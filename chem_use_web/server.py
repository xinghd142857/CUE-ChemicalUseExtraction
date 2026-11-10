from __future__ import annotations

import json
import mimetypes
import sys
import traceback
import urllib.parse
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .core import ChemicalUseWebApp, UserFacingError
from .settings import AppSettings, PACKAGE_ROOT, build_arg_parser, load_settings


STATIC_ROOT = PACKAGE_ROOT / "static"


def parse_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise UserFacingError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise UserFacingError("JSON body must be an object.")
    return payload


def parse_multipart(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise UserFacingError("Expected multipart/form-data upload.")
    length = int(handler.headers.get("Content-Length") or 0)
    raw_body = handler.rfile.read(length)
    raw_message = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + raw_body
    message = BytesParser(policy=default).parsebytes(raw_message)
    fields: dict[str, Any] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if filename:
            fields[name] = {"filename": filename, "content": data}
        else:
            fields[name] = data.decode(part.get_content_charset() or "utf-8", errors="replace")
    return fields


def make_handler(app: ChemicalUseWebApp) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        server_version = f"ChemUseWeb/{__version__}"

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                query = urllib.parse.parse_qs(parsed.query)
                if path == "/":
                    self._send_file(STATIC_ROOT / "index.html")
                elif path.startswith("/static/"):
                    self._send_file(STATIC_ROOT / path.removeprefix("/static/"))
                elif path == "/api/health":
                    self._send_json(app.health())
                elif path == "/api/settings":
                    self._send_json(app.settings.to_public_dict())
                elif path == "/api/debug/model":
                    self._send_json(app.model_manager.status())
                elif path == "/api/model/download/status":
                    self._send_json(app.model_download_status())
                elif path == "/api/rag/status":
                    self._send_json(app.rag_status())
                elif path == "/api/dev/name-to-smiles/status":
                    self._send_json(
                        {
                            "visible": False,
                            "status": "reserved",
                            "message": (
                                "Hidden placeholder for the future name-to-SMILES model. "
                                "No user-facing controls are enabled yet."
                            ),
                        }
                    )
                elif path == "/api/download":
                    job_id = (query.get("job_id") or [""])[0]
                    file_format = (query.get("format") or ["csv"])[0].lower()
                    content, content_type, filename = app.export_job(job_id, file_format)
                    self._send_bytes(content, content_type, filename=filename)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._handle_error(exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                if path == "/api/extract":
                    payload = parse_json_body(self)
                    result = app.extract_text(
                        str(payload.get("text") or ""),
                        enable_oecd=bool(payload.get("enable_oecd")),
                        enrich_chemicals=bool(payload.get("enrich_chemicals", True)),
                        include_debug=bool(payload.get("include_debug")),
                    )
                    self._send_json(result)
                elif path == "/api/preview-file":
                    fields = parse_multipart(self)
                    upload = fields.get("file")
                    if not upload:
                        raise UserFacingError("Upload a file first.")
                    result = app.preview_file(upload["filename"], upload["content"])
                    self._send_json(result)
                elif path == "/api/extract-file":
                    fields = parse_multipart(self)
                    upload = fields.get("file")
                    if not upload:
                        raise UserFacingError("Upload a file first.")
                    result = app.extract_file(
                        upload["filename"],
                        upload["content"],
                        column=str(fields.get("column") or "") or None,
                        enable_oecd=str(fields.get("enable_oecd") or "").lower() == "true",
                        enrich_chemicals=str(fields.get("enrich_chemicals") or "true").lower() == "true",
                        include_debug=str(fields.get("include_debug") or "").lower() == "true",
                    )
                    self._send_json(result)
                elif path == "/api/oecd/download":
                    result = app.oecd_classifier.download()
                    self._send_json(result)
                elif path == "/api/oecd/check":
                    result = app.oecd_classifier.check_provider()
                    self._send_json(result)
                elif path == "/api/model/download":
                    result = app.download_model()
                    self._send_json(result)
                elif path == "/api/training/plan":
                    payload = parse_json_body(self)
                    result = app.training_plan(payload)
                    self._send_json(result)
                elif path == "/api/rag/reindex":
                    payload = parse_json_body(self)
                    result = app.rag_reindex(
                        source_csv=str(payload.get("source_csv") or ""),
                        output_column=str(payload.get("output_column") or ""),
                        path_column=str(payload.get("path_column") or ""),
                        text_column=str(payload.get("text_column") or ""),
                    )
                    self._send_json(result)
                elif path == "/api/rag/search":
                    payload = parse_json_body(self)
                    result = app.rag_search(
                        str(payload.get("query") or ""),
                        top_k=int(payload.get("top_k") or app.settings.rag_default_top_k),
                    )
                    self._send_json(result)
                elif path == "/api/rag/answer":
                    payload = parse_json_body(self)
                    result = app.rag_answer(
                        str(payload.get("query") or ""),
                        top_k=int(payload.get("top_k") or app.settings.rag_default_top_k),
                    )
                    self._send_json(result)
                elif path == "/api/rag/enrich":
                    payload = parse_json_body(self)
                    results = payload.get("results") or []
                    limit = int(payload.get("limit") or len(results))
                    result = app.rag_enrich_results(results, limit=limit)
                    self._send_json(result)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._handle_error(exc)

        def _send_file(self, path: Path) -> None:
            try:
                resolved = path.resolve()
                if not str(resolved).startswith(str(STATIC_ROOT.resolve())):
                    self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                    return
                if not resolved.exists() or not resolved.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
                self._send_bytes(resolved.read_bytes(), content_type)
            except OSError as exc:
                self._handle_error(exc)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _send_bytes(
            self,
            content: bytes,
            content_type: str,
            *,
            status: int = 200,
            filename: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            if filename:
                safe_name = filename.replace('"', "")
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
            self.end_headers()
            self.wfile.write(content)

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, UserFacingError):
                self._send_json({"ok": False, "error": str(exc)}, status=exc.status_code)
                return
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    return Handler


def settings_from_args(args: Any) -> AppSettings:
    return load_settings(
        args.config,
        overrides={
            "host": args.host,
            "port": args.port,
            "model_path": args.model_path,
            "model_repo_id": args.model_repo_id,
            "model_revision": args.model_revision,
            "model_download_dir": args.model_download_dir,
            "extraction_backend": args.extraction_backend,
            "device": args.device,
            "gpu_id": args.gpu_id,
            "oecd_provider": args.oecd_provider,
            "oecd_ollama_base_url": args.oecd_ollama_base_url,
            "oecd_ollama_model": args.oecd_ollama_model,
            "oecd_vllm_base_url": args.oecd_vllm_base_url,
            "oecd_vllm_model": args.oecd_vllm_model,
            "rag_transformers_model": args.rag_transformers_model,
            "rag_transformers_device": args.rag_transformers_device,
        },
    )


def run_server(settings: AppSettings) -> int:
    app = ChemicalUseWebApp(settings)
    handler = make_handler(app)
    address = (settings.host, settings.port)
    httpd = ThreadingHTTPServer(address, handler)
    print(f"Chemical Use Extractor running at http://{settings.host}:{settings.port}")
    print(f"Extraction backend: {settings.extraction_backend}")
    print(f"Model path: {settings.model_path}")
    print(f"Model repo: {settings.model_repo_id}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    return run_server(settings)
