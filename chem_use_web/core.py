from __future__ import annotations

import csv
import io
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .inference import build_extraction_model_manager
from .oecd import OecdClassifier
from .parser import count_words, normalize_text, parse_extraction_output
from .rag import KirkRagSearch, RagUnavailableError
from .rag_answer import RagAnswerer, RagGenerationError
from .resolver import ChemicalResolver
from .settings import AppSettings, WORKSPACE_ROOT
from .training import TrainingPlanError, build_training_plan


TEXT_COLUMN_CANDIDATES = (
    "text",
    "paragraph",
    "description",
    "input",
    "prompt",
    "content",
    "abstract",
    "sentence",
)


class UserFacingError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def read_table_from_bytes(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(filename or "").suffix.lower()
    stream = io.BytesIO(payload)
    if suffix == ".csv":
        return pd.read_csv(stream)
    if suffix == ".tsv":
        return pd.read_csv(stream, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(stream)
    raise UserFacingError("Unsupported file type. Upload CSV, TSV, XLSX, or XLS.")


def infer_text_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    columns = [str(column) for column in frame.columns]
    if requested:
        if requested not in columns:
            raise UserFacingError(f"Column '{requested}' was not found in the uploaded file.")
        return requested
    lowered = {column.lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    object_columns = [
        column
        for column in columns
        if frame[column].dtype == object or pd.api.types.is_string_dtype(frame[column])
    ]
    if object_columns:
        return object_columns[0]
    if columns:
        return columns[0]
    raise UserFacingError("The uploaded file has no columns.")


def dataframe_preview(frame: pd.DataFrame, max_rows: int = 5) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": frame.head(max_rows).fillna("").astype(str).to_dict(orient="records"),
        "row_count": int(len(frame)),
        "inferred_text_column": infer_text_column(frame),
    }


def resolve_runtime_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((WORKSPACE_ROOT / path).resolve())


class ChemicalUseWebApp:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.model_manager = build_extraction_model_manager(settings)
        self.oecd_classifier = OecdClassifier(settings)
        self.resolver = ChemicalResolver(settings)
        self.rag_searcher = KirkRagSearch(settings)
        self.rag_answerer = RagAnswerer(settings)

    def health(self) -> dict[str, Any]:
        model_status = self.model_manager.status()
        return {
            "ok": True,
            "settings": self.settings.to_public_dict(),
            "hardware": model_status.get("hardware", {}),
            "model": model_status,
            "oecd": self.oecd_classifier.status(),
            "rag": self.rag_searcher.status(),
        }

    def download_model(self) -> dict[str, Any]:
        try:
            return self.model_manager.download_model()
        except (RuntimeError, ValueError) as exc:
            raise UserFacingError(str(exc)) from exc

    def model_download_status(self) -> dict[str, Any]:
        if hasattr(self.model_manager, "download_status"):
            return self.model_manager.download_status()
        return {"status": "unavailable", "active": False, "message": "Download status is unavailable."}

    def preview_file(self, filename: str, payload: bytes) -> dict[str, Any]:
        frame = read_table_from_bytes(filename, payload)
        return dataframe_preview(frame)

    def rag_status(self) -> dict[str, Any]:
        return self.rag_searcher.status()

    def training_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return build_training_plan(
                payload,
                hardware=self.model_manager.hardware_status(),
                base_dir=WORKSPACE_ROOT,
            )
        except TrainingPlanError as exc:
            raise UserFacingError(str(exc)) from exc

    def rag_reindex(
        self,
        *,
        source_csv: str | None = None,
        output_column: str | None = None,
        path_column: str | None = None,
        text_column: str | None = None,
    ) -> dict[str, Any]:
        if source_csv is not None:
            clean_source = source_csv.strip()
            self.settings.rag_source_csv = resolve_runtime_path(clean_source) if clean_source else ""
        if output_column is not None:
            self.settings.rag_output_column = output_column.strip()
        if path_column is not None:
            self.settings.rag_path_column = path_column.strip()
        if text_column is not None:
            self.settings.rag_text_column = text_column.strip()
        self.rag_searcher = KirkRagSearch(self.settings)
        try:
            return self.rag_searcher.build_index()
        except RagUnavailableError as exc:
            raise UserFacingError(str(exc)) from exc

    def rag_search(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        try:
            return self.rag_searcher.search(
                query,
                top_k=top_k or self.settings.rag_default_top_k,
            )
        except RagUnavailableError as exc:
            raise UserFacingError(str(exc)) from exc

    def rag_answer(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        retrieval = self.rag_search(query, top_k=top_k)
        results = retrieval.get("results") or []
        try:
            generated = self.rag_answerer.answer(query, results)
        except RagGenerationError as exc:
            raise UserFacingError(str(exc)) from exc
        return {
            "ok": True,
            "query": retrieval.get("query") or query,
            "answer": generated.get("answer", ""),
            "provider": generated.get("provider", ""),
            "model": generated.get("model", ""),
            "citations": generated.get("citations", []),
            "context_count": generated.get("context_count", 0),
            "message": generated.get("message", ""),
            "retrieval": retrieval,
            "results": results,
        }

    def rag_enrich_results(self, results: list[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
        if not isinstance(results, list):
            raise UserFacingError("RAG enrichment expects a list of visible results.")

        max_rows = min(max(int(limit or len(results) or 0), 0), 50)
        visible = results[:max_rows]
        local_cache: dict[str, Any] = {}
        enriched: list[dict[str, Any]] = []

        for row in visible:
            chemical = normalize_text(str(row.get("chemical") or row.get("chemicals") or ""))
            if not chemical:
                continue
            chemical_info = local_cache.get(chemical)
            if chemical_info is None:
                chemical_info = self.resolver.resolve(chemical)
                local_cache[chemical] = chemical_info
            enriched.append(
                {
                    "rowid": row.get("rowid"),
                    "chemical": chemical,
                    "chemical_info": chemical_info,
                }
            )

        return {
            "ok": True,
            "count": len(enriched),
            "unique_chemicals": len(local_cache),
            "results": enriched,
        }

    def extract_text(
        self,
        text: str,
        *,
        enable_oecd: bool,
        enrich_chemicals: bool,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        clean_text = normalize_text(text)
        if not clean_text:
            raise UserFacingError("Enter text before running extraction.")
        word_count = count_words(clean_text)
        if word_count > self.settings.direct_text_word_limit:
            raise UserFacingError(
                f"Direct text input is limited to {self.settings.direct_text_word_limit} words. "
                f"This input has {word_count} words."
            )
        result = self._process_sample(
            clean_text,
            source_row=0,
            enable_oecd=enable_oecd,
            enrich_chemicals=enrich_chemicals,
            include_debug=include_debug,
        )
        return self._save_job(
            {
                "mode": "text",
                "input_rows": 1,
                "records": result["records"],
                "summary": self._summarize(result["records"]),
                "warnings": result["warnings"],
            }
        )

    def extract_file(
        self,
        filename: str,
        payload: bytes,
        *,
        column: str | None,
        enable_oecd: bool,
        enrich_chemicals: bool,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        frame = read_table_from_bytes(filename, payload)
        if len(frame) > self.settings.file_sample_limit:
            raise UserFacingError(
                f"Uploaded file has {len(frame)} rows. The configured limit is "
                f"{self.settings.file_sample_limit} rows."
            )
        text_column = infer_text_column(frame, column)
        records: list[dict[str, Any]] = []
        warnings: list[str] = []
        original_rows = frame.fillna("").astype(str).to_dict(orient="records")

        for index, value in enumerate(frame[text_column].tolist()):
            text = "" if pd.isna(value) else normalize_text(value)
            if count_words(text) > self.settings.file_sample_recommended_word_limit:
                warnings.append(
                    f"Row {index + 1} is longer than the recommended "
                    f"{self.settings.file_sample_recommended_word_limit} words."
                )
            sample = self._process_sample(
                text,
                source_row=index,
                enable_oecd=enable_oecd,
                enrich_chemicals=enrich_chemicals,
                include_debug=include_debug,
            )
            for record in sample["records"]:
                record["original_row"] = original_rows[index]
            records.extend(sample["records"])
            warnings.extend(sample["warnings"])

        return self._save_job(
            {
                "mode": "file",
                "filename": filename,
                "input_rows": int(len(frame)),
                "text_column": text_column,
                "records": records,
                "summary": self._summarize(records),
                "warnings": warnings[:1000],
            }
        )

    def _process_sample(
        self,
        text: str,
        *,
        source_row: int,
        enable_oecd: bool,
        enrich_chemicals: bool,
        include_debug: bool,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        if not text:
            return {
                "records": [
                    self._status_record(source_row, text, "no_result", "Input row was empty.")
                ],
                "warnings": warnings,
            }

        try:
            raw_output = self.model_manager.generate(text)
        except Exception as exc:
            return {
                "records": [self._status_record(source_row, text, "inference_error", str(exc))],
                "warnings": warnings,
            }

        parsed = parse_extraction_output(raw_output)
        if parsed.status != "success":
            record = self._status_record(source_row, text, parsed.status, parsed.error)
            if include_debug:
                record["debug"] = {"raw_model_output": raw_output, "cleaned_output": parsed.cleaned_output}
            return {"records": [record], "warnings": warnings}

        records: list[dict[str, Any]] = []
        enrichment_cache: dict[str, Any] = {}
        for pair in parsed.pairs:
            chemical = pair["chemical"]
            chemical_info = {}
            if enrich_chemicals:
                chemical_info = enrichment_cache.get(chemical)
                if chemical_info is None:
                    chemical_info = self.resolver.resolve(chemical)
                    enrichment_cache[chemical] = chemical_info

            for use_text in pair["uses"]:
                oecd_labels: list[str] = []
                oecd_error = ""
                if enable_oecd:
                    try:
                        oecd_labels = self.oecd_classifier.classify(use_text)
                    except Exception as exc:
                        oecd_error = str(exc)
                record = {
                    "source_row": source_row,
                    "status": "success",
                    "input_preview": text[:240],
                    "chemical": chemical,
                    "use": use_text,
                    "oecd_labels": oecd_labels,
                    "oecd_error": oecd_error,
                    "chemical_info": chemical_info,
                }
                if include_debug:
                    record["debug"] = {"raw_model_output": raw_output, "cleaned_output": parsed.cleaned_output}
                records.append(record)
        return {"records": records, "warnings": warnings}

    @staticmethod
    def _status_record(source_row: int, text: str, status: str, error: str = "") -> dict[str, Any]:
        return {
            "source_row": source_row,
            "status": status,
            "input_preview": text[:240],
            "chemical": "",
            "use": "",
            "oecd_labels": [],
            "oecd_error": "",
            "chemical_info": {},
            "error": error,
        }

    @staticmethod
    def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        chemicals: set[str] = set()
        for record in records:
            status = str(record.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
            if record.get("chemical"):
                chemicals.add(str(record["chemical"]))
        return {
            "records": len(records),
            "unique_chemicals": len(chemicals),
            "statuses": statuses,
        }

    def _save_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        payload["job_id"] = job_id
        path = Path(self.settings.results_dir).expanduser() / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["download_urls"] = {
            "json": f"/api/download?job_id={job_id}&format=json",
            "csv": f"/api/download?job_id={job_id}&format=csv",
            "xlsx": f"/api/download?job_id={job_id}&format=xlsx",
        }
        return payload

    def load_job(self, job_id: str) -> dict[str, Any]:
        if not re_safe_job_id(job_id):
            raise UserFacingError("Invalid job id.")
        path = Path(self.settings.results_dir).expanduser() / f"{job_id}.json"
        if not path.exists():
            raise UserFacingError("Job not found.", status_code=404)
        return json.loads(path.read_text(encoding="utf-8"))

    def export_job(self, job_id: str, file_format: str) -> tuple[bytes, str, str]:
        payload = self.load_job(job_id)
        records = payload.get("records", [])
        rows = [flatten_record(record) for record in records]
        stem = f"chemical_use_results_{job_id[:8]}"
        if file_format == "json":
            return (
                json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
                "application/json",
                f"{stem}.json",
            )
        if file_format == "csv":
            buffer = io.StringIO()
            fieldnames = sorted({key for row in rows for key in row.keys()}) or ["status"]
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            return buffer.getvalue().encode("utf-8-sig"), "text/csv", f"{stem}.csv"
        if file_format == "xlsx":
            buffer = io.BytesIO()
            pd.DataFrame(rows).to_excel(buffer, index=False)
            return (
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                f"{stem}.xlsx",
            )
        raise UserFacingError("Unsupported download format.", status_code=400)


def re_safe_job_id(value: str) -> bool:
    return bool(value) and all(ch in "0123456789abcdef" for ch in value.lower()) and len(value) == 32


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    chemical_info = record.get("chemical_info") or {}
    identifiers = chemical_info.get("identifiers") or {}
    links = chemical_info.get("links") or {}
    row = {
        "source_row": record.get("source_row", ""),
        "status": record.get("status", ""),
        "chemical": record.get("chemical", ""),
        "use": record.get("use", ""),
        "oecd_labels": " | ".join(record.get("oecd_labels") or []),
        "oecd_error": record.get("oecd_error", ""),
        "error": record.get("error", ""),
        "input_preview": record.get("input_preview", ""),
        "pubchem_cid": identifiers.get("pubchem_cid", ""),
        "cas": " | ".join(identifiers.get("cas") or []),
        "inchi_key": identifiers.get("inchi_key", ""),
        "inchi": identifiers.get("inchi", ""),
        "smiles": identifiers.get("smiles", ""),
        "iupac_name": identifiers.get("iupac_name", ""),
        "molecular_formula": identifiers.get("molecular_formula", ""),
        "molecular_weight": identifiers.get("molecular_weight", ""),
        "chembl_id": identifiers.get("chembl_id", ""),
        "sources": " | ".join(chemical_info.get("sources") or []),
    }
    for name, url in links.items():
        row[f"link_{name}"] = url
    original = record.get("original_row") or {}
    for key, value in original.items():
        row[f"input_{key}"] = value
    return row
