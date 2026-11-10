from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


APP_DISPLAY_NAME = "Chemical Use Extractor"
PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_REPO_ID = "Jesolo/Llama-CUE"


def _default_model_download_dir() -> str:
    return str(WORKSPACE_ROOT / "models" / "Llama-CUE")


def _default_oecd_model_dir() -> str:
    return str(WORKSPACE_ROOT / "models" / "openai_gpt_oss_20b")


def _default_rag_source_csv() -> str:
    return ""


def _default_rag_index_path() -> str:
    return str(WORKSPACE_ROOT / "cache" / "kirk_r2_rag.sqlite")


def _resolve_workspace_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "chem_use_web":
        path = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()
    return str((WORKSPACE_ROOT / path).resolve())


@dataclass
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 8888
    model_path: str = _default_model_download_dir()
    model_repo_id: str = DEFAULT_MODEL_REPO_ID
    model_revision: str = "main"
    model_download_dir: str = _default_model_download_dir()
    extraction_backend: str = "transformers"
    device: str = "auto"
    gpu_id: int | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    prompt_template: str = (
        "<s>[INST] Extract compound & Use in Paragraph, the result should be a dict: "
        "{text} [/INST]"
    )
    direct_text_word_limit: int = 500
    file_sample_recommended_word_limit: int = 500
    file_sample_limit: int = 100000
    results_dir: str = str(WORKSPACE_ROOT / "runs")
    cache_dir: str = str(WORKSPACE_ROOT / "cache")
    chemical_enrichment_default: bool = True
    chemical_api_timeout_seconds: float = 8.0
    chemical_cache_ttl_seconds: int = 604800
    fp8_gpu_memory_threshold_gb: float = 16.0
    low_gpu_memory_warning_gb: float = 12.0
    oecd_provider: str = "ollama"
    oecd_model_id: str = "openai/gpt-oss-20b"
    oecd_model_dir: str = _default_oecd_model_dir()
    oecd_ollama_base_url: str = "http://127.0.0.1:11434/v1"
    oecd_ollama_model: str = "gpt-oss:20b"
    oecd_ollama_api_key: str = "ollama"
    oecd_ollama_timeout_seconds: float = 60.0
    oecd_vllm_base_url: str = "http://127.0.0.1:8000/v1"
    oecd_vllm_model: str = "openai/gpt-oss-20b"
    oecd_vllm_api_key: str = "EMPTY"
    oecd_vllm_timeout_seconds: float = 60.0
    oecd_probe_timeout_seconds: float = 3.0
    oecd_max_new_tokens: int = 96
    oecd_temperature: float = 0.0
    oecd_do_sample: bool = False
    rag_source_csv: str = _default_rag_source_csv()
    rag_index_path: str = _default_rag_index_path()
    rag_path_column: str = ""
    rag_text_column: str = "original paragraph"
    rag_output_column: str = "model outputs"
    rag_default_top_k: int = 10
    rag_transformers_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rag_transformers_device: str = "auto"
    rag_transformers_max_input_tokens: int = 4096
    rag_answer_max_context_chars: int = 9000
    rag_answer_max_tokens: int = 512
    debug_enabled: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["oecd_ollama_api_key"] = "set" if self.oecd_ollama_api_key else ""
        data["oecd_vllm_api_key"] = "set" if self.oecd_vllm_api_key else ""
        data["model_path_exists"] = bool(Path(self.model_path).expanduser().exists())
        data["model_download_dir_exists"] = bool(Path(self.model_download_dir).expanduser().exists())
        data["oecd_model_dir_exists"] = bool(Path(self.oecd_model_dir).expanduser().exists())
        return data


ENV_MAP: dict[str, tuple[str, Any]] = {
    "CHEM_USE_WEB_HOST": ("host", str),
    "CHEM_USE_WEB_PORT": ("port", int),
    "CHEM_USE_MODEL_PATH": ("model_path", str),
    "CHEM_USE_MODEL_REPO_ID": ("model_repo_id", str),
    "CHEM_USE_MODEL_REVISION": ("model_revision", str),
    "CHEM_USE_MODEL_DOWNLOAD_DIR": ("model_download_dir", str),
    "CHEM_USE_EXTRACTION_BACKEND": ("extraction_backend", str),
    "CHEM_USE_DEVICE": ("device", str),
    "CHEM_USE_GPU_ID": ("gpu_id", int),
    "CHEM_USE_MAX_NEW_TOKENS": ("max_new_tokens", int),
    "CHEM_USE_TEMPERATURE": ("temperature", float),
    "CHEM_USE_TOP_P": ("top_p", float),
    "CHEM_USE_DO_SAMPLE": ("do_sample", bool),
    "CHEM_USE_PROMPT_TEMPLATE": ("prompt_template", str),
    "CHEM_USE_RESULTS_DIR": ("results_dir", str),
    "CHEM_USE_CACHE_DIR": ("cache_dir", str),
    "CHEM_USE_FP8_GPU_MEMORY_THRESHOLD_GB": ("fp8_gpu_memory_threshold_gb", float),
    "CHEM_USE_LOW_GPU_MEMORY_WARNING_GB": ("low_gpu_memory_warning_gb", float),
    "CHEM_USE_OECD_PROVIDER": ("oecd_provider", str),
    "CHEM_USE_OECD_MODEL_ID": ("oecd_model_id", str),
    "CHEM_USE_OECD_MODEL_DIR": ("oecd_model_dir", str),
    "CHEM_USE_OECD_OLLAMA_BASE_URL": ("oecd_ollama_base_url", str),
    "CHEM_USE_OECD_OLLAMA_MODEL": ("oecd_ollama_model", str),
    "CHEM_USE_OECD_OLLAMA_API_KEY": ("oecd_ollama_api_key", str),
    "CHEM_USE_OECD_OLLAMA_TIMEOUT_SECONDS": ("oecd_ollama_timeout_seconds", float),
    "CHEM_USE_OECD_VLLM_BASE_URL": ("oecd_vllm_base_url", str),
    "CHEM_USE_OECD_VLLM_MODEL": ("oecd_vllm_model", str),
    "CHEM_USE_OECD_VLLM_API_KEY": ("oecd_vllm_api_key", str),
    "CHEM_USE_OECD_VLLM_TIMEOUT_SECONDS": ("oecd_vllm_timeout_seconds", float),
    "CHEM_USE_OECD_PROBE_TIMEOUT_SECONDS": ("oecd_probe_timeout_seconds", float),
    "CHEM_USE_RAG_SOURCE_CSV": ("rag_source_csv", str),
    "CHEM_USE_RAG_INDEX_PATH": ("rag_index_path", str),
    "CHEM_USE_RAG_PATH_COLUMN": ("rag_path_column", str),
    "CHEM_USE_RAG_TEXT_COLUMN": ("rag_text_column", str),
    "CHEM_USE_RAG_OUTPUT_COLUMN": ("rag_output_column", str),
    "CHEM_USE_RAG_DEFAULT_TOP_K": ("rag_default_top_k", int),
    "CHEM_USE_RAG_TRANSFORMERS_MODEL": ("rag_transformers_model", str),
    "CHEM_USE_RAG_TRANSFORMERS_DEVICE": ("rag_transformers_device", str),
    "CHEM_USE_RAG_TRANSFORMERS_MAX_INPUT_TOKENS": ("rag_transformers_max_input_tokens", int),
    "CHEM_USE_RAG_ANSWER_MAX_CONTEXT_CHARS": ("rag_answer_max_context_chars", int),
    "CHEM_USE_RAG_ANSWER_MAX_TOKENS": ("rag_answer_max_tokens", int),
}


def parse_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce(value: str, converter: Any) -> Any:
    if converter is bool:
        return parse_bool(value)
    if converter is int and value == "":
        return None
    return converter(value)


def read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def load_settings(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> AppSettings:
    settings = AppSettings()

    resolved_config = Path(
        config_path
        or os.environ.get("CHEM_USE_WEB_CONFIG", "")
        or WORKSPACE_ROOT / "config.local.json"
    ).expanduser()
    config_payload = read_json_config(resolved_config)
    for key, value in config_payload.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    for env_name, (field_name, converter) in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value is not None:
            setattr(settings, field_name, _coerce(value, converter))

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(settings, key):
                setattr(settings, key, value)

    settings.oecd_provider = str(settings.oecd_provider or "ollama").strip().lower()
    settings.extraction_backend = str(settings.extraction_backend or "transformers").strip().lower()
    settings.model_path = _resolve_workspace_path(settings.model_path)
    settings.model_download_dir = _resolve_workspace_path(settings.model_download_dir)
    settings.results_dir = _resolve_workspace_path(settings.results_dir)
    settings.cache_dir = _resolve_workspace_path(settings.cache_dir)
    settings.oecd_model_dir = _resolve_workspace_path(settings.oecd_model_dir)
    settings.rag_source_csv = _resolve_workspace_path(settings.rag_source_csv)
    settings.rag_index_path = _resolve_workspace_path(settings.rag_index_path)
    Path(settings.results_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(settings.cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(settings.rag_index_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return settings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Chemical Use Extractor web app.")
    parser.add_argument("--config", help="Path to a JSON config file. Defaults to config.local.json.")
    parser.add_argument("--host", help="Host interface to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, help="Port to bind. Default: 8888")
    parser.add_argument("--model-path", help="Path to the merged extraction model.")
    parser.add_argument("--model-repo-id", help="Hugging Face repo id for downloading the merged extraction model.")
    parser.add_argument("--model-revision", help="Optional Hugging Face revision for the merged extraction model.")
    parser.add_argument("--model-download-dir", help="Local directory used when downloading the merged extraction model.")
    parser.add_argument("--extraction-backend", choices=("transformers", "mlx"), help="Extraction runtime backend.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), help="Inference device.")
    parser.add_argument("--gpu-id", type=int, help="CUDA GPU index when using CUDA.")
    parser.add_argument("--oecd-provider", choices=("ollama", "vllm", "local"), help="OECD classifier provider.")
    parser.add_argument("--oecd-ollama-base-url", help="OpenAI-compatible Ollama base URL, for example http://127.0.0.1:11434/v1.")
    parser.add_argument("--oecd-ollama-model", help="Model name served by Ollama, for example gpt-oss:20b.")
    parser.add_argument("--oecd-vllm-base-url", help="OpenAI-compatible vLLM base URL, for example http://127.0.0.1:8000/v1.")
    parser.add_argument("--oecd-vllm-model", help="Model name served by vLLM, for example openai/gpt-oss-20b.")
    parser.add_argument("--rag-transformers-model", help="Small Hugging Face model/path loaded directly in Python for RAG answers.")
    parser.add_argument("--rag-transformers-device", choices=("auto", "cuda", "mps", "cpu"), help="Device for the direct Transformers RAG answer model.")
    return parser
