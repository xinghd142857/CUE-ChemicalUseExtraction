from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


GIB = 1024**3
DEFAULT_PROMPT_TEMPLATE = (
    "Extract chemical-use pairs from the paragraph.\n\n"
    "Paragraph:\n{input}\n\n"
    'Return a JSON object such as {"Water": "solvent | cleaning agent"}.'
)
VALID_STRATEGIES = {"lora", "dpo"}


class TrainingPlanError(ValueError):
    pass


def build_training_plan(payload: dict[str, Any], *, hardware: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    request = _normalize_request(payload)
    required_columns = _required_columns(request)
    dataset = _inspect_training_data(
        request["training_data_path"],
        base_dir=base_dir,
        required_columns=required_columns,
    )
    estimate = _estimate_memory(request, hardware)
    repo_id = _repo_id_from_model_link(request["model_link"])
    warnings = _build_warnings(request, hardware, estimate)

    training_config = {
        "framework": "CUE",
        "strategy": request["strategy"],
        "model": {
            "model_link": request["model_link"],
            "repo_id": repo_id,
            "estimated_parameters_b": request["model_size_b"],
        },
        "dataset": {
            "path": dataset["path"],
            "format": dataset["format"],
            "input_column": request["input_column"],
            "output_column": request["output_column"],
            "prompt_column": request["prompt_column"],
            "chosen_column": request["chosen_column"],
            "rejected_column": request["rejected_column"],
        },
        "prompt_template": request["prompt_template"],
        "hyperparameters": _public_hyperparameters(request),
    }

    return {
        "ok": True,
        "framework": "CUE",
        "strategy": request["strategy"],
        "strategy_label": "LoRA supervised fine-tuning" if request["strategy"] == "lora" else "DPO preference tuning",
        "model": {
            "model_link": request["model_link"],
            "repo_id": repo_id,
            "estimated_parameters_b": request["model_size_b"],
        },
        "hardware": _public_hardware(hardware),
        "dataset": dataset,
        "required_columns": required_columns,
        "prompt_template": request["prompt_template"],
        "hyperparameters": _public_hyperparameters(request),
        "estimate": estimate,
        "warnings": warnings,
        "training_config": training_config,
        "install_command": _install_command(hardware, request),
        "next_steps": _next_steps(request),
    }


def _normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
    strategy = str(payload.get("strategy") or "lora").strip().lower()
    if strategy not in VALID_STRATEGIES:
        raise TrainingPlanError("Choose either LoRA or DPO as the training strategy.")

    model_link = str(payload.get("model_link") or "").strip()
    if not model_link:
        raise TrainingPlanError("Paste a Hugging Face model link or repo id.")

    prompt_template = str(payload.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE).strip()
    if not prompt_template:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    model_size_b = _number(payload, "model_size_b", _infer_model_size_b(model_link), minimum=0.1)

    return {
        "strategy": strategy,
        "model_link": model_link,
        "model_size_b": model_size_b,
        "training_data_path": str(payload.get("training_data_path") or "").strip(),
        "input_column": str(payload.get("input_column") or "").strip(),
        "output_column": str(payload.get("output_column") or "").strip(),
        "prompt_column": str(payload.get("prompt_column") or "").strip(),
        "chosen_column": str(payload.get("chosen_column") or "").strip(),
        "rejected_column": str(payload.get("rejected_column") or "").strip(),
        "prompt_template": prompt_template,
        "learning_rate": _number(payload, "learning_rate", 2e-4, minimum=1e-8),
        "epochs": _number(payload, "epochs", 3.0, minimum=0.1),
        "per_device_batch_size": int(_number(payload, "per_device_batch_size", 1, minimum=1)),
        "gradient_accumulation_steps": int(_number(payload, "gradient_accumulation_steps", 8, minimum=1)),
        "max_seq_length": int(_number(payload, "max_seq_length", 2048, minimum=128)),
        "lora_rank": int(_number(payload, "lora_rank", 16, minimum=1)),
        "lora_alpha": int(_number(payload, "lora_alpha", 32, minimum=1)),
        "lora_dropout": _number(payload, "lora_dropout", 0.05, minimum=0.0),
        "dpo_beta": _number(payload, "dpo_beta", 0.1, minimum=0.001),
        "load_in_4bit": _bool(payload.get("load_in_4bit"), default=True),
        "gradient_checkpointing": _bool(payload.get("gradient_checkpointing"), default=True),
    }


def _number(payload: dict[str, Any], key: str, default: float, *, minimum: float) -> float:
    raw = payload.get(key)
    if raw in (None, ""):
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise TrainingPlanError(f"{key} must be a number.") from exc
    if not math.isfinite(value) or value < minimum:
        raise TrainingPlanError(f"{key} must be at least {minimum:g}.")
    return value


def _bool(value: Any, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _repo_id_from_model_link(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return value.strip()
    return value.strip().strip("/")


def _infer_model_size_b(value: str) -> float:
    repo_id = _repo_id_from_model_link(value)
    matches = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", repo_id)
    if matches:
        return float(matches[-1])
    return 8.0


def _required_columns(request: dict[str, Any]) -> list[str]:
    if request["strategy"] == "dpo":
        columns = [request["prompt_column"], request["chosen_column"], request["rejected_column"]]
        if not all(columns):
            raise TrainingPlanError("DPO needs prompt, chosen, and rejected columns.")
        return columns

    columns = [request["input_column"], request["output_column"]]
    if not all(columns):
        raise TrainingPlanError("LoRA needs one input column and one output column.")
    return columns


def _resolve_data_path(value: str, base_dir: Path) -> Path:
    if not value:
        raise TrainingPlanError("Enter the local training data file path.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _inspect_training_data(value: str, *, base_dir: Path, required_columns: list[str]) -> dict[str, Any]:
    path = _resolve_data_path(value, base_dir)
    if not path.exists() or not path.is_file():
        raise TrainingPlanError(f"Training data file was not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        preview_frame = pd.read_csv(path, sep=sep, nrows=5)
        row_count = _count_delimited_rows(path)
        data_format = suffix.removeprefix(".").upper()
    elif suffix in {".xlsx", ".xls"}:
        preview_frame = pd.read_excel(path, nrows=5)
        row_count = int(len(pd.read_excel(path, usecols=[0])))
        data_format = suffix.removeprefix(".").upper()
    elif suffix == ".jsonl":
        preview_rows, row_count = _read_jsonl_preview(path)
        preview_frame = pd.DataFrame(preview_rows)
        data_format = "JSONL"
    elif suffix == ".json":
        preview_frame, row_count = _read_json_preview(path)
        data_format = "JSON"
    else:
        raise TrainingPlanError("Training data must be CSV, TSV, XLSX, XLS, JSONL, or JSON.")

    columns = [str(column) for column in preview_frame.columns]
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise TrainingPlanError(
            "Missing required training column(s): "
            + ", ".join(missing)
            + ". Available columns: "
            + (", ".join(columns) or "none")
            + "."
        )

    return {
        "path": str(path),
        "exists": True,
        "format": data_format,
        "row_count": row_count,
        "columns": columns,
        "preview": preview_frame.head(5).fillna("").astype(str).to_dict(orient="records"),
    }


def _count_delimited_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0


def _read_jsonl_preview(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    row_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            clean = line.strip()
            if not clean:
                continue
            row_count += 1
            if len(rows) < 5:
                parsed = json.loads(clean)
                if not isinstance(parsed, dict):
                    raise TrainingPlanError("JSONL training rows must be objects.")
                rows.append(parsed)
    return rows, row_count


def _read_json_preview(path: Path) -> tuple[pd.DataFrame, int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise TrainingPlanError("JSON training data must be a list of objects or an object with a data list.")
    return pd.DataFrame(payload[:5]), len(payload)


def _estimate_memory(request: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    params_b = request["model_size_b"]
    quantized = request["load_in_4bit"]
    base_weight_gb = params_b * 1_000_000_000 * (0.55 if quantized else 2.0) / GIB

    sequence_factor = request["max_seq_length"] / 2048
    batch_factor = request["per_device_batch_size"]
    activation_gb = max(0.8, params_b * 0.22) * sequence_factor * batch_factor
    if request["strategy"] == "dpo":
        activation_gb *= 1.8
    if request["gradient_checkpointing"]:
        activation_gb *= 0.6

    adapter_gb = 0.45 + (params_b / 8.0) * (request["lora_rank"] / 16.0) * 0.35
    optimizer_gb = 0.75 if quantized else 1.2
    runtime_gb = 1.2 + params_b * 0.08
    total_gb = base_weight_gb + activation_gb + adapter_gb + optimizer_gb + runtime_gb
    available_gb = _available_memory_gb(hardware)

    if available_gb is None:
        status = "unknown"
        margin_gb = None
    else:
        margin_gb = available_gb - total_gb
        if available_gb >= total_gb * 1.15:
            status = "ok"
        elif available_gb >= total_gb:
            status = "tight"
        else:
            status = "too_large"

    return {
        "estimated_memory_gb": round(total_gb, 2),
        "available_memory_gb": round(available_gb, 2) if available_gb is not None else None,
        "margin_gb": round(margin_gb, 2) if margin_gb is not None else None,
        "status": status,
        "breakdown": {
            "base_weights_gb": round(base_weight_gb, 2),
            "activations_gb": round(activation_gb, 2),
            "lora_and_optimizer_gb": round(adapter_gb + optimizer_gb, 2),
            "runtime_overhead_gb": round(runtime_gb, 2),
        },
        "assumptions": [
            "Memory is a rough planning estimate, not a scheduler guarantee.",
            "LoRA/DPO estimates assume adapter training, not full fine-tuning.",
            "DPO uses paired chosen/rejected completions, so it needs more activation memory than LoRA SFT.",
        ],
    }


def _available_memory_gb(hardware: dict[str, Any]) -> float | None:
    value = hardware.get("total_memory_gb")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    bytes_value = hardware.get("total_memory_bytes")
    if isinstance(bytes_value, (int, float)) and bytes_value > 0:
        return float(bytes_value) / GIB
    return None


def _public_hardware(hardware: dict[str, Any]) -> dict[str, Any]:
    return {
        "accelerator": hardware.get("accelerator") or "unknown",
        "name": hardware.get("name") or "",
        "index": hardware.get("index"),
        "total_memory_gb": hardware.get("total_memory_gb"),
        "selected_quantization": hardware.get("selected_quantization") or "none",
        "warning": hardware.get("warning") or "",
    }


def _public_hyperparameters(request: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "learning_rate",
        "epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "dpo_beta",
        "load_in_4bit",
        "gradient_checkpointing",
    )
    return {key: request[key] for key in keys}


def _build_warnings(request: dict[str, Any], hardware: dict[str, Any], estimate: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if "{input}" not in request["prompt_template"] and request["strategy"] == "lora":
        warnings.append("The prompt template does not contain {input}; most LoRA datasets should include it.")

    accelerator = str(hardware.get("accelerator") or "").lower()
    if accelerator in {"cpu", "unknown"}:
        warnings.append("No CUDA/ROCm/MPS accelerator was selected. Training may be extremely slow.")
    if accelerator in {"mps", "mlx-metal"} and request["load_in_4bit"]:
        warnings.append("4-bit QLoRA usually depends on CUDA bitsandbytes. Apple Silicon may need an MLX LoRA workflow.")
    if request["strategy"] == "dpo" and accelerator in {"mps", "mlx-metal", "cpu", "unknown"}:
        warnings.append("DPO is usually most practical on a CUDA/ROCm GPU because each row has chosen and rejected outputs.")
    if estimate["status"] == "tight":
        warnings.append("Estimated memory is close to the detected limit. Reduce sequence length, batch size, or LoRA rank.")
    if estimate["status"] == "too_large":
        warnings.append("Estimated memory is above the detected limit. Use a smaller model, 4-bit loading, or a remote GPU.")
    return warnings


def _install_command(hardware: dict[str, Any], request: dict[str, Any]) -> str:
    packages = [
        '"datasets>=2.19"',
        '"peft>=0.11"',
        '"trl>=0.9"',
        '"accelerate>=0.30"',
    ]
    accelerator = str(hardware.get("accelerator") or "").lower()
    if request["load_in_4bit"] and accelerator in {"cuda", "rocm"}:
        packages.append("bitsandbytes")
    return "python -m pip install " + " ".join(packages)


def _next_steps(request: dict[str, Any]) -> list[str]:
    if request["strategy"] == "dpo":
        return [
            "Confirm every row has a prompt, chosen answer, and rejected answer.",
            "Keep the prompt direct; most quality comes from chosen/rejected examples.",
            "Start with one epoch and inspect a small validation sample before longer runs.",
        ]
    return [
        "Confirm the input column contains the paragraph or prompt input.",
        "Confirm the output column contains the exact target answer format you want.",
        "Start with a small LoRA rank and short sequence length, then increase only if quality needs it.",
    ]
