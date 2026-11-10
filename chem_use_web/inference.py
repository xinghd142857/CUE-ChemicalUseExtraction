from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from .hardware import detect_gpu_memory_policy
from .settings import AppSettings


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class ModelDependencyError(RuntimeError):
    pass


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _estimate_hf_repo_size(settings: AppSettings) -> int | None:
    if not settings.model_repo_id:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            repo_id=settings.model_repo_id,
            revision=settings.model_revision or None,
            files_metadata=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        return None

    total = 0
    for sibling in getattr(info, "siblings", []) or []:
        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        if size is None and isinstance(lfs, dict):
            size = lfs.get("size")
        if isinstance(size, int) and size > 0:
            total += size
    return total or None


class DownloadProgressTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "active": False,
            "message": "No model download is running.",
            "repo_id": "",
            "revision": "",
            "target": "",
            "bytes_downloaded": 0,
            "total_bytes": None,
            "percent": None,
            "started_at": None,
            "updated_at": None,
            "elapsed_seconds": 0.0,
            "error": "",
        }

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._state.get("active"))

    def start(self, *, repo_id: str, revision: str, target: Path, total_bytes: int | None) -> None:
        now = time.time()
        with self._lock:
            self._state = {
                "status": "downloading",
                "active": True,
                "message": "Downloading model files from Hugging Face.",
                "repo_id": repo_id,
                "revision": revision,
                "target": str(target),
                "bytes_downloaded": _directory_size(target),
                "total_bytes": total_bytes,
                "percent": None,
                "started_at": now,
                "updated_at": now,
                "elapsed_seconds": 0.0,
                "error": "",
            }

    def complete(self, message: str) -> None:
        now = time.time()
        with self._lock:
            target = Path(str(self._state.get("target") or ""))
            downloaded = _directory_size(target)
            total = self._state.get("total_bytes")
            self._state.update(
                {
                    "status": "complete",
                    "active": False,
                    "message": message,
                    "bytes_downloaded": downloaded,
                    "total_bytes": total or downloaded or None,
                    "percent": 100.0,
                    "updated_at": now,
                    "elapsed_seconds": now - float(self._state.get("started_at") or now),
                    "error": "",
                }
            )

    def fail(self, error: str) -> None:
        now = time.time()
        with self._lock:
            target = Path(str(self._state.get("target") or ""))
            self._state.update(
                {
                    "status": "error",
                    "active": False,
                    "message": "Model download failed.",
                    "bytes_downloaded": _directory_size(target),
                    "updated_at": now,
                    "elapsed_seconds": now - float(self._state.get("started_at") or now),
                    "error": error,
                }
            )

    def status(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            state = dict(self._state)
            target_text = str(state.get("target") or "")
            if target_text:
                state["bytes_downloaded"] = _directory_size(Path(target_text))
            total = state.get("total_bytes")
            downloaded = int(state.get("bytes_downloaded") or 0)
            if state.get("status") == "complete":
                state["percent"] = 100.0
            elif isinstance(total, int) and total > 0:
                state["percent"] = min(100.0, downloaded / total * 100)
            else:
                state["percent"] = None
            started_at = state.get("started_at")
            state["elapsed_seconds"] = now - float(started_at) if started_at else 0.0
            state["updated_at"] = now
            return state


def _require_huggingface_hub() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelDependencyError(
            "huggingface_hub is missing. Install requirements with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return snapshot_download


def _download_model_snapshot(settings: AppSettings, progress: DownloadProgressTracker | None = None) -> Path:
    if not settings.model_repo_id:
        raise ValueError("Set model_repo_id in config.local.json before downloading.")

    snapshot_download = _require_huggingface_hub()
    target = Path(settings.model_download_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.start(
            repo_id=settings.model_repo_id,
            revision=settings.model_revision or "main",
            target=target,
            total_bytes=_estimate_hf_repo_size(settings),
        )
    try:
        snapshot_download(
            repo_id=settings.model_repo_id,
            revision=settings.model_revision or None,
            local_dir=str(target),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        if progress:
            progress.fail(str(exc))
        raise
    return target


class ExtractionModelManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.device: str | None = None
        self.torch_dtype: Any | None = None
        self.download_progress = DownloadProgressTracker()

    def hardware_status(self) -> dict[str, Any]:
        return detect_gpu_memory_policy(self.settings)

    def status(self) -> dict[str, Any]:
        model_path = Path(self.settings.model_path).expanduser()
        model_download_dir = Path(self.settings.model_download_dir).expanduser()
        hardware = self.hardware_status()
        return {
            "loaded": self.model is not None and self.tokenizer is not None,
            "backend": "transformers",
            "model_path": str(model_path),
            "model_path_exists": model_path.exists(),
            "model_repo_id": self.settings.model_repo_id,
            "model_revision": self.settings.model_revision,
            "model_download_dir": str(model_download_dir),
            "model_download_dir_exists": model_download_dir.exists(),
            "device": self.device,
            "hardware": hardware,
            "selected_quantization": hardware["selected_quantization"],
        }

    def download_model(self) -> dict[str, Any]:
        if self.download_progress.is_active():
            raise RuntimeError("A model download is already running.")
        target = _download_model_snapshot(self.settings, progress=self.download_progress)
        self.settings.model_path = str(target)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.torch_dtype = None
        message = f"Downloaded merged model to {target}."
        self.download_progress.complete(message)
        return self.status() | {"message": message, "download": self.download_status()}

    def download_status(self) -> dict[str, Any]:
        return self.download_progress.status()

    def _require_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelDependencyError(
                "Model dependencies are missing. Install them with: "
                "python -m pip install -r requirements.txt"
            ) from exc
        return torch, AutoTokenizer, AutoModelForCausalLM

    def _resolve_device(self, torch: Any) -> tuple[str, Any]:
        preference = self.settings.device
        gpu_id = self.settings.gpu_id
        if preference == "auto":
            if torch.cuda.is_available():
                if gpu_id is not None:
                    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                        raise RuntimeError(f"CUDA GPU {gpu_id} is not available.")
                    return f"cuda:{gpu_id}", torch.float16
                return "cuda", torch.float16
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if gpu_id is not None:
                    raise RuntimeError("--gpu-id can only be used with CUDA.")
                return "mps", torch.float16
            return "cpu", torch.float32
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            if gpu_id is not None:
                if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA GPU {gpu_id} is not available.")
                return f"cuda:{gpu_id}", torch.float16
            return "cuda", torch.float16
        if preference == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS was requested but is not available.")
            return "mps", torch.float16
        return "cpu", torch.float32

    @staticmethod
    def _load_tokenizer(AutoTokenizer: Any, model_path: Path) -> Any:
        last_error: Exception | None = None
        for use_fast in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is None:
                        raise RuntimeError("Tokenizer has neither a pad token nor an eos token.")
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                return tokenizer
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Could not load tokenizer.") from last_error

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        torch, AutoTokenizer, AutoModelForCausalLM = self._require_dependencies()
        model_path = Path(self.settings.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Merged model path does not exist: {model_path}. "
                "Download the model or set CHEM_USE_MODEL_PATH."
            )

        device, torch_dtype = self._resolve_device(torch)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        if hasattr(model, "config"):
            model.config.use_cache = True
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = True
            model.generation_config.do_sample = self.settings.do_sample
            if self.settings.do_sample:
                model.generation_config.temperature = self.settings.temperature
                model.generation_config.top_p = self.settings.top_p
            else:
                model.generation_config.temperature = None
                model.generation_config.top_p = None

        model.eval()
        model.to(device)
        tokenizer = self._load_tokenizer(AutoTokenizer, model_path)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.torch_dtype = torch_dtype

    def build_prompt(self, text: str) -> str:
        return self.settings.prompt_template.format(text=text)

    def generate(self, text: str) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.device is not None

        prompt = self.build_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = prompt

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.settings.do_sample:
            kwargs["temperature"] = self.settings.temperature
            kwargs["top_p"] = self.settings.top_p

        import torch

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class MlxExtractionModelManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.download_progress = DownloadProgressTracker()

    def _mlx_quantization_label(self) -> str:
        model_path = str(self.settings.model_path).lower()
        if "8bit" in model_path or "8-bit" in model_path:
            return "mlx-8bit"
        if "4bit" in model_path or "4-bit" in model_path:
            return "mlx-4bit"
        return "mlx"

    def hardware_status(self) -> dict[str, Any]:
        hardware = detect_gpu_memory_policy(self.settings)
        if hardware.get("accelerator") != "mps":
            return hardware

        hardware = hardware.copy()
        hardware["accelerator"] = "mlx-metal"
        hardware["name"] = "MLX on Apple Metal"
        hardware["fp8_recommended"] = False
        hardware["selected_quantization"] = self._mlx_quantization_label()
        hardware["vllm_flags"] = []

        warnings: list[str] = []
        total_memory_gb = hardware.get("total_memory_gb")
        if (
            isinstance(total_memory_gb, (int, float))
            and total_memory_gb < self.settings.low_gpu_memory_warning_gb
        ):
            warnings.append(
                f"Detected {total_memory_gb:.1f} GB unified memory, below the "
                f"{self.settings.low_gpu_memory_warning_gb:.0f} GB warning threshold. "
                "MLX 8-bit may be tight; use short inputs or switch to MLX 4-bit."
            )
        hardware["warnings"] = warnings
        hardware["warning"] = " ".join(warnings)
        return hardware

    def status(self) -> dict[str, Any]:
        model_path = Path(self.settings.model_path).expanduser()
        model_download_dir = Path(self.settings.model_download_dir).expanduser()
        hardware = self.hardware_status()
        return {
            "loaded": self.model is not None and self.tokenizer is not None,
            "backend": "mlx",
            "model_path": str(model_path),
            "model_path_exists": model_path.exists(),
            "model_repo_id": self.settings.model_repo_id,
            "model_revision": self.settings.model_revision,
            "model_download_dir": str(model_download_dir),
            "model_download_dir_exists": model_download_dir.exists(),
            "device": hardware["accelerator"],
            "hardware": hardware,
            "selected_quantization": hardware["selected_quantization"],
        }

    def download_model(self) -> dict[str, Any]:
        if self.download_progress.is_active():
            raise RuntimeError("A model download is already running.")
        target = _download_model_snapshot(self.settings, progress=self.download_progress)
        message = (
            f"Downloaded merged Hugging Face model to {target}. "
            "The MLX backend needs an MLX-converted model; convert this folder with mlx_lm "
            "or set extraction_backend to transformers."
        )
        self.download_progress.complete(message)
        return self.status() | {
            "message": message,
            "downloaded_model_path": str(target),
            "download": self.download_status(),
        }

    def download_status(self) -> dict[str, Any]:
        return self.download_progress.status()

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ModelDependencyError(
                "mlx-lm is missing. Install it on Apple Silicon with: python -m pip install mlx-lm"
            ) from exc

        model_path = Path(self.settings.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"MLX model path does not exist: {model_path}")

        self.model, self.tokenizer = load(str(model_path))

    def build_prompt(self, text: str) -> str:
        return self.settings.prompt_template.format(text=text)

    def generate(self, text: str) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.build_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = prompt

        sampler = make_sampler(
            temp=self.settings.temperature,
            top_p=self.settings.top_p if self.settings.do_sample else 0.0,
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=input_text,
            max_tokens=self.settings.max_new_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()


def build_extraction_model_manager(settings: AppSettings) -> Any:
    if settings.extraction_backend == "mlx":
        return MlxExtractionModelManager(settings)
    if settings.extraction_backend != "transformers":
        raise ValueError(f"Unsupported extraction backend: {settings.extraction_backend}")
    return ExtractionModelManager(settings)
