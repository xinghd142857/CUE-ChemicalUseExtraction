from __future__ import annotations

import os
import platform
import sys
from typing import Any

from .settings import AppSettings


GIB = 1024**3


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / GIB, 2)


def build_gpu_memory_policy(
    *,
    accelerator: str,
    total_memory_bytes: int | None,
    name: str = "",
    index: int | None = None,
    compute_capability: str = "",
    fp8_supported: bool = False,
    fp8_threshold_gb: float = 16.0,
    low_memory_warning_gb: float = 12.0,
) -> dict[str, Any]:
    total_memory_gb = bytes_to_gib(total_memory_bytes)
    fp8_recommended = (
        total_memory_gb is not None
        and total_memory_gb <= fp8_threshold_gb
        and accelerator in {"cuda", "rocm"}
    )
    selected_quantization = "fp8" if fp8_recommended and fp8_supported else "none"
    warnings: list[str] = []

    if total_memory_gb is not None and total_memory_gb < low_memory_warning_gb:
        warnings.append(
            f"Detected {total_memory_gb:.1f} GB accelerator memory, below the "
            f"{low_memory_warning_gb:.0f} GB warning threshold. Use very short "
            "context, low concurrency, or a smaller quantized model."
        )

    if fp8_recommended and fp8_supported:
        warnings.append(
            f"Detected <= {fp8_threshold_gb:.0f} GB GPU memory. FP8 is selected "
            "for supported vLLM serving."
        )
    elif fp8_recommended and not fp8_supported:
        warnings.append(
            f"Detected <= {fp8_threshold_gb:.0f} GB GPU memory, but FP8 is not "
            "reported as supported on this accelerator. Use a supported vLLM GPU "
            "or a smaller quantized model."
        )
    elif accelerator == "mps":
        warnings.append(
            "Apple MPS uses unified memory and does not support this app's FP8 "
            "serving policy. Use the MLX backend with an 8-bit model on Mac, "
            "or a CUDA/ROCm vLLM server for FP8."
        )

    vllm_flags: list[str] = []
    if selected_quantization == "fp8":
        vllm_flags = ["--quantization", "fp8", "--kv-cache-dtype", "fp8"]

    return {
        "accelerator": accelerator,
        "name": name,
        "index": index,
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gb": total_memory_gb,
        "compute_capability": compute_capability,
        "fp8_supported": fp8_supported,
        "fp8_recommended": fp8_recommended,
        "selected_quantization": selected_quantization,
        "warning": " ".join(warnings),
        "warnings": warnings,
        "vllm_flags": vllm_flags,
    }


def detect_gpu_memory_policy(settings: AppSettings) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return _torchless_hardware_policy(settings) | {
            "detection_error": str(exc),
            "torch_available": False,
        }

    preference = str(settings.device or "auto").lower()
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        index = settings.gpu_id if settings.gpu_id is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        is_rocm = bool(getattr(torch.version, "hip", None))
        accelerator = "rocm" if is_rocm else "cuda"
        major = int(getattr(props, "major", 0) or 0)
        minor = int(getattr(props, "minor", 0) or 0)
        compute_capability = "" if is_rocm else f"{major}.{minor}"
        fp8_supported = is_rocm or major > 8 or (major == 8 and minor >= 9)
        return build_gpu_memory_policy(
            accelerator=accelerator,
            total_memory_bytes=int(props.total_memory),
            name=str(getattr(props, "name", "")),
            index=index,
            compute_capability=compute_capability,
            fp8_supported=fp8_supported,
            fp8_threshold_gb=settings.fp8_gpu_memory_threshold_gb,
            low_memory_warning_gb=settings.low_gpu_memory_warning_gb,
        )

    if preference in {"auto", "mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return build_gpu_memory_policy(
            accelerator="mps",
            total_memory_bytes=_system_memory_bytes(),
            name="Apple Metal Performance Shaders",
            fp8_supported=False,
            fp8_threshold_gb=settings.fp8_gpu_memory_threshold_gb,
            low_memory_warning_gb=settings.low_gpu_memory_warning_gb,
        )

    return build_gpu_memory_policy(
        accelerator="cpu",
        total_memory_bytes=_system_memory_bytes(),
        name="CPU",
        fp8_supported=False,
        fp8_threshold_gb=settings.fp8_gpu_memory_threshold_gb,
        low_memory_warning_gb=settings.low_gpu_memory_warning_gb,
    )


def _torchless_hardware_policy(settings: AppSettings) -> dict[str, Any]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return build_gpu_memory_policy(
            accelerator="mps",
            total_memory_bytes=_system_memory_bytes(),
            name="Apple Silicon unified memory",
            fp8_supported=False,
            fp8_threshold_gb=settings.fp8_gpu_memory_threshold_gb,
            low_memory_warning_gb=settings.low_gpu_memory_warning_gb,
        )

    return build_gpu_memory_policy(
        accelerator="cpu",
        total_memory_bytes=_system_memory_bytes(),
        name="CPU",
        fp8_supported=False,
        fp8_threshold_gb=settings.fp8_gpu_memory_threshold_gb,
        low_memory_warning_gb=settings.low_gpu_memory_warning_gb,
    )


def _system_memory_bytes() -> int | None:
    if sys.platform == "win32":
        return _windows_memory_bytes()
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            return page_size * page_count
        except (ValueError, OSError):
            pass
    return None


def _windows_memory_bytes() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)
