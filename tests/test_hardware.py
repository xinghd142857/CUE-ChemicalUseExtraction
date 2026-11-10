from __future__ import annotations

import unittest
from unittest.mock import patch

from chem_use_web.hardware import GIB, _system_memory_bytes, build_gpu_memory_policy


class HardwarePolicyTests(unittest.TestCase):
    def test_cuda_16gb_selects_fp8_when_supported(self) -> None:
        policy = build_gpu_memory_policy(
            accelerator="cuda",
            total_memory_bytes=16 * GIB,
            fp8_supported=True,
            compute_capability="8.9",
        )

        self.assertTrue(policy["fp8_recommended"])
        self.assertEqual(policy["selected_quantization"], "fp8")
        self.assertEqual(policy["vllm_flags"], ["--quantization", "fp8", "--kv-cache-dtype", "fp8"])

    def test_below_12gb_warns(self) -> None:
        policy = build_gpu_memory_policy(
            accelerator="cuda",
            total_memory_bytes=8 * GIB,
            fp8_supported=True,
            compute_capability="8.9",
        )

        self.assertEqual(policy["selected_quantization"], "fp8")
        self.assertIn("below the 12 GB warning threshold", policy["warning"])

    def test_large_gpu_keeps_default_precision(self) -> None:
        policy = build_gpu_memory_policy(
            accelerator="cuda",
            total_memory_bytes=24 * GIB,
            fp8_supported=True,
            compute_capability="8.9",
        )

        self.assertFalse(policy["fp8_recommended"])
        self.assertEqual(policy["selected_quantization"], "none")

    def test_mps_does_not_select_fp8(self) -> None:
        policy = build_gpu_memory_policy(
            accelerator="mps",
            total_memory_bytes=16 * GIB,
            fp8_supported=False,
        )

        self.assertFalse(policy["fp8_recommended"])
        self.assertEqual(policy["selected_quantization"], "none")
        self.assertIn("MPS", policy["warning"])

    def test_windows_system_memory_uses_windows_api(self) -> None:
        with patch("chem_use_web.hardware.sys.platform", "win32"), patch(
            "chem_use_web.hardware._windows_memory_bytes",
            return_value=32 * GIB,
        ):
            self.assertEqual(_system_memory_bytes(), 32 * GIB)


if __name__ == "__main__":
    unittest.main()
