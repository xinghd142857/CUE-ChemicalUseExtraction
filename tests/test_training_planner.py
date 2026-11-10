from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from chem_use_web.hardware import GIB, build_gpu_memory_policy
from chem_use_web.training.planner import TrainingPlanError, build_training_plan


class TrainingPlannerTests(unittest.TestCase):
    def _csv_path(self, tmpdir: str, fieldnames: list[str], row: dict[str, str]) -> Path:
        path = Path(tmpdir) / "train.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_lora_plan_parses_huggingface_url_and_validates_columns(self) -> None:
        hardware = build_gpu_memory_policy(
            accelerator="cuda",
            total_memory_bytes=24 * GIB,
            fp8_supported=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._csv_path(
                tmpdir,
                ["paragraph", "model outputs"],
                {
                    "paragraph": "Water is used as a solvent.",
                    "model outputs": '{"Water": "solvent"}',
                },
            )
            plan = build_training_plan(
                {
                    "strategy": "lora",
                    "model_link": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
                    "training_data_path": str(path),
                    "input_column": "paragraph",
                    "output_column": "model outputs",
                },
                hardware=hardware,
                base_dir=Path(tmpdir),
            )

        self.assertEqual(plan["model"]["repo_id"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(plan["model"]["estimated_parameters_b"], 8.0)
        self.assertEqual(plan["dataset"]["row_count"], 1)
        self.assertIn(plan["estimate"]["status"], {"ok", "tight"})

    def test_missing_dpo_columns_are_user_facing_errors(self) -> None:
        hardware = build_gpu_memory_policy(
            accelerator="cuda",
            total_memory_bytes=24 * GIB,
            fp8_supported=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._csv_path(
                tmpdir,
                ["prompt", "chosen"],
                {"prompt": "Extract uses", "chosen": '{"Water": "solvent"}'},
            )
            with self.assertRaises(TrainingPlanError) as context:
                build_training_plan(
                    {
                        "strategy": "dpo",
                        "model_link": "Qwen/Qwen2.5-7B-Instruct",
                        "training_data_path": str(path),
                        "prompt_column": "prompt",
                        "chosen_column": "chosen",
                        "rejected_column": "rejected",
                    },
                    hardware=hardware,
                    base_dir=Path(tmpdir),
                )

        self.assertIn("Missing required training column", str(context.exception))

    def test_dpo_estimate_is_larger_than_lora_for_same_model(self) -> None:
        hardware = build_gpu_memory_policy(
            accelerator="cuda",
            total_memory_bytes=16 * GIB,
            fp8_supported=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._csv_path(
                tmpdir,
                ["prompt", "chosen", "rejected", "paragraph", "model outputs"],
                {
                    "prompt": "Extract uses",
                    "chosen": '{"Water": "solvent"}',
                    "rejected": '{"Water": "unknown"}',
                    "paragraph": "Water is used as a solvent.",
                    "model outputs": '{"Water": "solvent"}',
                },
            )
            lora = build_training_plan(
                {
                    "strategy": "lora",
                    "model_link": "Qwen/Qwen2.5-7B-Instruct",
                    "training_data_path": str(path),
                    "input_column": "paragraph",
                    "output_column": "model outputs",
                },
                hardware=hardware,
                base_dir=Path(tmpdir),
            )
            dpo = build_training_plan(
                {
                    "strategy": "dpo",
                    "model_link": "Qwen/Qwen2.5-7B-Instruct",
                    "training_data_path": str(path),
                    "prompt_column": "prompt",
                    "chosen_column": "chosen",
                    "rejected_column": "rejected",
                },
                hardware=hardware,
                base_dir=Path(tmpdir),
            )

        self.assertGreater(dpo["estimate"]["estimated_memory_gb"], lora["estimate"]["estimated_memory_gb"])


if __name__ == "__main__":
    unittest.main()
