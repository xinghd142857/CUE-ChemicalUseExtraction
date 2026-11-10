from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chem_use_web.core import ChemicalUseWebApp
from chem_use_web.settings import load_settings


class FakeModelManager:
    def generate(self, text: str) -> str:
        if "empty" in text.lower():
            return "{}"
        return "{'Water': 'solvent | cleaning agent', 'Ethanol': 'fuel'}"

    def status(self) -> dict:
        return {"loaded": True}


class CorePipelineTests(unittest.TestCase):
    def test_real_text_pipeline_with_fake_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ChemicalUseWebApp(load_settings(overrides={"results_dir": tmpdir, "cache_dir": tmpdir}))
            app.model_manager = FakeModelManager()  # type: ignore[assignment]
            result = app.extract_text(
                "Water is used as a solvent and ethanol is used as a fuel.",
                enable_oecd=False,
                enrich_chemicals=False,
            )
        self.assertEqual(result["summary"]["statuses"], {"success": 3})
        self.assertEqual(result["records"][0]["chemical"], "Water")
        self.assertEqual(result["records"][0]["use"], "solvent")

    def test_empty_model_output_is_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ChemicalUseWebApp(load_settings(overrides={"results_dir": tmpdir, "cache_dir": tmpdir}))
            app.model_manager = FakeModelManager()  # type: ignore[assignment]
            result = app.extract_text("empty paragraph", enable_oecd=False, enrich_chemicals=False)
        self.assertEqual(result["summary"]["statuses"], {"no_result": 1})

    def test_download_model_uses_configured_hf_repo(self) -> None:
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "chem_use_web.inference._require_huggingface_hub",
            return_value=fake_snapshot_download,
        ), patch(
            "chem_use_web.inference._estimate_hf_repo_size",
            return_value=1234,
        ):
            download_dir = str(Path(tmpdir) / "base")
            app = ChemicalUseWebApp(
                load_settings(
                    overrides={
                        "results_dir": tmpdir,
                        "cache_dir": tmpdir,
                        "model_repo_id": "Jesolo/Llama-CUE",
                        "model_download_dir": download_dir,
                    }
                )
            )
            result = app.download_model()

        self.assertEqual(calls[0]["repo_id"], "Jesolo/Llama-CUE")
        self.assertEqual(calls[0]["revision"], "main")
        self.assertEqual(Path(calls[0]["local_dir"]).resolve(), Path(download_dir).resolve())
        self.assertTrue(result["model_path_exists"])
        self.assertEqual(result["download"]["status"], "complete")
        self.assertEqual(app.model_download_status()["status"], "complete")


if __name__ == "__main__":
    unittest.main()
