from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from chem_use_web.oecd import OecdClassifier
from chem_use_web.settings import load_settings


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "[\"Solvent\", \"Cleaning agent\", \"Other\"]",
                        }
                    }
                ]
            }
        ).encode("utf-8")


class FakeOllamaNativeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "message": {
                    "content": "{\"labels\": [\"Solvent\", \"Cleaning agent\", \"Other\"]}",
                }
            }
        ).encode("utf-8")


class FakeModelsResponse:
    def __init__(self, model_ids: list[str]) -> None:
        self.model_ids = model_ids

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps({"data": [{"id": model_id} for model_id in self.model_ids]}).encode("utf-8")


class FakeUnparseableResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "message": {
                    "content": "This appears to describe an energetic material.",
                }
            }
        ).encode("utf-8")


def fake_vllm_urlopen(request, timeout):
    assert request.full_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert timeout == 5.0
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 96
    return FakeResponse()


def fake_ollama_urlopen(request, timeout):
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert timeout == 7.0
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "gpt-oss:20b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["format"]["properties"]["labels"]["minItems"] == 3
    assert body["format"]["properties"]["labels"]["maxItems"] == 3
    return FakeOllamaNativeResponse()


def fake_ollama_models_urlopen(request, timeout):
    assert request.full_url == "http://127.0.0.1:11434/v1/models"
    assert timeout == 3.0
    return FakeModelsResponse(["gpt-oss:20b", "llama3.2"])


def fake_ollama_models_missing_urlopen(request, timeout):
    assert request.full_url == "http://127.0.0.1:11434/v1/models"
    assert timeout == 3.0
    return FakeModelsResponse(["llama3.2"])


def fake_ollama_unparseable_urlopen(request, timeout):
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    return FakeUnparseableResponse()


def fake_ollama_unreachable_urlopen(request, timeout):
    raise urllib.error.URLError("connection refused")


class OecdOpenAiCompatibleTests(unittest.TestCase):
    def test_classifies_with_vllm_chat_completions_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch("urllib.request.urlopen", fake_vllm_urlopen):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "vllm",
                    "oecd_vllm_base_url": "http://127.0.0.1:8000/v1",
                    "oecd_vllm_timeout_seconds": 5.0,
                }
            )
            labels = OecdClassifier(settings).classify("used as a solvent in coatings")
        self.assertEqual(labels, ["Solvent", "Cleaning agent", "Other"])

    def test_oecd_provider_vllm_stays_vllm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "vllm",
                    "oecd_vllm_base_url": "http://127.0.0.1:8000/v1",
                    "oecd_vllm_model": "legacy-model",
                    "oecd_vllm_api_key": "EMPTY",
                    "oecd_vllm_timeout_seconds": 11.0,
                }
            )
        self.assertEqual(settings.oecd_provider, "vllm")
        self.assertEqual(settings.oecd_vllm_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(settings.oecd_vllm_model, "legacy-model")
        self.assertEqual(settings.oecd_vllm_api_key, "EMPTY")
        self.assertEqual(settings.oecd_vllm_timeout_seconds, 11.0)

    def test_classifies_with_ollama_chat_completions_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch("urllib.request.urlopen", fake_ollama_urlopen):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "ollama",
                    "oecd_ollama_base_url": "http://127.0.0.1:11434/v1",
                    "oecd_ollama_timeout_seconds": 7.0,
                }
            )
            labels = OecdClassifier(settings).classify("used as a solvent in coatings")
        self.assertEqual(labels, ["Solvent", "Cleaning agent", "Other"])

    def test_ollama_probe_reports_ready_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "urllib.request.urlopen",
            fake_ollama_models_urlopen,
        ):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "ollama",
                }
            )
            result = OecdClassifier(settings).check_provider()
        self.assertTrue(result["provider_reachable"])
        self.assertTrue(result["model_available"])
        self.assertTrue(result["ready"])

    def test_ollama_probe_reports_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "urllib.request.urlopen",
            fake_ollama_models_missing_urlopen,
        ):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "ollama",
                }
            )
            result = OecdClassifier(settings).check_provider()
        self.assertTrue(result["provider_reachable"])
        self.assertFalse(result["model_available"])
        self.assertIn("ollama pull gpt-oss:20b", result["message"])

    def test_ollama_probe_reports_unreachable_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "urllib.request.urlopen",
            fake_ollama_unreachable_urlopen,
        ):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "ollama",
                }
            )
            result = OecdClassifier(settings).check_provider()
        self.assertFalse(result["provider_reachable"])
        self.assertFalse(result["ready"])
        self.assertIn("not reachable", result["message"])

    def test_unparseable_ollama_response_raises_visible_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "urllib.request.urlopen",
            fake_ollama_unparseable_urlopen,
        ):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "oecd_provider": "ollama",
                }
            )
            with self.assertRaisesRegex(RuntimeError, "no valid OECD labels"):
                OecdClassifier(settings).classify("used as an explosive")


if __name__ == "__main__":
    unittest.main()
