from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch

from chem_use_web.rag_answer import RagAnswerer, build_rag_messages
from chem_use_web.settings import load_settings


SAMPLE_RESULTS = [
    {
        "rowid": 11,
        "chemical": "TATB",
        "use": "improves the process forming performance of TATB-based explosive",
        "path": "patent-A",
        "source_row_number": 3,
        "pair_index": 1,
        "paragraph": "The invention modifies the surface of TATB crystals and improves process forming performance.",
        "paragraph_preview": "The invention modifies the surface of TATB crystals.",
    }
]


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeTorchModule:
    @staticmethod
    def inference_mode():
        return FakeInferenceMode()


class FakeInputIds:
    shape = (1, 3)


class FakeInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": FakeInputIds()})

    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "chat prompt"

    def __call__(self, text, return_tensors, truncation, max_length):
        self.call_args = {
            "text": text,
            "return_tensors": return_tensors,
            "truncation": truncation,
            "max_length": max_length,
        }
        return FakeInputs()

    def decode(self, token_ids, skip_special_tokens):
        self.decode_args = {
            "token_ids": token_ids,
            "skip_special_tokens": skip_special_tokens,
        }
        return "TATB is linked to explosive forming performance [1]."


class FakeModel:
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[101, 102, 103, 201, 202]]


class RagAnswerTests(unittest.TestCase):
    def test_builds_grounded_messages_with_citations(self) -> None:
        messages, citations = build_rag_messages(
            "What chemicals improve explosive forming performance?",
            SAMPLE_RESULTS,
            max_context_chars=4000,
        )

        self.assertIn("only the retrieved records", messages[0]["content"])
        self.assertIn("Do not use outside knowledge", messages[0]["content"])
        self.assertIn("TATB", messages[1]["content"])
        self.assertIn("[1]", messages[1]["content"])
        self.assertEqual(citations[0]["chemical"], "TATB")
        self.assertEqual(citations[0]["id"], 1)

    def test_answers_with_direct_transformers_provider(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(sys.modules, {"torch": FakeTorchModule()}):
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "rag_transformers_model": "fake-small-rag-model",
                    "rag_transformers_device": "cpu",
                    "rag_transformers_max_input_tokens": 1024,
                    "rag_answer_max_tokens": 64,
                }
            )
            answerer = RagAnswerer(settings)
            answerer._transformers_model = model
            answerer._transformers_tokenizer = tokenizer
            answerer._transformers_device = "cpu"

            result = answerer.answer("What is TATB used for?", SAMPLE_RESULTS)

        self.assertEqual(result["provider"], "transformers")
        self.assertEqual(result["model"], "fake-small-rag-model")
        self.assertEqual(result["device"], "cpu")
        self.assertIn("[1]", result["answer"])
        self.assertEqual(tokenizer.call_args["max_length"], 1024)
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 64)
        self.assertFalse(model.generate_kwargs["do_sample"])

    def test_empty_retrieval_does_not_call_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_settings(
                overrides={
                    "cache_dir": tmpdir,
                    "results_dir": tmpdir,
                    "rag_transformers_model": "fake-small-rag-model",
                }
            )
            result = RagAnswerer(settings).answer("missing?", [])

        self.assertEqual(result["provider"], "transformers")
        self.assertEqual(result["model"], "fake-small-rag-model")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["context_count"], 0)
        self.assertIn("No matching records", result["message"])


if __name__ == "__main__":
    unittest.main()
