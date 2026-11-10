from __future__ import annotations

import re
from typing import Any

from .parser import normalize_text
from .rag import snippet
from .settings import AppSettings


_HARMONY_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


class RagGenerationError(RuntimeError):
    pass


def clean_generated_answer(value: object) -> str:
    text = _HARMONY_TOKEN_RE.sub("", str(value or "")).replace("<|endoftext|>", "")
    return text.strip()


def build_rag_messages(
    query: str,
    results: list[dict[str, Any]],
    *,
    max_context_chars: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    clean_query = normalize_text(query)
    context, citations = build_rag_context(results, max_context_chars=max_context_chars)
    system = (
        "You are a careful chemical-use retrieval-augmented generation assistant. "
        "Answer using only the retrieved records supplied by the user. "
        "Do not use outside knowledge. "
        "Every factual claim about a chemical, use, or evidence must cite record ids like [1]. "
        "If the retrieved records do not support an answer, say that the retrieved records do not provide enough evidence."
    )
    user = f"""
Question:
\"\"\"{clean_query}\"\"\"

Retrieved records:
{context}

Write a concise grounded answer. Use record citations such as [1] or [2] next to the claims they support.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], citations


def build_rag_context(
    results: list[dict[str, Any]],
    *,
    max_context_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    max_chars = max(1200, int(max_context_chars or 0))
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    used_chars = 0

    for index, row in enumerate(results, start=1):
        chemical = normalize_text(row.get("chemical") or row.get("chemicals") or "")
        use_text = normalize_text(row.get("use") or row.get("uses") or "")
        evidence = snippet(row.get("paragraph") or row.get("paragraph_preview") or "", 900)
        source = normalize_text(row.get("path") or "")
        source_row_number = row.get("source_row_number") or ""
        pair_index = row.get("pair_index") or ""

        block = (
            f"[{index}]\n"
            f"Chemical: {chemical}\n"
            f"Use: {use_text}\n"
            f"Source: {source or 'not provided'}\n"
            f"Source row: {source_row_number}\n"
            f"Pair index: {pair_index}\n"
            f"Evidence: {evidence or 'not provided'}"
        )
        block_length = len(block) + 2
        if blocks and used_chars + block_length > max_chars:
            break
        if not blocks and block_length > max_chars:
            available = max(250, max_chars - (block_length - len(evidence)))
            evidence = snippet(evidence, available)
            block = (
                f"[{index}]\n"
                f"Chemical: {chemical}\n"
                f"Use: {use_text}\n"
                f"Source: {source or 'not provided'}\n"
                f"Source row: {source_row_number}\n"
                f"Pair index: {pair_index}\n"
                f"Evidence: {evidence or 'not provided'}"
            )
            block_length = len(block) + 2

        blocks.append(block)
        used_chars += block_length
        citations.append(
            {
                "id": index,
                "rowid": row.get("rowid"),
                "chemical": chemical,
                "use": use_text,
                "source": source,
                "source_row_number": source_row_number,
                "pair_index": pair_index,
            }
        )

    return "\n\n".join(blocks), citations


class RagAnswerer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._transformers_model: Any | None = None
        self._transformers_tokenizer: Any | None = None
        self._transformers_device: str | None = None

    def answer(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {
                "provider": "transformers",
                "model": self.settings.rag_transformers_model,
                "answer": "",
                "citations": [],
                "context_count": 0,
                "message": "No matching records were retrieved, so no grounded RAG answer was generated.",
            }

        messages, citations = build_rag_messages(
            query,
            results,
            max_context_chars=self.settings.rag_answer_max_context_chars,
        )
        return self._answer_with_transformers(messages, citations)

    def _resolve_transformers_device(self, torch: Any) -> tuple[str, Any]:
        preference = str(self.settings.rag_transformers_device or "auto").strip().lower()
        if preference == "auto":
            if torch.cuda.is_available():
                return "cuda", torch.float16
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", torch.float16
            return "cpu", torch.float32
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise RagGenerationError("RAG Transformers device is set to CUDA, but CUDA is not available.")
            return "cuda", torch.float16
        if preference == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RagGenerationError("RAG Transformers device is set to MPS, but MPS is not available.")
            return "mps", torch.float16
        if preference == "cpu":
            return "cpu", torch.float32
        raise RagGenerationError("rag_transformers_device must be one of auto, cuda, mps, or cpu.")

    def _load_transformers_model(self) -> tuple[Any, Any, str]:
        if self._transformers_model is not None and self._transformers_tokenizer is not None:
            assert self._transformers_device is not None
            return self._transformers_model, self._transformers_tokenizer, self._transformers_device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RagGenerationError(
                "Direct Python RAG answers require torch and transformers. "
                "Install requirements with: python -m pip install -r chem_use_web/requirements.txt"
            ) from exc

        model_name = str(self.settings.rag_transformers_model or "").strip()
        if not model_name:
            raise RagGenerationError("Set rag_transformers_model before using the Transformers RAG provider.")

        device, torch_dtype = self._resolve_transformers_device(torch)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as exc:
            raise RagGenerationError(
                f"Could not load RAG tokenizer '{model_name}'. "
                "Check the model name/path or download access."
            ) from exc
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except TypeError:
            load_kwargs.pop("low_cpu_mem_usage", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:
            raise RagGenerationError(
                f"Could not load RAG model '{model_name}'. "
                "If this is the first run, make sure the machine can download it from Hugging Face, "
                "or set rag_transformers_model to a local model directory."
            ) from exc

        model.eval()
        try:
            model.to(device)
        except Exception as exc:
            raise RagGenerationError(f"Could not move RAG model '{model_name}' to {device}.") from exc

        self._transformers_model = model
        self._transformers_tokenizer = tokenizer
        self._transformers_device = device
        return model, tokenizer, device

    def _answer_with_transformers(
        self,
        messages: list[dict[str, str]],
        citations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        model, tokenizer, device = self._load_transformers_model()
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = "\n\n".join(
                f"{message['role'].upper()}:\n{message['content']}" for message in messages
            ) + "\n\nASSISTANT:\n"

        max_input_tokens = max(512, int(self.settings.rag_transformers_max_input_tokens or 0))
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)

        generation_kwargs = {
            "max_new_tokens": int(self.settings.rag_answer_max_tokens),
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        try:
            import torch

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            raise RagGenerationError(f"Direct Transformers RAG generation failed: {exc}") from exc

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "provider": "transformers",
            "model": self.settings.rag_transformers_model,
            "device": device,
            "answer": clean_generated_answer(answer),
            "citations": citations,
            "context_count": len(citations),
        }
