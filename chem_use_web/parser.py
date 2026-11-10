from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
_HARMONY_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


@dataclass
class ParsedExtraction:
    status: str
    pairs: list[dict[str, Any]]
    cleaned_output: str
    error: str = ""


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def count_words(value: str) -> int:
    return len(re.findall(r"\b\S+\b", value or ""))


def clean_model_output(raw_output: Any) -> str:
    text = str(raw_output or "").strip()
    if "<|channel|>final<|message|>" in text:
        text = text.rsplit("<|channel|>final<|message|>", maxsplit=1)[-1]
    elif "<|message|>" in text:
        text = text.rsplit("<|message|>", maxsplit=1)[-1]
    text = _HARMONY_TOKEN_RE.sub("", text)
    text = text.replace("<|endoftext|>", "")
    text = _CODE_FENCE_RE.sub("", text)
    return text.strip()


def _extract_braced_candidate(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def _parse_literal(text: str) -> Any:
    candidate = _extract_braced_candidate(text)
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(candidate)
        except Exception:
            continue
    return None


def _split_uses(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        chunks = []
        for item in value:
            chunks.extend(_split_uses(item))
    elif isinstance(value, dict):
        for key in ("uses", "use", "function", "functions", "application", "applications"):
            if key in value:
                return _split_uses(value[key])
        chunks = [json.dumps(value, ensure_ascii=False)]
    else:
        chunks = [str(value)]

    seen: set[str] = set()
    uses: list[str] = []
    for chunk in chunks:
        cleaned = normalize_text(chunk)
        if cleaned and cleaned not in seen:
            uses.append(cleaned)
            seen.add(cleaned)
    return uses


def _pair_from_mapping(payload: dict[Any, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for raw_chemical, raw_uses in payload.items():
        chemical = normalize_text(raw_chemical)
        if not chemical:
            continue
        uses = _split_uses(raw_uses)
        if uses:
            pairs.append({"chemical": chemical, "uses": uses})
    return pairs


def _pair_from_sequence(payload: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            chemical = normalize_text(
                item.get("chemical")
                or item.get("compound")
                or item.get("name")
                or item.get("substance")
            )
            uses = _split_uses(
                item.get("uses")
                or item.get("use")
                or item.get("function")
                or item.get("application")
                or item.get("applications")
            )
            if chemical and uses:
                pairs.append({"chemical": chemical, "uses": uses})
    return pairs


def parse_extraction_output(raw_output: Any) -> ParsedExtraction:
    cleaned = clean_model_output(raw_output)
    empty_markers = {"", "{}", "[]", "none", "null", "no", "n/a"}
    if cleaned.strip().lower() in empty_markers:
        return ParsedExtraction(status="no_result", pairs=[], cleaned_output=cleaned)
    lowered = cleaned.strip().lower()
    if lowered.startswith("no ") and any(token in lowered for token in ("compound", "chemical", "pair", "result", "found")):
        return ParsedExtraction(status="no_result", pairs=[], cleaned_output=cleaned)

    payload = _parse_literal(cleaned)
    if payload is None:
        return ParsedExtraction(
            status="parse_error",
            pairs=[],
            cleaned_output=cleaned,
            error="Model output was not a parseable Python/JSON dictionary.",
        )

    if isinstance(payload, dict):
        pairs = _pair_from_mapping(payload)
    elif isinstance(payload, (list, tuple)):
        pairs = _pair_from_sequence(payload)
    else:
        pairs = []

    if not pairs:
        return ParsedExtraction(status="no_result", pairs=[], cleaned_output=cleaned)
    return ParsedExtraction(status="success", pairs=pairs, cleaned_output=cleaned)
