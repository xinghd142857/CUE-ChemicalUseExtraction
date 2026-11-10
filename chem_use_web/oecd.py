from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .parser import normalize_text
from .settings import AppSettings


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

OECD_CATEGORY_BLOCK = (
    "Ablative; Abrasive; Absorbent; Adhesion/cohesion promoter; Adsorbent; "
    "Aerating and deaerating agent; Alloying element; Anti-adhesive/cohesive; "
    "Anticaking agent; Anticondensation agent; Antifreeze agent; Antioxidant; "
    "Antiredeposition agent; Antiscaling agent; Antislip/friction agent; "
    "Antistain agent; Antistatic agent; Anti-streaking agent; Binder; Biocide; "
    "Bleaching agent; Brightener; Catalyst; Chain transfer agent; Chelating "
    "agent; Chemical reaction regulator; Cleaning agent; Cloud-point depressant; "
    "Coalescing agent; Compatibilizer; Conductive agent; Corrosion inhibitor; "
    "Crystal growth modifiers (nucleating agents); Deflocculant; Defoamer; "
    "Dehydrating agent (desiccant); Demulsifier; Density modifier; Deodoriser; "
    "Diluent; Dispersing agent; Drier; Dust suppressant; Dusting agent; Dye; "
    "Elasticiser; Embalming agent; Emulsifier; Energy releasers (explosives, "
    "motive propellant); Etching agent; Explosion inhibitor; Filler; Film "
    "former; Finishing agent; Fire extinguishing agent; Fixing agent "
    "(mordant); Flame retardant; Flavouring and nutrient; Flocculating agent; "
    "Flotation agent; Flow promoter; Flux agent; Foaming agent; Fragrance; "
    "Freeze-thaw additive; Fuel; Fuel additive; Gelling modifier; Hardener; "
    "Heat stabiliser; Heat transferring agent; Humectant; Hydraulic fluids; "
    "Impregnation agent; Incandescent agent; Insulators; Intermediate; Ion "
    "exchange agent; Leaching agent; Lubricating agent; Luminescent agent; "
    "Magnetic element; Monomers; Opacifer; Oxidizing agent; pH regulating "
    "agent; Photochemical; Pigment; Plasticiser; Plating agent; Polymerisation "
    "promoter; Preservative; Pressure transfer agent; Processing aids not "
    "otherwise specified; Propellants, non-motive (blowing agents); Reactive "
    "cleaning/removal agent; Reducing agent; Refrigerants; Sealant (barrier); "
    "Semiconductor and photovoltaic agent; Sizing agent; Softener; Soil "
    "amendments (including fertilisers); Solids separation (precipitating) "
    "agent; Solubility enhancer; Solvent; Stabilising agent; Surface modifier; "
    "Surfactant; Swelling agent; Tackifier; Tanning agents; Terminator/blocker; "
    "Thickener / thickening agent; Tracer; UV stabiliser; Vapour pressure "
    "modifier; Vehicle (carrier); Viscosity modifier; Waterproofing agent; "
    "Wetting agent (non-aqueous); X-Ray absorber; No specific technical "
    "function; Other."
)

OECD_CATEGORIES = [item.strip().rstrip(".") for item in OECD_CATEGORY_BLOCK.split(";")]
_LOOKUP = {re.sub(r"\s+", " ", item).casefold(): item for item in OECD_CATEGORIES}
_HARMONY_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
OECD_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": OECD_CATEGORIES},
            "minItems": 3,
            "maxItems": 3,
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def build_oecd_prompt(use_text: str) -> str:
    return f"""
You are an expert chemical classification model.
Classify the given chemical-use description into the top three OECD Functional Use Categories.

Categories:
{OECD_CATEGORY_BLOCK}

Rules:
1. Return ONLY a pure list with three category names.
2. Use category names exactly as written in the category list.
3. Do not explain.

Description:
\"\"\"{normalize_text(use_text)}\"\"\"
""".strip()


def build_oecd_structured_prompt(use_text: str) -> str:
    return f"""
You are an expert chemical classification model.
Classify the given chemical-use description into the top three OECD Functional Use Categories.

Categories:
{OECD_CATEGORY_BLOCK}

Rules:
1. Return ONLY a JSON object with this shape: {{"labels": ["label 1", "label 2", "label 3"]}}.
2. Use exactly three category names.
3. Use category names exactly as written in the category list.
4. Do not explain.

Description:
\"\"\"{normalize_text(use_text)}\"\"\"
""".strip()


def _canonicalize(value: str) -> str | None:
    key = re.sub(r"\s+", " ", normalize_text(value).rstrip(".")).casefold()
    return _LOOKUP.get(key)


def parse_oecd_output(text: str) -> list[str]:
    cleaned = _HARMONY_TOKEN_RE.sub("", str(text or "")).replace("<|endoftext|>", "").strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    candidate = cleaned[start : end + 1] if start != -1 and end != -1 and start < end else cleaned

    parsed: Any = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
            break
        except Exception:
            continue

    labels: list[str] = []
    if isinstance(parsed, dict):
        parsed = parsed.get("labels")
    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            label = _canonicalize(str(item))
            if label and label not in labels:
                labels.append(label)

    if labels:
        return labels[:3]

    lowered = cleaned.casefold()
    positions: list[tuple[int, str]] = []
    for category in OECD_CATEGORIES:
        pos = lowered.find(category.casefold())
        if pos != -1:
            positions.append((pos, category))
    for _, category in sorted(positions, key=lambda item: item[0]):
        if category not in labels:
            labels.append(category)
    return labels[:3]


class OecdClassifier:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def status(self) -> dict[str, Any]:
        model_dir = Path(self.settings.oecd_model_dir).expanduser()
        return {
            "provider": self.settings.oecd_provider,
            "model_id": self.settings.oecd_model_id,
            "model_dir": str(model_dir),
            "model_dir_exists": model_dir.exists(),
            "ollama_base_url": self.settings.oecd_ollama_base_url,
            "ollama_model": self.settings.oecd_ollama_model,
            "vllm_base_url": self.settings.oecd_vllm_base_url,
            "vllm_model": self.settings.oecd_vllm_model,
            "loaded": self.model is not None and self.tokenizer is not None,
            "temperature": self.settings.oecd_temperature,
            "do_sample": self.settings.oecd_do_sample,
        }

    def download(self) -> dict[str, Any]:
        if self.settings.oecd_provider == "ollama":
            status = self.status()
            status["message"] = (
                "OECD provider is Ollama. Pull the model with: ollama pull "
                f"{self.settings.oecd_ollama_model}"
            )
            return status
        if self.settings.oecd_provider == "vllm":
            status = self.status()
            status["message"] = "OECD provider is vLLM, so no local model download is needed."
            return status

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install requirements, then retry the download."
            ) from exc

        model_dir = Path(self.settings.oecd_model_dir).expanduser().resolve()
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.settings.oecd_model_id,
            local_dir=str(model_dir),
            token=os.environ.get("HF_TOKEN"),
        )
        return self.status()

    def check_provider(self) -> dict[str, Any]:
        status = self.status()
        provider = self.settings.oecd_provider
        if provider == "ollama":
            return status | self._check_openai_compatible_provider(
                provider_name="Ollama",
                base_url=self.settings.oecd_ollama_base_url,
                model_name=self.settings.oecd_ollama_model,
                api_key=self.settings.oecd_ollama_api_key,
                missing_model_hint=f"Run: ollama pull {self.settings.oecd_ollama_model}",
            )
        if provider == "vllm":
            return status | self._check_openai_compatible_provider(
                provider_name="vLLM",
                base_url=self.settings.oecd_vllm_base_url,
                model_name=self.settings.oecd_vllm_model,
                api_key=self.settings.oecd_vllm_api_key,
                missing_model_hint=(
                    f"Start vLLM with {self.settings.oecd_vllm_model}, or download that model first."
                ),
            )
        if provider == "local":
            ready = Path(self.settings.oecd_model_dir).expanduser().exists()
            return status | {
                "provider_reachable": ready,
                "model_available": ready,
                "ready": ready,
                "available_models": [],
                "message": (
                    "Local OECD model is available."
                    if ready
                    else "Local OECD model is missing. Download gpt-oss before enabling OECD labels."
                ),
            }
        raise ValueError("oecd_provider must be 'ollama', 'vllm', or 'local'.")

    def load(self) -> None:
        if self.settings.oecd_provider in {"ollama", "vllm"}:
            return

        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "OECD model dependencies are missing. Install requirements, then retry."
            ) from exc

        model_dir = Path(self.settings.oecd_model_dir).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(
                f"OECD model is not downloaded at {model_dir}. Use the download button first."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def classify(self, use_text: str) -> list[str]:
        if self.settings.oecd_provider == "ollama":
            return self._classify_with_ollama_native(
                use_text,
                base_url=self.settings.oecd_ollama_base_url,
                model_name=self.settings.oecd_ollama_model,
                timeout_seconds=self.settings.oecd_ollama_timeout_seconds,
            )
        if self.settings.oecd_provider == "vllm":
            return self._classify_with_openai_compatible(
                use_text,
                provider_name="vLLM",
                base_url=self.settings.oecd_vllm_base_url,
                model_name=self.settings.oecd_vllm_model,
                api_key=self.settings.oecd_vllm_api_key,
                timeout_seconds=self.settings.oecd_vllm_timeout_seconds,
            )
        if self.settings.oecd_provider != "local":
            raise ValueError("oecd_provider must be 'ollama', 'vllm', or 'local'.")

        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = build_oecd_prompt(use_text)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = next(self.model.parameters()).device
        inputs = inputs.to(model_device)

        import torch

        kwargs = {
            "max_new_tokens": self.settings.oecd_max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        completion_ids = outputs[0][prompt_length:]
        raw_output = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        return parse_oecd_output(raw_output)

    def _check_openai_compatible_provider(
        self,
        *,
        provider_name: str,
        base_url: str,
        model_name: str,
        api_key: str,
        missing_model_hint: str,
    ) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.oecd_probe_timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {
                "provider_reachable": False,
                "model_available": False,
                "ready": False,
                "available_models": [],
                "message": f"{provider_name} port check failed with HTTP {exc.code}: {detail}",
            }
        except urllib.error.URLError:
            return {
                "provider_reachable": False,
                "model_available": False,
                "ready": False,
                "available_models": [],
                "message": (
                    f"{provider_name} port is not reachable at {url}. "
                    f"Start {provider_name}; if gpt-oss is not installed, {missing_model_hint}."
                ),
            }
        except json.JSONDecodeError:
            return {
                "provider_reachable": True,
                "model_available": False,
                "ready": False,
                "available_models": [],
                "message": (
                    f"{provider_name} port is reachable, but /models returned invalid JSON. "
                    f"{missing_model_hint}."
                ),
            }

        models = self._extract_model_ids(data)
        model_available = model_name in models
        if model_available:
            message = f"{provider_name} port is working and {model_name} is available."
        else:
            message = (
                f"{provider_name} port is working, but {model_name} is not available. "
                f"{missing_model_hint}."
            )
        return {
            "provider_reachable": True,
            "model_available": model_available,
            "ready": model_available,
            "available_models": models,
            "message": message,
        }

    @staticmethod
    def _extract_model_ids(data: Any) -> list[str]:
        items = data.get("data", []) if isinstance(data, dict) else []
        models: list[str] = []
        if not isinstance(items, list):
            return models
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id and model_id not in models:
                models.append(model_id)
        return models

    def _classify_with_openai_compatible(
        self,
        use_text: str,
        *,
        provider_name: str,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout_seconds: float,
    ) -> list[str]:
        base_url = base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": build_oecd_prompt(use_text)}],
            "temperature": 0,
            "max_tokens": self.settings.oecd_max_new_tokens,
        }
        if self.settings.oecd_temperature:
            payload["temperature"] = self.settings.oecd_temperature

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{provider_name} OECD request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach {provider_name} at {url}. "
                f"Start {provider_name} or update the configured base URL."
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected {provider_name} response shape: {data}") from exc
        labels = parse_oecd_output(content)
        if not labels:
            raise RuntimeError(
                f"{provider_name} returned a response, but no valid OECD labels could be parsed."
            )
        return labels

    def _classify_with_ollama_native(
        self,
        use_text: str,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
    ) -> list[str]:
        url = f"{self._ollama_native_base_url(base_url)}/api/chat"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": build_oecd_structured_prompt(use_text)}],
            "stream": False,
            "think": False,
            "format": OECD_LABEL_SCHEMA,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama OECD request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {url}. Start Ollama or update the configured base URL."
            ) from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response shape: {data}") from exc
        labels = parse_oecd_output(content)
        if not labels:
            raise RuntimeError("Ollama returned a response, but no valid OECD labels could be parsed.")
        return labels

    @staticmethod
    def _ollama_native_base_url(base_url: str) -> str:
        trimmed = base_url.rstrip("/")
        return trimmed[:-3] if trimmed.endswith("/v1") else trimmed
