from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .parser import normalize_text
from .settings import AppSettings


CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_SSL_CONTEXT: ssl.SSLContext | None = None


def _valid_cas(value: str) -> bool:
    digits = value.replace("-", "")
    checksum = int(digits[-1])
    total = sum(int(digit) * multiplier for multiplier, digit in enumerate(reversed(digits[:-1]), start=1))
    return total % 10 == checksum


def _default_ssl_context() -> ssl.SSLContext:
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            import certifi

            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


def _json_get(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "chem-use-web/0.1"})
    with urllib.request.urlopen(request, timeout=timeout, context=_default_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def _text_get(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "chem-use-web/0.1"})
    with urllib.request.urlopen(request, timeout=timeout, context=_default_ssl_context()) as response:
        return response.read().decode("utf-8", errors="replace").strip()


class ChemicalResolver:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.cache_path = Path(settings.cache_dir).expanduser() / "chemical_cache.json"
        self.cache: dict[str, Any] = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def _cache_key(self, chemical: str) -> str:
        return normalize_text(chemical).casefold()

    def resolve(self, chemical: str) -> dict[str, Any]:
        clean = normalize_text(chemical)
        if not clean:
            return self._empty_result(clean)
        key = self._cache_key(clean)
        cached = self.cache.get(key)
        now = time.time()
        if cached and now - float(cached.get("cached_at", 0)) < self.settings.chemical_cache_ttl_seconds:
            payload = cached.get("payload") or {}
            if not self._failed_payload(payload):
                return payload

        result = self._empty_result(clean)
        self._resolve_pubchem(clean, result)
        self._resolve_cactus(clean, result)
        self._resolve_chembl(clean, result)
        result["links"].update(self._standard_links(clean, result))

        if not self._failed_payload(result):
            self.cache[key] = {"cached_at": now, "payload": result}
            self._save_cache()
        return result

    @staticmethod
    def _failed_payload(payload: dict[str, Any]) -> bool:
        identifiers = payload.get("identifiers") or {}
        has_identifier = any(
            identifiers.get(key)
            for key in ["pubchem_cid", "cas", "inchi_key", "inchi", "smiles", "chembl_id"]
        )
        return bool(payload.get("errors")) and not has_identifier and not payload.get("sources")

    @staticmethod
    def _empty_result(chemical: str) -> dict[str, Any]:
        return {
            "query": chemical,
            "identifiers": {
                "pubchem_cid": "",
                "cas": [],
                "inchi_key": "",
                "inchi": "",
                "smiles": "",
                "iupac_name": "",
                "molecular_formula": "",
                "molecular_weight": "",
                "chembl_id": "",
            },
            "links": {},
            "sources": [],
            "errors": [],
        }

    def _resolve_pubchem(self, chemical: str, result: dict[str, Any]) -> None:
        q = urllib.parse.quote(chemical, safe="")
        timeout = self.settings.chemical_api_timeout_seconds
        try:
            cid_payload = _json_get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON",
                timeout,
            )
            cids = cid_payload.get("IdentifierList", {}).get("CID", [])
            if not cids:
                return
            cid = str(cids[0])
            result["identifiers"]["pubchem_cid"] = cid
            result["links"]["pubchem"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"

            props = _json_get(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{cid}/property/IUPACName,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,"
                "MolecularFormula,MolecularWeight/JSON",
                timeout,
            )
            rows = props.get("PropertyTable", {}).get("Properties", [])
            if rows:
                row = rows[0]
                result["identifiers"]["iupac_name"] = str(row.get("IUPACName") or "")
                result["identifiers"]["smiles"] = str(row.get("CanonicalSMILES") or row.get("IsomericSMILES") or "")
                result["identifiers"]["inchi"] = str(row.get("InChI") or "")
                result["identifiers"]["inchi_key"] = str(row.get("InChIKey") or "")
                result["identifiers"]["molecular_formula"] = str(row.get("MolecularFormula") or "")
                result["identifiers"]["molecular_weight"] = str(row.get("MolecularWeight") or "")

            try:
                synonyms = _json_get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
                    timeout,
                )
                syn_list = synonyms.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                cas_numbers = []
                for item in syn_list:
                    for candidate in CAS_RE.findall(str(item)):
                        if _valid_cas(candidate) and candidate not in cas_numbers:
                            cas_numbers.append(candidate)
                    if len(cas_numbers) >= 10:
                        break
                result["identifiers"]["cas"] = cas_numbers
            except Exception as exc:
                result["errors"].append(f"PubChem synonyms: {exc}")

            result["sources"].append("PubChem")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            result["errors"].append(f"PubChem: {exc}")

    def _resolve_cactus(self, chemical: str, result: dict[str, Any]) -> None:
        q = urllib.parse.quote(chemical, safe="")
        timeout = self.settings.chemical_api_timeout_seconds
        base = f"https://cactus.nci.nih.gov/chemical/structure/{q}"
        try:
            if not result["identifiers"]["smiles"]:
                smiles = _text_get(f"{base}/smiles", timeout)
                if smiles and not smiles.startswith("<!DOCTYPE"):
                    result["identifiers"]["smiles"] = smiles
            if not result["identifiers"]["inchi_key"]:
                inchikey = _text_get(f"{base}/stdinchikey", timeout)
                if inchikey and not inchikey.startswith("<!DOCTYPE"):
                    result["identifiers"]["inchi_key"] = inchikey
            result["sources"].append("NCI CACTUS")
        except (urllib.error.URLError, TimeoutError) as exc:
            result["errors"].append(f"NCI CACTUS: {exc}")

    def _resolve_chembl(self, chemical: str, result: dict[str, Any]) -> None:
        timeout = self.settings.chemical_api_timeout_seconds
        query = result["identifiers"].get("inchi_key") or chemical
        q = urllib.parse.quote(query, safe="")
        try:
            payload = _json_get(
                f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={q}&limit=1",
                timeout,
            )
            molecules = payload.get("molecules", [])
            if molecules:
                chembl_id = molecules[0].get("molecule_chembl_id") or ""
                result["identifiers"]["chembl_id"] = chembl_id
                if chembl_id:
                    result["links"]["chembl"] = f"https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}"
                    result["sources"].append("ChEMBL")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            result["errors"].append(f"ChEMBL: {exc}")

    @staticmethod
    def _standard_links(chemical: str, result: dict[str, Any]) -> dict[str, str]:
        q = urllib.parse.quote(chemical)
        links = {
            "pubchem_search": f"https://pubchem.ncbi.nlm.nih.gov/#query={q}",
            "comptox_search": f"https://comptox.epa.gov/dashboard/chemical/search-results?search={q}",
            "cas_common_chemistry": f"https://commonchemistry.cas.org/results?q={q}",
            "cactus": f"https://cactus.nci.nih.gov/chemical/structure/{q}/names",
            "chembl_search": f"https://www.ebi.ac.uk/chembl/g/#search_results/all/query={q}",
        }
        inchikey = result.get("identifiers", {}).get("inchi_key")
        if inchikey:
            links["wikidata_search"] = (
                "https://www.wikidata.org/w/index.php?search="
                f"{urllib.parse.quote(inchikey)}"
            )
        return links
