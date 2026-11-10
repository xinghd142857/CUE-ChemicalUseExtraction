from __future__ import annotations

import csv
import io
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .parser import parse_extraction_output
from .settings import AppSettings


DEFAULT_CHUNKSIZE = 5000
INDEX_SCHEMA_VERSION = "5"
QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class RagUnavailableError(RuntimeError):
    pass


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_cell_text(value: object) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(clean_cell_text(item) for item in value if clean_cell_text(item))
    if isinstance(value, dict):
        return "; ".join(
            f"{clean_cell_text(key)}: {clean_cell_text(item)}"
            for key, item in value.items()
            if clean_cell_text(key) or clean_cell_text(item)
        )
    return " ".join(str(value).split())


def parse_extracted_pairs(value: object) -> list[dict[str, str]]:
    parsed = parse_extraction_output(value)
    if parsed.status != "success":
        return []

    pairs: list[dict[str, str]] = []
    for pair in parsed.pairs:
        chemical_text = clean_cell_text(pair.get("chemical"))
        for use in pair.get("uses") or []:
            use_text = clean_cell_text(use)
            if chemical_text and use_text:
                pairs.append({"chemical": chemical_text, "use": use_text})
    return pairs


def format_pair(pair: dict[str, str]) -> str:
    return f"{pair['chemical']}: {pair['use']}"


def make_fts_query(query: str) -> str:
    terms = QUERY_TOKEN_RE.findall(query.lower())
    if not terms:
        return ""
    return " OR ".join(
        f'"{term}"*' if len(term) >= 3 else f'"{term}"'
        for term in terms[:24]
    )


def snippet(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def iter_source_frames(path: Path, *, chunksize: int = DEFAULT_CHUNKSIZE):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=chunksize)
        return
    if suffix == ".tsv":
        yield from pd.read_csv(path, sep="\t", chunksize=chunksize)
        return
    if suffix in {".xlsx", ".xls"}:
        yield pd.read_excel(path)
        return
    raise RagUnavailableError("RAG knowledge file must be CSV, TSV, XLSX, or XLS.")


class KirkRagSearch:
    def __init__(self, settings: AppSettings) -> None:
        self.source_csv = Path(settings.rag_source_csv).expanduser() if settings.rag_source_csv else None
        self.index_path = Path(settings.rag_index_path).expanduser()
        self.path_column = settings.rag_path_column
        self.text_column = settings.rag_text_column
        self.output_column = settings.rag_output_column

    def status(self) -> dict[str, Any]:
        index_exists = self.index_path.exists()
        current_source_csv = str(self.source_csv) if self.source_csv else ""
        source_exists = bool(self.source_csv and self.source_csv.exists())
        pair_count = 0
        source_row_count = 0
        indexed_at = ""
        source_csv = ""
        schema_version = ""
        schema_ok = False
        source_ok = False
        output_column = ""
        output_column_ok = False
        path_column = ""
        text_column = ""
        path_column_ok = False
        text_column_ok = False

        if index_exists:
            try:
                with sqlite3.connect(self.index_path) as conn:
                    pair_count = int(conn.execute("SELECT COUNT(*) FROM kirk_fts").fetchone()[0])
                    meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
                    indexed_at = meta.get("indexed_at", "")
                    source_csv = meta.get("source_csv", "")
                    schema_version = meta.get("schema_version", "")
                    source_row_count = int(meta.get("source_row_count", "0") or 0)
                    output_column = meta.get("output_column", "")
                    path_column = meta.get("path_column", "")
                    text_column = meta.get("text_column", "")
                    schema_ok = schema_version == INDEX_SCHEMA_VERSION
                    source_ok = source_csv == current_source_csv
                    output_column_ok = output_column == self.output_column
                    path_column_ok = path_column == self.path_column
                    text_column_ok = text_column == self.text_column
            except sqlite3.Error:
                index_exists = False

        return {
            "available": (
                source_exists
                and index_exists
                and schema_ok
                and source_ok
                and output_column_ok
                and path_column_ok
                and text_column_ok
                and pair_count > 0
            ),
            "source_csv": current_source_csv,
            "source_exists": source_exists,
            "index_path": str(self.index_path),
            "index_exists": index_exists,
            "row_count": pair_count,
            "pair_count": pair_count,
            "source_row_count": source_row_count,
            "indexed_at": indexed_at,
            "indexed_source_csv": source_csv,
            "schema_version": schema_version,
            "expected_schema_version": INDEX_SCHEMA_VERSION,
            "indexed_output_column": output_column,
            "output_column": self.output_column,
            "indexed_path_column": path_column,
            "indexed_text_column": text_column,
            "path_column": self.path_column,
            "text_column": self.text_column,
            "needs_reindex": source_exists
            and (
                not index_exists
                or not schema_ok
                or not source_ok
                or not output_column_ok
                or not path_column_ok
                or not text_column_ok
                or pair_count == 0
            ),
        }

    def build_index(self, *, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
        if not self.source_csv:
            raise RagUnavailableError("Enter a RAG knowledge file path.")
        if not self.source_csv.exists():
            raise RagUnavailableError(f"RAG knowledge file was not found: {self.source_csv}")
        if not self.output_column:
            raise RagUnavailableError("Enter the prediction/model-output column name.")
        if not self.text_column:
            raise RagUnavailableError("Enter the evidence/original-paragraph column name.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        for stale_path in [
            self.index_path,
            Path(f"{self.index_path}-wal"),
            Path(f"{self.index_path}-shm"),
        ]:
            if stale_path.exists():
                stale_path.unlink()

        started = time.time()
        pair_count = 0
        source_row_count = 0
        with sqlite3.connect(self.index_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE VIRTUAL TABLE kirk_fts USING fts5(
                    source_path UNINDEXED,
                    source_row_number UNINDEXED,
                    pair_index UNINDEXED,
                    chemical,
                    use_text,
                    chemical_use_pair,
                    raw_output UNINDEXED,
                    paragraph UNINDEXED
                )
                """
            )
            conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")

            for chunk in iter_source_frames(self.source_csv, chunksize=chunksize):
                required_columns = [self.text_column, self.output_column]
                if self.path_column:
                    required_columns.append(self.path_column)
                missing = [
                    column
                    for column in required_columns
                    if column not in chunk.columns
                ]
                if missing:
                    raise RagUnavailableError(
                        f"RAG source CSV is missing columns {missing}. "
                        f"Available columns: {list(chunk.columns)}"
                    )

                chunk = chunk.fillna("")
                rows = []
                for source_index, row in chunk.iterrows():
                    source_row_count += 1
                    raw_output = str(row.get(self.output_column, ""))
                    paragraph = str(row.get(self.text_column, ""))
                    source_path = str(row.get(self.path_column, "")) if self.path_column else ""
                    pairs = parse_extracted_pairs(raw_output)
                    for pair_index, pair in enumerate(pairs, start=1):
                        rows.append(
                            (
                                source_path,
                                int(source_index) + 1,
                                pair_index,
                                pair["chemical"],
                                pair["use"],
                                format_pair(pair),
                                raw_output,
                                paragraph,
                            )
                        )

                if rows:
                    conn.executemany(
                        """
                        INSERT INTO kirk_fts(
                            source_path,
                            source_row_number,
                            pair_index,
                            chemical,
                            use_text,
                            chemical_use_pair,
                            raw_output,
                            paragraph
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    pair_count += len(rows)
                    conn.commit()

            metadata = {
                "source_csv": str(self.source_csv),
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "row_count": str(pair_count),
                "pair_count": str(pair_count),
                "source_row_count": str(source_row_count),
                "schema_version": INDEX_SCHEMA_VERSION,
                "output_column": self.output_column,
                "path_column": self.path_column,
                "text_column": self.text_column,
            }
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            conn.commit()

        elapsed_seconds = round(time.time() - started, 2)
        return {
            "ok": True,
            "message": (
                f"Indexed {pair_count:,} chemical-use pairs "
                f"from {source_row_count:,} source rows in {elapsed_seconds} seconds."
            ),
            "row_count": pair_count,
            "pair_count": pair_count,
            "source_row_count": source_row_count,
            "elapsed_seconds": elapsed_seconds,
            "index_path": str(self.index_path),
        }

    def search(self, query: str, *, top_k: int = 10) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise RagUnavailableError("Enter a search query.")
        if top_k < 1:
            raise RagUnavailableError("top_k must be at least 1.")
        if top_k > 50:
            top_k = 50

        if not self.index_path.exists():
            raise RagUnavailableError("RAG index has not been built yet.")

        status = self.status()
        if not status.get("available"):
            if not self.source_csv:
                indexed_source = str(status.get("indexed_source_csv") or "")
                detail = f" The existing index was built from {indexed_source}." if indexed_source else ""
                raise RagUnavailableError(
                    "Choose a RAG knowledge file path and click Build/update search index before searching."
                    + detail
                )
            if not status.get("source_exists"):
                raise RagUnavailableError(f"RAG knowledge file was not found: {self.source_csv}")
            if status.get("needs_reindex"):
                raise RagUnavailableError("RAG index needs to be rebuilt for chemical/use search.")
            raise RagUnavailableError("RAG index is not ready. Check the knowledge file and column settings.")
        if status.get("needs_reindex"):
            raise RagUnavailableError("RAG index needs to be rebuilt for chemical/use search.")

        fts_query = make_fts_query(query)
        if not fts_query:
            raise RagUnavailableError("Search query must include letters or numbers.")

        with sqlite3.connect(self.index_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    rowid,
                    source_path,
                    source_row_number,
                    pair_index,
                    chemical,
                    use_text,
                    chemical_use_pair,
                    paragraph,
                    raw_output,
                    bm25(kirk_fts) AS score
                FROM kirk_fts
                WHERE kirk_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()

        results = [
            {
                "rowid": int(row["rowid"]),
                "score": float(row["score"]),
                "relevance": -float(row["score"]),
                "path": row["source_path"],
                "source_row_number": int(row["source_row_number"] or 0),
                "pair_index": int(row["pair_index"] or 0),
                "chemical": row["chemical"],
                "chemicals": row["chemical"],
                "use": row["use_text"],
                "uses": row["use_text"],
                "chemical_use_pair": row["chemical_use_pair"],
                "chemical_use_pairs": row["chemical_use_pair"],
                "parsed_uses": row["chemical_use_pair"],
                "raw_output": row["raw_output"],
                "paragraph": row["paragraph"],
                "paragraph_preview": snippet(row["paragraph"], 700),
            }
            for row in rows
        ]
        return {
            "ok": True,
            "query": query,
            "fts_query": fts_query,
            "count": len(results),
            "results": results,
        }

    def export_results_csv(self, results: list[dict[str, Any]]) -> bytes:
        fieldnames = [
            "rowid",
            "score",
            "relevance",
            "path",
            "source_row_number",
            "pair_index",
            "chemical",
            "use",
            "chemical_use_pair",
            "raw_output",
            "paragraph",
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in results)
        return buffer.getvalue().encode("utf-8-sig")
