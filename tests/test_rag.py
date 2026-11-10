from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import unittest

from chem_use_web.rag import KirkRagSearch, RagUnavailableError, make_fts_query, parse_extracted_pairs
from chem_use_web.settings import AppSettings


class RagParsingTests(unittest.TestCase):
    def test_index_parser_splits_pipe_separated_uses(self) -> None:
        pairs = parse_extracted_pairs("{'TATB': 'surface modifier | energetic material additive'}")
        self.assertEqual(
            pairs,
            [
                {"chemical": "TATB", "use": "surface modifier"},
                {"chemical": "TATB", "use": "energetic material additive"},
            ],
        )


class RagSearchTests(unittest.TestCase):
    def test_fts_query_uses_prefix_matching_for_words(self) -> None:
        self.assertEqual(make_fts_query("insect"), '"insect"*')
        self.assertEqual(make_fts_query("TATB insect"), '"tatb"* OR "insect"*')

    def test_search_matches_insecticide_with_insect_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.csv"
            index_path = Path(tmpdir) / "rag.sqlite"
            with source_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["original paragraph", "model outputs"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "original paragraph": "Permethrin is described as an insecticide formulation.",
                        "model outputs": "{'Permethrin': 'insecticide | insect repellent'}",
                    }
                )

            settings = AppSettings(
                rag_source_csv=str(source_path),
                rag_index_path=str(index_path),
                rag_text_column="original paragraph",
                rag_output_column="model outputs",
            )
            searcher = KirkRagSearch(settings)
            searcher.build_index(chunksize=2)

            result = searcher.search("insect", top_k=10)

            self.assertEqual(result["count"], 2)
            self.assertEqual(
                [row["use"] for row in result["results"]],
                ["insecticide", "insect repellent"],
            )

    def test_search_rejects_stale_index_when_source_is_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.csv"
            index_path = Path(tmpdir) / "rag.sqlite"
            with source_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["original paragraph", "model outputs"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "original paragraph": "TATB was coated to improve processing.",
                        "model outputs": "{'TATB': 'surface modifier'}",
                    }
                )

            build_settings = AppSettings(
                rag_source_csv=str(source_path),
                rag_index_path=str(index_path),
                rag_text_column="original paragraph",
                rag_output_column="model outputs",
            )
            KirkRagSearch(build_settings).build_index(chunksize=2)

            stale_settings = AppSettings(
                rag_source_csv="",
                rag_index_path=str(index_path),
                rag_text_column="original paragraph",
                rag_output_column="model outputs",
            )

            with self.assertRaisesRegex(RagUnavailableError, "Choose a RAG knowledge file path"):
                KirkRagSearch(stale_settings).search("TATB", top_k=10)


if __name__ == "__main__":
    unittest.main()
