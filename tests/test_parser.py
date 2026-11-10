from __future__ import annotations

import unittest

from chem_use_web.parser import count_words, parse_extraction_output


class ParserTests(unittest.TestCase):
    def test_parses_python_dict_and_splits_pipe_uses(self) -> None:
        parsed = parse_extraction_output("{'TATB': 'coating modifier | explosive filler'}")
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.pairs[0]["chemical"], "TATB")
        self.assertEqual(parsed.pairs[0]["uses"], ["coating modifier", "explosive filler"])

    def test_empty_dict_is_no_result(self) -> None:
        parsed = parse_extraction_output("{}")
        self.assertEqual(parsed.status, "no_result")
        self.assertEqual(parsed.pairs, [])

    def test_empty_string_is_no_result(self) -> None:
        parsed = parse_extraction_output("")
        self.assertEqual(parsed.status, "no_result")

    def test_textual_empty_output_is_no_result(self) -> None:
        parsed = parse_extraction_output("No chemical-use pairs found.")
        self.assertEqual(parsed.status, "no_result")

    def test_malformed_output_is_parse_error(self) -> None:
        parsed = parse_extraction_output("TATB: coating modifier")
        self.assertEqual(parsed.status, "parse_error")

    def test_word_count(self) -> None:
        self.assertEqual(count_words("alpha beta gamma"), 3)


if __name__ == "__main__":
    unittest.main()
