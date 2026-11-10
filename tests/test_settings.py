from __future__ import annotations

from pathlib import Path
import unittest

from chem_use_web.settings import WORKSPACE_ROOT, _resolve_workspace_path


class SettingsPathTests(unittest.TestCase):
    def test_relative_paths_resolve_from_project_root(self) -> None:
        self.assertEqual(
            Path(_resolve_workspace_path("models/Llama-CUE")),
            (WORKSPACE_ROOT / "models" / "Llama-CUE").resolve(),
        )

    def test_legacy_chem_use_web_prefix_still_resolves_to_project_root(self) -> None:
        self.assertEqual(
            Path(_resolve_workspace_path("chem_use_web/models/Llama-CUE")),
            (WORKSPACE_ROOT / "models" / "Llama-CUE").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
