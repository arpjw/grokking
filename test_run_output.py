"""Tests for keeping prior training results intact."""
from __future__ import annotations

import os
import tempfile
import unittest

from run_output import prepare_run_dir


class RunOutputTests(unittest.TestCase):
    def test_creates_fresh_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            out_dir = prepare_run_dir("trial-1", root)
            self.assertEqual(out_dir, os.path.join(root, "trial-1"))
            self.assertTrue(os.path.isdir(out_dir))

    def test_refuses_existing_results(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            out_dir = prepare_run_dir("trial", root)
            marker = os.path.join(out_dir, "metrics.csv")
            with open(marker, "w", encoding="utf-8") as f:
                f.write("previous run")
            with self.assertRaises(FileExistsError):
                prepare_run_dir("trial", root)
            with open(marker, encoding="utf-8") as f:
                self.assertEqual(f.read(), "previous run")

    def test_rejects_paths_outside_results(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for name in ("", "..", "../other", "/tmp/other", "nested/run"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    prepare_run_dir(name, root)


if __name__ == "__main__":
    unittest.main()
