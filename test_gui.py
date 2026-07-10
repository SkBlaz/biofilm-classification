"""Tests for the local MicroICS GUI safety helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gui.app import folder_status, validate_config


class TestFolderStatus(unittest.TestCase):
    def test_missing_folder_is_safe_to_create(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "new-results"

            self.assertEqual(
                folder_status(str(missing)),
                {"path": str(missing), "exists": False, "item_count": 0, "items": [], "has_more": False},
            )

    def test_non_empty_folder_reports_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results = Path(temporary_directory)
            for name in ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv", "f.tsv"):
                (results / name).touch()

            status = folder_status(str(results))

            self.assertTrue(status["exists"])
            self.assertEqual(status["item_count"], 6)
            self.assertEqual(status["items"], ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"])
            self.assertTrue(status["has_more"])


class TestCleanupValidation(unittest.TestCase):
    def test_training_rejects_non_empty_results_without_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_images = root / "training-images"
            results = root / "results"
            training_images.mkdir()
            results.mkdir()
            (results / "previous-run.tsv").touch()

            config = {
                "workflow": "train",
                "training_images": str(training_images),
                "results_dir": str(results),
                "confirm_cleanup": False,
            }

            with self.assertRaisesRegex(ValueError, "not empty"):
                validate_config(config)

    def test_training_accepts_non_empty_results_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_images = root / "training-images"
            results = root / "results"
            training_images.mkdir()
            results.mkdir()
            (results / "previous-run.tsv").touch()

            config = validate_config(
                {
                    "workflow": "train",
                    "training_images": str(training_images),
                    "results_dir": str(results),
                    "confirm_cleanup": True,
                }
            )

            self.assertEqual(config["results_dir"], str(results.resolve()))
            self.assertTrue(config["confirm_cleanup"])


if __name__ == "__main__":
    unittest.main()
