"""Tests for the local MicroICS GUI safety helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gui.app import default_config, folder_status, generated_feature_path, gui_feature_validation_path, validate_config


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
    def test_feature_generation_rejects_non_empty_results_without_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_images = root / "training-images"
            results = root / "results"
            training_images.mkdir()
            results.mkdir()
            (results / "previous-run.tsv").touch()

            config = {
                "workflow": "features_labelled",
                "training_images": str(training_images),
                "results_dir": str(results),
                "confirm_cleanup": False,
            }

            with self.assertRaisesRegex(ValueError, "not empty"):
                validate_config(config)

    def test_feature_generation_accepts_non_empty_results_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_images = root / "training-images"
            results = root / "results"
            training_images.mkdir()
            results.mkdir()
            (results / "previous-run.tsv").touch()

            config = validate_config(
                {
                    "workflow": "features_labelled",
                    "training_images": str(training_images),
                    "results_dir": str(results),
                    "confirm_cleanup": True,
                }
            )

            self.assertEqual(config["results_dir"], str(results.resolve()))
            self.assertTrue(config["confirm_cleanup"])

    def test_train_models_reuses_existing_complete_table_without_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            results = root / "results"
            images.mkdir()
            results.mkdir()
            (results / "datafile.tsv").write_text("sampleName\tlabel\tfeature\nsample\tA\t1\n", encoding="utf-8")

            config = validate_config(
                {
                    "workflow": "train",
                    "training_images": str(images),
                    "results_dir": str(results),
                    "confirm_cleanup": False,
                }
            )

            self.assertEqual(config["workflow"], "train")
            self.assertFalse(config["confirm_cleanup"])

    def test_train_models_requires_a_complete_table(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            results = root / "results"
            images.mkdir()
            results.mkdir()

            with self.assertRaisesRegex(ValueError, "complete feature table"):
                validate_config(
                    {
                        "workflow": "train",
                        "training_images": str(images),
                        "results_dir": str(results),
                    }
                )

    def test_complete_feature_table_is_a_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            results = root / "results"
            feature_file = root / "complete.tsv"
            images.mkdir()
            results.mkdir()
            feature_file.write_text("sampleName\tlabel\tfeature\nsample\tA\t1\n", encoding="utf-8")

            config = validate_config(
                {
                    "workflow": "train",
                    "training_images": str(images),
                    "results_dir": str(results),
                    "feature_file": str(feature_file),
                }
            )

            self.assertEqual(config["feature_file"], str(feature_file.resolve()))

    def test_imaging_position_is_a_supported_replication_unit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            images = Path(temporary_directory)
            results = images / "results"

            config = validate_config(
                {
                    "workflow": "features_labelled",
                    "training_images": str(images),
                    "results_dir": str(results),
                    "replication_unit": "position",
                }
            )

            self.assertEqual(config["replication_unit"], "position")

    def test_unlabelled_generation_validates_unknown_features_table(self):
        path = generated_feature_path(
            {
                "workflow": "features_unlabelled",
                "results_dir": "/tmp/microics-unlabelled-results",
            }
        )

        self.assertEqual(path.name, "unknown_features.tsv")

    def test_gui_validation_report_avoids_docker_owned_validation_folder(self):
        path = gui_feature_validation_path({"results_dir": "/tmp/microics-results"})

        self.assertEqual(path, Path("/tmp/microics-results/gui_feature_validation.json"))


class TestWorkflowInterface(unittest.TestCase):
    def test_workflow_is_ordered_and_host_monitor_is_removed(self):
        html = (Path(__file__).parent / "gui" / "index.html").read_text(encoding="utf-8")
        javascript = (Path(__file__).parent / "gui" / "app.js").read_text(encoding="utf-8")

        workflows = [
            'class="workflow-stage"',
            'data-workflow="train"',
            'data-workflow="inference"',
            'data-workflow="full"',
        ]
        positions = [html.index(workflow) for workflow in workflows]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("Host monitor", html)
        self.assertNotIn("Host monitor", javascript)
        self.assertIn("Imaging position (pos)", html)
        self.assertIn("MicroICS uses this table as-is and does not extract or join features automatically.", html)
        self.assertIn("<strong>Images per label</strong>", javascript)
        self.assertIn('<details class="validation-secondary">', javascript)
        self.assertIn("<summary>Images per label per date</summary>", javascript)
        self.assertEqual(default_config()["workflow"], "features_labelled")


if __name__ == "__main__":
    unittest.main()
