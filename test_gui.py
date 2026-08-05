"""Tests for the local MicroICS GUI safety helpers."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gui.app import (
    command_for,
    create_env,
    default_config,
    folder_status,
    generated_feature_path,
    gui_feature_validation_path,
    receive_upload,
    reset_pipeline,
    runtime,
    state,
    validate_config,
)


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


class TestFeatureFileSelection(unittest.TestCase):
    def test_tab_separated_file_picker_stages_one_file(self):
        contents = b"sampleName\tlabel\tfeature\nsample-a\tA\t1\n"
        handler = SimpleNamespace(
            headers={
                "X-Upload-Name": "publication-data.tsv",
                "Content-Length": str(len(contents)),
            },
            rfile=io.BytesIO(contents),
        )

        with tempfile.TemporaryDirectory() as temporary_directory, patch("gui.app.UPLOAD_ROOT", Path(temporary_directory)):
            selected = receive_upload(handler, {".tsv", ".txt"}, "tab-separated .tsv or .txt files")

            selected_path = Path(selected["path"])
            self.assertEqual(selected_path.name, "publication-data.tsv")
            self.assertEqual(selected_path.read_bytes(), contents)


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

    def test_voxel_dimensions_are_validated_and_written_to_the_run_environment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            images = Path(temporary_directory) / "images"
            results = Path(temporary_directory) / "results"
            images.mkdir()
            config = validate_config(
                {
                    "workflow": "features_labelled",
                    "training_images": str(images),
                    "results_dir": str(results),
                    "voxel_size_x": "0.2",
                    "voxel_size_y": 0.3,
                    "voxel_size_z": 0.7,
                }
            )

            env_path = Path(create_env(config))
            try:
                env_text = env_path.read_text(encoding="utf-8")
            finally:
                os.unlink(env_path)

            self.assertEqual((config["voxel_size_x"], config["voxel_size_y"], config["voxel_size_z"]), (0.2, 0.3, 0.7))
            self.assertIn('IMAGINE_VOXEL_SIZE_X="0.2"', env_text)
            self.assertIn('IMAGINE_VOXEL_SIZE_Y="0.3"', env_text)
            self.assertIn('IMAGINE_VOXEL_SIZE_Z="0.7"', env_text)

    def test_voxel_dimensions_must_be_positive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            images = Path(temporary_directory)

            with self.assertRaisesRegex(ValueError, "Voxel size Z must be a positive number"):
                validate_config(
                    {
                        "workflow": "features_labelled",
                        "training_images": str(images),
                        "results_dir": str(images / "results"),
                        "voxel_size_z": 0,
                    }
                )


class TestRunReset(unittest.TestCase):
    def test_docker_step_uses_a_unique_run_container_name(self):
        command = command_for("/tmp/run.env", ["1", "datafile.tsv", "10", "generate_features"], {}, "microics-run-id")

        self.assertEqual(
            command,
            [
                "docker",
                "compose",
                "--env-file",
                "/tmp/run.env",
                "run",
                "--rm",
                "--no-TTY",
                "--name",
                "microics-run-id",
                "imagine",
                "1",
                "datafile.tsv",
                "10",
                "generate_features",
            ],
        )

    def test_reset_terminates_active_process_and_returns_to_ready_state(self):
        process = object()
        original_state = dict(state)
        original_runtime = dict(runtime)
        try:
            state.update(
                {
                    "status": "running",
                    "steps": [{"id": "features", "status": "running"}],
                    "logs": ["working"],
                    "artifacts": [{"path": "partial.tsv"}],
                    "artifact_roots": {"results": "/tmp/results"},
                    "validation": {"ok": True},
                }
            )
            runtime.update({"process": process, "active_run_id": "old-run", "stop_requested": False})

            runtime["container_name"] = "microics-old-run"
            with (
                patch("gui.app.terminate_process") as terminate,
                patch("gui.app.remove_run_container") as remove_container,
            ):
                self.assertTrue(reset_pipeline())

            terminate.assert_called_once_with(process, force=True)
            remove_container.assert_called_once_with("microics-old-run")
            self.assertEqual(state["status"], "idle")
            self.assertEqual(state["steps"], [])
            self.assertEqual(state["logs"], [])
            self.assertEqual(state["artifacts"], [])
            self.assertIsNone(state["validation"])
            self.assertEqual(state["progress"]["label"], "Ready")
            self.assertIsNone(runtime["active_run_id"])
        finally:
            state.clear()
            state.update(original_state)
            runtime.clear()
            runtime.update(original_runtime)


class TestWorkflowInterface(unittest.TestCase):
    def test_windows_has_one_visible_self_preparing_launcher(self):
        repository = Path(__file__).parent
        launchers = sorted(path.name for path in repository.glob("*.bat"))
        launcher = (repository / "MicroICS.bat").read_text(encoding="utf-8")

        self.assertEqual(launchers, ["MicroICS.bat"])
        self.assertIn("call :create_shortcut", launcher)
        self.assertIn("winget install --exact --id Python.Python.3.12", launcher)
        self.assertIn("winget install --exact --id Docker.DockerDesktop", launcher)
        self.assertIn("gui\\requirements.txt", launcher)

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
        self.assertIn('id="featureFilePicker"', html)
        self.assertIn("Choose file", html)
        self.assertIn('data-learner-mode="single"', html)
        self.assertIn('data-learner-mode="all"', html)
        self.assertIn("Benchmark every algorithm", html)
        self.assertIn('id="voxelSizeX"', html)
        self.assertIn('id="voxelSizeY"', html)
        self.assertIn('id="voxelSizeZ"', html)
        self.assertIn('id="resetButton"', html)
        self.assertIn('<img src="/logo.png" alt="MicroICS logo">', html)
        self.assertIn("filter:none", (Path(__file__).parent / "gui" / "styles.css").read_text(encoding="utf-8"))
        self.assertIn("Filenames found", javascript)
        self.assertIn("<strong>Images per label</strong>", javascript)
        self.assertIn('<details class="validation-secondary">', javascript)
        self.assertIn("<summary>Images per label per date</summary>", javascript)
        self.assertEqual(default_config()["workflow"], "features_labelled")
        self.assertEqual(
            (
                default_config()["voxel_size_x"],
                default_config()["voxel_size_y"],
                default_config()["voxel_size_z"],
            ),
            (0.13, 0.13, 0.5),
        )


if __name__ == "__main__":
    unittest.main()
