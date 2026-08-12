"""Tests for the standalone, container-local MicroICS GUI."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gui.app as app
import gui.execution as execution
from gui.app import (
    Handler,
    create_results_zip,
    preflight_error,
    receive_upload,
    replication_options,
    reset_pipeline,
    runtime,
    start_pipeline,
    state,
    validate_config,
)
from gui.execution import ExecutionStep, build_execution_steps, create_job, job_paths


@contextmanager
def temporary_jobs_root():
    with tempfile.TemporaryDirectory() as temporary_directory:
        jobs_root = Path(temporary_directory) / "jobs"
        with patch.object(execution, "JOBS_ROOT", jobs_root), patch.object(app, "JOBS_ROOT", jobs_root):
            yield jobs_root


def upload_handler(name: str, contents: bytes, job_id: str = ""):
    headers = {"X-Upload-Name": name, "Content-Length": str(len(contents))}
    if job_id:
        headers["X-Job-ID"] = job_id
    return SimpleNamespace(headers=headers, rfile=io.BytesIO(contents))


def labelled_image_name() -> str:
    return "01012026_s_Lm_st_L1_p_C01_pos001_tm_24_ch_Syto9_z_21.tif"


def training_table() -> bytes:
    return b"sampleName\tlabel\tfeature\n01012026--s--Lm--st--L1--p--C01--pos001--tm--24--ch--Syto9--z--21\tL1\t1\n"


def replication_table() -> bytes:
    rows = ["sampleName\tlabel\tfeature"]
    for label, prefix in (("L1", "C"), ("L2", "D")):
        for number in range(1, 6):
            name = f"01012026--s--Lm--st--{label}--p--{prefix}{number:02d}--pos001--tm--24--ch--Syto9--z--21"
            rows.append(f"{name}\t{label}\t{number}")
    return ("\n".join(rows) + "\n").encode()


class TestJobUploads(unittest.TestCase):
    def test_jobs_have_isolated_input_work_and_output_directories(self):
        with temporary_jobs_root():
            first = create_job()
            second = create_job()

            self.assertNotEqual(first.job_id, second.job_id)
            self.assertTrue(first.training_images.is_dir())
            self.assertTrue(first.work.is_dir())
            self.assertTrue(first.output.is_dir())
            self.assertNotEqual(first.root, second.root)

    def test_upload_strips_path_components_and_stays_in_job(self):
        with temporary_jobs_root():
            selected = receive_upload(upload_handler("../../../etc/escape.tif", b"TIFF"), "training")
            paths = job_paths(selected["job_id"])
            destination = paths.training_images / selected["file"]

            self.assertEqual(selected["file"], "escape.tif")
            self.assertEqual(destination.read_bytes(), b"TIFF")
            self.assertEqual(destination.resolve().parent, paths.training_images.resolve())

    def test_tiff_and_feature_table_uploads_are_supported(self):
        with temporary_jobs_root():
            image = receive_upload(upload_handler("volume.tiff", b"TIFF"), "inference")
            feature = receive_upload(
                upload_handler("features.tsv", training_table(), image["job_id"]),
                "feature",
                {".tsv", ".txt"},
                "tab-separated files",
            )

            self.assertEqual(image["job_id"], feature["job_id"])
            self.assertEqual(feature["file"], "features.tsv")


class TestDirectExecution(unittest.TestCase):
    def test_feature_generation_plan_calls_pipeline_directly(self):
        with temporary_jobs_root():
            paths = create_job()
            (paths.training_images / labelled_image_name()).touch()
            config = validate_config({"workflow": "features_labelled", "job_id": paths.job_id})

            steps = build_execution_steps(config)
            commands = [command for step in steps for command in step.commands]

            self.assertEqual([step.step_id for step in steps], ["features", "validate"])
            self.assertTrue(any(command[0] == "bash" and command[1].endswith("run_analysis.sh") for command in commands))
            self.assertFalse(any(token in {"docker", "compose", "docker-compose"} for command in commands for token in command))

    def test_training_plan_calls_existing_learning_entry_point(self):
        with temporary_jobs_root():
            paths = create_job()
            (paths.feature_files / "training.tsv").write_bytes(training_table())
            config = validate_config({"workflow": "train", "job_id": paths.job_id, "feature_file": "training.tsv"})

            steps = build_execution_steps(config)

            self.assertEqual([step.step_id for step in steps], ["models", "reports"])
            self.assertIn("feature_ranking_lite.py", " ".join(steps[0].commands[0]))
            self.assertIn("--save_models", steps[0].commands[0])

    def test_full_plan_rechecks_generated_features_with_selected_replication_unit(self):
        with temporary_jobs_root():
            paths = create_job()
            (paths.training_images / labelled_image_name()).touch()
            (paths.inference_images / "unknown.tif").touch()
            config = validate_config(
                {
                    "workflow": "full",
                    "job_id": paths.job_id,
                    "replication_unit": "well",
                    "learner": "rf",
                }
            )

            steps = build_execution_steps(config)
            validation_command = next(step.commands[0] for step in steps if step.step_id == "validate")

            self.assertIn("--replication-unit", validation_command)
            self.assertEqual(validation_command[validation_command.index("--replication-unit") + 1], "well")
            self.assertIn("--nested-cv", validation_command)

    def test_replication_options_block_date_before_training(self):
        with temporary_jobs_root():
            paths = create_job()
            (paths.feature_files / "training.tsv").write_bytes(replication_table())

            report = replication_options(
                {
                    "job_id": paths.job_id,
                    "feature_file": "training.tsv",
                    "replication_unit": "date",
                    "learner": "rf",
                    "all_learners": False,
                }
            )

            self.assertFalse(report["ok"])
            self.assertFalse(report["units"]["date"]["ok"])
            self.assertTrue(report["units"]["well"]["ok"])
            self.assertIn("Date cannot be used", report["errors"][0])

    def test_plate_is_not_accepted_as_a_replication_unit(self):
        with temporary_jobs_root():
            paths = create_job()
            (paths.feature_files / "training.tsv").write_bytes(replication_table())

            with self.assertRaisesRegex(ValueError, "available unit"):
                validate_config(
                    {
                        "workflow": "train",
                        "job_id": paths.job_id,
                        "feature_file": "training.tsv",
                        "replication_unit": "plate",
                    }
                )

    def test_inference_plan_accepts_precomputed_features_and_prior_models(self):
        with temporary_jobs_root():
            model_paths = create_job()
            models = model_paths.results / "models"
            models.mkdir(parents=True, exist_ok=True)
            (models / "rf_model.joblib").touch()
            paths = create_job()
            (paths.feature_files / "unknown_features.tsv").write_bytes(training_table().replace(b"\tlabel", b"\tprediction_label"))
            config = validate_config(
                {
                    "workflow": "inference",
                    "job_id": paths.job_id,
                    "feature_file": "unknown_features.tsv",
                    "model_job_id": model_paths.job_id,
                }
            )

            steps = build_execution_steps(config)
            command = steps[0].commands[0]

            self.assertEqual([step.step_id for step in steps], ["inference"])
            self.assertIn("inference.py", " ".join(command))
            self.assertIn("--features_file", command)
            self.assertEqual(config["models_dir"], str(models))


class TestDetailedTrainingProgress(unittest.TestCase):
    def test_structured_training_event_advances_within_model_stage(self):
        event = {
            "phase": "Benchmark all features",
            "phase_index": 2,
            "phase_total": 5,
            "completed": 5,
            "total": 20,
            "detail": "Benchmark all features: evaluation 6 of 20 — rf, fold 1/3",
            "sub_total": 10,
        }
        original_state = dict(state)
        original_runtime = dict(runtime)
        try:
            state.update(
                steps=[
                    {"id": "models", "label": "Train and benchmark models", "status": "running", "detail": "Starting"},
                    {"id": "reports", "label": "Create benchmark reports", "status": "pending", "detail": "Waiting"},
                ]
            )
            runtime["progress_context"] = {"model_step_fraction": 0.0}

            app.update_progress("models", f"2026-08-11 12:00:00 {app.PROGRESS_PREFIX}{json.dumps(event)}")

            self.assertEqual(state["steps"][0]["detail"], event["detail"])
            self.assertEqual(state["progress"]["percent"], 12.5)
        finally:
            state.clear()
            state.update(original_state)
            runtime.clear()
            runtime.update(original_runtime)

    def test_verbose_search_fit_updates_detail_and_fraction(self):
        event = {
            "phase": "Benchmark all features",
            "phase_index": 2,
            "phase_total": 5,
            "completed": 5,
            "total": 20,
            "detail": "Benchmark all features: evaluation 6 of 20 — rf, fold 1/3",
            "sub_total": 10,
        }
        original_state = dict(state)
        original_runtime = dict(runtime)
        try:
            state.update(steps=[{"id": "models", "label": "Train models", "status": "running", "detail": "Starting"}])
            runtime["progress_context"] = {"model_step_fraction": 0.0}
            app.update_progress("models", f"{app.PROGRESS_PREFIX}{json.dumps(event)}")
            initial_percent = state["progress"]["percent"]

            app.update_progress("models", "[CV] END max_depth=10, n_estimators=100; total time=1.2s")

            self.assertGreater(state["progress"]["percent"], initial_percent)
            self.assertIn("search fit 1 of 10 complete", state["steps"][0]["detail"])
        finally:
            state.clear()
            state.update(original_state)
            runtime.clear()
            runtime.update(original_runtime)


class TestPreflightFailureDetails(unittest.TestCase):
    def test_preflight_error_lists_exact_section_reasons(self):
        report = {
            "features": {
                "ok": False,
                "errors": ["Required label column is missing", "Could not parse feature columns: texture"],
                "invalid_cells_by_feature": {"texture": 3},
            },
            "ok": False,
        }

        message = preflight_error(report)

        self.assertEqual(
            message,
            "Preflight validation failed:\n"
            "- Features: Required label column is missing\n"
            "- Features: Could not parse feature columns: texture\n"
            "- Features: Feature 'texture' contains 3 non-numeric values",
        )

    def test_preflight_error_lists_invalid_image_filenames(self):
        report = {
            "images": {
                "ok": False,
                "invalid_images": 2,
                "invalid_filenames": ["bad-one.tif", "bad-two.tif"],
                "message": "Validated 0 of 2 image filenames",
            },
            "ok": False,
        }

        message = preflight_error(report)

        self.assertIn("- Images: Invalid image filename: bad-one.tif", message)
        self.assertIn("- Images: Invalid image filename: bad-two.tif", message)

    def test_preflight_error_lists_exact_replication_failure(self):
        report = {
            "replication": {
                "ok": False,
                "errors": [
                    "Date cannot be used for grouped training: "
                    "Cannot create grouped cross-validation splitter: at least 2 groups are required, got 1"
                ],
            },
            "ok": False,
        }

        message = preflight_error(report)

        self.assertIn("- Replication: Date cannot be used for grouped training", message)
        self.assertIn("at least 2 groups are required, got 1", message)


class TestResultsAndRecovery(unittest.TestCase):
    def test_training_start_does_not_wait_for_feature_preflight(self):
        with temporary_jobs_root():
            paths = create_job()
            (paths.feature_files / "training.tsv").write_bytes(training_table())
            release_pipeline = threading.Event()
            original_state = dict(state)
            original_runtime = dict(runtime)

            def hold_background_job(_config, _job_id):
                release_pipeline.wait(timeout=2)

            try:
                state["status"] = "idle"
                with patch("gui.app.preflight_config") as preflight, patch("gui.app.run_pipeline", side_effect=hold_background_job):
                    started, error = start_pipeline({"workflow": "train", "job_id": paths.job_id, "feature_file": "training.tsv"})

                self.assertTrue(started)
                self.assertIsNone(error)
                self.assertEqual(state["status"], "starting")
                self.assertEqual(state["progress"]["label"], "Validating inputs")
                self.assertTrue(state["progress"]["indeterminate"])
                preflight.assert_not_called()
            finally:
                release_pipeline.set()
                thread = runtime.get("thread")
                if thread:
                    thread.join(timeout=2)
                state.clear()
                state.update(original_state)
                runtime.clear()
                runtime.update(original_runtime)

    def test_job_output_is_downloadable_as_zip(self):
        with temporary_jobs_root():
            paths = create_job()
            result = paths.results / "datafile.tsv"
            result.write_text("sampleName\tfeature\nA\t1\n", encoding="utf-8")

            archive = create_results_zip(paths.job_id)

            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(bundle.namelist(), ["results/datafile.tsv"])

    def test_failed_subprocess_marks_job_failed_without_raising(self):
        with temporary_jobs_root():
            paths = create_job()
            config = {
                "job_id": paths.job_id,
                "workflow": "features_labelled",
                "training_images": str(paths.training_images),
                "inference_images": str(paths.inference_images),
                "results_dir": str(paths.results),
                "inference_output": str(paths.inference_output),
                "inference_work_dir": str(paths.work / "inference"),
                "feature_file": "",
                "models_dir": str(paths.results / "models"),
                "workers": 1,
                "top_features": 1,
                "replication_unit": "date",
                "learner": "rf",
                "all_learners": False,
                "correlation_threshold": 0.8,
                "voxel_size_x": 0.13,
                "voxel_size_y": 0.13,
                "voxel_size_z": 0.5,
            }
            failure = ExecutionStep("features", "Expected failure", ((sys.executable, "-c", "raise SystemExit(7)"),))
            original_state = dict(state)
            original_runtime = dict(runtime)
            try:
                state.update(status="starting", job_id=paths.job_id, steps=[], logs=[], artifacts=[], error=None)
                runtime.update(active_job_id=paths.job_id, stop_requested=False)
                with (
                    patch("gui.app.build_execution_steps", return_value=[failure]),
                    patch("gui.app.preflight_config", return_value={"ok": True}),
                ):
                    app.run_pipeline(config, paths.job_id)

                self.assertEqual(state["status"], "failed")
                self.assertIn("exit code 7", state["error"])
            finally:
                state.clear()
                state.update(original_state)
                runtime.clear()
                runtime.update(original_runtime)

    def test_reset_terminates_only_the_active_subprocess(self):
        process = object()
        original_state = dict(state)
        original_runtime = dict(runtime)
        try:
            state.update(status="running", job_id=None, steps=[{"id": "features"}], logs=["working"])
            runtime.update(process=process, active_job_id=None, stop_requested=False)
            with patch("gui.app.terminate_process") as terminate:
                self.assertTrue(reset_pipeline())
            terminate.assert_called_once_with(process, force=True)
            self.assertEqual(state["status"], "idle")
        finally:
            state.clear()
            state.update(original_state)
            runtime.clear()
            runtime.update(original_runtime)


class TestHTTPAndImage(unittest.TestCase):
    def test_health_endpoint_reports_ok(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/health") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.load(response), {"status": "ok"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_dockerfile_launches_one_self_contained_gui(self):
        repository = Path(__file__).parent
        dockerfile = (repository / "Dockerfile").read_text(encoding="utf-8")
        compose = (repository / "docker-compose.yml").read_text(encoding="utf-8")
        backend = (repository / "gui" / "app.py").read_text(encoding="utf-8")

        self.assertIn("COPY . /opt/microics", dockerfile)
        self.assertIn("EXPOSE 8765", dockerfile)
        self.assertIn('"--host", "0.0.0.0"', dockerfile)
        self.assertIn("MICROICS_DATA_ROOT=/data", dockerfile)
        self.assertNotIn("docker compose", backend.lower())
        self.assertNotIn('subprocess.run(["docker"', backend)
        self.assertEqual(compose.count("services:"), 1)
        self.assertNotIn("/var/run/docker.sock", compose)

    def test_browser_interface_uses_uploads_jobs_and_zip_downloads(self):
        repository = Path(__file__).parent
        html = (repository / "gui" / "index.html").read_text(encoding="utf-8")
        javascript = (repository / "gui" / "app.js").read_text(encoding="utf-8")

        self.assertIn('accept=".tif,.tiff,image/tiff"', html)
        self.assertIn('id="modelJob"', html)
        self.assertIn('id="downloadButton"', html)
        self.assertIn('id="resultDelivery"', html)
        self.assertIn("browser's configured download location", html)
        self.assertNotIn("Results folder", html)
        self.assertNotIn("Choose folder", html)
        self.assertIn("X-Job-ID", javascript)
        self.assertIn("download_url", javascript)
        self.assertNotIn("await post('/api/preflight'", javascript)
        self.assertIn("setFormMessage(nextState.error", javascript)
        self.assertIn("'is-error'", javascript)
        self.assertIn("progressBar.classList.toggle('is-indeterminate'", javascript)
        self.assertIn("Preflight check in progress", javascript)
        self.assertLess(javascript.index("status: 'starting'"), javascript.index("renderState(await post('/api/run'"))
        self.assertIn('id="copyLogButton"', html)
        self.assertIn('aria-label="Copy job log"', html)
        self.assertIn("navigator.clipboard?.writeText", javascript)
        self.assertIn("fallbackCopyText", javascript)
        self.assertIn("event.stopPropagation()", javascript)
        self.assertNotIn('value="plate"', html)
        self.assertIn("/api/replication-options", javascript)
        self.assertIn("Checking grouped cross-validation feasibility", javascript)


if __name__ == "__main__":
    unittest.main()
