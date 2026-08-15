#!/usr/bin/env python3
"""Web UI for the self-contained MicroICS runtime."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    from gui.execution import (
        JOBS_ROOT,
        SRC_DIR,
        ExecutionStep,
        build_execution_steps,
        create_job,
        execution_environment,
        job_paths,
        safe_uploaded_name,
        stage_training_features,
        unique_upload_path,
        validate_job_id,
    )
    from src.input_validation import validate_feature_table, validate_image_directory
except ModuleNotFoundError:  # Direct ``python gui/app.py`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gui.execution import (
        JOBS_ROOT,
        SRC_DIR,
        ExecutionStep,
        build_execution_steps,
        create_job,
        execution_environment,
        job_paths,
        safe_uploaded_name,
        stage_training_features,
        unique_upload_path,
        validate_job_id,
    )
    from src.input_validation import validate_feature_table, validate_image_directory

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = Path(__file__).resolve().parent
VERSION = "0.8.0"
MAX_LOG_LINES = 500
MAX_ARTIFACTS = 450
PROGRESS_PREFIX = "MICROICS_PROGRESS "
DEFAULT_VOXEL_DIMENSIONS = {"voxel_size_x": 0.13, "voxel_size_y": 0.13, "voxel_size_z": 0.5}

state_lock = threading.RLock()
runtime = {
    "process": None,
    "stop_requested": False,
    "progress_context": None,
    "active_job_id": None,
    "thread": None,
}
state = {
    "status": "idle",
    "job_id": None,
    "current_step": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "validation": None,
    "config": None,
    "steps": [],
    "progress": {"completed": 0, "total": 0, "percent": 0, "label": "Ready"},
    "logs": [],
    "artifacts": [],
    "download_url": None,
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def json_response(handler: BaseHTTPRequestHandler, payload: object, status=HTTPStatus.OK):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
        payload = json.loads(handler.rfile.read(length) or b"{}")
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be an object")
    return payload


def read_memory() -> tuple[int, int] | None:
    try:
        values = {}
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, value, *_ = line.split()
                if key in {"MemTotal:", "MemAvailable:"}:
                    values[key] = int(value) * 1024
        return values["MemTotal:"] - values["MemAvailable:"], values["MemTotal:"]
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def hardware_defaults() -> dict[str, object]:
    """Suggest conservative worker usage inside the current container."""
    cpu_count = os.cpu_count() or 1
    memory = read_memory()
    return {
        "workers": max(1, min(64, math.floor(cpu_count * 0.7))),
        "cpu_count": cpu_count,
        "memory_available_mib": math.floor((memory[1] - memory[0]) / (1024 * 1024)) if memory else None,
    }


def default_config() -> dict[str, object]:
    return {
        "workflow": "features_labelled",
        "job_id": "",
        "feature_file": "",
        "model_job_id": "",
        "workers": min(4, os.cpu_count() or 1),
        "top_features": 10,
        "correlation_threshold": 0.8,
        "all_learners": False,
        "learner": "rf",
        **DEFAULT_VOXEL_DIMENSIONS,
    }


def metadata_path(job_id: str) -> Path:
    return job_paths(job_id).root / "job.json"


def read_job_metadata(job_id: str) -> dict:
    try:
        payload = json.loads(metadata_path(job_id).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return {}


def write_job_metadata(job_id: str, **updates):
    paths = job_paths(job_id, create=True)
    metadata = read_job_metadata(job_id)
    metadata.update(updates)
    metadata.setdefault("job_id", job_id)
    metadata.setdefault("created_at", timestamp())
    metadata["updated_at"] = timestamp()
    temporary = paths.work / "job.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(paths.root / "job.json")


def list_jobs() -> list[dict]:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for root in sorted(JOBS_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not root.is_dir():
            continue
        try:
            paths = job_paths(root.name)
        except (ValueError, FileNotFoundError):
            continue
        metadata = read_job_metadata(root.name)
        model_files = sorted((paths.results / "models").glob("*_model.joblib"))
        has_models = bool(model_files)
        has_output = any(path.is_file() for path in paths.output.rglob("*"))
        jobs.append(
            {
                "job_id": root.name,
                "status": metadata.get("status", "draft"),
                "workflow": metadata.get("workflow", "draft"),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "has_models": has_models,
                "has_output": has_output,
                "download_url": f"/api/jobs/{root.name}/download" if has_output else None,
                "learner": metadata.get("learner"),
                "all_learners": bool(metadata.get("all_learners")),
                "model_count": len(model_files),
                "model_names": [path.name.replace("_model.joblib", "") for path in model_files],
            }
        )
    return jobs


def model_jobs() -> list[dict]:
    return [job for job in list_jobs() if job["has_models"]]


def upload_directory(job_id: str, kind: str) -> Path:
    paths = job_paths(job_id, create=True)
    directories = {
        "training": paths.training_images,
        "inference": paths.inference_images,
        "feature": paths.feature_files,
    }
    try:
        return directories[kind]
    except KeyError as exc:
        raise ValueError("Unsupported upload kind") from exc


def receive_upload(
    handler: BaseHTTPRequestHandler,
    kind: str = "training",
    allowed_suffixes: set[str] | None = None,
    file_description: str = ".tif or .tiff image files",
) -> dict[str, str]:
    allowed_suffixes = allowed_suffixes or {".tif", ".tiff"}
    raw_job_id = handler.headers.get("X-Job-ID", "")
    paths = job_paths(validate_job_id(raw_job_id), create=True) if raw_job_id else create_job()
    try:
        safe_name = safe_uploaded_name(handler.headers.get("X-Upload-Name", ""), allowed_suffixes)
    except ValueError as exc:
        raise ValueError(f"Only {file_description} can be selected") from exc
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Uploaded file size could not be determined") from exc
    if content_length <= 0:
        raise ValueError("The selected file is empty")

    destination = unique_upload_path(upload_directory(paths.job_id, kind), safe_name)
    remaining = content_length
    try:
        with destination.open("wb") as output:
            while remaining:
                chunk = handler.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("The upload ended before the complete file was received")
                output.write(chunk)
                remaining -= len(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    write_job_metadata(paths.job_id, status="draft")
    return {"job_id": paths.job_id, "file": destination.name, "kind": kind}


def uploaded_feature_path(paths, filename: object) -> Path | None:
    if not filename:
        return None
    safe_name = safe_uploaded_name(str(filename), {".tsv", ".txt"})
    candidate = (paths.feature_files / safe_name).resolve()
    if candidate.parent != paths.feature_files.resolve() or not candidate.is_file():
        raise ValueError("The selected feature table is not available in this job")
    return candidate


def has_images(directory: Path) -> bool:
    return any(path.is_file() and path.suffix.lower() in {".tif", ".tiff"} for path in directory.iterdir())


def validate_config(raw: dict) -> dict:
    config = default_config()
    config.update(raw)
    workflow = config.get("workflow")
    if workflow not in {"full", "train", "inference", "features_labelled", "features_unlabelled"}:
        raise ValueError("Choose a supported workflow")
    job_id = validate_job_id(str(config.get("job_id", "")))
    paths = job_paths(job_id)

    try:
        workers = int(config.get("workers", 4))
        top_features = int(config.get("top_features", 10))
        correlation_threshold = float(config.get("correlation_threshold", 0.8))
    except (TypeError, ValueError) as exc:
        raise ValueError("Workers, top features, and correlation threshold must be numeric") from exc
    if not 1 <= workers <= 64:
        raise ValueError("Workers must be between 1 and 64")
    if not 1 <= top_features <= 500:
        raise ValueError("Top features must be between 1 and 500")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("Correlation threshold must be between 0 and 1")

    voxel_dimensions = {}
    for axis in ("x", "y", "z"):
        key = f"voxel_size_{axis}"
        try:
            dimension = float(config.get(key, DEFAULT_VOXEL_DIMENSIONS[key]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Voxel size {axis.upper()} must be a positive number") from exc
        if not math.isfinite(dimension) or dimension <= 0:
            raise ValueError(f"Voxel size {axis.upper()} must be a positive number")
        voxel_dimensions[key] = dimension

    learner = str(config.get("learner", "rf"))
    if learner not in {"rf", "dummy", "decisiontree", "logistic", "xgb", "gridsearch"}:
        raise ValueError("Choose a supported learner")

    feature_file = uploaded_feature_path(paths, config.get("feature_file"))
    if workflow in {"features_labelled", "full"} and not feature_file and not has_images(paths.training_images):
        raise ValueError("Upload labelled training images or a complete training feature table")
    if workflow == "features_unlabelled" and not has_images(paths.inference_images):
        raise ValueError("Upload at least one inference image")
    if workflow == "train" and not feature_file:
        raise ValueError("Train models requires an uploaded complete feature table")
    if workflow == "full" and not has_images(paths.inference_images):
        raise ValueError("All together requires inference images to classify")
    if workflow == "inference" and not feature_file and not has_images(paths.inference_images):
        raise ValueError("Inference requires uploaded images or a compatible feature table")

    model_job_id = str(config.get("model_job_id", ""))
    if workflow == "full":
        models_dir = paths.results / "models"
    elif workflow == "inference":
        validate_job_id(model_job_id)
        model_paths = job_paths(model_job_id)
        models_dir = model_paths.results / "models"
        if not any(models_dir.glob("*_model.joblib")):
            raise ValueError("Select a completed training job containing models")
    else:
        models_dir = paths.results / "models"

    return {
        "workflow": workflow,
        "job_id": job_id,
        "training_images": str(paths.training_images),
        "inference_images": str(paths.inference_images),
        "feature_file": str(feature_file) if feature_file else "",
        "feature_file_name": feature_file.name if feature_file else "",
        "model_job_id": model_job_id,
        "models_dir": str(models_dir),
        "results_dir": str(paths.results),
        "inference_output": str(paths.inference_output),
        "inference_work_dir": str(paths.work / "inference"),
        "workers": workers,
        "top_features": top_features,
        "correlation_threshold": correlation_threshold,
        "all_learners": bool(config.get("all_learners")),
        "learner": learner,
        **voxel_dimensions,
    }


def preflight_config(config: dict) -> dict:
    workflow = config["workflow"]
    report: dict[str, object] = {}
    if workflow in {"features_labelled", "full"} and not config.get("feature_file"):
        report["images"] = validate_image_directory(config["training_images"], labelled=True)
    if workflow == "features_unlabelled":
        report["images"] = validate_image_directory(config["inference_images"], labelled=False)
    if workflow in {"train", "inference", "full"} and config.get("feature_file"):
        report["features"] = validate_feature_table(config["feature_file"], require_label=workflow != "inference")
    if workflow in {"inference", "full"} and not (workflow == "inference" and config.get("feature_file")):
        report["inference_images"] = validate_image_directory(config["inference_images"], labelled=False)
    report["ok"] = bool(report) and all(section.get("ok", False) for section in report.values() if isinstance(section, dict))
    return report


def preflight_error(report: dict) -> str:
    reasons = []
    for name, section in report.items():
        if not isinstance(section, dict) or section.get("ok", True):
            continue
        section_name = name.replace("_", " ").capitalize()
        section_reasons = [str(error) for error in section.get("errors", [])]
        invalid_cells = section.get("invalid_cells_by_feature", {})
        section_reasons.extend(
            f"Feature '{feature}' contains {count} non-numeric value{'s' if count != 1 else ''}" for feature, count in invalid_cells.items()
        )
        invalid_filenames = section.get("invalid_filenames", [])
        section_reasons.extend(f"Invalid image filename: {filename}" for filename in invalid_filenames)
        unlisted_filenames = max(0, int(section.get("invalid_images", 0)) - len(invalid_filenames))
        if unlisted_filenames:
            section_reasons.append(f"{unlisted_filenames} additional invalid image filename(s) were omitted")
        if not section_reasons and section.get("message"):
            section_reasons.append(str(section["message"]))
        reasons.extend(f"- {section_name}: {reason}" for reason in section_reasons)
    return "Preflight validation failed:\n" + "\n".join(reasons or ["- Check the uploaded inputs"])


def run_is_active(job_id: str) -> bool:
    with state_lock:
        return runtime["active_job_id"] == job_id


def add_log(message: str, job_id: str | None = None):
    entry = f"[{now()}] {message}"
    with state_lock:
        if job_id is not None and runtime["active_job_id"] != job_id:
            return
        state["logs"].append(entry)
        state["logs"] = state["logs"][-MAX_LOG_LINES:]
    if job_id:
        try:
            with (job_paths(job_id).work / "pipeline.log").open("a", encoding="utf-8") as log_file:
                log_file.write(entry + "\n")
        except OSError:
            pass


def progress_context(config: dict, steps: list[ExecutionStep]) -> dict:
    return {
        "step_count": len(steps),
        "feature_total": sum(1 for path in Path(config["training_images"]).glob("*.tif*")),
        "feature_writes": 0,
        "model_step_fraction": 0.0,
    }


def parse_pipeline_progress(line: str) -> dict | None:
    """Extract and validate a structured progress event from a subprocess log line."""
    marker = line.find(PROGRESS_PREFIX)
    if marker < 0:
        return None
    try:
        event = json.loads(line[marker + len(PROGRESS_PREFIX) :])
        numeric_fields = ("phase_index", "phase_total", "completed", "total")
        if not isinstance(event, dict) or not all(isinstance(event.get(field), int | float) for field in numeric_fields):
            return None
        if event["phase_total"] <= 0 or event["total"] <= 0 or not isinstance(event.get("detail"), str):
            return None
        return event
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def model_step_fraction(event: dict, sub_completed: int = 0) -> float:
    """Map work within a reported training phase to a fraction of the model step."""
    unit_fraction = event["completed"]
    sub_total = event.get("sub_total")
    if sub_total:
        unit_fraction += min(sub_completed / sub_total, 0.99)
    phase_fraction = min(1.0, unit_fraction / event["total"])
    return min(1.0, max(0.0, (event["phase_index"] - 1 + phase_fraction) / event["phase_total"]))


def update_progress(step_id: str, line: str | None = None, complete: bool = False):
    with state_lock:
        context = runtime["progress_context"] or {}
        current = next((item for item in state["steps"] if item["id"] == step_id), None)
        if line and current:
            detail = line[:180]
            if step_id == "features" and "writing " in line:
                context["feature_writes"] = context.get("feature_writes", 0) + 1
                detail = f"Generated feature outputs: {context['feature_writes']}"
            elif step_id == "models":
                event = parse_pipeline_progress(line)
                if event:
                    context["model_progress_event"] = event
                    context["model_sub_completed"] = 0
                    context["model_step_fraction"] = max(context.get("model_step_fraction", 0.0), model_step_fraction(event))
                    detail = event["detail"]
                elif "[CV" in line and "] END" in line and context.get("model_progress_event", {}).get("sub_total"):
                    event = context["model_progress_event"]
                    sub_completed = min(context.get("model_sub_completed", 0) + 1, event["sub_total"])
                    context["model_sub_completed"] = sub_completed
                    context["model_step_fraction"] = max(context.get("model_step_fraction", 0.0), model_step_fraction(event, sub_completed))
                    detail = f"{event['detail']} — search fit {sub_completed} of {event['sub_total']} complete"
                elif context.get("model_progress_event"):
                    detail = current.get("detail", detail)
            current["detail"] = detail
        completed = sum(item["status"] == "completed" for item in state["steps"])
        total = len(state["steps"])
        base = completed / total if total else 0
        if not complete and current and current["status"] == "running" and total:
            step_fraction = context.get("model_step_fraction", 0.0) if step_id == "models" else 0.5
            base += step_fraction / total
        state["progress"] = {
            "completed": completed,
            "total": total,
            "percent": min(99 if not complete else 100, round(base * 100, 1)),
            "label": f"Working: {current['label']}" if current and current["status"] == "running" else "Complete",
            "detail": current.get("detail", "Working") if current else "Working",
        }


def set_step(step_id: str, status: str, detail: str | None = None):
    with state_lock:
        for step in state["steps"]:
            if step["id"] != step_id:
                continue
            step["status"] = status
            if detail is not None:
                step["detail"] = detail
            if status == "running":
                step["started_at"] = now()
                state["current_step"] = step_id
            if status in {"completed", "failed", "cancelled"}:
                step["finished_at"] = now()
                state["current_step"] = None
            return


def terminate_process(process: subprocess.Popen, force: bool = False):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM) if os.name != "nt" else process.terminate()
    except ProcessLookupError:
        return
    if not force:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL) if os.name != "nt" else process.kill()
        except ProcessLookupError:
            pass


def execute_step(step: ExecutionStep, environment: dict[str, str], job_id: str) -> bool:
    if not run_is_active(job_id):
        return False
    set_step(step.step_id, "running", "Starting")
    update_progress(step.step_id, "Starting")
    for command in step.commands:
        add_log("$ " + " ".join(command), job_id)
        try:
            process = subprocess.Popen(
                command,
                cwd=SRC_DIR,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            set_step(step.step_id, "failed", str(exc))
            with state_lock:
                state["error"] = f"{step.label} could not start: {exc}"
            add_log(str(exc), job_id)
            return False
        with state_lock:
            if runtime["active_job_id"] == job_id:
                runtime["process"] = process
        assert process.stdout is not None
        for line in process.stdout:
            if not run_is_active(job_id):
                terminate_process(process, force=True)
                break
            line = line.rstrip()
            if line:
                progress_only = step.step_id == "models" and (PROGRESS_PREFIX in line or ("[CV" in line and "] END" in line))
                if not progress_only:
                    add_log(line, job_id)
                update_progress(step.step_id, line)
            with state_lock:
                if runtime["stop_requested"]:
                    terminate_process(process)
        process.stdout.close()
        exit_code = process.wait()
        with state_lock:
            if runtime["process"] is process:
                runtime["process"] = None
            stopped = runtime["stop_requested"] or runtime["active_job_id"] != job_id
        if stopped:
            if run_is_active(job_id):
                set_step(step.step_id, "cancelled", "Stopped")
            return False
        if exit_code:
            set_step(step.step_id, "failed", f"Exited with code {exit_code}")
            with state_lock:
                state["error"] = f"{step.label} failed with exit code {exit_code}"
            add_log(f"{step.label} failed with exit code {exit_code}", job_id)
            return False
    set_step(step.step_id, "completed", "Complete")
    update_progress(step.step_id, complete=True)
    add_log(f"{step.label} complete", job_id)
    return True


def artifact_kind(path: Path) -> str:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if path.suffix.lower() in {".html", ".htm"}:
        return "html"
    if path.suffix.lower() in {".tsv", ".csv", ".txt", ".log", ".json"}:
        return "text"
    return "file"


def refresh_artifacts(job_id: str | None):
    artifacts = []
    if job_id:
        output = job_paths(job_id).output
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(output).as_posix()
                size = path.stat().st_size
            except OSError:
                continue
            artifacts.append({"path": relative, "name": path.name, "size": size, "kind": artifact_kind(path)})
            if len(artifacts) >= MAX_ARTIFACTS:
                break
    with state_lock:
        state["artifacts"] = artifacts
        state["download_url"] = f"/api/jobs/{job_id}/download" if job_id and artifacts else None


def run_pipeline(config: dict, job_id: str):
    try:
        steps = build_execution_steps(config)
        if not steps:
            raise ValueError("The selected workflow has no executable steps")
        environment = execution_environment(config)
        validation_detail = "Validating the uploaded feature table" if config.get("feature_file") else "Validating uploaded images"
        with state_lock:
            if runtime["active_job_id"] != job_id:
                return
            state["status"] = "running"
            state["steps"] = [{"id": step.step_id, "label": step.label, "status": "pending", "detail": "Waiting"} for step in steps]
            state["progress"] = {
                "completed": 0,
                "total": len(steps),
                "percent": 0,
                "label": "Validating inputs",
                "detail": validation_detail,
                "indeterminate": True,
            }
            runtime["progress_context"] = progress_context(config, steps)
        write_job_metadata(job_id, status="running", workflow=config["workflow"])
        add_log("Pipeline started inside the MicroICS container", job_id)
        add_log(validation_detail, job_id)

        preflight = preflight_config(config)
        with state_lock:
            if runtime["active_job_id"] != job_id:
                return
            state["validation"] = preflight
        if not preflight["ok"]:
            raise ValueError(preflight_error(preflight))

        with state_lock:
            state["progress"].update(label="Preparing inputs", detail="Preparing the validated inputs for processing", indeterminate=True)
        stage_training_features(config)
        for step in steps:
            if not execute_step(step, environment, job_id):
                break
            refresh_artifacts(job_id)
        with state_lock:
            if runtime["active_job_id"] != job_id:
                return
            if runtime["stop_requested"]:
                final_status = "cancelled"
            elif any(step["status"] == "failed" for step in state["steps"]):
                final_status = "failed"
            else:
                final_status = "completed"
                state["progress"].update(percent=100, completed=len(state["steps"]), label="Complete")
            state["status"] = final_status
            state["finished_at"] = now()
            error = state["error"]
        write_job_metadata(job_id, status=final_status, workflow=config["workflow"], error=error)
        add_log("Pipeline complete" if final_status == "completed" else f"Pipeline {final_status}", job_id)
        refresh_artifacts(job_id)
    except Exception as exc:
        with state_lock:
            if runtime["active_job_id"] != job_id:
                return
            state["status"] = "failed"
            state["error"] = str(exc)
            state["finished_at"] = now()
        write_job_metadata(job_id, status="failed", workflow=config.get("workflow"), error=str(exc))
        add_log(f"Unexpected error: {exc}", job_id)
    finally:
        with state_lock:
            if runtime["active_job_id"] == job_id:
                runtime.update(process=None, progress_context=None, active_job_id=None, thread=None)


def start_pipeline(raw_config: dict) -> tuple[bool, str | None]:
    with state_lock:
        if state["status"] in {"starting", "running"}:
            return False, "A pipeline is already running"
    try:
        config = validate_config(raw_config)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)

    job_id = config["job_id"]
    with state_lock:
        runtime.update(stop_requested=False, active_job_id=job_id)
        state.update(
            {
                "status": "starting",
                "job_id": job_id,
                "current_step": None,
                "started_at": now(),
                "finished_at": None,
                "error": None,
                "validation": None,
                "config": config,
                "steps": [],
                "progress": {
                    "completed": 0,
                    "total": 0,
                    "percent": 0,
                    "label": "Validating inputs",
                    "detail": "Starting the preflight check",
                    "indeterminate": True,
                },
                "logs": [],
                "artifacts": [],
                "download_url": None,
            }
        )
    write_job_metadata(
        job_id,
        status="starting",
        workflow=config["workflow"],
        error=None,
        learner=config.get("learner"),
        all_learners=bool(config.get("all_learners")),
    )
    thread = threading.Thread(target=run_pipeline, args=(config, job_id), daemon=True)
    with state_lock:
        runtime["thread"] = thread
    thread.start()
    return True, None


def stop_pipeline() -> bool:
    with state_lock:
        if state["status"] not in {"starting", "running"}:
            return False
        runtime["stop_requested"] = True
        process = runtime["process"]
    if process:
        terminate_process(process, force=True)
    add_log("Stop requested", state.get("job_id"))
    return True


def reset_pipeline() -> bool:
    with state_lock:
        had_run = state["status"] != "idle" or bool(state["steps"] or state["logs"])
        process = runtime["process"]
        job_id = state.get("job_id")
        runtime.update(process=None, stop_requested=True, progress_context=None, active_job_id=None, thread=None)
        state.update(
            {
                "status": "idle",
                "job_id": None,
                "current_step": None,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "validation": None,
                "config": None,
                "steps": [],
                "progress": {"completed": 0, "total": 0, "percent": 0, "label": "Ready"},
                "logs": [],
                "artifacts": [],
                "download_url": None,
            }
        )
    if process:
        terminate_process(process, force=True)
    if job_id:
        metadata = read_job_metadata(job_id)
        if metadata.get("status") in {"starting", "running"}:
            write_job_metadata(job_id, status="cancelled")
    return had_run


def snapshot(refresh=False) -> dict:
    with state_lock:
        job_id = state["job_id"]
    if refresh and job_id:
        refresh_artifacts(job_id)
    with state_lock:
        payload = dict(state)
        payload["steps"] = [dict(step) for step in state["steps"]]
        payload["logs"] = list(state["logs"])
        payload["artifacts"] = [dict(artifact) for artifact in state["artifacts"]]
    payload["model_jobs"] = model_jobs()
    return payload


def resolve_artifact(job_id: str, relative: str) -> Path:
    output = job_paths(validate_job_id(job_id)).output.resolve()
    candidate = (output / unquote(relative)).resolve()
    if candidate.parent != output and output not in candidate.parents:
        raise PermissionError("Artifact path is outside the job output folder")
    if not candidate.is_file():
        raise FileNotFoundError("Artifact does not exist")
    return candidate


def create_results_zip(job_id: str) -> Path:
    paths = job_paths(validate_job_id(job_id))
    archive = paths.work / f"microics-results-{job_id}.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths.output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(paths.output))
    temporary.replace(archive)
    return archive


class Handler(BaseHTTPRequestHandler):
    server_version = "MicroICS-GUI/2.0"

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            json_response(self, {"status": "ok"})
            return
        if parsed.path == "/api/state":
            json_response(self, snapshot(parse_qs(parsed.query).get("refresh") == ["1"]))
            return
        if parsed.path == "/api/defaults":
            json_response(self, {"version": VERSION, "config": default_config(), "model_jobs": model_jobs()})
            return
        if parsed.path == "/api/hardware-defaults":
            json_response(self, hardware_defaults())
            return
        if parsed.path == "/api/jobs":
            json_response(self, {"jobs": list_jobs()})
            return
        if parsed.path in {"/api/artifact", "/api/preview"}:
            query = parse_qs(parsed.query)
            try:
                path = resolve_artifact(query.get("job", [""])[0], query.get("path", [""])[0])
            except (FileNotFoundError, PermissionError, ValueError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            if parsed.path == "/api/preview":
                try:
                    json_response(self, {"name": path.name, "content": path.read_text(errors="replace")[:24000]})
                except OSError as exc:
                    json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.serve_file(path)
            return
        job_download = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/download", parsed.path)
        if job_download:
            try:
                archive = create_results_zip(job_download.group(1))
            except (FileNotFoundError, OSError, ValueError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            self.serve_file(archive, download_name=archive.name)
            return
        if parsed.path in {"/", "/index.html"}:
            self.serve_file(GUI_DIR / "index.html")
            return
        if parsed.path == "/logo.png":
            self.serve_file(ROOT / "logo.png")
            return
        if parsed.path == "/logo_dark.png":
            self.serve_file(ROOT / "logo_dark.png")
            return
        static_path = (GUI_DIR / parsed.path.lstrip("/")).resolve()
        if static_path.parent == GUI_DIR or GUI_DIR in static_path.parents:
            if static_path.is_file():
                self.serve_file(static_path)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/preflight":
            try:
                config = validate_config(read_json(self))
                json_response(self, {"config": config, "report": preflight_config(config)})
            except (FileNotFoundError, ValueError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path in {"/api/upload", "/api/upload-feature"}:
            try:
                if parsed.path == "/api/upload-feature":
                    result = receive_upload(
                        self,
                        "feature",
                        {".tsv", ".txt"},
                        "tab-separated .tsv or .txt files",
                    )
                else:
                    kind = self.headers.get("X-Upload-Kind", "training")
                    result = receive_upload(self, kind)
                json_response(self, result, HTTPStatus.CREATED)
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except OSError as exc:
                json_response(self, {"error": f"Could not save the selected file: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/run":
            try:
                started, error = start_pipeline(read_json(self))
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if not started:
                status = HTTPStatus.CONFLICT if error and "already" in error else HTTPStatus.BAD_REQUEST
                json_response(self, {"error": error}, status)
                return
            json_response(self, snapshot(), HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/stop":
            stop_pipeline()
            json_response(self, snapshot())
            return
        if parsed.path == "/api/reset":
            reset_pipeline()
            json_response(self, snapshot())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def serve_file(self, path: Path, download_name: str | None = None):
        try:
            size = path.stat().st_size
            source = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with source:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            if download_name:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)


def main():
    parser = argparse.ArgumentParser(description="Run the self-contained MicroICS GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"MicroICS GUI running at {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MicroICS GUI", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
