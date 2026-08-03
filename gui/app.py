#!/usr/bin/env python3
"""Small local web UI for the Docker-backed MicroICS pipeline."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    from src.input_validation import validate_feature_table, validate_image_directory
except ModuleNotFoundError:  # Running gui/app.py directly puts gui/, not the repository root, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.input_validation import validate_feature_table, validate_image_directory

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = Path(__file__).resolve().parent
VERSION = "0.5.1"
MAX_LOG_LINES = 500
MAX_ARTIFACTS = 450
UPLOAD_ROOT = ROOT / ".gui_uploads"
GUI_LOG = ROOT / "microics-gui.log"

state_lock = threading.RLock()
runtime = {"process": None, "stop_requested": False, "progress_context": None}
state = {
    "status": "idle",
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
    "artifact_roots": {},
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def read_memory() -> tuple[int, int] | None:
    try:
        values = {}
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, value, *_ = line.split()
                if key in {"MemTotal:", "MemAvailable:"}:
                    values[key] = int(value) * 1024
        total = values["MemTotal:"]
        available = values["MemAvailable:"]
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None
    return total - available, total


def hardware_defaults() -> dict[str, object]:
    """Suggest conservative Docker limits based on currently available host resources."""
    cpu_count = os.cpu_count() or 1
    cpu_limit = round(cpu_count * 0.7, 2)
    memory = read_memory()
    if memory:
        used, total = memory
        available = max(total - used, 512 * 1024 * 1024)
        memory_limit_mib = max(512, math.floor(available * 0.7 / (1024 * 1024)))
    else:
        memory_limit_mib = None
    return {
        "workers": max(1, min(64, math.floor(cpu_limit))),
        "cpu_limit": cpu_limit,
        "memory_limit": f"{memory_limit_mib}m" if memory_limit_mib else None,
        "memory_limit_mib": memory_limit_mib,
        "cpu_count": cpu_count,
    }


def default_config() -> dict[str, object]:
    test_images = ROOT / "test_images"
    return {
        "workflow": "features_labelled",
        "feature_mode": "labelled",
        "training_images": str(test_images if test_images.is_dir() else ROOT),
        "results_dir": str(ROOT / "results"),
        "inference_images": str(test_images if test_images.is_dir() else ROOT),
        "inference_output": str(ROOT / "inference_results"),
        "workers": 4,
        "top_features": 10,
        "correlation_threshold": 0.8,
        "all_learners": False,
        "learner": "rf",
        "confirm_cleanup": False,
        "cpu_limit": None,
        "memory_limit": None,
        "replication_unit": "date",
        "feature_file": "",
    }


def demo_config() -> dict[str, str]:
    """Return absolute paths for the built-in installation check."""
    demo_images = ROOT / "examples" / "test_images"
    if not demo_images.is_dir():
        demo_images = ROOT / "test_images"
    demo_results = ROOT / ".gui_demo_results"
    return {
        "images": str(demo_images),
        "results": str(demo_results),
        "inference_output": str(demo_results / "inference"),
    }


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
        return json.loads(handler.rfile.read(length) or b"{}")
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Request body must be valid JSON") from exc


def receive_upload(
    handler: BaseHTTPRequestHandler,
    allowed_suffixes: set[str] | None = None,
    file_description: str = ".tif image files",
) -> dict[str, str]:
    allowed_suffixes = allowed_suffixes or {".tif"}
    filename = handler.headers.get("X-Upload-Name", "").replace("\\", "/")
    safe_name = Path(filename).name
    if not safe_name or Path(safe_name).suffix.lower() not in allowed_suffixes:
        raise ValueError(f"Only {file_description} can be selected")
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Uploaded file size could not be determined") from exc
    if content_length <= 0:
        raise ValueError("The selected file is empty")

    group = handler.headers.get("X-Upload-Group", "")
    if not re.fullmatch(r"[a-f0-9]{32}", group):
        group = uuid.uuid4().hex
    directory = UPLOAD_ROOT / group
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / safe_name
    if destination.exists():
        destination = directory / f"{destination.stem}-{uuid.uuid4().hex[:8]}{destination.suffix}"

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
    return {"directory": str(directory), "file": destination.name, "path": str(destination), "group": group}


def browse_directory(value: str | None) -> dict[str, object]:
    requested = Path(value).expanduser().resolve() if value else ROOT
    if requested.is_dir():
        current = requested
    elif requested.parent.is_dir():
        current = requested.parent
    else:
        raise ValueError("Folder does not exist and its parent cannot be opened")
    directories = []
    try:
        children = sorted(current.iterdir(), key=lambda path: path.name.lower())
    except OSError as exc:
        raise ValueError(f"Could not read folder: {exc}") from exc
    for child in children:
        try:
            if child.is_dir():
                directories.append({"name": child.name, "path": str(child)})
        except OSError:
            continue
    return {
        "path": str(current),
        "requested": str(requested),
        "parent": str(current.parent) if current != current.parent else None,
        "directories": directories,
    }


def folder_status(value: str | None) -> dict[str, object]:
    """Return the small amount of information needed before a destructive run."""
    requested = Path(value).expanduser().resolve() if value else ROOT
    if requested.exists() and not requested.is_dir():
        raise ValueError("Results folder must be a folder")
    if not requested.exists():
        return {"path": str(requested), "exists": False, "item_count": 0, "items": [], "has_more": False}
    try:
        items = sorted(requested.iterdir(), key=lambda path: path.name.lower())
    except OSError as exc:
        raise ValueError(f"Could not inspect the results folder: {exc}") from exc
    visible_items = [item.name for item in items[:5]]
    return {
        "path": str(requested),
        "exists": True,
        "item_count": len(items),
        "items": visible_items,
        "has_more": len(items) > len(visible_items),
    }


def add_log(message: str):
    entry = f"[{now()}] {message}"
    with state_lock:
        state["logs"].append(entry)
        state["logs"] = state["logs"][-MAX_LOG_LINES:]
    try:
        with GUI_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(entry + "\n")
    except OSError:
        pass


def tif_count(directory: str) -> int:
    try:
        return sum(path.is_file() and path.suffix.lower() == ".tif" for path in Path(directory).iterdir())
    except OSError:
        return 0


def estimate_model_work(config: dict) -> int:
    """Estimate benchmark sub-tasks so the long model step has useful movement."""
    data_file = Path(config["results_dir"]) / "datafile.tsv"
    feature_count = 256
    try:
        with data_file.open(encoding="utf-8") as data:
            feature_count = max(1, len(data.readline().rstrip("\n").split("\t")) - 1)
    except OSError:
        pass
    component_count = sum(component <= feature_count for component in (16, 32, 64, 128, 256, 512)) + 1
    model_count = 6 if config.get("all_learners") else 1
    cv_folds = 3
    simple_runs = 3 * component_count * 2 * cv_folds * model_count * 2
    rfe_runs = max(1, math.ceil(feature_count / 20) * cv_folds)
    return max(10, simple_runs + rfe_runs + model_count + 2)


def progress_context(config: dict, steps: list[tuple[str, str, list[str]]]) -> dict:
    step_ids = [step[0] for step in steps]
    if config["workflow"] == "full":
        weights = {"prepare": 0.07, "features": 0.39, "validate": 0.06, "models": 0.33, "inference": 0.15}
    elif config["workflow"] == "train":
        weights = {"prepare": 0.07, "features": 0.48, "validate": 0.06, "models": 0.39}
    elif config["workflow"] in {"features_labelled", "features_unlabelled"}:
        weights = {"prepare": 0.10, "features": 0.82, "validate": 0.08}
    else:
        weights = {"prepare": 0.12, "inference": 0.88}
    weights = {step_id: weights[step_id] for step_id in step_ids}
    weight_total = sum(weights.values()) or 1
    weights = {step_id: weight / weight_total for step_id, weight in weights.items()}
    image_directory = (
        config["inference_images"] if config["workflow"] in {"inference", "features_unlabelled"} else config["training_images"]
    )
    return {
        "weights": weights,
        "image_total": max(1, tif_count(image_directory)),
        "feature_writes": 0,
        "feature_post": 0,
        "model_events": 0,
        "model_total": estimate_model_work(config),
        "model_rfe": 0.0,
        "model_post": 0,
        "inference_images": 0,
        "inference_post": 0,
        "detail": "Preparing pipeline",
    }


def progress_fraction(step_id: str, context: dict) -> float:
    if step_id == "prepare":
        return 0.0
    if step_id == "features":
        total = context["image_total"] + 3
        return min(1.0, (min(context["image_total"], context["feature_writes"] // 2) + context["feature_post"]) / total)
    if step_id == "models":
        completed = min(context["model_total"] - 1, context["model_events"])
        completed += context["model_rfe"]
        completed += min(1, context["model_post"])
        return min(1.0, completed / context["model_total"])
    if step_id == "inference":
        total = context["image_total"] + 3
        completed = min(context["image_total"], context["inference_images"]) + context["inference_post"]
        return min(1.0, completed / total)
    return 0.0


def update_progress(step_id: str, line: str | None = None, complete: bool = False):
    with state_lock:
        context = runtime["progress_context"]
        if not context:
            return
        if line:
            context["detail"] = line[:160]
            if step_id == "features":
                if "writing " in line:
                    context["feature_writes"] += 1
                if "Running step: create joint dataframe" in line:
                    context["feature_post"] = max(context["feature_post"], 1)
                elif "Running step: compute aggregated features" in line:
                    context["feature_post"] = max(context["feature_post"], 2)
                elif "Running step: create final dataframe" in line:
                    context["feature_post"] = max(context["feature_post"], 3)
                if "writing " in line:
                    context["detail"] = (
                        f"Completed {min(context['image_total'], context['feature_writes'] // 2)} of {context['image_total']} input images"
                    )
            elif step_id == "models":
                if "Stored partial evaluation" in line or "Loaded existing partial evaluation" in line:
                    context["model_events"] += 1
                    context["detail"] = f"Completed {context['model_events']} benchmark evaluations"
                match = re.search(r"Testing top features: (\d+) out of (\d+)", line)
                if match:
                    current, total = (int(value) for value in match.groups())
                    context["model_rfe"] = current / max(1, total) * max(1, context["model_total"] * 0.15)
                    context["detail"] = f"RFE: {current} of {total} feature sets"
                if "Running step: visualize benchmark results" in line:
                    context["model_post"] = 1
            elif step_id == "inference":
                if "Generated features for " in line:
                    context["inference_images"] += 1
                    context["detail"] = f"Generated features for {context['inference_images']} of {context['image_total']} images"
                if "Running inference with model: " in line:
                    context["inference_images"] = context["image_total"]
                    context["inference_post"] = max(context["inference_post"], 1)
                    context["detail"] = "Running image classification"
                if "Generating SHAP explanations" in line:
                    context["inference_post"] = max(context["inference_post"], 2)
                    context["detail"] = "Generating explanations"
                if "Inference complete." in line:
                    context["inference_post"] = 3
        if complete:
            fraction = 1.0
        else:
            fraction = progress_fraction(step_id, context)
        current = next((item for item in state["steps"] if item["id"] == step_id), None)
        if current and line:
            current["detail"] = context["detail"]
        completed_stages = sum(item["status"] == "complete" for item in state["steps"])
        percent = sum(
            context["weights"].get(item["id"], 0)
            * (1 if item["status"] == "complete" else fraction if item["id"] == step_id and item["status"] == "running" else 0)
            for item in state["steps"]
        )
        state["progress"].update(
            completed=completed_stages,
            total=len(state["steps"]),
            percent=round(percent * 100),
            label=f"Working: {current['label']}"
            if current and current["status"] == "running"
            else state["progress"].get("label", "Waiting"),
            detail=context["detail"],
        )


def set_step(step_id: str, status: str, detail: str | None = None):
    with state_lock:
        for step in state["steps"]:
            if step["id"] == step_id:
                step["status"] = status
                if detail is not None:
                    step["detail"] = detail
                if status == "running":
                    step["started_at"] = now()
                    state["current_step"] = step_id
                if status in {"complete", "failed", "cancelled"}:
                    step["finished_at"] = now()
                    if state["current_step"] == step_id:
                        state["current_step"] = None
                completed = sum(item["status"] == "complete" for item in state["steps"])
                total = len(state["steps"])
                current = next((item for item in state["steps"] if item["status"] == "running"), None)
                state["progress"] = {
                    "completed": completed,
                    "total": total,
                    "percent": round(completed / total * 100) if total else 0,
                    "label": (
                        f"Working: {current['label']}"
                        if current
                        else "Complete"
                        if completed == total and total
                        else "Failed"
                        if status == "failed"
                        else "Stopped"
                        if status == "cancelled"
                        else "Waiting"
                    ),
                }
                return


def safe_path(value: object, label: str, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    path = Path(value).expanduser().resolve()
    if must_exist and not path.is_dir():
        raise ValueError(f"{label} must be an existing folder")
    return path


def safe_file(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} must be an existing file")
    return path


def generated_feature_path(config: dict) -> Path:
    filename = "unknown_features.tsv" if config["workflow"] == "features_unlabelled" else "datafile.tsv"
    return Path(config["results_dir"]) / filename


def gui_feature_validation_path(config: dict) -> Path:
    # Docker owns files it creates below results/validation on Unix. Keep the
    # host-side GUI report in the user-created results root so it is writable.
    return Path(config["results_dir"]) / "gui_feature_validation.json"


def validate_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Request body must be an object")
    config = default_config()
    config.update(raw)
    workflow = config.get("workflow")
    if workflow not in {"full", "train", "inference", "features_labelled", "features_unlabelled"}:
        raise ValueError("Choose a supported workflow")

    results_dir = safe_path(config.get("results_dir"), "Results folder", workflow == "inference")
    if workflow in {"inference", "features_unlabelled"}:
        training_images = safe_path(config.get("inference_images"), "Inference images", True)
    else:
        training_images = safe_path(config.get("training_images"), "Training images", True)
    if workflow in {"full", "inference"}:
        inference_images = safe_path(config.get("inference_images"), "Inference images", True)
        inference_output = safe_path(config.get("inference_output"), "Inference output")
    else:
        inference_images = training_images
        inference_output = results_dir

    try:
        workers = int(config.get("workers", 4))
        top_features = int(config.get("top_features", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("Workers and top features must be whole numbers") from exc
    if not 1 <= workers <= 64:
        raise ValueError("Workers must be between 1 and 64")
    if not 1 <= top_features <= 500:
        raise ValueError("Top features must be between 1 and 500")
    try:
        correlation_threshold = float(config.get("correlation_threshold", 0.8))
    except (TypeError, ValueError) as exc:
        raise ValueError("Correlation threshold must be a number") from exc
    if not 0 < correlation_threshold <= 1:
        raise ValueError("Correlation threshold must be between 0 and 1")

    cpu_limit = config.get("cpu_limit")
    if cpu_limit is None or cpu_limit == "":
        cpu_limit = None
    else:
        try:
            cpu_limit = float(cpu_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("CPU limit must be a positive number") from exc
        if not 0 < cpu_limit <= (os.cpu_count() or 1):
            raise ValueError("CPU limit must be between 0 and the number of logical CPUs")
        cpu_limit = round(cpu_limit, 2)

    memory_limit = config.get("memory_limit")
    if memory_limit is None or memory_limit == "":
        memory_limit = None
    elif not isinstance(memory_limit, str) or not re.fullmatch(r"[1-9][0-9]*(?:m|g)", memory_limit.lower()):
        raise ValueError("Memory limit must use megabytes or gigabytes, for example 4096m")

    resume = workflow in {"full", "train"} and (results_dir / "partial").is_dir() and (results_dir / "datafile.tsv").is_file()
    if (
        workflow in {"full", "features_labelled", "features_unlabelled"}
        and results_dir.exists()
        and any(results_dir.iterdir())
        and not config.get("confirm_cleanup")
        and not resume
        and not (workflow == "full" and config.get("feature_file"))
    ):
        raise ValueError("The results folder is not empty. Confirm cleanup before starting.")

    replication_unit = config.get("replication_unit", "date")
    if replication_unit not in {"position", "well", "plate", "date"}:
        raise ValueError("Replication unit must be position, well, plate, or date")
    feature_file = config.get("feature_file") or ""
    if feature_file:
        feature_file = str(safe_file(feature_file, "Feature table"))
    if workflow == "train" and not feature_file and not (results_dir / "datafile.tsv").is_file():
        raise ValueError("Train models requires a complete feature table. Select one or generate results/datafile.tsv first.")
    learner = str(config.get("learner", "rf"))
    if learner not in {"rf", "dummy", "decisiontree", "logistic", "xgb", "gridsearch"}:
        raise ValueError("Choose a supported learner")

    results_dir.mkdir(parents=True, exist_ok=True)
    inference_output.mkdir(parents=True, exist_ok=True)
    return {
        "workflow": workflow,
        "feature_mode": "unlabelled" if workflow == "features_unlabelled" else "labelled",
        "training_images": str(training_images),
        "results_dir": str(results_dir),
        "inference_images": str(inference_images),
        "inference_output": str(inference_output),
        "workers": workers,
        "top_features": top_features,
        "correlation_threshold": correlation_threshold,
        "all_learners": bool(config.get("all_learners")),
        "learner": learner,
        "confirm_cleanup": bool(config.get("confirm_cleanup")),
        "cpu_limit": cpu_limit,
        "memory_limit": memory_limit.lower() if memory_limit else None,
        "replication_unit": replication_unit,
        "feature_file": feature_file,
        "resume": resume,
    }


def preflight_config(config: dict) -> dict:
    image_path = config["inference_images"] if config["workflow"] in {"inference", "features_unlabelled"} else config["training_images"]
    labelled = config["workflow"] not in {"inference", "features_unlabelled"}
    report = {"images": validate_image_directory(image_path, labelled=labelled)}
    if config.get("feature_file"):
        report["features"] = validate_feature_table(config["feature_file"], require_label=labelled)
    elif config["workflow"] == "train":
        report["features"] = validate_feature_table(
            Path(config["results_dir"]) / "datafile.tsv",
            require_label=True,
        )
    report["ok"] = all(section.get("ok", False) for section in report.values())
    return report


def create_env(config: dict) -> str:
    values = {
        "IMAGINE_IMAGES": config["training_images"]
        if config["workflow"] not in {"inference", "features_unlabelled"}
        else config["inference_images"],
        "IMAGINE_REPLICATION_UNIT": config.get("replication_unit", "date"),
        "IMAGINE_RESULTS": config["results_dir"],
        "IMAGINE_INFERENCE_INPUTS": config["inference_images"],
        "IMAGINE_INFERENCE_OUTPUTS": config["inference_output"],
        "IMAGINE_INFERENCE_DATAFILE": config["results_dir"],
    }
    handle = tempfile.NamedTemporaryFile("w", prefix="microics-", suffix=".env", delete=False)
    try:
        for key, value in values.items():
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
            handle.write(f'{key}="{escaped}"\n')
    finally:
        handle.close()
    return handle.name


def command_for(env_file: str, args: list[str], config: dict | None = None) -> list[str]:
    command = ["docker", "compose", "--env-file", env_file, "run", "--rm", "--no-TTY"]
    if config and config.get("cpu_limit"):
        command.extend(["--cpus", str(config["cpu_limit"])])
    if config and config.get("memory_limit"):
        command.extend(["--memory", config["memory_limit"]])
    return [*command, "imagine", *args]


def terminate_process(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def execute_step(step_id: str, label: str, command: list[str]) -> bool:
    set_step(step_id, "running", "Running")
    update_progress(step_id, "Starting")
    add_log(f"$ {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        set_step(step_id, "failed", str(exc))
        with state_lock:
            state["error"] = f"{label} could not start: {exc}"
        add_log(f"Could not start {label}: {exc}")
        return False

    with state_lock:
        runtime["process"] = process
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                add_log(line)
                update_progress(step_id, line)
            with state_lock:
                if runtime["stop_requested"]:
                    terminate_process(process)
        exit_code = process.wait()
    finally:
        with state_lock:
            runtime["process"] = None

    with state_lock:
        stopped = runtime["stop_requested"]
    if stopped:
        set_step(step_id, "cancelled", "Stopped")
        return False
    if exit_code:
        set_step(step_id, "failed", f"Exited with code {exit_code}")
        with state_lock:
            state["error"] = f"{label} failed with exit code {exit_code}"
        add_log(f"{label} failed with exit code {exit_code}")
        return False
    set_step(step_id, "complete", "Complete")
    update_progress(step_id, complete=True)
    add_log(f"{label} complete")
    return True


def refresh_artifacts(config: dict | None):
    if not config:
        return
    roots = {"results": Path(config["results_dir"])}
    if config["workflow"] in {"full", "inference"}:
        roots["inference"] = Path(config["inference_output"])
    artifacts = []
    for root_id, root in roots.items():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(root)
                size = path.stat().st_size
            except OSError:
                continue
            parts = relative.parts
            visible = len(parts) == 1 or parts[0] in {"visualizations", "validation"}
            if parts and parts[0] == "explanations":
                visible = len(parts) <= 2 and ("summary" in path.name or "shap" in path.name or "importance" in path.name)
            if not visible:
                continue
            artifacts.append(
                {
                    "root": root_id,
                    "path": relative.as_posix(),
                    "name": path.name,
                    "size": size,
                    "kind": artifact_kind(path),
                }
            )
            if len(artifacts) >= MAX_ARTIFACTS:
                break
        if len(artifacts) >= MAX_ARTIFACTS:
            break
    with state_lock:
        state["artifacts"] = artifacts
        state["artifact_roots"] = {key: str(value) for key, value in roots.items()}


def artifact_kind(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if extension in {".html", ".htm"}:
        return "html"
    if extension in {".tsv", ".csv", ".txt", ".log"}:
        return "text"
    return "file"


# Show previously generated default outputs as soon as the browser connects.
refresh_artifacts(default_config())


def run_pipeline(config: dict):
    env_file = create_env(config)
    try:
        if config.get("feature_file") and config["workflow"] in {"full", "train", "inference"}:
            destination = Path(config["results_dir"]) / "datafile.tsv"
            source = Path(config["feature_file"])
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            add_log(f"Using validated feature table: {destination}")
        add_log("Resuming benchmark from saved partial evaluations" if config.get("resume") else "Pipeline started")
        steps = [("prepare", "Prepare Docker image", ["docker", "compose", "--env-file", env_file, "build", "imagine"])]
        if config["workflow"] in {"full", "train", "features_labelled", "features_unlabelled"}:
            steps.extend(
                [
                    (
                        "features",
                        "Generate features",
                        command_for(
                            env_file,
                            [
                                str(config["workers"]),
                                "datafile.tsv",
                                str(config["top_features"]),
                                "generate_features",
                                *(["--unlabelled"] if config["workflow"] == "features_unlabelled" else []),
                            ],
                            config,
                        ),
                    ),
                    (
                        "models",
                        "Train and save models",
                        command_for(
                            env_file,
                            [
                                str(config["workers"]),
                                "datafile.tsv",
                                str(config["top_features"]),
                                "learning_benchmark_save_models",
                                "--learner",
                                config.get("learner", "rf"),
                                "--correlation-threshold",
                                str(config.get("correlation_threshold", 0.8)),
                                *(["--all_learners"] if config["all_learners"] else []),
                            ],
                            config,
                        ),
                    ),
                ]
            )
            if config["workflow"] == "train":
                steps.pop(1)  # Train from the complete feature table already in the results folder.
            elif (config.get("feature_file") or config.get("resume")) and config["workflow"] == "full":
                steps.pop(1)  # Use the supplied, already validated feature table.
            elif config["workflow"] in {"features_labelled", "features_unlabelled"}:
                steps = steps[:2]  # Feature generation is an independent GUI module.
            if config["workflow"] in {"full", "train", "features_labelled", "features_unlabelled"}:
                validation_command = [
                    sys.executable,
                    str(ROOT / "src" / "validate_inputs.py"),
                    "--images",
                    config["training_images"],
                    "--features",
                    str(generated_feature_path(config)),
                    "--output",
                    str(gui_feature_validation_path(config)),
                ]
                if config["workflow"] == "features_unlabelled":
                    validation_command.append("--unlabelled")
                validation_index = next((index for index, step in enumerate(steps) if step[0] == "models"), len(steps))
                steps.insert(validation_index, ("validate", "Validate generated features", validation_command))
        if config["workflow"] in {"full", "inference"}:
            steps.append(
                (
                    "inference",
                    "Classify new images",
                    command_for(
                        env_file,
                        [
                            str(config["workers"]),
                            "/imagine/inference_datafile/datafile.tsv" if config.get("feature_file") else "-",
                            str(config["top_features"]),
                            "inference",
                        ],
                        config,
                    ),
                )
            )

        with state_lock:
            state["steps"] = [{"id": step_id, "label": label, "status": "pending", "detail": "Waiting"} for step_id, label, _ in steps]
            state["progress"] = {"completed": 0, "total": len(steps), "percent": 0, "label": "Preparing pipeline"}
            runtime["progress_context"] = progress_context(config, steps)
        for step_id, label, command in steps:
            if not execute_step(step_id, label, command):
                report_path = gui_feature_validation_path(config)
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    with state_lock:
                        state["validation"] = {**(state.get("validation") or {}), **report}
                except (OSError, json.JSONDecodeError):
                    pass
                refresh_artifacts(config)
                with state_lock:
                    cancelled = runtime["stop_requested"]
                add_log("Pipeline stopped" if cancelled else "Pipeline stopped after an error")
                break
            if step_id == "validate":
                report_path = gui_feature_validation_path(config)
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    with state_lock:
                        state["validation"] = {**(state.get("validation") or {}), **report}
                except (OSError, json.JSONDecodeError) as exc:
                    add_log(f"Could not load feature validation report: {exc}")
            refresh_artifacts(config)
        else:
            add_log("Pipeline complete")
        with state_lock:
            if runtime["stop_requested"]:
                state["status"] = "cancelled"
            elif any(step["status"] == "failed" for step in state["steps"]):
                state["status"] = "failed"
            else:
                state["status"] = "complete"
            state["finished_at"] = now()
        refresh_artifacts(config)
    except Exception as exc:  # Keep failures visible in the UI rather than killing the worker silently.
        with state_lock:
            state["status"] = "failed"
            state["error"] = str(exc)
            state["finished_at"] = now()
        add_log(f"Unexpected error: {exc}")
    finally:
        with state_lock:
            runtime["process"] = None
            runtime["progress_context"] = None
        try:
            os.unlink(env_file)
        except OSError:
            pass


def start_pipeline(raw_config: dict) -> tuple[bool, str | None]:
    with state_lock:
        if state["status"] == "running":
            return False, "A pipeline is already running"
    try:
        config = validate_config(raw_config)
        preflight = preflight_config(config)
        if not preflight["ok"]:
            errors = []
            image_report = preflight.get("images", {})
            if image_report.get("invalid_filenames"):
                errors.append("Malformed image filenames: " + ", ".join(image_report["invalid_filenames"][:5]))
            for section in preflight.values():
                if isinstance(section, dict):
                    errors.extend(section.get("errors", []))
            return False, "Preflight validation failed. " + "; ".join(errors or [image_report.get("message", "Check the input data")])
    except ValueError as exc:
        return False, str(exc)
    try:
        GUI_LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    with state_lock:
        runtime["stop_requested"] = False
        state.update(
            {
                "status": "running",
                "current_step": None,
                "started_at": now(),
                "finished_at": None,
                "error": None,
                "validation": preflight,
                "config": config,
                "steps": [],
                "progress": {"completed": 0, "total": 0, "percent": 0, "label": "Preparing pipeline"},
                "logs": [],
                "artifacts": [],
            }
        )
    thread = threading.Thread(target=run_pipeline, args=(config,), daemon=True)
    thread.start()
    return True, None


def stop_pipeline() -> bool:
    with state_lock:
        if state["status"] != "running":
            return False
        runtime["stop_requested"] = True
        process = runtime["process"]
    if process:
        terminate_process(process)
    add_log("Stop requested")
    return True


def snapshot(refresh=False) -> dict:
    with state_lock:
        config = state["config"] or default_config()
    if refresh and config:
        refresh_artifacts(config)
    with state_lock:
        return {
            key: state[key]
            for key in (
                "status",
                "current_step",
                "started_at",
                "finished_at",
                "error",
                "validation",
                "config",
                "steps",
                "progress",
                "logs",
                "artifacts",
                "artifact_roots",
            )
        }


def resolve_artifact(root_id: str, relative: str) -> Path:
    with state_lock:
        root_value = state["artifact_roots"].get(root_id)
    if not root_value:
        raise FileNotFoundError("Unknown artifact folder")
    root = Path(root_value).resolve()
    candidate = (root / unquote(relative)).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError("Artifact path is outside the results folder")
    if not candidate.is_file():
        raise FileNotFoundError("Artifact does not exist")
    return candidate


class Handler(BaseHTTPRequestHandler):
    server_version = "MicroICS-GUI/1.0"

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            json_response(self, snapshot(parse_qs(parsed.query).get("refresh") == ["1"]))
            return
        if parsed.path == "/api/defaults":
            json_response(self, {"version": VERSION, "config": default_config(), "demo": demo_config()})
            return
        if parsed.path == "/api/hardware-defaults":
            json_response(self, hardware_defaults())
            return
        if parsed.path == "/api/browse":
            try:
                path = parse_qs(parsed.query).get("path", [""])[0]
                json_response(self, browse_directory(path or None))
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/folder-status":
            try:
                path = parse_qs(parsed.query).get("path", [""])[0]
                json_response(self, folder_status(path or None))
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path in {"/api/artifact", "/api/preview"}:
            query = parse_qs(parsed.query)
            try:
                path = resolve_artifact(query.get("root", [""])[0], query.get("path", [""])[0])
            except (FileNotFoundError, PermissionError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            if parsed.path == "/api/preview":
                try:
                    text = path.read_text(errors="replace")[:24000]
                except OSError as exc:
                    json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                json_response(self, {"name": path.name, "content": text})
                return
            self.serve_file(path)
            return
        if parsed.path == "/" or parsed.path == "/index.html":
            self.serve_file(GUI_DIR / "index.html")
            return
        static_name = parsed.path.lstrip("/")
        static_path = (GUI_DIR / static_name).resolve()
        if GUI_DIR in static_path.parents and static_path.is_file():
            self.serve_file(static_path)
            return
        if parsed.path == "/logo.png":
            self.serve_file(ROOT / "logo.png")
            return
        if parsed.path == "/logo_dark.png":
            self.serve_file(ROOT / "logo_dark.png")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/open-folder":
            try:
                payload = read_json(self)
                folder = safe_path(payload.get("path"), "Output folder")
                folder.mkdir(parents=True, exist_ok=True)
                if os.name == "nt":
                    os.startfile(str(folder))  # type: ignore[attr-defined]
                elif shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    raise ValueError("No desktop folder opener is available")
                json_response(self, {"path": str(folder)})
            except (OSError, ValueError) as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/preflight":
            try:
                config = validate_config(read_json(self))
                json_response(self, {"config": config, "report": preflight_config(config)})
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/upload":
            try:
                json_response(self, receive_upload(self), HTTPStatus.CREATED)
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except OSError as exc:
                json_response(self, {"error": f"Could not save the selected file: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/upload-feature":
            try:
                json_response(
                    self,
                    receive_upload(self, {".tsv", ".txt"}, "tab-separated .tsv or .txt files"),
                    HTTPStatus.CREATED,
                )
            except ValueError as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except OSError as exc:
                json_response(self, {"error": f"Could not save the selected file: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/run":
            try:
                payload = read_json(self)
                started, error = start_pipeline(payload)
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
        self.send_error(HTTPStatus.NOT_FOUND)

    def serve_file(self, path: Path):
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Run the MicroICS local pipeline GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
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
