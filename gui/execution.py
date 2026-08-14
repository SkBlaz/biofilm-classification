"""Container-local execution plan for the MicroICS GUI.

This module only orchestrates existing MicroICS entry points. Scientific feature
generation, learning, and inference remain implemented in ``src/``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_ROOT = Path(os.environ.get("MICROICS_DATA_ROOT", ROOT / ".microics-data")).resolve()
JOBS_ROOT = DATA_ROOT / "jobs"
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    root: Path
    input: Path
    training_images: Path
    inference_images: Path
    feature_files: Path
    work: Path
    output: Path
    results: Path
    inference_output: Path


@dataclass(frozen=True)
class ExecutionStep:
    step_id: str
    label: str
    commands: tuple[tuple[str, ...], ...]


def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job ID")
    return job_id


def job_paths(job_id: str, create: bool = False) -> JobPaths:
    validate_job_id(job_id)
    root = (JOBS_ROOT / job_id).resolve()
    if root.parent != JOBS_ROOT.resolve():
        raise ValueError("Invalid job path")
    paths = JobPaths(
        job_id=job_id,
        root=root,
        input=root / "input",
        training_images=root / "input" / "training-images",
        inference_images=root / "input" / "inference-images",
        feature_files=root / "input" / "feature-files",
        work=root / "work",
        output=root / "output",
        results=root / "output" / "results",
        inference_output=root / "output" / "inference",
    )
    if create:
        for directory in (
            paths.training_images,
            paths.inference_images,
            paths.feature_files,
            paths.work,
            paths.results,
            paths.inference_output,
        ):
            directory.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise FileNotFoundError("Job does not exist")
    return paths


def create_job() -> JobPaths:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    return job_paths(uuid.uuid4().hex, create=True)


def safe_uploaded_name(filename: str, allowed_suffixes: set[str]) -> str:
    normalized = str(filename or "").replace("\\", "/")
    safe_name = Path(normalized).name
    if not safe_name or safe_name in {".", ".."} or "\x00" in safe_name:
        raise ValueError("The uploaded filename is invalid")
    if Path(safe_name).suffix.lower() not in allowed_suffixes:
        raise ValueError("The uploaded file type is not supported")
    return safe_name


def unique_upload_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = (directory / filename).resolve()
    if candidate.parent != directory.resolve():
        raise ValueError("The uploaded filename is outside the job input folder")
    if candidate.exists():
        candidate = directory / f"{candidate.stem}-{uuid.uuid4().hex[:8]}{candidate.suffix}"
    return candidate


def execution_environment(config: dict) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(SRC_DIR))),
            "MPLBACKEND": "Agg",
            "IMAGINE_IMAGES": config["training_images"],
            "IMAGINE_RESULTS": config["results_dir"],
            "IMAGINE_INFERENCE_INPUTS": config["inference_images"],
            "IMAGINE_INFERENCE_OUTPUTS": config["inference_output"],
            "IMAGINE_INFERENCE_DATAFILE": config["results_dir"],
            "IMAGINE_LEARNER": config["learner"],
            "IMAGINE_CORRELATION_THRESHOLD": str(config["correlation_threshold"]),
            "IMAGINE_VOXEL_SIZE_X": str(config["voxel_size_x"]),
            "IMAGINE_VOXEL_SIZE_Y": str(config["voxel_size_y"]),
            "IMAGINE_VOXEL_SIZE_Z": str(config["voxel_size_z"]),
        }
    )
    return environment


def stage_training_features(config: dict) -> Path | None:
    feature_file = config.get("feature_file")
    if not feature_file or config["workflow"] == "inference":
        return None
    destination = Path(config["results_dir"]) / "datafile.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(feature_file)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def build_execution_steps(config: dict) -> list[ExecutionStep]:
    workflow = config["workflow"]
    workers = str(config["workers"])
    top_features = str(config["top_features"])
    results = Path(config["results_dir"])
    datafile = results / "datafile.tsv"
    steps: list[ExecutionStep] = []

    should_generate = workflow in {"features_labelled", "features_unlabelled"} or (workflow == "full" and not config.get("feature_file"))
    if should_generate:
        arguments = ["bash", str(SRC_DIR / "run_analysis.sh"), workers, "datafile.tsv", top_features, "generate_features"]
        if workflow == "features_unlabelled":
            arguments.append("--unlabelled")
        steps.append(ExecutionStep("features", "Generate features", (tuple(arguments),)))
        generated_name = "unknown_features.tsv" if workflow == "features_unlabelled" else "datafile.tsv"
        validation = [
            sys.executable,
            str(SRC_DIR / "validate_inputs.py"),
            "--images",
            config["inference_images"] if workflow == "features_unlabelled" else config["training_images"],
            "--features",
            str(results / generated_name),
            "--output",
            str(results / "validation" / "feature_validation.json"),
        ]
        if workflow == "features_unlabelled":
            validation.append("--unlabelled")
        steps.append(ExecutionStep("validate", "Validate generated features", (tuple(validation),)))

    if workflow in {"train", "full"}:
        learner_args = [
            sys.executable,
            str(SRC_DIR / "feature_ranking_lite.py"),
            "--parallelism",
            workers,
            "--files",
            str(datafile),
            "--fout",
            str(results / "ranking.out"),
            "--save_models",
            "--learner",
            config["learner"],
            "--correlation-threshold",
            str(config["correlation_threshold"]),
        ]
        if config["all_learners"]:
            learner_args.append("--all_learners")
        steps.append(ExecutionStep("models", "Train and benchmark models", (tuple(learner_args),)))
        steps.append(
            ExecutionStep(
                "reports",
                "Create benchmark reports",
                ((sys.executable, str(SRC_DIR / "visualize_benchmark.py")),),
            )
        )

    if workflow in {"inference", "full"}:
        inference_args = [
            sys.executable,
            str(SRC_DIR / "inference.py"),
            config["models_dir"],
            config["inference_images"],
            config["inference_output"],
            "--temp_dir",
            config["inference_work_dir"],
        ]
        if workflow == "inference" and config.get("feature_file"):
            inference_args.extend(("--features_file", config["feature_file"]))
        steps.append(ExecutionStep("inference", "Classify samples", (tuple(inference_args),)))

    return steps
