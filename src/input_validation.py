"""Preflight checks shared by the command line pipeline and the GUI.

The checks in this module deliberately do not repair input.  A report is useful
only when it describes what will actually be passed to the learning code.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IMAGE_PATTERN = re.compile(
    r"^(?P<date>\d{8})[-_]s[-_](?P<species>[^_-]+)[-_]st[-_](?P<label>[^_-]+)"
    r"[-_]p[-_](?P<well>[^_-]+)[-_]pos(?P<position>[^_-]+)[-_]tm[-_](?P<time>[^_-]+)"
    r"[-_]ch[-_](?P<channel>[^_-]+)[-_]z[-_](?P<z>[^_-]+)$",
    re.IGNORECASE,
)

METADATA_COLUMNS = {"label", "date", "sampleName", "noPos", "Unnamed:", "Unnamed: 0"}
MICROICS_PREFIXES = (
    "counts",
    "diff",
    "max",
    "med",
    "mean",
    "std",
    "min",
    "minProp",
    "BioVolume",
    "Substratum",
    "Homogen",
    "Spreading",
    "GPT",
    "CustomAlgos",
    "DiffGlobal",
    "eigen",
    "globalMean",
    "mdiffs",
    "RoughnessThreshold",
    "ThicknessThreshold",
    "Area2D",
    "surf3D",
    "vol3D",
    "compactness",
)


def parse_image_name(filename: str) -> dict[str, str] | None:
    """Parse a MicroICS image name, accepting both ``_`` and ``--`` separators."""
    stem = Path(filename).stem
    match = IMAGE_PATTERN.match(stem)
    if not match:
        return None
    fields = match.groupdict()
    fields["plate"] = fields["label"]
    return fields


def sample_fields(sample_name: str) -> dict[str, str] | None:
    """Parse a data-table sample name after feature generation."""
    fields = parse_image_name(sample_name.replace("--", "_"))
    if fields:
        return fields
    # Some historical files omit the extension and use a mixed separator.
    tokens = [token for token in re.split(r"[-_]+", Path(sample_name).stem) if token]
    if len(tokens) >= 14 and tokens[0].isdigit() and len(tokens[0]) == 8:
        try:
            st_index = tokens.index("st")
            plate_index = tokens.index("p")
            return {
                "date": tokens[0],
                "species": tokens[2],
                "label": tokens[st_index + 1],
                "plate": tokens[st_index + 1],
                "well": tokens[plate_index + 1],
                "position": tokens[plate_index + 2].removeprefix("pos"),
                "time": tokens[plate_index + 3].removeprefix("tm"),
                "channel": tokens[plate_index + 4].removeprefix("ch"),
                "z": tokens[plate_index + 5].removeprefix("z"),
            }
        except (ValueError, IndexError):
            return None
    return None


def validate_image_directory(directory: str | Path, labelled: bool = True) -> dict[str, Any]:
    """Return a JSON-friendly filename, label, date, and replication report."""
    root = Path(directory)
    files = sorted(path for path in root.glob("*.tif") if path.is_file()) if root.is_dir() else []
    invalid: list[str] = []
    parsed: list[tuple[Path, dict[str, str]]] = []
    for path in files:
        fields = parse_image_name(path.name)
        label = fields.get("label", "").strip() if fields else ""
        if not fields or (labelled and (not label or label.lower() in {"unknown", "missing", "unlabelled", "unlabeled"})):
            invalid.append(path.name)
        else:
            parsed.append((path, fields))

    labels = Counter(fields["label"] for _, fields in parsed)
    per_date: dict[str, dict[str, int]] = defaultdict(dict)
    date_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for _, fields in parsed:
        date_labels[fields["date"]][fields["label"]] += 1
    for date, counts in sorted(date_labels.items()):
        per_date[date] = dict(sorted(counts.items()))

    return {
        "path": str(root.resolve()),
        "labelled": labelled,
        "total_images": len(files),
        "valid_images": len(parsed),
        "invalid_images": len(invalid),
        "invalid_filenames": invalid[:100],
        "unique_labels": sorted(labels),
        "images_per_label": dict(sorted(labels.items())),
        "images_per_label_per_date": per_date,
        "ok": bool(files) and not invalid,
        "message": (
            f"Validated {len(parsed)} of {len(files)} image filenames; found {len(labels)} unique label(s)."
            if files
            else "No .tif images were found."
        ),
    }


def _is_microics_feature(name: str) -> bool:
    return name.startswith(MICROICS_PREFIXES)


def validate_feature_table(path: str | Path, require_label: bool = True) -> dict[str, Any]:
    """Validate feature columns without silently coercing bad values."""
    source = Path(path)
    try:
        frame = pd.read_csv(source, sep="\t")
    except Exception as exc:  # pandas has several parser exception types
        return {"path": str(source), "ok": False, "errors": [f"Could not read feature table: {exc}"]}

    metadata = [column for column in frame.columns if column in METADATA_COLUMNS or str(column).startswith("Unnamed:")]
    feature_columns = [column for column in frame.columns if column not in metadata]
    microics = [column for column in feature_columns if _is_microics_feature(str(column))]
    external = [column for column in feature_columns if column not in microics]
    unparsed: list[str] = []
    empty_cells = 0
    nan_cells = 0
    infinite_cells = 0
    invalid_cells: dict[str, int] = {}

    for column in feature_columns:
        series = frame[column]
        blank = series.isna() | series.astype(str).str.strip().eq("")
        empty_cells += int(blank.sum())
        nan_cells += int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        non_numeric = (~blank) & numeric.isna()
        if non_numeric.any():
            unparsed.append(str(column))
            invalid_cells[str(column)] = int(non_numeric.sum())
        infinite = np.isinf(numeric.dropna().to_numpy()).sum()
        infinite_cells += int(infinite)

    errors = []
    if not feature_columns:
        errors.append("No feature columns were found")
    if "sampleName" not in frame.columns:
        errors.append("Required sampleName column is missing")
    else:
        sample_names = frame["sampleName"]
        missing_sample_names = sample_names.isna() | sample_names.astype(str).str.strip().eq("")
        if missing_sample_names.any():
            errors.append(f"{int(missing_sample_names.sum())} missing sampleName values found")
        duplicate_sample_names = sample_names.duplicated(keep=False)
        if duplicate_sample_names.any():
            errors.append(f"{int(duplicate_sample_names.sum())} rows have duplicate sampleName values")
    if require_label and "label" not in frame.columns:
        errors.append("Required label column is missing")
    elif require_label:
        labels = frame["label"]
        missing_labels = labels.isna() | labels.astype(str).str.strip().str.lower().isin(
            {"", "unknown", "missing", "unlabelled", "unlabeled"}
        )
        if missing_labels.any():
            errors.append(f"{int(missing_labels.sum())} missing/unknown label values found")
    if empty_cells:
        errors.append(f"{empty_cells} empty/NaN feature cells found")
    if infinite_cells:
        errors.append(f"{infinite_cells} infinite feature values found")
    if unparsed:
        errors.append("Could not parse feature columns: " + ", ".join(unparsed))

    return {
        "path": str(source.resolve()),
        "rows": int(len(frame)),
        "features_read": len(feature_columns),
        "microics_features": len(microics),
        "external_features": len(external),
        "microics_feature_names": [str(value) for value in microics],
        "external_feature_names": [str(value) for value in external],
        "unparsed_feature_names": unparsed,
        "empty_cells": empty_cells,
        "nan_cells": nan_cells,
        "infinite_cells": infinite_cells,
        "invalid_cells_by_feature": invalid_cells,
        "metadata_columns": [str(value) for value in metadata],
        "errors": errors,
        "ok": not errors,
    }


def replication_group(sample_name: str, unit: str) -> str:
    """Return the user-selected replication unit from a sample name."""
    if unit not in {"position", "well", "plate", "date"}:
        raise ValueError("Replication unit must be position, well, plate, or date")
    fields = sample_fields(sample_name)
    if not fields:
        raise ValueError(f"Could not parse replication fields from sample name: {sample_name}")
    return fields[unit]


def write_json_report(report: dict[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
