"""Shared cross-validation planning for preflight and model evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

MIN_CV_SPLITS = 2


def _class_counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Classification requires at least 2 classes")
    return classes, counts


def _groups_per_class(y: np.ndarray, groups: np.ndarray, classes: np.ndarray) -> dict[Any, int]:
    return {label: int(len(np.unique(groups[y == label]))) for label in classes}


def _all_classes_present(values: np.ndarray, classes: np.ndarray) -> bool:
    return np.array_equal(np.unique(values), classes)


def _valid_splits(splitter, y: np.ndarray, groups: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray]] | None:
    samples = np.zeros((len(y), 1))
    try:
        iterator = splitter.split(samples, y, groups) if groups is not None else splitter.split(samples, y)
        splits = list(iterator)
    except ValueError:
        return None
    classes = np.unique(y)
    for train_indices, test_indices in splits:
        if not _all_classes_present(y[train_indices], classes) or not _all_classes_present(y[test_indices], classes):
            return None
        if groups is not None and not set(groups[train_indices]).isdisjoint(groups[test_indices]):
            return None
    return splits


def get_adaptive_cv(y, max_splits: int = 5, groups=None):
    """Return the largest deterministic stratified splitter that is actually valid."""
    y_values = np.asarray(y)
    if len(y_values) < MIN_CV_SPLITS:
        raise ValueError(f"Cannot create cross-validation splitter: at least {MIN_CV_SPLITS} samples are required, got {len(y_values)}")

    classes, class_counts = _class_counts(y_values)
    class_count_info = ", ".join(str(int(count)) for count in sorted(class_counts))
    min_class_count = int(class_counts.min())
    if min_class_count < MIN_CV_SPLITS:
        raise ValueError(
            "Cannot create stratified cross-validation splitter: "
            f"each class requires at least {MIN_CV_SPLITS} samples (class counts: {class_count_info})"
        )

    if groups is None:
        n_splits = min(int(max_splits), min_class_count)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if _valid_splits(splitter, y_values, None) is None:
            raise ValueError("Cannot create stratified cross-validation folds containing every class")
        reason = (
            f"StratifiedKFold enabled because minimum class count is {min_class_count}; "
            f"selected {n_splits} folds (class counts: {class_count_info})"
        )
        return splitter, n_splits, min_class_count, "stratified", reason

    group_values = np.asarray(groups)
    if len(group_values) != len(y_values):
        raise ValueError("Replication groups must have one value per sample")
    unique_groups = np.unique(group_values)
    if len(unique_groups) < MIN_CV_SPLITS:
        raise ValueError(
            f"Cannot create grouped cross-validation splitter: at least {MIN_CV_SPLITS} groups are required, got {len(unique_groups)}"
        )

    per_class = _groups_per_class(y_values, group_values, classes)
    min_class_groups = min(per_class.values())
    group_count_info = ", ".join(f"{label}: {count}" for label, count in per_class.items())
    maximum = min(int(max_splits), len(unique_groups), min_class_groups)
    for n_splits in range(maximum, MIN_CV_SPLITS - 1, -1):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if _valid_splits(splitter, y_values, group_values) is not None:
            reason = (
                f"StratifiedGroupKFold enabled for {len(unique_groups)} replication groups; selected {n_splits} folds "
                f"(groups per class: {group_count_info})"
            )
            return splitter, n_splits, min_class_count, "stratified-group", reason

    raise ValueError(
        "Cannot create grouped cross-validation folds containing every class in both training and evaluation data "
        f"(groups per class: {group_count_info})"
    )


def assess_grouped_cv(labels, groups, require_nested_cv: bool = True) -> dict[str, Any]:
    """Dry-run the outer and, when needed, inner splitters used by training."""
    y_values = np.asarray(labels)
    group_values = np.asarray(groups)
    label_counts = Counter(str(value) for value in y_values)
    report: dict[str, Any] = {
        "samples": int(len(y_values)),
        "class_counts": dict(sorted(label_counts.items())),
        "group_count": int(len(np.unique(group_values))) if len(group_values) else 0,
        "groups_per_class": {str(label): int(len(np.unique(group_values[y_values == label]))) for label in np.unique(y_values)},
        "outer_folds": 0,
        "inner_folds": [],
        "warnings": [],
        "errors": [],
    }
    try:
        outer_cv, outer_folds, _minimum, strategy, reason = get_adaptive_cv(y_values, max_splits=3, groups=group_values)
        outer_splits = list(outer_cv.split(np.zeros((len(y_values), 1)), y_values, group_values))
        report.update(outer_folds=outer_folds, strategy=strategy, reason=reason)
        if require_nested_cv:
            inner_folds = []
            for fold_number, (train_indices, _test_indices) in enumerate(outer_splits, start=1):
                try:
                    _inner_cv, fold_count, _minimum, _strategy, _reason = get_adaptive_cv(
                        y_values[train_indices], max_splits=5, groups=group_values[train_indices]
                    )
                except ValueError as exc:
                    raise ValueError(f"outer fold {fold_number} cannot support nested model tuning: {exc}") from exc
                inner_folds.append(fold_count)
            report["inner_folds"] = inner_folds
        if outer_folds == 2 or (report["inner_folds"] and min(report["inner_folds"]) == 2):
            report["warnings"].append("Only two-fold evaluation is available for at least one training stage; estimates may be unstable")
        report["status"] = "limited" if report["warnings"] else "suitable"
        report["ok"] = True
    except ValueError as exc:
        report.update(status="unavailable", ok=False)
        report["errors"].append(str(exc))
    return report
