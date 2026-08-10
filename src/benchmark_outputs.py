"""Small, deterministic benchmark reports used by the GUI and CLI pipeline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

MODEL_DISPLAY_NAMES = {
    "decisiontree": "Decision tree",
    "dummy": "Majority baseline",
    "gridsearch": "KNN grid search",
    "logistic": "Logistic regression",
    "rf": "Random forest",
    "xgb": "XGBoost",
}


def _model_report_name(model_name: object, max_length: int = 80) -> tuple[str, str]:
    """Return readable, filesystem-safe names for current and legacy reports."""
    raw_name = re.sub(r"\s+", " ", str(model_name)).strip() or "model"
    display_name = MODEL_DISPLAY_NAMES.get(raw_name, raw_name)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_") or "model"
    if len(safe_name) > max_length:
        digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
        safe_name = f"{safe_name[: max_length - len(digest) - 1].rstrip('_')}_{digest}"
        display_name = f"Legacy model {digest}"
    return display_name, safe_name


def _ranking_features(rankings: pd.DataFrame, top_n: int) -> list[str]:
    feature_column = "feature" if "feature" in rankings.columns else rankings.columns[0]
    score_columns = [column for column in rankings.columns if column != feature_column]
    if not score_columns:
        return []
    return rankings.sort_values(score_columns[0], ascending=False)[feature_column].astype(str).head(top_n).tolist()


def write_confusion_matrices(classification_file: str | Path, output_dir: str | Path) -> list[Path]:
    """Write raw TSV and PNG confusion matrices for each benchmarked learner."""
    source = Path(classification_file)
    if not source.is_file():
        return []
    frame = pd.read_csv(source, sep="\t")
    required = {"model", "test_set", "predicted_set", "accuracy"}
    if not required.issubset(frame.columns):
        return []
    frame = frame[frame["predicted_set"].fillna("").astype(str).str.len() > 0].copy()
    if frame.empty:
        return []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for model_name, model_frame in frame.groupby("model", sort=True):
        model_frame = model_frame.copy()
        model_frame["configuration"] = model_frame["n_components"].astype(str) + "|" + model_frame["thr_features"].astype(str)
        best_configuration = model_frame.groupby("configuration")["accuracy"].mean().idxmax()
        selected = model_frame[model_frame["configuration"] == best_configuration]
        true_values = [value for values in selected["test_set"] for value in str(values).split(",")]
        predictions = [value for values in selected["predicted_set"] for value in str(values).split(",")]
        if len(true_values) != len(predictions):
            continue
        labels = sorted(set(true_values) | set(predictions))
        matrix = confusion_matrix(true_values, predictions, labels=labels)
        display_name, safe_name = _model_report_name(model_name)
        table = pd.DataFrame(matrix, index=labels, columns=labels)
        table.index.name = "true"
        table.to_csv(destination / f"confusion_matrix_{safe_name}.tsv", sep="\t")
        plt.figure(figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 0.9)))
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Held-out confusion matrix: {display_name}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        image_path = destination / f"confusion_matrix_{safe_name}.png"
        plt.savefig(image_path, dpi=180)
        plt.close()
        outputs.append(image_path)
    return outputs


def write_feature_correlation(
    data_file: str | Path, rankings_file: str | Path, output_dir: str | Path, top_n: int = 20, threshold: float = 0.8
) -> list[Path]:
    """Write a clustered correlation heatmap and connected correlation clusters."""
    data = pd.read_csv(data_file, sep="\t", index_col=0)
    rankings = pd.read_csv(rankings_file, sep="\t")
    features = [feature for feature in _ranking_features(rankings, top_n) if feature in data.columns]
    numeric = data[features].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if len(numeric.columns) < 2:
        return []
    correlation = numeric.replace([np.inf, -np.inf], np.nan).corr()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "top_feature_correlation.tsv"
    correlation.to_csv(matrix_path, sep="\t")
    plot_values = correlation.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(plot_values, 1.0)
    plot_correlation = pd.DataFrame(plot_values, index=correlation.index, columns=correlation.columns)
    clustered = sns.clustermap(plot_correlation, cmap="vlag", center=0, vmin=-1, vmax=1, figsize=(12, 10))
    heatmap_path = destination / "top_feature_correlation_clustered.png"
    clustered.savefig(heatmap_path, dpi=180)
    plt.close(clustered.fig)

    parent = {feature: feature for feature in correlation.columns}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            if abs(float(correlation.loc[left, right])) > threshold:
                union(left, right)
    clusters: dict[str, list[str]] = {}
    for feature in correlation.columns:
        clusters.setdefault(find(feature), []).append(feature)
    clusters_path = destination / "top_feature_correlation_clusters.txt"
    with clusters_path.open("w", encoding="utf-8") as output:
        output.write(f"Clusters at |r| > {threshold}\n")
        for number, members in enumerate(sorted(clusters.values(), key=lambda values: values[0]), start=1):
            output.write(f"Cluster {number}: {', '.join(members)}\n")
    return [matrix_path, heatmap_path, clusters_path]


def write_feature_boxplots(data_file: str | Path, rankings_file: str | Path, output_dir: str | Path, top_n: int = 10) -> list[Path]:
    """Create class-stratified feature-value boxplots for the top ranked features."""
    data = pd.read_csv(data_file, sep="\t", index_col=0)
    if "label" not in data.columns:
        return []
    rankings = pd.read_csv(rankings_file, sep="\t")
    destination = Path(output_dir) / "feature_values"
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for feature in _ranking_features(rankings, top_n):
        if feature not in data.columns:
            continue
        numeric_values = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = pd.DataFrame({"label": data["label"].astype(str), "value": numeric_values}).dropna()
        if values.empty:
            continue
        plt.figure(figsize=(max(6, len(values["label"].unique()) * 1.2), 5))
        sns.boxplot(data=values, x="label", y="value", color="#6c8ebf")
        sns.stripplot(data=values, x="label", y="value", color="#202b3c", alpha=0.35, size=3)
        plt.title(f"{feature} by class")
        plt.xlabel("Class")
        plt.ylabel(feature)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(feature))
        output = destination / f"{safe_name}.png"
        plt.savefig(output, dpi=180)
        plt.close()
        outputs.append(output)
    return outputs
