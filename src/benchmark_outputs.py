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


def canonical_model_name(model_name: object) -> str:
    """Return the stable learner key used by current and legacy result files."""
    raw_name = re.sub(r"\s+", " ", str(model_name)).strip()
    lowered = raw_name.lower()
    if lowered in MODEL_DISPLAY_NAMES:
        return lowered
    if "randomforestclassifier" in lowered:
        return "rf"
    if "dummyclassifier" in lowered:
        return "dummy"
    if "decisiontreeclassifier" in lowered:
        return "decisiontree"
    if "logisticregression" in lowered:
        return "logistic"
    if "xgbclassifier" in lowered:
        return "xgb"
    if "gridsearchcv" in lowered or "kneighborsclassifier" in lowered:
        return "gridsearch"
    return raw_name or "model"


def _model_report_name(model_name: object, max_length: int = 80) -> tuple[str, str]:
    """Return readable, filesystem-safe names for current and legacy reports."""
    raw_name = re.sub(r"\s+", " ", str(model_name)).strip() or "model"
    canonical_name = canonical_model_name(raw_name)
    display_name = MODEL_DISPLAY_NAMES.get(canonical_name, raw_name)
    safe_source = canonical_name if canonical_name in MODEL_DISPLAY_NAMES else raw_name
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", safe_source).strip("_") or "model"
    if len(safe_name) > max_length:
        digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
        safe_name = f"{safe_name[: max_length - len(digest) - 1].rstrip('_')}_{digest}"
        display_name = f"Legacy model {digest}"
    return display_name, safe_name


def _boolean_values(values: pd.Series) -> pd.Series:
    """Normalize booleans stored by pandas as bools or TSV strings."""
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _preferred_configuration(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the predeclared all-feature configuration without test-score cherry-picking."""
    candidates = frame.copy()
    if "thr_features" in candidates:
        with_thresholds = candidates[_boolean_values(candidates["thr_features"])]
        if not with_thresholds.empty:
            candidates = with_thresholds

    if "component_setting" in candidates:
        all_features = candidates[candidates["component_setting"].astype(str).str.lower() == "all"]
        if not all_features.empty:
            return all_features

    if "n_components" in candidates:
        all_features = candidates[candidates["n_components"].astype(str).str.lower() == "all"]
        if not all_features.empty:
            return all_features

    if "n_components" not in candidates:
        return candidates
    numeric_components = pd.to_numeric(candidates["n_components"], errors="coerce")
    if numeric_components.notna().any():
        return candidates[numeric_components == numeric_components.max()]
    return candidates


def _ranking_features(rankings: pd.DataFrame, top_n: int) -> list[str]:
    feature_column = "feature" if "feature" in rankings.columns else rankings.columns[0]
    score_columns = [column for column in rankings.columns if column != feature_column]
    if not score_columns:
        return []
    return rankings.sort_values(score_columns[0], ascending=False)[feature_column].astype(str).head(top_n).tolist()


def _classification_scope(source: Path) -> tuple[str, str]:
    """Return filename and title labels that keep benchmark variants separate."""
    scope = source.stem.removeprefix("classification_")
    if scope in {"", "classification"}:
        return "", "All generated columns"
    safe_scope = re.sub(r"[^A-Za-z0-9_-]+", "_", scope).strip("_")
    display_scope = {
        "all": "All generated columns",
        "no_counts_features": "Without count-derived columns",
    }.get(scope, scope.replace("_", " ").capitalize())
    return f"{safe_scope}_", display_scope


def write_confusion_matrices(classification_file: str | Path, output_dir: str | Path) -> list[Path]:
    """Write matrices for each learner's predeclared all-feature evaluation."""
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
    scope_prefix, scope_display = _classification_scope(source)
    outputs: list[Path] = []
    frame["canonical_model"] = frame["model"].map(canonical_model_name)
    for model_name, model_frame in frame.groupby("canonical_model", sort=True):
        selected = _preferred_configuration(model_frame)
        true_values = [value for values in selected["test_set"] for value in str(values).split(",")]
        predictions = [value for values in selected["predicted_set"] for value in str(values).split(",")]
        if len(true_values) != len(predictions):
            continue
        labels = sorted(set(true_values) | set(predictions))
        matrix = confusion_matrix(true_values, predictions, labels=labels)
        display_name, safe_name = _model_report_name(model_name)
        table = pd.DataFrame(matrix, index=labels, columns=labels)
        table.index.name = "true"
        report_stem = f"confusion_matrix_{scope_prefix}{safe_name}"
        table.to_csv(destination / f"{report_stem}.tsv", sep="\t")
        row_totals = table.sum(axis=1).replace(0, np.nan)
        normalized_table = table.div(row_totals, axis=0).fillna(0.0)
        normalized_table.index.name = "true"
        normalized_table.to_csv(destination / f"{report_stem}_normalized.tsv", sep="\t")
        plt.figure(figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 0.9)))
        sns.heatmap(normalized_table, annot=True, fmt=".1%", cmap="Blues", cbar=False, vmin=0, vmax=1)
        mean_accuracy = pd.to_numeric(selected["accuracy"], errors="coerce").mean()
        plt.title(f"Held-out confusion matrix: {display_name}\n{scope_display}; mean accuracy {mean_accuracy:.3f}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        image_path = destination / f"{report_stem}.png"
        plt.savefig(image_path, dpi=180)
        plt.close()
        outputs.append(image_path)
    return outputs


def write_classification_plot(classification_file: str | Path, output_dir: str | Path) -> Path | None:
    """Plot held-out accuracy for current short learner names and legacy estimator strings."""
    source = Path(classification_file)
    if not source.is_file():
        return None
    frame = pd.read_csv(source, sep="\t")
    required = {"model", "n_components", "accuracy"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
    frame = frame.dropna(subset=["accuracy"])
    if frame.empty:
        return None
    frame["canonical_model"] = frame["model"].map(canonical_model_name)
    frame["display_model"] = frame["canonical_model"].map(lambda value: MODEL_DISPLAY_NAMES.get(value, value))
    frame["components"] = frame["n_components"].astype(str)
    preferred_order = list(MODEL_DISPLAY_NAMES)
    present = frame["canonical_model"].drop_duplicates().tolist()
    model_order = [MODEL_DISPLAY_NAMES[key] for key in preferred_order if key in present]
    model_order.extend(MODEL_DISPLAY_NAMES.get(key, key) for key in present if key not in preferred_order)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, max(4, len(model_order) * 0.75 + 2)))
    sns.barplot(
        data=frame,
        y="display_model",
        x="accuracy",
        hue="components",
        order=model_order,
        palette="colorblind",
        alpha=0.7,
        capsize=0.15,
    )
    plt.title(source.stem)
    plt.ylabel("")
    plt.xlabel("Held-out accuracy")
    plt.xlim(0, 1)
    plt.legend(title="Components", loc="lower left")
    plt.tight_layout()
    output = destination / f"{source.stem}.pdf"
    plt.savefig(output, dpi=300)
    plt.close()
    return output


def write_ablation_plot(
    ablation_file: str | Path,
    output_dir: str | Path,
    total_features: int | None = None,
) -> Path | None:
    """Plot every evaluated RF subset and state that counts refer to generated columns."""
    source = Path(ablation_file)
    if not source.is_file():
        return None
    frame = pd.read_csv(source, sep="\t")
    if not {"top_n", "accuracy"}.issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["top_n"] = pd.to_numeric(frame["top_n"], errors="coerce")
    frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
    frame = frame.dropna(subset=["top_n", "accuracy"]).sort_values("top_n")
    if frame.empty:
        return None
    best = frame.loc[frame["accuracy"].idxmax()]
    if total_features is None:
        total_features = (
            int(frame["total_features"].dropna().iloc[0])
            if "total_features" in frame and frame["total_features"].notna().any()
            else int(frame["top_n"].max())
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=frame, x="top_n", y="accuracy")
    plt.vlines(best["top_n"], 0, best["accuracy"], color="red", linestyle="dashed")
    plt.plot(best["top_n"], best["accuracy"], "ro")
    plt.annotate(
        f"Highest observed subset (exploratory): {int(best['top_n'])} of {total_features} generated columns\n"
        f"Accuracy: {best['accuracy']:.3f}",
        xy=(best["top_n"], best["accuracy"]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    plt.title("RF feature-subset sensitivity (exploratory)")
    plt.xlabel("Top generated feature columns considered (RF ranking)")
    plt.ylabel("Mean held-out accuracy for each fixed subset size")
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    output = destination / "ablation_rf.pdf"
    plt.savefig(output, dpi=300)
    plt.close()
    return output


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
