"""Create standalone plots from completed MicroICS benchmark TSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from benchmark_outputs import write_ablation_plot, write_classification_plot, write_confusion_matrices
except ImportError:  # Package import from the repository root.
    from .benchmark_outputs import write_ablation_plot, write_classification_plot, write_confusion_matrices


def create_benchmark_plots(results_folder: str | Path) -> list[Path]:
    """Render all available classification files and the RF ablation report."""
    results = Path(results_folder)
    output_dir = results / "visualizations"
    outputs: list[Path] = []
    for classification_file in sorted(results.glob("classification*.tsv")):
        output = write_classification_plot(classification_file, output_dir)
        if output:
            outputs.append(output)
        outputs.extend(write_confusion_matrices(classification_file, output_dir))

    total_features = None
    data_file = results / "datafile.tsv"
    if data_file.is_file():
        data_columns = pd.read_csv(data_file, sep="\t", nrows=0).columns
        total_features = sum(column not in {"sampleName", "label"} for column in data_columns)
    ablation_output = write_ablation_plot(
        results / "ablation_ranking_all.tsv",
        output_dir,
        total_features=total_features,
    )
    if ablation_output:
        outputs.append(ablation_output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_folder",
        nargs="?",
        default=os.environ.get("IMAGINE_RESULTS", "/imagine/results"),
        help="Folder containing classification and ablation TSV files",
    )
    arguments = parser.parse_args()
    for output in create_benchmark_plots(arguments.results_folder):
        print(output.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
