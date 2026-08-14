"""Tests for deterministic benchmark report output names."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark_outputs import (
    _configuration_key,
    _model_report_name,
    canonical_model_name,
    write_ablation_plot,
    write_classification_plot,
    write_confusion_matrices,
    write_feature_boxplots,
    write_feature_correlation,
)


class TestBenchmarkOutputNames(unittest.TestCase):
    def test_plot_configuration_keeps_all_columns_and_threshold_only_separate(self):
        all_columns = pd.Series({"n_components": "all", "thr_features": True})
        threshold_only = pd.Series({"n_components": "all", "thr_features": False})

        self.assertEqual(_configuration_key(all_columns), "all_columns")
        self.assertEqual(_configuration_key(threshold_only), "threshold_only")

    def test_legacy_numeric_endpoints_are_not_mislabeled_as_svd(self):
        self.assertEqual(_configuration_key(pd.Series({"n_components": 2700, "thr_features": True})), "all_columns")
        self.assertEqual(_configuration_key(pd.Series({"n_components": 580, "thr_features": False})), "threshold_only")
        self.assertEqual(_configuration_key(pd.Series({"n_components": 32, "thr_features": False})), "svd_32")

    def test_known_learner_uses_short_readable_names(self):
        self.assertEqual(_model_report_name("rf"), ("Random forest", "rf"))

    def test_legacy_random_forest_name_maps_to_current_key(self):
        self.assertEqual(canonical_model_name("RandomizedSearchCV(estimator=RandomForestClassifier())"), "rf")

    def test_legacy_estimator_name_is_bounded_and_stable(self):
        legacy_name = "RandomizedSearchCV(" + "param_distributions={'n_estimators': [100, 200]}, " * 20 + ")"

        first = _model_report_name(legacy_name)
        second = _model_report_name(legacy_name)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first[1]), 80)
        self.assertRegex(first[1], r"_[0-9a-f]{12}$")

    def test_legacy_classification_file_writes_bounded_filenames(self):
        legacy_name = "RandomizedSearchCV(" + "very_long_parameter_description=" * 30 + ")"
        frame = pd.DataFrame(
            {
                "model": [legacy_name],
                "n_components": [16],
                "thr_features": [True],
                "accuracy": [0.5],
                "test_set": ["A,B"],
                "predicted_set": ["A,A"],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "classification.tsv"
            output = root / "visualizations"
            frame.to_csv(source, sep="\t", index=False)

            generated = write_confusion_matrices(source, output)

            self.assertEqual(len(generated), 1)
            self.assertTrue(generated[0].is_file())
            self.assertLessEqual(len(generated[0].name), 110)
            self.assertTrue(generated[0].with_suffix(".tsv").is_file())

    def test_confusion_matrix_uses_all_features_not_best_test_score(self):
        frame = pd.DataFrame(
            {
                "model": ["rf", "rf"],
                "n_components": [16, "all"],
                "thr_features": [True, True],
                "accuracy": [1.0, 0.5],
                "test_set": ["A,B", "A,B"],
                "predicted_set": ["A,B", "A,A"],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "classification.tsv"
            output = root / "visualizations"
            frame.to_csv(source, sep="\t", index=False)

            write_confusion_matrices(source, output)

            matrix = pd.read_csv(output / "confusion_matrix_rf.tsv", sep="\t", index_col=0)
            normalized = pd.read_csv(output / "confusion_matrix_rf_normalized.tsv", sep="\t", index_col=0)
            self.assertEqual(matrix.loc["B", "A"], 1)
            self.assertEqual(normalized.loc["B", "A"], 1.0)

    def test_confusion_matrices_for_feature_variants_do_not_overwrite_each_other(self):
        frame = pd.DataFrame(
            {
                "model": ["rf"],
                "n_components": ["all"],
                "thr_features": [True],
                "accuracy": [0.5],
                "test_set": ["A,B"],
                "predicted_set": ["A,A"],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "visualizations"
            all_features = root / "classification_all.tsv"
            no_counts = root / "classification_no_counts_features.tsv"
            frame.to_csv(all_features, sep="\t", index=False)
            frame.to_csv(no_counts, sep="\t", index=False)

            generated = write_confusion_matrices(all_features, output) + write_confusion_matrices(no_counts, output)

            self.assertEqual(
                {path.name for path in generated},
                {"confusion_matrix_all_rf.png", "confusion_matrix_no_counts_features_rf.png"},
            )
            self.assertTrue(all(path.is_file() for path in generated))

    def test_standalone_plots_support_short_rf_name_and_full_ablation_range(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "visualizations"
            classification = root / "classification_all.tsv"
            ablation = root / "ablation_ranking_all.tsv"
            pd.DataFrame(
                {
                    "model": ["rf", "rf"],
                    "n_components": [16, "all"],
                    "accuracy": [0.7, 0.8],
                }
            ).to_csv(classification, sep="\t", index=False)
            pd.DataFrame(
                {
                    "top_n": [1, 101, 265],
                    "accuracy": [0.5, 0.8, 0.75],
                    "selected_columns": [10, 1010, 2650],
                    "total_feature_families": [265, 265, 265],
                    "total_columns": [2650, 2650, 2650],
                }
            ).to_csv(ablation, sep="\t", index=False)

            classification_plot = write_classification_plot(classification, output)
            ablation_plot = write_ablation_plot(ablation, output)

            self.assertTrue(classification_plot.is_file())
            self.assertTrue(ablation_plot.is_file())

    def test_feature_reports_tolerate_constant_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_file = root / "data.tsv"
            rankings_file = root / "rankings.tsv"
            output = root / "visualizations"
            pd.DataFrame(
                {
                    "sampleName": [f"sample-{index}" for index in range(6)],
                    "varying": [1, 2, 3, 4, 5, 6],
                    "constant": [0, 0, 0, 0, 0, 0],
                    "nonfinite": [1, 2, np.nan, np.inf, 5, 6],
                    "label": ["A", "A", "A", "B", "B", "B"],
                }
            ).to_csv(data_file, sep="\t", index=False)
            pd.DataFrame({"feature": ["varying", "constant", "nonfinite"], "forest": [3, 2, 1]}).to_csv(
                rankings_file, sep="\t", index=False
            )

            correlations = write_feature_correlation(data_file, rankings_file, output)
            boxplots = write_feature_boxplots(data_file, rankings_file, output)

            self.assertEqual(len(correlations), 3)
            self.assertTrue(all(path.is_file() for path in correlations))
            self.assertEqual(len(boxplots), 3)
            self.assertTrue(all(path.is_file() for path in boxplots))


if __name__ == "__main__":
    unittest.main()
