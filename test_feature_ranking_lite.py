#!/usr/bin/env python3
"""
Unit tests for adaptive CV selection in feature_ranking_lite.py.
"""

import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# Add src directory to path
sys.path.insert(0, "src")

from feature_ranking_lite import (
    compute_rankings,
    convert_to_one_hot,
    get_adaptive_cv,
    get_benchmark_runtime_config,
    load_data,
    name_manipulator,
    name_manipulator_date,
)


class TestAdaptiveCV(unittest.TestCase):
    """Test adaptive cross-validation setup."""

    def test_uses_class_count_limited_stratified_kfold(self):
        """Should reduce n_splits to minimum class count when needed."""
        y = np.array([0, 0, 1, 1, 2, 2])
        cv, n_splits, min_class_count, strategy, reason = get_adaptive_cv(y, max_splits=5)

        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(n_splits, 2)
        self.assertEqual(cv.n_splits, 2)
        self.assertEqual(min_class_count, 2)
        self.assertEqual(strategy, "stratified")
        self.assertIn("StratifiedKFold enabled", reason)

    def test_falls_back_to_kfold_for_singleton_class(self):
        """Should fall back to KFold when stratification is impossible."""
        y = np.array([0, 0, 1])
        cv, n_splits, min_class_count, strategy, reason = get_adaptive_cv(y, max_splits=5)

        self.assertIsInstance(cv, KFold)
        self.assertEqual(n_splits, 2)
        self.assertEqual(cv.n_splits, n_splits)
        self.assertEqual(min_class_count, 1)
        self.assertEqual(strategy, "kfold")
        self.assertIn("Falling back to KFold", reason)

    def test_empty_target_raises_value_error(self):
        """Should reject empty target arrays."""
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            get_adaptive_cv(np.array([]), max_splits=5)

    def test_single_sample_raises_value_error(self):
        """Should reject a single-sample target array."""
        with self.assertRaisesRegex(ValueError, "at least 2 samples"):
            get_adaptive_cv(np.array([1]), max_splits=5)


class TestBenchmarkRuntimeConfig(unittest.TestCase):
    """Test benchmark runtime tuning for CI and local runs."""

    def test_uses_fast_runtime_config_in_ci(self):
        with patch.dict("os.environ", {"CI": "true"}, clear=False):
            config = get_benchmark_runtime_config()

        self.assertEqual(config["n_iter"], 2)
        self.assertEqual(config["repetitions"], 1)
        self.assertEqual(config["n_components"], ["all"])

    def test_uses_default_runtime_config_outside_ci(self):
        with patch.dict("os.environ", {}, clear=True):
            config = get_benchmark_runtime_config()

        self.assertEqual(config["n_iter"], 10)
        self.assertEqual(config["repetitions"], 3)
        self.assertEqual(config["n_components"], [16, 32, 64, 128, 256, 512, "all"])


class TestFeatureRankingDataUtilities(unittest.TestCase):
    """Test data preparation utilities used by ranking and benchmark code."""

    def test_convert_to_one_hot_includes_low_cardinality_categoricals(self):
        x_data = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "color": ["red", "blue", "red"],
                "unique_id": ["a", "b", "c"],
            }
        )

        converted, feature_groups = convert_to_one_hot(x_data)

        self.assertIn("numeric", converted.columns)
        self.assertIn("color_blue", converted.columns)
        self.assertIn("color_red", converted.columns)
        self.assertNotIn("unique_id_a", converted.columns)
        self.assertEqual(feature_groups["color"], ["color_blue", "color_red"])
        self.assertNotIn("unique_id", feature_groups)

    def test_load_data_replaces_nan_and_inf_and_writes_intermediary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = f"{temp_dir}/data.tsv"
            pd.DataFrame(
                {
                    "sampleName": [
                        "date--s--Lm--st--L628--p--C03--pos001",
                        "date--s--Lm--st--L628--p--C03--pos002",
                        "date--s--Lm--st--L628--p--C03--pos003",
                    ],
                    "numeric": [np.inf, 1.0, np.nan],
                    "label": ["L628", None, "L628"],
                }
            ).to_csv(data_path, sep="\t", index=False)

            loaded = load_data(data_path)

            self.assertAlmostEqual(loaded.loc["date--s--Lm--st--L628--p--C03--pos001", "numeric"], 4.14)
            self.assertEqual(loaded.loc["date--s--Lm--st--L628--p--C03--pos003", "numeric"], -666)
            self.assertEqual(loaded.loc["date--s--Lm--st--L628--p--C03--pos002", "label"], "missing")
            self.assertTrue(pd.io.common.file_exists(data_path + "intermediary_aggregated.tsv"))

    def test_load_data_handles_all_nonfinite_numeric_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = f"{temp_dir}/data.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["date--s--Lm--st--L628--p--C03--pos001", "date--s--Lm--st--L628--p--C03--pos002"],
                    "numeric": [np.inf, -np.inf],
                    "label": ["L628", "L628"],
                }
            ).to_csv(data_path, sep="\t", index=False)

            loaded = load_data(data_path)

            self.assertEqual(loaded["numeric"].tolist(), [-666.0, -666.0])

    def test_compute_rankings_skip_returns_feature_target_split(self):
        data = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0], "label": ["a", "b"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            x_data, y_data, scores = compute_rankings(data, f"{temp_dir}/data.tsv", skip=True)

        self.assertEqual(x_data.columns.tolist(), ["feature1", "feature2"])
        self.assertEqual(y_data.tolist(), ["a", "b"])
        self.assertIsNone(scores)

    def test_compute_rankings_loads_cached_ranking_file(self):
        data = pd.DataFrame({"feature1": [1.0, 2.0], "label": ["a", "b"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            ranking_path = f"{temp_dir}/rankings_label.tsv"
            pd.DataFrame({"feature": ["feature1"], "RandomForest(n=200, p=1.0)": [0.9]}).to_csv(ranking_path, sep="\t", index=False)

            _, _, scores = compute_rankings(data, f"{temp_dir}/data.tsv", target_col="label")

        self.assertEqual(scores["feature"].tolist(), ["feature1"])

    def test_name_manipulators_extract_expected_group_keys(self):
        sample = "13042023--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21"

        self.assertEqual(name_manipulator(sample), "L628")
        self.assertEqual(name_manipulator_date(sample), ("13042023", "L628"))


if __name__ == "__main__":
    unittest.main()
