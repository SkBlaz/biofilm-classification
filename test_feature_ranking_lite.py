#!/usr/bin/env python3
"""
Unit tests for adaptive CV selection in feature_ranking_lite.py.
"""

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# Add src directory to path
sys.path.insert(0, "src")

from feature_ranking_lite import convert_to_one_hot, get_adaptive_cv, get_benchmark_runtime_config


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


class TestConvertToOneHot(unittest.TestCase):
    """Test categorical preprocessing for feature ranking."""

    def test_expands_low_cardinality_categorical_columns(self):
        data = pd.DataFrame({"numeric": [1, 2, 3], "strain": ["A", "A", "B"]})

        converted, feature_groups = convert_to_one_hot(data)

        self.assertListEqual(list(converted.columns), ["numeric", "strain_A", "strain_B"])
        self.assertDictEqual(feature_groups, {"numeric": ["numeric"], "strain": ["strain_A", "strain_B"]})

    def test_skips_high_cardinality_categorical_columns(self):
        data = pd.DataFrame({"numeric": [1, 2, 3], "sample_id": ["A", "B", "C"]})

        converted, feature_groups = convert_to_one_hot(data)

        self.assertListEqual(list(converted.columns), ["numeric"])
        self.assertDictEqual(feature_groups, {"numeric": ["numeric"]})


if __name__ == "__main__":
    unittest.main()
