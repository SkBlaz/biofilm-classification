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

import feature_ranking_lite
from feature_ranking_lite import (
    compute_ablation_scores,
    configured_parallelism,
    emit_pipeline_progress,
    get_adaptive_cv,
    get_benchmark_runtime_config,
    partial_evaluation_path,
    prepare_fold_features,
    search_fit_count,
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

    def test_grouped_cv_never_splits_a_replication_group(self):
        y = np.tile(np.array([0, 1, 2]), 6)
        groups = np.repeat(np.arange(6), 3)

        cv, n_splits, _min_class_count, strategy, _reason = get_adaptive_cv(y, max_splits=5, groups=groups)

        self.assertEqual(n_splits, 5)
        self.assertIn(strategy, {"stratified-group", "group"})
        for train_indices, test_indices in cv.split(np.zeros((len(y), 1)), y, groups):
            self.assertTrue(set(groups[train_indices]).isdisjoint(groups[test_indices]))


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

    def test_configured_parallelism_honors_requested_worker_limit(self):
        with patch.object(feature_ranking_lite, "PARALLELISM", 3):
            self.assertEqual(configured_parallelism(), 3)

    def test_partial_cache_separates_replication_strategies(self):
        date_path = partial_evaluation_path("partial", "all", 0, 16, True, "rf", 0, "date")
        well_path = partial_evaluation_path("partial", "all", 0, 16, True, "rf", 0, "well")

        self.assertNotEqual(date_path, well_path)
        self.assertIn("_repdate_", date_path)
        self.assertIn("/v2_", date_path)

    def test_fold_preprocessing_is_reusable_across_learners(self):
        x_train = np.arange(60, dtype=float).reshape(10, 6)
        x_test = np.arange(24, dtype=float).reshape(4, 6)

        transformed_train, transformed_test, transformer, effective_components = prepare_fold_features(
            x_train,
            x_test,
            3,
            True,
            np.array([1, 4]),
        )

        self.assertEqual(transformed_train.shape, (10, 3))
        self.assertEqual(transformed_test.shape, (4, 3))
        self.assertEqual(effective_components, 3)
        self.assertIsNotNone(transformer)

    def test_threshold_only_fold_reports_actual_feature_count(self):
        x_train = np.arange(60, dtype=float).reshape(10, 6)
        x_test = np.arange(24, dtype=float).reshape(4, 6)

        transformed_train, transformed_test, transformer, effective_components = prepare_fold_features(
            x_train,
            x_test,
            "all",
            False,
            np.array([1, 4]),
        )

        self.assertEqual(transformed_train.shape, (10, 2))
        self.assertEqual(transformed_test.shape, (4, 2))
        self.assertEqual(effective_components, 2)
        self.assertIsNone(transformer)

    def test_search_fit_count_includes_candidates_and_folds(self):
        random_search = feature_ranking_lite.RandomizedSearchCV(
            feature_ranking_lite.DummyClassifier(), {"strategy": ["most_frequent", "prior"]}, n_iter=2
        )
        grid_search = feature_ranking_lite.GridSearchCV(feature_ranking_lite.DummyClassifier(), {"strategy": ["most_frequent", "prior"]})

        self.assertEqual(search_fit_count(random_search, 3), 6)
        self.assertEqual(search_fit_count(grid_search, 3), 6)

    def test_progress_event_is_machine_readable(self):
        with self.assertLogs(feature_ranking_lite.logger, level="INFO") as captured:
            emit_pipeline_progress("Benchmark", 2, 5, 3, 10, "Evaluation 4 of 10", sub_total=6)

        self.assertIn('MICROICS_PROGRESS {"phase":"Benchmark"', captured.output[0])
        self.assertIn('"sub_total":6', captured.output[0])


class TestAblationEvaluation(unittest.TestCase):
    def test_ablation_is_deterministic_and_honors_replication_groups(self):
        rng = np.random.default_rng(42)
        sample_names = [
            f"{date}--s--Lm--st--{label}--p--C01--pos001--tm--24--ch--Syto9--z--21"
            for date in ("01012026", "02012026", "03012026", "04012026")
            for label in ("L1", "L2")
            for _ in range(2)
        ]
        labels = pd.Series([label for _date in range(4) for label in ("L1", "L2") for _ in range(2)], index=sample_names)
        features = pd.DataFrame(rng.normal(size=(len(sample_names), 22)), index=sample_names)

        with patch.object(feature_ranking_lite, "PARALLELISM", 1):
            first = compute_ablation_scores(features, labels, replication_unit="date")
            second = compute_ablation_scores(features, labels, replication_unit="date")

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["top_n"].tolist(), [1, 21])


if __name__ == "__main__":
    unittest.main()
