#!/usr/bin/env python3
"""
Unit tests for adaptive CV selection in feature_ranking_lite.py.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add src directory to path
sys.path.insert(0, "src")

import feature_ranking_lite
from cv_planning import assess_grouped_cv
from feature_ranking_lite import (
    benchmark_configurations,
    compute_ablation_scores,
    configured_parallelism,
    convert_to_one_hot,
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

    def test_rejects_singleton_class_instead_of_reporting_invalid_scores(self):
        y = np.array([0, 0, 1])
        with self.assertRaisesRegex(ValueError, "each class requires at least 2 samples"):
            get_adaptive_cv(y, max_splits=5)

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

    def test_nested_plan_rejects_two_class_pure_groups_per_class(self):
        labels = np.array(["A", "A", "B", "B"])
        groups = np.array(["A1", "A2", "B1", "B2"])

        report = assess_grouped_cv(labels, groups, require_nested_cv=True)

        self.assertFalse(report["ok"])
        self.assertIn("nested model tuning", report["errors"][0])


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

    def test_partial_cache_is_pinned_to_published_main_protocol(self):
        date_path = partial_evaluation_path("partial", "all", 0, 16, True, "rf", 0, "date")
        well_path = partial_evaluation_path("partial", "all", 0, 16, True, "rf", 0, "well")

        self.assertEqual(date_path, well_path)
        self.assertIn("_reppublished-main_", date_path)
        self.assertIn("/v6_", date_path)

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

    def test_benchmark_configurations_preserve_published_threshold_flag_repetitions(self):
        configurations = benchmark_configurations([16, 32, "all"], feature_count=100, threshold_feature_count=20)

        self.assertEqual(
            configurations,
            [
                (16, True, "svd_16"),
                (16, False, "svd_16"),
                (32, True, "svd_32"),
                (32, False, "svd_32"),
                ("all", True, "all_columns"),
                ("all", False, "threshold_only"),
            ],
        )

    def test_classification_ignores_replication_groups_for_published_protocol(self):
        rng = np.random.default_rng(7)
        labels = pd.Series(["A"] * 9 + ["B"] * 9)
        features = pd.DataFrame(
            rng.normal(size=(len(labels), 4)),
            index=[f"sample-{index}" for index in range(len(labels))],
            columns=[f"feature-{index}" for index in range(4)],
        )
        runtime = {"n_iter": 1, "repetitions": 1, "n_components": ["all"]}

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(feature_ranking_lite, "get_benchmark_runtime_config", return_value=runtime),
        ):
            data_path = Path(temporary_directory) / "datafile.tsv"
            feature_ranking_lite.do_classification_simple(
                features,
                labels,
                str(data_path),
                learner="dummy",
                replication_unit="date",
            )
            classification = pd.read_csv(Path(temporary_directory) / "classification_all.tsv", sep="\t")

        self.assertEqual(len(classification), 6)
        self.assertEqual(set(classification["component_setting"]), {"all_columns"})

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


class TestConvertToOneHot(unittest.TestCase):
    """Keep the categorical-ranking coverage added on the default branch."""

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


class TestAblationEvaluation(unittest.TestCase):
    def test_ablation_preserves_published_column_counts_and_ignores_replication_groups(self):
        rng = np.random.default_rng(42)
        labels = pd.Series(["L1"] * 12 + ["L2"] * 12)
        features = pd.DataFrame(rng.normal(size=(len(labels), 42)), index=[f"sample-{index}" for index in range(len(labels))])

        np.random.seed(123)
        first = compute_ablation_scores(features, labels, replication_unit="date")
        np.random.seed(123)
        second = compute_ablation_scores(features, labels, replication_unit="date")

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.columns.tolist(), ["top_n", "accuracy"])
        self.assertEqual(first["top_n"].tolist(), [1, 21, 41])


if __name__ == "__main__":
    unittest.main()
