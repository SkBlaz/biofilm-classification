#!/usr/bin/env python3
"""
Unit tests for inference.py core functions.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add src directory to path
sys.path.insert(0, "src")

from inference import (
    format_predictions,
    generate_shap_explanations,
    load_models,
    run_inference,
    total_abs_diff_per_feature,
    validate_cli_inputs,
)


class RecordingModel:
    """Small inference test double that records the final numpy array."""

    def __init__(self, predictions=None):
        self.predictions = predictions
        self.seen_X = None

    def predict(self, X):
        self.seen_X = np.asarray(X)
        if self.predictions is not None:
            return np.array(self.predictions[: len(X)])
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        self.seen_X = np.asarray(X)
        return np.tile(np.array([[0.75, 0.25]]), (len(X), 1))


class TestLoadModels(unittest.TestCase):
    """Test model loading functionality."""

    def setUp(self):
        """Create temporary directory with test models."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple test model
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(self.temp_dir, "test_model_model.joblib")
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {"feature_names": ["feature1", "feature2"], "target_mapping": {0: "class_a", 1: "class_b"}}
        metadata_path = os.path.join(self.temp_dir, "test_model_metadata.joblib")
        joblib.dump(metadata, metadata_path)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_valid_models(self):
        """Test loading valid models from directory."""
        models, metadata = load_models(self.temp_dir)

        # Should load one model
        self.assertEqual(len(models), 1)
        self.assertIn("test_model", models)

        # Should load metadata
        self.assertEqual(len(metadata), 1)
        self.assertIn("test_model", metadata)
        self.assertIn("feature_names", metadata["test_model"])
        self.assertEqual(metadata["test_model"]["feature_names"], ["feature1", "feature2"])

    def test_empty_directory(self):
        """Test loading from empty directory."""
        empty_dir = tempfile.mkdtemp()

        try:
            with self.assertRaises(ValueError) as context:
                load_models(empty_dir)
            self.assertIn("No model files found", str(context.exception))
        finally:
            import shutil

            shutil.rmtree(empty_dir)

    def test_model_without_metadata(self):
        """Test loading model without metadata file."""
        # Create a new model without metadata
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        model.fit(X_train, y_train)

        model_path = os.path.join(self.temp_dir, "model2_model.joblib")
        joblib.dump(model, model_path)

        # Should still load the model
        models, metadata = load_models(self.temp_dir)

        # Should have both models but only one metadata
        self.assertEqual(len(models), 2)
        self.assertIn("test_model", models)
        self.assertIn("model2", models)


class TestFormatPredictions(unittest.TestCase):
    """Test prediction formatting functionality."""

    def setUp(self):
        """Create test models and data."""
        # Create a simple test model
        self.model = RandomForestClassifier(n_estimators=2, random_state=42)
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        self.model.fit(X_train, y_train)

        # Create test data
        self.X = pd.DataFrame({"feature1": [1.5, 3.5], "feature2": [2.5, 4.5]}, index=["sample1", "sample2"])

        # Create models dict
        self.models = {"model1": self.model}

        # Create metadata dict
        self.metadata = {"model1": {"feature_names": ["feature1", "feature2"], "target_mapping": {0: "class_a", 1: "class_b"}}}

    def test_basic_predictions(self):
        """Test basic prediction formatting."""
        results = format_predictions(self.models, self.metadata, self.X)

        # Should return a DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)

        # Each result should have sample_name
        self.assertIn("sample_name", results.columns)
        self.assertIn("sample1", results["sample_name"].values)
        self.assertIn("sample2", results["sample_name"].values)

    def test_empty_models(self):
        """Test with empty models dictionary."""
        results = format_predictions({}, {}, self.X)

        # Should return DataFrame with just sample names
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        self.assertIn("sample_name", results.columns)

    def test_missing_features(self):
        """Test when data is missing some features."""
        # Create data with only one feature
        X_missing = pd.DataFrame({"feature1": [1.5, 3.5]}, index=["sample1", "sample2"])

        # Should handle missing features by adding zeros
        results = format_predictions(self.models, self.metadata, X_missing)

        # Should still produce predictions
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)

    def test_extra_features(self):
        """Test when data has extra features not in metadata."""
        # Create data with extra feature
        X_extra = pd.DataFrame({"feature1": [1.5, 3.5], "feature2": [2.5, 4.5], "feature3": [0.5, 1.5]}, index=["sample1", "sample2"])

        # Should handle extra features gracefully
        results = format_predictions(self.models, self.metadata, X_extra)

        # Should still produce predictions
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)

    def test_model_without_predict_proba(self):
        """Test with a model that doesn't have predict_proba method."""

        # Create a simple mock model without predict_proba
        class SimpleModel:
            def predict(self, X):
                return np.array([0] * len(X))

        simple_model = SimpleModel()
        models = {"simple": simple_model}
        metadata_simple = {"simple": {"feature_names": ["feature1", "feature2"]}}

        # Should handle models without predict_proba
        results = format_predictions(models, metadata_simple, self.X)

        # Should still produce predictions
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)


class TestValidateCliInputs(unittest.TestCase):
    """Test user-facing CLI input validation messages."""

    @staticmethod
    def _create_test_file(directory, filename):
        test_file = Path(directory, filename)
        test_file.touch()
        if not test_file.exists():
            raise RuntimeError(f"Failed to create test file: {test_file}")

    def test_missing_models_directory(self):
        with tempfile.TemporaryDirectory() as images_dir:
            with self.assertRaises(ValueError) as context:
                validate_cli_inputs("/does/not/exist", images_dir)
            self.assertIn("Models directory does not exist", str(context.exception))
            self.assertIn("Usage: python inference.py", str(context.exception))

    def test_missing_model_files(self):
        with tempfile.TemporaryDirectory() as models_dir, tempfile.TemporaryDirectory() as images_dir:
            self._create_test_file(images_dir, "sample.tif")
            with self.assertRaises(ValueError) as context:
                validate_cli_inputs(models_dir, images_dir)
            self.assertIn("No model files matching '*_model.joblib'", str(context.exception))

    def test_missing_tif_files(self):
        with tempfile.TemporaryDirectory() as models_dir, tempfile.TemporaryDirectory() as images_dir:
            self._create_test_file(models_dir, "demo_model.joblib")
            with self.assertRaises(ValueError) as context:
                validate_cli_inputs(models_dir, images_dir)
            self.assertIn("No '.tif' images were found", str(context.exception))

    def test_valid_inputs(self):
        with tempfile.TemporaryDirectory() as models_dir, tempfile.TemporaryDirectory() as images_dir:
            self._create_test_file(models_dir, "demo_model.joblib")
            self._create_test_file(images_dir, "sample.tif")
            validate_cli_inputs(models_dir, images_dir)


class TestRunInference(unittest.TestCase):
    """Test the file-based inference workflow."""

    def test_run_inference_aligns_cleans_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            features_file = os.path.join(temp_dir, "features.tsv")
            output_dir = os.path.join(temp_dir, "out")
            pd.DataFrame(
                {
                    "feature1": [np.inf, 2.0],
                    "extra_feature": [100.0, 200.0],
                    "label": ["ignored_a", "ignored_b"],
                },
                index=["sample_a", "sample_b"],
            ).to_csv(features_file, sep="\t")

            model = RecordingModel()
            metadata = {"demo": {"feature_names": ["feature1", "missing_feature"], "target_mapping": {0: "class_a", 1: "class_b"}}}

            with patch("inference.generate_shap_explanations") as shap_mock:
                num_models = run_inference({"demo": model}, metadata, features_file, output_dir)

            self.assertEqual(num_models, 1)
            shap_mock.assert_called_once()
            np.testing.assert_allclose(model.seen_X, np.array([[5.14, 0.0], [2.0, 0.0]]))

            predictions = pd.read_csv(os.path.join(output_dir, "demo_predictions.tsv"), sep="\t")
            probabilities = pd.read_csv(os.path.join(output_dir, "demo_probabilities.tsv"), sep="\t", index_col=0)
            processed = pd.read_csv(os.path.join(output_dir, "demo_features.tsv"), sep="\t", index_col=0)
            summary = pd.read_csv(os.path.join(output_dir, "inference_summary.tsv"), sep="\t")

            self.assertEqual(predictions["sample_name"].tolist(), ["sample_a", "sample_b"])
            self.assertEqual(predictions["prediction"].tolist(), ["class_a", "class_a"])
            self.assertEqual(probabilities.columns.tolist(), ["class_a", "class_b"])
            self.assertEqual(processed.columns.tolist(), ["feature1", "missing_feature"])
            self.assertEqual(summary.loc[0, "num_predictions"], 2)

    def test_run_inference_skips_model_missing_required_svd_transformer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            features_file = os.path.join(temp_dir, "features.tsv")
            output_dir = os.path.join(temp_dir, "out")
            pd.DataFrame({"feature1": [1.0], "feature2": [2.0]}, index=["sample_a"]).to_csv(features_file, sep="\t")

            model = RecordingModel()
            metadata = {"demo": {"feature_names": ["feature1", "feature2"], "n_components": 2}}

            with patch("inference.generate_shap_explanations") as shap_mock:
                num_models = run_inference({"demo": model}, metadata, features_file, output_dir)

            self.assertEqual(num_models, 0)
            shap_mock.assert_called_once()
            self.assertIsNone(model.seen_X)
            self.assertFalse(os.path.exists(os.path.join(output_dir, "inference_summary.tsv")))

    def test_run_inference_applies_threshold_indices_before_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            features_file = os.path.join(temp_dir, "features.tsv")
            output_dir = os.path.join(temp_dir, "out")
            pd.DataFrame(
                {
                    "feature1": [1.0, 10.0],
                    "feature2": [2.0, 20.0],
                    "feature3": [3.0, 30.0],
                },
                index=["sample_a", "sample_b"],
            ).to_csv(features_file, sep="\t")

            model = RecordingModel()
            metadata = {
                "demo": {
                    "feature_names": ["feature1", "feature2", "feature3"],
                    "thr_features": False,
                    "n_components": "all",
                    "thr_indices": [0, 2],
                }
            }

            with patch("inference.generate_shap_explanations"):
                num_models = run_inference({"demo": model}, metadata, features_file, output_dir)

            self.assertEqual(num_models, 1)
            np.testing.assert_allclose(model.seen_X, np.array([[1.0, 3.0], [10.0, 30.0]]))


class TestInferenceExplanations(unittest.TestCase):
    """Test helper behavior around inference explanations."""

    def test_generate_shap_explanations_returns_when_shap_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(sys.modules, {"shap": None}):
                generate_shap_explanations({}, {}, {}, temp_dir)

            self.assertFalse(os.path.exists(os.path.join(temp_dir, "explanations")))

    def test_total_abs_diff_per_feature_reports_tree_path_differences(self):
        rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        rf.fit(X, y)

        total, per_feature, details = total_abs_diff_per_feature(rf, np.array([1.0, 0.0]), feature_names=["a", "b"])

        self.assertGreaterEqual(total, 0.0)
        self.assertTrue(set(per_feature).issubset({"a", "b"}))
        self.assertEqual(len(details), len([detail for detail in details if detail["feature"] in {"a", "b"}]))
        self.assertAlmostEqual(total, sum(detail["abs_diff"] for detail in details))


if __name__ == "__main__":
    unittest.main()
