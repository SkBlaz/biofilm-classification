#!/usr/bin/env python3
"""
Unit tests for inference.py core functions.
"""

import os
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add src directory to path
sys.path.insert(0, 'src')

from inference import format_predictions, load_models


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
        metadata = {
            'feature_names': ['feature1', 'feature2'],
            'target_mapping': {0: 'class_a', 1: 'class_b'}
        }
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
        self.assertIn('test_model', models)
        
        # Should load metadata
        self.assertEqual(len(metadata), 1)
        self.assertIn('test_model', metadata)
        self.assertIn('feature_names', metadata['test_model'])
        self.assertEqual(metadata['test_model']['feature_names'], ['feature1', 'feature2'])
    
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
        self.assertIn('test_model', models)
        self.assertIn('model2', models)


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
        self.X = pd.DataFrame({
            'feature1': [1.5, 3.5],
            'feature2': [2.5, 4.5]
        }, index=['sample1', 'sample2'])
        
        # Create models dict
        self.models = {'model1': self.model}
        
        # Create metadata dict
        self.metadata = {
            'model1': {
                'feature_names': ['feature1', 'feature2'],
                'target_mapping': {0: 'class_a', 1: 'class_b'}
            }
        }
    
    def test_basic_predictions(self):
        """Test basic prediction formatting."""
        results = format_predictions(self.models, self.metadata, self.X)
        
        # Should return a DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        
        # Each result should have sample_name
        self.assertIn('sample_name', results.columns)
        self.assertIn('sample1', results['sample_name'].values)
        self.assertIn('sample2', results['sample_name'].values)
    
    def test_empty_models(self):
        """Test with empty models dictionary."""
        results = format_predictions({}, {}, self.X)
        
        # Should return DataFrame with just sample names
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
        self.assertIn('sample_name', results.columns)
    
    def test_missing_features(self):
        """Test when data is missing some features."""
        # Create data with only one feature
        X_missing = pd.DataFrame({
            'feature1': [1.5, 3.5]
        }, index=['sample1', 'sample2'])
        
        # Should handle missing features by adding zeros
        results = format_predictions(self.models, self.metadata, X_missing)
        
        # Should still produce predictions
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)
    
    def test_extra_features(self):
        """Test when data has extra features not in metadata."""
        # Create data with extra feature
        X_extra = pd.DataFrame({
            'feature1': [1.5, 3.5],
            'feature2': [2.5, 4.5],
            'feature3': [0.5, 1.5]
        }, index=['sample1', 'sample2'])
        
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
        models = {'simple': simple_model}
        metadata_simple = {
            'simple': {
                'feature_names': ['feature1', 'feature2']
            }
        }
        
        # Should handle models without predict_proba
        results = format_predictions(models, metadata_simple, self.X)
        
        # Should still produce predictions
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 2)


if __name__ == '__main__':
    unittest.main()
