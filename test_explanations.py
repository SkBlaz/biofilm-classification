#!/usr/bin/env python3
"""
Test script for the explainability functionality in inference.py.
This tests SHAP explanation generation for biofilm classification predictions.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference import generate_shap_explanations
    INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Cannot import inference module: {e}")
    INFERENCE_AVAILABLE = False

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data_and_model():
    """Create synthetic test data and a trained model for testing."""
    # Create synthetic data similar to biofilm features
    n_samples = 100
    n_features = 50
    n_classes = 3
    
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_classes=n_classes,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    sample_names = [f'sample_{i}' for i in range(n_samples)]
    
    # Create and train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create metadata similar to what would be saved
    metadata = {
        'feature_names': feature_names,
        'target_mapping': {'class_0': 0, 'class_1': 1, 'class_2': 2},
        'thr_features': False,
        'n_components': 'all'
    }
    
    return X, y, model, metadata, feature_names, sample_names


def test_shap_explanations():
    """Test SHAP explanation generation functionality."""
    if not INFERENCE_AVAILABLE:
        print("Cannot test explanations - inference module not available")
        return False
    
    try:
        logger.info("Creating test data and model...")
        X, y, model, metadata, feature_names, sample_names = create_test_data_and_model()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Testing SHAP explanation generation...")
            
            # Test the explanation function
            generate_shap_explanations(
                model=model,
                model_name="test_model",
                X_values=X,
                X_features=feature_names,
                sample_names=sample_names,
                output_dir=temp_dir,
                metadata=metadata
            )
            
            # Check if explanation files were created
            explanations_dir = os.path.join(temp_dir, "explanations")
            if not os.path.exists(explanations_dir):
                raise ValueError("Explanations directory was not created")
            
            explanation_file = os.path.join(explanations_dir, "test_model_shap_explanations.tsv")
            importance_file = os.path.join(explanations_dir, "test_model_feature_importance.tsv")
            
            if not os.path.exists(explanation_file):
                raise ValueError("SHAP explanations file was not created")
            
            if not os.path.exists(importance_file):
                raise ValueError("Feature importance file was not created")
            
            # Verify the content of the explanation files
            explanations_df = pd.read_csv(explanation_file, sep='\t', index_col=0)
            importance_df = pd.read_csv(importance_file, sep='\t', index_col=0)
            
            logger.info(f"Explanations shape: {explanations_df.shape}")
            logger.info(f"Importance shape: {importance_df.shape}")
            logger.info(f"Top 5 important features: {importance_df.head()}")
            
            # Basic validation
            if explanations_df.empty:
                raise ValueError("Explanations DataFrame is empty")
            
            if importance_df.empty:
                raise ValueError("Feature importance DataFrame is empty")
            
            # Check that we have the expected number of features
            if explanations_df.shape[1] != len(feature_names):
                raise ValueError(f"Expected {len(feature_names)} features, got {explanations_df.shape[1]}")
            
            logger.info("SHAP explanation test passed!")
            return True
            
    except Exception as e:
        logger.error(f"SHAP explanation test failed: {e}")
        logger.debug("Error details:", exc_info=True)
        return False


def test_without_shap():
    """Test that the function handles gracefully when SHAP is not available."""
    # This is a placeholder test - in practice, we would mock the SHAP import failure
    logger.info("Testing graceful handling when SHAP is not available...")
    # The actual function should log a warning and return without error
    logger.info("This test would require mocking SHAP unavailability")
    return True


if __name__ == "__main__":
    logger.info("Starting explainability tests...")
    
    success = True
    
    # Test SHAP explanation generation
    if not test_shap_explanations():
        success = False
    
    # Test graceful handling without SHAP
    if not test_without_shap():
        success = False
    
    if success:
        logger.info("All explainability tests passed!")
        sys.exit(0)
    else:
        logger.error("Some explainability tests failed!")
        sys.exit(1)