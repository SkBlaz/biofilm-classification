#!/usr/bin/env python3
"""
Integration test for inference with explainability features.
This creates synthetic data and models to test the complete workflow.
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

from inference import run_inference

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def create_test_models_and_data(models_dir, features_file):
    """Create test models and feature data for integration testing."""
    logger.info("Creating test models and data...")
    
    # Create synthetic data similar to biofilm features
    n_samples = 50
    n_features = 100
    n_classes = 3
    
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_classes=n_classes,
        n_informative=30,
        n_redundant=20,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'biofilm_feature_{i}' for i in range(n_features)]
    sample_names = [f'biofilm_sample_{i}' for i in range(n_samples)]
    
    # Create features DataFrame
    features_df = pd.DataFrame(X, columns=feature_names, index=sample_names)
    
    # Save features file
    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    features_df.to_csv(features_file, sep='\t')
    logger.info(f"Created features file: {features_file}")
    
    # Create and save test models
    os.makedirs(models_dir, exist_ok=True)
    
    # Model 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
    rf_model.fit(X, y)
    
    rf_metadata = {
        'feature_names': feature_names,
        'target_mapping': {'strain_A': 0, 'strain_B': 1, 'strain_C': 2},
        'thr_features': False,
        'n_components': 'all',
        'model_type': 'RandomForest'
    }
    
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.joblib'))
    joblib.dump(rf_metadata, os.path.join(models_dir, 'random_forest_metadata.joblib'))
    
    # Model 2: Random Forest with different parameters
    rf2_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=123)
    rf2_model.fit(X, y)
    
    rf2_metadata = {
        'feature_names': feature_names,
        'target_mapping': {'strain_A': 0, 'strain_B': 1, 'strain_C': 2},
        'thr_features': False,
        'n_components': 'all',
        'model_type': 'RandomForest_v2'
    }
    
    joblib.dump(rf2_model, os.path.join(models_dir, 'random_forest_v2_model.joblib'))
    joblib.dump(rf2_metadata, os.path.join(models_dir, 'random_forest_v2_metadata.joblib'))
    
    logger.info("Created test models in %s", models_dir)
    return len(feature_names), len(sample_names)


def test_inference_integration():
    """Test the complete inference workflow with explainability."""
    logger.info("Running inference integration test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, "models")
        features_file = os.path.join(temp_dir, "test_features.tsv")
        output_dir = os.path.join(temp_dir, "results")
        
        # Create test data and models
        n_features, n_samples = create_test_models_and_data(models_dir, features_file)
        
        # Test 1: Regular inference (without explanations)
        logger.info("Testing inference without explanations...")
        from inference import load_models
        models, metadata = load_models(models_dir)
        
        num_models = run_inference(models, metadata, features_file, output_dir, enable_explanations=False)
        
        if num_models == 0:
            raise ValueError("No models produced predictions")
        
        # Check basic output files
        prediction_files = [f for f in os.listdir(output_dir) if f.endswith('_predictions.tsv')]
        if len(prediction_files) == 0:
            raise ValueError("No prediction files were created")
        
        logger.info(f"Successfully created {len(prediction_files)} prediction files")
        
        # Test 2: Inference with explanations
        logger.info("Testing inference with explanations...")
        output_dir_with_explanations = os.path.join(temp_dir, "results_with_explanations")
        
        num_models = run_inference(models, metadata, features_file, output_dir_with_explanations, enable_explanations=True)
        
        if num_models == 0:
            raise ValueError("No models produced predictions with explanations")
        
        # Check explanation files
        explanations_dir = os.path.join(output_dir_with_explanations, "explanations")
        if not os.path.exists(explanations_dir):
            raise ValueError("Explanations directory was not created")
        
        explanation_files = [f for f in os.listdir(explanations_dir) if f.endswith('_shap_explanations.tsv')]
        importance_files = [f for f in os.listdir(explanations_dir) if f.endswith('_feature_importance.tsv')]
        
        if len(explanation_files) == 0:
            raise ValueError("No SHAP explanation files were created")
        
        if len(importance_files) == 0:
            raise ValueError("No feature importance files were created")
        
        logger.info(f"Successfully created {len(explanation_files)} explanation files")
        logger.info(f"Successfully created {len(importance_files)} feature importance files")
        
        # Validate explanation content
        for explanation_file in explanation_files:
            explanation_path = os.path.join(explanations_dir, explanation_file)
            explanations_df = pd.read_csv(explanation_path, sep='\t', index_col=0)
            
            if explanations_df.empty:
                raise ValueError(f"Explanations file {explanation_file} is empty")
            
            logger.info(f"Explanation file {explanation_file}: {explanations_df.shape}")
        
        # Validate feature importance content
        for importance_file in importance_files:
            importance_path = os.path.join(explanations_dir, importance_file)
            importance_df = pd.read_csv(importance_path, sep='\t', index_col=0)
            
            if importance_df.empty:
                raise ValueError(f"Feature importance file {importance_file} is empty")
            
            logger.info(f"Feature importance file {importance_file}: {importance_df.shape}")
            logger.info(f"Top 3 important features: {importance_df.head(3).index.tolist()}")
        
        logger.info("Integration test completed successfully!")
        return True


if __name__ == "__main__":
    logger.info("Starting integration test for inference with explainability...")
    
    try:
        success = test_inference_integration()
        if success:
            logger.info("All integration tests passed!")
            sys.exit(0)
        else:
            logger.error("Integration test failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Integration test failed with error: {e}")
        logger.debug("Error details:", exc_info=True)
        sys.exit(1)