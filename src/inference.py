#!/usr/bin/env python3
"""
Inference mode for biofilm classification.
Loads pre-trained models and makes predictions on new .tif images.
"""

import argparse
import os
import sys
import glob
import logging
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_models(models_dir):
    """Load all trained models from the models directory."""
    models = {}
    model_files = glob.glob(os.path.join(models_dir, "*.pkl")) + glob.glob(os.path.join(models_dir, "*.joblib"))
    
    if not model_files:
        raise ValueError(f"No model files found in {models_dir}. Expected .pkl or .joblib files.")
    
    for model_file in model_files:
        model_name = Path(model_file).stem
        try:
            if model_file.endswith('.pkl'):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            elif model_file.endswith('.joblib'):
                model = joblib.load(model_file)
            
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load model {model_file}: {e}")
    
    if not models:
        raise ValueError(f"Failed to load any models from {models_dir}")
    
    return models

def generate_features_for_inference(images_dir, output_dir):
    """Generate features for input images using the existing feature generator."""
    import subprocess
    import tempfile
    
    # Create temporary directories for feature generation
    temp_feature_dir = os.path.join(output_dir, "temp_features")
    temp_raw_dir = os.path.join(output_dir, "temp_raw") 
    temp_analysis_dir = os.path.join(output_dir, "temp_analysis")
    
    os.makedirs(temp_feature_dir, exist_ok=True)
    os.makedirs(temp_raw_dir, exist_ok=True)
    os.makedirs(temp_analysis_dir, exist_ok=True)
    
    # Find all .tif files
    tif_files = glob.glob(os.path.join(images_dir, "*.tif"))
    if not tif_files:
        raise ValueError(f"No .tif files found in {images_dir}")
    
    logger.info(f"Found {len(tif_files)} .tif files to process")
    
    # Generate features for each image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for tif_file in tif_files:
        cmd = f"python {os.path.join(script_dir, 'feature_generator.py')} --outfolder {temp_feature_dir} --file {tif_file}"
        logger.info(f"Generating features for {tif_file}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Feature generation failed for {tif_file}: {result.stderr}")
            continue
    
    # Create joint dataset
    logger.info("Creating joint dataset from generated features")
    result = subprocess.run(f"python {os.path.join(script_dir, 'create_joint_df.py')} {temp_feature_dir} {temp_raw_dir}", 
                          shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Joint dataset creation failed: {result.stderr}")
        return None
    
    # Compute aggregated features  
    logger.info("Computing aggregated features")
    result = subprocess.run(f"python {os.path.join(script_dir, 'analysis.py')} {temp_raw_dir} {temp_analysis_dir}",
                          shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Feature analysis failed: {result.stderr}")
        return None
    
    # Create final dataset
    dataset_file = os.path.join(output_dir, "inference_data.tsv")
    logger.info("Creating final dataset")
    result = subprocess.run(f"python {os.path.join(script_dir, 'create_final_df_from_results.py')} {temp_analysis_dir} {dataset_file}",
                          shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Final dataset creation failed: {result.stderr}")
        return None
    
    # Clean up temporary directories
    import shutil
    try:
        shutil.rmtree(temp_feature_dir)
        shutil.rmtree(temp_raw_dir) 
        shutil.rmtree(temp_analysis_dir)
    except:
        pass
    
    return dataset_file

def make_predictions(models, data_file, output_dir):
    """Make predictions using loaded models."""
    # Load the feature data
    try:
        data = pd.read_csv(data_file, sep='\t', index_col=0)
        logger.info(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data from {data_file}: {e}")
        return
    
    # Prepare features (remove any label columns if present)
    feature_cols = [col for col in data.columns if col not in ['label', 'strain', 'date']]
    X = data[feature_cols]
    
    # Make predictions with each model
    predictions = {}
    prediction_probabilities = {}
    
    for model_name, model in models.items():
        try:
            logger.info(f"Making predictions with model: {model_name}")
            
            # Handle missing features gracefully
            model_features = getattr(model, 'feature_names_in_', None)
            if model_features is not None:
                # Use only features the model was trained on
                available_features = [f for f in model_features if f in X.columns]
                if len(available_features) != len(model_features):
                    logger.warning(f"Model {model_name} expects {len(model_features)} features, "
                                 f"but only {len(available_features)} are available")
                X_model = X[available_features]
            else:
                X_model = X
            
            # Make predictions
            y_pred = model.predict(X_model)
            predictions[model_name] = y_pred
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_model)
                prediction_probabilities[model_name] = y_proba
                
        except Exception as e:
            logger.error(f"Failed to make predictions with model {model_name}: {e}")
            continue
    
    if not predictions:
        logger.error("No predictions were made successfully")
        return
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions, index=data.index)
    predictions_file = os.path.join(output_dir, "predictions.tsv")
    predictions_df.to_csv(predictions_file, sep='\t')
    logger.info(f"Saved predictions to: {predictions_file}")
    
    # Save prediction probabilities if available
    if prediction_probabilities:
        for model_name, proba in prediction_probabilities.items():
            proba_df = pd.DataFrame(proba, index=data.index)
            proba_file = os.path.join(output_dir, f"probabilities_{model_name}.tsv")
            proba_df.to_csv(proba_file, sep='\t')
            logger.info(f"Saved probabilities for {model_name} to: {proba_file}")
    
    # Create summary
    summary = {
        'input_images': len(data),
        'models_used': list(predictions.keys()),
        'predictions_file': predictions_file,
        'feature_count': len(feature_cols)
    }
    
    summary_file = os.path.join(output_dir, "inference_summary.txt")
    with open(summary_file, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Inference completed. Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on new .tif images using pre-trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("models_dir", help="Directory containing pre-trained model files (.pkl or .joblib)")
    parser.add_argument("images_dir", help="Directory containing input .tif images")
    parser.add_argument("output_dir", help="Directory to save prediction results")
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.isdir(args.models_dir):
        logger.error(f"Models directory does not exist: {args.models_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.images_dir):
        logger.error(f"Images directory does not exist: {args.images_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load models
        logger.info(f"Loading models from: {args.models_dir}")
        models = load_models(args.models_dir)
        
        # Generate features for input images
        logger.info(f"Generating features for images in: {args.images_dir}")
        data_file = generate_features_for_inference(args.images_dir, args.output_dir)
        
        if data_file is None:
            logger.error("Feature generation failed")
            sys.exit(1)
        
        # Make predictions
        logger.info("Making predictions...")
        make_predictions(models, data_file, args.output_dir)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()