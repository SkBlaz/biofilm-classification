#!/usr/bin/env python3
"""
Inference script for biofilm classification models.

This script loads pre-trained models and runs inference on new .tif images.
"""

import os
import sys
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
import logging
import subprocess
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def load_models(models_dir):
    """Load all trained models from the specified directory."""
    models = {}
    metadata = {}
    
    model_files = glob.glob(os.path.join(models_dir, "*_model.joblib"))
    if not model_files:
        raise ValueError(f"No model files found in {models_dir}")
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace("_model.joblib", "")
        metadata_file = model_file.replace("_model.joblib", "_metadata.joblib")
        
        try:
            # Load model
            model = joblib.load(model_file)
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
            
            # Load metadata
            if os.path.exists(metadata_file):
                meta = joblib.load(metadata_file)
                metadata[model_name] = meta
                logger.info(f"Loaded metadata for: {model_name}")
            else:
                logger.warning(f"No metadata file found for {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded successfully")
        
    return models, metadata


def format_predictions(models, metadata, X):
    """Format predictions from multiple models into a single DataFrame."""
    results_list = []
    
    for sample_name in X.index:
        result = {'sample_name': sample_name}
        
        for model_name, model in models.items():
            try:
                meta = metadata[model_name]
                
                # Get single sample
                sample_data = X.loc[[sample_name]]
                
                # Align features 
                feature_names = meta.get('feature_names', [])
                if feature_names:
                    missing_features = set(feature_names) - set(sample_data.columns)
                    if missing_features:
                        logger.warning(f"Missing features for {model_name}: {missing_features}")
                        for feat in missing_features:
                            sample_data[feat] = 0
                    
                    sample_data = sample_data[feature_names]
                
                # Make prediction
                prediction = model.predict(sample_data)[0]
                
                # Get confidence (max probability)
                confidence = 0.0
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(sample_data)[0]
                    confidence = max(probabilities)
                
                # Convert prediction back to original label if mapping available
                if 'target_mapping' in meta:
                    target_mapping = meta['target_mapping']
                    code_to_label = {v: k for k, v in target_mapping.items()}
                    prediction = code_to_label.get(prediction, prediction)
                
                result[f'{model_name}_prediction'] = prediction
                result[f'{model_name}_confidence'] = confidence
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name} for {sample_name}: {e}")
                result[f'{model_name}_prediction'] = 'ERROR'
                result[f'{model_name}_confidence'] = 0.0
        
        results_list.append(result)
    
    return pd.DataFrame(results_list)


def generate_features_for_images(images_dir, temp_dir):
    """Generate features for all .tif images in the given directory."""
    logger.info(f"Generating features for images in {images_dir}")
    
    # Create temporary directories
    feature_generator_dir = os.path.join(temp_dir, "feature_generator")
    raw_dir = os.path.join(temp_dir, "raw") 
    analysis_dir = os.path.join(temp_dir, "analysis")
    
    os.makedirs(feature_generator_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Determine working directory (Docker vs local)
    if os.path.exists("/opt/imagine"):
        src_dir = "/opt/imagine"
    elif os.path.exists(os.path.join(os.path.dirname(__file__), "..", "src")):
        # Running locally, check if we're in the src directory or project root
        src_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # Assume we're running locally from project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = current_dir
    
    # Find all .tif files
    tif_files = glob.glob(os.path.join(images_dir, "*.tif"))
    if not tif_files:
        raise ValueError(f"No .tif files found in {images_dir}")
    
    logger.info(f"Found {len(tif_files)} .tif files")
    
    # Generate features for each image
    for tif_file in tif_files:
        try:
            cmd = [
                "python", os.path.join(src_dir, "feature_generator.py"),
                "--outfolder", feature_generator_dir,
                "--file", tif_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Feature generation failed for {tif_file}: {result.stderr}")
                continue
            logger.info(f"Generated features for {os.path.basename(tif_file)}")
        except Exception as e:
            logger.error(f"Error processing {tif_file}: {e}")
            continue
    
    # Create joint dataframe from individual feature files
    try:
        cmd = ["python", os.path.join(src_dir, "create_joint_df.py"), feature_generator_dir, raw_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Joint dataframe creation failed: {result.stderr}")
            raise RuntimeError("Failed to create joint dataframe")
    except Exception as e:
        logger.error(f"Error creating joint dataframe: {e}")
        raise
    
    # Compute aggregated features
    try:
        cmd = ["python", os.path.join(src_dir, "analysis.py"), raw_dir, analysis_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Analysis failed: {result.stderr}")
            raise RuntimeError("Failed to compute aggregated features")
    except Exception as e:
        logger.error(f"Error in analysis step: {e}")
        raise
    
    # Create final dataframe
    data_file = os.path.join(temp_dir, "inference_data.tsv")
    try:
        cmd = ["python", os.path.join(src_dir, "create_final_df_from_results.py"), analysis_dir, data_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Final dataframe creation failed: {result.stderr}")
            raise RuntimeError("Failed to create final dataframe")
    except Exception as e:
        logger.error(f"Error creating final dataframe: {e}")
        raise
    
    if not os.path.exists(data_file):
        raise RuntimeError("Final dataframe was not created successfully")
        
    logger.info(f"Features generated and saved to {data_file}")
    return data_file


def generate_shap_explanations(model, model_name, X_values, X_features, sample_names, output_dir, metadata):
    """Generate SHAP explanations for model predictions."""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - skipping explanations for %s", model_name)
        return
    
    try:
        logger.info(f"Generating SHAP explanations for {model_name}")
        
        # Create a wrapper function for the model that handles the preprocessing pipeline
        def model_wrapper(X_input):
            """Wrapper function that applies the same preprocessing as during training."""
            X_processed = X_input.copy()
            
            # Apply same preprocessing as in run_inference
            thr_features = metadata.get('thr_features', False)
            n_components = metadata.get('n_components', 'all')
            
            # Apply feature thresholding BEFORE SVD if conditions match training
            if not thr_features and n_components == "all" and 'thr_indices' in metadata:
                thr_indices = np.array(metadata['thr_indices'])
                if len(thr_indices) > 0 and max(thr_indices) < X_processed.shape[1]:
                    X_processed = X_processed[:, thr_indices]
            
            # Apply dimensionality reduction if used during training
            if 'svd_transformer' in metadata and metadata['svd_transformer'] is not None:
                svd_transformer = metadata['svd_transformer']
                X_processed = svd_transformer.transform(X_processed)
            
            return model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else model.predict(X_processed)
        
        # Use a subset for SHAP explanations to avoid memory issues
        n_background = min(100, len(X_values))  # Use up to 100 samples for background
        n_explain = min(20, len(X_values))  # Explain up to 20 samples
        
        # Create background dataset for SHAP
        background_indices = np.random.choice(len(X_values), size=n_background, replace=False)
        background_data = X_values[background_indices]
        
        # Choose samples to explain
        explain_indices = np.random.choice(len(X_values), size=n_explain, replace=False)
        explain_data = X_values[explain_indices]
        explain_names = [sample_names[i] for i in explain_indices]
        
        # Use Permutation explainer for model-agnostic explanations
        explainer = shap.PermutationExplainer(model_wrapper, background_data)
        shap_values = explainer(explain_data)
        
        # Save SHAP explanations
        explanations_dir = os.path.join(output_dir, "explanations")
        os.makedirs(explanations_dir, exist_ok=True)
        
        # Save summary values
        if hasattr(shap_values, 'values'):
            # For multi-class classification, take mean across classes
            if len(shap_values.values.shape) == 3:
                mean_shap_values = np.mean(np.abs(shap_values.values), axis=2)
            else:
                mean_shap_values = np.abs(shap_values.values)
            
            # Create explanations DataFrame
            explanations_df = pd.DataFrame(
                mean_shap_values,
                columns=[f'feature_{i}' for i in range(mean_shap_values.shape[1])],
                index=explain_names
            )
            
            # If we have feature names, use them
            if X_features is not None and len(X_features) == mean_shap_values.shape[1]:
                explanations_df.columns = X_features
            
            # Save individual explanations
            explanations_file = os.path.join(explanations_dir, f"{model_name}_shap_explanations.tsv")
            explanations_df.to_csv(explanations_file, sep="\t")
            logger.info(f"Saved SHAP explanations for {model_name} to {explanations_file}")
            
            # Save feature importance summary (mean absolute SHAP values)
            feature_importance = explanations_df.abs().mean().sort_values(ascending=False)
            importance_file = os.path.join(explanations_dir, f"{model_name}_feature_importance.tsv")
            feature_importance.to_frame('importance').to_csv(importance_file, sep="\t")
            logger.info(f"Saved feature importance for {model_name} to {importance_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate SHAP explanations for {model_name}: {e}")
        logger.debug("SHAP error details:", exc_info=True)


def run_inference(models, metadata, features_file, output_dir, enable_explanations=False):
    """Run inference on the prepared features using the loaded models."""
    logger.info(f"Running inference on {features_file}")
    
    # Load the features dataframe
    try:
        data = pd.read_csv(features_file, sep="\t", index_col=0)
        logger.info(f"Loaded features with shape: {data.shape}")
#        logger.info(f"Columns: {list(data.columns)}")
        
        # Remove label column if it exists
        if 'label' in data.columns:
            logger.info("Removing label column from features")
            data = data.drop('label', axis=1)
            logger.info(f"Features shape after removing label: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load features file: {e}")
        raise
    
    # Prepare results storage
    all_predictions = {}
    
    # Run inference with each model
    for model_name, model in models.items():
        logger.info(f"Running inference with model: {model_name}")
        
        try:
            # Get metadata for this model
            meta = metadata.get(model_name, {})
            
            # Prepare features (align with training features if metadata available)
            X = data.copy()
            if 'feature_names' in meta:
                # Ensure we have the same features as during training
                training_features = meta['feature_names']
                missing_features = set(training_features) - set(X.columns)
                extra_features = set(X.columns) - set(training_features)
                
                if missing_features:
                    logger.warning(f"Missing features for {model_name}: {missing_features}")
                    # Add missing features as zeros
                    for feat in missing_features:
                        X[feat] = 0
                        
                if extra_features:
                    logger.info(f"Dropping extra features for {model_name}: {len(extra_features)} features")
                    logger.info(f"Extra features being dropped: {list(extra_features)}")
                    X = X.drop(columns=list(extra_features))
                
                # Reorder columns to match training
                X = X[training_features]
                logger.info(f"Final feature shape for {model_name}: {X.shape}")
            
            # Handle NaN and infinity values before converting to numpy array (same as during training)
            # Following the same pattern as in feature_ranking_lite.py
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Replace inf with max + 3.14, then NaN with -666 (same as training)
                    max_val = X[col].replace([np.inf, -np.inf], np.nan).max()
                    if pd.isna(max_val):  # All values are NaN or inf
                        X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(-666)
                    else:
                        X[col] = X[col].replace([np.inf, -np.inf], [max_val + 3.14, -666]).fillna(-666)
                else:
                    # For non-numeric columns, replace NaN with "missing"
                    X[col] = X[col].fillna("missing")
            
            # Convert to numpy array
            X_values = X.values
            logger.info(f"X_values shape before preprocessing: {X_values.shape}")
            
            # Final check for any remaining NaN or infinity values
            if np.isnan(X_values).any() or np.isinf(X_values).any():
                nan_count = np.isnan(X_values).sum()
                inf_count = np.isinf(X_values).sum()
                if nan_count > 0 or inf_count > 0:
                    logger.warning(f"Found {nan_count} NaN values and {inf_count} infinity values after preprocessing - cleaning up")
                    X_values = np.nan_to_num(X_values, nan=-666.0, posinf=1000.0, neginf=-1000.0)
            
            # Apply same preprocessing as during training
            # Note: Feature thresholding should have been applied during training before SVD
            # The thr_indices refer to the original feature space, but we need to apply SVD first
            # since the saved SVD transformer was fitted on the full feature set
            
            # Apply preprocessing in the same order as during training
            # During training: 
            # - If thr_features=False and n_components="all": apply thr_indices first
            # - If n_components != "all": apply SVD (no thr_indices filtering)
            # - If thr_features=True: no thr_indices filtering at all
            
            thr_features = meta.get('thr_features', False)
            n_components = meta.get('n_components', 'all')
            
            # Apply feature thresholding BEFORE SVD if conditions match training
            if not thr_features and n_components == "all" and 'thr_indices' in meta:
                thr_indices = np.array(meta['thr_indices'])
                if len(thr_indices) > 0 and max(thr_indices) < X_values.shape[1]:
                    logger.info(f"Applying feature thresholding to original features for {model_name} (before SVD)")
                    X_values = X_values[:, thr_indices]
                    logger.info(f"After feature thresholding shape: {X_values.shape}")
            
            # Apply dimensionality reduction if used during training
            if 'svd_transformer' in meta and meta['svd_transformer'] is not None:
                svd_transformer = meta['svd_transformer']
                logger.info(f"Applying saved SVD transformation for {model_name}")
                logger.info(f"Input shape: {X_values.shape}, Expected output: {n_components} components")
                X_values = svd_transformer.transform(X_values)
                logger.info(f"After SVD shape: {X_values.shape}")
                            
            elif n_components != 'all' and n_components is not None:
                logger.error(f"Model {model_name} used dimensionality reduction (n_components={n_components}) but no SVD transformer was saved")
                logger.error("Cannot apply same SVD transformation without the fitted transformer")
                logger.error("This model cannot be used for inference - skipping")
                continue
                        
            else:
                logger.info(f"No dimensionality reduction applied for {model_name}")
            
            # Make predictions
            try:
                predictions = model.predict(X_values)
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_values)
                
                # Convert predictions back to original labels if mapping available
                if 'target_mapping' in meta:
                    target_mapping = meta['target_mapping'] 
                    # Reverse the mapping (from codes to labels)
                    code_to_label = {v: k for k, v in target_mapping.items()}
                    predictions = [code_to_label.get(pred, pred) for pred in predictions]
                
                # Store predictions
                all_predictions[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'sample_names': X.index.tolist(),
                    'processed_features': X_values,
                    'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None
                }
                
                logger.info(f"Successfully generated {len(predictions)} predictions with {model_name}")
                
                # Generate SHAP explanations if enabled
                if enable_explanations:
                    generate_shap_explanations(
                        model, 
                        model_name, 
                        X_values, 
                        X.columns.tolist() if hasattr(X, 'columns') else None,
                        X.index.tolist(), 
                        output_dir, 
                        meta
                    )
                
            except Exception as e:
                logger.error(f"Prediction failed with model {model_name}: {e}")
                continue
                
        except Exception as e:
            logger.error(f"Error running inference with {model_name}: {e}")
            continue
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, results in all_predictions.items():
        # Save predictions as TSV
        pred_df = pd.DataFrame({
            'sample_name': results['sample_names'],
            'prediction': results['predictions']
        })
        
        pred_file = os.path.join(output_dir, f"{model_name}_predictions.tsv")
        pred_df.to_csv(pred_file, sep="\t", index=False)
        logger.info(f"Saved predictions for {model_name} to {pred_file}")
        
        # Save probabilities if available
        if results['probabilities'] is not None:
            prob_df = pd.DataFrame(results['probabilities'], 
                                 index=results['sample_names'])
            prob_file = os.path.join(output_dir, f"{model_name}_probabilities.tsv") 
            prob_df.to_csv(prob_file, sep="\t")
            logger.info(f"Saved probabilities for {model_name} to {prob_file}")
    
    # Create summary file
    summary_data = []
    for model_name, results in all_predictions.items():
        summary_data.append({
            'model': model_name,
            'num_predictions': len(results['predictions']),
            'unique_predictions': len(set(results['predictions']))
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "inference_summary.tsv")
        summary_df.to_csv(summary_file, sep="\t", index=False)
        logger.info(f"Saved inference summary to {summary_file}")
    
    logger.info(f"Inference complete. Results saved to {output_dir}")
    return len(all_predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on biofilm images using pre-trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("models_dir", help="Directory containing trained models")
    parser.add_argument("images_dir", help="Directory containing .tif images for inference")
    parser.add_argument("output_dir", help="Directory to save inference results")
    parser.add_argument("--temp_dir", default="/tmp/inference", 
                       help="Temporary directory for feature generation")
    parser.add_argument("--explanations", action="store_true",
                       help="Generate SHAP explanations for predictions (requires SHAP)")
    
    args = parser.parse_args()
    
    # Check SHAP availability if explanations are requested
    if args.explanations and not SHAP_AVAILABLE:
        logger.error("SHAP explanations requested but SHAP is not installed. Install with: pip install shap")
        sys.exit(1)
    
    # Validate input directories
    if not os.path.isdir(args.models_dir):
        logger.error(f"Models directory does not exist: {args.models_dir}")
        sys.exit(1)
        
    if not os.path.isdir(args.images_dir):
        logger.error(f"Images directory does not exist: {args.images_dir}")
        sys.exit(1)
    
    try:
        # Load models
        logger.info("Loading trained models...")
        models, metadata = load_models(args.models_dir)
        logger.info(f"Loaded {len(models)} models")
        
        # Generate features for input images
        logger.info("Generating features for input images...")
        os.makedirs(args.temp_dir, exist_ok=True)
        features_file = generate_features_for_images(args.images_dir, args.temp_dir)
        
        # Run inference
        logger.info("Running inference...")
        if args.explanations:
            logger.info("SHAP explanations enabled")
        num_models = run_inference(models, metadata, features_file, args.output_dir, 
                                 enable_explanations=args.explanations)
        
        logger.info(f"Inference completed successfully using {num_models} models")
        if args.explanations:
            logger.info("SHAP explanations saved to explanations/ subdirectory")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
