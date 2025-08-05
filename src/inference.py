#!/usr/bin/env python3
"""
Inference script for biofilm classification.
Loads pre-trained models and generates predictions on new data.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def load_model_metadata(metadata_path):
    """Load model metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def preprocess_features(X, model_metadata):
    """
    Apply the same feature preprocessing used during training.
    """
    # Get preprocessing parameters
    n_components = model_metadata['n_components']
    thr_features = model_metadata['thr_features']
    threshold_indices = model_metadata.get('threshold_indices', [])
    
    X_processed = X.values.copy()
    
    # Apply threshold feature filtering if needed
    if not thr_features and n_components == "all" and threshold_indices:
        X_processed = X_processed[:, threshold_indices]
        logger.info(f"Applied threshold filtering, using {len(threshold_indices)} features")
    
    return X_processed


def make_predictions(data_path, models_dir, output_dir, model_names=None, images_folder=None):
    """
    Load trained models and make predictions on new data.
    
    Args:
        data_path: Path to the dataset (TSV file with features)
        models_dir: Directory containing saved models
        output_dir: Directory to save prediction results
        model_names: List of specific model names to use (optional)
        images_folder: Path to folder containing original .tif images (for mapping filenames)
    """
    logger.info(f"Starting inference on {data_path}")
    logger.info(f"Using models from {models_dir}")
    logger.info(f"Saving results to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    logger.info(f"Loading dataset from {data_path}")
    data = pd.read_csv(data_path, sep='\t', index_col=0)
    
    # Separate features from labels (if present)
    feature_cols = [col for col in data.columns if col != 'label']
    X = data[feature_cols]
    
    # Check if labels are present (for evaluation)
    has_labels = 'label' in data.columns
    if has_labels:
        y_true = data['label']
        logger.info(f"Dataset contains {len(X)} samples with {len(feature_cols)} features and labels")
    else:
        logger.info(f"Dataset contains {len(X)} samples with {len(feature_cols)} features (no labels)")
    
    # Find available models
    model_summaries = []
    for filter_mode in ['all', 'no_counts_features']:
        summary_path = os.path.join(models_dir, f"models_summary_{filter_mode}.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                for model_name, model_info in summary.items():
                    model_info['filter_mode'] = filter_mode
                    model_summaries.append((model_name, model_info))
    
    if not model_summaries:
        raise ValueError(f"No trained models found in {models_dir}")
    
    logger.info(f"Found {len(model_summaries)} trained models")
    
    # Filter models if specific names are requested
    if model_names:
        model_summaries = [(name, info) for name, info in model_summaries 
                          if name in model_names]
        logger.info(f"Using {len(model_summaries)} specified models: {model_names}")
    
    all_predictions = {}
    all_probabilities = {}
    
    for model_name, model_info in model_summaries:
        try:
            filter_mode = model_info['filter_mode']
            logger.info(f"Making predictions with {model_name} (filter_mode: {filter_mode})")
            
            # Load model metadata
            metadata_path = os.path.join(models_dir, f"{model_name}_{filter_mode}_metadata.json")
            metadata = load_model_metadata(metadata_path)
            
            # Load the model
            model_path = os.path.join(models_dir, f"{model_name}_{filter_mode}.joblib")
            model = joblib.load(model_path)
            
            # Apply feature filtering for no_counts_features mode
            X_filtered = X.copy()
            if filter_mode == 'no_counts_features':
                # Remove features containing 'counts'
                feature_cols_filtered = [col for col in X.columns if 'counts' not in col]
                X_filtered = X[feature_cols_filtered]
                logger.info(f"Filtered features for {model_name}: {len(feature_cols_filtered)} features")
            
            # Preprocess features
            X_processed = preprocess_features(X_filtered, metadata)
            
            # Apply SVD transformation if used during training
            if 'svd_path' in metadata:
                svd_path = os.path.join(models_dir, metadata['svd_path'])
                svd = joblib.load(svd_path)
                X_processed = svd.transform(X_processed)
                logger.info(f"Applied SVD transformation: {X_processed.shape[1]} components")
            
            # Make predictions
            predictions = model.predict(X_processed)
            
            # Convert predictions back to original labels
            class_mapping = metadata['class_mapping']
            # Reverse the mapping: code -> label
            reverse_mapping = {int(code): label for code, label in class_mapping.items()}
            predicted_labels = [reverse_mapping[pred] for pred in predictions]
            
            all_predictions[f"{model_name}_{filter_mode}"] = predicted_labels
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_processed)
                    all_probabilities[f"{model_name}_{filter_mode}"] = probabilities
                except:
                    logger.warning(f"Could not get probabilities for {model_name}")
            
            logger.info(f"Generated {len(predictions)} predictions with {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to make predictions with {model_name}: {e}")
            continue
    
    if not all_predictions:
        raise ValueError("No models were able to make predictions")
    
    # Map sample indices to TIF filenames if images_folder is provided
    sample_to_tif = {}
    if images_folder and os.path.exists(images_folder):
        # Get list of TIF files sorted to ensure consistent mapping
        tif_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.tif')])
        logger.info(f"Found {len(tif_files)} TIF files in images folder")
        
        # Create mapping from sample index to TIF filename
        # The sample indices in the dataframe should correspond to the TIF files processed
        sample_indices = list(data.index)
        
        for i, sample_idx in enumerate(sample_indices):
            if i < len(tif_files):
                # Use basename without extension to match with sample index pattern  
                tif_basename = os.path.splitext(tif_files[i])[0]
                # Try to match by sample index name first, fallback to order-based mapping
                if str(sample_idx) in tif_basename or tif_basename in str(sample_idx):
                    sample_to_tif[sample_idx] = tif_files[i]
                else:
                    # Fallback: map by order
                    sample_to_tif[sample_idx] = tif_files[i]
            else:
                sample_to_tif[sample_idx] = f"sample_{sample_idx}.tif"
                
        logger.info(f"Mapped {len(sample_to_tif)} samples to TIF files")
    else:
        # Use sample indices as filenames when no images folder provided
        for sample_idx in data.index:
            sample_to_tif[sample_idx] = f"{sample_idx}.tif"
    
    # Create results in the requested format: tif_filename, model_name, prediction
    predictions_list = []
    for sample_idx in data.index:
        tif_filename = sample_to_tif[sample_idx]
        for model_name in all_predictions.keys():
            prediction = all_predictions[model_name][data.index.get_loc(sample_idx)]
            predictions_list.append({
                'tif_filename': tif_filename,
                'model_name': model_name,
                'prediction': prediction
            })
    
    # Create the final predictions DataFrame
    predictions_df = pd.DataFrame(predictions_list)
    
    # Save predictions in the requested format
    predictions_path = os.path.join(output_dir, 'predictions.tsv')
    predictions_df.to_csv(predictions_path, sep='\t', index=False)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Also create the legacy format for backward compatibility
    results_df = pd.DataFrame(all_predictions, index=data.index)
    
    # Add true labels if available
    if has_labels:
        results_df.insert(0, 'true_label', y_true)
    
    # Save legacy format
    legacy_predictions_path = os.path.join(output_dir, 'predictions_legacy.tsv')
    results_df.to_csv(legacy_predictions_path, sep='\t')
    logger.info(f"Saved legacy format predictions to {legacy_predictions_path}")
    
    # Save probabilities if available
    if all_probabilities:
        prob_dir = os.path.join(output_dir, 'probabilities')
        os.makedirs(prob_dir, exist_ok=True)
        
        for model_name, probs in all_probabilities.items():
            prob_df = pd.DataFrame(probs, index=data.index)
            prob_path = os.path.join(prob_dir, f'{model_name}_probabilities.tsv')
            prob_df.to_csv(prob_path, sep='\t')
        
        logger.info(f"Saved prediction probabilities to {prob_dir}")
    
    # Calculate accuracy if labels are available
    if has_labels:
        logger.info("\nModel Performance:")
        for model_name in all_predictions.keys():
            y_pred = all_predictions[model_name]
            # Convert to same type for comparison
            y_true_list = y_true.tolist()
            accuracy = sum(1 for true, pred in zip(y_true_list, y_pred) if true == pred) / len(y_true_list)
            logger.info(f"  {model_name}: {accuracy:.4f}")
    
    # Create summary
    summary = {
        'dataset_path': data_path,
        'num_samples': len(X),
        'num_features': len(feature_cols),
        'models_used': list(all_predictions.keys()),
        'output_files': {
            'predictions': 'predictions.tsv',
            'probabilities': 'probabilities/' if all_probabilities else None
        }
    }
    
    summary_path = os.path.join(output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Inference completed. Results saved to {output_dir}")
    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained biofilm classification models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the dataset file (TSV format)"
    )
    parser.add_argument(
        "--models_dir", 
        required=True,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output_dir",
        required=True, 
        help="Directory to save prediction results"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model names to use (if not specified, uses all available models)"
    )
    parser.add_argument(
        "--images_folder",
        help="Path to folder containing original .tif images (for mapping filenames)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file not found: {args.data}")
    
    if not os.path.exists(args.models_dir):
        raise FileNotFoundError(f"Models directory not found: {args.models_dir}")
    
    make_predictions(
        data_path=args.data,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        model_names=args.models,
        images_folder=args.images_folder
    )


if __name__ == "__main__":
    main()