#!/usr/bin/env python3
"""
Inference script for biofilm classification models.

This script loads pre-trained models and runs inference on new .tif images.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)
CLI_USAGE = "Usage: python inference.py <models_dir> <images_dir> <output_dir>"


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


def validate_cli_inputs(models_dir, images_dir):
    """Validate CLI input paths and provide actionable error messages."""
    errors = []

    if not os.path.isdir(models_dir):
        errors.append(f"Models directory does not exist: {models_dir}")
    elif not any(Path(models_dir).glob("*_model.joblib")):
        errors.append(
            f"No model files matching '*_model.joblib' were found in: {models_dir}. "
            "Run learning_benchmark_save_models first or verify model naming."
        )

    if not os.path.isdir(images_dir):
        errors.append(f"Images directory does not exist: {images_dir}")
    elif not any(Path(images_dir).glob("*.tif")):
        errors.append(f"No '.tif' images were found in: {images_dir}")

    if errors:
        raise ValueError("Invalid CLI inputs:\n- " + "\n- ".join(errors) + f"\n{CLI_USAGE}")


def format_predictions(models, metadata, X):
    """Format predictions from multiple models into a single DataFrame."""
    results_list = []

    for sample_name in X.index:
        result = {"sample_name": sample_name}

        for model_name, model in models.items():
            try:
                meta = metadata[model_name]

                # Get single sample
                sample_data = X.loc[[sample_name]]

                # Align features
                feature_names = meta.get("feature_names", [])
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
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(sample_data)[0]
                    confidence = max(probabilities)

                # Convert prediction back to original label if mapping available
                if "target_mapping" in meta:
                    target_mapping = meta["target_mapping"]
                    # target_mapping is already code->label mapping
                    prediction = target_mapping.get(prediction, prediction)

                result[f"{model_name}_prediction"] = prediction
                result[f"{model_name}_confidence"] = confidence

            except Exception as e:
                logger.error(f"Error predicting with {model_name} for {sample_name}: {e}")
                result[f"{model_name}_prediction"] = "ERROR"
                result[f"{model_name}_confidence"] = 0.0

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
            cmd = ["python", os.path.join(src_dir, "feature_generator.py"), "--outfolder", feature_generator_dir, "--file", tif_file]
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


def run_inference(models, metadata, features_file, output_dir):
    """Run inference on the prepared features using the loaded models."""
    logger.info(f"Running inference on {features_file}")

    # Load the features dataframe
    try:
        data = pd.read_csv(features_file, sep="\t", index_col=0)
        logger.info(f"Loaded features with shape: {data.shape}")
        #        logger.info(f"Columns: {list(data.columns)}")

        # Remove label column if it exists
        if "label" in data.columns:
            logger.info("Removing label column from features")
            data = data.drop("label", axis=1)
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
            if "feature_names" in meta:
                # Ensure we have the same features as during training
                training_features = meta["feature_names"]
                missing_features = set(training_features) - set(X.columns)
                extra_features = set(X.columns) - set(training_features)

                if missing_features:
                    logger.warning(f"Missing features for {model_name}: {missing_features}")
                    # Add missing features as zeros
                    for feat in missing_features:
                        X[feat] = 0

                if extra_features:
                    logger.info(f"Dropping extra features for {model_name}: {len(extra_features)} features")
                    logger.info(
                        f"Extra features being dropped: {list(extra_features) if len(extra_features) < 10 else str(len(extra_features)) + ' features'}"
                    )
                    X = X.drop(columns=list(extra_features))

                # Reorder columns to match training
                X = X[training_features]
                logger.info(f"Final feature shape for {model_name}: {X.shape}")

            # Handle NaN and infinity values before converting to numpy array (same as during training)
            # Following the same pattern as in feature_ranking_lite.py
            for col in X.columns:
                if X[col].dtype in ["float64", "float32", "int64", "int32"]:
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

            thr_features = meta.get("thr_features", False)
            n_components = meta.get("n_components", "all")

            # Apply feature thresholding BEFORE SVD if conditions match training
            if not thr_features and n_components == "all" and "thr_indices" in meta:
                thr_indices = np.array(meta["thr_indices"])
                if len(thr_indices) > 0 and max(thr_indices) < X_values.shape[1]:
                    logger.info(f"Applying feature thresholding to original features for {model_name} (before SVD)")
                    X_values = X_values[:, thr_indices]
                    logger.info(f"After feature thresholding shape: {X_values.shape}")

            # Apply dimensionality reduction if used during training
            if "svd_transformer" in meta and meta["svd_transformer"] is not None:
                svd_transformer = meta["svd_transformer"]
                logger.info(f"Applying saved SVD transformation for {model_name}")
                logger.info(f"Input shape: {X_values.shape}, Expected output: {n_components} components")
                X_values = svd_transformer.transform(X_values)
                logger.info(f"After SVD shape: {X_values.shape}")

            elif n_components != "all" and n_components is not None:
                logger.error(
                    f"Model {model_name} used dimensionality reduction (n_components={n_components}) but no SVD transformer was saved"
                )
                logger.error("Cannot apply same SVD transformation without the fitted transformer")
                logger.error("This model cannot be used for inference - skipping")
                continue

            else:
                logger.info(f"No dimensionality reduction applied for {model_name}")

            # Make predictions
            try:
                predictions = model.predict(X_values)
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_values)

                # Convert predictions back to original labels if mapping available
                if "target_mapping" in meta:
                    target_mapping = meta["target_mapping"]
                    # target_mapping is already code->label mapping
                    predictions = [target_mapping.get(pred, pred) for pred in predictions]

                # Store predictions and processed features
                all_predictions[model_name] = {
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "sample_names": X.index.tolist(),
                    "processed_features": X,  # Store the processed features dataframe
                }

                logger.info(f"Successfully generated {len(predictions)} predictions with {model_name}")

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
        pred_df = pd.DataFrame({"sample_name": results["sample_names"], "prediction": results["predictions"]})

        pred_file = os.path.join(output_dir, f"{model_name}_predictions.tsv")
        pred_df.to_csv(pred_file, sep="\t", index=False)
        logger.info(f"Saved predictions for {model_name} to {pred_file}")

        # Save probabilities if available
        if results["probabilities"] is not None:
            # Get column names using target_mapping if available
            meta = metadata.get(model_name, {})
            column_names = None
            if "target_mapping" in meta:
                # target_mapping is code->label mapping
                target_mapping = meta["target_mapping"]
                num_classes = results["probabilities"].shape[1]
                column_names = [target_mapping.get(i, f"class_{i}") for i in range(num_classes)]

            prob_df = pd.DataFrame(results["probabilities"], index=results["sample_names"], columns=column_names)
            prob_file = os.path.join(output_dir, f"{model_name}_probabilities.tsv")
            prob_df.to_csv(prob_file, sep="\t")
            logger.info(f"Saved probabilities for {model_name} to {prob_file}")

        # Save processed features if available
        if "processed_features" in results:
            features_df = results["processed_features"]
            features_file = os.path.join(output_dir, f"{model_name}_features.tsv")
            features_df.to_csv(features_file, sep="\t")
            logger.info(f"Saved processed features for {model_name} to {features_file}")

    # Create summary file
    summary_data = []
    for model_name, results in all_predictions.items():
        summary_data.append(
            {"model": model_name, "num_predictions": len(results["predictions"]), "unique_predictions": len(set(results["predictions"]))}
        )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "inference_summary.tsv")
        summary_df.to_csv(summary_file, sep="\t", index=False)
        logger.info(f"Saved inference summary to {summary_file}")

    # Generate SHAP explanations
    try:
        generate_shap_explanations(models, metadata, all_predictions, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate SHAP explanations: {e}")
        logger.warning("Continuing without SHAP explanations")

    logger.info(f"Inference complete. Results saved to {output_dir}")
    return len(all_predictions)


def generate_shap_explanations(models, metadata, all_predictions, output_dir):
    """Generate SHAP explanations for model predictions.

    Args:
        models: Dictionary of loaded models
        metadata: Dictionary of model metadata
        all_predictions: Dictionary of prediction results including processed features
        output_dir: Directory to save SHAP explanations
    """
    logger.info("Generating SHAP explanations...")
    try:
        import shap
    except ImportError:
        logger.warning("SHAP is not installed; skipping explanation generation")
        return

    # Create explanations subdirectory
    explanations_dir = os.path.join(output_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)

    for model_name, model in models.items():
        if model_name not in all_predictions:
            logger.warning(f"Skipping SHAP for {model_name} - no predictions available")
            continue

        try:
            logger.info(f"Generating SHAP explanations for {model_name}...")

            results = all_predictions[model_name]
            X = results["processed_features"]
            # Convert to numpy array for SHAP
            X_values = X.values

            # Get metadata for this model
            meta = metadata.get(model_name, {})

            # Apply the same transformations as during inference
            thr_features = meta.get("thr_features", False)
            n_components = meta.get("n_components", "all")

            # Apply feature thresholding BEFORE SVD if conditions match training
            if not thr_features and n_components == "all" and "thr_indices" in meta:
                thr_indices = np.array(meta["thr_indices"])
                if len(thr_indices) > 0 and max(thr_indices) < X_values.shape[1]:
                    X_values = X_values[:, thr_indices]

            # Apply dimensionality reduction if used during training
            if "svd_transformer" in meta and meta["svd_transformer"] is not None:
                svd_transformer = meta["svd_transformer"]
                X_values = svd_transformer.transform(X_values)
                # For SVD-transformed features, use generic column names
                feature_names = [f"component_{i}" for i in range(X_values.shape[1])]
            else:
                # Use original feature names
                feature_names = list(X.columns)

            # Select an appropriate SHAP explainer based on model type
            explainer = None
            shap_values = None

            # Try TreeExplainer first (for tree-based models)
            if hasattr(model, "estimators_") or "Forest" in str(type(model)) or "XGB" in str(type(model)) or "Tree" in str(type(model)):
                try:
                    logger.info(f"Using TreeExplainer for {model_name}")

                    df_x = pd.DataFrame(X_values, columns=feature_names)
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(df_x)
                    explanation = explainer(df_x)

                    import matplotlib

                    matplotlib.use("Agg")  # Use non-interactive backend
                    import matplotlib.pyplot as plt

                    # Generate explanations for each example
                    for i, (idx, row) in enumerate(X.iterrows()):
                        logger.info(f"Generating explanation for model {model_name} and example {idx}")

                        row_dir = os.path.join(explanations_dir, idx)
                        os.makedirs(row_dir, exist_ok=True)
                        for class_idx in range(shap_values.shape[2]):
                            class_name = class_idx
                            if "target_mapping" in meta:
                                class_name = meta["target_mapping"].get(class_idx, class_idx)

                            flag = "_true" if class_name in idx else ""
                            flag += "_predicted" if results["predictions"][i] == class_name else ""

                            plt.figure(figsize=(10, 6))
                            shap.plots.decision(
                                explainer.expected_value[class_idx],
                                shap_values[i, :, class_idx],
                                feature_display_range=slice(None, -16, -1),
                                feature_names=list(df_x.columns),
                                show=False,
                            )

                            decision_plot_file = os.path.join(row_dir, f"{model_name}_{class_name}_decision{flag}.png")
                            plt.savefig(decision_plot_file, bbox_inches="tight", dpi=150)
                            plt.close()

                            plt.figure(figsize=(10, 6))
                            shap.plots.waterfall(explanation[i, :, class_idx], max_display=15, show=False)

                            waterfall_plot_file = os.path.join(row_dir, f"{model_name}_{class_name}_waterfall{flag}.png")
                            plt.savefig(waterfall_plot_file, bbox_inches="tight", dpi=150)
                            plt.close()

                    # TreeExplainer might return a list for multi-class or 3D array for binary
                    # Handle 3D array case (binary classification)
                    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        logger.info("Converting 3D SHAP values to 2D (taking positive class)")
                        # For binary classification, take the values for the positive class (index 1)
                        # This is shape (n_samples, n_features, n_classes) -> (n_samples, n_features)
                        shap_values = shap_values[:, :, 1]

                except Exception as e:
                    logger.warning(f"TreeExplainer failed for {model_name}: {e}")
                    shap_values = None

            # Fall back to KernelExplainer (model-agnostic but slower)
            if shap_values is None:
                try:
                    logger.info(f"Using KernelExplainer for {model_name}")
                    # Sample a subset of data as background for KernelExplainer
                    # Use min of 100 samples or all available
                    background_size = min(100, X_values.shape[0])
                    background_indices = np.random.choice(X_values.shape[0], background_size, replace=False)
                    background = X_values[background_indices]

                    # Create explainer with predict function
                    if hasattr(model, "predict_proba"):

                        def predict_fn(x):
                            return model.predict_proba(x)
                    else:

                        def predict_fn(x):
                            return model.predict(x).reshape(-1, 1)

                    explainer = shap.KernelExplainer(predict_fn, background)

                    # Compute SHAP values for all samples (can be slow)
                    # Limit to first 50 samples if dataset is large
                    n_samples = min(50, X_values.shape[0])
                    shap_values = explainer.shap_values(X_values[:n_samples])

                    # Handle 3D array case (binary classification) for KernelExplainer too
                    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        logger.info("Converting 3D SHAP values to 2D (taking positive class)")
                        shap_values = shap_values[:, :, 1]

                    # Adjust X_values and related data to match
                    if n_samples < X_values.shape[0]:
                        logger.warning(f"Limited SHAP computation to {n_samples} samples due to computational cost")
                        X_values = X_values[:n_samples]
                        results["sample_names"] = results["sample_names"][:n_samples]

                except Exception as e:
                    logger.error(f"KernelExplainer failed for {model_name}: {e}")
                    shap_values = None
                    continue

            if shap_values is None:
                logger.error(f"Could not generate SHAP values for {model_name}")
                continue

            # Handle multi-class case - shap_values might be a list
            if isinstance(shap_values, list):
                # For multi-class, take the class with highest probability for each sample
                # Or we can save explanations for each class
                logger.info(f"Multi-class model detected with {len(shap_values)} classes")

                # Save SHAP values for each class
                for class_idx, class_shap in enumerate(shap_values):
                    class_name = class_idx
                    if "target_mapping" in meta:
                        class_name = meta["target_mapping"].get(class_idx, class_idx)

                    # Create DataFrame with SHAP values
                    shap_df = pd.DataFrame(class_shap, index=results["sample_names"], columns=feature_names)

                    # Save to CSV
                    shap_file = os.path.join(explanations_dir, f"{model_name}_shap_class_{class_name}.csv")
                    shap_df.to_csv(shap_file)
                    logger.info(f"Saved SHAP values for {model_name} class {class_name}")

                # For summary, use the first class
                shap_values_summary = shap_values[0]
            else:
                shap_values_summary = shap_values

                # Create DataFrame with SHAP values
                shap_df = pd.DataFrame(shap_values_summary, index=results["sample_names"], columns=feature_names)

                # Save to CSV
                shap_file = os.path.join(explanations_dir, f"{model_name}_shap_values.csv")
                shap_df.to_csv(shap_file)
                logger.info(f"Saved SHAP values for {model_name}")

            # Generate summary plot and save as HTML
            try:
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend
                import matplotlib.pyplot as plt

                # Summary plot
                shap.summary_plot(shap_values_summary, X_values, feature_names=feature_names, show=False)
                summary_plot_file = os.path.join(explanations_dir, f"{model_name}_summary_plot.png")
                plt.gcf().set_size_inches(56, 12)
                plt.savefig(summary_plot_file, bbox_inches="tight", dpi=150)
                plt.close()
                logger.info(f"Saved summary plot for {model_name}")

                # Feature importance (mean absolute SHAP values)
                mean_abs_shap = np.abs(shap_values_summary).mean(axis=0)
                importance_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap}).sort_values(
                    "importance", ascending=False
                )

                importance_file = os.path.join(explanations_dir, f"{model_name}_feature_importance.csv")
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"Saved feature importance for {model_name}")

            except Exception as e:
                logger.warning(f"Could not generate plots for {model_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations for {model_name}: {e}")
            continue

    logger.info(f"SHAP explanations saved to {explanations_dir}")


def total_abs_diff_per_feature(rf, X_values, feature_names=None):
    """
    Computes total absolute difference between feature values and split thresholds
    along the decision path for a given sample, across all trees in a RandomForest.

    Returns both the global total and per-feature totals.

    Parameters
    ----------
    rf : fitted RandomForestClassifier or RandomForestRegressor
        The trained random forest.
    X_values : array-like of shape (n_features,)
        The feature vector for the sample.
    feature_names : list of str, optional
        Feature names for readability.

    Returns
    -------
    total_diff : float
        Sum of all absolute differences across all splits and trees.
    per_feature_diff : dict
        Mapping {feature_name or index: total_abs_difference}.
    details : list of dict
        Full details of each split (tree index, feature, threshold, value, abs_diff).
    """

    X_values = np.array(X_values).reshape(1, -1)
    total_diff = 0.0
    per_feature_diff = defaultdict(float)
    details = []

    for tree_idx, estimator in enumerate(rf.estimators_):
        tree = estimator.tree_
        node_indicator = estimator.decision_path(X_values)
        node_index = node_indicator.indices  # indices of nodes that the sample passes through

        for node_id in node_index:
            # Skip leaf nodes
            if tree.children_left[node_id] == -1:
                continue

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_value = X_values[0, feature_idx]
            diff = abs(feature_value - threshold)

            total_diff += diff
            key = feature_names[feature_idx] if feature_names is not None else feature_idx
            per_feature_diff[key] += diff

            details.append({"tree": tree_idx, "feature": key, "threshold": threshold, "feature_value": feature_value, "abs_diff": diff})

    return total_diff, dict(per_feature_diff), details


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on biofilm images using pre-trained models", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("models_dir", help="Directory containing trained models")
    parser.add_argument("images_dir", help="Directory containing .tif images for inference")
    parser.add_argument("output_dir", help="Directory to save inference results")
    parser.add_argument("--temp_dir", default="/tmp/inference", help="Temporary directory for feature generation")

    args = parser.parse_args()

    try:
        validate_cli_inputs(args.models_dir, args.images_dir)
    except ValueError as e:
        logger.error(str(e))
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
        num_models = run_inference(models, metadata, features_file, args.output_dir)

        logger.info(f"Inference completed successfully using {num_models} models")

    except Exception:
        logger.exception("Inference failed unexpectedly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
