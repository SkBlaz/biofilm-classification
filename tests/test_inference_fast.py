#!/usr/bin/env python3
"""
Fast inference test using pre-generated features.
This tests the core inference functionality without expensive TIF processing.
"""

import logging
import os
import shutil
import sys

# Add src directory to path
# Use /opt/imagine for Docker, fallback to relative path for local testing
if os.path.exists("/opt/imagine"):
    sys.path.insert(0, "/opt/imagine")
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))

from inference import load_models, run_inference

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def test_inference_with_pregenerated_features():
    """Test inference using pre-generated features from datafile.tsv."""

    models_dir = "/imagine/results/models"
    features_file = "/imagine/results/datafile.tsv"
    temp_output_dir = "/tmp/inference_test_output"

    # Create temporary output directory
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        logger.info("Testing inference with pre-generated features...")

        # Check if models exist
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Check if features file exists
        if not os.path.isfile(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")

        # Load models
        logger.info("Loading models...")
        models, metadata = load_models(models_dir)
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

        # Run inference
        logger.info("Running inference...")
        num_successful = run_inference(models, metadata, features_file, temp_output_dir)

        # Verify results
        if num_successful == 0:
            raise RuntimeError("No models produced successful predictions")

        # Check output files exist
        output_files = os.listdir(temp_output_dir)
        if not output_files:
            raise RuntimeError("No output files were generated")

        # Copy results to final location for verification
        final_output_dir = "/imagine/results/inference_test_output"
        if os.path.exists(final_output_dir):
            shutil.rmtree(final_output_dir)
        shutil.copytree(temp_output_dir, final_output_dir)

        logger.info("Inference test completed successfully!")
        logger.info(f"Generated {len(output_files)} output files")
        logger.info(f"Results saved to {final_output_dir}")

        return True

    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return False
    finally:
        # Clean up temp directory
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)


if __name__ == "__main__":
    success = test_inference_with_pregenerated_features()
    sys.exit(0 if success else 1)
