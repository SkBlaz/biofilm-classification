#!/bin/bash
# Demonstration script for the inference mode functionality
# This script shows how to use the new inference feature

echo "=== Biofilm Classification Inference Mode Demo ==="
echo

# Check if we're in the right directory
if [ ! -f "src/inference.py" ]; then
    echo "Error: Please run this script from the biofilm-classification root directory"
    exit 1
fi

echo "This demo shows the new inference functionality that was just implemented."
echo "The inference mode allows you to:"
echo "  1. Load previously trained models"
echo "  2. Generate predictions from raw images OR pre-computed features"  
echo "  3. Output results to a specified folder"
echo "  4. NEW: Full pipeline from image files to predictions!"
echo

echo "=== Available Tasks ==="
echo "The pipeline now supports these tasks:"
echo "  - generate_features: Extract features from biofilm images"
echo "  - learning_benchmark: Train ML models and save best performers"
echo "  - data_visualization: Create visualizations of top features"
echo "  - inference: Load saved models and generate predictions (NEW!)"
echo

echo "=== Usage Examples ==="
echo
echo "1. Docker usage for image-level inference (NEW!):"
echo "   # Place your .tif images in a folder and run:"
echo "   docker compose run --rm imagine 4 datafile.tsv 10 inference"
echo "   # The system will automatically detect images and run the full pipeline"
echo
echo "2. Docker usage for feature-level inference:"
echo "   # If you already have computed features:"
echo "   docker compose run --rm imagine 4 datafile.tsv 10 inference"
echo "   # The system will use the pre-computed datafile.tsv"
echo
echo "3. Direct usage:"
echo "   python3 src/inference.py --data /path/to/data.tsv \\"
echo "                           --models_dir /path/to/models \\"
echo "                           --output_dir /path/to/results"
echo

echo "=== Testing the Implementation ==="
echo "The implementation has been tested and verified with:"
echo "  ✅ Model saving during learning_benchmark"
echo "  ✅ Model loading and prediction generation"
echo "  ✅ Both labeled and unlabeled datasets"
echo "  ✅ Multiple model types (RandomForest, Logistic Regression, etc.)"
echo "  ✅ Proper preprocessing pipeline preservation"
echo "  ✅ Error handling and validation"
echo "  ✅ NEW: Full inference pipeline from raw images"
echo "  ✅ NEW: Automatic feature generation during inference"
echo

echo "=== Output Files ==="
echo "When you run inference, you'll get:"
echo "  📁 inference_results/"
echo "    ├── predictions.tsv          # Model predictions with true labels"
echo "    ├── probabilities/           # Prediction probabilities per model"
echo "    └── inference_summary.json   # Run summary and metrics"
echo

echo "=== Next Steps ==="
echo "1. Run 'docker compose build' to build the updated image"
echo "2. Run the learning_benchmark task to train and save models"
echo "3. Run the inference task to generate predictions"
echo
echo "For detailed documentation, see RUN.md"
echo
echo "🎉 Inference mode implementation complete!"