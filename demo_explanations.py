#!/usr/bin/env python3
"""
Demonstration script showing SHAP explainability output format.
Creates a sample explanation and saves it to show the format.
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_explanation():
    """Create a sample SHAP explanation to demonstrate the output format."""
    
    print("=== Biofilm Classification Inference - SHAP Explainability Demo ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Sample feature names (biofilm-related features)
    feature_names = [
        'cell_density_mean', 'cell_density_std', 'biovolume_total',
        'surface_coverage_ratio', 'thickness_mean', 'thickness_max',
        'roughness_coefficient', 'porosity_fraction', 'connectivity_index',
        'aggregation_measure', 'texture_entropy', 'shape_complexity'
    ]
    
    # Sample names
    sample_names = ['biofilm_001.tif', 'biofilm_002.tif', 'biofilm_003.tif']
    
    # Create sample SHAP values (feature contributions to predictions)
    np.random.seed(42)
    shap_values = np.random.randn(len(sample_names), len(feature_names)) * 0.1
    
    # Make some features more important than others
    shap_values[:, 0] += 0.3  # cell_density_mean is important
    shap_values[:, 4] += 0.2  # thickness_mean is important 
    shap_values[:, 7] += -0.25  # porosity_fraction has negative impact
    
    # Create explanations DataFrame
    explanations_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
        index=sample_names
    )
    
    print("Sample SHAP Explanations (feature contributions to predictions):")
    print("=" * 80)
    print(explanations_df.round(3))
    print()
    
    # Create feature importance (mean absolute SHAP values)
    feature_importance = explanations_df.abs().mean().sort_values(ascending=False)
    
    print("Feature Importance Ranking (based on mean absolute SHAP values):")
    print("=" * 60)
    for i, (feature, importance) in enumerate(feature_importance.head(6).items(), 1):
        print(f"{i:2d}. {feature:<25} {importance:.3f}")
    print()
    
    print("Interpretation:")
    print("- Positive values: Features that push prediction toward the predicted class")
    print("- Negative values: Features that push prediction away from the predicted class")
    print("- Larger absolute values: More influential features for that specific prediction")
    print()
    
    # Save sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        explanations_file = os.path.join(temp_dir, "sample_shap_explanations.tsv")
        importance_file = os.path.join(temp_dir, "sample_feature_importance.tsv")
        
        explanations_df.to_csv(explanations_file, sep='\t')
        feature_importance.to_frame('importance').to_csv(importance_file, sep='\t')
        
        print(f"Sample files saved to:")
        print(f"- Explanations: {explanations_file}")
        print(f"- Feature Importance: {importance_file}")
        print()
        
        print("To enable SHAP explanations in inference, use:")
        print("docker compose run --rm imagine 4 - 10 inference --explanations")
        print("or")
        print("python src/inference.py models_dir images_dir output_dir --explanations")


if __name__ == "__main__":
    create_sample_explanation()