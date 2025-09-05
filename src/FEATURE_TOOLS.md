# Feature Generator CLI Tools

This directory contains CLI-callable feature generation tools for biofilm image analysis.

## Available Tools

### feature_generator.py
Full-featured image processing with comprehensive feature extraction.

**Usage:**
```bash
python feature_generator.py --file input.tif --outfolder output_dir
```

**Features:**
- Complete 3D image processing pipeline
- Comprehensive feature set including:
  - Cell counting with multiple thresholds (0.01-0.3)
  - GPT-enhanced features (fractal dimension, volume, compactness)
  - Texture analysis (GLCM features)
  - Spatial spreading calculations
  - Thickness and roughness measurements
- Homogeneity calculations
- Full biovolume analysis

### feature_generator_lite.py
Lightweight version for faster processing with essential features only.

**Usage:**
```bash
python feature_generator_lite.py --file input.tif --outfolder output_dir
```

**Features:**
- Streamlined 3D image processing
- Essential feature set including:
  - Cell counting with 5 key thresholds (0.05-0.25)
  - Basic intensity statistics
  - Simplified biovolume calculations
  - Basic thickness and coverage measurements
- ~3x faster processing than full version
- Output compatible with downstream analysis

### feature_ranking.py
Comprehensive feature ranking and machine learning analysis.

**Usage:**
```bash
python feature_ranking.py --files data.tsv --fout rankings.tsv
```

### feature_ranking_lite.py
Optimized feature ranking with model saving capabilities.

**Usage:**
```bash
python feature_ranking_lite.py --files data.tsv --fout rankings.tsv [--save_models] [--all_learners]
```

## Integration with Analysis Pipeline

Both feature generators can be used in the main analysis pipeline:

```bash
# Full feature extraction
bash run_analysis.sh /path/to/images 4 datafile.tsv 10 generate_features

# Lite feature extraction (faster)
bash run_analysis.sh /path/to/images 4 datafile.tsv 10 generate_features_lite
```

## When to Use Which

- **feature_generator.py**: Use for comprehensive analysis, research, publication-quality results
- **feature_generator_lite.py**: Use for rapid prototyping, quick analysis, or when processing large datasets

Both tools produce compatible output formats that work with the downstream analysis pipeline.