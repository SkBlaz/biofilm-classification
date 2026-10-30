# How MicroICS works

MicroICS converts 2D and 3D biofilm microscopy images into measurements, learns which measurements distinguish known biological classes, and applies the trained models to new images. One Docker image supplies both the browser GUI and reproducible scientific environment; all computation runs in that same container.

## Using MicroICS through the GUI

The GUI presents the analysis as three biological questions rather than a list of scripts.

### 1. Generate features

Feature generation measures each 2D image or 3D TIFF stack and writes one row per image.

**Labelled images** are images whose filenames contain a known class, such as a strain. Choose this option when preparing a table for model training. MicroICS checks the filename convention, reports the number of images in each class and acquisition date, and writes `datafile.tsv`.

**Unlabelled images** are new or unknown samples whose class is not yet known. Choose this option when preparing images for prediction. Their filenames do not need to contain a class. MicroICS writes `unknown_features.tsv` while preserving the filenames so predictions can be traced back to images.

Before either calculation, enter the physical voxel dimensions X, Y, and Z in micrometres. MicroICS uses them for measurements expressed as physical area, volume, or thickness. The displayed defaults (0.13, 0.13, and 0.5 µm) should be replaced with the values recorded by the microscope when they differ. Z is ignored for 2D images; only X and Y spacing affect 2D measurements.

Feature families describe complementary aspects of biofilm structure, including:

- stack-normalized intensity and threshold-sweep object counts;
- layer-to-layer intensity differences and normalized dispersion;
- calibrated biomass, voxel-extent maximum height, mean thickness, compacted column thickness, roughness, and substratum coverage;
- substratum coverage/biomass area, object count/density/size, equivalent diameter, enclosed pore summaries, and anisotropy-aware nearest-neighbour summaries;
- biomass centre-of-mass height, vertical spread, surface area, surface-to-volume ratio, and diffusion distance;
- intensity quantiles, homogeneity, anisotropy-aware spatial spreading, and 32-level directional gray-level co-occurrence summaries.

Feature calculation is selected in the GUI: 2D inputs use one complete structural/intensity/texture package; 3D inputs offer basic biofilm, object, advanced geometry, intensity, grayscale geometry/texture, and serial-threshold groups. Object features require a valid multi-slice mask. Deselected packages are not calculated. Object and pore distributions accompany their numeric ML summaries.

Segmentation operates on a reproducible 8-bit conversion of the whole stack. Acquisition metadata or an explicit maximum (4095 for genuine 12-bit data) defines the range; absent metadata, non-8-bit images use the observed minimum and maximum across the complete stack. Existing unsigned 8-bit images remain unchanged. Conversion bounds and range mode are recorded; the CLI can explicitly restore storage-range conversion. The exact converted TIFF is available for ImageJ tuning. Original-precision intensity measurements remain separate. Automatic Otsu, three-class Otsu, BiofilmQ-style trimmed robust background, and manual cutoffs share fixed COMSTAT2 inclusive boundaries. Three-class Otsu chooses one of its two cutoffs according to the dim-class assignment; these are intensity classes, not biological cell states. Scikit-image rolling-ball subtraction has been removed; thresholding uses the converted stack directly. Whole-stack Otsu remains the default; an optional substratum source estimates one cutoff from the first N slices. BEM has been removed from selectable methods because its input restrictions repeatedly reject these data. Imported binary or integer-labelled masks bypass thresholding and preserve supplied object identities. Connectivity and physical-size filtering affect the actual biomass mask before QC and measurements.

QC exports full-stack biomass masks, reproducible random-colour RGB labels, RGB overlays, the exact 8-bit intensity histogram, a biomass-versus-threshold curve, and applied-setting metadata under `feature_generator/segmentation/`. Numeric object-label TIFFs and projection QC PNGs are omitted; RGB labels and full-stack overlays remain. Separate first/middle/top PNGs are also omitted. Both plots mark the base automatic and sensitivity-adjusted thresholds. Border QC measures edge foreground as a fraction of all foreground and excludes the singleton Z axis for 2D images. Imported masks have no threshold curve. Tune settings on a small upload before a full batch. Saved generation settings reproduce the segmentation/feature contract during raw-image inference; older models need compatible original feature tables or retraining. QC supports visual assessment and does not certify biological validity.

Spatial spreading uses the selected mask with independent x/y/z calibration. Surface area uses a voxel-face estimator; local thickness uses a calibrated digital voxel-centre sphere convention, not a continuous mesh. Local density samples a physical-radius sphere around biomass voxels. Internal porosity and pores exclude border-connected background. Synthetic masks check filled thickness, roughness, height extent, and calibrated geometry; exact agreement with exported COMSTAT measurements remains a separate validation task. See `RUN.md` for conversion, units, workflow, and algorithm details.

### 2. Train models

Training asks whether structural measurements can distinguish the known labels. It reads a complete Excel XLSX, CSV, TSV, or tab-separated TXT feature table containing `sampleName`, `label`, and numeric feature columns. The table can be generated by MicroICS or assembled externally. The GUI also accepts a second measurement table and joins numeric columns by exact `sampleName` before training. The tables must have identical sample sets and unique, nonempty names; row order may differ. Duplicate headers, overlapping feature names, and conflicting optional labels are rejected. Exact sample identifiers, including leading zeros, are preserved through training. The combined table is exported with the job results.

Learning defaults to the published main-branch evaluation protocol with ordinary stratified cross-validation. The GUI and CLI also offer optional grouped cross-validation by acquisition date, well, or imaging position. Grouped evaluation keeps related samples together across folds and can estimate performance on independent experimental groups. The selected unit must be statistically feasible for the dataset; it should also match the study's experimental design. Results from grouped and ordinary stratified protocols are not directly comparable.

**One learner** is the normal starting point. Random Forest is the default and provides a focused run with lower time and resource requirements. Other single learners are available for a planned comparison.

**All learners** benchmarks every available algorithm under the same folds. Use it when an algorithm comparison is scientifically relevant and enough compute time is available. It is substantially slower than one learner; the worker and Docker resource controls bound its CPU and memory use.

During training, MicroICS:

1. ranks image-derived features;
2. evaluates different feature subsets in held-out folds;
3. tunes the selected learner or learners;
4. writes classification tables and confusion matrices;
5. trains final models on all labelled data and saves them in `results/models`.

Generated feature columns are retained even when they contain zero, NaN, or infinite values, and the learning and inference loaders apply the established value-level imputation consistently. Missing sample identifiers, duplicate samples, non-numeric text, or missing training labels stop the run.

### 3. Inference

Inference answers: “Which learned class best matches each new biofilm image?” It loads the saved models, generates the same measurements for new images (or reads a complete precomputed feature table), aligns them to the features expected by each model, and writes predictions and explanations to the chosen inference folder.

Inference is not a substitute for independent biological validation. Predictions are most reliable when new images use acquisition and preprocessing conditions comparable to the training set.

### Combined workflow and recovery

**All together** runs labelled feature generation, model training, and inference in sequence. Separate stages are preferable while checking a new dataset because their intermediate tables are easier to inspect.

The GUI preflight report appears before computation. **Stop** interrupts the active scientific subprocess while keeping the web server and job log available. **New job** resets the transient interface state; completed files remain under that job ID when `/data` is persisted.

## Runtime architecture

The browser uploads data to a per-job directory under `/data`. `gui/execution.py` maps the request to the existing feature-generation, training, reporting, and inference entry points in `src/`. A worker thread supervises each scientific subprocess and records its output without blocking the HTTP server. Results remain associated with the job and can be downloaded as one ZIP archive. See [RUN.md](RUN.md) for the exact filesystem and commands.

## Methodological notes

The learning benchmark uses adaptive ordinary stratified cross-validation by default. When a replication unit is selected, the outer evaluation, nested tuning, and feature ablation use stratified group folds so related samples stay together. The published benchmark sequence is retained: three repetitions and both threshold-flag passes for every requested component count. Model selection is evaluated on held-out folds, while final saved models are refitted on all labelled samples using the chosen configuration. Dimensionality reduction and threshold-feature selection are fitted within training folds. Classification plots keep all-column, threshold-only, and SVD configurations separate, while confusion matrices use the all-column evaluation and display row-normalized percentages. Ablation ranks generated feature-table columns once using the complete labelled table, then evaluates increasing prefixes with the selected fold strategy. Aggregation-statistic columns remain independent ranked features. The dotted ablation marker is a descriptive plateau estimate (later observed scores within 0.01), not model selection.

The reported accuracy, weighted F1 results, class predictions, and confusion matrices should be interpreted together with class balance, replication design, and the number of independent biological experiments. High image counts do not compensate for too few independent experimental units.

Per-image scalar features are exported once, while slice measurements retain their aggregations. Both 2D imported-mask and 3D runs assemble complete feature tables. Mean thickness now counts filled voxels per occupied column, excluding internal gaps and empty columns. Roughness uses the filled column thickness and its whole-field mean, including zeros. Maximum height includes the full highest occupied voxel. `BiomassHeightP95_um` complements it with the slice-extent height containing 95% of biomass voxels. These definitions implement the requested comparison protocol without claiming exact agreement with every COMSTAT version. Feature schema 4 requires regenerated tables and retrained models for raw-image inference; original precomputed tables remain usable. Both image-based fractal features remain active: per-slice median-binarized boundary box counting (`FractalDimension`) and grayscale differential box counting (`FractalGrayscaleBoxDimension`). The additional `MaskFractalDimension` applies boundary box counting directly to the selected, size-filtered binary mask per slice, then aggregates across slices. Serial-threshold biomass measurements use per-slice `LayerArea_um2Thr*` (foreground pixels times calibrated pixel area) at every cutoff from 0.01 to 0.29; areas and stack-normalized percentages receive the usual slice aggregations. Whole-stack `BioVolumeThr*` is removed. `BiofilmExtentHeight_um`, `MeanThickness_COMSTAT_um`, `VerticalFillRatio`, and `FilledThicknessMean_um` are also removed as redundant or unsuitable features.

External measurements can be supplied as numeric columns in a complete Excel XLSX, CSV, TSV, or tab-separated TXT feature table, with one row per image, `sampleName`, and training `label`. Training no longer accepts image-settings JSON uploads. Image workflows still record generation settings automatically; models trained from uploaded tables require compatible precomputed inference tables containing the combined features. Newly saved models enforce complete feature columns at inference even without image-generation settings.
