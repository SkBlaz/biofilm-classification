# Running MicroICS

## Standalone container

Build the one production/runtime image:

```bash
docker build -t microics .
```

Start the web application:

```bash
docker run --rm -p 8765:8765 microics
```

Open <http://localhost:8765>. The liveness check is available at <http://localhost:8765/api/health>.

The image starts `gui/app.py` on `0.0.0.0:8765`. The GUI directly starts the existing scripts under `src/` inside the same container. It does not invoke Docker or Docker Compose and does not require a source mount or access to `/var/run/docker.sock`.

To persist runtime data, attach a volume to `/data`:

```bash
docker run --rm -p 8765:8765 -v microics-data:/data microics
```

Compose remains an optional development shortcut only:

```bash
docker compose up --build
```

## Browser workflow

Inputs are supplied with the browser, rather than by entering host paths. Image uploads accept `.tif` and `.tiff`; compatible precomputed feature tables accept `.tsv` and `.txt`.

Available operations are:

- **Generate labelled features**: upload labelled TIFFs and produce `results/datafile.tsv`.
- **Generate unlabelled features**: upload inference TIFFs and produce `results/unknown_features.tsv`.
- **Train models**: upload a complete labelled feature table in Excel (`.xlsx`, first worksheet), CSV, TSV, or tab-separated TXT format and write rankings, reports, and models below `results/`.
- **Inference**: choose an earlier persisted training job and upload either images or a compatible precomputed feature table (only one of the two is required). Predictions are written below `inference/`.
- **All together**: upload training and inference images and run generation, training, and inference sequentially.

Completed output is available from the job’s **Download results ZIP** link. A failed computation changes only that job to `failed`; the server remains available for another request.

Choose **2D images** for one complete feature package, or **3D stacks** to select basic metrics, object features, advanced geometry, intensity, grayscale geometry/texture, serial thresholds, or All. Deselected groups are not calculated. Do not mix single-slice images and stacks in one job. Physical lengths use micrometres (µm), areas µm², volumes µm³, and densities their reciprocal units.

The additional **Segmentation and threshold QC** section uses the images uploaded for the selected workflow. Choose Automatic (Otsu, robust background, three-class Otsu, BEM), Manual / tuned threshold, or Import masks. Only relevant threshold controls appear. COMSTAT2 (`intensity >= cutoff`) boundaries are fixed; there is no boundary selector or representative-image map. **Run segmentation and QC only** produces review artifacts without computing features or training models, and accepts arbitrary TIFF filenames. Tune a small representative upload first, then run the full batch with the same settings.

All segmentation uses an internally generated 8-bit TIFF. Existing unsigned 8-bit images remain unchanged. Other images use OME `SignificantBits` when available; without intensity metadata, the minimum and maximum pixel values across the complete stack define the conversion range. This measures the observed range rather than guessing the acquisition bit depth. Constant metadata-free images convert to zero. Conversion applies one range to the complete stack: `floor(255 * clip((raw - lower) / (upper - lower), 0, 1) + 0.5)`. Bounds, source, rounding rule, and clipped fraction are recorded in JSON. CLI users can override the acquisition maximum with `--conversion-max 4095` or restore storage-range conversion with `--conversion-range-mode storage`; saved settings record the mode. Download `segmentation/*_thresholding_uint8.tif` and tune that exact image in ImageJ. Manual units are 8-bit intensity **0–255**, or that intensity divided by 255 (**0–1**). Intensity features retain original source precision; explicit `RawIntensity*` columns accompany the legacy normalized summaries.

Three-class Otsu calculates two cutoffs and applies one: the lower includes dim intensity in biomass, the upper excludes it. Histogram Otsu's last background bin is advanced to the first foreground bin for COMSTAT2; automatic sensitivity multiplies the inclusive cutoff. Robust background follows [BiofilmQ v1.0.1](https://github.com/knutdrescher/BiofilmQ/tree/v1.0.1/includes): trim 5% from each intensity tail, then use fitted mean + 2 standard deviations. The standard deviation uses `ddof=1`, matching MATLAB's [uncensored normal fit](https://uk.mathworks.com/help/stats/fitdist.html). A zero cutoff is advanced above zero to exclude dark background. BEM follows the [published power-curve/slope criterion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6115396/), with adjustable tolerance (default 10%), a positive scale and negative power, and an actionable failure if fitting/convergence fails. BEM requires a zero-mode 8-bit histogram without saturated 255 voxels; conversion does not waive these assumptions. Three-class labels describe intensity only, not biological cell states.

Imported masks are grayscale TIFFs with the same pixel/voxel dimensions and axis order as their images: nonzero means biomass, or distinct positive integer labels identify objects. Name each mask like its image (`sample.tif`) or add `_mask` (`sample_mask.tif`); matching must be unique. Thresholding is skipped. Binary masks use connectivity to identify objects; labelled masks preserve distinct identities even when touching, with labels compacted internally. Connectivity defaults to 26 (faces, edges, corners); 18 includes faces/edges and 6 only faces. Minimum area/volume defaults to zero; size filtering is applied before all mask measurements and QC, so raising it removes noise from the actual mask. High values can remove real small cells. For 3D masks the volume cutoff applies; the area cutoff applies to 2D inputs.

Each image provides QC in `feature_generator/segmentation/`: a full biomass mask TIFF, a numeric object-label TIFF, a reproducible random-colour RGB object-label TIFF, a full-stack RGB overlay TIFF, a maximum-projection preview, an intensity histogram, a threshold curve, and `*_segmentation.json`. Single-slice inputs produce one-slice TIFFs. Separate first/middle/top PNGs are no longer exported. Imported masks have no selected cutoff or threshold curve. Histogram and threshold curve mark both the base automatic threshold and the sensitivity-adjusted cutoff, including when they coincide; the curve also marks biomass after size filtering. The border QC fraction is the number of foreground voxels on image edges divided by all foreground voxels, excluding the singleton Z axis for 2D. Object and pore distributions remain CSVs. Large spherical-density calculations use FFT convolution with integer count rounding. Advanced geometry includes digital local thickness: the largest voxel-centred sphere covering each biomass voxel, with radii measured to background voxel centres using calibrated distances; this uses a digital-grid convention rather than a continuous surface mesh. Radius sampling uses the finest voxel spacing, bounding diameter underestimation to less than twice that spacing; `LocalThicknessRadiusStep_um` records it.

Per-image scalar measurements are exported once through `ImageFeaturesCustomAlgos.tsv` and `SUMMARYImageFeaturesCustomAlgos.tsv.txt`; they do not receive slice-level min/mean/standard-deviation aggregations. Slice measurements retain their aggregations. Both 2D imported-mask and 3D workflows produce `datafile.tsv` (or `unknown_features.tsv` for unlabelled jobs). Column heights now stop at their own highest occupied slice, fixing mean COMSTAT thickness and roughness. COMSTAT heights retain the zero-based `z * dz` convention; extent height uses `(z + 1) * dz`, and compacted thickness counts occupied voxels. True maxima can still equal the stack height when some biomass reaches the top. `BiomassHeightP95_um` adds the height containing 95% of biomass voxels, measured from the substrate using slice extent, so sparse top-layer signal has less influence. Regenerate tables and retrain models to use the corrected measurements and scalar feature schema.

`run_summary.json` records GUI parameters, validation, and outcome. Repeated QC-only runs replace their previous segmentation artifacts and write `segmentation_settings.json`, preserving existing trained-table settings and model provenance. `feature_generation_settings.json` records reproducible image settings beside generated tables, and saved model metadata carries those settings. When uploading an image-derived table for a separate training job, upload its matching settings JSON with **Choose matching image settings**; external tables without that provenance can still train and use compatible precomputed inference tables. Raw-image inference restores saved segmentation, conversion, voxel dimensions, and feature groups; imported-mask inference still requires new masks paired with the inference images. GUI model jobs without saved generation settings require a compatible feature table. This changes the segmentation/feature schema: retrain models or provide a table generated with the model's original protocol; column names alone do not establish scientific compatibility. Missing columns in models with saved generation settings fail rather than being replaced with zeros.

The **Trained models** dropdown on the Inference step lists each training job by creation date and learner(s) used (for example "Aug 15 2026, 08:03 · All learners (5 models)"), not just its job ID, so you can pick the correct one at a glance. Inference automatically loads and evaluates every model saved in the chosen job — if that job was trained with **All learners**, every learner's predictions are reported side by side, so there is no separate "all learners" toggle at the inference step.

### Participant FAQ

- **Importing a pre-trained model:** open **Inference**, choose **Import .joblib model**, and select a model exported by MicroICS. The import creates a separate model job; select it in **Trained models** and provide either new images or a compatible feature table. If available, keep the matching `*_metadata.joblib` beside the model when transferring a complete job directory. A model is only compatible with the same feature names, preprocessing, and voxel dimensions used during training.
- **Deleting old models:** select the model job in **Trained models** and click **Delete selected job**. This removes that job's models, inputs, outputs, and logs from the persisted data volume; it cannot be undone from the GUI.
- **New job and downloads:** click **New job** before starting a genuinely separate analysis. This keeps uploads and outputs isolated by job. Existing jobs are not overwritten, and each job's ZIP contains only that job's output.
- **Threshold-derived columns:** the classification bar labelled **Threshold-derived features only (subset)** is a deliberately separate comparison. It uses only columns created from thresholded measurements; it is useful for an ablation/comparison question, but it is not the default full-feature model and can be ignored when that hypothesis is not relevant.
- **RF ablation:** the RF ranks all generated columns once, then evaluates increasing prefixes (1, 21, 41, …) with ordinary stratified folds. The dotted line in `ablation_rf.pdf` marks the first tested count whose later observed accuracies remain within 0.01 accuracy points; it is an interpretation aid, not an automatically selected optimum. A flat curve does not prove that the omitted features are biologically unimportant.
- **Feature-value plots:** small gray dots are individual measurements. The orange diamonds are class means; boxplots show the median and interquartile range. Outliers are not drawn as large ambiguous circles.
- **Feature compatibility:** do not assume any feature table can be used with any saved model. The table must contain `sampleName` and the exact numeric feature columns expected by the model; features derived from different image channels, voxel sizes, segmentation, or software versions may be scientifically or technically incompatible.
- **Inference explanations:** the `explanations/` folder contains the all-class beeswarm, class-specific beeswarms, SHAP CSV files, overall feature importance, and compact per-feature importance plots for each class. SHAP values show contribution to a model output, not biological causation.

### Sharing changes and extra interpretation scripts

To propose a code change, create a branch on GitHub, commit the change, push the branch, open **New pull request**, choose the repository's `main` branch as the base, describe what changed and how it was tested, then request Blaž as reviewer. Keep analysis scripts that are reusable parts of MicroICS in the repository (for example under `scripts/` or `src/` with a short README); keep one-off participant notebooks, large data, and generated results in the project resources or an external archive, not in the source tree.

Choosing a feature table selects the input only. No output folder needs to be selected: generated files appear in **Inspect results** in the browser, and **Download results ZIP** saves a copy to the browser’s configured download location. Internally, the job keeps them under `/data/jobs/<job-id>/output/`.

## Runtime filesystem

Each browser run receives an unpredictable UUID and owns this structure:

```text
/data/jobs/<job-id>/
├── input/
│   ├── training-images/
│   ├── inference-images/
│   └── feature-files/
├── work/
│   ├── pipeline.log
│   └── inference-features/
├── output/
│   ├── results/
│   └── inference/
└── job.json
```

Uploaded names are reduced to a filename and validated by extension before a destination inside the job input directory is created. Existing files receive a unique suffix, so uploads cannot traverse outside a job or silently overwrite another input.

## Direct scientific entry points

The GUI is the default image command, but scientific modules remain reusable. From a source development environment, the primary entry points are:

```text
src/run_analysis.sh                   feature generation and legacy orchestration
src/feature_generator.py              per-image features and segmentation QC
src/feature_ranking_lite.py           ranking, benchmarking, and saved models
src/inference.py                      image or precomputed-table inference
src/visualize_benchmark.py            benchmark reports
```

The GUI execution adapter is `gui/execution.py`; it prepares job-local paths and argument lists while leaving algorithms in `src/`.

For direct per-image generation, use `--segmentation-approach automatic|manual|import`, `--threshold-method otsu|manual|multi_otsu|bem|robust_background`, `--threshold-scale uint8|stack_normalized`, and `--threshold VALUE`. A manual value with no explicit approach selects manual segmentation, including when the requested method is Otsu; explicit Automatic plus a manual cutoff is rejected. The deprecated `raw` scale aliases 8-bit units, not higher-bit source intensities. Use `--conversion-max 4095` to override automatic observed-range conversion for a known 12-bit acquisition, `--masks-dir PATH` for matching masks, `--image-dimension auto|2d|3d`, `--feature-groups basic,objects,advanced,intensity,geometry,serial` (or `all`), and `--qc-only` for segmentation review. Three-class Otsu has `--dim-class foreground|background`; manual second cutoffs and representative maps were removed. Automatic methods accept `--threshold-sensitivity` except BEM, whose control is `--bem-tolerance`. Connectivity and size filters use `--connectivity-3d`, `--minimum-object-area-um2`, and `--minimum-object-volume-um3`.

Benchmark plots can be regenerated without repeating model training:

```bash
PYTHONPATH=src python src/visualize_benchmark.py /path/to/results
```

The command reads existing `classification_*.tsv` and `ablation_ranking_all.tsv` files and writes PDFs under the result folder's `visualizations/` directory.

## Input contracts

Labelled image filenames must retain the naming convention expected by `src/input_validation.py`. Unlabelled inference images do not require embedded class labels.

A complete training table is tab-separated and contains `sampleName`, `label`, and numeric feature columns. A compatible inference table contains `sampleName` and the numeric columns required by the selected saved model. Generated feature columns are retained even when they contain zero, `NaN`, or infinite values; the learning and inference loaders apply the established value-level imputation consistently.

Voxel sizes are measured in micrometres. Select the actual acquisition values in the GUI. Learning intentionally follows the published main-branch evaluation protocol: ordinary stratified cross-validation, three seeded benchmark repetitions, and the original generated-column ablation. Training tables therefore require `sampleName` and `label`, but `sampleName` does not need to encode a selectable replication unit.

## Local development checks

Run from the repository root after installing `src/requirements.docker.txt` and Ruff:

```bash
PYTHONPATH=src python run_tests.py
python -m ruff check .
python -m ruff format --check .
```

Shell scripts are pinned to LF with `.gitattributes`. Docker also removes carriage returns from copied `src/*.sh` scripts, protecting existing Windows checkouts. Rebuild the image after updating; changing global Git settings is unnecessary.

Feature spreadsheets use one row per image: `sampleName`, `label` for training, then numeric feature columns. Add external measurements as extra columns in the complete table. The optional JSON upload contains image-generation settings, not external feature values.
