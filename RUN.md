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
- **Train models**: upload a complete labelled feature table and write rankings, reports, and models below `results/`.
- **Inference**: choose an earlier persisted training job and upload either images or a compatible precomputed feature table (only one of the two is required). Predictions are written below `inference/`.
- **All together**: upload training and inference images and run generation, training, and inference sequentially.

Completed output is available from the job’s **Download results ZIP** link. A failed computation changes only that job to `failed`; the server remains available for another request.

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
src/feature_ranking_lite.py           ranking, benchmarking, and saved models
src/inference.py                      image or precomputed-table inference
src/visualize_benchmark.py            benchmark reports
```

The GUI execution adapter is `gui/execution.py`; it prepares job-local paths and argument lists while leaving algorithms in `src/`.

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
