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
- **Inference**: choose an earlier persisted training job and upload either images or a compatible precomputed feature table. Predictions are written below `inference/`.
- **All together**: upload training and inference images and run generation, training, and inference sequentially.

Completed output is available from the job’s **Download results ZIP** link. A failed computation changes only that job to `failed`; the server remains available for another request.

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

## Input contracts

Labelled image filenames must retain the naming convention expected by `src/input_validation.py`. Unlabelled inference images do not require embedded class labels.

A complete training table is tab-separated and contains `sampleName`, `label`, and numeric feature columns. A compatible inference table contains `sampleName` and the numeric columns required by the selected saved model. Generated feature columns are retained even when they contain zero, `NaN`, or infinite values; the learning and inference loaders apply the established value-level imputation consistently.

Voxel sizes are measured in micrometres. Select the actual acquisition values in the GUI. The replication unit (`position`, `well`, `plate`, or `date`) controls which related images remain grouped during model evaluation.

## Local development checks

Run from the repository root after installing `src/requirements.docker.txt` and Ruff:

```bash
PYTHONPATH=src python run_tests.py
python -m ruff check .
python -m ruff format --check .
```
