# AGENTS.md

This file is the bot onboarding guide for this repository.

## 1) Purpose

Use this repository to process 3D biofilm images and run feature generation, ranking, learning benchmarks, visualization, and inference pipelines.

## 2) Repository map (quick orientation)

- `src/`: main Python pipeline code
- `src/visualizations/`: visualization modules
- `test_*.py`: unit tests
- `run_tests.py`: test runner
- `src/segmentation.py`: shared settings, 8-bit conversion, imported TIFF masks, and size filtering
- `src/segmentation_qc.py`: full-stack mask/RGB overlay/coloured labels, threshold plots, and exact thresholding TIFF exports
- `gui/execution.py`: per-job adapter from GUI requests to existing scientific entry points
- `run_gui.sh`: optional Docker build/run convenience launcher for Unix-like systems
- `src/requirements.docker.txt`: runtime Python dependencies used in Docker
- `.github/workflows/`: CI workflows (Docker workflows + Ruff)
- `RUN.md`: usage and Docker execution guide
- `APPROACH.md`: methodology overview

## 3) Local setup and validation commands

Run from repository root:

```bash
python -m pip install -r src/requirements.docker.txt
python -m pip install ruff
PYTHONPATH=src python run_tests.py
python -m ruff check .
python -m ruff format --check .
```

Notes:
- Shell scripts use LF via `.gitattributes`; Docker normalizes existing Windows `src/*.sh` copies.
- Segmentation uses a fixed whole-stack 8-bit conversion and COMSTAT2 boundaries. Metadata-free non-8-bit TIFFs use the whole-stack observed intensity range; OME SignificantBits and explicit CLI acquisition maxima take precedence. Record conversion range mode and tune the exported 8-bit TIFF in ImageJ.
- The GUI supports segmentation/QC-only runs, matching imported TIFF masks, and dimension-aware feature-package execution. Preserve original-precision intensity features and model generation settings. Physical-unit spans must bypass CSS uppercase transforms (µ can otherwise display like M).
- Repeated QC-only tuning replaces QC artifacts and records `segmentation_settings.json` without overwriting trained-table/model provenance.
- QC lives in `feature_generator/segmentation/`. Scalar measurements bypass slice aggregations via `ImageFeaturesCustomAlgos`; 2D imported-mask runs must still produce training tables. Border fractions ignore singleton Z in 2D. Thickness counts filled voxels over occupied columns; roughness uses filled thickness over the whole field; maximum height includes the full highest occupied voxel; `BiomassHeightP95_um` complements true maxima. Regenerate tables and retrain after this feature schema change.
- Rolling-ball preprocessing is removed from GUI, CLI, segmentation, and provenance. Otsu defaults to whole-stack sampling with optional first-N-slice substratum sampling. BEM is no longer selectable. Record threshold source in GUI/CLI/model provenance. Feature schema 4 blocks older image settings from raw-image inference; use original tables or retrain. Both image-based fractal dimensions remain active, with additional per-slice boundary box counting on the selected binary mask (`MaskFractalDimension`).
- QC omits numeric object-label TIFFs and projection QC PNGs, retaining RGB labels and full-stack overlays. Decimal size controls accept comma/dot and adjust by 0.01; Z spacing is ignored for 2D measurements.
- Complete feature tables accept XLSX (first worksheet), CSV, TSV, and TXT; `openpyxl` is required for Excel. Train models optionally joins a second measurement table by exact `sampleName`, preserving primary row order and exporting the combined `datafile.tsv`. Require unique nonempty keys, identical sample sets, distinct feature columns, and agreeing optional labels. Image-settings JSON uploads are removed; image workflows still save provenance automatically. Uploaded-table models require compatible precomputed inference tables. Reject duplicate headers before pandas can rename them, and preserve exact sampleName strings through training. Newly saved models set `strict_feature_columns` so inference rejects missing features even without image-generation settings. Failed table replacements block Run until retried or cleared.
- Train models defaults to the published ordinary stratified CV protocol. An optional replication unit (`date`, `well`, or `position`) enables grouped outer evaluation, nested tuning, and ablation; grouped input must pass feasibility checks using the training labels and parseable sample names. Saved model metadata and the job summary record the selected protocol.
- Serial-threshold ML biomass features are per-slice `LayerArea_um2Thr*` (foreground pixels times `dx * dy`) and normalized percentages, aggregated across slices; whole-stack `BioVolumeThr*` is removed. Do not export the redundant `BiofilmExtentHeight_um`, `MeanThickness_COMSTAT_um`, `VerticalFillRatio`, or `FilledThicknessMean_um` features.
- Result ZIP downloads stream directly to the HTTP response with ZIP64 support; never stage full download archives in job work directories. Unix launcher and GUI Compose bind `.microics-data/` to `/data`; RUN.md includes Windows Desktop mounts, existing-data recovery, and scoped Docker cleanup. Downloads retain original jobs. New job and the first run after a page reload offer selective, explicitly confirmed job cleanup; no jobs are preselected, and active jobs/current run inputs and models are protected. Job sizes are calculated only for the cleanup listing, never during state polling. Linux bind mounts must run as the host user/group (Unix launcher does this; Compose accepts MICROICS_UID/MICROICS_GID) so saved jobs remain removable by the host user.
- Tests import modules from `src`, so keep `PYTHONPATH=src`.
- CI runs Ruff checks and format checks on `main`/`master` pushes and PRs. The Feature Generation workflow also runs the complete unit suite in the built Docker runtime.

## 4) Change workflow for bots

1. Understand the issue and identify the smallest safe change.
2. Run baseline checks before editing (tests/lint as available).
3. Edit only directly relevant files.
4. Re-run targeted validation after edits.
5. Update this `AGENTS.md` if onboarding/process information changed.

## 5) What to avoid

- Do not change unrelated code or tests.
- Do not commit generated artifacts or large temporary outputs.
- Do not remove existing tests to make CI pass.

## 6) Main-branch bot update rule (required)

Whenever a bot-authored PR is merged to `main` (or `master`), that PR **must** update this `AGENTS.md` with any onboarding/process changes introduced by the PR.

If nothing in onboarding/process changed, add a short entry in the log below saying "Reviewed, no onboarding changes required."

## 7) Bot main-branch touch log

Keep newest entry at the top.

| Date (UTC) | Bot/Agent | PR/Commit | AGENTS.md update summary |
|---|---|---|---|
| 2026-10-29 | Copilot | current change | Preflight accepts both replication-option spellings used by the CLI and shell pipeline; grouped feasibility checks remain enforced. |
| 2026-10-29 | Codex | current change | Restored optional grouped cross-validation by date, well, or imaging position while preserving published stratified CV as the default. |
| 2026-10-28 | Codex | current change | Fixed ZIP downloads with direct streaming; added confirmed selective job cleanup, host-owned bind mounts, recovery guidance, and scoped Docker image/cache cleanup. |
| 2026-10-25 | Codex | current change | Follow-up verification: duplicate-header rejection, exact numeric sample identifiers through learning, strict feature-column metadata for table-trained models, and failed-upload recovery documented. |
| 2026-10-25 | Codex | current change | Final feedback: removed rolling ball and JSON upload, added safe two-table training joins and binary-mask fractality, restored per-slice threshold area aggregation, removed redundant features, and advanced schema to 4. |
| 2026-10-16 | Codex | current change | Fifth-round feedback: rolling-ball preprocessing, optional substratum Otsu, fractional size controls, leaner QC exports, BEM removal, filled thickness/roughness and voxel-extent heights, both fractal features, and schema-3 inference compatibility documented. |
| 2026-10-06 | Codex | current change | Compact segmentation controls, observed-range conversion, full-stack RGB QC, corrected column heights/2D border fraction, scalar table exports, Excel/CSV input, and aligned methodology documentation recorded. |
| 2026-10-02 | Codex | current change | LF shell checkout/build protection, shared 8-bit segmentation settings, mask import, slice QC, dimension-aware feature selection, model provenance, and Docker unit-suite CI documented. |
| 2026-08-17 | Codex | current change | Imported/deleted model jobs, ablation plateau marker, explicit feature-value means, per-feature SHAP plots, and participant FAQ documented. |
| 2026-08-15 | Copilot CLI | current change | GUI inference step now clarifies images/feature table are alternatives, trained-model dropdown shows creation date and learner(s)/model count instead of only a job ID, and inference help text explains it auto-evaluates every saved model (no separate "all learners" toggle needed at inference). |
| 2026-08-14 | Codex | current change | Random Forest and KNN hyperparameter search grids trimmed to eliminate HPO scaling bottlenecks on high-dimensional feature tables. |
| 2026-08-13 | Codex | current change | GUI and CLI learning are pinned to the published main-branch protocol: ordinary stratified CV, three seeded benchmark repetitions, duplicate threshold-flag component passes, and full-table generated-column RF ablation. |
| 2026-08-12 | Codex | current change | Standalone benchmark replotting, stable learner names, non-duplicated seeded evaluation, distinct normalized fixed-configuration confusion matrices, and explicit ablation column counts documented. |
| 2026-08-10 | Codex | current change | Standalone single-container GUI/pipeline image, direct per-job execution, explicit browser result delivery, and `/data` persistence documented. |
| 2026-08-05 | Codex | current change | Single self-preparing Windows launcher, private GUI dependencies, bounded benchmark report names, and cross-platform installation/workflow guidance documented. |
| 2026-08-04 | Codex | current change | GUI voxel-dimension propagation, visible logo treatment, and stop-versus-reset recovery behavior documented. |
| 2026-08-03 | Codex | current change | Feature-table picker, legacy generated-value imputation contract, convention-free unlabelled preflight, prominent learner scope, and bounded all-learner resources documented. |
| 2026-07-29 | Codex | current change | Sequential GUI workflow, complete external-table contract, imaging-position replication, Windows diagnostics, and writable generated-feature validation documented. |
| 2026-07-21 | Codex | current change | Unix GUI launcher now diagnoses sudo misuse and Docker socket access. |
| 2026-07-21 | Codex | current change | GUI preflight validation, grouped replication-aware benchmarking, resumable learner runs, organized reports, demo workflow, citation metadata, and UTF-8 Windows launcher documented. |
| 2026-07-10 | Codex | current change | Windows GUI launcher and icon shortcut workflow documented. |
| 2026-05-15 | @copilot | current PR | Initial detailed onboarding guide added. |
