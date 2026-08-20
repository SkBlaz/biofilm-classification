<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img src="logo.png" alt="MicroICS logo" width="240">
  </picture>
</p>

# MicroICS

MicroICS is a tool for linking the 3D structure of biofilms to biological properties or questions of interest. It does this by measuring thousands of structural features from 3D biofilm images and using machine learning to identify which of them relate to the biological grouping linked to the question in interest. It also enables you to apply what it learns to classify new images.

The tool is built as three interoperable modules that can be used together as a pipeline or individually:
1. Feature generation – measures structural features from 3D biofilm images and writes them to a table.
2. Model training – trains and compares machine learning classifiers on those features, identifies the structural features that separate your groups, and saves the trained models.
3. Inference – applies a saved model to new, unseen images to predict their labels.

MicroICS can be installed in two ways: from the command line, as described below, or using Docker Desktop, which requires no command line setup and is explained in the accompanying guide resources/installation_and_running_MicroICS.

## Standalone web application

The production image contains the web interface, MicroICS source, Python, and the complete scientific environment. Docker is the only host-side runtime requirement; Python, Conda, Git, Docker Compose, repository mounts, and Docker-socket access are not required to run a built image.

Build and start it from a source checkout:

```bash
docker build -t microics .
docker run --rm -p 8765:8765 microics
```

Open <http://localhost:8765>. The browser uploads inputs into an isolated job, computation runs inside that same container, and the completed job can be downloaded as a ZIP file. To keep jobs, models, and results after removing the container, optionally attach a named volume:

```bash
docker run --rm -p 8765:8765 -v microics-data:/data microics
```

The application stores each run under `/data/jobs/<job-id>/` and never needs access to host filesystem paths. The image accepts `.tif` and `.tiff` uploads and compatible pre-generated `.tsv`/`.txt` feature tables.

For convenience on Unix-like development systems, `bash run_gui.sh` builds and starts the same standalone image. `docker-compose.yml` is an optional equivalent for development; the application does not invoke or depend on Compose.

## Workflows

The GUI supports the real MicroICS stages:

1. generate features from labelled or unlabelled images;
2. train one learner or compare all learners from a complete feature table;
3. run inference from uploaded images or a compatible precomputed data file using models from a completed training job;
4. run labelled feature generation, training, and image inference together.

Before generating features, enter the X, Y, and Z voxel dimensions from the microscope metadata. GUI and command-line learning use the published main-branch benchmark protocol so newly generated results remain directly comparable with the published results.

## Development validation

With the Python dependencies installed locally, run:

```bash
PYTHONPATH=src python run_tests.py
python -m ruff check .
python -m ruff format --check .
```

See [RUN.md](RUN.md) for the runtime and API details and [APPROACH.md](APPROACH.md) for the methodology.

## Citation

Janež, Škrlj, Osojnik et al., “MicroICS: predictive phenotyping of *Listeria monocytogenes* biofilms from three-dimensional structural features,” *npj Biofilms and Microbiomes* (2026). DOI: [10.1038/s41522-026-01083-8](https://doi.org/10.1038/s41522-026-01083-8).

The repository also includes [CITATION.cff](CITATION.cff) for GitHub’s **Cite this repository** feature.
