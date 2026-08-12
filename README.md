<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img src="logo.png" alt="MicroICS logo" width="240">
  </picture>
</p>

# MicroICS

MicroICS turns 3D biofilm TIFF images into quantitative structural features, compares machine-learning classifiers, saves trained models, and predicts labels for new images.

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

Before generating features, enter the X, Y, and Z voxel dimensions from the microscope metadata. For model evaluation, select the experimental replication unit that must remain together in each cross-validation fold. The GUI checks date, well, and imaging-position feasibility from the uploaded training data and explains why statistically invalid choices are unavailable.

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
