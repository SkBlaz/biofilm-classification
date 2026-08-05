<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img src="logo.png" alt="MicroICS logo" width="240">
  </picture>
</p>

# MicroICS

[![Feature Generation](https://github.com/SkBlaz/biofilm-classification/actions/workflows/feature-generation.yml/badge.svg)](https://github.com/SkBlaz/biofilm-classification/actions/workflows/feature-generation.yml)
[![Inference](https://github.com/SkBlaz/biofilm-classification/actions/workflows/inference.yml/badge.svg)](https://github.com/SkBlaz/biofilm-classification/actions/workflows/inference.yml)
[![Learning Benchmark](https://github.com/SkBlaz/biofilm-classification/actions/workflows/learning-benchmark.yml/badge.svg)](https://github.com/SkBlaz/biofilm-classification/actions/workflows/learning-benchmark.yml)
[![Data Visualization](https://github.com/SkBlaz/biofilm-classification/actions/workflows/visualization.yml/badge.svg)](https://github.com/SkBlaz/biofilm-classification/actions/workflows/visualization.yml)
[![Ruff](https://github.com/SkBlaz/biofilm-classification/actions/workflows/ruff.yml/badge.svg)](https://github.com/SkBlaz/biofilm-classification/actions/workflows/ruff.yml)

MicroICS turns 3D biofilm TIFF images into quantitative structural features, compares machine-learning classifiers, saves trained models, and predicts labels for new images. A local graphical interface guides the normal sequence:

1. generate features from labelled or unlabelled images;
2. train one learner or compare all learners;
3. run inference on new images.

The scientific Python stack runs inside Docker, so it does not need to be installed manually on the host computer.

## What you need

| Tool | Why it is needed |
|---|---|
| [Docker](https://docs.docker.com/get-docker/) | Runs the image-processing and machine-learning environment. |
| A web browser | Displays the local MicroICS interface. |
| Git (optional) | Downloads the repository and makes later updates easy. A ZIP download works without Git. |
| Python 3.11+ | Runs the small local interface. The Windows launcher installs it when missing. |

For an institutional computer, check software-installation and virtualization permissions before a workshop. Docker Desktop currently requires at least 8 GB RAM on Windows; consult the [official Windows requirements](https://docs.docker.com/desktop/setup/install/windows-install/#system-requirements). The first Docker build downloads several gigabytes and can take a while.

## Windows: easiest installation

No terminal experience or separate Python-library commands are required.

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/). Use the recommended WSL 2 setup. If asked for architecture, choose **x86_64/AMD64** for an x64 system and **Arm** for an ARM-based system; Windows Settings → System → About → System type shows which computer you have.
2. Start Docker Desktop once, accept its terms, and wait until it says the engine is running. Docker may ask Windows to enable WSL 2 or restart.
3. At the top of this GitHub page, select **Code → Download ZIP**, extract the ZIP, and open the extracted `biofilm-classification` folder.
4. Double-click **`MicroICS.bat`**.

The single launcher:

- creates or updates a **MicroICS** desktop shortcut automatically;
- installs Python 3.12 through Windows Package Manager if Python is missing;
- creates a private interface environment and installs its two small packages;
- offers the official Docker Desktop installer when Docker is missing;
- checks that Docker is running, then opens MicroICS in the browser.

Windows may show an administrator or security prompt while installing software. If the launcher reports that Docker is not ready, wait for Docker Desktop to finish starting and run `MicroICS.bat` again. On managed computers where `winget` or installations are blocked, ask IT to install Docker Desktop and Python 3.11 or newer.

### Windows with Git (optional)

Git is useful when you want to update MicroICS later. Open PowerShell and run:

```powershell
winget install --exact --id Git.Git --source winget
git clone https://github.com/SkBlaz/biofilm-classification.git
cd biofilm-classification
.\MicroICS.bat
```

Later, update with `git pull` from the same folder. `docker --version` and `git --version` can be used to check those installations.

## Linux installation (Ubuntu/Debian)

Open a terminal with <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>T</kbd>. These commands install Git, Python support for the interface, and Docker Engine:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker "$USER"
```

The Docker convenience script is intended for development machines; review [Docker's limitations and distribution-specific alternatives](https://docs.docker.com/engine/install/) if required by local IT policy. Membership in the `docker` group grants root-equivalent Docker access. Sign out and back in (or reboot) after adding the group, then run:

```bash
git clone https://github.com/SkBlaz/biofilm-classification.git
cd biofilm-classification
bash run_gui.sh
```

The launcher creates a private interface environment and installs its local packages on first use. Do not run it with `sudo`.

## macOS installation

This route follows the vendor installation methods but has not yet been validated by the MicroICS project team on physical Mac hardware.

1. Open Terminal and check for Homebrew with `brew --version`. If it is missing, install it using the command on [brew.sh](https://brew.sh/).
2. Install the required tools and download MicroICS:

```bash
brew install git python
brew install --cask docker
git clone https://github.com/SkBlaz/biofilm-classification.git
cd biofilm-classification
```

3. Open Docker from Applications and wait until it reports that Docker is running.
4. Start MicroICS:

```bash
bash run_gui.sh
```

## First run

MicroICS opens at <http://127.0.0.1:8765>. Choose **Run sample images** to check the installation. The first analysis run builds the Docker image; later starts are faster.

Before generating features, enter the X, Y, and Z voxel dimensions from the microscope acquisition metadata. The defaults are 0.13, 0.13, and 0.5 µm. For model evaluation, choose the experimental replication unit that should remain together in each cross-validation fold.

Use **Stop** to interrupt the current Docker step while keeping the run log. Use **Reset run** to terminate active work, clear the interface state, and return to Ready; it does not delete files already written to the selected output folder.

## Documentation

- [How MicroICS works and what each GUI choice means](APPROACH.md)
- [Detailed GUI, Docker, input-table, and command-line reference](RUN.md)
- [Continuous-integration checks](.github/CI.md)
- [Bot and contributor onboarding](AGENTS.md)

## Citation

Nika Janež et al., “MicroICS: predictive phenotyping of *Listeria monocytogenes* biofilms from three-dimensional structural features,” *npj Biofilms and Microbiomes* (2026). DOI: [10.1038/s41522-026-01083-8](https://doi.org/10.1038/s41522-026-01083-8).

The repository also includes [CITATION.cff](CITATION.cff) for GitHub’s **Cite this repository** feature.
