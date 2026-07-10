
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img src="logo.png" alt="MicroICS logo" width="240">
  </picture>
</p>

# Image classification suite MicroICS

![Feature Generation](https://github.com/SkBlaz/biofilm-classification/actions/workflows/feature-generation.yml/badge.svg)
![Inference](https://github.com/SkBlaz/biofilm-classification/actions/workflows/inference.yml/badge.svg)
![Learning Benchmark](https://github.com/SkBlaz/biofilm-classification/actions/workflows/learning-benchmark.yml/badge.svg)
![Data Visualization](https://github.com/SkBlaz/biofilm-classification/actions/workflows/visualization.yml/badge.svg)
![Ruff](https://github.com/SkBlaz/biofilm-classification/actions/workflows/ruff.yml/badge.svg)

Welcome to Image classification suite MicroICS, a Python-based software for efficient classification of 3D biofilm images. The following links lead to two main documentation components, software itself (engineering aspects, running it), and algorithmic aspects.

1. [Running the software](RUN.md)

2. [Approach overview](APPROACH.md)

3. [Continuous Integration](.github/CI.md)

```
TY  - JOUR
AU  - Janež, Nika
AU  - Škrlj, Blaž
AU  - Osojnik, Aljaž
AU  - Ladányi, Márta
AU  - Breskvar, Martin
AU  - Petković, Matej
AU  - Kokot, Boštjan
AU  - Čotar, Petra
AU  - Papić, Bojan
AU  - Golob, Majda
AU  - Peternel, Tjaša
AU  - Sabotič, Jerica
PY  - 2026
DA  - 2026/07/08
TI  - MicroICS: predictive phenotyping of Listeria monocytogenes biofilms from three-dimensional structural features
JO  - npj Biofilms and Microbiomes
AB  - Biofilms underpin microbial survival, yet their three-dimensional structure remains difficult to quantify systematically. We present the Microbial Image Classification Suite (MicroICS), an open-source framework for predictive phenotyping of microbial communities from 3D biofilm images. MicroICS consists of three independent but interoperable modules: feature extraction from 3D biofilm images, machine learning-based classification, and an inference module for applying trained models to new, unseen images. Each module can be run independently, and the classification module accepts externally generated features, enabling integration with existing quantitative image analysis tools. As a pilot study, we demonstrate the framework using eight epidemiologically diverse Listeria monocytogenes strains, with strain differentiation and trait-based grouping as proof-of-principle tasks. MicroICS extracted over 2,700 quantitative structural features from Syto 9-labelled biofilm images. An optimised random forest model achieved human baseline-level accuracy in strain classification on previously unseen images, including biofilms experimentally perturbed by food extracts. BiofilmQ-derived features incorporated into the classification module yielded higher accuracy with fewer features than either tool alone, confirming the framework’s extensibility. Extending the framework to clonal complex-based clinical grouping demonstrates utility beyond strain identity. MicroICS is applicable to any organism or condition for which suitable 3D biofilm images can be obtained.
SN  - 2055-5008
UR  - https://doi.org/10.1038/s41522-026-01083-8
DO  - 10.1038/s41522-026-01083-8
ID  - Janež2026
ER  - 
```
