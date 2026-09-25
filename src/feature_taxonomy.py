"""Machine-readable feature grouping for generated MicroICS columns."""

from __future__ import annotations

FEATURE_GROUPS = (
    "2D substratum structure",
    "3D basic biofilm metrics",
    "3D advanced geometry",
    "Threshold-free intensity",
    "Threshold-free geometry and texture",
    "Serial and multi-threshold",
    "Metadata and QC",
)

_THRESHOLD_PREFIXES = (
    "counts(",
    "BioVolumeThr",
    "ThicknessThreshold",
    "RoughnessThreshold",
    "NoBiomassDetected",
    "Biomass_",
    "BiomassVolume",
    "MaxBiofilmHeight",
    "BiofilmExtentHeight",
    "MeanThickness",
    "MaxCompactedColumnThickness",
    "VerticalFillRatio",
    "Roughness_RaStar",
    "SubstratumCoverage",
    "SubstratumObject",
    "SubstratumBiomassArea",
    "SubstratumInternalPore",
    "SubstratumInternalPorosity",
    "ObjectCount3D",
    "ObjectDensity3D",
    "ObjectVolume3D",
    "ObjectNearestNeighbor3D",
    "InternalPore",
    "InternalPorosity",
    "LocalBiomassDensity",
    "SurfaceArea_",
    "SurfaceToVolume",
    "MeanDiffusion",
    "MaxDiffusion",
    "LayerArea",
    "LayerOccupancy",
    "SpreadingHorizontal",
    "SpreadingVertical",
    "SpreadingTotal",
)


def is_threshold_derived_feature(name: str) -> bool:
    """Return whether a generated feature depends on a binary threshold mask."""
    return str(name).startswith(_THRESHOLD_PREFIXES)


def classify_feature(name: str) -> str:
    """Assign one primary taxonomy group without inferring cell biology."""
    feature = str(name)
    if feature.startswith(("counts(", "Substratum", "Area2D", "ObjectCount2D", "ObjectArea2D")):
        return "2D substratum structure"
    if feature.startswith(("BioVolumeThr", "ThicknessThreshold", "RoughnessThreshold")):
        return "Serial and multi-threshold"
    if feature.startswith(
        ("SurfaceArea", "SurfaceToVolume", "MeanDiffusion", "MaxDiffusion", "InternalPore", "InternalPorosity", "LocalBiomassDensity")
    ):
        return "3D advanced geometry"
    if feature.startswith(
        (
            "Biomass_",
            "BiomassVolume",
            "MaxBiofilmHeight",
            "BiofilmExtentHeight",
            "MeanThickness",
            "MaxCompactedColumnThickness",
            "VerticalFillRatio",
            "Roughness_RaStar",
            "LayerArea",
            "LayerOccupancy",
            "ObjectCount3D",
            "ObjectDensity3D",
            "ObjectVolume3D",
            "ObjectNearestNeighbor3D",
            "Spreading",
        )
    ):
        return "3D basic biofilm metrics"
    if feature.startswith(("diff", "max", "med", "mean", "std", "min", "minProp", "q10", "q25", "q75", "q90")):
        return "Threshold-free intensity"
    if feature.startswith(("Homogeneity", "Homogenity", "Spreading", "GPT", "Fractal", "GLCM")):
        return "Threshold-free geometry and texture"
    return "Metadata and QC"


def feature_catalog(names) -> dict[str, list[str]]:
    """Return ordered column names grouped for GUI/docs/report consumers."""
    result = {group: [] for group in FEATURE_GROUPS}
    for name in names:
        text = str(name)
        if text in {"sampleName", "label"} or text.startswith("Unnamed:"):
            continue
        result[classify_feature(text)].append(text)
    return result
