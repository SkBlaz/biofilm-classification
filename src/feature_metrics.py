"""Calibrated segmentation and biofilm geometry metrics.

Arrays use the axis order ``(z, y, x)``. COMSTAT-compatible heights use
layer-index coordinates, so layer zero is at height zero.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def otsu_threshold(image: np.ndarray) -> float:
    """Return ImageJ AutoThresholder-compatible Otsu for integer histograms."""
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("Cannot threshold an empty image")
    if not np.isfinite(values).all():
        raise ValueError("Image intensities must be finite")
    if np.min(values) == np.max(values):
        return float(np.nextafter(values.flat[0], np.inf))
    if np.issubdtype(values.dtype, np.integer) and np.min(values) >= 0 and np.max(values) <= 65535:
        histogram = np.bincount(values.astype(np.int64, copy=False).ravel(), minlength=int(values.max()) + 1)
        gray_levels = np.arange(histogram.size, dtype=float)
    else:
        histogram, edges = np.histogram(values, bins=256)
        gray_levels = (edges[:-1] + edges[1:]) / 2
    total_pixels = float(histogram.sum())
    if total_pixels == 0:
        raise ValueError("Cannot threshold an empty image")
    probabilities = histogram.astype(float) / total_pixels
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * gray_levels)
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    numerator = (total_mean * cumulative_probability - cumulative_mean) ** 2
    between_class_variance = np.divide(numerator, denominator, out=np.full_like(numerator, -np.inf), where=denominator > 0)
    # ImageJ keeps the first threshold when several bins share the maximum.
    threshold_index = int(np.argmax(between_class_variance))
    return float(gray_levels[threshold_index])


def multi_otsu_thresholds(image: np.ndarray) -> tuple[float, float]:
    """Return inclusive 8-bit cutoffs for background/dim/bright classes."""
    values = np.asarray(image)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Image must be non-empty and contain finite intensities")
    if np.unique(values).size < 3:
        raise ValueError("Three-class Otsu requires at least three intensity values")
    from skimage.filters import threshold_multiotsu

    cuts = threshold_multiotsu(values, classes=3)
    return float(cuts[0] + 1), float(cuts[1] + 1)


def multi_otsu_labels(image: np.ndarray, lower: float, upper: float, mode: str = "comstat2") -> np.ndarray:
    """Return background/dim/bright labels using the selected equality rule."""
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Multi-Otsu thresholds must be finite and strictly increasing")
    values = np.asarray(image)
    if mode == "comstat2":
        return np.where(values < lower, 0, np.where(values < upper, 1, 2)).astype(np.uint8)
    if mode == "comstat1":
        return np.where(values <= lower, 0, np.where(values <= upper, 1, 2)).astype(np.uint8)
    raise ValueError("mode must be 'comstat1' or 'comstat2'")


def multi_otsu_biomass_mask(labels: np.ndarray, dim_class: str) -> np.ndarray:
    """Collapse intensity classes to biomass without assigning biological states."""
    classes = np.asarray(labels)
    if dim_class == "foreground":
        return classes > 0
    if dim_class == "background":
        return classes == 2
    raise ValueError("dim_class must be 'background' or 'foreground'")


def robust_background_threshold(image: np.ndarray) -> float:
    """BiofilmQ 1.0.1: trim 5% per tail, mean + 2 sigma.

    Sigma is the normal-distribution maximum-likelihood estimate (ddof=0).
    All-zero trimmed backgrounds use the first positive 8-bit bin to avoid
    classifying zero-intensity background as biomass under COMSTAT2.
    """
    values = np.sort(np.asarray(image, dtype=float).ravel())
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Image must be non-empty and contain finite intensities")
    trim = int(np.floor(values.size / 20 + 0.5))
    capped = values[trim : values.size - trim] if trim else values
    cutoff = float(capped.mean() + 2 * capped.std())
    return cutoff if cutoff > 0 else float(np.nextafter(0.0, np.inf))


def bem_threshold(image: np.ndarray, change_tolerance: float = 0.10) -> int:
    """Select an 8-bit Biovolume Elasticity Method threshold.

    BEM fits ``a*T**b + c`` to voxel biovolume over thresholds 1..254, then
    chooses the first threshold where successive fitted-slope changes are
    below the published relative tolerance. The source histogram must have
    its mode at zero and no saturated 255 voxels.
    """
    from scipy.optimize import curve_fit

    values = np.asarray(image)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("BEM requires a non-empty finite image")
    if values.dtype != np.uint8:
        raise ValueError("BEM requires an unsigned 8-bit image stack")
    quantized = values
    histogram = np.bincount(quantized.ravel(), minlength=256)
    if histogram[0] != histogram.max():
        raise ValueError("BEM requires the histogram mode at intensity zero")
    if histogram[255] > 0:
        raise ValueError("BEM is not valid for stacks containing saturated 255 voxels")
    if not math.isfinite(change_tolerance) or not 0 < change_tolerance < 1:
        raise ValueError("change_tolerance must be between zero and one")

    thresholds = np.arange(1, 256, dtype=float)
    biovolume = np.cumsum(histogram[::-1])[::-1][1:256].astype(float)

    def power_curve(threshold, a, b, c):
        return a * np.power(threshold, b) + c

    parameters, _ = curve_fit(
        power_curve,
        thresholds,
        biovolume,
        p0=(max(float(biovolume[0]), 1.0), -0.5, 0.0),
        bounds=([0.0, -10.0, -np.inf], [np.inf, -0.001, np.inf]),
        maxfev=20000,
    )
    a, b, _ = parameters
    slopes = a * b * np.power(thresholds, b - 1)
    if not np.isfinite(slopes).all() or np.all(np.abs(slopes) < np.finfo(float).eps):
        raise ValueError("BEM fit has no finite nonzero slope; choose another method")
    relative_changes = np.abs(np.diff(slopes)) / np.maximum(np.abs(slopes[:-1]), np.finfo(float).eps)
    acceptable = np.flatnonzero(relative_changes < change_tolerance)
    if not acceptable.size:
        raise ValueError("BEM slope criterion was not reached")
    return int(thresholds[acceptable[0]])


def threshold_mask(image: np.ndarray, cutoff: float, mode: str = "comstat2") -> np.ndarray:
    """Segment foreground using the selected COMSTAT threshold boundary rule."""
    if not math.isfinite(float(cutoff)):
        raise ValueError("Threshold must be finite")
    if mode == "comstat1":
        return np.asarray(image) > cutoff
    if mode == "comstat2":
        return np.asarray(image) >= cutoff
    raise ValueError("mode must be 'comstat1' or 'comstat2'")


def comstat_metrics(mask: np.ndarray, voxel_size: tuple[float, float, float], include_geometry: bool = True) -> dict[str, object]:
    """Calculate core COMSTAT-style metrics for a ``(z, y, x)`` binary mask.

    ``voxel_size`` is ``(dx, dy, dz)`` in micrometres. Empty biomass-dependent
    measurements are NaN and ``NoBiomassDetected`` is true.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("mask must have shape (z, y, x)")
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must contain (dx, dy, dz)")
    dx, dy, dz = (float(value) for value in voxel_size)
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy, dz)):
        raise ValueError("voxel dimensions must be positive finite values")

    nz, ny, nx = binary.shape
    occupied_per_layer = binary.sum(axis=(1, 2), dtype=np.int64)
    layer_area_um2 = occupied_per_layer.astype(float) * dx * dy
    layer_occupancy_pct = occupied_per_layer.astype(float) * (100.0 / (nx * ny))
    total_voxels = int(binary.sum())
    has_biomass = total_voxels > 0
    volume_um3 = total_voxels * dx * dy * dz

    surface_area_um2 = surface_to_volume_um_inv = float("nan")
    mean_diffusion_distance_um = max_diffusion_distance_um = float("nan")
    if include_geometry:
        padded = np.pad(binary, 1, constant_values=False)
        face_changes_z = np.count_nonzero(padded[1:, 1:-1, 1:-1] != padded[:-1, 1:-1, 1:-1])
        face_changes_y = np.count_nonzero(padded[1:-1, 1:, 1:-1] != padded[1:-1, :-1, 1:-1])
        face_changes_x = np.count_nonzero(padded[1:-1, 1:-1, 1:] != padded[1:-1, 1:-1, :-1])
        surface_area_um2 = face_changes_z * dx * dy + face_changes_y * dx * dz + face_changes_x * dy * dz
        surface_to_volume_um_inv = surface_area_um2 / volume_um3 if has_biomass else float("nan")
        if has_biomass:
            padded_distance = ndimage.distance_transform_edt(np.pad(binary, 1, constant_values=False), sampling=(dz, dy, dx))
            diffusion_distances = padded_distance[1:-1, 1:-1, 1:-1][binary]
            mean_diffusion_distance_um = float(diffusion_distances.mean())
            max_diffusion_distance_um = float(diffusion_distances.max())
        else:
            mean_diffusion_distance_um = float("nan")
            max_diffusion_distance_um = float("nan")

    heights = np.zeros((ny, nx), dtype=float)
    if has_biomass:
        occupied_z = np.any(binary, axis=(1, 2))
        max_layer_index = int(np.flatnonzero(occupied_z)[-1])
        max_height_um = float(max_layer_index * dz)
        biofilm_extent_height_um = float((max_layer_index + 1) * dz)
        for z in range(nz):
            heights[np.any(binary[: z + 1], axis=0)] = z * dz
        mean_thickness_um = float(heights.mean())
        biomass_columns = heights[np.any(binary, axis=0)]
        mean_biomass_column_thickness_um = float(biomass_columns.mean()) if biomass_columns.size else 0.0
        compacted_um = float(binary.sum(axis=0).max() * dz)
        vertical_fill_ratio = compacted_um / biofilm_extent_height_um
        roughness = float(np.mean(np.abs(heights - mean_thickness_um)) / mean_thickness_um) if mean_thickness_um > 0 else float("nan")
        layer_indices = np.arange(nz, dtype=float)
        z_weights = occupied_per_layer.astype(float)
        z_com_um = float(np.dot(layer_indices * dz, z_weights) / total_voxels)
        z_spread_um = float(np.sqrt(np.dot((layer_indices * dz - z_com_um) ** 2, z_weights) / total_voxels))
    else:
        max_height_um = float("nan")
        biofilm_extent_height_um = float("nan")
        mean_thickness_um = float("nan")
        mean_biomass_column_thickness_um = float("nan")
        compacted_um = float("nan")
        vertical_fill_ratio = float("nan")
        roughness = float("nan")
        z_com_um = float("nan")
        z_spread_um = float("nan")

    return {
        "NoBiomassDetected": not has_biomass,
        "Biomass_um3_per_um2": volume_um3 / (nx * ny * dx * dy),
        "BiomassVolume_um3": volume_um3,
        "SurfaceArea_voxel_faces_um2": float(surface_area_um2),
        "SurfaceToVolumeRatio_um_inv": float(surface_to_volume_um_inv),
        "MeanDiffusionDistance_um": mean_diffusion_distance_um,
        "MaxDiffusionDistance_um": max_diffusion_distance_um,
        "MaxBiofilmHeight_COMSTAT_um": max_height_um,
        "BiofilmExtentHeight_um": biofilm_extent_height_um,
        "MeanThickness_COMSTAT_um": mean_thickness_um,
        "MeanThickness_BiomassColumns_um": mean_biomass_column_thickness_um,
        "MaxCompactedColumnThickness_um": compacted_um,
        "VerticalFillRatio": vertical_fill_ratio,
        "Roughness_RaStar": roughness,
        "SubstratumCoverage_pct": float(layer_occupancy_pct[0]) if nz else float("nan"),
        "LayerArea_um2": layer_area_um2,
        "LayerOccupancy_pct": layer_occupancy_pct,
        "BiomassCenterOfMassHeight_um": z_com_um,
        "BiomassVerticalSpread_um": z_spread_um,
    }


def connected_object_metrics_2d(
    mask: np.ndarray,
    voxel_size: tuple[float, float, float],
    min_area_um2: float = 0.0,
    object_labels: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure 8-connected 2D objects and anisotropic centroid nearest distances."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    dx, dy, _ = (float(value) for value in voxel_size)
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy)):
        raise ValueError("x/y voxel dimensions must be positive finite values")
    if not math.isfinite(min_area_um2) or min_area_um2 < 0:
        raise ValueError("min_area_um2 must be a non-negative finite value")

    labels = object_labels
    if labels is None:
        labels, _ = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    areas: list[float] = []
    centroids: list[tuple[float, float]] = []
    for object_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        coords = np.argwhere(labels[bounds] == object_id)
        if not coords.size:
            continue
        coords[:, 0] += bounds[0].start
        coords[:, 1] += bounds[1].start
        area = float(len(coords) * dx * dy)
        if area < min_area_um2:
            continue
        centroids.append((float(coords[:, 0].mean() * dy), float(coords[:, 1].mean() * dx)))
        areas.append(area)

    points = np.asarray(centroids, dtype=float).reshape((-1, 2))
    if len(points) > 1:
        from scipy.spatial import cKDTree

        distances = cKDTree(points).query(points, k=2)[0][:, 1]
    else:
        distances = np.full(len(points), np.nan, dtype=float)
    field_area = binary.shape[0] * binary.shape[1] * dx * dy
    return {
        "ObjectCount": int(len(areas)),
        "ObjectDensity_per_um2": float(len(areas) / field_area),
        "ObjectAreas_um2": np.asarray(areas, dtype=float),
        "ObjectCentroids_um": points,
        "NearestNeighborDistances_um": distances,
    }


def connected_object_metrics_3d(
    mask: np.ndarray,
    voxel_size: tuple[float, float, float],
    connectivity: int = 26,
    min_volume_um3: float = 0.0,
    object_labels: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure 3D biomass objects with explicit 6/18/26 connectivity."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("mask must have shape (z, y, x)")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be 6, 18, or 26")
    dx, dy, dz = (float(value) for value in voxel_size)
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy, dz)):
        raise ValueError("voxel dimensions must be positive finite values")
    if not math.isfinite(min_volume_um3) or min_volume_um3 < 0:
        raise ValueError("min_volume_um3 must be non-negative and finite")
    rank_connectivity = {6: 1, 18: 2, 26: 3}[connectivity]
    labels = object_labels
    if labels is None:
        labels, _ = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, rank_connectivity))
    counts = np.bincount(labels.ravel())[1:]
    volumes = counts.astype(float) * dx * dy * dz
    keep = (counts > 0) & (volumes >= min_volume_um3)
    object_ids = np.flatnonzero(keep) + 1
    volumes = volumes[keep]
    centers = ndimage.center_of_mass(binary, labels, object_ids.tolist()) if object_ids.size else []
    centers_um = np.asarray([(x * dx, y * dy, z * dz) for z, y, x in centers], dtype=float).reshape((-1, 3))
    if len(centers_um) > 1:
        from scipy.spatial import cKDTree

        distances = cKDTree(centers_um).query(centers_um, k=2)[0][:, 1]
    else:
        distances = np.full(len(centers_um), np.nan, dtype=float)
    field_volume = binary.size * dx * dy * dz
    return {
        "ObjectCount3D": int(len(volumes)),
        "ObjectDensity3D_per_um3": float(len(volumes) / field_volume),
        "ObjectVolumes3D_um3": volumes,
        "ObjectCentroids3D_um": centers_um,
        "NearestNeighborDistances3D_um": distances,
        "ObjectConnectivity3D": connectivity,
        "MinimumObjectVolume_um3": min_volume_um3,
    }


def internal_pore_metrics_3d(
    mask: np.ndarray,
    voxel_size: tuple[float, float, float],
    connectivity: int = 26,
) -> dict[str, object]:
    """Measure internal voids, excluding background connected to any volume face."""
    biomass = np.asarray(mask, dtype=bool)
    if biomass.ndim != 3:
        raise ValueError("mask must have shape (z, y, x)")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be 6, 18, or 26")
    dx, dy, dz = (float(value) for value in voxel_size)
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy, dz)):
        raise ValueError("voxel dimensions must be positive finite values")
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    background_labels, _ = ndimage.label(~biomass, structure=structure)
    border_ids = np.unique(
        np.concatenate(
            (
                background_labels[0].ravel(),
                background_labels[-1].ravel(),
                background_labels[:, 0, :].ravel(),
                background_labels[:, -1, :].ravel(),
                background_labels[:, :, 0].ravel(),
                background_labels[:, :, -1].ravel(),
            )
        )
    )
    internal = (~biomass) & ~np.isin(background_labels, border_ids)
    pore_labels, pore_count = ndimage.label(internal, structure=structure)
    pore_voxels = np.bincount(pore_labels.ravel())[1 : pore_count + 1]
    voxel_volume = dx * dy * dz
    pore_volumes = pore_voxels.astype(float) * voxel_volume
    biomass_volume = int(biomass.sum()) * voxel_volume
    internal_volume = float(pore_volumes.sum())
    biofilm_roi_volume = biomass_volume + internal_volume
    return {
        "InternalPoreCount3D": int(pore_count),
        "InternalPorosity3D": internal_volume / biofilm_roi_volume if biofilm_roi_volume else float("nan"),
        "InternalPoreVolumes_um3": pore_volumes,
        "InternalVoidFraction_pct": 100.0 * internal_volume / biofilm_roi_volume if biofilm_roi_volume else float("nan"),
    }


def internal_pore_metrics_2d(mask: np.ndarray, voxel_size: tuple[float, float, float]) -> dict[str, object]:
    """Measure enclosed 8-connected 2D voids, excluding border-connected background."""
    biomass = np.asarray(mask, dtype=bool)
    if biomass.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    dx, dy, _ = (float(value) for value in voxel_size)
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy)):
        raise ValueError("x/y voxel dimensions must be positive finite values")
    structure = ndimage.generate_binary_structure(2, 2)
    background_labels, _ = ndimage.label(~biomass, structure=structure)
    border_ids = np.unique(np.concatenate((background_labels[0], background_labels[-1], background_labels[:, 0], background_labels[:, -1])))
    internal = (~biomass) & ~np.isin(background_labels, border_ids)
    pore_labels, pore_count = ndimage.label(internal, structure=structure)
    pore_pixels = np.bincount(pore_labels.ravel())[1 : pore_count + 1]
    pore_areas = pore_pixels.astype(float) * dx * dy
    biomass_area = int(biomass.sum()) * dx * dy
    internal_area = float(pore_areas.sum())
    enclosed_roi_area = biomass_area + internal_area
    return {
        "InternalPoreCount2D": int(pore_count),
        "InternalPorosity2D": internal_area / enclosed_roi_area if enclosed_roi_area else float("nan"),
        "InternalPoreAreas_um2": pore_areas,
    }


def local_biomass_density(
    mask: np.ndarray,
    voxel_size: tuple[float, float, float],
    radius_um: float,
) -> dict[str, object]:
    """Return calibrated spherical-neighborhood density at biomass voxels."""
    biomass = np.asarray(mask, dtype=bool)
    if biomass.ndim != 3:
        raise ValueError("mask must have shape (z, y, x)")
    dx, dy, dz = (float(value) for value in voxel_size)
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy, dz)):
        raise ValueError("voxel dimensions must be positive finite values")
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be positive and finite")
    rz, ry, rx = (int(math.floor(radius_um / step)) for step in (dz, dy, dx))
    oz, oy, ox = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    footprint = (oz * dz) ** 2 + (oy * dy) ** 2 + (ox * dx) ** 2 <= radius_um**2 + np.finfo(float).eps
    if biomass.size * footprint.size > 1_000_000:
        from scipy.signal import fftconvolve

        kernel = footprint.astype(np.float32)
        # Counts are integers; remove FFT roundoff before density division.
        biomass_count = np.rint(fftconvolve(biomass.astype(np.float32), kernel, mode="same")).astype(float)
        total_count = np.rint(fftconvolve(np.ones(biomass.shape, dtype=np.float32), kernel, mode="same")).astype(float)
    else:
        biomass_count = ndimage.convolve(biomass.astype(float), footprint.astype(float), mode="constant", cval=0.0)
        total_count = ndimage.convolve(np.ones(biomass.shape, dtype=float), footprint.astype(float), mode="constant", cval=0.0)
    density_map = np.divide(biomass_count, total_count, out=np.zeros_like(biomass_count), where=total_count > 0)
    values = density_map[biomass]
    return {
        "LocalBiomassDensityMap": density_map,
        "LocalBiomassDensityRadius_um": radius_um,
        "LocalBiomassDensityMean": float(values.mean()) if values.size else float("nan"),
        "LocalBiomassDensityStd": float(values.std()) if values.size else float("nan"),
        "LocalBiomassDensityIQR": float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else float("nan"),
    }


def local_thickness_metrics(mask, dimensions):
    """Digital sphere diameters, with voxel-resolution radius sampling."""
    binary = np.asarray(mask, dtype=bool)
    spacing = tuple(reversed(dimensions))
    padded = np.pad(binary, 1)
    radii = ndimage.distance_transform_edt(padded, sampling=spacing)
    thickness = np.zeros(padded.shape, dtype=float)
    step = min(spacing)
    # Round radii down; diameter error is less than two finest voxels.
    maximum = int(np.floor(radii.max() / step + 1e-12))
    for radius in np.arange(maximum, 0, -1) * step:
        covered = ndimage.distance_transform_edt(radii < radius, sampling=spacing) <= radius
        thickness[(thickness == 0) & padded & covered] = 2 * radius
    values = thickness[1:-1, 1:-1, 1:-1][binary]
    return {
        "LocalThicknessMean_um": float(values.mean()) if values.size else float("nan"),
        "LocalThicknessMedian_um": float(np.median(values)) if values.size else float("nan"),
        "LocalThicknessMax_um": float(values.max()) if values.size else float("nan"),
        "LocalThicknessRadiusStep_um": step,
    }
