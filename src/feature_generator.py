import argparse
import json
import logging
import math
import os
import warnings

import matplotlib.pyplot as plt
import multipagetiff as mtif
import numba
import numpy as np
import pandas as pd
from scipy.ndimage import label

from feature_metrics import (
    bem_threshold,
    comstat_metrics,
    connected_object_metrics_2d,
    connected_object_metrics_3d,
    internal_pore_metrics_2d,
    internal_pore_metrics_3d,
    local_biomass_density,
    multi_otsu_biomass_mask,
    multi_otsu_labels,
    multi_otsu_thresholds,
    otsu_threshold,
    robust_background_threshold,
    threshold_mask,
)
from input_validation import parse_image_name

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# create logger
logger = logging.getLogger("imGenLogger")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

CONNECTION_KERNEL = np.ones((3, 3), dtype=np.int32)

DEFAULT_VOXEL_DIMENSIONS = (0.13, 0.13, 0.5)


def read_voxel_dimensions(environ=None) -> tuple[float, float, float]:
    """Read positive X/Y/Z voxel dimensions in micrometres from the environment."""
    environ = os.environ if environ is None else environ
    dimensions = []
    for axis, default in zip(("X", "Y", "Z"), DEFAULT_VOXEL_DIMENSIONS):
        name = f"IMAGINE_VOXEL_SIZE_{axis}"
        try:
            value = float(environ.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        dimensions.append(value)
    return tuple(dimensions)


voxel_size_x, voxel_size_y, voxel_size_z = read_voxel_dimensions()
voxel_size = voxel_size_x * voxel_size_y * voxel_size_z
pixel_area = voxel_size_x * voxel_size_y


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gpt_fractal_dimension(Z):
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    Z = Z < np.median(Z)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def calculate_spatial_spreading(image_stack, dimensions=(voxel_size_x, voxel_size_y, voxel_size_z)):
    """Return physical x-y, z, and 3D standard-deviation spreads in µm."""
    coordinates = np.argwhere(np.asarray(image_stack) > 0)
    if coordinates.size == 0:
        return float("nan"), float("nan"), float("nan")
    dx, dy, dz = dimensions
    # Array order is z, y, x; convert each axis independently to µm.
    physical = coordinates.astype(float) * np.array([dz, dy, dx])
    sigma_z, sigma_y, sigma_x = np.var(physical, axis=0)
    horizontal = float(np.sqrt(sigma_x + sigma_y))
    vertical = float(np.sqrt(sigma_z))
    total = float(np.sqrt(sigma_x + sigma_y + sigma_z))
    return horizontal, vertical, total


def get_cell_count(intensity_matrix, threshold=0.3, visualizations=False):
    intensity_matrix = np.asarray(intensity_matrix)
    maximum = np.max(intensity_matrix) if intensity_matrix.size else 0
    if maximum <= 0:
        norm_mat = np.zeros_like(intensity_matrix, dtype=np.uint8)
    else:
        norm_mat = (intensity_matrix / maximum > threshold).astype(np.uint8)
    labeled, ncomponents = label(norm_mat, CONNECTION_KERNEL)

    if visualizations:
        plt.imshow(labeled)
        plt.savefig(f"snapshot{ncomponents}.png", dpi=300)
        plt.clf()

    return ncomponents, labeled


def get_transition_matrix(raw_image: np.ndarray, n_bins=None) -> np.ndarray:
    if n_bins is None:
        n_bins = 256
    bins = np.linspace(0, np.max(raw_image) + 1, n_bins - 1)
    binned_image = np.digitize(raw_image, bins)
    return transition_matrix_helper(binned_image, n_bins)


@numba.njit
def transition_matrix_helper(binned_image: np.ndarray, n_bins: int) -> np.ndarray:
    matrix = np.zeros((n_bins, n_bins))
    a, b, c = binned_image.shape
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for x in range(a):
        for y in range(b):
            for z in range(c):
                for dx, dy, dz in neighbors:
                    x1 = x + dx
                    y1 = y + dy
                    z1 = z + dz
                    if not (0 <= x1 < a and 0 <= y1 < b and 0 <= z1 < c):
                        continue
                    this = int(binned_image[x, y, z])
                    other = int(binned_image[x1, y1, z1])
                    matrix[this, other] += 1
                    matrix[other, this] += 1
    total = np.sum(matrix)
    if total == 0:
        return matrix
    return matrix / total


def get_homogenity(raw_3d_image: np.ndarray) -> float:
    """
    Implements the formula (5) from Beyenal et al.
    It is important to note that below, index = value
    """
    transition_matrix = get_transition_matrix(raw_3d_image)
    n_a, n_b = transition_matrix.shape
    a, b = np.indices((n_a, n_b))
    return np.sum(transition_matrix / (1 + (a - b) ** 2))


def subspace_fragmentation(labeled, pixel_size):
    actual_ary = labeled[0]
    unique_qry = np.unique(actual_ary, return_counts=True)
    unique_counts = unique_qry[1] * pixel_size
    individual = len(np.where(unique_counts < 5)[0])
    aggregates = len(np.where(50 > unique_counts.all() >= 5)[0])
    colonies = len(np.where(unique_counts >= 50)[0])
    return (individual, aggregates, colonies)


def gpt_calculate_volume(labeled_image: np.ndarray, voxel_size: float) -> np.ndarray:
    unique_labels, counts = np.unique(labeled_image, return_counts=True)
    volumes = counts * voxel_size
    volume_dict = dict(zip(unique_labels, volumes))
    return np.median(np.array(list(volume_dict.values())))


def gpt_measure_compactness(labeled_image: np.ndarray, voxel_size: float) -> np.ndarray:
    volumes = gpt_calculate_volume(labeled_image, voxel_size)
    surface_areas = gpt_calculate_surface_area(labeled_image)
    compactness = volumes / surface_areas
    return compactness


def gpt_calculate_surface_area(labeled_image: np.ndarray) -> np.ndarray:
    surface_area_dict = {}
    unique_labels = np.unique(labeled_image)

    for lbl in unique_labels:
        if lbl == 0:  # Skip background
            continue

        # Create a binary mask for the current label
        mask = (labeled_image == lbl).astype(np.uint8)
        # Use the scipy function to calculate the surface area
        surface_area = np.sum(np.gradient(np.array(mask))[0].astype(np.float32) > 0)  # Approximation
        surface_area_dict[lbl] = surface_area

    return np.mean(np.array(list(surface_area_dict.values())))


def gpt_calculate_texture_features(
    image_stack: np.ndarray, distances=(1,), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
) -> pd.DataFrame:
    from skimage.feature import graycomatrix, graycoprops

    texture_features = []
    property_names = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")

    for z in range(len(image_stack)):
        slice_image = np.clip(np.rint(np.asarray(image_stack[z]) * 31), 0, 31).astype(np.uint8)
        glcm = graycomatrix(slice_image, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
        row = {}
        for prop in property_names:
            values = graycoprops(glcm, prop)
            for distance_index, distance in enumerate(distances):
                direction_values = []
                for angle_index, angle in enumerate(angles):
                    value = float(values[distance_index, angle_index])
                    degrees = int(round(np.degrees(angle)))
                    row[f"GLCM{prop.title()}_d{distance}_angle{degrees}"] = value
                    direction_values.append(value)
                row[f"GLCM{prop.title()}_d{distance}_mean"] = float(np.mean(direction_values))
        texture_features.append(row)

    return pd.DataFrame(texture_features)


def segment(
    filepath,
    outfolder,
    threshold_mode="comstat2",
    threshold_method="otsu",
    threshold_scale="stack_normalized",
    manual_threshold=None,
    upper_threshold=None,
    dim_class="foreground",
    representative_images=None,
    local_density_radius_um=2.0,
    connectivity_3d=26,
    minimum_object_area_um2=0.0,
    minimum_object_volume_um3=0.0,
):
    image = mtif.read_stack(filepath, units="um")

    actual_final_df = []
    all_values = []
    all_evalues = []

    all_images = []

    raw_layers = []
    for enx, sub_image in enumerate(image):
        print(f"Processing sub-image {enx} for {filepath} ..")
        if len(sub_image.shape) == 3:
            sub_image = np.median(sub_image, axis=-1)
        raw_layers.append(np.asarray(sub_image))

    raw_stack = np.asarray(raw_layers)
    representative_images = representative_images or {}
    parsed_metadata = parse_image_name(os.path.basename(filepath))
    image_label = parsed_metadata["label"] if parsed_metadata else None
    representative_image = representative_images.get(image_label)
    stack_max = max(float(np.max(layer)) for layer in raw_layers)
    if stack_max > 0:
        all_images = [layer / stack_max for layer in raw_layers]
    else:
        all_images = [np.zeros_like(layer, dtype=float) for layer in raw_layers]
    image_space = np.asarray(all_images)

    for enx, sub_image in enumerate(all_images):
        max_eig = 1  # np.max(eigenvalues).real
        all_evalues.append(max_eig)
        row = {}

        # stevilo
        for cutoff in np.arange(0.01, 0.3, 0.01):
            cell_count, labeled = get_cell_count(sub_image, cutoff)
            row[f"counts(inten<{cutoff}"] = cell_count

            intensity_range = np.max(sub_image) - np.min(sub_image)
            tmp_img = (sub_image - np.mean(sub_image)) / intensity_range if intensity_range else np.zeros_like(sub_image)

            row[f"counts(Norminten<{cutoff}"], _ = get_cell_count(tmp_img, cutoff)

        # Intensity range
        row["diff"] = np.max(sub_image) - np.min(sub_image)

        # Max intensity
        row["max"] = np.max(sub_image)

        # Median value
        row["med"] = np.median(sub_image)

        # Stddev
        row["std"] = np.std(sub_image)
        row["q10"] = np.percentile(sub_image, 10)
        row["q25"] = np.percentile(sub_image, 25)
        row["q75"] = np.percentile(sub_image, 75)
        row["q90"] = np.percentile(sub_image, 90)

        # Mean pixel
        row["mean"] = np.mean(sub_image)

        # Min pixel
        row["min"] = np.min(sub_image)

        # Emptyness
        row["minProp"] = len(np.where(sub_image < row["mean"])[0]) / (sub_image.shape[0] * sub_image.shape[1])

        actual_final_df.append(row)
        for j in sub_image.reshape(-1):
            all_values.append(j)

    print(f"Computing global features for {filepath} ..")
    all_values = np.array(all_values)
    out_df = pd.DataFrame(actual_final_df)

    for cutoff in np.arange(0.01, 0.3, 0.01):
        threshold_mask_for_volume = threshold_mask(image_space, cutoff, mode=threshold_mode)
        biomass_per_area = threshold_mask_for_volume.sum() * voxel_size / (image_space.shape[1] * image_space.shape[2] * pixel_area)
        out_df[f"BioVolumeThr{cutoff}"] = biomass_per_area

    for col in out_df.columns:
        out_df[f"{col}Normalized"] = 100 * (out_df[col] / out_df[col].sum())

    try:
        out_df["Homogeneity"] = get_homogenity(image_space)  # inspired by (10.1016/j.mimet.2004.08.003)
    except Exception:
        out_df["Homogeneity"] = 0

    try:
        global_texture_features = gpt_calculate_texture_features(all_images).mean(axis=0)
        for k, v in global_texture_features.items():
            out_df[k] = v
    except Exception:
        # 2d case
        pass

    class_thresholds = None
    actual_threshold_scale = "stack_normalized"
    if threshold_method == "otsu":
        selected_threshold = otsu_threshold(raw_stack)
        binary_mask = threshold_mask(raw_stack, selected_threshold, mode=threshold_mode)
        selected_threshold_qc = selected_threshold / stack_max if stack_max else 1.0
        actual_threshold_scale = "raw_intensity"
    elif threshold_method == "manual":
        if manual_threshold is None:
            raise ValueError("Manual threshold method requires a threshold")
        selected_threshold = float(manual_threshold)
        threshold_image = raw_stack if threshold_scale == "raw" else image_space
        actual_threshold_scale = threshold_scale
        binary_mask = threshold_mask(threshold_image, selected_threshold, mode=threshold_mode)
        selected_threshold_qc = selected_threshold / stack_max if threshold_scale == "raw" and stack_max else selected_threshold
    elif threshold_method == "robust_background":
        selected_threshold = robust_background_threshold(image_space)
        binary_mask = threshold_mask(image_space, selected_threshold, mode=threshold_mode)
        selected_threshold_qc = selected_threshold
    elif threshold_method == "bem":
        selected_threshold = float(bem_threshold(raw_stack))
        binary_mask = threshold_mask(raw_stack, selected_threshold, mode=threshold_mode)
        selected_threshold_qc = selected_threshold / stack_max if stack_max else 1.0
        actual_threshold_scale = "raw_uint8"
    elif threshold_method == "multi_otsu":
        if (manual_threshold is None) != (upper_threshold is None):
            raise ValueError("Multi-Otsu needs both manual thresholds or neither")
        if manual_threshold is None:
            lower, upper = multi_otsu_thresholds(image_space)
            class_image = image_space
            actual_threshold_scale = "stack_normalized"
        else:
            lower, upper = float(manual_threshold), float(upper_threshold)
            class_image = raw_stack if threshold_scale == "raw" else image_space
            actual_threshold_scale = threshold_scale
        class_thresholds = (lower, upper)
        class_labels = multi_otsu_labels(class_image, lower, upper, mode=threshold_mode)
        binary_mask = multi_otsu_biomass_mask(class_labels, dim_class)
        selected_threshold = lower if dim_class == "foreground" else upper
        selected_threshold_qc = selected_threshold / stack_max if threshold_scale == "raw" and stack_max else selected_threshold
        if actual_threshold_scale == "stack_normalized":
            selected_threshold_qc = selected_threshold
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}")
    horizontal_spreading, vertical_spreading, total_spreading = calculate_spatial_spreading(binary_mask)
    out_df["SpreadingHorizontal"] = horizontal_spreading
    out_df["SpreadingVertical"] = vertical_spreading
    out_df["SpreadingTotal"] = total_spreading
    spatial_metrics = comstat_metrics(binary_mask, (voxel_size_x, voxel_size_y, voxel_size_z))
    for name, value in spatial_metrics.items():
        if np.isscalar(value):
            out_df[name] = value
    out_df["LayerArea_um2"] = spatial_metrics["LayerArea_um2"]
    out_df["LayerOccupancy_pct"] = spatial_metrics["LayerOccupancy_pct"]

    substratum_objects = connected_object_metrics_2d(
        binary_mask[0], (voxel_size_x, voxel_size_y, voxel_size_z), min_area_um2=minimum_object_area_um2
    )
    object_areas = substratum_objects["ObjectAreas_um2"]
    object_nnd = substratum_objects["NearestNeighborDistances_um"]
    out_df["SubstratumObjectCount"] = substratum_objects["ObjectCount"]
    out_df["SubstratumObjectDensity_per_um2"] = substratum_objects["ObjectDensity_per_um2"]
    out_df["SubstratumObjectAreaMean_um2"] = float(np.mean(object_areas)) if object_areas.size else float("nan")
    out_df["SubstratumObjectAreaMedian_um2"] = float(np.median(object_areas)) if object_areas.size else float("nan")
    out_df["SubstratumObjectAreaIQR_um2"] = (
        float(np.percentile(object_areas, 75) - np.percentile(object_areas, 25)) if object_areas.size else float("nan")
    )
    out_df["SubstratumObjectAreaStd_um2"] = float(np.std(object_areas)) if object_areas.size else float("nan")
    out_df["SubstratumObjectAreaP90_um2"] = float(np.percentile(object_areas, 90)) if object_areas.size else float("nan")
    object_diameters = 2.0 * np.sqrt(object_areas / np.pi)
    out_df["SubstratumObjectEquivalentDiameterMean_um"] = float(object_diameters.mean()) if object_diameters.size else float("nan")
    out_df["SubstratumBiomassArea_um2"] = float(binary_mask[0].sum() * pixel_area)
    valid_nnd = object_nnd[np.isfinite(object_nnd)]
    out_df["SubstratumObjectNearestNeighborMean_um"] = float(valid_nnd.mean()) if valid_nnd.size else float("nan")
    out_df["SubstratumObjectNearestNeighborMedian_um"] = float(np.median(valid_nnd)) if valid_nnd.size else float("nan")
    out_df["SubstratumObjectNearestNeighborStd_um"] = float(valid_nnd.std()) if valid_nnd.size else float("nan")
    out_df["SubstratumObjectNearestNeighborIQR_um"] = (
        float(np.percentile(valid_nnd, 75) - np.percentile(valid_nnd, 25)) if valid_nnd.size else float("nan")
    )
    out_df["SubstratumObjectNearestNeighborP90_um"] = float(np.percentile(valid_nnd, 90)) if valid_nnd.size else float("nan")
    pores_2d = internal_pore_metrics_2d(binary_mask[0], (voxel_size_x, voxel_size_y, voxel_size_z))
    pore_areas_2d = pores_2d["InternalPoreAreas_um2"]
    out_df["SubstratumInternalPoreCount2D"] = pores_2d["InternalPoreCount2D"]
    out_df["SubstratumInternalPorosity2D"] = pores_2d["InternalPorosity2D"]
    out_df["SubstratumInternalPoreAreaMean_um2"] = float(pore_areas_2d.mean()) if pore_areas_2d.size else float("nan")
    out_df["SubstratumInternalPoreAreaMedian_um2"] = float(np.median(pore_areas_2d)) if pore_areas_2d.size else float("nan")

    objects_3d = connected_object_metrics_3d(
        binary_mask,
        (voxel_size_x, voxel_size_y, voxel_size_z),
        connectivity=connectivity_3d,
        min_volume_um3=minimum_object_volume_um3,
    )
    volumes_3d = objects_3d["ObjectVolumes3D_um3"]
    nnd_3d = objects_3d["NearestNeighborDistances3D_um"]
    valid_nnd_3d = nnd_3d[np.isfinite(nnd_3d)]
    out_df["ObjectCount3D"] = objects_3d["ObjectCount3D"]
    out_df["ObjectDensity3D_per_um3"] = objects_3d["ObjectDensity3D_per_um3"]
    out_df["ObjectVolume3DMean_um3"] = float(volumes_3d.mean()) if volumes_3d.size else float("nan")
    out_df["ObjectVolume3DMedian_um3"] = float(np.median(volumes_3d)) if volumes_3d.size else float("nan")
    out_df["ObjectVolume3DStd_um3"] = float(volumes_3d.std()) if volumes_3d.size else float("nan")
    out_df["ObjectNearestNeighbor3DMean_um"] = float(valid_nnd_3d.mean()) if valid_nnd_3d.size else float("nan")
    out_df["ObjectNearestNeighbor3DMedian_um"] = float(np.median(valid_nnd_3d)) if valid_nnd_3d.size else float("nan")
    out_df["ObjectNearestNeighbor3DIQR_um"] = (
        float(np.percentile(valid_nnd_3d, 75) - np.percentile(valid_nnd_3d, 25)) if valid_nnd_3d.size else float("nan")
    )

    pores_3d = internal_pore_metrics_3d(binary_mask, (voxel_size_x, voxel_size_y, voxel_size_z), connectivity=connectivity_3d)
    pore_volumes = pores_3d["InternalPoreVolumes_um3"]
    out_df["InternalPoreCount3D"] = pores_3d["InternalPoreCount3D"]
    out_df["InternalPorosity3D"] = pores_3d["InternalPorosity3D"]
    out_df["InternalPoreVolumeMean_um3"] = float(pore_volumes.mean()) if pore_volumes.size else float("nan")
    out_df["InternalPoreVolumeMedian_um3"] = float(np.median(pore_volumes)) if pore_volumes.size else float("nan")
    density_3d = local_biomass_density(binary_mask, (voxel_size_x, voxel_size_y, voxel_size_z), local_density_radius_um)
    for key in ("LocalBiomassDensityMean", "LocalBiomassDensityStd", "LocalBiomassDensityIQR"):
        out_df[key] = density_3d[key]
    qc_stem = os.path.splitext(os.path.basename(filepath))[0]
    qc_overlay_path = os.path.join(outfolder, f"{qc_stem}_segmentation_qc.png")
    qc_curve_path = os.path.join(outfolder, f"{qc_stem}_threshold_curve.png")
    os.makedirs(outfolder, exist_ok=True)
    if image_space.ndim == 3 and image_space.shape[0]:
        biomass_by_layer = binary_mask.sum(axis=(1, 2))
        representative_layer = int(np.argmax(biomass_by_layer))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image_space[representative_layer], cmap="gray")
        if class_thresholds:
            dim_layer = class_labels[representative_layer] == 1
            bright_layer = class_labels[representative_layer] == 2
            axes[0].imshow(np.ma.masked_where(~dim_layer, dim_layer), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
            axes[0].imshow(np.ma.masked_where(~bright_layer, bright_layer), cmap="Oranges", alpha=0.55, vmin=0, vmax=1)
        else:
            axes[0].imshow(binary_mask[representative_layer], cmap="autumn", alpha=0.35)
        method_name = threshold_method.replace("_", " ").title()
        axes[0].set_title(f"{method_name} segmentation, layer {representative_layer}")
        axes[0].axis("off")
        axes[1].hist(image_space.ravel(), bins=256, color="0.35")
        axes[1].axvline(selected_threshold_qc, color="red", linestyle="--", label=f"T={selected_threshold_qc:.4g}")
        qc_class_thresholds = class_thresholds
        if class_thresholds and actual_threshold_scale == "raw" and stack_max:
            qc_class_thresholds = tuple(value / stack_max for value in class_thresholds)
        if qc_class_thresholds:
            axes[1].axvline(qc_class_thresholds[0], color="blue", linestyle=":", label="Dim cutoff")
            axes[1].axvline(qc_class_thresholds[1], color="orange", linestyle=":", label="Bright cutoff")
        axes[1].set_xlabel("Stack-normalized intensity")
        axes[1].set_ylabel("Voxel count")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(qc_overlay_path, dpi=180)
        plt.close(fig)

        sweep = np.linspace(0.0, 1.0, 101)
        biomass_fraction = np.asarray([threshold_mask(image_space, value, threshold_mode).mean() for value in sweep])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sweep, biomass_fraction)
        ax.axvline(selected_threshold_qc, color="red", linestyle="--", label=f"Selected T={selected_threshold_qc:.4g}")
        ax.set_xlabel("Stack-normalized threshold")
        ax.set_ylabel("Biomass fraction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(qc_curve_path, dpi=180)
        plt.close(fig)

    foreground_coords = np.argwhere(binary_mask)
    border_voxels = (
        (
            (foreground_coords[:, 0] == 0)
            | (foreground_coords[:, 0] == binary_mask.shape[0] - 1)
            | (foreground_coords[:, 1] == 0)
            | (foreground_coords[:, 1] == binary_mask.shape[1] - 1)
            | (foreground_coords[:, 2] == 0)
            | (foreground_coords[:, 2] == binary_mask.shape[2] - 1)
        )
        if foreground_coords.size
        else np.array([], dtype=bool)
    )
    source_limit = np.iinfo(raw_stack.dtype).max if np.issubdtype(raw_stack.dtype, np.integer) else None
    saturation_fraction = float(np.mean(raw_stack == source_limit)) if source_limit is not None else None
    segmentation_record = {
        "image": os.path.basename(filepath),
        "method": threshold_method,
        "threshold_stack_normalized": selected_threshold_qc,
        "selected_threshold_value": selected_threshold,
        "selected_threshold_units": actual_threshold_scale,
        "class_thresholds_stack_normalized": (
            [value / stack_max for value in class_thresholds]
            if class_thresholds and actual_threshold_scale == "raw" and stack_max
            else class_thresholds
        ),
        "threshold_boundary_mode": threshold_mode,
        "dim_class_assignment": dim_class if threshold_method == "multi_otsu" else None,
        "image_class": image_label,
        "representative_image_for_class": representative_image,
        "voxel_size_um": {"x": voxel_size_x, "y": voxel_size_y, "z": voxel_size_z},
        "connectivity_2d": 8,
        "connectivity_3d": connectivity_3d,
        "minimum_object_area_um2": minimum_object_area_um2,
        "minimum_object_volume_um3": minimum_object_volume_um3,
        "local_density_radius_um": local_density_radius_um,
        "array_axis_order": ["z", "y", "x"],
        "no_biomass_detected": spatial_metrics["NoBiomassDetected"],
        "foreground_fraction": float(binary_mask.mean()),
        "foreground_touching_image_border_fraction": float(border_voxels.mean()) if border_voxels.size else None,
        "top_layer_occupancy_percent": float(binary_mask[-1].mean() * 100) if binary_mask.shape[0] else None,
        "saturation_fraction": saturation_fraction,
        "three_class_fractions": {
            "background": float(np.mean(class_labels == 0)),
            "dim-intensity": float(np.mean(class_labels == 1)),
            "bright-intensity": float(np.mean(class_labels == 2)),
        }
        if class_thresholds
        else None,
        "qc_overlay": os.path.basename(qc_overlay_path),
        "threshold_curve": os.path.basename(qc_curve_path),
    }
    with open(os.path.join(outfolder, f"{qc_stem}_segmentation.json"), "w", encoding="utf-8") as summary_file:
        json.dump(segmentation_record, summary_file, indent=2)

    # Threshold sweep uses immutable masks and the complete x-y column population.
    for cutoff in np.arange(0.01, 0.3, 0.01):
        sweep_mask = threshold_mask(image_space, cutoff, mode=threshold_mode)
        sweep_metrics = comstat_metrics(sweep_mask, (voxel_size_x, voxel_size_y, voxel_size_z))
        out_df[f"ThicknessThreshold={cutoff}"] = sweep_metrics["MeanThickness_COMSTAT_um"]
        out_df[f"RoughnessThreshold={cutoff}"] = sweep_metrics["Roughness_RaStar"]

    out_file_name = (os.path.basename(filepath) + "CustomAlgos").replace(".tif", "") + ".txt"
    out_file_name = f"{outfolder}/{out_file_name}"
    print(f"writing {out_file_name}")
    out_df.to_csv(out_file_name, sep="\t")

    global_df = []
    try:
        global_mdiff = np.mean(np.diff(out_df["mean"].values))
        global_madiff = np.max(np.diff(out_df["mean"].values))
        global_miadiff = np.min(np.diff(out_df["mean"].values))
        global_eigen = np.mean(np.array(all_evalues))
        global_df.append(
            {
                "globalMean": np.mean(all_values),
                "mdiffs": global_mdiff,
                "maxdiffs": global_madiff,
                "mindiffs": global_miadiff,
                "eigen": global_eigen,
            }
        )
        dfx = pd.DataFrame(global_df)
        out_file_name = out_file_name.replace("CustomAlgos", "DiffGlobal")
        print(f"writing {out_file_name}")
        dfx.to_csv(out_file_name, sep="\t")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FeatureGen")
    parser.add_argument("-f", "--file", help="File name", required=True)
    parser.add_argument("-of", "--outfolder", help="Output folder", required=True)
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "manual", "bem", "robust_background", "multi_otsu"),
        default=os.environ.get("IMAGINE_THRESHOLD_METHOD", "otsu"),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("comstat1", "comstat2"),
        default=os.environ.get("IMAGINE_THRESHOLD_MODE", "comstat2"),
    )
    parser.add_argument(
        "--threshold-scale",
        choices=("stack_normalized", "raw"),
        default=os.environ.get("IMAGINE_THRESHOLD_SCALE", "stack_normalized"),
    )
    parser.add_argument("--threshold", type=float, default=os.environ.get("IMAGINE_THRESHOLD_VALUE") or None)
    parser.add_argument("--threshold-upper", type=float, default=os.environ.get("IMAGINE_THRESHOLD_UPPER_VALUE") or None)
    parser.add_argument(
        "--dim-class",
        choices=("background", "foreground"),
        default=os.environ.get("IMAGINE_DIM_CLASS_ASSIGNMENT", "foreground"),
    )
    parser.add_argument(
        "--representative-images-json",
        default=os.environ.get("IMAGINE_REPRESENTATIVE_IMAGES_JSON", "{}"),
    )
    parser.add_argument(
        "--local-density-radius-um",
        type=float,
        default=float(os.environ.get("IMAGINE_LOCAL_DENSITY_RADIUS_UM", "2.0")),
    )
    parser.add_argument("--connectivity-3d", type=int, choices=(6, 18, 26), default=int(os.environ.get("IMAGINE_CONNECTIVITY_3D", "26")))
    parser.add_argument("--minimum-object-area-um2", type=float, default=float(os.environ.get("IMAGINE_MIN_OBJECT_AREA_UM2", "0")))
    parser.add_argument("--minimum-object-volume-um3", type=float, default=float(os.environ.get("IMAGINE_MIN_OBJECT_VOLUME_UM3", "0")))
    args = vars(parser.parse_args())
    logger.info(
        "Using voxel dimensions X=%s µm, Y=%s µm, Z=%s µm",
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
    )
    threshold_max = 1 if args["threshold_scale"] == "stack_normalized" else math.inf
    if args["threshold"] is not None and not 0 <= args["threshold"] <= threshold_max:
        parser.error("--threshold is outside the selected intensity scale")
    if args["threshold_upper"] is not None and not 0 <= args["threshold_upper"] <= threshold_max:
        parser.error("--threshold-upper is outside the selected intensity scale")
    if args["threshold_method"] == "manual" and args["threshold"] is None:
        parser.error("--threshold-method manual requires --threshold")
    if args["threshold_method"] == "multi_otsu" and ((args["threshold"] is None) != (args["threshold_upper"] is None)):
        parser.error("multi_otsu requires both manual thresholds or neither")
    try:
        representative_images = json.loads(args["representative_images_json"])
    except json.JSONDecodeError as exc:
        parser.error(f"invalid representative threshold JSON: {exc}")
    if not isinstance(representative_images, dict):
        parser.error("representative images must be a JSON object")
    segment(
        args["file"],
        args["outfolder"],
        args["threshold_mode"],
        args["threshold_method"],
        args["threshold_scale"],
        args["threshold"],
        args["threshold_upper"],
        args["dim_class"],
        representative_images,
        args["local_density_radius_um"],
        args["connectivity_3d"],
        args["minimum_object_area_um2"],
        args["minimum_object_volume_um3"],
    )
