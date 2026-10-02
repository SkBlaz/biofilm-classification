import argparse
import logging
import math
import os
import warnings

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy.ndimage import label

from feature_metrics import (
    comstat_metrics,
    connected_object_metrics_2d,
    connected_object_metrics_3d,
    internal_pore_metrics_2d,
    internal_pore_metrics_3d,
    local_biomass_density,
    local_thickness_metrics,
)
from feature_taxonomy import classify_feature
from segmentation import (
    GROUPS,
    environment_settings,
    filter_mask,
    generate_mask,
    mask_for_image,
    normalize_settings,
    read_mask,
    read_tiff_stack,
    to_uint8,
)
from segmentation_qc import write_qc

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


def grayscale_fractal_dimension(stack):
    """Differential box counting on the grayscale intensity graph."""
    values = np.asarray(stack, dtype=float)
    if values.shape[0] == 1:
        values = values[0]
    span = float(np.ptp(values))
    if span == 0 or min(values.shape) < 4:
        return float("nan")
    values = (values - values.min()) / span
    scales, counts = [], []
    maximum = max(values.shape)
    for size in (2**exponent for exponent in range(1, int(np.log2(min(values.shape))) + 1)):
        count = 0
        for origin in np.ndindex(*(int(np.ceil(length / size)) for length in values.shape)):
            block = values[tuple(slice(index * size, min((index + 1) * size, length)) for index, length in zip(origin, values.shape))]
            count += int(np.floor(block.max() * maximum / size) - np.floor(block.min() * maximum / size) + 1)
        scales.append(np.log(maximum / size))
        counts.append(np.log(count))
    return float(np.polyfit(scales, counts, 1)[0]) if len(scales) > 1 else float("nan")


def add_distribution(row, prefix, values, unit):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    for name, function in (
        ("Mean", np.mean),
        ("Median", np.median),
        ("Std", np.std),
        ("IQR", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        ("P90", lambda x: np.percentile(x, 90)),
    ):
        row[f"{prefix}{name}_{unit}"] = float(function(values)) if values.size else float("nan")


def segment(
    filepath,
    outfolder,
    threshold_mode="comstat2",
    threshold_method="otsu",
    threshold_scale="uint8",
    manual_threshold=None,
    upper_threshold=None,
    dim_class="foreground",
    representative_images=None,
    local_density_radius_um=2.0,
    connectivity_3d=26,
    minimum_object_area_um2=0.0,
    minimum_object_volume_um3=0.0,
    *,
    segmentation_approach=None,
    threshold_sensitivity=1.0,
    bem_tolerance=0.10,
    conversion_max=None,
    masks_dir="",
    image_dimension="auto",
    feature_groups=None,
    qc_only=False,
):
    if threshold_mode != "comstat2":
        raise ValueError("Segmentation uses COMSTAT2 boundaries")
    if upper_threshold is not None or representative_images:
        raise ValueError("Use one manual threshold on uploaded images; representative maps and second cutoffs are no longer supported")
    settings = normalize_settings(
        {
            "segmentation_approach": segmentation_approach,
            "threshold_method": threshold_method,
            "threshold_scale": threshold_scale,
            "threshold_value": manual_threshold,
            "threshold_sensitivity": threshold_sensitivity,
            "bem_tolerance": bem_tolerance,
            "dim_class_assignment": dim_class,
            "conversion_max": conversion_max,
            "masks_dir": masks_dir,
            "image_dimension": image_dimension,
            "feature_groups": feature_groups or ["all"],
            "qc_only": qc_only,
            "connectivity_3d": connectivity_3d,
            "minimum_object_area_um2": minimum_object_area_um2,
            "minimum_object_volume_um3": minimum_object_volume_um3,
            "local_density_radius_um": local_density_radius_um,
        }
    )
    raw_stack, bits = read_tiff_stack(filepath)
    actual_dimension = "2d" if raw_stack.shape[0] == 1 else "3d"
    if image_dimension != "auto" and image_dimension != actual_dimension:
        raise ValueError(f"{os.path.basename(filepath)} is {actual_dimension.upper()}; choose matching image dimensions")
    if actual_dimension == "2d":
        settings["feature_groups"] = ["2d", "intensity", "geometry"]
    image8, conversion = to_uint8(raw_stack, conversion_max, bits)
    imported_labels = None
    mask_path = None
    if settings["segmentation_approach"] == "import":
        mask_path = mask_for_image(filepath, masks_dir)
        imported_mask, labelled = read_mask(mask_path, raw_stack.shape)
        imported_labels = imported_mask if labelled else None
    else:
        imported_mask = None
    binary_mask, result = generate_mask(image8, settings, imported_mask)
    dimensions = (voxel_size_x, voxel_size_y, voxel_size_z)
    binary_mask, object_labels = filter_mask(binary_mask, dimensions, settings, imported_labels)
    record = write_qc(filepath, outfolder, image8, binary_mask, object_labels, settings, result, conversion, dimensions, mask_path)
    if settings["qc_only"]:
        return record
    groups = set(settings["feature_groups"])
    stem = os.path.splitext(os.path.basename(filepath))[0]
    maximum = float(raw_stack.max())
    image_space = raw_stack.astype(float) / maximum if maximum > 0 else np.zeros_like(raw_stack, dtype=float)
    threshold_space = image8.astype(float) / 255
    rows = [{} for _ in raw_stack]
    if "intensity" in groups:
        for row, layer, raw_layer in zip(rows, image_space, raw_stack):
            for name, function in (
                ("diff", np.ptp),
                ("max", np.max),
                ("med", np.median),
                ("std", np.std),
                ("mean", np.mean),
                ("min", np.min),
            ):
                row[name] = float(function(layer))
                row[f"RawIntensity{name.title()}"] = float(function(raw_layer))
            for percentile in (10, 25, 75, 90):
                row[f"q{percentile}"] = float(np.percentile(layer, percentile))
            row["minProp"] = float(np.mean(layer < row["mean"]))
    if "serial" in groups:
        for row, layer in zip(rows, threshold_space):
            span = np.ptp(layer)
            normalized = (layer - layer.mean()) / span if span else np.zeros_like(layer)
            for cutoff in np.arange(0.01, 0.3, 0.01):
                row[f"counts(inten<{cutoff}"] = get_cell_count(layer, cutoff)[0]
                row[f"counts(Norminten<{cutoff}"] = get_cell_count(normalized, cutoff)[0]
    out_df = pd.DataFrame(rows, index=np.arange(len(rows)))
    if "serial" in groups:
        for cutoff in np.arange(0.01, 0.3, 0.01):
            sweep_mask = threshold_space >= cutoff
            sweep_mask, _ = filter_mask(sweep_mask, dimensions, settings)
            metrics = comstat_metrics(sweep_mask, dimensions, include_geometry=False)
            out_df[f"BioVolumeThr{cutoff}"] = metrics["Biomass_um3_per_um2"]
            out_df[f"ThicknessThreshold={cutoff}"] = metrics["MeanThickness_COMSTAT_um"]
            out_df[f"RoughnessThreshold={cutoff}"] = metrics["Roughness_RaStar"]
    totals = out_df.sum(axis=0).replace(0, np.nan)
    normalized = (100 * out_df.div(totals)).fillna(0)
    normalized.columns = [f"{column}Normalized" for column in normalized.columns]
    out_df = pd.concat([out_df, normalized], axis=1).copy()
    if "geometry" in groups:
        out_df["Homogeneity"] = get_homogenity(image_space)
        for name, value in gpt_calculate_texture_features(image_space).mean(axis=0).items():
            out_df[name] = value
        out_df["FractalGrayscaleBoxDimension"] = grayscale_fractal_dimension(raw_stack)
    if "basic" in groups or "advanced" in groups:
        metrics = comstat_metrics(binary_mask, dimensions, include_geometry="advanced" in groups)
        for name, value in metrics.items():
            group = "advanced" if classify_feature(name) == "3D advanced geometry" else "basic"
            if group in groups:
                out_df[name] = value
        if "basic" in groups:
            horizontal, vertical, total = calculate_spatial_spreading(binary_mask)
            out_df["SpreadingHorizontal"] = horizontal
            out_df["SpreadingVertical"] = vertical
            out_df["SpreadingTotal"] = total
            out_df["FilledThicknessMean_um"] = float(binary_mask.sum(axis=0).mean() * voxel_size_z)
    if "2d" in groups or groups == set(GROUPS):
        out_df = out_df.copy()
        mask2d = binary_mask[0]
        # For 3D stacks, this package describes the substratum slice.
        labels2d = object_labels[0] if actual_dimension == "2d" else None
        objects = connected_object_metrics_2d(mask2d, dimensions, object_labels=labels2d)
        row = {
            "SubstratumCoverage_pct": float(mask2d.mean() * 100),
            "SubstratumBiomassArea_um2": float(mask2d.sum() * pixel_area),
            "SubstratumObjectCount": objects["ObjectCount"],
            "SubstratumObjectDensity_per_um2": objects["ObjectDensity_per_um2"],
        }
        areas = objects["ObjectAreas_um2"]
        distances = objects["NearestNeighborDistances_um"]
        diameters = 2 * np.sqrt(areas / np.pi)
        add_distribution(row, "SubstratumObjectArea", areas, "um2")
        add_distribution(row, "SubstratumObjectEquivalentDiameter", diameters, "um")
        add_distribution(row, "SubstratumObjectNearestNeighbor", distances, "um")
        pores = internal_pore_metrics_2d(mask2d, dimensions)
        row["SubstratumInternalPoreCount2D"] = pores["InternalPoreCount2D"]
        row["SubstratumInternalPorosity2D"] = pores["InternalPorosity2D"]
        add_distribution(row, "SubstratumInternalPoreArea", pores["InternalPoreAreas_um2"], "um2")
        for name, value in row.items():
            out_df[name] = value
        pd.DataFrame({"area_um2": areas, "equivalent_diameter_um": diameters, "nearest_neighbor_um": distances}).to_csv(
            os.path.join(outfolder, f"{stem}_objects_2d.csv"), index=False
        )
        pd.DataFrame({"area_um2": pores["InternalPoreAreas_um2"]}).to_csv(os.path.join(outfolder, f"{stem}_pores_2d.csv"), index=False)
    if "objects" in groups and actual_dimension == "3d":
        objects = connected_object_metrics_3d(binary_mask, dimensions, connectivity_3d, object_labels=object_labels)
        row = {"ObjectCount3D": objects["ObjectCount3D"], "ObjectDensity3D_per_um3": objects["ObjectDensity3D_per_um3"]}
        add_distribution(row, "ObjectVolume3D", objects["ObjectVolumes3D_um3"], "um3")
        add_distribution(row, "ObjectNearestNeighbor3D", objects["NearestNeighborDistances3D_um"], "um")
        for name, value in row.items():
            out_df[name] = value
        pd.DataFrame(
            {"volume_um3": objects["ObjectVolumes3D_um3"], "nearest_neighbor_um": objects["NearestNeighborDistances3D_um"]}
        ).to_csv(os.path.join(outfolder, f"{stem}_objects_3d.csv"), index=False)
    if "advanced" in groups:
        out_df = out_df.copy()
        pores = internal_pore_metrics_3d(binary_mask, dimensions, connectivity=connectivity_3d)
        row = {"InternalPoreCount3D": pores["InternalPoreCount3D"], "InternalPorosity3D": pores["InternalPorosity3D"]}
        add_distribution(row, "InternalPoreVolume", pores["InternalPoreVolumes_um3"], "um3")
        row.update(
            {
                name: value
                for name, value in local_biomass_density(binary_mask, dimensions, local_density_radius_um).items()
                if np.isscalar(value)
            }
        )
        row.update(local_thickness_metrics(binary_mask, dimensions))
        for name, value in row.items():
            out_df[name] = value
        pd.DataFrame({"volume_um3": pores["InternalPoreVolumes_um3"]}).to_csv(os.path.join(outfolder, f"{stem}_pores_3d.csv"), index=False)
    out_df.to_csv(os.path.join(outfolder, f"{stem}CustomAlgos.txt"), sep="\t")
    if "intensity" in groups:
        means = np.mean(image_space, axis=(1, 2))
        differences = np.diff(means)
        global_row = {
            "globalMean": float(image_space.mean()),
            "mdiffs": float(differences.mean()) if differences.size else 0.0,
            "maxdiffs": float(differences.max()) if differences.size else 0.0,
            "mindiffs": float(differences.min()) if differences.size else 0.0,
            "eigen": 1.0,
        }
        pd.DataFrame([global_row]).to_csv(os.path.join(outfolder, f"{stem}DiffGlobal.txt"), sep="\t")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate calibrated features and segmentation QC")
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-of", "--outfolder", required=True)
    parser.add_argument("--segmentation-approach", choices=("automatic", "manual", "import"))
    parser.add_argument("--threshold-method", choices=("otsu", "manual", "bem", "robust_background", "multi_otsu"))
    parser.add_argument("--threshold-scale", choices=("stack_normalized", "uint8", "raw"))
    parser.add_argument("--threshold", dest="threshold_value", type=float)
    parser.add_argument("--threshold-sensitivity", type=float)
    parser.add_argument("--bem-tolerance", type=float)
    parser.add_argument("--conversion-max", type=float, help="Acquisition intensity maximum, e.g. 4095 for 12-bit")
    parser.add_argument("--dim-class", dest="dim_class_assignment", choices=("background", "foreground"))
    parser.add_argument("--masks-dir")
    parser.add_argument("--image-dimension", choices=("auto", "2d", "3d"))
    parser.add_argument("--feature-groups", help="Comma-separated basic,objects,advanced,intensity,geometry,serial or all")
    parser.add_argument("--qc-only", action="store_true", default=None)
    parser.add_argument("--local-density-radius-um", type=float)
    parser.add_argument("--connectivity-3d", type=int, choices=(6, 18, 26))
    parser.add_argument("--minimum-object-area-um2", type=float)
    parser.add_argument("--minimum-object-volume-um3", type=float)
    args = vars(parser.parse_args())
    raw_settings = environment_settings()
    raw_settings.update({key: value for key, value in args.items() if value is not None})
    try:
        settings = normalize_settings(raw_settings)
        segment(
            args["file"],
            args["outfolder"],
            threshold_method=settings["threshold_method"],
            threshold_scale=settings["threshold_scale"],
            manual_threshold=settings["threshold_value"],
            dim_class=settings["dim_class_assignment"],
            local_density_radius_um=settings["local_density_radius_um"],
            connectivity_3d=settings["connectivity_3d"],
            minimum_object_area_um2=settings["minimum_object_area_um2"],
            minimum_object_volume_um3=settings["minimum_object_volume_um3"],
            segmentation_approach=settings["segmentation_approach"],
            threshold_sensitivity=settings["threshold_sensitivity"],
            bem_tolerance=settings["bem_tolerance"],
            conversion_max=settings["conversion_max"],
            masks_dir=settings["masks_dir"],
            image_dimension=settings["image_dimension"],
            feature_groups=settings["feature_groups"],
            qc_only=settings["qc_only"],
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
