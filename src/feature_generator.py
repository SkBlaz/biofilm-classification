import argparse
import logging
import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import multipagetiff as mtif
import numba
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label
from scipy.spatial import cKDTree

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

# Keep the historical threshold grid so existing generated columns remain
# compatible with previously prepared feature tables.
SEGMENTATION_THRESHOLDS = np.arange(0.01, 0.3, 0.01)

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


def calculate_spatial_spreading(image_stack):
    Dx, Dy, Dz = [], [], []
    for z, slice_ in enumerate(image_stack):
        indices = np.argwhere(slice_ > 0)
        for index in indices:
            x, y = index
            Dx.append(x)
            Dy.append(y)
            Dz.append(z)

    Dx = np.array(Dx)
    Dy = np.array(Dy)
    Dz = np.array(Dz)

    sigma_x = np.var(Dx)
    sigma_y = np.var(Dy)
    sigma_z = np.var(Dz)

    Sz = sigma_z  # horizontal
    Sxy = np.sqrt(sigma_x + sigma_y)  # vertical
    Sxyz = np.sqrt(sigma_x + sigma_y + sigma_z)  # total

    return Sxy, Sz, Sxyz


def get_cell_count(intensity_matrix, threshold=0.3, visualizations=False):
    norm_mat = normalize_intensity(intensity_matrix)
    labeled, ncomponents = label(norm_mat > threshold, CONNECTION_KERNEL)

    if visualizations:
        plt.imshow(labeled)
        plt.savefig(f"snapshot{ncomponents}.png", dpi=300)
        plt.clf()

    return ncomponents, labeled


def get_cell_count_only(intensity_matrix: np.ndarray, threshold: float, workspace: np.ndarray) -> int:
    """Count components using a caller-owned label buffer.

    The threshold scan calls this hundreds of times per image. Reusing the
    output buffer avoids retaining a large temporary label array for every
    candidate threshold.
    """
    norm_mat = normalize_intensity(intensity_matrix)
    return int(label(norm_mat > threshold, CONNECTION_KERNEL, output=workspace))


def normalize_intensity(intensity_matrix: np.ndarray) -> np.ndarray:
    """Normalize an image to ``[0, 1]`` without failing on empty images."""
    image = np.asarray(intensity_matrix, dtype=float)
    maximum = np.nanmax(image) if image.size else 0.0
    if not np.isfinite(maximum) or maximum <= 0:
        return np.zeros_like(image, dtype=float)
    return np.nan_to_num(image / maximum, nan=0.0, posinf=0.0, neginf=0.0)


def optimal_threshold_from_counts(thresholds: np.ndarray, counts: np.ndarray) -> float:
    """Return the threshold at the peak of a raw connected-component curve.

    ``np.argmax`` deliberately chooses the first threshold for a plateau. It
    is the conservative end of a plateau and keeps the selected segmentation
    reproducible when several thresholds produce the same count.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if thresholds.ndim != 1 or counts.ndim != 1 or thresholds.size != counts.size or not thresholds.size:
        raise ValueError("thresholds and counts must be non-empty one-dimensional arrays of equal length")
    finite_counts = np.where(np.isfinite(counts), counts, -np.inf)
    if np.all(np.isneginf(finite_counts)):
        return float(thresholds[0])
    return float(thresholds[int(np.argmax(finite_counts))])


def _summary(values: np.ndarray, prefix: str, include_min: bool = False) -> dict[str, float]:
    """Create stable scalar summaries for a variable-length distribution."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        result = {
            f"{prefix}Mean": 0.0,
            f"{prefix}Median": 0.0,
            f"{prefix}Std": 0.0,
            f"{prefix}Largest": 0.0,
        }
        if include_min:
            result[f"{prefix}Min"] = 0.0
        return result
    result = {
        f"{prefix}Mean": float(np.mean(values)),
        f"{prefix}Median": float(np.median(values)),
        f"{prefix}Std": float(np.std(values)),
        f"{prefix}Largest": float(np.max(values)),
    }
    if include_min:
        result[f"{prefix}Min"] = float(np.min(values))
    return result


def object_size_features(labeled_image: np.ndarray) -> dict[str, float]:
    """Summarize foreground object sizes in pixels, excluding label 0."""
    _, sizes = np.unique(labeled_image[labeled_image > 0], return_counts=True)
    result = {"OptimalObjectCount": float(sizes.size)}
    result.update(_summary(sizes, "ObjectSize"))
    return result


def nearest_neighbor_features(labeled_image: np.ndarray) -> dict[str, float]:
    """Summarize each object's nearest centroid distance in pixels."""
    object_labels = np.unique(labeled_image[labeled_image > 0])
    if object_labels.size < 2:
        distances = np.array([], dtype=float)
    else:
        centroids = np.asarray(center_of_mass(labeled_image > 0, labeled_image, object_labels), dtype=float)
        distances = cKDTree(centroids).query(centroids, k=2)[0][:, 1]
    return _summary(distances, "NearestNeighborDistance", include_min=True)


def void_size_features(labeled_image: np.ndarray) -> dict[str, float]:
    """Summarize connected empty-region sizes in pixels.

    The zero label is biomass in this inverse segmentation and is therefore
    excluded from the void distribution.
    """
    void_labels, sizes = np.unique(label(labeled_image == 0, CONNECTION_KERNEL)[0], return_counts=True)
    sizes = sizes[void_labels > 0]
    result = {"VoidCount": float(sizes.size)}
    result.update(_summary(sizes, "VoidSize"))
    return result


def optimal_segmentation_features(labeled_image: np.ndarray) -> dict[str, float]:
    """Return object, nearest-neighbour, and pore summaries for one layer."""
    features = {}
    features.update(object_size_features(labeled_image))
    features.update(nearest_neighbor_features(labeled_image))
    features.update(void_size_features(labeled_image))
    return features


def biomass_height_features(labeled_layers: list[np.ndarray]) -> dict[str, float]:
    """Calculate biomass-weighted centre height and vertical spread.

    Heights are reported in zero-based layer-index units, matching the z
    coordinate used by the image stack. Empty stacks return zeroes.
    """
    layer_biomass = np.asarray([np.count_nonzero(layer) for layer in labeled_layers], dtype=float)
    total_biomass = float(layer_biomass.sum())
    if total_biomass == 0 or layer_biomass.size == 0:
        return {"BiomassCenterOfMassHeight": 0.0, "BiomassVerticalSpread": 0.0}
    z_positions = np.arange(layer_biomass.size, dtype=float)
    center = float(np.average(z_positions, weights=layer_biomass))
    spread = float(np.sqrt(np.average((z_positions - center) ** 2, weights=layer_biomass)))
    return {"BiomassCenterOfMassHeight": center, "BiomassVerticalSpread": spread}


def save_segmentation_diagnostics(
    normalized_layers: list[np.ndarray],
    thresholds: np.ndarray,
    counts: np.ndarray,
    outfolder: str,
    image_name: str,
    optimal_threshold: float,
) -> None:
    """Save the count curve and one binary mask for every candidate threshold."""
    diagnostic_dir = Path(outfolder) / "segmentation_diagnostics" / Path(image_name).stem
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    representative_index = int(np.argmax([np.sum(layer) for layer in normalized_layers])) if normalized_layers else 0
    representative = normalized_layers[representative_index] if normalized_layers else np.zeros((1, 1))

    curve = pd.DataFrame({"threshold": thresholds, "raw_count": counts, "is_optimal": thresholds == optimal_threshold})
    curve.to_csv(diagnostic_dir / "threshold_curve.tsv", sep="\t", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, counts, marker="o")
    plt.axvline(optimal_threshold, color="tab:red", linestyle="--", label=f"optimal={optimal_threshold:.2f}")
    plt.xlabel("Normalized intensity threshold")
    plt.ylabel("Connected-component count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(diagnostic_dir / "raw_counts_vs_threshold.png", dpi=180)
    plt.close()

    for threshold in thresholds:
        mask = normalize_intensity(representative) > threshold
        threshold_label = f"{threshold:.2f}".replace(".", "p")
        plt.imsave(diagnostic_dir / f"mask_threshold_{threshold_label}.png", mask, cmap="gray", vmin=0, vmax=1)


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
                    y1 = x + dy
                    z1 = z + dz
                    if not (0 <= x1 < a and 0 <= y1 < b and 0 <= z1 < c):
                        continue
                    this = int(binned_image[x, y, z])
                    other = int(binned_image[x1, y1, z1])
                    matrix[this, other] += 1
                    matrix[other, this] += 1
    return matrix / np.sum(matrix)


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
        surface_area_dict[label] = surface_area

    return np.mean(np.array(list(surface_area_dict.values())))


def gpt_calculate_texture_features(image_stack: np.ndarray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]) -> pd.DataFrame:
    from skimage.feature import graycomatrix, graycoprops

    texture_features = []

    for z in range(len(image_stack)):
        # Calculate the co-occurrence matrix for each slice
        slice_image = image_stack[z].astype(np.uint8)
        glcm = graycomatrix(slice_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Extract features from GLCM
        for distance in distances:
            contrast = graycoprops(glcm, "contrast")[0, 0]
            dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
            homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
            energy = graycoprops(glcm, "energy")[0, 0]
            correlation = graycoprops(glcm, "correlation")[0, 0]
            asm = graycoprops(glcm, "ASM")[0, 0]

            # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops
            texture_features.append(
                {
                    "GPTContrast": contrast,
                    "GPTDissimilarity": dissimilarity,
                    "GPTHomogeneity": homogeneity,
                    "GPTEnergy": energy,
                    "GPTCorrelation": correlation,
                    "GPTASM": asm,
                }
            )

    return pd.DataFrame(texture_features)


def gpt_thresholded_volume(binary_mask: np.ndarray, voxel_size: float) -> float:
    """Calculate the physical volume represented by foreground voxels."""
    return float(np.count_nonzero(binary_mask) * voxel_size)


def segment(filepath, outfolder):
    image = mtif.read_stack(filepath, units="um")

    all_values = []
    all_evalues = []

    # lower bound for defining "biomass"
    biomass_thr = 0.03
    biomass_counts_per_layers = []
    normalized_layers = []

    for enx, original_sub_image in enumerate(image):
        print(f"Processing sub-image {enx} for {filepath} ..")
        sub_image = np.asarray(original_sub_image, dtype=float)
        if len(sub_image.shape) == 3:
            sub_image = np.median(sub_image, axis=-1)
        sub_image = normalize_intensity(sub_image)
        normalized_layers.append(sub_image)
        all_evalues.append(1)
        all_values.extend(sub_image.reshape(-1))

    if not normalized_layers:
        raise ValueError(f"No image layers found in {filepath}")

    threshold_counts_by_layer = []
    normalized_threshold_counts_by_layer = []
    threshold_voxel_counts_by_layer = []
    for sub_image in normalized_layers:
        raw_counts = []
        normalized_counts = []
        voxel_counts = []
        component_workspace = np.empty(sub_image.shape, dtype=np.int32)
        for threshold in SEGMENTATION_THRESHOLDS:
            raw_counts.append(get_cell_count_only(sub_image, threshold, component_workspace))
            voxel_counts.append(int(np.count_nonzero(component_workspace)))

            tmp_img = (sub_image - np.mean(sub_image)) / (np.max(sub_image) - np.min(sub_image))
            normalized_counts.append(get_cell_count_only(tmp_img, threshold, component_workspace))
        threshold_counts_by_layer.append(raw_counts)
        normalized_threshold_counts_by_layer.append(normalized_counts)
        threshold_voxel_counts_by_layer.append(voxel_counts)

    raw_count_curve = np.sum(np.asarray(threshold_counts_by_layer), axis=0)
    optimal_threshold = optimal_threshold_from_counts(SEGMENTATION_THRESHOLDS, raw_count_curve)
    optimal_labels = [get_cell_count(layer, optimal_threshold)[1] for layer in normalized_layers]
    optimal_image_volume = sum(gpt_thresholded_volume(layer > 0, voxel_size) for layer in optimal_labels)
    height_features = biomass_height_features(optimal_labels)

    save_segmentation_diagnostics(
        normalized_layers,
        SEGMENTATION_THRESHOLDS,
        raw_count_curve,
        outfolder,
        os.path.basename(filepath),
        optimal_threshold,
    )

    actual_final_df = []
    for layer_index, sub_image in enumerate(normalized_layers):
        row = {}
        raw_counts = threshold_counts_by_layer[layer_index]
        normalized_counts = normalized_threshold_counts_by_layer[layer_index]

        for threshold_index, (threshold, raw_count, normalized_count) in enumerate(
            zip(SEGMENTATION_THRESHOLDS, raw_counts, normalized_counts)
        ):
            # Preserve the historical raw and normalized count feature names.
            row[f"counts(inten<{threshold}"] = raw_count
            row[f"counts(Norminten<{threshold}"] = normalized_count
            row[f"GPTVolumeThr{threshold}"] = float(threshold_voxel_counts_by_layer[layer_index][threshold_index] * voxel_size)

        optimal_label = optimal_labels[layer_index]
        row["OptimalThreshold"] = optimal_threshold
        row.update(optimal_segmentation_features(optimal_label))
        row["GPTVolumeOptimalThreshold"] = gpt_thresholded_volume(optimal_label > 0, voxel_size)
        row["GPTVolumeOptimalThresholdImage"] = optimal_image_volume
        row.update(height_features)

        biomass_count, _ = get_cell_count(sub_image, biomass_thr)
        biomass_counts_per_layers.append(biomass_count)

        # Intensity statistics
        row["diff"] = np.max(sub_image) - np.min(sub_image)
        row["max"] = np.max(sub_image)
        row["med"] = np.median(sub_image)
        row["std"] = np.std(sub_image)
        row["mean"] = np.mean(sub_image)
        row["min"] = np.min(sub_image)
        row["minProp"] = len(np.where(sub_image < row["mean"])[0]) / sub_image.size

        # GPT-generated legacy features
        row["GPTFractalDim"] = gpt_fractal_dimension(sub_image)
        row["GPTVolume"] = gpt_calculate_volume(sub_image, voxel_size)
        actual_final_df.append(row)

    print(f"Computing global features for {filepath} ..")
    all_values = np.array(all_values)
    out_df = pd.DataFrame(actual_final_df)

    for threshold in SEGMENTATION_THRESHOLDS:
        pixel_count = out_df[f"counts(inten<{threshold}"].sum()
        biovolume = (pixel_count * voxel_size) / 134**2
        out_df[f"BioVolumeThr{threshold}"] = biovolume  # inspired by (10.1099/00221287-146-10-2395)

    for col in out_df.columns:
        out_df[f"{col}Normalized"] = 100 * (out_df[col] / out_df[col].sum())

    all_pixels_layer = normalized_layers[0].size
    out_df["SubstratumRelativeCoverage"] = 100 * (
        biomass_counts_per_layers[0] / all_pixels_layer
    )  # inspired by (doi:10.1088/1367-2630/17/3/033017)
    try:
        out_df["Homogenity"] = get_homogenity(image.raw_images)  # inspired by (10.1016/j.mimet.2004.08.003)
    except Exception:
        out_df["Homogeneity"] = 0

    horizontal_spreding, vertical_spreading, total_spreading = 0, 0, 0
    try:
        horizontal_spreding, vertical_spreading, total_spreading = calculate_spatial_spreading(image.raw_images)
    except Exception:
        pass

    out_df["SpreadingHorizontal"] = horizontal_spreding
    out_df["SpreadingVertical"] = vertical_spreading
    out_df["SpreadingTotal"] = total_spreading

    try:
        global_texture_features = gpt_calculate_texture_features(np.asarray(normalized_layers)).mean(axis=0)
        for k, v in global_texture_features.items():
            out_df[k] = v
    except Exception:
        # 2d case
        pass

    # Mean thickness - basically highest vertical line per pixel - avg-d
    image_space = np.asarray(normalized_layers)

    # For each threshold, compute derived (vol) features
    for min_thr in SEGMENTATION_THRESHOLDS:
        thresholded_matrices = []
        thicknesses = []

        for el in image_space:
            thresholded_matrices.append(np.where(el > min_thr, el, 0))

        substrate_matrix = thresholded_matrices[0]
        where_biomass = np.where(substrate_matrix > 0)

        # For each pixel, measure verticals
        for x, y in zip(where_biomass[0], where_biomass[1]):
            layer = 1

            while True:
                for el in thresholded_matrices[1:]:
                    if el[x, y] != 0:
                        layer += 1
                    else:
                        break
                break
            thicknesses.append(layer)

        out_df[f"ThicknessThreshold={min_thr}"] = np.mean(thicknesses)

        # Point thickness, Li, is measured by locating the highest biomass voxel along the normal of the substrate (locating the highest point (μm)
        # above each (x,y) pixel in the bottom layer containing biomass)
        out_df[f"RoughnessThreshold={min_thr}"] = np.mean(np.diff(thicknesses))

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
    args = vars(parser.parse_args())
    logger.info(
        "Using voxel dimensions X=%s µm, Y=%s µm, Z=%s µm",
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
    )
    segment(args["file"], args["outfolder"])
