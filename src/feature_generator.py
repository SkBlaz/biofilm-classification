import argparse
import logging
import os.path
import warnings

import matplotlib.pyplot as plt
import multipagetiff as mtif
import numba
import numpy as np
import pandas as pd
from scipy.ndimage import label

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

# Constants
voxel_size = 0.13 * 0.13 * 0.5
pixel_area = 0.13**2


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
    norm_mat = intensity_matrix / np.max(intensity_matrix)
    norm_mat[np.where(norm_mat > threshold)] = 1
    norm_mat[np.where(norm_mat <= threshold)] = 0
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


def segment(filepath, outfolder):
    image = mtif.read_stack(filepath, units="um")

    actual_final_df = []
    all_values = []
    all_evalues = []

    # lower bound for defining "biomass"
    biomass_thr = 0.03
    biomass_counts_per_layers = []
    all_images = []

    for enx, sub_image in enumerate(image):
        print(f"Processing sub-image {enx} for {filepath} ..")
        all_images.append(sub_image / np.max(sub_image))
        if len(sub_image.shape) == 3:
            sub_image = np.median(sub_image, axis=-1)
            sub_image = sub_image / np.max(sub_image)

        max_eig = 1  # np.max(eigenvalues).real
        all_pixels_layer = sub_image.shape[0] * sub_image.shape[1]
        all_evalues.append(max_eig)
        row = {}

        # stevilo
        for threshold in np.arange(0.01, 0.3, 0.01):
            cell_count, labeled = get_cell_count(sub_image, threshold)
            row[f"counts(inten<{threshold}"] = cell_count

            tmp_img = (sub_image - np.mean(sub_image)) / (np.max(sub_image) - np.min(sub_image))

            row[f"counts(Norminten<{threshold}"], _ = get_cell_count(tmp_img, threshold)

        tmp_count, _ = get_cell_count(sub_image, biomass_thr)
        biomass_counts_per_layers.append(tmp_count)

        # Intensity range
        row["diff"] = np.max(sub_image) - np.min(sub_image)

        # Max intensity
        row["max"] = np.max(sub_image)

        # Median value
        row["med"] = np.median(sub_image)

        # Stddev
        row["std"] = np.std(sub_image)

        # Mean pixel
        row["mean"] = np.mean(sub_image)

        # Min pixel
        row["min"] = np.min(sub_image)

        # Emptyness
        row["minProp"] = len(np.where(sub_image < row["mean"])[0]) / (sub_image.shape[0] * sub_image.shape[1])

        # GPT-generated
        row["GPTFractalDim"] = gpt_fractal_dimension(sub_image)
        row["GPTVolume"] = gpt_calculate_volume(sub_image, voxel_size)
        # row['GPTCompactness'] = gpt_measure_compactness(sub_image, voxel_size)
        # row['GPTSurface'] = gpt_calculate_surface_area(sub_image)

        actual_final_df.append(row)
        for j in sub_image.reshape(-1):
            all_values.append(j)

    print(f"Computing global features for {filepath} ..")
    all_values = np.array(all_values)
    out_df = pd.DataFrame(actual_final_df)

    for threshold in np.arange(0.01, 0.3, 0.01):
        pixel_count = out_df[f"counts(inten<{threshold}"].sum()
        biovolume = (pixel_count * voxel_size) / 134**2
        out_df[f"BioVolumeThr{threshold}"] = biovolume  # inspired by (10.1099/00221287-146-10-2395)

    for col in out_df.columns:
        out_df[f"{col}Normalized"] = 100 * (out_df[col] / out_df[col].sum())

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
        global_texture_features = gpt_calculate_texture_features(all_images).mean(axis=0)
        for k, v in global_texture_features.items():
            out_df[k] = v
    except Exception:
        # 2d case
        pass

    # Mean thickness - basically highest vertical line per pixel - avg-d
    image_space = np.array(all_images)

    # For each threshold, compute derived (vol) features
    for min_thr in np.arange(0.01, 0.3, 0.01):
        thresholded_matrices = []
        thicknesses = []

        for el in image_space:
            el[el <= min_thr] = 0
            thresholded_matrices.append(el)

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
    segment(args["file"], args["outfolder"])
