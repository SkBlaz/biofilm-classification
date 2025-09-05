import argparse
import os.path

import pandas as pd
import numpy as np
import multipagetiff as mtif
import logging
from scipy.ndimage import label
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_cell_count(intensity_matrix, threshold=0.3, visualizations=False):
    norm_mat = intensity_matrix / np.max(intensity_matrix)
    norm_mat[np.where(norm_mat > threshold)] = 1
    norm_mat[np.where(norm_mat <= threshold)] = 0
    labeled, ncomponents = label(norm_mat, CONNECTION_KERNEL)
    return ncomponents, labeled


def segment_lite(filepath, outfolder):
    """
    Lightweight version of the segment function that generates fewer features
    for faster processing.
    """
    
    image = mtif.read_stack(filepath, units="um")

    actual_final_df = []
    all_values = []

    # lower bound for defining "biomass"
    biomass_thr = 0.03
    biomass_counts_per_layers = []

    for enx, sub_image in enumerate(image):
        print(f"Processing sub-image {enx} for {filepath} ..")
        if len(sub_image.shape) == 3:
            sub_image = np.median(sub_image, axis=-1)
            sub_image = sub_image / np.max(sub_image)

        all_pixels_layer = sub_image.shape[0] * sub_image.shape[1]
        row = {}

        # Basic count features (simplified set)
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
            cell_count, labeled = get_cell_count(sub_image, threshold)
            row[f"counts(inten<{threshold})"] = cell_count

        tmp_count, _ = get_cell_count(sub_image, biomass_thr)
        biomass_counts_per_layers.append(tmp_count)

        # Basic intensity statistics
        row["diff"] = np.max(sub_image) - np.min(sub_image)
        row["max"] = np.max(sub_image)
        row["med"] = np.median(sub_image)
        row["std"] = np.std(sub_image)
        row["mean"] = np.mean(sub_image)
        row["min"] = np.min(sub_image)

        # Emptyness ratio
        row["minProp"] = len(np.where(sub_image < row["mean"])[0]) / (
            sub_image.shape[0] * sub_image.shape[1])

        actual_final_df.append(row)
        for j in sub_image.reshape(-1):
            all_values.append(j)

    print(f"Computing global features for {filepath} ..")
    all_values = np.array(all_values)
    out_df = pd.DataFrame(actual_final_df)
    
    # Basic biovolume calculations (simplified)
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
        pixel_count = out_df[f"counts(inten<{threshold})"].sum()
        biovolume = (pixel_count * voxel_size) / 134 ** 2
        out_df[f"BioVolumeThr{threshold}"] = biovolume

    # Normalized features
    for col in out_df.columns:
        if col.startswith("counts") or col.startswith("BioVolume"):
            out_df[f"{col}Normalized"] = 100 * (out_df[col] / out_df[col].sum())

    # Basic coverage calculation
    out_df["SubstratumRelativeCoverage"] = 100 * (biomass_counts_per_layers[0] / all_pixels_layer)

    # Basic thickness calculation (simplified)
    if len(biomass_counts_per_layers) > 1:
        out_df["MeanThickness"] = np.mean([len(biomass_counts_per_layers)])
        out_df["Roughness"] = np.std(biomass_counts_per_layers)
    else:
        out_df["MeanThickness"] = 1
        out_df["Roughness"] = 0

    out_file_name = (os.path.basename(filepath) + "LiteAlgos").replace(".tif", "") + ".txt"
    out_file_name = f"{outfolder}/{out_file_name}"
    print(f"writing {out_file_name}")
    out_df.to_csv(out_file_name, sep="\t")

    # Global statistics
    global_df = []
    try:
        global_mdiff = np.mean(np.diff(out_df["mean"].values))
        global_madiff = np.max(np.diff(out_df["mean"].values))
        global_miadiff = np.min(np.diff(out_df["mean"].values))
        global_df.append(
            {
                "globalMean": np.mean(all_values),
                "mdiffs": global_mdiff,
                "maxdiffs": global_madiff,
                "mindiffs": global_miadiff,
            }
        )
        dfx = pd.DataFrame(global_df)
        out_file_name = out_file_name.replace("LiteAlgos", "LiteGlobal")
        print(f"writing {out_file_name}")
        dfx.to_csv(out_file_name, sep="\t")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FeatureGenLite - Lightweight Feature Generator")
    parser.add_argument("-f", "--file", help="File name", required=True)
    parser.add_argument("-of", "--outfolder", help="Output folder", required=True)
    args = vars(parser.parse_args())
    segment_lite(args['file'], args["outfolder"])