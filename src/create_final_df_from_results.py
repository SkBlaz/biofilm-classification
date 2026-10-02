import glob
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from feature_taxonomy import feature_catalog
except ImportError:
    from .feature_taxonomy import feature_catalog

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)


def drop_export_artifact_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove accidental CSV index columns without changing generated features.

    Empty, NaN, and infinite feature values are intentionally retained here. The
    learning and inference loaders apply the legacy value-level imputation, so
    dropping a whole feature because one sample needs imputation changes the
    feature contract and makes GUI-generated tables differ from CLI tables.
    """
    artifact_columns = [str(column) for column in frame.columns if str(column).startswith("Unnamed:")]
    return frame.drop(columns=artifact_columns), artifact_columns


if __name__ == "__main__":
    results_folder_analysis = sys.argv[1]
    outfile = sys.argv[2]
    unlabelled = "--unlabelled" in sys.argv[3:] or "--unlabeled" in sys.argv[3:]

    all_dfs = []

    for results_file in glob.glob(f"{results_folder_analysis}/*.txt"):
        if os.path.isdir(results_file):
            continue
        logging.info(f"Analyzing file: {results_file}")
        # if "Area2D" not in results_file:
        #     continue
        contents = pd.read_csv(results_file, sep="\t")
        if "sampleName" in contents.columns:
            non_sname = contents.columns[1:]
            # fid = str(hash(results_file))
            fid = results_file.split("/")[-1].replace(".txt.", "")
            non_sname = [x + "-" + fid for x in non_sname]
            contents.columns = ["sampleName"] + non_sname
            melted_df = pd.melt(contents, id_vars=["sampleName"])
            all_dfs.append(melted_df)

    df_final = pd.concat(all_dfs)
    df_final = df_final.reset_index().pivot(index="sampleName", columns="variable", values="value")
    if unlabelled:
        df_final["label"] = "unlabelled"
    else:
        df_final["label"] = [x.split("--")[4] for x in df_final.index.tolist()]
    ubound = df_final.isnull().sum(axis=1) / df_final.shape[1]
    df_final = df_final[ubound < 0.8]
    df_final, removed_columns = drop_export_artifact_columns(df_final)
    if removed_columns:
        logging.warning(
            "Excluded %d accidental index columns: %s",
            len(removed_columns),
            ", ".join(removed_columns),
        )
    #    df_final.to_csv(f"../prepared_data/{date.today()}-{tag}.tsv", sep="\t")
    logging.info(f"Writing {outfile}")
    df_final.to_csv(outfile, sep="\t")
    taxonomy_path = Path(outfile).with_name("feature_taxonomy.json")
    taxonomy_path.write_text(json.dumps(feature_catalog(df_final.columns), indent=2), encoding="utf-8")
    logging.info("Writing %s", taxonomy_path)

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (Path(results_folder_analysis).parent / "feature_generator").glob("*_segmentation.json")
    ]
    if records:
        settings = []
        for record in records:
            if record.get("segmentation_version") != 2:
                raise ValueError("Cannot mix legacy and 8-bit segmentation outputs")
            settings.append(
                {
                    "segmentation_version": 2,
                    "segmentation_approach": record["segmentation_approach"],
                    "threshold_method": record["requested_method"],
                    "threshold_scale": record["submitted_threshold_units"] or "uint8",
                    "threshold_value": record["submitted_threshold_value"],
                    "threshold_sensitivity": record["configured_threshold_sensitivity"],
                    "bem_tolerance": record["configured_bem_tolerance"],
                    "conversion_max": record["configured_conversion_max"],
                    "dim_class_assignment": record["dim_class_assignment"] or "foreground",
                    "image_dimension": record["image_dimension"],
                    "feature_groups": record["feature_groups"],
                    "connectivity_3d": record["connectivity_3d"],
                    "minimum_object_area_um2": record["minimum_object_area_um2"],
                    "minimum_object_volume_um3": record["minimum_object_volume_um3"],
                    "local_density_radius_um": record["local_density_radius_um"],
                    **{f"voxel_size_{axis}": value for axis, value in record["voxel_size_um"].items()},
                }
            )
        if any(item != settings[0] for item in settings):
            raise ValueError("Images in one feature table must use matching generation settings")
        Path(outfile).with_name("feature_generation_settings.json").write_text(json.dumps(settings[0], indent=2), encoding="utf-8")
