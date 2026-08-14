import glob
import logging
import os
import sys

import pandas as pd

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
