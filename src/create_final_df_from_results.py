import glob
import logging
import os
import sys

import pandas as pd

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)


def create_final_dataframe(results_folder_analysis, outfile, include_label=True):
    """Create a sample-by-feature TSV from aggregated feature files.

    When include_label is False, the output is suitable for unknown inference
    images whose sample names do not encode a known class label.
    """
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

    if not all_dfs:
        raise ValueError(f"No aggregated feature files found in {results_folder_analysis}")

    df_final = pd.concat(all_dfs)
    df_final = df_final.reset_index().pivot(index="sampleName", columns="variable", values="value")
    if include_label:
        df_final["label"] = [x.split("--")[4] for x in df_final.index.tolist()]
    ubound = df_final.isnull().sum(axis=1) / df_final.shape[1]
    df_final = df_final[ubound < 0.8]
    #    df_final.to_csv(f"../prepared_data/{date.today()}-{tag}.tsv", sep="\t")
    logging.info(f"Writing {outfile}")
    df_final.to_csv(outfile, sep="\t")
    return df_final


if __name__ == "__main__":
    include_label = "--unlabeled" not in sys.argv
    positional_args = [arg for arg in sys.argv[1:] if arg != "--unlabeled"]
    if len(positional_args) != 2:
        raise SystemExit("Usage: python create_final_df_from_results.py <analysis_dir> <outfile> [--unlabeled]")

    results_folder_analysis = positional_args[0]
    outfile = positional_args[1]
    create_final_dataframe(results_folder_analysis, outfile, include_label=include_label)
