import glob
import logging
import sys

import pandas as pd
import tqdm

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)

STATISTICS = [
    "median",
    "min",
    "max",
    "mean",
    "q10",
    "q25",
    "q75",
    "q90",
    "std",
    "var",
]


def aggregate_raw_features(results_raw_folder, results_analysis_folder, statistics=None):
    if statistics is None:
        statistics = STATISTICS

    output_files = []
    for fname in tqdm.tqdm(glob.glob(f"{results_raw_folder}/*")):
        s3d = fname
        NAME = "SUMMARY" + fname.split("/")[-1].replace(".txt", "")
        logging.info("Processing {fname} -- computing aggregates")
        try:
            for statistic in statistics:
                dfx = pd.read_csv(s3d, sep="\t")
                if statistic == "median":
                    dfx2 = dfx.groupby(["sampleName"]).median().reset_index()
                elif statistic == "min":
                    dfx2 = dfx.groupby(["sampleName"]).min().reset_index()
                elif statistic == "max":
                    dfx2 = dfx.groupby(["sampleName"]).max().reset_index()
                elif statistic == "mean":
                    dfx2 = dfx.groupby(["sampleName"]).mean().reset_index()
                elif statistic == "q10":
                    dfx2 = dfx.groupby(["sampleName"]).quantile(0.10).reset_index()
                elif statistic == "q25":
                    dfx2 = dfx.groupby(["sampleName"]).quantile(0.25).reset_index()
                elif statistic == "q75":
                    dfx2 = dfx.groupby(["sampleName"]).quantile(0.75).reset_index()
                elif statistic == "q90":
                    dfx2 = dfx.groupby(["sampleName"]).quantile(0.90).reset_index()
                elif statistic == "std":
                    dfx2 = dfx.groupby(["sampleName"]).std().reset_index()
                elif statistic == "var":
                    dfx2 = dfx.groupby(["sampleName"]).var().reset_index()

                outfile = f"{results_analysis_folder}/{NAME}_{statistic}.txt"
                logging.info(f"writing {outfile}")
                dfx2.to_csv(outfile, sep="\t", index=False)
                output_files.append(outfile)

        except Exception as es:
            logging.error(es)

    logging.info("Done!")
    return output_files


if __name__ == "__main__":
    results_raw_folder = sys.argv[1]
    results_analysis_folder = sys.argv[2]

    aggregate_raw_features(results_raw_folder, results_analysis_folder)
