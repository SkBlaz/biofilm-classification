import pandas as pd
import tqdm
import glob
import sys
import logging

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)


if __name__ == "__main__":
    results_raw_folder = sys.argv[1]
    results_analysis_folder = sys.argv[2]

    for fname in tqdm.tqdm(glob.glob(f"{results_raw_folder}/*")):
        s3d = fname
        NAME = "SUMMARY" + fname.split("/")[-1].replace(".txt", "")
        logging.info(f"Processing {fname} -- computing aggregates")
        try:
            for statistic in [
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
            ]:
                dfx = pd.read_csv(s3d, sep="\t")
                
                # Map statistic names to pandas methods
                stat_methods = {
                    "median": lambda df: df.groupby(["sampleName"]).median(),
                    "min": lambda df: df.groupby(["sampleName"]).min(),
                    "max": lambda df: df.groupby(["sampleName"]).max(),
                    "mean": lambda df: df.groupby(["sampleName"]).mean(),
                    "q10": lambda df: df.groupby(["sampleName"]).quantile(0.10),
                    "q25": lambda df: df.groupby(["sampleName"]).quantile(0.25),
                    "q75": lambda df: df.groupby(["sampleName"]).quantile(0.75),
                    "q90": lambda df: df.groupby(["sampleName"]).quantile(0.90),
                    "std": lambda df: df.groupby(["sampleName"]).std(),
                    "var": lambda df: df.groupby(["sampleName"]).var(),
                }
                
                dfx2 = stat_methods[statistic](dfx).reset_index()

                outfile = f"{results_analysis_folder}/{NAME}_{statistic}.txt"
                logging.info(f"writing {outfile}")
                dfx2.to_csv(outfile, sep="\t", index=False)

        except Exception as es:
            logging.error(es)

    logging.info("Done!")
