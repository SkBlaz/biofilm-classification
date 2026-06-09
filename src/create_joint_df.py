import glob
import logging
import sys
from collections import defaultdict

import pandas as pd
import tqdm

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)


def extract_data(fname, namespace_identifiers):
    for identifier in namespace_identifiers:
        if identifier in fname:
            dfx = pd.read_csv(fname, sep="\t")
            return identifier, dfx
    return None, None


def create_joint_dataframe(feature_generator_folder, results_raw_folder, namespace_identifiers=None):
    if namespace_identifiers is None:
        namespace_identifiers = ["CustomAlgos", "DiffGlobal"]

    all_files = glob.glob(f"{feature_generator_folder}/*.txt")

    all_dfs = defaultdict(list)
    for fname in tqdm.tqdm(all_files[:]):
        logging.info(f"Processing {fname} ..")
        namespace, df = extract_data(fname, namespace_identifiers)
        if df is not None:
            sample_name = fname.split("/")[-1]
            for el in namespace_identifiers:
                sample_name = sample_name.replace(el, "")
            sample_name = sample_name.replace(".txt", "")
            sample_name = "--".join(sample_name.split("_"))

            df["sampleName"] = sample_name
            all_dfs[namespace].append(df.iloc[:100000])

    outputs = {}
    for k, v in all_dfs.items():
        try:
            dfx_f = pd.concat(v, axis=0)
            output_file = f"{results_raw_folder}/{k}.tsv"
            logging.warning(f"writing {output_file}")
            dfx_f.to_csv(output_file, sep="\t")
            outputs[k] = dfx_f
        except Exception as es:
            logging.error(es)
    return outputs


if __name__ == "__main__":
    feature_generator_folder = sys.argv[1]
    results_raw_folder = sys.argv[2]

    create_joint_dataframe(feature_generator_folder, results_raw_folder)
