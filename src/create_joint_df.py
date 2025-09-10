import pandas as pd
import logging
import sys
import tqdm
from collections import defaultdict
import glob

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)


def extract_data(fname, namespace_identifiers):
    for identifier in namespace_identifiers:
        if identifier in fname:
            dfx = pd.read_csv(fname, sep="\t")
            return identifier, dfx
    return None, None


if __name__ == "__main__":
    feature_generator_folder = sys.argv[1]
    results_raw_folder = sys.argv[2]

    # dpath = "../raw_dat_march_23/wetransfer_d_01022023_o_nj_p_d04_s_lm_st_394_gm_lb_gms_ns_sub_nc_t_24_ch_sy9_tret_1um_ista_pozicija-lsm_2023-03-10_1214"
    all_files = glob.glob(f"{feature_generator_folder}/*.txt")
    #    namespace_identifiers = ["CustomAlgos"]
    namespace_identifiers = ["CustomAlgos", "DiffGlobal"]
    #    namespace_identifiers = ["vol3D", "surf3D", "Area2DLayers", "histAll", "CustomAlgos"]

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

    for k, v in all_dfs.items():
        try:
            dfx_f = pd.concat(v, axis=0)
            logging.warn(f"writing {results_raw_folder}/{k}.tsv")
            dfx_f.to_csv(f"{results_raw_folder}/{k}.tsv", sep="\t")
        except Exception as es:
            logging.error(es)
