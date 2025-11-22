"""
The code for computing the saturation point of the classifiers.
Basically the copy of ``feature_ranking.do_classification`` with the extension
of having multiple training sets per given test set. The training sets vary in size.
"""

import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange


def get_out_dir(sub="how_much_is_enough"):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), sub)


def convert_to_one_hot(x_data: pd.DataFrame):
    categorical_features = list(x_data.select_dtypes(exclude="number").columns)
    numeric_data = x_data.drop(categorical_features, axis=1)
    parts = [numeric_data]
    feature_groups = {col: [col] for col in numeric_data.columns}
    n_rows = x_data.shape[0]
    for c in categorical_features:
        one_hot = pd.get_dummies(x_data[[c]])
        if len(one_hot.columns) < n_rows:
            print(f"Will include categorical {c}")
            parts.append(one_hot)
            feature_groups[c] = list(one_hot.columns)
    return pd.concat(parts, axis=1), feature_groups


def load_data(path_to_data: str, target_col="label"):
    data = pd.read_csv(path_to_data, sep="\t", index_col="sampleName")
    # inf --> max + 3.14
    for c in data.columns:
        max_val = data[c].replace(np.inf, np.nan).max()
        if isinstance(max_val, str):
            data[c] = data[c].fillna("missing")
        else:
            data[c] = data[c].replace(np.inf, max_val + 3.14).fillna(-666)

    x_columns = list(filter(lambda c: c != target_col, data.columns))
    x_data = data[x_columns]
    y_data = data[target_col]
    x_data, _ = convert_to_one_hot(x_data)
    return x_data, y_data


def extract_parts(sample_name):
    pool_pattern = r"st[-_]{1,2}([^-_]+)[-_]?"
    sub_pool_pat = r"(\d{8}).+p[-_]{1,2}([^-_]+)[-_]?"
    p_match = re.search(pool_pattern, sample_name)
    s_match = re.search(sub_pool_pat, sample_name)
    if p_match is None or s_match is None:
        raise ValueError(f"Sample name {sample_name} is weird.")
    p = p_match.group(1)
    s = (s_match.group(1), s_match.group(2))
    return p, s


def extract_pools(index: pd.Index, r_seed: int):
    pools = {}
    group_ids = []
    for sample_name in index:
        g_id = extract_parts(sample_name)
        p, s = g_id
        if p not in pools:
            pools[p] = set()
        pools[p].add(s)
        group_ids.append(g_id)
    pools = {x: sorted(y) for x, y in pools.items()}
    random.seed(r_seed)
    for p in pools:
        random.shuffle(pools[p])
    assert len({len(ps) for ps in pools.values()}) == 1, {p: len(y) for p, y in pools.items()}
    return pools, group_ids


def filter_data(pools, n_pools, index: list[str], index_test: list[str]):
    ok_positions = []
    test_p: str | None = None
    for i, sample_name in enumerate(index_test):
        p, s = extract_parts(sample_name)
        if s in pools[p][:n_pools]:
            test_p = p
            break
    for i, sample_name in enumerate(index):
        p, s = extract_parts(sample_name)
        up_to = n_pools if p != test_p else n_pools + 1
        if s in pools[p][:up_to]:
            ok_positions.append(i)
    return ok_positions


def do_experiment(path_to_data: str):
    out_dir = get_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    exp_name = os.path.basename(path_to_data)
    out_file = os.path.join(out_dir, f"incremental_bagging_{exp_name}")
    if os.path.exists(out_file):
        return out_file
    x_data, y_data = load_data(path_to_data)
    results = pd.DataFrame(columns=["bazenčki", "primeri", "točnost RF"])
    for r_seed in range(1000, 1001):
        pools, group_ids = extract_pools(x_data.index, r_seed)
        do_classification(x_data, y_data, pools, group_ids, results)
    results = results.groupby(["bazenčki"]).agg(["mean", "std"])
    results.columns = results.columns.map("_".join)
    results.to_csv(out_file, sep="\t", index=False)
    return out_file


def prepare_groups(group_ids: list[tuple[str, str]]):
    unique_ids = sorted(set(group_ids))
    unique = {value: i for i, value in enumerate(unique_ids)}
    group_to_indices = {group: [] for group in unique}
    for i, name in enumerate(group_ids):
        group_to_indices[name].append(i)
    for group, test_indices in group_to_indices.items():
        train_indices = []
        for other_group, other_indices in group_to_indices.items():
            if other_group != group:
                train_indices.extend(other_indices)
        yield train_indices, test_indices


def do_classification(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    pools: dict[str, list[str]],
    group_ids: list[tuple[str, str]],
    results: pd.DataFrame,
):
    model = RandomForestClassifier(n_estimators=200, max_features=1.0, random_state=1234, n_jobs=-1)
    max_pools = max(len(ps) for ps in pools.values())

    sample_names = list(x_data.index)
    x_datav = x_data.values
    y_datav = y_data.values

    for train_ind, test_ind in tqdm(list(prepare_groups(group_ids))):
        x_train = x_datav[train_ind]
        y_train = y_datav[train_ind]
        x_test = x_datav[test_ind]
        y_test = y_datav[test_ind]
        train_sample_names = [sample_names[i] for i in train_ind]
        test_sample_names = [sample_names[i] for i in test_ind]
        for n_pools in trange(1, max_pools + 1, leave=False):
            ok_positions = filter_data(pools, n_pools, train_sample_names, test_sample_names)
            x_train_filt = x_train[ok_positions]
            y_train_filt = y_train[ok_positions]
            model.fit(x_train_filt, y_train_filt)
            y_hat = model.predict(x_test)
            acc = accuracy_score(y_test, y_hat)
            results.loc[len(results)] = {
                "bazenčki": n_pools,
                "primeri": len(ok_positions),
                "točnost RF": acc,
            }
    return results


def show_results(results_file):
    import matplotlib.pyplot as plt

    results = pd.read_csv(results_file, sep="\t")
    ys = results["točnost RF_mean"]
    stds = results["točnost RF_std"]
    xs = results.index + 1
    plt.figure(figsize=(10, 6))
    plt.errorbar(xs, ys, yerr=stds, fmt="o")
    plt.xticks(xs)
    plt.yticks(np.arange(0.0, 1.01, 0.05))
    plt.grid()
    name = os.path.basename(results_file)
    plt.title(f"Točnost +- std za\n{name}")
    plt.xlabel("Število bazenčkov na kategorijo")
    plt.ylabel("Točnost")
    plt.savefig(results_file.replace(".tsv", ".svg"))
    # plt.show()


if __name__ == "__main__":
    this_dir = os.path.abspath(os.path.dirname(__file__))
    base_files = [
        "../prepared_data/2023-09-01-extra-features-final",
        "../prepared_data/2023-08-05-paral2-final",
        "../prepared_data/2023-08-06-extra-features-final_only_custom",
    ]
    files = [base_file + appendix + ".tsv" for base_file in base_files for appendix in [""]]
    for data_file in files:
        print(f"Processing {data_file}")
        the_file = os.path.join(this_dir, data_file)
        out_file = do_experiment(the_file)
        show_results(out_file)
