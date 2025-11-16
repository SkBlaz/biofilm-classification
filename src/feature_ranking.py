import argparse

# from tabpfn import TabPFNClassifier
# import tpot
import logging
import os
import re

import numpy as np

# from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBClassifier

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
np.random.seed(123)


def get_out_dir(sub="ranking_results"):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), sub)


def convert_to_one_hot(x_data: pd.DataFrame):
    """
    Converts every nominal feature x with values {A1, A2, ..., AN} into
    N features.

    :param x_data: data frame to modify
    :return: modified dataframe and feature groups {old name: list of new names, ...}
    """
    categorical_features = list(x_data.select_dtypes(exclude="number").columns)
    numeric_data = x_data.drop(categorical_features, axis=1)
    parts = [numeric_data]
    feature_groups = {col: [col] for col in numeric_data.columns}
    n_rows = x_data.shape[0]
    for c in categorical_features:
        one_hot = pd.get_dummies(x_data[[c]])
        if len(one_hot.columns) < n_rows:
            logger.info(f"Will include categorical {c}")
            parts.append(one_hot)
            feature_groups[c] = list(one_hot.columns)
    return pd.concat(parts, axis=1), feature_groups


class Ranking:
    """
    Superclass of feature ranking methods. Every subclass must implement compute_scores.
    """

    def __init__(self, name):
        self.name = name
        self.features: list[str] | None = None
        self.scores: list[str] | None = None

    def fit(self, xs: pd.DataFrame, y: pd.Series):
        """
        Does some preprocessing (one-hot etc.), computes the scores (needs to be implemented
        in a subclass), and does some paostprocessing - sums up the scores of nominal features
        to obtain the importances of the original features.
        """
        xs, feature_groups = convert_to_one_hot(xs)
        self.features = list(feature_groups)
        self.scores = [0.0 for _ in self.features]
        partial_scores = self.compute_scores(xs, y)
        for i, c_original in enumerate(self.features):
            self.scores[i] = sum(partial_scores[c] for c in feature_groups[c_original])

    def compute_scores(self, xs: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Actually computes the scores for input features ``xs`` and target values ``y``

        :return: a dictionary ``{feature name: feature importance, ...}``
        """
        raise NotImplementedError()

    @property
    def names_and_scores(self):
        return self.features, self.scores


class ForestRanking(Ranking):
    """
    Random forest feature ranking via bagging of 200 trees (by default).
    """

    def __init__(self, n_estimators=200, max_features=1.0):
        super().__init__(f"RandomForest(n={n_estimators}, p={max_features})")
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=1234)

    def compute_scores(self, xs: pd.DataFrame, y: pd.Series):
        self.model.fit(xs.values, y.values)
        return dict(zip(xs.columns, self.model.feature_importances_))


class MutualInformationRanking(Ranking):
    """
    Mutual information feature ranking.
    """

    def __init__(self):
        super().__init__("MutualInformation")

    def compute_scores(self, xs: pd.DataFrame, y: pd.Series):
        scores = mutual_info_classif(xs, y, random_state=1234)
        return dict(zip(xs.columns, scores))


def load_data(path_to_data: str):
    """
    Loads the data into pandas dataframe, replaces nans and infinities
    with something negative and (max of finite) + something, respectively.
    """
    data = pd.read_csv(path_to_data, sep="\t", index_col="sampleName")
    # inf --> max + 3.14
    # nan --> -666
    for c in data.columns:
        max_val = data[c].replace(np.inf, np.nan).max()
        if isinstance(max_val, str):
            data[c] = data[c].fillna("missing")
        else:
            data[c] = data[c].replace(np.inf, max_val + 3.14).fillna(-666)
    # data['noPos'] = ["--".join(x.split("--")[:7]) for x in  data.index.values.tolist()]
    # data.index = range(len(data))
    # data = data.groupby('noPos').max().reset_index()
    # data = data.drop('noPos', axis=1)
    print(data.shape)
    data.to_csv("intermediary_aggregated.tsv", sep="\t")
    return data


def compute_rankings(path_to_data: str, target_col="label", skip: bool = False, fout: str = ""):
    """
    Computes feature rankings for the data found at ``path_to_data``,
    where the target column is ``target_col``.

    Saves the rankings into files (csv and pdf) to the output directory
    (see ``get_out_dir``).
    """
    data = load_data(path_to_data)
    rf_model = ForestRanking()
    mi_model = MutualInformationRanking()
    models = [rf_model, mi_model]
    x_columns = list(filter(lambda c: c != target_col, data.columns))
    x_data = data[x_columns]
    y_data = data[target_col]
    if skip:
        return x_data, y_data, None
    scores_data = pd.DataFrame(index=x_columns, columns=[model.name for model in models], data=0.0)
    for model in models:
        logger.info(f"Computing rankings for {model} - shape: {x_data.shape}")
        model.fit(x_data, y_data)
        for feature, score in zip(*model.names_and_scores):
            scores_data.at[feature, model.name] = score
    scores_data = scores_data.sort_values(rf_model.name, ascending=False)
    out_dir = get_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    file_appendix = os.path.basename(path_to_data)
    file_appendix = file_appendix[: file_appendix.rfind(".")]
    show_rankings(scores_data, os.path.join(out_dir, f"rankings_{file_appendix}.pdf"))

    if fout:
        output_file = fout
    else:
        output_file = os.path.join(out_dir, f"rankings_{file_appendix}.tsv")
    logger.info(f"Saving the ranking results to {output_file}")
    scores_data.reset_index().rename(columns={"index": "feature"}).to_csv(output_file, sep="\t", index=False)
    return x_data, y_data, scores_data


def show_rankings(rankings_data: pd.DataFrame, out_file: str):
    """Creates a bar-plot of feature importances and saves it to the output file."""
    ax = rankings_data.plot.bar(rot=90, figsize=(60, 8))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_file)


def name_manipulator(value: str):
    """
    Extracts (L628, C03, 001, 24) from the names such as

    ``12042023_s_Lm_st_L628_p_C03_pos001_tm_24_ch_Syto9_z_11_`` -> This we deprecated ..

    or

    --> This is theconvention indeed
    ``13042023--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21``

    Used to compute training/testing examples. This extractor will produce
    optimistic splits (it turns out that the same date for given sev leaks some
    data from, for example, bazenček C03 to bazenček C04).
    """

    return value.split("--")[4]
    # TODO -> get rid of this hack and unify beforehand, this is an antipattern
    # pattern = re.compile(r"st[-_]+(\w+)[-_]+p[-_]+(\w+)[-_]+pos(\d+)[-_]+tm[-_]+(\d+)")

    # match = pattern.search(value)
    # if match is None:
    #     raise ValueError(f"Sample name {value} is weird.")
    # return tuple(match.group(i) for i in [1, 2, 4])


def name_manipulator_date(value: str):
    """
    Extracts the date and sev from the names as in the ``name_manipulator``.
    This computes better training/test splits.
    """
    return (value[:8], re.search(r"st--([^-_]+)-", value).group(1))


def human_grouping(sample_names: pd.Series):
    """
    Uses file human_split.csv to load only some examples (80) and split them
    into train:test (40:40).
    """
    name_to_i = {name: i for i, name in enumerate(sample_names)}
    train_indices = []
    test_indices = []
    human_split_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_split.csv")
    with open(human_split_file, encoding="utf-8") as f:
        for line in f:
            example, part = line.strip().split(",")
            if example not in name_to_i:
                raise ValueError(f"Example {example} is not present in your data.")
            i = name_to_i[example]
            if part == "tr":
                train_indices.append(i)
            elif part == "te":
                test_indices.append(i)
            else:
                raise ValueError(f"Wrong part: 'tr' or 'te' expected, got '{part}'")
    return train_indices, test_indices


def prepare_groups(sample_name_data: pd.Series, splitter: str):
    """
    Prepares groups of training and testing examples. Uses one of the
    name manipulators, acording to the plitter value.
    """
    if splitter == "optimistic":
        y_transformed = sample_name_data.apply(name_manipulator)
    elif splitter == "date":
        y_transformed = sample_name_data.apply(name_manipulator_date)
    elif splitter == "human":
        yield human_grouping(sample_name_data)
        return
    else:
        raise ValueError(f"Wrong splitter: {splitter} (optimistic, date, human allowed)")

    unique = {value: i for i, value in enumerate(y_transformed.unique())}
    group_to_indices = {group: [] for group in unique}
    for i, name in enumerate(y_transformed):
        group_to_indices[name].append(i)
    # group_sizes = {group: len(value) for group, value in group_to_indices.items()}
    # logger.info(f"Data groups: {group_sizes}")
    for group, test_indices in group_to_indices.items():
        train_indices = []
        for other_group, other_indices in group_to_indices.items():
            if other_group != group:
                train_indices.extend(other_indices)
        yield train_indices, test_indices


def do_classification_simple(X, ys):
    X = X.values
    y = pd.Categorical(ys.values).codes
    catmap = dict(zip(y, ys.values))
    skf = StratifiedKFold(n_splits=10)
    upsampling = 1
    n_components = 32
    # tpot.TPOTClassifier(generations=10, population_size=10, verbosity=2, config_dict="TPOT NN"), TabPFNClassifier(device='cpu', N_ensemble_configurations=32), XGBClassifier()
    models = [RandomForestClassifier(), DummyClassifier()]
    # models = [RandomForestClassifier(), DummyClassifier(), tpot.TPOTClassifier(generations=10, population_size=10, verbosity=2, config_dict="TPOT NN"), TabPFNClassifier(device='cpu', N_ensemble_configurations=32), XGBClassifier()]
    for reduce_dim in [True, False]:
        for model in models:
            for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                x_train = X[train_index]
                x_test = X[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                if reduce_dim or "TabPFN" in str(model):
                    svd = TruncatedSVD(n_components=n_components, n_iter=15, random_state=42).fit(x_train)
                    x_train = svd.transform(x_train)
                    x_test = svd.transform(x_test)
                else:
                    n_components = x_train.shape[1]

                model.fit(x_train, y_train)
                y_hat = model.predict(x_test)
                acc = accuracy_score(y_test, y_hat)
                test_map = ",".join([catmap[x] for x in y_test])
                output = ["RESULT", str(model).replace("\n", ""), upsampling, n_components, i, acc, test_map]
                print("\t".join(str(x) for x in output))


def do_classification(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    in_file: str,
    model_names: list[str],
    splitter: str,
    extension: str,
):
    """
    Uses bagging of 200 trees (and the dummy classifier) to predict the values of
    ``y_data`` from ``x_data``.

    Input file ``in_file`` is only used to compute the experiment name and the name of
    the output files, which are extended by ``extension``.
    """
    # prepare models and results data
    models = {
        "dummy": DummyClassifier(),
        "bagging": RandomForestClassifier(n_estimators=200, max_features=1.0, random_state=1234),
        "LR": LogisticRegression(max_iter=1000, random_state=1234),
        # "tpot": tpot.TPOTClassifier(
        #     generations=10, population_size=10, verbosity=2, config_dict="TPOT NN"
        # ),
    }
    models = {name: model for name, model in models.items() if name in model_names}
    {model: [] for model in models}
    list(x_data.index)
    # preprocess data
    logger.info("Converting to one-hot")
    x_data, _ = convert_to_one_hot(x_data)
    sample_names = x_data.index.to_series()
    list(x_data.columns)
    x_data = x_data.values
    y_data = y_data.values
    # data frame of wrong predictions
    pd.DataFrame(columns=["group", "name", "true", "predicted", "decision makers"])

    # for name, model in models.items():
    #     # for each model
    #     logger.info(f"Learning {name}")
    #     upsampling = 15
    #     n_components = 10
    encoding = {}
    for enx, el in enumerate(set(y_data.tolist())):
        encoding[el] = enx
    y_data = np.array([encoding[x] for x in y_data])

    for model in [RandomForestClassifier(), LogisticRegression(max_iter=1000), DummyClassifier(), XGBClassifier()]:
        for n_components in range(1, 25):
            for upsampling in [1, 5, 10, 15]:
                for train_ind, test_ind in tqdm(prepare_groups(sample_names, splitter)):
                    # and each training-testing split

                    x_train = x_data[train_ind].copy()
                    y_train = y_data[train_ind].copy()
                    x_test = x_data[test_ind].copy()
                    y_test = y_data[test_ind].copy()
                    # learn the model
                    x_train = np.repeat(x_train, upsampling, axis=0)
                    y_train = np.repeat(y_train, upsampling, axis=0)
                    perm = np.random.permutation(x_train.shape[0])
                    x_train = x_train[perm]
                    y_train = y_train[perm]
                    svd = TruncatedSVD(n_components=n_components, n_iter=15, random_state=42).fit(x_train)
                    x_train = svd.transform(x_train)
                    x_test = svd.transform(x_test)

                    model.fit(x_train, y_train)
                    y_hat = model.predict(x_test)
                    # evaluate it
                    acc = np.round(accuracy_score(y_test, y_hat), 4)
                    print("RESULT", acc, n_components, upsampling, model)
    #         if name == "bagging":
    #             # and report misses
    #             for i, (true, predicted) in enumerate(zip(y_test, y_hat)):
    #                 decision_makers = forest_by_hand(model, attribute_names, x_test[i])
    #                 if true != predicted:
    #                     example_name = names[test_ind[i]]
    #                     df_wrong.loc[len(df_wrong)] = {
    #                         "group": name_manipulator(example_name),
    #                         "name": example_name,
    #                         "true": true,
    #                         "predicted": predicted,
    #                         "decision makers": decision_makers,
    #                     }
    #         results[name].append(acc)
    #     # save the results: accuracy and misses
    #     out_dir = get_out_dir("classification_results")
    #     os.makedirs(out_dir, exist_ok=True)
    #     exp_name = os.path.basename(in_file)
    #     exp_name = exp_name[: exp_name.rfind(".")] + extension
    #     out_file = os.path.join(out_dir, f"classification_bagging_{exp_name}.tsv")
    #     with open(out_file, "w", encoding="utf-8") as f:
    #         print("model;upsampling;dim;mean accuracy;std accuracy", file=f)
    #         for name, values in results.items():
    #             acc = np.mean(values)
    #             std = np.std(values)
    #             print(f"RESULT;{name};{upsampling};{n_components};{acc};{std}")
    #             print(f"{name};{upsampling};{n_components};{acc};{std}", file=f)
    # out_file_wrong = os.path.join(out_dir, f"missed_{exp_name}.tsv")
    # df_wrong.to_csv(out_file_wrong, sep="\t", index=False)


if __name__ == "__main__":
    # let's turn this beast into a command-line puppy.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Prepared datasets to analyze (paths to them)",
    )
    parser.add_argument("--data-modifiers", nargs="*", default=[], help="For example, 'z--11 z--21'")
    parser.add_argument(
        "--splitter",
        default="date",
        choices=["human", "date", "optimistic"],
        help="Which split to use",
    )
    parser.add_argument(
        "--fext",
        default="",
        help="File extension for the output files, e.g., '_final_experiments'. Ignored in ranking outputs if fout is specified as well.",
    )
    parser.add_argument(
        "--models",
        default=["dummy", "bagging"],
        choices=["dummy", "bagging", "tpot", "LR"],
        nargs="*",
        help="Which ML models do you want to run?",
    )
    parser.add_argument(
        "--fout",
        default="",
        help="Ranking output file, e.g., 'ranking.out' (specify the full path)",
    )

    try:
        arguments = parser.parse_args()
    except SystemExit:
        parser.print_help()
        exit(999)

    files = [
        base_file[: base_file.rfind(".")] + appendix + ".tsv"
        for base_file in arguments.files
        for appendix in [""] + arguments.data_modifiers
    ]
    # for every file, compute feature ranking and accuracy
    simple_classif = True
    for file in files:
        logger.info(f"Processing {file}")
        xs, y, _ = compute_rankings(file, skip=False, fout=arguments.fout)

        if simple_classif:
            # just 10fCV
            do_classification_simple(xs, y)
        # do_classification(
        #     xs, y, file, arguments.models, arguments.splitter, arguments.fext
        # )
