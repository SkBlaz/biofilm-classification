import argparse
import gc
import logging
import os
import re
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, RandomizedSearchCV, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - only older scikit-learn versions
    StratifiedGroupKFold = None
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

try:
    from benchmark_outputs import write_confusion_matrices, write_feature_boxplots, write_feature_correlation
except ImportError:  # Package import from the repository root.
    from .benchmark_outputs import write_confusion_matrices, write_feature_boxplots, write_feature_correlation

warnings.simplefilter(action="ignore", category=DataConversionWarning)

try:
    from tpot import TPOTClassifier

    TPOT_AVAILABLE = False
except ImportError:
    TPOT_AVAILABLE = False
    TPOTClassifier = None

try:
    from autogluon.tabular import TabularPredictor

    AUTOGLUON_AVAILABLE = False
except ImportError:
    AUTOGLUON_AVAILABLE = False
    TabularPredictor = None

if not TPOT_AVAILABLE or not AUTOGLUON_AVAILABLE:
    print("Skipping tpot and autogluon, uncomment to enable (takes a lot of time)")


parameters = {
    "n_neighbors": list(range(3, 50, 2)),  # Test odd values for better balancing in ties
    "metric": ["minkowski", "euclidean", "manhattan", "chebyshev", "hamming", "jaccard"],
    "weights": ["uniform", "distance"],  # Test both weighting strategies
    "p": [1, 2, 3],  # Minkowski distance with Manhattan (1), Euclidean (2), etc.
}


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
np.random.seed(123)

PARALLELISM = -1
MIN_CV_SPLITS = 2
PARTIAL_CACHE_VERSION = 2


def get_benchmark_runtime_config():
    """Return runtime settings for the learning benchmark."""
    if os.getenv("CI", "").lower() == "true":
        return {
            "n_iter": 2,
            "repetitions": 1,
            "n_components": ["all"],
        }
    return {
        "n_iter": 10,
        "repetitions": 3,
        "n_components": [16, 32, 64, 128, 256, 512, "all"],
    }


def configured_parallelism() -> int:
    """Return the requested worker count for estimators and model searches."""
    return PARALLELISM if PARALLELISM == -1 else max(1, PARALLELISM)


def partial_evaluation_path(partial_dir, filter_mode, repetition, n_components, thr_features, model_name, fold, replication_unit):
    """Return a cache path that cannot mix incompatible CV strategies."""
    replication_tag = replication_unit or "none"
    filename = (
        f"v{PARTIAL_CACHE_VERSION}_{filter_mode}_partial_{repetition}_n{n_components}_"
        f"thr{thr_features}_{model_name}_rep{replication_tag}_fold{fold}.tsv"
    )
    return os.path.join(partial_dir, filename)


def prepare_fold_features(x_train, x_test, n_components, thr_features, thr_indices):
    """Prepare one fold once so every learner reuses the same transformed arrays."""
    if not thr_features and n_components == "all" and len(thr_indices) > 0:
        x_train = x_train[:, thr_indices]
        x_test = x_test[:, thr_indices]

    svd_transformer = None
    if n_components != "all":
        svd_transformer = TruncatedSVD(n_components=int(n_components), n_iter=15, random_state=42).fit(x_train)
        x_train = svd_transformer.transform(x_train)
        x_test = svd_transformer.transform(x_test)

    return x_train, x_test, svd_transformer, x_train.shape[1]


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


def load_data(path_to_data: str):
    """
    Loads the data into pandas dataframe, replaces nans and infinities
    with something negative and (max of finite) + something, respectively.
    """
    data = pd.read_csv(path_to_data, sep="\t", index_col="sampleName")
    # inf --> max + 3.14
    # nan --> -666
    for c in data.columns:
        if pd.api.types.is_numeric_dtype(data[c]):
            finite = data[c].replace([np.inf, -np.inf], np.nan)
            max_val = finite.max()
            if pd.isna(max_val):
                data[c] = finite.fillna(-666)
            else:
                data[c] = data[c].replace([np.inf, -np.inf], [max_val + 3.14, -666]).fillna(-666)
        else:
            data[c] = data[c].fillna("missing")

    if "label" in data.columns:
        missing_label_mask = data["label"].isna() | (data["label"].astype(str).str.strip().str.lower() == "missing")
        if missing_label_mask.any():
            dropped_count = int(missing_label_mask.sum())
            logger.warning(f"Dropping {dropped_count} rows with missing 'label' from {path_to_data}")
            data = data.loc[~missing_label_mask].copy()

    data = data.copy()

    # Using haccs to get poss aggregated ..
    data["noPos"] = ["--".join([j for j in x.split("--") if "pos" not in j]) for x in data.index.values.tolist()]
    # data.index = range(len(data))
    # cols = [x for x in data.columns if x != 'label']
    # fil = {x: 'mean' for x in cols}
    # fil['label'] = 'first'
    # fil['noPos'] = 'first'
    # data = data.groupby('noPos').agg(fil)
    data.to_csv(path_to_data + "intermediary_aggregated.tsv", sep="\t")
    return data.drop(["noPos"], axis=1)


#    return data


def validate_target_labels(data: pd.DataFrame, target_col: str, source_file: str):
    """Fail early if the supervised target contains missing labels."""
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in {source_file}. Available columns: {list(data.columns)}")

    target = data[target_col]
    missing_mask = target.isna() | (target.astype(str).str.strip().str.lower() == "missing")
    if missing_mask.any():
        missing_samples = data.index[missing_mask].astype(str).tolist()
        preview = ", ".join(missing_samples[:10])
        more = "" if len(missing_samples) <= 10 else f", ... ({len(missing_samples)} total)"
        raise ValueError(
            f"Target column '{target_col}' in {source_file} contains missing labels for: {preview}{more}. "
            "Fix or remove these rows before training; missing labels must not become a learned class."
        )


def compute_rankings(data: str, path_to_data: str, target_col="label", skip: bool = False, fout: str = ""):
    """
    Computes feature rankings for the data found at ``path_to_data``,
    where the target column is ``target_col``.

    Saves the rankings into files (csv and pdf) to the output directory
    (see ``get_out_dir``).
    """
    validate_target_labels(data, target_col, path_to_data)

    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    fout = base_dir + f"/rankings_{target_col}.tsv"
    logger.info(fout)

    output_file = fout

    x_columns = list(filter(lambda c: c != target_col, data.columns))
    x_columns = list(filter(lambda c: c != "Unnamed:", x_columns))
    x_columns = list(filter(lambda c: c != "Unnamed: ", x_columns))
    x_data = data[x_columns]
    y_data = data[target_col]

    if os.path.isfile(output_file):
        logger.info(f"Loading ranking from {output_file}")
        scores_data = pd.read_csv(output_file, sep="\t")
        return x_data, y_data, scores_data

    rf_model = ForestRanking()
    models = [rf_model]

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


def name_manipulator_date(value: str):
    """
    Extracts the date and sev from the names as in the ``name_manipulator``.
    This computes better training/test splits.
    """
    return (value[:8], re.search(r"st--([^-_]+)-", value).group(1))


def get_adaptive_cv(y, max_splits=5, groups=None):
    """Build an adaptive CV splitter for classification targets.

    Returns a tuple ``(cv_splitter, n_splits, min_class_count, strategy, reason)``.
    Uses ``StratifiedKFold`` when each class has at least ``MIN_CV_SPLITS`` samples;
    otherwise falls back to ``KFold`` with the same adaptive split count.
    """
    y = np.asarray(y)
    n_samples = len(y)
    if n_samples < MIN_CV_SPLITS:
        raise ValueError(f"Cannot create cross-validation splitter: at least {MIN_CV_SPLITS} samples are required, got {n_samples}")

    group_values = np.asarray(groups) if groups is not None else None
    if group_values is not None and len(group_values) != n_samples:
        raise ValueError("Replication groups must have one value per sample")
    unique_groups = np.unique(group_values) if group_values is not None else None
    if unique_groups is not None and len(unique_groups) < MIN_CV_SPLITS:
        raise ValueError(f"Cannot create grouped cross-validation splitter: at least {MIN_CV_SPLITS} groups are required")
    target_splits = min(max_splits, max(MIN_CV_SPLITS, n_samples // 2))
    if unique_groups is not None:
        target_splits = min(target_splits, len(unique_groups))
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()

    if unique_groups is not None:
        n_splits = min(target_splits, len(unique_groups))
        if StratifiedGroupKFold is not None and min_class_count >= MIN_CV_SPLITS:
            class_count_info = ", ".join(str(count) for count in sorted(class_counts))
            reason = (
                f"StratifiedGroupKFold enabled for {len(unique_groups)} replication groups; selected {n_splits} folds "
                f"(class counts: {class_count_info})"
            )
            return (
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42),
                n_splits,
                min_class_count,
                "stratified-group",
                reason,
            )
        reason = f"GroupKFold enabled for {len(unique_groups)} replication groups; selected {n_splits} folds"
        logger.warning(reason)
        return GroupKFold(n_splits=n_splits), n_splits, min_class_count, "group", reason

    if min_class_count >= MIN_CV_SPLITS:
        class_count_info = ", ".join(str(count) for count in sorted(class_counts))
        n_splits = min(target_splits, min_class_count)
        reason = (
            f"StratifiedKFold enabled because minimum class count is {min_class_count}; selected {n_splits} folds from target {target_splits} "
            f"(class counts: {class_count_info})"
        )
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits, min_class_count, "stratified", reason

    class_count_info = ", ".join(str(count) for count in sorted(class_counts))
    reason = (
        f"Falling back to KFold because minimum class count is {min_class_count}; using {target_splits} folds "
        f"with minimum class count {min_class_count} (class counts: {class_count_info})"
    )
    logger.warning(reason)
    return KFold(n_splits=target_splits, shuffle=True, random_state=42), target_splits, min_class_count, "kfold", reason


def do_classification_simple(
    X, ys, path_to_data, filter_mode="all", save_models=False, all_learners=False, replication_unit=None, learner="rf"
):
    all_cols = X.columns
    sample_names = X.index.astype(str).tolist()
    thr_indices = []
    for enx, x in enumerate(all_cols):
        if "Threshold" in x:
            thr_indices.append(enx)
    thr_indices = np.array(thr_indices)
    X = X.values
    y = pd.Categorical(ys.values).codes
    catmap = dict(zip(y, ys.values))
    upsampling = 1

    # Define hyperparameter grid for RandomForest tuning
    rf_param_dist = {
        "n_estimators": np.arange(100, 1001, 100),
        "max_features": ["sqrt", "log2", None] + list(np.arange(0.1, 0.6, 0.1)),
        "max_depth": [None] + list(np.arange(5, 31, 5)),
        "min_samples_split": np.arange(2, 21, 2),
        "min_samples_leaf": np.arange(1, 11, 1),
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    # Create tuned RandomForest using RandomizedSearchCV
    # Adapt CV folds and iterations based on dataset and class size
    n_samples = X.shape[0]
    group_ids = None
    if replication_unit:
        try:
            from input_validation import replication_group
        except ImportError:
            from .input_validation import replication_group

        group_ids = np.array([replication_group(name, replication_unit) for name in sample_names])
    cv_rf, n_splits, min_class_count, cv_strategy, cv_reason = get_adaptive_cv(y, max_splits=5, groups=group_ids)
    runtime_config = get_benchmark_runtime_config()
    n_iter = runtime_config["n_iter"]
    logger.info(
        f"Using {n_splits} CV folds and {n_iter} iterations for dataset with {n_samples} samples "
        f"(strategy: {cv_strategy}, minimum class count: {min_class_count}). {cv_reason}"
    )

    search_jobs = configured_parallelism()
    # Parallelize the search, not both levels. Nested ``n_jobs=-1`` caused each
    # outer worker to start another full worker pool for large feature tables.
    rf_base = RandomForestClassifier(random_state=42, n_jobs=1)
    tuned_rf = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        n_iter=n_iter,
        scoring=make_scorer(f1_score, average="weighted"),
        cv=cv_rf,
        verbose=0,  # Reduced verbosity for faster execution
        random_state=42,
        n_jobs=search_jobs,
        pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
    )

    if all_learners:
        models = {
            "dummy": DummyClassifier(),
            "decisiontree": tree.DecisionTreeClassifier(),
            "logistic": LogisticRegression(),
            "rf": tuned_rf,
            "xgb": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1, objective="binary:logistic", n_jobs=search_jobs),
            "gridsearch": GridSearchCV(
                KNeighborsClassifier(),
                parameters,
                n_jobs=search_jobs,
                pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
            ),
            #'tpot': TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, n_jobs=PARALLELISM, memory='auto'),
        }

        # Add TPOT only if available
        if TPOT_AVAILABLE:
            models["tpot"] = TPOTClassifier(
                generations=5,
                population_size=20,
                cv=5,
                random_state=42,
                verbosity=2,
                n_jobs=PARALLELISM,
                memory="auto",
            )
    else:
        # Default behavior: only RandomForest (fast)
        learner_models = {
            "rf": tuned_rf,
            "dummy": DummyClassifier(),
            "decisiontree": tree.DecisionTreeClassifier(),
            "logistic": LogisticRegression(),
            "xgb": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1, objective="binary:logistic", n_jobs=search_jobs),
            "gridsearch": GridSearchCV(
                KNeighborsClassifier(),
                parameters,
                n_jobs=search_jobs,
                pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
            ),
        }
        if learner not in learner_models:
            raise ValueError(f"Unsupported learner: {learner}")
        models = {learner: learner_models[learner]}

    # Add autogluon model only if available
    if AUTOGLUON_AVAILABLE:
        models["autogluon"] = TabularPredictor(label="label")

    outputs = []
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    partial_dir = base_dir + "/partial/"
    if not os.path.isdir(partial_dir):
        os.mkdir(partial_dir)

    # Store CV results to find best hyperparameters for final model training
    cv_results = []
    failed_models = []

    skf, outer_n_splits, outer_min_class_count, outer_cv_strategy, outer_cv_reason = get_adaptive_cv(y, max_splits=3, groups=group_ids)
    logger.info(
        f"Outer evaluation uses {outer_n_splits} folds (strategy: {outer_cv_strategy}, "
        f"minimum class count: {outer_min_class_count}). {outer_cv_reason}"
    )

    for repetition in range(runtime_config["repetitions"]):
        for n_components in runtime_config["n_components"]:
            desc_components = n_components

            # Skip if n_components exceeds available features
            if n_components != "all" and n_components > X.shape[1]:
                logger.info(f"Skipping n_components={n_components} as it exceeds available features ({X.shape[1]})")
                continue

            for thr_features in [True, False]:
                split_iterator = skf.split(X, y, group_ids) if group_ids is not None else skf.split(X, y)
                for i, (train_index, test_index) in enumerate(split_iterator):
                    x_train = X[train_index]
                    x_test = X[test_index]
                    y_train = y[train_index]
                    y_test = y[test_index]
                    groups_train = group_ids[train_index] if group_ids is not None else None

                    x_train_model, x_test_model, svd_transformer, effective_components = prepare_fold_features(
                        x_train,
                        x_test,
                        desc_components,
                        thr_features,
                        thr_indices,
                    )

                    for model_name, model in models.items():
                        partial_path = partial_evaluation_path(
                            partial_dir,
                            filter_mode,
                            repetition,
                            desc_components,
                            thr_features,
                            model_name,
                            i,
                            replication_unit,
                        )

                        if os.path.isfile(partial_path):
                            with open(partial_path) as f:
                                output = f.read().strip().split("\t")
                            if len(output) == 8:  # partials written by pre-resume versions
                                output.append("")
                            outputs.append(output)
                            if save_models:
                                cv_results.append(
                                    {
                                        "model_name": model_name,
                                        "repetition": repetition,
                                        "n_components": desc_components,
                                        "thr_features": thr_features,
                                        "fold": i,
                                        "accuracy": float(output[5]),
                                        "svd_transformer": None,
                                    }
                                )
                            logger.info(f"Loaded existing partial evaluation from {partial_path}, skipping model evaluation")
                            continue

                        model_for_fold = clone(model)
                        if isinstance(model, RandomizedSearchCV | GridSearchCV):
                            cv_inner, n_splits_inner, min_class_count_inner, inner_cv_strategy, inner_cv_reason = get_adaptive_cv(
                                y_train, max_splits=5, groups=groups_train
                            )
                            logger.info(
                                f"Using {n_splits_inner} inner CV folds for {model_name} on fold {i} "
                                f"(strategy: {inner_cv_strategy}, minimum class count: {min_class_count_inner}). {inner_cv_reason}"
                            )
                            model_for_fold.set_params(cv=cv_inner)

                        logger.info(
                            f"Running {desc_components} ({effective_components} features) "
                            f"{' '.join(str(model_for_fold).split())}, fold: {i}, filter mode: {filter_mode}"
                        )

                        model_failed = False
                        if "TabularPredictor" in str(model_for_fold) and AUTOGLUON_AVAILABLE:
                            # if desc_components == "all":
                            #    continue
                            x_train_ag = pd.DataFrame(x_train_model)
                            x_test_ag = pd.DataFrame(x_test_model)
                            y_train_ag = pd.DataFrame(y_train)
                            y_test_ag = pd.DataFrame(y_test)
                            y_train_ag.columns = ["label"]
                            y_test_ag.columns = ["label"]

                            train_data = pd.concat([x_train_ag, y_train_ag], axis=1)
                            test_data = pd.concat([x_test_ag, y_test_ag], axis=1)

                            model_for_fold = TabularPredictor(label="label")
                            predictor = (
                                model_for_fold.fit(train_data, ag_args_fit={"num_cpus": PARALLELISM})
                                if PARALLELISM != -1
                                else model_for_fold.fit(train_data)
                            )
                            # predictor = model.fit(train_data)
                            y_hat = predictor.predict(test_data)
                            acc = accuracy_score(y_test_ag, y_hat)
                            mname = str(model_for_fold)
                            del model_for_fold
                            model_for_fold = mname
                            gc.collect()
                        elif "TabularPredictor" in str(model_for_fold) and not AUTOGLUON_AVAILABLE:
                            # Skip autogluon if not available
                            logger.warning(f"Skipping {model_name} - autogluon not available")
                            continue
                        else:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    fit_kwargs = (
                                        {"groups": groups_train}
                                        if groups_train is not None and isinstance(model_for_fold, RandomizedSearchCV | GridSearchCV)
                                        else {}
                                    )
                                    model_for_fold.fit(x_train_model, y_train, **fit_kwargs)
                                    y_hat = model_for_fold.predict(x_test_model)
                            except Exception as e:
                                model_failed = True
                                failed_models.append(f"{model_name} fold {i}: {e}")
                                logger.warning(
                                    f"Repetition {repetition} with {desc_components} components (THR: {thr_features}) model {model_name} fold {i} raised {e} (filter mode {filter_mode})"
                                )
                                y_hat = np.ones(len(x_test_model))

                            acc = accuracy_score(y_test, y_hat)

                        if model_failed:
                            continue

                        # Store CV results for later use in final model training
                        if save_models:
                            cv_results.append(
                                {
                                    "model_name": model_name,
                                    "repetition": repetition,
                                    "n_components": desc_components,
                                    "thr_features": thr_features,
                                    "fold": i,
                                    "accuracy": acc,
                                    "svd_transformer": svd_transformer,
                                }
                            )

                        test_map = ",".join([str(catmap[x]) for x in y_test])
                        prediction_map = ",".join([str(catmap.get(int(x), x)) for x in np.asarray(y_hat)])
                        output = [
                            "RESULT",
                            model_name,
                            upsampling,
                            effective_components,
                            i,
                            acc,
                            test_map,
                            thr_features,
                            prediction_map,
                        ]
                        with open(partial_path, "w") as f:
                            f.write("\t".join([str(x) for x in output]))
                            logger.info(f"Stored partial evaluation to {partial_path}")
                        outputs.append([str(x) for x in output])
    if failed_models:
        details = "; ".join(failed_models[:5])
        logger.error(f"Benchmark incomplete; retaining {partial_dir} for resume. Failures: {details}")
        raise RuntimeError(f"Benchmark incomplete ({len(failed_models)} failed evaluations). Re-run to resume from partial results.")
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)
        logger.info("All model evaluation complete, deleted partial results")
    dfx = pd.DataFrame(outputs)
    dfx.columns = [
        "tag",
        "model",
        "upsampling",
        "n_components",
        "fold",
        "accuracy",
        "test_set",
        "thr_features",
        "predicted_set",
    ]
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    fout = base_dir + f"/classification_{filter_mode}.tsv"
    dfx = dfx.sort_values(by=["accuracy"])
    dfx.to_csv(fout, sep="\t")

    logger.info(f"Wrote classification outputs to {fout}")

    # Train final models on all data using best hyperparameters from cross-validation
    if save_models and cv_results:
        logger.info("Training final models on all data using best hyperparameters from cross-validation")

        # Convert cv_results to DataFrame for easier analysis
        cv_df = pd.DataFrame(cv_results)

        # Find best hyperparameters for each model (prioritize 'all' features and thr_features=True)
        best_configs = {}
        for model_name in cv_df["model_name"].unique():
            model_results = cv_df[cv_df["model_name"] == model_name]

            # First try with 'all' features and thr_features=True
            preferred_results = model_results[(model_results["n_components"] == "all") & (model_results["thr_features"])]

            if len(preferred_results) > 0:
                # Use mean accuracy across folds for the preferred configuration
                best_config = preferred_results.groupby(["n_components", "thr_features"])["accuracy"].mean().reset_index()
                best_config = best_config.loc[best_config["accuracy"].idxmax()]
            else:
                # Fall back to best overall configuration
                best_config = model_results.groupby(["n_components", "thr_features"])["accuracy"].mean().reset_index()
                best_config = best_config.loc[best_config["accuracy"].idxmax()]

            best_configs[model_name] = {
                "n_components": best_config["n_components"],
                "thr_features": best_config["thr_features"],
                "accuracy": best_config["accuracy"],
            }

        # Handle case where path_to_data has no directory separator
        path_parts = path_to_data.split("/")[:-1]
        if path_parts:
            base_dir = "/".join(path_parts)
        else:
            base_dir = "."  # Current directory if no path is specified
        models_dir = base_dir + "/models"
        os.makedirs(models_dir, exist_ok=True)

        # Train final models using best configurations on ALL data
        for model_name, config in best_configs.items():
            logger.info(
                f"Training final {model_name} model with n_components={config['n_components']}, thr_features={config['thr_features']}"
            )

            # Prepare data using the same logic from CV
            X_final = X.copy()
            if not config["thr_features"] and config["n_components"] == "all" and len(thr_indices) > 0:
                X_final = X_final[:, thr_indices]

            # Create a fresh model instance
            if model_name in models:
                if model_name == "dummy":
                    final_model = DummyClassifier()
                elif model_name == "decisiontree":
                    final_model = tree.DecisionTreeClassifier()
                elif model_name == "logistic":
                    final_model = LogisticRegression()
                elif model_name == "rf":
                    # Use tuned RandomForest for final model with adaptive CV folds
                    cv_final, n_splits_final, min_class_count_final, final_cv_strategy, final_cv_reason = get_adaptive_cv(
                        y, max_splits=5, groups=group_ids
                    )
                    n_iter_final = runtime_config["n_iter"]
                    logger.info(
                        f"Using {n_splits_final} CV folds and {n_iter_final} iterations for final model training with {X_final.shape[0]} samples "
                        f"(strategy: {final_cv_strategy}, minimum class count: {min_class_count_final}). {final_cv_reason}"
                    )

                    final_model = RandomizedSearchCV(
                        estimator=RandomForestClassifier(random_state=42, n_jobs=1),
                        param_distributions=rf_param_dist,
                        n_iter=n_iter_final,
                        scoring=make_scorer(f1_score, average="weighted"),
                        cv=cv_final,
                        verbose=0,
                        random_state=42,
                        n_jobs=search_jobs,
                        pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
                    )
                elif model_name == "xgb":
                    final_model = XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=1,
                        objective="binary:logistic",
                        n_jobs=search_jobs,
                    )
                elif model_name == "gridsearch":
                    cv_final, n_splits_final, min_class_count_final, final_cv_strategy, final_cv_reason = get_adaptive_cv(
                        y, max_splits=5, groups=group_ids
                    )
                    logger.info(
                        f"Using {n_splits_final} CV folds for final KNN tuning with {X_final.shape[0]} samples "
                        f"(strategy: {final_cv_strategy}, minimum class count: {min_class_count_final}). "
                        f"{final_cv_reason}"
                    )
                    final_model = GridSearchCV(
                        KNeighborsClassifier(),
                        parameters,
                        cv=cv_final,
                        n_jobs=search_jobs,
                        pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
                    )
                elif model_name == "tpot" and TPOT_AVAILABLE:
                    final_model = TPOTClassifier(
                        generations=5,
                        population_size=20,
                        cv=5,
                        random_state=42,
                        verbosity=2,
                        n_jobs=PARALLELISM,
                        memory="auto",
                    )
                else:
                    logger.warning(f"Unknown model type {model_name}, skipping final model training")
                    continue

                # Apply SVD transformation if needed
                svd_transformer = None
                n_components_val = config["n_components"]
                if config["n_components"] != "all":
                    n_components_val = int(config["n_components"])
                    svd_transformer = TruncatedSVD(n_components=n_components_val, n_iter=15, random_state=42)
                    X_final = svd_transformer.fit_transform(X_final)
                else:
                    n_components_val = X_final.shape[1]

                # Skip autogluon for final model training as it's complex to handle
                if model_name == "autogluon":
                    logger.info(f"Skipping final model training for {model_name} (autogluon)")
                    continue

                try:
                    # Train final model on all data
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit_kwargs = (
                            {"groups": group_ids}
                            if group_ids is not None and isinstance(final_model, RandomizedSearchCV | GridSearchCV)
                            else {}
                        )
                        final_model.fit(X_final, y, **fit_kwargs)

                    # For RandomizedSearchCV models, log the best parameters and use best estimator
                    if hasattr(final_model, "best_params_"):
                        logger.info(f"Best parameters for {model_name}: {final_model.best_params_}")
                        logger.info(f"Best CV score for {model_name}: {final_model.best_score_}")
                        # Save the best estimator instead of the search object
                        model_to_save = final_model.best_estimator_
                    else:
                        model_to_save = final_model

                    # Save the final model
                    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
                    joblib.dump(model_to_save, model_path)
                    logger.info(f"Saved final model {model_name} to {model_path}")

                    # Save metadata
                    metadata = {
                        "feature_names": list(all_cols),
                        "target_mapping": catmap,
                        "n_components": config["n_components"],
                        "thr_features": config["thr_features"],
                        "thr_indices": thr_indices.tolist() if len(thr_indices) > 0 else [],
                        "filter_mode": filter_mode,
                        "model_name": model_name,
                        "cv_accuracy": config["accuracy"],
                        "svd_transformer": svd_transformer,
                    }

                    # Add hyperparameter tuning info for RandomizedSearchCV models
                    if hasattr(final_model, "best_params_"):
                        metadata["best_params"] = final_model.best_params_
                        metadata["best_cv_score"] = final_model.best_score_

                    metadata_path = os.path.join(models_dir, f"{model_name}_metadata.joblib")
                    joblib.dump(metadata, metadata_path)
                    logger.info(f"Saved metadata for final model {model_name} to {metadata_path}")

                except Exception as e:
                    logger.error(f"Failed to train final model {model_name}: {e}")
                    continue


def compute_ablation_scores(xs, y, replication_unit=None):
    """Evaluate top-feature subsets without leakage and with repeatable forests."""
    sample_names = xs.index.astype(str).tolist()
    X_init = xs.values
    y_init = pd.Categorical(y.values).codes
    group_ids = None
    if replication_unit:
        try:
            from input_validation import replication_group
        except ImportError:
            from .input_validation import replication_group

        group_ids = np.array([replication_group(name, replication_unit) for name in sample_names])

    skf, rfe_n_splits, rfe_min_class_count, rfe_cv_strategy, rfe_cv_reason = get_adaptive_cv(y_init, max_splits=3, groups=group_ids)
    logger.info(
        f"RFE evaluation uses {rfe_n_splits} folds (strategy: {rfe_cv_strategy}, "
        f"minimum class count: {rfe_min_class_count}). {rfe_cv_reason}"
    )
    split_iterator = skf.split(X_init, y_init, group_ids) if group_ids is not None else skf.split(X_init, y_init)
    splits = list(split_iterator)
    top_feature_counts = list(range(1, X_init.shape[1], 20))
    fold_accuracies = {top_n: [] for top_n in top_feature_counts}
    forest_jobs = configured_parallelism()

    for train_index, test_index in splits:
        # Rank only on the training fold. Ranking on all samples before CV leaks
        # information from the held-out fold into every ablation configuration.
        ranking_model = RandomForestClassifier(random_state=1234, n_jobs=forest_jobs)
        ranking_model.fit(X_init[train_index], y_init[train_index])
        sorted_indices = np.argsort(ranking_model.feature_importances_)[::-1]

        for top_n in top_feature_counts:
            selected = sorted_indices[:top_n]
            fresh_model = RandomForestClassifier(random_state=1234, n_jobs=forest_jobs)
            fresh_model.fit(X_init[train_index][:, selected], y_init[train_index])
            y_hat = fresh_model.predict(X_init[test_index][:, selected])
            fold_accuracies[top_n].append(accuracy_score(y_init[test_index], y_hat))

    out_df = []
    for top_n in top_feature_counts:
        mean_acc = np.mean(fold_accuracies[top_n])
        logger.info(f"Testing top features: {top_n} out of {X_init.shape[1]} (acc: {mean_acc})")
        out_df.append({"top_n": top_n, "accuracy": mean_acc})
    return pd.DataFrame(out_df)


def do_classification_rfe(xs, y, path_to_data, tagname="all", replication_unit=None):
    dfx_out = compute_ablation_scores(xs, y, replication_unit=replication_unit)
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    fout = base_dir + f"/ablation_ranking_{tagname}.tsv"
    dfx_out.to_csv(fout, sep="\t", index=False)
    print(dfx_out)


if __name__ == "__main__":
    # let's turn this beast into a command-line puppy.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Prepared datasets to analyze (paths to them)",
    )
    parser.add_argument(
        "--fout",
        default="",
        help="Ranking output file, e.g., 'ranking.out' (specify the full path)",
    )

    parser.add_argument(
        "--parallelism",
        default=-1,
        help="Number of concurrent threads for parallel approaches",
    )

    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Save trained models for inference (default: False)",
    )

    parser.add_argument(
        "--all_learners",
        action="store_true",
        help="Enable all machine learning algorithms (default: False, only RandomForest)",
    )
    parser.add_argument(
        "--replication_unit",
        "--replication-unit",
        choices=["position", "well", "plate", "date"],
        default=None,
        help="Keep all samples from the selected replication unit in one CV fold",
    )
    parser.add_argument(
        "--learner",
        choices=["rf", "dummy", "decisiontree", "logistic", "xgb", "gridsearch"],
        default="rf",
        help="Single learner to benchmark when --all_learners is not used",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
        help="Absolute correlation threshold used to group redundant top features",
    )

    try:
        arguments = parser.parse_args()
    except SystemExit:
        parser.print_help()
        exit(999)

    PARALLELISM = int(arguments.parallelism)
    if not 0 < arguments.correlation_threshold <= 1:
        parser.error("--correlation-threshold must be between 0 and 1")
    save_models = arguments.save_models
    all_learners = arguments.all_learners

    files = [base_file[: base_file.rfind(".")] + appendix + ".tsv" for base_file in arguments.files for appendix in [""]]
    # for every file, compute feature ranking and accuracy
    simple_classif = True
    for file in files:
        logger.info(f"Processing {file}")
        data = load_data(file)
        dates = []

        for date in data.index.tolist():
            dates.append(date.split("--")[0])

        data = data.copy()
        data["date"] = dates
        xs, y, _ = compute_rankings(data, file, skip=False, target_col="date")
        data = data.drop("date", axis=1)

        xs, y, _ = compute_rankings(data, file, skip=False, target_col="label")

        assert "date" not in xs.columns

        do_classification_simple(
            xs,
            y,
            file,
            save_models=save_models,
            all_learners=all_learners,
            replication_unit=arguments.replication_unit,
            learner=arguments.learner,
        )

        xs_cols = [x for x in xs.columns.tolist() if "counts" not in x]
        xs_no_counts = xs[xs_cols]

        do_classification_simple(
            xs_no_counts,
            y,
            file,
            "no_counts_features",
            save_models=save_models,
            all_learners=all_learners,
            replication_unit=arguments.replication_unit,
            learner=arguments.learner,
        )
        do_classification_rfe(xs, y, file, replication_unit=arguments.replication_unit)

        output_dir = os.path.join(os.path.dirname(file), "visualizations")
        write_confusion_matrices(os.path.join(os.path.dirname(file), "classification_all.tsv"), output_dir)
        write_confusion_matrices(os.path.join(os.path.dirname(file), "classification_no_counts_features.tsv"), output_dir)
        rankings_file = os.path.join(os.path.dirname(file), "rankings_label.tsv")
        if os.path.isfile(rankings_file):
            write_feature_correlation(file, rankings_file, output_dir, threshold=arguments.correlation_threshold)
            write_feature_boxplots(file, rankings_file, output_dir)

# Ref run
# conda activate imagine; python feature_ranking_lite.py --files ../results_30_12_2023_2/data.tsv --fout ../benchmark
