import os
import gc
import re
import numpy as np
import argparse
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

import shutil

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter(action='ignore', category=DataConversionWarning)

# Import HalvingRandomSearchCV (available in sklearn >= 1.0)
try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV
    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False
    HalvingRandomSearchCV = None

# Import Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = False
except:
    TPOT_AVAILABLE = False
    TPOTClassifier = None

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = False
except:
    AUTOGLUON_AVAILABLE = False
    TabularPredictor = None

if not TPOT_AVAILABLE or not AUTOGLUON_AVAILABLE:
    print("Skipping tpot and autogluon, uncomment to enable (takes a lot of time)")


parameters = {
    'n_neighbors': list(range(3, 50, 2)),  # Test odd values for better balancing in ties
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming', 'jaccard'],
    'weights': ['uniform', 'distance'],   # Test both weighting strategies
    'p': [1, 2, 3],                       # Minkowski distance with Manhattan (1), Euclidean (2), etc.
}


import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
np.random.seed(123)

PARALLELISM = -1


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
            self.scores[i] = sum(partial_scores[c]
                                 for c in feature_groups[c_original])

    def compute_scores(self, xs: pd.DataFrame,
                       y: pd.Series) -> dict[str, float]:
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
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_features=max_features,
                                            random_state=1234)

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
        max_val = data[c].replace(np.inf, np.nan).max()
        if isinstance(max_val, str):
            data[c] = data[c].fillna("missing")
        else:
            data[c] = data[c].replace(np.inf, max_val + 3.14).fillna(-666)

    data = data.copy()

    # Using haccs to get poss aggregated ..
    data['noPos'] = [
        "--".join([j for j in x.split("--") if "pos" not in j])
        for x in data.index.values.tolist()
    ]
    # data.index = range(len(data))
    # cols = [x for x in data.columns if x != 'label']
    # fil = {x: 'mean' for x in cols}
    # fil['label'] = 'first'
    # fil['noPos'] = 'first'
    # data = data.groupby('noPos').agg(fil)
    data.to_csv(path_to_data + "intermediary_aggregated.tsv", sep="\t")
    return data.drop(['noPos'], axis=1)
#    return data


def compute_rankings(data: str,
                     path_to_data: str,
                     target_col="label",
                     skip: bool = False,
                     fout: str = ""):
    """
    Computes feature rankings for the data found at ``path_to_data``,
    where the target column is ``target_col``.

    Saves the rankings into files (csv and pdf) to the output directory
    (see ``get_out_dir``).
    """
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    fout = base_dir + f"/rankings_{target_col}.tsv"
    logger.info(fout)

    if fout:
        output_file = fout
    else:
        output_file = os.path.join(out_dir, f"rankings_{file_appendix}.tsv")

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
    scores_data = pd.DataFrame(index=x_columns,
                               columns=[model.name for model in models],
                               data=0.0)
    for model in models:
        logger.info(f"Computing rankings for {model} - shape: {x_data.shape}")
        model.fit(x_data, y_data)
        for feature, score in zip(*model.names_and_scores):
            scores_data.at[feature, model.name] = score
    scores_data = scores_data.sort_values(rf_model.name, ascending=False)
    out_dir = get_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    file_appendix = os.path.basename(path_to_data)
    file_appendix = file_appendix[:file_appendix.rfind(".")]
    show_rankings(scores_data,
                  os.path.join(out_dir, f"rankings_{file_appendix}.pdf"))

    logger.info(f"Saving the ranking results to {output_file}")
    scores_data.reset_index().rename(columns={
        "index": "feature"
    }).to_csv(output_file, sep="\t", index=False)
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


def rf_search_space(for_halving=False):
    """
    Returns a comprehensive hyperparameter search space for RandomForest.
    
    This search space includes:
    - n_estimators: 200-1600 trees (not included if for_halving=True)
    - max_features: sqrt, log2, None, and float values 0.1-0.9
    - max_depth: None (unlimited) or 6-40
    - min_samples_split: 2-20
    - min_samples_leaf: 1-20
    - bootstrap: True/False
    - class_weight: None, balanced, balanced_subsample
    - max_samples: None or 0.5-0.95 (only valid when bootstrap=True)
    
    Args:
        for_halving: If True, excludes n_estimators and max_samples (for HalvingRandomSearchCV)
    """
    space = {
        "max_features": ["sqrt", "log2", None] + list(np.arange(0.1, 1.0, 0.1)),
        "max_depth": [None] + list(range(6, 41, 2)),
        "min_samples_split": list(range(2, 21, 1)),
        "min_samples_leaf": list(range(1, 21, 1)),
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    
    if not for_halving:
        space["n_estimators"] = list(range(200, 1601, 100))
        # Note: max_samples is handled specially - only used when bootstrap=True
        # We'll filter this during sampling
    
    return space


def make_groups_from_index(index, mode="date"):
    """
    Extract groups from sample names to prevent data leakage in CV.
    
    Args:
        index: pandas Index or list of sample names
        mode: "date" to group by date+sev, "sev" to group by sev only
        
    Returns:
        numpy array of group labels (integers)
    """
    if mode == "date":
        # Extract date (first 8 chars) and sev (st--XXX)
        groups = []
        for sample_name in index:
            try:
                date = sample_name[:8]
                sev_match = re.search(r"st--([^-_]+)", sample_name)
                if sev_match:
                    sev = sev_match.group(1)
                    groups.append(f"{date}_{sev}")
                else:
                    # Fallback: use just the date
                    groups.append(date)
            except:
                # Fallback: use the full sample name
                groups.append(sample_name)
        
        # Convert to integer labels
        unique_groups = sorted(set(groups))
        group_to_int = {g: i for i, g in enumerate(unique_groups)}
        return np.array([group_to_int[g] for g in groups])
    
    elif mode == "sev":
        # Extract only sev
        groups = []
        for sample_name in index:
            try:
                sev_match = re.search(r"st--([^-_]+)", sample_name)
                if sev_match:
                    sev = sev_match.group(1)
                    groups.append(sev)
                else:
                    groups.append(sample_name)
            except:
                groups.append(sample_name)
        
        # Convert to integer labels
        unique_groups = sorted(set(groups))
        group_to_int = {g: i for i, g in enumerate(unique_groups)}
        return np.array([group_to_int[g] for g in groups])
    
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'date' or 'sev'")


def tune_rf_optuna(X, y, groups=None, time_budget_sec=600):
    """
    Tune RandomForest using Optuna Bayesian optimization.
    
    Features:
    - Bayesian optimization with TPE sampler
    - Median pruner for early stopping of unpromising trials
    - Time-budgeted optimization
    - Group-aware CV if groups are provided
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target labels (numpy array or pandas Series)
        groups: Optional group labels for GroupKFold CV
        time_budget_sec: Time budget in seconds (default: 600)
        
    Returns:
        (best_estimator, study): Best RandomForest model and Optuna study object
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, cannot use tune_rf_optuna")
        return None, None
    
    logger.info(f"Starting Optuna RF tuning with time budget {time_budget_sec}s")
    
    # Ensure X is numpy array
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    def objective(trial):
        # Sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 15, 20, 25, 30, 35, 40]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Add max_features (categorical + continuous)
        max_features_choice = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "none", "float"])
        if max_features_choice == "sqrt":
            params["max_features"] = "sqrt"
        elif max_features_choice == "log2":
            params["max_features"] = "log2"
        elif max_features_choice == "none":
            params["max_features"] = None
        else:  # float
            params["max_features"] = trial.suggest_float("max_features_value", 0.1, 0.9)
        
        # Add max_samples if bootstrap is True
        if params["bootstrap"]:
            max_samples_choice = trial.suggest_categorical("max_samples_type", ["none", "float"])
            if max_samples_choice == "float":
                params["max_samples"] = trial.suggest_float("max_samples_value", 0.5, 0.95)
            else:
                params["max_samples"] = None
        
        # Create model
        rf = RandomForestClassifier(**params)
        
        # Setup CV
        if groups is not None:
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                cv = GroupKFold(n_splits=min(5, n_groups))
                cv_splits = list(cv.split(X, y, groups))
            else:
                # Fall back to StratifiedKFold if not enough groups
                logger.warning(f"Only {n_groups} group(s) found, falling back to StratifiedKFold")
                n_splits = min(5, max(2, len(X) // 2))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_splits = list(cv.split(X, y))
        else:
            n_splits = min(5, max(2, len(X) // 2))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_splits = list(cv.split(X, y))
        
        # Cross-validation with pruning
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            score = f1_score(y_val, y_pred, average="weighted")
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(np.mean(scores), fold_idx)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    # Create study with pruner and sampler
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    # Optimize with time budget
    study.optimize(objective, timeout=time_budget_sec, n_jobs=1, show_progress_bar=False)
    
    logger.info(f"Optuna completed {len(study.trials)} trials")
    logger.info(f"Best trial score: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Build best estimator
    best_params = study.best_params.copy()
    
    # Reconstruct max_features
    if best_params["max_features_type"] == "sqrt":
        max_features = "sqrt"
    elif best_params["max_features_type"] == "log2":
        max_features = "log2"
    elif best_params["max_features_type"] == "none":
        max_features = None
    else:
        max_features = best_params.get("max_features_value", 0.5)
    
    # Reconstruct max_samples
    max_samples = None
    if best_params.get("bootstrap", False):
        if best_params.get("max_samples_type") == "float":
            max_samples = best_params.get("max_samples_value", None)
    
    # Remove auxiliary keys
    for key in ["max_features_type", "max_features_value", "max_samples_type", "max_samples_value"]:
        best_params.pop(key, None)
    
    best_params["max_features"] = max_features
    if "bootstrap" in best_params and best_params["bootstrap"]:
        best_params["max_samples"] = max_samples
    
    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X, y)
    
    return best_rf, study


def tune_rf_halving(X, y, groups=None):
    """
    Tune RandomForest using HalvingRandomSearchCV with successive halving.
    
    Uses n_estimators as the resource parameter - starts with small forests
    and progressively increases size for promising candidates.
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target labels (numpy array or pandas Series)
        groups: Optional group labels for GroupKFold CV
        
    Returns:
        (best_estimator, search): Best RandomForest model and search object
    """
    if not HALVING_AVAILABLE:
        logger.warning("HalvingRandomSearchCV not available, cannot use tune_rf_halving")
        return None, None
    
    logger.info("Starting HalvingRandomSearchCV RF tuning")
    
    # Ensure proper format
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Create search space (exclude n_estimators as it's used as resource)
    search_space = rf_search_space(for_halving=True)
    
    # Setup CV
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            cv = GroupKFold(n_splits=min(5, n_groups))
        else:
            # Fall back to StratifiedKFold if not enough groups
            logger.warning(f"Only {n_groups} group(s) found, falling back to StratifiedKFold")
            n_splits = min(5, max(2, len(X) // 2))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = min(5, max(2, len(X) // 2))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Create base estimator
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Create halving search
    search = HalvingRandomSearchCV(
        estimator=rf_base,
        param_distributions=search_space,
        resource="n_estimators",
        max_resources=1600,
        min_resources=200,
        scoring=make_scorer(f1_score, average="weighted"),
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X, y, groups=groups if groups is not None else None)
    
    logger.info(f"HalvingRandomSearchCV completed")
    logger.info(f"Best score: {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")
    
    return search.best_estimator_, search


from sklearn.model_selection import ParameterSampler

def tune_rf_randomized(X, y, groups=None, n_iter=50):
    """
    Tune RandomForest using RandomizedSearchCV with rich search space.
    
    Fallback method when Optuna or Halving are not available.
    Uses manual cross-validation to handle conditional parameters properly.
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target labels (numpy array or pandas Series)
        groups: Optional group labels for GroupKFold CV
        n_iter: Number of parameter settings sampled (default: 50)
        
    Returns:
        (best_estimator, search): Best RandomForest model and search object
    """
    logger.info(f"Starting RandomizedSearchCV RF tuning with {n_iter} iterations")
    
    # Ensure proper format
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Get base search space
    search_space = rf_search_space(for_halving=False)
    
    # Setup CV
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            cv = GroupKFold(n_splits=min(5, n_groups))
            cv_splits = list(cv.split(X, y, groups))
        else:
            # Fall back to StratifiedKFold if not enough groups
            logger.warning(f"Only {n_groups} group(s) found, falling back to StratifiedKFold")
            n_splits = min(5, max(2, len(X) // 2))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_splits = list(cv.split(X, y))
    else:
        n_splits = min(5, max(2, len(X) // 2))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X, y))
    
    # Sample parameters manually, handling bootstrap/max_samples constraint
    best_score = -np.inf
    best_params = None
    best_estimator = None
    
    np.random.seed(42)
    for i in range(n_iter):
        # Sample parameters
        params = {}
        for key, values in search_space.items():
            if isinstance(values, list):
                params[key] = np.random.choice(values)
        
        # Handle max_samples conditionally
        if params.get('bootstrap', True):
            # When bootstrap=True, randomly choose max_samples
            if np.random.random() < 0.5:
                params['max_samples'] = None
            else:
                params['max_samples'] = np.random.uniform(0.5, 0.95)
        else:
            # When bootstrap=False, max_samples must be None
            params['max_samples'] = None
        
        # Create model with these parameters
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        # Cross-validate
        scores = []
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            rf_clone = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            rf_clone.fit(X_train, y_train)
            y_pred = rf_clone.predict(X_val)
            score = f1_score(y_val, y_pred, average="weighted")
            scores.append(score)
        
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_estimator = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    # Fit best estimator on full data
    best_estimator.fit(X, y)
    
    logger.info(f"RandomizedSearchCV completed")
    logger.info(f"Best score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Create a mock search object for compatibility
    class MockSearch:
        def __init__(self, best_estimator, best_params, best_score):
            self.best_estimator_ = best_estimator
            self.best_params_ = best_params
            self.best_score_ = best_score
    
    search = MockSearch(best_estimator, best_params, best_score)
    
    return best_estimator, search


def build_best_rf(X, y, sample_index=None, time_budget_sec=300):
    """
    Build the best RandomForest model using the best available tuning method.
    
    Priority order:
    1. Optuna (if available) - most sophisticated
    2. HalvingRandomSearchCV (if available) - efficient resource allocation
    3. RandomizedSearchCV - reliable fallback
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target labels (numpy array or pandas Series)
        sample_index: Optional pandas Index for extracting groups
        time_budget_sec: Time budget for Optuna (default: 300s)
        
    Returns:
        Best RandomForest estimator (not the search object)
    """
    # Extract groups if sample_index is provided
    groups = None
    if sample_index is not None:
        try:
            groups = make_groups_from_index(sample_index, mode="date")
            logger.info(f"Created {len(np.unique(groups))} groups for group-aware CV")
        except Exception as e:
            logger.warning(f"Could not extract groups from index: {e}")
            groups = None
    
    # Try methods in order of preference
    if OPTUNA_AVAILABLE:
        logger.info("Using Optuna for hyperparameter optimization")
        best_rf, study = tune_rf_optuna(X, y, groups=groups, time_budget_sec=time_budget_sec)
        if best_rf is not None:
            return best_rf
    
    if HALVING_AVAILABLE:
        logger.info("Using HalvingRandomSearchCV for hyperparameter optimization")
        best_rf, search = tune_rf_halving(X, y, groups=groups)
        if best_rf is not None:
            return best_rf
    
    # Fallback to RandomizedSearchCV
    logger.info("Using RandomizedSearchCV for hyperparameter optimization")
    best_rf, search = tune_rf_randomized(X, y, groups=groups, n_iter=50)
    return best_rf


def do_classification_simple(X, ys, path_to_data, filter_mode="all", save_models=False, all_learners=False):

    all_cols = X.columns
    sample_index = ys.index  # Preserve the index for group extraction
    thr_indices = []
    for enx, x in enumerate(all_cols):
        if "Threshold" in x:
            thr_indices.append(enx)
    thr_indices = np.array(thr_indices)
    X = X.values
    y = pd.Categorical(ys.values).codes
    catmap = dict(zip(y, ys.values))
    upsampling = 1

    # Build best RandomForest using advanced HPO (Optuna -> Halving -> RandomizedSearch)
    # Use adaptive time budget based on dataset size
    n_samples = X.shape[0]
    if n_samples < 20:
        # Small dataset (e.g., CI tests): use minimal time budget
        time_budget_sec = 30
    elif n_samples < 100:
        # Small dataset: moderate time budget
        time_budget_sec = 60
    elif n_samples < 500:
        # Medium dataset: standard time budget
        time_budget_sec = 180
    else:
        # Large dataset: extended time budget
        time_budget_sec = 300
    
    logger.info(f"Building best RandomForest model using advanced hyperparameter optimization (time budget: {time_budget_sec}s for {n_samples} samples)")
    tuned_rf = build_best_rf(X, y, sample_index=sample_index, time_budget_sec=time_budget_sec)

    if all_learners:
        models = {
            'dummy': DummyClassifier(),
            'decisiontree': tree.DecisionTreeClassifier(),
            'logistic': LogisticRegression(),
            'rf': tuned_rf,
            'xgb': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1, objective='binary:logistic'),
            'gridsearch': GridSearchCV(KNeighborsClassifier(), parameters, n_jobs=PARALLELISM),
            #'tpot': TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, n_jobs=PARALLELISM, memory='auto'),
        }
        
        # Add TPOT only if available
        if TPOT_AVAILABLE:
            models['tpot'] = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, n_jobs=PARALLELISM, memory='auto')
    else:
        # Default behavior: only RandomForest (fast)
        models = {
            'rf': tuned_rf,
        }
    
    # Add autogluon model only if available
    if AUTOGLUON_AVAILABLE:
        models['autogluon'] = TabularPredictor(label="label")
    
    outputs = []
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    partial_dir = base_dir + f"/partial/"
    if not os.path.isdir(partial_dir):
        os.mkdir(partial_dir)
    
    # Store CV results to find best hyperparameters for final model training
    cv_results = []
        
    for repetition in range(3):
        for n_components in [16, 32, 64, 128, 256, 512, "all"]:
            desc_components = n_components
            
            # Skip if n_components exceeds available features
            if n_components != "all" and n_components > X.shape[1]:
                logger.info(f"Skipping n_components={n_components} as it exceeds available features ({X.shape[1]})")
                continue
                
            for thr_features in [True, False]:
                skf = StratifiedKFold(n_splits=3)
                
                for i, (train_index, test_index) in enumerate(skf.split(X, y)):

                    

                    x_train = X[train_index]
                    x_test = X[test_index]
                    y_train = y[train_index]
                    y_test = y[test_index]

                    if not thr_features and desc_components == "all" and len(thr_indices) > 0:
                        x_train = x_train[:, thr_indices]
                        x_test = x_test[:, thr_indices]                

                    for model_name, model in models.items():
                        partial_path = partial_dir + f"{filter_mode}_partial_{repetition}_n{desc_components}_thr{thr_features}_{model_name}_fold{i}.tsv"
                        
                        if os.path.isfile(partial_path):
                            with open(partial_path) as f:
                                output = f.read().strip().split('\t')
                            outputs.append(output)
                            logger.info(f"Loaded existing partial evaluation from {partial_path}, skipping model evaluation")
                            continue

                        logger.info(f"Running {n_components} {' '.join(str(model).split())}, fold: {i}, filter mode: {filter_mode}")

                        # Prepare data with SVD if needed
                        x_train_model = x_train.copy()
                        x_test_model = x_test.copy()
                        svd_transformer = None
                        
                        if desc_components != "all" or "TabPFN" in str(model):
                            svd_transformer = TruncatedSVD(n_components=n_components,
                                                n_iter=15,
                                                random_state=42).fit(x_train_model)
                            x_train_model = svd_transformer.transform(x_train_model)
                            x_test_model = svd_transformer.transform(x_test_model)
                        else:
                            n_components = x_train_model.shape[1]

                        if "TabularPredictor" in str(model) and AUTOGLUON_AVAILABLE:
                            #if desc_components == "all":
                            #    continue
                            x_train_ag = pd.DataFrame(x_train_model)
                            x_test_ag = pd.DataFrame(x_test_model)
                            y_train_ag = pd.DataFrame(y_train)
                            y_test_ag = pd.DataFrame(y_test)
                            y_train_ag.columns = ['label']
                            y_test_ag.columns = ['label']

                            train_data = pd.concat([x_train_ag, y_train_ag], axis=1)
                            test_data = pd.concat([x_test_ag, y_test_ag], axis=1)

                            model = TabularPredictor(label="label")
                            predictor = model.fit(train_data, ag_args_fit={'num_cpus': PARALLELISM}) if PARALLELISM != -1 else model.fit(train_data)
                            #predictor = model.fit(train_data)
                            y_hat = predictor.predict(test_data)
                            acc = accuracy_score(y_test_ag, y_hat)
                            mname = str(model)
                            del model
                            model = mname
                            gc.collect()
                        elif "TabularPredictor" in str(model) and not AUTOGLUON_AVAILABLE:
                            # Skip autogluon if not available
                            logger.warning(f"Skipping {model_name} - autogluon not available")
                            continue
                        else:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    model.fit(x_train_model, y_train)
                                    y_hat = model.predict(x_test_model)
                            except Exception as e:
                                logger.warning(f"Repetition {repetition} with {desc_components} components (THR: {thr_features}) model {model_name} fold {i} raised {e} (filter mode {filter_mode})")
                                y_hat = np.ones(len(x_test_model))

                            acc = accuracy_score(y_test, y_hat)
                        
                        # Store CV results for later use in final model training
                        if save_models:
                            cv_results.append({
                                'model_name': model_name,
                                'repetition': repetition,
                                'n_components': desc_components,
                                'thr_features': thr_features,
                                'fold': i,
                                'accuracy': acc,
                                'svd_transformer': svd_transformer
                            })
                        
                        test_map = ",".join([catmap[x] for x in y_test])
                        output = [
                            "RESULT",
                            str(model).replace("\n", ""), upsampling, n_components, i,
                            acc, test_map, thr_features
                        ]
                        with open(partial_path, 'w') as f:
                            f.write('\t'.join([str(x) for x in output]))
                            logger.info(f"Stored partial evaluation to {partial_path}")
                        outputs.append([str(x) for x in output])
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)
        logger.info("All model evaluation complete, deleted partial results")
    dfx = pd.DataFrame(outputs)
    dfx.columns = ['tag', 'model', 'upsampling', 'n_components', 'fold', 'accuracy', 'test_set', 'thr_features']
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    fout = base_dir + f"/classification_{filter_mode}.tsv"
    dfx = dfx.sort_values(by=['accuracy'])
    dfx.to_csv(fout, sep="\t")

    logger.info(f"Wrote classification outputs to {fout}")
    
    # Train final models on all data using best hyperparameters from cross-validation
    if save_models and cv_results:
        logger.info("Training final models on all data using best hyperparameters from cross-validation")
        
        # Convert cv_results to DataFrame for easier analysis
        cv_df = pd.DataFrame(cv_results)
        
        # Find best hyperparameters for each model (prioritize 'all' features and thr_features=True)
        best_configs = {}
        for model_name in cv_df['model_name'].unique():
            model_results = cv_df[cv_df['model_name'] == model_name]
            
            # First try with 'all' features and thr_features=True
            preferred_results = model_results[
                (model_results['n_components'] == 'all') & 
                (model_results['thr_features'] == True)
            ]
            
            if len(preferred_results) > 0:
                # Use mean accuracy across folds for the preferred configuration
                best_config = preferred_results.groupby(['n_components', 'thr_features'])['accuracy'].mean().reset_index()
                best_config = best_config.loc[best_config['accuracy'].idxmax()]
            else:
                # Fall back to best overall configuration
                best_config = model_results.groupby(['n_components', 'thr_features'])['accuracy'].mean().reset_index()
                best_config = best_config.loc[best_config['accuracy'].idxmax()]
            
            best_configs[model_name] = {
                'n_components': best_config['n_components'],
                'thr_features': best_config['thr_features'],
                'accuracy': best_config['accuracy']
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
            logger.info(f"Training final {model_name} model with n_components={config['n_components']}, thr_features={config['thr_features']}")
            
            # Prepare data using the same logic from CV
            X_final = X.copy()
            if not config['thr_features'] and config['n_components'] == "all" and len(thr_indices) > 0:
                X_final = X_final[:, thr_indices]
            
            # Create a fresh model instance
            model_already_fitted = False
            if model_name in models:
                if model_name == 'dummy':
                    final_model = DummyClassifier()
                elif model_name == 'decisiontree':
                    final_model = tree.DecisionTreeClassifier()
                elif model_name == 'logistic':
                    final_model = LogisticRegression()
                elif model_name == 'rf':
                    # Use build_best_rf for final model training with advanced HPO
                    logger.info(f"Using build_best_rf for final RF model training")
                    # Create index from sample_index for group extraction
                    final_sample_index = sample_index if sample_index is not None else None
                    # Adaptive time budget based on dataset size
                    n_samples_final = X_final.shape[0]
                    if n_samples_final < 20:
                        time_budget_final = 30
                    elif n_samples_final < 100:
                        time_budget_final = 60
                    elif n_samples_final < 500:
                        time_budget_final = 180
                    else:
                        time_budget_final = 300
                    # build_best_rf returns an already fitted model
                    final_model = build_best_rf(X_final, y, sample_index=final_sample_index, time_budget_sec=time_budget_final)
                    # Skip the fit step for this model as it's already fitted
                    model_already_fitted = True
                elif model_name == 'xgb':
                    final_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1, objective='binary:logistic')
                elif model_name == 'gridsearch':
                    final_model = GridSearchCV(KNeighborsClassifier(), parameters, n_jobs=PARALLELISM)
                elif model_name == 'tpot' and TPOT_AVAILABLE:
                    final_model = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, n_jobs=PARALLELISM, memory='auto')
                else:
                    logger.warning(f"Unknown model type {model_name}, skipping final model training")
                    continue
                
                # Apply SVD transformation if needed
                svd_transformer = None
                n_components_val = config['n_components']
                if config['n_components'] != "all":
                    n_components_val = int(config['n_components'])
                    svd_transformer = TruncatedSVD(n_components=n_components_val, n_iter=15, random_state=42)
                    X_final = svd_transformer.fit_transform(X_final)
                else:
                    n_components_val = X_final.shape[1]
                
                # Skip autogluon for final model training as it's complex to handle
                if model_name == 'autogluon':
                    logger.info(f"Skipping final model training for {model_name} (autogluon)")
                    continue
                
                try:
                    # Train final model on all data (unless already fitted)
                    if not model_already_fitted:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            final_model.fit(X_final, y)
                    
                    # For RandomizedSearchCV models, log the best parameters and use best estimator
                    if hasattr(final_model, 'best_params_'):
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
                        'feature_names': list(all_cols),
                        'target_mapping': catmap,
                        'n_components': config['n_components'],
                        'thr_features': config['thr_features'],
                        'thr_indices': thr_indices.tolist() if len(thr_indices) > 0 else [],
                        'filter_mode': filter_mode,
                        'model_name': model_name,
                        'cv_accuracy': config['accuracy'],
                        'svd_transformer': svd_transformer
                    }
                    
                    # Add hyperparameter tuning info for RandomizedSearchCV models
                    if hasattr(final_model, 'best_params_'):
                        metadata['best_params'] = final_model.best_params_
                        metadata['best_cv_score'] = final_model.best_score_
                    
                    metadata_path = os.path.join(models_dir, f"{model_name}_metadata.joblib")
                    joblib.dump(metadata, metadata_path)
                    logger.info(f"Saved metadata for final model {model_name} to {metadata_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to train final model {model_name}: {e}")
                    continue


def do_classification_rfe(xs, y, path_to_data, tagname="all"):

    # Do feature ranking on everything (yeah, this is just an ablation)
    # Do learning with top n
    model = RandomForestClassifier()
    model.fit(xs, y)
    importances = np.array(model.feature_importances_)
    sorted_indices = np.argsort(importances)[::-1]
        
    skf = StratifiedKFold(n_splits=3)
    X_init = xs.values
    y_init = pd.Categorical(y.values).codes
    out_df = []
    for j in range(1, len(sorted_indices), 20):

        X = X_init[:, sorted_indices[:j]]
        y = y_init

        accs = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):

            fresh_model = RandomForestClassifier()

            x_train = X[train_index]
            y_train = y[train_index]

            x_test = X[test_index]
            y_test = y[test_index]

            fresh_model.fit(x_train, y_train)
            y_hat = fresh_model.predict(x_test)        
            acc = accuracy_score(y_test, y_hat)
            accs.append(acc)
        mean_acc = np.mean(accs)
        logger.info(f"Testing top features: {j} out of {len(sorted_indices)} (acc: {mean_acc})")
        out_df.append({"top_n": j, "accuracy": mean_acc})
    dfx_out = pd.DataFrame(out_df)
    # Handle case where path_to_data has no directory separator
    path_parts = path_to_data.split("/")[:-1]
    if path_parts:
        base_dir = "/".join(path_parts)
    else:
        base_dir = "."  # Current directory if no path is specified
    fout = base_dir + f"/ablation_ranking_{tagname}.tsv"
    dfx_out.to_csv(fout, sep="\t")    
    print(dfx_out)

if __name__ == "__main__":
    # let's turn this beast into a command-line puppy.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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


    try:
        arguments = parser.parse_args()
    except:
        parser.print_help()
        exit(999)

    PARALLELISM = int(arguments.parallelism)
    save_models = arguments.save_models
    all_learners = arguments.all_learners

    files = [
        base_file[:base_file.rfind(".")] + appendix + ".tsv"
        for base_file in arguments.files for appendix in [""]
    ]
    # for every file, compute feature ranking and accuracy
    simple_classif = True
    for file in files:
        logger.info(f"Processing {file}")
        data = load_data(file)
        dates = []
        
        for date in data.index.tolist():
            dates.append(date.split("--")[0])
        
        data = data.copy()    
        data['date'] = dates
        xs, y, _ = compute_rankings(data, file, skip=False, target_col="date")
        data = data.drop('date', axis=1)
        
        xs, y, _ = compute_rankings(data, file, skip=False, target_col="label")

        assert "date" not in xs.columns
        
        do_classification_simple(xs, y, file, save_models=save_models, all_learners=all_learners)
        
        xs_cols = [x for x in xs.columns.tolist() if "counts" not in x]
        xs_no_counts = xs[xs_cols]

        do_classification_simple(xs_no_counts, y, file, "no_counts_features", save_models=save_models, all_learners=all_learners)
        do_classification_rfe(xs, y, file)

# Ref run
# conda activate imagine; python feature_ranking_lite.py --files ../results_30_12_2023_2/data.tsv --fout ../benchmark
