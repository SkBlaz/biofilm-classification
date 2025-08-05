import os
import gc
import re
import numpy as np
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

import shutil
import joblib
import json

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter(action='ignore', category=DataConversionWarning)

try:
    from tpot import TPOTClassifier
    from autogluon.tabular import TabularPredictor
except:
    print("Skipping tpot and autogluon, uncomment to enable (takes a lot of time")


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
    fout = "/".join(path_to_data.split("/")[:-1]) + f"/rankings_{target_col}.tsv"
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


def save_best_models(outputs, X, ys, path_to_data, filter_mode="all"):
    """
    Save the best performing models for each algorithm type.
    """
    results_dir = "/".join(path_to_data.split("/")[:-1])
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Convert outputs to DataFrame for analysis
    dfx = pd.DataFrame(outputs)
    dfx.columns = ['tag', 'model', 'upsampling', 'n_components', 'fold', 'accuracy', 'test_set', 'thr_features']
    dfx['accuracy'] = dfx['accuracy'].astype(float)
    
    # Find best performing configuration for each model type
    best_models = dfx.groupby('model')['accuracy'].max().to_dict()
    
    logger.info(f"Training and saving best models to {models_dir}")
    
    # Re-train best models on full dataset
    all_cols = X.columns
    thr_indices = []
    for enx, x in enumerate(all_cols):
        if "Threshold" in x:
            thr_indices.append(enx)
    thr_indices = np.array(thr_indices)
    
    X_vals = X.values
    y_vals = pd.Categorical(ys.values).codes
    catmap = dict(zip(y_vals, ys.values))
    
    models = {
        'dummy': DummyClassifier(),
        'decisiontree': tree.DecisionTreeClassifier(),
        'logistic': LogisticRegression(),
        'rf': RandomForestClassifier(),
        'xgb': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1, objective='binary:logistic'),
        'gridsearch': GridSearchCV(KNeighborsClassifier(), parameters, n_jobs=PARALLELISM),
    }
    
    saved_models = {}
    
    for model_name, model in models.items():
        if model_name in best_models:
            # Get best configuration for this model
            best_config = dfx[dfx['model'] == model_name].loc[dfx['accuracy'].idxmax()]
            n_components = best_config['n_components']
            thr_features = best_config['thr_features']
            
            logger.info(f"Training best {model_name} with {n_components} components, thr_features={thr_features}")
            
            # Prepare data with same preprocessing as best configuration
            X_train = X_vals.copy()
            
            if not thr_features and n_components == "all":
                X_train = X_train[:, thr_indices]
            
            svd = None
            if n_components != "all":
                n_components = int(n_components)
                svd = TruncatedSVD(n_components=n_components, n_iter=15, random_state=42)
                X_train = svd.fit_transform(X_train)
            
            # Train model on full dataset
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_vals)
                
                # Save model and metadata
                model_info = {
                    'model_name': model_name,
                    'model_type': str(type(model).__name__),
                    'n_components': n_components,
                    'thr_features': thr_features,
                    'best_accuracy': float(best_config['accuracy']),
                    'feature_names': list(X.columns),
                    'class_mapping': catmap,
                    'filter_mode': filter_mode,
                    'threshold_indices': thr_indices.tolist() if thr_indices.size > 0 else []
                }
                
                # Save model
                model_path = os.path.join(models_dir, f"{model_name}_{filter_mode}.joblib")
                joblib.dump(model, model_path)
                
                # Save SVD transformer if used
                if svd is not None:
                    svd_path = os.path.join(models_dir, f"{model_name}_{filter_mode}_svd.joblib")
                    joblib.dump(svd, svd_path)
                    model_info['svd_path'] = f"{model_name}_{filter_mode}_svd.joblib"
                
                # Save metadata
                metadata_path = os.path.join(models_dir, f"{model_name}_{filter_mode}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                saved_models[model_name] = {
                    'model_path': model_path,
                    'metadata_path': metadata_path,
                    'accuracy': float(best_config['accuracy'])
                }
                
                logger.info(f"Saved {model_name} model (accuracy: {best_config['accuracy']:.4f}) to {model_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save model {model_name}: {e}")
    
    # Save overall model summary
    summary_path = os.path.join(models_dir, f"models_summary_{filter_mode}.json")
    with open(summary_path, 'w') as f:
        json.dump(saved_models, f, indent=2)
    
    logger.info(f"Saved {len(saved_models)} models to {models_dir}")
    return saved_models


def do_classification_simple(X, ys, path_to_data, filter_mode="all"):

    all_cols = X.columns
    thr_indices = []
    for enx, x in enumerate(all_cols):
        if "Threshold" in x:
            thr_indices.append(enx)
    thr_indices = np.array(thr_indices)
    X = X.values
    y = pd.Categorical(ys.values).codes
    catmap = dict(zip(y, ys.values))
    upsampling = 1

    models = {
        'dummy': DummyClassifier(),
        'decisiontree': tree.DecisionTreeClassifier(),
        'logistic': LogisticRegression(),
        'rf': RandomForestClassifier(),
        'xgb': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1, objective='binary:logistic'),
        'gridsearch': GridSearchCV(KNeighborsClassifier(), parameters, n_jobs=PARALLELISM),
        #'tpot': TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, n_jobs=PARALLELISM, memory='auto'),
        'autogluon': TabularPredictor(label="label"),
    }
    
    outputs = []
    partial_dir = "/".join(path_to_data.split("/")[:-1]) + f"/partial/"
    if not os.path.isdir(partial_dir):
        os.mkdir(partial_dir)
        
    for repetition in range(3):
        for n_components in [16, 32, 64, 128, 256, 512, "all"]:
            desc_components = n_components
            for thr_features in [True, False]:
                skf = StratifiedKFold(n_splits=3)
                
                for i, (train_index, test_index) in enumerate(skf.split(X, y)):

                    

                    x_train = X[train_index]
                    x_test = X[test_index]
                    y_train = y[train_index]
                    y_test = y[test_index]

                    if not thr_features and desc_components == "all":
                        x_train = x_train[:, thr_indices]
                        x_test = x_test[:, thr_indices]                

                    if desc_components != "all" or "TabPFN" in str(model):
                        svd = TruncatedSVD(n_components=n_components,
                                            n_iter=15,
                                            random_state=42).fit(x_train)
                        x_train = svd.transform(x_train)
                        x_test = svd.transform(x_test)
                    else:
                        n_components = x_train.shape[1]

                    for model_name, model in models.items():
                        partial_path = partial_dir + f"{filter_mode}_partial_{repetition}_n{desc_components}_thr{thr_features}_{model_name}_fold{i}.tsv"
                        
                        if os.path.isfile(partial_path):
                            with open(partial_path) as f:
                                output = f.read().strip().split('\t')
                            outputs.append(output)
                            logger.info(f"Loaded existing partial evaluation from {partial_path}, skipping model evaluation")
                            continue

                        logger.info(f"Running {n_components} {' '.join(str(model).split())}, fold: {i}, filter mode: {filter_mode}")

                        if "TabularPredictor" in str(model):
                            #if desc_components == "all":
                            #    continue
                            x_train_ag = pd.DataFrame(x_train)
                            x_test_ag = pd.DataFrame(x_test)
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
                        else:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    model.fit(x_train, y_train)
                                    y_hat = model.predict(x_test)
                            except Exception as e:
                                logger.warning(f"Repetition {repetition} with {desc_components} components (THR: {thr_features}) model {model_name} fold {i} raised {e} (filter mode {filter_mode})")
                                y_hat = np.ones(len(x_test))

                            acc = accuracy_score(y_test, y_hat)
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
    fout = "/".join(path_to_data.split("/")[:-1]) + f"/classification_{filter_mode}.tsv"
    dfx = dfx.sort_values(by=['accuracy'])
    dfx.to_csv(fout, sep="\t")

    logger.info(f"Wrote classification outputs to {fout}")
    
    # Save best models for inference
    save_best_models(outputs, X, ys, path_to_data, filter_mode)


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
    fout = "/".join(path_to_data.split("/")[:-1]) + f"/ablation_ranking_{tagname}.tsv"
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


    try:
        arguments = parser.parse_args()
    except:
        parser.print_help()
        exit(999)

    PARALLELISM = int(arguments.parallelism)

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
        
        do_classification_simple(xs, y, file)
        
        xs_cols = [x for x in xs.columns.tolist() if "counts" not in x]
        xs_no_counts = xs[xs_cols]

        do_classification_simple(xs_no_counts, y, file, "no_counts_features")
        do_classification_rfe(xs, y, file)

# Ref run
# conda activate imagine; python feature_ranking_lite.py --files ../results_30_12_2023_2/data.tsv --fout ../benchmark
