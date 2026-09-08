#!/usr/bin/env python
"""Top-k feature-selection table for the SHAP-IMV examples.

For every dataset and every estimator family, four attribution baselines rank the
features; the top ``k`` of each ranking are then used to retrain the same
estimator, and accuracy/precision are averaged over the folds of a repeated
stratified cross-validation.

    ranking method  ->  top-k features  ->  refit  ->  accuracy / precision

The rankings are global, so each local method is aggregated into one importance
score per feature:

* ``SHAP-IMV`` mean exact Shapley value with IMV as the coalition value, read
  from the notebook artefacts written by ``src/empirical/shap_imv/*.ipynb``
  (recomputed with :class:`imvpy.BinaryIMV` when the artefact is absent).
* ``SHAP``     mean absolute SHAP value over the explained rows.
* ``LIME``     mean absolute local surrogate weight over the explained rows.
* ``Anchors``  how often a feature appears in the anchor of an explained row,
               tie-broken by its mean position inside the anchor.

Datasets, features, preprocessing, estimators and seeds are copied from the
notebooks so the rankings line up with the published SHAP-IMV figures.

Usage:
    python src/empirical/shap_imv/top_k_features.py                 # all examples
    python src/empirical/shap_imv/top_k_features.py --example shap_imv_titanic
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Paths and protocol
# --------------------------------------------------------------------------- #

PROJECT_ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents)
    if (path / "requirements.txt").is_file()
    and (path / "src" / "figure_utils.py").is_file()
)
CACHE = Path(os.environ.get("IMV_CACHE_HOME", Path.home() / ".cache" / "imv"))
DATA_HOME = Path(os.environ.get("IMV_DATA_CACHE", CACHE / "datasets")) / "openml"
NOTEBOOK_ARTIFACTS = Path(os.environ.get("IMV_ARTIFACT_CACHE", CACHE / "notebook_artifacts"))
RANKING_CACHE = NOTEBOOK_ARTIFACTS / "shap_imv_top_k" / "rankings"
OUTPUT_DIR = PROJECT_ROOT / "output" / "examples" / "shap"

# The notebooks' protocol: ten seeds, so every number is a mean over ten
# independent fold partitions rather than one lucky split.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
TOP_K = 5
EVAL_SPLITS = 10          # "averaged over 10 folds"
RANK_SPLITS = 5           # unused for scoring; kept for the SHAP-IMV fallback

# Explanation budget. SHAP is exact per row and cheap; LIME and especially
# Anchors resample the model, so they get a smaller, seeded row sample.
SHAP_ROWS = 2000
SHAP_BACKGROUND = 200
LIME_ROWS = 200
LIME_SAMPLES = 2000
ANCHOR_ROWS = 100
ANCHOR_THRESHOLD = 0.95

MODEL_PARAMETERS = {
    "logistic_regression": {"max_iter": 5000},
    "xgboost": {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2,
                "verbosity": 0, "tree_method": "hist", "n_jobs": 1},
    "lightgbm": {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2,
                 "verbose": -1, "n_jobs": 1},
}
MODEL_NAMES = tuple(MODEL_PARAMETERS)
METHODS = ("shap_imv", "shap", "lime", "anchors")

FORCE_RECOMPUTE = os.environ.get("IMV_FORCE_RECOMPUTE", "0").strip().lower() in {
    "1", "true", "yes",
}


def build_model(model_name: str, seed: int):
    """One estimator instance, parameterised exactly as in the notebooks."""
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=seed, **MODEL_PARAMETERS[model_name])
    if model_name == "xgboost":
        return XGBClassifier(random_state=seed, **MODEL_PARAMETERS[model_name])
    if model_name == "lightgbm":
        return LGBMClassifier(random_state=seed, **MODEL_PARAMETERS[model_name])
    raise ValueError(f"unknown model: {model_name}")


# --------------------------------------------------------------------------- #
# Datasets — preprocessing transcribed from the notebooks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Example:
    """One notebook: its display name, dataset slug, features and loader."""
    name: str            # e.g. shap_imv_titanic; also the output file stem
    dataset: str         # DATASET in the notebook; names the cached artefacts
    title: str
    features: tuple
    loader: Callable[[], pd.DataFrame]


def load_titanic() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = fetch_openml("titanic", version=1, as_frame=True,
                             data_home=DATA_HOME).frame.copy()
    frame["target"] = frame["survived"].astype(int)
    frame["Sex"] = (frame["sex"].astype(str) == "female").astype(int)
    frame["Class"] = pd.to_numeric(frame["pclass"], errors="coerce")
    frame["Age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["Age"] = frame["Age"].fillna(frame["Age"].median())
    frame["Fare"] = pd.to_numeric(frame["fare"], errors="coerce")
    frame["Fare"] = frame["Fare"].fillna(frame["Fare"].median())
    frame["Alone"] = ((pd.to_numeric(frame["sibsp"], errors="coerce").fillna(0)
                       + pd.to_numeric(frame["parch"], errors="coerce").fillna(0)) == 0).astype(int)
    frame["Embarked"] = frame["embarked"].astype(str).map({"C": 0, "Q": 1, "S": 2}).fillna(2)
    frame["AgeClass"] = frame["Age"] * frame["Class"]
    title = frame["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title.replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
                           "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    frame["Title"] = title.map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
                                "Rare": 5}).fillna(0).astype(int)
    return frame


def load_breast_cancer() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    raw = fetch_ucirepo(id=17)
    frame = raw.data.features.copy()
    frame["target"] = (raw.data.targets.iloc[:, 0].astype(str).str.strip() == "M").astype(int)
    return frame


def load_adult_income() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    raw = fetch_ucirepo(id=2)
    frame = raw.data.features.copy()
    target = raw.data.targets.iloc[:, 0].astype(str).str.strip().str.rstrip(".")
    frame["target"] = (target == ">50K").astype(int)
    frame["sex_female"] = (frame["sex"].astype(str).str.strip() == "Female").astype(int)
    frame["married"] = (frame["marital-status"].astype(str).str.strip()
                        .str.startswith("Married").astype(int))
    return frame


EXAMPLES = {
    example.name: example
    for example in (
        Example("shap_imv_titanic", "titanic", "Titanic",
                ("Sex", "Title", "Class", "AgeClass", "Fare", "Embarked"),
                load_titanic),
        Example("shap_imv_breast_cancer", "breast_cancer", "Breast Cancer",
                ("radius1", "texture1", "smoothness1", "compactness1",
                 "symmetry1", "fractal_dimension1"),
                load_breast_cancer),
        Example("shap_imv_adult_income", "adult_income", "Adult Income",
                ("age", "education-num", "hours-per-week", "capital-gain",
                 "sex_female", "married"),
                load_adult_income),
    )
}


def prepare(example: Example):
    """Notebook preprocessing: select, drop non-numeric rows, standardise."""
    frame = example.loader()
    features = list(example.features)
    missing = [column for column in features if column not in frame.columns]
    if missing:                       # column naming varies between UCI mirrors
        raise KeyError(f"{example.name}: missing columns {missing}")
    data = (frame[features + ["target"]]
            .apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True))
    # Standardisation is fit once on the full dataset, exactly as the notebooks
    # do: with no row subsampling the seed selects the fold partition only.
    data[features] = StandardScaler().fit_transform(data[features])
    return data[features].to_numpy(dtype=float), data["target"].to_numpy(dtype=int), features


# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #

def shap_imv_importance(example: Example, model_name: str, features, X, y) -> np.ndarray:
    """Mean SHAP-IMV per feature, from the notebook artefact when available."""
    summary = (NOTEBOOK_ARTIFACTS / example.name / "results"
               / f"{example.dataset}_shap_imv_summary.csv")
    if summary.is_file():
        table = pd.read_csv(summary)
        table = table[table["model"] == model_name].set_index("feature")["mean"]
        if set(table.index) == set(features):
            return table.reindex(features).to_numpy(dtype=float)
        print(f"  {summary.name}: feature set differs from the notebook; recomputing")
    return _shap_imv_from_scratch(model_name, features, X, y)


def _shap_imv_from_scratch(model_name, features, X, y) -> np.ndarray:
    """Fallback: exact SHAP-IMV over the power set, averaged over the seeds."""
    from imv import BinaryIMV

    frame = pd.DataFrame(X, columns=features)
    frame["target"] = y
    totals = np.zeros(len(features))
    for seed in SEEDS:
        evaluator = BinaryIMV(
            frame, "target", list(features), lambda: build_model(model_name, seed),
            split_method="stratified_kfold", n_splits=RANK_SPLITS,
            random_seed=seed, n_jobs=1, verbose=False,
        )
        evaluator.run_evaluation()
        totals += [evaluator.calculate_imvshapley_value(name) for name in features]
    return totals / len(SEEDS)


def _fit_full(model_name: str, seed: int, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_model(model_name, seed).fit(X, y)


def _sample_rows(X, n_rows, seed):
    rng = np.random.default_rng(seed)
    if len(X) <= n_rows:
        return X
    return X[rng.choice(len(X), size=n_rows, replace=False)]


def shap_importance(model, model_name, X, seed) -> np.ndarray:
    """Mean |SHAP value| per feature over a seeded row sample.

    XGBoost is read through its own ``pred_contribs`` interface rather than
    ``shap.TreeExplainer``: both run the same exact TreeSHAP, but shap 0.49 (the
    last release that supports numpy < 2, which alibi still pins) cannot parse
    the list-valued ``base_score`` written by xgboost 3.
    """
    import shap

    rows = _sample_rows(X, SHAP_ROWS, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name == "logistic_regression":
            background = _sample_rows(X, SHAP_BACKGROUND, seed)
            values = shap.LinearExplainer(model, background).shap_values(rows)
        elif model_name == "xgboost":
            import xgboost

            contributions = model.get_booster().predict(
                xgboost.DMatrix(rows), pred_contribs=True)
            values = np.asarray(contributions)[:, :-1]   # drop the bias column
        else:
            values = shap.TreeExplainer(model).shap_values(rows)
    values = np.asarray(values[1] if isinstance(values, list) else values)
    if values.ndim == 3:               # (rows, features, classes) -> positive class
        values = values[..., -1]
    return np.abs(values).mean(axis=0)


def lime_importance(model, features, X, seed) -> np.ndarray:
    """Mean |local surrogate weight| per feature over a seeded row sample."""
    from lime.lime_tabular import LimeTabularExplainer

    # Binary columns are declared categorical: quartile bins over a two-valued
    # standardised column are degenerate and would hide the feature from LIME.
    categorical = [index for index in range(X.shape[1])
                   if len(np.unique(X[:, index])) <= 2]
    explainer = LimeTabularExplainer(
        X, feature_names=list(features), class_names=["0", "1"],
        categorical_features=categorical, discretize_continuous=True,
        mode="classification", random_state=seed,
    )
    totals = np.zeros(X.shape[1])
    rows = _sample_rows(X, LIME_ROWS, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in rows:
            explanation = explainer.explain_instance(
                row, model.predict_proba, num_features=X.shape[1],
                num_samples=LIME_SAMPLES, labels=(1,),
            )
            for index, weight in explanation.as_map()[1]:
                totals[index] += abs(weight)
    return totals / len(rows)


def anchor_importance(model, features, X, seed) -> np.ndarray:
    """How often each feature anchors a prediction, tie-broken by its position.

    The count is the primary signal; anchors are grown greedily, so the mean
    position inside the anchor breaks ties between equally frequent features.
    It is scaled below 1 so it can never outrank a difference in count.
    """
    import logging

    from alibi.explainers import AnchorTabular

    # Rows whose anchor misses the precision threshold are reported one line at a
    # time; across 10 seeds x 100 rows that buries everything else.
    logging.getLogger("alibi").setLevel(logging.ERROR)

    counts = np.zeros(X.shape[1])
    positions = np.zeros(X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = AnchorTabular(
            lambda rows: model.predict(np.asarray(rows)),
            feature_names=list(features), seed=seed,
        )
        explainer.fit(X, disc_perc=(25, 50, 75))
        rows = _sample_rows(X, ANCHOR_ROWS, seed)
        for row in rows:
            try:
                explanation = explainer.explain(row, threshold=ANCHOR_THRESHOLD)
            except Exception as error:            # a row the search cannot anchor
                print(f"  anchors: skipping a row ({type(error).__name__}: {error})")
                continue
            anchor = explanation.data["raw"]["feature"]
            for position, index in enumerate(anchor):
                counts[index] += 1
                positions[index] += 1.0 / (position + 1)
    denominator = max(len(rows), 1)
    return counts / denominator + 0.5 * positions / (denominator * max(counts.max(), 1))


def ranking_path(example: Example, model_name: str, method: str, seed: int) -> Path:
    return (RANKING_CACHE / example.name
            / f"{method}_{model_name}_seed_{seed}.json")


def compute_ranking(example: Example, model_name: str, method: str, seed: int,
                    features, X, y) -> np.ndarray:
    """Importance per feature for one (model, method, seed), with a disk cache."""
    path = ranking_path(example, model_name, method, seed)
    if not FORCE_RECOMPUTE and path.is_file():
        try:
            cached = json.loads(path.read_text())
            if cached["features"] == list(features):
                return np.asarray(cached["importance"], dtype=float)
        except (ValueError, KeyError, OSError):
            pass                                   # fall through and recompute

    model = _fit_full(model_name, seed, X, y)
    if method == "shap":
        importance = shap_importance(model, model_name, X, seed)
    elif method == "lime":
        importance = lime_importance(model, features, X, seed)
    elif method == "anchors":
        importance = anchor_importance(model, features, X, seed)
    else:
        raise ValueError(f"unknown method: {method}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"features": list(features),
                                     "importance": [float(v) for v in importance]}))
    os.replace(temporary, path)
    return np.asarray(importance, dtype=float)


def select_top_k(features, importance, k=TOP_K):
    """Top-k feature names, most important first; ties break on feature order."""
    order = sorted(range(len(features)), key=lambda i: (-importance[i], i))
    return [features[i] for i in order[:k]], [features[i] for i in order[k:]]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model_name, X, y, columns):
    """Accuracy and precision per fold, over EVAL_SPLITS folds x every seed."""
    subset = X[:, columns]
    accuracy, precision = [], []
    for seed in SEEDS:
        folds = StratifiedKFold(n_splits=EVAL_SPLITS, shuffle=True, random_state=seed)
        seed_accuracy, seed_precision = [], []
        for train, test in folds.split(subset, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build_model(model_name, seed).fit(subset[train], y[train])
                predicted = model.predict(subset[test])
            seed_accuracy.append(accuracy_score(y[test], predicted))
            seed_precision.append(precision_score(y[test], predicted, zero_division=0))
        accuracy.append(seed_accuracy)
        precision.append(seed_precision)
    return np.asarray(accuracy), np.asarray(precision)


def run_example(example: Example, n_jobs: int) -> pd.DataFrame:
    started = perf_counter()
    print(f"\n=== {example.name} ===")
    X, y, features = prepare(example)
    print(f"rows {X.shape[0]}, features {features}, positive rate {y.mean():.4f}")

    # Rankings: SHAP-IMV comes from the notebook artefact, the three local
    # baselines are recomputed per seed and averaged, so every ranking is a
    # ten-seed mean.
    jobs = [(model_name, method, seed)
            for model_name in MODEL_NAMES
            for method in METHODS if method != "shap_imv"
            for seed in SEEDS]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        computed = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(compute_ranking)(example, model_name, method, seed, features, X, y)
            for model_name, method, seed in jobs)
    per_seed = {}
    for (model_name, method, seed), importance in zip(jobs, computed):
        per_seed.setdefault((model_name, method), []).append(importance)

    rows, evaluated = [], {}
    for model_name in MODEL_NAMES:
        for method in METHODS:
            if method == "shap_imv":
                importance = shap_imv_importance(example, model_name, features, X, y)
            else:
                importance = np.mean(per_seed[(model_name, method)], axis=0)
            selected, dropped = select_top_k(features, importance)
            columns = [features.index(name) for name in selected]
            # Six features choose five leaves at most six distinct subsets, and
            # the methods often agree, so each subset is evaluated once.
            key = (model_name, tuple(sorted(columns)))
            if key not in evaluated:
                evaluated[key] = evaluate(model_name, X, y, columns)
            accuracy, precision = evaluated[key]
            rows.append({
                "example": example.name,
                "dataset": example.title,
                "model": model_name,
                "method": method,
                "k": TOP_K,
                "selected_features": ";".join(selected),
                "dropped_features": ";".join(dropped),
                "importance": ";".join(f"{importance[features.index(n)]:.6g}"
                                       for n in selected),
                "accuracy_mean": accuracy.mean(),
                "accuracy_std_across_seeds": accuracy.mean(axis=1).std(ddof=1),
                "precision_mean": precision.mean(),
                "precision_std_across_seeds": precision.mean(axis=1).std(ddof=1),
                "n_seeds": len(SEEDS),
                "n_splits": EVAL_SPLITS,
            })
            print(f"{model_name:>20} {method:>9}  "
                  f"acc {100 * accuracy.mean():6.2f}  "
                  f"prec {100 * precision.mean():6.2f}  [{', '.join(selected)}]")

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{example.name}_top_{TOP_K}_features.csv"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"wrote {destination} in {perf_counter() - started:.1f}s")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", action="append", choices=sorted(EXAMPLES),
                        help="run one example; repeatable (default: all)")
    parser.add_argument("--n-jobs", type=int,
                        default=int(os.environ.get("IMV_N_JOBS", 0)) or None,
                        help="workers for the ranking sweep (default: cores - 1)")
    arguments = parser.parse_args()

    n_jobs = arguments.n_jobs or max(1, (os.cpu_count() or 2) - 1)
    names = arguments.example or sorted(EXAMPLES)
    for name in names:
        run_example(EXAMPLES[name], n_jobs)


if __name__ == "__main__":
    main()
