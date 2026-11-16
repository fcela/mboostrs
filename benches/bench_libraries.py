"""Held-out accuracy of mboostrs against EBM, XGBoost, LightGBM, CatBoost and an MLP.

The comparison rests on three choices, all of which change the answer.

*Every model sees the same splits.* Outer evaluation is `REPEATS` random 75/25
splits with seeds shared across libraries, so no model is scored on an easier
partition than another.

*"Fair" has two definitions and they disagree.* Under `configs`, every library
draws the same number of hyperparameter configurations and is scored by the same
inner cross-validation — equal search effort, which ignores that one library
fits in 0.07s and another in 18s. Under `seconds`, every library gets the same
tuning wall-clock and draws configurations until it runs out — equal compute,
which lets a fast library search further. Both are run; the tables report both.

*Two of the models cannot fit interactions.* mboostrs `GAMBoost` and a
depth-1 XGBoost are additive by construction: the fitted function is a sum of
one-dimensional terms. The depth-1 XGBoost is in the list as a control, so that
the gap between additive and non-additive models can be read separately from the
gap between libraries.

Preprocessing is identical for every model: median imputation, one-hot encoding
of categoricals, and standardisation applied only to the MLP, which requires it.

    python benches/bench_libraries.py --out docs/book/cases/bench_libraries.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, fetch_openml, load_breast_cancer, load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPEATS = 3
N_CONFIGS = 12
INNER_FOLDS = 3
TIME_BUDGET = 15.0
MAX_ROWS = 5000
SEED = 20240811


# --------------------------------------------------------------------- data
def _openml(data_id, target):
    frame = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
    return frame.drop(columns=[target]), frame[target]


def _encoder(X):
    """An unfitted encoder for a raw frame: median imputation of numeric
    columns, most-frequent imputation and one-hot coding (levels under 2%
    pooled) of the others. It is fitted inside every fit, so imputation
    values, vocabularies and rare-level thresholds come from that fit's
    training rows only."""
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "bool")]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False, min_frequency=0.02)), cat),
    ], remainder="drop", sparse_threshold=0.0)


def with_encoding(build, X):
    """`build` wrapped so each fitted model carries its own encoder when the
    data are a raw frame; unchanged for a numeric array."""
    if not hasattr(X, "columns"):
        return build
    return lambda params: make_pipeline(_encoder(X), build(params))


def take(X, rows):
    """Rows of an array or a frame."""
    return X.iloc[rows] if hasattr(X, "iloc") else X[rows]


def final_estimator(model):
    """The model inside an encoding pipeline, or the model itself."""
    return model[-1] if hasattr(model, "steps") else model


def datasets():
    out = {}

    d = load_diabetes()
    out["diabetes"] = ("regression", d.data.astype(float), d.target.astype(float))

    X, y = _openml(197, "usr")
    out["cpu_act"] = ("regression", X, y.to_numpy(float))

    c = fetch_california_housing()
    out["california"] = ("regression", c.data.astype(float), c.target.astype(float))

    b = load_breast_cancer()
    out["breast_cancer"] = ("classification", b.data.astype(float), b.target.astype(int))

    X, y = _openml(31, "class")
    out["credit_g"] = ("classification", X, (y.to_numpy() == "good").astype(int))

    X, y = _openml(151, "class")
    out["electricity"] = ("classification", X, (y.to_numpy() == "UP").astype(int))

    rng = np.random.default_rng(SEED)
    for name, (task, X, y) in list(out.items()):
        if len(y) > MAX_ROWS:
            keep = rng.choice(len(y), MAX_ROWS, replace=False)
            out[name] = (task, take(X, keep), y[keep])
    return out


# ------------------------------------------------------------------- models
def encoded_datasets():
    """`datasets()` with every raw frame encoded as numbers, the encoder fitted
    on all of its rows. For describing the columns and for single-split
    illustrations in the book; the benchmark itself fits the encoder inside
    every fit (`with_encoding`), from that fit's training rows only."""
    out = {}
    for name, (task, X, y) in datasets().items():
        if hasattr(X, "columns"):
            X = np.asarray(_encoder(X).fit_transform(X), dtype=float)
        out[name] = (task, X, y)
    return out


def model_space(task):
    """(builder, sampler) per library. The sampler draws one config from a grid
    of comparable generosity: a rate, a capacity knob and a regularisation knob."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from mboostrs.sklearn import GAMBoostClassifier, GAMBoostRegressor
    from xgboost import XGBClassifier, XGBRegressor

    reg = task == "regression"

    def draw(rng, grid):
        return {k: (v[rng.integers(len(v))] if isinstance(v, list) else v)
                for k, v in grid.items()}

    space = {}

    space["mboostrs GAMBoost"] = (
        lambda p: (GAMBoostRegressor if reg else GAMBoostClassifier)(**p),
        lambda rng: draw(rng, {"mstop": [100, 200, 400, 800, 1500],
                               "nu": [0.05, 0.1, 0.3],
                               "df": [3.0, 4.0, 6.0]}),
        "additive",
    )
    space["EBM"] = (
        lambda p: (ExplainableBoostingRegressor if reg else ExplainableBoostingClassifier)(
            random_state=0, n_jobs=-1, **p),
        lambda rng: draw(rng, {"max_rounds": [2000, 5000, 8000],
                               "learning_rate": [0.01, 0.02, 0.05],
                               "interactions": [0, 5, 10],
                               "max_leaves": [2, 3]}),
        "additive + pairs",
    )
    space["XGBoost"] = (
        lambda p: (XGBRegressor if reg else XGBClassifier)(
            verbosity=0, n_jobs=-1, tree_method="hist", **p),
        lambda rng: draw(rng, {"n_estimators": [200, 500, 1000],
                               "learning_rate": [0.02, 0.05, 0.1],
                               "max_depth": [3, 4, 6, 8],
                               "subsample": [0.7, 1.0],
                               "reg_lambda": [0.1, 1.0, 10.0]}),
        "interactions",
    )
    space["XGBoost depth 1"] = (
        lambda p: (XGBRegressor if reg else XGBClassifier)(
            verbosity=0, n_jobs=-1, tree_method="hist", max_depth=1, **p),
        lambda rng: draw(rng, {"n_estimators": [200, 500, 1000, 2000],
                               "learning_rate": [0.02, 0.05, 0.1],
                               "subsample": [0.7, 1.0],
                               "reg_lambda": [0.1, 1.0, 10.0]}),
        "additive",
    )
    space["LightGBM"] = (
        # `subsample` has no effect unless bagging runs: subsample_freq=1
        # resamples the rows at every iteration.
        lambda p: (LGBMRegressor if reg else LGBMClassifier)(verbose=-1, n_jobs=-1,
                                                             subsample_freq=1, **p),
        lambda rng: draw(rng, {"n_estimators": [200, 500, 1000],
                               "learning_rate": [0.02, 0.05, 0.1],
                               "num_leaves": [7, 15, 31, 63],
                               "subsample": [0.7, 1.0],
                               "reg_lambda": [0.1, 1.0, 10.0]}),
        "interactions",
    )
    space["CatBoost"] = (
        lambda p: (CatBoostRegressor if reg else CatBoostClassifier)(
            verbose=0, allow_writing_files=False, thread_count=-1, **p),
        lambda rng: draw(rng, {"iterations": [200, 500, 1000],
                               "learning_rate": [0.02, 0.05, 0.1],
                               "depth": [4, 6, 8],
                               "l2_leaf_reg": [1.0, 3.0, 10.0]}),
        "interactions",
    )
    space["MLP"] = (
        lambda p: make_pipeline(
            StandardScaler(),
            (MLPRegressor if reg else MLPClassifier)(
                max_iter=600, early_stopping=True, random_state=0, **p)),
        lambda rng: draw(rng, {"hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
                               "alpha": [1e-5, 1e-3, 1e-1],
                               "learning_rate_init": [1e-3, 3e-3, 1e-2]}),
        "interactions",
    )
    return space


def clipped(task, model, Xfit_y, Xv):
    """Predictions, held to the range of the responses the model was trained on.

    A prespecified policy, applied to every library and using the training
    responses only. No model family is bounded by that range: a boosted tree
    ensemble adds corrections rather than averaging observed responses, and a
    spline surface or a neural net can extrapolate far outside it, which a
    squared-error metric punishes out of proportion to the number of rows
    involved. The unclipped score is recorded next to the clipped one.
    """
    raw = np.asarray(model.predict(Xv)).ravel()
    return np.clip(raw, float(np.min(Xfit_y)), float(np.max(Xfit_y)))


def score(task, model, X, y, Xv, yv):
    model.fit(X, y)
    if task == "regression":
        return float(r2_score(yv, clipped(task, model, y, Xv)))
    return float(roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))


def inner_cv(task, build, params, X, y, folds):
    splitter = (KFold(folds, shuffle=True, random_state=0) if task == "regression"
                else StratifiedKFold(folds, shuffle=True, random_state=0))
    got = []
    for tr, va in splitter.split(X, y):
        got.append(score(task, build(params), take(X, tr), y[tr], take(X, va), y[va]))
    return float(np.mean(got))


def tune(task, build, sample, X, y, mode, rng):
    """Return (best params, configs tried, tuning seconds)."""
    best, best_score, tried = None, -np.inf, 0
    t0 = time.perf_counter()
    while True:
        if mode == "configs" and tried >= N_CONFIGS:
            break
        if mode == "seconds" and time.perf_counter() - t0 >= TIME_BUDGET and tried >= 1:
            break
        params = sample(rng)
        try:
            got = inner_cv(task, build, params, X, y, INNER_FOLDS)
        except Exception:
            tried += 1
            continue
        tried += 1
        if got > best_score:
            best, best_score = params, got
    return best, tried, time.perf_counter() - t0


def run(mode, out_path, only=None):
    results = []
    for name, (task, X, y) in datasets().items():
        if only and name not in only:
            continue
        space = model_space(task)
        for repeat in range(REPEATS):
            strat = y if task == "classification" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=SEED + repeat, stratify=strat)
            for library, (build, sample, family) in space.items():
                build = with_encoding(build, X)
                rng = np.random.default_rng(SEED + repeat)
                t0 = time.perf_counter()
                params, tried, tune_s = tune(task, build, sample, Xtr, ytr, mode, rng)
                if params is None:
                    continue
                final = build(params)
                fit0 = time.perf_counter()
                final.fit(Xtr, ytr)
                fit_s = time.perf_counter() - fit0
                held = (r2_score(yte, clipped(task, final, ytr, Xte))
                        if task == "regression"
                        else roc_auc_score(yte, final.predict_proba(Xte)[:, 1]))
                held_raw = (r2_score(yte, np.asarray(final.predict(Xte)).ravel())
                            if task == "regression" else held)
                results.append({
                    "dataset": name, "task": task, "n": int(len(y)),
                    "p": int(X.shape[1]), "repeat": repeat, "library": library,
                    "family": family, "mode": mode, "score": float(held),
                    "score_unclipped": float(held_raw),
                    "configs_tried": tried, "tune_seconds": round(tune_s, 2),
                    "refit_seconds": round(fit_s, 3),
                    "total_seconds": round(time.perf_counter() - t0, 2),
                })
                print(f"[{mode}] {name:14s} r{repeat} {library:18s} "
                      f"score {held:.4f}  configs {tried:3d}  tune {tune_s:7.1f}s",
                      flush=True)
                with open(out_path, "w") as fh:
                    json.dump({"protocol": {
                        "repeats": REPEATS, "n_configs": N_CONFIGS,
                        "inner_folds": INNER_FOLDS, "time_budget_s": TIME_BUDGET,
                        "max_rows": MAX_ROWS, "seed": SEED}, "results": results}, fh, indent=1)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/book/cases/bench_libraries.json")
    ap.add_argument("--mode", default="both", choices=["configs", "seconds", "both"])
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset; default is all six")
    args = ap.parse_args()
    modes = ["configs", "seconds"] if args.mode == "both" else [args.mode]
    only = args.datasets.split(",") if args.datasets else None
    everything = []
    for mode in modes:
        everything += run(mode, args.out + f".{mode}.partial", only=only)
    from _jobs import provenance

    protocol = {"repeats": REPEATS, "n_configs": N_CONFIGS,
                "inner_folds": INNER_FOLDS, "time_budget_s": TIME_BUDGET,
                "max_rows": MAX_ROWS, "seed": SEED}
    with open(args.out, "w") as fh:
        json.dump({"provenance": provenance(
                       protocol=protocol, repeats={"outer_splits": REPEATS},
                       uncertainty="per-split scores; summarised by mean and standard error"),
                   "protocol": protocol, "results": everything}, fh, indent=1)
    print(f"\nwrote {args.out}: {len(everything)} rows")
