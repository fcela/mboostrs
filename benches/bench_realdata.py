"""Real regression data with default specifications: OpenML-CTR23.

The OpenML-CTR23 suite (Fischer, Feurer and Bischl 2023) collects 35 curated
tabular regression tasks. This study uses the 24 with at most 35 features and
at most 30,000 rows, and caps the training rows of each split at 5,000 (a
random subsample of the training part) so every method fits every split.
Each data set is split five times at random, 80% training and 20% test (the
test part capped at 5,000 rows).

Every method starts from zero: the data frame and the name of the response,
with each library's default way of handling numeric and categorical columns.
mboostrs: `autospec(budget=200)` fitted with `rec.fit`, the same with
`interactions="adopt"`, the same followed by `refit="reml"`, the same with
`deselect=0.025` (a sparser specification), and `tune_mstop("y ~ .")`; mgcv with `s(x)` for every numeric column with more
than ten distinct values, a linear term otherwise, factors as factors, REML;
EBM defaults; LightGBM with early stopping on a 20% validation part over 15
and 63 leaves; XGBoost with the same early stopping over depths 4 and 8; CatBoost defaults; ridge regression on one-hot categoricals as
the linear baseline.

The score is the test R^2, 1 - MSE / Var(y_test), on the response's own
scale. When autospec chooses a distributional family (GaussianLSS when the
spread moves with a covariate), the prediction scored is the fitted location
parameter mu; each result row names the component it scored in
``scored_component`` ("response" or "mu").

    python benches/bench_realdata.py --out docs/book/cases/bench_realdata.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASETS = {
    44956: "abalone", 44957: "airfoil_self_noise", 44958: "auction_verification",
    44959: "concrete_compressive_strength", 44966: "solar_flare",
    44969: "naval_propulsion_plant", 44971: "white_wine", 44972: "red_wine",
    44973: "grid_stability", 44977: "california_housing", 44978: "cpu_activity",
    44980: "kin8nm", 44981: "pumadyn32nh", 44983: "miami_housing", 44984: "cps88wages",
    44987: "socmob", 44990: "brazilian_houses", 44993: "health_insurance",
    41021: "Moneyball", 44960: "energy_efficiency", 44962: "forest_fires",
    44967: "student_performance_por", 44970: "QSAR_fish_toxicity", 45402: "space_ga",
}
#: Held back from development: CTR23 regression data sets not among the 24
#: above, used to confirm policies chosen on those (``--suite confirmation``).
#: QSAR_TID_11 (1,025 columns) is left out: the per-column smooth methods
#: cannot fit it within the job timeout.
CONFIRMATION = {
    44961: "sgemm_gpu_kernel_performance", 44963: "physiochemical_protein",
    44964: "superconductivity", 44965: "geographical_origin_of_music",
    44968: "synchronous_machine", 44974: "video_transcoding", 44975: "wave_energy",
    44976: "sarcos", 44979: "diamonds", 44986: "nyc_taxi_green", 44989: "kings_county",
    44992: "fps_benchmark", 44994: "cars", 45012: "fifa",
}
ALL_DATASETS = {**DATASETS, **CONFIRMATION}
SPLITS = 5
MAX_TRAIN = 5000
MAX_TEST = 5000


def load(data_id):
    """(features, response, categorical column names) as a pandas frame."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    target = bunch.target_names[0] if bunch.target_names else frame.columns[-1]
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    frame.columns = [_clean(c) for c in frame.columns]
    target = _clean(target)
    categorical = []
    for column in frame.columns:
        if column == target:
            continue
        import pandas as pd

        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
            categorical.append(column)
        else:
            frame[column] = frame[column].astype(float)
    frame[target] = frame[target].astype(float)
    return frame, target, categorical


def _clean(name):
    import re

    name = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return name if name[0].isalpha() else f"v_{name}"


def split(frame, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    cut = int(0.8 * len(frame))
    train = order[:cut][:MAX_TRAIN]
    test = order[cut:][:MAX_TEST]
    return frame.iloc[train].reset_index(drop=True), frame.iloc[test].reset_index(drop=True)


# ------------------------------------------------------------------ methods
def fit_mboostrs(train, test, target, categorical, seed, variant):
    import polars as pl

    import mboostrs as mb

    tr, te = pl.from_pandas(train), pl.from_pandas(test.drop(columns=[target]))
    # One missing-data policy for every data set, fixed before any data is
    # seen: median imputation with an indicator, replayed on new rows.
    kwargs = {"na_action": "impute"}
    # A random split can put a label in the test part that the training part
    # lacks; predicting it at the average group is the analyst's next step
    # after the default refusal.
    if variant == "y~.":
        model = mb.tune_mstop(f"{target} ~ .", tr, seed=seed, **kwargs).model
        return _point_prediction(model, te)
    rec = mb.autospec(tr, target, budget=200, seed=seed, search="fast",
                      interactions="adopt" if variant in ("adopt", "tree") else "report",
                      residual_tree=variant == "tree",
                      **({"deselect": 0.025} if variant == "sparse" else {}))
    if variant == "refit":
        return _point_prediction(rec.fit(tr, refit="reml", **kwargs), te)
    return _point_prediction(rec.fit(tr, **kwargs), te)


def _point_prediction(model, newdata):
    """(test predictions, the component they are) for a fitted mboostrs model.

    An ordinary fit (`GAMBoostModel`, `GamRefit`) predicts one array, the
    response, recorded as ``"response"``. A distributional fit (`LSSModel`,
    which autospec chooses when its family probe finds the spread moving with
    a covariate) predicts a dict with one array per distribution parameter. The harness scores the location
    parameter ``mu`` on the response scale, recorded as ``"mu"``, so its
    squared error compares with the other methods' point predictions. For
    GaussianLSS, GammaLSS and BetaLSS ``mu`` is the predicted mean; for
    StudentTLSS it is the centre of the distribution, which is the mean when
    its degrees of freedom exceed 1.
    """
    pred = model.predict(newdata=newdata, type="response", unseen_levels="zero_effect")
    if isinstance(pred, dict):
        return np.asarray(pred["mu"], float), "mu"
    return np.asarray(pred, float), "response"


def _pandas_categories(train, test, categorical):
    import pandas as pd

    # The vocabulary is the training rows' own; a test label they lack is
    # missing, which LightGBM routes like any missing value.
    X_tr, X_te = train.copy(), test.copy()
    for column in categorical:
        levels = pd.Categorical(X_tr[column]).categories
        X_tr[column] = pd.Categorical(X_tr[column], categories=levels)
        X_te[column] = pd.Categorical(X_te[column], categories=levels)
    return X_tr, X_te


def fit_lgbm(train, test, target, categorical, seed):
    import lightgbm as lgb

    X_tr, X_te = _pandas_categories(train.drop(columns=[target]), test.drop(columns=[target]),
                                    categorical)
    y = train[target].to_numpy()
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.8 * len(y))
    best = None
    for leaves in (15, 63):
        params = {"objective": "regression", "num_leaves": leaves, "learning_rate": 0.05,
                  "num_threads": 1, "seed": seed, "verbose": -1}
        booster = lgb.train(params, lgb.Dataset(X_tr.iloc[idx[:cut]], y[idx[:cut]]), 5000,
                            valid_sets=[lgb.Dataset(X_tr.iloc[idx[cut:]], y[idx[cut:]])],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        score = next(iter(booster.best_score["valid_0"].values()))
        if best is None or score < best[0]:
            best = (score, params, booster.best_iteration)
    booster = lgb.train(best[1], lgb.Dataset(X_tr, y), best[2])
    return booster.predict(X_te)


def fit_xgboost(train, test, target, categorical, seed):
    """XGBoost with native categorical support: depth 4 or 8 and the number
    of rounds chosen by early stopping on the same 20% validation part of the
    training rows as LightGBM, then refitted on all training rows."""
    import xgboost as xgb

    X_tr, X_te = _pandas_categories(train.drop(columns=[target]), test.drop(columns=[target]),
                                    categorical)
    y = train[target].to_numpy()
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.8 * len(y))

    def matrix(X, label=None):
        return xgb.DMatrix(X, label=label, enable_categorical=True)

    best = None
    for depth in (4, 8):
        params = {"objective": "reg:squarederror", "max_depth": depth, "eta": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                  "nthread": 1, "seed": seed}
        booster = xgb.train(params, matrix(X_tr.iloc[idx[:cut]], y[idx[:cut]]), 5000,
                            evals=[(matrix(X_tr.iloc[idx[cut:]], y[idx[cut:]]), "valid")],
                            early_stopping_rounds=100, verbose_eval=False)
        if best is None or booster.best_score < best[0]:
            best = (booster.best_score, params, booster.best_iteration + 1)
    booster = xgb.train(best[1], matrix(X_tr, y), best[2])
    return booster.predict(matrix(X_te))


def fit_perpetual(train, test, target, categorical, seed):
    """PerpetualBooster (pure-Rust gradient boosting with one ``budget``
    parameter in place of hyperparameter tuning), squared loss, its default
    budget 0.5, native categorical columns, one thread."""
    from perpetual import PerpetualBooster

    X_tr, X_te = _pandas_categories(train.drop(columns=[target]), test.drop(columns=[target]),
                                    categorical)
    model = PerpetualBooster(objective="SquaredLoss", budget=0.5, num_threads=1, seed=seed,
                             categorical_features=categorical or None)
    model.fit(X_tr, train[target].to_numpy(float))
    return np.asarray(model.predict(X_te), float)


def fit_catboost(train, test, target, categorical, seed):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(random_seed=seed, verbose=0, thread_count=1,
                              allow_writing_files=False,
                              cat_features=categorical or None)
    model.fit(train.drop(columns=[target]), train[target])
    return model.predict(test.drop(columns=[target]))


def fit_ebm(train, test, target, categorical, seed):
    from interpret.glassbox import ExplainableBoostingRegressor

    model = ExplainableBoostingRegressor(n_jobs=1, random_state=seed)
    model.fit(train.drop(columns=[target]), train[target])
    return model.predict(test.drop(columns=[target]))


def fit_ridge(train, test, target, categorical, seed):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Indicator columns come from the training labels; a test label they
    # lack gets all-zero indicators.
    X_tr = pd.get_dummies(train.drop(columns=[target]), columns=categorical, dtype=float)
    X_te = pd.get_dummies(test.drop(columns=[target]), columns=categorical, dtype=float)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    model = make_pipeline(SimpleImputer(), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(X_tr, train[target])
    return model.predict(X_te)


def _linear_design(train, test, target, categorical):
    """Training-label indicators, training-median imputation and training
    standardization, shared by the ridge and lasso baselines."""
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_tr = pd.get_dummies(train.drop(columns=[target]), columns=categorical, dtype=float)
    X_te = pd.get_dummies(test.drop(columns=[target]), columns=categorical, dtype=float)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    imputer = SimpleImputer(strategy="median").fit(X_tr)
    scaler = StandardScaler().fit(imputer.transform(X_tr))
    return (scaler.transform(imputer.transform(X_tr)),
            scaler.transform(imputer.transform(X_te)), list(X_tr.columns))


def fit_lasso(train, test, target, categorical, seed, variant="lasso"):
    """Lasso-type baselines on the standardized design, penalty by 5-fold CV:

    * ``lasso``: LassoCV;
    * ``elastic net``: ElasticNetCV over l1_ratio 0.1, 0.5, 0.9, 1;
    * ``relaxed lasso``: the lasso's selected columns refitted by least
      squares (Meinshausen's relaxed lasso at gamma = 0);
    * ``adaptive lasso``: LassoCV on columns rescaled by the absolute ridge
      coefficients (Zou 2006), so large effects are penalized less;
    * ``spline lasso``: LassoCV on a cubic B-spline expansion of each numeric
      column (6 basis functions) plus the indicators, a nonlinear additive
      lasso.
    """
    from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV

    y = train[target].to_numpy(float)
    kw = {"cv": 5, "random_state": seed, "max_iter": 20000, "n_jobs": 1}
    if variant == "spline lasso":
        from sklearn.preprocessing import SplineTransformer

        numeric = [c for c in train.columns if c != target and c not in categorical]
        tr, te = train.copy(), test.copy()
        for c in numeric:
            median = tr[c].median()
            tr[c], te[c] = tr[c].fillna(median), te[c].fillna(median)
        blocks_tr, blocks_te = [], []
        for c in numeric:
            if tr[c].nunique() < 10:
                continue
            spline = SplineTransformer(n_knots=4, degree=3, extrapolation="linear")
            blocks_tr.append(spline.fit_transform(tr[[c]]))
            blocks_te.append(spline.transform(te[[c]]))
        X_tr, X_te, _ = _linear_design(tr, te, target, categorical)
        if blocks_tr:
            spl_tr, spl_te = np.hstack(blocks_tr), np.hstack(blocks_te)
            mu, sd = spl_tr.mean(0), spl_tr.std(0) + 1e-12
            X_tr = np.hstack([X_tr, (spl_tr - mu) / sd])
            X_te = np.hstack([X_te, (spl_te - mu) / sd])
        return LassoCV(**kw).fit(X_tr, y).predict(X_te)
    X_tr, X_te, _ = _linear_design(train, test, target, categorical)
    if variant == "lasso":
        return LassoCV(**kw).fit(X_tr, y).predict(X_te)
    if variant == "elastic net":
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], **kw).fit(X_tr, y).predict(X_te)
    if variant == "relaxed lasso":
        keep = np.flatnonzero(LassoCV(**kw).fit(X_tr, y).coef_)
        if keep.size == 0:
            return np.full(len(X_te), y.mean())
        return LinearRegression().fit(X_tr[:, keep], y).predict(X_te[:, keep])
    if variant == "adaptive lasso":
        scale = np.abs(RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_tr, y).coef_) + 1e-8
        return LassoCV(**kw).fit(X_tr * scale, y).predict(X_te * scale)
    raise ValueError(variant)


def fit_mlp(train, test, target, categorical, seed):
    """A tuned multilayer perceptron (PyTorch), the tabular baseline of
    Gorishniy et al. (2021): numeric columns median-imputed and mapped to
    normal scores by a quantile transform fitted on the training rows,
    training-label indicators for categoricals, a standardized target; three
    hidden layers of 256 ReLU units, AdamW (lr 1e-3, weight decay 1e-5),
    batch 256. Dropout 0 or 0.2 and the epoch count (up to 400, patience 30)
    are chosen on the same 20% validation part of the training rows as the
    tree libraries, then the network is refitted on all training rows for
    that many epochs."""
    import pandas as pd
    import torch
    from sklearn.preprocessing import QuantileTransformer

    torch.set_num_threads(1)
    numeric = [c for c in train.columns if c != target and c not in categorical]
    tr, te = train.copy(), test.copy()
    for c in numeric:
        median = tr[c].median()
        tr[c], te[c] = tr[c].fillna(median), te[c].fillna(median)
    blocks_tr, blocks_te = [], []
    if numeric:
        qt = QuantileTransformer(output_distribution="normal", random_state=seed,
                                 n_quantiles=min(1000, len(tr)))
        blocks_tr.append(qt.fit_transform(tr[numeric]))
        blocks_te.append(qt.transform(te[numeric]))
    if categorical:
        d_tr = pd.get_dummies(tr[categorical].astype(str), dtype=float)
        d_te = pd.get_dummies(te[categorical].astype(str), dtype=float).reindex(
            columns=d_tr.columns, fill_value=0.0)
        blocks_tr.append(d_tr.to_numpy())
        blocks_te.append(d_te.to_numpy())
    X_tr = np.hstack(blocks_tr).astype(np.float32)
    X_te = np.hstack(blocks_te).astype(np.float32)
    y = tr[target].to_numpy(float)
    y_mu, y_sd = y.mean(), y.std() + 1e-12
    z = ((y - y_mu) / y_sd).astype(np.float32)
    idx = np.random.default_rng(seed).permutation(len(z))
    cut = int(0.8 * len(z))

    def network(dropout):
        torch.manual_seed(seed)
        layers, width = [], X_tr.shape[1]
        for _ in range(3):
            layers += [torch.nn.Linear(width, 256), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            width = 256
        return torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))

    def train_net(dropout, rows, epochs, valid=None):
        net = network(dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xt, zt = torch.from_numpy(X_tr[rows]), torch.from_numpy(z[rows])
        gen = torch.Generator().manual_seed(seed)
        best, best_epoch, wait = np.inf, 0, 0
        for epoch in range(1, epochs + 1):
            net.train()
            for batch in torch.randperm(len(rows), generator=gen).split(256):
                opt.zero_grad()
                loss = torch.mean((net(Xt[batch]).squeeze(1) - zt[batch]) ** 2)
                loss.backward()
                opt.step()
            if valid is not None:
                net.eval()
                with torch.no_grad():
                    Xv, zv = torch.from_numpy(X_tr[valid]), torch.from_numpy(z[valid])
                    v = float(torch.mean((net(Xv).squeeze(1) - zv) ** 2))
                if v < best:
                    best, best_epoch, wait = v, epoch, 0
                else:
                    wait += 1
                    if wait >= 30:
                        break
        return net, best, best_epoch

    choice = min(((train_net(d, idx[:cut], 400, idx[cut:])[1:], d) for d in (0.0, 0.2)),
                 key=lambda item: item[0][0])
    (_, epochs), dropout = choice
    net, _, _ = train_net(dropout, np.arange(len(z)), max(epochs, 1))
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(X_te)).squeeze(1).numpy()
    return pred * y_sd + y_mu


def _nn_frame(train, test, target, categorical):
    """Numeric columns median-imputed from the training rows; categoricals as
    pandas categories with the training vocabulary."""
    X_tr, X_te = _pandas_categories(train.drop(columns=[target]), test.drop(columns=[target]),
                                    categorical)
    for c in X_tr.columns:
        if c not in categorical:
            median = X_tr[c].median()
            X_tr[c], X_te[c] = X_tr[c].fillna(median), X_te[c].fillna(median)
    return X_tr, X_te


def fit_realmlp(train, test, target, categorical, seed):
    """RealMLP with its tuned defaults (pytabkit ``RealMLP_TD_Regressor``,
    Holzmüller et al. 2024): its own preprocessing, categorical embeddings and
    validation-based stopping on the training rows."""
    import torch
    from pytabkit import RealMLP_TD_Regressor

    torch.set_num_threads(1)
    X_tr, X_te = _nn_frame(train, test, target, categorical)
    model = RealMLP_TD_Regressor(random_state=seed, n_threads=1, device="cpu", verbosity=0)
    model.fit(X_tr, train[target].to_numpy(float), cat_col_names=categorical or None)
    return np.asarray(model.predict(X_te), float).ravel()


def fit_tabpfn(train, test, target, categorical, seed):
    """TabPFN, a transformer pretrained on synthetic tabular tasks that
    predicts by in-context learning on the training rows, with its defaults
    on CPU. Its pretraining data are synthetic, so it has not seen these
    data sets."""
    import torch
    from tabpfn import TabPFNRegressor

    torch.set_num_threads(1)
    X_tr, X_te = _nn_frame(train, test, target, categorical)
    cat_idx = [X_tr.columns.get_loc(c) for c in categorical]
    for c in categorical:
        X_tr[c], X_te[c] = X_tr[c].cat.codes, X_te[c].cat.codes
    model = TabPFNRegressor(random_state=seed, device="cpu", n_jobs=1,
                            categorical_features_indices=cat_idx or None,
                            ignore_pretraining_limits=True)
    model.fit(X_tr.to_numpy(float), train[target].to_numpy(float))
    return np.asarray(model.predict(X_te.to_numpy(float)), float)


def fit_mgcv(train, test, target, categorical, seed):
    import pandas as pd

    # mgcv drops rows with a missing predictor and cannot predict them; the
    # analyst imputes the training medians first.
    train, test = train.copy(), test.copy()
    for column in train.columns:
        if column != target and column not in categorical and (
                train[column].isna().any() or test[column].isna().any()):
            median = train[column].median()
            train[column] = train[column].fillna(median)
            test[column] = test[column].fillna(median)
    terms = []
    for column in train.columns:
        if column == target:
            continue
        if column in categorical:
            terms.append(column)
        else:
            distinct = train[column].nunique()
            terms.append(f"s({column})" if distinct > 10 else column)
    # A test label the training rows lack predicts NA in mgcv; the analyst
    # assigns it the most frequent training level.
    factors = "".join(f"tr${c} <- factor(tr${c}); nd${c}[!(nd${c} %in% levels(tr${c}))] <- "
                      f"names(which.max(table(tr${c}))); nd${c} <- factor(nd${c}, "
                      f"levels = levels(tr${c}))\n" for c in categorical)
    script = ("suppressMessages(library(mgcv))\nargs <- commandArgs(TRUE)\n"
              "tr <- read.csv(args[1]); nd <- read.csv(args[2])\n" + factors +
              f"fit <- bam(as.formula('{target} ~ {' + '.join(terms)}'), data = tr, "
              "method = 'fREML', discrete = TRUE, nthreads = 1)\n"
              "p <- as.numeric(predict(fit, nd))\n"
              "write.csv(data.frame(p = p), args[3], row.names = FALSE)\n")
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / n for n in ("tr.csv", "nd.csv", "out.csv", "fit.R")]
        train.to_csv(paths[0], index=False)
        test.drop(columns=[target]).to_csv(paths[1], index=False)
        paths[3].write_text(script)
        res = subprocess.run(["Rscript", str(paths[3]), *map(str, paths[:3])],
                             capture_output=True, text=True,
                             env={**os.environ, "OMP_NUM_THREADS": "1"})
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-1500:])
        return pd.read_csv(paths[2])["p"].to_numpy(float)


METHODS = {
    "mboostrs autospec": lambda *a: fit_mboostrs(*a, "autospec"),
    "mboostrs autospec + pairs": lambda *a: fit_mboostrs(*a, "adopt"),
    "mboostrs autospec + REML refit": lambda *a: fit_mboostrs(*a, "refit"),
    "mboostrs autospec (deselect 2.5%)": lambda *a: fit_mboostrs(*a, "sparse"),
    "mboostrs autospec + pairs + residual tree": lambda *a: fit_mboostrs(*a, "tree"),
    "mboostrs y ~ .": lambda *a: fit_mboostrs(*a, "y~."),
    "mgcv": fit_mgcv,
    "EBM": fit_ebm,
    "LightGBM": fit_lgbm,
    "XGBoost": fit_xgboost,
    "Perpetual": fit_perpetual,
    "MLP (PyTorch)": fit_mlp,
    "RealMLP": fit_realmlp,
    "CatBoost": fit_catboost,
    "ridge": fit_ridge,
    "lasso": lambda *a: fit_lasso(*a, variant="lasso"),
    "elastic net": lambda *a: fit_lasso(*a, variant="elastic net"),
    "relaxed lasso": lambda *a: fit_lasso(*a, variant="relaxed lasso"),
    "adaptive lasso": lambda *a: fit_lasso(*a, variant="adaptive lasso"),
    "spline lasso": lambda *a: fit_lasso(*a, variant="spline lasso"),
}


# TabPFN downloads its weights only after a one-time licence acceptance
# (https://ux.priorlabs.ai); with TABPFN_TOKEN set it joins the comparison.
if os.environ.get("TABPFN_TOKEN"):
    METHODS["TabPFN"] = fit_tabpfn


def run_job(spec):
    data_id, method, seed = spec
    frame, target, categorical = load(int(data_id))
    train, test = split(frame, seed)
    start = time.perf_counter()
    try:
        out = METHODS[method](train, test, target, categorical, seed)
        # An mboostrs method returns (predictions, scored component); the
        # other libraries return point predictions of the response.
        pred, component = out if isinstance(out, tuple) else (out, "response")
        pred = np.asarray(pred, float)
    except Exception as exc:
        return {"data_id": data_id, "dataset": ALL_DATASETS[int(data_id)], "method": method,
                "seed": seed, "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
    y = test[target].to_numpy(float)
    mse = float(np.mean((pred - y) ** 2))
    return {"data_id": data_id, "dataset": ALL_DATASETS[int(data_id)], "method": method,
            "seed": seed, "seconds": time.perf_counter() - start,
            "n_train": len(train), "n_test": len(test), "p": frame.shape[1] - 1,
            "n_categorical": len(categorical), "scored_component": component, "mse": mse,
            "r2": float(1 - mse / np.var(y)), "finite": bool(np.all(np.isfinite(pred)))}


def main():
    from _jobs import provenance, run_jobs

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("development", "confirmation"),
                        default="development")
    parser.add_argument("--out", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    suite = CONFIRMATION if args.suite == "confirmation" else DATASETS
    if args.datasets is None:
        args.datasets = ",".join(str(d) for d in suite)
    if args.out is None:
        args.out = ("docs/book/cases/bench_realdata.json" if args.suite == "development"
                    else "docs/book/cases/bench_realdata_confirmation.json")
    for data_id in args.datasets.split(","):
        load(int(data_id))          # download once, before the workers start
    specs = [(int(d), method, seed) for d in args.datasets.split(",")
             for method in args.methods.split(",") for seed in range(args.splits)]
    env = dict(os.environ, OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    rows = run_jobs("bench_realdata", specs, HERE / ".cache" / "realdata",
                    workers=args.workers, timeout=3600, env=env,
                    uses_mboostrs=lambda spec: "mboostrs" in str(spec[1]))
    import importlib.metadata as md

    versions = {}
    for package in ("mboostrs", "interpret", "lightgbm", "xgboost", "catboost", "scikit-learn",
                    "torch", "pytabkit", "tabpfn", "perpetual"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            pass
    versions["mgcv"] = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("mgcv")))'],
        capture_output=True, text=True).stdout.strip()
    protocol = {"source": "OpenML-CTR23 (study 353)",
                "suite": args.suite,
                "datasets": {str(k): v for k, v in suite.items()},
                "splits": args.splits, "train_share": 0.8, "max_train": MAX_TRAIN,
                "max_test": MAX_TEST,
                "score": "test R^2 = 1 - MSE / Var(y_test) on the response scale; a "
                         "distributional (LSS) fit is scored on its location parameter "
                         "mu (scored_component 'mu' in its rows)",
                "tuning": "mboostrs: autospec(budget=200), rec.fit; interactions='adopt'; "
                          "interactions='adopt' with residual_tree=True; "
                          "refit='reml'; deselect=0.025; tune_mstop('y ~ .'); mgcv: bam(fREML, discrete) "
                          "with s(x) for numeric columns with > 10 values; EBM, CatBoost: "
                          "defaults; LightGBM: early stopping on 20% of the training part "
                          "over 15/63 leaves; XGBoost: the same early stopping over depths 4/8; MLP: 3x256 ReLU, "
                          "AdamW, quantile-normal inputs, dropout 0/0.2 and epochs by the same "
                          "early stopping; Perpetual: budget 0.5 (default), squared loss; "
                          "RealMLP: pytabkit RealMLP_TD defaults; TabPFN: "
                          "TabPFNRegressor defaults on CPU; ridge: RidgeCV on standardized one-hot "
                          "features; lasso, elastic net, relaxed lasso, adaptive lasso: "
                          "the same design, penalty by 5-fold CV; spline lasso: LassoCV on "
                          "cubic B-spline expansions of numeric columns plus indicators"}
    Path(args.out).write_text(json.dumps({
        "provenance": provenance(protocol=protocol, repeats={"splits": args.splits},
                                 uncertainty="mean over splits; per-data-set ranks",
                                 versions=versions),
        "protocol": protocol, "results": rows}, default=float))
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
