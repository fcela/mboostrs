"""The methods compared. Each arm sees the development rows and the test
predictors, never the test response, and returns a Forecast.

Every arm finishes with a model fitted on all development rows. Arms that
choose a boosting step count on held-out rows use an inner 80/20 split of the
development rows, then refit on all of them with the chosen count. Arms whose
library has its own cross-validation (mboostrs `tune_mstop`) use it instead.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from tasks import LEVELS

# Quantile levels fitted by arms that fit one model per level. The forecast at
# the 99 scoring levels is linear interpolation in tau between them, constant
# beyond 0.01 and 0.99 (which are fitted, so no extrapolation is needed).
QR_LEVELS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      0.95, 0.975, 0.99])
MAX_CATEGORY_LEVELS = 50
#: "v1" (response divided by its SD for every arm) or "v2" (raw response).
SHOOTOUT_VERSION = __import__("os").environ.get("SHOOTOUT_VERSION", "v2")
EARLY_STOP = 50
MAX_ROUNDS = 2000


@dataclass
class Forecast:
    quantiles: np.ndarray            # (n, 99), non-decreasing in each row
    mean: np.ndarray                 # (n,)
    logpdf: object = None            # callable y -> (n,) log density, parametric arms only
    info: dict = field(default_factory=dict)


@dataclass
class Prepared:
    dev: pd.DataFrame                # imputed; categoricals as strings; includes "y"
    test: pd.DataFrame               # same columns without "y"
    numeric: list
    categorical: list
    x_dev: np.ndarray                # numeric matrix with one-hot categoricals
    x_test: np.ndarray
    y: np.ndarray
    positive: bool
    seed: int
    scale: float = 1.0               # the response was divided by this (see prepare)


# ----------------------------------------------------------------- preparation
def prepare(dev, test, seed):
    """Shared preprocessing, identical for every arm.

    Numeric missing values take the development median; categorical missing
    values become the level "NA". Categoricals keep their 50 most frequent
    development levels and map the rest, and unseen test levels, to "_other"
    (or to the development mode when "_other" does not occur in development).
    Columns constant on the development rows are dropped. The response is
    divided by its development standard deviation.
    """
    dev, test = dev.copy(), test.drop(columns=["y"]).copy()
    predictors = [c for c in dev.columns if c != "y"]
    numeric, categorical = [], []
    for c in predictors:
        if pd.api.types.is_bool_dtype(dev[c]) or not pd.api.types.is_numeric_dtype(dev[c]):
            categorical.append(c)
        else:
            numeric.append(c)
    for c in numeric:
        median = float(np.nanmedian(dev[c].astype(float))) if dev[c].notna().any() else 0.0
        dev[c] = dev[c].astype(float).fillna(median)
        test[c] = test[c].astype(float).fillna(median)
    for c in categorical:
        d = dev[c].astype(object).where(dev[c].notna(), "NA").astype(str)
        t = test[c].astype(object).where(test[c].notna(), "NA").astype(str)
        keep = list(d.value_counts().index[:MAX_CATEGORY_LEVELS])
        d = d.where(d.isin(keep), "_other")
        fallback = "_other" if (d == "_other").any() else d.mode().iloc[0]
        t = t.where(t.isin(set(d)), fallback)
        dev[c], test[c] = d, t
    constant = [c for c in numeric + categorical if dev[c].nunique() < 2]
    numeric = [c for c in numeric if c not in constant]
    categorical = [c for c in categorical if c not in constant]
    dev = dev[numeric + categorical + ["y"]]
    test = test[numeric + categorical]
    x_dev, x_test = _design(dev, test, numeric, categorical)
    # Version 1 gave every arm the response divided by its development SD
    # (run_one multiplied the forecast back), because boosted LSS fits were
    # not scale-equivariant: their mean moved by about nu per step in response
    # units, so a response in the hundreds of thousands never left its offset.
    # Version 2 (the default) gives every arm the raw response: mboostrs now
    # scales its distributional and quantile fits itself.
    raw = dev["y"].to_numpy(float)
    scale = (float(np.std(raw)) or 1.0) if SHOOTOUT_VERSION == "v1" else 1.0
    dev["y"] = raw / scale
    y = dev["y"].to_numpy(float)
    return Prepared(dev, test, numeric, categorical, x_dev, x_test, y,
                    bool(np.min(y) > 0), seed, scale)


def _design(dev, test, numeric, categorical):
    blocks_d = [dev[numeric].to_numpy(float)]
    blocks_t = [test[numeric].to_numpy(float)]
    for c in categorical:
        levels = sorted(set(dev[c]))
        blocks_d.append((dev[c].to_numpy()[:, None] == np.array(levels)[None, :]).astype(float))
        blocks_t.append((test[c].to_numpy()[:, None] == np.array(levels)[None, :]).astype(float))
    return np.hstack(blocks_d), np.hstack(blocks_t)


def inner_split(n, seed):
    order = np.random.default_rng(10_000 + seed).permutation(n)
    cut = int(round(0.8 * n))
    return np.sort(order[:cut]), np.sort(order[cut:])


# ------------------------------------------------------------------ utilities
def interpolate_levels(q_fitted, fitted_levels=QR_LEVELS):
    q_fitted = np.sort(q_fitted, axis=1)  # rearrangement (Chernozhukov et al. 2010)
    out = np.empty((q_fitted.shape[0], len(LEVELS)))
    for i in range(q_fitted.shape[0]):
        out[i] = np.interp(LEVELS, fitted_levels, q_fitted[i])
    return out


def weighted_quantiles(weights, values):
    """Lower quantiles of row-wise weighted empirical distributions.

    weights: (n, m) non-negative rows summing to 1 over sorted values (m,).
    Returns quantiles at LEVELS and the means.
    """
    cdf = np.cumsum(weights, axis=1)
    cdf /= cdf[:, -1:]
    idx = np.empty((weights.shape[0], len(LEVELS)), dtype=np.int64)
    for j, tau in enumerate(LEVELS):
        idx[:, j] = np.argmax(cdf >= tau - 1e-12, axis=1)
    return values[idx], weights @ values / weights.sum(axis=1)


def _parametric(q_fn, n):
    return np.column_stack([np.asarray(q_fn(tau), float).reshape(n) for tau in LEVELS])


def _formula(prep):
    terms = []
    for c in prep.numeric:
        terms.append(f"bbs({c})" if prep.dev[c].nunique() >= 12 else f"bols({c})")
    terms += [f"bols({c})" for c in prep.categorical]
    return "y ~ " + " + ".join(terms)


def _polars(frame):
    import polars as pl
    return pl.from_pandas(frame)


# ------------------------------------------------------------------- mboostrs
def _lss_forecast(model, test, family):
    params = model.predict_parameters(test)
    n = len(test)
    q = _parametric(lambda tau: family.quantile(np.full(n, tau), params), n)
    mean = np.asarray(family.mean(params), float).reshape(n)
    return Forecast(np.sort(q, axis=1), mean,
                    lambda y: np.log(np.asarray(family.pdf(np.asarray(y, float), params), float)),
                    {"parameters": list(params)})


def mboostrs_fit(prep, **kwargs):
    """`mb.fit(dev, "y")`: the library's automated entry point."""
    import mboostrs as mb

    fit = mb.fit(_polars(prep.dev), "y", seed=prep.seed, **kwargs)
    model = fit.model
    test = _polars(prep.test)
    family = getattr(model, "_family", None)
    info = {"final": fit.trail.final,
            "family": type(family).__name__ if family is not None else None}
    if hasattr(model, "predict_parameters") and family is not None and hasattr(family, "param_names"):
        f = _lss_forecast(model, test, family)
        f.info.update(info, route="distributional")
        return f
    # A mean model: its family's predictive distribution at the fitted mean,
    # with a Gaussian residual standard deviation from the development rows.
    mean = np.asarray(model.predict(test, type="response"), float).reshape(-1)
    fitted = np.asarray(model.predict(_polars(prep.dev.drop(columns=["y"])), type="response"),
                        float).reshape(-1)
    name = info["family"] or ""
    n = len(mean)
    if "Poisson" in name:
        q = _parametric(lambda tau: stats.poisson.ppf(tau, mean), n)
        return Forecast(q, mean, lambda y: stats.poisson.logpmf(np.round(y), mean),
                        {**info, "route": "mean model, Poisson"})
    sd = float(np.std(prep.y - fitted, ddof=1))
    q = _parametric(lambda tau: stats.norm.ppf(tau, mean, sd), n)
    return Forecast(q, mean, lambda y: stats.norm.logpdf(y, mean, sd),
                    {**info, "route": "mean model, Gaussian residual sd", "sd": sd})


def mboostrs_fit_lss_trees(prep):
    """`mb.fit(dev, "y", lss_trees=True)`: the automated entry point, with the
    joint-tree step offered to an adopted distributional model."""
    return mboostrs_fit(prep, lss_trees=True)


def _tuned_lss(prep, family, method="cyclic", tree_depth=None):
    import mboostrs as mb

    dev = _polars(prep.dev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # A distributional specification is one formula per parameter; every
        # parameter gets every term, and with `tree_depth` a joint tree over
        # the numeric predictors.
        rhs = _formula(prep)
        if tree_depth is not None and len(prep.numeric) >= 2:
            rhs += f" + btree({', '.join(prep.numeric)}, maxdepth={tree_depth})"
        formula = {name: rhs for name in family.param_names}
        tuned = mb.tune_mstop(formula, dev, family=family, folds=5, seed=prep.seed,
                              method=method)
    return tuned


def mboostrs_gaussian_lss(prep):
    """GaussianLSS with one smooth (bbs) or linear (bols) term per predictor,
    step count by `tune_mstop` 5-fold CV."""
    import mboostrs as mb

    tuned = _tuned_lss(prep, mb.GaussianLSS())
    f = _lss_forecast(tuned.model, _polars(prep.test), tuned.model._family)
    f.info.update(mstop=_mstop(tuned), at_boundary=bool(getattr(tuned, "at_boundary", False)))
    return f


def _lss_candidates(prep):
    import mboostrs as mb

    cands = {"GaussianLSS": mb.GaussianLSS(), "StudentTLSS": mb.StudentTLSS()}
    if prep.positive:
        cands.update(GammaLSS=mb.GammaLSS(), LogNormalLSS=mb.LogNormalLSS())
    return cands


def mboostrs_lss_select(prep, method="cyclic", tree_depth=None):
    """As mboostrs_gaussian_lss for each candidate family (Gaussian, Student t,
    and gamma and log-normal for a positive response); the family with the
    lowest cross-validated negative log-likelihood is kept."""
    best, scores = None, {}
    for name, family in _lss_candidates(prep).items():
        try:
            tuned = _tuned_lss(prep, family, method=method, tree_depth=tree_depth)
            scores[name] = float(_cv_risk(tuned))
        except Exception as exc:  # a candidate that cannot be fitted is skipped
            scores[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        if best is None or scores[name] < scores[best[0]]:
            best = (name, tuned)
    if best is None:
        raise RuntimeError(f"no candidate family fitted: {scores}")
    f = _lss_forecast(best[1].model, _polars(prep.test), best[1].model._family)
    f.info.update(chosen=best[0], cv_scores=scores, mstop=_mstop(best[1]))
    return f


def mboostrs_lss_select_noncyclic(prep):
    """mboostrs.LSS_select with the non-cyclic algorithm, its total step count
    tuned by `tune_mstop` (#4)."""
    return mboostrs_lss_select(prep, method="noncyclic")


def mboostrs_lss_select_tree(prep):
    """mboostrs.LSS_select with a joint depth-6 tree over the numeric
    predictors added to every parameter."""
    return mboostrs_lss_select(prep, tree_depth=6)


def _mstop(tuned):
    m = getattr(tuned, "mstop", None)
    return m if isinstance(m, (int, dict)) else str(m)


def _cv_risk(tuned):
    # The aggregated out-of-fold family risk (mean negative log-likelihood per
    # row) along the grid; its minimum is the risk at the chosen step count.
    return float(np.min(np.asarray(tuned.risk, float)))


def mboostrs_quantiles(prep):
    """One additive quantile model (Quantile family, same terms) per level in
    QR_LEVELS, each step count by `tune_mstop` 5-fold CV of the check loss,
    then sorted per row."""
    import mboostrs as mb

    dev, test = _polars(prep.dev), _polars(prep.test)
    cols, mstops = [], {}
    for tau in QR_LEVELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tuned = mb.tune_mstop(_formula(prep), dev, family=mb.Quantile(tau=float(tau)),
                                  folds=5, seed=prep.seed)
        cols.append(np.asarray(tuned.model.predict(test), float).reshape(-1))
        mstops[float(tau)] = _mstop(tuned)
    q = np.column_stack(cols)
    crossing = float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))
    qq = interpolate_levels(q)
    return Forecast(qq, qq.mean(axis=1), None, {"mstop": mstops, "crossing_share": crossing})


# ------------------------------------------------------------------ hessboost
def _hessboost_fit(prep, family, rounds=None):
    import hessboost as hb

    params = {"objective": f"dist:{family}", "nthread": 1, "seed": prep.seed}
    x = prep.x_dev.astype(np.float32)
    y = prep.y.astype(np.float32)
    if rounds is None:
        tr, va = inner_split(len(y), prep.seed)
        dtr = hb.DMatrix(x[tr], label=y[tr])
        dva = hb.DMatrix(x[va], label=y[va])
        booster = hb.train(params, dtr, MAX_ROUNDS, evals=[(dva, "val")],
                           early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        rounds = int(booster.best_iteration) + 1
        dist = booster.predict_distribution(dva)
        val_nll = float(-np.mean(dist.log_prob(y[va].astype(np.float64))))
        return rounds, val_nll
    booster = hb.train(params, hb.DMatrix(x, label=y), rounds, verbose_eval=False)
    return booster


def _hessboost_forecast(prep, family):
    import hessboost as hb

    rounds, val_nll = _hessboost_fit(prep, family)
    booster = _hessboost_fit(prep, family, rounds)
    dist = booster.predict_distribution(hb.DMatrix(prep.x_test.astype(np.float32)))
    n = prep.x_test.shape[0]
    q = _parametric(lambda tau: dist.quantile(float(tau)), n)
    return Forecast(np.sort(q, axis=1), np.asarray(dist.mean(), float),
                    lambda y: np.asarray(dist.log_prob(np.asarray(y, float)), float),
                    {"family": family, "rounds": rounds, "inner_val_nll": val_nll})


def hessboost_normal(prep):
    """hessboost `dist:normal`, library defaults, rounds by inner early stopping."""
    return _hessboost_forecast(prep, "normal")


HESSBOOST_GRID = [{"max_depth": d, "min_child_weight": w, "eta": 0.05}
                  for d in (2, 4, 6) for w in (1, 20)]


def hessboost_tuned(prep):
    """hessboost over families (normal, student_t, and gamma and lognormal for
    a positive response) x HESSBOOST_GRID, each with rounds by inner early
    stopping on NLL; the configuration with the lowest inner-holdout CRPS is
    refitted on all development rows."""
    import hessboost as hb

    fams = ["normal", "student_t"] + (["gamma", "lognormal"] if prep.positive else [])
    x, y = prep.x_dev.astype(np.float32), prep.y.astype(np.float32)
    tr, va = inner_split(len(y), prep.seed)
    dtr, dva = hb.DMatrix(x[tr], label=y[tr]), hb.DMatrix(x[va], label=y[va])
    trials = []
    for fam in fams:
        for grid in HESSBOOST_GRID:
            params = {"objective": f"dist:{fam}", "nthread": 1, "seed": prep.seed, **grid}
            try:
                booster = hb.train(params, dtr, MAX_ROUNDS, evals=[(dva, "val")],
                                   early_stopping_rounds=EARLY_STOP, verbose_eval=False)
                rounds = int(booster.best_iteration) + 1
                dist = booster.predict_distribution(dva)
                q = np.sort(_parametric(lambda tau, d=dist: d.quantile(float(tau)), len(va)), axis=1)
                score = float(crps_quantiles(q, prep.y[va]).mean())
            except Exception as exc:
                trials.append({"family": fam, **grid, "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            trials.append({"family": fam, **grid, "rounds": rounds, "inner_crps": score})
    ok = [t for t in trials if np.isfinite(t.get("inner_crps", np.nan))]
    if not ok:
        raise RuntimeError(f"no hessboost configuration fitted: {trials[:3]}")
    best = min(ok, key=lambda t: t["inner_crps"])
    params = {"objective": f"dist:{best['family']}", "nthread": 1, "seed": prep.seed,
              **{k: best[k] for k in ("max_depth", "min_child_weight", "eta")}}
    booster = hb.train(params, hb.DMatrix(x, label=y), best["rounds"], verbose_eval=False)
    dist = booster.predict_distribution(hb.DMatrix(prep.x_test.astype(np.float32)))
    n = prep.x_test.shape[0]
    q = _parametric(lambda tau: dist.quantile(float(tau)), n)
    return Forecast(np.sort(q, axis=1), np.asarray(dist.mean(), float),
                    lambda yy: np.asarray(dist.log_prob(np.asarray(yy, float)), float),
                    {"chosen": best, "trials": trials})


def crps_quantiles(q, y):
    from metrics import crps_from_quantiles
    return crps_from_quantiles(q, np.asarray(y, float))


def _select_on_inner(prep, configs, fit_forecast):
    """Fit each config on the inner 80%, score CRPS on the inner 20%, refit the
    best on all development rows."""
    tr, va = inner_split(len(prep.y), prep.seed)
    inner = Prepared(prep.dev.iloc[tr], prep.dev.iloc[va].drop(columns=["y"]), prep.numeric,
                     prep.categorical, prep.x_dev[tr], prep.x_dev[va], prep.y[tr],
                     prep.positive, prep.seed)
    trials = []
    for cfg in configs:
        f = fit_forecast(inner, cfg)
        trials.append({**cfg, "inner_crps": float(crps_quantiles(f.quantiles, prep.y[va]).mean())})
    best = min(trials, key=lambda t: t["inner_crps"])
    cfg = {k: v for k, v in best.items() if k != "inner_crps"}
    f = fit_forecast(prep, cfg)
    f.info.update(chosen=cfg, trials=trials)
    return f


# -------------------------------------------------------------------- NGBoost
def ngboost_normal(prep):
    """NGBoost Normal, library defaults, rounds by inner early stopping."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    tr, va = inner_split(len(prep.y), prep.seed)
    probe = NGBRegressor(Dist=Normal, n_estimators=MAX_ROUNDS, random_state=prep.seed,
                         verbose=False)
    probe.fit(prep.x_dev[tr], prep.y[tr], X_val=prep.x_dev[va], Y_val=prep.y[va],
              early_stopping_rounds=EARLY_STOP)
    rounds = int(probe.best_val_loss_itr or MAX_ROUNDS) + 1
    model = NGBRegressor(Dist=Normal, n_estimators=rounds, random_state=prep.seed,
                         verbose=False).fit(prep.x_dev, prep.y)
    dist = model.pred_dist(prep.x_test)
    mu, sd = np.asarray(dist.params["loc"]), np.asarray(dist.params["scale"])
    q = _parametric(lambda tau: stats.norm.ppf(tau, mu, sd), len(mu))
    return Forecast(q, mu, lambda y: stats.norm.logpdf(y, mu, sd), {"rounds": rounds})


# ------------------------------------------------------------------- CatBoost
def catboost_uncertainty(prep):
    """CatBoost RMSEWithUncertainty (Gaussian mean and variance), defaults,
    native categoricals, iterations by inner early stopping."""
    from catboost import CatBoostRegressor, Pool

    x = prep.dev.drop(columns=["y"])
    tr, va = inner_split(len(prep.y), prep.seed)
    cats = prep.categorical
    common = dict(loss_function="RMSEWithUncertainty", thread_count=1,
                  random_seed=prep.seed, verbose=0, allow_writing_files=False)
    probe = CatBoostRegressor(**common, iterations=MAX_ROUNDS)
    probe.fit(Pool(x.iloc[tr], prep.y[tr], cat_features=cats),
              eval_set=Pool(x.iloc[va], prep.y[va], cat_features=cats),
              early_stopping_rounds=EARLY_STOP)
    rounds = int(probe.get_best_iteration()) + 1
    model = CatBoostRegressor(**common, iterations=rounds)
    model.fit(Pool(x, prep.y, cat_features=cats))
    pred = np.asarray(model.predict(Pool(prep.test, cat_features=cats)))
    mu, sd = pred[:, 0], np.sqrt(np.maximum(pred[:, 1], 1e-300))
    q = _parametric(lambda tau: stats.norm.ppf(tau, mu, sd), len(mu))
    return Forecast(q, mu, lambda y: stats.norm.logpdf(y, mu, sd), {"rounds": rounds})


# ------------------------------------------------------------------- LightGBM
def lightgbm_quantiles(prep):
    """LightGBM `objective="quantile"` per level in QR_LEVELS, defaults,
    rounds by inner early stopping on the check loss, then sorted."""
    import lightgbm as lgb

    tr, va = inner_split(len(prep.y), prep.seed)
    cols, rounds = [], {}
    for tau in QR_LEVELS:
        common = dict(objective="quantile", alpha=float(tau), n_jobs=1,
                      random_state=prep.seed, verbose=-1)
        probe = lgb.LGBMRegressor(**common, n_estimators=MAX_ROUNDS)
        probe.fit(prep.x_dev[tr], prep.y[tr], eval_set=[(prep.x_dev[va], prep.y[va])],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        r = int(probe.best_iteration_ or MAX_ROUNDS)
        model = lgb.LGBMRegressor(**common, n_estimators=r).fit(prep.x_dev, prep.y)
        cols.append(model.predict(prep.x_test))
        rounds[float(tau)] = r
    q = np.column_stack(cols)
    crossing = float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))
    qq = interpolate_levels(q)
    return Forecast(qq, qq.mean(axis=1), None, {"rounds": rounds, "crossing_share": crossing})


# -------------------------------------------------------------------- forests
def quantile_forest(prep):
    """quantile-forest RandomForestQuantileRegressor defaults (Meinshausen QRF)."""
    from quantile_forest import RandomForestQuantileRegressor

    model = RandomForestQuantileRegressor(n_jobs=1, random_state=prep.seed)
    model.fit(prep.x_dev, prep.y)
    q = np.asarray(model.predict(prep.x_test, quantiles=list(LEVELS)), float)
    mean = np.asarray(model.predict(prep.x_test, quantiles="mean"), float).reshape(-1)
    return Forecast(np.sort(q, axis=1), mean, None, {})


def _drforest(prep, criterion, **kwargs):
    from drforest import DistributionalRandomForest

    model = DistributionalRandomForest(criterion=criterion, random_state=prep.seed, **kwargs)
    model.fit(prep.x_dev, prep.y)
    q = np.asarray(model.predict_quantiles(prep.x_test, list(LEVELS)), float)
    q = q.reshape(prep.x_test.shape[0], len(LEVELS))
    mean = np.asarray(model.predict(prep.x_test), float).reshape(-1)
    return Forecast(np.sort(q, axis=1), mean, None, {"criterion": criterion})


def drforest_mmd(prep):
    """drforest DistributionalRandomForest, default MMD splitting."""
    return _drforest(prep, "mmd")


def drforest_cart(prep):
    """drforest DistributionalRandomForest with CART (mean) splitting."""
    return _drforest(prep, "cart")


def drforest_tuned(prep):
    """drforest over criterion {mmd, cart} x min_samples_leaf {5, 20, 50},
    chosen by inner-holdout CRPS."""
    configs = [{"criterion": c, "min_samples_leaf": m} for c in ("mmd", "cart") for m in (5, 20, 50)]
    return _select_on_inner(prep, configs, lambda p, cfg: _drforest(
        p, cfg["criterion"], min_samples_leaf=cfg["min_samples_leaf"]))


DISTREE = dict(n_trees=100, subsample=0.6, max_depth=6)


def distreebution_crps_tuned(prep):
    """CRPS-RF over max_depth {4, 6, 10}, chosen by inner-holdout CRPS."""
    import distreebu_rs as rs

    configs = [{"max_depth": d} for d in (4, 6, 10)]
    return _select_on_inner(prep, configs, lambda p, cfg: _distree_forest(
        p, lambda: rs.RegressionTreeCRPS(max_depth=cfg["max_depth"], loo=True),
        max_depth=cfg["max_depth"]))


def _distree_forest(prep, make_tree, max_depth=None):
    """Bagged DisTreebution trees as in DisTreebution's `UQ.train_trees`: each
    tree sees a 60% subsample without replacement; a test row's predictive
    distribution pools the training responses of its leaf in every tree, each
    tree with equal weight and uniform within the leaf."""
    y = prep.y
    values = np.unique(y)
    x_dev = prep.x_dev
    trees = []
    rng = np.random.default_rng(prep.seed)
    n_sub = int(DISTREE["subsample"] * len(y))
    for _ in range(DISTREE["n_trees"]):
        idx = rng.permutation(len(y))[:n_sub]
        tree = make_tree()
        tree.fit(x_dev[idx].tolist(), y[idx].tolist())
        trees.append(tree)
    n = prep.x_test.shape[0]
    q = np.empty((n, len(LEVELS)))
    mean = np.empty(n)
    chunk = 500
    for start in range(0, n, chunk):
        rows = np.arange(start, min(n, start + chunk))
        w = np.zeros((len(rows), len(values)))
        xt = prep.x_test[rows].tolist()
        for tree in trees:
            for leaf_rows, leaf_y in tree.get_values_leaf(xt, list(range(len(rows)))):
                pos = np.searchsorted(values, np.asarray(leaf_y, float))
                share = 1.0 / len(leaf_y)
                counts = np.bincount(pos, minlength=len(values)) * share
                w[np.asarray(leaf_rows, int)] += counts[None, :]
        w /= len(trees)
        q[rows], mean[rows] = weighted_quantiles(w, values)
    return Forecast(q, mean, None, {**DISTREE, "max_depth": max_depth or DISTREE["max_depth"]})


def distreebution_crps(prep):
    """DisTreebution CRPS-RF: trees split on leave-one-out CRPS, bagged."""
    import distreebu_rs as rs

    return _distree_forest(prep, lambda: rs.RegressionTreeCRPS(
        max_depth=DISTREE["max_depth"], loo=True))


def distreebution_pmqrf(prep):
    """DisTreebution PMQRF: trees split on the summed pinball loss at the 99
    levels with the leave-one-out correction, bagged."""
    import distreebu_rs as rs

    return _distree_forest(prep, lambda: rs.RegressionTreeQuantile(
        quantiles=list(LEVELS), max_depth=DISTREE["max_depth"], loo=True))


# ----------------------------------------------------------------- references
def reference_gaussian_linear(prep):
    """Least squares on the design matrix with one constant residual sd."""
    x = np.column_stack([np.ones(len(prep.y)), prep.x_dev])
    xt = np.column_stack([np.ones(prep.x_test.shape[0]), prep.x_test])
    beta, *_ = np.linalg.lstsq(x, prep.y, rcond=None)
    resid = prep.y - x @ beta
    rank = np.linalg.matrix_rank(x)
    sd = float(np.sqrt(resid @ resid / max(len(prep.y) - rank, 1)))
    mu = xt @ beta
    q = _parametric(lambda tau: stats.norm.ppf(tau, mu, sd), len(mu))
    return Forecast(q, mu, lambda y: stats.norm.logpdf(y, mu, sd), {"sd": sd})


def reference_climatology(prep):
    """The empirical distribution of the development responses, for every row."""
    n = prep.x_test.shape[0]
    q = np.quantile(prep.y, LEVELS, method="inverted_cdf")
    return Forecast(np.repeat(q[None, :], n, axis=0), np.full(n, prep.y.mean()), None, {})


ARMS = {
    "mboostrs.fit": mboostrs_fit,
    "mboostrs.fit_lss_trees": mboostrs_fit_lss_trees,
    "mboostrs.GaussianLSS": mboostrs_gaussian_lss,
    "mboostrs.LSS_select": mboostrs_lss_select,
    "mboostrs.LSS_select_noncyclic": mboostrs_lss_select_noncyclic,
    "mboostrs.LSS_select_tree": mboostrs_lss_select_tree,
    "mboostrs.quantiles": mboostrs_quantiles,
    "hessboost.normal": hessboost_normal,
    "hessboost.tuned": hessboost_tuned,
    "ngboost.normal": ngboost_normal,
    "catboost.uncertainty": catboost_uncertainty,
    "lightgbm.quantiles": lightgbm_quantiles,
    "qrf": quantile_forest,
    "drforest.mmd": drforest_mmd,
    "drforest.cart": drforest_cart,
    "drforest.tuned": drforest_tuned,
    "distreebution.crps_rf": distreebution_crps,
    "distreebution.crps_rf_tuned": distreebution_crps_tuned,
    "distreebution.pmqrf": distreebution_pmqrf,
    "reference.gaussian_linear": reference_gaussian_linear,
    "reference.climatology": reference_climatology,
}


def unscale(forecast, scale):
    """Return a forecast for the original response from one for response / scale."""
    logpdf = forecast.logpdf
    if logpdf is not None:
        def logpdf_original(y, _f=logpdf, _s=scale):
            return np.asarray(_f(np.asarray(y, float) / _s), float) - np.log(_s)
        logpdf = logpdf_original
    return Forecast(forecast.quantiles * scale, forecast.mean * scale, logpdf,
                    {**forecast.info, "response_scale": scale})
