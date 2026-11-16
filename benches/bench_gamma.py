"""Gamma means: GammaReg against a Gamma GLM, and a smooth mean a linear GLM misses.

Two simulated tasks. The response is Gamma with mean ``mu`` and shape 4
(variance ``mu**2 / 4``, so the coefficient of variation is 1/2). Rows are
independent. Predictors are standard normal; ``noise`` is unused.

* ``linear``: ``log mu = 0.15 + 0.55 x1 - 0.30 x2``.
* ``smooth``: ``log mu = 0.05 + sin(1.5 x1) + 0.30 x2``. The mean is not
  linear in ``x1``.

Every method sees the same 1,500 training rows and is scored on an independent
4,000-row test sample. Four training seeds (0, 1, 2, 3). mboostrs step counts
are chosen by 5-fold cross-validation over every step of the fitted path.
statsmodels fits a Gamma GLM with a log link by IRLS, linear or with a
B-spline in ``x1``. scikit-learn's ``GammaRegressor(alpha=0)`` is the
unpenalised linear gamma regressor. R ``gamboost`` is not installed.

Scores on the test rows:

* mean gamma negative log-likelihood (nats per row), using that fit's mean
  and its own shape estimate. A Gaussian fit has no gamma density; its row
  leaves this blank and reports its own Gaussian negative log-likelihood
  instead;
* mean absolute error of the predicted mean;
* RMSE of the predicted log mean against the generating log mean, when every
  prediction is positive.

``GammaReg``'s shape is the ``nuisance()`` value (``Var = mu**2 / shape``).
statsmodels reports the reciprocal of that number as the GLM scale.

    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benches/bench_gamma.py --out docs/book/cases/bench_gamma.json

Polars holds the frames. statsmodels, patsy and scikit-learn receive NumPy
columns taken from those frames; pandas is not used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import patsy
import polars as pl
import statsmodels.api as sm
from scipy.special import gammaln
from sklearn.linear_model import GammaRegressor

import mboostrs as mb

N_TRAIN = 1500
N_TEST = 4000
SEEDS = (0, 1, 2, 3)
SHAPE = 4.0
NU = 0.1
FOLDS = 5
MSTOP = {"glm": 400, "gam": 1000, "gaussian": 800}
COLUMNS = ("x1", "x2", "noise")
SPLINE = "bs(x1, df=8, lower_bound=-6, upper_bound=6) + x2 + noise"
HERE = Path(__file__).resolve().parent


def simulate(seed, n, task):
    rng = np.random.default_rng(seed)
    frame = {name: rng.normal(size=n) for name in COLUMNS}
    if task == "linear":
        log_mu = 0.15 + 0.55 * frame["x1"] - 0.30 * frame["x2"]
    elif task == "smooth":
        log_mu = 0.05 + np.sin(1.5 * frame["x1"]) + 0.30 * frame["x2"]
    else:
        raise ValueError(task)
    mu = np.exp(log_mu)
    frame["y"] = rng.gamma(SHAPE, scale=mu / SHAPE)
    return pl.DataFrame(frame), mu, log_mu


def columns(frame):
    return {name: frame[name].to_numpy() for name in frame.columns}


def gamma_nll(y, mu, shape):
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, None)
    y = np.asarray(y, dtype=np.float64)
    shape = float(shape)
    return float(
        np.mean(
            gammaln(shape) + shape * np.log(mu) - shape * np.log(shape) - (shape - 1.0) * np.log(y) + shape * y / mu
        )
    )


def gaussian_nll(y, mu, sigma2):
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma2 = float(sigma2)
    return float(0.5 * np.log(2.0 * np.pi * sigma2) + 0.5 * np.mean((y - mu) ** 2) / sigma2)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def log_rmse(mu_hat, log_mu):
    mu_hat = np.asarray(mu_hat, dtype=np.float64)
    if np.any(mu_hat <= 0.0):
        return None
    return float(np.sqrt(np.mean((np.log(mu_hat) - np.asarray(log_mu)) ** 2)))


def design_pair(formula, train, test):
    train_cols = columns(train)
    design = patsy.dmatrix(formula, train_cols)
    test_x = np.asarray(patsy.build_design_matrices([design.design_info], columns(test))[0])
    return np.asarray(design), test_x, list(design.design_info.column_names)


def fit_statsmodels(train, test, formula):
    y = train["y"].to_numpy()
    x_train, x_test, names = design_pair(formula, train, test)
    started = time.perf_counter()
    result = sm.GLM(y, x_train, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    seconds = time.perf_counter() - started
    mu = np.asarray(result.predict(x_test), dtype=np.float64)
    # Pearson scale is the dispersion phi with Var = phi * mu**2, so the
    # shape GammaReg reports is 1/phi. `llf` uses this same scale.
    shape = 1.0 / float(result.scale)
    return {
        "mu": mu,
        "shape": shape,
        "seconds": seconds,
        "llf": float(result.llf),
        "params": {name: float(value) for name, value in zip(names, result.params, strict=True)},
        "converged": bool(getattr(result, "converged", True)),
    }


def fit_sklearn(train, test):
    names = ("x1", "x2", "noise")
    x_train = np.column_stack([train[name].to_numpy() for name in names])
    x_test = np.column_stack([test[name].to_numpy() for name in names])
    y = train["y"].to_numpy()
    started = time.perf_counter()
    # alpha=0 is unpenalised. sklearn still parameterises the mean with a log
    # link; it does not return a shape.
    model = GammaRegressor(alpha=0.0, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(x_train, y)
    seconds = time.perf_counter() - started
    mu_train = np.asarray(model.predict(x_train), dtype=np.float64)
    # Pearson dispersion on the training rows, matching a Gamma GLM's scale
    # up to the residual degrees of freedom.
    resid_df = max(len(y) - x_train.shape[1] - 1, 1)
    phi = float(np.sum(((y - mu_train) / mu_train) ** 2) / resid_df)
    return {
        "mu": np.asarray(model.predict(x_test), dtype=np.float64),
        "shape": 1.0 / phi,
        "seconds": seconds,
        "intercept": float(model.intercept_),
        "coef": {name: float(value) for name, value in zip(names, model.coef_, strict=True)},
    }


def _shape_of(model):
    nuisance = model.nuisance()
    if nuisance is None:
        return None
    values = np.ravel(np.asarray(nuisance, dtype=np.float64))
    if values.size != 1:
        return None
    return float(values[0])


def fit_boost(kind, formula, train, test, seed, family):
    mstop = MSTOP["gaussian" if kind == "gaussian" else kind]
    ctor = mb.glmboost if kind in ("glm", "gaussian") else mb.gamboost
    started = time.perf_counter()
    fit = ctor(formula, train, family=family, control=mb.boost_control(mstop, NU))
    cv = fit.cvrisk(type="kfold", B=FOLDS, seed=seed)
    chosen = int(cv.optimal_mstop)
    tuned = fit[:chosen]
    predict_kwargs = {"type": "response"}
    if kind == "gam":
        predict_kwargs["extrapolation"] = "allow"
    mu = np.asarray(tuned.predict(test, **predict_kwargs), dtype=np.float64).ravel()
    seconds = time.perf_counter() - started
    return {
        "mu": mu,
        "shape": _shape_of(tuned),
        "seconds": seconds,
        "mstop": chosen,
        "mstop_cap": mstop,
        "at_boundary": bool(cv.at_boundary()),
        "mstop_1se": int(cv.mstop_1se),
        "model": tuned,
    }


def score_row(task, seed, method, *, mu, shape, seconds, y, mu_true, log_mu, extra):
    mu = np.asarray(mu, dtype=np.float64).ravel()
    row = {
        "task": task,
        "seed": seed,
        "method": method,
        "gamma_nll": None if shape is None else gamma_nll(y, mu, shape),
        "nll_at_true_shape": gamma_nll(y, mu, SHAPE) if np.all(mu > 0.0) else None,
        "mae": mae(mu, mu_true),
        "log_rmse": log_rmse(mu, log_mu),
        "shape": shape,
        "seconds": seconds,
        "n_nonpositive": int(np.sum(mu <= 0.0)),
    }
    row.update(extra)
    return row


def run_task(task, seed):
    train, _, _ = simulate(seed, N_TRAIN, task)
    # The test sample does not depend on the training seed's rows. A fixed
    # test seed per task keeps the four fits on the same rows.
    test, mu_true, log_mu = simulate(10_000 + (0 if task == "linear" else 1), N_TEST, task)
    y = test["y"].to_numpy()
    rows = result_rows()
    common = dict(y=y, mu_true=mu_true, log_mu=log_mu)

    rows.append(
        score_row(
            task,
            seed,
            "generating Gamma",
            mu=mu_true,
            shape=SHAPE,
            seconds=0.0,
            extra={"mstop": None, "at_boundary": None},
            **common,
        )
    )

    linear = fit_statsmodels(train, test, "x1 + x2 + noise")
    rows.append(
        score_row(
            task,
            seed,
            "statsmodels Gamma GLM",
            mu=linear["mu"],
            shape=linear["shape"],
            seconds=linear["seconds"],
            extra={"converged": linear["converged"]},
            **common,
        )
    )

    spline = fit_statsmodels(train, test, SPLINE)
    rows.append(
        score_row(
            task,
            seed,
            "statsmodels Gamma spline GLM",
            mu=spline["mu"],
            shape=spline["shape"],
            seconds=spline["seconds"],
            extra={"converged": spline["converged"]},
            **common,
        )
    )

    learned = fit_sklearn(train, test)
    rows.append(
        score_row(
            task,
            seed,
            "sklearn GammaRegressor",
            mu=learned["mu"],
            shape=learned["shape"],
            seconds=learned["seconds"],
            extra={},
            **common,
        )
    )

    glm = fit_boost("glm", "y ~ x1 + x2 + noise", train, test, seed, mb.GammaReg())
    rows.append(
        score_row(
            task,
            seed,
            "mboostrs glmboost GammaReg",
            mu=glm["mu"],
            shape=glm["shape"],
            seconds=glm["seconds"],
            extra={"mstop": glm["mstop"], "at_boundary": glm["at_boundary"], "mstop_1se": glm["mstop_1se"]},
            **common,
        )
    )

    gam = fit_boost("gam", "y ~ bbs(x1) + bols(x2) + bols(noise)", train, test, seed, mb.GammaReg())
    rows.append(
        score_row(
            task,
            seed,
            "mboostrs gamboost GammaReg",
            mu=gam["mu"],
            shape=gam["shape"],
            seconds=gam["seconds"],
            extra={"mstop": gam["mstop"], "at_boundary": gam["at_boundary"], "mstop_1se": gam["mstop_1se"]},
            **common,
        )
    )

    gauss = fit_boost("gaussian", "y ~ x1 + x2 + noise", train, test, seed, mb.Gaussian())
    mu_train = np.asarray(gauss["model"].predict(type="response"), dtype=np.float64).ravel()
    sigma2 = float(np.mean((train["y"].to_numpy() - mu_train) ** 2))
    rows.append(
        score_row(
            task,
            seed,
            "mboostrs glmboost Gaussian",
            mu=gauss["mu"],
            shape=None,
            seconds=gauss["seconds"],
            extra={
                "mstop": gauss["mstop"],
                "at_boundary": gauss["at_boundary"],
                "mstop_1se": gauss["mstop_1se"],
                "gaussian_nll": gaussian_nll(y, gauss["mu"], sigma2),
                "sigma2": sigma2,
            },
            **common,
        )
    )

    recovery = None
    if task == "linear" and seed == 0:
        coef = glm["model"].coef(off2int=True)
        raw = glm["model"].coef(off2int=False)
        recovery = {
            "statsmodels": linear["params"],
            "glmboost_off2int": {key: float(value) for key, value in coef.items()},
            "glmboost_raw_intercept": float(raw["(Intercept)"]),
            "offset": float(glm["model"].offset) if np.ndim(glm["model"].offset) == 0 else None,
            "sklearn_intercept": learned["intercept"],
            "sklearn_coef": learned["coef"],
            "glmboost_shape": glm["shape"],
            "statsmodels_shape": linear["shape"],
            "glmboost_loglik": float(glm["model"].logLik()),
            "statsmodels_llf": linear["llf"],
        }
    return rows, recovery


def _call(label, fn):
    try:
        value = fn()
        detail = value if isinstance(value, str) else f"{type(value).__name__}: {value!r}"
        return {"label": label, "ok": True, "detail": detail[:500]}
    except Exception as exc:
        return {"label": label, "ok": False, "error": type(exc).__name__, "detail": str(exc)[:500]}


def usability(train):
    probes = []
    probes.append(
        _call(
            "gamboost(family=GammaLSS())",
            lambda: mb.gamboost("y ~ bbs(x1)", train, family=mb.GammaLSS(), control=mb.boost_control(5, 0.1)),
        )
    )
    probes.append(
        _call(
            "glmboost(family=GammaLSS())",
            lambda: mb.glmboost("y ~ x1", train, family=mb.GammaLSS(), control=mb.boost_control(5, 0.1)),
        )
    )
    probes.append(
        _call(
            "GammaReg rejects a zero",
            lambda: mb.glmboost(
                "y ~ x1",
                pl.DataFrame({"y": [1.0, 0.0, 2.0], "x1": [0.1, 0.2, 0.3]}),
                family=mb.GammaReg(),
                control=mb.boost_control(5, 0.1),
            ),
        )
    )

    fit = mb.glmboost("y ~ x1 + x2 + noise", train, family=mb.GammaReg(), control=mb.boost_control(30, 0.1))
    link = np.asarray(fit.predict(type="link"), dtype=np.float64).ravel()
    response = np.asarray(fit.predict(type="response"), dtype=np.float64).ravel()
    probes.append(
        {
            "label": "predict default is the log mean",
            "ok": True,
            "detail": (
                "type='link' matches log(type='response') to "
                f"{float(np.max(np.abs(link - np.log(response)))):.2e}; "
                "the default type is 'link', so a caller who treats predict() as E[Y] "
                "scores the log mean."
            ),
        }
    )
    nuisance = fit.nuisance()
    probes.append(
        {
            "label": "GammaReg nuisance is the shape",
            "ok": nuisance is not None and np.asarray(nuisance).size == 1,
            "detail": (
                f"nuisance() = {float(np.ravel(nuisance)[0]):.4f}. "
                "Variance is mu**2 / shape. A Gamma GLM's scale is 1/shape, not this number."
            ),
        }
    )

    gaussian = mb.glmboost("y ~ x1", train, family=mb.Gaussian(), control=mb.boost_control(20, 0.1))
    gaussian_nuisance = gaussian.nuisance()
    probes.append(
        {
            "label": "glmboost Gaussian nuisance()",
            "ok": gaussian_nuisance is None,
            "detail": f"returned {gaussian_nuisance!r}.",
        }
    )

    def _continue_then_cvrisk():
        extended = fit.continue_to(40)
        extended.cvrisk(type="kfold", B=3, seed=0)
        return "cvrisk after continue_to returned"

    probes.append(_call("cvrisk after glmboost.continue_to", _continue_then_cvrisk))

    gam = mb.gamboost("y ~ bbs(x1)", train, family=mb.GammaReg(), control=mb.boost_control(15, 0.1))

    def _gam_continue_then_cvrisk():
        extended = gam.continue_to(25)
        extended.cvrisk(type="kfold", B=3, seed=0)
        return "cvrisk after gamboost.continue_to returned"

    probes.append(_call("cvrisk after gamboost.continue_to", _gam_continue_then_cvrisk))

    recommendation = mb.autospec(train, response="y")
    probes.append(
        {
            "label": "autospec on a gamma sample",
            "ok": type(recommendation.family).__name__ == "GammaReg",
            "detail": (
                f"family={type(recommendation.family).__name__}(), "
                f"formula={recommendation.formula!r}, "
                f"defaulted={recommendation.defaulted!r}, "
                f"priority={recommendation.priority!r}"
            ),
        }
    )

    caught = []

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        caught.append(f"{category.__name__}: {message}")

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _showwarning
        mb.gamboost(
            "y ~ bbs(x1) + bols(x2)",
            train,
            family=mb.GammaReg(),
            control=mb.boost_control(20, 0.1),
        ).predict(
            pl.DataFrame(
                {
                    "x1": np.array([6.0]),
                    "x2": np.array([0.0]),
                }
            ),
            type="response",
        )
    probes.append(
        {
            "label": "bbs predict outside the training range",
            "ok": True,
            "detail": caught[0][:500] if caught else "no warning",
        }
    )
    return probes


from benches._quality_provenance import attach_provenance, begin_run, result_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    begin_run(__file__, simulate, N_TRAIN, N_TEST, ("linear", "smooth"))
    results = result_rows()
    recovery = None
    for task in ("linear", "smooth"):
        for seed in SEEDS:
            print(f"{task} seed {seed}", flush=True)
            rows, found = run_task(task, seed)
            results.extend(rows)
            if found is not None:
                recovery = found
            for row in rows:
                nll = row["gamma_nll"]
                nll_text = "na" if nll is None else f"{nll:.4f}"
                print(
                    f"  {row['method']}: nll={nll_text} mae={row['mae']:.4f} "
                    f"shape={row['shape']} sec={row['seconds']:.3f} "
                    f"mstop={row.get('mstop')} boundary={row.get('at_boundary')}",
                    flush=True,
                )
    train, _, _ = simulate(0, N_TRAIN, "linear")
    payload = {
        "design": {
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "seeds": list(SEEDS),
            "shape": SHAPE,
            "nu": NU,
            "folds": FOLDS,
            "mstop": MSTOP,
            "linear": "log mu = 0.15 + 0.55 x1 - 0.30 x2",
            "smooth": "log mu = 0.05 + sin(1.5 x1) + 0.30 x2",
            "spline": SPLINE,
        },
        "results": results,
        "recovery": recovery,
        "usability": {"probes": usability(train)},
    }
    attach_provenance(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
