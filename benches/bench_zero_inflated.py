"""Zero-inflated counts: ZIPoLSS against a ZIP MLE, and the models an agent reaches first.

Two simulated tasks, each with four predictors of which three matter. Rows are
independent. The response is zero-inflated Poisson:

* ``linear``: ``log mu = 0.15 + 0.55 x1 - 0.30 x2``,
  ``logit pi = -0.35 + 0.95 z``. ``noise`` is unused.
* ``smooth``: ``log mu = 0.05 + sin(1.5 x1) + 0.30 x2``, same inflation
  equation. The mean is not linear in ``x1``.

``pi`` is the probability of a structural zero. The observed count is 0 with
probability ``pi`` and Poisson(``mu``) otherwise, so
``E[Y] = (1 - pi) mu`` and
``P(Y = 0) = pi + (1 - pi) exp(-mu)``.

Every method sees the same 1,500 training rows and is scored on an independent
4,000-row test sample. Four training seeds (0, 1, 2, 3). mboostrs step counts
are chosen by 5-fold cross-validation on the training rows; if the optimum is
the largest budget, the path is extended once and cross-validated again.
statsmodels fits the same linear (or B-spline) design by maximum likelihood,
with no selection. R ``gamboostLSS`` is the boosting twin of ``ZIPoLSS`` and
is not installed here; the MLE is the accuracy reference for this design, not
a second boosting implementation.

The hurdle column is the two-part model ``Hurdle``'s docstring tells a caller
to build: ``Binomial`` for ``y > 0``, and ``Hurdle`` (zero-truncated negative
binomial) on the positive counts. It is a different likelihood from ZIP.

Scores, all on the test rows:

* mean negative log probability of the observed count (nats per row);
* mean absolute error of the predicted expectation ``E[Y | x]``;
* RMSE of predicted ``P(Y = 0)`` against the generating probability;
* for ZIP fits only, RMSE of the structural-zero probability and of the
  Poisson mean.

    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benches/bench_zero_inflated.py --out docs/book/cases/bench_zero_inflated.json

Polars holds the frames. statsmodels and patsy receive NumPy columns taken
from those frames; pandas is not used.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import patsy
import polars as pl
from scipy.special import gammaln
from statsmodels.discrete.count_model import ZeroInflatedPoisson

import mboostrs as mb

N_TRAIN = 1500
N_TEST = 4000
SEEDS = (0, 1, 2, 3)
MSTOP_CAP = 1600
MSTOP_EXTEND = 3200
NU = 0.1
FOLDS = 5
COLUMNS = ("x1", "x2", "z", "noise")
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
    logit_pi = -0.35 + 0.95 * frame["z"]
    mu = np.exp(log_mu)
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    y = np.where(rng.random(n) < pi, 0.0, rng.poisson(mu))
    frame["y"] = y
    return pl.DataFrame(frame), mu, pi


def _clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)


def _clip_mu(mu):
    return np.clip(np.asarray(mu, dtype=np.float64), 1e-12, None)


def zip_logp(y, mu, pi):
    mu, pi = _clip_mu(mu), _clip_prob(pi)
    p0 = pi + (1.0 - pi) * np.exp(-mu)
    return np.where(
        y == 0,
        np.log(np.clip(p0, 1e-300, None)),
        np.log(1.0 - pi) + y * np.log(mu) - mu - gammaln(y + 1.0),
    )


def pois_logp(y, mu):
    mu = _clip_mu(mu)
    return y * np.log(mu) - mu - gammaln(y + 1.0)


def nb_logp(y, mu, sigma):
    """Negative binomial with size ``sigma`` and variance ``mu + mu**2/sigma``."""
    mu = _clip_mu(mu)
    size = max(float(np.asarray(sigma).ravel()[0]), 1e-8)
    p = size / (size + mu)
    return (gammaln(y + size) - gammaln(size) - gammaln(y + 1.0)
            + size * np.log(p) + y * np.log(1.0 - p))


def hurdle_logp(y, p_pos, mu, sigma):
    """Binomial gate times a zero-truncated negative binomial.

    ``Hurdle``'s ``sigma`` is the dispersion (variance ``mu + sigma mu**2``),
    so the size is ``1/sigma``. ``NBinomial.sigma`` is that size, not this one.
    """
    p_pos = _clip_prob(p_pos)
    mu = _clip_mu(mu)
    size = 1.0 / max(float(np.asarray(sigma).ravel()[0]), 1e-8)
    p = size / (size + mu)
    log_nb = (gammaln(y + size) - gammaln(size) - gammaln(y + 1.0)
              + size * np.log(p) + y * np.log(1.0 - p))
    log_p0 = size * (np.log(size) - np.log(size + mu))
    log_trunc = log_nb - np.log(np.clip(1.0 - np.exp(log_p0), 1e-300, None))
    return np.where(y == 0, np.log(1.0 - p_pos), np.log(p_pos) + log_trunc)


def mean_nll(logp):
    return float(-np.mean(logp))


def p0_zip(mu, pi):
    return pi + (1.0 - pi) * np.exp(-_clip_mu(mu))


def p0_pois(mu):
    return np.exp(-_clip_mu(mu))


def p0_nb(mu, sigma):
    mu = _clip_mu(mu)
    size = max(float(np.asarray(sigma).ravel()[0]), 1e-8)
    return np.exp(size * (np.log(size) - np.log(size + mu)))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _as_matrix(design):
    return np.asarray(design, dtype=np.float64)


def design_matrices(task, train, test):
    columns = {name: train[name].to_numpy() for name in COLUMNS}
    test_columns = {name: test[name].to_numpy() for name in COLUMNS}
    if task == "linear":
        formula = "x1 + x2 + z + noise"
    else:
        formula = "bs(x1, df=4, lower_bound=-6, upper_bound=6) + x2 + z + noise"
    # DesignMatrix is a NumPy subclass; no pandas frame is built.
    train_design = patsy.dmatrix(formula, columns)
    test_design = patsy.build_design_matrices(
        [train_design.design_info], test_columns)[0]
    return _as_matrix(train_design), _as_matrix(test_design)


def fit_statsmodels(task, train, test):
    x_train, x_test = design_matrices(task, train, test)
    y = train["y"].to_numpy()
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ZeroInflatedPoisson(
            y, x_train, exog_infl=x_train, inflation="logit"
        ).fit(method="bfgs", maxiter=400, disp=False)
    seconds = time.perf_counter() - started
    mu = np.asarray(result.predict(x_test, exog_infl=x_test, which="mean-main"), float)
    # prob-main is the probability the count is drawn from the Poisson, so
    # the structural-zero probability is its complement. Checked against the
    # generating pi on the linear task (RMSE well under 0.05 at this n).
    pi = 1.0 - np.asarray(
        result.predict(x_test, exog_infl=x_test, which="prob-main"), float)
    expectation = np.asarray(result.predict(x_test, exog_infl=x_test, which="mean"), float)
    return {
        "mu": mu, "pi": pi, "mean": expectation,
        "seconds": seconds,
        "converged": bool(result.mle_retvals.get("converged", False)),
        "mstop": None,
        "extended": False,
    }


def _cvrisk(model, *, seed, lss):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if lss:
            result = model.cvrisk(B=FOLDS, seed=seed, length_out=6)
        else:
            result = model.cvrisk(type="kfold", B=FOLDS, seed=seed)
    notes = [str(item.message) for item in caught]
    return result, any("largest budget" in note or "where the search stopped" in note for note in notes)


def _optimal_scalar(result, lss):
    optimal = result.optimal_mstop
    if lss:
        values = list(np.asarray(optimal).ravel())
        return int(max(values)), optimal
    return int(optimal), int(optimal)


def tune_standard(fit_fn, seed):
    started = time.perf_counter()
    fit = fit_fn(MSTOP_CAP)
    _result, at_boundary = _cvrisk(fit, seed=seed, lss=False)
    extended = False
    if at_boundary or int(fit.mstop) == MSTOP_CAP and _result.optimal_mstop == MSTOP_CAP:
        # cvrisk's own boundary warning is the signal; also extend when the
        # chosen step is the cap even if the warning text changes.
        extended = int(_result.optimal_mstop) >= MSTOP_CAP
    if extended and hasattr(fit, "continue_to"):
        fit = fit.continue_to(MSTOP_EXTEND)
        _result, at_boundary = _cvrisk(fit, seed=seed, lss=False)
    chosen, mstop = _optimal_scalar(_result, lss=False)
    tuned = fit[:chosen]
    return tuned, {
        "seconds": time.perf_counter() - started,
        "mstop": mstop,
        "extended": extended,
        "at_boundary": bool(at_boundary),
    }


def tune_lss(fit_fn, seed):
    started = time.perf_counter()
    fit = fit_fn(MSTOP_CAP)
    result, at_boundary = _cvrisk(fit, seed=seed, lss=True)
    chosen, mstop = _optimal_scalar(result, lss=True)
    extended = chosen >= MSTOP_CAP
    if extended:
        fit = fit_fn(MSTOP_EXTEND)
        result, at_boundary = _cvrisk(fit, seed=seed, lss=True)
        chosen, mstop = _optimal_scalar(result, lss=True)
    return fit, {
        "seconds": time.perf_counter() - started,
        "mstop": mstop if isinstance(mstop, int) else [int(v) for v in np.asarray(mstop).ravel()],
        "predict_mstop": mstop,
        "extended": extended,
        "at_boundary": bool(at_boundary or chosen >= MSTOP_EXTEND),
    }


def formula(task, kind):
    if kind == "glm":
        return "y ~ x1 + x2 + z + noise"
    if task == "linear":
        return "y ~ bols(x1) + bols(x2) + bols(z) + bols(noise)"
    return "y ~ bbs(x1, df=4, knots=10) + bols(x2) + bols(z) + bols(noise)"


def positive_frame(data, response_name):
    """Positive counts, with the binary response the hurdle docstring names."""
    return data.filter(pl.col("y") > 0)


def score_row(task, seed, method, *, nll, expectation, p0, mu, pi, meta, y, mu_true, pi_true):
    row = {
        "task": task,
        "seed": seed,
        "method": method,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "nll": nll,
        "mae": mae(y, expectation),
        "p0_rmse": rmse(p0, p0_zip(mu_true, pi_true)),
        "seconds": meta["seconds"],
        "mstop": meta.get("mstop"),
        "extended": meta.get("extended", False),
        "at_boundary": meta.get("at_boundary", False),
    }
    if mu is not None:
        row["mu_rmse"] = rmse(mu, mu_true)
    if pi is not None:
        row["pi_rmse"] = rmse(pi, pi_true)
    if "converged" in meta:
        row["converged"] = meta["converged"]
    return row


def run_task(task, seed):
    train, _mu_tr, _pi_tr = simulate(10_000 + seed, N_TRAIN, task)
    test, mu_true, pi_true = simulate(80_000 + seed, N_TEST, task)
    y = test["y"].to_numpy()
    rows = result_rows()

    oracle_pi = pi_true
    rows.append(score_row(
        task, seed, "generating ZIP",
        nll=mean_nll(zip_logp(y, mu_true, oracle_pi)),
        expectation=(1.0 - oracle_pi) * mu_true,
        p0=p0_zip(mu_true, oracle_pi),
        mu=mu_true, pi=oracle_pi,
        meta={"seconds": 0.0, "mstop": None, "extended": False, "at_boundary": False},
        y=y, mu_true=mu_true, pi_true=pi_true,
    ))

    sm = fit_statsmodels(task, train, test)
    if task == "linear" and sm["pi"] is not None and rmse(sm["pi"], pi_true) > 0.15:
        raise RuntimeError(
            f"statsmodels structural-zero RMSE is {rmse(sm['pi'], pi_true):.3f}; "
            "prob-main may not be 1 - pi")
    rows.append(score_row(
        task, seed, "statsmodels ZIP MLE",
        nll=mean_nll(zip_logp(y, sm["mu"], sm["pi"])),
        expectation=sm["mean"],
        p0=p0_zip(sm["mu"], sm["pi"]),
        mu=sm["mu"], pi=sm["pi"], meta=sm,
        y=y, mu_true=mu_true, pi_true=pi_true,
    ))

    if task == "linear":
        def zip_fit(mstop):
            return mb.glmboostLSS(
                formula(task, "glm"), train, family=mb.ZIPoLSS(),
                control=mb.boost_control(mstop, NU))

        def pois_fit(mstop):
            return mb.glmboost(
                formula(task, "glm"), train, family=mb.Poisson(),
                control=mb.boost_control(mstop, NU))

        def nb_fit(mstop):
            return mb.glmboost(
                formula(task, "glm"), train, family=mb.NBinomial(),
                control=mb.boost_control(mstop, NU))

        def gate_fit(mstop, data):
            return mb.glmboost(
                "hit ~ x1 + x2 + z + noise", data, family=mb.Binomial(),
                control=mb.boost_control(mstop, NU))

        def count_fit(mstop, data):
            return mb.glmboost(
                formula(task, "glm"), data, family=mb.Hurdle(),
                control=mb.boost_control(mstop, NU))
    else:
        def zip_fit(mstop):
            return mb.gamboostLSS(
                formula(task, "gam"), train, family=mb.ZIPoLSS(),
                control=mb.boost_control(mstop, NU))

        def pois_fit(mstop):
            return mb.gamboost(
                formula(task, "gam"), train, family=mb.Poisson(),
                control=mb.boost_control(mstop, NU))

        def nb_fit(mstop):
            return mb.gamboost(
                formula(task, "gam"), train, family=mb.NBinomial(),
                control=mb.boost_control(mstop, NU))

        def gate_fit(mstop, data):
            return mb.gamboost(
                "hit ~ bbs(x1, df=4, knots=10) + bols(x2) + bols(z) + bols(noise)",
                data, family=mb.Binomial(), control=mb.boost_control(mstop, NU))

        def count_fit(mstop, data):
            return mb.gamboost(
                formula(task, "gam"), data, family=mb.Hurdle(),
                control=mb.boost_control(mstop, NU))

    zip_model, zip_meta = tune_lss(zip_fit, seed)
    params = zip_model.predict(test, type="response", mstop=zip_meta["predict_mstop"])
    mixture = np.asarray(zip_model._family.mean(params), float)
    mean_api = np.asarray(
        zip_model.predict(test, type="mean", mstop=zip_meta["predict_mstop"]), float)
    if not np.allclose(mixture, mean_api):
        raise RuntimeError("ZIPoLSS predict(type='mean') disagrees with family.mean")
    # In-sample density must be the risk the fit minimises, or the test NLL
    # is scoring a different distribution than the one that was fitted.
    train_params = zip_model.predict(type="response", mstop=zip_meta["predict_mstop"])
    train_nll = mean_nll(zip_logp(train["y"].to_numpy(), train_params["mu"], train_params["sigma"]))
    risk_nll = float(np.asarray(zip_model.risk_)[int(np.max(zip_meta["predict_mstop"]))] / N_TRAIN)
    # Cyclic risk is recorded after each full cycle; index mstop is the risk
    # at that per-parameter budget. A mismatch larger than 1e-4 means the
    # scoring pmf is not the fitted pmf.
    if abs(train_nll - risk_nll) > 1e-3:
        raise RuntimeError(
            f"ZIP scoring NLL {train_nll:.6f} disagrees with fit risk {risk_nll:.6f}")
    row = score_row(
        task, seed, "mboostrs ZIPoLSS",
        nll=mean_nll(zip_logp(y, params["mu"], params["sigma"])),
        expectation=mixture,
        p0=p0_zip(params["mu"], params["sigma"]),
        mu=params["mu"], pi=params["sigma"], meta=zip_meta,
        y=y, mu_true=mu_true, pi_true=pi_true,
    )
    row["mae_if_mu_used_as_mean"] = mae(y, params["mu"])
    rows.append(row)
    if task == "linear" and seed == SEEDS[0]:
        coefficient_note = _coefficient_note(zip_model, zip_meta["predict_mstop"])
    else:
        coefficient_note = None

    pois, pois_meta = tune_standard(pois_fit, seed)
    mu_hat = np.asarray(pois.predict(test, type="response"), float)
    train_mu = np.asarray(pois.predict(type="response"), float)
    train_nll = mean_nll(pois_logp(train["y"].to_numpy(), train_mu))
    risk_nll = float(pois.risk_[pois.mstop] / N_TRAIN)
    if abs(train_nll - risk_nll) > 1e-3:
        raise RuntimeError(
            f"Poisson scoring NLL {train_nll:.6f} disagrees with fit risk {risk_nll:.6f}")
    rows.append(score_row(
        task, seed, "mboostrs Poisson",
        nll=mean_nll(pois_logp(y, mu_hat)),
        expectation=mu_hat, p0=p0_pois(mu_hat),
        mu=mu_hat, pi=None, meta=pois_meta,
        y=y, mu_true=mu_true, pi_true=pi_true,
    ))

    nb, nb_meta = tune_standard(nb_fit, seed)
    mu_hat = np.asarray(nb.predict(test, type="response"), float)
    sigma = float(np.asarray(nb.nuisance()).ravel()[0])
    train_mu = np.asarray(nb.predict(type="response"), float)
    train_nll = mean_nll(nb_logp(train["y"].to_numpy(), train_mu, sigma))
    risk_nll = float(nb.risk_[nb.mstop] / N_TRAIN)
    if abs(train_nll - risk_nll) > 1e-3:
        raise RuntimeError(
            f"NBinomial scoring NLL {train_nll:.6f} disagrees with fit risk "
            f"{risk_nll:.6f} at sigma={sigma}")
    rows.append(score_row(
        task, seed, "mboostrs NBinomial",
        nll=mean_nll(nb_logp(y, mu_hat, sigma)),
        expectation=mu_hat, p0=p0_nb(mu_hat, sigma),
        mu=mu_hat, pi=None, meta=nb_meta,
        y=y, mu_true=mu_true, pi_true=pi_true,
    ))

    hit = train.with_columns((pl.col("y") > 0).cast(pl.Float64).alias("hit"))
    pos = positive_frame(train, "y")
    started = time.perf_counter()
    gate, gate_meta = tune_standard(lambda mstop: gate_fit(mstop, hit), seed)
    count, count_meta = tune_standard(lambda mstop: count_fit(mstop, pos), seed)
    seconds = time.perf_counter() - started
    # tune_standard already counted its own time; the sum is the hurdle cost.
    seconds = gate_meta["seconds"] + count_meta["seconds"]
    p_pos = np.asarray(gate.predict(test, type="response"), float)
    mu_hat = np.asarray(count.predict(test, type="response"), float)
    sigma = float(np.asarray(count.nuisance()).ravel()[0])
    # Hurdle sigma is dispersion; size is its reciprocal. Checked below
    # against the fit's own risk on the positive rows.
    size = 1.0 / max(sigma, 1e-8)
    train_mu = np.asarray(count.predict(type="response"), float)
    train_y = pos["y"].to_numpy()
    # The hurdle family's risk is the truncated NLL, not the two-part NLL.
    from scipy.special import gammaln as _gammaln
    _mu = _clip_mu(train_mu)
    _p = size / (size + _mu)
    _log_nb = (_gammaln(train_y + size) - _gammaln(size) - _gammaln(train_y + 1.0)
               + size * np.log(_p) + train_y * np.log(1.0 - _p))
    _log_p0 = size * (np.log(size) - np.log(size + _mu))
    _log_trunc = _log_nb - np.log(np.clip(1.0 - np.exp(_log_p0), 1e-300, None))
    trunc_nll = float(-np.mean(_log_trunc))
    risk_nll = float(count.risk_[count.mstop] / pos.height)
    if abs(trunc_nll - risk_nll) > 1e-3:
        raise RuntimeError(
            f"Hurdle truncated NLL {trunc_nll:.6f} disagrees with fit risk {risk_nll:.6f}")
    nb_zero = np.exp(size * (np.log(size) - np.log(size + _clip_mu(mu_hat))))
    # Truncated negative-binomial mean is mu / (1 - P(Y=0)).
    expectation = p_pos * mu_hat / np.clip(1.0 - nb_zero, 1e-12, None)
    rows.append(score_row(
        task, seed, "mboostrs hurdle (Binomial + Hurdle)",
        nll=mean_nll(hurdle_logp(y, p_pos, mu_hat, sigma)),
        expectation=expectation,
        p0=1.0 - p_pos,
        mu=None, pi=None,
        meta={"seconds": seconds, "mstop": {"gate": gate_meta["mstop"], "count": count_meta["mstop"]},
              "extended": gate_meta["extended"] or count_meta["extended"],
              "at_boundary": gate_meta["at_boundary"] or count_meta["at_boundary"]},
        y=y, mu_true=mu_true, pi_true=pi_true,
    ))
    return rows, coefficient_note


def _coefficient_note(model, mstop):
    """What ``coef()`` reports for the inflation intercept, next to the offset.

    ``coef()`` has no ``mstop`` argument, so this is the intercept at the
    budget the model was fitted to (the cap or the extension), not a sliced one.
    """
    del mstop
    coef = model.coef()
    sigma_intercept = float(coef["sigma"]["(Intercept)"])
    mu_intercept = float(coef["mu"]["(Intercept)"])
    sigma_offset = float(np.asarray(model._params[model.param_names.index("sigma")]["offset"]).ravel()[0])
    mu_offset = float(np.asarray(model._params[model.param_names.index("mu")]["offset"]).ravel()[0])
    return {
        "sigma_intercept_coef": sigma_intercept,
        "sigma_offset": sigma_offset,
        "sigma_intercept_plus_offset": sigma_intercept + sigma_offset,
        "sigma_intercept_truth": -0.35,
        "mu_intercept_coef": mu_intercept,
        "mu_offset": mu_offset,
        "mu_intercept_plus_offset": mu_intercept + mu_offset,
        "mu_intercept_truth": 0.15,
        "coef_doc_first_line": (model.coef.__doc__ or "").strip().splitlines()[0],
    }


def _call(label, fn):
    try:
        value = fn()
        return {"label": label, "outcome": "ok", "detail": repr(value)[:500]}
    except Exception as exc:
        return {"label": label, "outcome": type(exc).__name__, "detail": str(exc)[:800]}


def usability(train):
    """Calls an agent makes from the class name, and what comes back."""
    probes = [
        _call(
            "gamboost(family=ZIPoLSS())",
            lambda: mb.gamboost("y ~ bols(x1) + bols(z)", train, family=mb.ZIPoLSS(),
                                control=mb.boost_control(5, 0.1)),
        ),
        _call(
            "glmboost(family=ZIPoLSS())",
            lambda: mb.glmboost("y ~ x1 + z", train, family=mb.ZIPoLSS(),
                                control=mb.boost_control(5, 0.1)),
        ),
        _call(
            "gamboostLSS(family=Poisson())",
            lambda: mb.gamboostLSS("y ~ bols(x1)", train, family=mb.Poisson(),
                                   control=mb.boost_control(5, 0.1)),
        ),
        _call(
            "glmboostLSS(family=Poisson())",
            lambda: mb.glmboostLSS("y ~ x1", train, family=mb.Poisson(),
                                   control=mb.boost_control(5, 0.1)),
        ),
    ]
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = mb.autospec(train, "y", budget=80, mstop=80, seed=0)
    autospec_seconds = time.perf_counter() - started
    report = record.report()
    family_name = type(record.family).__name__
    scan_started = time.perf_counter()
    ranked = mb.scan_lss_families(
        "y ~ x1 + x2 + z + noise", train, support="counts", metric="cvrisk",
        fit_func="glmboostLSS", control=mb.boost_control(80, 0.1),
    )
    scan_seconds = time.perf_counter() - scan_started
    def _num(value):
        return None if value is None else float(value)

    scan_top = [
        {"family": row.get("family"), "status": row.get("status"),
         "cvrisk": _num(row.get("cvrisk")), "risk": _num(row.get("risk")),
         "error": row.get("error")}
        for row in ranked[:8]
    ]
    return {
        "probes": probes,
        "autospec_seconds": autospec_seconds,
        "scan_lss_seconds": scan_seconds,
        "scan_lss_top": scan_top,
        "autospec_family": family_name,
        "autospec_formula": record.formula,
        "autospec_mentions_zip": "ZIPo" in report or "zero-inflat" in report.lower(),
        "autospec_warnings": [str(item.message) for item in caught],
        "autospec_report_excerpt": report[:1200],
        "zip_doc_first_paragraph": (mb.ZIPoLSS.__doc__ or "").strip().split("\n\n")[0],
        "hurdle_doc_excerpt": " ".join((mb.Hurdle.__doc__ or "").split()),
    }


from benches._quality_provenance import attach_provenance, begin_run, result_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE.parent / "docs/book/cases/bench_zero_inflated.json")
    parser.add_argument("--tasks", nargs="+", default=["linear", "smooth"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    begin_run(__file__, simulate, N_TRAIN, N_TEST, ("linear", "smooth"),
              train_seed_offset=10_000, test_seed_offset=80_000, test_shared=False)
    # Smooth fits warn when a few test rows sit outside the training range of
    # bbs(x1), and when a long path spends the spline's penalty. Both are
    # recorded in the case study; they are not part of the score table.
    warnings.filterwarnings("ignore", message=".*outside the range they were fitted on.*")
    warnings.filterwarnings("ignore", message=".*undone much of the penalty.*")
    rows = result_rows()
    coefficient_note = None
    for task in args.tasks:
        for seed in args.seeds:
            print(f"task={task} seed={seed}", flush=True)
            task_rows, note = run_task(task, seed)
            rows.extend(task_rows)
            if note is not None:
                coefficient_note = note
            for row in task_rows:
                if row["method"] == "generating ZIP":
                    continue
                print(
                    f"  {row['method']:36s} nll={row['nll']:.4f} mae={row['mae']:.4f} "
                    f"p0={row['p0_rmse']:.4f} sec={row['seconds']:.2f} mstop={row['mstop']}",
                    flush=True,
                )
    train, _, _ = simulate(10_000, N_TRAIN, "linear")
    print("usability", flush=True)
    use = usability(train)
    print("autospec family", use["autospec_family"], flush=True)
    document = {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "seeds": list(args.seeds),
        "nu": NU,
        "folds": FOLDS,
        "mstop_cap": MSTOP_CAP,
        "mstop_extend": MSTOP_EXTEND,
        "tasks": {
            "linear": "log mu = 0.15 + 0.55 x1 - 0.30 x2; logit pi = -0.35 + 0.95 z",
            "smooth": "log mu = 0.05 + sin(1.5 x1) + 0.30 x2; logit pi = -0.35 + 0.95 z",
        },
        "competitors_not_run": [
            "R gamboostLSS::ZIPoLSS is not installed. statsmodels ZeroInflatedPoisson "
            "is the maximum-likelihood reference for the same design matrix."
        ],
        "results": rows,
        "coefficient_reading": coefficient_note,
        "usability": use,
    }
    attach_provenance(document)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
