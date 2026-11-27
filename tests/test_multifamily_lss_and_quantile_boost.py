"""Tests for Milestone 3 (C6 multi-family noncyclic Stage 3 LSS challenger & C7-Python quantile_boost)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._rust import crps_normal
from mboostrs.distribution_evaluation import _crps


def _gaussian_oob_baseline_scores(train, test, formula, mstop, nu=0.3):
    """Fit a constant-scale Gaussian model on train and evaluate held-out CRPS and NLL on test."""
    m = mb.gamboost(formula, data=train, family=mb.Gaussian(), control=mb.boost_control(mstop, nu))
    res_tr = np.asarray(train["y"], dtype=float) - np.asarray(m.predict(train, type="response"), dtype=float)
    sd = max(float(np.std(res_tr)), 1e-8)
    mu_te = np.asarray(m.predict(test, type="response"), dtype=float)
    y_te = np.asarray(test["y"], dtype=float)
    nll = float(np.mean(0.5 * (math.log(2.0 * math.pi * sd * sd) + ((y_te - mu_te) / sd) ** 2)))
    crps = float(np.mean(np.asarray(crps_normal(y_te, mu_te, np.full_like(y_te, sd)), dtype=float)))
    return crps, nll


# ─── C6: Multi-Family Noncyclic Stage 3 Distributional Challenger ────────────


def test_c6_student_t_adopts_studenttlss_and_beats_gaussian_crps_and_logscore():
    rng = np.random.default_rng(101)
    n_train, n_test = 700, 400
    n = n_train + n_test
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-2.0, 2.0, size=n)
    x3 = rng.normal(size=n)
    mu_true = 2.0 * np.sin(x1) + 1.2 * x2
    y = mu_true + 0.7 * rng.standard_t(3.0, size=n)

    train = {"x1": x1[:n_train], "x2": x2[:n_train], "x3": x3[:n_train], "y": y[:n_train]}
    test = {"x1": x1[n_train:], "x2": x2[n_train:], "x3": x3[n_train:], "y": y[n_train:]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.fit(train, "y", seed=0)

    assert isinstance(fit.model._family, mb.StudentTLSS)
    assert fit.model._method == "noncyclic"

    preds = fit.predict(test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (n_test,)
    assert np.all(np.isfinite(preds))

    params_te = fit.model.predict(test, type="response")
    y_te = test["y"]
    lss_crps = float(np.mean(np.asarray(_crps(fit.model._family, y_te, params_te), dtype=float)))
    lss_nll = float(0.5 * np.mean(fit.model._family.deviance_increment(y_te, params_te)))

    g_crps, g_nll = _gaussian_oob_baseline_scores(
        train, test, "y ~ bbs(x1) + bbs(x2) + bols(x3)", mstop=150
    )
    assert lss_crps < g_crps
    assert lss_nll < g_nll


def test_c6_lognormal_adopts_lss_and_predict_plus_interval_work_end_to_end():
    rng = np.random.default_rng(202)
    n_train, n_test = 700, 400
    n = n_train + n_test
    x1 = rng.uniform(-1.5, 1.5, size=n)
    x2 = rng.uniform(-1.5, 1.5, size=n)
    x3 = rng.normal(size=n)
    log_mu = 0.6 * np.sin(x1) + 0.4 * x2
    sigma = 0.35 + 0.25 * np.abs(x1)
    y = np.exp(log_mu + sigma * rng.normal(size=n))

    train = {"x1": x1[:n_train], "x2": x2[:n_train], "x3": x3[:n_train], "y": y[:n_train]}
    test = {"x1": x1[n_train:], "x2": x2[n_train:], "x3": x3[n_train:], "y": y[n_train:]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.fit(train, "y", seed=0)

    assert isinstance(fit.model._family, (mb.LogNormalLSS, mb.GaussianLSS))
    assert fit.model._method == "noncyclic"

    preds = fit.predict(test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (n_test,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0.0)

    ci = fit.predict_interval(test, level=0.90)
    assert ci.lower.shape == (n_test,)
    assert ci.upper.shape == (n_test,)
    assert np.all(np.isfinite(ci.lower))
    assert np.all(np.isfinite(ci.upper))
    assert np.all(ci.upper >= ci.lower)


def test_c6_heteroscedastic_s3_recovers_scale_only_covariate():
    rng = np.random.default_rng(303)
    n_train, n_test = 800, 200
    n = n_train + n_test
    x0 = rng.uniform(-1.5, 1.5, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    x2 = rng.uniform(-1.5, 1.5, size=n)  # scale-only covariate
    x3 = rng.uniform(-1.5, 1.5, size=n)  # noise
    sigma = np.exp(-0.5 + 0.9 * x2)
    y = 1.5 * np.sin(2.0 * x0) + 0.8 * x1 + sigma * rng.normal(size=n)

    train = {"x0": x0[:n_train], "x1": x1[:n_train], "x2": x2[:n_train], "x3": x3[:n_train], "y": y[:n_train]}
    test = {"x0": x0[n_train:], "x1": x1[n_train:], "x2": x2[n_train:], "x3": x3[n_train:], "y": y[n_train:]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.fit(train, "y", seed=0)

    assert isinstance(fit.model._family, mb.GaussianLSS)
    assert fit.model._method == "noncyclic"
    sigma_formula = fit.model._formula["sigma"]
    assert "x2" in sigma_formula
    preds = fit.predict(test)
    assert preds.shape == (n_test,)
    assert np.all(np.isfinite(preds))


# ─── C7-Python: MultiQuantileModel & mb.quantile_boost ───────────────────────


def test_c7_quantile_boost_rearrangement_crps_cqr_and_missing_unseen():
    rng = np.random.default_rng(404)
    n_train, n_cal, n_test = 600, 300, 400
    n = n_train + n_cal + n_test
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2 = rng.uniform(-2.0, 2.0, size=n)
    # Skewed heteroscedastic response (exponential noise scaled by x1, x2)
    loc = 1.5 * np.sin(x1) + 0.7 * x2 + 0.8 * (x1 * x2 > 0).astype(float)
    scale = 0.4 + 0.4 * np.abs(x1)
    y = loc + scale * (rng.exponential(1.0, size=n) - 1.0)

    train = {"x1": x1[:n_train], "x2": x2[:n_train], "y": y[:n_train]}
    cal = {"x1": x1[n_train : n_train + n_cal], "x2": x2[n_train : n_train + n_cal], "y": y[n_train : n_train + n_cal]}
    test = {"x1": x1[n_train + n_cal :], "x2": x2[n_train + n_cal :], "y": y[n_train + n_cal :]}

    taus = [0.05, 0.25, 0.5, 0.75, 0.95]
    mq = mb.quantile_boost(
        "y ~ bbs(x1) + bbs(x2) + btree(x1, x2, maxdepth=3, split_method=histogram)",
        train,
        quantiles=taus,
        folds=5,
        max_mstop=300,
        seed=0,
    )
    assert isinstance(mq, mb.MultiQuantileModel)
    assert mq.quantiles == tuple(taus)
    assert set(mq.models.keys()) == set(taus)
    assert all(1 <= mq.optimal_mstop[t] <= 300 for t in taus)

    # Rearranged predictions have zero quantile crossings
    rearr = mq.predict(test, rearrange=True)
    pred_df = rearr.predictions
    q_cols = np.column_stack([np.asarray(pred_df[col], dtype=float) for col in pred_df.columns])
    assert q_cols.shape == (n_test, len(taus))
    assert np.all(np.diff(q_cols, axis=1) >= -1e-12)

    # Raw array option when rearrange=False
    raw_arr = mq.predict(test, rearrange=False)
    assert isinstance(raw_arr, np.ndarray)
    assert raw_arr.shape == (n_test, len(taus))

    # Mean prediction and CRPS
    mean_pred = mq.predict_mean(test)
    assert mean_pred.shape == (n_test,)
    assert np.all(np.isfinite(mean_pred))
    crps_val = mq.crps(test)
    assert math.isfinite(crps_val) and crps_val > 0.0
    # CRPS should easily beat a naive constant-median baseline
    med_const = float(np.median(train["y"]))
    naive_crps = float(np.mean(np.abs(test["y"] - med_const))) * 0.5
    assert crps_val < naive_crps

    # CQR calibrated interval achieves ~90% coverage
    iv_cqr = mq.predict_interval(test, level=0.90, calibration=cal)
    assert iv_cqr["method"] == "cqr"
    assert iv_cqr["level"] == pytest.approx(0.90)
    cov = float(np.mean((test["y"] >= iv_cqr["lower"]) & (test["y"] <= iv_cqr["upper"])))
    assert 0.84 <= cov <= 0.96, f"Expected ~0.90 CQR coverage, got {cov:.3f}"
    assert np.all(iv_cqr["upper"] >= iv_cqr["lower"])

    # Uncalibrated quantile interval
    iv_q = mq.predict_interval(test, level=0.90)
    assert iv_q["method"] == "quantile"
    assert iv_q["cqr_adjustment"] == 0.0

    # Serialization / repr
    d = mq.to_dict()
    assert d["quantiles"] == taus
    assert "MultiQuantileModel" in repr(mq)


def test_c7_quantile_boost_handles_missing_values_and_unseen_categoricals():
    rng = np.random.default_rng(505)
    n_train, n_test = 300, 60
    x1 = rng.normal(size=n_train)
    x2 = rng.normal(size=n_train)
    x2[:25] = np.nan
    cat = rng.choice(["a", "b", "c"], size=n_train).astype(object)
    cat[:10] = None
    y = 1.5 * x1 + 0.5 * np.nan_to_num(x2) + np.where(cat == "a", 1.0, 0.0) + rng.normal(0.0, 0.5, size=n_train)
    train = {"x1": x1, "x2": x2, "cat": cat, "y": y}

    x1_te = rng.normal(size=n_test)
    x2_te = rng.normal(size=n_test)
    x2_te[:10] = np.nan
    cat_te = rng.choice(["a", "b", "unseen_z", None], size=n_test)
    test = {"x1": x1_te, "x2": x2_te, "cat": cat_te}

    mq = mb.quantile_boost(
        "y ~ bbs(x1) + bbs(x2) + brandom(cat)",
        train,
        quantiles=(0.1, 0.5, 0.9),
        folds=3,
        max_mstop=120,
        na_action="impute",
        seed=0,
    )
    res = mq.predict(test, unseen_levels="average")
    assert res.n == n_test
    iv = mq.predict_interval(test, level=0.80, unseen_levels="average")
    assert np.all(np.isfinite(iv["lower"]))
    assert np.all(np.isfinite(iv["upper"]))
    assert np.all(np.isfinite(mq.predict_mean(test, unseen_levels="average")))
