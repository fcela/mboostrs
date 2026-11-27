"""Tests for Milestone 4 (C9 + C10): calibrated DR/orthogonal CATE intervals,
estimand='ato' overlap weights + thin-overlap companion, .digest(), and
multi-topic `python -m mboostrs guide` CLI.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.families import Gaussian


def _smooth_dgp(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x0, x1, x2 = rng.uniform(-1.5, 1.5, (3, n))
    e = 1.0 / (1.0 + np.exp(-(0.9 * x0 - 0.7 * x1)))
    t = rng.binomial(1, e).astype(float)
    tau = 1.5 * np.sin(2.0 * x0) + 0.8 * x1
    m0 = x0**2 + x2
    y = m0 + tau * t + rng.normal(0.0, 0.8, n)
    return {"t": t, "y": y, "x0": x0, "x1": x1, "x2": x2}, tau


def _joint_dgp(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x0, x1, x2 = rng.uniform(-1.5, 1.5, (3, n))
    e = 1.0 / (1.0 + np.exp(-(0.6 * x0 - 0.5 * x1)))
    t = rng.binomial(1, e).astype(float)
    tau = 2.0 * ((x0 > 0.0) & (x1 > 0.0)).astype(float)
    m0 = np.sin(x0) + 0.5 * x1 + 0.3 * x2
    y = m0 + tau * t + rng.normal(0.0, 0.7, n)
    return {"t": t, "y": y, "x0": x0, "x1": x1, "x2": x2}, tau


def _spec(estimand: str = "ate") -> mb.CausalSpec:
    return mb.CausalSpec(
        treatment="t",
        outcome="y",
        covariates=("x0", "x1", "x2"),
        estimand=estimand,
        time_zero="assignment",
        follow_up="1 year",
        adjustment_rationale="x0, x1, x2 are baseline confounders",
    )


def test_smooth_dr_orthogonal_monte_carlo_coverage_90():
    """20 Monte Carlo replications of nonlinear CATE achieve ~90% coverage in [0.85, 0.96]."""
    spec = _spec()
    hits = 0
    total = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in range(20):
            train, _ = _smooth_dgp(800, seed=1000 + rep)
            test, tau_test = _smooth_dgp(100, seed=2000 + rep)
            test_x = {k: test[k] for k in ("x0", "x1", "x2")}
            fx = mb.effect_model(
                train, spec, terms="smooth", folds=5, seed=rep,
                interval_method="dr_orthogonal",
            )
            assert fx.interval_method == "dr_orthogonal"
            pred = fx.predict(test_x, interval=True, level=0.90)
            lo = pred["lower"].to_numpy()
            hi = pred["upper"].to_numpy()
            hits += int(np.sum((lo <= tau_test) & (tau_test <= hi)))
            total += tau_test.size
    coverage = hits / total
    assert 0.85 <= coverage <= 0.96, f"smooth 90% coverage={coverage:.4f} outside [0.85, 0.96]"


def test_dr_orthogonal_with_aipw_nuisance_uses_kennedy_dr_pseudo_outcome():
    """Passing nuisance=AIPWResult uses Kennedy (2023) DR pseudo-outcomes and calibrates well."""
    spec = _spec()
    hits = 0
    total = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in range(6):
            train, _ = _smooth_dgp(800, seed=3000 + rep)
            test, tau_test = _smooth_dgp(150, seed=4000 + rep)
            test_x = {k: test[k] for k in ("x0", "x1", "x2")}
            nuis = mb.aipw(train, spec, folds=5, seed=rep, smooth=True, tune=False, mstop=100)
            fx = mb.effect_model(
                train, spec, nuisance=nuis, terms="smooth", seed=rep,
                interval_method="dr_orthogonal",
            )
            pred = fx.predict(test_x, interval=True, level=0.90)
            lo = pred["lower"].to_numpy()
            hi = pred["upper"].to_numpy()
            assert np.all(hi > lo)
            hits += int(np.sum((lo <= tau_test) & (tau_test <= hi)))
            total += tau_test.size
    coverage = hits / total
    assert 0.85 <= coverage <= 0.97, f"DR-learner coverage={coverage:.4f} outside [0.85, 0.97]"


def test_joint_and_stumps_dr_orthogonal_intervals_cover_interactive_cate():
    """Boosted 'joint' (and 'stumps') models get calibrated intervals under dr_orthogonal."""
    spec = _spec()
    hits = 0
    total = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in range(6):
            train, _ = _joint_dgp(1200, seed=5000 + rep)
            test, tau_test = _joint_dgp(250, seed=6000 + rep)
            test_x = {k: test[k] for k in ("x0", "x1", "x2")}
            fx = mb.effect_model(
                train, spec, terms="joint", folds=5, seed=rep,
                interval_method="dr_orthogonal",
            )
            assert fx.interval_method == "dr_orthogonal"
            assert fx.diagnostics["interval"]["dr_draws"] == 40
            pred = fx.predict(test_x, interval=True, level=0.90)
            lo = pred["lower"].to_numpy()
            hi = pred["upper"].to_numpy()
            assert np.all(hi > lo)
            hits += int(np.sum((lo <= tau_test) & (tau_test <= hi)))
            total += tau_test.size
    coverage = hits / total
    assert coverage >= 0.83, f"joint 90% coverage={coverage:.4f} < 0.83"

    # Also verify terms='stumps' works with dr_orthogonal
    train_s, _ = _joint_dgp(600, seed=7001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx_stumps = mb.effect_model(
            train_s, spec, terms="stumps", folds=4, seed=1,
            interval_method="dr_orthogonal",
        )
    pred_s = fx_stumps.predict({k: train_s[k][:20] for k in ("x0", "x1", "x2")},
                               interval=True, level=0.90)
    assert np.all(pred_s["upper"].to_numpy() > pred_s["lower"].to_numpy())


def test_estimand_ato_and_thin_overlap_companion():
    """estimand='ato' uses overlap weights with zero clipping, and estimand='ate' attaches ato_companion."""
    rng = np.random.default_rng(77)
    n = 1500
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Extreme propensity in [~0.002, ~0.998]
    e_true = 1.0 / (1.0 + np.exp(-(3.2 * x0 + 0.5 * x1)))
    t = rng.binomial(1, e_true).astype(float)
    tau = 1.0 + 1.5 * x0
    y = 2.0 * x0 + x1 + 0.5 * x2 + tau * t + rng.normal(0.0, 0.8, n)
    data = {"t": t, "y": y, "x0": x0, "x1": x1, "x2": x2}

    spec_ato = _spec("ato")
    spec_ate = _spec("ate")
    assert "ATO" in spec_ato.describe()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_ato = mb.aipw(data, spec_ato, folds=4, smooth=True, tune=False, mstop=80, seed=3)
        res_ate = mb.aipw(data, spec_ate, folds=4, smooth=True, tune=False, mstop=80, seed=3)

    assert res_ato.diagnostics["target"] == "overlap"
    assert res_ato.diagnostics["propensity_clip_applied"] is None
    assert res_ato.diagnostics["n_clipped_propensity"] == 0
    assert res_ato.ato_companion is None

    # Thin overlap triggers ato_companion on estimand='ate'
    comp = res_ate.ato_companion
    assert isinstance(comp, dict)
    assert set(comp) == {"estimate", "se", "ci_lower", "ci_upper", "outside_overlap_share"}
    assert comp["outside_overlap_share"] >= 0.05
    assert comp["estimate"] == pytest.approx(res_ato.estimate, rel=1e-10)
    assert comp["se"] == pytest.approx(res_ato.standard_error, rel=1e-10)
    assert comp["se"] < res_ate.standard_error
    assert res_ate.to_dict()["ato_companion"] == comp

    # Healthy overlap leaves ato_companion as None
    healthy_data, _ = _smooth_dgp(600, seed=88)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_healthy = mb.aipw(healthy_data, spec_ate, folds=3, tune=False, mstop=40, seed=1)
    assert res_healthy.ato_companion is None


def test_digest_across_all_models_and_top_level():
    """mb.digest() and .digest() produce deterministic <=25-line summary cards."""
    rng = np.random.default_rng(9)
    n = 150
    x1, x2 = rng.normal(size=(2, n))
    y = 2.0 * x1 + np.sin(x2) + rng.normal(0.0, 0.5, n)
    data = {"y": y, "x1": x1, "x2": x2}

    fit_res = mb.fit(data, "y", budget="quick", uncertainty="conformal", seed=0)
    gam = mb.gamboost("y ~ bbs(x1) + bbs(x2)", data=data, family=Gaussian(),
                      control=mb.boost_control(20, 0.1))
    glm = mb.glmboost("y ~ x1 + x2", data=data, family=Gaussian(),
                      control=mb.boost_control(20, 0.1))
    lss = mb.gamboostLSS(
        {"mu": "y ~ bbs(x1) + bbs(x2)", "sigma": "y ~ bbs(x1)"},
        data=data, family=mb.GaussianLSS(), control=mb.boost_control(15, 0.1),
    )
    qm = mb.quantile_boost("y ~ bbs(x1) + bbs(x2)", data=data,
                           quantiles=(0.1, 0.5, 0.9), folds=3, max_mstop=30, seed=0)

    c_data, _ = _smooth_dgp(240, seed=12)
    spec = _spec("ate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aipw_res = mb.aipw(c_data, spec, folds=3, tune=False, mstop=30, seed=1)
        fx_res = mb.effect_model(c_data, spec, terms="smooth", folds=3, seed=1,
                                 interval_method="dr_orthogonal")

    for obj in (fit_res, gam, glm, lss, qm, aipw_res, fx_res):
        card1 = obj.digest()
        card2 = mb.digest(obj)
        assert card1 == card2
        lines = card1.splitlines()
        assert 2 <= len(lines) <= 25
        assert lines[0].startswith("[mboostrs ")

    assert "conformal" in fit_res.digest()
    assert "quantiles=(0.1, 0.5, 0.9)" in qm.digest()
    assert "dr_orthogonal" in fx_res.digest()

    with pytest.raises(TypeError, match="digest"):
        mb.digest({"not": "a model"})


def test_cli_guide_supports_multiple_topics():
    """python -m mboostrs guide <t1> <t2> outputs both topics separated by ---."""
    proc = subprocess.run(
        [sys.executable, "-m", "mboostrs", "guide", "fit-and-tune", "causal"],
        capture_output=True, text=True, check=True,
    )
    assert "\n---\n" in proc.stdout
    assert "tune_mstop" in proc.stdout
    assert "CausalSpec" in proc.stdout

    # Unknown second topic fails cleanly
    bad = subprocess.run(
        [sys.executable, "-m", "mboostrs", "guide", "causal", "no-such-topic"],
        capture_output=True, text=True,
    )
    assert bad.returncode != 0
