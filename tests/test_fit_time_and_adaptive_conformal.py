"""Tests for Milestone 1 (C5 bounded-integer measurement typing + C8 time/uncertainty/adaptive conformal)."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


# ─── C5: Bounded-integer measurement vs sparse grouping codes ────────────────


def test_c5_early_stage_diabetes_age_is_not_misclassified_as_labels_or_groups():
    """Integer Age 16..90 with noisy cohort prevalence stays continuous/bbs, not integer-labels."""
    rng = np.random.default_rng(17)
    n = 520
    # Non-zero-origin dense integer span (16..90), ~51 distinct values over span 75.
    age = rng.integers(16, 91, size=n)
    # Deliberately add noisy level-wise jitter so naive ANOVA F is high,
    # mimicking cohort prevalence noise in Early Stage Diabetes.
    level_noise = rng.normal(0.0, 0.8, size=91)[age]
    polyuria = rng.integers(0, 2, size=n)
    y = ((0.03 * (age - 45) + 1.5 * polyuria + level_noise + rng.normal(0.0, 0.3, n)) > 1.0).astype(float)
    df = pl.DataFrame({"Age": age, "Polyuria": polyuria, "class": y})

    kinds = {c.name: c for c in mb.classify_columns(df, response="class")}
    assert kinds["Age"].kind == "continuous"
    assert kinds["Age"].term.startswith("bbs(Age")

    ev = mb.label_evidence("Age", age.astype(float), y)
    assert ev is not None
    assert ev.labels is False

    rec = mb.autospec(df, "class", budget=40, seed=0)
    assert not any(h.column == "Age" and h.kind == "integer-labels" for h in rec.hygiene)
    assert "grouped folds" not in rec.folds_description
    assert "brandom(Age" not in rec.formula


def test_c5_dense_unnamed_measurement_span_and_explicit_code_tokens():
    """Unnamed dense span (>12 levels, low>1, span<=200, >=50% density) is measurement;
    explicit code names and sparse codes remain categorical/grouped."""
    rng = np.random.default_rng(29)
    n = 800
    # Unnamed measurement column spanning 20..80 (61 levels, span 60, density ~1.0)
    x_meas = rng.integers(20, 81, size=n)
    # Explicit code name even though values are dense 10..49
    site_id = rng.integers(10, 50, size=n)
    # Sparse district codes 0, 3, ..., 117 (40 levels, span 117, density 0.34)
    idx = rng.integers(0, 40, size=n)
    g = (idx * 3).astype(float)
    y = (
        0.05 * x_meas
        + rng.normal(0.0, 1.5, size=50)[site_id]
        + rng.normal(0.0, 1.5, size=40)[idx]
        + rng.normal(0.0, 0.4, size=n)
    )
    df = pl.DataFrame({"x_meas": x_meas, "site_id": site_id, "g": g, "y": y})

    kinds = {c.name: c for c in mb.classify_columns(df, response="y")}
    assert kinds["x_meas"].kind == "continuous"
    assert kinds["x_meas"].term.startswith("bbs(x_meas")
    assert kinds["site_id"].kind == "categorical"
    assert kinds["site_id"].term.startswith("brandom(site_id")
    assert kinds["g"].kind == "categorical"
    assert kinds["g"].term.startswith("brandom(g")

    # Sparse code column triggers integer-labels + grouped folds and fast autospec preserves brandom(g)
    rec = mb.autospec(df.select(["g", "x_meas", "y"]), "y", budget=40, search="fast", seed=0)
    assert any(h.column == "g" and h.kind == "integer-labels" for h in rec.hygiene)
    assert "grouped folds" in rec.folds_description
    assert "brandom(g" in rec.formula


# ─── C8: mb.fit(time=..., uncertainty=...) + adaptive conformal ──────────────


def test_c8_fit_time_and_uncertainty_conformal_predict_interval():
    rng = np.random.default_rng(101)
    n = 300
    t = np.arange(n, dtype=float)
    x = rng.normal(size=n)
    y = 1.5 * x + 0.02 * t + rng.normal(0.0, 0.5, size=n)
    df = pl.DataFrame({"t": t, "x": x, "y": y})

    fitted = mb.fit(df, "y", time="t", uncertainty="conformal", seed=0)
    assert fitted.trail.validate()
    trail_dict = fitted.trail.to_dict()
    assert "rolling" in trail_dict["folds"]["scheme"].lower()
    assert trail_dict["uncertainty"]["method"] == "conformal"
    assert fitted.conformal is not None

    test_df = pl.DataFrame({
        "t": np.arange(n, n + 40, dtype=float),
        "x": rng.normal(size=40),
    })
    ci = fitted.predict_interval(test_df, level=0.8)
    assert ci.method == "adaptive"
    assert ci.level == 0.8
    assert ci.lower.shape == (40,)
    assert ci.reliability is not None
    assert any("Barber" in a for a in ci.assumptions)


def test_c8_time_validation_and_fit_uncertainty_stability_bootstrap_and_binary():
    rng = np.random.default_rng(202)
    n = 220
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 * x1 - 0.8 * x2 + rng.normal(0.0, 0.4, size=n)
    data = {"x1": x1, "x2": x2, "y": y}

    with pytest.raises(ValueError, match="not found"):
        mb.fit(data, "y", time="missing_col", seed=0)
    with pytest.raises(ValueError, match="response"):
        mb.fit(data, "y", time="y", seed=0)

    fit_stab = mb.fit(data, "y", uncertainty="stability", seed=0)
    assert fit_stab.trail.validate()
    assert fit_stab.stability is not None
    assert fit_stab.trail.to_dict()["uncertainty"]["method"] == "stability"

    fit_boot = mb.fit(data, "y", uncertainty="bootstrap", seed=0)
    assert fit_boot.trail.validate()
    assert fit_boot.bootstrap is not None
    assert fit_boot.trail.to_dict()["uncertainty"]["method"] == "bootstrap"

    # Binary classification with uncertainty="conformal" computes ECE cleanly without crashing
    p = 1.0 / (1.0 + np.exp(-(1.2 * x1 - 0.6 * x2)))
    y_bin = rng.binomial(1, p).astype(float)
    fit_bin = mb.fit({"x1": x1, "x2": x2, "y": y_bin}, "y", uncertainty="conformal", seed=0)
    assert fit_bin.trail.validate()
    assert isinstance(fit_bin.conformal, dict)
    assert fit_bin.conformal["kind"] == "classification_calibration"
    assert 0.0 <= fit_bin.conformal["ece"] <= 1.0


def test_c8_adaptive_conformal_and_stream_improve_winter_coverage_under_scale_shift():
    """Low-noise summer -> 3x-noise winter regime shift (Beijing PM2.5 style)."""
    rng = np.random.default_rng(77)
    n_summer = 400
    n_winter_cal = 60
    n_winter_test = 400
    n_total = n_summer + n_winter_cal + n_winter_test

    t = np.arange(n_total, dtype=float)
    x = rng.normal(size=n_total)
    noise_sd = np.concatenate([
        np.full(n_summer, 0.5),
        np.full(n_winter_cal + n_winter_test, 1.5),
    ])
    y = 2.0 * x + np.sin(x) + rng.normal(0.0, noise_sd)

    n_train_cal = n_summer + n_winter_cal
    train_data = {"t": t[:n_train_cal], "x": x[:n_train_cal], "y": y[:n_train_cal]}
    test_data = {"t": t[n_train_cal:], "x": x[n_train_cal:], "y": y[n_train_cal:]}

    model = mb.gamboost(
        "y ~ bbs(x, df=4)",
        data=train_data,
        family=mb.Gaussian(),
        control=mb.boost_control(mstop=120, nu=0.1),
    )

    level = 0.80
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci_split = mb.conformal_interval(model, test_data, level=level, method="split", seed=0)

    with pytest.warns(mb.ConformalReliabilityWarning, match="temporal_scale_shift"):
        ci_adaptive = mb.conformal_interval(
            model, test_data, level=level, method="adaptive", calib_fraction=0.25, seed=0
        )

    assert ci_adaptive.reliability is not None
    assert ci_adaptive.reliability["status"] == "caution"
    assert any("temporal_scale_shift" in w for w in ci_adaptive.reliability["warnings"])

    cov_split = ci_split.empirical_coverage(test_data["y"])
    cov_adaptive = ci_adaptive.empirical_coverage(test_data["y"])

    stream_cold = mb.adaptive_conformal_stream(
        ci_split.predicted,
        test_data["y"],
        ci_split.width / 2.0,
        level=level,
        lr=0.05,
    )
    stream_warm = mb.adaptive_conformal_stream(
        ci_adaptive.predicted,
        test_data["y"],
        ci_adaptive.width / 2.0,
        level=level,
        lr=0.02,
    )
    assert stream_cold.lower.shape == (n_winter_test,)
    assert stream_cold.upper.shape == (n_winter_test,)
    assert stream_cold.alpha_path.shape == (n_winter_test,)
    assert stream_cold.multipliers.shape == (n_winter_test,)

    # Both Barber et al. weighted adaptive conformal and Gibbs & Candes ACI stream
    # substantially recover nominal 0.80 winter coverage compared to exchangeable split.
    assert cov_split < 0.50
    assert cov_adaptive > cov_split + 0.30
    assert stream_cold.coverage > cov_split + 0.25
    assert 0.72 <= stream_warm.coverage <= 0.88


def test_c8_adaptive_conformal_local_score_and_lss_and_held_out_calibration():
    rng = np.random.default_rng(88)
    n = 260
    x = rng.normal(size=n)
    sigma_true = 0.3 + 0.6 * np.abs(x)
    y = 1.5 * x + rng.normal(0.0, sigma_true)
    train = {"x": x[:200], "y": y[:200]}
    cal = {"x": x[200:230], "y": y[200:230]}
    test = {"x": x[230:], "y": y[230:]}

    m_gam = mb.gamboost("y ~ bbs(x, df=4)", data=train, control=mb.boost_control(60, 0.1))
    ci_local = mb.conformal_interval(m_gam, test, level=0.8, method="adaptive", score="local")
    assert ci_local.method == "adaptive"
    assert ci_local.score == "local"
    assert ci_local.lower.shape == (30,)

    ci_cal = mb.conformal_interval(m_gam, test, level=0.8, method="adaptive", calibration=cal)
    assert ci_cal.method == "adaptive"
    assert ci_cal.n_calibration == 30

    m_lss = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bbs(x, df=4)"},
        data=train,
        family=mb.GaussianLSS(),
        control=mb.boost_control(40, 0.1),
    )
    ci_lss = mb.conformal_interval(m_lss, test, level=0.8, method="adaptive")
    assert ci_lss.method == "adaptive"
    assert ci_lss.score == "lss_standardized"
    assert np.all(np.isfinite(ci_lss.lower)) and np.all(np.isfinite(ci_lss.upper))
