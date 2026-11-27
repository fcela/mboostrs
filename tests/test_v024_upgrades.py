"""Regression and feature tests for v0.2.4 upgrades:
- All 4 D6 robustness fixes (LSS+%X% conformal, subsample domain safety, %X% missing margins, constant spline subsets)
- Mixed numeric+categorical histogram btree
- D9 Anchor 5: ConformalInterval.reliability & ConformalReliabilityWarning
- D10 Anchor 4: mb.reproduce(), Fit.reproduce(), and CLI `python -m mboostrs reproduce`
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pytest

import mboostrs as mb


def test_lss_interaction_conformal_interval_succeeds():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.lognormal(0.5, 0.5, n)
    z = rng.uniform(0.0, 2.0, n)
    y = np.log(x) * z + rng.normal(0.0, 0.2 + 0.3 * z, n)
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(log(x)) %X% bbs(z)", "sigma": "y ~ bbs(z)"},
        data={"y": y, "x": x, "z": z},
        family=mb.GaussianLSS(),
        control=mb.boost_control(40, 0.1),
        na_action="impute",
    )
    ci = mb.conformal_interval(fit, method="split", level=0.9, seed=1)
    assert np.all(np.isfinite(ci.lower))
    assert np.all(np.isfinite(ci.upper))
    assert ci.reliability is not None
    assert ci.reliability["status"] in ("reliable", "caution")


def test_interaction_brandom_missing_values_and_unseen_test_nan_impute():
    rng = np.random.default_rng(11)
    n = 200
    g = np.array(["a", "b", "c", None] * (n // 4), dtype=object)
    x = rng.normal(size=n)
    y = x + np.where(g == "a", 1.0, 0.0) + rng.normal(0.0, 0.3, n)
    fit = mb.gamboost(
        "y ~ bbs(x) %X% brandom(g)",
        data={"y": y, "x": x, "g": g},
        control=mb.boost_control(40, 0.1),
        na_action="impute",
    )
    # Test row has BOTH a missing numeric x (complete in train!) AND None in g
    pred = fit.predict(
        newdata={"x": np.array([0.0, np.nan]), "g": np.array(["a", None], dtype=object)}
    )
    assert np.all(np.isfinite(pred))


def test_conformal_refit_handles_sparse_column_collapsing_to_constant():
    rng = np.random.default_rng(19)
    n = 250
    x1 = rng.normal(size=n)
    # Only 2 non-zero entries in training — almost certain to collapse on some splits
    x2 = np.zeros(n)
    x2[0] = 0.5
    x2[1] = 1.0
    y = 2.0 * x1 + rng.normal(0.0, 0.5, n)
    fit = mb.gamboost(
        "y ~ bbs(x1) + bbs(sqrt(x2))",
        data={"y": y, "x1": x1, "x2": x2},
        control=mb.boost_control(30, 0.1),
    )
    # Try several seeds to ensure a split where both non-zero rows land in calib still succeeds
    for s in range(10):
        ci = mb.conformal_interval(fit, method="split", level=0.9, seed=s)
        assert np.all(np.isfinite(ci.lower))


def test_fit_subsample_domain_safety_for_rare_zero_and_negative():
    rng = np.random.default_rng(23)
    n = 3500
    # Skewed positive column with a single zero at row 3499
    x_pos = rng.lognormal(1.0, 1.5, n)
    x_pos[-1] = 0.0
    # Skewed non-negative column with a single negative outlier at row 3498
    x_neg = rng.lognormal(1.0, 1.5, n)
    x_neg[-2] = -5.0
    y = np.log1p(x_pos) + 0.1 * np.maximum(x_neg, 0.0) + rng.normal(0.0, 0.5, n)
    fitted = mb.fit({"y": y, "x_pos": x_pos, "x_neg": x_neg}, "y", seed=0)
    pred = fitted.predict({"x_pos": x_pos[:10], "x_neg": x_neg[:10]})
    assert np.all(np.isfinite(pred))


def test_mixed_categorical_histogram_btree():
    rng = np.random.default_rng(31)
    n = 600
    x = rng.normal(size=n)
    g = rng.choice(["red", "green", "blue", "gold"], size=n)
    y = np.where(g == "red", 2.0 * x, -1.5 * x) + rng.normal(0.0, 0.3, n)
    data = {"y": y, "x": x, "g": g}
    m_hist = mb.gamboost(
        "y ~ btree(x, g, maxdepth=4, selection=sse, split_method=histogram)",
        data=data,
        control=mb.boost_control(60, 0.1),
    )
    m_exact = mb.gamboost(
        "y ~ btree(x, g, maxdepth=4, selection=sse, split_method=exact)",
        data=data,
        control=mb.boost_control(60, 0.1),
    )
    p_hist = np.asarray(m_hist.predict(newdata=data), dtype=float)
    p_exact = np.asarray(m_exact.predict(newdata=data), dtype=float)
    assert np.corrcoef(p_hist, p_exact)[0, 1] > 0.99
    assert float(np.mean((y - p_hist) ** 2)) < 0.25


def test_conformal_reliability_diagnostics_clean_vs_shifted_and_small():
    rng = np.random.default_rng(41)
    n = 500
    x = rng.normal(0.0, 1.0, n)
    y = 1.5 * x + rng.normal(0.0, 0.5, n)
    model = mb.gamboost("y ~ bols(x)", data={"y": y, "x": x}, control=mb.boost_control(50, 0.1))

    # 1. Clean in-distribution query with ample calibration split -> reliable
    x_id = rng.normal(0.0, 0.5, 100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", mb.ConformalReliabilityWarning)
        ci_ok = mb.conformal_interval(model, newdata={"x": x_id}, level=0.9, method="split", seed=0)
    assert ci_ok.reliability is not None
    assert ci_ok.reliability["status"] == "reliable"
    assert ci_ok.reliability["out_of_range_rows"] == 0
    assert len(caught) == 0

    # 2. Shifted / extrapolating query -> caution + ConformalReliabilityWarning
    x_ood = rng.normal(5.0, 1.0, 100)
    with pytest.warns(mb.ConformalReliabilityWarning, match="reliability caution"):
        ci_ood = mb.conformal_interval(
            model, newdata={"x": x_ood}, level=0.9, method="split", seed=0, extrapolation="allow"
        )
    assert ci_ood.reliability["status"] == "caution"
    assert ci_ood.reliability["out_of_range_fraction"] > 0.5
    assert "x" in ci_ood.reliability["shifted_columns"]

    # 3. Small calibration set -> caution + ConformalReliabilityWarning
    n_small = 50
    xs = rng.normal(size=n_small)
    ys = 1.5 * xs + rng.normal(0.0, 0.5, n_small)
    m_small = mb.gamboost("y ~ bols(x)", data={"y": ys, "x": xs}, control=mb.boost_control(30, 0.1))
    with pytest.warns(mb.ConformalReliabilityWarning, match="small calibration split"):
        ci_small = mb.conformal_interval(m_small, level=0.9, method="split", seed=0)
    assert ci_small.reliability["status"] == "caution"
    assert ci_small.reliability["calibration_coverage_se"] > 0.05


def test_reproduce_api_and_cli_match_bit_identically(tmp_path):
    rng = np.random.default_rng(53)
    n = 250
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 * x1 - np.sin(x2) + rng.normal(0.0, 0.4, n)
    data = {"y": y, "x1": x1, "x2": x2}

    fitted = mb.fit(data, "y", seed=7)
    res1 = fitted.reproduce(data)
    assert res1.matched is True
    assert res1.differences == ()
    assert len(res1.prediction_sha256) == 64
    assert "mboostrs fit" in res1.report

    # Round-trip through JSON file + mb.reproduce(path, data)
    trail_path = tmp_path / "trail.json"
    trail_path.write_text(fitted.trail.to_json(), encoding="utf-8")
    res2 = mb.reproduce(trail_path, data)
    assert res2.matched is True
    assert res2.prediction_sha256 == res1.prediction_sha256

    # One-command CLI reproduction
    import pandas as pd

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    out_md = tmp_path / "report.md"
    proc = subprocess.run(
        [sys.executable, "-m", "mboostrs", "reproduce", str(trail_path), str(csv_path), "--output", str(out_md)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_md.is_file()
    assert "# mboostrs fit" in out_md.read_text(encoding="utf-8")


def test_bbs_quantile_ospline_fits_linear_and_nonlinear_and_centered():
    rng = np.random.default_rng(71)
    n = 600
    x = np.exp(rng.normal(0.0, 1.2, n))
    y_lin = 0.5 * x + rng.normal(0.0, 0.5, n)
    m_lin = mb.gamboost(
        "y ~ bbs(x, knots=15, df=4, knot_placement='quantile')",
        data={"y": y_lin, "x": x},
        control=mb.boost_control(200, 0.1),
    )
    x_test = np.linspace(float(np.min(x)), float(np.max(x)) * 1.3, 50)
    pred = np.asarray(m_lin.predict(newdata={"x": x_test}, extrapolation="allow"), dtype=float)
    assert np.all(np.isfinite(pred))
    assert float(np.mean((pred - 0.5 * x_test) ** 2)) < 0.25

    # Centered quantile O-spline + nonlinear signed heavy-tailed curve
    s = rng.standard_t(2.5, n)
    y_nl = np.sin(s) + rng.normal(0.0, 0.2, n)
    m_nl = mb.gamboost(
        "y ~ bols(s) + bbs(s, knots=15, df=3, center=True, knot_placement=quantile)",
        data={"y": y_nl, "s": s},
        control=mb.boost_control(200, 0.1),
    )
    pred_nl = np.asarray(m_nl.predict(newdata={"s": s}), dtype=float)
    assert np.all(np.isfinite(pred_nl))
    assert float(np.mean((pred_nl - np.sin(s)) ** 2)) < 0.12


def test_high_cardinality_diag_brandom_is_fast_and_exact():
    import time

    rng = np.random.default_rng(83)
    n = 2500
    n_levels = 1200
    groups = rng.integers(0, n_levels, size=n)
    effects = rng.normal(0.0, 2.0, size=n_levels)
    y = effects[groups] + rng.normal(0.0, 0.5, size=n)
    labels = np.array([f"g{g}" for g in groups], dtype=object)

    t0 = time.perf_counter()
    m = mb.gamboost(
        "y ~ brandom(g, df=4)",
        data={"y": y, "g": labels},
        control=mb.boost_control(100, 0.1),
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"DiagBrandom took too long: {elapsed:.2f}s"
    pred = np.asarray(m.predict(newdata={"g": labels}, unseen_levels="average"), dtype=float)
    assert np.all(np.isfinite(pred))
    assert float(np.corrcoef(pred, effects[groups])[0, 1]) > 0.75


def test_fit_predict_clips_spline_extrapolation_by_default():
    rng = np.random.default_rng(91)
    n = 300
    x = rng.uniform(0.0, 2.0, n)
    y = np.sin(3.0 * x) + rng.normal(0.0, 0.2, n)
    fitted = mb.fit({"y": y, "x": x}, "y", seed=0)
    # Far out-of-range test value should be clipped flat at boundary by default
    p_default = np.asarray(fitted.predict({"x": np.array([10.0])}), dtype=float)
    p_edge = np.asarray(fitted.predict({"x": np.array([float(np.max(x))])}), dtype=float)
    assert np.allclose(p_default, p_edge, atol=1e-6)
    # Caller can still opt back into linear extrapolation explicitly
    p_allow = np.asarray(fitted.predict({"x": np.array([10.0])}, extrapolation="allow"), dtype=float)
    assert np.all(np.isfinite(p_allow))


def test_autospec_nominal_categorical_uses_random_folds_not_grouped():
    rng = np.random.default_rng(97)
    n = 400
    colors = np.array([f"color_{i}" for i in rng.integers(0, 14, size=n)], dtype=object)
    x = rng.normal(size=n)
    y = x + np.where(colors == "color_0", 2.0, 0.0) + rng.normal(0.0, 0.3, n)
    rec = mb.autospec({"y": y, "x": x, "color": colors}, "y", budget=60, search="fast", seed=0)
    assert "random folds" in rec.folds_description
    assert "grouped folds" not in rec.folds_description

