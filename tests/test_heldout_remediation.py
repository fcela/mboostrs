"""Regression tests for the held-out benchmark remediation fixes (D6, D1b, D11, D5)."""

from __future__ import annotations

import re
import warnings

import numpy as np
import polars as pl

import mboostrs as mb


def test_fit_handles_every_row_missing_and_unseen_levels_and_conformal():
    """D6: mb.fit defaults to na_action='impute' and unseen_levels='average'."""
    rng = np.random.default_rng(42)
    n = 300
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    cat_a = rng.choice(["red", "green", "blue"], size=n).astype(object)
    cat_b = rng.choice(["low", "high"], size=n).astype(object)
    y = 2.0 * x0 - 1.5 * x1 + np.where(cat_a == "red", 1.0, 0.0) + rng.normal(0, 0.4, size=n)

    # Make EVERY row have at least one missing predictor so na_action='omit' would drop 100% of rows.
    x0_miss = x0.copy()
    x1_miss = x1.copy()
    x0_miss[: n // 2] = np.nan
    x1_miss[n // 2 :] = np.nan
    cat_a[:20] = None

    train = pl.DataFrame({"x0": x0_miss, "x1": x1_miss, "cat_a": cat_a, "cat_b": cat_b, "y": y})
    test = pl.DataFrame({
        "x0": [0.5, np.nan, -1.0, 0.2],
        "x1": [np.nan, 0.3, 0.4, -0.5],
        "cat_a": ["red", "unseen_color", None, "blue"],
        "cat_b": ["low", "unseen_tier", None, "high"],
        "y": [1.0, 0.0, -1.0, 0.5],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mb.fit(train, "y", seed=0)
        preds = res.predict(test, type="response")
        if isinstance(preds, dict):
            preds = res.model.predict(test, type="mean")
        preds = np.asarray(preds, dtype=float).ravel()
        assert preds.shape == (4,)
        assert np.all(np.isfinite(preds))

        ci = mb.conformal_interval(res.model, test, level=0.9, method="split", seed=0)
        assert np.all(np.isfinite(ci.lower))
        assert np.all(np.isfinite(ci.upper))
        assert np.all(ci.upper >= ci.lower)


def test_fit_complete_training_categorical_handles_test_null_and_unseen():
    """D6: a categorical column complete in training still handles None and unseen levels at predict."""
    rng = np.random.default_rng(7)
    n = 250
    x = rng.normal(size=n)
    g = rng.choice(["a", "b", "c", "d"], size=n)
    y = x + np.where(g == "a", 1.0, -0.5) + rng.normal(0, 0.3, size=n)
    train = pl.DataFrame({"x": x, "g": pl.Series("g", g).cast(pl.Categorical), "y": y})
    test = pl.DataFrame({
        "x": [0.0, np.nan, 1.0],
        "g": pl.Series("g", [None, "z_unseen", "a"]).cast(pl.Categorical),
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mb.fit(train, "y", seed=0)
        preds = np.asarray(res.predict(test, type="response"), dtype=float).ravel()
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))


def test_fit_null_task_reports_intercept_only_formula():
    """D1b: pure noise task reports an intercept-only final formula with zero false discoveries."""
    rng = np.random.default_rng(50000)
    n = 600
    X = rng.uniform(-1, 1, (n, 12))
    y = rng.normal(0, 1, n)
    train = pl.DataFrame({f"x{i}": X[:, i] for i in range(12)} | {"y": y})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mb.fit(train, "y", seed=0)
    formula = res.trail.final["formula"]
    claimed = [f"x{i}" for i in range(12) if re.search(rf"\bx{i}\b", formula)]
    assert claimed == [], f"Expected zero claimed noise columns on null data, got {claimed} in {formula!r}"


def test_fit_adopts_guarded_residual_tree_on_pure_interaction_and_stays_additive_on_s1():
    """D11/D1b: adopts btree on S2 (3*sign(x0*x1)+x2) and stays additive on S1."""
    # S2: pure interaction
    rng = np.random.default_rng(30000)
    X = rng.uniform(-1, 1, (2500, 12))
    y = 3.0 * np.sign(X[:, 0] * X[:, 1]) + X[:, 2] + rng.normal(0, 0.5, 2500)
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(12)} | {"y": y})
    train, test = df[:1500], df[1500:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_s2 = mb.fit(train, "y", seed=0)
    preds = np.asarray(res_s2.predict(test, type="response"), dtype=float).ravel()
    mse = float(np.mean((test["y"].to_numpy() - preds) ** 2))
    assert mse < 1.2, f"Expected S2 MSE < 1.2 with guarded residual tree, got {mse:.3f}"
    formula_s2 = res_s2.trail.final["formula"]
    assert "btree(" in formula_s2
    assert "x0" in formula_s2 and "x1" in formula_s2

    # S1: purely additive
    rng1 = np.random.default_rng(20000)
    X1 = rng1.uniform(-1, 1, (2500, 12))
    y1 = 2.0 * np.sin(2.0 * X1[:, 0]) + 2.0 * (X1[:, 1] ** 2 - 1.0 / 3.0) + X1[:, 2] + rng1.normal(0, 0.5, 2500)
    df1 = pl.DataFrame({f"x{i}": X1[:, i] for i in range(12)} | {"y": y1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_s1 = mb.fit(df1[:1500], "y", seed=0)
    preds1 = np.asarray(res_s1.predict(df1[1500:], type="response"), dtype=float).ravel()
    mse1 = float(np.mean((df1[1500:]["y"].to_numpy() - preds1) ** 2))
    assert mse1 < 0.35
    assert "btree(" not in res_s1.trail.final["formula"]


def test_fit_subsamples_search_on_large_table_and_fits_full_rows():
    """D5: large tables run specification search on a subsample, then fit on all rows."""
    rng = np.random.default_rng(11)
    n = 8000
    X = rng.uniform(-1, 1, (n, 10))
    y = 2.0 * np.sin(2.0 * X[:, 0]) + X[:, 1] + rng.normal(0, 0.5, n)
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(10)} | {"y": y})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mb.fit(df, "y", seed=0)
    fitted = np.asarray(res.model.predict(type="response"), dtype=float).ravel()
    assert fitted.shape == (n,)
    trail_step = res.trail.to_dict()["steps"][-1]
    assert "subsample" in trail_step["reason"]
