"""Tests for the unified fast `autospec(..., search='fast')` pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _make_additive(n: int = 800, p: int = 15, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] + np.sin(2.0 * X[:, 1]) + 0.5 * X[:, 2] ** 2 + rng.normal(scale=0.5, size=n)
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(p)} | {"y": y})


def _make_xor(n: int = 1000, p: int = 12, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] * X[:, 1] + rng.normal(scale=0.5, size=n)
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(p)} | {"y": y})


def _make_null(n: int = 600, p: int = 12, seed: int = 2) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(p)} | {"y": y})


def _make_heteroscedastic(n: int = 1000, p: int = 12, *, product: bool = False, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    scale = np.exp(0.65 * X[:, 2] * X[:, 3]) if product else np.exp(0.65 * X[:, 2])
    y = 1.5 * X[:, 0] + np.sin(X[:, 1]) + rng.normal(scale=scale, size=n)
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(p)} | {"y": y})


def test_fast_autospec_additive_weeds_noise_and_fits():
    df = _make_additive()
    rec = mb.autospec(df, "y", budget=200, interactions="adopt", residual_tree=True, search="fast", seed=0)
    radar = rec.evidence["fast_radar"]
    assert len(radar["dropped_columns"]) >= 5
    for col in ("x0", "x1", "x2"):
        assert col in radar["kept_columns"]
    assert isinstance(rec.family, mb.Gaussian)
    assert "btree(" not in rec.formula
    fit = rec.fit(df)
    preds = np.asarray(fit.predict(type="response"), dtype=float)
    r2 = 1.0 - float(np.mean((df["y"].to_numpy() - preds) ** 2)) / float(np.var(df["y"].to_numpy()))
    assert r2 > 0.85


def test_fast_autospec_pure_xor_adopts_interaction_tree():
    df = _make_xor()
    tr, te = df[:800], df[800:]
    rec = mb.autospec(tr, "y", budget=200, interactions="adopt", residual_tree=True, search="fast", seed=0)
    assert "x0" in rec.evidence["fast_radar"]["kept_columns"]
    assert "x1" in rec.evidence["fast_radar"]["kept_columns"]
    assert rec.evidence.get("fast_interaction", {}).get("adopted") is True
    assert "btree(" in rec.formula
    fit = rec.fit(tr)
    preds = np.asarray(fit.predict(newdata=te.drop("y"), type="response"), dtype=float)
    y_te = te["y"].to_numpy()
    r2 = 1.0 - float(np.mean((y_te - preds) ** 2)) / float(np.var(y_te))
    assert r2 > 0.80


def test_fast_autospec_null_refuses_interactions_and_lss():
    df = _make_null()
    rec = mb.autospec(df, "y", budget=200, interactions="adopt", residual_tree=True, search="fast", seed=0)
    assert isinstance(rec.family, mb.Gaussian)
    assert not rec.evidence.get("fast_interaction", {}).get("adopted", False)
    assert not rec.evidence.get("fast_lss", {}).get("adopted", False)


@pytest.mark.parametrize("product", [False, True])
def test_fast_autospec_heteroscedastic_adopts_gaussian_lss(product):
    df = _make_heteroscedastic(product=product)
    rec = mb.autospec(df, "y", budget=200, interactions="adopt", residual_tree=True, search="fast", seed=0)
    assert type(rec.family).__name__ == "GaussianLSS"
    assert isinstance(rec.formula, dict) and "sigma" in rec.formula
    fit = rec.fit(df)
    mu = np.asarray(fit.predict(parameter="mu", type="response"), dtype=float)
    sigma = np.asarray(fit.predict(parameter="sigma", type="response"), dtype=float)
    assert np.all(np.isfinite(mu)) and np.all(sigma > 0)


def test_fast_autospec_preserves_mandatory_and_declared():
    df = _make_additive()
    rec = mb.autospec(
        df, "y", budget=200, search="fast", seed=0,
        mandatory=["x14"], monotone={"x13": "increasing"},
    )
    assert "x14" in rec.evidence["fast_radar"]["kept_columns"]
    assert "x13" in rec.evidence["fast_radar"]["kept_columns"]
    assert "bmono(x13" in rec.formula


def test_fast_autospec_binary_classification():
    rng = np.random.default_rng(7)
    n = 600
    X = rng.normal(size=(n, 10))
    logits = 1.5 * X[:, 0] - 1.2 * X[:, 1] ** 2
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(10)} | {"y": y})
    rec = mb.autospec(df, "y", budget=200, search="fast", seed=0)
    assert isinstance(rec.family, mb.Binomial)
    fit = rec.fit(df)
    probs = np.asarray(fit.predict(type="response"), dtype=float)
    assert np.all((probs > 0) & (probs < 1))
