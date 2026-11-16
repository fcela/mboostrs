"""Verify Gaussian Gram (fast) path is active and produces reasonable results."""

import mboostrs as mboost
import numpy as np
from numpy.testing import assert_allclose

rng = np.random.default_rng(20250609)
n = 100
p = 5
X = {f"x{i}": rng.normal(0, 1, n) for i in range(p)}
X["y"] = rng.normal(0, 1, n)


def test_gram_fitted_finite():
    """Gram path produces finite fitted values."""
    model = mboost.glmboost(
        "y ~ " + " + ".join(f"x{i}" for i in range(p)),
        data=X, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    f = model.fitted()
    assert np.all(np.isfinite(f))
    assert len(f) == n


def test_gram_vs_standard_similar():
    """Gram (inbag) and non-Gram (oobag) converge to similar coefs."""
    model_gram = mboost.glmboost(
        "y ~ " + " + ".join(f"x{i}" for i in range(p)),
        data=X, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    coef_gram = model_gram.coef()
    # Refit with out-of-bag risk (forces non-Gram path)
    model_nogram = mboost.glmboost(
        "y ~ " + " + ".join(f"x{i}" for i in range(p)),
        data=X, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1, "oobag", False),
    )
    coef_nogram = model_nogram.coef()
    for name in coef_gram:
        if name != "(Intercept)":
            assert_allclose(coef_gram[name], coef_nogram[name],
                            rtol=1e-4, err_msg=f"Gram vs non-Gram: {name} diverges")


def test_gram_slice_non_gram():
    """Slicing the Gram-path model works."""
    model = mboost.glmboost(
        "y ~ " + " + ".join(f"x{i}" for i in range(p)),
        data=X, family=mboost.Gaussian(),
        control=mboost.boost_control(200, 0.1),
    )
    half = model[:50]
    assert len(half.selected) == 50
    assert len(half.risk_) == 51
    assert len(half.fitted()) == n


def test_gram_convergence():
    """Gram path should converge toward true coefficients."""
    rng2 = np.random.default_rng(20250610)
    n2 = 200
    x1 = rng2.normal(size=n2)
    x2 = rng2.normal(size=n2)
    y = 0.3 * x1 + 1.2 * x2 + rng2.normal(scale=0.5, size=n2)
    d = {"y": y, "x1": x1, "x2": x2}
    model = mboost.glmboost(
        "y ~ x1 + x2", data=d, family=mboost.Gaussian(),
        control=mboost.boost_control(250, 0.05),
    )
    c = model.coef()
    assert_allclose(c["x1"], 0.3, atol=0.15)
    assert_allclose(c["x2"], 1.2, atol=0.15)
