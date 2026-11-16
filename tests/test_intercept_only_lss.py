"""Intercept-only parameter formulas: `sigma: "y ~ 1"` (X03).

A constant distribution parameter used to require a fabricated constant
column: the rent case built `one=1.0` and fitted
``bols(one, intercept=False)``. The term ``1`` is now the intercept, as in
R's ``y ~ 1``: the parser emits one coefficient on a column of ones, and no
column of the frame is needed.

A real column actually called ``1`` keeps referring to the column, the
convention `expand_formula` documents for ``0``/``1`` terms: the intercept
interpretation applies only when no column holds the name.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _frame(seed=7, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + 0.5 * rng.normal(size=n)
    return pl.DataFrame({"x": x, "y": y})


def test_the_intercept_term_matches_the_fabricated_constant_column():
    """`y ~ 1` and `bols(one, intercept=False)` fit the same model."""
    data = _frame()
    with_one = data.with_columns(pl.lit(1.0).alias("one"))
    new = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                         data, family=mb.GaussianLSS())
    old = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(one, intercept=False)"},
        with_one, family=mb.GaussianLSS())
    for parameter in ("mu", "sigma"):
        a = np.asarray(new.predict(newdata=data, type="response")[parameter])
        b = np.asarray(old.predict(newdata=with_one, type="response")[parameter])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_a_constant_scale_is_estimated_not_frozen():
    """The sigma predictions move off the family offset, all by the same
    amount: one coefficient was estimated for the constant."""
    data = _frame()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                         data, family=mb.GaussianLSS())
    sigma = np.asarray(fit.predict(newdata=data, type="response")["sigma"])
    assert np.ptp(sigma) == pytest.approx(0.0, abs=1e-12)
    sigma_coefficients = fit.coef()["sigma"]
    assert isinstance(sigma_coefficients, dict)
    assert set(sigma_coefficients) == {"1", "(offset)"}
    assert sigma_coefficients["1"] != pytest.approx(0.0, abs=1e-3)


def test_new_data_needs_no_constant_column():
    """Prediction rows carry the predictors only; the ones column is
    supplied, so a frame with different row counts works."""
    data = _frame()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                         data, family=mb.GaussianLSS())
    fresh = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    prediction = fit.predict(newdata=fresh, type="response")
    assert len(prediction["sigma"]) == 3
    assert np.all(np.isfinite(np.asarray(prediction["mu"])))


def test_cross_validation_runs_on_the_intercept_only_parameter():
    data = _frame()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                         data, family=mb.GaussianLSS())
    risk = fit.cvrisk()
    assert len(risk.optimal_mstop) == 2
    assert all(value >= 1 for value in risk.optimal_mstop)


def test_glmboostlss_accepts_the_intercept_term():
    """The linear-LSS frontend reads `1` through the same recipe: one
    constant column in the design, no fabricated `one`."""
    data = _frame()
    fit = mb.glmboostLSS({"mu": "y ~ x", "sigma": "y ~ 1"}, data,
                         family=mb.GaussianLSS())
    sigma = np.asarray(fit.predict(newdata=data, type="response")["sigma"])
    assert np.ptp(sigma) == pytest.approx(0.0, abs=1e-12)


def test_a_real_column_named_one_still_refers_to_the_column():
    """A frame holding a column called `1` keeps naming the column; only a
    missing `1` column turns the term into the intercept. Both parameters
    are fitted on the real column, so both predictions vary with it, where
    the intercept pseudo-column would leave them flat."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 50)
    data = pl.DataFrame({"1": t, "y": 2 * t + 0.1 * rng.normal(size=50)})
    fit = mb.gamboostLSS({"mu": "y ~ bols(1)", "sigma": "y ~ bols(1)"},
                         data, family=mb.GaussianLSS())
    prediction = fit.predict(newdata=data, type="response")
    mu = np.asarray(prediction["mu"])
    sigma = np.asarray(prediction["sigma"])
    assert np.ptp(mu) > 1.0
    assert np.corrcoef(mu, t)[0, 1] > 0.9
    assert np.ptp(sigma) > 0


def test_gamboost_fits_intercept_only_as_an_offset_model():
    """`gamboost("y ~ 1")` fits no learner: the constant is the family's offset
    (#357). The LSS functions keep the intercept as a parameter learner."""
    data = _frame()
    fit = mb.gamboost("y ~ 1", data, family=mb.Gaussian())
    assert list(fit.selected) == []
