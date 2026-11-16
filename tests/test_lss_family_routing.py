"""Distributional families must name the LSS fit, and the mean must be E[Y]."""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _zip_frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    mu = np.exp(0.2 + 0.4 * x)
    pi = 1 / (1 + np.exp(-0.3 * z))
    y = np.where(rng.random(n) < pi, 0.0, rng.poisson(mu))
    return pl.DataFrame({"x": x, "z": z, "y": y})


def test_gamboost_rejects_zip_by_naming_gamboostlss():
    data = _zip_frame()
    with pytest.raises(TypeError, match=r"gamboostLSS"):
        mb.gamboost("y ~ bols(x)", data, family=mb.ZIPoLSS(),
                    control=mb.boost_control(5, 0.1))


def test_glmboost_rejects_zip_by_naming_gamboostlss():
    data = _zip_frame()
    with pytest.raises(TypeError, match=r"glmboostLSS"):
        mb.glmboost("y ~ x", data, family=mb.ZIPoLSS(),
                    control=mb.boost_control(5, 0.1))


def test_gamboostlss_rejects_poisson_by_naming_gamboost():
    data = _zip_frame()
    with pytest.raises(TypeError, match=r"gamboost\(\)"):
        mb.gamboostLSS("y ~ bols(x)", data, family=mb.Poisson(),
                       control=mb.boost_control(5, 0.1))


def test_glmboostlss_rejects_poisson_by_naming_glmboost():
    data = _zip_frame()
    with pytest.raises(TypeError, match=r"glmboost\(\)"):
        mb.glmboostLSS("y ~ x", data, family=mb.Poisson(),
                       control=mb.boost_control(5, 0.1))


def test_zip_mean_is_mixture_not_poisson_mean():
    data = _zip_frame(n=200)
    fit = mb.glmboostLSS(
        "y ~ x + z", data, family=mb.ZIPoLSS(), control=mb.boost_control(30, 0.1))
    params = fit.predict(type="response")
    mixture = (1.0 - params["sigma"]) * params["mu"]
    np.testing.assert_allclose(fit.mean(), mixture)
    np.testing.assert_allclose(fit.predict(type="mean"), mixture)
    # The Poisson component is not the expected count when inflation is present.
    assert np.mean(np.abs(params["mu"] - mixture)) > 1e-6
