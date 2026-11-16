"""Regression tests for #229: predict() warns once on the default link scale.

The default stays ``type="link"`` (R mboost parity); the first default-scale
prediction on a non-identity-link family must name the scale it returns.
"""
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


@pytest.fixture()
def gamma_data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = np.exp(1 + 0.5 * x) * rng.gamma(shape=5.0, scale=0.2, size=n)
    return pl.DataFrame({"x": x, "y": y})


@pytest.fixture()
def gaussian_data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"x": x, "y": y})


def _scale_warnings(w):
    return [x for x in w if "default type='link'" in str(x.message)]


def test_gamma_default_predict_warns_once(gamma_data):
    fit = mb.glmboost("y ~ bols(x)", data=gamma_data, family=mb.GammaReg(),
                      control=mb.boost_control(mstop=50, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit.predict()
        fit.predict()  # second call must not warn again
    msgs = _scale_warnings(w)
    assert len(msgs) == 1
    assert "log-mean" in str(msgs[0].message)
    assert "GammaReg" in str(msgs[0].message)


def test_explicit_type_never_warns(gamma_data):
    fit = mb.glmboost("y ~ bols(x)", data=gamma_data, family=mb.GammaReg(),
                      control=mb.boost_control(mstop=50, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit.predict(type="link")
        fit.predict(type="response")
    assert _scale_warnings(w) == []


def test_gaussian_default_predict_does_not_warn(gaussian_data):
    fit = mb.gamboost("y ~ bols(x)", data=gaussian_data, family=mb.Gaussian(),
                      control=mb.boost_control(mstop=50, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit.predict()
        fit.predict_parameters()
    assert _scale_warnings(w) == []


def test_gamboost_gamma_warns(gamma_data):
    fit = mb.gamboost("y ~ bbs(x, df=4)", data=gamma_data, family=mb.GammaReg(),
                      control=mb.boost_control(mstop=50, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit.predict()
    assert len(_scale_warnings(w)) == 1


def test_lss_predict_parameters_defaults_to_response_without_warning():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    y = np.exp(0.5 + 0.3 * x + rng.normal(scale=0.4, size=n))
    data = pl.DataFrame({"x": x, "y": y})
    fit = mb.glmboostLSS("y ~ bols(x)", data=data, family=mb.GammaLSS(),
                         control=mb.boost_control(mstop=30, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pp = fit.predict_parameters()
        fit.predict_parameters()
    # #326: an omitted type means the response scale, and never warns.
    assert _scale_warnings(w) == []
    response = fit.predict(type="response")
    assert set(pp) == {"mu", "sigma"}
    for name in ("mu", "sigma"):
        np.testing.assert_allclose(pp[name], np.asarray(response[name]))
    # predict() itself keeps the link-scale default and its #229 warning.
    with pytest.warns(UserWarning, match="default type='link'") as record:
        link = fit.predict()
    message = str(record[0].message)
    assert "predict()" in message
    assert "mu: log-mean" in message
    assert "sigma: log-shape" in message
    for name in ("mu", "sigma"):
        np.testing.assert_allclose(np.asarray(link[name]), np.log(np.asarray(pp[name])))


def test_summary_reports_prediction_scale(gamma_data, gaussian_data):
    fit = mb.glmboost("y ~ bols(x)", data=gamma_data, family=mb.GammaReg(),
                      control=mb.boost_control(mstop=50, nu=0.1))
    d = fit.summary(as_dict=True)
    assert d["prediction_scale"] == "link (log-mean)"
    assert "Default predict scale" in str(fit.summary())

    fitg = mb.glmboost("y ~ bols(x)", data=gaussian_data, family=mb.Gaussian(),
                       control=mb.boost_control(mstop=50, nu=0.1))
    assert fitg.summary(as_dict=True)["prediction_scale"] == \
        "link (identity: same as response)"
