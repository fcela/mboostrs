"""#326: ``predict_parameters()`` defaults to the response/natural scale.

``predict()`` keeps its link-scale default (R ``mboost`` parity) and the
#229 warning that names the scale. ``predict_parameters()`` is the call whose
output gets published, so an omitted ``type`` there now means every parameter
on the scale it is named for — an NB ``sigma`` is the size, not ``log(size)``
— and no link-scale warning fires.
"""
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _nb_frame(seed=0, n=600):
    """NB2 counts: ``mu = exp(1 + x)`` with ``x ~ U(0, 1)`` and a known size 3."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    mu = np.exp(1.0 + x)
    size = 3.0
    y = rng.negative_binomial(n=size, p=size / (size + mu))
    return pl.DataFrame({"x": x, "y": y.astype(float)})


@pytest.fixture(scope="module")
def nb_fit():
    return mb.gamboostLSS("y ~ bbs(x, df=4)", data=_nb_frame(),
                           families=mb.NBinomialLSS(),
                           control=mb.boost_control(100, 0.1))


def test_nb_lss_sigma_defaults_to_size_not_log_size(nb_fit):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the call
        params = nb_fit.predict_parameters()
    sigma = np.asarray(params["sigma"], dtype=float)
    # True size is 3: the response scale, not its log (~1.1).
    assert 2.0 < np.median(sigma) < 4.5


def test_nb_lss_response_is_exp_of_link(nb_fit):
    response = nb_fit.predict_parameters()
    link = nb_fit.predict_parameters(type="link")
    for name in ("mu", "sigma"):
        np.testing.assert_allclose(
            np.asarray(response[name], dtype=float),
            np.exp(np.asarray(link[name], dtype=float)),
            rtol=0.0, atol=1e-10)


def test_standard_model_scales_match_predict():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.poisson(np.exp(0.5 + x)).astype(float)
    data = pl.DataFrame({"x": x, "y": y})
    fit = mb.glmboost("y ~ x", data=data, family=mb.Poisson(),
                      control=mb.boost_control(50, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the call
        params = fit.predict_parameters()
    np.testing.assert_allclose(
        np.asarray(params["f"], dtype=float),
        np.asarray(fit.predict(type="response"), dtype=float),
        rtol=0.0, atol=1e-12)
    # Counts, not log-counts: the fitted means run around exp(1).
    assert np.median(np.asarray(params["f"], dtype=float)) > 1.5
    link = fit.predict_parameters(type="link")
    np.testing.assert_allclose(
        np.asarray(link["f"], dtype=float),
        np.asarray(fit.predict(type="link"), dtype=float),
        rtol=0.0, atol=1e-12)


def test_predict_default_warning_names_predict_and_the_size_label():
    # A fresh fit, so predict() with an omitted type is its first link-scale
    # call and the #229 warning fires here.
    fit = mb.gamboostLSS("y ~ bbs(x, df=4)", data=_nb_frame(),
                        families=mb.NBinomialLSS(),
                        control=mb.boost_control(100, 0.1))
    with pytest.warns(UserWarning, match="default type='link'") as record:
        fit.predict()
    message = str(record[0].message)
    assert "predict()" in message
    assert "predict_parameters" not in message
    # The corrected label: sigma is a size, so the old "log-dispersion"
    # wording is gone. (The label's parenthetical still says which way the
    # size points, so the check is on the old term, not the word.)
    assert "log-size" in message
    assert "log-dispersion" not in message
