"""Distributional fits divide the response by a scale and map the model back.

Boosted LSS fitting moves a location parameter by about ``nu`` per step in
response units once gradients are stabilized, and by ``nu (y - mu)/sigma^2``
when they are not, so the same data in different units gave different fits
and a response in large units barely left its offset. ``response_scale``
(default ``"auto"``: the weighted SD of the response each fit sees) fits
``y / s`` and maps every parameter's predictor back, so a fit is the same in
any units. ``"none"`` (or ``None``) is R gamboostLSS's raw-units path.
"""

import pickle

import numpy as np
import pytest

import mboostrs as mb

C = 1000.0


def _data(unit=1.0, n=300, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = unit * (1.0 + np.sin(2 * x) + np.exp(-1.0 + 0.6 * x) * rng.standard_normal(n))
    return {"x": x, "y": y}


def _gam(data, **kwargs):
    return mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"}, data,
                          family=mb.GaussianLSS(), control=mb.boost_control(150, 0.1),
                          **kwargs)


@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
def test_a_fit_in_other_units_is_the_same_model_in_those_units(method):
    small, large = _gam(_data(1.0), method=method), _gam(_data(C), method=method)
    p, q = small.predict(type="response"), large.predict(type="response")
    np.testing.assert_allclose(q["mu"], C * p["mu"], rtol=1e-9)
    np.testing.assert_allclose(q["sigma"], C * p["sigma"], rtol=1e-9)
    n = len(p["mu"])
    assert float(large.logLik()) == pytest.approx(float(small.logLik()) - n * np.log(C),
                                                  rel=1e-10)
    assert list(large._selection_order) == list(small._selection_order)


def _t3(unit, n=400, seed=4):
    # Student t errors with 3 df, so the df parameter is identified.
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return {"x": x, "y": unit * (1.0 + x + 0.5 * rng.standard_t(3, n))}


def test_the_linear_fit_is_the_same_model_in_other_units():
    fit = lambda d: mb.glmboostLSS("y ~ x", d, families=mb.StudentTLSS(),  # noqa: E731
                                   control=mb.boost_control(80, 0.1))
    small, large = fit(_t3(1.0)), fit(_t3(C))
    p, q = small.predict(type="response"), large.predict(type="response")
    # Student t's offsets come from a Brent search, whose iterates rounding
    # moves at the 1e-7 level.
    np.testing.assert_allclose(q["mu"], C * p["mu"], rtol=1e-6)
    np.testing.assert_allclose(q["sigma"], C * p["sigma"], rtol=1e-6)
    np.testing.assert_allclose(q["df"], p["df"], rtol=1e-6)


def test_the_raw_path_stalls_in_large_units():
    data = _data(1e5)
    raw = _gam(data, response_scale="none")
    scaled = _gam(data)
    y = data["y"]
    rmse = lambda m: float(np.sqrt(np.mean((y - m.predict(type="response")["mu"]) ** 2)))  # noqa: E731
    # The raw fit's mean barely leaves the offset (its RMSE is the response
    # SD); the scaled fit explains most of the mean.
    assert rmse(raw) > 0.99 * float(np.std(y))
    assert rmse(scaled) < 0.8 * float(np.std(y))


def test_none_and_the_string_none_are_the_raw_path():
    data = _data(7.0)
    a, b = _gam(data, response_scale=None), _gam(data, response_scale="none")
    np.testing.assert_array_equal(a.predict(type="link")["mu"], b.predict(type="link")["mu"])
    assert a._response_scale == "none" and a._response_scale_used is None


def test_the_automatic_scale_is_the_weighted_sd_and_is_recorded():
    data = _data(7.0)
    fit = _gam(data)
    assert fit._response_scale == "auto"
    assert fit._response_scale_used == pytest.approx(float(np.std(data["y"])), rel=1e-12)
    fixed = _gam(data, response_scale=fit._response_scale_used)
    np.testing.assert_allclose(fixed.predict(type="link")["mu"], fit.predict(type="link")["mu"],
                               rtol=1e-12)


def test_a_refit_and_a_saved_model_keep_the_scale_request():
    data = _data(50.0)
    fit = _gam(data)
    again = fit.update(mstop=150)
    np.testing.assert_allclose(again.predict(type="link")["mu"], fit.predict(type="link")["mu"],
                               rtol=1e-12)
    restored = pickle.loads(pickle.dumps(fit))
    assert restored._response_scale == "auto"
    np.testing.assert_allclose(restored.predict(type="link")["mu"],
                               fit.predict(type="link")["mu"], rtol=1e-12)


def test_count_families_are_not_scaled():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 200)
    data = {"x": x, "y": rng.poisson(np.exp(1 + x)).astype(float)}
    fit = mb.glmboostLSS("y ~ x", data, families=mb.NBinomialLSS(),
                         control=mb.boost_control(30, 0.1))
    assert fit._response_scale_used is None


@pytest.mark.parametrize("bad", ["sd", -1.0, 0.0, float("nan")])
def test_an_invalid_scale_is_refused(bad):
    with pytest.raises(ValueError, match="response_scale"):
        _gam(_data(), response_scale=bad)


def test_noncyclic_tune_mstop_searches_total_steps():
    data = _data(20.0)
    spec = {"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"}
    tuned = mb.tune_mstop(spec, data, family=mb.GaussianLSS(), seed=0, method="noncyclic")
    assert tuned.model._method == "noncyclic"
    folds = mb.cv_weights(len(data["y"]), type="kfold", B=5, seed=0)
    largest = int(tuned.largest)
    reference = mb.gamboostLSS(spec, data, family=mb.GaussianLSS(), method="noncyclic",
                               control=mb.boost_control(largest, 0.1)).cvrisk(
        folds=folds, grid=list(range(1, largest + 1)))
    best = int(np.asarray(reference.optimal_mstop).ravel()[0])
    assert tuned.mstop == best
    assert sum(len(p["selected"]) for p in tuned.model._params) == tuned.mstop


@pytest.mark.parametrize(("family", "rtol"), [
    (lambda **k: mb.Quantile(tau=0.8, **k), 1e-12),
    # The Laplace offset is R's Brent search for the median, and with an even
    # number of rows any value between the two middle observations minimises
    # the loss: the search stops somewhere in that flat stretch, and rounding
    # (including x86_64 vs arm64 FMA differences) decides where. The fit is
    # the same up to that choice, which also flips the sign of rows lying
    # closer to the fit than the stretch is wide (measured: 0.15%-0.6% of SD).
    (lambda **k: mb.Laplace(**k), 1e-2),
])
def test_quantile_and_laplace_fits_are_the_same_model_in_other_units(family, rtol):
    # Their gradient (tau - 1{y < f}, sign(y - f)) does not grow with the
    # response; with the default scale a fit in other units is the same
    # model in those units, and R's raw-units fit is not.
    small, large = _data(1.0), _data(C)
    fit = lambda d, **k: mb.gamboost("y ~ bbs(x)", d, family=family(**k),  # noqa: E731
                                     control=mb.boost_control(200, 0.1))
    a, b = fit(small).predict(), fit(large).predict()
    # Measured against the response's spread, so values near zero do not
    # inflate a relative difference.
    spread = C * float(np.std(small["y"]))
    assert float(np.max(np.abs(np.asarray(b) - C * np.asarray(a)))) <= rtol * spread
    raw_a = fit(small, response_scale=None).predict()
    raw_b = fit(large, response_scale=None).predict()
    assert not np.allclose(np.asarray(raw_b), C * np.asarray(raw_a), rtol=0.05)


def test_quantile_defaults_start_at_tau_and_step_at_point_three():
    q = mb.Quantile(tau=0.9)
    assert q.qoffset == 0.9 and q.default_nu == 0.3 and q.response_scale == "auto"
    fit = mb.glmboost("y ~ x", _data(), family=q)
    assert fit._control[1] == pytest.approx(0.3)
