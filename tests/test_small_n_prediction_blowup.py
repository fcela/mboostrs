"""Mean predictions far outside the training range are flagged on every path.

A spline basis continues at its edge slope beyond its boundary knots. When the
training sample is narrow and the covariate is heavy-tailed, a test row far past
the edge gets a linear extrapolation that can be hundreds of times the response
scale: in an 800-row claim-cost draw one row's predicted mean was 338 while the
other means lay in [-0.5, 1.0]. The default policy stays ``"warn"`` (the R
mboost continuation, documented and relied on for parity), so the value is
returned with one warning that names the covariate and the rows. Every mean
path must use that same default, must not swallow the warning, and must accept
``extrapolation="clip"``, which holds the covariate at the training edge.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning

FAR = np.array([250.0])


@pytest.fixture(scope="module")
def narrow():
    rng = np.random.default_rng(7)
    x = rng.uniform(1.0, 6.0, 300)
    y = 0.4 * np.log(x) + rng.normal(0.0, 0.2, 300)
    data = {"x": x, "y": y}
    lss = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"}, data=data,
        family=mb.GaussianLSS(), control=mb.boost_control(mstop=200, nu=0.1))
    gam = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(mstop=200, nu=0.1))
    return {"x": x, "y": y, "lss": lss, "gam": gam,
            "edge": np.array([x.max()]), "gam_refit": mb.gam_refit(gam, which="all"),
            "lss_refit": mb.gamlss_refit(lss, which="all")}


def _mu(out):
    return np.asarray(out["mu"] if isinstance(out, dict) else out, dtype=float).ravel()


# Each mean path as (name, call), where call(narrow, newdata_x, policy) returns
# the mean (or the mu column) at covariate values x. ``policy=None`` leaves the
# keyword out, so the default is what is exercised.
def _lss_predict(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.predict(newdata={"x": x}, type="response", **kw)["mu"]


def _lss_predict_parameters(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.predict_parameters({"x": x}, **kw)["mu"]


def _lss_params(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.params(newdata={"x": x}, **kw)["mu"]


def _lss_predict_all(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.predict_all(newdata={"x": x}, output="dict", **kw)["mu"]


def _lss_mean(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.mean(newdata={"x": x}, **kw)


def _gam_predict(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.predict(newdata={"x": x}, type="response", **kw)


def _gam_predict_over_mstop(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return fit.predict_over_mstop([200], newdata={"x": x}, type="response", **kw)


def _gam_predict_all(fit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    rows = fit.predict_all(newdata={"x": x}, output="records", **kw)
    return np.array([float(row["fitted"]) for row in rows])


def _lss_refit(refit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return refit.predict({"x": x}, parameter="mu", type="response", **kw)


def _gam_refit(refit, x, policy):
    kw = {} if policy is None else {"extrapolation": policy}
    return refit.predict({"x": x}, type="response", **kw)


PATHS = [
    ("lss.predict", "lss", _lss_predict),
    ("lss.predict_parameters", "lss", _lss_predict_parameters),
    ("lss.params", "lss", _lss_params),
    ("lss.predict_all", "lss", _lss_predict_all),
    ("lss.mean", "lss", _lss_mean),
    ("gam.predict", "gam", _gam_predict),
    ("gam.predict_over_mstop", "gam", _gam_predict_over_mstop),
    ("gam.predict_all", "gam", _gam_predict_all),
    ("gamlss_refit.predict", "lss_refit", _lss_refit),
    ("gam_refit.predict", "gam_refit", _gam_refit),
]


def _fit(narrow, key):
    return narrow[key]


@pytest.mark.parametrize("name, key, call", PATHS, ids=[p[0] for p in PATHS])
def test_default_warns_and_matches_the_explicit_warn(narrow, name, key, call):
    fit = _fit(narrow, key)
    with pytest.warns(ExtrapolationWarning):
        default = call(fit, FAR, None)
    with pytest.warns(ExtrapolationWarning):
        explicit = call(fit, FAR, "warn")
    np.testing.assert_allclose(np.asarray(default, dtype=float),
                               np.asarray(explicit, dtype=float))


@pytest.mark.parametrize("name, key, call", PATHS, ids=[p[0] for p in PATHS])
def test_clip_holds_the_value_at_the_training_edge(narrow, name, key, call):
    fit, edge = _fit(narrow, key), narrow["edge"]
    with pytest.warns(ExtrapolationWarning, match="clipped"):
        clipped = call(fit, FAR, "clip")
    at_edge = call(fit, edge, "clip")
    np.testing.assert_allclose(np.asarray(clipped, dtype=float),
                               np.asarray(at_edge, dtype=float))


def test_warn_keeps_the_edge_slope_and_is_far_from_the_clipped_value(narrow):
    fit, edge = narrow["lss"], narrow["edge"]
    with pytest.warns(ExtrapolationWarning):
        warned = _mu(fit.predict(newdata={"x": FAR}, type="response", extrapolation="warn"))
    edge_value = _mu(fit.predict(newdata={"x": edge}, type="response"))
    assert abs(float(warned[0])) > 10 * abs(float(edge_value[0]))


def test_allow_is_silent_and_gives_the_warn_value(narrow):
    fit = narrow["lss"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        allowed = _mu(fit.predict(newdata={"x": FAR}, type="response", extrapolation="allow"))
    with pytest.warns(ExtrapolationWarning):
        warned = _mu(fit.predict(newdata={"x": FAR}, type="response", extrapolation="warn"))
    np.testing.assert_allclose(allowed, warned)


def test_warning_names_the_number_of_rows_outside(narrow):
    fit = narrow["lss"]
    rows = {"x": np.array([250.0, 260.0, 270.0, 3.0])}
    with pytest.warns(ExtrapolationWarning, match="3 above"):
        fit.predict(newdata=rows, type="response")


def test_conformal_fitted_mean_warns_and_follows_the_policy(narrow):
    fit, edge = narrow["lss"], narrow["edge"]
    newdata = {"x": np.concatenate([FAR, edge])}
    with pytest.warns(ExtrapolationWarning):
        default = mb.conformal_interval(fit, newdata=newdata, level=0.9,
                                        method="distributional", seed=1)
    assert np.all(np.isfinite(default.fitted_mean))
    with pytest.warns(ExtrapolationWarning):
        expected = _mu(fit.predict(newdata={"x": FAR}, type="response"))
    np.testing.assert_allclose(default.fitted_mean[:1], expected)
    with pytest.warns(ExtrapolationWarning, match="clipped"):
        clipped = mb.conformal_interval(fit, newdata=newdata, level=0.9,
                                        method="distributional", seed=1,
                                        extrapolation="clip")
    np.testing.assert_allclose(clipped.fitted_mean[0], clipped.fitted_mean[1])
    np.testing.assert_allclose(clipped.fitted_mean[1],
                               _mu(fit.predict(newdata={"x": edge}, type="response")))
    assert np.all(np.isfinite(clipped.lower)) and np.all(np.isfinite(clipped.upper))
    assert np.max(np.abs(clipped.upper[0] - clipped.lower[0])) < 10 * (
        np.max(clipped.upper[1] - clipped.lower[1]) + 1.0)


def test_conformal_split_fitted_mean_warns_for_a_standard_model(narrow):
    fit = narrow["gam"]
    newdata = {"x": FAR}
    with pytest.warns(ExtrapolationWarning):
        interval = mb.conformal_interval(fit, newdata=newdata, level=0.9, seed=2)
    assert np.all(np.isfinite(interval.fitted_mean))


def test_tensor_margin_warns_by_default_and_clips_on_request():
    rng = np.random.default_rng(11)
    x = rng.uniform(1.0, 6.0, 300)
    z = rng.uniform(0.0, 1.0, 300)
    y = 0.4 * np.log(x) + z + rng.normal(0.0, 0.2, 300)
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x) %X% bbs(z)", "sigma": "y ~ 1"},
                         data={"x": x, "y": y, "z": z}, family=mb.GaussianLSS(),
                         control=mb.boost_control(mstop=150, nu=0.1))
    newdata = {"x": FAR, "z": np.array([0.5])}
    with pytest.warns(ExtrapolationWarning, match="x"):
        default = fit.predict(newdata=newdata, type="response")["mu"]
    with pytest.warns(ExtrapolationWarning, match="clipped"):
        clipped = fit.predict(newdata=newdata, type="response", extrapolation="clip")["mu"]
    with pytest.warns(ExtrapolationWarning):
        explicit = fit.predict(newdata=newdata, type="response", extrapolation="warn")["mu"]
    np.testing.assert_allclose(default, explicit)
    edge = fit.predict(newdata={"x": np.array([x.max()]), "z": np.array([0.5])},
                       type="response")["mu"]
    np.testing.assert_allclose(clipped, edge)


def test_draw_like_the_claim_cost_blowup_is_flagged_and_bounded_on_request():
    rng = np.random.default_rng(5)
    n = 800
    # The training covariate is moderate; the test covariate has a heavy right
    # tail, as average occupancy does in census block groups.
    train_x = np.exp(rng.normal(1.0, 0.5, n))
    test_x = np.exp(rng.normal(1.0, 1.4, 200))
    y = 0.3 * np.log(train_x) + rng.normal(0.0, 0.3, n)
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
                         data={"x": train_x, "y": y}, family=mb.GaussianLSS(),
                         control=mb.boost_control(mstop=300, nu=0.1))
    with pytest.warns(ExtrapolationWarning):
        mu_default = fit.predict_parameters({"x": test_x})["mu"]
    assert np.all(np.isfinite(mu_default))
    with pytest.warns(ExtrapolationWarning):
        mu_clip = fit.predict_parameters({"x": test_x}, extrapolation="clip")["mu"]
    assert np.max(np.abs(mu_clip)) < 3 * np.max(np.abs(y))
