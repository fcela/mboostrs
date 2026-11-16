"""Interval for the summed prediction of a refit (issue #416, item 6).

`GamRefit.predict(interval=True)` and `GamlssRefit.predict(interval=True)`
give the standard error of the sum of the intercept and the terms at new
rows, from the full covariance of those coefficients, ``sqrt(x' V x)``. A
term's own interval (`effects`) leaves out the intercept, so the two agree
only after the intercept's variance is added.
"""

import warnings
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

LEVEL = 0.9
Z = NormalDist().inv_cdf(0.5 + LEVEL / 2)


def _gaussian_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(3 * x) + 0.5 * z + rng.normal(0, 0.3, n)
    return {"x": x, "z": z, "y": y}


@pytest.fixture(scope="module")
def one_term_refit():
    data = _gaussian_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x)", data=data, control=mb.boost_control(200))
        return mb.gam_refit(model, which="all"), data


@pytest.fixture(scope="module")
def two_term_refit():
    data = _gaussian_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, control=mb.boost_control(200))
        return mb.gam_refit(model, which="all"), data


def test_summed_estimate_is_the_intercept_plus_the_effects(two_term_refit):
    refit, data = two_term_refit
    nd = {"x": data["x"][:7], "z": data["z"][:7]}
    rows = refit.predict(nd, interval=True, output="records")
    estimate = np.array([r["estimate"] for r in rows])
    np.testing.assert_allclose(estimate, refit.predict(nd), rtol=0, atol=1e-10)
    effects = refit.effects(nd, level=LEVEL)
    total = refit._intercept + sum(est for est, _, _ in effects.values())
    np.testing.assert_allclose(estimate, total, rtol=0, atol=1e-10)


def test_one_term_se_is_the_hand_computed_quadratic_form(one_term_refit):
    """With one term and a Gaussian fit the intercept and the term have no
    covariance (the centred term is orthogonal to the constant), so
    x' V x = V_00 + z' V_TT z, and z' V_TT z is the squared term standard error
    from `effects`. The check reads both from public outputs."""
    refit, data = one_term_refit
    nd = {"x": data["x"][:9]}
    V = refit.vcov()
    cross = V[0, 1:]
    assert np.max(np.abs(cross)) < 1e-12 * np.max(np.abs(V))
    est, lower, upper = refit.effects(nd, level=LEVEL)["bbs(x)"]
    term_se = (upper - lower) / (2 * Z)
    expected = np.sqrt(V[0, 0] + term_se ** 2)
    rows = refit.predict(nd, interval=True, level=LEVEL, output="records")
    np.testing.assert_allclose([r["se"] for r in rows], expected, rtol=1e-9)
    np.testing.assert_allclose([r["lower"] for r in rows],
                               [r["estimate"] - Z * r["se"] for r in rows], rtol=1e-12)
    np.testing.assert_allclose([r["upper"] for r in rows],
                               [r["estimate"] + Z * r["se"] for r in rows], rtol=1e-12)


def test_summed_interval_is_wider_than_the_term_interval(one_term_refit):
    refit, data = one_term_refit
    nd = {"x": data["x"][:9]}
    rows = refit.predict(nd, interval=True, level=LEVEL, output="records")
    _, lower, upper = refit.effects(nd, level=LEVEL)["bbs(x)"]
    term_width = upper - lower
    summed_width = np.array([r["upper"] - r["lower"] for r in rows])
    assert np.all(summed_width > term_width)


def test_corrected_interval_is_not_narrower(one_term_refit):
    refit, data = one_term_refit
    nd = {"x": data["x"][:9]}
    plain = refit.predict(nd, interval=True, output="records")
    corrected = refit.predict(nd, interval=True, corrected=True, output="records")
    assert all(c["se"] >= p["se"] - 1e-12 for p, c in zip(plain, corrected, strict=True))


def test_response_scale_bounds_are_the_link_bounds_for_identity_link(two_term_refit):
    """A Gaussian refit has the identity link, so the response-scale interval
    equals the link-scale one."""
    refit, data = two_term_refit
    nd = {"x": data["x"][:5], "z": data["z"][:5]}
    link = refit.predict(nd, interval=True, output="records")
    response = refit.predict(nd, type="response", interval=True, output="records")
    for a, b in zip(link, response, strict=True):
        assert b["estimate"] == pytest.approx(a["estimate"], abs=1e-12)
        assert b["lower"] == pytest.approx(a["lower"], abs=1e-12)
        assert b["upper"] == pytest.approx(a["upper"], abs=1e-12)
        assert b["se"] == pytest.approx(a["se"], abs=1e-12)


def test_binomial_interval_brackets_the_probability():
    rng = np.random.default_rng(3)
    n = 400
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    p = 1 / (1 + np.exp(-(2 * np.sin(3 * x) + z - 0.5)))
    data = {"x": x, "z": z, "y": (rng.uniform(size=n) < p).astype(float)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, family=mb.Binomial(),
                            control=mb.boost_control(150))
        refit = mb.gam_refit(model, which="all")
    nd = {"x": x[:6], "z": z[:6]}
    rows = refit.predict(nd, type="response", interval=True, output="records")
    assert all(0 < r["lower"] <= r["estimate"] <= r["upper"] < 1 for r in rows)
    link = refit.predict(nd)
    np.testing.assert_allclose([r["estimate"] for r in rows], 1 / (1 + np.exp(-link)),
                               rtol=1e-10)
    assert all(r["se"] > 0 for r in rows)


def test_interval_level_is_validated(one_term_refit):
    refit, data = one_term_refit
    with pytest.raises(ValueError, match="level must be between 0 and 1"):
        refit.predict({"x": data["x"][:3]}, interval=True, level=1.5)


def test_gamlss_per_parameter_interval_matches_posterior_draws():
    """Each parameter's summed interval uses the covariance of that parameter's
    coefficients. Under the LSS weights the intercept and the term are
    correlated, so the one-term formula of the Gaussian gam test does not
    apply; the independent check is the spread of posterior draws. mu has the
    identity link, so the sd of its draws is its link-scale se; sigma has the
    log link, so the sd of log(sigma) over the draws is its link-scale se.
    Draws are Monte Carlo, so the tolerance is 3%."""
    rng = np.random.default_rng(0)
    n = 300
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(3 * x) + rng.normal(0, np.exp(-1 + 0.8 * z), n)
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"}, data=df,
                             families=mb.GaussianLSS(), control=mb.boost_control(mstop=200))
        refit = mb.gamlss_refit(fit)
    nd = {"x": x[:6], "z": z[:6]}
    frames = refit.predict(nd, interval=True, level=LEVEL, output="records")
    assert set(frames) == {"mu", "sigma"}
    draws = refit.draws(nd, n_draws=20000, seed=7)
    mu_se = np.array([r["se"] for r in frames["mu"]])
    sigma_se = np.array([r["se"] for r in frames["sigma"]])
    np.testing.assert_allclose(mu_se, draws["mu"].std(axis=0), rtol=0.03)
    np.testing.assert_allclose(sigma_se, np.log(draws["sigma"]).std(axis=0), rtol=0.03)
    for name in ("mu", "sigma"):
        estimate = np.array([r["estimate"] for r in frames[name]])
        link = refit.predict(nd, parameter=name)
        np.testing.assert_allclose(estimate, link, atol=1e-10)


def test_gamlss_single_parameter_returns_one_frame():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0, 1, n)
    df = pl.DataFrame({"x": x, "y": np.sin(3 * x) + rng.normal(0, 0.3, n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}, data=df,
                             families=mb.GaussianLSS(), control=mb.boost_control(mstop=100))
        refit = mb.gamlss_refit(fit, which="all")
    frame = refit.predict({"x": x[:4]}, parameter="mu", interval=True, output="records")
    assert isinstance(frame, list) and len(frame) == 4
    assert set(frame[0]) == {"estimate", "se", "lower", "upper", "outside_range"}
