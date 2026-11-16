"""A linear boost can stop, by cross-validation, short of the same-family GLM.

The check reports that gap. It does not extend the path, and it does not
move the step the tuning score chose. Synthetic responses only: the batting
and wine cases that motivated it are not required.
"""

import warnings

import numpy as np
import pytest
from mboostrs.cvrisk import CvRiskResult

import mboostrs as mb


def _gaussian(n=80, mstop=5, nu=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.4, size=n)
    data = {"y": y, "x1": x1, "x2": x2}
    model = mb.glmboost(
        "y ~ x1 + x2", data=data, family=mb.Gaussian(),
        control=mb.boost_control(mstop, nu),
    )
    return model


def test_cross_validation_stopped_short_of_the_gaussian_glm():
    model = _gaussian(mstop=5, nu=0.1)
    before = model.mstop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cv = model.cvrisk(type="kfold", B=3, seed=1)
    report = mb.same_family_gap(model, cv)
    assert model.mstop == before
    assert report.available
    assert report.mstop == cv.optimal_mstop
    assert report.mstop_source == "cvrisk"
    assert report.gap > 1.0
    assert report.material
    assert report.comparison_loglik > report.boost_loglik
    assert "Cross-validation stopped" in report.message
    assert "short of the same-family GLM" in report.message
    assert "was not changed" in report.message


def test_a_boost_that_reached_the_glm_reports_a_negligible_gap():
    model = _gaussian(mstop=250, nu=0.5)
    report = mb.same_family_gap(model)
    assert report.available
    assert not report.material
    assert report.gap == pytest.approx(0.0, abs=1e-4)
    assert "not short" in report.message
    # Judged at the cross-validated step when one is supplied, including a
    # step earlier than the path's end. The original fit stays where it was.
    stopped = CvRiskResult(
        np.ones((2, 2)), np.array([1.0, 0.5]), np.array([2, model.mstop]),
        2, "2-fold kfold",
    )
    early = mb.same_family_gap(model, stopped)
    assert model.mstop == 250
    assert early.mstop == 2
    assert early.material
    assert early.gap > 1.0
    assert "Cross-validation stopped" in early.message


def test_proportional_odds_and_beta_are_the_same_comparison():
    rng = np.random.default_rng(1)
    n = 90
    x = rng.normal(size=n)
    u = rng.uniform(size=n)
    eta = 1.2 * x
    category = (
        1
        + (u > 1 / (1 + np.exp(0.2 + eta))).astype(int)
        + (u > 1 / (1 + np.exp(-0.5 + eta))).astype(int)
    )
    ordered = {"y": category.astype(float), "x": x}
    short = mb.glmboost(
        "y ~ x", data=ordered, family=mb.PropOdds(n_levels=3),
        control=mb.boost_control(2, 0.1),
    )
    long = mb.glmboost(
        "y ~ x", data=ordered, family=mb.PropOdds(n_levels=3),
        control=mb.boost_control(120, 0.3),
    )
    early = mb.same_family_gap(short)
    reached = mb.same_family_gap(long)
    assert early.material and early.gap > 1.0
    assert early.comparison == "proportional-odds model"
    assert not reached.material
    assert reached.gap == pytest.approx(0.0, abs=1e-3)

    mu = 1 / (1 + np.exp(-(0.2 + 0.8 * x)))
    phi = 8.0
    y = rng.beta(mu * phi, (1 - mu) * phi)
    beta_data = {"y": y, "x": x}
    beta_short = mb.glmboostLSS(
        "y ~ x", data=beta_data, families=mb.BetaLSS(),
        control=mb.boost_control(6, 0.1),
    )
    beta_long = mb.glmboostLSS(
        "y ~ x", data=beta_data, families=mb.BetaLSS(),
        control=mb.boost_control(80, 0.2),
    )
    cv = type("Cv", (), {"optimal_mstop": [6, 6]})()
    beta_early = mb.same_family_gap(beta_short, cv)
    beta_reached = mb.same_family_gap(beta_long)
    assert beta_short.mstop == 6
    assert beta_early.material and beta_early.gap > 1.0
    assert beta_early.comparison == "beta regression"
    assert "Cross-validation stopped" in beta_early.message
    assert not beta_reached.material
    assert beta_reached.gap == pytest.approx(0.0, abs=1e-2)


def test_a_supplied_comparison_loglik_is_used_and_the_path_is_not_extended():
    model = _gaussian(mstop=5, nu=0.1)
    fitted = mb.same_family_gap(model)
    accepted = mb.same_family_gap(model, comparison_loglik=fitted.comparison_loglik)
    assert accepted.comparison == "supplied log-likelihood"
    assert accepted.gap == pytest.approx(fitted.gap)
    assert accepted.material
    assert model.mstop == 5
    with pytest.raises(ValueError, match="does not extend"):
        mb.same_family_gap(model, type("Cv", (), {"optimal_mstop": 40})())


def test_families_without_a_same_family_glm_say_so():
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    y = np.exp(0.2 + 0.3 * x)
    data = {"y": y, "x": x}
    laplace = mb.glmboost(
        "y ~ x", data=data, family=mb.Laplace(), control=mb.boost_control(8, 0.02),
    )
    refused = mb.same_family_gap(laplace)
    assert not refused.available
    assert not refused.material
    assert "not a log-likelihood" in refused.message

    smooth = mb.gamboost(
        "y ~ bbs(x)", data=data, family=mb.Gaussian(), control=mb.boost_control(15, 0.1),
    )
    basis = mb.same_family_gap(smooth)
    assert not basis.available
    assert "same basis" in basis.message
    frame = basis.to_frame()
    assert frame.columns == [
        "available", "family", "mstop", "mstop_source", "boost_loglik",
        "comparison_loglik", "gap", "gap_per_observation", "material",
        "comparison", "message", "tol", "n_obs",
    ]


def _rust_port_fixtures():
    rng = np.random.default_rng(821)
    n = 150
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    w = rng.integers(1, 4, size=n).astype(float)
    eta = 0.3 + 0.7 * x1 - 0.4 * x2
    ctl = mb.boost_control(30, 0.1)

    def fit(label, y, family, **kwargs):
        data = {"y": y, "x1": x1, "x2": x2}
        return label, mb.glmboost(
            "y ~ x1 + x2", data=data, family=family, control=ctl, **kwargs,
        )

    yield fit("Gaussian", eta + rng.normal(size=n), mb.Gaussian())
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    yield fit("Binomial", yb, mb.Binomial())
    yield fit("Binomial-w", yb, mb.Binomial(), weights=w)
    yp = rng.poisson(np.exp(eta)).astype(float)
    yield fit("Poisson", yp, mb.Poisson())
    yield fit("Poisson-w", yp, mb.Poisson(), weights=w)
    yield fit("GammaReg", rng.gamma(3.0, np.exp(eta) / 3.0), mb.GammaReg())
    ynb = rng.negative_binomial(4.0, 4.0 / (4.0 + np.exp(eta))).astype(float)
    yield fit("NBinomial", ynb, mb.NBinomial())
    u = rng.uniform(size=n)
    cat = (
        1 + (u > 1 / (1 + np.exp(0.2 + eta))).astype(int)
        + (u > 1 / (1 + np.exp(-0.6 + eta))).astype(int)
    )
    yield fit("PropOdds", cat.astype(float), mb.PropOdds(n_levels=3))
    mu = 1 / (1 + np.exp(-(0.2 + 0.8 * x1)))
    data = {"y": rng.beta(mu * 8, (1 - mu) * 8), "x1": x1, "x2": x2}
    yield "BetaLSS", mb.glmboostLSS(
        "y ~ x1 + x2", data=data, families=mb.BetaLSS(), control=ctl,
    )


# The comparison log-likelihoods the NumPy fitter returned on these fits
# before the fit moved to the Rust core (issue #821).
_NUMPY_COMPARISON_LOGLIK = {
    "Gaussian": -216.9073189805079,
    "Binomial": -90.9776531426489,
    "Binomial-w": -170.5957757855348,
    "Poisson": -253.64340103607566,
    "Poisson-w": -473.29306921993714,
    "GammaReg": -179.62463084954987,
    "NBinomial": -239.10011615534472,
    "PropOdds": -137.27783893390725,
    "BetaLSS": 74.9177186287759,
}


def test_the_core_fit_matches_the_numpy_fitter_it_replaced():
    seen = {}
    for label, model in _rust_port_fixtures():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            report = mb.same_family_gap(model)
        assert report.available, (label, report.message)
        seen[label] = report.comparison_loglik
    assert seen.keys() == _NUMPY_COMPARISON_LOGLIK.keys()
    for label, value in _NUMPY_COMPARISON_LOGLIK.items():
        assert seen[label] == pytest.approx(value, rel=1e-8), label
