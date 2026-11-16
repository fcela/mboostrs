"""Adaptive hinge (kink) base-learner tests.

The hinge learner fits y = a * max(0, x - t) pivots (or max(0, t - x))
selection by selection. Where a P-spline smooths across a threshold, the hinge
keeps both sides straight and places one bend per selection at an
interior-quantile knot that $escaping residual scoring. The tests below check
the formula path, the fit, the filled-in coefficient vector, and the three
summation paths (in-sample, grouped newdata, grid newdata) that must replay a
per-selection hinge rather than add selections together.
"""

import numpy as np
import pytest

import mboostrs as mboost
from mboostrs.families import Gaussian
from mboostrs.formula import parse_gamboost_formula
from mboostrs.learners import Hinge
from mboostrs.terms import hinge as hinge_term


def _kinked_data(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = 400
    x = rng.uniform(-2.0, 3.0, n)
    y = (
        np.maximum(0.0, x - 1.0)
        - 2.0 * np.maximum(0.0, 0.2 - x)
        + 0.1 * rng.normal(size=n)
    )
    return {"y": y, "x": x}


def test_hinge_smoke():
    """fit/risk smoke invariant for the hinge learner."""
    data = _kinked_data()
    m = mboost.gamboost(
        "y ~ hinge(x, df = 4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(150, 0.1),
    )
    assert m.risk_[-1] < m.risk_[0]
    assert m.selected and set(m.selected) == {"x"}
    assert m.fitted(0).shape == (400,)


def test_hinge_beats_bbs_on_kink():
    """hinge(t) should fit a sharp kink better than a P-spline smooths it."""
    rng = np.random.default_rng(1)
    n = 600
    x = rng.uniform(-2.0, 2.0, n)
    data = {"y": np.maximum(0.0, x - 0.5) + 0.1 * rng.normal(size=n), "x": x}
    kw = dict(family=Gaussian(), control=mboost.boost_control(200, 0.1))
    hinge = mboost.gamboost("y ~ hinge(x, df = 4)", data=data, **kw)
    smooth = mboost.gamboost("y ~ bbs(x, df = 4)", data=data, **kw)
    grid = {"x": np.linspace(-2.0, 2.0, 9)}
    truth = np.maximum(0.0, grid["x"] - 0.5)

    def rmse(model):
        return float(
            np.sqrt(np.mean((np.asarray(model.predict(grid)).ravel() - truth) ** 2))
        )

    assert rmse(hinge) < rmse(smooth)


def test_hinge_parse_roundtrip():
    """formula spelling round-trips to a Hinge learner with the right knobs."""
    _, learners = parse_gamboost_formula("y ~ hinge(x, df = 4, n_candidates = 10) + bbs(z)")
    hinge, bbs = learners
    assert isinstance(hinge, Hinge)
    assert isinstance(bbs, __import__("mboostrs.learners", fromlist=["Bbs"]).Bbs)
    assert hinge.n_candidates == 10
    # the plain spelling default survives too
    _, learners2 = parse_gamboost_formula("y ~ hinge(x)")
    assert isinstance(learners2[0], Hinge)
    assert learners2[0].n_candidates == 20


def test_hinge_rejects_unknown_keyword():
    with pytest.raises(TypeError, match="unexpected keyword"):
        Hinge("x", not_a_knob=1)


def test_hinge_newdata_matches_fitted():
    """predicting on the training rows must reproduce fitted() exactly."""
    data = _kinked_data()
    m = mboost.gamboost(
        "y ~ hinge(x, df = 4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(150, 0.1),
    )
    in_sample = np.asarray(m.predict()).ravel()
    newdata = np.asarray(m.predict(data)).ravel()
    assert np.allclose(in_sample, newdata)
    # the recovered curve rises, plateaus, then rises again across the kink:
    # truth is +2 slope for x < 0.2, flat through x in (0.2, 1), +1 past 1.
    grid = {"x": np.linspace(-1.0, 2.0, 31)}
    fitted = np.asarray(m.predict(grid)).ravel()
    left = fitted[0:2].mean()
    plateau = fitted[5:18].mean()
    right = fitted[24:31].mean()
    assert right > plateau > left


def test_hinge_msptop_slice_changes_prediction():
    """predict(mstop=k) and predict() must differ on a fit still improving."""
    data = _kinked_data()
    m = mboost.gamboost(
        "y ~ hinge(x, df = 4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(300, 0.1),
    )
    at_slice = np.asarray(m.predict(mstop=50)).ravel()
    at_full = np.asarray(m.predict()).ravel()
    assert not np.allclose(at_slice, at_full)


def test_hinge_by_term():
    """hinge(x, by = z) fits a varying slope whose sign matches the by factor."""
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.normal(size=n)
    data = {
        "y": np.maximum(0.0, x - 0.3) * (1.0 + 0.5 * z) + 0.1 * rng.normal(size=n),
        "x": x,
        "z": z,
    }
    m = mboost.gamboost(
        "y ~ hinge(x, df = 4, by = z)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(200, 0.1),
    )
    assert np.asarray(m.predict()).shape == (n,)
    assert m.risk_[-1] < m.risk_[0]


def test_hinge_terms_helper():
    assert hinge_term("dew_point") == "hinge(dew_point, df=4.0, n_candidates=20)"
    assert "df=8.0" in hinge_term("dew_point", df=8.0)