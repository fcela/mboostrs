"""Cyclic bbs with a factor by= (issue #239).

A cyclic smooth modulated by a two-level factor must fit, and must match
the equivalent numeric 0/1 by= construction: the factor route builds one
wrapped basis per level, which is the indicator-weighted basis the numeric
route multiplies in directly.
"""

import numpy as np

import mboostrs as mb


def _cyclic_by_data(n=300, seed=0):
    """A periodic signal present only in level "b": the factor by= has to
    learn a flat zero in "a" and a sine in "b"."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, n)
    g = np.where(rng.random(n) < 0.5, "a", "b")
    y = np.sin(x) * (g == "b") + rng.normal(scale=0.2, size=n)
    return {
        "x": x, "g": g, "y": y,
        "g_a": (g == "a").astype(float),
        "g_b": (g == "b").astype(float),
    }


def test_cyclic_bbs_factor_by_fits():
    """The case #239 refused now builds and predicts."""
    data = _cyclic_by_data()
    fit = mb.gamboost(
        "y ~ bbs(x, cyclic=TRUE, df=4, by=g)", data=data,
        family=mb.Gaussian(), control=mb.boost_control(300, 0.1),
    )
    pred = np.asarray(fit.predict()).ravel()
    assert pred.shape == (300,)
    assert np.isfinite(pred).all()


def test_cyclic_bbs_factor_by_matches_numeric_by():
    """Two-level factor by= equals the per-level numeric 0/1 by= fit: the
    per-level wrapped bases span the same space as the indicator-weighted
    basis."""
    data = _cyclic_by_data()
    ctrl = mb.boost_control(300, 0.1)
    factor_fit = mb.gamboost(
        "y ~ bbs(x, cyclic=TRUE, df=4, by=g)", data=data,
        family=mb.Gaussian(), control=ctrl,
    )
    numeric_fit = mb.gamboost(
        "y ~ bbs(x, cyclic=TRUE, df=4, by=g_a)"
        " + bbs(x, cyclic=TRUE, df=4, by=g_b)",
        data=data, family=mb.Gaussian(), control=ctrl,
    )
    p_factor = np.asarray(factor_fit.predict()).ravel()
    p_numeric = np.asarray(numeric_fit.predict()).ravel()
    # Same model space, same data: the fits agree closely. They are not
    # bit-identical -- the factor route boosts one buser learner over both
    # levels while the numeric route boosts two separate learners -- so the
    # tolerance allows for the greedy path difference.
    assert np.allclose(p_factor, p_numeric, atol=0.3)
    assert np.corrcoef(p_factor, p_numeric)[0, 1] > 0.99


def test_cyclic_bbs_factor_by_predict_newdata():
    """Prediction on new data rebuilds the per-level cyclic bases."""
    data = _cyclic_by_data()
    fit = mb.gamboost(
        "y ~ bbs(x, cyclic=TRUE, df=4, by=g)", data=data,
        family=mb.Gaussian(), control=mb.boost_control(300, 0.1),
    )
    new = {
        "x": np.linspace(0, 2 * np.pi, 50),
        "g": np.array(["a", "b"] * 25),
    }
    pred = np.asarray(fit.predict(newdata=new)).ravel()
    assert pred.shape == (50,)
    assert np.isfinite(pred).all()
    # Level "a" carries no signal: its curve stays near zero while "b"
    # follows the sine.
    assert np.abs(pred[::2]).max() < 0.5
    assert np.abs(pred[1::2]).max() > 0.5
