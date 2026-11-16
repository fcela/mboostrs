"""Non-R smoke tests for by= (varying-coefficient) new-data prediction in
gamboostLSS.

Mirrors tests/test_by_parity/test_by_gamboost.py for the ordinary gamboost
by= newdata path, but exercises the gamboostLSS Rust `LearnerStore.predict_newdata`
route (BasisInfo::predict_into with a by_override), which previously raised
NotImplementedError for any learner using by=.
"""

import numpy as np
from mboostrs.gamlss import GaussianLSS, gamboostLSS


def _data(seed=42, n=200):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    z = rng.binomial(1, 0.5, n).astype(np.float64)
    w = rng.uniform(-2, 2, n)
    y = 3.0 + np.sin(x) * z + 0.5 * w + rng.normal(0, 0.5, n)
    return {"y": y, "x": x, "z": z, "w": w}


def _fit(data):
    return gamboostLSS(
        "y ~ bbs(x, df=4, by=z) + bols(w)",
        data=data,
        families=GaussianLSS(),
        control=(50, 0.1),
    )


def test_by_bbs_predict_newdata_no_longer_raises():
    data = _data()
    model = _fit(data)

    # A small newdata frame disjoint from the training rows.
    newdata = {
        "x": np.array([1.0, 3.0, 5.0, 7.0, 9.0]),
        "z": np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        "w": np.array([-1.0, 0.5, 0.0, 1.5, -0.5]),
    }
    for parameter in model.param_names:
        pred = model.predict(parameter=parameter, newdata=newdata)
        pred = np.asarray(pred, dtype=np.float64)
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))


def test_by_bbs_predict_newdata_z0_differs_from_z1():
    """Rows with z=0 must switch off the bbs(x, by=z) contribution: same x,
    different z should give different predictions for "mu", and the z=0 rows
    should equal the (by-independent) offset + bols(w) contribution only."""
    data = _data()
    model = _fit(data)

    x_vals = np.array([2.0, 4.0, 6.0, 8.0])
    w_vals = np.array([0.3, -0.3, 0.6, -0.6])
    newdata_z0 = {"x": x_vals, "z": np.zeros(4), "w": w_vals}
    newdata_z1 = {"x": x_vals, "z": np.ones(4), "w": w_vals}

    pred_z0 = np.asarray(model.predict(parameter="mu", newdata=newdata_z0), dtype=np.float64)
    pred_z1 = np.asarray(model.predict(parameter="mu", newdata=newdata_z1), dtype=np.float64)

    assert not np.allclose(pred_z0, pred_z1)

    # partial_effects for the bbs(x, by=z) learner must vanish for z=0 rows
    # (design = by * basis, and by=0 zeroes out the contribution) and be
    # nonzero for at least some z=1 rows.
    _, effect_z0 = model.partial_effects("mu", which="x", newdata=newdata_z0)
    _, effect_z1 = model.partial_effects("mu", which="x", newdata=newdata_z1)
    np.testing.assert_allclose(effect_z0, 0.0, atol=1e-10)
    assert np.any(np.abs(effect_z1) > 1e-8)


def test_by_bbs_partial_effects_newdata_matches_manual_prediction():
    """partial_effects(newdata=...) contributions for all learners should sum
    (with the offset) to the same value as predict(newdata=...)."""
    data = _data()
    model = _fit(data)

    newdata = {
        "x": np.array([1.0, 3.0, 5.0]),
        "z": np.array([1.0, 0.0, 1.0]),
        "w": np.array([0.2, -0.4, 0.6]),
    }
    pred = np.asarray(model.predict(parameter="mu", newdata=newdata, type="link"), dtype=np.float64)

    effects = model.partial_effects("mu", newdata=newdata)
    offset = model.coef("mu")["(offset)"]
    reconstructed = np.full(3, offset, dtype=np.float64)
    for _, effect in effects.values():
        reconstructed += np.asarray(effect, dtype=np.float64)

    np.testing.assert_allclose(reconstructed, pred, rtol=1e-10, atol=1e-10)
