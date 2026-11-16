"""Parity tests for by= varying-coefficient support vs R mboost."""

import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.gamboost import gamboost


def test_by_bols_gaussian_smoke():
    """Smoke test: bols with by= trains and predicts correctly."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.uniform(0, 10, n)
    z = rng.binomial(1, 0.5, n).astype(np.float64)
    y = 3.0 + 2.0 * x1 * z + rng.normal(0, 0.5, n)

    data = {"y": y, "x1": x1, "z": z}

    m = gamboost(
        "y ~ bols(x1, by=z, intercept=False)",
        data=data,
        family=Gaussian(),
        control=(50, 0.1),
    )

    assert m.mstop == 50
    assert len(m.fitted_) == n
    assert not np.any(np.isnan(m.fitted_))

    # Predict on same data should match fitted
    pred_new = m.predict(newdata=data)
    np.testing.assert_allclose(pred_new, m.fitted_, rtol=1e-10)

    # z=0 obs should all have the same fitted value
    # (because design = 0 - centering for all z=0 obs)
    z0 = z == 0
    z0_fitted = m.fitted_[z0]
    np.testing.assert_allclose(z0_fitted, z0_fitted[0], rtol=1e-14)

    # Predict at fewer iterations
    pred_half = m.predict(newdata=data, mstop=25)
    assert len(pred_half) == n


def test_by_bbs_gaussian_smoke():
    """Smoke test: bbs with by= trains and predicts correctly."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.uniform(0, 10, n)
    z = rng.binomial(1, 0.5, n).astype(np.float64)
    y = 3.0 + np.sin(x1) * z + rng.normal(0, 0.5, n)

    data = {"y": y, "x1": x1, "z": z}

    m = gamboost(
        "y ~ bbs(x1, by=z, df=4)",
        data=data,
        family=Gaussian(),
        control=(50, 0.1),
    )

    assert m.mstop == 50
    assert len(m.fitted_) == n
    assert not np.any(np.isnan(m.fitted_))

    pred_new = m.predict(newdata=data)
    np.testing.assert_allclose(pred_new, m.fitted_, rtol=1e-10)


def test_by_bols_vs_no_by():
    """bols with by=z should differ from bols without by."""
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.uniform(0, 10, n)
    z = rng.binomial(1, 0.5, n).astype(np.float64)
    y = 3.0 + 2.0 * x1 * z + rng.normal(0, 0.5, n)
    data = {"y": y, "x1": x1, "z": z}

    m_by = gamboost(
        "y ~ bols(x1, by=z, intercept=False)",
        data=data,
        family=Gaussian(),
        control=(20, 0.1),
    )
    m_no = gamboost(
        "y ~ bols(x1, intercept=False)",
        data=data,
        family=Gaussian(),
        control=(20, 0.1),
    )

    # With by=z, z=0 obs all have same fitted value; without by, they don't
    z0 = z == 0
    assert np.ptp(m_by.fitted_[z0]) < 1e-10
    assert np.ptp(m_no.fitted_[z0]) > 0.1


def test_by_bols_predict_newdata():
    """Predict on new data with by= should work correctly."""
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.uniform(0, 10, n)
    z = rng.binomial(1, 0.5, n).astype(np.float64)
    y = 3.0 + 2.0 * x1 * z + rng.normal(0, 0.5, n)
    data = {"y": y, "x1": x1, "z": z}

    m = gamboost(
        "y ~ bols(x1, by=z, intercept=False)",
        data=data,
        family=Gaussian(),
        control=(20, 0.1),
    )

    # New data where z=0 for all obs
    newdata_z0 = {"x1": np.array([1.0, 5.0, 9.0]), "z": np.array([0.0, 0.0, 0.0])}
    pred_z0 = m.predict(newdata=newdata_z0)
    # All predictions should be identical (design = 0 - centering)
    np.testing.assert_allclose(pred_z0, pred_z0[0], rtol=1e-10)

    # New data where z=1 for all obs
    newdata_z1 = {"x1": np.array([1.0, 5.0, 9.0]), "z": np.array([1.0, 1.0, 1.0])}
    pred_z1 = m.predict(newdata=newdata_z1)
    # Predictions should differ with x
    assert np.ptp(pred_z1) > 0.1


# --------------------------------------------------------------------------
# a factor by=
# --------------------------------------------------------------------------


def _factor_by_data(n=200, seed=3):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(0, 1, n)
    g = np.array(["a", "b", "c"])[rng.randint(0, 3, n)]
    slope = {"a": 0.0, "b": 2.0, "c": -2.0}
    y = np.array([slope[v] for v in g]) * x1 + rng.normal(0, 0.2, n)
    return {"y": y, "x1": x1, "g": g}


def test_factor_by_fits_one_block_per_non_reference_level():
    """R codes a factor `by` with treatment contrasts and stacks the design.

    The first level is the reference: its curve is zero for this term, and the
    other levels fit their difference from it. The design is therefore
    `(k - 1)` copies of the learner's own, which is what makes the term able to
    separate the group slopes at all — multiplying by integer level codes,
    which is what a factor `by` used to reach, cannot.
    """
    data = _factor_by_data()
    model = gamboost(
        "y ~ bols(x1, by=g)", data=data, family=Gaussian(), control=(300, 0.1)
    )
    # slope(a) = 0 (reference), slope(b) = 2, slope(c) = -2 at x1 = 1
    newdata = {"x1": np.array([1.0, 1.0, 1.0]), "g": np.array(["a", "b", "c"])}
    at_one = np.asarray(model.predict(newdata=newdata)).ravel()
    newdata_zero = {"x1": np.zeros(3), "g": np.array(["a", "b", "c"])}
    at_zero = np.asarray(model.predict(newdata=newdata_zero)).ravel()
    np.testing.assert_allclose(at_one - at_zero, [0.0, 2.0, -2.0], atol=0.15)


def test_factor_by_reports_the_term_the_user_wrote():
    data = _factor_by_data()
    model = gamboost(
        "y ~ bbs(x1, by=g, df=6) + bols(x1)", data=data, family=Gaussian(),
        control=(30, 0.1),
    )
    assert model._learner_names == ["bbs(x1, by=g)", "x1"]


def test_factor_by_survives_a_round_trip(tmp_path):
    """A model that cannot rebuild its design cannot predict on new data."""
    import pickle

    from mboostrs.serialize import load_model, save_model

    data = _factor_by_data()
    model = gamboost(
        "y ~ bbs(x1, by=g, df=6)", data=data, family=Gaussian(), control=(40, 0.1)
    )
    newdata = {"x1": np.linspace(0.1, 0.9, 5), "g": np.array(["a", "b", "c", "a", "b"])}
    expected = np.asarray(model.predict(newdata=newdata)).ravel()

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        np.asarray(restored.predict(newdata=newdata)).ravel(), expected, rtol=0, atol=0
    )

    path = save_model(model, tmp_path / "model.json")
    np.testing.assert_allclose(
        np.asarray(load_model(path).predict(newdata=newdata)).ravel(),
        expected, rtol=0, atol=0,
    )


def test_factor_by_is_not_confused_by_an_encoded_level_column():
    """The same factor as another learner's column must stay a factor here.

    `brandom(g)` makes the frame encode `g` to integer codes before the
    learners are built. A `by=` reading those codes multiplies the design by
    0, 1, 2 — the levels as a measurement — which is a wrong answer rather
    than a failure, so the factor-ness is recorded before the encoding.
    """
    data = _factor_by_data()
    encoded = gamboost(
        "y ~ brandom(g, df=2) + bols(x1, by=g)", data=data, family=Gaussian(),
        control=(200, 0.1),
    )
    plain = gamboost(
        "y ~ bols(x1, by=g)", data=data, family=Gaussian(), control=(200, 0.1)
    )
    newdata = {"x1": np.array([1.0, 1.0, 1.0]), "g": np.array(["a", "b", "c"])}
    # The by= term recovers the group slopes either way; reading level codes
    # would force one common slope scaled by 0, 1, 2.
    for model in (encoded, plain):
        at_one = np.asarray(model.predict(newdata=newdata)).ravel()
        zero = np.asarray(
            model.predict(newdata={"x1": np.zeros(3), "g": newdata["g"]})
        ).ravel()
        np.testing.assert_allclose(at_one - zero, [0.0, 2.0, -2.0], atol=0.2)


def test_factor_by_rejects_a_df_below_the_stacked_null_space():
    data = _factor_by_data()
    with pytest.raises(ValueError, match="unpenalized part"):
        gamboost("y ~ bbs(x1, by=g, df=3)", data=data, family=Gaussian(),
                 control=(10, 0.1))


def test_factor_by_on_an_unsupported_learner_is_refused():
    data = dict(_factor_by_data())
    data["x2"] = data["x1"] ** 2
    with pytest.raises(NotImplementedError, match="factor by="):
        gamboost("y ~ bspatial(x1, x2, by=g, df=6)", data=data,
                 family=Gaussian(), control=(10, 0.1))


def test_factor_by_is_refused_by_the_linear_frontend():
    # gamboostLSS fits a factor by= (issue #176, tests/test_lss_factor_by.py).
    from mboostrs.glmboost import glmboost

    data = _factor_by_data()
    with pytest.raises(NotImplementedError, match="factor by="):
        glmboost("y ~ bols(x1, by=g)", data=data, family=Gaussian(),
                 control=(10, 0.1))
