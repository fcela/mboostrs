"""Discovery reports the fitted scale beside the mean (issue #589).

``result.predictions`` carried ``row``, ``mean``, ``lower`` and ``upper``
only, so a caller scoring a location-scale fit rebuilt the scale from the
interval band as ``(upper - lower) / (2 * z_level)``. That band is a
conformal widening of a refit's central quantiles, not ``mean +/- z * sigma``
of the final model, so the rebuild was never the fitted ``sigma``. The frame
now carries ``sigma`` for a Gaussian location-scale fit, and only then.
"""

import json
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _hetero(seed=0, n=400):
    """A mean driven by x, a noise scale that falls as the positive h rises.

    h is right-skewed, so discover drafts it on log(h) and the spread stage
    adopts a log-scale term for it.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    sd = 0.15 + 0.6 * np.sqrt(np.exp(5.0) / h)
    y = 0.6 * x + sd * rng.normal(size=n)
    return pl.DataFrame({"x": x, "h": h, "y": y})


@pytest.fixture(scope="module")
def spread():
    frame = _hetero()
    new = _hetero(seed=9, n=24).drop("y")
    result = mb.discover(frame, "y", newdata=new, seed=0)
    return result, new


def test_spread_result_predictions_carry_the_fitted_scale(spread):
    result, _ = spread
    assert result.spread_formula is not None
    preds = result.predictions
    assert preds.columns == ["row", "mean", "sigma", "lower", "upper"]
    assert (preds["sigma"] > 0).all() and preds["sigma"].is_finite().all()
    # A location-scale fit: every row has its own scale.
    assert preds["sigma"].n_unique() == preds.height


def test_sigma_is_the_final_model_own_fitted_scale(spread):
    result, new = spread
    sigma = result.predictions["sigma"].to_numpy()
    fitted = result.model.predict(new, parameter="sigma", type="response",
                                  extrapolation="clip")
    assert np.allclose(sigma, np.asarray(fitted, dtype=float).ravel())


def test_sigma_is_not_the_scale_rebuilt_from_the_interval_band(spread):
    # The old workaround: rebuild sigma from the conformal band. The band
    # belongs to a refit calibrated on a training split and is widened by an
    # empirical quantile, so it does not reproduce the fitted scale.
    result, _ = spread
    preds = result.predictions
    z = NormalDist().inv_cdf(0.5 + result.level / 2)
    band = (preds["upper"].to_numpy() - preds["lower"].to_numpy()) / (2 * z)
    assert not np.allclose(band, preds["sigma"].to_numpy(), rtol=1e-6)


def test_sigma_tracks_the_true_scale_of_each_row(spread):
    result, new = spread
    sigma = result.predictions["sigma"].to_numpy()
    order = np.argsort(new["h"].to_numpy())
    third = len(order) // 3
    low_h = sigma[order[:third]]   # true scale near 1.4
    high_h = sigma[order[-third:]]  # true scale near 0.43
    assert low_h.mean() > 1.5 * high_h.mean()


def test_to_dict_carries_the_fitted_scale(spread):
    record = json.loads(json.dumps(spread[0].to_dict()))
    row = record["predictions"][0]
    assert set(row) == {"row", "mean", "sigma", "lower", "upper"}
    assert row["sigma"] == pytest.approx(spread[0].predictions["sigma"][0])


def test_a_constant_scale_fit_has_no_sigma_column():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    y = 0.6 * x + 0.5 * rng.normal(size=n)
    new = pl.DataFrame({"x": [0.0, 1.0], "h": [np.exp(5.0), np.exp(5.0)]})
    result = mb.discover(pl.DataFrame({"x": x, "h": h, "y": y}), "y",
                         newdata=new, seed=0, spread=False, interactions=False)

    assert result.spread_formula is None
    assert result.predictions.columns == ["row", "mean", "lower", "upper"]


def test_a_family_without_a_distributional_counterpart_has_no_sigma():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * x))).astype(float)
    new = pl.DataFrame({"x": [0.0, 1.0], "z": [0.0, 0.0]})
    result = mb.discover(pl.DataFrame({"x": x, "z": z, "y": y}), "y",
                         newdata=new, seed=0, interactions=False)

    assert result.family == "Binomial"
    assert result.spread_formula is None
    assert result.predictions.columns == ["row", "mean", "lower", "upper"]
