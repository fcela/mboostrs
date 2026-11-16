"""Independent score inversion and proper-training-only Gaussian calibration."""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust


def fixture():
    rng = np.random.default_rng(63011)
    x = rng.uniform(-1, 1, 240)
    return {"x": x, "y": np.sin(4 * x) + (0.3 + np.abs(x)) * rng.normal(size=len(x))}


def fit(data):
    return mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bbs(x, df=4)"},
        data,
        family=mb.GaussianLSS(),
        control=mb.boost_control(60),
    )


def test_native_matches_independent_score_set_and_affine_changes():
    rng = np.random.default_rng(63012)
    mu = rng.normal(size=80)
    sigma = np.exp(rng.normal(size=80))
    y = mu + sigma * rng.normal(size=80)
    expected = np.abs(y - mu) / sigma
    actual = _rust.conformal_location_scale_scores(y, mu, sigma)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
    q = np.sort(expected)[int(np.ceil(81 * 0.9)) - 1]
    lower, upper = _rust.conformal_location_scale_band(mu, sigma, q)
    np.testing.assert_array_equal(lower, mu - q * sigma)
    np.testing.assert_array_equal(upper, mu + q * sigma)
    np.testing.assert_array_equal(expected <= q, (np.asarray(lower) <= y) & (y <= np.asarray(upper)))
    shifted = _rust.conformal_location_scale_scores(7 + 3 * y, 7 + 3 * mu, 3 * sigma)
    np.testing.assert_allclose(shifted, expected, rtol=1e-14, atol=1e-14)


def test_public_held_out_matches_independent_parameters_and_scores():
    data = fixture()
    training = {key: value[:120] for key, value in data.items()}
    calibration = {key: value[120:] for key, value in data.items()}
    query = {"x": np.linspace(-0.8, 0.8, 12)}
    model = fit(training)
    cal = model.params(calibration, type="response", extrapolation="clip")
    pred = model.params(query, type="response", extrapolation="clip")
    score = np.abs(calibration["y"] - cal["mu"]) / cal["sigma"]
    q = np.sort(score)[int(np.ceil((len(score) + 1) * 0.9)) - 1]
    band = mb.conformal_interval(
        model, query, calibration=calibration, score="gaussian_normalized", extrapolation="clip"
    )
    np.testing.assert_allclose(band.lower, pred["mu"] - q * pred["sigma"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(band.upper, pred["mu"] + q * pred["sigma"], rtol=0, atol=1e-12)
    assert band.score == band.requested_score == "gaussian_normalized"
    assert band.quantile == pytest.approx(q)


def test_internal_split_equals_a_fit_on_the_known_proper_training_rows():
    data = fixture()
    model = fit(data)
    perm = np.random.default_rng(63013).permutation(240)
    held, training = perm[:120], perm[120:]
    proper = fit({key: value[training] for key, value in data.items()})
    calibration = {key: value[held] for key, value in data.items()}
    query = {"x": np.linspace(-0.7, 0.7, 9)}
    internal = mb.conformal_interval(model, query, score="gaussian_normalized", seed=63013, extrapolation="clip")
    external = mb.conformal_interval(
        proper, query, calibration=calibration, score="gaussian_normalized", extrapolation="clip"
    )
    np.testing.assert_allclose(internal.lower, external.lower, rtol=0, atol=1e-12)
    np.testing.assert_allclose(internal.upper, external.upper, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_native_rejects_invalid_scales(scale):
    with pytest.raises(ValueError, match="strictly positive finite scales"):
        _rust.conformal_location_scale_scores(np.zeros(2), np.zeros(2), np.array([1.0, scale]))


def test_score_scope_and_legacy_default_are_explicit():
    data = fixture()
    model = fit(data)
    query = {"x": np.array([0.0])}
    default = mb.conformal_interval(model, query, seed=63014)
    legacy = mb.conformal_interval(model, query, seed=63014, score="absolute")
    np.testing.assert_array_equal(default.lower, legacy.lower)
    standard = mb.gamboost("y ~ bbs(x)", data)
    with pytest.raises(ValueError, match="GaussianLSS"):
        mb.conformal_interval(standard, query, score="gaussian_normalized")
    transformed = mb.gamboostLSS(
        {"mu": "center(y) ~ bbs(x)", "sigma": "center(y) ~ bbs(x)"},
        data,
        family=mb.GaussianLSS(),
        control=mb.boost_control(20),
    )
    with pytest.raises(ValueError, match="untransformed"):
        mb.conformal_interval(transformed, query, score="gaussian_normalized")
