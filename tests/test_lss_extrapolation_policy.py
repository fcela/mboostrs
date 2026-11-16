"""gamboostLSS prediction honours `extrapolation=` (issue #67).

LSS prediction took no `extrapolation` argument and continued every spline
basis at its edge slope silently, 50% beyond the training range included.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = np.sin(3 * x) + rng.normal(size=200) * 0.3
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"},
                         data={"x": x, "y": y}, family=mb.GaussianLSS(), mstop=50)
    return fit, x


def test_default_warns_for_every_parameter(model):
    fit, _ = model
    with pytest.warns(ExtrapolationWarning) as caught:
        fit.predict(newdata={"x": np.array([1.5])})
    text = " ".join(str(w.message) for w in caught)
    assert "[mu]" in text and "[sigma]" in text


def test_error_refuses_a_small_excursion(model):
    fit, x = model
    with pytest.raises(ValueError, match="outside the range"):
        fit.predict(newdata={"x": np.array([x.max() + 0.04])}, extrapolation="error")


def test_clip_equals_the_boundary_prediction(model):
    fit, x = model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped = fit.predict(newdata={"x": np.array([1.5])}, extrapolation="clip")
    boundary = fit.predict(newdata={"x": np.array([x.max()])})
    for name in clipped:
        np.testing.assert_allclose(clipped[name], boundary[name])


def test_inside_the_range_is_quiet_and_unknown_policy_refused(model):
    fit, _ = model
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        fit.predict(newdata={"x": np.array([0.5])}, extrapolation="error")
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        fit.predict(newdata={"x": np.array([0.5])}, extrapolation="nope")
