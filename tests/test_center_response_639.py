"""Original-response quantiles from a centered response (#639)."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

import mboostrs as mb
from mboostrs.transforms import response_forward, response_inverse


@pytest.fixture
def centered():
    rng = np.random.default_rng(639)
    x = rng.uniform(-1, 1, 200)
    data = {"x": x, "y": 4 + x + rng.normal(scale=.3, size=200)}
    model = mb.gamboostLSS({"mu": "center(y) ~ bols(x)", "sigma": "center(y) ~ bols(x)"},
                          data, family=mb.GaussianLSS(), control=mb.boost_control(40))
    return model, data


def test_center_response_round_trip(centered):
    model, data = centered
    step = model._response_transform
    assert len(step.params) == 2
    np.testing.assert_allclose(response_inverse(step, response_forward(step, data["y"])), data["y"])


def test_legacy_single_mean_inverse(centered):
    model, _ = centered
    step = replace(model._response_transform, params=(model._response_transform.params[0],))
    np.testing.assert_allclose(response_inverse(step, np.array([0., 1.])),
                               np.array([0., 1.])+step.params[0])


def test_original_quantiles_match_gaussian_reference(centered):
    model, data = centered
    query = {"x": np.array([-.5, 0., .5])}
    params = model.params(query)
    interval = model.prediction_interval(level=.9, newdata=query)
    for probability, name in ((.05, "lower"), (.5, "median"), (.95, "upper")):
        expected = params["mu"] + norm.ppf(probability)*params["sigma"] + data["y"].mean()
        np.testing.assert_allclose(np.asarray(interval[name]), expected, atol=1e-12)
