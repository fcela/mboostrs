"""`prediction_profile` conditions its support check on the fitted predictors (issue #75).

A supplied column the model does not use stays fixed while the profiled
variable is replaced. When it is derived from that variable it reports
unsupported combinations the model never evaluates (CF-19: an old-building
indicator beside construction year raised the far fraction from 0.07 to 0.79).
"""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    n = 400
    yearc = rng.uniform(1900, 2000, n)
    area = rng.normal(70, 20, n) + 0.1 * (yearc - 1950)
    data = {
        "yearc": yearc,
        "area": area,
        "old_1918": (yearc < 1920).astype(float),
        "y": 0.01 * area + 0.002 * (yearc - 1950) + rng.normal(0, 0.3, n),
    }
    model = mb.gamboost("y ~ bbs(yearc) + bbs(area)", data=data,
                        control=mb.boost_control(100))
    return model, data


def test_default_conditions_on_fitted_predictors_only(fitted):
    model, data = fitted
    profile = mb.prediction_profile(model, data, "yearc", grid=[1905, 1950, 1995])
    assert profile.support["conditioning_columns"] == ("area",)
    assert profile.support["excluded_columns"] == ("old_1918",)
    assert profile.support_fraction_far[0] < 0.1


def test_explicit_support_columns_are_honoured(fitted):
    model, data = fitted
    profile = mb.prediction_profile(model, data, "yearc", grid=[1905, 1950, 1995],
                                    support_columns=["area", "old_1918"])
    assert profile.support["conditioning_columns"] == ("area", "old_1918")
    assert profile.support_fraction_far[0] > 0.5


def test_unknown_support_column_is_refused(fitted):
    model, data = fitted
    with pytest.raises(ValueError, match="support_columns"):
        mb.prediction_profile(model, data, "yearc", support_columns=["nope"])


def test_dot_formula_uses_every_supplied_column(fitted):
    _, data = fitted
    model = mb.glmboost("y ~ .", data=data, control=mb.boost_control(20))
    profile = mb.prediction_profile(model, data, "yearc", grid=[1950])
    assert set(profile.support["conditioning_columns"]) == {"area", "old_1918"}
