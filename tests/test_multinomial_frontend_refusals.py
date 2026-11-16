"""Multinomial refusals name the route that works (issue #98, item 3).

`glmboost(family=Multinomial())` has no multinomial path: string labels were
told to "fit gamboost or glmboost with family=Multinomial()", which is what
the caller had done, and integer labels reached the core and failed with
"y must be 0 or 1 (dummy-coded)". mboost's glmboost refuses Multinomial too.
A two-class response in gamboost reported the core's "n_classes must be at
least 3, got 2" without saying that Binomial fits it.
"""

import numpy as np
import pytest

import mboostrs as mb


def _data(labels):
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    return {"y": np.asarray(labels)[rng.integers(0, len(labels), n)], "x": x}


@pytest.mark.parametrize("labels", [list("abc"), [0, 1, 2]])
def test_glmboost_refuses_multinomial_and_names_gamboost(labels):
    with pytest.raises(ValueError, match="glmboost does not fit family=Multinomial"):
        mb.glmboost("y ~ x", data=_data(labels), family=mb.Multinomial())


def test_two_classes_point_to_binomial():
    with pytest.raises(ValueError, match="family=Binomial"):
        mb.gamboost("y ~ bols(x)", data=_data(list("ab")), family=mb.Multinomial())


def test_string_response_message_no_longer_points_back_to_glmboost():
    with pytest.raises(ValueError) as info:
        mb.glmboost("y ~ x", data=_data(list("ab")))
    assert "or glmboost" not in str(info.value)
    assert "0/1" in str(info.value)


def test_three_classes_still_fit():
    fit = mb.gamboost("y ~ bols(x)", data=_data(list("abc")), family=mb.Multinomial(),
                      control=mb.boost_control(20))
    probabilities = np.asarray(fit.predict(type="response"))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
