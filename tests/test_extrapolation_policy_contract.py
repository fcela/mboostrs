"""Strict numerical policies are independent of warning presentation."""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning, apply_policy, range_rows


@pytest.mark.parametrize("value", [-.04, 1.04, np.nextafter(0., -1.), np.nextafter(1., 2.)])
def test_error_rejects_every_representable_excursion(value):
    values = {0: np.array([value])}
    rows = range_rows([("bbs(x)", "x", values[0], 0., 1.)])
    with pytest.raises(ValueError, match="outside the range"):
        apply_policy(rows, values, "error")


def test_clip_applies_to_small_and_large_excursions_without_mutating_inputs():
    values = {0: np.array([-.04, .5, 1.04]), 1: np.array([-2., .5, 3.])}
    original = {k: v.copy() for k, v in values.items()}
    rows = range_rows([(f"bbs(x{k})", f"x{k}", v, 0., 1.) for k, v in values.items()])
    with pytest.warns(ExtrapolationWarning, match="clipped") as caught:
        clipped = apply_policy(rows, values, "clip")
    for k in values:
        np.testing.assert_array_equal(clipped[k], [0., .5, 1.])
        np.testing.assert_array_equal(values[k], original[k])
    # Only the material excursion contributes to the warning, but both clip.
    assert [r.variable for r in caught[0].message.rows] == ["x1"]


@pytest.mark.parametrize("policy", ["warn", "allow", "clip"])
def test_small_excursions_need_no_warning(policy):
    values = {0: np.array([-.04, 1.04])}
    rows = range_rows([("bbs(x)", "x", values[0], 0., 1.)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        result = apply_policy(rows, values, policy)
    np.testing.assert_array_equal(result[0], [0., 1.] if policy == "clip" else values[0])


def test_zero_span_extrapolation_is_reported_and_clipped():
    values = {0: np.array([1.9, 2., 2.1])}
    rows = range_rows([("bbs(x)", "x", values[0], 2., 2.)])
    assert rows[0].worst_distance == float("inf")
    with pytest.warns(ExtrapolationWarning):
        result = apply_policy(rows, values, "clip")
    np.testing.assert_array_equal(result[0], np.full(3, 2.))


def test_custom_warning_threshold_does_not_weaken_clip_or_error():
    values = {0: np.array([-1., 2.])}
    rows = range_rows([("bbs(x)", "x", values[0], 0., 1.)])
    with pytest.raises(ValueError, match="outside the range"):
        apply_policy(rows, values, "error", material=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        result = apply_policy(rows, values, "clip", material=100)
    np.testing.assert_array_equal(result[0], [0., 1.])


def test_prediction_clips_to_explicit_boundary_knots():
    x = np.linspace(0, 1, 100)
    fit = mb.gamboost("y ~ bbs(x, boundary.knots=c(-1, 2))",
                      data={"x": x, "y": np.sin(x)}, control=mb.boost_control(30))
    # Outside observed data, but inside the declared basis interval, is allowed.
    fit.predict(newdata={"x": np.array([1.5])}, extrapolation="error")
    query = {"x": np.array([-1.01, 2.01])}
    with pytest.raises(ValueError, match="outside the range"):
        fit.predict(newdata=query, extrapolation="error")
    actual = fit.predict(newdata=query, extrapolation="clip")
    expected = fit.predict(newdata={"x": np.array([-1., 2.])})
    np.testing.assert_allclose(actual, expected, atol=1e-12)
