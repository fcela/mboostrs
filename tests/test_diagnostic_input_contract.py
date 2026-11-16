"""Diagnostic summaries reject inputs that would change their meaning."""

import numpy as np
import pytest

from mboostrs.diagnostics import coverage, interval_metrics, ipc_weights


@pytest.mark.parametrize("observed,lower,upper", [
    ([0., 10.], [[0.], [10.]], [[0.], [10.]]),
    ([0., 1.], [-1.], [2.]),
    (0., -1., 1.),
    ([], [], []),
])
def test_coverage_rejects_broadcasting_and_empty_inputs(observed, lower, upper):
    with pytest.raises(ValueError, match="non-empty.*one-dimensional.*same shape"):
        coverage(observed, lower, upper)


@pytest.mark.parametrize("observed,lower,upper", [
    ([np.nan], [0.], [1.]),
    ([np.inf], [0.], [1.]),
    ([0.], [np.nan], [1.]),
    ([0.], [-1.], [np.nan]),
    ([0.], [1.], [-1.]),
    ([0.], [np.inf], [np.inf]),
    ([0.], [-np.inf], [-np.inf]),
])
def test_coverage_rejects_nonfinite_observations_and_invalid_bounds(observed, lower, upper):
    with pytest.raises(ValueError, match="observed|bound"):
        coverage(observed, lower, upper)


def test_unbounded_intervals_and_inclusive_endpoints_are_supported():
    assert coverage([0., 1.], [-np.inf, 1.], [0., np.inf]) == (1., np.inf)
    assert coverage([0., 1., 2.], [0., 0., 0.], [0., .5, 2.]) == pytest.approx((2 / 3, 2.5 / 3))


def test_weighted_interval_metrics_match_the_proper_score_formula():
    observed = np.array([0., 2., 5.])
    lower = np.array([0., 1., 2.])
    upper = np.array([1., 3., 4.])
    weights = np.array([1., 2., 1.])
    result = interval_metrics(
        observed, lower, upper, level=.8, weights=weights, retain_scores=True
    )
    expected_scores = np.array([1., 2., 12.])
    np.testing.assert_allclose(result["interval_scores"], expected_scores)
    assert result["interval_scores"].flags.writeable is False
    assert result["empirical_coverage"] == pytest.approx(3 / 4)
    assert result["lower_miss_rate"] == 0
    assert result["upper_miss_rate"] == pytest.approx(1 / 4)
    assert result["mean_width"] == pytest.approx(7 / 4)
    assert result["mean_interval_score"] == pytest.approx(17 / 4)
    assert result["nominal_coverage"] == .8


@pytest.mark.parametrize("level,weights,message", [
    (0, None, "level"),
    (1, None, "level"),
    (.9, [1, -1, 1], "weights"),
    (.9, [1, 1], "weights length"),
])
def test_interval_metrics_validate_level_and_weights(level, weights, message):
    with pytest.raises(ValueError, match=message):
        interval_metrics([0, 1, 2], [-1, 0, 1], [1, 2, 3],
                         level=level, weights=weights)


@pytest.mark.parametrize("maxweight", [-1., 0., np.nan, np.inf, True])
def test_ipc_clip_limit_must_be_positive_and_finite(maxweight):
    with pytest.raises((ValueError, TypeError), match="maxweight"):
        ipc_weights([1., 2.], [1., 0.], maxweight=maxweight)


@pytest.mark.parametrize("time,status", [([], []), (1., 1.), ([[1.], [2.]], [[1.], [0.]])])
def test_ipc_requires_nonempty_vectors(time, status):
    with pytest.raises(ValueError, match="non-empty.*one-dimensional"):
        ipc_weights(time, status)
