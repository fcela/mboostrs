"""Known count support preserves membership, including empty sets (#636)."""
import numpy as np
import polars as pl

from benches.study_poisson_conformal import count_metrics


def test_integer_boundaries_and_empty_sets_have_distinct_size_and_span():
    lower = np.array([-1.2, .1, 2., -2., 3.])
    upper = np.array([2.3, .9, 2., -1., 1.])
    observed = np.array([0, 1, 2, 0, 2])
    frame = pl.DataFrame({'y': observed})
    result = count_metrics(lower, upper, np.ones(5), frame, np.arange(1., 6.))
    assert result['count_coverage_identical']
    assert result['coverage'] == .4
    assert result['count_size'] == .8  # {0,1,2}, empty, {2}, empty, empty
    assert result['count_span'] == .4
    assert result['empty_count_sets'] == 3
    np.testing.assert_allclose(result['support_length'], .62)


def test_support_membership_at_every_nearby_integer():
    # Include exactly integral endpoints, a singleton, and endpoints that
    # leave no integer even though the real interval is nonempty.
    lower = np.array([-3., -.01, 0., .01, .9, 1., 1.01, 4.5, 6.])
    upper = np.array([-1., 0., 0., .99, 1., 1., 1.99, 4.9, 3.])
    for value in range(8):
        result = count_metrics(lower, upper, np.zeros(9),
            pl.DataFrame({'y': np.full(9, value)}), np.arange(1., 10.))
        assert result['count_coverage_identical']
