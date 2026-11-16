"""The interval study retains valid empty CQR sets without negative widths."""
import numpy as np
import pytest

from benches.study_nonmonotone_spread import score


def test_empty_set_width_coverage_and_two_quantile_score():
    # First set is empty, second covers, third misses by two response units.
    result = score(np.array([1., 0., 4.]), np.array([0., 2., 6.]),
                   {'y': np.array([.5, 1., 8.])}, np.ones(3), 'constant')
    assert result['empty_sets'] == 1
    assert result['max_crossing'] == 1
    assert result['width'] == pytest.approx(4/3)
    assert result['coverage'] == pytest.approx(1/3)
    # Scaled pinball sums are 19, 2 and 42, including the crossed pair.
    assert result['interval_score'] == 21
    assert result['width_scale_rank'] is None
