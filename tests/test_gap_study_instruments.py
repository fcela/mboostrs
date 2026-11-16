"""Study instruments exercise real public APIs before expensive measurement."""
import numpy as np

from benches.study_sharp_mean_controls import draw, oracle_diagnostic
from benches.study_poisson_tuning_consistency import draw as count_draw


def test_oracle_residual_control_uses_only_observed_predictor_candidates():
    frame, mean, _ = draw('axis_constant', 17, 160)
    result = oracle_diagnostic(frame, mean)
    assert set(result['by_column']) == {'x1', 'x2', 'x3', 'n0', 'n1'}
    assert all('t' in row and 'curve_removed' in row for row in result['by_column'].values())


def test_control_truth_and_count_support_are_retained():
    frame, mean, sd = draw('shifted_constant', 17, 160)
    expected = (3*np.sign(frame['x1'].to_numpy()-.25)*np.sign(frame['x2'].to_numpy()+.2)
                +frame['x3'].to_numpy())
    np.testing.assert_array_equal(mean, expected)
    np.testing.assert_array_equal(sd, np.full(160, .5))
    for panel in ('low', 'original', 'high', 'overdispersed', 'zero_inflated'):
        counts, mu, sigma = count_draw(panel, 17, 160)
        assert np.all(mu > 0) and np.all(sigma > 0)
        assert np.all(counts['y'].to_numpy() >= 0)
        assert np.all(counts['y'].to_numpy() == np.floor(counts['y'].to_numpy()))
