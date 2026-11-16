"""The group-effect core centers an indicator design by one shared shift."""
import numpy as np
import pytest

from mboostrs import _rust


def test_effects_match_centered_indicator_design():
    beta = np.array([1., 3., -2.])
    centers = np.array([.1, .2, .7])
    expected = (np.eye(3) - centers) @ beta
    got = _rust.centered_group_effects(beta.tolist(), centers.tolist())
    np.testing.assert_allclose(got, expected, atol=1e-15)
    assert np.dot(centers, got) == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(np.diff(got), np.diff(beta), atol=1e-15)


def test_uncentered_and_never_selected_groups():
    assert _rust.centered_group_effects([1., 3.], []) == [1., 3.]
    assert _rust.centered_group_effects([0., 0.], [.3, .7]) == [0., 0.]
    assert _rust.centered_group_effects([], []) == []


@pytest.mark.parametrize("beta,centers", [([1., 2.], [.1]), ([np.nan], [1.])])
def test_invalid_group_dimensions_or_values_are_refused(beta, centers):
    with pytest.raises(ValueError):
        _rust.centered_group_effects(beta, centers)
