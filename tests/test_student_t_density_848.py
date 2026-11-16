"""Student-t density helpers use the fitting likelihood without gamma overflow."""
import numpy as np
import pytest
from scipy.stats import t

from mboostrs import StudentTLSS


@pytest.mark.parametrize("df", [0.5, 1, 4, 340, 400, 1000, 100000])
def test_student_t_density_matches_scipy(df):
    y = np.array([-30., -2., 0., 1., 12.])
    got = StudentTLSS().pdf(y, {"mu": 1., "sigma": 2., "df": df})
    expected = t.pdf((y - 1) / 2, df) / 2
    np.testing.assert_allclose(got, expected, rtol=2e-9, atol=1e-14)
    assert np.isfinite(got).all()


def test_student_t_density_broadcasts_and_preserves_shape():
    y = np.array([[-2.], [2.]])
    df = np.array([4., 400., 1000.])
    got = StudentTLSS().pdf(y, {"mu": 0., "sigma": 1., "df": df})
    np.testing.assert_allclose(got, t.pdf(y, df), rtol=1e-11)
    assert got.shape == (2, 3)
    assert StudentTLSS().pdf([], {"mu": 0., "sigma": 1., "df": 400.}).size == 0


@pytest.mark.parametrize("parameter", ["sigma", "df"])
def test_student_t_density_refuses_nonpositive_parameters(parameter):
    params = {"mu": 0., "sigma": 1., "df": 400.}
    params[parameter] = 0
    with pytest.raises(ValueError, match="requires sigma"):
        StudentTLSS().pdf([0.], params)
