"""Special functions computed by the core (issue #66, items 3 and 7).

The distribution helpers' incomplete gamma and beta functions, Student-t CDF
and quantile, gamma and beta quantiles and the normal CDF, quantile and
log-CDF are `mboostrs_core::special`, reached through `gamlss._special`.
Each is checked against SciPy here, and an iteration limit reached before the
tolerance is reported as a warning instead of a silent value.
"""

import warnings

import numpy as np
import pytest
from scipy import special, stats

import mboostrs as mb
from mboostrs.gamlss import _special

rng = np.random.default_rng(0)
A = rng.uniform(0.2, 20, 400)
B = rng.uniform(0.2, 20, 400)
X = rng.uniform(0.0, 30, 400)
U = rng.uniform(1e-6, 1 - 1e-6, 400)
DF = rng.uniform(0.5, 60, 400)


def test_incomplete_gamma_and_beta():
    np.testing.assert_allclose(_special("gamma_p", A, X), special.gammainc(A, X),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(_special("beta_i", A, B, U), special.betainc(A, B, U),
                               rtol=1e-10, atol=1e-14)


def test_student_t_cdf_and_quantile():
    t = rng.normal(0, 4, 400)
    np.testing.assert_allclose(_special("t_cdf", t, DF), stats.t.cdf(t, DF),
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(_special("t_quantile", U, DF), stats.t.ppf(U, DF),
                               rtol=1e-9, atol=1e-10)


def test_gamma_and_beta_quantiles():
    np.testing.assert_allclose(_special("gamma_quantile", U, A), stats.gamma.ppf(U, A),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_special("beta_quantile", U, A, B), stats.beta.ppf(U, A, B),
                               rtol=1e-9, atol=1e-12)


def test_normal_functions():
    z = np.concatenate([rng.normal(0, 3, 300), [-40.0, -12.0, 9.0]])
    # libm's erfc is within 3e-14 relative in the far tail (z = -12).
    np.testing.assert_allclose(_special("normal_cdf", z), stats.norm.cdf(z), rtol=1e-13,
                               atol=1e-300)
    np.testing.assert_allclose(_special("normal_log_cdf", z), stats.norm.logcdf(z),
                               rtol=1e-8)
    p = np.concatenate([U, [1e-300, 1 - 1e-16]])
    np.testing.assert_allclose(_special("normal_quantile", p), stats.norm.ppf(p),
                               rtol=1e-14)


def test_broadcasting_and_shape():
    out = _special("gamma_p", np.array([[1.0], [2.0]]), np.array([0.5, 1.0, 2.0]))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, special.gammainc([[1.0], [2.0]], [0.5, 1.0, 2.0]),
                               rtol=1e-13)


def test_an_iteration_limit_is_reported():
    with pytest.warns(RuntimeWarning, match="iteration limit"):
        # The iteration limit grows with sqrt(a) up to a cap; a shape of 1e14
        # needs about 1e7 series terms, past it.
        _special("gamma_p", 1.0e14, 1.0e14 - 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _special("gamma_p", 3.0, 2.0)


@pytest.mark.parametrize("family,params", [
    (mb.GammaLSS(), {"mu": 2.0, "sigma": 3.0}),
    (mb.StudentTLSS(), {"mu": 1.0, "sigma": 2.0, "df": 4.0}),
    (mb.BetaLSS(), {"mu": 0.3, "phi": 7.0}),
    (mb.ZeroAdjustedGammaLSS(), {"mu": 2.0, "sigma": 0.6, "nu": 0.2}),
])
def test_family_quantiles_invert_their_cdfs(family, params):
    p = np.array([0.25, 0.5, 0.9, 0.99])
    q = family.quantile(p, params)
    np.testing.assert_allclose(family.cdf(q, params), p, atol=1e-9)
