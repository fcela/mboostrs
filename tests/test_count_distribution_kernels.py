"""Count-family CDFs and quantiles computed in the core, and the quantile
searches they share with the continuous families (issue #66).

PoissonLSS, NBinomialLSS, NegativeBinomialLSS, ZIPoLSS, ZINBLSS,
ZeroAdjustedPoissonLSS and BinomialLSS evaluated their CDFs by summing the
mass one count at a time in Python and their quantiles by walking that sum,
which refused means above a few thousand. The core computes the CDFs through
the regularized incomplete gamma and beta functions and finds each quantile
as the smallest k with a + b F(k) >= p (a = 0, b = 1 for the plain families;
the zero-inflated and zero-adjusted ones are affine in F).
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

FAMILIES = [
    (mb.PoissonLSS(), {"mu": 3.5}),
    (mb.NBinomialLSS(), {"mu": 3.5, "sigma": 0.7}),
    (mb.NegativeBinomialLSS(), {"mu": 3.5, "sigma": 0.7}),
    (mb.ZIPoLSS(), {"mu": 3.5, "sigma": 0.3}),
    (mb.ZINBLSS(), {"mu": 3.5, "sigma": 0.7, "nu": 0.3}),
    (mb.ZeroAdjustedPoissonLSS(), {"mu": 3.5, "sigma": 0.3}),
]


@pytest.mark.parametrize("family,params", FAMILIES, ids=lambda v: type(v).__name__)
def test_quantile_is_the_smallest_count_reaching_p(family, params):
    p = np.linspace(0.001, 0.999, 400)
    q = family.quantile(p, params)
    assert np.all(q == np.floor(q))
    assert np.all(family.cdf(q, params) >= p)
    below = q > 0
    assert np.all(family.cdf(q[below] - 1.0, params) < p[below])


@pytest.mark.parametrize("family,params", FAMILIES, ids=lambda v: type(v).__name__)
def test_cdf_is_the_cumulative_mass(family, params):
    k = np.arange(0, 60, dtype=float)
    np.testing.assert_allclose(family.cdf(k, params), np.cumsum(family.pdf(k, params)),
                               rtol=0, atol=1e-13)
    assert family.cdf(-1.0, params) == 0.0


def test_large_means_are_supported():
    """The summing helper refused a mean of 1e6; the core does not."""
    fam = mb.PoissonLSS()
    q = fam.quantile(np.array([0.025, 0.5, 0.975]), {"mu": 1.0e6})
    # R: qpois(c(0.025, 0.5, 0.975), 1e6)
    assert list(q) == [998041, 1000000, 1001960]


def test_binomial_matches_the_closed_form():
    fam = mb.BinomialLSS()
    k = np.arange(-1, 12, dtype=float)
    from math import comb

    exact = [sum(comb(10, j) * 0.3 ** j * 0.7 ** (10 - j) for j in range(int(kk) + 1))
             if kk >= 0 else 0.0 for kk in k]
    np.testing.assert_allclose(fam.cdf(k, {"mu": 0.3}, size=10), np.minimum(exact, 1.0),
                               atol=1e-14)
    assert list(fam.quantile(np.array([0.1, 0.5, 0.99]), {"mu": 0.3}, size=10)) == [1, 3, 7]


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_cdfs_and_quantiles_match_r():
    import rpy2.robjects as ro

    rng = np.random.default_rng(5)
    mu = np.exp(rng.uniform(np.log(0.05), np.log(2e5), 500))
    size = np.exp(rng.uniform(np.log(0.05), np.log(1e3), 500))
    k = np.floor(mu * np.exp(rng.normal(0, 0.2, 500)))
    p = rng.uniform(0.001, 0.999, 500)
    fv = ro.FloatVector
    np.testing.assert_allclose(mb.PoissonLSS().cdf(k, {"mu": mu}),
                               np.asarray(ro.r["ppois"](fv(k), fv(mu))), atol=1e-12)
    np.testing.assert_allclose(
        mb.NBinomialLSS().cdf(k, {"mu": mu, "sigma": size}),
        np.asarray(ro.r("function(q, s, m) pnbinom(q, size = s, mu = m)")(fv(k), fv(size), fv(mu))),
        atol=1e-11)
    assert np.array_equal(mb.PoissonLSS().quantile(p, {"mu": mu}),
                          np.asarray(ro.r["qpois"](fv(p), fv(mu))))
    assert np.array_equal(
        mb.NBinomialLSS().quantile(p, {"mu": mu, "sigma": size}),
        np.asarray(ro.r("function(p, s, m) qnbinom(p, size = s, mu = m)")(fv(p), fv(size), fv(mu))))


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_bcpe_quantile_and_inverse_gaussian_cdf_match_gamlss_dist():
    """Two defects found while moving the loops into the core. The BCPE
    quantile inverts a gamma quantile that is near 1e-13 for p close to 1/2
    and a large tau; the search stopped at an absolute width of 1e-12, so the
    BCPE quantile was off by up to 22% (pBCPE of it missed p by 0.08). The
    inverse-Gaussian CDF used a three-term Mills-ratio series from z = -8,
    whose truncation error (105 / z^8) put it 2.5e-7 from pIG."""
    import rpy2.robjects as ro

    ro.r("suppressMessages(library(gamlss.dist))")
    fv = ro.FloatVector
    rng = np.random.default_rng(2)
    n = 300
    mu = np.exp(rng.normal(1, 0.5, n))
    sigma = np.exp(rng.normal(-1.5, 0.4, n))
    nu = rng.choice([-1.5, -0.5, 0.0, 0.7, 2.0], n)
    tau = np.exp(rng.normal(1.5, 0.5, n))
    p = rng.uniform(0.01, 0.99, n)
    upstream = np.asarray(ro.r["qBCPE"](fv(p), mu=fv(mu), sigma=fv(sigma), nu=fv(nu), tau=fv(tau)))
    ours = mb.BCPELSS().quantile(p, {"mu": mu, "sigma": sigma, "nu": nu, "tau": tau})
    np.testing.assert_allclose(ours, upstream, rtol=1e-12)

    y = mu * np.exp(rng.normal(0, 0.3, n))
    s2 = np.exp(rng.normal(-1, 0.5, n))
    np.testing.assert_allclose(
        mb.InverseGaussianLSS().cdf(y, {"mu": mu, "sigma": s2}),
        np.asarray(ro.r["pIG"](fv(y), mu=fv(mu), sigma=fv(s2))), rtol=0, atol=1e-14)
    # qIG solves with uniroot at its default tolerance (about 1e-4), so its
    # own round trip misses p by up to 3e-5; the quantile is checked through
    # pIG instead.
    q = mb.InverseGaussianLSS().quantile(p, {"mu": mu, "sigma": s2})
    np.testing.assert_allclose(np.asarray(ro.r["pIG"](fv(q), mu=fv(mu), sigma=fv(s2))), p,
                               rtol=0, atol=1e-13)


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_gamma_and_beta_quantiles_match_r_in_the_tails():
    import rpy2.robjects as ro
    from mboostrs.gamlss import _special

    fv = ro.FloatVector
    rng = np.random.default_rng(1)
    n = 3000
    p = np.concatenate([rng.uniform(1e-12, 1e-3, n // 3), rng.uniform(0.001, 0.999, n // 3),
                        1 - rng.uniform(1e-12, 1e-3, n - 2 * (n // 3))])
    a = np.exp(rng.uniform(np.log(0.02), np.log(1e4), n))
    b = np.exp(rng.uniform(np.log(0.02), np.log(1e4), n))
    np.testing.assert_allclose(_special("gamma_quantile", p, a),
                               np.asarray(ro.r["qgamma"](fv(p), fv(a))), rtol=1e-10)
    np.testing.assert_allclose(_special("beta_quantile", p, a, b),
                               np.asarray(ro.r["qbeta"](fv(p), fv(a), fv(b))), rtol=1e-10)
