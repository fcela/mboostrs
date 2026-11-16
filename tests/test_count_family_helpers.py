"""Count-family helpers describe the distribution the fit estimates (issue #66).

`NegativeBinomialLSS` and `ZINBLSS` fit gamlss.dist NBI / ZINBI, whose
`sigma` is the dispersion (variance `mu + sigma * mu**2`, size `1/sigma`).
Their `pdf` used that size, but `pmf`, `cdf`, `quantile` and `simulate` (and
the R helpers) used the size `sigma`: at mu = 4, sigma = 0.5 the CDF at 0 was
0.333 instead of 0.111, and simulated counts had variance 36 instead of 12.
Count masses were also interpolated between integers (Poisson `pdf(0.5)` was
0.216) and raised at negative integers.
"""

import numpy as np
import pytest
from scipy import stats

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

COUNT_FAMILIES = {
    "PoissonLSS": {"mu": 4.0},
    "NBinomialLSS": {"mu": 4.0, "sigma": 0.5},
    "NegativeBinomialLSS": {"mu": 4.0, "sigma": 0.5},
    "GeometricLSS": {"mu": 4.0},
    "ZIPoLSS": {"mu": 4.0, "sigma": 0.3},
    "ZINBLSS": {"mu": 4.0, "sigma": 0.5, "nu": 0.3},
}
K = np.arange(0.0, 800.0)


@pytest.mark.parametrize("name", sorted(COUNT_FAMILIES))
def test_mass_cdf_quantile_and_draws_agree(name):
    family = getattr(mb, name)()
    params = COUNT_FAMILIES[name]
    mass = family.pdf(K, params)
    assert mass.sum() == pytest.approx(1.0, abs=1e-10)
    if hasattr(family, "pmf"):
        np.testing.assert_allclose(family.pmf(K[:20], params), mass[:20], rtol=1e-12)
    np.testing.assert_allclose(family.cdf(K[:20], params), np.cumsum(mass[:20]),
                               rtol=1e-10, atol=1e-14)
    probabilities = np.array([0.05, 0.3, 0.5, 0.8, 0.97])
    q = family.quantile(probabilities, params)
    assert np.all(family.cdf(q - 1, params) < probabilities)
    assert np.all(probabilities <= family.cdf(q, params) + 1e-12)
    mean = float((K * mass).sum())
    variance = float(((K - mean) ** 2 * mass).sum())
    draws = family.simulate({k: np.full(200_000, v) for k, v in params.items()},
                            rng=np.random.default_rng(0))
    assert draws.mean() == pytest.approx(mean, rel=0.02)
    assert draws.var() == pytest.approx(variance, rel=0.05)


@pytest.mark.parametrize("name", sorted(COUNT_FAMILIES))
def test_mass_is_zero_off_the_non_negative_integers(name):
    family = getattr(mb, name)()
    mass = family.pdf(np.array([-2.0, -1.0, 0.5, 2.5]), COUNT_FAMILIES[name])
    np.testing.assert_array_equal(mass, 0.0)


def test_binomial_mass_only_at_its_codings():
    family = mb.BinomialLSS()
    mass = family.pdf(np.array([-1.0, 0.0, 1.0, 0.5, 2.0]), {"mu": 0.3})
    np.testing.assert_allclose(mass, [0.7, 0.7, 0.3, 0.0, 0.0])


def test_nbi_and_zinbi_against_scipy():
    k = np.arange(0.0, 15.0)
    size = 1.0 / 0.5
    nbi = stats.nbinom(size, size / (size + 4.0))
    family = mb.NegativeBinomialLSS()
    params = {"mu": 4.0, "sigma": 0.5}
    np.testing.assert_allclose(family.pmf(k, params), nbi.pmf(k), rtol=1e-12)
    np.testing.assert_allclose(family.cdf(k, params), nbi.cdf(k), rtol=1e-12)
    zinb = mb.ZINBLSS()
    expected = 0.7 * nbi.pmf(k) + np.where(k == 0, 0.3, 0.0)
    np.testing.assert_allclose(zinb.pmf(k, {**params, "nu": 0.3}), expected, rtol=1e-12)


def test_inverse_gaussian_density_is_zero_below_support():
    density = mb.InverseGaussianLSS().pdf(np.array([-1.0, 0.0, 1.0]),
                                          {"mu": 2.0, "sigma": 0.5})
    assert density[0] == 0.0 and density[1] == 0.0
    assert density[2] == pytest.approx(0.4839414, rel=1e-6)  # gamlss.dist::dIG


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_nb_zinb_and_pareto2_against_gamlss_dist():
    import rpy2.robjects as ro

    ro.r("suppressMessages(library(gamlss.dist))")
    k = ro.FloatVector(list(range(12)))
    ks = np.arange(12.0)
    params = {"mu": 4.0, "sigma": 0.5}
    np.testing.assert_allclose(
        mb.NegativeBinomialLSS().cdf(ks, params),
        np.asarray(ro.r["pNBI"](k, mu=4.0, sigma=0.5)), rtol=1e-12)
    np.testing.assert_allclose(
        mb.ZINBLSS().cdf(ks, {**params, "nu": 0.3}),
        np.asarray(ro.r["pZINBI"](k, mu=4.0, sigma=0.5, nu=0.3)), rtol=1e-12)
    y = np.array([0.5, 1.0, 2.0, 5.0])
    # Pareto2LSS's shape is the reciprocal of gamlss PARETO2's sigma.
    np.testing.assert_allclose(
        mb.Pareto2LSS().pdf(y, params),
        np.asarray(ro.r["dPARETO2"](ro.FloatVector(y), mu=4.0, sigma=2.0)), rtol=1e-12)
