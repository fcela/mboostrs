"""Support, atoms, PIT, and mean calibration for mixed LSS distributions."""

import math

import numpy as np
import pytest

import mboostrs as mb


PARAMS = {"mu": np.array([2., 2., 2.]), "sigma": np.array([.5, .5, .5]),
          "nu": np.array([.3, .3, .3])}


def test_zero_adjusted_gamma_is_zero_below_its_support():
    family = mb.ZeroAdjustedGammaLSS()
    y = np.array([-1., np.nextafter(0., -1.), 0.])
    np.testing.assert_array_equal(family.pdf(y, PARAMS), [0., 0., .3])
    np.testing.assert_array_equal(family.cdf(y, PARAMS), [0., 0., .3])
    np.testing.assert_array_equal(family.cdf_left(y, PARAMS), [0., 0., 0.])


def fitted_zero_adjusted_gamma(seed=9, n=500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    is_zero = rng.random(n) < .35
    y = np.where(is_zero, 0., rng.gamma(4., .5, size=n))
    return mb.glmboostLSS(
        {"mu": "y ~ x", "sigma": "y ~ x", "nu": "y ~ x"},
        data={"x": x, "y": y}, family=mb.ZeroAdjustedGammaLSS(),
        control=mb.boost_control(20),
    ), is_zero


def test_pit_randomizes_only_the_point_mass():
    fit, is_zero = fitted_zero_adjusted_gamma()
    first = fit.pit(random_state=1)
    second = fit.pit(random_state=2)
    assert np.max(np.abs(first[is_zero] - second[is_zero])) > .05
    np.testing.assert_array_equal(first[~is_zero], second[~is_zero])
    deterministic = fit.pit(randomized=False)
    params = fit.params(type="response")
    np.testing.assert_allclose(deterministic[is_zero], .5 * params["nu"][is_zero])


def test_randomized_pit_of_generated_mixture_is_uniform():
    rng = np.random.default_rng(11)
    n = 10_000
    family = mb.ZeroAdjustedGammaLSS()
    params = {"mu": np.full(n, 1.6), "sigma": np.full(n, .8), "nu": np.full(n, .3)}
    zero = rng.random(n) < .3
    y = np.where(zero, 0., rng.gamma(1 / .8**2, 1.6 * .8**2, size=n))
    upper = family.cdf(y, params)
    lower = family.cdf_left(y, params)
    u = rng.uniform(lower, upper)
    ks = np.max(np.abs(np.sort(u) - (np.arange(1, n + 1) - .5) / n))
    assert ks < 1.36 / np.sqrt(n)


def test_calibration_uses_unconditional_mixture_mean():
    fit, _ = fitted_zero_adjusted_gamma()
    params = fit.params(type="response")
    expected = (1 - params["nu"]) * params["mu"]
    rows = fit.calibration_frame(bins=1, output="records")
    assert rows[0]["fitted_mean"] == pytest.approx(expected.mean())
    assert rows[0]["fitted_mean"] != pytest.approx(params["mu"].mean(), rel=1e-3)


def test_gumbel_calibration_uses_mean_not_location():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    y = .5 + 2 * rng.gumbel(size=300)
    fit = mb.glmboostLSS(
        {"mu": "y ~ x", "sigma": "y ~ x"}, data={"x": x, "y": y},
        family=mb.GumbelLSS(), control=mb.boost_control(10),
    )
    params = fit.params(type="response")
    row = fit.calibration_frame(bins=1, output="records")[0]
    assert row["fitted_mean"] == pytest.approx(
        np.mean(params["mu"] + np.euler_gamma * params["sigma"])
    )


def test_undefined_student_mean_is_refused():
    family = mb.StudentTLSS()
    with pytest.raises(ValueError, match="no finite mean"):
        family.mean({"mu": np.array([0.]), "sigma": np.array([1.]), "df": np.array([1.])})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_quantile_probabilities_must_be_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        mb.ZeroAdjustedGammaLSS().quantile(
            [bad], {"mu": np.array([1.]), "sigma": np.array([1.]), "nu": np.array([.3])}
        )


def test_logistic_density_is_finite_in_both_extreme_tails():
    family = mb.LogisticLSS()
    density = family.pdf([-1000., 1000.], {"mu": 0., "sigma": 1.})
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0)


@pytest.mark.parametrize("family,params,expected", [
    (mb.GaussianKnownVarLSS(), {"mu": [1., 2.], "sigma": [.5, 1.]}, [1., 2.]),
    (mb.BinomialLSS(), {"mu": [.2, .8]}, [.2, .8]),
    (mb.PoissonLSS(), {"mu": [2., 4.]}, [2., 4.]),
    (mb.GeometricLSS(), {"mu": [2., 4.]}, [2., 4.]),
    (mb.NBinomialLSS(), {"mu": [2., 4.], "sigma": [1., 2.]}, [2., 4.]),
    (mb.NegativeBinomialLSS(), {"mu": [2., 4.], "sigma": [1., 2.]}, [2., 4.]),
    (mb.PIGLSS(), {"mu": [2., 4.], "sigma": [1., 2.]}, [2., 4.]),
    (mb.ZIPoLSS(), {"mu": [2., 4.], "sigma": [.25, .5]}, [1.5, 2.]),
    (mb.ZINBLSS(), {"mu": [2., 4.], "sigma": [1., 2.], "nu": [.25, .5]},
     [1.5, 2.]),
    (mb.JSULSS(), {"mu": [1., 2.], "sigma": [1., 1.], "nu": [0., .5],
                   "tau": [1., 2.]}, [1., 2.]),
])
def test_declared_direct_and_zero_inflated_means(family, params, expected):
    np.testing.assert_allclose(family.mean(params), expected)


def test_zero_adjusted_poisson_mean_uses_the_truncated_positive_component():
    family = mb.ZeroAdjustedPoissonLSS()
    mu = np.array([.5, 2.])
    sigma = np.array([.2, .6])
    expected = (1 - sigma) * mu / (1 - np.exp(-mu))
    np.testing.assert_allclose(family.mean({"mu": mu, "sigma": sigma}), expected)


def test_pareto_and_loglog_means_enforce_moment_existence():
    pareto = mb.Pareto2LSS()
    np.testing.assert_allclose(
        pareto.mean({"mu": np.array([2.]), "sigma": np.array([3.])}), [1.]
    )
    with pytest.raises(ValueError, match="sigma > 1"):
        pareto.mean({"mu": np.array([2.]), "sigma": np.array([1.])})
    loglog = mb.LogLogLSS()
    expected = np.exp(.2) * np.pi * .5 / np.sin(np.pi * .5)
    np.testing.assert_allclose(
        loglog.mean({"mu": np.array([.2]), "sigma": np.array([.5])}), [expected]
    )
    with pytest.raises(ValueError, match="sigma < 1"):
        loglog.mean({"mu": np.array([.2]), "sigma": np.array([1.])})


def test_lognormal_and_weibull_aft_means_match_closed_forms():
    params = {"mu": np.array([.2, .5]), "sigma": np.array([.4, .8])}
    np.testing.assert_allclose(
        mb.LogNormalLSS().mean(params), np.exp(params["mu"] + params["sigma"]**2 / 2)
    )
    np.testing.assert_allclose(
        mb.WeibullLSS().mean(params),
        np.exp(params["mu"] + np.vectorize(math.lgamma)(1 + params["sigma"])),
    )


def test_bccg_keeps_median_distinct_from_an_unimplemented_mean():
    assert not hasattr(mb.BCCGLSS(), "mean")
