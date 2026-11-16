"""Distribution, link, simulation, and workflow contracts for BCT and BCPE."""

import numpy as np
import pytest

import mboostrs as mb


FAMILIES = [mb.BCTLSS, mb.BCPELSS]
PARAMETERS = [
    {"mu": 1.7, "sigma": .4, "nu": .3, "tau": 5.},
    {"mu": 2.4, "sigma": .7, "nu": -.5, "tau": 12.},
    {"mu": .9, "sigma": .25, "nu": .01, "tau": 2.2},
    {"mu": 1.2, "sigma": .6, "nu": 0., "tau": 4.},
]


def broadcast(params, n):
    return {name: np.full(n, value) for name, value in params.items()}


@pytest.mark.parametrize("family_cls", FAMILIES)
def test_links_and_mean_boundary(family_cls):
    family = family_cls()
    eta = np.array([-.5, 0., .7])
    np.testing.assert_array_equal(family.response(0, eta), eta)
    np.testing.assert_allclose(family.response(1, eta), np.exp(eta))
    np.testing.assert_array_equal(family.response(2, eta), eta)
    np.testing.assert_allclose(family.response(3, eta), np.exp(eta))
    with pytest.raises(NotImplementedError, match="location parameter"):
        family.mean(broadcast(PARAMETERS[0], 3))


@pytest.mark.parametrize("family_cls", FAMILIES)
@pytest.mark.parametrize("params", PARAMETERS)
def test_distribution_support_monotonicity_and_quantile_roundtrip(family_cls, params):
    family = family_cls()
    grid = np.geomspace(1e-5, 100, 1200)
    values = family.cdf(grid, broadcast(params, len(grid)))
    assert np.all(np.diff(values) >= -2e-12)
    assert values[0] < .02 and values[-1] > .98
    assert np.all((values >= 0) & (values <= 1))
    np.testing.assert_array_equal(
        family.pdf(np.array([-1., 0.]), broadcast(params, 2)), [0., 0.]
    )
    np.testing.assert_array_equal(
        family.cdf(np.array([-1., 0.]), broadcast(params, 2)), [0., 0.]
    )
    probabilities = np.array([.01, .1, .5, .9, .99])
    par = broadcast(params, len(probabilities))
    quantiles = family.quantile(probabilities, par)
    assert np.all(quantiles > 0)
    np.testing.assert_allclose(
        family.cdf(quantiles, par), probabilities, atol=3e-9, rtol=0
    )


@pytest.mark.parametrize("family_cls", FAMILIES)
@pytest.mark.parametrize("params", PARAMETERS[:3])
def test_density_normalizes_numerically(family_cls, params):
    family = family_cls()
    grid = np.geomspace(1e-9, 1e12, 50_000)
    density = family.pdf(grid, broadcast(params, len(grid)))
    assert np.all(np.isfinite(density)) and np.all(density >= 0)
    assert np.trapezoid(density, grid) == pytest.approx(1, abs=3e-4)


@pytest.mark.parametrize("family_cls", FAMILIES)
def test_inverse_cdf_simulation_has_uniform_pit(family_cls):
    family = family_cls()
    n = 3000
    params = broadcast(PARAMETERS[0], n)
    first = family.simulate(params, rng=np.random.default_rng(17))
    second = family.simulate(params, rng=np.random.default_rng(17))
    np.testing.assert_array_equal(first, second)
    pit = np.sort(family.cdf(first, params))
    expected = (np.arange(1, n + 1) - .5) / n
    assert np.max(np.abs(pit - expected)) < 1.5 / np.sqrt(n)


@pytest.mark.parametrize("family_cls", FAMILIES)
@pytest.mark.parametrize("field,value", [
    ("mu", 0.), ("sigma", 0.), ("tau", 0.), ("nu", np.nan),
])
def test_invalid_parameters_refuse_before_numerics(family_cls, field, value):
    family = family_cls()
    params = dict(PARAMETERS[0])
    params[field] = value
    with pytest.raises(ValueError, match="requires finite"):
        family.pdf([1.], params)


@pytest.mark.parametrize("family_cls", FAMILIES)
def test_fitted_family_supports_distributional_workflow(family_cls):
    rng = np.random.default_rng(42)
    n = 90
    x = np.linspace(-1.0, 1.0, n)
    source = family_cls()
    params = {
        "mu": 2.0 + 0.25 * x,
        "sigma": np.full(n, 0.18),
        "nu": np.full(n, 0.5),
        "tau": np.full(n, 5.0 if family_cls is mb.BCTLSS else 2.5),
    }
    y = source.simulate(params, rng=rng)
    model = mb.glmboostLSS(
        "y ~ x",
        data={"y": y, "x": x},
        families=family_cls(),
        control=mb.boost_control(8, 0.05),
    )

    fitted_params = model.params()
    assert set(fitted_params) == {"mu", "sigma", "nu", "tau"}
    assert all(np.all(np.isfinite(values)) for values in fitted_params.values())
    assert np.all(np.isfinite(model.risk))
    assert np.all(np.isfinite(model.pit(randomized=False)))
    assert np.all(np.isfinite(model.quantile_residuals(randomized=False)))

    centiles = model.centiles(probs=[0.1, 0.5, 0.9])
    assert {"q10", "q50", "q90"} <= set(centiles.columns)
    assert len(centiles) == n
    interval = model.prediction_interval(level=0.8)
    assert {"lower", "median", "upper"} <= set(interval.columns)
    assert len(interval) == n
