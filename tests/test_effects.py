"""Scale, averaging, support, and reconstruction contracts for contrasts."""

import numpy as np
import pytest

import mboostrs as mb


def gaussian_fit(n=200):
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, size=n)
    z = rng.normal(size=n)
    y = 2 * x + .8 * z + rng.normal(scale=.1, size=n)
    data = {"x": x, "z": z, "y": y}
    fit = mb.glmboost(
        "y ~ x + z", data, family=mb.Gaussian(),
        control=mb.boost_control(150, .1),
    )
    return fit, data


def test_linear_response_contrast_is_reconstructible_from_rows():
    fit, data = gaussian_fit()
    result = mb.prediction_contrast(
        fit, data, "x", 0.2, 0.8, type="response",
        observation_ids=[f"row-{i}" for i in range(200)],
    )
    expected = result.alternative_prediction - result.reference_prediction
    np.testing.assert_allclose(result.individual, expected)
    assert result.estimate == pytest.approx(np.average(expected, weights=result.weights))
    assert result.estimate == pytest.approx(1.2, abs=.08)
    assert len(result.to_frame(output="records")) == 200
    assert "not a causal effect" in result.describe()
    assert result.individual.flags.writeable is False
    assert repr(result).startswith("ContrastResult(x:")
    assert "array(" not in repr(result)


def test_weighted_contrast_uses_the_declared_standardization_population():
    fit, data = gaussian_fit()
    weights = np.linspace(.1, 2, 200)
    result = mb.prediction_contrast(
        fit, data, "x", .2, .8, weights=weights,
        population="survey-weighted study sample",
    )
    assert result.estimate == pytest.approx(
        np.average(result.individual, weights=weights)
    )
    assert result.population == "survey-weighted study sample"


def test_response_contrast_transforms_the_complete_binomial_predictor():
    rng = np.random.default_rng(8)
    x = rng.uniform(-1, 1, 500)
    z = rng.normal(size=500)
    probability = 1 / (1 + np.exp(-(.7 * x + 1.2 * z)))
    y = rng.binomial(1, probability)
    data = {"x": x, "z": z, "y": y}
    fit = mb.glmboost(
        "y ~ x + z", data, family=mb.Binomial(),
        control=mb.boost_control(150, .1),
    )
    response = mb.prediction_contrast(fit, data, "x", -.5, .5)
    link = mb.prediction_contrast(fit, data, "x", -.5, .5, type="link")
    assert np.std(response.individual) > 0
    assert np.std(link.individual) == pytest.approx(0, abs=1e-12)
    assert response.estimate != pytest.approx(link.estimate)


def test_support_policy_applies_to_linear_terms_too():
    fit, data = gaussian_fit()
    with pytest.raises(ValueError, match="outside training support"):
        mb.prediction_contrast(fit, data, "x", 0, 1.2)
    with pytest.warns(RuntimeWarning, match="depends on extrapolation"):
        allowed = mb.prediction_contrast(
            fit, data, "x", 0, 1.2, extrapolation="warn"
        )
    clipped = mb.prediction_contrast(
        fit, data, "x", 0, 1.2, extrapolation="clip"
    )
    boundary = mb.prediction_contrast(
        fit, data, "x", data["x"].min(), data["x"].max()
    )
    assert allowed.support["alternative"]["outside"] is True
    assert clipped.support["alternative"]["effective"] == data["x"].max()
    assert clipped.estimate == pytest.approx(boundary.estimate)


def test_factor_contrast_uses_original_training_levels_not_internal_codes():
    rng = np.random.default_rng(14)
    group = np.resize(np.array(["control", "middle", "high"]), 180)
    means = {"control": 0.0, "middle": 1.0, "high": 2.0}
    y = np.array([means[value] for value in group]) + rng.normal(scale=.1, size=180)
    data = {"group": group, "y": y}
    fit = mb.gamboost(
        "y ~ brandom(group, df=2)", data,
        control=mb.boost_control(100, .1),
    )
    result = mb.prediction_contrast(
        fit, data, "group", "control", "high", type="response"
    )
    assert result.estimate == pytest.approx(2, abs=.15)
    assert result.support["reference"]["levels"] == ("control", "high", "middle")
    with pytest.raises(ValueError, match="unseen level"):
        mb.prediction_contrast(fit, data, "group", "control", "new")


def test_distributional_contrast_requires_and_records_a_parameter():
    rng = np.random.default_rng(9)
    x = rng.uniform(-1, 1, 160)
    y = 1 + x + rng.normal(scale=np.exp(.2 * x), size=160)
    data = {"x": x, "y": y}
    fit = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=mb.boost_control(80, .1),
    )
    with pytest.raises(ValueError, match="parameter is required"):
        mb.prediction_contrast(fit, data, "x", -.5, .5)
    result = mb.prediction_contrast(
        fit, data, "x", -.5, .5, parameter="mu", type="response"
    )
    assert result.parameter == "mu"
    assert result.individual.shape == (160,)


def test_linear_average_marginal_effect_reconstructs_the_known_slope():
    fit, data = gaussian_fit()
    result = mb.average_marginal_effect(fit, data, "x", step=.01)
    expected = ((result.upper_prediction - result.lower_prediction)
                / (result.upper_value - result.lower_value))
    np.testing.assert_allclose(result.individual, expected)
    assert result.estimate == pytest.approx(2, abs=.12)
    assert result.support["one_sided_rows"] >= 0
    assert len(result.to_frame(output="records")) == 200
    assert result.individual.flags.writeable is False
    assert repr(result).startswith("MarginalEffectResult(")


def test_effect_plots_keep_row_heterogeneity_separate_from_the_average():
    pytest.importorskip("altair")
    fit, data = gaussian_fit()
    contrast = mb.prediction_contrast(fit, data, "x", .2, .8)
    derivative = mb.average_marginal_effect(fit, data, "x", step=.01)
    contrast_chart = mb.plot.plot_prediction_contrast(contrast).to_dict()
    derivative_chart = mb.plot.plot_marginal_effect(derivative).to_dict()
    assert contrast_chart["title"] == "Prediction contrast: x"
    assert derivative_chart["title"] == "Marginal effect: x"


def test_response_marginal_effect_varies_under_a_nonidentity_link():
    rng = np.random.default_rng(22)
    x = rng.uniform(-1, 1, 500)
    z = rng.normal(size=500)
    y = rng.binomial(1, 1 / (1 + np.exp(-(.8 * x + 1.1 * z))))
    data = {"x": x, "z": z, "y": y}
    fit = mb.glmboost(
        "y ~ x + z", data, family=mb.Binomial(),
        control=mb.boost_control(150, .1),
    )
    response = mb.average_marginal_effect(fit, data, "x", step=.01)
    link = mb.average_marginal_effect(fit, data, "x", step=.01, type="link")
    assert np.std(response.individual) > 1e-4
    assert np.std(link.individual) == pytest.approx(0, abs=1e-10)
    assert response.estimate != pytest.approx(link.estimate)


def test_marginal_effect_uses_one_sided_differences_at_support_boundaries():
    fit, data = gaussian_fit()
    boundary_data = {name: values[:3].copy() for name, values in data.items()}
    boundary_data["x"] = np.array([data["x"].min(), .5, data["x"].max()])
    result = mb.average_marginal_effect(
        fit, boundary_data, "x", step=.1, extrapolation="error"
    )
    assert result.lower_value[0] == data["x"].min()
    assert result.upper_value[-1] == data["x"].max()
    assert result.support["one_sided_rows"] == 2
    assert np.all(result.upper_value > result.lower_value)


def test_distributional_average_marginal_effect_requires_a_parameter():
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, 120)
    y = x + rng.normal(size=120)
    data = {"x": x, "y": y}
    fit = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=mb.boost_control(60, .1),
    )
    with pytest.raises(ValueError, match="parameter is required"):
        mb.average_marginal_effect(fit, data, "x")
    result = mb.average_marginal_effect(fit, data, "x", parameter="mu")
    assert result.parameter == "mu"
    assert np.isfinite(result.estimate)


def test_distributional_contrasts_use_all_fitted_parameters():
    rng = np.random.default_rng(44)
    x = rng.uniform(-1, 1, 320)
    y = 1 + x + np.exp(.6 * x) * rng.normal(size=320)
    data = {"x": x, "y": y}
    fit = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=mb.boost_control(140, .1),
    )
    mean = mb.distributional_contrast(
        fit, data, "x", -.5, .5, quantity="mean"
    )
    upper = mb.distributional_contrast(
        fit, data, "x", -.5, .5, quantity="quantile", value=.9
    )
    exceedance = mb.distributional_contrast(
        fit, data, "x", -.5, .5, quantity="exceedance_probability", value=1.5
    )
    assert mean.parameter == "mean"
    assert upper.parameter == "quantile(0.9)"
    assert exceedance.parameter == "Pr(response>1.5)"
    assert upper.estimate != pytest.approx(mean.estimate)
    assert np.all((exceedance.reference_prediction >= 0)
                  & (exceedance.reference_prediction <= 1))
    assert np.all((exceedance.alternative_prediction >= 0)
                  & (exceedance.alternative_prediction <= 1))
    for result in (mean, upper, exceedance):
        assert result.estimate == pytest.approx(np.mean(result.individual))


@pytest.mark.parametrize("quantity, value, message", [
    ("quantile", None, "requires"),
    ("quantile", 1.0, r"in \(0, 1\)"),
    ("exceedance_probability", np.nan, "finite"),
    ("mean", 0, "not used"),
    ("median", None, "quantity must"),
])
def test_distributional_contrast_validates_the_requested_functional(
        quantity, value, message):
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 100)
    y = x + rng.normal(size=100)
    data = {"x": x, "y": y}
    fit = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=mb.boost_control(30, .1),
    )
    with pytest.raises((ValueError, NotImplementedError), match=message):
        mb.distributional_contrast(
            fit, data, "x", -.5, .5, quantity=quantity, value=value
        )


def test_distributional_contrast_refuses_a_standard_model():
    fit, data = gaussian_fit()
    with pytest.raises(ValueError, match="requires an LSS model"):
        mb.distributional_contrast(
            fit, data, "x", .2, .8, quantity="mean"
        )


def test_accumulated_local_effect_recovers_a_linear_fitted_slope():
    fit, data = gaussian_fit()
    result = mb.accumulated_local_effect(fit, data, "x", bins=8)
    slopes = np.diff(result.effect) / np.diff(result.grid)
    assert np.allclose(slopes, 2, atol=.12)
    assert np.average(result.observation_effect, weights=result.weights) == pytest.approx(
        0, abs=1e-14
    )
    assert result.counts.sum() == 200
    assert result.weight_totals.sum() == pytest.approx(200)
    assert len(result.to_frame(output="records")) == len(result.effect)
    assert len(result.observation_frame(output="records")) == 200
    assert result.support["effective_bins"] == len(result.effect)
    assert repr(result).startswith("ALE1DResult(")


def test_ale_plot_exposes_interval_support():
    pytest.importorskip("altair")
    fit, data = gaussian_fit()
    result = mb.accumulated_local_effect(fit, data, "x", bins=8)
    chart = mb.plot.plot_accumulated_local_effect(result).to_dict()
    assert chart["title"] == "Accumulated local effect: x"


def test_weighted_ale_is_centered_on_the_declared_population():
    fit, data = gaussian_fit()
    weights = np.linspace(.1, 2, 200)
    result = mb.accumulated_local_effect(
        fit, data, "x", bins=7, weights=weights,
        population="weighted target sample",
    )
    assert np.average(result.observation_effect, weights=weights) == pytest.approx(
        0, abs=1e-14
    )
    assert result.population == "weighted target sample"


def test_response_scale_ale_varies_under_a_binomial_link():
    rng = np.random.default_rng(62)
    x = rng.uniform(-1, 1, 400)
    z = .8 * x + rng.normal(scale=.5, size=400)
    y = rng.binomial(1, 1 / (1 + np.exp(-(.9 * x + z))))
    data = {"x": x, "z": z, "y": y}
    fit = mb.glmboost(
        "y ~ x + z", data, family=mb.Binomial(),
        control=mb.boost_control(120, .1),
    )
    response = mb.accumulated_local_effect(fit, data, "x", bins=8)
    link = mb.accumulated_local_effect(fit, data, "x", bins=8, type="link")
    assert np.all(np.isfinite(response.effect))
    assert not np.allclose(response.effect, link.effect)


def test_ale_refuses_indicators_and_collapsed_quantile_intervals():
    rng = np.random.default_rng(7)
    indicator = np.resize(np.array([0, 1]), 100)
    y = indicator + rng.normal(size=100)
    data = {"indicator": indicator, "y": y}
    fit = mb.glmboost(
        "y ~ indicator", data, control=mb.boost_control(30, .1)
    )
    with pytest.raises(ValueError, match="prediction_contrast"):
        mb.accumulated_local_effect(fit, data, "indicator", bins=5)


def test_poisson_elasticity_and_semi_elasticity_are_reconstructible():
    rng = np.random.default_rng(73)
    x = rng.uniform(.2, 2, 800)
    y = rng.poisson(np.exp(.2 + .6 * x))
    data = {"x": x, "y": y}
    fit = mb.glmboost(
        "y ~ x", data, family=mb.Poisson(),
        control=mb.boost_control(180, .1),
    )
    elasticity = mb.average_elasticity(fit, data, "x", log_step=1e-3)
    semi = mb.average_elasticity(
        fit, data, "x", kind="semi_elasticity", log_step=1e-3
    )
    expected_elasticity = (
        np.log(elasticity.upper_prediction) - np.log(elasticity.lower_prediction)
    ) / (np.log(elasticity.upper_value) - np.log(elasticity.lower_value))
    expected_semi = (
        np.log(semi.upper_prediction) - np.log(semi.lower_prediction)
    ) / (semi.upper_value - semi.lower_value)
    np.testing.assert_allclose(elasticity.individual, expected_elasticity)
    np.testing.assert_allclose(semi.individual, expected_semi)
    assert np.std(semi.individual) == pytest.approx(0, abs=1e-9)
    assert np.std(elasticity.individual) > .05
    assert elasticity.estimate == pytest.approx(np.mean(elasticity.individual))
    assert len(elasticity.to_frame(output="records")) == 800


def test_elasticity_requires_positive_predictors_and_predictions():
    fit, data = gaussian_fit()
    shifted = {name: values.copy() for name, values in data.items()}
    shifted["x"][0] = 0
    with pytest.raises(ValueError, match="positive predictor"):
        mb.average_elasticity(fit, shifted, "x")
    with pytest.raises(ValueError, match="positive response predictions"):
        mb.average_elasticity(fit, data, "x")


@pytest.mark.parametrize("kind, step, message", [
    ("ratio", 1e-3, "kind"),
    ("elasticity", 0, "log_step"),
    ("elasticity", np.nan, "log_step"),
])
def test_elasticity_validates_its_definition(kind, step, message):
    fit, data = gaussian_fit()
    with pytest.raises(ValueError, match=message):
        mb.average_elasticity(fit, data, "x", kind=kind, log_step=step)


def test_prediction_profile_reconstructs_weighted_partial_dependence_and_ice():
    fit, data = gaussian_fit()
    weights = np.linspace(.2, 2, 200)
    result = mb.prediction_profile(
        fit, data, "x", points=9, weights=weights,
        observation_ids=[f"row-{i}" for i in range(200)],
    )
    np.testing.assert_allclose(
        result.average, np.average(result.ice, axis=0, weights=weights)
    )
    slopes = np.diff(result.average) / np.diff(result.grid)
    assert np.allclose(slopes, 2, atol=.12)
    assert len(result.to_frame(output="records")) == 9
    assert len(result.ice_frame(output="records")) == 200 * 9
    assert result.ice.flags.writeable is False


def test_prediction_profile_plot_keeps_average_ice_and_support_separate():
    pytest.importorskip("altair")
    rng = np.random.default_rng(89)
    x = rng.normal(size=300)
    z = x + rng.normal(scale=.1, size=300)
    y = x + z + rng.normal(size=300)
    data = {"x": x, "z": z, "y": y}
    fit = mb.glmboost("y ~ x + z", data, control=mb.boost_control(80, .1))
    result = mb.prediction_profile(fit, data, "x", points=7)
    chart = mb.plot.plot_prediction_profile(result, max_ice=12).to_dict()
    assert len(chart["vconcat"]) == 2
    assert chart["vconcat"][0]["title"] == "Prediction profile: x"


def test_profile_joint_support_flags_extreme_combinations_for_correlated_predictors():
    rng = np.random.default_rng(88)
    x = rng.normal(size=500)
    z = x + rng.normal(scale=.08, size=500)
    y = x + z + rng.normal(scale=.1, size=500)
    data = {"x": x, "z": z, "y": y}
    fit = mb.glmboost(
        "y ~ x + z", data, control=mb.boost_control(100, .1)
    )
    grid = np.array([np.quantile(x, .05), np.median(x), np.quantile(x, .95)])
    result = mb.prediction_profile(fit, data, "x", grid=grid)
    assert result.support["method"] == "weighted_linear_conditional_residual"
    assert result.support_fraction_far[0] > result.support_fraction_far[1]
    assert result.support_fraction_far[2] > result.support_fraction_far[1]
    assert "nonlinear" in result.support["limitation"]


def test_factor_profile_uses_training_labels_and_refuses_unseen_grid_values():
    rng = np.random.default_rng(19)
    group = np.resize(np.array(["a", "b", "c"]), 180)
    y = np.array([{"a": 0, "b": 1, "c": 2}[item] for item in group])
    y = y + rng.normal(scale=.1, size=180)
    data = {"group": group, "y": y}
    fit = mb.gamboost(
        "y ~ brandom(group, df=2)", data,
        control=mb.boost_control(100, .1),
    )
    result = mb.prediction_profile(fit, data, "group")
    assert result.grid.tolist() == ["a", "b", "c"]
    assert np.all(np.isnan(result.support_fraction_far))
    with pytest.raises(ValueError, match="unseen level"):
        mb.prediction_profile(fit, data, "group", grid=["a", "new"])


def test_compact_profile_omits_ice_without_losing_the_average():
    fit, data = gaussian_fit()
    result = mb.prediction_profile(fit, data, "x", points=7, retain_ice=False)
    assert result.ice is None
    assert len(result.average) == 7
    with pytest.raises(ValueError, match="retain_ice"):
        result.ice_frame()


def test_distributional_profile_requires_a_parameter():
    rng = np.random.default_rng(15)
    x = rng.uniform(-1, 1, 120)
    y = x + rng.normal(size=120)
    data = {"x": x, "y": y}
    fit = mb.glmboostLSS(
        "y ~ bols(x)", data, families=mb.GaussianLSS(),
        control=mb.boost_control(50, .1),
    )
    with pytest.raises(ValueError, match="parameter is required"):
        mb.prediction_profile(fit, data, "x", points=5)
    result = mb.prediction_profile(fit, data, "x", points=5, parameter="mu")
    assert result.parameter == "mu"
    assert result.ice.shape == (120, 5)


@pytest.mark.parametrize("step", [0, -1, np.nan])
def test_marginal_effect_requires_a_positive_finite_step(step):
    fit, data = gaussian_fit()
    with pytest.raises(ValueError, match="step"):
        mb.average_marginal_effect(fit, data, "x", step=step)


@pytest.mark.parametrize("arguments, message", [
    ({"variable": "missing", "reference": 0, "alternative": 1}, "not present"),
    ({"variable": "x", "reference": np.nan, "alternative": 1}, "finite"),
    ({"variable": "x", "reference": 0, "alternative": 1, "type": "probability"},
     "type"),
    ({"variable": "x", "reference": 0, "alternative": 1,
      "observation_ids": np.zeros(200)}, "uniquely"),
])
def test_contrast_rejects_ambiguous_or_invalid_inputs(arguments, message):
    fit, data = gaussian_fit()
    with pytest.raises(ValueError, match=message):
        mb.prediction_contrast(fit, data, **arguments)
