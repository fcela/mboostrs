"""Predictive distribution evaluation on explicitly declared data roles."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def gaussian_distribution_fit():
    rng = np.random.default_rng(202)
    x_train = rng.uniform(-1, 1, 500)
    sigma_train = np.exp(.35 * x_train)
    y_train = 1 + .8 * x_train + sigma_train * rng.normal(size=500)
    train = {"x": x_train, "y": y_train}
    fit = mb.glmboostLSS(
        "y ~ bols(x)", train, families=mb.GaussianLSS(),
        control=mb.boost_control(160, .1),
    )
    x_test = rng.uniform(-1, 1, 300)
    sigma_test = np.exp(.35 * x_test)
    y_test = 1 + .8 * x_test + sigma_test * rng.normal(size=300)
    return fit, {"x": x_test, "y": y_test}


def test_external_gaussian_distribution_scores_are_row_reconstructible(
        gaussian_distribution_fit):
    fit, test = gaussian_distribution_fit
    result = mb.evaluate_distribution(
        fit, test, data_role="external", levels=(.8, .9),
        thresholds=(0, 2), random_state=17,
    )
    assert isinstance(result, mb.DistributionEvaluationResult)
    assert result.family == "GaussianLSS"
    assert result.data_role == "external"
    assert result.observed.shape == (300,)
    assert set(result.parameters) == {"mu", "sigma"}
    assert np.all(result.density_or_mass > 0)
    np.testing.assert_allclose(
        result.negative_log_score, -np.log(result.density_or_mass)
    )
    assert np.all((result.pit >= 0) & (result.pit <= 1))
    assert result.metrics["pit_mean"] == pytest.approx(.5, abs=.12)
    assert result.metrics["mean_log_score"] == pytest.approx(
        np.mean(result.negative_log_score)
    )
    assert result.metrics["mean_crps"] == pytest.approx(np.mean(result.crps))
    assert np.all(result.crps >= 0)
    assert len(result.intervals) == 2
    assert [row["nominal_coverage"] for row in result.intervals] == [.8, .9]
    assert len(result.exceedances) == 2
    assert all(0 <= row["predicted_rate"] <= 1 for row in result.exceedances)
    assert len(result.to_frame(output="records")) == 300
    assert len(result.interval_frame(output="records")) == 2
    assert len(result.exceedance_frame(output="records")) == 2
    assert "external outcomes were excluded" in result.describe()
    assert "array(" not in repr(result)


def test_distribution_dashboard_uses_retained_scores_and_data_role(
        gaussian_distribution_fit):
    pytest.importorskip("altair")
    fit, test = gaussian_distribution_fit
    result = mb.evaluate_distribution(
        fit, test, data_role="external", levels=(.5, .8, .95)
    )
    chart = mb.plot.plot_distribution_evaluation(result).to_dict()
    assert len(chart["hconcat"]) == 3
    assert chart["hconcat"][0]["title"] == "PIT against uniform density"
    assert chart["hconcat"][2]["title"] == "Interval calibration · external"


def test_distribution_pit_randomization_is_seeded(gaussian_distribution_fit):
    fit, test = gaussian_distribution_fit
    first = mb.evaluate_distribution(
        fit, test, data_role="external", levels=(.8,), random_state=9
    )
    second = mb.evaluate_distribution(
        fit, test, data_role="external", levels=(.8,), random_state=9
    )
    np.testing.assert_array_equal(first.pit, second.pit)


def test_distribution_weights_define_every_aggregate(gaussian_distribution_fit):
    fit, test = gaussian_distribution_fit
    weights = np.linspace(.1, 2, 300)
    result = mb.evaluate_distribution(
        fit, test, data_role="external", weights=weights,
        levels=(.8,), thresholds=(1,),
    )
    assert result.metrics["weight_total"] == pytest.approx(weights.sum())
    assert result.metrics["mean_log_score"] == pytest.approx(
        np.average(result.negative_log_score, weights=weights)
    )
    assert result.intervals[0]["weight_total"] == pytest.approx(weights.sum())


def test_training_role_is_descriptive_and_retained(gaussian_distribution_fit):
    fit, _test = gaussian_distribution_fit
    training = fit._data
    result = mb.evaluate_distribution(
        fit, training, response="y", data_role="training", levels=(.8,)
    )
    assert result.data_role == "training"
    assert "do not estimate future" in result.assumptions[0]


def test_distribution_evaluation_requires_role_lss_and_valid_requests(
        gaussian_distribution_fit):
    fit, test = gaussian_distribution_fit
    with pytest.raises(ValueError, match="data_role"):
        mb.evaluate_distribution(fit, test, data_role="validation")
    with pytest.raises(ValueError, match="interval level"):
        mb.evaluate_distribution(fit, test, data_role="external", levels=(1,))
    with pytest.raises(ValueError, match="thresholds must be finite"):
        mb.evaluate_distribution(
            fit, test, data_role="external", levels=(.8,), thresholds=(np.nan,)
        )
    standard = mb.glmboost(
        "y ~ x", test, family=mb.Gaussian(), control=mb.boost_control(20, .1)
    )
    with pytest.raises(ValueError, match="requires an LSS model"):
        mb.evaluate_distribution(
            standard, test, data_role="external", levels=(.8,)
        )


def test_mixed_distribution_pit_randomizes_only_the_zero_atom():
    class FixedZaga:
        _family = mb.ZeroAdjustedGammaLSS()
        _response_name = "y"

        @staticmethod
        def predict(*, newdata, type, mstop=None):
            assert type == "response"
            n = len(newdata["y"])
            return {
                "mu": np.full(n, 2.0),
                "sigma": np.full(n, .5),
                "nu": np.full(n, .3),
            }

    data = {"y": np.array([0., 0., .5, 1., 3.])}
    first = mb.evaluate_distribution(
        FixedZaga(), data, data_role="external", levels=(.8,), random_state=1
    )
    second = mb.evaluate_distribution(
        FixedZaga(), data, data_role="external", levels=(.8,), random_state=2
    )
    assert np.all((first.pit[:2] >= 0) & (first.pit[:2] <= .3))
    assert not np.array_equal(first.pit[:2], second.pit[:2])
    np.testing.assert_array_equal(first.pit[2:], second.pit[2:])
    np.testing.assert_allclose(first.density_or_mass[:2], .3)
    # CRPS is integrated over the atom and the continuous part alike.
    assert first.crps is not None and np.all(first.crps >= 0)
    assert first.metrics["mean_crps"] == pytest.approx(np.mean(first.crps))
