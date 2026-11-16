"""Hand-computed contrasts, cluster uncertainty and joint support for effects.

The models here are small stand-ins with known predictors. Each one exposes
the `predict` signature and the `_data` training frame that the effects
functions read, so every expected value below can be worked out by hand.
"""

import json
import math
import warnings
from statistics import NormalDist

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.effects import ContrastResult

# The cluster tests use few clusters on purpose; their interval warning is
# asserted once in test_few_clusters_warn_that_the_percentile_interval_is_unreliable.
pytestmark = pytest.mark.filterwarnings("ignore:only .* clusters:RuntimeWarning")


class Linear:
    """Predictor eta = f(newdata) on the link scale, with an optional inverse link."""

    def __init__(self, eta, *, data, formula, inverse=None):
        self._eta = eta
        self._data = data
        self.formula = formula
        self._inverse = inverse

    def predict(self, newdata, type="link", mstop=None):
        eta = np.asarray(self._eta(newdata), dtype=np.float64)
        if type == "link" or self._inverse is None:
            return eta
        return self._inverse(eta)


class NormalFamily:
    """Normal distribution with mean mu and standard deviation sigma per row."""

    def mean(self, params):
        return params["mu"]

    def quantile(self, probability, params):
        z = NormalDist().inv_cdf(probability)
        return params["mu"] + params["sigma"] * z

    def cdf(self, threshold, params):
        return np.array([
            NormalDist(mu, sigma).cdf(threshold)
            for mu, sigma in zip(params["mu"], params["sigma"], strict=True)
        ])


class NormalLSS:
    """Distributional stand-in: mu = x, sigma = 1 + x (both read `x`)."""

    def __init__(self, data, *, mean=None):
        self._data = data
        self.formula = {"mu": "y ~ x", "sigma": "y ~ x"}
        self._family = NormalFamily() if mean is None else mean

    def predict(self, newdata, type="link", mstop=None, parameter=None):
        x = np.asarray(newdata["x"], dtype=np.float64)
        return {"mu": x.copy(), "sigma": 1.0 + x}


def wide_training():
    """A training frame whose x spans [0, 1], so contrasts from 0 to 1 are inside support."""
    return {"x": np.array([0.0, 1.0]), "z": np.array([-2.0, 2.0]), "y": np.zeros(2)}


def linear_data(n=6):
    rng = np.random.default_rng(11)
    return {
        "x": np.linspace(0.0, 1.0, n),
        "z": rng.normal(size=n),
        "y": rng.normal(size=n),
    }


# Hand-computed link-scale, identity-link and nonidentity-link contrasts.


def test_identity_link_contrast_is_the_slope_times_the_step():
    data = linear_data()
    model = Linear(lambda d: 1 + 2 * d["x"] + 0.5 * d["z"], data=data, formula="y ~ x + z")
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link")
    # eta(1) - eta(0) = 2 for every row, whatever z is.
    np.testing.assert_allclose(result.individual, 2.0)
    assert result.estimate == pytest.approx(2.0)


def test_interaction_contrast_is_averaged_with_the_supplied_weights():
    # eta = 1 + 2x + 3xz, so eta(1) - eta(0) = 2 + 3z. With z = (0, 1) and
    # weights (1, 3) the contrasts are (2, 5) and the weighted mean is
    # (1*2 + 3*5) / (1 + 3) = 17 / 4 = 4.25.
    data = {"x": np.array([0.0, 1.0]), "z": np.array([0.0, 1.0]), "y": np.zeros(2)}
    model = Linear(lambda d: 1 + 2 * d["x"] + 3 * d["x"] * d["z"], data=data, formula="y ~ x + z")
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                    weights=np.array([1.0, 3.0]))
    np.testing.assert_allclose(result.individual, [2.0, 5.0])
    assert result.estimate == pytest.approx(4.25)


def test_log_link_contrast_transforms_the_full_predictor():
    # mu = exp(0.5 + 0.8 x). The response contrast from x = 0.2 to x = 0.7 is
    # exp(0.5)(exp(0.56) - exp(0.16)); exponentiating only the slope term
    # would give exp(0.8 * 0.5) = exp(0.4) instead.
    data = {"x": np.array([0.0, 0.2, 0.7, 1.0]), "y": np.zeros(4)}
    model = Linear(lambda d: 0.5 + 0.8 * d["x"], data=data, formula="y ~ x",
                   inverse=np.exp)
    result = mb.prediction_contrast(model, data, "x", 0.2, 0.7, type="response")
    expected = math.exp(0.5) * (math.exp(0.56) - math.exp(0.16))
    np.testing.assert_allclose(result.individual, expected)
    assert result.estimate == pytest.approx(expected)
    assert expected != pytest.approx(math.exp(0.4))


def test_logit_link_contrast_is_the_difference_of_the_probabilities():
    # p = 1 / (1 + exp(-(-0.5 + 1.2 x))). Contrast from x = 0 to x = 1.
    data = {"x": np.array([0.0, 0.5, 1.0]), "y": np.zeros(3)}
    sigmoid = lambda e: 1 / (1 + np.exp(-e))  # noqa: E731
    model = Linear(lambda d: -0.5 + 1.2 * d["x"], data=data, formula="y ~ x",
                   inverse=sigmoid)
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="response")
    expected = sigmoid(0.7) - sigmoid(-0.5)
    np.testing.assert_allclose(result.individual, expected)
    assert result.estimate == pytest.approx(expected)


def test_factor_contrast_is_the_difference_of_level_effects():
    # level effects: a = 0, b = 0.5, c = -0.25. The contrast a -> c is -0.25
    # on every row, and b -> c is -0.75.
    effects = {"a": 0.0, "b": 0.5, "c": -0.25}
    data = {
        "g": np.array(["a", "b", "c", "a", "c", "b"]),
        "x": np.linspace(0, 1, 6),
        "y": np.zeros(6),
    }
    model = Linear(
        lambda d: 1 + np.array([effects[v] for v in d["g"]]) + 0.3 * d["x"],
        data=data, formula="y ~ x + g",
    )
    up = mb.prediction_contrast(model, data, "g", "a", "c", type="link")
    np.testing.assert_allclose(up.individual, -0.25)
    down = mb.prediction_contrast(model, data, "g", "b", "c", type="link")
    np.testing.assert_allclose(down.individual, -0.75)


def test_unseen_factor_level_is_refused_by_name():
    data = {"g": np.array(["a", "b"]), "x": np.zeros(2), "y": np.zeros(2)}
    model = Linear(lambda d: np.zeros(len(d["x"])), data=data, formula="y ~ x + g")
    with pytest.raises(ValueError, match="unseen level"):
        mb.prediction_contrast(model, data, "g", "a", "zzz", type="link")


def test_average_marginal_effect_of_an_interaction_is_the_weighted_slope():
    # d eta / d x = 2 + 3z, exact for a linear predictor, so the finite
    # difference returns it on every row, boundary rows included.
    data = {"x": np.array([0.0, 1.0]), "z": np.array([0.0, 1.0]), "y": np.zeros(2)}
    model = Linear(lambda d: 1 + 2 * d["x"] + 3 * d["x"] * d["z"], data=data, formula="y ~ x + z")
    result = mb.average_marginal_effect(model, data, "x", type="link",
                                        weights=np.array([1.0, 3.0]))
    np.testing.assert_allclose(result.individual, [2.0, 5.0], rtol=1e-6)
    assert result.estimate == pytest.approx(4.25, rel=1e-6)


# Infinite and undefined moments of a distributional model.


def test_a_mean_that_is_infinite_on_a_row_is_refused():
    class Heavy(NormalFamily):
        def mean(self, params):
            return np.where(params["mu"] > 0.5, np.inf, params["mu"])

    data = {"x": np.array([0.0, 1.0]), "y": np.zeros(2)}
    model = NormalLSS(data, mean=Heavy())
    with pytest.raises(ValueError, match="one finite value per row"):
        mb.distributional_contrast(model, data, "x", 0.0, 1.0, quantity="mean")


def test_a_family_without_a_mean_is_refused_by_name():
    class NoMean:
        def quantile(self, probability, params):
            return params["mu"]

    data = {"x": np.array([0.0, 1.0]), "y": np.zeros(2)}
    model = NormalLSS(data, mean=NoMean())
    with pytest.raises(NotImplementedError, match="does not declare a response mean"):
        mb.distributional_contrast(model, data, "x", 0.0, 1.0, quantity="mean")


# Distributional contrasts against the normal distribution.


def test_mean_and_quantile_contrasts_follow_the_normal_formulas():
    data = {"x": np.array([0.0, 0.5, 1.0]), "y": np.zeros(3)}
    model = NormalLSS(data)
    mean = mb.distributional_contrast(model, data, "x", 0.0, 1.0, quantity="mean")
    # mu = x, so the mean moves by one; sigma does not enter the mean.
    np.testing.assert_allclose(mean.individual, 1.0)
    # With z = inv_cdf(0.975): at x = 0 the quantile is 0 + 1 * z = z (mu = 0,
    # sigma = 1); at x = 1 it is 1 + 2 * z (mu = 1, sigma = 2). The contrast
    # is 1 + z.
    upper = mb.distributional_contrast(model, data, "x", 0.0, 1.0,
                                       quantity="quantile", value=0.975)
    assert upper.estimate == pytest.approx(1 + NormalDist().inv_cdf(0.975))


def test_exceedance_contrast_is_strict_and_hand_computed():
    # P(Y > 2) at x = 0 (sigma 1, mu 0) is 1 - Phi(2); at x = 1 (mu 1, sigma 2)
    # it is 1 - Phi(0.5).
    data = {"x": np.array([0.0, 1.0]), "y": np.zeros(2)}
    model = NormalLSS(data)
    result = mb.distributional_contrast(model, data, "x", 0.0, 1.0,
                                        quantity="exceedance_probability", value=2.0)
    expected = (1 - NormalDist().cdf(0.5)) - (1 - NormalDist().cdf(2.0))
    assert result.estimate == pytest.approx(expected)
    assert result.parameter == "Pr(response>2)"


# Cluster-level uncertainty.


def interaction_model_and_data():
    # eta = 1 + 2x + 3xz, so the contrast 0 -> 1 is 2 + 3z on each row.
    rows = {"x": np.array([0.0, 1.0, 0.0, 1.0]), "z": np.array([0.0, 0.0, 1.0, 1.0]),
            "y": np.zeros(4)}
    model = Linear(lambda d: 1 + 2 * d["x"] + 3 * d["x"] * d["z"], data=rows,
                   formula="y ~ x + z")
    return model, rows, None


def test_cluster_sandwich_matches_a_hand_computation():
    # Four standardization rows, z = (0, 1, 1, 2): contrasts (2, 5, 5, 8),
    # mean 5. Clusters (0, 0, 1, 2) give scores -3, 0, 3, so the sandwich
    # variance is G/(G-1) * (9 + 0 + 9) / 4**2 and the SE is sqrt(27) / 4.
    data = {"x": np.zeros(4), "z": np.array([0.0, 1.0, 1.0, 2.0]), "y": np.zeros(4)}
    model = Linear(lambda d: 1 + 2 * d["x"] + 3 * d["x"] * d["z"], data=wide_training(),
                   formula="y ~ x + z")
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                    cluster=np.array(["p", "p", "q", "r"]), n_draws=400)
    assert result.estimate == pytest.approx(5.0)
    assert result.dependence["se_sandwich"] == pytest.approx(math.sqrt(27) / 4)
    assert result.dependence["clusters"] == 3
    assert result.uncertainty == "cluster_bootstrap"
    lower, upper = result.interval
    assert lower < 5.0 < upper
    assert result.dependence["band"] == "pointwise"


def test_clustering_widens_the_standard_error_when_rows_share_a_cluster():
    # Rows in a cluster share z, so their contrasts are alike. The naive SE
    # treats the 12 rows as independent; the sandwich SE does not.
    rng = np.random.default_rng(2)
    cluster_z = rng.normal(size=4)
    labels = np.repeat(np.arange(4), 3)
    z = cluster_z[labels]
    data = {"x": np.zeros(12), "z": z, "y": np.zeros(12)}
    model = Linear(lambda d: 1 + 2 * d["x"] + 3 * d["x"] * d["z"], data=wide_training(),
                   formula="y ~ x + z")
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                    cluster=labels, n_draws=300)
    individual = result.individual
    naive = np.std(individual, ddof=1) / math.sqrt(12)
    assert result.dependence["se_sandwich"] > naive


def test_cluster_uncertainty_is_deterministic_under_the_seed():
    data = linear_data(n=40)
    data["z"] = np.random.default_rng(9).normal(size=40)
    model = Linear(lambda d: 1 + 2 * d["x"] + 3 * d["x"] * d["z"], data=data,
                   formula="y ~ x + z")
    labels = np.arange(40) % 8
    first = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                   cluster=labels, n_draws=200, seed=5)
    second = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                    cluster=labels, n_draws=200, seed=5)
    other = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                   cluster=labels, n_draws=200, seed=6)
    assert first.interval == second.interval
    assert first.interval != other.interval


def test_one_cluster_is_refused_and_a_length_mismatch_is_named():
    data = linear_data()
    model = Linear(lambda d: 2 * d["x"], data=data, formula="y ~ x")
    with pytest.raises(ValueError, match="at least two clusters"):
        mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link", cluster=np.zeros(6))
    with pytest.raises(ValueError, match="cluster has shape"):
        mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link", cluster=np.arange(5))


def test_few_clusters_warn_that_the_percentile_interval_is_unreliable():
    data = linear_data()
    model = Linear(lambda d: 2 * d["x"], data=data, formula="y ~ x")
    with pytest.warns(RuntimeWarning, match="only 3 clusters"):
        mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                               cluster=np.array([0, 0, 1, 1, 2, 2]), n_draws=100)


def test_without_clusters_there_is_no_dependence_record():
    data = linear_data()
    model = Linear(lambda d: 2 * d["x"], data=data, formula="y ~ x")
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link")
    assert result.dependence is None
    assert result.interval is None
    assert result.uncertainty == "none"
    assert "No sampling or model-selection uncertainty" in result.describe()


def test_marginal_effect_and_distributional_contrast_accept_clusters():
    data = {"x": np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.6]), "z": np.arange(6.0),
            "y": np.zeros(6)}
    model = Linear(lambda d: 2 * d["x"] + 0.1 * d["z"], data=data, formula="y ~ x + z")
    labels = np.array([0, 0, 1, 1, 2, 2])
    marginal = mb.average_marginal_effect(model, data, "x", type="link", cluster=labels,
                                          n_draws=100)
    assert marginal.estimate == pytest.approx(2.0, rel=1e-6)
    assert marginal.dependence["clusters"] == 3
    lss = NormalLSS({"x": np.array([0.0, 1.0, 0.5, 0.25]), "y": np.zeros(4)})
    contrast = mb.distributional_contrast(
        lss, {"x": np.array([0.0, 1.0, 0.5, 0.25]), "y": np.zeros(4)}, "x", 0.0, 1.0,
        quantity="mean", cluster=np.array([0, 0, 1, 1]), n_draws=100,
    )
    assert contrast.dependence["clusters"] == 2


def test_cluster_record_is_json_safe_and_round_trips_through_to_dict():
    data = linear_data()
    model = Linear(lambda d: 2 * d["x"], data=data, formula="y ~ x")
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link",
                                    cluster=np.array([0, 0, 1, 1, 2, 2]), n_draws=100)
    record = result.to_dict()
    assert record["dependence"]["clusters"] == 3
    json.dumps(record)


def test_a_cluster_result_without_its_record_is_refused():
    data = linear_data()
    model = Linear(lambda d: 2 * d["x"], data=data, formula="y ~ x")
    base = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link")
    with pytest.raises(ValueError, match="must agree"):
        ContrastResult(
            estimate=base.estimate, individual=base.individual,
            reference_prediction=base.reference_prediction,
            alternative_prediction=base.alternative_prediction, weights=base.weights,
            observation_ids=base.observation_ids, variable="x", reference=0.0,
            alternative=1.0, scale="link", parameter=None, population="supplied rows",
            support={}, uncertainty="cluster_bootstrap", dependence=None,
        )


# Joint support of counterfactual combinations.


def test_correlated_predictors_expose_unsupported_combinations():
    # x2 follows x1 closely in the training data. Setting x1 = 0.9 for a row
    # with x2 near 0.1 asks for a combination the data never contains.
    rng = np.random.default_rng(3)
    x1 = rng.uniform(0, 1, 400)
    x2 = np.clip(x1 + rng.normal(scale=0.03, size=400), 0, 1)
    training = {"x1": x1, "x2": x2, "y": np.zeros(400)}
    rows = {"x1": x1[:60], "x2": x2[:60], "y": np.zeros(60)}
    model = Linear(lambda d: 2 * d["x1"] + d["x2"], data=training, formula="y ~ x1 + x2")
    correlated = mb.prediction_contrast(model, rows, "x1", 0.1, 0.9, type="link")
    joint = correlated.support["joint"]
    assert joint["supported_fraction"] < 0.6
    assert joint["unsupported_rows"] > 0
    assert joint["conditioning_columns"] == ("x2",)
    assert joint["training_source"] == "training_data"

    independent_x2 = rng.uniform(0, 1, 400)
    uncorrelated = {"x1": x1, "x2": independent_x2, "y": np.zeros(400)}
    model_u = Linear(lambda d: 2 * d["x1"] + d["x2"], data=uncorrelated, formula="y ~ x1 + x2")
    spread = {"x1": x1[:60], "x2": independent_x2[:60], "y": np.zeros(60)}
    ok = mb.prediction_contrast(model_u, spread, "x1", 0.1, 0.9, type="link")
    assert ok.support["joint"]["supported_fraction"] > 0.9


def test_a_counterfactual_on_the_empty_centre_of_a_circle_is_unsupported():
    # Training rows lie on the circle (cos t, sin t). A row at (1, 0) with x1
    # set to 0 asks for (0, 0), the centre: every training row is at least one
    # unit away, which is 1.41 sd on the standardized scale (sd of cos = sd of
    # sin = 0.707). A linear distance from the cloud's mean misses this.
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    training = {"x1": np.cos(t), "x2": np.sin(t), "y": np.zeros(400)}
    rows = {"x1": np.array([1.0, 0.0]), "x2": np.array([0.0, 1.0]), "y": np.zeros(2)}
    model = Linear(lambda d: d["x1"] + d["x2"], data=training, formula="y ~ x1 + x2")
    row_on_circle = mb.prediction_contrast(
        model, rows, "x1", 0.0, 1.0, type="link", weights=np.array([0.0, 1.0]),
    )
    assert row_on_circle.support["joint"]["supported_fraction"] == pytest.approx(1.0)
    row_at_centre = mb.prediction_contrast(
        model, rows, "x1", 0.0, 1.0, type="link", weights=np.array([1.0, 0.0]),
    )
    assert row_at_centre.support["joint"]["supported_fraction"] == pytest.approx(0.0)


def test_an_unobserved_factor_combination_is_unsupported_away_from_its_region():
    # Level "rare" occurs only for x <= 0.2. A row with x near 1 asks for
    # "rare" at x = 1, and the nearest "rare" row is 0.8 away, which is
    # about 2.8 training sd, so the counterfactual is unsupported.
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 300)
    level = np.where(rng.uniform(size=300) < 0.5, "common", "rare")
    level[x > 0.2] = "common"
    training = {"x": x, "g": level, "y": np.zeros(300)}
    model = Linear(lambda d: np.where(d["g"] == "rare", 1.0, 0.0) + d["x"],
                   data=training, formula="y ~ x + g")
    rows = {"x": np.array([0.05, 0.95]), "g": np.array(["common", "common"]),
            "y": np.zeros(2)}
    near = mb.prediction_contrast(model, rows, "g", "common", "rare", type="link",
                                  weights=np.array([1.0, 0.0]))
    far = mb.prediction_contrast(model, rows, "g", "common", "rare", type="link",
                                 weights=np.array([0.0, 1.0]))
    assert near.support["joint"]["supported_fraction"] == pytest.approx(1.0)
    assert far.support["joint"]["supported_fraction"] == pytest.approx(0.0)
    assert far.support["joint"]["categorical_cell_columns"] == ("g",)


def test_joint_policy_error_and_off_and_warn():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 200)
    level = np.where(x > 0.2, "common", np.where(rng.uniform(size=200) < 0.5, "common", "rare"))
    training = {"x": x, "g": level, "y": np.zeros(200)}
    model = Linear(lambda d: np.zeros(len(d["x"])), data=training, formula="y ~ x + g")
    rows = {"x": np.array([0.05, 0.95]), "g": np.array(["common", "common"]),
            "y": np.zeros(2)}
    with pytest.raises(ValueError, match="same categorical cell"):
        mb.prediction_contrast(model, rows, "g", "common", "rare", type="link",
                               joint_support="error")
    with pytest.warns(RuntimeWarning, match="same categorical cell"):
        mb.prediction_contrast(model, rows, "g", "common", "rare", type="link",
                               joint_support="warn")
    off = mb.prediction_contrast(model, rows, "g", "common", "rare", type="link",
                                 joint_support="off")
    assert off.support["joint"]["method"] == "not_computed"
    with pytest.raises(ValueError, match="joint_support must be one of"):
        mb.prediction_contrast(model, rows, "g", "common", "rare", type="link",
                               joint_support="maybe")


def test_joint_support_uses_the_supplied_rows_when_the_model_keeps_no_training_frame():
    model = Linear(lambda d: 2 * d["x"], data=None, formula="y ~ x")
    model._data = None
    data = {"x": np.linspace(0, 1, 20), "y": np.zeros(20)}
    result = mb.prediction_contrast(model, data, "x", 0.0, 1.0, type="link")
    assert result.support["joint"]["training_source"] == "standardization_rows"


def test_joint_support_shows_in_the_description():
    model, rows, _ = interaction_model_and_data()
    result = mb.prediction_contrast(model, rows, "x", 0.0, 1.0, type="link")
    assert "Joint support:" in result.describe()
    marginal = mb.average_marginal_effect(model, rows, "x", type="link")
    assert "Joint support:" in marginal.describe()


# Refit-posterior intervals for distributional contrasts.


def fitted_lss_and_refit():
    rng = np.random.default_rng(0)
    n = 240
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = 2 * x + rng.normal(scale=0.3 + 0.4 * z, size=n)
    data = {"x": x, "z": z, "y": y}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"}, data=data,
                             families=mb.GaussianLSS(), control=mb.boost_control(mstop=150))
        refit = mb.gamlss_refit(fit)
    return fit, refit, data


def test_refit_draws_give_a_posterior_interval_for_a_quantile_contrast():
    fit, refit, data = fitted_lss_and_refit()
    result = mb.distributional_contrast(
        fit, data, "x", 0.2, 0.8, quantity="quantile", value=0.9,
        refit=refit, n_draws=200, seed=1,
    )
    assert result.uncertainty == "refit_draws"
    assert result.dependence["method"] == "refit_draws"
    assert result.dependence["draws"] == 200
    lower, upper = result.interval
    assert lower < upper
    assert "Refit-posterior" in result.describe()
    again = mb.distributional_contrast(
        fit, data, "x", 0.2, 0.8, quantity="quantile", value=0.9,
        refit=refit, n_draws=200, seed=1,
    )
    assert again.interval == result.interval


def test_refit_and_cluster_intervals_are_not_combined_silently():
    fit, refit, data = fitted_lss_and_refit()
    with pytest.raises(ValueError, match="one of them"):
        mb.distributional_contrast(
            fit, data, "x", 0.2, 0.8, quantity="mean", refit=refit,
            cluster=np.arange(len(data["x"])) % 4, n_draws=50,
        )


def test_refit_must_be_a_gamlss_refit():
    fit, _, data = fitted_lss_and_refit()
    with pytest.raises(TypeError, match="gamlss_refit"):
        mb.distributional_contrast(fit, data, "x", 0.2, 0.8, quantity="mean",
                                   refit=object(), n_draws=50)
