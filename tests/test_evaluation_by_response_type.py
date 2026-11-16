"""Evaluation by response type: counts, CRPS, AUC uncertainty, and named limits.

Every expected value below comes from a closed form or from scipy, not from
the code under test. The Rust kernels are exercised through the Python
wrappers that call them.
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

import mboostrs as mb
from mboostrs.distribution_evaluation import _crps, _crps_quadrature
from mboostrs.gamlss import (
    GammaLSS,
    GaussianLSS,
    NBinomialLSS,
    PoissonLSS,
    ZeroAdjustedGammaLSS,
)


def _result(records, *, prediction_scale="response", family="Poisson",
            independent_unit="record", score_name="mse"):
    n = len(records)
    return mb.EvaluationResult(
        score=.4, fold_scores=np.array([.4]), formula="y ~ x",
        response="y", family=family, score_name=score_name,
        score_parameter=None, prediction_scale=prediction_scale,
        loss_units="units", aggregation="equal_fold_mean",
        fold_scheme="supplied", evaluation_task="new_exchangeable_observation",
        independent_unit=independent_unit, n_independent_units=n, seed=1,
        tuning="fixed_mstop=1", data_fingerprint="3" * 64,
        fingerprint_scope="all_columns", n_observations=n, n_evaluations=n,
        fit_count=1, excluded_record_ids=(), records=tuple(records),
        folds=({"fold": 0, "score": .4},),
    )


def _records(observed, prediction, weights=None):
    weights = np.ones(len(observed)) if weights is None else weights
    return tuple({
        "record_id": f"r{i:04d}", "row": i, "fold": 0,
        "observed": float(observed[i]), "prediction": float(prediction[i]),
        "evaluation_weight": float(weights[i]), "loss": np.nan,
    } for i in range(len(observed)))


class _FixedModel:
    """Stands in for a fitted LSS model whose parameters are given directly."""

    def __init__(self, family, response, params):
        self._family = family
        self._response_name = response
        self._params = params

    def predict(self, *, newdata, type, mstop=None):
        assert type == "response"
        return {name: np.asarray(value, dtype=np.float64) * np.ones(len(newdata["y"]))
                for name, value in self._params.items()}


# Counts ---------------------------------------------------------------------


def test_poisson_log_score_and_deviance_match_scipy_with_exposure():
    rng = np.random.default_rng(31)
    n = 60
    exposure = rng.uniform(.5, 3.0, n)
    true_rate = np.exp(-.4 + .3 * rng.normal(size=n))
    observed = rng.poisson(true_rate * exposure).astype(float)
    predicted_rate = true_rate * 1.1
    weights = rng.uniform(.5, 2.0, n)
    result = _result(_records(observed, predicted_rate, weights))

    metrics = result.count_metrics(exposure=exposure, prediction_quantity="rate")

    expected_count = predicted_rate * exposure
    log_score = -stats.poisson.logpmf(observed, expected_count)
    deviance = 2 * (stats.poisson.logpmf(observed, np.maximum(observed, 1e-300))
                    - stats.poisson.logpmf(observed, expected_count))
    assert metrics["mean_poisson_log_score"] == pytest.approx(
        np.average(log_score, weights=weights), rel=1e-12)
    assert metrics["mean_poisson_deviance"] == pytest.approx(
        np.average(deviance, weights=weights), rel=1e-10)
    assert metrics["observed_rate"] == pytest.approx(
        observed @ weights / (exposure @ weights), rel=1e-12)
    assert metrics["count_unit"] == "events per row exposure"


def test_count_scores_without_exposure_treat_each_row_as_one_unit():
    observed = np.array([0.0, 3.0, 1.0, 4.0])
    expected = np.array([.5, 2.0, 1.5, 3.0])
    result = _result(_records(observed, expected), family="Poisson")
    metrics = result.count_metrics()
    assert metrics["exposure_supplied"] is False
    assert metrics["mean_poisson_log_score"] == pytest.approx(
        np.mean(-stats.poisson.logpmf(observed, expected)), rel=1e-12)


# CRPS -----------------------------------------------------------------------


def test_gaussian_quadrature_reproduces_the_closed_form():
    rng = np.random.default_rng(5)
    y = rng.normal(size=40)
    mu = rng.normal(size=40) * .5
    sigma = np.exp(rng.normal(size=40) * .3)
    params = {"mu": mu, "sigma": sigma}
    closed = _crps(GaussianLSS(), y, params)
    quadrature = _crps_quadrature(GaussianLSS(), y, params)
    np.testing.assert_allclose(quadrature, closed, rtol=0, atol=1e-7)


def test_gaussian_crps_matches_the_standard_normal_definition():
    y, mu, sigma = np.array([-1.0]), np.array([0.0]), np.array([0.7])
    left = quad(lambda x: stats.norm.cdf(x, 0, .7) ** 2, -np.inf, -1.0)[0]
    right = quad(lambda x: (1 - stats.norm.cdf(x, 0, .7)) ** 2, -1.0, np.inf)[0]
    got = _crps(GaussianLSS(), y, {"mu": mu, "sigma": sigma})
    assert got[0] == pytest.approx(left + right, abs=1e-9)


def test_gamma_crps_at_shape_one_matches_the_exponential_closed_form():
    # Gamma with shape 1 is exponential with mean m: CRPS = y - 3m/2 + 2m exp(-y/m).
    m = 2.5
    y = np.array([0.0, .4, 2.5, 9.0])
    params = {"mu": np.full(4, m), "sigma": np.ones(4)}
    got = _crps(GammaLSS(), y, params)
    expected = y - 1.5 * m + 2 * m * np.exp(-y / m)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_poisson_crps_matches_the_scipy_threshold_sum():
    mu = np.array([.7, 4.0, 12.0])
    y = np.array([0.0, 3.0, 15.0])
    thresholds = np.arange(0, 400)
    expected = []
    for observed, rate in zip(y, mu, strict=True):
        cdf = stats.poisson.cdf(thresholds, rate)
        expected.append(np.sum((cdf - (observed <= thresholds)) ** 2))
    got = _crps(PoissonLSS(), y, {"mu": mu})
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_negative_binomial_crps_matches_scipy_and_its_cdf_parameterization():
    mu, size = 3.0, 2.0
    y = np.array([0.0, 2.0, 7.0])
    scipy_nb = stats.nbinom(size, size / (size + mu))
    thresholds = np.arange(0, 600)
    family = NBinomialLSS()
    params = {"mu": np.full(3, mu), "sigma": np.full(3, size)}
    np.testing.assert_allclose(
        family.cdf(np.full(3, 4.0), params), scipy_nb.cdf(4.0), atol=1e-12)
    expected = [np.sum((scipy_nb.cdf(thresholds) - (value <= thresholds)) ** 2)
                for value in y]
    np.testing.assert_allclose(_crps(family, y, params), expected, rtol=0, atol=1e-10)


def test_zero_adjusted_gamma_crps_matches_an_independent_integral():
    mu, sigma, nu = 2.0, .8, .3
    shape, scale = 1 / sigma**2, mu * sigma**2
    gamma = stats.gamma(shape, scale=scale)

    def cdf(t):
        return nu + (1 - nu) * gamma.cdf(t)

    y = np.array([0.0, .5, 3.0])
    params = {"mu": np.full(3, mu), "sigma": np.full(3, sigma), "nu": np.full(3, nu)}
    got = _crps(ZeroAdjustedGammaLSS(), y, params)
    for index, observed in enumerate(y):
        left = 0.0 if observed == 0 else quad(lambda t: cdf(t) ** 2, 0, observed)[0]
        right = quad(lambda t: (1 - cdf(t)) ** 2, observed, np.inf)[0]
        assert got[index] == pytest.approx(left + right, abs=1e-6)


def test_distribution_result_reports_crps_for_a_non_gaussian_family():
    model = _FixedModel(
        ZeroAdjustedGammaLSS(), "y",
        {"mu": 2.0, "sigma": .8, "nu": .3},
    )
    data = {"y": np.array([0.0, .5, 1.0, 3.0])}
    result = mb.evaluate_distribution(
        model, data, data_role="external", levels=(.8,), random_state=3,
    )
    assert result.crps is not None
    assert result.metrics["mean_crps"] == pytest.approx(np.mean(result.crps))
    assert np.all(result.crps >= 0)
    assert not any("CRPS is not reported" in note for note in result.assumptions)


def test_count_crps_is_unavailable_for_non_integer_outcomes():
    params = {"mu": np.array([2.0, 2.0])}
    assert _crps(PoissonLSS(), np.array([1.5, 2.0]), params) is None
    with pytest.raises(ValueError, match="non-negative integer"):
        mb.evaluate_distribution(
            _FixedModel(PoissonLSS(), "y", {"mu": 2.0}),
            {"y": np.array([0.0, 1.5, 2.0])}, data_role="external", levels=(.8,),
        )


def test_crps_is_withheld_for_a_transformed_response():
    rng = np.random.default_rng(8)
    x = rng.uniform(-1, 1, 200)
    y = np.exp(.5 + .4 * x + .3 * rng.normal(size=200))
    fit = mb.glmboostLSS(
        "log(y) ~ bols(x)", {"x": x, "y": y}, families=mb.GaussianLSS(),
        control=mb.boost_control(60, .1),
    )
    holdout = {"x": x[:50], "y": y[:50]}
    result = mb.evaluate_distribution(fit, holdout, data_role="external", levels=(.8,))
    assert result.crps is None
    assert result.metrics["mean_crps"] is None
    assert any("CRPS is not reported" in note and "transform" in note
               for note in result.assumptions)


# AUC uncertainty ------------------------------------------------------------


def _delong_oracle(observed, scores):
    """Unweighted DeLong variance, computed by enumerating every pair."""
    observed = np.asarray(observed, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positives = scores[:, observed == 1]
    negatives = scores[:, observed == 0]
    m, n = positives.shape[1], negatives.shape[1]
    psi = np.zeros((scores.shape[0], m, n))
    for a in range(scores.shape[0]):
        p = positives[a][:, None]
        q = negatives[a][None, :]
        psi[a] = (p > q) + .5 * (p == q)
    v = psi.mean(axis=2)
    u = psi.mean(axis=1)
    auc = v.mean(axis=1)
    cov = np.atleast_2d(np.cov(v, ddof=1)) / m + np.atleast_2d(np.cov(u, ddof=1)) / n
    return auc, cov


def test_auc_interval_matches_the_pairwise_delong_oracle_under_integer_weights():
    rng = np.random.default_rng(77)
    n = 40
    observed = rng.binomial(1, .4, n).astype(float)
    score = observed * .8 + rng.normal(size=n)
    frequency = rng.integers(1, 4, n)
    result = _result(_records(observed, np.clip(1 / (1 + np.exp(-score)), 0, 1),
                              weights=frequency.astype(float)),
                     family="Binomial", score_name="log_loss")
    interval = result.auc_interval(level=.9)

    # Expanding each row into its frequency gives the same estimator without weights.
    expanded_y = np.repeat(observed, frequency)
    expanded_s = np.repeat(1 / (1 + np.exp(-score)), frequency)
    auc, cov = _delong_oracle(expanded_y, expanded_s[None, :])
    assert interval["auc"] == pytest.approx(auc[0], abs=1e-12)
    assert interval["standard_error"] == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-10)
    z = 1.6448536269514722  # two-sided 90% normal quantile
    assert interval["lower"] == pytest.approx(auc[0] - z * math.sqrt(cov[0, 0]), abs=1e-9)
    assert interval["upper"] == pytest.approx(min(1, auc[0] + z * math.sqrt(cov[0, 0])), abs=1e-9)


def test_paired_auc_difference_uses_the_shared_records_covariance():
    rng = np.random.default_rng(12)
    n = 50
    observed = rng.binomial(1, .5, n).astype(float)
    a_score = observed + rng.normal(scale=1.0, size=n)
    b_score = observed + rng.normal(scale=1.2, size=n)
    to_prob = lambda s: 1 / (1 + np.exp(-s))  # noqa: E731
    first = _result(_records(observed, to_prob(a_score)), family="Binomial",
                    score_name="log_loss")
    second = _result(_records(observed, to_prob(b_score)), family="Binomial",
                     score_name="log_loss")
    got = first.auc_difference(second, level=.95)

    auc, cov = _delong_oracle(observed, np.vstack([to_prob(a_score), to_prob(b_score)]))
    assert got["difference"] == pytest.approx(auc[0] - auc[1], abs=1e-12)
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    assert got["standard_error"] == pytest.approx(math.sqrt(variance), rel=1e-9)


def test_paired_auc_difference_refuses_unpaired_records():
    observed = np.array([0.0, 1.0, 0.0, 1.0])
    prob = np.array([.2, .7, .4, .9])
    first = _result(_records(observed, prob), family="Binomial", score_name="log_loss")
    other = _result(_records(observed[:3], prob[:3]), family="Binomial",
                    score_name="log_loss")
    with pytest.raises(ValueError, match="paired"):
        first.auc_difference(other)


def test_auc_interval_names_the_record_independence_it_assumes():
    observed = np.array([0.0, 1.0, 0.0, 1.0])
    prob = np.array([.2, .7, .4, .9])
    grouped = _result(_records(observed, prob), family="Binomial",
                      score_name="log_loss", independent_unit="group")
    with pytest.raises(ValueError, match="independent unit"):
        grouped.auc_interval()
    with pytest.raises(ValueError, match="level"):
        _result(_records(observed, prob), family="Binomial",
                score_name="log_loss").auc_interval(level=1.5)


def test_delong_refuses_a_class_without_enough_weight():
    observed = np.array([0.0, 1.0, 0.0])
    prob = np.array([.2, .7, .4])
    result = _result(_records(observed, prob), family="Binomial",
                     score_name="log_loss")
    with pytest.raises(ValueError, match="total weight above one"):
        result.auc_interval()
