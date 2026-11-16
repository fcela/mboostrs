"""Median-rule and bootstrap intervals: aipw repeats and effect_model interval_method."""

from statistics import NormalDist

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust


def spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2", "x3"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes",
    )


def data_set(n=300, seed=7):
    """Confounded treatment and an additive effect tau(x) = 1 + 0.8 x1 + 0.5 sin(2 x2)."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.uniform(-1.5, 1.5, (3, n))
    e = 1 / (1 + np.exp(-(0.8 * x1 + 0.6 * x2)))
    a = rng.binomial(1, e).astype(float)
    tau = 1 + 0.8 * x1 + 0.5 * np.sin(2 * x2)
    y = np.sin(x1) + 0.7 * x2 + 0.3 * x3 + tau * a + rng.normal(0, 0.5, n)
    return {"a": a, "y": y, "x1": x1, "x2": x2, "x3": x3}


def manual_median_rule(estimates, ses):
    """The rule of Chernozhukov et al. (2018), written out for checking the binding."""
    estimates, ses = np.asarray(estimates, float), np.asarray(ses, float)
    center = np.median(estimates, axis=0)
    variance = np.median(ses**2 + (estimates - center) ** 2, axis=0)
    return center, np.sqrt(variance)


def test_the_binding_applies_the_median_rule():
    estimates = np.array([[1.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    ses = np.array([[0.5, 0.0], [0.5, 0.0], [1.0, 0.0]])
    center, combined = _rust.median_repeat_combine(estimates, ses)
    # Median of 1, 2, 4 is 2; variance terms 1.25, 0.25, 5 have median 1.25.
    np.testing.assert_allclose(center, [2.0, 5.0])
    np.testing.assert_allclose(combined, [np.sqrt(1.25), 0.0])


def test_the_binding_matches_the_rule_on_random_input():
    rng = np.random.default_rng(3)
    for repeats in (2, 5):
        estimates = rng.normal(size=(repeats, 17))
        ses = rng.uniform(0.1, 2.0, size=(repeats, 17))
        center, combined = _rust.median_repeat_combine(estimates, ses)
        want_center, want_se = manual_median_rule(estimates, ses)
        np.testing.assert_allclose(center, want_center, rtol=1e-12)
        np.testing.assert_allclose(combined, want_se, rtol=1e-12)


def test_the_binding_gives_type_seven_percentile_limits():
    draws = np.random.default_rng(0).normal(size=(40, 6))
    lower, upper, se = _rust.bootstrap_percentile_interval(draws, 0.9)
    np.testing.assert_allclose(lower, np.quantile(draws, 0.05, axis=0))
    np.testing.assert_allclose(upper, np.quantile(draws, 0.95, axis=0))
    np.testing.assert_allclose(se, draws.std(axis=0, ddof=1))


@pytest.fixture(scope="module")
def aipw_fits():
    data = data_set()
    kwargs = dict(folds=3, repeats=3, seed=11, tune=False, mstop=60,
                  outcome_missing="error")
    return {
        "average": mb.aipw(data, spec(), **kwargs),
        "median": mb.aipw(data, spec(), repeat_aggregation="median", **kwargs),
        "single": mb.aipw(data, spec(), folds=3, repeats=1, seed=11, tune=False, mstop=60),
        "single_median": mb.aipw(data, spec(), folds=3, repeats=1, seed=11, tune=False,
                                 mstop=60, repeat_aggregation="median"),
    }


def test_repeat_aggregation_is_average_or_median():
    with pytest.raises(ValueError, match="repeat_aggregation"):
        mb.aipw(data_set(), spec(), folds=3, repeats=2, tune=False, mstop=20,
                repeat_aggregation="mean")


def test_the_average_default_reports_each_repeat_and_keeps_the_averaged_scores(aipw_fits):
    fit = aipw_fits["average"]
    assert len(fit.diagnostics["repeat_estimates"]) == 3
    assert len(fit.diagnostics["repeat_standard_errors"]) == 3
    assert fit.nuisance["split_randomness_in_variance"] is False
    assert fit.estimate == pytest.approx(np.mean(fit.scores))


def test_the_median_estimate_and_standard_error_follow_the_rule(aipw_fits):
    fit = aipw_fits["median"]
    estimates = fit.diagnostics["repeat_estimates"]
    ses = fit.diagnostics["repeat_standard_errors"]
    center, combined = manual_median_rule([[e] for e in estimates], [[s] for s in ses])
    assert fit.estimate == pytest.approx(center[0])
    assert fit.standard_error == pytest.approx(combined[0])
    assert fit.nuisance["split_randomness_in_variance"] is True
    z = NormalDist().inv_cdf(0.975)
    assert fit.interval[0] == pytest.approx(fit.estimate - z * fit.standard_error)
    assert fit.interval[1] == pytest.approx(fit.estimate + z * fit.standard_error)


def test_the_median_rule_is_the_average_when_there_is_one_repeat(aipw_fits):
    single, single_median = aipw_fits["single"], aipw_fits["single_median"]
    assert single_median.estimate == pytest.approx(single.estimate)
    assert single_median.standard_error == pytest.approx(single.standard_error)


@pytest.fixture(scope="module")
def effect_fits():
    data = data_set()
    common = dict(terms="smooth", seed=5, folds=3)
    return {
        "data": data,
        "refit": mb.effect_model(data, spec(), interval_method="refit", **common),
        "repeats": mb.effect_model(data, spec(), interval_method="repeats",
                                   interval_repeats=3, **common),
        "bootstrap": mb.effect_model(data, spec(), interval_method="bootstrap",
                                     bootstrap_draws=4, **common),
    }


NEW_ROWS = {"x1": np.linspace(-1.2, 1.2, 40), "x2": np.linspace(1.0, -1.0, 40),
            "x3": np.linspace(-0.5, 0.5, 40)}


def test_refit_has_no_refits(effect_fits):
    fit = effect_fits["refit"]
    assert fit.interval_method == "refit"
    assert fit.interval_fits == ()
    assert fit.second_stage is None
    assert fit.diagnostics["interval"]["method"] == "refit"
    explicit = mb.effect_model(effect_fits["data"], spec(), terms="smooth", seed=5, folds=3,
                               interval_method="refit")
    np.testing.assert_allclose(fit.predict(NEW_ROWS), explicit.predict(NEW_ROWS))


def test_the_default_interval_is_the_second_stage_bootstrap(effect_fits):
    """interval_method="auto" (the default) gives the second-stage bootstrap for a smooth fit."""
    fit = mb.effect_model(effect_fits["data"], spec(), terms="smooth", seed=5, folds=3,
                          bootstrap_draws=20)
    assert fit.interval_method == "second_stage_bootstrap"
    assert fit.second_stage.n_draws == 20
    assert fit.interval_fits == ()
    refit = effect_fits["refit"]
    # The point estimate is the full-sample fit under either method.
    np.testing.assert_allclose(fit.predict(NEW_ROWS), refit.predict(NEW_ROWS))


def test_repeats_fit_one_refit_per_extra_split_seed(effect_fits):
    fit = effect_fits["repeats"]
    assert len(fit.interval_fits) == 2
    assert fit.diagnostics["interval"]["repeats"] == 3
    assert not np.allclose(fit.interval_fits[0].tau, fit.tau)


def test_repeats_estimate_is_the_median_over_the_splits(effect_fits):
    fit = effect_fits["repeats"]
    taus = np.vstack([f.predict(NEW_ROWS) for f in (fit, *fit.interval_fits)])
    np.testing.assert_allclose(fit.predict(NEW_ROWS), np.median(taus, axis=0))


def test_repeats_interval_follows_the_median_rule(effect_fits):
    fit = effect_fits["repeats"]
    frame = fit.predict(NEW_ROWS, interval=True, level=0.9)
    parts = [f.predict(NEW_ROWS, interval=True, level=0.9)
             for f in (fit, *fit.interval_fits)]
    estimates = np.vstack([p["estimate"].to_numpy() for p in parts])
    ses = np.vstack([p["se"].to_numpy() for p in parts])
    center, combined = manual_median_rule(estimates, ses)
    z = NormalDist().inv_cdf(0.95)
    np.testing.assert_allclose(frame["estimate"].to_numpy(), center, rtol=1e-10)
    np.testing.assert_allclose(frame["se"].to_numpy(), combined, rtol=1e-10)
    np.testing.assert_allclose(frame["lower"].to_numpy(), center - z * combined, rtol=1e-10)
    np.testing.assert_allclose(frame["upper"].to_numpy(), center + z * combined, rtol=1e-10)


def test_bootstrap_refits_on_resampled_rows_of_the_same_size(effect_fits):
    fit = effect_fits["bootstrap"]
    n = len(effect_fits["data"]["a"])
    assert len(fit.interval_fits) == 4
    assert all(len(f.tau) == n for f in fit.interval_fits)
    assert all(f.terms == "smooth" for f in fit.interval_fits)


def test_bootstrap_interval_is_the_percentile_of_the_refits(effect_fits):
    fit = effect_fits["bootstrap"]
    draws = np.vstack([f.predict(NEW_ROWS) for f in fit.interval_fits])
    frame = fit.predict(NEW_ROWS, interval=True, level=0.9)
    np.testing.assert_allclose(frame["estimate"].to_numpy(), fit.predict(NEW_ROWS))
    np.testing.assert_allclose(frame["lower"].to_numpy(), np.quantile(draws, 0.05, axis=0))
    np.testing.assert_allclose(frame["upper"].to_numpy(), np.quantile(draws, 0.95, axis=0))
    np.testing.assert_allclose(frame["se"].to_numpy(), draws.std(axis=0, ddof=1))


def test_the_refit_methods_refuse_boosted_candidates_and_supplied_nuisance():
    data = data_set()
    with pytest.raises(ValueError, match="interval_method must be one of"):
        mb.effect_model(data, spec(), interval_method="jackknife")
    with pytest.raises(ValueError, match="smooth second stage"):
        mb.effect_model(data, spec(), terms="stumps", interval_method="repeats")
    nuisance = mb.aipw(data, spec(), folds=3, tune=False, mstop=20)
    with pytest.raises(ValueError, match="nuisance=None"):
        mb.effect_model(data, spec(), nuisance=nuisance, interval_method="bootstrap")
