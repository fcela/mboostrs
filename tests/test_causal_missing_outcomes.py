"""Missing outcomes, observation weights and sensitivity inputs for the causal
estimators.

The simulated designs have a known average effect. The outcome is missing at
random given the baseline covariates and the treatment, and the effect varies
with a covariate, so a complete-case analysis targets a different population.
"""

import inspect
from statistics import NormalDist

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.folds import FoldScheme, stratified_folds


def expit(x):
    return 1 / (1 + np.exp(-x))


def mar_data(seed, n=2000):
    """Treatment depends on x1 and x2. The effect is 1 + x1, whose mean over the
    population is 1. The outcome is observed with probability expit(0.5 + 1.5 x1
    - 0.5 a), so the observed rows over-represent large x1 where the effect is
    larger, and the complete-case average effect exceeds 1."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, expit(0.8 * x1 - 0.5 * x2))
    y = x1 + 0.5 * x2 + a * (1 + x1) + rng.normal(0, 1, n)
    observed = rng.binomial(1, expit(0.5 + 1.5 * x1 - 0.5 * a)).astype(bool)
    y_obs = np.where(observed, y, np.nan)
    return {"a": a, "y": y_obs, "x1": x1, "x2": x2}, 1.0


def mar_spec(estimand="ate"):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"), estimand=estimand,
        target_population="simulated population", unit="person",
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )


FAST = dict(folds=5, tune=False, mstop=200)


def test_error_message_names_the_column_the_count_and_both_remedies():
    data, _ = mar_data(seed=0)
    missing = int(np.isnan(data["y"]).sum())
    with pytest.raises(ValueError) as caught:
        mb.aipw(data, mar_spec(), **FAST)
    message = str(caught.value)
    assert "'y'" in message
    assert f"missing for {missing} of 2000 rows" in message
    assert "outcome_missing='ipw'" in message
    assert "drop" in message


def test_complete_case_aipw_is_biased_under_missing_at_random():
    data, truth = mar_data(seed=0)
    observed = ~np.isnan(data["y"])
    complete = {name: values[observed] for name, values in data.items()}
    result = mb.aipw(complete, mar_spec(), **FAST)
    assert result.estimate - truth > 0.25


def test_ipw_route_recovers_the_effect_within_its_interval_in_most_seeds():
    covered, errors = 0, []
    for seed in range(20):
        data, truth = mar_data(seed)
        result = mb.aipw(data, mar_spec(), outcome_missing="ipw", **FAST)
        lower, upper = result.interval
        covered += lower <= truth <= upper
        errors.append(result.estimate - truth)
    assert covered >= 16, f"covered {covered} of 20"
    assert np.mean(np.abs(errors)) < 0.15


def test_ipw_result_reports_the_response_model_and_targets_all_rows():
    data, _ = mar_data(seed=3)
    result = mb.aipw(data, mar_spec(), outcome_missing="ipw", **FAST)
    observed = ~np.isnan(data["y"])
    assert len(result.scores) == len(observed)
    np.testing.assert_array_equal(result.observed, observed.astype(float))
    d = result.diagnostics
    assert d["outcome_missing"] == "ipw"
    assert d["n_missing_outcome"] == int((~observed).sum())
    assert d["response_model"] == "fitted"
    assert 0 < d["response_propensity_min"] <= d["response_propensity_max"] < 1
    assert 1 < d["response_effective_sample_size"] <= observed.sum()
    assert result.response_propensity.min() == pytest.approx(d["response_propensity_min"])


def test_ipw_score_is_the_augmented_formula_with_clipped_propensities():
    data, _ = mar_data(seed=5)
    result = mb.aipw(data, mar_spec(), outcome_missing="ipw", **FAST)
    a, y = result.treatment, result.outcome
    observed = result.observed
    lo, hi = result.diagnostics["propensity_clip"]
    e = np.clip(result.propensity, lo, hi)
    pi = np.clip(result.response_propensity, lo, hi)
    r = observed
    y0 = np.where(r == 1, y, 0.0)
    expected = (result.mu1 - result.mu0
                + r * a / (pi * e) * (y0 - result.mu1)
                - r * (1 - a) / (pi * (1 - e)) * (y0 - result.mu0))
    np.testing.assert_allclose(result.scores, expected, rtol=1e-12, atol=1e-12)
    assert result.estimate == pytest.approx(expected.mean(), rel=1e-12)


def test_outcome_missing_with_no_missing_rows_is_the_complete_case_estimate():
    data, _ = mar_data(seed=1)
    observed = ~np.isnan(data["y"])
    complete = {name: values[observed] for name, values in data.items()}
    plain = mb.aipw(complete, mar_spec(), **FAST)
    routed = mb.aipw(complete, mar_spec(), outcome_missing="ipw", **FAST)
    assert routed.diagnostics["response_model"].startswith("not fitted")
    assert routed.estimate == pytest.approx(plain.estimate, rel=1e-12)


def test_outcome_missing_argument_is_checked():
    data, _ = mar_data(seed=0, n=400)
    with pytest.raises(ValueError, match="outcome_missing must be 'error' or 'ipw'"):
        mb.aipw(data, mar_spec(), outcome_missing="drop", **FAST)


def test_integer_weights_equal_the_estimator_on_replicated_rows():
    """The nuisance boosting paths of a weighted and a replicated fit agree once
    they have converged (mstop=1000 here). At a few hundred steps they differ
    by about 1e-3, because the learners centre their design by the unweighted
    mean; that limitation is stated in the aipw docstring."""
    data, _ = mar_data(seed=7, n=300)
    rng = np.random.default_rng(11)
    w = rng.integers(1, 4, size=300)
    rep = {name: np.repeat(values, w) for name, values in data.items()}

    base = stratified_folds(data["a"], B=4, seed=3)
    scheme_rep = FoldScheme(
        weights=np.repeat(base.weights, w, axis=0),
        held_out=np.repeat(base.held_out, w, axis=0),
        scheme=base.scheme, seed=base.seed,
    )
    kwargs = dict(outcome_missing="ipw", tune=False, mstop=1000)
    weighted = mb.aipw(data, mar_spec(), folds=base, weights=w, **kwargs)
    replicated = mb.aipw(rep, mar_spec(), folds=scheme_rep, **kwargs)
    assert weighted.estimate == pytest.approx(replicated.estimate, rel=1e-6)
    assert weighted.standard_error == pytest.approx(replicated.standard_error, rel=1e-6)
    assert weighted.diagnostics["weight_total"] == pytest.approx(w.sum())


def test_weights_must_match_the_sample():
    data, _ = mar_data(seed=0, n=400)
    observed = ~np.isnan(data["y"])
    data = {name: values[observed] for name, values in data.items()}
    with pytest.raises(ValueError, match="weights has shape"):
        mb.aipw(data, mar_spec(), weights=np.ones(10), **FAST)
    with pytest.raises(ValueError, match="non-negative"):
        mb.aipw(data, mar_spec(), weights=-np.ones(len(data["a"])), **FAST)


def test_effect_model_with_ipw_recovers_the_modifier():
    data, _ = mar_data(seed=2, n=1500)
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )
    fx = mb.effect_model(data, spec, outcome_missing="ipw", terms="smooth", seed=1)
    assert fx.diagnostics["outcome_missing"] == "ipw"
    assert fx.diagnostics["response_model"] == "fitted"
    assert np.corrcoef(fx.tau, data["x1"])[0, 1] > 0.8


def test_propensity_clip_default_is_the_same_in_aipw_and_effect_model():
    # Both defer to the spec's propensity_clip, whose default is (0.01, 0.99).
    from mboostrs.causal import FROM_SPEC

    aipw_default = inspect.signature(mb.aipw).parameters["propensity_clip"].default
    effect_default = inspect.signature(mb.effect_model).parameters["propensity_clip"].default
    assert aipw_default is effect_default is FROM_SPEC
    assert mar_spec().propensity_clip == (0.01, 0.99)


def test_confounding_sensitivity_estimate_path_matches_the_result_path():
    data, _ = mar_data(seed=4, n=800)
    result = mb.aipw(data, mar_spec(), outcome_missing="ipw", **FAST)
    from_result = mb.confounding_sensitivity(
        result, outcome_effects=[0, 1, 2], imbalances=[0, .5], output="records")
    from_estimate = mb.confounding_sensitivity(
        estimate=result.estimate, se=result.standard_error, level=result.level,
        outcome_effects=[0, 1, 2], imbalances=[0, .5], output="records")
    from_scores = mb.confounding_sensitivity(
        scores=result.scores, level=result.level,
        outcome_effects=[0, 1, 2], imbalances=[0, .5], output="records")
    assert from_result[0]["lower"] is not None
    for other in (from_estimate, from_scores):
        assert len(other) == len(from_result)
        for row, reference in zip(other, from_result, strict=True):
            for key, value in reference.items():
                if isinstance(value, float):
                    assert row[key] == pytest.approx(value, rel=1e-10, abs=1e-12), key
                else:
                    assert row[key] == value, key


def test_confounding_sensitivity_without_se_has_no_interval():
    rows = mb.confounding_sensitivity(
        estimate=1.5, outcome_effects=[1], imbalances=[0.5], output="records")
    assert rows[0]["adjusted_estimate"] == pytest.approx(1.0)
    assert rows[0]["lower"] is None and rows[0]["interval_includes_zero"] is None


def test_confounding_sensitivity_interval_uses_the_normal_critical_value():
    rows = mb.confounding_sensitivity(
        estimate=1.0, se=0.5, level=.9, outcome_effects=[0], imbalances=[0],
        output="records")
    critical = NormalDist().inv_cdf(.95)
    assert rows[0]["lower"] == pytest.approx(1.0 - critical * 0.5)
    assert rows[0]["upper"] == pytest.approx(1.0 + critical * 0.5)


def test_confounding_sensitivity_needs_exactly_one_kind_of_input():
    data, _ = mar_data(seed=0, n=400)
    result = mb.aipw(data, mar_spec(), outcome_missing="ipw", **FAST)
    with pytest.raises(TypeError, match="not both"):
        mb.confounding_sensitivity(result, estimate=1.0, outcome_effects=[0],
                                   imbalances=[0])
    with pytest.raises(TypeError, match="pass result"):
        mb.confounding_sensitivity(outcome_effects=[0], imbalances=[0])


def test_effect_model_refuses_a_missing_outcome_by_default():
    data, _ = mar_data(seed=2, n=300)
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )
    with pytest.raises(ValueError, match="outcome 'y' is missing for"):
        mb.effect_model(data, spec, terms="smooth")


def test_effect_model_reuses_ipw_nuisances_only_with_the_same_option():
    data, _ = mar_data(seed=2, n=800)
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )
    result = mb.aipw(data, spec, outcome_missing="ipw", **FAST)
    fx = mb.effect_model(data, spec, nuisance=result, outcome_missing="ipw",
                         terms="smooth", modifiers=("x1",), seed=1)
    np.testing.assert_allclose(fx.propensity, result.propensity)
    np.testing.assert_allclose(fx.outcome_mean,
                               result.propensity * result.mu1
                               + (1 - result.propensity) * result.mu0)
    # On complete rows the option is not needed to fit, so a mismatch between
    # the result and the call is the error that is reached.
    observed = ~np.isnan(data["y"])
    complete = {name: values[observed] for name, values in data.items()}
    complete_result = mb.aipw(complete, spec, outcome_missing="ipw", **FAST)
    with pytest.raises(ValueError, match="outcome_missing='ipw'"):
        mb.effect_model(complete, spec, nuisance=complete_result, terms="smooth",
                        modifiers=("x1",))
