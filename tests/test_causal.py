"""Causal specifications and cross-fitted AIPW estimators."""

import numpy as np
import pytest

import mboostrs as mb


def confounded(n=1200, seed=4, effect=2.):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-1.5 * x))
    a = (rng.random(n) < e).astype(float)
    y = effect * a + 2 * x + rng.normal(size=n)
    return {"x": x, "a": a, "y": y}


def spec(estimand="ate"):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",), estimand=estimand,
        target_population="study sample", unit="person",
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x causes both treatment and outcome",
    )


def _clipped(result):
    """The propensity aipw used: clipped to the result's declared bounds."""
    lower, upper = result.diagnostics["propensity_clip"]
    return np.clip(result.propensity, lower, upper)


def test_spec_records_the_intervention_and_assumptions():
    item = spec()
    text = item.describe()
    assert "setting a from 0 to 1" in text
    assert "ATE" in text and "one year" in text
    assert any("exchangeability" in assumption for assumption in item.assumptions)


def test_post_treatment_adjustment_is_refused():
    with pytest.raises(ValueError, match="post-treatment"):
        mb.CausalSpec("a", "y", ("mediator",), post_treatment=("mediator",),
                      time_zero="assignment", follow_up="one year",
                      adjustment_rationale="baseline design")


@pytest.mark.parametrize("missing", ["time", "follow_up", "rationale"])
def test_causal_design_requires_timing_and_adjustment_reason(missing):
    arguments = dict(treatment="a", outcome="y", covariates=("x",),
                     time_zero="assignment", follow_up="one year",
                     adjustment_rationale="x is the common cause")
    arguments[{"time": "time_zero", "follow_up": "follow_up",
               "rationale": "adjustment_rationale"}[missing]] = ""
    with pytest.raises(ValueError, match="time_zero|adjustment_rationale"):
        mb.CausalSpec(**arguments)


def test_cross_fitted_ate_recovers_known_effect_and_beats_naive_difference():
    data = confounded()
    result = mb.aipw(data, spec(), folds=5, mstop=150, tune=False)
    naive = data["y"][data["a"] == 1].mean() - data["y"][data["a"] == 0].mean()
    assert result.estimate == pytest.approx(2., abs=.18)
    assert abs(result.estimate - 2) < abs(naive - 2)
    assert result.interval[0] < 2 < result.interval[1]
    assert set(result.fold_id) == set(range(5))
    assert repr(result).startswith("AIPWResult(ate=")
    assert "array(" not in repr(result)


def test_arbitrary_treatment_labels_are_recoded_for_the_propensity_model():
    data = confounded(n=600)
    data["a"] = np.where(data["a"] == 1, "treated", "control")
    item = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",),
        treatment_contrast=("control", "treated"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="x is the common cause",
    )
    result = mb.aipw(data, item, folds=4, mstop=100, tune=False)
    assert result.estimate == pytest.approx(2, abs=.3)
    assert set(result.treatment) == {0, 1}


def test_binary_outcome_estimates_a_marginal_risk_difference():
    rng = np.random.default_rng(18)
    n = 2400
    x = rng.normal(size=n)
    a = rng.binomial(1, .5, size=n)
    p0 = 1 / (1 + np.exp(-(-.7 + .8 * x)))
    p1 = 1 / (1 + np.exp(-(-.7 + .8 * x + .9)))
    y = rng.binomial(1, np.where(a == 1, p1, p0))
    item = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",), outcome_type="binary",
        time_zero="randomization", follow_up="six months",
        adjustment_rationale="x is a prespecified prognostic baseline variable",
    )
    result = mb.aipw({"x": x, "a": a, "y": y}, item, folds=5,
                     mstop=100, tune=False)
    assert result.estimate == pytest.approx(float(np.mean(p1 - p0)), abs=.04)
    assert result.diagnostics["effect_scale"] == "risk_difference"
    assert result.interval[0] < np.mean(p1 - p0) < result.interval[1]


def test_binary_outcome_requires_zero_one_coding():
    data = confounded(n=100)
    data["y"] = np.where(data["y"] > np.median(data["y"]), 2, 0)
    item = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",), outcome_type="binary",
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="x is the common cause",
    )
    with pytest.raises(ValueError, match="coded with values 0 and 1"):
        mb.aipw(data, item, tune=False)


def test_row_scores_reconstruct_the_ate_exactly():
    result = mb.aipw(confounded(n=500), spec(), folds=4, mstop=100, tune=False)
    a, y, m0, m1 = (result.treatment, result.outcome, result.mu0, result.mu1)
    e = _clipped(result)
    expected = m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)
    np.testing.assert_allclose(result.scores, expected)
    assert result.estimate == pytest.approx(expected.mean())
    assert result.influence.mean() == pytest.approx(0, abs=1e-14)
    assert len(result.to_frame(output="records")) == len(y)


def test_repeated_cross_fitting_averages_only_out_of_fold_nuisance_predictions():
    data = confounded(n=500)
    result = mb.aipw(data, spec(), folds=4, repeats=3, seed=19,
                     mstop=60, tune=False)
    assert result.fold_ids.shape == (500, 3)
    assert np.array_equal(result.fold_id, result.fold_ids[:, 0])
    assert all(set(result.fold_ids[:, repeat]) == set(range(4))
               for repeat in range(3))
    assert result.nuisance["fold_repeats"] == 3
    assert len(result.nuisance["selected_mstop_by_repeat"]) == 3
    assert result.nuisance["split_randomness_in_variance"] is False

    a, y, m0, m1 = (result.treatment, result.outcome, result.mu0, result.mu1)
    e = _clipped(result)
    expected = m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)
    np.testing.assert_allclose(result.scores, expected)
    assert result.estimate == pytest.approx(expected.mean())
    rows = result.to_frame(output="records")
    assert all(len(row["fold_assignments"]) == 3 for row in rows)


def test_repeated_cross_fitting_is_deterministic_for_its_seed():
    data = confounded(n=300)
    first = mb.aipw(data, spec(), folds=3, repeats=2, seed=8,
                    mstop=40, tune=False)
    second = mb.aipw(data, spec(), folds=3, repeats=2, seed=8,
                     mstop=40, tune=False)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    np.testing.assert_allclose(first.propensity, second.propensity)
    np.testing.assert_allclose(first.scores, second.scores)


def test_repeats_require_drawable_integer_partitions():
    data = confounded(n=100)
    scheme = mb.grouped_folds(np.repeat(np.arange(25), 4), B=5)
    with pytest.raises(ValueError, match="integer folds count"):
        mb.aipw(data, spec(), folds=scheme, repeats=2, tune=False)


def test_att_score_recovers_known_effect():
    result = mb.aipw(confounded(), spec("att"), folds=5, mstop=150, tune=False)
    assert result.estimate == pytest.approx(2., abs=.2)
    a, y, m0 = (result.treatment, result.outcome, result.mu0)
    e = _clipped(result)
    share = a.mean()
    numerator = a * (y - m0) - (1 - a) * e / (1 - e) * (y - m0)
    expected_influence = (numerator - result.estimate * a) / share
    np.testing.assert_allclose(result.influence, expected_influence)
    assert result.influence.mean() == pytest.approx(0, abs=1e-14)


def test_inner_tuning_is_recorded_and_stays_inside_outer_training():
    result = mb.aipw(confounded(n=240), spec(), folds=3, mstop=30, max_mstop=30,
                     tune=True, inner_folds=3)
    assert result.nuisance["tuned"] is True
    assert len(result.nuisance["selected_mstop"]) == 3
    assert all(all(1 <= stop <= 30 for stop in row.values())
               for row in result.nuisance["selected_mstop"])


def test_grouped_cross_fitting_keeps_groups_whole():
    data = confounded(n=400)
    groups = np.repeat(np.arange(100), 4)
    folds = mb.grouped_folds(groups, B=5, seed=2)
    result = mb.aipw(data, spec(), folds=folds, mstop=80, tune=False)
    for group in np.unique(groups):
        assert len(np.unique(result.fold_id[groups == group])) == 1
    assert result.standard_error > 0 and result.interval is not None
    assert result.diagnostics["variance"] == "cluster_robust_influence"
    assert result.diagnostics["independent_units"] == 100
    assert result.diagnostics["degrees_of_freedom"] == 99

    totals = np.array([
        result.influence[groups == group].sum() for group in np.unique(groups)
    ])
    expected_variance = 100 / 99 * np.sum(totals**2) / len(groups) ** 2
    assert result.standard_error == pytest.approx(np.sqrt(expected_variance))


def test_cluster_robust_uncertainty_uses_clusters_not_fold_labels():
    data = confounded(n=360)
    groups = np.repeat(np.arange(60), 6)
    first = mb.aipw(data, spec(), folds=mb.grouped_folds(groups, B=3, seed=7),
                    mstop=60, tune=False)
    second = mb.aipw(data, spec(), folds=mb.grouped_folds(groups, B=6, seed=7),
                     mstop=60, tune=False)
    assert first.diagnostics["independent_units"] == 60
    assert second.diagnostics["independent_units"] == 60
    assert first.diagnostics["degrees_of_freedom"] == 59
    assert second.diagnostics["degrees_of_freedom"] == 59


def test_diagnostics_report_overlap_and_effective_sample_size():
    result = mb.aipw(confounded(n=500), spec(), folds=5, mstop=100, tune=False)
    assert 0 < result.diagnostics["propensity_min"]
    assert result.diagnostics["propensity_max"] < 1
    assert 0 < result.diagnostics["ipw_effective_sample_size"] <= 500
    assert "conditional exchangeability" in result.describe()
    balance = result.balance_frame(output="records")
    assert [row["covariate"] for row in balance] == ["x"]
    assert abs(balance[0]["smd_after"]) < abs(balance[0]["smd_before"])
    overlap = result.overlap_frame(output="records")
    assert len(overlap) == 500
    assert {row["treatment_label"] for row in overlap} == {"control", "treated"}
    assert max(row["analysis_weight"] for row in overlap) == pytest.approx(
        result.diagnostics["ipw_max"]
    )


def test_att_overlap_weights_match_the_att_target_population():
    result = mb.aipw(confounded(n=500), spec("att"), folds=5,
                     mstop=100, tune=False)
    rows = result.overlap_frame(output="records")
    assert all(row["analysis_weight"] == 1 for row in rows
               if row["treatment"] == 1)
    expected = np.where(
        result.treatment == 1,
        1,
        _clipped(result) / (1 - _clipped(result)),
    )
    assert result.diagnostics["ipw_max"] == pytest.approx(expected.max())
    assert result.diagnostics["ipw_effective_sample_size"] == pytest.approx(
        expected.sum() ** 2 / np.sum(expected**2)
    )
    assert result.diagnostics["treated_weight_effective_sample_size"] == pytest.approx(
        result.treatment.sum()
    )
    controls = (1 - result.treatment) * expected
    assert result.diagnostics["control_weight_effective_sample_size"] == pytest.approx(
        controls.sum() ** 2 / np.sum(controls**2)
    )


def test_causal_overlap_and_balance_plots_expose_the_diagnostic_quantities():
    pytest.importorskip("altair")
    result = mb.aipw(confounded(n=300), spec(), folds=3, mstop=60, tune=False)
    overlap = mb.plot.plot_causal_overlap(result).to_dict()
    balance = mb.plot.plot_causal_balance(result).to_dict()
    assert len(overlap["hconcat"]) == 2
    assert overlap["hconcat"][0]["title"] == "Observed propensity overlap"
    assert balance["title"] == "Covariate balance"


@pytest.mark.parametrize("bad", [
    mb.CausalSpec("a", "y", ("x",), treatment_contrast=(0, 2),
                  time_zero="assignment", follow_up="one year",
                  adjustment_rationale="x is the common cause"),
    mb.CausalSpec("a", "y", ("x",), time_zero="assignment", follow_up="one year",
                  adjustment_rationale="x is the common cause"),
])
def test_invalid_treatment_data_are_refused(bad):
    data = confounded(n=100)
    if bad.treatment_contrast == (0, 1):
        data["a"][:] = 1
    with pytest.raises(ValueError, match="treatment|both treatment"):
        mb.aipw(data, bad, tune=False)


def test_repeated_or_incomplete_folds_are_refused():
    data = confounded(n=100)
    folds = mb.repeated_kfold_folds(100, k=5, repeats=2)
    with pytest.raises(ValueError, match="held out exactly once"):
        mb.aipw(data, spec(), folds=folds, tune=False)


def test_confounding_sensitivity_is_an_exact_bias_function():
    result = mb.aipw(confounded(n=500), spec(), folds=5, mstop=100, tune=False)
    rows = mb.confounding_sensitivity(
        result, outcome_effects=[0, 1, 2], imbalances=[0, .5], output="records"
    )
    unchanged = next(row for row in rows if row["outcome_effect"] == 0)
    assert unchanged["adjusted_estimate"] == result.estimate
    shifted = next(row for row in rows
                   if row["outcome_effect"] == 2 and row["imbalance"] == .5)
    assert shifted["assumed_bias"] == 1
    assert shifted["adjusted_estimate"] == pytest.approx(result.estimate - 1)
    assert shifted["lower"] == pytest.approx(result.interval[0] - 1)


def test_measured_confounder_benchmark_calibrates_the_shift_scale():
    rng = np.random.default_rng(27)
    n = 1000
    confounder = rng.normal(size=n)
    prognostic = rng.normal(size=n)
    irrelevant = rng.normal(size=n)
    e = 1 / (1 + np.exp(-1.5 * confounder))
    a = rng.binomial(1, e)
    y = 2 * a + 2 * confounder + .8 * prognostic + rng.normal(size=n)
    data = {"a": a, "y": y, "confounder": confounder,
            "prognostic": prognostic, "irrelevant": irrelevant}
    item = mb.CausalSpec(
        treatment="a", outcome="y",
        covariates=("confounder", "prognostic", "irrelevant"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="confounder and prognostic variables precede treatment",
    )
    result = mb.aipw(data, item, folds=4, mstop=100, tune=False)
    rows = mb.confounding_benchmark(
        data, result, variables=("confounder", "irrelevant"), output="records"
    )
    by_name = {row["omitted_covariate"]: row for row in rows}
    assert abs(by_name["confounder"]["estimate_shift"]) > .5
    assert abs(by_name["confounder"]["estimate_shift"]) > 5 * abs(
        by_name["irrelevant"]["estimate_shift"]
    )
    assert all(row["benchmark"] == "leave_one_measured_covariate_out" for row in rows)


def test_confounder_benchmark_requires_a_reconstructible_fold_design():
    data = confounded(n=200)
    data["z"] = np.random.default_rng(2).normal(size=200)
    item = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x", "z"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="x is a common cause and z is a baseline precision variable",
    )
    scheme = mb.grouped_folds(np.repeat(np.arange(50), 4), B=5)
    result = mb.aipw(data, item, folds=scheme, mstop=50, tune=False)
    with pytest.raises(ValueError, match="same scheme"):
        mb.confounding_benchmark(data, result, variables=("x",), output="records")
    rows = mb.confounding_benchmark(
        data, result, variables=("x",), folds=scheme, output="records"
    )
    assert len(rows) == 1


def test_confounder_benchmark_validates_its_covariate_set():
    data = confounded(n=200)
    result = mb.aipw(data, spec(), folds=4, mstop=50, tune=False)
    with pytest.raises(ValueError, match="at least two covariates"):
        mb.confounding_benchmark(data, result, output="records")
    with pytest.raises(ValueError, match="not in the adjustment"):
        mb.confounding_benchmark(data, result, variables=("missing",), output="records")


def test_negative_control_records_expected_null_without_claiming_validation():
    rng = np.random.default_rng(77)
    n = 1000
    x = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-x)))
    placebo = .8 * x + rng.normal(size=n)
    item = mb.CausalSpec(
        treatment="a", outcome="placebo", covariates=("x",),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="x is a common cause of treatment and the placebo outcome",
    )
    control = mb.negative_control(
        {"x": x, "a": a, "placebo": placebo}, item, kind="outcome",
        rationale="treatment occurs after the process generating the placebo outcome",
        folds=5, mstop=100, tune=False,
    )
    assert control.interval_excludes_expected is False
    assert "does not establish identification" in control.describe()
    assert control.effect.spec is item


def test_negative_control_can_flag_a_deliberately_unadjusted_bias_path():
    rng = np.random.default_rng(91)
    n = 1200
    common_cause = rng.normal(size=n)
    irrelevant = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-1.6 * common_cause)))
    placebo = 1.5 * common_cause + rng.normal(scale=.5, size=n)
    item = mb.CausalSpec(
        treatment="a", outcome="placebo", covariates=("irrelevant",),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="deliberately incomplete adjustment for the falsification test",
    )
    control = mb.negative_control(
        {"irrelevant": irrelevant, "a": a, "placebo": placebo}, item,
        kind="outcome",
        rationale="the placebo precedes treatment and shares its omitted common cause",
        folds=5, mstop=80, tune=False,
    )
    assert control.interval_excludes_expected is True
    assert "evidence against" in control.describe()


@pytest.mark.parametrize("arguments, message", [
    ({"kind": "placebo", "rationale": "known null"}, "kind"),
    ({"kind": "outcome", "rationale": ""}, "rationale"),
    ({"kind": "outcome", "rationale": "known null", "expected_effect": np.nan},
     "expected_effect"),
])
def test_negative_control_requires_an_explicit_valid_design(arguments, message):
    with pytest.raises((TypeError, ValueError), match=message):
        mb.negative_control(confounded(n=100), spec(), tune=False, **arguments)


def modified_spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2", "x3"),
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )


def modified(n=1500, seed=3, step=False, joint=False):
    """Confounded data whose effect depends on x1 only: 1 + 0.8 x1, a step
    0.5 + 1(x1 > 0), or 1 + 1.5 1(x1 x2 > 0), which turns on two modifiers
    at once and so leaves each modifier's own effect flat."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.uniform(-1.5, 1.5, (3, n))
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 + 0.6 * x2))))
    if joint:
        tau = 1 + 1.5 * (x1 * x2 > 0)
    else:
        tau = 0.5 + (x1 > 0) if step else 1 + 0.8 * x1
    y = np.sin(x1) + 0.7 * x2 + tau * a + rng.normal(0, 0.5, n)
    return {"a": a, "y": y, "x1": x1, "x2": x2, "x3": x3}, tau


def test_effect_model_recovers_a_modifier_and_beats_the_naive_contrast():
    data, tau = modified()
    fx = mb.effect_model(data, modified_spec(), seed=1)
    assert np.sqrt(np.mean((fx.tau - tau) ** 2)) < 0.25
    assert np.corrcoef(fx.tau, data["x1"])[0, 1] > 0.9
    flat = fx.predict({"x1": [0.0, 0.0], "x2": [-1.0, 1.0], "x3": [-1.0, 1.0]})
    assert abs(flat[1] - flat[0]) < 0.2
    assert fx.predict({"x1": [-1.0, 1.0], "x2": [0, 0], "x3": [0, 0]})[1] > 1.5


def test_effect_model_chooses_stumps_for_a_step_effect():
    data, tau = modified(step=True)
    fx = mb.effect_model(data, modified_spec(), seed=1)
    assert fx.terms in ("stumps", "joint")
    assert set(fx.cv_rloss) == {"smooth", "stumps", "joint"}
    assert np.sqrt(np.mean((fx.tau - tau) ** 2)) < 0.3


def test_effect_model_finds_an_effect_that_needs_two_modifiers():
    """1 + 1.5 1(x1 x2 > 0) has a flat marginal effect in each of x1 and x2,
    so no additive dictionary can represent it; the joint candidate does."""
    data, tau = modified(joint=True)
    additive = mb.effect_model(data, modified_spec(), terms="stumps", seed=1)
    fx = mb.effect_model(data, modified_spec(), seed=1)
    joint = mb.effect_model(data, modified_spec(), terms="joint", seed=1)
    additive_rmse = np.sqrt(np.mean((additive.tau - tau) ** 2))
    joint_rmse = np.sqrt(np.mean((joint.tau - tau) ** 2))
    assert joint.cv_rloss["joint"] < additive.cv_rloss["stumps"]
    assert joint_rmse < 0.7 * additive_rmse
    assert fx.terms == "joint"
    assert np.sqrt(np.mean((fx.tau - tau) ** 2)) < 0.6
    assert joint.diagnostics["joint_depth"] == 3


def test_effect_model_joint_needs_two_modifiers():
    data, _ = modified(n=400)
    with pytest.raises(ValueError, match="at least two modifiers"):
        mb.effect_model(data, modified_spec(), terms="joint", modifiers=("x1",))
    # one modifier leaves nothing for the joint tree to split on, so "auto"
    # does not compare it.
    fx = mb.effect_model(data, modified_spec(), modifiers=("x1",), seed=1)
    assert set(fx.cv_rloss) == {"smooth", "stumps"}


def test_effect_model_reuses_aipw_nuisances():
    data, tau = modified()
    res = mb.aipw(data, modified_spec(), smooth=True, seed=2)
    fx = mb.effect_model(data, modified_spec(), nuisance=res, terms="smooth", modifiers=("x1",))
    np.testing.assert_allclose(fx.propensity, res.propensity)
    np.testing.assert_allclose(fx.outcome_mean,
                               res.propensity * res.mu1 + (1 - res.propensity) * res.mu0)
    assert fx.modifiers == ("x1",) and fx.terms == "smooth"
    assert np.sqrt(np.mean((fx.tau - tau) ** 2)) < 0.3
    assert "supplied aipw result" in fx.describe()
    assert fx.to_dict()["ate_plugin"] == pytest.approx(fx.ate_plugin)


def test_effect_model_refuses_bad_arguments():
    data, _ = modified(n=200)
    with pytest.raises(ValueError, match="modifiers"):
        mb.effect_model(data, modified_spec(), modifiers=("z",))
    with pytest.raises(ValueError, match="terms"):
        mb.effect_model(data, modified_spec(), terms="trees")
    with pytest.raises(TypeError, match="nuisance"):
        mb.effect_model(data, modified_spec(), nuisance="aipw")


def test_rloss_transform_equals_the_r_loss():
    from mboostrs._rust import rloss, rloss_transform

    rng = np.random.default_rng(0)
    y, m, tau = rng.normal(size=(3, 50))
    t = rng.binomial(1, 0.5, 50).astype(float)
    e = rng.uniform(0.001, 0.999, 50)
    z, w, clipped = rloss_transform(y, t, m, e, (0.01, 0.99))
    assert clipped == int(np.sum((e < 0.01) | (e > 0.99)))
    assert np.mean(w * (z - tau) ** 2) == pytest.approx(rloss(y, t, m, e, tau, (0.01, 0.99)))


def test_ci_test_separates_a_proxy_from_its_driver():
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.uniform(-1, 1, n)
    p1 = x1 + rng.normal(0, .35, n)
    z = rng.uniform(-1, 1, n)
    y = np.sin(3 * x1) + z ** 2 + rng.normal(0, .5, n)
    data = {"x1": x1, "p1": p1, "z": z, "y": y}
    assert mb.ci_test(data, "p1", "y").p_value < 1e-6
    assert mb.ci_test(data, "p1", "y", given=("x1",)).p_value > 0.01
    assert mb.ci_test(data, "x1", "y", given=("p1",)).p_value < 1e-6
    # A non-monotone driver (y on z^2) is detected by the joint test of x
    # and its quartile hinges; its single covariance with y is zero.
    result = mb.ci_test(data, "z", "y", given=("x1",))
    assert result.functions == 4 and result.p_value < 1e-6
    assert "not proof of independence" in result.describe()


def test_ci_test_holds_its_level_under_the_null():
    rejections = 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        n = 400
        z = rng.uniform(-1, 1, n)
        x = np.sin(2 * z) + rng.normal(0, .5, n)
        y = z ** 2 + rng.normal(0, .5, n)
        rejections += mb.ci_test({"x": x, "y": y, "z": z}, "x", "y", given=("z",),
                                 seed=seed).p_value < 0.05
    # About 2 of 40 at the 5% level; 7 or more would be a clear excess.
    assert rejections <= 6
