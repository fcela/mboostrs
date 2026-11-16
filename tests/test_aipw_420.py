"""aipw for issue #420: sparse-arm check, pooled outcome model, step cap,
learner choice for low-cardinality and factor baselines, and the ATT with
missing outcomes.

The simulated designs are small so the file runs in seconds. Each test that
depends on a stochastic outcome uses fixed seeds.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import causal
from mboostrs.folds import FoldScheme, stratified_folds


def expit(x):
    return 1 / (1 + np.exp(-x))


def sparse_design(seed, n=1200):
    """No treatment effect. Treatment becomes rare where z is large, and the
    outcome has a convex tail in z. Treated rows are almost absent in the tail,
    so a treated-arm model fitted on the treated rows extrapolates there, while
    the control rows cover the tail."""
    rng = np.random.default_rng(seed)
    x1, z = rng.normal(size=(2, n))
    a = rng.binomial(1, expit(-0.2 + 0.8 * x1 - 3.0 * z))
    y = x1 + 0.5 * z + 4.0 * np.maximum(z - 1.0, 0) ** 2 + rng.normal(size=n)
    return {"a": a, "y": y, "x1": x1, "z": z}


def sparse_spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "z"), target_population="simulated",
        unit="person", time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and z cause both treatment and outcome",
    )


SPARSE_FAST = dict(smooth=True, folds=5, tune=False, mstop=200)


def att_design(seed, n=2000):
    """Missing at random given x and a. The effect is 1 + x1, so the ATT, the
    mean effect among treated rows, differs from the ATE."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, expit(0.8 * x1 - 0.5 * x2))
    y = x1 + 0.5 * x2 + a * (1 + x1) + rng.normal(size=n)
    observed = rng.binomial(1, expit(0.5 + 1.5 * x1 - 0.5 * a)).astype(bool)
    return {"a": a, "y": np.where(observed, y, np.nan), "x1": x1, "x2": x2}


def att_truth():
    """E[1 + x1 | A = 1] under the design above, over a large Monte Carlo
    population drawn from the same covariate and treatment model."""
    rng = np.random.default_rng(2026)
    x1, x2 = rng.normal(size=(2, 1_000_000))
    p = expit(0.8 * x1 - 0.5 * x2)
    return float(np.sum(p * (1 + x1)) / np.sum(p))


def att_spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"), estimand="att",
        target_population="simulated treated population", unit="person",
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )


def sparse_warnings(caught):
    return [w for w in caught if "propensity bin" in str(w.message)]


# Sparse arms ---------------------------------------------------------------

def test_separate_models_warn_and_name_the_sparse_bin():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.aipw(sparse_design(0), sparse_spec(), outcome_model="separate",
                         **SPARSE_FAST)
    messages = sparse_warnings(caught)
    assert len(messages) == 1
    text = str(messages[0].message)
    assert "propensity" in text and "z in [" in text and "outcome_model='pooled'" in text
    report = result.diagnostics["sparse_arms"]
    assert report["outcome_model"] == "separate" and report["warned"] is True
    assert len(report["bins"]) == 6  # n = 1,200: one bin per 200 rows
    assert sum(row["n_treated"] + row["n_control"] for row in report["bins"]) == 1200
    assert sum(row["n_treated"] for row in report["bins"]) == int(result.treatment.sum())
    flagged = report["bins"][report["flagged_bins"][0]]
    assert flagged["n_treated"] < 20 or flagged["n_control"] < 20
    assert flagged["share_of_target"] > 0.05


def test_pooled_outcome_model_covers_zero_in_most_seeds_where_separate_warns():
    covered, separate_errors, pooled_errors = [], [], []
    for seed in range(10):
        data = sparse_design(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            separate = mb.aipw(data, sparse_spec(), outcome_model="separate", **SPARSE_FAST)
        assert sparse_warnings(caught), f"seed {seed}: separate did not warn"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pooled = mb.aipw(data, sparse_spec(), outcome_model="pooled", **SPARSE_FAST)
        assert not sparse_warnings(caught), f"seed {seed}: pooled warned"
        lower, upper = pooled.interval
        covered.append(lower <= 0 <= upper)
        separate_errors.append(abs(separate.estimate))
        pooled_errors.append(abs(pooled.estimate))
    assert sum(covered) >= 8
    assert np.mean(pooled_errors) < np.mean(separate_errors)


def test_outcome_model_options_are_checked():
    data = sparse_design(0)
    with pytest.raises(ValueError, match="outcome_model must be"):
        mb.aipw(data, sparse_spec(), outcome_model="shared", **SPARSE_FAST)
    with pytest.raises(ValueError, match="applies to outcome_model='pooled'"):
        mb.aipw(data, sparse_spec(), outcome_model="separate", outcome_interactions=True,
                **SPARSE_FAST)


def test_pooled_interactions_put_the_treatment_into_each_baseline_term():
    data = sparse_design(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = mb.aipw(data, sparse_spec(), outcome_model="pooled", **SPARSE_FAST)
        interacted = mb.aipw(data, sparse_spec(), outcome_model="pooled",
                             outcome_interactions=True, **SPARSE_FAST)
    assert plain.nuisance["terms"]["outcome"] == "bbs(x1, df=4) + bbs(z, df=4) + a"
    assert interacted.nuisance["terms"]["outcome"] == (
        "bbs(x1, df=4) + bbs(z, df=4) + a + bbs(x1, df=4, by=a) + bbs(z, df=4, by=a)")
    assert "sparse_arms" in interacted.diagnostics


def test_pooled_model_without_interactions_records_the_flag_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.aipw(sparse_design(2), sparse_spec(), outcome_model="pooled",
                         **SPARSE_FAST)
    assert not sparse_warnings(caught)
    report = result.diagnostics["sparse_arms"]
    assert report["warned"] is False and report["flagged_bins"]


def balanced_fit_design(seed, n=1500):
    """Both arms cover the whole covariate range. The true propensity runs from
    about 0.2 to 0.8, so no bin has a thin arm."""
    rng = np.random.default_rng(seed)
    x1, z = rng.normal(size=(2, n))
    a = rng.binomial(1, expit(0.3 * x1 - 0.3 * z))
    y = x1 + 0.5 * z + a * (1 + 0.5 * x1) + rng.normal(size=n)
    return {"a": a, "y": y, "x1": x1, "z": z}


def test_bin_count_is_one_per_200_rows_between_4_and_10():
    assert causal._sparse_bin_count(500) == 4
    assert causal._sparse_bin_count(1500) == 7
    assert causal._sparse_bin_count(2000) == 10
    assert causal._sparse_bin_count(50_000) == 10


def test_balanced_design_at_n_1500_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.aipw(balanced_fit_design(0), sparse_spec(), outcome_model="separate",
                         **SPARSE_FAST)
    assert 0.1 < float(result.propensity.min()) and float(result.propensity.max()) < 0.9
    assert not sparse_warnings(caught)
    report = result.diagnostics["sparse_arms"]
    assert len(report["bins"]) == 7
    assert report["flagged_bins"] == [] and report["warned"] is False


def test_arm_absent_from_high_propensity_region_still_warns_at_n_1500():
    data = sparse_design(0, n=1500)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.aipw(data, sparse_spec(), outcome_model="separate", **SPARSE_FAST)
    assert len(sparse_warnings(caught)) == 1
    report = result.diagnostics["sparse_arms"]
    assert len(report["bins"]) == 7 and report["flagged_bins"]


def test_weighted_fit_and_replicated_rows_give_the_same_sparse_arm_flags():
    data = sparse_design(0, n=600)
    w = np.random.default_rng(5).integers(1, 4, size=600)
    replicated = {name: np.repeat(values, w) for name, values in data.items()}
    base = stratified_folds(data["a"], B=3, seed=2)
    scheme_rep = FoldScheme(
        weights=np.repeat(base.weights, w, axis=0),
        held_out=np.repeat(base.held_out, w, axis=0),
        scheme=base.scheme, seed=base.seed,
    )
    kwargs = dict(outcome_model="separate", smooth=True, tune=False, mstop=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = mb.aipw(data, sparse_spec(), folds=base, weights=w, **kwargs)
        expanded = mb.aipw(replicated, sparse_spec(), folds=scheme_rep, **kwargs)
    report, expanded_report = (weighted.diagnostics["sparse_arms"],
                               expanded.diagnostics["sparse_arms"])
    assert report["flagged_bins"] == expanded_report["flagged_bins"]
    assert len(report["bins"]) == len(expanded_report["bins"])
    for row, other in zip(report["bins"], expanded_report["bins"], strict=True):
        assert row["sparse_arms"] == other["sparse_arms"]
        assert row["n_treated"] == pytest.approx(other["n_treated"])
        assert row["n_control"] == pytest.approx(other["n_control"])
        assert row["propensity_min"] == pytest.approx(other["propensity_min"])
        assert row["propensity_max"] == pytest.approx(other["propensity_max"])


# Step cap ------------------------------------------------------------------

def capped_design(n=240):
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    a = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(float)
    y = 2 * a + 2 * x + rng.normal(size=n)
    return {"x": x, "a": a, "y": y}


def one_covariate_spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",), target_population="study sample",
        unit="person", time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x causes both treatment and outcome",
    )


def test_step_search_doubles_from_mstop_and_warns_naming_max_mstop():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.aipw(capped_design(), one_covariate_spec(), folds=3, mstop=10,
                         max_mstop=30, tune=True, inner_folds=3)
    capped = [w for w in caught if "max_mstop=30" in str(w.message)]
    assert capped, "expected a warning that names aipw's max_mstop argument"
    assert "aipw(max_mstop=" in str(capped[0].message)
    record = result.diagnostics["nuisance_mstop"]
    assert any(row["at_max_mstop"] for row in record.values())
    for row in record.values():
        assert row["first_grid"] == 10 and row["max_mstop"] == 30
        assert all(1 <= stop <= 30 for stop in row["mstop_by_fold"])


def test_untuned_nuisances_run_exactly_mstop_steps():
    result = mb.aipw(capped_design(), one_covariate_spec(), folds=3, mstop=25,
                     tune=False)
    record = result.diagnostics["nuisance_mstop"]
    for row in record.values():
        assert row["mstop_by_fold"] == [25, 25, 25]
        assert row["at_max_mstop"] is False and row["max_mstop"] is None


def test_max_mstop_below_mstop_is_refused():
    with pytest.raises(ValueError, match="max_mstop=20 is below mstop=50"):
        mb.aipw(capped_design(), one_covariate_spec(), mstop=50, max_mstop=20)


# Learner choice ------------------------------------------------------------

def test_smooth_uses_linear_terms_below_ten_distinct_values():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    d = rng.integers(0, 2, n).astype(float)
    count = rng.integers(0, 4, n).astype(float)
    a = (rng.random(n) < expit(x + d - 0.5 * count)).astype(float)
    y = a + x + d + count + rng.normal(size=n)
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x", "d", "count"),
        target_population="sample", unit="person", time_zero="assignment",
        follow_up="one year", adjustment_rationale="all three cause treatment and outcome")
    # Separate models, so the outcome terms are the per-arm terms; under "auto"
    # this n = 400 design is flagged and the pooled model would be fitted instead.
    result = mb.aipw({"a": a, "y": y, "x": x, "d": d, "count": count}, spec,
                     outcome_model="separate", smooth=True, folds=3, tune=False, mstop=50)
    assert result.nuisance["terms"]["propensity"] == (
        "bbs(x, df=4) + bols(d) + bols(count)")
    assert result.nuisance["terms"]["outcome"] == result.nuisance["terms"]["propensity"]


def test_string_and_categorical_baselines_are_accepted_and_balanced():
    rng = np.random.default_rng(7)
    n = 1500
    site = rng.choice(np.array(["north", "south", "east"], dtype=object), n)
    shift = np.where(site == "north", 1.0, np.where(site == "south", -0.5, 0.0))
    x = rng.normal(size=n) + shift
    a = (rng.random(n) < expit(0.8 * x - 0.5 * (site == "east"))).astype(float)
    y = 2.0 * a + x + np.where(site == "south", 1.5, 0.0) + rng.normal(size=n)
    data = {"a": a, "y": y, "x": x, "site": site}
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("site", "x"), target_population="sample",
        unit="person", time_zero="assignment", follow_up="one year",
        adjustment_rationale="site and x cause treatment and outcome")
    result = mb.aipw(data, spec, smooth=True, folds=5, tune=False, mstop=200)
    lower, upper = result.interval
    assert lower <= 2.0 <= upper
    assert abs(result.estimate - 2.0) < 0.3
    assert result.nuisance["terms"]["propensity"] == "bols(site) + bbs(x, df=4)"
    labels = [row["covariate"] for row in result.diagnostics["balance"]]
    assert {"site=east", "site=north", "site=south", "x"} <= set(labels)


def test_categorical_baseline_with_missing_values_is_refused():
    data = {"a": np.array([0., 1., 0., 1.] * 10), "y": np.arange(40, dtype=float),
            "site": np.array(["a", None, "b", "a"] * 10, dtype=object)}
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("site",), target_population="sample",
        unit="person", time_zero="assignment", follow_up="one year",
        adjustment_rationale="site causes treatment and outcome")
    with pytest.raises(ValueError, match="categorical covariate 'site' has"):
        mb.aipw(data, spec, tune=False, mstop=20)


# ATT with missing outcomes -------------------------------------------------

def test_att_with_missing_outcomes_recovers_the_known_att():
    truth = att_truth()
    covered, estimates, complete_case = [], [], []
    for seed in range(5):
        data = att_design(seed)
        result = mb.aipw(data, att_spec(), outcome_missing="ipw", folds=5, tune=False,
                         mstop=200)
        lower, upper = result.interval
        covered.append(lower <= truth <= upper)
        estimates.append(result.estimate)
        observed = np.isfinite(data["y"])
        subset = {name: values[observed] for name, values in data.items()}
        complete_case.append(mb.aipw(subset, att_spec(), folds=5, tune=False,
                                     mstop=200).estimate)
    assert sum(covered) >= 4
    assert abs(np.mean(estimates) - truth) < 0.1
    # Dropping the rows with a missing outcome changes the population, and here
    # it moves the ATT by more than the response weights do.
    assert np.mean(complete_case) - truth > 0.25


def test_att_with_missing_outcomes_targets_all_treated_rows():
    data = att_design(0)
    result = mb.aipw(data, att_spec(), outcome_missing="ipw", folds=5, tune=False,
                     mstop=200)
    treated = result.treatment == 1
    assert int(treated.sum()) > int(np.isfinite(result.outcome[treated]).sum())
    assert result.diagnostics["n_missing_outcome"] > 0
    assert result.diagnostics["response_model"] == "fitted"
    # The estimate is the mean of the row scores, each divided by the treated share.
    assert result.estimate == pytest.approx(float(np.mean(result.scores)), rel=1e-12)


def test_att_complete_case_score_is_unchanged_by_the_response_form():
    data = att_design(3)
    complete = np.isfinite(data["y"])
    subset = {name: values[complete] for name, values in data.items()}
    # Linear nuisances keep the propensity inside the clip, so the unclipped
    # propensity stored on the result reproduces the score.
    result = mb.aipw(subset, att_spec(), folds=5, tune=False, mstop=200, smooth=False)
    numerator = (result.treatment * (result.outcome - result.mu0)
                 - (1 - result.treatment) * result.propensity
                 / (1 - result.propensity) * (result.outcome - result.mu0))
    share = result.treatment.mean()
    assert np.allclose(result.scores, numerator / share, rtol=1e-10, atol=1e-12)

