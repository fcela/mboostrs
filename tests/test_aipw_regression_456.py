"""Issue #456: the aipw default must not produce an absurd ATE, and the two-part
outcome model is bounded; effect_model's terms="auto" does not return a candidate
that is no better than a constant.

The design in ``_failing_design`` has the features of the job-training draws on
which the default gave an ATE error above 100,000 dollars: a heavy-tailed positive
confounder in dollars, a thin treated tail that the sparse-arm check flags, and an
outcome that is zero for part of the sample. Its ATE is known from the conditional
means, so the tests can grade the estimate.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from mboostrs import causal
from mboostrs.causal import CausalSpec, aipw


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


#: The outcome's log-scale standard deviation in the failing design.
_SIGMA_LOG = 0.4


def _failing_design(n=3000, seed=11):
    """A frame with columns treated, earnings, prior and x2, and the sample-average
    treatment effect of its conditional means."""
    rng = np.random.default_rng(seed)
    x2 = rng.normal(size=n)
    log_prior = rng.normal(10.0, 1.1, n)
    prior = np.exp(log_prior)
    z = log_prior - 10.0
    tail = prior > np.quantile(prior, 0.97)
    treated = (rng.random(n) < _sigmoid(-3.2 + 0.9 * z + 0.3 * x2 - 3.0 * tail)).astype(float)
    employed = rng.random(n) < _sigmoid(-0.2 + 0.5 * z + 0.3 * treated)
    level = 9.5 + 0.7 * z + 0.2 * x2 + 0.25 * treated
    earnings = np.where(employed, np.exp(level + _SIGMA_LOG * rng.normal(size=n)), 0.0)
    frame = pl.DataFrame({"treated": treated, "earnings": earnings, "prior": prior, "x2": x2})

    def conditional_mean(a):
        employment = _sigmoid(-0.2 + 0.5 * z + 0.3 * a)
        mean_log = 9.5 + 0.7 * z + 0.2 * x2 + 0.25 * a
        return employment * np.exp(mean_log + _SIGMA_LOG ** 2 / 2)

    truth = float(np.mean(conditional_mean(np.ones(n)) - conditional_mean(np.zeros(n))))
    return frame, truth


_SPEC = CausalSpec(
    treatment="treated", outcome="earnings", covariates=("prior", "x2"),
    estimand="ate", outcome_type="continuous", time_zero="assignment",
    follow_up="one year after assignment",
    adjustment_rationale="prior earnings and x2 drive both assignment and earnings",
    outcome_missing="error")


@pytest.fixture(scope="module")
def failing_fit():
    """The default fit and the two-part fit on the failing design, computed once."""
    frame, truth = _failing_design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The defaults are smooth=False and outcome_model="cv". The "auto" fit
        # is the rule the issue measured; spline terms (smooth=True) on the
        # heavy-tailed dollar confounder extrapolate.
        default = aipw(frame, _SPEC, folds=5, seed=7, level=0.9)
        auto = aipw(frame, _SPEC, folds=5, seed=7, level=0.9, outcome_model="auto")
        two_part = aipw(frame, _SPEC, folds=5, seed=7, level=0.9,
                        outcome_model="two_part")
    return {"truth": truth, "default": default, "auto": auto, "two_part": two_part}


def test_the_failing_design_flags_a_thin_bin_and_has_zeros(failing_fit):
    # The fixture must reproduce the sparse-arm condition that selected the
    # two-part model in the failing draws.
    default = failing_fit["default"]
    assert default.diagnostics["sparse_arms"]["flagged_bins"]
    assert np.any(np.asarray(default.outcome) == 0)


def test_default_ate_on_the_failing_design_is_finite_and_within_half_the_truth(failing_fit):
    result = failing_fit["default"]
    truth = failing_fit["truth"]
    assert np.isfinite(result.estimate)
    assert abs(result.estimate - truth) < 0.5 * abs(truth)


def test_auto_does_not_select_the_two_part_model_on_a_flagged_zero_inflated_outcome(failing_fit):
    assert failing_fit["auto"].diagnostics["outcome_model"] == "pooled"
    assert failing_fit["auto"].diagnostics["outcome_model_choice"]["outcome_interactions"]


def test_auto_ate_on_the_failing_design_is_within_half_the_truth(failing_fit):
    assert abs(failing_fit["auto"].estimate - failing_fit["truth"]) < 0.5 * abs(failing_fit["truth"])


def test_default_cv_choice_on_the_failing_design_includes_the_two_part_candidate(failing_fit):
    cv = failing_fit["default"].diagnostics["outcome_model_cv"]
    assert "two_part" in cv["candidates"]
    assert cv["chosen"] == "two_part"
    assert failing_fit["default"].diagnostics["outcome_model"] == "two_part"


def test_two_part_ate_on_the_failing_design_is_finite_and_within_half_the_truth(failing_fit):
    result = failing_fit["two_part"]
    truth = failing_fit["truth"]
    assert result.diagnostics["outcome_model"] == "two_part"
    assert np.isfinite(result.estimate)
    assert abs(result.estimate - truth) < 0.5 * abs(truth)


def test_two_part_means_stay_within_the_outcome_range(failing_fit):
    record = failing_fit["two_part"].diagnostics["outcome_range"]
    assert record["rows_outside"] == {"mu0": 0, "mu1": 0}
    assert np.max(np.abs(failing_fit["two_part"].mu1)) <= record["upper_bound"]
    assert np.max(np.abs(failing_fit["two_part"].mu0)) <= record["upper_bound"]


def test_two_part_log_model_enters_the_skewed_baseline_on_the_log_scale(failing_fit):
    terms = failing_fit["two_part"].nuisance["terms"]["outcome"]
    employment, log_part = terms.split("; log outcome: ")
    assert "_log_prior" in log_part
    assert "_log_x2" not in log_part and "bols(prior)" not in log_part
    assert "bols(prior)" in employment


def test_log_baselines_selects_only_the_strictly_positive_skewed_column():
    rng = np.random.default_rng(0)
    values = {
        "prior": np.exp(rng.normal(10.0, 1.1, 2000)),
        "x2": rng.normal(size=2000),
        "education": rng.integers(0, 18, 2000).astype(float),
    }
    assert causal._log_baselines(values, ("prior", "x2", "education")) == frozenset({"prior"})


def test_outcome_range_check_warns_only_for_means_beyond_the_bound():
    y = np.array([0.0, 100.0, 200.0, 0.0])
    observed = np.ones(4)
    inside = np.array([10.0, 150.0, 190.0, 5.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = causal._outcome_range_check(y, observed, inside, inside, "separate")
    assert record["rows_outside"] == {"mu0": 0, "mu1": 0}
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    beyond = np.array([10.0, 150.0, 900.0, 5.0])
    with pytest.warns(RuntimeWarning, match="lie outside"):
        record = causal._outcome_range_check(y, observed, inside, beyond, "two_part")
    assert record["rows_outside"] == {"mu0": 0, "mu1": 1}
    # The observed range is [0, 200]; one width beyond its upper end is 400.
    assert record["upper_bound"] == pytest.approx(400.0)


def test_auto_terms_refuses_a_boosted_candidate_that_stops_at_zero_steps():
    # stumps has the lower loss, but its tuned step count is 1: a near-constant fit.
    rng = np.random.default_rng(1)
    n = 400
    base = rng.normal(size=n)
    constant = 1.0 + base
    loss_rows = {
        "smooth": constant - 0.20 + 0.05 * rng.normal(size=n),
        "stumps": constant - 0.30 + 0.05 * rng.normal(size=n),
    }
    cv_rloss = {k: float(np.mean(v)) for k, v in loss_rows.items()}
    chosen, record = causal._auto_terms(cv_rloss, loss_rows, constant, {"stumps": 1})
    assert chosen == "smooth"
    assert record["flat"]["stumps"] is True and record["eligible"]["stumps"] is False
    assert record["lowest_loss"] == "stumps"


def test_auto_terms_prefers_smooth_when_the_difference_is_within_one_standard_error():
    rng = np.random.default_rng(2)
    n = 400
    constant = np.zeros(n)
    # Independent noise in each candidate, so the standard error of their
    # difference is positive (about 0.035) and larger than the 0.01 gap.
    loss_rows = {
        "smooth": constant - 0.2 + 0.5 * rng.normal(size=n),
        "stumps": constant - 0.21 + 0.5 * rng.normal(size=n),
    }
    cv_rloss = {k: float(np.mean(v)) for k, v in loss_rows.items()}
    chosen, record = causal._auto_terms(cv_rloss, loss_rows, constant, {"stumps": 40})
    assert chosen == "smooth"
    assert record["tie_with_smooth"] is True
    assert "one standard error" in record["rule"]


def test_auto_terms_keeps_a_candidate_that_beats_smooth_by_more_than_one_standard_error():
    rng = np.random.default_rng(3)
    n = 400
    constant = np.zeros(n)
    noise = 0.3 * rng.normal(size=n)
    loss_rows = {
        "smooth": constant - 0.2 + noise,
        "stumps": constant - 0.9 + noise,
    }
    cv_rloss = {k: float(np.mean(v)) for k, v in loss_rows.items()}
    chosen, record = causal._auto_terms(cv_rloss, loss_rows, constant, {"stumps": 40})
    assert chosen == "stumps"
    assert record["tie_with_smooth"] is False


def test_auto_terms_returns_smooth_when_no_candidate_beats_the_constant():
    rng = np.random.default_rng(4)
    n = 400
    constant = np.zeros(n)
    loss_rows = {
        "smooth": constant + 0.1 + 0.5 * rng.normal(size=n),
        "stumps": constant + 0.2 + 0.5 * rng.normal(size=n),
    }
    cv_rloss = {k: float(np.mean(v)) for k, v in loss_rows.items()}
    chosen, record = causal._auto_terms(cv_rloss, loss_rows, constant, {"stumps": 40})
    assert chosen == "smooth"
    assert not any(record["eligible"].values())
    assert "no candidate beats the constant effect" in record["rule"]


def _heterogeneous_effect_data(n=800, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * x + 0.5 * z)))).astype(float)
    y = 1.0 + 0.5 * t + t * 1.5 * (x > 0) + rng.normal(0, 1, n)
    spec = CausalSpec(
        treatment="t", outcome="y", covariates=("x", "z"),
        time_zero="baseline", follow_up="end of study",
        adjustment_rationale="x and z are baseline covariates that precede treatment")
    return {"t": t, "y": y, "x": x, "z": z}, spec


def test_effect_model_auto_records_its_rule_and_the_smaller_step_cap():
    data, spec = _heterogeneous_effect_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = causal.effect_model(data, spec, terms="auto")
    assert fx.diagnostics["max_mstop"] == 1600
    selection = fx.diagnostics["terms_selection"]
    assert selection["rule"]
    assert set(selection["gain_over_constant"]) == set(fx.cv_rloss)
    assert fx.terms in selection["gain_over_constant"]
    assert "terms='auto' rule:" in fx.describe()
    for stops in fx.diagnostics["nuisance_mstop"]:
        assert stops["propensity"] <= 1600


def test_effect_model_with_named_terms_records_that_it_was_requested():
    data, spec = _heterogeneous_effect_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = causal.effect_model(data, spec, terms="smooth")
    selection = fx.diagnostics["terms_selection"]
    assert selection["rule"] == "requested"
    assert selection["lowest_loss"] == "smooth"
    # The gains of every candidate are reported for a named choice as well.
    assert set(selection["gain_over_constant"]) == set(fx.cv_rloss)
    assert set(selection["gain_se"]) == set(fx.cv_rloss)
    assert "smooth" in selection["gain_over_constant"]
    assert "terms='auto' rule:" not in fx.describe()
