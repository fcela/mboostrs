"""Tools from issue #429, items 5 to 10.

Covers ci_test with categorical columns, the cv_risk result and grid, the
GAMBoost fold count, the tune_mstop fold scores, rank_by_effect ties,
the argument order of confounding_benchmark, the agent-guide text, and
iv_effect against a known complier effect and against linear 2SLS.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.explore import cv_risk, tune_mstop
from mboostrs.families import Binomial, Gaussian

ROOT = Path(__file__).resolve().parents[1]


def _spec(treatment="d", outcome="y", covariates=("x",)):
    return mb.CausalSpec(
        treatment=treatment, outcome=outcome, covariates=covariates,
        time_zero="assignment", follow_up="end of study",
        adjustment_rationale="baseline covariates precede the instrument",
    )


# Item 5: ci_test with a categorical x.


def _factor_data(seed, dependent):
    rng = np.random.default_rng(seed)
    n = 500
    z = rng.normal(size=n)
    g = rng.integers(0, 3, size=n)
    x = np.array(["a", "b", "c"])[g]
    shift = (1.0 * (g == 1) - 0.8 * (g == 2)) if dependent else 0.0
    y = shift + 0.5 * z + rng.normal(size=n)
    return {"x": x, "y": y, "z": z}


def test_ci_test_tests_a_three_level_factor_jointly_with_one_df_per_indicator():
    dependent = mb.ci_test(_factor_data(3, True), "x", "y", given=["z"], folds=4)
    assert dependent.functions == 2
    assert dependent.p_value < 1e-6
    independent = mb.ci_test(_factor_data(4, False), "x", "y", given=["z"], folds=4)
    assert independent.functions == 2
    assert independent.p_value > 0.01


def test_ci_test_accepts_a_two_level_categorical_y_and_refuses_three_levels():
    data = _factor_data(5, True)
    binary_y = np.where(data["x"] == "a", "yes", "no")
    test = mb.ci_test({"x": data["y"], "y": binary_y, "z": data["z"]}, "x", "y",
                      given=["z"], folds=3)
    assert 0 <= test.p_value <= 1
    assert test.functions == 4  # a continuous x with many values: x and three hinges
    with pytest.raises(TypeError, match="one level at a time"):
        mb.ci_test(data, "y", "x", given=["z"], folds=3)


# Item 6: cv_risk result and grid, GAMBoost fold count, tune_mstop fold scores.


def _gauss_data(seed=5, n=240):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.normal(size=n)
    mu = np.sin(x) + 0.5 * z
    return {"y": mu + rng.normal(scale=0.5, size=n),
            "yb": (rng.uniform(size=n) < 1 / (1 + np.exp(-mu))).astype(float),
            "x": x, "z": z}


def test_cv_risk_grid_matches_the_float_at_each_step_count():
    data = _gauss_data()
    grid = [5, 20, 40]
    frame = cv_risk("bbs(x) + z", data, "y", family=Gaussian(), folds=4, seed=0,
                    mstop_grid=grid)
    records = frame.to_dicts() if hasattr(frame, "to_dicts") else frame.to_dict("records")
    assert [row["mstop"] for row in records] == grid
    for row, m in zip(records, grid, strict=True):
        result = cv_risk("bbs(x) + z", data, "y", family=Gaussian(), folds=4, seed=0,
                         mstop=m, return_result=True)
        assert row["risk"] == pytest.approx(float(result.score), rel=1e-9, abs=1e-12)
        assert row["fold_se"] > 0
        # The per-fold scores ride along, so a caller can pair two
        # specifications' grid rows on the shared folds (#500).
        assert row["fold_scores"] == pytest.approx(
            list(np.asarray(result.fold_scores, dtype=float)), rel=1e-9, abs=1e-12)


def test_cv_risk_grid_refuses_a_result_object_and_scores_response_losses():
    data = _gauss_data()
    with pytest.raises(ValueError, match="one risk per grid point"):
        cv_risk("bbs(x) + z", data, "y", family=Gaussian(), folds=4, mstop_grid=[5],
                return_result=True)
    # The grid scores the response losses along the path too, with the same
    # recoding and weighting a single-mstop call uses, so a response-score
    # grid row equals the float the same call returns at that step count.
    # The brier pass uses signed {-1, +1} labels, which both routes must
    # recode to {0, 1} the same way before the loss reads them.
    grid = [5, 20, 40]
    signed = dict(data)
    signed["yb"] = [2 * v - 1 for v in data["yb"]]
    for response, family, score, frame_data in (
            ("y", Gaussian(), "mse", data),
            ("yb", Binomial(), "brier", signed)):
        frame = cv_risk("bbs(x) + z", frame_data, response, family=family,
                        folds=4, seed=0, mstop_grid=grid, score=score)
        records = frame.to_dicts() if hasattr(frame, "to_dicts") else frame.to_dict("records")
        for row, m in zip(records, grid, strict=True):
            float_risk = cv_risk("bbs(x) + z", frame_data, response, family=family,
                                 folds=4, seed=0, mstop=m, score=score)
            assert row["risk"] == pytest.approx(float_risk, rel=1e-9, abs=1e-12)


def test_evaluation_result_aggregate_is_the_float_the_call_returns():
    data = _gauss_data()
    result = cv_risk("bbs(x) + z", data, "y", family=Gaussian(), folds=4, seed=0,
                     mstop=20, return_result=True)
    value = cv_risk("bbs(x) + z", data, "y", family=Gaussian(), folds=4, seed=0, mstop=20)
    assert result.aggregate == pytest.approx(float(value), rel=0, abs=1e-12)
    assert result.aggregate == result.score == float(result)


def test_gamboost_cvrisk_accepts_an_integer_fold_count():
    data = _gauss_data()
    model = mb.gamboost("y ~ bbs(x) + z", data=data, family=Gaussian(),
                        control=mb.boost_control(30, 0.1))
    result = model.cvrisk(folds=4, seed=1)
    assert result.fold_type == "4-fold kfold"
    assert result.risk_matrix.shape == (4, 31)
    assert result.optimal_mstop >= 1


def test_tune_mstop_reports_one_fold_score_per_fold_whose_mean_is_the_curve():
    data = _gauss_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuning = tune_mstop("yb ~ bbs(x) + z", data, family=Binomial(), folds=4,
                            start=40, seed=2, max_mstop=160)
    assert len(tuning.fold_scores) == 4
    assert np.mean(tuning.fold_scores) == pytest.approx(tuning.risk[tuning.mstop], abs=1e-10)


# Item 8: rank_by_effect ties at the selection boundary.


def _ranking(**columns):
    frame = pl.DataFrame(columns)
    return mb.rank_by_effect(frame, {}, share=0.5, propensity=[0.5] * frame.height)


def test_rank_by_effect_reports_ties_and_breaks_them_by_narrower_interval():
    # Rows 0, 1 and 2 share the estimate 0.5, and the cut (3 of 6 rows) splits them.
    ranking = _ranking(estimate=[0.5, 0.5, 0.5, 0.3, 0.9, 0.1],
                       se=[0.2, 0.1, 0.3, 0.1, 0.1, 0.1])
    assert ranking.chosen == (4, 1, 0)
    assert ranking.tied_rows == 3
    assert "standard error" in ranking.tie_break
    assert "3 row(s) tie at the selection boundary" in ranking.describe()
    tied = [row["tied_at_boundary"] for row in ranking.rows]
    assert tied == [True, True, True, False, False, False]


def test_rank_by_effect_breaks_ties_by_row_order_without_an_interval():
    ranking = _ranking(estimate=[0.5, 0.5, 0.5, 0.3, 0.9, 0.1])
    assert ranking.chosen == (4, 0, 1)
    assert ranking.tied_rows == 3
    assert "row order" in ranking.tie_break


def test_rank_by_effect_says_when_nothing_ties():
    ranking = _ranking(estimate=[0.4, 0.3, 0.2, 0.1], se=[0.1, 0.1, 0.1, 0.1])
    assert ranking.tied_rows == 0
    assert "No rows tie at the selection boundary" in ranking.describe()


# Item 9: argument order of confounding_benchmark.


def _aipw_fixture():
    rng = np.random.default_rng(9)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.7 * x1))).astype(float)
    y = 1.0 * a + x1 + 0.5 * x2 + rng.normal(size=n)
    data = {"a": a, "y": y, "x1": x1, "x2": x2}
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="assignment", follow_up="end of study",
        adjustment_rationale="x1 and x2 are measured before treatment",
    )
    result = mb.aipw(data, spec, folds=3, mstop=60, tune=False, seed=1)
    return data, result


def test_confounding_benchmark_accepts_either_argument_order():
    data, result = _aipw_fixture()
    canonical = mb.confounding_benchmark(data, result, output="records")
    reversed_order = mb.confounding_benchmark(result, data, output="records")
    assert canonical == reversed_order
    assert [row["omitted_covariate"] for row in canonical] == ["x1", "x2"]


def test_confounding_benchmark_refuses_two_results_or_no_result():
    data, result = _aipw_fixture()
    with pytest.raises(TypeError, match="confounding_benchmark"):
        mb.confounding_benchmark(data, data)
    with pytest.raises(TypeError, match="confounding_benchmark"):
        mb.confounding_benchmark(data, result=None)


def _custom_fixture(**terms):
    data, _ = _aipw_fixture()
    spec = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="assignment", follow_up="end of study",
        adjustment_rationale="x1 and x2 are measured before treatment",
    )
    result = mb.aipw(data, spec, folds=3, mstop=60, tune=False, seed=1, **terms)
    return data, result


def test_confounding_benchmark_refits_custom_terms_minus_the_omitted_covariate():
    data, result = _custom_fixture(propensity_terms="bbs(x1, df=4) + x2",
                                   outcome_terms="x1 + x2")
    assert result.nuisance["custom_terms"] == {"propensity": True, "outcome": True}
    rows = {row["omitted_covariate"]: row for row in
            mb.confounding_benchmark(data, result, output="records")}
    # Omitting x1 keeps the x2 term of the propensity formula and the pooled outcome.
    assert rows["x1"]["propensity_terms"] == "x2"
    assert rows["x1"]["outcome_terms"] == "x2"
    # Omitting x2 keeps the spline of x1 and drops the outcome term of x2.
    assert rows["x2"]["propensity_terms"] == "bbs(x1, df=4)"
    assert rows["x2"]["outcome_terms"] == "x1"


def test_confounding_benchmark_reports_default_formulas_as_none():
    data, result = _custom_fixture(outcome_terms="x1 + x2")
    rows = mb.confounding_benchmark(data, result, output="records")
    assert all(row["propensity_terms"] is None for row in rows)
    assert all(row["outcome_terms"] is not None for row in rows)


def test_confounding_benchmark_refuses_a_custom_formula_left_with_no_terms():
    data, result = _custom_fixture(propensity_terms="x1")
    with pytest.raises(ValueError, match="custom propensity_terms='x1'") as caught:
        mb.confounding_benchmark(data, result, variables=["x1"])
    assert "'x1'" in str(caught.value)
    # Omitting the other covariate leaves the custom term, so that call succeeds.
    rows = mb.confounding_benchmark(data, result, variables=["x2"], output="records")
    assert rows[0]["propensity_terms"] == "x1"


# Item 10: iv_effect.


def _late_data(seed=7, n=4000, effect=2.0):
    """Binary instrument, monotone compliance and a confounder shared by D and Y.

    With one effect for every unit, the complier effect equals ``effect``.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.binomial(1, 0.5, size=n).astype(float)
    u = rng.uniform(size=n)
    d = (u < 0.3 + 0.4 * z + 0.1 * x1).astype(float)
    y = effect * d + x1 + 0.5 * x2 + 3.0 * u + rng.normal(size=n)
    return {"y": y, "d": d, "z": z, "x1": x1, "x2": x2}


def test_iv_effect_recovers_a_known_complier_effect():
    data = _late_data()
    iv = mb.iv_effect(data, _spec(covariates=("x1", "x2")), "z", folds=5, seed=0,
                      tune=False, mstop=200)
    assert abs(iv.estimate - 2.0) < 0.25
    assert iv.standard_error > 0
    lower, upper = iv.interval
    assert lower < iv.estimate < upper
    assert iv.first_stage_f > 100
    assert iv.weak_instrument is False
    assert iv.first_stage_r2 > 0.1


def test_iv_effect_matches_linear_2sls_on_a_linear_design():
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    p = 0.4 + 0.1 * x + 0.3 * z + 0.2 * u
    d = (rng.uniform(size=n) < np.clip(p, 0.02, 0.98)).astype(float)
    y = 1.5 * d + x + 0.5 * u + rng.normal(size=n)
    design = np.column_stack([np.ones(n), x])

    def residual(v):
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    tsls = (residual(z) @ residual(y)) / (residual(z) @ residual(d))
    iv = mb.iv_effect({"y": y, "d": d, "z": z, "x": x}, _spec(covariates=("x",)), "z",
                      folds=5, seed=3, tune=False, mstop=200)
    assert iv.estimate == pytest.approx(tsls, abs=0.03)


def test_iv_effect_accepts_a_binary_instrument_and_reports_the_first_stage():
    data = _late_data(seed=8, n=3000)
    data["z"] = data["z"].astype(float)
    iv = mb.iv_effect(data, _spec(covariates=("x1", "x2")), "z", folds=4, seed=2,
                      tune=False, mstop=150)
    assert iv.nuisance["binary_instrument"] is True
    assert abs(iv.estimate - 2.0) < 0.4


def test_iv_effect_warns_for_a_weak_instrument():
    rng = np.random.default_rng(12)
    n = 1500
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = (rng.uniform(size=n) < np.clip(0.4 + 0.2 * u + 0.01 * z, 0.02, 0.98)).astype(float)
    y = 1.5 * d + x + 0.5 * u + rng.normal(size=n)
    with pytest.warns(RuntimeWarning, match="weak"):
        iv = mb.iv_effect({"y": y, "d": d, "z": z, "x": x}, _spec(covariates=("x",)), "z",
                          folds=4, tune=False, mstop=150)
    assert iv.weak_instrument is True
    assert "WEAK INSTRUMENT" in iv.describe()


def test_iv_describe_states_the_estimand_and_the_assumptions():
    data = _late_data(seed=9, n=1500)
    iv = mb.iv_effect(data, _spec(covariates=("x1", "x2")), "z", folds=3, tune=False,
                      mstop=120)
    text = iv.describe()
    assert "complier" in text
    assert "monotonicity" in text
    assert "exclusion" in text
    assert "independence" in text
    assert "partial F" in text


def test_iv_result_exports_to_json_and_a_frame():
    data = _late_data(seed=10, n=800)
    iv = mb.iv_effect(data, _spec(covariates=("x1", "x2")), "z", folds=3, tune=False,
                      mstop=100)
    record = iv.to_dict()
    json.dumps(record)
    assert record["instrument"] == "z"
    assert "influence" not in record
    frame = iv.to_frame(output="records")
    assert len(frame) == 800
    assert {"influence", "fold", "instrument_residual"} <= set(frame[0])


def test_iv_effect_refuses_a_bad_instrument():
    data = _late_data(seed=11, n=300)
    with pytest.raises(ValueError, match="distinct"):
        mb.iv_effect(data, _spec(covariates=("x1", "x2")), "x1", tune=False, mstop=50)
    categorical = dict(data, z=np.where(data["z"] > 0, "on", "off"))
    with pytest.raises(TypeError, match="categorical"):
        mb.iv_effect(categorical, _spec(covariates=("x1", "x2")), "z", tune=False, mstop=50)
    constant = dict(data, z=np.ones(300))
    with pytest.raises(ValueError, match="constant"):
        mb.iv_effect(constant, _spec(covariates=("x1", "x2")), "z", tune=False, mstop=50)


# Exports and the agent guide.


def test_new_names_are_exported():
    assert "iv_effect" in mb.__all__
    assert "IVResult" in mb.__all__
    assert callable(mb.iv_effect)


def test_causal_and_uncertainty_guides_are_byte_identical_across_both_copies():
    for topic in ("causal.md", "uncertainty.md"):
        source = (ROOT / "docs/skills/analyzing-data-with-mboostrs/topics" / topic).read_bytes()
        copy = (ROOT / "python/mboostrs/agent_docs/topics" / topic).read_bytes()
        assert source == copy, topic


def test_guide_shows_gam_refit_effects_as_a_method_and_the_iv_estimator():
    uncertainty = (ROOT / "python/mboostrs/agent_docs/topics/uncertainty.md").read_text()
    instruments = (ROOT / "python/mboostrs/agent_docs/topics/causal-instruments.md").read_text()
    heterogeneity = (ROOT / "python/mboostrs/agent_docs/topics/causal-heterogeneity.md").read_text()
    assert "refit.effects(df.head(5), level=0.9)" in uncertainty
    assert "## Term effect intervals: gam_refit(...).effects" in uncertainty
    assert "## Instrumental variables with iv_effect" in instruments
    assert "tied_rows" in heterogeneity
