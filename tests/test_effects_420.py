"""Issue #420, items 5 to 9: refit extrapolation, effect_model's constant
comparison and warnings, result output, penalty-drift aggregation and the
overlap-aware ranking."""

import json
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import causal
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs._model_api import penalty_drift_collecting
from mboostrs.families import Gaussian
from mboostrs.selection import gam_refit, gamlss_refit


def _refit_fixture():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(5 * x) + 0.5 * z + rng.normal(0, 0.2, n)
    data = {"x": x, "z": z, "y": y}
    model = mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=data, family=Gaussian(),
                        control=mb.boost_control(50, .1))
    return gam_refit(model, which="all")


#: One row inside the fitted range of x (0, 1), one row past its upper edge.
NEWDATA = {"x": np.array([0.2, 1.4]), "z": np.array([0.1, 0.3])}


# ----------------------------------------------------------- item 5: refits

def test_refit_interval_flags_the_row_outside_the_training_range():
    refit = _refit_fixture()
    with pytest.warns(ExtrapolationWarning, match="covariate x"):
        frame = refit.predict(NEWDATA, interval=True, output="records")
    assert [row["outside_range"] for row in frame] == [False, True]


def test_refit_clip_holds_the_covariate_at_its_edge():
    refit = _refit_fixture()
    edge = refit.predict({"x": np.array([refit._ranges["x"][1]]), "z": np.array([0.3])},
                         interval=True, output="records")[0]
    with pytest.warns(ExtrapolationWarning):
        clipped = refit.predict(NEWDATA, interval=True, extrapolation="clip",
                                output="records")
    assert clipped[1]["estimate"] == pytest.approx(edge["estimate"], rel=1e-10)
    # The flag still describes the values as given, before clipping.
    assert clipped[1]["outside_range"] is True


def test_refit_error_refuses_and_allow_is_silent():
    refit = _refit_fixture()
    with pytest.raises(ValueError, match="covariate x"):
        refit.predict(NEWDATA, interval=True, extrapolation="error")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        refit.predict(NEWDATA, interval=True, extrapolation="allow")
    assert not [w for w in caught if issubclass(w.category, ExtrapolationWarning)]


def test_refit_rejects_an_unknown_policy():
    refit = _refit_fixture()
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        refit.predict(NEWDATA, extrapolation="linear")


def test_refit_without_spline_terms_has_no_range_to_check():
    rng = np.random.default_rng(5)
    n = 120
    z = rng.uniform(0, 1, n)
    data = {"z": z, "y": 2 * z + rng.normal(0, 0.1, n)}
    model = mb.gamboost("y ~ bols(z)", data=data, family=Gaussian(),
                        control=mb.boost_control(20, .1))
    refit = gam_refit(model, which="all")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = refit.predict({"z": np.array([5.0])}, interval=True, output="records")
    assert frame[0]["outside_range"] is False
    assert not [w for w in caught if issubclass(w.category, ExtrapolationWarning)]


def test_gamlss_refit_flags_rows_per_parameter():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    df = pl.DataFrame({"x": x, "z": z,
                       "y": np.sin(3 * x) + rng.normal(0, 0.2 + 0.4 * z, n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"}, data=df,
                             families=mb.GaussianLSS(), control=mb.boost_control(mstop=100))
        refit = gamlss_refit(fit)
    newdata = {"x": np.array([0.5, 2.0]), "z": np.array([0.5, 0.5])}
    with pytest.warns(ExtrapolationWarning, match="covariate x"):
        frames = refit.predict(newdata, interval=True, output="records")
    assert [row["outside_range"] for row in frames["mu"]] == [False, True]
    # sigma reads only z, which is inside its range for both rows.
    assert [row["outside_range"] for row in frames["sigma"]] == [False, False]


def test_effect_model_predict_passes_the_policy_to_the_refit():
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.6 * x))).astype(float)
    y = 1.0 + t * (1.0 + x) + rng.normal(0, 0.5, n)
    spec = causal.CausalSpec(
        treatment="t", outcome="y", covariates=("x", "z"),
        time_zero="baseline", follow_up="end of study",
        adjustment_rationale="x and z are baseline covariates that precede treatment")
    fx = causal.effect_model({"t": t, "y": y, "x": x, "z": z}, spec, terms="smooth")
    far = {"x": np.array([0.0, 5.0]), "z": np.array([0.0, 0.0])}
    with pytest.raises(ValueError, match="covariate x"):
        fx.predict(far, interval=True, extrapolation="error")
    with pytest.warns(ExtrapolationWarning):
        frame = fx.predict(far, interval=True, output="records")
    assert [row["outside_range"] for row in frame] == [False, True]


# ------------------------------------------------- item 6: constant effect

def _causal_data(seed, effect):
    rng = np.random.default_rng(seed)
    n = 800
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * x + 0.5 * z)))).astype(float)
    y = 1.0 + effect(t, x) + rng.normal(0, 1, n)
    spec = causal.CausalSpec(
        treatment="t", outcome="y", covariates=("x", "z"),
        time_zero="baseline", follow_up="end of study",
        adjustment_rationale="x and z are baseline covariates that precede treatment")
    return {"t": t, "y": y, "x": x, "z": z}, spec


def test_effect_model_reports_the_constant_effect_beside_the_candidates():
    data, spec = _causal_data(7, lambda t, x: 0.5 * t + t * 1.5 * (x > 0))
    fx = causal.effect_model(data, spec, terms="auto")
    constant = fx.diagnostics["constant_effect"]
    assert np.isfinite(fx.constant_rloss)
    assert constant["rloss"] == pytest.approx(fx.constant_rloss)
    # The true average effect is 0.5 + 1.5 * P(x > 0) = 1.25.
    assert constant["ate"] == pytest.approx(1.25, abs=0.3)
    assert "constant effect (the ATE)" in fx.describe()
    # The constant is reported beside the candidates, not among them, so the
    # choice among candidates is unchanged by it.
    assert set(fx.cv_rloss) <= {"smooth", "stumps", "joint"}


def test_effect_model_warns_when_the_candidate_does_not_beat_the_constant():
    data, spec = _causal_data(7, lambda t, x: 0.5 * t)
    with pytest.warns(UserWarning, match="does not beat a constant effect"):
        fx = causal.effect_model(data, spec, terms="auto")
    assert fx.diagnostics["constant_effect"]["beats_by_se"] is False


def test_effect_model_stays_silent_when_the_candidate_beats_the_constant():
    data, spec = _causal_data(7, lambda t, x: 0.5 * t + t * 1.5 * (x > 0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fx = causal.effect_model(data, spec, terms="stumps")
    assert fx.diagnostics["constant_effect"]["beats_by_se"] is True
    assert not [w for w in caught if "constant" in str(w.message)]


def test_effect_model_warns_when_boosted_candidate_stops_near_zero_steps():
    data, spec = _causal_data(7, lambda t, x: 0.5 * t)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fx = causal.effect_model(data, spec, terms="stumps")
    assert fx.diagnostics["tau_mstop"]["stumps"] < 5
    assert [w for w in caught if "close to a constant" in str(w.message)]


# ------------------------------------------------- item 7: result output

def _confounded_data():
    rng = np.random.default_rng(2)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x1 + 0.5 * x2)))).astype(float)
    y = 1.0 + 0.5 * t + x1 + 0.5 * x2 + rng.normal(0, 1, n)
    spec = causal.CausalSpec(
        treatment="t", outcome="y", covariates=("x1", "x2"),
        time_zero="baseline", follow_up="end of study",
        adjustment_rationale="x1 and x2 are baseline covariates that precede treatment")
    return {"t": t, "y": y, "x1": x1, "x2": x2}, spec


def test_confounding_sensitivity_output_records_is_json_safe():
    rows = causal.confounding_sensitivity(
        estimate=0.5, se=0.1, outcome_effects=[0.5, 1.0], imbalances=[-0.2, 0.3],
        output="records")
    assert isinstance(rows, list) and len(rows) == 4
    assert json.loads(json.dumps(rows)) == rows


def test_confounding_sensitivity_default_is_a_polars_frame():
    frame = causal.confounding_sensitivity(
        estimate=0.5, se=0.1, outcome_effects=[0.5], imbalances=[0.3])
    assert isinstance(frame, pl.DataFrame)
    assert causal.confounding_sensitivity(
        estimate=0.5, se=0.1, outcome_effects=[0.5], imbalances=[0.3],
        output="pandas").shape[0] == 1


def test_confounding_benchmark_output_records_is_json_safe():
    data, spec = _confounded_data()
    result = causal.aipw(data, spec, folds=3, mstop=50, tune=False)
    rows = causal.confounding_benchmark(data, result, output="records")
    assert [row["omitted_covariate"] for row in rows] == ["x1", "x2"]
    assert json.loads(json.dumps(rows)) == rows
    assert isinstance(causal.confounding_benchmark(data, result), pl.DataFrame)


# -------------------------------------------- item 8: penalty-drift summary

def _drifting_fit():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + rng.normal(0, 0.3, n)
    return mb.gamboost("y ~ bbs(x, df=4)", data={"x": x, "y": y}, family=Gaussian(),
                       control=mb.boost_control(4000, .1))


def test_penalty_drift_collecting_reports_one_summary_with_the_fit_count():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with penalty_drift_collecting():
            for _ in range(3):
                _drifting_fit()
    drift = [w for w in caught if "penalty" in str(w.message)]
    assert len(drift) == 1
    assert "in 3 fit(s)" in str(drift[0].message)


def test_a_fit_outside_any_routine_warns_for_itself():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _drifting_fit()
    assert len([w for w in caught if "penalty" in str(w.message)]) == 1


# ------------------------------------------- item 9: rank_by_effect

def _frame_ranking(estimate, propensity, **kwargs):
    frame = pl.DataFrame({"estimate": estimate})
    return causal.rank_by_effect(frame, {}, propensity=propensity, **kwargs)


def test_rank_by_effect_ranks_only_rows_inside_the_overlap():
    estimate = np.array([9.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0])
    # Rows 0 and 1 have the largest effects but lie outside the bounds.
    propensity = np.array([0.001, 0.999, 0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5, 0.5])
    ranking = _frame_ranking(estimate, propensity, share=0.2)
    assert ranking.chosen == (2, 3)
    assert ranking.outside == (0, 1)
    assert 0 not in ranking.chosen and 1 not in ranking.chosen
    assert ranking.gain == pytest.approx(3.5)
    assert ranking.gain_interval is None and ranking.level is None
    assert ranking.shortfall == 0


def test_rank_by_effect_reports_a_shortfall_when_overlap_is_small():
    estimate = np.array([3.0, 2.0, 1.0, 0.0])
    propensity = np.array([0.5, 0.001, 0.001, 0.001])
    ranking = _frame_ranking(estimate, propensity, share=0.75)
    assert ranking.chosen == (0,)
    assert ranking.shortfall == 2
    assert "unfilled" in ranking.describe()


def test_rank_by_effect_rejects_bad_arguments():
    estimate = pl.DataFrame({"estimate": [1.0, 2.0]})
    with pytest.raises(ValueError, match="share"):
        causal.rank_by_effect(estimate, {}, propensity=[0.5, 0.5], share=0.0)
    with pytest.raises(ValueError, match="one per row"):
        causal.rank_by_effect(estimate, {}, propensity=[0.5])
    with pytest.raises(ValueError, match="propensity="):
        causal.rank_by_effect(estimate, {})
    with pytest.raises(ValueError, match="bounds"):
        causal.rank_by_effect(estimate, {}, propensity=[0.5, 0.5], bounds=(0.5, 0.2))


def test_rank_by_effect_record_is_json_safe_and_tabulated():
    ranking = _frame_ranking(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.001]))
    record = ranking.to_dict()
    assert record["n_chosen"] == len(ranking.chosen)
    assert "rows" not in record
    assert json.loads(json.dumps(ranking.to_dict(include_arrays=True)))
    table = ranking.table(output="records")
    assert [row["in_overlap"] for row in table] == [True, True, False]
    # One of the two overlap rows is chosen: row 1, the larger estimate.
    assert [row["chosen"] for row in table] == [False, True, False]
    assert [row["rank"] for row in table] == [2, 1, None]


def test_rank_by_effect_on_a_smooth_effect_model_gives_a_gain_interval():
    data, spec = _causal_data(7, lambda t, x: 0.5 * t + t * 1.5 * (x > 0))
    fx = causal.effect_model(data, spec, terms="smooth")
    rows = {"x": data["x"][:200], "z": data["z"][:200]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranking = causal.rank_by_effect(fx, rows, share=0.2)
    assert ranking.gain_interval is not None
    lower, upper = ranking.gain_interval
    assert lower <= ranking.gain <= upper
    assert ranking.level == pytest.approx(0.95)
    assert len(ranking.chosen) == 40
    assert "interval" in ranking.describe()


def test_rank_by_effect_is_exported():
    assert "rank_by_effect" in mb.__all__ and "OverlapRanking" in mb.__all__
    assert mb.rank_by_effect is causal.rank_by_effect
