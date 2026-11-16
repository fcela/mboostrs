"""aipw and effect_model: spec defaults, nuisance warnings, nuisance formulas and
outcome columns that hold None (issue #429, items 1 to 4).

The simulated designs are small, and the nuisance models run with ``tune=False``
unless a test is about the tuning path.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs.causal import FROM_SPEC, _NuisanceNotes
from mboostrs.explore import FLAT_CURVE_TOLERANCE


def expit(x):
    return 1 / (1 + np.exp(-x))


def simulated(seed=0, n=300):
    """Treatment depends on x1 and x2; the effect is 2 everywhere."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    a = rng.binomial(1, expit(0.8 * x1 - 0.5 * x2)).astype(float)
    y = 1 + 2 * a + x1 + 0.5 * x2 + rng.normal(size=n)
    return {"a": a, "y": y, "x1": x1, "x2": x2}


def with_missing_outcome(data, rows=(3, 7, 11)):
    out = dict(data)
    y = np.array(data["y"], dtype=float)
    y[list(rows)] = np.nan
    out["y"] = y
    return out


def with_outlier(data):
    """Row 0 sits far outside the rest of x1. When its fold holds it out, the
    spline of that fold's nuisance is asked to predict beyond its range."""
    out = dict(data)
    x1 = np.array(data["x1"], dtype=float)
    x1[0] = 30.0
    out["x1"] = x1
    return out


def spec_with(**overrides):
    fields = dict(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="treatment assignment", follow_up="one year",
        adjustment_rationale="x1 and x2 cause both treatment and outcome",
    )
    fields.update(overrides)
    return mb.CausalSpec(**fields)


FAST = dict(folds=5, tune=False, mstop=100)


# Item 1: the CausalSpec supplies propensity_clip, outcome_missing and overlap_bounds.

def test_aipw_takes_propensity_clip_from_the_spec_when_the_call_does_not():
    spec = spec_with(propensity_clip=(0.05, 0.95))
    result = mb.aipw(simulated(), spec, **FAST)
    assert result.diagnostics["propensity_clip"] == (0.05, 0.95)


def test_aipw_explicit_propensity_clip_wins_over_the_spec():
    spec = spec_with(propensity_clip=(0.05, 0.95))
    assert mb.aipw(simulated(), spec, propensity_clip=(0.02, 0.98),
                   **FAST).diagnostics["propensity_clip"] == (0.02, 0.98)
    assert mb.aipw(simulated(), spec, propensity_clip=None,
                   **FAST).diagnostics["propensity_clip"] is None


def test_aipw_takes_outcome_missing_from_the_spec_when_the_call_does_not():
    data = with_missing_outcome(simulated())
    spec = spec_with(outcome_missing="ipw")
    result = mb.aipw(data, spec, **FAST)
    assert result.diagnostics["outcome_missing"] == "ipw"
    assert result.diagnostics["n_missing_outcome"] == 3


def test_aipw_explicit_outcome_missing_wins_over_the_spec():
    data = with_missing_outcome(simulated())
    spec = spec_with(outcome_missing="ipw")
    with pytest.raises(ValueError, match="missing for 3 of 300 rows"):
        mb.aipw(data, spec, outcome_missing="error", **FAST)


def test_aipw_keeps_the_error_rule_when_the_spec_keeps_its_default():
    data = with_missing_outcome(simulated())
    with pytest.raises(ValueError, match="missing for 3 of 300 rows"):
        mb.aipw(data, spec_with(), **FAST)


def test_aipw_takes_overlap_bounds_from_the_spec_when_the_call_does_not():
    spec = spec_with(overlap_bounds=(0.1, 0.9))
    result = mb.aipw(simulated(), spec, **FAST)
    assert tuple(result.diagnostics["overlap"]["bounds"]) == (0.1, 0.9)
    explicit = mb.aipw(simulated(), spec, overlap_bounds=(0.02, 0.98), **FAST)
    assert tuple(explicit.diagnostics["overlap"]["bounds"]) == (0.02, 0.98)


def test_effect_model_takes_propensity_clip_from_the_spec_when_the_call_does_not():
    spec = spec_with(propensity_clip=(0.05, 0.95))
    fx = mb.effect_model(simulated(), spec, folds=3, terms="smooth")
    assert fx.diagnostics["propensity_clip"] == (0.05, 0.95)
    explicit = mb.effect_model(simulated(), spec, folds=3, terms="smooth",
                               propensity_clip=(0.02, 0.98))
    assert explicit.diagnostics["propensity_clip"] == (0.02, 0.98)


def test_effect_model_takes_outcome_missing_and_overlap_bounds_from_the_spec():
    data = with_missing_outcome(simulated())
    spec = spec_with(outcome_missing="ipw", overlap_bounds=(0.1, 0.9))
    fx = mb.effect_model(data, spec, folds=3, terms="smooth")
    assert fx.diagnostics["outcome_missing"] == "ipw"
    assert fx.overlap_bounds == (0.1, 0.9)


def test_the_flat_tolerance_default_is_the_one_tune_mstop_uses():
    import inspect

    default = inspect.signature(mb.aipw).parameters["flat_tolerance"].default
    assert default == FLAT_CURVE_TOLERANCE
    default = inspect.signature(mb.effect_model).parameters["flat_tolerance"].default
    assert default == FLAT_CURVE_TOLERANCE


def test_the_spec_defaults_are_the_sentinel_in_both_signatures():
    import inspect

    for function in (mb.aipw, mb.effect_model):
        parameters = inspect.signature(function).parameters
        assert parameters["propensity_clip"].default is FROM_SPEC
        assert parameters["outcome_missing"].default is FROM_SPEC
    assert inspect.signature(mb.aipw).parameters["overlap_bounds"].default is FROM_SPEC
    assert inspect.signature(mb.effect_model).parameters["overlap_bounds"].default is FROM_SPEC


# Item 2: nuisance warnings are issued once per call, naming the nuisance and the argument.

def _extrapolation_warnings(caught):
    return [w for w in caught if issubclass(w.category, ExtrapolationWarning)]


def test_aipw_extrapolation_warning_names_the_nuisance_and_the_argument_once():
    data = with_outlier(simulated())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.aipw(data, spec_with(), smooth=True, tune=False, mstop=100)
    found = _extrapolation_warnings(caught)
    assert found, "the outlier row should extrapolate in one fold"
    for item in found:
        assert "aipw, " in str(item.message)
        assert "aipw(extrapolation='allow')" in str(item.message)
    propensity = [w for w in found if "propensity model" in str(w.message)]
    assert len(propensity) == 1


def test_aipw_extrapolation_allow_is_silent():
    data = with_outlier(simulated())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.aipw(data, spec_with(), smooth=True, tune=False, mstop=100,
                extrapolation="allow")
    assert not _extrapolation_warnings(caught)


def test_aipw_extrapolation_error_refuses_the_nuisance_prediction():
    data = with_outlier(simulated())
    with pytest.raises(ValueError, match="outside the range"):
        mb.aipw(data, spec_with(), smooth=True, tune=False, mstop=100,
                extrapolation="error")


def test_aipw_records_the_extrapolation_policy_it_used():
    result = mb.aipw(simulated(), spec_with(), **FAST)
    assert result.nuisance["extrapolation"] == "warn"
    assert result.nuisance["flat_tolerance"] == FLAT_CURVE_TOLERANCE


def test_aipw_flat_curve_warning_names_the_nuisance_and_the_argument():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.aipw(simulated(), spec_with(), smooth=True, tune=True, folds=3,
                inner_folds=3, mstop=50, flat_tolerance=10.0)
    flat = [w for w in caught if "flat cross-validated curve" in str(w.message)]
    assert flat, "a tolerance of 10 makes every curve flat"
    assert all(issubclass(w.category, RuntimeWarning) for w in flat)
    assert any("propensity model" in str(w.message) for w in flat)
    assert all("aipw(flat_tolerance=None)" in str(w.message) for w in flat)
    assert all("tune_mstop" not in str(w.message) for w in flat)


def test_aipw_flat_tolerance_none_turns_the_flat_check_off():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.aipw(simulated(), spec_with(), smooth=True, tune=True, folds=3,
                inner_folds=3, mstop=50, flat_tolerance=None)
    assert not [w for w in caught if "flat cross-validated curve" in str(w.message)]


def test_effect_model_flat_curve_warning_names_its_nuisance_and_argument():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.effect_model(simulated(), spec_with(), folds=3, flat_tolerance=10.0)
    flat = [w for w in caught if "flat cross-validated curve" in str(w.message)]
    assert any("effect_model, propensity model" in str(w.message) for w in flat)
    assert all("effect_model(flat_tolerance=None)" in str(w.message) for w in flat)


def test_effect_model_extrapolation_warning_names_the_nuisance_and_argument():
    data = with_outlier(simulated())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.effect_model(data, spec_with(), folds=5)
    # The second stage's own extrapolation note is not a nuisance warning and
    # keeps its wording; only the nuisance warnings are checked here.
    nuisance = [w for w in _extrapolation_warnings(caught)
                if str(w.message).startswith("effect_model, ")]
    assert any("propensity model" in str(w.message) for w in nuisance)
    assert all("effect_model(extrapolation='allow')" in str(w.message) for w in nuisance)


def test_effect_model_extrapolation_error_refuses_the_nuisance_prediction():
    data = with_outlier(simulated())
    with pytest.raises(ValueError, match="outside the range"):
        mb.effect_model(data, spec_with(), folds=5, extrapolation="error")


def test_notes_flat_check_uses_the_tolerance_it_was_given():
    class Tuned:
        risk = (0.70, 0.69, 0.68)
        mstop = 2

    # |0.68 - 0.70| / 0.70 is 2.9 percent.
    loose = _NuisanceNotes("aipw", 0.05)
    loose.observe_tuning("propensity", Tuned())
    assert loose._flat["propensity"][0][0] == 2
    tight = _NuisanceNotes("aipw", 0.01)
    tight.observe_tuning("propensity", Tuned())
    assert "propensity" not in tight._flat
    off = _NuisanceNotes("aipw", None)
    off.observe_tuning("propensity", Tuned())
    assert not off._flat


def test_notes_name_a_penalty_drift_warning_by_nuisance():
    notes = _NuisanceNotes("aipw", None)
    with notes.scope("mu1"):
        warnings.warn("boosting has undone much of the penalty of 1 learner(s): "
                      "bbs(x1): 2 of 3 df after 40 selections", UserWarning, stacklevel=2)
    with pytest.warns(UserWarning,
                      match=r"aipw, treated-arm outcome model: boosting has undone"):
        notes.emit()


def test_notes_pass_other_warnings_through_unchanged():
    notes = _NuisanceNotes("aipw", None)
    with notes.scope("propensity"):
        warnings.warn("an unrelated note", UserWarning, stacklevel=2)
    with pytest.warns(UserWarning, match=r"^an unrelated note$"):
        notes.emit()


def test_notes_collect_extrapolation_rows_into_one_warning_per_nuisance():
    from mboostrs._extrapolation import RangeRow, _emit

    notes = _NuisanceNotes("aipw", None)
    row = RangeRow(learner="bbs(x1)", variable="x1", train_low=-2.0, train_high=2.0,
                   new_low=30.0, new_high=30.0, n_below=0, n_above=1)
    # Two predict calls report the same row; the scope collects both.
    for _ in range(2):
        with notes.scope("propensity"):
            _emit(ExtrapolationWarning("x", rows=[row]), stacklevel=2)
    with pytest.warns(ExtrapolationWarning, match="propensity model") as caught:
        notes.emit()
    assert len(caught) == 1
    assert "2 fold predictions" in str(caught[0].message)
    assert "aipw(extrapolation='allow')" in str(caught[0].message)


# Item 3: nuisance formulas.

def test_aipw_propensity_terms_replace_the_built_formula_and_are_recorded():
    terms = "bbs(x1, df=4) + btree(x2, maxdepth=1)"
    result = mb.aipw(simulated(), spec_with(), tune=False, mstop=100,
                     propensity_terms=terms)
    assert result.nuisance["terms"]["propensity"] == terms
    assert result.nuisance["custom_terms"] == {"propensity": True, "outcome": False}
    assert np.isfinite(result.estimate)


def test_aipw_outcome_terms_in_a_separate_model_are_the_whole_formula():
    result = mb.aipw(simulated(), spec_with(), tune=False, mstop=100,
                     outcome_model="separate", outcome_terms="bbs(x1, df=4)")
    assert result.nuisance["terms"]["outcome"] == "bbs(x1, df=4)"


def test_aipw_outcome_terms_in_a_pooled_model_add_the_treatment():
    result = mb.aipw(simulated(), spec_with(), tune=False, mstop=100,
                     outcome_model="pooled", outcome_terms="bbs(x1, df=4)")
    assert result.nuisance["terms"]["outcome"] == "bbs(x1, df=4) + a"


def test_aipw_outcome_terms_refuse_outcome_interactions():
    with pytest.raises(ValueError, match="by= terms"):
        mb.aipw(simulated(), spec_with(), tune=False, mstop=100,
                outcome_model="pooled", outcome_interactions=True,
                outcome_terms="bbs(x1, df=4)")


def test_aipw_custom_terms_are_checked_before_fitting():
    data = simulated()
    with pytest.raises(ValueError, match="right-hand side only"):
        mb.aipw(data, spec_with(), tune=False, propensity_terms="a ~ x1 + x2")
    with pytest.raises(ValueError, match="bare term"):
        mb.aipw(data, spec_with(), tune=False, outcome_terms="bbs(x1) + a")
    with pytest.raises(ValueError, match="non-empty"):
        mb.aipw(data, spec_with(), tune=False, propensity_terms="   ")


def test_aipw_smooth_can_be_set_per_nuisance():
    result = mb.aipw(simulated(), spec_with(), tune=False, mstop=100,
                     outcome_model="separate",
                     smooth={"propensity": False, "outcome": True})
    assert result.nuisance["terms"]["propensity"] == "bols(x1) + bols(x2)"
    assert result.nuisance["terms"]["outcome"] == "bbs(x1, df=4) + bbs(x2, df=4)"
    assert result.nuisance["smooth_by_nuisance"] == {
        "propensity": False, "outcome": True, "response": False}


def test_aipw_smooth_dict_rejects_unknown_nuisances():
    with pytest.raises(ValueError, match="unknown nuisance"):
        mb.aipw(simulated(), spec_with(), tune=False, smooth={"propensty": True})


def test_aipw_bool_smooth_still_applies_to_every_nuisance():
    result = mb.aipw(simulated(), spec_with(), tune=False, mstop=100, smooth=True,
                     outcome_model="separate")
    assert result.nuisance["smooth_by_nuisance"] == {
        "propensity": True, "outcome": True, "response": True}
    assert result.nuisance["terms"]["propensity"] == "bbs(x1, df=4) + bbs(x2, df=4)"


def test_aipw_refuses_an_unknown_extrapolation_policy_and_a_negative_tolerance():
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        mb.aipw(simulated(), spec_with(), **FAST, extrapolation="loud")
    with pytest.raises(ValueError, match="flat_tolerance"):
        mb.aipw(simulated(), spec_with(), **FAST, flat_tolerance=-1.0)


# Item 4: an outcome column with None values.

def test_none_in_a_numeric_outcome_is_a_missing_value_under_ipw():
    data = simulated()
    y = [float(v) for v in data["y"]]
    for row in (2, 5, 9):
        y[row] = None
    data = {**data, "y": y}
    result = mb.aipw(data, spec_with(), outcome_missing="ipw", **FAST)
    assert result.diagnostics["n_missing_outcome"] == 3


def test_none_in_a_numeric_outcome_follows_the_error_rule():
    data = simulated()
    y = [float(v) for v in data["y"]]
    for row in (2, 5, 9):
        y[row] = None
    with pytest.raises(ValueError, match="missing for 3 of 300 rows"):
        mb.aipw({**data, "y": y}, spec_with(), **FAST)


def test_none_and_nan_are_both_missing_in_a_numeric_outcome():
    data = simulated()
    y = [float(v) for v in data["y"]]
    y[2] = None
    y[5] = float("nan")
    result = mb.aipw({**data, "y": y}, spec_with(), outcome_missing="ipw", **FAST)
    assert result.diagnostics["n_missing_outcome"] == 2


def test_a_boolean_outcome_with_none_is_read_as_zero_and_one():
    data = simulated()
    y = [bool(v > 0) for v in data["y"]]
    y[0] = None
    result = mb.aipw({**data, "y": y}, spec_with(outcome_type="binary"),
                     outcome_missing="ipw", **FAST)
    assert result.diagnostics["n_missing_outcome"] == 1
    assert set(np.unique(result.outcome[~np.isnan(result.outcome)])) <= {0.0, 1.0}


def test_a_text_outcome_with_none_says_missing_values_must_be_null_or_nan():
    data = simulated()
    y = ["high" if v > 0 else "low" for v in data["y"]]
    y[0] = None
    with pytest.raises(NotImplementedError,
                       match="Missing values must be null or NaN in a numeric column"):
        mb.aipw({**data, "y": y}, spec_with(), **FAST)


def test_effect_model_reads_a_none_outcome_the_same_way():
    data = simulated()
    y = [float(v) for v in data["y"]]
    y[4] = None
    fx = mb.effect_model({**data, "y": y}, spec_with(outcome_missing="ipw"), folds=3,
                         terms="smooth")
    assert fx.diagnostics["n_missing_outcome"] == 1
