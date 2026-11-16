"""Causal messages from issue 487: the argument named by effect_model's
TypeError, the missing-column error of aipw, the documented keys of
diagnostics["overlap"], and extrapolation warnings in Python syntax.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import causal
from mboostrs._extrapolation import range_rows, summarize


def _spec(**extra):
    fields = dict(
        treatment="t", outcome="y", covariates=("x",), estimand="ate",
        outcome_type="continuous", target_population="p", unit="u",
        time_zero="t0", follow_up="1y", treatment_contrast=(0, 1),
        adjustment_rationale="r")
    fields.update(extra)
    return mb.CausalSpec(**fields)


def _data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
    y = 2 * t + x + rng.normal(size=n)
    return {"t": t, "y": y, "x": x}


# -- 5. effect_model(level=) ---------------------------------------------------

def test_effect_model_level_error_names_predict_and_still_rejects_the_call():
    with pytest.raises(TypeError, match="argument 'level'") as info:
        mb.effect_model(_data(), _spec(), level=0.9)
    message = str(info.value)
    assert "predict" in message
    assert "interval=True, level=0.9" in message


def test_effect_model_other_unknown_keyword_keeps_the_plain_message():
    with pytest.raises(TypeError, match="unexpected keyword argument 'lvl'"):
        mb.effect_model(_data(), _spec(), lvl=0.9)


# -- 7. aipw with a covariate that is not a column -----------------------------

def test_aipw_missing_covariate_error_names_the_argument_and_the_column():
    spec = _spec(covariates=("nope",))
    with pytest.raises(ValueError, match="covariates='nope'") as info:
        mb.aipw(_data(), spec, folds=2, mstop=20)
    assert "data has no column" in str(info.value)


def test_aipw_missing_treatment_error_names_the_treatment_argument():
    spec = _spec(treatment="missing_t")
    with pytest.raises(ValueError, match="treatment='missing_t'"):
        mb.aipw(_data(), spec, folds=2, mstop=20)


# -- 6. diagnostics["overlap"] -------------------------------------------------

def test_aipw_overlap_diagnostic_is_a_dict_with_the_documented_keys():
    fit = mb.aipw(_data(), _spec(), folds=2, mstop=20)
    overlap = fit.diagnostics["overlap"]
    assert isinstance(overlap, dict)
    expected = {"bounds", "n", "n_below", "n_above", "regions", "estimand",
                "estimate", "standard_error", "estimate_inside",
                "standard_error_inside", "n_inside", "n_outside", "share_outside"}
    assert expected <= set(overlap)


def test_aipw_docstring_describes_the_overlap_dict_by_its_keys():
    doc = causal.aipw.__doc__
    assert "overlap_report\" of the result" not in doc
    assert "is a plain dict" in doc
    for key in ("n_outside", "share_outside", "estimate_inside"):
        assert f"``{key}``" in doc


# -- 8. extrapolation warnings in Python syntax --------------------------------

def test_extrapolation_warning_shows_a_boundary_knots_term_in_python_form():
    rows = range_rows([(
        "bbs(prior_earnings, df=4, boundary.knots=c(8.21,11.63))",
        "prior_earnings", np.array([12.0, 13.0]), 8.21, 11.63)])
    text = str(summarize(rows, "effect_model").args[0])
    assert "boundary.knots" not in text
    assert "c(8.21" not in text
    assert "bbs(prior_earnings, df=4) fitted on [8.21, 11.63]" in text
    assert "covariate prior_earnings: 2 above 11.63" in text


def test_extrapolation_warning_leaves_a_plain_term_label_unchanged():
    rows = range_rows([("bbs(x, df=4)", "x", np.array([5.0]), 0.0, 1.0)])
    text = str(summarize(rows, None).args[0])
    assert "bbs(x, df=4), covariate x: 1 above 1" in text
    assert "fitted on [" not in text
