"""Messages and defaults from issue #416, items 7 to 12.

7   `extrapolation=` on tune_mstop, compare_specifications and term_report.
8   A formula without `~` names the response it needs, before any polars call.
9   ci_test accepts any conditioning column name.
10  A bare categorical name warns once, naming the column and the coding.
11  The Binomial parity note says what it changes in the summary.
12  The penalty-drift warning says what it does and does not change.
"""

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import explore
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs.causal import ci_test

REPO = Path(__file__).resolve().parents[1]


def _outlier_frame(n=200, seed=0):
    """One x value far past the rest, so a fold's held-out row falls outside
    the training range of that fold's spline."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    x[np.argmax(x)] = 5.0
    y = np.sin(3 * np.minimum(x, 1)) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y, "z": rng.uniform(size=n)}


def _extrapolation_warnings(caught):
    return [w for w in caught if issubclass(w.category, ExtrapolationWarning)]


# ---- item 7: extrapolation= ------------------------------------------------

@pytest.mark.parametrize("function", [explore.tune_mstop, explore.compare_specifications,
                                      explore.term_report])
def test_the_three_routines_take_extrapolation_with_warn_as_the_default(function):
    parameter = inspect.signature(function).parameters["extrapolation"]
    assert parameter.default == "warn"


def test_tune_mstop_refuses_a_fold_prediction_outside_its_range_under_error():
    data = _outlier_frame()
    with pytest.raises(ValueError, match="outside the range"):
        explore.tune_mstop("y ~ bbs(x) + bols(z)", data, folds=4, start=50, max_mstop=100,
                           fold_local=True, extrapolation="error")


def test_tune_mstop_warning_names_the_argument_on_tune_mstop():
    data = _outlier_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.tune_mstop("y ~ bbs(x) + bols(z)", data, folds=4, start=50, max_mstop=100,
                           fold_local=True)
    rows = _extrapolation_warnings(caught)
    assert len(rows) == 1
    assert "tune_mstop(extrapolation='allow')" in str(rows[0].message)


def test_tune_mstop_clip_is_accepted_and_says_the_rows_were_clipped():
    data = _outlier_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuning = explore.tune_mstop("y ~ bbs(x) + bols(z)", data, folds=4, start=50,
                                    max_mstop=100, fold_local=True, extrapolation="clip")
    assert tuning.mstop >= 1
    rows = _extrapolation_warnings(caught)
    assert len(rows) == 1 and "were clipped to the training range" in str(rows[0].message)


def test_tune_mstop_default_path_is_unchanged_by_extrapolation():
    """The default family path builds every fold's basis from all rows, so no
    held-out row leaves it: the argument does not change the answer."""
    data = _outlier_frame()
    warn = explore.tune_mstop("y ~ bbs(x) + bols(z)", data, folds=4, start=50, max_mstop=100)
    error = explore.tune_mstop("y ~ bbs(x) + bols(z)", data, folds=4, start=50,
                               max_mstop=100, extrapolation="error")
    assert warn.mstop == error.mstop
    np.testing.assert_allclose(warn.risk, error.risk, rtol=0, atol=0)


def test_term_report_warns_once_naming_term_report_and_allow_is_silent():
    data = _outlier_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.term_report("y ~ bbs(x) + bols(z)", data, response="y", folds=4, mstop=50)
    rows = _extrapolation_warnings(caught)
    assert len(rows) == 1
    assert "term_report(extrapolation='allow')" in str(rows[0].message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.term_report("y ~ bbs(x) + bols(z)", data, response="y", folds=4, mstop=50,
                            extrapolation="allow")
    assert not _extrapolation_warnings(caught)


def test_compare_specifications_passes_the_policy_to_each_fold():
    data = _outlier_frame()
    specs = ["bbs(x) + bols(z)", "bols(x) + bols(z)"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.compare_specifications(specs, data, "y", folds=4, mstop=50)
    rows = _extrapolation_warnings(caught)
    assert len(rows) == 1
    assert "compare_specifications(extrapolation='allow')" in str(rows[0].message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.compare_specifications(specs, data, "y", folds=4, mstop=50,
                                       extrapolation="allow")
    assert not _extrapolation_warnings(caught)


def test_an_unknown_extrapolation_value_is_refused():
    data = _outlier_frame()
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        explore.tune_mstop("y ~ bbs(x)", data, folds=4, start=50, max_mstop=100,
                           extrapolation="bogus")


# ---- item 8: a formula without a response -----------------------------------

def test_tune_mstop_without_a_response_names_the_fix_before_polars():
    data = _outlier_frame()
    with pytest.raises(ValueError) as caught:
        explore.tune_mstop("bbs(x) + bols(z)", data, folds=4, start=50)
    message = str(caught.value)
    assert "has no response" in message
    assert "'y ~ bbs(x) + bols(z)'" in message
    assert "tune_mstop" in message


def test_gamboost_without_a_response_names_the_fix():
    data = _outlier_frame()
    with pytest.raises(ValueError, match=r"has no response.*'y ~ bbs\(x\)'"):
        mb.gamboost("bbs(x)", data=data)


def test_term_report_without_a_response_argument_keeps_its_own_message():
    data = _outlier_frame()
    with pytest.raises(ValueError, match="response must be given"):
        explore.term_report("bbs(x) + bols(z)", data, folds=4, mstop=50)


def test_a_formula_with_a_response_is_not_refused():
    data = _outlier_frame()
    tuning = explore.tune_mstop("y ~ bbs(x)", data, folds=3, start=30, max_mstop=60)
    assert tuning.mstop >= 1


# ---- item 9: ci_test conditioning names -------------------------------------

def test_ci_test_accepts_names_with_equals_and_spaces():
    data = _outlier_frame(n=160)
    rng = np.random.default_rng(4)
    data["smsa=yes"] = (rng.uniform(size=160) < 0.5).astype(float)
    data["urban area"] = rng.uniform(size=160)
    result = ci_test(data, "x", "y", given=["smsa=yes", "urban area"], folds=3)
    assert result.given == ("smsa=yes", "urban area")
    assert np.isfinite(result.p_value)


def test_ci_test_name_with_equals_gives_the_same_result_as_a_plain_name():
    """The formula is built from internal aliases, so the name is not parsed
    and the statistic is the one the plain name gives."""
    data = _outlier_frame(n=160)
    rng = np.random.default_rng(5)
    flag = (rng.uniform(size=160) < 0.5).astype(float)
    with_equals = {**data, "smsa=yes": flag}
    plain = {**data, "smsa_yes": flag}
    a = ci_test(with_equals, "x", "y", given=["smsa=yes"], folds=3)
    b = ci_test(plain, "x", "y", given=["smsa_yes"], folds=3)
    assert a.statistic == pytest.approx(b.statistic, rel=0, abs=1e-12)
    assert a.p_value == pytest.approx(b.p_value, rel=0, abs=1e-12)


def test_ci_test_docstring_states_the_naming_rule():
    doc = " ".join(ci_test.__doc__.split())
    assert "Naming rule:" in doc
    assert "internal alias" in doc


# ---- item 10: bare categorical names ----------------------------------------

def _categorical_frame(n=150, seed=6):
    rng = np.random.default_rng(seed)
    group = rng.choice(["a", "b", "c"], n).tolist()
    x = rng.uniform(size=n)
    y = (np.array(group) == "b") * 1.0 + (np.array(group) == "c") * 2.0 + rng.normal(0, 0.2, n)
    return {"g": group, "x": x, "y": y}


def test_a_bare_categorical_name_warns_once_naming_the_column_and_the_coding():
    data = _categorical_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.gamboost("y ~ g + bbs(x)", data=data, control=mb.boost_control(20))
    messages = [str(w.message) for w in caught if "bare" in str(w.message)]
    assert len(messages) == 1
    assert "'g'" in messages[0]
    assert "brandom(name, lambda=0)" in messages[0]
    assert "bols(g)" in messages[0] and "brandom(g)" in messages[0]


def test_an_explicit_bols_does_not_warn_and_fits_the_same():
    data = _categorical_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explicit = mb.gamboost("y ~ bols(g) + bbs(x)", data=data,
                               control=mb.boost_control(20))
    assert not [w for w in caught if "bare" in str(w.message)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = mb.gamboost("y ~ g + bbs(x)", data=data, control=mb.boost_control(20))
    np.testing.assert_allclose(np.asarray(bare.predict(data)).ravel(),
                               np.asarray(explicit.predict(data)).ravel(), rtol=0, atol=1e-12)


def test_the_guides_no_longer_say_a_bare_categorical_name_is_refused():
    # The step-3 sentence moved from SKILL.md to topics/fit-and-tune.md.
    for path in ("docs/skills/analyzing-data-with-mboostrs/topics/fit-and-tune.md",
                 "docs/skills/analyzing-data-with-mboostrs/topics/categoricals.md"):
        text = (REPO / path).read_text()
        assert "refused rather than silently coded" not in text
        assert "is refused, not silently coded" not in text
        assert re.search(r"brandom\(name, lambda=0\)|brandom\(g, lambda=0\)", text), path


def test_the_packaged_guides_match_the_repository_guides():
    for path in ("SKILL.md", "topics/categoricals.md"):
        source = (REPO / "docs/skills/analyzing-data-with-mboostrs" / path).read_bytes()
        packaged = (REPO / "python/mboostrs/agent_docs" / path).read_bytes()
        assert source == packaged, path


# ---- item 11: the Binomial parity note --------------------------------------

def _binomial_model():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(size=n)
    p = 1 / (1 + np.exp(-2 * np.sin(3 * x)))
    data = {"x": x, "y": (rng.uniform(size=n) < p).astype(float)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost("y ~ bbs(x)", data=data, family=mb.Binomial(),
                           control=mb.boost_control(40, 0.1))


def test_the_binomial_summary_says_what_the_log2_factor_changes():
    text = str(_binomial_model().summary())
    assert "log 2" in text or "log(2)" in text
    assert "1.44" in text
    assert "unscaled" in text


def test_the_reported_binomial_risk_is_the_negative_log_likelihood():
    model = _binomial_model()
    assert float(np.asarray(model.risk_)[-1]) == pytest.approx(-model.logLik(), rel=1e-10)


# ---- item 12: the penalty-drift warning -------------------------------------

def test_the_penalty_drift_warning_says_it_changes_no_fitted_value():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(size=n)
    data = {"x": x, "y": np.sin(3 * x) + rng.normal(0, 0.3, n)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.gamboost("y ~ bbs(x, df=4)", data=data, control=mb.boost_control(3000, 0.1))
    drift =[str(w.message) for w in caught if "undone much of the penalty" in str(w.message)]
    assert drift, "the fixture should trigger the drift warning"
    assert "changes no fitted value" in drift[0]
    assert "cross-validation" in drift[0]


def test_calling_the_drift_check_leaves_the_fit_unchanged():
    from mboostrs._model_api import warn_if_penalty_drifted

    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(size=n)
    data = {"x": x, "y": np.sin(3 * x) + rng.normal(0, 0.3, n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x, df=4)", data=data, control=mb.boost_control(3000, 0.1))
    before = np.asarray(model.predict(data)).copy()
    # The fit above already showed the drift warning once for this process.
    from mboostrs._extrapolation import reset_shown_summaries
    reset_shown_summaries()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_penalty_drifted(model)
    assert any("undone much of the penalty" in str(w.message) for w in caught)
    np.testing.assert_array_equal(before, np.asarray(model.predict(data)))
