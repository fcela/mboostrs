"""Claims benchmark rerun 2 friction (issue #447): spread candidates, discover's
tuning cap, extrapolation messages, factor pairs, validate's transformed columns,
and per-stage time estimates."""

import importlib
import types
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs._extrapolation import CollectedRows, forward_extrapolation, row_for
from mboostrs.explore import FACTOR_TENSOR_MAX_LEVELS, _spanned_drop_note, pair_term_for

discover_module = importlib.import_module("mboostrs.discover")


def _households_spread(seed=0, n=600, null=False):
    """A mean driven by x, with a noise scale driven by h, a column the mean never uses.

    h is positive and right-skewed, so discover drafts it on log(h). With
    ``null=True`` the noise is constant and no column drives the spread.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    if null:
        sd = np.full(n, 0.5)
    else:
        sd = 0.15 + 0.6 * np.sqrt(np.exp(5.0) / h)
    y = 0.6 * x + sd * rng.normal(size=n)
    return pl.DataFrame({"x": x, "h": h, "y": y})


def _fit_gaussian(frame):
    return mb.gamboost("y ~ bbs(x, df=4)", frame, family=mb.Gaussian(),
                       control=mb.boost_control(200, 0.1))


# ---------------------------------------------------------------------------
# Items 1/4: the spread test covers every candidate column of the data
# ---------------------------------------------------------------------------


def test_heteroscedasticity_tests_a_column_the_mean_model_does_not_use():
    frame = _households_spread()
    het = mb.heteroscedasticity(_fit_gaussian(frame), frame, transforms={"h": "log"})

    assert "h" in het["tested_columns"]
    assert het["by_column"]["h"]["transform"] == "log"
    assert abs(het["by_column"]["h"]["t"]) >= 2
    assert het["strongest"] == "h"


def test_heteroscedasticity_result_names_the_untested_columns_and_why():
    frame = _households_spread().with_columns(pl.Series("g", ["a", "b", "c", "d"] * 150))
    het = mb.heteroscedasticity(_fit_gaussian(frame), frame, transforms={"h": "log"})

    assert "y" not in het["tested_columns"]
    assert set(het["tested_columns"]) == {"x", "h"}
    assert "g" in het["untested_columns"]
    assert "categorical" in het["untested_columns"]["g"]


def test_heteroscedasticity_log_transform_needs_positive_values():
    frame = _households_spread().with_columns(pl.col("h") - 1e6)
    het = mb.heteroscedasticity(_fit_gaussian(frame), frame, transforms={"h": "log"})

    assert "h" not in het["by_column"]
    assert "positive" in het["untested_columns"]["h"]


def test_heteroscedasticity_columns_argument_limits_the_candidates():
    frame = _households_spread()
    het = mb.heteroscedasticity(_fit_gaussian(frame), frame, columns=["x"])

    assert het["tested_columns"] == ["x"]


def test_discover_nominates_the_spread_driver_outside_the_mean_model():
    result = mb.discover(_households_spread(seed=3), "y", seed=0)

    assert result.spread_formula is not None
    assert "log(h)" in result.spread_formula
    assert result.spread_drivers and result.spread_drivers[0]["variables"] == ["h"]
    stage = [e for e in result.log if e["stage"] == "spread"]
    assert any("tested the spread against 2 column(s)" in e["reason"] for e in stage)


def test_discover_finds_a_spread_driver_when_the_mean_has_no_terms():
    # No mean structure: the mean model is y ~ 1, a constant fit. The spread
    # test still runs against every column.
    rng = np.random.default_rng(7)
    n = 600
    x = rng.uniform(-2, 2, n)
    h = np.exp(rng.normal(5.0, 1.0, n))
    sd = 0.15 + 0.6 * np.sqrt(np.exp(5.0) / h)
    y = sd * rng.normal(size=n)
    result = mb.discover(pl.DataFrame({"x": x, "h": h, "y": y}), "y", seed=0)

    stage = [e for e in result.log if e["stage"] == "spread"]
    assert any("tested the spread against 2 column(s)" in e["reason"] for e in stage)
    assert result.spread_formula is not None and "log(h)" in result.spread_formula


def test_discover_adopts_no_spread_driver_on_null_data():
    result = mb.discover(_households_spread(seed=4, null=True), "y", seed=0)

    assert result.spread_formula is None
    assert result.spread_drivers == []


# ---------------------------------------------------------------------------
# Item 2: a main effect spanned by a tensor is named in its row
# ---------------------------------------------------------------------------


def test_spanned_drop_note_names_the_tensor_and_says_the_fit_is_unchanged():
    reference = "bbs(x, df=4) + bbs(z, df=4) + bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)"
    row = "bbs(z, df=4) + bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)"
    note = _spanned_drop_note(reference, row, 0.0, 0.0)

    assert note == ("the term's columns are spanned by bbs(x, df=2.5, knots=5) %X% "
                    "bbs(z, df=2.5, knots=5); dropping it does not change the fit")


def test_spanned_drop_note_without_zero_change_gives_only_the_spanning_clause():
    reference = "bbs(x, df=4) + bbs(z, df=4) + bbs(x, df=2.5) %X% bbs(z, df=2.5)"
    row = "bbs(z, df=4) + bbs(x, df=2.5) %X% bbs(z, df=2.5)"
    note = _spanned_drop_note(reference, row, -0.01, 0.002)

    assert note == "the term's columns are spanned by bbs(x, df=2.5) %X% bbs(z, df=2.5)"


def test_spanned_drop_note_is_none_for_an_unspanned_drop_and_for_the_reference():
    reference = "bbs(x, df=4) + bbs(z, df=4)"
    assert _spanned_drop_note(reference, reference, 0.0, 0.0) is None
    assert _spanned_drop_note(reference, "bbs(z, df=4)", 0.0, 0.0) is None


def test_spanned_drop_note_ignores_distributional_specifications():
    reference = {"mu": "bbs(x, df=4) + bbs(z, df=4)", "sigma": "1"}
    row = {"mu": "bbs(z, df=4)", "sigma": "1"}
    assert _spanned_drop_note(reference, row, 0.0, 0.0) is None


def test_compare_specifications_rows_carry_a_note_key():
    frame = _households_spread()
    rows = mb.compare_specifications(
        ["bbs(x, df=4)", "bbs(x, df=4) + bbs(h, df=4)"], frame, "y", folds=3, mstop=30, seed=0)

    assert all("note" in row for row in rows)
    assert rows[0]["note"] is None


# ---------------------------------------------------------------------------
# Item 5: the boundary message names max_mstop, and max_mstop reaches the tuning
# ---------------------------------------------------------------------------


def test_max_mstop_caps_the_mean_tuning_grid_and_the_message_names_it(monkeypatch):
    frame = _households_spread()
    calls = []

    def still_falling(formula, data, **kwargs):
        calls.append(kwargs["start"])
        return types.SimpleNamespace(mstop=kwargs["start"], at_boundary=True)

    monkeypatch.setattr(discover_module, "tune_mstop", still_falling)
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False, max_mstop=150)

    assert calls == [100, 150]
    boundary = [w for w in result.warnings if w.startswith("boundary")]
    assert boundary and "discover(max_mstop=...) above 150" in boundary[0]


def test_default_max_mstop_keeps_the_400_step_grid(monkeypatch):
    frame = _households_spread()
    calls = []

    def still_falling(formula, data, **kwargs):
        calls.append(kwargs["start"])
        return types.SimpleNamespace(mstop=kwargs["start"], at_boundary=True)

    monkeypatch.setattr(discover_module, "tune_mstop", still_falling)
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False)

    assert calls == [100, 200, 400]
    assert any("above 400" in w for w in result.warnings if w.startswith("boundary"))


def test_max_mstop_must_be_a_positive_whole_number():
    frame = _households_spread()
    with pytest.raises(ValueError, match="max_mstop"):
        mb.discover(frame, "y", max_mstop=0)


# ---------------------------------------------------------------------------
# Item 6: extrapolation is an argument of discover and its warnings name it
# ---------------------------------------------------------------------------


def test_discover_rejects_an_unknown_extrapolation_policy():
    with pytest.raises(ValueError, match="extrapolation"):
        mb.discover(_households_spread(), "y", extrapolation="sometimes")


def test_discover_passes_extrapolation_to_its_fold_fits(monkeypatch):
    seen = []
    real = discover_module.compare_specifications

    def record(*args, **kwargs):
        seen.append(kwargs.get("extrapolation"))
        return real(*args, **kwargs)

    monkeypatch.setattr(discover_module, "compare_specifications", record)
    mb.discover(_households_spread(n=200), "y", seed=0, interactions=False,
                spread=False, extrapolation="clip")

    assert seen and set(seen) == {"clip"}


def test_extrapolation_warning_from_a_stage_names_discover_once(monkeypatch):
    real = discover_module.compare_specifications

    def noisy(*args, **kwargs):
        out = real(*args, **kwargs)
        rows = CollectedRows()
        rows.append(row_for("bbs(x)", "x", [9.0, 9.5], 0.0, 1.0))
        rows.calls = 1
        forward_extrapolation(rows)
        forward_extrapolation(rows)
        return out

    monkeypatch.setattr(discover_module, "compare_specifications", noisy)
    result = mb.discover(_households_spread(n=200), "y", seed=0, interactions=False,
                         spread=False)

    named = [w for w in result.warnings if "outside the range" in w]
    assert len(named) == 1
    assert "discover(extrapolation='allow')" in named[0]
    assert "compare_specifications(" not in named[0]


# ---------------------------------------------------------------------------
# Item 7: factor pairs are bols-coded; the grouped-folds caveat is for many levels
# ---------------------------------------------------------------------------


def _factor_frame(seed=0, n=400, levels=4):
    rng = np.random.default_rng(seed)
    a = rng.choice([f"L{i}" for i in range(levels)], n)
    b = rng.choice(["u", "v", "w"], n)
    x = rng.normal(size=n)
    y = np.where((a == "L0") & (b == "u"), 1.0, 0.0) + 0.5 * x + rng.normal(0, 0.3, n)
    return pl.DataFrame({"a": a, "b": b, "x": x, "y": y})


def test_few_level_factor_pair_is_a_bols_term_not_brandom():
    frame = _factor_frame()
    term = pair_term_for("a", "b", data=frame)

    assert term.startswith("bols(") and "by=" in term
    assert "brandom" not in term


def test_factor_pair_term_fits_and_predicts():
    frame = _factor_frame()
    term = pair_term_for("a", "b", data=frame)
    model = mb.gamboost(f"y ~ bols(x) + bols(a) + bols(b) + {term}", frame,
                        family=mb.Gaussian(), control=mb.boost_control(100, 0.1))

    assert np.all(np.isfinite(np.asarray(model.predict(frame.head(5)))))


def test_many_level_factor_pair_keeps_the_group_term():
    levels = FACTOR_TENSOR_MAX_LEVELS + 5
    frame = _factor_frame(levels=levels)
    term = pair_term_for("a", "b", data=frame)

    assert term.startswith("brandom(")


# ---------------------------------------------------------------------------
# Item 9: columns a transform reads are read through the transform
# ---------------------------------------------------------------------------


def test_a_raw_column_read_only_through_a_transform_is_not_listed_as_unread():
    from mboostrs._preflight import _extra_column_issues, _Recipe

    # log__x is built from x; no term names x itself.
    recipe = _Recipe(tensor_inputs={}, tensor_names={}, sources={"log__x": ("x",)},
                     imputed=frozenset(), factors=())
    issues = _extra_column_issues(["x", "z", "junk"], ["log__x", "z"], {"y"}, recipe, 20)

    through = [i for i in issues if i["kind"] == "read_through_transform"]
    extra = [i for i in issues if i["kind"] == "extra_column"]
    assert through and through[0]["column"] == "x" and through[0]["severity"] == "info"
    assert through[0]["message"] == "read through derived columns: x (log__x)"
    assert extra and extra[0]["column"] == "junk"
    assert "'x'" not in extra[0]["message"]


def test_validate_still_lists_a_column_no_term_reads():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    junk = rng.normal(size=n)
    y = x + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "junk": junk, "y": y})
    model = mb.gamboost("y ~ bbs(x, df=4)", frame, family=mb.Gaussian(),
                        control=mb.boost_control(100, 0.1))
    issues = mb.validate(model, frame.select(["x", "junk"]).head(20), output="records")

    extra = [i for i in issues if i["kind"] == "extra_column"]
    assert extra and extra[0]["column"] == "junk"


# ---------------------------------------------------------------------------
# Items 16/17: a time estimate per stage, logged before the stage runs; budgets
# ---------------------------------------------------------------------------


def test_each_stage_logs_its_estimated_seconds_before_it_runs():
    result = mb.discover(_households_spread(n=300), "y", seed=0)

    estimates = [e for e in result.log if e["stage"] == "estimate"]
    stages = {e["candidate"] for e in estimates}
    assert {"mean effects", "final model"} <= stages
    assert all(isinstance(e["estimated_seconds"], float) for e in estimates)
    # The mean-effects estimate precedes the first mean-effects entry.
    first_estimate = next(i for i, e in enumerate(result.log)
                          if e["stage"] == "estimate" and e["candidate"] == "mean effects")
    first_entry = next(i for i, e in enumerate(result.log) if e["stage"] == "mean effects")
    assert first_estimate < first_entry


def test_budget_skips_optional_stages_and_still_fits_the_final_model():
    result = mb.discover(_households_spread(n=300), "y", seed=0, budget_seconds=0.0)

    skipped = [e for e in result.log if "skipped" in e["reason"]]
    assert skipped and all(e["estimated_seconds"] is not None for e in skipped)
    assert result.formula  # the final model is fitted regardless
    assert any(e["stage"] == "final model" for e in result.log)


def test_repeated_extrapolation_warnings_of_other_kinds_are_grouped(monkeypatch):
    real = discover_module.compare_specifications

    def noisy(*args, **kwargs):
        out = real(*args, **kwargs)
        for _ in range(3):
            warnings.warn("the same wording 12 times over", UserWarning, stacklevel=2)
        return out

    monkeypatch.setattr(discover_module, "compare_specifications", noisy)
    result = mb.discover(_households_spread(n=200), "y", seed=0, interactions=False,
                         spread=False)

    grouped = [w for w in result.warnings if "same wording" in w]
    assert len(grouped) == 1
