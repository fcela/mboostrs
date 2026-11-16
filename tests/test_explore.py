"""Tests for the fit-inspect-respecify loop (`mboostrs.explore`)."""
import numpy as np
import pytest

import mboostrs as mboost
from mboostrs.explore import (
    cv_risk,
    sensitivity,
    split_terms,
    suggest_specifications,
    term_report,
    term_variable,
)


@pytest.fixture
def curved():
    """y bends in x, is linear in z, and does not depend on noise at all."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 10, n)
    z = rng.normal(0, 1, n)
    return {"x": x, "z": z, "noise": rng.normal(0, 1, n),
            "y": np.sin(x) + 0.8 * z + rng.normal(0, 0.3, n)}


def test_split_terms_respects_nesting():
    assert split_terms("bbs(a, df=4) + bols(b)") == ["bbs(a, df=4)", "bols(b)"]
    assert split_terms("bbs(a, df=2) %X% bbs(b, df=2) + bols(c)") == [
        "bbs(a, df=2) %X% bbs(b, df=2)", "bols(c)"]


def test_term_variable_reads_the_column():
    assert term_variable("bbs(area, df=4)") == "area"
    assert term_variable("btree(a, b, maxdepth=3)") == "a"


def test_cv_risk_is_reproducible_and_fold_matched(curved):
    a = cv_risk("bbs(x, df=4) + bols(z)", curved, "y", folds=3, mstop=100)
    b = cv_risk("bbs(x, df=4) + bols(z)", curved, "y", folds=3, mstop=100)
    assert a == b


def test_a_term_that_matters_earns_more_than_one_that_does_not(curved):
    report = term_report("y ~ bbs(x, df=4) + bols(z) + bols(noise)", curved,
                         folds=3, mstop=200, shapes=False)
    earns = {row["variable"]: row["earns"] for row in report.contributions}
    assert earns["x"] > earns["noise"]
    assert earns["z"] > earns["noise"]


def test_a_curved_effect_prefers_a_curve(curved):
    report = term_report("y ~ bbs(x, df=4) + bols(z)", curved, folds=3, mstop=200)
    shape = {row["variable"]: row for row in report.shapes}["x"]
    assert shape["as bbs"] < shape["as bols"]      # lower risk is better


def test_a_linear_effect_does_not_need_a_curve():
    rng = np.random.default_rng(1)
    n = 600
    straight = {"x": rng.uniform(0, 10, n), "w": rng.normal(0, 1, n)}
    straight["y"] = 2.0 * straight["x"] + rng.normal(0, 0.5, n)
    report = term_report("y ~ bbs(x, df=4) + bols(w)", straight, folds=3, mstop=200)
    shape = {row["variable"]: row for row in report.shapes}["x"]
    assert shape["best"] == "bols"


def test_binary_columns_are_not_offered_other_shapes(curved):
    frame = dict(curved)
    frame["flag"] = (frame["z"] > 0).astype(float)
    report = term_report("y ~ bbs(x, df=4) + bols(flag)", frame, folds=3, mstop=100)
    assert "flag" not in {row["variable"] for row in report.shapes}


def test_report_needs_a_response_when_the_formula_has_none(curved):
    with pytest.raises(ValueError, match="response must be given"):
        term_report("bbs(x, df=4)", curved, folds=2, mstop=50)


def test_term_report_repr_labels_the_score_it_used(curved):
    """The default baseline is a family risk; only an explicit `score="mse"`
    reports squared error. The old repr said "out-of-fold MSE" for both."""
    family = term_report("bols(x) + bols(z)", curved, "y", folds=2, mstop=50)
    assert "out-of-fold family risk" in repr(family)
    assert "MSE" not in repr(family)
    mse = term_report("bols(x) + bols(z)", curved, "y", folds=2, mstop=50,
                      score="mse")
    assert "out-of-fold MSE" in repr(mse)


def test_suggestions_are_ranked_and_name_the_change(curved):
    report = term_report("y ~ bbs(x, df=4) + bols(z) + bols(noise)", curved,
                         folds=3, mstop=200)
    suggestions = suggest_specifications(report)
    assert suggestions == sorted(suggestions, key=lambda s: -s.expected_gain)
    assert any(s.action == "drop" and "noise" in s.term for s in suggestions)


def test_suggestions_accept_screened_pairs(curved):
    report = term_report("y ~ bbs(x, df=4) + bols(z)", curved, folds=3,
                         mstop=100, shapes=False)
    pairs = mboost.screen_interactions(
        {k: v for k, v in curved.items() if k != "y"},
        residual=curved["x"] * curved["z"])
    # #298 phase 1: the screen delegates to the Rust pass, and the planted
    # x*z product still comes back first and positive.
    assert (pairs[0].left, pairs[0].right) == ("x", "z")
    assert pairs[0].removed > 0
    suggestions = suggest_specifications(report, screened=pairs[:1])
    assert any(s.action == "add" for s in suggestions)


def _numpy_reference_screen(values, residual, bins=4, seed=0, pairs=()):
    """The numpy pass `screen_interactions` ran before the Rust delegate
    (#298 phase 1): same design — quantile cells fit on one half, MSE removed
    measured on the other — kept as an A/B reference."""
    rng = np.random.default_rng(seed)
    first = rng.permutation(len(residual)) < len(residual) // 2
    codes = np.empty(values.shape, dtype=np.int64)
    cells = np.full(values.shape[1], bins, dtype=np.int64)
    for j in range(values.shape[1]):
        edges = np.quantile(values[first, j], np.linspace(0, 1, bins + 1)[1:-1])
        codes[:, j] = np.searchsorted(edges, values[:, j])
    held = residual[~first]
    baseline = float(np.mean((held - held.mean()) ** 2))
    scored = []
    for a, b in pairs:
        size = int(cells[a] * cells[b])
        cell_fit = codes[first, a] * cells[b] + codes[first, b]
        cell_held = codes[~first, a] * cells[b] + codes[~first, b]
        total = np.bincount(cell_fit, weights=residual[first], minlength=size)
        count = np.bincount(cell_fit, minlength=size)
        means = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        scored.append(baseline - float(np.mean((held - means[cell_held]) ** 2)))
    return scored


def test_screen_agrees_with_the_numpy_reference_it_replaced():
    """The Rust delegate and the numpy pass it replaced share the design, so
    on planted structure they agree on the ranking and, within the noise a
    different half-split introduces, on the scores (#298 phase 1)."""
    rng = np.random.default_rng(7)
    n = 3000
    frame = {c: rng.uniform(-1, 1, n) for c in ("a", "b", "c", "d", "e")}
    residual = frame["a"] * frame["b"] + rng.normal(0, 0.1, n)
    values = np.column_stack(list(frame.values()))
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]

    rust = {(p.left, p.right): p.removed
            for p in mboost.screen_interactions(frame, residual=residual, seed=0)}
    reference = _numpy_reference_screen(values, residual, pairs=pairs)

    # The planted pair wins for both, by name and by column index.
    assert max(rust, key=rust.get) == ("a", "b")
    assert pairs[int(np.argmax(reference))] == (0, 1)
    # Scores agree within half-split noise: loosely for noise pairs (the
    # tolerance), tightly for the planted one (a fifth of its score).
    names = list(frame)
    for (a, b), removed in zip(pairs, reference, strict=False):
        by_name = (names[a], names[b])
        assert abs(rust[by_name] - removed) < max(0.02, 0.2 * abs(removed))


def test_sensitivity_reports_range_direction_and_turning(curved):
    model = mboost.gamboost("y ~ bbs(x, df=4) + bols(z)", data=curved,
                            family=mboost.Gaussian(),
                            control=mboost.boost_control(300, 0.1))
    rows = {r["variable"]: r for r in
            sensitivity(model, {k: v for k, v in curved.items() if k != "y"},
                        variables=["x", "z"])}
    assert rows["x"]["turns"] is True       # sin over 0..10 turns
    assert rows["z"]["turns"] is False      # a straight line does not
    assert rows["x"]["range of the effect"] > 0


def test_sensitivity_uses_the_models_own_predictors(curved):
    """A frame column the model never saw must not be averaged or gridded."""
    model = mboost.gamboost("y ~ bbs(x, df=4)", data=curved,
                            family=mboost.Gaussian(),
                            control=mboost.boost_control(200, 0.1))
    data = dict(curved)
    data["unrelated"] = np.linspace(0, 1, len(curved["y"]))
    rows = sensitivity(model, data)
    assert [r["variable"] for r in rows] == ["x"]
    with pytest.raises(ValueError, match="not a predictor"):
        sensitivity(model, data, variables=["unrelated"])


def test_sensitivity_grids_a_categorical_over_its_levels(curved):
    """A categorical predictor is evaluated at its actual levels, and an
    unordered one carries no slope or direction (no scale, no order)."""
    rng = np.random.default_rng(0)
    n = len(curved["y"])
    fac = rng.choice(["a", "b", "c"], size=n)
    data = {"x": curved["x"], "y": curved["y"], "f": fac}
    model = mboost.gamboost("y ~ bbs(x, df=4) + brandom(f)", data=data,
                            family=mboost.Gaussian(),
                            control=mboost.boost_control(200, 0.1))
    rows = {r["variable"]: r for r in sensitivity(model, data)}
    row = rows["f"]
    assert row["range of the effect"] >= 0
    # Unordered: no scale for a slope, no order for a direction.
    assert row["slope, 10% to 90%"] != row["slope, 10% to 90%"]  # is nan
    assert row["turns"] != row["turns"]                          # is nan
    assert rows["x"]["turns"] is True


def test_sensitivity_names_a_string_bols_by_its_source_column(curved):
    """`bols(<string column>)` is fitted through a derived `__by0` design
    column; sensitivity must report the source column the caller passed and
    treat it as categorical, not ask for the derived one (issue #309)."""
    rng = np.random.default_rng(0)
    n = len(curved["y"])
    data = {"x": curved["x"], "y": curved["y"],
            "gender": rng.choice(["female", "male"], size=n)}
    model = mboost.gamboost("y ~ bbs(x, df=4) + bols(gender)", data=data,
                            family=mboost.Gaussian(),
                            control=mboost.boost_control(200, 0.1))
    rows = {r["variable"]: r for r in sensitivity(model, data)}
    # The source column, in formula order -- never the derived `__by0`.
    assert set(rows) == {"x", "gender"}
    # A two-level string column is an unordered categorical: held at its
    # modal level for the other terms, gridded over its levels, no slope.
    assert rows["gender"]["slope, 10% to 90%"] != rows["gender"]["slope, 10% to 90%"]
    assert rows["gender"]["turns"] != rows["gender"]["turns"]
    assert rows["x"]["turns"] is True
    with pytest.raises(ValueError, match="not a predictor"):
        sensitivity(model, data, variables=["__by0"])


def test_sensitivity_reads_a_datetime_predictor(curved):
    """A time axis is gridded as a time axis, and reported numerically."""
    import datetime as dt
    import polars as pl

    n = len(curved["y"])
    base = dt.date(2020, 1, 1)
    df = pl.DataFrame({"d": [base + dt.timedelta(days=i) for i in range(n)],
                       "y": curved["y"]})
    model = mboost.gamboost("y ~ bbs(d, df=4)", data=df,
                            family=mboost.Gaussian(),
                            control=mboost.boost_control(100, 0.1))
    rows = sensitivity(model, df)
    assert [r["variable"] for r in rows] == ["d"]
    assert rows[0]["range of the effect"] >= 0
    assert np.isfinite(rows[0]["slope, 10% to 90%"])


def test_suggestions_mark_detected_signals_as_weaker_evidence(curved):
    """A screened pair is evidence of structure, not a measured gain."""
    report = term_report("y ~ bbs(x, df=4) + bols(z)", curved, folds=3,
                         mstop=100, shapes=False)
    pairs = mboost.screen_interactions(
        {k: v for k, v in curved.items() if k != "y"},
        residual=curved["x"] * curved["z"])
    suggestions = suggest_specifications(report, screened=pairs[:1])
    kinds = {s.action: s.basis for s in suggestions}
    assert kinds.get("add") == "detected"
    assert all(s.basis == "measured" for s in suggestions if s.action != "add")


def test_measured_suggestions_outrank_detected_ones(curved):
    report = term_report("y ~ bbs(x, df=4) + bols(z) + bols(noise)", curved,
                         folds=3, mstop=200)
    pairs = mboost.screen_interactions(
        {k: v for k, v in curved.items() if k != "y"},
        residual=curved["x"] * curved["z"] * 50.0)      # a large detected score
    suggestions = suggest_specifications(report, screened=pairs[:1])
    bases = [s.basis for s in suggestions]
    assert bases == sorted(bases, key=lambda b: b != "measured")


def test_compare_specifications_shares_folds_and_reports_change(curved):
    rows = mboost.compare_specifications(
        {"linear": "bols(x) + bols(z)",
         "smooth": "bbs(x, df=4) + bols(z)"},
        curved, "y", folds=3, mstop=200)
    assert [r["specification"] for r in rows] == ["linear", "smooth"]
    assert rows[0]["change"] == 0.0
    # x enters through a sine, so the smooth must beat the line
    assert rows[1]["change"] < 0



def test_compare_specifications_reads_a_whole_formula_as_its_right_hand_side(curved):
    """An agent passed ``"y ~ bbs(x) + ..."`` and got ``cannot parse formula
    term``. A formula for the same response scores exactly as its right-hand
    side does; one for another response is refused with the fix spelled out."""
    rhs = mboost.compare_specifications(
        ["bols(x) + bols(z)", "bbs(x, df=4) + bols(z)"], curved, "y", folds=3, mstop=200)
    whole = mboost.compare_specifications(
        ["y ~ bols(x) + bols(z)", "y ~ bbs(x, df=4) + bols(z)"], curved, "y", folds=3, mstop=200)
    assert [r["specification"] for r in whole] == [r["specification"] for r in rhs]
    assert [r["out-of-fold risk"] for r in whole] == [r["out-of-fold risk"] for r in rhs]
    named = mboost.compare_specifications(
        {"linear": "y ~ bols(x) + bols(z)", "smooth": "bbs(x, df=4) + bols(z)"},
        curved, "y", folds=3, mstop=200)
    assert named[1]["rhs"] == "bbs(x, df=4) + bols(z)"
    assert named[0]["rhs"] == "bols(x) + bols(z)"
    with pytest.raises(ValueError, match="takes right-hand sides"):
        mboost.compare_specifications(["z ~ bols(x)"], curved, "y", folds=3, mstop=50)

def test_sensitivity_holds_a_group_term_at_a_level_not_a_mean():
    """`brandom(g)` reads its column as group labels whatever its dtype, so
    neither the grid nor the held-fixed value may be an interpolation: the
    engine refuses a level of 1.44, and would be wrong to accept one. An
    integer-coded group column used to be gridded by linspace and held at its
    mean, which stopped the fit with "Brandom level ... must be a finite
    integer"."""
    import mboostrs as mb

    rng = np.random.default_rng(0)
    n = 120
    data = {"x": rng.normal(size=n), "g": rng.integers(0, 3, size=n).astype(float)}
    data["y"] = np.sin(data["x"]) + 0.5 * data["g"] + rng.normal(scale=0.3, size=n)
    model = mb.gamboost("y ~ bbs(x) + brandom(g)", data=data,
                        family=mb.Gaussian(), control=mb.boost_control(40, 0.1))

    rows = {row["variable"]: row for row in mb.sensitivity(model, data)}
    assert set(rows) == {"x", "g"}
    # Unordered levels have no scale to take a slope along.
    assert np.isnan(rows["g"]["slope, 10% to 90%"])
    assert np.isnan(rows["g"]["turns"])
    # The measured predictor keeps a real slope.
    assert np.isfinite(rows["x"]["slope, 10% to 90%"])


def test_tune_mstop_returns_the_fitted_model_at_the_chosen_step():
    """One call tunes and fits: `.model` is the sweep's full-data fit cut to
    `mstop`, identical to fitting that many steps directly."""
    rng = np.random.default_rng(0)
    data = {"x1": rng.uniform(-2, 2, 300), "x2": rng.uniform(-2, 2, 300)}
    data["y"] = np.sin(data["x1"]) + rng.normal(0, 0.3, 300)

    tuning = mboost.tune_mstop("y ~ bbs(x1) + bbs(x2)", data)
    direct = mboost.gamboost("y ~ bbs(x1) + bbs(x2)", data=data,
                         control=mboost.boost_control(mstop=tuning.mstop, nu=0.1))

    assert tuning.model.mstop == tuning.mstop
    np.testing.assert_allclose(np.asarray(tuning.model.predict()).ravel(),
                               np.asarray(direct.predict()).ravel())


def test_tune_mstop_keeps_doubling_until_the_risk_turns():
    """A 50-level random effect with df=4 converges slowly: the old default of
    three doublings stopped at 800 steps, silently, when about 4,000 were
    needed."""
    import warnings

    rng = np.random.default_rng(1)
    b = rng.normal(size=50)
    g = np.repeat(np.arange(50), 20)
    x = 0.8 * b[g] + rng.normal(scale=0.6, size=1000)
    data = {"x": x, "g": g, "y": 0.5 * x + b[g] + rng.normal(scale=0.5, size=1000)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tuning = mboost.tune_mstop("y ~ bbs(x, df=4) + brandom(g)", data)
    assert not tuning.at_boundary
    assert tuning.mstop > 800


def test_tune_mstop_warns_when_the_search_ends_still_falling():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 300)
    data = {"x": x, "y": np.sin(3 * x) + rng.normal(scale=0.1, size=300)}
    with pytest.warns(RuntimeWarning, match="still falling at 20 steps"):
        tuning = mboost.tune_mstop("y ~ bbs(x)", data, start=10, max_mstop=20)
    assert tuning.at_boundary


def test_tune_mstop_aligns_folds_with_complete_cases_under_omit():
    """Issue #39: with na_action='omit' the tuning skeleton estimates on
    complete cases, so fold weights drawn over the full frame name rows it
    does not have and the sweep dies on a shape check. The folds must be
    subset to the rows the fit keeps; tuning the pre-dropped frame with the
    subset scheme then gives the same answer."""
    rng = np.random.default_rng(5)
    n = 200
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=n)
    y = np.sin(3 * x) + .5 * z + rng.normal(scale=.1, size=n)
    holed = np.where(rng.random(n) < .15, np.nan, x)
    data = {"x": holed, "z": z, "y": y}
    keep = np.isfinite(holed)
    assert keep.sum() < n

    # One shared fold design: the same matrix, subset to the rows each fit
    # has. Drawing folds afresh on the smaller frame would compare two
    # different splits, not two routes over the same one.
    from mboostrs.cvrisk import cv_weights

    matrix = cv_weights(n, type="kfold", B=4, seed=1)
    full = mboost.tune_mstop("y ~ bbs(x) + bols(z)", data, folds=matrix,
                             start=20, seed=1, na_action="omit")
    complete = {name: values[keep] for name, values in data.items()}
    reference = mboost.tune_mstop("y ~ bbs(x) + bols(z)", complete, folds=matrix[keep],
                                  start=20, seed=1, na_action="omit")
    assert full.mstop == reference.mstop
    np.testing.assert_allclose(full.risk, reference.risk)
