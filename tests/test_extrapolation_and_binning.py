"""Where the model stops knowing anything, and where it is approximating.

A P-spline basis is defined between its boundary knots. Outside them the
basis continues by its edge slope, which is a decision the fit does not
constrain. `predict(newdata=)` reported that with `mboost`'s wording:

    Some 'x' values are beyond 'boundary.knots'; Linear extrapolation used.

The `'x'` is the literal string, not the column, so a model over `area` and
`rooms` was told about `'x'`. It fired once per call however many learners
were involved. And it said nothing about how far outside: one row a hair past
the edge and a thousand rows two training ranges out read identically.
Measured on a fit over `x` in `[-2.40, 2.00]`: predicting at `x = -6` returned
2.60 and at `x = +6` returned 3.16 — finite, plausible numbers, differing from
the flat continuation by 3.5 on a response whose residual sd is 0.3.

Binning has the mirror problem. `bins=50` on a column taking seven distinct
values bins nothing; the effective count is seven and nothing said so.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning


@pytest.fixture(scope="module")
def curved():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    z = rng.uniform(0.0, 1.0, size=n)
    return {"x": x, "z": z,
            "y": 2 * np.sin(x) + z + rng.normal(size=n) * 0.3}


@pytest.fixture(scope="module")
def model(curved):
    return mb.gamboost("y ~ bbs(x, df=4) + bbs(z, df=4)", data=curved,
                       control=mb.boost_control(150, 0.1))


@pytest.fixture(scope="module")
def outside(curved):
    return {"x": np.array([-6.0, 0.0, 6.0]),
            "z": np.array([0.5, 0.5, 0.5])}


# ─── the warning says what and how far ─────────────────────────────


def test_the_warning_names_the_learner_and_the_distance(model, outside):
    with pytest.warns(ExtrapolationWarning) as caught:
        model.predict(newdata=outside)
    message = str(caught[0].message)
    assert "bbs(x)" in message
    assert "1 below" in message and "1 above" in message
    assert "training ranges out" in message
    assert "'x' values" not in message, (
        "the old message said the literal string 'x' whatever the column was"
    )


def test_a_learner_inside_its_range_is_not_named(model, outside):
    with pytest.warns(ExtrapolationWarning) as caught:
        model.predict(newdata=outside)
    assert "bbs(z)" not in str(caught[0].message)


def test_every_offending_learner_is_named_not_just_the_first(model):
    """It broke out of the loop after the first, so a second out-of-range
    learner was silent."""
    both = {"x": np.array([-6.0]), "z": np.array([9.0])}
    with pytest.warns(ExtrapolationWarning) as caught:
        model.predict(newdata=both)
    message = str(caught[0].message)
    assert "bbs(x)" in message and "bbs(z)" in message
    assert message.startswith("2 learner(s)")


# ─── the four policies ─────────────────────────────────────────────


def test_allow_is_silent_and_predicts(model, outside):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        predicted = np.asarray(model.predict(newdata=outside,
                                             extrapolation="allow")).ravel()
    assert np.all(np.isfinite(predicted))


def test_error_refuses_with_the_same_content(model, outside):
    with pytest.raises(ValueError, match="outside the range they were fitted"):
        model.predict(newdata=outside, extrapolation="error")


def test_clip_holds_the_effect_flat_outside_the_range(model, curved, outside):
    """The clipped prediction must equal the prediction at the training edge,
    which is what "flat outside" means."""
    with pytest.warns(ExtrapolationWarning, match="clipped to the training range"):
        clipped = np.asarray(model.predict(newdata=outside,
                                           extrapolation="clip")).ravel()
    edges = {"x": np.array([curved["x"].min(), 0.0, curved["x"].max()]),
             "z": np.array([0.5, 0.5, 0.5])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at_edges = np.asarray(model.predict(newdata=edges)).ravel()
    np.testing.assert_allclose(clipped, at_edges, atol=1e-12)


def test_the_two_continuations_differ_enough_to_matter(model, outside):
    """If they agreed the policy would be decoration."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linear = np.asarray(model.predict(newdata=outside,
                                          extrapolation="allow")).ravel()
        flat = np.asarray(model.predict(newdata=outside,
                                        extrapolation="clip")).ravel()
    assert np.max(np.abs(linear - flat)) > 1.0
    assert linear[1] == pytest.approx(flat[1]), "the in-range row is untouched"


def test_an_unknown_policy_is_refused(model, outside):
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        model.predict(newdata=outside, extrapolation="extrapolate")


def test_the_policy_is_checked_even_when_nothing_is_out_of_range(curved):
    """A typo should not pass silently just because this particular call had
    nothing to warn about."""
    linear = mb.gamboost("y ~ bols(x)", data=curved,
                         control=mb.boost_control(20, 0.1))
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        linear.predict(newdata=curved, extrapolation="nonsense")


def test_a_prediction_inside_the_range_says_nothing(model, curved):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        model.predict(newdata=curved)


# ─── a structured warning and a materiality line ───────────────────


def test_the_warning_is_structured_and_carries_the_rows(model, outside):
    from mboostrs._extrapolation import ExtrapolationWarning

    with pytest.warns(ExtrapolationWarning) as caught:
        model.predict(newdata=outside)
    rows = caught[0].message.rows
    assert len(rows) == 1
    assert rows[0].learner == "bbs(x)"
    assert rows[0].n_outside == 2
    assert rows[0].worst_distance > 0.5


def test_small_extrapolations_are_quiet_but_strict_policies_still_act(model, curved):
    """Warning materiality must not change rejection or clipping behavior."""
    from mboostrs._extrapolation import ExtrapolationWarning

    span = curved["x"].max() - curved["x"].min()
    near = {"x": np.array([curved["x"].max() + 0.001 * span]),
            "z": np.array([0.5])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        model.predict(newdata=near)
    with pytest.raises(ValueError, match="outside the range"):
        model.predict(newdata=near, extrapolation="error")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        clipped = model.predict(newdata=near, extrapolation="clip")
    edge = {"x": np.array([curved["x"].max()]), "z": np.array([0.5])}
    np.testing.assert_allclose(clipped, model.predict(newdata=edge), atol=1e-12)


def test_the_report_still_counts_every_row_exactly(model, curved, outside):
    """Warning materiality does not remove small excursions from the report."""
    span = curved["x"].max() - curved["x"].min()
    near = {"x": np.array([curved["x"].max() + 0.001 * span, -6.0]),
            "z": np.array([0.5, 0.5])}
    rows = {row["learner"]: row
            for row in model.range_report(near, output="records")}
    assert rows["bbs(x)"]["n_below"] == 1
    assert rows["bbs(x)"]["n_above"] == 1


# ─── fold loops collect once and summarize once ────────────────────


def test_a_fold_loop_warns_one_summary_for_a_material_leak():
    """A fold that holds an isolated far row rebuilds its basis without it,
    and predicts that row far outside -- a genuinely material extrapolation.
    The fold loop emits one summary, not a paragraph per fold: `cv_risk` has
    no `extrapolation=` to turn off, so one summary states the fact and the
    rest of the folds stay quiet."""
    from mboostrs import explore
    from mboostrs._extrapolation import ExtrapolationWarning

    rng = np.random.default_rng(7)
    x = np.concatenate([rng.uniform(0.0, 1.0, 199), [5.0]])
    data = {"x": x, "y": np.sin(x) + rng.normal(size=200) * 0.3}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.cv_risk("bbs(x, df=4)", data, "y", folds=5, mstop=30,
                        seed=0, score="mse")
    extrapolated = [w for w in caught
                    if issubclass(w.category, ExtrapolationWarning)]
    assert len(extrapolated) == 1, (
        "the fold holding the far row produces one summary, not five "
        "paragraphs (one per fold)"
    )
    assert "bbs(x)" in str(extrapolated[0].message)


def test_conformal_interval_warns_at_most_once(curved):
    from mboostrs._extrapolation import ExtrapolationWarning

    model = mb.gamboost("y ~ bbs(x, df=4)", data=curved,
                        control=mb.boost_control(40, 0.1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.conformal_interval(model, newdata=curved, method="cv+",
                              n_folds=4, seed=1, level=0.9)
    extrapolated = [w for w in caught
                    if issubclass(w.category, ExtrapolationWarning)]
    assert len(extrapolated) <= 1


def test_the_full_workflow_stays_quiet_on_ordinary_data(curved):
    """The budget: a complete supported workflow -- fit, cross-validation,
    model comparison, conformal interval, prediction -- on data whose columns
    cover their own folds emits no extrapolation warning at all. Random folds
    rebuilding the basis on a range that ends a few percent short can
    extrapolate without exceeding the warning threshold."""
    from mboostrs import explore
    from mboostrs._extrapolation import ExtrapolationWarning

    model = mb.gamboost("y ~ bbs(x, df=4) + bbs(z, df=4)", data=curved,
                        control=mb.boost_control(60, 0.1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explore.cv_risk("bbs(x, df=4)", curved, "y",
                        mstop=30, folds=5)
        explore.compare_specifications(
            ["bbs(x, df=4)", "bbs(x, df=4) + bbs(z, df=4)"],
            curved, "y", folds=3, mstop=30,
        )
        mb.conformal_interval(model, method="cv+", n_folds=4,
                              seed=2, level=0.9)
        model.predict(newdata=curved)
    extrapolated = [w for w in caught
                    if issubclass(w.category, ExtrapolationWarning)]
    assert not extrapolated, (
        f"the ordinary workflow should not warn, got "
        f"{[str(w.message) for w in extrapolated]}"
    )


# ─── the structured report ─────────────────────────────────────────


def test_the_range_report_gives_a_row_per_spline_learner(model, outside):
    rows = {row["learner"]: row for row in
            model.range_report(outside, output="records")}
    assert set(rows) == {"bbs(x)", "bbs(z)"}
    x_row = rows["bbs(x)"]
    assert x_row["n_below"] == 1 and x_row["n_above"] == 1
    assert x_row["n_outside"] == 2
    assert rows["bbs(z)"]["n_outside"] == 0


def test_the_distance_is_in_training_ranges(model, curved):
    """The number that separates a boundary case from a different question."""
    span = curved["x"].max() - curved["x"].min()
    far = {"x": np.array([curved["x"].max() + 2 * span]),
           "z": np.array([0.5])}
    row = next(r for r in model.range_report(far, output="records")
               if r["learner"] == "bbs(x)")
    assert row["worst_distance"] == pytest.approx(2.0, rel=1e-6)


def test_the_report_on_training_data_reads_off_the_ranges(model, curved):
    rows = {row["learner"]: row for row in model.range_report(output="records")}
    assert rows["bbs(x)"]["n_outside"] == 0
    assert rows["bbs(x)"]["train_low"] == pytest.approx(curved["x"].min())
    assert rows["bbs(x)"]["train_high"] == pytest.approx(curved["x"].max())


@pytest.mark.parametrize("output", ["polars", "pandas"])
def test_the_report_honours_output(model, outside, output):
    frame = model.range_report(outside, output=output)
    assert type(frame).__module__.split(".")[0] == output


def test_an_explicit_boundary_knot_is_the_range_that_counts(curved):
    """`boundary_knots=` sets where the basis ends, so it is what the report
    compares against, not the column's own range."""
    model = mb.gamboost("y ~ bbs(x, df=4, boundary.knots=c(-4, 4))",
                        data=curved, control=mb.boost_control(20, 0.1))
    row = model.range_report(output="records")[0]
    assert (row["train_low"], row["train_high"]) == (-4.0, 4.0)
    assert row["n_outside"] == 0


# ─── sparse regions ────────────────────────────────────────────────


def test_a_gap_in_the_covariate_is_shaded():
    from mboostrs.plot import _sparse_bands

    rng = np.random.default_rng(0)
    values = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(4, 5, 200)])
    bands = _sparse_bands(values, np.linspace(0, 5, 100))
    assert bands is not None and bands.height == 1
    start, end = bands["start"][0], bands["end"][0]
    assert 0.9 < start < 1.2 and 3.8 < end < 4.1


def test_evenly_spread_data_is_not_shaded():
    from mboostrs.plot import _sparse_bands

    rng = np.random.default_rng(1)
    assert _sparse_bands(rng.uniform(0, 5, 400), np.linspace(0, 5, 100)) is None


def test_the_tails_of_a_normal_covariate_shade_as_two_bands_not_five():
    """Cell counts near the threshold flip on one observation. Without
    closing single-cell gaps the tail of a normal came out as four
    disconnected slivers."""
    from mboostrs.plot import _sparse_bands

    rng = np.random.default_rng(0)
    bands = _sparse_bands(rng.normal(size=400), np.linspace(-3.5, 3.5, 100))
    assert bands is not None and bands.height == 2


def test_the_partial_effect_chart_carries_the_shading():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(4, 5, 200)])
    data = {"x": x, "y": np.sin(x) + rng.normal(size=x.size) * 0.2}
    model = mb.gamboost("y ~ bbs(x, df=4)", data=data,
                        control=mb.boost_control(40, 0.1))
    chart = model.plot_partial_effects()
    # sparse band, residual points, rug, line, zero rule
    assert len(chart.layer) == 5


def test_a_chart_over_evenly_spread_data_carries_no_shading(curved):
    """The shading has to be absent when there is nothing to say, or it stops
    meaning anything."""
    model = mb.gamboost("y ~ bbs(z, df=4)", data=curved,
                        control=mb.boost_control(40, 0.1))
    assert len(model.plot_partial_effects().layer) == 4


# ─── binning: what was requested against what happened ─────────────


@pytest.fixture(scope="module")
def wide():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    few = rng.integers(0, 7, size=n).astype(float)
    return {"x": x, "few": few,
            "y": np.sin(x) + 0.3 * few + rng.normal(size=n) * 0.3}


def test_the_report_shows_effective_bins_not_the_request(wide):
    """`bins=50` on a column taking seven values bins nothing."""
    model = mb.gamboost("y ~ bbs(x, df=4, bins=50) + bbs(few, df=4, bins=50)",
                        data=wide, control=mb.boost_control(20, 0.1))
    rows = {row["learner"]: row
            for row in model.binning_report(output="records")}
    assert rows["bbs(x)"]["requested_bins"] == 50
    assert rows["bbs(x)"]["effective_bins"] == 50
    assert rows["bbs(few)"]["requested_bins"] == 50
    assert rows["bbs(few)"]["distinct_values"] == 7
    assert rows["bbs(few)"]["effective_bins"] == 7, (
        "seven distinct values cannot fill fifty bins"
    )
    assert rows["bbs(few)"]["ties_collapsed"] == 0


def test_the_report_names_the_terms_binning_did_not_reach(wide):
    """A `by=` term has no binned design, so `bins=` skips it. The fit warns
    and the report shows which learners kept their exact storage."""
    with pytest.warns(UserWarning, match="was not applied"):
        model = mb.gamboost("y ~ bbs(x, df=4) + bbs(x, df=4, by=few)",
                            data=wide, bins=64,
                            control=mb.boost_control(20, 0.1))
    binned = [row["binned"] for row in model.binning_report(output="records")]
    assert binned == [True, False]


def test_a_learner_with_no_binned_storage_passes_through_quietly(wide):
    """`btree` has no binned representation at all, which is not a limitation
    of the request -- there is nothing to skip."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = mb.gamboost("y ~ bbs(x, df=4) + btree(few)", data=wide,
                            bins=64, control=mb.boost_control(20, 0.1))
    rows = {row["learner"]: row
            for row in model.binning_report(output="records")}
    assert rows["bbs(x)"]["binned"]
    assert not rows["btree(few)"]["binned"]


# ─── bins="auto" ───────────────────────────────────────────────────


def test_auto_sizes_the_bins_against_the_basis(wide):
    model = mb.gamboost("y ~ bbs(x, df=4, bins='auto')", data=wide,
                        control=mb.boost_control(20, 0.1))
    row = model.binning_report(output="records")[0]
    # 20 knots + degree 3 = 23 basis columns, four bins per column.
    assert row["requested_bins"] == 92
    assert row["ties_collapsed"] == 20000 - 92
    assert "4 bins per basis column" in row["auto_reason"]


def test_auto_declines_to_bin_a_column_with_few_distinct_values(wide):
    """Binning seven values into ninety-two bins collapses nothing, and the
    decision is reported rather than left as a silent no-op."""
    model = mb.gamboost("y ~ bbs(few, df=4, bins='auto')", data=wide,
                        control=mb.boost_control(20, 0.1))
    row = model.binning_report(output="records")[0]
    assert row["requested_bins"] is None
    assert not row["binned"]
    assert "collapse nothing" in row["auto_reason"]


def test_auto_declines_when_there_are_too_few_rows_per_bin():
    from mboostrs.learners import auto_bins

    assert auto_bins(n_rows=200, n_distinct=200, basis_columns=23) is None
    assert auto_bins(n_rows=20000, n_distinct=20000, basis_columns=23) == 92


def test_auto_respects_a_memory_budget():
    from mboostrs.learners import auto_bins

    generous = auto_bins(20000, 20000, 23)
    tight = auto_bins(20000, 20000, 23, memory_budget=8 * 23 * 30)
    assert tight is not None and tight < generous


def test_auto_works_as_a_global_argument(wide):
    model = mb.gamboost("y ~ bbs(x, df=4) + bbs(few, df=4)", data=wide,
                        bins="auto", control=mb.boost_control(20, 0.1))
    rows = {row["learner"]: row
            for row in model.binning_report(output="records")}
    assert rows["bbs(x)"]["requested_bins"] == 92
    assert rows["bbs(few)"]["requested_bins"] is None


def test_auto_saves_memory_against_the_exact_fit(wide):
    binned = mb.gamboost("y ~ bbs(x, df=4, bins='auto')", data=wide,
                         control=mb.boost_control(20, 0.1))
    exact = mb.gamboost("y ~ bbs(x, df=4)", data=wide,
                        control=mb.boost_control(20, 0.1))
    saved = (exact.execution_report().retained_learner_bytes
             - binned.execution_report().retained_learner_bytes)
    assert saved > 0
    assert (binned.execution_report().retained_learner_bytes
            < 0.1 * exact.execution_report().retained_learner_bytes)


def test_an_invalid_bins_string_is_refused():
    with pytest.raises(ValueError, match="positive integer or 'auto'"):
        mb.Bbs("x", bins="lots")


# ─── the approximation check states its own scope ──────────────────


def test_check_binning_reports_the_saving_and_its_scope(wide):
    model = mb.gamboost("y ~ bbs(x, df=4, bins=32)", data=wide,
                        control=mb.boost_control(30, 0.1))
    result = model.check_binning(n=1500, seed=1)
    assert result["bytes_saved"] > 0
    assert result["binned_retained_learner_bytes"] < result[
        "exact_retained_learner_bytes"]
    assert "1500 of 20000 rows" in result["scope"]
    assert "not a property of the bin count" in result["scope"]
    assert set(result) >= {"max_abs_fitted_difference", "relative_to_residual_sd",
                           "first_selection_divergence",
                           "selection_sequence_identical"}


def test_check_binning_refuses_a_model_with_nothing_to_check(curved):
    model = mb.gamboost("y ~ bbs(x, df=4)", data=curved,
                        control=mb.boost_control(10, 0.1))
    with pytest.raises(ValueError, match="no approximately binned learner"):
        model.check_binning()


def test_the_report_lists_linear_terms_without_acting_on_them():
    from mboostrs._extrapolation import ExtrapolationWarning

    """Issue #67: a bols(w) term is reported with its distance, marked
    "linear", while predict() neither warns about it nor clips it. Raw
    categorical new data passes through the fit's recipe first (it used to
    fail as "not numbers")."""
    rng = np.random.default_rng(0)
    n = 200
    d = {"x": rng.uniform(0, 1, n), "w": rng.normal(size=n), "g": rng.choice(list("abc"), n)}
    d["y"] = np.sin(3 * d["x"]) + d["w"] + rng.normal(size=n)
    fit = mb.gamboost("y ~ bbs(x) + bols(w) + bols(g)", data=d, control=mb.boost_control(50))
    span = d["w"].max() - d["w"].min()
    new = {"x": np.array([0.5, 0.6]), "w": np.array([0.0, d["w"].max() + span]),
           "g": np.array(["a", "b"])}
    rows = {row["learner"]: row for row in fit.range_report(new, output="records")}
    assert rows["bbs(x)"]["term"] == "spline basis" and rows["bbs(x)"]["n_outside"] == 0
    assert rows["bols(w)"]["term"] == "linear"
    assert rows["bols(w)"]["worst_distance"] == pytest.approx(1.0, rel=1e-9)
    assert "bols(g)" not in rows
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        clipped = fit.predict(newdata=new, extrapolation="clip")
    np.testing.assert_array_equal(clipped, fit.predict(newdata=new, extrapolation="allow"))


def test_contribution_predictions_follow_the_policy():
    """Issue #67: predict(which=...) and predict(aggregate="none") returned
    without applying extrapolation=, even "error"; partial_effects takes the
    policy too (default "allow")."""
    import polars as pl

    from mboostrs._extrapolation import ExtrapolationWarning

    rng = np.random.default_rng(0)
    n = 200
    d = {"x": rng.uniform(0, 1, n), "w": rng.normal(size=n)}
    d["y"] = np.sin(3 * d["x"]) + d["w"] + rng.normal(size=n)
    fit = mb.gamboost("y ~ bbs(x) + bols(w)", data=d, control=mb.boost_control(50))
    new = {"x": np.array([0.5, 2.0]), "w": np.array([0.0, 0.0])}
    edge = {"x": np.array([0.5, d["x"].max()]), "w": np.array([0.0, 0.0])}
    with pytest.raises(ValueError, match="outside the range"):
        fit.predict(newdata=new, which="x", extrapolation="error")
    with pytest.raises(ValueError, match="outside the range"):
        fit.predict(newdata=new, aggregate="none", extrapolation="error")
    with pytest.raises(ValueError, match="outside the range"):
        fit.partial_effects(which="x", newdata=new, extrapolation="error")
    with pytest.warns(ExtrapolationWarning):
        fit.predict(newdata=new, which="x")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        fit.partial_effects(which="x", newdata=new)
    for frame in (new, pl.DataFrame(new)):
        with pytest.warns(ExtrapolationWarning):
            clipped = fit.predict(newdata=frame, which="x", extrapolation="clip")
        np.testing.assert_allclose(
            np.asarray(clipped), np.asarray(fit.predict(newdata=edge, which="x")))
