"""specification_curve scores the candidates a search touched on one fold partition."""

import json

import altair as alt
import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def panel(seed=3, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=n)
    w = rng.lognormal(0, 1, n)
    y = np.sin(3 * x) + 0.5 * z + 0.2 * np.log1p(w) + rng.normal(0, 0.3, n)
    return {"x": x, "z": z, "w": w, "y": y}


def searched_session(data):
    """Declared: the starting specification and the first comparison. Raised: the second comparison."""
    search = mb.SearchSession(data, "y", family=mb.Gaussian(), score="mse", folds=4,
                              mstop=60, seed=0)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("compare", candidates={"smooth x": "bbs(x, df=4) + bols(z)"})
    search.run("compare", candidates={"log w": "bols(x) + bbs(log1p(w), df=4)"})
    return search


def by_rhs(curve):
    return {row["rhs"]: row for row in curve.records}


def test_session_curve_scores_every_touched_specification_on_one_partition():
    data = panel()
    curve = mb.specification_curve(searched_session(data), data, "y")
    assert isinstance(curve, mb.SpecificationCurve)
    rows = by_rhs(curve)
    assert set(rows) == {"bols(x) + bols(z)", "bbs(x, df=4) + bols(z)",
                         "bols(x) + bbs(log1p(w), df=4)"}
    assert curve.partitions == 1 and curve.folds_per_partition == 4
    assert curve.fits == 3 * 4
    assert sum(row["role"] == "selected" for row in curve.records) == 1
    selected = rows["bols(x) + bols(z)"]
    assert selected["role"] == "selected" and selected["gap vs selected"] == 0.0
    assert selected["gap se"] is None and selected["within two se"] is None
    for row in curve.records:
        if row["role"] == "candidate":
            assert row["gap se"] > 0
            assert row["gap vs selected"] == pytest.approx(
                row["out-of-fold score"] - selected["out-of-fold score"])
    # Records are sorted by out-of-fold score, lowest first.
    scores = [row["out-of-fold score"] for row in curve.records]
    assert scores == sorted(scores)
    assert [row["rank"] for row in curve.records] == [1, 2, 3]


def test_choices_are_flagged_declared_or_raised_by_observed_results():
    data = panel()
    curve = mb.specification_curve(searched_session(data), data, "y")
    rows = by_rhs(curve)
    assert rows["bols(x) + bols(z)"]["origin"] == "declared"
    assert rows["bbs(x, df=4) + bols(z)"]["origin"] == "declared"
    raised = rows["bols(x) + bbs(log1p(w), df=4)"]
    assert raised["origin"] == "raised"
    assert raised["shape origin"] == "raised"
    assert raised["transform"] == "bbs(log1p(w), df=4)"
    assert raised["transform origin"] == "raised"
    assert raised["family origin"] == "declared"
    assert raised["fold scheme origin"] == "declared"
    assert curve.post_hoc is True
    assert curve.label.startswith("post hoc: 1 of 3")
    assert rows["bols(x) + bols(z)"]["transform origin"] == "none"


def test_a_search_with_no_observed_result_before_its_candidates_is_declared():
    data = panel()
    search = mb.SearchSession(data, "y", family=mb.Gaussian(), score="mse", folds=4,
                              mstop=60, seed=0)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("compare", candidates={"smooth x": "bbs(x, df=4) + bols(z)"})
    curve = mb.specification_curve(search, data, "y")
    assert all(row["origin"] == "declared" for row in curve.records)
    assert curve.post_hoc is False
    assert curve.label.startswith("declared before scoring")
    assert curve.joint_test["available"] is True


def test_joint_test_is_a_sign_flip_test_on_the_shared_folds():
    data = panel()
    curve = mb.specification_curve(searched_session(data), data, "y")
    joint = curve.joint_test
    assert joint["available"] is True
    assert joint["folds_reused"] is True and joint["confirmatory"] is False
    assert joint["units"] == 4 and joint["exact"] is True and joint["draws"] == 16
    assert 0 < joint["p_value"] <= 1
    # The statistic is the mean per-fold margin of the selected row over the
    # median row, so it is positive exactly when the median sits above it.
    assert joint["statistic"] == pytest.approx(
        joint["curve_median_risk"] - joint["selected_risk"], abs=0.5)


def test_a_single_specification_has_no_joint_test():
    data = panel()
    search = mb.SearchSession(data, "y", family=mb.Gaussian(), score="mse", folds=4,
                              mstop=60, seed=0)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    curve = mb.specification_curve(search, data, "y")
    assert len(curve.records) == 1
    assert curve.joint_test["available"] is False
    assert "one specification" in curve.joint_test["reason"]


def test_budget_refuses_before_any_specification_is_scored():
    data = panel()
    search = searched_session(data)
    with pytest.raises(ValueError, match=r"needs 12 fits.*budget=5"):
        mb.specification_curve(search, data, "y", budget=5)
    assert mb.specification_curve(search, data, "y", budget=12).fits == 12


def test_transcript_reproduces_the_session_curve_exactly():
    data = panel()
    search = searched_session(data)
    record = json.loads(json.dumps(search.export_transcript(), allow_nan=False))
    from_session = mb.specification_curve(search, data, "y")
    from_record = mb.specification_curve(record, data, "y")
    assert from_record.records == from_session.records
    assert from_record.joint_test == from_session.joint_test
    assert from_record.label == from_session.label


def test_transcript_refuses_other_data_and_an_unreproducible_partition():
    data = panel()
    record = json.loads(json.dumps(searched_session(data).export_transcript(),
                                   allow_nan=False))
    other = {**data, "y": data["y"] + 1.0}
    with pytest.raises(ValueError, match="differ from the rows the transcript"):
        mb.specification_curve(record, other, "y")
    older = {key: value for key, value in record.items() if key != "fold_count"}
    with pytest.raises(ValueError, match="predates"):
        mb.specification_curve(older, data, "y")
    altered = {**record, "family_id": "not-the-family"}
    with pytest.raises(ValueError, match="arguments the transcript does not record"):
        mb.specification_curve(altered, data, "y")


def test_session_refuses_other_data_and_a_session_without_a_selected_specification():
    data = panel()
    with pytest.raises(ValueError, match="differ from the rows the session searched"):
        mb.specification_curve(searched_session(data), {**data, "y": data["y"] * 2}, "y")
    empty = mb.SearchSession(data, "y", family=mb.Gaussian(), score="mse", folds=4, seed=0)
    with pytest.raises(ValueError, match="no selected specification"):
        mb.specification_curve(empty, data, "y")


def test_unknown_sources_are_refused():
    with pytest.raises(TypeError, match="source must be"):
        mb.specification_curve({"not": "a transcript"}, panel(), "y")


def test_families_with_different_losses_are_scored_by_mse_and_the_result_says_so():
    data = panel()
    positive = {**data, "y": np.exp(data["y"] / 2.0)}
    search = mb.SearchSession(positive, "y", family=mb.Gaussian(), score="family",
                              folds=3, mstop=40, seed=0)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("compare", candidates={"smooth x": "bbs(x, df=4) + bols(z)"})
    search.run("set_family", family=mb.GammaReg())
    search.run("compare", candidates={"log w": "bols(x) + bbs(log1p(w), df=4)"})
    curve = mb.specification_curve(search, positive, "y")
    assert curve.score == "mse"
    assert any("mean squared error" in note for note in curve.notes)
    families = {row["family"] for row in curve.records}
    assert families == {"Gaussian", "GammaReg"}
    gamma = next(row for row in curve.records if row["family"] == "GammaReg")
    assert gamma["family origin"] == "raised"


def test_output_selects_the_type_of_rows_and_to_dict_is_plain():
    data = panel()
    search = searched_session(data)
    records = mb.specification_curve(search, data, "y", output="records")
    assert isinstance(records.rows, list) and isinstance(records.rows[0], dict)
    frame = mb.specification_curve(search, data, "y", output="polars")
    assert isinstance(frame.rows, pl.DataFrame)
    assert frame.rows.height == 3
    plain = json.loads(json.dumps(frame.to_dict(), allow_nan=False))
    assert plain["selected"] == plain["rows"][[row["role"] for row in plain["rows"]]
                                               .index("selected")]["specification"]
    assert plain["post_hoc"] is True and plain["fold_scheme"] == frame.fold_scheme


def test_plot_returns_an_altair_chart_of_the_gaps():
    data = panel()
    curve = mb.specification_curve(searched_session(data), data, "y")
    chart = curve.plot()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    assert "layer" in spec and len(spec["layer"]) == 3
    assert "joint test" in spec["title"]["subtitle"]


def test_an_autospec_recommendation_is_a_source():
    data = panel()
    rec = mb.autospec(data, "y")
    curve = mb.specification_curve(rec, data, "y", seed=0)
    selected = next(row for row in curve.records if row["role"] == "selected")
    assert selected["rhs"] == rec.formula.partition("~")[2].strip()
    assert curve.fold_scheme and curve.partitions == 1
    assert any("5-fold" in note for note in curve.notes)


def test_a_discover_result_is_a_source_and_its_draft_is_declared():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200)
    junk = rng.normal(size=200)
    y = np.sin(x) + rng.normal(0, 0.3, 200)
    data = {"x": x, "junk": junk, "y": y}
    result = mb.discover(data, "y", seed=0, interactions=False, spread=False)
    curve = mb.specification_curve(result, data, "y")
    draft = next(entry["candidate"] for entry in result.log if entry["stage"] == "draft")
    rows = by_rhs(curve)
    assert draft in rows and rows[draft]["origin"] == "declared"
    selected = next(row for row in curve.records if row["role"] == "selected")
    assert selected["rhs"] == result.formula
    assert curve.post_hoc is (len(curve.records) > 1)


def test_a_discover_result_with_a_log_scale_driver_is_scored_on_the_joint_score():
    # Heteroscedastic noise: discover adopts a log-scale term, so the curve mixes
    # the Gaussian mean candidates with GaussianLSS candidates.
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    g = rng.integers(0, 2, n).astype(float)
    junk = rng.normal(size=n)
    y = np.sin(x) + 1.5 * z * g + rng.normal(0, 0.5, n) * np.exp(0.5 * x)
    data = {"x": x, "z": z, "g": g, "junk": junk, "y": y}
    result = mb.discover(data, "y", seed=0)
    assert result.spread_formula is not None
    curve = mb.specification_curve(result, data, "y")
    assert curve.score == "family"
    assert any("joint score of GaussianLSS" in note for note in curve.notes)
    assert {row["family"] for row in curve.records} == {"GaussianLSS"}
    selected = next(row for row in curve.records if row["role"] == "selected")
    assert selected["rhs"] == f"mu: {result.formula}; sigma: {result.spread_formula}"
    assert len(curve.records) > 2
