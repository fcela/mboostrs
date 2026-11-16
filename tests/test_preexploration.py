"""Preliminary exploration proposes comparisons and shares its fit across probes."""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._autospec import _preexploration
from mboostrs._frame import read_column


def line_data():
    x, noise = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    return {"x": x.ravel(), "noise": noise.ravel(), "y": 3 * x.ravel()}


def test_exploration_is_reused_for_shapes_and_pairs(monkeypatch):
    import mboostrs._autospec as internals

    calls = []
    real = internals._preexploration

    def recorded(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(internals, "_preexploration", recorded)
    rec = mb.autospec(line_data(), "y", family=mb.Gaussian(), budget=100,
                      deselect=False, seed=17)
    assert len(calls) == 1
    assert rec.evidence["preexploration"]["fits"] == 1
    assert rec.evidence["preexploration"]["seed"] == 17
    shapes = {row.column: row for row in rec.evidence["shapes"]}
    assert {s.shape for s in shapes["x"].scores} == {"bbs", "bols"}
    assert {s.shape for s in shapes["noise"].scores} == {"bbs", "bols", "btree"}
    # Every retained candidate still has real fold risks, including the incumbent.
    assert all(len(s.fold_risk) == 5 for s in shapes["x"].scores)
    assert "held-out comparison" in rec.report()


@pytest.mark.parametrize("kwargs", [{"budget": 0}, {"budget": 2},
                                     {"budget": 100, "preexploration": "off"}])
def test_disabled_or_unaffordable_exploration_does_not_fit(monkeypatch, kwargs):
    import mboostrs._autospec as internals

    def forbidden(*args, **kwargs):
        pytest.fail("pre-exploration must not fit")

    monkeypatch.setattr(internals, "_preexploration", forbidden)
    rec = mb.autospec(line_data(), "y", family=mb.Gaussian(), **kwargs)
    assert "preexploration" not in rec.evidence
    if kwargs["budget"] == 0:
        assert rec.fits == 0
    if kwargs.get("preexploration") == "off":
        x = next(r for r in rec.evidence["shapes"] if r.column == "x")
        assert {s.shape for s in x.scores} == {"bbs", "bols", "btree"}


def test_declared_constraint_retains_all_cost_comparisons():
    rec = mb.autospec(line_data(), "y", family=mb.Gaussian(), monotone={"x": "increasing"},
                      budget=100, deselect=False)
    x = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert x.chosen == x.incumbent and "bmono" in x.incumbent
    assert {"bbs", "bols", "btree"}.issubset({s.shape for s in x.scores})


def test_categorical_codes_and_missing_values_reach_both_engines():
    data = line_data()
    data["group"] = np.where(data["noise"] > 0, "a", "b")
    data["x"][0] = np.nan
    columns = [read_column(data, name) for name in ("x", "noise", "group")]
    native = _preexploration(data, "y", columns, mstop=60)
    assert native["columns"][2]["column"] == "group"
    assert native["columns"][2]["reason"] == "ambiguous"
    try:
        optional = _preexploration(data, "y", columns, engine="hessboost", mstop=60)
    except (ValueError, RuntimeError) as exc:
        assert "hessboost Cargo feature" in str(exc)
    else:
        assert optional["engine"] == "hessboost"
        assert optional["priority"][0] == "x"
        assert optional["training_r2"] > 0.95


def test_invalid_engine_is_rejected_even_without_a_budget():
    with pytest.raises(ValueError, match="preexploration"):
        mb.autospec(line_data(), "y", preexploration="unknown")


@pytest.mark.parametrize("engine,cost", [("neural", 1), ("dual", 2)])
def test_neural_engines_restore_names_and_account_fits(engine, cost):
    data = line_data()
    data["y"] = data["x"] * data["noise"]
    columns = [read_column(data, name) for name in ("x", "noise")]
    result = _preexploration(data, "y", columns, engine=engine, seed=9, mstop=150)
    assert result["fits"] == cost and result["engine"] == engine
    assert set(result["pairs"][0][:2]) == {"x", "noise"}
    assert result["pairs"][0][2] > 0.7
    if engine == "dual":
        assert result["pair_sources"][0]["columns"] == ["x", "noise"]
        assert "neural" in result["pair_sources"][0]["sources"]


def test_dual_exploration_is_charged_once_and_reused(monkeypatch):
    import mboostrs._autospec as internals

    calls = []
    real = internals._preexploration

    def recorded(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(internals, "_preexploration", recorded)
    rec = mb.autospec(line_data(), "y", family=mb.Gaussian(), budget=100,
                      preexploration="dual", deselect=False)
    assert len(calls) == 1 and calls[0]["fits"] == 2
    assert rec.evidence["preexploration"]["fits"] == 2
    assert "2 probe fit(s)" in rec.report()


def test_neural_hints_do_not_prune_shapes():
    rec = mb.autospec(line_data(), "y", family=mb.Gaussian(), budget=100,
                      preexploration="neural", deselect=False)
    assert not rec.evidence["preexploration"]["shape_pruning"]
    assert {s.shape for s in rec.evidence["shapes"][0].scores} == {"bbs", "bols", "btree"}


def test_wide_exploration_caps_partial_dependence_without_discarding_shapes():
    import json

    from mboostrs._rust import preexplore_json

    x = np.random.default_rng(3).normal(size=(40, 34))
    result = json.loads(preexplore_json(x, np.ascontiguousarray(3 * x[:, 0]),
                                       [0] * 34, [0] * 34, rounds=5, pairs=False))
    untouched = [row for row in result["columns"] if row["partial"] is None]
    assert len(untouched) == 2
    assert all(row["shapes"] == ["bols", "bbs", "btree"] for row in untouched)
    assert result["pairs"] == []


def test_response_scale_hints_do_not_prune_log_link_shapes(monkeypatch):
    import mboostrs._autospec as internals

    def misleading(*args, **kwargs):
        return {"engine": "native", "fits": 1, "priority": ["x"], "pairs": [],
                "columns": [{"column": "x", "shapes": ["bols"], "reason": "line"}]}

    monkeypatch.setattr(internals, "_preexploration", misleading)
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 300)
    rec = mb.autospec({"x": x, "y": rng.poisson(np.exp(1 + x))}, "y",
                      family=mb.Poisson(), budget=100, step="fixed")
    assert not rec.evidence["preexploration"]["shape_pruning"]
    assert {s.shape for s in rec.evidence["shapes"][0].scores} == {"bbs", "bols", "btree"}
    assert "shapes are not pruned" in rec.report()


@pytest.mark.parametrize("budget", [20, 40, 100])
def test_pair_adoption_never_bypasses_validation_when_budget_is_tight(monkeypatch, budget):
    import importlib

    from mboostrs._autospec import InteractionLead

    module = importlib.import_module("mboostrs.autospec")
    term = "bols(x, by=noise)"
    lead = InteractionLead(left="x", right="noise", removed=10.0, term=term, kind="varying")
    monkeypatch.setattr(module, "probe_interactions", lambda *a, **kw: ((lead,), 0, 0))
    validated = []
    real = module._held_out_pairs

    def record(*args, **kwargs):
        validated.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_held_out_pairs", record)
    rec = mb.autospec(line_data(), "y", family=mb.Gaussian(), budget=budget,
                      preexploration="off", interactions="adopt", deselect=False)
    if not validated:
        assert term not in rec.formula
        assert not rec.evidence.get("adopted_interactions")
    if rec.evidence.get("unvalidated_interactions"):
        assert "suggested only" in rec.report()
