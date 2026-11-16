"""The modeler loop: deterministic search, judgment seam, causal lane."""

from __future__ import annotations

import types

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.modeler import LLMJudge, _extract_json, modeler_loop


def _predictive_frame(n=150, seed=4):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = rng.uniform(-1, 1, n)
    y = 3.0 + np.sin(x1) + 0.8 * x2 + 0.0 * x3 + rng.normal(scale=0.3, size=n)
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y}


def _causal_frame(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 4, n)
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * x1)))).astype(float)
    y = 1.5 * a + np.sin(x2) + 0.7 * x1 + rng.normal(scale=0.4, size=n)
    mediator = 0.9 * a + rng.normal(scale=0.2, size=n)
    x3 = rng.normal(size=n)
    return {"x1": x1, "x2": x2, "x3": x3, "a": a, "mediator": mediator, "y": y}


def _causal_spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x1", "x2"),
        time_zero="admission", follow_up="30-day readmission",
        adjustment_rationale="x1 and x2 are measured at admission and cover "
        "the confounders the design names; nothing else pre-treatment was "
        "recorded.",
        post_treatment=("mediator",),
    )


class _StubClient:
    """An OpenAI-compatible client serving scripted replies, no network."""

    def __init__(self, replies):
        completions = types.SimpleNamespace(
            create=lambda **kwargs: {"choices": [
                {"message": {"content": replies.pop(0)}}]})
        self.chat = types.SimpleNamespace(completions=completions)


def _steps(result, step):
    return [s for s in result.transcript.steps if s["step"] == step]


def test_deterministic_loop_records_every_step():
    result = modeler_loop(_predictive_frame(), budget=30, max_rounds=2,
                          tune_start=100)
    assert result.judgment_calls == 0
    assert not _steps(result, "judgment")
    kinds = [s["step"] for s in result.transcript.steps]
    assert kinds[0] == "autospec"
    assert kinds[-1] == "final"
    assert result.fits > 0
    assert result.mstop >= 1
    mb.formula.parse_gamboost_formula(f"y ~ {result.spec}")
    assert {item.action for item in result.action_evidence} >= {
        "set_spec", "fit", "term_report", "interaction_screen", "compare", "tune"}
    assert result.to_dict()["action_evidence"]


@pytest.mark.parametrize("limited", [False, True])
def test_declared_family_reaches_search_and_budget_fallback(limited):
    frame = _predictive_frame(n=60)
    frame["y"] = np.round(frame["y"] + 8)
    result = modeler_loop(
        frame, family=mb.Gaussian(), budget=0, max_rounds=0, mstop=10,
        tune_start=10, workflow_budget=mb.CostBudget(fits=1) if limited else None)
    assert isinstance(result.family, mb.Gaussian)
    assert np.isfinite(result.model.predict()).all()
    if limited:
        assert result.stop_reason == "budget_fits"
    else:
        assert _steps(result, "autospec")[0]["family"] == "Gaussian"


def test_bench_agent_loop_delegates_to_modeler_loop():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root / "benches") not in sys.path:
        sys.path.insert(0, str(root / "benches"))
    from bench_agent_search import agent_loop

    spec, family, mstop, groups, transcript = agent_loop(_predictive_frame(120))
    assert spec and mstop >= 1
    assert transcript.steps[0]["step"] == "autospec"


def test_judge_proposal_is_measured_and_audited():
    reply = (
        "```json\n"
        '{"reasoning": "x3 looks like noise", "drop": ["x3"], "add": [],'
        ' "lock": [], "constraints": {}, "flags": [], "assumptions": [],'
        ' "estimand_note": "predictive"}\n'
        "```"
    )
    judge = LLMJudge(client=_StubClient([reply] * 3), retain_text=True)
    result = modeler_loop(_predictive_frame(), judgment=judge, budget=30,
                          max_rounds=2, tune_start=100)
    assert result.judgment_calls >= 1
    step = _steps(result, "judgment")[0]
    assert step["drop"] == 1 and "judgment_sha256" in step
    assert step["candidates_added"] == 1
    assert all("locked" not in r["reason"] for r in result.refusals
               if "x3" in r["change"])
    from mboostrs.modeler import _columns_of

    assert _steps(result, "accept")
    assert "x3" not in _columns_of(result.spec)


def test_judge_failure_is_logged_and_loop_survives():
    judge = LLMJudge(client=_StubClient(["I cannot answer that."]))
    result = modeler_loop(_predictive_frame(), judgment=judge, budget=30,
                          max_rounds=1, tune_start=100)
    failed = _steps(result, "judgment-failed")
    assert failed and "no JSON" in failed[0]["error"]
    assert result.spec


def test_causal_lane_locks_adjustment_and_excludes_treatment():
    reply = (
        '{"reasoning": "trim", "drop": ["x1", "x3"], "add": ["bols(a)"],'
        ' "lock": [], "constraints": {}, "flags": [{"kind": "leakage",'
        ' "term": "mediator", "note": "post-treatment"}],'
        ' "assumptions": [{"statement": "no interference between units",'
        ' "confidence": "medium"}], "estimand_note": "ate"}'
    )
    judge = LLMJudge(client=_StubClient([reply] * 3))
    result = modeler_loop(_causal_frame(), judgment=judge,
                          causal=_causal_spec(), budget=30, max_rounds=2,
                          tune_start=100)
    refused = {r["change"] for r in result.refusals}
    assert "drop x1" in refused
    assert "add bols(a)" in refused
    from mboostrs.modeler import _columns_of

    columns = _columns_of(result.spec)
    assert "x1" in columns and "x2" in columns
    assert "a" not in columns and "mediator" not in columns
    provenance = {e["provenance"] for e in result.ledger.to_list()}
    assert "modeler-stated" in provenance
    assert "llm-proposed-unconfirmed" in provenance
    assert any(e["statement"] == "no interference between units"
               for e in result.ledger.to_list())
    assert result.causal_result is not None
    assert np.isfinite(result.causal_result.estimate)
    kinds = [s["step"] for s in result.transcript.steps]
    assert kinds[0] == "causal-design" and "causal-result" in kinds
    assert _steps(result, "judgment-flag")


def test_extra_locks_are_enforced():
    result = modeler_loop(_predictive_frame(), locks=("x4",), budget=30,
                          max_rounds=2, tune_start=100)
    from mboostrs.modeler import _columns_of

    assert "x4" in _columns_of(result.spec)


def test_ledger_provenance_rules():
    ledger = mb.AssumptionLedger()
    entry = ledger.add("exchangeability holds", "llm-proposed-unconfirmed")
    assert ledger.confirm("exchangeability holds") is entry
    with pytest.raises(KeyError):
        ledger.confirm("never proposed")
    with pytest.raises(ValueError):
        ledger.add("", "unresolved")
    with pytest.raises(ValueError):
        ledger.add("x", "invented-by-nobody")


def test_llm_judge_needs_key_or_client(monkeypatch):
    for name in ("MBOOSTRS_LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    judge = mb.LLMJudge()
    with pytest.raises(ValueError, match="API key"):
        judge({"round": 0})


def test_extract_json_tolerates_fences_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        _extract_json("no object here")


def _elicit_reply(overrides=None, **patch):
    proposal = {
        "treatment": "a", "outcome": "y", "time_zero": "admission",
        "follow_up": "30-day readmission", "covariates": ["x1", "x2"],
        "post_treatment": ["mediator"], "adjustment_rationale":
        "x1 and x2 are measured at admission; the design names no other "
        "pre-treatment confounder.",
        "questions": ["was x3 measured before treatment?"],
        "unresolved": ["whether units interfere"],
    }
    proposal.update(overrides or {})
    import json
    return json.dumps(proposal)


def test_elicitation_drafts_a_confirmable_spec():
    judge = mb.LLMJudge(client=_StubClient([_elicit_reply()]))
    draft = mb.elicit_causal_spec(_causal_frame(), judge,
                                  domain_context="hospital cohort")
    assert draft.spec is not None and not draft.problems
    assert draft.spec.treatment == "a"
    assert draft.questions == ["was x3 measured before treatment?"]
    assert draft.unresolved == ["whether units interfere"]
    entry = draft.ledger.entries[0]
    assert entry.provenance == "llm-proposed-unconfirmed"
    draft.ledger.confirm(entry.statement)
    assert draft.ledger.entries[0].provenance == "llm-proposed-confirmed"


def test_elicitation_refuses_columns_outside_the_frame():
    judge = mb.LLMJudge(client=_StubClient([
        _elicit_reply({"covariates": ["x1", "not_a_column"]})]))
    draft = mb.elicit_causal_spec(_causal_frame(), judge)
    assert draft.spec is None
    assert any("absent from the frame" in p for p in draft.problems)


def test_elicitation_refuses_a_non_binary_treatment():
    judge = mb.LLMJudge(client=_StubClient([_elicit_reply({"treatment": "x1"})]))
    draft = mb.elicit_causal_spec(_causal_frame(), judge)
    assert draft.spec is None
    assert any("binary treatment" in p for p in draft.problems)


def test_elicitation_requires_a_rationale():
    judge = mb.LLMJudge(client=_StubClient([
        _elicit_reply({"adjustment_rationale": ""})]))
    draft = mb.elicit_causal_spec(_causal_frame(), judge)
    assert draft.spec is None
    assert any("adjustment_rationale" in p for p in draft.problems)


def test_elicitation_survives_a_failing_judge():
    judge = mb.LLMJudge(client=_StubClient(["no JSON here"]))
    draft = mb.elicit_causal_spec(_causal_frame(), judge)
    assert draft.spec is None and draft.problems


def test_modeler_loop_refuses_an_unidentified_causal_design():
    frame = _causal_frame()
    frame["a"] = np.zeros(len(frame["a"]))
    result = modeler_loop(frame, judgment=None,
                          causal=_causal_spec(), budget=30, max_rounds=1,
                          tune_start=100)
    assert result.causal_result is None
    assert result.trap_report is not None
    assert "no_treatment_variation" in {f.name for f in result.trap_report.blocking}
    assert any(r.get("severity") == "blocking" for r in result.refusals)
    assert "causal-traps" in [e["step"] for e in result.transcript.steps]


def test_modeler_loop_records_advisory_traps_alongside_the_estimate():
    result = modeler_loop(_causal_frame(), judgment=None,
                          causal=_causal_spec(), budget=30, max_rounds=1,
                          tune_start=100)
    assert result.causal_result is not None
    assert result.trap_report is not None
    # A numeric frame carries no timestamp, so the timing claim stays
    # unverified: the estimate ships with that recorded.
    assert {f.name for f in result.trap_report.advisory} == {"timing_unverifiable"}
