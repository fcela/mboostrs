"""Judgment schemas, measured edits, and provider audit boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import mboostrs as mb
import mboostrs.modeler as modeler


def apply(proposal, spec="bbs(x) + bols(z)", locks=()):
    refusals = []
    candidates, extra = modeler._apply_judgment(
        proposal, spec=spec, response="y", locks=locks,
        forbidden={"y", "treatment"}, columns={"x", "z", "y", "treatment"},
        refusals=refusals, ledger=mb.AssumptionLedger(), round_index=0,
        t=mb.Transcript())
    return candidates, extra, refusals


@pytest.mark.parametrize("proposal", [
    [], None, {"drop": "x"}, {"add": "bbs(x)"}, {"lock": [3]},
    {"replace": []}, {"flags": {}}, {"constraints": []},
    {"constraints": {"monotone": ["x"]}},
    {"constraints": {"monotone": {"x": "up"}}}, {"unexpected": True},
])
def test_malformed_judgment_is_rejected(proposal):
    with pytest.raises(ValueError):
        mb.SpecificationJudgment.from_mapping(proposal)


def test_constraint_becomes_a_candidate_not_an_unmeasured_adoption():
    candidates, _, refusals = apply({"constraints": {"monotone": {"x": "decreasing"}}})
    assert candidates == {"judge constrain x": 'bols(z) + bmono(x, constraint="decreasing")'}
    assert not refusals


def test_constraint_refuses_to_ignore_an_interaction():
    candidates, _, refusals = apply(
        {"constraints": {"monotone": {"x": "increasing"}}},
        spec="bbs(x) + bbs(x) %X% bbs(z)")
    assert not candidates and "interactions" in refusals[0]["reason"]


def test_replacement_preserves_locked_columns():
    assert apply({"replace": {"bbs(x)": "bols(x)"}}, locks=("x",))[0] == {
        "judge replace bbs(x)": "bols(x) + bols(z)"}
    candidates, _, refusals = apply({"replace": {"bbs(x)": "bbs(z)"}}, locks=("x",))
    assert not candidates and "locked" in refusals[0]["reason"]


def test_term_drop_cannot_remove_a_locked_by_column():
    candidates, _, refusals = apply({"drop": ["bols(x, by=z)"]},
                                     spec="bols(x, by=z)", locks=("z",))
    assert not candidates and "locked" in refusals[0]["reason"]


@pytest.mark.parametrize("term", ["bbs(log(y))", "bbs(x) %X% bbs(treatment)", "bols(unknown)"])
def test_forbidden_or_unknown_columns_never_reach_measurement(term):
    candidates, _, refusals = apply({"add": [term]})
    assert not candidates and refusals


def test_tensor_proposal_is_validated_by_both_margins():
    candidates, _, refusals = apply({"add": ["bbs(x) %X% bbs(z)"]})
    assert candidates and not refusals


def test_single_term_drop_returns_offset_only_candidate():
    candidates, _, refusals = apply({"drop": ["x"]}, spec="bbs(x)")
    assert candidates == {"judge drop x": "1"} and not refusals


def client(reply, usage=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: {"choices": [{"message": {"content": reply}}], "usage": usage})))


def test_client_usage_and_invalid_reply_are_audited():
    judge = mb.LLMJudge(client=client("not JSON", {"total_tokens": 37}), max_calls=1)
    with pytest.raises(ValueError, match="JSON"):
        judge({"evidence_id": "abc"})
    assert judge.tokens_used == 37
    assert judge.calls[0]["status"] == "failed"
    assert judge.calls[0]["reply_sha256"]
    assert judge.calls[0]["evidence_id"] == "abc"
    with pytest.raises(RuntimeError, match="budget"):
        judge({})
    assert len(judge.calls) == 1


@pytest.mark.parametrize("usage", [None, {"total_tokens": 100}])
def test_token_budget_stops_when_exhausted_or_unmeasurable(usage):
    judge = mb.LLMJudge(client=client("{}", usage), max_total_tokens=100)
    judge({})
    with pytest.raises(RuntimeError, match="token budget"):
        judge({})
    assert len(judge.calls) == 1


def frame():
    rng = np.random.default_rng(81)
    x = rng.uniform(-1, 1, 80)
    return {"x": x, "z": rng.normal(size=80), "y": x + rng.normal(scale=0.1, size=80)}


def test_monotone_edit_is_measured_then_returned_with_a_fitted_model(monkeypatch):
    import mboostrs.search_session as session

    original_tune = session.tune_mstop
    measured = []

    def compare(specs, data, response, **kwargs):
        measured.append((specs, kwargs))
        return [{"specification": label, "rhs": rhs, "change": -float(i),
                 "out-of-fold risk": 1.0 - i,
                 "change se": 0.1, "cumulative fit count": 5 * (i + 1)}
                for i, (label, rhs) in enumerate(specs.items())]

    def tune(*args, **kwargs):
        kwargs.update(start=20, extensions=0)
        return original_tune(*args, **kwargs)

    monkeypatch.setattr(session, "compare_specifications", compare)
    monkeypatch.setattr(modeler, "suggest_specifications", lambda *a, **k: [])
    monkeypatch.setattr(session, "screen_interactions", lambda *a, **k: [])
    monkeypatch.setattr(session, "tune_mstop", tune)
    result = modeler.modeler_loop(
        frame(), budget=0, mstop=20, max_rounds=1, seed=24,
        judgment=lambda d: mb.SpecificationJudgment(
            evidence_id=d["evidence_id"], monotone={"x": "increasing"}))
    assert len(measured) == 1
    assert measured[0][1]["seed"] == 24
    assert measured[0][1]["nu"] == 0.3
    assert 'bmono(x, constraint="increasing")' in result.spec
    assert result.model is not None
    predicted = result.model.predict({"x": np.linspace(-0.9, 0.9, 40), "z": np.zeros(40)})
    assert np.all(np.diff(predicted) >= -1e-9)
    accept = next(s for s in result.transcript.steps if s["step"] == "accept")
    comparison = next(s for s in result.transcript.steps if s["step"] == "comparison")
    assert accept["comparison_id"] == comparison["comparison_id"]
    assert accept["spec"] == result.spec
    json.dumps(result.to_dict())


@pytest.mark.parametrize("reply", [{"add": "x"}, {"evidence_id": "stale"}, []])
def test_invalid_judgment_does_not_abort_search(reply):
    result = modeler.modeler_loop(frame(), budget=0, mstop=20, max_rounds=1,
                                  judgment=lambda d: reply, tune_start=20)
    assert any(s["step"] == "judgment-failed" for s in result.transcript.steps)
    assert result.model is not None


def test_invalid_locks_fail_before_discovery(monkeypatch):
    monkeypatch.setattr(modeler, "autospec", lambda *a, **k: pytest.fail("should not fit"))
    with pytest.raises(ValueError, match="locks"):
        modeler.modeler_loop(frame(), locks=("y",))


def test_direct_judgment_constructor_validates_direction():
    with pytest.raises(ValueError, match="monotone"):
        mb.SpecificationJudgment(monotone={"x": "sideways"})


def test_polars_input_returns_a_usable_model():
    import polars as pl

    data = pl.DataFrame(frame())
    result = modeler.modeler_loop(data, budget=0, mstop=20, max_rounds=0, tune_start=20)
    prediction = result.model.predict(data.drop("y"))
    assert np.asarray(prediction).shape == (len(data),)


def test_causal_interview_uses_design_prompt_not_predictive_edit_prompt():
    messages = []

    def send(**kwargs):
        messages.extend(kwargs["messages"])
        return {"choices": [{"message": {"content": "{}"}}]}

    supplied = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    mb.elicit_causal_spec(frame(), mb.LLMJudge(client=supplied))
    assert "eliciting a causal analysis design" in messages[0]["content"]
    assert json.loads(messages[1]["content"])["purpose"] == "causal_design"


def test_dual_prescan_reaches_judge_as_proposals():
    seen = []

    def judge(digest):
        seen.append(digest)
        return {"evidence_id": digest["evidence_id"]}

    result = modeler.modeler_loop(frame(), budget=100, mstop=30, max_rounds=1,
                                  preexploration="dual", judgment=judge)
    assert seen and seen[0]["pre_scan"]["engine"] == "dual"
    assert seen[0]["pre_scan"]["fits"] == 2
    assert "proposals only" in seen[0]["pre_scan"]["interpretation"]
    assert result.model is not None


def test_search_benchmark_scores_the_returned_model(monkeypatch):
    from benches import bench_specification_validation as benchmark

    scenario = benchmark.search.SCENARIOS["linear_additive"]
    data, _, _ = scenario.build(np.random.default_rng(31000), 400,
                                np.random.default_rng(31001))
    result = mb.modeler_loop(data, budget=0, max_rounds=0, mstop=30, tune_start=20)
    monkeypatch.setattr(benchmark.search, "agent_loop", lambda *a, **k: result)
    original = mb.gamboost
    controls = []

    def fit(*args, **kwargs):
        controls.append(kwargs["control"])
        return original(*args, **kwargs)

    monkeypatch.setattr(mb, "gamboost", fit)
    row = benchmark.run("linear_additive", 0, 0, 0)
    # Only the initial baseline is refitted. The final score uses the exact
    # model returned by the search, so a different learning rate cannot
    # silently change the measured procedure.
    assert len(controls) == 1
    assert controls[0].nu == result.transcript.steps[0]["nu"] == 0.3
    assert row["final_mstop"] == result.mstop
