"""Whole-search admission, failure accounting and usable budget fallbacks."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _workflow
from mboostrs._workflow import BudgetExhausted, Workflow, call_fit


def data():
    rng = np.random.default_rng(74)
    x = rng.normal(size=90)
    return {"x": x, "z": rng.normal(size=90), "y": 1.2 * x + rng.normal(size=90)}


@pytest.mark.parametrize("limit", [1, 7, 30])
def test_modeler_respects_total_fit_budget_and_returns_predictions(limit):
    events = []
    result = mb.modeler_loop(data(), budget=30, mstop=20, max_rounds=2, tune_start=20,
                             workflow_budget=mb.CostBudget(fits=limit), progress=events.append)
    assert result.stop_reason == "budget_fits"
    assert 1 <= result.fits <= limit
    assert result.fits == sum(e["fit_charges"] for e in events)
    assert np.isfinite(result.model.predict()).all()
    assert result.budget_report["fit_charges"] == result.fits
    assert all(e["status"] != "running" for e in result.budget_report["operations"])
    assert result.transcript.steps[-1]["step"] == "workflow-budget"


def test_large_budget_preserves_the_unbudgeted_model():
    options = dict(budget=0, mstop=20, max_rounds=0, tune_start=20, seed=6)
    reference = mb.modeler_loop(data(), **options)
    bounded = mb.modeler_loop(data(), **options, workflow_budget=mb.CostBudget(fits=500))
    assert bounded.spec == reference.spec and bounded.mstop == reference.mstop
    assert bounded.stop_reason == reference.stop_reason
    np.testing.assert_array_equal(bounded.model.predict(), reference.model.predict())


@pytest.mark.parametrize("fit_fn,formula", [(mb.gamboost, "y ~ bbs(x)"), (mb.glmboost, "y ~ x")])
def test_native_cv_batch_is_admitted_as_a_whole(fit_fn, formula):
    with Workflow(mb.CostBudget(fits=5)) as workflow:
        fit = fit_fn(formula, data(), control=mb.boost_control(10))
        with pytest.raises(BudgetExhausted) as exc:
            fit.cvrisk(B=5, type="kfold")
        assert exc.value.reason == "fits"
        assert workflow.charges == 1
        assert len(workflow.events) == 1  # no fold was dispatched


def test_admitted_cv_batch_counts_its_folds():
    with Workflow(mb.CostBudget(fits=6)) as workflow:
        fit = mb.gamboost("y ~ bols(x)", data(), control=mb.boost_control(10))
        fit.cvrisk(B=5, type="kfold")
        assert workflow.charges == 6
        assert [e["fit_charges"] for e in workflow.events] == [1, 5]


def test_failed_batch_keeps_its_charge_and_reports_uncertainty():
    def failed_batch():
        raise ValueError("one fold failed")

    with Workflow(mb.CostBudget(fits=4)) as workflow:
        with pytest.raises(ValueError, match="fold failed"):
            call_fit(failed_batch, _fit_count=4)
        assert workflow.charges == 4
        assert workflow.events[0]["status"] == "failed"
        assert workflow.events[0]["actual_completed_fits"] is None
        with pytest.raises(BudgetExhausted):
            call_fit(lambda: pytest.fail("must not run"))


def test_context_is_restored_after_exception():
    with pytest.raises(ValueError):
        with Workflow(mb.CostBudget(fits=1)) as workflow:
            call_fit(lambda: (_ for _ in ()).throw(ValueError("failed")))
    mb.glmboost("y ~ x", data(), control=mb.boost_control(10))
    assert workflow.charges == 1


def test_thread_scopes_do_not_share_fit_counters():
    def run(limit):
        with Workflow(mb.CostBudget(fits=limit)) as workflow:
            for _ in range(limit):
                mb.glmboost("y ~ x", data(), control=mb.boost_control(5))
            return workflow.charges

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [1, 3])) == [1, 3]


def test_time_exhaustion_keeps_completed_fallback(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_workflow.time, "perf_counter", lambda: clock[0])

    def after_fit(event):
        clock[0] = 2.0

    result = mb.modeler_loop(data(), mstop=20, workflow_budget=mb.CostBudget(seconds=1),
                             progress=after_fit)
    assert result.stop_reason == "budget_seconds"
    assert result.fits == 1 and np.isfinite(result.model.predict()).all()
    assert result.budget_report["elapsed_seconds"] == 2.0


def test_memory_exhaustion_keeps_completed_fallback(monkeypatch):
    peak = [50.0]
    monkeypatch.setattr(_workflow, "_peak_memory_mb", lambda: peak[0])
    result = mb.modeler_loop(data(), mstop=20, workflow_budget=mb.CostBudget(memory_mb=100),
                             progress=lambda event: peak.__setitem__(0, 150.0))
    assert result.stop_reason == "budget_memory_mb" and result.fits == 1
    assert result.budget_report["peak_memory_mb"] == 150.0


def test_unaffordable_initial_model_is_a_structured_error(monkeypatch):
    monkeypatch.setattr(_workflow, "_peak_memory_mb", lambda: 150.0)
    with pytest.raises(BudgetExhausted) as exc:
        mb.modeler_loop(data(), workflow_budget=mb.CostBudget(memory_mb=100))
    assert exc.value.code == "workflow_budget"
    assert exc.value.report["fit_charges"] == 0


def test_dual_prescan_counts_both_native_fits():
    from mboostrs._autospec import _preexploration
    from mboostrs._frame import read_column

    frame = data()
    columns = [read_column(frame, name) for name in ("x", "z")]
    with Workflow(mb.CostBudget(fits=2)) as workflow:
        _preexploration(frame, "y", columns, engine="dual", mstop=10)
        assert workflow.charges == 2
    with Workflow(mb.CostBudget(fits=1)) as workflow:
        with pytest.raises(BudgetExhausted):
            _preexploration(frame, "y", columns, engine="dual", mstop=10)
        assert workflow.charges == 0


def test_reml_refit_is_charged():
    with Workflow(mb.CostBudget(fits=2)) as workflow:
        fit = mb.gamboost("y ~ bols(x)", data(), control=mb.boost_control(10))
        mb.gam_refit(fit)
        assert workflow.charges == 2
        assert workflow.events[-1]["operation"] == "gam_refit_core"


def test_joint_lss_fit_and_cv_are_charged():
    with Workflow(mb.CostBudget(fits=4)) as workflow:
        fit = mb.gamboostLSS({"mu": "y ~ bols(x)", "sigma": "y ~ bols(z)"}, data(),
                             families=mb.GaussianLSS(), control=mb.boost_control(5))
        fit.cvrisk(folds=mb.cv_weights(90, type="kfold", B=3), grid=[(2, 2), (5, 5)])
        assert workflow.charges == 4


@pytest.mark.parametrize("limit", [1, 7, 30])
def test_policy_budget_returns_an_honestly_unscored_fallback(limit):
    import json

    result = mb.auto_policy(data(), "y", mstop=20, budget=mb.CostBudget(fits=limit))
    assert result.fits <= limit
    assert np.isfinite(result.model.predict()).all()
    record = result.to_dict()
    if result.score is None:
        assert record["evaluation_status"] == "not_evaluated"
        assert not result.trail
    assert record["budget_report"]["fit_charges"] == result.fits
    json.dumps(record, allow_nan=False)


def test_policy_fallback_keeps_declared_groups():
    frame = data()
    frame["g"] = np.tile(["a", "b", "c"], 30)
    result = mb.auto_policy(frame, "y", groups=("g",), mstop=20,
                            budget=mb.CostBudget(fits=1))
    assert "brandom(g)" in result.spec
    assert np.isfinite(result.model.predict()).all()


def test_cv_failure_reports_the_request_that_could_not_be_admitted():
    with Workflow(mb.CostBudget(fits=2)) as workflow:
        fit = mb.gamboost("y ~ bols(x)", data(), control=mb.boost_control(10))
        with pytest.raises(BudgetExhausted) as error:
            fit.cvrisk(B=5, type="kfold")
        assert error.value.requested_fit_charges == 5
        assert workflow.report()["blocked_request"] == {
            "operation": "cvrisk_gamboost_specs", "fit_charges": 5, "reason": "fits"}


def test_no_judge_request_after_fit_budget_exhaustion():
    result = mb.modeler_loop(data(), budget=20, mstop=20,
                             workflow_budget=mb.CostBudget(fits=1),
                             judgment=lambda digest: pytest.fail("budget is already spent"))
    assert result.judgment_calls == 0


def test_causal_budget_stop_carries_design_without_claiming_an_estimate():
    frame = data()
    frame["a"] = np.tile([0., 1.], 45)
    spec = mb.CausalSpec(treatment="a", outcome="y", covariates=("x",),
                         time_zero="baseline", follow_up="one month",
                         adjustment_rationale="x is the declared pre-treatment adjustment set")
    result = mb.modeler_loop(frame, causal=spec, mstop=20,
                             workflow_budget=mb.CostBudget(fits=1))
    assert result.causal_result is None and result.ledger.entries
    assert "a" not in result.spec
    assert result.stop_reason == "budget_fits"


def test_policy_progress_reports_actual_fit_charges():
    seen = []
    result = mb.auto_policy(data(), "y", mstop=20, budget=mb.CostBudget(fits=30),
                            progress=lambda stage, spent: seen.append((stage, spent.fits)))
    assert seen[-1] == ("workflow", result.fits)
    assert all(fits <= 30 for _, fits in seen)


def test_judge_sees_bounded_actual_resource_evidence():
    seen = []
    result = mb.modeler_loop(data(), budget=0, mstop=10, max_rounds=1,
                             workflow_budget=mb.CostBudget(fits=200),
                             judgment=lambda digest: seen.append(digest) or {})
    assert seen
    resources = seen[0]["workflow_budget"]
    assert resources["fit_charges"] == seen[0]["cumulative_fits"]
    assert resources["remaining_fit_charges"] == 200 - resources["fit_charges"]
    assert "operations" not in resources
    assert result.fits <= 200


def test_judge_is_skipped_when_remaining_fits_cannot_price_a_proposal():
    seen = []
    options = dict(budget=0, mstop=10, max_rounds=1, tune_start=10)
    mb.modeler_loop(data(), **options, workflow_budget=mb.CostBudget(fits=200),
                    judgment=lambda digest: seen.append(digest) or {})
    limit = seen[0]["workflow_budget"]["fit_charges"] + 1
    result = mb.modeler_loop(
        data(), **options, workflow_budget=mb.CostBudget(fits=limit),
        judgment=lambda digest: pytest.fail("one charge cannot price a five-fold comparison"))
    assert result.judgment_calls == 0 and result.fits < limit
    assert result.budget_report["blocked_request"]["operation"] == "judgment_followup_comparison"
    assert result.budget_report["blocked_request"]["fit_charges"] == 10
