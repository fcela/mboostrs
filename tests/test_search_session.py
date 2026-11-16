"""Agent-directed search must keep its fold, budget and test boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import mboostrs as mb


def sample(seed=13, groups=6, rows_per_group=18):
    rng = np.random.default_rng(seed)
    n = groups * rows_per_group
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=n)
    group = np.repeat([f"state_{i}" for i in range(groups)], rows_per_group)
    y = 4*x*x + .6*z + rng.normal(0, .12, n)
    return {"x": x, "z": z, "state": group, "y": y}


def test_agent_chooses_actions_and_replays_grouped_search_after_resume():
    train = sample()
    search = mb.open_search(
        train, "y", family=mb.Gaussian(), groups="state", folds=3,
        mstop=40, budget=mb.CostBudget(fits=80), locks=("x",))
    assert search.folds.check_leakage()["clean"]
    assert search.state()["available_actions"]["classify"]["fit_charges"] == "0"
    assert search.run("classify").fit_charges == 0
    search.run("hypothesis", variable="x", role="measurement",
               statement="The input may have curvature.",
               would_count_against="A spline increases held-out MSE.")
    search.run("set_spec", rhs="bols(x) + bols(z) + brandom(state)")
    compared = search.run("compare", candidates={
        "curve": "bbs(x) + bols(z) + brandom(state)"})
    assert compared.status == "measured"
    assert compared.value[1]["change"] < 0
    assert compared.fit_charges == 6

    saved = json.loads(json.dumps(search.checkpoint(), allow_nan=False))
    resumed = mb.SearchSession.resume(train, saved, family=mb.Gaussian())
    assert resumed.state()["state_id"] == search.state()["state_id"]
    assert resumed.state()["budget"]["fit_charges"] == 6
    assert resumed.folds.check_leakage()["clean"]
    chosen = resumed.run("adopt", comparison_id=compared.evidence_id, candidate="curve")
    assert chosen.summary["spec"].startswith("bbs(x)")
    fitted = resumed.run("fit")
    assert fitted.fit_charges == 1
    assert resumed.run("interaction_screen").status == "screening_only"
    resumed.run("freeze")
    test = sample(seed=99, groups=2, rows_per_group=12)
    scored = resumed.run("final_evaluate", data=test)
    assert np.isfinite(scored.summary["score"])
    assert scored.summary["n_test"] == len(test["y"])
    with pytest.raises(ValueError, match="unavailable"):
        resumed.run("final_evaluate", data=test)


def test_session_refuses_stale_comparison_and_preserves_fitted_fallback_on_budget():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2,
                            budget=mb.CostBudget(fits=1), mstop=12)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("fit")
    candidate = search.run("compare", candidates={"smooth": "bbs(x) + bols(z)"})
    assert candidate.status == "refused"
    assert candidate.summary["blocked_request"]["reason"] == "fits"
    assert search.state()["budget"]["fit_charges"] == 1
    assert search.model is not None
    search.run("freeze")

    separate = mb.open_search(data, "y", family=mb.Gaussian(), folds=2, mstop=12)
    separate.run("set_spec", rhs="bols(x) + bols(z)")
    measured = separate.run("compare", candidates={"smooth": "bbs(x) + bols(z)"})
    separate.run("set_spec", rhs="bols(x)")
    with pytest.raises(ValueError, match="stale"):
        separate.run("adopt", comparison_id=measured.evidence_id, candidate="smooth")


def test_session_rejects_mutated_training_data_and_forged_resume():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2)
    saved = search.checkpoint()
    changed = {**data, "y": data["y"].copy()}
    changed["y"][0] += 1
    with pytest.raises(ValueError, match="do not match"):
        mb.SearchSession.resume(changed, saved, family=mb.Gaussian())
    data["y"][0] += 1
    with pytest.raises(ValueError, match="changed after"):
        search.run("classify")


def test_checkpoint_rejects_changed_comparison_or_charge():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2, mstop=12)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("compare", candidates={"smooth": "bbs(x) + bols(z)"})
    saved = search.checkpoint()
    changed = json.loads(json.dumps(saved))
    changed["evidence"][-1]["comparison_rows"][1]["change"] = -1000.0
    with pytest.raises(ValueError, match="differs from its evidence"):
        mb.SearchSession.resume(data, changed, family=mb.Gaussian())
    changed = json.loads(json.dumps(saved))
    changed["budget_report"]["fit_charges"] += 1
    with pytest.raises(ValueError, match="fit charges"):
        mb.SearchSession.resume(data, changed, family=mb.Gaussian())
    changed = json.loads(json.dumps(saved))
    changed["budget_report"]["budget"]["fits"] = 999
    with pytest.raises(ValueError, match="identity chain"):
        mb.SearchSession.resume(data, changed, family=mb.Gaussian())


def test_agent_can_choose_different_next_tool_after_the_same_comparison():
    choices = []
    for curved in (False, True):
        rng = np.random.default_rng(8)
        x = rng.uniform(-1, 1, 120)
        y = (3*x*x if curved else 3*x) + rng.normal(0, .1, len(x))
        data = {"x": x, "y": y}
        search = mb.open_search(data, "y", family=mb.Gaussian(),
                                folds=4, mstop=60, budget=mb.CostBudget(fits=100))
        search.run("set_spec", rhs="bols(x)")
        compared = search.run("compare", candidates={"smooth": "bbs(x)"})
        row = compared.value[1]
        if row["change"] < -2*row["change se"]:
            search.run("adopt", comparison_id=compared.evidence_id, candidate="smooth")
            next_evidence = search.run("tune", start=60)
        else:
            next_evidence = search.run("term_report", shapes=False)
        choices.append(next_evidence.action)
        assert search.state()["fold_id"] == compared.summary["fold_id"]
    assert choices == ["term_report", "tune"]


def test_dual_prescan_is_charged_as_screening_evidence():
    rng = np.random.default_rng(3)
    x, z = rng.normal(size=(2, 80))
    data = {"x": x, "z": z, "y": x*z + rng.normal(0, .1, 80)}
    search = mb.open_search(data, "y", family=mb.Gaussian(),
                            budget=mb.CostBudget(fits=2))
    result = search.run("pre_scan", engine="dual", rounds=100)
    assert result.status == "screening_only"
    assert result.fit_charges == 2
    assert search.spec is None
    assert search.state()["budget"]["fit_charges"] == 2


def test_pair_candidates_accepts_true_or_a_prescan_report():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2,
                            mstop=25, budget=mb.CostBudget(fits=20))
    prescan = search.run("pre_scan", engine="native", rounds=40)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("fit")
    fresh = search.run("pair_candidates", surrogate=True)
    reused = search.run("pair_candidates", surrogate=prescan.value)
    assert fresh.status == reused.status == "screening_only"
    assert fresh.fit_charges == 1
    assert reused.fit_charges == 0
    assert isinstance(fresh.summary["leads"], list)


def test_compare_lists_all_two_se_candidates_and_soft_nested_removals():
    data = sample(groups=5)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=5, mstop=60)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    comparison = search.run("compare", candidates={
        "smooth four": "bbs(x, df=4) + bols(z)",
        "smooth three": "bbs(x, df=3) + bols(z)",
    })
    names = [row["candidate"] for row in comparison.summary["two_se_candidates"]]
    assert names == [row["candidate"] for row in sorted(
        comparison.summary["two_se_candidates"], key=lambda row: row["change"])]
    assert set(names) == {"smooth four", "smooth three"}
    assert comparison.summary["also_cleared"] == comparison.summary["two_se_candidates"][1:]

    linear = mb.open_search({"x": data["x"], "z": data["z"],
                             "y": data["x"] + .02 * data["z"]},
                            "y", family=mb.Gaussian(), folds=5, mstop=30)
    linear.run("set_spec", rhs="bols(x) + bols(z)")
    drops = linear.run("compare", candidates={"drop z": "bols(x)"})
    assert drops.summary["soft_removal_candidates"][0]["candidate"] == "drop z"
    adopted = linear.run("adopt", comparison_id=drops.evidence_id,
                         candidate="drop z", simplification_se=2)
    assert adopted.summary["spec"] == "bols(x)"


def test_frozen_transcript_exports_metadata_but_cannot_resume():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2, mstop=12)
    search.run("set_spec", rhs="bols(x)")
    search.run("fit")
    search.run("freeze")
    with pytest.raises(ValueError, match="before freeze"):
        search.checkpoint()
    search.run("final_evaluate", data=sample(seed=99, groups=2, rows_per_group=12))
    transcript = json.loads(json.dumps(search.export_transcript(), allow_nan=False))
    assert transcript["kind"] == "search_transcript"
    assert transcript["frozen"] and transcript["evaluated"]
    assert transcript["final_evaluation"]["score_name"] == "mse"
    assert transcript["budget_report"]["fit_charges"] == 1
    assert "folds" not in transcript and "model" not in transcript
    with pytest.raises(ValueError, match="cannot resume fitting"):
        mb.SearchSession.resume(data, transcript, family=mb.Gaussian())


def test_term_report_flags_a_collinear_proxy_without_dropping_it():
    rng = np.random.default_rng(41)
    x = rng.normal(size=120)
    proxy = x + rng.normal(0, .01, len(x))
    data = {"x": x, "proxy": proxy, "y": 2*x + rng.normal(0, .08, len(x))}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3,
                            mstop=45, locks=("x",))
    search.run("set_spec", rhs="bols(x) + bols(proxy)")
    report = search.run("term_report", shapes=False, proxy_hygiene=True)
    notes = report.summary["proxy_notes"]
    assert any(row["term"] == "bols(proxy)" and row["locked_sibling"] == "x"
               and row["drop_candidate"] == "bols(x)" for row in notes)
    assert search.spec == "bols(x) + bols(proxy)"


def test_interaction_screen_marks_a_binary_threshold_sibling_as_advisory():
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    switch = rng.integers(0, 2, len(x)).astype(float)
    data = {"x": x, "switch": switch,
            "y": 3*x*switch + rng.normal(0, .1, len(x))}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3, mstop=40)
    search.run("set_spec", rhs="bols(x) + bols(switch)")
    search.run("fit")
    screened = search.run("interaction_screen", pairs=[("x", "switch")])
    lead = screened.summary["top_pairs"][0]
    assert lead["thresholded_sibling_suggested"]
    assert lead["threshold_column"] == "switch"
    assert lead["sibling_term"] == "bols(x, by=switch)"
    assert "not a paired two-SE" in screened.summary["interpretation"]
    assert search.spec == "bols(x) + bols(switch)"


def test_transform_candidates_nominates_log_and_one_cutpoint_with_a_charged_probe():
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(size=120))
    z = rng.normal(size=120)
    data = {"x": x, "z": z,
            "y": 2*np.log1p(x) + x*(z > 0) + rng.normal(0, .1, len(x))}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2,
                            mstop=30, budget=mb.CostBudget(fits=20))
    search.run("set_spec", rhs="bols(x) + bols(z)")
    nominated = search.run("transform_candidates", rounds=60)
    assert nominated.status == "screening_only" and nominated.fit_charges == 1
    by_kind = {row["kind"]: row for row in nominated.summary["leads"]}
    assert by_kind["log1p_predictor"]["term"] == "bols(log1p(x))"
    assert by_kind["single_cutpoint_interaction"]["term"] == "btree(z, by=x, maxdepth=1)"
    measured = search.run("compare", candidates={
        "log": by_kind["log1p_predictor"]["candidate_rhs"],
        "cutpoint": by_kind["single_cutpoint_interaction"]["candidate_rhs"],
    })
    assert measured.status == "measured" and measured.fit_charges == 6
    assert search.spec == "bols(x) + bols(z)"


def test_response_probe_compares_back_transformed_predictions_on_original_mse():
    rng = np.random.default_rng(24)
    x = rng.uniform(0, 3, 100)
    y = np.expm1(1 + .7*x) + rng.normal(0, .05, len(x))
    data = {"x": x, "y": y}
    search = mb.open_search(data, "y", family=mb.Gaussian(), score="mse",
                            folds=4, mstop=40, budget=mb.CostBudget(fits=8))
    search.run("set_spec", rhs="bols(x)")
    measured = search.run("response_probe", transforms=("identity", "log1p"))
    assert measured.fit_charges == 8 and measured.status == "measured"
    rows = measured.summary["rows"]
    assert rows[0]["formula"] == "y ~ bols(x)"
    assert rows[1]["formula"] == "log1p(y) ~ bols(x)"
    assert rows[1]["mse"] < rows[0]["mse"]
    assert measured.summary["decision"] == "measured only; no response or family switch"
    assert search.spec == "bols(x)" and search.family.__class__ is mb.Gaussian
    assert search.state()["budget"]["fit_charges"] == 8


def test_transform_candidates_flags_spread_without_switching_family():
    rng = np.random.default_rng(22)
    x = rng.uniform(0, 3, 180)
    data = {"x": x, "y": 1 + x + np.exp(x)*rng.normal(size=len(x))}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3,
                            mstop=60, budget=mb.CostBudget(fits=8))
    search.run("set_spec", rhs="bols(x)")
    search.run("fit")
    hints = search.run("transform_candidates", rounds=40)
    assert hints.fit_charges == 1
    assert hints.summary["spread_hint"]["strongest"] == "x"
    assert "no family switch" in hints.summary["spread_hint"]["suggestion"]
    assert search.family.__class__ is mb.Gaussian


def test_inprocess_judge_chooses_tools_and_cannot_request_test_evaluation():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2, mstop=12,
                            budget=mb.CostBudget(fits=20))
    seen = []

    def choose(digest):
        assert "data" not in digest and "test" not in digest
        assert "final_evaluate" not in digest["state"]["available_actions"]
        seen.append(digest["state"]["state_id"])
        action, arguments = (("set_spec", {"rhs": "bols(x) + bols(z)"})
                             if len(seen) == 1 else ("stop", {}))
        return {"action": action, "arguments": arguments,
                "state_id": seen[-1], "reasoning": "check a small specification"}

    assert search.drive(choose, max_actions=3) == "judge_stopped"
    assert [item.action for item in search.evidence] == ["judge", "set_spec", "judge"]
    assert search.spec == "bols(x) + bols(z)"
    assert seen[0] != seen[1]

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            assert "one next action" in kwargs["messages"][0]["content"]
            digest = json.loads(kwargs["messages"][1]["content"])
            answer = {"action": "stop", "arguments": {},
                      "state_id": digest["state"]["state_id"],
                      "reasoning": "sufficient for this demonstration"}
            return {"choices": [{"message": {"content": json.dumps(answer)}}],
                    "usage": {"total_tokens": 20}}

    provider = mb.LLMJudge(client=FakeClient(), max_calls=1,
                           max_total_tokens=100)
    assert search.drive(provider, max_actions=1) == "judge_stopped"
    record = search.evidence[-1].summary["provider_call"]
    assert record["status"] == "received"
    assert record["usage"]["total_tokens"] == 20
    assert record["prompt_sha256"] and record["reply_sha256"]


def test_failed_provider_or_stale_action_leaves_session_usable():
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2)

    class FailingClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **_kwargs):
            raise RuntimeError("provider unavailable")

    judge = mb.LLMJudge(client=FailingClient(), max_calls=1)
    assert search.drive(judge, max_actions=1) == "judge_failed"
    assert search.evidence[-1].summary["provider_call"]["status"] == "failed"
    assert search.spec is None

    def stale(_digest):
        return {"action": "set_spec", "arguments": {"rhs": "bols(x)"},
                "state_id": "old state", "reasoning": "stale"}
    assert search.drive(stale, max_actions=1) == "judge_failed"
    assert search.spec is None
    assert search.run("set_spec", rhs="bols(x) + bols(z)").status == "unmeasured"


def test_family_can_be_declared_after_open_and_replayed():
    data = sample(groups=4)
    search = mb.open_search(data, "y", folds=2)
    assert "set_spec" not in search.state()["available_actions"]
    search.run("set_family", family=mb.Gaussian())
    search.run("set_spec", rhs="bols(x)")
    checkpoint = json.loads(json.dumps(search.checkpoint(), allow_nan=False))
    resumed = mb.SearchSession.resume(data, checkpoint, family=mb.Gaussian())
    assert resumed.state()["state_id"] == search.state()["state_id"]
    assert resumed.spec == "bols(x)"


def test_tree_selection_keyword_is_not_mistaken_for_a_data_column():
    from mboostrs.formula import term_variables

    assert term_variables("bols(sse)") == ("sse",)
    assert term_variables("btree(z, selection=sse)") == ("z",)
    data = sample(groups=4)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2)
    result = search.run("set_spec", rhs="bols(x) + btree(z, selection=sse)")
    assert "btree(z, selection=sse)" in result.summary["spec"]


def test_causal_session_excludes_treatment_and_post_treatment_from_search():
    rng = np.random.default_rng(17)
    x = rng.normal(size=80)
    treatment = rng.integers(0, 2, 80).astype(float)
    mediator = treatment + rng.normal(size=80)
    data = {"x": x, "a": treatment, "mediator": mediator,
            "y": 1.2*treatment + x + rng.normal(size=80)}
    design = mb.CausalSpec(
        treatment="a", outcome="y", covariates=("x",),
        time_zero="enrolment", follow_up="one year",
        adjustment_rationale="x is measured before treatment",
        post_treatment=("mediator",))
    search = mb.open_search(data, "y", family=mb.Gaussian(),
                            causal=design, folds=2)
    names = {kind.name for kind in search.run("classify").value}
    assert "x" in names and "a" not in names and "mediator" not in names
    with pytest.raises(ValueError, match="forbidden"):
        search.run("set_spec", rhs="bols(x) + bols(a)")
    with pytest.raises(ValueError, match="forbidden"):
        search.run("set_spec", rhs="bols(x) + bols(mediator)")
    with pytest.raises(ValueError, match="locked"):
        search.run("set_spec", rhs="1")
    assert search.run("causal_traps").status == "screening_only"
    search.run("set_spec", rhs="bols(x)")
    recovered = mb.SearchSession.resume(
        data, json.loads(json.dumps(search.checkpoint())), family=mb.Gaussian())
    assert recovered.state()["causal_design"] == search.state()["causal_design"]


def test_na_omit_sessions_run_residual_actions_and_tune_on_complete_cases():
    """Issue #39: with na_action='omit' the retained fit has complete-case
    rows, while interaction_screen, pair_candidates and tune read the full
    frame. Every one of them raised on the length mismatch; they must instead
    use the rows the model was actually fitted on."""
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    group = np.repeat([f"g{i}" for i in range(6)], 20)
    y = x + z + rng.normal(0, .1, n)
    holed = np.where(rng.random(n) < .2, np.nan, x)
    data = {"x": holed, "z": z, "grp": group, "y": y}
    keep = np.isfinite(holed)
    assert keep.sum() < n

    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3, groups="grp",
                            seed=0, na_action="omit", mstop=25, nu=.3)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    search.run("fit")
    assert len(search.model.residuals()) == int(keep.sum())

    screened = search.run("interaction_screen", pairs=[("x", "z")])
    assert screened.status == "screening_only"
    assert screened.summary["top_pairs"][0]["left"] == "x"
    pairs = search.run("pair_candidates")
    assert pairs.status == "screening_only"
    tuned = search.run("tune", start=25)
    assert tuned.status == "measured" and search.mstop >= 1

    # The same tuning on the pre-dropped complete-case frame agrees when both
    # sessions see the same fold design (the subset scheme of the full frame).
    complete = {name: values[keep] for name, values in data.items()}
    dropped = mb.open_search(complete, "y", family=mb.Gaussian(),
                             folds=search.folds.take(np.flatnonzero(keep)), seed=0,
                             mstop=25, nu=.3)
    dropped.run("set_spec", rhs="bols(x) + bols(z)")
    dropped.run("tune", start=25)
    assert search.mstop == dropped.mstop


def test_final_evaluate_scores_family_loss_on_the_holdout():
    """Issue #40: a session opened with score='family' (the natural CV risk
    for count families) used to be refused at final_evaluate. The holdout now
    reports the weighted mean family loss with the fitted nuisance restored —
    the quantity cvrisk reports for a fold."""
    rng = np.random.default_rng(7)
    n = 80
    x = rng.uniform(0, 2, n)
    counts = rng.poisson(np.exp(.9 * x)).astype(float)
    train = {"x": x[:60], "y": counts[:60]}
    test = {"x": x[60:], "y": counts[60:]}

    search = mb.open_search(train, "y", family=mb.NBinomial(), score="family",
                            folds=3, mstop=30)
    search.run("set_spec", rhs="bols(x)")
    search.run("fit")
    search.run("freeze")
    final = search.run("final_evaluate", data=test)

    assert final.status == "measured"
    assert final.summary["score_name"] == "family"
    assert np.isfinite(final.summary["score"]) and final.summary["score"] > 0
    assert final.summary["n_test"] == len(test["y"])
    assert search.evaluated

    # Oracle: the NBinomial loss with the fitted dispersion. The dispersion
    # the fit estimated is huge, so the NLL collapses onto the Poisson NLL:
    # mean(y*log(mu) - mu - log(y!)) with mu = exp(eta) on the test rows.
    from scipy.special import gammaln

    mu = np.exp(np.asarray(search.model.predict(test, type="link"), dtype=float))
    y = test["y"]
    poisson_nll = float(-np.mean(y*np.log(mu) - mu - gammaln(y + 1)))
    # The fitted dispersion sits at its cap (100), where the NBinomial loss is
    # the Poisson NLL up to the finite-dispersion correction.
    assert np.isclose(final.summary["score"], poisson_nll, atol=1e-2)

    # One-shot: a second final_evaluate is refused.
    with pytest.raises(ValueError):
        search.run("final_evaluate", data=test)


def test_final_evaluate_family_score_matches_mse_for_gaussian():
    """Gaussian family loss is the squared error, so a Gaussian session's
    family-scored holdout must equal its mse-scored holdout exactly — the
    two routes compute the same number through different machinery."""
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 3, 70)
    y = 1 + .6*x + rng.normal(0, .2, 70)
    train, test = ({"x": x[:50], "y": y[:50]}, {"x": x[50:], "y": y[50:]})

    sessions = {}
    for score in ("mse", "family"):
        search = mb.open_search(train, "y", family=mb.Gaussian(), score=score,
                                folds=3, mstop=25)
        search.run("set_spec", rhs="bols(x) + bbs(x, df=3)")
        search.run("fit")
        search.run("freeze")
        sessions[score] = search.run("final_evaluate", data=test).summary["score"]
    # The same quantity through two summation orders: equal to rounding.
    assert np.isclose(sessions["mse"], sessions["family"], rtol=1e-12)


def test_final_evaluate_refuses_family_score_where_undefined():
    """A multinomial family has no scalar family loss on a holdout frame; the
    refusal names the alternative instead of crashing. (Distributional LSS
    families cannot open a session: the evidence cache cannot fingerprint
    them.)"""
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 2, 60)
    group = np.repeat(["a", "b", "c"], 20)
    y = np.where(x > 1, group, np.roll(group, 1))
    train, test = ({"x": x[:45], "cls": y[:45]}, {"x": x[45:], "cls": y[45:]})
    search = mb.open_search(train, "cls", family=mb.Multinomial(), score="family",
                            folds=2, mstop=15)
    search.run("set_spec", rhs="bols(x)")
    search.run("fit")
    search.run("freeze")
    with pytest.raises(NotImplementedError, match="log_loss"):
        search.run("final_evaluate", data=test)


def test_structure_candidates_surfaces_typed_basis_leads():
    """Issue #42: latent structure in an engineered basis (log1p slopes,
    cutpoint indicators, hinges) is invisible to raw-column pair screens. The
    charged screening action builds the basis, screens it against the
    incumbent's residualized residual, and returns typed terms for compare."""
    rng = np.random.default_rng(1)
    n = 240
    x = np.exp(rng.uniform(-1, 3, n))          # skewed positive slope column
    z = rng.uniform(-1, 1, n)                  # threshold at its median, 0
    w = rng.uniform(0, 1, n)                   # hinge at its median, 0.5
    y = (2.0*np.log1p(x) * (z >= 0) + 3.0*np.maximum(w - .5, 0)
         + rng.normal(0, .15, n))
    data = {"x": x, "z": z, "w": w, "y": y}

    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3,
                            mstop=60, nu=.3, budget=mb.CostBudget(fits=20))
    search.run("set_spec", rhs="bols(x) + bols(z) + bols(w)")
    search.run("fit")
    nominated = search.run("structure_candidates", rounds=60)

    assert nominated.status == "screening_only"
    assert nominated.fit_charges == 1
    leads = nominated.summary["leads"]
    by_kind = {}
    for lead in leads:
        by_kind.setdefault(lead["kind"], []).append(lead)

    interaction = by_kind["basis_interaction"][0]  # ranked first overall
    assert leads[0]["kind"] == "basis_interaction"
    assert interaction["slope_column"] == "x"
    assert interaction["threshold_column"] == "z"
    assert abs(interaction["cutpoint"]) < .15      # the median split
    assert interaction["term"] == "btree(z, by=log1p(x), maxdepth=1)"
    assert interaction["screens_positive"]

    hinge_columns = {lead["column"] for lead in by_kind["hinge_term"]}
    assert "w" in hinge_columns                     # the true hinge column
    assert all(lead["term"].startswith("bols(pmax(")
               for lead in by_kind["hinge_term"])
    assert search.spec == "bols(x) + bols(z) + bols(w)"  # screening only

    measured = search.run("compare", candidates={
        "interaction": interaction["candidate_rhs"],
        "hinge": by_kind["hinge_term"][0]["candidate_rhs"],
    })
    assert measured.status == "measured"
    improved = [row for row in measured.summary["rows"]
                if row["specification"] != "current"
                and row["change"] < -2 * row["change se"]]
    assert any("by=log1p(x)" in row["rhs"] for row in improved)


def test_structure_candidates_reuses_a_prescan_report_at_zero_cost():
    data = sample(seed=5)
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=2,
                            mstop=25, budget=mb.CostBudget(fits=10))
    prescan = search.run("pre_scan", engine="native", rounds=40)
    search.run("set_spec", rhs="bols(x) + bols(z)")
    fresh = search.run("structure_candidates", report=prescan.value)
    again = search.run("structure_candidates", report=prescan.value)
    assert fresh.fit_charges == again.fit_charges == 0
    assert fresh.status == again.status == "screening_only"
    assert fresh.summary["source_engine"] == "native"
    restricted = search.run("structure_candidates", report=prescan.value,
                            columns=("x", "z"))
    assert set(restricted.summary["screened_columns"]) <= {"x", "z"}


def test_count_sessions_attribute_mean_versus_dispersion_covariates():
    """Issue #44: a count session that parks a dispersion covariate on the
    mean buys nothing and loses the sigma path. transform_candidates now
    attributes each covariate: signed Pearson residual lines up with a
    missing mean term, log of its square with dispersion structure; and the
    action no longer crashes for count families, which heteroscedasticity
    refuses by design."""
    rng = np.random.default_rng(3)
    n = 240
    x = rng.normal(size=n)          # mean covariate, already in the spec
    w = rng.normal(size=n)          # mean covariate, missing from the spec
    v = rng.normal(size=n)          # dispersion-only covariate
    noise = rng.normal(size=n)      # neither
    mu = np.exp(.5 * x + .4 * w)
    theta = np.exp(1.0 + .8 * v)
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    data = {"x": x, "w": w, "v": v, "noise": noise, "y": y}

    for family in (mb.Poisson(), mb.NBinomial()):
        search = mb.open_search(data, "y", family=family, folds=3,
                                mstop=40, nu=.3)
        search.run("set_spec", rhs="bols(x)")
        search.run("fit")
        probe = search.run("transform_candidates", rounds=40)
        assert probe.status == "screening_only"
        hint = probe.summary["dispersion_hint"]
        assert hint["dispersion_columns"] == ["v"]
        by_column = {row["column"]: row for row in hint["rows"]}
        assert by_column["w"]["attribution"] == "mean"
        assert by_column["w"]["signed_residual_r2"] > .01
        assert by_column["v"]["attribution"] == "dispersion"
        assert by_column["v"]["log_square_r2"] > .01
        assert by_column["v"]["signed_residual_r2"] < .01
        assert "x" not in by_column and "noise" not in by_column
        assert "gamboostLSS" in hint["recipe"]
        if type(family).__name__ == "NBinomial":
            assert hint["theta"] and np.isfinite(hint["theta"])
        assert search.spec == "bols(x)"  # screening changed nothing


def test_coef_and_btree_splits_expose_the_fitted_tree():
    """Issue #46, narrowed: coef() crashed for every btree-containing model
    (tree coefficient vectors encode structure and do not add), and nothing
    public read a fitted tree's split points. Both now work: per-step
    decoded node records under the disambiguated learner name, and the
    cutpoints each tree actually used with their boosting-step counts."""
    rng = np.random.default_rng(2)
    n = 240
    x = rng.uniform(0, 3, n)
    z = rng.uniform(0, 1, n)
    y = x * (z >= .3) + rng.normal(0, .12, n)
    model = mb.gamboost("y ~ bols(x) + btree(x, z, maxdepth=1)",
                        data={"x": x, "z": z, "y": y}, family=mb.Gaussian(),
                        control=mb.boost_control(40, .3))

    coefficients = model.coef()
    # A bols(x) reports [intercept, slope], as mboost's coef does.
    linear = coefficients["x (bols)"]
    assert np.shape(linear) == (2,)
    records = coefficients["x (btree)"]
    assert records and all("nodes" in record for record in records)
    kinds = {node["kind"] for record in records for node in record["nodes"]}
    assert kinds <= {"split", "leaf", "level_split"} and "split" in kinds
    raw = model.coef(aggregate="none")["x (btree)"]
    assert isinstance(raw[0], list)  # decoded per step, not the flat encoding

    splits = model.btree_splits()["x (btree)"]
    assert splits["variables"] == ["x", "z"]
    on_z = [row for row in splits["splits"] if row["variable"] == "z"]
    assert on_z and min(abs(row["cutpoint"] - .3) for row in on_z) < .05
    assert all(row["steps"] >= 1 for row in splits["splits"])


def test_structure_candidates_distill_a_soft_tree_soak():
    """Issue #46's desk-ops acceptance, narrowed to what was genuinely
    missing: a stump soak partially absorbs threshold structure; the screen
    on its residual still ranks the typed interaction first, and the soak's
    own splits record the true cutpoint. The emitted term learns its cutpoint
    at compare time, so the distilled form is measurable against the soak."""
    rng = np.random.default_rng(2)
    n = 240
    x = np.exp(rng.uniform(-1, 3, n))
    z = rng.uniform(0, 1, n)          # threshold at 0.3, between quartiles
    w = rng.uniform(0, 1, n)
    y = (2.0*np.log1p(x) * (z >= .3) + 3.0*np.maximum(w - .5, 0)
         + rng.normal(0, .15, n))
    data = {"x": x, "z": z, "w": w, "y": y}

    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=5,
                            mstop=60, nu=.3, budget=mb.CostBudget(fits=24))
    search.run("set_spec",
               rhs="bols(x) + bols(z) + bols(w) + btree(x, z, maxdepth=1)")
    search.run("fit")
    soak_splits = search.model.btree_splits()["x (btree)"]["splits"]
    on_z = [row for row in soak_splits if row["variable"] == "z"]
    assert on_z and min(abs(row["cutpoint"] - .3) for row in on_z) < .05

    distilled = search.run("structure_candidates", rounds=60)
    top = distilled.summary["leads"][0]
    assert top["kind"] == "basis_interaction"
    assert top["slope_column"] == "x" and top["threshold_column"] == "z"
    assert top["term"] == "btree(z, by=log1p(x), maxdepth=1)"

    measured = search.run("compare", candidates={"distilled":
                                                 top["candidate_rhs"]})
    row = next(item for item in measured.summary["rows"]
               if item["specification"] == "distilled")
    assert row["change"] < -2 * row["change se"]


def test_structure_candidates_survives_columns_without_usable_cutpoints():
    """A column whose quartiles all sit against the edge of its range — a
    zero-inflated offices-style count with a clipped tail — contributes no
    hinge and no indicator. The screen used to crash unpacking the missing
    pair; it must skip the column and keep the rest of the basis."""
    rng = np.random.default_rng(6)
    n = 160
    spike = np.where(rng.random(n) < .95, 0.0, 200.0)   # every quartile is 0
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(0, .1, n)
    data = {"spike": spike, "x": x, "y": y}

    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3,
                            mstop=25, nu=.3, budget=mb.CostBudget(fits=6))
    search.run("set_spec", rhs="bols(x)")
    search.run("fit")
    screened = search.run("structure_candidates", rounds=40)
    assert screened.status == "screening_only"
    # spike contributes no hinge or indicator lead (its only cut is degenerate)
    assert not any(lead.get("column") == "spike"
                   and lead["kind"] in ("hinge_term", "cutpoint_indicator")
                   for lead in screened.summary["leads"])


def test_compare_reports_paired_winner_gaps():
    """Issue #58: prefer-interpretable needs the gap between two clearing
    candidates, in paired SE on the folds they share — a number the rows
    (each measured against the incumbent) never carried. The summary's
    winner_gaps must agree with a manual paired SE on the evaluations."""
    from mboostrs.explore import (_paired_se, _partitions,
                                   compare_specifications)

    rng = np.random.default_rng(10)
    n = 240
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    data = {"x": x, "z": z,
            "y": np.sin(2*x) + .5*z + np.maximum(x, 0) + rng.normal(0, .1, n)}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=5,
                            mstop=60, nu=.3, budget=mb.CostBudget(fits=30))
    search.run("set_spec", rhs="bols(x) + bols(z)")
    comparison = search.run("compare", candidates={
        "typed": "bols(x) + bbs(x, df=4) + bols(z)",
        "catch-all": "bols(x) + bols(z) + btree(x, z, maxdepth=3)",
    })
    gaps = comparison.summary["winner_gaps"]
    clearing = [c["candidate"] for c in comparison.summary["two_se_candidates"]]
    assert len(gaps) == len(clearing) - 1
    best = comparison.summary["two_se_candidates"][0]
    # An independent scoring of the same candidates supplies the fold scores
    # the manual oracle needs; the session's evidence value carries plain
    # rows only, so the gaps must agree with this outside computation.
    independent = compare_specifications(
        {"current": "bols(x) + bols(z)",
         "typed": "bols(x) + bbs(x, df=4) + bols(z)",
         "catch-all": "bols(x) + bols(z) + btree(x, z, maxdepth=3)"},
        data, "y", family=mb.Gaussian(), folds=search.folds, mstop=60,
        nu=.3, seed=0, score="mse", include_evaluations=True)
    by_name = {row["specification"]: row for row in independent}
    for gap in gaps:
        assert gap["candidate"] != best["candidate"]
        manual = float(_paired_se(
            by_name[gap["candidate"]]["evaluation"].fold_scores,
            by_name[best["candidate"]]["evaluation"].fold_scores,
            _partitions(search.folds)))
        assert np.isclose(gap["paired_se"], manual, rtol=1e-8)
        assert np.isclose(gap["gap_vs_best"],
                          by_name[gap["candidate"]]["change"]
                          - by_name[best["candidate"]]["change"], rtol=1e-8)
        assert gap["within_two_se"] == (gap["gap_vs_best"] <= 2 * gap["paired_se"])
    # A single clearing candidate leaves nothing to gap.
    solo = mb.open_search(data, "y", family=mb.Gaussian(), folds=3,
                          mstop=40, nu=.3, budget=mb.CostBudget(fits=12))
    solo.run("set_spec", rhs="bols(x) + bols(z)")
    one = solo.run("compare", candidates={"typed": "bbs(x, df=4) + bols(z)"})
    assert one.summary["winner_gaps"] == []


def test_term_report_reports_concurvity_without_dropping():
    """Issue #118: a near function-space duplicate (x2 = x + small noise)
    is reported with mgcv's concurvity measures, and the specification is
    left as it is; an unrelated smooth is not reported. The pairwise feed
    names the duplicated pair itself, once, from the boosted fit."""
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    data = {"x": x, "x2": x + rng.normal(0, 0.02, n), "z": z,
            "y": np.sin(2 * x) + z + rng.normal(0, 0.3, n)}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3, mstop=150)
    search.run("set_spec", rhs="bbs(x) + bbs(x2) + bbs(z)")
    report = search.run("term_report", shapes=False, concurvity=True)
    flagged = {row["term"]: row for row in report.summary["concurvity_notes"]}
    assert set(flagged) == {"bbs(x)", "bbs(x2)"}
    assert all(row["worst"] > 0.99 for row in flagged.values())
    pairs = report.summary["concurvity_pairs"]
    assert [set(row["pair"]) for row in pairs] == [{"bbs(x)", "bbs(x2)"}]
    assert pairs[0]["worst"] > 0.99
    assert set(pairs[0]) == {"pair", "worst", "observed", "estimate", "interpretation"}
    assert search.spec == "bbs(x) + bbs(x2) + bbs(z)"


def test_term_report_concurvity_pairs_empty_below_threshold():
    """A specification whose terms are mutually orthogonal reports no
    pairs, and the term report still returns the key for callers."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    data = {"x": x, "z": z,
            "y": np.sin(2 * x) + z + rng.normal(0, .3, n)}
    search = mb.open_search(data, "y", family=mb.Gaussian(), folds=3, mstop=100)
    search.run("set_spec", rhs="bbs(x) + bbs(z)")
    report = search.run("term_report", shapes=False, concurvity=True)
    assert report.summary["concurvity_pairs"] == []
    assert report.summary["concurvity_notes"] == []
