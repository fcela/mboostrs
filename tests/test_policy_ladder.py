"""Stage-specific policy edits and preservation of accepted structure."""

from types import SimpleNamespace

import numpy as np
import pytest

import mboostrs as mb
import mboostrs.policy as policy
import mboostrs.search_session as actions


ADDITIVE = "bbs(x) + bols(z)"
PAIR = "bbs(x) %X% bols(z)"
TREE = "btree(x, z, maxdepth=5, selection=sse)"


def test_pair_stage_keeps_accepted_main_effects_and_deduplicates_pairs():
    current = f"{ADDITIVE} + {PAIR}"
    discovered = SimpleNamespace(formula=(
        "y ~ bols(x) + bbs(z) + bbs(x)%X%bols(z) + bols(x) %X% bbs(z)"))
    assert policy._rung_rhs("pairs", current, discovered) == (
        current + " + bols(x) %X% bbs(z)")


def test_tree_stage_keeps_accepted_pair_and_ignores_changed_shapes():
    current = f"{ADDITIVE} + {PAIR}"
    discovered = SimpleNamespace(formula=f"y ~ btree(x) + {TREE}",
                                 evidence={"adopted_tree": TREE})
    assert policy._rung_rhs("hybrid", current, discovered) == current + " + " + TREE


def test_shape_tree_is_not_mislabeled_as_an_adopted_residual_tree():
    discovered = SimpleNamespace(formula="y ~ btree(x) + bols(z)", evidence={})
    assert policy._rung_rhs("hybrid", ADDITIVE, discovered) == ADDITIVE


def test_pair_nominations_include_varying_coefficients_despite_family_rejection():
    from mboostrs._autospec import InteractionLead

    discovered = SimpleNamespace(formula="y ~ " + ADDITIVE, evidence={
        "interactions": (
            InteractionLead("x", "g", 1.0, "bbs(x, by=g)", "varying"),
            InteractionLead("h", "z", 0.5, "brandom(h, by=z)", "varying"),
            InteractionLead("x", "z", 1.0, "", "duplicate-axis"),
        ),
        "rejected_interactions": ("bbs(x, by=g)", "brandom(h, by=z)"),
    })
    assert policy._rung_rhs("pairs", ADDITIVE, discovered) == (
        ADDITIVE + " + bbs(x, by=g) + brandom(h, by=z)")


def test_formula_only_pair_recommendation_keeps_varying_coefficients():
    discovered = SimpleNamespace(formula="y ~ bbs(x) + bbs(x, by = g)")
    assert policy._rung_rhs("pairs", ADDITIVE, discovered) == ADDITIVE + " + bbs(x, by = g)"


def test_family_rejected_tree_is_still_nominated_for_requested_metric():
    discovered = SimpleNamespace(formula="y ~ " + ADDITIVE, evidence={
        "residual_tree": {"term": TREE, "adopted": False, "mstop": 100,
                          "relative_gain": -0.01},
    })
    assert policy._rung_rhs("hybrid", ADDITIVE, discovered) == ADDITIVE + " + " + TREE


def test_failed_tree_sweep_is_not_retried_as_a_successful_nomination():
    discovered = SimpleNamespace(formula="y ~ " + ADDITIVE, evidence={
        "residual_tree": {"term": TREE, "adopted": False,
                          "reason": "the candidate's sweep failed"},
    })
    assert policy._rung_rhs("hybrid", ADDITIVE, discovered) == ADDITIVE


@pytest.mark.parametrize("additive_loss,pair_loss,hybrid_loss,adopted,expected_profile", [
    (0.8, 0.6, 0.4, [True, True, True, True], "hybrid"),
    (0.8, 0.6, 0.9, [True, True, True, False], "tensor"),
    (1.1, 0.6, 0.9, [True, False, True, False], "tensor"),
    (1.1, 1.1, 0.4, [True, False, False, True], "hybrid"),
    (1.1, 1.1, 1.1, [True, False, False, False], "linear"),
])
def test_full_ladder_scores_nested_proposals_and_retains_best_model(
        monkeypatch, additive_loss, pair_loss, hybrid_loss, adopted, expected_profile):
    data = {"x": np.linspace(-1, 1, 40), "z": np.linspace(-2, 2, 40),
            "y": np.linspace(-3, 3, 40)}
    scored = []
    models = {}
    folds_seen = []
    discoveries = []

    def discover(*args, **kwargs):
        discoveries.append(kwargs["budget"])
        if not kwargs["budget"]:
            return SimpleNamespace(formula="y ~ " + ADDITIVE, family=mb.Gaussian(), fits=0)
        folds_seen.append(kwargs["folds"])
        assert kwargs["interactions"] == "report" and not kwargs["residual_tree"]
        assert kwargs["joint_tree"] is False
        return SimpleNamespace(formula="y ~ " + ADDITIVE, family=mb.Gaussian(),
                               fits=2, evidence={})

    def tune(formula, *args, **kwargs):
        folds_seen.append(kwargs["folds"])
        rhs = policy._rhs_of(formula)
        models[rhs] = object()
        return SimpleNamespace(mstop=10, model=models[rhs], fits=2)

    def score(rhs, *args, **kwargs):
        folds_seen.append(kwargs["folds"])
        assert kwargs["score"] == "mse"
        scored.append(rhs)
        loss = (hybrid_loss if TREE in rhs else pair_loss if PAIR in rhs
                else additive_loss if "bbs" in rhs else 1.0)
        return SimpleNamespace(score=loss, fold_scores=[loss, loss], fit_count=2)

    monkeypatch.setattr(actions, "autospec", discover)
    monkeypatch.setattr(actions, "tune_mstop", tune)
    monkeypatch.setattr(actions, "cv_risk", score)

    def incumbent_leads(frame, response, family, rhs, fitted, mstop, groups, seed, surrogate):
        assert fitted is models[rhs]
        assert mstop == 10
        return (SimpleNamespace(term=PAIR, kind="tensor", source="residual"),), 0

    monkeypatch.setattr(policy, "_incumbent_pair_leads", incumbent_leads)
    result = policy.auto_policy(data, "y", folds=2, metric="mse")
    linear = "bols(x) + bols(z)"
    pair_base = ADDITIVE if adopted[1] else linear
    tree_base = pair_base + " + " + PAIR if adopted[2] else pair_base
    assert scored == [linear, ADDITIVE, pair_base + " + " + PAIR, tree_base + " + " + TREE]
    assert all(folds is folds_seen[0] for folds in folds_seen)
    assert discoveries == [0, 100]  # schema and one measured shape discovery
    assert result.profile == expected_profile
    assert result.model is models[result.spec]
    assert [row.rung for row in result.trail] == ["linear", "additive", "pairs", "hybrid"]
    assert [row.adopted for row in result.trail] == adopted
    assert result.score == {"hybrid": hybrid_loss, "tensor": pair_loss, "linear": 1.0}[expected_profile]


def test_pair_screen_reuses_the_actual_short_incumbent_path(monkeypatch):
    x = np.linspace(-1, 1, 36)
    z = np.tile([-1., 0., 1.], 12)
    y = x * z + .1 * x
    data = {"x": x, "z": z, "y": y}
    observed = []

    def screen(_data, residual, **kwargs):
        observed.append(np.asarray(residual))
        return [SimpleNamespace(left="x", right="z", removed=1.0)]

    monkeypatch.setattr(mb, "screen_interactions", screen)
    fitted = SimpleNamespace(mstop=10, predict=lambda *, type: np.zeros(len(y)))
    leads, fits = policy._incumbent_pair_leads(
        data, "y", mb.Gaussian(), "bols(x) + bols(z)", fitted, 10,
        (), 0, False)
    assert fits == 0 and len(leads) == 2
    assert {lead.kind for lead in leads} == {"tensor", "joint_tree"}
    assert all((lead.left, lead.right) == ("x", "z") for lead in leads)
    assert "%X%" in next(lead.term for lead in leads if lead.kind == "tensor")
    assert next(lead.term for lead in leads if lead.kind == "joint_tree") == \
        "btree(x, z, maxdepth=3, selection=sse)"
    assert len(observed) == 1, "both forms must reuse a single residual screen"
    np.testing.assert_array_equal(observed[0], y)


def test_pair_prescan_does_not_reintroduce_columns_absent_from_incumbent(monkeypatch):
    from mboostrs._autospec import probe_interactions
    from mboostrs._frame import read_column

    data = {"x": np.arange(36., dtype=float), "z": np.tile([0., 1., 2.], 12),
            "w": np.linspace(-1, 1, 36), "y": np.tile([0., 1., 2.], 12)}
    monkeypatch.setattr(mb, "screen_interactions", lambda *args, **kwargs: [])
    surrogate = {"engine": "dual", "pair_sources": [], "pairs": [
        ("x", "w", .9, 1.0), ("x", "z", .8, 1.0)]}
    fitted = SimpleNamespace(mstop=10, predict=lambda *, type: np.zeros(36))
    leads, fits, reused = probe_interactions(
        data, "y", "y ~ bols(x) + bols(z)", mb.Gaussian(),
        [read_column(data, "x"), read_column(data, "z")],
        ["bols(x)", "bols(z)"], mstop=10, nu=.1, top=1,
        main_fit=fitted, surrogate=surrogate, exact_fit=True)
    assert fits == 0 and reused == 1
    assert len(leads) == 2
    assert {lead.kind for lead in leads} == {"tensor", "joint_tree"}
    assert all((lead.left, lead.right) == ("x", "z") for lead in leads)


@pytest.mark.parametrize("fit_limit", [70, 400])
def test_native_interaction_after_rejected_shapes_and_budget_checkpoint(fit_limit):
    # This recorded failure stopped at MSE 46.44 before pair screening.
    # Use the original fixture unchanged to exercise the complete native path.
    rng = np.random.default_rng(37)
    x, z = rng.uniform(-2, 2, (2, 160))
    data = {"x": x, "z": z,
            "y": 3*np.sin(x) + 2*z + 5*x*z + rng.normal(0, .3, 160)}
    result = policy.auto_policy(
        data, "y", family=mb.Gaussian(), folds=3, mstop=100,
        discovery_fits=100, preexploration="dual", budget=mb.CostBudget(fits=fit_limit))
    assert np.isfinite(result.model.predict()).all()
    assert result.fits <= fit_limit
    assert result.trail[1].rung == "additive" and not result.trail[1].adopted
    if fit_limit == 70:
        assert result.stopped_early == "fits"
        assert result.profile == "linear" and result.score is not None
    else:
        assert result.trail[2].rung == "pairs" and result.trail[2].adopted
        assert result.profile == "tensor" and result.score < 1.0
        assert result.trail[2].baseline_rhs == "bols(x) + bols(z)"


def test_native_varying_coefficient_is_tested_and_retained_over_tree():
    rng = np.random.default_rng(41)
    x = rng.uniform(-1, 1, 180)
    b = rng.integers(0, 2, 180)
    data = {"x": x, "b": b, "y": 8*(2*b-1)*x + rng.normal(0, .2, 180)}
    result = policy.auto_policy(
        data, "y", family=mb.Gaussian(), folds=3, mstop=100,
        discovery_fits=100, preexploration="dual", budget=mb.CostBudget(fits=400))
    assert result.profile == "varying"
    assert "bbs(x, by=b)" in result.spec
    assert result.score < .2 and result.fits <= 400
    assert result.trail[-1].rung == "hybrid" and not result.trail[-1].adopted
    assert "btree(" in result.trail[-1].candidate_rhs
    assert np.isfinite(result.model.predict()).all()


def test_budgeted_policy_records_action_charges_after_its_fallback():
    rng = np.random.default_rng(11)
    x, z = rng.normal(size=(2, 65))
    data = {"x": x, "z": z, "y": x*x + z + rng.normal(0, .2, len(x))}
    result = policy.auto_policy(
        data, "y", family=mb.Gaussian(), folds=2, mstop=20,
        complexity="simple", discovery_fits=30, budget=mb.CostBudget(fits=100))
    assert result.action_evidence
    assert result.fits == result.budget_report["fit_charges"]
    assert result.fits == sum(item.fit_charges for item in result.action_evidence)
    assert {item.action for item in result.action_evidence} >= {
        "autospec", "set_spec", "fit", "tune", "evaluate"}
