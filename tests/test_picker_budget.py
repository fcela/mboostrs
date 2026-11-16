"""The form picker's budget earmark holds at tight budgets (#511).

The A/B rerun found the picker starved: on 4/8 product-interaction replicates
the screen ranked x0×x1 first and the joint-tree lead was proposed, but the
`budget - spent >= 2*n_folds` gate failed because the pair_reserve the shape
probe left was consumed downstream by the post-shape mstop probe and the
deselection sweeps. These tests pin the fix:
(a) at budget 32 the picker runs and the true pair is adopted (starvation
    regression -- fails on the pre-fix code);
(b) the earmark is protected: the mstop probe and deselection cannot spend
    below it, and total fits never exceed the budget;
(c) the picker's cost is exactly the stated 3*n_folds fits.
"""
import numpy as np

import mboostrs as mb

N_FOLDS = 5  # fold_matrix() default: 5-fold
BUDGET = 32
PICKER_NEED = 3 * N_FOLDS


def _starvation_data(seed=511004):
    """The A/B battery's interaction panel (seed 511004 starved pre-fix)."""
    rng = np.random.default_rng(seed)
    n = 500
    X = rng.uniform(0, 1, size=(n, 6))
    y = (3.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
         + 1.5 * np.sin(np.pi * X[:, 2]) + 1.0 * X[:, 3]
         + rng.normal(scale=0.7, size=n))
    return {f"x{i}": X[:, i] for i in range(6)} | {"y": y}


def _run():
    return mb.autospec(
        _starvation_data(), "y", family=mb.Gaussian(),
        budget=BUDGET, nu=0.1, interactions="adopt",
        residual_tree=False, seed=511004,
    )


def test_picker_runs_at_budget_32_starvation_regression():
    # Pre-fix, the picker gate failed on this seed: the screen proposed the
    # joint-tree lead but budget - spent < 2*n_folds, so nothing was adopted.
    rec = _run()
    ev = rec.evidence
    assert ev.get("competing_pairs") == ("x0_x_x1",)
    form = ev.get("pair_form_x0_x_x1")
    assert form is not None, "picker must run at budget 32"
    assert form["winner"] == "tensor", form
    adopted = ev.get("adopted_interactions") or ()
    assert any("x0" in t and "x1" in t and "%X%" in t for t in adopted), \
        f"true pair not adopted: {adopted}"


def test_earmark_survives_downstream_spend():
    # The mstop probe runs (its evidence exists) and total fits respect the
    # budget, yet the picker still ran: downstream stages left the earmark.
    rec = _run()
    ev = rec.evidence
    assert "mstop" in ev, "mstop probe must still run under the earmark"
    assert ev.get("pair_form_x0_x_x1") is not None, "picker starved"
    assert rec.fits <= BUDGET, f"overspent: {rec.fits} > {BUDGET}"


def test_picker_cost_bounded_by_stated_budget():
    # Three candidates on the shared folds: exactly 3*n_folds fits, no more.
    rec = _run()
    form = rec.evidence.get("pair_form_x0_x_x1")
    assert form is not None
    assert form["picker_fits"] == PICKER_NEED, \
        f"picker cost {form['picker_fits']} != stated {PICKER_NEED}"


def test_picker_warrant_rejects_chance_pair():
    # On additive truth the picker runs the comparison but adopts nothing:
    # the shape rule finds no pair form beating the additive incumbent.
    rng = np.random.default_rng(7)
    n = 500
    X = rng.uniform(0, 1, size=(n, 6))
    y = 1.5 * np.sin(np.pi * X[:, 2]) + 1.0 * X[:, 3] + rng.normal(scale=0.7, size=n)
    data = {f"x{i}": X[:, i] for i in range(6)} | {"y": y}
    rec = mb.autospec(
        data, "y", family=mb.Gaussian(),
        budget=BUDGET, nu=0.1, interactions="adopt",
        residual_tree=False, seed=7,
    )
    adopted = rec.evidence.get("adopted_interactions") or ()
    pair_terms = [t for t in adopted if "%X%" in t or t.startswith("btree(")]
    assert not pair_terms, f"chance pair adopted: {pair_terms}"
