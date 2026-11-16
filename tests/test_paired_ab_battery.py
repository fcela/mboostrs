"""Accounting regression tests for the paired A/B battery (issue #511).

These encode the #509 reporting-audit rules that the instrument depends on:
tree terms are identified by btree_splits() membership (never name parsing);
candidate vs used columns are recorded separately; used columns for trees
are the union of numeric splits and level_split_variables; split columns
are structure, never effect proof.
"""

import numpy as np
import pytest

mb = pytest.importorskip("mboostrs")
from benches.paired_battery import (
    N_TOTAL,
    PANELS,
    _fit_gamboost_path,
    _frame_slice,
    _partition,
    _select_mstop_validation,
    paired_ci_halfwidth,
    partial_formula,
    run_arm_A,
    structural_scores,
    supplied_basis_floor,
    term_accounting,
)
from benches.bench_paired_estimation_discovery import report_text


def _fit(formula, data, mstop=20, nu=0.1):
    return mb.gamboost(formula, data=data,
                       control=mb.boost_control(mstop, nu=nu))


def test_joint_tree_accounting_seed_509_repro():
    """The #509 seed-509 repro: btree(x, z) must report both columns used.

    The term is named plain 'x' (no '(btree)' marker); identification must
    come from btree_splits() membership, not the name.
    """
    rng = np.random.default_rng(509)
    x, z = rng.normal(size=200), rng.normal(size=200)
    m = _fit("y ~ btree(x, z, maxdepth=2, selection=sse)",
             {"x": x, "z": z, "y": 2 * (x > 0) + 3 * (z > 0)})
    terms = term_accounting(m)
    assert len(terms) == 1
    t = terms[0]
    assert t["is_tree"] is True
    assert set(t["candidate_columns"]) == {"x", "z"}
    assert set(t["used_columns"]) == {"x", "z"}
    # numeric_cutpoint_count counts cutpoint records, nothing else
    info = m.btree_splits()[t["term"]]
    assert t["numeric_cutpoint_count"] == len(info["splits"])


def test_btree_identified_by_membership_not_name():
    """A single tree term carries no '(btree)' marker; still a tree."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    m = _fit("y ~ btree(x, maxdepth=2, selection=sse)",
             {"x": x, "y": 3 * (x > 0) + rng.normal(size=120)})
    names = m.base_learner_names()
    assert not any("(btree)" in n for n in names), names
    terms = term_accounting(m)
    assert terms[0]["is_tree"] is True


def test_categorical_level_splits_in_used_set():
    """Astra #509 item 3: level_split_variables join the used set.

    A categorical driver the tree actually splits on must be reported as
    used, not unused.
    """
    cat = np.array(["a", "b", "c"] * 40)
    y = np.tile([0.0, 2.0, 5.0], 40).astype(float)
    m = _fit("y ~ btree(g, maxdepth=2, selection=sse)",
             {"g": cat, "y": y}, mstop=5, nu=0.1)
    terms = term_accounting(m)
    assert len(terms) == 1
    t = terms[0]
    assert t["is_tree"] is True
    assert "g" in t["candidate_columns"]
    assert "g" in t["used_columns"]
    # no numeric cutpoints here; the count says what it counts
    assert t["numeric_cutpoint_count"] == 0


def test_mixed_numeric_categorical_used_set():
    """Mixed fit: both numeric and categorical split columns land in the
    used set; the candidate set stays complete."""
    rng = np.random.default_rng(11)
    n = 240
    x = rng.normal(size=n)
    xn = rng.normal(size=n)  # pure noise candidate
    cat = np.array(["a", "b", "c"] * 80)
    y = 2.0 * (x > 0) + np.where(cat == "b", 3.0, 0.0) + rng.normal(size=n)
    m = _fit("y ~ btree(x, g, xn, maxdepth=2, selection=sse)",
             {"x": x, "g": cat, "xn": xn, "y": y}, mstop=30)
    terms = term_accounting(m)
    t = terms[0]
    assert t["is_tree"] is True
    assert "x" in t["used_columns"]
    assert "g" in t["used_columns"]
    assert set(t["candidate_columns"]) == {"x", "g", "xn"}


def test_tree_terms_excluded_from_effect_recovery():
    """Structural columns are not effect proof.

    The accounting record marks tree terms so that any per-column effect
    recovery can exclude them; structural_scores returns recall/FDP over
    columns only and asserts no effect magnitudes.
    """
    rng = np.random.default_rng(509)
    x, z = rng.normal(size=200), rng.normal(size=200)
    m = _fit("y ~ btree(x, z, maxdepth=2, selection=sse)",
             {"x": x, "z": z, "y": 2 * (x > 0) + 3 * (z > 0)})
    terms = term_accounting(m)
    assert all(t["is_tree"] for t in terms)
    sc = structural_scores(terms, [frozenset({"x"}), frozenset({"z"})])
    assert set(sc) == {"recall", "fdp", "used_columns",
                       "n_driver_groups_recovered", "n_driver_groups"}
    assert sc["recall"] == 1.0


def test_scalar_learner_used_only_when_selected():
    """A scalar learner never selected reports empty used columns."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=100)
    xn = rng.normal(size=100)
    m = _fit("y ~ bols(x) + bols(xn)", {"x": x, "xn": xn, "y": 2 * x},
             mstop=5, nu=0.5)
    terms = {t["term"]: t for t in term_accounting(m)}
    # single learner per column: names are bare labels
    assert set(terms) == {"x", "xn"}
    assert terms["x"]["used_columns"] == ["x"]
    assert terms["x"]["is_tree"] is False
    assert terms["x"]["numeric_cutpoint_count"] is None
    # xn may or may not be selected at tiny mstop; either way the
    # candidate/used contract holds
    txn = terms["xn"]
    assert txn["candidate_columns"] == ["xn"]
    assert txn["used_columns"] == (["xn"] if txn["updates"] > 0 else [])


def test_brandom_term_reports_column_not_term_string():
    """brandom(g) records 'g' in used_columns, never the term string.

    Frozen-baseline defect (#511): _learner_columns returned
    'brandom(g, df=4.0)' as a pseudo-column, so linear_factors reported
    structural recall 0.75 / FDP 0.25 on every replicate -- an accounting
    artifact, not a discovery failure (it also misattributed rerun #2's
    x3-substitute pattern). Predictive gaps were unaffected.
    """
    rng = np.random.default_rng(11)
    g = rng.integers(0, 5, size=200).astype(str)
    y = (np.array([1.0, -2.0, 0.5, 3.0, -1.0])[rng.integers(0, 5, 200)]
         + rng.normal(scale=0.1, size=200))
    m = _fit("y ~ brandom(g, df=4)", {"g": g, "y": y}, mstop=30, nu=0.3)
    terms = term_accounting(m)
    assert len(terms) == 1
    t = terms[0]
    assert t["is_tree"] is False
    assert t["updates"] > 0
    assert t["candidate_columns"] == ["g"]
    assert t["used_columns"] == ["g"]
    # the invariant: no term strings ever leak into used columns
    for term in terms:
        for col in term["used_columns"]:
            assert "(" not in col and ")" not in col, col
    # end-to-end: the driver group is recovered with FDP 0
    sc = structural_scores(terms, [frozenset({"g"})])
    assert sc["recall"] == 1.0
    assert sc["fdp"] == 0.0

mb = pytest.importorskip("mboostrs")


def test_partitions_four_way_disjoint():
    """60/15/15/10 of n=1000: disjoint, covering, paired by construction."""
    tr, se, ca, te = _partition(N_TOTAL, 12345)
    assert (len(tr), len(se), len(ca), len(te)) == (600, 150, 150, 100)
    all_idx = np.concatenate([tr, se, ca, te])
    assert len(np.unique(all_idx)) == N_TOTAL == len(all_idx)
    # same seed -> identical partition (pairing across arms)
    tr2, _, _, _ = _partition(N_TOTAL, 12345)
    assert np.array_equal(np.sort(tr), np.sort(tr2))


def test_growth_triggers_from_tiny_cap():
    """The growth mechanism: a boundary minimum doubles the cap until interior."""
    spec = PANELS["interaction"]
    cols = spec["generate"](511000)
    tr, se, _, _ = _partition(len(cols["y"]), 511001)
    model = _fit_gamboost_path(spec["formula_A"], _frame_slice(cols, tr),
                               mstop_cap=40)
    m_opt, info = _select_mstop_validation(
        model, _frame_slice(cols, se),
        mstop_cap=40, mstop_ceiling=160, grow=True)
    assert info["growth_steps"] > 0
    assert info["final_cap"] > 40
    assert info["at_boundary"] is False
    assert m_opt < int(0.95 * info["final_cap"])
    # frozen behavior without growth: stuck at the boundary
    _, info2 = _select_mstop_validation(
        model, _frame_slice(cols, se),
        mstop_cap=40, mstop_ceiling=160, grow=False)
    assert info2["at_boundary"] is True
    assert info2["growth_steps"] == 0


def test_a_arm_growth_interior_on_interaction():
    """v2's truncation finding: the A arm must not sit at a fixed cap.

    v1.0's fixed-cap-1000 argmin truncated 3/8 interaction replicates;
    the grown arm records an interior minimum.
    """
    spec = PANELS["interaction"]
    cols = spec["generate"](511000)
    tr, se, _, te = _partition(len(cols["y"]), 511001)
    rec = run_arm_A(spec["formula_A"], cols, tr, se, te, seed=511000)
    assert rec["at_boundary"] is False
    assert rec["final_cap"] >= 1000
    assert np.isfinite(rec["test_mse"]) and rec["test_mse"] > 0


def test_interaction_panel_no_marginal_signal():
    """The hard discovery case by construction: the product term has no
    linear marginal signal in x0 or x1, so main-effects bases cannot
    capture it. The port must not change this."""
    cols = PANELS["interaction"]["generate"](777, noiseless=True)
    x0 = np.asarray(cols["x0"])
    x1 = np.asarray(cols["x1"])
    inter = 3.0 * (x0 - 0.5) * (x1 - 0.5)
    assert abs(np.corrcoef(inter, x0)[0, 1]) < 0.05
    assert abs(np.corrcoef(inter, x1)[0, 1]) < 0.05
    A = np.column_stack([x0, x1, np.ones_like(x0)])
    beta, *_ = np.linalg.lstsq(A, inter, rcond=None)
    r2 = 1 - np.var(inter - A @ beta) / np.var(inter)
    assert r2 < 0.01
    # the panel contract itself is unchanged from v1.0
    spec = PANELS["interaction"]
    assert "%X%" in spec["formula_A"]
    assert frozenset({"x0", "x1"}) in spec["joint_groups"]
    assert spec["sigma2"] == 0.49


def test_threshold_jump_floor_separates():
    """Evaluator-only floors: the partial basis pays for the jump, the
    supplied basis does not — the representation gap, a priori."""
    fs = supplied_basis_floor("threshold_jump", draws=2, mstop=1500)
    fp = supplied_basis_floor("threshold_jump",
                              formula=partial_formula("threshold_jump"),
                              draws=2, mstop=1500)
    assert fs["rmse"] < 0.05
    assert fp["rmse"] > 3 * fs["rmse"]
    assert fp["rmse"] > 0.05
    assert fs["n_rows"] == 2 * N_TOTAL
    # on a smooth task both bases are adequate: no such gap
    ss = supplied_basis_floor("smooth_additive", draws=2, mstop=1500)
    sp = supplied_basis_floor("smooth_additive",
                              formula=partial_formula("smooth_additive"),
                              draws=2, mstop=1500)
    assert sp["rmse"] < 2 * ss["rmse"] + 0.02


def test_floors_have_documented_keys():
    f = supplied_basis_floor("kinked", draws=1, mstop=100)
    assert set(f) == {"rmse", "n_rows", "mstop", "nu", "formula", "note"}
    assert f["n_rows"] == N_TOTAL
    assert np.isfinite(f["rmse"])


def test_report_renders_from_records():
    """The report renderer must survive failed arms, missing arms and
    boundary info (regression: a summary-shape change once crashed
    rendering after a completed battery)."""
    scales = {"interaction": 0.8784, "kinked": 1.3683}
    records = [
        {"panel": "interaction", "replicate": 0, "seed": 511000,
         "instrument": "paired-ab/v1.1",
         "partition_digest": "abc",
         "arms": {
             "A_supplied": {"test_mse": 0.55, "mstop": 70,
                            "at_boundary": False, "final_cap": 1000,
                            "growth_steps": 0, "terms": []},
             "A_partial": {"test_mse": 1.40, "mstop": 120,
                           "at_boundary": False, "final_cap": 1000,
                           "growth_steps": 0, "terms": []},
             "B_discovery": {"test_mse": 0.54, "mstop": 90, "terms": [],
                             "structural": {"recall": 1.0, "fdp": 0.0,
                                            "used_columns": ["x0", "x1"],
                                            "n_driver_groups_recovered": 3,
                                            "n_driver_groups": 3}},
         },
         "failures": {}},
        {"panel": "kinked", "replicate": 0, "seed": 511000,
         "instrument": "paired-ab/v1.1",
         "partition_digest": "def",
         "arms": {
             "A_supplied": {"test_mse": 0.30, "mstop": 3200,
                            "at_boundary": True, "final_cap": 3200,
                            "growth_steps": 2, "terms": []},
         },
         "failures": {"B_discovery": "RuntimeError: boom"}},
    ]
    text = report_text(records, scales)
    assert "paired-ab/v1.1" in text
    assert "boundary" in text
    assert "precision contract" in text
    assert "arm failures: 1" in text
    # boundary rate picks up the ceiling hit
    assert "rate=1.00" in text


def test_paired_ci_halfwidth_contract():
    """The precision contract's unit: half-width shrinks with replicates,
    NaN when there is nothing to resolve."""
    assert np.isnan(paired_ci_halfwidth([0.05]))
    hw8 = paired_ci_halfwidth([0.01] * 8)
    assert hw8 == 0.0  # no variation -> nothing to resolve
    rng = np.random.default_rng(0)
    hw = paired_ci_halfwidth(rng.normal(0, 0.1, size=8))
    assert 0 < hw < 0.2
