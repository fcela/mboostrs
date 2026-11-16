"""select_capacity: the capacity ladder keeps only interaction terms that pay.

The tests plant three response shapes and check the rung each one selects:

* a smooth additive response keeps the base formula (the extra capacity does
  not pay two standard errors, which is what the cancer panel of the
  competitive study showed with a joint tree that was always on);
* a bilinear response keeps the screened tensor rung;
* a smooth-times-step interaction -- structure a tensor's bilinear surface
  cannot make but a joint tree makes in two splits -- keeps the tree rung.
"""

import json

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _frame(seed, n=500):
    rng = np.random.default_rng(seed)
    return rng, pl.DataFrame({
        "x1": rng.uniform(-2, 2, n),
        "x2": rng.uniform(-2, 2, n),
        "x3": rng.uniform(-2, 2, n),
        "z": rng.normal(size=n),
    })


def test_smooth_response_keeps_base():
    rng, data = _frame(11)
    y = (np.sin(data["x1"].to_numpy())
         + 0.8 * data["x2"].to_numpy() ** 2
         + 0.5 * rng.normal(size=data.height))
    data = data.with_columns(y=y)
    pick = mb.select_capacity(
        "y ~ bbs(x1, df=8) + bbs(x2, df=6) + bbs(x3)", data,
        mstop=300, pairs=2)
    assert pick.selected == "base"
    # The tuned budget: the rung is scored and refitted at the step count its
    # own held-out score turns at, not at the shared 300.
    assert pick.model.mstop == pick.rungs[0]["mstop"] < 300
    assert pick.rungs[0]["score"] == pytest.approx(
        mb.cv_risk(pick.rungs[0]["rhs"], data, "y", mstop=pick.model.mstop))
    labels = [row["label"] for row in pick.rungs]
    assert labels[0] == "base"
    assert pick.rungs[0]["selected"] is True


def test_bilinear_response_adds_tensors():
    rng, data = _frame(12)
    y = (1.5 * data["x1"].to_numpy() * data["x2"].to_numpy()
         + 0.3 * data["x3"].to_numpy()
         + 0.25 * rng.normal(size=data.height))
    data = data.with_columns(y=y)
    pick = mb.select_capacity(
        "y ~ bbs(x1) + bbs(x2) + bbs(x3)", data, mstop=300)
    assert pick.selected == "+tensors"
    assert pick.screened_pairs
    assert pick.screened_pairs[0]["left"] in {"x1", "x2"}
    tensor_rung = next(r for r in pick.rungs if r["label"] == "+tensors")
    assert "%X%" in tensor_rung["rhs"]
    assert tensor_rung["selected"] is True
    base_rung = pick.rungs[0]
    assert base_rung["score"] > tensor_rung["score"]


def test_step_interaction_adds_tree():
    # 2 sin(x1) 1(x2 > 0): the tensor rung cannot make the step in its x2
    # margin, the joint tree makes it in two splits, so the tree rung is
    # adopted decisively.
    rng, data = _frame(13, n=800)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    y = (2.0 * np.sin(x1) * (x2 > 0)
         + 0.4 * data["x3"].to_numpy()
         + 0.3 * rng.normal(size=data.height))
    data = data.with_columns(y=y)
    pick = mb.select_capacity(
        "y ~ bbs(x1) + bbs(x2) + bbs(x3)", data, mstop=300, tree_depth=3)
    assert pick.selected == "+tree"
    tree_rung = next(r for r in pick.rungs if r["label"] == "+tree")
    assert "btree(x1, x2, x3" in tree_rung["rhs"]
    assert tree_rung["selected"] is True


def test_zero_noise_free_data_selects_highest_paying_rung():
    # With no noise the quadrant structure is exact, so the tree rung's
    # advantage is far beyond two standard errors.
    rng, data = _frame(14)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=1.5 * ((x1 > 0) == (x2 > 0)))
    pick = mb.select_capacity(
        "y ~ bbs(x1) + bbs(x2)", data, mstop=300)
    assert pick.selected == "+tree"


def test_pure_noise_proposes_no_pair_and_keeps_base():
    rng, data = _frame(15)
    data = data.with_columns(y=rng.normal(size=data.height))
    pick = mb.select_capacity(
        "y ~ bbs(x1) + bbs(x2) + bbs(x3)", data, mstop=200, pairs=2)
    assert pick.selected == "base"
    # No pair removed held-out variation, so the tensor rung is not built.
    assert [row["label"] for row in pick.rungs] == ["base", "+tree"]
    assert pick.screened_pairs == ()


def test_same_seed_repeats_exactly():
    rng, data = _frame(16)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=x1 * x2 + 0.3 * rng.normal(size=data.height))
    first = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=250)
    second = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=250)
    assert first.selected == second.selected
    assert [r["score"] for r in first.rungs] == [r["score"] for r in second.rungs]


def test_result_record_is_json_safe():
    rng, data = _frame(17)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=x1 * x2 + 0.3 * rng.normal(size=data.height))
    pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=200)
    record = pick.to_dict()
    assert record["selected"] == pick.selected
    assert json.dumps(record)  # every value is JSON-safe


def test_pairs_zero_and_no_tree_refuses():
    rng, data = _frame(18)
    data = data.with_columns(y=rng.normal(size=data.height))
    with pytest.raises(ValueError, match="no rung"):
        mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data,
                           mstop=100, pairs=0, tree=False)


def test_composed_response_is_refused():
    rng, data = _frame(19)
    data = data.with_columns(y=rng.normal(size=data.height))
    with pytest.raises(ValueError, match="plain response"):
        mb.select_capacity("Surv(time, status) ~ bbs(x1)", data, mstop=100)


def test_extrapolation_argument_reaches_the_folds():
    rng, data = _frame(20)
    data = data.with_columns(y=data["x1"].to_numpy() * data["x2"].to_numpy())
    pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data,
                              mstop=150, extrapolation="clip")
    assert pick.selected in {"base", "+tensors", "+tree"}


def test_factor_interaction_selects_tree():
    # A threshold whose location depends on a string factor: the additive
    # base (a smooth in x plus level shifts) cannot fit it, a numeric-only
    # tree cannot split on the factor, but a joint tree over all formula
    # columns can. The shootout rerun missed hour x working-day on the bike
    # panels for exactly this reason (#493).
    rng = np.random.default_rng(21)
    n = 800
    x = rng.uniform(-2, 2, n)
    regime = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    threshold = {"a": -1.0, "b": 0.0, "c": 1.0}
    y = (2.5 * (x > np.array([threshold[r] for r in regime]))
         + 0.4 * rng.normal(size=n))
    data = pl.DataFrame({"x": x, "regime": regime, "y": y})
    pick = mb.select_capacity("y ~ bbs(x, df=8) + bols(regime)", data,
                              mstop=400)
    assert pick.selected == "+tree", pick.selected
    tree_rung = next(r for r in pick.rungs if r["label"] == "+tree")
    assert "regime" in tree_rung["rhs"], tree_rung["rhs"]


def test_binary_column_cannot_be_a_tensor_margin():
    # A 0/1 column in a real interaction: the screen proposes the pair, but
    # a binary column's spline boundary knots collapse (the rerun crashed
    # on this, #493), so the guard must drop it from the tensor rung without
    # an exception -- the tree rung still covers the structure.
    rng = np.random.default_rng(22)
    n = 600
    x = rng.uniform(-2, 2, n)
    flag = rng.integers(0, 2, n).astype(float)
    y = (3.0 * flag * (x > 0.0) + 0.5 * x + 0.3 * rng.normal(size=n))
    data = pl.DataFrame({"x": x, "flag": flag, "y": y})
    pick = mb.select_capacity("y ~ bbs(x, df=8) + bols(flag)", data,
                              mstop=400)
    assert pick.selected in {"+tree", "+tensors"}, pick.selected
    for pair in pick.screened_pairs:
        assert "flag" not in (pair["left"], pair["right"]), pair
    assert all(np.isfinite(r["score"]) for r in pick.rungs)


def _slow_tree_frame(seed=13, n=1200):
    """The panel of #493's third gap: a fine interaction the joint tree
    absorbs slowly, so its held-out score is still falling at a small cap
    and only a budget search past that cap reveals its advantage."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.uniform(-2, 2, n) for _ in range(3))
    y = (1.5 * np.sin(6 * x1) * (x2 > 0) + 0.4 * x3
         + 0.3 * rng.normal(size=n))
    return pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def test_tuned_budget_adopts_a_rung_the_shared_cap_hides():
    # #493: every rung used to be scored at one shared mstop, and on this
    # panel the tree rung is still far from its optimum there. Under the
    # paired adoption rule of #500 its advantage -- real in every fold, since
    # the panel plants the interaction -- is visible even at the truncated
    # budget, so +tree is adopted under both budgets; the budget search is
    # what finds the rung's real optimum past the cap.
    data = _slow_tree_frame()
    formula = "y ~ bbs(x1) + bbs(x2) + bbs(x3)"
    shared = mb.select_capacity(formula, data, mstop=25, nu=0.02,
                                budget="shared")
    assert shared.selected == "+tree"
    with pytest.warns(RuntimeWarning, match="still falling"):
        tuned = mb.select_capacity(formula, data, mstop=25, nu=0.02)
    assert tuned.selected == "+tree"
    shared_tree = next(r for r in shared.rungs if r["label"] == "+tree")
    tree_rung = next(r for r in tuned.rungs if r["label"] == "+tree")
    # The shared cap scored the tree before its advantage matured; the tuned
    # budget searched past the cap (25 steps) to the rung's own optimum.
    assert shared_tree["mstop"] == 25
    assert tree_rung["mstop"] > 25
    assert tuned.model.mstop == tree_rung["mstop"]
    assert tree_rung["score"] < shared_tree["score"]


def test_shared_budget_keeps_the_equal_step_contract():
    # budget="shared" reproduces the original comparison exactly: every
    # boosting rung scored at mstop by cv_risk, on the shared folds.
    rng, data = _frame(23)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=x1 * x2 + 0.3 * rng.normal(size=data.height))
    pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data,
                              mstop=150, budget="shared")
    for row in pick.rungs:
        assert row["estimator"] == "boosting"
        assert row["mstop"] == 150
        assert row["n_folds"] == 5
        assert row["score"] == pytest.approx(
            mb.cv_risk(row["rhs"], data, "y", mstop=150))


def test_tuned_scores_never_exceed_the_shared_ones():
    # The tuned budget searches a grid that contains the shared mstop, so a
    # rung's tuned score is its best point of that grid and never worse than
    # its score at mstop itself. Checked on a response score, which the grid
    # route scores with the same losses a single-mstop call uses.
    rng, data = _frame(24)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=x1 * x2 + 0.3 * rng.normal(size=data.height))
    shared = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=200,
                                budget="shared", score="mse")
    tuned = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=200,
                               score="mse")
    by_rhs = {row["rhs"]: row for row in shared.rungs}
    for row in tuned.rungs:
        assert isinstance(row["mstop"], int)
        # Same folds, same route: the tuned grid contains the shared mstop,
        # so the tuned score is exact at the points and never above it.
        assert row["score"] <= by_rhs[row["rhs"]]["score"] + 1e-12


def test_tree_depth_sequence_scores_one_rung_per_depth():
    # tree_depth=(2, 4) is the depth search the shootout battery asked for
    # (#493): one +tree rung per depth, ordered as given, each searched for
    # its own budget, the smallest depth within tolerance kept.
    rng, data = _frame(25)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=2.0 * np.sin(x1) * (x2 > 0)
                             + 0.3 * rng.normal(size=data.height))
    pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
                              mstop=300, tree_depth=(2, 4))
    tree_rows = [r for r in pick.rungs if r["label"] == "+tree"]
    assert ["maxdepth=2" in r["rhs"] for r in tree_rows] == [True, False]
    assert "maxdepth=4" in tree_rows[1]["rhs"]
    assert all(np.isfinite(r["score"]) for r in tree_rows)
    if pick.selected == "+tree":
        selected = next(r for r in tree_rows if r["selected"])
        assert "maxdepth=" in selected["rhs"]


def test_tuned_falls_back_to_the_shared_budget_with_a_warning():
    # The grid route cannot score a per-row offset, so the call says so once
    # and compares the rungs at the shared mstop, exactly as budget="shared".
    rng, data = _frame(26)
    x1, x2 = data["x1"].to_numpy(), data["x2"].to_numpy()
    data = data.with_columns(y=x1 * x2 + 0.3 * rng.normal(size=data.height))
    with pytest.warns(RuntimeWarning, match="budget='tuned' cannot search"):
        pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=150,
                                  offset=np.ones(data.height))
    shared = mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=150,
                               budget="shared", offset=np.ones(data.height))
    assert [r["score"] for r in pick.rungs] == [r["score"] for r in shared.rungs]
    assert all(r["mstop"] == 150 for r in pick.rungs)


def test_max_mstop_below_mstop_is_refused():
    rng, data = _frame(27)
    data = data.with_columns(y=rng.normal(size=data.height))
    with pytest.raises(ValueError, match="max_mstop"):
        mb.select_capacity("y ~ bbs(x1) + bbs(x2)", data, mstop=100,
                           max_mstop=50)


def _period_panel(seed, amplitude, n=1200):
    """Three time-ordered blocks whose noise scales differ, with a planted
    interaction of the given amplitude.

    Every rung scores far worse on the loud middle block, so each fold's
    marginal spread is large -- the setting of #500 -- while the folds share
    that period effect, which the paired fold difference between rungs
    cancels."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.uniform(-1, 1, n) for _ in range(3))
    period = np.arange(n) // (n // 3)
    sd = np.where(period == 1, 1.4, 0.35)
    y = (0.8 * np.sin(2 * np.pi * x1) + 0.4 * x3 + amplitude * np.sign(x1 * x2)
         + sd * rng.normal(size=n))
    return pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def _rolling(n=1200):
    from mboostrs.folds import rolling_folds

    return rolling_folds(n, B=3, initial=300, horizon=300)


def test_paired_tolerance_adopts_a_gain_the_marginal_bar_hides():
    # #500: on few time-ordered folds the best rung's marginal fold spread is
    # dominated by the periods the folds cover, so the old two-standard-error
    # bar sat far above any real capacity gain. Here the planted interaction's
    # gain is an order of magnitude below that old bar -- the marginal rule
    # kept the base -- while the paired fold difference against the best rung,
    # which cancels the shared period effect, adopts it.
    data = _period_panel(30, 0.5)
    pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
                              mstop=250, folds=_rolling(), budget="shared")
    assert pick.selected in {"+tensors", "+tree"}, pick.selected
    rows = {row["label"]: row for row in pick.rungs}
    best = min(rows.values(), key=lambda row: row["score"])
    # The old rule's bar: the best rung's marginal spread, which this gain
    # never cleared.
    assert 2 * best["fold_se"] > rows["base"]["score"] - best["score"]


def test_paired_tolerance_keeps_base_within_paired_noise():
    # The honest no-adoption of the PM2.5 panel: a smaller interaction that
    # no consistent per-fold gain carries stays within the paired bar, and
    # the base is kept -- the paired rule sharpens the threshold without
    # adopting noise.
    data = _period_panel(30, 0.3)
    pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
                              mstop=250, folds=_rolling(), budget="shared")
    assert pick.selected == "base"


def test_rungs_carry_per_fold_scores():
    # Every rung record carries each fold's score at the step count the rung
    # was scored at, on both budget routes, so a reader can recompute the
    # paired differences the tolerance reads (#500).
    data = _period_panel(31, 0.5)
    for budget in ("shared", "tuned"):
        pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
                                  mstop=120, folds=_rolling(), budget=budget)
        for row in pick.rungs:
            assert len(row["fold_scores"]) == row["n_folds"] == 3
            assert np.all(np.isfinite(row["fold_scores"]))
        # The mean of a rung's per-fold scores is its recorded score.
        for row in pick.rungs:
            assert np.mean(row["fold_scores"]) == pytest.approx(row["score"])
