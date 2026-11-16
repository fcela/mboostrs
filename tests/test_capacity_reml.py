"""Opt-in direct REML candidates in capacity selection."""

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.explore import _folds
from mboostrs._frame import take_rows


def test_smooth_uses_reml_when_boosting_budget_is_short():
    rng = np.random.default_rng(1)
    n = 250
    x = rng.uniform(-3, 3, n)
    z = rng.normal(size=n)
    data = pl.DataFrame({"x": x, "z": z,
                         "y": np.sin(x) + 0.2 * z + rng.normal(0, 0.1, n)})
    pick = mb.select_capacity(
        "y ~ bbs(x, df=6) + bbs(z, df=4)", data,
        estimators=("boosting", "reml"), folds=4, mstop=1, pairs=0, tree=False)
    assert pick.selected == "base"
    assert pick.selected_estimator == "reml"
    assert pick.to_dict()["mstop"] is None
    assert all(row["estimator"] in {"boosting", "reml"} for row in pick.rungs)


def test_reml_only_menu_still_scores_boosted_intercept():
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    data = pl.DataFrame({"x": x, "y": 2.0 + rng.normal(size=x.size)})
    pick = mb.select_capacity("y ~ bbs(x)", data, estimators=("reml",),
                              pairs=0, tree=False, folds=4)
    intercept = next(row for row in pick.rungs if row["label"] == "intercept")
    assert intercept["estimator"] == "boosting"


def test_three_way_interaction_uses_boosted_tree():
    rng = np.random.default_rng(10)
    n = 600
    x1, x2, x3 = (rng.uniform(-2, 2, n) for _ in range(3))
    y = 2 * np.prod([(x > 0) * 2 - 1 for x in (x1, x2, x3)], axis=0)
    data = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    pick = mb.select_capacity(
        "y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
        estimators=("boosting", "reml"), folds=5, mstop=500, tree_depth=3)
    assert pick.selected == "+tree"
    assert pick.selected_estimator == "boosting"
    assert not any(r["label"] == "+tree" and r["estimator"] == "reml"
                   for r in pick.rungs)


def test_null_response_keeps_intercept_baseline_and_repeats_json():
    rng = np.random.default_rng(15)
    x = rng.normal(size=300)
    data = pl.DataFrame({"x": x, "y": rng.normal(size=x.size)})
    kwargs = dict(estimators=("boosting", "reml"), folds=5, mstop=200,
                  pairs=0, tree=False)
    one = mb.select_capacity("y ~ bbs(x)", data, **kwargs)
    two = mb.select_capacity("y ~ bbs(x)", data, **kwargs)
    assert one.selected == "intercept"
    assert one.selected_estimator == "boosting"
    assert one.to_dict() == two.to_dict()
    json.dumps(one.to_dict())


def test_reml_cv_matches_manual_fold_fits_with_weights_offset_and_imputation():
    rng = np.random.default_rng(31)
    n = 30
    x = np.linspace(0, 1, n)
    x[5] = np.nan
    offset = np.linspace(-0.2, 0.2, n)
    weights = np.linspace(0.5, 1.5, n)
    y = np.sin(np.nan_to_num(x, nan=0.5)) + offset + rng.normal(0, 0.03, n)
    data = pl.DataFrame({"x": x, "y": y})
    base = np.column_stack([weights * (np.arange(n) < 20),
                            weights * ((np.arange(n) >= 10) & (np.arange(n) < 30))])
    held = np.column_stack([np.arange(n) >= 20, np.arange(n) < 10])
    judge = held.copy()
    judge[25:28, 0] = False  # structured unused rows are not scored
    scheme = mb.FoldScheme(weights=base, held_out=held, judge=judge,
                           base_weights=weights, scheme="manual rolling-like")
    result = mb.cv_risk("bbs(x, df=5)", data, "y", folds=scheme,
                        estimator="reml", score="mse", weights=weights,
                        na_action="impute", offset=offset, return_result=True)
    manual = []
    for fit_mask, _held_mask, fold_weights in _folds(n, scheme, 0, weights):
        # _folds already returns complete training weights and judge membership.
        b = len(manual)
        held_mask = judge[:, b]
        model = mb.gam("y ~ bbs(x, df=5)", data=take_rows(data, np.flatnonzero(fit_mask)),
                       weights=fold_weights[fit_mask], offset=offset[fit_mask],
                       na_action="impute")
        pred = model.predict(take_rows(data, np.flatnonzero(held_mask)),
                             type="link", offset=offset[held_mask])
        wh = weights[held_mask]
        manual.append(float(np.average((y[held_mask] - pred) ** 2, weights=wh)))
    np.testing.assert_allclose(result.fold_scores, manual, rtol=1e-10, atol=1e-12)
    assert all(row["estimator"] == "reml" and row["mstop"] is None
               for row in result.folds)
    assert result.folds[0]["n_evaluation"] == int(judge[:, 0].sum())


@pytest.mark.parametrize("kwargs,match", [
    ({"score": "brier"}, "score='family' or 'mse'"),
    ({"family": mb.Poisson()}, "Gaussian only"),
])
def test_reml_rejects_unsupported_comparison(kwargs, match):
    data = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match=match):
        mb.select_capacity("y ~ bbs(x)", data, estimators=("reml",), **kwargs)


def test_nonconvergent_reml_folds_are_recorded_and_cannot_win(monkeypatch):
    import mboostrs.direct_gam as direct_gam

    monkeypatch.setattr(direct_gam, "gam", lambda *a, **k: SimpleNamespace(
        convergence={"converged": False, "stop_reason": "iteration_limit"}))
    rng = np.random.default_rng(8)
    x = rng.normal(size=100)
    data = pl.DataFrame({"x": x, "y": x + rng.normal(0, 0.1, x.size)})
    pick = mb.select_capacity("y ~ bbs(x)", data, estimators=("boosting", "reml"),
                              folds=4, mstop=200, pairs=0, tree=False)
    reml = next(row for row in pick.rungs
                if row["label"] == "base" and row["estimator"] == "reml")
    assert np.isnan(reml["score"])
    assert "did not converge" in reml["error"]
    assert pick.selected_estimator == "boosting"


def test_nonconvergent_final_reml_refit_is_refused(monkeypatch):
    import mboostrs.direct_gam as direct_gam

    real_gam = direct_gam.gam
    monkeypatch.setattr(direct_gam, "gam", real_gam)
    monkeypatch.setattr(mb, "gam", lambda *a, **k: SimpleNamespace(
        convergence={"converged": False, "stop_reason": "iteration_limit"}))
    rng = np.random.default_rng(9)
    x = rng.uniform(-1, 1, 120)
    data = pl.DataFrame({"x": x, "y": np.sin(3 * x) + rng.normal(0, 0.05, x.size)})
    with pytest.raises(ValueError, match="selected direct REML model did not converge"):
        mb.select_capacity("y ~ bbs(x)", data, estimators=("reml",),
                           folds=4, pairs=0, tree=False)


def test_reml_cv_honors_unseen_factor_policy():
    from mboostrs.folds import FoldScheme

    data = pl.DataFrame({"group": ["A"] * 4 + ["B"] * 4 + ["C"],
                         "y": [0.0] * 4 + [1.0] * 4 + [2.0]})
    weights = np.ones((9, 2))
    weights[:, 0] = 0
    weights[:8, 0] = 1
    weights[:, 1] = 1
    weights[0, 1] = 0
    held = weights == 0
    scheme = FoldScheme(weights=weights, held_out=held, scheme="unseen level")
    average = mb.cv_risk("bols(group)", data, "y", folds=scheme,
                         estimator="reml", score="mse", unseen_levels="average",
                         return_result=True)
    assert np.isfinite(average.score)
    with pytest.raises(ValueError, match="unseen|level"):
        mb.cv_risk("bols(group)", data, "y", folds=scheme,
                   estimator="reml", score="mse", unseen_levels="error",
                   return_result=True)
