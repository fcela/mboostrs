"""Stopping metrics and shared, fold-local validation designs."""

import numpy as np
import pytest

import mboostrs as mb
import mboostrs.explore as explore
from mboostrs import _rust
from mboostrs.policy import _policy_folds


def data(n=90):
    rng = np.random.default_rng(33)
    x = rng.uniform(-1, 1, n)
    return {"x": x, "z": rng.normal(size=n), "y": 2 * x + rng.normal(size=n)}


@pytest.mark.parametrize("score,tau", [("mse", None), ("pinball", .2), ("expectile", .8)])
def test_score_path_matches_independent_fold_predictions(score, tau):
    frame = data()
    weights = np.linspace(.5, 2, 90)
    tuned = mb.tune_mstop("y ~ bbs(x, knots=4, df=3)", frame, folds=3, start=20,
                          extensions=0, weights=weights, score=score, score_parameter=tau)
    oracle = []
    for fitting, held, train_weights in explore._folds(90, 3, 0, weights):
        train = {k: v[fitting] for k, v in frame.items()}
        test = {k: v[held] for k, v in frame.items()}
        fit = mb.gamboost("y ~ bbs(x, knots=4, df=3)", train, weights=train_weights[fitting],
                          control=mb.boost_control(20, .1))
        curve = []
        for stop in range(21):
            residual = test["y"] - fit.predict(test, mstop=stop, type="response")
            if score == "mse":
                loss = residual**2
            elif score == "pinball":
                loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual)
            else:
                loss = np.where(residual >= 0, tau, 1 - tau) * residual**2
            curve.append(np.average(loss, weights=weights[held]))
        oracle.append(curve)
    np.testing.assert_allclose(tuned.fold_scores, oracle, rtol=1e-11, atol=1e-12)
    mean = np.mean(oracle, axis=0)
    assert tuned.mstop == 1 + np.argmin(mean[1:])
    assert tuned.score == score and tuned.score_parameter == tau
    assert tuned.total_fits == 4


@pytest.mark.parametrize("family", [mb.Binomial(), mb.AdaExp(), mb.HingeLoss()])
def test_binary_streamed_loss_matches_public_prediction(family):
    frame = data()
    frame["y"] = np.where(frame["y"] > 0, 1., -1.)
    fit = mb.gamboost("y ~ bols(x)", frame, family=family, control=mb.boost_control(15))
    observed = (frame["y"] + 1) / 2
    path = fit._response_score_path(observed, np.ones(90), frame, "log_loss")
    oracle = []
    for stop in range(16):
        p = np.clip(fit.predict(frame, type="response", mstop=stop), np.finfo(float).tiny,
                    1 - np.finfo(float).eps)
        oracle.append(np.mean(-observed * np.log(p) - (1 - observed) * np.log1p(-p)))
    np.testing.assert_allclose(path, oracle, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("formula", ["y ~ bbs(x) + bols(z)",
                                     "y ~ bbs(x, knots=3) %X% bbs(z, knots=3)",
                                     "y ~ btree(x, z, maxdepth=2)"])
def test_streamed_path_supports_splines_tensors_and_trees(formula):
    frame = data()
    fit = mb.gamboost(formula, frame, control=mb.boost_control(40))
    path = fit._response_score_path(frame["y"], np.ones(90), frame, "mse")
    for stop in (0, 1, 20, 40):
        expected = np.mean((frame["y"] - fit.predict(frame, mstop=stop))**2)
        assert path[stop] == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_learned_predictor_transforms_are_refitted_within_each_fold():
    frame = data()
    tuned = mb.tune_mstop("y ~ bols(scale(x))", frame, folds=3, start=12,
                          extensions=0, score="mse")
    # The public fixed-step evaluation independently recreates every recipe.
    for stop in (1, 6, 12):
        reference = mb.cv_risk("bols(scale(x))", frame, "y", folds=3, mstop=stop,
                               score="mse", return_result=True)
        np.testing.assert_allclose(np.asarray(tuned.fold_scores)[:, stop], reference.fold_scores,
                                   rtol=1e-11, atol=1e-12)


def test_rolling_judge_masks_and_unused_rows_are_preserved():
    frame = data()
    folds = mb.rolling_folds(90, B=2, initial=30, horizon=10)
    first = mb.tune_mstop("y ~ bols(x)", frame, folds=folds, start=10, extensions=0, score="mse")
    unused = ~(np.any(folds.weights > 0, axis=1) | np.any(folds.held_out, axis=1))
    changed = {k: v.copy() for k, v in frame.items()}
    changed["y"][unused] = 1e6
    second = mb.tune_mstop("y ~ bols(x)", changed, folds=folds, start=10, extensions=0, score="mse")
    np.testing.assert_array_equal(first.fold_scores, second.fold_scores)


def test_nested_tuning_uses_requested_metric_and_training_weights(monkeypatch):
    from mboostrs import _metric_tuning

    frame = data()
    folds = mb.grouped_folds(np.repeat(np.arange(9), 10), B=3, weights=np.linspace(.5, 2, 90))
    seen = []
    original = _metric_tuning.tune_response_mstop

    def record(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(_metric_tuning, "tune_response_mstop", record)
    monkeypatch.setattr(explore, "_MAX_TUNED_MSTOP", 8)
    result = mb.cv_risk("bols(x)", frame, "y", folds=folds, mstop="cv",
                        score="pinball", score_parameter=.2, return_result=True)
    assert len(seen) == 3
    assert all(k["score"] == "pinball" and k["score_parameter"] == .2 for k in seen)
    assert all(np.array_equal(k["folds"].base_weights, k["weights"]) for k in seen)
    assert result.fit_count == 12


def test_policy_materializes_one_scheme_and_holds_each_declared_group():
    frame = data()
    frame["a"] = np.repeat(np.arange(9), 10)
    frame["b"] = np.repeat(np.arange(9), 10) + 100
    folds = _policy_folds(frame, "y", None, ("a", "b"), 4, None)
    for train, held, _ in explore._folds(90, folds, 4):
        for name in ("a", "b"):
            assert not set(frame[name][train]) & set(frame[name][held])
    assert _policy_folds(frame, "y", folds, ("a",), 5, None) is folds


def test_connected_crossed_groups_are_never_split():
    codes = np.array([[0., 0.], [0., 1.], [1., 1.], [2., 2.]])
    assert _rust.group_components(codes) == [0, 0, 0, 1]


def test_policy_propagates_metric_parameter_and_exact_fold_object(monkeypatch):
    import mboostrs.search_session as actions

    folds = _policy_folds(data(), "y", 3, (), 0, None)
    calls = []
    original = actions.tune_mstop

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        kwargs.update(extensions=0)
        return original(*args, **kwargs)

    monkeypatch.setattr(actions, "tune_mstop", record)
    result = mb.auto_policy(data(), "y", folds=folds, metric="pinball", score_parameter=.2,
                            family=mb.Quantile(.2), mstop=12, complexity="simple")
    assert calls and all(c["folds"] is folds and c["score"] == "pinball"
                         and c["score_parameter"] == .2 for c in calls)
    assert result.tuning.score == "pinball"


def test_score_parameter_is_part_of_cache_identity():
    cache = mb.FitCache()
    args = dict(rhs="bols(x)", data=data(), family=mb.Quantile(.2), mstop=10, folds=3,
                metric="pinball")
    assert cache.key(**args, score_parameter=.2) != cache.key(**args, score_parameter=.8)


def test_multiclass_tuning_and_maximum_path_bound():
    frame = data()
    frame["y"] = np.tile(["a", "b", "c"], 30)
    folds = mb.stratified_folds(frame["y"], B=3)
    tuned = mb.tune_mstop("y ~ bols(x)", frame, family=mb.Multinomial(), folds=folds,
                          score="log_loss", start=100, max_mstop=8)
    assert tuned.largest == 8 and 1 <= tuned.mstop <= 8
    assert np.asarray(tuned.fold_scores).shape == (3, 9)


def test_small_ordered_inner_scheme_is_refused_instead_of_randomized():
    folds = mb.rolling_folds(20, B=2, initial=8, horizon=4)
    with pytest.raises(ValueError, match="ordered inner"):
        explore._inner_folds(folds, folds.weights[:, 0] > 0, 0)


def test_requested_metric_changes_stopping_for_an_overdispersed_count_sample():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 150)
    y = rng.poisson(np.exp(x)).astype(float)
    y[0] += 30
    frame = {"x": x, "y": y}
    options = dict(family=mb.Poisson(), folds=3, start=120, extensions=0, nu=.03)
    family = mb.tune_mstop("y ~ bols(x)", frame, score="family", fold_local=True, **options)
    mse = mb.tune_mstop("y ~ bols(x)", frame, score="mse", **options)
    assert mse.mstop < family.mstop
    assert mse.risk[mse.mstop] < mse.risk[family.mstop]


def test_row_offsets_use_the_same_fold_slices_in_tuning_and_scoring():
    frame = data()
    offset = np.linspace(-.5, .5, 90)
    tuned = mb.tune_mstop("y ~ bols(x)", frame, score="mse", folds=3,
                          start=12, extensions=0, offset=offset)
    for stop in (1, 6, 12):
        result = mb.cv_risk("bols(x)", frame, "y", score="mse", folds=3,
                            mstop=stop, offset=offset, return_result=True)
        np.testing.assert_allclose(np.asarray(tuned.fold_scores)[:, stop], result.fold_scores,
                                   rtol=1e-11, atol=1e-12)


def test_declared_groups_reach_every_policy_stage(monkeypatch):
    import mboostrs.search_session as actions

    frame = data()
    frame["g"] = np.repeat(np.arange(9), 10)
    seen = []
    discovery_fits = []
    original = actions.autospec

    def discovery(*args, **kwargs):
        seen.append(kwargs.get("folds"))
        recommendation = original(*args, **kwargs)
        discovery_fits.append(recommendation.fits)
        return recommendation

    monkeypatch.setattr(actions, "autospec", discovery)
    result = mb.auto_policy(frame, "y", groups="g", mstop=10, complexity="simple")
    assert seen and all(isinstance(folds, mb.FoldScheme) for folds in seen)
    assert all(folds is seen[0] for folds in seen)
    assert discovery_fits[0] == 0 and any(count > 0 for count in discovery_fits[1:])
    assert "brandom(g)" in result.spec
