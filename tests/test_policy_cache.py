"""Specification cache isolation and accounting across policy calls."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import mboostrs as mb
import mboostrs.policy as policy
import mboostrs.search_session as actions


def _request(**changes):
    request = dict(rhs="bols(x)", data={"x": np.arange(8.), "y": np.arange(8.)},
                   response="y", family=mb.Gaussian(), mstop=10, folds=2)
    request.update(changes)
    return request


def test_native_build_identity_is_used():
    assert policy.FitCache().build == mb.build_info()["build_id"]


def test_equal_configuration_reuses_a_value():
    cache = policy.FitCache()
    first = cache.key(**_request())
    second = cache.key(**_request())
    assert first is not None and first == second
    value = object()
    assert cache.get_or_run(first, lambda: value) is value
    assert cache.get_or_run(second, lambda: pytest.fail("unexpected refit")) is value
    assert (cache.hits, cache.misses) == (1, 1)


@pytest.mark.parametrize("change", [
    {"seed": 1}, {"response": "x"}, {"metric": "family"},
    {"na_action": "impute"}, {"unseen_levels": "average"},
    {"rhs": "bbs(x)"}, {"mstop": 20}, {"folds": 4},
    {"weights": np.arange(1., 9.)},
    {"transform": {"center": 2.}}, {"fit_kwargs": {"nu": 0.2}},
    {"data": {"x": np.arange(8.)[::-1], "y": np.arange(8.)}},
])
def test_changed_request_cannot_reuse_score(change):
    cache = policy.FitCache()
    assert cache.key(**_request()) != cache.key(**_request(**change))


def test_parameterized_family_is_keyed_by_value():
    cache = policy.FitCache()
    request = _request(family=mb.Quantile(0.25))
    first = cache.key(**request)
    assert first == cache.key(**_request(family=mb.Quantile(0.25)))
    request["family"].tau = 0.75
    assert first != cache.key(**request)


def test_distinct_builds_do_not_share_keys():
    assert (policy.FitCache(build="a").key(**_request())
            != policy.FitCache(build="b").key(**_request()))


def test_category_labels_are_part_of_data_identity():
    cache = policy.FitCache()
    first = {"x": ["a", "b"] * 4, "y": np.arange(8.)}
    second = {"x": ["c", "d"] * 4, "y": np.arange(8.)}
    assert cache.key(**_request(data=first)) != cache.key(**_request(data=second))


@pytest.mark.parametrize("field", ["held_out", "judge", "base_weights", "repeats"])
def test_fold_evaluation_boundary_is_part_of_identity(field):
    weights = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    # Judge one of each fold's two held-out rows, so the held-out and judge
    # masks each have a valid alternative to change to.
    judge = np.array([[False, True], [False, False], [True, False], [False, False]])
    folds = mb.FoldScheme(weights, weights == 0, "test",
                          judge=judge, base_weights=np.ones(4))
    changes = {"held_out": judge.copy(),
               "judge": np.array([[False, False], [False, True], [False, False], [True, False]]),
               "base_weights": np.arange(1., 5.), "repeats": 2}
    cache = policy.FitCache()
    assert (cache.key(**_request(folds=folds))
            != cache.key(**_request(folds=replace(folds, **{field: changes[field]}))))


def test_array_shape_is_part_of_identity():
    cache = policy.FitCache()
    weights = np.arange(8.)
    assert (cache.key(**_request(folds=weights.reshape(4, 2)))
            != cache.key(**_request(folds=weights.reshape(2, 4))))


def test_opaque_configuration_bypasses_cache():
    cache = policy.FitCache()
    key = cache.key(**_request(fit_kwargs={"callback": lambda: None}))
    assert key is None
    assert cache.get_or_run(key, lambda: 1) == 1
    assert cache.get_or_run(key, lambda: 2) == 2
    assert cache.to_dict()["entries"] == 0


def test_policy_cache_hit_does_not_charge_historical_fits(monkeypatch):
    calls = []
    model = object()

    def tune(*args, **kwargs):
        calls.append(kwargs["seed"])
        return SimpleNamespace(mstop=10, model=model, fits=3)

    monkeypatch.setattr(actions, "autospec", lambda *args, **kwargs:
                        SimpleNamespace(formula="y ~ bols(x)", family=mb.Gaussian(),
                                        fits=int(kwargs["budget"] > 0)))
    monkeypatch.setattr(actions, "tune_mstop", tune)
    monkeypatch.setattr(actions, "cv_risk", lambda *args, **kwargs:
                        SimpleNamespace(score=0.25, fold_scores=[0.2, 0.3], fit_count=2))
    monkeypatch.setattr(policy, "_linear_rhs", lambda *args: "bols(x)")
    cache = policy.FitCache()
    kwargs = dict(complexity="simple", folds=2, mstop=10, cache=cache)
    data = _request()["data"]
    first = policy.auto_policy(data, "y", **kwargs)
    second = policy.auto_policy(data, "y", **kwargs)
    assert first.model is second.model is model
    assert first.score == second.score
    assert (first.fits, second.fits) == (6, 1)  # the probe still runs
    assert calls == [0]
    policy.auto_policy(data, "y", seed=1, **kwargs)
    policy.auto_policy(data, "y", metric="family", **kwargs)
    assert calls == [0, 1, 0]


def test_discovery_starts_after_scored_checkpoint_and_simple_profile_stops_after_shapes(monkeypatch):
    events = []
    checkpointed = []

    def discover(*args, **kwargs):
        events.append(("discover", kwargs["budget"]))
        if kwargs["budget"]:
            assert checkpointed and checkpointed[-1].score == 1.0
            assert kwargs["preexploration"] == "dual"
            assert kwargs["nu"] == 0.2
            assert kwargs["step"] == "line_search"
            return SimpleNamespace(formula="y ~ bbs(x)", family=mb.Gaussian(), fits=6)
        return SimpleNamespace(formula="y ~ bbs(x)", family=mb.Gaussian(), fits=0)

    def score(rhs, *args, **kwargs):
        events.append(("score", rhs))
        loss = 1.1 if "bbs" in rhs else 1.0
        return SimpleNamespace(score=loss, fold_scores=[loss, loss], fit_count=2)

    monkeypatch.setattr(actions, "autospec", discover)
    monkeypatch.setattr(actions, "tune_mstop", lambda *args, **kwargs:
                        SimpleNamespace(mstop=10, model=object(), fits=2))
    monkeypatch.setattr(actions, "cv_risk", score)
    result = policy._auto_policy(
        {"x": np.linspace(-1, 1, 40), "y": np.linspace(-1, 1, 40)},
        "y", folds=2, discovery_fits=12, preexploration="dual", complexity="simple",
        nu=0.2, step="line_search", _checkpoint=checkpointed.append)
    assert events == [("discover", 0), ("score", "bols(x)"),
                      ("discover", 12), ("score", "bbs(x)")]
    assert result.spec == "bols(x)" and result.score == 1.0
    assert result.trail[1].rung == "additive"
    assert all(not row.adopted for row in result.trail[1:])
    assert result.fits == 14


@pytest.mark.parametrize("kwargs", [
    {"discovery_fits": -1}, {"discovery_fits": 1.5},
    {"preexploration": "unknown"},
])
def test_invalid_discovery_settings_fail_before_fitting(monkeypatch, kwargs):
    monkeypatch.setattr(actions, "autospec", lambda *args, **kw: pytest.fail("unexpected discovery"))
    with pytest.raises((ValueError, TypeError)):
        policy.auto_policy(_request()["data"], "y", **kwargs)


def test_cached_cv_predictions_equal_fresh_computation():
    from mboostrs.explore import cv_risk

    rng = np.random.default_rng(24)
    data = {"x": rng.normal(size=40), "y": rng.normal(size=40)}
    cache = policy.FitCache()
    request = _request(data=data)

    def compute():
        return cv_risk("bols(x)", data, "y", family=mb.Gaussian(), folds=2,
                       mstop=10, score="mse", return_result=True)

    key = cache.key(**request)
    original = cache.get_or_run(key, compute)
    cached = cache.get_or_run(key, lambda: pytest.fail("unexpected refit"))
    fresh = compute()
    assert cached is original
    np.testing.assert_array_equal(cached.fold_scores, fresh.fold_scores)
    assert cached.records == fresh.records
