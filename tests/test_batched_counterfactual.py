"""Counterfactual predictions are stacked into one `predict` call (#296).

`prediction_profile`, `accumulated_local_effect` and the fallback path of
`GAMBoostModel.predict_over_grid` predict the same rows once per replaced
value. Each test compares the stacked result with an explicit loop of one
`predict` call per value, counts the `predict` calls the stacked path makes,
and checks that repeated calls reuse the model's prepared learner store.
"""

import functools
import pickle
import sys
import warnings

import numpy as np
import pytest

import mboostrs as mb
import mboostrs._counterfactual as counterfactual
from mboostrs.gamlss import GaussianLSS

gamboost_module = sys.modules["mboostrs.gamboost"]

RTOL = 1e-12


def _data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    z = rng.uniform(0, 1, n)
    w = rng.normal(size=n)
    k = rng.integers(0, 5, n).astype(float)
    g = rng.choice(["a", "b", "c"], n)
    y = (5 + np.sin(x) + 0.5 * z * x + (g == "b") * 1.0 + 0.3 * w + 0.2 * k
         + rng.normal(scale=0.3, size=n))
    return {"x": x, "z": z, "w": w, "k": k, "g": g, "y": y}


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.fixture(scope="module")
def gam(data):
    return mb.gamboost("y ~ bbs(x, df=5) + bbs(z, df=4) + bols(w) + brandom(k) + bols(g)",
                       data=data, control=mb.boost_control(mstop=120))


@pytest.fixture(scope="module")
def glm(data):
    numeric = {key: value for key, value in data.items() if key != "g"}
    return mb.glmboost("y ~ x + z + w + k", data=numeric,
                       control=mb.boost_control(mstop=120))


@pytest.fixture(scope="module")
def lss(data):
    return mb.gamboostLSS({"mu": "y ~ bbs(x, df=4) + bbs(z, df=3) + bols(w)",
                           "sigma": "y ~ bbs(x, df=3)"},
                          data=data, families=GaussianLSS(),
                          control=mb.boost_control(mstop=80))


def _loop(model, variable, values, frame, **kwargs):
    """One `predict` call per value, the computation the stacked path replaces."""
    columns = {name: np.asarray(column) for name, column in frame.items()}
    n = len(columns[variable])
    return np.vstack([
        np.asarray(model.predict(newdata={**columns, variable: np.full(n, value)},
                                 **kwargs), dtype=float)
        for value in values
    ])


def _count_predict(monkeypatch, model):
    """Wrap `model.predict` on the instance and return the call counter."""
    calls = {"n": 0}
    original = model.predict

    @functools.wraps(original)
    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "predict", counted, raising=False)
    return calls


def test_glmboost_profile_matches_the_loop_in_one_call(monkeypatch, glm, data):
    frame = {key: value for key, value in data.items() if key != "g"}
    grid = np.linspace(1.0, 9.0, 20)
    want = _loop(glm, "x", grid, frame, type="response")
    calls = _count_predict(monkeypatch, glm)
    result = mb.prediction_profile(glm, frame, "x", grid=grid, extrapolation="allow")
    assert calls["n"] == 1
    np.testing.assert_allclose(result.ice, want.T, rtol=RTOL, atol=0)


def test_lss_profile_matches_the_loop_in_one_call(monkeypatch, lss, data):
    grid = np.linspace(0.5, 9.5, 20)
    want = _loop(lss, "x", grid, data, parameter="mu", type="response")
    calls = _count_predict(monkeypatch, lss)
    result = mb.prediction_profile(lss, data, "x", grid=grid, parameter="mu",
                                   extrapolation="allow")
    assert calls["n"] == 1
    np.testing.assert_allclose(result.ice, want.T, rtol=RTOL, atol=0)


def test_factor_grid_fallback_matches_the_loop_in_one_call(monkeypatch, gam, data):
    levels = ["a", "b", "c"]
    assert gam._grid_plan("g", levels, "link") is None
    want = _loop(gam, "g", levels, data, type="link")
    calls = _count_predict(monkeypatch, gam)
    got = gam.predict_over_grid("g", levels, newdata=data, output="numpy")
    assert calls["n"] == 1
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


def test_factor_profile_on_gamboost_matches_the_loop(gam, data):
    result = mb.prediction_profile(gam, data, "g", type="response")
    want = _loop(gam, "g", list(result.grid), data, type="response")
    np.testing.assert_allclose(result.ice, want.T, rtol=RTOL, atol=0)


def test_interaction_fallback_matches_the_loop(monkeypatch, data):
    fitted = mb.gamboost("y ~ bbs(x, df=4) %X% bbs(z, df=4) + bols(w)", data=data,
                         control=mb.boost_control(mstop=60))
    grid = np.linspace(1.0, 9.0, 8)
    assert fitted._grid_plan("x", list(grid), "link") is None
    want = _loop(fitted, "x", grid, data, type="link")
    calls = _count_predict(monkeypatch, fitted)
    got = fitted.predict_over_grid("x", grid, newdata=data, output="numpy")
    assert calls["n"] == 1
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


def test_fallback_repeats_a_per_row_offset_for_each_value(data):
    offset = np.linspace(-0.5, 0.5, len(data["x"]))
    fitted = mb.gamboost("y ~ bbs(x, df=4) + bols(g)", data=data, offset=offset,
                         control=mb.boost_control(mstop=40))
    new_offset = offset + 0.25
    levels = ["a", "c"]
    want = _loop(fitted, "g", levels, data, type="link", offset=new_offset)
    got = fitted.predict_over_grid("g", levels, newdata=data, offset=new_offset,
                                   output="numpy")
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


def test_ale_matches_one_predict_per_bin_in_one_call(monkeypatch, gam, data):
    calls = _count_predict(monkeypatch, gam)
    result = mb.accumulated_local_effect(gam, data, "x", bins=8, type="response")
    assert calls["n"] == 1
    columns = {name: np.asarray(column) for name, column in data.items()}
    want = []
    for index in range(len(result.edges) - 1):
        rows = result.bin_index == index
        subset = {name: column[rows] for name, column in columns.items()}
        m = int(rows.sum())
        lower = gam.predict(newdata={**subset, "x": np.full(m, result.edges[index])},
                            type="response", extrapolation="error")
        upper = gam.predict(newdata={**subset, "x": np.full(m, result.edges[index + 1])},
                            type="response", extrapolation="error")
        want.append(np.mean(upper - lower))
    np.testing.assert_allclose(result.local_change, want, rtol=1e-10, atol=0)


def test_small_chunks_give_the_same_values(monkeypatch, glm, data):
    frame = {key: value for key, value in data.items() if key != "g"}
    grid = np.linspace(1.0, 9.0, 7)
    full = mb.prediction_profile(glm, frame, "x", grid=grid, extrapolation="allow")
    monkeypatch.setattr(counterfactual, "MAX_STACKED_ROWS", 2 * len(frame["x"]))
    calls = _count_predict(monkeypatch, glm)
    chunked = mb.prediction_profile(glm, frame, "x", grid=grid, extrapolation="allow")
    assert calls["n"] == 4
    np.testing.assert_array_equal(chunked.ice, full.ice)


def test_predict_frames_splits_frames_of_different_lengths():
    frames = [{"a": np.arange(3.0)}, {"a": np.arange(5.0) + 10}, {"a": np.arange(2.0)}]
    seen = []

    def predict(frame, copies):
        seen.append(copies)
        return frame["a"] * 2

    out = counterfactual.predict_frames(predict, frames)
    assert seen == [3]
    for frame, got in zip(frames, out, strict=True):
        np.testing.assert_array_equal(got, frame["a"] * 2)


@pytest.fixture
def count_store_builds(monkeypatch):
    builds = {"n": 0}
    real = gamboost_module._prepare_learner_store

    def counting(*args, **kwargs):
        builds["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gamboost_module, "_prepare_learner_store", counting)
    return builds


@pytest.mark.parametrize("restored", [False, True])
def test_repeated_counterfactual_calls_reuse_the_learner_store(
        count_store_builds, gam, data, restored):
    model = pickle.loads(pickle.dumps(gam)) if restored else gam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            model.predict_over_grid("x", [2.0, 5.0], newdata=data, output="numpy")
            model.predict_over_grid("g", ["a", "b"], newdata=data, output="numpy")
            mb.prediction_profile(model, data, "z", points=5)
            mb.accumulated_local_effect(model, data, "x", bins=4)
    store = model._replay_store()
    # A fitted model keeps the store its fit prepared; a restored model, which
    # cannot unpickle one, prepares it once and keeps it.
    assert count_store_builds["n"] == (1 if restored else 0)
    assert model._replay_store() is store
