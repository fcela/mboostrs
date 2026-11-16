"""`predict_over_grid` evaluates a value grid in one pass and matches `predict` per value (#296).

Each test compares the one-pass result with a loop of `predict` calls on a
frame whose variable column is replaced by the grid value, which is what the
method replaces. `prediction_profile` is compared with the per-value loop it
used before the rewire, reproduced by removing the method from the class.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.gamboost import GAMBoostModel
from mboostrs.gamlss import GaussianLSS


def _data(n=600, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    z = rng.uniform(0, 1, n)
    w = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n)
    y = (np.sin(x) + 0.5 * z * x + (g == "b") * 1.0 + 0.5 * w
         + rng.normal(scale=0.3, size=n))
    return {"x": x, "z": z, "w": w, "g": g, "y": y}


@pytest.fixture(scope="module")
def model():
    data = _data()
    formula = ("y ~ bbs(x, df=5) + bbs(z, df=4) + bbs(x, by=z, df=4) "
               "+ bols(w) + brandom(g)")
    return mb.gamboost(formula, data=data, control=mb.boost_control(mstop=80))


def _columns(frame):
    return {name: np.asarray(frame[name]) for name in frame}


def _per_value(fitted, variable, grid, frame, *, object_column=False, **kwargs):
    columns = _columns(frame)
    n = len(next(iter(columns.values())))
    rows = []
    for value in grid:
        dtype = object if object_column else None
        replaced = {**columns, variable: np.full(n, value, dtype=dtype)}
        rows.append(np.asarray(fitted.predict(newdata=replaced, **kwargs), dtype=float))
    return np.vstack(rows)


@pytest.mark.parametrize("variable", ["x", "z", "w"])
@pytest.mark.parametrize("scale", ["link", "response"])
def test_one_pass_matches_predict_per_value(model, variable, scale):
    frame = {k: v[:250] for k, v in _data().items()}
    grid = np.linspace(0.2, 0.8, 5) if variable == "z" else np.linspace(0.5, 9.5, 6)
    if variable == "w":
        grid = np.linspace(-1.5, 1.5, 5)
    assert model._grid_plan(variable, list(grid), scale) is not None
    got = model.predict_over_grid(variable, grid, newdata=frame, type=scale,
                                  output="numpy")
    want = _per_value(model, variable, grid, frame, type=scale)
    assert got.shape == (len(grid), 250)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


@pytest.mark.parametrize("label, formula, env, variable, grid", [
    ("bmono", "y ~ bmono(x, constraint='increasing', df=4) + bols(w)", None,
     "x", [1.0, 4.0, 8.0]),
    ("bns", "y ~ bns(x, df=4) + bbs(z, df=4, center=False)", None,
     "x", [1.0, 4.0, 8.0]),
    ("brandom_by", "y ~ bbs(x, df=4) + brandom(k, by=w)", None,
     "w", [-0.5, 1.0]),
    ("brandom_grid", "y ~ bbs(x, df=4) + brandom(k)", None,
     "k", [0.0, 3.0]),
    ("bconstrained", "y ~ bconstrained(x, C=constr, knots=20, degree=3) + bols(z)",
     {"constr": np.eye(24)}, "x", [1.0, 4.0, 8.0]),
])
def test_one_pass_matches_for_each_univariate_learner(label, formula, env, variable, grid):
    rng = np.random.default_rng(31)
    n = 400
    data = {
        "x": rng.uniform(0, 10, n),
        "z": rng.uniform(0, 1, n),
        "w": rng.normal(size=n),
        "k": rng.integers(0, 6, n).astype(float),
    }
    data["y"] = (np.log1p(data["x"]) + 2 * data["z"] + 0.3 * data["w"]
                 + 0.2 * data["k"] + rng.normal(scale=0.2, size=n))
    fitted = mb.gamboost(formula, data=data, env=env,
                         control=mb.boost_control(mstop=50))
    frame = {key: value[:120] for key, value in data.items()}
    assert fitted._grid_plan(variable, list(grid), "link") is not None, label
    got = fitted.predict_over_grid(variable, grid, newdata=frame, output="numpy")
    want = _per_value(fitted, variable, grid, frame, type="link")
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_factor_grid_falls_back_and_matches(model):
    frame = {k: v[:200] for k, v in _data().items()}
    levels = ["a", "b", "c"]
    assert model._grid_plan("g", levels, "link") is None
    got = model.predict_over_grid("g", levels, newdata=frame, output="numpy")
    want = _per_value(model, "g", levels, frame, object_column=True, type="link")
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_truncated_mstop_matches(model):
    frame = {k: v[:200] for k, v in _data().items()}
    grid = np.linspace(0.5, 9.5, 4)
    got = model.predict_over_grid("x", grid, newdata=frame, mstop=25, output="numpy")
    want = _per_value(model, "x", grid, frame, type="link", mstop=25)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_offset_model_takes_the_new_rows_offset():
    data = _data(n=300, seed=5)
    offset = np.linspace(-0.5, 0.5, 300)
    fitted = mb.gamboost("y ~ bbs(x, df=4) + bbs(z, df=3)", data=data,
                         offset=offset, control=mb.boost_control(mstop=40))
    frame = {k: v[:120] for k, v in data.items()}
    new_offset = offset[:120] + 0.25
    grid = np.linspace(1.0, 9.0, 4)
    got = fitted.predict_over_grid("x", grid, newdata=frame, offset=new_offset,
                                   output="numpy")
    want = _per_value(fitted, "x", grid, frame, type="link", offset=new_offset)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_offset_model_refuses_rows_without_their_offset():
    data = _data(n=300, seed=5)
    fitted = mb.gamboost("y ~ bbs(x, df=4)", data=data,
                         offset=np.linspace(-0.5, 0.5, 300),
                         control=mb.boost_control(mstop=20))
    frame = {k: v[:50] for k, v in data.items()}
    with pytest.raises(ValueError, match="offset"):
        fitted.predict_over_grid("x", [1.0, 2.0], newdata=frame)


@pytest.mark.parametrize("policy", ["allow", "clip", "warn"])
def test_out_of_range_grid_follows_the_predict_policy(model, policy):
    frame = {k: v[:150] for k, v in _data().items()}
    grid = [-2.0, 5.0, 14.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = model.predict_over_grid("x", grid, newdata=frame,
                                      extrapolation=policy, output="numpy")
    with warnings.catch_warnings(record=True) as expected_caught:
        warnings.simplefilter("always")
        want = _per_value(model, "x", grid, frame, type="link", extrapolation=policy)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
    if policy == "warn":
        assert any("training range" in str(w.message) for w in caught)
        assert any("training range" in str(w.message) for w in expected_caught)


def test_out_of_range_grid_errors_under_error_policy(model):
    frame = {k: v[:50] for k, v in _data().items()}
    with pytest.raises(ValueError):
        model.predict_over_grid("x", [5.0, 14.0], newdata=frame, extrapolation="error")
    with pytest.raises(ValueError):
        _per_value(model, "x", [5.0, 14.0], frame, type="link", extrapolation="error")


def test_records_output_names_columns_by_grid_value(model):
    frame = {k: v[:20] for k, v in _data().items()}
    rows = model.predict_over_grid("x", [1.0, 2.5], newdata=frame, output="records")
    assert len(rows) == 20
    assert set(rows[0]) == {"row", "x=1", "x=2.5"}


def test_unknown_variable_is_refused(model):
    frame = {k: v[:20] for k, v in _data().items()}
    with pytest.raises(ValueError, match="not a column"):
        model.predict_over_grid("nope", [1.0], newdata=frame)


def test_empty_grid_is_refused(model):
    frame = {k: v[:20] for k, v in _data().items()}
    with pytest.raises(ValueError, match="at least one value"):
        model.predict_over_grid("x", [], newdata=frame)


def _profile_before_and_after(monkeypatch, fitted, data, variable, **kwargs):
    """The result with the one-pass method, and with it removed (the per-value loop)."""
    after = mb.prediction_profile(fitted, data, variable, **kwargs)
    monkeypatch.delattr(GAMBoostModel, "predict_over_grid")
    before = mb.prediction_profile(fitted, data, variable, **kwargs)
    return before, after


@pytest.mark.parametrize("variable", ["x", "z", "g"])
def test_prediction_profile_is_unchanged_on_gamboost(monkeypatch, model, variable):
    data = _data()
    before, after = _profile_before_and_after(
        monkeypatch, model, data, variable, points=7, type="response")
    np.testing.assert_allclose(after.average, before.average, rtol=0, atol=1e-10)
    np.testing.assert_allclose(after.ice, before.ice, rtol=0, atol=1e-10)
    assert after.grid.tolist() == before.grid.tolist()


def test_prediction_profile_is_unchanged_with_weights_and_link_scale(monkeypatch, model):
    data = _data()
    weights = np.linspace(0.2, 2.0, len(data["x"]))
    before, after = _profile_before_and_after(
        monkeypatch, model, data, "z", points=9, type="link", weights=weights)
    np.testing.assert_allclose(after.average, before.average, rtol=0, atol=1e-10)
    np.testing.assert_allclose(after.ice, before.ice, rtol=0, atol=1e-10)
    np.testing.assert_allclose(after.support_fraction_far, before.support_fraction_far)


def test_prediction_profile_on_gamboostlss_matches_per_value_predict():
    data = _data(n=400, seed=21)
    fitted = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4) + bbs(z, df=3)", "sigma": "y ~ bols(w)"},
        data=data, families=GaussianLSS(), control=mb.boost_control(mstop=40))
    result = mb.prediction_profile(fitted, data, "x", points=5,
                                   parameter="mu", type="response")
    columns = {k: np.asarray(v) for k, v in data.items()}
    n = len(data["x"])
    want = np.vstack([
        np.asarray(fitted.predict(newdata={**columns, "x": np.full(n, value)},
                                  parameter="mu", type="response"), dtype=float)
        for value in result.grid
    ])
    np.testing.assert_allclose(result.ice, want.T, rtol=0, atol=1e-10)
