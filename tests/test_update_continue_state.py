"""`update(**kwargs)`, `continue_to` with a vector offset, refit state that
survives pickling and `save_model`, and centered `Buser` partial effects
(issue #820).

1. `update(nu=0.5)` returned the model unchanged: the keyword arguments were
   accepted and ignored. They now refit the model with those arguments
   replaced, and an unknown name raises `TypeError`.
2. `continue_to` on a gamboost fit with a per-observation offset raised
   `TypeError` from `float(offset)` in the resume path. A vector offset now
   takes the cold refit.
3. `__getstate__` dropped the refit bundle (and, on a GAM, the centered
   `Buser` means and the residual), so after pickling or loading a full
   artifact `continue_to` refused and `formula` returned the expanded
   formula.
4. `partial_effects()` on a `Buser(center=True)` term raised `TypeError`,
   because the stored means are a list.
"""

import pickle

import numpy as np
import pytest

import mboostrs as mb


def _frame(seed=7, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    y = np.sin(x) + 0.5 * z + 0.3 * rng.normal(size=n)
    return {"x": x, "z": z, "w": w, "y": y}


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _restore(model, route, tmp_path):
    if route == "pickle":
        return pickle.loads(pickle.dumps(model))
    path = tmp_path / "model.json"
    mb.save_model(model, path)
    return mb.load_model(path)


# ─── 1. update(**kwargs) ───────────────────────────────────────────

@pytest.mark.parametrize("engine,formula", [
    ("glmboost", "y ~ x + z"),
    ("gamboost", "y ~ bbs(x) + bols(z)"),
])
def test_update_with_nu_refits_at_that_step_length(engine, formula):
    data = _frame()
    fit = getattr(mb, engine)(formula, data=data, control=mb.boost_control(60, 0.1))
    updated = fit.update(nu=0.5)
    cold = getattr(mb, engine)(formula, data=data, control=mb.boost_control(60, 0.5))
    assert updated.mstop == 60
    assert updated.control.nu == 0.5
    assert _maxdiff(updated.fitted_, fit.fitted_) > 1e-3
    assert _maxdiff(updated.fitted_, cold.fitted_) <= 1e-10


def test_update_with_mstop_and_a_keyword_refits_to_that_mstop():
    data = _frame()
    fit = mb.glmboost("y ~ x + z", data=data, control=mb.boost_control(60, 0.1))
    updated = fit.update(30, nu=0.3)
    cold = mb.glmboost("y ~ x + z", data=data, control=mb.boost_control(30, 0.3))
    assert updated.mstop == 30
    assert _maxdiff(updated.fitted_, cold.fitted_) <= 1e-10


def test_update_with_a_fit_argument_refits_with_it():
    data = _frame()
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data=data,
                      control=mb.boost_control(40, 0.1))
    updated = fit.update(family=mb.Laplace())
    cold = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, family=mb.Laplace(),
                       control=mb.boost_control(40, 0.1))
    assert isinstance(updated.family, mb.Laplace)
    assert _maxdiff(updated.fitted_, cold.fitted_) <= 1e-10


@pytest.mark.parametrize("engine,formula", [
    ("glmboost", "y ~ x + z"),
    ("gamboost", "y ~ bbs(x) + bols(z)"),
])
def test_update_refuses_an_unsupported_keyword(engine, formula):
    fit = getattr(mb, engine)(formula, data=_frame(), control=mb.boost_control(20, 0.1))
    with pytest.raises(TypeError, match="'not_an_argument'"):
        fit.update(not_an_argument=1)
    with pytest.raises(TypeError, match="'data'"):
        fit.update(data=_frame(seed=8))


def test_update_validates_the_control_it_builds():
    fit = mb.glmboost("y ~ x + z", data=_frame(), control=mb.boost_control(20, 0.1))
    with pytest.raises(ValueError):
        fit.update(nu=2.0)


def test_update_with_mstop_alone_still_slices():
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data=_frame(),
                      control=mb.boost_control(60, 0.1))
    sliced = fit.update(mstop=25)
    assert sliced.mstop == 25
    assert _maxdiff(sliced.fitted_, fit[:25].fitted_) == 0.0


# ─── 2. continue_to with a vector offset ───────────────────────────

def test_continue_to_with_a_vector_offset_takes_the_cold_refit():
    data = _frame()
    offset = 0.5 * np.random.default_rng(1).normal(size=len(data["y"]))
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, offset=offset,
                      family=mb.Laplace(), control=mb.boost_control(40, 0.1))
    grown = fit.continue_to(80)
    cold = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, offset=offset,
                       family=mb.Laplace(), control=mb.boost_control(80, 0.1))
    assert grown.mstop == 80
    assert _maxdiff(grown.fitted_, cold.fitted_) <= 1e-10
    assert _maxdiff(grown[:40].fitted_, fit.fitted_) <= 1e-10


# ─── 3. refit state survives pickling and save_model ───────────────

@pytest.mark.parametrize("route", ["pickle", "artifact"])
@pytest.mark.parametrize("engine,formula,family", [
    ("glmboost", "y ~ .", mb.Gaussian()),
    ("gamboost", "y ~ bbs(x) + bols(z) + bols(w)", mb.Laplace()),
    ("gamboost", "y ~ bbs(x) + bols(z) + bols(w)", mb.Gaussian()),
])
def test_a_restored_model_continues_like_the_original(route, engine, formula,
                                                      family, tmp_path):
    data = _frame()
    fit = getattr(mb, engine)(formula, data=data, family=family,
                              control=mb.boost_control(40, 0.1))
    restored = _restore(fit, route, tmp_path)
    assert restored.formula == fit.formula == formula
    assert restored.unavailable_operations() == []
    grown = restored.continue_to(80)
    expected = fit.continue_to(80)
    assert grown.mstop == 80
    assert _maxdiff(grown.fitted_, expected.fitted_) <= 1e-10
    updated = restored.update(nu=0.3)
    assert _maxdiff(updated.fitted_, fit.update(nu=0.3).fitted_) <= 1e-10


@pytest.mark.parametrize("route", ["pickle", "artifact"])
def test_a_restored_gam_keeps_its_residual_and_buser_means(route, tmp_path):
    data = _frame()
    design = np.column_stack([data["w"], data["w"] ** 2])
    fit = mb.gamboost("y ~ bbs(x)", data=data, family=mb.Laplace(),
                      learners=[mb.Bbs("x"), mb.Buser("u", X=design, K=np.eye(2),
                                                     center=True)],
                      control=mb.boost_control(40, 0.1))
    restored = _restore(fit, route, tmp_path)
    np.testing.assert_array_equal(restored._residual, fit._residual)
    np.testing.assert_allclose(restored._buser_centering_means["u"],
                               fit._buser_centering_means["u"], rtol=0, atol=0)


def test_a_scoring_only_artifact_still_refuses_refits(tmp_path):
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data=_frame(), family=mb.Laplace(),
                      control=mb.boost_control(40, 0.1))
    path = tmp_path / "model.json"
    mb.save_model(fit, path, scoring_only=True)
    loaded = mb.load_model(path)
    assert loaded._residual is None
    assert {row["operation"] for row in loaded.unavailable_operations()} >= {
        "continue_to", "update"}
    with pytest.raises(ValueError, match="did not retain the inputs"):
        loaded.continue_to(80)


def test_the_refit_bundle_is_outside_the_portable_state(tmp_path):
    """The Rust and R readers decode the whole `state` object and refuse a
    training frame in it, so the bundle sits in its own top-level key."""
    import json

    import polars as pl

    fit = mb.glmboost("y ~ x + z", data=pl.DataFrame(_frame()),
                      control=mb.boost_control(20, 0.1))
    path = tmp_path / "model.json"
    mb.save_model(fit, path)
    document = json.loads(path.read_text())
    assert "refit_bundle" not in document["state"]
    assert document["refit_bundle"] is not None
    assert '"frame"' not in json.dumps(document["state"])
    assert mb.load_model(path).continue_to(40).mstop == 40


# ─── 4. centered Buser partial effects ─────────────────────────────

@pytest.mark.parametrize("route", [None, "pickle", "artifact"])
def test_partial_effects_on_a_centered_buser_term(route, tmp_path):
    data = _frame()
    design = np.column_stack([data["w"], data["w"] ** 2])
    fit = mb.gamboost("y ~ bbs(x)", data=data,
                      learners=[mb.Bbs("x"), mb.Buser("u", X=design, K=np.eye(2),
                                                     center=True)],
                      control=mb.boost_control(60, 0.1))
    model = fit if route is None else _restore(fit, route, tmp_path)
    _, effect = model.partial_effects(which="u")["u"]
    effect = np.asarray(effect)
    # The design is centered at its weighted training mean, so the effect
    # averages to zero over the training rows.
    assert effect.shape == (len(data["y"]),)
    assert abs(float(np.mean(effect))) <= 1e-10
    assert float(np.max(np.abs(effect))) > 0.0
