"""stabsel fits each fold only as far as its result depends on.

A fold's contribution to `phat` is fixed once its q-th distinct learner has
entered: the q-truncation rewrites everything after the (q+1)-th distinct
entry, and `phat` records each learner's first appearance, so later iterations
can change nothing. Each path stops there instead of running to `mstop` --
`_selected_paths` refits the standard models on a smaller budget, and the LSS
engine breaks out of its own loop at the q-th distinct component. These tests
pin that both shortcuts are exact -- the reference runs every fold to the
model's full `mstop`, which is what the implementation used to do.
"""

import importlib

import numpy as np
import pytest

import mboostrs as mb

# `mboostrs.stabsel` is shadowed by the re-exported function of the same name.
stabsel_module = importlib.import_module("mboostrs.stabsel")


@pytest.fixture
def full_path_stabsel(monkeypatch):
    """`mb.stabsel` with every fold fitted to the model's full mstop."""
    def _full_paths(model, folds, n_fits, is_glmboost, n_jobs, q_val, m):
        if is_glmboost and stabsel_module._HAS_SELECTED_ENGINE:
            return stabsel_module._run_glmboost_selected(model, folds, None)
        return stabsel_module._run_refit_selected(model, folds, is_glmboost,
                                                  n_jobs, None)

    def _apply():
        monkeypatch.setattr(stabsel_module, "_selected_paths", _full_paths)

    return _apply


def _data(seed=3):
    rng = np.random.default_rng(seed)
    n = 120
    x = {f"x{j}": rng.normal(size=n) for j in range(6)}
    x["y"] = 2.0 * x["x0"] - 1.5 * x["x1"] + 0.5 * x["x2"] + rng.normal(size=n)
    return x


def _identical(reference, shortcut):
    assert np.array_equal(reference.phat, shortcut.phat)
    assert np.array_equal(reference.max_, shortcut.max_)
    assert reference.selected == shortcut.selected


def test_gamboost_early_stop_matches_full_paths(full_path_stabsel):
    data = _data()
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(6))
    model = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                        control=mb.boost_control(400, 0.1))

    shortcut = mb.stabsel(model, q=3, PFER=1.0, B=8, assumption="unimodal",
                          verbose=False)
    full_path_stabsel()
    reference = mb.stabsel(model, q=3, PFER=1.0, B=8, assumption="unimodal",
                           verbose=False)
    _identical(reference, shortcut)


def test_glmboost_early_stop_matches_full_paths(full_path_stabsel):
    data = _data()
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(6))
    model = mb.glmboost(formula, data=data, family=mb.Gaussian(),
                        control=mb.boost_control(400, 0.1))

    shortcut = mb.stabsel(model, q=3, PFER=1.0, B=8, assumption="unimodal",
                          verbose=False)
    full_path_stabsel()
    reference = mb.stabsel(model, q=3, PFER=1.0, B=8, assumption="unimodal",
                           verbose=False)
    _identical(reference, shortcut)


def test_result_does_not_depend_on_n_jobs():
    data = _data(seed=11)
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(6))
    model = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                        control=mb.boost_control(300, 0.1))

    serial = mb.stabsel(model, q=3, PFER=1.0, B=6, assumption="unimodal",
                        verbose=False, n_jobs=1)
    threaded = mb.stabsel(model, q=3, PFER=1.0, B=6, assumption="unimodal",
                          verbose=False, n_jobs=-1)
    _identical(serial, threaded)


def test_short_mstop_still_warns_when_q_is_unreachable():
    """A model too short to select q distinct learners keeps warning."""
    data = _data(seed=5)
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(6))
    model = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                        control=mb.boost_control(2, 0.1))

    with pytest.warns(UserWarning, match="mstop too small"):
        mb.stabsel(model, q=4, PFER=1.0, B=4, assumption="unimodal")


@pytest.fixture
def full_path_lss_stabsel(monkeypatch):
    """LSS stabsel with every fold fitted to the full mstop.

    The engine caps a fold at its q-th distinct component; passing `q=None`
    instead is what the implementation did before the cap existed, and is the
    reference these tests compare against. `replaced` collects the caps that
    were dropped, so a test can tell that the reference really ran uncapped.
    """
    from mboostrs import gamlss as gamlss_module

    replaced = []

    def _apply():
        for name in ("_stabsel_selected_gamboostlss_mixed",
                     "_stabsel_selected_glmboostlss"):
            capped = getattr(gamlss_module, name)

            def uncapped(*args, _capped=capped, **kwargs):
                # q is the second-to-last positional argument (before
                # stabilization); nu_per_param arrives as a keyword after it.
                replaced.append(args[-2])
                return _capped(*args[:-2], None, args[-1], **kwargs)

            monkeypatch.setattr(gamlss_module, name, uncapped)
        return replaced

    return _apply


def _lss_data(seed=4):
    rng = np.random.default_rng(seed)
    n = 300
    x = {f"x{j}": rng.uniform(0.0, 1.0, n) for j in range(6)}
    x["y"] = rng.normal(1.0 + 2.0 * x["x0"] - 1.5 * x["x1"],
                        np.exp(-0.5 + 0.6 * x["x2"]))
    return x


def test_gamboostlss_early_stop_matches_full_paths(full_path_lss_stabsel):
    data = _lss_data()
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(6))
    model = mb.gamboostLSS({"mu": formula, "sigma": formula}, data=data,
                           families=mb.GaussianLSS(),
                           control=mb.boost_control(200, 0.1),
                           method="noncyclic")

    shortcut = model.stabsel(q=4, PFER=1.0, B=6, seed=17)
    replaced = full_path_lss_stabsel()
    reference = model.stabsel(q=4, PFER=1.0, B=6, seed=17)
    assert replaced == [4], "the reference run must not be capped"
    _identical(reference, shortcut)


def test_glmboostlss_early_stop_matches_full_paths(full_path_lss_stabsel):
    data = _lss_data(seed=9)
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(6))
    model = mb.glmboostLSS({"mu": formula, "sigma": formula}, data=data,
                           families=mb.GaussianLSS(),
                           control=mb.boost_control(200, 0.1),
                           method="noncyclic")

    shortcut = model.stabsel(q=4, PFER=1.0, B=6, seed=23)
    replaced = full_path_lss_stabsel()
    reference = model.stabsel(q=4, PFER=1.0, B=6, seed=23)
    assert replaced == [4], "the reference run must not be capped"
    _identical(reference, shortcut)


@pytest.mark.parametrize("q", [2, 4, 6])
def test_lss_capped_fold_path_is_a_prefix_that_stops_at_q(q):
    """The capped path is the full path cut at its q-th distinct component."""
    from mboostrs._rust import stabsel_selected_gamboostlss_mixed as selected
    from mboostrs.formula import parse_gamboost_formula
    from mboostrs.gamlss import _build_lss_response, _gamboostlss_learner_specs

    data = _lss_data(seed=13)
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(6))
    learners = parse_gamboost_formula(formula)[1]
    specs = _gamboostlss_learner_specs([learners, learners], data)
    y, n = _build_lss_response(data, "y", False, False, False)
    folds = np.ascontiguousarray(stabsel_module._subsample_folds(n, 6, seed=2))
    weights = np.ones(n)

    full = selected(specs, y, weights, 200, 0.1, "gaussianlss", folds, None)
    capped = selected(specs, y, weights, 200, 0.1, "gaussianlss", folds, q)

    for capped_fold, full_fold in zip(capped, full, strict=True):
        assert list(capped_fold) == list(full_fold[:len(capped_fold)])
        assert len(set(capped_fold)) == q
        # Stopping any earlier would drop a distinct component phat needs.
        assert len(set(capped_fold[:-1])) == q - 1


def test_lss_short_mstop_still_warns_when_q_is_unreachable():
    data = _lss_data(seed=21)
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=3)" for j in range(6))
    model = mb.gamboostLSS({"mu": formula, "sigma": formula}, data=data,
                           families=mb.GaussianLSS(),
                           control=mb.boost_control(2, 0.1),
                           method="noncyclic")

    with pytest.warns(UserWarning, match="mstop too small"):
        model.stabsel(q=4, PFER=1.0, B=4)
