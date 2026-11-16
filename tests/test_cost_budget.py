"""fit_budget refusals happen before any fit, and estimate_cost counts the fits
each call performs."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import importlib

import mboostrs as mb
from mboostrs.cost import CostEstimate, estimate_cost
from mboostrs.gamlss import GaussianLSS

# The package re-exports functions named like these modules, so the module
# objects are reached through importlib.
conformal_module = importlib.import_module("mboostrs.conformal")
explore = importlib.import_module("mboostrs.explore")
gamboost_module = importlib.import_module("mboostrs.gamboost")
stabsel_module = importlib.import_module("mboostrs.stabsel")


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(0, 1, n)
    g = rng.integers(0, 3, n)
    y = np.sin(x) + 0.5 * z + rng.normal(0, 0.4, n)
    return {"y": y, "x": x, "z": z, "g": g}


def counting(monkeypatch, module, name):
    """Replace `module.name` with a wrapper that counts its calls."""
    original = getattr(module, name)
    calls = []

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


# ---- tune_mstop ----------------------------------------------------------

def test_tune_mstop_refuses_before_any_fit(monkeypatch):
    frame = data()
    fits = counting(monkeypatch, gamboost_module, "gamboost")
    with pytest.raises(ValueError, match=r"tune_mstop needs 4 fits.*fit_budget=3"):
        mb.tune_mstop("y ~ bbs(x, df=4)", frame, folds=3, start=20, fit_budget=3)
    assert fits == []


def test_tune_mstop_budget_at_the_count_runs(monkeypatch):
    frame = data()
    result = mb.tune_mstop("y ~ bbs(x, df=4)", frame, folds=3, start=20, fit_budget=4)
    assert result.fits == 3


def test_tune_mstop_estimate_is_the_refusal_count():
    cost = estimate_cost("tune_mstop", "y ~ bbs(x, df=4)", data(), folds=3, start=20)
    assert cost.fits == 4  # three fold fits plus the final fit


# ---- term_report ---------------------------------------------------------

FORMULA = "y ~ bbs(x, df=4) + bols(z) + brandom(g)"


def test_term_report_refuses_before_any_fit(monkeypatch):
    frame = data()
    needed = estimate_cost("term_report", FORMULA, frame, folds=3, mstop=20).fits
    cv_calls = counting(monkeypatch, explore, "cv_risk")
    with pytest.raises(ValueError, match=rf"term_report needs {needed} fits.*fit_budget={needed - 1}"):
        explore.term_report(FORMULA, frame, folds=3, mstop=20, fit_budget=needed - 1)
    assert cv_calls == []


def test_term_report_budget_at_the_count_runs_and_matches_the_report():
    frame = data()
    needed = estimate_cost("term_report", FORMULA, frame, folds=3, mstop=20).fits
    report = explore.term_report(FORMULA, frame, folds=3, mstop=20, fit_budget=needed)
    assert report.fits == needed


def test_term_report_estimate_follows_the_shapes_flag():
    frame = data()
    with_shapes = estimate_cost("term_report", FORMULA, frame, folds=3, mstop=20).fits
    without = estimate_cost("term_report", FORMULA, frame, folds=3, mstop=20,
                            shapes=False)
    report = explore.term_report(FORMULA, frame, folds=3, mstop=20, shapes=False)
    assert without.fits == report.fits
    assert without.fits < with_shapes


# ---- stabsel -------------------------------------------------------------

def fitted_model(frame):
    return mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=frame,
                       control=mb.boost_control(40, 0.1))


def test_stabsel_refuses_before_any_subsample_is_fitted(monkeypatch):
    model = fitted_model(data())
    subsamples = counting(monkeypatch, stabsel_module, "make_folds")
    selected = counting(monkeypatch, stabsel_module, "_selected_paths")
    # B=5 under the default "SS" is 10 fits.
    with pytest.raises(ValueError, match=r"stabsel needs 10 fits.*fit_budget=9"):
        mb.stabsel(model, B=5, PFER=2, q=1, verbose=False, fit_budget=9)
    assert subsamples == [] and selected == []


def test_stabsel_mb_counts_one_fit_per_subsample():
    model = fitted_model(data())
    assert stabsel_module.stabsel_planned_fits(5, None, "MB") == 5
    with pytest.raises(ValueError, match=r"stabsel needs 5 fits.*fit_budget=4"):
        mb.stabsel(model, B=5, PFER=2, q=1, verbose=False, sampling_type="MB",
                   fit_budget=4)


def test_stabsel_supplied_folds_count_their_initial_columns():
    frame = data()
    model = fitted_model(frame)
    folds = np.ones((len(frame["y"]), 4))
    # Four initial subsamples under SS are eight fits, whatever B says.
    assert stabsel_module.stabsel_planned_fits(None, folds, "SS") == 8
    with pytest.raises(ValueError, match=r"stabsel needs 8 fits.*fit_budget=7"):
        mb.stabsel(model, folds=folds, PFER=2, q=1, verbose=False, fit_budget=7)


def test_stabsel_lss_refuses_before_any_subsample_is_fitted(monkeypatch):
    frame = data()
    subsamples = counting(monkeypatch, stabsel_module, "make_folds")
    with pytest.raises(ValueError, match=r"stabsel_glmboostLSS needs 10 fits.*fit_budget=5"):
        mb.stabsel_glmboostLSS("y ~ bols(x) + bols(z)", frame, family=GaussianLSS(),
                               B=5, PFER=2, q=1, fit_budget=5)
    assert subsamples == []


def test_stabsel_gamboostLSS_refuses_with_the_same_message():
    frame = data()
    with pytest.raises(ValueError, match=r"stabsel_gamboostLSS needs 10 fits.*fit_budget=5"):
        mb.stabsel_gamboostLSS("y ~ bbs(x, df=3)", frame, family=GaussianLSS(),
                               B=5, PFER=2, q=1, fit_budget=5)


# ---- compare_specifications: same counter, same message -------------------

def test_compare_refusal_message_names_the_remedy(monkeypatch):
    frame = data(60)
    evaluations = counting(monkeypatch, explore, "evaluate_specification")
    with pytest.raises(ValueError, match=r"needs 6 fits.*fit_budget=5.*raise fit_budget to at least 6"):
        mb.compare_specifications(["bols(x)", "bbs(x, df=3)"], frame, "y", folds=3,
                                  mstop=20, fit_budget=5)
    assert evaluations == []


# ---- estimate_cost -------------------------------------------------------

def test_estimate_cost_returns_the_record_with_consistent_fields():
    cost = estimate_cost("gamboost", "y ~ bols(x)", data())
    assert isinstance(cost, CostEstimate)
    assert cost.call == "gamboost"
    assert cost.fits == 1
    assert cost.seconds_per_fit > 0
    assert cost.estimated_seconds == pytest.approx(cost.fits * cost.seconds_per_fit)
    assert "\n" not in cost.explanation


def test_estimate_cost_compare_matches_the_comparison_it_describes():
    frame = data(60)
    cost = estimate_cost("compare_specifications", ["bols(x)", "bbs(x, df=3)"], frame,
                         "y", folds=3, mstop=20)
    rows = mb.compare_specifications(["bols(x)", "bbs(x, df=3)"], frame, "y", folds=3,
                                     mstop=20)
    assert cost.fits == rows[-1]["cumulative fit count"] == 6


def test_estimate_cost_compare_counts_inner_folds_for_cv():
    frame = data(60)
    cost = estimate_cost("compare_specifications", ["bols(x)"], frame, "y", folds=2,
                         mstop="cv")
    assert cost.fits == 12


def test_estimate_cost_term_report_matches_the_report():
    frame = data()
    cost = estimate_cost("term_report", FORMULA, frame, folds=3, mstop=20)
    report = explore.term_report(FORMULA, frame, folds=3, mstop=20)
    assert cost.fits == report.fits


def test_estimate_cost_stabsel_matches_the_result():
    frame = data()
    model = fitted_model(frame)
    for sampling_type in ("SS", "MB"):
        cost = estimate_cost("stabsel", model, B=4, PFER=2, q=1,
                             sampling_type=sampling_type)
        result = mb.stabsel(model, B=4, PFER=2, q=1, verbose=False,
                            sampling_type=sampling_type)
        assert cost.fits == result.n_fits


def test_estimate_cost_conformal_matches_the_refits(monkeypatch):
    frame = data()
    model = fitted_model(frame)
    refits = counting(monkeypatch, conformal_module, "_refit")
    estimate = estimate_cost("conformal_interval", model, method="cv+", n_folds=3)
    assert estimate.fits == 3
    mb.conformal_interval(model, method="cv+", n_folds=3, mstop=40, seed=1)
    assert len(refits) == 3


def test_conformal_split_and_cqr_counts(monkeypatch):
    frame = data()
    model = fitted_model(frame)
    refits = counting(monkeypatch, conformal_module, "_refit")
    assert estimate_cost("conformal_interval", model, method="split").fits == 1
    mb.conformal_interval(model, method="split", mstop=40, seed=1)
    assert len(refits) == 1
    refits.clear()
    assert estimate_cost("conformal_interval", model, method="cqr", mstop=40).fits == 2
    mb.conformal_interval(model, method="cqr", mstop=40, seed=1)
    assert len(refits) == 2


def test_conformal_calibration_rows_cost_no_refit(monkeypatch):
    frame = data()
    model = fitted_model(frame)
    cal = {key: value[:40] for key, value in frame.items()}
    refits = counting(monkeypatch, conformal_module, "_refit")
    cost = estimate_cost("conformal_interval", model, calibration=cal)
    assert cost.fits == 0
    assert cost.estimated_seconds == 0.0
    mb.conformal_interval(model, calibration=cal, seed=1)
    assert refits == []


def test_estimate_cost_rejects_an_unknown_call():
    with pytest.raises(ValueError, match="cannot cost 'cvrisk'.*gamboost"):
        estimate_cost("cvrisk", "y ~ bols(x)", data())
