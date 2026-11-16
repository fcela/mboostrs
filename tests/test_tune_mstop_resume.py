"""`tune_mstop` resumes each fold's path across the doubling grids.

Every route is compared with the refit-from-zero search it replaced: each grid
refits every fold from the first iteration and reads the held-out risk off that
fit. The reference below is written with public calls only, so a regression in
the resumed path shows up as a difference from it. Agreement is required to
1e-12.
"""

import sys
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import Binomial, Gaussian, GaussianLSS, Poisson
from mboostrs._control import boost_control
from mboostrs._frame import take_rows
from mboostrs._rust import response_score, select_score_path
from mboostrs.cvrisk import BOUNDARY_SHARE, cv_weights, is_at_boundary
from mboostrs.explore import _evaluation_weights, _fitted_family_score, _folds, _response_values

TOL = 1e-12

GAMBOOST = sys.modules["mboostrs.gamboost"]


def lss_frame(n=200, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    mu = np.sin(x) + 0.3 * x
    sigma = 0.5 + 0.2 * np.cos(x / 2) ** 2
    return {"x": x, "y": mu + sigma * rng.standard_normal(n)}


def count_frame(n=200, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return {"x": x, "y": rng.poisson(np.exp(0.4 * np.sin(x) + 0.05 * x)).astype(float)}


def binary_frame(n=200, seed=9):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    p = 1.0 / (1.0 + np.exp(-(1.2 * np.sin(x))))
    return {"x": x, "y": (rng.uniform(size=n) < p).astype(float)}


def refit_lss_search(formulas, data, family, matrix, *, start, nu, extensions, max_mstop):
    """The refit-from-zero LSS search: one full cross-validation per grid."""
    from mboostrs.gamlss import diagonal_grid, gamboostLSS, make_grid_lss

    largest, left, ends = start, extensions, []
    while True:
        fit = gamboostLSS(formulas, data=data, family=family,
                          control=boost_control(largest, nu))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            curve = np.asarray(fit.cvrisk(
                folds=matrix,
                grid=diagonal_grid(largest, family.n_params, length_out=largest + 1)
            ).aggregate, dtype=float)
        ends.append(largest)
        values = make_grid_lss(largest, length_out=largest + 1)
        best = int(values[int(np.argmin(np.where(np.isfinite(curve), curve, np.inf)))])
        at_boundary = is_at_boundary(best, largest)
        if not at_boundary or left == 0 or largest >= max_mstop:
            break
        largest = min(largest * 2, max_mstop)
        if left is not None:
            left -= 1
    return {"risk": curve, "mstop": best, "largest": largest,
            "at_boundary": at_boundary, "ends": ends}


def refit_response_search(formula, data, family, *, folds, seed, score, parameter,
                          start, nu, extensions, max_mstop):
    """The refit-from-zero response search: every grid refits every fold."""
    from mboostrs.gamboost import gamboost

    response = formula.split("~", 1)[0].strip()
    y = np.asarray(_response_values(data, response), dtype=float)
    weights = _evaluation_weights(len(y), folds, None)
    partitions = _folds(len(y), folds, seed, weights)
    largest, left, ends = start, extensions, []
    while True:
        fold_scores = []
        for training_mask, held_mask, training_weights in partitions:
            train_rows, held_rows = np.flatnonzero(training_mask), np.flatnonzero(held_mask)
            model = gamboost(formula, data=take_rows(data, train_rows), family=family,
                             weights=training_weights[train_rows],
                             control=boost_control(largest, nu))
            held = take_rows(data, held_rows)
            path = []
            for k in range(largest + 1):
                if score == "family":
                    prediction = np.asarray(model.predict(held, mstop=k, type="link"), float)
                    path.append(_fitted_family_score(model, family, y[held_rows], prediction,
                                                     weights[held_rows], mstop=k))
                    continue
                prediction = np.asarray(model.predict(held, mstop=k, type="response"), float)
                prediction = prediction.reshape(len(held_rows), -1)
                mean, _ = response_score(y[held_rows], np.ascontiguousarray(prediction),
                                         weights[held_rows], score, parameter)
                path.append(mean)
            fold_scores.append(path)
        ends.append(largest)
        curve, best = select_score_path(np.ascontiguousarray(fold_scores, dtype=float))
        at_boundary = is_at_boundary(best, largest)
        if not at_boundary or left == 0 or largest >= max_mstop:
            break
        largest = min(largest * 2, max_mstop)
        if left is not None:
            left -= 1
    return {"risk": np.asarray(curve), "mstop": int(best), "largest": largest,
            "at_boundary": at_boundary, "ends": ends,
            "fold_scores": np.asarray(fold_scores)}


# --- distributional specifications -------------------------------------------

LSS_FORMULAS = {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bbs(x, df=3)"}


@pytest.mark.parametrize("folds, extensions, max_mstop, start", [
    (4, None, 64, 8),       # doubles until the curve turns or the cap is reached
    (4, 1, 64, 8),          # stops after one extension
    (3, None, 40, 20),      # the cap cuts the last grid
    (5, None, 80, 60),      # the first grid already holds the minimum
])
def test_lss_resumed_search_equals_refit_from_zero(folds, extensions, max_mstop, start):
    data = lss_frame()
    matrix = cv_weights(len(data["y"]), type="kfold", B=folds, seed=3)
    reference = refit_lss_search(LSS_FORMULAS, data, GaussianLSS(), matrix, start=start,
                                 nu=0.1, extensions=extensions, max_mstop=max_mstop)
    tuning = mb.tune_mstop(LSS_FORMULAS, data, GaussianLSS(), folds=matrix, start=start,
                           nu=0.1, extensions=extensions, max_mstop=max_mstop, seed=3)
    assert tuning.mstop == reference["mstop"]
    assert tuning.largest == reference["largest"]
    assert tuning.at_boundary == reference["at_boundary"]
    assert tuning.grid_ends == tuple(reference["ends"])
    np.testing.assert_allclose(np.asarray(tuning.risk), reference["risk"], rtol=0, atol=TOL)


def test_lss_search_fits_each_fold_once_and_one_full_fit():
    data = lss_frame()
    tuning = mb.tune_mstop(LSS_FORMULAS, data, GaussianLSS(), folds=4, start=8, nu=0.1,
                           max_mstop=64, seed=3)
    assert len(tuning.grid_ends) >= 2, "the search must grow for this check to mean anything"
    assert tuning.fits == 4
    assert tuning.full_data_fits == 1
    assert tuning.total_fits == 5


def test_lss_search_budget_is_folds_plus_one():
    data = lss_frame()
    with pytest.raises(ValueError, match=r"needs 5 fits"):
        mb.tune_mstop(LSS_FORMULAS, data, GaussianLSS(), folds=4, start=8, max_mstop=64,
                      seed=3, fit_budget=4)
    mb.tune_mstop(LSS_FORMULAS, data, GaussianLSS(), folds=4, start=8, max_mstop=64,
                  seed=3, fit_budget=5)


# --- response scores and fold-local family scores ----------------------------

@pytest.mark.parametrize("formula, family, score, parameter, data, folds, start, max_mstop", [
    ("y ~ bbs(x, df=4)", Gaussian(), "mse", None, lss_frame(), 4, 8, 64),
    ("y ~ bols(x)", Gaussian(), "mse", None, lss_frame(), 3, 8, 32),
    ("y ~ bbs(x, df=4)", Binomial(), "brier", None, binary_frame(), 4, 8, 64),
    ("y ~ bbs(x, df=4)", Gaussian(), "pinball", 0.5, lss_frame(), 4, 8, 64),
])
def test_response_score_resumed_search_equals_refit_from_zero(
        formula, family, score, parameter, data, folds, start, max_mstop):
    reference = refit_response_search(formula, data, family, folds=folds, seed=2, score=score,
                                      parameter=parameter, start=start, nu=0.1,
                                      extensions=None, max_mstop=max_mstop)
    tuning = mb.tune_mstop(formula, data, family, folds=folds, start=start, nu=0.1,
                           max_mstop=max_mstop, seed=2, score=score, score_parameter=parameter)
    assert tuning.mstop == reference["mstop"]
    assert tuning.largest == reference["largest"]
    assert tuning.at_boundary == reference["at_boundary"]
    assert tuning.grid_ends == tuple(reference["ends"])
    np.testing.assert_allclose(np.asarray(tuning.risk), reference["risk"], rtol=0, atol=TOL)
    np.testing.assert_allclose(np.asarray(tuning.fold_scores), reference["fold_scores"],
                               rtol=0, atol=TOL)


def test_fold_local_family_score_resumed_search_equals_refit_from_zero():
    data = count_frame()
    reference = refit_response_search("y ~ bbs(x, df=4)", data, Poisson(), folds=4, seed=2,
                                      score="family", parameter=None, start=8, nu=0.1,
                                      extensions=None, max_mstop=64)
    tuning = mb.tune_mstop("y ~ bbs(x, df=4)", data, Poisson(), folds=4, start=8, nu=0.1,
                           max_mstop=64, seed=2, fold_local=True)
    assert tuning.mstop == reference["mstop"]
    assert tuning.largest == reference["largest"]
    assert tuning.grid_ends == tuple(reference["ends"])
    np.testing.assert_allclose(np.asarray(tuning.risk), reference["risk"], rtol=0, atol=TOL)


def record_fit_routes(monkeypatch):
    """Record the numerical route of every model `gamboost` returns while tuning.

    The tuner builds its fold models and the final fit through `gamboost`, so
    the list holds one entry per fit, in order, named by the route it used.
    """
    real_gamboost = GAMBOOST.gamboost
    routes = []

    def recording(*args, **kwargs):
        model = real_gamboost(*args, **kwargs)
        report = model.execution_report()
        routes.append(None if report is None else report.route)
        return model

    monkeypatch.setattr(GAMBOOST, "gamboost", recording)
    return routes


def test_gram_route_response_folds_are_refitted_at_each_grid(monkeypatch):
    # A Gaussian bbs fit takes the Gaussian Gram route, which keeps no residual,
    # so each fold is refitted from zero at every grid. That is the fit count
    # the refit-from-zero search spent, and no fold runs on the general route.
    data = lss_frame()
    routes = record_fit_routes(monkeypatch)
    tuning = mb.tune_mstop("y ~ bbs(x, df=4)", data, Gaussian(), folds=4, start=8, nu=0.1,
                           max_mstop=64, seed=2, score="mse")
    grids = len(tuning.grid_ends)
    assert grids >= 2, "the search must grow for this check to mean anything"
    assert tuning.fits == 4 * grids
    assert routes == ["gaussian_gram"] * (4 * grids + 1)  # the folds, then the final fit


def test_general_route_response_folds_resume_from_grid_to_grid(monkeypatch):
    # A Poisson fit is not Gram-eligible, so each fold is continued from where
    # the previous grid stopped: one fit per fold, and the final fit.
    data = count_frame()
    routes = record_fit_routes(monkeypatch)
    tuning = mb.tune_mstop("y ~ bbs(x, df=4)", data, Poisson(), folds=4, start=8, nu=0.1,
                           max_mstop=64, seed=2, fold_local=True)
    assert len(tuning.grid_ends) >= 2
    assert tuning.fits == 4
    assert len(routes) == 4 + 1
    assert "gaussian_gram" not in routes


def test_response_budget_counts_a_gram_refit_at_every_grid():
    data = lss_frame()
    # 8 -> 16 -> 32 -> 64 is four grids: the worst case is 4 folds * 4 grids + 1.
    with pytest.raises(ValueError, match=r"tune_mstop needs 17 fits.*fit_budget=16"):
        mb.tune_mstop("y ~ bbs(x, df=4)", data, Gaussian(), folds=4, start=8, max_mstop=64,
                      seed=2, score="mse", fit_budget=16)
    mb.tune_mstop("y ~ bbs(x, df=4)", data, Gaussian(), folds=4, start=8, max_mstop=64,
                  seed=2, score="mse", fit_budget=17)


def test_patience_is_still_refused_on_the_resumed_routes():
    data = lss_frame()
    with pytest.raises(NotImplementedError, match="patience"):
        mb.tune_mstop("y ~ bbs(x, df=4)", data, Gaussian(), folds=3, start=8, score="mse",
                      patience=3)
    with pytest.raises(NotImplementedError, match="patience"):
        mb.tune_mstop(LSS_FORMULAS, data, GaussianLSS(), folds=3, start=8, patience=3)


# --- the family score, already resumed, is unchanged -------------------------

def test_family_score_search_equals_one_cross_validation_on_the_final_grid():
    data = lss_frame()
    tuning = mb.tune_mstop("y ~ bbs(x, df=4)", data, Gaussian(), folds=4, start=8, nu=0.1,
                           max_mstop=64, seed=2)
    assert len(tuning.grid_ends) >= 2
    final = tuning.largest
    cold = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=Gaussian(),
                       control=boost_control(final, 0.1))
    matrix = cv_weights(len(data["y"]), type="kfold", B=4, seed=2)
    reference = cold.cvrisk(folds=matrix, _inner=True).aggregate
    np.testing.assert_allclose(np.asarray(tuning.risk), np.asarray(reference)[:final + 1],
                               rtol=0, atol=TOL)
    assert tuning.fits == 4


# --- continue_to, the warm path the general-route folds use -----------------

@pytest.mark.parametrize("formula", ["y ~ bbs(x, df=4)", "y ~ bols(x)"])
def test_continue_to_equals_a_cold_fit_over_two_steps(formula):
    data = lss_frame()
    model = mb.gamboost(formula, data=data, family=Gaussian(), control=boost_control(20, 0.1))
    grown = model.continue_to(50).continue_to(120)
    cold = mb.gamboost(formula, data=data, family=Gaussian(), control=boost_control(120, 0.1))
    assert grown.mstop == cold.mstop == 120
    np.testing.assert_allclose(grown.predict(data, type="link"),
                               cold.predict(data, type="link"), rtol=0, atol=TOL)
    np.testing.assert_allclose(np.asarray(grown._risk), np.asarray(cold._risk),
                               rtol=0, atol=TOL)


def test_continue_to_a_weighted_linear_fit_matches_a_cold_fit():
    data = lss_frame()
    w = (np.arange(len(data["y"])) % 3 != 0).astype(float)
    model = mb.gamboost("y ~ bols(x)", data=data, family=Gaussian(), weights=w,
                        control=boost_control(20, 0.1))
    grown = model.continue_to(60).continue_to(100)
    cold = mb.gamboost("y ~ bols(x)", data=data, family=Gaussian(), weights=w,
                       control=boost_control(100, 0.1))
    np.testing.assert_allclose(grown.predict(data, type="link"),
                               cold.predict(data, type="link"), rtol=0, atol=TOL)


def test_continue_to_a_gram_route_fit_refits_and_matches_a_cold_fit():
    # A fit on the Gram route keeps no residual, so continue_to refits it from
    # zero. The tuner therefore refits such folds itself rather than resuming.
    data = lss_frame()
    model = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=Gaussian(),
                        control=boost_control(20, 0.1))
    assert len(model._residual) == 0
    grown = model.continue_to(40)
    cold = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=Gaussian(),
                       control=boost_control(40, 0.1))
    np.testing.assert_allclose(grown.predict(data, type="link"),
                               cold.predict(data, type="link"), rtol=0, atol=TOL)


def test_boundary_share_is_the_documented_five_percent():
    assert BOUNDARY_SHARE == 0.05
