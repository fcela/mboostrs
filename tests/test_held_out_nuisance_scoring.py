"""Held-out family loss uses the nuisance value fitted on the training rows.

NBinomial's dispersion, GammaReg's shape and PropOdds' thresholds are
estimated while boosting. The core `cvrisk` scores each fold at the fold
fit's own estimate; the Python assistants built the family from its
constructor arguments and scored at the default instead (issue #62).
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.explore import cv_risk


def _two_folds(n):
    folds = np.ones((n, 2))
    folds[: n // 2, 0] = 0
    folds[n // 2:, 1] = 0
    return folds


def _data(family_name, n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    mu = np.exp(1 + x)
    if family_name == "NBinomial":
        y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
        family = mb.NBinomial()
    elif family_name == "GammaReg":
        y = rng.gamma(2.0, mu / 2.0)
        family = mb.GammaReg()
    elif family_name == "Huber":
        # delta unset: the family re-estimates it from the residuals at every
        # step, so a fold must be scored at its own final delta.
        y = np.sin(2 * x) + rng.standard_t(2, n) * 0.3
        family = mb.Huber()
    else:
        y = np.digitize(x + rng.normal(0, 0.5, n), [-0.5, 0.0, 0.5]).astype(float) + 1
        family = mb.PropOdds(n_levels=4)
    return {"x": x, "y": y}, family


@pytest.mark.parametrize("family_name", ["NBinomial", "GammaReg", "PropOdds", "Huber"])
def test_cv_risk_family_score_matches_core_cvrisk(family_name):
    data, family = _data(family_name)
    folds = _two_folds(len(data["y"]))
    full = mb.gamboost("y ~ bols(x)", data=data, family=family,
                       control=mb.boost_control(50))
    core = np.asarray(full.cvrisk(folds=folds).risk_matrix)[:, 50].mean()
    assistant = cv_risk("bols(x)", data, "y", family=family, folds=folds, mstop=50)
    # The core fits each fold through in-bag weights on all rows; the
    # assistant fits the training rows alone. GammaReg's shape is a Brent
    # minimisation whose tolerance separates the two routes by about 1e-5
    # (1.952027 vs 1.952000 on fold 2). Scoring at the constructor default,
    # the previous behaviour, was off by 10% on this data.
    # Huber's delta is the unweighted median absolute residual, as in
    # mboost, so the core's weighted fold fit lets held-out rows into delta
    # while the assistant's row-subset fit does not: about 1e-2 apart here
    # (bols(x) is the block [1, x] for a non-squared loss, as in mboost).
    rtol = {"GammaReg": 1e-5, "Huber": 2e-2}.get(family_name, 1e-10)
    assert assistant == pytest.approx(core, rel=rtol, abs=1e-12)


def test_nbinomial_score_differs_from_constructor_default_dispersion():
    # Guards against the test above passing because the dispersion does not
    # matter on this data: scoring at the constructor default gives another number.
    from mboostrs.explore import _score

    data, family = _data("NBinomial")
    folds = _two_folds(len(data["y"]))
    train = folds[:, 0] > 0
    fit = mb.gamboost("y ~ bols(x)", data={k: v[train] for k, v in data.items()},
                      family=family, control=mb.boost_control(50))
    held = ~train
    eta = np.asarray(fit.predict(newdata={"x": data["x"][held]}, type="link"))
    default = _score(family, data["y"][held], eta, np.ones(held.sum()))
    fold_score = cv_risk("bols(x)", data, "y", family=family,
                         folds=folds[:, :1], mstop=50)
    assert abs(default - fold_score) > 1e-3


def test_fold_local_tuning_path_matches_core_cvrisk():
    data, family = _data("NBinomial")
    folds = _two_folds(len(data["y"]))
    core = mb.tune_mstop("y ~ bols(x)", data, family, folds=folds, start=40,
                         extensions=0, max_mstop=40)
    local = mb.tune_mstop("y ~ bols(x)", data, family, folds=folds, start=40,
                          extensions=0, max_mstop=40, fold_local=True)
    np.testing.assert_allclose(np.asarray(local.risk)[1:], np.asarray(core.risk)[1:],
                               rtol=1e-10)


def test_working_residuals_use_fitted_dispersion():
    # mboost's NBinomial negative gradient: y - (y + theta) * mu / (mu + theta),
    # with theta the dispersion the fit estimated (`nuisance()`).
    data, family = _data("NBinomial")
    fit = mb.gamboost("y ~ bols(x)", data=data, family=family,
                      control=mb.boost_control(50))
    theta = float(np.ravel(fit.nuisance())[0])
    mu = np.exp(np.asarray(fit.predict(type="link")))
    y = data["y"]
    expected = y - (y + theta) * mu / (mu + theta)
    np.testing.assert_allclose(fit.residuals(type="working"), expected, rtol=1e-10)


def test_autospec_held_out_risk_matches_core_cvrisk():
    from mboostrs._autospec import _held_out_risk

    data, family = _data("NBinomial")
    folds = _two_folds(len(data["y"]))
    probe, _ = _held_out_risk(data, "y", "y ~ bols(x)", family, folds, 50, 0.1)
    full = mb.gamboost("y ~ bols(x)", data=data, family=family,
                       control=mb.boost_control(50))
    risk = np.asarray(full.cvrisk(folds=folds).risk_matrix)[:, 50]
    # `_held_out_risk` pools rows; with equal folds that is the fold mean.
    assert probe == pytest.approx(risk.mean(), rel=1e-10)
