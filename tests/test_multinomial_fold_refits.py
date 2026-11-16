"""Multinomial fold refits prepare the learners the fit prepared (issue #98).

The fit sends a Multinomial `bbs` to the core in its Kronecker form (penalty
K + I, df calibrated over the class blocks). cvrisk and stability selection
built an ordinary `bbs` instead, whose single-class lambda the Kronecker lift
then reused, so each fold fitted a different smoother: fold 0's held-out
negative log-likelihood per observation was 1.0627 at step 10 against 1.0885
for the same fold refitted with gamboost(weights=). Linear learners were
unaffected.
"""

import numpy as np
import pytest

import mboostrs as mb

CLASSES = ["a", "b", "c"]


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    eta = np.column_stack([np.sin(2 * x), -x + z, np.zeros(n)])
    p = np.exp(eta)
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(CLASSES, p=pi) for pi in p])
    return {"y": y, "x": x, "z": z}


@pytest.mark.parametrize("formula", ["y ~ bbs(x) + bols(z)", "y ~ bbs(x, df=6) + bbs(z)"])
def test_cvrisk_fold_equals_a_weighted_refit(formula):
    d = _data()
    n = len(d["y"])
    control = mb.boost_control(60, 0.1)
    folds = np.asarray(mb.cv_weights(n, "kfold", 5))
    fit = mb.gamboost(formula, data=d, family=mb.Multinomial(), control=control)
    risk = np.asarray(fit.cvrisk(folds=folds).risk_matrix)
    codes = np.array([CLASSES.index(v) for v in d["y"]])
    for fold in (0, 3):
        weights = folds[:, fold]
        held = weights == 0
        refit = mb.gamboost(formula, data=d, family=mb.Multinomial(), weights=weights,
                            control=control)
        for k in (1, 10, 60):
            probs = np.asarray(refit[k].predict(newdata=d, type="response"))
            nll = -np.log(probs[np.arange(n), codes])[held].sum() / held.sum()
            assert risk[fold, k] == pytest.approx(nll, rel=1e-9), (fold, k)


def test_stability_selection_fold_paths_equal_weighted_refits():
    from mboostrs.stabsel import _run_gamboost_specs_selected

    d = _data()
    n = len(d["y"])
    control = mb.boost_control(40, 0.1)
    formula = "y ~ bbs(x) + bbs(z) + bols(x) + bols(z)"
    fit = mb.gamboost(formula, data=d, family=mb.Multinomial(), control=control)
    folds = np.asarray(mb.cv_weights(n, "subsampling", 3, seed=4))
    paths = _run_gamboost_specs_selected(fit, folds, 40)
    for fold in range(3):
        refit = mb.gamboost(formula, data=d, family=mb.Multinomial(),
                            weights=folds[:, fold], control=control)
        assert list(paths[fold]) == list(refit._selected), fold
