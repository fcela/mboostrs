"""Weighted LSS cross-validation refits and scores with the model's weights
(issue #68).

`LSSModel.cvrisk` built its default folds from the model's weights but scored
held-out rows unweighted, and explicit 0/1 folds dropped the weights from the
fold refits too: fold risks were 1.349 against 1.377 for a direct refit of
the same fold. Each fold now equals `gamboostLSS(..., weights=fold * w)`
scored with `w` on its held-out rows, as the standard cvrisk does, with
non-default stabilization and per-parameter step lengths carried through.
"""

import numpy as np
import pytest

import mboostrs as mb

FORMULA = {"mu": "y ~ bbs(x) + bols(z)", "sigma": "y ~ bols(z) + bbs(x)"}
NU = {"mu": 0.1, "sigma": 0.05}


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    return {"x": x, "z": z, "y": rng.normal(np.sin(x), np.exp(0.3 * z))}


def _direct_fold_risk(data, fit_weights, eval_weights, held, mstop, **kw):
    family = mb.GaussianLSS()
    fit = mb.gamboostLSS(FORMULA, data=data, families=family, weights=fit_weights,
                         control=mb.boost_control(mstop), nu=NU, **kw)
    nll = -np.log(family.pdf(data["y"], fit.predict(newdata=data, type="response")))
    return float(np.sum((eval_weights * nll)[held]) / np.sum(eval_weights[held]))


@pytest.mark.parametrize("stabilization", ["MAD", "L2"])
@pytest.mark.parametrize("explicit", [True, False])
def test_cyclic_folds_equal_weighted_refits(stabilization, explicit):
    data = _data()
    w = np.where(np.arange(200) % 5 == 0, 2.0, 1.0)
    model = mb.gamboostLSS(FORMULA, data=data, families=mb.GaussianLSS(), weights=w,
                           control=mb.boost_control(40), nu=NU, stabilization=stabilization)
    if explicit:
        mask = np.asarray(mb.cv_weights(200, "kfold", 4, seed=3))
        result = model.cvrisk(folds=mask, grid=[[40, 40]])
        fit_columns = mask * w[:, None]
    else:
        result = model.cvrisk(grid=[[40, 40]])
        fit_columns = np.asarray(mb.cv_weights(200, type="kfold", B=5, seed=42, weights=w))
    for b in range(fit_columns.shape[1]):
        held = fit_columns[:, b] == 0
        expected = _direct_fold_risk(data, fit_columns[:, b], w, held, 40,
                                     stabilization=stabilization)
        assert result.risk_matrix[b, 0] == pytest.approx(expected, rel=1e-10), b


def test_noncyclic_folds_equal_weighted_refits():
    data = _data()
    w = np.where(np.arange(200) % 5 == 0, 2.0, 1.0)
    model = mb.gamboostLSS(FORMULA, data=data, families=mb.GaussianLSS(), weights=w,
                           control=mb.boost_control(40), nu=NU, method="noncyclic")
    mask = np.asarray(mb.cv_weights(200, "kfold", 3, seed=4))
    result = model.cvrisk(folds=mask, grid=[40])
    for b in range(3):
        held = mask[:, b] == 0
        expected = _direct_fold_risk(data, mask[:, b] * w, w, held, 40, method="noncyclic")
        assert result.risk_matrix[b, -1] == pytest.approx(expected, rel=1e-8), b
