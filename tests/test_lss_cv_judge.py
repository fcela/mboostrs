"""LSS cross-validation scores only a FoldScheme's judged rows (issue #61).

A rolling origin trains on a window and scores the block after it; the rows
beyond that block are held out of the fit but not scored (the scheme's judge
mask). LSS cvrisk coerced the scheme to its weight matrix, dropped the mask
and scored those rows too: 1.392 against 1.138 for the judged rows alone on
the fixture below.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.gamlss import diagonal_grid

FORMULA = {"mu": "y ~ bbs(x)", "sigma": "y ~ bols(x)"}


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    return {"x": x, "y": rng.normal(np.sin(4 * x), 0.3 + 0.5 * x)}


def _judged_nll(data, scheme, mstop, **fit_kwargs):
    family = mb.GaussianLSS()
    values = []
    for b in range(scheme.n_folds):
        fit = mb.gamboostLSS(FORMULA, data=data, families=family,
                             control=mb.boost_control(mstop),
                             weights=scheme.weights[:, b], **fit_kwargs)
        probs = fit.predict(newdata=data, type="response")
        values.append(-np.log(family.pdf(data["y"], probs))[scheme.judge[:, b]].mean())
    return float(np.mean(values))


def test_cyclic_cvrisk_scores_judged_rows_only():
    data = _data()
    scheme = mb.rolling_folds(200, B=4)
    assert scheme.judge is not None
    model = mb.gamboostLSS(FORMULA, data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(50))
    curve = model.cvrisk(folds=scheme, grid=diagonal_grid(50, 2, length_out=51)).aggregate
    assert curve[-1] == pytest.approx(_judged_nll(data, scheme, 50), rel=1e-10)


def test_noncyclic_cvrisk_scores_judged_rows_only():
    data = _data()
    scheme = mb.rolling_folds(200, B=3)
    model = mb.gamboostLSS(FORMULA, data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(60), method="noncyclic")
    result = model.cvrisk(folds=scheme, grid=[60])
    expected = _judged_nll(data, scheme, 60, method="noncyclic")
    assert result.aggregate[-1] == pytest.approx(expected, rel=1e-8)


def test_glmboostlss_refuses_a_judge_mask_it_cannot_apply():
    data = _data()
    with pytest.raises(NotImplementedError, match="judged rows"):
        mb.cvrisk_glmboostLSS({"mu": "y ~ x", "sigma": "y ~ x"}, data=data,
                              families=mb.GaussianLSS(), control=mb.boost_control(20),
                              folds=mb.rolling_folds(200, B=3))
