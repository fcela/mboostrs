"""A non-cyclic LSS fit is refitted at its total number of updates (issue #68).

`LSSModel.mstop` reports per-parameter counts and collapses equal counts to
one integer. For a non-cyclic fit with 45 updates of each of two parameters
that integer is 45 while the total is 90; `update()`, `continue_to()` and
the default `cvrisk` grid read it as the total and halved the model.
"""

import numpy as np
import pytest

import mboostrs as mb

FORMULAS = {"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"}


@pytest.fixture(scope="module")
def tied():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-1, 1, n)
    data = {"x": x, "y": np.sin(2 * x) + np.exp(0.5 * x) * rng.normal(size=n)}
    model = mb.gamboostLSS(FORMULAS, data=data, mstop=90, family=mb.GaussianLSS(),
                           method="noncyclic", stabilization="MAD",
                           nu={"mu": 0.1, "sigma": 0.02}, response_scale=None)
    # The fixture is only useful while the counts tie.
    assert model.mstop == 45
    return model, data


def test_update_keeps_the_total(tied):
    model, data = tied
    weights = np.ones(len(data["y"]))
    weights[0] = 2.0
    refit = model.update(weights=weights)
    assert len(refit.risk_) - 1 == 90


def test_continue_to_refuses_to_shrink(tied):
    model, _ = tied
    with pytest.raises(ValueError, match="only grows"):
        model.continue_to(60)
    assert len(model.continue_to(100).risk_) - 1 == 100


def test_default_cvrisk_grid_reaches_the_total(tied):
    model, data = tied
    folds = np.ones((len(data["y"]), 1))
    folds[:100, 0] = 0
    cv = model.cvrisk(folds=folds)
    assert max(point[0] for point in cv.grid) == 90
