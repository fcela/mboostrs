"""LSS continuation with unequal per-parameter budgets equals a direct fit (#68).

A cyclic fit with mu=30, sigma=10 continued to a scalar or a per-parameter
target must give the fit a direct call with that target gives.
"""

import numpy as np
import pytest

import mboostrs as mb

FORMULA = {"mu": "y ~ bbs(x)", "sigma": "y ~ bols(x)"}


def _data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    return {"x": x, "y": rng.normal(np.sin(x), np.exp(0.3 * x))}


@pytest.mark.parametrize("target", [40, {"mu": 40, "sigma": 20}, {"mu": 30, "sigma": 25}])
def test_continue_to_equals_a_direct_fit(target):
    data = _data()
    model = mb.gamboostLSS(FORMULA, data=data, families=mb.GaussianLSS(),
                           mstop={"mu": 30, "sigma": 10})
    continued = model.continue_to(target)
    direct = mb.gamboostLSS(FORMULA, data=data, families=mb.GaussianLSS(), mstop=target)
    assert continued.mstop == direct.mstop
    for name in ("mu", "sigma"):
        np.testing.assert_array_equal(continued.predict(type="link")[name],
                                      direct.predict(type="link")[name])
