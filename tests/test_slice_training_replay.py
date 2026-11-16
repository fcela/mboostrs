"""Slicing a gamboost path rebuilds the training fit by one product per learner.

`model[:k]` and `with_mstop(k)` recompute the in-sample linear predictor at
the shorter path. Every learner but a tree is linear in its coefficients, so
the steps that chose it are summed first and applied once; a tree's steps are
replayed one by one. The sliced fit must equal a cold fit to `k` steps for
every learner kind the store holds, and the fit at the full path must equal
the model's own fitted values.
"""

import numpy as np
import pytest

import mboostrs as mb


def _data(n=400, seed=3):
    rng = np.random.default_rng(seed)
    d = {
        "x": rng.uniform(0.0, 1.0, n),
        "u": rng.uniform(0.0, 1.0, n),
        "v": rng.uniform(0.0, 1.0, n),
        "z": rng.normal(size=n),
        "g": np.array(["a", "b", "c", "d"])[rng.integers(0, 4, n)],
    }
    d["y"] = (np.sin(6 * d["x"]) + np.cos(4 * d["u"] * d["v"]) + 0.3 * d["z"]
              + (d["g"] == "b") * 0.5 + rng.normal(scale=0.2, size=n))
    return d


FORMULAS = [
    "y ~ bspatial(u, v, knots=6, df=6) + bbs(x, df=4) + bols(z)",
    "y ~ bbs(x, df=4) + bols(g) + brandom(g) + bbs(x, by=z, df=4)",
    "y ~ bbs(x, df=4) + btree(u, v) + bols(z)",
]


@pytest.mark.parametrize("formula", FORMULAS)
@pytest.mark.parametrize("k", [1, 37, 150])
def test_sliced_fit_equals_a_cold_fit(formula, k):
    data = _data()
    long = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                       control=mb.boost_control(300, 0.1))
    cold = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                       control=mb.boost_control(k, 0.1))
    sliced = long.with_mstop(k)
    assert sliced.selected == cold.selected
    np.testing.assert_allclose(sliced.fitted_, cold.fitted_, rtol=0, atol=1e-10)
    np.testing.assert_allclose(long[:k].fitted_, cold.fitted_, rtol=0, atol=1e-10)


@pytest.mark.parametrize("formula", FORMULAS)
def test_replay_to_the_full_path_matches_the_fitted_values(formula):
    data = _data()
    model = mb.gamboost(formula, data=data, family=mb.Gaussian(),
                        control=mb.boost_control(200, 0.1))
    replayed = model._predict_training(model.mstop)
    np.testing.assert_allclose(replayed, model.fitted_, rtol=0, atol=1e-10)


def test_sliced_poisson_fit_matches_on_the_link_scale():
    data = _data()
    rng = np.random.default_rng(0)
    data["count"] = rng.poisson(np.exp(0.5 + np.sin(6 * data["x"])))
    formula = "count ~ bspatial(u, v, knots=6, df=6) + bbs(x, df=4)"
    long = mb.gamboost(formula, data=data, family=mb.Poisson(),
                       control=mb.boost_control(250, 0.1))
    cold = mb.gamboost(formula, data=data, family=mb.Poisson(),
                       control=mb.boost_control(90, 0.1))
    np.testing.assert_allclose(long[:90].fitted_, cold.fitted_, rtol=0, atol=1e-10)
