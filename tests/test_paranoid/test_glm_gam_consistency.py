"""GLMBoost with bols() should match GAMBoost with bols()."""

import mboostrs as mboost
import numpy as np
from numpy.testing import assert_allclose

rng = np.random.default_rng(20250609)
n = 100
x1 = rng.normal(2.0, 3.0, n)
x2 = rng.normal(-1.0, 2.0, n)
y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
data = {"y": y, "x1": x1, "x2": x2}


def test_glm_vs_gam_bols_fitted():
    model_glm = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    model_gam = mboost.gamboost(
        "y ~ bols(x1) + bols(x2)", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    assert_allclose(model_glm.fitted(), model_gam.fitted(), atol=1e-10)


def test_glm_vs_gam_bols_risk():
    model_glm = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    model_gam = mboost.gamboost(
        "y ~ bols(x1) + bols(x2)", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    assert_allclose(model_glm.risk_, model_gam.risk_, atol=1e-10)


def test_glm_vs_gam_bols_selected():
    model_glm = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    model_gam = mboost.gamboost(
        "y ~ bols(x1) + bols(x2)", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )
    assert model_glm.selected == model_gam.selected
