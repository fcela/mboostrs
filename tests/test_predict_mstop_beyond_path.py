"""predict(mstop=k) beyond the fitted path is refused, not clipped (issue #548)."""

import numpy as np
import pytest

import mboostrs as mb


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = np.sin(3 * x) + rng.standard_normal(n)
    return {"y": y, "x": x}


@pytest.mark.parametrize("fit_fn", [mb.gamboost, mb.glmboost])
def test_predict_beyond_the_fitted_path_is_refused(fit_fn):
    data = _data()
    formula = "y ~ bbs(x, df=4)" if fit_fn is mb.gamboost else "y ~ bols(x)"
    fit = fit_fn(formula, data, family=mb.Gaussian())
    with pytest.raises(ValueError, match=f"beyond the model's fitted {fit.mstop} steps"):
        fit.predict(newdata={"x": data["x"][:5]}, mstop=fit.mstop + 400)
    with pytest.raises(ValueError, match="beyond the model's fitted"):
        fit.predict(mstop=fit.mstop + 400)


@pytest.mark.parametrize("fit_fn", [mb.gamboost, mb.glmboost])
def test_predict_at_or_below_the_fitted_path_is_unchanged(fit_fn):
    data = _data()
    formula = "y ~ bbs(x, df=4)" if fit_fn is mb.gamboost else "y ~ bols(x)"
    fit = fit_fn(formula, data, family=mb.Gaussian())
    at = fit.predict(newdata={"x": data["x"][:5]}, mstop=fit.mstop)
    default = fit.predict(newdata={"x": data["x"][:5]})
    np.testing.assert_allclose(at, default)
    below = fit.predict(newdata={"x": data["x"][:5]}, mstop=fit.mstop // 2)
    assert np.all(np.isfinite(below))
