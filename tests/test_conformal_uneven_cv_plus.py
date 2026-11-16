"""CV+ reports covariate-uneven coverage, as split conformal does (#543)."""
import warnings

import numpy as np

import mboostrs as mb


def _fit(scale, seed=0, n=1000):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + scale(x1) * rng.standard_normal(n)
    return mb.gamboost("y ~ bbs(x1, df=4) + bols(x2)", {"y": y, "x1": x1, "x2": x2},
                       family=mb.Gaussian())


def _uneven_warnings(fit, method):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.conformal_interval(fit, newdata={"x1": np.zeros(3), "x2": np.zeros(3)},
                              level=0.9, seed=0, method=method)
    return [str(w.message) for w in caught if "coverage is uneven" in str(w.message)]


def test_cv_plus_warns_when_scale_grows_with_a_covariate():
    fit = _fit(lambda x1: 0.3 + 1.5 * np.abs(x1))
    messages = _uneven_warnings(fit, "cv+")
    assert messages and messages[0].startswith("CV+ coverage is uneven")


def test_cv_plus_is_silent_on_homoscedastic_noise():
    fit = _fit(lambda x1: np.ones_like(x1))
    assert _uneven_warnings(fit, "cv+") == []


def test_cv_plus_reports_the_scale_model_view():
    # The fitted log-residual scale is binned on the kept rows' out-of-fold
    # predictions, so CV+ names the scale heteroscedasticity as such, not
    # only through the covariate third that happens to align (#543).
    fit = _fit(lambda x1: 0.3 + 1.5 * np.abs(x1))
    messages = _uneven_warnings(fit, "cv+")
    assert messages and "fitted scale" in messages[0]
