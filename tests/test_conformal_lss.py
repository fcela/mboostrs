"""Conformal intervals for distributional fits (issue #111).

method="distributional" refits the LSS specification on the proper-training
rows, takes the fitted distribution's central band at each calibration row,
and widens it by the conformal quantile of the CQR score, so coverage holds
even when the family is wrong. Over 20 splits of a t(2.5) response fitted as
Gaussian it covered 0.899 at level 0.9; the parametric band covered 0.920,
and 0.885 when the family was right.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb


def _fit(seed, noise):
    rng = np.random.default_rng(seed)
    n = 600
    x = rng.uniform(-1, 1, n)
    y = np.sin(2 * x) + (0.3 + 0.5 * (x > 0)) * noise(rng, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"},
                               data={"x": x[:400], "y": y[:400]},
                               families=mb.GaussianLSS(), control=mb.boost_control(100))
    return model, {"x": x[400:]}, y[400:]


def test_coverage_holds_when_the_family_is_wrong():
    coverage = []
    for seed in range(8):
        model, newdata, y = _fit(seed, lambda rng, n: rng.standard_t(2.5, n))
        ci = mb.conformal_interval(model, newdata=newdata, level=0.9, seed=seed)
        coverage.append(np.mean((y >= ci.lower) & (y <= ci.upper)))
    assert 0.86 <= np.mean(coverage) <= 0.94


def test_the_result_names_its_method_and_guarantee():
    model, newdata, _ = _fit(0, lambda rng, n: rng.normal(size=n))
    ci = mb.conformal_interval(model, newdata=newdata, level=0.8, seed=1)
    assert ci.method == "distributional"
    assert ci.guaranteed_coverage == 0.8
    assert ci.n_calibration == 200 and ci.n_training == 200
    assert np.all(ci.lower <= ci.predicted) and np.all(ci.predicted <= ci.upper)
    explicit = mb.conformal_interval(model, newdata=newdata, level=0.8, seed=1,
                                     method="distributional")
    np.testing.assert_array_equal(explicit.lower, ci.lower)


def test_other_methods_and_unsupported_designs_refuse():
    model, newdata, _ = _fit(0, lambda rng, n: rng.normal(size=n))
    with pytest.raises(NotImplementedError, match="distributional"):
        mb.conformal_interval(model, newdata=newdata, method="cqr")
    with pytest.raises(NotImplementedError, match="tune"):
        mb.conformal_interval(model, newdata=newdata, mstop="tune")
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 2, 200)
    transformed = mb.gamboostLSS({"mu": "y ~ bbs(scale(x))", "sigma": "y ~ bols(x)"},
                                 data={"x": x, "y": x + rng.normal(size=200)},
                                 families=mb.GaussianLSS(), control=mb.boost_control(30))
    # A fitted transform (scale) is estimated again on the proper-training rows.
    interval = mb.conformal_interval(transformed)
    assert "scale(" in interval.calibration_source
    assert "re-estimated on the proper-training rows" in interval.calibration_source
    glm =mb.glmboost("y ~ x", data={"x": x, "y": x + rng.normal(size=200)})
    with pytest.raises(ValueError, match="glmboostLSS/gamboostLSS"):
        mb.conformal_interval(glm, method="distributional")
