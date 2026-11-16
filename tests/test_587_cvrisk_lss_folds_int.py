"""`cvrisk_gamboostLSS(folds=5)` accepts an integer fold count (issue #587).

Passing an integer used to reach `_fold_design`, which read the 0-d array as a
bootstrap matrix and indexed ``shape[1]``, so the call died with
``IndexError: tuple index out of range``. An integer now builds k-fold weights
internally, matching ``mb.cv_weights(n, type='kfold', B=k)``, as the non-LSS
``cvrisk`` does.
"""

import numpy as np
import pytest

import mboostrs as mb


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    return {
        "y": rng.normal(1 + z, np.exp(0.3 * z)),
        "x": z,
        "z": rng.uniform(-1, 1, n),
    }


def _smooth():
    return {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bbs(z, df=4)"}


def test_cvrisk_gamboostlss_accepts_an_integer_fold_count():
    data = _data()
    control = mb.boost_control(20, 0.1)
    cv = mb.cvrisk_gamboostLSS(_smooth(), data, family=mb.GaussianLSS(),
                               folds=5, control=control)
    risk = np.asarray(cv.risk_matrix, dtype=float)
    assert risk.shape[0] == 5
    assert np.all(np.isfinite(risk))


def test_integer_folds_build_the_same_kfold_matrix_as_cv_weights():
    data = _data()
    control = mb.boost_control(20, 0.1)
    integer = mb.cvrisk_gamboostLSS(_smooth(), data, family=mb.GaussianLSS(),
                                    folds=5, control=control)
    explicit = mb.cvrisk_gamboostLSS(
        _smooth(), data, family=mb.GaussianLSS(),
        folds=mb.cv_weights(len(data["y"]), type="kfold", B=5), control=control)
    np.testing.assert_allclose(
        np.asarray(integer.risk_matrix, dtype=float),
        np.asarray(explicit.risk_matrix, dtype=float), rtol=0, atol=0)


def test_cvrisk_glmboostlss_accepts_an_integer_fold_count():
    data = _data()
    cv = mb.cvrisk_glmboostLSS({"mu": "y ~ x", "sigma": "y ~ z"}, data,
                               family=mb.GaussianLSS(), folds=4,
                               control=mb.boost_control(20, 0.1))
    assert np.asarray(cv.risk_matrix, dtype=float).shape[0] == 4


def test_too_small_integer_fold_count_names_the_cv_weights_call():
    with pytest.raises(ValueError, match=r"cv_weights\(n, type='kfold', B=5\)"):
        mb.cvrisk_gamboostLSS(_smooth(), _data(), family=mb.GaussianLSS(),
                              folds=1, control=mb.boost_control(20, 0.1))
