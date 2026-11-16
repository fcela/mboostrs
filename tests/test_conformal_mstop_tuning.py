"""Conformal calibration with the stopping point tuned inside proper training (issue #64).

A model tuned by cross-validation on all its rows has a step count that
depended on the rows split conformal later uses for calibration; the
guarantee assumes it did not. `mstop="tune"` re-chooses the step count for
each proper-training fit from that fit's own rows, and `mstop_source` says
which rule applied.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 240


def _tuned_on_all_rows(y, x):
    data = {"x": x, "y": y}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboost("y ~ bbs(x)", data=data, control=mb.boost_control(400, 0.1))
        return fit[int(fit.cvrisk(type="kfold", B=5, seed=0).optimal_mstop)]


def _pair():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2, 2, N)
    y = np.sin(x) + rng.normal(scale=0.3, size=N)
    # Rows split conformal (seed 1, calib_fraction 0.5) calibrates on.
    calibration = np.random.default_rng(1).permutation(N)[: N // 2]
    altered = y.copy()
    altered[calibration] += rng.normal(scale=3.0, size=calibration.size)
    return _tuned_on_all_rows(y, x), _tuned_on_all_rows(altered, x)


def test_tuned_fits_do_not_depend_on_calibration_responses():
    first, second = _pair()
    assert first.mstop != second.mstop  # the all-row tuning saw them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mb.conformal_interval(first, level=0.8, seed=1, mstop="tune")
        b = mb.conformal_interval(second, level=0.8, seed=1, mstop="tune")
    np.testing.assert_allclose(a.predicted, b.predicted)
    assert "inside each proper-training set" in a.mstop_source


def test_default_path_records_its_assumption():
    first, second = _pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mb.conformal_interval(first, level=0.8, seed=1)
        b = mb.conformal_interval(second, level=0.8, seed=1)
    assert not np.allclose(a.predicted, b.predicted)
    assert "fitted path" in a.mstop_source


@pytest.mark.parametrize("method", ["cv+", "cqr"])
def test_other_methods_accept_tuning(method):
    first, _ = _pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mb.conformal_interval(first, level=0.8, seed=1, method=method,
                                       mstop="tune", n_folds=5)
    assert np.all(result.lower <= result.upper)
    assert "proper-training" in result.mstop_source


def test_invalid_mstop_is_refused():
    first, _ = _pair()
    with pytest.raises(ValueError, match="mstop must be"):
        mb.conformal_interval(first, mstop="auto")
