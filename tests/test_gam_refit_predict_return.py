"""Return types of `GamRefit.predict` (issue #487, item 12).

Without ``interval=True`` the method returns a NumPy array. With it, the
default ``output="auto"`` returns a Polars frame with the columns
``estimate``, ``se``, ``lower``, ``upper`` and ``outside_range``.
"""

import warnings

import numpy as np
import polars as pl

import mboostrs as mb

INTERVAL_COLUMNS = ["estimate", "se", "lower", "upper", "outside_range"]


def _refit_and_frame():
    rng = np.random.default_rng(487)
    n = 300
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(3 * x) + 0.5 * z + rng.normal(0, 0.3, n)
    data = {"x": x, "z": z, "y": y}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, control=mb.boost_control(200))
        refit = mb.gam_refit(model, which="all")
    newdata = {"x": x[:12], "z": z[:12]}
    return refit, newdata


def test_predict_without_interval_returns_a_numpy_array():
    refit, newdata = _refit_and_frame()
    eta = refit.predict(newdata=newdata)

    assert isinstance(eta, np.ndarray)
    assert eta.shape == (12,)


def test_predict_with_interval_returns_a_polars_frame_with_the_documented_columns():
    refit, newdata = _refit_and_frame()
    frame = refit.predict(newdata=newdata, interval=True)

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == INTERVAL_COLUMNS
    assert frame.height == 12
    assert frame["outside_range"].dtype == pl.Boolean
    assert (frame["lower"] <= frame["estimate"]).all()
    assert (frame["estimate"] <= frame["upper"]).all()


def test_predict_with_interval_and_records_output_returns_dicts_with_the_same_keys():
    refit, newdata = _refit_and_frame()
    rows = refit.predict(newdata=newdata, interval=True, output="records")

    assert isinstance(rows, list)
    assert len(rows) == 12
    assert list(rows[0]) == INTERVAL_COLUMNS
