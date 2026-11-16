"""Regression tests for the array-valued predict/replay marshaling (#285).

The predict/replay boundary now takes contiguous ``float64`` buffers instead
of lists-of-lists. These tests pin the value-preserving property: an integer
or strided ``newdata`` column produces exactly the same eta as its
``ascontiguousarray(..., dtype=float64)`` twin, and the Rust boundary's
list fallback still agrees with the array path.
"""

import numpy as np
import pytest

import mboostrs as mboost
from mboostrs.families import Gaussian


def _fit():
    rng = np.random.default_rng(285)
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.normal(size=n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(scale=0.3, size=n)
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)",
        data={"x1": x1, "x2": x2, "y": y},
        family=Gaussian(),
        control=mboost.boost_control(40, 0.1),
    )
    return model, rng


def _newdata(rng, n=120):
    return rng.uniform(0, 1, n), rng.normal(size=n)


def test_strided_newdata_matches_contiguous():
    model, rng = _fit()
    x1, x2 = _newdata(rng)
    ref = model.predict({"x1": x1, "x2": x2})
    strided1 = np.repeat(x1, 2)[::2]
    strided2 = np.repeat(x2, 2)[::2]
    assert not strided1.flags["C_CONTIGUOUS"]
    assert not strided2.flags["C_CONTIGUOUS"]
    got = model.predict({"x1": strided1, "x2": strided2})
    np.testing.assert_array_equal(got, ref)


def test_integer_dtype_newdata_matches_float():
    model, _ = _fit()
    n = 120
    x1 = np.linspace(0, 1, n)
    x2 = np.arange(n, dtype=np.int64) - n // 2
    ref = model.predict({"x1": x1, "x2": x2.astype(np.float64)})
    got = model.predict({"x1": x1, "x2": x2})
    np.testing.assert_array_equal(got, ref)


def test_pandas_frame_newdata_matches_dict():
    pd = pytest.importorskip("pandas")
    model, rng = _fit()
    x1, x2 = _newdata(rng)
    ref = model.predict({"x1": x1, "x2": x2})
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    got = model.predict(frame)
    np.testing.assert_array_equal(got, ref)


def test_polars_frame_newdata_matches_dict():
    pl = pytest.importorskip("polars")
    model, rng = _fit()
    x1, x2 = _newdata(rng)
    ref = model.predict({"x1": x1, "x2": x2})
    frame = pl.DataFrame({"x1": x1, "x2": x2})
    got = model.predict(frame)
    np.testing.assert_array_equal(got, ref)


def test_rust_boundary_list_fallback_matches_array_path():
    model, rng = _fit()
    store = model._replay_store()
    x1, x2 = _newdata(rng)
    x_data, by_data = model._replay_slices({"x1": x1, "x2": x2})
    selected = model._selected_rust()
    coefs = model._coefs_rust()

    arrays = store.predict_newdata_grouped(0.0, selected, coefs, x_data, None, by_data)
    lists = store.predict_newdata_grouped(
        0.0,
        selected,
        [np.asarray(c).tolist() for c in coefs],
        [np.asarray(v).tolist() for v in x_data],
        None,
        [None if b is None else np.asarray(b).tolist() for b in by_data]
        if by_data is not None else None,
    )
    np.testing.assert_array_equal(lists, arrays)