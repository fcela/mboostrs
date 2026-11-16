"""The real-data harness scores a distributional fit on its location parameter (#838)."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

import mboostrs as mb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benches"))
import bench_realdata


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    y = 1.0 + 2.0 * x + np.exp(0.8 * z) * rng.normal(size=n)
    return pl.DataFrame({"x": x, "z": z, "y": y})


def test_lss_fit_is_scored_on_mu():
    train, test = _data(300, 0), _data(50, 1).drop("y")
    model = mb.gamboostLSS("y ~ bbs(x) + bbs(z)", data=train, families=mb.GaussianLSS(),
                           control=mb.boost_control(mstop=50))
    pred, component = bench_realdata._point_prediction(model, test)
    assert component == "mu"
    expected = model.predict(newdata=test, parameter="mu", type="response")
    np.testing.assert_array_equal(pred, np.asarray(expected, float))
    assert pred.shape == (50,)


def test_point_fit_is_scored_on_response():
    train, test = _data(300, 0), _data(50, 1).drop("y")
    model = mb.gamboost("y ~ bbs(x) + bbs(z)", data=train,
                        control=mb.boost_control(mstop=50))
    pred, component = bench_realdata._point_prediction(model, test)
    assert component == "response"
    np.testing.assert_array_equal(pred, np.asarray(model.predict(newdata=test,
                                                                 type="response"), float))
