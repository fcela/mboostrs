"""A restored model with a per-observation offset predicts like the original.

With a vector offset every `bols` term is fitted in the two-column block form
`[1, z]`, because the centered form takes its constant from a scalar offset.
A pickled or loaded model rebuilds its learner store from the stored learner
specifications; that rebuild chose the centered form, read the block-form
coefficients against a one-column design, and moved `predict(newdata=)` by up
to 0.89 on the fit below (issue #819).
"""

import pickle

import numpy as np
import pytest

import mboostrs as mb


def _fit_and_new_rows():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(-2, 2, n)
    z = rng.normal(size=n)
    offset = 0.5 * rng.normal(size=n)
    y = np.sin(x) + 0.5 * z + offset + 0.3 * rng.normal(size=n)
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data={"x": x, "z": z, "y": y},
                      offset=offset, control=mb.boost_control(80, 0.1))
    newdata = {"x": rng.uniform(-2, 2, 30), "z": rng.normal(size=30)}
    new_offset = rng.normal(size=30)
    return fit, newdata, new_offset


@pytest.mark.parametrize("route", ["pickle", "artifact"])
def test_restored_vector_offset_model_predicts_new_rows_identically(route, tmp_path):
    fit, newdata, new_offset = _fit_and_new_rows()
    expected = np.asarray(fit.predict(newdata=newdata, offset=new_offset))
    if route == "pickle":
        restored = pickle.loads(pickle.dumps(fit))
    else:
        path = tmp_path / "model.json"
        mb.save_model(fit, path)
        restored = mb.load_model(path)
    got = np.asarray(restored.predict(newdata=newdata, offset=new_offset))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)
