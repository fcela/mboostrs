"""Every model kind survives pickle and the JSON artifact with equal predictions.

A model with a `btree` term saved fine and then failed to load: the
decoder passed `Btree.categorical`, an attribute the fit sets, back to the
constructor. This matrix covers each learner kind, response transforms,
families, binning and the distributional models.
"""

import pickle
import warnings

import numpy as np
import pytest

import mboostrs as mb


def _data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, n)
    z = rng.normal(size=n)
    f = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    y = np.sin(3 * x) + 0.5 * z + (f == "b") + rng.normal(0, 0.3, n)
    return {"x": x, "z": z, "g": rng.integers(0, 4, n), "f": f, "y": y,
            "cnt": rng.poisson(np.exp(0.5 + x)).astype(float),
            "yb": (y > 0.3).astype(float)}


CASES = {
    "glm": lambda d: mb.glmboost("y ~ x + f", data=d),
    "glm binomial": lambda d: mb.glmboost("yb ~ x + z", data=d, family=mb.Binomial()),
    "bbs bols": lambda d: mb.gamboost("y ~ bbs(x) + bols(z)", data=d),
    "brandom": lambda d: mb.gamboost("y ~ bbs(x) + brandom(g)", data=d),
    "tensor": lambda d: mb.gamboost("y ~ bbs(x) %X% bbs(z)", data=d),
    "by": lambda d: mb.gamboost("y ~ bbs(x, by=z)", data=d),
    "btree numeric": lambda d: mb.gamboost("y ~ btree(x, z)", data=d),
    "btree factor": lambda d: mb.gamboost("y ~ bbs(x) + btree(z, f, maxdepth=2)", data=d),
    "bmono": lambda d: mb.gamboost("y ~ bmono(x)", data=d),
    "transform": lambda d: mb.gamboost("y ~ bbs(scale(x))", data=d),
    "log1p response": lambda d: mb.gamboost("log1p(cnt) ~ bbs(x)", data=d),
    "nbinomial": lambda d: mb.gamboost("cnt ~ bbs(x)", data=d, family=mb.NBinomial()),
    "binned": lambda d: mb.gamboost("y ~ bbs(x) + bbs(z)", data=d, bins=32),
    "gamboostLSS": lambda d: mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z)"},
                                            data=d, family=mb.GaussianLSS()),
    "glmboostLSS": lambda d: mb.glmboostLSS({"mu": "y ~ x + z", "sigma": "y ~ x"},
                                            data=d, family=mb.GaussianLSS()),
}


def _predict(model, newdata):
    value = model.predict(newdata=newdata)
    if isinstance(value, dict):
        return np.concatenate([np.asarray(v, dtype=float) for v in value.values()])
    return np.asarray(value, dtype=float)


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_predictions_are_equal(name, tmp_path):
    data = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CASES[name](data)
        newdata = {k: v[:30] for k, v in data.items()}
        expected = _predict(model, newdata)
        np.testing.assert_array_equal(_predict(pickle.loads(pickle.dumps(model)), newdata),
                                      expected)
        for scoring_only in (False, True):
            path = tmp_path / f"model-{scoring_only}.json"
            mb.save_model(model, path, scoring_only=scoring_only)
            np.testing.assert_array_equal(_predict(mb.load_model(path), newdata), expected)
