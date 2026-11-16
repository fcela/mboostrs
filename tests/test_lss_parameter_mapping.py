"""Per-parameter mappings must name every distribution parameter, and only those.

A StudentTLSS formula mapping without ``df`` raised a bare ``KeyError: 'df'``
from inside the fit, and a misspelled key was ignored.
"""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    return {"x": rng.uniform(size=100), "y": rng.normal(size=100)}


@pytest.mark.parametrize("entry,term", [(mb.gamboostLSS, "bbs(x)"), (mb.glmboostLSS, "x")])
def test_missing_parameter_is_named(data, entry, term):
    with pytest.raises(ValueError, match=r"missing \['df'\]"):
        entry({"mu": f"y ~ {term}", "sigma": f"y ~ {term}"}, data=data,
              family=mb.StudentTLSS())


@pytest.mark.parametrize("entry,term", [(mb.gamboostLSS, "bbs(x)"), (mb.glmboostLSS, "x")])
def test_unknown_parameter_is_named(data, entry, term):
    with pytest.raises(ValueError, match=r"unknown \['sgma'\]"):
        entry({"mu": f"y ~ {term}", "sigma": "y ~ 1", "sgma": "y ~ 1"}, data=data,
              family=mb.GaussianLSS())


def test_mstop_mapping_is_checked(data):
    with pytest.raises(ValueError, match="an mstop mapping"):
        mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}, data=data,
                       family=mb.GaussianLSS(), mstop={"mu": 10})
