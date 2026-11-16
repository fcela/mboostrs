"""autospec interaction rules: heredity, forbidden pairs, mandatory terms (#74).

On a main-effects-plus-pair truth and a pure interaction (sin(3 x1 x2)),
over four seeds at budget 150, every rule adopted the true pair; on the pure
interaction the heredity rules cost a little held-out accuracy (test MSE
0.094 against 0.092 with heredity="none"), because they keep the parents'
main effects, which carry nothing there.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb


def _data(seed=0, n=600, pure=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 4))
    y = (np.sin(3 * X[:, 0] * X[:, 1]) if pure
         else np.sin(2 * X[:, 0]) + X[:, 1] + 1.5 * X[:, 0] * X[:, 1])
    return {f"x{j + 1}": X[:, j] for j in range(4)} | {"y": y + rng.normal(0, 0.3, n)}


def _spec(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.autospec(_data(**kwargs.pop("data", {})), "y", budget=150, seed=0,
                           interactions="adopt", **kwargs)


def _pairs(rec):
    return [t for t in rec.evidence.get("adopted_interactions", ()) if "x1" in t and "x2" in t]


def test_a_forbidden_pair_is_neither_listed_nor_adopted():
    rec = _spec(forbid=[("x2", "x1")])
    assert not _pairs(rec)
    assert all({lead.left, lead.right} != {"x1", "x2"} for lead in rec.evidence["interactions"])
    assert "forbidden pairs x1 x x2" in rec.report()


def test_strong_heredity_keeps_both_parents():
    rec = _spec(data={"pure": True}, heredity="strong")
    assert _pairs(rec)
    terms = [t.strip() for t in rec.formula.split("~")[1].split("+")]
    mains = {t for t in terms if "%X%" not in t}
    assert any("x1" in t for t in mains) and any("x2" in t for t in mains)
    assert "heredity strong" in rec.report()


def test_mandatory_columns_survive_deselection():
    # x4 is noise, and deselection drops it unless it is mandatory.
    assert "x4" not in _spec().formula
    assert "x4" in _spec(mandatory=["x4"]).formula


def test_bad_arguments_refuse():
    with pytest.raises(ValueError, match="heredity"):
        _spec(heredity="sometimes")
    with pytest.raises(ValueError, match="mandatory"):
        _spec(mandatory=["nope"])
