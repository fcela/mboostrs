"""Coordinate detection must preserve a supported single holdout (#538)."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.autospec import _spatial_cell_scheme
from mboostrs.folds import FoldScheme


def _data():
    rng = np.random.default_rng(538)
    n = 240
    lon, lat = rng.uniform(-1, 1, (2, n))
    y = 3 * (lon > 0) + lat + rng.normal(0, .3, n)
    return {"lon": lon, "lat": lat, "y": y}


def _holdout(structured):
    weights = np.zeros((240, 1))
    weights[:150, 0] = 1
    if not structured:
        return weights
    # Later unused rows carry no validation credit. Observation weights are
    # complete fit weights, and evaluation weights are retained separately.
    base = np.linspace(.5, 1.5, 240)
    judge = np.zeros_like(weights, dtype=bool)
    judge[150:210] = True
    return FoldScheme(weights * base[:, None], weights == 0, "single future holdout",
                      judge=judge, ordered=True, base_weights=base)


@pytest.mark.parametrize("structured", [False, True])
def test_spatial_adapter_leaves_single_holdout_unchanged(structured):
    folds = _holdout(structured)
    columns = [SimpleNamespace(name=c) for c in ("lon", "lat")]
    spatial, cells = _spatial_cell_scheme(
        _data(), columns, np.asarray(folds), folds if structured else None)
    assert spatial is None and cells == 0


@pytest.mark.parametrize("structured", [False, True])
def test_autospec_residual_tree_executes_on_single_holdout(monkeypatch, structured):
    import sys

    module = sys.modules["mboostrs.autospec"]
    original = module._spatial_cell_scheme
    calls = []

    def record(data, modelled, folds, scheme):
        calls.append((np.asarray(folds).copy(), scheme))
        return original(data, modelled, folds, scheme)

    monkeypatch.setattr(module, "_spatial_cell_scheme", record)
    folds = _holdout(structured)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(_data(), "y", family=mb.Gaussian(), folds=folds,
                          budget=60, mstop=40, nu=.1, seed=538,
                          preexploration="off", interactions="report",
                          residual_tree=True, joint_tree=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.asarray(folds))
    if structured:
        assert calls[0][1] is folds
        assert rec.evidence["residual_tree"]["validation"] == "shared single future holdout folds"
    else:
        assert calls[0][1] is None
        assert rec.evidence["residual_tree"]["validation"] == "shared row folds"
    assert rec.fits <= 60
    assert np.isfinite(rec.fit(_data()).predict(_data())).all()
