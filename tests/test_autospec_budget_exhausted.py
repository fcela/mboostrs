"""BudgetExhausted propagates out of the one-term backfit sites in probe_shapes.

The two `except Exception` blocks around `_FoldBackfit` construction and
`backfit.risk_scan` must let BudgetExhausted through instead of falling back
to the full-formula scoring path. Other exceptions keep the fallback.
"""

import numpy as np
import polars as pl
import pytest

from mboostrs import _autospec
from mboostrs._workflow import BudgetExhausted
from mboostrs.autospec import _term_for
from mboostrs._frame import read_column
from mboostrs.families import Gaussian


def _probe_inputs(n=200):
    rng = np.random.default_rng(822)
    x1 = rng.uniform(0, 1, n)
    y = 3.0 * (x1 > 0.5) + rng.normal(size=n)
    frame = pl.DataFrame({"x1": x1, "y": y})
    folds = np.ones((n, 1))
    folds[np.random.default_rng(10_822).permutation(n)[: n // 4], 0] = 0.0
    column = read_column(frame, "x1")
    return frame, column, [_term_for(column, 4.0, "bbs")], folds


def _probe(frame, column, terms, folds):
    return _autospec.probe_shapes(
        frame, "y", [column], terms, Gaussian(), folds=folds, mstop=20, nu=0.1,
        df=4.0, max_fits=400, fixed_names=(), env=None)


def _budget_exhausted():
    return BudgetExhausted("fit", "probe_shapes", 1, {})


def test_budget_exhausted_from_fold_backfit_construction_propagates(monkeypatch):
    frame, column, terms, folds = _probe_inputs()

    def exhausted(*args, **kwargs):
        raise _budget_exhausted()

    monkeypatch.setattr(_autospec, "_FoldBackfit", exhausted)
    with pytest.raises(BudgetExhausted):
        _probe(frame, column, terms, folds)


def test_budget_exhausted_from_risk_scan_propagates(monkeypatch):
    frame, column, terms, folds = _probe_inputs()

    class ExhaustedBackfit:
        made = 0

        def __init__(self, *args, **kwargs):
            pass

        def risk_scan(self, *args, **kwargs):
            raise _budget_exhausted()

    monkeypatch.setattr(_autospec, "_FoldBackfit", ExhaustedBackfit)
    with pytest.raises(BudgetExhausted):
        _probe(frame, column, terms, folds)


def test_other_risk_scan_error_still_falls_back(monkeypatch):
    frame, column, terms, folds = _probe_inputs()
    scored = []

    class FailingBackfit:
        made = 0

        def __init__(self, *args, **kwargs):
            pass

        def risk_scan(self, *args, **kwargs):
            raise RuntimeError("backfit failed")

    def fallback(*args, **kwargs):
        scored.append(args[1])
        return 1.0, 1.0, (), False

    monkeypatch.setattr(_autospec, "_FoldBackfit", FailingBackfit)
    monkeypatch.setattr(_autospec, "_path_held_out_scan", fallback)
    _probe(frame, column, terms, folds)
    assert scored, "fallback scoring path was not used"
