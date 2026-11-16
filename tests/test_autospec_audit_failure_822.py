"""A failing row-level audit propagates instead of being accepted (#822)."""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs import _autospec


class AuditFailure(RuntimeError):
    pass


def _step_frame(n=364):
    rng = np.random.default_rng(702)
    x1 = rng.uniform(0, 1, n)
    y = 3.0 * (x1 > 0.5) + rng.normal(size=n)
    folds = np.ones((n, 1))
    folds[np.random.default_rng(10_002).permutation(n)[: n // 4], 0] = 0.0
    return pl.DataFrame({"x1": x1, "y": y}), folds


def _probe(frame, folds):
    from mboostrs._frame import read_column
    from mboostrs.autospec import _term_for

    column = read_column(frame, "x1")
    return _autospec.probe_shapes(
        frame, "y", [column], [_term_for(column, 4.0, "bbs")], mb.Gaussian(),
        folds=folds, mstop=200, nu=0.1, df=4.0, max_fits=400,
        fixed_names=(), env=None)


def test_shape_audit_failure_propagates(monkeypatch):
    frame, folds = _step_frame()

    def fail(*args, **kwargs):
        raise AuditFailure("shape audit")

    monkeypatch.setattr(_autospec, "_shape_row_z", fail)
    with pytest.raises(AuditFailure):
        _probe(frame, folds)


@pytest.mark.parametrize("failing_call", [0, 1])
def test_row_level_check_failure_propagates(monkeypatch, failing_call):
    frame, folds = _step_frame()
    calls = []

    def fail_nth(*args, **kwargs):
        calls.append(1)
        if len(calls) - 1 == failing_call:
            raise AuditFailure("row losses")
        return np.zeros(3), 1

    monkeypatch.setattr(_autospec, "_fold_row_losses", fail_nth)
    with pytest.raises(AuditFailure):
        _autospec._held_out_pair_z(
            frame, "y", "y ~ bbs(x1)", 100, 100, 0.1, "y ~ btree(x1)", 100,
            100, 0.1, mb.Gaussian(), folds)
