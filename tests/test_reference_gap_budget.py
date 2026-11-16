"""Regression tests for #496: ``reference_gap`` must be bounded and never silent.

During the v4 hostile-workflows benchmark the tree reference hung twice on
heteroscedastic data (killed after 9+ and 4+ minutes at 0% CPU) with no
traceback, no timeout and no progress signal. ``reference_gap`` now takes a
``time_budget`` (default 600s): the budget is checked between the computation
stages, and the tree reference scan — the most expensive stage — runs with
the remaining budget as a hard timeout. Exceeding it raises ``TimeoutError``
naming the stage instead of hanging silently.
"""

import time

import numpy as np
import polars as pl
import pytest

from mboostrs.reference_gap import _run_stage, reference_gap


def _heteroscedastic_frame(n=200, seed=496):
    """Small-scale version of the v4 heteroscedastic case from #496.

    Mean 2*x0 + 1.5*sin(pi*x1), sigma 0.25 + 1.5*x2: the variance driver x2
    carries no mean signal, as in the issue's data.
    """
    rng = np.random.default_rng(seed)
    x0, x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    x3, x4 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = 2 * x0 + 1.5 * np.sin(np.pi * x1) + rng.normal(0, 0.25 + 1.5 * x2)
    return pl.DataFrame({"y": y, "x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4})


def test_nonpositive_time_budget_raises():
    frame = _heteroscedastic_frame()
    with pytest.raises(ValueError, match="time_budget"):
        reference_gap("y ~ bbs(x0, df=4)", frame, time_budget=0)
    with pytest.raises(ValueError, match="time_budget"):
        reference_gap("y ~ bbs(x0, df=4)", frame, time_budget=-5)


def test_exhausted_budget_raises_timeout_naming_the_stage():
    # A budget smaller than any real setup can meet must fail fast and say
    # which stage it never reached — never hang silently.
    frame = _heteroscedastic_frame()
    with pytest.raises(TimeoutError, match="specification scan"):
        reference_gap("y ~ bbs(x0, df=4)", frame, time_budget=1e-9,
                      max_mstop=100, folds=3)


def test_heteroscedastic_data_completes_within_budget():
    # The #496 shape at small scale: must return a result, not hang.
    frame = _heteroscedastic_frame()
    started = time.monotonic()
    res = reference_gap("y ~ bbs(x0, df=4) + bbs(x1, df=4)", frame,
                        folds=3, seed=11, max_mstop=300, time_budget=600)
    assert time.monotonic() - started < 600
    assert np.isfinite(res.gap)
    assert res.reference_mstop > 0


def test_run_stage_timeout_names_the_stage():
    with pytest.raises(TimeoutError, match="tree reference scan"):
        _run_stage(lambda: time.sleep(30), 0.05, "tree reference scan", ["a"])


def test_run_stage_returns_value_and_propagates_errors():
    assert _run_stage(lambda: 42, 5.0, "s", []) == 42
    with pytest.raises(ValueError, match="boom"):
        def fail():
            raise ValueError("boom")
        _run_stage(fail, 5.0, "s", [])


def test_run_stage_none_timeout_runs_directly():
    assert _run_stage(lambda: "ok", None, "s", []) == "ok"


def test_default_budget_is_six_hundred_seconds():
    import inspect
    assert inspect.signature(reference_gap).parameters["time_budget"].default == 600
