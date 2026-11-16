"""Issue #443: aipw(interval="bootstrap") resamples the whole procedure."""

import warnings

import numpy as np
import pytest
from test_aipw_regression_456 import _SPEC, _failing_design

import mboostrs as mb
from mboostrs import causal


def _fit(**kw):
    frame, _ = _failing_design(n=500, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.aipw(frame, _SPEC, folds=3, seed=5, mstop=20, tune=False,
                       outcome_model="pooled", **kw)


def test_bootstrap_interval_is_percentile_of_draws():
    r = _fit(interval="bootstrap", bootstrap_draws=8, level=0.8)
    b = r.diagnostics["bootstrap"]
    draws = b["draws"]
    assert draws.shape == (8,) and b["n_failed"] == 0 and b["n_used"] == 8
    lo, hi = np.quantile(draws, [0.1, 0.9])
    assert r.interval == pytest.approx((lo, hi))
    assert r.standard_error == pytest.approx(draws.std(ddof=1))
    assert b["influence_standard_error"] is not None
    assert r.diagnostics["interval_method"] == "bootstrap"


def test_default_is_influence_and_a_seed_fixes_the_draws():
    default = _fit()
    assert default.diagnostics["bootstrap"] is None
    one = _fit(interval="bootstrap", bootstrap_draws=4)
    two = _fit(interval="bootstrap", bootstrap_draws=4)
    np.testing.assert_array_equal(one.diagnostics["bootstrap"]["draws"],
                                  two.diagnostics["bootstrap"]["draws"])
    assert one.estimate == default.estimate


def test_failed_draws_are_counted(monkeypatch):
    calls = {"n": 0}
    original = causal.aipw.__wrapped__

    def flaky(*a, **k):
        calls["n"] += 1
        if k.get("interval") == "influence" and calls["n"] in (3, 4):
            raise ValueError("synthetic failure")
        return original(*a, **k)

    monkeypatch.setattr(causal.aipw, "__wrapped__", flaky)
    # the primary call is also routed through the wrapper, not __wrapped__
    with pytest.warns(RuntimeWarning, match="failed and were dropped"):
        r = mb.aipw(*_args(), folds=3, seed=5, mstop=20, tune=False,
                    outcome_model="pooled", interval="bootstrap", bootstrap_draws=6)
    assert r.diagnostics["bootstrap"]["n_failed"] == 2
    assert r.diagnostics["bootstrap"]["n_used"] == 4


def _args():
    frame, _ = _failing_design(n=500, seed=3)
    return frame, _SPEC


def test_bootstrap_refuses_repeats_and_bad_interval():
    with pytest.raises(ValueError, match="repeats=1"):
        _fit(interval="bootstrap", repeats=2)
    with pytest.raises(ValueError, match="interval must be"):
        _fit(interval="wald")


def test_cost_multiplies_by_draws():
    frame, _ = _failing_design(n=300, seed=3)
    base = mb.estimate_cost("aipw", frame, _SPEC, folds=3)
    boot = mb.estimate_cost("aipw", frame, _SPEC, folds=3, interval="bootstrap",
                            bootstrap_draws=10)
    assert boot.fits == base.fits * 11
