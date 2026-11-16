"""Stopping and returned fits must follow the same update path (#659)."""
import importlib

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


@pytest.mark.parametrize('family,rate', [(mb.Gaussian(), .1), (mb.Poisson(), .02),
                                        (mb.Binomial(), .02), (mb.GammaReg(), .02)])
def test_actual_tuning_and_refit_use_the_same_rate(monkeypatch, family, rate):
    module = importlib.import_module('mboostrs.discover')
    rng = np.random.default_rng(659)
    x = rng.uniform(-1, 1, 160)
    if isinstance(family, mb.Poisson):
        y = rng.poisson(np.exp(2+.7*x))
    elif isinstance(family, mb.Binomial):
        y = rng.binomial(1, 1/(1+np.exp(-3*x)))
    elif isinstance(family, mb.GammaReg):
        y = rng.gamma(5, np.exp(.7*x)/5)
    else:
        y = x+rng.normal(0, .2, len(x))
    frame = pl.DataFrame({'x': x, 'y': y})
    captured, original = [], module.tune_mstop
    estimates, original_cost = [], module.estimate_cost

    def record(*args, **kwargs):
        tuned = original(*args, **kwargs)
        captured.append((kwargs.get('nu', .1), tuned))
        return tuned

    monkeypatch.setattr(module, 'tune_mstop', record)

    def cost(call, *args, **kwargs):
        if call == 'tune_mstop':
            estimates.append(kwargs.get('nu', .1))
        return original_cost(call, *args, **kwargs)

    monkeypatch.setattr(module, 'estimate_cost', cost)
    result = mb.discover(frame, 'y', family=family, spread=False, interactions=False,
                         seed=659, max_mstop=100, mandatory_terms=['x'])
    assert captured
    assert all(nu == rate for nu, _ in captured)
    assert estimates and all(nu == rate for nu in estimates)
    assert result.model._control.nu == rate
    tuned = captured[-1][1]
    # The retained tuning fit, truncated at its own stop, must predict exactly
    # as the returned fit. This catches path mismatch beyond the passed keyword.
    np.testing.assert_allclose(result.model.predict(frame, type='response'),
                               tuned.model.predict(frame, type='response'), atol=1e-12, rtol=0)


def test_rate_is_retained_when_the_tuning_grid_doubles(monkeypatch):
    from types import SimpleNamespace

    module = importlib.import_module('mboostrs.discover')
    analysis = object.__new__(module._Analysis)
    analysis.response, analysis.frame, analysis.family = 'y', {}, mb.Poisson()
    analysis.folds, analysis.seed, analysis.na_action = 5, 659, 'error'
    analysis.max_mstop, analysis.fold_extrapolation = 200, 'clip'
    analysis.extra_warnings = []
    analysis._admit_tune = lambda *_: True
    analysis._note = lambda *_a, **_k: None
    calls = []

    def tuner(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(mstop=kwargs['start']-10,
                               at_boundary=kwargs['start'] < 200)

    monkeypatch.setattr(module, 'tune_mstop', tuner)
    analysis._tune_mean('bols(x)')
    assert [c['start'] for c in calls] == [100, 200]
    assert [c['nu'] for c in calls] == [.02, .02]
    assert analysis.mstop == 190
