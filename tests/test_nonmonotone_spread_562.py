"""Nonmonotone variance must reach the existing held-out scale comparison."""
import numpy as np
import polars as pl

import mboostrs as mb


def _draw(seed=0, n=1000, constant=False):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-1, 1, (2, n))
    sd = np.ones(n) if constant else .3 + 1.5*np.abs(x1)
    y = np.sin(2*np.pi*x1) + .5*x2 + sd*rng.normal(size=n)
    return pl.DataFrame({'x1': x1, 'x2': x2, 'y': y})


def test_u_shaped_scale_has_curve_evidence_with_no_signed_slope():
    data = _draw()
    fit = mb.gamboost('y ~ bbs(x1, df=4) + bbs(x2, df=4)', data)
    first = mb.heteroscedasticity(fit, columns=['x1'])
    second = mb.heteroscedasticity(fit, columns=['x1'])
    driver = first['by_column']['x1']
    assert abs(driver['t']) < 2
    assert driver['curve_removed'] > .3
    assert first == second
    assert first['tested_columns'] == ['x1']


def test_discovery_adopts_a_validated_u_shaped_scale():
    data = _draw()
    new = pl.DataFrame({'x1': [-.9, 0., .9], 'x2': [0., 0., 0.]})
    result = mb.discover(data, 'y', newdata=new, seed=0)
    assert result.spread_formula == 'bbs(x1, df=4)'
    assert result.predictions['sigma'][0] > 2*result.predictions['sigma'][1]
    assert result.predictions['sigma'][2] > 2*result.predictions['sigma'][1]
    nomination = next(row for row in result.log if 'nomination_scores' in row)
    assert nomination['nomination_scores']['x1']['curve_removed'] > 0
    selected = next(row for row in result.log if row['stage'] == 'spread' and row.get('adopted'))
    assert selected['change'] <= -2*selected['se']


def test_curve_nomination_respects_sigma_candidates_and_null_control():
    restricted = mb.discover(_draw(), 'y', seed=0, candidates={'sigma': ['x2']})
    assert restricted.spread_formula is None
    null = mb.discover(_draw(seed=4, constant=True), 'y', seed=0)
    assert null.spread_formula is None
    no_budget = mb.discover(_draw(), 'y', seed=0, budget_seconds=0)
    assert no_budget.spread_formula is None
