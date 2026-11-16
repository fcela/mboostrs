"""GamlssRefit exposes summary() like the other fit objects (#506)."""
import numpy as np
import polars as pl

import mboostrs as mb


def _refit():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    y = 0.4 * x + np.exp(0.3 * x) * rng.normal(size=n)
    df = pl.DataFrame({"x": x, "y": y})
    model = mb.gamboostLSS("y ~ bbs(x)", data=df, family=mb.GaussianLSS(), mstop=50)
    return mb.gamlss_refit(model)


def test_summary_reports_every_parameter_term_once():
    refit = _refit()
    summary = refit.summary()
    rows = summary.as_dict()["tables"][0]["rows"]
    expected = {(p, t) for p in refit.parameters for t in refit.edf[p]}
    assert {(r[0], r[1]) for r in rows} == expected
    assert len(rows) == len(expected)


def test_summary_values_match_the_refit():
    refit = _refit()
    rows = refit.summary().as_dict()["tables"][0]["rows"]
    for parameter, term, edf, smoothing in rows:
        assert edf == refit.edf[parameter][term]
        assert smoothing == refit.smoothing[parameter][term]
