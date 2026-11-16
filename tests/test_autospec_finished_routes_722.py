"""autospec compares the finished routes before adopting a distributional
family (#722).

A distributional candidate that passes the family probe's checks has its
shape, pair and residual-tree probes run on `mu`; the mean family's probes
run in full; and the two finished models are scored on the same folds. The
distributional model is adopted only when it passes the log-score check and
the mean check against the finished mean model.
"""

import importlib

import numpy as np

import mboostrs as mb

# `mboostrs.autospec` the attribute is the function; the module is imported by name.
module = importlib.import_module("mboostrs.autospec")

N = 400


def _gamma_step_in_mean(seed):
    """Gamma data whose mean has a step in x and whose shape moves with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, N)
    z = rng.normal(size=N)
    mu = np.exp(0.4 + 1.2 * (x > 0.3) + 0.2 * z)
    shape = np.exp(0.8 + 0.7 * x)
    return {"y": rng.gamma(shape, mu / shape), "x": x, "z": z}


def _gamma_moving_shape(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, N)
    mu = np.exp(0.4 + 0.5 * x)
    shape = 1.0 / np.exp(0.1 + 0.45 * x) ** 2
    return {"y": rng.gamma(shape, mu / shape), "x": x}


def test_both_finished_routes_are_scored_and_the_winner_is_recorded():
    rec = mb.autospec(_gamma_step_in_mean(3), "y", budget=200, mstop=100, seed=0)
    record = rec.evidence["finished_routes"]

    assert record["compared"]
    assert record["winner"] == "distributional"
    assert isinstance(rec.family, mb.GammaLSS)
    for key in ("lss_score", "mean_score", "lss_mse", "mean_mse"):
        assert np.isfinite(record[key])
    assert record["log_score_check"] == "passed"
    assert record["mean_check"] == "passed"
    assert "nats/row" in record["sentence"]
    assert "GammaLSS" in rec.evidence["family"].sentence


def test_the_budget_is_split_and_never_exceeded():
    rec = mb.autospec(_gamma_step_in_mean(3), "y", budget=200, mstop=100, seed=0)
    split = rec.evidence["finished_routes"]["budget"]

    assert split["set_aside_for_comparison"] == 25      # five folds x five
    expected = (split["remaining_at_family_check"]
                - split["set_aside_for_comparison"]) // 2
    assert split["distributional_route_share"] == expected
    assert split["distributional_route_fits"] <= split["distributional_route_share"]
    assert rec.fits <= 200


def test_the_mean_route_stands_when_it_wins_the_comparison(monkeypatch):
    real = module.compare_finished_routes

    def mean_wins(*args, **kwargs):
        out = real(*args, **kwargs)
        return {**out, "winner": "mean", "log_score_check": "failed"}

    monkeypatch.setattr(module, "compare_finished_routes", mean_wins)
    data = _gamma_step_in_mean(3)
    rec = mb.autospec(data, "y", budget=200, mstop=100, seed=0)

    assert rec.evidence["finished_routes"]["winner"] == "mean"
    assert not hasattr(rec.family, "param_names")
    assert rec.evidence["family"].recommended is rec.family
    assert type(rec.fit(data)).__name__ != "LSSModel"


def test_a_short_budget_keeps_the_family_probe_verdict():
    """Below the fits a comparison needs, the candidate's own checks decide
    and the evidence says the routes were not compared."""
    rec = mb.autospec(_gamma_moving_shape(7), "y", budget=50, mstop=100, seed=0)
    record = rec.evidence["finished_routes"]

    assert isinstance(rec.family, mb.GammaLSS)
    assert record["compared"] is False
    assert "fits remained" in record["reason"]
