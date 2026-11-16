"""Regression tests for #202: autospec must not recommend Poisson when exact
zeros dominate the response.

On zero-inflated counts the family probe runs the count-family scan and
adopts a zero-capable family instead of the dispersion rule's Poisson.
"""
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs._autospec import _linearize_formula


@pytest.fixture()
def zero_inflated_data():
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.normal(size=n)
    mu = np.exp(0.15 + 0.55 * x1 - 0.30 * x2)
    pi = 1 / (1 + np.exp(0.35 - 0.95 * z))
    y = np.where(rng.random(n) < pi, 0, rng.poisson(mu)).astype(float)
    assert (y == 0).mean() > 0.5
    return pl.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y})


@pytest.fixture()
def plain_poisson_data():
    rng = np.random.default_rng(3)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.5 + 0.4 * x1 - 0.3 * x2)).astype(float)
    assert (y == 0).mean() < 0.5
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_linearize_formula():
    assert _linearize_formula("y ~ bbs(x1, df=4.0) + btree(x2)") == \
        "y ~ bols(x1) + bols(x2)"
    assert _linearize_formula("y ~ bols(x1) + x2") == "y ~ bols(x1) + x2"


def test_zero_inflated_recommends_zero_capable_family(zero_inflated_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(zero_inflated_data, "y", budget=100, seed=0)
    name = type(rec.family).__name__
    assert name in {"NBinomial", "ZIPoLSS", "ZeroAdjustedPoissonLSS",
                    "ZINBLSS"}, name
    assert name != "Poisson"
    sentence = rec.evidence["family"].sentence
    assert "exact zeros" in sentence
    assert "PoissonLSS" in sentence  # the scan baseline is named


def test_zero_inflated_fit_works(zero_inflated_data):
    """The recommendation must survive the downstream probes and fit()."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(zero_inflated_data, "y", budget=100, seed=0)
        model = rec.fit(zero_inflated_data)
    assert model is not None


def test_plain_poisson_unaffected(plain_poisson_data):
    """Below the 50% trigger the old behavior is unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(plain_poisson_data, "y", budget=100, seed=0)
    name = type(rec.family).__name__
    assert name in {"Poisson", "NBinomial"}, name
    assert "exact zeros" not in rec.evidence["family"].sentence


def test_scan_rows_report_where_the_cvrisk_minimum_landed(zero_inflated_data):
    """The scan records the step of its out-of-bag risk minimum; equal to
    the budget means the curve was still falling at the cap, so the ranking
    is provisional (#202)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = mb.scan_lss_families(
            "y ~ x1 + x2 + z", zero_inflated_data, support="counts",
            metric="cvrisk", fit_func="glmboostLSS",
            control=mb.boost_control(60, 0.1))
    ok = [r for r in rows
          if r.get("status") == "ok" and r.get("cvrisk") is not None]
    assert ok
    for row in ok:
        assert isinstance(row["cvrisk_mstop"], int)
        assert 1 <= row["cvrisk_mstop"] <= 60
    zip_row = next(r for r in ok if r["family"] == "ZIPoLSS")
    # 60 steps leave the zero-inflation component far from converged: its
    # minimum sits on the cap, which is the boundary signal the router
    # escalates on.
    assert zip_row["cvrisk_mstop"] == 60


def test_route_adopts_only_once_the_comparison_turns(zero_inflated_data):
    """A budget that leaves the risk curves falling at the cap must not
    produce a family verdict; an escalated budget that turns adopts."""
    from mboostrs import _autospec

    y = zero_inflated_data["y"].to_numpy()
    zero_share = float((y == 0).mean())
    formula = "y ~ bbs(x1) + bbs(x2) + bbs(z)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.MonkeyPatch.context() as stalled:
            stalled.setattr(_autospec, "_ZERO_SCAN_BUDGETS", (80,))
            evidence = _autospec._route_zero_inflated(
                zero_inflated_data, "y", formula, None, y, zero_share)
    assert evidence.declined is True
    assert type(evidence.recommended).__name__ == "Poisson"
    assert "declines to choose" in evidence.sentence
    assert "provisional ranking" in evidence.sentence
    assert "80-step cap" in evidence.sentence

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evidence = _autospec._route_zero_inflated(
            zero_inflated_data, "y", formula, None, y, zero_share)
    assert evidence.declined is False
    assert type(evidence.recommended).__name__ == "NBinomial"
    assert "interior at" in evidence.sentence
    assert "PoissonLSS" in evidence.sentence
    # Turning took more than one rung, and the adopted sentence names the
    # budget that actually decided.
    budget = int(evidence.sentence.split("cvrisk, ")[1].split(" steps")[0])
    assert budget in _autospec._ZERO_SCAN_BUDGETS and budget > 80
