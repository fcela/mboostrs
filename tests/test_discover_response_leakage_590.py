"""Issue #590: discover must not admit a copy or transform of the response.

``mb.discover`` used every non-response column as a candidate predictor, so a
frame that still held ``wages`` when the response was ``log(wages)`` selected
``bbs(log(wages))`` as a mean effect, with no warning. A unique row identifier
such as ``rownames`` was admitted too. The fix drops a numeric predictor whose
absolute rank (Spearman) correlation with the response is at least 0.999, and
warns about a whole-number predictor with one distinct value per row.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.discover import MIN_RANK_PAIRS, _rank_correlation


def _centered_frame(seed=0, n=300):
    """A response with a real ``x`` effect plus a monotone transform of it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    logwage = 1.5 * np.sin(x) + rng.normal(0, 0.6, n)
    # wages is exp(const + logwage): a strictly increasing function of the
    # response, exactly the leak the issue reports.
    wages = np.exp(10.5 + logwage)
    return pl.DataFrame({"x": x, "wages": wages, "logwage": logwage})


def _leak_only_frame(seed=0, n=300):
    """A response whose only candidate predictor is its own transform.

    Without the fix, discovery selects ``bbs(log(wages))`` on this frame (the
    response is ``log(wages)`` exactly); ``noise`` gives the stage a second,
    useless candidate so the leak is a real choice, not the only column.
    """
    rng = np.random.default_rng(seed)
    logwage = rng.normal(size=n)
    wages = np.exp(10.5 + logwage)
    noise = rng.normal(size=n)
    return pl.DataFrame({"wages": wages, "noise": noise, "logwage": logwage})


def test_response_transform_is_dropped_and_alone_leaves_an_intercept():
    result = mb.discover(_leak_only_frame(), "logwage", seed=0,
                         spread=False, interactions=False)
    assert result.formula == "1"
    assert any("dropped predictor 'wages'" in w for w in result.warnings)
    hygiene = [r for r in result.log if r["stage"] == "hygiene"]
    assert any(r["candidate"] == "wages" and "rank correlation" in r["reason"]
               for r in hygiene)


def test_a_genuine_predictor_survives_the_hygiene_drop():
    result = mb.discover(_centered_frame(), "logwage", seed=0,
                         spread=False, interactions=False)
    assert "wages" not in result.formula
    assert any("dropped predictor 'wages'" in w for w in result.warnings)
    # The genuine predictor is still found and used.
    variables = {v for e in result.mean_effects for v in e["variables"]}
    assert "x" in variables
    assert "wages" not in variables


def test_exact_copy_of_the_response_is_dropped():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "y_copy": y.copy(), "y": y})
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False)
    assert "y_copy" not in result.formula
    assert any("dropped predictor 'y_copy'" in w for w in result.warnings)
    assert "x" in {v for e in result.mean_effects for v in e["variables"]}


def test_row_identifier_is_warned_but_kept():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "rownames": np.arange(n, dtype=float), "y": y})
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False)
    assert any("looks like a row identifier" in w and "rownames" in w
               for w in result.warnings)
    # An identifier may be an ordered quantity, so it is kept, not dropped.
    assert not any("dropped predictor 'rownames'" in w for w in result.warnings)
    hygiene = [r for r in result.log if r["stage"] == "hygiene"]
    assert any(r["candidate"] == "rownames" for r in hygiene)


def test_genuine_predictors_are_not_flagged():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-2, 2, n)
    z = rng.normal(size=n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": x, "z": z, "y": y})
    result = mb.discover(frame, "y", seed=0, spread=False, interactions=False)
    assert not any("dropped predictor" in w for w in result.warnings)
    assert not any("looks like a row identifier" in w for w in result.warnings)


def test_mandatory_response_copy_raises():
    with pytest.raises(ValueError, match="mandatory_terms"):
        mb.discover(_centered_frame(), "logwage", seed=0, spread=False,
                    interactions=False, mandatory_terms=["wages"])


def test_rank_correlation_reads_monotone_transforms():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    assert _rank_correlation(pl.Series(y), pl.Series(np.exp(y))) == pytest.approx(1.0)
    # The measure is absolute, so a decreasing transform is caught too.
    assert _rank_correlation(pl.Series(y), pl.Series(-y)) == pytest.approx(1.0)
    assert _rank_correlation(pl.Series(y), pl.Series(rng.normal(size=200))) < 0.5


def test_rank_correlation_needs_enough_pairs():
    y = np.arange(MIN_RANK_PAIRS - 1, dtype=float)
    assert _rank_correlation(pl.Series(y), pl.Series(np.exp(y))) is None
