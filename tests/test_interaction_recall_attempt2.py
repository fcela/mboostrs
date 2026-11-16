"""Interaction recall and adoption in ``discover`` (attempt 2 of the claims benchmark).

Covers the changes made after the second claims-discovery rerun:

* an interaction is tested against the additive model with both of its continuous
  main effects at the tensor's margin smooth, ``bbs(x, df=2.5, knots=5)``, rather
  than against the draft's ``df=4`` mains;
* a candidate whose addition leaves the out-of-fold risk exactly unchanged (change
  and its standard error both 0, because the boosting never selects its term) is
  not adopted;
* a whole-number column with more distinct values than ``LABEL_MAX_LEVELS`` is not
  read as labels, so it is not drafted as a ``brandom`` term, whose fit cost grows
  with the square of the number of levels.
"""

import importlib
import time
import warnings

import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs.terms import LABEL_MAX_LEVELS, classify_columns, label_evidence

discover_module = importlib.import_module("mboostrs.discover")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _many_level_frame(n, levels, seed, signal):
    """A continuous predictor, a whole-number column with ``levels`` distinct values, y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    # Sparse identifiers, not 0..levels-1: a dense run of whole numbers reads as a count.
    ids = rng.choice(np.arange(10_000, 1_000_000), size=levels, replace=False)
    index = rng.integers(0, levels, n)
    effect = rng.normal(0.0, signal, levels) if signal else np.zeros(levels)
    y = np.sin(2 * x) + effect[index] + rng.normal(0.0, 0.5, n)
    return pl.DataFrame({"x": x, "code": ids[index].astype(float), "y": y})


def _age_income_frame(n, seed):
    """Two predictors with the claims benchmark's threshold interaction, and two noise columns.

    The planted term is ``0.9 * sigmoid((age - 35) / 2.5) * sigmoid((3 - log income) / 0.6)``,
    which is the shape of the benchmark's age x income effect on a log-drafted income.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(1.0, 52.0, n)
    income = np.exp(rng.normal(1.1, 0.45, n))
    noise_a = rng.normal(size=n)
    noise_b = rng.normal(size=n)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    f = 0.9 * sig((age - 35.0) / 2.5) * sig((3.0 - income) / 0.6)
    y = f + rng.normal(0.0, 0.2, n)
    return pl.DataFrame({"housing_median_age": age, "median_income": income,
                         "noise_a": noise_a, "noise_b": noise_b, "y": y})


def test_candidate_with_zero_change_and_zero_se_is_rejected(monkeypatch):
    """A pair the boosting never selects has change 0 and se 0; it must not be adopted.

    ``0 <= -bar * 0`` holds, so before the guard such a pair was adopted on every
    draw where the screen ranked it and the boosting never used it.
    """
    frame = _many_level_frame(400, 4, seed=1, signal=0.0).with_columns(
        pl.Series("z", np.random.default_rng(2).normal(size=400)))
    real = discover_module.compare_specifications

    def zero_for_pairs(specs, *args, **kwargs):
        rows = real(specs, *args, **kwargs)
        if any("%X%" in s for s in specs):
            for row in rows:
                row["change"] = 0.0
                row["change se"] = 0.0
        return rows

    monkeypatch.setattr(discover_module, "compare_specifications", zero_for_pairs)
    monkeypatch.setattr(discover_module, "_screen_pairs",
                        lambda *a, **k: [discover_module.ScreenedPair("x", "z", 1.0)])
    result = _quiet(mb.discover, frame, "y", seed=0)
    assert result.interactions == []
    rejected = [e for e in result.log if e["stage"] == "interactions" and e["candidate"]]
    assert rejected and all(not e["adopted"] for e in rejected)


def test_interaction_is_tested_against_both_mains_at_the_tensor_margin(monkeypatch):
    """The reference carries both continuous mains at ``df=2.5, knots=5``, not the draft df."""
    frame = _age_income_frame(600, seed=7)
    seen = []
    real = discover_module.compare_specifications

    def record(specs, *args, **kwargs):
        if any("%X%" in s for s in specs):
            seen.append(list(specs))
        return real(specs, *args, **kwargs)

    monkeypatch.setattr(discover_module, "compare_specifications", record)
    _quiet(mb.discover, frame, "y", seed=0)
    assert seen, "the interaction stage did not compare any pair"
    # The interaction stage compares the reference, the tensor alternative and
    # the actual incumbent it would replace (#679); the tensor must be the one
    # carrying the %X% term.
    reference, alternative, incumbent = seen[0]
    assert "%X%" not in incumbent
    assert "bbs(housing_median_age, df=2.5, knots=5)" in reference
    assert "bbs(log(median_income), df=2.5, knots=5)" in reference
    assert "df=4" not in " + ".join(t for t in reference.split(" + ")
                                   if "housing_median_age" in t or "median_income" in t)
    assert alternative.startswith(reference)
    assert "%X%" in alternative[len(reference):]


def test_planted_age_income_interaction_is_adopted_on_a_small_draw():
    """The planted pair is adopted, and the noise columns carry no interaction."""
    frame = _age_income_frame(1500, seed=11)
    result = _quiet(mb.discover, frame, "y", seed=0)
    pairs = [frozenset(i["variables"]) for i in result.interactions]
    assert frozenset(("housing_median_age", "median_income")) in pairs
    assert all("noise_a" not in p and "noise_b" not in p for p in pairs)


def test_many_level_whole_number_column_is_not_read_as_labels():
    """Above LABEL_MAX_LEVELS distinct values, the label test does not apply."""
    frame = _many_level_frame(6000, LABEL_MAX_LEVELS + 300, seed=3, signal=0.8)
    assert label_evidence("code", frame["code"].to_numpy(), frame["y"].to_numpy()) is None
    kinds = {k.name: k for k in classify_columns(frame, response="y", df=4.0)}
    assert not kinds["code"].term.startswith("brandom(")


def test_label_test_still_reads_a_column_within_the_cap_as_labels():
    """The cap removes only the columns above it."""
    frame = _many_level_frame(6000, 120, seed=4, signal=0.8)
    evidence = label_evidence("code", frame["code"].to_numpy(), frame["y"].to_numpy())
    assert evidence is not None and evidence.labels
    kinds = {k.name: k for k in classify_columns(frame, response="y", df=4.0)}
    assert kinds["code"].term.startswith("brandom(")


def test_discover_on_many_level_column_finishes_within_time_bound():
    """Time bound on a small reproducer of the slow draft.

    With the label rule uncapped, this reproducer (2,500 rows, 600 sparse identifiers
    with real level effects) drafted ``brandom(code)``, and discover took 47 s on the
    development machine, against 4.6 s with the cap. The bound is 30 s.
    """
    frame = _many_level_frame(2500, 600, seed=5, signal=0.8)
    kinds = {k.name: k for k in classify_columns(frame, response="y", df=4.0)}
    assert not kinds["code"].term.startswith("brandom(")
    start = time.perf_counter()
    result = _quiet(mb.discover, frame, "y", seed=0)
    elapsed = time.perf_counter() - start
    assert "brandom(code" not in result.formula
    assert elapsed < 30.0, f"discover took {elapsed:.1f} s; the bound is 30 s"
