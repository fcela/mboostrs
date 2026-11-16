"""Tests for mboostrs.insights (issues #292, #305, #307).

These tests assert rankings, effect directions, and behavior -- never exact
screening scores, so they pass both before and after the #298 Rust
reimplementation of screen_interactions (different RNG stream, same order on
real structure).

Every pipeline test runs on both a pandas and a Polars frame (#305): Polars
is the package's core frame type and insights() must read either.
"""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mboostrs.insights import insights

BOTH_FRAMES = pytest.mark.parametrize("frame", ["pandas", "polars"])


def _as(frame, df):
    """The same data in the frame type under test."""
    return pl.from_pandas(df) if frame == "polars" else df


def _known_structure(seed=0, n=600):
    """y = 2*x0 + sin(1.5*x1) + 0.8*x2*x3 + noise; x4..x7 are noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 8))
    y = (2.0 * X[:, 0] + np.sin(1.5 * X[:, 1]) + 0.8 * X[:, 2] * X[:, 3]
         + rng.normal(0, 0.5, n))
    cols = [f"x{i}" for i in range(8)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    return df


def _noise_only(seed=1, n=500):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    df["y"] = rng.normal(size=n)
    return df


@BOTH_FRAMES
def test_finds_true_mains_with_correct_directions(frame):
    df = _as(frame, _known_structure())
    findings = insights(df, "y", seed=0)
    assert findings, "expected findings on structured data"
    cols_found = set()
    for f in findings:
        cols_found.update(f.evidence.get("columns", []))
    assert "x0" in cols_found and "x1" in cols_found
    # x0's effect is increasing: direction must say so where measured
    x0 = [f for f in findings if f.evidence.get("columns") == ["x0"]]
    assert x0 and "increasing" in x0[0].finding


@BOTH_FRAMES
def test_finds_true_pair(frame):
    df = _as(frame, _known_structure())
    findings = insights(df, "y", seed=0)
    pairs = [f for f in findings
             if set(f.evidence.get("columns", [])) == {"x2", "x3"}]
    assert pairs, "expected the x2*x3 interaction among the findings"


@BOTH_FRAMES
def test_every_finding_has_evidence_and_next_action(frame):
    df = _as(frame, _known_structure())
    for f in insights(df, "y", seed=0):
        assert f.evidence, f"empty evidence: {f.finding}"
        assert f.next_action.strip(), f"empty next action: {f.finding}"
        assert f.basis in ("measured", "detected")


@BOTH_FRAMES
def test_no_duplicates_across_stages(frame):
    df = _as(frame, _known_structure())
    findings = insights(df, "y", seed=0)
    keys = []
    for f in findings:
        cols = f.evidence.get("columns")
        keys.append(tuple(sorted(cols)) if cols else f.finding)
    assert len(keys) == len(set(keys)), f"duplicate findings: {keys}"


@BOTH_FRAMES
def test_to_dict_is_json_safe(frame):
    df = _as(frame, _known_structure())
    for f in insights(df, "y", seed=0):
        d = f.to_dict()
        json.dumps(d)  # raises if not JSON-safe


@BOTH_FRAMES
def test_noise_only_returns_little(frame):
    df = _as(frame, _noise_only())
    findings = insights(df, "y", seed=1)
    # The held-out screen should keep junk out; allow a couple of weak ones.
    assert len(findings) <= 2, f"too many findings on noise: {len(findings)}"


@BOTH_FRAMES
def test_measured_before_detected(frame):
    df = _as(frame, _known_structure())
    findings = insights(df, "y", seed=0)
    bases = [f.basis for f in findings]
    # All measured findings come before any detected one.
    seen_detected = False
    for b in bases:
        if b == "detected":
            seen_detected = True
        if seen_detected:
            assert b == "detected"


@BOTH_FRAMES
def test_family_recorded_in_evidence(frame):
    df = _as(frame, _known_structure())
    findings = insights(df, "y", seed=0)
    assert findings
    for f in findings:
        assert f.evidence.get("family") == "Gaussian", f.finding


def _bernoulli(seed=3, n=400):
    """y ~ Bernoulli(sigmoid(1.5*x0)); x1..x3 are noise (#307)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    p = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0]))
    y = (rng.uniform(size=n) < p).astype(float)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df["y"] = y
    return df


@BOTH_FRAMES
def test_bernoulli_response_has_no_spread_finding(frame):
    df = _as(frame, _bernoulli())
    findings = insights(df, "y", seed=0)
    assert findings, "expected the x0 effect among the findings"
    assert not [f for f in findings if "spread" in f.finding], [
        f.finding for f in findings]
    for f in findings:
        assert f.evidence.get("family") == "Binomial", f.finding


def _overdispersed_counts(seed=5, n=400):
    """Poisson counts whose rate carries a lognormal multiplier, so the
    variance exceeds the mean (#307)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    rate = np.exp(0.5 * X[:, 0]) * rng.lognormal(0.0, 1.0, size=n)
    y = rng.poisson(rate).astype(float)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df["y"] = y
    return df


@BOTH_FRAMES
def test_overdispersed_counts_flagged(frame):
    df = _as(frame, _overdispersed_counts())
    findings = insights(df, "y", seed=0)
    assert findings
    for f in findings:
        assert f.evidence.get("family") == "Poisson", f.finding
        assert "spread" not in f.finding
    over = [f for f in findings if "overdispersed" in f.finding]
    assert over, "expected the overdispersion finding on mixed counts"
    assert "NegBinomial" in over[0].next_action
    assert over[0].effect_size > 1.5
