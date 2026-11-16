"""What a published timing has to say about itself.

P3.5: "Use the paired alternating protocol from
`done/performance/stabsel-perf-baseline-2026-09-04.md`. Refresh benchmark
artifacts after current changes; older July artifacts cannot support current
headlines. Pin thread count, CPU target, versions, dirty-tree state, input and
output digests, warmup, repetitions, uncertainty, and memory scope."

Two gaps, of different kinds.

The protocol that produced the only stable numbers in this repository lived in
a document and in two scripts that are not in `benches/`. The session before it
measured a 6% speedup where it expected several-fold and could not tell signal
from noise, because repeated runs of the same benchmark varied by 3x; running
both arms alternating and reporting the median of the *paired* ratios brought
that to 1-5%. Any runner wanting it had to reimplement it.

And the contract that pins the provenance exists —
`docs/schemas/benchmark-result.schema.json` requires commit, platform,
versions, threads, warmup and digests — while the eight artifacts the book
actually renders carry none of it. One records in prose that two numbers were
re-measured because "their first measurement overlapped another job", which is
the drift the paired protocol cancels, caught by hand.
"""

import json
import pathlib
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.benchmark_contract import (  # noqa: E402
    BenchmarkContractError,
    paired_ratios,
)

CHECK = ROOT / "tools" / "check_benchmark_provenance.py"


# --- the paired alternating protocol ----------------------------------

def _work(seconds, value=(1.0, 2.0, 3.0)):
    def run():
        time.sleep(seconds)
        return list(value)
    return run


def test_it_reports_the_ratio_of_the_paired_arms():
    result = paired_ratios({"slow": _work(0.004), "fast": _work(0.002)},
                           rounds=7)
    assert result["protocol"] == "paired-alternating"
    assert result["arms"] == ["slow", "fast"]
    assert 0.35 < result["ratio"]["median"] < 0.75, result["ratio"]


def test_the_spread_is_reported_beside_the_median():
    """The number that told the earlier session its measurement was unusable
    was the spread, not the median. Reporting only a point estimate is how a
    3x-varying benchmark reads as a 6% speedup."""
    result = paired_ratios({"a": _work(0.003), "b": _work(0.003)}, rounds=5)
    assert "spread" in result["ratio"]
    assert result["ratio"]["spread"] >= 1.0
    assert len(result["paired_ratios"]) == 5


def test_arms_that_compute_different_things_are_refused():
    """A speed ratio between arms that computed different results measures
    nothing, and is the mistake P3.5's fourth paragraph is about."""
    with pytest.raises(BenchmarkContractError, match="computed different"):
        paired_ratios({"a": _work(0.001, (1.0,)),
                       "b": _work(0.001, (2.0,))}, rounds=4)


def test_an_arm_that_is_not_deterministic_is_refused():
    counter = {"n": 0}

    def drifting():
        counter["n"] += 1
        return [float(counter["n"])]

    with pytest.raises(BenchmarkContractError, match="same output on every"):
        paired_ratios({"a": drifting, "b": _work(0.001, (1.0,))}, rounds=4)


def test_too_few_rounds_are_refused():
    with pytest.raises(BenchmarkContractError, match="at least three rounds"):
        paired_ratios({"a": _work(0.001), "b": _work(0.001)}, rounds=2)


def test_it_needs_exactly_two_arms():
    with pytest.raises(BenchmarkContractError, match="exactly two arms"):
        paired_ratios({"a": _work(0.001)}, rounds=4)


def test_the_timings_are_kept_not_only_the_ratio():
    """A ratio alone cannot say whether both arms got slower."""
    result = paired_ratios({"a": _work(0.002), "b": _work(0.002)}, rounds=4)
    assert set(result["seconds"]) == {"a", "b"}
    assert all(len(values) == 4 for values in result["seconds"].values())


# --- the provenance ratchet -------------------------------------------

def test_the_provenance_check_passes():
    result = subprocess.run([sys.executable, str(CHECK)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_the_gap_is_recorded_rather_than_invented():
    """Re-running these is a machine-dependent job for one quiet box. The
    check records what is missing and fails when it grows; it does not fill
    the fields in."""
    baseline = json.loads(
        (ROOT / "docs" / "book" / "benchmark_provenance_baseline.json")
        .read_text())
    assert baseline["artifacts"] == len(baseline["missing"])
    assert baseline["missing_fields"] == sum(len(m) for m in baseline["missing"].values())
    # The seven artifacts written before the job harness recorded provenance
    # still say nothing about when or where they were measured.
    undated = [name for name, missing in baseline["missing"].items()
               if "created_at" in missing]
    assert len(undated) == 7
    for name in undated:
        assert "versions" in baseline["missing"][name], name


def test_the_dates_come_from_git():
    """The artifacts carry no measurement date, and inventing one would be
    worse than having none. The commit date is recoverable and is an upper
    bound on when the numbers were measured."""
    baseline = json.loads(
        (ROOT / "docs" / "book" / "benchmark_provenance_baseline.json")
        .read_text())
    for name, record in baseline["last_change"].items():
        if not record:
            # Not committed yet: git has no date for it, and none is invented.
            continue
        assert record["last_committed"].startswith("202"), name
        assert len(record["last_commit"]) == 12, name


def test_most_artifacts_do_pin_the_comparison_terms():
    """The half that is right: P3.5's fourth paragraph asks that a library
    comparison hold sample cap, tuning budget, metric, split and search fixed,
    and `bench_libraries.json` does."""
    protocol = json.loads(
        (ROOT / "docs" / "book" / "cases" / "bench_libraries.json")
        .read_text())["protocol"]
    for term in ("max_rows", "time_budget_s", "n_configs", "inner_folds",
                 "seed"):
        assert term in protocol, term


def test_a_new_artifact_without_provenance_fails(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    from tools import check_benchmark_provenance as provenance

    monkeypatch.setattr(provenance, "ARTIFACTS", tmp_path)
    (tmp_path / "bench_invented.json").write_text('{"results": []}')
    monkeypatch.setattr(provenance, "load_baseline", dict)
    monkeypatch.setattr(sys, "argv", ["check_benchmark_provenance.py"])
    assert provenance.main() == 1
