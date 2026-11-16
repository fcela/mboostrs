"""What a dense interaction design costs — P3.4 item 7's own gate.

P3.4 item 7: "replace dense Buser interaction materialization with a
structured tensor recipe **when memory measurements show the need**."

The measurement shows the need and its resolution. A `bbs(x) %X% brandom(g)`
interaction with 24 spline columns over 20 groups materialises 480 columns,
and at 20,000 rows the unfactored fit retained 232.6 MB. The item is now
implemented: the fit step keeps the two factors apart, so the retained design
drops to 4.3 MB in the worst measured case while the 480 materialised columns
(which decide the fit) are still reported as `total_basis_columns`. The
40-column version remains the arithmetic label only.

This file records the number the decision rests on and fails if it moves, so
the case for the work and its effect do not have to be re-derived.
"""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEASUREMENT = ROOT / "benches" / "interaction_memory.json"
RUNNER = ROOT / "benches" / "bench_interaction_memory.py"


@pytest.fixture(scope="module")
def measured():
    return json.loads(MEASUREMENT.read_text())


def test_the_committed_measurement_matches_a_fresh_run():
    result = subprocess.run([sys.executable, str(RUNNER), "--check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_the_dense_design_is_the_product_of_the_two_factors(measured):
    """480 columns for 24 splines over 20 groups: every pair is materialised."""
    case = next(c for c in measured["cases"]
                if c["rows"] == 8_000 and c["levels"] == 20)
    assert case["dense_columns"] == case["spline_columns"] * case["levels"]
    assert case["structured_columns_if_factored"] == (
        case["spline_columns"] + case["levels"])


def test_the_measurement_shows_the_dense_baseline(measured):
    """The design still materialises 480 columns for 24 splines over 20 groups:
    the static case for the structured recipe is unchanged."""
    assert measured["summary"]["factor_if_structured"] > 5
    assert measured["summary"]["largest_dense_columns"] == 480


def test_the_structured_recipe_resolves_the_retention_need(measured):
    """Item 7's resolution: the fit no longer retains the 232.6 MB baseline.
    The worst measured case stays an order of magnitude below it."""
    assert measured["summary"]["largest_retained_bytes"] < 10e6


def test_the_cost_grows_with_the_rows(measured):
    by_rows = {c["rows"]: c["retained_bytes"] for c in measured["cases"]
               if c["levels"] == 20}
    ordered = [by_rows[n] for n in sorted(by_rows)]
    assert ordered == sorted(ordered)
    assert ordered[-1] > 4 * ordered[0]


def test_the_factored_column_count_is_labelled_as_arithmetic(measured):
    """It is not a measurement — no structured path exists to measure — and
    the field name says so rather than sitting beside the measured ones as
    though it had been observed."""
    for case in measured["cases"]:
        assert "if_factored" in " ".join(
            key for key in case if key.startswith("structured"))


def test_it_claims_no_timing(measured):
    """Memory only: mixing an unmeasured timing into a memory artifact is how
    the numbers in `docs/book` came to have no provenance."""
    assert "no timing is measured here" in measured["design"]["note"]
    body = json.dumps(measured["cases"])
    for word in ("seconds", "_ms", "elapsed"):
        assert word not in body, word


def test_it_carries_its_provenance(measured):
    assert measured["metadata"]["commit"]
    assert "working_tree" in measured["metadata"]
