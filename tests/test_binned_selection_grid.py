"""Binning changes which learner is selected, and by how much.

P3.3 gates its work on a grid: "Do not add Gaussian pairwise cross-tabs until
a low-`p` benchmark beats the current compact scatter path after accounting
for memory. Complete the `n/p/K`, wall-time, peak-RSS, and selection-agreement
grid first."

`benches/binned_base_learners.json` held one point of it — n = 10,000, 40
learners, 200 bins, Binomial — reporting `sequence_identical: true`. Read
alone, that says binning preserves the fit and the 6x speedup it records
(83.4 ms to 13.8 ms for bbs at 100 bins) is free.

Across 18 cells it is not. Selection diverges at 50 and 200 bins in every cell
tried, usually inside the first dozen iterations, and the fitted values differ
by more than one response unit. So the timing comparison those benchmarks make
is between a fit and a *different fit*, which is the mistake P3.5's fourth
paragraph is about, arrived at from the other direction.

The correctness dimension is cheap — every fit here is milliseconds — and it
is the dimension that decides whether the timing dimension means anything.
"""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID = ROOT / "benches" / "binned_selection_grid.json"
RUNNER = ROOT / "benches" / "bench_binned_selection_grid.py"


@pytest.fixture(scope="module")
def grid():
    return json.loads(GRID.read_text())


def test_the_committed_grid_matches_a_fresh_run():
    result = subprocess.run([sys.executable, str(RUNNER), "--check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_it_covers_the_three_axes_p3_3_names(grid):
    design = grid["design"]
    assert len(design["rows"]) >= 3
    assert len(design["learners"]) >= 2
    assert len(design["bins"]) >= 3
    assert len(grid["cells"]) == (len(design["rows"]) * len(design["learners"])
                                  * len(design["bins"]))


def test_binning_changes_the_selection_sequence(grid):
    """The finding. A single agreeing point read as though agreement were
    general; most of the grid disagrees."""
    identical = [cell for cell in grid["cells"] if cell["sequence_identical"]]
    assert len(identical) < len(grid["cells"]), (
        "if every cell now agrees, binning changed and the benchmark "
        "conclusions should be revisited")
    assert len(identical) == grid["summary"]["sequence_identical"]


def test_the_coarse_bin_counts_never_agree(grid):
    """50 and 200 bins are the counts `binned_base_learners.json` measured
    speedups at. Neither preserves the selection sequence anywhere here."""
    for cell in grid["cells"]:
        if cell["bins"] in (50, 200):
            assert not cell["sequence_identical"], cell


def test_divergence_starts_early(grid):
    """Not a late tie-break: the paths part inside the first dozen steps, so a
    truncated fit is affected as much as a long one."""
    divergences = [cell["first_divergence"] for cell in grid["cells"]
                   if cell["first_divergence"] is not None]
    assert divergences
    assert min(divergences) <= 7
    assert sorted(divergences)[len(divergences) // 2] <= 20


def test_the_error_grows_with_n_at_a_fixed_bin_count(grid):
    """The counter-intuitive one, and the reason a bin count cannot be a
    constant: at fixed `K`, more rows put more distinct values in each bin, so
    the approximation gets worse as the data grows."""
    by_rows = {}
    for cell in grid["cells"]:
        if cell["bins"] == 800 and cell["learners"] == 4:
            by_rows[cell["n"]] = cell["max_abs_fitted_difference"]
    ordered = [by_rows[n] for n in sorted(by_rows)]
    assert len(ordered) >= 3
    assert ordered == sorted(ordered), by_rows
    assert ordered[-1] > 2 * ordered[0]


def test_the_grid_carries_its_provenance(grid):
    """P3.5's contract, applied where the number is produced rather than
    after the fact."""
    metadata = grid["metadata"]
    assert metadata["commit"]
    assert "working_tree" in metadata
    assert grid["design"]["seed"] == 0
    assert "not measured here" in grid["design"]["note"]


def test_it_does_not_claim_a_timing(grid):
    """Wall time and peak RSS need a quiet machine and the paired protocol.
    Mixing an untimed sweep into a timing artifact is how the numbers in
    `docs/book` came to have no provenance."""
    cells = json.dumps(grid["cells"])
    for word in ("seconds", "median_ms", "wall", "rss"):
        assert word not in cells.lower(), word
    # The note may say what is *not* measured; a cell may not carry a timing.
    assert "not measured here" in grid["design"]["note"]
