#!/usr/bin/env python3
"""The `n`/`p`/`K` selection-agreement grid P3.3 gates its work on.

P3.3: "Do not add Gaussian pairwise cross-tabs until a low-`p` benchmark beats
the current compact scatter path after accounting for memory. Complete the
`n/p/K`, wall-time, peak-RSS, and selection-agreement grid first."

`benches/binned_base_learners.json` holds one point of that grid: n = 10,000,
40 learners, 200 bins, Binomial, reporting `sequence_identical: true`. Read
alone it says binning preserves the fit. Across a grid it does not: binning
changes which learner is selected, from the first handful of iterations, at
every bin count that benchmark measured speedups for.

This runs the correctness dimension, which is cheap -- every fit here is
milliseconds -- and is the dimension that decides whether the timing
dimension means anything. A binned fit that selects a different model is not
a faster way to compute the same answer.

Wall-time and peak-RSS are not measured here. They need a quiet machine and
the paired protocol in `benches/benchmark_contract.py`; mixing an untimed
correctness sweep into a timing artifact is how the timings in `docs/book`
came to have no provenance.

Run from the repository root:

    python benches/bench_binned_selection_grid.py
    python benches/bench_binned_selection_grid.py --check
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

OUTPUT = ROOT / "benches" / "binned_selection_grid.json"

ROWS = (1_000, 4_000, 16_000)
LEARNERS = (4, 12)
BINS = (50, 200, 800)
MSTOP = 60
NU = 0.1
SEED = 0


def one_cell(n: int, p: int, bins: int) -> dict:
    import mboostrs as mb

    rng = np.random.default_rng(SEED)
    data = {f"x{i}": rng.normal(size=n) for i in range(p)}
    data["y"] = (data["x0"] + np.sin(2 * data["x1"])
                 + rng.normal(scale=0.3, size=n))
    formula = "y ~ " + " + ".join(f"bbs(x{i}, df=4)" for i in range(p))
    control = mb.boost_control(MSTOP, NU)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = mb.gamboost(formula, data, control=control)
        binned = mb.gamboost(formula, data, control=control, bins=bins)

    exact_selection = list(exact.selected_which())
    binned_selection = list(binned.selected_which())
    divergence = next(
        (index for index, (left, right)
         in enumerate(zip(exact_selection, binned_selection, strict=True))
         if left != right),
        None,
    )
    return {
        "n": n,
        "learners": p,
        "bins": bins,
        "mstop": MSTOP,
        "family": "Gaussian",
        "sequence_identical": exact_selection == binned_selection,
        "first_divergence": divergence,
        "max_abs_fitted_difference": float(
            np.max(np.abs(exact.fitted_ - binned.fitted_))),
    }


def build() -> dict:
    from benches.benchmark_contract import machine_metadata

    cells = [one_cell(n, p, bins)
             for n, p, bins in itertools.product(ROWS, LEARNERS, BINS)]
    agree = [cell for cell in cells if cell["sequence_identical"]]
    return {
        "schema_version": 1,
        "kind": "binned-selection-grid",
        "metadata": machine_metadata(),
        "design": {
            "rows": list(ROWS),
            "learners": list(LEARNERS),
            "bins": list(BINS),
            "mstop": MSTOP,
            "nu": NU,
            "seed": SEED,
            "learner": "bbs(df=4)",
            "note": "correctness only; wall time and peak RSS are not "
                    "measured here and need a quiet machine",
        },
        "summary": {
            "cells": len(cells),
            "sequence_identical": len(agree),
            "identical_at_bins": sorted({cell["bins"] for cell in agree}),
            "max_fitted_difference": max(cell["max_abs_fitted_difference"]
                                         for cell in cells),
        },
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed grid's findings changed")
    arguments = parser.parse_args()

    grid = build()
    if arguments.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT.name} is missing", file=sys.stderr)
            return 1
        committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
        # The machine metadata differs by design; the findings must not.
        if committed["cells"] != grid["cells"]:
            print(f"{OUTPUT.name} no longer matches a fresh run. Binning's "
                  f"effect on selection changed; regenerate and read the "
                  f"diff before treating it as noise.", file=sys.stderr)
            return 1
        print(f"Binned selection grid: {grid['summary']['cells']} cells, "
              f"{grid['summary']['sequence_identical']} with an identical "
              f"selection sequence")
        return 0

    OUTPUT.write_text(json.dumps(grid, indent=1) + "\n", encoding="utf-8")
    summary = grid["summary"]
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {summary['cells']} cells, "
          f"{summary['sequence_identical']} identical, largest fitted "
          f"difference {summary['max_fitted_difference']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
