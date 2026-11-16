#!/usr/bin/env python3
"""What a dense interaction design costs, which is P3.4 item 7's gate.

P3.4 item 7: "replace dense Buser interaction materialization with a
structured tensor recipe **when memory measurements show the need**."

The measurement had not been made, so the item could neither be done nor
dropped. This makes it.

A `%X%` interaction materialises the full tensor: `p1 * p2` columns, one per
pair of basis functions. The structured form would keep the two factors apart
— `p1 + p2` columns — and take their product only where the arithmetic needs
it, the way the Multinomial Kronecker path already does.

Reported per configuration: the dense column count, the design's own bytes,
and what the fit retains. The structured column count is arithmetic, not a
measurement, and is labelled as such.

Run from the repository root:

    python benches/bench_interaction_memory.py
    python benches/bench_interaction_memory.py --check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

OUTPUT = ROOT / "benches" / "interaction_memory.json"

#: `(rows, spline columns, group levels)`. The spline count is `bbs`'s default
#: basis size; the group counts span a factor an interaction is plausibly
#: written against.
CASES = ((2_000, 24, 5), (2_000, 24, 20), (8_000, 24, 20), (20_000, 24, 20))


def one_case(n: int, spline_columns: int, levels: int) -> dict:
    import mboostrs as mb

    rng = np.random.default_rng(0)
    data = {"x": rng.normal(size=n), "g": rng.integers(0, levels, n).astype(float)}
    data["y"] = data["x"] + rng.normal(scale=0.3, size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboost("y ~ bbs(x, df=4) %X% brandom(g, df=4)", data,
                          control=mb.boost_control(10, 0.1))
    report = fit.execution_report()
    dense_columns = int(report.total_basis_columns)
    structured_columns = spline_columns + levels
    return {
        "rows": n,
        "spline_columns": spline_columns,
        "levels": levels,
        "dense_columns": dense_columns,
        "dense_design_bytes": n * dense_columns * 8,
        "retained_bytes": int(report.retained_learner_bytes),
        # Arithmetic, not a measurement: what the two factors would occupy if
        # they were kept apart.
        "structured_columns_if_factored": structured_columns,
        "structured_design_bytes_if_factored": n * structured_columns * 8,
    }


def build() -> dict:
    from benches.benchmark_contract import machine_metadata

    cases = [one_case(*case) for case in CASES]
    worst = max(cases, key=lambda case: case["retained_bytes"])
    return {
        "schema_version": 1,
        "kind": "interaction-memory",
        "metadata": machine_metadata(),
        "design": {
            "formula": "y ~ bbs(x, df=4) %X% brandom(g, df=4)",
            "mstop": 10,
            "note": "memory only; no timing is measured here",
        },
        "summary": {
            "cases": len(cases),
            "largest_retained_bytes": worst["retained_bytes"],
            "largest_dense_columns": worst["dense_columns"],
            "factor_if_structured": round(
                worst["dense_columns"] / worst["structured_columns_if_factored"], 1
            ),
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed measurement has moved")
    arguments = parser.parse_args()

    measured = build()
    if arguments.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT.name} is missing", file=sys.stderr)
            return 1
        committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if committed["cases"] != measured["cases"]:
            print(f"{OUTPUT.name} no longer matches a fresh run: the "
                  f"interaction design's size changed. Read the diff before "
                  f"regenerating.", file=sys.stderr)
            return 1
        print(f"Interaction memory: {measured['summary']['cases']} cases, "
              f"largest retained "
              f"{measured['summary']['largest_retained_bytes'] / 1e6:.1f} MB")
        return 0

    OUTPUT.write_text(json.dumps(measured, indent=1) + "\n", encoding="utf-8")
    summary = measured["summary"]
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {summary['cases']} cases, "
          f"largest retained {summary['largest_retained_bytes'] / 1e6:.1f} MB, "
          f"{summary['factor_if_structured']}x more columns than a factored "
          f"design would need")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
