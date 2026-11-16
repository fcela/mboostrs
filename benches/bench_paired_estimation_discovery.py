"""Run the paired Track A / Track B battery (issue #511).

Usage:
    python -m benches.bench_paired_estimation_discovery --run [--panels ...] [--replicates N]
    python -m benches.bench_paired_estimation_discovery --report
    python -m benches.bench_paired_estimation_discovery --floors

Results land in docs/benchmarks/paired_ab/.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.paired_battery import (  # noqa: E402
    INSTRUMENT_VERSION,
    N_REPLICATES,
    PANELS,
    _pilot_scale,
    partial_formula,
    run_replicate,
    summarize,
    supplied_basis_floor,
)

OUTDIR = ROOT / "docs" / "benchmarks" / "paired_ab"


def cmd_run(args):
    import os
    for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    panels = args.panels or list(PANELS)
    n_rep = args.replicates or N_REPLICATES
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scales = {p: _pilot_scale(p) for p in panels}
    (OUTDIR / "scales.json").write_text(json.dumps(
        {"instrument": INSTRUMENT_VERSION, "scales": scales,
         "note": "s = Var(y) from 50k pilot draws (50 x 1000 rows)"}, indent=1))
    records = []
    for panel in panels:
        for rep in range(n_rep):
            print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
                  f"{panel} replicate {rep}", flush=True)
            rec = run_replicate(panel, rep)
            records.append(rec)
            (OUTDIR / "results.json").write_text(json.dumps(records, indent=1))
    print(f"done: {len(records)} replicates -> {OUTDIR / 'results.json'}")


def cmd_floors(args):
    """Evaluator-only basis floors per panel (supplied + partial formulas)."""
    panels = args.panels or list(PANELS)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    floors = {}
    for panel in panels:
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
              f"floor {panel}", flush=True)
        floors[panel] = {
            "supplied": supplied_basis_floor(panel),
            "partial": supplied_basis_floor(panel,
                                            formula=partial_formula(panel)),
        }
    (OUTDIR / "floors.json").write_text(json.dumps(
        {"instrument": INSTRUMENT_VERSION, "floors": floors}, indent=1))
    print(f"done -> {OUTDIR / 'floors.json'}")


def report_text(records, scales):
    """Render the battery report as text (regression-tested; see tests)."""
    import numpy as np
    buf = io.StringIO()
    summary = summarize(records, scales)
    buf.write(f"paired-estimation-discovery {INSTRUMENT_VERSION}\n")
    buf.write(json.dumps(summary, indent=1))
    buf.write("\nstructural (B_discovery: recall / FDP medians):\n")
    for panel in sorted({r["panel"] for r in records}):
        recs = [r["arms"]["B_discovery"]["structural"] for r in records
                if r["panel"] == panel and "B_discovery" in r["arms"]
                and "structural" in r["arms"]["B_discovery"]]
        if recs:
            buf.write(f"  {panel}: recall={np.median([x['recall'] for x in recs]):.2f} "
                      f"fdp={np.median([x['fdp'] for x in recs]):.2f} (n={len(recs)})\n")
    buf.write("\nstopping boundaries (A arms: mean boundary rate / mean growth steps):\n")
    for panel in sorted(summary):
        parts = []
        for arm in ("A_supplied", "A_partial"):
            key = f"boundary_rate_{arm}"
            if key in summary[panel]:
                e = summary[panel][key]
                g = summary[panel].get(f"growth_steps_{arm}", {})
                parts.append(f"{arm}: rate={e['mean']:.2f} "
                             f"growth_steps~{g.get('mean', float('nan')):.1f}")
        if parts:
            buf.write(f"  {panel}: " + "; ".join(parts) + "\n")
    buf.write("\nprecision contract: each gap's paired_ci_halfwidth is the "
              "smallest absolute gap this design resolves at 8 replicates; "
              "a gap of interest narrower than that needs more replicates, "
              "not weaker claims.\n")
    n_fail = sum(len(r.get("failures", {})) for r in records)
    buf.write(f"\narm failures: {n_fail}\n")
    return buf.getvalue()


def cmd_report(args):
    records = json.loads((OUTDIR / "results.json").read_text())
    scales = json.loads((OUTDIR / "scales.json").read_text())["scales"]
    text = report_text(records, scales)
    print(text)
    (OUTDIR / "report.txt").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--floors", action="store_true")
    ap.add_argument("--panels", nargs="*")
    ap.add_argument("--replicates", type=int)
    args = ap.parse_args()
    if args.run:
        cmd_run(args)
    elif args.report:
        cmd_report(args)
    elif args.floors:
        cmd_floors(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
