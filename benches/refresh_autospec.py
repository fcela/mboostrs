"""Remeasure the autospec rows of the real-data study on the current build.

The real-data study (``benches/bench_realdata.py``) compares mboostrs's
autospec workflows with mgcv, EBM, gradient-boosted trees, neural networks and
penalized linear models on 24 OpenML-CTR23 data sets. Its comparison methods
take hours to refit and do not change when mboostrs changes, so a refresh
remeasures only the rows whose method contains ``autospec`` and keeps every
other row byte for byte, marked ``"retained original measurement"``.

    python benches/refresh_autospec.py --studies realdata --workers 8

The jobs run through ``bench_realdata.run_job`` with the same data sets,
splits and thread settings as the original study. The assembled file,
``docs/book/cases/bench_realdata_preexploration.json``, records the source
artifact's sha256 and its original provenance next to the refresh's own;
``tools/check_realdata_refresh.py`` validates the assembly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

SOURCE = ROOT / "docs/book/cases/bench_realdata.json"
OUT = ROOT / "docs/book/cases/bench_realdata_preexploration.json"


def refresh_realdata(workers: int) -> None:
    from _jobs import provenance, run_jobs
    import bench_realdata

    source_bytes = SOURCE.read_bytes()
    original = json.loads(source_bytes)
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = original["results"]
    specs = sorted({(int(r["data_id"]), r["method"], int(r["seed"]))
                    for r in rows if "autospec" in r["method"]})
    for data_id in sorted({spec[0] for spec in specs}):
        bench_realdata.load(data_id)  # download once, before the workers start
    env = dict(os.environ, OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    fresh = run_jobs("bench_realdata", specs, HERE / ".cache" / "realdata",
                     workers=workers, timeout=3600, env=env,
                     uses_mboostrs=lambda spec: "mboostrs" in str(spec[1]))
    by_key = {}
    for spec, r in zip(specs, fresh, strict=True):
        key = (int(spec[0]), spec[1], int(spec[2]))
        if "data_id" not in r:
            r = {**r, "data_id": key[0], "method": key[1], "seed": key[2],
                 "error": str(r.get("failure") or r.get("_status") or "failed")}
        by_key[key] = r
    results = []
    for row in rows:
        if "autospec" in row["method"]:
            new = dict(by_key[(int(row["data_id"]), row["method"], int(row["seed"]))])
            new.setdefault("dataset", row.get("dataset"))
            results.append({**new, "refresh_study": "realdata",
                            "refresh_status": "current autospec"})
        else:
            results.append({**row, "refresh_status": "retained original measurement"})
    protocol = dict(original["protocol"])
    protocol["preexploration"] = previous.get("protocol", {}).get("preexploration", "native")
    protocol["search"] = "fast"
    protocol["refresh"] = {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rows_remeasured": len(specs),
        "scope": "all rows whose method contains autospec",
        "retained_rows": "original values, timestamps and fingerprints unchanged",
        "timing": "mixed measurement dates; no fixed-workload speed ratio",
    }
    import importlib.metadata as md

    document = {
        "provenance": provenance(protocol=protocol,
                                 repeats={"splits": original["protocol"]["splits"]},
                                 uncertainty="mean over splits; per-data-set ranks",
                                 versions={"mboostrs": md.version("mboostrs")}),
        "protocol": protocol,
        "original_provenance": original["provenance"],
        "results": results,
    }
    OUT.write_text(json.dumps(document, default=float))
    print(f"wrote {OUT}: {len(specs)} autospec rows remeasured, "
          f"{len(results) - len(specs)} retained")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", default=["realdata"], choices=["realdata"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for study in args.studies:
        if study == "realdata":
            refresh_realdata(args.workers)


if __name__ == "__main__":
    main()
