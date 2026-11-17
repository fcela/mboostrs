"""Run one analyst's pipeline on every replicate, then time follow-ups, large n and reproducibility.

python evaluate.py --lib mboostrs --bench <claims_discovery dir> --hidden <hidden dir>
Results go to <bench>/eval/<lib>/.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

THREADS = "4"
TIMEOUT = 600
REPS = ["dev", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "N1", "N2", "S1", "S2"]
DROPS = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
         "median_income", "median_house_value", "ave_rooms", "ave_bedrooms", "ave_occup"]


def env():
    e = dict(os.environ)
    e.update(OMP_NUM_THREADS=THREADS, RAYON_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS,
             OPENBLAS_NUM_THREADS=THREADS, PYTHONHASHSEED="0")
    return e


def run_pipeline(run_dir: Path, train: Path, test: Path, out: Path, seed: int = 1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, "run.py", "--train", str(train), "--test", str(test), "--out", str(out), "--seed", str(seed)],
            cwd=run_dir, env=env(), capture_output=True, text=True, timeout=TIMEOUT,
        )
        rc, err = p.returncode, p.stderr[-3000:]
    except subprocess.TimeoutExpired:
        rc, err = "timeout", ""
    secs = time.perf_counter() - t0
    ok = rc == 0 and (out / "findings.json").exists() and (out / "predictions.parquet").exists()
    return {"rc": rc, "ok": ok, "seconds": secs, "stderr_tail": err}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def same_outputs(a: Path, b: Path) -> dict:
    out = {}
    try:
        fa, fb = json.loads((a / "findings.json").read_text()), json.loads((b / "findings.json").read_text())
        out["findings_equal"] = fa == fb
    except Exception as e:  # noqa: BLE001
        out["findings_equal"] = f"error: {e}"
    try:
        pa = pl.read_parquet(a / "predictions.parquet").sort("row_id")
        pb = pl.read_parquet(b / "predictions.parquet").sort("row_id")
        diff = max(float(np.max(np.abs(pa[c].to_numpy() - pb[c].to_numpy()))) for c in ("mean", "lower90", "upper90"))
        out["pred_max_abs_diff"] = diff
    except Exception as e:  # noqa: BLE001
        out["pred_max_abs_diff"] = f"error: {e}"
    return out


def load_refit(run_dir: Path):
    spec = importlib.util.spec_from_file_location("refit_mod", run_dir / "refit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(run_dir))
    spec.loader.exec_module(mod)
    return mod.fit_predict


def followups(run_dir: Path, dev: Path) -> dict:
    for k, v in env().items():
        os.environ[k] = v
    train, test = pl.read_parquet(dev / "train.parquet"), pl.read_parquet(dev / "test.parquet")
    res = {"times": [], "errors": []}
    try:
        fp = load_refit(run_dir)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"import: {e!r}")
        return res
    for col in DROPS:
        t0 = time.perf_counter()
        try:
            pred = np.asarray(fp(train, test, [col]))
            assert pred.shape[0] == test.height
            res["times"].append(time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"drop {col}: {e!r}"[:500])
    return res


def large_n(run_dir: Path, dev: Path, n: int = 500_000) -> dict:
    train, test = pl.read_parquet(dev / "train.parquet"), pl.read_parquet(dev / "test.parquet")
    rng = np.random.default_rng(7)
    big = train[rng.integers(0, train.height, n)]
    jit = ["longitude", "latitude", "median_income", "ave_rooms", "ave_occup"]
    big = big.with_columns(
        [(pl.col(c) * pl.Series(1 + rng.normal(0, 0.002, n))).alias(c) for c in jit]
        + [pl.Series("row_id", np.arange(n))]
    )
    try:
        fp = load_refit(run_dir)
        t0 = time.perf_counter()
        pred = np.asarray(fp(big, test, []))
        secs = time.perf_counter() - t0
        return {"seconds": secs, "ok": bool(pred.shape[0] == test.height), "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"seconds": None, "ok": False, "error": repr(e)[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--skip", default="", help="comma list of stages to skip: reps,followup,largen,repro")
    a = ap.parse_args()
    bench, hidden = Path(a.bench), Path(a.hidden)
    run_dir = bench / "runs" / a.lib
    ev = bench / "eval" / a.lib
    ev.mkdir(parents=True, exist_ok=True)
    skip = set(a.skip.split(","))
    if "largen_inner" in a.skip.split(":"):
        print(json.dumps(large_n(run_dir, bench / "data" / "dev")))
        return
    log = json.loads((ev / "runlog.json").read_text()) if (ev / "runlog.json").exists() else {}

    if "reps" not in skip:
        for rep in REPS:
            if log.get(rep, {}).get("ok") and (ev / rep / "findings.json").exists():
                continue  # resume: this replicate already ran
            rd = hidden / "reps" / rep
            log[rep] = run_pipeline(run_dir, rd / "train.parquet", rd / "test.parquet", ev / rep)
            print(a.lib, rep, {k: v for k, v in log[rep].items() if k != "stderr_tail"}, flush=True)
            (ev / "runlog.json").write_text(json.dumps(log, indent=2, default=str))
    if "repro" not in skip:
        dev = hidden / "reps" / "dev"
        log["repro_run2"] = run_pipeline(run_dir, dev / "train.parquet", dev / "test.parquet", ev / "dev_repeat")
        log["repro_grader_vs_grader"] = same_outputs(ev / "dev", ev / "dev_repeat")
        log["repro_grader_vs_analyst"] = same_outputs(ev / "dev", run_dir / "dev_out")
        (ev / "runlog.json").write_text(json.dumps(log, indent=2, default=str))
    if "followup" not in skip:
        log["followup"] = followups(run_dir, bench / "data" / "dev")
        print(a.lib, "followup", log["followup"], flush=True)
        (ev / "runlog.json").write_text(json.dumps(log, indent=2, default=str))
    if "largen" not in skip:
        try:
            p = subprocess.run([sys.executable, __file__, "--lib", a.lib, "--bench", a.bench,
                                "--hidden", a.hidden, "--skip", "largen_inner:"],
                               env=env(), capture_output=True, text=True, timeout=600)
            line = [x for x in p.stdout.splitlines() if x.startswith("{")]
            log["large_n"] = json.loads(line[-1]) if line else {"ok": False, "seconds": None,
                                                               "error": p.stderr[-500:]}
        except subprocess.TimeoutExpired:
            log["large_n"] = {"ok": False, "seconds": None, "error": "timeout 600 s"}
        print(a.lib, "large_n", log["large_n"], flush=True)
        (ev / "runlog.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
