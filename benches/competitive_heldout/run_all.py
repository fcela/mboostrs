"""Run every (task, seed, arm) job of PROTOCOL.md in its own process, 900 s each."""
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REAL = HERE.parent / "fit_heldout" / "data"
PY, WORKERS, OUT = sys.argv[1], int(sys.argv[2]), HERE / sys.argv[3]
ARMS = ("mboostrs", "catboost", "lightgbm", "xgboost", "ebm", "reference")
real = [str(t["id"]) for f in ("manifest.json", "manifest_uci.json", "manifest_mb.json")
        for t in json.load(open(REAL / f))["tasks"] if (REAL / f"{t['id']}.parquet").exists()]
synth = [t["id"] for t in json.load(open(HERE / "data/manifest_synth.json"))["tasks"]]
jobs = [(t, s, a) for t in real for s in (5000, 5001, 5002) for a in ARMS]
jobs += [(t, 5000, a) for t in synth for a in ARMS]
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((str(r["id"]), r["seed"], r["arm"]))
env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1")


def run(job):
    t, s, a = job
    try:
        p = subprocess.run([PY, "-I", str(HERE / "run_one.py"), t, str(s), a], capture_output=True,
                           text=True, timeout=900, env=env, cwd=str(HERE))
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if line:
            return line[-1]
        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
        return json.dumps(dict(id=t, seed=s, arm=a, status=f"error: crash: {err}"[:400], loss=None))
    except subprocess.TimeoutExpired:
        return json.dumps(dict(id=t, seed=s, arm=a, status="error: timeout 900 s", loss=None))


todo = [j for j in jobs if j not in done]
print(f"{len(todo)} jobs", flush=True)
with ThreadPoolExecutor(WORKERS) as pool, open(OUT, "a") as fh:
    for i, line in enumerate(pool.map(run, todo), 1):
        fh.write(line + "\n")
        fh.flush()
        if i % 50 == 0:
            print(f"{i}/{len(todo)}", flush=True)
print("done", flush=True)
