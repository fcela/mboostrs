"""Run every (dataset, seed, arm) job in its own process with a 600 s limit."""
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.argv[1]
WORKERS = int(sys.argv[2])
ids = [t["id"] for f in ("manifest.json", "manifest_uci.json", "manifest_mb.json")
       for t in json.load(open(HERE / "data" / f))["tasks"]
       if (HERE / f"data/{t['id']}.parquet").exists()]
SEEDS = tuple(int(s) for s in sys.argv[3].split(","))
OUT = HERE / sys.argv[4]
jobs = [(d, s, a) for d in ids for s in SEEDS for a in ("old", "new", "quick")]
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((r["id"], r["seed"], r["arm"]))
env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def run(job):
    d, s, a = job
    try:
        p = subprocess.run([PY, "-I", str(HERE / "run_one.py"), str(d), str(s), a],
                           capture_output=True, text=True, timeout=600, env=env,
                           cwd=str(HERE))
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if line:
            return line[-1]
        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
        return json.dumps(dict(id=d, seed=s, arm=a, status=f"error: crash: {err}"[:400],
                               loss=None, seconds=None))
    except subprocess.TimeoutExpired:
        return json.dumps(dict(id=d, seed=s, arm=a, status="error: timeout 600 s",
                               loss=None, seconds=600))


todo = [j for j in jobs if j not in done]
print(f"{len(todo)} jobs", flush=True)
with ThreadPoolExecutor(WORKERS) as pool, open(OUT, "a") as fh:
    for i, line in enumerate(pool.map(run, todo), 1):
        fh.write(line + "\n")
        fh.flush()
        if i % 25 == 0:
            print(f"{i}/{len(todo)}", flush=True)
print("done", flush=True)
