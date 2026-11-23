"""D10 of PROTOCOL.md: rerun ten real datasets twice per arm with seed 5000."""
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent
REAL = HERE.parent / "fit_heldout" / "data"
PY = sys.argv[1]
ARMS = ("mboostrs", "catboost", "lightgbm", "xgboost", "ebm", "reference")
rows = [json.loads(line) for line in open(HERE / "results.jsonl")]
ok = {(str(r["id"]), r["seed"], r["arm"]) for r in rows if r["status"] == "ok"}
order = [str(t["id"]) for f in ("manifest.json", "manifest_uci.json", "manifest_mb.json")
         for t in json.load(open(REAL / f))["tasks"] if (REAL / f"{t['id']}.parquet").exists()]
chosen = [t for t in order if all((t, 5000, a) in ok for a in ARMS)][:10]
env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
def run(job):
    t, a, rep = job
    p = subprocess.run([PY, "-I", str(HERE / "run_one_d10.py"), t, "5000", a], capture_output=True, text=True, env=env, timeout=900)
    r = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1])
    return t, a, rep, r.get("prediction_sha256"), r["status"][:60]
jobs = [(t, a, k) for t in chosen for a in ARMS for k in (0, 1)]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(run, jobs))
out = {}
for t in chosen:
    for a in ARMS:
        h = [r[3] for r in res if r[0] == t and r[1] == a]
        out.setdefault(a, []).append(dict(task=t, identical=len(set(h)) == 1 and h[0] is not None))
summary = {a: f"{sum(x['identical'] for x in v)}/{len(v)}" for a, v in out.items()}
json.dump(dict(chosen=chosen, detail=out, summary=summary), open(HERE / "d10.json", "w"), indent=1)
print(chosen)
print(summary)
