"""Run every (task, seed, arm) job in its own process, in parallel.

python -I run_all.py PYTHON WORKERS OUT.jsonl [ARM,ARM,...]

Jobs already present in OUT are skipped, so an interrupted run resumes.
A job over TIMEOUT seconds is killed and recorded as a timeout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arms  # noqa: E402
import tasks  # noqa: E402

TIMEOUT = 3600


def run(python, job):
    kind, name, seed, arm = job
    begin = time.perf_counter()
    try:
        proc = subprocess.run([python, "-I", str(HERE / "run_one.py"), kind, name, str(seed), arm],
                              capture_output=True, text=True, timeout=TIMEOUT, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"kind": kind, "task": name, "seed": seed, "arm": arm, "status": "timeout",
                "seconds": time.perf_counter() - begin}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"kind": kind, "task": name, "seed": seed, "arm": arm, "status": "crash",
                "returncode": proc.returncode, "stderr": proc.stderr[-2000:],
                "seconds": time.perf_counter() - begin}
    return json.loads(lines[-1])


def main(python, workers, out, arm_filter=None):
    out = Path(out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["kind"], r["task"], int(r["seed"]), r["arm"]))
    selected = arm_filter.split(",") if arm_filter else list(arms.ARMS)
    jobs = [(k, n, s, a) for (k, n, s) in tasks.all_tasks() for a in selected
            if (k, n, s, a) not in done]
    print(f"{len(jobs)} jobs to run, {len(done)} already recorded", flush=True)
    with ThreadPoolExecutor(int(workers)) as pool, out.open("a") as sink:
        futures = [pool.submit(run, python, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            crps = r.get("scores", {}).get("crps")
            print(f"[{i}/{len(jobs)}] {r['task']} s{r['seed']} {r['arm']}: {r['status']} "
                  f"{r.get('seconds', 0):.1f}s crps={crps}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
