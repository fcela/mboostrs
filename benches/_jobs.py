"""Run benchmark jobs as separate processes, cached and resumable.

A job is one call `module.run_job(spec)` in a fresh interpreter. The parent
keeps `workers` of them running from a thread pool and writes each job's
result to `cache_dir/<key>.json` as it finishes, so

* a job that crashes the interpreter (a segfault, an abort in native code, the
  OS killing it for memory) is recorded with its exit code and the tail of its
  stderr, instead of leaving a process pool waiting forever;
* a job that runs longer than `timeout` seconds is killed and recorded;
* rerunning the script skips a job only when its result file exists *and*
  was measured under the same identity: the runner and the local modules it
  imports, this harness, the installed mboostrs (Python sources and compiled
  extension, for jobs that use it), package and R versions, thread settings
  and platform. Anything else is measured again. A row without an identity
  (written before identities were recorded) is never reused.

Rows are strict JSON: a non-finite number is stored as null and listed in
``_nonfinite``. Every row carries ``_status`` (``success``, ``fit_failure``
when `run_job` reported an error, ``crash`` or ``timeout``) and ``_identity`` (the
fingerprint, its components and the time the job ran), so an assembled
artifact can say when and under which build each number was measured.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_CHILD = """
import json, sys
sys.path.insert(0, {bench_dir!r})
module = __import__({module!r})
spec = json.loads(sys.argv[1])
result = module.run_job(spec)
with open(sys.argv[2], "w") as fh:
    json.dump(result, fh, default=float)
"""


def job_key(spec):
    return "__".join(str(v) for v in spec)


def _sha(parts):
    import hashlib

    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else str(part).encode())
    return digest.hexdigest()


def _local_modules(module, bench_dir):
    """The runner and every benches/ module it imports, transitively."""
    import re

    seen, stack = [], [module]
    while stack:
        name = stack.pop()
        path = Path(bench_dir) / f"{name}.py"
        if name in seen or not path.exists():
            continue
        seen.append(name)
        text = path.read_text()
        for match in re.finditer(r"^\s*(?:from|import)\s+(?:benches\.)?([A-Za-z_]\w*)", text, re.M):
            stack.append(match.group(1))
    return sorted(set(seen) | {"_jobs"})


def _mboostrs_identity():
    """Hash of the mboostrs Python sources and compiled extension that a
    child process started with this interpreter imports."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import mboostrs, mboostrs._rust as r; print(mboostrs.__file__); print(r.__file__)"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        return {"error": probe.stderr[-300:]}
    package_file, extension = probe.stdout.split()
    package = Path(package_file).parent
    sources = sorted(package.rglob("*.py"))
    return {"package": str(package), "extension": extension,
            "hash": _sha([p.read_bytes() for p in sources] + [Path(extension).read_bytes()])}


_PACKAGES = ("numpy", "polars", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost",
             "catboost", "interpret", "shap", "aplr", "pygam", "ngboost",
             "statsmodels", "econml", "doubleml")
_R_PACKAGES = ("mgcv", "gamlss", "glmnet", "knockoff", "Boruta", "grf", "scam",
               "mboost", "survival")


def _environment_identity(env):
    import importlib.metadata as metadata
    import platform

    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    r_script = ("cat(R.version.string, '\\n'); for (p in c(" +
                ",".join(repr(p) for p in _R_PACKAGES) + ")) "
                "if (requireNamespace(p, quietly=TRUE)) cat(p, as.character(packageVersion(p)), '\\n')")
    try:
        r = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True,
                           timeout=120).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        r = "unavailable"
    threads = {k: env.get(k) for k in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS",
                                       "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "r": r, "threads": threads}


def _strict(value, path, found):
    """`value` with every non-finite float replaced by None; the paths of the
    replaced entries are appended to `found`."""
    if isinstance(value, float) and value != value or value in (float("inf"), float("-inf")):
        found.append(path)
        return None
    if isinstance(value, dict):
        return {k: _strict(v, f"{path}.{k}" if path else str(k), found) for k, v in value.items()}
    if isinstance(value, list):
        return [_strict(v, f"{path}[{i}]", found) for i, v in enumerate(value)]
    return value


def _row_status(row):
    if "failure" in row:
        return "timeout" if "timeout_s" in row["failure"] else "crash"
    if "error" in row:
        return "fit_failure"
    return "success"


def run_jobs(module, specs, cache_dir, *, workers=8, timeout=3600, env=None,
             uses_mboostrs=None):
    """Run every spec without a cached result of the same identity; return
    the list of results.

    `uses_mboostrs(spec)` says whether a job imports mboostrs; its identity
    then includes the installed mboostrs build. It defaults to every job, so
    a competitor-only job is re-measured after an mboostrs change unless the
    runner says it does not use it.
    """
    import datetime

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = str(Path(__file__).resolve().parent)
    child = _CHILD.format(bench_dir=bench_dir, module=module)
    child_env = dict(os.environ, **(env or {}))
    uses_mboostrs = uses_mboostrs or (lambda spec: True)

    modules = _local_modules(module, bench_dir)
    runner_hash = _sha([(Path(bench_dir) / f"{name}.py").read_bytes() for name in modules])
    environment = _environment_identity(child_env)
    mboostrs = _mboostrs_identity()
    base = {"runner_modules": modules, "runner_hash": runner_hash,
            "environment": environment}

    def identity(spec):
        components = dict(base)
        if uses_mboostrs(spec):
            components["mboostrs"] = mboostrs
        return {"fingerprint": _sha([json.dumps(components, sort_keys=True),
                                     json.dumps(list(spec))]),
                "components": components}

    def cached(spec):
        path = cache_dir / f"{job_key(spec)}.json"
        if not path.exists():
            return False
        try:
            row = json.loads(path.read_text())
        except ValueError:
            return False
        stored = row.get("_identity", {}).get("fingerprint")
        return stored == identity(spec)["fingerprint"]

    todo = [s for s in specs if not cached(s)]
    started = time.time()
    print(f"{len(specs) - len(todo)} cached, {len(todo)} to run", flush=True)

    def one(spec):
        out = cache_dir / f"{job_key(spec)}.json"
        tmp = out.with_suffix(".part")
        begin = time.time()
        stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        try:
            res = subprocess.run([sys.executable, "-c", child, json.dumps(spec), str(tmp)],
                                 capture_output=True, text=True, timeout=timeout, env=child_env)
            if res.returncode == 0 and tmp.exists():
                row = json.loads(tmp.read_text())
                failure = None
            else:
                failure = {"exit_code": res.returncode, "stderr": res.stderr[-3000:]}
        except subprocess.TimeoutExpired:
            failure = {"timeout_s": timeout}
        if failure is not None:
            failure["seconds"] = time.time() - begin
            row = {"job": list(spec), "failure": failure}
        # Strict JSON: a NaN or infinity becomes null, and the row says where.
        found = []
        row = _strict(row, "", found)
        if found:
            row["_nonfinite"] = found
        row["_status"] = _row_status(row)
        row["_identity"] = {**identity(spec), "measured_at": stamp,
                            "wall_seconds": time.time() - begin}
        out.write_text(json.dumps(row, default=float, allow_nan=False))
        tmp.unlink(missing_ok=True)
        return spec, failure

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, s) for s in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            spec, failure = future.result()
            status = "FAILED " + json.dumps(failure)[:300] if failure else "ok"
            print(f"[{done}/{len(todo)}] {job_key(spec)} {status} "
                  f"({time.time() - started:.0f}s)", flush=True)

    results = []
    for s in specs:
        path = cache_dir / f"{job_key(s)}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    return results


def provenance(*, protocol, repeats, uncertainty, versions=None):
    """The fields `tools/check_benchmark_provenance.py` requires of a
    published timing: when, which commit, which machine and versions, how
    many threads, the protocol, the number of repetitions and what spread is
    reported."""
    import datetime
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_contract import machine_metadata

    meta = machine_metadata()
    merged_versions = dict(meta["versions"])
    merged_versions.update(versions or {})
    return {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "commit": {"sha": meta["commit"], **meta["working_tree"]},
        "platform": meta["platform"],
        "versions": merged_versions,
        "threads": {**meta["threads"], "per_fit": 1},
        "protocol": protocol,
        "repeats": repeats,
        "uncertainty": uncertainty,
    }
