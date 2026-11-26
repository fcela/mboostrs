"""One (task, seed, arm) run; prints one JSON line.

python -I run_one.py KIND NAME SEED ARM
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

for _k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import arms  # noqa: E402
import metrics  # noqa: E402
import tasks  # noqa: E402


def main(kind, name, seed, arm):
    seed = int(seed)
    record = {"kind": kind, "task": name, "seed": seed, "arm": arm}
    dev, test, truth = tasks.load(kind, name, seed)
    y_test = test["y"].to_numpy(float)
    prep = arms.prepare(dev, test, seed)
    record.update(response_scale=prep.scale, n_dev=len(dev), n_test=len(test), n_predictors=len(prep.numeric) + len(prep.categorical),
                  n_design_columns=int(prep.x_dev.shape[1]), positive=prep.positive)
    begin = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecast = arms.unscale(arms.ARMS[arm](prep), prep.scale)
        record["seconds"] = time.perf_counter() - begin
        record["scores"] = metrics.score(forecast, y_test, truth, prep.test)
        record["info"] = _jsonable(forecast.info)
        record["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}"
                                     for w in caught})[:20]
        record["status"] = "ok"
    except Exception as exc:
        record["seconds"] = time.perf_counter() - begin
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        record["traceback"] = traceback.format_exc()[-2000:]
    record["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        2**20 if sys.platform == "darwin" else 2**10)
    print(json.dumps(record))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    main(*sys.argv[1:5])
