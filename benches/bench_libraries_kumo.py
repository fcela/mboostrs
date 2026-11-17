"""Kumo Tabular fixed baseline for the library-comparison case study.

Install structured-data-models==0.2.1, pandas and scikit-learn, then run:
    python benches/bench_libraries_kumo.py --cache-dir /tmp/kumo-library-jobs

Uses the original runner's datasets, outer splits and training-only encoding.
There is no hyperparameter search: small pretrained model, eight estimators.
CPU execution is intentional; GPU timings must be published as a separate run.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from bench_libraries import REPEATS, SEED, MAX_ROWS, _encoder, datasets, take

LIBRARY = "NVIDIA Kumo Tabular (small)"
MODEL_REVISION = "v1.0.1"
NUM_ESTIMATORS = 8


def run_job(spec):
    import pandas as pd
    import sdm
    import torch
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    name, repeat = spec
    task, X, y = datasets()[name]
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=SEED + repeat,
        stratify=y if task == "classification" else None)
    Xtr, Xte = take(X, train), take(X, test)
    ytr, yte = y[train], y[test]

    # Weight loading (including the first download) is reported separately.
    load0 = time.perf_counter()
    model = sdm.models.KumoTabular(task=task, size="small", device="cpu")
    load_seconds = time.perf_counter() - load0
    start = time.perf_counter()
    if hasattr(X, "columns"):
        encoder = _encoder(X)
        Xtr = encoder.fit_transform(Xtr)
        Xte = encoder.transform(Xte)
    columns = [f"f{i}" for i in range(Xtr.shape[1])]

    def features(values):
        return sdm.TableTensor.from_pandas(
            pd.DataFrame(values, columns=columns),
            stypes={c: "numerical" for c in columns}, device="cpu")

    context, query = features(Xtr), features(Xte)
    target = sdm.TableTensor.from_pandas(
        pd.DataFrame({"target": ytr}),
        stypes={"target": "categorical" if task == "classification" else "numerical"},
        device="cpu")
    with torch.inference_mode():
        prediction = model(
            x_context=context, y_context=target, x_query=query,
            num_estimators=NUM_ESTIMATORS, estimator_batch_size=1,
            generator=torch.Generator(device="cpu").manual_seed(SEED + repeat))
    if task == "classification":
        # SDM may reorder classes: select the positive label by name.
        raw = prediction.to_pandas()["1"].to_numpy(dtype=float)
        held = held_raw = roc_auc_score(yte, raw)
    else:
        # The mean of equally spaced predictive quantiles approximates the
        # predictive expectation, the point forecast appropriate for R².
        raw = prediction.numerical.mean(dim=-1).cpu().numpy().ravel()
        held_raw = r2_score(yte, raw)
        held = r2_score(yte, np.clip(raw, ytr.min(), ytr.max()))
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    for array in (train, test, Xtr, Xte, ytr, yte):
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "dataset": name, "repeat": repeat, "task": task, "n": len(y),
        "p": Xtr.shape[1], "library": LIBRARY, "family": "pretrained interactions",
        "mode": "pretrained", "score": float(held), "score_unclipped": float(held_raw),
        "load_seconds": load_seconds, "total_seconds": elapsed,
        "input_digest": digest.hexdigest(),
    }


def main():
    from _jobs import run_jobs

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/book/cases/bench_libraries_kumo.json")
    ap.add_argument("--cache-dir", default="/tmp/kumo-library-jobs")
    ap.add_argument("--datasets", default="diabetes,cpu_act,california,breast_cancer,credit_g,electricity")
    args = ap.parse_args()
    versions = {p: importlib.metadata.version(p)
                for p in ("structured-data-models", "torch", "numpy", "pandas", "scikit-learn")}
    if versions["structured-data-models"] != "0.2.1":
        raise RuntimeError("This benchmark pins structured-data-models==0.2.1")
    protocol = {
        "repeats": REPEATS, "max_rows": MAX_ROWS, "seed": SEED,
        "test_size": 0.25, "mode": "pretrained", "n_configs": 1,
        "model": "nvidia/Kumo-Tabular", "revision": MODEL_REVISION,
        "size": "small", "num_estimators": NUM_ESTIMATORS, "device": "cpu",
        "regression_point": "mean of q001 through q999",
        "timing": "training-only encoding, table conversion, forward pass and scoring; weights loaded separately",
        "warmup": "none; one forward pass per split", "datasets": args.datasets.split(","),
    }
    specs = [(d, r) for d in protocol["datasets"] for r in range(REPEATS)]
    rows = run_jobs("bench_libraries_kumo", specs, args.cache_dir,
                    workers=1, timeout=3600, uses_mboostrs=lambda _: False,
                    env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                         "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    artifact = {
        "provenance": {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "commit": {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine(), "python": platform.python_version()},
            "versions": versions, "threads": {"per_fit": 1, "torch_interop": 1},
            "protocol": protocol, "repeats": {"outer_splits": REPEATS},
            "uncertainty": "per-split scores; report mean and sample standard deviation",
        },
        "protocol": protocol, "results": rows,
    }
    Path(args.out).write_text(json.dumps(artifact, indent=1, allow_nan=False) + "\n")
    if any(row["_status"] != "success" for row in rows):
        raise SystemExit("Kumo jobs failed; inspect the artifact before publishing scores")


if __name__ == "__main__":
    main()
