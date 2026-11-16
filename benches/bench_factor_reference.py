"""Fresh paired controls for categorical factor-reference dictionary completion."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

THREADS = {k: "1" for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
os.environ.update(THREADS)
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/factor_reference"
CASES = ("opposed", "relabelled", "smooth", "null", "numeric", "factor_main", "logged")


def data(case, seed):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(618300 + 10000 * CASES.index(case) + seed)
    x, z, n0, n1 = rng.uniform(-1, 1, (4, 2500))
    group = rng.integers(0, 2, len(x))
    labels = np.where(group == 0, "a", "b")
    if case in ("opposed", "relabelled"):
        mu = 2 * np.sin(np.pi * x) * (1 - 2 * group) + .5 * z
        if case == "relabelled":
            labels = np.where(group == 0, "zeta", "alpha")
    elif case == "smooth":
        mu = 2 * np.sin(np.pi * x) + .5 * z
    elif case == "null":
        mu = np.zeros(len(x))
    elif case == "numeric":
        labels = group
        mu = 2 * np.sin(np.pi * x) * group + .5 * z
    elif case == "factor_main":
        mu = 1.5 * np.sin(np.pi * x) + 2 * group + .5 * z
    else:
        u = rng.uniform(0, 4, len(x))
        x = np.exp(u)
        mu = 2 * np.sin(np.pi * (u / 2 - 1)) * (1 - 2 * group) + .5 * z
    frame = pd.DataFrame({"x": x, "z": z, "n0": n0, "n1": n1, "g": labels,
                          "y": mu + rng.normal(0, .5, len(x))})
    return frame.iloc[:500].copy(), frame.iloc[500:].copy()


def run_job(spec):
    import mboostrs as mb
    import numpy as np
    from benches.benchmark_contract import data_digest

    case, seed = spec
    train, test = data(case, seed)
    result = mb.discover(train, "y", seed=seed, folds=3, max_mstop=800, spread=False)
    prediction = np.asarray(result.model.predict(test.drop(columns="y"), type="response", extrapolation="clip")).ravel()
    return {"case": case, "seed": seed, "mse": float(np.mean((prediction - test["y"].to_numpy()) ** 2)),
            "formula": result.formula, "log": result.log, "warnings": result.warnings,
            "mstop": result.mstop, "interactions": result.interactions,
            "input_digest": data_digest([train.to_dict("list"), test.to_dict("list")]),
            "prediction_digest": data_digest(prediction)}


def main():
    import mboostrs as mb
    from benches._jobs import run_jobs
    from benches.study_evidence import evidence_document, write_evidence

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    protocol = DIRECTORY / "PROTOCOL.md"
    specs = [(case, seed) for case in CASES for seed in range(6)]
    library = Path(mb.__file__).resolve().parents[2]
    document = evidence_document({"id": "factor-reference/1", "phase": args.phase, "specs": specs,
        "document_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "library_source_revision": subprocess.check_output(["git", "-C", str(library), "rev-parse", "HEAD"], text=True).strip()},
        {"draws_per_family": 6}, [Path(__file__)])
    document["results"] = run_jobs("bench_factor_reference", specs,
        ROOT / f"benches/.cache/factor_reference/{args.phase}", workers=1, timeout=240, env=THREADS)
    write_evidence(document, DIRECTORY / f"{args.phase}.json")


if __name__ == "__main__":
    main()
