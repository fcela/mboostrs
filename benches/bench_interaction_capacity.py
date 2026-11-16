"""Frozen learner-form diagnosis for the largest historical interaction gap."""
from __future__ import annotations

import argparse
import hashlib
import os
import warnings
from pathlib import Path

from benches.bench_predictive_tracks import competitor

THREADS = {key: "1" for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
os.environ.update(THREADS)
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/interaction_capacity"
TASKS = ("axis_xor", "shifted_xor", "smooth_product", "additive", "null")
METHODS = ("mboostrs_known", "mboostrs_discover", "catboost", "lightgbm", "xgboost", "histgb", "ridge")
CONTROL_TASKS = ("linear", "curved", "product", "factor", "null", "step")
FORMULAS = {
    "axis_xor": "y ~ btree(x1, x2, maxdepth=3, selection=sse) + bols(x3)",
    "shifted_xor": "y ~ btree(x1, x2, maxdepth=3, selection=sse) + bols(x3)",
    "smooth_product": "y ~ bbs(x1, df=4, knots=8) %X% bbs(x2, df=4, knots=8) + bols(x3)",
    "additive": "y ~ bbs(x1, df=4) + bols(x2)",
    "null": "y ~ 1",
}
TENSOR = ("y ~ bbs(x1, df=2.5, knots=5) + bbs(x2, df=2.5, knots=5) + "
          "bbs(x1, df=2.5, knots=5) %X% bbs(x2, df=2.5, knots=5) + bols(x3)")


def draw(task, seed, control=False):
    import numpy as np
    import pandas as pd

    names = CONTROL_TASKS if control else TASKS
    rng = np.random.default_rng((202710170 if control else 202610170) + 10000*names.index(task) + seed)
    x = rng.uniform(-1, 1, (2200 if control else 2800, 5))
    frame = pd.DataFrame(x, columns=["x1", "x2", "x3", "n0", "n1"])
    if task == "axis_xor":
        mu = 3*np.sign(x[:, 0])*np.sign(x[:, 1]) + x[:, 2]
    elif task == "shifted_xor":
        mu = 3*np.sign(x[:, 0]-.25)*np.sign(x[:, 1]+.15) + x[:, 2]
    elif task == "smooth_product":
        mu = 2*np.sin(np.pi*x[:, 0]*x[:, 1]) + .5*x[:, 2]
    elif task == "additive":
        mu = 2*np.sin(np.pi*x[:, 0]) + .5*x[:, 1]
    elif task == "linear":
        mu = 1.5*x[:, 0]+.5*x[:, 1]
    elif task == "curved":
        mu = np.sin(2*x[:, 0])+x[:, 1]**2
    elif task == "product":
        mu = 2*np.sin(2*x[:, 0]*x[:, 1])+.5*x[:, 2]
    elif task == "factor":
        b = rng.integers(0, 2, len(x))
        frame["g"] = np.where(b == 0, "a", "b")
        mu = 1.5*np.sin(2*x[:, 0])*(1-2*b)+.5*x[:, 1]
    elif task == "step":
        mu = 2*np.sign(x[:, 0]+.3)*np.sign(x[:, 1]-.2)+.5*x[:, 2]
    else:
        mu = np.zeros(len(x))
    frame["y"] = mu+rng.normal(0, .6 if control else .5, len(x))
    ndev = 600 if control else 800
    return frame.iloc[:ndev].copy(), frame.iloc[ndev:].copy(), mu[ndev:]


def run_job(spec):
    import mboostrs as mb
    import numpy as np
    import pandas as pd

    from benches.benchmark_contract import data_digest

    task, seed, method = spec
    control = method == "control"
    data, test, truth = draw(task, seed, control)
    newdata = test.drop(columns="y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method in ("mboostrs_known", "tensor1600", "tensor6400"):
            formula = FORMULAS[task] if method == "mboostrs_known" else TENSOR
            cap = 6400 if method == "tensor6400" else 1600
            tuned = mb.tune_mstop(formula, data, family=mb.Gaussian(), folds=5,
                                 start=100, max_mstop=cap, nu=.1, seed=seed, score="mse")
            model = tuned.model
            prediction = model.predict(newdata, type="response", extrapolation="clip")
            evidence = {"formula": formula, "family": "Gaussian", "tuning": tuned.to_dict(include_arrays=True)}
        elif method in ("mboostrs_discover", "control"):
            result = mb.discover(data, "y", seed=seed, folds=3 if control else 5,
                                 max_mstop=800 if control else 1600)
            kwargs = {"parameter": "mu"} if result.spread_formula is not None else {}
            prediction = result.model.predict(newdata, type="response", extrapolation="clip", **kwargs)
            evidence = result.to_dict()
            evidence["refit_available"] = result.refit is not None
        else:
            fit, evidence = competitor(method, data, seed)
            for column in newdata:
                if newdata[column].dtype == object:
                    newdata[column] = pd.Categorical(newdata[column])
            prediction = fit.predict(newdata)
    prediction = np.asarray(prediction).ravel()
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError("outer predictions must be finite and aligned")
    return {"task": task, "seed": seed, "method": method,
            "mse": float(np.mean((prediction-test["y"].to_numpy())**2)),
            "oracle_mse": float(np.mean((truth-test["y"].to_numpy())**2)),
            "mean_function_mse": float(np.mean((prediction-truth)**2)),
            "input_digest": data_digest([data.to_dict("list"), test.to_dict("list")]),
            "prediction_digest": data_digest(prediction),
            "warnings": [str(w.message) for w in caught], "evidence": evidence}


def main():
    from benches._jobs import run_jobs
    from benches.study_evidence import evidence_document, write_evidence

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--suite", choices=("primary", "diagnostic", "controls"), default="primary")
    args = parser.parse_args()
    specs = ([(t, s, m) for t in TASKS for s in range(8) for m in METHODS] if args.suite == "primary" else
             [(t, s, m) for t in TASKS[:2] for s in range(8) for m in ("tensor1600", "tensor6400")]
             if args.suite == "diagnostic" else
             [(t, s, "control") for t in CONTROL_TASKS for s in range(6)])
    protocol = DIRECTORY / "PROTOCOL.md"
    document = evidence_document({"id": "interaction-capacity/1", "phase": args.phase, "suite": args.suite,
        "specs": specs, "document_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "scorecard_sha256": hashlib.sha256((ROOT / "docs/scorecard/scorecard.md").read_bytes()).hexdigest(),
        "guide_sha256": hashlib.sha256((ROOT / "docs/book/benchmarks/BENCHMARK_GUIDANCE.md").read_bytes()).hexdigest(),
        "timeout_seconds": 240, "workers": 1, "target_gap_pct": 1.},
        {"draws": 6 if args.suite == "controls" else 8},
        [Path(__file__), ROOT / "benches/bench_predictive_tracks.py"])
    target = DIRECTORY / f"{args.phase}_{args.suite}.json"
    write_evidence(document, target.with_suffix(".planned.json"))
    document["results"] = run_jobs("bench_interaction_capacity", specs,
        ROOT / f"benches/.cache/interaction_capacity/{args.phase}/{args.suite}",
        workers=1, timeout=240, env=THREADS)
    write_evidence(document, target)


if __name__ == "__main__":
    main()
