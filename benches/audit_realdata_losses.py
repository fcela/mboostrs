"""Training-only S1.4 ablations for naval propulsion and kin8nm.

The benchmark outer test rows are excluded before discovery or scoring.
Each audit fold discovers its own specification on its fitting rows. Path
scores are descriptive development evidence, not unbiased estimates for a
variant selected using these scores. No confirmation dataset is accepted.

    .venv/bin/python benches/audit_realdata_losses.py --out /tmp/loss-audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

if __package__:
    from .bench_realdata import DATASETS, MAX_TRAIN, load
else:
    from bench_realdata import DATASETS, MAX_TRAIN, load

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DATASETS = (44969, 44980)
GRID = (100, 400, 1600, 6400)


def training_indices(n, seed):
    """Exactly the benchmark's training indices, without reading test rows."""
    return np.random.default_rng(seed).permutation(n)[:int(0.8 * n)][:MAX_TRAIN]


def audit_partitions(n, folds, seed):
    if not 2 <= folds <= n:
        raise ValueError("audit folds must be between 2 and the training row count")
    order = np.random.default_rng(seed).permutation(n)
    for held in np.array_split(order, folds):
        yield np.setdiff1d(np.arange(n), held), held


def with_knots(formula, knots):
    def change(match):
        args = re.sub(r",\s*knots\s*=\s*\d+", "", match[1])
        return f"bbs({args}, knots={knots})"
    return re.sub(r"\bbbs\(([^()]*)\)", change, formula)


def measurement_formula(frame, response, knots=20):
    terms = []
    for name in frame.columns:
        if name == response or frame[name].n_unique() <= 1:
            continue
        # Low-cardinality numeric predictors are deliberately measurements
        # in this arm; binary predictors retain a linear contrast.
        terms.append(f"bols({name})" if frame[name].n_unique() == 2
                     else f"bbs({name}, df=4, knots={knots})")
    return f"{response} ~ " + (" + ".join(terms) or "1")


def score_predictions(observed, predicted):
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("predictions must be finite and match the audit rows")
    error = (predicted - observed) ** 2
    return {"mse": float(error.mean()), "rmse": float(np.sqrt(error.mean()))}


def run_fold(train, response, fit_rows, held_rows, seed, fold, budget):
    import mboostrs as mb

    fitting, held = train[fit_rows], train[held_rows]
    observed = held[response].to_numpy()
    predictors = held.drop(response)
    started = time.perf_counter()
    # All adaptive decisions, including inner tuning and family inference,
    # see only fitting rows. The audit response is used solely for scores.
    rec = mb.autospec(fitting, response, budget=budget, seed=seed,
                      interactions="report", residual_tree=False)
    discovery_seconds = time.perf_counter() - started
    grid = sorted(set((*GRID, int(rec.mstop))))
    control = rec.control._replace(mstop=max(grid))
    metadata = {
        "fold": fold, "fit_rows": fit_rows.tolist(), "held_rows": held_rows.tolist(),
        "formula": rec.formula, "family": type(rec.family).__name__,
        "recommended_mstop": int(rec.mstop), "nu": float(control.nu),
        "inner_folds": rec.folds_description, "discovery_seconds": discovery_seconds,
        "discovery_reported_fits": rec.fits, "discovery_report": rec.report(),
    }
    variants = [
        ("discovered_inferred", rec.formula, rec.family),
        ("discovered_gaussian", rec.formula, mb.Gaussian()),
        ("discovered_gaussian_knots40", with_knots(rec.formula, 40), mb.Gaussian()),
        ("measurements_gaussian_knots20", measurement_formula(fitting, response), mb.Gaussian()),
    ]
    results = []
    for name, formula, family in variants:
        start = time.perf_counter()
        try:
            fit = mb.gamboost(formula, data=fitting, family=family, control=control,
                              na_action="error")
            prediction = fit.predict_over_mstop(
                grid, newdata=predictors, type="response", output="numpy",
                unseen_levels="zero_effect")
            scores = [{"mstop": step, **score_predictions(observed, pred)}
                      for step, pred in zip(grid, prediction, strict=True)]
            result = {"variant": name, "formula": formula, "status": "success",
                      "scores": scores, "seconds": time.perf_counter() - start}
            # The refit uses all declared measurement terms, avoiding a
            # second selection based on audit scores or path endpoint.
            if name == "measurements_gaussian_knots20":
                refit_start = time.perf_counter()
                try:
                    refitted = mb.gam_refit(fit, which="all")
                    result["reml"] = {
                        "status": "success",
                        **score_predictions(observed, refitted.predict(predictors)),
                        "seconds": time.perf_counter() - refit_start,
                    }
                except Exception as exc:
                    result["reml"] = {"status": "failure", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
        except Exception as exc:
            results.append({"variant": name, "formula": formula, "status": "failure",
                            "error": f"{type(exc).__name__}: {exc}",
                            "seconds": time.perf_counter() - start})
    return {**metadata, "variants": results}


def provenance():
    import mboostrs as mb

    sources = [Path(__file__).resolve(), ROOT / "benches/bench_realdata.py",
               *sorted((ROOT / "python/mboostrs").glob("*.py"))]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "native_build": dict(mb.build_info()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "threads": {name: os.environ.get(name) for name in
                    ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, nargs="+", choices=AUDIT_DATASETS,
                        default=list(AUDIT_DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import mboostrs as mb
    import polars as pl

    if mb.build_info()["profile"] != "release":
        parser.error("loss ablations require a release extension")
    artifact = {
        "provenance": provenance(),
        "protocol": {"suite": "development_training_only", "outer_seed": args.seed,
                     "audit_folds": args.folds, "autospec_budget": args.budget,
                     "grid": list(GRID), "outer_test_scored": False,
                     "confirmation_loaded": False,
                     "interpretation": "descriptive ablations; audit scores may steer development, not confirmation claims"},
        "datasets": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out)

    save()
    for data_id in args.datasets:
        frame, response, categorical = load(data_id)
        indices = training_indices(len(frame), args.seed)
        train = pl.from_pandas(frame.iloc[indices].reset_index(drop=True))
        if categorical or not np.isfinite(train.to_numpy()).all():
            raise ValueError("this first audit supports complete numeric predictors only")
        record = {"data_id": data_id, "dataset": DATASETS[data_id], "response": response,
                  "training_indices": indices.tolist(), "n_train": len(train),
                  "data_sha256": hashlib.sha256(train.to_numpy().tobytes()).hexdigest(),
                  "folds": []}
        artifact["datasets"].append(record)
        for fold, (fit_rows, held_rows) in enumerate(audit_partitions(len(train), args.folds, args.seed)):
            print(f"{DATASETS[data_id]} fold {fold + 1}/{args.folds}", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = run_fold(train, response, fit_rows, held_rows, args.seed,
                                      fold, args.budget)
                except Exception as exc:
                    result = {"fold": fold, "status": "failure",
                              "error": f"{type(exc).__name__}: {exc}"}
                result["warnings"] = sorted({str(w.message) for w in caught})
            record["folds"].append(result)
            save()
            print(json.dumps({"dataset": DATASETS[data_id], "fold": fold,
                              "family": result.get("family"),
                              "mstop": result.get("recommended_mstop"),
                              "error": result.get("error")}), flush=True)


if __name__ == "__main__":
    main()
