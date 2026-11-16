"""Follow up S1.4's singular all-smooth refit without changing its artifact.

Remove only exact duplicate predictors, detected on each fitting partition;
preserve the response and audit rows. Compare the same deduplicated formula
at 10,000 and 102,400 boosting steps and with a joint REML fit. This is a
development diagnostic prompted by the initial naval fold failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

from audit_realdata_losses import (
    AUDIT_DATASETS,
    ROOT,
    measurement_formula,
    provenance,
    score_predictions,
)
from bench_realdata import load


def duplicate_predictors(frame, response):
    kept, dropped = [], {}
    for name in frame.columns:
        if name == response:
            continue
        values = frame[name].to_numpy()
        match = next((other for other in kept
                      if np.array_equal(values, frame[other].to_numpy())), None)
        if match is None:
            kept.append(name)
        else:
            dropped[name] = match
    return dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import mboostrs as mb
    import polars as pl

    if mb.build_info()["profile"] != "release":
        parser.error("conditioning audit requires a release extension")
    raw = args.source.read_bytes()
    source = json.loads(raw)
    if (source["protocol"]["suite"] != "development_training_only"
            or any(d["data_id"] not in AUDIT_DATASETS for d in source["datasets"])):
        parser.error("only the development loss-audit datasets are supported")
    artifact = {
        "provenance": provenance(), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol": {"deduplication": "exact equality of predictor values on fitting rows",
                     "grid": [10000, 102400], "nu": 0.5,
                     "outer_test_scored": False, "confirmation_loaded": False,
                     "scope": "development follow-up; no inference or defaults changed"},
        "results": [],
    }
    artifact["provenance"]["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for dataset in source["datasets"]:
        frame, response, _ = load(dataset["data_id"])
        train = pl.from_pandas(frame.iloc[dataset["training_indices"]].reset_index(drop=True))
        for fold in dataset["folds"]:
            if "fit_rows" not in fold:
                continue
            fitting = train[np.asarray(fold["fit_rows"])]
            held = train[np.asarray(fold["held_rows"])]
            duplicates = duplicate_predictors(fitting, response)
            fitting = fitting.drop(list(duplicates))
            predictors = held.drop([response, *duplicates])
            observed = held[response].to_numpy()
            formula = measurement_formula(fitting, response)
            start = time.perf_counter()
            result = {"dataset": dataset["dataset"], "fold": fold["fold"],
                      "duplicates": duplicates, "formula": formula}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = mb.gamboost(formula, data=fitting, family=mb.Gaussian(),
                                      control=mb.boost_control(102400, 0.5))
                    predictions = fit.predict_over_mstop(
                        [10000, 102400], newdata=predictors, type="response", output="numpy")
                    result["boosting"] = [
                        {"mstop": step, **score_predictions(observed, prediction)}
                        for step, prediction in zip([10000, 102400], predictions, strict=True)]
                    refitted = mb.gam_refit(fit, which="all")
                    result["reml"] = {**score_predictions(observed, refitted.predict(predictors)),
                                      "convergence": refitted.convergence,
                                      "edf": refitted.edf, "sigma": refitted.sigma}
                    result["status"] = ("success" if refitted.convergence.get("converged")
                                        else "nonconverged")
                except Exception as exc:
                    result.update(status="failure", error=f"{type(exc).__name__}: {exc}")
                result["warnings"] = sorted({str(w.message) for w in caught})
            result["seconds"] = time.perf_counter() - start
            artifact["results"].append(result)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
            print(json.dumps({k: result.get(k) for k in
                              ("dataset", "fold", "duplicates", "status", "boosting", "error")}),
                  flush=True)


if __name__ == "__main__":
    main()
