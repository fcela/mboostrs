"""Frozen paired encoding views in supplied-structure and discovery tracks."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import warnings
from pathlib import Path

from benches.bench_predictive_tracks import METHODS, THREADS, competitor

os.environ.update(THREADS)
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/encoding_tracks"
ENCODINGS = ("categorical", "indicator", "boolean", "signed", "shifted")
BASE_FORMULA = "y ~ bbs(x1, df=4) + bbs(x1, df=4, by=g) + bols(x2)"
FORMULAS = {encoding: BASE_FORMULA for encoding in ENCODINGS}
FORMULAS["signed"] = "y ~ bbs(x1, df=4, by=g) + bols(x2)"


def draw(encoding, seed):
    import numpy as np
    import pandas as pd
    from benches.benchmark_contract import data_digest

    rng = np.random.default_rng(619000 + seed)
    x = rng.uniform(-1, 1, (2800, 5))
    group = rng.integers(0, 2, len(x))
    mu = 2 * np.sin(np.pi * x[:, 0]) * (1 - 2 * group) + .5 * x[:, 1]
    response = mu + rng.normal(size=len(x))
    variants = {"categorical": np.where(group == 0, "a", "b"), "indicator": group,
                "boolean": group == 1, "signed": 1 - 2 * group, "shifted": group + 2}
    frame = pd.DataFrame(x, columns=["x1", "x2", "x3", "n0", "n1"])
    frame["g"] = variants[encoding]
    frame["y"] = response
    latent = data_digest([x, group, response])
    return frame.iloc[:800].copy(), frame.iloc[800:].copy(), mu[800:], latent


def run_job(spec):
    import mboostrs as mb
    import numpy as np
    import pandas as pd
    from benches.benchmark_contract import data_digest

    encoding, seed, method = spec
    data, test, truth, latent = draw(encoding, seed)
    newdata = test.drop(columns="y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "mboostrs_known":
            tuned = mb.tune_mstop(FORMULAS[encoding], data, family=mb.Gaussian(),
                folds=5, start=100, max_mstop=1600, nu=.1, seed=seed, score="mse")
            prediction = tuned.model.predict(newdata, type="response", extrapolation="clip")
            evidence = {"formula": FORMULAS[encoding], "family": "Gaussian",
                        "tuning": tuned.to_dict(include_arrays=True)}
        elif method == "mboostrs_discover":
            result = mb.discover(data, "y", seed=seed, folds=5, max_mstop=1600)
            options = {"parameter": "mu"} if result.spread_formula is not None else {}
            prediction = result.model.predict(newdata, type="response", extrapolation="clip", **options)
            evidence = {"formula": result.formula, "family": result.family,
                        "spread_formula": result.spread_formula, "mstop": result.mstop,
                        "log": result.log, "warnings": result.warnings,
                        "mean_effects": result.mean_effects, "interactions": result.interactions}
        else:
            fit, evidence = competitor(method, data, seed)
            for c in newdata.columns:
                if newdata[c].dtype == object:
                    newdata[c] = pd.Categorical(newdata[c])
            prediction = fit.predict(newdata)
    prediction = np.asarray(prediction).ravel()
    if prediction.shape != (2000,) or not np.isfinite(prediction).all():
        raise ValueError("outer predictions must be 2000 finite means")
    return {"task": encoding, "seed": seed, "method": method,
            "mse": float(np.mean((prediction - test["y"].to_numpy()) ** 2)),
            "oracle_mse": float(np.mean((truth - test["y"].to_numpy()) ** 2)),
            "mean_function_mse": float(np.mean((prediction - truth) ** 2)),
            "input_digest": data_digest([data.to_dict("list"), test.to_dict("list")]),
            "latent_digest": latent, "prediction_digest": data_digest(prediction),
            "warnings": [str(w.message) for w in caught], "evidence": evidence}


def main():
    import mboostrs as mb
    from benches._jobs import run_jobs
    from benches.study_evidence import evidence_document, write_evidence

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    specs = [(encoding, seed, method) for encoding in ENCODINGS for seed in range(8) for method in METHODS]
    library = Path(mb.__file__).resolve().parents[2]
    document = evidence_document({"id": "encoding-tracks/1", "phase": args.phase, "specs": specs,
        "document_sha256": hashlib.sha256((DIRECTORY / "PROTOCOL.md").read_bytes()).hexdigest(),
        "scorecard_sha256": hashlib.sha256((ROOT / "docs/scorecard/scorecard.md").read_bytes()).hexdigest(),
        "guide_sha256": hashlib.sha256((ROOT / "docs/book/benchmarks/BENCHMARK_GUIDANCE.md").read_bytes()).hexdigest(),
        "library_source_revision": subprocess.check_output(["git", "-C", str(library), "rev-parse", "HEAD"], text=True).strip(),
        "timeout_seconds": 240, "workers": 1, "target_gap_pct": 1.},
        {"paired_latent_draws": 8, "encodings_per_draw": 5}, [Path(__file__), ROOT / "benches/bench_predictive_tracks.py"])
    document["results"] = run_jobs("bench_encoding_tracks", specs,
        ROOT / f"benches/.cache/encoding_tracks/{args.phase}", workers=1, timeout=240, env=THREADS)
    write_evidence(document, DIRECTORY / f"{args.phase}.json")


if __name__ == "__main__":
    main()
