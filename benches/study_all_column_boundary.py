"""Original ten-column #608 task, with frozen representation controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import time
import traceback
import warnings

os.environ.update(
    {
        k: "1"
        for k in (
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
)

import mboostrs as mb
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from benches.study_evidence import data_digest, evidence_document
from mboostrs._output import json_safe

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/all_column_boundary"
METHODS = (
    "original_hybrid",
    "additive",
    "supplied_pair",
    "supplied_pair_all_mains",
    "supplied_tuned",
    "discovery",
    "catboost",
    "lightgbm",
)


def draw(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (2000, 10))
    mean = 3 * np.sign(x[:, 0] * x[:, 1]) + x[:, 2]
    y = mean + rng.normal(0, 0.5, 2000)
    fit, test = train_test_split(np.arange(2000), test_size=0.3, random_state=0)
    frame = pl.DataFrame({**{f"x{i}": x[:, i] for i in range(10)}, "y": y})
    return frame, mean, fit, test


def predict(method, train, test):
    all_mains = " + ".join(f"bbs(x{i})" for i in range(10))
    pair = "btree(x0, x1, maxdepth=3, selection=sse)"
    formula = {
        "original_hybrid": all_mains + " + btree(" + ",".join(f"x{i}" for i in range(10)) + ", maxdepth=6)",
        "additive": all_mains,
        "supplied_pair": pair + " + bols(x2)",
        "supplied_pair_all_mains": all_mains + " + " + pair,
        "supplied_tuned": pair + " + bols(x2)",
    }
    if method in formula:
        if method == "supplied_tuned":
            tuning = mb.tune_mstop(
                "y ~ " + formula[method], train, family=mb.Gaussian(), nu=0.1, folds=5, start=100, max_mstop=500, seed=0
            )
            model = tuning.model
            details = dict(formula=formula[method], mstop=tuning.mstop, risk=list(tuning.risk))
        else:
            model = mb.gamboost("y ~ " + formula[method], train, control=mb.boost_control(500))
            details = dict(formula=formula[method], mstop=500)
        return model.predict(test, type="response"), details
    if method == "discovery":
        result = mb.discover(train, "y", seed=0)
        prediction = result.model.predict(test, type="response")
        mean = prediction["mu"] if isinstance(prediction, dict) else prediction
        return mean, dict(
            formula=result.formula,
            spread_formula=result.spread_formula,
            family=result.family,
            mstop=result.mstop,
            log=result.log,
            provenance=result.provenance,
        )
    x, xt = train.drop("y").to_numpy(), test.to_numpy()
    if method == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_seed=0,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=200, max_depth=-1, num_leaves=31, learning_rate=0.05, random_state=0, n_jobs=1, verbosity=-1
        )
    return model.fit(x, train["y"].to_numpy()).predict(xt), dict(settings="PROTOCOL.md")


def run(phase, output, start=None, stop=None):
    first, last = (6087100, 6087110) if phase == "development" else (6088100, 6088150)
    seeds = range(first if start is None else start, last if stop is None else stop)
    assert all(first <= s < last for s in seeds)
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = evidence_document(
        {
            "id": "all-column-boundary/1",
            "phase": phase,
            "sha256": hashlib.sha256((DIRECTORY / "PROTOCOL.md").read_bytes()).hexdigest(),
        },
        {"seeds": list(seeds)},
        [Path(__file__)],
    )
    package = Path(mb.__file__).parent
    metadata["loaded_sources"] = {
        n: hashlib.sha256((package / n).read_bytes()).hexdigest() for n in ("discover.py", "explore.py", "gamboost.py")
    }
    metadata.pop("results")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    count, digest = 0, hashlib.sha256()
    with lzma.open(output, "xb", filters=[{"id": lzma.FILTER_LZMA2, "preset": 3, "dict_size": 16 * 1024**2}]) as stream:
        for seed in seeds:
            frame, truth, training, testing = draw(seed)
            train, test = frame[training], frame[testing]
            base = dict(
                seed=seed,
                input_digest=data_digest([frame.to_numpy(), truth]),
                training_rows=training.tolist(),
                test_rows=testing.tolist(),
                test_response=test["y"].to_list(),
                true_mean=truth[testing].tolist(),
            )
            for method in METHODS:
                row = {**base, "method": method}
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        point, details = predict(method, train, test.drop("y"))
                        point = np.asarray(point, dtype=float).ravel()
                        y = test["y"].to_numpy()
                        assert point.shape == y.shape and np.isfinite(point).all()
                        error = float(np.mean((point - y) ** 2))
                        row.update(
                            status="ok",
                            details=details,
                            prediction=point.tolist(),
                            mse=error,
                            true_mean_mse=float(np.mean((point - truth[testing]) ** 2)),
                            nrmse=float(np.sqrt(error) / y.std(ddof=0)),
                        )
                    except Exception:
                        row.update(status="failed", traceback=traceback.format_exc())
                row.update(elapsed=time.perf_counter() - started, warnings=[str(w.message) for w in caught])
                line = (json.dumps(json_safe(row), allow_nan=False) + "\n").encode()
                digest.update(line)
                stream.write(line)
                stream.flush()
                count += 1
            print(json.dumps(dict(phase=phase, seed=seed, records=count)), flush=True)
    metadata.update(
        records=count,
        compressed_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        uncompressed_sha256=digest.hexdigest(),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    run(args.phase, args.output, args.start, args.stop)
