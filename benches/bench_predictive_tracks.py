"""Frozen D11 comparison with supplied-structure and data-alone tracks."""
from __future__ import annotations

import argparse
import hashlib
import os
import warnings
from pathlib import Path

THREADS = {k: "1" for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
os.environ.update(THREADS)
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/book/benchmarks/predictive_tracks/v2"
TASKS = ("smooth", "interaction", "factor")
METHODS = ("mboostrs_known", "mboostrs_discover", "catboost", "lightgbm", "xgboost", "histgb", "ridge")
FORMULAS = {
    "smooth": "y ~ bbs(x1, df=4) + bols(x2)",
    "interaction": "y ~ bbs(x1, df=4, knots=8) %X% bbs(x2, df=4, knots=8) + bols(x3)",
    "factor": "y ~ bbs(x1, df=4) + bbs(x1, df=4, by=g) + bols(x2)",
}


def draw(task, seed):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(539400 + 10000 * TASKS.index(task) + seed)
    x = rng.uniform(-1, 1, (2800, 5))
    frame = pd.DataFrame(x, columns=["x1", "x2", "x3", "n0", "n1"])
    if task == "smooth":
        mu = 2 * np.sin(np.pi * x[:, 0]) + .5 * x[:, 1]
    elif task == "interaction":
        mu = 2 * np.sin(np.pi * x[:, 0] * x[:, 1]) + .5 * x[:, 2]
    else:
        g = rng.choice(["a", "b"], len(x))
        frame["g"] = g
        mu = 2 * np.sin(np.pi * x[:, 0]) * np.where(g == "a", 1, -1) + .5 * x[:, 1]
    frame["y"] = mu + rng.normal(size=len(x))
    return frame.iloc[:800].copy(), frame.iloc[800:].copy(), mu[800:]


def competitor(method, data, seed):
    import numpy as np
    import pandas as pd

    x = data.drop(columns="y").copy()
    for c in x.columns:
        if x[c].dtype == object:
            x[c] = pd.Categorical(x[c])
    y = data["y"].to_numpy()
    choices = []
    for size in (400, 1000) if method != "ridge" else (None,):
        for capacity in ((3, 6) if method == "catboost" else (2, 4) if method == "xgboost"
                         else (7, 15) if method in ("lightgbm", "histgb") else (.1, 1., 10., 100.)):
            choices.append({"size": size, "capacity": capacity})

    def build(setting):
        size, cap = setting["size"], setting["capacity"]
        if method == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(loss_function="RMSE", iterations=size, depth=cap,
                learning_rate=.05, verbose=False, random_seed=seed, thread_count=1,
                allow_writing_files=False, cat_features=[c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)])
        if method == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=size, num_leaves=cap, learning_rate=.05,
                                  min_child_samples=20, verbosity=-1, random_state=seed, n_jobs=1)
        if method == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=size, max_depth=cap, learning_rate=.05,
                reg_lambda=1., tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=1)
        if method == "histgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(max_iter=size, max_leaf_nodes=cap,
                learning_rate=.05, min_samples_leaf=20, early_stopping=False, random_state=seed)
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        numeric = [c for c in x if not isinstance(x[c].dtype, pd.CategoricalDtype)]
        factor = [c for c in x if c not in numeric]
        prep = ColumnTransformer([("numeric", StandardScaler(), numeric),
                                  ("factor", OneHotEncoder(handle_unknown="ignore", sparse_output=False), factor)])
        return make_pipeline(prep, Ridge(alpha=cap))

    candidates = []
    for setting in choices:
        fit = build(setting).fit(x.iloc[:600], y[:600])
        candidates.append({**setting, "validation_mse": float(np.mean((fit.predict(x.iloc[600:]) - y[600:]) ** 2))})
    chosen = min(candidates, key=lambda c: c["validation_mse"])
    fit = build(chosen).fit(x, y)
    return fit, {"candidates": candidates, "chosen": chosen, "fit_count": 5}


def run_job(spec):
    import mboostrs as mb
    import numpy as np
    import pandas as pd

    from benches.benchmark_contract import data_digest

    task, seed, method = spec
    data, test, truth = draw(task, seed)
    newdata = test.drop(columns="y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "mboostrs_known":
            tuned = mb.tune_mstop(FORMULAS[task], data, family=mb.Gaussian(),
                folds=5, start=100, max_mstop=1600, nu=.1, seed=seed, score="mse")
            model = tuned.model
            prediction = model.predict(newdata, type="response", extrapolation="clip")
            evidence = {"formula": FORMULAS[task], "family": "Gaussian",
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
    return {"task": task, "seed": seed, "method": method,
            "mse": float(np.mean((prediction - test["y"].to_numpy()) ** 2)),
            "oracle_mse": float(np.mean((truth - test["y"].to_numpy()) ** 2)),
            "mean_function_mse": float(np.mean((prediction - truth) ** 2)),
            "input_digest": data_digest([data.to_dict("list"), test.to_dict("list")]),
            "prediction_digest": data_digest(prediction),
            "warnings": [str(w.message) for w in caught], "evidence": evidence}


def main():
    from benches._jobs import run_jobs
    from benches.study_evidence import evidence_document, write_evidence

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    specs = [(task, seed, method) for task in TASKS for seed in range(8) for method in METHODS]
    protocol = DIRECTORY / "PROTOCOL.md"
    document = evidence_document({"id": "predictive-tracks/2", "phase": args.phase, "specs": specs,
        "document_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "scorecard_sha256": hashlib.sha256((ROOT / "docs/scorecard/scorecard.md").read_bytes()).hexdigest(),
        "guide_sha256": hashlib.sha256((ROOT / "docs/book/benchmarks/BENCHMARK_GUIDANCE.md").read_bytes()).hexdigest(),
        "timeout_seconds": 240, "workers": 1, "target_gap_pct": 1.},
        {"independent_draws_per_process": 8}, [Path(__file__)])
    write_evidence(document, DIRECTORY / f"{args.phase}.planned.json")
    document["results"] = run_jobs("bench_predictive_tracks", specs,
        ROOT / f"benches/.cache/predictive_tracks/v2/{args.phase}", workers=1, timeout=240, env=THREADS)
    write_evidence(document, DIRECTORY / f"{args.phase}.json")


if __name__ == "__main__":
    main()
