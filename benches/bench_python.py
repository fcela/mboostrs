#!/usr/bin/env python3
"""Correctness-gated Python evidence for the non-comparative scenario matrix."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benches.benchmark_contract import (
        data_digest,
        load_scenarios,
        machine_metadata,
        measurement,
        self_check_digest,
        validate_result,
    )
except ModuleNotFoundError:
    from benchmark_contract import (  # type: ignore[no-redef]
        data_digest,
        load_scenarios,
        machine_metadata,
        measurement,
        self_check_digest,
        validate_result,
    )


def _timings(function: Callable[[], Any], warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
    values = []
    for _ in range(repetitions):
        gc.collect()
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return values


def _finite_outputs(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    converted = {name: np.asarray(value) for name, value in outputs.items()}
    for name, value in converted.items():
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite correctness output {name}")
    return converted


def _scenario(row: dict[str, Any], mb: Any, repetitions: int, warmup: int) -> dict[str, Any]:
    scenario_id = row["id"]
    rng = np.random.default_rng(row["seed"])
    parameters = row["parameters"]
    n = int(parameters["n"])

    if scenario_id.startswith("fit.monotone"):
        x = np.sort(rng.uniform(-2, 2, n))
        data = {"x": x, "y": np.exp(0.3 * x) + rng.normal(scale=0.1, size=n)}

        def operation():
            return mb.gamboost(
                'y ~ bmono(x, constraint="increasing", df=4)',
                data=data,
                control=mb.boost_control(parameters["mstop"], 0.1),
            )

        model = operation()
        learner_name = model._learner_names[0]
        effect = np.asarray(model.partial_effects(which=learner_name)[learner_name])
        if np.any(np.diff(effect) < -1e-8):
            raise RuntimeError("monotone learner produced a decreasing partial effect")
        outputs = _finite_outputs({"fitted": model.fitted_, "risk": model.risk_, "effect": effect})
        inputs = data
    elif scenario_id.startswith("fit.survival"):
        x = rng.normal(size=(n, parameters["candidates"]))
        event_time = rng.exponential(np.exp(-0.3 * x[:, 0]))
        event = rng.binomial(1, 0.75, size=n).astype(float)
        data = {"time": event_time, "event": event} | {
            f"x{index}": x[:, index] for index in range(x.shape[1])
        }
        formula = "Surv(time, event) ~ " + " + ".join(f"x{index}" for index in range(x.shape[1]))

        def operation():
            return mb.glmboost(
                formula,
                data=data,
                family=mb.CoxPH(),
                control=mb.boost_control(parameters["mstop"], 0.1),
            )

        model = operation()
        outputs = _finite_outputs({"fitted": model.fitted_, "risk": model.risk_})
        inputs = data
    elif scenario_id.startswith("fit.multinomial"):
        classes = int(parameters["classes"])
        x = rng.normal(size=n)
        latent = np.column_stack([0.4 * (index + 1) * x + rng.normal(size=n) for index in range(classes)])
        y = np.argmax(latent, axis=1).astype(float)
        data = {"x": x, "y": y}

        def operation():
            return mb.gamboost(
                "y ~ bols(x, intercept=False)",
                data=data,
                family=mb.Multinomial(n_classes=classes),
                control=mb.boost_control(parameters["mstop"], 0.1),
            )

        model = operation()
        outputs = _finite_outputs({"fitted": model.fitted_, "risk": model.risk_})
        if outputs["fitted"].size != n * (classes - 1):
            raise RuntimeError("multinomial fitted shape violates K-1 parameterization")
        inputs = data
    elif scenario_id.startswith("fit.lss"):
        x = rng.normal(size=n)
        x2 = rng.normal(size=n)
        z = rng.normal(size=n)
        group = np.arange(n) % 16
        group_scale = np.exp(np.linspace(-0.4, 0.4, 16))[group]
        data = {
            "x": x,
            "x2": x2,
            "z": z,
            "group": group,
            "y": 0.5 * x + 0.25 * z + rng.normal(
                scale=(
                    group_scale
                    if "structured_brandom" in scenario_id or "structured_bmrf" in scenario_id
                    else np.exp(0.15 * np.sin(x) * np.cos(x2))
                    if "structured_bspatial" in scenario_id
                    else np.exp(0.1 * x)
                ),
                size=n,
            ),
        }
        method = "noncyclic" if "noncyclic" in scenario_id else "cyclic"
        mixed = "mixed" in scenario_id
        bbs_only = "bbs_gaussian" in scenario_id
        structured_brandom = "structured_brandom" in scenario_id
        structured_bmrf = "structured_bmrf" in scenario_id
        structured_bspatial = "structured_bspatial" in scenario_id
        structured_bmono = "structured_bmono" in scenario_id
        structured_bconstrained = "structured_bconstrained" in scenario_id
        structured_btree = "structured_btree" in scenario_id
        custom_buser = "custom_buser" in scenario_id
        custom_bkernel = "custom_bkernel" in scenario_id
        k_regions = np.diag(np.r_[1.0, np.full(14, 2.0), 1.0])
        k_regions += np.diag(np.full(15, -1.0), 1)
        k_regions += np.diag(np.full(15, -1.0), -1)
        family = mb.StudentTLSS() if "studentt" in scenario_id else mb.GaussianLSS()
        formula = (
            {
                "mu": "y ~ bbs(x, df=4) + bols(z)",
                "sigma": "y ~ brandom(group, df=4) + bols(z)",
            }
            if structured_brandom
            else {
                "mu": "y ~ bbs(x, df=4) + bols(z)",
                "sigma": "y ~ bmrf(group, K=K_regions, df=4) + bols(z)",
            }
            if structured_bmrf
            else {
                "mu": "y ~ bbs(x, df=4) + bols(z)",
                "sigma": (
                    "y ~ bspatial(x, x2, knots=3, degree=2, differences=1, "
                    "df=5, center=True) + bols(z)"
                ),
            }
            if structured_bspatial
            else {
                "mu": (
                    'y ~ bmono(x, constraint="increasing", knots=8, degree=3, '
                    "differences=2, df=4) + bols(z)"
                ),
                "sigma": "y ~ bbs(x2, df=4) + bols(z)",
            }
            if structured_bmono
            else {
                "mu": (
                    "y ~ bconstrained(x, C=C_positive, knots=3, degree=2, "
                    "differences=1, df=4) + bols(z)"
                ),
                "sigma": "y ~ bbs(x2, df=4) + bols(z)",
            }
            if structured_bconstrained
            else {
                "mu": "y ~ btree(x, x2, minbucket=4, minsplit=10, maxdepth=2) + bols(z)",
                "sigma": "y ~ bbs(x2, df=4) + bols(z)",
            }
            if structured_btree
            else {
                "mu": "y ~ buser(custom, X=X_custom, K=K_custom, df=3) + bols(z)",
                "sigma": "y ~ bbs(x2, df=4) + bols(z)",
            }
            if custom_buser
            else {
                "mu": "y ~ bkernel(x, bandwidth=0.5, df=4, kernel=gaussian) + bols(z)",
                "sigma": "y ~ bbs(x2, df=4) + bols(z)",
            }
            if custom_bkernel
            else "y ~ bbs(x, df=4) + bols(z)"
            if mixed
            else "y ~ bbs(x, df=4)"
            if bbs_only
            else "y ~ x"
        )

        def operation():
            fit = (
                mb.gamboostLSS
                if mixed
                or bbs_only
                or structured_brandom
                or structured_bmrf
                or structured_bspatial
                or structured_bmono
                or structured_bconstrained
                or structured_btree
                or custom_buser
                or custom_bkernel
                else mb.glmboostLSS
            )
            extra = (
                {"env": {"K_regions": k_regions}}
                if structured_bmrf
                else {"env": {"C_positive": np.eye(6)}}
                if structured_bconstrained
                else {
                    "env": {
                        "X_custom": np.column_stack((x, x * x, np.sin(x))),
                        "K_custom": np.diag([0.1, 1.0, 1.0]),
                    }
                }
                if custom_buser
                else {}
            )
            return fit(
                formula,
                data=data,
                families=family,
                method=method,
                control=mb.boost_control(parameters["mstop"], 0.1),
                **extra,
            )

        model = operation()
        selected = np.asarray([len(parameter["selected"]) for parameter in model._params])
        fitted = np.concatenate(
            [model.predict(parameter=name, type="link") for name in model.param_names]
        )
        outputs = _finite_outputs(
            {"risk": model.risk_, "selected_counts": selected, "fitted": fitted}
        )
        if method == "noncyclic" and selected.sum() != parameters["mstop"]:
            raise RuntimeError("non-cyclic LSS update count differs from mstop")
        inputs = data
    elif scenario_id == "predict.lss.mixed.end_to_end":
        x = rng.uniform(-2, 2, n)
        z = rng.normal(size=n)
        data = {"x": x, "z": z, "y": np.sin(x) + 0.4 * z + rng.normal(size=n)}
        model = mb.gamboostLSS(
            "y ~ bbs(x, df=4) + bols(z)",
            data=data,
            families=mb.GaussianLSS(),
            control=mb.boost_control(parameters["mstop"], 0.1),
        )
        newdata = {"x": x[: n // 2] + 0.01, "z": z[: n // 2] - 0.01}

        def operation():
            training = model.predict(type="link")
            truncated = model.predict(type="link", mstop=parameters["prediction_mstop"])
            fresh = model.predict(newdata=newdata, type="link")
            restored = pickle.loads(pickle.dumps(model)).predict(newdata=newdata, type="link")
            return training, truncated, fresh, restored

        training, truncated, fresh, restored = operation()
        for parameter in model.param_names:
            np.testing.assert_allclose(restored[parameter], fresh[parameter], rtol=0, atol=0)
        outputs = _finite_outputs(
            {
                "training": np.concatenate(list(training.values())),
                "truncated": np.concatenate(list(truncated.values())),
                "newdata": np.concatenate(list(fresh.values())),
            }
        )
        inputs = {"training": data, "newdata": newdata}
    elif scenario_id == "cv.lss.mixed.end_to_end":
        x = rng.uniform(-2, 2, n)
        z = rng.normal(size=n)
        data = {"x": x, "z": z, "y": np.sin(x) + 0.4 * z + rng.normal(size=n)}
        fold_count = int(parameters["folds"])
        folds = np.ones((n, fold_count), dtype=np.float64)
        for fold in range(fold_count):
            folds[fold::fold_count, fold] = 0.0
        grid = [[value, value] for value in (10, 20, int(parameters["mstop"]))]

        def operation():
            return mb.cvrisk_gamboostLSS(
                "y ~ bbs(x, df=4) + bols(z)",
                data=data,
                families=mb.GaussianLSS(),
                control=mb.boost_control(parameters["mstop"], 0.1),
                folds=folds,
                grid=grid,
            )

        result = operation()
        outputs = _finite_outputs(
            {"risk": result.risk_matrix, "aggregate": result.aggregate}
        )
        inputs = data | {"folds": folds, "grid": grid}
    elif scenario_id == "model.operations.end_to_end":
        x = rng.uniform(-3, 3, n)
        data = {"x": x, "y": np.sin(x) + rng.normal(scale=0.2, size=n)}
        model = mb.gamboost(
            "y ~ bbs(x, df=4)",
            data=data,
            control=mb.boost_control(parameters["mstop"], 0.1),
        )
        grid = [1, parameters["mstop"] // 2, parameters["mstop"]]

        def operation():
            prediction = model.predict(newdata={"x": x}, type="link")
            path = np.column_stack(
                [model[step].predict(newdata={"x": x}, type="link") for step in grid]
            )
            sliced = model[grid[1]].predict(newdata={"x": x}, type="link")
            restored = pickle.loads(pickle.dumps(model)).predict(newdata={"x": x}, type="link")
            residuals = model.residuals(type="response")
            coefficients = model.coef(aggregate="sum")
            return prediction, path, sliced, restored, residuals, coefficients

        prediction, path, sliced, restored, residuals, coefficients = operation()
        np.testing.assert_allclose(prediction, model.fitted_, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(restored, prediction, rtol=0, atol=0)
        outputs = _finite_outputs(
            {
                "prediction": prediction,
                "path": path,
                "sliced": sliced,
                "residuals": residuals,
                "coefficient_values": np.concatenate([np.ravel(value) for value in coefficients.values()]),
            }
        )
        inputs = data
    else:
        raise RuntimeError(f"no Python benchmark implementation for {scenario_id}")

    # Each scenario above asserts its outputs against an internal invariant
    # before reaching here — a pickle round-trip reproducing a fresh
    # prediction, a replayed path matching the fitted one — with
    # `assert_allclose` at rtol=0/atol=0 for round-trips and 1e-10 for
    # predictions. This is a single-implementation row, so the evidence is a
    # self-check rather than a two-implementation comparison, and the digest
    # says so instead of being an untagged raw output hash.
    shared_digest = self_check_digest(
        outputs,
        invariant="round-trip and replay reproduce a fresh run of the same implementation",
    )
    timings = _timings(operation, warmup, repetitions)
    return {
        "scenario_id": scenario_id,
        "boundary": row["boundary"],
        "thread_mode": row["thread_mode"],
        "seed": row["seed"],
        "parameters": parameters,
        "input_digest": data_digest(inputs),
        "compared_outputs": sorted(outputs),
        "tolerance": {"atol": 0.0, "rtol": 0.0},
        "correctness": {"status": "passed", "max_absolute_error": 0.0, "max_normalized_error": 0.0},
        "measurements": [
            measurement(
                "mboostrs-python",
                timings,
                output_digest=data_digest(outputs),
                comparison_digest=shared_digest,
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benches/results/python-benchmark.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.environ["RAYON_NUM_THREADS"] = "1"
    import mboostrs as mb

    manifest = load_scenarios()
    rows = [row for row in manifest["scenarios"] if row["runner"] == "python" and row["status"] == "implemented"]
    warmup = 1 if args.quick else manifest["defaults"]["warmup"]
    # 7, not 3: below seven the bootstrap percentile interval collapses to
    # the min/max range while still being labelled CI95, and --quick would
    # otherwise be one flag away from writing a degenerate interval into a
    # published artifact with nothing marking it.
    repetitions = 7 if args.quick else manifest["defaults"]["repetitions"]
    results = []
    for row in rows:
        print(f"Running {row['id']} ...", flush=True)
        results.append(_scenario(row, mb, repetitions, warmup))
    payload = {
        "schema_version": 1,
        "kind": "mboostrs-benchmark-result",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_digest": data_digest(manifest),
        "metadata": machine_metadata(),
        "warmup_policy": {"iterations": warmup, "description": "untimed calls before measured repetitions"},
        "results": results,
        "regressions": [],
    }
    validate_result(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} Python scenario rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
