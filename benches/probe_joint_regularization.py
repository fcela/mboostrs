"""Bounded simulation comparing boosting and joint spline regularization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mboostrs as mb  # noqa: E402
from mboostrs import _rust  # noqa: E402

TERMS = [f"bbs(x{j}, df=4, knots=12)" for j in range(8)]
SCENARIOS = ("sparse", "correlated", "dense", "null")


def make_data(seed, n=600, test_n=2000):
    rng = np.random.default_rng(seed)

    def xdraw(size):
        dist, count = size
        x = rng.normal(size=(count, 8)) if dist == "normal" else rng.uniform(-1, 1, (count, 8))
        if dist == "normal":
            x[:, 3] = 0.9 * x[:, 0] + np.sqrt(0.19) * rng.normal(size=count)
        return x

    out = {}
    for scenario in SCENARIOS:
        dist = "normal" if scenario == "correlated" else "uniform"
        train_x, test_x = xdraw((dist, n)), xdraw((dist, test_n))

        def signal(x, scenario=scenario):
            if scenario in ("sparse", "correlated"):
                return np.sin(3 * x[:, 0]) + x[:, 1] ** 2 + 0.5 * x[:, 2]
            if scenario == "dense":
                return sum(np.sin((1 + j % 3) * x[:, j]) for j in range(8)) / np.sqrt(8)
            return np.zeros(x.shape[0])

        train_f, test_f = signal(train_x), signal(test_x)
        train_y = train_f + rng.normal(size=n)
        frame = pl.DataFrame({**{f"x{j}": train_x[:, j] for j in range(8)}, "y": train_y})
        test = pl.DataFrame({**{f"x{j}": test_x[:, j] for j in range(8)}, "y": test_f})
        out[scenario] = (frame, test, test_f, train_x, train_y)
    return out


def null_projector(B, K):
    c = np.asarray(B, float).T @ np.ones(B.shape[0])
    C = np.linalg.qr(c[:, None], mode="complete")[0][:, 1:]
    restricted = C.T @ np.asarray(K, float) @ C
    values, vectors = np.linalg.eigh(restricted)
    tol = max(float(values[-1]), 0.0) * values.size * np.finfo(float).eps * 100
    U0 = vectors[:, values <= tol]
    P0 = C @ U0 @ U0.T @ C.T
    return P0, C, U0


def fit_core(data, y, extra=False):
    designs, penalties, names, variables, raws = [], [], [], [], []
    for j, term in enumerate(TERMS):
        d = mb.learner_design(term, data)
        B, K = np.asarray(d.design, float), np.asarray(d.penalty, float)
        names.append(term)
        designs.append(B)
        raws.append(d)
        variables.append([f"x{j}"])
        plist = [K]
        if extra:
            P0, C, U0 = null_projector(B, K)
            assert np.linalg.norm(C.T @ K @ C @ U0, ord="fro") < 1e-8
            assert np.linalg.norm(P0 @ C @ (C.T @ K @ C), ord="fro") < 1e-8
            plist.append(P0)
        penalties.append(plist)
    core = _rust.gam_refit_core(names, designs, penalties, [None] * 8, np.asarray(y, float), variables=variables)
    return core, raws


def summarize_arm(scenario, seed, method, train, test, truth, xtrain, y, fitfn):
    row = {
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "status": "error",
        "warnings": [],
        "signal_mse": None,
        "wall_seconds": None,
        "tuning_seconds": None,
        "refit_seconds": None,
        "edf": None,
        "training_effect_rms": None,
        "selected_count": None,
        "tuning_cap": None,
        "converged": None,
        "error": None,
    }
    start = time.perf_counter()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred, meta = fitfn()
        row.update(meta)
        row["signal_mse"] = float(np.mean((np.asarray(pred) - truth) ** 2))
        row["status"] = "ok"
    except Exception as exc:  # retained as a row for reproducible failures
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["warnings"] = [str(w.message) for w in caught]
    row["wall_seconds"] = row.pop("total_training_seconds", time.perf_counter() - start)
    row["data_sha256"] = hashlib.sha256(
        np.ascontiguousarray(np.column_stack([xtrain, y])).view(np.uint8).tobytes()
        + np.ascontiguousarray(test.select([f"x{j}" for j in range(8)]).to_numpy()).view(np.uint8).tobytes()
        + np.ascontiguousarray(truth).view(np.uint8).tobytes()
    ).hexdigest()
    return row


def run_seed(seed, n=600, test_n=2000):
    rows = []
    for scenario, (train, test, truth, xtrain, y) in make_data(seed, n, test_n).items():
        formula = "y ~ " + " + ".join(TERMS)
        tuned_state = {}

        def get_tuned(state=tuned_state, formula=formula, train=train, seed=seed):
            if not state:
                t0 = time.perf_counter()
                state["result"] = mb.tune_mstop(formula, train, folds=3, start=100, max_mstop=3000, nu=0.1, seed=seed)
                state["seconds"] = time.perf_counter() - t0
            return state["result"], state["seconds"]

        def boosted(test=test):
            tuned, tuning = get_tuned()
            pred = tuned.model.predict(newdata=test, type="response")
            selected = len(set(tuned.model.selected))
            return pred, {
                "tuning_seconds": tuning,
                "selected_count": selected,
                "mstop": int(tuned.mstop),
                "tuning_cap": bool(tuned.mstop >= 3000),
                "converged": True,
                "total_training_seconds": tuning,
            }

        def selected(train=train, test=test):
            tuned, ts = get_tuned()
            r = time.perf_counter()
            refit = mb.gam_refit(tuned.model, which="selected")
            refit_seconds = time.perf_counter() - r
            pred = refit.predict(test, extrapolation="allow")
            effects = refit.effects(train)
            return pred, {
                "tuning_seconds": ts,
                "refit_seconds": refit_seconds,
                "total_training_seconds": ts + refit_seconds,
                "selected_count": len(set(tuned.model.selected)),
                "mstop": int(tuned.mstop),
                "tuning_cap": tuned.mstop >= 3000,
                "edf": dict(refit.edf),
                "training_effect_rms": {k: float(np.sqrt(np.mean(np.asarray(v[0]) ** 2))) for k, v in effects.items()},
                "converged": bool(refit.convergence.get("converged", True)),
            }

        def core_arm(extra, train=train, y=y, test=test):
            t = time.perf_counter()
            core, recipes = fit_core(train, y, extra)
            elapsed = time.perf_counter() - t
            pred = np.asarray(core.prediction([d.transform(test) for d in recipes], [None] * 8, [False] * 8)[0])
            blocks = list(core.blocks())
            edf = {b["name"]: float(b["edf"]) for b in blocks}
            effects = {}
            for i, recipe in enumerate(recipes):
                effect = core.effect(i, recipe.design, None)[0]
                effects[TERMS[i]] = float(np.sqrt(np.mean(np.asarray(effect) ** 2)))
            return pred, {
                "refit_seconds": elapsed,
                "total_training_seconds": elapsed,
                "edf": edf,
                "training_effect_rms": effects,
                "converged": bool(core.convergence().get("converged", True)),
            }

        for method, fn in (
            ("tuned_boost", boosted),
            ("selected_refit", selected),
            ("joint_reml", lambda: core_arm(False)),
            ("double_penalty", lambda: core_arm(True)),
        ):
            rows.append(summarize_arm(scenario, seed, method, train, test, truth, xtrain, y, fn))
        reference = rows[-4]["signal_mse"]
        for row in rows[-4:]:
            row["signal_mse_delta_vs_boost"] = (
                None if row["signal_mse"] is None or reference is None else row["signal_mse"] - reference
            )
    return rows


def native_source_digest():
    """Match crates/mboostrs-core/build.rs, including bindings and manifests."""
    paths = list((ROOT / "crates").rglob("*.rs"))
    paths += [ROOT / name for name in ("Cargo.toml", "Cargo.lock", ".cargo/config.toml")
              if (ROOT / name).is_file()]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(ROOT).parts):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return "sha256:" + digest.hexdigest()


def evidence():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = None
    import mboostrs

    build = dict(mb.build_info())
    return {
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_git_sha": sha,
        "build_info": build,
        "native_build_matches_checkout": build.get("git_revision") == sha,
        "checkout_native_source_digest": native_source_digest(),
        "native_sources_match": build.get("source_digest") == native_source_digest(),
        "python": sys.version,
        "numpy": np.__version__,
        "threads": {
            k: os.environ.get(k)
            for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "platform": platform.platform(),
        "package_version": mboostrs.__version__,
    }


def write_results(rows, output, n, test_n, seeds):
    grouped = {}
    for s in SCENARIOS:
        for m in ("tuned_boost", "selected_refit", "joint_reml", "double_penalty"):
            rs = [r for r in rows if r["scenario"] == s and r["method"] == m and r["status"] == "ok"]
            a = [r["signal_mse"] for r in rs]
            grouped[f"{s}/{m}"] = {
                "n": len(a),
                "mean_signal_mse": float(np.mean(a)) if a else None,
                "se_signal_mse": float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
                "median_wall_seconds": float(np.median([r["wall_seconds"] for r in rs])) if rs else None,
            }
    Path(output).write_text(
        json.dumps(
            {
                "experiment": "joint_regularization_probe",
                "n": n,
                "test_n": test_n,
                "seeds": seeds,
                "no_coverage_claims": True,
                "evidence": evidence(),
                "rows": rows,
                "summary": grouped,
            },
            indent=2,
        )
        + "\n"
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--test-n", type=int, default=2000)
    p.add_argument("--out", default="joint_regularization_probe.json")
    a = p.parse_args(argv)
    if mb.build_info()["source_digest"] != native_source_digest():
        raise RuntimeError("native source digest differs from checkout; rebuild before benchmarking")
    if a.seeds < 1 or a.n < 20 or a.test_n < 1:
        p.error("require seeds >= 1, n >= 20, test-n >= 1")
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    rows = [r for seed in seeds for r in run_seed(seed, a.n, a.test_n)]
    write_results(rows, a.out, a.n, a.test_n, seeds)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
