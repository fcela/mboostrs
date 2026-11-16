"""Identified and deliberately unidentified causal screening panels."""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.benchmark_contract import data_digest  # noqa: E402
from benches.shootout_models import search  # noqa: E402

SCENARIOS = ["smooth_effect", "joint_effect", "weak_overlap", "omitted_confounder"]
METHODS = ["unadjusted", "mboost_rlearner", "mboost_aipw", "catboost_tlearner", "grf"]


def causal_data(scenario, seed, n=1500, n_test=2000):
    rng = np.random.default_rng(seed + 50000)
    X = rng.uniform(-1, 1, (n, 12))
    Xt = rng.uniform(-1, 1, (n_test, 12))
    def tau(x):
        if scenario == "joint_effect":
            return 1 + 1.5 * np.where(x[:, 0] * x[:, 1] > 0, 1, -1)
        return 1 + .8 * np.sin(2 * x[:, 0])
    U = rng.normal(size=n)
    score = (5 if scenario == "weak_overlap" else 1.5) * X[:, 0] + X[:, 1]
    if scenario == "omitted_confounder":
        score += 2 * U
    e = expit(score)
    t = rng.binomial(1, e)
    mu = 2 * np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + X[:, 2]
    y = mu + t * tau(X) + rng.normal(0, .5, n)
    if scenario == "omitted_confounder":
        y += 2 * U
    frame = pl.DataFrame({**{f"x{i}": X[:, i] for i in range(12)}, "y": y, "t": t})
    test = pl.DataFrame({f"x{i}": Xt[:, i] for i in range(12)})
    return frame, test, tau(Xt), e


GRF_SCRIPT = r'''
suppressMessages(library(grf))
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1]); nd <- read.csv(args[2])
xs <- grep("^x", names(d), value=TRUE)
f <- causal_forest(as.matrix(d[xs]), d$y, d$t, num.trees=2000,
                  tune.parameters="all", num.threads=1, seed=as.integer(args[3]))
p <- predict(f, as.matrix(nd[xs]), estimate.variance=TRUE)
a <- average_treatment_effect(f)
cat(jsonlite::toJSON(list(tau=as.numeric(p$predictions),
                         variance=as.numeric(p$variance.estimates),
                         ate=as.numeric(a[1]), se=as.numeric(a[2])), auto_unbox=TRUE, digits=NA))
'''


def run_job(spec):
    import mboostrs as mb
    from mboostrs.causal import CausalSpec, aipw, effect_model

    scenario, seed, method = spec
    if method == "catboost_tlearner":
        import catboost  # noqa: F401
    frame, test, true_tau, propensity = causal_data(scenario, int(seed))
    names = [f"x{i}" for i in range(12)]
    spec = CausalSpec(treatment="t", outcome="y", covariates=tuple(names),
                     target_population="independent uniform(-1, 1) baseline covariates",
                     unit="independently generated person", time_zero="before treatment",
                     follow_up="one outcome after the assigned binary treatment",
                     adjustment_rationale="baseline causes of treatment and outcome are observed; "
                     "the omitted-confounder negative control deliberately violates this")
    begin = time.perf_counter()
    info = {}
    if method == "unadjusted":
        a = frame.filter(pl.col("t") == 1)["y"].mean() - frame.filter(pl.col("t") == 0)["y"].mean()
        tau = np.full(len(test), a)
        ate, se = float(a), None
    elif method == "mboost_rlearner":
        fitted = effect_model(frame, spec, folds=3, seed=int(seed), max_mstop=1600)
        tau = np.asarray(fitted.predict(test), float)
        ate, se = float(tau.mean()), None
        info = fitted.diagnostics
        info.update(terms=fitted.terms, mstop=int(fitted.model.mstop),
                    selected_terms=list(fitted.model.coef()))
    elif method == "mboost_aipw":
        fitted = aipw(frame, spec, smooth=True, folds=3, inner_folds=3, mstop=800,
                      seed=int(seed), propensity_clip=(.02, .98))
        ate, se = float(fitted.estimate), float(fitted.standard_error)
        tau = None
        info = {"n_clipped": int(np.sum((fitted.propensity < .02) | (fitted.propensity > .98))),
                "nuisance": fitted.nuisance}
    elif method == "catboost_tlearner":
        models = []
        for arm in (0, 1):
            arm_data = frame.filter(pl.col("t") == arm).drop("t")
            order = np.random.default_rng(int(seed) + arm).permutation(len(arm_data))
            cut = int(.8 * len(order))
            fits = search("catboost", arm_data[order[:cut].tolist()], arm_data[order[cut:].tolist()],
                          [], "regression", int(seed), tier="tuned")
            models.append(fits[-1][1])
        tau = models[1].predict(test) - models[0].predict(test)
        ate, se = float(tau.mean()), None
        info = {"selected_configs": [m.config for m in models]}
    elif method == "grf":
        with tempfile.TemporaryDirectory(prefix="shootout-grf-") as tmp:
            tmp = Path(tmp)
            frame.write_csv(tmp / "train.csv")
            test.write_csv(tmp / "test.csv")
            (tmp / "fit.R").write_text(GRF_SCRIPT)
            res = subprocess.run(["Rscript", str(tmp / "fit.R"), str(tmp / "train.csv"),
                                  str(tmp / "test.csv"), str(seed)], capture_output=True,
                                 text=True, check=True, timeout=200)
            prediction = json.loads(res.stdout)
        tau = np.array(prediction["tau"])
        ate, se = float(prediction["ate"]), float(prediction["se"])
        variance = np.asarray(prediction["variance"])
        info["conditional_coverage95"] = float(np.mean(np.abs(tau - true_tau) <= 1.96 * np.sqrt(variance)))
    else:
        raise ValueError(method)
    scores = {"ate": ate, "ate_error": ate - 1., "ate_se": se,
              "ate_covered95": None if se is None else bool(abs(ate - 1) <= 1.96 * se),
              "pehe": None if tau is None else float(np.sqrt(np.mean((tau - true_tau) ** 2)))}
    return {"scenario": scenario, "seed": int(seed), "method": method, "scores": scores,
            "seconds": time.perf_counter() - begin, "details": info,
            "identified_by_design": scenario != "omitted_confounder",
            "input_digest": data_digest(frame.to_dict(as_series=False)),
            "truth_digest": data_digest(true_tau),
            "true_ate": 1., "n_train": len(frame), "n_test": len(test),
            "propensity_range": [float(propensity.min()), float(propensity.max())],
            "conditional_intervals": "native grf only" if method == "grf" else "unavailable",
            "native_build": dict(mb.build_info()),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             * (1 if sys.platform == "darwin" else 1024)),
            "child_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                                   * (1 if sys.platform == "darwin" else 1024))}


if __name__ == "__main__":
    import argparse

    from benches._jobs import run_jobs
    from benches.bench_competitive_shootout import THREADS
    from benches.shootout_data import CACHE
    from benches.study_evidence import evidence_document, write_evidence

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=7)
    args = parser.parse_args()
    jobs = []
    for i, scenario in enumerate(SCENARIOS):
        for seed in range(args.seeds):
            shift = (i + seed) % len(METHODS)
            jobs.extend((scenario, seed, m) for m in METHODS[shift:] + METHODS[:shift])
    document = evidence_document({"document": "docs/benchmarks/competitive_shootout_protocol.md",
                                  "causal_scenarios": SCENARIOS,
                                  "specs": [list(s) for s in jobs], "timeout_seconds": 240},
                                 {"independent_draws": args.seeds}, [Path(__file__)])
    document["results"] = run_jobs("bench_shootout_causal", jobs, CACHE / "causal", workers=1,
                                    timeout=240, env=THREADS)
    write_evidence(document, ROOT / "docs/benchmarks/competitive_shootout_causal.json")
