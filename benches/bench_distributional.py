"""Gaussian distributional regression with shared validation data.

python -m pip install -r benches/requirements-distributional.txt
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m benches.bench_distributional --repeats 5

Three tracks. `additive`: no interaction for any method (depth-1 trees for
NGBoost). `supplied`: every method receives the product x1*x2 -- a bilinear
term for the smooth models, an extra column for NGBoost, still with depth-1
trees. `discovery`: nothing is supplied; NGBoost searches with depth-3 trees
and the smooth models, which have no interaction search in their LSS fit,
stay additive. Each library gets three configurations, the same
three inner folds and independent test data. This compares procedures, not
identical estimators; elapsed costs are not implementation speed ratios.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from scipy.stats import norm

import mboostrs as mb
from benches.study_evidence import data_digest, evidence_document, gaussian_scores, write_evidence

LIBRARIES = ("mboostrs", "mboostrs + REML refit", "mboostrs autospec pairs + REML refit",
             "gamboostLSS", "mssm", "NGBoost")
SCENARIOS = ("additive", "interaction")
TRACKS = ("additive", "supplied", "discovery")
FEATURES = ("x1", "x2", "x3", "noise")
STEPS = (100, 300, 800)


def sample(seed, n, scenario):
    rng = np.random.default_rng(seed)
    d = {name: rng.uniform(-1, 1, n) for name in FEATURES}
    mu = 1.5 * np.sin(np.pi * d["x1"]) + .6 * d["x2"]
    log_sigma = -.4 + .45 * d["x3"]
    if scenario == "interaction":
        mu += 1.2 * d["x1"] * d["x2"]
        log_sigma += .6 * d["x1"] * d["x2"]
    sigma = np.exp(log_sigma)
    d["y"] = mu + sigma * rng.normal(size=n)
    return d, mu, sigma


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def fit_predict(library, training, test, configuration, track, seed):
    """Return response-scale mean and standard deviation plus model size."""
    if library in ("mboostrs", "mboostrs + REML refit", "mboostrs autospec pairs + REML refit"):
        rhs = " + ".join(f"bbs({name}, df=4, knots=10)" for name in FEATURES)
        if track == "supplied":
            rhs += " + bols(x1, by=x2, intercept=False)"
        if library == "mboostrs autospec pairs + REML refit":
            # Pair discovery on the training rows only: autospec screens the
            # mean model's residual and keeps the pairs held-out risk
            # supports; they enter both parameters.
            rec = mb.autospec(training, "y", budget=200, seed=seed, interactions="adopt")
            pairs = list(rec.evidence.get("adopted_interactions", ()))
            if pairs:
                rhs += " + " + " + ".join(pairs)
        if library in ("mboostrs + REML refit", "mboostrs autospec pairs + REML refit"):
            # The non-cyclic fit records the risk reduction of every learner,
            # which deselection needs: terms under 1% of it are dropped, then
            # the kept terms of both parameters are refitted with REML
            # smoothing parameters.
            fit = mb.gamboostLSS("y ~ " + rhs, training, family=mb.GaussianLSS(),
                                control=mb.boost_control(2 * STEPS[configuration], .1),
                                method="noncyclic")
            kept = mb.deselect(fit, seed=seed)
            refit = mb.gamlss_refit(kept.model)
            prediction = refit.predict(test, type="response")
            size = {"selected_components": {p: len(v) for p, v in kept.variables.items()}}
            return prediction["mu"], prediction["sigma"], size
        fit = mb.gamboostLSS("y ~ " + rhs, training, family=mb.GaussianLSS(),
                            control=mb.boost_control(STEPS[configuration], .05),
                            stabilization="L2")
        prediction = fit.predict(newdata=test, type="response")
        size = {"selected_components": {p: len(set(names)) for p, names in
                                         fit.selected_which().items()}}
        return prediction["mu"], prediction["sigma"], size
    if library == "gamboostLSS":
        from rpy2 import robjects as ro

        train_r = ro.DataFrame({k: ro.FloatVector(v) for k, v in training.items()})
        test_r = ro.DataFrame({k: ro.FloatVector(v) for k, v in test.items()})
        fn = ro.r('''function(train, test, steps, interaction) {
            notes <- character()
            result <- withCallingHandlers({
            rhs <- paste(sprintf("bbs(%s, df=4, knots=10)",
                                 c("x1", "x2", "x3", "noise")), collapse=" + ")
            if (interaction) rhs <- paste(rhs,
              "+ bols(x1, by=x2, intercept=FALSE)")
            fit <- gamboostLSS::gamboostLSS(as.formula(paste("y ~", rhs)), data=train,
              families=gamboostLSS::GaussianLSS(stabilization="L2"),
              control=mboost::boost_control(mstop=steps, nu=0.05))
            p <- predict(fit, newdata=test, type="response")
            list(mu=as.numeric(p$mu), sigma=as.numeric(p$sigma))
            }, warning=function(w) {
                notes <<- c(notes, conditionMessage(w)); invokeRestart("muffleWarning")
            })
            result$notes <- unique(notes)
            result
        }''')
        result = fn(train_r, test_r, STEPS[configuration], track == "supplied")
        for note in result.rx2("notes"):
            warnings.warn("R gamboostLSS: " + str(note), RuntimeWarning, stacklevel=2)
        return np.asarray(result.rx2("mu")), np.asarray(result.rx2("sigma")), {}
    if library == "mssm":
        # mssm requires pandas frames. The book's own frames remain Polars.
        import pandas as pd
        from mssm.models import GAMMLSS, GAUMLSS, Formula, f, i, l, lhs

        train_pd, test_pd = pd.DataFrame(training), pd.DataFrame(test)
        train_pd["product"] = train_pd["x1"] * train_pd["x2"]
        test_pd["product"] = test_pd["x1"] * test_pd["x2"]
        nk = (8, 12, 16)[configuration]
        formulas = []
        for _ in range(2):
            terms = [i(), *(f([name], nk=nk, penalize_null=True) for name in FEATURES)]
            if track == "supplied":
                terms.append(l(["product"]))
            formulas.append(Formula(lhs("y"), terms, data=train_pd, print_warn=False))
        fit = GAMMLSS(formulas, GAUMLSS())
        fit.fit(progress_bar=False, n_cores=1, seed=seed)
        parameters = [np.asarray(fit.family.links[p].fi(
            fit.predict(None, test_pd, par=p)[0])).ravel() for p in range(2)]
        return *parameters, {"coefficient_count": int(np.size(fit.coef))}
    if library == "NGBoost":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from sklearn.tree import DecisionTreeRegressor

        columns = list(FEATURES)
        x = np.column_stack([training[name] for name in columns])
        xt = np.column_stack([test[name] for name in columns])
        if track == "supplied":
            # The same product the smooth models receive, as a column.
            x = np.column_stack([x, training["x1"] * training["x2"]])
            xt = np.column_stack([xt, test["x1"] * test["x2"]])
        fit = NGBRegressor(Dist=Normal, n_estimators=STEPS[configuration],
                          learning_rate=.05, random_state=seed, verbose=False,
                          Base=DecisionTreeRegressor(max_depth=3 if track == "discovery" else 1,
                                                     min_samples_leaf=10, random_state=seed))
        fit.fit(x, training["y"])
        prediction = fit.pred_dist(xt).params
        return prediction["loc"], prediction["scale"], {"boosting_rounds": len(fit.base_models)}
    raise ValueError(library)


def run(library, scenario, track, repeat):
    seed = 72000 + 101 * repeat
    training, _, _ = sample(seed, 500, scenario)
    test, true_mu, true_sigma = sample(seed + 1, 2000, scenario)
    assignment = np.random.default_rng(seed + 2).permutation(500) % 3
    inner = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for configuration in range(3):
            scores = []
            errors = []
            for fold in range(3):
                held = assignment == fold
                try:
                    mu, sigma, _ = fit_predict(library, subset(training, ~held),
                                              subset(training, held), configuration, track, seed)
                    scores.extend(gaussian_scores(training["y"][held], mu, sigma)["log_score"])
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            inner.append({"configuration": configuration,
                          "log_score": float(np.mean(scores)) if not errors else None,
                          "errors": errors})
        valid = [row for row in inner if row["log_score"] is not None]
        if not valid:
            raise RuntimeError(f"all configurations failed: {inner}")
        best = min(valid, key=lambda row: row["log_score"])["configuration"]
        mu, sigma, size = fit_predict(library, training, test, best, track, seed)
        scores = gaussian_scores(test["y"], mu, sigma)
    # Independent scipy evaluation is an oracle for the shared scoring code.
    np.testing.assert_allclose(scores["log_score"], -norm.logpdf(test["y"], mu, sigma),
                               rtol=1e-12, atol=1e-12)
    return {
        "library": library, "scenario": scenario, "track": track, "repeat": repeat,
        "seed": seed, "n_train": 500, "n_test": 2000,
        "input_digest": data_digest({"training": training, "test": test, "folds": assignment}),
        "configuration": best, "inner": inner, "size": size,
        "metrics": {key: float(np.mean(value)) for key, value in scores.items()},
        "mean_function_mse": float(np.mean((mu - true_mu)**2)),
        "log_scale_mse": float(np.mean((np.log(sigma) - np.log(true_sigma))**2)),
        "oracle_log_score": float(np.mean(gaussian_scores(test["y"], true_mu,
                                                          true_sigma)["log_score"])),
        "seconds": time.perf_counter() - started,
        "warnings": sorted({str(w.message) for w in caught}),
        "output_digest": data_digest({"mu": mu, "sigma": sigma}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--libraries", nargs="+", default=list(LIBRARIES), choices=LIBRARIES)
    parser.add_argument("--out", default="docs/book/cases/bench_distributional.json")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two independent draws are required")
    # Imports and R package initialization are outside the workflow timings.
    if "gamboostLSS" in args.libraries:
        from rpy2 import robjects as ro
        ro.r('suppressPackageStartupMessages(library(gamboostLSS))')
    if "mssm" in args.libraries:
        import mssm.models  # noqa: F401
    if "NGBoost" in args.libraries:
        import ngboost  # noqa: F401
    protocol = {
        "libraries": args.libraries, "family": "Gaussian mean and standard deviation",
        "scenarios": list(SCENARIOS), "tracks": list(TRACKS),
        "train_rows": 500, "test_rows": 2000, "inner_folds": 3, "configurations": 3,
        "boosting_mstop": list(STEPS), "learning_rate": .05, "mssm_basis_sizes": [8, 12, 16],
        "selection": "lowest pooled inner-fold negative Gaussian log density",
        "basis": "mboostrs/R use bbs; mssm uses REML with null-space penalties; bases and penalties differ",
        "interactions": ("additive: none; supplied: the product x1*x2 for every method (a term "
                         "for smooth models, a column for NGBoost with depth-1 trees); discovery: "
                         "nothing supplied, NGBoost depth-3 trees, smooth models additive"),
        "test_access": "after all configuration choices; no response clipping",
        "warmup": "dependency imports only; no repeated fixed-workload timing or speed ratio",
        "timed_boundary": "all inner fits, final refit, prediction and score calculation",
    }
    doc = evidence_document(protocol, args.repeats, [__file__, "benches/study_evidence.py"])
    if "gamboostLSS" in args.libraries:
        doc["versions"]["gamboostLSS"] = str(ro.r('as.character(packageVersion("gamboostLSS"))')[0])
    for scenario in SCENARIOS:
        for track in TRACKS:
            for repeat in range(args.repeats):
                # Rotate execution order to reduce a fixed ordering advantage.
                order = args.libraries[repeat % len(args.libraries):] + args.libraries[:repeat % len(args.libraries)]
                for library in order:
                    try:
                        row = run(library, scenario, track, repeat)
                        row["status"] = "ok"
                        print(scenario, track, repeat, library,
                              f"NLL={row['metrics']['log_score']:.4f}",
                              f"{row['seconds']:.1f}s", flush=True)
                    except Exception as exc:
                        row = {"library": library, "scenario": scenario, "track": track,
                               "repeat": repeat, "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"}
                        print(library, row["error"], flush=True)
                    doc["results"].append(row)
                    write_evidence(doc, args.out)


if __name__ == "__main__":
    main()
