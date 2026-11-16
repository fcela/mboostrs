"""Repeat the documented search on independent training/test draws.

Run from the repository root with the repository environment:
  python -m benches.bench_specification_validation --repeats 12

No test outcomes enter the search. The existing agent_loop is reused rather
than replaced with a more favorable policy. The initial autospec fit and the
final search fit predict the same test rows. A full transcript is retained.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import mboostrs as mb
from benches import bench_agent_search as search
from benches.study_evidence import data_digest, evidence_document, write_evidence

SCENARIOS = ("linear_additive", "correlated", "pure_interaction", "small_n")


def run(scenario_name, repeat, budget, rounds):
    scenario = search.SCENARIOS[scenario_name]
    seed = 31000 + repeat * 101
    n = min(scenario.n, 400)
    training, _, _ = scenario.build(np.random.default_rng(seed), n,
                                    np.random.default_rng(seed + 1))
    # Independent data are generated here, but never passed to agent_loop.
    test, truth, _ = scenario.build(np.random.default_rng(seed + 2), 2000,
                                    np.random.default_rng(seed + 3))
    search.BUDGET, search.MAX_ROUNDS = budget, rounds
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = search.agent_loop(training, return_result=True)
        spec, family, mstop, groups, transcript = (
            result.spec, result.family, result.mstop, result.groups, result.transcript)
        initial = transcript.steps[0]
        initial_fit = mb.gamboost(initial["formula"], training, family=family,
                                 control=mb.boost_control(initial["mstop"], initial["nu"]))
        final_fit = result.model
        p0 = np.asarray(initial_fit.predict(newdata=test, type="response"))
        p1 = np.asarray(final_fit.predict(newdata=test, type="response"))
    if groups or type(family).__name__ != initial["family"]:
        raise ValueError("this study requires an unchanged family and ungrouped scenarios")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("nonfinite test predictions")
    initial_main, initial_pairs, _ = search._structure(initial["formula"].split("~", 1)[1])
    final_main, final_pairs, _ = search._structure(spec)
    initial_present = initial_main.union(*initial_pairs) if initial_pairs else initial_main
    final_present = final_main.union(*final_pairs) if final_pairs else final_main
    row = {
        "scenario": scenario_name, "repeat": repeat, "seed": seed,
        "n_train": n, "n_test": len(truth),
        "input_digest": data_digest({"training": training, "test": test, "truth": truth}),
        "initial_formula": initial["formula"], "final_formula": "y ~ " + spec,
        "family": type(family).__name__,
        "initial_mstop": initial["mstop"], "final_mstop": int(mstop),
        "initial_terms": len(search.split_terms(initial["formula"].split("~", 1)[1])),
        "final_terms": len(search.split_terms(spec)),
        "initial_noise": sorted(set(scenario.noise) & initial_present),
        "final_noise": sorted(set(scenario.noise) & final_present),
        "initial_signal_found": sorted(set(scenario.signal) & initial_present),
        "final_signal_found": sorted(set(scenario.signal) & final_present),
        "final_true_pairs": len(set(scenario.pairs) & final_pairs),
        "final_false_pairs": len(final_pairs - set(scenario.pairs)),
        "initial_selected": sorted(set(initial_fit.selected)),
        "final_selected": sorted(set(final_fit.selected)),
        "initial_mse": float(np.mean((test["y"] - p0)**2)),
        "final_mse": float(np.mean((test["y"] - p1)**2)),
        "oracle_mse": float(np.mean((test["y"] - truth)**2)),
        "initial_mean_error": float(np.mean((truth - p0)**2)),
        "final_mean_error": float(np.mean((truth - p1)**2)),
        "seconds": time.perf_counter() - started,
        "warnings": sorted({str(w.message) for w in caught}),
        "transcript": transcript.steps,
        "output_digest": data_digest({"initial": p0, "final": p1}),
    }
    row["mse_change"] = row["final_mse"] - row["initial_mse"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--out", default="docs/book/cases/bench_specification_validation.json")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two independent draws are required")
    protocol = {
        "scenarios": args.scenarios, "max_train_rows": 400, "test_rows": 2000,
        "autospec_budget": args.budget, "search_rounds": args.rounds,
        "budget_scope": "autospec probes only; subsequent report/search/refit calls are additional",
        "inner_folds": 5, "outer": "fresh independent training and test draws",
        "acceptance": "existing agent_loop: improvement beyond two fold SEs, else removal within two SEs",
        "baseline": "the starting autospec formula at its own tuned mstop",
        "test_access": "only after all specification and stopping choices",
        "warmup": "none; timings include first-use costs and are descriptive",
    }
    doc = evidence_document(protocol, args.repeats,
                            [__file__, search.__file__, "benches/study_evidence.py"])
    for name in args.scenarios:
        for repeat in range(args.repeats):
            try:
                row = run(name, repeat, args.budget, args.rounds)
                row["status"] = "ok"
                print(name, repeat, f"delta={row['mse_change']:+.4f}",
                      f"terms={row['initial_terms']}->{row['final_terms']}", flush=True)
            except Exception as exc:
                row = {"scenario": name, "repeat": repeat, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
                print(name, repeat, row["error"], flush=True)
            doc["results"].append(row)
            write_evidence(doc, args.out)


if __name__ == "__main__":
    main()
