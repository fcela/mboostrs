"""The documented specification search, run against known generating processes.

Each scenario draws a training frame and a 4000-row test frame from one data
generating process. `agent_loop` then does what an agent reading the
documentation would do, with public functions only:

    classify_columns       read the frame
    autospec               measure family, shapes, step count, interaction leads;
                           rerun with groups= when the hygiene scan names a group
    term_report            what each term earns, with paired fold standard errors
    screen_interactions    joint structure in the residual, group columns by level
    suggest_specifications ranked changes
    compare_specifications score the changes on the fold design of the task;
                           accept a change that beats the current specification
                           by two paired standard errors, or else a removal
                           whose cost is within two paired standard errors:
                           when the folds cannot tell two specifications
                           apart, the smaller one is kept
    tune_mstop             the final step count, growing the grid as needed

The final specification is scored against the truth: which true main effects,
interactions and groups it carries, how many pure-noise columns it keeps (in
the formula, and among the learners the fit selects), whether its family is
defensible, and its held-out squared error on the test frame against the error
of the true mean function on the same task. For a grouped task the test frame
holds groups the fit never saw, so the true mean function excludes the group
effects, which nothing can predict for a new group.

    python benches/bench_agent_search.py --out docs/book/cases/bench_agent_search.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

import mboostrs as mb
from mboostrs.explore import split_terms

BUDGET = 250
MAX_ROUNDS = 4
TRAIN_SEED = 11
TEST_ROWS = 4000

#: Added to group codes, and prefixed to group labels, in a test frame for a
#: new-group task, so its groups are levels the fit never saw.
FRESH_GROUPS = {"offset": 0}


# ─── scenarios ───────────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    description: str
    build: object                    # (rng, n, group_rng) -> (frame, mean, group part)
    n: int = 1000
    signal: tuple = ()               # columns with a true main effect
    pairs: tuple = ()                # frozensets of interacting columns
    groups: tuple = ()               # grouping columns
    noise: tuple = ()                # columns with no effect
    family: str = "Gaussian"         # the loss that matches the process
    also_ok: tuple = ()              # other defensible losses
    new_groups: bool = True          # the task is prediction for new groups
    notes: str = ""


def _normal(rng, n, names):
    return {name: rng.normal(size=n) for name in names}


def _codes(rng, n, k):
    return rng.integers(0, k, n).astype(float) + FRESH_GROUPS["offset"]


def _index(codes):
    return codes.astype(int) - FRESH_GROUPS["offset"]


def s_notebook(rng, n, grng):
    f = _normal(rng, n, [f"x{i}" for i in range(1, 9)])
    f["g"] = _codes(rng, n, 40)
    b = grng.normal(size=40)[_index(f["g"])]
    mean = 2 * np.sin(2 * f["x1"]) + 1.5 * (f["x2"] > 0) + 0.8 * f["x1"] * f["x4"]
    f["y"] = mean + b + rng.normal(scale=0.8, size=n)
    return f, mean, b


def s_negbin_sites(rng, n, grng):
    f = {k: rng.uniform(-2, 2, size=n) for k in "abcde"}
    f["site"] = _codes(rng, n, 25)
    u = grng.normal(scale=0.5, size=25)[_index(f["site"])]
    eta = 0.5 + 0.6 * np.log1p(np.exp(2 * f["a"])) - 0.4 * f["b"] ** 2
    mu = np.exp(np.clip(eta + u, -5, 4))
    f["y"] = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    return f, np.exp(eta), None


def s_varying_binary(rng, n, grng):
    f = _normal(rng, n, ["x", "w", "z1", "z2"])
    f["grp"] = rng.integers(0, 2, n).astype(float)
    mean = np.where(f["grp"] == 1, 1.5 * f["x"], -0.5 * f["x"]) + np.cos(f["w"])
    f["y"] = mean + rng.normal(size=n) * np.exp(0.2 + 0.5 * f["w"])
    return f, mean, None


def s_random_slopes(rng, n, grng):
    f = _normal(rng, n, ["x", "z", "noise1", "noise2"])
    f["clinic"] = _codes(rng, n, 30)
    index = _index(f["clinic"])
    slopes = grng.normal(size=30)[index]
    intercepts = grng.normal(scale=0.7, size=30)[index]
    mean = f["x"] + np.sin(f["z"])
    f["y"] = mean + slopes * f["x"] + intercepts + rng.normal(scale=0.5, size=n)
    return f, mean, slopes * f["x"] + intercepts


def s_many_noise(rng, n, grng):
    names = [f"v{i:02d}" for i in range(30)]
    f = _normal(rng, n, names)
    mean = 1.2 * f["v00"] + np.tanh(2 * f["v01"]) + 0.6 * f["v02"] ** 2
    f["y"] = mean + rng.normal(size=n)
    return f, mean, None


def s_correlated(rng, n, grng):
    x1 = rng.normal(size=n)
    f = {"x1": x1, "x1_proxy": x1 + rng.normal(scale=0.15, size=n),
         "x3": rng.normal(size=n), "x4": rng.normal(size=n)}
    mean = np.exp(0.5 * f["x1"]) + 0.5 * f["x3"]
    f["y"] = mean + rng.normal(scale=0.7, size=n)
    return f, mean, None


def s_binary_logit(rng, n, grng):
    f = _normal(rng, n, ["age", "dose", "bmi", "junk1", "junk2"])
    p = 1 / (1 + np.exp(-(-0.5 + 1.2 * np.tanh(f["dose"]) + 0.8 * f["age"] * f["bmi"])))
    f["y"] = rng.binomial(1, p).astype(float)
    return f, p, None


def s_monotone(rng, n, grng):
    f = {"dose": rng.uniform(0, 10, size=n), "weight": rng.normal(size=n),
         "noise": rng.normal(size=n)}
    mean = 3 * (1 - np.exp(-0.6 * f["dose"])) + 0.4 * f["weight"]
    f["y"] = mean + rng.normal(scale=0.6, size=n)
    return f, mean, None


def s_string_groups(rng, n, grng):
    f = {"x": rng.normal(size=n), "day": rng.uniform(0, 365, size=n),
         "noise": rng.normal(size=n)}
    codes = rng.integers(0, 20, n)
    prefix = "new_" if FRESH_GROUPS["offset"] else ""
    f["store"] = np.array([f"{prefix}store_{c:02d}" for c in range(20)])[codes]
    u = grng.normal(scale=0.8, size=20)[codes]
    mean = 0.5 * f["x"] + 0.004 * f["day"] + 0.5 * np.sin(2 * np.pi * f["day"] / 365)
    f["y"] = mean + u + rng.normal(scale=0.5, size=n)
    return f, mean, u


def s_heavy_tails(rng, n, grng):
    f = _normal(rng, n, ["x", "z", "noise"])
    mean = 2 * np.sin(f["x"]) + 0.7 * f["z"]
    f["y"] = mean + rng.standard_t(2, size=n) * 0.6
    return f, mean, None


def s_small_n(rng, n, grng):
    f = _normal(rng, n, ["a", "b", "c", "d", "e", "f", "g"])
    mean = 1.5 * f["a"] + np.sin(2 * f["b"])
    f["y"] = mean + rng.normal(scale=0.8, size=n)
    return f, mean, None


def s_linear_additive(rng, n, grng):
    f = _normal(rng, n, ["p", "q", "r", "s", "t"])
    mean = f["p"] - 0.8 * f["q"] + 0.5 * f["r"]
    f["y"] = mean + rng.normal(size=n)
    return f, mean, None


def s_pure_interaction(rng, n, grng):
    f = {"u": rng.uniform(-2, 2, size=n), "v": rng.uniform(-2, 2, size=n),
         "w": rng.normal(size=n), "k": rng.normal(size=n)}
    mean = 1.5 * np.tanh(2 * f["u"]) * np.tanh(2 * f["v"]) + 0.5 * f["w"]
    f["y"] = mean + rng.normal(scale=0.5, size=n)
    return f, mean, None


def s_group_covariate(rng, n, grng):
    f = {"x": rng.normal(size=n), "noise": rng.normal(size=n)}
    codes = rng.integers(0, 35, n)
    f["region"] = codes.astype(float) + FRESH_GROUPS["offset"]
    f["income"] = grng.normal(size=35)[codes]
    u = grng.normal(scale=0.5, size=35)[codes]
    mean = 0.8 * f["x"] + f["income"]
    f["y"] = mean + u + rng.normal(scale=0.6, size=n)
    return f, mean, u


def s_missing(rng, n, grng):
    f = _normal(rng, n, ["x", "z", "noise"])
    mean = np.sin(2 * f["x"]) + 0.8 * f["z"]
    f["y"] = mean + rng.normal(scale=0.5, size=n)
    f["z"] = np.where(rng.uniform(size=n) < 0.12, np.nan, f["z"])
    return f, mean, None


def s_exposure(rng, n, grng):
    f = {"x": rng.normal(size=n), "hours": rng.uniform(1, 40, size=n),
         "noise": rng.normal(size=n)}
    mu = np.exp(-1.0 + 0.6 * f["x"]) * f["hours"]
    f["y"] = rng.poisson(mu).astype(float)
    return f, mu, None


def s_zero_inflated(rng, n, grng):
    f = _normal(rng, n, ["x", "z", "noise"])
    mu = np.exp(0.8 + 0.7 * f["x"])
    zero = 1 / (1 + np.exp(-(-0.5 + 1.2 * f["z"])))
    f["y"] = np.where(rng.uniform(size=n) < zero, 0.0, rng.poisson(mu)).astype(float)
    return f, (1 - zero) * mu, None


def s_gamma(rng, n, grng):
    f = _normal(rng, n, ["x", "z", "noise"])
    mu = np.exp(1.0 + 0.5 * f["x"] + 0.3 * np.sin(2 * f["z"]))
    f["y"] = rng.gamma(3.0, mu / 3.0)
    return f, mu, None


def s_age_integer(rng, n, grng):
    f = {"age": rng.integers(18, 80, n).astype(float), "x": rng.normal(size=n),
         "noise": rng.normal(size=n)}
    mean = 0.002 * (f["age"] - 50) ** 2 + 0.03 * f["age"] + 0.7 * f["x"]
    f["y"] = mean + rng.normal(scale=0.5, size=n)
    return f, mean, None


def s_narrow_positive(rng, n, grng):
    f = _normal(rng, n, ["x", "z", "noise"])
    mean = 76.0 + f["x"] + 0.5 * np.sin(2 * f["z"])
    f["y"] = mean + rng.normal(scale=0.8, size=n)
    return f, mean, None


SCENARIOS = {s.name: s for s in [
    Scenario("notebook", "sine, threshold and a product interaction, 40 group intercepts",
             s_notebook, n=1200, signal=("x1", "x2"), pairs=(frozenset({"x1", "x4"}),),
             groups=("g",), noise=("x3", "x5", "x6", "x7", "x8")),
    Scenario("negbin_sites", "overdispersed counts across 25 sites", s_negbin_sites,
             n=1500, signal=("a", "b"), groups=("site",), noise=("c", "d", "e"),
             family="NBinomial"),
    Scenario("varying_binary", "a slope that changes sign with a binary column; spread grows with w",
             s_varying_binary, n=1200, signal=("x", "w"), pairs=(frozenset({"x", "grp"}),),
             noise=("z1", "z2"), new_groups=False),
    Scenario("random_slopes", "30 clinics with random intercepts and random slopes",
             s_random_slopes, n=1500, signal=("x", "z"), pairs=(frozenset({"x", "clinic"}),),
             groups=("clinic",), noise=("noise1", "noise2"),
             notes="a new clinic's slope cannot be predicted, so grouped folds cannot credit the slope term"),
    Scenario("many_noise", "3 signals among 30 columns", s_many_noise, n=600,
             signal=("v00", "v01", "v02"), noise=tuple(f"v{i:02d}" for i in range(3, 30)),
             new_groups=False),
    Scenario("correlated", "a signal and its near-copy", s_correlated, n=800,
             signal=("x1", "x3"), noise=("x4",), new_groups=False,
             notes="x1_proxy is neither signal nor pure noise"),
    Scenario("binary_logit", "a logistic response with a product interaction", s_binary_logit,
             n=2000, signal=("dose",), pairs=(frozenset({"age", "bmi"}),),
             noise=("junk1", "junk2"), family="Binomial", new_groups=False),
    Scenario("monotone", "a saturating dose response", s_monotone, n=600,
             signal=("dose", "weight"), noise=("noise",), new_groups=False),
    Scenario("string_groups", "20 stores as text labels, a trend and a season", s_string_groups,
             n=1200, signal=("x", "day"), groups=("store",), noise=("noise",)),
    Scenario("heavy_tails", "t-distributed noise with two degrees of freedom", s_heavy_tails,
             n=800, signal=("x", "z"), noise=("noise",), new_groups=False),
    Scenario("small_n", "150 rows, two signals, five noise columns", s_small_n, n=150,
             signal=("a", "b"), noise=("c", "d", "e", "f", "g"), new_groups=False),
    Scenario("linear_additive", "three straight lines, two noise columns", s_linear_additive,
             n=800, signal=("p", "q", "r"), noise=("s", "t"), new_groups=False),
    Scenario("pure_interaction", "a product of two saturating curves with no main effects",
             s_pure_interaction, n=1000, signal=("w",), pairs=(frozenset({"u", "v"}),),
             noise=("k",), new_groups=False),
    Scenario("group_covariate", "a group-level covariate and group intercepts", s_group_covariate,
             n=1400, signal=("x", "income"), groups=("region",), noise=("noise",)),
    Scenario("missing", "12% of a signal column missing", s_missing, n=800,
             signal=("x", "z"), noise=("noise",), new_groups=False),
    Scenario("exposure", "Poisson counts over unequal exposure hours", s_exposure, n=1000,
             signal=("x", "hours"), noise=("noise",), family="Poisson", new_groups=False,
             notes="the process is an offset log(hours), which mboostrs cannot fit as an offset"),
    Scenario("zero_inflated", "structural zeros driven by z", s_zero_inflated, n=1200,
             signal=("x", "z"), noise=("noise",), family="Poisson", also_ok=("NBinomial",),
             new_groups=False),
    Scenario("gamma", "a positive skewed response with a log-linear mean", s_gamma, n=900,
             signal=("x", "z"), noise=("noise",), family="GammaReg", also_ok=("Gaussian",),
             new_groups=False),
    Scenario("age_integer", "a whole-number age with a curved effect", s_age_integer, n=1200,
             signal=("age", "x"), noise=("noise",), new_groups=False),
    Scenario("narrow_positive", "a positive response that varies by 1% of its level",
             s_narrow_positive, n=800, signal=("x", "z"), noise=("noise",), new_groups=False),
]}


# ─── the agent loop ──────────────────────────────────────────────────────


@dataclass
class Transcript:
    steps: list = field(default_factory=list)

    def log(self, step, **details):
        self.steps.append({"step": step, **details})


def _numeric(values):
    return np.asarray(values).dtype.kind in "fiub"


def _complete(frame):
    keep = np.all([np.isfinite(np.asarray(v, dtype=float)) for v in frame.values()
                   if _numeric(v)], axis=0)
    return {c: np.asarray(v)[keep] for c, v in frame.items()}


def _folds(frame, groups):
    if groups:
        return mb.grouped_folds(np.asarray(frame[groups[0]], dtype=object), B=5, seed=0)
    return 5


def agent_loop(frame, response="y", transcript=None, *, return_result=False):
    """The documented search, now the library's :func:`mboostrs.modeler_loop`."""
    result = mb.modeler_loop(frame, response, transcript=transcript,
                             budget=BUDGET, max_rounds=MAX_ROUNDS)
    if return_result:
        return result
    return (result.spec, result.family, result.mstop, result.groups,
            result.transcript)


# ─── scoring ─────────────────────────────────────────────────────────────

_CALL = re.compile(r"(\w+)\(([^()]*)\)")


def _structure(spec):
    """Main-effect columns, column pairs, and group columns of a specification."""
    main, pairs, grouped = set(), set(), set()
    for term in split_terms(spec):
        names = []
        for learner, args in _CALL.findall(term):
            parts = [a.strip() for a in args.split(",")]
            positional = [p for p in parts if p and "=" not in p]
            by = [p.split("=", 1)[1].strip() for p in parts if p.replace(" ", "").startswith("by=")]
            names.extend(positional + by)
            if learner == "brandom":
                grouped.update(positional)
        names = list(dict.fromkeys(names))
        if len(names) >= 2:
            pairs.add(frozenset(names[:2]))
        elif names:
            main.add(names[0])
    return main, pairs, grouped


def _mentions(name, learners):
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    return any(pattern.search(learner) for learner in learners)


def run(name):
    scenario = SCENARIOS[name]
    frame, _, _ = scenario.build(np.random.default_rng(TRAIN_SEED), scenario.n,
                                 np.random.default_rng(TRAIN_SEED + 1000))
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = agent_loop(frame, return_result=True)
        spec, family, mstop, t = (
            result.spec, result.family, result.mstop, result.transcript)
        fit = result.model
        selected = set(fit.selected)

        FRESH_GROUPS["offset"] = 1000 if scenario.new_groups else 0
        try:
            group_rng = np.random.default_rng(TRAIN_SEED + (2000 if scenario.new_groups else 1000))
            test, mean, group_part = scenario.build(np.random.default_rng(TRAIN_SEED + 1),
                                                    TEST_ROWS, group_rng)
        finally:
            FRESH_GROUPS["offset"] = 0
        keep = np.all([np.isfinite(np.asarray(v, dtype=float)) for c, v in test.items()
                       if _numeric(v) and c != "y"], axis=0)
        test = {c: np.asarray(v)[keep] for c, v in test.items()}
        truth = np.asarray(mean)[keep]
        if not scenario.new_groups and group_part is not None:
            truth = truth + np.asarray(group_part)[keep]
        predicted = np.asarray(fit.predict(newdata=test, type="response",
                                           unseen_levels="average"), dtype=float).ravel()
        y = np.asarray(test["y"], dtype=float)
        mse = float(np.mean((y - predicted) ** 2))
        floor = float(np.mean((y - truth) ** 2))

    main, pairs, grouped = _structure(spec)
    present = main.union(*pairs) if pairs else set(main)
    return {
        "scenario": name,
        "description": scenario.description,
        "notes": scenario.notes,
        "rows": scenario.n,
        "final": spec,
        "family": type(family).__name__,
        "family_ok": type(family).__name__ in (scenario.family, *scenario.also_ok),
        "mstop": int(mstop),
        "signal": list(scenario.signal),
        "signal_found": sorted(v for v in scenario.signal if v in present),
        "pairs": [sorted(p) for p in scenario.pairs],
        "pairs_found": [sorted(p) for p in scenario.pairs if p in pairs],
        "extra_pairs": [sorted(p) for p in pairs if p not in scenario.pairs],
        "groups": list(scenario.groups),
        "groups_found": sorted(g for g in scenario.groups if g in grouped),
        "noise": len(scenario.noise),
        "noise_in_formula": sorted(v for v in scenario.noise if v in present),
        "noise_selected": sorted(v for v in scenario.noise if _mentions(v, selected)),
        "test_mse": mse,
        "floor_mse": floor,
        "excess": mse / floor - 1.0,
        "seconds": round(time.time() - started, 1),
        "steps": t.steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/book/cases/bench_agent_search.json")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.scenarios))) as pool:
        results = list(pool.map(run, args.scenarios))
    for r in results:
        print(f"{r['scenario']:18s} excess {r['excess']:+.3f}  signal {len(r['signal_found'])}/"
              f"{len(r['signal'])}  pairs {len(r['pairs_found'])}/{len(r['pairs'])}  groups "
              f"{len(r['groups_found'])}/{len(r['groups'])}  noise kept {len(r['noise_in_formula'])}/"
              f"{r['noise']}  family {r['family']}  {r['seconds']}s", flush=True)
    try:
        from benchmark_contract import machine_metadata
    except ImportError:
        from benches.benchmark_contract import machine_metadata
    metadata = machine_metadata()
    protocol = {"budget": BUDGET, "max_rounds": MAX_ROUNDS, "train_seed": TRAIN_SEED,
                "test_rows": TEST_ROWS,
                "acceptance": "a change that lowers the out-of-fold risk by more than two "
                              "paired fold standard errors; otherwise a removal whose cost "
                              "is within two standard errors"}
    document = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "commit": metadata["commit"],
        "working_tree": metadata["working_tree"],
        "platform": metadata["platform"],
        "versions": metadata["versions"],
        "threads": metadata["threads"],
        "protocol": protocol,
        # One training draw and one test draw per process: the table is twenty
        # cases, not an error rate.
        "repeats": 1,
        "uncertainty": {
            "per_change": "each accepted change records the paired fold standard error "
                          "of its difference in risk (results[].steps[].se)",
            "across_draws": "not measured: one training draw per process",
        },
        "results": results,
    }
    with open(args.out, "w") as handle:
        json.dump(document, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
