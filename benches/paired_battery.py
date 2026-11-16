"""Paired Track A / Track B estimation-discovery battery (issue #511).

Instrument: paired-estimation-discovery, version paired-ab/v1.1.
See docs/benchmarks/paired_estimation_discovery_protocol.md for the frozen
contract (information contracts, paired decomposition, panels, arms,
stopping rule, structural accounting rules, failure accounting).

Change log (published lineage; the laptop-local v1.1/v1.2 drafts were
never published and are not ancestors):
- paired-ab/v1.1: ported measurement-validity pieces from the v2 design
  (branch feat/paired-estimation-discovery-511, PR #527, not merged):
  (1) A-arm stopping grows by doubling (continue_to) until the validation
  minimum is interior, ceiling 3200 — the v1.0 fixed-cap-1000 argmin
  truncated 3/8 interaction replicates and contaminated estimation excess
  with a tuning limitation (follows the #509 stopping-ablation evidence
  that boundary growth is never materially worse); at_boundary,
  final_cap and growth_steps are recorded per replicate. The
  B_discovery procedure is untouched. (2) Four-way
  train/selection/calibration/test partitions (60/15/15/10, n=1000):
  selection rows choose stopping iterations; calibration rows are held
  out disjoint from selection, reserved for future interval calibration
  (this battery does not calibrate intervals yet); test rows are touched
  once. All arms share the identical partition per replicate (paired).
  (3) Evaluator-only supplied-basis floors per panel: noiseless fits of
  the supplied and partial formulas, separating an inadequate basis from
  estimation error within the basis. (4) New task threshold_jump: a
  discontinuous threshold the default smooth basis cannot follow — the
  first task where discovery is expected to struggle, so the instrument
  demonstrates it can detect a gap. Deliberately not ported: v2's
  Task-dataclass redesign; linear_sparse/mixed_factor_proxy (redundant
  with linear_factors); v2's frozen baseline artifacts (a fresh frozen
  run follows after this lands). v1.0's kinked and the pure
  product-interaction panels are unchanged — the interaction panel keeps
  its hard discovery case (product term with no marginal signal).

Structural accounting follows the #509 audit rules:
- tree terms are identified by btree_splits() membership, never by name
  parsing (a btree(x, z) may be named plain "x" with no marker);
- candidate columns and used columns are recorded separately per term;
- used columns for trees = union of numeric splits[*]["variable"] and
  level_split_variables (categorical split variables);
- split columns are structure, never effect proof;
- no oracle minima anywhere (test touched once, after all choices).
"""

from __future__ import annotations

import hashlib
import time
import traceback
import warnings
from collections import Counter

import numpy as np

# The test set is wider than train by design; extrapolation warnings are
# expected and would drown the logs. The predictions stand as computed.
warnings.filterwarnings("ignore", message=".*predict outside the range.*")

INSTRUMENT_VERSION = "paired-ab/v1.1"

# ---------------------------------------------------------------------------
# Design constants (frozen with the instrument)
# ---------------------------------------------------------------------------

N_TOTAL = 1000
PARTITION = {"train": 0.60, "selection": 0.15, "calibration": 0.15,
             "test": 0.10}
N_REPLICATES = 8
SEED_BASE = 511000
MSTOP_START = 1000      # initial A-arm path length
MSTOP_CEILING = 3200    # growth ceiling; at_boundary recorded if reached
NU = 0.1                # frozen step size, every arm

# t_{0.975, n-1} for the paired CI half-width; n=8 is frozen with the
# instrument, so the exact value is pinned, not approximated.
_T_0975 = {8: 2.365}


def paired_ci_halfwidth(vals):
    """Half-width of the CI for the mean of paired replicate gaps.

    This is the smallest absolute gap the design resolves at the frozen
    replicate count: a gap of interest narrower than this needs more
    replicates, not weaker claims (see the protocol's precision
    contract).
    """
    a = np.asarray(vals, dtype=float)
    n = len(a)
    if n < 2 or not np.all(np.isfinite(a)):
        return float("nan")
    t = _T_0975.get(n, 1.96)
    return float(t * np.std(a, ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Panels: DGPs + evaluator-only truth
# ---------------------------------------------------------------------------

def gen_linear_factors(seed, noiseless=False):
    """Linear effects + factor + correlated substitute + nulls."""
    rng = np.random.default_rng(seed)
    n = N_TOTAL
    X = rng.normal(size=(n, 10))
    g = rng.choice(["a", "b", "c"], size=n)
    x3 = X[:, 0] + rng.normal(scale=0.1, size=n)  # correlated substitute for x0
    X[:, 3] = x3
    noise = rng.normal(scale=0.5, size=n) * (0.0 if noiseless else 1.0)
    y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 2]
         + 1.2 * (g == "b") - 0.8 * (g == "c")
         + noise)
    cols = {f"x{i}": X[:, i] for i in range(10)}
    cols["g"] = g
    cols["y"] = y
    return cols


def gen_smooth_additive(seed, noiseless=False):
    """Smooth + linear effects + correlated substitute + nulls."""
    rng = np.random.default_rng(seed)
    n = N_TOTAL
    X = rng.uniform(-2, 2, size=(n, 8))
    x3 = X[:, 0] + rng.normal(scale=0.05, size=n)
    X[:, 3] = x3
    noise = rng.normal(scale=0.5, size=n) * (0.0 if noiseless else 1.0)
    y = (2.0 * np.sin(np.pi * X[:, 0]) + 1.5 * X[:, 1] ** 2 - X[:, 1]
         + 0.8 * X[:, 2] + noise)
    cols = {f"x{i}": X[:, i] for i in range(8)}
    cols["y"] = y
    return cols


def gen_kinked(seed, noiseless=False):
    """Hinges + smooth nuisance + nulls (mirrors datasets.kinked_threshold)."""
    rng = np.random.default_rng(seed)
    n = N_TOTAL
    X = rng.uniform(0, 1, size=(n, 7))
    noise = rng.normal(scale=0.5, size=n) * (0.0 if noiseless else 1.0)
    y = (3.0 * np.maximum(0, X[:, 0] - 0.3)
         - 2.0 * np.maximum(0, 0.7 - X[:, 1])
         + 2.0 * np.sin(np.pi * X[:, 2])
         + noise)
    cols = {f"x{i}": X[:, i] for i in range(7)}
    cols["y"] = y
    return cols


def gen_interaction(seed, noiseless=False):
    """Product interaction + smooth + linear mains + nulls
    (mirrors datasets.interaction_surface).

    The interaction is a pure product: 3.0*(x0-0.5)*(x1-0.5) has zero
    linear marginal signal in x0 or x1 (independent centered factors),
    so main-effects bases cannot capture it. This is the hard discovery
    case by construction; the port must not change it.
    """
    rng = np.random.default_rng(seed)
    n = N_TOTAL
    X = rng.uniform(0, 1, size=(n, 6))
    noise = rng.normal(scale=0.7, size=n) * (0.0 if noiseless else 1.0)
    y = (3.0 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
         + 1.5 * np.sin(np.pi * X[:, 2]) + 1.0 * X[:, 3]
         + noise)
    cols = {f"x{i}": X[:, i] for i in range(6)}
    cols["y"] = y
    return cols


def gen_threshold_jump(seed, noiseless=False):
    """Discontinuous threshold in x1 that smoothness cannot follow.

    Ported from the v2 design (threshold_jump): the supplied arm declares
    x1 threshold-like and expresses it as a shallow tree (exact for a
    step); the threshold location is NOT supplied. A continuous-kink
    variant was rejected on floor evidence — df=4 splines follow a hinge
    too well to discriminate — which is exactly the kind of decision the
    floors exist to inform.
    """
    rng = np.random.default_rng(seed)
    n = N_TOTAL
    X = rng.uniform(-1, 1, size=(n, 10))
    noise = rng.normal(scale=0.5, size=n) * (0.0 if noiseless else 1.0)
    y = (1.8 * np.sin(1.5 * X[:, 0])
         + 1.2 * (X[:, 1] > 0.2).astype(float)
         + noise)
    cols = {f"x{i}": X[:, i] for i in range(10)}
    cols["y"] = y
    return cols


PANELS = {
    "linear_factors": {
        "generate": gen_linear_factors,
        "sigma2": 0.25,
        "driver_groups": [frozenset({"x0", "x3"}), frozenset({"x1"}),
                          frozenset({"x2"}), frozenset({"g"})],
        "formula_A": "y ~ bols(x0) + bols(x1) + bols(x2) + brandom(g)",
        "candidates": [f"x{i}" for i in range(10)] + ["g"],
        "numeric_candidates": [f"x{i}" for i in range(10)],
    },
    "smooth_additive": {
        "generate": gen_smooth_additive,
        "sigma2": 0.25,
        "driver_groups": [frozenset({"x0", "x3"}), frozenset({"x1"}),
                          frozenset({"x2"})],
        "formula_A": "y ~ bbs(x0) + bbs(x1) + bols(x2)",
        "candidates": [f"x{i}" for i in range(8)],
        "numeric_candidates": [f"x{i}" for i in range(8)],
    },
    "kinked": {
        "generate": gen_kinked,
        "sigma2": 0.25,
        "driver_groups": [frozenset({"x0"}), frozenset({"x1"}),
                          frozenset({"x2"})],
        "formula_A": "y ~ hinge(x0) + hinge(x1) + bbs(x2)",
        "candidates": [f"x{i}" for i in range(7)],
        "numeric_candidates": [f"x{i}" for i in range(7)],
    },
    "interaction": {
        "generate": gen_interaction,
        "sigma2": 0.49,
        "driver_groups": [frozenset({"x0", "x1"}), frozenset({"x2"}),
                          frozenset({"x3"})],
        "joint_groups": [frozenset({"x0", "x1"})],
        "formula_A": "y ~ bbs(x0) %X% bbs(x1) + bbs(x2) + bols(x3)",
        "candidates": [f"x{i}" for i in range(6)],
        "numeric_candidates": [f"x{i}" for i in range(6)],
    },
    "threshold_jump": {
        "generate": gen_threshold_jump,
        "sigma2": 0.25,
        "driver_groups": [frozenset({"x0"}), frozenset({"x1"})],
        "formula_A": "y ~ bbs(x0) + btree(x1, maxdepth=2, selection=sse)",
        "candidates": [f"x{i}" for i in range(10)],
        "numeric_candidates": [f"x{i}" for i in range(10)],
    },
}


def _pilot_scale(panel_name, seed=511999, draws=50):
    """Frozen task scale s = Var(y) from a large pilot draw.

    Concatenates `draws` replicates of N_TOTAL rows (50k rows total),
    so s is a stable population quantity, not a single-draw estimate.
    """
    ys = [np.asarray(PANELS[panel_name]["generate"](seed + d)["y"])
          for d in range(draws)]
    return float(np.var(np.concatenate(ys)))


def partial_formula(panel_name):
    """The partial-knowledge arm's formula: all candidates smooth, no interactions."""
    spec = PANELS[panel_name]
    terms = " + ".join(f"bbs({c})" for c in spec["numeric_candidates"])
    if "g" in spec["candidates"]:
        terms += " + brandom(g)"
    return f"y ~ {terms}"


def supplied_basis_floor(panel_name, formula=None, draws=5, mstop=3000,
                         seed_base=99001):
    """Evaluator-only basis floor.

    Fits a specification (default: the panel's supplied formula) to large
    noiseless draws from the DGP (y = truth exactly) and returns the RMSE
    against the truth. This is the floor the basis itself imposes,
    separating an inadequate basis from estimation error within the
    basis. Computing it for both the supplied and the partial arm's
    formula quantifies the representation gap a priori. Evaluator-only:
    arms never see these draws and the result never enters arm scoring.
    The floor includes a small optimization shortfall (a finite-mstop
    boosting fit), not pure basis error; settings are recorded.
    """
    import mboostrs as mb
    spec = PANELS[panel_name]
    rhs = spec["formula_A"] if formula is None else formula
    frames = [spec["generate"](seed_base + offset, noiseless=True)
              for offset in range(draws)]
    big = {k: np.concatenate([np.asarray(f[k]) for f in frames])
           for k in frames[0]}
    truth = big["y"]  # noiseless: y == truth
    model = mb.gamboost(rhs, big, control=mb.boost_control(mstop, nu=0.3))
    fitted = np.asarray(model.predict(big))
    rmse = float(np.sqrt(np.mean((fitted - truth) ** 2)))
    return {"rmse": rmse, "n_rows": len(truth), "mstop": mstop, "nu": 0.3,
            "formula": rhs,
            "note": ("noiseless fit of the formula; includes a small "
                     "optimization shortfall, not pure basis error")}


# ---------------------------------------------------------------------------
# Structural accounting (the #509 audit rules)
# ---------------------------------------------------------------------------

def _term_candidate_columns(model, index):
    """Data columns a non-tree learner may draw on.

    ``model._learner_columns`` falls back to splitting the learner's
    display label for specs it doesn't special-case. A random-effect
    term labelled ``brandom(g, df=4.0)`` then reports the whole term
    string as one pseudo-column (its comma sits inside parens, so the
    label never splits) -- the frozen-baseline defect that inflated
    linear_factors FDP to 0.25 (#511). The spec object carries the real
    column (``Brandom.name``, plus ``by`` when present), so random-effect
    learners resolve through the spec; everything else keeps the
    ``_learner_columns`` contract.
    """
    specs = getattr(model, "_learners_spec", None) or []
    spec = specs[index] if index < len(specs) else None
    if type(spec).__name__ == "Brandom":
        cols = [spec.name]
        if getattr(spec, "by", None):
            cols.append(spec.by)
        return sorted(set(cols))
    return sorted(model._learner_columns(index))


def term_accounting(model):
    """Per-term candidate vs used columns for one fitted model.

    Rules (from the #509 reporting audit):
    - A term is a tree term iff its disambiguated name is a key of
      ``model.btree_splits()`` — membership, never name parsing. A
      ``btree(x, z)`` named plain ``"x"`` (no ``(btree)`` marker) is
      still identified as a tree.
    - ``candidate_columns``: every column the term may draw on
      (``spec.names`` for trees via ``_learner_columns``; the spec's
      declared column for random-effect terms, never the term string).
    - ``used_columns``: columns the term actually used. For trees, the
      union of numeric ``splits[*]["variable"]`` and
      ``level_split_variables`` (categorical split variables are
      reported there, never in ``splits``). For other learners, the
      learner's columns when selected at least once, else empty.
    - ``numeric_cutpoint_count``: ``len(splits)`` for trees — a count of
      numeric cutpoint records, not of split nodes, boosting updates,
      or categorical splits. None for non-trees.
    - Split columns are structure, never effect proof: a recorded split
      may have zero predictive effect, and this function asserts
      nothing about effects.
    """
    splits = model.btree_splits()  # name -> {variables, splits, level_split_variables}
    names = model.base_learner_names()
    counts = Counter(model._selected)
    terms = []
    for index, name in enumerate(names):
        is_tree = name in splits
        candidate = _term_candidate_columns(model, index)
        updates = int(counts.get(index, 0))
        if is_tree:
            info = splits[name]
            used = sorted(
                {s["variable"] for s in info["splits"]}
                | set(info.get("level_split_variables") or [])
            )
            n_cutpoints = len(info["splits"])
        else:
            used = candidate if updates > 0 else []
            n_cutpoints = None
        terms.append({
            "term": name,
            "is_tree": is_tree,
            "candidate_columns": candidate,
            "used_columns": used,
            "numeric_cutpoint_count": n_cutpoints,
            "updates": updates,
        })
    return terms


def structural_scores(terms, driver_groups, joint_groups=()):
    """Driver-group recall and false discovery proportion.

    - A driver group is recovered if >= 1 of its columns is used, except
      joint groups (e.g. an interaction pair), which require every
      member used.
    - FDP = fraction of touched groups outside all driver groups, where
      a touched group is a driver group intersecting the used columns,
      and each used column outside every driver group counts as its own
      singleton group.
    Returns (recall, fdp, used_columns).
    """
    used_set = {c for t in terms for c in t["used_columns"]}
    joint_set = set(joint_groups)
    recovered = 0
    for group in driver_groups:
        if group in joint_set:
            hit = group <= used_set
        else:
            hit = bool(group & used_set)
        recovered += hit
    recall = recovered / len(driver_groups) if driver_groups else float("nan")
    touched_driver = {g for g in driver_groups if g & used_set}
    touched_false = [{c} for c in used_set
                     if not any(c in g for g in driver_groups)]
    total = len(touched_driver) + len(touched_false)
    fdp = len(touched_false) / total if total else 0.0
    return {"recall": recall, "fdp": fdp,
            "used_columns": sorted(used_set),
            "n_driver_groups_recovered": recovered,
            "n_driver_groups": len(driver_groups)}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _frame_slice(cols, idx):
    return {k: np.asarray(v)[idx] for k, v in cols.items()}


def _fit_gamboost_path(formula, train, mstop_cap=MSTOP_START, nu=NU, seed=0):
    import mboostrs as mb
    model = mb.gamboost(
        formula, data=train,
        control=mb.boost_control(mstop_cap, nu=nu),
    )
    return model


def _select_mstop_validation(model, val, grid_step=10, mstop_cap=MSTOP_START,
                             mstop_ceiling=MSTOP_CEILING, grow=True):
    """mstop by validation argmin; optionally grow the path until interior.

    With grow=True (the A arms), the boosting path doubles via
    ``continue_to`` until the validation minimum is interior (below 95%
    of the cap) or the ceiling is reached. The v1.0 fixed-cap-1000 argmin
    truncated 3/8 interaction replicates and reported a tuning
    limitation as estimation excess; growth follows the #509
    stopping-ablation evidence that boundary growth is never materially
    worse. The ceiling (3200 = 3.2x the v1.0 cap) bounds the cost to at
    most two doublings; at_boundary records whether the minimum was
    still at the cap when the ceiling bound.
    With grow=False (the frozen B arm), the plain fixed-cap argmin is
    used and no growth bookkeeping is recorded.
    Returns (m_opt, info) with at_boundary, final_cap, growth_steps.
    Test is never consulted (no oracle).
    """
    cap = mstop_cap
    growth_steps = 0
    while True:
        grid = list(range(0, cap + 1, grid_step))
        # output="numpy" gives (len(grid), n); the default frame prepends a
        # "row" index column, which must not enter the argmin.
        preds = np.asarray(model.predict_over_mstop(grid, newdata=val,
                                                    output="numpy"))
        assert preds.shape == (len(grid), len(np.asarray(val["y"]))), preds.shape
        y = np.asarray(val["y"])
        mse = ((preds - y[None, :]) ** 2).mean(axis=1)
        m_opt = int(grid[int(np.argmin(mse))])
        at_boundary = m_opt >= int(0.95 * cap)
        if not grow or not at_boundary or cap >= mstop_ceiling:
            break
        cap = min(2 * cap, mstop_ceiling)
        model = model.continue_to(cap)
        growth_steps += 1
    return m_opt, {"at_boundary": at_boundary, "final_cap": cap,
                   "growth_steps": growth_steps}


def run_arm_A(formula, cols, idx_train, idx_sel, idx_test, seed=0):
    """Supplied-specification arm: fit on train, grow+validate mstop, test once."""
    import mboostrs as mb  # noqa: F401 (import keeps arm deps explicit)
    t0 = time.perf_counter()
    train = _frame_slice(cols, idx_train)
    sel = _frame_slice(cols, idx_sel)
    test = _frame_slice(cols, idx_test)
    model = _fit_gamboost_path(formula, train, seed=seed)
    m_opt, stop_info = _select_mstop_validation(model, sel)
    preds = np.asarray(model.predict(test, mstop=m_opt))
    mse = float(((preds - np.asarray(test["y"])) ** 2).mean())
    terms = term_accounting(model)
    return {"test_mse": mse, "mstop": m_opt, "terms": terms,
            "at_boundary": stop_info["at_boundary"],
            "final_cap": stop_info["final_cap"],
            "growth_steps": stop_info["growth_steps"],
            "seconds": time.perf_counter() - t0}


def run_arm_partial(panel, cols, idx_train, idx_sel, idx_test, seed=0):
    """Partial-knowledge arm: all candidates as smooth terms, no interactions."""
    return run_arm_A(partial_formula(panel), cols, idx_train, idx_sel,
                     idx_test, seed=seed)


def run_arm_B(cols, idx_train, idx_sel, idx_test, seed=0):
    """Discovery arm: frozen autospec on train only, then fit + validate + test.

    The B procedure is frozen: fixed-cap validation argmin (no growth),
    exactly as in v1.0.
    """
    import mboostrs as mb
    t0 = time.perf_counter()
    train = _frame_slice(cols, idx_train)
    sel = _frame_slice(cols, idx_sel)
    test = _frame_slice(cols, idx_test)
    rec = mb.autospec(
        train, response="y", family=mb.Gaussian(),
        budget=32, nu=0.1, interactions="adopt", residual_tree="auto",
        seed=seed,
    )
    model = rec.fit(train)
    m_opt, _ = _select_mstop_validation(model, sel, grow=False)
    preds = np.asarray(model.predict(test, mstop=m_opt))
    mse = float(((preds - np.asarray(test["y"])) ** 2).mean())
    terms = term_accounting(model)
    spec_text = ""
    try:
        spec_text = str(rec.report())[:2000]
    except Exception:
        pass
    return {"test_mse": mse, "mstop": m_opt, "terms": terms,
            "seconds": time.perf_counter() - t0, "spec": spec_text}


# ---------------------------------------------------------------------------
# Paired runner + scoring
# ---------------------------------------------------------------------------

def _partition(n, seed):
    """Four-way disjoint partition: train / selection / calibration / test.

    Selection rows choose stopping iterations; calibration rows are held
    out disjoint from selection, reserved for future interval
    calibration (this battery does not calibrate intervals yet); test
    rows are touched once, after all choices. All arms share the
    identical partition per replicate (paired).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(PARTITION["train"] * n)
    n_sel = int(PARTITION["selection"] * n)
    n_cal = int(PARTITION["calibration"] * n)
    b1, b2, b3 = n_train, n_train + n_sel, n_train + n_sel + n_cal
    return idx[:b1], idx[b1:b2], idx[b2:b3], idx[b3:]


def run_replicate(panel_name, replicate, seed_base=SEED_BASE):
    """One paired replicate: all arms on identical partitions.

    Returns a raw record; failures are captured, never raised.
    """
    spec = PANELS[panel_name]
    seed = seed_base + replicate
    record = {"panel": panel_name, "replicate": replicate, "seed": seed,
              "instrument": INSTRUMENT_VERSION, "arms": {}, "failures": {}}
    try:
        cols = spec["generate"](seed)
        n = len(cols["y"])
        idx_train, idx_sel, idx_cal, idx_test = _partition(n, seed + 1)
        record["partition_digest"] = hashlib.sha256(
            np.concatenate([idx_train, idx_sel, idx_cal, idx_test]).tobytes()
        ).hexdigest()[:16]
        record["partition_sizes"] = {
            "train": len(idx_train), "selection": len(idx_sel),
            "calibration": len(idx_cal), "test": len(idx_test)}
        # Arm A: supplied specification
        try:
            record["arms"]["A_supplied"] = run_arm_A(
                spec["formula_A"], cols, idx_train, idx_sel, idx_test, seed=seed)
        except Exception:  # noqa: BLE001 - failure accounting
            record["failures"]["A_supplied"] = traceback.format_exc(limit=5)
        # Arm P: partial knowledge
        try:
            record["arms"]["A_partial"] = run_arm_partial(
                panel_name, cols, idx_train, idx_sel, idx_test, seed=seed)
        except Exception:  # noqa: BLE001
            record["failures"]["A_partial"] = traceback.format_exc(limit=5)
        # Arm B: frozen discovery
        try:
            record["arms"]["B_discovery"] = run_arm_B(
                cols, idx_train, idx_sel, idx_test, seed=seed)
        except Exception:  # noqa: BLE001
            record["failures"]["B_discovery"] = traceback.format_exc(limit=5)
        # Structural scores for B (and A_partial for reference)
        for arm in ("B_discovery", "A_partial"):
            if arm in record["arms"]:
                record["arms"][arm]["structural"] = structural_scores(
                    record["arms"][arm]["terms"], spec["driver_groups"],
                    spec.get("joint_groups", ()))
    except Exception:  # noqa: BLE001
        record["failures"]["replicate"] = traceback.format_exc(limit=5)
    return record


def score_record(record, s):
    """Paired decomposition for one replicate.

    estimation_excess = (R_A - R*) / s
    discovery_gap    = (R_B - R_A) / s
    partial_gap      = (R_P - R_A) / s
    end_to_end       = (R_B - R*) / s
    R* = sigma2 (known); s = frozen pilot Var(y).
    """
    spec = PANELS[record["panel"]]
    r_star = spec["sigma2"]
    arms = record["arms"]
    out = {}
    if "A_supplied" in arms:
        r_a = arms["A_supplied"]["test_mse"]
        out["estimation_excess"] = (r_a - r_star) / s
        out["R_A"] = r_a
    if "B_discovery" in arms and "A_supplied" in arms:
        r_b = arms["B_discovery"]["test_mse"]
        out["discovery_gap"] = (r_b - arms["A_supplied"]["test_mse"]) / s
        out["end_to_end_excess"] = (r_b - r_star) / s
        out["R_B"] = r_b
    if "A_partial" in arms and "A_supplied" in arms:
        r_p = arms["A_partial"]["test_mse"]
        out["partial_knowledge_gap"] = (r_p - arms["A_supplied"]["test_mse"]) / s
        out["R_P"] = r_p
    out["R_star"] = r_star
    out["s"] = s
    return out


def summarize(records, scales):
    """Median + IQR of each measure across replicates, per panel.

    Also aggregates the A-arm stopping-boundary rates (mean across
    replicates): a nonzero rate means the growth ceiling bound, and the
    estimation excess on that panel mixes in a tuning limitation.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for rec in records:
        s = scales[rec["panel"]]
        sc = score_record(rec, s)
        for k, v in sc.items():
            if k.startswith(("estimation", "discovery", "partial", "end_to_end")):
                agg[rec["panel"]][k].append(v)
        for arm in ("A_supplied", "A_partial"):
            info = rec["arms"].get(arm, {})
            if "at_boundary" in info:
                agg[rec["panel"]][f"boundary_rate_{arm}"].append(
                    1.0 if info["at_boundary"] else 0.0)
                agg[rec["panel"]][f"growth_steps_{arm}"].append(
                    float(info.get("growth_steps", 0)))
    summary = {}
    for panel, measures in agg.items():
        summary[panel] = {}
        for k, vals in measures.items():
            a = np.asarray(vals)
            entry = {
                "median": float(np.median(a)),
                "q1": float(np.quantile(a, 0.25)),
                "q3": float(np.quantile(a, 0.75)),
                "n": len(a),
            }
            if k.startswith(("boundary_rate_", "growth_steps_")):
                entry["mean"] = float(np.mean(a))
            else:
                entry["paired_ci_halfwidth"] = paired_ci_halfwidth(vals)
            summary[panel][k] = entry
    return summary
