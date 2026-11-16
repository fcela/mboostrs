"""The automatic policy layer: what to fit, under what metric, at what cost.

:func:`~mboostrs.autospec` measures candidate terms on the training folds
and reports what it measured, but it does not decide the *shape of the
answer*: which rungs of the ladder to climb, when a more complex
specification has paid for its extra terms, and what the whole procedure
is allowed to spend. :func:`auto_policy` makes those decisions explicit and
returns the decision trail with the model.

The ladder is linear baseline -> adaptive shapes -> selected pairs ->
residual tree. A strong simple candidate is priced first and kept: a rung is
adopted only when the paired per-fold risk reduction clears
``threshold_se`` standard errors of its own paired difference (the
two-fold-SE rule of the specification-search literature). That rule is a
search heuristic, not a hypothesis test -- it measures how consistently the
folds agree that the complexity pays, and it has no nominal level. The
guardrail it must be able to reproduce is the specification-search case
where an additive model lost 0.5% to a richer one and the policy kept the
additive model: a gain of that size does not clear two standard errors
unless the folds agree closely.

Costs are accounted in :class:`CostBudget` and :class:`Spent`, and repeated
specifications are served from :class:`FitCache`, whose keys carry the
specification, the data, the fold assignment, the weights, the transform and
the build identity -- so a cache hit can only be the same computation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace

import numpy as np

from mboostrs._frame import column_names, read_column
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._output import export_result, json_safe
from mboostrs._validate import check_int_at_least, check_positive
from mboostrs.explore import split_terms
from mboostrs.formula import term_variables
from mboostrs.terms import classify_columns

__all__ = [
    "CostBudget",
    "FitCache",
    "PairedTest",
    "PolicyResult",
    "Spent",
    "auto_policy",
]


#: The rungs, cheapest first. Rejection retains the incumbent; subsequent
#: stages can test different structure without inheriting rejected terms.
_LADDER = ("linear", "additive", "pairs", "hybrid")

#: ``threshold_se`` per complexity preference. "rich" asks for less evidence
#: per rung; "simple" never climbs past the shapes.
_COMPLEXITY = {
    "simple": (2.0, frozenset({"additive"})),
    "balanced": (2.0, frozenset({"additive", "pairs", "hybrid"})),
    "rich": (1.0, frozenset({"additive", "pairs", "hybrid"})),
}

#: The profile label each rung produces. A result that depends on the
#: residual tree is a hybrid-predictor profile, which is a different claim
#: from the compact-effect profile a smooth-only fit makes.
_PROFILE = {
    "linear": "linear",
    "additive": "additive",
    "pairs": "tensor",
    "hybrid": "hybrid",
}


@dataclass(frozen=True)
class CostBudget:
    """Limits on a procedure's fit count, run time and memory. ``None`` means no limit on that field.

    A budget covers everything the procedure does -- probes, fold fits,
    grid sweeps, the final fit -- not one stage of it. Counting model fits
    is the portable measure; the seconds and memory ceilings are checked
    between stages, since neither is observable from inside a fit.
    """

    seconds: float | None = None
    fits: int | None = None
    memory_mb: float | None = None

    def __post_init__(self):
        if self.seconds is not None:
            object.__setattr__(self, "seconds", check_positive("seconds", self.seconds))
        if self.fits is not None:
            object.__setattr__(self, "fits", check_int_at_least("fits", self.fits, 1))
        if self.memory_mb is not None:
            object.__setattr__(self, "memory_mb",
                               check_positive("memory_mb", self.memory_mb))

    def to_dict(self, *, include_arrays=False):
        """The limits, as a JSON-safe record."""
        return {
            "seconds": None if self.seconds is None else float(self.seconds),
            "fits": None if self.fits is None else int(self.fits),
            "memory_mb": None if self.memory_mb is None else float(self.memory_mb),
        }


@dataclass
class Spent:
    """Fit, time and memory charges recorded during a procedure."""

    seconds: float = 0.0
    fits: int = 0
    memory_mb: float | None = None
    budget: CostBudget | None = None
    #: One line per stage that recorded its cost, for progress reporting.
    stages: list = field(default_factory=list)

    def record(self, label, *, seconds=0.0, fits=0, memory_mb=None):
        """Add one stage's cost, and return ``self`` for chaining."""
        self.seconds += float(seconds)
        self.fits += int(fits)
        if memory_mb is not None:
            self.memory_mb = (max(self.memory_mb or 0.0, float(memory_mb)))
        self.stages.append({"stage": label, "seconds": round(self.seconds, 3),
                            "fits": self.fits})
        return self

    def exceeded(self):
        """The name of the limit passed, or ``None`` while inside budget."""
        if self.budget is None:
            return None
        if self.budget.fits is not None and self.fits >= self.budget.fits:
            return "fits"
        if self.budget.seconds is not None and self.seconds >= self.budget.seconds:
            return "seconds"
        if (self.budget.memory_mb is not None and self.memory_mb is not None
                and self.memory_mb >= self.budget.memory_mb):
            return "memory_mb"
        return None

    def remaining(self):
        """Remaining allowance for each budget limit; ``None`` where unlimited."""
        if self.budget is None:
            return {"seconds": None, "fits": None, "memory_mb": None}
        return {
            "seconds": (None if self.budget.seconds is None
                        else max(self.budget.seconds - self.seconds, 0.0)),
            "fits": (None if self.budget.fits is None
                     else max(self.budget.fits - self.fits, 0)),
            "memory_mb": (None if self.budget.memory_mb is None
                          else max(self.budget.memory_mb - (self.memory_mb or 0.0), 0.0)),
        }

    def to_dict(self, *, include_arrays=False):
        """Spent against budget, with the stage log."""
        return {
            "budget": None if self.budget is None else self.budget.to_dict(),
            "spent_seconds": round(self.seconds, 3),
            "spent_fits": self.fits,
            "peak_memory_mb": self.memory_mb,
            "remaining": self.remaining(),
            "exceeded": self.exceeded(),
            "stages": list(self.stages),
        }


def _cache_value(value):
    """Encode supported configuration by value; refuse opaque mutable state."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return [type(value).__name__, value]
    if isinstance(value, np.generic):
        return _cache_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            payload = _cache_value(value.tolist())
        else:
            payload = hashlib.blake2b(value.tobytes(), digest_size=32).hexdigest()
        return ["array", value.dtype.str, list(value.shape), payload]
    if isinstance(value, Mapping):
        pairs = [(_cache_value(k), _cache_value(v)) for k, v in value.items()]
        return ["mapping", sorted(pairs, key=lambda pair: json.dumps(pair[0]))]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_cache_value(v) for v in value]]
    cls = type(value)
    kind = f"{cls.__module__}.{cls.__qualname__}"
    if is_dataclass(value) and not isinstance(value, type):
        return [kind, _cache_value({f.name: getattr(value, f.name) for f in fields(value)})]
    if cls.__module__ == "mboostrs.families":
        return [kind, _cache_value(vars(value))]
    raise TypeError(f"cache cannot fingerprint {kind}")


def _fingerprint(value):
    encoded = json.dumps(_cache_value(value), separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=32).hexdigest()


def _data_fingerprint(data):
    """A content hash of the frame a computation ran on.

    Column names, dtypes and every value, so two frames share a fingerprint
    only when they are the same data. Cost is one pass over the values.
    """
    columns = []
    for name in column_names(data):
        column = read_column(data, name)
        columns.append((name, column.kind, column.origin, column.unit,
                        column.ordered, column.levels, column.values))
    return _fingerprint(columns)


class FitCache:
    """Memo for evaluated specifications within a Python process.

    The key carries the right-hand side, a content hash of the data, the
    family and step count, the fold assignment, the weights, the transform
    and the build identity. Two calls that differ in any of these are
    different computations and cannot share an entry; a hit therefore
    returns the value the same computation produced. Opaque configuration
    (such as a callable whose closure cannot be fingerprinted) bypasses the
    cache. Prepared learners are not shared between distinct specifications.
    """

    def __init__(self, *, build=None, enabled=True):
        self.enabled = bool(enabled)
        self._entries = {}
        self.hits = 0
        self.misses = 0
        from mboostrs import build_info

        self.build = build if build is not None else build_info()["build_id"]

    def key(self, *, rhs, data, family, mstop, folds, weights=None, transform=None,
            response=None, seed=0, metric="mse", na_action="error",
            unseen_levels="error", fit_kwargs=None, score_parameter=None):
        """Key the full computation, or return ``None`` to bypass caching."""
        try:
            return _fingerprint({
                "rhs": rhs, "response": response, "data": _data_fingerprint(data),
                "family": family, "mstop": int(mstop), "folds": folds,
                "seed": seed, "metric": metric, "weights": weights,
                "transform": transform, "na_action": na_action,
                "unseen_levels": unseen_levels, "fit_kwargs": fit_kwargs,
                "build": self.build,
                "score_parameter": score_parameter,
            })
        except TypeError:
            return None

    def get_or_run(self, key, compute):
        """Return the cached value for `key`, computing it on a miss."""
        if not self.enabled or key is None:
            self.misses += 1
            return compute()
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        value = compute()
        self._entries[key] = value
        return value

    def clear(self):
        """Drop every entry and reset the counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def to_dict(self, *, include_arrays=False):
        """Cache size and hit/miss counts, as a JSON-safe record."""
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses, "build": self.build, "enabled": self.enabled}


@dataclass(frozen=True)
class PairedTest:
    """One rung's decision: the paired fold comparison behind an adoption."""

    rung: str
    baseline_rhs: str
    candidate_rhs: str
    baseline: float
    candidate: float
    gain: float
    se: float
    t: float
    threshold_se: float
    passed: bool
    worse_folds: int
    n_folds: int
    adopted: bool
    reason: str

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the decision."""
        record = export_result(
            self,
            include_arrays=include_arrays,
            extras={
                "gain": self.gain,
                "gain_percent": (100.0 * self.gain / self.baseline
                                 if self.baseline else float("nan")),
                "passed": self.passed,
            },
        )
        # The initial candidate has no scored predecessor, and a zero-SE
        # gain has no finite standardized ratio. JSON null preserves that
        # distinction without emitting nonstandard NaN/Infinity tokens.
        return {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in record.items()}

    def __str__(self):  # pragma: no cover - display only
        verdict = "adopt" if self.adopted else "keep"
        return (f"{self.rung}: {verdict} ({self.baseline:.6g} -> {self.candidate:.6g}, "
                f"gain {self.gain:+.4g} = {self.t:.2f} SE, rule {self.threshold_se:g} SE)")


def _paired_test(rung, baseline_rhs, candidate_rhs, before, after, *, partitions,
                 threshold_se, adopted, reason):
    from mboostrs import _rust

    record = _rust.paired_gain(np.asarray(before, dtype=float),
                               np.asarray(after, dtype=float),
                               int(partitions), float(threshold_se))
    return PairedTest(
        rung=rung, baseline_rhs=baseline_rhs, candidate_rhs=candidate_rhs,
        baseline=float(record["baseline"]), candidate=float(record["candidate"]),
        gain=float(record["gain"]), se=float(record["se"]), t=float(record["t"]),
        threshold_se=float(threshold_se), passed=bool(record["passed"]),
        worse_folds=int(record["worse_folds"]), n_folds=int(record["n_folds"]),
        adopted=bool(adopted), reason=reason)


def _rhs_of(formula):
    """The right-hand side of a formula that may carry a response.

    :func:`~mboostrs.autospec` returns a full formula while the fold and
    tuning entry points take a right-hand side, so the response named by the
    specification is dropped here; the response used for every fit is the one
    the caller passed to :func:`auto_policy`.
    """
    if "~" in formula:
        return partition_tilde(formula)[2].strip()
    return formula.strip()


def _linear_rhs(data, response, groups=()):
    """The strong simple candidate: a linear term per measurement, a group
    effect per category."""
    terms = []
    held = set(groups)
    for kind in classify_columns(data, response=response):
        if kind.name in held or kind.kind in ("categorical", "label"):
            terms.append(f"brandom({quote_name(kind.name)})")
        else:
            terms.append(f"bols({quote_name(kind.name)})")
    return " + ".join(terms) if terms else "1"


def _rung_rhs(rung, current, screened, *, leads=None, tree=None):
    """Extend the accepted model only with the current rung's structure.

    A discovery rerun can choose different main effects or remove a pair.
    Those edits must not enter a comparison labeled as adding pairs or a
    residual tree. Each combined proposal is still tuned and measured by
    the policy on its shared folds before adoption.
    """
    proposed = _rhs_of(screened.formula)
    if rung == "additive":
        return proposed
    terms = split_terms(current)
    evidence = getattr(screened, "evidence", {})
    if rung == "pairs":
        if leads is not None:
            additions = [lead.term for lead in leads
                         if lead.term and lead.kind != "duplicate-axis"]
        elif "interactions" in evidence:
            # Leads are nominations, including varying coefficients and
            # random slopes. Family-loss adoption inside autospec must not
            # veto testing them under the policy's requested metric.
            additions = [lead.term for lead in evidence["interactions"]
                         if lead.term and lead.kind != "duplicate-axis"]
        else:
            additions = [term for term in split_terms(proposed)
                         if "%X%" in term or re.search(r"\bby\s*=", term)]
    elif rung == "hybrid":
        # A completed family-loss comparison nominates a tree even if it
        # rejected it. The policy measures the combined candidate itself.
        # Failed sweeps and univariate shape trees do not nominate a tree.
        residual = evidence.get("residual_tree", {})
        tree = tree or evidence.get("adopted_tree")
        if tree is None and "mstop" in residual:
            tree = residual.get("term")
        additions = [tree] if tree else []
    else:
        raise ValueError(f"unknown policy rung: {rung!r}")
    seen = {"".join(term.split()) for term in terms}
    for term in additions:
        key = "".join(term.split())
        if key not in seen:
            terms.append(term)
            seen.add(key)
    return " + ".join(terms)


def _incumbent_columns(rhs, data, response, groups):
    """Columns and their main terms in the actually accepted specification."""
    names = set(column_names(data)) - {response}
    main = {}
    for term in split_terms(rhs):
        used = [name for name in term_variables(term) if name in names]
        if len(used) == 1:
            main.setdefault(used[0], term)
    columns = [read_column(data, name) for name in column_names(data) if name in main]
    terms = [main[column.name] if column.name not in groups
             else f"brandom({quote_name(column.name)})" for column in columns]
    return columns, terms


def _incumbent_pair_leads(data, response, family, rhs, fitted, mstop, groups,
                          seed, surrogate):
    """Screen response residuals of the fitted incumbent and reuse its path."""
    from mboostrs._autospec import probe_interactions

    columns, terms = _incumbent_columns(rhs, data, response, groups)
    if len(columns) < 2 or type(family).__name__ == "Multinomial":
        return (), 0
    leads, fits, _ = probe_interactions(
        data, response, f"{quote_response(response)} ~ {rhs}", family, columns, terms,
        mstop=mstop, nu=0.1, top=6, main_fit=fitted, surrogate=surrogate,
        seed=seed, exact_fit=True)
    return leads, fits


def _incumbent_tree_term(data, response, rhs):
    """One conditional tree candidate over the accepted model's columns."""
    used = set(term_variables(rhs)) - {response}
    names = [name for name in column_names(data)
             if name in used and read_column(data, name).kind in
             ("numeric", "boolean", "categorical")]
    if len(names) < 2:
        return None
    return f"btree({', '.join(quote_name(_c) for _c in names)}, maxdepth=5, selection=sse)"


def _policy_folds(data, response, folds, groups, seed, weights):
    """Materialize one validation design for discovery, tuning and scoring."""
    from mboostrs import _rust
    from mboostrs.explore import _evaluation_weights, _folds
    from mboostrs.folds import FoldScheme, grouped_folds

    n = len(data[response])
    if set(groups) - (set(column_names(data)) - {response}):
        raise ValueError("groups must name predictor columns")
    count = folds is None or isinstance(folds, (int, float, np.integer, np.floating))
    if groups and count:
        base = _evaluation_weights(n, 5, weights)
        encoded = []
        for name in groups:
            if read_column(data, name).n_missing:
                raise ValueError(f"group column {name!r} has missing labels; supply an explicit fold design")
            labels = {}
            encoded.append([labels.setdefault(value, len(labels)) for value in np.asarray(data[name]).tolist()])
        positive = base > 0
        connected = _rust.group_components(np.ascontiguousarray(np.column_stack(encoded)[positive], dtype=float))
        codes = np.full(n, -1, dtype=int)
        codes[positive] = connected
        count = min(5, len(set(connected))) if folds is None else folds
        return grouped_folds(codes, B=count, seed=seed, weights=base)
    folds = 5 if folds is None else folds
    base = _evaluation_weights(n, folds, weights)
    parts = _folds(n, folds, seed, base)  # validate explicit schemes as well
    if isinstance(folds, FoldScheme):
        return folds
    return FoldScheme(
        np.column_stack([w for _, _, w in parts]), np.column_stack([h for _, h, _ in parts]),
        scheme=f"policy {len(parts)}-fold" if count else "policy supplied folds",
        seed=seed, base_weights=base)


def _auto_policy(data, response, *, family=None, metric="mse", complexity="balanced",
                folds=5, seed=0, mstop=100, df=4.0, budget=None, threshold_se=None,
                cache=None, progress=None, groups=(), na_action="error",
                unseen_levels="average", _checkpoint=None, score_parameter=None,
                discovery_fits=100, preexploration="native", _runner=None,
                **fit_kwargs):
    """Choose a specification by climbing a ladder, and fit it.

    The procedure prices the strong simple candidate first, then the shape
    catalogue, then selected pairs, then the residual tree. Each rung is
    priced against the incumbent on the *same* folds, and adopted only when
    the paired per-fold gain clears ``threshold_se`` standard errors.
    Rejection retains the incumbent and proceeds to the next allowed rung:
    interactions can help even when flexible main effects do not. Rejected
    terms are not carried into later proposals. The linear candidate is
    returned when no rung clears. Stopping occurs at the last allowed rung
    or at budget exhaustion.

    Costs are accounted against `budget` across every stage, and a cache hit
    replays the same computation's value. When the budget is exhausted the
    procedure returns the best specification found so far, with
    ``stopped_early`` naming the limit it passed.

    Parameters
    ----------
    data : mapping of column name to array
    response : str
    family : family instance, optional
        Probed when omitted, and reused for every rung so the comparison
        holds the loss fixed.
    metric : {"mse", "family", "brier", "log_loss", "pinball", "expectile"}
        The scoring rule the rungs are compared on. It is the metric the
        user cares about, and the same metric tunes `mstop` afterwards.
    complexity : {"simple", "balanced", "rich"}
        ``"simple"`` stops after the shapes; ``"rich"`` asks for one
        standard error instead of two.
    folds : int, ndarray or FoldScheme
    seed : int
    mstop : int
        Where the step-count sweep starts, for both the rungs and the final
        fit.
    df : float
        Degrees of freedom the shape rung offers each column.
    budget : CostBudget, optional
    threshold_se : float, optional
        Overrides the complexity preference's threshold.
    cache : FitCache, optional
        A cache shared across calls; one is created when omitted.
    progress : callable, optional
        Called as ``progress(stage, spent)`` after each stage.
    groups : sequence of str, optional
        Columns fitted as ``brandom`` effects and held out whole.
    na_action : {"error", "omit", "impute", "drop"}
    unseen_levels : {"error", "average", "zero_effect", "refuse"}

    Returns
    -------
    PolicyResult
    """
    if complexity not in _COMPLEXITY:
        raise ValueError(
            f"complexity must be one of {sorted(_COMPLEXITY)}, got {complexity!r}")
    default_threshold, allowed = _COMPLEXITY[complexity]
    threshold = default_threshold if threshold_se is None else float(threshold_se)
    if threshold < 0:
        raise ValueError(f"threshold_se must be non-negative, got {threshold_se!r}")
    cache = FitCache() if cache is None else cache
    spent = Spent(budget=budget)
    partitions = int(getattr(folds, "repeats", 1) or 1)
    trail = []
    if _runner is None:
        from mboostrs._workflow import _CURRENT
        from mboostrs.search_session import open_search

        active = _CURRENT.get()
        runner = open_search(
            data, response, family=family, score=metric, folds=folds,
            seed=seed, groups=groups, weights=fit_kwargs.get("weights"),
            budget=None if active is None else active.budget,
            na_action=na_action, unseen_levels=unseen_levels,
            score_parameter=score_parameter, nu=fit_kwargs.get("nu", 0.1),
            mstop=mstop)
    else:
        runner = _runner
    extra_fit_options = {key: value for key, value in fit_kwargs.items()
                         if key not in ("nu", "weights")}

    def stage(name, *, seconds=0.0, fits=0):
        """Record one stage's cost and name the limit it passed, if any."""
        spent.record(name, seconds=seconds, fits=fits)
        if progress is not None:
            progress(name, spent)
        return spent.exceeded()

    def evaluate(rhs):
        """Price one right-hand side: its tuned step count and its score.

        Every rung is tuned before it is compared. A specification with more
        basis functions needs more iterations to reach the fit a simpler one
        reaches early, so comparing rungs at one shared step count prices the
        flexible rung for being under-fitted rather than for being worse. The
        fold scores are read at the rung's own tuned step count, on the folds
        both rungs were tuned and scored on, so the paired comparison is
        between two tuned specifications.

        The key carries the right-hand side, the data contents, the family,
        the starting step count, the fold assignment, the weights, the
        transform and the build identity, so a repeated specification is
        served from the cache instead of being refitted.
        """
        key = cache.key(rhs=rhs, data=data, family=family, mstop=mstop, folds=folds,
                        weights=fit_kwargs.get("weights"),
                        transform=fit_kwargs.get("transform"), response=response,
                        seed=seed, metric=metric, na_action=na_action,
                        unseen_levels=unseen_levels, fit_kwargs=fit_kwargs, score_parameter=score_parameter)
        charged_fits = 0

        def compute():
            nonlocal charged_fits
            started = time.perf_counter()
            runner.run("set_spec", rhs=rhs)
            tuning = runner.run("tune", start=mstop, fold_local=True,
                                fit_options=extra_fit_options).value
            result = runner.run("evaluate", mstop=tuning.mstop,
                                fit_options=extra_fit_options).value
            charged_fits = int(getattr(tuning, "total_fits", tuning.fits)) + int(result.fit_count)
            return Priced(
                rhs=rhs, score=float(result.score),
                fold_scores=np.asarray(result.fold_scores, dtype=float),
                mstop=int(tuning.mstop), model=tuning.model, tuning=tuning,
                fits=charged_fits,
                seconds=time.perf_counter() - started)

        started = time.perf_counter()
        priced = cache.get_or_run(key, compute)
        return replace(priced, fits=charged_fits, seconds=time.perf_counter() - started)

    def decide(rung, candidate_rhs, candidate, incumbent):
        """Compare one rung with the incumbent and record the decision."""
        if np.shape(incumbent.fold_scores) != np.shape(candidate.fold_scores):
            raise ValueError(
                f"fold counts differ for {rung!r}: the incumbent and the candidate "
                f"must be scored on the same folds, got "
                f"{np.size(incumbent.fold_scores)} and "
                f"{np.size(candidate.fold_scores)}")
        record = _paired_gain(incumbent.fold_scores, candidate.fold_scores,
                              partitions, threshold)
        passed = bool(record["passed"])
        test = PairedTest(
            rung=rung, baseline_rhs=incumbent.rhs, candidate_rhs=candidate_rhs,
            baseline=incumbent.score, candidate=candidate.score,
            gain=float(record["gain"]),
            se=float(record["se"]), t=float(record["t"]), threshold_se=threshold,
            passed=passed, worse_folds=int(record["worse_folds"]),
            n_folds=int(record["n_folds"]), adopted=passed,
            reason=("the paired per-fold gain clears the stated "
                    f"{threshold:g}-SE rule" if passed
                    else (f"the paired gain is {record['t']:.2f} SE, short of "
                          f"the {threshold:g}-SE rule")))
        trail.append(test)
        return test.adopted

    # The family is chosen once and then held fixed across the rungs: a
    # comparison across rungs is only a comparison of specifications when
    # every rung pays the same loss.
    probe = runner.run("autospec", mstop=mstop, df=df, budget=0,
                       interactions="report", residual_tree=False).value
    family = family if family is not None else probe.family
    if runner.family is None:
        runner.run("set_family", family=family)
    stopped = stage("schema", fits=probe.fits)

    current = _linear_rhs(data, response, groups)
    incumbent = evaluate(current)
    stopped = stage("linear", seconds=incumbent.seconds,
                    fits=incumbent.fits) or stopped
    trail.append(PairedTest(
        rung="linear", baseline_rhs="1", candidate_rhs=current,
        baseline=float("nan"), candidate=incumbent.score, gain=0.0, se=0.0, t=0.0,
        threshold_se=threshold, passed=True, worse_folds=0,
        n_folds=int(np.size(incumbent.fold_scores)), adopted=True,
        reason="the strong simple candidate is priced first and kept as the "
               "fallback every rung has to beat"))

    def result():
        rung_adopted = next((t.rung for t in reversed(trail) if t.adopted), "linear")
        profile = _PROFILE[rung_adopted]
        if rung_adopted == "pairs" and "%X%" not in current:
            profile = "varying"
        return PolicyResult(
            spec=current, family=family, mstop=incumbent.mstop,
            profile=profile, model=incumbent.model, trail=tuple(trail),
            spent=spent, cache=cache, score=incumbent.score, threshold_se=threshold,
            complexity=complexity, stopped_early=stopped, tuning=incumbent.tuning,
            response=response, metric=metric,
            action_evidence=tuple(runner.evidence))

    if _checkpoint is not None:
        _checkpoint(result())

    discovery = probe
    for rung in _LADDER[1:]:
        if stopped:
            break
        if rung not in allowed:
            trail.append(PairedTest(
                rung=rung, baseline_rhs=current, candidate_rhs=current,
                baseline=incumbent.score, candidate=incumbent.score, gain=0.0,
                se=0.0, t=0.0, threshold_se=threshold, passed=False, worse_folds=0,
                n_folds=int(np.size(incumbent.fold_scores)), adopted=False,
                reason=f"complexity={complexity!r} stops before this rung"))
            continue
        started = time.perf_counter()
        screen_fits = 0
        if rung == "additive":
            discovery = runner.run(
                "autospec", mstop=mstop, df=df, budget=discovery_fits,
                preexploration=preexploration,
                nu=fit_kwargs.get("nu", 0.1),
                step=fit_kwargs.get("step", "fixed"),
                interactions="report", residual_tree=False, joint_tree=False).value
            candidate_rhs = _rung_rhs(rung, current, discovery)
            screen_fits = discovery.fits
        elif rung == "pairs":
            surrogate = discovery.evidence.get("preexploration", False)
            nominated = runner.run(
                "pair_candidates", rhs=current, model=incumbent.model,
                mstop=incumbent.mstop, surrogate=surrogate)
            leads, screen_fits = nominated.value, nominated.fit_charges
            candidate_rhs = _rung_rhs(rung, current, discovery, leads=leads)
        else:
            tree = runner.run("tree_candidate", rhs=current).value
            candidate_rhs = _rung_rhs(rung, current, discovery, tree=tree)
        stopped = stage(f"{rung}-screen", fits=screen_fits,
                        seconds=time.perf_counter() - started) or stopped
        if stopped:
            break
        if candidate_rhs == current:
            trail.append(PairedTest(
                rung=rung, baseline_rhs=current, candidate_rhs=candidate_rhs,
                baseline=incumbent.score, candidate=incumbent.score, gain=0.0,
                se=0.0, t=0.0, threshold_se=threshold, passed=False, worse_folds=0,
                n_folds=int(np.size(incumbent.fold_scores)), adopted=False,
                reason="this rung proposes the incumbent's own terms"))
            if _checkpoint is not None:
                _checkpoint(result())
            continue
        candidate = evaluate(candidate_rhs)
        stopped = stage(rung, seconds=candidate.seconds,
                        fits=candidate.fits) or stopped
        if decide(rung, candidate_rhs, candidate, incumbent):
            current, incumbent = candidate_rhs, candidate
        # Preserve rejected decisions too: later budget exhaustion must not
        # erase the measured evidence for retaining this incumbent.
        if _checkpoint is not None:
            _checkpoint(result())

    # The ladder stops where the budget ran out, and then the returned model is
    # the incumbent's -- priced at the step count that rung's own sweep chose,
    # so the reported score and the returned model are the same computation.
    return result()


def auto_policy(data, response, *, family=None, metric="mse", complexity="balanced",
                folds=None, seed=0, mstop=100, df=4.0, budget=None, threshold_se=None,
                cache=None, progress=None, groups=(), na_action="error",
                unseen_levels="average", score_parameter=None, discovery_fits=100,
                preexploration="native", **fit_kwargs):
    """Choose a specification with measured comparisons and a workflow budget.

    ``budget=CostBudget(...)`` reserves fit charges across discovery,
    tuning and evaluation. A simple fitted model is retained before that
    search begins; if no CV score was completed, the returned ``score`` is
    None and its evaluation status is ``not_evaluated``. Fit charges are
    hard admission limits, including failed native batches. Seconds and
    process-lifetime high-water RSS (MiB) are cooperative checkpoints and
    can overshoot within a call or preparation step. ``budget_report``
    records every admitted native fit operation and the blocked request.

    ``metric`` scores candidates and selects stopping iterations on one
    shared fold design. Pinball/expectile require ``score_parameter``.
    Without explicit folds, declared groups are held whole (connected
    components for multiple grouping columns); otherwise five random folds
    are used. The paired-SE rule is a search heuristic, not an inferential test.
    A rejected stage retains the incumbent while later allowed stages test
    other structure; no rejection forces adoption of its proposed terms.
    Without a budget the existing ladder runs without admission limits.

    ``discovery_fits`` is the main-effect probe allowance (default 100);
    ``0`` skips fitted shape discovery. Pairs still screen the accepted
    model's residual, and a tree can still be compared under the whole-workflow
    ``budget``. Discovery begins after the linear candidate has been scored
    and retained. ``preexploration`` may
    be ``"native"``, ``"neural"``, ``"dual"`` or ``"off"``. Discovery
    currently ranks shape probes by family loss; the requested ``metric``
    governs each complete candidate's stopping iteration and final adoption.
    """
    if complexity not in _COMPLEXITY:
        raise ValueError(f"complexity must be one of {sorted(_COMPLEXITY)}")
    check_int_at_least("discovery_fits", discovery_fits, 0)
    if preexploration not in ("native", "neural", "dual", "off"):
        raise ValueError('preexploration must be "native", "neural", "dual" or "off"')
    if threshold_se is not None and (not np.isfinite(threshold_se) or threshold_se < 0):
        raise ValueError("threshold_se must be finite and non-negative")
    if metric == "family":
        if score_parameter is not None:
            raise ValueError("score_parameter is only used by pinball or expectile loss")
    else:
        from mboostrs import _rust

        _rust.response_score(np.array([0.]), np.array([[0.5]]), np.ones(1), metric, score_parameter)
    groups = (groups,) if isinstance(groups, str) else tuple(groups)
    folds = _policy_folds(data, response, folds, groups, seed, fit_kwargs.get("weights"))
    arguments = dict(family=family, metric=metric, complexity=complexity, folds=folds,
                     seed=seed, mstop=mstop, df=df, threshold_se=threshold_se,
                     cache=cache, progress=progress, groups=groups, na_action=na_action,
                     unseen_levels=unseen_levels, score_parameter=score_parameter,
                     discovery_fits=discovery_fits, preexploration=preexploration, **fit_kwargs)
    if budget is None:
        return _auto_policy(data, response, **arguments)
    if not isinstance(budget, CostBudget):
        raise TypeError("budget must be a CostBudget")
    from mboostrs._workflow import Workflow
    from mboostrs.search_session import open_search

    tracker = Workflow(budget)
    with tracker:
        runner = open_search(
            data, response, family=family, score=metric, folds=folds,
            seed=seed, groups=groups, weights=fit_kwargs.get("weights"),
            budget=budget, na_action=na_action, unseen_levels=unseen_levels,
            score_parameter=score_parameter, nu=fit_kwargs.get("nu", 0.1),
            mstop=mstop)
        initial = runner.run("autospec", budget=0).value
        if runner.family is None:
            runner.run("set_family", family=initial.family)
        rhs = _linear_rhs(data, response, groups)
        fallback_kwargs = dict(fit_kwargs)
        fallback_kwargs.pop("nu", None)
        fallback_kwargs.pop("weights", None)
        runner.run("set_spec", rhs=rhs)
        model = runner.run("fit", fit_options=fallback_kwargs).value
        cache = cache if cache is not None else FitCache()
        best = PolicyResult(spec=rhs, family=initial.family, mstop=mstop, profile="linear",
                            model=model, trail=(), spent=Spent(budget=budget), cache=cache,
                            score=None, threshold_se=(_COMPLEXITY[complexity][0]
                                                     if threshold_se is None else threshold_se),
                            complexity=complexity, stopped_early=None, response=response, metric=metric,
                            action_evidence=tuple(runner.evidence))

        def checkpoint(result):
            nonlocal best
            best = result
            tracker.checkpoint()

        arguments.update(family=initial.family, cache=cache,
                         _checkpoint=checkpoint, _runner=runner)
        if progress is not None:
            def measured_progress(stage, _reported):
                measured = tracker.report()
                progress(stage, Spent(seconds=measured["elapsed_seconds"],
                                      fits=measured["fit_charges"],
                                      memory_mb=measured["peak_memory_mb"], budget=budget))
            arguments["progress"] = measured_progress
        try:
            result = _auto_policy(data, response, **arguments)
            best = result
            tracker.checkpoint()
        except Exception:
            if tracker.stopped is None:
                raise
            result = replace(best, stopped_early=tracker.stopped)
        report = tracker.report()
        result.spent.budget = budget
        result.spent.fits = report["fit_charges"]
        result.spent.seconds = report["elapsed_seconds"]
        result.spent.memory_mb = report["peak_memory_mb"]
        result.spent.stages.append({"stage": "workflow", "fits": result.spent.fits,
                                    "seconds": result.spent.seconds})
        result = replace(result, budget_report=report,
                         action_evidence=tuple(runner.evidence))
        if progress is not None:
            progress("workflow", result.spent)
        return result


def _paired_gain(before_folds, after_folds, partitions, threshold):
    """The paired per-fold comparison, as the core computes it.

    `partitions` is the number of times the scheme resampled the folds; a
    repeated scheme's standard error is averaged over the repeats rather
    than pooled across them, so repeated and single schemes are not scored
    on the same scale by accident.
    """
    from mboostrs import _rust

    return _rust.paired_gain(np.asarray(before_folds, dtype=float),
                             np.asarray(after_folds, dtype=float),
                             int(partitions), float(threshold))


@dataclass(frozen=True)
class Priced:
    """One priced specification: its tuned step count, its score and its fit."""

    rhs: str
    score: float
    fold_scores: np.ndarray
    mstop: int
    model: object
    tuning: object
    fits: int
    seconds: float


@dataclass(frozen=True)
class PolicyResult:
    """Specification selected by the policy, with evidence for each step.

    ``profile`` is linear, additive, tensor, varying (interactions using
    ``by=`` without a tensor), or hybrid (an accepted residual tree).
    """

    spec: str
    family: object
    mstop: int
    profile: str
    model: object
    trail: tuple
    spent: Spent
    cache: FitCache
    score: float | None
    threshold_se: float
    complexity: str
    stopped_early: str | None
    tuning: object = None
    response: str = ""
    metric: str = "mse"
    budget_report: dict | None = None
    action_evidence: tuple = ()

    @property
    def adopted(self):
        """The rungs the ladder climbed, in order."""
        return tuple(t.rung for t in self.trail if t.adopted)

    @property
    def fits(self):
        """Model fits the whole procedure spent."""
        return self.spent.fits

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record: the specification, the trail and the cost."""
        return json_safe({
                "spec": self.spec,
                "family": type(self.family).__name__,
                "response": self.response,
                "metric": self.metric,
                "mstop": self.mstop,
                "profile": self.profile,
                "score": None if self.score is None else float(self.score),
                "evaluation_status": "not_evaluated" if self.score is None else "inner_cv_descriptive",
                "adopted": list(self.adopted),
                "trail": [t.to_dict() for t in self.trail],
                "spent": self.spent.to_dict(),
                "cache": self.cache.to_dict(),
                "stopped_early": self.stopped_early,
                "budget_report": self.budget_report,
                "action_evidence": [item.to_dict() for item in self.action_evidence],
                "threshold_se": self.threshold_se,
                "complexity": self.complexity,
            })

    def __repr__(self):  # pragma: no cover - display only
        score = "not evaluated" if self.score is None else f"{self.score:.6g}"
        return (f"PolicyResult({self.spec!r}, profile={self.profile!r}, "
                f"mstop={self.mstop}, {self.metric}={score}, "
                f"adopted={list(self.adopted)}, fits={self.fits})")
