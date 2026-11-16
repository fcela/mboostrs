"""What happens when new data leaves the range the model was fitted on.

A P-spline basis is defined between its boundary knots. Outside them the
basis extrapolates linearly, which is a decision, not a fact about the data:
the fitted curve is continued by its slope at the edge, and nothing in the fit
constrains it. `predict(newdata=)` said so with

    Some 'x' values are beyond 'boundary.knots'; Linear extrapolation used.

which is `mboost`'s wording and has three problems. The `'x'` is the literal
string, not the column -- a model over `area` and `rooms` is told about
`'x'`. It fires once per call however many learners are involved, so the
second one is silent. And it says nothing about *how far* outside: one row a
hair past the edge and a thousand rows two training ranges out produce the
same sentence.

:func:`range_rows` measures it instead: per learner, the training range, the
new range, how many rows fall each side, and the worst distance expressed in
training ranges, which is the number that says whether the extrapolation is a
rounding artefact or a different question.

The policy says what to do about it:

* ``"warn"`` (default) -- predict, and say which learners, how many rows and
  how far. One message, naming everything. The warning is an
  :class:`ExtrapolationWarning` (a plain ``Warning``, not a ``UserWarning``),
  whose ``.rows`` carry the offending :class:`RangeRow` objects for callers
  that want the distances rather than the paragraph.
* ``"error"`` -- refuse, with the same content. For a pipeline where an
  out-of-range row means the input is wrong.
* ``"allow"`` -- predict silently. For a caller who has decided that linear
  continuation is what they want.
* ``"clip"`` -- evaluate the basis at the nearest training value instead, so
  the prediction is flat outside the range rather than continuing the edge
  slope, and say so once. Neither is more correct in general: a flat
  continuation assumes the effect stops changing, a linear one assumes it goes
  on changing at the edge rate. Clipping makes the assumption visible and
  bounded.

``MATERIAL_DISTANCE`` (5% of the training span) controls warnings only.
Smaller excursions still extrapolate, but do not emit a warning. ``error``
rejects every value strictly outside the interval and ``clip`` clamps every
such value, independent of that threshold. ``range_report()`` counts all
excursions, including those one representable float beyond the boundary.

A loop that calls ``predict`` once per fold -- cross-validation, conformal
calibration -- wraps its body in :func:`collecting` and lets the per-fold rows
accumulate, then prints a single :func:`summarize` paragraph naming the
farthest offender. The fold loop itself never warns.

Coverage: the policy checks univariate spline learners in full-model
newdata predictions of Python ``gamboost`` and ``gamboostLSS`` (every
distribution parameter) and of R ``predict.mboostrs_gamboost``, where R also
checks each coordinate of ``bspatial``; R gamboostLSS applies it per
parameter. A ``by=`` variable of a spline term is checked against its own
training range, because the term's contribution is the basis times that
variable: an out-of-range ``by=`` value scales the effect and is clipped
like any other covariate. Linear terms continue the fitted line, so the
policy leaves them alone; ``range_report()`` lists them as ``term="linear"``
rows with their distance.

Contribution predictions (``predict(which=)``, ``prediction_contributions()``)
and Multinomial predictions take the same check. A ``bspatial`` term is checked
per coordinate, against each coordinate's training range. The B-spline margins
of a ``%X%`` term are checked against their own training ranges; under
``"clip"`` the clamped values are the ones the margin's basis is evaluated at,
and the other terms that read the same column keep the raw values.

``btree``, ``bkernel``, ``buser`` and ``brad`` have no training-range boundary:
a tree is constant past its outermost split, a kernel decays towards zero, and
a user-supplied design has no coordinates. For those terms the policy is not
applied, and ``validate()`` lists them as ``not_range_checked``. The
scikit-learn adapters predict through the same
``gamboost.predict``, so an extrapolation inside scikit-learn's *own*
cross-validation loop (which this package does not run) warns once per fold,
like a direct ``predict`` call.
"""

from __future__ import annotations

import functools
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from mboostrs.agent_docs import guide_hint

__all__ = [
    "POLICIES", "RangeRow", "ExtrapolationWarning", "apply_policy",
    "CollectedRows", "collecting", "forward_extrapolation", "one_summary",
    "range_rows", "routine_warnings", "summarize",
]

#: The policies `extrapolation=` accepts.
POLICIES = ("warn", "error", "allow", "clip")

#: Minimum excursion, in training spans, that emits a warning. Small fold
#: boundary excursions remain in range reports and still obey strict error
#: and clip policies. This is a presentation threshold, not a numeric tolerance.
MATERIAL_DISTANCE = 0.05


class ExtrapolationWarning(Warning):
    """Predicting outside the training range of a spline basis.

    A direct subclass of ``Warning``, not of ``UserWarning``. A common
    blanket filter, ``warnings.filterwarnings("ignore", category=UserWarning)``,
    silences every ``UserWarning`` and so would silence this one; the
    extrapolation warning is filtered only by its own class,
    ``category=ExtrapolationWarning``. ``.rows`` holds the material
    :class:`RangeRow` objects that triggered the warning.
    """

    def __init__(self, message, rows=None):
        super().__init__(message)
        self.rows = list(rows) if rows else []


class CollectedRows(list):
    """The rows a fold loop collected, and how many predict calls flagged them.

    ``calls`` counts the predict calls that reported at least one material
    row. A cross-validation loop makes one predict call per fold, so ``calls``
    is the number of fold predictions affected.
    """

    calls: int = 0
    #: Whether the collected rows were clipped (``extrapolation="clip"``)
    #: rather than extrapolated, which changes the sentence the summary gives.
    clipped: bool = False


#: While :func:`collecting` is active, :func:`apply_policy` records the rows
#: here instead of printing a warning per call. A fold loop calls ``predict``
#: once per fold; one summary at the end says what all of them found.
_quiet_rows: CollectedRows | None = None


@contextmanager
def collecting():
    """Collect extrapolation evidence during a fold loop instead of warning
    per call; the caller re-emits one summary with :func:`summarize`.

    A cross-validation or conformal loop calls ``predict`` once per fold, and
    each call that leaves the training range would print its own paragraph.
    Inside this context the offending rows accumulate on the yielded list;
    the calling code runs :func:`summarize` over it once the loop is done.
    """
    global _quiet_rows
    previous = _quiet_rows
    _quiet_rows = CollectedRows()
    try:
        yield _quiet_rows
    finally:
        _quiet_rows = previous


def forward_extrapolation(rows, stacklevel: int = 3, owner: str | None = None) -> None:
    """Pass a finished collection up to the enclosing routine, or warn now.

    A routine that collected its own fold predictions (`cv_risk`) calls this
    when it ends. Inside a :func:`routine_warnings` block the rows and their
    count join the enclosing collection, and that routine issues the single
    warning. Outside any routine the summary of `rows` is issued here.
    """
    global _quiet_rows
    if _quiet_rows is not None:
        _quiet_rows.extend(rows)
        _quiet_rows.calls += getattr(rows, "calls", 0)
        _quiet_rows.clipped = _quiet_rows.clipped or getattr(rows, "clipped", False)
        return
    aggregated = summarize(rows, owner)
    if aggregated is not None:
        warnings.warn(aggregated, stacklevel=stacklevel)


def one_summary(fn):
    """Decorate a routine so its extrapolation warnings are issued once, at its end.

    Equivalent to running the body inside :func:`routine_warnings`. The wrapped
    function keeps its name, docstring and signature.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with routine_warnings(stacklevel=3, owner=fn.__name__):
            return fn(*args, **kwargs)

    return wrapper


@contextmanager
def routine_warnings(stacklevel: int = 3, owner: str | None = None):
    """Issue one extrapolation warning for a whole routine call.

    A routine such as `cv_risk`, `tune_mstop` or `compare_specifications` runs
    many fold fits, and some of them predict outside the training range. Each
    of those fits reports to the collector and the routine issues one warning
    when it ends, naming the learners and the number of fold predictions
    affected. A routine that runs inside another routine adds its rows to the
    outer collection, and the outer routine issues the single warning.
    """
    global _quiet_rows
    if _quiet_rows is not None:
        yield _quiet_rows
        return
    # The penalty-drift warnings of the routine's fits are counted in the same
    # way, and the drift summary follows the extrapolation summary.
    from mboostrs._model_api import penalty_drift_collecting

    with penalty_drift_collecting(stacklevel=stacklevel + 1), collecting() as rows:
        yield rows
    aggregated = summarize(rows, owner)
    if aggregated is not None:
        warnings.warn(aggregated, stacklevel=stacklevel)


@dataclass(frozen=True)
class RangeRow:
    """One learner's training range against the data it is being asked about."""

    learner: str
    variable: str
    train_low: float
    train_high: float
    new_low: float
    new_high: float
    n_below: int
    n_above: int

    @property
    def n_outside(self) -> int:
        return self.n_below + self.n_above

    @property
    def worst_distance(self) -> float:
        """How far outside, in training ranges.

        ``0`` is inside; ``1`` means a row sits a whole training range beyond
        the edge, which is a different question rather than a boundary case.
        """
        span = self.train_high - self.train_low
        if span <= 0:
            return float("inf") if self.n_outside else 0.0
        below = (self.train_low - self.new_low) / span
        above = (self.new_high - self.train_high) / span
        return float(max(below, above, 0.0))

    def as_dict(self) -> dict:
        return {
            "learner": self.learner,
            "variable": self.variable,
            "train_low": self.train_low,
            "train_high": self.train_high,
            "new_low": self.new_low,
            "new_high": self.new_high,
            "n_below": self.n_below,
            "n_above": self.n_above,
            "n_outside": self.n_outside,
            "worst_distance": self.worst_distance,
        }


def row_for(learner: str, variable: str, values, train_low, train_high) -> RangeRow:
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([train_low, train_high], dtype=np.float64)
    return RangeRow(
        learner=learner, variable=variable,
        train_low=float(train_low), train_high=float(train_high),
        new_low=float(finite.min()), new_high=float(finite.max()),
        n_below=int(np.sum(finite < train_low)),
        n_above=int(np.sum(finite > train_high)),
    )


def range_rows(entries) -> list[RangeRow]:
    """``RangeRow`` per ``(learner, variable, values, low, high)`` entry."""
    return [row_for(*entry) for entry in entries]


def _material(rows, mind):
    """The rows far enough outside the training range to warn about."""
    return [row for row in rows if row.n_outside and row.worst_distance >= mind]


#: The R-only ``boundary.knots=c(lo, hi)`` argument of a term label. Python
#: users state the fitted range in words instead, so the warning shows it so.
_BOUNDARY_KNOTS_ARG = re.compile(r",?\s*boundary\.knots\s*=\s*c\([^)]*\)")


def _python_label(learner: str) -> tuple[str, bool]:
    """The term label as a Python user writes it, and whether it held a
    ``boundary.knots=`` argument. The label ``bbs(x, df=4, boundary.knots=c(8.2,
    11.6))`` becomes ``bbs(x, df=4)``; the fitted range is printed separately."""
    label, count = _BOUNDARY_KNOTS_ARG.subn("", learner)
    return label, count > 0


def _describe(rows: list[RangeRow], calls: int | None = None,
              owner: str | None = None, clipped: bool = False) -> str:
    outside = [row for row in rows if row.n_outside]
    worst = max(outside, key=lambda row: row.worst_distance)
    parts = []
    for row in sorted(outside, key=lambda r: -r.worst_distance):
        where = []
        if row.n_below:
            where.append(f"{row.n_below} below {row.train_low:.6g}")
        if row.n_above:
            where.append(f"{row.n_above} above {row.train_high:.6g}")
        label, explicit_range = _python_label(row.learner)
        fitted = (f" fitted on [{row.train_low:.6g}, {row.train_high:.6g}]"
                  if explicit_range else "")
        parts.append(
            f"{label}{fitted}, covariate {row.variable}: {' and '.join(where)} "
            f"(worst {row.worst_distance:.2g} training ranges out)"
        )
    learners = len({row.learner for row in outside})
    if calls is None:
        lead = (f"{learners} learner(s) are asked to predict outside the range "
                f"they were fitted on. ")
    else:
        folds = f"{calls} fold predictions" if calls != 1 else "1 fold prediction"
        lead = (f"{folds} in this call asked {learners} learner(s) to predict "
                f"outside the range they were fitted on. ")
    return (
        lead + "; ".join(parts) + ". "
        f"A spline basis continues by its edge slope outside its boundary "
        f"knots, which the fit does not constrain -- at "
        f"{worst.worst_distance:.2g} training ranges out that is an "
        f"assumption, not an estimate. " + _remedy(owner, clipped)
    )


def _remedy(owner: str | None, clipped: bool = False) -> str:
    """The sentence naming the `extrapolation=` values, or what clipping did.

    With `owner` set (the name of the routine the user called, such as
    ``term_report``), the values are named on that routine, which is where
    the user can pass them. Without it, they are named on ``predict``.
    `clipped` says the rows were already clipped to the training range.
    """
    if clipped:
        return ("Those rows were clipped to the training range, so the effect is "
                "held flat outside it rather than continuing at the edge slope.")
    if owner is None:
        return ("Pass extrapolation='allow' to accept it, 'clip' to hold the "
                "effect flat outside the range instead, or 'error' to refuse."
                + guide_hint("modeling-tactics"))
    return (f"Pass {owner}(extrapolation='allow') to accept it, or "
            f"extrapolation='clip' to hold the effect flat outside the range "
            f"instead, or extrapolation='error' to refuse." + guide_hint("modeling-tactics"))


def _emit(warning: ExtrapolationWarning, stacklevel: int, clipped: bool = False) -> None:
    """Print the warning -- unless a fold loop is collecting, in which case
    the rows are saved for the one summary at the end."""
    if _quiet_rows is None:
        warnings.warn(warning, stacklevel=stacklevel)
    else:
        _quiet_rows.extend(warning.rows)
        _quiet_rows.calls += 1
        _quiet_rows.clipped = _quiet_rows.clipped or clipped


def _merge(rows):
    """Rows deduplicated per learner, keeping the farthest per learner."""
    best = {}
    for row in rows:
        key = (row.learner, row.variable, row.train_low, row.train_high)
        if key not in best or row.worst_distance > best[key].worst_distance:
            best[key] = row
    return sorted(best.values(), key=lambda r: -r.worst_distance)


#: Columns of ``extrapolation_report()`` in the order they are returned.
REPORT_COLUMNS = (
    "parameter", "term", "covariate", "kind", "n_below", "n_above", "n_outside",
    "train_low", "train_high", "new_low", "new_high", "worst_distance",
)


def report_records(entries) -> list[dict]:
    """The report records for the rows that have values outside their range.

    ``entries`` is a sequence of ``(parameter, kind, row)`` triples, where
    ``row`` is a :class:`RangeRow` and ``kind`` names the kind of term
    (``"spline basis"`` or ``"linear"``). Rows with no value outside the
    training range are dropped: the report lists exceedances only.
    """
    records = []
    for parameter, kind, row in entries:
        if not row.n_outside:
            continue
        records.append({
            "parameter": parameter,
            "term": row.learner,
            "covariate": row.variable,
            "kind": kind,
            "n_below": row.n_below,
            "n_above": row.n_above,
            "n_outside": row.n_outside,
            "train_low": row.train_low,
            "train_high": row.train_high,
            "new_low": row.new_low,
            "new_high": row.new_high,
            "worst_distance": row.worst_distance,
        })
    return records


#: The summaries :func:`show_summary_once` has issued in this process, keyed by
#: their wording with the numbers replaced.
_shown_summaries: set = set()


def show_summary_once(summary, stacklevel: int = 2) -> None:
    """Issue an extrapolation summary unless one with the same wording (numbers
    aside) was already issued in this process.

    A loop of calls that each leave the training range (for example
    :func:`~mboostrs.ci_test` over many seeds) would otherwise print the same
    paragraph with different counts each time. Inside
    :func:`~mboostrs.collect_warnings` every summary is issued, and the block
    groups them."""
    from mboostrs._warning_collect import collecting_active, warning_key

    if not collecting_active():
        key = (type(summary), warning_key(summary))
        if key in _shown_summaries:
            return
        _shown_summaries.add(key)
    warnings.warn(summary, stacklevel=stacklevel + 1)


def reset_shown_summaries() -> None:
    """Forget the summaries :func:`show_summary_once` has issued (for tests)."""
    _shown_summaries.clear()


def summarize(rows, owner: str | None = None) -> ExtrapolationWarning | None:
    """One warning over rows collected from a fold loop, or ``None``.

    `owner` is the name of the user-facing routine, which the remedy names."""
    merged = _merge(rows)
    if not merged:
        return None
    return ExtrapolationWarning(
        _describe(merged, getattr(rows, "calls", None), owner,
                  clipped=getattr(rows, "clipped", False)), rows=merged)


def apply_policy(rows: list[RangeRow], values_by_index, policy: str,
                 *, stacklevel: int = 3, material: float = MATERIAL_DISTANCE):
    """Act on the policy, and return the values to evaluate the basis at.

    ``values_by_index`` maps the position of a row in ``rows`` to the array
    that row measured. Under ``"clip"`` the returned mapping holds clamped
    copies; under every other policy it is returned unchanged.

    ``material`` is the minimum distance, in training ranges, that emits a
    warning. Error and clip policies act on every strictly out-of-range
    value. A fold loop can collect warnings and summarize them once.
    """
    if policy not in POLICIES:
        raise ValueError(
            f"extrapolation must be one of {POLICIES}, got {policy!r}"
        )
    outside = [row for row in rows if row.n_outside]
    if not outside or policy == "allow":
        return values_by_index
    if policy == "error":
        raise ValueError(_describe(outside))
    warn_about = _material(outside, material)
    if policy == "warn":
        if warn_about:
            _emit(ExtrapolationWarning(_describe(warn_about), rows=warn_about), stacklevel)
        return values_by_index

    clipped = dict(values_by_index)
    for index, row in enumerate(rows):
        if not row.n_outside or index not in clipped:
            continue
        clipped[index] = np.clip(
            np.asarray(clipped[index], dtype=np.float64),
            row.train_low, row.train_high,
        )
    if warn_about:
        _emit(ExtrapolationWarning(_describe(warn_about, clipped=True), rows=warn_about),
              stacklevel, clipped=True)
    return clipped
