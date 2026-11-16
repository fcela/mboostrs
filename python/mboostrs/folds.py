"""Fold constructors for data that is not a simple random sample.

``cv_weights`` offers three schemes -- bootstrap, k-fold, subsampling -- and
all three assume the rows are exchangeable. A great deal of data is not:

* **Classes are unbalanced.** A plain 10-fold split of 200 rows with 12
  positives gives folds holding 0, 1, 2 or 3 of them. The out-of-fold risk of
  a fold with no positive case is not an estimate of anything, and the spread
  across folds is dominated by that accident rather than by the model.
* **Rows are clustered.** Repeated measurements on the same patient, several
  quarters of the same firm, pupils in one school. Splitting rows at random
  puts the same cluster on both sides of the split, so the held-out rows are
  partly predicted by their own cluster-mates and the cross-validated risk is
  optimistic. This is the leak the scheme has to prevent, not merely record.
* **Rows are ordered in time.** A random split trains on the future to predict
  the past.

The constructors here return a :class:`FoldScheme`: the ``(n, B)`` in-bag
weight matrix `cvrisk` and `stabsel` already take -- ``np.asarray(scheme)``
gives it, so a scheme can be passed anywhere a matrix could -- plus what the
matrix alone cannot say. Which rows each fold holds out, what the scheme is,
what seed produced it, and which grouping it respects, so
:meth:`FoldScheme.check_leakage` can be asked about any grouping afterwards,
including one the scheme was not built around.

Every constructor follows the same weight convention as ``cv_weights``:
column ``b`` carries the model's weights on the in-bag rows and zero on the
rows held out. A row whose model weight is zero is not in the sample at all,
and is not counted as held out anywhere.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from mboostrs._output import export_result
from mboostrs._validate import check_int_at_least
from mboostrs._weights import normalize_weights

__all__ = [
    "FoldScheme",
    "blocked_folds",
    "grouped_folds",
    "known_group_folds",
    "loo_folds",
    "repeated_kfold_folds",
    "rolling_folds",
    "stratified_folds",
]


def _base_weights(n: int, weights) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(
            f"weights has shape {arr.shape}, expected ({n},) to match the sample"
        )
    # The public weight contract: finite, non-negative, and not all zero.
    return normalize_weights(arr, n, policy="raw").given


def _check_B(B: int, upper: int, what: str) -> int:
    checked = check_int_at_least("B", B, 2, allow_none=False)
    # pyright narrow: allow_none=False and a non-None value never return None.
    B = int(checked) if checked is not None else 2
    if B > upper:
        raise ValueError(
            f"B={B} exceeds the {upper} {what} available; each fold would hold "
            f"out nothing"
        )
    return B


@dataclass(frozen=True)
class FoldScheme:
    """An ``(n, B)`` in-bag weight matrix, and what it means.

    Passing one where a matrix is expected works: ``np.asarray(scheme)``
    returns the weights, so ``model.cvrisk(folds=scheme)`` needs no
    unwrapping.

    Build a scheme with a constructor such as ``kfold_folds`` or
    ``grouped_folds`` rather than by hand. Every array is copied and made
    read-only at construction, and construction raises ``ValueError`` when a
    shape disagrees with ``weights``, a fold trains on a row it also holds out,
    a fold trains on or scores no rows, or ``repeats`` does not divide the
    number of folds.

    Attributes
    ----------
    weights : ndarray, shape (n, B)
        In-bag weights: the model's weights on the rows a fold trains on, zero
        on the rows it holds out.
    held_out : ndarray of bool, shape (n, B)
        Whether the row is held out in that fold. A row whose model weight is
        zero is out of the sample and is held out nowhere.
    scheme : str
        The scheme's name, for example ``"stratified 10-fold"``.
    seed : int or None
        The seed, when the scheme draws one.
    groups : ndarray or None
        The grouping the scheme kept whole, when it was built around one.
    strata : ndarray or None
        The strata whose proportions the scheme preserved.
    ordered : bool
        Whether the scheme orders folds in time, so that the check for
        training on the future applies.
    judge : ndarray of bool, shape (n, B), or None
        The rows whose out-of-fold risk is scored, when that is not every
        zero-weight row (rolling-origin folds). It must be a subset of
        ``held_out``.
    base_weights : ndarray, shape (n,), or None
        Observation weights before folds were assigned, used to weight
        evaluation losses. ``None`` means unit weights.
    repeats : int
        Number of stacked partitions; ``B // repeats`` folds each.
    known_groups : ndarray or None
        Grouping spread across folds on purpose (``known_group_folds``).
    times : ndarray or None
        Timestamps the order of a ``rolling_folds`` scheme was taken from.
    embargo : float, numpy.timedelta64 or None
        Minimum gap, in the units of ``times``, between the last training time
        and the first judged time of a fold.
    """

    #: ``(n, B)`` in-bag weights: the model's weights on the rows a fold
    #: trains on, zero on the rows it holds out.
    weights: np.ndarray
    #: ``(n, B)`` boolean: whether the row is *held out* in that fold. Not
    #: simply ``weights == 0`` -- a row whose model weight is zero is out of
    #: the sample entirely and is held out nowhere.
    held_out: np.ndarray
    #: The scheme's name, e.g. ``"stratified 10-fold"``.
    scheme: str
    #: The seed, when the scheme draws one. Deterministic schemes have none.
    seed: int | None = None
    #: The grouping the scheme kept intact, when it was built around one.
    groups: np.ndarray | None = None
    #: The strata whose proportions the scheme preserved, when it has any.
    strata: np.ndarray | None = None
    #: Whether the scheme orders folds in time, so that the check for
    #: training on the future applies.
    ordered: bool = False
    #: ``(n, B)`` boolean, when a fold's risk is not "every zero-weight row".
    #: The CV engines score every in-sample row whose fold weight is zero, which
    #: is the right judge set for k-fold, blocked, and bootstrap splits -- all of
    #: them train on one part and score the rest. A rolling-origin fold leaves
    #: rows that are *neither* trained on *nor* judged (the future beyond the
    #: horizon, and the rows before a sliding window), and those must not be
    #: scored. ``judge`` names the exact rows whose out-of-fold risk is
    #: computed; unset means "score the zero-weight rows", which is what the
    #: documented schemes want anyway.
    judge: np.ndarray | None = None
    _note: str = field(default="", repr=False)
    #: Observation weights before assigning folds, including rows that never
    #: enter a rolling training window. Constructors retain an independent
    #: copy. In specification scoring these weight evaluation losses;
    #: ``weights`` already contains the complete training weights.
    #: For a manually constructed scheme, None means evaluation uses unit
    #: weights unless the caller supplies them explicitly.
    base_weights: np.ndarray | None = field(default=None, repr=False)
    #: Partitions of the rows stacked side by side, in order: ``B`` columns
    #: hold ``repeats`` partitions of ``B // repeats`` folds each. Folds from
    #: different partitions reuse the same rows, so the specification-scoring
    #: functions compute a paired standard error within each partition rather
    #: than over all columns as if each fold were a separate draw.
    repeats: int = 1
    #: The grouping whose members are spread across folds on purpose, so that
    #: every held-out row has a group-mate in its fold's training rows. Set by
    #: :func:`known_group_folds`; the score then measures prediction for groups
    #: the model has seen. Unlike ``groups``, nothing is kept whole.
    known_groups: np.ndarray | None = field(default=None, repr=False)
    #: Timestamps (numeric or ``datetime64``) that the time order of a
    #: :func:`rolling_folds` scheme was taken from. Unset means the row order
    #: is the time order.
    times: np.ndarray | None = field(default=None, repr=False)
    #: Minimum gap, in the units of ``times``, between the last training time
    #: and the first judged time of a fold. Rows inside the gap are neither
    #: trained on nor judged.
    embargo: float | np.timedelta64 | None = None

    def __post_init__(self):
        """Validate the scheme and freeze its arrays.

        Checked here rather than when a fold loop first reads the scheme, so a
        malformed scheme fails where it is built. Every array is stored as a
        read-only copy: a scheme is fingerprinted and shared between
        comparisons, and an in-place edit after construction would change
        what an earlier fingerprint describes.
        """
        def frozen(name, value, dtype):
            array = np.array(value, dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
            return array

        weights = frozen("weights", self.weights, np.float64)
        if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
            raise ValueError(
                f"FoldScheme weights must be a non-empty (n, B) matrix, got shape "
                f"{weights.shape}")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("FoldScheme weights must be finite and non-negative")
        n, folds = weights.shape
        held = frozen("held_out", self.held_out, np.bool_)
        if held.shape != weights.shape:
            raise ValueError(
                f"FoldScheme held_out has shape {held.shape}; weights have "
                f"{weights.shape}")
        if np.any(held & (weights > 0)):
            raise ValueError("FoldScheme holds out rows it also trains on; a held-out "
                             "row must have weight zero in that fold")
        if self.judge is not None:
            judge = frozen("judge", self.judge, np.bool_)
            if judge.shape != weights.shape:
                raise ValueError(
                    f"FoldScheme judge has shape {judge.shape}; weights have "
                    f"{weights.shape}")
            if np.any(judge & (weights > 0)):
                raise ValueError("FoldScheme judges rows it trains on")
            if np.any(judge & ~held):
                raise ValueError("FoldScheme judge must be a subset of held_out")
            scored = judge
        else:
            scored = held
        empty_fit = np.flatnonzero(~np.any(weights > 0, axis=0))
        if empty_fit.size:
            raise ValueError(f"FoldScheme fold {int(empty_fit[0])} trains on no rows")
        empty_score = np.flatnonzero(~np.any(scored, axis=0))
        if empty_score.size:
            raise ValueError(f"FoldScheme fold {int(empty_score[0])} scores no rows")
        for name in ("groups", "strata", "known_groups", "times"):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, copy=True)
                if array.shape[:1] != (n,):
                    raise ValueError(
                        f"FoldScheme {name} has {array.shape[0] if array.ndim else 0} "
                        f"entries for {n} rows")
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        if self.base_weights is not None:
            base = frozen("base_weights", self.base_weights, np.float64)
            if base.shape != (n,):
                raise ValueError(
                    f"FoldScheme base_weights has shape {base.shape}, expected ({n},)")
            if not np.all(np.isfinite(base)) or np.any(base < 0):
                raise ValueError("FoldScheme base_weights must be finite and non-negative")
        if int(self.repeats) < 1 or folds % int(self.repeats):
            raise ValueError(
                f"FoldScheme repeats={self.repeats} must be a positive divisor of "
                f"its {folds} folds")

    def __array__(self, dtype=None, copy=None):
        values = self.weights
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights.shape

    @property
    def n_folds(self) -> int:
        return int(self.weights.shape[1])

    def __len__(self) -> int:
        return int(self.weights.shape[0])

    def __repr__(self) -> str:
        return (f"FoldScheme({len(self.weights)} rows x {self.n_folds} folds, "
                f"{self.scheme}, ordered={self.ordered})"
                + (f", seed={self.seed}" if self.seed is not None else ""))

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record with the weight evidence only on request.

        Parameters
        ----------
        include_arrays : bool, default False
            Include ``weights``, ``held_out``, ``groups``, ``strata``,
            ``known_groups`` and ``times`` as nested lists. Without it the
            record holds scalars plus ``n_folds`` and ``n_rows``.

        Returns
        -------
        dict
            JSON-serialisable record.
        """
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("weights", "held_out", "groups", "strata", "known_groups", "times"),
            extras={"n_folds": self.n_folds, "n_rows": len(self)},
        )

    def fold(self, b: int) -> np.ndarray:
        """The row indices fold ``b`` holds out.

        Parameters
        ----------
        b : int
            Zero-based fold (column) index.

        Returns
        -------
        ndarray of int
            Sorted row positions with ``held_out[:, b]`` true.
        """
        return np.flatnonzero(self.held_out[:, b])

    def fold_sizes(self) -> np.ndarray:
        """How many rows each fold holds out.

        Returns
        -------
        ndarray of int, shape (B,)
            Count of held-out rows per fold.
        """
        return self.held_out.sum(axis=0)

    def take(self, rows):
        """The same scheme restricted to `rows` (positions or boolean mask).

        A model fitted with ``na_action='omit'`` estimates on the complete
        cases, so a scheme drawn over the full frame must drop the same rows
        before its weights can drive that model's cross-validation. Every
        per-row field -- weights, held-out masks, the judge set, group and
        stratum labels, base weights -- is subset together, so fold
        membership and judge semantics are preserved exactly.

        Parameters
        ----------
        rows : array_like of int or bool
            Row positions to keep, or a boolean mask of length ``n``.

        Returns
        -------
        FoldScheme
            A new scheme over the selected rows, with the same fold count.

        Raises
        ------
        ValueError
            If the restricted scheme leaves a fold with no training or no
            scored rows (raised by the constructor).
        """
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        rows = np.asarray(rows, dtype=int)

        def _subset(array):
            return None if array is None else np.ascontiguousarray(array[rows])

        return FoldScheme(
            weights=np.ascontiguousarray(self.weights[rows]),
            held_out=np.ascontiguousarray(self.held_out[rows]),
            scheme=self.scheme,
            seed=self.seed,
            groups=_subset(self.groups),
            strata=_subset(self.strata),
            ordered=self.ordered,
            judge=_subset(self.judge),
            _note=self._note,
            base_weights=_subset(self.base_weights),
            repeats=self.repeats,
            known_groups=_subset(self.known_groups),
            times=_subset(self.times),
            embargo=self.embargo,
        )

    def check_leakage(self, groups=None, *, raise_on_leak: bool = True) -> dict:
        """Report rows that a fold both trains on and is judged by.

        With no argument this checks the grouping the scheme was built around
        -- for a grouped scheme, that no cluster straddles a split; for a time
        scheme, that no fold trains on rows later than the ones it holds out.

        Pass ``groups`` to ask about a grouping the scheme knows nothing
        about, which is the useful case: a plain k-fold split of repeated
        measurements leaks by subject, and the matrix cannot say so.

        Parameters
        ----------
        groups : array_like, shape (n,), optional
            Group label per row. Defaults to the grouping the scheme was built
            around (``self.groups``).
        raise_on_leak : bool, default True
            Raise when a leak is found. With ``False`` the report is returned
            instead.

        Returns
        -------
        dict
            ``{"scheme": str, "leaks": list of dict, "clean": bool}``. Each
            leak has ``fold``, ``kind`` (``"group"``, ``"time"`` or
            ``"embargo"``) and the groups or row positions involved.

        Raises
        ------
        ValueError
            If ``groups`` does not have one entry per row, or if a leak is
            found and ``raise_on_leak`` is true.
        """
        report: dict = {"scheme": self.scheme, "leaks": []}
        grouping = groups if groups is not None else self.groups
        if grouping is not None:
            grouping = np.asarray(grouping)
            if grouping.shape[0] != len(self):
                raise ValueError(
                    f"groups has length {grouping.shape[0]}, expected {len(self)}"
                )
            in_bag_mask = (self.weights > 0)
            for b in range(self.n_folds):
                out_groups = set(grouping[self.held_out[:, b]].tolist())
                in_groups = set(grouping[in_bag_mask[:, b]].tolist())
                straddling = sorted(out_groups & in_groups, key=repr)
                if straddling:
                    report["leaks"].append(
                        {"fold": b, "kind": "group", "groups": straddling}
                    )
        if self.ordered:
            # Compare timestamps when the scheme has them, row positions when not.
            clock = (np.arange(len(self)) if self.times is None
                     else np.asarray(self.times))
            for b in range(self.n_folds):
                held = self.fold(b)
                trained = np.flatnonzero(self.weights[:, b] > 0)
                if not (held.size and trained.size):
                    continue
                if clock[trained].max() > clock[held].min():
                    report["leaks"].append(
                        {"fold": b, "kind": "time",
                         "trains_on_row": int(trained[np.argmax(clock[trained])]),
                         "judged_from_row": int(held[np.argmin(clock[held])])}
                    )
                elif self.embargo is not None and np.any(
                        clock[trained] > clock[held].min() - self.embargo):
                    report["leaks"].append(
                        {"fold": b, "kind": "embargo",
                         "trains_on_row": int(trained[np.argmax(clock[trained])]),
                         "judged_from_row": int(held[np.argmin(clock[held])])}
                    )
        report["clean"] = not report["leaks"]
        if report["leaks"] and raise_on_leak:
            first = report["leaks"][0]
            if first["kind"] == "group":
                detail = (
                    f"fold {first['fold']} trains on and is judged by "
                    f"{len(first['groups'])} of the same groups, the first "
                    f"being {first['groups'][0]!r}"
                )
            elif first["kind"] == "embargo":
                detail = (
                    f"fold {first['fold']} trains on row "
                    f"{first['trains_on_row']}, which lies inside the embargo "
                    f"before row {first['judged_from_row']} that it is judged on"
                )
            else:
                detail = (
                    f"fold {first['fold']} trains on row "
                    f"{first['trains_on_row']}, which comes after row "
                    f"{first['judged_from_row']} that it is judged on"
                )
            raise ValueError(
                f"{self.scheme}: {len(report['leaks'])} folds leak. {detail}. "
                f"The cross-validated risk of such a split is optimistic: the "
                f"held-out rows are partly predicted by information the model "
                f"already saw. Use grouped_folds(groups=...) to keep each "
                f"group whole, or rolling_folds() for ordered data."
            )
        return report

    def describe(self) -> str:
        """A short text summary: scheme name, fold count, held-out sizes.

        Returns
        -------
        str
            Multi-line description including the seed and the number of
            groups kept whole when the scheme has them.
        """
        sizes = self.fold_sizes()
        lines = [
            f"{self.scheme}: {self.n_folds} folds over {len(self)} rows",
            f"  held out per fold: min {int(sizes.min())}, "
            f"max {int(sizes.max())}",
        ]
        if self.seed is not None:
            lines.append(f"  seed: {self.seed}")
        if self.groups is not None:
            lines.append(f"  groups kept whole: {len(np.unique(self.groups))}")
        if self.known_groups is not None:
            lines.append(
                f"  groups spread across folds (known-group design): "
                f"{len(np.unique(self.known_groups))}")
        if self.times is not None:
            lines.append("  rows ordered by the supplied timestamps")
        if self.embargo is not None:
            lines.append(f"  embargo before each judged block: {self.embargo}")
        if self.strata is not None:
            levels, counts = np.unique(self.strata, return_counts=True)
            lines.append(
                "  strata: "
                + ", ".join(f"{lv}={c}" for lv, c in zip(levels, counts,
                                                         strict=False))
            )
        if self._note:
            lines.append(f"  {self._note}")
        return "\n".join(lines)

    def to_frame(self, output="auto"):
        """One row per (row, fold): whether the row is in-bag and its weight.

        Parameters
        ----------
        output : {"auto", "pandas", "polars", "records"}, default "auto"
            Container for the result; ``"auto"`` is Polars when installed,
            otherwise pandas, otherwise records.

        Returns
        -------
        frame or list of dict
            Columns ``row``, ``fold`` (zero-based), ``weight`` and ``held_out``;
            ``n * B`` rows.
        """
        from mboostrs._output import output_frame

        rows = []
        for b in range(self.n_folds):
            for i in range(len(self)):
                rows.append({
                    "row": i,
                    "fold": b,
                    "weight": float(self.weights[i, b]),
                    "held_out": bool(self.held_out[i, b]),
                })
        return output_frame(rows, output,
                            columns=("row", "fold", "weight", "held_out"))


def _scheme_from_assignment(assignment, base, scheme, **extra) -> FoldScheme:
    """Build a scheme where each row is held out in exactly one fold."""
    assignment = np.asarray(assignment, dtype=np.intp)
    n = assignment.shape[0]
    B = int(assignment.max()) + 1
    held_out = np.zeros((n, B), dtype=bool)
    held_out[np.arange(n), assignment] = True
    held_out &= (base > 0)[:, None]
    weights = np.where(held_out, 0.0, base[:, None])
    return FoldScheme(weights=np.ascontiguousarray(weights),
                      held_out=held_out, scheme=scheme,
                      base_weights=base.copy(), **extra)


def stratified_folds(y, B=10, *, seed=42, weights=None) -> FoldScheme:
    """K-fold splits that preserve the class proportions of ``y``.

    A plain 10-fold split of 200 rows carrying 12 positives leaves folds with
    0, 1, 2 or 3 of them; a fold holding none produces an out-of-fold risk
    that estimates nothing, and the spread across folds measures the split
    rather than the model. Here each class is partitioned separately and the
    pieces are combined, so every fold holds out as near as possible the same
    share of each class.

    ``y`` is the stratifying variable: the response for a classification fit,
    but any label vector works -- a treatment arm, a site.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
    n = y.shape[0]
    base = _base_weights(n, weights)
    in_sample = base > 0
    levels, counts = np.unique(y[in_sample], return_counts=True)
    if levels.size == 0:
        raise ValueError("every row carries zero weight, so there is nothing to fold")
    smallest = int(counts.min())
    B = _check_B(B, smallest, f"members of the smallest class ({levels[int(np.argmin(counts))]!r})")

    rng = np.random.default_rng(seed)
    assignment = np.zeros(n, dtype=np.intp)
    load = np.zeros(B, dtype=np.int64)
    for level in levels:
        # Deal only the in-sample rows of the class. A zero-weight row is out
        # of the sample entirely: it is masked out of every fold's held-out
        # set, so counting it here would shrink B and tilt the balance over
        # rows no fold ever judges.
        rows = np.flatnonzero((y == level) & in_sample)
        rng.shuffle(rows)
        # Deal the class's rows round-robin, but start from the fold that is
        # currently smallest. Dealing every class from fold 0 piles all the
        # remainders onto the same early folds: with 188 negatives and 12
        # positives over ten folds that gives sizes 21, 21, 20 ... 19, 19,
        # where starting from the smallest gives 20 everywhere.
        order = np.argsort(load, kind="stable")
        chosen = order[np.arange(rows.size) % B]
        assignment[rows] = chosen
        np.add.at(load, chosen, 1)
    return _scheme_from_assignment(
        assignment, base, f"stratified {B}-fold", seed=int(seed), strata=y,
    )



def _scalar(value):
    """A NumPy scalar as its Python value; a Python object (a `str` label from
    an object array) as it is."""
    return value.item() if hasattr(value, "item") else value

def grouped_folds(groups, B=10, *, seed=42, weights=None, repeats=1) -> FoldScheme:
    """K-fold splits that keep each group entirely on one side.

    For repeated measurements, clustered sampling, or any design where rows
    within a group share more than the model explains. Splitting such rows at
    random leaves a group's own members in the training data whenever one of
    them is held out, so the held-out row is partly predicted by itself and
    the cross-validated risk is optimistic.

    Groups are dealt to folds largest-first, which keeps the fold sizes close
    when the groups are of unequal size; the seed orders groups of equal size.

    With ``repeats`` greater than one the groups are dealt that many times,
    each deal with the next draw of the seed's generator, and the deals are
    stacked side by side as `repeated_kfold_folds` stacks row partitions: every
    group is held out once in each deal. With few groups, which groups share a
    fold changes a comparison by as much as the difference between two
    specifications; scoring over several deals averages that out.
    `compare_specifications` and `term_report` compute their paired standard
    errors within each deal, since the deals reuse the same groups.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError(f"groups must be one-dimensional, got shape {groups.shape}")
    n = groups.shape[0]
    base = _base_weights(n, weights)
    in_sample = base > 0
    if not in_sample.any():
        raise ValueError("every row carries zero weight, so there is nothing to fold")
    labels, full_sizes = np.unique(groups, return_counts=True)
    effective = {_scalar(lab): int(sz) for lab, sz in
                 zip(*np.unique(groups[in_sample], return_counts=True),
                     strict=False)}
    if not effective:
        raise ValueError("every row carries zero weight, so there is nothing to fold")
    B = _check_B(B, len(effective), "distinct groups")

    # Size each group by its in-sample membership only: a group whose members
    # all carry zero weight trains on nothing and is judged on nothing, so it
    # must neither consume a fold of budget nor tilt the load balancing.
    eff_sizes = np.array([effective.get(_scalar(lab), 0) for lab in labels],
                         dtype=np.int64)
    repeats = check_int_at_least("repeats", repeats, 1, allow_none=False)
    assert repeats is not None
    rng = np.random.default_rng(seed)
    deals = []
    for _ in range(repeats):
        order = rng.permutation(labels.size)
        order = order[np.argsort(-eff_sizes[order], kind="stable")]

        load = np.zeros(B, dtype=np.int64)
        fold_of_group: dict = {}
        for index in order:
            target = int(np.argmin(load))
            fold_of_group[_scalar(labels[index])] = target
            load[target] += int(eff_sizes[index])

        assignment = np.array([fold_of_group[_scalar(g)] for g in groups],
                              dtype=np.intp)
        deals.append(_scheme_from_assignment(
            assignment, base, f"grouped {B}-fold", seed=int(seed), groups=groups,
        ))
    if repeats == 1:
        return deals[0]
    return FoldScheme(
        weights=np.ascontiguousarray(np.hstack([deal.weights for deal in deals])),
        held_out=np.hstack([deal.held_out for deal in deals]),
        scheme=f"{repeats}x repeated grouped {B}-fold", seed=int(seed), groups=groups,
        _note=f"{B * repeats} columns: every group is held out {repeats} times",
        base_weights=base.copy(), repeats=repeats,
    )


def blocked_folds(n, B=5, *, weights=None) -> FoldScheme:
    """Split rows into contiguous held-out blocks.

    For data with local dependence -- a time series, a spatial transect --
    where neighbouring rows are nearly the same observation. A random split
    holds out a row whose immediate neighbours are in the training data, which
    for smooth serial dependence is close to holding out nothing.

    Training may include rows after a held-out block, so these folds measure
    prediction across local blocks, not forecasting into future periods.
    Use :func:`rolling_folds` to evaluate forecasting.
    """
    n = int(n)
    base = _base_weights(n, weights)
    B = _check_B(B, n, "rows")
    # `array_split` gives blocks differing by at most one row.
    assignment = np.concatenate([
        np.full(block.size, b, dtype=np.intp)
        for b, block in enumerate(np.array_split(np.arange(n), B))
    ])
    return _scheme_from_assignment(
        assignment, base, f"blocked {B}-fold", ordered=True,
        _note=("trains on rows after the block it is judged on; use "
               "rolling_folds for forecasting"),
    )


def loo_folds(n, *, weights=None) -> FoldScheme:
    """Leave-one-out: fold ``i`` holds out row ``i`` alone (``n`` folds).

    mboost's ``cv(type = "kfold", B = n)``. Every row is judged by a fit to
    all the others, so the estimate of prediction error has little bias, at
    the cost of ``n`` fits -- one per row -- and a high variance between
    neighbouring rows' contributions. A row with zero model weight is out of
    the sample and is held out by no fold. For grouped data use
    :func:`grouped_folds`, which leaves out whole groups.
    """
    n = int(n)
    base = _base_weights(n, weights)
    if n < 2:
        raise ValueError(f"leave-one-out needs at least 2 rows, got {n}")
    if n > 5000:
        warnings.warn(
            f"leave-one-out over {n} rows costs {n} fits; k-fold with "
            "B=10 usually estimates the same risk at a hundredth of the cost",
            RuntimeWarning, stacklevel=2)
    # Rows with zero model weight are out of the sample: no fold for them.
    in_sample = np.flatnonzero(base > 0)
    if in_sample.size < 2:
        raise ValueError("leave-one-out needs at least 2 rows with positive weight")
    assignment = np.full(n, -1, dtype=np.intp)
    assignment[in_sample] = np.arange(in_sample.size)
    return _scheme_from_assignment(assignment, base, f"leave-one-out ({in_sample.size} folds)")


def _check_embargo(embargo, times):
    """The embargo in the units of `times`, or None when there is none."""
    if embargo is None:
        return None
    if times is None:
        raise ValueError(
            "embargo needs time=: a gap is measured in timestamps, and row "
            "positions do not give one. Pass the timestamps as time=, or drop "
            "embargo=")
    if np.issubdtype(times.dtype, np.datetime64):
        if not isinstance(embargo, np.timedelta64):
            raise ValueError(
                "embargo for datetime timestamps must be a numpy timedelta64, "
                "such as np.timedelta64(7, 'D')")
        value = embargo
        if value < np.timedelta64(0, "s"):
            raise ValueError("embargo must be non-negative")
        return value
    if isinstance(embargo, np.timedelta64):
        raise ValueError("embargo for numeric timestamps must be a number in the same units as time")
    value = float(embargo)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"embargo must be a finite, non-negative number, got {embargo!r}")
    return value


def rolling_folds(n, B=5, *, horizon=None, initial=None, expanding=True,
                  weights=None, time=None, embargo=None) -> FoldScheme:
    """Rolling-origin splits: train on the past, judge on what comes next.

    Fold ``b`` trains on rows up to an origin and is judged on the ``horizon``
    rows after it, so no fold ever sees a row later than the ones it predicts.
    This is the only scheme here that measures forecasting.

    ``expanding=True`` grows the training window from the first row each time;
    ``expanding=False`` slides a window of fixed length, which is what to use
    when the relationship itself drifts and old rows should stop counting.

    ``horizon`` defaults to an even division of the rows after ``initial``;
    ``initial`` defaults to half the rows. Both count rows in time order.

    ``time`` gives each row a timestamp, numeric or ``datetime64``. The rows
    are then ordered by it (ties keep their array order), so the data need not
    be sorted and "before" means "earlier in time", not "earlier in the
    array". Without ``time`` the array order is the time order.

    ``embargo`` is a minimum gap between the last training timestamp and the
    first judged timestamp of each fold. Training rows inside the gap are
    dropped from that fold and are neither trained on nor judged. It needs
    ``time``: a gap is measured in timestamps, not in row positions. Numeric
    timestamps take a number in their own units; ``datetime64`` timestamps
    take a ``numpy.timedelta64``, such as ``np.timedelta64(7, "D")``.
    """
    n = int(n)
    base = _base_weights(n, weights)
    B = _check_B(B, n, "rows")
    initial = check_int_at_least("initial", initial, 1, allow_none=False) \
        if initial is not None else max(1, n // 2)
    assert initial is not None
    if initial >= n:
        raise ValueError(
            f"initial={initial} leaves no rows to judge on out of {n}"
        )
    horizon = check_int_at_least("horizon", horizon, 1, allow_none=False) \
        if horizon is not None else max(1, (n - initial) // B)
    assert horizon is not None
    if initial + B * horizon > n:
        raise ValueError(
            f"initial={initial} plus {B} folds of horizon={horizon} needs "
            f"{initial + B * horizon} rows but only {n} are available. Lower "
            f"B, lower horizon, or lower initial."
        )

    if time is None:
        times = None
        order = np.arange(n)
    else:
        times = np.asarray(time)
        if times.shape != (n,):
            raise ValueError(
                f"time has shape {times.shape}, expected ({n},) to match the sample")
        if np.issubdtype(times.dtype, np.datetime64):
            if np.any(np.isnat(times)):
                raise ValueError(
                    "time contains missing timestamps (NaT); drop those rows or "
                    "give them a timestamp before folding")
        else:
            times = times.astype(np.float64)
            if not np.all(np.isfinite(times)):
                raise ValueError("time must be finite: drop or impute the missing timestamps")
        order = np.argsort(times, kind="stable")
    gap = _check_embargo(embargo, times)

    # Positions below are time-ordered; `order` maps them back to rows.
    base_sorted = base[order]
    held_sorted = np.zeros((n, B), dtype=bool)
    fold_sorted = np.zeros((n, B), dtype=np.float64)
    for b in range(B):
        origin = initial + b * horizon
        start = 0 if expanding else b * horizon
        fold_sorted[start:origin, b] = base_sorted[start:origin]
        held_sorted[origin:origin + horizon, b] = True
    held_sorted &= (base_sorted > 0)[:, None]
    fold_sorted[held_sorted] = 0.0
    if gap is not None:
        times_sorted = times[order]
        for b in range(B):
            judged = held_sorted[:, b]
            if not judged.any():
                continue
            first_judged = times_sorted[judged].min()
            inside_gap = times_sorted > first_judged - gap
            fold_sorted[inside_gap, b] = 0.0

    held_out = np.zeros((n, B), dtype=bool)
    fold_weights = np.zeros((n, B), dtype=np.float64)
    held_out[order] = held_sorted
    fold_weights[order] = fold_sorted
    window = "expanding" if expanding else "sliding"
    if times is None:
        ordering = ""
    else:
        ordering = ", in the supplied time order"
        if gap is not None:
            ordering += f", embargo {gap}"
    return FoldScheme(
        weights=np.ascontiguousarray(fold_weights), held_out=held_out,
        judge=held_out,
        scheme=f"rolling-origin {B}-fold ({window} window{ordering})", ordered=True,
        _note=f"initial={initial}, horizon={horizon}",
        base_weights=base.copy(),
        times=times, embargo=gap,
    )


def known_group_folds(groups, B=10, *, seed=42, weights=None) -> FoldScheme:
    """K-fold splits in which every held-out row has a group-mate in training.

    The complement of :func:`grouped_folds`. Rows are dealt to folds one group
    at a time, so the members of one group land in different folds, and each
    held-out row keeps at least one group-mate among its fold's training rows.
    The score then measures prediction for a group the model has already
    observed, the setting of a `brandom` or `bmrf` effect estimated for clusters
    that are in the data. :func:`grouped_folds` measures prediction for a
    group the model has not seen.

    Each group needs at least two rows with positive weight, so that a
    held-out row has a training sibling; a group with one row is refused.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError(f"groups must be one-dimensional, got shape {groups.shape}")
    n = groups.shape[0]
    base = _base_weights(n, weights)
    in_sample = base > 0
    if not in_sample.any():
        raise ValueError("every row carries zero weight, so there is nothing to fold")
    levels, counts = np.unique(groups[in_sample], return_counts=True)
    if counts.min() < 2:
        raise ValueError(
            f"known-group folds need at least two rows with positive weight in "
            f"every group; group {_scalar(levels[int(np.argmin(counts))])!r} has one. "
            f"Drop that row, or use grouped_folds for a group seen only once")
    B = _check_B(B, int(in_sample.sum()), "rows")

    rng = np.random.default_rng(seed)
    assignment = np.zeros(n, dtype=np.intp)
    load = np.zeros(B, dtype=np.int64)
    for level in levels:
        rows = np.flatnonzero((groups == level) & in_sample)
        rng.shuffle(rows)
        # Same round-robin dealing as stratified_folds: a group's rows go to
        # distinct folds while the group has no more rows than there are folds.
        order = np.argsort(load, kind="stable")
        chosen = order[np.arange(rows.size) % B]
        assignment[rows] = chosen
        np.add.at(load, chosen, 1)
    return _scheme_from_assignment(
        assignment, base, f"known-group {B}-fold", seed=int(seed),
        known_groups=groups,
    )


def repeated_kfold_folds(n, k=10, repeats=5, *, seed=42,
                         weights=None) -> FoldScheme:
    """``repeats`` independent k-fold partitions, side by side.

    One k-fold split has a variance of its own: the risk it reports depends on
    which partition was drawn, and with a small sample that dependence can be
    larger than the difference between two models. Repeating the split and
    averaging over partitions reduces it. The columns are all ``k * repeats``
    folds, so the aggregate is over every fold of every repeat.
    """
    n = int(n)
    base = _base_weights(n, weights)
    k = _check_B(k, n, "rows")
    repeats = check_int_at_least("repeats", repeats, 1, allow_none=False)
    assert repeats is not None

    rng = np.random.default_rng(seed)
    columns = []
    held = []
    for _ in range(repeats):
        rows = rng.permutation(n)
        assignment = np.empty(n, dtype=np.intp)
        assignment[rows] = np.arange(n) % k
        block = np.zeros((n, k), dtype=bool)
        block[np.arange(n), assignment] = True
        block &= (base > 0)[:, None]
        held.append(block)
        columns.append(np.where(block, 0.0, base[:, None]))
    return FoldScheme(
        weights=np.ascontiguousarray(np.hstack(columns)),
        held_out=np.hstack(held),
        scheme=f"{repeats}x repeated {k}-fold", seed=int(seed),
        _note=f"{k * repeats} columns: every row is held out {repeats} times",
        base_weights=base.copy(), repeats=repeats,
    )


def warn_if_leaky(folds, groups, *, scheme_name="the supplied folds") -> None:
    """Warn when a fold matrix splits ``groups`` across the two sides.

    For a plain matrix, which carries no record of what it respects.
    """
    matrix = np.asarray(folds, dtype=np.float64)
    groups = np.asarray(groups)
    straddling = 0
    for b in range(matrix.shape[1]):
        in_bag = set(groups[matrix[:, b] > 0].tolist())
        out = set(groups[matrix[:, b] == 0].tolist())
        straddling += len(in_bag & out)
    if straddling:
        warnings.warn(
            f"{scheme_name} split {straddling} group memberships across the "
            f"training and held-out sides. The held-out rows are then partly "
            f"predicted by their own group-mates, so the cross-validated risk "
            f"is optimistic. Build folds with "
            f"mboostrs.grouped_folds(groups, B=...) to keep each group whole.",
            RuntimeWarning,
            stacklevel=2,
        )
