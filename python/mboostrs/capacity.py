"""Select model capacity by comparing additive, tensor and tree specifications.

Every candidate is scored on shared folds, and a larger model is selected
only when its held-out improvement clears the specified standard-error margin.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from mboostrs._frame import column_names, read_column
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._output import export_result
from mboostrs.explore import FoldError, cv_risk, split_terms, term_variable
from mboostrs.formula import expand_formula
from mboostrs.terms import TENSOR_MARGINAL_DF, pearson_residual, screen_interactions

__all__ = ["CapacitySelection", "select_capacity"]

_BASE = "base"
_TENSORS = "+tensors"
_TREE = "+tree"


def _plain_response(lhs: str) -> str:
    """The response column name, refusing a composed left-hand side."""
    response = lhs.strip()
    if "(" in response or ")" in response or "," in response:
        raise ValueError(
            f"select_capacity needs a plain response column, got {response!r}; "
            "a composed response such as Surv() is not supported")
    if not response:
        raise ValueError("the formula has no left-hand side")
    return response


def _numeric_columns(rhs, data, response):
    """Numeric data columns the formula's terms read, in first-appearance order."""
    present = set(column_names(data))
    seen, out = set(), []
    for term in split_terms(rhs):
        variable = term_variable(term)
        if (variable is None or variable in seen or variable == response
                or variable not in present):
            continue
        seen.add(variable)
        if read_column(data, variable).kind == "numeric":
            out.append(variable)
    return out


def _formula_columns(rhs, data, response):
    """Data columns the formula's terms read, of any kind, in first-appearance
    order. A joint tree splits on these, and a factor column can be part of
    an interaction that a numeric-only tree cannot express."""
    present = set(column_names(data))
    seen, out = set(), []
    for term in split_terms(rhs):
        variable = term_variable(term)
        if (variable is None or variable in seen or variable == response
                or variable not in present):
            continue
        seen.add(variable)
        out.append(variable)
    return out


def _distinct_values(data, name) -> int:
    """Distinct finite values of a numeric column."""
    values = np.asarray(read_column(data, name).values, dtype=float)
    return int(np.unique(values[np.isfinite(values)]).size)


def _tensor_term(left: str, right: str) -> str:
    return (f"bbs({quote_name(left)}, df={TENSOR_MARGINAL_DF}) %X% "
            f"bbs({quote_name(right)}, df={TENSOR_MARGINAL_DF})")


def _tree_term(columns, depth: int) -> str:
    return f"btree({', '.join(quote_name(_c) for _c in columns)}, maxdepth={depth}, selection=sse)"


def _fold_se(fold_scores) -> float:
    """The standard error of the equal-fold mean score."""
    scores = np.asarray(fold_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size < 2:
        return 0.0
    return float(np.std(scores, ddof=1) / np.sqrt(scores.size))


def _paired_se(row, best):
    """The standard error of the paired fold difference ``row - best``, or
    None when the two rungs' fold scores cannot be paired.

    Every rung is scored on the same folds. The standard error of the
    per-fold differences reflects variation in the comparison while
    accounting for variation shared by the folds.
    """
    left, right = row.get("fold_scores"), best.get("fold_scores")
    if not left or not right:
        return None
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    difference = difference[np.isfinite(difference)]
    if difference.size < 2:
        return None
    return float(np.std(difference, ddof=1) / np.sqrt(difference.size))


#: Shares of the current grid end a budget search scores: a geometric ladder
#: from one thirty-second of the end to the end itself, so a doubling grid
#: keeps its resolution and never scores more than seven points per call.
_BUDGET_SHARES = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0)


def _budget_ladder(end: int) -> list[int]:
    """The step counts a budget search scores for a grid that ends at ``end``."""
    return sorted({max(1, round(end * share)) for share in _BUDGET_SHARES})


def _fold_count(folds) -> int:
    """How many folds the shared partition splits the rows into."""
    if isinstance(folds, (int, np.integer)):
        return int(folds)
    return int(np.asarray(folds, dtype=float).shape[1])


def _tuned_budget(rhs, data, response, *, family, folds, seed, nu, score, mstop,
                  max_mstop, extrapolation, fit_kwargs):
    """One rung's held-out score at the selected step count.

    The budget is searched, not fixed: the grid starts at ``mstop`` steps and
    doubles while the rung's held-out score is still falling at its largest
    point, up to ``max_mstop``, and every point is scored by `cv_risk`'s
    step-count grid -- one fit per fold per grid end, on the same folds every
    rung shares. The returned score is the rung's best point of the grid, the
    value ``cv_risk(mstop=m)`` returns there, so tuned and shared rungs stay
    comparable. Returns ``(score, fold_se, fold_scores, mstop, at_boundary)``
    -- ``fold_scores`` holds each fold's score at that point, in fold order,
    used to compare rungs -- or ``None`` when no point could be scored and
    the caller must fall back to the shared single-budget evaluation, which
    records the fold failures.
    """
    scored: dict[int, tuple[float, float | None, tuple[float, ...]]] = {}
    end = int(mstop)
    while True:
        frame = cv_risk(rhs, data, response, family=family, folds=folds, nu=nu,
                        seed=seed, score=score, extrapolation=extrapolation,
                        mstop_grid=[m for m in _budget_ladder(end) if m not in scored],
                        **fit_kwargs)
        records = (frame.to_dicts() if hasattr(frame, "to_dicts")
                   else frame.to_dict("records"))
        for row in records:
            scored[int(row["mstop"])] = (
                float(row["risk"]),
                None if row["fold_se"] is None else float(row["fold_se"]),
                tuple(float(v) for v in row.get("fold_scores", ())))
        finite = {m: values for m, values in scored.items() if np.isfinite(values[0])}
        if not finite:
            return None
        best = min(finite, key=lambda m: finite[m][0])
        # The score still falling at the largest point means the rung has not
        # finished using the grid; double the end and score the added points.
        if best < end or end >= max_mstop:
            return (*finite[best], best, best == end and end >= max_mstop)
        end = min(2 * end, max_mstop)


def _tuned_unsupported(family, fit_kwargs):
    """Why the step-count grid cannot score this call, or None when it can.

    `cv_risk`'s grid route fits one model per fold and cuts its path, which
    needs a standard single-response family, ``na_action="error"`` and no
    per-row offset; outside that set the rungs keep the shared budget.
    """
    from mboostrs.families import Multinomial

    if hasattr(family, "param_names"):
        return f"{type(family).__name__} is a distributional family"
    if isinstance(family, Multinomial):
        return "Multinomial is a multi-column response family"
    if fit_kwargs.get("na_action") not in (None, "error"):
        return f"na_action={fit_kwargs['na_action']!r}"
    offset = fit_kwargs.get("offset")
    if offset is not None and np.ndim(offset) > 0:
        return "a per-row offset"
    return None


def _evaluate(rhs, data, response, *, family, folds, seed, mstop, nu, score,
              extrapolation, fit_kwargs, estimator="boosting"):
    """The out-of-fold score of one rung, on the shared fold partition."""
    return cv_risk(rhs, data, response, family=family, folds=folds, mstop=mstop,
                   nu=nu, seed=seed, score=score, extrapolation=extrapolation,
                   estimator=estimator, return_result=True, retain_arrays=False,
                   **fit_kwargs)


@dataclass(frozen=True)
class CapacitySelection:
    """The capacity ladder :func:`select_capacity` measured, and what it kept.

    ``rungs`` is one record per estimator/specification evaluated, in increasing
    capacity: ``label`` (``"intercept"``, ``"base"``, ``"+tensors"``,
    ``"+tree"``), ``estimator``, ``rhs``,
    ``formula``, the held-out ``score``, its marginal standard error
    ``fold_se`` across folds, ``fold_scores`` (each fold's score at the step
    count the rung was scored at, in fold order -- the paired differences
    against the best rung are what ``tolerance`` reads), ``n_folds``, the
    ``mstop`` the rung was scored at (its tuned step count, the shared budget
    for ``budget="shared"``, or ``None`` for a direct REML fit, which has no
    step count), whether the rung was ``selected``, and a ``note``. With more
    than one ``tree_depth`` there is one ``"+tree"`` record per depth, in the
    order given.
    ``screened_pairs`` lists the pairs the residual screen proposed (the ones
    with a positive held-out gain, up to ``pairs=``), each with the mean
    squared error it removed out of sample.

    ``model`` is the selected estimator refitted on all rows. A boosting fit
    uses its rung's tuned step count under ``budget="tuned"`` and the
    comparison's ``mstop`` under ``budget="shared"``; a direct REML fit has no
    boosting step count. ``selected_estimator`` identifies the fitted
    estimator.
    """

    #: The label of the rung kept: ``"base"``, ``"+tensors"`` or ``"+tree"``.
    selected: str
    #: The selected rung's formula.
    formula: str
    #: The selected rung's right-hand side.
    rhs: str
    #: Standard errors of each smaller rung's paired fold difference against
    #: the best rung -- ``tolerance`` of which the kept rung may sit above
    #: the best one.
    tolerance: float
    #: One record per rung evaluated: ``label``, ``rhs``, ``formula``,
    #: ``score``, ``fold_se``, ``n_folds``, ``selected``, ``note``.
    rungs: tuple[dict[str, Any], ...]
    #: The pairs the screen proposed: ``left``, ``right``, ``removed``.
    screened_pairs: tuple[dict[str, Any], ...]
    #: The selected form refitted on all rows with its selected estimator.
    model: object = field(repr=False, compare=False)
    #: Estimator used by the selected candidate; legacy instances are boosting.
    selected_estimator: str = "boosting"

    @property
    def variables(self):
        """The data columns the selected formula reads, in first-seen order.

        A ``%X%`` term names both of its margins."""
        out: list[str] = []
        for term in split_terms(self.rhs):
            for margin in term.split("%X%"):
                name = term_variable(margin)
                if name is not None and name not in out:
                    out.append(name)
        return out

    def to_dict(self, *, include_arrays: bool = False):
        """A JSON-safe record; the model itself is not exported."""
        mstop = getattr(self.model, "mstop", None)
        return export_result(
            self, include_arrays=include_arrays,
            extras={"mstop": (None if mstop is None else
                              int(mstop) if not isinstance(mstop, dict) else mstop)},
        )


def select_capacity(formula, data, family=None, *, folds=5, seed=0, mstop=500,
                    nu=0.1, pairs=3, tree=True, tree_depth=3, budget="tuned",
                    max_mstop=None, tolerance=2.0, score="family",
                    extrapolation="warn", estimators=("boosting",),
                    **fit_kwargs):
    """Extend a formula with interaction capacity only where it pays.

    Three specifications are built on top of ``formula`` and scored out of
    fold on one shared partition:

    ``base``
        The formula as given.
    ``+tensors``
        The base plus one penalized tensor per interaction pair the residual
        screen proposed. The screen bins half the rows, takes cell means of
        the base fit's Pearson residual there and measures the mean squared
        error those means remove on the other half, so a pair that fits only
        noise removes nothing out of sample. A pair is proposed only when it
        removes a positive amount, and at most ``pairs`` of them are taken,
        best first. Each pair enters as
        ``bbs(a, df=2.5) %X% bbs(b, df=2.5)`` (``TENSOR_MARGINAL_DF``; squared
        it clears the tensor penalty's null space of 4, see the ``pitfalls``
        guide topic). A column with two or fewer distinct values cannot be a
        spline margin -- its boundary knots collapse and the fit refuses --
        so pairs over such columns are never proposed.
    ``+tree``
        The ``+tensors`` rung plus one joint tree, ``btree(..., maxdepth=
        tree_depth, selection=sse)``, over every column the base formula
        reads, factors included: a factor can be the half of an interaction
        no numeric-only tree can express, which is how the bike panels'
        hour-by-working-day structure hid from the first chooser. The tree
        can split on several columns at once, so it stands for interaction
        structure that the pair screen did not nominate.

    The rungs are nested: ``base`` is inside ``+tensors``, which is inside
    ``+tree``. By default every boosting rung is searched for its own best
    step count out of fold and scored there (``budget="tuned"``): the grid
    starts at ``mstop`` steps and doubles while the rung's held-out score is
    still falling at its largest point, up to ``max_mstop`` (four times
    ``mstop`` by default), and the rung is scored at the best point of that
    grid, by :func:`~mboostrs.cv_risk`, which shares one fold partition
    across calls with the same ``seed``, ``folds`` and row count -- so the
    scores are paired fold by fold. With ``budget="shared"``, every rung
    is fitted and scored at
    the same ``mstop`` and ``nu``, and a rung that needs more steps than
    that to show its advantage is not adopted.

    The retained rung is the smallest whose mean score is within
    ``tolerance`` standard errors of the best rung's mean. The standard
    error is calculated from paired fold differences on the shared split.
    If a difference cannot be paired (one fold or a failed fold), the best
    rung's marginal fold spread is used. A larger rung is retained only when
    its held-out score improves enough to satisfy this rule.

    Parameters
    ----------
    formula : str
        The base formula, ``"y ~ bbs(x1) + bbs(x2) + f"``. Bare columns,
        penalized splines and factor terms are all accepted; the rungs only
        add to it. The response must be a plain column.
    data : polars or pandas DataFrame, or a mapping of column to values
        The training rows.
    family : Family, optional
        The family of every fit; ``Gaussian()`` when omitted.
    folds : int, ndarray or FoldScheme, optional
        The fold partition, passed to :func:`~mboostrs.cv_risk`. Random folds
        favour the ``+tree`` rung on time-ordered rows -- a tree can memorize
        the neighbours of a held-out row -- so pass a scheme from
        ``mboostrs.folds`` for grouped or time-ordered data, as the
        ``modeling-tactics`` guide topic requires of any reference
        comparison.
    seed : int
        Seed of the default random folds, and of the screen's half-split.
    mstop : int
        Under ``budget="tuned"`` (the default), the first grid end of every
        rung's step-count search, which doubles from there while the rung's
        score still falls. Under ``budget="shared"``, the step count every
        rung is fitted to, an equal budget, and a rung that needs more steps
        than this to lower held-out risk enough to meet ``tolerance`` is not
        adopted.
    nu : float
        Step length of every fit.
    pairs : int
        How many screened pairs at most enter the ``+tensors`` rung. ``0``
        skips the rung.
    tree : bool, default True
        Whether to build and score the ``+tree`` rung.
    tree_depth : int or sequence of int
        ``maxdepth`` of the joint tree. A sequence, such as ``(3, 6)``,
        scores one ``+tree`` rung per depth and keeps the smallest depth
        within ``tolerance`` of the best score.
    budget : {"tuned", "shared"}, default "tuned"
        Whether each boosting rung is scored at its own out-of-fold-tuned
        step count ("tuned") or every rung at the same ``mstop`` ("shared").
        Tuning runs on `cv_risk`'s step-count grid, which needs a standard
        single-response family, ``na_action="error"`` and no per-row offset;
        a call outside that set is scored at the shared budget and says so
        in one warning.
    max_mstop : int, optional
        The largest step count a ``budget="tuned"`` search may reach; four
        times ``mstop`` by default. A rung whose score is still falling there
        is scored and fitted at the cap and the result warns that its
        advantage may be understated.
    tolerance : float
        How many standard errors, of each smaller rung's paired fold
        difference against the best rung's mean, that rung may sit above
        the best and still be kept. The default ``2.0`` keeps only what
        clearly pays, the same two-standard-error rule the ``modeling-tactics``
        guide topic states for a measured gap. Pairing uses the same folds
        for every rung and removes fold variation shared by the two scores.
        ``tolerance=1.0`` gives the one-standard-error rule.
    estimators : sequence of {"boosting", "reml"}
        Estimators compared for the base and tensor forms, in tie-break order.
        The default ``("boosting",)`` preserves the original ladder. Requesting
        ``"reml"`` adds direct Gaussian `mb.gam` candidates and an
        intercept-only boosted baseline. Trees remain boosting candidates.
    score : {"family", "mse", "brier", "log_loss", "pinball", "expectile"}
        The held-out score, passed to :func:`~mboostrs.cv_risk`. Every rung
        is scored on one target, so the comparison is on one scale.
    extrapolation : {"warn", "allow", "clip", "error"}
        The out-of-range policy for fold predictions.
    **fit_kwargs
        Passed to every fit (``weights``, ``na_action`` and the rest).

    Returns
    -------
    CapacitySelection
        The selected form and estimator, the candidate table with scores,
        standard errors and the step count each boosting rung was scored at,
        the screened pairs, and the selected model refitted on all rows.
        A boosting fit uses its rung's tuned step count; REML estimates its
        smoothness.

    Raises
    ------
    ValueError
        When the formula's left-hand side is not a plain column, the base
        rung's cross-validation failed on every fold, or no rung beyond the
        base was requested or buildable.

    Examples
    --------
    A smooth-times-step interaction, ``2 sin(x1) 1(x2 > 0)``: the tensor of
    the ``+tensors`` rung cannot make the step in its ``x2`` margin, the
    joint tree makes it in two splits, so the tree rung is the one that pays:

    >>> import numpy as np, polars as pl, mboostrs as mb
    >>> rng = np.random.default_rng(3)
    >>> n = 800
    >>> x1, x2, x3 = (rng.uniform(-2, 2, n) for _ in range(3))
    >>> data = pl.DataFrame({
    ...     "x1": x1, "x2": x2, "x3": x3,
    ...     "y": 2.0 * np.sin(x1) * (x2 > 0) + 0.4 * x3
    ...          + 0.3 * rng.normal(size=n)})
    >>> pick = mb.select_capacity("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
    ...                           mstop=300)
    >>> pick.selected
    '+tree'

    A smooth response, where the extra capacity does not pay and the base is
    kept:

    >>> y = np.sin(data["x1"].to_numpy()) + data["x2"].to_numpy() ** 2
    >>> smooth = data.with_columns(y=y + 0.4 * rng.normal(size=n))
    >>> pick = mb.select_capacity("y ~ bbs(x1, df=8) + bbs(x2, df=6) + bbs(x3)",
    ...                           smooth, mstop=300)
    >>> pick.selected
    'base'
    """
    import mboostrs as mb

    estimators = tuple(estimators)
    if not estimators or any(x not in ("boosting", "reml") for x in estimators):
        raise ValueError("estimators must contain 'boosting' and/or 'reml'")
    if len(set(estimators)) != len(estimators):
        raise ValueError("estimators must not contain duplicates")
    if "reml" in estimators:
        from mboostrs.families import Gaussian
        if not isinstance(family or Gaussian(), Gaussian):
            raise ValueError("estimator='reml' currently supports Gaussian only")
        if score not in ("family", "mse"):
            raise ValueError("estimator='reml' supports score='family' or 'mse'")
        if set(fit_kwargs) - {"weights", "offset", "na_action"}:
            raise ValueError("estimator='reml' supports only weights, offset, and na_action fit options")
    if pairs < 0:
        raise ValueError("pairs must be at least 0")
    if budget not in ("tuned", "shared"):
        raise ValueError(f"budget must be 'tuned' or 'shared', got {budget!r}")
    if isinstance(tree_depth, (int, np.integer)) and not isinstance(tree_depth, bool):
        depths = (int(tree_depth),)
    else:
        try:
            raw = list(tree_depth)
        except TypeError:
            raw = None
        if (raw is None or not raw
                or any(float(depth) != int(depth) for depth in raw)):
            raise ValueError(
                "tree_depth must be a positive whole depth or a sequence of them")
        depths = tuple(int(depth) for depth in raw)
    if any(depth < 1 for depth in depths):
        raise ValueError("tree_depth must be at least 1")
    if len(set(depths)) != len(depths):
        raise ValueError("tree_depth must not name a depth twice")
    if max_mstop is None:
        max_mstop = 4 * int(mstop)
    if int(max_mstop) != max_mstop or max_mstop < mstop:
        raise ValueError(f"max_mstop must be a whole number at least mstop={mstop}")
    max_mstop = int(max_mstop)
    if not tree and pairs == 0 and estimators == ("boosting",):
        raise ValueError(
            "tree=False and pairs=0 leave no rung to compare against the base")

    expanded = expand_formula(formula, data)
    lhs, _, rhs = partition_tilde(expanded)
    response = _plain_response(lhs)
    columns = _numeric_columns(rhs, data, response)
    tree_columns = _formula_columns(rhs, data, response)
    family = family if family is not None else mb.Gaussian()
    reason = _tuned_unsupported(family, fit_kwargs) if budget == "tuned" else None
    if reason is not None:
        warnings.warn(
            f"select_capacity: budget='tuned' cannot search the step counts here "
            f"({reason}); the rungs are compared at the shared mstop={int(mstop)} "
            f"instead. Pass budget='shared' to silence this warning.",
            RuntimeWarning, stacklevel=2)
    tuned = budget == "tuned" and reason is None

    # The ladder: nested rungs of increasing capacity.
    plan = [(_BASE, rhs, "the formula as given")]
    screened, survivors = [], []
    if pairs > 0 and len(columns) >= 2:
        base_fit = mb.gamboost(f"{quote_response(response)} ~ {rhs}", data, family=family,
                               control=mb.boost_control(mstop=mstop, nu=nu),
                               **fit_kwargs)
        residual = pearson_residual(base_fit, data, response)
        # A column with two or fewer distinct values cannot be a spline
        # tensor margin: its boundary knots collapse and the fit refuses
        # a screened pair with a 0/1 column, so such columns are screened out.
        marginable = [c for c in columns if _distinct_values(data, c) > 2]
        screened = screen_interactions(
            data, residual, response=response, seed=seed,
            pairs=[(a, b) for a, b in combinations(marginable, 2)]
            if len(marginable) >= 2 else [])
        survivors = [p for p in screened if p.removed > 0][:pairs]
    if survivors:
        tensor_rhs = rhs + " + " + " + ".join(
            _tensor_term(p.left, p.right) for p in survivors)
        note = ("screened on the base fit's Pearson residual; removed MSE "
                + ", ".join(f"{p.left} x {p.right}: {p.removed:+.3g}"
                            for p in survivors))
        plan.append((_TENSORS, tensor_rhs, note))
    if tree and tree_columns and "boosting" in estimators:
        # Every depth builds on the same +tensors form; a deeper tree is the
        # larger capacity, so the depths keep the order the caller gave.
        tree_base = plan[-1][1]
        for depth in depths:
            plan.append((_TREE, tree_base + " + " + _tree_term(tree_columns, depth),
                         f"one joint tree over {', '.join(tree_columns)} "
                         f"(maxdepth={depth})"))
    if len(plan) == 1 and estimators == ("boosting",):
        why = ("no column in the formula for a tree"
               if tree else "no screened pair removed held-out variation")
        raise ValueError(f"no rung beyond the base could be built: {why}")

    # Every rung on the same folds; a tuned rung is scored at the step count
    # its own out-of-fold score turns at, a shared one at the given mstop.
    if "reml" in estimators:
        plan.insert(0, ("intercept", "1", "intercept-only boosting baseline"))
    rows = []
    for label, rung_rhs, note in plan:
        candidate_estimators = ("boosting",) if label == "intercept" else estimators
        for estimator in candidate_estimators:
            if label == "+tree" and estimator != "boosting":
                continue
            row = {"label": label, "estimator": estimator, "rhs": rung_rhs,
                   "formula": f"{quote_response(response)} ~ {rung_rhs}",
                   "selected": False, "note": note}
            found = None
            if tuned and estimator == "boosting" and label != "intercept":
                # The intercept rung takes no boosting step, so it has no
                # budget to tune; the shared evaluation scores it as before.
                try:
                    found = _tuned_budget(rung_rhs, data, response, family=family,
                                          folds=folds, seed=seed, nu=nu, score=score,
                                          mstop=mstop, max_mstop=max_mstop,
                                          extrapolation=extrapolation,
                                          fit_kwargs=fit_kwargs)
                except FoldError:
                    # A fold the grid route could not score: the shared
                    # evaluation records the failure instead of raising past
                    # it, as it always did.
                    found = None
            if found is not None:
                score_value, fold_se, fold_scores, at_mstop, at_boundary = found
                row.update(score=score_value, fold_se=fold_se or 0.0,
                           n_folds=_fold_count(folds), mstop=at_mstop, error=None,
                           fold_scores=fold_scores)
                if at_boundary:
                    warnings.warn(
                        f"select_capacity: the {label} rung's held-out score was "
                        f"still falling at {at_mstop} steps, the largest its budget "
                        f"search reached (max_mstop={max_mstop}), so it is scored "
                        f"and fitted there and its advantage may be understated. "
                        f"Raise max_mstop to search further.",
                        RuntimeWarning, stacklevel=2)
            else:
                result = _evaluate(rung_rhs, data, response, family=family,
                                   folds=folds, seed=seed, mstop=mstop, nu=nu,
                                   score=score, extrapolation=extrapolation,
                                   fit_kwargs=fit_kwargs, estimator=estimator)
                row.update(score=float(result.score),
                           fold_se=_fold_se(result.fold_scores),
                           n_folds=int(np.asarray(result.fold_scores).size),
                           mstop=int(mstop) if estimator == "boosting" else None,
                           fold_scores=tuple(float(v) for v in result.fold_scores),
                           error="; ".join(
                               str(fold["error"]) for fold in result.folds
                               if fold.get("status") == "failed") or None)
            rows.append(row)

    finite = [row for row in rows if np.isfinite(row["score"])]
    if not finite:
        raise ValueError(
            "the base rung's cross-validation failed on every fold; "
            "select_capacity cannot compare specifications")
    best = min(finite, key=lambda row: row["score"])

    def limit_of(row):
        """The bar `row` must clear: `tolerance` standard errors of its
        paired fold difference against the best rung, which cancels what
        the folds share. Without a pairable difference -- one fold, or a
        failed fold on either side -- the best rung's marginal fold spread
        decides."""
        paired = _paired_se(row, best)
        return best["score"] + tolerance * (
            paired if paired is not None else (best["fold_se"] or 0.0))

    form_order = {"intercept": -1, _BASE: 0, _TENSORS: 1, _TREE: 2}
    estimator_order = {name: i for i, name in enumerate(estimators)}
    estimator_order.setdefault("boosting", len(estimator_order))
    chosen = next(row for row in sorted(
        rows, key=lambda r: (form_order[r["label"]], estimator_order[r["estimator"]]))
        if np.isfinite(row["score"]) and row["score"] <= limit_of(row))
    chosen["selected"] = True

    chosen_rhs = chosen["rhs"]
    if chosen["estimator"] == "reml":
        model = mb.gam(f"{quote_response(response)} ~ {chosen_rhs}", data, family=family, **fit_kwargs)
        if not model.convergence.get("converged", True):
            raise ValueError(
                "the selected direct REML model did not converge on all rows: "
                f"{model.convergence.get('stop_reason', 'unknown')}")
    else:
        model = mb.gamboost(f"{quote_response(response)} ~ {chosen_rhs}", data, family=family,
                            control=mb.boost_control(chosen["mstop"], nu),
                            **fit_kwargs)
    return CapacitySelection(
        selected=chosen["label"], formula=chosen["formula"], rhs=chosen_rhs,
        selected_estimator=chosen["estimator"],
        tolerance=float(tolerance), rungs=tuple(rows),
        screened_pairs=tuple({"left": p.left, "right": p.right,
                              "removed": float(p.removed)} for p in survivors),
        model=model,
    )
