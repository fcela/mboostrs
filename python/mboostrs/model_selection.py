"""Model comparison and family scanning workflows (Milestone 9).

Provides high-level helpers for comparing fitted models and scanning
candidate LSS families, inspired by R's ``gamboostLSS`` model selection
ecosystem but without cloning every classical stepwise helper.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from mboostrs._comparison_identity import (
    check_comparison,
    check_cv_folds,
    common_sample_plan,
    family_scale,
    refit_on_common_sample,
)
from mboostrs._dispatch import is_lss
from mboostrs._validate import check_int_at_least, check_positive

__all__ = [
    "gaic_table",
    "compare_models",
    "scan_lss_families",
    "profile_mstop",
]


def _model_label(model, index: int) -> str:
    """Generate a human-readable label for a model."""
    formula = getattr(model, "_formula", None)
    if formula:
        if isinstance(formula, dict):
            formula = str(formula)
        return str(formula)[:60]
    family = getattr(model, "_family", None)
    fname = family.__class__.__name__ if family else "Unknown"
    return f"{fname}#{index}"


def _aic_at_optimum(result) -> float | None:
    """The AIC at the model's own optimal iteration.

    Reading `result.aic[-1]` gives the criterion at whatever `mstop` the model
    happened to be fitted with, so a table used to rank models ranked them by
    their AIC at their fitted length. That is only sound when every model in
    the table was fitted to the same `mstop`, which a model-comparison table
    invites you to violate. R's `AIC()` returns a scalar at the optimum for the
    same reason.
    """
    path = np.asarray(result.aic, dtype=float)
    if path.size == 0:
        return None
    best = getattr(result, "mstop", None)
    if best is not None and 1 <= int(best) <= path.size:
        return float(path[int(best) - 1])
    return float(np.nanmin(path))


def _criterion_row(model, k):
    """The criterion and all companion quantities at the same fitted step."""
    k = check_positive("k", k)
    if is_lss(model):
        count = sum(len(p["selected"]) for p in model._params) + 1
        return {
            "criterion": "lss_update_count_heuristic", "normalization": "total",
            "df_convention": "update_count_plus_one", "df": count,
            "penalty": k, "value": 2 * float(model._risk[-1]) + k * count,
            "mstop": model.mstop, "risk": float(model._risk[-1]),
            "logLik": float(model.logLik()),
        }
    method = "corrected" if type(model._family).__name__ == "Gaussian" else "classical"
    result = model.aic(method=method, k=k)
    step = int(result.mstop)
    fitted = model if step == model.mstop else model[:step]
    return {
        "criterion": result.criterion, "normalization": result.normalization,
        "df_convention": result.df_convention, "df": float(result.df[step - 1]),
        "penalty": result.penalty, "value": float(result.aic[step - 1]),
        "mstop": step, "risk": float(fitted._risk[-1]),
        "logLik": float(fitted.logLik()),
    }


def _require_same_response_and_weights(models):
    """Data-level sample checks for models without a comparison identity record.

    Compares the retained response, effective weights and shared columns of
    every model with the first, in row order. Used only when a model has no
    ``_comparison_identity``; a model that has one is compared through it.
    """
    from mboostrs._frame import read_column

    reference = None
    for model in models:
        frame = getattr(model, "_data", None)
        name = getattr(model, "_response_name", None)
        if frame is None or name is None:
            raise ValueError("comparison needs retained responses and observation weights")
        names = (name,) if isinstance(name, str) else tuple(name)
        response = np.column_stack([read_column(frame, part).values for part in names])
        weights = np.asarray(model.model_weights())
        if reference is not None:
            previous, previous_w = reference
            if (response.shape != previous.shape or not np.array_equal(response, previous)
                    or not np.array_equal(weights, previous_w)):
                raise ValueError("comparison requires the same observed responses, row order and effective weights")
        else:
            reference = response, weights
    mismatch = _shared_column_mismatch(models)
    if mismatch is not None:
        raise ValueError(mismatch)


def _shared_column_mismatch(models):
    """The first shared column whose values or levels differ between the models, or None.

    Compares every column the models' retained frames have in common, in row
    order, against the first model. Numeric encodings alone cannot identify
    factor labels, so the levels are compared too. A model without retained
    data is skipped.
    """
    from mboostrs._frame import column_names, read_column

    reference = None
    for model in models:
        frame = getattr(model, "_data", None)
        if frame is None:
            continue
        columns = set(column_names(frame))
        if reference is not None:
            previous_frame, previous_columns = reference
            for column in sorted(columns & previous_columns):
                left, right = read_column(frame, column), read_column(previous_frame, column)
                if (left.levels != right.levels
                        or not np.array_equal(left.values, right.values, equal_nan=True)):
                    return f"the models differ in shared column {column!r}; use the same rows in order"
        else:
            reference = frame, columns
    return None


def _get_metric(model, metric: str, k: float = 2.0) -> float | None:
    """Extract a single metric from a model.

    Returns None if the metric is unavailable for this model.
    """
    if metric == "risk":
        risk = getattr(model, "_risk", None)
        if risk is not None:
            return float(risk[-1])
        return None

    if metric == "logLik":
        try:
            return float(model.logLik())
        except NotImplementedError:
            return None

    if metric in ("GAIC", "AIC"):
        # A distributional fit has `gaic` and no `aic`; a standard fit the
        # other way round. Branching on the model kind rather than on the
        # presence of a name means adding `aic` to the LSS class one day
        # cannot silently reroute this.
        try:
            return _criterion_row(model, k)["value"]
        except NotImplementedError:
            pass
        return None

    if metric == "mstop":
        return float(model.mstop)

    if metric == "selected_learners":
        return float(_distinct_learners(model))

    if metric == "n":
        weights = getattr(model, "_weights", None)
        if weights is not None:
            return float(len(weights))
        fitted = getattr(model, "_fitted", None)
        if fitted is not None:
            return float(len(fitted))
        return None

    return None


def _resolve_sample(models, labels, *, requested, helper, cv=None):
    """The models a table scores, and the common-sample plan when one was asked for.

    With ``requested`` false the models come back unchanged and no plan is
    made. With it true, the plan names the rows every model can use. When some
    model loses rows to the others, every model is refitted on those rows,
    since a score on a different row set is not comparable; the excluded rows
    are reported in a warning and in each row by :func:`_sample_fields`.
    A cross-validation table cannot be refitted in place, so it is refused.
    """
    if not requested:
        return list(models), None
    plan = common_sample_plan(models, labels)
    if not plan.changed:
        return list(models), plan
    if cv is not None and any(c is not None for c in cv):
        raise ValueError(
            f"{helper}() with common_sample=True refits the models on the "
            f"{plan.n_rows} rows every candidate can use, and the cv= results were "
            "computed on the full sample. Compute each model's cvrisk() after "
            "refitting, or drop cv= from this call.")
    refitted = [refit_on_common_sample(model, plan) for model in models]
    detail = "; ".join(
        f"{label} loses {count} of its own rows" + (f", dropped by {by}" if by else "")
        for label, count, by in zip(labels, plan.excluded, plan.excluded_by, strict=True)
        if count)
    warnings.warn(
        f"{helper}(common_sample=True) scores every model on the {plan.n_rows} rows "
        f"all candidates share, out of {plan.n_input} input rows. Rows removed from "
        f"a candidate's own sample: {detail}. Each candidate was refitted on the "
        f"common rows; the excluded rows are reported in each row's "
        f"'common sample excluded rows' and 'common sample excluded by' fields.",
        UserWarning, stacklevel=3)
    return refitted, plan


def _sample_fields(plan, index) -> dict[str, Any]:
    """The common-sample columns of one row: the rows scored, and those this model lost."""
    return {
        "common sample rows": plan.n_rows,
        "common sample excluded rows": plan.excluded[index],
        "common sample excluded by": plan.excluded_by[index],
    }


def gaic_table(
    *models,
    k: float | tuple = 2.0,
    labels: list[str] | None = None,
    as_frame: bool | None = None,
    output: str | None = None,
    common_sample: bool = False,
) -> list[dict[str, Any]] | Any:
    """Compare same-sample fits using an explicitly named stopping criterion.

    Parameters
    ----------
    *models
        Fitted model objects (GLMBoostModel, GAMBoostModel, or LSSModel).
    k : float or tuple
        Penalty multiplier. If a float, use that value. If a tuple
        ``(k1, "logn")``, compute GAIC with penalty ``k1`` and also
        ``log(n)`` (BIC-like).
    labels : list of str, optional
        Custom model labels. Defaults to formula/family.
    as_frame : bool, optional
        Superseded by ``output=``. ``True`` returns a data frame, ``False``
        returns records. Leaving it unset — the default — returns records,
        which is what this function has always returned.
    output : {"auto", "pandas", "polars", "records"}, optional
        Output format. Overrides ``as_frame`` if both are given.
    common_sample : bool, default False
        Score every model on the rows all of them can use. A model drops the
        rows its ``na_action`` removes, so two candidates can keep different
        rows; with ``common_sample=True`` the intersection of the kept rows is
        taken, every model is refitted on it, and the criteria are computed
        there. Rows are never dropped silently: a warning names the rows each
        candidate loses, and each row reports ``common sample rows``,
        ``common sample excluded rows`` (how many of its own rows the common
        sample removed) and ``common sample excluded by`` (which other
        candidates dropped them). With the default ``False``, models on
        different rows raise ``ValueError``.

    Returns
    -------
    DataFrame or list of dict
        Gaussian rows use ``mboost_corrected``, a per-observation criterion
        with a fixed penalty; k must be 2. Likelihood rows use ``GAIC(k)``.
        LSS rows use ``heuristic_GAIC(k)``: their penalty counts updates plus
        one, not effective df. These three definitions cannot be ranked
        together. Rows report criterion, normalization, df convention and the
        evaluated mstop; risk and logLik refer to that same step.

        Models fitted to different rows, response values, row order or
        effective weights raise ``ValueError``. Models that share their rows
        but differ in response transform (``log(y)`` against ``y``) or weight
        policy are returned unsorted, each row with a ``comparison`` column
        giving the reason, because their likelihoods are over different scales.
    """
    if not models:
        raise ValueError("gaic_table() requires at least one model")
    if labels is not None and len(labels) != len(models):
        raise ValueError("labels must have one entry per model")
    labels = list(labels) if labels is not None else [
        _model_label(model, i) for i, model in enumerate(models)]
    models, sample_plan = _resolve_sample(models, labels, requested=common_sample,
                                          helper="gaic_table")

    # A GAIC table ranks models by `-2 * loglik + k * df`. That ranking means
    # something only when every row's first term is a log-likelihood and all of
    # them are taken with respect to the same quantity: otherwise the constants
    # do not cancel and a difference between two rows is not a likelihood
    # ratio. Checked once, for the whole table, so the refusal can name which
    # model broke the comparison.
    from mboostrs._likelihood import comparable_measures

    families = [getattr(model, "_family", None) for model in models]
    if all(family is not None for family in families):
        comparable, reason = comparable_measures(families)
        if not comparable:
            raise NotImplementedError(
                f"gaic_table() cannot rank these models: {reason}. Compare "
                f"them with cvrisk() on a shared fold matrix instead, which "
                f"needs no likelihood and no shared constant."
            )

    comparison = check_comparison(models, helper="gaic_table")
    if isinstance(k, tuple):
        if len(k) != 2 or k[1] != "logn":
            raise ValueError("k must be a positive number or (positive_number, 'logn')")
        penalties = [check_positive("k", k[0]), "logn"]
    else:
        penalties = [check_positive("k", k)]

    rows = []
    contracts = set()
    sort_key = None
    for i, model in enumerate(models):
        row: dict[str, Any] = {"label": labels[i]}

        for pen in penalties:
            if pen == "logn":
                n = _get_metric(model, "n")
                if n is not None and n > 0:
                    pen_val = math.log(n)
                    key = "GAIC(logn)"
                else:
                    continue
            else:
                pen_val = float(pen)
                key = f"GAIC({pen_val:g})"

            detail = _criterion_row(model, pen_val)
            contracts.add((detail["criterion"], detail["normalization"], detail["df_convention"]))
            if len(contracts) > 1:
                raise ValueError(
                    "these models use different criterion definitions or df conventions "
                    "and cannot be ranked together"
                )
            if detail["criterion"] == "mboost_corrected":
                key = "mboost_corrected"
            elif detail["criterion"] == "lss_update_count_heuristic":
                key = "heuristic_" + key
            row[key] = detail["value"]
            if sort_key is None:
                sort_key = key
            if "criterion" not in row:
                row.update({name: value for name, value in detail.items() if name != "value"})
            # A different penalty can select a different stopping point.
            row[f"{key}_mstop"] = detail["mstop"]
        row["n"] = _get_metric(model, "n")
        if sample_plan is not None:
            row.update(_sample_fields(sample_plan, i))
        rows.append(row)

    if comparison.verdict == "segregate":
        for row in rows:
            row["comparison"] = comparison.notice
    elif sort_key:
        rows.sort(key=lambda r: (np.isnan(r[sort_key]), r[sort_key] if not np.isnan(r[sort_key]) else float("inf")))

    from mboostrs._output import _normalize_output, output_frame
    fmt = _normalize_output(output, as_frame)
    if output is None and as_frame is None:
        # These two functions have always returned records, so an unset
        # `as_frame` keeps doing that and callers iterating `for row in result`
        # are unaffected. An explicit `as_frame=True` is honoured.
        fmt = "records"
    return output_frame(rows, fmt)


def _cv_optimum_row(cv_result) -> dict[str, Any]:
    """The cross-validated stopping point and risk for one model, or blanks."""
    from mboostrs.cvrisk import _optimum_index

    if cv_result is None:
        return {"cv_mstop": None, "cv_risk": None}
    aggregate = np.asarray(cv_result.aggregate, dtype=float)
    grid = np.asarray(cv_result.grid)
    best = _optimum_index(aggregate, grid)
    return {"cv_mstop": int(grid[best]), "cv_risk": float(aggregate[best])}


def _sort_comparison_rows(rows, sort, risk_refusal):
    """Order comparison rows by one column, refusing a sort that would rank incomparable values."""
    if not all(sort in row for row in rows):
        raise ValueError(f"sort={sort!r} is not a column of this table")
    if sort == "risk" and risk_refusal is not None:
        raise ValueError(f"cannot sort by risk: {risk_refusal}")
    if sort in ("GAIC", "AIC"):
        criteria = {row["criterion"] for row in rows if row.get("criterion") is not None}
        if len(criteria) > 1:
            raise ValueError(
                f"cannot sort by {sort}: the rows use different criteria "
                f"({', '.join(sorted(criteria))}) that are not on one scale"
            )
    return sorted(rows, key=lambda row: _criterion_sort_key(row, sort))


#: Metrics where a larger value is better. Every other metric is read as a
#: loss or a count, and ``sort=`` orders it smallest first.
_HIGHER_IS_BETTER = frozenset({"logLik"})


def _distinct_learners(model) -> int:
    """How many different base learners the path updated, up to the model's mstop.

    A distributional model counts the learners of each distribution parameter
    separately and adds them, because each parameter has its own path.
    """
    if is_lss(model):
        return sum(len({int(s) for s in p["selected"]}) for p in model._params)
    return len({int(s) for s in model._selected})


def _risk_scale(model) -> tuple:
    """The identity of the loss a model's ``risk`` is measured in.

    Two risks are on one scale when the family class and every constructor
    setting match: ``Quantile(tau=0.1)`` and ``Quantile(tau=0.5)`` are different
    losses, and so are ``Gaussian`` and ``Binomial`` even on the same response.
    Settings are read from the constructor signature, so nuisance values a fit
    estimates (a dispersion, a threshold) do not split one family into two.
    """
    family = getattr(model, "_family", None)
    if family is None:
        return (type(model).__name__,)
    return family_scale(family)


def _criterion_sort_key(row, sort):
    """Sort key for one row: best value first, missing values last."""
    value = row.get(sort)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (1, 0.0)
    direction = -1.0 if sort in _HIGHER_IS_BETTER else 1.0
    return (0, direction * float(value))


def compare_models(
    *models,
    metrics: tuple[str, ...] = ("risk", "logLik", "GAIC", "mstop", "selected_learners"),
    labels: list[str] | None = None,
    sort: str | None = None,
    cv: list | None = None,
    as_frame: bool | None = None,
    output: str | None = None,
    common_sample: bool = False,
) -> list[dict[str, Any]] | Any:
    """Compare models across multiple metrics.

    Parameters
    ----------
    *models
        Fitted model objects.
    metrics : tuple of str
        Metrics to evaluate: "risk", "logLik", "GAIC", "AIC", "mstop", "n",
        "selected_learners".

        ``logLik`` follows the family's likelihood contract and is unavailable
        for non-likelihood objectives. Legacy ``GAIC``/``AIC`` column names
        carry the selected criterion value; always read the accompanying
        ``criterion``, ``normalization`` and ``df_convention``. Gaussian
        corrected criteria and LSS update-count heuristics are not on a
        shared GAIC scale. When a criterion is requested, all metrics in a
        row refer to its selected stopping point.

        ``selected_learners`` is the number of distinct base learners the
        path updated, up to the model's own mstop; a distributional model
        adds the counts of its parameters.

        ``risk`` is withheld (``None``, with a ``risk_unavailable`` reason on
        every row) when the models do not share one loss: different families,
        or one family with different settings, such as two quantile levels.
        Risks in different units cannot be ranked.
    labels : list of str, optional
        Custom model labels.
    sort : str, optional
        A column to order the rows by, best first: smallest first for every
        metric except ``logLik``, largest first. Rows whose value is missing
        go last. Sorting by ``risk`` when it is withheld, or by ``GAIC`` or
        ``AIC`` when the rows use different criteria, raises ``ValueError``.
        Unset keeps the order the models were given in.
    cv : list of CvRiskResult or None, optional
        One cross-validation result per model, or ``None`` for a model without
        one. Adds ``cv_mstop`` and ``cv_risk``: the stopping point the
        cross-validated risk prefers and the risk there. All supplied results
        must carry the same non-empty ``fold_fingerprint``, so every model is
        scored on the same folds.
    as_frame : bool, optional
        Superseded by ``output=``. ``True`` returns a data frame, ``False``
        returns records. Leaving it unset — the default — returns records,
        which is what this function has always returned.
    output : {"auto", "pandas", "polars", "records"}, optional
        Output format. Overrides ``as_frame`` if both are given.
    common_sample : bool, default False
        Score every model on the rows all of them can use, refitting each on
        the intersection of their kept rows, as :func:`gaic_table` does. The
        excluded rows are reported per row. Cannot be combined with ``cv=``
        when rows are excluded, because the cross-validation results were
        computed on the full sample.

    Returns
    -------
    DataFrame or list of dict
        One row per model with requested metrics. Models on different rows,
        responses or effective weights raise ``ValueError``. Models that share
        their rows but differ in response transform or weight policy keep the
        given order, each row carries a ``comparison`` reason, and ``sort=``
        raises, since the rows are not on one scale.
    """
    if not models:
        raise ValueError("compare_models() requires at least one model")
    if labels is not None and len(labels) != len(models):
        raise ValueError("labels must have one entry per model")
    if cv is not None and len(cv) != len(models):
        raise ValueError("cv must have one entry per model (None for a model without one)")
    labels = list(labels) if labels is not None else [
        _model_label(model, i) for i, model in enumerate(models)]
    models, sample_plan = _resolve_sample(models, labels, requested=common_sample,
                                          helper="compare_models", cv=cv)
    comparison = check_comparison(models, helper="compare_models")
    if cv is not None:
        check_cv_folds(cv, helper="compare_models")

    scales = {_risk_scale(model) for model in models}
    risk_refusal = None
    if len(scales) > 1:
        names = sorted({scale[0] for scale in scales})
        risk_refusal = (
            f"risk is a loss in each family's own units, and these models use "
            f"{', '.join(names)}; compare them by cross-validated risk on shared folds"
        )

    rows = []
    for i, model in enumerate(models):
        row: dict[str, Any] = {"label": labels[i]}
        detail = None
        if any(metric in ("GAIC", "AIC") for metric in metrics):
            try:
                detail = _criterion_row(model, 2.0)
            except NotImplementedError as exc:
                row["criterion_unavailable"] = str(exc)
        for metric in metrics:
            if metric in ("GAIC", "AIC"):
                row[metric] = None if detail is None else detail["value"]
            elif detail is not None and metric in detail:
                row[metric] = detail[metric]
            else:
                row[metric] = _get_metric(model, metric)
        if detail is not None:
            row.update({name: detail[name] for name in
                        ("criterion", "normalization", "df_convention", "df", "penalty")})
        if risk_refusal is not None and "risk" in metrics:
            row["risk"] = None
            row["risk_unavailable"] = risk_refusal
        if cv is not None:
            row.update(_cv_optimum_row(cv[i]))
        if sample_plan is not None:
            row.update(_sample_fields(sample_plan, i))
        rows.append(row)

    if comparison.verdict == "segregate":
        # The rows are on different response scales: every column is shown,
        # none is ordered, and sorting would rank them anyway.
        if sort == "risk" and risk_refusal is not None:
            # The more specific reason: the risks are different losses.
            raise ValueError(f"cannot sort by risk: {risk_refusal}")
        if sort is not None:
            raise ValueError(
                f"cannot sort by {sort!r}: {comparison.notice}. Remove sort=, or "
                f"refit the models on one response scale to rank them")
        for row in rows:
            row["comparison"] = comparison.notice
    elif sort is not None:
        rows = _sort_comparison_rows(rows, sort, risk_refusal)

    from mboostrs._output import _normalize_output, output_frame
    fmt = _normalize_output(output, as_frame)
    if output is None and as_frame is None:
        # These two functions have always returned records, so an unset
        # `as_frame` keeps doing that and callers iterating `for row in result`
        # are unaffected. An explicit `as_frame=True` is honoured.
        fmt = "records"
    return output_frame(rows, fmt)


def _combined_formula(sides) -> str:
    """One formula reading every column any of ``sides`` reads."""
    from mboostrs.formula import partition_tilde

    sides = [sides] if isinstance(sides, str) else list(sides)
    lhs = partition_tilde(sides[0])[0].strip()
    rhs = " + ".join(partition_tilde(side)[2].strip() for side in sides
                     if partition_tilde(side)[2].strip())
    return f"{lhs} ~ {rhs}" if rhs else f"{lhs} ~ 1"


def _scan_common_sample(formula, data, families):
    """The data restricted to the rows every candidate family can be fitted to.

    Every candidate reads a subset of the entries of ``formula`` (a family with
    fewer parameters reads fewer of them), so the rows complete in every entry
    are the rows complete for every candidate. Returns the restricted data and
    a record of the input size, the kept row positions and the original data,
    from which each row's own exclusions are counted.
    """
    from mboostrs.formula import apply_na_action

    sides = list(formula.values()) if hasattr(formula, "keys") else [formula]
    reduced, dropped = apply_na_action(_combined_formula(sides), data, "omit")
    first = (next(iter(data.values())) if isinstance(data, dict)
             else data[list(data.columns)[0]])
    n_input = len(np.asarray(first).ravel())
    keep = np.setdiff1d(np.arange(n_input), np.asarray(dropped, dtype=np.intp))
    if dropped:
        warnings.warn(
            f"scan_lss_families(na_action='omit') fits and scores every candidate "
            f"family on the {keep.size} of {n_input} rows complete in every column "
            f"the formulas read; {len(dropped)} rows with a missing value were removed "
            f"before any fit. Each row reports 'common sample excluded rows': how many "
            f"rows its own formulas would have kept.", UserWarning, stacklevel=3)
    return reduced, {"n_input": n_input, "n_rows": int(keep.size), "keep": keep,
                     "dropped": frozenset(int(i) for i in dropped), "data": data}


def _scan_rows_excluded(sample, spec) -> int:
    """Rows the common sample removed that this candidate's own formulas keep."""
    from mboostrs.formula import apply_na_action

    sides = list(spec.values()) if hasattr(spec, "keys") else [spec]
    _, own = apply_na_action(_combined_formula(sides), sample["data"], "omit")
    return len(sample["dropped"] - {int(i) for i in own})


def scan_lss_families(
    formula,
    data,
    families: list | None = None,
    control=None,
    metric: str = "GAIC",
    fit_func="glmboostLSS",
    k: float = 2.0,
    support: str | None = None,
    folds=None,
    na_action: str = "error",
) -> list[dict[str, Any]]:
    """Scan candidate LSS families and rank by a metric.
    Usage::

        import mboostrs as mb
        rows = mb.scan_lss_families("y ~ bbs(x)", df, support="realplus")
        rows[0]  # best candidate first, or the first error if all failed

    The counterpart of ``gamlss::chooseDist``: choosing the response
    distribution is usually the least-informed decision in a distributional
    model, and fitting the alternatives is a more direct way to make it than
    reasoning about the response's shape.

    Parameters
    ----------
    formula : str or mapping
        The model specification, shared by every candidate: a whole formula
        (``"y ~ bbs(x)"``), or a mapping with one whole formula per
        distribution parameter (``{"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}``).
        Families differ in how many parameters they have, so a mapping is
        matched by position and any parameter beyond its length reuses the
        last entry. A bare right-hand side is refused before anything is fitted.
    data : dict or DataFrame
    families : list, optional
        LSS family instances or names to try (e.g. ``[GaussianLSS(), "GammaLSS"]``).
        Defaults to the candidate set for ``support``.
    support : str, optional
        The response support whose candidate set is used when ``families`` is
        not given: ``"realline"``, ``"realplus"``, ``"real0to1"``, ``"counts"``
        or ``"zeroplus"``. The sets are ``FAMILY_SETS`` in
        ``mboostrs.model_selection`` (also exported as ``mb.FAMILY_SETS``);
        ``mb.family_set(support)`` returns the family names for one support.
    folds : int or array, optional
        Fold weights for ``metric="cvrisk"``: an ``(n, B)`` weight matrix, or a
        number of folds for a seeded k-fold split (`cv_weights`). Defaults to
        5-fold.
    control : boost_control tuple, optional
        Defaults to ``boost_control(100, 0.1)``.
    metric : {"cvrisk", "GAIC", "risk", "logLik"}
        What to rank by. Lower is better for every one but ``logLik``.

        ``"cvrisk"`` is out-of-bag risk from cross-validation, and is the
        metric to prefer. ``"GAIC"`` matches ``chooseDist`` but see the warning
        below.
    fit_func : {"glmboostLSS", "gamboostLSS"}
        Which LSS fit function to use.
    k : float
        Penalty for GAIC.
    na_action : {"error", "omit"}
        ``"error"`` (the default) refuses a missing value in any column a
        formula reads. ``"omit"`` fixes one sample before any candidate is
        fitted: the rows complete in every column that any entry of
        ``formula`` reads. A one-parameter family reads only the first entry
        of a mapping and would keep rows a three-parameter family drops; every
        family is fitted and scored on the common rows instead, so no
        candidate scores better by having dropped rows. ``folds`` given as a
        matrix has one row per input row and is restricted to the kept rows.
        A ``UserWarning`` names the number of rows removed, and each row
        reports ``common sample rows`` and ``common sample excluded rows``
        (how many rows its own formulas would have kept that the common
        sample removed).

    Returns
    -------
    list of dict
        Sorted by metric (best first). Failed fits have ``error`` set and sort
        last, so a candidate that could not fit is recorded rather than
        silently missing. When every candidate fails, a ``RuntimeWarning``
        names the first error: the first row is then not a winner. With ``metric="cvrisk"`` each row also carries
        ``cvrisk_mstop``, the grid step where the out-of-bag minimum landed;
        it equals the budget when the risk curve was still falling at the cap,
        which makes the ranking provisional. When the fitted candidates differ in
        response transform, weight policy or reference measure (a count family's
        probability mass next to a continuous family's density), the rows keep
        candidate order, each fitted row carries a ``comparison`` reason, and
        ``metric`` does not order them. Candidates fitted to different rows or
        weights raise ``ValueError``.

    Notes
    -----
    Any comparison across families rests on every family's risk being the exact
    negative log-likelihood of the same data: one that dropped a constant term
    would appear better or worse by that constant.
    **Legacy GAIC score.** Its penalty counts boosting
    updates plus one; it is not effective df and is no longer reported as
    likelihood df. Under cyclic fitting every parameter is
    updated once per iteration, so that count is `mstop * n_params + 1`: it
    measures the *boosting budget*, not how much structure the model absorbed.
    A two-parameter family at `mstop=60` scores df 121 against a one-parameter
    family's 61 whatever either of them fitted, so GAIC ranks largely by
    parameter count.

    """
    import mboostrs as mb

    if control is None:
        control = mb.boost_control(100, 0.1)

    if families is None:
        if support is None:
            raise ValueError(
                f"give either families= or support=; support is one of "
                f"{sorted(FAMILY_SETS)}"
            )
        families = family_set(support)

    # Every family is fitted from these formulas, so a bare right-hand side
    # would fail identically in each candidate; refuse it once, here.
    sides = list(formula.values()) if hasattr(formula, "keys") else [formula]
    for side in sides:
        if isinstance(side, str) and "~" not in side:
            raise ValueError(
                f"scan_lss_families takes whole formulas, got the right-hand "
                f"side {side!r}; write it as 'response ~ {side}'")

    fit_fn = getattr(mb, fit_func, None)
    if fit_fn is None:
        raise ValueError(f"Unknown fit_func: {fit_func!r}")

    valid_metrics = ("cvrisk", "GAIC", "risk", "logLik")
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {metric!r}")
    if na_action not in ("error", "omit"):
        raise ValueError(f"na_action must be 'error' or 'omit', got {na_action!r}")

    sample = None
    if na_action == "omit":
        data, sample = _scan_common_sample(formula, data, families)

    fold_weights = None
    if metric == "cvrisk":
        first = (next(iter(data.values())) if isinstance(data, dict)
                 else data[list(data.columns)[0]])
        n_obs = len(np.asarray(first).ravel())
        if folds is None:
            fold_weights = mb.cv_weights(n_obs, type="kfold", B=5)
        elif isinstance(folds, (int, np.integer)) and not isinstance(folds, bool):
            fold_weights = mb.cv_weights(n_obs, type="kfold", B=int(folds))
        else:
            fold_weights = np.asarray(folds, dtype=np.float64)
            if sample is not None and fold_weights.shape[0] == sample["n_input"]:
                fold_weights = fold_weights[sample["keep"]]

    rows = []
    fitted = []
    for candidate in families:
        family = getattr(mb, candidate)() if isinstance(candidate, str) else candidate
        fname = family.__class__.__name__
        row: dict[str, Any] = {"family": fname}
        # A mapping specification is matched to this family's parameter count.
        if hasattr(formula, "keys"):
            sides = list(formula.values())
            spec = {
                name: sides[min(index, len(sides) - 1)]
                for index, name in enumerate(family.param_names)
            }
        else:
            spec = formula
        if sample is not None:
            row["common sample rows"] = sample["n_rows"]
            row["common sample excluded rows"] = _scan_rows_excluded(sample, spec)
        try:
            model = fit_fn(spec, data=data, families=family, control=control)
            row["status"] = "ok"
            detail = _criterion_row(model, k)
            row["GAIC"] = detail["value"]
            row.update({name: detail[name] for name in
                        ("criterion", "normalization", "df_convention", "df", "penalty")})
            row["risk"] = _get_metric(model, "risk")
            row["logLik"] = _get_metric(model, "logLik")
            row["mstop"] = model.mstop
            row["n"] = _get_metric(model, "n")
            if metric == "cvrisk":
                # The same budget for every candidate, along the diagonal --
                # the direction cross-validation can reuse one fitted path for.
                budget = len(np.asarray(model.risk_)) - 1
                steps = sorted({int(round(v)) for v in
                                np.linspace(1, budget, min(budget, 8))})
                grid = [[step] * len(family.param_names) for step in steps]
                result = model.cvrisk(folds=fold_weights, grid=grid)
                aggregate = np.asarray(result["aggregate"])
                row["cvrisk"] = float(np.min(aggregate))
                # The grid step where the out-of-bag minimum landed. Equal to
                # the budget means the curve was still falling at the cap, so
                # the ranking is provisional: a larger budget can reorder it.
                row["cvrisk_mstop"] = int(steps[int(np.argmin(aggregate))])
        # Broad on purpose: the row *is* the typed missing value. It records
        # `status="error"` and the message, so a candidate family that cannot
        # be fitted on this data is reported as such next to the ones that
        # were, rather than ending the comparison.
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)[:200]
            row["GAIC"] = None
            row["risk"] = None
            row["logLik"] = None
            row["mstop"] = None
            row["n"] = None
            row["cvrisk"] = None
            row["cvrisk_mstop"] = None
        if row["status"] == "ok":
            fitted.append((row, model))
        rows.append(row)

    comparison = check_comparison([model for _row, model in fitted],
                                  helper="scan_lss_families")

    def _sort_key(r):
        status_ok = r["status"] == "ok"
        val = r.get(metric)
        if val is None:
            val = float("inf")
        if metric == "logLik":
            val = -val
        return (not status_ok, val)

    if comparison.verdict == "segregate":
        # Every fitted candidate is shown in candidate order with the reason;
        # failed fits still go last.
        for row, _model in fitted:
            row["comparison"] = comparison.notice
        rows.sort(key=lambda r: r["status"] != "ok")
    else:
        rows.sort(key=_sort_key)
    if rows and all(r["status"] == "error" for r in rows):
        warnings.warn(
            f"every candidate family failed to fit; the first error was: "
            f"{rows[0]['error']}", RuntimeWarning, stacklevel=2)
    return rows


def profile_mstop(
    model,
    grid: list[int] | None = None,
    metric: str = "risk",
    k: float = 2.0,
) -> list[dict[str, Any]]:
    """Evaluate a model at different mstop values.

    Parameters
    ----------
    model
        A fitted GLMBoostModel, GAMBoostModel, or LSSModel.
    grid : list of int, optional
        mstop values to evaluate. If None, uses 10 evenly-spaced points
        from 1 to model.mstop.
    metric : {"risk", "GAIC", "logLik"}
        Metric to evaluate at each mstop.
    k : float
        Penalty for GAIC.

    Returns
    -------
    list of dict
        One row per grid point with mstop and metric value.
    """
    if grid is None:
        ms = model.mstop
        grid = list(range(1, ms + 1, max(1, ms // 10)))
        if ms not in grid:
            grid.append(ms)

    if metric not in ("risk", "GAIC", "logLik"):
        raise ValueError(f"metric must be 'risk', 'GAIC', or 'logLik', got {metric!r}")
    grid = [check_int_at_least("grid mstop", step, 1, allow_none=False) for step in grid]
    if not grid or any(step > model.mstop for step in grid):
        raise ValueError("grid must contain fitted iterations between 1 and model.mstop")
    k = check_positive("k", k)

    rows = []
    for mstop in sorted(set(grid)):
        row: dict[str, Any] = {"mstop": mstop}

        if metric == "risk":
            risk = getattr(model, "_risk", None)
            if risk is not None and mstop < len(risk):
                row["risk"] = float(risk[mstop])
            else:
                row["risk"] = None

        elif metric == "GAIC":
            if is_lss(model):
                # An LSS path has to be refitted to reach a different mstop.
                try:
                    # `LSSModel` has no `__getitem__`: a distributional path
                    # is one mstop per parameter, so it is refitted.
                    sub = model.update(mstop=mstop)
                    detail = _criterion_row(sub, k)
                    row["GAIC"] = detail["value"]
                    row.update({name: detail[name] for name in
                                ("criterion", "normalization", "df_convention", "df", "penalty")})
                except NotImplementedError as exc:
                    row["GAIC"] = None
                    row["unavailable_reason"] = str(exc)
            else:
                try:
                    method = "corrected" if type(model._family).__name__ == "Gaussian" else "classical"
                    aic_result = model.aic(method=method, k=k)
                    row["GAIC"] = float(aic_result.aic[mstop - 1])
                    row.update({name: getattr(aic_result, name) for name in
                                ("criterion", "normalization", "df_convention", "penalty")})
                    row["df"] = float(aic_result.df[mstop - 1])
                except NotImplementedError as exc:
                    row["GAIC"] = None
                    row["unavailable_reason"] = str(exc)

        elif metric == "logLik":
            try:
                sub = model.update(mstop=mstop) if is_lss(model) else model[:mstop]
                row["logLik"] = float(sub.logLik())
            except NotImplementedError as exc:
                row["logLik"] = None
                row["unavailable_reason"] = str(exc)

        rows.append(row)

    return rows


def r_squared(model, kind="both"):
    """Generalized R-squared for a distributional model, as ``gamlss::Rsq``.

    Both forms are likelihood ratios against the model's own null — for a
    boosted fit, iteration zero, where the predictor is the offset alone and
    nothing has been selected:

    * Cox-Snell: ``1 - exp((2/n) * (loglik_null - loglik_fit))``. Bounded above
      by ``1 - exp((2/n) * loglik_null)``, which is below 1 for a discrete
      response, so it cannot reach 1 however good the fit.
    * Cragg-Uhler (also called Nagelkerke): the same quantity divided by that
      bound, so 1 is attainable.

    Parameters
    ----------
    model : LSSModel
        A distributional fit. The measure needs a risk path that is a negative
        log-likelihood, which is what an LSS risk is.
    kind : {"both", "cox-snell", "cragg-uhler"}

    Returns
    -------
    float or dict

    Raises
    ------
    NotImplementedError
        For a standard `glmboost` or `gamboost` fit. Their `risk_` is the
        family's loss, and for most families that is not a log-likelihood:
        Gaussian's is the residual sum of squares, so `-risk_` is short of the
        log-likelihood by the normalizing terms, and `Quantile`, `Expectile`
        and `Huber` are not likelihoods at all. `logLik()` provides the
        value at the fitted `mstop`, but no matching likelihood is available
        at iteration zero for the ratio.

    Notes
    -----
    Descriptive, not adjusted for the number of selected terms. Boosting
    selects terms from the data, so this reads higher than a cross-validated
    estimate would; use :func:`cvrisk` to compare models.
    """
    valid = {"both", "cox-snell", "cragg-uhler"}
    if kind not in valid:
        raise ValueError(f"kind must be one of {sorted(valid)}, got {kind!r}")

    from mboostrs._dispatch import is_lss

    if not is_lss(model):
        raise NotImplementedError(
            "r_squared is defined for distributional (LSS) models, whose risk "
            "is a negative log-likelihood. A standard model's risk_ is the "
            "family's loss: for Gaussian that is the residual sum of squares, "
            "not a log-likelihood, and for Quantile, Expectile and Huber there "
            "is no likelihood at all."
        )

    risk = np.asarray(model.risk_, dtype=np.float64)
    if risk.size < 2:
        raise ValueError("r_squared needs a risk path with at least one iteration")
    # `fitted()` on an LSS model is a mapping of parameter name to vector, so
    # take the observation count from one parameter, not from the mapping.
    fitted = model.fitted()
    first = next(iter(fitted.values())) if isinstance(fitted, dict) else fitted
    n = len(np.asarray(first, dtype=np.float64).ravel())
    if n == 0:
        raise ValueError("r_squared needs a fitted model with observations")

    loglik_null = -float(risk[0])
    loglik_fit = -float(risk[-1])
    cox_snell = 1.0 - np.exp((2.0 / n) * (loglik_null - loglik_fit))
    ceiling = 1.0 - np.exp((2.0 / n) * loglik_null)
    cragg_uhler = cox_snell / ceiling if ceiling != 0 else float("nan")

    if kind == "cox-snell":
        return float(cox_snell)
    if kind == "cragg-uhler":
        return float(cragg_uhler)
    return {"cox_snell": float(cox_snell), "cragg_uhler": float(cragg_uhler)}


#: Candidate sets by response support, following gamlss' `chooseDist(type=)`.
#: A family is listed under a support only if it can actually fit that data;
#: `choose_family` still verifies by fitting, and records why any candidate was
#: skipped rather than dropping it silently.
FAMILY_SETS = {
    "realline": ("GaussianLSS", "LogisticLSS", "StudentTLSS", "GumbelLSS", "JSULSS"),
    "realplus": ("GammaLSS", "InverseGaussianLSS", "Pareto2LSS", "BCCGLSS",
                 "LogNormalLSS", "WeibullLSS", "LogLogLSS"),
    "real0to1": ("BetaLSS",),
    "counts": ("PoissonLSS", "GeometricLSS", "NBinomialLSS", "NegativeBinomialLSS",
               "ZIPoLSS", "ZINBLSS", "ZeroAdjustedPoissonLSS"),
    "zeroplus": ("ZeroAdjustedGammaLSS",),
}


def family_set(support: str) -> list[str]:
    """Return the candidate LSS family names for one response support.

    ``support`` is a key of ``FAMILY_SETS``: ``"realline"`` (the whole real
    line), ``"realplus"`` (positive continuous), ``"real0to1"`` (proportions in
    the open unit interval), ``"counts"`` (non-negative integers) or
    ``"zeroplus"`` (non-negative continuous with a point mass at zero).

    The result is a new list, so changing it does not change ``FAMILY_SETS``.
    Pass it as ``families=`` to ``scan_lss_families``, or read it before
    scanning to see which families a support covers.

    Raises
    ------
    ValueError
        If ``support`` is not one of the keys of ``FAMILY_SETS``. The message
        lists the valid supports.
    """
    if not isinstance(support, str) or support not in FAMILY_SETS:
        raise ValueError(
            f"support must be one of {sorted(FAMILY_SETS)}, got {support!r}; "
            f"pass one of those names, for example family_set('realplus')"
        )
    return list(FAMILY_SETS[support])


def _gaic_column(k: float) -> str:
    return f"gaic_k{k:g}".replace(".", "_")
