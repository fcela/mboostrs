"""Residual checks that name the revision they point to.

Three checks run on the residuals of a fitted mean model (`glmboost` or
`gamboost`):

* :func:`cluster_residual_check` asks whether residuals agree within clusters.
* :func:`time_residual_check` asks whether residuals, ordered by time, agree
  with their neighbours.
* :func:`influence_check` asks whether a few rows move the fitted values.

Each returns a :class:`RevisionFinding` (or :class:`InfluenceFinding`) that
states what was found, the number behind it and the reference it is judged
against, the assumption the pattern challenges, the alternative explanations,
the revision it points to, and the held-out comparison that would support the
revision. :func:`revision_checks` runs all three and returns a
:class:`RevisionReport`.

The numbers come from the Rust core (`mboostrs_core::residual_structure`).
Each check also reports its false-alarm rate: the share of simulated datasets
with no structure in which the same rule fires, for the same design. A
revision that a check suggests is therefore never reported without the rate
at which that suggestion would appear on data that needs no revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mboostrs._dispatch import is_lss as _is_lss
from mboostrs._names import quote_name

#: Default significance level of every check.
ALPHA = 0.05


@dataclass(frozen=True, repr=False)
class RevisionFinding:
    """One residual check: what it found, the evidence, and the revision.

    ``flagged`` is the decision at ``alpha``. ``statistic``, ``value`` and
    ``units`` name the number the decision rests on; ``p_value`` and
    ``reference`` say how it is judged. ``false_alarm_rate`` is the share of
    ``false_alarm_draws`` simulated null datasets, with the same design, on
    which the same rule fires. ``revision`` is the change to try when
    ``flagged`` is true, and a statement that no change is indicated when it
    is not.
    """

    #: Which check produced the finding: ``"cluster"``, ``"time"`` or ``"influence"``.
    check: str
    #: Whether the pattern is present at ``alpha``.
    flagged: bool
    #: What was found, in plain words, with the numbers.
    pattern: str
    #: Name of the statistic the decision rests on.
    statistic: str
    #: Value of that statistic.
    value: float
    #: Units of ``value``.
    units: str
    #: The p-value of the decision, or None where the check has none.
    p_value: float | None
    #: How ``p_value`` is computed and from how many draws.
    reference: str
    #: Share of simulated null datasets in which the rule fires.
    false_alarm_rate: float
    #: Number of simulated null datasets behind ``false_alarm_rate``.
    false_alarm_draws: int
    #: The modelling assumption the pattern challenges.
    assumption: str
    #: Explanations other than the one the revision assumes.
    alternatives: tuple[str, ...]
    #: The change to make to the fit, or the statement that none is indicated.
    revision: str
    #: The comparison on held-out data that would support the revision.
    held_out_check: str
    #: Further numbers behind the finding, keyed by name.
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """The finding as a readable block: pattern, evidence, guide."""
        p = "" if self.p_value is None else f", p = {self.p_value:.3g}"
        lines = [
            f"[{self.check}] {self.pattern}",
            f"  evidence: {self.statistic} = {self.value:.4g} ({self.units}){p}; "
            f"reference: {self.reference}",
            f"  false-alarm rate on {self.false_alarm_draws} null datasets with "
            f"this design: {self.false_alarm_rate:.3f}",
            f"  assumption challenged: {self.assumption}",
            "  alternatives: " + "; ".join(self.alternatives),
            f"  revision: {self.revision}",
            f"  held-out check: {self.held_out_check}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary

    def __repr__(self) -> str:
        return self.summary


@dataclass(frozen=True, repr=False, eq=False)
class InfluenceFinding(RevisionFinding):
    """The influence check, with the per-row measures and exact deletion refits.

    ``rows`` are the flagged row positions (zero-based, in data order).
    ``cooks``, ``loo_residual``, ``studentized`` and ``leverage`` hold one value
    per row. ``refits`` holds one record per refitted row: the row, and the
    largest and mean change in fitted values on the rows that were kept.
    """

    rows: tuple[int, ...] = ()
    cooks: np.ndarray | None = None
    loo_residual: np.ndarray | None = None
    studentized: np.ndarray | None = None
    leverage: np.ndarray | None = None
    refits: tuple[dict, ...] = ()


@dataclass(frozen=True, repr=False)
class RevisionReport:
    """The three checks run together, in the order cluster, time, influence.

    A check that was not asked for (no ``cluster=`` or no ``time=``) is absent
    from ``findings``. ``revisions`` lists the revision of every flagged check.
    """

    findings: tuple[RevisionFinding, ...]

    @property
    def flagged(self) -> tuple[RevisionFinding, ...]:
        """The findings whose pattern is present at ``alpha``."""
        return tuple(f for f in self.findings if f.flagged)

    @property
    def revisions(self) -> tuple[str, ...]:
        """The revision of each flagged finding, in finding order."""
        return tuple(f.revision for f in self.flagged)

    @property
    def summary(self) -> str:
        """Every finding, one block each."""
        if not self.findings:
            return "No residual check was requested; pass cluster= or time=."
        return "\n\n".join(f.summary for f in self.findings)

    def __str__(self) -> str:
        return self.summary

    def __repr__(self) -> str:
        return self.summary


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _require_mean_model(model) -> None:
    if _is_lss(model):
        raise ValueError(
            "the residual checks read the residuals of one mean model; this is a "
            "distributional (LSS) model with one residual set per parameter. Run "
            "the check on a glmboost or gamboost fit of the mean"
        )


def _residuals(model) -> np.ndarray:
    values = np.asarray(model.residuals(type="response"), dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "the model has non-finite residuals; refit with na_action='omit' or "
            "'impute' before running the residual checks"
        )
    return values


def _column(model, values, what: str, n: int):
    """A column named by string (read from the model's data) or one value per row.

    Returns the values and the column kind: ``"categorical"`` for a text or
    categorical column, ``"numeric"`` otherwise.
    """
    kind = "numeric"
    if isinstance(values, str):
        data = getattr(model, "_data", None)
        if data is None:
            raise ValueError(
                f"{what}={values!r} names a column, but the model kept no data; pass "
                "the values themselves, one per row"
            )
        from mboostrs._frame import column_names, read_column

        if values not in column_names(data):
            raise ValueError(
                f"{what}={values!r} names no column of the data; pass a column name "
                "or one value per row"
            )
        column = read_column(data, values)
        array = np.asarray(column.values)
        if column.kind == "categorical":
            kind = "categorical"
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "biufM":
            kind = "categorical"
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(
            f"{what} has shape {array.shape}; it needs one value per row ({n} rows)"
        )
    return array, kind


def _name(values, default: str) -> str:
    return values if isinstance(values, str) else default


def _cluster_codes(model, cluster, n: int) -> tuple[np.ndarray, int]:
    labels, _ = _column(model, cluster, "cluster", n)
    if labels.dtype.kind == "f" and np.any(~np.isfinite(labels)):
        raise ValueError("cluster has missing values; drop or impute those rows first")
    _, codes = np.unique(labels, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64).ravel()
    return codes, int(codes.max()) + 1


def _numeric_time(model, time, n: int) -> np.ndarray:
    values, kind = _column(model, time, "time", n)
    if kind == "categorical":
        raise ValueError(
            "time must be numeric (an index, a year, a day count); a text or categorical "
            "time must be converted first, for example to a day count or an integer index"
        )
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("time has missing values; drop or impute those rows first")
    return values


def _is_in_formula(model, name: str | None) -> bool:
    if name is None:
        return False
    formula = getattr(model, "formula", None)
    if formula is None:
        return False
    text = formula if isinstance(formula, str) else str(formula)
    return name in text


# ---------------------------------------------------------------------------
# Cluster check
# ---------------------------------------------------------------------------


def cluster_residual_check(model, cluster, *, alpha: float = ALPHA,
                           permutations: int = 999, draws: int = 200,
                           seed: int = 0) -> RevisionFinding:
    """Measure the share of response-residual variance between clusters.

    The statistic is the one-way ANOVA intra-cluster correlation (ICC) of the
    response residuals: the share of residual variance that lies between
    clusters, after allowing for unequal cluster sizes. An ICC near zero means
    the residuals of one cluster are no more alike than those of different
    clusters. A positive ICC means rows in the same cluster share a common
    departure from the fitted mean, which a random intercept per cluster
    (``brandom(cluster)``) would absorb.

    Parameters
    ----------
    model : glmboost or gamboost fit
    cluster : str or array
        Column name of the model's data, or one cluster label per row.
    alpha : float
        Significance level of the decision.
    permutations : int
        Shuffles of the residuals across rows, for the permutation p-value.
        Cluster sizes are kept, so the reference is exact under exchangeable
        residuals.
    draws, seed
        Null datasets for the false-alarm rate, and the random seed.

    Returns
    -------
    RevisionFinding
        ``flagged`` is true when the ICC is positive and the permutation
        p-value is below ``alpha``. ``evidence`` holds the cluster count, the
        cluster sizes and the cluster mean residuals.

    Notes
    -----
    A cluster check needs at least two clusters and some cluster with two or
    more rows. With few clusters (under five) a random intercept is weakly
    identified, so a flag then argues for grouped evaluation more than for a
    particular intercept.
    """
    _require_mean_model(model)
    residuals = _residuals(model)
    n = residuals.shape[0]
    codes, groups = _cluster_codes(model, cluster, n)
    name = _name(cluster, "cluster")

    out = _rust_call(
        "residual_cluster_icc", residuals, codes, permutations, seed)
    icc, f_stat, df_between, df_within, f_p, perm_p, perms, _, sizes, means = out
    flagged = bool(icc > 0.0 and perm_p < alpha)
    rate = _rust_call(
        "residual_cluster_false_alarm_rate", codes, draws, permutations, seed + 1, alpha)

    share = f"{icc:.3f}"
    pattern = (
        f"Residuals agree within {name} clusters: intra-cluster correlation {share} "
        f"over {groups} clusters (between-cluster F = {f_stat:.3g} on "
        f"({df_between:.0f}, {df_within:.0f}) df)."
        if flagged else
        f"Residuals do not agree within {name} clusters: intra-cluster correlation "
        f"{share} over {groups} clusters, not distinguishable from zero at alpha = {alpha}."
    )
    if flagged:
        revision = f"add brandom({quote_name(name)}) to the formula, so each cluster has its own intercept"
        if groups < 5:
            revision += (
                f"; with {groups} clusters the intercept is weakly identified, so "
                "evaluate with grouped_folds on the cluster column as well"
            )
        revision += (
            ", then rerun this check on the new fit. Boosting moves a random intercept "
            "by nu per step, so a short mstop can leave a positive ICC after the term is "
            "added; check the ICC again at a larger mstop"
        )
    else:
        revision = (
            "no cluster term is indicated by this check; keep the formula, and "
            "evaluate with grouped_folds if the rows are still clustered by design"
        )
    return RevisionFinding(
        check="cluster",
        flagged=flagged,
        pattern=pattern,
        statistic="intra-cluster correlation",
        value=float(icc),
        units="share of residual variance between clusters; dimensionless",
        p_value=float(perm_p),
        reference=(
            f"permutation p-value from {perms} shuffles of the residuals across rows, "
            "cluster sizes kept; the rule is ICC > 0 and p < alpha"
        ),
        false_alarm_rate=float(rate),
        false_alarm_draws=int(draws),
        assumption="rows are independent given the predictors",
        alternatives=(
            f"a cluster effect the model lacks (add brandom({quote_name(name)}))",
            "a predictor that varies between clusters and is missing from the formula",
            "batch or measurement effects that happen to line up with the cluster labels",
        ),
        revision=revision,
        held_out_check=(
            f"score the formula with and without brandom({quote_name(name)}) by cvrisk on "
            f"grouped_folds({name!r}); the term is supported when the grouped risk falls"
        ),
        evidence={
            "clusters": int(groups),
            "cluster_sizes": [int(s) for s in sizes],
            "cluster_means": [float(m) for m in means],
            "f_statistic": float(f_stat),
            "f_p_value": float(f_p),
            "df_between": float(df_between),
            "df_within": float(df_within),
            "permutations": int(perms),
            "residuals": "response",
        },
    )


# ---------------------------------------------------------------------------
# Time check
# ---------------------------------------------------------------------------


def time_residual_check(model, time, *, max_lag: int | None = None,
                        alpha: float = ALPHA, permutations: int = 999,
                        draws: int = 200, seed: int = 0) -> RevisionFinding:
    """Test whether residuals are correlated across ordered time points.

    The residuals are ordered by ``time`` and averaged within each distinct
    time value, so the series has one point per time value and a lag is a step
    between consecutive time values. The statistic is the Ljung-Box sum of the
    squared sample autocorrelations over lags ``1..max_lag``. Its reference is a
    permutation distribution: the series is shuffled and the statistic
    recomputed, which is exact when the residuals are exchangeable under no
    serial dependence. The chi-squared p-value is reported as well.

    Parameters
    ----------
    model : glmboost or gamboost fit
    time : str or array
        Column name of the model's data, or one numeric time per row.
    max_lag : int, optional
        Highest lag in the statistic. Defaults to ``10 * log10(m)``, rounded
        down, where ``m`` is the number of distinct time values, as R's
        ``acf`` does.
    alpha, permutations, draws, seed
        As in :func:`cluster_residual_check`.

    Returns
    -------
    RevisionFinding
        ``evidence`` holds the autocorrelations by lag, the white-noise band
        ``1.96 / sqrt(m)`` and the lags whose autocorrelation lies outside it.

    Notes
    -----
    A time term already in the formula is noted in the revision text, because
    a remaining dependence then points at the error process rather than at
    a missing smooth.
    """
    _require_mean_model(model)
    residuals = _residuals(model)
    n = residuals.shape[0]
    times = _numeric_time(model, time, n)
    distinct = int(np.unique(times).shape[0])
    if distinct < 4:
        raise ValueError(
            f"the time check needs at least four distinct time values; this column has "
            f"{distinct}. Pass a time column with more distinct values"
        )
    if max_lag is None:
        max_lag = min(int(np.floor(10.0 * np.log10(distinct))), distinct - 2)
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    name = _name(time, "time")

    out = _rust_call(
        "residual_serial_dependence", residuals, times, int(max_lag), permutations, seed)
    acf, q, lags, chisq_p, perm_p, perms, length, bound, unique_times = out
    acf = np.asarray(acf, dtype=np.float64)
    flagged = bool(perm_p < alpha)
    rate = _rust_call(
        "residual_serial_false_alarm_rate", times, int(max_lag), draws, permutations,
        seed + 1, alpha)
    beyond = [int(k + 1) for k, a in enumerate(acf) if abs(a) > bound]
    lag_one = float(acf[0])
    in_formula = _is_in_formula(model, _name(time, None))

    if flagged:
        pattern = (
            f"Residuals are serially correlated in {name}: Ljung-Box Q = {q:.4g} over "
            f"lags 1 to {lags}, lag-1 autocorrelation {lag_one:.3f}"
            + (f", lags outside the white-noise band: {beyond}." if beyond else ".")
        )
        if in_formula:
            revision = (
                f"{name} already enters the formula, so the remaining dependence is in "
                "the error process; a time smooth in the mean does not remove it. "
                "Evaluate on contiguous time blocks (blocked_folds) and treat interval "
                "widths from this fit as too narrow"
            )
        else:
            revision = (
                f"add a time smooth: bbs({quote_name(name)}) for a non-linear trend, or "
                f"bols({quote_name(name)}) for a linear one, then rerun this check"
            )
    else:
        pattern = (
            f"Residuals show no serial correlation in {name}: Ljung-Box Q = {q:.4g} over "
            f"lags 1 to {lags}, permutation p = {perm_p:.3g}."
        )
        revision = (
            "no time term is indicated by this check; keep the formula, and evaluate "
            "on time-ordered folds if the prediction target is later periods"
        )
    return RevisionFinding(
        check="time",
        flagged=flagged,
        pattern=pattern,
        statistic="Ljung-Box Q",
        value=float(q),
        units=f"chi-squared-type statistic over {lags} lags; dimensionless",
        p_value=float(perm_p),
        reference=(
            f"permutation p-value from {perms} shuffles of the time-ordered series "
            f"(chi-squared p-value with {lags} df: {chisq_p:.3g})"
        ),
        false_alarm_rate=float(rate),
        false_alarm_draws=int(draws),
        assumption="rows are independent given the predictors, so residuals are uncorrelated across time",
        alternatives=(
            f"a smooth trend in {name} the model lacks (add bbs({quote_name(name)}) or bols({quote_name(name)}))",
            "an autoregressive error: the noise itself carries the dependence, and a "
            "time smooth reduces but does not remove it",
            "a level shift at a known date (add a step term at that date, or split the sample)",
        ),
        revision=revision,
        held_out_check=(
            f"compare cvrisk on blocked_folds (contiguous blocks ordered by {name}) before "
            "and after the time term; the revision is supported when the blocked risk falls "
            "and this check's p-value rises above alpha. For forecasting, use rolling_folds"
        ),
        evidence={
            "autocorrelation": [float(a) for a in acf],
            "lag_one": lag_one,
            "lags": int(lags),
            "white_noise_band": float(bound),
            "lags_outside_band": beyond,
            "series_length": int(length),
            "times": [float(t) for t in unique_times],
            "chisq_p_value": float(chisq_p),
            "permutations": int(perms),
            "residuals": "response",
        },
    )


# ---------------------------------------------------------------------------
# Influence check
# ---------------------------------------------------------------------------


def influence_check(model, *, alpha: float = ALPHA, refit_top: int = 5,
                    draws: int = 200, seed: int = 0) -> InfluenceFinding:
    """Use leverage and deletion refits to assess changes associated with individual rows.

    The measures come from the hat-matrix diagonal, with no refit: Cook's
    distance ``D_i = r_i^2 h_i / (df sigma^2 (1 - h_i)^2)`` (the size of the
    change in every fitted value when row ``i`` is deleted), the leave-one-out
    residual ``r_i / (1 - h_i)`` and the externally studentized deleted
    residual ``t_i``, where ``r_i`` is the response residual, ``h_i`` the
    leverage, ``df`` the trace of the hat matrix and ``sigma`` the residual
    standard deviation. A row is flagged when the two-sided Student-t p-value
    of ``t_i``, on ``n - df - 1`` degrees of freedom and Bonferroni-adjusted
    over the rows, is below ``alpha``. The flag uses ``t_i`` because its
    reference is exact for a linear smoother with normal errors, where a
    Cook's distance reference is not.

    The top ``refit_top`` flagged rows are then removed one at a time and the
    model is refitted with the same arguments, so the approximation is checked
    against an exact deletion. The refit's change in fitted values on the
    remaining rows is reported per row.

    Parameters
    ----------
    model : glmboost or gamboost fit with Gaussian (normal-errors) family
        The leave-one-out formulas are for a Gaussian linear smoother. Other
        families are refused by name.
    alpha : float
        Bonferroni-adjusted level of the flag.
    refit_top : int
        How many flagged rows to refit without, largest Cook's distance first.
    draws, seed
        Null datasets for the false-alarm rate, and the random seed.

    Returns
    -------
    InfluenceFinding
        ``rows`` are the flagged positions. ``refits`` gives the exact deletion
        change for each refitted row.

    Notes
    -----
    The false-alarm rate simulates residuals with variance
    ``sigma^2 (1 - h_i)`` and treats them as independent, because only the
    hat-matrix diagonal is available. The Student-t reference is exact for a
    linear smoother with normal errors and approximate for a boosting fit, so
    the simulated rate is the number to read, not the nominal ``alpha``.
    """
    _require_mean_model(model)
    from mboostrs.explore import _is_gaussian_contract

    family = getattr(model, "_family", None)
    if family is not None and not _is_gaussian_contract(family):
        raise ValueError(
            "influence_check() is defined for a Gaussian (normal-errors) fit, because "
            "its leave-one-out measures are linear-smoother formulas. This model was "
            f"fitted with {type(family).__name__}. Refit with Gaussian or GaussReg, "
            "or use a refit-based deletion check on the rows of interest"
        )
    weights = getattr(model, "_weights", None)
    if weights is not None and not np.allclose(np.asarray(weights, dtype=float), 1.0):
        raise NotImplementedError(
            "influence_check() does not yet take case weights: the leave-one-out "
            "formulas assume unit weights. Fit without weights= to run it"
        )

    residuals = _residuals(model)
    hat = np.asarray(model.hatvalues(), dtype=np.float64).ravel()
    n = residuals.shape[0]
    out = _rust_call("residual_influence", residuals, hat, alpha)
    df, sigma, loo, cooks, studentized, row_p, flagged_rows = out
    loo = np.asarray(loo)
    cooks = np.asarray(cooks)
    row_p = np.asarray(row_p)
    studentized = np.asarray(studentized)
    rate = _rust_call("residual_influence_false_alarm_rate", hat, sigma, draws, seed, alpha)

    rows = tuple(int(i) for i in flagged_rows)
    refits = ()
    if rows and refit_top > 0:
        top = sorted(rows, key=lambda i: -cooks[i])[:refit_top]
        refits = tuple(_deletion_refit(model, i) for i in top)

    flagged = bool(rows)
    if flagged:
        worst = max(rows, key=lambda i: cooks[i])
        pattern = (
            f"{len(rows)} of {n} rows move the fitted values beyond the Bonferroni bound "
            f"at alpha = {alpha}; the largest Cook's distance is {cooks[worst]:.3g} at "
            f"row {worst} (leverage {hat[worst]:.3g}, leave-one-out residual "
            f"{loo[worst]:.4g} response units)."
        )
        listed = ", ".join(str(i) for i in rows[:10])
        if len(rows) > 10:
            listed += ", ..."
        revision = (
            f"inspect rows [{listed}] (zero-based positions) for data-entry or "
            "measurement errors first. If they are errors, correct or drop them and refit"
        )
        if refits:
            drop = ", ".join(str(r["row"]) for r in refits)
            biggest = max(r["max_change"] for r in refits)
            revision += (
                f". If they are real, refit without rows [{drop}] and report both fits: "
                f"refitting without them changes the fitted values on the kept rows by "
                f"up to {biggest:.4g} response units"
            )
    else:
        pattern = (
            f"No row moves the fitted values beyond the Bonferroni bound at alpha = {alpha}; "
            f"the largest Cook's distance is {float(np.max(cooks)):.3g}."
        )
        revision = (
            "no row is indicated by this check; keep all rows, since deleting none of "
            "them changes the fit beyond the bound"
        )
    return InfluenceFinding(
        check="influence",
        flagged=flagged,
        pattern=pattern,
        statistic="largest Cook's distance",
        value=float(np.max(cooks)),
        units="Cook's distance; dimensionless",
        p_value=float(np.min(np.minimum(1.0, n * row_p))),
        reference=(
            f"Student-t reference on the studentized deleted residual with {n - df - 1:.3g} df, "
            f"Bonferroni-adjusted over {n} rows (smallest adjusted p-value shown)"
        ),
        false_alarm_rate=float(rate),
        false_alarm_draws=int(draws),
        assumption="no few rows drive the fit; the fitted function is stable under deleting any one row",
        alternatives=(
            "a data-entry or measurement error in the flagged rows",
            "a true extreme value that a heavier-tailed family would fit without the rows "
            "moving the mean (compare Laplace or Huber by held-out loss)",
            "a predictor value no other row has, where the fitted curve is extrapolated",
        ),
        revision=revision,
        held_out_check=(
            "score the full fit and each deletion refit on rows neither of them used in "
            "fitting (holdout or cv_risk folds); the deletion is supported when its "
            "held-out loss is no worse than the full fit's"
        ),
        evidence={
            "df": float(df),
            "sigma": float(sigma),
            "leverage_sum": float(np.sum(hat)),
            "mean_leverage": float(np.mean(hat)),
            "high_leverage_threshold": float(2.0 * df / n),
            "high_leverage_rows": [int(i) for i in np.flatnonzero(hat > 2.0 * df / n)],
            "refit_top": int(refit_top),
        },
        rows=rows,
        cooks=cooks,
        loo_residual=loo,
        studentized=studentized,
        leverage=hat,
        refits=refits,
    )


def _deletion_refit(model, row: int) -> dict:
    """Refit the model without ``row``, and measure the change on the kept rows."""
    from mboostrs._refit import require_row_subsettable

    spec = require_row_subsettable(model, "influence_check")
    n = int(np.asarray(model.fitted()).shape[0])
    keep = np.delete(np.arange(n), row)
    refit = spec.subset(keep).build()
    full = np.asarray(model.fitted(), dtype=np.float64).ravel()[keep]
    changed = np.abs(np.asarray(refit.fitted(), dtype=np.float64).ravel() - full)
    return {
        "row": int(row),
        "max_change": float(changed.max()),
        "mean_change": float(changed.mean()),
        "rows_kept": int(keep.shape[0]),
    }


# ---------------------------------------------------------------------------
# All checks
# ---------------------------------------------------------------------------


def revision_checks(model, *, cluster=None, time=None, alpha: float = ALPHA,
                    permutations: int = 999, draws: int = 200, seed: int = 0,
                    refit_top: int = 5) -> RevisionReport:
    """Run the cluster, time and influence checks and report their revisions.

    ``cluster`` and ``time`` are passed to :func:`cluster_residual_check` and
    :func:`time_residual_check`; a check whose argument is None is skipped.
    The influence check always runs. Every check uses the same ``alpha`` and
    the same seed, so a rerun on the same fit gives the same report.

    Returns
    -------
    RevisionReport
        ``findings`` in the order cluster, time, influence.
    """
    findings: list[RevisionFinding] = []
    if cluster is not None:
        findings.append(cluster_residual_check(
            model, cluster, alpha=alpha, permutations=permutations, draws=draws, seed=seed))
    if time is not None:
        findings.append(time_residual_check(
            model, time, alpha=alpha, permutations=permutations, draws=draws, seed=seed))
    findings.append(influence_check(
        model, alpha=alpha, refit_top=refit_top, draws=draws, seed=seed))
    return RevisionReport(findings=tuple(findings))


def _rust_call(name: str, *args):
    from mboostrs import _rust

    return getattr(_rust, name)(*args)


__all__ = [
    "InfluenceFinding",
    "RevisionFinding",
    "RevisionReport",
    "cluster_residual_check",
    "influence_check",
    "revision_checks",
    "time_residual_check",
]
