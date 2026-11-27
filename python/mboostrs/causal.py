"""Causal estimands built from explicit assumptions and cross-fitted nuisances.

This module does not turn a predictive model into a causal one. A
:class:`CausalSpec` records the intervention, estimand, target population and
adjustment assumptions. :func:`aipw` then estimates a binary-treatment ATE or
ATT with nuisance predictions fitted outside each observation's fold.
Identification still depends on consistency, conditional exchangeability,
positivity and no interference; the result carries those assumptions.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np

from mboostrs._extrapolation import (
    POLICIES,
    CollectedRows,
    ExtrapolationWarning,
    collecting,
    one_summary,
    summarize,
)
from mboostrs._frame import column_names, read_column
from mboostrs._names import quote_name, quote_response
from mboostrs._output import export_result, json_safe, output_frame
from mboostrs._validate import check_int_at_least, check_positive, check_unit_interval
from mboostrs._warning_collect import once_per_call
from mboostrs.folds import FoldScheme, stratified_folds
from mboostrs.selection import GamRefit

__all__ = [
    "AIPWResult",
    "CausalSpec",
    "EffectModel",
    "NegativeControlResult",
    "TrapFinding",
    "TrapReport",
    "aipw",
    "causal_trap_report",
    "confounding_benchmark",
    "confounding_sensitivity",
    "effect_model",
    "CITest",
    "ci_test",
    "negative_control",
    "OverlapReport",
    "overlap_report",
    "OverlapRanking",
    "rank_by_effect",
    "IVResult",
    "iv_effect",
]


_IDENTIFICATION_ASSUMPTIONS = (
    "consistency: each observed outcome equals the potential outcome under the observed treatment",
    "conditional exchangeability: potential outcomes are independent of treatment given the baseline covariates",
    "positivity: both treatment levels have positive probability across the target covariate support",
    "no interference: one unit's treatment does not change another unit's outcome",
)


#: Propensity bounds that define the overlap region when none are given.
DEFAULT_OVERLAP_BOUNDS = (0.02, 0.98)

class _FromSpec:
    """The default of an argument that a :class:`CausalSpec` can supply."""

    def __repr__(self):  # pragma: no cover - display only
        return "from_spec"


#: Default of the arguments ``propensity_clip``, ``outcome_missing`` and
#: ``overlap_bounds``. A call that leaves one unset takes the value the
#: :class:`CausalSpec` carries; a value the caller passes always wins.
FROM_SPEC = _FromSpec()

@dataclass(frozen=True)
class TrapFinding:
    """One design problem a causal claim would rest on.

    ``severity`` is ``"blocking"`` when the declared estimand is not
    identified on this data (a column is absent, an arm is empty, one arm
    of a level never appears), and ``"advisory"`` when the estimate is
    computable but describes a smaller or different population than the
    spec's wording suggests.
    """

    name: str
    severity: str
    message: str

    def __str__(self):  # pragma: no cover - display only
        return f"[{self.severity}] {self.name}: {self.message}"


@dataclass(frozen=True)
class TrapReport:
    """Findings from checking a declared causal design against the supplied data."""

    findings: tuple[TrapFinding, ...] = ()
    spec: dict = field(default_factory=dict)

    @property
    def blocking(self):
        """Findings that leave the estimand unidentified."""
        return tuple(f for f in self.findings if f.severity == "blocking")

    @property
    def advisory(self):
        """Findings that change which population the estimate describes."""
        return tuple(f for f in self.findings if f.severity == "advisory")

    @property
    def clean(self):
        """True when the design raises no finding at all."""
        return not self.findings

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the findings."""
        return export_result(
            self,
            include_arrays=include_arrays,
            extras={
                "n_blocking": len(self.blocking),
                "n_advisory": len(self.advisory),
                "findings": [
                    {"name": f.name, "severity": f.severity, "message": f.message}
                    for f in self.findings
                ],
            },
        )

    def __str__(self):  # pragma: no cover - display only
        if self.clean:
            return "causal design: no finding"
        return "\n".join(str(f) for f in self.findings)


#: A categorical covariate with at most this many levels is checked arm by
#: arm for positivity; beyond it, the levels are too sparse for the check to
#: mean anything.
_POSITIVITY_MAX_LEVELS = 20

#: Bins a continuous adjustment column is cut into for the same check.
_POSITIVITY_BINS = 10


def causal_trap_report(frame, spec, *, min_per_arm=5, min_arm_share=0.01,
                       max_missing_share=0.2, outcome_missing=FROM_SPEC):
    """Check a declared causal design against the data before estimating it.

    A :class:`CausalSpec` fixes the intervention, the timing and the
    adjustment set in words. This function reports where the frame
    contradicts that account, and where the frame cannot check it at all --
    causes are not in the data, so a spec whose timing cannot be verified
    says so rather than passing silently. Run it before :func:`aipw`: a
    blocking finding means the estimand is not identified on these rows,
    not that the estimator would have returned a wider interval.

    Parameters
    ----------
    frame : mapping of column name to array
    spec : CausalSpec
        The design to check.
    min_per_arm : int
        Rows the smaller arm must contain.
    min_arm_share : float
        Share of rows the smaller arm must reach.
    max_missing_share : float
        Missing share above which an adjustment column changes the
        population the adjustment set represents.
    outcome_missing : {"error", "ipw"}, optional
        How the outcome's missing values are handled by :func:`aipw`. Defaults
        to the spec's ``outcome_missing``. With ``"ipw"`` the estimate targets
        all rows, and the report says so in place of the subpopulation note
        it gives for the default.

    Returns
    -------
    TrapReport
    """
    from mboostrs._frame import column_names, read_column

    outcome_policy = _check_outcome_missing(
        _spec_default(spec, "outcome_missing", outcome_missing))
    min_per_arm = check_int_at_least("min_per_arm", min_per_arm, 1)
    min_arm_share = check_unit_interval("min_arm_share", min_arm_share)
    # A threshold of 1.0 is meaningful here ("flag any missingness"), so this
    # bound is closed where `check_unit_interval` is open.
    if not 0.0 <= max_missing_share <= 1.0:
        raise ValueError(
            f"max_missing_share must be in [0, 1], got {max_missing_share!r}")
    columns = set(column_names(frame))
    findings = []

    # post_treatment columns are not checked here: a frame may omit them on
    # purpose (the training file holds them, the test file does not), and the
    # design does not adjust for them, so their absence blocks nothing.
    named = [spec.treatment, spec.outcome, *spec.covariates]
    absent = sorted({n for n in named if n not in columns})
    if absent:
        findings.append(TrapFinding(
            "columns_absent", "blocking",
            f"the design names columns the frame does not have: {absent}"))

    def flag(name, severity, message):
        findings.append(TrapFinding(name, severity, message))

    if spec.treatment in columns:
        treatment = read_column(frame, spec.treatment)
        values = treatment.values
        control, treated = spec.treatment_contrast
        observed = set(np.unique(values[~np.isnan(values)]).tolist())
        if len(observed) < 2:
            flag("no_treatment_variation", "blocking",
                 f"{spec.treatment!r} takes {len(observed)} distinct value(s); "
                 "no contrast is observed")
        else:
            absent_labels = [arm for arm in (control, treated)
                            if arm not in observed]
            if absent_labels:
                flag("contrast_absent", "blocking",
                     f"the declared treatment contrast {control!r} vs {treated!r} "
                     f"has no rows for {absent_labels}; observed values are "
                     f"{sorted(observed)[:6]}")
            if len(observed) > 2:
                extra = sorted(observed - {control, treated})[:6]
                flag("extra_treatment_levels", "advisory",
                     f"{spec.treatment!r} takes {len(observed)} values; rows at "
                     f"{extra} sit outside the declared contrast, so the "
                     "estimate describes only the declared pair's population")
            arms = [np.sum(values == label) for label in (control, treated)]
            for label, count in zip((control, treated), arms, strict=True):
                share = count / max(len(values), 1)
                if count < min_per_arm or share < min_arm_share:
                    flag("arm_too_small", "blocking",
                         f"arm {label!r} has {count} rows ({share:.3%} of the "
                         f"frame); the design needs at least {min_per_arm} and "
                         f"{min_arm_share:.1%}")
        if treatment.n_missing:
            flag("treatment_missing", "advisory",
                 f"{spec.treatment!r} is missing for {treatment.n_missing} rows; "
                 "those rows are treated as untreated or dropped, and which it "
                 "is changes the estimand")

    if spec.outcome in columns:
        outcome = read_column(frame, spec.outcome)
        if outcome.n_missing and outcome_policy == "ipw":
            flag("outcome_missing_ipw", "advisory",
                 f"{spec.outcome!r} is missing for {outcome.n_missing} rows. With "
                 f"outcome_missing='ipw' the estimate targets all {len(outcome.values)} "
                 "rows, observed or not: a response model for P(outcome observed | "
                 "covariates, treatment) is cross-fitted and its inverse weights the "
                 "observed rows. That assumes the outcome is missing at random given "
                 "those variables, which these rows cannot check")
        elif outcome.n_missing:
            flag("outcome_missing", "advisory",
                 f"{spec.outcome!r} is missing for {outcome.n_missing} rows; "
                 f"the estimate then covers the {len(outcome.values) - outcome.n_missing} "
                 "rows with an observed outcome, a subpopulation of the spec's target")
    treatment_values = (read_column(frame, spec.treatment).values
                        if spec.treatment in columns else None)

    def one_arm_groups(values, groups, labels):
        """The groups in which one declared arm has no row at all."""
        sparse = []
        for group, label in zip(groups, labels, strict=True):
            rows = values == group
            for arm in spec.treatment_contrast:
                if np.sum(rows & (treatment_values == arm)) == 0:
                    sparse.append(label)
                    break
        return sparse

    for name in spec.covariates:
        if name not in columns:
            continue
        column = read_column(frame, name)
        share = column.n_missing / max(len(column.values), 1)
        if share > max_missing_share:
            flag("covariate_missing", "advisory",
                 f"adjustment column {name!r} is missing for {share:.1%} of rows; "
                 f"imputing it changes what {name!r} contributes to exchangeability")
        if treatment_values is None:
            continue
        values = column.values
        finite = values[~np.isnan(values)] if values.dtype.kind == "f" else values
        distinct = np.unique(finite)
        if column.kind == "categorical" and 1 < column.n_levels <= _POSITIVITY_MAX_LEVELS:
            sparse = one_arm_groups(values, np.arange(column.n_levels),
                                    list(column.levels or range(column.n_levels)))
        elif 1 < len(distinct) <= _POSITIVITY_MAX_LEVELS:
            # A numeric column with a handful of values is a category in
            # all but name: arm support is checked value by value.
            sparse = one_arm_groups(values, distinct, [float(v) for v in distinct])
        elif values.dtype.kind == "f" and len(distinct) > _POSITIVITY_MAX_LEVELS:
            # A continuous covariate is checked by decile, which is what a
            # propensity histogram shows the analyst anyway.
            edges = np.quantile(finite, np.linspace(0, 1, _POSITIVITY_BINS + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                continue
            codes = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
            labels = [f"decile {i + 1}" for i in range(len(edges) - 1)]
            sparse = one_arm_groups(codes.astype(float), np.arange(len(labels)), labels)
        else:
            sparse = []
        if sparse:
            flag("positivity_by_level", "advisory",
                 f"level(s) {sparse[:6]} of {name!r} appear under only one "
                 "treatment arm; within those levels the contrast is not "
                 "observed, and the estimate rests on the levels that are")

    has_time = any(read_column(frame, n).kind == "datetime" for n in columns)
    if not has_time:
        flag("timing_unverifiable", "advisory",
             f"no column in the frame is a timestamp, so time_zero "
             f"({spec.time_zero!r}) and the follow-up window ({spec.follow_up!r}) "
             "stay a modeler-stated claim: nothing in these rows can check them")

    return TrapReport(tuple(findings), spec.to_dict())


def _spec_default(spec, name, value):
    """``value`` when the caller passed one, otherwise the spec's field `name`.

    With no spec (a bare propensity array), or when the spec does not carry
    the field, the field's dataclass default is used. Only :data:`FROM_SPEC`
    is replaced, so an explicit ``None`` (for example ``propensity_clip=None``)
    still means no clipping.
    """
    if value is not FROM_SPEC:
        return value
    if spec is None:
        return CausalSpec.__dataclass_fields__[name].default
    return getattr(spec, name)


@dataclass(frozen=True)
class CausalSpec:
    """A binary-treatment causal question and its adjustment design."""

    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    estimand: str = "ate"
    outcome_type: str = "continuous"
    target_population: str = "sample population"
    unit: str = "observation"
    time_zero: str | None = None
    follow_up: str | None = None
    treatment_contrast: tuple[object, object] = (0, 1)
    post_treatment: tuple[str, ...] = ()
    adjustment_rationale: str = ""
    assumptions: tuple[str, ...] = _IDENTIFICATION_ASSUMPTIONS
    #: Propensity clipping ``(lower, upper)``, or ``None`` for no clipping. The
    #: default matches :func:`effect_model` and :func:`aipw`.
    propensity_clip: tuple[float, float] | None = (0.01, 0.99)
    #: How a missing outcome is handled: ``"error"`` refuses it, ``"ipw"``
    #: targets all rows through a response model. The default is ``"error"``.
    outcome_missing: str = "error"
    #: Propensity bounds that define the overlap region, read by
    #: :func:`overlap_report` when the call does not pass ``bounds``.
    overlap_bounds: tuple[float, float] = DEFAULT_OVERLAP_BOUNDS

    def __post_init__(self):
        if not self.treatment or not self.outcome:
            raise ValueError("treatment and outcome must be non-empty column names")
        if self.treatment == self.outcome:
            raise ValueError("treatment and outcome must be different columns")
        covariates = tuple(self.covariates)
        if not covariates or any(not isinstance(name, str) or not name for name in covariates):
            raise ValueError("covariates must contain baseline column names")
        if len(set(covariates)) != len(covariates):
            raise ValueError("covariates must not contain duplicate columns")
        forbidden = {self.treatment, self.outcome} & set(covariates)
        if forbidden:
            raise ValueError(f"treatment/outcome cannot also be covariates: {sorted(forbidden)}")
        post = set(self.post_treatment) & set(covariates)
        if post:
            raise ValueError(
                f"post-treatment variables cannot enter the baseline adjustment set: {sorted(post)}"
            )
        if self.estimand not in ("ate", "att", "ato"):
            raise ValueError("estimand must be 'ate', 'att', or 'ato'")
        if self.outcome_type not in ("continuous", "binary"):
            raise ValueError("outcome_type must be 'continuous' or 'binary'")
        if not self.time_zero or not self.follow_up:
            raise ValueError("time_zero and follow_up must be stated for a causal estimand")
        if not self.adjustment_rationale.strip():
            raise ValueError(
                "adjustment_rationale must explain why the baseline covariates suffice; "
                "predictive importance is not an identification argument"
            )
        if len(self.treatment_contrast) != 2 or self.treatment_contrast[0] == self.treatment_contrast[1]:
            raise ValueError("treatment_contrast must contain distinct control and treated labels")
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "post_treatment", tuple(self.post_treatment))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        clip = self.propensity_clip
        object.__setattr__(self, "propensity_clip",
                           _check_clip(tuple(clip) if isinstance(clip, list) else clip))
        object.__setattr__(self, "outcome_missing", _check_outcome_missing(self.outcome_missing))
        bounds = self.overlap_bounds
        object.__setattr__(self, "overlap_bounds",
                           _check_bounds(tuple(bounds) if isinstance(bounds, list) else bounds))

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; every field is a small scalar or tuple."""
        return export_result(
            self,
            include_arrays=include_arrays,
            extras={"n_covariates": len(self.covariates)},
        )

    def describe(self) -> str:
        """The intervention and identification claim in plain text."""
        control, treated = self.treatment_contrast
        timing = ""
        if self.time_zero:
            timing += f" Time zero: {self.time_zero}."
        if self.follow_up:
            timing += f" Follow-up: {self.follow_up}."
        return (
            f"{self.estimand.upper()} of setting {self.treatment} from {control!r} "
            f"to {treated!r} on {self.outcome} ({self.outcome_type}), "
            f"for {self.target_population}; "
            f"unit: {self.unit}. Adjust for baseline {list(self.covariates)}.{timing}"
        )


#: Columns of ``AIPWResult.to_frame``, kept for the zero-row frame.
AIPW_COLUMNS = (
    "row", "fold", "fold_assignments", "treatment", "outcome", "propensity",
    "mu0", "mu1", "score", "influence",
)

#: Columns of ``AIPWResult.subgroup_effect``.
SUBGROUP_COLUMNS = (
    "level", "n", "weight", "estimate", "standard_error", "lower", "upper",
)


@dataclass(frozen=True)
class AIPWResult:
    """Cross-fitted AIPW estimate with reconstructible row contributions."""

    estimate: float
    standard_error: float | None
    interval: tuple[float, float] | None
    level: float
    spec: CausalSpec
    treatment: np.ndarray
    outcome: np.ndarray
    #: Cross-fitted P(treatment = treated | covariates) of each row, unclipped,
    #: averaged over repeats when repeats > 1. Each value comes from a model
    #: fitted on the folds that exclude the row, and the estimate is built from
    #: these values. Use this field for the training rows. For new rows use
    #: :meth:`AIPWResult.predict_propensity`, which applies a model refit on all
    #: rows and clips the values, so its values on the training rows differ from
    #: this field.
    propensity: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    scores: np.ndarray
    influence: np.ndarray
    fold_id: np.ndarray
    diagnostics: dict
    nuisance: dict
    fold_ids: np.ndarray | None = None
    #: 1 where the outcome is observed, 0 where it is missing.
    observed: np.ndarray | None = None
    #: Cross-fitted P(outcome observed | covariates, treatment); all ones when
    #: no outcome was missing.
    response_propensity: np.ndarray | None = None
    #: The observation weights, when the estimate was weighted.
    weights: np.ndarray | None = None
    #: Zero-extra-fit overlap-weighted (ATO) companion when ``estimand="ate"``,
    #: ``target="all"``, and at least 5% of rows fall outside ``overlap_bounds``.
    ato_companion: dict | None = None
    #: The propensity model fitted on all rows, which :meth:`predict_propensity`
    #: scores new rows with. None when the result was built without one.
    _propensity_model: object = field(default=None, repr=False, compare=False)
    #: The training data, read by :meth:`subgroup_effect` (column names) and
    #: :meth:`effect_surface` (covariates). A reference, not a copy.
    _data: object = field(default=None, repr=False, compare=False)
    #: Cluster labels of the fold scheme, or None; they define the variance.
    _groups: object = field(default=None, repr=False, compare=False)
    #: Whether the fold scheme was ordered (no variance is defined then).
    _ordered: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        n = len(np.asarray(self.scores))
        for name in ("treatment", "outcome", "propensity", "mu0", "mu1",
                     "scores", "influence", "fold_id"):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("observed", "response_propensity", "weights"):
            if getattr(self, name) is None:
                continue
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        fold_ids = (np.asarray(self.fold_id)[:, None] if self.fold_ids is None
                    else np.asarray(self.fold_ids).copy())
        if fold_ids.ndim != 2 or fold_ids.shape[0] != n:
            raise ValueError(
                f"fold_ids must have shape ({n}, repeats), got {fold_ids.shape}"
            )
        if not np.array_equal(fold_ids[:, 0], self.fold_id):
            raise ValueError("fold_id must equal the first column of fold_ids")
        fold_ids.setflags(write=False)
        object.__setattr__(self, "fold_ids", fold_ids)

    def to_frame(self, output="auto"):
        """One row per observation, sufficient to reconstruct the estimate."""
        rows = [
            {
                "row": i, "fold": int(self.fold_id[i]),
                "fold_assignments": self.fold_ids[i].tolist(),
                "treatment": float(self.treatment[i]), "outcome": float(self.outcome[i]),
                "propensity": float(self.propensity[i]), "mu0": float(self.mu0[i]),
                "mu1": float(self.mu1[i]), "score": float(self.scores[i]),
                "influence": float(self.influence[i]),
            }
            for i in range(len(self.scores))
        ]
        return output_frame(rows, output, columns=AIPW_COLUMNS)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row contribution tracks only on request.

        The scalar estimate, standard error, interval, level and specification
        are always present; the per-row propensity, means, scores, influence
        and fold ids are gated behind ``include_arrays``.
        """
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("treatment", "outcome", "propensity", "mu0", "mu1",
                    "scores", "influence", "fold_id", "fold_ids", "observed",
                    "response_propensity", "weights"),
            extras={"n": len(self.scores)},
        )

    def __repr__(self) -> str:
        interval = "available" if self.interval is not None else "unavailable"
        return (
            f"AIPWResult({self._score_estimand()}={self.estimate:.6g}, "
            f"se={self.standard_error!r}, interval={interval!r}, "
            f"n={len(self.scores)}, repeats={self.fold_ids.shape[1]})"
        )

    def predict_propensity(self, newdata):
        """Estimated P(treatment = treated | covariates) of each row of ``newdata``.

        The model is the propensity model that :func:`aipw` fitted on all rows,
        with the same formula, smooth flag, step count and tuning as its
        cross-fitted propensity fits. The values are clipped to the
        ``propensity_clip`` of the fit, and the result is a NumPy array with one
        value per row, in row order. ``newdata`` needs every covariate of the
        specification; a missing covariate raises ``ValueError`` naming it. Use
        :attr:`propensity` for the training rows, where the estimate was computed;
        use this method for new rows or to check the model on any rows.
        """
        if self._propensity_model is None:
            raise ValueError(
                "this AIPWResult has no stored propensity model, so it cannot score new "
                "rows; refit with aipw() to get one")
        names = column_names(newdata)
        missing = [name for name in self.spec.covariates if name not in names]
        if missing:
            raise ValueError(
                f"predict_propensity needs every covariate of the specification, and "
                f"newdata lacks {missing}; add those columns to newdata")
        covariates = _modifier_values(newdata, self.spec.covariates)
        e = _nuisance_predict(self._propensity_model, covariates,
                              self.nuisance["extrapolation"])
        e = np.asarray(e, dtype=np.float64).ravel()
        clip = self.diagnostics["propensity_clip"]
        if clip is not None:
            e = np.clip(e, *clip)
        return e

    def _check_score_mean_estimand(self, owner):
        if self.target == "overlap" or self.spec.estimand != "ate":
            raise ValueError(
                f"{owner} works on the ATE augmented scores, whose mean is the ATE "
                f"estimate; this result targets {self._label()}, whose estimate is "
                "not a mean of the scores. Refit with aipw(target='all') for an ATE")

    def subgroup_effect(self, by, *, level=None, output="auto"):
        """Average treatment effect within subgroups, with a standard error and interval.

        Each subgroup's effect is the weighted mean of the augmented (AIPW)
        scores of its rows, :attr:`scores`. Its standard error is computed
        as :func:`aipw` computes the standard error of the overall estimate,
        with the scores of the subgroup as influence values: frequency weights
        give ``sum(w (score - mean)^2) / (W (W - 1))``, and a grouped fold
        scheme gives the cluster-robust sandwich over the clusters present in
        the subgroup, with a Student-t interval on ``clusters - 1`` degrees of
        freedom. A subgroup that holds every row therefore reproduces
        :attr:`estimate` and :attr:`standard_error`. The standard error is None
        for a subgroup with total weight of one or less or fewer than two
        clusters, and for an ordered fold scheme.

        Parameters
        ----------
        by : str or array-like
            A boolean mask over the training rows (one subgroup, the rows where
            it is true), the name of a column of the training data (one subgroup
            per distinct value), or an array of labels with one entry per
            training row. Missing values in the labels are refused.
        level : float, optional
            Interval level; default is :attr:`level`, the level ``aipw`` used.
        output : {"auto", "pandas", "polars", "records"}
            Result format.

        Returns
        -------
        One row per subgroup with ``level`` (the subgroup value; ``True`` for a
        mask), ``n`` (rows), ``weight`` (total observation weight), ``estimate``,
        ``standard_error``, ``lower`` and ``upper``.

        Notes
        -----
        The subgroup effects use nuisance models fitted on all rows. The
        variance does not account for choosing subgroups after looking at the
        data; a subgroup effect is as credible as the subgroup is prespecified.
        The method needs an ATE result (``target='all'``). With
        ``repeat_aggregation='median'`` the overall estimate is the median over
        repeats, and a subgroup holding every row gives the mean of the
        averaged scores, which differs from it.
        """
        from mboostrs import _rust
        from mboostrs.cvrisk import _t_quantile

        self._check_score_mean_estimand("subgroup_effect")
        level = self.level if level is None else check_unit_interval("level", level)
        n = len(self.scores)
        if isinstance(by, str):
            if self._data is None or by not in column_names(self._data):
                raise ValueError(
                    f"subgroup_effect(by={by!r}): the training data has no column "
                    f"{by!r}; pass a column name of the data given to aipw, or a "
                    "boolean mask with one entry per training row")
            labels = _modifier_values(self._data, [by])[by]
        else:
            labels = np.asarray(by)
            if labels.shape != (n,):
                raise ValueError(
                    f"by must have one entry per training row ({n}); got shape {labels.shape}")
        if labels.dtype == bool:
            if not labels.any():
                raise ValueError("the boolean mask selects no rows")
            levels = np.array([True])
            codes = np.where(labels, 0, -1).astype(np.int64)
        else:
            if labels.dtype.kind == "f" and np.isnan(labels).any():
                raise ValueError("by has missing values; fill or drop them first")
            levels, codes = np.unique(labels, return_inverse=True)
            codes = codes.astype(np.int64)
        weights = (np.ones(n) if self.weights is None
                   else np.asarray(self.weights, dtype=np.float64))
        groups = None
        if self._groups is not None:
            groups = np.unique(np.asarray(self._groups), return_inverse=True)[1]
            groups = groups.astype(np.uint64)
        summary = _rust.subgroup_scores(
            np.asarray(self.scores, dtype=np.float64), weights, codes, len(levels), groups)
        probability = .5 + level / 2
        rows = []
        for value, (count, weight, mean, variance, n_clusters) in zip(levels, summary, strict=True):
            se = lower = upper = None
            if variance is not None and not self._ordered:
                se = float(np.sqrt(max(variance, 0.0)))
                critical = (NormalDist().inv_cdf(probability) if n_clusters is None
                            else _t_quantile(probability, n_clusters - 1))
                lower, upper = float(mean - critical * se), float(mean + critical * se)
            rows.append({
                "level": value.item() if hasattr(value, "item") else value,
                "n": int(count), "weight": float(weight), "estimate": float(mean),
                "standard_error": se, "lower": lower, "upper": upper,
            })
        return output_frame(rows, output, columns=SUBGROUP_COLUMNS)

    def effect_surface(self, formula, *, mstop=200, nu=0.1, **gamboost_args):
        """Doubly robust second stage: regress the augmented scores on covariates.

        Fits ``gamboost`` with the augmented scores :attr:`scores` as the
        response and ``formula`` as the right-hand side, on the training data
        and the observation weights of the ``aipw`` fit. The expected score of
        a row given its covariates is the conditional average treatment effect,
        so ``predict`` of the returned model estimates the effect as a function
        of the covariates (the DR-learner of Kennedy, 2023). The scores are
        stored under a generated column name that differs from every column of
        the data.

        Parameters
        ----------
        formula : str
            The right-hand side, for example ``"bbs(age, df=4) + bols(sex)"``;
            a leading ``"~"`` is accepted. Every variable must be a column of
            the training data.
        mstop, nu : int, float
            Boosting steps and step length of the second stage. Choose ``mstop``
            by cross-validation (``mb.cvrisk``) for a tuned surface.
        **gamboost_args
            Passed to ``mboostrs.gamboost`` (``family`` defaults to Gaussian).

        Returns
        -------
        The fitted ``GAMBoostModel``; ``model.predict(newdata)`` gives the
        estimated conditional effect.

        Notes
        -----
        The scores are noisy pseudo-outcomes: their mean given the covariates
        is the effect, but each score has a variance far above that of an
        outcome, so the surface is estimated with wide uncertainty. Intervals
        or standard errors from the second-stage model treat the scores as
        fixed observed data and ignore the error of the nuisance models that
        produced them and of the choice of ``mstop``; they understate the
        uncertainty. The surface describes the effect only through the
        covariates in ``formula``. The method needs an ATE result
        (``target='all'``).
        """
        import mboostrs as mb
        from mboostrs.families import Gaussian

        self._check_score_mean_estimand("effect_surface")
        if self._data is None:
            raise ValueError("this AIPWResult holds no training data; refit with aipw()")
        rhs = formula.strip().lstrip("~").strip() if isinstance(formula, str) else ""
        if not rhs:
            raise ValueError("formula must be the right-hand side string, e.g. 'bbs(x, df=4)'")
        names = column_names(self._data)
        response = "_aipw_score"
        while response in names:
            response = "_" + response
        used = [name for name in names
                if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_.])", rhs)]
        if not used:
            raise ValueError(
                f"formula {formula!r} names no column of the training data {names}")
        data = _modifier_values(self._data, used)
        data[response] = np.asarray(self.scores, dtype=np.float64)
        gamboost_args.setdefault("family", Gaussian())
        if self.weights is not None:
            gamboost_args.setdefault("weights", np.asarray(self.weights, dtype=np.float64))
        if "control" not in gamboost_args:
            gamboost_args["control"] = mb.boost_control(mstop, nu)
        return mb.gamboost(f"{quote_response(response)} ~ {rhs}", data=data, **gamboost_args)

    def balance_frame(self, output="auto"):
        """Standardized mean differences before and after treatment weighting."""
        return output_frame(self.diagnostics["balance"], output)

    def overlap_frame(self, output="auto"):
        """Row-level propensity support and estimand-specific analysis weights.

        Plot or summarize this table by treatment arm. A propensity histogram
        can show whether the two arms occupy the same covariate-support region;
        the weight column shows which observations dominate the requested ATE
        or ATT. Propensities used for estimation include the explicitly
        recorded clipping policy, when one was requested.
        """
        clip = self.diagnostics.get("propensity_clip_applied",
                                    self.diagnostics["propensity_clip"])
        propensity_used = (np.clip(self.propensity, *clip)
                           if clip is not None else self.propensity)
        weights = _effect_weights(self.treatment, propensity_used,
                                  self._score_estimand())
        if self.weights is not None:
            weights = weights * self.weights
        rows = [
            {
                "row": i,
                "treatment": int(self.treatment[i]),
                "treatment_label": ("treated" if self.treatment[i] == 1
                                    else "control"),
                "propensity": float(self.propensity[i]),
                "propensity_used": float(propensity_used[i]),
                "analysis_weight": float(weights[i]),
            }
            for i in range(len(self.scores))
        ]
        return output_frame(rows, output)

    @property
    def target(self):
        """The population the estimate describes: ``"all"`` (every row, or
        every treated row for the ATT) or ``"overlap"`` (the overlap-weighted
        ATE). Set by ``aipw(target=...)``."""
        return self.diagnostics.get("target", "all")

    def _score_estimand(self):
        if self.target == "overlap" or self.spec.estimand == "ato":
            return "overlap"
        return self.spec.estimand

    def _label(self):
        if self.spec.estimand == "ato":
            return "ATO"
        return "OVERLAP-WEIGHTED ATE" if self.target == "overlap" else self.spec.estimand.upper()

    def _target_text(self):
        if self.target == "overlap" or self.spec.estimand == "ato":
            return ("overlap-weighted ATE over all rows, each row weighted by e (1 - e) "
                    "with its cross-fitted propensity e; rows with propensity near 0 or 1 "
                    "count little, so this is not the ATE over every row")
        population = "every treated row" if self.spec.estimand == "att" else "every row"
        text = f"{self.spec.estimand.upper()} over {population}"
        share = (self.diagnostics["fraction_below_0_05"]
                 + self.diagnostics["fraction_above_0_95"])
        if share > 0.05:
            text += (f"; {share:.1%} of rows have a cross-fitted propensity below 0.05 or "
                     "above 0.95, where the outcome model is extrapolated and the estimate "
                     "is sensitive to it. aipw(target='overlap') estimates the "
                     "overlap-weighted effect instead")
        return text

    def describe(self) -> str:
        """Estimate, target, interval, overlap and the assumptions it depends on."""
        uncertainty = (
            f"{self.level:.0%} interval [{self.interval[0]:.6g}, {self.interval[1]:.6g}]"
            if self.interval is not None else
            "interval unavailable for this dependence design"
        )
        boot = self.diagnostics.get("bootstrap")
        if boot is not None:
            uncertainty += (f" (percentile interval of {boot['n_used']} of {boot['n_draws']} "
                            f"bootstrap draws of the whole procedure; {boot['n_failed']} failed)")
        return (
            f"{self._label()}={self.estimate:.6g}, {uncertainty}, "
            f"n={len(self.scores)}; propensity range "
            f"[{self.diagnostics['propensity_min']:.4g}, "
            f"{self.diagnostics['propensity_max']:.4g}].\n"
            + "Target: " + self._target_text() + ".\n"
            + "Outcome model: " + self._outcome_model_text() + ".\n"
            + "Propensity: " + self._propensity_choice_text() + ".\n"
            + "Assumptions: " + "; ".join(self.spec.assumptions)
        )

    def _propensity_choice_text(self):
        record = self.diagnostics.get("propensity_learner")
        if record is None:
            return "formula as specified"
        chosen = record["chosen"]
        if chosen == "custom":
            return "custom formula from propensity_terms"
        text = f"formula {chosen!r}"
        if len(record["candidates"]) > 1:
            deviance = record["out_of_fold_deviance"]
            others = ", ".join(f"{label!r} {value:.6g}" for label, value in deviance.items()
                               if label != chosen)
            text += (f", chosen by propensity_learner={record['requested']!r} on out-of-fold "
                     f"binomial deviance ({deviance[chosen]:.6g} against {others})")
        return text

    def _outcome_model_text(self):
        choice = self.diagnostics.get("outcome_model_choice")
        if choice is None:
            return self.diagnostics.get("outcome_model", "separate")
        used = choice["used"]
        if used == "pooled" and choice["outcome_interactions"]:
            used = "pooled with treatment interactions"
        if used == "two_part":
            used = "two-part (positive-outcome probability times the mean of the log outcome)"
        if choice["requested"] in ("auto", "cv"):
            return (f"{used}, chosen by outcome_model={choice['requested']!r} because "
                    f"{choice['reason']}")
        return f"{used}, because outcome_model={choice['requested']!r} was requested"

    def digest(self) -> str:
        """Return a concise, deterministic summary card of this AIPW estimate."""
        d = self.diagnostics
        overlap = d.get("overlap") or {}
        ci_text = (f"[{self.interval[0]:.6g}, {self.interval[1]:.6g}]"
                   if self.interval is not None else "unavailable")
        se_text = f"{self.standard_error:.6g}" if self.standard_error is not None else "None"
        prop_choice = (d.get("propensity_learner") or {}).get("chosen", "smooth")
        lines = [
            f"[mboostrs AIPWResult] estimand={self._label()}, n={len(self.scores)}, "
            f"treatment={self.spec.treatment}, outcome={self.spec.outcome}",
            f"estimate={self.estimate:.6g}, se={se_text}, {self.level:.0%} CI={ci_text}",
            f"folds={self.nuisance.get('fold_count')} (repeats={self.nuisance.get('fold_repeats')}), "
            f"outcome_model={d.get('outcome_model')}, propensity_learner={prop_choice}",
            f"propensity=[{d.get('propensity_min', float('nan')):.4g}, "
            f"{d.get('propensity_max', float('nan')):.4g}], "
            f"outside_overlap={overlap.get('share_outside', 0.0):.1%}, "
            f"ess={d.get('ipw_effective_sample_size', float('nan')):.1f}",
        ]
        if self.ato_companion is not None:
            c = self.ato_companion
            c_se = f"{c['se']:.6g}" if c["se"] is not None else "None"
            c_ci = (f"[{c['ci_lower']:.6g}, {c['ci_upper']:.6g}]"
                    if c["ci_lower"] is not None else "unavailable")
            lines.append(
                f"ato_companion={c['estimate']:.6g} (se={c_se}, CI={c_ci}, "
                f"outside_share={c['outside_overlap_share']:.1%})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class NegativeControlResult:
    """A prespecified negative-control estimate and its expected null value."""

    kind: str
    rationale: str
    expected_effect: float
    effect: AIPWResult

    def __post_init__(self):
        if self.kind not in ("exposure", "outcome"):
            raise ValueError("kind must be 'exposure' or 'outcome'")
        if not self.rationale.strip():
            raise ValueError("rationale must explain why the control effect should be null")
        if not np.isfinite(self.expected_effect):
            raise ValueError("expected_effect must be finite")
        if not isinstance(self.effect, AIPWResult):
            raise TypeError("effect must be an AIPWResult")

    @property
    def interval_excludes_expected(self) -> bool | None:
        """Whether the reported interval excludes the prespecified null."""
        if self.effect.interval is None:
            return None
        lower, upper = self.effect.interval
        return not lower <= self.expected_effect <= upper

    def to_dict(self, *, include_arrays=False):
        """The falsification verdict and its expected-null justification.

        The nested effect record keeps its scalar estimate, standard error and
        interval; its per-row contribution arrays are gated by
        ``include_arrays``, as in :meth:`AIPWResult.to_dict`.
        """
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=(),
            extras={"interval_excludes_expected": self.interval_excludes_expected},
        )

    def __repr__(self) -> str:
        return (
            f"NegativeControlResult(kind={self.kind!r}, "
            f"estimate={self.effect.estimate:.6g}, expected={self.expected_effect:.6g}, "
            f"interval_excludes_expected={self.interval_excludes_expected!r})"
        )

    def describe(self) -> str:
        """Interpret the falsification result without treating a null as proof."""
        excluded = self.interval_excludes_expected
        if excluded is True:
            conclusion = (
                "The interval excludes the expected control effect, which is evidence "
                "against at least one control-design, measurement, or identification assumption."
            )
        elif excluded is False:
            conclusion = (
                "The interval includes the expected control effect. This check did not "
                "detect the targeted bias, but it does not establish identification."
            )
        else:
            conclusion = (
                "No interval is available for this dependence design, so the control is descriptive."
            )
        return (
            f"Negative-control {self.kind}; expected effect={self.expected_effect:.6g}; "
            f"estimated effect={self.effect.estimate:.6g}. {conclusion} "
            f"Design rationale: {self.rationale}"
        )


# The message for an outcome with missing rows under ``outcome_missing="error"``.
def _missing_outcome_message(name, missing, n):
    return (
        f"outcome {name!r} is missing for {missing} of {n} rows, and the causal "
        "estimators need an observed outcome for every row. Either pass "
        "outcome_missing='ipw' to model the outcome as missing at random given the "
        "covariates and treatment, with inverse-probability weighting, or drop the "
        "rows with a missing outcome before the call, which changes the target "
        "population to the rows that have an observed outcome."
    )


def _numeric_outcome(data, name):
    """The outcome column as float64, with NaN for each None or NaN, when every
    other value is a number or a boolean; ``(None, offending value)`` otherwise.

    A column from ``to_dict()`` that holds None among its numbers is an object
    column, which :func:`~mboostrs._frame.read_column` reads as categorical. Its
    present values decide whether it is an outcome."""
    raw = np.asarray(data[name], dtype=object).ravel()
    out = np.empty(raw.size, dtype=np.float64)
    for i, value in enumerate(raw.tolist()):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out[i] = np.nan
        elif isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
            out[i] = float(value)
        else:
            return None, value
    return out, None


def _columns(data, spec: CausalSpec, *, outcome_missing="error", categorical=False,
             owner="aipw"):
    """Outcome and baseline columns as arrays.

    Numeric columns (and booleans, as 0/1) are float arrays. The outcome may
    hold NaN only under ``outcome_missing="ipw"``; a None in an otherwise
    numeric outcome column is read as NaN. A categorical baseline is refused
    unless ``categorical=True``, in which case it is an object array of label
    strings: the boosting formulas then read it as a factor, as they do for any
    string column. ``owner`` names the routine in messages."""
    values = {}
    present = set(column_names(data))
    roles = [("treatment", spec.treatment), ("outcome", spec.outcome),
             *(("covariates", name) for name in spec.covariates)]
    missing = [(role, name) for role, name in roles if name not in present]
    if missing:
        listed = "; ".join(f"{role}={name!r}" for role, name in missing)
        raise ValueError(
            f"{owner}: data has no column for {listed}. Each name in the CausalSpec "
            "arguments treatment, outcome and covariates must be a column of data.")
    for name in (spec.outcome, *spec.covariates):
        column = read_column(data, name)
        is_factor = column.levels is not None and column.kind != "boolean"
        if is_factor and name == spec.outcome:
            numeric, offending = _numeric_outcome(data, name)
            if numeric is None:
                raise NotImplementedError(
                    f"{owner} requires a numeric outcome; {name!r} is categorical (it holds "
                    f"{offending!r}). Missing values must be null or NaN in a numeric column.")
            column_values = numeric
        else:
            if is_factor and not categorical:
                raise NotImplementedError(
                    f"{owner} currently requires numeric baseline columns; {name!r} is categorical"
                )
            if is_factor:
                values[name] = _factor_labels(column, name)
                continue
            column_values = np.asarray(column.values, dtype=np.float64)
        values[name] = column_values
        if values[name].ndim != 1:
            raise ValueError(f"{name!r} must be a one-dimensional column")
        if name == spec.outcome:
            missing = np.isnan(values[name])
            if outcome_missing == "error" and missing.any():
                raise ValueError(_missing_outcome_message(
                    name, int(missing.sum()), missing.size))
            if np.any(np.isinf(values[name])):
                raise ValueError(f"{name!r} must be finite where observed (NaN marks a missing value)")
        elif not np.all(np.isfinite(values[name])):
            raise ValueError(f"{name!r} must be a finite one-dimensional column")
    return values


def _factor_labels(column, name):
    """The label of each row of a categorical column, as an object array of strings."""
    if column.n_missing:
        raise ValueError(
            f"categorical covariate {name!r} has {column.n_missing} missing "
            "values; drop those rows or fill them before the call")
    return np.array([str(column.label(code)) for code in column.values], dtype=object)


def _modifier_values(data, names):
    """Columns as the fitted models read them: numeric columns as float arrays and
    categorical columns as label arrays, so new rows are coded as the fit was."""
    values = {}
    for name in names:
        column = read_column(data, name)
        if column.levels is not None and column.kind != "boolean":
            values[name] = _factor_labels(column, name)
        else:
            values[name] = np.asarray(column.values, dtype=np.float64)
    return values


def _check_clip(propensity_clip):
    """``None`` or a ``(lower, upper)`` pair strictly inside (0, 1)."""
    if propensity_clip is None:
        return None
    if (not isinstance(propensity_clip, tuple) or len(propensity_clip) != 2
            or not 0 < propensity_clip[0] < propensity_clip[1] < 1):
        raise ValueError("propensity_clip must be None or a tuple (lower, upper) inside (0, 1)")
    return (float(propensity_clip[0]), float(propensity_clip[1]))


def _check_outcome_missing(outcome_missing):
    if outcome_missing not in ("error", "ipw"):
        raise ValueError("outcome_missing must be 'error' or 'ipw'")
    return outcome_missing


def _check_outcome_model(outcome_model):
    if outcome_model not in ("auto", "cv", "separate", "pooled", "two_part"):
        raise ValueError("outcome_model must be 'auto' (a row-count rule and the sparse-arm check pick the model), "
                         "'cv' (the model with the lowest out-of-fold loss), "
                         "'separate' (one outcome model per arm), 'pooled' (one model with "
                         "the treatment as a term) or 'two_part' (for a zero-inflated outcome "
                         "that is zero or positive: P(positive) times the mean of the log "
                         "outcome among the positive rows)")
    return outcome_model


#: The default largest step count of a nuisance search, three doublings past
#: the default first grid of 200 steps.
_MAX_MSTOP_DEFAULT = 1600

#: How aipw's messages name each nuisance model.
_NUISANCE_LABELS = {
    "propensity": "propensity",
    "response": "response (observation)",
    "outcome": "pooled outcome",
    "employment": "employment (positive outcome) model",
    "log_outcome": "log outcome among positive rows",
    "mu0": "control-arm outcome",
    "mu1": "treated-arm outcome",
}


def _sample_weights(weights, n):
    """Observation weights as frequencies: a row with weight ``w`` counts as
    ``w`` identical rows. ``None`` gives unit weights."""
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights has shape {w.shape}, expected ({n},) to match the sample")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    return w


def _smooth_baselines(values, covariates):
    """The covariates that enter a smooth term: numeric columns with at least
    ``_SMOOTH_MIN_VALUES`` distinct values. The count is taken on all rows, so
    every fold uses the same rule. A factor, or a numeric column with fewer
    distinct values, enters linearly."""
    return frozenset(name for name in covariates
                     if values[name].dtype != object
                     and len(np.unique(values[name])) >= _SMOOTH_MIN_VALUES)


#: The nuisance names whose step counts a per-nuisance ``mstop`` mapping may set.
_STEP_NAMES = ("propensity", "response", "outcome", "employment", "log_outcome",
               "mu0", "mu1")


def _check_fixed_steps(mstop, tune):
    """The step count of each nuisance when ``mstop`` is a mapping, else None.

    A mapping is the ``nuisance_mstop_summary`` of a tuned call. It fixes every
    nuisance's steps, so it needs ``tune=False``."""
    if not isinstance(mstop, Mapping):
        return None
    if tune:
        raise ValueError(
            "mstop is a mapping of per-nuisance step counts, which fixes each nuisance's "
            "steps; pass tune=False with it. diagnostics['nuisance_mstop_summary'] of a "
            "tuned call is the mapping to pass")
    unknown = sorted(set(mstop) - set(_STEP_NAMES))
    if unknown:
        raise ValueError(
            f"mstop names unknown nuisance(s) {unknown}; the names are {list(_STEP_NAMES)}, "
            "as in diagnostics['nuisance_mstop_summary']")
    return {name: check_int_at_least(f"mstop[{name!r}]", value, 1, allow_none=False)
            for name, value in mstop.items()}


def _rhs(covariates, smooth, df, *, smooth_names=(), linear=(), interactions=()):
    """The right-hand side of a nuisance formula.

    With ``smooth`` a baseline in ``smooth_names`` enters as ``bbs``. Every
    other baseline is a linear ``bols(name)`` term, whatever ``smooth`` is. In
    gamboost a plain name means a spline, so a linear term must be spelled
    ``bols``; ``bols(name)`` gives the same fit as a plain name in
    :func:`~mboostrs.glmboost`. A factor is also ``bols(name)``, which gamboost
    codes with treatment contrasts (see :func:`_fit_nuisance`). ``linear`` names
    are added as plain terms. Each name in ``interactions`` multiplies every
    baseline term by that column (``by=``), so the baseline shape can differ
    between the two values of a binary treatment."""
    terms = []
    for name in covariates:
        if smooth and name in smooth_names:
            terms.append(f"bbs({quote_name(name)}, df={df:g})")
        else:
            terms.append(f"bols({quote_name(name)})")
    by_terms = []
    for by in interactions:
        for name in covariates:
            if smooth and name in smooth_names:
                by_terms.append(f"bbs({quote_name(name)}, df={df:g}, by={quote_name(by)})")
            else:
                by_terms.append(f"bols({quote_name(name)}, by={quote_name(by)})")
    return " + ".join([*terms, *linear, *by_terms])


#: The column of each ``bols(column ...)`` term in a formula.
_BOLS_COLUMN = re.compile(r"\bbols\(\s*([A-Za-z_][A-Za-z0-9_.]*)")


def _has_factor_term(rhs, data):
    """Whether a ``bols`` term of ``rhs`` names a categorical column of ``data``.

    glmboost refuses ``bols`` on a factor and codes a bare factor name instead;
    gamboost codes ``bols`` on a factor with treatment contrasts. A formula with
    a factor is therefore fitted by gamboost."""
    return any(name in data and np.asarray(data[name]).dtype == object
               for name in _BOLS_COLUMN.findall(rhs))


#: The start of a penalty-drift warning (``_model_api.warn_if_penalty_drifted``).
_DRIFT_TEXT = "boosting has undone much of the penalty"

#: The nuisance models that ``smooth=`` can switch on, by name.
_SMOOTH_NUISANCES = ("propensity", "outcome", "response")

#: A bare column name, as a term of a formula.
_BARE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _check_extrapolation(extrapolation):
    if extrapolation not in POLICIES:
        raise ValueError(f"extrapolation must be one of {POLICIES}, got {extrapolation!r}")
    return extrapolation


def _check_flat_tolerance(flat_tolerance):
    if flat_tolerance is None:
        return None
    value = float(flat_tolerance)
    if not np.isfinite(value) or value < 0:
        raise ValueError("flat_tolerance must be None or a finite number >= 0")
    return value


def _smooth_by_nuisance(smooth):
    """``smooth`` as one flag per nuisance model. A bool applies to all three
    models; a dict names the models it sets, and a model it does not name takes
    False."""
    if isinstance(smooth, dict):
        unknown = sorted(set(smooth) - set(_SMOOTH_NUISANCES))
        if unknown:
            raise ValueError(
                f"smooth names unknown nuisance model(s) {unknown}; the keys are "
                "'propensity', 'outcome' and 'response'")
        flags = {}
        for key in _SMOOTH_NUISANCES:
            value = smooth.get(key, False)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"smooth[{key!r}] must be True or False, got {value!r}")
            flags[key] = bool(value)
        return flags
    return {key: bool(smooth) for key in _SMOOTH_NUISANCES}


def _top_level_terms(rhs):
    """The ``+``-separated terms of a right-hand side. A ``+`` inside parentheses
    does not separate terms."""
    terms, depth, start = [], 0, 0
    for i, ch in enumerate(rhs):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "+" and depth == 0:
            terms.append(rhs[start:i].strip())
            start = i + 1
    terms.append(rhs[start:].strip())
    return [term for term in terms if term]


def _check_terms(terms, argument, treatment):
    """A user-supplied right-hand side, checked for the two mistakes that a
    formula cannot catch by itself: a response left in, and the treatment named
    as a bare term."""
    if terms is None:
        return None
    if not isinstance(terms, str) or not terms.strip():
        raise ValueError(
            f"{argument} must be a non-empty string of right-hand-side terms, for example "
            "'bbs(age) + btree(prior_earnings, maxdepth=1)'")
    if "~" in terms:
        raise ValueError(
            f"{argument} is the right-hand side only; remove the response and '~' from {terms!r}")
    bare = {term for term in _top_level_terms(terms) if _BARE_NAME.fullmatch(term)}
    if treatment in bare:
        raise ValueError(
            f"{argument} names the treatment {treatment!r} as a bare term. The treatment is "
            "the response of the propensity model, and aipw adds it to the pooled outcome "
            f"model itself. For an interaction, write by={quote_name(treatment)} inside a term instead")
    return terms.strip()


#: A strictly positive numeric baseline whose sample skewness exceeds this value
#: enters the log-outcome model of the two-part fit on the log scale.
_LOG_SKEW = 1.0

#: The name prefix of a log-scale copy of a baseline in a nuisance formula.
_LOG_PREFIX = "_log_"

#: A fitted mean outside the observed outcome range, widened by this many ranges
#: on each side, is reported by aipw as a warning.
_OUTCOME_RANGE_MULTIPLE = 1.0


def _log_baselines(values, covariates):
    """The strictly positive numeric baselines whose sample skewness exceeds
    ``_LOG_SKEW``, counted on all rows so that every fold uses the same rule.
    Earnings and distances are the typical case; a baseline with a zero or a
    negative value is never logged."""
    names = []
    for name in covariates:
        column = values[name]
        if column.dtype == object or column.min() <= 0:
            continue
        centred = column - column.mean()
        spread = np.sqrt(np.mean(centred ** 2))
        if spread > 0 and np.mean(centred ** 3) / spread ** 3 > _LOG_SKEW:
            names.append(name)
    return frozenset(names)


def _model_names(covariates, log_names):
    """The formula name of each baseline: its log-scale copy when it is logged."""
    return [_LOG_PREFIX + name if name in log_names else name for name in covariates]


def _with_log_columns(columns, log_names):
    """``columns`` with the log-scale copy of each logged baseline added."""
    out = dict(columns)
    for name in log_names:
        out[_LOG_PREFIX + name] = np.log(np.asarray(columns[name], dtype=np.float64))
    return out


def _within_training_range(newdata, train):
    """``newdata`` with each numeric column held to the range that the same
    column has in ``train``, so a nuisance is never asked to extrapolate."""
    out = dict(newdata)
    for name, column in newdata.items():
        if name not in train:
            continue
        reference = np.asarray(train[name])
        if reference.dtype == object or np.asarray(column).dtype == object:
            continue
        out[name] = np.clip(np.asarray(column, dtype=np.float64),
                            float(reference.min()), float(reference.max()))
    return out


def _nuisance_predict(model, newdata, extrapolation, train=None):
    """Response-scale predictions of a nuisance model.

    With the training columns ``train`` of the model, every numeric column of
    ``newdata`` is first held to its training range. aipw passes ``train`` only
    for the two-part model, whose mean takes an exponential of the log model's
    prediction, so an extrapolated input there is amplified. Otherwise
    ``extrapolation`` goes to a boosted model with spline terms; a linear
    glmboost has no spline range to check and predicts from its coefficients."""
    from mboostrs.gamboost import GAMBoostModel

    if train is not None:
        newdata = _within_training_range(newdata, train)
    if isinstance(model, GAMBoostModel):
        return model.predict(newdata=newdata, type="response", extrapolation=extrapolation)
    return model.predict(newdata=newdata, type="response")


class _NuisanceNotes:
    """The warnings that the nuisance fits of one aipw or effect_model call raise.

    Each nuisance model runs inside :meth:`scope`. Within the scope the
    extrapolation rows of its predictions and tuning are collected rather than
    warned, and every other warning is held. :meth:`observe_tuning` checks a
    tuned step count's cross-validated curve for flatness against
    ``flat_tolerance``. :meth:`emit` issues the held warnings once for the call,
    each naming the nuisance model and the argument of ``owner`` that controls
    it.
    """

    def __init__(self, owner, flat_tolerance):
        self.owner = owner
        self.flat_tolerance = flat_tolerance
        self._rows = {}
        self._tuned = {}
        self._flat = {}
        self._held = {}

    @contextmanager
    def scope(self, label):
        with warnings.catch_warnings(record=True) as caught, collecting() as rows:
            warnings.simplefilter("always")
            yield
        merged = self._rows.setdefault(label, CollectedRows())
        merged.extend(rows)
        merged.calls += rows.calls
        merged.clipped = merged.clipped or rows.clipped
        self._held.setdefault(label, []).extend(caught)

    def absorb(self, other):
        """Take over the rows, tuning records and held warnings of ``other``, a
        notes object of the same call whose nuisance fits were kept."""
        for label, rows in other._rows.items():
            merged = self._rows.setdefault(label, CollectedRows())
            merged.extend(rows)
            merged.calls += rows.calls
            merged.clipped = merged.clipped or rows.clipped
        for label, count in other._tuned.items():
            self._tuned[label] = self._tuned.get(label, 0) + count
        for label, items in other._flat.items():
            self._flat.setdefault(label, []).extend(items)
        for label, items in other._held.items():
            self._held.setdefault(label, []).extend(items)

    def observe_tuning(self, label, tuning):
        """Record one tuned step count, and note it when its curve is flat: the
        chosen count's risk lies within ``flat_tolerance`` of the risk at step 0,
        as :func:`~mboostrs.explore.tune_mstop` decides on its own curve."""
        self._tuned[label] = self._tuned.get(label, 0) + 1
        if self.flat_tolerance is None:
            return
        curve = np.asarray(tuning.risk, dtype=np.float64)
        step = max(1, int(tuning.mstop))
        if step >= curve.size:
            return
        intercept, chosen = float(curve[0]), float(curve[step])
        if not (np.isfinite(intercept) and np.isfinite(chosen)) or intercept == 0.0:
            return
        gap = abs(chosen - intercept) / abs(intercept)
        if gap <= self.flat_tolerance:
            self._flat.setdefault(label, []).append((step, intercept, chosen, gap))

    def emit(self):
        """Issue the held warnings: one extrapolation warning, one flat-curve
        warning and the penalty-drift warnings per nuisance model that raised
        them, then any other warning as it was raised."""
        for label, display in _NUISANCE_LABELS.items():
            model = f"{self.owner}, {display} model: "
            rows = self._rows.get(label)
            if rows:
                aggregated = summarize(rows, self.owner)
                if aggregated is not None:
                    warnings.warn(ExtrapolationWarning(model + aggregated.args[0],
                                                       rows=aggregated.rows),
                                  stacklevel=3)
            flats = self._flat.get(label)
            if flats:
                step, intercept, chosen, gap = flats[0]
                tolerance = self.flat_tolerance
                warnings.warn(
                    f"{model}the tuned step count lies on a flat cross-validated curve in "
                    f"{len(flats)} of {self._tuned[label]} fold fit(s): for example "
                    f"mstop={step} has risk {chosen:.6g}, within {tolerance:.1%} of the "
                    f"intercept-only risk {intercept:.6g} (relative gap {gap:.2%}). This "
                    f"{display} model is close to a constant. Pass "
                    f"{self.owner}(flat_tolerance=None) to turn this check off, or a "
                    "different tolerance.",
                    RuntimeWarning, stacklevel=3)
            for item in self._held.get(label, []):
                if str(item.message).startswith(_DRIFT_TEXT):
                    warnings.warn(f"{model}{item.message}", UserWarning, stacklevel=3)
                else:
                    warnings.warn(item.message, item.category, stacklevel=3)


def _fit_nuisance(response, data, rhs, family, *, smooth, mstop, max_mstop, tune,
                  inner_folds, seed, weights=None, notes=None, label=None,
                  extrapolation="warn"):
    """Fit one nuisance model and return it with a record of its step count.

    Without ``tune`` the model runs ``mstop`` steps. With ``tune`` the step
    count comes from inner cross-validation on the rows given: the grid starts
    at ``mstop`` steps and doubles while the cross-validated risk still falls,
    never past ``max_mstop``. The record holds the chosen count, the first grid
    and whether the search stopped at ``max_mstop`` with the risk still falling.

    ``smooth`` selects the boosted-spline fit (``gamboost``) over the linear one
    (``glmboost``). A formula with a categorical ``bols`` term is fitted by
    gamboost, which codes the factor; glmboost would refuse it. A user-supplied
    ``rhs`` is passed with ``smooth=True``, since its terms may need the boosted
    fit. With ``notes`` the tuned curve is checked for flatness under ``label``.
    """
    import mboostrs as mb
    from mboostrs.explore import tune_mstop

    formula = f"{quote_response(response)} ~ {rhs}"
    if not tune:
        fit_fn = mb.gamboost if smooth or _has_factor_term(rhs, data) else mb.glmboost
        model = fit_fn(formula, data=data, family=family,
                       control=mb.boost_control(mstop, .1), weights=weights)
        return model, {"mstop": int(model.mstop), "first_grid": int(mstop),
                       "at_max_mstop": False}
    folds = min(inner_folds, len(np.asarray(data[response])) // 2)
    if folds < 2:
        raise ValueError("inner nuisance tuning needs at least four training observations")
    with warnings.catch_warnings():
        # aipw reports a search that reached max_mstop, naming its own argument.
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="the cross-validated risk was still falling")
        # The flat-curve check is made by the caller's notes, which name the
        # nuisance and the argument, so tune_mstop's own check is switched off.
        tuning = tune_mstop(formula, data, family, folds=folds, start=mstop,
                            max_mstop=max_mstop, seed=seed, weights=weights,
                            extrapolation=extrapolation, flat_tolerance=None)
    if notes is not None:
        notes.observe_tuning(label, tuning)
    return tuning.model, {"mstop": int(tuning.mstop), "first_grid": int(mstop),
                          "at_max_mstop": bool(tuning.at_boundary)}


def _response_name(spec: CausalSpec):
    """A column name for the observed-outcome indicator that no declared
    column uses."""
    name = "outcome_observed"
    taken = {spec.treatment, spec.outcome, *spec.covariates}
    while name in taken:
        name = "_" + name
    return name


def _crossfit_folds(a, folds, seed):
    n = len(a)
    scheme = stratified_folds(a, B=folds, seed=seed) if isinstance(folds, int) else folds
    if not isinstance(scheme, FoldScheme):
        raise TypeError("folds must be an integer or FoldScheme")
    if scheme.weights.shape[0] != n:
        raise ValueError(f"folds has {scheme.weights.shape[0]} rows, expected {n}")
    held = scheme.judge if scheme.judge is not None else scheme.held_out
    held = np.asarray(held, dtype=bool)
    if held.shape != scheme.weights.shape or np.any(held.sum(axis=1) != 1):
        raise ValueError("causal cross-fitting requires every row held out exactly once")
    if scheme.base_weights is not None and not np.allclose(scheme.base_weights, 1):
        raise NotImplementedError("causal sample weights require a target-population variance design")
    return scheme, held


def _weighted_mean_variance(x, w):
    total = w.sum()
    mean = float(np.sum(w * x) / total)
    variance = float(np.sum(w * (x - mean) ** 2) / total)
    return mean, variance


def _effect_weights(a, e, estimand):
    """The analysis weight of each row: 1/e for a treated row and 1/(1 - e) for
    a control row (ATE); 1 for a treated row and e/(1 - e) for a control row
    (ATT); 1 - e for a treated row and e for a control row (overlap-weighted
    ATE)."""
    if estimand == "ate":
        return a / e + (1 - a) / (1 - e)
    if estimand == "overlap":
        return a * (1 - e) + (1 - a) * e
    return a + (1 - a) * e / (1 - e)


#: The targets :func:`aipw` can estimate. ``"all"`` is the ATE over every row;
#: ``"overlap"`` is the overlap-weighted ATE, which weights each row by
#: ``e (1 - e)``.
_TARGETS = ("all", "overlap")


def _check_target(target, estimand):
    if target not in _TARGETS:
        raise ValueError(
            "target must be 'all' (the ATE over every row) or 'overlap' (the "
            "overlap-weighted ATE, each row weighted by e (1 - e))")
    if target == "overlap" and estimand not in ("ate", "ato"):
        raise ValueError(
            "target='overlap' is defined for estimand='ate' only; for the ATT, pass "
            "target='all' (the default), which estimates the effect on all treated rows")
    return "overlap" if estimand == "ato" else target


def _balance_terms(values, covariates):
    """(label, numeric array) for each balance row: a numeric covariate is its
    own row, and a factor gives one 0/1 indicator row per level."""
    for name in covariates:
        if values[name].dtype == object:
            for level in sorted(set(values[name].tolist())):
                yield f"{name}={level}", (values[name] == level).astype(np.float64)
        else:
            yield name, values[name]


def _balance(values, spec, a, e, estimand, sample_weight):
    weights = sample_weight * _effect_weights(a, e, estimand)
    wt, wc = a * weights, (1 - a) * weights
    rows = []
    for name, x in _balance_terms(values, spec.covariates):
        mt, vt = _weighted_mean_variance(x, a * sample_weight)
        mc, vc = _weighted_mean_variance(x, (1 - a) * sample_weight)
        scale = np.sqrt((vt + vc) / 2)
        before = (mt - mc) / scale if scale > 0 else 0.0
        mat, vat = _weighted_mean_variance(x, wt)
        mac, vac = _weighted_mean_variance(x, wc)
        adjusted_scale = np.sqrt((vat + vac) / 2)
        after = (mat - mac) / adjusted_scale if adjusted_scale > 0 else 0.0
        rows.append({
            "covariate": name, "treated_mean": mt, "control_mean": mc,
            "smd_before": float(before), "weighted_treated_mean": mat,
            "weighted_control_mean": mac, "smd_after": float(after),
        })
    return rows


def _influence_uncertainty(influence, scheme, level, sample_weight):
    """Return an influence-function standard error for the declared design.

    IID rows use the variance of the weighted mean of the influence values:
    with frequency weights ``w`` and ``W = sum(w)``, ``sum(w (IF - mean)^2) /
    (W (W - 1))``. This is the usual ``var(IF, ddof=1) / n`` on unit weights.
    When ``groups`` are present, rows within a group may be arbitrarily
    dependent and groups are treated as independent sampling units. The
    cluster sandwich for a weighted mean is ``G/(G-1) * sum_g(sum_{i in g}
    w_i (IF_i - mean))^2 / W^2``. A Student-t critical value with ``G-1``
    degrees of freedom makes the finite number of independent units visible in
    the interval.

    Ordered folds do not by themselves specify a time-series variance model,
    so they remain an explicit refusal rather than receiving an IID interval.
    """
    from mboostrs import _rust

    influence = np.asarray(influence, dtype=np.float64)
    probability = .5 + level / 2
    if scheme.groups is not None:
        labels, inverse = np.unique(np.asarray(scheme.groups), return_inverse=True)
        n_groups = int(labels.size)
        if n_groups < 2:
            return None, None, {
                "variance": "unavailable_fewer_than_two_clusters",
                "independent_units": n_groups,
            }
        variance = _rust.influence_variance(
            influence, sample_weight, inverse.astype(np.uint64))
        se = float(np.sqrt(max(variance, 0.0)))
        # Reuse the package's dependency-free exact Student-t implementation.
        from mboostrs.cvrisk import _t_quantile

        critical = _t_quantile(probability, n_groups - 1)
        return se, critical, {
            "variance": "cluster_robust_influence",
            "independent_units": n_groups,
            "degrees_of_freedom": n_groups - 1,
            "finite_cluster_correction": n_groups / (n_groups - 1),
        }
    if scheme.ordered:
        return None, None, {
            "variance": "unavailable_for_ordered_dependence",
            "independent_units": None,
        }
    variance = _rust.influence_variance(influence, sample_weight)
    se = float(np.sqrt(max(variance, 0.0)))
    units = (int(influence.size) if np.all(sample_weight == 1)
             else float(sample_weight.sum()))
    return se, NormalDist().inv_cdf(probability), {
        "variance": "iid_influence",
        "independent_units": units,
        "degrees_of_freedom": None,
    }


#: The sparse-arm check splits the propensity into bins of equal total weight.
#: The bin count is one per _SPARSE_ROWS_PER_BIN units of weight, clamped to
#: [_SPARSE_MIN_BINS, _SPARSE_MAX_BINS], so a bin keeps enough rows for a
#: 20-row arm count to be reachable at moderate n.
_SPARSE_MIN_BINS = 4
_SPARSE_MAX_BINS = 10
_SPARSE_ROWS_PER_BIN = 200
#: A bin holding fewer rows than this in one arm is sparse for that arm.
_SPARSE_MIN_ROWS = 20
#: A sparse bin is reported only when it holds more than this share of the
#: target rows (all rows for the ATE, treated rows for the ATT).
_SPARSE_TARGET_SHARE = 0.05


def _sparse_bin_count(n):
    """Number of propensity bins for the sparse-arm check, given n: the total
    observation weight (the row count when unweighted). One bin per 200 units,
    at least 4 and at most 10."""
    return int(min(max(int(n) // _SPARSE_ROWS_PER_BIN, _SPARSE_MIN_BINS), _SPARSE_MAX_BINS))


def _weighted_bin_edges(e, w, bins):
    """Propensity values that cut the weighted sample into ``bins`` groups of
    equal total weight. The cut for share k/bins is the smallest propensity whose
    cumulative weight reaches that share. A row with integer weight w counts as
    w identical rows, so a weighted sample and its replicated rows get the same
    edges."""
    order = np.argsort(e, kind="stable")
    x = e[order]
    cumulative = np.cumsum(w[order])
    targets = cumulative[-1] * np.arange(1, bins) / bins
    cut = x[np.minimum(np.searchsorted(cumulative, targets, side="left"), x.size - 1)]
    return np.unique(np.concatenate(([x[0]], cut, [x[-1]])))


def _bin_span(x):
    """The range of a numeric covariate in one bin, or the factor levels."""
    if x.dtype == object:
        return sorted(set(x.tolist()))
    return [float(x.min()), float(x.max())]


def _sparse_arms(e, a, values, covariates, estimand, sample_weight):
    """Weighted treated and control counts per propensity bin, and the bins in
    which an arm is too thin for its outcome model. Bins hold roughly equal
    total weight; their count comes from _sparse_bin_count. A row counts with
    its observation weight, so integer weights match replicated rows.

    For the ATE both outcome models are evaluated
    at every row, so both arms are checked. For the ATT only the control model
    matters: the treated residual terms cancel in the numerator, and the
    treated rows enter through ``mu0`` alone.
    """
    n = a.size
    w = np.asarray(sample_weight, dtype=np.float64)
    edges = _weighted_bin_edges(e, w, _sparse_bin_count(w.sum()))
    if edges.size < 2:
        edges = np.array([e.min(), e.max()])
    bin_of = np.clip(np.searchsorted(edges, e, side="right") - 1, 0, edges.size - 2)
    target = (a == 1) if estimand == "att" else np.ones(n, dtype=bool)
    checked = ("control",) if estimand == "att" else ("treated", "control")
    n_target = float(w[target].sum())
    bins = []
    for b in range(edges.size - 1):
        member = bin_of == b
        counts = {"treated": float(w[member & (a == 1)].sum()),
                  "control": float(w[member & (a == 0)].sum())}
        share = float(w[member & target].sum()) / n_target
        sparse = [arm for arm in checked
                  if share > _SPARSE_TARGET_SHARE and counts[arm] < _SPARSE_MIN_ROWS]
        in_bin = e[member]
        bins.append({
            "bin": b,
            "propensity_min": float(in_bin.min()) if in_bin.size else None,
            "propensity_max": float(in_bin.max()) if in_bin.size else None,
            "n_treated": counts["treated"],
            "n_control": counts["control"],
            "share_of_target": share,
            "covariate_span": {name: _bin_span(values[name][member])
                               for name in covariates} if in_bin.size else {},
            "sparse_arms": sparse,
        })
    return {
        "target": "treated rows" if estimand == "att" else "all rows",
        "arms_checked": list(checked),
        "bins": bins,
        "flagged_bins": [row["bin"] for row in bins if row["sparse_arms"]],
        "min_rows_per_arm": _SPARSE_MIN_ROWS,
        "target_share": _SPARSE_TARGET_SHARE,
    }


#: Row count up to which ``outcome_model="auto"`` fits the pooled outcome model
#: without treatment interactions. The value is the 1,500 rows of the small
#: draws in docs/book/benchmarks/causal_discovery (``N_SMALL`` in prepare.py),
#: the largest row count the measurement covers: over 25 such draws the mean
#: absolute ATE error was 1,404 for pooled without
#: interactions and 2,487 for the previous rule. At 12,000 rows the measurement
#: covers 5 draws, too few to extend the threshold.
_AUTO_POOLED_MAX_ROWS = 1500


def _outcome_oof_loss(outcome_type, y, a, observed, mu0, mu1, weights):
    """Out-of-fold loss of the outcome predictions at each observed row's own
    treatment arm, scored by the Rust family scorer (binary outcome: weighted
    mean binomial deviance; continuous: the Gaussian family loss, the weighted mean squared error)."""
    rows = observed == 1
    mu = np.where(a == 1, mu1, mu0)[rows]
    if outcome_type == "binary":
        from mboostrs import _rust
        return float(_rust.binomial_deviance(
            np.ascontiguousarray(y[rows]), np.ascontiguousarray(mu),
            np.ascontiguousarray(weights[rows])))
    from mboostrs.explore import _score
    from mboostrs.families import Gaussian
    return float(_score(Gaussian(), y[rows], mu, weights[rows]))


def _resolve_outcome_model(requested, interactions, sparse, custom=False, n_rows=None):
    """The outcome model a fit uses, whether its treatment interactions are on,
    and the reason for that choice.

    ``"auto"`` takes the pooled model without interactions when the data have
    at most ``_AUTO_POOLED_MAX_ROWS`` rows. With more rows it takes the pooled
    model with treatment interactions when the sparse-arm check flagged a
    propensity bin, and separate models per arm otherwise.
    ``outcome_interactions=True`` with ``"auto"`` selects the pooled
    model, since interactions exist only there. With ``custom`` (a user's
    ``outcome_terms``) the pooled model has no interactions, because aipw
    cannot build them from a string the user wrote. ``"auto"`` never selects
    the two-part model: it is requested by name only.
    """
    flagged = sparse["flagged_bins"]
    if requested == "two_part":
        return "two_part", False, "outcome_model='two_part' was requested"
    if requested == "auto":
        if interactions:
            return "pooled", True, "outcome_interactions=True needs the pooled model"
        if n_rows is not None and n_rows <= _AUTO_POOLED_MAX_ROWS:
            return "pooled", False, (
                f"the data have {n_rows} rows, at most {_AUTO_POOLED_MAX_ROWS}, where the "
                "pooled model without treatment interactions had the smaller ATE error "
                "in simulation")
        if flagged and custom:
            return "pooled", False, (
                f"the sparse-arm check flagged propensity bin(s) {flagged}, and outcome_terms "
                "is a formula you wrote, so the pooled model has no treatment interactions")
        if flagged:
            return "pooled", True, f"the sparse-arm check flagged propensity bin(s) {flagged}"
        return "separate", False, "the sparse-arm check flagged no propensity bin"
    used_interactions = bool(interactions) and requested == "pooled"
    return requested, used_interactions, f"outcome_model={requested!r} was requested"


def _sparse_arm_message(report, covariates, model, interactions, requested):
    """One warning naming each flagged bin, its propensity range, its arm counts
    and the covariate span of its rows, the outcome model that was fitted, and
    the remedy."""
    details = []
    for row in report["bins"]:
        if not row["sparse_arms"]:
            continue
        spans = []
        for name in covariates:
            span = row["covariate_span"][name]
            if isinstance(span, list) and len(span) == 2 and not isinstance(span[0], str):
                spans.append(f"{name} in [{span[0]:.4g}, {span[1]:.4g}]")
            else:
                spans.append(f"{name} in {{{', '.join(map(str, span))}}}")
        arms = " and ".join(f"{row['n_' + arm]:g} {arm}" for arm in row["sparse_arms"])
        details.append(
            f"bin {row['bin']} with propensity in "
            f"[{row['propensity_min']:.4g}, {row['propensity_max']:.4g}] holds "
            f"{row['share_of_target']:.1%} of {report['target']} and only {arms} "
            f"rows (counts: {row['n_treated']} treated, {row['n_control']} control); "
            f"covariates {'; '.join(spans)}")
    if model == "pooled" and interactions:
        used = ("outcome_model='pooled' with outcome_interactions=True (one model with the "
                "treatment times each baseline term)")
        if requested == "auto":
            remedy = ("Pass outcome_model='pooled' for one model without interactions, or "
                      "outcome_model='separate' for one model per arm, to choose the model "
                      "yourself, or state the target population without that region")
        else:
            remedy = ("Pass outcome_interactions=False so the sparse arm borrows the other "
                      "arm's shape, or state the target population without that region")
    else:
        used = "outcome_model='separate' (one outcome model per arm)"
        remedy = ("Pass outcome_model='pooled' to fit one outcome model with the treatment "
                  "as a term, so the sparse arm borrows the other arm's shape, or state "
                  "the target population without that region")
    chosen = (", chosen by outcome_model='auto' because this check flagged the bins"
              if requested == "auto" else "")
    return (
        f"aipw: {len(details)} propensity bin(s) hold more than "
        f"{report['target_share']:.0%} of {report['target']} but fewer than "
        f"{report['min_rows_per_arm']} rows of an arm. The outcome model of that arm is "
        "fitted from too few rows there and extrapolates, so the estimate and its "
        "interval are unreliable for those rows. The fitted model is "
        f"{used}{chosen}. "
        + " ".join(f"({i}) {text}." for i, text in enumerate(details, 1))
        + " " + remedy + "."
    )


def _outcome_range_check(y, observed, mu0, mu1, model):
    """Count the fitted outcome means that lie beyond the observed outcome range
    by more than ``_OUTCOME_RANGE_MULTIPLE`` times its width, on either side,
    and warn once when any do. The estimate is not changed."""
    seen = y[observed == 1]
    low, high = float(seen.min()), float(seen.max())
    width = high - low
    bound_low = low - _OUTCOME_RANGE_MULTIPLE * width
    bound_high = high + _OUTCOME_RANGE_MULTIPLE * width
    rows_outside = {}
    for name, mu in (("mu0", mu0), ("mu1", mu1)):
        mu = np.asarray(mu, dtype=np.float64)
        rows_outside[name] = int(np.sum(~np.isfinite(mu) | (mu < bound_low) | (mu > bound_high)))
    record = {
        "observed_min": low, "observed_max": high,
        "multiple": _OUTCOME_RANGE_MULTIPLE,
        "lower_bound": bound_low, "upper_bound": bound_high,
        "rows_outside": rows_outside, "outcome_model": model,
    }
    total = sum(rows_outside.values())
    if total:
        warnings.warn(
            f"aipw: {total} fitted outcome mean(s) lie outside [{bound_low:.6g}, "
            f"{bound_high:.6g}], more than {_OUTCOME_RANGE_MULTIPLE:g} times the width of the "
            f"observed outcome range [{low:.6g}, {high:.6g}] beyond either end (mu0: "
            f"{rows_outside['mu0']} rows, mu1: {rows_outside['mu1']} rows). The "
            f"{model} outcome model extrapolates for those rows, and the estimate uses "
            "these means, so an error in them enters the estimate. Inspect those rows in "
            "diagnostics['overlap'], or refit with outcome_model='separate' or 'pooled' "
            "and compare the estimates.",
            RuntimeWarning, stacklevel=3)
    return record


@once_per_call
def aipw(data, spec: CausalSpec, *, folds=5, repeats=1, seed=42, smooth=False, df=4,
         mstop=200, max_mstop=None, tune=True, inner_folds=5, level=.95,
         propensity_clip=FROM_SPEC, outcome_missing=FROM_SPEC, weights=None,
         outcome_model="cv", outcome_interactions=False,
         overlap_bounds=FROM_SPEC, target="all", propensity_terms=None,
         outcome_terms=None, extrapolation="warn", flat_tolerance=0.01,
         repeat_aggregation="average", propensity_learner="smooth",
         interval="influence", bootstrap_draws=200):
    """Cross-fitted augmented inverse-probability estimate for binary treatment.

    The supported outcome types are a continuous mean difference and a binary
    risk difference, both with numeric baseline covariates and ATE/ATT.
    Nuisance models are fitted outside each observation's fold; with
    ``tune=True`` their stopping iterations are chosen by an inner
    cross-validation using outer-training rows only. Ordinary folds use IID
    influence-function uncertainty. A grouped fold scheme uses the retained
    group labels for cluster-robust uncertainty. Ordered dependence needs an
    explicit time-series variance model and therefore has no interval.

    ``interval`` chooses how the standard error and interval are computed.
    ``"influence"`` (the default) uses the influence function of the final
    estimate, treating the cross-fitted nuisance predictions as fixed. That
    variance leaves out the error of the nuisance fits where the propensity is
    near 0 or 1 and the data-dependent choice of the outcome model, and on a
    thin-overlap design its interval can cover less than ``level``.
    ``"bootstrap"`` resamples the whole procedure. Each of ``bootstrap_draws``
    draws (default 200) resamples rows with replacement, or whole clusters when
    the fold scheme is grouped, then reruns the cross-fitting on the resample,
    including the ``outcome_model="cv"`` choice and the tuning of every
    nuisance, and records the ATE. The reported interval is the percentile
    interval of the draws at ``level`` and ``standard_error`` is their standard
    deviation. Rows are drawn by the core's ``group_bootstrap_counts`` stream
    seeded by ``seed``, so a seed fixes the draws. In a draw, the copies of one
    original row (or cluster) are kept in one cross-fitting fold, so no copy is
    predicted by a model trained on another copy; the folds of a draw are
    therefore grouped folds with the number of folds of the call. A draw that
    fails (for example a fold without a treated row) is dropped: the number of
    failed draws is in ``diagnostics["bootstrap"]["n_failed"]`` and in a
    warning, and the interval uses the draws that finished. The influence
    standard error and interval remain in ``diagnostics["bootstrap"]``. The
    bootstrap needs ``repeats=1`` and an unordered fold scheme. The cost is
    ``bootstrap_draws`` times the cost of one call;
    :func:`estimate_cost` counts it.
    The draws run one after another. Each draw's nuisance fits use the core's
    thread pool, and the warning bookkeeping of the fits is not safe to share
    between Python threads, so there is no worker-thread option.

    ``level`` is the coverage of the reported interval. The default is 0.95, a
    95% interval; pass ``level=0.9`` for a 90% interval.

    ``repeats>1`` draws independent stratified partitions and averages each
    observation's out-of-fold nuisance predictions before constructing its
    score. This reduces dependence on one arbitrary partition. The reported
    influence interval is conditional on that repeated-fitting procedure; it
    does not add a separate variance component for split randomness.

    ``repeat_aggregation`` says how ``repeats>1`` combines the partitions. The
    default ``"average"`` is the averaged-nuisance estimator described above:
    ``estimate``, ``scores`` and ``influence`` all come from the averaged
    nuisances, so the mean of ``scores`` equals ``estimate``. ``"median"`` is
    the median rule of Chernozhukov et al. (2018). Each repeat computes its own
    estimate ``theta_r`` and influence-function standard error ``se_r`` from its
    own nuisances. The reported estimate is the median of the ``theta_r``, and
    the variance is the median of ``se_r^2 + (theta_r - estimate)^2``. The
    second term carries the spread of the partitions into the interval, so the
    interval includes split-to-split variation. ``estimate`` and the interval
    then differ from ``mean(scores)``, and ``scores`` and ``influence`` remain
    those of the averaged nuisances. ``diagnostics["repeat_estimates"]`` and
    ``diagnostics["repeat_standard_errors"]`` list each repeat's estimate and
    standard error whenever ``repeats>1``. The median rule needs an integer
    ``folds``, as ``repeats>1`` does.

    ``smooth=False`` (the default) fits each nuisance with linear terms only.
    ``smooth=True`` fits a penalized spline (``bbs``) for every numeric baseline
    that has at least ten distinct values, and a linear ``bols`` term for the
    others: a two-valued indicator, a low-cardinality count, or a factor. This
    is the rule :func:`effect_model` uses. On 10 simulated draws at n=12,000 with
    90% intervals, the median and largest absolute ATE error, the coverage and
    the median seconds were: ``smooth=False`` with ``outcome_model="cv"`` 434 and
    2,322, 0.9, 2.7 s; ``smooth=True`` with ``"cv"`` 550 and 1,706, 0.9, 8.0 s;
    ``smooth=False`` with ``"auto"`` 790 and 2,322, 0.6, 1.5 s; ``smooth=True``
    with ``"auto"`` 1,137 and 8,205, 0.6, 5.9 s. Baselines
    may be numeric, boolean, string or categorical; a factor is coded with
    treatment contrasts, as :func:`~mboostrs.gamboost` codes a string column,
    and ``diagnostics["balance"]`` reports one row per factor level.
    ``smooth`` also takes a dict with keys ``"propensity"``, ``"outcome"`` and
    ``"response"``, for example ``smooth={"propensity": False, "outcome": True}``.
    Each named nuisance gets its own flag; a nuisance the dict does not name
    takes ``False``.

    ``propensity_terms`` and ``outcome_terms`` replace the formula that
    ``smooth`` builds for the propensity model and for the outcome model. Each is
    a right-hand side as a string, for example
    ``"bbs(age) + btree(prior_earnings, maxdepth=1)"``; the response and ``~``
    are not included. A step in the propensity at an eligibility limit is
    fitted by a tree term, which the default formula does not contain. A
    custom right-hand side is fitted by ``gamboost``, whatever ``smooth`` says.
    The response model keeps the formula that ``smooth`` builds. With
    ``outcome_model="pooled"`` the treatment is added to ``outcome_terms`` as a
    linear term, and ``outcome_interactions=True`` is refused with
    ``outcome_terms``. An interaction is written inside the string with ``by=``
    set to the treatment column: ``"bbs(age, df=4, by=treated)"`` when
    ``treated`` is the treatment column. A custom term that names the treatment
    as a bare term is refused. The formulas used are in ``nuisance["terms"]``.

    ``propensity_learner`` chooses the propensity formula when
    ``propensity_terms`` is not given. ``"smooth"`` is the formula that the
    ``smooth`` argument builds: ``bbs`` terms when it is on, and
    linear terms when it is off. ``"smooth+trees"`` adds
    ``btree(name, maxdepth=1)`` for each numeric baseline with more than two
    distinct values, and is fitted by ``gamboost``. A depth-one tree makes one
    split, so it can place a step such as an eligibility cutoff, which a
    spline spreads over several basis functions. ``"smooth"`` (the default)
    keeps the formula that ``smooth`` builds.
    ``"auto"`` fits both formulas on the same folds and keeps the one with the
    lower out-of-fold binomial deviance; it changes the propensities, and so the
    estimate, when the tree formula wins. The two deviances and the choice are in
    ``diagnostics["propensity_learner"]``, and :meth:`AIPWResult.describe`
    states them. With ``propensity_terms`` the learner is not used, and an
    explicit value other than ``"smooth"`` is refused.

    ``mstop`` is the first grid of each nuisance's step count. With ``tune=True``
    the grid doubles while the inner cross-validated risk still falls, up to
    ``max_mstop`` (default 1600, and never below ``mstop``). A nuisance whose
    risk is still falling at ``max_mstop`` triggers a warning that names this
    argument. ``diagnostics["nuisance_mstop"]`` records, for each nuisance, the
    step counts of its folds and whether any fold reached the cap.
    ``diagnostics["nuisance_mstop_summary"]`` gives one step count per nuisance,
    the median of its fold counts. With ``outcome_model="cv"`` it also holds the
    nuisances of the candidates that were not chosen (a nuisance shared by two
    candidates takes the chosen candidate's count), so it can be passed back to
    another ``"cv"`` call. Passing that mapping as ``mstop`` with
    ``tune=False`` refits every nuisance at its own fixed step count, without a
    second tuned pass. With ``tune=False`` every nuisance runs exactly ``mstop``
    steps.

    Warnings from the nuisance fits are issued once per call, each naming its
    nuisance model (propensity, response, pooled outcome, control-arm outcome or
    treated-arm outcome) and the argument that controls it. ``extrapolation``
    is the policy for nuisance predictions outside a spline's training range,
    as in :meth:`~mboostrs.gamboost.GAMBoostModel.predict`: ``"warn"`` (the
    default) predicts and warns, ``"clip"`` holds the prediction flat at the
    range edge, ``"allow"`` is silent and ``"error"`` refuses. The two-part
    outcome model is the exception: its covariates are always held to the
    training range, because its mean takes an exponential of the log model's
    prediction. ``flat_tolerance``
    is the relative gap at which a tuned nuisance's cross-validated curve counts
    as flat, as in :func:`~mboostrs.explore.tune_mstop`: a flat curve means the
    nuisance is close to a constant. ``None`` turns that check off.

    ``outcome_model`` says how the outcome models are fitted. ``"cv"`` (the
    default) is described below. ``"auto"`` fits ``"pooled"`` without treatment interactions when the data
    have at most 1,500 rows. That threshold is the 1,500 rows of the small
    simulated draws in ``docs/book/benchmarks/causal_discovery``, the largest
    row count the measurement covers: over 25 such draws the mean absolute ATE
    error was 1,404 for the pooled model without interactions and 2,487 for the
    rule that applies above the threshold. With
    more rows ``"auto"`` runs the sparse-arm check described below on the
    cross-fitted propensity. If the check flags a bin, the fit uses ``"pooled"``
    with ``outcome_interactions=True`` (with ``outcome_terms``, ``"pooled"``
    without interactions); if it flags no bin, the fit uses ``"separate"``.
    ``outcome_interactions=True`` with ``"auto"`` always gives the pooled model
    with interactions. ``"auto"``
    never chooses ``"two_part"``, which must be requested by name.
    ``outcome_model="cv"`` fits every candidate outcome model on the
    cross-fitting folds the call already uses and keeps the one with the lowest
    out-of-fold loss on the observed outcomes: the weighted mean binomial
    deviance for a binary outcome, the weighted mean squared error (the Gaussian
    family loss) for a continuous one, each computed from the
    cross-fitted ``mu0`` and ``mu1`` at the arm a row received. The candidates
    are ``"pooled"``, ``"separate"``, ``"two_part"`` when the outcome is
    continuous, zero or positive with at least one zero, and ``"pooled"`` with
    interactions (labelled ``"pooled_interactions"``) when the ``"auto"`` rule
    would pick it. ``diagnostics["outcome_model_cv"]`` lists the candidates, the
    loss of each and the chosen label. The outcome models are fitted once per
    candidate, so the outcome step costs that many times as much. The chosen
    model and the reason are in
    ``diagnostics["outcome_model_choice"]``, and :meth:`AIPWResult.describe`
    states them. ``"separate"`` fits one outcome model per arm, each on that
    arm's rows, so the treated model is fitted only where treated rows are.
    ``"pooled"`` fits one outcome model on all observed rows with the treatment
    as a linear term. The treated arm then borrows the control arm's baseline
    shape and adds the estimated shift. With ``outcome_interactions=True`` every
    baseline term is also multiplied by the treatment, which lets the shape
    differ by arm again; that option needs the rows of the thinner arm to fit
    its own shape. ``outcome_interactions=True`` with ``"separate"`` is refused.
    With ``"auto"`` and ``outcome_interactions=True`` the fit uses the pooled
    model with interactions. A covariate set refitted by
    :func:`confounding_benchmark` uses the model chosen for the full set.

    ``outcome_model="two_part"`` is for an outcome that is zero or positive with
    some zeros, such as earnings with unemployed people at zero, and a continuous
    outcome. The employment model is a Bernoulli model for P(Y > 0 | x, a) on every
    observed row. The log model is a Gaussian model for log(Y) on the positive rows.
    Both have the treatment as a linear term, as the pooled model does. The mean is
    ``P(Y > 0 | x, a) * exp(m(x, a) + s2 / 2)``, where ``m`` is the log model's mean and
    ``s2`` the residual variance of log Y on the training rows. A treatment effect
    that scales with the baseline level of a positive outcome is a constant shift on
    the log scale, so the log model carries it to baseline levels that have few treated
    rows. ``outcome_interactions=True`` and ``outcome_model="separate"`` are refused
    with it. The log model enters each strictly positive baseline whose sample
    skewness exceeds 1 on the log scale (for example prior earnings in dollars
    enters as its logarithm), because a linear term in dollars gives a log mean
    that grows with the raw value and the exponential then amplifies it. The
    employment model keeps the raw baselines. Both models hold every numeric
    covariate to its training range at prediction time, whatever ``extrapolation``
    says.
    ``nuisance["terms"]["outcome"]`` gives the formula of both parts, and a custom
    ``outcome_terms`` is used as written, with no log-scale copy. The model has
    no treatment interactions. ``"auto"`` does not select this model; on a
    flagged sparse arm it uses the pooled model with interactions instead.

    ``diagnostics["outcome_range"]`` checks the fitted means ``mu0`` and ``mu1``.
    A mean lying more than one observed outcome range beyond the range of the
    observed outcome on either side is counted there and warned about, naming the
    remedy. The check reports a model that extrapolates; it does not change the
    estimate.

    ``diagnostics["sparse_arms"]`` counts treated and control rows in each
    propensity bin of the cross-fitted propensity, with the propensity range and
    the covariate span of the rows in the bin. The bins hold roughly equal total
    weight: one bin per 200 units of weight (the row count when unweighted),
    between 4 and 10 bins. Arm counts and shares are weighted sums, so a row with
    weight 3 counts as three rows. A bin is flagged for an arm when it holds more
    than 5% of the target weight (all rows for the ATE, treated rows for the ATT)
    and fewer than 20 weighted rows of that arm. A flagged bin warns
    whenever the fit uses separate models or pooled interactions, because those
    fit an outcome curve for the thin arm from its own rows. The warning names
    the model that was fitted. A pooled model without interactions records the
    flag without warning.

    ``outcome_missing`` says what a missing outcome (NaN) means. When the call
    does not pass it, the ``outcome_missing`` of the :class:`CausalSpec` is used.
    ``"error"`` (the default of the spec) requires an observed outcome in every
    row. ``"ipw"`` treats
    the outcome as missing at random given the baseline covariates and the
    treatment: a response model for P(R = 1 | x, a) is cross-fitted like the
    propensity, outcome means are fitted on observed rows, and the augmented
    score divides the observed-row terms by the response propensity pi. The
    estimand keeps its target population: the ATE is over all rows, and the
    ATT is over all treated rows, observed or not. The ATT numerator of row i
    is ``A (mu1 - mu0) + R A (Y - mu1)/pi - R (1 - A) e/(1 - e) (Y - mu0)/pi``,
    divided by the treated share. Its influence-function variance treats the
    cross-fitted nuisances as fixed, as the ATE variance does. The result
    reports the range of pi and its effective sample size.

    ``weights`` are observation weights read as frequencies: a row with weight
    2 counts as two identical rows. They enter the nuisance fits and the
    average of the scores, and the uncertainty is the one for the weighted
    sample. The weights do not describe a survey design with clustering or
    stratification; a grouped ``FoldScheme`` supplies cluster-robust variance
    for dependence among rows. The learners centre their design by the
    unweighted mean, so a weighted nuisance fit follows a different boosting
    path from the fit on replicated rows. The two agree once the boosting
    converges: on a 400-row example the estimates differed by 1e-3 at 100
    steps and by 1e-8 at 1000 steps.

    ``propensity_clip`` bounds the propensity and, under ``outcome_missing="ipw"``,
    the response propensity. When the call does not pass it, the value is the
    ``propensity_clip`` of the :class:`CausalSpec`, which defaults to
    ``(0.01, 0.99)``, the same default as :func:`effect_model`, so the two
    estimators weight rows alike. An argument that is passed always wins. Pass
    ``None`` to disable clipping. Clipping the propensity changes the mean of the score in the rows
    it moves: a treated row with propensity 0.005 gets weight 100 instead of
    200, so the augmentation no longer removes the outcome-model error in those
    rows. Where the propensity is below the bound for many rows, use
    ``target="overlap"``, which does not divide by the propensity.

    ``target`` says which population the estimate describes. ``"all"`` (the
    default) is the ATE over every row, as described above. ``"overlap"`` is the
    overlap-weighted ATE: the effect averaged over all rows with each row
    weighted by ``e (1 - e)``, where ``e`` is the cross-fitted propensity. Its
    weights are at most 1/4 and vanish where the propensity is near 0 or 1, so
    the rows that have almost no treated (or almost no control) counterparts
    contribute little instead of dominating the estimate. It describes a
    population whose weights depend on the propensity, not the ATE over all
    rows: it is the effect for the covariate region where both arms are likely,
    and it equals the ATE when the propensity is 1/2 everywhere. Only
    ``estimand="ate"`` supports it. The propensity clip does not apply to it,
    since no propensity divides a residual. The result, its ``describe()`` and
    ``diagnostics["target"]`` state the target used. The default ``"all"`` is
    unchanged, so existing estimates do not move.

    ``diagnostics["overlap"]`` is a plain dict, the :func:`overlap_report` of the
    result after ``OverlapReport.to_dict()``. Its keys are:

    - ``bounds``: the ``(lower, upper)`` propensity bounds used.
    - ``n``: the number of rows; ``n_below`` and ``n_above``: rows whose
      propensity is below ``lower`` or above ``upper``; ``n_outside`` is their sum
      and ``share_outside`` is ``n_outside / n``.
    - ``regions``: one entry per covariate, giving where most rows lie outside
      the bounds (numeric covariates) or the share outside for each level
      (categorical covariates).
    - ``estimand``, ``estimate``, ``standard_error``: the estimand and the
      full-population estimate of this result.
    - ``n_inside``, ``estimate_inside``, ``standard_error_inside``: the row count
      and the same estimand restricted to rows whose propensity lies inside the
      bounds.
    """
    if not isinstance(spec, CausalSpec):
        raise TypeError("spec must be a CausalSpec")
    overlap_bounds = _check_bounds(_spec_default(spec, "overlap_bounds", overlap_bounds))
    if interval not in ("influence", "bootstrap"):
        raise ValueError(
            f"interval must be 'influence' or 'bootstrap'; got {interval!r}. 'influence' "
            "is the influence-function interval, 'bootstrap' resamples the whole procedure")
    bootstrap_draws = check_int_at_least("bootstrap_draws", bootstrap_draws, 2,
                                         allow_none=False)
    redo = dict(mstop=mstop, max_mstop=max_mstop, tune=tune, inner_folds=inner_folds,
                smooth=smooth, df=df, level=level, outcome_model=outcome_model,
                outcome_interactions=outcome_interactions, target=target,
                propensity_terms=propensity_terms, outcome_terms=outcome_terms,
                extrapolation=extrapolation, flat_tolerance=flat_tolerance,
                propensity_learner=propensity_learner, repeat_aggregation=repeat_aggregation)
    mstop_fixed = _check_fixed_steps(mstop, tune)
    if mstop_fixed is None:
        mstop = check_int_at_least("mstop", mstop, 1, allow_none=False)
    else:
        mstop = max(mstop_fixed.values())

    def steps_for(name):
        """The first step count of one nuisance: ``mstop``, or its entry in a mapping."""
        if mstop_fixed is None:
            return mstop
        if name not in mstop_fixed:
            raise ValueError(
                f"mstop has no step count for the {name!r} nuisance, which this call fits. "
                "Pass diagnostics['nuisance_mstop_summary'] from a call with the same "
                "outcome_model, outcome_missing, smooth and propensity_learner settings")
        return mstop_fixed[name]

    if max_mstop is None:
        max_mstop = max(_MAX_MSTOP_DEFAULT, mstop)
    else:
        max_mstop = check_int_at_least("max_mstop", max_mstop, 1, allow_none=False)
        if max_mstop < mstop:
            raise ValueError(
                f"max_mstop={max_mstop} is below mstop={mstop}; the step search starts at "
                "mstop and cannot end below it. Pass max_mstop >= mstop")
    inner_folds = check_int_at_least("inner_folds", inner_folds, 2, allow_none=False)
    repeats = check_int_at_least("repeats", repeats, 1, allow_none=False)
    if repeat_aggregation not in ("average", "median"):
        raise ValueError(
            f"repeat_aggregation must be 'average' or 'median'; got {repeat_aggregation!r}. "
            "Pass 'median' for the Chernozhukov et al. median rule over repeats")
    df = check_positive("df", df)
    level = check_unit_interval("level", level)
    propensity_clip = _check_clip(_spec_default(spec, "propensity_clip", propensity_clip))
    target_used = _check_target(target, spec.estimand)
    # The overlap target has bounded weights, so the propensity is not clipped
    # for it; the response clip still applies to the response propensity.
    score_estimand = "overlap" if target_used == "overlap" else spec.estimand
    score_clip = None if target_used == "overlap" else propensity_clip
    outcome_missing = _check_outcome_missing(
        _spec_default(spec, "outcome_missing", outcome_missing))
    requested_model = _check_outcome_model(outcome_model)
    if outcome_interactions and requested_model in ("separate", "two_part"):
        raise ValueError(
            "outcome_interactions=True applies to outcome_model='pooled' or 'auto'; the "
            "separate models already have their own baseline shape in each arm, and the "
            "two-part model has no treatment interactions")
    propensity_custom = _check_terms(propensity_terms, "propensity_terms", spec.treatment)
    outcome_custom = _check_terms(outcome_terms, "outcome_terms", spec.treatment)
    _check_propensity_learner(propensity_learner)
    if propensity_custom is not None and propensity_learner != "smooth":
        raise ValueError(
            f"propensity_learner={propensity_learner!r} chooses among propensity formulas, "
            "and propensity_terms replaces them. Pass one or the other; leave "
            "propensity_learner at 'smooth' when propensity_terms is given")
    if outcome_custom is not None and outcome_interactions:
        raise ValueError(
            "outcome_interactions=True builds its by= terms from the covariates, which "
            "outcome_terms replaces. Write the interaction inside outcome_terms with by= "
            "set to the treatment column, and leave outcome_interactions=False")
    _check_extrapolation(extrapolation)
    flat_tolerance = _check_flat_tolerance(flat_tolerance)
    smooth_flags = _smooth_by_nuisance(smooth)
    notes = _NuisanceNotes("aipw", flat_tolerance)
    values = _columns(data, spec, outcome_missing=outcome_missing, categorical=True)
    smooth_names = _smooth_baselines(values, spec.covariates)
    # Each nuisance's right-hand side: the user's string when one is given,
    # otherwise the formula that its smooth flag selects. A custom string is
    # fitted by gamboost, since its terms may need the boosted fit.
    rhs_base = propensity_custom or _rhs(
        spec.covariates, smooth_flags["propensity"], df,
        smooth_names=smooth_names)
    rhs_outcome_base = outcome_custom or _rhs(
        spec.covariates, smooth_flags["outcome"], df,
        smooth_names=smooth_names)
    rhs_response = _rhs(spec.covariates, smooth_flags["response"], df,
                        smooth_names=smooth_names, linear=(spec.treatment,))
    control_label, treated_label = spec.treatment_contrast
    raw_a = np.asarray(data[spec.treatment])
    allowed = np.isin(raw_a, (control_label, treated_label))
    if not np.all(allowed):
        raise ValueError("treatment contains values outside treatment_contrast")
    a = (raw_a == treated_label).astype(float)
    n = len(a)
    y = values[spec.outcome]
    observed = (~np.isnan(y)).astype(float)
    if spec.outcome_type == "binary" and not np.all(np.isin(y[observed == 1], (0, 1))):
        raise ValueError("a binary causal outcome must be coded with values 0 and 1")
    if a.sum() == 0 or a.sum() == len(a):
        raise ValueError("both treatment levels must be observed")
    observed_y = y[observed == 1]
    zero_inflated = bool(spec.outcome_type == "continuous" and np.all(observed_y >= 0)
                         and np.any(observed_y == 0))
    if requested_model == "two_part" and not zero_inflated:
        reason = ("a binary outcome" if spec.outcome_type == "binary"
                  else "negative values" if np.any(observed_y < 0)
                  else "no zero values among the observed rows")
        raise ValueError(
            "outcome_model='two_part' needs a continuous outcome that is zero or positive, "
            f"with at least one zero among the observed rows; this outcome has {reason}. "
            "Pass outcome_model='pooled' or 'separate' for this outcome")
    sample_weight = _sample_weights(weights, n)
    use_response = outcome_missing == "ipw" and bool(np.any(observed == 0))
    response_name = _response_name(spec)
    folds_were_integer = isinstance(folds, (int, np.integer))
    if repeats > 1 and not folds_were_integer:
        raise ValueError(
            "repeats>1 requires an integer folds count so each partition can be drawn; "
            "pass repeats=1 with an explicit FoldScheme"
        )
    schemes = [
        _crossfit_folds(a, folds, seed + 104729 * repeat)
        for repeat in range(repeats)
    ]
    scheme = schemes[0][0]
    if interval == "bootstrap" and (repeats > 1 or scheme.ordered):
        raise ValueError(
            "interval='bootstrap' needs repeats=1 and an unordered fold scheme: a draw "
            "reruns one cross-fitting partition on the resample. Pass repeats=1, or use "
            "interval='influence'")
    e_repeats = np.empty((repeats, n))
    pi_repeats = np.ones((repeats, n))
    mu0_repeats = np.empty((repeats, n))
    mu1_repeats = np.empty((repeats, n))
    fold_ids = np.empty((n, repeats), dtype=int)
    # The step counts of the nuisances fitted for each repeat and fold.
    arm_stops_all = [[{} for _ in range(repeat_scheme.n_folds)]
                     for repeat_scheme, _ in schemes]
    # The step-count record of every fitted nuisance, across repeats and folds.
    records = {"propensity": [], "response": [], "outcome": [], "employment": [],
               "log_outcome": [], "mu0": [], "mu1": []}
    from mboostrs import _rust
    from mboostrs.families import Binomial, Gaussian

    outcome_family = Binomial() if spec.outcome_type == "binary" else Gaussian()

    # Pass 1 fits each propensity candidate on every fold. The candidates share
    # the folds and seeds, so the choice between them compares out-of-fold
    # predictions of the same rows. The outcome models wait for the sparse-arm
    # check, which reads the cross-fitted propensity.
    if propensity_custom is not None:
        prop_candidates = {"custom": propensity_custom}
    else:
        prop_candidates = _propensity_candidates(
            propensity_learner, rhs_base, values, spec.covariates)
    prop_runs = {}
    for label, rhs in prop_candidates.items():
        # The base formula follows the smooth flag; the tree formula and a
        # user's formula are fitted by gamboost, whose terms may need it.
        smooth_fit = smooth_flags["propensity"] if label == "smooth" else True
        candidate_notes = _NuisanceNotes("aipw", flat_tolerance)
        e_candidate = np.empty((repeats, n))
        candidate_stops = [[None] * repeat_scheme.n_folds for repeat_scheme, _ in schemes]
        candidate_records = []
        for repeat, (repeat_scheme, held) in enumerate(schemes):
            repeat_seed = seed + 104729 * repeat
            for b in range(repeat_scheme.n_folds):
                test = held[:, b]
                train = np.asarray(repeat_scheme.weights[:, b]) > 0
                if not np.any(test):
                    raise ValueError(f"repeat {repeat}, fold {b} has no evaluation rows")
                # Fit the propensity to the declared contrast recoded as 0/1.
                # Passing original labels made non-0/1 contrasts unusable.
                train_data = {
                    spec.treatment: a[train],
                    **{name: values[name][train] for name in spec.covariates},
                }
                test_data = {name: values[name][test] for name in spec.covariates}
                with candidate_notes.scope("propensity"):
                    propensity, propensity_info = _fit_nuisance(
                        spec.treatment, train_data, rhs, Binomial(),
                        smooth=smooth_fit,
                        mstop=steps_for("propensity"), max_mstop=max_mstop, tune=tune,
                        inner_folds=inner_folds,
                        seed=repeat_seed + 3 * b, weights=sample_weight[train],
                        notes=candidate_notes, label="propensity",
                        extrapolation=extrapolation,
                    )
                    e_candidate[repeat, test] = _nuisance_predict(
                        propensity, test_data, extrapolation)
                candidate_stops[repeat][b] = propensity_info["mstop"]
                candidate_records.append(propensity_info)
                fold_ids[test, repeat] = b
        prop_runs[label] = {"rhs": rhs, "e": e_candidate, "stops": candidate_stops,
                            "records": candidate_records, "notes": candidate_notes}
    prop_deviance = {
        label: float(_rust.binomial_deviance(
            a, run["e"].mean(axis=0), sample_weight))
        for label, run in prop_runs.items()
    }
    chosen_prop = min(prop_deviance, key=prop_deviance.get)
    chosen_run = prop_runs[chosen_prop]
    notes.absorb(chosen_run["notes"])
    rhs_base = chosen_run["rhs"]
    e_repeats = chosen_run["e"]
    for repeat, stops in enumerate(chosen_run["stops"]):
        for b, step in enumerate(stops):
            arm_stops_all[repeat][b]["propensity"] = step
    records["propensity"] = chosen_run["records"]

    # Pass 1b fits the response model, when outcomes are missing, on the same
    # folds. The response model sees the treatment as a linear term, so a
    # binary column is not handed to a smooth term.
    if use_response:
        for repeat, (repeat_scheme, held) in enumerate(schemes):
            repeat_seed = seed + 104729 * repeat
            for b in range(repeat_scheme.n_folds):
                test = held[:, b]
                train = np.asarray(repeat_scheme.weights[:, b]) > 0
                test_data = {name: values[name][test] for name in spec.covariates}
                response_data = {
                    response_name: observed[train],
                    spec.treatment: a[train],
                    **{name: values[name][train] for name in spec.covariates},
                }
                with notes.scope("response"):
                    response, response_info = _fit_nuisance(
                        response_name, response_data, rhs_response, Binomial(),
                        smooth=smooth_flags["response"], mstop=steps_for("response"),
                        max_mstop=max_mstop, tune=tune, inner_folds=inner_folds,
                        seed=repeat_seed + 3 * b + 3, weights=sample_weight[train],
                        notes=notes, label="response", extrapolation=extrapolation,
                    )
                    pi_repeats[repeat, test] = _nuisance_predict(
                        response, {**test_data, spec.treatment: a[test]}, extrapolation)
                arm_stops_all[repeat][b]["response"] = response_info["mstop"]
                records["response"].append(response_info)

    e = e_repeats.mean(axis=0)
    sparse = _sparse_arms(e, a, values, spec.covariates, spec.estimand, sample_weight)

    def fit_outcome_models(model, interactions):
        """Fit the outcome model ``model`` on every cross-fitting fold and
        repeat, and return its out-of-fold ``mu0`` and ``mu1`` (repeats by rows),
        the step-count records, the step counts by repeat and fold, the
        right-hand sides it used and its notes. It changes none of the
        call's shared state, so a candidate that is not chosen leaves no trace."""
        mu0_run = np.empty((repeats, n))
        mu1_run = np.empty((repeats, n))
        run_records = {key: [] for key in records}
        run_notes = _NuisanceNotes("aipw", flat_tolerance)
        stops_all = [[{} for _ in range(repeat_scheme.n_folds)]
                     for repeat_scheme, _ in schemes]
        # The right-hand side of the pooled outcome model, with or without the
        # treatment interactions chosen above. A user's outcome_terms carries no
        # interactions that aipw could add.
        if outcome_custom is not None:
            rhs_pooled = f"{outcome_custom} + {spec.treatment}"
        else:
            rhs_pooled = _rhs(spec.covariates, smooth_flags["outcome"], df,
                              smooth_names=smooth_names, linear=(spec.treatment,),
                              interactions=(spec.treatment,) if interactions else ())
        # The two-part log model's formula: the same baselines, with each skewed
        # positive one entered as its log-scale copy. A user's outcome_terms is used
        # as written, so no copy is made for it.
        log_names = (_log_baselines(values, spec.covariates)
                     if model == "two_part" and outcome_custom is None else frozenset())
        log_smooth_names = frozenset(_model_names(smooth_names, log_names)) if log_names else smooth_names
        if outcome_custom is not None:
            rhs_log = rhs_pooled
        else:
            rhs_log = _rhs(_model_names(spec.covariates, log_names), smooth_flags["outcome"], df,
                           smooth_names=log_smooth_names, linear=(spec.treatment,))


        # Pass 2 fits the outcome models on the same folds and seeds as pass 1.
        for repeat, (repeat_scheme, held) in enumerate(schemes):
            repeat_seed = seed + 104729 * repeat
            for b in range(repeat_scheme.n_folds):
                test = held[:, b]
                train = np.asarray(repeat_scheme.weights[:, b]) > 0
                test_data = {name: values[name][test] for name in spec.covariates}
                arm_stops = stops_all[repeat][b]
                if model == "pooled":
                    # One outcome model on every observed row; the treatment column
                    # shifts the baseline curve, and its prediction at each arm's
                    # value gives mu0 and mu1 for the test rows.
                    pool = train & (observed == 1)
                    for arm in (0, 1):
                        if not np.any(pool & (a == arm)):
                            raise ValueError(
                                f"repeat {repeat}, fold {b} has no training rows with an "
                                f"observed outcome in treatment arm {arm}")
                    pooled_data = {
                        spec.outcome: y[pool],
                        spec.treatment: a[pool],
                        **{name: values[name][pool] for name in spec.covariates},
                    }
                    with run_notes.scope("outcome"):
                        outcome, outcome_info = _fit_nuisance(
                            spec.outcome, pooled_data, rhs_pooled, outcome_family,
                            smooth=smooth_flags["outcome"] or outcome_custom is not None,
                            mstop=steps_for("outcome"), max_mstop=max_mstop, tune=tune,
                            inner_folds=inner_folds, seed=repeat_seed + 3 * b + 1,
                            weights=sample_weight[pool], notes=run_notes, label="outcome",
                            extrapolation=extrapolation,
                        )
                        for arm, target in ((0, mu0_run), (1, mu1_run)):
                            arm_newdata = {
                                **test_data,
                                spec.treatment: np.full(int(test.sum()), float(arm)),
                            }
                            target[repeat, test] = _nuisance_predict(
                                outcome, arm_newdata, extrapolation)
                    arm_stops["outcome"] = outcome_info["mstop"]
                    run_records["outcome"].append(outcome_info)
                elif model == "two_part":
                    # Two models on the same rows: a Bernoulli model for P(Y > 0 | x, a)
                    # on every observed row, and a Gaussian model for log Y on the
                    # positive rows. Both have the treatment as a linear term. The
                    # mean of Y is P(Y > 0) * exp(m + s2 / 2), where s2 is the residual
                    # variance of log Y in the training rows.
                    pool = train & (observed == 1)
                    positive = pool & (y > 0)
                    if positive.sum() < 2:
                        raise ValueError(
                            f"repeat {repeat}, fold {b} has fewer than two training rows with a "
                            "positive observed outcome, so the two-part model cannot be fitted")
                    employed_data = {
                        "employed": (y[pool] > 0).astype(float),
                        spec.treatment: a[pool],
                        **{name: values[name][pool] for name in spec.covariates},
                    }
                    with run_notes.scope("employment"):
                        employment, employment_info = _fit_nuisance(
                            "employed", employed_data, rhs_pooled, Binomial(),
                            smooth=smooth_flags["outcome"] or outcome_custom is not None,
                            mstop=steps_for("employment"), max_mstop=max_mstop, tune=tune,
                            inner_folds=inner_folds, seed=repeat_seed + 3 * b + 5,
                            weights=sample_weight[pool], notes=run_notes, label="employment",
                            extrapolation=extrapolation,
                        )
                    log_rows = _with_log_columns({
                        "log_outcome": np.log(y[positive]),
                        spec.treatment: a[positive],
                        **{name: values[name][positive] for name in spec.covariates},
                    }, log_names)
                    with run_notes.scope("log_outcome"):
                        log_model, log_info = _fit_nuisance(
                            "log_outcome", log_rows, rhs_log, Gaussian(),
                            smooth=smooth_flags["outcome"] or outcome_custom is not None,
                            mstop=steps_for("log_outcome"), max_mstop=max_mstop, tune=tune,
                            inner_folds=inner_folds, seed=repeat_seed + 3 * b + 7,
                            weights=sample_weight[positive], notes=run_notes, label="log_outcome",
                            extrapolation=extrapolation,
                        )
                    fitted_log = _nuisance_predict(
                        log_model, _with_log_columns(
                            {name: values[name][positive] for name in spec.covariates}
                            | {spec.treatment: a[positive]}, log_names),
                        extrapolation, train=log_rows)
                    s2 = _rust.log_residual_variance(
                        np.log(y[positive]), fitted_log, sample_weight[positive])
                    for arm, target in ((0, mu0_run), (1, mu1_run)):
                        arm_newdata = {
                            **test_data,
                            spec.treatment: np.full(int(test.sum()), float(arm)),
                        }
                        employed_prob = _nuisance_predict(
                            employment, arm_newdata, extrapolation, train=employed_data)
                        log_mean = _nuisance_predict(
                            log_model, _with_log_columns(arm_newdata, log_names),
                            extrapolation, train=log_rows)
                        target[repeat, test] = _rust.two_part_mean(employed_prob, log_mean, s2)
                    arm_stops["employment"] = employment_info["mstop"]
                    arm_stops["log_outcome"] = log_info["mstop"]
                    run_records["employment"].append(employment_info)
                    run_records["log_outcome"].append(log_info)
                else:
                    for arm, target in ((0, mu0_run), (1, mu1_run)):
                        arm_rows = train & (a == arm) & (observed == 1)
                        if arm_rows.sum() < 4:
                            raise ValueError(
                                f"repeat {repeat}, fold {b} has fewer than four training "
                                f"rows with an observed outcome in treatment arm {arm}"
                            )
                        arm_data = {
                            spec.outcome: y[arm_rows],
                            **{name: values[name][arm_rows] for name in spec.covariates},
                        }
                        with run_notes.scope(f"mu{arm}"):
                            outcome, outcome_info = _fit_nuisance(
                                spec.outcome, arm_data, rhs_outcome_base, outcome_family,
                                smooth=smooth_flags["outcome"] or outcome_custom is not None,
                                mstop=steps_for(f"mu{arm}"), max_mstop=max_mstop, tune=tune,
                                inner_folds=inner_folds, seed=repeat_seed + 3 * b + arm + 1,
                                weights=sample_weight[arm_rows], notes=run_notes,
                                label=f"mu{arm}", extrapolation=extrapolation,
                            )
                            target[repeat, test] = _nuisance_predict(
                                outcome, test_data, extrapolation)
                        arm_stops[f"mu{arm}"] = outcome_info["mstop"]
                        run_records[f"mu{arm}"].append(outcome_info)

        return {"mu0": mu0_run, "mu1": mu1_run, "records": run_records,
                "stops": stops_all, "rhs_pooled": rhs_pooled, "rhs_log": rhs_log,
                "notes": run_notes}

    unchosen_records = {}
    if requested_model == "cv":
        # Fit each candidate outcome model on the folds above and keep the one
        # with the lowest out-of-fold loss on the observed outcomes. The loss is
        # computed by the Rust scorer; Python only takes the minimum.
        auto_model, auto_interactions, _ = _resolve_outcome_model(
            "auto", outcome_interactions, sparse, custom=outcome_custom is not None,
            n_rows=n)
        candidate_specs = {"pooled": ("pooled", False), "separate": ("separate", False)}
        if auto_model == "pooled" and auto_interactions and outcome_custom is None:
            candidate_specs["pooled_interactions"] = ("pooled", True)
        if zero_inflated:
            candidate_specs["two_part"] = ("two_part", False)
        cv_runs, cv_loss = {}, {}
        for label, (cand_model, cand_interactions) in candidate_specs.items():
            cv_runs[label] = fit_outcome_models(cand_model, cand_interactions)
            cv_loss[label] = _outcome_oof_loss(
                spec.outcome_type, y, a, observed, cv_runs[label]["mu0"].mean(axis=0),
                cv_runs[label]["mu1"].mean(axis=0), sample_weight)
        chosen_label = min(cv_loss, key=cv_loss.get)
        outcome_model, outcome_interactions = candidate_specs[chosen_label]
        run = cv_runs[chosen_label]
        choice_reason = (
            f"it had the lowest out-of-fold outcome loss ({cv_loss[chosen_label]:.6g}) among "
            f"{', '.join(f'{k} ({v:.6g})' for k, v in cv_loss.items())}")
        # Step counts of the candidates that were not chosen, so that the
        # summary of this fit can drive a cv refit, which fits every candidate.
        for label, cand_run in cv_runs.items():
            if label == chosen_label:
                continue
            for name, rows in cand_run["records"].items():
                if rows:
                    unchosen_records.setdefault(name, []).extend(rows)
        outcome_cv = {
            "candidates": list(candidate_specs), "loss": cv_loss, "chosen": chosen_label,
            "loss_name": ("mean binomial deviance" if spec.outcome_type == "binary"
                          else "weighted mean squared error"),
        }
    else:
        outcome_model, outcome_interactions, choice_reason = _resolve_outcome_model(
            requested_model, outcome_interactions, sparse, custom=outcome_custom is not None,
            n_rows=n)
        run = fit_outcome_models(outcome_model, outcome_interactions)
        outcome_cv = None
    sparse["outcome_model"] = outcome_model
    sparse["outcome_interactions"] = outcome_interactions
    notes.absorb(run["notes"])
    mu0_repeats, mu1_repeats = run["mu0"], run["mu1"]
    rhs_pooled, rhs_log = run["rhs_pooled"], run["rhs_log"]
    for name, rows in run["records"].items():
        records[name].extend(rows)
    for repeat, repeat_stops in enumerate(run["stops"]):
        for b, step_record in enumerate(repeat_stops):
            arm_stops_all[repeat][b].update(step_record)
    selected_by_repeat = arm_stops_all
    notes.emit()

    capped =[_NUISANCE_LABELS[name] for name, rows in records.items()
              if any(row["at_max_mstop"] for row in rows)]
    if capped:
        warnings.warn(
            f"aipw: the cross-validated risk of the {', '.join(capped)} model(s) "
            f"was still falling at max_mstop={max_mstop} steps, the largest "
            f"step count the search may reach, so mstop={max_mstop} is where that "
            "search stopped, not where the risk turns. The search starts at "
            f"mstop={mstop} and doubles while the risk falls. Raise "
            f"aipw(max_mstop=...) above {max_mstop} to search further.",
            RuntimeWarning, stacklevel=2)

    pi = pi_repeats.mean(axis=0)
    mu0 = mu0_repeats.mean(axis=0)
    mu1 = mu1_repeats.mean(axis=0)
    outcome_range = _outcome_range_check(y, observed, mu0, mu1, outcome_model)
    fold_id = fold_ids[:, 0]
    sparse["warned"] = bool(sparse["flagged_bins"]) and (
        outcome_model == "separate" or outcome_interactions)
    if sparse["warned"]:
        warnings.warn(_sparse_arm_message(sparse, spec.covariates, outcome_model,
                                          outcome_interactions, requested_model),
                      UserWarning, stacklevel=2)

    from mboostrs import _rust

    (score, estimate, influence, _treated_share, e_used, pi_used, n_clipped,
     n_clipped_response, response_ess) = _rust.augmented_scores(
        y, a, observed, sample_weight, e, pi, mu0, mu1, score_estimand,
        score_clip, propensity_clip if use_response else None,
    )
    se, critical, variance_info = _influence_uncertainty(
        influence, scheme, level, sample_weight)
    repeat_estimates = repeat_ses = None
    if repeats > 1:
        # Each repeat's own estimate and standard error, from its own nuisances.
        # The averaged-nuisance estimate above is the default; "median" replaces
        # it with the median rule over these.
        repeat_estimates, repeat_ses = [], []
        for r in range(repeats):
            _, est_r, inf_r, *_ = _rust.augmented_scores(
                y, a, observed, sample_weight, e_repeats[r], pi_repeats[r],
                mu0_repeats[r], mu1_repeats[r], score_estimand, score_clip,
                propensity_clip if use_response else None,
            )
            se_r, _, _ = _influence_uncertainty(inf_r, schemes[r][0], level, sample_weight)
            repeat_estimates.append(float(est_r))
            repeat_ses.append(float(se_r))
        if repeat_aggregation == "median":
            combined_estimate, combined_se = _rust.median_repeat_combine(
                np.asarray(repeat_estimates, dtype=np.float64)[:, None],
                np.asarray(repeat_ses, dtype=np.float64)[:, None],
            )
            estimate = float(combined_estimate[0])
            se = float(combined_se[0])
    interval_ci = ((estimate - critical * se, estimate + critical * se)
                   if se is not None else None)
    ato_companion = None
    if spec.estimand == "ate" and target_used == "all":
        lo_b, hi_b = overlap_bounds
        outside_share = float(np.mean((e < lo_b) | (e > hi_b)))
        if outside_share >= 0.05:
            _, ato_est, ato_inf, *_ = _rust.augmented_scores(
                y, a, observed, sample_weight, e, pi, mu0, mu1,
                "overlap", None, propensity_clip if use_response else None,
            )
            ato_se, ato_crit, _ = _influence_uncertainty(
                ato_inf, scheme, level, sample_weight)
            if ato_se is not None and ato_crit is not None:
                ato_lo = float(ato_est - ato_crit * ato_se)
                ato_hi = float(ato_est + ato_crit * ato_se)
            else:
                ato_lo = ato_hi = None
            ato_companion = {
                "estimate": float(ato_est),
                "se": None if ato_se is None else float(ato_se),
                "ci_lower": ato_lo,
                "ci_upper": ato_hi,
                "outside_overlap_share": outside_share,
            }
    bootstrap_record = None
    if interval == "bootstrap":
        bootstrap_record = _aipw_bootstrap(
            spec, values, raw_a, y, sample_weight, weights is not None, scheme,
            bootstrap_draws, seed, level,
            dict(redo, propensity_clip=propensity_clip, outcome_missing=outcome_missing,
                 overlap_bounds=overlap_bounds))
        bootstrap_record["influence_standard_error"] = se
        bootstrap_record["influence_interval"] = interval_ci
        se = bootstrap_record["standard_error"]
        interval_ci = bootstrap_record["interval"]
    ipw = _effect_weights(a, e_used, score_estimand)
    treated_weights = a * ipw
    control_weights = (1 - a) * ipw

    def _effective_sample_size(weight):
        return float(_rust.effective_sample_size(sample_weight * weight))

    diagnostics = {
        "propensity_min": float(e.min()), "propensity_max": float(e.max()),
        "fraction_below_0_05": float(np.mean(e < .05)),
        "fraction_above_0_95": float(np.mean(e > .95)),
        "ipw_max": float(ipw.max()),
        "ipw_effective_sample_size": _effective_sample_size(ipw),
        "treated_weight_effective_sample_size": _effective_sample_size(treated_weights),
        "control_weight_effective_sample_size": _effective_sample_size(control_weights),
        "propensity_clip": propensity_clip,
        "propensity_clip_applied": score_clip,
        "target": target_used,
        "repeat_aggregation": repeat_aggregation if repeats > 1 else None,
        "repeat_estimates": repeat_estimates,
        "repeat_standard_errors": repeat_ses,
        "n_clipped_propensity": int(n_clipped),
        "outcome_missing": outcome_missing,
        "n_missing_outcome": int(n - observed.sum()),
        "response_model": ("fitted" if use_response
                           else "not fitted: no outcome is missing"),
        "response_propensity_min": float(pi.min()) if use_response else None,
        "response_propensity_max": float(pi.max()) if use_response else None,
        "n_clipped_response": int(n_clipped_response),
        "response_effective_sample_size": float(response_ess),
        "sample_weights": weights is not None,
        "weight_total": float(sample_weight.sum()),
        "effect_scale": ("risk_difference" if spec.outcome_type == "binary"
                         else "mean_difference"),
        "balance": _balance(values, spec, a, e_used, score_estimand, sample_weight),
        "sparse_arms": sparse,
        "outcome_range": outcome_range,
        "outcome_model": outcome_model,
        "outcome_model_choice": {
            "requested": requested_model,
            "used": outcome_model,
            "outcome_interactions": outcome_interactions,
            "reason": choice_reason,
        },
        "outcome_model_cv": outcome_cv,
        "propensity_learner": _propensity_record(
            propensity_learner, propensity_custom is not None, chosen_prop,
            prop_deviance, prop_runs),
        "nuisance_mstop": {
            name: {
                "first_grid": steps_for(name),
                "max_mstop": max_mstop if tune else None,
                "mstop_by_fold": [row["mstop"] for row in rows],
                "at_max_mstop": any(row["at_max_mstop"] for row in rows),
            }
            for name, rows in records.items() if rows
        },
        "nuisance_mstop_summary": {
            name: int(round(float(np.median([row["mstop"] for row in rows]))))
            for name, rows in {**unchosen_records,
                               **{k: v for k, v in records.items() if v}}.items()
            if rows
        },
        **variance_info,
        "interval_method": interval,
        "bootstrap": bootstrap_record,
    }
    # The propensity model of all rows, which predict_propensity scores new rows
    # with. It uses the formula, smooth flag, first step count and tuning of the
    # cross-fitted propensity fits. It does not enter the estimate.
    full_smooth = smooth_flags["propensity"] if chosen_prop == "smooth" else True
    full_propensity, _ = _fit_nuisance(
        spec.treatment,
        {spec.treatment: a, **{name: values[name] for name in spec.covariates}},
        rhs_base, Binomial(), smooth=full_smooth,
        mstop=steps_for("propensity"), max_mstop=max_mstop, tune=tune,
        inner_folds=inner_folds, seed=seed, weights=sample_weight,
        extrapolation=extrapolation,
    )

    result = AIPWResult(
        estimate=estimate, standard_error=se, interval=interval_ci, level=level,
        spec=spec, treatment=a, outcome=y, propensity=e, mu0=mu0, mu1=mu1,
        scores=score, influence=influence, fold_id=fold_id,
        diagnostics=diagnostics,
        nuisance={
            "smooth": smooth, "smooth_by_nuisance": smooth_flags, "df": df,
            "tuned": tune, "target": target_used,
            "mstop": mstop, "max_mstop": max_mstop,
            "extrapolation": extrapolation, "flat_tolerance": flat_tolerance,
            # The resolved model, read by confounding_benchmark, which refits with it.
            "outcome_model": outcome_model,
            "outcome_model_requested": requested_model,
            "outcome_interactions": outcome_interactions,
            # Whether a user's propensity_terms or outcome_terms replaced a formula.
            # confounding_benchmark refits on fewer covariates and cannot carry them.
            "custom_terms": {"propensity": propensity_custom is not None,
                             "outcome": outcome_custom is not None},
            # The right-hand side of each nuisance formula, as fitted.
            "terms": {
                "propensity": rhs_base,
                "response": rhs_response if use_response else None,
                "outcome": (f"employment: {rhs_pooled}; log outcome: {rhs_log}"
                            if outcome_model == "two_part"
                            else rhs_pooled if outcome_model == "pooled"
                            else rhs_outcome_base),
            },
            "inner_folds": inner_folds if tune else None,
            "selected_mstop": (selected_by_repeat[0] if repeats == 1
                               else selected_by_repeat),
            "selected_mstop_by_repeat": selected_by_repeat,
            "fold_scheme": scheme.scheme, "fold_repeats": repeats, "seed": seed,
            "fold_count": scheme.n_folds,
            "fold_input": "integer" if folds_were_integer else "scheme",
            "outcome_missing": outcome_missing,
            "split_randomness_in_variance": repeats > 1 and repeat_aggregation == "median",
        },
        fold_ids=fold_ids,
        observed=observed,
        response_propensity=pi,
        weights=None if weights is None else sample_weight,
        ato_companion=ato_companion,
        _propensity_model=full_propensity,
        _data=data,
        _groups=None if scheme.groups is None else np.asarray(scheme.groups),
        _ordered=bool(scheme.ordered),
    )
    result.diagnostics["overlap"] = overlap_report(
        result, data={name: values[name] for name in spec.covariates},
        bounds=overlap_bounds).to_dict()
    return result


def _aipw_bootstrap(spec, values, raw_treatment, y, sample_weight, weighted, scheme,
                    draws, seed, level, kwargs):
    """The bootstrap of the whole :func:`aipw` procedure.

    Draw ``b`` resamples the units (rows, or the clusters of ``scheme``) with
    the core's ``group_bootstrap_counts`` stream, builds the resampled data,
    and reruns ``aipw`` on it with grouped folds that keep the copies of one
    unit in one fold. The percentile interval and the standard error come from
    the core's ``bootstrap_percentile_interval``; the tally of failed draws
    goes through ``bootstrap_outcome``.
    """
    import time

    from mboostrs import _rust
    from mboostrs._bootstrap import FAILED, BootstrapTally, bootstrap_outcome
    from mboostrs._bootstrap import group_bootstrap_counts as unit_counts
    from mboostrs.folds import grouped_folds

    n = len(y)
    units = np.arange(n) if scheme.groups is None else np.asarray(scheme.groups)
    counts = unit_counts(units, sample_weight > 0, draws, int(seed) + 2024)
    n_folds = scheme.n_folds
    tally = BootstrapTally(draws)
    inner = aipw.__wrapped__

    def one(b):
        rows = np.repeat(np.arange(n), counts[:, b].astype(int))
        data = {
            spec.treatment: raw_treatment[rows], spec.outcome: y[rows],
            **{name: values[name][rows] for name in spec.covariates},
        }
        try:
            folds_b = grouped_folds(units[rows], B=n_folds,
                                    seed=int(seed) + _REPEAT_SEED_STEP * (b + 1))
            fit = inner(data, spec, folds=folds_b, repeats=1,
                        seed=int(seed) + _REPEAT_SEED_STEP * (b + 1),
                        weights=sample_weight[rows] if weighted else None,
                        interval="influence", **kwargs)
            value = float(fit.estimate)
            if not np.isfinite(value):
                raise FloatingPointError("the draw's estimate is not finite")
            return value
        except Exception as exc:  # a failed draw is counted, not fatal
            tally.mark(b, FAILED, exc)
            return float("nan")

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimates = np.asarray([one(b) for b in range(draws)], dtype=np.float64)
    seconds = time.perf_counter() - started
    counts_out = bootstrap_outcome(tally, B=draws, what="ATE")
    kept = estimates[np.isfinite(estimates)]
    if kept.size < 2:
        raise RuntimeError(
            f"only {kept.size} of {draws} bootstrap draws finished, too few for an interval")
    lower, upper, se = _rust.bootstrap_percentile_interval(kept[:, None], float(level))
    return {
        "draws": estimates, "n_draws": int(draws), "n_failed": counts_out["n_failed"],
        "n_used": counts_out["n_used"],
        "failed_messages": sorted({f"{type(e).__name__}: {e}" for e in tally.failed_errors}),
        "standard_error": float(se[0]), "interval": (float(lower[0]), float(upper[0])),
        "seconds": seconds, "unit": "cluster" if scheme.groups is not None else "row",
    }


#: A column with fewer distinct values than this enters an effect or
#: nuisance model linearly: a smooth needs values to bend through.
_SMOOTH_MIN_VALUES = 10


#: Quantile bins used to locate the numeric covariate region with most rows
#: outside the bounds.
_OVERLAP_BINS = 20


def _check_bounds(bounds):
    if (not isinstance(bounds, tuple) or len(bounds) != 2
            or not 0 < bounds[0] < bounds[1] < 1):
        raise ValueError("overlap bounds must be a tuple (lower, upper) with "
                         "0 < lower < upper < 1")
    return float(bounds[0]), float(bounds[1])


def _covariate_region(column, outside):
    """Covariate regions containing rows outside the specified overlap bounds.

    A numeric covariate is cut into quantile bins of about 5% of the rows
    each; the region is the span of the values in the bins where more than
    half of the rows are outside. A categorical covariate lists the levels
    where more than half of the rows are outside. ``outside`` is the boolean
    mask of rows outside the bounds."""
    if column.levels is not None:
        levels = []
        for code, level in enumerate(column.levels):
            member = column.values == code
            if not member.any():
                continue
            share = float(outside[member].mean())
            levels.append({"level": level, "rows": int(member.sum()),
                           "share_outside": share, "flagged": share > 0.5})
        return {"covariate": column.name, "kind": "categorical",
                "levels": tuple(levels),
                "flagged_levels": tuple(row["level"] for row in levels if row["flagged"])}
    x = column.values
    finite = np.isfinite(x)
    edges = np.unique(np.quantile(x[finite], np.linspace(0, 1, _OVERLAP_BINS + 1))) \
        if finite.sum() >= 2 else np.array([])
    if edges.size < 2:
        return {"covariate": column.name, "kind": "numeric", "region": None,
                "rows_in_region": 0, "share_outside_in_region": None,
                "share_outside_elsewhere": None}
    bin_of = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    flagged = []
    for b in range(edges.size - 1):
        member = (bin_of == b) & finite
        if member.any() and outside[member].mean() > 0.5:
            flagged.append(b)
    if not flagged:
        return {"covariate": column.name, "kind": "numeric", "region": None,
                "rows_in_region": 0, "share_outside_in_region": None,
                "share_outside_elsewhere": float(outside[finite].mean())}
    in_region = np.isin(bin_of, flagged) & finite
    elsewhere = finite & ~in_region
    return {
        "covariate": column.name, "kind": "numeric",
        "region": (float(x[in_region].min()), float(x[in_region].max())),
        "rows_in_region": int(in_region.sum()),
        "share_outside_in_region": float(outside[in_region].mean()),
        "share_outside_elsewhere": (float(outside[elsewhere].mean())
                                    if elsewhere.any() else None),
    }


def _weighted_mean_se(values, weight):
    """The weighted mean and its standard error, with the weights read as
    frequencies: the variance of the mean is ``sum w (x - m)^2 / (W (W - 1))``
    for total weight ``W``, which is the usual ``s^2 / n`` for unit weights."""
    total = float(weight.sum())
    mean = float(np.sum(weight * values) / total)
    if total <= 1:
        return mean, None
    variance = float(np.sum(weight * (values - mean) ** 2) / (total * (total - 1)))
    return mean, float(np.sqrt(variance))


def _restricted_aipw(result, inside):
    """The estimand among the rows inside the bounds, with a standard error
    that treats the inside rows as an independent sample. For the ATE it is
    the weighted mean of the augmented score over those rows, which estimates
    ``E[tau(X) | X in region]``. For the ATT it is the ratio of the numerator
    sums over the treated rows in the region, which estimates
    ``E[e(X) tau(X) | X in region] / E[e(X) | X in region]``, the effect on
    the treated in that region. For the overlap target the scores are not
    the ATE scores, so no restricted estimate is reported."""
    if result.target == "overlap" or int(inside.sum()) < 2:
        return None, None
    n = inside.size
    weight = (np.ones(n) if result.weights is None
              else np.asarray(result.weights, dtype=np.float64))
    w_in = weight[inside]
    if result.spec.estimand == "ate":
        return _weighted_mean_se(result.scores[inside], w_in)
    treated_share = float(np.sum(weight * result.treatment) / weight.sum())
    numerator = result.scores[inside] * treated_share
    treated = result.treatment[inside]
    denominator = float(np.sum(w_in * treated))
    if denominator <= 0:
        return None, None
    estimate = float(np.sum(w_in * numerator) / denominator)
    mean_treated = denominator / float(w_in.sum())
    influence = (numerator - estimate * treated) / mean_treated
    _, se = _weighted_mean_se(influence, w_in)
    return estimate, se


@dataclass(frozen=True)
class OverlapReport:
    """Rows whose estimated propensity lies outside ``bounds``, where they
    are, and the estimate among the rows inside the bounds.

    ``regions`` has one entry per covariate passed with ``data``. Numeric
    entries give the ``region`` (span of values) where most rows are outside,
    with the share outside inside and outside it; categorical entries list
    each level with its share outside. ``estimate`` and ``standard_error`` are
    the full-population estimate of a result (``None`` for a bare propensity
    array), and ``estimate_inside`` the same estimand restricted to rows whose
    propensity is inside ``bounds``. The restricted standard error treats the
    inside rows as an independent sample and does not account for the
    cross-fitting or for choosing the bounds after seeing the propensities.
    """

    bounds: tuple
    n: int
    n_below: int
    n_above: int
    regions: tuple
    estimand: str | None
    estimate: float | None
    standard_error: float | None
    estimate_inside: float | None
    standard_error_inside: float | None
    n_inside: int

    @property
    def n_outside(self) -> int:
        return self.n_below + self.n_above

    @property
    def share_outside(self) -> float:
        return self.n_outside / self.n

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the counts, regions and estimates."""
        return export_result(
            self, include_arrays=include_arrays,
            extras={"n_outside": self.n_outside, "share_outside": self.share_outside},
        )

    def __repr__(self) -> str:
        return (f"OverlapReport(share_outside={self.share_outside:.4g}, "
                f"n={self.n}, bounds={self.bounds}, estimate={self.estimate!r}, "
                f"estimate_inside={self.estimate_inside!r})")

    def describe(self) -> str:
        """The share outside the bounds, the flagged regions and both estimates."""
        lower, upper = self.bounds
        text = (f"{self.n_outside} of {self.n} rows ({self.share_outside:.1%}) have "
                f"propensity outside [{lower:g}, {upper:g}]: {self.n_below} below, "
                f"{self.n_above} above.")
        for region in self.regions:
            if region["kind"] == "numeric" and region["region"] is not None:
                low, high = region["region"]
                text += (f" {region['covariate']} in [{low:.4g}, {high:.4g}] holds "
                         f"{region['rows_in_region']} rows, "
                         f"{region['share_outside_in_region']:.0%} outside.")
            elif region["kind"] == "categorical" and region["flagged_levels"]:
                text += (f" {region['covariate']} levels "
                         f"{list(region['flagged_levels'])} are mostly outside.")
        if self.estimate is not None and self.estimate_inside is not None:
            text += (f" {self.estimand.upper() if self.estimand else 'estimate'}: full "
                     f"{self.estimate:.6g}, inside the bounds {self.estimate_inside:.6g} "
                     f"({self.n_inside} rows).")
        return text


def overlap_report(source, data=None, bounds=FROM_SPEC, *, covariates=None):
    """Summarize where the estimated propensity leaves the overlap bounds.

    Parameters
    ----------
    source : array, AIPWResult or EffectModel
        Cross-fitted propensities ``P(treated | x)`` for each row, or a result
        that carries them. For a result, the covariates default to its
        specification's and the estimate is reported next to the restricted
        one.
    data : mapping or data frame, optional
        The rows' covariate columns. Without it the covariate regions are not
        computed.
    bounds : tuple, optional
        ``(lower, upper)``: rows with propensity below ``lower`` or above
        ``upper`` are outside. Defaults to the ``overlap_bounds`` of the
        result's spec, and to ``(0.02, 0.98)`` for a bare array or an
        :class:`EffectModel` without one. ``(0.02, 0.98)`` is a convention,
        not a theorem; state the bounds used.
    covariates : sequence of str, optional
        Columns of `data` to summarize when `source` is a bare array. A result
        uses its own covariates.

    Returns
    -------
    OverlapReport
        ``share_outside`` is the fraction of rows outside the bounds.
        ``regions`` locates them by covariate. ``estimate_inside`` is the
        estimand of the result restricted to the rows inside the bounds: the
        mean augmented score for an ATE, the treated-row ratio for an ATT, and
        the mean effect over the fitting rows for an EffectModel. For
        ``aipw(target="overlap")`` it is None, since the overlap scores are not
        ATE scores.
    """
    spec = source.spec if isinstance(source, (AIPWResult, EffectModel)) else None
    lower, upper = _check_bounds(_spec_default(spec, "overlap_bounds", bounds))
    if isinstance(source, (AIPWResult, EffectModel)):
        propensity = np.asarray(source.propensity, dtype=np.float64)
        names = source.spec.covariates if covariates is None else tuple(covariates)
    else:
        propensity = np.asarray(source, dtype=np.float64).ravel()
        if data is not None and covariates is None:
            raise ValueError("pass covariates=... with a propensity array and data, "
                             "naming the columns to summarize")
        names = () if covariates is None else tuple(covariates)
    if propensity.size == 0 or not np.all(np.isfinite(propensity)) \
            or np.any((propensity < 0) | (propensity > 1)):
        raise ValueError("propensities must be finite values in [0, 1]")
    below = propensity < lower
    above = propensity > upper
    outside = below | above
    inside = ~outside
    regions = []
    if data is not None:
        for name in names:
            column = read_column(data, name)
            if column.values.size != propensity.size:
                raise ValueError(f"column {name!r} has {column.values.size} rows, "
                                 f"the propensities {propensity.size}")
            regions.append(_covariate_region(column, outside))
    estimand = estimate = standard_error = None
    inside_estimate = inside_se = None
    if isinstance(source, AIPWResult):
        estimand = source._score_estimand()
        estimate = float(source.estimate)
        standard_error = source.standard_error
        inside_estimate, inside_se = _restricted_aipw(source, inside)
    elif isinstance(source, EffectModel):
        estimand = "ate"
        estimate = source.ate_plugin
        if inside.any():
            inside_estimate = float(np.mean(source.tau[inside]))
    return OverlapReport(
        bounds=(lower, upper), n=int(propensity.size), n_below=int(below.sum()),
        n_above=int(above.sum()), regions=tuple(regions), estimand=estimand,
        estimate=estimate, standard_error=standard_error,
        estimate_inside=inside_estimate, standard_error_inside=inside_se,
        n_inside=int(inside.sum()),
    )


def _propensity_of(propensity_model, spec, newdata):
    """``e(x)`` for new rows from the propensity model of an effect fit.

    Raises ``ValueError`` when ``newdata`` lacks a covariate of ``spec``."""
    if propensity_model is None:
        raise ValueError("this EffectModel has no propensity model; refit it with "
                         "effect_model() to get one")
    names = column_names(newdata)
    missing = [name for name in spec.covariates if name not in names]
    if missing:
        raise ValueError(
            f"propensity needs every covariate of the specification, and newdata lacks "
            f"{missing}; add those columns to newdata")
    covariates = _modifier_values(newdata, spec.covariates)
    return np.asarray(propensity_model.predict(newdata=covariates, type="response"),
                      dtype=np.float64).ravel()


class _Propensities(np.ndarray):
    """The cross-fitted propensities of the fitting rows, which can also be called.

    The array is ``EffectModel.propensity``: it is the value that :func:`aipw`
    and :func:`overlap_report` read as an array. Calling it with new rows,
    ``propensity(newdata)``, returns the estimated propensity ``e(x)`` of each
    row from the propensity model fitted on all rows."""

    def __new__(cls, values, predict):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj._predict = predict
        return obj

    def __array_finalize__(self, obj):
        self._predict = getattr(obj, "_predict", None)

    def __call__(self, newdata):
        """The estimated propensity ``e(x) = P(treatment = treated | x)`` of new rows.

        ``newdata`` needs every covariate of the specification (``spec.covariates``),
        as a mapping or a data frame. The values come from the propensity model that
        :func:`effect_model` fitted on all rows, with the same smooth terms and step
        count as its cross-fitted nuisances. The result is an array with one
        probability per row, in the same row order as ``newdata``. Use it to rank rows
        by overlap, as :func:`rank_by_effect` does, or to check where new rows lie
        relative to ``overlap_bounds``.

        Raises ``ValueError`` when ``newdata`` lacks a covariate or when the model has
        no propensity model.
        """
        if self._predict is None:
            raise ValueError("this propensity array has no model for new rows; use the "
                             "propensity of an EffectModel returned by effect_model()")
        return self._predict(newdata)


@dataclass(frozen=True)
class EffectModel:
    """A conditional average treatment effect ``tau(x)``, fitted by the
    R-learner.

    ``terms`` names the candidate whose cross-validated R-loss was lowest,
    and ``cv_rloss`` holds every candidate's score. ``tau`` holds the
    in-sample predictions and ``ate_plugin`` their unweighted mean; use
    :func:`aipw` for an average effect with an interval.

    ``model`` is the fitted ``tau``. For ``terms="smooth"`` it is a
    :class:`~mboostrs.selection.GamRefit`: a penalized additive model whose
    smoothing parameters are chosen by REML, so ``predict(..., interval=True)``
    gives a per-row interval. For ``"stumps"`` and ``"joint"`` it is a boosted
    model, which has no coefficient covariance and therefore no per-row
    interval.

    ``propensity_model`` is a boosted model of ``e(x)`` fitted on all rows; it
    flags new rows whose estimated propensity lies outside ``overlap_bounds``
    when :meth:`predict` is called. ``propensity`` holds the cross-fitted
    propensities of the fitting rows as an array; calling it, as
    ``propensity(newdata)``, returns ``e(x)`` for new rows. ``overlap`` in ``diagnostics`` summarizes
    the cross-fitted propensities of the fitting rows.

    The ``smooth`` and ``stumps`` candidates are additive in the modifiers:
    each modifier contributes its own terms and none of them can depend on
    another. The ``joint`` candidate is a single tree over all modifiers, so
    each modifier's share of it depends on the others; a fitted ``joint``
    model reports a combined ``partial_effects`` row rather than one row per
    modifier.
    """

    model: object = field(repr=False)
    spec: CausalSpec
    modifiers: tuple[str, ...]
    terms: str
    cv_rloss: dict
    tau: np.ndarray
    propensity: np.ndarray
    outcome_mean: np.ndarray
    diagnostics: dict
    propensity_model: object = field(default=None, repr=False, compare=False)
    overlap_bounds: tuple = DEFAULT_OVERLAP_BOUNDS
    #: Out-of-fold R-loss of the constant effect (``tau`` equal to the ATE on
    #: each fold's training rows), on the same folds as ``cv_rloss``.
    constant_rloss: float = float("nan")
    #: Scale and central range of each smooth modifier, as the smooth second stage
    #: enters it; empty unless the chosen candidate is ``"smooth"``.
    scale_plan: dict = field(default_factory=dict, repr=False, compare=False)
    #: How ``predict(..., interval=True)`` builds the per-row interval: ``"refit"``
    #: (the second stage's covariance given the nuisances), ``"repeats"`` (median
    #: rule over ``interval_fits``) or ``"bootstrap"`` (percentile interval over
    #: ``interval_fits``).
    interval_method: str = "refit"
    #: The refits behind the ``"repeats"`` and ``"bootstrap"`` intervals, each an
    #: :class:`EffectModel` fitted on another split seed or on resampled rows.
    #: Empty for ``"refit"`` and ``"second_stage_bootstrap"``.
    interval_fits: tuple = field(default=(), repr=False, compare=False)
    #: The second-stage row bootstrap behind ``"second_stage_bootstrap"``: a
    #: :class:`~mboostrs.selection.GamRefitBootstrap` with one refit per draw.
    second_stage: object = field(default=None, repr=False, compare=False)
    #: Calibrated DR/orthogonal interval state for ``interval_method="dr_orthogonal"``.
    _dr_state: dict | None = field(default=None, repr=False, compare=False)

    def _model_inputs(self, newdata):
        """The modifier columns of new rows on the scale the fitted ``tau`` uses."""
        return _apply_plan(_modifier_values(newdata, self.modifiers), self.scale_plan)

    def _outside_overlap(self, newdata):
        """Boolean flags for the new rows whose estimated propensity is outside
        ``overlap_bounds``, or ``None`` when the rows lack a covariate the
        propensity model needs. Warns once with the count of flagged rows."""
        if self.propensity_model is None:
            return None
        names = column_names(newdata)
        missing = [name for name in self.spec.covariates if name not in names]
        if missing:
            warnings.warn(
                f"predict cannot check overlap: the new rows lack {missing}, which the "
                "propensity model needs. Pass every covariate of the specification to "
                "check overlap; the returned effects carry no overlap flag.",
                stacklevel=3)
            return None
        e = _propensity_of(self.propensity_model, self.spec, newdata)
        lower, upper = self.overlap_bounds
        outside = (e < lower) | (e > upper)
        count = int(outside.sum())
        if count:
            warnings.warn(
                f"{count} of {outside.size} new rows have an estimated propensity outside "
                f"[{lower:g}, {upper:g}]: their effects are extrapolated from rows the "
                "other treatment arm rarely reaches. The rows are flagged in the "
                "outside_overlap column when predict(..., interval=True) is used.",
                stacklevel=3)
        return outside

    def _tau_at(self, newdata, extrapolation):
        """This fit's ``tau`` at the rows of ``newdata``, as a 1-D array."""
        return np.asarray(self.model.predict(newdata=self._model_inputs(newdata),
                                             extrapolation=extrapolation),
                          dtype=np.float64).ravel()

    def _interval_arrays(self, newdata, level, extrapolation):
        """The per-row estimate and standard error of this fit's own second-stage
        covariance, as arrays, with the range flag."""
        records = self.model.predict(newdata=self._model_inputs(newdata), interval=True,
                                     level=level, output="records",
                                     extrapolation=extrapolation)
        return {
            "estimate": np.array([float(r["estimate"]) for r in records]),
            "se": np.array([float(r["se"]) for r in records]),
            "outside_range": np.array([bool(r["outside_range"]) for r in records]),
        }

    def _second_stage_draws(self, newdata, extrapolation):
        """Each bootstrap draw's ``tau`` at the rows of ``newdata``, as a
        ``draws x rows`` array, on the scale the smooth second stage uses."""
        return self.second_stage.predict(self._model_inputs(newdata),
                                         extrapolation=extrapolation)

    def predict(self, newdata, interval=False, level=0.95, output="auto",
                extrapolation=None):
        """``tau(x)`` for new rows.

        The return type is set by ``output`` and ``interval``:

        * ``interval=False, output="auto"`` (the default) returns a NumPy array
          of the estimates, as earlier versions did.
        * ``interval=False`` with ``output="numpy"`` returns the same array.
        * ``interval=False`` with ``output="polars"``, ``"pandas"`` or
          ``"records"`` returns a frame, or a list of dicts, with one column,
          ``estimate``.
        * ``interval=True`` returns a frame by default, with ``estimate``,
          ``se``, ``lower``, ``upper`` (the pointwise interval at `level`),
          ``outside_overlap``, a boolean flag that is true where the estimated
          propensity lies outside ``overlap_bounds``, and ``outside_range``, true
          where a modifier lies outside the training range of its spline. The
          interval needs ``terms="smooth"``. With ``output="numpy"`` it returns a
          dict that maps each of those names to a NumPy array; the
          ``outside_overlap`` entry is None when the propensity is not available.

        Every frame output has an ``estimate`` column, so code that reads
        ``pred["estimate"]`` works with ``interval=True`` and with
        ``output="polars"``. Indexing the array of the default return by name
        does not work; use ``output="polars"`` for named columns.

        How the interval is built follows ``interval_method`` of
        :func:`effect_model`. ``"refit"`` uses the second stage's covariance
        given the nuisances. ``"repeats"`` takes the estimate and standard error
        at each row from every refit over a different split seed, and combines
        them by the median rule of Chernozhukov et al. (2018): the estimate is
        the median over refits, and the variance is the median of
        ``se^2 + (estimate_r - estimate)^2``. With ``"repeats"``, the array
        returned without ``interval`` is the same median. ``"bootstrap"``
        reports the full-sample estimate with the percentile interval of the
        refits on resampled rows. For ``"repeats"`` and ``"bootstrap"``,
        ``outside_range`` is the flag of the full-sample fit.

        ``extrapolation`` is passed to the fitted ``tau`` model, which applies
        it to the modifiers that have a spline basis (see
        :meth:`mboostrs.selection.GamRefit.predict`). ``"clip"``, the default
        for ``terms="smooth"``, holds each modifier at its central range edge,
        the 1st or 99th percentile of the fitting rows, so the effect is flat
        beyond the range the spline was fitted on. The default for the boosted
        candidates is ``"warn"``, which predicts and warns once. ``"error"``
        refuses; ``"allow"`` is silent and continues the spline's edge slope.

        The modifier columns are required. The covariates of the propensity
        model are required only to check overlap; without them the effects are
        returned, with no flag and a warning.

        A row outside the bounds still gets an effect; its estimate is an
        extrapolation. The call warns once with the number of such rows.
        """
        from mboostrs import _rust

        if output not in ("auto", "numpy", "polars", "pandas", "records"):
            raise ValueError("output must be 'auto', 'numpy', 'polars', 'pandas' or "
                             f"'records'; got {output!r}")
        if extrapolation is None:
            extrapolation = "clip" if isinstance(self.model, GamRefit) else "warn"
        outside = self._outside_overlap(newdata)
        if not interval:
            if self.interval_method == "repeats":
                taus = np.vstack([f._tau_at(newdata, extrapolation)
                                  for f in (self, *self.interval_fits)])
                estimates = _rust.median_repeat_combine(taus, np.zeros_like(taus))[0]
            else:
                estimates = self._tau_at(newdata, extrapolation)
            if output in ("auto", "numpy"):
                return np.asarray(estimates)
            return output_frame([{"estimate": float(value)} for value in estimates], output,
                                columns=("estimate",))
        if not isinstance(self.model, GamRefit) and self.interval_method != "dr_orthogonal":
            raise NotImplementedError(
                f"interval=True needs a penalized smooth fit, but this model chose "
                f"terms={self.terms!r}, a boosted fit with no coefficient covariance. "
                "Refit with effect_model(..., terms='smooth') to get per-row intervals.")
        if self.interval_method == "dr_orthogonal":
            level = check_unit_interval("level", level)
            if self._dr_state is None:
                raise RuntimeError("dr_orthogonal interval state is missing")
            if self._dr_state["kind"] == "smooth":
                estimate, se, lower, upper, outside_range = _dr_smooth_predict(
                    self, newdata, level, extrapolation)
            else:
                from mboostrs.cvrisk import _t_quantile

                inputs = self._model_inputs(newdata)
                estimate = self._tau_at(newdata, extrapolation)
                draws = np.vstack([
                    np.asarray(m.predict(newdata=inputs, extrapolation=extrapolation),
                               dtype=np.float64).ravel()
                    for m in self._dr_state["models"]
                ])
                n_draws = draws.shape[0]
                se = np.std(draws, axis=0, ddof=1) * np.sqrt(
                    (n_draws + 1.0) / max(n_draws - 1.0, 1.0))
                tcrit = _t_quantile(0.5 + float(level) / 2.0, max(1, n_draws - 1))
                lower = estimate - tcrit * se
                upper = estimate + tcrit * se
                outside_range = np.zeros(len(estimate), dtype=bool)
        elif self.interval_method == "second_stage_bootstrap":
            draws = self._second_stage_draws(newdata, extrapolation)
            lower, upper, se = _rust.bootstrap_percentile_interval(draws, level)
            full = self._interval_arrays(newdata, level, extrapolation)
            estimate, outside_range = full["estimate"], full["outside_range"]
        elif self.interval_method == "bootstrap":
            draws = np.vstack([f._tau_at(newdata, extrapolation) for f in self.interval_fits])
            lower, upper, se = _rust.bootstrap_percentile_interval(draws, level)
            full = self._interval_arrays(newdata, level, extrapolation)
            estimate, outside_range = full["estimate"], full["outside_range"]
        elif self.interval_method == "repeats":
            parts = [f._interval_arrays(newdata, level, extrapolation)
                     for f in (self, *self.interval_fits)]
            estimate, se = _rust.median_repeat_combine(
                np.vstack([p["estimate"] for p in parts]),
                np.vstack([p["se"] for p in parts]))
            z = NormalDist().inv_cdf(0.5 + level / 2)
            lower, upper = estimate - z * se, estimate + z * se
            outside_range = parts[0]["outside_range"]
        else:
            full = self._interval_arrays(newdata, level, extrapolation)
            estimate, se, outside_range = full["estimate"], full["se"], full["outside_range"]
            z = NormalDist().inv_cdf(0.5 + level / 2)
            lower, upper = estimate - z * se, estimate + z * se
        if output == "numpy":
            return {
                "estimate": np.asarray(estimate, dtype=float),
                "se": np.asarray(se, dtype=float),
                "lower": np.asarray(lower, dtype=float),
                "upper": np.asarray(upper, dtype=float),
                "outside_overlap": None if outside is None else np.asarray(outside, dtype=bool),
                "outside_range": np.asarray(outside_range, dtype=bool),
            }
        rows = [
            {"estimate": float(estimate[i]), "se": float(se[i]),
             "lower": float(lower[i]), "upper": float(upper[i]),
             "outside_overlap": None if outside is None else bool(outside[i]),
             "outside_range": bool(outside_range[i])}
            for i in range(len(estimate))
        ]
        return output_frame(rows, output)

    @property
    def ate_plugin(self) -> float:
        """The unweighted mean of the fitted ``tau(x)`` over the fitting rows.

        Each row counts once; rows are not weighted by overlap (the R-loss
        weights ``e(x)(1 - e(x))`` are not applied). The value is an estimate
        of the population average effect only as far as the fitted ``tau`` is
        accurate. It has no standard error. Use :func:`aipw` for an average
        effect with an interval.
        """
        return float(np.mean(self.tau))

    def overlap_report(self, bounds=None, data=None):
        """The overlap summary of the fitting rows; see :func:`overlap_report`."""
        return overlap_report(self, data=data,
                              bounds=self.overlap_bounds if bounds is None else bounds)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; per-row arrays only on request."""
        return export_result(
            self, include_arrays=include_arrays,
            arrays=("tau", "propensity", "outcome_mean"),
            extras={"n": len(self.tau), "ate_plugin": self.ate_plugin},
        )

    def __repr__(self) -> str:
        return (f"EffectModel(terms={self.terms!r}, modifiers={list(self.modifiers)}, "
                f"ate_plugin={self.ate_plugin:.6g}, n={len(self.tau)})")

    def describe(self) -> str:
        """The fitted effect, how it was chosen, and what it assumes."""
        d = self.diagnostics
        losses = ", ".join(f"{k} {v:.6g}" for k, v in self.cv_rloss.items())
        shape = ("one joint tree over" if self.terms == "joint"
                 else "additive in")
        if isinstance(self.model, GamRefit):
            fit = (f"penalized smooths (REML, edf {sum(self.model.edf.values()):.3g} "
                   "in total)")
            interval = _EFFECT_INTERVAL_TEXT[self.interval_method]
            if self.second_stage is not None:
                interval = interval.format(draws=self.second_stage.n_draws)
            interval += (" (these cover the sampling variation of the second stage; a bias "
                         "of the estimate, such as one from the nuisance models or from a "
                         "penalized smooth that flattens a sharp effect, is not covered)")
        else:
            fit = f"{self.model.mstop} boosting steps"
            interval = (_EFFECT_INTERVAL_TEXT["dr_orthogonal"]
                        if self.interval_method == "dr_orthogonal"
                        else "no per-row interval (boosted fit)")
        overlap = d.get("overlap", {})
        share = overlap.get("share_outside")
        overlap_text = ("" if share is None else
                        f" {share:.1%} of fitting rows have propensity outside "
                        f"{tuple(overlap.get('bounds', ()))}.")
        constant = d.get("constant_effect")
        if constant:
            constant_text = (
                f" A constant effect (the ATE) has out-of-fold R-loss "
                f"{constant['rloss']:.6g}, against {self.cv_rloss[self.terms]:.6g} for "
                f"{self.terms}: the constant's loss minus {self.terms}'s is "
                f"{constant['gain']:.6g} (standard error {constant['gain_se']:.6g}).")
        else:
            constant_text = ""
        scale_text = ""
        if self.scale_plan:
            logged = [name for name, entry in self.scale_plan.items() if entry["log"]]
            scale_text = (
                " Smooth modifiers are fitted over their 1st to 99th percentile range, and "
                "their effect is held flat beyond it"
                + (f"; {logged} enter on the log scale" if logged else "") + ".")
        bounds = d.get("pseudo_bounds")
        learner_text = (
            f" Pseudo-outcome {d.get('learner', 'r')}"
            + ("" if bounds is None else
               f", clipped to [{bounds[0]:.6g}, {bounds[1]:.6g}] for "
               f"{d.get('n_winsorized', 0)} rows")
            + ".")
        selection = d.get("terms_selection") or {}
        selection_text = (f" terms='auto' rule: {selection['rule']}."
                          if selection.get("rule") not in (None, "requested") else "")
        return (
            f"tau(x) for {self.spec.treatment} on {self.spec.outcome}, {shape} "
            f"{list(self.modifiers)}; {self.terms} terms chosen by cross-validated "
            f"R-loss ({losses}); {fit}; {interval}.{selection_text}{constant_text}{learner_text}"
            f"{scale_text} "
            f"Mean in-sample effect "
            f"{self.ate_plugin:.6g} (no interval; see aipw). Propensities clipped to "
            f"{d['propensity_clip']} for {d['n_clipped']} rows; nuisances from "
            f"{d['nuisance_source']}.{_propensity_text(d.get('propensity_learner'))}"
            f"{overlap_text} Identification assumes "
            + "; ".join(self.spec.assumptions) + "."
        )

    def digest(self) -> str:
        """Return a concise, deterministic summary card of this CATE model."""
        d = self.diagnostics
        overlap = d.get("overlap") or {}
        ce = d.get("constant_effect") or {}
        losses = ", ".join(f"{k}={v:.4g}" for k, v in self.cv_rloss.items())
        fit_info = (f"edf={sum(self.model.edf.values()):.3g}"
                    if isinstance(self.model, GamRefit)
                    else f"mstop={self.model.mstop}")
        lines = [
            f"[mboostrs EffectModel] terms={self.terms}, n={len(self.tau)}, "
            f"modifiers={list(self.modifiers)}, treatment={self.spec.treatment}, "
            f"outcome={self.spec.outcome}",
            f"ate_plugin={self.ate_plugin:.6g}, {fit_info}, interval_method={self.interval_method}",
            f"cv_rloss=({losses}), constant_rloss={self.constant_rloss:.6g}, "
            f"beats_constant_by_se={ce.get('beats_by_se')}",
            f"outside_overlap={overlap.get('share_outside', 0.0):.1%}, "
            f"learner={d.get('learner')}, nuisance={d.get('nuisance_source')}",
        ]
        return "\n".join(lines)


#: Depth of the joint candidate. Three levels can express an effect that turns
#: on one modifier within the range of another, and each step of such a tree
#: fits more of the residual than one split does, so the shared step count is
#: spent on the interaction instead of on rebuilding the additive part.
_JOINT_DEPTH = 3


def _rhs_for(modifiers, values, terms):
    """The right-hand side of one candidate specification of ``tau``."""
    if terms == "joint":
        # ``sse`` and not the default conditional-inference selection: the
        # marginal association of each modifier can be near zero when its
        # effect needs another modifier to appear, and a rule that ranks
        # columns by that association would never offer it as the first
        # split.
        return (f"btree({', '.join(quote_name(_c) for _c in modifiers)}, maxdepth={_JOINT_DEPTH:g}, "
                "selection=sse)")
    return " + ".join(f"btree({quote_name(name)})" for name in modifiers)


#: How much lower the joint candidate's cross-validated R-loss must be, as a
#: fraction of the best additive score, before it is preferred. Cross-validated
#: losses within a few percent of each other do not separate the dictionaries,
#: and the additive form is the one whose effect can be read as one curve per
#: modifier. The margin is well below the 27-38 percent by which the joint
#: candidate wins an effect that needs two modifiers, and well above the
#: under-2 percent by which it loses or ties on the others.
_JOINT_MARGIN = 0.05


def _choose_candidate(cv_rloss):
    """The candidate with the lowest R-loss, subject to the joint margin."""
    additive = {k: v for k, v in cv_rloss.items() if k != "joint"}
    best_additive = min(additive, key=additive.get) if additive else None
    if ("joint" in cv_rloss and best_additive is not None
            and cv_rloss["joint"] < cv_rloss[best_additive] * (1 - _JOINT_MARGIN)):
        return "joint"
    return best_additive or "joint"


#: The candidate that ``terms="auto"`` returns when no candidate is eligible.
_AUTO_FALLBACK = "smooth"


def _gain_and_se(diff, sample_weight):
    """Mean of the per-row loss differences ``diff`` and its standard error,
    with the observation weights as frequencies (``_rust.rloss_gain``).
    ``None`` weights are unit weights."""
    from mboostrs import _rust

    diff = np.ascontiguousarray(diff, dtype=np.float64)
    weight = (np.ones_like(diff) if sample_weight is None
              else np.ascontiguousarray(sample_weight, dtype=np.float64))
    gain, se = _rust.rloss_gain(diff, weight)
    return float(gain), float(se)


def _candidate_gains(loss_rows, constant_loss, mstops, sample_weight=None):
    """Each candidate's out-of-fold R-loss gain over the constant effect, the
    standard error of that gain over the rows (observation weights
    ``sample_weight`` counted as frequencies), whether its step count is too
    small to be a fit (``flat``), and whether it is eligible: its gain exceeds
    its standard error and it is not flat. Returns four dicts keyed by candidate."""
    gain, gain_se, flat, eligible = {}, {}, {}, {}
    for kind, rows in loss_rows.items():
        gain[kind], gain_se[kind] = _gain_and_se(constant_loss - rows, sample_weight)
        flat[kind] = bool(kind in mstops and mstops[kind] < _CLOSE_TO_CONSTANT_STEPS)
        eligible[kind] = bool(gain[kind] > gain_se[kind] and not flat[kind])
    return gain, gain_se, flat, eligible


def _auto_terms(cv_rloss, loss_rows, constant_loss, mstops, sample_weight=None):
    """The candidate that ``terms="auto"`` returns, and how it was chosen.

    A candidate is eligible when its out-of-fold R-loss gain over the constant
    effect exceeds the standard error of that gain, and its tuned step count is
    not below ``_CLOSE_TO_CONSTANT_STEPS``. Those are the two checks that make
    :func:`effect_model` warn that a fit is no better than a constant. Among the
    eligible candidates the lowest R-loss is taken, with the joint margin of
    :func:`_choose_candidate`. The smooth candidate replaces that choice when
    the chosen candidate beats it by no more than one standard error of the
    loss difference, since such a tie does not show that the extra flexibility
    helps. When no candidate is eligible, the smooth candidate is returned and
    the constant-effect warning names it.

    Standard errors count the observation weights ``sample_weight`` as
    frequencies; ``None`` means unit weights.

    Returns the candidate and a record of the gains, standard errors, flags and
    the rule that decided the choice.
    """
    gain, gain_se, flat, eligible = _candidate_gains(loss_rows, constant_loss, mstops,
                                                     sample_weight)
    record = {
        "lowest_loss": _choose_candidate(cv_rloss),
        "gain_over_constant": gain, "gain_se": gain_se, "flat": flat,
        "eligible": eligible, "tie_with_smooth": False,
    }
    pool = {kind: cv_rloss[kind] for kind in cv_rloss if eligible.get(kind)}
    if not pool:
        record["rule"] = (f"no candidate beats the constant effect by more than its standard "
                          f"error, so {_AUTO_FALLBACK!r} is returned")
        return _AUTO_FALLBACK, record
    chosen = _choose_candidate(pool)
    if chosen != _AUTO_FALLBACK and _AUTO_FALLBACK in pool:
        # Positive when the chosen candidate has the lower loss.
        advantage, se = _gain_and_se(loss_rows[_AUTO_FALLBACK] - loss_rows[chosen],
                                     sample_weight)
        if advantage <= se:
            record["tie_with_smooth"] = True
            chosen = _AUTO_FALLBACK
    if record["tie_with_smooth"]:
        record["rule"] = (f"{record['lowest_loss']!r} had the lowest R-loss, but it does not "
                          f"beat {_AUTO_FALLBACK!r} by more than one standard error, so "
                          f"{_AUTO_FALLBACK!r} is returned")
    elif chosen != record["lowest_loss"]:
        record["rule"] = (f"{record['lowest_loss']!r} had the lowest R-loss among all "
                          f"candidates, but it is not eligible; {chosen!r} had the lowest "
                          "R-loss among the eligible candidates")
    else:
        record["rule"] = f"{chosen!r} had the lowest R-loss among the eligible candidates"
    return chosen, record


def _nuisance_rhs(spec, values, df):
    """The smooth dictionary of the outcome and propensity nuisances."""
    return " + ".join(
        f"bbs({quote_name(name)}, df={df:g})" if len(np.unique(values[name])) >= _SMOOTH_MIN_VALUES
        else f"bols({quote_name(name)})"
        for name in spec.covariates)


#: The values of ``propensity_learner`` in :func:`aipw` and :func:`effect_model`.
_PROPENSITY_LEARNERS = ("auto", "smooth", "smooth+trees")


def _check_propensity_learner(value):
    if value not in _PROPENSITY_LEARNERS:
        raise ValueError(
            "propensity_learner must be 'auto', 'smooth' or 'smooth+trees'; "
            f"got {value!r}")
    return value


def _propensity_candidates(learner, base, values, covariates):
    """The propensity formulas that a ``propensity_learner`` may choose from.

    ``"smooth"`` is the base formula. ``"smooth+trees"`` is the base plus
    ``btree(name, maxdepth=1)`` for each numeric covariate with more than two
    distinct values: a depth-one tree makes one split, so it can place a step
    that a spline spreads over several basis functions. ``"auto"`` offers both.
    Without a numeric covariate there is no tree candidate, and the base is the
    only one. Returns ``{label: right-hand side}`` in the order the labels are
    compared, the base first."""
    names = [name for name in covariates
             if values[name].dtype != object and len(np.unique(values[name])) > 2]
    candidates = {}
    if learner in ("auto", "smooth"):
        candidates["smooth"] = base
    if learner in ("auto", "smooth+trees") and names:
        trees = " + ".join(f"btree({quote_name(name)}, maxdepth=1)" for name in names)
        candidates["smooth+trees"] = f"{base} + {trees}"
    if not candidates:
        candidates["smooth"] = base
    return candidates


def _propensity_record(requested, custom, chosen, deviance, runs):
    """The ``diagnostics["propensity_learner"]`` entry: the requested learner,
    the candidate chosen, each candidate's out-of-fold binomial deviance (lower
    is better) and each candidate's formula."""
    return {
        "requested": "custom" if custom else requested,
        "chosen": chosen,
        "candidates": list(runs),
        "out_of_fold_deviance": {label: deviance[label] for label in runs},
        "terms": {label: run["rhs"] for label, run in runs.items()},
    }


def _crossfit_means(values, spec, a, y, observed, sample_weight, fold_scheme, held, seed, df,
                    notes, extrapolation="warn", response_name=None,
                    propensity_learner="smooth", max_mstop=_MAX_MSTOP_DEFAULT):
    """Out-of-fold ``E[y | x]`` (fitted on observed rows), ``P(t = 1 | x)`` and,
    when ``response_name`` is given, ``P(observed | x, t)``, each tuned by inner
    cross-validation. Each nuisance runs inside its ``notes`` scope, so its
    warnings are issued by the caller under its own name."""
    from mboostrs import _rust
    from mboostrs.explore import tune_mstop
    from mboostrs.families import Binomial, Gaussian

    rhs = _nuisance_rhs(spec, values, df)
    outcome_family = Binomial() if spec.outcome_type == "binary" else Gaussian()
    m_hat, e_hat = np.empty(len(a)), np.empty(len(a))
    pi_hat = np.ones(len(a))
    # Pass 1: each propensity candidate is tuned and predicted on every fold,
    # on the same folds and seeds, and the one with the lower out-of-fold
    # binomial deviance is kept. Each candidate has its own notes, and only
    # the kept candidate's warnings are reported.
    runs = {}
    for label, prop_rhs in _propensity_candidates(
            propensity_learner, rhs, values, spec.covariates).items():
        candidate_notes = _NuisanceNotes(notes.owner, notes.flat_tolerance)
        e_candidate = np.empty(len(a))
        steps = []
        for b in range(fold_scheme.n_folds):
            test = held[:, b]
            train = np.asarray(fold_scheme.weights[:, b]) > 0
            train_data = {name: values[name][train] for name in spec.covariates}
            test_data = {name: values[name][test] for name in spec.covariates}
            # tune_mstop's flat-curve check is off here; the notes check it under
            # each nuisance's name and with the caller's flat_tolerance.
            with candidate_notes.scope("propensity"):
                propensity = tune_mstop(f"{spec.treatment} ~ {prop_rhs}",
                                        {**train_data, spec.treatment: a[train]},
                                        Binomial(), folds=5, seed=seed + b,
                                        weights=sample_weight[train], max_mstop=max_mstop,
                                        extrapolation=extrapolation, flat_tolerance=None)
                candidate_notes.observe_tuning("propensity", propensity)
                e_candidate[test] = _nuisance_predict(propensity.model, test_data,
                                                      extrapolation)
            steps.append(propensity.mstop)
        runs[label] = {"rhs": prop_rhs, "e": e_candidate, "stops": steps,
                       "notes": candidate_notes}
    deviance = {label: float(_rust.binomial_deviance(a, run["e"], sample_weight))
                for label, run in runs.items()}
    chosen = min(deviance, key=deviance.get)
    notes.absorb(runs[chosen]["notes"])
    e_hat[:] = runs[chosen]["e"]
    # Pass 2: the outcome model, and the response model when outcomes are
    # missing, on the same folds.
    stops = []
    for b in range(fold_scheme.n_folds):
        test = held[:, b]
        train = np.asarray(fold_scheme.weights[:, b]) > 0
        train_data = {name: values[name][train] for name in spec.covariates}
        test_data = {name: values[name][test] for name in spec.covariates}
        arm_train = train & (observed == 1)
        arm_data = {name: values[name][arm_train] for name in spec.covariates}
        # tune_mstop's flat-curve check is off here; the notes check it under
        # each nuisance's name and with the caller's flat_tolerance.
        with notes.scope("outcome"):
            outcome = tune_mstop(f"{spec.outcome} ~ {rhs}",
                                 {**arm_data, spec.outcome: y[arm_train]},
                                 outcome_family, folds=5, seed=seed + b,
                                 weights=sample_weight[arm_train], max_mstop=max_mstop,
                                 extrapolation=extrapolation, flat_tolerance=None)
            notes.observe_tuning("outcome", outcome)
            m_hat[test] = _nuisance_predict(outcome.model, test_data, extrapolation)
        entry = {"outcome": outcome.mstop, "propensity": runs[chosen]["stops"][b]}
        if response_name is not None:
            with notes.scope("response"):
                response = tune_mstop(
                    f"{quote_response(response_name)} ~ {rhs} + {spec.treatment}",
                    {**train_data, spec.treatment: a[train], response_name: observed[train]},
                    Binomial(), folds=5, seed=seed + b + 1, weights=sample_weight[train],
                    max_mstop=max_mstop, extrapolation=extrapolation, flat_tolerance=None)
                notes.observe_tuning("response", response)
                pi_hat[test] = _nuisance_predict(
                    response.model, {**test_data, spec.treatment: a[test]}, extrapolation)
            entry["response"] = response.mstop
        stops.append(entry)
    record = _propensity_record(propensity_learner, False, chosen, deviance, runs)
    return m_hat, e_hat, pi_hat, stops, record


def _scale_plan(values, modifiers):
    """How each smooth modifier is entered: on the log scale when it is a
    heavy-tailed positive column, and over its central range.

    The plan is computed from the covariates of the fitting rows only; the
    outcome does not enter it. Only numeric modifiers with enough distinct
    values, the ones that receive a spline basis, have a plan."""
    from mboostrs import _rust

    plan = {}
    for name in modifiers:
        column = np.asarray(values[name])
        if column.dtype.kind == "O" or len(np.unique(column)) < _SMOOTH_MIN_VALUES:
            continue
        log_scale, lower, upper = _rust.modifier_scale(
            np.ascontiguousarray(column, dtype=np.float64), _SMOOTH_TRIM_TAIL, _SKEW_RATIO)
        plan[name] = {"log": bool(log_scale), "lower": float(lower), "upper": float(upper)}
    return plan


def _apply_plan(values, plan):
    """The modifier columns on the scale the smooth second stage uses. A column
    without a plan is returned as it is; a log-scale column must be positive."""
    out = dict(values)
    for name, entry in plan.items():
        column = np.asarray(values[name], dtype=np.float64)
        if entry["log"]:
            if np.any(column <= 0):
                raise ValueError(
                    f"{name!r} is entered on the log scale because it is positive in the "
                    "fitting rows, and the new rows contain zero or negative values; "
                    "drop or recode them before predicting")
            column = np.log(column)
        out[name] = column
    return out


def _inside_plan(values, plan):
    """Rows whose plan modifiers lie in their central range. `values` must
    already be on the plan's scale."""
    rows = np.ones(len(next(iter(values.values()))), dtype=bool)
    for name, entry in plan.items():
        rows &= (values[name] >= entry["lower"]) & (values[name] <= entry["upper"])
    return rows


def _smooth_rhs(modifiers, values, df, plan=None):
    """The additive dictionary of the penalized second stage: a basis with
    REML-chosen smoothing for each modifier with enough distinct values, a
    linear term for the rest. A planned modifier's basis spans its central
    range, so its equally spaced knots fall over the range the rows cover."""
    terms = []
    for name in modifiers:
        if len(np.unique(values[name])) < _SMOOTH_MIN_VALUES:
            terms.append(f"bols({quote_name(name)})")
        elif plan and name in plan:
            lo, hi = plan[name]["lower"], plan[name]["upper"]
            terms.append(f"bbs({quote_name(name)}, df={df:g}, boundary.knots=c({lo:.17g},{hi:.17g}))")
        else:
            terms.append(f"bbs({quote_name(name)}, df={df:g})")
    return " + ".join(terms)


def _fit_tau(kind, modifiers, values, response, z, w, rows, df, mstop, plan=None):
    """``tau`` fitted on `rows`: a penalized additive model with REML smoothing
    for ``"smooth"``, otherwise a boosted model run for `mstop` steps.

    For ``"smooth"`` with a `plan`, `values` are on the plan's scale and the
    rows outside the planned central ranges are left out of the fit."""
    import mboostrs as mb
    from mboostrs.families import Gaussian
    from mboostrs.selection import gam_refit

    if kind == "smooth" and plan:
        rows = rows & _inside_plan(values, plan)
    data = {name: values[name][rows] for name in modifiers}
    data[response] = z[rows]
    if kind == "smooth":
        # gam_refit reads the designs and penalties of the boosted fit, not its
        # coefficients, so a single boosting step is enough to build the refit.
        model = mb.gamboost(f"{quote_response(response)} ~ {_smooth_rhs(modifiers, values, df, plan)}",
                            data=data, family=Gaussian(), weights=w[rows],
                            control=mb.boost_control(1, .1))
        return gam_refit(model, which="all")
    model = mb.gamboost(f"{quote_response(response)} ~ {_rhs_for(modifiers, values, kind)}",
                        data=data, family=Gaussian(), weights=w[rows],
                        control=mb.boost_control(mstop, .1))
    return model


def _oof_predicted(kind, modifiers, values, response, z, w, scheme, held, df, mstop, plan=None):
    """Out-of-fold predictions of ``tau`` for one candidate.

    Each candidate is refitted on the training rows of every fold and scored
    on that fold's held rows, so the candidates share one fold partition. A
    boosted candidate uses the step count tuned on the same folds; the smooth
    candidate has no tuned quantity. A smooth candidate with a `plan` holds
    held-out modifiers at the edge of their central range, as the final fit
    does."""
    predicted = np.empty(z.size)
    extrapolation = {"extrapolation": "clip"} if kind == "smooth" and plan else {}
    for b in range(scheme.n_folds):
        train = np.asarray(scheme.weights[:, b]) > 0
        test = held[:, b]
        model = _fit_tau(kind, modifiers, values, response, z, w, train, df, mstop, plan)
        predicted[test] = np.asarray(
            model.predict(newdata={name: values[name][test] for name in modifiers},
                          **extrapolation),
            dtype=np.float64).ravel()
    return predicted


def _oof_constant(z, w, scheme, held):
    """Out-of-fold predictions of the constant effect.

    On each fold's training rows the constant that minimizes the R-loss
    ``sum w (z - tau)^2`` is the weighted mean ``sum w z / sum w``, an
    overlap-tilted average of the pseudo-outcome rather than the plain ATE
    (it differs from ``ate_plugin`` when overlap is uneven). It is scored on
    the fold's held rows, like the candidates."""
    predicted = np.empty(z.size)
    for b in range(scheme.n_folds):
        train = np.asarray(scheme.weights[:, b]) > 0
        test = held[:, b]
        predicted[test] = float(np.sum(w[train] * z[train]) / np.sum(w[train]))
    return predicted


def _loss_rows(z, w, predicted):
    """Per-row R-loss ``w_i (z_i - tau_i)^2``; its mean is the candidate's score."""
    return w * (z - predicted) ** 2


#: A boosted candidate tuned to fewer steps than this is close to a constant:
#: its first few selections fit little of the pseudo-outcome's variation.
_CLOSE_TO_CONSTANT_STEPS = 5

#: The pseudo-outcome choices of :func:`effect_model`'s ``learner`` argument.
_LEARNERS = ("r_winsorized", "r")

#: Share of positive-weight rows clipped in each tail of the pseudo-outcome by
#: ``learner="r_winsorized"``; the bounds are the 1st and 99th percentiles.
_PSEUDO_TAIL = 0.01

#: A fitted effect whose absolute value exceeds this multiple of the outcome's
#: interquartile range triggers a warning.
_EFFECT_RANGE_MULTIPLE = 10.0

#: Share of fitting rows trimmed from each tail of a smooth modifier. The smooth
#: second stage is fitted over the central range (1st to 99th percentile), and its
#: predictions are held at the range edge outside it.
_SMOOTH_TRIM_TAIL = 0.01

#: A positive modifier whose 99th percentile is at least this multiple of its
#: median is entered on the log scale by the smooth second stage.
_SKEW_RATIO = 3.0


#: The ways ``effect_model`` can build a per-row interval; see its ``interval_method``.
_INTERVAL_METHODS = ("auto", "refit", "repeats", "bootstrap", "second_stage_bootstrap",
                     "dr_orthogonal")

#: Seed step between the splits of repeated fits, the same step :func:`aipw` uses.
_REPEAT_SEED_STEP = 104729


def _interval_refits(method, count, spec, *, data, values, treatment, outcome, modifiers,
                     weights, folds, seed, df, propensity_clip, max_mstop, outcome_missing,
                     overlap_bounds, learner):
    """The refits behind a ``"repeats"`` or ``"bootstrap"`` interval.

    ``"repeats"`` refits the whole pipeline (nuisances, tuning and second stage)
    with split seeds ``seed + 104729 * r`` for ``r = 1 .. count - 1``; the
    primary fit is repeat 0 and is not refitted here. ``"bootstrap"`` draws
    ``count`` resamples of the rows with replacement and refits the pipeline on
    each, with the same seed rule. Every refit uses the smooth second stage, so
    each one has a coefficient covariance. The refits run with their warnings
    suppressed; the primary fit reports the warnings of the call.
    """
    # Each refit reports only its estimate and covariance-based standard error,
    # so it is fitted with interval_method="refit" and no second-stage bootstrap.
    kwargs = dict(terms="smooth", folds=folds, df=df, propensity_clip=propensity_clip,
                  interval_method="refit",
                  max_mstop=max_mstop, outcome_missing=outcome_missing,
                  overlap_bounds=overlap_bounds, learner=learner)
    fits = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "repeats":
            for r in range(1, count):
                fits.append(effect_model(
                    data, spec, modifiers=modifiers, weights=weights,
                    seed=seed + _REPEAT_SEED_STEP * r, **kwargs))
        else:
            rng = np.random.default_rng([seed, 2024])
            n = len(treatment)
            weight_array = None if weights is None else np.asarray(weights, dtype=np.float64)
            for b in range(count):
                rows = rng.integers(0, n, n)
                resample = {
                    spec.treatment: treatment[rows],
                    spec.outcome: outcome[rows],
                    **{name: values[name][rows] for name in spec.covariates},
                }
                fits.append(effect_model(
                    resample, spec, modifiers=modifiers,
                    weights=None if weight_array is None else weight_array[rows],
                    seed=seed + _REPEAT_SEED_STEP * (b + 1), **kwargs))
    return fits


def _level_belongs_to_predict(fn):
    """Reject ``level=`` with a message naming the ``predict`` method, where the
    interval level is chosen. The wrapped function keeps its signature."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "level" in kwargs:
            raise TypeError(
                f"{fn.__name__}() has no argument 'level'. 'level' is an argument of the "
                "predict method, which returns the interval at that level: "
                "fit.predict(newdata, interval=True, level=0.9)")
        return fn(*args, **kwargs)
    return wrapper


@_level_belongs_to_predict
@once_per_call
@one_summary
def effect_model(data, spec: CausalSpec, *, nuisance=None, modifiers=None, terms="auto",
                 folds=5, seed=42, df=4, propensity_clip=FROM_SPEC, max_mstop=None,
                 outcome_missing=FROM_SPEC, weights=None,
                 overlap_bounds=FROM_SPEC, learner="r_winsorized",
                 extrapolation="warn", flat_tolerance=0.01, propensity_learner="smooth",
                 interval_method="auto", interval_repeats=5, bootstrap_draws=200):
    """Estimate the conditional average treatment effect ``tau(x)``.

    The R-learner (Nie and Wager, 2021) fits ``tau`` by minimizing
    ``sum w (y - m(x) - (t - e(x)) tau(x))^2``, where ``m(x) = E[y | x]`` and
    ``e(x) = P(t = 1 | x)`` are predicted out of fold. This equals a weighted
    regression of the pseudo-outcome ``(y - m)/(t - e)`` with weights
    ``w (t - e)^2``. Because ``y - m`` and ``t - e`` are residuals, a
    confounder's effect on the outcome does not leak into ``tau`` when the
    nuisances are accurate.

    Intervals. With ``terms="smooth"`` the second stage is a penalized
    additive model (:func:`~mboostrs.selection.gam_refit`) whose smoothing is
    chosen by REML. The pseudo-outcome's error variance is proportional to
    ``1 / (t - e)^2``, so the weights ``(t - e)^2`` make the fit weighted least
    squares with the correct error variance, and the coefficient covariance
    is the one that weighting implies. ``predict(newdata, interval=True)``
    returns, per row, the estimate ``tau(x)``, its standard error from the
    full coefficient covariance (the covariance between the smooth terms is
    included) and a pointwise interval. The interval covers the sampling
    variation of the second stage given the cross-fitted nuisances: it treats
    ``m`` and ``e`` as fixed at their out-of-fold predictions. It does not
    cover error in those nuisances, the choice of modifiers or ``terms``, or
    the bias a penalized smooth carries where the true effect bends sharply.
    Its coverage is near nominal when the additive form holds; the test suite
    checks it on simulated effects. The ``"stumps"`` and ``"joint"``
    candidates are boosted fits, which have no coefficient covariance, so
    their per-row intervals are refused.

    ``interval_method="second_stage_bootstrap"`` keeps the pseudo-outcome and
    the nuisance predictions fixed and resamples the fitting rows with
    replacement ``bootstrap_draws`` times. Each resample is a multinomial draw
    of the rows, used as prior weights on the full-sample spline basis, and the
    second stage is refitted on it with its smoothing chosen by REML. The
    interval is the percentile interval of the draws, and the estimate is the
    full-sample fit. It captures the variation of the second stage, including
    its smoothing choice, at the cost of one penalized refit per draw; it does
    not refit the nuisances, so it accepts ``nuisance=``. ``"auto"``, the
    default, is this method for a smooth second stage and ``"refit"`` for the
    boosted candidates.

    ``interval_method="repeats"`` refits the nuisances and the second stage on
    ``interval_repeats`` split seeds (the primary fit is one of them). At each
    row it takes the estimate and standard error from every fit and combines
    them by the median rule of Chernozhukov et al. (2018): the estimate is the
    median over fits, and the variance is the median of ``se_r^2 + (tau_r -
    estimate)^2``. The second term adds the split-to-split spread to each fit's
    own sampling variance. ``interval_method="bootstrap"`` resamples the rows
    with replacement ``bootstrap_draws`` times, refits the whole pipeline on
    each resample, and reports the full-sample estimate with the percentile
    interval of the refitted effects. Both methods refit the nuisances, so they
    refuse ``nuisance=``, and both need the smooth second stage. Neither removes
    the bias of the nuisances or of the penalized smooth.

    Checks. The constant effect, ``tau`` equal to the ATE, is scored on the same
    folds (``constant_rloss``; ``diagnostics["constant_effect"]`` gives its
    difference from the chosen candidate and that difference's standard error).
    A warning is issued when the chosen candidate does not beat the constant by
    more than its standard error, and when a boosted candidate is tuned to fewer
    than five steps, which makes it close to a constant. A warning is also issued
    when the largest absolute fitted effect over the fitting rows exceeds 10 times
    the interquartile range of the observed outcome. Extrapolation warnings from
    the fold fits are reported once for the call.

    Bounded second stage. The R-learner pseudo-outcome ``(y - m) / (t - e)``
    has no bound: where ``|t - e|`` is small it is a large multiple of the
    residual. With ``learner="r_winsorized"`` (the default) it is clipped to its
    1st and 99th percentiles over the rows with positive weight before the
    second stage, so that no row's pseudo-outcome lies beyond those percentiles.
    The clipping does not drop rows, and the bounds are reported in
    ``diagnostics["pseudo_bounds"]``. Clipping also shrinks the effects of the rows
    in the tails when those tails are not heavy: on a design with light tails the
    per-row 90% coverage falls by one to two points. ``learner="r"`` keeps the
    unbounded pseudo-outcome.

    Bounded modifiers. The smooth candidate fits each modifier's spline over its
    central range, the 1st and 99th percentiles of the fitting rows, so its
    equally spaced knots fall over the rows and not over a few extreme values.
    Rows outside a range are left out of the fit, and predictions outside it are
    held at the range edge (``extrapolation="clip"``, the default of
    :meth:`EffectModel.predict` for ``terms="smooth"``). A positive modifier whose
    99th percentile is at least 3 times its median is entered as its logarithm,
    which spreads its long right tail over the basis. The chosen scales and ranges
    are in ``diagnostics["smooth_scale"]``. The boosted candidates use the raw
    columns.

    Parameters
    ----------
    data : mapping or data frame
        Outcome, treatment and covariate columns. Covariates may be categorical
        (strings or factors); they are read as factors, as the boosting formulas
        read them. A missing categorical value is an error.
    spec : CausalSpec
        The treatment, outcome, covariates and assumptions.
    nuisance : AIPWResult, optional
        Reuse the cross-fitted propensity, outcome and response models of
        :func:`aipw` on the same rows (``m = e mu1 + (1 - e) mu0``). By default
        they are cross-fitted here with smooth terms and a tuned step count.
        The result's ``outcome_missing`` and weights must match these arguments.
    modifiers : sequence of str, optional
        The covariates ``tau`` may depend on; all of ``spec.covariates`` by
        default.
    terms : {"auto", "smooth", "stumps", "joint"}
        ``"smooth"`` gives each modifier a penalized smooth (a linear term for
        a modifier with fewer than ten distinct values), with REML smoothing
        and per-row intervals; ``"stumps"`` a one-split tree for effects that
        jump at a threshold; ``"joint"`` one depth-3 tree that may split on any
        modifier. The joint tree is the only candidate that can represent an
        effect of two modifiers at once, such as ``1 + 1.5 * (x0 * x1 > 0)``,
        for which each modifier's own effect is flat; it selects its split by
        squared error because a modifier whose joint effect is strong can have
        a marginal association near zero. ``"auto"`` scores every applicable
        candidate by its out-of-fold R-loss, on the same folds, and then
        restricts the choice to the candidates that beat a constant effect by
        more than their standard error and whose tuned step count is not close
        to zero (a boosted fit that stops within 5 steps is close to a
        constant). Among those it keeps the lowest R-loss, except that the joint
        tree must be at least 5 percent lower than the best additive candidate
        to be preferred. A choice that beats ``"smooth"`` by no more than one
        standard error of their loss difference is replaced by ``"smooth"``. If
        no candidate is eligible, ``"smooth"`` is returned and the constant-effect
        warning says so. ``diagnostics["terms_selection"]`` gives the gains, the
        flags and the rule that decided the choice, and :meth:`EffectModel.describe`
        states it. The candidates are not mixed in one model. The joint candidate
        needs two or more modifiers.
    folds : int
        Outer cross-fitting folds for the nuisances and the folds that score
        and tune ``tau``, stratified by treatment.
    df : float
        The ``df`` of the boosted smooths in the nuisance models. The smooth
        candidate's basis and smoothing come from the REML refit, which does
        not read ``df``.
    outcome_missing : {"error", "ipw"}, optional
        When the call does not pass it, the ``outcome_missing`` of the
        :class:`CausalSpec` is used, which defaults to ``"error"``: an observed
        outcome is required in every row. ``"ipw"`` treats the outcome as
        missing at random given the covariates and treatment: the outcome mean
        is fitted on observed rows, and the R-loss weight of each row is
        multiplied by ``R / pi``, where ``pi`` is the cross-fitted response
        propensity. The effect is then estimated for all rows, as in
        :func:`aipw`.
    weights : array-like, optional
        Observation weights read as frequencies. They enter the nuisance fits
        and the R-loss.
    propensity_clip : tuple or None, optional
        Bounds for ``e(x)`` and, under ``outcome_missing="ipw"``, for the
        response propensity ``pi``. When the call does not pass it, the
        ``propensity_clip`` of the :class:`CausalSpec` is used, which defaults
        to ``(0.01, 0.99)``, the same as :func:`aipw`, so the two estimators
        weight rows alike. An argument that is passed always wins. The count of
        clipped rows is reported. ``None`` disables clipping.
    extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
        The policy for the cross-fitted nuisance predictions of a row outside a
        spline's training range, as in :func:`aipw`. The warning names the
        nuisance model and this argument.
    flat_tolerance : float or None, default 0.01
        The relative gap at which a tuned nuisance's cross-validated curve
        counts as flat (see :func:`aipw`). ``None`` turns that check off. The
        second stage's own tuning keeps the default check of
        :func:`~mboostrs.explore.tune_mstop`.
    max_mstop : int, optional
        The largest step count searched by every boosted fit: the outer
        cross-validated tuning of each nuisance model and the tuning of a boosted
        ``tau`` candidate. ``None`` (the default) is 1600, the cap that
        :func:`aipw` uses. Before 1600 was the default the cap was 10,000, and a
        propensity model reached it in every fold, so the nuisance tuning took
        most of the fit time.
    overlap_bounds : tuple
        Propensity bounds ``(lower, upper)`` that define the overlap region.
        Rows outside them are reported in ``diagnostics["overlap"]`` and, for
        new rows, flagged by :meth:`EffectModel.predict`.
    learner : {"r_winsorized", "r"}
        The pseudo-outcome the second stage fits. ``"r_winsorized"`` (the
        default) clips it as described above. ``"r"`` uses the unbounded
        pseudo-outcome; where overlap is thin, a flexible second stage can
        then place effects far outside the outcome's range.
    propensity_learner : {"smooth", "auto", "smooth+trees"}
        Which propensity formula is fitted. ``"smooth"``, the default, fits the
        base formula only. ``"auto"`` compares the base formula with the base plus
        a depth-one tree per numeric covariate, on the same folds, and keeps the
        one with the lower out-of-fold binomial deviance. ``"smooth+trees"`` fits
        the tree formula only. The formulas and the deviances are in
        ``diagnostics["propensity_learner"]``.
        The base formula of :func:`aipw` is the one its ``smooth`` argument
        builds, linear by default; here it is the spline dictionary.
    interval_method : {"auto", "refit", "second_stage_bootstrap", "repeats", "bootstrap"}
        How ``predict(..., interval=True)`` builds the per-row interval. ``"auto"``,
        the default, is ``"second_stage_bootstrap"`` for a smooth second stage and
        ``"refit"`` for the boosted candidates, as described under Intervals.
        ``"repeats"`` and ``"bootstrap"`` refit the whole pipeline and need the
        smooth second stage.
    interval_repeats : int
        With ``interval_method="repeats"``, the number of split seeds, the primary
        fit included. The fit costs that many times the single fit. Ignored otherwise.
    bootstrap_draws : int
        With ``interval_method="bootstrap"``, the number of row resamples, each
        refitted in full; with ``"second_stage_bootstrap"``, the number of
        resamples of the second stage alone. Ignored otherwise.

    Returns
    -------
    EffectModel
    """
    from mboostrs import _rust
    from mboostrs.explore import tune_mstop
    from mboostrs.families import Binomial, Gaussian

    if not isinstance(spec, CausalSpec):
        raise TypeError("spec must be a CausalSpec")
    if terms not in ("auto", "smooth", "stumps", "joint"):
        raise ValueError("terms must be 'auto', 'smooth', 'stumps' or 'joint'")
    if learner not in _LEARNERS:
        raise ValueError(f"learner must be 'r_winsorized' or 'r'; got {learner!r}")
    if max_mstop is None:
        max_mstop = _MAX_MSTOP_DEFAULT
    max_mstop = check_int_at_least("max_mstop", max_mstop, 1, allow_none=False)
    folds = check_int_at_least("folds", folds, 2, allow_none=False)
    df = check_positive("df", df)
    propensity_clip = _check_clip(_spec_default(spec, "propensity_clip", propensity_clip))
    outcome_missing = _check_outcome_missing(
        _spec_default(spec, "outcome_missing", outcome_missing))
    overlap_bounds = _check_bounds(_spec_default(spec, "overlap_bounds", overlap_bounds))
    _check_extrapolation(extrapolation)
    flat_tolerance = _check_flat_tolerance(flat_tolerance)
    modifiers = tuple(spec.covariates if modifiers is None else modifiers)
    unknown = set(modifiers) - set(spec.covariates)
    if not modifiers or unknown:
        raise ValueError(f"modifiers must be a non-empty subset of spec.covariates; "
                         f"not covariates: {sorted(unknown)}")
    if terms == "joint" and len(modifiers) < 2:
        raise ValueError("a joint tree needs at least two modifiers")
    if interval_method not in _INTERVAL_METHODS:
        raise ValueError(
            f"interval_method must be one of {', '.join(_INTERVAL_METHODS)}; got "
            f"{interval_method!r}")
    _check_propensity_learner(propensity_learner)
    if nuisance is not None and propensity_learner != "smooth":
        raise ValueError(
            f"propensity_learner={propensity_learner!r} chooses the propensity model that "
            "effect_model fits, and nuisance= supplies a fitted one. Pass nuisance=None, or "
            "leave propensity_learner at 'smooth' to keep the aipw result's choice")
    interval_repeats = check_int_at_least("interval_repeats", interval_repeats, 2,
                                          allow_none=False)
    bootstrap_draws = check_int_at_least("bootstrap_draws", bootstrap_draws, 2,
                                         allow_none=False)
    if interval_method == "auto" and terms in ("stumps", "joint"):
        interval_method = "refit"
    if interval_method not in ("auto", "refit", "dr_orthogonal"):
        if terms in ("stumps", "joint"):
            raise ValueError(
                f"interval_method={interval_method!r} needs the smooth second stage, and "
                f"terms={terms!r} is a boosted model with no coefficient covariance. Pass "
                "terms='smooth' or terms='auto'")
        if interval_method in ("repeats", "bootstrap") and isinstance(nuisance, AIPWResult):
            raise ValueError(
                f"interval_method={interval_method!r} refits the nuisance models on each "
                "split, so it cannot reuse nuisance=; pass nuisance=None")
    values = _columns(data, spec, outcome_missing=outcome_missing, categorical=True,
                      owner="effect_model")
    control_label, treated_label = spec.treatment_contrast
    raw_a = np.asarray(data[spec.treatment])
    if not np.all(np.isin(raw_a, (control_label, treated_label))):
        raise ValueError("treatment contains values outside treatment_contrast")
    a = (raw_a == treated_label).astype(float)
    n = len(a)
    if a.sum() == 0 or a.sum() == n:
        raise ValueError("both treatment levels must be observed")
    y = values[spec.outcome]
    observed = (~np.isnan(y)).astype(float)
    use_response = outcome_missing == "ipw" and bool(np.any(observed == 0))
    sample_weight_given = weights is not None or (
        isinstance(nuisance, AIPWResult) and nuisance.weights is not None)
    sample_weight = _sample_weights(weights, n)
    notes = _NuisanceNotes("effect_model", flat_tolerance)

    if nuisance is None:
        fold_scheme, held = _crossfit_folds(a, folds, seed)
        m_hat, e_hat, pi_hat, stops, propensity_record = _crossfit_means(
            values, spec, a, y, observed, sample_weight, fold_scheme, held, seed, df,
            notes, extrapolation=extrapolation,
            response_name=_response_name(spec) if use_response else None,
            propensity_learner=propensity_learner, max_mstop=max_mstop)
        notes.emit()
        propensity_formula = propensity_record["terms"][propensity_record["chosen"]]
        source = f"{fold_scheme.n_folds}-fold cross-fitting with smooth terms"
    elif isinstance(nuisance, AIPWResult):
        if nuisance.spec != spec:
            raise ValueError("nuisance was estimated for a different CausalSpec")
        if not (np.array_equal(nuisance.treatment, a)
                and np.array_equal(nuisance.outcome, y, equal_nan=True)):
            raise ValueError("nuisance was estimated on different rows")
        if nuisance.nuisance.get("outcome_missing", "error") != outcome_missing:
            raise ValueError(
                "nuisance was estimated with outcome_missing="
                f"{nuisance.nuisance.get('outcome_missing', 'error')!r}; pass the same option")
        if nuisance.weights is not None:
            if weights is None:
                sample_weight = np.asarray(nuisance.weights, dtype=np.float64)
            elif not np.array_equal(sample_weight, nuisance.weights):
                raise ValueError("nuisance was estimated with different weights")
        elif weights is not None:
            raise ValueError("nuisance was estimated without weights")
        e_hat = np.asarray(nuisance.propensity, dtype=np.float64)
        m_hat = e_hat * nuisance.mu1 + (1 - e_hat) * nuisance.mu0
        pi_hat = (np.ones(n) if nuisance.response_propensity is None
                  else np.asarray(nuisance.response_propensity, dtype=np.float64))
        stops = nuisance.nuisance.get("selected_mstop")
        # The overlap model below keeps the smooth dictionary; the aipw
        # result's propensity choice is recorded, not refitted.
        propensity_record = nuisance.diagnostics.get("propensity_learner")
        propensity_formula = None
        source = "the supplied aipw result"
        unique_folds = np.unique(nuisance.fold_id)
        held = (np.column_stack([nuisance.fold_id == f for f in unique_folds])
                if unique_folds.size >= 2 else np.ones((n, 1), dtype=bool))
    else:
        raise TypeError("nuisance must be None or an AIPWResult")

    # Without a clip bound the R-loss uses the propensity as given; the
    # response bound applies only when a response model was fitted, since
    # its complete-case value 1 would otherwise be pulled to the bound.
    clip = propensity_clip if propensity_clip is not None else (0.0, 1.0)
    response_clip = propensity_clip if use_response else None
    pseudo_bounds, n_winsorized = None, 0
    if interval_method == "dr_orthogonal":
        clip_base = propensity_clip if propensity_clip is not None else (0.01, 0.99)
        stab_clip = (max(0.01, clip_base[0]), min(0.99, clip_base[1]))
        if isinstance(nuisance, AIPWResult):
            e_stab = np.clip(e_hat, stab_clip[0], stab_clip[1])
            pi_stab = np.clip(pi_hat, stab_clip[0], stab_clip[1]) if use_response else np.ones(n)
            mu_t = np.where(a == 1.0, nuisance.mu1, nuisance.mu0)
            y_obs = np.where(observed == 1.0, y, 0.0)
            obs_factor = (observed / pi_stab) if use_response else 1.0
            pseudo = ((nuisance.mu1 - nuisance.mu0)
                      + obs_factor * ((a - e_stab) / (e_stab * (1.0 - e_stab))) * (y_obs - mu_t))
            weights_r = sample_weight.copy()
            n_clipped = int(np.sum((e_hat < stab_clip[0]) | (e_hat > stab_clip[1])))
            n_clipped_response = (int(np.sum((pi_hat < stab_clip[0]) | (pi_hat > stab_clip[1])))
                                  if use_response else 0)
        else:
            pseudo, weights_r, n_clipped, n_clipped_response = _rust.rloss_transform_weighted(
                np.ascontiguousarray(y, dtype=np.float64), a,
                np.ascontiguousarray(m_hat), np.ascontiguousarray(e_hat), stab_clip,
                sample_weight, observed, np.ascontiguousarray(pi_hat),
                stab_clip if use_response else None,
            )
            pseudo = np.asarray(pseudo, dtype=np.float64)
            weights_r = np.asarray(weights_r, dtype=np.float64)
    else:
        pseudo, weights_r, n_clipped, n_clipped_response = _rust.rloss_transform_weighted(
            np.ascontiguousarray(y, dtype=np.float64), a,
            np.ascontiguousarray(m_hat), np.ascontiguousarray(e_hat), clip,
            sample_weight, observed, np.ascontiguousarray(pi_hat), response_clip,
        )
        pseudo = np.asarray(pseudo, dtype=np.float64)
        weights_r = np.asarray(weights_r, dtype=np.float64)
        if learner == "r_winsorized":
            pseudo, lower_bound, upper_bound, n_winsorized = _rust.winsorize_pseudo_outcome(
                pseudo, weights_r, _PSEUDO_TAIL)
            pseudo = np.asarray(pseudo, dtype=np.float64)
            pseudo_bounds = (float(lower_bound), float(upper_bound))
    response = "tau_pseudo_outcome"
    while response in modifiers:
        response = "_" + response
    fit_data = {name: values[name] for name in modifiers}
    fit_data[response] = pseudo
    tau_scheme, tau_held = _crossfit_folds(a, folds, seed + 1)
    candidates = ("smooth", "stumps", "joint") if terms == "auto" else (terms,)
    if terms == "auto" and len(modifiers) < 2:
        candidates = ("smooth", "stumps")
    # The smooth candidate enters its heavy-tailed positive modifiers on the log
    # scale and spans each modifier over its central range; the boosted
    # candidates keep the raw columns.
    scale_plan = _scale_plan(values, modifiers)
    smooth_values = _apply_plan(values, scale_plan)
    tuned, cv_rloss, cv_predicted, loss_rows = {}, {}, {}, {}
    for kind in candidates:
        mstop = None
        if kind != "smooth":
            tuning = tune_mstop(f"{quote_response(response)} ~ {_rhs_for(modifiers, values, kind)}", fit_data,
                                Gaussian(), folds=tau_scheme, weights=weights_r,
                                max_mstop=max_mstop, seed=seed)
            tuned[kind] = tuning
            mstop = tuning.mstop
        if kind == "smooth":
            cv_predicted[kind] = _oof_predicted(kind, modifiers, smooth_values, response,
                                                pseudo, weights_r, tau_scheme, tau_held, df,
                                                mstop, scale_plan)
        else:
            cv_predicted[kind] = _oof_predicted(kind, modifiers, values, response, pseudo,
                                                weights_r, tau_scheme, tau_held, df, mstop)
        loss_rows[kind] = _loss_rows(pseudo, weights_r, cv_predicted[kind])
        cv_rloss[kind] = float(np.mean(loss_rows[kind]))
    # The constant effect on the same folds, and the chosen candidate's margin
    # over it. The margin's standard error is over the rows' loss differences.
    constant_loss = _loss_rows(pseudo, weights_r,
                               _oof_constant(pseudo, weights_r, tau_scheme, tau_held))
    constant_rloss = float(np.mean(constant_loss))
    if terms == "auto":
        chosen, terms_selection = _auto_terms(
            cv_rloss, loss_rows, constant_loss, {k: t.mstop for k, t in tuned.items()},
            sample_weight)
    else:
        chosen = _choose_candidate(cv_rloss)
        # The gains of every candidate are reported for a named choice too, so the
        # record reads the same as terms="auto"; the choice itself is the one asked for.
        gains, gain_ses, flats, eligibles = _candidate_gains(
            loss_rows, constant_loss, {k: t.mstop for k, t in tuned.items()}, sample_weight)
        terms_selection = {"rule": "requested", "lowest_loss": chosen,
                           "gain_over_constant": gains, "gain_se": gain_ses,
                           "flat": flats, "eligible": eligibles, "tie_with_smooth": False}
    gain_rows = constant_loss - _loss_rows(pseudo, weights_r, cv_predicted[chosen])
    gain, gain_se = _gain_and_se(gain_rows, sample_weight)
    everything = np.ones(n, dtype=bool)
    if chosen == "smooth":
        model = _fit_tau("smooth", modifiers, smooth_values, response, pseudo, weights_r,
                         everything, df, None, scale_plan)
        tau = np.asarray(model.predict(newdata={name: smooth_values[name] for name in modifiers},
                                       extrapolation="clip"), dtype=np.float64).ravel()
    else:
        model = tuned[chosen].model
        tau = np.asarray(model.predict(newdata={name: values[name] for name in modifiers}),
                         dtype=np.float64).ravel()

    # The propensity of new rows, for the overlap flag in predict. Its step
    # count is the median of the cross-fitted ones, so it needs no new tuning.
    prop_steps = ([s["propensity"] for s in stops if isinstance(s, dict)]
                  if isinstance(stops, list) else [])
    prop_mstop = max(1, int(np.median(prop_steps))) if prop_steps else 100
    import mboostrs as mb

    propensity_model = mb.gamboost(
        f"{spec.treatment} ~ {propensity_formula or _nuisance_rhs(spec, values, df)}",
        data={**{name: values[name] for name in spec.covariates}, spec.treatment: a},
        family=Binomial(), weights=sample_weight,
        control=mb.boost_control(prop_mstop, .1))

    observed_outcome = y[observed == 1]
    outcome_iqr = (float(np.quantile(observed_outcome, 0.75) - np.quantile(observed_outcome, 0.25))
                   if observed_outcome.size else float("nan"))
    max_abs_tau = float(np.max(np.abs(tau)))
    beyond_range = bool(outcome_iqr > 0 and max_abs_tau > _EFFECT_RANGE_MULTIPLE * outcome_iqr)

    diagnostics = {
        "learner": learner,
        "pseudo_bounds": pseudo_bounds,
        "n_winsorized": int(n_winsorized),
        "smooth_scale": {name: dict(entry) for name, entry in scale_plan.items()},
        "smooth_rows_outside_range": int(n - np.count_nonzero(
            _inside_plan(smooth_values, scale_plan))) if scale_plan else 0,
        "effect_range": {
            "max_abs_effect": max_abs_tau,
            "outcome_iqr": outcome_iqr,
            "multiple": _EFFECT_RANGE_MULTIPLE,
            "beyond_range": beyond_range,
        },
        "propensity_clip": propensity_clip,
        "n_clipped": int(n_clipped),
        "propensity_min": float(e_hat.min()),
        "propensity_max": float(e_hat.max()),
        "outcome_missing": outcome_missing,
        "n_missing_outcome": int(n - observed.sum()),
        "response_model": ("fitted" if use_response
                           else "not fitted: no outcome is missing"),
        "response_propensity_min": float(pi_hat.min()) if use_response else None,
        "response_propensity_max": float(pi_hat.max()) if use_response else None,
        "n_clipped_response": int(n_clipped_response),
        "sample_weights": sample_weight_given,
        "nuisance_source": source,
        "nuisance_mstop": stops,
        "tau_fit": "penalized_reml" if chosen == "smooth" else "boosted",
        "tau_mstop": {k: t.mstop for k, t in tuned.items()},
        "tau_at_boundary": {k: t.at_boundary for k, t in tuned.items()},
        "candidates": list(candidates),
        "terms_selection": terms_selection,
        "max_mstop": max_mstop,
        "joint_depth": _JOINT_DEPTH,
        "effect_scale": ("risk_difference" if spec.outcome_type == "binary"
                         else "mean_difference"),
        "constant_effect": {
            "ate": float(np.sum(weights_r * pseudo) / np.sum(weights_r)),
            "rloss": constant_rloss,
            "gain": gain,
            "gain_se": gain_se,
            "beats_by_se": bool(gain > gain_se),
        },
    }
    if not diagnostics["constant_effect"]["beats_by_se"]:
        warnings.warn(
            f"terms={chosen!r} does not beat a constant effect by more than its standard "
            f"error. Out-of-fold R-loss {cv_rloss[chosen]:.6g} for {chosen!r} against "
            f"{constant_rloss:.6g} for the constant effect (the ATE), a difference of "
            f"{gain:.6g} with standard error {gain_se:.6g}. The modifiers add no reliable "
            "heterogeneity at this sample size: report the constant effect from aipw, or "
            "pass terms= to name a candidate explicitly. The R-loss comparison has low power "
            "for detecting effect modifiers, so a real but small modifier can fail it. To test "
            "who benefits more, use effect_modifiers(result, data): its BLP and GATES tests "
            "ask whether the average effect changes along each covariate.",
            UserWarning, stacklevel=3)
    if chosen in tuned and tuned[chosen].mstop < _CLOSE_TO_CONSTANT_STEPS:
        warnings.warn(
            f"the chosen {chosen!r} candidate was tuned to {tuned[chosen].mstop} boosting "
            f"steps, below {_CLOSE_TO_CONSTANT_STEPS}, so tau(x) is close to a constant. "
            "Read the estimate as the average effect (aipw gives it with an interval), not "
            "as a heterogeneous effect.",
            UserWarning, stacklevel=3)
    if diagnostics["effect_range"]["beyond_range"]:
        warnings.warn(
            f"the fitted effect reaches {max_abs_tau:.6g} in absolute value on the fitting "
            f"rows, more than {_EFFECT_RANGE_MULTIPLE:g} times the interquartile range of the "
            f"outcome ({outcome_iqr:.6g}). The second stage is extrapolating where the "
            "overlap is thin. Inspect the rows with the largest |tau(x)| in predict() "
            "output, check their propensity with propensity(), and report the constant "
            "effect from aipw() beside this one.",
            UserWarning, stacklevel=3)
    from functools import partial

    if interval_method == "auto":
        interval_method = _AUTO_INTERVAL_SMOOTH if chosen == "smooth" else "refit"
    interval_fits = ()
    second_stage = None
    dr_state = None
    if interval_method in ("repeats", "bootstrap", "second_stage_bootstrap"):
        if chosen != "smooth":
            raise ValueError(
                f"interval_method={interval_method!r} needs the smooth second stage, but this "
                f"fit chose terms={chosen!r}, a boosted model with no coefficient covariance. "
                "Pass terms='smooth' to get intervals from the refits")
    if interval_method in ("repeats", "bootstrap"):
        interval_fits = tuple(_interval_refits(
            interval_method, interval_repeats if interval_method == "repeats"
            else bootstrap_draws, spec, data=data, values=values, treatment=raw_a,
            outcome=y, modifiers=modifiers, weights=weights, folds=folds, seed=seed,
            df=df, propensity_clip=propensity_clip, max_mstop=max_mstop,
            outcome_missing=outcome_missing, overlap_bounds=overlap_bounds,
            learner=learner))
    elif interval_method == "second_stage_bootstrap":
        # The second stage's rows: the smooth fit leaves out rows outside the
        # central range of its planned modifiers, and the resamples draw from
        # the same rows.
        fit_rows = (_inside_plan(smooth_values, scale_plan) if scale_plan
                    else np.ones(n, dtype=bool))
        second_stage = _second_stage_bootstrap(
            model, pseudo, weights_r, fit_rows, bootstrap_draws, seed)
    elif interval_method == "dr_orthogonal":
        if chosen == "smooth":
            fit_rows = (_inside_plan(smooth_values, scale_plan) if scale_plan
                        else np.ones(n, dtype=bool))
            dr_state = _build_dr_smooth_state(
                model, pseudo[fit_rows], weights_r[fit_rows], held[fit_rows])
        else:
            dr_state = _build_dr_boosted_state(
                chosen, modifiers, values, response, pseudo, weights_r, held,
                model.mstop, seed)
    fx = EffectModel(model=model, spec=spec, modifiers=modifiers, terms=chosen,
                     cv_rloss=cv_rloss, tau=tau,
                     propensity=_Propensities(
                         e_hat, partial(_propensity_of, propensity_model, spec)),
                     outcome_mean=np.asarray(m_hat), diagnostics=diagnostics,
                     propensity_model=propensity_model, overlap_bounds=overlap_bounds,
                     constant_rloss=constant_rloss,
                     scale_plan=scale_plan if chosen == "smooth" else {},
                     interval_method=interval_method, interval_fits=interval_fits,
                     second_stage=second_stage, _dr_state=dr_state)
    fx.diagnostics["overlap"] = overlap_report(
        fx, data={name: values[name] for name in spec.covariates}).to_dict()
    fx.diagnostics["interval"] = {
        "method": interval_method,
        "refits": len(interval_fits),
        "repeats": interval_repeats if interval_method == "repeats" else None,
        "bootstrap_draws": bootstrap_draws if interval_method == "bootstrap" else None,
        "second_stage_draws": (None if second_stage is None else second_stage.n_draws),
        "second_stage_converged": (None if second_stage is None
                                   else int(sum(second_stage.converged()))),
        "dr_draws": (dr_state["n_draws"]
                     if dr_state is not None and dr_state["kind"] == "boosted" else None),
        "repeat_seed_step": _REPEAT_SEED_STEP if interval_method in ("repeats", "bootstrap")
        else None,
    }
    fx.diagnostics["propensity_learner"] = propensity_record
    return fx


#: What ``interval_method="auto"`` means for a smooth second stage.
_AUTO_INTERVAL_SMOOTH = "second_stage_bootstrap"

#: How :meth:`EffectModel.describe` names the interval of each method.
_EFFECT_INTERVAL_TEXT = {
    "refit": "per-row intervals from the penalized covariance, given the nuisances",
    "second_stage_bootstrap": (
        "per-row percentile intervals from {draws} row resamples of the second stage, with "
        "the pseudo-outcome held fixed"),
    "repeats": "per-row intervals from the median rule over split-seed refits",
    "bootstrap": "per-row percentile intervals from full-pipeline row resamples",
    "dr_orthogonal": (
        "per-row calibrated DR/orthogonal intervals with nuisance fold-perturbation uncertainty"),
}


def _propensity_text(record):
    """One sentence on the propensity formula that was chosen, with the
    out-of-fold deviances it was chosen on; empty when there was no choice."""
    if not record or len(record.get("candidates", ())) < 2:
        return ""
    deviance = record["out_of_fold_deviance"]
    chosen = record["chosen"]
    others = ", ".join(f"{label!r} {value:.6g}" for label, value in deviance.items()
                       if label != chosen)
    return (f" Propensity formula {chosen!r} chosen by out-of-fold binomial deviance "
            f"({deviance[chosen]:.6g} against {others}).")


def _second_stage_bootstrap(model, response, weights, rows, draws, seed):
    """The row bootstrap of a smooth second stage: ``draws`` resamples of the
    fitting rows, each a multinomial draw of ``len(rows)`` rows, refitted by
    :meth:`~mboostrs.selection.GamRefit.bootstrap` with the pseudo-outcome and
    its weights held fixed.

    The resamples come from ``numpy`` seeded by ``seed``; the refits and their
    REML smoothing choices are computed in the Rust core."""
    fit_index = np.flatnonzero(rows)
    n_fit = fit_index.size
    rng = np.random.default_rng([seed, 2025])
    counts = np.empty((draws, n_fit))
    probability = np.full(n_fit, 1.0 / n_fit)
    for b in range(draws):
        counts[b] = rng.multinomial(n_fit, probability)
    return model.bootstrap(counts, np.asarray(response)[fit_index],
                           np.asarray(weights)[fit_index])


def _solve_linear(mat, rhs):
    """Solve ``mat @ x = rhs`` with a minimum-norm pseudoinverse fallback."""
    try:
        return np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(mat, rhs, rcond=None)[0]


def _build_dr_smooth_state(model, z_fit, w_fit, held_fit):
    """Precompute HC3 sandwich covariance and K leave-one-fold-out refits."""
    z_fit = np.asarray(z_fit, dtype=np.float64)
    w_fit = np.asarray(model._weights if model._weights is not None else w_fit, dtype=np.float64)
    held_fit = np.asarray(held_fit, dtype=bool)
    n_fit = z_fit.size
    Xs = [np.ones((n_fit, 1))]
    Ss = [np.zeros((1, 1))]
    Z_list = []
    train_data = model._training_data
    for (_term, design, by), blk in zip(model._blocks, model._core.blocks(), strict=True):
        X_raw = np.asarray(design.design, dtype=np.float64)
        if by is not None:
            X_raw = X_raw * np.asarray(read_column(train_data, by).values, dtype=np.float64)[:, None]
        width = int(blk["end"]) - int(blk["start"])
        if width < X_raw.shape[1]:
            c = X_raw.T @ w_fit
            Q, _ = np.linalg.qr(c[:, None], mode="complete")
            Z = Q[:, X_raw.shape[1] - width:]
            X_c = X_raw @ Z
            Ks = [Z.T @ np.asarray(K, dtype=np.float64) @ Z for K in design.penalties]
        else:
            Z = None
            X_c = X_raw
            Ks = [np.asarray(K, dtype=np.float64) for K in design.penalties]
        Z_list.append(Z)
        Xs.append(X_c)
        S_blk = np.zeros((width, width))
        for lam, K in zip(blk["lambda"], Ks, strict=False):
            S_blk = S_blk + float(lam) * K
        Ss.append(S_blk)
    X_S = np.hstack(Xs)
    p_tot = X_S.shape[1]
    S_lam = np.zeros((p_tot, p_tot))
    pos = 0
    for s_blk in Ss:
        k = s_blk.shape[0]
        S_lam[pos:pos + k, pos:pos + k] = s_blk
        pos += k
    H = X_S.T @ (w_fit[:, None] * X_S) + S_lam
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)
    beta_hat = H_inv @ (X_S.T @ (w_fit * z_fit))
    res = z_fit - X_S @ beta_hat
    h_ii = w_fit * np.sum((X_S @ H_inv) * X_S, axis=1)
    scale_hc = (w_fit ** 2) * (res ** 2) / np.maximum(1.0 - h_ii, 0.2) ** 2
    V_hc = H_inv @ (X_S.T @ (scale_hc[:, None] * X_S)) @ H_inv

    K_folds = held_fit.shape[1] if held_fit.ndim == 2 else 1
    if K_folds >= 2:
        betas_loo = np.empty((K_folds, p_tot), dtype=np.float64)
        for k_fold in range(K_folds):
            keep = ~held_fit[:, k_fold]
            if not np.any(keep):
                betas_loo[k_fold] = beta_hat
                continue
            X_k = X_S[keep]
            w_k = w_fit[keep]
            H_k = X_k.T @ (w_k[:, None] * X_k) + S_lam
            betas_loo[k_fold] = _solve_linear(H_k, X_k.T @ (w_k * z_fit[keep]))
    else:
        betas_loo = beta_hat[None, :]
    df_eff = max(1.0, float(model._core.residual_df)
                 if model._core.residual_df is not None else float(n_fit))
    return {
        "kind": "smooth",
        "V_hc": V_hc,
        "Z_list": tuple(Z_list),
        "betas_loo": betas_loo,
        "df_eff": df_eff,
    }


def _dr_smooth_predict(effect_model, newdata, level, extrapolation):
    """Evaluate calibrated HC3 + fold-perturbation intervals on a smooth EffectModel."""
    from mboostrs.cvrisk import _t_quantile
    from mboostrs.selection import _apply_range_policy, _range_rows

    full = effect_model._interval_arrays(newdata, level, extrapolation)
    estimate, outside_range = full["estimate"], full["outside_range"]
    inputs = effect_model._model_inputs(newdata)
    pairs = _range_rows(effect_model.model._ranges, inputs)
    policy = "allow" if extrapolation == "warn" else extrapolation
    clipped_inputs = _apply_range_policy(pairs, inputs, policy)
    raws, bys, _ = effect_model.model._design_rows(
        effect_model.model._imputed(clipped_inputs), "error")
    dr_state = effect_model._dr_state
    Xs_te = [np.ones((len(estimate), 1))]
    for raw, by, Z in zip(raws, bys, dr_state["Z_list"], strict=True):
        if by is not None:
            raw = raw * by[:, None]
        Xs_te.append(raw @ Z if Z is not None else raw)
    X_te = np.hstack(Xs_te)
    v_hc = np.maximum(np.sum((X_te @ dr_state["V_hc"]) * X_te, axis=1), 0.0)
    betas_loo = dr_state["betas_loo"]
    K_folds = betas_loo.shape[0]
    if K_folds >= 2:
        taus_loo = betas_loo @ X_te.T
        v_loo = ((K_folds - 1.0) / K_folds) * np.sum(
            (taus_loo - taus_loo.mean(axis=0, keepdims=True)) ** 2, axis=0)
    else:
        v_loo = 0.0
    se = np.sqrt(v_hc + 0.25 * v_loo)
    tcrit = _t_quantile(0.5 + float(level) / 2.0, dr_state["df_eff"])
    lower = estimate - tcrit * se
    upper = estimate + tcrit * se
    return estimate, se, lower, upper, outside_range


def _build_dr_boosted_state(chosen, modifiers, values, response, pseudo, weights_r,
                            held, mstop, seed, n_draws=40):
    """Fit ``n_draws`` Bayesian Exp(1) * fold-perturbed boosted second-stage models."""
    import mboostrs as mb
    from mboostrs.families import Gaussian

    rhs = _rhs_for(modifiers, values, chosen)
    formula = f"{quote_response(response)} ~ {rhs}"
    n = len(pseudo)
    K_folds = held.shape[1] if held.ndim == 2 else 1
    mstop_use = max(1, min(int(mstop), 150))
    rng = np.random.default_rng([int(seed), 2026])
    base_data = {name: values[name] for name in modifiers}
    base_data[response] = np.asarray(pseudo, dtype=np.float64)
    w_base = np.asarray(weights_r, dtype=np.float64)
    models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_draws):
            exp_w = rng.exponential(1.0, size=n)
            if K_folds >= 2:
                fold_mult = rng.exponential(1.0, size=K_folds)
                fold_mult = fold_mult / max(float(np.mean(fold_mult)), 1e-12)
                row_fold = held @ fold_mult
            else:
                row_fold = 1.0
            w_draw = w_base * exp_w * row_fold
            if float(np.sum(w_draw)) <= 0.0:
                w_draw = w_base
            models.append(mb.gamboost(
                formula, data=base_data, family=Gaussian(), weights=w_draw,
                control=mb.boost_control(mstop_use, 0.1),
            ))
    return {"kind": "boosted", "models": tuple(models), "n_draws": int(n_draws)}


#: Level of the interval on the gain of the top group.
_RANK_LEVEL = 0.95


@dataclass(frozen=True)
class OverlapRanking:
    """The top group of rows by estimated effect, ranked inside the overlap region.

    ``chosen`` holds the positions in ``newdata`` of the top group, best first.
    ``outside`` holds the positions whose propensity lies outside ``bounds``;
    they are listed here and are never in ``chosen``. ``gain`` is the mean
    estimated effect over the chosen rows. ``gain_interval`` is its interval at
    ``level``, computed from the covariance of the fitted effect model; it is
    ``None`` when the effect has no coefficient covariance (a boosted
    candidate, or a plain frame of estimates). ``shortfall`` counts the places
    the top group could not fill because fewer rows lie inside the overlap
    than ``share`` asks for. ``rows`` holds one record per row of ``newdata``
    with its estimate, interval, propensity, overlap flag, rank among the
    overlap rows and whether it is chosen.
    """

    chosen: tuple
    outside: tuple
    gain: float
    gain_interval: tuple | None
    level: float | None
    share: float
    bounds: tuple
    shortfall: int
    rows: tuple = field(repr=False)
    #: Rows whose estimate equals the estimate at the cut between the top group
    #: and the rest, when that estimate is shared across the cut; otherwise 0.
    tied_rows: int = 0
    #: The rule that orders rows with equal estimates: interval width (or
    #: standard error) and then row order.
    tie_break: str = "row order"

    def table(self, output="auto"):
        """One row per row of ``newdata``; see the ``rows`` field. The
        ``tied_at_boundary`` column marks the tied rows."""
        return output_frame(list(self.rows), output)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row records only on request."""
        return export_result(self, include_arrays=include_arrays, arrays=("rows",),
                             extras={"n_chosen": len(self.chosen),
                                     "n_outside": len(self.outside)})

    def describe(self) -> str:
        """The top group, its gain, and the rows left out for lack of overlap."""
        lower, upper = self.bounds
        text = (f"Top {len(self.chosen)} of {len(self.rows)} rows by estimated effect, "
                f"ranked among rows with propensity in [{lower:g}, {upper:g}]. "
                f"Mean estimated effect of the top group {self.gain:.6g}")
        if self.gain_interval is not None:
            text += (f" with {self.level:.0%} interval [{self.gain_interval[0]:.6g}, "
                     f"{self.gain_interval[1]:.6g}]")
        text += f". {len(self.outside)} row(s) outside the overlap region are listed "
        text += "separately and were not ranked."
        if self.tied_rows:
            text += (f" {self.tied_rows} row(s) tie at the selection boundary, all with the "
                     f"same estimate; ties are broken by {self.tie_break}.")
        else:
            text += f" No rows tie at the selection boundary; ties are broken by {self.tie_break}."
        if self.shortfall:
            text += (f" The overlap region holds fewer rows than the share asks for: "
                     f"{self.shortfall} place(s) of the top group are unfilled.")
        return text


def rank_by_effect(effect, newdata, *, share=0.2, propensity=None,
                   bounds=DEFAULT_OVERLAP_BOUNDS):
    """Rank new rows by estimated effect, using only rows with overlap.

    Propensity values near 0 or 1 indicate that one treatment arm has few
    comparable rows. Effect estimates in those regions rely on extrapolation.
    This function ranks rows inside ``bounds`` and lists rows outside the
    overlap region separately.

    Parameters
    ----------
    effect : EffectModel, BLPProjection or data frame
        An :class:`EffectModel`, a :class:`~mboostrs.effect_modifiers.BLPProjection`
        from :meth:`~mboostrs.effect_modifiers.EffectModifiers.projection` (its
        linear projection of the effect on the covariates), or a frame of
        estimated effects with an ``estimate`` column, one row per row of
        ``newdata`` (for example the output of ``EffectModel.predict``). A
        projection or a frame needs ``propensity``.
    newdata : mapping or data frame
        The rows to rank. For an ``EffectModel`` it needs the modifiers and, when
        ``propensity`` is not given, every covariate of the propensity model.
    share : float
        The top group holds ``round(share * n)`` rows, where ``n`` counts every
        row of ``newdata``, so the default ranks the top fifth of the rows. If
        fewer rows lie inside ``bounds``, the group is the inside rows and
        ``shortfall`` says how many places are unfilled.
    propensity : array-like, optional
        The propensity of each row of ``newdata``. By default an
        :class:`EffectModel` computes it from its propensity model.
    bounds : tuple
        ``(lower, upper)``: a row is inside the overlap region when its
        propensity lies in the closed interval. The default ``(0.02, 0.98)``
        matches :func:`overlap_report`.

    Returns
    -------
    OverlapRanking
        ``chosen`` and ``outside`` are positions in ``newdata``. The gain
        interval is the mean effect's own interval (level 0.95), computed
        from the full coefficient covariance, and is available only for an
        :class:`EffectModel` with ``terms="smooth"``.

    Ties: rows with exactly equal estimates are ordered by interval width (the
    span of ``lower`` to ``upper``, or ``se`` when only that is available),
    narrower first, and then by row position. A discrete modifier gives many
    rows the same estimate, so the boundary can split a tie. ``tied_rows``
    counts the rows at the cut that share its estimate, and ``tie_break`` names
    the rule; both appear in :meth:`OverlapRanking.describe`.
    """
    lower, upper = _check_bounds(bounds)
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1]; got {share!r}")
    from mboostrs.effect_modifiers import BLPProjection

    mean_interval = None
    if isinstance(effect, BLPProjection):
        if propensity is None:
            raise ValueError(
                "rank_by_effect on a BLPProjection needs propensity=; the projection has no "
                "propensity model of its own")
        estimate = effect.predict(newdata)
        se = lo = hi = None
    elif isinstance(effect, EffectModel):
        values = effect._model_inputs(newdata)
        if isinstance(effect.model, GamRefit):
            records = effect.model.predict(newdata=values, interval=True, level=_RANK_LEVEL,
                                           output="records", extrapolation="clip")
            estimate, se, lo, hi = (np.array([r[key] for r in records], dtype=np.float64)
                                    for key in ("estimate", "se", "lower", "upper"))

            def mean_interval(rows):
                subset = {name: value[rows] for name, value in values.items()}
                return effect.model._mean_interval(subset, level=_RANK_LEVEL)[1:]
        else:
            estimate = np.asarray(effect.model.predict(newdata=values),
                                  dtype=np.float64).ravel()
            se = lo = hi = None
        if propensity is None:
            try:
                propensity = effect.propensity(newdata)
            except ValueError as error:
                raise ValueError(
                    f"rank_by_effect needs the propensity of the new rows: {error}; "
                    "or pass propensity=") from error
    else:
        if propensity is None:
            raise ValueError("rank_by_effect on a frame of estimates needs propensity=")
        if "estimate" not in column_names(effect):
            raise ValueError("an effect frame needs an 'estimate' column; pass the output of "
                             "EffectModel.predict or an EffectModel")
        estimate = np.asarray(read_column(effect, "estimate").values, dtype=np.float64)
        names = column_names(effect)
        se, lo, hi = (np.asarray(read_column(effect, key).values, dtype=np.float64)
                      if key in names else None for key in ("se", "lower", "upper"))
    e = np.asarray(propensity, dtype=np.float64).ravel()
    n = estimate.size
    if e.size != n:
        raise ValueError(f"propensity has {e.size} values for {n} rows; pass one per row")
    inside = np.isfinite(e) & (e >= lower) & (e <= upper) & np.isfinite(estimate)
    k = max(1, int(round(share * n)))
    candidates = np.flatnonzero(inside)
    # Rows with equal estimates are ordered by interval width, narrower first, and
    # then by row position. The width is the interval's span when the estimates
    # carry one, else the standard error; with neither, the order is row order.
    if lo is not None and hi is not None:
        width = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
        tie_break = "lower interval width, then row order"
    elif se is not None:
        width = np.asarray(se, dtype=np.float64)
        tie_break = "lower standard error, then row order"
    else:
        width = np.zeros(n)
        tie_break = "row order (these estimates carry no interval width)"
    width = np.where(np.isfinite(width), width, np.inf)
    # np.lexsort takes its last key as the primary one.
    order = np.lexsort((candidates, width[candidates], -estimate[candidates]))
    ranked = candidates[order]
    chosen = ranked[:k]
    shortfall = int(k - chosen.size)
    # A tie at the boundary is an estimate shared by a chosen row and an unchosen
    # row. The rows sharing that estimate are marked; the rest of the ranking is
    # not a tie.
    tied = np.zeros(n, dtype=bool)
    if chosen.size and ranked.size > chosen.size:
        cut = estimate[ranked[chosen.size - 1]]
        if np.any(estimate[ranked[chosen.size:]] == cut):
            tied[candidates[estimate[candidates] == cut]] = True
    tied_rows = int(tied.sum())
    outside = np.flatnonzero(~inside)
    rank_of = {int(row): position + 1 for position, row in enumerate(ranked)}
    chosen_set = {int(row) for row in chosen}
    rows = []
    for i in range(n):
        record = {
            "row": i, "estimate": float(estimate[i]),
            "propensity": float(e[i]), "in_overlap": bool(inside[i]),
            "rank": rank_of.get(i), "chosen": i in chosen_set,
            "tied_at_boundary": bool(tied[i]),
        }
        for key, column in (("se", se), ("lower", lo), ("upper", hi)):
            record[key] = None if column is None else float(column[i])
        rows.append(record)
    gain = float(np.mean(estimate[chosen])) if chosen.size else float("nan")
    gain_interval = None
    level = None
    if mean_interval is not None and chosen.size:
        gain_lower, gain_upper = mean_interval(chosen)
        gain_interval = (gain_lower, gain_upper)
        level = _RANK_LEVEL
    return OverlapRanking(
        chosen=tuple(int(row) for row in chosen),
        outside=tuple(int(row) for row in outside),
        gain=gain, gain_interval=gain_interval, level=level, share=float(share),
        bounds=(lower, upper), shortfall=shortfall, rows=tuple(rows),
        tied_rows=tied_rows, tie_break=tie_break)


@dataclass(frozen=True)
class CITest:
    """A test of ``x`` independent of ``y`` given the columns in ``given``.

    ``status`` is ``"tested"`` when the test ran, and ``"dependent"`` when x or y
    is an exact linear combination of the conditioning set. In that case
    ``skipped`` is True, nothing was tested, and ``statistic``, ``p_value`` and
    ``residual_correlation`` are NaN. ``notes`` names the column to drop to break
    the identity, the one ``ci_test`` suggests. Read a skipped result as
    undetermined: no p-value exists for it.
    """

    x: str
    y: str
    given: tuple[str, ...]
    statistic: float
    p_value: float
    residual_correlation: float
    n: int
    folds: int
    #: Functions of ``x`` tested jointly: ``x`` alone (1) or ``x`` and three
    #: hinge functions at its quartiles (4, a chi-squared test with 4 df).
    functions: int = 1
    #: Conditioning columns removed because they are exact linear combinations
    #: of the others; ``given`` still lists what the caller asked for.
    dropped: tuple[str, ...] = ()
    #: One line per decision the test made beyond the statistic: a dropped
    #: column and what it depends on, a regression that reached the step cap,
    #: or x or y being a combination of the conditioning set.
    notes: tuple[str, ...] = ()
    #: True when the test was not run because x or y is an exact linear
    #: combination of the conditioning set. The statistic and p-value are then NaN.
    skipped: bool = False
    #: ``"tested"``, or ``"dependent"`` when ``skipped`` is True.
    status: str = "tested"
    #: How rows with a missing outcome were handled: ``"error"`` or ``"ipw"``.
    outcome_missing: str = "error"
    #: Rows with a missing y value (zero unless ``outcome_missing="ipw"``).
    n_missing_outcome: int = 0

    def to_dict(self, *, include_arrays=False):
        return export_result(self, include_arrays=include_arrays)

    def __repr__(self):
        return (f"CITest(status={self.status!r}, x={self.x!r}, y={self.y!r}, "
                f"given={self.given!r}, statistic={self.statistic:.6g}, "
                f"p_value={self.p_value:.6g}, n={self.n})")

    def describe(self) -> str:
        cond = f" given {list(self.given)}" if self.given else ""
        if self.skipped:
            body = (f"UNDETERMINED (status 'dependent'): {self.x} vs {self.y}{cond} was not "
                    "tested, because x or y is an exact linear combination of the "
                    "conditioning set. The statistic and p-value are NaN: no test was run, "
                    "so there is no p-value to read as evidence either way.")
        else:
            kind = ("GCM statistic" if self.functions == 1
                    else f"chi-squared statistic ({self.functions} functions of {self.x})")
            body = (f"{self.x} vs {self.y}{cond}: {kind} {self.statistic:.3g}, "
                    f"p = {self.p_value:.3g}, residual correlation "
                    f"{self.residual_correlation:.3g} ({self.n} rows, {self.folds}-fold "
                    f"cross-fitted boosted regressions). A small p-value says the two "
                    f"are associated after the conditioning columns; a large one says "
                    f"no association was detected, which is not proof of independence.")
        return body + "".join(f" Note: {note}" for note in self.notes)


#: Largest step count a residual regression in :func:`ci_test` may reach. The
#: grid doubles from 100 steps, so the cap is reached after four doublings. A
#: regression whose out-of-fold risk still falls at the cap is reported in the
#: result's ``notes``. On the benchmark's 12,000-row file, caps of 1,000 and
#: 10,000 moved the statistics by under 1.5%; the cap is what sets the cost.
CI_MAX_MSTOP = 1000

#: Relative tolerance for an exact linear dependence among the conditioning
#: columns, and between x or y and them: a column whose component outside the
#: others is at most this share of its own norm is treated as a combination.
CI_DEPENDENCE_TOL = 1e-10

#: The bounds on the probability that an outcome is observed when
#: :func:`ci_test` weights the observed rows by its inverse. The same default
#: as the propensity clip of :func:`aipw`.
CI_RESPONSE_CLIP = (0.01, 0.99)

_CI_ALIAS = re.compile(r"__ci_given(\d+)\b")


def _unalias(text, names):
    """``text`` with each internal ``__ci_givenK`` alias replaced by the
    caller's column name ``names[K]``."""
    def name_of(match):
        k = int(match.group(1))
        return names[k] if k < len(names) else match.group(0)
    return _CI_ALIAS.sub(name_of, str(text))


def _fit_values(data, name):
    """The values of a conditioning column as the learner should see them.

    A categorical column is passed as its labels: its numeric codes would be
    read as an ordered number."""
    from mboostrs._frame import _raw

    column = read_column(data, name)
    if column.kind != "categorical":
        return column.values
    raw = _raw(data, name)
    return np.asarray(raw.to_numpy() if hasattr(raw, "to_numpy") else raw)


def _missing_values_message(name, count, role):
    """The refusal for a column with missing values, naming the remedies for its role."""
    if role == "outcome":
        return (
            f"{name!r} has {count} missing values, and ci_test needs an observed outcome for "
            "every row. Pass outcome_missing='ipw' to keep every row and weight the observed "
            "rows by the inverse of a fitted response model, which assumes the outcome is "
            "missing at random given the conditioning columns and x. Or drop the rows with a "
            f"missing {name!r} before the call. Dropping them can bias a role test: when "
            "whether the outcome is observed depends on a column affected by x or y (a "
            "post-treatment column), the test then conditions on that column.")
    return (f"{name!r} has {count} missing values; ci_test does not drop rows. Remove them "
            "from the data before the call.")


def _checked_numeric(data, name, *, allow_missing=False, role="x"):
    """A numeric x or y column. Missing values are refused unless ``allow_missing``,
    in which case they are NaN in the returned array."""
    column = read_column(data, name)
    if column.kind == "categorical":
        raise TypeError(
            f"ci_test needs {name!r} to be numeric; it is categorical. Pass a "
            "0/1 indicator for a two-level column, or name it under given=")
    if column.n_missing and not allow_missing:
        raise ValueError(_missing_values_message(name, column.n_missing, role))
    return np.asarray(column.values, dtype=np.float64)


def _given_design(data, given, n):
    """The intercept and the design columns of a conditioning set.

    Each numeric or boolean column is one design column. A categorical column
    contributes one indicator per level after its first observed level, which
    is how a learner codes a factor with treatment contrasts. ``owner`` gives
    the index in ``given`` that each design column came from, and ``None`` for
    the intercept. Levels with no rows are skipped, since their indicator is
    all zeros.
    """
    columns, labels, owner = [np.ones(n)], ["intercept"], [None]
    for k, name in enumerate(given):
        column = read_column(data, name)
        if column.n_missing:
            raise ValueError(
                f"the conditioning column {name!r} has {column.n_missing} missing "
                "values; ci_test does not drop rows. Remove them before the call.")
        if column.kind == "categorical":
            counts = [int(np.sum(column.values == level)) for level in range(column.n_levels)]
            present = [level for level, count in enumerate(counts) if count > 0]
            for level in present[1:]:
                columns.append((column.values == level).astype(np.float64))
                labels.append(f"{name}[{column.levels[level]}]")
                owner.append(k)
        else:
            columns.append(np.asarray(column.values, dtype=np.float64))
            labels.append(name)
            owner.append(k)
    return columns, labels, owner


def _expression(target, labels, depends_on):
    """``target = c1 * label1 + c2 * label2 ...`` for a dependent column."""
    terms = " ".join(f"{coef:+.6g} * {labels[j]}" for j, coef in depends_on)
    return f"{target} = {terms or '0'}"


def _is_categorical(data, name):
    """Whether a column is categorical (a factor), as opposed to numeric or boolean."""
    return read_column(data, name).kind == "categorical"


def _categorical_indicators(data, name, *, allow_missing=False):
    """The level indicators of a categorical column, and their labels.

    One 0/1 column per level after the first observed level, which is how a
    factor enters a boosted model with treatment contrasts. Levels with no rows
    are skipped. The labels name each indicator by its level. A row with a
    missing level is NaN in every indicator when ``allow_missing`` is set, and
    is refused otherwise.
    """
    column = read_column(data, name)
    if column.n_missing and not allow_missing:
        raise ValueError(_missing_values_message(name, column.n_missing, "x"))
    present = [level for level in range(column.n_levels)
               if np.any(column.values == level)]
    if len(present) < 2:
        raise ValueError(
            f"{name!r} has fewer than two observed levels, so it is constant and cannot "
            "be tested; drop the column or pass a column that varies")
    missing = np.isnan(column.values)
    columns = [(column.values == level).astype(np.float64) for level in present[1:]]
    for indicator in columns:
        indicator[missing] = np.nan
    labels = [f"{name}[{column.label(level)}]" for level in present[1:]]
    return np.column_stack(columns), labels


def _span_dependence(kept_columns, values, n):
    """The ``(j, coefficient)`` pairs over the kept columns when ``values`` is an
    exact combination of them, else ``None``. ``j`` indexes ``kept_columns``."""
    from mboostrs._rust import linear_dependence

    ordered = kept_columns + [values]
    hit = linear_dependence(np.concatenate(ordered), n, len(ordered), CI_DEPENDENCE_TOL)
    target_norm = float(np.linalg.norm(values))
    for position, depends_on in hit:
        if position == len(kept_columns):
            # A coefficient whose term is below rounding relative to the target is
            # numerical residue of the elimination, not part of the identity.
            return [(j, c) for j, c in depends_on
                    if abs(c) * float(np.linalg.norm(kept_columns[j]))
                    > CI_DEPENDENCE_TOL * target_norm]
    return None


def _identity_advice(target, involved):
    """The sentence that names the columns whose identity with ``target`` must be
    broken, and the one ``ci_test`` suggests dropping.

    ``involved`` lists the conditioning column names in the identity, in the
    order the caller gave them. The suggestion is the last of them, the column
    the conditioning-set rule drops first. Removing it leaves ``target`` as a
    combination of fewer columns, so the test can run."""
    if not involved:
        return ""
    names = [*involved, target]
    listed = ", ".join(f"{name!r}" for name in names[:-1]) + f" or {names[-1]!r}"
    suggestion = involved[-1]
    return (f"Breaking the identity needs one of {listed} to leave the set; ci_test "
            f"suggests dropping {suggestion!r}, the last of the conditioning columns in it.")


def _span_expression(kept_columns, kept_labels, values, target, n):
    """``target = c1 * label1 + ...`` when ``values`` is a combination of the kept
    columns, else ``None``."""
    depends_on = _span_dependence(kept_columns, values, n)
    if depends_on is None:
        return None
    return _expression(target, kept_labels, depends_on)


def _reduce_conditioning(data, given, x_name, x_matrix, x_labels, y_name, y_values, n,
                         *, observed=None):
    """Find exact linear dependence in the conditioning set and between x or y
    and it.

    ``x_matrix`` holds the columns of x that enter the test: one column for a
    numeric x, one 0/1 indicator per level for a categorical x, with
    ``x_labels`` naming them. ``observed`` marks the rows that have a y value;
    y is checked for dependence on those rows only. Returns ``(kept, dropped,
    notes, degenerate, x_keep)``: the conditioning columns kept, the ones dropped
    as combinations of the kept columns, one note per decision, whether the test
    has nothing to test, and which columns of ``x_matrix`` are kept. An x column
    that is a combination of the kept conditioning columns has a zero residual,
    so it is dropped from the test. The test is degenerate when no x column is
    left, or when y is itself such a combination: its residual is zero by
    construction.
    """
    from mboostrs._rust import linear_dependence

    if observed is None:
        observed = np.ones(n, dtype=bool)
    columns, labels, owner = _given_design(data, given, n)
    found = linear_dependence(np.concatenate(columns), n, len(columns), CI_DEPENDENCE_TOL)
    dropped, notes = [], []
    for position, depends_on in found:
        k = owner[position]
        if k is None:
            continue
        if k not in dropped:
            dropped.append(k)
        expression = _expression(labels[position],
                                 labels, [(j, c) for j, c in depends_on])
        notes.append(
            f"{given[k]!r} was dropped from the conditioning set: it is an exact "
            f"linear combination of the columns before it ({expression}, with "
            "the intercept as a term), so it adds nothing once they are in the set.")
    kept_positions = [i for i, k in enumerate(owner) if k is None or k not in dropped]
    kept_columns = [columns[i] for i in kept_positions]
    kept_labels = [labels[i] for i in kept_positions]
    # The given column each kept design column came from (None for the intercept).
    kept_owner = [owner[i] for i in kept_positions]
    kept = tuple(name for k, name in enumerate(given) if k not in dropped)

    def involved_names(depends_on):
        ks = sorted({kept_owner[j] for j, _ in depends_on if kept_owner[j] is not None})
        return [given[k] for k in ks]

    degenerate = False
    x_keep = np.ones(x_matrix.shape[1], dtype=bool)
    x_expressions = []
    for j in range(x_matrix.shape[1]):
        depends_on = _span_dependence(kept_columns, x_matrix[:, j], n)
        if depends_on is not None:
            x_keep[j] = False
            expression = _expression(x_labels[j], kept_labels, depends_on)
            x_expressions.append((x_labels[j], expression, involved_names(depends_on)))
    if not x_keep.any():
        degenerate = True
        for label, expression, involved in x_expressions:
            advice = _identity_advice(label, involved)
            notes.append(
                f"{label!r} is an exact linear combination of the conditioning set "
                f"({expression}), so its regression residual is zero and no test was run: "
                f"the statistic and p_value are NaN. {advice}".rstrip())
    else:
        for label, expression, involved in x_expressions:
            advice = _identity_advice(label, involved)
            notes.append(
                f"{label!r} is an exact linear combination of the conditioning set "
                f"({expression}), so its regression residual is zero; it is dropped "
                f"from the test. {advice}".rstrip())
    observed_count = int(np.sum(observed))
    y_depends = _span_dependence([column[observed] for column in kept_columns],
                                 y_values[observed], observed_count)
    if y_depends is not None:
        degenerate = True
        y_expression = _expression(y_name, kept_labels, y_depends)
        advice = _identity_advice(y_name, involved_names(y_depends))
        notes.append(
            f"{y_name!r} is an exact linear combination of the conditioning set "
            f"({y_expression}), so its regression residual is zero and no test was run: "
            f"the statistic and p_value are NaN. {advice}".rstrip())
    return kept, tuple(given[k] for k in dropped), tuple(notes), degenerate, x_keep


def _crossfit_prediction(data, given, values, fold, seed, *, family=None, fit_rows=None,
                         extrapolation="warn"):
    """Out-of-fold predictions of ``values`` from a boosted regression on ``given``,
    and whether any fold's regression reached :data:`CI_MAX_MSTOP` with its risk
    still falling.

    ``family`` is the response family (None for Gaussian); a binomial family
    returns probabilities. ``fit_rows`` marks the rows the regressions are fitted
    on, so rows with a missing ``values`` entry can be left out of the fit. Each
    conditioning column is copied under the alias ``__ci_givenK`` (its position in
    ``given``) before the formula is built, because the formula is text and a
    column name may contain characters the grammar reads as syntax. Every column
    keeps its type: a categorical column is passed as its labels. The step count
    of each fold's regression is tuned by inner cross-validation on the training
    rows, with a fold-level step cap of :data:`CI_MAX_MSTOP`.
    """
    import warnings

    from mboostrs._frame import take_rows
    from mboostrs.explore import tune_mstop
    from mboostrs.transforms import _with_columns as with_columns

    values = np.asarray(values, dtype=np.float64)
    fit_rows = (np.ones(values.size, dtype=bool) if fit_rows is None
                else np.asarray(fit_rows, dtype=bool))
    aliases = [f"__ci_given{k}" for k in range(len(given))]
    columns = {"__ci_target": values,
               **{alias: _fit_values(data, name)
                  for alias, name in zip(aliases, given, strict=True)}}
    data = with_columns(data, columns)
    predicted_all = np.empty(values.size)
    # A numeric column takes the default learner (bbs); a factor takes bols, which
    # codes it with treatment contrasts, as the default learner does.
    terms = [f"bols({quote_name(alias)})" if read_column(data, alias).kind == "categorical" else alias
             for alias in aliases]
    formula = "__ci_target ~ " + " + ".join(terms)
    predict_kwargs = {} if family is None else {"type": "response"}
    capped = False
    for k in np.unique(fold):
        train = np.flatnonzero((fold != k) & fit_rows)
        test = np.flatnonzero(fold == k)
        with warnings.catch_warnings():
            # The cap is reported through `capped`, and the curve of a residual
            # regression is not a finding, so tune_mstop's flat-curve note is off.
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message="the cross-validated risk was still falling")
            tuning = tune_mstop(formula, take_rows(data, train), family, folds=5,
                                seed=seed + int(k), nu=0.3, max_mstop=CI_MAX_MSTOP,
                                flat_tolerance=None)
        capped = capped or bool(tuning.at_boundary)
        predicted = np.asarray(tuning.model.predict(
            newdata=take_rows(data, test), unseen_levels="zero_effect",
            extrapolation=extrapolation, **predict_kwargs), dtype=float)
        predicted_all[test] = predicted.ravel()
    return predicted_all, capped


def _crossfit_residual(data, given, values, fold, seed, *, extrapolation="warn",
                       fit_rows=None):
    """Out-of-fold residuals of ``values`` after a boosted regression on ``given``,
    and whether any fold's regression reached :data:`CI_MAX_MSTOP` with its risk
    still falling. A row outside ``fit_rows`` has residual 0."""
    values = np.asarray(values, dtype=np.float64)
    fit_rows = (np.ones(values.size, dtype=bool) if fit_rows is None
                else np.asarray(fit_rows, dtype=bool))
    if not given:
        residual = np.zeros(values.size)
        residual[fit_rows] = values[fit_rows] - values[fit_rows].mean()
        return residual, False
    predicted, capped = _crossfit_prediction(data, given, values, fold, seed,
                                             fit_rows=fit_rows, extrapolation=extrapolation)
    return np.where(fit_rows, values - predicted, 0.0), capped


def ci_test(data, x, y, given=(), *, folds=5, seed=0, extrapolation="warn",
            outcome_missing="error"):
    """Test whether ``x`` and ``y`` are independent given ``given``.

    The generalised covariance measure (Shah and Peters, 2020): regress ``x``
    on ``given`` and ``y`` on ``given`` with boosted additive models, each
    predicted out of fold, and test whether the products of the two residuals
    have mean zero. A single covariance misses a dependence that is not
    monotone (``y`` on ``x^2`` for a symmetric ``x``), so a numeric ``x`` with
    ten or more distinct values is expanded into ``x`` and hinge functions at
    its quartiles, each residualized on ``given``, and the four residual
    products are tested jointly (a chi-squared test with four degrees of
    freedom, the multivariate form of the same test). The statistic is approximately standard normal under
    conditional independence when both regressions are accurate; it needs no
    linear model and no distributional assumption on ``x`` or ``y``. A
    column that predicts the response only through a driver it is correlated
    with (a proxy) shows no association once the driver is in ``given``.

    Parameters
    ----------
    data : mapping or data frame
    x, y : str
        A numeric column (a binary 0/1 column is fine: its residual is taken
        from its conditional mean), or a categorical column.

        A categorical ``x`` with more than two levels is tested jointly as a
        factor: each level after the first observed level is one 0/1
        indicator, residualized on ``given`` like a numeric ``x``, and the
        indicators are tested together with a chi-squared test whose degrees
        of freedom equal the number of indicators. A two-level ``x`` is one
        indicator, tested as a binary column.

        A categorical ``y`` with two observed levels is one indicator, tested
        as a binary column. A ``y`` with more than two levels is refused, with
        the per-level route named: pass ``data[y] == level`` as a 0/1 column for
        each level of interest.
    given : sequence of str
        Conditioning columns, numeric, boolean, categorical or string. A
        categorical column enters as a factor, coded with treatment contrasts
        as :func:`~mboostrs.gamboost` codes a string column. Naming rule: any
        column name is accepted, including one with ``=``, a space or a dot.
        The names are never parsed as formula text; each column is copied
        under an internal alias (``__ci_given0``, ...) before the boosted
        regressions are fitted, with its values and type unchanged. ``x``,
        ``y`` and ``given`` must be distinct names.
    folds : int
        Cross-fitting folds.
    extrapolation : {"warn", "error", "allow", "clip"}
        How a held-out row outside the training range of a conditioning
        column is predicted, as in :meth:`GAMBoostModel.predict`. The
        regressions' warnings are combined into one per call, which names the
        columns of ``given`` rather than their internal aliases.
    outcome_missing : {"error", "ipw"}
        What a missing value of ``y`` means. ``"error"`` (the default) refuses
        it, because dropping those rows conditions the test on the rows whose
        outcome was observed, which can bias it when that depends on a
        post-treatment column. ``"ipw"`` keeps every row and weights each
        observed row by the inverse of a boosted model of P(y observed | given,
        x), clipped to 0.01 to 0.99. The test then assumes the outcome is
        missing at random given ``given`` and x. Missing values in x or the
        conditioning columns are refused under either setting.

    Returns
    -------
    CITest
        ``status`` is ``"dependent"`` and ``skipped`` is True when x or y is an
        exact linear combination of the conditioning set: the test is then not
        run, and ``statistic``, ``p_value`` and ``residual_correlation`` are NaN.
        The ``notes`` name the columns of that identity and the one to drop to
        break it (the last conditioning column in the identity). Otherwise
        ``status`` is ``"tested"``. ``dropped`` names conditioning columns removed
        because they are exact linear combinations of the others, and ``notes``
        records each such decision, every regression that reached the step cap,
        and any case in which x or y is itself a combination of the conditioning
        set. The
        ``functions`` field is the number of residual functions of x that were
        tested: 1 for a numeric x with fewer than ten distinct values, 4 with
        the quartile hinges, and the number of indicators for a categorical x.
        ``residual_correlation`` is the correlation of the first function of x
        (x itself, or the first indicator of a categorical x) with y's residual.
    """
    import warnings
    from dataclasses import replace as dc_replace

    from mboostrs._extrapolation import CollectedRows, collecting, show_summary_once, summarize
    from mboostrs._rust import gcm_test, gcm_test_multi
    from mboostrs._validate import check_int_at_least
    from mboostrs.families import Binomial

    given = tuple(given)
    folds = check_int_at_least("folds", folds, 2, allow_none=False)
    if extrapolation not in ("warn", "error", "allow", "clip"):
        raise ValueError("extrapolation must be one of 'warn', 'error', 'allow', "
                         f"'clip'; got {extrapolation!r}")
    outcome_missing = _check_outcome_missing(outcome_missing)
    if len(set(given)) != len(given):
        raise ValueError("given must not name a column twice")
    if x == y or x in given or y in given:
        raise ValueError("x, y and the given columns must be distinct")
    ipw = outcome_missing == "ipw"
    x_factor = _is_categorical(data, x)
    y_factor = _is_categorical(data, y)
    if x_factor:
        x_matrix, x_labels = _categorical_indicators(data, x)
        x_values = None
    else:
        x_values = _checked_numeric(data, x)
        x_matrix, x_labels = x_values[:, None], [x]
    if y_factor:
        y_matrix, _ = _categorical_indicators(data, y, allow_missing=ipw)
        if y_matrix.shape[1] != 1:
            raise TypeError(
                f"ci_test tests a categorical y only when it has two observed levels; {y!r} "
                f"has {y_matrix.shape[1] + 1}. Test one level at a time: "
                f"pass (data[{y!r}] == level) as a 0/1 column for each level of interest, "
                "as y, and read the tests together.")
        y_values = y_matrix[:, 0]
    else:
        y_values = _checked_numeric(data, y, allow_missing=ipw, role="outcome")
    observed = ~np.isnan(y_values)
    missing_outcomes = int(np.sum(~observed))
    weighted = ipw and missing_outcomes > 0
    n = x_matrix.shape[0]
    fold = np.random.default_rng(seed).permutation(n) % folds
    capped_targets = []
    kept = given
    notes = []
    skipped = False
    with warnings.catch_warnings(record=True) as caught, collecting() as rows:
        warnings.simplefilter("always")
        try:
            kept, dropped, found, degenerate, x_keep = _reduce_conditioning(
                data, given, x, x_matrix, x_labels, y, y_values, n, observed=observed)
            notes = list(found)
            if degenerate:
                statistic, p_value, correlation, used, functions = (
                    float("nan"), float("nan"), float("nan"), n, 1)
                skipped = True
            else:
                r_y, capped = _crossfit_residual(data, kept, y_values, fold, seed + 7,
                                                 extrapolation=extrapolation,
                                                 fit_rows=observed)
                if capped:
                    capped_targets.append(y)
                r_y_raw = r_y
                if weighted:
                    # The probability that y is observed, given the conditioning set and x.
                    # Each observed row enters with weight 1 / probability; a missing
                    # row has residual 0 and so contributes nothing.
                    observed_prob, capped = _crossfit_prediction(
                        data, (*kept, x), observed.astype(np.float64), fold, seed + 11,
                        family=Binomial(), extrapolation=extrapolation)
                    if capped:
                        capped_targets.append(f"the probability that {y!r} is observed")
                    r_y = r_y / np.clip(observed_prob, *CI_RESPONSE_CLIP)
                basis = x_matrix[:, x_keep] if x_factor else _x_basis(x_values)
                functions = int(basis.shape[1])
                if functions == 1:
                    r_x, capped = _crossfit_residual(data, kept, basis[:, 0], fold, seed,
                                                     extrapolation=extrapolation)
                    if capped:
                        capped_targets.append(x)
                    statistic, p_value, correlation, used = gcm_test(
                        np.ascontiguousarray(r_x), np.ascontiguousarray(r_y))
                    first_residual = r_x
                else:
                    columns = []
                    for j in range(functions):
                        r_j, capped = _crossfit_residual(
                            data, kept, basis[:, j], fold, seed + j,
                            extrapolation=extrapolation)
                        columns.append(r_j)
                        if capped:
                            capped_targets.append(f"{x} (basis function {j + 1})")
                    statistic, p_value, correlation, used = gcm_test_multi(
                        np.ascontiguousarray(np.column_stack(columns).ravel()), functions,
                        np.ascontiguousarray(r_y))
                    first_residual = columns[0]
                if weighted:
                    # The correlation describes the rows that have an outcome; the
                    # weights are not part of it.
                    _, _, correlation, _ = gcm_test(
                        np.ascontiguousarray(first_residual[observed]),
                        np.ascontiguousarray(r_y_raw[observed]))
        except ValueError as error:
            raise ValueError(_unalias(error, (*kept, x))) from error
    # The aliases number the conditioning columns in order: the residual fits
    # use the kept columns, and the response model of x and y uses them followed
    # by x, so every alias maps to a caller's name through (*kept, x).
    names = (*kept, x)
    for item in caught:
        if issubclass(item.category, RuntimeWarning) and "still falling" in str(item.message):
            continue
        text = _unalias(item.message, names)
        if text == str(item.message):
            warnings.warn(item.message, item.category, stacklevel=2)
        elif isinstance(item.message, ExtrapolationWarning):
            warnings.warn(ExtrapolationWarning(text, rows=item.message.rows),
                          item.category, stacklevel=2)
        else:
            warnings.warn(text, item.category, stacklevel=2)
    if rows:
        renamed = CollectedRows(
            dc_replace(row, learner=_unalias(row.learner, names),
                       variable=_unalias(row.variable, names))
            for row in rows)
        renamed.calls = rows.calls
        renamed.clipped = rows.clipped
        aggregated = summarize(renamed, owner="ci_test")
        if aggregated is not None:
            show_summary_once(aggregated, stacklevel=2)
    if weighted:
        notes.append(
            f"{y!r} is missing for {missing_outcomes} of {n} rows. The test keeps every row "
            f"and weights each observed row by 1 / P({y!r} is observed | {', '.join(map(repr, (*kept, x)))}), "
            f"with the probability from a boosted response model clipped to "
            f"{CI_RESPONSE_CLIP[0]:g} to {CI_RESPONSE_CLIP[1]:g}. This assumes the outcome is "
            "missing at random given those columns.")
    if capped_targets:
        notes.append(
            f"the boosted regression of {', '.join(map(str, capped_targets))} on the "
            f"conditioning set reached the {CI_MAX_MSTOP}-step cap with its out-of-fold "
            "risk still falling; its residuals come from that fit, not from a converged "
            "one.")
    return CITest(x=x, y=y, given=given, statistic=float(statistic), p_value=float(p_value),
                  residual_correlation=float(correlation), n=int(used), folds=folds,
                  functions=functions, dropped=dropped, notes=tuple(notes),
                  skipped=skipped, status="dependent" if skipped else "tested",
                  outcome_missing=outcome_missing, n_missing_outcome=missing_outcomes)


def _x_basis(values):
    """``x`` and hinge functions at its quartiles, for a column with enough
    distinct values; ``x`` alone otherwise. The single covariance of ``x``
    with ``y`` misses a dependence such as ``y`` on ``x^2`` for a symmetric
    ``x``; the hinges carry it."""
    if np.unique(values).size < 10:
        return values[:, None]
    knots = np.quantile(values, [0.25, 0.5, 0.75])
    return np.column_stack([values] + [np.maximum(values - q, 0.0) for q in knots])


def confounding_sensitivity(result: AIPWResult | None = None, *, outcome_effects,
                            imbalances, estimate=None, se=None, scores=None,
                            level=.95, output="auto"):
    """Bias grid for one omitted baseline confounder under an additive model.

    ``outcome_effects`` gives the change in the outcome caused by a one-unit
    increase in the omitted confounder, conditional on measured covariates.
    ``imbalances`` gives its adjusted treated-minus-control mean difference in
    the same confounder units. Their product is the assumed omitted-variable
    bias, so ``adjusted_estimate = estimate - outcome_effect * imbalance``.

    The estimate comes from one of two inputs:

    * ``result``, an :class:`AIPWResult`. Its estimate and, when available,
      its interval (the design-aware one, including cluster-robust intervals)
      are shifted by the bias.
    * ``estimate``, a plain number from any estimator, with ``se``, its
      standard error. The interval is ``estimate +/- z * se`` at ``level``
      with a normal critical value. Without ``se`` the rows carry no interval.
    * ``scores``, a vector of per-row scores whose mean is the estimate. The
      standard error is the IID one, ``sd(scores, ddof=1) / sqrt(n)``, unless
      ``se`` is also given. When ``estimate`` is given with ``scores``, the
      estimate is used as given.

    Pass exactly one of ``result`` and ``estimate``/``se``/``scores``.

    This is a transparent bias function, not identification or a calibrated
    bound. Choose ranges from subject-matter knowledge or benchmark them to
    measured covariates, and report the units. The confidence interval, when
    available, is shifted by the assumed bias and does not include uncertainty
    in the sensitivity parameters.

    ``output`` chooses the return type, as for the other frames of the
    package: ``"auto"`` (the default) is a Polars frame when Polars is
    installed and a pandas frame otherwise, ``"polars"`` and ``"pandas"`` force
    one, and ``"records"`` returns a list of dicts holding plain Python
    numbers, booleans and ``None``, which ``json.dumps`` accepts. The rows
    are one per (outcome effect, imbalance) pair.
    """
    from mboostrs import _rust

    if result is not None:
        if not isinstance(result, AIPWResult):
            raise TypeError("result must be an AIPWResult")
        if estimate is not None or se is not None or scores is not None:
            raise TypeError("pass either result or estimate/se/scores, not both")
        center = float(result.estimate)
        interval = result.interval
    else:
        if estimate is None and scores is None:
            raise TypeError(
                "pass result (an AIPWResult), or estimate= with se=, or scores=")
        if scores is not None:
            score_values = np.asarray(scores, dtype=np.float64).ravel()
            score_mean, score_se = _rust.mean_and_iid_se(score_values)
            if estimate is None:
                estimate = score_mean
            if se is None:
                se = score_se
        center = float(estimate)
        if not np.isfinite(center):
            raise ValueError("estimate must be finite")
        if se is None:
            interval = None
        else:
            se = float(se)
            if not np.isfinite(se) or se < 0:
                raise ValueError("se must be finite and non-negative")
            level = check_unit_interval("level", level)
            critical = NormalDist().inv_cdf(.5 + level / 2)
            interval = (center - critical * se, center + critical * se)
    effects = np.asarray(outcome_effects, dtype=np.float64).ravel()
    differences = np.asarray(imbalances, dtype=np.float64).ravel()
    if effects.size == 0 or differences.size == 0:
        raise ValueError("outcome_effects and imbalances must be non-empty")
    if not np.all(np.isfinite(effects)) or not np.all(np.isfinite(differences)):
        raise ValueError("outcome_effects and imbalances must be finite")
    rows = []
    for effect in effects:
        for imbalance in differences:
            bias = float(effect * imbalance)
            adjusted = center - bias
            row = {
                "outcome_effect": float(effect), "imbalance": float(imbalance),
                "assumed_bias": bias, "adjusted_estimate": adjusted,
                "changes_sign": bool(np.signbit(adjusted) != np.signbit(center)
                                     and adjusted != 0),
            }
            if interval is not None:
                row["lower"] = interval[0] - bias
                row["upper"] = interval[1] - bias
                row["interval_includes_zero"] = row["lower"] <= 0 <= row["upper"]
            else:
                row.update(lower=None, upper=None, interval_includes_zero=None)
            rows.append(row)
    return output_frame([json_safe(row) for row in rows], output)


def confounding_benchmark(data, result=None, *, variables=None,
                          folds=None, output="auto"):
    """Benchmark sensitivity by omitting one measured covariate at a time.

    Each row reruns the same AIPW configuration after deleting one declared
    baseline covariate. The observed estimate shift gives a scale for choosing
    sensitivity scenarios: for example, an assumed unmeasured bias twice the
    largest measured-covariate shift. It is not a bound. Omission can change
    overlap and the entire nuisance fit, measured covariates need not resemble
    an unmeasured one, and correlated measured covariates can mask each other's
    contribution.

    The canonical call is ``confounding_benchmark(data, result)``: the data
    frame first, then the :class:`AIPWResult`. The reversed order,
    ``confounding_benchmark(result, data)``, is also accepted; the two are told
    apart by type, so either order gives the same rows. (The companion
    :func:`confounding_sensitivity` takes the result first, because its
    estimate can instead be given as plain numbers.)

    Integer folds are reconstructed from the retained count and seed. If the
    original estimate used an explicit grouped or ordered ``FoldScheme``, pass
    that same object as ``folds`` so its sampling-unit semantics are preserved.
    Each refit uses the result's outcome model (the one ``"auto"`` resolved to),
    its treatment interactions, and its ``mstop`` and ``max_mstop``.

    A result fitted with ``propensity_terms`` or ``outcome_terms`` is refitted
    with the same formulas minus every term that names the omitted covariate;
    the rows report the reduced formulas in ``propensity_terms`` and
    ``outcome_terms`` (``None`` where the model used its default formula). When
    every term of a custom formula names the omitted covariate, the call refuses
    before any refit, naming the custom formula, because a refit with no
    formula is not a model.

    ``output`` chooses the return type, as for the other frames of the
    package: ``"auto"`` (the default) is a Polars frame when Polars is
    installed and a pandas frame otherwise, ``"polars"`` and ``"pandas"`` force
    one, and ``"records"`` returns a list of dicts holding plain Python
    numbers, booleans and ``None``, which ``json.dumps`` accepts. One row is
    returned per omitted covariate. A Polars frame's ``to_dict()`` returns
    Series; use ``output="records"`` for plain values.
    """
    if isinstance(data, AIPWResult) and not isinstance(result, AIPWResult):
        data, result = result, data
    if isinstance(data, AIPWResult):
        raise TypeError(
            "pass one AIPWResult and the data frame: confounding_benchmark(data, result)")
    if not isinstance(result, AIPWResult):
        raise TypeError(
            "result must be an AIPWResult; the call is confounding_benchmark(data, result)")
    available = result.spec.covariates
    selected = available if variables is None else tuple(variables)
    if not selected:
        raise ValueError("variables must name at least one measured covariate")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"benchmark variables are not in the adjustment set: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("benchmark variables must not contain duplicates")
    if len(available) < 2:
        raise ValueError(
            "leave-one-covariate-out benchmarking needs at least two covariates "
            "because the current nuisance formula requires a non-empty adjustment set"
        )
    if folds is None:
        if result.nuisance.get("fold_input") != "integer":
            raise ValueError(
                "the original estimate used an explicit FoldScheme; pass that same "
                "scheme as folds to preserve its design"
            )
        folds = result.nuisance["fold_count"]

    options = {
        "folds": folds,
        "repeats": result.nuisance["fold_repeats"],
        "seed": result.nuisance["seed"],
        "smooth": result.nuisance["smooth"],
        "df": result.nuisance["df"],
        "mstop": result.nuisance["mstop"],
        "max_mstop": result.nuisance["max_mstop"],
        "tune": result.nuisance["tuned"],
        # The resolved model, not the request: under "auto" the check on the
        # reduced covariate set could pick another model than the full set did.
        "outcome_model": result.nuisance["outcome_model"],
        "outcome_interactions": result.nuisance["outcome_interactions"],
        "inner_folds": result.nuisance["inner_folds"] or 5,
        "level": result.level,
        "propensity_clip": result.diagnostics["propensity_clip"],
        "target": result.diagnostics.get("target", "all"),
        "outcome_missing": result.nuisance.get("outcome_missing", "error"),
        "weights": result.weights,
    }
    # Custom formulas are reduced before any fit, so a refusal names the custom
    # terms and no refit has run when it is raised.
    custom = _custom_term_texts(result)
    plans = []
    for variable in selected:
        reduced = tuple(name for name in available if name != variable)
        terms = {}
        for model, text in custom.items():
            terms[f"{model}_terms"] = _without_variable(text, variable, model)
        plans.append((variable, reduced, terms))
    rows = []
    for variable, reduced, terms in plans:
        reduced_spec = replace(result.spec, covariates=reduced)
        estimate = aipw(data, reduced_spec, **options, **terms)
        shift = float(estimate.estimate - result.estimate)
        rows.append({
            "omitted_covariate": variable,
            "full_estimate": result.estimate,
            "omitted_estimate": estimate.estimate,
            "estimate_shift": shift,
            "absolute_shift": abs(shift),
            "shift_in_full_standard_errors": (
                abs(shift) / result.standard_error
                if result.standard_error not in (None, 0) else None
            ),
            "omitted_propensity_min": estimate.diagnostics["propensity_min"],
            "omitted_propensity_max": estimate.diagnostics["propensity_max"],
            "propensity_terms": terms.get("propensity_terms"),
            "outcome_terms": terms.get("outcome_terms"),
            "benchmark": "leave_one_measured_covariate_out",
        })
    return output_frame([json_safe(row) for row in rows], output)


def _custom_term_texts(result):
    """The custom right-hand sides a result was fitted with, by nuisance model.

    ``aipw(propensity_terms=, outcome_terms=)`` records which models got a custom
    formula in ``nuisance["custom_terms"]`` and the formula as fitted in
    ``nuisance["terms"]``. A pooled outcome formula has the treatment appended, so
    the suffix is removed to give back the string the user passed.
    """
    flags = result.nuisance.get("custom_terms") or {}
    fitted = result.nuisance.get("terms") or {}
    texts = {}
    if flags.get("propensity") and fitted.get("propensity"):
        texts["propensity"] = fitted["propensity"]
    if flags.get("outcome") and fitted.get("outcome"):
        text = fitted["outcome"]
        suffix = f" + {result.spec.treatment}"
        if text.endswith(suffix):
            text = text[: -len(suffix)]
        texts["outcome"] = text
    return texts


def _without_variable(text, variable, model):
    """``text`` with every top-level term that names ``variable`` removed.

    A term names a variable when the variable's name appears in it as a whole
    identifier, so ``bbs(age, df=4)`` names ``age`` and ``bols(z, by=age)`` names
    both. Raises when no term is left, naming the custom formula: a refit with an
    empty right-hand side is not a model.
    """
    pattern = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(variable)}(?![A-Za-z0-9_.])")
    kept = [term for term in _top_level_terms(text) if not pattern.search(term)]
    if not kept:
        argument = f"{model}_terms"
        raise ValueError(
            f"confounding_benchmark cannot omit {variable!r}: the {model} model of this "
            f"result uses the custom {argument}={text!r}, and every term of it names "
            f"{variable!r}, so the refit would have no {model} formula. Drop {variable!r} "
            f"from the benchmark variables, or refit by hand with aipw(..., {argument}=...) "
            "on the covariates that remain.")
    return " + ".join(kept)


def negative_control(data, spec: CausalSpec, *, kind, rationale,
                     expected_effect=0.0, **aipw_options):
    """Estimate a prespecified negative-control exposure or outcome effect.

    ``spec`` is the complete causal specification for the control analysis,
    including its treatment contrast, outcome type, timing and adjustment set.
    ``rationale`` must state the scientific reason its effect should equal
    ``expected_effect``. For example, a negative-control outcome must share
    relevant measurement/confounding pathways while being causally unaffected
    by treatment; merely choosing an unrelated variable is uninformative.

    An incompatible interval can reveal residual confounding, selection,
    measurement error or a failed control assumption. A compatible interval
    cannot prove the primary effect is identified because the control may not
    probe the relevant bias pathway.
    """
    if not isinstance(spec, CausalSpec):
        raise TypeError("spec must be a CausalSpec for the negative-control analysis")
    if kind not in ("exposure", "outcome"):
        raise ValueError("kind must be 'exposure' or 'outcome'")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must explain why the control effect should be null")
    expected_effect = float(expected_effect)
    if not np.isfinite(expected_effect):
        raise ValueError("expected_effect must be finite")
    effect = aipw(data, spec, **aipw_options)
    return NegativeControlResult(kind, rationale, expected_effect, effect)


#: First-stage partial F below which :func:`iv_effect` reports a weak instrument.
#: 10 is the rule of thumb of Staiger and Stock (1997) for one instrument; it
#: is a screen, not a test.
IV_WEAK_F = 10.0

#: The assumptions an IV estimate rests on, stated in every :class:`IVResult`.
_IV_ASSUMPTIONS = (
    "relevance: the instrument moves the probability of treatment after the covariates",
    "independence: the instrument shares no unmeasured cause with the outcome, given the covariates",
    "exclusion: the instrument affects the outcome only through the treatment, given the covariates",
    "monotonicity: no unit is moved out of treatment by the instrument, so the units it moves all move the same way",
    "consistency: each observed outcome equals the potential outcome under the observed treatment and instrument value",
)


@dataclass(frozen=True)
class IVResult:
    """A cross-fitted instrumental-variable estimate, its interval and its first-stage diagnostics."""

    treatment: str
    outcome: str
    instrument: str
    covariates: tuple
    estimate: float
    standard_error: float
    level: float
    interval: tuple
    first_stage_r2: float
    first_stage_f: float
    weak_threshold: float
    weak_instrument: bool
    n: int
    folds: int
    seed: int
    nuisance: dict
    influence: np.ndarray
    outcome_residual: np.ndarray
    treatment_residual: np.ndarray
    instrument_residual: np.ndarray
    fold: np.ndarray
    assumptions: tuple = _IV_ASSUMPTIONS

    def __post_init__(self):
        if self.n < 3:
            raise ValueError(
                f"an IV estimate needs at least three rows; this result has {self.n}")
        for name in ("influence", "outcome_residual", "treatment_residual",
                     "instrument_residual", "fold"):
            shape = np.shape(getattr(self, name))
            if shape != (self.n,):
                raise ValueError(f"{name} has shape {shape}, expected ({self.n},) for "
                                 f"{self.n} rows")

    def __repr__(self) -> str:
        return (
            f"IVResult(estimate={self.estimate:.6g}, se={self.standard_error:.6g}, "
            f"first_stage_f={self.first_stage_f:.4g}, weak_instrument={self.weak_instrument})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row residuals and influence values only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("influence", "outcome_residual", "treatment_residual",
                    "instrument_residual", "fold"),
            extras={"n_folds": self.folds},
        )

    def to_frame(self, output="auto"):
        """One row per observation: its fold, its three residuals and its influence value."""
        records = [
            {
                "row": int(i), "fold": int(self.fold[i]),
                "outcome_residual": float(self.outcome_residual[i]),
                "treatment_residual": float(self.treatment_residual[i]),
                "instrument_residual": float(self.instrument_residual[i]),
                "influence": float(self.influence[i]),
            }
            for i in range(self.n)
        ]
        return output_frame(records, output)

    def describe(self) -> str:
        """The estimand, the estimate with its interval, the first stage and the assumptions."""
        lower, upper = self.interval
        text = (
            f"Partially linear IV (DML-IV) of {self.treatment} on {self.outcome}, "
            f"instrument {self.instrument}, adjusting for {list(self.covariates)}; "
            f"{self.n} rows, {self.folds}-fold cross-fitting. "
            "Estimand: a weighted average of the effect of "
            f"{self.treatment} among compliers, the units whose {self.treatment} "
            "the instrument switches on, under monotonicity; with an effect that "
            "is the same for every unit, that effect. "
            f"Estimate {self.estimate:.6g}, {self.level:.0%} interval "
            f"[{lower:.6g}, {upper:.6g}] (influence-function standard error "
            f"{self.standard_error:.6g}). "
            f"First stage: partial R2 {self.first_stage_r2:.4g}, partial F "
            f"{self.first_stage_f:.4g}. "
        )
        if self.weak_instrument:
            text += (
                f"WEAK INSTRUMENT: the partial F is below {self.weak_threshold:g}. "
                "The estimate and its interval can be badly biased and the interval "
                "too narrow; use a stronger instrument or a larger sample. "
            )
        else:
            text += (f"The partial F is at or above {self.weak_threshold:g}, the rule "
                     "of thumb for a strong instrument. ")
        if self.nuisance.get("any_at_max_mstop"):
            text += ("A nuisance step search reached max_mstop with its risk still "
                     "falling; raise max_mstop. ")
        n_missing = self.nuisance.get("n_missing_outcome") or 0
        if n_missing:
            text += (f"The outcome is missing for {n_missing} of {self.n} rows; those rows "
                     "carry weight zero, and the observed rows are weighted by the inverse "
                     "of a fitted response model for P(outcome observed | covariates, "
                     "treatment, instrument), which assumes the outcome is missing at random "
                     "given those columns. ")
        text += "Assumptions the data cannot check: " + "; ".join(self.assumptions) + "."
        return text


def _iv_nuisance_fit(alias, train_data, rhs, family, values_test, *, smooth, mstop, max_mstop,
                     tune, inner_folds, seed, weights=None):
    """Fit one nuisance model on the training rows and predict the held-out rows.

    A nuisance that is close to a constant (an instrument nearly unrelated to the
    covariates) has a flat risk curve, which is the expected state of a residual
    regression, so tune_mstop's flat-curve note is switched off here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="tune_mstop: the chosen mstop")
        model, info = _fit_nuisance(alias, train_data, rhs, family, smooth=smooth,
                                    mstop=mstop, max_mstop=max_mstop, tune=tune,
                                    inner_folds=inner_folds, seed=seed, weights=weights)
    return model.predict(newdata=values_test, type="response"), info


def iv_effect(data, spec: CausalSpec, instrument, *, folds=5, seed=0, smooth=False, df=4,
              mstop=200, max_mstop=None, tune=True, inner_folds=5, level=.95,
              weak_threshold=IV_WEAK_F, outcome_missing=FROM_SPEC, propensity_clip=FROM_SPEC):
    """Cross-fitted partially linear instrumental-variable estimate for a binary treatment.

    The model is ``Y = theta D + g(X) + U`` with a binary treatment ``D``,
    baseline covariates ``X`` and an instrument ``Z`` that is related to ``D``
    but reaches ``Y`` only through ``D``. Three boosted nuisance models are
    fitted out of fold on the covariates: the outcome mean ``E[Y|X]``, the
    treatment mean ``E[D|X]`` and the instrument mean ``E[Z|X]``. Each
    observation's residuals are taken from the fold that did not fit them
    (``Y - m(X)``, ``D - e(X)``, ``Z - r(X)``), and the estimate is the ratio of
    residual covariances, ``sum(Z~ Y~) / sum(Z~ D~)`` (DML-IV, Chernozhukov et
    al., 2018). With the covariates linear, this is the two-stage least squares
    estimate of ``theta`` on the covariates.

    The estimand is a weighted average of the effect of ``D`` among compliers,
    the units whose treatment the instrument switches on, under monotonicity.
    With an effect that does not vary across units it is that effect. The
    estimand is the complier effect whatever ``spec.estimand`` says; the field
    is not used.

    A missing outcome is handled as in :func:`aipw`: with
    ``outcome_missing="ipw"`` a response model for ``P(outcome observed | X, D, Z)``
    is cross-fitted on the same folds, the outcome mean is fitted on the observed
    rows weighted by ``1 / pi``, and each observed row's moment is multiplied by
    ``1 / pi``, where ``pi`` is clipped to ``propensity_clip``. This assumes the
    outcome is missing at random given the covariates, the treatment and the
    instrument. Rows with a missing outcome get weight zero.

    Uncertainty is the sandwich variance from the influence function
    ``W Z~ (Y~ - theta D~) / mean(W Z~ D~)``, with ``W`` the inverse response
    weight (one when every outcome is observed) and a normal interval at
    ``level``. It treats the nuisance predictions as fixed and the rows as
    independent draws, and it does not include uncertainty from the choice of
    fold split.

    The first stage is the squared correlation of ``Z~`` and ``D~``, the partial
    R2 of the treatment on the instrument after the covariates, and the partial
    F ``n R2 / (1 - R2)`` from the same residuals. When F is below
    ``weak_threshold`` (10 by default) the result is marked
    ``weak_instrument`` and a RuntimeWarning is raised: a weak instrument makes
    the ratio biased toward the confounded regression and the interval too
    narrow.

    Parameters
    ----------
    data : mapping or data frame
        Treatment, outcome, instrument and covariate columns. The outcome must
        be numeric; it may hold NaN or None only under ``outcome_missing="ipw"``.
    spec : CausalSpec
        Names the treatment (binary, read through ``treatment_contrast``), the
        outcome and the covariates. ``adjustment_rationale``, ``time_zero`` and
        ``follow_up`` are required as for every causal spec, and the
        ``estimand`` field is not used.
    instrument : str
        The instrument column. It may be continuous or binary (0/1). A
        categorical instrument is refused.
    folds : int or FoldScheme
        Cross-fitting folds, stratified on the treatment. A FoldScheme must hold
        every row out exactly once.
    seed : int
        Seed of the fold split and of the inner step-count searches.
    smooth, df, mstop, max_mstop, tune, inner_folds
        Nuisance-model options with the meanings they have in :func:`aipw`:
        ``smooth`` (default False) fits linear terms only; ``smooth=True`` fits a
        penalized spline for each numeric covariate with at least ten distinct
        values and a linear term for the others. ``mstop`` is the first step grid
        of each nuisance and ``tune`` chooses its step count by inner
        cross-validation.
    level : float
        Coverage of the interval, between 0 and 1.
    weak_threshold : float
        The first-stage partial F below which the instrument is reported weak.
    outcome_missing : {"error", "ipw"}, optional
        What a missing outcome (NaN) means. ``"error"`` refuses a missing outcome
        and names this argument. ``"ipw"`` keeps every row through the inverse
        response weighting described above. When the call does not pass it, the
        ``outcome_missing`` of the :class:`CausalSpec` is used.
    propensity_clip : tuple or None, optional
        Bounds for the response probability ``pi`` under ``outcome_missing="ipw"``.
        When the call does not pass it, the ``propensity_clip`` of the spec is used.

    Returns
    -------
    IVResult
        ``estimate``, ``standard_error``, ``interval``, the first-stage
        ``first_stage_r2`` and ``first_stage_f``, ``weak_instrument``, the
        per-row residuals and influence values, and the nuisance step counts.
    """
    if not isinstance(spec, CausalSpec):
        raise TypeError("spec must be a CausalSpec")
    if not isinstance(instrument, str) or not instrument:
        raise TypeError("instrument must be the name of one column")
    if instrument in (spec.treatment, spec.outcome, *spec.covariates):
        raise ValueError(
            f"the instrument {instrument!r} must be a column distinct from the treatment, "
            "the outcome and the covariates")
    from mboostrs import _rust
    from mboostrs.families import Binomial, Gaussian

    mstop = check_int_at_least("mstop", mstop, 1, allow_none=False)
    if max_mstop is None:
        max_mstop = max(_MAX_MSTOP_DEFAULT, mstop)
    else:
        max_mstop = check_int_at_least("max_mstop", max_mstop, 1, allow_none=False)
        if max_mstop < mstop:
            raise ValueError(
                f"max_mstop={max_mstop} is below mstop={mstop}; pass max_mstop >= mstop")
    inner_folds = check_int_at_least("inner_folds", inner_folds, 2, allow_none=False)
    level = check_unit_interval("level", level)
    weak_threshold = check_positive("weak_threshold", weak_threshold)
    outcome_missing = _check_outcome_missing(
        _spec_default(spec, "outcome_missing", outcome_missing))
    clip = _check_clip(_spec_default(spec, "propensity_clip", propensity_clip))

    values = _columns(data, spec, outcome_missing=outcome_missing, categorical=True,
                      owner="iv_effect")
    smooth_names = _smooth_baselines(values, spec.covariates)
    rhs = _rhs(spec.covariates, smooth, df, smooth_names=smooth_names)

    z_column = read_column(data, instrument)
    if z_column.levels is not None and z_column.kind != "boolean":
        raise TypeError(
            f"the instrument {instrument!r} is categorical; pass a numeric column, or a "
            "0/1 column for a binary instrument")
    if z_column.n_missing or not np.all(np.isfinite(np.asarray(z_column.values, dtype=float))):
        raise ValueError(
            f"the instrument {instrument!r} has missing or non-finite values; iv_effect "
            "does not drop rows. Remove them from the data before the call.")
    z = np.asarray(z_column.values, dtype=np.float64)
    if np.unique(z).size < 2:
        raise ValueError(f"the instrument {instrument!r} is constant; it cannot move the treatment")
    binary_instrument = bool(np.all(np.isin(z, (0.0, 1.0))))

    control_label, treated_label = spec.treatment_contrast
    raw_d = np.asarray(data[spec.treatment])
    if not np.all(np.isin(raw_d, (control_label, treated_label))):
        raise ValueError("treatment contains values outside treatment_contrast")
    d = (raw_d == treated_label).astype(np.float64)
    if d.sum() == 0 or d.sum() == d.size:
        raise ValueError("both treatment levels must be observed")
    y = values[spec.outcome]
    observed = (~np.isnan(y)).astype(np.float64)
    if spec.outcome_type == "binary" and not np.all(np.isin(y[observed == 1], (0, 1))):
        raise ValueError("a binary causal outcome must be coded with values 0 and 1")
    n = d.size
    use_response = outcome_missing == "ipw" and bool(np.any(observed == 0))
    # Missing outcomes carry no value into the residual products; their rows
    # get weight zero, so the filled value does not reach the estimate.
    y_filled = np.where(observed == 1, y, 0.0)
    scheme, held = _crossfit_folds(d, folds, seed)

    outcome_family = Binomial() if spec.outcome_type == "binary" else Gaussian()
    instrument_family = Binomial() if binary_instrument else Gaussian()
    residual_outcome = np.empty(n)
    residual_treatment = np.empty(n)
    residual_instrument = np.empty(n)
    fold_of = np.empty(n, dtype=np.int64)
    records = {"outcome": [], "treatment": [], "instrument": [], "response": []}
    # The alias names keep the nuisance formulas clear of every declared column.
    aliases = {"outcome": "__iv_outcome", "treatment": "__iv_treatment",
               "instrument": "__iv_instrument", "response": "__iv_response"}
    targets = {"outcome": (y_filled, outcome_family), "treatment": (d, Binomial()),
               "instrument": (z, instrument_family)}
    # Pass 1 (missing outcomes only): the response model P(observed | X, D, Z),
    # cross-fitted on the same folds as the nuisances, with the treatment and the
    # instrument as linear terms.
    response_pi = np.ones(n)
    if use_response:
        rhs_response = _rhs(spec.covariates, smooth, df, smooth_names=smooth_names,
                            linear=(spec.treatment, instrument))
        for b in range(scheme.n_folds):
            train = np.asarray(scheme.weights[:, b]) > 0
            test = held[:, b]
            response_data = {
                aliases["response"]: observed[train],
                spec.treatment: d[train],
                instrument: z[train],
                **{name: values[name][train] for name in spec.covariates},
            }
            response, info = _fit_nuisance(
                aliases["response"], response_data, rhs_response, Binomial(),
                smooth=smooth, mstop=mstop, max_mstop=max_mstop, tune=tune,
                inner_folds=inner_folds, seed=seed + 1000 + b)
            records["response"].append(info)
            response_pi[test] = _nuisance_predict(
                response, {**{name: values[name][test] for name in spec.covariates},
                           spec.treatment: d[test], instrument: z[test]}, "warn")
        if clip is not None:
            response_pi = np.clip(response_pi, clip[0], clip[1])
        if np.any(response_pi <= 0):
            raise ValueError("the fitted response probability is zero for some rows; "
                             "set propensity_clip to a positive lower bound")
    # Inverse response weight of each row: 1 / pi when the outcome is observed, 0 otherwise.
    if use_response:
        inverse_weight = np.where(observed == 1, 1.0 / response_pi, 0.0)
    else:
        inverse_weight = np.ones(n)
    for b in range(scheme.n_folds):
        train = np.asarray(scheme.weights[:, b]) > 0
        test = held[:, b]
        if not np.any(test):
            raise ValueError(f"fold {b} has no evaluation rows")
        test_data = {name: values[name][test] for name in spec.covariates}
        for k, key in enumerate(("outcome", "treatment", "instrument")):
            target, family = targets[key]
            rows = train
            weights = None
            if key == "outcome" and use_response:
                # The outcome mean is fitted on the observed training rows, weighted by 1 / pi.
                rows = train & (observed == 1)
                weights = inverse_weight[rows]
            train_data = {aliases[key]: target[rows],
                          **{name: values[name][rows] for name in spec.covariates}}
            prediction, info = _iv_nuisance_fit(
                aliases[key], train_data, rhs, family, test_data, smooth=smooth,
                mstop=mstop, max_mstop=max_mstop, tune=tune, inner_folds=inner_folds,
                seed=seed + 3 * b + k, weights=weights)
            records[key].append(info)
            residual = target[test] - np.asarray(prediction, dtype=np.float64).ravel()
            if key == "outcome":
                residual_outcome[test] = residual
            elif key == "treatment":
                residual_treatment[test] = residual
            else:
                residual_instrument[test] = residual
        fold_of[test] = b

    estimate, variance, influence, r2, f_stat, _used = _rust.iv_ratio_weighted(
        np.ascontiguousarray(residual_outcome), np.ascontiguousarray(residual_treatment),
        np.ascontiguousarray(residual_instrument), np.ascontiguousarray(inverse_weight))
    standard_error = float(np.sqrt(max(variance, 0.0)))
    critical = NormalDist().inv_cdf(.5 + level / 2)
    interval = (float(estimate - critical * standard_error),
                float(estimate + critical * standard_error))
    weak = bool(f_stat < weak_threshold)
    if weak:
        warnings.warn(
            f"iv_effect: the first-stage partial F is {f_stat:.4g}, below {weak_threshold:g}; "
            f"the instrument {instrument!r} is weak. The IV estimate can be badly biased "
            "and its interval too narrow. Use a stronger instrument, or check the "
            "first stage before reading the estimate.",
            RuntimeWarning, stacklevel=2)
    any_at_max = any(info["at_max_mstop"] for stage in records.values() for info in stage)
    if any_at_max:
        warnings.warn(
            "iv_effect: a nuisance step search reached max_mstop with its risk still "
            "falling; raise max_mstop to let it converge.", RuntimeWarning, stacklevel=2)
    return IVResult(
        treatment=spec.treatment, outcome=spec.outcome, instrument=instrument,
        covariates=tuple(spec.covariates), estimate=float(estimate),
        standard_error=standard_error, level=float(level), interval=interval,
        first_stage_r2=float(r2), first_stage_f=float(f_stat),
        weak_threshold=float(weak_threshold), weak_instrument=weak, n=int(n),
        folds=int(scheme.n_folds), seed=int(seed),
        nuisance={"any_at_max_mstop": bool(any_at_max), "steps": records,
                  "binary_instrument": binary_instrument, "tuned": bool(tune),
                  "outcome_missing": outcome_missing,
                  "n_missing_outcome": int(n - observed.sum()),
                  "response_model": "fitted" if use_response else None,
                  "response_propensity_min": float(response_pi.min()) if use_response else None,
                  "response_propensity_max": float(response_pi.max()) if use_response else None},
        influence=np.asarray(influence), outcome_residual=residual_outcome,
        treatment_residual=residual_treatment, instrument_residual=residual_instrument,
        fold=fold_of)
