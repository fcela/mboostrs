"""Held-out gap between a formula and a flexible tree reference, and its sources.

:func:`reference_gap` fits two boosted models on the same held-out rows: the
specification the caller wrote, and a reference built from mboostrs's own
``btree`` learner over every predictor column of the data (``reference_columns``
selects the formula's covariates only, or an explicit list), plus the lagged
and rolling covariates when the rows are time-ordered. Each model is tuned to
its own best step count on the folds. The mean difference of the fold losses,
with its standard error across folds, is the gap, and ``material`` reports
whether it exceeds two standard errors.

The function also ranks terms that could close the gap: interaction pairs of
the formula's covariates, columns the formula omits, and, for time-ordered
data, lagged and rolling covariates. The ranking uses the family loss: the
screen is run on the negative gradient of the family's loss at the
specification's out-of-fold predictions, scaled by the square root of the
loss's curvature, which is the ordinary residual for a Gaussian fit. The top
``top`` candidates are each added to the specification and refitted once,
whether or not the gap is material, so every candidate has a measured
held-out gain.
"""

from __future__ import annotations

import dataclasses
import threading
import time as _time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import polars as pl

from mboostrs._names import partition_tilde, quote_name, quote_response

#: A gap is material when it is larger than this many standard errors.
MATERIAL_SE = 2.0

#: The fraction of ``max_mstop`` at which a best step count counts as "at the
#: cap". It matches the warning ``gamboost`` gives for the same condition.
CAP_SHARE = 0.95

_REFERENCES = ("trees",)


@dataclass
class ReferenceGapResult:
    """Result of comparing a specification with a tree reference, including candidate revisions.

    Losses are the family's risk per held-out row, on the scale the family
    fits (the link scale for the log1p target of a quantile model, for example).
    A positive ``gap`` means the reference has the lower loss.

    Attributes
    ----------
    gap : float
        Mean over folds of ``model loss - reference loss`` at each model's own
        best step count. Positive when the reference is better.
    gap_se : float
        Standard error of ``gap`` across folds.
    material : bool
        Whether ``gap > 2 * gap_se``.
    relative_gain : float
        ``gap`` as a share of the specification's loss above the saturated
        floor (see ``_risk_floor``). Zero for families whose floor is zero.
    reference_loss, model_loss : float
        Fold-aggregated held-out loss of the reference and the specification.
    model_formula, reference_formula : str
        The specification, and the reference ``btree`` formula.
    model_mstop, reference_mstop : int
        Best step count of each model on the folds.
    model_at_cap, reference_at_cap : bool
        Whether the best step count sits within 5% of ``max_mstop``. Such a
        value is unfinished tuning: the loss was still falling at the cap.
    reference_in_sample_loss : float
        Loss of the reference fitted at ``reference_mstop`` on all analysis
        rows and scored on the same rows. Compare with ``reference_loss``.
    candidates : polars.DataFrame
        The top candidate terms in rank order, one row each. Columns:
        ``rank``, ``kind`` (``"interaction"``, ``"column"`` for a column the
        formula omits, ``"lag"`` or ``"user"``), ``term``,
        ``source``, ``screen_share`` (out-of-sample share of the mean square of
        the screening residual that the term removes in the screen; the
        residual is described under ``screen_residual``), ``measured_gain``
        and ``gain_se`` (held-out gain of the specification plus this term,
        over the specification, with its fold standard error),
        ``relative_gain``, ``closed_share`` (``measured_gain / gap``),
        ``mstop`` and ``at_cap``. Candidates are measured whether or not the
        gap is material; ``closed_share`` is meaningful only when it is.
    closed_share : float or None
        Share of the gap closed by adding every candidate with a positive
        measured gain together, in one refit. ``None`` when the gap is not
        material or no candidate was measured. ``combined_gain`` is still
        reported when the gap is not material.
    combined_formula : str or None
        The specification plus those candidates.
    combined_gain, combined_gain_se : float or None
        Held-out gain of ``combined_formula`` over the specification.
    folds : int
        Number of fold columns the gap rests on.
    family : str
        Name of the family the models were fitted with.
    notes : list of str
        Plain statements about the checks that need attention, such as an
        overfit reference, a gap resting on fewer than three folds, or a
        candidate pair that cannot be turned into a term.
    data : polars.DataFrame
        The rows the models were fitted on: time-ordered when ``time=`` was
        given, with the lag and rolling columns the candidates refer to.
    fold_losses : dict
        ``"model"`` and ``"reference"``: the held-out loss of each fold, at
        each model's best step count. ``gap`` is the mean of their difference.
    reference_columns : list of str
        The columns the reference ``btree`` was given, including lag and
        rolling features built with ``time=``.
    reference_extra_columns : list of str
        The members of ``reference_columns`` that the formula does not use.
        A gap that appears with these columns and not without them points at
        information the specification leaves out.
    screen_residual : str
        ``"family"`` when the candidate screen ran on the scaled negative
        gradient of the family loss, ``"response"`` when it ran on
        ``y - mean`` (Gaussian fits, and families whose loss couples rows), or
        ``"none"`` when no screen ran.
    """

    gap: float
    gap_se: float
    material: bool
    relative_gain: float
    reference_loss: float
    model_loss: float
    model_formula: str
    reference_formula: str
    model_mstop: int
    reference_mstop: int
    model_at_cap: bool
    reference_at_cap: bool
    reference_in_sample_loss: float
    candidates: pl.DataFrame
    closed_share: float | None
    combined_formula: str | None
    combined_gain: float | None
    combined_gain_se: float | None
    folds: int
    family: str
    notes: list[str] = field(default_factory=list)
    data: pl.DataFrame | None = field(default=None, repr=False)
    #: Held-out loss of each fold for the specification (``"model"``) and the
    #: reference (``"reference"``), at each model's best step count.
    fold_losses: dict = field(default_factory=dict, repr=False)
    reference_columns: list = field(default_factory=list)
    reference_extra_columns: list = field(default_factory=list)
    screen_residual: str = "none"

    def __repr__(self) -> str:
        return (f"ReferenceGapResult(gap={self.gap:.4g}, gap_se={self.gap_se:.4g}, "
                f"material={self.material}, candidates={self.candidates.height}, "
                f"closed_share={self.closed_share})")

    def describe(self) -> str:
        """Plain-text summary: the gap, its source and what to try next."""
        if self.gap >= 0:
            direction = (f"The reference's loss is lower by {self.gap:.6g} "
                         f"(SE {self.gap_se:.6g}), a relative gain of "
                         f"{100 * self.relative_gain:.1f}%.")
        else:
            direction = (f"The reference's loss is higher by {-self.gap:.6g} "
                         f"(SE {self.gap_se:.6g}): the specification is ahead of the reference "
                         f"by {-self.relative_gain * 100:.1f}% of its loss.")
        lines = [
            f"Held-out loss ({self.family}, {self.folds} folds): specification "
            f"{self.model_loss:.6g}, tree reference {self.reference_loss:.6g}. " + direction,
        ]
        if self.reference_extra_columns:
            lines.append("The reference also saw columns the formula does not use: "
                         + ", ".join(self.reference_extra_columns) + ".")
        else:
            lines.append("The reference saw only columns the formula uses.")
        lines.append(f"Reference overfit check: its held-out loss is {self.reference_loss:.6g} "
                     f"and its loss on the rows it was fitted to is "
                     f"{self.reference_in_sample_loss:.6g}, at {self.reference_mstop} steps.")
        if not np.isfinite(self.gap_se):
            lines.append("The gap has no standard error, so it cannot be judged. "
                         "Use more folds, or check the notes.")
        elif self.material:
            lines.append(f"The gap is above {MATERIAL_SE:g} standard errors, so the "
                         "specification leaves structure that the tree reference finds.")
        else:
            lines.append(f"The gap is not above {MATERIAL_SE:g} standard errors. The "
                         "specification is as good as the tree reference on these folds.")
        if self.model_at_cap:
            lines.append(f"The specification's best step count is at the search cap "
                         f"({self.model_mstop}). Its tuning is unfinished: raise max_mstop.")
        if self.reference_at_cap:
            lines.append(f"The reference's best step count is at the search cap "
                         f"({self.reference_mstop}). Its tuning is unfinished: raise max_mstop.")
        if self.candidates.height:
            lines.append("Candidate terms, each added alone to the specification "
                         "(measured whether or not the gap is material):")
            for row in self.candidates.iter_rows(named=True):
                closed = ""
                if self.material and row["closed_share"] is not None:
                    closed = f", closes {100 * row['closed_share']:.0f}% of the gap"
                gain, se = row["measured_gain"], row["gain_se"]
                clear = (" This is above 2 standard errors."
                         if np.isfinite(se) and se > 0 and gain > MATERIAL_SE * se else "")
                lines.append(f"  {row['rank']}. {row['term']} ({row['kind']}): held-out gain "
                             f"{gain:.6g} (SE {se:.6g}){closed}.{clear}")
            if self.material and self.closed_share is not None:
                lines.append(f"Adding every candidate with a positive gain together closes "
                             f"{100 * self.closed_share:.0f}% of the gap "
                             f"(gain {self.combined_gain:.6g}, SE {self.combined_gain_se:.6g}).")
        if self.material and self.candidates.height:
            lines.append("Next: add the candidates with a positive gain to the formula, "
                         "refit, and run reference_gap again on the new formula to see what "
                         "gap is left.")
        elif self.material:
            lines.append("Next: no candidate could be measured. Check the notes, then try a "
                         "different reference depth or a longer path.")
        elif self.candidates.height:
            lines.append("Next: the gap does not ask for a change to the specification. "
                         "Add a candidate only if its measured gain is above 2 standard "
                         "errors.")
        else:
            lines.append("Next: nothing in this check asks for a change to the specification.")
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        """The result as JSON-ready data. The fitted models are not included.

        ``include_arrays=True`` adds ``"fold_losses"``: the per-fold held-out
        loss of the specification and of the reference.
        """
        from mboostrs.discover import _plain

        record = {
            "gap": self.gap,
            "gap_se": self.gap_se,
            "material": self.material,
            "relative_gain": self.relative_gain,
            "reference_loss": self.reference_loss,
            "model_loss": self.model_loss,
            "model_formula": self.model_formula,
            "reference_formula": self.reference_formula,
            "model_mstop": self.model_mstop,
            "reference_mstop": self.reference_mstop,
            "model_at_cap": self.model_at_cap,
            "reference_at_cap": self.reference_at_cap,
            "reference_in_sample_loss": self.reference_in_sample_loss,
            "candidates": self.candidates.to_dicts(),
            "closed_share": self.closed_share,
            "combined_formula": self.combined_formula,
            "combined_gain": self.combined_gain,
            "combined_gain_se": self.combined_gain_se,
            "folds": self.folds,
            "family": self.family,
            "notes": self.notes,
            "reference_columns": list(self.reference_columns),
            "reference_extra_columns": list(self.reference_extra_columns),
            "screen_residual": self.screen_residual,
        }
        if include_arrays:
            record["fold_losses"] = {name: [float(v) for v in values]
                                     for name, values in self.fold_losses.items()}
        return _plain(record)


# ---------------------------------------------------------------------------
# Inputs


def _split_formula(formula: str) -> tuple[str, str]:
    if "~" not in formula:
        raise ValueError(f"formula {formula!r} has no '~'. Write it as 'y ~ x + ...'.")
    lhs, rhs = (lambda _p: (_p[0], _p[2]))(partition_tilde(formula))
    return lhs.strip(), rhs.strip()


def _covariates(formula: str, response: str, frame: pl.DataFrame) -> list[str]:
    """Columns the formula's terms read, in the order they first appear."""
    from mboostrs.explore import split_terms
    from mboostrs.formula import term_variables

    _, rhs = _split_formula(formula)
    names: list[str] = []
    for term in split_terms(rhs):
        for name in term_variables(term):
            if name != response and name not in names:
                names.append(name)
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise ValueError(
            f"the formula uses {absent}, which are not columns of data. Add them to the "
            "data, or remove the terms that use them.")
    if not names:
        raise ValueError("the formula has no covariates. Add at least one predictor "
                         "to the right-hand side.")
    return names


def _reference_columns(frame: pl.DataFrame, spec, covariates: list[str], response: str,
                       time: str | None, exclude: list[str], notes: list[str]) -> list[str]:
    """Columns the reference sees besides the formula's covariates.

    ``spec`` is ``"all"`` (every predictor column of ``frame``), ``"formula"``
    (none) or a list of column names. The response, the time column and the
    caller's ``exclude`` columns are never added. Columns the tree cannot use
    (dates, missing values, a single value, strings with more than 50 levels)
    are skipped with a note.
    """
    if isinstance(spec, str):
        if spec == "formula":
            return []
        if spec != "all":
            raise ValueError("reference_columns must be 'all', 'formula' or a list of "
                             f"column names, got {spec!r}")
        wanted = list(frame.columns)
        named = False
    else:
        wanted = [str(c) for c in spec]
        absent = [c for c in wanted if c not in frame.columns]
        if absent:
            raise ValueError(f"reference_columns {absent} are not columns of data")
        named = True
    skip = {response, *exclude}
    if time is not None:
        skip.add(time)
    extra: list[str] = []
    dropped: list[str] = []
    for name in wanted:
        if name in covariates or name in skip or name in extra:
            continue
        column = frame[name]
        dtype = column.dtype
        unusable = (column.null_count() > 0 or column.drop_nulls().n_unique() < 2
                    or dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time, pl.Object)
                    or (dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum))
                    and column.n_unique() > 50
                    or not (dtype.is_numeric() or dtype == pl.Boolean
                            or dtype in (pl.String, pl.Categorical)
                            or isinstance(dtype, pl.Enum)))
        if unusable:
            dropped.append(name)
        else:
            extra.append(name)
    if dropped and not named:
        notes.append("the reference did not use column(s) " + ", ".join(dropped)
                     + " (missing values, a single value, a date, or more than 50 levels)")
    elif dropped:
        raise ValueError(f"reference_columns {dropped} cannot be used by the tree: they have "
                         "missing values, a single value, a date type or more than 50 levels")
    return extra


def _is_factor(frame: pl.DataFrame, name: str) -> bool:
    """A string, categorical or boolean column, or a numeric one with two values."""
    dtype = frame[name].dtype
    if dtype in (pl.String, pl.Categorical, pl.Boolean) or isinstance(dtype, pl.Enum):
        return True
    return frame[name].drop_nulls().n_unique() <= 2


def _is_lag_candidate(frame: pl.DataFrame, name: str) -> bool:
    """A floating-point column with more than two distinct values.

    Integer-typed columns (calendar fields, row indices, hour of day) and
    binary flags are not lagged. A measurement stored as a float is lagged even
    when its values are whole numbers, as a temperature recorded in degrees is.
    """
    if frame[name].dtype not in (pl.Float32, pl.Float64):
        return False
    return frame[name].drop_nulls().n_unique() > 2


def _lag_and_rolling_features(frame: pl.DataFrame, times: np.ndarray,
                              columns: list[str], lags: tuple[int, ...]) -> dict:
    """Lagged and trailing rolling-mean columns, in the frame's own row order.

    The steps are rows in time order, so a missing hour in the data moves the
    lag. The rolling mean is over the current and the previous ``max(lags) - 1``
    rows and is missing until the window is full.
    """
    order = np.argsort(times, kind="stable")
    window = max(lags)
    features: dict[str, np.ndarray] = {}
    for name in columns:
        ordered = pl.Series(frame[name].to_numpy().astype(float)[order])
        for k in lags:
            shifted = np.full(len(order), np.nan)
            shifted[order] = ordered.shift(k).to_numpy()
            features[f"{name}_lag{k}"] = shifted
        rolled = np.full(len(order), np.nan)
        rolled[order] = ordered.rolling_mean(window, min_samples=window).to_numpy()
        features[f"{name}_roll{window}"] = rolled
    return features


def _slice_scheme(scheme, mask: np.ndarray):
    """A fold scheme restricted to the rows where ``mask`` is true."""
    kwargs: dict[str, Any] = {"weights": scheme.weights[mask],
                              "held_out": scheme.held_out[mask]}
    if scheme.judge is not None:
        kwargs["judge"] = scheme.judge[mask]
    if scheme.base_weights is not None:
        kwargs["base_weights"] = scheme.base_weights[mask]
    for name in ("groups", "strata", "known_groups", "times"):
        value = getattr(scheme, name)
        if value is not None:
            kwargs[name] = value[mask]
    return dataclasses.replace(scheme, **kwargs)


def _make_folds(n: int, folds, fold_scheme, time_values, seed: int):
    from mboostrs.folds import FoldScheme, repeated_kfold_folds, rolling_folds

    if fold_scheme is not None:
        if not isinstance(fold_scheme, FoldScheme):
            raise TypeError("fold_scheme must be a mboostrs.folds FoldScheme, such as "
                            "rolling_folds(...), grouped_folds(...) or blocked_folds(...)")
        return fold_scheme
    if isinstance(folds, bool) or not isinstance(folds, (int, np.integer)):
        raise TypeError(f"folds must be an integer or a fold_scheme, got {folds!r}")
    if folds < 2:
        raise ValueError(f"folds must be at least 2 for a standard error, got {folds}")
    if time_values is not None:
        return rolling_folds(n, B=int(folds), time=time_values)
    return repeated_kfold_folds(n, k=int(folds), repeats=1, seed=seed)


# ---------------------------------------------------------------------------
# Fitting and scoring


def _scan(frame, response, formula, family, scheme, max_mstop, nu) -> dict:
    """Held-out loss of one formula at its best step count, with per-fold losses.

    This is the computation of ``mboostrs._autospec._path_held_out_scan``, with
    the best step count returned as well.
    """
    import warnings

    from mboostrs import gamboost
    from mboostrs._autospec import _control

    fit = gamboost(formula, data=frame, family=family, control=_control(max_mstop, nu))
    with warnings.catch_warnings():
        # A best step count at the cap is reported in the result as unfinished
        # tuning, so cvrisk's own warning for it is not repeated here.
        warnings.filterwarnings("ignore", message="optimal mstop", category=UserWarning)
        cv = fit.cvrisk(folds=scheme, _inner=True)
    aggregate = np.asarray(cv.aggregate, dtype=float)[1:]
    if not np.any(np.isfinite(aggregate)):
        return {"loss": float("nan"), "mstop": 0, "at_cap": False,
                "folds": np.full(np.asarray(scheme.weights).shape[1], np.nan)}
    best = int(np.nanargmin(aggregate))
    per_fold = np.asarray(cv.risk_matrix, dtype=float)[:, 1:][:, best]
    step = best + 1
    return {"loss": float(aggregate[best]), "mstop": step,
            "at_cap": step >= CAP_SHARE * max_mstop, "folds": per_fold}


def _paired(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of ``first - second`` across folds."""
    from mboostrs._rust import paired_mean_se

    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    if first.size < 2 or not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        return float("nan"), float("nan")
    mean, se = paired_mean_se(first, second)
    return float(mean), float(se)


def _in_sample_loss(frame, response, formula, family, mstop, nu) -> float:
    """Loss of ``formula`` fitted at ``mstop`` on all rows and scored on those rows."""
    from mboostrs import gamboost
    from mboostrs._autospec import _control
    from mboostrs._rust import family_diagnostics
    from mboostrs.explore import _fitted_nuisance
    from mboostrs.families import _family_params, _family_rust_name

    fit = gamboost(formula, data=frame, family=family, control=_control(mstop, nu))
    y = np.ascontiguousarray(frame[response].to_numpy().astype(float))
    eta = np.ascontiguousarray(np.asarray(fit.predict(type="link"), dtype=float).ravel())
    scalar, vector = _fitted_nuisance(fit) if hasattr(fit, "nuisance") else (None, None)
    _, _, risk = family_diagnostics(_family_rust_name(family), _family_params(family), y,
                                    eta, np.ones(y.size), scalar, vector)
    return float(risk) / y.size


def _working_residual(fit, family, y: np.ndarray, eta: np.ndarray,
                      mu: np.ndarray) -> tuple[np.ndarray, str]:
    """The residual the candidate screen ranks on, with the name of its kind.

    A Gaussian fit uses ``y - mu``, whose squared bin means are the family loss
    up to a constant. For any other family the residual is the negative
    gradient of the family's per-row loss with respect to the linear predictor
    ``eta``, divided by the square root of the loss's curvature when that is
    positive. A cell of rows that shares a shift of ``eta`` lowers the loss by
    about ``(sum of gradients)^2 / (2 * sum of curvatures)``, so the squared bin
    mean of this scaled gradient ranks terms by the family loss they remove
    (for a Poisson fit it is the Pearson residual). A loss without curvature,
    such as the pinball loss, uses the gradient itself. The gradient and
    curvature are central differences of ``family_held_out_rows``, evaluated at
    the fit's own nuisance value. A family whose loss couples rows falls back to
    ``y - mu`` (kind ``"response"``).
    """
    from mboostrs._rust import family_held_out_rows
    from mboostrs.explore import _fitted_nuisance
    from mboostrs.families import Gaussian, GaussReg, _family_params, _family_rust_name

    if isinstance(family, (Gaussian, GaussReg)):
        return y - mu, "response"
    scalar, vector = _fitted_nuisance(fit) if hasattr(fit, "nuisance") else (None, None)
    name, params = _family_rust_name(family), _family_params(family)
    step = 1e-3

    def rows(shift: float):
        out = family_held_out_rows(name, params, np.ascontiguousarray(y),
                                   np.ascontiguousarray(eta + shift), scalar, vector)
        return None if out is None else np.asarray(out, dtype=float)

    plus, centre, minus = rows(step), rows(0.0), rows(-step)
    if plus is None or centre is None or minus is None \
            or not all(np.all(np.isfinite(v)) for v in (plus, centre, minus)):
        return y - mu, "response"
    gradient = -(plus - minus) / (2 * step)
    curvature = (plus - 2 * centre + minus) / step ** 2
    positive = curvature > 1e-8
    if positive.mean() > 0.5:
        floor = 1e-6 * float(np.median(curvature[positive]))
        return gradient / np.sqrt(np.maximum(curvature, max(floor, 1e-12))), "family"
    return gradient, "family"


def _out_of_fold_residuals(frame, response, formula, family, scheme, mstop,
                           nu) -> tuple[np.ndarray, str]:
    """Working residual of each row (see ``_working_residual``), from the fold
    that held it out, and its kind.

    A row is held out in at most one fold of the schemes used here, so the
    average is over one value. Rows no fold judges are NaN.
    """
    from mboostrs import gamboost
    from mboostrs._autospec import _control

    y = frame[response].to_numpy().astype(float)
    weights = np.asarray(scheme.weights, dtype=float)
    judged = np.asarray(scheme.judge if scheme.judge is not None else scheme.held_out,
                        dtype=bool)
    total = np.zeros(y.size)
    count = np.zeros(y.size)
    kinds = set()
    for column in range(weights.shape[1]):
        rows = judged[:, column]
        if not rows.any():
            continue
        fit = gamboost(formula, data=frame, family=family, control=_control(mstop, nu),
                       weights=weights[:, column])
        eta = np.asarray(fit.predict(type="link"), dtype=float).ravel()
        mu = np.asarray(fit.predict(type="response"), dtype=float).ravel()
        resid, kind = _working_residual(fit, family, y[rows], eta[rows], mu[rows])
        kinds.add(kind)
        total[rows] += resid
        count[rows] += 1
    kind = "family" if "family" in kinds else "response"
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), kind


def _interaction_term(left: str, right: str, factors: set[str]) -> str | None:
    """The formula term for an interaction pair, or None when it is two factors."""
    left_factor, right_factor = left in factors, right in factors
    if left_factor and right_factor:
        return None
    if left_factor or right_factor:
        numeric, factor = (right, left) if left_factor else (left, right)
        return f"bbs({quote_name(numeric)}, by={quote_name(factor)}, df=6)"
    return f"bbs({quote_name(left)}) %X% bbs({quote_name(right)})"


def _factor_codes(series: pl.Series) -> np.ndarray:
    """Integer level codes of a factor column, as floats."""
    if series.dtype in (pl.String, pl.Categorical, pl.Boolean) or isinstance(series.dtype, pl.Enum):
        values = series.cast(pl.String).fill_null("").to_numpy()
    else:
        values = series.to_numpy().astype(float)
    _, codes = np.unique(values, return_inverse=True)
    return codes.astype(float)


def _screen_candidates(frame, response, covariates, residual, factors, main_names,
                       seed: int, notes: list[str], lag_names=()) -> list[dict]:
    """Interaction and main-effect candidates with their out-of-sample screen shares.

    ``main_names`` are the columns the formula does not use (omitted columns and
    the lag and rolling features); each is scored as a main effect on the
    residual. Interaction pairs are scored over the formula's covariates.
    """
    from mboostrs._rust import screen_main_effects
    from mboostrs.terms import screen_interactions

    rows = np.isfinite(residual)
    resid = residual[rows]
    mean_square = float(np.mean(resid ** 2))
    if not mean_square > 0:
        notes.append("the out-of-fold residuals have no variation, so no candidate was screened")
        return []
    sub = frame.filter(pl.Series(rows))
    screened = screen_interactions(
        sub.select(covariates), resid, response=None, seed=seed,
        groups=[c for c in covariates if c in factors])
    candidates: list[dict] = []
    skipped = 0
    for pair in screened:
        if pair.removed <= 0:
            continue
        term = _interaction_term(pair.left, pair.right, factors)
        if term is None:
            skipped += 1
            continue
        candidates.append({"kind": "interaction", "term": term,
                           "source": f"{pair.left} x {pair.right}",
                           "share": pair.removed / mean_square})
    if skipped:
        notes.append(f"{skipped} interaction pair(s) of two factors were screened but have "
                     "no term here; fit them with bols() interactions if they matter")
    if main_names:
        columns, levels = [], []
        for name in main_names:
            if name in factors:
                codes = _factor_codes(sub[name])
                columns.append(codes)
                levels.append(int(codes.max()) + 1)
            else:
                columns.append(sub[name].to_numpy().astype(float))
                levels.append(0)
        X = np.column_stack(columns)
        scored = screen_main_effects(np.ascontiguousarray(X), np.ascontiguousarray(resid),
                                     4, int(seed), levels, None)
        for column, removed in scored:
            if removed <= 0:
                continue
            name = main_names[column]
            if name in factors:
                term, kind = f"bols({quote_name(name)})", "column"
            else:
                term, kind = f"bbs({quote_name(name)}, df=4)", "lag" if name in lag_names else "column"
            candidates.append({"kind": kind, "term": term,
                               "source": name, "share": removed / mean_square})
    candidates.sort(key=lambda c: c["share"], reverse=True)
    return candidates


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "rank": pl.Int64, "kind": pl.String, "term": pl.String, "source": pl.String,
        "screen_share": pl.Float64, "measured_gain": pl.Float64, "gain_se": pl.Float64,
        "relative_gain": pl.Float64, "closed_share": pl.Float64, "mstop": pl.Int64,
        "at_cap": pl.Boolean})


_T = TypeVar("_T")


def _check_budget(start: float, budget: float | None, stage: str,
                  completed: list[str]) -> None:
    """Raise ``TimeoutError`` if the wall-clock budget is spent before ``stage``."""
    if budget is None:
        return
    elapsed = _time.monotonic() - start
    if elapsed >= budget:
        done = ", ".join(completed) if completed else "none"
        raise TimeoutError(
            f"reference_gap exceeded its time budget of {budget:g}s before the "
            f"'{stage}' stage (elapsed {elapsed:.0f}s; completed: {done}). To allow "
            f"more time, raise time_budget; to reduce the cost, lower max_mstop "
            f"or folds.")


def _run_stage(fn: Callable[[], _T], timeout: float | None, stage: str,
               completed: list[str]) -> _T:
    """Run ``fn`` with a hard wall-clock timeout, in a daemon worker thread.

    The Rust engine releases the GIL during long computes, so a call that
    blocks indefinitely (rather than merely computing slowly) cannot wedge
    the caller: the timeout fires and names the stage instead of hanging
    silently. A thread that never returns cannot be killed; it is a daemon
    thread holding no Python resources, so it neither blocks the caller nor
    interpreter shutdown. ``timeout=None`` runs ``fn`` directly.
    """
    if timeout is None:
        return fn()
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True,
                              name=f"reference_gap:{stage}")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        done = ", ".join(completed) if completed else "none"
        raise TimeoutError(
            f"reference_gap exceeded its time budget during the '{stage}' stage "
            f"(no result after {timeout:g}s; completed: {done}). The stage did "
            f"not return; to allow more time, raise time_budget, or lower "
            f"max_mstop or folds to reduce the cost.") from None
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


# ---------------------------------------------------------------------------
# Public function


def reference_gap(formula_or_model, data, *, family=None, folds=5, fold_scheme=None,
                  time=None, reference="trees", maxdepth=4, max_mstop=5000, seed=0,
                  candidates=None, top=5, lags=(1, 3, 24), nu=0.1, lag_columns=None,
                  reference_columns="all", exclude_columns=None,
                  extrapolation="warn", time_budget: float | None = 600) -> ReferenceGapResult:
    """Compare a specification with a tree reference and rank candidate revisions.

    Two boosted models are scored on the same held-out rows. The specification
    is the formula (or the formula of a fitted ``gamboost`` model). The
    reference is a single ``btree`` term with depth ``maxdepth`` and the same
    family and step length. By default (``reference_columns="all"``) it is
    given every predictor column of ``data`` besides the response, the time
    column and ``exclude_columns``, and, with ``time=``, the lag and
    rolling-mean features described below; ``reference_columns="formula"``
    gives it only the covariates the formula reads. Each model's step count is
    the one that minimises its mean held-out loss, found on the folds. The gap
    is the mean over folds of the specification's loss minus the reference's,
    and its standard error is taken across folds. The result lists the columns
    the reference used and those among them that the formula does not use.

    The function then ranks terms that could close the gap, whether or not the
    gap is above two standard errors (``material`` reports that). The screen
    runs on the specification's out-of-fold working residual: the negative
    gradient of the family's loss at the out-of-fold linear predictor, divided
    by the square root of the loss's curvature where that is positive. For a
    Gaussian fit this is the ordinary residual ``y - fitted``, so its ranking
    is the residual mean-square ranking. For other families the ranking is by
    the family loss a term removes, not by squared error on the response scale.
    Interaction pairs of the formula's covariates are screened with
    :func:`mboostrs.screen_interactions`. A numeric pair becomes
    ``bbs(a) %X% bbs(b)``; a numeric and a factor become
    ``bbs(x, by=f, df=6)``; two factors are not turned into a term. A column of
    ``data`` that the formula does not use is screened as a main effect and
    becomes ``bbs(col, df=4)`` (``bols(col)`` for a factor). With ``time=``,
    every floating-point column with more than two distinct values gets lagged and
    rolling-mean features (``lags`` steps back, and a rolling mean over
    ``max(lags)`` rows), screened together with the other columns, and the
    candidates are ``bbs(<feature>, df=4)``. The best ``top`` candidates are
    each added to the specification and refitted once, and the candidates with
    a positive gain are then added together in one more refit.

    Parameters
    ----------
    formula_or_model : str or fitted gamboost model
        The specification, as ``"y ~ bbs(x, df=4) + ..."``. A fitted
        ``gamboost`` model supplies its formula and, when ``family`` is None,
        its family.
    data : polars or pandas DataFrame, or a mapping of column to values
        The training rows. Every column the formula reads must be present and
        complete: missing values raise, as they do in ``gamboost``.
    family : Family, optional
        The loss. ``None`` infers it from the response as ``autospec`` does.
    folds : int, default 5
        Number of folds. Without ``time=`` the folds are a random k-fold split
        with ``seed``. With ``time=`` they are rolling-origin blocks: fold ``b``
        trains on the rows before its block and is scored on the block, so
        no fold trains on later rows. The first block starts after half the
        rows.
    fold_scheme : FoldScheme, optional
        An explicit scheme from :mod:`mboostrs.folds`, such as
        ``grouped_folds`` or ``blocked_folds``. Overrides ``folds`` and
        ``time`` for the split. Its rows must match ``data`` in order. With
        ``time=`` and lags, rows in the first ``max(lags)`` of time order are
        removed from the analysis, and the scheme is restricted to the rows
        that remain.
    time : str, optional
        Name of a timestamp column (numeric or datetime). The folds become
        time-ordered, and lag and rolling-mean candidates are built in time
        order.
    reference : {"trees"}, default "trees"
        The flexible reference. ``"trees"`` is the ``btree`` term described
        above.
    maxdepth : int, default 4
        Depth of the reference tree.
    max_mstop : int, default 5000
        Largest step count searched for each model. A best step count within
        5% of it is reported as unfinished tuning.
    seed : int, default 0
        Seed of the random k-fold split and of the residual screen.
    candidates : list of str, optional
        Terms to measure in place of the screened candidates, for example
        ``["bbs(TEMP_lag1, df=4)"]``. Without this argument the screened
        candidates are measured.
    top : int, default 5
        Number of candidates measured, one refit each, whatever the gap.
        ``0`` skips candidate measurement.
    lags : tuple of int, default (1, 3, 24)
        Lag steps, and the rolling-mean window is ``max(lags)``. Used only with
        ``time=``.
    nu : float, default 0.1
        Step length of both models.
    lag_columns : list of str, optional
        Columns to lag. ``None`` lags every floating-point column with more
        than two distinct values, other than the response and the time column.
        Integer-typed columns and binary flags are not lagged.
    reference_columns : {"all", "formula"} or list of str, default "all"
        The columns the reference ``btree`` sees besides the formula's
        covariates. ``"all"`` adds every other predictor column of ``data``
        (columns with missing values, a single value, a date type or more than
        50 levels are skipped and noted) and, with ``time=``, the lag and
        rolling features. ``"formula"`` is the earlier behaviour: only the
        covariates the formula reads. A list names the extra columns.
    exclude_columns : list of str, optional
        Columns that are never given to the reference, such as weight, offset
        or identifier columns.
    extrapolation : {"warn", "clip", "error", "allow"}, default "warn"
        Checked and stored for the same calls as ``discover``. Every score
        here is on rows the models were fitted on, so no extrapolation policy
        acts on it; the setting has no effect on the numbers returned.
    time_budget : float or None, default 600
        Wall-clock seconds for the whole call. The budget is checked between
        the computation stages, and the tree reference scan — the most
        expensive stage — runs with the remaining budget as a hard timeout:
        exceeding it raises ``TimeoutError`` naming the stage instead of
        hanging silently. ``None`` disables the budget (not recommended).

    Returns
    -------
    ReferenceGapResult
        The gap, its standard error, the two losses, the candidate table, the
        share of the gap the candidates close together, and the checks in
        ``notes``. ``describe()`` says what to try next.

    Examples
    --------
    Measure the gap for a model on random folds, then read the result::

        res = reference_gap("y ~ bbs(x1, df=4) + bols(g)", data, folds=5, seed=1)
        print(res.describe())
        res.candidates          # ranked candidate terms, with measured gains
    """
    from mboostrs._autospec import _relative_gain, _risk_floor
    from mboostrs._extrapolation import POLICIES
    from mboostrs.autospec import _infer_family
    from mboostrs.discover import _as_frame
    from mboostrs.families import _family_rust_name

    if extrapolation not in POLICIES:
        raise ValueError(f"extrapolation must be one of {POLICIES}, got {extrapolation!r}")
    if reference not in _REFERENCES:
        raise ValueError(f"reference must be one of {_REFERENCES}, got {reference!r}. "
                         "'trees' is the only reference this version builds.")
    if isinstance(maxdepth, bool) or not isinstance(maxdepth, (int, np.integer)) \
            or not 1 <= maxdepth <= 10:
        raise ValueError(f"maxdepth must be an integer from 1 to 10, got {maxdepth!r}")
    if max_mstop < 100:
        raise ValueError(f"max_mstop must be at least 100, got {max_mstop}")
    if top < 0:
        raise ValueError(f"top must be zero or more, got {top}")
    if not nu > 0:
        raise ValueError(f"nu must be positive, got {nu}")
    if time_budget is not None and not time_budget > 0:
        raise ValueError(f"time_budget must be positive or None, got {time_budget!r}")

    started = _time.monotonic()
    completed: list[str] = []

    if isinstance(formula_or_model, str):
        formula = formula_or_model
    elif hasattr(formula_or_model, "formula"):
        formula = formula_or_model.formula
        if family is None:
            family = formula_or_model.family
    else:
        raise TypeError("formula_or_model must be a formula string or a fitted gamboost model")
    response, _ = _split_formula(formula)

    frame = _as_frame(data)
    if response not in frame.columns:
        raise ValueError(f"response {response!r} is not a column of data")
    if family is None:
        family = _infer_family(frame, response)
    family_name = _family_rust_name(family)
    covariates = _covariates(formula, response, frame)
    exclude = [] if exclude_columns is None else [str(c) for c in exclude_columns]
    absent_excluded = [c for c in exclude if c not in frame.columns]
    if absent_excluded:
        raise ValueError(f"exclude_columns {absent_excluded} are not columns of data")
    notes: list[str] = []
    extra_columns = _reference_columns(frame, reference_columns, covariates, response, time,
                                       exclude, notes)
    # Columns the formula omits are candidates whatever the reference was given.
    omitted = _reference_columns(frame, "all", covariates, response, time, exclude, [])
    factors = {c for c in [*covariates, *omitted] if _is_factor(frame, c)}

    lag_names: list[str] = []
    if time is not None:
        if time not in frame.columns:
            raise ValueError(f"time column {time!r} is not a column of data")
        if frame[time].null_count():
            raise ValueError(f"time column {time!r} has missing values; drop or fill them "
                             "before calling reference_gap")
        times_all = frame[time].to_numpy()
        columns = (list(lag_columns) if lag_columns is not None else
                   [c for c in frame.columns
                    if c not in (response, time, *exclude) and _is_lag_candidate(frame, c)])
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"lag_columns {missing} are not columns of data")
        lag_steps = tuple(sorted({int(k) for k in lags}))
        if not lag_steps or lag_steps[0] < 1:
            raise ValueError(f"lags must be positive whole steps, got {tuple(lags)}")
        if columns:
            features = _lag_and_rolling_features(frame, times_all, columns, lag_steps)
            frame = frame.with_columns([pl.Series(name, values) for name, values
                                        in features.items()])
            lag_names = list(features)
            complete = np.ones(frame.height, dtype=bool)
            for values in features.values():
                complete &= np.isfinite(values)
            dropped = int((~complete).sum())
            if dropped:
                notes.append(f"{dropped} row(s) have no full lag or rolling window (the first "
                             "rows in time order, or rows with a missing value in a lagged "
                             "column) and are not in the analysis")
            if not complete.any():
                raise ValueError("no row has a full lag window; use shorter lags or more rows")
            frame = frame.filter(pl.Series(complete))
            if fold_scheme is not None:
                fold_scheme = _slice_scheme(fold_scheme, complete)
            time_values = times_all[complete]
        else:
            time_values = times_all
    else:
        time_values = None
    n = frame.height
    if n < 10:
        raise ValueError(f"reference_gap needs at least 10 analysis rows, got {n}")

    scheme = _make_folds(n, folds, fold_scheme, time_values, seed)
    if fold_scheme is not None and np.asarray(scheme.weights).shape[0] != n:
        raise ValueError(f"fold_scheme has {np.asarray(scheme.weights).shape[0]} rows but the "
                         f"analysis has {n}. Build the scheme on the same rows, in the same "
                         "order, as data.")
    n_folds = int(np.asarray(scheme.weights).shape[1])
    if n_folds < 3:
        notes.append(f"the gap rests on {n_folds} fold(s), fewer than 3, so its standard "
                     "error is unreliable")

    if isinstance(reference_columns, str) and reference_columns == "formula":
        lag_used: list[str] = []
    else:
        lag_used = list(lag_names)
    used_columns = [*covariates, *extra_columns, *lag_used]
    reference_formula = (f"{quote_response(response)} ~ btree({', '.join(quote_name(_c) for _c in used_columns)}, "
                         f"maxdepth={int(maxdepth)}, selection=sse)")
    floor = _risk_floor(family, frame, response)

    _check_budget(started, time_budget, "specification scan", completed)
    model = _scan(frame, response, formula, family, scheme, max_mstop, nu)
    completed.append("specification scan")

    # The tree reference is the most expensive stage and the one whose cost
    # the caller controls least (a btree over every covariate at max_mstop
    # steps with its own fold tuning). It runs with the remaining budget as
    # a hard timeout so it errors instead of hanging silently.
    _check_budget(started, time_budget, "tree reference scan", completed)
    ref_stage_start = _time.monotonic()
    remaining = (None if time_budget is None
                 else max(time_budget - (ref_stage_start - started), 0.0))
    ref = _run_stage(
        lambda: _scan(frame, response, reference_formula, family, scheme,
                      max_mstop, nu),
        remaining, "tree reference scan", completed)
    completed.append("tree reference scan")
    if time_budget is not None:
        ref_elapsed = _time.monotonic() - ref_stage_start
        if ref_elapsed > 0.5 * time_budget:
            warnings.warn(
                f"the tree reference scan took {ref_elapsed:.0f}s, more than half "
                f"of the {time_budget:g}s time budget. To leave room for the "
                f"candidate measurement, raise time_budget or lower max_mstop.",
                UserWarning, stacklevel=2)

    gap, gap_se = _paired(model["folds"], ref["folds"])
    relative = _relative_gain(model["loss"], ref["loss"], floor)
    material = bool(np.isfinite(gap_se) and gap_se > 0 and gap > MATERIAL_SE * gap_se)

    _check_budget(started, time_budget, "reference in-sample loss", completed)
    in_sample = _in_sample_loss(frame, response, reference_formula, family, ref["mstop"], nu)
    completed.append("reference in-sample loss")
    if model["at_cap"]:
        notes.append(f"the specification's best step count ({model['mstop']}) is within 5% of "
                     "max_mstop; its loss was still falling at the cap")
    if ref["at_cap"]:
        notes.append(f"the reference's best step count ({ref['mstop']}) is within 5% of "
                     "max_mstop; its loss was still falling at the cap")

    result_candidates = _empty_candidates()
    closed_share = None
    combined_formula = None
    combined_gain = None
    combined_se = None
    screen_kind = "none"
    # Candidates are measured whatever the gap; ``material`` is reported, not a gate.
    if candidates is not None or top > 0:
        _check_budget(started, time_budget, "candidate screening", completed)
        if candidates is not None:
            pool = [{"kind": "user", "term": str(t), "source": "candidates=", "share": None}
                    for t in candidates]
        else:
            residual, screen_kind = _out_of_fold_residuals(
                frame, response, formula, family, scheme, model["mstop"], nu)
            main_names = [*omitted, *lag_names]
            pool = _screen_candidates(frame, response, covariates, residual, factors,
                                      main_names, seed, notes, lag_names=set(lag_names))
        pool = [c for c in pool if c["term"] not in formula]
        chosen = pool[:top]
        rows = []
        measured = []
        for rank, cand in enumerate(chosen, start=1):
            scan = _scan(frame, response, f"{formula} + {cand['term']}", family, scheme,
                         max_mstop, nu)
            gain, se = _paired(model["folds"], scan["folds"])
            closed = gain / gap if gap else None
            rows.append({"rank": rank, "kind": cand["kind"], "term": cand["term"],
                         "source": cand["source"], "screen_share": cand["share"],
                         "measured_gain": gain, "gain_se": se,
                         "relative_gain": _relative_gain(model["loss"], scan["loss"], floor),
                         "closed_share": closed, "mstop": scan["mstop"],
                         "at_cap": bool(scan["at_cap"])})
            measured.append((cand["term"], gain, scan))
        if rows:
            result_candidates = pl.DataFrame(rows, schema=_empty_candidates().schema)
        positive = [(term, scan) for term, gain, scan in measured
                    if np.isfinite(gain) and gain > 0]
        if positive:
            combined_formula = formula + " + " + " + ".join(term for term, _ in positive)
            if len(positive) == 1:
                scan = positive[0][1]
            else:
                scan = _scan(frame, response, combined_formula, family, scheme, max_mstop, nu)
            combined_gain, combined_se = _paired(model["folds"], scan["folds"])
            if gap:
                closed_share = combined_gain / gap
            if scan["at_cap"]:
                notes.append("the combined formula's best step count is within 5% of "
                             "max_mstop; its loss was still falling at the cap")
        else:
            closed_share = 0.0 if gap else None
        completed.append("candidate screening")

    return ReferenceGapResult(
        gap=gap, gap_se=gap_se, material=material, relative_gain=float(relative),
        reference_loss=ref["loss"], model_loss=model["loss"],
        model_formula=formula, reference_formula=reference_formula,
        model_mstop=int(model["mstop"]), reference_mstop=int(ref["mstop"]),
        model_at_cap=bool(model["at_cap"]), reference_at_cap=bool(ref["at_cap"]),
        reference_in_sample_loss=in_sample,
        candidates=result_candidates,
        closed_share=closed_share if material else None,
        combined_formula=combined_formula, combined_gain=combined_gain,
        combined_gain_se=combined_se, folds=n_folds, family=family_name,
        notes=notes, data=frame,
        fold_losses={"model": model["folds"], "reference": ref["folds"]},
        reference_columns=used_columns, reference_extra_columns=[*extra_columns, *lag_used],
        screen_residual=screen_kind)
