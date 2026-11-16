from __future__ import annotations

import builtins
import itertools
import warnings
from dataclasses import dataclass

import numpy as np

from mboostrs._output import export_result, json_safe
from mboostrs._results import MappingResultMixin
from mboostrs._validate import check_int_at_least, check_unit_interval

try:
    from mboostrs._rust import cv_weights as _cv_weights
    from mboostrs._rust import cvrisk_glmboost as _cvrisk_glmboost
    _HAS_RUST_ENGINE = True
except ImportError:
    _HAS_RUST_ENGINE = False
    _cv_weights = None
    _cvrisk_glmboost = None


#: Share of a grid's end that counts as the boundary. An optimum within this
#: band of the end has not been seen to turn: the risk was still falling, or
#: flat, where the grid stopped. Every search that grows a grid (`tune_mstop`,
#: `cv_risk` and `compare_specifications` with ``mstop="cv"``) extends the grid
#: while the optimum is in this band, and the boundary warning fires on the same
#: band, so a warning is never raised for an optimum the search would extend.
BOUNDARY_SHARE = 0.05


def boundary_tolerance(end) -> int:
    """How many steps before the end of a grid of `end` steps still count as
    the boundary band: `BOUNDARY_SHARE` of the end, rounded half up, and at
    least one step."""
    return max(1, int(np.floor(BOUNDARY_SHARE * float(end) + 0.5)))


def is_at_boundary(optimum, end) -> bool:
    """Whether an optimum at `optimum` steps lies in the boundary band of a
    grid that ends at `end` steps."""
    return float(optimum) + boundary_tolerance(end) >= float(end)


def _optimum_index(aggregate, grid) -> int:
    """Index of the lowest aggregated risk among grid points with ``mstop >= 1``.

    Iteration 0 is the offset-only model. R's ``mstop(cvrisk_object)`` never
    returns it, and neither does `CvRiskResult.optimal_mstop`; every helper
    that reads an optimum off a curve goes through this function so that the
    one-standard-error rule, the boundary check, summaries and plots agree
    with it. Non-finite risks are skipped; ties take the lowest index, as the
    core does on finite curves. Ties are exact computed equality; no risk
    tolerance is used, so roundoff-different curves can report different stops.
    A grid with no positive point falls back to the whole grid.
    """
    values = np.asarray(aggregate, dtype=np.float64)
    steps = np.asarray(list(grid))
    eligible = np.flatnonzero((steps > 0) & np.isfinite(values))
    if eligible.size == 0:
        eligible = np.flatnonzero(np.isfinite(values))
    if eligible.size == 0:
        return 0
    return int(eligible[int(np.argmin(values[eligible]))])


@dataclass(frozen=True)
class CvReport:
    """Summary of a cross-validation run for standard and distributional models.

    `folds` describes the fold design in words and `n_folds` counts its
    columns. `grid` lists the budgets evaluated: step counts for a standard
    model, one tuple per grid point (a budget per parameter) for a
    distributional one. `optimum` is the chosen budget, an int or
    ``{parameter: steps}``; `risk_at_optimum` is the mean held-out risk
    there, in the units `score` names. `at_boundary` is true when the optimum
    is the largest budget searched, so the search stopped at the grid's edge.
    """

    folds: str
    n_folds: int
    grid: tuple
    optimum: object
    risk_at_optimum: float
    score: str
    at_boundary: bool
    preprocessing: str = "fixed"


def _score_label(provenance) -> str:
    family = getattr(provenance, "family", None)
    loss = f"the {family} family's loss" if family else "the family's loss"
    return f"mean held-out risk: {loss}, per unit of held-out observation weight"


class CvRiskResult(MappingResultMixin):
    """Cross-validated risk along the boosting path.

    Returned by a model's ``cvrisk()``. Slice the model to
    ``optimal_mstop`` to get the tuned fit.

    Attributes
    ----------
    risk_matrix : ndarray, shape (n_folds, n_grid)
        Out-of-fold risk for every fold at every grid point. Two models
        evaluated on the same folds can be compared using their paired
        per-fold differences.
    aggregate : ndarray, shape (n_grid,)
        Mean risk across folds at each grid point.
    grid : ndarray
        The iteration counts evaluated.
    optimal_mstop : int
        The grid point minimising ``aggregate``.
    fold_type : str
        How the folds were produced — "10-fold kfold" for folds this call
        generated, the scheme's own name for a `FoldScheme`, and "10
        supplied folds" for a bare matrix, whose scheme this object cannot
        know.
    provenance : Provenance or None
        The model, family, row count, budget and weight policy this curve was
        computed under. ``result.describe()`` prints it.
    held_out_fraction : float
        The average share of the sample a fold holds out: ``1/k`` for k-fold,
        about 0.37 for a bootstrap draw. It scales the Nadeau-Bengio
        correction in :meth:`compare_to`, which is why it is measured rather
        than assumed.
    data_fingerprint : str
        Digest of the response and weights the folds were fitted to
        (`data_fingerprint`); empty for a result built without one.
    data_identity : str
        SHA-256 digest of the observation IDs, observed response and effective
        weights of the model the curve was computed for, read from its
        comparison identity; empty when the model recorded none.
        :meth:`compare_to` refuses two curves whose digests differ.
    fold_fingerprint : str
        A digest of training fold weights, evaluation masks and model
        observation weights. Matching digests are necessary for pairing;
        callers must also use the same observations in the same order and
        comparable scoring targets. The digest does not identify the data.
    folds : FoldScheme or None
        The scheme, when one was passed. A generated or bare matrix is not
        retained -- it is `n * B` floats and the caller already has it.
    design : str
        The fold design this curve was computed under: the scheme's own name
        for a `FoldScheme`, "N supplied folds" for a bare partition matrix,
        and "N bootstrap folds" when the matrix carries the multiplicity
        weights a bootstrap draw leaves behind. :meth:`compare_to` refuses an
        interval for the designs the Nadeau-Bengio correction does not cover.
    preprocessing : {"fixed", "refit"}
        Whether the full-data recipe was reused or preprocessing was learned
        within each training fold.
    """

    __slots__ = ("risk_matrix", "aggregate", "grid", "optimal_mstop",
                 "fold_type", "provenance", "fold_fingerprint", "data_fingerprint", "folds",
                 "held_out_fraction", "design", "preprocessing", "boundary_warning",
                 "comparison_identity")

    _MAPPING_KEYS = ("risk_matrix", "aggregate", "grid", "optimal_mstop",
                     "fold_type", "risk", "mstop", "standard_error",
                     "mstop_1se", "preprocessing", "boundary_warning")
    _MAPPING_ALIASES = {"risk": "risk_matrix", "mstop": "optimal_mstop"}

    def __init__(self, risk_matrix, aggregate, grid, optimal_mstop, fold_type,
                 provenance=None, *, fold_fingerprint="", data_fingerprint="", folds=None,
                 held_out_fraction=None, design=None, preprocessing="fixed",
                 warn_boundary=True, comparison_identity=None):
        self.risk_matrix = risk_matrix
        self.aggregate = aggregate
        self.grid = grid
        self.optimal_mstop = optimal_mstop
        self.fold_type = fold_type
        self.provenance = provenance
        self.fold_fingerprint = fold_fingerprint
        self.data_fingerprint = data_fingerprint
        self.folds = folds
        self.held_out_fraction = held_out_fraction
        self.design = design or fold_type
        self.preprocessing = preprocessing
        # The comparison identity of the model the curve was computed for, or
        # None. `compare_cv` reads it to decide whether two curves can be paired.
        self.comparison_identity = comparison_identity
        self.boundary_warning = self._boundary_warning(grid, optimal_mstop)
        # A search that grows its own grid (`tune_mstop`, `cv_risk` with
        # mstop="cv") passes warn_boundary=False: it extends the grid or
        # warns in its own words, and naming `cvrisk` here would be wrong.
        if self.boundary_warning is not None and warn_boundary:
            warnings.warn(self.boundary_warning, UserWarning, stacklevel=2)

    @staticmethod
    def _boundary_warning(grid, optimal_mstop):
        """Warn when the CV optimum sits at the edge of the searched grid.

        An optimum in the boundary band (`is_at_boundary`, `BOUNDARY_SHARE`
        of the grid's end) means the search range was exhausted: the risk
        was still falling (or flat) where the grid ended, so the "optimal"
        mstop is the cap, not an interior minimum.
        """
        try:
            cap = float(np.max(grid)) if len(grid) else float("nan")
        except TypeError:
            return None
        if cap != cap or cap <= 0:
            return None
        if is_at_boundary(optimal_mstop, cap):
            return (
                f"optimal mstop {int(optimal_mstop)} is within "
                f"{BOUNDARY_SHARE:.0%} of the search cap {cap:g}; the search "
                "range was exhausted \u2014 consider increasing mstop and "
                "re-running cvrisk."
            )
        return None

    def __repr__(self):
        base = f"CvRiskResult(mstop_opt={self.optimal_mstop}, folds={self.risk_matrix.shape[0]})"
        if self.boundary_warning:
            base += " [boundary: search range exhausted]"
        return base

    @property
    def data_identity(self) -> str:
        """SHA-256 digest of the observation IDs, response and weights of the
        model this curve was computed for (`_comparison_identity.data_identity`),
        or ``""`` when the model recorded none."""
        from mboostrs._comparison_identity import record_data_identity

        return record_data_identity(getattr(self, "comparison_identity", None))

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the raw fold-risk matrix only on request.

        Parameters
        ----------
        include_arrays : bool, default False
            Include ``risk_matrix``, the ``(n_folds, n_grid)`` out-of-fold
            risks. The aggregate, grid and standard error are always included.

        Returns
        -------
        dict
            JSON-serialisable record of the optimum, aggregate curve, fold
            design and fingerprints.
        """
        record = {
            "optimal_mstop": self.optimal_mstop,
            "mstop_1se": self.mstop_1se,
            "boundary_warning": self.boundary_warning,
            "aggregate": json_safe(self.aggregate),
            "grid": json_safe(self.grid),
            "standard_error": json_safe(self.standard_error),
            "fold_type": self.fold_type,
            "design": self.design,
            "preprocessing": self.preprocessing,
            "held_out_fraction": self.held_out_fraction,
            "fold_fingerprint": self.fold_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "data_identity": self.data_identity,
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
            "folds": self.folds.to_dict() if self.folds is not None else None,
        }
        if include_arrays:
            record["risk_matrix"] = json_safe(self.risk_matrix)
        return record

    # ── the spread across folds, not only its mean ────────────────────

    @property
    def standard_error(self) -> np.ndarray:
        """Standard error of the mean risk at each grid point.

        The across-fold standard deviation divided by the square root of the
        fold count. The estimate describes variation across this set of folds;
        the folds share training rows, so it is not a confidence bound for
        variation across datasets. :meth:`compare_to` applies the
        Nadeau-Bengio correction when comparing two models under its supported
        fold assumptions.
        """
        risk = np.asarray(self.risk_matrix, dtype=np.float64)
        if risk.shape[0] < 2:
            return np.zeros(risk.shape[1])
        return risk.std(axis=0, ddof=1) / np.sqrt(risk.shape[0])

    @property
    def mstop_1se(self) -> int:
        """The one-standard-error rule: the smallest `mstop` whose risk is
        within one standard error of the minimum.

        The rule adds the standard error at the minimum to the minimum risk,
        then returns the smallest positive grid point whose risk is no greater
        than that threshold. For boosting, the smallest grid point is the
        earliest stopping iteration within the rule's threshold.
        """
        aggregate = np.asarray(self.aggregate, dtype=np.float64)
        grid = np.asarray(list(self.grid))
        best = _optimum_index(aggregate, grid)
        threshold = aggregate[best] + float(self.standard_error[best])
        within = np.flatnonzero((aggregate <= threshold) & (grid > 0))
        return int(grid[int(within.min())]) if within.size else int(grid[best])

    def at_boundary(self) -> bool:
        """Whether the curve is still falling at the last grid point.

        The minimum lies at the last evaluated grid point. The selected step
        count may therefore depend on the grid limit; extend the path to check
        whether risk declines further.
        """
        aggregate = np.asarray(self.aggregate, dtype=np.float64)
        grid = list(self.grid)
        if len(aggregate) < 2:
            return False
        return (_optimum_index(aggregate, grid) == len(aggregate) - 1
                and int(self.optimal_mstop) == int(grid[-1]))

    def report(self) -> CvReport:
        """Fold design, grid, optimum, held-out score and boundary flag as a
        :class:`CvReport`, the shape an LSS result reports too."""
        provenance = self.provenance
        grid = tuple(int(g) for g in self.grid)
        optimum = int(self.optimal_mstop)
        aggregate = np.asarray(self.aggregate, dtype=np.float64)
        return CvReport(
            folds=(getattr(provenance, "folds", None) or str(self.design)),
            n_folds=int(np.asarray(self.risk_matrix).shape[0]),
            grid=grid, optimum=optimum,
            risk_at_optimum=float(aggregate[grid.index(optimum)]),
            score=_score_label(provenance), at_boundary=bool(self.at_boundary()),
            preprocessing=self.preprocessing)

    def compare_to(self, other, *, mstop=None, level=0.95):
        """Compare two models fold by fold, on the folds they share.

        Pairing uses the per-fold differences to account for variation shared
        by the two models on the same folds. The folds must match: the
        fingerprints are checked and a mismatch raises an error.

        ``mstop`` selects an explicit grid point (default: each model at its
        reported optimum). Selected optima are always compared descriptively,
        even when they coincide. An explicit point enables an approximate
        interval under the random-fold assumptions; choosing that point after
        inspecting the curves still incurs selection bias.

        Parameters
        ----------
        other : CvRiskResult
            The second curve. The difference reported is ``self - other``, so
            a negative mean favours ``self`` (risk is a loss).
        mstop : int, optional
            Grid point to compare both models at. Supplying it enables the
            approximate Nadeau-Bengio interval; omitting it compares each
            model at its own ``optimal_mstop`` and leaves ``interval`` None.
        level : float, default 0.95
            Confidence level of the interval, using Student-t quantiles on
            ``n_folds - 1`` degrees of freedom.

        Returns
        -------
        CvComparison
            Mean paired difference, naive and corrected standard errors, the
            t statistic and (when ``mstop`` was given) the interval. Curves on
            different response scales return NaN differences with a notice.

        Raises
        ------
        ValueError
            If the two curves have different fold fingerprints or fold
            counts, fewer than two folds, or a fold design the
            Nadeau-Bengio correction does not cover.
        """
        return compare_cv(self, other, mstop=mstop, level=level)

    def plot(self):
        """Plot cross-validated risk.

        Returns
        -------
        alt.LayerChart
        """
        from mboostrs.plot import plot_cvrisk
        return plot_cvrisk(self)

    def to_frame(self, output="auto"):
        """Long-format table: one row per (fold, mstop) with its risk.

        Columns: ``fold`` (1-based), ``mstop``, ``risk``, ``aggregate_risk``
        (the mean risk at that ``mstop`` across folds), and ``is_optimal``
        (whether that ``mstop`` is ``self.optimal_mstop``).

        Parameters
        ----------
        output : {"auto", "pandas", "polars", "records"}, default "auto"
            Container for the result.

        Returns
        -------
        frame or list of dict
            ``n_folds * n_grid`` rows.
        """
        from mboostrs.gamlss import _records_frame

        grid = list(self.grid)
        rows = []
        for fold_idx in range(self.risk_matrix.shape[0]):
            for col, mstop in enumerate(grid):
                rows.append({
                    "fold": fold_idx + 1,
                    "mstop": int(mstop),
                    "risk": float(self.risk_matrix[fold_idx, col]),
                    "aggregate_risk": float(self.aggregate[col]),
                    "is_optimal": bool(int(mstop) == int(self.optimal_mstop)),
                })
        return _records_frame(
            rows, output,
            columns=("fold", "mstop", "risk", "aggregate_risk", "is_optimal"),
        )


def _nb_interval_supported(result) -> tuple[bool, str]:
    """Whether the Nadeau-Bengio interval is valid for the fold design a
    curve recorded.

    The correction is derived for the setting where every fold is a random
    split of exchangeable rows: k-fold and sub-sampling. Rolling origins and
    blocked folds give up on exchangeability, grouped folds keep dependent
    rows together, repeated k-fold correlates folds across repeats, and a
    bootstrap draw is not a partition of the sample at all. For those designs
    the correction factor is not the right one and there is no calibration to
    claim, so an interval is refused instead of being reported as one.
    """
    design = getattr(result, "design", None) or ""
    design = design.lower()
    if any(token in design for token in (
            "rolling", "blocked", "grouped", "repeated", "bootstrap",
            "multiplicity")):
        return False, design
    return True, design


def _nb_refusal(design: str) -> str:
    return (
        f"the Nadeau-Bengio interval assumes every fold is a random split of "
        f"exchangeable rows, which {design} folds are not, so the interval's "
        f"coverage would not be what it claims. Rebuild both curves on the "
        f"same k-fold (or sub-sampling) fold matrix to read the interval as "
        f"a confidence bound."
    )


@dataclass(frozen=True)
class CvComparison:
    """Two models compared on the folds they share.

    The naive standard error of a paired cross-validation difference is known
    to be too small: the folds' training sets overlap, so their differences
    are positively correlated and the across-fold variance understates the
    variability of the whole procedure. Nadeau and Bengio (2003, *Machine
    Learning* 52:239-281) give the correction used here -- multiply the
    variance by ``1/k + n_test/n_train`` instead of ``1/k`` -- which for
    10-fold cross-validation inflates the standard error by about
    ``sqrt(1 + 10 * (1/9)) = 1.45``. Both are reported, because the naive one
    is what most software prints and the difference between them is the point.

    The interval is an approximation for a prespecified comparison using
    random splits of exchangeable rows. Rolling, blocked, grouped, repeated
    k-fold, and bootstrap designs remain unsupported by this comparison.
    Pass ``mstop=`` to compare at an explicit point. Without it, the result
    is descriptive and ``interval`` is None, even if the selected minima
    coincide. Selecting the same grid value on both curves does not remove
    tuning bias. An explicit value chosen after looking at the curves also
    falls outside the prespecified-comparison interpretation.

    The confidence interval uses Student-t critical values on
    ``n_folds - 1`` degrees of freedom, not normal ones: with the handful of
    folds real comparisons run, normal quantiles understate the width
    materially (2.78 vs 1.96 for a 95% interval on five folds).
    """

    #: Mean of ``risk_a - risk_b`` across folds at the compared grid point.
    #: Negative means the first model has the lower risk.
    mean_difference: float
    #: Per-fold differences, in fold order.
    differences: np.ndarray
    #: Across-fold standard error, uncorrected.
    se_naive: float
    #: The same, with the Nadeau-Bengio correction for the training overlap.
    se_corrected: float
    #: ``mean_difference / se_corrected``, a descriptive statistic in every
    #: mode.
    t: float
    #: Approximate confidence interval for the mean difference, from the corrected
    #: standard error and a t distribution on ``n_folds - 1`` degrees of
    #: freedom. ``None`` when the comparison is descriptive (the two models
    #: were compared at their selected optima); see :attr:`inferential`.
    interval: tuple[float, float] | None
    level: float
    n_folds: int
    #: The grid value the first model was compared at.
    mstop_a: int
    #: The grid value the second model was compared at.
    mstop_b: int
    #: How the correction was scaled: the held-out share of a fold.
    test_fraction: float
    #: The fold design the comparison was computed under, e.g.
    #: ``"10-fold kfold"`` or ``"rolling-origin 5-fold (expanding window)"``.
    design: str
    #: Whether an explicit mstop enabled an approximate interval. This flag
    #: does not establish that the caller prespecified the comparison.
    #: False for all comparisons at data-selected optima.
    inferential: bool
    #: Why the two curves are not ranked against each other, or "" when they
    #: are. A non-empty reason means the curves are on different response
    #: scales (``log(y)`` against ``y``), so there is no difference to report:
    #: the numeric fields are NaN, ``interval`` is None and ``inferential`` is
    #: False.
    comparison: str = ""

    def __repr__(self):
        if self.comparison:
            return f"CvComparison(not ranked, folds={self.n_folds}, comparison={self.comparison!r})"
        return (f"CvComparison(mean_difference={self.mean_difference:.6g}, "
                f"t={self.t:.3f}, folds={self.n_folds}, "
                f"inferential={self.inferential})")

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-fold differences only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("differences",),
        )

    def describe(self) -> str:
        if self.comparison:
            return (f"{self.n_folds} folds, mstop {self.mstop_a} vs {self.mstop_b}\n"
                    f"  not ranked: {self.comparison}")
        favours = "the first model" if self.mean_difference < 0 else "the second"
        lines = [
            f"{self.n_folds} paired folds, mstop {self.mstop_a} vs {self.mstop_b}",
            f"  mean risk difference: {self.mean_difference:.6g} "
            f"(favours {favours})",
            f"  standard error: {self.se_corrected:.6g} corrected, "
            f"{self.se_naive:.6g} uncorrected",
        ]
        if self.inferential and self.interval is not None:
            low, high = self.interval
            separates = "excludes" if low > 0 or high < 0 else "includes"
            lines.append(
                f"  {self.level:.0%} approximate interval: [{low:.6g}, {high:.6g}] "
                f"-- {separates} zero"
            )
            lines.append(
                "  The correction is Nadeau-Bengio: cross-validation folds "
                "share training rows, so the uncorrected error is too small."
            )
            lines.append(
                f"  Interval validity assumes random folds ({self.design}) "
                "and models and mstop chosen before inspecting these curves."
            )
        else:
            lines.append(
                "  No calibrated interval: the two models were compared at "
                "their own selected optima, so the interval would not cover "
                "the tuning bias. The comparison is descriptive."
            )
        return "\n".join(lines)


def _grid_index(grid, mstop: int) -> int:
    """The column whose grid value is ``mstop``, raising when it is absent."""
    grid = np.asarray(list(grid))
    matches = np.flatnonzero(grid == mstop)
    if matches.size == 0:
        raise ValueError(
            f"mstop={mstop} is on neither curve's grid, so the models cannot "
            f"be compared at it"
        )
    return int(matches[0])


def compare_cv(a, b, *, mstop=None, level=0.95) -> CvComparison:
    """Paired comparison of two cross-validation curves on the same folds.

    ``mstop`` selects the fixed grid point to compare at; it must lie on both
    curves' grids. Without it, each model is compared at its reported optimum
    descriptively, including when the selected grid values coincide. The
    Nadeau-Bengio overlap correction does not remove tuning bias. Its interval
    is approximate and requires a prespecified comparison: supplying an
    explicit mstop after inspecting the curves does not satisfy that condition.
    """
    from mboostrs._comparison_identity import check_curves

    level = check_unit_interval("level", level)
    if mstop is not None:
        checked = check_int_at_least("mstop", mstop, 0, allow_none=False)
        # pyright narrow: allow_none=False and a non-None value never return None.
        mstop = int(checked) if checked is not None else 0
        column_a = _grid_index(a.grid, mstop)
        column_b = _grid_index(b.grid, mstop)
    else:
        column_a = _grid_index(a.grid, a.optimal_mstop)
        column_b = _grid_index(b.grid, b.optimal_mstop)

    identities = [getattr(a, "comparison_identity", None), getattr(b, "comparison_identity", None)]
    if all(identity is not None for identity in identities):
        verdict = check_curves(identities, helper="compare_cv")
        if verdict.verdict == "segregate":
            return _segregated_comparison(a, b, column_a, column_b, level,
                                          "; ".join(verdict.reasons))

    if not a.fold_fingerprint or not b.fold_fingerprint:
        raise ValueError(
            "paired comparisons require recorded fold fingerprints on both "
            "curves. Recompute cvrisk with the same folds for both models."
        )
    if a.fold_fingerprint != b.fold_fingerprint:
        raise ValueError(
            "these two curves were computed on different folds, so pairing "
            "them compares the folds as much as the models: "
            f"{a.fold_type} ({a.fold_fingerprint}) against {b.fold_type} "
            f"({b.fold_fingerprint}). Build one fold matrix -- with "
            "mboostrs.cv_weights or one of the fold constructors -- and pass "
            "it to both models."
        )
    risk_a = np.asarray(a.risk_matrix, dtype=np.float64)
    risk_b = np.asarray(b.risk_matrix, dtype=np.float64)
    if risk_a.shape[0] != risk_b.shape[0]:
        raise ValueError(
            f"{risk_a.shape[0]} folds against {risk_b.shape[0]}: these curves "
            "cannot be paired"
        )
    k = risk_a.shape[0]
    if k < 2:
        raise ValueError("a paired comparison needs at least two folds")

    for curve in (a, b):
        supported, design = _nb_interval_supported(curve)
        if not supported:
            raise ValueError(_nb_refusal(design))

    differences = risk_a[:, column_a] - risk_b[:, column_b]
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))

    se_naive = float(np.sqrt(variance / k))
    # Nadeau-Bengio: replace the 1/k in the variance of the mean by
    # 1/k + n_test/n_train. `test_fraction` is the held-out share, so
    # n_test/n_train is f / (1 - f).
    fraction = _test_fraction(a)
    inflation = 1.0 / k + (fraction / (1.0 - fraction) if fraction < 1 else 0.0)
    se_corrected = float(np.sqrt(variance * inflation))

    design = getattr(a, "design", None) or a.fold_type
    mstop_a = int(a.grid[column_a])
    mstop_b = int(b.grid[column_b])
    # Matching data-selected optima are still selected using these fold scores.
    inferential = mstop is not None
    interval = None
    if inferential:
        critical = _t_quantile(0.5 + level / 2.0, k - 1)
        half = critical * se_corrected
        interval = (mean - half, mean + half)
    return CvComparison(
        mean_difference=mean,
        differences=differences,
        se_naive=se_naive,
        se_corrected=se_corrected,
        t=mean / se_corrected if se_corrected > 0 else 0.0,
        interval=interval,
        level=level,
        n_folds=k,
        mstop_a=mstop_a,
        mstop_b=mstop_b,
        test_fraction=fraction,
        design=design,
        inferential=inferential,
    )


def _segregated_comparison(a, b, column_a, column_b, level, notice) -> CvComparison:
    """The result for two curves on different response scales: no difference."""
    nan = float("nan")
    k = int(np.asarray(a.risk_matrix).shape[0])
    fraction = _test_fraction(a)
    return CvComparison(
        mean_difference=nan,
        differences=np.full(k, nan),
        se_naive=nan,
        se_corrected=nan,
        t=nan,
        interval=None,
        level=level,
        n_folds=k,
        mstop_a=int(a.grid[column_a]),
        mstop_b=int(b.grid[column_b]),
        test_fraction=fraction,
        design=getattr(a, "design", None) or a.fold_type,
        inferential=False,
        comparison=notice,
    )


def _test_fraction(result) -> float:
    """The held-out share of a fold, for the Nadeau-Bengio correction.

    Measured from the fold matrix when the curve was built (``1/k`` for
    k-fold, about 0.37 for a bootstrap draw, which is why it is not assumed).
    Falls back to ``1/k`` for a curve from before this was recorded.
    """
    fraction = getattr(result, "held_out_fraction", None)
    if fraction is not None and 0.0 < fraction < 1.0:
        return float(fraction)
    return 1.0 / max(int(np.asarray(result.risk_matrix).shape[0]), 2)


def held_out_fraction(matrix, judge=None) -> float:
    """The average share of the sample a fold holds out.

    Rows already carrying zero weight are out of the sample entirely and are
    not counted as held out. When the fold scheme carries an explicit judge
    mask (rolling origins), the fraction measures the judged rows instead of
    assuming every zero-weight row is scored.
    """
    if judge is not None:
        mask = np.asarray(judge, dtype=bool)
        if mask.shape[0] == 0:
            return 0.0
        return float(mask.mean(axis=0).mean())
    array = np.asarray(matrix, dtype=np.float64)
    in_sample = array.max(axis=1) > 0
    if not in_sample.any():
        return 0.0
    held = (array[in_sample] == 0).mean(axis=0)
    return float(held.mean())


def _t_cdf(t: float, dof: float) -> float:
    """P(T <= t) for a Student-t with ``dof`` degrees of freedom (NaN for
    ``dof <= 0``), from the core (`mboostrs_core::special`)."""
    from mboostrs.gamlss import _special

    return float(_special("t_cdf", t, dof))


def _t_quantile(p: float, dof: float) -> float:
    """The ``p``-quantile of a Student-t, from the core's inversion of its CDF
    (`mboostrs_core::special::student_t_quantile`)."""
    from mboostrs.gamlss import _special

    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    if dof <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {dof}")
    return float(_special("t_quantile", p, dof))


def data_fingerprint(response, weights) -> str:
    """Digest of the response and weights a fit was trained on.

    A cross-validation result carries it so a consumer that is handed the
    result and a model (an explanation prompt, a comparison) can check that
    the two describe the same data: a result computed for another response,
    or the same rows reordered, has another digest.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=8)
    for value in (response, weights):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fold_fingerprint(matrix, *, judge=None, weights=None, response=None) -> str:
    """Digest of training/evaluation membership, observation weights and response.

    Two cross-validation results are paired only when their digests match.
    Including the response means the same fold matrix applied to different
    data -- another response, or the same rows reordered -- no longer pairs.
    Explicit and implicit evaluation masks have distinct digests, even when
    equivalent. Matching digests do not establish that the predictors or the
    scoring rule agree.
    """
    import hashlib

    array = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
    digest = hashlib.blake2b(array.tobytes(), digest_size=8)
    digest.update(str(array.shape).encode())
    for name, value, dtype in (("judge", judge, np.bool_), ("weights", weights, np.float64),
                               ("response", response, np.float64)):
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
        else:
            array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def warn_if_at_boundary(result) -> None:
    """Warn when the cross-validated risk is still falling at the last
    iteration evaluated.

    Then the minimum is wherever the path was cut off. `optimal_mstop` is
    reported all the same, and reads exactly like a minimum that the data
    chose.
    """
    if not result.at_boundary():
        return
    grid = list(result.grid)
    warnings.warn(
        f"the cross-validated risk is still falling at mstop={grid[-1]}, the "
        f"largest iteration evaluated, so the reported optimal_mstop="
        f"{result.optimal_mstop} is where the path stopped rather than where "
        f"the risk turns. Refit with a larger mstop and cross-validate again "
        f"on the same folds: `model.continue_to({2 * int(grid[-1])})` then "
        f"`cvrisk(folds=...)`. (`continue_to` resumes the retained gradient "
        f"where the fit kept one, and otherwise costs a full path; "
        f"`parity_notes(include_global=True)` says which routes resume.)",
        RuntimeWarning,
        stacklevel=3,
    )


#: A fold whose held-out risk at the last step exceeds the median fold's by
#: this factor is treated as deciding `mstop` on its own.
_DOMINANT_FOLD_RATIO = 10.0


def warn_if_one_fold_dominates(result) -> None:
    """Warn when one fold's held-out risk is far above the others'.

    The aggregate is a mean over folds, so a fold whose risk explodes pulls
    the minimum to wherever that fold is still small, usually very early. On
    a 5-fold search of California housing with raw `AveOccup` (skewness near
    40), one fold reached 160 while the others stayed near 0.4, and the search
    stopped at 11 of 2000 steps although held-out error kept falling to 2000.
    """
    rows = np.asarray(result.risk_matrix, dtype=float)
    if rows.ndim != 2 or rows.shape[0] < 3:
        return
    last = rows[:, -1]
    median = float(np.median(last))
    worst = int(np.argmax(last))
    if not np.isfinite(median) or median <= 0 or not last[worst] > _DOMINANT_FOLD_RATIO * median:
        return
    warnings.warn(
        f"fold {worst + 1} of {rows.shape[0]} has a held-out risk of "
        f"{last[worst]:.3g} at the last step evaluated, {last[worst] / median:.0f} "
        f"times the median fold's {median:.3g}, so the chosen optimal_mstop="
        f"{result.optimal_mstop} follows that one fold. A few held-out rows with "
        f"extreme predictor values usually cause this: a spline is extended "
        f"across a range its fold's training rows never covered. Fit heavy-tailed "
        f"predictors on the log scale (bbs(log(x))) or on their training "
        f"quantiles (bbs(ecdf(x)), knots at quantiles and a flat effect beyond "
        f"the training range), or check those rows.",
        RuntimeWarning,
        stacklevel=3,
    )


def truncated_at_mstop(result, at_mstop):
    """A copy of `result` judged at a step count the caller chose.

    The curve, the fold rows and the grid end at `at_mstop`, and the
    boundary check is the caller's to make: "was the minimum a truncation
    artifact" is the question the caller answered by fixing the step, so
    the caller of `cvrisk(..., at_mstop=...)` gets the truncated curve
    without the warning a full-path search owes. The `at_boundary` method
    still reports what the truncated curve says, for whoever wants to ask.
    """
    grid = list(result.grid)
    if at_mstop not in grid:
        raise ValueError(
            f"at_mstop={at_mstop} is not a step count this path evaluated; "
            f"the grid runs to {grid[-1]}"
        )
    index = grid.index(at_mstop)
    aggregate = np.asarray(result.aggregate, dtype=np.float64)[:index + 1]
    rows = np.asarray(result.risk_matrix, dtype=np.float64)[:, :index + 1]
    optimum = (int(grid[_optimum_index(aggregate, grid[:index + 1])])
               if len(aggregate) > 1 else at_mstop)
    return CvRiskResult(
        rows, aggregate, grid[:index + 1], optimum, result.fold_type,
        result.provenance,
        fold_fingerprint=result.fold_fingerprint,
        data_fingerprint=getattr(result, "data_fingerprint", ""),
        comparison_identity=getattr(result, "comparison_identity", None),
        held_out_fraction=result.held_out_fraction,
        folds=result.folds,
        design=result.design,
        preprocessing=result.preprocessing,
    )


def _fold_design(folds, folds_matrix) -> str:
    """Name the fold design a curve was built on.

    A :class:`~mboostrs.folds.FoldScheme` names itself. A bare matrix cannot
    know whether it came out of ``cv_weights`` as k-fold, sub-sampling, or a
    bootstrap draw -- with one exception: a bootstrap draw leaves in-bag
    multiplicity weights behind (a row may appear twice in the same fold),
    which a partition never does. Seeing those weights is the one structural
    fact a matrix can state on its own.
    """
    from mboostrs.folds import FoldScheme

    if isinstance(folds, FoldScheme):
        return folds.scheme
    arr = np.asarray(folds_matrix, dtype=np.float64)
    if arr.size and np.nanmax(np.abs(arr)) > 1.0 + 1e-12:
        return f"{arr.shape[1]} bootstrap folds (multiplicity weights)"
    return fold_label_for(folds, folds_matrix)


def fold_label_for(folds, matrix) -> str:
    """Name the folds actually used.

    A :class:`~mboostrs.folds.FoldScheme` knows what it is -- "grouped 5-fold",
    "rolling-origin 5-fold (expanding window)" -- and that name is the thing a
    reader of the result needs. A bare matrix carries no such record, so the
    column count is all there is to say.
    """
    from mboostrs.folds import FoldScheme

    if isinstance(folds, FoldScheme):
        return folds.scheme
    return f"{matrix.shape[1]} supplied folds"


def cv_weights(n, type="bootstrap", B=25, prob=0.5, seed=42, weights=None):
    """Generate cross-validation fold weights.

    Parameters
    ----------
    n : int
        Number of observations.
    type : str
        Resampling type ("bootstrap", "kfold", "subsampling").
    B : int
        Number of folds / bootstrap replications.
    prob : float
        Proportion of observations per fold (bootstrap/sub-sampling).
    seed : int
        Random seed.
    weights : array-like or None
        Observation weights (passed through to folds).

    Returns
    -------
    np.ndarray
        (n, B) matrix of fold membership weights.
    """
    if not _HAS_RUST_ENGINE:
        raise RuntimeError("Rust engine not available")
    w = np.asarray(weights, dtype=np.float64) if weights is not None else None
    return np.asarray(_cv_weights(n, type, B, prob, seed, w), dtype=np.float64)


def _as_fold_matrix(folds, n, weight_policy="upstream", model_weights=None):
    """Return folds as a contiguous (n, B) float64 matrix.

    Each column is a weight vector for one refit, so it goes through the same
    scale policy the model's own weights did: upstream runs every fold through
    ``mboost_fit``, which rescales fractional weights. Folds drawn from unit
    weights are 0/1 and pass through unchanged.

    A 0/1 matrix given to a model with non-constant ``model_weights`` is a
    fold mask: the model's weights are multiplied in, so each fold refits with
    them. A matrix that already equals the weights on its in-bag rows (from
    ``cv_weights(..., weights=w)``) and any count or fractional matrix are
    used as given.
    """
    from mboostrs._weights import rescale_fold_weights

    arr = np.asarray(folds, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("folds must be a 2-D matrix")
    if arr.shape[0] != n and arr.shape[1] == n:
        arr = arr.T
    if arr.shape[0] != n:
        raise ValueError(f"fold matrix has {arr.shape[0]} rows but n = {n}")
    if arr.shape[1] == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("folds must have columns of finite, non-negative weights")
    from mboostrs.folds import FoldScheme

    if isinstance(folds, FoldScheme):
        # A scheme carries complete fit weights; normalization or composition
        # here would change the scheme the caller explicitly supplied.
        return np.ascontiguousarray(arr)
    arr = _with_model_weights(arr, model_weights)
    return np.ascontiguousarray(rescale_fold_weights(arr, weight_policy))


def _with_model_weights(arr, model_weights):
    """Multiply non-constant model weights into a 0/1 fold mask."""
    if model_weights is None:
        return arr
    w = np.asarray(model_weights, dtype=np.float64).ravel()
    if w.shape[0] != arr.shape[0] or np.ptp(w) == 0:
        return arr
    in_bag = arr > 0
    if not np.all(in_bag | (arr == 0)) or not np.all(arr[in_bag] == 1):
        return arr
    carried = np.where(in_bag, w[:, None], 0.0)
    if np.array_equal(arr, carried):
        return arr
    return carried


def make_grid(max, length_out=10, min=None, log=True, dense_mu_grid=True):
    """Build a per-parameter ``mstop`` grid for distributional cross-validation.

    The counterpart of ``gamboostLSS::make.grid``. Tuning an LSS model means
    choosing one ``mstop`` per distribution parameter, so the search space is a
    product grid rather than a line, and it has to be built deliberately.

    Parameters
    ----------
    max : int or mapping
        The largest ``mstop`` to consider. An integer returns a single vector;
        a mapping of parameter name to integer returns the product grid over
        those parameters, in the mapping's order.
    length_out : int or mapping
        How many points per parameter.
    min : int or mapping, optional
        The smallest ``mstop``. Defaults to 1.
    log : bool
        Space the points logarithmically. Boosting improves quickly at first
        and slowly later, so a log grid puts its resolution where the risk
        actually changes. Points are rounded to integers, and duplicates
        produced by rounding are dropped.
    dense_mu_grid : bool
        Fill in every integer along the first parameter above the largest value
        of the others. Cross-validation can extend a fitted path along that
        direction without refitting, so those points are nearly free; the
        remaining grid points each cost a fit.

    Returns
    -------
    list[int] or list[dict]
        A list of ``mstop`` values for a scalar ``max``; otherwise a list of
        grid points, each a mapping of parameter name to ``mstop``, ready to
        pass as ``cvrisk(grid=...)``.

    Examples
    --------
    >>> make_grid(100, length_out=6)
    [1, 3, 6, 16, 40, 100]
    """
    scalar = not hasattr(max, "keys")
    names = ["mstop"] if scalar else list(max.keys())
    highest = [int(max)] if scalar else [int(v) for v in max.values()]

    def _spread(value, default):
        if value is None:
            return [default] * len(names)
        if hasattr(value, "keys"):
            return [int(value[name]) for name in names]
        return [int(value)] * len(names)

    lowest = _spread(min, 1)
    counts = _spread(length_out, 10)

    axes = []
    for name, lo, hi, count in zip(names, lowest, highest, counts, strict=True):
        if lo >= hi:
            raise ValueError(
                f"min must be below max for {name!r}, got min={lo}, max={hi}"
            )
        if count < 2:
            raise ValueError(f"length_out must be at least 2, got {count} for {name!r}")
        if log:
            if lo < 1:
                raise ValueError(
                    f"a logarithmic grid needs min >= 1, got {lo} for {name!r}"
                )
            points = np.exp(np.linspace(np.log(lo), np.log(hi), count))
        else:
            points = np.linspace(lo, hi, count)
        # Rounding a log grid collapses neighbours at the low end; keep one.
        axes.append(sorted({int(round(float(p))) for p in points}))

    if scalar:
        return axes[0]

    points = [dict(zip(names, combination, strict=True))
              for combination in itertools.product(*axes)]

    if dense_mu_grid and len(names) > 1:
        first = names[0]
        others = names[1:]
        ceiling = highest[0]
        filled = {tuple(point.values()): point for point in points}
        for point in list(points):
            floor = builtins.max(point[name] for name in others)
            for value in range(floor, ceiling + 1):
                candidate = dict(point)
                candidate[first] = value
                filled.setdefault(tuple(candidate.values()), candidate)
        points = list(filled.values())

    points.sort(key=lambda point: tuple(point[name] for name in names))
    return points


def weighted_median(x, w=1.0, na_rm=False):
    """The weighted median, matching ``gamboostLSS:::weighted.median``.

    Observations with zero weight are dropped, the rest are sorted, and the
    value where the cumulative weight first passes half the total is returned.
    When the weights are whole numbers summing to an even total the two central
    values are averaged, matching the unweighted median's convention; otherwise
    the larger of the two is returned, which is what the R implementation does.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    weights = np.asarray(w, dtype=np.float64).ravel()
    if weights.size == 1:
        weights = np.full(values.shape, float(weights[0]))
    if weights.shape != values.shape:
        raise ValueError(
            f"w has length {weights.size} but x has length {values.size}"
        )

    keep = weights != 0
    values, weights = values[keep], weights[keep]
    if na_rm:
        finite = ~(np.isnan(values) | np.isnan(weights))
        values, weights = values[finite], weights[finite]
    elif np.any(np.isnan(values)) or np.any(np.isnan(weights)):
        return float("nan")
    if values.size == 0:
        return float("nan")

    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    above = int(np.argmax(cumulative > 0.5))
    below = 0 if above == 0 else int(np.max(np.nonzero(cumulative <= 0.5)[0]))
    total = float(np.sum(weights))
    if total % 1 == 0 and total % 2 == 0:
        return float(np.mean([values[above], values[below]]))
    return float(builtins.max(values[above], values[below]))
