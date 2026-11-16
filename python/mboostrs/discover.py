"""``mboostrs.discover``: one rule-driven, budgeted, null-safe analysis.

``discover(data, response)`` runs the analysis that a careful analyst would
write by hand, with each decision made by a stated rule and recorded in
``result.log``:

1. Draft one term per column with :func:`~mboostrs.terms.classify_columns`.
   Before the draft, a numeric column whose absolute rank (Spearman)
   correlation with the response is at least 0.999 is dropped, with a warning:
   it is a copy or a monotone transform of the response, and it would leak the
   response into the model. A whole-number column with one distinct value per
   row is kept but warned about as a possible row identifier. Then every
   continuous column gets a P-spline at the same ``df``, so the
   comparisons between terms are fair. Two rules change the draft: an
   unordered categorical column with at most ten levels is a ``bols`` term,
   and a positive column with sample skewness above 1 gets its P-spline on
   ``log(x)``, which the formula replays for new rows. Both are logged.
2. Mean effects. Starting from the draft, drop every term whose removal does
   not raise the out-of-fold risk by more than two ``change se``
   (:func:`~mboostrs.explore.compare_specifications`). Each round drops the
   most removable term; rounds repeat until no term qualifies. When every
   term is dropped the model is ``y ~ 1``, which the boosting functions fit
   as an offset-only model.
3. Interactions. :func:`~mboostrs.terms.screen_interactions` ranks the pairs
   of columns by how much of the residual they explain out of sample, averaged
   over five row splits. The top five with a positive score are tried one at a
   time with :func:`~mboostrs.explore.pair_term_for`, whose smooths take each
   column's main-term scale (``log(x)`` for a column drafted on the log).
   Each pair is tested against the mean model with both of its continuous
   main effects refitted at the tensor's own margin smooth,
   ``bbs(x, df=2.5, knots=5)``, so the comparison measures what the tensor adds
   and not the difference between a ``df=4`` main effect and a ``df=2.5``
   margin. For a binary partner paired with a continuous predictor, a
   baseline smooth is included when needed so the varying smooth need not
   represent both level curves alone. The partner keeps its numeric or factor
   coding. The compared baseline terms are
   recorded in the interaction log. A pair is added only if it lowers the out-of-fold risk by
   ``2 * sqrt(log(5))``, about 2.54, ``change se``, and the change is a
   decrease: a pair the boosting never selects has change and ``change se``
   both exactly 0, which is not evidence. The bar for five candidates is the
   multiple-comparison scaling that autospec also uses.
4. Spread. For a Gaussian response, :func:`~mboostrs.explore.heteroscedasticity`
   tests every numeric predictor of the data, mean model or not, on the scale it
   is drafted on (``log(x)`` for a right-skewed column), and nominates those whose
   residual spread moves with them (``|t| >= 2``), or whose eight-bin
   curves reduce held-out log-squared-residual error on average over five
   fixed half-splits. The latter route can detect symmetric scale patterns
   with zero slope. The log lists tested columns and nomination scores.
   A distributional model with a smooth log-scale term for one of them is
   adopted only if it beats the constant-scale model by two ``change se``
   under the joint score of :class:`~mboostrs.gamlss.GaussianLSS`, first at
   the screening fit's 200 steps and then again in a confirmation that
   tunes each specification's stopping inside the folds: a win that only
   reflects the screening fit's unfinished mean is not adopted.
5. The stopping iteration of the adopted mean model is chosen by
   :func:`~mboostrs.explore.tune_mstop` on a grid of 100 steps. A grid whose
   minimum lies in the last 5% of the grid is doubled, at most twice. If the
   minimum still lies there, the stage is recorded as a ``boundary``, the full
   adopted term set is kept, and the stopping iteration found is used. A
   distributional model is tuned by its own cross-validated risk,
   :meth:`LSSModel.cvrisk`, with the same doubling.
6. The adopted terms are refitted by REML (:func:`~mboostrs.selection.gam_refit`
   or :func:`~mboostrs.selection.gamlss_refit`) to give effect intervals and
   term tests. In a mixed ordinary model, unsupported terms such as trees
   retain their boosted contributions; the refit is conditional on those
   contributions. An all-tree model has no supported effect to refit.
   Distributional refits with unsupported terms are recorded as unavailable.
7. ``newdata`` rows are checked with ``extrapolation_report``, predicted with
   ``extrapolation="clip"``, and given intervals by
   :func:`~mboostrs.conformal.conformal_interval`. When a spread model is
   adopted, ``result.predictions`` also carries ``sigma``, the fitted scale
   of each row, next to ``mean`` (the fitted mean), so a score reads the two
   distribution parameters instead of rebuilding the scale from the interval
   band.

Before each stage that refits models, :func:`~mboostrs.cost.estimate_cost`
gives the expected seconds, and the log records that estimate (stage
``"estimate"``) before the stage runs. A stage that would take the run past
``budget_seconds`` is skipped and the log says so. The final model is fitted
whatever the budget, and the log says when that fit went past it.

The tuning stages stop at ``max_mstop`` steps (400 for the mean model by
default). Extrapolation warnings from the fold fits are issued once per call and
name ``discover(extrapolation=...)``.

Screens and comparisons are computed on the rows that have every predictor
and the response. Fits use ``na_action``. A NaN counts as missing, as a null
does. Under ``"impute"`` a numeric interaction partner's missing values are
filled with its training median in every fit, and the distributional fits of
the spread stage fill their predictors the same way.

Three optional constraints act on the stages: ``heredity`` admits an
interaction only when its main effects are in the mean model (``"strong"``)
or one of them is (``"weak"``); ``forbidden_pairs`` are never nominated;
``mandatory_terms`` survive backward elimination, and a mandatory spread
column is included without the spread test. ``candidates`` limits the
columns each parameter's search may use. Each accepts a value for the mean
model or a mapping keyed by ``"mu"`` or ``"sigma"``. The rule in force is the
``"rule"`` entry of ``result.log``.

The adoption rules are fixed (two standard errors for mean terms; for
spread, two standard errors at the screening comparison and again at the
tuned-stopping confirmation,
``2 * sqrt(log(5))`` for interactions, five interaction candidates,
``|t| >= 2`` or positive held-out curve gain for spread nomination). The screen averaging and the reference of an
interaction test were chosen on simulated draws of the claims benchmark
generator, with 20 full-size draws and 4 null draws, not
on the data at hand. The remaining rules were not tuned, and the reported
``change se`` understates the variation between data sets
(see :func:`~mboostrs.explore.compare_specifications`). On permuted responses
the rules kept no term in the checks run for this module (eight seeds at 300
rows). ``DiscoveryResult.log`` records each candidate that was rejected.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from mboostrs._autospec import _validate_against_gaussian, _validate_gamma, probe_family
from mboostrs._constraints import (
    admits,
    check_forbidden_pairs,
    check_heredity,
    check_mandatory,
    main_columns,
    pair_label,
)
from mboostrs._control import boost_control
from mboostrs._errors import MboostrsError
from mboostrs._extrapolation import POLICIES, routine_warnings
from mboostrs._names import quote_name, quote_response
from mboostrs._provenance import provenance
from mboostrs._validate import check_int_at_least, check_whole_int
from mboostrs._warning_collect import collect_warnings
from mboostrs.autospec import autospec
from mboostrs.conformal import _min_calibration_size, conformal_interval
from mboostrs.cost import estimate_cost
from mboostrs.cvrisk import cv_weights, is_at_boundary
from mboostrs.effects import prediction_contrast
from mboostrs.explore import (
    _effects,
    compare_specifications,
    heteroscedasticity,
    pair_term_for,
    tune_mstop,
)
from mboostrs.families import Gaussian, NBinomial, Poisson, _family_rust_name
from mboostrs.formula import term_variables
from mboostrs.gamboost import gamboost
from mboostrs.gamlss import GaussianLSS, LSSModel, gamboostLSS
from mboostrs.selection import GamlssRefit, GamRefit, gam_refit, gamlss_refit
from mboostrs.terms import (
    TENSOR_MARGINAL_DF,
    ScreenedPair,
    classify_columns,
    pearson_residual,
    screen_interactions,
)

__all__ = ["DiscoveryContrast", "DiscoveryResult", "discover"]

#: Iterations of the screening fits (leave-one-out, interactions, spread).
SCREEN_MSTOP = 200
#: Iterations and step length of the family probe's out-of-fold fits, the
#: probe's own values: it walks each candidate's path to its cross-validated
#: optimum, where the 0.1 of the screening fits needs an order of magnitude
#: more steps for the same held-out accuracy (#585).
FAMILY_MSTOP = 100
FAMILY_NU = 0.3
#: Iterations of the boosted model fitted when no stopping iteration is tuned.
DEFAULT_MSTOP = 100
#: Largest iteration grid of the distributional model's cross-validation.
LSS_GRID_MSTOP = 400
#: First iteration grid of the mean model's tuning. A grid whose risk is still
#: falling at its end is doubled, at most MAX_GRID_EXTENSIONS times.
TUNE_START = 100
MAX_GRID_EXTENSIONS = 2
#: Largest mstop grid the tuning stages search when ``max_mstop`` is not given.
DEFAULT_MAX_MSTOP = TUNE_START * 2 ** MAX_GRID_EXTENSIONS
#: An unordered categorical column with at most this many levels is drafted as
#: ``bols`` (one coefficient per level), not ``brandom``.
CATEGORICAL_BOLS_MAX_LEVELS = 10
#: A positive predictor whose sample skewness exceeds this is drafted as a
#: smooth of its log.
LOG_SKEW = 1.0
#: Learning rate of every boosting fit.
NU = 0.1
#: Learning rate for screening fits on families with non-constant loss curvature.
#: Poisson, Gamma, Binomial and other non-Gaussian families have a Hessian that
#: varies with the fitted values; the Gaussian-tuned NU = 0.1 overshoots where
#: the curvature is high, and the risk path oscillates instead of descending
#: (quine Poisson: 99 of 199 steps rising, minimum at step 3). The smaller step
#: descends monotonically.
NU_NONGAUSSIAN = 0.02


def _screening_nu(family):
    """Learning rate for discovery screening fits, by family.

    Gaussian loss has constant curvature, so the default step is stable.
    Other families retain their smaller 0.02 rate. Screening, tuning and
    the returned fit use the same shared-core choice. The count-complexity
    policy does not change these existing rates.
    """
    from mboostrs import _rust

    # Preserve Gaussian subclass behavior while passing all numeric defaults
    # through the shared core policy used by both bindings.
    key = "gaussian" if isinstance(family, Gaussian) else _family_rust_name(family)
    return _rust.discovery_learning_rate(key)


def _mean_budget_factor(mean_budget, family) -> int:
    """Multiplier of the mean model's step counts under ``mean_budget``.

    ``"fixed"`` keeps the step counts (factor 1). ``"rate_scaled"`` multiplies
    them by ``NU / nu`` rounded to an integer, where ``nu`` is the family's
    screening rate, so that every family gets the boosting budget
    ``mstop * nu`` that the step counts give a Gaussian fit at ``NU``: 5 for
    the 0.02 rate of the count families, 1 for a Gaussian response.
    """
    if mean_budget == "fixed":
        return 1
    return max(1, int(round(NU / _screening_nu(family))))


#: Degrees of freedom of every draft smooth.
DRAFT_DF = 4.0
#: Interaction candidates tried, at most.
MAX_INTERACTIONS = 5
#: Half-splits of the rows whose interaction screen scores are averaged.
SCREEN_SPLITS = 5
#: Standard errors a change must clear to be adopted.
SE_RULE = 2.0
#: Standard errors an interaction must clear. Up to MAX_INTERACTIONS pairs are
#: tried, so the bar is 2 * sqrt(log(MAX_INTERACTIONS)), the same
#: multiple-comparison bar as autospec's pair adoption (2.54 for five pairs).
#: On the claims benchmark, with the interaction test against the matched
#: reference, the bar recalls the planted pair on 20 of 20 full draws at 0.15
#: false pairs per draw (a bar of 2.0 gives 0.20 on the same runs). No pair was
#: adopted on 4 null draws, and on 8 permuted-response draws the screen
#: nominated no pair.
INTERACTION_SE_RULE = 2.0 * math.sqrt(math.log(MAX_INTERACTIONS))
#: |t| of the spread slope needed to nominate a predictor as a spread driver.
SPREAD_NOMINATION_T = 2.0
#: Report line for the interaction stage. A pair whose two members are both brandom
#: terms is never nominated (see `_is_brandom_pair`).
FACTOR_PAIR_NOTE = (
    "Interactions between two brandom factors are not searched: a factor with more "
    "than ten levels is a brandom term, and its out-of-fold score rewards memorized "
    "group effects. To add one by hand, write the cell term with both main effects "
    "in the formula, for example bols(g, by=b) for a factor g with more than ten "
    "levels and a two-valued column b.")
#: Posterior draws for the contrast interval of a distributional refit.
CONTRAST_DRAWS = 1000
#: Rows needed for the analysis at all.
MIN_ROWS = 30
#: Points of the fitted curve used to read the shape of an effect.
CURVE_POINTS = 25
#: Absolute Spearman (rank) correlation at or above which a predictor is read
#: as a copy or monotone transform of the response. Rank correlation is
#: invariant to a monotone transform, so the one threshold catches both a
#: duplicated column and a column such as ``wages`` when the response is
#: ``log(wages)``.
RESPONSE_COPY_RANK = 0.999
#: A whole-number predictor with at least this share of distinct values per
#: row names the rows; it is kept but flagged as a possible identifier.
UNIQUE_INTEGER_SHARE = 0.99
#: Complete pairs two columns need before their rank correlation is read.
MIN_RANK_PAIRS = 10

_EXCEPTIONS = (MboostrsError, ValueError, RuntimeError, FloatingPointError,
               np.linalg.LinAlgError, ZeroDivisionError)


@dataclass
class DiscoveryContrast:
    """The average change in the prediction when one variable moves.

    ``estimate`` is the mean over the training rows of the prediction at
    ``alternative`` minus the prediction at ``reference``, on the scale named
    by ``scale``. ``lower`` and ``upper`` bound it at ``level``. Both are
    ``None`` when the fit has no refit to give an interval or its refit holds
    unsupported contributions fixed. ``note`` states
    what the numbers are, including when the variable was not selected.
    """

    variable: str
    reference: Any
    alternative: Any
    parameter: str | None
    estimate: float
    lower: float | None
    upper: float | None
    level: float
    scale: str
    note: str

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        """The contrast as JSON-ready data. It holds no arrays, so the flag has no effect."""
        return _plain(self.__dict__)


@dataclass(eq=False)
class DiscoveryResult:
    """What :func:`discover` found, with the decisions that produced it.

    Attributes
    ----------
    model : GAMBoostModel or LSSModel
        The final boosted model, fitted on the adopted terms at the tuned
        stopping iteration. When prediction bands are requested, this model
        uses only the selection rows; calibration rows never enter it.
    refit : GamRefit, GamlssRefit or None
        The same terms refitted by REML, with effect intervals and term tests.
        ``None`` when the refit was skipped or refused; the log gives the reason.
    formula : str
        Right-hand side of the adopted mean model (``"1"`` for an offset-only
        model).
    spread_formula : str or None
        Right-hand side of the adopted log-scale model, or ``None``.
    mean_effects : list of dict
        One record per adopted mean term: ``term``, ``variables``, ``shape``,
        ``range`` and ``slope`` (from :func:`~mboostrs.explore.sensitivity`),
        ``transform`` (``"log"`` when the smooth is on ``log(x)``, else
        ``None``; ``range`` and ``slope`` are on the original scale of ``x``),
        and ``change``, ``change se`` (the out-of-fold evidence for keeping it).
    interactions : list of dict
        One record per adopted interaction: ``term``, ``variables``,
        ``change``, ``change se``.
    spread_drivers : list of dict
        One record per adopted log-scale term: ``term``, ``variables``,
        ``parameter``, ``shape``, ``direction`` (``"increasing"`` when the
        log scale rises across the range of the variable), ``change``,
        ``change se``.
    predictions : polars.DataFrame or None
        One row per ``newdata`` row, with columns ``row`` (its position, from
        0), ``mean``, ``lower`` and ``upper``. ``mean`` is the fitted mean on
        the response scale (the ``mu`` of a location-scale fit); ``lower`` and
        ``upper`` bound it at ``level``. A fit with a spread model
        (``spread_formula`` is not ``None``, the Gaussian location-scale
        model) also carries ``sigma``, the fitted scale of that row on the
        response scale, so a score (CRPS, log score) reads ``mean`` and
        ``sigma`` directly instead of rebuilding the scale from the interval.
        The column is absent for a constant-scale fit and for a family with no
        distributional counterpart, so ``spread_formula is not None`` says
        whether it is there. ``None`` when no ``newdata`` was given.
    extrapolation : polars.DataFrame or None
        :meth:`extrapolation_report` of the model for ``newdata``.
    warnings : list of str
        Every warning raised while the analysis ran, as text.
    log : list of dict
        Every decision, with keys ``stage``, ``candidate``, ``score`` (the
        out-of-fold risk, or ``None``), ``change``, ``se``, ``adopted`` and
        ``reason``.
    provenance : dict
        :func:`mboostrs.provenance` of the data and ``seed``.
        ``partition`` records selection/calibration row positions and their
        input namespaces. Positions refer to the original supplied frames,
        before missing-data handling or predictor hygiene.
    interval : ConformalInterval or None
        The calibration result behind ``predictions``, including its marginal
        coverage bound, assumptions and calibration source. ``None`` when no
        band was requested or budget/error prevented calibration.
    stage_seconds : dict
        Measured wall-clock seconds of each stage, keyed by stage name
        (``"mean effects"``, ``"interactions"``, ``"spread"``, ``"tune"``,
        ``"refit"``, and so on, as in ``log``), with ``"predict"`` for the
        prediction and calibration step. A stage that did not run is absent.
        ``mboostrs.estimate_cost("discover", ...)`` gives the first-pass fit
        count of each stage before a run; this is what the stages took.
    constraints : dict
        The rule in force: ``heredity`` and ``forbidden_pairs`` (for the mean
        model), and ``mandatory_terms`` and ``candidates`` keyed by parameter.
        A ``None`` candidate list means every predictor was allowed.
    """

    model: Any
    refit: Any
    formula: str
    spread_formula: str | None
    family: str
    mstop: int
    folds: int
    seed: int
    level: float
    mean_effects: list[dict]
    interactions: list[dict]
    spread_drivers: list[dict]
    predictions: pl.DataFrame | None
    extrapolation: pl.DataFrame | None
    warnings: list[str]
    log: list[dict]
    provenance: dict
    _frame: pl.DataFrame = field(repr=False)
    _variables: frozenset = field(repr=False, default=frozenset())
    constraints: dict = field(default_factory=dict)
    _dropped_refit: Any = field(repr=False, default=None)
    _dropped_variables: frozenset = field(repr=False, default=frozenset())
    interval: Any = field(repr=False, default=None)
    stage_seconds: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"DiscoveryResult(formula={self.formula!r}, "
                f"spread_formula={self.spread_formula!r}, "
                f"mean_effects={len(self.mean_effects)}, "
                f"interactions={len(self.interactions)}, "
                f"spread_drivers={len(self.spread_drivers)}, "
                f"mstop={self.mstop}, decisions={len(self.log)}, "
                f"seconds={sum(self.stage_seconds.values()):.1f})")

    def contrast(self, variable: str, reference, alternative, *,
                 parameter: str | None = None) -> DiscoveryContrast:
        """The average change in the prediction when ``variable`` moves.

        The rows are the training rows with every predictor present, and
        ``variable`` is set to ``reference`` in one copy and to ``alternative``
        in another. The estimate is the mean of the row-wise difference.

        With a Gaussian, Binomial, Poisson or other family the refit is a
        REML GAM, which keeps no posterior draws. Its interval is the sum of
        the pointwise half-widths of the affected effects at each row, a
        conservative bound under the refit's Gaussian coefficient
        approximation. It is conditional on the adopted terms and does not
        account for the selection that chose them. With a distributional
        refit, the interval is the quantiles of the average over
        ``CONTRAST_DRAWS``
        posterior draws of the ``parameter`` (default ``"mu"``).

        When an ordinary refit holds unsupported terms such as trees at their
        boosted contributions, contrasts use the boosted estimate without an
        interval. The refit estimates no uncertainty for the held terms.

        A variable whose draft term backward elimination dropped is marked
        "not selected" in the note. Its estimate is read from the REML refit of
        the model before elimination, and its interval is that refit's interval
        widened to include 0. This applies to a GAM refit. When the adopted model
        is distributional, or the variable had no draft term, the contrast is
        exactly 0 with a note that it was not selected. The scale is ``"link"``
        for a GAM refit and ``"response"`` for a distributional one.
        """
        level = self.level
        if variable not in self._variables and variable in self._dropped_variables \
                and self._dropped_refit is not None:
            return self._dropped_contrast(variable, reference, alternative)
        if variable not in self._variables:
            return DiscoveryContrast(
                variable, reference, alternative, parameter, 0.0, 0.0, 0.0, level,
                "link", f"{variable!r} is not in the model: it was not selected, "
                        "so its contrast is exactly 0.")
        frame = self._frame
        ref_frame = _set_value(frame, variable, reference)
        alt_frame = _set_value(frame, variable, alternative)
        held = getattr(self.refit, "held", ())
        if self.refit is None or held:
            scale = "response"
            kwargs = {"parameter": parameter or "mu"} if self.spread_formula else {}
            result = prediction_contrast(self.model, frame, variable, reference,
                                         alternative, type="response", **kwargs)
            return DiscoveryContrast(
                variable, reference, alternative, kwargs.get("parameter"),
                float(result.estimate), None, None, level, scale,
                ("the REML refit holds fitted contributions fixed and does not estimate "
                 "their uncertainty; the estimate is from the boosted fit without an interval."
                 if held else "no refit was fitted for this model, so there is no interval; "
                 "the estimate is from the boosted fit."))
        if isinstance(self.refit, GamlssRefit):
            return _contrast_lss(self.refit, ref_frame, alt_frame, variable,
                                 reference, alternative, parameter or "mu", level,
                                 self.seed)
        return _contrast_gam(self.refit, ref_frame, alt_frame, variable,
                             reference, alternative, level)

    def _dropped_contrast(self, variable, reference, alternative) -> DiscoveryContrast:
        """Contrast of a variable whose term backward elimination dropped.

        Read from the REML refit of the model before elimination (the draft
        terms, with the adopted interactions), so the interval carries the
        uncertainty of the variable's effect, not an exact zero.
        """
        ref_frame = _set_value(self._frame, variable, reference)
        alt_frame = _set_value(self._frame, variable, alternative)
        c = _contrast_gam(self._dropped_refit, ref_frame, alt_frame, variable, reference,
                          alternative, self.level)
        # The interval is widened to include zero: backward elimination kept the
        # term out, so zero is one of the values the data do not rule out.
        return DiscoveryContrast(
            variable, reference, alternative, None, c.estimate,
            min(c.lower, 0.0), max(c.upper, 0.0), self.level, "link",
            f"{variable!r} is not selected: backward elimination dropped its term. The "
            "estimate is from the REML refit of the model before elimination, and the "
            "interval is that refit's interval widened to include 0. It shows what the term "
            "would contribute, not that it is in the model.")

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        """The result as JSON-ready data (``json.dumps`` accepts it).

        Models and refits are not included; the fitted objects stay on
        ``model`` and ``refit``. Records keep the ``provenance`` entry.
        ``include_arrays=True`` adds ``"fitted"``: the final model's mean on
        the training rows that have every predictor present, on the response
        scale.
        """
        record = {
            "formula": self.formula,
            "spread_formula": self.spread_formula,
            "family": self.family,
            "mstop": self.mstop,
            "folds": self.folds,
            "seed": self.seed,
            "level": self.level,
            "mean_effects": self.mean_effects,
            "interactions": self.interactions,
            "spread_drivers": self.spread_drivers,
            "predictions": None if self.predictions is None
            else self.predictions.to_dicts(),
            "extrapolation": None if self.extrapolation is None
            else self.extrapolation.to_dicts(),
            "warnings": self.warnings,
            "log": self.log,
            "provenance": self.provenance,
            "constraints": self.constraints,
            "interval": None if self.interval is None else self.interval.to_dict(),
            "stage_seconds": dict(self.stage_seconds),
        }
        if include_arrays:
            record["fitted"] = self._fitted()
        return _plain(record)

    def _fitted(self) -> np.ndarray:
        if isinstance(self.model, LSSModel):
            values = self.model.predict(self._frame, parameter="mu", type="response")
        else:
            values = self.model.predict(self._frame, type="response")
        return np.asarray(values, dtype=float).ravel()


def discover(data, response: str, *, newdata=None, calibration=None, family=None, seed: int = 0,
             level: float = 0.9, folds: int = 5, na_action: str = "impute",
             budget_seconds: float | None = None, spread: bool = True,
             simplify_mean: bool = True, mean_step: str = "fixed", mean_budget: str = "fixed",
             interactions: bool = True, heredity: str | Mapping = "none",
             forbidden_pairs=(), mandatory_terms=(),
             candidates: Mapping | None = None, max_mstop: int | None = None,
             extrapolation: str = "warn") -> DiscoveryResult:
    """Run the rule-driven analysis of ``response`` on ``data``.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    Parameters
    ----------
    data : polars or pandas DataFrame, or a mapping of column to values
        The analysis rows. ``response`` names the column to explain; every
        other column is a candidate predictor. With ``newdata`` and no explicit
        ``calibration``, a Rust two-fold assignment reserves half before any
        response-dependent operation. The remaining rows select and fit the
        returned model. Without a prediction band, all rows reach analysis.
    response : str
        Name of the response column.
    newdata : frame, optional
        Rows to predict, with intervals. Each needs the predictor columns the
        final model uses.
    calibration : frame or mapping, optional
        Independently held-out rows with the response and needed predictors.
        They calibrate the already selected fit without a further split or
        refit, allowing all ``data`` rows to reach selection. The caller must
        reserve these rows before choosing the model. Otherwise, ``newdata``
        triggers an automatic half holdout. Row positions are recorded in
        ``result.provenance['partition']``; separate input frames have separate
        position namespaces. Calibration responses affect band endpoints only.
    family : Family, optional
        The loss. ``None`` infers it from the response as
        :func:`~mboostrs.autospec` does.
    seed : int
        Seed of every fold assignment and posterior draw. The boosting itself
        is deterministic.
    level : float
        Coverage of the effect, contrast and prediction intervals.
    folds : int
        Cross-validation folds. The count is lowered for small data (at least
        three folds, one per 50 rows); the log records the lowered count.
    na_action : {"impute", "impute_model", "impute_model_split", "omit", "error"}
        Passed to every fit. Screens use the complete rows. ``"impute_model"``
        fills a missing numeric predictor from a model on the other predictors
        (see ``gamboost``); the screens still fill it with the median.
        ``"impute_model_split"`` adds the opt-in observed/filled split of
        ``gamboost``.
    budget_seconds : float, optional
        Seconds the stages may take in total, by :func:`~mboostrs.estimate_cost`.
        A stage whose estimate would take the run past this is skipped and
        logged. ``None`` runs every stage.
    simplify_mean : bool, default True
        For standard Poisson/NBinomial fits, compare the autogenerated smooth
        mean draft with a linear backbone on the same fitting folds. Prefer
        the simpler draft when its risk increase is within two paired fold
        standard errors. Mandatory terms and predictor transformations are
        preserved. This precedes ordinary variable removal; it is a selection
        heuristic, not a curvature test. ``False`` retains the smooth draft.
    mean_step : {"auto", "fixed", "line_search"}, default "fixed"
        Step rule for ordinary mean screening, comparison, stopping and final
        fits. ``"fixed"`` is the default. ``"auto"`` uses the shared Rust line search for NBinomial and
        fixed steps for other families. Line search minimizes fitting loss
        along the selected direction before shrinking by the existing ``nu``;
        its effective update therefore changes. Family inference and distributional fitting retain
        their existing update rules. The learning-rate values and stopping
        caps are unchanged.
    mean_budget : {"fixed", "rate_scaled"}, default "fixed"
        Step counts of the ordinary mean model's screening, comparison and
        tuning fits. ``"fixed"`` (the default) uses 200 steps for each
        screening and comparison fit and a tuning grid that starts at 100
        steps and is capped at 400. ``"rate_scaled"`` multiplies all three by
        ``0.1 / nu`` for the family's learning rate ``nu``: by 5 for the
        count families, whose rate is 0.02, so that their fits move as far
        along the boosting path (``mstop * nu`` = 20 at screening, 40 at the
        cap) as a Gaussian fit at rate 0.1 does; a Gaussian fit is unchanged.
        It costs about five times the boosting steps of a count fit. An
        explicit ``max_mstop`` still sets the cap. Family inference and the
        distributional stages are unchanged. This is an opt-in candidate
        that has not been evaluated on the package's full benchmark suite.
    spread : bool
        Whether to test a log-scale (distributional) model for a Gaussian
        response.
    interactions : bool
        Whether to screen and test pairwise interactions. For two continuous
        columns, the smooth tensor is tested first, then a depth-three joint
        tree is compared against the current model on the same folds. The tree
        replaces that pair's tensor only when it clears the same paired-risk
        margin. A selected tree has no smooth REML effect intervals; the refit
        refusal is recorded in the log, and contrasts use the boosted model.
        After an interaction is adopted, discarded draft main terms are
        reconsidered at the existing two-SE mean-effect margin, since the
        interaction can explain variation that masked a weaker main effect.
    heredity : {"none", "weak", "strong"}, or mapping, default "none"
        When an interaction may enter the mean model. ``"strong"`` admits a
        pair only when both of its columns have a main-effect term in the mean
        model at the time the pair is tested; ``"weak"`` when at least one
        does. Backward elimination runs before the interaction stage, so the
        main effects a pair needs are the ones that survived it. ``"none"``
        admits any pair the screen nominates and the comparison adopts, which
        is how a pure interaction such as ``sin(3 * x1 * x2)`` is found.
    forbidden_pairs : iterable of (str, str), or mapping, optional
        Column pairs never nominated and never adopted as interactions.
    mandatory_terms : iterable of str, or mapping, optional
        Columns whose mean term backward elimination never drops. Under the
        key ``"sigma"``, one column whose log-scale term is included without
        the spread test.
    candidates : mapping, optional
        Keys ``"mu"`` and ``"sigma"``: the predictor columns each parameter's
        search may use. ``None`` for a parameter allows every predictor.
        Mandatory columns are always candidates.
    max_mstop : int, optional
        Largest stopping iteration the tuning stages search. The mean model's
        grid starts at 100 steps and doubles while the cross-validated minimum
        lies in the last 5% of the grid; the distributional model's grid starts
        at 400 and doubles the same way. When the minimum still lies there at
        the cap, discover warns and keeps the grid it has. The risk may be
        falling or flat there, so raise ``max_mstop`` when the process needs
        more steps.
        ``None`` (the default) caps the mean model at
        400 steps and the distributional model at 1600. A value sets both caps:
        above the default lets the search double further, below 100 limits the
        first grid. A tuning stopped at its cap says so in ``result.warnings``
        and names this argument.
    extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
        How the cross-validation folds of the screens and of the tuning treat
        a held-out row outside a term's training range. The warning names
        ``discover(extrapolation=...)``, and it is issued once per call, with
        the number of fold predictions it covers. The final ``newdata``
        predictions always use ``"clip"``.

    A bare value of ``heredity`` or ``forbidden_pairs`` applies to the mean
    model (``"mu"``), since interactions are only searched there; a bare
    ``mandatory_terms`` or ``candidates`` applies to ``"mu"`` too. A mapping
    must use the parameter names above and raises on any other key.

    Returns
    -------
    DiscoveryResult

    Raises
    ------
    ValueError
        When ``response`` is not a column, ``level`` is outside (0, 1), or the
        selection data have fewer than 30 complete rows. An automatic holdout
        therefore needs at least 60 rows before missing-data handling. The
        held-out set must also meet the conformal minimum for ``level``.

    Notes
    -----
    Warnings are recorded in ``result.warnings`` and are not re-emitted.
    A numeric predictor that is a copy or a monotone transform of the response
    (absolute rank correlation at least 0.999) is dropped before the draft and
    named in ``result.warnings``, so a response such as ``y = log(wages)`` can
    no longer select ``wages`` as a predictor. A whole-number predictor with
    one distinct value per row is only warned about, because it may be an
    ordered quantity rather than an identifier. Optional stages that fail, for
    example a spread test on a constant response, are logged with their error
    and the run continues. When backward
    elimination leaves no mean term, the mean model is intercept-only
    (``formula == "1"``, that is ``y ~ 1``) and ``result.warnings`` says so and
    names the family; every prediction is then the mean of the response.

    Examples
    --------
    >>> import numpy as np, polars as pl, mboostrs as mb
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> junk = rng.normal(size=300)
    >>> y = np.sin(x) + rng.normal(0, 0.3, 300)
    >>> result = mb.discover(pl.DataFrame({"x": x, "junk": junk, "y": y}), "y", seed=0)
    >>> [e["variables"] for e in result.mean_effects]
    [['x']]
    """
    if calibration is data and calibration is not None:
        raise ValueError("calibration must be held-out data, not the same object used for discovery")
    frame = _as_frame(data)
    if response not in frame.columns:
        raise ValueError(
            f"response {response!r} is not a column of data; pass the name of "
            f"the response column, one of {frame.columns}")
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}; "
                         "pass 0.9 for a 90% interval")
    folds = check_whole_int("folds", folds, 2)
    if max_mstop is not None:
        max_mstop = check_whole_int("max_mstop", max_mstop, 1)
    if na_action not in ("impute", "impute_model", "impute_model_split", "omit", "error"):
        raise ValueError("na_action must be 'impute', 'impute_model', 'impute_model_split', 'omit' or 'error', "
                         f"got {na_action!r}")
    if budget_seconds is not None and budget_seconds < 0:
        raise ValueError("budget_seconds must be zero or positive; pass None for no budget")
    if extrapolation not in POLICIES:
        raise ValueError(f"extrapolation must be one of {POLICIES}, got {extrapolation!r}; "
                         "pass 'warn' (the default), 'allow', 'clip' or 'error'")
    if max_mstop is not None:
        max_mstop = check_int_at_least("max_mstop", max_mstop, 1, allow_none=False)
    if isinstance(family, type):
        family = family()
    rules = _parameter_rules(frame.columns, response, heredity=heredity,
                             forbidden_pairs=forbidden_pairs,
                             mandatory_terms=mandatory_terms, candidates=candidates)
    new_frame = None if newdata is None else _as_frame(newdata, "newdata")
    frame, calibration_frame, partition = _calibration_partition(
        frame, calibration, new_frame is not None, seed, level)
    if calibration_frame is not None and response not in calibration_frame.columns:
        raise ValueError(f"calibration rows have no response column {response!r}")

    fit_na = na_action
    # With a reserved holdout, keep the caller's fitted missing-data recipe.
    # Neither calibration nor prediction rows choose the fitting policy.
    no_missing = not _has_nulls(frame) and calibration_frame is None
    if na_action in ("impute", "impute_model", "impute_model_split") and no_missing:
        # "impute" adds a fitted imputation to every model, which the
        # distributional intervals refuse. With no missing value anywhere,
        # "omit" fits the same rows without it.
        fit_na = "omit"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Every extrapolation warning of the stages is collected by the call and
        # issued once, naming discover(extrapolation=...); the repeated ones of
        # other kinds are grouped by collect_warnings.
        with collect_warnings(), routine_warnings(stacklevel=2, owner="discover"):
            analysis = _Analysis(frame, response, family=family, seed=seed, level=level,
                                 folds=folds, na_action=fit_na,
                                 budget_seconds=budget_seconds, spread=spread, simplify_mean=simplify_mean,
                                 mean_step=mean_step, mean_budget=mean_budget,
                                 interactions=interactions, rules=rules,
                                 max_mstop=max_mstop, extrapolation=extrapolation,
                                 partition=partition)
            if fit_na != na_action:
                analysis._note("setup", None, "no missing values in the data or newdata: fits "
                               "use na_action='omit', the same rows without the imputation "
                               "terms that 'impute' adds to every model")
            analysis.run()
            predict_start = time.perf_counter()
            analysis.predict(new_frame, calibration_frame)
            if new_frame is not None:
                analysis.stage_seconds["predict"] = time.perf_counter() - predict_start
        result = analysis.result([])
        result.warnings = _unique([*analysis.extra_warnings,
                                   *(str(w.message) for w in caught)])
    return result


def plan_discover_cost(args, kwargs, bound):
    """(fits, runner, explanation, stages) for ``estimate_cost("discover", ...)``.

    The stages are counted on their first pass, from the draft terms and the
    folds ``discover`` would use. The runner is one screening fit of the draft
    mean model on the complete rows.
    """
    a, _extra = bound(discover, args, kwargs)
    frame = _as_frame(a["data"])
    response = a["response"]
    if response not in frame.columns:
        raise ValueError(f"response {response!r} is not a column of data")
    family = a["family"]
    if isinstance(family, type):
        family = family()
    rules = _parameter_rules(frame.columns, response, heredity=a["heredity"],
                             forbidden_pairs=a["forbidden_pairs"],
                             mandatory_terms=a["mandatory_terms"], candidates=a["candidates"])
    new_frame = None if a["newdata"] is None else _as_frame(a["newdata"], "newdata")
    frame, calibration_frame, partition = _calibration_partition(
        frame, a["calibration"], new_frame is not None, a["seed"], a["level"])
    fit_na = a["na_action"]
    if fit_na in ("impute", "impute_model", "impute_model_split") \
            and not _has_nulls(frame) and calibration_frame is None:
        fit_na = "omit"
    analysis = _Analysis(
        frame, response, family=family, seed=a["seed"], level=a["level"], folds=a["folds"],
        na_action=fit_na, budget_seconds=None, spread=a["spread"],
        simplify_mean=a["simplify_mean"], mean_budget=a["mean_budget"],
        interactions=a["interactions"], rules=rules,
        max_mstop=a["max_mstop"], extrapolation=a["extrapolation"], partition=partition)
    terms, _notes = analysis._draft_terms()
    if analysis.mu_columns is not None:
        terms = [t for t in terms if set(term_variables(t)) <= analysis.mu_columns]
    folds, n_terms = analysis.folds, len(terms)
    stages: dict[str, int] = {}
    if analysis.do_simplify_mean and isinstance(analysis.family, (Poisson, NBinomial)):
        simple = [analysis._count_linear_alternatives.get(t, t)
                  if not (set(term_variables(t)) & analysis.mandatory_mu) else t
                  for t in terms]
        if simple != terms:
            stages["count mean forms"] = 2 * folds
    stages["mean effects"] = (n_terms + 1) * folds if n_terms else 0
    stages["interactions"] = 1 if analysis.do_interactions else 0
    gaussian_spread = analysis.do_spread and isinstance(analysis.family, Gaussian)
    stages["spread"] = 1 if gaussian_spread and not analysis.mandatory_sigma else 0
    stages["tune"] = folds + 1 if n_terms else 0
    stages["final model"] = 1
    fits = sum(stages.values())
    analysis.mean_terms = list(terms)

    def run():
        analysis._fit_mean()

    why = ("first pass of each stage, " + ", ".join(
        f"{stage} {count}" for stage, count in stages.items() if count)
        + f" ({n_terms} draft terms, {folds} folds); rounds of backward elimination, "
        "interaction candidates, spread tests and the tuning of a spread model add fits "
        "that depend on the data")
    return fits, run, why, stages


def _calibration_partition(frame, calibration, predict, seed, level):
    """Separate raw rows before any response-dependent discovery operation."""
    n = frame.height
    if calibration is not None:
        held = _as_frame(calibration, "calibration")
        selection_rows, calibration_rows = list(range(n)), list(range(held.height))
        origin = "caller-supplied calibration frame; independent holdout is the caller's responsibility"
        calibration_input = "calibration"
    elif predict:
        if n < 2:
            raise ValueError("discovery calibration needs both selection and calibration rows")
        weights = cv_weights(n, type="kfold", B=2, seed=seed)[:, 0]
        selection_rows = np.flatnonzero(weights > 0).tolist()
        calibration_rows = np.flatnonzero(weights == 0).tolist()
        frame, held = frame[selection_rows], frame[calibration_rows]
        origin = "first held-out fold of a Rust two-fold assignment, reserved before discovery"
        calibration_input = "data"
    else:
        selection_rows, calibration_rows = list(range(n)), []
        held, origin, calibration_input = None, "no prediction band requested", None
    if held is not None:
        if frame.height < MIN_ROWS:
            raise ValueError(f"the calibration holdout leaves {frame.height} selection rows; "
                             f"discover needs at least {MIN_ROWS} complete selection rows. "
                             "Add data or supply a separate calibration= frame.")
        minimum = _min_calibration_size(level)
        if held.height < minimum:
            raise ValueError(f"level={level} needs at least {minimum} calibration rows; got {held.height}")
    return frame, held, {"selection_input": "data", "selection_rows": selection_rows,
                         "calibration_input": calibration_input, "calibration_rows": calibration_rows,
                         "source": origin, "seed": seed, "input_rows": n}


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


class _Analysis:
    """The stages of :func:`discover`, sharing one log and one budget clock."""

    def __init__(self, frame, response, *, family, seed, level, folds, na_action,
                 budget_seconds, spread, interactions, rules, max_mstop=None, simplify_mean=True,
                 mean_step="fixed", mean_budget="fixed",
                 extrapolation="warn", partition=None):
        self.frame = frame
        self.response = response
        self.seed = seed
        self.level = level
        self.na_action = na_action
        self.budget = budget_seconds
        #: Largest stopping iteration the tuning stages search; None for the defaults.
        self.max_mstop = max_mstop
        #: ``extrapolation=`` of the fold fits (the screens and the tuning).
        self.fold_extrapolation = extrapolation
        self.do_spread = spread
        self.do_simplify_mean = simplify_mean
        if not isinstance(mean_step, str) or mean_step not in ("auto", "fixed", "line_search"):
            raise ValueError("mean_step must be 'auto', 'fixed' or 'line_search'")
        self.requested_mean_step = mean_step
        if not isinstance(mean_budget, str) or mean_budget not in ("fixed", "rate_scaled"):
            raise ValueError("mean_budget must be 'fixed' or 'rate_scaled'")
        self.mean_budget = mean_budget
        self.do_interactions = interactions
        self.t0 = time.perf_counter()
        #: Measured seconds of each stage, by stage name.
        self.stage_seconds: dict[str, float] = {}
        self.log: list[dict] = []
        self._estimated_stages: set[str] = set()
        self.partition = partition
        if partition is not None:
            self._note("calibration", None,
                       f"{partition['source']}; {len(partition['selection_rows'])} selection rows, "
                       f"{len(partition['calibration_rows'])} calibration rows")
        self.predictors = [c for c in frame.columns if c != response]
        # A predictor that is a copy or monotone transform of the response is
        # dropped before the draft; a whole-number predictor that names rows is
        # kept but warned about. Both are recorded in ``result.warnings`` and the
        # log, so a caller reading only the result sees what was left out.
        self.dropped_predictors, self.identifier_columns = _predictor_hygiene(
            frame, response, self.predictors)
        if self.dropped_predictors:
            dropped = [c for c, _ in self.dropped_predictors]
            frame = frame.drop(dropped)
            self.frame = frame
            self.predictors = [c for c in self.predictors if c not in set(dropped)]
        self.kind_of = {ck.name: ck.kind for ck in
                        classify_columns(frame, response=response, df=DRAFT_DF)}
        self.complete = frame.drop_nulls(subset=[response, *self.predictors])
        n = self.complete.height
        if n < MIN_ROWS:
            raise ValueError(
                f"discover needs at least {MIN_ROWS} rows with a response and every "
                f"predictor present; found {n}. Add rows or fit the model directly.")
        self.folds = min(folds, max(3, n // 50))
        if self.folds != folds:
            self._note("setup", None, f"{folds} folds lowered to {self.folds} for {n} rows "
                       "(at least three folds, one per 50 rows)")
        self.rules = rules
        self.heredity = rules["heredity"]["mu"]
        self.forbidden = rules["forbidden_pairs"]["mu"]
        self.mandatory_mu = set(rules["mandatory_terms"]["mu"])
        self.mandatory_sigma = rules["mandatory_terms"]["sigma"]
        mu_candidates = rules["candidates"]["mu"]
        sigma_candidates = rules["candidates"]["sigma"]
        self.mu_columns = None if mu_candidates is None else set(mu_candidates)
        self.sigma_columns = None if sigma_candidates is None else set(sigma_candidates)
        self.mean_terms: list[str] = []
        self.draft_terms: list[str] = []
        self.dropped_refit = None
        self.dropped_variables: set[str] = set()
        #: Predictor name -> transform its smooth is drafted on ("log").
        self.transforms: dict[str, str] = {}
        self._count_linear_alternatives: dict[str, str] = {}
        if family is None:
            family = self._infer_family()
        self.family = family
        self.family_name = type(family).__name__
        from mboostrs import _rust

        family_key = "gaussian" if isinstance(family, Gaussian) else _family_rust_name(family)
        self._resolved_mean_step = (_rust.discovery_mean_step(family_key)
                                    if mean_step == "auto" else mean_step)
        #: Multiplier of the mean model's screening and tuning step counts.
        self._budget_factor = _mean_budget_factor(mean_budget, family)
        if self.mandatory_sigma and not (self.do_spread and isinstance(family, Gaussian)):
            raise ValueError(
                "mandatory_terms names a log-scale (sigma) column, which needs the spread "
                "stage: a Gaussian response with spread=True. This fit is "
                f"{self.family_name} with spread {'on' if self.do_spread else 'off'}; "
                "remove the 'sigma' entry from mandatory_terms")
        #: Boundary notes that go to ``result.warnings`` next to the library's warnings.
        self.extra_warnings: list[str] = []
        self.evidence: dict[str, tuple[float, float]] = {}
        self.interaction_records: list[dict] = []
        self.spread_term: str | None = None
        self.spread_record: dict | None = None
        self.model = None
        self.refit = None
        self.mstop = DEFAULT_MSTOP
        self.predictions = None
        self.interval = None
        self.extrapolation = None

    # -- bookkeeping -------------------------------------------------------

    def _infer_family(self):
        """The response's family: the dtype rule's candidate, checked by the probe.

        The dtype rule names a candidate from the values alone. A positive
        response whose standard deviation is more than 10% of its mean is
        read as GammaReg, because a Gamma's variance is ``mu**2 / shape``:
        its spread grows with its level, while a Gaussian's is constant.
        That is a statement about the variance function, not about how the
        family behaves under boosting, so the family probe measures it before
        the candidate is trusted.

        The probe fits the draft mean model under Gaussian on the analysis
        folds and correlates the absolute out-of-fold residual with the
        fitted mean. The correlation is zero under constant variance and
        positive when the spread tracks the level; a correlation above
        0.25 says Gamma, below it Gaussian, and within two standard errors
        of the line the probe declines to choose and the dtype candidate
        stands.

        A Gamma (or count) verdict is then checked for what boosting does to
        it. GammaReg's gradient is the residual scaled by the family's shape
        parameter, which is recalibrated at every step and grows as the fit
        improves, so a fixed step length overshoots and the risk path stops
        descending. The stages that follow read the path (leave-one-out
        changes at a fixed step count), and on such a path their numbers
        carry no signal: on SLID's log-wages every term was dropped and the
        model emptied (#585). The check refits the candidate and Gaussian,
        each at its own cross-validated step count, and compares their
        out-of-fold squared error: the candidate is replaced by Gaussian when
        it is worse by 2% and by two paired standard errors, or by half
        again whatever the standard error (a single diverged fold inflates
        the error enough to excuse anything).

        The probe's distributional candidates (GammaLSS, BetaLSS) are not
        offered here. The stages that follow fit one mean model, and the
        spread stage is this analysis's own answer to the scale question.
        The probe returns ``None`` when the response leaves nothing to
        compare (a binary or a categorical response); the dtype candidate
        then stands, as it does when the probe itself fails.
        """
        inferred = autospec(self.frame, self.response)[2]
        terms, _ = self._draft_terms()
        if self.mu_columns is not None:
            terms = [t for t in terms if set(term_variables(t)) <= self.mu_columns]
        formula = f"{quote_response(self.response)} ~ {_join(terms)}"
        weights = cv_weights(self.complete.height, type="kfold", B=self.folds,
                            seed=self.seed)
        try:
            evidence = probe_family(
                self.complete, self.response, formula, inferred,
                folds=weights, mstop=FAMILY_MSTOP, nu=FAMILY_NU)
            if evidence is not None:
                name = type(evidence.recommended).__name__
                if name == "GammaReg":
                    evidence = _validate_gamma(
                        self.complete, self.response, formula, weights,
                        FAMILY_NU, None, evidence)
                elif name in ("Poisson", "NBinomial"):
                    evidence = _validate_against_gaussian(
                        evidence.recommended, self.complete, self.response,
                        formula, weights, FAMILY_NU, None, evidence)
        except _EXCEPTIONS as exc:
            self._note("family", type(inferred).__name__,
                       f"the family probe failed ({type(exc).__name__}: {exc}), so the "
                       "dtype-inferred family stands")
            return inferred
        if evidence is None:
            self._note("family", type(inferred).__name__,
                       "the probe has no comparison to make on this response (a binary "
                       "or a categorical response), so the dtype-inferred family stands")
            return inferred
        family = evidence.recommended
        if getattr(family, "param_names", None) is not None:
            # probe_family, called without validate, never recommends a
            # distributional family; the check keeps a later change honest.
            family = inferred
        self._note("family", type(family).__name__, evidence.sentence, adopted=True)
        return family

    def _note(self, stage, candidate, reason, *, score=None, change=None, se=None,
              adopted=False, estimated_seconds=None):
        self.log.append({"stage": stage, "candidate": candidate, "score": _num(score),
                         "change": _num(change), "se": _num(se),
                         "adopted": bool(adopted), "reason": reason,
                         "estimated_seconds": _num(estimated_seconds)})

    def _spent(self) -> float:
        return time.perf_counter() - self.t0

    def _estimate(self, stage, call, *args, **kwargs) -> float:
        """Estimated seconds of one call, logged once per stage before it runs."""
        try:
            seconds = estimate_cost(call, *args, **kwargs).estimated_seconds
        except _EXCEPTIONS as exc:
            # The stage itself reports the failure when it runs; the estimate
            # is only a budget input, so it counts as zero here.
            self._log_estimate(stage, 0.0, f"the first {call} of this stage, which could not "
                               f"be estimated ({type(exc).__name__}); it runs without a time "
                               "check")
            return 0.0
        self._log_estimate(stage, seconds, f"the first {call} of this stage")
        return seconds

    def _log_estimate(self, stage, seconds, what):
        """One log line per stage with its estimated time, written before the stage runs."""
        if stage in self._estimated_stages:
            return
        self._estimated_stages.add(stage)
        self._note("estimate", stage, f"about {seconds:.2f} s for {what} (estimate_cost), "
                   "logged before the stage runs", estimated_seconds=seconds)

    def _admit(self, stage, call, *args, **kwargs) -> bool:
        """Whether the stage fits the budget; logs a skip when it does not."""
        return self._admit_seconds(stage, self._estimate(stage, call, *args, **kwargs))

    def _admit_seconds(self, stage, estimate: float) -> bool:
        if self.budget is None:
            return True
        if self._spent() + estimate <= self.budget:
            return True
        self._note(stage, None, f"skipped: about {estimate:.2f} s more would take the run past "
                   f"budget_seconds={self.budget:g} ({self._spent():.2f} s used)",
                   estimated_seconds=estimate)
        return False

    def _required(self, stage, call, *args, **kwargs):
        """Estimate of a stage the result cannot be built without; never skipped."""
        seconds = self._estimate(stage, call, *args, **kwargs)
        if self.budget is not None and self._spent() + seconds > self.budget:
            self._note(stage, None, f"not skipped: the final model is required, so this "
                       f"stage runs although about {seconds:.2f} s more takes the run past "
                       f"budget_seconds={self.budget:g}", estimated_seconds=seconds)
        return seconds

    def _fail(self, stage, candidate, exc):
        self._note(stage, candidate, f"error: {type(exc).__name__}: {exc}")

    # -- stages ------------------------------------------------------------

    def run(self):
        self._note("rule", None, _rule_sentence(self.rules))
        self._record_predictor_hygiene()
        terms = self._draft()
        if self.mu_columns is not None:
            terms = [t for t in terms if set(term_variables(t)) <= self.mu_columns]
        drafted = {v for t in terms for v in term_variables(t)}
        absent = sorted(self.mandatory_mu - drafted)
        if absent:
            raise ValueError(
                f"mandatory_terms names {absent}, which the draft gives no mean term (the "
                "column is constant, a copy or monotone transform of the response that the "
                "hygiene check dropped, or outside candidates); remove it from mandatory_terms")
        self._note("draft", _join(terms), f"{len(terms)} terms from the columns, each continuous "
                   f"column a P-spline at df={DRAFT_DF:g}")
        self.draft_terms = list(terms)
        self.mean_terms = list(terms)
        if self.do_simplify_mean and isinstance(self.family, (Poisson, NBinomial)):
            self._guard("count mean forms", self._count_mean_forms)
            terms = list(self.mean_terms)
        self._guard("mean effects", lambda: self._mean_effects(terms))
        if self.do_interactions:
            self._guard("interactions", self._interactions)
        else:
            self._note("interactions", None, "not run: interactions=False")
        if self.interaction_records:
            self._guard("mean recovery", self._recover_mean_effects)
        if self.do_spread and isinstance(self.family, Gaussian):
            self._guard("spread", self._spread)
        elif self.do_spread:
            self._note("spread", None, f"not run: {self.family_name} has no distributional "
                       "counterpart in mboostrs; spread drivers are tested for Gaussian only")
        else:
            self._note("spread", None, "not run: spread=False")
        self._guard("tune", self._tune_and_fit)
        if self.model is None:
            self._note("tune", None, "error in the tuning stage: the final model is fitted at "
                       f"mstop={DEFAULT_MSTOP} without tuning")
            if self.spread_term is None:
                self.model = self._fit_final(DEFAULT_MSTOP)
            else:
                self.mstop = DEFAULT_MSTOP
                self.model = gamboostLSS(self._lss_formulas(), self.frame, family=GaussianLSS(),
                                         mstop=DEFAULT_MSTOP, na_action=self.na_action)
        self._guard("refit", self._refit)
        self._guard("dropped terms", self._refit_draft)
        self._warn_if_empty_mean()

    def _warn_if_empty_mean(self) -> None:
        """Record a warning when no mean term survived, so callers cannot miss it.

        An intercept-only mean model (``formula == "1"``) predicts the response
        mean for every row. That is a real result, not a failure, but the log
        note alone is easy to overlook, so it also goes to ``result.warnings``.
        """
        if self.mean_terms:
            return
        self.extra_warnings.append(
            f"no mean term survived: the mean model is {quote_response(self.response)} ~ 1, an "
            f"intercept-only fit of family {self.family_name}. Every prediction is "
            f"the mean of {self.response}; read the log for why each term was "
            "dropped, or pass mandatory_terms to keep one")

    def _record_predictor_hygiene(self) -> None:
        """Log and warn about response copies and row identifiers among the columns.

        ``_Analysis.__init__`` drops a response copy from the frame before the
        draft and keeps the row identifiers. Both are recorded here, in the log
        and in ``result.warnings``, so a caller reading only the result sees what
        discovery left out and why.
        """
        for column, correlation in self.dropped_predictors:
            self._note("hygiene", column, f"dropped: rank correlation {correlation:.3f} with "
                       f"{self.response}, at or above {RESPONSE_COPY_RANK:g}; the column is a "
                       "copy or monotone transform of the response, so it would leak it into "
                       "the model", adopted=False)
            self.extra_warnings.append(
                f"dropped predictor {column!r}: its rank correlation with the response "
                f"{self.response!r} is {correlation:.3f}, at or above {RESPONSE_COPY_RANK:g}. "
                "The column is a copy or a monotone transform of the response, so it cannot be "
                "known before the response and would predict it trivially. Pass the column as "
                "the response, or leave it out; fit it deliberately only if that leakage is "
                "the intended model")
        for column, detail in self.identifier_columns:
            self._note("hygiene", column, f"warning: {detail}; the column looks like a row "
                       "identifier, which carries no effect to share across rows", adopted=False)
            self.extra_warnings.append(
                f"predictor {column!r} looks like a row identifier ({detail}). A column that "
                "names rows has no effect to share across them; keep it only if its values are "
                "an ordered quantity measured before the response, such as a year or an index")

    def _draft_terms(self) -> tuple[list[str], list[tuple[str, str]]]:
        """One mean term per predictor, with the notes that explain them.

        An unordered categorical column with at most CATEGORICAL_BOLS_MAX_LEVELS
        levels is a ``bols`` term. A positive predictor with sample skewness
        above LOG_SKEW has its smooth on ``log(x)``, written into the formula,
        so the log is replayed on new rows and contrasts take the original
        scale. The notes are returned instead of logged, so a caller that
        needs the terms alone (the family probe) writes no log entry.

        The draft says nothing about the family: it depends only on the
        columns, so it is the same before and after the family is inferred.
        """
        terms: list[str] = []
        notes: list[tuple[str, str]] = []
        for kind in classify_columns(self.frame, response=self.response, df=DRAFT_DF):
            term = kind.term
            if (kind.kind == "categorical" and kind.dtype == "categorical"
                    and not kind.ordered and kind.levels <= CATEGORICAL_BOLS_MAX_LEVELS):
                term = f"bols({quote_name(kind.name)})"
                notes.append((term, f"{kind.levels} unordered levels: one coefficient per "
                              "level as bols; brandom shrinks level effects and is "
                              "for more levels"))
            elif kind.kind == "continuous" and term.startswith(f"bbs({quote_name(kind.name)}, "):
                skew = _log_skew(self.frame[kind.name])
                if skew is not None:
                    values = self.frame[kind.name].drop_nulls().cast(pl.Float64)
                    self.transforms[kind.name] = "log"
                    term = term.replace(f"bbs({quote_name(kind.name)}, ", f"bbs(log({quote_name(kind.name)}), ", 1)
                    notes.append((f"log({quote_name(kind.name)})", "right-skewed positive predictor: "
                                  f"sample skewness {skew:.2f} (above {LOG_SKEW:g}), "
                                  f"minimum {float(values.min()):g} above 0, so its "
                                  f"smooth is fitted on log({quote_name(kind.name)}); contrasts and "
                                  "predictions take the original scale"))
            if kind.kind == "continuous" and term.startswith("bbs("):
                argument = (f"log({quote_name(kind.name)})" if self.transforms.get(kind.name) == "log"
                            else quote_name(kind.name))
                self._count_linear_alternatives[term] = f"bols({argument})"
            terms.append(term)
        return terms, notes

    def _draft(self) -> list[str]:
        """The draft terms of every predictor; the notes are logged under 'draft'."""
        terms, notes = self._draft_terms()
        for candidate, reason in notes:
            self._note("draft", candidate, reason)
        return terms

    def _refit_draft(self):
        """REML refit of the model before elimination, for the contrasts of dropped terms.

        The model is the draft terms plus the adopted interactions. A draft term
        that elimination dropped has no term in the result, so its contrast comes
        from this refit. Nothing is refitted when elimination dropped no term.
        """
        dropped = [t for t in self.draft_terms if t not in self.mean_terms]
        if not dropped or self.spread_term is not None:
            return
        full = list(self.draft_terms) + [t for t in self.mean_terms
                                         if t not in self.draft_terms]
        control = boost_control(max(self.mstop, 1), _screening_nu(self.family), step=self.mean_step)
        if not self._admit("refit", "gamboost", f"{quote_response(self.response)} ~ {_join(full)}", self.frame,
                           family=self.family, control=control, na_action=self.na_action):
            return
        model = gamboost(f"{quote_response(self.response)} ~ {_join(full)}", self.frame, family=self.family,
                         control=control,
                         na_action=self.na_action)
        self.dropped_refit = self._ordinary_refit(model, terms=full, stage="dropped terms")
        if self.dropped_refit is None:
            return
        self.dropped_variables = {v for t in dropped for v in term_variables(t)}
        self._note("refit", _join(dropped), "REML refit of the model before elimination, used "
                   "only for the contrasts of the dropped terms" + self._held_note(self.dropped_refit),
                   adopted=False)

    def _guard(self, stage, fn):
        start = time.perf_counter()
        try:
            fn()
        except _EXCEPTIONS as exc:
            self._fail(stage, None, exc)
        finally:
            self.stage_seconds[stage] = (self.stage_seconds.get(stage, 0.0)
                                         + time.perf_counter() - start)

    def _count_mean_forms(self):
        """Plan paired fits; the shared core computes the complexity decision."""
        from mboostrs import _rust

        current = list(self.mean_terms)
        simple = [self._count_linear_alternatives.get(term, term)
                  if not (set(term_variables(term)) & self.mandatory_mu) else term
                  for term in current]
        if simple == current:
            return
        specifications = [_join(current), _join(simple)]
        rate = _screening_nu(self.family)
        if not self._admit("count mean forms", "compare_specifications", specifications,
                           self.frame, self.response, family=self.family, folds=self.folds,
                           mstop=self.screen_mstop, nu=rate, step=self.mean_step, seed=self.seed,
                           na_action=self.na_action):
            return
        rows = compare_specifications(specifications, self.frame, self.response,
            family=self.family, folds=self.folds, mstop=self.screen_mstop, nu=rate, step=self.mean_step, seed=self.seed,
            na_action=self.na_action, extrapolation=self.fold_extrapolation, include_evaluations=True)
        if any(row.get("error") is not None or row.get("evaluation") is None for row in rows):
            self._note("count mean forms", None, "not adopted: a paired shape comparison could not be evaluated")
            return
        rich_scores, simple_scores = (np.asarray(row["evaluation"].fold_scores, dtype=np.float64) for row in rows)
        chosen, change, se = _rust.prefer_simpler_cv(simple_scores, rich_scores, SE_RULE)
        self._note("count mean forms", specifications[1],
            "adopted: linear backbone within two paired fold standard errors" if chosen else
            "kept smooth backbone: its risk improvement exceeds two paired fold standard errors",
            change=change, se=se, adopted=chosen)
        self.log[-1].update(rich=specifications[0], simple=specifications[1],
            simple_scores=simple_scores.tolist(), rich_scores=rich_scores.tolist(), nu=rate,
            step=self.mean_step)
        if chosen:
            self.mean_terms = list(simple)
            self.draft_terms = list(simple)

    def _mean_effects(self, terms: list[str]) -> None:
        """Backward elimination on the leave-one-out comparisons.

        Each round drops the least useful removable term; `self.mean_terms`
        is updated after every drop, so a failure leaves the terms kept so far.
        """
        current = list(terms)
        while current:
            specs = [_join(current)] + [_join(current[:i] + current[i + 1:])
                                        for i in range(len(current))]
            if not self._admit("mean effects", "compare_specifications", specs, self.frame,
                               self.response, family=self.family, folds=self.folds,
                               mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step,
                               seed=self.seed, na_action=self.na_action):
                break
            rows = compare_specifications(
                specs, self.frame, self.response, family=self.family, folds=self.folds,
                mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step,
                seed=self.seed, na_action=self.na_action,
                extrapolation=self.fold_extrapolation)
            drops = rows[1:]
            for term, row in zip(current, drops, strict=True):
                self.evidence[term] = (row["change"], row["change se"])
            protected = {term for term in current
                         if set(term_variables(term)) & self.mandatory_mu}
            removable = [i for i, row in enumerate(drops)
                         if row["change"] <= SE_RULE * row["change se"]
                         and current[i] not in protected]
            if not removable:
                for term, row in zip(current, drops, strict=True):
                    reason = ("kept: mandatory (mandatory_terms)" if term in protected else
                              "kept: removing it raises out-of-fold risk by more than "
                              f"{SE_RULE:g} change se")
                    self._note("mean effects", term, reason,
                               score=rows[0]["out-of-fold risk"],
                               change=row["change"], se=row["change se"], adopted=True)
                break
            pick = min(removable, key=lambda i: drops[i]["change"])
            term = current[pick]
            self._note("mean effects", term, "dropped: removing it does not raise out-of-fold "
                       f"risk by more than {SE_RULE:g} change se (the least useful term this round)",
                       score=drops[pick]["out-of-fold risk"], change=drops[pick]["change"],
                       se=drops[pick]["change se"], adopted=False)
            current.pop(pick)
            self.mean_terms = list(current)
        if terms and not current:
            self._note("mean effects", "1", "no mean term survived: the mean model is y ~ 1")

    @property
    def mean_step(self):
        """Resolved ordinary-mean optimizer; partial internal fixtures retain fixed steps."""
        return getattr(self, "_resolved_mean_step", "fixed")

    @property
    def screen_mstop(self) -> int:
        """Steps of each ordinary-mean screening and comparison fit."""
        return SCREEN_MSTOP * getattr(self, "_budget_factor", 1)

    @property
    def tune_start(self) -> int:
        """First grid of the mean model's stopping-iteration search."""
        return TUNE_START * getattr(self, "_budget_factor", 1)

    @property
    def tune_cap(self) -> int:
        """Largest grid of that search: ``max_mstop`` when given."""
        return self.max_mstop or DEFAULT_MAX_MSTOP * getattr(self, "_budget_factor", 1)

    def _fit_mean(self):
        """The screening fit of the current mean model, on the complete rows."""
        control = boost_control(self.screen_mstop, _screening_nu(self.family), step=self.mean_step)
        formula = f"{quote_response(self.response)} ~ {_join(self.mean_terms)}"
        if not self._admit("mean screening", "gamboost", formula, self.complete,
                           family=self.family, control=control, na_action=self.na_action):
            return None
        return gamboost(formula, self.complete, family=self.family, control=control,
                        na_action=self.na_action)

    def _interactions(self):
        # With no mean term left, the mean model is y ~ 1 and the residual is
        # the centred response, so a pair that carries all of the signal (a
        # pure interaction) is still screened and tested.
        self._note("interactions", None, FACTOR_PAIR_NOTE)
        fit = self._fit_mean()
        if fit is None:
            return
        residual = pearson_residual(fit, self.complete, self.response)
        screened: list[ScreenedPair] = _screen_pairs(
            self.complete, residual, response=self.response, seed=self.seed,
            na_action=self.na_action)
        positive = [p for p in screened if p.removed > 0]
        candidates = self._admissible_pairs(positive)
        if not candidates:
            self._note("interactions", None,
                       "no pair explains residual variation out of sample" if not positive
                       else "no pair that explains residual variation is admissible under "
                       "the constraints")
            return
        for pair in candidates:
            label = f"{pair.left} x {pair.right}"
            try:
                # The pair's smooths use the scale of each main term (log for a
                # column drafted on log(x)), so the tensor's knots fall where the
                # main term's do.
                term = pair_term_for(pair.left, pair.right, data=self.complete,
                                     transforms=self.transforms)
            except _EXCEPTIONS as exc:
                self._fail("interactions", label, exc)
                continue
            if term in self.mean_terms:
                continue
            reference = self._pair_reference(pair)
            specs = [_join(reference), _join(reference + [term])]
            incumbent = list(self.mean_terms)
            check_incumbent = reference != incumbent
            if check_incumbent:
                specs.append(_join(incumbent))
            if not self._admit("interactions", "compare_specifications", specs, self.frame,
                               self.response, family=self.family, folds=self.folds,
                               mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step,
                               seed=self.seed, na_action=self.na_action):
                return
            rows = compare_specifications(
                specs, self.frame, self.response, family=self.family, folds=self.folds,
                mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step,
                seed=self.seed, na_action=self.na_action,
                extrapolation=self.fold_extrapolation, include_evaluations=check_incumbent)
            row = rows[1]
            # A pair the boosting never selects leaves the risk unchanged: change and
            # its se are both exactly 0, and 0 <= -bar * 0 holds. Such a pair has no
            # evidence, so the change must also be a decrease.
            adopted = (row["change"] < 0
                       and row["change"] <= -INTERACTION_SE_RULE * row["change se"])
            incumbent_check = None
            if adopted and check_incumbent:
                from mboostrs import _rust

                if any(r.get("error") is not None or r.get("evaluation") is None for r in (row, rows[2])):
                    adopted = False
                    incumbent_check = dict(strictly_improves_incumbent=False,
                                           reason="actual-incumbent comparison unavailable")
                else:
                    candidate_scores = np.asarray(row["evaluation"].fold_scores, dtype=np.float64)
                    incumbent_scores = np.asarray(rows[2]["evaluation"].fold_scores, dtype=np.float64)
                    prefer_incumbent, difference, difference_se = _rust.prefer_simpler_cv(
                        incumbent_scores, candidate_scores, 0.0)
                    adopted = not prefer_incumbent
                    incumbent_check = dict(
                        incumbent_scores=incumbent_scores.tolist(), candidate_scores=candidate_scores.tolist(),
                        incumbent_minus_candidate=difference, paired_se=difference_se,
                        strictly_improves_incumbent=adopted)
            reason = (f"adopted: lowers out-of-fold risk by more than "
                      f"{INTERACTION_SE_RULE:.2f} change se"
                      if adopted else f"rejected: does not lower out-of-fold risk by "
                      f"{INTERACTION_SE_RULE:.2f} change se")
            if incumbent_check is not None and not adopted:
                reason = "rejected: matched-margin gain does not improve the actual incumbent"
            self._note("interactions", term, f"{reason} (screen score {pair.removed:.4g})",
                       score=row["out-of-fold risk"], change=row["change"],
                       se=row["change se"], adopted=adopted)
            self.log[-1]["reference_terms"] = list(reference)
            self.log[-1].update(incumbent_terms=incumbent, incumbent_check=incumbent_check)
            if adopted:
                self.mean_terms = reference + [term]
                self.interaction_records.append({
                    "term": term, "variables": [pair.left, pair.right],
                    "change": _num(row["change"]), "change se": _num(row["change se"])})
            # A smooth tensor cannot represent a sharp pair boundary with a
            # short, fixed spline basis. Test the existing joint tree against
            # the actual incumbent, retaining it only for a paired CV gain.
            if all(self.kind_of.get(c) == "continuous" for c in (pair.left, pair.right)):
                self._pair_tree(pair, term if adopted else None)

    def _recover_mean_effects(self):
        """Reconsider dropped main terms conditional on the found interactions.

        A dominant interaction can mask a weaker main effect during the first
        elimination. Compare each absent draft term with the actual incumbent
        after the interaction stage, in draft order, at the same screening
        budget. The existing main-effect margin and fold design still apply.
        """
        for term in self.draft_terms:
            if term in self.mean_terms:
                continue
            # Pair references may have retained the same main function at a
            # different margin df. Do not offer a duplicate of that column.
            variables = term_variables(term)
            if any(term_variables(t) == variables for t in self.mean_terms):
                continue
            specs = [_join(self.mean_terms), _join(self.mean_terms + [term])]
            if not self._admit("mean recovery", "compare_specifications", specs, self.frame,
                               self.response, family=self.family, folds=self.folds,
                               mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step,
                               seed=self.seed, na_action=self.na_action):
                return
            row = compare_specifications(
                specs, self.frame, self.response, family=self.family, folds=self.folds,
                mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step, seed=self.seed,
                na_action=self.na_action, extrapolation=self.fold_extrapolation)[1]
            adopted = row["change"] < 0 and row["change"] <= -SE_RULE * row["change se"]
            reason = ("recovered" if adopted else "rejected") + (
                f": adding this main term after interactions must lower out-of-fold "
                f"risk by more than {SE_RULE:g} change se")
            self._note("mean recovery", term, reason, score=row["out-of-fold risk"],
                       change=row["change"], se=row["change se"], adopted=adopted)
            self.log[-1]["reference_terms"] = list(self.mean_terms)
            if adopted:
                self.mean_terms.append(term)
                # Effect records describe the loss from removal, the negative
                # of the measured addition gain; the log keeps the addition.
                self.evidence[term] = (-row["change"], row["change se"])

    def _pair_tree(self, pair, tensor):
        """Compare a continuous pair's tree form with the current mean model.

        A tree replaces this pair's just-adopted tensor, if present. All main
        effects and other interactions stay. Rejection, budget refusal or a
        failed comparison leaves the incumbent intact. Numeric fitting and
        paired risk differences use the existing core-backed comparison.
        """
        tree = f"btree({quote_name(pair.left)}, {quote_name(pair.right)}, maxdepth=3, selection=sse)"
        reference = list(self.mean_terms)
        alternative = [term for term in reference if term != tensor] + [tree]
        specs = [_join(reference), _join(alternative)]
        if not self._admit("interaction form", "compare_specifications", specs, self.frame,
                           self.response, family=self.family, folds=self.folds,
                           mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step,
                               seed=self.seed, na_action=self.na_action):
            return
        row = compare_specifications(
            specs, self.frame, self.response, family=self.family, folds=self.folds,
            mstop=self.screen_mstop, nu=_screening_nu(self.family), step=self.mean_step, seed=self.seed,
            na_action=self.na_action, extrapolation=self.fold_extrapolation)[1]
        adopted = (row["change"] < 0
                   and row["change"] <= -INTERACTION_SE_RULE * row["change se"])
        reason = ("adopted" if adopted else "rejected") + (
            f": tree form must lower the incumbent's out-of-fold risk by more than "
            f"{INTERACTION_SE_RULE:.2f} change se; comparison at mstop={self.screen_mstop}")
        self._note("interaction form", tree, reason, score=row["out-of-fold risk"],
                   change=row["change"], se=row["change se"], adopted=adopted)
        self.log[-1].update(reference_terms=reference, replaces=tensor if adopted else None)
        if adopted:
            self.mean_terms = alternative
            self.interaction_records = [r for r in self.interaction_records if r["term"] != tensor]
            self.interaction_records.append({"term": tree, "variables": [pair.left, pair.right],
                                            "change": _num(row["change"]), "change se": _num(row["change se"])})

    def _pair_reference(self, pair) -> list[str]:
        """The mean model an interaction is tested against.

        The tensor's margins are ``bbs(x, df=2.5, knots=5)`` (`pair_term_for`).
        For a pair of continuous columns, each main effect of the pair is put in
        the reference at that same smooth, replacing its draft or screened
        smooth, so the test compares the additive model at the tensor's margin
        flexibility with the same model plus the tensor. Without this, the
        draft's ``df=4`` main effects differ from the tensor's margins, and the
        paired fold differences measure that difference as well as the
        interaction. A binary varying smooth can also need its continuous
        partner's baseline when marginal elimination removed that partner.
        Together the two terms permit independent curves at the two levels,
        while retaining the original scalar or factor coding.
        """
        columns = {pair.left, pair.right}
        for factor, partner in ((pair.left, pair.right), (pair.right, pair.left)):
            if (self.kind_of.get(factor) in ("binary", "categorical")
                    and self.complete[factor].n_unique() == 2
                    and self.kind_of.get(partner) == "continuous"):
                reference = list(self.mean_terms)
                if not any(term_variables(term) == (partner,) for term in reference):
                    reference.append(f"bbs({self._margin_scale(partner)}, df={DRAFT_DF:g})")
                return reference
        if not all(self.kind_of.get(c) == "continuous" for c in columns):
            return list(self.mean_terms)
        kept = [t for t in self.mean_terms
                if not (len(term_variables(t)) == 1 and term_variables(t)[0] in columns)]
        for column in (pair.left, pair.right):
            kept.append(f"bbs({self._margin_scale(column)}, df={TENSOR_MARGINAL_DF:g}, knots=5)")
        return kept

    def _margin_scale(self, column: str) -> str:
        kind = self.transforms.get(column)
        return f"{kind}({column})" if kind else column

    def _admissible_pairs(self, positive):
        """The screened pairs the constraints allow, best first, at most five.

        A forbidden pair is dropped without a log entry of its own; the rule
        in force already names it. A pair the heredity rule bars is logged,
        because the rule's effect on a result is what a reader needs to see.
        """
        mains = main_columns(self.mean_terms)
        kept, forbidden = [], 0
        for pair in positive:
            columns = frozenset((pair.left, pair.right))
            if self.mu_columns is not None and not columns <= self.mu_columns:
                continue
            if columns in self.forbidden:
                forbidden += 1
                continue
            if self._is_brandom_pair(pair):
                continue
            if not admits(columns, mains, self.heredity):
                needed = ", ".join(sorted(columns - mains))
                self._note("interactions", f"{pair.left} x {pair.right}",
                           f"not nominated: heredity {self.heredity} needs a main effect "
                           f"of {needed} in the mean model")
                continue
            kept.append(pair)
        if forbidden:
            self._note("interactions", None, f"{forbidden} forbidden pair(s) not nominated")
        return kept[:MAX_INTERACTIONS]

    def _is_brandom_pair(self, pair) -> bool:
        """Whether `pair_term_for` builds a tensor of two brandom terms for this pair.

        Such a pair is not nominated: a brandom term's folds hold out rows of a
        group while its other rows train the fold, so the score rewards memorized
        group effects.
        """
        try:
            term = pair_term_for(pair.left, pair.right, data=self.complete,
                                 transforms=self.transforms)
        except _EXCEPTIONS:
            return False
        parts = term.split(" %X% ")
        return len(parts) == 2 and all(part.startswith("brandom(") for part in parts)

    def _spread(self):
        if self.mandatory_sigma:
            column = self.mandatory_sigma[0]
            term = self._spread_term(column)
            self.spread_term = term
            self.spread_record = {"term": term, "variables": [column], "parameter": "sigma",
                                  "change": None, "change se": None, "mandatory": True}
            self._note("spread", f"sigma: {term}", "adopted: mandatory (mandatory_terms "
                       "names it); included without the spread test", adopted=True)
            return
        fit = self._fit_mean()
        if fit is None:
            return
        # Every predictor of the data is a candidate, not only the mean model's:
        # a column that moves the spread and not the mean is found here.
        het = heteroscedasticity(fit, self.complete, transforms=self.transforms)
        by_column = het.get("by_column", {})
        tested = [c for c in self.predictors if c in by_column]
        untested = self._spread_untested(het)
        nominated = [c for c, stats in by_column.items()
                     if c in self.predictors
                     and ((_finite(stats.get("t")) and abs(stats["t"]) >= SPREAD_NOMINATION_T)
                          or (_finite(stats.get("curve_removed")) and stats["curve_removed"] > 0))
                     and (self.sigma_columns is None or c in self.sigma_columns)]
        if not nominated:
            self._note("spread", None, "no predictor's residual spread has |t| >= "
                       f"{SPREAD_NOMINATION_T:g} or positive held-out curve gain; constant scale kept. "
                       "Tested the spread against "
                       f"{len(tested)} column(s): {', '.join(tested) or 'none'}. {untested}")
            return
        mu = _join(self.mean_terms)
        sigma_terms = {c: self._spread_term(c) for c in nominated}
        specs = [{"mu": mu, "sigma": "1"}] + [{"mu": mu, "sigma": s}
                                              for s in sigma_terms.values()]
        if not self._admit("spread", "compare_specifications", specs, self.frame, self.response,
                           family=GaussianLSS(), folds=self.folds, mstop=SCREEN_MSTOP,
                           seed=self.seed, na_action=self.na_action):
            return
        self._note("spread", None, f"tested the spread against {len(tested)} column(s) "
                   f"({', '.join(tested)}); {len(nominated)} nominated at |t| >= "
                   f"{SPREAD_NOMINATION_T:g} or positive held-out curve gain: {', '.join(nominated)}. "
                   f"{untested}")
        self.log[-1]["nomination_scores"] = by_column
        rows = compare_specifications(
            specs, self.frame, self.response, family=GaussianLSS(), folds=self.folds,
            mstop=SCREEN_MSTOP, nu=NU, seed=self.seed, na_action=self.na_action,
            extrapolation=self.fold_extrapolation)
        best_col, best_row = min(zip(nominated, rows[1:], strict=True), key=lambda p: p[1]["change"])
        passes = best_row["change"] <= -SE_RULE * best_row["change se"]
        confirm_status, confirm_row = None, None
        if passes:
            confirm_status, confirm_row = self._confirm_spread(mu, sigma_terms[best_col])
        for col, row in zip(nominated, rows[1:], strict=True):
            if col != best_col:
                self._note("spread", f"sigma: {sigma_terms[col]}",
                           "rejected: a weaker log-scale term than the best nominated one",
                           score=row["out-of-fold risk"], change=row["change"],
                           se=row["change se"], adopted=False)
                continue
            if not passes:
                self._note("spread", f"sigma: {sigma_terms[col]}",
                           f"rejected: the best log-scale term does not beat the constant scale "
                           f"by {SE_RULE:g} change se",
                           score=row["out-of-fold risk"], change=row["change"],
                           se=row["change se"], adopted=False)
            elif confirm_status == "confirmed":
                self._note("spread", f"sigma: {sigma_terms[col]}",
                           f"adopted: beats the constant scale by more than {SE_RULE:g} "
                           "change se at the screening comparison and again in the "
                           "mean-adequacy confirmation, which tunes each specification's "
                           "stopping inside the folds; the best of the nominated "
                           "log-scale terms",
                           score=confirm_row["out-of-fold risk"], change=confirm_row["change"],
                           se=confirm_row["change se"], adopted=True)
            elif confirm_status == "rejected":
                self._note("spread", f"sigma: {sigma_terms[col]}",
                           f"rejected: beats the constant scale by more than {SE_RULE:g} "
                           f"change se at the screening comparison (change "
                           f"{best_row['change']:.4g}), but not in the mean-adequacy "
                           "confirmation, which tunes each specification's stopping "
                           "inside the folds: against a mean fitted to its tuned "
                           "stopping the win does not survive, so the screening win "
                           "reads as mean approximation error, not varying scale; "
                           "constant scale kept",
                           score=confirm_row["out-of-fold risk"], change=confirm_row["change"],
                           se=confirm_row["change se"], adopted=False)
            else:
                self._note("spread", f"sigma: {sigma_terms[col]}",
                           f"rejected: beats the constant scale by more than {SE_RULE:g} "
                           f"change se at the screening comparison (change "
                           f"{best_row['change']:.4g}), but the mean-adequacy "
                           "confirmation could not be run or scored within the "
                           "budget, so the win is unconfirmed; constant scale kept",
                           score=row["out-of-fold risk"], change=row["change"],
                           se=row["change se"], adopted=False)
        if passes and confirm_status == "confirmed":
            col = best_col
            self.spread_term = sigma_terms[col]
            self.spread_record = {"term": self.spread_term, "variables": [col],
                                  "parameter": "sigma", "change": _num(confirm_row["change"]),
                                  "change se": _num(confirm_row["change se"])}

    def _confirm_spread(self, mu: str, term: str):
        """Whether a screening spread win survives an adequately fitted mean.

        The screening comparison fits both specifications at SCREEN_MSTOP,
        while the adopted model is fitted at its tuned stopping; a
        log-scale term can win the screening comparison by absorbing
        structure the short mean fit leaves in the residuals, which reads
        as varying scale although the generating scale is constant. The
        confirmation scores the same two specifications
        with each candidate's stopping tuned inside every outer fold
        (``mstop="cv"``), capped where the final distributional tuning
        caps it, so the mean in both specifications is fitted as fully as
        the final model fits it. Returns ("confirmed", row) when the term
        again beats the constant scale by SE_RULE change se,
        ("rejected", row) when it does not, and ("unavailable", None)
        when the budget refuses the comparison or it cannot be scored.
        """
        cap = self.max_mstop or LSS_GRID_MSTOP * 2 ** MAX_GRID_EXTENSIONS
        specs = [{"mu": mu, "sigma": "1"}, {"mu": mu, "sigma": term}]
        if not self._admit("spread", "compare_specifications", specs, self.frame,
                           self.response, family=GaussianLSS(), folds=self.folds,
                           mstop="cv", max_mstop=cap, seed=self.seed,
                           na_action=self.na_action):
            return "unavailable", None
        try:
            rows = compare_specifications(
                specs, self.frame, self.response, family=GaussianLSS(), folds=self.folds,
                mstop="cv", max_mstop=cap, nu=NU, seed=self.seed,
                na_action=self.na_action, extrapolation=self.fold_extrapolation)
        except _EXCEPTIONS:
            return "unavailable", None
        row = rows[1]
        if not (_finite(row.get("change")) and _finite(row.get("change se"))):
            return "unavailable", None
        if row["change"] <= -SE_RULE * row["change se"]:
            return "confirmed", row
        return "rejected", row

    def _tune_and_fit(self):
        """Stopping iteration of the adopted model, then the final fit."""
        if self.spread_term is None:
            rhs = _join(self.mean_terms)
            if not self.mean_terms:
                self._note("tune", None, "not tuned: the model has no terms, so no stopping "
                           f"iteration is chosen; mstop={DEFAULT_MSTOP}")
                self.mstop = DEFAULT_MSTOP
            else:
                self._tune_mean(rhs)
            self.model = self._fit_final(self.mstop)
            return
        formulas = self._lss_formulas()
        model = self._tune_lss(formulas)
        if model is None:
            self.mstop = DEFAULT_MSTOP
            model = gamboostLSS(formulas, self.frame, family=GaussianLSS(),
                                mstop=self.mstop, na_action=self.na_action)
        self.model = model

    def _tune_mean(self, rhs: str) -> None:
        """Set ``self.mstop`` from a cross-validated grid, doubling the grid while the risk falls.

        The grid starts at TUNE_START steps (times the ``mean_budget`` factor,
        as is the cap). A minimum in the boundary band of
        the grid doubles it, at most MAX_GRID_EXTENSIONS times, and each doubling
        is admitted by the budget. If the minimum is still at the end, the stage
        is recorded as a boundary, the full adopted term set stays, and the
        stopping iteration is the minimum found. No term is dropped for it.
        """
        formula = f"{quote_response(self.response)} ~ {rhs}"
        cap = self.tune_cap
        grid, extensions = min(self.tune_start, cap), 0
        if not self._admit_tune(formula, grid):
            # The budget refused the first grid; the log already says so.
            self.mstop = DEFAULT_MSTOP
            return
        while True:
            with warnings.catch_warnings():
                # The stage writes its own boundary entry and warning, so the
                # library's per-grid "still falling" warning is not repeated.
                warnings.filterwarnings("ignore", message="the cross-validated risk was still falling")
                tuning = tune_mstop(formula, self.frame, family=self.family, folds=self.folds,
                                    seed=self.seed, na_action=self.na_action, start=grid,
                                    nu=_screening_nu(self.family), step=self.mean_step,
                                    extensions=0, max_mstop=cap,
                                    extrapolation=self.fold_extrapolation)
            if not tuning.at_boundary or grid >= cap:
                break
            next_grid = min(2 * grid, cap)
            if not self._admit_tune(formula, next_grid):
                break
            grid, extensions = next_grid, extensions + 1
        self.mstop = max(1, int(tuning.mstop))
        if tuning.at_boundary:
            reason = (f"boundary: the cross-validated minimum lies in the last 5% of the "
                      f"mstop grid ({grid} steps after {extensions} doubling(s)); the full adopted "
                      f"term set is kept, at mstop={self.mstop}, and the stopping iteration is "
                      "not resolved")
            self.extra_warnings.append(self._boundary_message("mean model", grid, cap))
        else:
            reason = (f"cross-validated stopping iteration over {self.folds} folds on a grid of "
                      f"{grid} steps" + (f" after {extensions} doubling(s)" if extensions else ""))
        self._note("tune", rhs, reason, score=None)

    def _admit_tune(self, formula: str, grid: int) -> bool:
        """Whether a mean-model tuning on a grid of ``grid`` steps fits the budget."""
        return self._admit("tune", "tune_mstop", formula, self.frame, family=self.family,
                           folds=self.folds, seed=self.seed, na_action=self.na_action,
                           nu=_screening_nu(self.family), step=self.mean_step,
                           start=grid, extensions=0,
                           max_mstop=self.tune_cap)

    def _boundary_message(self, model_name: str, grid: int, cap: int) -> str:
        """The boundary note, naming the argument that extends the search."""
        if grid >= cap:
            remedy = (f"to search further, pass discover(max_mstop=...) above {cap}; "
                      "a larger value lets the grid double past it")
        else:
            remedy = (f"the budget stopped the search before the grid reached max_mstop={cap}; "
                      "raise budget_seconds to search further")
        return (f"boundary: the {model_name}'s stopping iteration is at the end of the mstop "
                f"grid ({grid} steps); the full adopted term set is kept at mstop={self.mstop}; "
                f"{remedy}")

    def _fit_final(self, mstop):
        rhs = _join(self.mean_terms)
        control = boost_control(mstop, _screening_nu(self.family), step=self.mean_step)
        self._required("final model", "gamboost", f"{quote_response(self.response)} ~ {rhs}", self.frame,
                       family=self.family, control=control, na_action=self.na_action)
        return gamboost(f"{quote_response(self.response)} ~ {rhs}", self.frame, family=self.family,
                        control=control, na_action=self.na_action)

    def _lss_formulas(self):
        # gamboostLSS takes the intercept-only mean as "1" (it fits the intercept
        # as the parameter's learner); an empty right-hand side is refused.
        return {"mu": f"{quote_response(self.response)} ~ {_join(self.mean_terms) or '1'}",
                "sigma": f"{quote_response(self.response)} ~ {self.spread_term}"}

    def _spread_untested(self, het: dict) -> str:
        """The sentence that names the frame columns the spread test did not cover.

        A predictor is not tested when :func:`heteroscedasticity` lists it in
        ``untested_columns`` with a reason, and a response copy is dropped before
        the draft and so is never a candidate. The sentence names each one with
        its reason, so the report states which columns could not drive a spread
        term.
        """
        by_column = het.get("by_column", {})
        reasons = het.get("untested_columns", {})
        parts = [f"{c} ({reasons.get(c, 'not tested')})"
                 for c in self.predictors if c not in by_column]
        parts += [f"{c} (response copy, dropped before the draft)"
                  for c, _ in self.dropped_predictors]
        if not parts:
            return "Every predictor in the data was tested as a spread candidate."
        return ("Not tested as spread candidates: " + ", ".join(parts)
                + ". A spread driver among them cannot be nominated.")

    def _spread_term(self, column: str) -> str:
        return _spread_term(column, self.complete, self.transforms.get(column))

    def _lss_cv_seconds(self, formulas, grid: int) -> float:
        """Estimated seconds of the cross-validation at one grid."""
        try:
            one_fit = estimate_cost("gamboostLSS", formulas, self.frame, family=GaussianLSS(),
                                    mstop=grid, na_action=self.na_action).estimated_seconds
        except _EXCEPTIONS:
            return 0.0
        return self.folds * one_fit

    def _tune_lss(self, formulas) -> LSSModel | None:
        """The stopping iteration of both parameters, the grid doubled while either falls.

        Returns the model fitted at the larger of the two stopping iterations, or
        None when the budget refuses the first grid. Doubling follows the rule of
        :meth:`_tune_mean`.
        """
        # The distributional grid starts at LSS_GRID_MSTOP and, by default, doubles
        # to LSS_GRID_MSTOP * 2 ** MAX_GRID_EXTENSIONS; max_mstop sets the ceiling.
        cap = self.max_mstop or LSS_GRID_MSTOP * 2 ** MAX_GRID_EXTENSIONS
        grid, extensions = min(LSS_GRID_MSTOP, cap), 0
        seconds = self._lss_cv_seconds(formulas, grid)
        self._log_estimate("tune", seconds,
                           "the cross-validation of the distributional model at its first grid")
        if not self._admit_seconds("tune", seconds):
            return None
        while True:
            full = gamboostLSS(formulas, self.frame, family=GaussianLSS(), mstop=grid,
                               na_action=self.na_action)
            n_model = len(full.predict(parameter="mu", type="response"))
            folds = cv_weights(n_model, type="kfold", B=self.folds, seed=self.seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="the lowest cross-validated risk is at")
                cv = full.cvrisk(folds=folds)
            optimal = [int(v) for v in np.atleast_1d(cv.optimal_mstop)]
            at_boundary = any(is_at_boundary(v, grid) for v in optimal)
            if not at_boundary or grid >= cap:
                break
            next_grid = min(2 * grid, cap)
            if not self._admit_seconds("tune", self._lss_cv_seconds(formulas, next_grid)):
                break
            grid, extensions = next_grid, extensions + 1
        self.mstop = max(1, max(optimal))
        names = dict(zip(cv.param_names, optimal, strict=True))
        if at_boundary:
            reason = (f"boundary: the cross-validated minimum lies in the last 5% of the mstop "
                      f"grid ({grid} steps after {extensions} doubling(s)) for "
                      f"{names}; the full adopted term set is kept, at mstop={self.mstop} for "
                      "both parameters, and the stopping iteration is not resolved")
            self.extra_warnings.append(
                self._boundary_message("distributional model", grid, cap))
        else:
            reason = (f"cross-validated stopping iteration for each distribution parameter {names}; "
                      f"the larger, {self.mstop}, is used for both")
        self._note("tune", "mu and sigma", reason, score=None)
        self._required("final model", "gamboostLSS", formulas, self.frame, family=GaussianLSS(),
                       mstop=self.mstop, na_action=self.na_action)
        return gamboostLSS(formulas, self.frame, family=GaussianLSS(), mstop=self.mstop,
                           na_action=self.na_action)

    @staticmethod
    def _held_note(refit):
        return (f"; conditional on the held boosted contributions {list(refit.held)}; "
                "their uncertainty is not estimated" if refit.held else "")

    def _ordinary_refit(self, model, *, terms=None, stage="refit"):
        try:
            return gam_refit(model, terms=terms, unsupported="hold")
        except ValueError as exc:
            if not str(exc).startswith("no term the refit can represent; all of"):
                raise
            self._note(stage, None, "not run: every selected term is held; "
                       "no supported effect remains for a REML refit. "
                       "Boosted predictions remain available without refit intervals.")
            return None

    def _refit(self):
        if isinstance(self.model, LSSModel):
            if not self._admit("refit", "gamboostLSS", self._lss_formulas(), self.frame,
                               family=GaussianLSS(), mstop=self.mstop,
                               na_action=self.na_action):
                return
            try:
                self.refit = gamlss_refit(self.model)
            except NotImplementedError as exc:
                if not str(exc).startswith("gamlss_refit supports ") or "cannot refit" not in str(exc):
                    raise
                self._note("refit", "mu and sigma", "not run: a joint distributional REML "
                           f"refit is unavailable for these terms ({exc}). "
                           "Boosted parameter predictions remain available without refit intervals.")
                return
            self._note("refit", "mu and sigma", "REML refit of the adopted terms", adopted=True)
            return
        if not self.mean_terms:
            self._note("refit", None, "not run: intercept-only model has no terms to refit")
            return
        if not self._admit("refit", "gamboost", f"{quote_response(self.response)} ~ {_join(self.mean_terms)}",
                           self.frame, family=self.family, na_action=self.na_action):
            return
        self.refit = self._ordinary_refit(self.model)
        if self.refit is None:
            return
        self._note("refit", _join(self.mean_terms),
                   "REML refit of the adopted terms" + self._held_note(self.refit), adopted=True)

    def predict(self, new_frame, calibration=None):
        if new_frame is None:
            return
        model = self.model
        is_lss = isinstance(model, LSSModel)
        parameters = None
        try:
            self.extrapolation = model.extrapolation_report(new_frame)
            if is_lss:
                # Every distribution parameter at once, on the response scale:
                # mu becomes "mean", and the rest (sigma here) becomes its own
                # column, so a caller scores the fit without rebuilding the
                # scale from the interval band.
                parameters = model.predict_parameters(new_frame, type="response",
                                                      extrapolation="clip")
                mean = parameters["mu"]
            else:
                mean = model.predict(new_frame, type="response", extrapolation="clip")
        except _EXCEPTIONS as exc:
            self._fail("predict", None, exc)
            return
        mean = np.asarray(mean, dtype=float).ravel()
        n = new_frame.height
        lower = upper = np.full(n, np.nan)
        method = "distributional" if is_lss else "split"
        if self._admit("predict", "conformal_interval", model, new_frame, level=self.level,
                       method=method, seed=self.seed, calibration=calibration):
            try:
                interval = conformal_interval(model, new_frame, level=self.level,
                                              method=method, seed=self.seed,
                                              calibration=calibration, extrapolation="clip")
                self.interval = interval
                lower = np.asarray(interval.lower, dtype=float).ravel()
                upper = np.asarray(interval.upper, dtype=float).ravel()
                self._note("predict", None, f"intervals: {interval.method}, guarantee "
                           f"'{interval.guarantee}'", adopted=True)
            except _EXCEPTIONS as exc:
                self._fail("predict", "intervals", exc)
        columns = {"row": np.arange(n), "mean": mean}
        if parameters is not None:
            for name, values in parameters.items():
                if name != "mu":
                    columns[name] = np.asarray(values, dtype=float).ravel()
        columns["lower"], columns["upper"] = lower, upper
        self.predictions = pl.DataFrame(columns)

    def result(self, caught_warnings: list[str]) -> DiscoveryResult:
        spread_formula = self.spread_term
        mean_effects = self._mean_records()
        spread_drivers = self._spread_records()
        variables = {v for rec in mean_effects + self.interaction_records + spread_drivers
                     for v in rec["variables"]}
        prov = provenance(self.frame, seed=self.seed)
        prov["partition"] = self.partition
        prov["simplify_mean"] = self.do_simplify_mean
        prov["mean_step"] = self.mean_step
        prov["mean_step_requested"] = self.requested_mean_step
        prov["mean_budget"] = getattr(self, "mean_budget", "fixed")
        return DiscoveryResult(
            model=self.model, refit=self.refit, formula=_join(self.mean_terms),
            spread_formula=spread_formula, family=self.family_name, mstop=self.mstop,
            folds=self.folds, seed=self.seed, level=self.level,
            mean_effects=mean_effects, interactions=self.interaction_records,
            spread_drivers=spread_drivers, predictions=self.predictions,
            extrapolation=self.extrapolation, warnings=caught_warnings, log=self.log,
            provenance=prov, _frame=self.complete, _variables=frozenset(variables),
            constraints=_rule_record(self.rules), _dropped_refit=self.dropped_refit,
            _dropped_variables=frozenset(self.dropped_variables), interval=self.interval,
            stage_seconds={stage: round(seconds, 6)
                           for stage, seconds in self.stage_seconds.items()})

    # -- labels ------------------------------------------------------------

    def _mean_records(self) -> list[dict]:
        interaction_terms = {r["term"] for r in self.interaction_records}
        records = []
        for term in self.mean_terms:
            if term in interaction_terms:
                continue
            variables = _term_variables(term, self.predictors)
            change, se = self.evidence.get(term, (None, None))
            record = {"term": term, "variables": variables, "shape": None, "range": None,
                      "slope": None, "direction": None, "transform": self._transform_of(variables),
                      "change": _num(change), "change se": _num(se)}
            if len(variables) == 1:
                record.update(self._describe(variables[0], parameter=self._default_parameter()))
            records.append(record)
        return records

    def _spread_records(self) -> list[dict]:
        if self.spread_record is None:
            return []
        record = dict(self.spread_record)
        record.update(self._describe(record["variables"][0], parameter="sigma"))
        record["transform"] = self._transform_of(record["variables"])
        return [record]

    def _transform_of(self, variables) -> str | None:
        """The transform of a one-variable term's predictor, or None."""
        if len(variables) != 1:
            return None
        return self.transforms.get(variables[0])

    def _default_parameter(self) -> str | None:
        return "mu" if isinstance(self.model, LSSModel) else None

    def _describe(self, variable: str, *, parameter) -> dict:
        """Shape label, range, slope and direction of one variable's effect.

        The range, slope and turning flag come from :func:`sensitivity`, and
        the shape word is read from the same curve it evaluates.
        """
        empty = {"shape": None, "range": None, "slope": None, "direction": None}
        if self.kind_of.get(variable) in ("binary", "categorical"):
            return {**empty, "shape": "categorical"}
        log_grid = (variable,) if variable in self.transforms else ()
        ((sens, curve),) = _effects(self.model, self.complete, [variable], CURVE_POINTS,
                                    (0.1, 0.9), parameter, log_grid=log_grid)
        rng = _finite_or_none(sens.get("range of the effect"))
        slope = _finite_or_none(sens.get("slope, 10% to 90%"))
        turns = sens.get("turns")
        if turns is None or (isinstance(turns, float) and math.isnan(turns)):
            return {**empty, "shape": "categorical", "range": rng, "slope": slope}
        shape = _label_curve(curve)
        direction = "increasing" if curve[-1] >= curve[0] else "decreasing"
        return {"shape": shape, "range": rng, "slope": slope, "direction": direction}


#: Parameters a constraint may name. ``mu`` is the mean model and ``sigma``
#: the log-scale model that the spread stage of a Gaussian response may add.
PARAMETERS = ("mu", "sigma")


def _by_parameter(value, name, allowed, default):
    """``{parameter: value}`` for a bare value (it applies to ``mu``) or a mapping."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValueError(f"{name} names {unknown}; it accepts the keys {list(allowed)}, "
                             "the mean model 'mu' and the log scale 'sigma'")
        return {p: value.get(p, default) for p in allowed}
    return {p: (value if p == "mu" else default) for p in allowed}


def _parameter_rules(columns, response, *, heredity, forbidden_pairs, mandatory_terms,
                     candidates):
    """The validated constraints, keyed by parameter, as plain values."""
    columns = list(columns)
    heredity_by = _by_parameter(heredity, "heredity", ("mu",), "none")
    forbidden_by = _by_parameter(forbidden_pairs, "forbidden_pairs", ("mu",), ())
    mandatory_by = _by_parameter(mandatory_terms, "mandatory_terms", PARAMETERS, ())
    candidates_by = _by_parameter(candidates, "candidates", PARAMETERS, None)

    mandatory = {}
    for p in PARAMETERS:
        value = mandatory_by[p]
        if isinstance(value, str):
            value = (value,)
        mandatory[p] = check_mandatory(value, columns, response,
                                       name=f"mandatory_terms[{p!r}]")
    if len(mandatory["sigma"]) > 1:
        raise ValueError("mandatory_terms['sigma'] names "
                         f"{len(mandatory['sigma'])} columns; discover fits one log-scale "
                         "term, so name at most one")

    allowed = {}
    for p in PARAMETERS:
        value = candidates_by[p]
        if value is None:
            allowed[p] = None
            continue
        names = check_mandatory((value,) if isinstance(value, str) else value, columns,
                                response, name=f"candidates[{p!r}]")
        if not names:
            raise ValueError(f"candidates[{p!r}] is empty; name the columns the {p} search "
                             "may use, or omit the key to allow every predictor")
        allowed[p] = tuple(sorted(set(names) | set(mandatory[p])))

    forbidden = forbidden_by["mu"]
    if isinstance(forbidden, str):
        raise ValueError("forbidden_pairs takes pairs such as ('x1', 'x2'), not a string")
    return {
        "heredity": {"mu": check_heredity(heredity_by["mu"], "heredity")},
        "forbidden_pairs": {"mu": check_forbidden_pairs(
            list(forbidden or ()), columns, response, name="forbidden_pairs")},
        "mandatory_terms": mandatory,
        "candidates": allowed,
    }


def _rule_record(rules):
    """The rule in force as JSON-ready data, keyed by parameter."""
    return {
        "heredity": dict(rules["heredity"]),
        "forbidden_pairs": {"mu": [sorted(pair) for pair in rules["forbidden_pairs"]["mu"]]},
        "mandatory_terms": {p: list(v) for p, v in rules["mandatory_terms"].items()},
        "candidates": {p: None if v is None else list(v)
                       for p, v in rules["candidates"].items()},
    }


def _rule_sentence(rules):
    """The rule in force, stated in one line for the log."""
    parts = [f"mu heredity {rules['heredity']['mu']}"]
    if rules["forbidden_pairs"]["mu"]:
        parts.append("forbidden pairs (mu): " + ", ".join(
            pair_label(pair) for pair in rules["forbidden_pairs"]["mu"]))
    for p in PARAMETERS:
        if rules["mandatory_terms"][p]:
            parts.append(f"mandatory ({p}): " + ", ".join(rules["mandatory_terms"][p]))
    for p in PARAMETERS:
        if rules["candidates"][p] is not None:
            parts.append(f"candidates ({p}): " + ", ".join(rules["candidates"][p]))
    return "rule in force: " + "; ".join(parts)


#: A turn whose change is smaller than this share of the curve's range is
#: wiggle from the spline basis, not a turn.
SMALL_TURN = 0.05


def _label_curve(curve: np.ndarray) -> str:
    """The shape word of a fitted curve on an evenly spaced grid.

    The curve is split into runs of one sign of change. A run smaller than
    `SMALL_TURN` of the range is merged into its neighbours, so that a
    noisy spline on a linear effect reads as monotone. One remaining turn
    is a U (a minimum) or an inverted U (a maximum); two or more are
    nonmonotone. A curve with one jump and no other change is a threshold.
    """
    diffs = np.diff(curve)
    span = float(np.ptp(curve))
    tol = 1e-9 * max(1.0, span)
    nonzero = diffs[np.abs(diffs) > tol]
    if nonzero.size == 0:
        return "increasing" if curve[-1] >= curve[0] else "decreasing"
    runs: list[float] = []
    for d in nonzero:
        if runs and np.sign(runs[-1]) == np.sign(d):
            runs[-1] += d
        else:
            runs.append(float(d))
    if len(runs) == 1 and nonzero.size <= 2:
        return "threshold_increase" if runs[0] > 0 else "threshold_decrease"
    while len(runs) > 1:
        small = min(range(len(runs)), key=lambda i: abs(runs[i]))
        if abs(runs[small]) >= SMALL_TURN * span:
            break
        if 0 < small < len(runs) - 1:
            runs[small - 1:small + 2] = [runs[small - 1] + runs[small] + runs[small + 1]]
        else:
            del runs[small]
    turns = len(runs) - 1
    if turns == 0:
        return "increasing" if runs[0] > 0 else "decreasing"
    if turns == 1:
        return "u_shaped" if runs[0] < 0 else "inverted_u"
    return "nonmonotone"


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def _contrast_gam(refit: GamRefit, ref_frame, alt_frame, variable, reference, alternative,
                  level) -> DiscoveryContrast:
    eff_ref = refit.effects(ref_frame, level=level)
    eff_alt = refit.effects(alt_frame, level=level)
    estimate = 0.0
    half = 0.0
    touched = False
    for term in eff_ref:
        if term not in eff_alt:
            continue
        est_r, lo_r, hi_r = (np.asarray(v, dtype=float) for v in eff_ref[term])
        est_a, lo_a, hi_a = (np.asarray(v, dtype=float) for v in eff_alt[term])
        if np.array_equal(est_r, est_a):
            continue
        touched = True
        estimate += float(np.mean(est_a - est_r))
        half += float(np.mean(((hi_a - lo_a) + (hi_r - lo_r)) / 2.0))
    if not touched:
        return DiscoveryContrast(variable, reference, alternative, None, 0.0, 0.0, 0.0, level,
                                 "link", f"{variable!r} has no effect in the refit: it was not "
                                         "selected, so its contrast is exactly 0.")
    return DiscoveryContrast(
        variable, reference, alternative, None, estimate, estimate - half, estimate + half,
        level, "link",
        "average over the training rows of the change in the linear predictor; the interval "
        "is a conservative bound (the sum of pointwise half-widths of the affected effects)."
        + (" Conditional on the held boosted contributions; their uncertainty is not estimated."
           if refit.held else ""))


def _contrast_lss(refit: GamlssRefit, ref_frame, alt_frame, variable, reference, alternative,
                  parameter, level, seed) -> DiscoveryContrast:
    draws_ref = refit.draws(ref_frame, n_draws=CONTRAST_DRAWS, seed=seed)
    draws_alt = refit.draws(alt_frame, n_draws=CONTRAST_DRAWS, seed=seed)
    if parameter not in draws_ref:
        raise ValueError(f"parameter {parameter!r} is not a parameter of this model; "
                         f"pass one of {sorted(draws_ref)}")
    per_draw = (np.asarray(draws_alt[parameter]) - np.asarray(draws_ref[parameter])).mean(axis=1)
    point_alt = np.asarray(refit.predict(alt_frame, parameter=parameter, type="response"), dtype=float)
    point_ref = np.asarray(refit.predict(ref_frame, parameter=parameter, type="response"), dtype=float)
    estimate = float(np.mean(point_alt - point_ref))
    lower, upper = np.quantile(per_draw, [(1 - level) / 2, (1 + level) / 2])
    return DiscoveryContrast(
        variable, reference, alternative, parameter, estimate, float(lower), float(upper),
        level, "response",
        f"average over the training rows of the change in {parameter!r} on its response scale; "
        f"the interval is the {level:g} quantiles of {CONTRAST_DRAWS} posterior draws.")


def _set_value(frame: pl.DataFrame, variable: str, value) -> pl.DataFrame:
    if variable not in frame.columns:
        raise ValueError(f"variable {variable!r} is not a column of the data; "
                         f"pass one of {frame.columns}")
    try:
        literal = pl.lit(value).cast(frame.schema[variable])
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise ValueError(f"value {value!r} cannot be used for {variable!r} "
                         f"({frame.schema[variable]}): {exc}") from exc
    return frame.with_columns(literal.alias(variable))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _screen_pairs(complete, residual, *, response, seed, na_action) -> list[ScreenedPair]:
    """:func:`~mboostrs.terms.screen_interactions` averaged over SCREEN_SPLITS half-splits.

    One half-split of the rows gives each pair a noisy score, so the rank of a
    pair that carries real structure changes with the split seed. The score of
    each pair here is the mean of its scores over the splits with seeds
    ``seed`` to ``seed + SCREEN_SPLITS - 1``, so the top candidates are the
    pairs that remove residual variation on most splits.
    """
    totals: dict[frozenset, float] = {}
    order: dict[frozenset, tuple[str, str]] = {}
    for offset in range(SCREEN_SPLITS):
        for pair in screen_interactions(complete, residual, response=response,
                                        seed=seed + offset, na_action=na_action):
            key = frozenset((pair.left, pair.right))
            totals[key] = totals.get(key, 0.0) + pair.removed / SCREEN_SPLITS
            order.setdefault(key, (pair.left, pair.right))
    ranked = sorted(totals, key=lambda key: (-totals[key], order[key]))
    return [ScreenedPair(left=order[key][0], right=order[key][1], removed=totals[key])
            for key in ranked]


def _has_nulls(frame: pl.DataFrame) -> bool:
    return any(frame[c].null_count() > 0 for c in frame.columns)


def _as_frame(data, name: str = "data") -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        frame = data
    elif isinstance(data, dict):
        frame = pl.DataFrame(data)
    elif type(data).__module__.startswith("pandas"):
        frame = pl.from_pandas(data)
    else:
        raise TypeError(f"{name} must be a polars or pandas DataFrame or a mapping of column "
                        f"to values, not {type(data).__name__}")
    return _nan_as_null(frame)


def _nan_as_null(frame: pl.DataFrame) -> pl.DataFrame:
    """Float NaN becomes null, so every stage counts and drops one missing marker."""
    floats = [c for c, dtype in frame.schema.items() if dtype in (pl.Float32, pl.Float64)]
    if not floats:
        return frame
    return frame.with_columns([pl.col(c).fill_nan(None) for c in floats])


def _join(terms: list[str]) -> str:
    return " + ".join(terms) if terms else "1"


def _log_skew(column: pl.Series) -> float | None:
    """The sample skewness of a positive column when it draws the log smooth, else None.

    The column qualifies when its present values are all above 0 and their
    skewness (the mean of the cubed standardised values) exceeds LOG_SKEW.
    """
    present = column.drop_nulls().cast(pl.Float64).to_numpy()
    present = present[np.isfinite(present)]
    if present.size <= 3 or float(present.min()) <= 0:
        return None
    scale = float(np.std(present))
    if scale == 0:
        return None
    skew = float(np.mean(((present - present.mean()) / scale) ** 3))
    return skew if skew > LOG_SKEW else None


def _predictor_hygiene(frame: pl.DataFrame, response: str, predictors: list[str],
                       ) -> tuple[list[tuple[str, float]], list[tuple[str, str]]]:
    """Response copies and row identifiers among the candidate predictors.

    Returns ``(copies, identifiers)``. ``copies`` is one ``(column,
    correlation)`` pair per numeric predictor whose absolute Spearman (rank)
    correlation with a numeric response is at least ``RESPONSE_COPY_RANK``: rank
    correlation compares the ordering, so it is 1 when the column is any
    increasing or decreasing function of the response, a copy as well as a
    transform. ``identifiers`` is one ``(column, detail)`` pair per whole-number
    predictor whose distinct values cover at least ``UNIQUE_INTEGER_SHARE`` of
    its rows, which names rows rather than measuring a quantity. A response copy
    is not also reported as an identifier.
    """
    if not frame.schema[response].is_numeric():
        y = None
    else:
        y = frame[response]
    copies: list[tuple[str, float]] = []
    identifiers: list[tuple[str, str]] = []
    for name in predictors:
        dtype = frame.schema[name]
        column = frame[name]
        present = column.drop_nulls()
        if present.len() == 0:
            continue
        if y is not None and dtype.is_numeric():
            correlation = _rank_correlation(y, column)
            if correlation is not None and correlation >= RESPONSE_COPY_RANK:
                copies.append((name, correlation))
                continue
        if dtype.is_numeric() and present.len() >= MIN_RANK_PAIRS:
            values = present.cast(pl.Float64).to_numpy()
            distinct = present.n_unique()
            if (distinct / present.len() >= UNIQUE_INTEGER_SHARE
                    and bool(np.all(values == np.round(values)))):
                identifiers.append((name, f"{distinct} distinct whole numbers in "
                                          f"{present.len()} rows"))
    return copies, identifiers


def _rank_correlation(first: pl.Series, second: pl.Series) -> float | None:
    """Absolute Spearman rank correlation over the two columns' complete pairs.

    Ranks use average ties, so the result is the Pearson correlation of the
    ranks. It is ``None`` when fewer than ``MIN_RANK_PAIRS`` pairs are present
    or either column is constant over them.
    """
    pairs = pl.DataFrame({"first": first, "second": second}).drop_nulls()
    if pairs.height < MIN_RANK_PAIRS:
        return None
    a = pairs["first"].cast(pl.Float64).rank().to_numpy()
    b = pairs["second"].cast(pl.Float64).rank().to_numpy()
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return abs(float(np.corrcoef(a, b)[0, 1]))



def _spread_term(column: str, frame: pl.DataFrame, transform: str | None = None) -> str:
    """The log-scale term of one predictor: ``bols`` for two values, else a smooth.

    A smooth on a drafted log transform (``transform="log"``) is written as
    ``bbs(log(x))``, the same as the mean term of that column.
    """
    if frame[column].n_unique() <= 2:
        return f"bols({quote_name(column)})"
    argument = f"log({quote_name(column)})" if transform == "log" else quote_name(column)
    return f"bbs({argument}, df={DRAFT_DF:g})"


def _term_variables(term: str, predictors: list[str]) -> list[str]:
    from mboostrs.formula import term_variables

    read = set(term_variables(term))
    return [name for name in predictors if name in read]


def _unique(messages: list[str]) -> list[str]:
    """The messages with repeats removed, first occurrence kept in order."""
    return list(dict.fromkeys(messages))


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_none(value):
    return float(value) if _finite(value) else None


def _num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain(value):
    """Recursively convert numpy and polars values to JSON-ready Python values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return _num(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
