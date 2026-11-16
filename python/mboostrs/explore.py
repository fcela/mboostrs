"""Tools for the loop between fitting a model and changing its specification.

Fitting is one call. Deciding what to fit next is the part that takes the
afternoon: which terms are earning their place, whether a smooth is buying
anything a straight line would not, which variables the fit is most sensitive
to, and what is worth trying next. Each of those questions is answerable by
refitting and comparing, and each is tedious enough by hand that it usually does
not get asked.

    report = term_report("rent ~ bbs(area, df=4) + bols(bath)", data, "rent")
    report.contributions      # what each term earns out of fold
    report.shapes             # what each term would earn as a line, a spline, a tree
    suggest_specifications(report)   # ranked changes worth trying

Every number is out-of-fold. Comparing specifications on the training risk
rewards flexibility rather than fit, and the whole point of the loop is to
decide between specifications of different flexibility.
"""

from __future__ import annotations

import re
import warnings
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

from mboostrs._comparison_identity import check_evaluations
from mboostrs._control import boost_control
from mboostrs._extrapolation import one_summary
from mboostrs._names import NAME_SRC as N
from mboostrs._names import mask_names, partition_tilde, quote_name, quote_response, split_top_level, unquote_name
from mboostrs._names import split_terms as _split_terms
from mboostrs._output import export_result, output_frame
from mboostrs._validate import check_fit_budget, check_int_at_least, check_unit_interval
from mboostrs._weights import normalize_weights
from mboostrs.agent_docs import guide_hint
from mboostrs.cvrisk import BOUNDARY_SHARE, is_at_boundary
from mboostrs.families import (
    AdaExp,
    Binomial,
    Cindex,
    CoxPH,
    Gaussian,
    GaussReg,
    Gehan,
    HingeLoss,
    Loglog,
    Lognormal,
    Multinomial,
    Weibull,
)
from mboostrs.folds import FoldScheme, grouped_folds, rolling_folds, stratified_folds
from mboostrs.formula import _SURV_RE, term_variables


class FoldError(ValueError):
    """A fold has no usable rows: its training or its evaluation side is empty
    once rows with a missing value are removed.

    ``cv_risk(..., return_result=True)`` keeps such a fold as a failed record.
    Any other error raises, including a response or family the fit cannot read,
    because it would fail every fold alike and recording it would leave a
    result that looks complete.
    """


#: Families whose response is a `Surv(time, status)` pair.
_SURVIVAL_FAMILIES = (CoxPH, Weibull, Loglog, Lognormal, Gehan, Cindex)

_TERM_RE = re.compile(rf"^\s*(\w+)\s*\(\s*({N})")

#: Shapes a continuous term is tried in, cheapest first.
SHAPE_CANDIDATES = ("bols", "bbs", "btree")


def split_terms(rhs):
    """The terms of a right-hand side, split on top-level ``+``."""
    return _split_terms(rhs)


def term_variable(term):
    """The column a term is built on, or None if it cannot be read."""
    m = _TERM_RE.match(term)
    return unquote_name(m.group(2)) if m else (unquote_name(term) or None)


def _shape_of(term):
    m = _TERM_RE.match(term)
    return m.group(1) if m else None


def _as_shape(term, shape, df=4.0):
    """The same variable, fitted with a different learner."""
    variable = term_variable(term)
    if shape == "bols":
        return f"bols({quote_name(variable)})"
    if shape == "btree":
        return f"btree({quote_name(variable)})"
    if shape == "brandom":
        return f"brandom({quote_name(variable)}, df={df})"
    return f"bbs({quote_name(variable)}, df={df})"


def _shape_candidates(column):
    """The learners worth trying for a column, given what it is.

    An unordered factor has only one: its codes are labels, so a slope, a
    spline or a threshold over them fits a curve through an alphabetical
    accident. `bols` on such a column is refused by the fit path for the same
    reason, and offering it here only produced that refusal mid-report.

    An ordered factor keeps all three -- its codes are a scale, so a slope
    across levels and a cut between two of them both mean something.
    """
    if column.levels is not None and not column.ordered:
        return ("brandom",)
    return SHAPE_CANDIDATES


def _evaluation_weights(n, folds, weights):
    """Resolve observation weights without recomposing a scheme's fit weights."""
    base = getattr(folds, "base_weights", None) if isinstance(folds, FoldScheme) else None
    for name, value in (("weights", weights), ("FoldScheme.base_weights", base)):
        if value is not None and np.shape(value) != (n,):
            raise ValueError(f"{name} has shape {np.shape(value)}, expected ({n},) to match the data")
    if base is not None:
        base = normalize_weights(base, n, policy="raw").fit
        if weights is not None:
            supplied = normalize_weights(weights, n, policy="raw").fit
            if not np.array_equal(supplied, base):
                raise ValueError(
                    "weights must match the FoldScheme.base_weights. Build "
                    "the scheme with weights= to change both training and "
                    "evaluation weights together."
                )
        return base
    return normalize_weights(weights, n, policy="raw").fit


def _fold_mask(value, shape, name):
    mask = np.asarray(value)
    if mask.shape != shape or mask.dtype != np.dtype(bool):
        raise ValueError(f"FoldScheme.{name} must be a boolean mask of shape {shape}, got {mask.shape}")
    return mask


def _folds(n, folds, seed, weights=None):
    """``(fit_mask, held_mask, complete_fit_weights)`` per fold.

    Bare matrices are resampling multipliers, including bootstrap counts.
    A FoldScheme already contains complete training weights and explicit
    evaluation membership. Its judge mask can restrict the held-out set.
    Rows outside both sets remain unused, as required by rolling validation.
    """
    base = _evaluation_weights(n, folds, weights)
    if isinstance(folds, (int, float, np.integer, np.floating)):
        count = check_int_at_least("folds", folds, 2, allow_none=False)
        if count > n:
            raise ValueError(f"folds={count} exceeds the {n} observations")
        rng = np.random.default_rng(seed)
        assignment = rng.permutation(n) % count
        matrix = np.where(assignment[:, None] != np.arange(count), base[:, None], 0.0)
        held_out = assignment[:, None] == np.arange(count)
    else:
        matrix = np.asarray(folds, dtype=np.float64)
        held_out = None
    if matrix.ndim != 2 or matrix.shape[0] != n or matrix.shape[1] == 0:
        raise ValueError(
            f"folds must be a fold count or an ({n}, B) in-bag weight matrix; "
            f"got shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("folds must contain finite, non-negative weights")
    if isinstance(folds, FoldScheme):
        held_out = _fold_mask(folds.held_out, matrix.shape, "held_out")
        if np.any(held_out & (matrix > 0)):
            raise ValueError("FoldScheme rows cannot both train and be held out in the same fold")
        if folds.judge is not None:
            judge = _fold_mask(folds.judge, matrix.shape, "judge")
            if np.any(judge & ~held_out):
                raise ValueError("FoldScheme.judge must be a subset of held_out")
            held_out = judge
    elif held_out is None:
        held_out = matrix == 0
        with np.errstate(over="ignore"):
            matrix = matrix * base[:, None]
        if not np.all(np.isfinite(matrix)):
            raise ValueError("folds times weights must remain finite and non-negative")

    held_out = held_out & (base > 0)[:, None]
    result = []
    for b in range(matrix.shape[1]):
        fit_weights = np.where(base > 0, matrix[:, b], 0.0)
        fit_mask = fit_weights > 0
        if not fit_mask.any():
            raise ValueError(f"fold {b} has no positive-weight training rows")
        if not held_out[:, b].any():
            raise ValueError(f"fold {b} has no positive-weight evaluation rows")
        result.append((fit_mask, held_out[:, b], fit_weights))
    return result


#: Learners whose column identifies a group: rows sharing the value are one
#: sampling unit, so a fold that splits the value across its two sides scores
#: memory as well as generalization.
_GROUP_LEARNERS = ("brandom", "bmrf")


def _group_columns(rhs):
    """Columns a right-hand side treats as groups, as ``(learner, column)``."""
    columns = []
    rhs_list = list(rhs.values()) if isinstance(rhs, dict) else [rhs]
    for one_rhs in rhs_list:
        for term in split_terms(one_rhs):
            learner = _shape_of(term)
            if learner in _GROUP_LEARNERS:
                columns.append((learner, term_variable(term)))
    return columns


def _warn_if_folds_split_groups(rhs, data, response, fold_list, usable):
    """Warn when a group term is scored on folds that split its groups.

    A `brandom` term whose groups appear on both sides of a fold is fitted to
    the very rows it is scored on, so the fold's held-out score partly rewards
    memorized level effects. The score is still computed as asked; this names
    what it measures, and the folds that would measure prediction in new
    groups instead.
    """
    from mboostrs._frame import read_column

    group_columns = _group_columns(rhs)
    if not group_columns:
        return
    n_folds = len(fold_list)
    for learner, column in group_columns:
        if column is None or column not in _frame_column_set(data):
            continue
        try:
            codes = read_column(data, column).values
        except (TypeError, ValueError, KeyError):
            continue
        split_folds = 0
        for fit_rows, held_rows, _fold_weights in fold_list:
            trained = codes[fit_rows & usable]
            evaluated = codes[held_rows & usable]
            trained = trained[np.isfinite(trained)]
            evaluated = evaluated[np.isfinite(evaluated)]
            if evaluated.size and trained.size:
                if np.intersect1d(trained, evaluated).size:
                    split_folds += 1
        if split_folds:
            warnings.warn(
                f"{learner}({column}) is scored on folds that split its groups: "
                f"in {split_folds} of {n_folds} folds, rows of a group are held "
                "out while other rows of the same group train the fold, so the "
                "score partly rewards memorized group effects. Folds that keep "
                "each group on one side, such as "
                "mboostrs.folds.grouped_folds, measure prediction in new groups "
                "instead." + guide_hint("pitfalls"),
                stacklevel=3,
            )


def _frame_column_set(data):
    from mboostrs._frame import column_names

    try:
        return set(column_names(data))
    except (TypeError, ValueError):
        return set()


def _score(family, y, predicted, weights):
    """Mean family loss evaluated on link-scale predictions."""
    from mboostrs._rust import family_diagnostics
    from mboostrs.families import _family_params, _family_rust_name

    _offset, _gradient, risk = family_diagnostics(
        _family_rust_name(family),
        _family_params(family),
        np.ascontiguousarray(np.asarray(y, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(predicted, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(weights, dtype=np.float64)),
    )
    return float(risk) / max(float(np.sum(weights)), 1e-12)


def _fitted_nuisance(model, mstop=None):
    """`(scalar, vector)` nuisance arguments for the native family calls.

    The fit's nuisance value at step `mstop` (default: the model's own step
    count), split the way `family_held_out_score` and `family_diagnostics`
    take it: one value as `scalar`, several (PropOdds thresholds) as
    `vector`, `(None, None)` for a family without one.
    """
    position = model.mstop if mstop is None else int(mstop)
    # GAMBoostModel and GLMBoostModel keep the per-step path under
    # different names; `nuisance()` reads it at the model's own step count.
    path = (getattr(model, "_family_sigma_path", None)
            or getattr(model, "_nuisance_vec_path", None))
    if path and position < len(path):
        nuisance = path[position]
    else:
        nuisance = model.nuisance()
    values = (np.ravel(np.asarray(nuisance, dtype=np.float64))
              if nuisance is not None else np.array([]))
    scalar = float(values[0]) if values.size == 1 else None
    vector = values.tolist() if values.size > 1 else None
    return scalar, vector


def _fitted_family_score(model, family, y, predicted, weights, mstop=None):
    """Mean family loss on held-out rows, using the fit's own nuisance values.

    A family with a nuisance parameter (NBinomial's dispersion, GammaReg's
    shape, PropOdds' thresholds, the parametric survival families' scale)
    estimates it on the training rows while boosting. Held-out loss must be
    evaluated at that training estimate, at the step count being scored;
    `_score` builds the family from its constructor arguments and so scores
    at the constructor default instead. Multinomial and LSS fits carry no
    scalar or vector nuisance on this path and keep `_score`.
    """
    from mboostrs._rust import family_held_out_score
    from mboostrs.families import Multinomial, _family_params, _family_rust_name

    if isinstance(family, Multinomial) or not hasattr(model, "nuisance"):
        return _score(family, y, predicted, weights)
    scalar, vector = _fitted_nuisance(model, mstop)
    return float(family_held_out_score(
        _family_rust_name(family), _family_params(family),
        np.ascontiguousarray(np.asarray(y, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(predicted, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(weights, dtype=np.float64)),
        scalar, vector))


def _fitted_family_rows(model, family, y, predicted, mstop=None):
    """Each held-out row's family loss at the fit's nuisance values, so the
    records reconstruct the fold score (their weighted mean). NaN rows when
    the loss couples observations (CoxPH, AUC, Cindex, Gehan) and for
    Multinomial, whose held-out rows are scored by a separate route."""
    from mboostrs._rust import family_held_out_rows
    from mboostrs.families import Multinomial, _family_params, _family_rust_name

    n = len(np.asarray(predicted))
    if isinstance(family, Multinomial) or not hasattr(model, "nuisance"):
        return np.full(n, np.nan)
    scalar, vector = _fitted_nuisance(model, mstop)
    rows = family_held_out_rows(
        _family_rust_name(family), _family_params(family),
        np.ascontiguousarray(np.asarray(y, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(predicted, dtype=np.float64)),
        scalar, vector)
    return np.full(n, np.nan) if rows is None else np.asarray(rows, dtype=np.float64)


def _survival_names(response, family):
    """``(time, status)`` column names when `family` is a survival family.

    A survival family takes a ``Surv(time, status)`` response, given as that
    text or as a ``(time, status)`` pair. A Surv response with a family that
    is not a survival family is refused, so that the time column is never
    fitted as an ordinary response by mistake. Returns None for other families.
    """
    survival = isinstance(family, _SURVIVAL_FAMILIES)
    names = None
    if isinstance(response, tuple) and len(response) == 2:
        names = (str(response[0]), str(response[1]))
    elif isinstance(response, str):
        match = _SURV_RE.match(response.strip())
        if match:
            names = (match.group(1), match.group(2))
    if survival and names is None:
        raise ValueError(
            f"{type(family).__name__} requires a Surv(time, status) response: pass "
            "response='Surv(time, status)' or response=('time', 'status')")
    if not survival and names is not None and isinstance(response, str):
        raise ValueError(
            f"a Surv(time, status) response needs a survival family such as "
            f"CoxPH() or Weibull(), not {type(family).__name__}")
    return names if survival else None


def _family_target(y, survival_event, rows):
    """The response `family_held_out_score` takes for `rows`.

    A survival family takes ``[time, status]`` stacked (length 2 per row);
    every other family takes the response itself.
    """
    if survival_event is None:
        return y
    return np.concatenate([y, survival_event[rows]])


def _lss_held_out_rows(model, y_held, link, fold):
    """Per-row negative log-likelihood and response-scale parameters of an LSS fit.

    `link` is the fit's link-scale prediction for the held-out rows, keyed by
    parameter name. The losses come from the core family's `row_nll`, the
    per-observation terms the fit and the core cross-validated risk sum, so the
    fold score is the risk the core reports for those rows. The parameters are
    an ``(n, K)`` array in the family's parameter order. A response the family
    refuses (outside its support) fails the fold.
    """
    from mboostrs._rust import lss_row_nll

    names = tuple(model.param_names)
    etas = [np.ascontiguousarray(link[name], dtype=np.float64) for name in names]
    try:
        loss, parameters = lss_row_nll(
            model._family._rust_name, len(names),
            np.ascontiguousarray(y_held, dtype=np.float64), etas)
    except ValueError as exc:
        raise FoldError(f"fold {fold} held-out rows cannot be scored: {exc}") from exc
    return np.asarray(loss), np.column_stack([np.asarray(p) for p in parameters])


def _failed_fold_row(b, fit_rows, held_rows, fold_weights, exc):
    """The fold record of a fold that could not be fitted or scored.

    The row keeps the fold's sizes and weights, so the failure can be read
    against the design, and names the exception. Its score is NaN, and so is
    the result's score: an equal-fold mean that left this fold out would
    describe a different set of folds.
    """
    evaluation_weight = np.asarray(fold_weights, dtype=float)[np.asarray(held_rows, bool)]
    return {
        "fold": b, "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "score": float("nan"),
        "n_training": int(np.sum(fit_rows)),
        "n_evaluation": int(np.sum(held_rows)),
        "training_weight_total": float(np.sum(np.asarray(fold_weights)[np.asarray(fit_rows, bool)])),
        "evaluation_weight_total": float(np.sum(evaluation_weight)),
        "n_unseen_level_rows": None,
        "mstop": None,
        "inner_mstop": None,
        "inner_at_boundary": None,
        "fit_count": None,
        "training_rows": None,
        "evaluation_rows": None,
    }


def _scoring_loss(score, y, predicted, parameter):
    """Per-observation loss for scores defined outside the fitted family."""
    from mboostrs._rust import response_score

    _, losses = response_score(np.asarray(y, dtype=float),
                               np.asarray(predicted, dtype=float).reshape(-1, 1),
                               np.ones(len(y)), score, parameter)
    return np.asarray(losses)


def cv_risk(rhs, data, response, family=None, folds=5, mstop=500, nu=0.1,
            seed=0, weights=None, score="family", na_action="error",
            score_parameter=None, observation_ids=None, return_result=False,
            retain_arrays=True, unseen_levels="average", extrapolation="warn",
            step="fixed", response_transform=None, max_mstop=None, mstop_grid=None,
            estimator="boosting",
            **fit_kwargs):
    """Out-of-fold risk of a right-hand side.

    With a distributional family (`GaussianLSS`, ...), `rhs` may instead be a
    mapping of parameter name to right-hand side — ``{"mu": "bbs(x)",
    "sigma": "1"}`` — or a plain string shared by every parameter, which is
    what a single-formula `gamboostLSS` call means. The score is then the
    family's own joint proper score, the negative log predictive density
    (``score='family'`` is required; the single-quantity scores name a
    different target), and the result's per-record predictions are one value
    per distribution parameter, in the family's `param_names` order. A row
    missing in any parameter's formula is dropped on both sides of the fold
    under ``na_action='omit'``.

    A survival family (`CoxPH`, `Weibull`, `Loglog`, `Lognormal`, `Gehan`,
    `Cindex`) takes a ``Surv(time, status)`` response, given as that text or as
    a ``(time, status)`` pair. It is scored by its family loss
    (``score='family'``) on the link scale, and each held-out record carries
    the ``status`` as ``event``.

    With ``return_result=True``, a fold whose training or evaluation side is
    empty after ``na_action='omit'`` stays in the result as a record with
    ``status='failed'`` and its error. The result's score is then NaN, because
    a mean over the remaining folds would describe fewer folds. The float return
    raises instead. Other errors (a response the family cannot read, an
    ``unseen_levels='error'`` refusal) raise in both cases.

    The same folds are used for every call with the same ``seed``, ``folds``
    and row count, so two specifications are compared on the same partition.

    Parameters
    ----------
    folds : int, ndarray or FoldScheme
        A fold count, an ``(n, B)`` in-bag weight matrix, or a scheme from
        `mboostrs.folds`. Pass a scheme to score a specification the way the
        data is actually structured -- grouped, stratified, or in time order.
        A bare matrix contains resampling multipliers: multiply its column
        by ``weights`` once to obtain training weights. Bootstrap counts
        retain their multiplicities. A ``FoldScheme`` already contains
        complete training weights and explicit held-out/judge masks; rows
        outside both training and evaluation remain unused.
    weights : array-like, optional
        Observation weights for training and evaluation. With a constructed
        ``FoldScheme``, defaults to its retained ``base_weights``; an explicit
        vector must match them and is not multiplied into training twice.
        For a manually constructed scheme without ``base_weights``, this
        vector weights evaluation and excludes zero-weight rows, while the
        scheme's training weights are used as supplied.
    score : {"family", "mse", "brier", "log_loss", "pinball", "expectile"}
        ``"family"`` evaluates family loss on link predictions. ``"mse"``
        evaluates squared error on response predictions; for Binomial this
        is the Brier score, with signed {-1, +1} labels converted to {0, 1}.
        ``"brier"`` names that probability score explicitly and extends it to
        the sum of squared class-probability errors for Multinomial.
        ``"log_loss"`` evaluates Bernoulli log loss on response probabilities.
        ``"pinball"`` and ``"expectile"`` evaluate their named asymmetric
        response losses at ``score_parameter``. Scores outside ``"family"``
        require one response prediction per observation and can compare fitted
        families that target the same held-out quantity.
    score_parameter : float, optional
        Required strictly between zero and one for pinball and expectile loss;
        rejected for other scores so a silently ignored target cannot enter a
        comparison.
    na_action : {"error", "omit"}
        Passed to the fit. ``"omit"`` drops rows with missing values in the
        columns the formula uses, on both sides: a held-out row with a missing
        predictor cannot be scored either.
    observation_ids : array-like, optional
        Unique record identifiers retained with out-of-fold predictions. Row
        positions are used by default. These identify records, not independent
        sampling units; use a grouped ``FoldScheme`` for clustered data.
    return_result : bool, default False
        Return an :class:`EvaluationResult` with predictions and provenance.
        The default float preserves the historical ``cv_risk`` contract.
    retain_arrays : bool, default True
        When returning a result, retain one record per held-out prediction and
        the training/evaluation indices per fold. Set false for compact
        provenance when row-level arrays are unnecessary.
    unseen_levels : {"average", "zero_effect", "error"}, default "average"
        How a held-out row is predicted when its ``brandom`` level does not
        occur in the fold's training rows, which is every row of a held-out
        group under a grouped fold scheme. ``"average"`` predicts it as a
        member of an average group, the prediction for a new group (see
        :meth:`GAMBoostModel.predict`). ``"zero_effect"`` gives the level a
        zero coefficient, which reproduces what ``model.cvrisk`` computes on
        the same folds. ``"error"`` refuses the fold. Each fold record counts
        the rows this applied to as ``n_unseen_level_rows``.
    extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
        How a held-out row outside its fold's training range is predicted,
        as in :meth:`GAMBoostModel.predict`. ``"warn"`` predicts and reports
        the fold rows that left the range, once per call; ``"clip"`` evaluates
        the spline at the nearest training value; ``"error"`` refuses; and
        ``"allow"`` predicts silently.
    mstop : int or "cv"
        A fixed budget, or ``"cv"`` to tune it *inside* each outer fold. A
        fixed budget compares specifications at one number of iterations,
        which favours whichever of them happens to suit it; tuning inside the
        fold compares each specification at its own best stop, and the outer
        score stays honest because the tuning never sees the held-out rows.
        It costs an inner cross-validation per outer fold.
    max_mstop : int, optional
        With ``mstop="cv"``, the largest step count the inner tuning may
        reach; the default is 500. The inner search starts at 500 steps, or at
        this cap when it is smaller, and doubles while the inner risk is still
        in the boundary band at the end of the grid, up to the cap. A fold
        whose search reaches the cap with the risk still falling is reported
        in the fold records (``inner_at_boundary``) and the call warns. Passing
        it with a fixed ``mstop`` is an error.
    mstop_grid : sequence of int, optional
        The risk along a grid of step counts from one call. Each fold is
        fitted once to ``max(mstop_grid)`` steps and its path is cut to every
        grid value, so the grid costs the fits of one `cv_risk` call. ``mstop``
        is then not used. Supports every ``score`` on a standard
        single-response family with ``na_action='error'``; other combinations
        raise. The return value is a frame with one row per grid value:
        ``mstop``, ``risk`` (the equal-fold mean, the number `cv_risk(mstop=m)`
        returns for that value), ``fold_se`` (its standard error over folds)
        and ``fold_scores`` (each fold's score at that step count, in fold
        order, so paired comparisons across specifications can be made on the
        shared folds).
    estimator : {"boosting", "reml"}, default "boosting"
        ``"boosting"`` retains the usual fixed-step fit. ``"reml"`` fits a
        direct Gaussian GAM independently on each training fold; it accepts
        ``score="family"`` or ``"mse"`` and does not use boosting-step,
        response-transform, or other boosting fit options.

    Returns
    -------
    float
        Arithmetic mean of the fold scores. Each fold score is normalized
        by its evaluation weight total, so folds contribute equally even
        when their sizes differ. An empty effective training or evaluation
        set raises rather than silently removing that fold from the average.
        With ``return_result=True`` the same number is
        :attr:`EvaluationResult.score`, also readable as
        :attr:`EvaluationResult.aggregate`, so the float and the result agree.
    """
    # `_warn_as` is internal: compare_specifications passes None and warns in
    # its own words. It sits in **fit_kwargs so the public signature stays as is.
    warn_as = fit_kwargs.pop("_warn_as", "cv_risk")
    if estimator not in ("boosting", "reml"):
        raise ValueError("estimator must be 'boosting' or 'reml'")
    if estimator == "reml":
        if not isinstance(family or Gaussian(), Gaussian):
            raise ValueError("estimator='reml' currently supports Gaussian only")
        if score not in ("family", "mse"):
            raise ValueError("estimator='reml' supports score='family' or 'mse'")
        if (mstop == "cv" or mstop_grid is not None or max_mstop is not None
                or step != "fixed" or response_transform is not None):
            raise ValueError("estimator='reml' does not use boosting step or response-tuning options")
        unsupported = set(fit_kwargs) - {"offset"}
        if unsupported:
            raise ValueError(f"estimator='reml' does not support fit option(s): {', '.join(sorted(unsupported))}")
    family = Gaussian() if family is None else family
    from mboostrs._frame import take_rows
    from mboostrs.gamboost import _check_unseen_levels, gamboost

    unseen_levels = _check_unseen_levels(unseen_levels)

    if response_transform is not None:
        if response_transform not in ("log", "log1p"):
            raise ValueError("response_transform must be 'log', 'log1p' or None")
        if not isinstance(family, Gaussian) or score != "mse" or mstop == "cv":
            raise ValueError("response-transform comparison needs Gaussian, score='mse' and fixed mstop")
    fit_formula = (f"{response_transform}({response}) ~ {rhs}"
                   if response_transform else f"{quote_response(response)} ~ {rhs}")
    if getattr(family, "_multivariate", False) or getattr(family, "_known_variance", False):
        raise ValueError(
            f"{type(family).__name__} has a multi-column or (outcome, variance) response, "
            "which evaluate_specification and cv_risk do not read; score the model with "
            "its cvrisk() method instead")
    survival_names = _survival_names(response, family)
    if survival_names is not None:
        fit_formula = f"Surv({survival_names[0]}, {survival_names[1]}) ~ {rhs}"
        response = f"Surv({survival_names[0]}, {survival_names[1]})"

    valid_scores = ("family", "mse", "brier", "log_loss", "pinball", "expectile")
    if score not in valid_scores:
        raise ValueError(f"score must be one of {valid_scores}, got {score!r}")
    if survival_names is not None and score != "family":
        raise ValueError(
            f"{type(family).__name__} is scored by its family loss (score='family'); "
            f"score={score!r} is a response-scale score and names a different target")
    if score in ("pinball", "expectile"):
        if score_parameter is None:
            raise ValueError(f"score_parameter is required for score={score!r}")
        score_parameter = check_unit_interval("score_parameter", score_parameter)
    elif score_parameter is not None:
        raise ValueError(f"score_parameter is only used by pinball or expectile loss, got score={score!r}")
    if mstop_grid is not None:
        # Reached only after the arguments above are validated, so a grid call
        # refuses a bad score or a stray score_parameter exactly as a single
        # mstop call does.
        if return_result:
            raise ValueError(
                "mstop_grid returns one risk per grid point; return_result=True describes "
                "one specification at one step count. Call cv_risk with mstop=<value> for that")
        if response_transform is not None:
            raise NotImplementedError(
                "mstop_grid does not yet take a response transform; call cv_risk "
                "once per mstop for a transformed response")
        return _cv_risk_grid(rhs, data, response, family, folds, mstop_grid, nu=nu,
                             seed=seed, weights=weights, score=score,
                             score_parameter=score_parameter, na_action=na_action,
                             unseen_levels=unseen_levels, extrapolation=extrapolation,
                             step=step, **fit_kwargs)
    class_labels = None
    survival_event = None
    if isinstance(family, Multinomial):
        if score not in ("brier", "log_loss"):
            raise ValueError(
                "Multinomial evaluation supports score='brier' or 'log_loss'; "
                "scalar MSE, pinball, expectile, and partial family risk are not comparable"
            )
        y, class_labels = _multinomial_response(data, response)
        if family.n_classes is not None and family.n_classes != len(class_labels):
            raise ValueError(
                f"Multinomial n_classes={family.n_classes} but evaluation data has "
                f"{len(class_labels)} observed classes; every probability column needs a label"
            )
        family = Multinomial(n_classes=len(class_labels))
    elif survival_names is not None:
        # `y` holds the observed times; the status column is kept beside it and
        # stacked under the times only where the family's loss is computed.
        y = np.asarray(_response_values(data, survival_names[0]), dtype=float)
        survival_event = np.asarray(_response_values(data, survival_names[1]), dtype=float)
        if survival_event.shape != y.shape:
            raise ValueError("the time and status columns must have the same length")
    else:
        y = np.asarray(_response_values(data, response), dtype=float)
    if response_transform is not None:
        bound = 0.0 if response_transform == "log1p" else 1.0
        invalid = (y < 0 if response_transform == "log1p" else y <= 0)
        if np.any(invalid) or np.any(~np.isfinite(y)):
            raise ValueError(f"{response_transform} response requires finite "
                             f"{'nonnegative' if bound == 0.0 else 'positive'} observations")
    lss_formulas = None
    is_lss = hasattr(family, "param_names") and hasattr(family, "pdf")
    if isinstance(rhs, dict) and not is_lss:
        raise ValueError(
            "a right-hand side mapping names distribution parameters; it needs a "
            "distributional family, e.g. family=GaussianLSS(), not "
            f"{type(family).__name__}")
    if is_lss:
        if score != "family":
            raise ValueError(
                "a distributional family is scored by its own joint proper score "
                "(score='family': the negative log predictive density); 'mse' and "
                "the other single-quantity scores name a different target")
        lss_formulas = _lss_formulas(rhs, response, family)
    n = len(y)
    if observation_ids is None:
        observation_ids = np.arange(n)
    else:
        observation_ids = np.asarray(observation_ids)
        if observation_ids.shape != (n,):
            raise ValueError(
                f"observation_ids has shape {observation_ids.shape}, expected ({n},)"
            )
        if len(set(observation_ids.tolist())) != n:
            raise ValueError("observation_ids must uniquely identify the supplied rows")
    w = _evaluation_weights(n, folds, weights)
    # Taken from the observed response, before any recoding below.
    identity = _evaluation_data_identity(y, survival_event, w, observation_ids)
    if score != "family" and isinstance(family, (Binomial, AdaExp, HingeLoss)) and np.any(y[w > 0] < 0):
        y = (y + 1.0) / 2.0
    vector_offset = None
    if lss_formulas is None and fit_kwargs.get("offset") is not None and np.ndim(fit_kwargs["offset"]) > 0:
        vector_offset = np.asarray(fit_kwargs["offset"], dtype=float)
        if vector_offset.shape != (n,):
            raise ValueError("offset must match response rows")
    cap = _check_max_mstop(mstop, max_mstop)
    first_grid = None if cap is None else min(_MAX_TUNED_MSTOP, cap)
    errors = []
    records = []
    fold_rows = []
    evaluated = np.zeros(n, dtype=bool)
    # A held-out fold can contain values outside the range the fold's basis was
    # built on; the learner extrapolates and says so. That is expected here and
    # not something the caller can act on, so every fold's `predict` reports to
    # the collector instead of burning one warning paragraph each, and the loop
    # prints a single summary naming the farthest offender.
    import warnings as _warnings

    from mboostrs._extrapolation import collecting, forward_extrapolation, summarize

    with collecting() as extrapolated:
        fold_list = _folds(n, folds, seed, w)
        if _group_columns(rhs):
            # Rows a fold would drop under na_action='omit' are on neither
            # side of it, so they cannot split a group that is being scored.
            if na_action == "omit":
                formulas = (list(lss_formulas.values()) if lss_formulas is not None
                            else [fit_formula])
                usable = ~_any_missing_rows(formulas, data, np.arange(n))
            else:
                usable = np.ones(n, dtype=bool)
            # A known-group scheme spreads groups over folds by design, so the
            # memorization warning describes what the caller asked for.
            if not (isinstance(folds, FoldScheme) and folds.known_groups is not None):
                _warn_if_folds_split_groups(rhs, data, response, fold_list, usable)
        for b, (fit_rows, held_rows, fold_weights) in enumerate(fold_list):
            # A training fold without every class is a scheme the caller must
            # change, not a fit that failed, so it is refused rather than recorded.
            if isinstance(family, Multinomial):
                present = set(y[fit_rows].astype(int).tolist())
                required = set(range(len(class_labels)))
                if present != required:
                    missing_classes = [
                        class_labels[index] for index in sorted(required - present)
                    ]
                    raise ValueError(
                        f"fold {b} training rows omit classes {missing_classes!r}; "
                        "use stratified folds with every class represented in training"
                    )
            try:
                train = take_rows(data, fit_rows)
                weights_fit = fold_weights[fit_rows]
                fold_options = dict(fit_kwargs)
                if vector_offset is not None:
                    fold_options["offset"] = vector_offset[fit_rows]
                if lss_formulas is not None:
                    # With na_action="omit" the rows the formulas need complete
                    # are taken out here, on both sides of the fold, and the
                    # counts below stay aligned. "impute" and "error" keep every
                    # row and are applied by gamboostLSS itself.
                    fit_index = np.flatnonzero(fit_rows)
                    if na_action == "omit":
                        fit_index = fit_index[~_any_missing_rows(
                            list(lss_formulas.values()), data, fit_index)]
                        if fit_index.size == 0:
                            raise FoldError(
                                f"fold {b} has no training rows after na_action='omit'")
                        train = take_rows(data, fit_index)
                        weights_fit = fold_weights[fit_index]
                budget = cap if mstop == "cv" else int(mstop)
                if lss_formulas is not None:
                    from mboostrs.gamlss import gamboostLSS

                    model = gamboostLSS(lss_formulas, data=train, family=family,
                                        weights=weights_fit,
                                        control=boost_control(budget, nu),
                                        na_action=na_action, **fit_kwargs)
                elif estimator == "reml":
                    from mboostrs.direct_gam import gam
                    model = gam(f"{quote_response(response)} ~ {rhs}", data=train, family=family,
                                weights=weights_fit, offset=fold_options.pop("offset", None),
                                na_action=na_action)
                    if not model.convergence.get("converged", True):
                        raise FoldError(
                            "direct REML fit did not converge: "
                            f"{model.convergence.get('stop_reason', 'unknown')}")
                else:
                    model = gamboost(fit_formula, data=train, family=family,
                                     weights=weights_fit, na_action=na_action,
                                     control=boost_control(budget, nu, step=step), **fold_options)
                fold_fit_count = 1
                inner_mstop = inner_at_boundary = None
                if mstop == "cv":
                    inner_train = fit_rows
                    if na_action == "omit":
                        # `model` sees `train` minus the rows the fit dropped, so
                        # the inner scheme must be drawn over exactly those rows or
                        # the fold matrix will not match the model's row count.
                        if lss_formulas is not None:
                            kept = fit_index
                        else:
                            missing = _missing_rows(fit_formula, data,
                                                    np.flatnonzero(fit_rows))
                            kept = np.flatnonzero(fit_rows)[~missing]
                        inner_train = np.zeros_like(fit_rows)
                        inner_train[kept] = True
                        if not inner_train.any():
                            raise FoldError(f"fold {b} has no training rows after na_action='omit'")
                    inner = _inner_folds(folds, inner_train, seed,
                                         training_weights=fold_weights[inner_train])
                    if score == "family":
                        model, search = _tuned(model, seed, inner,
                                               start=first_grid, max_mstop=cap)
                        fold_fit_count += search.fits
                    else:
                        from mboostrs._metric_tuning import tune_response_mstop

                        inner_options = dict(fit_kwargs)
                        if vector_offset is not None:
                            inner_options["offset"] = vector_offset[inner_train]
                        tuned = tune_response_mstop(
                            fit_formula, take_rows(data, inner_train), family,
                            folds=inner if inner is not None else 5, start=first_grid, nu=nu,
                            extensions=None, max_mstop=cap, seed=seed, step=step,
                            score=score, score_parameter=score_parameter,
                            unseen_levels=unseen_levels, full_fit=model,
                            weights=fold_weights[inner_train], na_action=na_action,
                            _warn=False, **inner_options)
                        fold_fit_count += tuned.total_fits
                        model = tuned.model
                        search = _InnerSearch(tuned.mstop, tuned.largest,
                                              tuned.at_boundary, tuned.total_fits)
                    inner_mstop, inner_at_boundary = search.mstop, bool(search.at_boundary)
                held_index = np.flatnonzero(held_rows)
                if na_action == "omit":
                    # A held-out row with a missing predictor cannot be scored either,
                    # so the same rule applies on both sides. Dropping it only from
                    # the fit left `predict` refusing the fold.
                    try:
                        if lss_formulas is not None:
                            keep = ~_any_missing_rows(list(lss_formulas.values()),
                                                       data, held_index)
                        else:
                            keep = ~_missing_rows(fit_formula, data, held_index)
                    except ValueError as exc:
                        raise FoldError(f"fold {b} evaluation with na_action='omit': {exc}") from exc
                    held_index = held_index[keep]
                    if held_index.size == 0:
                        raise FoldError(f"fold {b} has no evaluation rows after na_action='omit'")
                held = take_rows(data, held_index)
                n_unseen = _unseen_level_rows(model, train, held)
                if lss_formulas is not None:
                    # The held-out loss and the response-scale parameters both
                    # come from the core family, from the link-scale predictors.
                    row_loss, predicted = _lss_held_out_rows(
                        model, y[held_index], model.predict(
                            newdata=held, type="link", unseen_levels=unseen_levels,
                            extrapolation=extrapolation), b)
                else:
                    predicted = np.asarray(model.predict(
                        newdata=held, type="link" if score == "family" else "response",
                        unseen_levels=unseen_levels, extrapolation=extrapolation,
                        **({"offset": vector_offset[held_index]} if vector_offset is not None else {}),
                    ))
                    if response_transform is not None:
                        from mboostrs._rust import inverse_response_transform

                        predicted = np.asarray(inverse_response_transform(
                            np.ascontiguousarray(predicted, dtype=float), response_transform),
                            dtype=float)
                    expected_shape = (
                        (len(held_index), len(class_labels))
                        if isinstance(family, Multinomial) else (len(held_index),)
                    )
                    if predicted.shape != expected_shape:
                        raise NotImplementedError(
                            f"score={score!r} received {type(family).__name__} prediction "
                            f"shape {predicted.shape}, expected {expected_shape}"
                        )
                y_held, w_held = y[held_index], w[held_index]
                if lss_formulas is not None:
                    fold_score = float(np.sum(np.where(w_held > 0, w_held * row_loss, 0.0))
                                       / np.sum(w_held))
                    if not np.isfinite(fold_score):
                        bad = np.flatnonzero((w_held > 0) & ~np.isfinite(row_loss))
                        if bad.size == 0:
                            raise FoldError("the held-out score is not finite")
                        raise FoldError(
                            f"the held-out density of {bad.size} row(s) is zero or undefined, "
                            "so their negative log-likelihood is not finite (first at row "
                            f"{int(held_index[bad[0]])}): "
                            f"{type(family).__name__} predicts a spread too small for "
                            "those rows. Use extrapolation='clip', which holds the spline "
                            "flat outside its training range, or a smaller spread model.")
                elif score != "family":
                    from mboostrs._rust import response_score

                    values = predicted[:, None] if predicted.ndim == 1 else predicted
                    fold_score, row_loss = response_score(
                        np.asarray(y_held, dtype=float), np.ascontiguousarray(values, dtype=float),
                        np.asarray(w_held, dtype=float), score, score_parameter)
                    row_loss = np.asarray(row_loss)
                else:
                    target = _family_target(y_held, survival_event, held_index)
                    fold_score = _fitted_family_score(model, family, target, predicted, w_held)
                    row_loss = _fitted_family_rows(model, family, target, predicted)
                evaluated[held_index] = True
                if retain_arrays:
                    records.extend({
                        "record_id": observation_ids[row].item()
                        if isinstance(observation_ids[row], np.generic)
                        else observation_ids[row],
                        "row": int(row), "fold": b,
                        "observed": float(y_held[j]),
                        "prediction": (predicted[j].tolist()
                                       if predicted.ndim == 2 else float(predicted[j])),
                        "evaluation_weight": float(w_held[j]),
                        "loss": float(row_loss[j]),
                        **({"event": float(survival_event[row])}
                           if survival_event is not None else {}),
                    } for j, row in enumerate(held_index))
                fold_rows.append({
                    "fold": b, "status": "ok", "error": None, "score": fold_score,
                    "estimator": estimator,
                    "n_training": int(np.sum(fit_rows)),
                    "n_evaluation": int(len(held_index)),
                    "training_weight_total": float(np.sum(fold_weights[fit_rows])),
                    "evaluation_weight_total": float(np.sum(w_held)),
                    "n_unseen_level_rows": n_unseen,
                    "mstop": None if estimator == "reml" else int(model.mstop),
                    "inner_mstop": inner_mstop,
                    "inner_at_boundary": inner_at_boundary,
                    "fit_count": fold_fit_count,
                    "training_rows": tuple(np.flatnonzero(fit_rows).tolist())
                    if retain_arrays else None,
                    "evaluation_rows": tuple(held_index.tolist()) if retain_arrays else None,
                })
            except FoldError as exc:
                # A fold with no usable rows is recorded, not dropped: the result
                # keeps its row and the score becomes NaN. The float contract
                # raises instead, as it always has.
                if not return_result:
                    raise
                failed = _failed_fold_row(b, fit_rows, held_rows, fold_weights, exc)
                failed["estimator"] = estimator
                fold_rows.append(failed)

    # One score per fold, in fold order; a failed fold carries NaN.
    errors = [row["score"] for row in fold_rows]
    if mstop == "cv" and warn_as is not None:
        flagged = sum(1 for row in fold_rows if row["inner_at_boundary"])
        if flagged:
            _warnings.warn(
                _inner_boundary_warning(warn_as, cap, flagged, len(fold_rows)),
                RuntimeWarning, stacklevel=2)
    aggregated = summarize(extrapolated)
    forward_extrapolation(extrapolated, stacklevel=2)
    result_score = float(np.mean(errors))
    if not return_result:
        return result_score
    from mboostrs.evaluation import EvaluationResult

    fold_scheme = folds.scheme if isinstance(folds, FoldScheme) else (
        f"random {int(folds)}-fold" if isinstance(folds, (int, float, np.integer,
                                                          np.floating))
        else "supplied weight matrix"
    )
    if isinstance(folds, FoldScheme) and folds.groups is not None:
        evaluation_task = "new_group"
        independent_unit = "group"
        n_independent_units = int(np.unique(folds.groups).size)
    elif isinstance(folds, FoldScheme) and folds.known_groups is not None:
        # Records are the units scored; their group-mates train the same folds,
        # so the records are not independent of one another within a group.
        evaluation_task = "known_group"
        independent_unit = "record"
        n_independent_units = n
    elif isinstance(folds, FoldScheme) and folds.ordered:
        evaluation_task = "future_horizon"
        independent_unit = "ordered_record"
        n_independent_units = n
    else:
        evaluation_task = "new_exchangeable_observation"
        independent_unit = "record"
        n_independent_units = n
    from mboostrs._provenance import fingerprint as _digest

    return EvaluationResult(
        score=result_score, fold_scores=np.asarray(errors),
        response_transform=response_transform or "identity",
        weights_fingerprint=_digest({"weight": np.asarray(w, dtype=np.float64)})["sha256"],
        data_identity=identity,
        formula=("; ".join(lss_formulas.values()) if lss_formulas is not None
                 else fit_formula),
        response=response,
        family=type(family).__name__, score_name=score,
        score_parameter=score_parameter,
        prediction_scale=("distributional parameters, response scale"
                          if lss_formulas is not None
                          else "link" if score == "family" else "response"),
        loss_units=("nats_per_observation" if lss_formulas is not None else {
            "family": "family_loss",
            "mse": "squared_response_units",
            "brier": "squared_probability_units",
            "log_loss": "nats_per_observation",
            "pinball": "response_units",
            "expectile": "squared_response_units",
        }[score]),
        aggregation="equal_fold_mean", fold_scheme=fold_scheme,
        evaluation_task=evaluation_task, independent_unit=independent_unit,
        n_independent_units=n_independent_units, seed=int(seed),
        tuning="inner_cv" if mstop == "cv" else f"fixed_mstop={int(mstop)}",
        data_fingerprint=_holdout_aware_fingerprint(rhs, data),
        fingerprint_scope="all_columns",
        n_observations=n, n_evaluations=int(sum(row["n_evaluation"] for row in fold_rows)),
        fit_count=int(sum(row["fit_count"] or 0 for row in fold_rows)),
        excluded_record_ids=tuple(
            value.item() if isinstance(value, np.generic) else value
            for value in observation_ids[~evaluated]
        ),
        records=tuple(records), folds=tuple(fold_rows),
        warnings=(() if aggregated is None else (aggregated,)),
        class_labels=class_labels,
        build_id=_build_id(),
    )


def _cv_risk_grid(rhs, data, response, family, folds, mstop_grid, *, nu, seed, weights,
                  score, score_parameter=None, na_action, unseen_levels, extrapolation,
                  step="fixed", **fit_kwargs):
    """`cv_risk` at every step count of ``mstop_grid``, from one fit per fold.

    Each fold's model is fitted once, to the largest step count of the grid, on
    the same training rows and recipe `cv_risk` uses at a single ``mstop``. Its
    path is then cut to each grid value (``model[:m]``) and scored on the
    held-out rows, so the grid costs the same number of fits as one `cv_risk`
    call. The returned frame has one row per grid value with the equal-fold mean
    loss, the value `cv_risk(mstop=m)` returns for that value. ``score='family'``
    evaluates the family loss on link predictions, as `cv_risk` does; every
    other score evaluates its response-scale loss with the same recoding and
    weighting a single-``mstop`` call uses.
    """
    from mboostrs._frame import take_rows
    from mboostrs.gamboost import _check_unseen_levels, gamboost

    unseen_levels = _check_unseen_levels(unseen_levels)
    if np.ndim(fit_kwargs.get("offset")) > 0:
        raise NotImplementedError(
            "mstop_grid does not yet take a per-row offset; call cv_risk once per mstop")
    if na_action != "error":
        raise NotImplementedError(
            "mstop_grid does not yet take na_action='omit'; drop the rows with missing "
            "values before the call, or call cv_risk once per mstop")
    family = Gaussian() if family is None else family
    if (isinstance(rhs, dict) or hasattr(family, "param_names")
            or isinstance(family, Multinomial)
            or getattr(family, "_multivariate", False)
            or getattr(family, "_known_variance", False)
            or _survival_names(response, family) is not None):
        raise NotImplementedError(
            f"mstop_grid supports a standard single-response family; {type(family).__name__} "
            "or this response is not supported here. Call cv_risk once per mstop")
    values = sorted({check_int_at_least("mstop_grid", m, 1, allow_none=False)
                     for m in mstop_grid})
    if not values:
        raise ValueError("mstop_grid must name at least one step count")
    y = np.asarray(_response_values(data, response), dtype=float)
    n = len(y)
    w = _evaluation_weights(n, folds, weights)
    if score != "family" and isinstance(family, (Binomial, AdaExp, HingeLoss)) and np.any(y[w > 0] < 0):
        # The same recoding a single-mstop call applies: signed {-1, +1}
        # labels become {0, 1} before a response-scale loss reads them.
        y = (y + 1.0) / 2.0
    fold_list = _folds(n, folds, seed, w)
    scores = np.empty((len(fold_list), len(values)))
    formula = f"{quote_response(response)} ~ {rhs}"
    for b, (fit_rows, held_rows, fold_weights) in enumerate(fold_list):
        train = take_rows(data, fit_rows)
        model = gamboost(formula, data=train, family=family,
                         weights=fold_weights[fit_rows], na_action=na_action,
                         control=boost_control(values[-1], nu, step=step), **fit_kwargs)
        held_index = np.flatnonzero(held_rows)
        held = take_rows(data, held_index)
        y_held, w_held = y[held_index], w[held_index]
        for j, m in enumerate(values):
            # An intercept-only model takes no boosting step, so it has no
            # path to slice; its prediction is the same at every step count.
            cut = model[:m] if model.mstop else model
            if score == "family":
                predicted = np.asarray(cut.predict(newdata=held, type="link",
                                                   unseen_levels=unseen_levels,
                                                   extrapolation=extrapolation))
                scores[b, j] = _fitted_family_score(cut, family, y_held, predicted, w_held)
                continue
            from mboostrs._rust import response_score

            predicted = np.asarray(cut.predict(newdata=held, type="response",
                                               unseen_levels=unseen_levels,
                                               extrapolation=extrapolation))
            columns = predicted[:, None] if predicted.ndim == 1 else predicted
            scores[b, j], _ = response_score(
                y_held, np.ascontiguousarray(columns, dtype=float), w_held,
                score, score_parameter)
    risk = scores.mean(axis=0)
    fold_se = (scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
               if scores.shape[0] > 1 else np.full(len(values), np.nan))
    rows = [{"mstop": int(m), "risk": float(risk[j]),
             "fold_se": None if np.isnan(fold_se[j]) else float(fold_se[j]),
             "fold_scores": [float(scores[b, j]) for b in range(scores.shape[0])]}
            for j, m in enumerate(values)]
    return output_frame(rows, "auto")


def _build_id():
    """The native extension's build identity, or "" when it reports none."""
    try:
        from mboostrs import build_info

        return str(build_info().get("build_id", ""))
    except Exception:  # noqa: BLE001 -- provenance must not break an evaluation
        return ""


def _evaluation_data_identity(y, survival_event, weights, observation_ids) -> str:
    """The `data_identity` of the rows `cv_risk` scores, or ``""`` when the
    response is not numeric."""
    from mboostrs._comparison_identity import data_identity

    response = [y] if survival_event is None else [y, survival_event]
    try:
        return data_identity(response, weights, observation_ids)
    except (TypeError, ValueError):
        return ""


def _holdout_aware_fingerprint(rhs, data):
    """The data fingerprint, having counted a use of a registered test part."""
    from mboostrs.evaluation import data_fingerprint

    fingerprint = data_fingerprint(data)
    _record_holdout_use(rhs, fingerprint)
    return fingerprint


def evaluate_specification(rhs, data, response, **kwargs):
    """Evaluate one specification and retain its held-out evidence.

    Arguments are those of :func:`cv_risk`. The returned
    :class:`~mboostrs.evaluation.EvaluationResult` records the exact score,
    prediction scale, equal-fold aggregation, row identifiers, exclusions,
    fold membership, weights and a data fingerprint. Set
    ``retain_arrays=False`` to omit row memberships and predictions.
    """
    kwargs.pop("return_result", None)
    return cv_risk(rhs, data, response, return_result=True, **kwargs)


#: Test parts created by :func:`holdout`, keyed by data fingerprint, holding
#: the specifications already fitted or scored on each. Scoring a second
#: specification on the same test rows makes the best of those scores
#: optimistic by an unknown amount, so the evaluation assistants say so.
_HOLDOUT_SCORED: dict[str, set[str]] = {}


def _record_holdout_use(rhs, fingerprint):
    """Count one more specification evaluated on a registered test part."""
    specs = _HOLDOUT_SCORED.get(fingerprint)
    if specs is None:
        return
    specs.add(repr(rhs))
    if len(specs) > 1:
        warnings.warn(
            f"this evaluation refits and scores data held out by holdout(): "
            f"{len(specs)} specifications have now used those test rows, so "
            "the best of their scores is optimistic by an amount this count "
            "does not measure, and refitting has already spent them. Fit on "
            "the search part and score the test rows through predictions "
            "instead", stacklevel=3)


def _column_or_array(data, values, what):
    """A column named by string, or the array itself, as float codes."""
    from mboostrs._frame import column_names, read_column

    if isinstance(values, str):
        if values not in column_names(data):
            raise ValueError(
                f"{what}={values!r} names no column of the data; pass a "
                "column name or one value per row")
        values = read_column(data, values).values
    return np.asarray(values, dtype=float)


def holdout(data, fraction=0.2, *, groups=None, time=None, seed=0):
    """Split a frame into a search part and a test part.

    The search part is where specifications are compared and tuned; the test
    part is scored once, at the end, by predicting it from a model fitted on
    the search part. ``fraction`` is the test part's share of the rows, or of
    the groups under ``groups=``.

    The split follows the evaluation task the scores are meant to describe:

    * With neither ``groups`` nor ``time``, rows are dealt at random under
      ``seed``: the test part measures prediction for new observations like
      the sampled ones.
    * ``groups=`` (a column name, or one group code per row) deals whole
      groups to one side: the test part holds groups the search part never
      saw, so it measures prediction in new groups.
    * ``time=`` (a column name, or one value per row) splits on the time
      axis: the test part is the most recent ``fraction`` of the rows, so it
      measures prediction of the future.

    ``groups`` and ``time`` name different tasks and cannot be combined.

    Returns ``(search, test)``, each the same kind of frame as ``data``. The
    test part is remembered by its content: an evaluation assistant
    (:func:`evaluate_specification`, :func:`compare_specifications`,
    :func:`term_report`, :func:`tune_mstop`) later run on exactly those rows
    warns, from the second specification onward, how many have used them.
    A hand-written predict-and-score loop is not intercepted; score the test
    part once and keep the result with the search that produced it.
    """
    from mboostrs._frame import column_names, read_column, take_rows
    from mboostrs.evaluation import data_fingerprint

    n = read_column(data, column_names(data)[0]).values.size
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie strictly between 0 and 1, got {fraction!r}")
    if groups is not None and time is not None:
        raise ValueError(
            "groups= and time= name different evaluation tasks (new groups "
            "against the future); split for one of them")
    rng = np.random.default_rng(seed)
    if groups is not None:
        codes = _column_or_array(data, groups, "groups")
        levels = np.unique(codes)
        if len(levels) < 2:
            raise ValueError(
                "holdout needs at least two groups to put one on each side; "
                f"groups has {len(levels)}")
        n_test = max(1, int(round(len(levels) * fraction)))
        n_test = min(n_test, len(levels) - 1)
        test_levels = set(rng.permutation(levels)[:n_test].tolist())
        test = np.array([code in test_levels for code in codes], dtype=bool)
    elif time is not None:
        times = _column_or_array(data, time, "time")
        order = np.argsort(times, kind="stable")
        n_test = max(1, int(round(n * fraction)))
        n_test = min(n_test, n - 1)
        test = np.zeros(n, dtype=bool)
        test[order[-n_test:]] = True
    else:
        if n < 2:
            raise ValueError("holdout needs at least two rows")
        n_test = max(1, int(round(n * fraction)))
        n_test = min(n_test, n - 1)
        test = np.zeros(n, dtype=bool)
        test[rng.permutation(n)[:n_test]] = True
    if not test.any() or test.all():
        raise ValueError("the split produced an empty side; choose a different fraction")
    search, test_part = take_rows(data, ~test), take_rows(data, test)
    _HOLDOUT_SCORED.setdefault(data_fingerprint(test_part), set())
    return search, test_part


@dataclass(frozen=True)
class MstopTuning:
    """A cross-validated step count, and whether the grid grew to find it."""

    #: The iteration with the lowest aggregated out-of-fold risk.
    mstop: int
    #: Aggregated out-of-fold risk at iterations ``0 .. largest``.
    risk: tuple
    #: The largest iteration of the last grid evaluated.
    largest: int
    #: The largest iteration of the first grid.
    first_largest: int
    #: True when the minimum lies in the boundary band at the end of the last
    #: grid (`mboostrs.cvrisk.BOUNDARY_SHARE` of its length), where the curve
    #: has not been seen to turn.
    at_boundary: bool
    #: Model fits spent on the folds. A grown grid resumes each fold's path,
    #: so the family-score search costs one fit per fold.
    fits: int
    #: The full-data fit of the last grid, when the specification is a
    #: standard model: the same formula, family, data and step length the
    #: sweep cross-validated, at `largest` iterations. Truncating its path
    #: (`fit[:k]`) is the main-effect fit any later step needs, so a caller
    #: that wants the fitted model anyway need not refit. Distributional
    #: specifications store the model refitted at `mstop` (a distributional
    #: path cannot be sliced, so the refit is the one `model` returns). Not
    #: part of the tuning's identity.
    _fit: object = field(default=None, repr=False, compare=False)
    #: True when `_fit` is already the model at `mstop` (distributional
    #: specifications), so `model` returns it without slicing.
    _fit_at_mstop: bool = field(default=False, repr=False, compare=False)
    score: str = "family"
    score_parameter: float | None = None
    full_data_fits: int = 2
    #: Out-of-fold family risk of each fold at the chosen ``mstop``, one value
    #: per fold in fold order. Their mean is the aggregate ``risk[mstop]``.
    fold_scores: tuple = ()
    #: The grid's end after the first fit and after each extension.
    grid_ends: tuple = ()
    #: The patience window the search stopped on, or None for the doubling rule.
    patience: int | None = None

    @property
    def total_fits(self):
        """Fold fits plus the preparation/final fits performed by the tuner."""
        return self.fits + self.full_data_fits

    def __repr__(self) -> str:
        return (
            f"MstopTuning(mstop={self.mstop}, largest={self.largest}, "
            f"extended={self.extended}, at_boundary={self.at_boundary}, "
            f"fits={self.fits})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record. The full risk path is exported only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("risk", "fold_scores"),
            extras={"extended": self.extended, "total_fits": self.total_fits},
        )

    @property
    def extended(self) -> bool:
        return self.largest > self.first_largest

    @property
    def model(self):
        """The full-data fit at the chosen step count, ready to predict.

        For a standard specification the sweep already fitted the model on
        all rows to `largest` iterations; this is that fit cut to `mstop`, so
        ``tune_mstop(...)`` both tunes and fits in one call. For a
        distributional specification the model is refitted at `mstop` (every
        parameter moves by the same step count), and that refit is returned.
        An offset-only fit is returned directly: its constant score path has
        a positive selected grid index but the model has zero actual updates.
        """
        if self._fit is None:
            return None
        # An offset-only fit has no updates to slice, even though the
        # constant score path selects the first positive grid position.
        if self._fit_at_mstop or self._fit.mstop == 0:
            return self._fit
        return self._fit[: self.mstop]

    @property
    def sentence(self) -> str:
        grew = (f"the risk was still falling at {self.first_largest}, so the grid "
                f"was extended to {self.largest}; " if self.extended else "")
        if len(self.risk) == self.largest + 1 and self.risk[0] <= self.risk[self.mstop]:
            return (f"no evaluated positive iteration improves on the offset-only fit; "
                    f"iteration {self.mstop} is the best eligible positive step, not "
                    f"evidence of predictive improvement. Check the specification, "
                    f"family and step length before interpreting the selected model")
        if self.mstop <= 1 and len(self.risk) > 2:
            return (f"the cross-validated risk is lowest at iteration {self.mstop} "
                    f"and rises from there: every boosting step makes held-out "
                    f"predictions worse. Either nothing in the specification "
                    f"predicts the response, or the family or the step length does "
                    f"not suit it and the fit diverges; check the training risk path, "
                    f"a smaller `nu`, or another family before reading any other "
                    f"measurement made at this setting")
        if self.at_boundary and self.patience is not None:
            return grew + (
                f"the cross-validated risk is lowest at {self.mstop} of the "
                f"{self.largest} iterations tried, fewer than the patience of "
                f"{self.patience} steps before the end of the search, so it has "
                f"not been seen to turn: treat it as a floor rather than an "
                f"optimum, and raise `max_mstop` to cross-validate further")
        if self.at_boundary:
            return grew + (
                f"the cross-validated risk is lowest at {self.mstop} of the "
                f"{self.largest} iterations tried, within the last "
                f"{BOUNDARY_SHARE:.0%} of the grid, so it has not been seen to "
                f"turn: treat it as a floor rather than an optimum, and raise "
                f"`mstop` to cross-validate further")
        return grew + (f"the cross-validated risk turns at {self.mstop} of "
                       f"{self.largest} iterations tried")


#: Relative gap between the chosen step count's cross-validated risk and the
#: intercept-only risk (step count 0) below which `tune_mstop` reports a flat
#: curve. The gap is ``|risk(mstop) - risk(0)| / |risk(0)|``.
FLAT_CURVE_TOLERANCE = 0.01


@one_summary
def tune_mstop(formula, data, family=None, *, folds=5, start=100, nu=0.1,
               extensions=None, max_mstop=10_000, seed=0, step="fixed",
               score="family", score_parameter=None, unseen_levels="average",
               fold_local=False, patience=None, fit_budget=None,
               extrapolation="warn", flat_tolerance=FLAT_CURVE_TOLERANCE, **fit_kwargs):
    """Cross-validate the step count, doubling the grid while the risk still falls.
    Usage::

        import mboostrs as mb
        df = mb.datasets.bodyfat()
        tuned = mb.tune_mstop("DEXfat ~ bbs(age, df=4)", df)
        tuned.mstop  # tuned.at_boundary says whether the risk had turned

    The model is fitted once per fold at ``start`` iterations and the
    aggregated out-of-fold risk is read at every iteration of that path. If the
    minimum lies in the boundary band at the end of the grid (the last
    `mboostrs.cvrisk.BOUNDARY_SHARE`, 5%, of its length), the risk had not
    turned: the grid is doubled and the sweep repeated until the minimum falls
    before the band, never past `max_mstop` (the last grid is cut to it). For
    example, a curve lowest at 399 of 400 iterations has not been seen to turn.
    A search that stops at its limit with the minimum still in the band warns,
    naming the limit that applied (`max_mstop` or `extensions`), because its
    `mstop` is where the grid stopped rather than where the data turned.

    A grown grid resumes each fold's path where the previous grid stopped: the
    default family score, a distributional specification, and a response score
    or ``fold_local=True`` on a model that is not Gaussian Gram-route. A
    response-score fold on the Gaussian Gram route keeps no residual to resume
    from, and its general-route continuation costs more per step than the Gram
    refit, so that fold is refitted from zero at each grid, as the refit search
    did.

    The fit count is ``folds + 1`` when every fold resumes: one fit per fold for
    the sweep, and one fit on the full data (the final fit at the chosen grid,
    or for a distributional specification the fit the search's refit arguments
    come from). A Gram-route response fold adds one fit per later grid.
    ``fit_budget`` is checked before any fit against the worst case, which for
    a response score counts every grid up to the limit.

    Parameters
    ----------
    formula : str
        A ``gamboost`` formula.
    folds : int, ndarray or FoldScheme
        A fold count (random k-fold on `seed`), an ``(n, B)`` in-bag weight
        matrix, or a scheme from `mboostrs.folds`, such as
        `grouped_folds` for prediction in new groups.
    start : int
        The first grid's largest iteration.
    extensions : int, optional
        How many times the grid may double; ``None`` (default) doubles until
        the risk turns or ``max_mstop`` is reached.
    max_mstop : int
        The largest step count searched.
    step : {"fixed", "line_search"}
        The step rule (see :func:`~mboostrs.boost_control`). ``"line_search"``
        scales each step by a one-dimensional search, which keeps a Poisson,
        negative binomial or gamma fit of a large response from overflowing
        its inverse link in the first steps.
    fit_budget : int, optional
        Maximum model fits allowed for the first grid, ``folds + 1`` (see the
        note above). The call refuses before any fit when the count is larger.
    **fit_kwargs
        Passed to `gamboost`: ``weights``, ``env``, ``na_action`` and so on.
    score : {"family", "mse", "brier", "log_loss", "pinball", "expectile"}
        Stopping criterion. Response scores fit each fold's recipes on its
        training rows and evaluate every path position on its held-out rows.
        Pinball/expectile require ``score_parameter``. Response-score tuning
        currently requires a plain response column, scalar predictions or
        Multinomial probabilities; transformed-response scales are refused.
    unseen_levels : {"average", "zero_effect", "error"}
        Held-out group policy for response-score paths.
    fold_local : bool, default False
        Also evaluate family loss through fold-local prediction, using the
        same recipe, held-out mask and unseen-level semantics as `cv_risk`.
        Response scores always use this route. The default family path
        retains the native mboost-compatible CV behavior.
    patience : int, optional
        Stop once the aggregated held-out risk has not improved for this many
        iterations, growing the grid only to the minimum plus `patience`
        rather than doubling it. The selected `mstop` equals the full grid's
        unless the risk falls below its minimum again more than `patience`
        steps later. ``None`` (default) uses the doubling rule above. Applies
        to the default family score.
    extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
        How a held-out row outside its fold's training range is predicted, as
        in :meth:`GAMBoostModel.predict`. It applies to the fold-refit
        predictions: ``fold_local=True`` and the response scores. The default
        family path builds every fold's basis from the full data, so no
        held-out row can leave it and the argument has no effect there.
    flat_tolerance : float or None, default 0.01
        On the default family path, warn when the chosen step count's
        cross-validated risk is within this relative gap of the intercept-only
        risk at step 0, that is ``|risk(mstop) - risk(0)| <= flat_tolerance *
        |risk(0)|``. Such a curve is flat: the fit is close to a constant, and
        the warning says so. ``None`` turns the check off.

    Returns
    -------
    MstopTuning
        ``mstop`` and the curve, with ``at_boundary`` true when the risk was
        still falling at the end of the last grid.
    """
    from mboostrs.cvrisk import cv_weights
    from mboostrs.formula import require_response
    from mboostrs.gamboost import gamboost

    require_response(formula, "tune_mstop")
    if extrapolation not in ("warn", "error", "allow", "clip"):
        raise ValueError("extrapolation must be one of 'warn', 'error', 'allow', "
                         f"'clip'; got {extrapolation!r}")
    family = Gaussian() if family is None else family
    start = check_int_at_least("start", start, 1, allow_none=False)
    max_mstop = check_int_at_least("max_mstop", max_mstop, 1, allow_none=False)
    start = min(start, max_mstop)
    extensions = check_int_at_least("extensions", extensions, 0, allow_none=True)
    patience = check_int_at_least("patience", patience, 1, allow_none=True)
    if patience is not None and (score != "family" or fold_local or isinstance(formula, dict)):
        raise NotImplementedError(
            "patience applies to the default family score of a standard "
            "specification; response scores, fold_local and distributional "
            "specifications use the doubling rule")
    # A response-score fold fitted on the Gaussian Gram route is refitted at
    # every later grid, so that route's worst case counts each grid.
    response_route = (score != "family" or fold_local) and not isinstance(formula, dict)
    grids = len(_doubling_ends(start, max_mstop, extensions)) if response_route else 1
    check_fit_budget(
        _tune_mstop_fits(folds, grids=grids), fit_budget, "tune_mstop",
        detail=(f" for a first grid of {start} steps, worst case {grids} grids"
                if grids > 1 else f" for a first grid of {start} steps"),
        remedy="use fewer folds, or raise fit_budget")
    if score == "family" and score_parameter is not None:
        raise ValueError("score_parameter is only used by pinball or expectile loss")
    if score != "family" or fold_local:
        from mboostrs._metric_tuning import tune_response_mstop
        from mboostrs.gamboost import _check_unseen_levels

        if isinstance(formula, dict):
            raise NotImplementedError("distributional tuning currently requires score='family'")
        return tune_response_mstop(
            formula, data, family, folds=folds, start=start, nu=nu,
            extensions=extensions, max_mstop=max_mstop, seed=seed, step=step,
            score=score, score_parameter=score_parameter,
            unseen_levels=_check_unseen_levels(unseen_levels),
            extrapolation=extrapolation, **fit_kwargs)
    if score_parameter is not None:
        raise ValueError("score_parameter is only used by pinball or expectile loss")
    is_lss = hasattr(family, "param_names") and hasattr(family, "n_params")
    if isinstance(formula, dict):
        if not is_lss:
            raise ValueError(
                "a mapping specification names distribution parameters; it "
                f"needs a distributional family, e.g. family=GaussianLSS(), "
                f"not {type(family).__name__}")
        if step != "fixed":
            raise ValueError("a distributional specification is tuned with the fixed step")
        return _tune_mstop_lss(formula, data, family, folds=folds, start=start,
                               nu=nu, extensions=extensions,
                               max_mstop=max_mstop, seed=seed, **fit_kwargs)
    if isinstance(folds, (int, np.integer)):
        response = partition_tilde(formula)[0].strip()
        n = len(np.asarray(_response_values(data, response)))
        matrix = cv_weights(n, type="kfold", B=int(folds), seed=seed)
        scheme = None
    elif isinstance(folds, FoldScheme):
        matrix = np.asarray(folds, dtype=float)
        scheme = folds
    else:
        matrix = np.asarray(folds, dtype=float)
        scheme = None
    if fit_kwargs.get("na_action") == "omit":
        # The skeleton below estimates on complete cases only, so fold
        # weights drawn over the full frame name rows it does not have.
        # Subset every per-row fold field to the rows the fit will keep,
        # preserving fold membership and any judge set.
        n_full = len(np.asarray(_response_values(data, partition_tilde(formula)[0].strip())))
        missing = _missing_rows(formula, data, np.arange(n_full))
        if missing.any():
            kept_rows = np.flatnonzero(~missing)
            matrix = matrix[kept_rows]
            scheme = None if scheme is None else scheme.take(kept_rows)
    n_folds = int(matrix.shape[1])
    first = check_int_at_least("start", start, 1, allow_none=False)
    # The sweep cross-validates from the formula, data and learners alone, so
    # the model it starts from is fitted at one step; the full path is fitted
    # once, after the sweep.
    skeleton = gamboost(formula, data=data, family=family,
                        control=boost_control(1, nu, step=step), **fit_kwargs)
    saved = skeleton._control
    skeleton._control = saved._replace(mstop=int(first))
    try:
        # A FoldScheme keeps its judge set: a rolling-origin fold is judged
        # on its horizon block, not on every zero-weight row.
        result, grid_ends, at_boundary = skeleton.cvrisk(
            folds=scheme if scheme is not None else matrix, _inner=True,
            _extend={"max_mstop": int(max_mstop), "patience": patience,
                     "boundary_share": BOUNDARY_SHARE, "max_extensions": extensions})
    finally:
        skeleton._control = saved
    curve = np.asarray(result.aggregate, dtype=float)
    largest = int(grid_ends[-1])
    from mboostrs._rust import select_score_path

    _aggregate, best = select_score_path(np.ascontiguousarray(curve[None, :]))
    fit = gamboost(formula, data=data, family=family,
                   control=boost_control(largest, nu, step=step), **fit_kwargs)
    _warn_search_ended_falling(at_boundary, largest, max_mstop, extensions)
    _warn_flat_curve(curve, max(1, best), flat_tolerance)
    # One out-of-fold risk per fold at the chosen step count: the spread across
    # these is what the aggregate hides.
    chosen = max(1, best)
    fold_risks = np.asarray(result.risk_matrix, dtype=np.float64)[:, chosen]
    return MstopTuning(mstop=chosen, risk=tuple(float(v) for v in curve),
                       largest=largest, first_largest=first,
                       at_boundary=bool(at_boundary), fits=n_folds, _fit=fit,
                       grid_ends=grid_ends, patience=patience,
                       fold_scores=tuple(float(v) for v in fold_risks))


def _tune_mstop_fits(folds, grids=1) -> int:
    """Model fits `tune_mstop` spends, counted before any fit.

    Each fold is fitted once and its path resumed from grid to grid, and one
    fit on the full data is added, so the count is ``folds + 1``. A response
    score fold on the Gaussian Gram route is refitted at every grid instead;
    `grids` is then the number of grids the search may evaluate, and the count
    is the worst case ``folds * grids + 1``.
    """
    if isinstance(folds, (int, np.integer)):
        count = int(folds)
    else:
        count = int(np.asarray(folds, dtype=float).shape[1])
    return count * int(grids) + 1


def cvrisk_at(model, folds, mstop):
    """`model.cvrisk(folds)` along a path of `mstop` steps, whatever step count
    `model` itself was fitted to: the cross-validation refits every fold from
    the model's formula, data and learners, and reads nothing of its path."""
    saved = model._control
    model._control = saved._replace(mstop=int(mstop))
    try:
        return model.cvrisk(folds=folds, _inner=True)
    finally:
        model._control = saved


def _grid_growths(start, max_mstop, extensions=None) -> int:
    """The most times a grid that starts at `start` steps can grow: it doubles,
    is cut to `max_mstop`, and stops at `extensions` growths when that is set."""
    growths, largest = 0, int(start)
    while largest < int(max_mstop) and (extensions is None or growths < extensions):
        largest = min(2 * largest, int(max_mstop))
        growths += 1
    return growths


def _doubling_ends(start, max_mstop, extensions=None):
    """The grid ends a doubling search evaluates, in order.

    The first end is `start`. Each next end is twice the last, cut to
    `max_mstop`, and the list stops at `max_mstop` or after `extensions`
    growths, whichever comes first.
    """
    ends = [int(start)]
    while ends[-1] < int(max_mstop) and (extensions is None or len(ends) - 1 < extensions):
        ends.append(min(2 * ends[-1], int(max_mstop)))
    return ends


def _check_max_mstop(mstop, max_mstop):
    """The step cap of ``mstop="cv"`` tuning, or None for a fixed `mstop`.

    The cap is the largest grid the inner search may grow to. The first grid
    is `_MAX_TUNED_MSTOP`, or the cap when that is smaller. It is an error to
    give a cap with a fixed `mstop`, because the cap would be ignored.
    """
    if mstop != "cv":
        if max_mstop is not None:
            raise ValueError(
                "max_mstop caps the inner search of mstop='cv' and has no effect "
                f"with a fixed mstop={mstop}; pass mstop='cv' or drop max_mstop")
        return None
    if max_mstop is None:
        return int(_MAX_TUNED_MSTOP)
    return check_int_at_least("max_mstop", max_mstop, 1, allow_none=False)


def _warn_flat_curve(curve, mstop, tolerance):
    """Warn that the chosen step count barely improves on the constant model.

    `curve` is the aggregated out-of-fold risk at steps ``0 .. largest``, so
    ``curve[0]`` is the intercept-only risk. When the chosen step count lies
    within `tolerance` of it, the fitted learners add little the cross-validated
    risk can see, and the fit is close to a constant.
    """
    if tolerance is None or mstop >= len(curve):
        return
    intercept, chosen = float(curve[0]), float(curve[mstop])
    if not np.isfinite(intercept) or intercept == 0.0 or not np.isfinite(chosen):
        return
    gap = abs(chosen - intercept) / abs(intercept)
    if gap > tolerance:
        return
    warnings.warn(
        f"tune_mstop: the chosen mstop={mstop} has a cross-validated risk of "
        f"{chosen:.6g}, within {tolerance:.1%} of the intercept-only risk of "
        f"{intercept:.6g} (mstop=0; relative gap {gap:.2%}). The curve is flat and "
        "the fit is close to a constant: the terms add little that the "
        "out-of-fold risk can see at this sample size. Compare with the constant "
        "model before reporting the fitted effects. Pass flat_tolerance=None to "
        "silence this check, or a different tolerance.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_search_ended_falling(at_boundary, largest, max_mstop, extensions=None):
    """Warn that a search stopped at its limit with the risk not yet turned.

    The limit named in the warning is the one that applied: `max_mstop` when
    the grid reached it, otherwise the `extensions` count.
    """
    if not at_boundary:
        return
    if largest >= max_mstop:
        limit = f"max_mstop={max_mstop}"
        remedy = f"raise max_mstop above {max_mstop}"
    else:
        limit = f"extensions={extensions}"
        remedy = "set extensions=None or a larger count"
    warnings.warn(
        f"the cross-validated risk was still falling at {largest} steps, when the "
        f"search reached its limit ({limit}), so mstop={largest} is where the grid "
        f"stopped rather than where the risk turns. To search further, {remedy}, "
        f"or raise nu to take larger steps." + guide_hint("fit-and-tune"),
        RuntimeWarning,
        stacklevel=3,
    )


def _inner_boundary_warning(call, cap, flagged, folds, label=None):
    """The warning for `flagged` outer folds whose inner mstop search stopped at
    the cap `cap` with the risk still falling there. It names the call that
    ran the search and the argument that raises the cap."""
    where = f"{call} for specification {label!r}" if label is not None else call
    return (
        f"{where}: the inner mstop search stopped at max_mstop={cap} in "
        f"{flagged} of {folds} outer folds, with the cross-validated risk still "
        f"falling there, so those folds' mstop values are floors rather than "
        f"optima. To search further, pass a larger max_mstop (now {cap})."
    )


def _tune_mstop_lss(formulas, data, family, *, folds, start, nu, extensions,
                    max_mstop, seed, **fit_kwargs):
    """`tune_mstop` for a distributional specification.

    The same doubling grid, read off the LSS diagonal: every parameter moves
    by the same step count. The core fits each fold once and resumes its path
    from one grid to the next, so a search that doubles k times still costs one
    fit per fold. The one full-data fit supplies the refit arguments of the
    specification. `formulas` is the mapping `gamboostLSS` itself takes, full
    ``y ~ ...`` right-hand sides sharing one response.
    """
    from mboostrs.cvrisk import cv_weights
    from mboostrs.formula import parse_response
    from mboostrs.gamlss import gamboostLSS, make_grid_lss

    formulas = {name: str(f) for name, f in formulas.items()}
    responses = {parse_response(f) for f in formulas.values()}
    if len(responses) != 1 or not all(isinstance(r, str) for r in responses):
        raise ValueError(
            "every parameter formula must share one plain response")
    response = next(iter(responses))
    if isinstance(folds, (int, np.integer)):
        n = len(np.asarray(_response_values(data, response)))
        matrix = cv_weights(n, type="kfold", B=int(folds), seed=seed)
    elif isinstance(folds, FoldScheme):
        # Passed whole to the search, so its judge mask is honoured.
        matrix = folds
    else:
        matrix = np.asarray(folds, dtype=float)
    n_folds = int(matrix.shape[1])
    first = check_int_at_least("start", start, 1, allow_none=False)
    ends = _doubling_ends(first, max_mstop, extensions)
    grids = [make_grid_lss(end, length_out=end + 1) for end in ends]
    # The sweep needs only the specification, so the model it starts from is
    # fitted at one step; the full path is fitted once, at the chosen count.
    fit = gamboostLSS(formulas, data=data, family=family,
                      control=boost_control(1, nu), **fit_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        found = fit._cvrisk_search(matrix, ends=ends, grids=grids,
                                   boundary_share=BOUNDARY_SHARE)
    used = found["ends"]
    largest = int(used[-1])
    _warn_search_ended_falling(found["at_boundary"], largest, max_mstop, extensions)
    chosen = max(1, int(found["mstop"]))
    # An LSS path cannot be sliced, so the model is refitted at the chosen
    # count, as `mstop<-` does for an LSS fit.
    final = fit.update(mstop=chosen)
    return MstopTuning(mstop=chosen,
                       risk=tuple(float(v) for v in found["curve"]),
                       largest=largest, first_largest=first,
                       at_boundary=bool(found["at_boundary"]), fits=n_folds,
                       full_data_fits=1, grid_ends=tuple(int(e) for e in used),
                       _fit=final, _fit_at_mstop=True)


#: The first grid of the inner tuning searched when ``mstop="cv"``, and the
#: default step cap (``max_mstop``) of that search. Large enough that the inner
#: curve usually turns; when it does not, the search warns at the cap.
_MAX_TUNED_MSTOP = 500


def _inner_folds(folds, fit_mask, seed, training_weights=None):
    """Inner tuning folds faithful to the outer scheme's structure, or None.

    Random inner folds over a grouped or time-ordered outer scheme re-admit
    the exact leakage the scheme exists to prevent: the tuning would select
    the mstop that wins by seeing the rows the outer scheme holds apart. A
    grouped outer scheme gets group-whole inner folds, a stratified one
    preserved proportions, and an ordered one rolling-origin folds over the
    training prefix -- the same class of resampling, drawn from the outer
    fold's training rows only. A random outer scheme (a fold count, a bare
    matrix, k-fold, subsampling, bootstrap, repeated) has no structure for
    random inner folds to violate, so None keeps the ordinary tuning.
    """
    if not isinstance(folds, FoldScheme):
        return None
    train = np.flatnonzero(fit_mask)
    stratified = None if folds.strata is None else np.asarray(folds.strata)[train]
    groups = None if folds.groups is None else np.asarray(folds.groups)[train]
    m = train.size
    if folds.ordered:
        if m < 15:
            raise ValueError("ordered inner tuning needs at least 15 training rows; "
                             "use fixed mstop or a larger training window")
        horizon = max(1, m // 5)
        initial = m - 3 * horizon
        if initial < 1:
            initial = 1
        # The inner scheme keeps the outer scheme's time order and embargo, so
        # the training rows are ordered by their timestamps, not their positions.
        train_times = None if folds.times is None else np.asarray(folds.times)[train]
        return rolling_folds(m, B=3, initial=initial, horizon=horizon, weights=training_weights,
                             time=train_times, embargo=folds.embargo)
    if stratified is not None:
        return stratified_folds(stratified, B=3, seed=seed, weights=training_weights)
    if groups is not None:
        _, groups = np.unique(groups, return_inverse=True)
        return grouped_folds(groups, B=min(3, len(np.unique(groups))), seed=seed, weights=training_weights)
    return None


#: The inner search of one outer fold: the step count it chose, the largest
#: grid it reached, whether that grid's end was still in the boundary band,
#: and the fits it spent.
_InnerSearch = namedtuple("_InnerSearch", "mstop largest at_boundary fits")


def _tuned(model, seed, inner=None, *, start, max_mstop):
    """The model sliced to the `mstop` an inner cross-validation prefers, and
    the search that chose it.

    The inner folds are drawn from the outer fold's *training* rows only, so
    the outer score never sees a row the tuning used. When the outer scheme
    is grouped, stratified or ordered, `inner` reuses that structure so the
    tuning cannot pick an `mstop` through the same leakage the outer scheme
    prevents; an ordinary random outer scheme keeps random inner folds.

    The search starts at `start` steps and grows while its optimum is in the
    boundary band, up to `max_mstop`, by the same rule as `tune_mstop`. The
    model must have been fitted to at least `max_mstop` steps, since the
    returned model is sliced from its path.
    """
    import warnings as _warnings

    from mboostrs._rust import select_score_path

    if hasattr(model, "_params"):
        return _tuned_lss(model, seed, inner, start=start, max_mstop=max_mstop)
    saved = model._control
    model._control = saved._replace(mstop=int(start))
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            # The grown grid is resumed fold by fold, so the search costs one
            # fit per inner fold however far it grows.
            result, grid_ends, at_boundary = model.cvrisk(
                folds=inner, type="kfold", B=5, seed=seed, _inner=True,
                _extend={"max_mstop": int(max_mstop), "patience": None,
                         "boundary_share": BOUNDARY_SHARE, "max_extensions": None})
    finally:
        model._control = saved
    curve = np.asarray(result.aggregate, dtype=float)
    _aggregate, best = select_score_path(np.ascontiguousarray(curve[None, :]))
    optimal = max(1, int(best))
    fits = inner.n_folds if inner is not None else 5
    return model[:optimal], _InnerSearch(optimal, int(grid_ends[-1]),
                                         bool(at_boundary), fits)


def _tuned_lss(model, seed, inner, *, start, max_mstop):
    """`_tuned` for a distributional model.

    Every parameter takes the same step count, read from the diagonal grid.
    The grid is grown while its best point is in the boundary band. Each grid
    refits every inner fold, and the outer fit already covers `max_mstop`, so
    no fit of the full data is repeated.
    """
    from mboostrs.gamlss import diagonal_grid

    n_params = len(model.param_names)
    fits_per_grid = inner.n_folds if inner is not None else 5
    largest, grids_run, fits = int(start), 0, 0
    while True:
        grid = diagonal_grid(largest, n_params, length_out=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tuned = model.cvrisk(folds=inner, B=5, seed=seed, grid=grid)
        curve = np.asarray(tuned.aggregate, dtype=float)
        finite = np.isfinite(curve)
        if not finite.any():
            raise ValueError("the cross-validated risk is not finite at any iteration")
        point = grid[int(np.argmin(np.where(finite, curve, np.inf)))]
        best = int(max(point)) if isinstance(point, (tuple, list, np.ndarray)) else int(point)
        at_boundary = is_at_boundary(best, largest)
        grids_run += 1
        fits += fits_per_grid
        if not at_boundary or largest >= max_mstop:
            break
        largest = min(2 * largest, int(max_mstop))
    optimal = max(1, best)
    return model[:optimal], _InnerSearch(optimal, largest, bool(at_boundary), fits)


def _missing_rows(formula, data, index):
    """Which of `index`'s rows carry a missing value in a formula column."""
    from mboostrs._frame import take_rows
    from mboostrs.formula import apply_na_action

    subset = take_rows(data, index)
    _kept, dropped = apply_na_action(formula, subset, "omit")
    mask = np.zeros(len(index), dtype=bool)
    mask[np.asarray(dropped, dtype=int)] = True
    return mask


def _specification_name(rhs):
    """The row name of an unnamed candidate: the right-hand side itself, or for
    a distributional mapping ``"mu: bbs(x); sigma: 1"``."""
    if isinstance(rhs, dict):
        return "; ".join(f"{param}: {part}" for param, part in rhs.items())
    return rhs


def _lss_formulas(rhs, response, family):
    """Per-parameter formulas for one distributional candidate.

    A mapping names each parameter's right-hand side; a plain string is the
    right-hand side of every parameter, which is what a single-formula
    `gamboostLSS` call means. The keys are checked against the family's own
    `param_names`, so `{"mu": ..., "scale": ...}` is refused at the call
    instead of fitting a parameter that does not exist.
    """
    names = tuple(family.param_names)
    if isinstance(rhs, dict):
        missing = [name for name in names if name not in rhs]
        extra = [name for name in rhs if name not in names]
        if missing or extra:
            raise ValueError(
                f"the candidate names parameters {sorted(rhs)}, but "
                f"{type(family).__name__} takes {list(names)}")
        return {name: f"{quote_response(response)} ~ {rhs[name]}" for name in names}
    return {name: f"{quote_response(response)} ~ {rhs}" for name in names}


def _any_missing_rows(formulas, data, index):
    """Rows of `index` missing in at least one of several formulas."""
    mask = np.zeros(len(index), dtype=bool)
    for formula in formulas:
        mask |= _missing_rows(formula, data, index)
    return mask


def _unseen_level_rows(model, train, held):
    """Held-out rows whose ``brandom`` level the fold's training rows lack."""
    from mboostrs.learners import Brandom

    def raw(frame, name):
        values = frame[name]
        return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)

    # A distributional model holds one learner list per parameter rather than
    # the flat `gamboost` attribute, so both shapes are walked.
    specs = list(getattr(model, "_learners_spec", None) or ())
    for parameter in getattr(model, "_params", None) or ():
        if isinstance(parameter, dict):
            specs.extend(parameter.get("learners") or ())

    unseen = None
    for spec in specs:
        if not isinstance(spec, Brandom):
            continue
        # Set membership rather than np.isin: a label column need not sort.
        seen = set(raw(train, spec.name).tolist())
        rows = np.array([value not in seen for value in raw(held, spec.name).tolist()], dtype=bool)
        unseen = rows if unseen is None else unseen | rows
    return 0 if unseen is None else int(np.sum(unseen))


def _response_values(data, response):
    """The response column, whatever the frame is."""
    from mboostrs._frame import read_column

    return read_column(data, response).values


def _multinomial_response(data, response):
    """Integer response codes and their stable sorted labels."""
    raw = data[response]
    if hasattr(raw, "to_numpy"):
        raw = raw.to_numpy()
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Multinomial response must be a non-empty one-dimensional column")
    missing = np.array([
        value is None
        or (isinstance(value, (float, np.floating)) and np.isnan(value))
        for value in raw.tolist()
    ])
    if np.any(missing):
        raise ValueError("Multinomial response contains missing class labels")
    try:
        labels = tuple(np.unique(raw).tolist())
    except TypeError as exc:
        raise ValueError(
            "Multinomial class labels must have one mutually orderable type"
        ) from exc
    if len(labels) < 3:
        raise ValueError("Multinomial evaluation requires at least three observed classes")
    index = {label: position for position, label in enumerate(labels)}
    return np.asarray([index[value] for value in raw.tolist()], dtype=np.float64), labels


@dataclass
class TermReport:
    """Held-out contribution and alternative-shape scores for each term."""

    formula: str
    response: str
    baseline: float
    score: str = "family"
    #: Model fits spent producing this report: the baseline refit plus one
    #: per removal and shape candidate, each on every fold. A search that
    #: spends its autospec `budget=` and then runs reports outside it can
    #: total its real cost from this field, as the comparison rows and
    #: `MstopTuning.fits` already let it total theirs.
    fits: int = 0
    contributions: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    #: Distinct values per column of the data, so a suggestion can build the
    #: interaction term a column's type calls for.
    levels: dict = field(default_factory=dict)
    #: Columns the specification fits as groups (``brandom`` terms).
    groups: tuple = ()
    #: Candidate specifications the report compared: the full specification,
    #: each one with a term removed, and each shape alternative.
    candidates: int = 0

    def __repr__(self) -> str:      # pragma: no cover - display only
        score = ("out-of-fold MSE" if self.score == "mse"
                 else "out-of-fold family risk")
        lines = [f"TermReport({quote_response(self.response)} ~ ..., {score} "
                 f"{self.baseline:.4g}; {self.candidates} candidate specifications)",
                 "  what each term earns by being there:"]
        for row in self.contributions:
            note = ("  (it predicts the average group on these folds, so this is"
                    " what removing it does to the other terms)"
                    if row.get("measurable", True) is False else "")
            lines.append(f"    {row['term']:32s} {row['earns']:+12.4g}"
                         f" (se {row.get('se', float('nan')):.2g}){note}")
        if self.shapes:
            lines.append("  what each term would earn in another shape:")
            for row in self.shapes:
                lines.append(f"    {row['term']:32s} best: {row['best']:6s} "
                             f"({row['gain over current']:+.4g}, "
                             f"se {row.get('se', float('nan')):.2g})")
        return "\n".join(lines)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of what each term is worth, and in what shape."""
        return export_result(self, include_arrays=include_arrays)


_TermCandidate = namedtuple("_TermCandidate", "kind name term shape group spec")


def _term_report_setup(formula, data, response, family, score):
    """Parse term_report's specification and check it against the family.

    Returns ``(distributional, parts, rhs, formula_text, response)``. `parts`
    maps each distribution parameter (in the family's order) to its terms; a
    standard specification has the single key None. `rhs` is the standard
    right-hand side as written, and `formula_text` is what the report names.
    """
    is_lss = hasattr(family, "param_names") and hasattr(family, "pdf")
    if isinstance(formula, dict):
        if not is_lss:
            raise ValueError(
                "a mapping specification names distribution parameters; it "
                f"needs a distributional family, e.g. family=GaussianLSS(), "
                f"not {type(family).__name__}")
        if score != "family":
            raise ValueError(
                "a distributional specification is scored by its own joint "
                "proper score (score='family'); the single-quantity scores "
                "name a different target")
        per_param_rhs = {name: str(rhs) for name, rhs in formula.items()}
        unknown = [name for name in per_param_rhs
                   if name not in tuple(family.param_names)]
        if unknown:
            raise ValueError(
                f"the specification names parameters {sorted(per_param_rhs)}, "
                f"but {type(family).__name__} takes {list(family.param_names)}")
        missing = [name for name in family.param_names
                   if name not in per_param_rhs]
        if missing:
            raise ValueError(
                f"the specification omits parameters {missing} that "
                f"{type(family).__name__} fits; supply a right-hand side for "
                f"every parameter")
        if response is None:
            raise ValueError("response must be given with a mapping "
                             "specification")
        parts = {name: split_terms(per_param_rhs[name])
                 for name in family.param_names}
        formula_text = "; ".join(f"{name}: {quote_response(response)} ~ {rhs}"
                                 for name, rhs in per_param_rhs.items())
        return True, parts, None, formula_text, response

    if "~" in formula:
        left, _, rhs = partition_tilde(formula)
        response = response or left.strip()
    else:
        rhs = formula
    if response is None:
        raise ValueError("response must be given when formula has no '~'")
    return (False, {None: split_terms(rhs)}, rhs, f"{quote_response(response)} ~ {rhs}",
            response)


def _term_report_render(parts):
    """The candidate specification a parts mapping stands for: one string for
    a standard model, one right-hand side per parameter for a distributional one."""
    if None in parts:
        return " + ".join(parts[None])
    return {name: " + ".join(terms) for name, terms in parts.items()}


def _term_report_plan(parts, data, *, shapes, df):
    """Every specification term_report scores, baseline first, in scoring order.

    A candidate is the full specification (``baseline``), the specification
    with one term removed (``without``), or with one term replaced by one
    shape (``shape``). A removal that would leave a standard model empty, or
    a distributional parameter with nothing but its intercept, is not a
    candidate; a term with no other shape to compare, or whose column is
    binary or constant, gets none. Shape candidates of one term share a
    `group` id. Only the data's column names and values are read here.
    """
    distributional = None not in parts
    plan = [_TermCandidate("baseline", None, None, None, None,
                           _term_report_render(parts))]
    for name in parts:
        for term in parts[name]:
            kept = [t for t in parts[name] if t != term]
            if not kept:
                if not distributional or term.strip() == "1":
                    continue
                kept = ["1"]
            without = dict(parts)
            without[name] = kept
            plan.append(_TermCandidate("without", name, term, None, None,
                                       _term_report_render(without)))
    if shapes:
        from mboostrs._frame import column_names, read_column

        available = set(column_names(data))
        group = 0
        for name in parts:
            for term in parts[name]:
                variable = term_variable(term)
                if variable is None or variable not in available:
                    continue
                column = read_column(data, variable)
                values = column.values
                if len(np.unique(values[~np.isnan(values)])) <= 2:
                    continue
                candidates = _shape_candidates(column)
                if len(candidates) == 1 and candidates[0] == _shape_of(term):
                    # Nothing to compare it against; reporting a zero gain
                    # would read as "no better shape found" rather than "no
                    # other shape applies".
                    continue
                for shape in candidates:
                    swapped = [(_as_shape(t, shape, df) if t == term else t)
                               for t in parts[name]]
                    changed = dict(parts)
                    changed[name] = swapped
                    plan.append(_TermCandidate("shape", name, term, shape, group,
                                               _term_report_render(changed)))
                group += 1
    return plan


def _term_report_plan_for(formula, data, response, family, *, folds, mstop,
                          seed, weights, na_action, shapes, df, score):
    """Parse, plan and count a term_report call without fitting anything.

    Returns ``(setup, plan, fit_count)``, where `setup` is what
    `_term_report_setup` returns. term_report and `estimate_cost` both read the
    count from here, so a refusal and an estimate cannot disagree.
    """
    setup = _term_report_setup(formula, data, response, family, score)
    distributional, parts, _rhs, _text, response = setup
    plan = _term_report_plan(parts, data, shapes=shapes, df=df)
    n = len(np.asarray(_response_values(data, response)))
    fit_count = 0
    for candidate in plan:
        if distributional:
            formulas = list(_lss_formulas(candidate.spec, response, family).values())
        else:
            formulas = [f"{quote_response(response)} ~ {candidate.spec}"]
        fit_count += _planned_fit_count(
            n, folds, seed, weights, mstop, data=data, formulas=formulas,
            na_action=na_action)
    return setup, plan, fit_count


@one_summary
def term_report(formula, data, response=None, family=None, folds=5, mstop=500,
                nu=0.1, seed=0, shapes=True, df=4.0, weights=None,
                score="family", na_action="error", fit_budget=None,
                extrapolation="warn"):
    """Refit a specification without each term, and each term in other shapes.

    The comparison uses shared folds for all candidate specifications.

    **Term contribution.** The model is refitted without each term. The
    change in out-of-fold error estimates the term's predictive contribution
    under this fold design; a negative removal difference favors excluding it.

    **Alternative shapes.** Each term on a column with more than two
    distinct values is refitted as `bols`, `bbs` and `btree` in turn, the rest
    of the specification held fixed.

    The cost is one cross-validation per candidate specification: the full
    specification, each specification with one term removed, and each term in
    each other shape. A cross-validation costs `folds` fits (more with
    ``mstop="cv"``), so the call costs ``folds x (1 + n_terms + n_shapes)``
    fits, where ``n_shapes`` counts the shapes tried for every term that has
    one. ``fit_budget``, when given, refuses the call before any fit if that
    count is larger; `estimate_cost` returns the same count. The call is a
    tool for a specification of tens of terms, not thousands.

    Every gain carries ``se``, the standard error of the paired difference in
    fold scores between the two specifications. Both are scored on the same
    folds, so the fold-to-fold variation they share cancels; a gain within two
    standard errors of zero is not distinguishable from noise on these folds.

    A group term (``brandom``) on folds that hold its groups out is reported
    with ``measurable=False``. Every held-out row then belongs to a group the
    fold's fit never saw, so the term predicts the average group for it, and
    none of the score is the term's own prediction. What the row reports is
    still what removing the term costs: fitted without it, the other terms
    take up the variation it was absorbing, which changes what they predict
    for a new group. What these folds cannot say is whether the groups differ,
    which shows on folds that keep groups together.

    ``folds``, ``weights``, ``score``, ``na_action`` and ``mstop="cv"`` are
    passed to `cv_risk` and mean what they mean there: pass a
    `mboostrs.folds` scheme to score on grouped or time-ordered validation
    rather than a random split.

    ``extrapolation`` is passed to the held-out predictions of each fold, as
    in `cv_risk`: ``"warn"`` (default) reports the fold rows outside their
    training range in one warning, ``"clip"`` holds the spline flat at the
    range edge, ``"error"`` refuses, and ``"allow"`` predicts silently.
    """
    fit_budget = check_int_at_least("fit_budget", fit_budget, 1, allow_none=True)
    setup, plan, needed = _term_report_plan_for(
        formula, data, response, family, folds=folds, mstop=mstop, seed=seed,
        weights=weights, na_action=na_action, shapes=shapes, df=df, score=score)
    distributional, parts, _rhs, formula_text, response = setup
    check_fit_budget(
        needed, fit_budget, "term_report",
        detail=f" for {len(plan)} candidate specifications",
        remedy="use fewer folds, a fixed mstop, or shapes=False")

    scoring = {"weights": weights, "score": score, "na_action": na_action,
               "extrapolation": extrapolation}
    fits_spent = 0
    outcomes = []
    for candidate in plan:
        result = cv_risk(candidate.spec, data, response, family, folds, mstop, nu,
                         seed, return_result=True, retain_arrays=False, **scoring)
        fits_spent += result.fit_count
        outcomes.append((result.score, np.asarray(result.fold_scores, dtype=float)))
    baseline, baseline_folds = outcomes[0]

    report = TermReport(
        formula=formula_text, response=response, baseline=baseline, score=score,
        levels=_column_levels(data),
        groups=tuple(term_variable(t) for name in parts for t in parts[name]
                     if t.strip().startswith("brandom(")),
        candidates=len(plan))

    for index, candidate in enumerate(plan):
        if candidate.kind != "without":
            continue
        risk, fold_scores = outcomes[index]
        row = {}
        if distributional:
            row["parameter"] = candidate.name
        row.update({"term": candidate.term,
                    "variable": term_variable(candidate.term),
                    "without it": risk, "earns": risk - baseline,
                    "se": _paired_se(fold_scores, baseline_folds, _partitions(folds)),
                    "measurable": True})
        if candidate.term.strip().startswith("brandom(") and row["variable"] is not None:
            share = _unseen_share(data, row["variable"], folds, seed, weights)
            if share >= _UNMEASURABLE_UNSEEN_SHARE:
                row["measurable"] = False
                where = f" of {candidate.name}" if distributional else ""
                row["note"] = (
                    f"the folds hold every group of {row['variable']} out, so "
                    f"this term predicts the average group for every held-out "
                    f"row; what removing it changes is how the other terms{where} "
                    f"are fitted, which is the number reported here")
        report.contributions.append(row)
    report.contributions.sort(key=lambda r: r["earns"], reverse=True)

    # Shape candidates of one term share a group id; collect them in plan order.
    shape_groups = {}
    for index, candidate in enumerate(plan):
        if candidate.kind == "shape":
            shape_groups.setdefault(candidate.group, []).append(index)
    for indices in shape_groups.values():
        first = plan[indices[0]]
        scored = {plan[i].shape: outcomes[i][0] for i in indices}
        scored_folds = {plan[i].shape: outcomes[i][1] for i in indices}
        best = min(scored, key=scored.get)
        row = {}
        if distributional:
            row["parameter"] = first.name
        row.update({"term": first.term, "variable": term_variable(first.term),
                    "current": _shape_of(first.term), "best": best,
                    "gain over current": baseline - scored[best],
                    "se": _paired_se(scored_folds[best], baseline_folds,
                                     _partitions(folds)),
                    **{f"as {s}": v for s, v in scored.items()}})
        report.shapes.append(row)
    report.shapes.sort(key=lambda r: r["gain over current"], reverse=True)
    report.fits = fits_spent
    return report


#: A group term is not measurable on folds where at least this share of held-out
#: rows belongs to a group missing from the fold's training rows.
_UNMEASURABLE_UNSEEN_SHARE = 0.99


def _unseen_share(data, column, folds, seed, weights):
    """Share of held-out rows whose level of `column` the fold's fit never saw."""
    from mboostrs._frame import read_column

    try:
        values = read_column(data, column).values
        splits = _folds(len(values), folds, seed, weights)
    except (KeyError, TypeError, ValueError):
        return 0.0
    held_total = unseen = 0
    for fit_mask, held_mask, _ in splits:
        seen = set(values[fit_mask].tolist())
        held = values[held_mask].tolist()
        held_total += len(held)
        unseen += sum(value not in seen for value in held)
    return unseen / held_total if held_total else 0.0


def _spanned_drop_note(reference_rhs, rhs, change, change_se):
    """The note of a row whose specification drops a main effect a tensor spans.

    A term of the reference that `rhs` lacks is named when every column it reads
    is read by a ``%X%`` tensor that `rhs` keeps. When the row's change and its
    standard error are both exactly zero, the note says that dropping the term
    does not change the fit; a row that does change gets only the spanning
    clause. Returns None when no dropped term is spanned.
    """
    # A distributional specification is a mapping of parameter to formula, not a
    # single right-hand side, so only mean-model strings are read.
    if not (isinstance(reference_rhs, str) and isinstance(rhs, str)) or reference_rhs == rhs:
        return None
    kept = split_terms(rhs)
    for term in split_terms(reference_rhs):
        if term in kept:
            continue
        columns = set(term_variables(term))
        if not columns:
            continue
        for tensor in kept:
            if "%X%" in tensor and columns <= set(term_variables(tensor)):
                clause = f"the term's columns are spanned by {tensor}"
                if change == 0 and change_se == 0:
                    return f"{clause}; dropping it does not change the fit"
                return clause
    return None


def _paired_se(first, second, partitions=1):
    """Standard error of the mean paired difference between two fold-score vectors.

    With ``partitions`` greater than one, the vectors hold that many partitions
    of the rows stacked in order, as `repeated_kfold_folds` builds them. Folds
    of different partitions reuse the same rows, so a standard error over every
    column shrinks as partitions are added, much faster than the uncertainty
    about new data does. The variance of the differences is computed within
    each partition, as for one k-fold split, and averaged over the partitions.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.size < 2:
        return float("nan")
    difference = first - second
    partitions = int(partitions or 1)
    if (partitions > 1 and difference.size % partitions == 0
            and difference.size // partitions >= 2):
        within = difference.reshape(partitions, -1)
        variance = np.mean(np.var(within, axis=1, ddof=1))
        return float(np.sqrt(variance / within.shape[1]))
    return float(np.std(difference, ddof=1) / np.sqrt(difference.size))


def _partitions(folds):
    """How many stacked partitions a fold scheme holds; one for anything else."""
    return int(getattr(folds, "repeats", 1) or 1)


def _column_levels(data):
    """Distinct non-missing values per column, for choosing interaction terms."""
    from mboostrs._frame import column_names, read_column

    levels = {}
    for name in column_names(data):
        try:
            levels[name] = read_column(data, name).n_levels
        except (TypeError, ValueError):
            continue
    return levels


#: Most levels a factor may have for two factors to be joined as a ``bols``
#: tensor (one coefficient per pair of levels) rather than as ``brandom`` groups.
FACTOR_TENSOR_MAX_LEVELS = 10


def pair_term_for(left, right, *, levels=None, groups=(), categorical=(),
                  transforms=None, data=None):
    """The learner that carries an interaction between two columns.

    Two continuous columns get a tensor-product spline surface. A two-valued
    column is an indicator, so a continuous partner gets a smooth that switches
    on with it, ``bbs(x, by=b)``, and the term fits how the effect of ``x``
    differs between the two values. A group column (a ``brandom`` term, or any
    column in `groups`) gets a random slope, ``brandom(g, by=x)``: one
    shrunken slope on ``x`` per group. ``bbs(x, by=g)`` is not used for a
    multi-level group: on a numeric group code, ``by=`` multiplies the smooth
    by the code's value, which is an arbitrary label. (On a column of
    categorical dtype, ``by=`` instead expands into one smooth per level; the
    random slope is the term that means the same thing for both encodings.)
    Two label columns with at most `FACTOR_TENSOR_MAX_LEVELS` levels each get
    ``bols(a, by=b)``: one coefficient per combination of their levels, fitted
    as fixed effects (the cell design of the pair). A group column with more levels is a
    ``brandom`` term instead; a group paired with a two-valued column of labels
    (`categorical`) gets ``brandom(g) %X% brandom(b)``, since a label column is
    not a number a slope could multiply. `transforms` maps a column to the scale its main term
    uses (``{"x": "log"}``); its smooth in the pair term uses the same scale,
    so the pair does not extrapolate on a raw scale the main term avoided.

    `levels`, `groups` and `categorical` describe the two columns. Columns
    omitted from these arguments are assumed continuous. Pass the frame as
    ``data=`` to infer level counts and identify string, enum, categorical
    and boolean columns as labels. Columns with more than two levels are
    also classified as groups. With ``data=None``, the supplied descriptions
    determine the term.
    """
    transforms = transforms or {}

    def smooth(column):
        kind = transforms.get(column)
        return f"{kind}({quote_name(column)})" if kind else quote_name(column)

    levels = dict(levels or {})
    groups = set(groups)
    categorical = set(categorical)
    if data is not None:
        from mboostrs._frame import read_column

        for column_name in (left, right):
            if column_name in levels:
                continue
            column = read_column(data, column_name)
            levels[column_name] = column.n_levels
            if column.kind in ("categorical", "boolean"):
                categorical.add(column_name)
                if column.n_levels > 2:
                    groups.add(column_name)
    left_binary = levels.get(left, 3) <= 2
    right_binary = levels.get(right, 3) <= 2
    left_group = left in groups
    right_group = right in groups
    if left_binary and right_binary:
        return f"bols({quote_name(left)}, by={quote_name(right)})"
    if (left in categorical and right in categorical
            and levels.get(left, FACTOR_TENSOR_MAX_LEVELS + 1) <= FACTOR_TENSOR_MAX_LEVELS
            and levels.get(right, FACTOR_TENSOR_MAX_LEVELS + 1) <= FACTOR_TENSOR_MAX_LEVELS):
        # Two factors with few levels: one coefficient per combination of
        # levels. `bols(f, by=g)` is that cell design for a factor pair (the
        # %X% tensor of two factors is not supported); the factor with fewer
        # levels is the `by=`, which sets the number of blocks. Fixed effects
        # are not held out of the folds as a brandom group is, so the
        # grouped-folds caveat does not apply.
        smaller, larger = sorted((left, right), key=lambda c: levels.get(
            c, FACTOR_TENSOR_MAX_LEVELS + 1))
        return f"bols({quote_name(larger)}, by={quote_name(smaller)})"
    if left_binary or right_binary:
        indicator, other = (left, right) if left_binary else (right, left)
        if other in groups:
            if indicator in categorical:
                return f"brandom({quote_name(other)}) %X% brandom({quote_name(indicator)})"
            return f"brandom({quote_name(other)}, by={quote_name(indicator)})"
        return f"bbs({smooth(other)}, by={quote_name(indicator)})"
    if left_group and right_group:
        return f"brandom({quote_name(left)}) %X% brandom({quote_name(right)})"
    if left_group or right_group:
        group, other = (left, right) if left_group else (right, left)
        return f"brandom({quote_name(group)}, by={quote_name(other)})"
    # Five interior knots per margin (8 basis functions each, 64 columns):
    # with the 20-knot default a tensor has 576 columns, every step solves a
    # 576-dimensional system and each fold re-calibrates lambda on it -- 25
    # times slower on `benches/bench_insights.py`'s interaction task for the
    # same held-out error. mgcv's te() defaults to 5 functions per margin.
    return (f"bbs({smooth(left)}, df=2.5, knots=5)"
            f" %X% bbs({smooth(right)}, df=2.5, knots=5)")


def pair_terms_for(left, right, *, levels=None, groups=(), categorical=(), transforms=None):
    """Every learner worth trying for one interaction between two columns.

    For two continuous columns this returns three candidates in order: the
    tensor-product surface `pair_term_for` already builds, then
    ``bols(left, by=right)`` and ``bols(right, by=left)``, the two 1-df
    varying-coefficient directions. The tensor costs 6.25 df at its default
    margin and can miss a bilinear interaction that only reaches about one
    standard error under it; the cheap check is a single varying coefficient,
    and offering all three lets `compare_specifications` select among them.

    The `smooth()` handling is shared with the tensor, so a column fitted on
    the log or square-root scale (``transforms={"x": "log"}``) keeps that
    scale in every candidate. For any other combination of column types
    (binary, group, categorical) the answer is a one-element list holding
    exactly `pair_term_for`'s output, which stays unchanged.
    """
    transforms = transforms or {}

    def smooth(column):
        kind = transforms.get(column)
        return f"{kind}({quote_name(column)})" if kind else quote_name(column)

    levels = levels or {}
    groups = set(groups)
    left_special = levels.get(left, 3) <= 2 or left in groups
    right_special = levels.get(right, 3) <= 2 or right in groups
    if not left_special and not right_special:
        return [
            pair_term_for(left, right, levels=levels, groups=groups,
                          categorical=categorical, transforms=transforms),
            f"bols({smooth(left)}, by={smooth(right)})",
            f"bols({smooth(right)}, by={smooth(left)})",
        ]
    return [pair_term_for(left, right, levels=levels, groups=groups,
                          categorical=categorical, transforms=transforms)]


@dataclass(frozen=True)
class Suggestion:
    """One change worth trying, and the evidence for it.

    ``basis`` says what kind of number ``expected_gain`` is, and the two are not
    interchangeable. ``"measured"`` means the change was fitted and scored on
    held-out folds, so the gain is what that change did. ``"detected"`` means
    something was found in the residual — joint structure in a pair of columns,
    say — without any model of it being fitted. A detected signal says there is
    something there; it does not say which learner will capture it, and the
    wrong learner can make the fit worse than leaving it out.
    """

    action: str
    term: str
    expected_gain: float
    reason: str
    basis: str = "measured"
    #: The full right-hand side this change produces, ready to pass to
    #: `compare_specifications` or `gamboost`. A suggestion that names a
    #: change without producing the specification leaves the caller to apply
    #: it by hand, which is where the transcription errors come from.
    specification: str = ""
    #: The numbers behind the ranking, so a suggestion can be checked rather
    #: than taken.
    evidence: dict = field(default_factory=dict)

    def __repr__(self) -> str:      # pragma: no cover - display only
        return (f"{self.action} {self.term}: {self.expected_gain:+.4g} "
                f"[{self.basis}] ({self.reason})")

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the change and the evidence for it."""
        return export_result(self, include_arrays=include_arrays)


def _apply_shape(rhs_terms, term, replacement):
    """The right-hand side with one term swapped."""
    return " + ".join(replacement if t == term else t for t in rhs_terms)


def _is_gaussian_contract(family):
    """The only families whose raw residual spread is a heteroscedasticity
    test. Squared-error but heavy-tailed losses (Huber, Laplace, quantiles)
    do not make ``log(residual^2)`` a ``log(chi^2_1)``; count and rate
    families have spread that legitimately moves with the mean."""
    return isinstance(family, Gaussian) or isinstance(family, GaussReg)


def heteroscedasticity(model, data=None, *, columns=None, transforms=None) -> dict:
    """Measure how residual spread changes across fitted values.

    A mean model assumes it does not. When it does, the fit is still the best
    guess at the mean and every interval around it is wrong -- too narrow
    where the spread is large, too wide where it is small -- so the answer is
    a distributional model (`glmboostLSS`/`gamboostLSS`), which gives the
    scale its own predictor rather than one number.

    The measurement is the slope of ``log(residual^2)`` on the fitted value.
    Under constant variance that slope is zero and ``log(residual^2)`` is
    ``log`` of a chi-square with one degree of freedom, whose variance is
    ``pi^2 / 2`` whatever the scale of the data. That fixed number gives the
    slope a standard error without estimating anything, so ``t`` below is
    comparable across data sets -- which an R-squared here is not: the
    irreducible variance of ``log(chi^2_1)`` is so large that a real and
    strong pattern still explains only a few percent of it.

    ``spread_ratio`` is the residual spread at the 90th percentile of the
    fitted values against the 10th. It is the number that says whether the
    pattern matters in practice, where ``t`` says whether it is there at all.

    The variance claim holds only under the normal-errors contract that
    :class:`~mboostrs.families.Gaussian` and
    :class:`~mboostrs.families.GaussReg` fit. A Gamma or Poisson fit is
    expected to show a spread that grows with the mean even when it is exactly
    right, so the raw residual spread is not a test of heteroscedasticity
    there; this diagnostic refuses those families by name rather than return a
    number that reads as one.

    ``by_column`` repeats the slope test against each numeric predictor of the
    data, whether or not the mean model uses it. A predictor that moves the
    spread without moving the mean (a column that enters only the variance,
    such as the number of households behind a block-group average) is found
    there and not in the fitted-value slope. ``columns`` names the predictors
    to test; by default every column that :func:`~mboostrs.classify_columns`
    describes as a predictor is a candidate, with the response left out.
    ``transforms`` maps a column to the scale its term is fitted on
    (``{"households": "log"}`` tests the slope on ``log(households)``). A
    candidate that cannot be tested is listed in ``untested_columns`` with the
    reason. ``tested_columns`` lists the columns that have a row in
    ``by_column``, and each row gives the ``transform`` it was tested on.

    Each column also has ``curve_removed``: the mean held-out reduction of
    squared log-squared-residual error from eight quantile-bin means over
    five fixed half-splits (seeds 0 through 4). Edges and means use only each
    fit half. A positive value nominates a possible varying scale even when
    its signed slope cancels, as for a symmetric U-shaped pattern. The five
    splits overlap; this number is a screening score, not a significance
    test or an independent-fold standard error. Discovery still requires its
    held-out Gaussian location-scale comparison before adopting the term.
    """
    #: Variance of `log(chi^2_1)`, which is `pi^2 / 2` -- the trigamma
    #: function at one half.
    log_chisq_variance = np.pi ** 2 / 2.0

    family = getattr(model, "_family", None)
    if family is not None and not _is_gaussian_contract(family):
        raise ValueError(
            "heteroscedasticity() is defined for a model with Gaussian "
            f"(normal-errors) residuals, but this model was fitted with "
            f"{type(family).__name__}, whose spread is expected to move with "
            "the mean even under a correct specification. The "
            "log(residual^2)-on-fitted-value test would therefore read a "
            "well-specified fit as heteroscedastic. Use the Gaussian or "
            "GaussReg family, or check the fit's own residuals instead"
        )

    fitted = np.asarray(model.fitted(), dtype=np.float64).ravel()
    residual = np.asarray(model.residuals(type="response"),
                          dtype=np.float64).ravel()
    log_square = np.log(np.maximum(residual ** 2, 1e-12))
    centred = fitted - fitted.mean()
    denominator = float(np.sum(centred ** 2))
    by_column, untested = _spread_by_column(
        model, data, log_square, log_chisq_variance, columns=columns, transforms=transforms)
    strongest = max(by_column, key=lambda name: abs(by_column[name]["t"]), default=None)
    listing = {"tested_columns": list(by_column), "untested_columns": untested,
               "by_column": by_column, "strongest": strongest}
    if denominator <= 0:
        # A constant fit (an intercept-only mean) has no fitted-value slope, but
        # its residual spread can still move with a predictor, so the column
        # tests run.
        return {"slope": 0.0, "se": 0.0, "t": 0.0, "spread_ratio": 1.0, **listing}
    slope = float(np.sum(centred * (log_square - log_square.mean())) / denominator)
    se = float(np.sqrt(log_chisq_variance / denominator))
    low, high = np.quantile(fitted, [0.1, 0.9])
    ratio = float(np.exp(0.5 * slope * (high - low)))
    return {"slope": slope, "se": se, "t": slope / se if se > 0 else 0.0,
            "spread_ratio": ratio, **listing}


def _response_name(model) -> str | None:
    """The response column named by a model's formula, or None when unknown."""
    formula = getattr(model, "_formula", None)
    if isinstance(formula, str) and "~" in formula:
        return partition_tilde(formula)[0].strip() or None
    return None


def _spread_by_column(model, data, log_square, log_chisq_variance, *, columns=None,
                      transforms=None):
    """The spread test against each numeric predictor of the data.

    A variance that depends on a predictor need not track the fitted value:
    when the mean depends on other columns, or on this one through a curve
    that is flat on average, the fitted-value slope is near zero while the
    spread doubles across the predictor's range. The slope of
    ``log(residual^2)`` on the column has the same known variance as the
    fitted-value slope, so each column gets a ``t`` on the same scale.

    Returns the rows by column, and a mapping of each candidate that was not
    tested to the reason. The candidates are `columns` when given, otherwise
    every column `classify_columns` describes, with the response left out.
    """
    from mboostrs._frame import read_column
    from mboostrs.terms import classify_columns

    source = data if data is not None else getattr(model, "_data", None)
    if source is None:
        return {}, {}
    transforms = dict(transforms or {})
    if columns is None:
        try:
            names = [kind.name for kind in
                     classify_columns(source, response=_response_name(model))]
        except (ValueError, TypeError):
            try:
                names = _model_predictor_schema(model)
            except ValueError:
                return {}, {}
    else:
        names = list(columns)
    rows, untested = {}, {}
    curve_values = []
    for name in names:
        try:
            column = read_column(source, name)
        except Exception:  # noqa: BLE001 -- a frame library's missing-column error
            untested[name] = "column not found in the data"
            continue
        if column.kind != "numeric":
            untested[name] = f"{column.kind} column: the slope test takes a numeric column"
            continue
        if column.n_levels <= 2:
            untested[name] = "two or fewer distinct values: the slope test needs a numeric scale"
            continue
        values = np.asarray(column.values, dtype=np.float64)
        if values.shape != log_square.shape or not np.all(np.isfinite(values)):
            untested[name] = "missing or non-finite values in the fitted rows"
            continue
        transform = transforms.get(name)
        if transform == "log":
            if not np.all(values > 0):
                untested[name] = "log transform needs positive values"
                continue
            values = np.log(values)
        elif transform is not None:
            untested[name] = f"unknown transform {transform!r}"
            continue
        centred = values - values.mean()
        denominator = float(np.sum(centred ** 2))
        if denominator <= 0:
            untested[name] = "constant column"
            continue
        slope = float(np.sum(centred * (log_square - log_square.mean())) / denominator)
        se = float(np.sqrt(log_chisq_variance / denominator))
        low, high = np.quantile(values, [0.1, 0.9])
        rows[name] = {"slope": slope, "se": se, "t": slope / se, "transform": transform,
                      "spread_ratio": float(np.exp(0.5 * slope * (high - low)))}
        curve_values.append(values)
    if curve_values:
        from mboostrs import _rust

        # This nominates a curve; it is not a calibrated test. All binning,
        # held-half scoring and averaging use existing numeric core APIs.
        design = np.column_stack(curve_values)
        gains = {i: [] for i in range(len(curve_values))}
        for seed in range(5):
            for index, removed in _rust.screen_main_effects(
                    design, log_square, 8, seed, [0] * len(curve_values)):
                gains[index].append(removed)
        for index, row in enumerate(rows.values()):
            row["curve_removed"] = _rust.paired_mean_se(
                np.asarray(gains[index]), np.zeros(5))[0]
    return rows, untested


def _interaction_reason(left, right, candidate, index):
    """The words for one interaction suggestion, naming which learner it is.

    A continuous x continuous pair carries three candidates (see
    `pair_terms_for`); ``index`` identifies which one this is, so the reason
    can say whether the suggestion is the 6.25-df tensor or one of the 1-df
    varying coefficients. Every other pair carries a single candidate and
    keeps the generic trade-off wording.
    """
    opening = ("the pair explains held-out residual variation that "
               "neither main effect does.")
    if candidate.startswith("bbs(") and "%X%" in candidate:
        return (opening + " Fit it as a tensor-product spline surface "
                "(6.25 df): the flexible surface for a genuinely curved "
                "interaction. btree fits axis-aligned thresholds and costs "
                "no dense design")
    if index == 0:
        return (opening + " A tensor surface fits a smooth interaction; "
                "btree fits axis-aligned thresholds and costs no dense design")
    if index == 1:
        return (opening + " Fit it as a 1-df varying coefficient: the effect "
                f"of {left} varies linearly with {right} — the cheap check "
                "for a bilinear interaction")
    return (opening + " Fit it as a 1-df varying coefficient: the effect "
            f"of {right} varies linearly with {left} — the cheap check "
            "for a bilinear interaction")


def suggest_specifications(report, screened=None, drop_below=0.0, top=None,
                           model=None, noise=2.0):
    """Rank candidate changes by their expected held-out gain.

    Draws on a `term_report` and, optionally, the output of
    `terms.screen_interactions`.

    Two warnings travel with the output. Gains are measured one change at a
    time and do not add up, because two changes to the same specification
    interact. And a suggestion marked ``detected`` rather than ``measured``
    reports evidence of structure, not the effect of any particular term — see
    `Suggestion`. Both mean the same thing in practice: apply one change,
    refit, and score it before believing it. `compare_specifications` is the
    way to do that.

    A term is suggested for removal when removing it does not raise the
    out-of-fold error, or when what it earns is less than `noise` standard
    errors above zero: on these folds such a term cannot be told from one that
    earns nothing. The second rule prefers the smaller specification, and a
    genuinely weak effect can be removed by it; score the removal before
    accepting it. Terms suggested for removal are also offered together, as one
    specification without all of them, because removing them one per round
    takes as many rounds as there are terms. A group term that `term_report`
    could not measure on its folds is never suggested for removal.

    A shape change whose measured gain is less than `noise` standard errors
    (the paired fold standard error `term_report` records) is not suggested:
    on these folds it cannot be told apart from no change. Pass ``noise=0`` to
    list every positive gain. A drop is suggested whenever removing the term
    does not raise the out-of-fold error, since that is a claim about no
    change.

    An interaction's term follows the columns' types (see `pair_term_for`):
    a tensor surface for two continuous columns, a smooth switched by a
    two-valued column, a random slope for a group column. A continuous x
    continuous pair yields *three* suggestions — the tensor plus both
    varying-coefficient directions, ``bols(a, by=b)`` and ``bols(b, by=a)`` —
    because the tensor costs 6.25 df and can miss a bilinear interaction that
    a 1-df varying coefficient captures. Score them against each other
    with `compare_specifications` rather than taking the first.
    """
    _, _, rhs = partition_tilde(report.formula)
    terms = split_terms(rhs)

    out = []
    for row in report.shapes:
        se = row.get("se", float("nan"))
        within_noise = np.isfinite(se) and row["gain over current"] < noise * se
        if (row["best"] != row["current"] and row["gain over current"] > 0
                and not within_noise):
            replacement = _as_shape(row["term"], row["best"])
            out.append(Suggestion(
                action="refit", term=row["term"],
                expected_gain=row["gain over current"],
                reason=(f"as {row['best']} instead of {row['current']}: "
                        + _SHAPE_TRADE_OFF.get(row["best"], "")),
                basis="measured",
                specification=_apply_shape(terms, row["term"], replacement),
                evidence={**{key: value for key, value in row.items()
                             if key.startswith("as ")},
                          "standard error": se}))
    idle = []
    for row in report.contributions:
        if row.get("measurable", True) is False:
            # These folds cannot say whether the groups differ, so a group term
            # is never removed for earning nothing. A removal that *lowers* the
            # error beyond noise is measured evidence, and is offered.
            se = row.get("se", float("nan"))
            if np.isfinite(se) and se > 0 and row["earns"] < -noise * se:
                out.append(Suggestion(
                    action="drop", term=row["term"], expected_gain=-row["earns"],
                    reason=(f"the specification predicts held-out groups better without "
                            f"it: removing it lowers out-of-fold error by "
                            f"{-row['earns']:.3g}, more than {noise:g} standard errors "
                            f"({se:.2g}). On these folds the term predicts the average "
                            f"group, so what it changes is how the other terms are fitted"),
                    basis="measured",
                    specification=" + ".join(t for t in terms if t != row["term"]),
                    evidence={"with it": report.baseline,
                              "without it": row["without it"],
                              "standard error": se}))
            continue
        se = row.get("se", float("nan"))
        within_noise = bool(np.isfinite(se) and se > 0 and row["earns"] < noise * se)
        if row["earns"] <= drop_below or within_noise:
            idle.append(row)
            reason = ("removing it does not raise out-of-fold error"
                      if row["earns"] <= drop_below else
                      f"what it earns, {row['earns']:.3g}, is less than {noise:g} "
                      f"standard errors ({se:.2g}) above zero, so on these folds "
                      f"it cannot be told from a term that earns nothing")
            out.append(Suggestion(
                action="drop", term=row["term"], expected_gain=-row["earns"],
                reason=reason,
                basis="measured",
                specification=" + ".join(t for t in terms if t != row["term"]),
                evidence={"with it": report.baseline,
                          "without it": row["without it"],
                          "standard error": row.get("se", float("nan"))}))
    remaining = [t for t in terms if t not in {row["term"] for row in idle}]
    if len(idle) >= 2 and remaining:
        out.append(Suggestion(
            action="drop", term=f"{len(idle)} terms that earn nothing",
            expected_gain=float(-sum(row["earns"] for row in idle)),
            reason=("each of these terms earns nothing that the folds can tell "
                    "from zero when removed on its own. Removed together they are "
                    "one change, to be scored as one: correlated terms can each be "
                    "replaced by the others and still be needed together, and on a "
                    "frame of many correlated columns removing them one at a time "
                    "can remove information the model needs. Terms: "
                    + ", ".join(row["term"] for row in idle)),
            basis="measured",
            specification=" + ".join(remaining),
            evidence={"terms": [row["term"] for row in idle]}))
    levels = getattr(report, "levels", None) or {}
    group_columns = getattr(report, "groups", ()) or ()
    for pair in (screened or []):
        if getattr(pair, "removed", 0) > 0:
            candidates = pair_terms_for(pair.left, pair.right, levels=levels,
                                        groups=group_columns)
            for index, interaction in enumerate(candidates):
                out.append(Suggestion(
                    action="add", term=f"{pair.left} x {pair.right}",
                    expected_gain=float(pair.removed),
                    reason=_interaction_reason(pair.left, pair.right,
                                               interaction, index),
                    basis="detected",
                    specification=" + ".join([*terms, interaction]),
                    evidence={"residual MSE removed": float(pair.removed)}))
    if model is not None:
        family = getattr(model, "_family", None)
        if family is not None and _is_gaussian_contract(family):
            spread = heteroscedasticity(model)
            driver = None
            column_hits = {
                name: row for name, row in spread.get("by_column", {}).items()
                if abs(row["t"]) > 3.5
                and max(row["spread_ratio"], 1 / row["spread_ratio"]) > 1.3}
            if column_hits:
                driver = max(column_hits, key=lambda name: abs(column_hits[name]["t"]))
            fitted_hit = abs(spread["t"]) > 3.0 and spread["spread_ratio"] > 1.3
            if fitted_hit or driver is not None:
                if driver is not None and (
                        not fitted_hit
                        or abs(column_hits[driver]["t"]) > abs(spread["t"])):
                    ratio = column_hits[driver]["spread_ratio"]
                    ratio = max(ratio, 1 / ratio)
                    where = f"across the range of `{driver}`"
                else:
                    ratio = spread["spread_ratio"]
                    where = "across the fitted range"
                out.append(Suggestion(
                    action="model", term="distributional response",
                    expected_gain=0.0,
                    reason=(f"the residual spread changes by a factor of "
                            f"{ratio:.2g} {where}, which a mean model treats "
                            f"as constant. The "
                            f"fit stays the best guess at the mean; every "
                            f"interval around it is too narrow where the spread "
                            f"is large and too wide where it is small. A "
                            f"distributional fit gives the scale its own "
                            f"predictor"),
                    basis="detected",
                    specification=f"gamboostLSS({report.formula!r}, "
                                  f"families=GaussianLSS())",
                    evidence=spread))
    # Measured gains first: a detected signal is weaker evidence whatever its
    # magnitude, and sorting the two together would hide that.
    out.sort(key=lambda s: (s.basis == "measured", s.expected_gain), reverse=True)
    return out[:top] if top else out


#: What each shape assumes, in one clause, so a suggestion says what it is
#: asking the caller to believe rather than only that it scored better.
_SHAPE_TRADE_OFF = {
    "bols": "one slope, so the effect is assumed straight",
    "bbs": "a curve that may bend, penalised toward a straight line",
    "btree": "axis-aligned thresholds, so a jump is fitted and a gradual "
             "change is not",
    "brandom": "a mean per level, shrunk toward the overall mean",
}


def sensitivity(model, data, variables=None, points=25, quantiles=(0.1, 0.9),
                parameter=None):
    """How much the fitted function moves across each variable's range.

    The predictors are the fitted model's own -- the columns its terms name
    -- never every column in `data`. For each one the fitted partial effect
    is evaluated on a grid with the other predictors held at their means, and
    three numbers are reported: the total rise and fall of the effect, its
    slope between the given quantiles, and whether it turns. ``turns`` is
    true when the effect rises and then falls (or the reverse), with each of
    the two moves larger than 5% of the effect's range. Smaller reversals,
    such as spline wiggles on a nearly monotone curve, are not turns. The
    flag is what distinguishes a monotone effect that can be summarised by a
    single number from one that cannot.

    A categorical predictor is gridded over its actual levels, not a numeric
    interpolation between codes ("level 0.37" is not a category). For an
    unordered one the slope and the turn count are ``nan``: without a
    declared order there is no scale to take a slope along, and the
    "direction" of the effect would depend on an arbitrary level ordering. A
    datetime predictor is a numeric time axis and is reported in its unit.

    A distributional model (``gamboostLSS()`` or ``glmboostLSS()``) has a
    separate predictor set for each distribution parameter, so it is measured
    one parameter at a time. ``parameter`` names that parameter (``"mu"``,
    ``"sigma"``, ...); the default is the family's first parameter. The effect
    is the parameter's linear predictor on its link scale (the log scale for
    a log-link parameter such as ``sigma``), and each row names the parameter
    it was measured on. ``parameter`` is refused for a gamboost()/glmboost()
    model, which has one predictor set.

    The columns that the other parameters' formulas read are held at their
    means (or most frequent levels) on the grid, as the non-gridded columns
    of the measured parameter are, so a parameter is measured whatever the
    other formulas contain. Those columns must be in `data`.

    A model fitted with ``na_action="impute"`` has missing-value indicator
    columns (``<col>_missing``, or ``<col>_missing_`` when the data already
    has a column of that name). They are built from `<col>` when the grid is
    predicted, so `data` supplies only the original columns.
    """
    rows = [row for row, _ in _effects(model, data, variables, points,
                                       quantiles, parameter)]
    rows.sort(key=lambda r: r["range of the effect"], reverse=True)
    return rows


def _effects(model, data, variables, points, quantiles, parameter, log_grid=()):
    """The sensitivity row and the fitted curve of each variable, in order.

    The curve is the centred linear predictor at each grid point; the row's
    range and slope are read from it. :func:`sensitivity` returns the rows
    sorted by range, and :mod:`mboostrs.discover` reads the curve's shape
    from the same grid. The grid is evenly spaced on the column's own scale,
    except for the names in ``log_grid``, whose grid is geometric (evenly
    spaced on the log scale) so a right-skewed positive column is covered
    evenly.
    """
    from mboostrs._frame import frame_columns
    from mboostrs.gamlss import LSSModel

    distributional = isinstance(model, LSSModel)
    if parameter is not None and not distributional:
        raise ValueError(
            "parameter= names a distribution parameter of a gamboostLSS() or "
            "glmboostLSS() model, and this is a gamboost()/glmboost() model "
            "with one predictor set. Drop parameter= to measure all of its "
            "predictors.")
    if distributional:
        k = 0 if parameter is None else model._param_index(parameter)
        fit_parameter = model.param_names[k]
        schema = _parameter_predictor_schema(model, k)
        hold_schema = _distributional_predictor_schema(model)
        specs = model._params[k].get("learners") or ()
        hold_specs = [spec for p in model._params for spec in p.get("learners") or ()]
        scope = f"parameter {fit_parameter!r}"
    else:
        schema = _model_predictor_schema(model)
        hold_schema = schema
        specs = None
        hold_specs = None
        scope = "this model"
    columns = {c.name: c for c in frame_columns(data)}
    # Prediction replays every parameter's recipe, so the columns of the other
    # parameters are needed too, and they are held at their typical values.
    needed = list(dict.fromkeys([*schema, *hold_schema]))
    missing = [name for name in needed if name not in columns]
    if missing:
        raise ValueError(_missing_predictor_message(model, missing))
    level_valued = _level_valued_predictors(model, specs)
    hold_level_valued = _level_valued_predictors(model, hold_specs)
    variables = list(variables or schema)
    unknown = [name for name in variables if name not in schema]
    if unknown:
        raise ValueError(
            f"not a predictor of {scope} (its terms name {schema}): "
            f"{unknown[0] if len(unknown) == 1 else unknown}")
    out = []
    for name in variables:
        column = columns[name]
        present = column.values[~np.isnan(column.values)]
        if present.size == 0:
            raise ValueError(f"column {name!r} has no non-missing values")
        levels_kind = column.kind in ("categorical", "boolean")
        if levels_kind:
            codes = np.sort(np.unique(present))
            axis = codes
            grid = np.array([column.label(int(c)) for c in codes], dtype=object)
        elif name in level_valued:
            # Gridding group labels by interpolation asks the model about a
            # level it has no coefficient for: `brandom` refuses a level of
            # 1.44, and would be wrong to accept it.
            axis = np.sort(np.unique(present))
            grid = axis
        elif name in log_grid and np.min(present) > 0:
            axis = np.geomspace(np.min(present), np.max(present), points)
            grid = axis
        else:
            axis = np.linspace(np.min(present), np.max(present), points)
            grid = axis
        others = {}
        for other, data_column in columns.items():
            if other not in hold_schema:
                continue
            values = data_column.values[~np.isnan(data_column.values)]
            if values.size == 0:
                raise ValueError(
                    f"column {other!r} has no non-missing values to hold "
                    "the other predictors at")
            if data_column.kind in ("categorical", "boolean"):
                label = data_column.label(int(_modal_code(values)))
                others[other] = np.full(grid.size, label, dtype=object)
            elif other in hold_level_valued:
                # The mean of group labels is not a label, so hold the term at
                # its most frequent level, as a typed categorical column gets.
                others[other] = np.full(grid.size, float(_modal_code(values)))
            else:
                others[other] = np.full(grid.size, float(np.mean(values)))
        others[name] = grid
        if distributional:
            fitted = np.asarray(model.predict(
                newdata=others, parameter=fit_parameter, type="link")).ravel()
        else:
            fitted = np.asarray(model.predict(newdata=others)).ravel()
        fitted = fitted - fitted.mean()
        lo, hi = np.quantile(present, quantiles)
        at = np.interp([lo, hi], axis, fitted)
        if (column.kind == "categorical" and not column.ordered) or name in level_valued:
            slope = float("nan")
            turns = float("nan")
        else:
            slope = (float((at[1] - at[0]) / (hi - lo)) if hi > lo
                     else float("nan"))
            turns = _has_turn(fitted)
        row = {"variable": name}
        if distributional:
            row["parameter"] = fit_parameter
        row["range of the effect"] = float(fitted.max() - fitted.min())
        row[f"slope, {quantiles[0]:.0%} to {quantiles[1]:.0%}"] = slope
        row["turns"] = turns
        out.append((row, fitted))
    return out


def _missing_predictor_message(model, missing: list[str]) -> str:
    """The sensitivity error for columns the data lacks, naming each term.

    A column is named with the term the caller wrote for it (`bols(roof_class)`,
    not the learner it was fitted as) and, when the model generated it, the
    column it was generated from.
    """
    described = []
    for name in missing:
        source = _generated_indicator_source(model, name)
        if source is not None:
            described.append(
                f"{name!r}, a missing-value indicator that na_action='impute' "
                f"generates from {source!r}")
            continue
        term = _written_term(model, name)
        described.append(f"{name!r} (in {term})" if term != name else repr(name))
    noun = "it" if len(missing) == 1 else "them"
    return ("sensitivity needs every column the model's terms read, and the "
            "data is missing " + noun + ": " + "; ".join(described)
            + ". Add " + ("it" if len(missing) == 1 else "them")
            + " to the data; a generated indicator needs only its source column.")


def _generated_indicator_source(model, name: str) -> str | None:
    """The column a missing-value indicator was generated from, else None."""
    recipe = getattr(model, "_transform_recipe", None)
    for step in getattr(recipe, "steps", None) or ():
        if getattr(step, "kind", None) == "missing_indicator" and step.name == name:
            return step.source
    return None


def _written_term(model, column: str) -> str:
    """The term the caller wrote for `column`, such as `bols(roof_class)`.

    The model keeps the formula as written (one per parameter for a
    distributional model). A column with no call in the formula, or no
    formula at all, is returned as itself.
    """
    from mboostrs.formula import _signed_terms

    formulas = getattr(model, "_formula", None)
    if isinstance(formulas, dict):
        texts = list(formulas.values())
    else:
        texts = [formulas]
    for text in texts:
        if not isinstance(text, str) or "~" not in text:
            continue
        _, _, rhs = partition_tilde(text)
        for _, term in _signed_terms(rhs):
            term = term.strip()
            if term != column and re.search(rf"\b{re.escape(column)}\b", term):
                return term
    return column


#: A turn in a fitted curve counts only when the curve rises, and then falls
#: (or the reverse), by more than this share of the curve's range on both
#: sides of the reversal. Smaller reversals are wiggles of the fitted spline
#: or noise in a nearly monotone effect.
TURN_SHARE = 0.05


def _has_turn(curve: np.ndarray, share: float = TURN_SHARE) -> bool:
    """Whether the curve reverses direction with a leg larger than `share` of its range.

    The curve is reduced to its turning points (the start, the end, and each
    point where the direction changes). Each leg between two turning points
    is a rise or a fall of a given size. A turning point is a turn when the
    legs on both sides of it are each larger than `share` times the curve's
    range.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.size < 3 or not np.all(np.isfinite(curve)):
        return False
    tol = share * float(curve.max() - curve.min())
    if not tol > 0.0:
        return False
    steps = np.diff(curve)
    moving = np.flatnonzero(steps != 0.0)
    if moving.size < 2:
        return False
    # Flat stretches are skipped: a turn is a sign change between the first and
    # last moving steps of the curve. Each turning point sits at the curve
    # index where one run of moves ends and the next begins.
    flips = np.flatnonzero(np.diff(np.sign(steps[moving])) != 0)
    points = np.concatenate((
        [moving[0]], moving[flips] + 1, [moving[-1] + 1]))
    legs = np.abs(np.diff(curve[points]))
    return bool(np.any((legs[:-1] > tol) & (legs[1:] > tol)))


def _modal_code(values: np.ndarray) -> int:
    """The most frequent code, which stands in for the mean of a category."""
    codes, counts = np.unique(values, return_counts=True)
    return int(codes[np.argmax(counts)])


#: Learner kinds whose column carries group labels rather than a measured
#: quantity. The fitted function is defined at the observed levels and nowhere
#: between them, whatever the column's dtype: `brandom(district)` on an integer
#: column is still a set of groups.
_LEVEL_VALUED_LEARNERS = ("Brandom", "Bmrf")


def _level_valued_predictors(model, specs=None) -> set[str]:
    """Predictors the model reads as group labels rather than as numbers.

    `specs` are the learner objects to read; by default they are the model's
    own learner specs.
    """
    if specs is None:
        specs = getattr(model, "_learners_spec", None) or ()
    names = set()
    for spec in specs:
        if type(spec).__name__ in _LEVEL_VALUED_LEARNERS:
            name = getattr(spec, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _model_predictor_schema(model) -> list[str]:
    """The columns the fitted model's terms reference, in formula order.

    The model stores this as its learner specs; a restored (pickled) model
    persists those too, so a fresh fit and a reloaded one report the same
    schema. Only when a model has neither is the formula string parsed.
    """
    specs = getattr(model, "_learners_spec", None)
    if specs:
        return _derived_column_sources(model, _learner_spec_schema(specs))
    formula = getattr(model, "_formula", None)
    if isinstance(formula, str) and formula:
        # A model without learner specs keeps its formula, and an imputed one
        # names the generated `<col>_missing` indicator there; it maps back to
        # the column the caller supplies, as the spec path does.
        return _derived_column_sources(model, _formula_predictor_schema(formula))
    raise ValueError(
        "sensitivity needs the model that fitted it (a gamboost()/glmboost() "
        "model, or a gamboostLSS()/glmboostLSS() model with parameter=); this "
        "object exposes neither learner specs nor a formula. For a "
        "distributional model, rank one parameter's predictors with "
        "sensitivity(model, data, parameter='sigma'), or contrast one input "
        "with prediction_contrast(model, data, variable, reference, "
        "alternative, parameter='sigma') or distributional_contrast(..., "
        "quantity='mean')")


def _parameter_predictor_schema(model, k: int) -> list[str]:
    """The columns the terms of one distributional parameter reference.

    A gamboostLSS parameter keeps its learner objects, read the same way as a
    gamboost model's specs. A glmboostLSS parameter keeps only its column
    names, which are its linear terms.
    """
    parameter = model._params[k]
    learners = parameter.get("learners")
    if learners:
        names = _learner_spec_schema(learners)
    else:
        names = [name for name in parameter.get("predictor_names") or ()
                 if isinstance(name, str)]
    return _derived_column_sources(model, names)


def _distributional_predictor_schema(model) -> list[str]:
    """The columns the terms of every distribution parameter reference.

    Prediction for any one parameter replays the model's whole transform
    recipe, and the other parameters' learners are part of the same model, so
    a grid for one parameter holds these columns as well. Parameter order is
    kept, and a column two parameters share is listed once.
    """
    names: list[str] = []
    for k in range(len(model._params)):
        for name in _parameter_predictor_schema(model, k):
            if name not in names:
                names.append(name)
    return names


def _derived_column_sources(model, names: list[str]) -> list[str]:
    """Derived columns replaced by the columns the caller passed them through.

    A `bols(<string column>)` term is fitted through a derived design column
    (`__by0`; a tensor `bbs(x) %X% bbs(z)` uses `__ix0`), and a transform
    term such as `bbs(log(b))` through a column computed from `b` (`log__b`).
    The learner specs name the derived column, not the one in the caller's
    frame. Reporting the derived name asks the user for a column they never
    created and cannot supply. The model's interaction records and transform
    recipe name the source columns each derived column reads, so the derived
    column maps back to them.
    """
    interactions = getattr(model, "_interactions", None) or []
    sources: dict[str, list[str]] = {}
    for item in interactions:
        derived = getattr(item, "derived", None)
        if not isinstance(derived, str) or not derived:
            continue
        columns: list[str] = []
        for side in (getattr(item, "left", None), getattr(item, "right", None)):
            column = getattr(side, "column", None)
            if isinstance(column, str) and column and column not in columns:
                columns.append(column)
        if columns:
            sources[derived] = columns
    recipe = getattr(model, "_transform_recipe", None)
    for step in getattr(recipe, "steps", None) or ():
        derived = getattr(step, "name", None)
        raw = [s for s in (getattr(step, "sources", None) or ()) if isinstance(s, str)]
        if isinstance(derived, str) and derived and raw and derived not in raw:
            sources.setdefault(derived, raw)
    if not sources:
        return names
    resolved: dict[str, tuple[str, ...]] = {}

    def original_columns(name, path=()):
        if name in resolved:
            return resolved[name]
        if name in path:
            raise ValueError(
                "cyclic derived-column metadata: " + " -> ".join((*path, name)))
        if name not in sources:
            return (name,)
        columns = tuple(dict.fromkeys(
            column for source in sources[name]
            for column in original_columns(source, (*path, name))))
        resolved[name] = columns
        return columns

    return list(dict.fromkeys(
        column for name in names for column in original_columns(name)))


def _learner_spec_schema(specs) -> list[str]:
    """Column names a learner spec references: its variables plus any `by`.

    A spatial learner reads two coordinate columns and a tree reads a list of
    columns; their `name` joins those with commas, which is a label and not a
    column, so the columns themselves are listed instead.
    """
    names = []
    for spec in specs:
        if getattr(spec, "name1", None) is not None:
            raw = [spec.name1, getattr(spec, "name2", None)]
        elif getattr(spec, "names", None):
            raw = list(spec.names)
        else:
            raw = getattr(spec, "name", None)
            raw = [raw] if isinstance(raw, str) else (list(raw) if raw else [])
        by = getattr(spec, "by", None)
        for name in [*raw, by]:
            # "1" is the intercept (a formula's `y ~ 1`), which reads no column.
            if isinstance(name, str) and name and name != "1" and name not in names:
                names.append(name)
    return names


def _formula_predictor_schema(formula: str) -> list[str]:
    """The bare column names a learner-form formula references.

    `bbs(x, df=4)` names `x`, `bols(z, by=g)` names `z` and `g`. The analysis
    stops at the first `=` in a call so a keyword argument (whose value may
    itself contain commas, as in `boundary.knots = c(0, 10)`) is never read
    as a column.
    """
    from mboostrs.formula import _split_formula

    _, terms = _split_formula(formula)
    names = []
    for term in terms:
        term = term.strip()
        call = None
        masked = mask_names(term)
        for opener in ("(", "["):
            cut = masked.find(opener)
            if cut != -1:
                call = (term[:cut], term[cut + 1:].rstrip(")]"))
                break
        if call is None:
            if re.fullmatch(rf"{N}", term):
                names.append(unquote_name(term))
            else:
                raise ValueError(
                    f"cannot read the columns named by formula term {term!r}")
            continue
        for by in re.findall(rf"\bby\s*=\s*({N})", call[1]):
            names.append(unquote_name(by))
        # Positional arguments are the columns; a keyword argument (`df=4`,
        # `intercept=False`, `by=g`) is never a column, so each comma-separated
        # piece is read on its own and the keyword ones are skipped.
        for _, token in split_top_level(call[1], ","):
            token = token.strip()
            if "=" in mask_names(token):
                continue
            if re.fullmatch(rf"{N}", token) and unquote_name(token) not in names:
                names.append(unquote_name(token))
    return names


def _common_sample(data, formulas, weights, observation_ids, folds):
    """Restrict a comparison to the rows complete for every formula.

    Returns ``(data, weights, observation_ids, folds, excluded)`` with every
    row-aligned input subset to the kept rows; a fold matrix or `FoldScheme`
    is subset as a whole so its weights, held-out and judge masks stay paired.
    """
    from mboostrs.formula import apply_na_action
    from mboostrs.stabsel import _take_rows

    dropped = set()
    for formula in formulas:
        _kept, rows = apply_na_action(formula, data, "omit")
        dropped.update(int(i) for i in rows)
    if not dropped:
        return data, weights, observation_ids, folds, 0
    n = len(next(iter(data.values())) if isinstance(data, dict) else data)
    keep = np.setdiff1d(np.arange(n), np.fromiter(dropped, dtype=int))
    data = _take_rows(data, keep)
    if weights is not None and np.ndim(weights) > 0:
        weights = np.asarray(weights)[keep]
    if observation_ids is not None:
        observation_ids = np.asarray(observation_ids)[keep]
    if isinstance(folds, FoldScheme):
        from mboostrs.autospec import _scheme_rows

        folds = _scheme_rows(folds, keep)
    elif folds is not None and np.ndim(folds) == 2:
        folds = np.asarray(folds)[keep]
    return data, weights, observation_ids, folds, len(dropped)


@dataclass(frozen=True)
class _ComparePlan:
    """Comparison plan fixed before any specification is fitted."""

    pairs: list
    data: object
    weights: object
    observation_ids: object
    folds: object
    excluded: int
    n: int
    planned_counts: list
    planned_total: int
    #: The inner step cap of mstop="cv", or None for a fixed mstop.
    max_tuned: object = None


def _level_balanced_folds(folds, data, formulas, n, seed, weights):
    """Turn a fold count into a scheme that keeps every factor level in each training fold.

    A random split of a small sample can hold every row of a rare level out of
    one fold's training set. Its effect is then unestimated in that fold, and
    a candidate that carries the factor can fail there. Only the factor
    columns the candidates use are considered. Each combination of their
    levels (a single level when there is one factor column) is dealt
    round-robin across the folds, starting from the fold with
    the fewest rows so far, so a combination with two or more rows has at
    least one of them in every training fold. A combination with one row is
    held out of exactly one fold, so that fold's training set lacks it; no
    split of that row avoids this. Returns `folds` unchanged when it is not a
    count or no factor column is used.
    """
    if isinstance(folds, bool) or not isinstance(folds, (int, np.integer)):
        return folds
    from mboostrs.folds import _base_weights, _scheme_from_assignment
    from mboostrs.formula import term_variables
    from mboostrs.interactions import factor_columns

    factors = sorted(
        column for column in factor_columns(data)
        if any(column in term_variables(formula) for formula in formulas))
    if not factors:
        return folds
    count = check_int_at_least("folds", folds, 2, allow_none=False)
    if count > n:
        raise ValueError(f"folds={count} exceeds the {n} observations")
    columns = []
    for column in factors:
        values = data[column]
        columns.append(np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values,
                                  dtype=object))
    labels = np.array(["|".join(str(v) for v in key) for key in zip(*columns, strict=True)],
                      dtype=object)
    _, codes = np.unique(labels, return_inverse=True)
    base = _base_weights(n, weights)
    rng = np.random.default_rng(seed)
    assignment = np.zeros(n, dtype=np.intp)
    load = np.zeros(count, dtype=np.int64)
    for code in range(int(codes.max()) + 1):
        rows = np.flatnonzero((codes == code) & (base > 0))
        rng.shuffle(rows)
        order = np.argsort(load, kind="stable")
        chosen = order[np.arange(rows.size) % count]
        assignment[rows] = chosen
        np.add.at(load, chosen, 1)
    return _scheme_from_assignment(
        assignment, base, f"level-balanced {count}-fold ({', '.join(factors)})",
        seed=int(seed))


def _compare_plan(specifications, data, response, family, folds, mstop, seed,
                  weights, na_action, observation_ids, max_mstop=None,
                  score="family"):
    """Parse the candidates and count their fits, without fitting anything.

    Returns the ``(name, rhs)`` pairs, the frame and fold inputs on the common
    sample (rows complete for every candidate under ``na_action='omit'``), the
    number of rows that sample leaves out, and the fit count of each candidate
    and of the whole comparison. `compare_specifications` and `estimate_cost`
    both read the count from here.
    """
    def right_hand_side(rhs):
        # A whole formula for the same response is read as its right-hand
        # side: agents and people write the formula they would fit.
        if isinstance(rhs, dict):
            return {param: right_hand_side(part) for param, part in rhs.items()}
        if not isinstance(rhs, str) or "~" not in rhs:
            return rhs
        lhs, _, right = partition_tilde(rhs)
        if lhs.strip() != str(response):
            raise ValueError(
                f"specification {rhs!r} is a whole formula for {lhs.strip()!r}, "
                f"but response={response!r}; compare_specifications takes "
                "right-hand sides, e.g. "
                f"{right.strip()!r}")
        return right.strip()

    if hasattr(specifications, "items"):
        pairs = [(name, right_hand_side(rhs)) for name, rhs in specifications.items()]
    else:
        # A sequence names each row by its right-hand side; a distributional
        # candidate's mapping is named by its parameters' right-hand sides,
        # since a dict cannot itself be a row name.
        pairs = [(_specification_name(rhs), rhs)
                 for rhs in map(right_hand_side, specifications)]
    if not pairs:
        raise ValueError("no specifications to compare")
    n = len(np.asarray(_response_values(data, response)))
    is_lss = hasattr(family, "param_names") and hasattr(family, "pdf")
    if not is_lss and any(isinstance(rhs, dict) for _name, rhs in pairs):
        raise ValueError(
            "a right-hand side mapping names distribution parameters; it needs "
            "a distributional family, e.g. family=GaussianLSS(), not "
            f"{type(family).__name__}")

    def candidate_formulas(rhs):
        """The formulas one candidate fits: one per distribution parameter,
        or the single formula of a standard-family candidate."""
        if is_lss:
            return list(_lss_formulas(rhs, response, family).values())
        return [f"{quote_response(response)} ~ {rhs}"]

    # One common sample: under na_action='omit' each candidate would drop the
    # rows missing in its own columns, and candidates scored on different rows
    # cannot be ranked (a candidate could improve by dropping harder rows).
    # Every candidate is scored on the rows complete for all of them, and the
    # rows left out are reported.
    excluded = 0
    if na_action == "omit":
        data, weights, observation_ids, folds, excluded = _common_sample(
            data, [f for _name, rhs in pairs for f in candidate_formulas(rhs)],
            weights, observation_ids, folds)
        n = len(np.asarray(_response_values(data, response)))
    folds = _level_balanced_folds(
        folds, data, [f for _name, rhs in pairs for f in candidate_formulas(rhs)],
        n, seed, weights)
    max_tuned = _check_max_mstop(mstop, max_mstop)
    planned_counts = [
        _planned_fit_count(
            n, folds, seed, weights, mstop, data=data,
            formulas=candidate_formulas(rhs), na_action=na_action,
            max_mstop=max_tuned, refit=(score != "family" or is_lss),
        )
        for _name, rhs in pairs
    ]
    return _ComparePlan(pairs=pairs, data=data, weights=weights,
                        observation_ids=observation_ids, folds=folds,
                        excluded=excluded, n=n, planned_counts=planned_counts,
                        planned_total=sum(planned_counts), max_tuned=max_tuned)


@one_summary
def compare_specifications(specifications, data, response, family=None,
                           folds=5, mstop=500, nu=0.1, seed=0, weights=None,
                           score="family", score_parameter=None,
                           na_action="error", observation_ids=None,
                           include_evaluations=False, fit_budget=None,
                           output="records", unseen_levels="average",
                           history=None, max_mstop=None, extrapolation="warn",
                           step="fixed"):
    """Score several right-hand sides on the same folds.
    Usage::

        import mboostrs as mb
        rows = mb.compare_specifications(
            {"base": "bbs(x, df=4)", "smooth": "bbs(x, df=6)"}, df, "y")
        rows[0]["specification"]  # the first entry is the reference

    ``step`` selects the ordinary boosting step rule (``"fixed"`` or
    ``"line_search"``), forwarded to every candidate's core-backed evaluator.
    It uses the same meaning as ``cv_risk(step=...)``; the default is unchanged.

    The loop's closing step. `term_report` says what is worth changing and
    `suggest_specifications` orders the candidates, but neither has fitted the
    changed model: a suggestion is a hypothesis until the specification carrying
    it is scored. This scores them together, on one partition, so the numbers
    are comparable to each other and to the specification they came from.

    With a distributional family, a candidate may be a mapping of parameter
    name to right-hand side — ``{"mu": "bbs(x)", "sigma": "1"}`` — or of
    parameter name to a whole formula for the response,
    ``{"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}``. The response is inserted into a
    right-hand side, so both forms score the same candidate. The constant-scale
    and varying-scale specifications a spread diagnosis proposes can be
    compared directly this way. Every candidate is scored by the
    family's joint proper score (negative log predictive density) on the same
    folds, and a candidate row's ``rhs`` is still the thing to edit and pass
    back. A plain string candidate with a distributional family shares the
    right-hand side across parameters, as a single-formula `gamboostLSS`
    call does.

    Parameters
    ----------
    specifications : mapping or sequence
        A mapping of name to right-hand side, or a plain sequence of
        right-hand sides, which are then named by the string itself. The first
        entry is the reference the others are compared against. A whole
        formula whose left-hand side is `response` is accepted and read as its
        right-hand side; a different left-hand side raises. In a plain
        sequence, a distributional candidate (a mapping of parameter to
        right-hand side) is named ``"mu: ...; sigma: ..."``.
    output : {"records", "auto", "polars", "pandas"}
        ``"records"`` -- the default, and what this always returned -- is a
        list of dicts, which stays editable: a row's ``rhs`` is the string to
        change and pass back in.
    include_evaluations : bool, default False
        Include each reconstructible :class:`EvaluationResult` in records
        output. Frame output refuses this option because an opaque Python model
        result has no stable tabular representation.
    fit_budget : int, optional
        Maximum model fits allowed for the complete comparison. The requirement
        is derived before fitting: one fit per outer fold at fixed ``mstop``;
        with ``mstop="cv"``, the outer fit plus every structure-aware inner
        fold. An inner search that grows past its first grid (see
        ``max_mstop``) refits every inner fold at each grid under any score
        other than the default family score, so the requirement counts every
        grid up to ``max_mstop``. The comparison refuses before work begins
        when the budget is insufficient.
    max_mstop : int, optional
        With ``mstop="cv"``, the largest step count the inner tuning of each
        candidate may reach; the default is 500. The inner search starts at
        500 steps, or at this cap when it is smaller, and doubles while its
        optimum is in the boundary band at the end of the grid, up to the cap.
        A candidate whose inner search reaches the cap with the risk still
        falling gets a warning naming ``max_mstop``, and its row reports
        ``inner_at_boundary``. Passing it with a fixed ``mstop`` is an error.
    unseen_levels : {"average", "zero_effect", "error"}, default "average"
        Passed to :func:`cv_risk`: how a held-out row is predicted when its
        ``brandom`` level does not occur in the fold's training rows.
    extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
        Passed to :func:`cv_risk`: how a held-out row outside its fold's
        training range is predicted. ``"warn"`` reports the rows in one
        warning for the whole comparison; ``"clip"`` holds the spline flat at
        the range edge; ``"error"`` refuses; ``"allow"`` predicts silently.
    history : list of dict, optional
        Rows from earlier ``compare_specifications(...,
        include_evaluations=True)`` calls, in the order they were run, on the
        same data, folds, family and score. The comparison then reports the
        whole search: ``search optimism`` and ``competitive set`` cover the
        history's candidates together with the current ones, and every row
        gains ``search candidates`` and ``search fits``, the number of
        candidates and model fits across every call, history first. Rows
        without their retained evaluation are refused, as are rows whose data,
        folds, score or evaluation weights do not pair with the current call.
        The same
        specification scored twice counts twice; pass each candidate once.

    Returns
    -------
    One row per specification, in the order given. Every row has
    ``specification`` (its name) and ``rhs``, the right-hand side it was
    scored with: a string for a string candidate, and a dict of parameter to
    right-hand side for a mapping candidate. The row also has its out-of-fold
    risk, the change from the reference with its paired fold standard error
    (``change se``), and the settings the score was computed under. A row is
    the evidence *and* the specification, so a comparison can be edited and
    rerun without reconstructing what was scored.

    A candidate that cannot be scored (for example, a fit refused on its
    data) does not stop the comparison. Its row has ``error``, a message that
    names the candidate and the cause, and NaN in ``out-of-fold risk``,
    ``change`` and ``change se``; the others are still compared against the
    reference, which is the first candidate that scores. The function raises
    only when every candidate fails.

    ``change se`` is the standard deviation of the fold-by-fold differences
    divided by the square root of the number of folds. On a scheme that stacks
    several partitions (`repeated_kfold_folds`, or `grouped_folds` with
    ``repeats=``) it is computed within each partition and the variances are
    averaged, because folds of different partitions reuse the same rows. Even
    for one partition it understates how much the change would vary across
    data sets, since the training sets of the folds overlap; a
    two-standard-error threshold is weaker than a normal table suggests.

    With ``mstop="cv"`` each candidate's step count is tuned inside every
    outer fold, so a row compares tuning procedures, and the paired standard
    error covers that tuning. The row reports the inner step counts across
    the outer folds as ``inner_mstop_min``, ``inner_mstop_median`` and
    ``inner_mstop_max``, and ``inner_at_boundary`` is true when any outer
    fold's inner search stopped at ``max_mstop`` with the risk still falling.
    With a fixed ``mstop`` these four fields are None. Its calibration was measured by simulation
    (``tools/simulate_nested_comparison.py``: n = 200, five outer and five
    inner folds, ``bbs(x)`` against ``bbs(x) + bols(u)``). When ``u`` carries
    no signal, ``change / change se`` over 1,000 replicates had standard
    deviation 1.15 rather than 1; the extra term was judged better by two
    standard errors in 2.4% of replicates (Monte Carlo SE 0.5%) and
    ``|change| > 2 change se`` occurred in 6.3% (SE 0.8%), against 4.6% for a
    standard normal statistic. With a coefficient of 0.15 on ``u`` (noise
    standard deviation 0.7), the two-standard-error rule detected it in 35%
    of 400 replicates. Read ``change se`` as about 15% too small at this
    sample size. Grouped, rolling-origin and repeated schemes are not covered
    by this check.

    Two fields on every row describe the search rather than one candidate.
    ``search optimism`` is the Tibshirani & Tibshirani (2009) estimate of how
    optimistic the winning score is because it is the best of the candidates
    compared: for each fold, the winner's score minus the lowest score any
    candidate reached on that fold, averaged over folds, in the same units as
    the score. A single-candidate comparison, or a winner that won every fold,
    has zero optimism; the more candidates tried and the fewer folds the
    winner won, the larger it grows. ``competitive set`` lists, winner
    included, every candidate whose out-of-fold risk is within one paired
    fold standard error of the winner's: the folds cannot tell those
    candidates apart, so preferring the winner among them is a choice the
    comparison does not support.
    """
    if include_evaluations and output != "records":
        raise ValueError("include_evaluations=True requires output='records'")
    plan = _compare_plan(specifications, data, response, family, folds, mstop,
                         seed, weights, na_action, observation_ids,
                         max_mstop=max_mstop, score=score)
    pairs, data, weights = plan.pairs, plan.data, plan.weights
    observation_ids, folds = plan.observation_ids, plan.folds
    planned_counts, planned_total = plan.planned_counts, plan.planned_total
    excluded = plan.excluded
    if plan.excluded:
        warnings.warn(
            f"compare_specifications scored every specification on the {plan.n} rows "
            f"complete for all of them; {plan.excluded} row(s) with a missing value in "
            f"a column some specification uses were excluded from every score",
            UserWarning, stacklevel=2)
    fit_budget = check_int_at_least("fit_budget", fit_budget, 1, allow_none=True)
    check_fit_budget(
        planned_total, fit_budget, "comparison",
        detail=f" for {len(pairs)} specifications under this outer/inner design",
        remedy="drop a specification, or use a fixed mstop instead of mstop=\"cv\"")

    rows, reference, reference_evaluation, cumulative_fits = [], None, None, 0
    reference_rhs = None
    evaluations, scored_pairs, failures = [], [], []
    for candidate_index, (name, rhs) in enumerate(pairs):
        try:
            evaluation = evaluate_specification(
                rhs, data, response, family=family, folds=folds, mstop=mstop,
                nu=nu, seed=seed, weights=weights, score=score,
                score_parameter=score_parameter, na_action=na_action,
                observation_ids=observation_ids, unseen_levels=unseen_levels,
                extrapolation=extrapolation, step=step,
                max_mstop=plan.max_tuned, _warn_as=None,
            )
            # A fold that could not be scored leaves the candidate without a
            # score; it is a failed candidate, with the fold's cause.
            failed = [row for row in evaluation.folds if row.get("status") == "failed"]
            if failed:
                raise ValueError(f"fold {failed[0]['fold']}: {failed[0]['error']}")
        except Exception as exc:
            # A budget refusal stops the whole comparison, as it does for every
            # other caller; it is not a property of one candidate.
            from mboostrs._workflow import BudgetExhausted

            if isinstance(exc, BudgetExhausted):
                raise
            # A candidate that cannot be scored is recorded on its own row, and
            # the rest of the comparison continues. The comparison raises only
            # when no candidate scores.
            message = f"candidate {name!r} could not be scored: {exc}"
            failures.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            rows.append(_failed_compare_row(
                name, rhs, message, score=score, score_parameter=score_parameter,
                planned=planned_counts[candidate_index], planned_total=planned_total,
                fit_budget=fit_budget, folds=folds, mstop=mstop, nu=nu, step=step,
                weighted=weights is not None, excluded=excluded,
                cumulative=cumulative_fits))
            continue
        scored_pairs.append((name, rhs))
        flagged = sum(1 for fold in evaluation.folds if fold["inner_at_boundary"])
        if flagged:
            warnings.warn(
                _inner_boundary_warning("compare_specifications", plan.max_tuned,
                                        flagged, len(evaluation.folds), label=name),
                RuntimeWarning, stacklevel=2)
        risk = evaluation.score
        cumulative_fits += evaluation.fit_count
        if reference is None:
            reference = risk
            reference_evaluation = evaluation
            reference_rhs = rhs
        else:
            check_evaluations([reference_evaluation, evaluation],
                              helper="compare_specifications")
        change = risk - reference
        change_se = (0.0 if evaluation is reference_evaluation else
                     _paired_se(evaluation.fold_scores, reference_evaluation.fold_scores,
                                _partitions(folds)))
        row = {
            "specification": name,
            # The editable half: this is the string to change and pass back.
            "rhs": rhs,
            "common sample excluded rows": excluded,
            "out-of-fold risk": risk,
            "change": change,
            # The standard error of `change`: both specifications are scored on
            # the same folds, so the difference is paired fold by fold.
            "change se": change_se,
            "note": _spanned_drop_note(reference_rhs, rhs, change, change_se),
            # The evidence half: what the number was computed under, so two
            # comparisons run days apart can be told apart.
            "score": score,
            "score parameter": score_parameter,
            "prediction scale": evaluation.prediction_scale,
            "loss units": evaluation.loss_units,
            "aggregation": evaluation.aggregation,
            "evaluation task": evaluation.evaluation_task,
            "data fingerprint": evaluation.data_fingerprint,
            "n evaluations": evaluation.n_evaluations,
            "fit count": evaluation.fit_count,
            "planned fit count": planned_counts[candidate_index],
            "cumulative fit count": cumulative_fits,
            "planned total fits": planned_total,
            "fit budget": fit_budget,
            "folds": _folds_label(folds),
            "mstop": mstop,
            **_inner_summary(evaluation.folds),
            "nu": nu,
            "step": step,
            "weighted": weights is not None,
            "error": None,
        }
        if include_evaluations:
            row["evaluation"] = evaluation
        rows.append(row)
        evaluations.append(evaluation)
    if reference_evaluation is None:
        raise ValueError(
            "no specification could be scored: " + "; ".join(failures)
            + ". Each row of a failed candidate names its error; fix the "
            "candidates or the data, then rerun.")
    history_names, history_evaluations, history_fits = _read_history(history)
    for past in history_evaluations:
        # Rows from another call join this one only if they pair: same data,
        # folds, family, score and weights. The shared check compares the
        # retained per-record fold membership, so a different seed or fold
        # count is refused, not averaged in.
        check_evaluations([reference_evaluation, past],
                          helper="compare_specifications")
    all_pairs = ([(name, None) for name in history_names] + scored_pairs)
    all_evaluations = history_evaluations + evaluations
    search_optimism, competitive_set = _search_summary(
        all_pairs, all_evaluations, _partitions(folds))
    competitive_set = list(dict.fromkeys(competitive_set))
    for row in rows:
        # Properties of the search, carried on every row: how optimistic the
        # winner's score is as the best of these candidates, and which
        # candidates the folds cannot tell from the winner. With `history=`
        # the search is every call's candidates, in the order run.
        row["search optimism"] = search_optimism
        row["competitive set"] = competitive_set
        row["search candidates"] = len(all_pairs)
        row["search fits"] = history_fits + row["cumulative fit count"]
    if output == "records":
        return rows
    from mboostrs._output import output_frame

    return output_frame(rows, output)


def _failed_compare_row(name, rhs, message, *, score, score_parameter, planned,
                        planned_total, fit_budget, folds, mstop, nu, step, weighted,
                        excluded, cumulative):
    """The row of a candidate that could not be scored.

    It keeps the candidate's name and right-hand side, the settings the comparison
    ran under, and its planned fit count. Its risk and change fields are NaN, and
    ``error`` carries the message that names the candidate. The evidence fields
    that come from a scored fit are None, and the fit count is None because the
    fits that ran before the failure are not reported per candidate.
    """
    nan = float("nan")
    return {
        "specification": name,
        "rhs": rhs,
        "note": None,
        "error": message,
        "common sample excluded rows": excluded,
        "out-of-fold risk": nan,
        "change": nan,
        "change se": nan,
        "score": score,
        "score parameter": score_parameter,
        "prediction scale": None,
        "loss units": None,
        "aggregation": None,
        "evaluation task": None,
        "data fingerprint": None,
        "n evaluations": None,
        "fit count": None,
        "planned fit count": planned,
        "cumulative fit count": cumulative,
        "planned total fits": planned_total,
        "fit budget": fit_budget,
        "folds": _folds_label(folds),
        "mstop": mstop,
        "inner_mstop_min": None,
        "inner_mstop_median": None,
        "inner_mstop_max": None,
        "inner_at_boundary": None,
        "nu": nu,
        "step": step,
        "weighted": weighted,
    }


def _read_history(history):
    """Names, retained evaluations and fit counts from earlier comparison rows."""
    if history is None:
        return [], [], 0
    if not isinstance(history, (list, tuple)) or not history:
        raise ValueError(
            "history must be the rows list of an earlier "
            "compare_specifications call; pass history=rows with "
            "include_evaluations=True on that call")
    names, evaluations, fits = [], [], 0
    for row in history:
        evaluation = row.get("evaluation") if isinstance(row, dict) else None
        if evaluation is None or "specification" not in row:
            raise ValueError(
                "every history row needs its retained evaluation and "
                "specification name; produce history rows with "
                "compare_specifications(..., include_evaluations=True)")
        names.append(row["specification"])
        evaluations.append(evaluation)
        fits += int(row.get("fit count", 0))
    return names, evaluations, fits


def _search_summary(pairs, evaluations, partitions):
    """Search optimism and competitive set of one comparison.

    The optimism is the Tibshirani & Tibshirani (2009) estimate of the bias in
    the best of many cross-validated scores: for each fold, the winner's score
    on that fold minus the lowest score any candidate reached on it, averaged
    over folds. A comparison of one candidate, or a winner that won every
    fold, has zero optimism; a winner that scraped home on a minority of folds
    against many candidates carries a larger one. With stacked partitions
    (repeated k-fold) it is computed within each partition and averaged, as
    the paired standard error is.

    The competitive set is every candidate whose out-of-fold risk is within
    one paired fold standard error of the winner's, winner included: the folds
    cannot tell them apart, so a claim that the winner is better needs more
    than this comparison.
    """
    scores = np.vstack([np.asarray(e.fold_scores, dtype=float)
                        for e in evaluations])
    winner = int(np.argmin([e.score for e in evaluations]))
    optimism = scores[winner] - scores.min(axis=0)
    optimism = _partition_mean(optimism, partitions)
    winner_scores = scores[winner]
    winner_risk = evaluations[winner].score
    competitive = [
        name
        for (name, _rhs), evaluation in zip(pairs, evaluations, strict=True)
        if evaluation is evaluations[winner]
        or (evaluation.score - winner_risk
            <= _paired_se(evaluation.fold_scores, winner_scores, partitions))
    ]
    return optimism, competitive


def _partition_mean(values, partitions):
    """Mean over folds, within stacked partitions, as `_paired_se` averages."""
    values = np.asarray(values, dtype=float)
    partitions = int(partitions or 1)
    if (partitions > 1 and values.size % partitions == 0
            and values.size // partitions >= 2):
        within = values.reshape(partitions, -1)
        return float(np.mean(within.mean(axis=1)))
    return float(values.mean())


def _planned_fit_count(n, folds, seed, weights, mstop, *, data,
                       formulas, na_action, max_mstop=None, refit=False):
    """Worst-case number of model fits used by one specification evaluation.

    `formulas` is a list because a distributional candidate carries one
    formula per parameter; a row missing in any of them is a row the fit and
    the inner tuning never see, so the count must drop it for every formula.
    With ``mstop="cv"`` the inner search may grow its grid up to `max_mstop`;
    a refitting search (`refit`: a score other than the family score, or a
    distributional candidate) refits every inner fold at each grid, so the
    count covers every grid up to the cap. The family score resumes its
    folds, so its count is one fit per inner fold however far the grid grows.
    """
    outer = _folds(n, folds, seed, weights)
    if mstop != "cv":
        return len(outer)
    cap = _MAX_TUNED_MSTOP if max_mstop is None else int(max_mstop)
    grids = 1 + _grid_growths(min(_MAX_TUNED_MSTOP, cap), cap)
    count = 0
    for fit_rows, _held_rows, _fit_weights in outer:
        inner_train = fit_rows
        if na_action == "omit":
            missing = _any_missing_rows(formulas, data, np.flatnonzero(fit_rows))
            kept = np.flatnonzero(fit_rows)[~missing]
            inner_train = np.zeros_like(fit_rows)
            inner_train[kept] = True
        inner = _inner_folds(folds, inner_train, seed)
        inner_fits = inner.n_folds if inner is not None else 5
        count += 1 + inner_fits * (grids if refit else 1)
    return count


def _inner_summary(fold_rows):
    """The inner step counts across the outer folds, and whether any fold's
    inner search stopped at its cap with the risk still falling. Every field is
    None when no fold tuned its step count."""
    stops = [row["inner_mstop"] for row in fold_rows if row["inner_mstop"] is not None]
    if not stops:
        return {"inner_mstop_min": None, "inner_mstop_median": None,
                "inner_mstop_max": None, "inner_at_boundary": None}
    return {
        "inner_mstop_min": int(min(stops)),
        "inner_mstop_median": float(np.median(stops)),
        "inner_mstop_max": int(max(stops)),
        "inner_at_boundary": any(row["inner_at_boundary"] for row in fold_rows),
    }


def _folds_label(folds) -> str:
    """How the folds were made, in words, for the comparison's own record."""
    scheme = getattr(folds, "scheme", None)
    if scheme is not None:
        return str(scheme)
    if isinstance(folds, (int, np.integer)):
        return f"{int(folds)}-fold random split"
    return f"{np.asarray(folds).shape[1]} supplied folds"
