"""Method forms of the model-level functions, shared by the standard models.

`GLMBoostModel` and `GAMBoostModel` historically exposed diagnostics,
inference, prediction interval, model-selection, and serialization operations
only as module-level functions taking the model as the first argument
(``mb.pit(fit, ...)``). This mixin adds the method forms (``fit.pit(...)``)
that a caller expects, delegating to the same functions. The functions remain
for a functional style and for backward compatibility.

Imports are deferred inside each method so this module can be a base class of
the model modules without an import cycle.
"""

from __future__ import annotations

import functools
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mboostrs._weights import WeightPolicyMixin
from mboostrs.agent_docs import guide_hint
from mboostrs.families import _TYPE_DEFAULT

if TYPE_CHECKING:
    from mboostrs.report import ExecutionReport


def capture_fit_warnings(fn):
    """Record the warnings a fit call shows and store them on its model.

    For the duration of the call, ``warnings.showwarning`` is wrapped: each
    warning that reaches it is recorded as a ``WarningMessage`` and then
    handed to the handler that was installed before, so the caller's filters
    decide what is shown exactly as they would without this decorator
    (``ignore`` and ``module=`` filters, once-per-location deduplication under
    the ``default`` action, and ``error``, which raises at the warning inside
    the fit). The recorded warnings are attached to the returned model as
    ``_fit_warnings`` and readable through ``model.fit_warnings``. A warning
    the caller's filters suppress is not recorded.

    The capture only runs on the main thread. ``warnings.showwarning`` is
    process-global, so a fit on a worker thread (the bootstrap and
    stability-selection refits run on a thread pool) installing and
    restoring its own wrapper would race with the main thread's. Worker-thread
    fits therefore call ``fn`` plain; their warnings reach whatever handler
    is active, which during a main-thread fit is that fit's recorder.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            return fn(*args, **kwargs)
        caught = []
        previous = warnings.showwarning

        def record(message, category, filename, lineno, file=None, line=None):
            caught.append(warnings.WarningMessage(
                message, category, filename, lineno, file, line))
            previous(message, category, filename, lineno, file, line)

        warnings.showwarning = record
        try:
            result = fn(*args, **kwargs)
        finally:
            if warnings.showwarning is record:
                warnings.showwarning = previous
        if result is not None:
            result._fit_warnings = tuple(caught)
        return result

    return wrapper


def warn_if_risk_rose(model) -> None:
    """Warn when a fitted path's training risk rose at a fixed step.

    A boosting step moves ``nu`` along the least-squares fit to the negative
    gradient. For a strongly curved loss (Poisson with small counts at
    ``nu=0.5``, say) that step can overshoot, and the training risk then
    oscillates instead of falling while the fit still returns: on a 300-row
    Poisson example the risk rose on 56 of 100 steps, by up to 2.3 times.
    The core selects the reporting tolerance and comparable-loss policy.
    Likelihood families that profile a nuisance minimize the same objective.
    Their final risk is also checked against the path minimum, so gradual
    backslide is reported; an adaptive loss such as Huber remains omitted.
    """
    import warnings

    import numpy as np

    from mboostrs import _rust
    from mboostrs.families import _family_params, _family_rust_name

    control = getattr(model, "_control", None)
    if control is None or getattr(control, "risk", "inbag") != "inbag":
        return
    if getattr(control, "step", "fixed") != "fixed":
        return
    try:
        name = _family_rust_name(model._family)
    except (AttributeError, KeyError, ValueError):
        return
    risk = np.asarray(getattr(model, "_risk", ()), dtype=np.float64).ravel()
    if risk.size < 3:
        return
    found = _rust.risk_path_rises(name, _family_params(model._family), risk.tolist())
    if found is None:
        return
    count, largest, final_above_best, best_step = found
    rises = (
        f"the training risk rose on {count} of {risk.size - 1} boosting steps "
        f"(largest rise {largest:.3g} of the risk before it)"
        if count else "the training risk rose above an earlier minimum"
    )
    warnings.warn(
        f"{rises}; the final risk "
        f"is {final_above_best:.3%} above its minimum at step {best_step}: the step "
        f"nu={float(control.nu):g} is too large for this loss's curvature, so the "
        "path does not consistently descend. Use a smaller nu, or "
        "step='line_search', which chooses each step's length on the loss.",
        RuntimeWarning, stacklevel=3)


#: Share of the gap between a learner's penalized and unpenalized df at which
#: `warn_if_penalty_drifted` warns. A `bbs(x, df=4)` term selected at
#: `nu=0.1` reaches about 0.2 after 1,000 selections and 0.35 after 3,000.
DRIFT_SHARE = 1 / 3

#: The drift is computed only when some learner has `nu * selections` at least
#: this large; below it no penalized learner has drifted far, and the check
#: (one `p x p` Gram and eigendecomposition per learner) is skipped.
DRIFT_MIN_STEPS = 50.0


def _learner_label(model, formula_idx):
    """The term as written, so `bbs(x)` and `bols(x)` stay apart."""
    names = list(model._learner_names)
    specs = list(getattr(model, "_learners_spec", None) or ())
    spec = specs[formula_idx] if formula_idx < len(specs) else None
    name = names[formula_idx]
    if spec is None or type(spec).__name__ == "Buser":
        return name
    return f"{type(spec).__name__.lower()}({name})"


def penalty_drift_rows(model) -> list[dict]:
    """One row per learner: selections and its df along the path.

    ``single_step_df`` is ``tr(S)`` for the learner's smoother ``S``;
    ``boosted_df`` is the df of its accumulated contribution after its
    selections, ``sum_k 1 - (1 - nu s_k)^m`` over the eigenvalues ``s_k`` of
    ``S``; ``unpenalized_df`` is the rank of its design, the df the
    contribution approaches as selections grow; ``drift_share`` is the part of
    the gap between the first and the last that boosting has used. Learners
    without a closed-form smoother (trees, monotone splines) are omitted.
    Computed by the core (`mboostrs_core::penalty_drift`).
    """
    store = getattr(model, "_learner_store", None)
    if store is None or not hasattr(store, "penalty_drift"):
        return []
    drift = store.penalty_drift(list(model._selected_rust()))
    rows = []
    for rust_idx, entry in enumerate(drift):
        if entry is None:
            continue
        selections, single, boosted, unpenalized = entry
        gap = unpenalized - single
        rows.append({
            "learner": _learner_label(model, model._rust_to_formula(rust_idx)),
            "selections": int(selections),
            "single_step_df": float(single),
            "boosted_df": float(boosted),
            "unpenalized_df": float(unpenalized),
            "drift_share": float(max((boosted - single) / gap, 0.0)) if gap > 1e-8 else 0.0,
        })
    return rows


def concurvity_rows(model, pairwise: bool) -> list[dict]:
    """One row per term (or per ordered pair when `pairwise`): mgcv's three
    concurvity measures of the boosted model's terms, computed by the core
    (`mboostrs_core::boosted_concurvity`).

    Concurvity is the additive-model form of collinearity: how far one
    term's function space reaches into another's. ``worst`` is the largest
    fraction of a term function's squared norm the other terms can
    reproduce, over every coefficient vector; ``observed`` is that fraction
    at the fitted coefficients (NaN for a term that was never selected);
    ``estimate`` averages over coefficient directions. Each term is first
    reduced by the constant direction, as mgcv measures its constrained
    smooths. Trees and `%O%` Kronecker learners are omitted.
    """
    store = getattr(model, "_learner_store", None)
    if store is None or not hasattr(store, "concurvity"):
        store = model._replay_store()
    if store is None or not hasattr(store, "concurvity"):
        return []
    included, summary, pairs = store.concurvity(
        model._selected_rust(), model._coefs_rust())
    labels = [_learner_label(model, model._rust_to_formula(i)) for i in included]
    if not pairwise:
        return [
            {"term": term, "worst": float(w), "observed": float(o), "estimate": float(e)}
            for term, (w, o, e) in zip(labels, summary, strict=True)
        ]
    rows = []
    for i, (against, row) in enumerate(zip(labels, pairs, strict=True)):
        for j, (term, (w, o, e)) in enumerate(zip(labels, row, strict=True)):
            if i == j:
                continue
            rows.append({"term": term, "against": against,
                         "worst": float(w), "observed": float(o),
                         "estimate": float(e)})
    return rows


#: Decorator frames between a caller and `gamboost` (warning capture and
#: argument checking), skipped when looking for the caller.
_FIT_DECORATORS = {("mboostrs._model_api", "wrapper"), ("mboostrs._workflow", "checked")}


def _fitted_for_the_package() -> bool:
    """Whether the `gamboost()` call being finished came from mboostrs itself.

    Searches, probes, bootstrap and cross-validation refits and the
    scikit-learn adapter fit long paths on purpose, many times; the drift
    warning is advice to a caller choosing ``mstop`` and would only repeat
    there. Walks out past the ``gamboost`` frame and its decorators
    (`_FIT_DECORATORS`) and reports whether the next caller is a ``mboostrs``
    module.
    """
    import sys

    frame = sys._getframe(1)
    passed_fit = False
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        if not passed_fit:
            if module == "mboostrs.gamboost" and name == "gamboost":
                passed_fit = True
        elif (module, name) not in _FIT_DECORATORS:
            return module == "mboostrs" or module.startswith("mboostrs.")
        frame = frame.f_back
    return False


def warn_if_penalty_drifted(model) -> None:
    """Warn when boosting has undone much of a learner's penalty.

    Each selection of a penalized learner (a P-spline, a ridge term) fits a
    shrunken step, but the steps accumulate: the learner's contribution
    approaches the unpenalized least-squares fit on its basis as it is
    selected again and again (Schulte and Rügamer, 2025). The `df=` given to
    the learner fixes one step, not the fit. This warns when some learner has
    used at least `DRIFT_SHARE` of the gap between the df of one step and the
    df of the unpenalized fit.

    Each distinct warning (wording with numbers replaced) is shown once per
    Python process, so a script fitting many models sees it once. Inside
    `collect_warnings()` every instance is recorded, and inside a routine that
    runs many fits (cross-validation, tuning) the routine issues one summary.
    """
    from collections import Counter

    if _fitted_for_the_package():
        return
    selected = list(getattr(model, "_selected", ()) or ())
    nu = float(getattr(model, "_nu", 0.0) or getattr(getattr(model, "_control", None), "nu", 0.0))
    if not selected or nu * max(Counter(selected).values()) < DRIFT_MIN_STEPS:
        return
    try:
        rows = penalty_drift_rows(model)
    except (AttributeError, TypeError, ValueError):
        return
    drifted = sorted((r for r in rows if r["drift_share"] >= DRIFT_SHARE),
                     key=lambda r: -r["drift_share"])
    if not drifted:
        return
    parts = "; ".join(
        f"{r['learner']}: {r['boosted_df']:.3g} of {r['unpenalized_df']:.3g} df after "
        f"{r['selections']} selections (one step: {r['single_step_df']:.3g})"
        for r in drifted)
    with _drift_lock:
        tally = _drift_tally
        if tally is not None:
            tally.add(len(drifted), parts)
            return
    from mboostrs._extrapolation import show_summary_once

    # Shown once per process for each wording (numbers aside), as the
    # extrapolation summaries are; collect_warnings() shows every instance.
    # stacklevel 3 points past show_summary_once and this function at the
    # gamboost() caller's caller.
    show_summary_once(
        f"boosting has undone much of the penalty of {len(drifted)} learner(s): "
        f"{parts}. " + _DRIFT_ADVICE, stacklevel=3)


#: The advice that closes the penalty-drift warning, shared by the single-fit
#: and the aggregated forms.
_DRIFT_ADVICE = (
    "Each selection fits a shrunken step, and the steps add up to "
    "the unpenalized fit on the learner's basis, so a penalty's df= no "
    "longer sets the term's flexibility. The predictions at this mstop "
    "already include that flexibility; the warning changes no fitted value. "
    "Choose mstop by cross-validation (cv_risk) rather than fitting this "
    "long, because cross-validation scores that flexibility out of fold. "
    "penalty_drift() lists every learner." + guide_hint("fit-and-tune")
)


class _DriftTally:
    """The drift warnings of the fits one routine ran, counted for one summary."""

    def __init__(self):
        self.fits = 0
        self.worst = ""
        self.worst_learners = 0

    def add(self, learners: int, parts: str) -> None:
        self.fits += 1
        if learners > self.worst_learners:
            self.worst, self.worst_learners = parts, learners

    def message(self) -> str:
        return (
            f"boosting has undone much of the penalty in {self.fits} fit(s) of this call "
            f"(the most drifted: {self.worst}). " + _DRIFT_ADVICE)


#: The tally of the routine now running, if any. While it is set,
#: `warn_if_penalty_drifted` counts instead of warning, and the routine emits one
#: warning when it ends. Guarded by `_drift_lock` because fits may run on threads.
_drift_tally: _DriftTally | None = None
_drift_lock = threading.Lock()


@contextmanager
def penalty_drift_collecting(stacklevel: int = 3):
    """Report the penalty-drift warnings of a routine's fits as one warning.

    A routine that runs many fits (cross-validation, tuning, comparisons)
    enters this context. Each fit whose drift check fires adds to the count,
    and when the routine ends one ``UserWarning`` names how many fits were
    affected and the most drifted of them. A routine nested in another adds to
    the outer count, and the outer routine issues the warning.
    """
    global _drift_tally
    with _drift_lock:
        outer = _drift_tally
        if outer is None:
            _drift_tally = tally = _DriftTally()
    if outer is not None:
        yield outer
        return
    try:
        yield tally
    finally:
        with _drift_lock:
            _drift_tally = None
    if tally.fits:
        warnings.warn(tally.message(), UserWarning, stacklevel=stacklevel)


@dataclass(frozen=True)
class StoppingBudget:
    """How far a fitted model was boosted, in the same shape for every model.

    `scheme` is ``"single"`` for a standard model (one predictor ``f``),
    ``"cyclic"`` for a distributional model that updates every parameter once
    per iteration, and ``"noncyclic"`` for one that picks a parameter per
    update. `per_parameter` maps each predictor to its number of updates;
    `total` is their sum, which is the budget a non-cyclic refit takes.
    """

    scheme: str
    total: int
    per_parameter: dict


#: The name of a standard model's single predictor in `predict_parameters`,
#: `selected_terms` and `stopping_budget`: mboost's ``f``.
STANDARD_PREDICTOR = "f"


class ModelAPIMixin(WeightPolicyMixin):
    """Method forms of the model-level operations for standard fits."""

    #: Warnings raised while this model was fitted; set by the fit functions.
    _fit_warnings: tuple = ()

    @property
    def fit_warnings(self) -> tuple:
        """Warnings the fit showed, as ``warnings.WarningMessage`` objects.

        Empty for a quiet fit. The warnings are also emitted normally during
        the fit call itself, so reading this attribute afterwards is an
        alternative to wrapping the call with ``warnings.catch_warnings``.
        A warning your filters suppress is not recorded: under Python's
        default filters that includes ``ResourceWarning`` and a warning the
        same line already showed once.

        Not persisted: ``save_model``/``load_model`` restore the model without
        the fit-time warnings, because the artifact format carries plain data
        only and the warning types are interpreter-specific.
        """
        return self._fit_warnings

    # ── parity notes ──────────────────────────────────────────────────

    def parity_notes(self, *, include_global=False, output="auto"):
        """Where this fit differs from what its names refer to.

        A user reaching for `Gehan`, `Bkernel` or an AFT `logLik` has a meaning
        in mind, taken from `mboost` or from the literature. Sometimes mboostrs
        computes something else; sometimes `mboost` does too and mboostrs
        reproduces it, so the object differs from its *name* rather than from
        upstream. Each note says which, because the two call for different
        responses: one might be a bug to report, the other is a property of
        the method as implemented everywhere.

        ``include_global=True`` adds the notes that describe the package rather
        than this fit — `continue_to` resuming the retained gradient when the
        route kept one and otherwise refitting from zero, the wheel's CPU
        baseline. ``output="text"`` returns the printable form.
        """
        from mboostrs._parity_notes import format_notes, notes_for

        notes = notes_for(self, include_global=include_global)
        if output == "text":
            return format_notes(notes)
        if output == "records":
            return [
                {
                    "name": note.name,
                    "mechanism": note.mechanism,
                    "consequence": note.consequence,
                    "shared_with_upstream": note.shared_with_upstream,
                    "evidence": note.evidence,
                }
                for note in notes
            ]
        if output in ("auto", "polars", "pandas"):
            from mboostrs._output import output_frame

            return output_frame(
                self.parity_notes(include_global=include_global,
                                  output="records"),
                "auto" if output == "auto" else output,
            )
        raise ValueError(
            f"output must be 'auto', 'text', 'records', 'polars' or 'pandas', "
            f"got {output!r}"
        )

    # ── prediction intervals ──────────────────────────────────────────

    def conformal_interval(
        self,
        newdata=None,
        *,
        level=0.9,
        method="split",
        seed=None,
        calib_fraction=0.5,
        n_folds=10,
        mstop=None,
        calibration=None,
        extrapolation="warn",
    ):
        """Distribution-free prediction interval. See :func:`mboostrs.conformal_interval`."""
        from mboostrs.conformal import conformal_interval

        return conformal_interval(
            self, newdata, level=level, method=method, seed=seed,
            calib_fraction=calib_fraction, n_folds=n_folds, mstop=mstop,
            calibration=calibration, extrapolation=extrapolation,
        )

    # ── serialization ─────────────────────────────────────────────────

    def save(self, path, *, indent=None):
        """Write a portable JSON artifact. See :func:`mboostrs.save_model`.

        Load it with :func:`mboostrs.load_model`.
        """
        from mboostrs.serialize import save_model

        return save_model(self, path, indent=indent)

    # ── prediction / coefficients across learners ─────────────────────

    def predict_all(self, newdata=None, type="link", mstop=None,
                    include_response=True, output="auto", extrapolation="warn"):
        """Per-learner predictions. See :func:`mboostrs.predict_all`."""
        from mboostrs.diagnostics import predict_all

        return predict_all(self, newdata, type, mstop, include_response, output,
                           extrapolation=extrapolation)

    def coef_all(self, output="auto"):
        """Coefficients for every learner. See :func:`mboostrs.coef_all`."""
        from mboostrs.diagnostics import coef_all

        return coef_all(self, output)

    # ── distributional diagnostics ────────────────────────────────────

    def deviance_increment(self, newdata=None):
        """Per-observation deviance. See :func:`mboostrs.deviance_increment`."""
        from mboostrs.diagnostics import deviance_increment

        return deviance_increment(self, newdata)

    def pit(self, newdata=None, mstop=None, randomized=True, random_state=None):
        """Probability integral transform. See :func:`mboostrs.pit`."""
        from mboostrs.diagnostics import pit

        return pit(self, newdata, mstop, randomized, random_state)

    def quantile_residuals(self, newdata=None, mstop=None, randomized=True, random_state=None):
        """Randomized quantile residuals. See :func:`mboostrs.quantile_residuals`."""
        from mboostrs.diagnostics import quantile_residuals

        return quantile_residuals(self, newdata, mstop, randomized, random_state)

    def residual_frame(
        self, kind="response", newdata=None, mstop=None, randomized=True,
        random_state=None, output="auto",
    ):
        """Residuals as a frame. See :func:`mboostrs.residual_frame`."""
        from mboostrs.diagnostics import residual_frame

        return residual_frame(
            self, kind, newdata, mstop, randomized, random_state, output
        )

    def q_stats(self, xvar=None, n_intervals=10, newdata=None, mstop=None,
                randomized=True, random_state=None, output="auto"):
        """Binned quantile-residual statistics. See :func:`mboostrs.q_stats`."""
        from mboostrs.diagnostics import q_stats

        return q_stats(
            self, xvar, n_intervals, newdata, mstop, randomized, random_state,
            output,
        )

    def worm_data(self, xvar=None, n_intervals=4, newdata=None, mstop=None,
                  randomized=True, random_state=None, output="auto"):
        """Worm-plot data. See :func:`mboostrs.worm_data`."""
        from mboostrs.diagnostics import worm_data

        return worm_data(
            self, xvar, n_intervals, newdata, mstop, randomized, random_state,
            output,
        )

    def calibration_frame(self, xvar=None, bins=10, newdata=None, mstop=None,
                          output="auto"):
        """Calibration table. See :func:`mboostrs.calibration_frame`."""
        from mboostrs.diagnostics import calibration_frame

        return calibration_frame(self, xvar, bins, newdata, mstop, output)

    def outlier_frame(self, newdata=None, mstop=None, output="auto"):
        """Flagged outliers. See :func:`mboostrs.outlier_frame`."""
        from mboostrs.diagnostics import outlier_frame

        return outlier_frame(self, newdata, mstop, output)

    # ── descriptions for a language model ─────────────────────────────

    def llm_explain_prompt(self, **kwargs):
        """Render this fit as a prompt for a language model.

        See :func:`mboostrs.llm_explain_prompt` for the arguments
        (``which``, ``top``, ``n_points``, ``dataset_description``, ``cv``,
        ``stabsel``, ``output``, ...).
        """
        from mboostrs.explain import llm_explain_prompt

        return llm_explain_prompt(self, **kwargs)

    def llm_describe(self, **kwargs):
        """Send this fit's description to a language model and return the reply.

        Makes a network request. See :func:`mboostrs.llm_describe` for the
        arguments (``llm``, ``base_url``, ``api_key``, ``client``, ``reasoning``,
        plus everything :func:`mboostrs.llm_explain_prompt` takes).
        """
        from mboostrs.explain import llm_describe

        return llm_describe(self, **kwargs)

    # ── model selection ───────────────────────────────────────────────

    def stabsel(self, **kwargs):
        """Stability selection. See :func:`mboostrs.stabsel` for the arguments
        (``cutoff``, ``q``, ``PFER``, ``B``, ``folds``, ...)."""
        from mboostrs.stabsel import stabsel

        return stabsel(self, **kwargs)

    def profile_mstop(self, grid=None, metric="risk", k=2.0):
        """Metric over the mstop path. See :func:`mboostrs.profile_mstop`."""
        from mboostrs.model_selection import profile_mstop

        return profile_mstop(self, grid=grid, metric=metric, k=k)

    # ── members the two standard model classes shared verbatim ────────
    #
    # `GLMBoostModel` and `GAMBoostModel` agree on 46 method names, but that is
    # mostly *interface* agreement: 26 of the 46 are genuinely different
    # implementations, including the three largest (`cvrisk` 291 lines and 15%
    # similar, `predict` 272 lines and 28%, `__init__` 131 lines and 43%). A
    # shared base class for the whole surface would be the wrong move, which is
    # what enhancement-hunt 3.3 concluded.
    #
    # These seventeen were the exception: byte-identical in both files,
    # decorators included. They read only attributes both classes set, so a
    # mixin is the right home and a change to one can no longer miss the other.

    @property
    def mstop(self) -> int:
        return len(self._selected)

    # A common fitted-model protocol (issue #169): the same three calls answer
    # "how far was it boosted", "what did it select" and "what does it
    # predict" for a standard model and a distributional one, with one return
    # shape each. A standard model has one predictor, named "f".

    def support_report(self, newdata, threshold=0.01, output="auto"):
        """Whether each new row lies where the training data has rows.

        One row per new observation. ``distance_sq`` is its squared
        Mahalanobis distance from the training rows over the model's numeric
        predictors (the training mean and the pseudo-inverse of the training
        covariance); ``training_share_beyond`` is the share of training rows
        at least that far, and ``outside_joint_support`` is true when that
        share is below `threshold`. A row inside every predictor's range can
        still be outside: a high ``x1`` with a low ``x2`` when the two move
        together in the data. ``unseen_levels`` lists ``column=level`` for
        categorical values the training data never had. The distance is
        linear in the predictors; a curved or clustered training cloud can
        leave gaps it does not see. Computed by the core
        (`mboostrs_core::support::joint_support`).
        """
        from mboostrs import _rust
        from mboostrs._frame import column_names
        from mboostrs._output import output_frame
        from mboostrs.formula import _formula_columns, parse_response
        from mboostrs.interactions import _raw_column
        from mboostrs.transforms import _column, _is_categorical

        bundle = getattr(self, "_refit_bundle", None) or {}
        train = bundle.get("raw_data")
        formula = bundle.get("raw_formula") or self._formula
        if train is None:
            train = self._data
        response = parse_response(formula)
        response = {response} if isinstance(response, str) else set(response or ())
        columns = [c for c in _formula_columns(formula, train)
                   if c not in response and c in set(column_names(newdata))]
        numeric = [c for c in columns if not _is_categorical(train, c)]
        categorical = [c for c in columns if c not in numeric]
        n_new = len(_raw_column(newdata, columns[0])) if columns else 0
        distance = np.full(n_new, np.nan)
        share = np.full(n_new, np.nan)
        if numeric and n_new:
            x_train = np.column_stack([np.asarray(_column(train, c), dtype=np.float64)
                                       for c in numeric])
            x_new = np.column_stack([np.asarray(_column(newdata, c), dtype=np.float64)
                                     for c in numeric])
            keep = np.all(np.isfinite(x_train), axis=1)
            result = _rust.joint_support(np.ascontiguousarray(x_train[keep]),
                                         np.ascontiguousarray(x_new))
            if result is not None:
                distance, share = (np.asarray(v, dtype=np.float64) for v in result)
        levels = {c: set(np.asarray(_raw_column(train, c), dtype=object).tolist())
                  for c in categorical}
        new_labels = {c: np.asarray(_raw_column(newdata, c), dtype=object)
                      for c in categorical}
        rows = []
        for i in range(n_new):
            unseen = [f"{c}={new_labels[c][i]}" for c in categorical
                      if new_labels[c][i] not in levels[c]]
            rows.append({
                "row": i,
                "distance_sq": float(distance[i]),
                "training_share_beyond": float(share[i]),
                "outside_joint_support": bool(share[i] < threshold),
                "unseen_levels": ", ".join(unseen),
            })
        return output_frame(rows, output)

    def stopping_budget(self) -> StoppingBudget:
        """How far this model was boosted; see :class:`StoppingBudget`."""
        steps = len(self._selected)
        return StoppingBudget(scheme="single", total=steps,
                              per_parameter={STANDARD_PREDICTOR: steps})

    def selected_terms(self, output="auto"):
        """One row per boosting update: ``order`` (1-based, across the whole
        fit), ``parameter`` (``"f"`` here), ``iteration`` (1-based, within the
        parameter; equal to ``order`` for a standard model) and ``learner``
        (the name :meth:`base_learner_names` gives it)."""
        from mboostrs._output import output_frame

        names = list(self.base_learner_names())
        rows = [{"order": step, "parameter": STANDARD_PREDICTOR, "iteration": step,
                 "learner": names[int(index)]}
                for step, index in enumerate(self._selected, start=1)]
        return output_frame(rows, output)

    def predict_parameters(self, newdata=None, type=_TYPE_DEFAULT) -> dict:
        """Return the single fitted predictor as ``{"f": array}``.

        The default is the response scale: the quantity the family is named
        for (a Gaussian ``f`` is the mean, a Poisson ``f`` the expected
        count, a Binomial ``f`` a probability), and no link-scale warning
        fires for an omitted ``type``. Pass ``type="link"`` for the linear
        predictor; ``predict()`` keeps the link scale as its own (warning)
        default.
        """
        if type is _TYPE_DEFAULT:
            type = "response"
        return {STANDARD_PREDICTOR: np.asarray(self.predict(newdata=newdata, type=type))}

    @property
    def offset(self):
        """The model offset: the constant the boosting path started from.

        A property, matching the LSS classes, where it always was one. The
        method form `fit.offset()` still works and warns; see
        `mboostrs._compat.DEPRECATIONS`.
        """
        from mboostrs._compat import still_callable

        return still_callable(float(self._offset),
                              call=lambda: float(self._offset),
                              deprecation="offset-call")

    @property
    def offset_(self) -> float:
        """The scalar model offset; the same value as :attr:`offset`."""
        return float(self._offset)

    @property
    def risk_(self) -> np.ndarray:
        """The empirical risk after each iteration, length ``mstop + 1``.

        Read-only: this is the fit's own array, not a copy. See
        :mod:`mboostrs._state`.
        """
        from mboostrs._state import readonly

        return readonly(self._risk)

    @property
    def risk(self):
        """The empirical risk after each iteration, as a read-only array.

        The same values as :attr:`risk_`, under the name the distributional
        classes use. Both spellings exist on all three classes and are
        properties, so a helper written against either works on any fit; the
        trailing-underscore form is the scikit-learn convention for fitted
        state, and this one is what the boosting literature calls it.

        Calling it, ``fit.risk()``, returns the same values as a fresh
        writable ndarray, so either spelling works.
        """
        from mboostrs._state import callable_readonly

        return callable_readonly(self._risk)

    def stopping_report(self) -> dict:
        """Where the boosting path stopped, and whether that looks finished.

        Reads the in-sample risk path the fit already stores — no refit — and
        reports the best step, the risk there versus at the final ``mstop``,
        and whether the risk rose after its best step. Two patterns get a
        textual flag:

        * best step well short of ``mstop`` with rising risk afterwards (the
          #220 signature: further steps at this ``nu`` do not help — try a
          smaller ``nu`` or ``step="line_search"``);
        * risk still falling at the ``mstop`` cap (consider increasing
          ``mstop``).

        When a cheap same-family GLM exists (:func:`same_family_gap`), the
        report also compares in-sample ``logLik`` at the final step with that
        GLM and flags a material shortfall (#226): cross-validation can stop
        while the predictor is still a shrunk version of the MLE. Nothing is
        refit and the chosen step is never moved; this only describes it.

        Returns a JSON-safe dict.
        """
        return _stopping_report_for(self)

    @property
    def selected(self) -> list[str]:
        """Learner names selected at each iteration."""
        return [self._learner_names[i] for i in self._selected]

    @property
    def size(self) -> float:
        """Fitted NBinomial size, with ``Var(Y) = mu + mu**2 / size``.

        The scalar at this model's step count, also returned by
        ``nuisance()[0]``. statsmodels NB2 uses ``alpha = 1 / size``.
        Raises ``AttributeError`` for other response families, including
        Hurdle, whose nuisance uses a different parameterization.
        """
        from mboostrs.families import NBinomial

        if not isinstance(self.family, NBinomial):
            raise AttributeError("size is available for fitted NBinomial models")
        return float(self.nuisance()[0])

    def selected_which(self, which=None) -> list[str]:
        """Learner names selected at each iteration, optionally filtered.

        which: int, str, or list thereof. If None, return all selections.
               If given, only return selections matching these learner
               indices (int) or names (str).
        """
        names = self.selected
        if which is None:
            return names
        if isinstance(which, int | str):
            which = [which]
        allowed = set()
        for w in which:
            if isinstance(w, int):
                allowed.add(self._learner_names[w])
            else:
                allowed.add(w)
        return [n for n in names if n in allowed]

    def bnames(self):
        """Alias for :meth:`base_learner_names`."""
        return self.base_learner_names()

    @property
    def control(self):
        """The `boost_control` this model was fitted with.

        The 4-tuple `(mstop, nu, risk, trace)` that `boost_control` returns.
        A property, matching the LSS classes; the method form `fit.control()`
        still works and warns.
        """
        from mboostrs._compat import still_callable
        from mboostrs._control import BoostControl

        value = BoostControl(*self._control)
        return still_callable(value,
                              call=lambda: BoostControl(*self._control),
                              deprecation="control-call")

    def unavailable_operations(self, output="records"):
        """Operations this model cannot perform, and why.

        Empty for a model fitted in this session, and for one unpickled or
        loaded from a full artifact. A model loaded from a scoring-only
        artifact carries the fitted path but not the call that produced it,
        so everything that refits is unavailable.
        Asking here beats finding out five times at five call sites, with a
        different message each time.
        """
        from mboostrs._output import output_frame
        from mboostrs._refit import unavailable_operations

        rows = unavailable_operations(self)
        return rows if output == "records" else output_frame(rows, output)

    def execution_report(self) -> ExecutionReport | None:
        """Return the versioned numerical execution report for this fit."""
        return self._execution_report

    def model_weights(self):
        """Return observation weights used for fitting."""
        if self._weights is not None:
            return self._weights.copy()
        return np.ones(len(self._fitted))

    def model_frame(self):
        """Return the data used for fitting as a dict of arrays."""
        if self._data is None:
            raise ValueError("model_frame() requires stored data")
        if isinstance(self._data, dict):
            return {k: np.asarray(v, dtype=np.float64).ravel() for k, v in self._data.items()}
        result = {}
        for col in self._data.columns:
            result[col] = np.asarray(self._data[col], dtype=np.float64).ravel()
        return result

    def with_mstop(self, k):
        """Return a copy truncated to ``k`` boosting iterations.

        Equivalent to ``model[:k]`` but more readable.
        """
        return self[:k]

    def survfit(self, newdata=None):
        """Breslow baseline-survival curve for a fitted CoxPH model.

        Matches R's ``mboost:::survFit.mboost``. Requires ``family=CoxPH()``
        and unit case weights (the Breslow estimator used here, like R's,
        does not yet support weighted fits).

        Parameters
        ----------
        newdata : optional
            If given, returns one survival curve per row of ``newdata``
            (evaluated at ``predict(newdata=newdata, type="link")``) instead
            of the single baseline curve.

        Returns
        -------
        dict with keys ``"time"`` (event times), ``"n_event"`` (event count
        at each time), and ``"surv"`` (survival matrix, one row per event
        time, one column per curve).
        """
        # Lazy: `glmboost` imports this module, so these cannot be resolved at
        # import time.
        from mboostrs._rust import breslow_curves

        baseline = self._breslow_or_raise()
        linpred_new = (
            np.asarray(self.predict(newdata=newdata, type="link"), dtype=np.float64)
            if newdata is not None
            else None
        )
        surv = np.asarray(breslow_curves(
            np.ascontiguousarray(baseline["event_times"], dtype=np.float64),
            np.ascontiguousarray(baseline["n_event"], dtype=np.float64),
            np.ascontiguousarray(baseline["cumhaz"], dtype=np.float64),
            float(baseline["center"]),
            None if linpred_new is None else np.ascontiguousarray(linpred_new),
        ))
        return {
            "time": np.asarray(baseline["event_times"], dtype=np.float64),
            "n_event": np.asarray(baseline["n_event"], dtype=np.float64),
            "surv": surv,
        }

    def survival_at(self, times, newdata=None):
        """Predicted survival probability S(t | x) at fixed times, per subject.

        Requires a CoxPH model with a ``Surv(time, status)`` response. The
        Breslow baseline cumulative hazard ``H0(t)`` is a right-continuous step
        function: it is the cumulative hazard at the last event time at or
        before ``t``, and zero before the first event. A subject's survival is
        ``exp(-H0(t) * exp(eta - mean(eta_train)))``, where ``eta`` is the
        linear predictor of ``newdata`` (the training rows when ``newdata`` is
        None) and the mean is over the training linear predictor the baseline
        was centered on.

        Parameters
        ----------
        times : number or array-like
            Non-negative, finite horizons on the time scale of the response.
        newdata : dict or DataFrame, optional
            Subjects to predict. The response columns are not read.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_subjects, n_times)``. Values lie in [0, 1] and are
            non-increasing along each row, because ``H0`` is non-decreasing.
        """
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("times must be a non-empty finite one-dimensional sequence")
        if np.any(times < 0):
            raise ValueError("times must be non-negative")
        linpred = np.asarray(self.predict(newdata=newdata, type="link"), dtype=np.float64)
        return self._survival_from_linpred(times, linpred)

    def _survival_from_linpred(self, times, linpred):
        from mboostrs._rust import breslow_horizon_survival

        baseline = self._breslow_or_raise()
        return np.asarray(breslow_horizon_survival(
            np.ascontiguousarray(baseline["event_times"], dtype=np.float64),
            np.ascontiguousarray(baseline["n_event"], dtype=np.float64),
            np.ascontiguousarray(baseline["cumhaz"], dtype=np.float64),
            float(baseline["center"]),
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(np.asarray(linpred, dtype=np.float64).ravel()),
        ))

    def _breslow_or_raise(self):
        """The stored Breslow baseline, after checking the model can have one."""
        from mboostrs.families import CoxPH

        if not isinstance(self._family, CoxPH):
            raise NotImplementedError(
                f"survfit() requires family=CoxPH(); got {self._family.__class__.__name__}"
            )
        if not isinstance(self._response_name, tuple):
            raise ValueError("survfit() requires a Surv(time, status) response")
        baseline = getattr(self, "_breslow_baseline", None)
        if baseline is None:
            if self._weights is not None and not np.allclose(self._weights, 1.0):
                raise NotImplementedError("survfit() cannot (yet) deal with weights")
            raise ValueError(
                "this model has no stored Breslow baseline. It was either saved before "
                "the baseline was stored, or truncated from a model that was itself "
                "loaded without its training rows. Refit with glmboost() or gamboost() "
                "at the mstop you need"
            )
        return baseline

    def plot_risk(self):
        """Plot the risk path.

        Returns
        -------
        alt.Chart
        """
        from mboostrs.plot import plot_risk

        return plot_risk(self)

    def plot_selected(self):
        """Plot selection frequency per base-learner.

        Returns
        -------
        alt.Chart
        """
        from mboostrs.plot import plot_selected

        return plot_selected(self)

    def plot_varimp(self, sort=True):
        """Plot variable importance as a horizontal bar chart.

        Parameters
        ----------
        sort : bool
            Sort by importance descending (most important at top).

        Returns
        -------
        alt.Chart
        """
        from mboostrs.plot import plot_varimp

        return plot_varimp(self, sort=sort)

    def plot_residuals(self):
        """Plot residuals vs fitted values.

        Returns
        -------
        alt.Chart
        """
        from mboostrs.plot import plot_residuals

        return plot_residuals(self)


def _stopping_report_for(model) -> dict:
    """Implement :meth:`ModelAPIMixin.stopping_report` for any fitted model.

    Kept module-level so ``LSSModel`` (which does not share the mixin) can
    offer the same report. Reads only the stored in-sample risk path.
    """
    import numpy as _np

    risk = _np.asarray(model.risk_, dtype=float).ravel()
    ms = len(risk) - 1
    report = {
        "mstop": int(ms),
        "best_step": None,
        "risk_at_best": None,
        "risk_at_final": float(risk[-1]) if ms >= 0 else None,
        "risk_rose_after_best": False,
        "flag": None,
        "glm_gap_nats": None,
        "glm_gap_material": None,
    }
    if ms < 1 or not _np.isfinite(risk).any():
        report["flag"] = "path too short to judge stopping"
        return report

    finite = _np.isfinite(risk)
    best = int(_np.argmin(_np.where(finite, risk, _np.inf)))
    risk_best = float(risk[best])
    tol = 1e-8 * max(1.0, abs(risk_best))
    rose = bool(_np.any(risk[best + 1:] > risk_best + tol)) if best < ms else False

    report["best_step"] = best
    report["risk_at_best"] = risk_best
    report["risk_rose_after_best"] = rose

    if best < 0.5 * ms and rose:
        # The #220 signature: the path's best step is far behind the cap and
        # the risk climbs afterwards, so more steps at this nu cannot repair
        # it. A validation rule that stops on the first flat stretch books
        # the shrunk fit.
        report["flag"] = (
            f"in-sample risk was best at step {best} of {ms} and rose "
            "afterwards; further steps at this nu do not approach the "
            "unpenalized fit — try a smaller nu or step='line_search'"
        )
    elif best >= ms - 1:
        report["flag"] = (
            f"in-sample risk was still falling at the mstop cap ({ms}); "
            "consider increasing mstop"
        )

    try:
        from mboostrs._same_family import same_family_gap

        gap = same_family_gap(model)
    except Exception:
        gap = None
    if gap is not None and gap.available:
        report["glm_gap_nats"] = gap.gap
        report["glm_gap_material"] = bool(gap.material)
        if gap.material:
            extra = (
                f"in-sample logLik at mstop is {gap.gap:.2f} nats short of "
                f"the same-family {gap.comparison}; stopping left a shrunk "
                "predictor, not the MLE"
            )
            report["flag"] = extra if report["flag"] is None else report["flag"] + "; " + extra
    return report
