"""Distribution-free prediction intervals for fitted models.

An LSS model carries a fitted distribution, so `prediction_interval()` reads
its centiles directly. A standard `gamboost`/`glmboost` fit models only the
conditional mean, and `confint()` describes uncertainty about *effects* — it
says nothing about where the next observation will fall.

Split conformal and CQR target finite-sample marginal coverage

    P(Y_new in C(X_new)) >= level

without a distributional assumption. The specification and stopping rule must
be fixed without using calibration responses. CV+ has a different finite-sample
lower bound, generally below its nominal level. Results report the applicable
bound and assumptions. Coverage is marginal over X, not conditional at one x.

Three methods:

* ``"split"`` — fit on half the data, calibrate scores on the other half.
  Absolute scores give constant width; count scores can vary with the mean.
* ``"cv+"`` — K-fold; uses every observation for calibration at K extra fits.
  Its fold-specific predictions can give width that varies with x.
* ``"cqr"`` — conformalized quantile regression (Romano, Patterson & Candès,
  2019). Fits two `Quantile` models and conformalizes the gap between them,
  giving intervals whose **width varies with x**. This is the one worth
  reaching for when the noise is heteroscedastic.

References
----------
Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*.
Lei et al. (2018), "Distribution-Free Predictive Inference for Regression".
Barber, Candès, Ramdas & Tibshirani (2021), "Predictive inference with the
jackknife+" (CV+).
Romano, Patterson & Candès (2019), "Conformalized Quantile Regression".
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

import numpy as np

from mboostrs import _rust
from mboostrs._extrapolation import routine_warnings
from mboostrs._names import quote_name
from mboostrs._output import export_result

__all__ = ["ConformalInterval", "conformal_interval"]


def _f64(values) -> np.ndarray:
    """A flat float64 array, the form the Rust conformal core takes."""
    return np.ascontiguousarray(values, dtype=np.float64).ravel()


def _vec(values) -> np.ndarray:
    """A Rust-returned vector as a float64 array (the bindings return lists)."""
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class ConformalInterval:
    """Prediction intervals with their calibration evidence attached."""

    lower: np.ndarray
    upper: np.ndarray
    predicted: np.ndarray
    level: float
    method: str
    n_calibration: int
    quantile: float | None = field(default=None)
    guaranteed_coverage: float | None = None
    guarantee: str = ""
    assumptions: tuple[str, ...] = ()
    n_training: int | None = None
    n_folds: int | None = None
    #: Where each calibration fit's step count came from: tuned by
    #: cross-validation inside its own proper-training rows, fixed by the
    #: caller, or the model's fitted path. The last is clean only if that
    #: path was not chosen using the rows later used for calibration.
    mstop_source: str = ""
    #: How the calibration rows were chosen: a random split of the model's
    #: training rows, or rows the caller held out before fitting.
    calibration_source: str = ""
    #: Training rows CV+ left out so that every fold has the same size.
    n_set_aside: int = 0
    #: The full model's prediction at each interval row, on the response
    #: scale (the location parameter `mu` for a distributional model). It is
    #: the model's own fitted value, not the centre of the interval:
    #: ``predicted`` comes from the proper-training refit that calibrated the
    #: interval, so the two differ by the refit's estimation noise. Unavailable
    #: for response-transformed fits: an inverse location need not be a mean.
    fitted_mean: np.ndarray | None = None
    #: Standard-model split score; normalized corrections use residual-scale
    #: units, while absolute and quantile corrections use response units.
    score: str | None = None
    requested_score: str | None = None
    support: str = "real"
    reference_distribution: str | None = None
    zero_probability: float | None = None
    reference_dispersion: float | None = None
    randomization_seed: int | None = None

    def __post_init__(self):
        if self.support not in ("real", "nonnegative_integers"):
            raise ValueError("support must be 'real' or 'nonnegative_integers'")
        arrays = {}
        for name in ("lower", "upper", "predicted"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"{name} must be a non-empty one-dimensional array")
            if np.any(np.isnan(value)):
                raise ValueError(f"{name} must not contain NaN")
            value = value.copy()
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)
        if len({value.shape for value in arrays.values()}) != 1:
            raise ValueError("lower, upper and predicted must have the same shape")
        if np.any(arrays["lower"] > arrays["upper"]):
            raise ValueError("every conformal lower endpoint must be <= its upper endpoint")
        _require_level(self.level)
        if self.guaranteed_coverage is not None and not 0 <= self.guaranteed_coverage <= 1:
            raise ValueError("guaranteed_coverage must be in [0, 1] or None")
        if self.fitted_mean is not None:
            mean = np.asarray(self.fitted_mean, dtype=np.float64).ravel()
            if mean.shape != arrays["lower"].shape:
                raise ValueError("fitted_mean must have one value per interval row")
            mean = mean.copy()
            mean.setflags(write=False)
            object.__setattr__(self, "fitted_mean", mean)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def count_size(self) -> np.ndarray | None:
        """Number of nonnegative integers covered, including zero for empty sets."""
        if self.support != "nonnegative_integers":
            return None
        return _vec(_rust.conformal_count_cardinalities(_f64(self.lower), _f64(self.upper)))

    def to_dict(self, *, include_arrays=False):
        """A plain-Python record; lower, upper and predicted arrays only on request.

        Nonfinite numerical values are preserved for backward compatibility;
        use ``to_json()`` for strict JSON that distinguishes them from nulls. The
        ``"provenance"`` entry carries the library and environment; it has no
        data fingerprint because the interval holds no training rows."""
        extras = {"n": len(self.lower), "mean_width": float(np.mean(self.width))}
        if include_arrays and self.support == "nonnegative_integers":
            extras["count_size"] = self.count_size.tolist()
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("lower", "upper", "predicted"),
            extras=extras,
            with_provenance=True,
        )

    def to_json(self, *, include_arrays=False, indent=None):
        """Strict JSON preserving infinities, NaN and genuine missing values.

        Decode with ``mboostrs.load_result_json``. The versioned envelope
        records nonfinite categories separately from its null placeholders.
        ``include_arrays=True`` includes row endpoints and predictions.
        """
        from mboostrs.result_json import dump_result_json

        return dump_result_json(self.to_dict(include_arrays=include_arrays), indent=indent)

    def contains(self, observed) -> np.ndarray:
        observed = np.asarray(observed, dtype=np.float64).ravel()
        if observed.shape[0] != self.lower.shape[0]:
            raise ValueError(
                f"observed has {observed.shape[0]} rows, intervals have "
                f"{self.lower.shape[0]}"
            )
        contained = (observed >= self.lower) & (observed <= self.upper)
        if self.support == "nonnegative_integers":
            contained &= np.isfinite(observed) & (observed >= 0) & (observed == np.floor(observed))
        return contained

    def empirical_coverage(self, observed) -> float:
        return float(np.mean(self.contains(observed)))

    def to_frame(self, output="auto"):
        """The interval as a table with one row per interval row.

        Columns: ``predicted`` (the interval centre), ``lower``, ``upper``,
        ``width`` (``upper - lower``), ``median``, ``level`` (the coverage
        level, repeated on each row), and ``fitted_mean`` when the interval
        carries it. ``median`` repeats ``predicted`` under the name that
        ``prediction_interval`` gives its centre column. For
        ``method="distributional"`` that centre is the fitted distribution's
        median. For split and CV+ it is the point function used for calibration;
        for CQR it is the midpoint of the calibrated band. Inverting a response
        transform does not in general produce a conditional mean or median.
        """
        columns = {
            "predicted": self.predicted,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "median": self.predicted,
            "level": np.full(self.lower.shape, float(self.level)),
        }
        if self.support == "nonnegative_integers":
            count_size = self.count_size
            columns["count_size"] = count_size
            columns["empty_count_set"] = count_size == 0
        if self.fitted_mean is not None:
            columns["fitted_mean"] = self.fitted_mean
        from mboostrs.datasets import _frame  # same output= convention

        return _frame(columns, output)

    def __repr__(self) -> str:
        return (
            f"ConformalInterval(method={self.method!r}, level={self.level}, "
            f"n={self.lower.size}, n_calibration={self.n_calibration}, "
            f"guaranteed_coverage={self.guaranteed_coverage}, "
            f"mean_width={float(np.mean(self.width)):.4g})"
        )


# --------------------------------------------------------------------------
# refitting helpers
# --------------------------------------------------------------------------




def _as_column_dict(frame) -> dict[str, np.ndarray]:
    """A dict of 1-D column arrays from a dict, pandas or polars frame."""
    if isinstance(frame, dict):
        return {k: np.asarray(v).ravel() for k, v in frame.items()}
    return {col: np.asarray(frame[col]).ravel() for col in frame.columns}


def _training_data(model) -> dict[str, np.ndarray]:
    data = getattr(model, "_data", None)
    if data is None:
        raise ValueError(
            "conformal intervals need the training data, but this model did "
            "not retain it"
        )
    return _as_column_dict(data)


def _is_lss(model) -> bool:
    return type(model).__name__ == "LSSModel" or hasattr(model, "param_names")


def _check_supported_model(model) -> None:
    """Conformal intervals are for a model with one prediction per row.

    A distributional fit predicts a mapping -- one vector per distribution
    parameter -- and there is no single residual to rank, so a conformity
    score is not defined for it without first saying which quantity the
    interval is about. Handed one, the scoring reached `float()` on a dict and
    raised `TypeError: float() argument must be a string or a real number, not
    'dict'`, which describes the crash rather than the reason.

    `glmboostLSS` and `gamboostLSS` report `prediction_interval` and
    `centiles` from the fitted distribution directly, which is the answer a
    caller asking for an interval on a distributional fit wants.
    """
    kind = type(model).__name__
    if _is_lss(model):
        raise NotImplementedError(
            f"method='split', 'cv+' and 'cqr' are not defined for a "
            f"distributional fit: {kind} predicts one vector per distribution "
            f"parameter, so there is no single residual to rank. Use "
            f"method='distributional', which conformalizes the fitted "
            f"distribution's own quantiles."
        )


def _check_refittable(model) -> None:
    """Conformal calibration refits on a subset of the rows.

    Everything that makes such a subset invalid is named by the model's own
    refit specification, so the refusal says which learner or setting is the
    problem. This used to check for `Buser` alone, while `Bmrf`, an
    externally supplied constraint matrix, an interaction recipe, and
    row subsampling all have the same difficulty.
    """
    from mboostrs._refit import require_row_subsettable

    require_row_subsettable(model, "conformal calibration", refit_preprocessing=True)


def _warn_if_weighted(model) -> None:
    """Name the sampling assumption that a weight vector cannot settle."""
    if getattr(model, "_input_weights", None) is None:
        return
    warnings.warn(
        "this model was fitted with case weights. Calibration scores are "
        "unweighted, so the reported guarantee applies only if rows remain "
        "exchangeable for the target population and weights are fixed inputs "
        "to a permutation-symmetric fitting algorithm. Frequency, survey, "
        "inverse-probability, or outcome-derived weights require their own "
        "sampling argument; dropping weights alone would not prove validity.",
        UserWarning,
        stacklevel=3,
    )


def _subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def _ordered(lower, upper):
    """Sort crossed quantile predictions (Chernozhukov, Fernandez-Val and
    Galichon's rearrangement for two quantiles): the lower one is the smaller.
    A fixed function of the fits, so the calibration scores stay
    exchangeable."""
    lo, hi = _rust.conformal_ordered_pair(_f64(lower), _f64(upper))
    return _vec(lo), _vec(hi)


def _quantile_refit(model, rows, tau, seed, mstop):
    """The model's specification refitted to the `tau` quantile on `rows`.

    The step count the model was tuned to is a Gaussian (or other) loss's;
    the check loss of a quantile moves by a step of `nu` times a sign and
    needs more steps, so reusing it under-fitted both quantiles and widened
    the conformal band. Unless the caller fixes `mstop`, each quantile's step
    count is cross-validated on the training rows alone, doubling the grid
    while the risk still falls; the calibration rows are not used, which is
    what the coverage guarantee needs.
    """
    from mboostrs.families import Quantile

    family = Quantile(tau=tau)
    if isinstance(mstop, int):
        return _refit(model, None, rows=rows, family=family,
                      control=_refit_control(model, mstop))
    return _cv_tuned_refit(model, rows, seed, family=family)


_TUNING_CAPS = ContextVar("conformal_tuning_caps", default=None)


@contextmanager
def _tuning_warnings():
    """Report a stopping cap once per public calibration call."""
    token = _TUNING_CAPS.set(set())
    try:
        yield
    finally:
        _TUNING_CAPS.reset(token)


def _cv_tuned_refit(model, rows, seed, family=None):
    """The specification refitted on `rows`, stopped where 5-fold CV on those
    rows alone puts the minimum.

    The grid starts at the model's own step count (at least 1600) and doubles
    while the cross-validated risk is still falling at its end, up to
    `_QUANTILE_MSTOP_CAP`. No other row's response enters the choice, which
    is what makes the subsequent calibration clean of the stopping decision.
    """
    from mboostrs.cvrisk import cv_weights
    from mboostrs.gamboost import GAMBoostModel

    control = model._control
    largest = _rust.conformal_tuned_mstop_start(int(control.mstop))
    folds = cv_weights(len(rows), "kfold", _TUNE_FOLDS, seed=0 if seed is None else int(seed))
    # One fit on the proper-training rows is continued to each grid end, and the
    # per-fold fits and predictions are kept between grid ends (#578). The stop
    # rule is unchanged: a grid that ends at its optimum doubles, up to the cap.
    fit = _refit(model, None, rows=rows, family=family, control=control._replace(mstop=largest))
    fold_cache = {}
    while True:
        if fit._control.mstop != largest:
            fit = fit.continue_to(largest)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # A boundary optimum is what triggers the next doubling, so the
            # search-range warning is raised here only when the cap is reached.
            warnings.filterwarnings("ignore", message=".*search range was exhausted.*",
                                    category=UserWarning)
            # Option 2: a centred or scaled response keeps the transform learned on
            # `rows` for every fold; per-fold relearning is what the refit mode refuses.
            transformed = getattr(fit, "_response_transform", None) is not None
            if transformed:
                cv = fit.cvrisk(folds=folds, preprocessing="fixed")
            else:
                # Only the GAM refit path keeps per-fold fits between calls.
                extra = {"_path_cache": fold_cache} if isinstance(fit, GAMBoostModel) else {}
                cv = fit.cvrisk(folds=folds, preprocessing="refit", **extra)
        if not cv.at_boundary() or largest >= _QUANTILE_MSTOP_CAP:
            warned = _TUNING_CAPS.get()
            if cv.at_boundary() and (warned is None or largest not in warned):
                warnings.warn(
                    f"CQR quantile step count reached the cap {_QUANTILE_MSTOP_CAP}; "
                    "the cross-validated risk was still falling there.",
                    UserWarning, stacklevel=2)
                if warned is not None:
                    warned.add(largest)
            return fit[: _rust.conformal_tuned_mstop_choice(int(cv.optimal_mstop))]
        largest = _rust.conformal_tuned_mstop_next(largest, _QUANTILE_MSTOP_CAP)


_TUNED_SOURCE = ("tuned by 5-fold cross-validation inside each proper-training "
                 "set; calibration responses unused")


def _mstop_source(mstop, model) -> str:
    if mstop == "tune":
        return _TUNED_SOURCE
    if mstop is not None:
        return f"fixed by the caller (mstop={int(mstop)})"
    return (f"the model's fitted path ({int(model._control.mstop)} steps); "
            "clean only if it was not tuned on the calibration rows")


def _refit_control(model, mstop):
    """The control override for a refit at a caller-fixed step count, or None.

    None keeps the refit specification's own control. A refit must run to the
    requested step count: predicting at a count beyond the refit's own path
    silently returns the path's last step, so the intervals would describe a
    different model from the one requested.
    """
    if isinstance(mstop, int):
        return model._control._replace(mstop=int(mstop))
    return None


def _require_fitted_steps(model, mstop):
    """Refuse a held-out calibration that asks for more steps than were fitted.

    A held-out calibration uses the caller's model as fitted and never refits
    it, so there are no steps to add.
    """
    fitted = int(model._control.mstop)
    if isinstance(mstop, int) and mstop > fitted:
        raise ValueError(
            f"mstop={mstop} is beyond the model's fitted {fitted} steps. A "
            "held-out calibration uses the model as fitted and cannot add "
            "steps; refit the model with that mstop, or use method='split' "
            "or 'cv+', which refit with it.")


def _fit_for(model, rows, seed, mstop):
    """A proper-training fit, tuned inside `rows` when `mstop == "tune"`."""
    if mstop == "tune":
        return _cv_tuned_refit(model, rows, seed), None
    return _refit(model, None, rows=rows, control=_refit_control(model, mstop)), mstop


#: Largest step count the CQR quantile fits search.
_QUANTILE_MSTOP_CAP = 10_000

#: Inner folds of the cross-validation that picks a tuned refit's step count.
_TUNE_FOLDS = 5


def conformal_fit_count(model, *, method="split", mstop=None, n_folds=10,
                        calibration=None) -> int:
    """Model fits one `conformal_interval` call performs, from its arguments.

    A tuned refit (``mstop="tune"``, or any CQR quantile fit whose ``mstop``
    is not an integer) costs one fit plus `_TUNE_FOLDS` inner fits for its
    cross-validation. That is the first pass; the step grid doubles while the
    risk still falls, and each doubling adds the same cost again, so the count
    is a lower bound for tuned calls. ``calibration=`` uses the fitted model
    without refitting, so it costs nothing.

    The count covers the boosting refits. The uneven-coverage warning also
    fits one small direct-REML scale model to the calibration rows (``split``
    and ``cv+`` with at least `_SCALE_MODEL_MIN_ROWS` of them and a usable
    numeric covariate), which this count leaves out.
    """
    if calibration is not None:
        return 0
    if method not in ("split", "distributional", "cqr", "cv+", "cv_plus"):
        raise ValueError(f"method must be 'split', 'cv+', or 'cqr', got {method!r}")
    tuned = 1 + _TUNE_FOLDS
    if _is_lss(model) and method in ("split", "distributional"):
        return 1
    if method == "split":
        return tuned if mstop == "tune" else 1
    if method in ("cv+", "cv_plus"):
        return int(n_folds) * (tuned if mstop == "tune" else 1)
    # CQR: two quantile fits, each tuned unless the caller fixed an integer.
    fixed = isinstance(mstop, int) and not isinstance(mstop, bool)
    return 2 * (1 if fixed else tuned)


def _refit(model, data: dict[str, np.ndarray], *, family=None, control=None,
           rows=None):
    """Refit the model's own specification, on ``data`` or on ``rows`` of it.

    Goes through the model's `RefitSpec`, so every hyperparameter the fit
    accepted is carried: the weights, `center`, `env`, `bins`, explicitly
    supplied learners, and the subsampling settings. Rebuilding the call from
    the formula, data, family and control alone -- which this did -- produced
    a calibration model that differed from the fitted one in whichever of
    those the caller had set, and reported the difference as a coverage
    guarantee.

    ``rows`` subsets the specification, which keeps the weights aligned with
    the data; ``data`` is accepted for the callers that already hold a subset
    frame, and then carries the full-length weights forward only if the frame
    is full length.
    """
    from mboostrs._refit import refit_spec

    spec = refit_spec(model)
    if rows is not None:
        spec = spec.subset(rows, refit_preprocessing=True)
    else:
        spec = replace(spec, data=data)
        weights = spec.arguments.get("weights")
        if weights is not None and len(np.asarray(weights)) != _n_rows(data):
            raise ValueError(
                "conformal refit was given a data subset without the matching "
                "row indices, so the model's weights cannot be aligned with "
                "it. This is an internal error; pass rows= instead."
            )

    overrides = {}
    if family is not None:
        overrides["family"] = family
    if control is not None:
        overrides["control"] = control
    return spec.build(**overrides)


def _n_rows(data) -> int:
    for value in data.values():
        return len(np.asarray(value).ravel())
    return 0


#: Bytes of fold predictions held at once, so the (folds x target rows)
#: matrix passed to the core stays near 64 MB.
_CV_PLUS_POOL_BYTES = 64 * 1024 * 1024


def _cv_plus_chunk(n_folds: int, n_new: int) -> int:
    """How many target rows to hold at once, at least one and at most all.

    The core takes each target row's endpoints independently, so the target
    rows can be passed in blocks without changing the result.
    """
    per_row = max(n_folds * 8, 1)
    return int(np.clip(_CV_PLUS_POOL_BYTES // per_row, 1, max(n_new, 1)))


def _subset_columns(target, start: int, stop: int):
    """Rows ``start:stop`` of a prediction frame."""
    from mboostrs.stabsel import _take_rows

    return _take_rows(target, np.arange(start, stop))


def _original_response_name(model):
    step = getattr(model, "_response_transform", None)
    return step.source if step is not None else model._response_name


def _response(model, data: dict[str, np.ndarray]) -> np.ndarray:
    """Observed response in the common, original units used for calibration."""
    name = _original_response_name(model)
    if not isinstance(name, str):
        raise NotImplementedError(
            "conformal intervals are defined for a single numeric response; "
            f"this model has response {name!r}"
        )
    if name not in data:
        raise ValueError(f"conformal calibration needs the original response column {name!r}")
    return np.asarray(data[name], dtype=np.float64).ravel()


def _encode(model, newdata, unseen_levels="error"):
    """Rows as the training frame holds them: the model's predictor recipe
    (factor codes, imputation, derived columns) applied once.

    Calibration rows come from that frame, already encoded, so new rows are
    brought to the same coding before any refit predicts them.
    """
    if newdata is None:
        return None
    if hasattr(model, "_replay_newdata"):
        return model._replay_newdata(newdata, unseen_levels=unseen_levels)
    from mboostrs.glmboost import _replay_transforms

    return _replay_transforms(model, newdata, unseen_levels=unseen_levels)


@contextmanager
def _recipe_bypassed(fit):
    """Predict already-encoded rows without replaying the recipe on them."""
    saved = getattr(fit, "_transform_recipe", None)
    if saved is None:
        yield
        return
    fit._transform_recipe = None
    try:
        yield
    finally:
        fit._transform_recipe = saved


def _policy_kwargs(model, extrapolation, unseen_levels="error") -> dict:
    """The ``extrapolation=`` and ``unseen_levels=`` keywords for `model.predict`
    when the model accepts them."""
    import inspect

    params = inspect.signature(model.predict).parameters
    out = {}
    if "extrapolation" in params:
        out["extrapolation"] = extrapolation
    if "unseen_levels" in params:
        out["unseen_levels"] = unseen_levels
    return out


def _predict(fit, newdata, mstop=None, extrapolation="warn", unseen_levels="error") -> np.ndarray:
    """Response-scale predictions at rows already in the training coding."""
    with _recipe_bypassed(fit):
        return np.asarray(
            fit.predict(newdata=newdata, mstop=mstop, type="response",
                        **_policy_kwargs(fit, extrapolation, unseen_levels)), dtype=np.float64
        ).ravel()


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The finite-sample-valid order statistic of the calibration scores.

    The core takes the ceil((n+1) * level)-th smallest score. That +1 is what
    makes the coverage guarantee hold at finite n rather than only
    asymptotically; if it exceeds n the guarantee is unattainable and the
    interval is infinite.
    """
    return float(_rust.conformal_quantile(_f64(scores), float(level)))


def _require_level(level: float) -> None:
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")


def _min_calibration_size(level: float) -> int:
    """Smallest calibration set that can attain `level` at all."""
    return int(_rust.conformal_min_calibration_size(float(level)))


_BASE_ASSUMPTIONS = (
    "training and future rows are exchangeable",
    "the fitting algorithm is permutation symmetric",
    "the specification, transformations, hyperparameters and mstop were fixed without calibration responses",
)
_COUNT_ASSUMPTIONS = _BASE_ASSUMPTIONS + (
    "auxiliary uniforms are independent of responses; coverage averages over randomization",
)
_COUNT_GUARANTEE = "finite-sample marginal split-conformal coverage over observations and auxiliary randomness"


def _cv_plus_guaranteed_coverage(n: int, k: int, level: float) -> float | None:
    """Balanced K-fold CV+ lower bound from Barber et al. (2021); None when
    n is not a multiple of k."""
    return _rust.conformal_cv_plus_guaranteed_coverage(int(n), int(k), float(level))


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------


def _absolute_residuals(y, predicted) -> np.ndarray:
    return _vec(_rust.conformal_absolute_residuals(_f64(y), _f64(predicted)))


def _symmetric_band(point, q):
    lower, upper = _rust.conformal_symmetric_band(_f64(point), float(q))
    return _vec(lower), _vec(upper)


def _quantile_scores(lower, upper, y) -> np.ndarray:
    return _vec(_rust.conformal_quantile_scores(_f64(lower), _f64(upper), _f64(y)))


def _widen(lower, upper, q):
    lo, hi = _rust.conformal_widen_band(_f64(lower), _f64(upper), float(q))
    return _vec(lo), _vec(hi)


def _midpoint(lower, upper) -> np.ndarray:
    return _vec(_rust.conformal_band_midpoint(_f64(lower), _f64(upper)))


def _clip(values, lower, upper) -> np.ndarray:
    return _vec(_rust.conformal_clip_to_band(_f64(values), _f64(lower), _f64(upper)))


#: A covariate third whose calibration coverage sits this many binomial
#: standard errors below nominal is reported.
_UNEVEN_COVERAGE_Z = -4.0

#: Thirds of each covariate's calibration rows, ordered by covariate value.
_COVARIATE_BINS = 3
_BIN_NAMES = ("lowest", "middle", "highest")


#: Calibration rows a fitted scale model needs before its thirds are read.
#: Below this, one REML term per covariate says more about the fit than about
#: the noise, and the thirds are too few rows wide to compare with the
#: binomial margin above.
_SCALE_MODEL_MIN_ROWS = 60

#: Covariates the fitted scale model keeps on a wide frame. The column count
#: drives the REML design's width, so a frame with more numeric columns than
#: this keeps the ones most associated with the log residual, by the same
#: value-and-distance views the covariate thirds read.
_SCALE_MODEL_MAX_COLUMNS = 12


def _fitted_scale(predictions, y, columns):
    """The fitted log-residual scale of the calibration rows, or None.

    ``log(residual^2)`` is regressed on the numeric covariates by direct
    REML -- the scale model #543's proposed check names -- so scale
    structure that no single covariate's thirds isolate (several weak
    drivers, noise that grows with the fitted mean) is pooled into one
    fitted value per row. Binning the coverage by that fit sees the
    high-scale rows a constant-width band quietly loses: on the study's
    heteroscedastic draw the top true-scale third covered 0.78 against a
    nominal 0.90 while every overall number looked right. The rows the
    residual is read from are the rows the fit is made on, so the fit
    absorbs a small share of the realized noise; the binomial margin of
    the check stays far above it. Returns None whenever the fit is not
    available: the coverage warning is a diagnostic and must not break
    the interval it diagnoses.
    """
    if predictions is None or y.size < _SCALE_MODEL_MIN_ROWS:
        return None
    splittable = [np.asarray(column, dtype=np.float64) for column in columns
                  if np.unique(column).size > 2]
    if not splittable:
        return None
    target = _vec(_rust.conformal_scale_target(_f64(y), _f64(predictions)))
    if len(splittable) > _SCALE_MODEL_MAX_COLUMNS:
        ranked = _rust.conformal_rank_scale_columns(
            np.ascontiguousarray(np.vstack(splittable)), target, _SCALE_MODEL_MAX_COLUMNS)
        splittable = [splittable[i] for i in ranked]
    frame = {"_log_resid_sq": target}
    terms = []
    for index, column in enumerate(splittable):
        name = f"_x{index}"
        frame[name] = column
        terms.append(f"bbs({quote_name(name)}, df=4)")
    try:
        from mboostrs.direct_gam import gam

        fit = gam(f"_log_resid_sq ~ {' + '.join(terms)}", frame)
        if not fit.convergence.get("converged", True):
            return None
        return np.asarray(fit.predict(frame, extrapolation="allow"),
                          dtype=np.float64).ravel()
    except Exception:  # noqa: BLE001 -- a diagnostic must not break the interval
        return None


def _warn_uneven_coverage(model, calib_data, lower, upper, level, label="split-conformal",
                          predictions=None):
    """Warn when a constant-width band covers some calibration rows too rarely.

    A marginal guarantee allows the band to be right on average and wrong for
    part of the covariate range. Each numeric covariate is cut into thirds of
    the calibration rows, once by its value and once by its distance from its
    median, so noise that grows toward both tails is also seen. A third
    covering far below nominal is reported with its coverage.

    A second view reads the same margin off a scale model fitted to the
    calibration rows (`_fitted_scale`): coverage thirds of its fitted log
    residual scale, which pools heteroscedasticity that no single
    covariate's thirds isolate. Both views report into one warning.
    """
    response = _original_response_name(model)
    y = _response(model, calib_data)
    flagged = []
    numeric = []
    for name, values in calib_data.items():
        if name == response:
            continue
        x = np.asarray(values)
        if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
            continue
        x = x.astype(np.float64)
        numeric.append(x)
        for view, column in ((name, x), (f"|{name} - median|", np.abs(x - np.median(x)))):
            z, bin_index, coverage, count = _rust.conformal_worst_covariate_bin(
                column, lower, upper, y, level, _COVARIATE_BINS)
            if z < _UNEVEN_COVERAGE_Z:
                flagged.append(f"{view} ({_BIN_NAMES[bin_index]} third: coverage "
                               f"{coverage:.2f} over {count} rows)")
    scale = _fitted_scale(predictions, y, numeric)
    if scale is not None:
        z, bin_index, coverage, count = _rust.conformal_worst_covariate_bin(
            scale, lower, upper, y, level, _COVARIATE_BINS)
        if z < _UNEVEN_COVERAGE_Z:
            flagged.append(f"fitted scale ({_BIN_NAMES[bin_index]} third: coverage "
                           f"{coverage:.2f} over {count} rows)")
    if flagged:
        warnings.warn(
            f"{label} coverage is uneven, against a nominal {level}: "
            f"{'; '.join(flagged)}. The band has one width, so it does not "
            "follow noise that changes across the data. Use "
            "method='cqr' for that case.",
            UserWarning, stacklevel=3)


def _split_scores(y, predicted, level, score):
    if score == "absolute":
        return _absolute_residuals(y, predicted)
    return _vec(_rust.conformal_poisson_scores(_f64(y), _f64(predicted), level, score))


def _split_band(predicted, q, level, score):
    if score == "absolute":
        return _symmetric_band(predicted, q)
    lower, upper = _rust.conformal_poisson_band(_f64(predicted), level, q, score)
    return _vec(lower), _vec(upper)


def _calibrated_split_band(fit, y_cal, calibration_prediction, point, level, seed,
                           mstop, extrapolation, score, count_reference):
    """Argument/result plumbing; reference fitting and calibration run in Rust."""
    metadata = {}
    if score == "count_auto":
        from mboostrs.families import _family_rust_name

        # Resolve eligibility from this fit's rows, never calibration outcomes.
        score = _rust.conformal_count_default_score(
            _family_rust_name(fit.family), _f64(_response(fit, _training_data(fit))))
    if score in ("count_auto", "count_rank"):
        training_y = _response(fit, _training_data(fit))
        training_prediction = _predict_raw(fit, None, mstop, extrapolation)
        weights = getattr(fit, "_weights", None)
        if weights is None:
            weights = np.ones(training_y.size)
        pi, ratio, automatic = _rust.conformal_count_reference(
            _f64(training_y), _f64(training_prediction), _f64(weights),
            count_reference == "zero_inflated_poisson")
        metadata.update(reference_distribution=count_reference, zero_probability=pi,
                        reference_dispersion=ratio)
        if score == "count_auto":
            score = automatic
        if score == "count_rank":
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                                     or not 0 <= int(seed) <= 2**53-1):
                raise ValueError("count randomization seed must be an integer in [0, 2^53-1]")
            actual_seed, cal_uniforms = _rust.conformal_count_uniforms(
                y_cal.size, None if seed is None else int(seed), 0)
            _, query_uniforms = _rust.conformal_count_uniforms(point.size, actual_seed, 1)
            scores = _vec(_rust.conformal_count_rank_scores(
                _f64(y_cal), _f64(calibration_prediction), pi, _f64(cal_uniforms), level))
            q = _conformal_quantile(scores, level)
            lower, upper = _rust.conformal_count_rank_band(
                _f64(point), pi, q, _f64(query_uniforms), level)
            metadata.update(randomization_seed=actual_seed, support="nonnegative_integers")
            return _vec(lower), _vec(upper), q, score, metadata
    scores = _split_scores(y_cal, calibration_prediction, level, score)
    q = _conformal_quantile(scores, level)
    lower, upper = _split_band(point, q, level, score)
    if score != "absolute":
        metadata["support"] = "nonnegative_integers"
    return lower, upper, q, score, metadata


def _split_conformal(model, newdata, level, seed, calib_fraction, mstop, extrapolation="warn",
                     score="absolute", count_reference="zero_inflated_poisson",
                     unseen_levels="error"):
    data = _training_data(model)
    y = _response(model, data)
    n = y.size
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_calib = _rust.conformal_calibration_count(n, float(calib_fraction))

    if n_calib < _min_calibration_size(level):
        raise ValueError(
            f"level={level} needs at least {_min_calibration_size(level)} "
            f"calibration points; this split has {n_calib}. Lower the level, "
            f"raise calib_fraction, or use method='cv+'."
        )
    calib_idx, train_idx = perm[:n_calib], perm[n_calib:]

    fit, mstop = _fit_for(model, train_idx, seed, mstop)
    calib_data = _subset(data, calib_idx)
    if (getattr(fit, "_transform_recipe", None)
            or getattr(fit, "_response_transform", None) is not None):
        # The refit learned its predictor recipe on proper-training rows.
        # Replay that recipe on raw rows, never the original fit's coding.
        raw = _raw_refit_frame(model, n)
        calib_data = _subset(raw, calib_idx)
        calib_pred = _predict_raw(fit, calib_data, mstop, extrapolation, unseen_levels)
        point = _predict_raw(fit, raw if newdata is None else newdata, mstop,
                             extrapolation, unseen_levels)
    else:
        calib_pred = _predict(fit, calib_data, mstop, extrapolation, unseen_levels)
        target = _encode(model, newdata, unseen_levels) if newdata is not None else data
        point = _predict(fit, target, mstop, extrapolation, unseen_levels)
    lower, upper, q, score, metadata = _calibrated_split_band(
        fit, _response(model, calib_data), calib_pred, point, level, seed,
        mstop, extrapolation, score, count_reference)
    if score == "absolute":
        _warn_uneven_coverage(model, calib_data, *_symmetric_band(calib_pred, q), level,
                              predictions=calib_pred)

    return ConformalInterval(
        lower=lower, upper=upper, predicted=point, level=level,
        method="split", n_calibration=n_calib, quantile=q, score=score,
        guaranteed_coverage=level,
        guarantee=(_COUNT_GUARANTEE
                   if score == "count_rank" else "finite-sample marginal split-conformal coverage"),
        assumptions=_COUNT_ASSUMPTIONS if score == "count_rank" else _BASE_ASSUMPTIONS,
        n_training=n - n_calib,
        calibration_source=f"random split of the model's {n} training rows (seed={seed})",
        **metadata,
    )


def _calibration_response(calibration, name, level):
    """Validate the observed calibration response without inspecting fit rows."""
    if not isinstance(name, str):
        raise NotImplementedError(
            "conformal intervals are defined for a single numeric response; "
            f"this model has response {name!r}")
    names = calibration if isinstance(calibration, dict) else getattr(calibration, "columns", ())
    if name not in names:
        raise ValueError(f"calibration rows have no response column {name!r}")
    y_cal = np.asarray(calibration[name], dtype=np.float64).ravel()
    if not np.all(np.isfinite(y_cal)):
        raise ValueError(f"calibration response {name!r} must contain only finite observations")
    n_calib = y_cal.size
    if n_calib < _min_calibration_size(level):
        raise ValueError(
            f"level={level} needs at least {_min_calibration_size(level)} "
            f"calibration rows; got {n_calib}")
    return y_cal


def _held_out_split(model, newdata, level, mstop, calibration, extrapolation="warn",
                    score="absolute", count_reference="zero_inflated_poisson", seed=None,
                    unseen_levels="error"):
    """Split conformal with caller-held-out calibration rows and no refit."""
    _require_fitted_steps(model, mstop)
    y_cal = _calibration_response(calibration, _original_response_name(model), level)
    n_calib = y_cal.size
    point = _predict_raw(model, newdata, mstop, extrapolation, unseen_levels)
    lower, upper, q, score, metadata = _calibrated_split_band(
        model, y_cal, _predict_raw(model, calibration, mstop, extrapolation, unseen_levels), point,
        level, seed, mstop, extrapolation, score, count_reference)
    steps = int(mstop) if mstop is not None else int(model._control.mstop)
    return ConformalInterval(
        lower=lower, upper=upper, predicted=point, level=level,
        method="split", n_calibration=n_calib, quantile=q, score=score,
        guaranteed_coverage=level,
        guarantee=(_COUNT_GUARANTEE
                   if score == "count_rank" else "finite-sample marginal split-conformal coverage"),
        assumptions=_COUNT_ASSUMPTIONS if score == "count_rank" else _BASE_ASSUMPTIONS,
        n_training=len(next(iter(_training_data(model).values()))),
        mstop_source=f"the model as fitted ({steps} steps), chosen without the calibration rows",
        calibration_source="rows supplied by the caller, held out before the model was specified and fitted",
        **metadata,
    )


def _distributional_band(model, rows, level, mstop, extrapolation, unseen_levels="error"):
    """Original-response lower quantile, median and upper quantile."""
    params = model.params(rows, type="response", mstop=mstop, extrapolation=extrapolation,
                          unseen_levels=unseen_levels)
    shape = np.asarray(next(iter(params.values()))).shape
    quantiles = [model._family.quantile(np.full(shape, p), params)
                 for p in ((1-level)/2, .5, (1+level)/2)]
    response_step = getattr(model, "_response_transform", None)
    if response_step is not None:
        from mboostrs.transforms import response_inverse

        quantiles = [response_inverse(response_step, q) for q in quantiles]
    return tuple(_f64(q) for q in quantiles)


def _normalized_distributional_band(calibration, query, response, level):
    """Thin argument/result conversion for the shared-core location-scale score."""
    scores = _rust.conformal_location_scale_scores(
        _f64(response), _f64(calibration["mu"]), _f64(calibration["sigma"]))
    q = _conformal_quantile(_vec(scores), level)
    lower, upper = _rust.conformal_location_scale_band(_f64(query["mu"]), _f64(query["sigma"]), q)
    return _vec(lower), _vec(query["mu"]), _vec(upper), q


def _held_out_distributional(model, newdata, level, mstop, calibration, extrapolation,
                             score="absolute", unseen_levels="error"):
    """Calibrate an already selected LSS fit without refitting or splitting it."""
    family = model._family
    if (getattr(family, "_survival", False) or getattr(family, "_multivariate", False)
            or getattr(family, "_known_variance", False)):
        raise NotImplementedError("held-out distributional calibration needs a single observed response")
    budget = model.stopping_budget()
    limit = (budget.total if budget.scheme == "noncyclic" else
             min((count for count in budget.per_parameter.values() if count), default=0))
    if mstop is not None and mstop > limit:
        raise ValueError(f"mstop={mstop} is beyond this LSS model's fitted {limit} steps; "
                         "held-out calibration cannot add steps")
    response_step = getattr(model, "_response_transform", None)
    name = response_step.source if response_step is not None else model._response_name
    y = _calibration_response(calibration, name, level)

    if score == "gaussian_normalized":
        calibration_params = model.params(calibration, type="response", mstop=mstop,
                                          extrapolation=extrapolation,
                                          unseen_levels=unseen_levels)
        query_params = model.params(newdata, type="response", mstop=mstop,
                                    extrapolation=extrapolation,
                                    unseen_levels=unseen_levels)
        low, middle, high, q = _normalized_distributional_band(calibration_params, query_params, y, level)
    else:
        low, _, high = _distributional_band(model, calibration, level, mstop, extrapolation,
                                            unseen_levels)
        q = _conformal_quantile(_quantile_scores(low, high, y), level)
        low, middle, high = _distributional_band(model, newdata, level, mstop, extrapolation,
                                                 unseen_levels)
        low, high = _widen(low, high, q)
    return ConformalInterval(
        low, high, _clip(middle, low, high), level, "distributional", y.size, quantile=q, score=score,
        guaranteed_coverage=level,
        guarantee=("finite-sample marginal split-conformal coverage with standardized residuals"
                   if score == "gaussian_normalized" else
                   "finite-sample marginal split-conformal coverage of the fitted distribution's central band"),
        assumptions=_BASE_ASSUMPTIONS,
        n_training=len(next(iter(_training_data(model).values()))),
        mstop_source="the fitted distributional path, chosen without the calibration rows",
        calibration_source="rows supplied by the caller, held out before the model was specified and fitted",
    )


def _raw_refit_frame(model, n):
    """Raw retained rows, aligned with the indices used for calibration."""
    from mboostrs._refit import refit_spec

    source = refit_spec(model).raw_data
    if source is None:
        raise NotImplementedError("conformal refit prediction needs retained raw training inputs")
    frame = _as_column_dict(source)
    if not frame or any(len(values) != n for values in frame.values()):
        raise NotImplementedError(
            "conformal refit prediction needs raw inputs aligned with the retained training rows")
    return frame


def _predict_raw(fit, rows, mstop=None, extrapolation="warn", unseen_levels="error"):
    """Original-response point function using this fit's own fitted recipes.

    Inverting a fitted transformed location defines a point function for
    calibration; it does not in general recover an original-response mean.
    """
    values = fit.predict(newdata=rows, mstop=mstop, type="response",
                         **_policy_kwargs(fit, extrapolation, unseen_levels))
    step = getattr(fit, "_response_transform", None)
    if step is not None:
        from mboostrs.transforms import response_inverse

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = response_inverse(step, values)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"conformal response inversion for {step.expression!r} needs "
                "finite predictions in the inverse transform's domain")
    return np.asarray(values, dtype=np.float64).ravel()


def _cv_plus(model, newdata, level, seed, n_folds, mstop, extrapolation="warn",
             unseen_levels="error"):
    data = _training_data(model)
    y = _response(model, data)
    n = y.size
    if not (2 <= n_folds <= n):
        raise ValueError(f"n_folds must be in [2, n={n}], got {n_folds}")
    if n < _min_calibration_size(level):
        raise ValueError(
            f"level={level} needs at least {_min_calibration_size(level)} "
            f"observations; got {n}"
        )

    # Barber et al.'s finite-K bound is stated for folds of equal size. When
    # n is not a multiple of K, n mod K randomly chosen rows are left out of
    # the calibration; the rows kept are still exchangeable with a test point,
    # so the bound applies to them.
    rng = np.random.default_rng(seed)
    fold_size = n // n_folds
    kept = rng.permutation(n)[: fold_size * n_folds]
    folds = kept.reshape(n_folds, fold_size)
    n_set_aside = n - kept.size
    n_kept = kept.size
    raw = _raw_refit_frame(model, n)
    target = raw if newdata is None else newdata

    # The full-data fit CV+ needs for its point prediction is the model that
    # was passed in. Refitting it -- twice, as this did, once to learn the
    # prediction width and once for the point estimate -- repeats the whole
    # boosting path for a result already in hand.
    if isinstance(mstop, int) and mstop > int(model._control.mstop):
        # A step count beyond the model's own path needs a full-data refit to that
        # count; predicting the passed-in model there would clip to its last step.
        full = _refit(model, None, rows=np.arange(n), control=_refit_control(model, mstop))
        point = _predict_raw(full, target, mstop, extrapolation, unseen_levels)
    else:
        point = _predict_raw(model, target, None if mstop == "tune" else mstop,
                             extrapolation, unseen_levels)
    n_new = point.size
    tune = mstop == "tune"
    if tune:
        mstop = None

    # Fold k is held out and its model is fitted on the other kept rows. The
    # out-of-fold residual R_i of each kept row, and the fold model's
    # prediction mu_k(x) at every target row x, are the inputs the core
    # combines into the CV+ endpoints (Barber et al. 2021).
    residual_blocks = []
    prediction_blocks = []
    fold_fits = []
    for fold in folds:
        mask = np.zeros(n, dtype=bool)
        mask[kept] = True
        mask[fold] = False
        rows = np.flatnonzero(mask)
        fit = (_cv_tuned_refit(model, rows, seed) if tune
               else _refit(model, None, rows=rows, control=_refit_control(model, mstop)))
        held = _subset(raw, fold)
        held_pred = _predict_raw(fit, _subset(raw, fold), mstop, extrapolation, unseen_levels)
        prediction_blocks.append(held_pred)
        residual_blocks.append(_absolute_residuals(_response(model, held), held_pred))
        fold_fits.append(fit)
    # `folds` is the kept rows in order, fold by fold, so the concatenated
    # residuals and the fold index of each position line up with `kept`.
    residuals = np.concatenate(residual_blocks)
    # The same covariate-coverage check the split path runs, on the kept rows:
    # each row's out-of-fold prediction with the residual quantile as its band.
    kept_predictions = np.concatenate(prediction_blocks)
    _warn_uneven_coverage(model, _subset(raw, kept), *_symmetric_band(
        kept_predictions, _conformal_quantile(residuals, level)), level,
        label="CV+", predictions=kept_predictions)
    fold_of = np.repeat(np.arange(n_folds, dtype=np.int64), fold_size)

    lower = np.empty(n_new, dtype=np.float64)
    upper = np.empty(n_new, dtype=np.float64)
    chunk = _cv_plus_chunk(n_folds, n_new)
    for start in range(0, n_new, chunk):
        stop = min(start + chunk, n_new)
        fold_predictions = np.vstack([
            _predict_raw(fit, _subset_columns(target, start, stop), mstop,
                         extrapolation, unseen_levels)
            for fit in fold_fits
        ])
        lo, hi = _rust.conformal_cv_plus_bounds(residuals, fold_of, fold_predictions, float(level))
        lower[start:stop] = _vec(lo)
        upper[start:stop] = _vec(hi)

    guaranteed = _cv_plus_guaranteed_coverage(n_kept, n_folds, level)
    return ConformalInterval(
        lower=lower, upper=upper, predicted=point, level=level,
        method="cv+", n_calibration=n_kept, quantile=None,
        guaranteed_coverage=guaranteed,
        guarantee="balanced K-fold finite-sample CV+ lower bound",
        assumptions=_BASE_ASSUMPTIONS, n_training=n_kept - fold_size,
        n_folds=n_folds, n_set_aside=n_set_aside,
        calibration_source=(
            f"{n_kept} of the model's {n} training rows in {n_folds} folds of "
            f"{fold_size} (seed={seed})"
            + (f"; {n_set_aside} rows set aside so the folds have equal size"
               if n_set_aside else "")),
    )


def _cqr(model, newdata, level, seed, calib_fraction, mstop, extrapolation="warn",
         unseen_levels="error"):
    data = _training_data(model)
    y = _response(model, data)
    n = y.size
    alpha = 1.0 - level
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_calib = _rust.conformal_calibration_count(n, float(calib_fraction))
    if n_calib < _min_calibration_size(level):
        raise ValueError(
            f"level={level} needs at least {_min_calibration_size(level)} "
            f"calibration points; this split has {n_calib}."
        )
    calib_idx, train_idx = perm[:n_calib], perm[n_calib:]

    raw = _raw_refit_frame(model, n)
    lo_fit = _quantile_refit(model, train_idx, alpha / 2.0, seed, mstop)
    hi_fit = _quantile_refit(model, train_idx, 1.0 - alpha / 2.0, seed, mstop)
    if mstop == "tune":
        mstop = None

    raw_calib = _subset(raw, calib_idx)
    y_calib = _response(model, raw_calib)
    lo_calib, hi_calib = _ordered(_predict_raw(lo_fit, raw_calib, mstop, extrapolation, unseen_levels),
                                  _predict_raw(hi_fit, raw_calib, mstop, extrapolation, unseen_levels))
    # Romano et al.'s conformity score: how far outside the predicted band the
    # observation fell (negative when comfortably inside).
    scores = _quantile_scores(lo_calib, hi_calib, y_calib)
    q = _conformal_quantile(scores, level)

    target = raw if newdata is None else newdata
    lo, hi = _ordered(_predict_raw(lo_fit, target, mstop, extrapolation, unseen_levels),
                      _predict_raw(hi_fit, target, mstop, extrapolation, unseen_levels))
    # A negative correction narrows the band, and where the two quantiles are
    # closer than 2|q| it inverts it: the conformal set is then empty. The
    # core replaces an inverted row by its midpoint, which contains the empty
    # set, so the coverage guarantee still holds.
    lo, hi = _widen(lo, hi, q)
    return ConformalInterval(
        lower=lo, upper=hi, predicted=_midpoint(lo, hi), level=level,
        method="cqr", n_calibration=n_calib, quantile=q,
        guaranteed_coverage=level,
        guarantee="finite-sample marginal split-CQR coverage",
        assumptions=_BASE_ASSUMPTIONS, n_training=n - n_calib,
        calibration_source=f"random split of the model's {n} training rows (seed={seed})",
    )


#: Recipe steps that compute no parameter from the values, so a refit on a
#: subset of the rows reads them as the full fit did.
_UNESTIMATED_STEPS = ("factor", "dummy", "datetime", "impute_level", "missing_indicator",
                       "observed_indicator")


def _preprocessing_refit_spec(model, fitted):
    """The model's spec, for refitting its transforms on a subset of its rows.

    `fitted` lists the recipe steps whose parameters (a median, a log floor, a
    centre) are estimated from the values. The refit needs the inputs before
    those transforms, and each step's input must be a column of those inputs
    or the output of another step; a step that fails either test is named
    alone, since that is the one the caller has to change.
    """
    from mboostrs._refit import refit_spec, require_row_subsettable

    # A median recorded for a column that was complete at fit time is re-estimated
    # with the rest, but it is not named: it is not what a refusal is about.
    recipe = getattr(model, "_transform_recipe", None)
    indicated = {step.sources for step in (recipe.steps if recipe else ())
                 if step.kind == "missing_indicator"}
    names = [f"{step.kind}({step.name})" for step in fitted
             if step.kind not in ("impute", "impute_model") or step.sources in indicated]
    spec = refit_spec(model)
    if spec.raw_formula is None or spec.raw_data is None:
        raise NotImplementedError(
            "method='distributional' re-estimates the fitted transformations "
            f"{'(' + ', '.join(names) + ') ' if names else ''}"
            "on the proper-training rows, which needs the "
            "model's inputs before those transforms. This model did not keep "
            "them: it has an interaction term (%X%) or an env= matrix. Fit it "
            "again without them to calibrate a distributional interval.")
    raw_columns = set(_as_column_dict(spec.raw_data))
    recipe = getattr(model, "_transform_recipe", None)
    produced = {step.name for step in (recipe.steps if recipe else ())}
    for step in fitted:
        for source in step.sources:
            if source not in raw_columns and source not in produced:
                raise NotImplementedError(
                    "method='distributional' cannot re-estimate the transform "
                    f"{step.kind}({step.name}) on the proper-training rows: its "
                    f"input {source!r} is neither a column of the training data "
                    "nor the output of another transform. Add that column to the "
                    "data before fitting.")
    return require_row_subsettable(model, "conformal calibration",
                                   refit_preprocessing=True)


def _distributional(model, newdata, level, seed, calib_fraction, mstop, extrapolation="warn",
                    score="absolute", unseen_levels="error"):
    """Split conformal calibration of an LSS fit's own quantile band.

    The specification is refitted on the proper-training rows; at each
    calibration row the fitted distribution gives the alpha/2 and 1 - alpha/2
    quantiles, and the CQR score max(q_lo - y, y - q_hi) measures how far the
    observation fell outside that band. Widening the band by the conformal
    quantile of those scores gives finite-sample marginal coverage whatever
    the family: a correctly specified family needs little correction, a wrong
    one is corrected by the data (Romano, Patterson & Candes 2019, with the
    quantiles read off the fitted distribution, as in Chernozhukov, Wuthrich
    & Zhu 2021).

    When the formula has fitted transformations (an imputation median, a log
    floor), the refit estimates them again from the proper-training rows, so
    the calibration rows never shape the transforms the band is built with.
    """
    from mboostrs._refit import refit_spec

    family = model._family
    if (getattr(family, "_survival", False) or getattr(family, "_multivariate", False)
            or getattr(family, "_known_variance", False)):
        raise NotImplementedError(
            f"method='distributional' needs a single observed response; "
            f"{type(family).__name__} has a censored, multivariate or "
            f"known-variance response.")
    recipe = getattr(model, "_transform_recipe", None)
    fitted = [step for step in (recipe.steps if recipe else ())
              if step.kind not in _UNESTIMATED_STEPS]
    response_step = getattr(model, "_response_transform", None)
    if response_step is not None:
        # Reparse the raw formula so response centering/scaling is learned
        # only on proper-training rows and retained for inverse quantiles.
        fitted.append(response_step)
    if mstop == "tune":
        raise NotImplementedError(
            "mstop='tune' is not available for a distributional fit; pass None "
            "(each refit's full path) or an integer")

    data = _training_data(model)
    y = np.asarray(data[model._response_name], dtype=np.float64).ravel()
    n = y.size
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_calib = _rust.conformal_calibration_count(n, float(calib_fraction))
    if n_calib < _min_calibration_size(level):
        raise ValueError(
            f"level={level} needs at least {_min_calibration_size(level)} "
            f"calibration points; this split has {n_calib}.")
    calib_idx, train_idx = perm[:n_calib], perm[n_calib:]
    if fitted:
        spec = _preprocessing_refit_spec(model, fitted)
        # The rows before the transforms: the refit encodes them with the
        # transforms it estimated on its own rows.
        frame = _as_column_dict(spec.raw_data)
        if next(iter(frame.values())).shape[0] != n:
            raise NotImplementedError(
                "method='distributional' needs the model's inputs and its "
                f"training rows to line up ({next(iter(frame.values())).shape[0]} "
                f"input rows, {n} training rows); the row indices of the split "
                "would refer to different rows.")
        fit = spec.subset(train_idx, refit_preprocessing=True).build()
        calib_rows = _subset(frame, calib_idx)
        name = response_step.source if response_step is not None else model._response_name
        y_calib = _calibration_response(calib_rows, name, level)
        target_rows = frame if newdata is None else newdata
    else:
        fit = refit_spec(model).subset(train_idx).build()
        y_calib = y[calib_idx]

    def band(target):
        def predict():
            if score == "gaussian_normalized":
                return fit.params(target, type="response", mstop=mstop,
                                  extrapolation=extrapolation,
                                  unseen_levels=unseen_levels)
            return _distributional_band(fit, target, level, mstop, extrapolation,
                                        unseen_levels)
        if fitted:
            return predict()
        with _recipe_bypassed(fit):
            return predict()

    # Calibration rows and query rows use the same proper-training recipe.
    cal_target = calib_rows if fitted else _subset(data, calib_idx)
    target = (target_rows if fitted
              else (_encode(model, newdata, unseen_levels) if newdata is not None else data))
    if score == "gaussian_normalized":
        lo, mid, hi, q = _normalized_distributional_band(band(cal_target), band(target), y_calib, level)
    else:
        lo_c, _, hi_c = band(cal_target)
        scores = _quantile_scores(lo_c, hi_c, y_calib)
        q = _conformal_quantile(scores, level)
        lo, mid, hi = band(target)
        lo, hi = _widen(lo, hi, q)
    names = ", ".join(f"{step.kind}({step.name})" for step in fitted)
    source = f"random split of the model's {n} training rows (seed={seed})"
    if fitted:
        source += f"; {names} re-estimated on the proper-training rows"
    return ConformalInterval(
        lower=lo, upper=hi, predicted=_clip(mid, lo, hi), level=level,
        method="distributional", n_calibration=n_calib, quantile=q, score=score,
        guaranteed_coverage=level,
        guarantee=("finite-sample marginal split-conformal coverage with standardized residuals"
                   if score == "gaussian_normalized" else
                   "finite-sample marginal split-conformal coverage of the fitted distribution's central band"),
        assumptions=_BASE_ASSUMPTIONS, n_training=n - n_calib,
        mstop_source="fixed by the caller" if mstop is not None
        else "each refit's full path, the model's fitted step count",
        calibration_source=source,
    )


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def conformal_interval(
    model,
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
    score="auto",
    count_reference="zero_inflated_poisson",
    unseen_levels="error",
):
    """Distribution-free prediction intervals for a fitted standard model.
    Usage::

        import mboostrs as mb
        fit = mb.gamboost("y ~ bbs(x, df=4)", data, family=mb.Gaussian())
        ci = mb.conformal_interval(fit, level=0.9, seed=1)
        ci.lower, ci.upper

    Parameters
    ----------
    model : GLMBoostModel, GAMBoostModel or LSSModel
        A fitted model. It is used as a *specification*: the method refits it
        on subsets of the training data, so the intervals are calibrated on
        data the corresponding fit never saw.
    newdata : frame or dict, optional
        Where to predict. Defaults to the training data.
    level : float, default 0.9
        Nominal marginal coverage. CV+ reports its generally lower
        finite-sample bound separately when folds are balanced.
    method : {"split", "cv+", "cqr", "distributional"}, default "split"
        See the module docstring. Use ``"cqr"`` for heteroscedastic noise,
        ``"cv+"`` when data are scarce, ``"split"`` when fits are expensive.
        ``"split"`` stays the default: its band has one width, so it warns
        when a covariate range or a fitted scale region is under-covered,
        and ``"cqr"`` or ``"cv+"`` are the remedies. ``"cqr"`` is the
        slowest of the three: it refits
        two quantile models, each with its own cross-validated stopping
        search, about 3.5 s against 0.01 s for ``"split"`` at n = 10,000.
        Pass ``mstop=`` to fix the step count and skip the search.
        For a ``glmboostLSS``/``gamboostLSS`` fit the method is
        ``"distributional"`` (the default ``"split"`` selects it): the fitted
        distribution's central band, from a refit on the proper-training
        rows, is widened by the conformal quantile of how far calibration
        responses fell outside it. Coverage then holds even if the family is
        wrong.
    seed : int, optional
        Seed for the split or fold assignment, and for the count-rank
        score's tie randomization when the default ``score="auto"`` resolves
        to it: the calibration rows and the query rows each draw one uniform
        per row from a stream keyed by the seed, so the same rows and seed
        reproduce the interval exactly -- an internal split and its
        equivalent held-out calibration agree only when both carry the same
        seed. ``None`` (the default) draws and reports a seed in
        ``randomization_seed`` for the count score, and draws the split or
        fold assignment, so the call is not reproducible until a seed is
        given.
    calib_fraction : float, default 0.5
        Share of training rows held out for calibration (``"split"``, ``"cqr"``).
    n_folds : int, default 10
        Number of folds (``"cv+"``).
    mstop : int or "tune", optional
        Iteration to predict at. ``None`` (default) uses each refit's full
        path, the model's fitted step count; the guarantee then assumes that
        count was not chosen using the calibration rows, which is false if
        the model was tuned by cross-validation on all its training rows.
        ``"tune"`` re-chooses the step count for every proper-training fit
        by 5-fold cross-validation on that fit's own rows, so no calibration
        response enters the stopping decision. ``method="cqr"`` always tunes
        its quantile fits this way unless an integer is given. The returned
        ``mstop_source`` records which applied.
        Inner cross-validation relearns preprocessing within each training
        fold. CV+ and CQR predict held-out and query rows through each
        corresponding refit's own recipe. Response centering/scaling currently
        requires an integer step count (or None for split/CV+): inner CV refuses
        family risks whose response units differ between folds. CQR with either
        transform therefore requires an explicit integer.
    calibration : frame or dict, optional
        Rows held out before the model's specification, tuning and fit were
        chosen, with the response and every predictor. With
        ``method="split"``, the model is used as fitted, without a refit: the
        calibration scores use ``score`` on these rows. Every
        choice made on the fitting rows (formula search, family, ``mstop``)
        is then free of calibration responses, which is what the guarantee
        needs and what a refit of a specification chosen on all rows cannot
        provide. The caller is responsible for these rows never having been
        used to fit or select the model; ``calibration_source`` records that
        claim.
        For an LSS model, ``"split"`` and ``"distributional"`` calibrate its
        fitted central quantiles, applying its existing predictor recipe to
        the held-out rows without learning it again. Neither path refits or
        retunes the model. Distributional calibration also supports a response
        transform by returning the quantiles to the original response scale.
        In that transformed-response case ``predicted`` is the original-scale
        median clipped to the band, and ``fitted_mean`` is unavailable: the
        inverse-transformed location is generally not a response mean.
    extrapolation : {"warn", "clip", "error", "allow"}, default "warn"
        Policy for rows whose spline covariates lie outside the training range
        of the refit or model that predicts them, as in ``predict``. It applies
        to the interval's own predictions, the calibration predictions and the
        ``fitted_mean``, so they all use one policy. ``"clip"`` holds such a
        covariate at the nearest training value, which bounds the interval and
        the fitted mean; ``"warn"`` continues the spline's edge slope.
    score : str, default "auto"
        Nonconformity score for standard-model split calibration. Automatic
        scoring uses count ranks for standard Poisson/NBinomial fits with an
        untransformed count response, and absolute scoring for other models.
        Its reference and eligibility are learned on proper-training rows.
        A proper-training residual-to-reference variance ratio above 2 uses
        normalized scoring instead. Explicit legacy scores keep their definitions.
        The
        normalized count score divides absolute residuals by
        ``sqrt(max(predicted_mean, 1e-12))`` and multiplies the calibrated
        correction by that scale at prediction. The quantile score calibrates
        the central Poisson quantile band at the fitted mean. Both require
        nonnegative integer calibration responses and finite nonnegative
        predicted means. They retain marginal conformal validity without
        assuming the Poisson variance model is correct; conditional coverage
        is not guaranteed. Endpoints remain real-valued. Intersect every
        compared method with known count support when measuring count-set
        size. These options require an untransformed response and do not
        apply to CV+, CQR or LSS models. ``quantile`` uses standardized
        residual units for the normalized score and response units otherwise.
        ``count_rank`` uses a mean-preserving fitted count CDF and independent
        auxiliary uniforms. Its correction is in probability-rank units, its
        bounds are real and its target is nonnegative integers. The result
        records ``support``, ``count_size``, ``zero_probability``,
        ``reference_dispersion`` and ``randomization_seed``. A singleton below
        zero represents an empty count set. Coverage averages over responses
        and auxiliary randomness; it does not promise conditional coverage.
        Supply ``seed`` in [0, 2**53-1] to reproduce the randomization.
        ``gaussian_normalized`` is an experimental GaussianLSS split score:
        absolute residual divided by the fitted positive scale, inverted as
        location plus/minus the calibration cutoff times scale. It requires an
        untransformed response and uses standardized residual units. Its
        calibration is marginal and does not identify an irreducible noise scale.
    count_reference : {"zero_inflated_poisson", "poisson"}, default "zero_inflated_poisson"
        Reference for count-rank scores. The first fits one zero-inflation
        probability on proper-training rows while retaining the fitted mean.
        The second fixes that probability at zero. Neither assumes that the
        reference distribution is correct for marginal conformal validity.
    unseen_levels : {"error", "average", "zero_effect"}, default "error"
        Policy for factor or group levels present in ``newdata`` or
        ``calibration`` that were absent from training, as in ``predict``.

    Returns
    -------
    ConformalInterval

    Notes
    -----
    Coverage is marginal over a fresh exchangeable test point, not training
    coverage and not conditional coverage at each x. The default absolute
    split score gives constant width; Poisson scores, CV+ and CQR can vary
    with x. The specification and mstop
    must be fixed without calibration responses: pass ``calibration=`` with
    rows held out before the model was chosen to make that hold by
    construction. Read ``guarantee``, ``guaranteed_coverage``,
    ``assumptions``, ``mstop_source`` and ``calibration_source`` on the
    returned object.

    ``predicted`` is the point prediction attached to the interval. For ``method="split"``,
    ``"cqr"`` and ``"distributional"`` it comes from the proper-training refit,
    the fit on the rows not used for calibration: the refit's prediction for
    split, the midpoint of its two quantile fits for CQR, and its fitted
    median for distributional. For ``method="cv+"`` and for ``calibration=``
    it is the full model's prediction. ``fitted_mean`` holds the full model's
    prediction at the same rows, whatever the method, so the two can be
    compared. With ``score="poisson_quantile"``, ``predicted`` remains the
    fitted mean and need not be the centre of the calibrated quantile band.
    With a response transform, bounds, calibration scores and ``predicted``
    use original-response units. Each proper-training fit inverts predictions
    with its own fitted response parameters before scoring the raw observed
    response. CQR orders the inverse-transformed candidate functions before
    calibration. Response centering/scaling and predictor preprocessing are
    re-estimated on proper-training rows. ``fitted_mean`` is None for transformed
    fits: the original-response expectation is not in general the inverse of
    the fitted location. A standard fit's inverse prediction is a point
    function; a distributional fit supplies an original-response median.
    Inversion outside a transform's domain, or a nonfinite inverse prediction,
    raises ValueError. Standard ``predict(type='response')`` retains its usual
    transformed-response meaning; this conversion belongs to calibration.

    Every method treats rows as exchangeable. Rows from the same cluster,
    repeated measurements of one unit, and time-ordered rows are not: a
    calibration row from a cluster that also appears in training understates
    the error for a new cluster, and a later time point is not drawn like an
    earlier one. None of the methods here applies to such data, and none
    accepts a grouping or time argument.

    CV+ needs folds of equal size for its finite-K bound. When the number of
    rows is not a multiple of ``n_folds``, the remainder (at most
    ``n_folds - 1`` rows, chosen at random with ``seed``) is left out of the
    calibration and reported as ``n_set_aside``.

    Examples
    --------
    >>> import numpy as np, mboostrs as mb
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 500)
    >>> data = {"x": x, "y": np.sin(x) + rng.normal(scale=0.3, size=500)}
    >>> fit = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
    ...                   control=mb.boost_control(100, 0.1))
    >>> ci = mb.conformal_interval(fit, level=0.9, seed=1)
    >>> ci.lower.shape == ci.upper.shape == (500,)
    True
    """
    _require_level(level)
    if unseen_levels not in ("error", "average", "zero_effect"):
        raise ValueError(
            "unseen_levels must be 'error', 'average', or 'zero_effect', "
            f"got {unseen_levels!r}"
        )
    requested_score = score
    if count_reference not in ("zero_inflated_poisson", "poisson"):
        raise ValueError("count_reference must be 'zero_inflated_poisson' or 'poisson'")
    if score == "auto":
        score = "absolute"
        if (method == "split" and not _is_lss(model)
                and getattr(model, "_response_transform", None) is None):
            score = "count_auto"
    if score not in ("absolute", "poisson_normalized", "poisson_quantile", "count_auto", "count_rank",
                      "gaussian_normalized"):
        raise ValueError("score must be 'auto', 'absolute', 'poisson_normalized', 'poisson_quantile', "
                         "'count_rank' or 'gaussian_normalized'")
    if score == "gaussian_normalized":
        if (not _is_lss(model) or type(model._family).__name__ != "GaussianLSS"
                or getattr(model, "_response_transform", None) is not None
                or method not in ("split", "distributional")):
            raise ValueError("gaussian_normalized needs an untransformed GaussianLSS split/distributional fit")
    elif score != "absolute":
        if method != "split" or _is_lss(model):
            raise ValueError("Poisson scores are available only for standard-model method='split'")
        if getattr(model, "_response_transform", None) is not None:
            raise ValueError("Poisson scores require an untransformed count response")
    if not (mstop is None or mstop == "tune"
            or (isinstance(mstop, (int, np.integer)) and not isinstance(mstop, bool)
                and int(mstop) >= 1)):
        raise ValueError(f"mstop must be None, 'tune' or a positive integer, got {mstop!r}")
    if isinstance(mstop, np.integer):
        mstop = int(mstop)
    if not (0.0 < calib_fraction < 1.0):
        raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")
    if calibration is not None:
        if method not in (("split", "distributional") if _is_lss(model) else ("split",)):
            raise NotImplementedError(
                "calibration= supports split conformal on a standard model "
                "or distributional calibration of an LSS model; CV+ and CQR "
                "refit on subsets of the training rows instead")
        if mstop == "tune":
            raise ValueError(
                "mstop='tune' re-chooses the step count in refits; with "
                "calibration= the model is used as fitted, so choose its mstop "
                "on the fitting rows before calling")
        if _is_lss(model):
            _warn_if_weighted(model)
            with routine_warnings(stacklevel=2):
                interval = _held_out_distributional(
                    model, newdata, level, mstop, calibration, extrapolation, score,
                    unseen_levels=unseen_levels)
                return _with_fitted_mean(model, newdata,
                    replace(interval, requested_score=requested_score), extrapolation,
                    unseen_levels=unseen_levels)
        _check_supported_model(model)
        with routine_warnings(stacklevel=2):
            interval = _held_out_split(model, newdata, level, mstop, calibration, extrapolation,
                                       score, count_reference, seed, unseen_levels=unseen_levels)
            return _with_fitted_mean(model, newdata, replace(interval, requested_score=requested_score),
                                     extrapolation, unseen_levels=unseen_levels)
    if _is_lss(model) and method in ("split", "distributional"):
        # A distributional fit has one method: the default "split" means it.
        _warn_if_weighted(model)
        with routine_warnings(stacklevel=2):
            interval = _distributional(model, newdata, level, seed, calib_fraction, mstop,
                                       extrapolation, score, unseen_levels=unseen_levels)
            return _with_fitted_mean(model, newdata,
                replace(interval, requested_score=requested_score), extrapolation,
                unseen_levels=unseen_levels)
    if method == "distributional":
        raise ValueError("method='distributional' is for glmboostLSS/gamboostLSS fits")
    _check_supported_model(model)
    _check_refittable(model)
    _warn_if_weighted(model)

    # Every calibration fit predicts its held-out rows, and with `newdata=` a
    # target a fold leaves the training range. One summary at the end says
    # that, not a paragraph per fold.
    with routine_warnings(stacklevel=2), _tuning_warnings():
        if method == "split":
            result = _split_conformal(model, newdata, level, seed,
                                      calib_fraction, mstop, extrapolation, score,
                                      count_reference, unseen_levels=unseen_levels)
        elif method in ("cv+", "cv_plus"):
            result = _cv_plus(model, newdata, level, seed, n_folds, mstop, extrapolation,
                              unseen_levels=unseen_levels)
        elif method == "cqr":
            result = _cqr(model, newdata, level, seed, calib_fraction, mstop, extrapolation,
                          unseen_levels=unseen_levels)
        else:
            raise ValueError(
                f"method must be 'split', 'cv+', or 'cqr', got {method!r}"
            )
        # CQR always tunes its quantile fits unless the caller fixes mstop.
        source_key = "tune" if (method == "cqr" and mstop is None) else mstop
        return _with_fitted_mean(
            model, newdata, replace(result, mstop_source=_mstop_source(source_key, model),
                                    requested_score=requested_score),
            extrapolation, unseen_levels=unseen_levels)


def _with_fitted_mean(model, newdata, interval: ConformalInterval,
                      extrapolation="warn", unseen_levels="error") -> ConformalInterval:
    """`interval` with the full model's prediction at the same rows attached.

    Callers run this inside :func:`routine_warnings`, so the full model's
    extrapolation rows join the interval's rows in one summary; the policy is
    the one the interval used.
    """
    if getattr(model, "_response_transform", None) is not None:
        return interval
    with routine_warnings(stacklevel=3):
        mean = _full_model_mean(model, newdata, extrapolation, unseen_levels)
    return replace(interval, fitted_mean=mean)


def _full_model_mean(model, newdata, extrapolation="warn", unseen_levels="error") -> np.ndarray:
    """The full model's prediction at the interval rows, on the response scale.

    The rows are `newdata`, or the training rows when it is None. A
    distributional model reports its location parameter `mu`. Its
    extrapolation rows are reported under `extrapolation`, as the interval's
    own predictions are.
    """
    if _is_lss(model):
        params = model.params(newdata=newdata, type="response", extrapolation=extrapolation,
                              unseen_levels=unseen_levels)
        key = "mu" if "mu" in params else next(iter(params))
        mean = params[key]
    else:
        mean = model.predict(newdata=newdata, type="response",
                             **_policy_kwargs(model, extrapolation, unseen_levels))
    return np.asarray(mean, dtype=np.float64).ravel()
