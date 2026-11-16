"""Full-model contrasts with explicit scale, population, weights, and support."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from mboostrs._counterfactual import predict_frames, replaced_frames
from mboostrs._frame import column_names
from mboostrs._names import partition_tilde
from mboostrs._output import export_result, output_frame
from mboostrs._weights import normalize_weights

__all__ = [
    "ALE1DResult",
    "ContrastResult",
    "ElasticityResult",
    "MarginalEffectResult",
    "PredictionProfileResult",
    "average_marginal_effect",
    "accumulated_local_effect",
    "average_elasticity",
    "distributional_contrast",
    "prediction_profile",
    "prediction_contrast",
]


def _weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values, kind="stable")
    x = values[order]
    w = weights[order]
    positive = w > 0
    x, w = x[positive], w[positive]
    cumulative = (np.cumsum(w) - .5 * w) / np.sum(w)
    return np.interp(probabilities, cumulative, x, left=x[0], right=x[-1])


def _raw_values(data, name):
    values = data[name]
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values)


def _as_mapping(data):
    return {name: _raw_values(data, name).copy() for name in column_names(data)}


def _record_ids(n, observation_ids):
    if observation_ids is None:
        return np.arange(n)
    values = np.asarray(observation_ids)
    if values.shape != (n,):
        raise ValueError(f"observation_ids has shape {values.shape}, expected ({n},)")
    if len(set(values.tolist())) != n:
        raise ValueError("observation_ids must uniquely identify the supplied rows")
    return values.copy()


def _support_values(model, data, variable):
    recipe = getattr(model, "_transform_recipe", None)
    for item in getattr(recipe, "steps", ()):
        if item.kind == "factor" and item.source == variable:
            return np.asarray(item.params, dtype=object)
    training = getattr(model, "_data", None)
    source = training if training is not None and variable in column_names(training) else data
    return _raw_values(source, variable)


def _effective_value(value, training, variable, policy):
    if np.ndim(value) != 0:
        raise ValueError("reference and alternative must each be one scalar value")
    if training.dtype.kind in "iufb":
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{variable} contrast values must be finite")
        finite = np.asarray(training, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"{variable!r} has no finite training support")
        low, high = float(finite.min()), float(finite.max())
        outside = numeric < low or numeric > high
        if outside and policy == "error":
            raise ValueError(
                f"{variable}={numeric:g} is outside training support [{low:g}, {high:g}]"
            )
        if outside and policy == "warn":
            warnings.warn(
                f"{variable}={numeric:g} is outside training support [{low:g}, {high:g}]; "
                "the contrast depends on extrapolation",
                RuntimeWarning, stacklevel=3,
            )
        effective = float(np.clip(numeric, low, high)) if policy == "clip" else numeric
        return effective, {
            "kind": "numeric", "training_min": low, "training_max": high,
            "requested": numeric, "effective": effective, "outside": outside,
        }
    levels = tuple(np.unique(training).tolist())
    if value not in levels:
        raise ValueError(
            f"{variable}={value!r} is an unseen level; observed levels are {levels!r}"
        )
    return value, {
        "kind": "categorical", "levels": levels,
        "requested": value, "effective": value, "outside": False,
    }


def _predict(model, data, *, scale, parameter, mstop, extrapolation):
    arguments = {"newdata": data, "type": scale, "mstop": mstop}
    signature = inspect.signature(model.predict)
    if "parameter" in signature.parameters:
        if parameter is None:
            raise ValueError("parameter is required for a distributional model contrast")
        arguments["parameter"] = parameter
    elif parameter is not None:
        raise ValueError("parameter is only valid for a distributional model contrast")
    if "extrapolation" in signature.parameters:
        arguments["extrapolation"] = extrapolation
    predicted = model.predict(**arguments)
    if isinstance(predicted, dict):
        raise ValueError("parameter must select one distributional predictor")
    values = np.asarray(predicted, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"contrast requires one finite prediction per row, got shape {values.shape}"
        )
    return values


def _predict_stacked(model, frames, *, scale, parameter, mstop, extrapolation):
    """`_predict` of each frame, from one call per stacked chunk of frames."""
    return predict_frames(
        lambda frame, _copies: _predict(
            model, frame, scale=scale, parameter=parameter, mstop=mstop,
            extrapolation=extrapolation),
        frames,
    )


def _distribution_quantity(model, data, quantity, value, mstop):
    signature = inspect.signature(model.predict)
    if "parameter" not in signature.parameters:
        raise ValueError("distributional_contrast requires an LSS model")
    params = model.predict(newdata=data, type="response", mstop=mstop)
    if not isinstance(params, dict):
        raise ValueError("distributional model prediction did not return parameter arrays")
    family = getattr(model, "_family", None)
    if family is None:
        raise ValueError("distributional model does not expose its fitted family")
    return _quantity_from_params(family, params, quantity, value)


def _quantity_from_params(family, params, quantity, value):
    """The requested mean, quantile or exceedance probability for each row.

    `params` maps each distribution parameter to its natural-scale array, one
    value per row. Raises when the family cannot give the quantity or gives a
    value that is not finite on every row, such as an undefined or infinite
    mean."""
    if quantity == "mean":
        if value is not None:
            raise ValueError("value is not used when quantity='mean'")
        function = getattr(family, "mean", None)
        if function is None:
            raise NotImplementedError(f"{type(family).__name__} does not declare a response mean")
        result = function(params)
        label = "mean"
    elif quantity == "quantile":
        if value is None or not np.ndim(value) == 0:
            raise ValueError("quantity='quantile' requires one scalar probability in value")
        probability = float(value)
        if not np.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("quantile probability must be finite and in (0, 1)")
        function = getattr(family, "quantile", None)
        if function is None:
            raise NotImplementedError(f"{type(family).__name__} does not implement quantiles")
        result = function(probability, params)
        label = f"quantile({probability:g})"
    elif quantity == "exceedance_probability":
        if value is None or not np.ndim(value) == 0:
            raise ValueError(
                "quantity='exceedance_probability' requires one scalar threshold in value"
            )
        threshold = float(value)
        if not np.isfinite(threshold):
            raise ValueError("exceedance threshold must be finite")
        function = getattr(family, "cdf", None)
        if function is None:
            raise NotImplementedError(f"{type(family).__name__} does not implement a CDF")
        # The convention is strict exceedance, P(Y > threshold) = 1-F(threshold).
        result = 1 - np.asarray(function(threshold, params), dtype=np.float64)
        label = f"Pr(response>{threshold:g})"
    else:
        raise ValueError(
            "quantity must be 'mean', 'quantile', or 'exceedance_probability'"
        )
    result = np.asarray(result, dtype=np.float64)
    n = len(next(iter(params.values())))
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{label} must produce one finite value per row, got shape {result.shape}"
        )
    return result, label


_JOINT_POLICIES = ("report", "warn", "error", "off")
_JOINT_METHOD = "same_categorical_cell_and_nearest_standardized_training_row"
_CLUSTER_WARNING_BELOW = 10

# Columns of each ``to_frame`` table, so a zero-row result still returns a
# frame with these names. ``ElasticityResult`` appends its ``kind`` column.
_ROW_VALUE_COLUMNS = (
    "record_id", "weight", "lower_value", "upper_value",
    "lower_prediction", "upper_prediction",
)
CONTRAST_COLUMNS = (
    "record_id", "weight", "reference_prediction", "alternative_prediction",
    "contrast",
)
MARGINAL_EFFECT_COLUMNS = _ROW_VALUE_COLUMNS + ("marginal_effect",)
ALE1D_COLUMNS = (
    "bin", "lower", "upper", "grid", "effect", "local_change", "n",
    "weight_total",
)
PREDICTION_PROFILE_COLUMNS = ("grid", "average_prediction", "support_fraction_far")


def _cluster_dependence(individual, weights, cluster, *, level, n_draws, seed):
    """Cluster-level uncertainty of the weighted mean of `individual`, or None.

    Clusters are the sampling units. The mean's cluster-robust (sandwich) standard
    error and a cluster bootstrap percentile interval are computed in the Rust
    core (`mboostrs_core::contrast::cluster_average`). The fitted model is held
    fixed: the interval describes the averaging population as a sample of
    clusters, not the variability of the model fit.
    """
    if cluster is None:
        return None
    from mboostrs import _rust

    n = len(individual)
    labels = np.asarray(cluster)
    if labels.shape != (n,):
        raise ValueError(f"cluster has shape {labels.shape}, expected ({n},)")
    if labels.dtype.kind == "O" and any(v is None for v in labels.tolist()):
        raise ValueError("cluster must not contain missing values")
    _, codes = np.unique(labels, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64).ravel()
    groups = int(codes.max()) + 1
    if groups < 2:
        raise ValueError(
            "cluster uncertainty needs at least two clusters; this cluster column "
            "has one. Pass cluster=None for the row-wise result"
        )
    n_draws = int(n_draws)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")
    estimate, se_sandwich, se_bootstrap, lower, upper, clusters, draws = _rust.cluster_average(
        np.ascontiguousarray(individual, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        codes, n_draws, int(seed), float(level),
    )
    if clusters < _CLUSTER_WARNING_BELOW:
        warnings.warn(
            f"only {clusters} clusters: a bootstrap percentile interval from fewer than "
            f"{_CLUSTER_WARNING_BELOW} clusters is unreliable; read the sandwich "
            "standard error with care",
            RuntimeWarning, stacklevel=3,
        )
    return MappingProxyType({
        "method": "cluster_bootstrap", "band": "pointwise",
        "level": float(level), "clusters": int(clusters), "draws": int(draws),
        "seed": int(seed), "estimate": float(estimate),
        "se_bootstrap": float(se_bootstrap), "se_sandwich": float(se_sandwich),
        "lower": float(lower), "upper": float(upper),
        "held_fixed": "fitted model; rows are a sample of clusters",
    })


def _refit_dependence(model, refit, reference_data, alternative_data, quantity, value,
                      weights, *, level, n_draws, seed):
    """Interval of the average contrast from a `gamlss_refit` posterior, or None.

    `refit.draws` gives every distribution parameter at both settings from the
    same coefficient draws (one seed), so each draw yields one contrast in
    which the coefficient uncertainty is shared by the two settings. The
    interval is the percentile range of those contrasts. The point estimate
    stays with the boosted model; the refit's own centre is reported beside it.
    """
    if refit is None:
        return None
    if type(refit).__name__ != "GamlssRefit":
        raise TypeError("refit must be the result of mb.gamlss_refit(...)")
    family = getattr(model, "_family", None)
    if family is None:
        raise ValueError("distributional model does not expose its fitted family")
    n_draws = int(n_draws)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")
    reference = refit.draws(reference_data, n_draws=n_draws, seed=int(seed))
    alternative = refit.draws(alternative_data, n_draws=n_draws, seed=int(seed))
    contrasts = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):
        at_reference, _ = _quantity_from_params(
            family, {name: arr[d] for name, arr in reference.items()}, quantity, value)
        at_alternative, _ = _quantity_from_params(
            family, {name: arr[d] for name, arr in alternative.items()}, quantity, value)
        contrasts[d] = np.average(at_alternative - at_reference, weights=weights)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(contrasts, [tail, 1.0 - tail])
    return MappingProxyType({
        "method": "refit_draws", "band": "pointwise", "level": float(level),
        "draws": n_draws, "seed": int(seed),
        "center": float(np.median(contrasts)), "se_draws": float(np.std(contrasts, ddof=1)),
        "lower": float(lower), "upper": float(upper),
        "held_fixed": "fitted family and refit terms; coefficient draws share one seed",
    })


def _cell_codes(columns, tables, *, extend):
    """Integer codes of categorical columns, one table per column.

    With `extend`, unseen values get the next code (training rows); otherwise
    an unseen value gets -1, which matches no training cell."""
    n = len(next(iter(columns.values()))) if columns else 0
    out = np.empty((n, len(tables)), dtype=np.float64)
    for j, (name, table) in enumerate(tables.items()):
        values = np.asarray(columns[name]).tolist()
        if extend:
            out[:, j] = [table.setdefault(v, len(table)) for v in values]
        else:
            out[:, j] = [table.get(v, -1) for v in values]
    return out


def _numeric_matrix(columns, names, n):
    if not names:
        return np.empty((n, 0), dtype=np.float64)
    return np.column_stack([np.asarray(columns[name], dtype=np.float64) for name in names])


def _joint_training(model, mapping, names):
    """The model's training rows for `names` in the supplied rows' representation.

    A factor in the training frame is stored as integer codes, and the
    transform recipe holds its labels, so codes are decoded to labels. Returns
    None when the model keeps no training frame, when a name is missing from
    it, or when a column is numeric in one frame and not the other; the caller
    then falls back to the supplied rows.
    """
    frame = getattr(model, "_data", None)
    if frame is None or not all(name in column_names(frame) for name in names):
        return None
    training = _as_mapping(frame)
    recipe = getattr(model, "_transform_recipe", None)
    for item in getattr(recipe, "steps", ()):
        if item.kind != "factor" or item.source not in training:
            continue
        levels = np.asarray(item.params, dtype=object).ravel()
        try:
            codes = np.asarray(training[item.source], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if codes.size and (np.any(codes < 0) or np.any(codes >= levels.size)
                           or np.any(codes != np.floor(codes))):
            continue
        training[item.source] = levels[codes.astype(np.int64)]
    for name in names:
        if (training[name].dtype.kind in "iufb") != (mapping[name].dtype.kind in "iufb"):
            return None
    return training


def _joint_support(model, mapping, variable, variants, weights, *, threshold,
                   support_columns):
    """Share of standardization rows whose counterfactual rows are jointly observed.

    The counterfactual row of a standardization row sets `variable` to one of
    `variants` (one array per evaluation point) and keeps the row's other
    values. The counterfactual is supported when the training data has a row
    that (a) has the same values in every categorical conditioning column, and
    the same level of `variable` when `variable` is categorical, and (b) lies
    within `threshold` training standard deviations of the counterfactual on
    the numeric conditioning columns (and on `variable` when numeric). Training
    data is `model._data` when the model keeps it; otherwise the supplied rows,
    which is recorded in the result.
    """
    from mboostrs import _rust

    chosen, excluded = _support_columns(model, mapping, variable, support_columns)
    training = _joint_training(model, mapping, [variable, *chosen])
    if training is None:
        training = mapping
        source = "standardization_rows"
    else:
        source = "training_data"
    # The representation of a column follows the supplied rows: a numeric
    # column is standardized, anything else is a categorical cell member.
    numeric_names = [name for name in chosen if mapping[name].dtype.kind in "iufb"]
    cell_names = [name for name in chosen if name not in numeric_names]
    variable_numeric = mapping[variable].dtype.kind in "iufb"
    if variable_numeric:
        numeric_names = numeric_names + [variable]
    else:
        cell_names = cell_names + [variable]
    tables = {name: {} for name in cell_names}
    n_train = len(training[variable])
    train_cells = _cell_codes(training, tables, extend=True)
    train_numeric = _numeric_matrix(training, numeric_names, n_train)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    supported = np.ones(n, dtype=bool)
    for values in variants:
        # The counterfactual keeps every other column of the standardization
        # row and replaces only `variable`.
        columns = dict(mapping)
        if variable_numeric:
            columns[variable] = np.asarray(values, dtype=np.float64)
        else:
            columns[variable] = np.asarray(values, dtype=object)
        query_cells = _cell_codes(columns, tables, extend=False)
        query_numeric = _numeric_matrix(columns, numeric_names, n)
        distance = np.asarray(_rust.cell_nearest_distance(
            np.ascontiguousarray(train_cells), np.ascontiguousarray(train_numeric),
            np.ascontiguousarray(query_cells), np.ascontiguousarray(query_numeric),
        ), dtype=np.float64)
        supported &= distance <= threshold
    fraction = float(np.average(supported, weights=weights)) if n else float("nan")
    return {
        "method": _JOINT_METHOD,
        "threshold_sd": float(threshold),
        "conditioning_columns": tuple(chosen),
        "categorical_cell_columns": tuple(cell_names),
        "excluded_columns": tuple(excluded),
        "training_source": source,
        "training_rows": int(n_train),
        "supported_fraction": fraction,
        "unsupported_rows": int(np.sum(~supported)),
        "rows": int(n),
    }


def _joint_check(model, mapping, variable, variants, weights, *, policy, threshold,
                 support_columns, stacklevel):
    """Joint-support record for a contrast, applying the requested policy.

    ``"report"`` records the result; ``"warn"`` also warns when some rows are
    unsupported; ``"error"`` raises; ``"off"`` skips the computation."""
    if policy not in _JOINT_POLICIES:
        raise ValueError(f"joint_support must be one of {_JOINT_POLICIES}, got {policy!r}")
    if policy == "off":
        return {"method": "not_computed", "policy": "off"}
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("joint_threshold must be finite and positive")
    record = _joint_support(model, mapping, variable, variants, weights,
                            threshold=threshold, support_columns=support_columns)
    record["policy"] = policy
    missing = record["unsupported_rows"]
    if missing:
        message = (
            f"{missing} of {record['rows']} standardization rows set {variable!r} to a value "
            f"with no training row in the same categorical cell within {threshold:g} "
            f"standard deviations of the other predictors "
            f"{list(record['conditioning_columns'])}; those rows' contributions depend on "
            "extrapolation. Pass joint_support='report' to keep them with this record, or "
            "joint_support='off' to skip the check"
        )
        if policy == "error":
            raise ValueError(message)
        if policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
    return record


def _uncertainty_sentence(dependence, level):
    if dependence is None:
        return ("No sampling or model-selection uncertainty is attached; the "
                "contrast is a point summary of the fitted model over the rows.")
    if dependence["method"] == "refit_draws":
        return (
            f"Refit-posterior {level:.0%} interval [{dependence['lower']:.6g}, "
            f"{dependence['upper']:.6g}] from {dependence['draws']} draws of the "
            f"gamlss_refit coefficients ({dependence['band']} band). The point estimate "
            f"is the boosted model's; the refit's own centre is {dependence['center']:.6g}, "
            "and an interval that excludes the point estimate means the refit and the "
            "boosted fit disagree on this contrast. The family and the refit's terms are fixed."
        )
    return (
        f"Cluster-bootstrap {level:.0%} interval [{dependence['lower']:.6g}, "
        f"{dependence['upper']:.6g}] over {dependence['clusters']} clusters "
        f"({dependence['band']} band, cluster-robust SE {dependence['se_sandwich']:.6g}); "
        "the fitted model is held fixed."
    )


def _joint_sentence(joint):
    if not joint or joint.get("policy") == "off":
        return ""
    return (
        f" Joint support: {joint['supported_fraction']:.1%} of the weighted rows have "
        f"both counterfactual settings observed within {joint['threshold_sd']:g} sd "
        f"(unsupported rows: {joint['unsupported_rows']})."
    )


_UNCERTAINTY_KINDS = ("none", "cluster_bootstrap", "refit_draws")


def _check_uncertainty(uncertainty, dependence):
    if uncertainty not in _UNCERTAINTY_KINDS:
        raise ValueError(f"uncertainty must be one of {_UNCERTAINTY_KINDS}")
    if (uncertainty == "none") != (dependence is None):
        raise ValueError("uncertainty and dependence must agree: a result with an interval "
                         "carries its dependence record and an unintervaled one carries none")


@dataclass(frozen=True)
class ContrastResult:
    """A standardized full-model prediction contrast and its row contributions."""

    estimate: float
    individual: np.ndarray
    reference_prediction: np.ndarray
    alternative_prediction: np.ndarray
    weights: np.ndarray
    observation_ids: np.ndarray
    variable: str
    reference: object
    alternative: object
    scale: str
    parameter: str | None
    population: str
    support: dict
    uncertainty: str = "none"
    causal: bool = False
    #: Cluster-level uncertainty when `cluster=` was given, else None. See
    #: :func:`prediction_contrast`.
    dependence: dict | None = None

    def __post_init__(self):
        n = len(np.asarray(self.individual))
        for name in ("individual", "reference_prediction", "alternative_prediction",
                     "weights", "observation_ids"):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.scale not in ("link", "response"):
            raise ValueError("scale must be 'link' or 'response'")
        _check_uncertainty(self.uncertainty, self.dependence)
        object.__setattr__(self, "support", MappingProxyType(dict(self.support)))
        if self.dependence is not None:
            object.__setattr__(self, "dependence", MappingProxyType(dict(self.dependence)))

    @property
    def interval(self):
        """``(lower, upper)`` of the cluster-bootstrap interval, or None."""
        if self.dependence is None:
            return None
        return (self.dependence["lower"], self.dependence["upper"])

    def to_frame(self, output="auto"):
        """One row per standardized prediction contrast."""
        rows = [{
            "record_id": self.observation_ids[i].item()
            if isinstance(self.observation_ids[i], np.generic)
            else self.observation_ids[i],
            "weight": float(self.weights[i]),
            "reference_prediction": float(self.reference_prediction[i]),
            "alternative_prediction": float(self.alternative_prediction[i]),
            "contrast": float(self.individual[i]),
        } for i in range(len(self.individual))]
        return output_frame(rows, output, columns=CONTRAST_COLUMNS)

    def __repr__(self) -> str:
        return (
            f"ContrastResult({self.variable}: {self.reference!r} -> "
            f"{self.alternative!r}, {self.scale}={self.estimate:.6g}, "
            f"n={len(self.individual)})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row evidence only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("individual", "reference_prediction", "alternative_prediction",
                    "weights", "observation_ids"),
        )

    def describe(self) -> str:
        """State the estimand, scale, averaging population and uncertainty limit."""
        parameter = f" for parameter {self.parameter}" if self.parameter else ""
        level = self.dependence["level"] if self.dependence else 0.0
        return (
            f"Average full-model {self.scale}-scale contrast{parameter}: "
            f"set {self.variable} from {self.reference!r} to {self.alternative!r}, "
            f"estimate={self.estimate:.6g} over {self.population}. "
            f"{_uncertainty_sentence(self.dependence, level)}"
            f"{_joint_sentence(self.support.get('joint'))} "
            "This adjusted prediction contrast is not a causal effect without "
            "identification assumptions."
        )


@dataclass(frozen=True)
class MarginalEffectResult:
    """Average and row-specific finite derivatives of the full prediction."""

    estimate: float
    individual: np.ndarray
    lower_value: np.ndarray
    upper_value: np.ndarray
    lower_prediction: np.ndarray
    upper_prediction: np.ndarray
    weights: np.ndarray
    observation_ids: np.ndarray
    variable: str
    scale: str
    parameter: str | None
    population: str
    step: float
    support: dict
    uncertainty: str = "none"
    causal: bool = False
    #: Cluster-level uncertainty when `cluster=` was given, else None.
    dependence: dict | None = None

    def __post_init__(self):
        n = len(np.asarray(self.individual))
        for name in (
            "individual", "lower_value", "upper_value", "lower_prediction",
            "upper_prediction", "weights", "observation_ids",
        ):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.scale not in ("link", "response"):
            raise ValueError("scale must be 'link' or 'response'")
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("step must be finite and positive")
        _check_uncertainty(self.uncertainty, self.dependence)
        object.__setattr__(self, "support", MappingProxyType(dict(self.support)))
        if self.dependence is not None:
            object.__setattr__(self, "dependence", MappingProxyType(dict(self.dependence)))

    @property
    def interval(self):
        """``(lower, upper)`` of the cluster-bootstrap interval, or None."""
        if self.dependence is None:
            return None
        return (self.dependence["lower"], self.dependence["upper"])

    def to_frame(self, output="auto"):
        """One row per local finite derivative."""
        rows = [{
            "record_id": self.observation_ids[i].item()
            if isinstance(self.observation_ids[i], np.generic)
            else self.observation_ids[i],
            "weight": float(self.weights[i]),
            "lower_value": float(self.lower_value[i]),
            "upper_value": float(self.upper_value[i]),
            "lower_prediction": float(self.lower_prediction[i]),
            "upper_prediction": float(self.upper_prediction[i]),
            "marginal_effect": float(self.individual[i]),
        } for i in range(len(self.individual))]
        return output_frame(rows, output, columns=MARGINAL_EFFECT_COLUMNS)

    def __repr__(self) -> str:
        return (
            f"MarginalEffectResult(variable={self.variable!r}, "
            f"{self.scale}={self.estimate:.6g}, step={self.step:.4g}, "
            f"n={len(self.individual)})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row evidence only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("individual", "lower_value", "upper_value", "lower_prediction",
                    "upper_prediction", "weights", "observation_ids"),
        )

    def describe(self) -> str:
        """State the derivative target, scale, step, population, and limits."""
        parameter = f" for parameter {self.parameter}" if self.parameter else ""
        level = self.dependence["level"] if self.dependence else 0.0
        return (
            f"Average full-model derivative of {self.scale}-scale prediction{parameter} "
            f"with respect to {self.variable}: {self.estimate:.6g} over "
            f"{self.population}; requested finite-difference step={self.step:.6g}. "
            "Boundary rows use one-sided differences. "
            f"{_uncertainty_sentence(self.dependence, level)}"
            f"{_joint_sentence(self.support.get('joint'))} "
            "The derivative is not causal without identification assumptions."
        )


@dataclass(frozen=True)
class ElasticityResult:
    """Average and row-specific finite log-response derivatives."""

    estimate: float
    individual: np.ndarray
    lower_value: np.ndarray
    upper_value: np.ndarray
    lower_prediction: np.ndarray
    upper_prediction: np.ndarray
    weights: np.ndarray
    observation_ids: np.ndarray
    variable: str
    kind: str
    parameter: str | None
    population: str
    log_step: float
    support: dict
    uncertainty: str = "none"
    causal: bool = False

    def __post_init__(self):
        n = len(np.asarray(self.individual))
        for name in (
            "individual", "lower_value", "upper_value", "lower_prediction",
            "upper_prediction", "weights", "observation_ids",
        ):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.kind not in ("elasticity", "semi_elasticity"):
            raise ValueError("kind must be 'elasticity' or 'semi_elasticity'")
        if not np.isfinite(self.log_step) or self.log_step <= 0:
            raise ValueError("log_step must be finite and positive")
        object.__setattr__(self, "support", MappingProxyType(dict(self.support)))

    def to_frame(self, output="auto"):
        """One row per finite elasticity or semi-elasticity."""
        rows = [{
            "record_id": self.observation_ids[i].item()
            if isinstance(self.observation_ids[i], np.generic)
            else self.observation_ids[i],
            "weight": float(self.weights[i]),
            "lower_value": float(self.lower_value[i]),
            "upper_value": float(self.upper_value[i]),
            "lower_prediction": float(self.lower_prediction[i]),
            "upper_prediction": float(self.upper_prediction[i]),
            self.kind: float(self.individual[i]),
        } for i in range(len(self.individual))]
        return output_frame(rows, output,
                            columns=_ROW_VALUE_COLUMNS + (self.kind,))

    def __repr__(self) -> str:
        return (
            f"ElasticityResult(kind={self.kind!r}, variable={self.variable!r}, "
            f"estimate={self.estimate:.6g}, n={len(self.individual)})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row evidence only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("individual", "lower_value", "upper_value", "lower_prediction",
                    "upper_prediction", "weights", "observation_ids"),
        )

    def describe(self) -> str:
        """State the logarithmic derivative, averaging population, and limits."""
        if self.kind == "elasticity":
            definition = "d log(response prediction) / d log(variable)"
        else:
            definition = "d log(response prediction) / d variable"
        parameter = f" for parameter {self.parameter}" if self.parameter else ""
        return (
            f"Average {self.kind}{parameter}: {self.estimate:.6g} over "
            f"{self.population}; {definition}, finite log-step={self.log_step:.6g}. "
            "No sampling, selection, or causal uncertainty is attached."
        )


@dataclass(frozen=True)
class ALE1DResult:
    """A centered univariate accumulated-local-effect curve and its support."""

    variable: str
    edges: np.ndarray
    grid: np.ndarray
    effect: np.ndarray
    local_change: np.ndarray
    counts: np.ndarray
    weight_totals: np.ndarray
    bin_index: np.ndarray
    observation_effect: np.ndarray
    weights: np.ndarray
    observation_ids: np.ndarray
    scale: str
    parameter: str | None
    population: str
    support: dict

    def __post_init__(self):
        n_bins = len(np.asarray(self.effect))
        for name in ("grid", "effect", "local_change", "counts", "weight_totals"):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n_bins,):
                raise ValueError(f"{name} must have shape ({n_bins},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        edges = np.asarray(self.edges, dtype=np.float64).copy()
        if edges.shape != (n_bins + 1,) or np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing with one more value than bins")
        edges.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        n = len(np.asarray(self.observation_effect))
        for name in ("bin_index", "observation_effect", "weights", "observation_ids"):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "support", MappingProxyType(dict(self.support)))

    def to_frame(self, output="auto"):
        """One row per local interval with its effect and empirical support."""
        rows = [{
            "bin": index, "lower": float(self.edges[index]),
            "upper": float(self.edges[index + 1]), "grid": float(self.grid[index]),
            "effect": float(self.effect[index]),
            "local_change": float(self.local_change[index]),
            "n": int(self.counts[index]),
            "weight_total": float(self.weight_totals[index]),
        } for index in range(len(self.effect))]
        return output_frame(rows, output, columns=ALE1D_COLUMNS)

    def __repr__(self) -> str:
        return (
            f"ALE1DResult(variable={self.variable!r}, scale={self.scale!r}, "
            f"bins={len(self.effect)}, n={len(self.observation_effect)})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; per-bin curves always, per-row evidence on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("bin_index", "observation_effect", "weights", "observation_ids"),
        )

    def observation_frame(self, output="auto"):
        """One row per standardization observation and its assigned ALE value."""
        rows = [{
            "record_id": self.observation_ids[i].item()
            if isinstance(self.observation_ids[i], np.generic)
            else self.observation_ids[i],
            "bin": int(self.bin_index[i]), "weight": float(self.weights[i]),
            "effect": float(self.observation_effect[i]),
        } for i in range(len(self.observation_effect))]
        return output_frame(rows, output)

    def describe(self) -> str:
        """State scale, population, centering and the sparsest local interval."""
        parameter = f" for parameter {self.parameter}" if self.parameter else ""
        return (
            f"Accumulated local effect of {self.variable} on {self.scale}-scale "
            f"prediction{parameter}, centered over {self.population}; "
            f"{len(self.effect)} empirical intervals, sparsest n={int(self.counts.min())}. "
            "The curve is descriptive fitted structure and carries no sampling, "
            "selection, or causal uncertainty."
        )


@dataclass(frozen=True)
class PredictionProfileResult:
    """A weighted partial-dependence curve with optional ICE predictions."""

    variable: str
    grid: np.ndarray
    average: np.ndarray
    ice: np.ndarray | None
    weights: np.ndarray
    observation_ids: np.ndarray
    scale: str
    parameter: str | None
    population: str
    support_fraction_far: np.ndarray
    support: dict

    def __post_init__(self):
        grid = np.asarray(self.grid).copy()
        average = np.asarray(self.average, dtype=np.float64).copy()
        far = np.asarray(self.support_fraction_far, dtype=np.float64).copy()
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("grid must be a non-empty one-dimensional array")
        if average.shape != grid.shape or far.shape != grid.shape:
            raise ValueError("average and support_fraction_far must match grid")
        n = len(np.asarray(self.weights))
        for name in ("weights", "observation_ids"):
            value = np.asarray(getattr(self, name)).copy()
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.ice is not None:
            ice = np.asarray(self.ice, dtype=np.float64).copy()
            if ice.shape != (n, grid.size):
                raise ValueError(f"ice must have shape ({n}, {grid.size}), got {ice.shape}")
            ice.setflags(write=False)
            object.__setattr__(self, "ice", ice)
        for value in (grid, average, far):
            value.setflags(write=False)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "average", average)
        object.__setattr__(self, "support_fraction_far", far)
        object.__setattr__(self, "support", MappingProxyType(dict(self.support)))

    def to_frame(self, output="auto"):
        """One row per grid value with its average and support diagnostic."""
        rows = [{
            "grid": self.grid[index].item()
            if isinstance(self.grid[index], np.generic) else self.grid[index],
            "average_prediction": float(self.average[index]),
            "support_fraction_far": (
                None if np.isnan(self.support_fraction_far[index])
                else float(self.support_fraction_far[index])
            ),
        } for index in range(len(self.grid))]
        return output_frame(rows, output, columns=PREDICTION_PROFILE_COLUMNS)

    def __repr__(self) -> str:
        return (
            f"PredictionProfileResult(variable={self.variable!r}, "
            f"scale={self.scale!r}, points={len(self.grid)}, "
            f"n={len(self.weights)}, ice={self.ice is not None})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the curve always, ICE rows only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("ice", "weights", "observation_ids"),
        )

    def ice_frame(self, output="auto"):
        """One row per observation-grid prediction when ICE was retained."""
        if self.ice is None:
            raise ValueError("ICE predictions were omitted; rerun with retain_ice=True")
        rows = []
        for row in range(self.ice.shape[0]):
            record_id = self.observation_ids[row]
            if isinstance(record_id, np.generic):
                record_id = record_id.item()
            for column, grid_value in enumerate(self.grid):
                rows.append({
                    "record_id": record_id,
                    "grid": grid_value.item() if isinstance(grid_value, np.generic)
                    else grid_value,
                    "prediction": float(self.ice[row, column]),
                    "weight": float(self.weights[row]),
                })
        return output_frame(rows, output)

    def describe(self) -> str:
        """State scale, population, retained rows, and support diagnostic."""
        ice = "ICE retained" if self.ice is not None else "ICE omitted"
        return (
            f"Weighted partial-dependence profile of {self.variable} on the "
            f"{self.scale} scale over {self.population}; {len(self.grid)} grid values, "
            f"{ice}. Joint support uses {self.support['method']}; it is a diagnostic, "
            "not proof that every counterfactual combination is observed."
        )


def prediction_contrast(model, data, variable, reference, alternative, *,
                        type="response", parameter=None, weights=None,
                        observation_ids=None, population="supplied rows",
                        extrapolation="error", mstop=None, cluster=None,
                        level=0.95, n_draws=1000, seed=0, joint_support="report",
                        joint_threshold=1.0, support_columns=None):
    """Average a full-model prediction change over the supplied rows.

    Two copies of ``data`` are made. ``variable`` is set to ``reference`` in
    one and ``alternative`` in the other; every other predictor retains each
    row's observed value. The returned estimate is the weighted mean of the
    row-wise full-prediction differences. On a nonidentity link, response-scale
    contrasts transform the complete predictor before subtraction rather than
    exponentiating or transforming an isolated component.

    ``cluster`` names a cluster label per supplied row (a column's values, an
    array of length n). When it is given, the result carries a cluster-level
    uncertainty: the cluster-robust standard error of the weighted mean and a
    cluster-bootstrap percentile interval at ``level``, from ``n_draws``
    resamples of clusters with ``seed``. The clusters are the sampling units,
    and the fitted model is held fixed.

    ``joint_support`` checks the counterfactual combinations. Setting
    ``variable`` to ``reference`` keeps every other predictor at its row value,
    and that combination may never occur in the training data (a categorical
    level seen only with other covariate values, or a high ``x1`` with a low
    correlated ``x2``). A row is jointly supported when the training data has a
    row in the same categorical cell within ``joint_threshold`` training
    standard deviations on the numeric predictors the model reads. The policy
    ``"report"`` records the supported share in ``support["joint"]``;
    ``"warn"`` also warns, ``"error"`` raises, and ``"off"`` skips the check.
    ``support_columns`` names the conditioning columns, as in
    :func:`prediction_profile`. The check costs one pass over the training rows
    per standardization row and per counterfactual setting; pass
    ``joint_support="off"`` for very large data.

    This is standardization over ``population``, not a causal effect. A causal
    reading additionally needs a well-defined intervention, baseline adjustment
    set, exchangeability, positivity, consistency and an interference design.
    """
    if type not in ("link", "response"):
        raise ValueError("type must be 'link' or 'response'")
    if extrapolation not in ("error", "warn", "allow", "clip"):
        raise ValueError("extrapolation must be 'error', 'warn', 'allow', or 'clip'")
    names = column_names(data)
    if variable not in names:
        raise ValueError(f"variable {variable!r} is not present in data")
    mapping = _as_mapping(data)
    n = len(mapping[variable])
    if n == 0 or any(len(values) != n for values in mapping.values()):
        raise ValueError("data columns must have one common nonzero length")
    ids = _record_ids(n, observation_ids)
    normalized = normalize_weights(weights, n, policy="raw").fit
    training = _support_values(model, data, variable)
    reference_used, reference_support = _effective_value(
        reference, training, variable, extrapolation
    )
    alternative_used, alternative_support = _effective_value(
        alternative, training, variable, extrapolation
    )
    reference_data = {name: values.copy() for name, values in mapping.items()}
    alternative_data = {name: values.copy() for name, values in mapping.items()}
    reference_data[variable] = np.full(n, reference_used)
    alternative_data[variable] = np.full(n, alternative_used)
    joint = _joint_check(
        model, mapping, variable,
        [reference_data[variable], alternative_data[variable]], normalized,
        policy=joint_support, threshold=joint_threshold,
        support_columns=support_columns, stacklevel=2,
    )
    prediction_at_reference = _predict(
        model, reference_data, scale=type, parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    prediction_at_alternative = _predict(
        model, alternative_data, scale=type, parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    individual = prediction_at_alternative - prediction_at_reference
    estimate = float(np.average(individual, weights=normalized))
    dependence = _cluster_dependence(
        individual, normalized, cluster, level=level, n_draws=n_draws, seed=seed,
    )
    return ContrastResult(
        estimate=estimate, individual=individual,
        reference_prediction=prediction_at_reference,
        alternative_prediction=prediction_at_alternative,
        weights=normalized, observation_ids=ids, variable=variable,
        reference=reference, alternative=alternative, scale=type,
        parameter=parameter, population=population,
        support={"reference": reference_support, "alternative": alternative_support,
                 "joint": joint},
        uncertainty="none" if dependence is None else "cluster_bootstrap",
        dependence=dependence,
    )


def average_marginal_effect(model, data, variable, *, step=None, type="response",
                            parameter=None, weights=None, observation_ids=None,
                            population="supplied rows", extrapolation="error",
                            mstop=None, cluster=None, level=0.95, n_draws=1000,
                            seed=0, joint_support="report", joint_threshold=1.0,
                            support_columns=None):
    """Average row-specific finite derivatives of the full fitted prediction.

    Other predictors retain each row's observed value. Interior rows use a
    centered finite difference; rows within half a step of the training boundary
    use the available one-sided interval. The denominator is each row's actual
    upper-minus-lower value, so boundary truncation does not silently change the
    derivative scale. ``step`` defaults to one ten-thousandth of the nonzero
    training span and is recorded in the result.

    On a nonidentity link, response derivatives transform the complete predictor
    before differencing. The returned average standardizes a fitted derivative
    over ``population``; it is neither a coefficient nor a causal effect by
    itself. ``cluster``, ``level``, ``n_draws`` and ``seed`` give the
    cluster-level uncertainty of the average, as in :func:`prediction_contrast`;
    without ``cluster`` the average has no sampling uncertainty attached.
    ``joint_support`` checks the evaluation points against the training
    combinations, as in :func:`prediction_contrast`.
    """
    if type not in ("link", "response"):
        raise ValueError("type must be 'link' or 'response'")
    if extrapolation not in ("error", "warn", "allow", "clip"):
        raise ValueError("extrapolation must be 'error', 'warn', 'allow', or 'clip'")
    names = column_names(data)
    if variable not in names:
        raise ValueError(f"variable {variable!r} is not present in data")
    mapping = _as_mapping(data)
    values = mapping[variable]
    if values.dtype.kind not in "iufb":
        raise ValueError("average_marginal_effect requires a numeric or boolean variable")
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0 or any(len(column) != n for column in mapping.values()):
        raise ValueError("data columns must have one common nonzero length")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{variable!r} must be finite on every standardization row")
    ids = _record_ids(n, observation_ids)
    normalized = normalize_weights(weights, n, policy="raw").fit
    training = np.asarray(_support_values(model, data, variable), dtype=np.float64)
    training = training[np.isfinite(training)]
    if training.size == 0:
        raise ValueError(f"{variable!r} has no finite training support")
    low, high = float(training.min()), float(training.max())
    span = high - low
    if span <= 0:
        raise ValueError(f"{variable!r} has constant training support, so its derivative is undefined")
    step = span * 1e-4 if step is None else float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")

    outside = (values < low) | (values > high)
    if np.any(outside) and extrapolation == "error":
        raise ValueError(
            f"{int(outside.sum())} {variable!r} rows are outside training support "
            f"[{low:g}, {high:g}]"
        )
    if np.any(outside) and extrapolation == "warn":
        warnings.warn(
            f"{int(outside.sum())} {variable!r} rows are outside training support; "
            "their derivatives depend on extrapolation",
            RuntimeWarning, stacklevel=2,
        )
    centers = np.clip(values, low, high) if extrapolation == "clip" else values
    lower = centers - step / 2
    upper = centers + step / 2
    inside = (centers >= low) & (centers <= high)
    lower[inside] = np.maximum(lower[inside], low)
    upper[inside] = np.minimum(upper[inside], high)
    denominator = upper - lower
    if np.any(denominator <= 0):
        raise ValueError("finite-difference interval collapsed at the training boundary")
    joint = _joint_check(
        model, mapping, variable, [lower, upper], normalized,
        policy=joint_support, threshold=joint_threshold,
        support_columns=support_columns, stacklevel=2,
    )
    lower_data = {name: column.copy() for name, column in mapping.items()}
    upper_data = {name: column.copy() for name, column in mapping.items()}
    lower_data[variable] = lower
    upper_data[variable] = upper
    lower_prediction = _predict(
        model, lower_data, scale=type, parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    upper_prediction = _predict(
        model, upper_data, scale=type, parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    individual = (upper_prediction - lower_prediction) / denominator
    estimate = float(np.average(individual, weights=normalized))
    dependence = _cluster_dependence(
        individual, normalized, cluster, level=level, n_draws=n_draws, seed=seed,
    )
    one_sided = (lower == low) | (upper == high)
    return MarginalEffectResult(
        estimate=estimate, individual=individual, lower_value=lower,
        upper_value=upper, lower_prediction=lower_prediction,
        upper_prediction=upper_prediction, weights=normalized,
        observation_ids=ids, variable=variable, scale=type,
        parameter=parameter, population=population, step=step,
        support={
            "training_min": low, "training_max": high,
            "outside_rows": int(outside.sum()),
            "one_sided_rows": int(one_sided.sum()),
            "extrapolation": extrapolation,
            "joint": joint,
        },
        uncertainty="none" if dependence is None else "cluster_bootstrap",
        dependence=dependence,
    )


def average_elasticity(model, data, variable, *, kind="elasticity", log_step=1e-3,
                       parameter=None, weights=None, observation_ids=None,
                       population="supplied rows", extrapolation="error",
                       mstop=None):
    """Average a finite elasticity or semi-elasticity of response prediction.

    ``kind="elasticity"`` computes ``d log(prediction) / d log(variable)``;
    both the predictor and prediction must be positive. ``kind="semi_elasticity"``
    computes ``d log(prediction) / d variable`` and only the prediction must be
    positive. Evaluation points are multiplicatively separated by ``log_step``
    for elasticity and use the corresponding local additive separation for the
    semi-elasticity. Training-boundary rows use one-sided intervals.

    The result describes the fitted response surface over ``population``. It is
    not a coefficient when the response link is nonlinear, and it is not a
    causal effect without an identified intervention.
    """
    if kind not in ("elasticity", "semi_elasticity"):
        raise ValueError("kind must be 'elasticity' or 'semi_elasticity'")
    if extrapolation not in ("error", "warn", "allow", "clip"):
        raise ValueError("extrapolation must be 'error', 'warn', 'allow', or 'clip'")
    log_step = float(log_step)
    if not np.isfinite(log_step) or log_step <= 0:
        raise ValueError("log_step must be finite and positive")
    names = column_names(data)
    if variable not in names:
        raise ValueError(f"variable {variable!r} is not present in data")
    mapping = _as_mapping(data)
    raw = mapping[variable]
    if raw.dtype.kind not in "iuf" or np.unique(raw).size < 3:
        raise ValueError(
            "average_elasticity requires a numeric variable with at least three "
            "distinct values; use prediction_contrast for indicators or factors"
        )
    values = np.asarray(raw, dtype=np.float64)
    n = len(values)
    if n == 0 or any(len(column) != n for column in mapping.values()):
        raise ValueError("data columns must have one common nonzero length")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{variable!r} must be finite on every standardization row")
    if kind == "elasticity" and np.any(values <= 0):
        raise ValueError("elasticity requires positive predictor values on every row")
    ids = _record_ids(n, observation_ids)
    normalized = normalize_weights(weights, n, policy="raw").fit
    training = np.asarray(_support_values(model, data, variable), dtype=np.float64)
    training = training[np.isfinite(training)]
    if training.size == 0:
        raise ValueError(f"{variable!r} has no finite training support")
    low, high = float(training.min()), float(training.max())
    outside = (values < low) | (values > high)
    if np.any(outside) and extrapolation == "error":
        raise ValueError(
            f"{int(outside.sum())} {variable!r} rows are outside training support "
            f"[{low:g}, {high:g}]"
        )
    if np.any(outside) and extrapolation == "warn":
        warnings.warn(
            f"{int(outside.sum())} {variable!r} rows are outside training support; "
            "their elasticities depend on extrapolation",
            RuntimeWarning, stacklevel=2,
        )
    centers = np.clip(values, low, high) if extrapolation == "clip" else values
    if kind == "elasticity":
        lower = centers * np.exp(-log_step / 2)
        upper = centers * np.exp(log_step / 2)
    else:
        # A relative step retains numerical resolution across predictor scales,
        # while the denominator below remains in the predictor's original unit.
        additive_step = np.maximum(np.abs(centers), max(high - low, 1.0)) * log_step
        lower = centers - additive_step / 2
        upper = centers + additive_step / 2
    inside = (centers >= low) & (centers <= high)
    lower[inside] = np.maximum(lower[inside], low)
    upper[inside] = np.minimum(upper[inside], high)
    if kind == "elasticity" and np.any(lower <= 0):
        raise ValueError(
            "elasticity evaluation reached a non-positive predictor boundary; "
            "restrict data to positive support or use semi_elasticity"
        )
    denominator = (np.log(upper) - np.log(lower) if kind == "elasticity"
                   else upper - lower)
    if np.any(denominator <= 0):
        raise ValueError("elasticity evaluation interval collapsed at the training boundary")
    lower_data = {name: column.copy() for name, column in mapping.items()}
    upper_data = {name: column.copy() for name, column in mapping.items()}
    lower_data[variable] = lower
    upper_data[variable] = upper
    lower_prediction = _predict(
        model, lower_data, scale="response", parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    upper_prediction = _predict(
        model, upper_data, scale="response", parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    if np.any(lower_prediction <= 0) or np.any(upper_prediction <= 0):
        raise ValueError(
            f"{kind} requires positive response predictions at every evaluation point"
        )
    individual = (np.log(upper_prediction) - np.log(lower_prediction)) / denominator
    one_sided = (lower == low) | (upper == high)
    return ElasticityResult(
        estimate=float(np.average(individual, weights=normalized)),
        individual=individual, lower_value=lower, upper_value=upper,
        lower_prediction=lower_prediction, upper_prediction=upper_prediction,
        weights=normalized, observation_ids=ids, variable=variable, kind=kind,
        parameter=parameter, population=population, log_step=log_step,
        support={
            "training_min": low, "training_max": high,
            "outside_rows": int(outside.sum()),
            "one_sided_rows": int(one_sided.sum()),
            "extrapolation": extrapolation,
        },
    )


def distributional_contrast(model, data, variable, reference, alternative, *,
                            quantity, value=None, weights=None,
                            observation_ids=None, population="supplied rows",
                            extrapolation="error", mstop=None, cluster=None,
                            level=0.95, n_draws=1000, seed=0,
                            joint_support="report", joint_threshold=1.0,
                            support_columns=None, refit=None):
    """Contrast a fitted LSS mean, quantile, or exceedance probability.

    All response-scale distribution parameters are predicted under the two
    settings before the requested functional is evaluated. A quantile can
    therefore change through location, scale, or shape; an exceedance
    probability can change even when the mean does not. Exceedance uses the
    strict convention ``P(Y > value) = 1 - F(value)``, including discrete and
    mixed families.

    The averaging, weighting, support and causal limits are those of
    :func:`prediction_contrast`. ``value`` is the probability for a quantile,
    the response threshold for an exceedance probability, and must be omitted
    for a mean.

    ``refit`` is a :func:`mboostrs.gamlss_refit` result for the same model. Its
    coefficient draws (``n_draws`` of them, from ``seed``) give one contrast per
    draw, and the interval is their ``level`` percentile range. This is the
    refit's approximate posterior, which includes the coefficient uncertainty of
    the distributional fit; the point estimate still comes from ``model``. It
    cannot be combined with ``cluster=``.
    """
    if extrapolation not in ("error", "warn", "allow", "clip"):
        raise ValueError("extrapolation must be 'error', 'warn', 'allow', or 'clip'")
    names = column_names(data)
    if variable not in names:
        raise ValueError(f"variable {variable!r} is not present in data")
    mapping = _as_mapping(data)
    n = len(mapping[variable])
    if n == 0 or any(len(values) != n for values in mapping.values()):
        raise ValueError("data columns must have one common nonzero length")
    ids = _record_ids(n, observation_ids)
    normalized = normalize_weights(weights, n, policy="raw").fit
    training = _support_values(model, data, variable)
    reference_used, reference_support = _effective_value(
        reference, training, variable, extrapolation
    )
    alternative_used, alternative_support = _effective_value(
        alternative, training, variable, extrapolation
    )
    reference_data = {name: values.copy() for name, values in mapping.items()}
    alternative_data = {name: values.copy() for name, values in mapping.items()}
    reference_data[variable] = np.full(n, reference_used)
    alternative_data[variable] = np.full(n, alternative_used)
    joint = _joint_check(
        model, mapping, variable,
        [reference_data[variable], alternative_data[variable]], normalized,
        policy=joint_support, threshold=joint_threshold,
        support_columns=support_columns, stacklevel=2,
    )
    reference_prediction, label = _distribution_quantity(
        model, reference_data, quantity, value, mstop
    )
    alternative_prediction, other_label = _distribution_quantity(
        model, alternative_data, quantity, value, mstop
    )
    assert label == other_label
    individual = alternative_prediction - reference_prediction
    estimate = float(np.average(individual, weights=normalized))
    if cluster is not None and refit is not None:
        raise ValueError("cluster= and refit= give two different intervals; pass one of them")
    if refit is not None:
        dependence = _refit_dependence(
            model, refit, reference_data, alternative_data, quantity, value, normalized,
            level=level, n_draws=n_draws, seed=seed,
        )
    else:
        dependence = _cluster_dependence(
            individual, normalized, cluster, level=level, n_draws=n_draws, seed=seed,
        )
    return ContrastResult(
        estimate=estimate,
        individual=individual, reference_prediction=reference_prediction,
        alternative_prediction=alternative_prediction, weights=normalized,
        observation_ids=ids, variable=variable, reference=reference,
        alternative=alternative, scale="response", parameter=label,
        population=population,
        support={"reference": reference_support, "alternative": alternative_support,
                 "joint": joint},
        uncertainty="none" if dependence is None else dependence["method"],
        dependence=dependence,
    )


def accumulated_local_effect(model, data, variable, *, bins=10, type="response",
                             parameter=None, weights=None, observation_ids=None,
                             population="supplied rows", extrapolation="error",
                             mstop=None):
    """Estimate a centered univariate accumulated local effect.

    Weighted empirical quantiles partition ``variable``. Within each interval,
    the model predicts its resident rows at the interval's lower and upper
    boundaries while keeping their other predictors unchanged. Weighted local
    differences are accumulated across intervals and centered to weighted mean
    zero over ``population``. This avoids the global predictor combinations of
    partial dependence, but sparse intervals and extrapolated input rows remain
    visible in the returned counts and support policy.

    The interval effect is placed at its midpoint by accumulating prior local
    changes plus half the current one. This is a finite-bin approximation to
    the ALE integral. It describes the fitted model and carries no sampling,
    model-selection, or causal uncertainty.
    """
    from mboostrs._validate import check_int_at_least

    if type not in ("link", "response"):
        raise ValueError("type must be 'link' or 'response'")
    if extrapolation not in ("error", "warn", "allow", "clip"):
        raise ValueError("extrapolation must be 'error', 'warn', 'allow', or 'clip'")
    bins = check_int_at_least("bins", bins, 2, allow_none=False)
    names = column_names(data)
    if variable not in names:
        raise ValueError(f"variable {variable!r} is not present in data")
    mapping = _as_mapping(data)
    raw = mapping[variable]
    if raw.dtype.kind not in "iuf" or np.unique(raw).size < 3:
        raise ValueError(
            "accumulated_local_effect requires a numeric variable with at least "
            "three distinct values; use prediction_contrast for indicators or factors"
        )
    values = np.asarray(raw, dtype=np.float64)
    n = len(values)
    if n == 0 or any(len(column) != n for column in mapping.values()):
        raise ValueError("data columns must have one common nonzero length")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{variable!r} must be finite on every standardization row")
    ids = _record_ids(n, observation_ids)
    normalized = normalize_weights(weights, n, policy="raw").fit
    training = np.asarray(_support_values(model, data, variable), dtype=np.float64)
    training = training[np.isfinite(training)]
    if training.size == 0:
        raise ValueError(f"{variable!r} has no finite training support")
    low, high = float(training.min()), float(training.max())
    outside = (values < low) | (values > high)
    if np.any(outside) and extrapolation == "error":
        raise ValueError(
            f"{int(outside.sum())} {variable!r} rows are outside training support "
            f"[{low:g}, {high:g}]"
        )
    if np.any(outside) and extrapolation == "warn":
        warnings.warn(
            f"{int(outside.sum())} {variable!r} rows are outside training support; "
            "their ALE contributions depend on extrapolation",
            RuntimeWarning, stacklevel=2,
        )
    effective = np.clip(values, low, high) if extrapolation == "clip" else values
    edges = _weighted_quantiles(
        effective, normalized, np.linspace(0, 1, bins + 1)
    )
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError(
            "weighted quantile boundaries collapsed to fewer than two intervals; "
            "lower bins or use prediction_contrast"
        )
    n_bins = edges.size - 1
    bin_index = np.searchsorted(edges, effective, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    local_change = np.empty(n_bins, dtype=np.float64)
    counts = np.empty(n_bins, dtype=np.int64)
    weight_totals = np.empty(n_bins, dtype=np.float64)
    resident, frames = [], []
    for index in range(n_bins):
        rows = bin_index == index
        positive = rows & (normalized > 0)
        if not np.any(positive):
            raise ValueError(
                f"ALE interval {index} [{edges[index]:g}, {edges[index + 1]:g}] "
                "has no positive-weight observations"
            )
        subset = {name: column[positive] for name, column in mapping.items()}
        m = int(positive.sum())
        frames.append({**subset, variable: np.full(m, edges[index])})
        frames.append({**subset, variable: np.full(m, edges[index + 1])})
        resident.append(positive)
        counts[index] = int(rows.sum())
        weight_totals[index] = float(normalized[rows].sum())
    # Every interval's rows at both of its boundaries, predicted together.
    predictions = _predict_stacked(
        model, frames, scale=type, parameter=parameter, mstop=mstop,
        extrapolation=extrapolation,
    )
    for index, positive in enumerate(resident):
        lower_prediction, upper_prediction = predictions[2 * index:2 * index + 2]
        local_change[index] = np.average(
            upper_prediction - lower_prediction, weights=normalized[positive]
        )
    midpoint_effect = np.cumsum(local_change) - .5 * local_change
    raw_observation_effect = midpoint_effect[bin_index]
    center = float(np.average(raw_observation_effect, weights=normalized))
    effect = midpoint_effect - center
    observation_effect = effect[bin_index]
    return ALE1DResult(
        variable=variable, edges=edges, grid=(edges[:-1] + edges[1:]) / 2,
        effect=effect, local_change=local_change, counts=counts,
        weight_totals=weight_totals, bin_index=bin_index,
        observation_effect=observation_effect, weights=normalized,
        observation_ids=ids, scale=type, parameter=parameter,
        population=population,
        support={
            "training_min": low, "training_max": high,
            "outside_rows": int(outside.sum()),
            "extrapolation": extrapolation,
            "requested_bins": int(bins), "effective_bins": int(n_bins),
        },
    )


def _fitted_predictor_columns(model, names):
    """Supplied columns the fitted model's formula reads, or None if unknown.

    None means "use every supplied column": a model without a formula, or a
    formula with a ``.`` term, which reads every column of its data.
    """
    from mboostrs.explore import split_terms
    from mboostrs.formula import _IDENTIFIER_RE

    formula = getattr(model, "formula", None)
    if isinstance(formula, dict):
        formulas = list(formula.values())
    elif isinstance(formula, str):
        formulas = [formula]
    else:
        return None
    columns = []
    for text in formulas:
        if "~" not in text:
            return None
        rhs = partition_tilde(text)[2]
        if any(term.strip() == "." for term in split_terms(rhs)):
            return None
        for name in _IDENTIFIER_RE.findall(rhs):
            if name in names and name not in columns:
                columns.append(name)
    return columns


def _support_columns(model, mapping, variable, support_columns):
    """`(conditioning column names, supplied columns left out)` for the support check."""
    response = getattr(model, "_response_name", None)
    responses = set(response) if isinstance(response, (list, tuple)) else {response}
    supplied = [name for name in mapping if name != variable and name not in responses]
    if support_columns is not None:
        requested = ([support_columns] if isinstance(support_columns, str)
                     else list(support_columns))
        missing = [name for name in requested if name not in mapping]
        if missing:
            raise ValueError(f"support_columns names columns not in data: {missing}")
        chosen = [name for name in requested if name != variable]
    else:
        fitted = _fitted_predictor_columns(model, list(mapping))
        chosen = supplied if fitted is None else [
            name for name in supplied if name in fitted]
    return chosen, [name for name in supplied if name not in chosen]


def _linear_conditional_support(model, mapping, variable, grid, weights, threshold,
                                support_columns=None):
    """Fraction of standardized rows far from a weighted linear support model."""
    chosen, excluded = _support_columns(model, mapping, variable, support_columns)
    candidates = []
    for name, raw in mapping.items():
        if name not in chosen:
            continue
        values = np.asarray(raw)
        if values.dtype.kind in "iufb" and np.all(np.isfinite(values)):
            candidates.append(np.asarray(values, dtype=np.float64))
    if not candidates or np.asarray(grid).dtype.kind not in "iuf":
        return np.full(len(grid), np.nan), {
            "method": "unavailable_without_other_finite_numeric_columns",
            "threshold_residual_sd": threshold,
            "numeric_conditioning_columns": len(candidates),
            "conditioning_columns": tuple(chosen),
            "excluded_columns": tuple(excluded),
        }
    target = np.asarray(mapping[variable], dtype=np.float64)
    design = np.column_stack([np.ones(len(target)), *candidates])
    root_weight = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, None], target * root_weight, rcond=None
    )
    expected = design @ coefficients
    residual = target - expected
    scale = float(np.sqrt(np.average(residual**2, weights=weights)))
    if scale <= np.finfo(np.float64).eps * max(1.0, float(np.std(target))):
        far = np.array([
            np.average(np.abs(value - expected) > 1e-12, weights=weights)
            for value in np.asarray(grid, dtype=np.float64)
        ])
    else:
        far = np.array([
            np.average(np.abs(value - expected) / scale > threshold, weights=weights)
            for value in np.asarray(grid, dtype=np.float64)
        ])
    return far, {
        "method": "weighted_linear_conditional_residual",
        "threshold_residual_sd": threshold,
        "residual_scale": scale,
        "numeric_conditioning_columns": len(candidates),
        "conditioning_columns": tuple(chosen),
        "excluded_columns": tuple(excluded),
        "limitation": "nonlinear and categorical dependence may be missed",
    }


def prediction_profile(model, data, variable, *, grid=None, points=25,
                       type="response", parameter=None, weights=None,
                       observation_ids=None, population="supplied rows",
                       extrapolation="error", retain_ice=True,
                       support_threshold=2.5, support_columns=None, mstop=None):
    """Weighted partial dependence and optional individual conditional curves.

    At each grid value, ``variable`` is replaced for every supplied row while
    all other predictors retain their row values. The weighted average is the
    partial-dependence curve; the prediction matrix contains individual
    conditional-expectation (ICE) curves. The default grid uses weighted
    empirical quantiles and therefore remains dense where the declared
    population has observations.

    ``support_fraction_far`` is the weighted fraction of counterfactual rows
    whose replaced value lies more than ``support_threshold`` residual standard
    deviations from a weighted linear regression of ``variable`` on other
    finite numeric columns. By default those are the supplied columns the
    fitted model's formula reads; a column the model does not use is left
    out, because it stays fixed while ``variable`` is replaced and, if it is
    derived from ``variable`` (an indicator of old buildings beside the
    construction year), it would report combinations as unsupported that the
    model never sees. ``support_columns=`` names the conditioning columns
    explicitly. A formula with a ``.`` term reads every column, so then every
    supplied column is used. ``support["conditioning_columns"]`` and
    ``support["excluded_columns"]`` record the choice. This detects linear
    joint-support problems; nonlinear and categorical dependence may be
    missed. ALE is preferable when global replacements create unsupported
    combinations.
    """
    from mboostrs._validate import check_int_at_least, check_positive

    if type not in ("link", "response"):
        raise ValueError("type must be 'link' or 'response'")
    if extrapolation not in ("error", "warn", "allow", "clip"):
        raise ValueError("extrapolation must be 'error', 'warn', 'allow', or 'clip'")
    if not isinstance(retain_ice, (bool, np.bool_)):
        raise TypeError("retain_ice must be boolean")
    support_threshold = check_positive("support_threshold", support_threshold)
    names = column_names(data)
    if variable not in names:
        raise ValueError(f"variable {variable!r} is not present in data")
    mapping = _as_mapping(data)
    values = mapping[variable]
    n = len(values)
    if n == 0 or any(len(column) != n for column in mapping.values()):
        raise ValueError("data columns must have one common nonzero length")
    ids = _record_ids(n, observation_ids)
    normalized = normalize_weights(weights, n, policy="raw").fit
    training = _support_values(model, data, variable)
    numeric = training.dtype.kind in "iufb"
    if grid is None:
        if numeric:
            points = check_int_at_least("points", points, 2, allow_none=False)
            source = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(source)):
                raise ValueError(f"{variable!r} must be finite to construct a profile grid")
            requested_grid = np.unique(_weighted_quantiles(
                source, normalized, np.linspace(0, 1, points)
            ))
        else:
            requested_grid = np.asarray(np.unique(training), dtype=object)
    else:
        requested_grid = np.asarray(list(grid) if not np.isscalar(grid) else [grid])
        if requested_grid.ndim != 1 or requested_grid.size == 0:
            raise ValueError("grid must be a non-empty one-dimensional sequence")
    effective_grid, support_rows = [], []
    for requested in requested_grid:
        value, support = _effective_value(requested, training, variable, extrapolation)
        effective_grid.append(value)
        support_rows.append(support)
    effective_grid = np.asarray(effective_grid,
                                dtype=np.float64 if numeric else object)
    if numeric:
        effective_grid = np.unique(effective_grid)
    if parameter is None and hasattr(model, "predict_over_grid"):
        # One pass over the grid: the other predictors are replayed once and
        # only the learners that read `variable` are re-evaluated per value.
        grid_predictions = np.asarray(model.predict_over_grid(
            variable, list(effective_grid), newdata=mapping, type=type,
            mstop=mstop, extrapolation=extrapolation, output="numpy",
        ), dtype=np.float64)
        if grid_predictions.shape != (len(effective_grid), n) or not np.all(
                np.isfinite(grid_predictions)):
            raise ValueError(
                "contrast requires one finite prediction per row, got shape "
                f"{grid_predictions.shape}"
            )
        prediction = np.ascontiguousarray(grid_predictions.T)
    else:
        # The replaced frames are stacked row-wise and predicted together, which
        # pays the per-call replay cost once instead of once per grid value.
        columns = _predict_stacked(
            model, replaced_frames(mapping, variable, effective_grid, n),
            scale=type, parameter=parameter, mstop=mstop, extrapolation=extrapolation,
        )
        prediction = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    average = np.average(prediction, axis=0, weights=normalized)
    far, support = _linear_conditional_support(
        model, mapping, variable, effective_grid, normalized, support_threshold,
        support_columns,
    )
    support.update({
        "grid": tuple(support_rows),
        "extrapolation": extrapolation,
        "requested_points": int(len(requested_grid)),
        "effective_points": int(len(effective_grid)),
    })
    return PredictionProfileResult(
        variable=variable, grid=effective_grid, average=average,
        ice=prediction if retain_ice else None, weights=normalized,
        observation_ids=ids, scale=type, parameter=parameter,
        population=population, support_fraction_far=far, support=support,
    )
