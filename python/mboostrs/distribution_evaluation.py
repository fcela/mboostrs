"""Predictive evaluation for fitted distributional boosting models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from mboostrs._frame import read_column
from mboostrs._output import export_result, output_frame
from mboostrs._validate import check_unit_interval
from mboostrs._weights import normalize_weights
from mboostrs.diagnostics import interval_metrics
from mboostrs.evaluation import data_fingerprint

__all__ = ["DistributionEvaluationResult", "evaluate_distribution"]

# Leading columns of ``to_frame``; one column per distribution parameter follows.
_ROW_COLUMNS = (
    "row", "observed", "weight", "density_or_mass", "negative_log_score",
    "deviance", "pit", "crps",
)


def _readonly(value, *, shape=None, name="array"):
    result = np.asarray(value, dtype=np.float64).copy()
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DistributionEvaluationResult:
    """Row-level distribution scores and aggregate predictive diagnostics."""

    family: str
    response: str
    data_role: str
    data_fingerprint: str
    observed: np.ndarray
    weights: np.ndarray
    parameters: dict[str, np.ndarray]
    density_or_mass: np.ndarray
    negative_log_score: np.ndarray
    deviance: np.ndarray
    pit: np.ndarray
    crps: np.ndarray | None
    metrics: dict
    intervals: tuple[dict, ...]
    exceedances: tuple[dict, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        n = len(np.asarray(self.observed))
        for name in (
            "observed", "weights", "density_or_mass", "negative_log_score",
            "deviance", "pit",
        ):
            object.__setattr__(self, name, _readonly(
                getattr(self, name), shape=(n,), name=name
            ))
        if self.crps is not None:
            object.__setattr__(self, "crps", _readonly(
                self.crps, shape=(n,), name="crps"
            ))
        params = {}
        for name, value in self.parameters.items():
            params[name] = _readonly(value, shape=(n,), name=f"parameter {name}")
        object.__setattr__(self, "parameters", MappingProxyType(params))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "intervals", tuple(dict(row) for row in self.intervals))
        object.__setattr__(self, "exceedances", tuple(dict(row) for row in self.exceedances))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def __repr__(self) -> str:
        return (
            f"DistributionEvaluationResult(family={self.family!r}, "
            f"role={self.data_role!r}, n={len(self.observed)}, "
            f"mean_log_score={self.metrics['mean_log_score']:.6g})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the per-row score arrays and distribution
        parameters only on request, the small metrics always."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("observed", "weights", "parameters", "density_or_mass",
                    "negative_log_score", "deviance", "pit", "crps"),
            extras={"n": len(self.observed)},
        )

    def to_frame(self, output="auto"):
        """One row per evaluated outcome with parameters and scores."""
        rows = []
        for index in range(len(self.observed)):
            row = {
                "row": index, "observed": float(self.observed[index]),
                "weight": float(self.weights[index]),
                "density_or_mass": float(self.density_or_mass[index]),
                "negative_log_score": float(self.negative_log_score[index]),
                "deviance": float(self.deviance[index]),
                "pit": float(self.pit[index]),
                "crps": None if self.crps is None else float(self.crps[index]),
            }
            row.update({name: float(values[index])
                        for name, values in self.parameters.items()})
            rows.append(row)
        columns = _ROW_COLUMNS + tuple(self.parameters)
        return output_frame(rows, output, columns=columns)

    def interval_frame(self, output="auto"):
        """One row per requested central interval level."""
        return output_frame(list(self.intervals), output)

    def exceedance_frame(self, output="auto"):
        """One row per requested strict exceedance threshold."""
        return output_frame(list(self.exceedances), output)

    def describe(self) -> str:
        """State the evaluated distribution, data role, scores, and scope."""
        return (
            f"{self.family} distribution evaluation on {self.data_role} data: "
            f"n={len(self.observed)}, mean log score={self.metrics['mean_log_score']:.6g}, "
            f"PIT mean={self.metrics['pit_mean']:.4g}. "
            + " ".join(self.assumptions)
        )


def _response_name(model, response):
    if response is not None:
        return response
    # A transformed response (`log(rent) ~ ...`) stores the *derived* name
    # the fit read; the outcome the caller measured lives in the source
    # column the recipe step records, and that is the column to score.
    step = getattr(model, "_response_transform", None)
    if step is not None:
        return step.source
    name = getattr(model, "_response_name", None)
    if not isinstance(name, str):
        raise ValueError("response must name the scalar outcome column")
    return name


def _family_function(family, name):
    function = getattr(family, name, None)
    if function is None:
        raise NotImplementedError(
            f"{type(family).__name__} does not implement {name}; check family_capabilities()"
        )
    return function


def _model_has_group_terms(model) -> bool:
    """Whether any parameter's learner menu holds a ``brandom`` term, whose
    levels an evaluated row can fall outside of."""
    from mboostrs.learners import Brandom

    specs = list(getattr(model, "_learners_spec", None) or ())
    for parameter in getattr(model, "_params", None) or ():
        if isinstance(parameter, dict):
            specs.extend(parameter.get("learners") or ())
    return any(isinstance(spec, Brandom) for spec in specs)


# Families whose outcome is a count. Their CRPS is a sum over integer thresholds;
# every other family with a cdf and a quantile is integrated over t.
_COUNT_FAMILIES = frozenset({
    "PoissonLSS", "NBinomialLSS", "GeometricLSS", "BinomialLSS",
    "ZIPoLSS", "ZINBLSS", "ZeroAdjustedPoissonLSS",
})
_CRPS_TAIL = 1e-10
_CRPS_COUNT_TAIL = 1e-12
_CRPS_NODES = 1201


def _broadcast_rows(value, n):
    return np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)).copy()


def _crps_count(family, observed, params):
    """CRPS of an integer predictive distribution: the sum over thresholds
    k = 0..K of (F(k) - 1{y <= k})^2, with K past the upper 1e-12 quantile."""
    from mboostrs._rust import crps_discrete_sum

    n = observed.size
    if np.any(observed < 0) or np.any(observed != np.floor(observed)):
        return None
    upper = _broadcast_rows(family.quantile(1 - _CRPS_COUNT_TAIL, params), n)
    top = int(max(np.max(observed), np.max(np.ceil(upper)))) + 1
    cdf = np.column_stack([
        np.clip(_broadcast_rows(family.cdf(np.full(n, float(k)), params), n), 0.0, 1.0)
        for k in range(top)
    ])
    return np.asarray(crps_discrete_sum(observed, cdf))


def _crps_quadrature(family, observed, params, nodes=_CRPS_NODES):
    """CRPS of a continuous or mixed distribution, by Simpson quadrature of
    the integral of (F(t) - 1{y <= t})^2 over [q(1e-10), y] and [y, q(1 - 1e-10)].

    A point mass at the lower support bound (for example a zero-adjusted
    family's atom at zero) sits at the left endpoint, where the cdf is
    right-continuous, so the quadrature stays accurate for mixed families.
    """
    from mboostrs._rust import crps_cdf_quadrature

    n = observed.size
    lower = np.minimum(_broadcast_rows(family.quantile(_CRPS_TAIL, params), n), observed)
    upper = np.maximum(_broadcast_rows(family.quantile(1 - _CRPS_TAIL, params), n), observed)
    step_below = (observed - lower) / (nodes - 1)
    step_above = (upper - observed) / (nodes - 1)
    cdf_below = np.empty((n, nodes))
    cdf_above = np.empty((n, nodes))
    for k in range(nodes):
        at_below = observed - (nodes - 1 - k) * step_below
        at_above = observed + k * step_above
        cdf_below[:, k] = np.clip(_broadcast_rows(family.cdf(at_below, params), n), 0.0, 1.0)
        cdf_above[:, k] = np.clip(_broadcast_rows(family.cdf(at_above, params), n), 0.0, 1.0)
    return np.asarray(crps_cdf_quadrature(cdf_below, cdf_above, step_below, step_above))


def _crps(family, observed, params):
    """Row-level CRPS for a fitted family, or None when the family cannot supply it.

    GaussianLSS uses its closed form. Count families use the threshold sum.
    Every other family with a cdf and a quantile uses quadrature. A count family
    evaluated on outcomes that are not non-negative integers returns None.
    """
    from mboostrs._rust import crps_normal

    name = type(family).__name__
    if name == "GaussianLSS":
        if np.any(np.asarray(params["sigma"]) <= 0):
            raise ValueError("GaussianLSS CRPS requires positive sigma")
        return np.asarray(crps_normal(observed, params["mu"], params["sigma"]))
    if getattr(family, "cdf", None) is None or getattr(family, "quantile", None) is None:
        return None
    if name in _COUNT_FAMILIES:
        return _crps_count(family, observed, params)
    return _crps_quadrature(family, observed, params)


def evaluate_distribution(model, data, response=None, *, data_role,
                          weights=None, levels=(.5, .8, .95), thresholds=(),
                          random_state=0, mstop=None, unseen_levels="error"):
    """Evaluate a fitted scalar-response LSS distribution on supplied data.

    ``data_role`` must be ``"external"`` or ``"training"``. External means
    the outcomes were not used for fitting, tuning, specification search, or
    preprocessing choices; this is a caller declaration retained in the result,
    not something an already-fitted object can prove. Training results diagnose
    fit but do not estimate future predictive performance.

    The family must implement density/mass, CDF, quantile, and deviance
    increments. PIT randomizes uniformly between ``F(y-)`` and ``F(y)``, so
    discrete and mixed atoms are handled without subtracting an assumed unit.
    Central interval metrics use the proper interval score. Strict exceedance
    probabilities use ``1-F(threshold)``. CRPS, the integral of the squared
    gap between the predictive cdf and the step at the outcome, is reported
    for every family with a cdf and a quantile: GaussianLSS by its closed
    form, count families (Poisson, negative binomial, geometric, binomial,
    zero-inflated and zero-adjusted Poisson) as a sum over integer thresholds,
    and every other family, including the zero-adjusted gamma, by Simpson
    quadrature. CRPS is withheld for transformed responses, because it is not
    invariant to the transform; the result's assumptions name the reason.

    ``unseen_levels`` is the point policy of ``model.predict``: how a row
    whose ``brandom`` level the fit never saw is scored — ``"average"`` as a
    member of an average group, ``"zero_effect"`` with a zero coefficient,
    or ``"error"`` (the default) refused. A non-refusing policy is recorded
    in the result's assumptions, because the scores then condition on a point
    prediction for a group the fit never saw and do not integrate new-group
    uncertainty.
    """
    if data_role not in ("external", "training"):
        raise ValueError("data_role must be 'external' or 'training'")
    response = _response_name(model, response)
    observed = np.asarray(read_column(data, response).values, dtype=np.float64)
    if observed.ndim != 1 or observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("distribution evaluation requires a non-empty finite scalar response")
    w = normalize_weights(weights, len(observed), policy="raw").fit
    # A transformed response (`log(rent) ~ ...`) fits the family to g(y), so
    # the scores below compare g(y) with that family. The density then
    # carries the change-of-variables factor |g'(y)| — the `+ log(rent)` of
    # the euro-scale negative log density — so every reported score is on
    # the original scale the caller measured.
    from mboostrs.transforms import response_forward, response_jacobian

    response_step = getattr(model, "_response_transform", None)
    density_scale = None
    if response_step is not None:
        density_scale = np.asarray(
            response_jacobian(response_step, observed), dtype=np.float64)
        observed = np.asarray(
            response_forward(response_step, observed), dtype=np.float64)
        if observed.shape != density_scale.shape or np.any(~np.isfinite(observed)) \
                or np.any(density_scale <= 0.0):
            raise ValueError(
                "the response transform produced a non-finite or non-positive "
                "density scale; check the response values against "
                f"{response_step.expression!r}")
    import inspect

    accepts_policy = "unseen_levels" in inspect.signature(model.predict).parameters
    if unseen_levels != "error" and not accepts_policy:
        raise ValueError(
            f"{type(model).__name__}.predict() does not accept unseen_levels=, so "
            f"the requested policy {unseen_levels!r} for group levels the fit "
            "never saw cannot be applied")
    params = model.predict(newdata=data, type="response", mstop=mstop,
                           **({"unseen_levels": unseen_levels} if accepts_policy else {}))
    if not isinstance(params, dict) or not params:
        raise ValueError("evaluate_distribution requires an LSS model with parameter predictions")
    params = {name: np.asarray(value, dtype=np.float64) for name, value in params.items()}
    if any(value.shape != observed.shape or not np.all(np.isfinite(value))
           for value in params.values()):
        raise ValueError("every distribution parameter must be finite and aligned with the response")
    family = getattr(model, "_family", None)
    if family is None:
        raise ValueError("model does not expose a fitted distribution family")
    density = np.asarray(_family_function(family, "pdf")(observed, params), dtype=np.float64)
    if density_scale is not None:
        density = density * density_scale
    if density.shape != observed.shape or np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("family density or mass must be finite, non-negative, and row-aligned")
    negative_log_score = np.full(observed.shape, np.inf, dtype=np.float64)
    positive_density = density > 0
    negative_log_score[positive_density] = -np.log(density[positive_density])
    deviance = np.asarray(
        _family_function(family, "deviance_increment")(observed, params),
        dtype=np.float64,
    )
    if deviance.shape != observed.shape or np.any(np.isnan(deviance)):
        raise ValueError("family deviance increments must be row-aligned and not NaN")
    cdf = _family_function(family, "cdf")
    upper = np.asarray(cdf(observed, params), dtype=np.float64)
    cdf_left = getattr(family, "cdf_left", None)
    lower = (np.asarray(cdf_left(observed, params), dtype=np.float64)
             if cdf_left is not None else upper)
    if (upper.shape != observed.shape or lower.shape != observed.shape
            or np.any(~np.isfinite(upper)) or np.any(~np.isfinite(lower))
            or np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper)):
        raise ValueError("family CDF limits must satisfy 0 <= F(y-) <= F(y) <= 1")
    rng = np.random.default_rng(random_state)
    pit = lower + rng.random(len(observed)) * (upper - lower)

    quantile = _family_function(family, "quantile")
    interval_rows = []
    for level in levels:
        level = check_unit_interval("interval level", level)
        alpha = 1 - level
        lower_bound = np.asarray(quantile(alpha / 2, params), dtype=np.float64)
        upper_bound = np.asarray(quantile(1 - alpha / 2, params), dtype=np.float64)
        # Coverage is scale-free (a monotone transform moves the observation
        # and the bounds together), but width is not: bounds reported to the
        # caller are on the original scale, like the density above.
        if response_step is not None:
            from mboostrs.transforms import response_inverse

            lower_bound = np.asarray(
                response_inverse(response_step, lower_bound), dtype=np.float64)
            upper_bound = np.asarray(
                response_inverse(response_step, upper_bound), dtype=np.float64)
        metrics = interval_metrics(
            observed, lower_bound, upper_bound, level=level, weights=w
        )
        interval_rows.append(metrics)

    exceedance_rows = []
    total = float(w.sum())
    for threshold in thresholds:
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError("exceedance thresholds must be finite")
        probability = 1 - np.asarray(cdf(threshold, params), dtype=np.float64)
        event = observed > threshold
        clipped = np.clip(
            probability, np.finfo(np.float64).tiny,
            1 - np.finfo(np.float64).eps,
        )
        exceedance_rows.append({
            "threshold": threshold,
            "observed_rate": float(np.sum(w * event) / total),
            "predicted_rate": float(np.sum(w * probability) / total),
            "brier": float(np.sum(w * (event - probability) ** 2) / total),
            "log_loss": float(np.sum(
                w * -(event * np.log(clipped) + (~event) * np.log1p(-clipped))
            ) / total),
        })

    finite_log = np.isfinite(negative_log_score)
    finite_effective_log = finite_log | (w == 0)
    metrics = {
        "mean_log_score": (float(np.sum(w[finite_log] * negative_log_score[finite_log]) / total)
                           if np.all(finite_effective_log) else float("inf")),
        "zero_density_weight": float(np.sum(w[~finite_log])),
        "mean_deviance": float(np.sum(w * deviance) / total),
        "pit_mean": float(np.sum(w * pit) / total),
        "pit_variance": float(np.sum(w * (pit - np.sum(w * pit) / total) ** 2) / total),
        "weight_total": total,
    }
    # CRPS is not invariant to a response transform: on the fitted scale it
    # would score g(y), not y, so it is withheld for transformed responses.
    crps = None if response_step is not None else _crps(family, observed, params)
    if crps is not None:
        metrics["mean_crps"] = float(np.sum(w * crps) / total)
    else:
        metrics["mean_crps"] = None
    assumptions = (
        "The caller declares that external outcomes were excluded from fitting, tuning, and specification choices."
        if data_role == "external"
        else "Training-data scores diagnose fit and do not estimate future predictive performance.",
        "Weights define the evaluation population; dependence-aware uncertainty is not attached.",
    )
    if response_step is not None:
        assumptions = assumptions + (
            f"The response was fitted as {response_step.expression!r}; densities and "
            "scores are reported on the original scale, carrying the "
            "change-of-variables factor of that transform.",
        )
    if unseen_levels != "error" and _model_has_group_terms(model):
        policy = ("a member of an average group" if unseen_levels == "average"
                  else "a zero group effect")
        assumptions = assumptions + (
            f"Rows whose group level the fit never saw are scored as {policy} "
            f"(unseen_levels={unseen_levels!r}); every score conditions on that "
            "point prediction and none integrates new-group uncertainty.",
        )
    if crps is None:
        if response_step is not None:
            reason = "the response transform is applied and CRPS is not invariant to it"
        else:
            reason = (f"{type(family).__name__} has no cdf or quantile, or a count "
                      "family was given outcomes that are not non-negative integers")
        assumptions = assumptions + (f"CRPS is not reported: {reason}.",)
    return DistributionEvaluationResult(
        family=type(family).__name__, response=response, data_role=data_role,
        data_fingerprint=data_fingerprint(data), observed=observed, weights=w,
        parameters=params, density_or_mass=density,
        negative_log_score=negative_log_score, deviance=deviance, pit=pit,
        crps=crps, metrics=metrics, intervals=tuple(interval_rows),
        exceedances=tuple(exceedance_rows), assumptions=assumptions,
    )
