"""Fixed-horizon evaluation of CoxPH models on supplied outcomes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mboostrs._output import export_result, output_frame
from mboostrs.evaluation import data_fingerprint

__all__ = ["SurvivalEvaluationResult", "evaluate_survival"]

_ON_LOW_CENSORING = ("error", "warn")


def _readonly(value, *, shape=None, name="array"):
    result = np.asarray(value, dtype=np.float64).copy()
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SurvivalEvaluationResult:
    """Fixed-horizon Brier score and time-dependent AUC of a CoxPH model.

    One row of :meth:`horizon_frame` per horizon. The Brier score measures
    absolute predicted survival ``S(t | x)``; the AUC measures how well the
    linear predictor ranks subjects who have an event by ``t`` above subjects
    still event-free at ``t``. Both use inverse-probability-of-censoring
    weights, so a row censored before ``t`` is down-weighted rather than
    dropped.
    """

    family: str
    response: tuple
    data_role: str
    data_fingerprint: str
    horizons: tuple
    n_observations: int
    total_weight: float
    censoring_floor: float
    on_low_censoring: str
    time: np.ndarray
    event: np.ndarray
    weights: np.ndarray
    linear_predictor: np.ndarray
    survival: np.ndarray
    horizon_rows: tuple[dict, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        n = self.n_observations
        for name in ("time", "event", "weights", "linear_predictor"):
            object.__setattr__(self, name, _readonly(
                getattr(self, name), shape=(n,), name=name))
        object.__setattr__(self, "survival", _readonly(
            self.survival, shape=(n, len(self.horizons)), name="survival"))
        object.__setattr__(self, "horizon_rows",
                           tuple(dict(row) for row in self.horizon_rows))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "horizons", tuple(float(t) for t in self.horizons))

    def __repr__(self) -> str:
        briers = ", ".join(
            f"t={row['horizon']:g}: brier={row['brier']:.4g}, auc={row['auc']:.4g}"
            for row in self.horizon_rows
        )
        return f"SurvivalEvaluationResult(data_role={self.data_role!r}, {briers})"

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; per-subject arrays only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("time", "event", "weights", "linear_predictor", "survival"),
            extras={"n": self.n_observations},
        )

    def horizon_frame(self, output="auto"):
        """One row per horizon: IPCW Brier score, AUC, censoring survival and counts.

        Columns: ``brier`` (IPCW Brier score, lower is better), ``auc``
        (cumulative/dynamic IPCW AUC, 0.5 is chance), ``censoring_survival``
        (Kaplan-Meier ``G(t)`` on these rows), ``below_censoring_floor``,
        ``case_weight`` and ``control_weight`` (the weight totals behind the
        AUC), ``events_by_horizon`` and ``at_risk_at_horizon`` (weighted counts),
        and ``predicted_survival_mean`` (weighted mean of ``S(t | x)``).
        """
        return output_frame(list(self.horizon_rows), output)

    def survival_frame(self, output="auto"):
        """One row per subject and horizon with the predicted ``S(t | x)``."""
        rows = []
        for index in range(self.n_observations):
            for position, horizon in enumerate(self.horizons):
                rows.append({
                    "row": index, "horizon": horizon,
                    "survival": float(self.survival[index, position]),
                    "linear_predictor": float(self.linear_predictor[index]),
                    "time": float(self.time[index]), "event": float(self.event[index]),
                    "weight": float(self.weights[index]),
                })
        return output_frame(rows, output)

    def describe(self) -> str:
        """State the horizons, scores, censoring floor and data role."""
        parts = [
            f"{row['horizon']:g}: Brier={row['brier']:.4g}, AUC={row['auc']:.4g}, "
            f"G(t)={row['censoring_survival']:.3g}"
            for row in self.horizon_rows
        ]
        return (
            f"{self.family} fixed-horizon evaluation on {self.data_role} data, "
            f"n={self.n_observations}: " + "; ".join(parts) + ". "
            + " ".join(self.assumptions)
        )


def _check_horizons(horizons):
    values = np.atleast_1d(np.asarray(horizons, dtype=np.float64))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("horizons must be a non-empty finite one-dimensional sequence")
    if np.any(values < 0):
        raise ValueError("horizons must be non-negative")
    return values


def _check_weights(weights, n):
    if weights is None:
        return np.ones(n, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, with a positive total")
    return values


def evaluate_survival(model, data, *, horizons, data_role, weights=None,
                      censoring_floor=0.05, on_low_censoring="error"):
    """Brier score and time-dependent AUC of a CoxPH model at fixed horizons.

    ``data`` supplies the predictors and the ``Surv(time, status)`` columns the
    model was fitted with. The predicted survival ``S(t | x)`` comes from the
    Breslow baseline stored on ``model`` and the linear predictor of ``data``.
    The censoring survival ``G(t)`` is a Kaplan-Meier estimate on the rows of
    ``data``.

    ``data_role`` must be ``"external"`` or ``"training"``. External means the
    outcomes were not used to fit, tune or choose the model. This is a caller
    declaration recorded in the result, not something the fitted object can
    check. Training results describe fit and do not estimate prediction on new
    subjects.

    ``censoring_floor`` is the smallest ``G(t)`` at which the IPCW weights are
    considered stable. Below it, ``on_low_censoring="error"`` (default) raises
    and ``"warn"`` issues a warning and records ``below_censoring_floor`` in
    the result. Rows with ``G(t) = 0`` raise in either mode, because their
    weights are infinite.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        A fit with ``family=CoxPH()`` and a ``Surv(time, status)`` response.
    data : dict or DataFrame
        Rows to evaluate. Event status must be coded 0 (censored) or 1 (event).
    horizons : number or array-like
        Non-negative times on the response scale.
    data_role : {"external", "training"}
    weights : array-like, optional
        Non-negative evaluation weights, one per row of ``data``.
    censoring_floor : float, default 0.05
        Threshold on ``G(t)`` in ``[0, 1)``.
    on_low_censoring : {"error", "warn"}, default "error"

    Returns
    -------
    SurvivalEvaluationResult
    """
    from mboostrs._rust import ipcw_auc_at, ipcw_brier_at
    from mboostrs.glmboost import _extract_column

    if data_role not in ("external", "training"):
        raise ValueError("data_role must be 'external' or 'training'")
    if on_low_censoring not in _ON_LOW_CENSORING:
        raise ValueError("on_low_censoring must be 'error' or 'warn'")
    censoring_floor = float(censoring_floor)
    if not 0 <= censoring_floor < 1:
        raise ValueError("censoring_floor must lie in [0, 1)")
    if not hasattr(model, "_survival_from_linpred"):
        raise TypeError("evaluate_survival requires a GLMBoostModel or GAMBoostModel")
    response = getattr(model, "_response_name", None)
    if not isinstance(response, tuple):
        raise ValueError("evaluate_survival requires a Surv(time, status) response")
    horizon_values = _check_horizons(horizons)

    time = np.asarray(_extract_column(data, response[0]), dtype=np.float64).ravel()
    event = np.asarray(_extract_column(data, response[1]), dtype=np.float64).ravel()
    n = time.size
    if n == 0 or not np.all(np.isfinite(time)) or np.any(time < 0):
        raise ValueError("survival times must be finite and non-negative")
    if event.shape != (n,) or np.any((event != 0) & (event != 1)):
        raise ValueError(
            f"event status in {response[1]!r} must be coded 0 (censored) or 1 (event)")
    evaluation_weights = _check_weights(weights, n)

    linpred = np.asarray(model.predict(newdata=data, type="link"), dtype=np.float64).ravel()
    survival = model._survival_from_linpred(horizon_values, linpred)

    rows = []
    below = []
    for position, horizon in enumerate(horizon_values):
        brier, g_brier, floor_brier = ipcw_brier_at(
            time, event, evaluation_weights,
            np.ascontiguousarray(survival[:, position]), float(horizon), censoring_floor)
        auc, g_auc, floor_auc, case_weight, control_weight = ipcw_auc_at(
            time, event, evaluation_weights, linpred, float(horizon), censoring_floor)
        at_risk = evaluation_weights[time > horizon]
        event_by = (event == 1) & (time <= horizon)
        if floor_brier or floor_auc:
            below.append(float(horizon))
        rows.append({
            "horizon": float(horizon),
            "brier": float(brier),
            "auc": float(auc),
            "censoring_survival": float(g_brier),
            "below_censoring_floor": bool(floor_brier or floor_auc),
            "case_weight": float(case_weight),
            "control_weight": float(control_weight),
            "events_by_horizon": float(evaluation_weights[event_by].sum()),
            "at_risk_at_horizon": float(at_risk.sum()),
            "predicted_survival_mean": float(
                np.sum(evaluation_weights * survival[:, position])
                / evaluation_weights.sum()),
        })

    assumptions = [
        "Predicted survival uses the Breslow baseline stored with the fitted model, "
        "not the rows evaluated here.",
        "Brier and AUC weight each row by inverse probability of censoring; the "
        "censoring survival is a Kaplan-Meier estimate on the evaluated rows.",
        "The AUC ranks subjects by the linear predictor; the Brier score uses the "
        "absolute survival probability.",
    ]
    if data_role == "external":
        assumptions.append(
            "data_role='external' is the caller's declaration that these outcomes "
            "were not used to fit, tune or select the model.")
    else:
        assumptions.append(
            "data_role='training' diagnoses fit on rows the model was fitted to; it "
            "does not estimate prediction on new subjects.")
    if below:
        listed = ", ".join(f"{t:g}" for t in below)
        message = (
            f"censoring survival G(t) is below censoring_floor={censoring_floor:g} at "
            f"horizon(s) {listed}; IPCW weights there are unstable. Use earlier "
            "horizons, or pass on_low_censoring='warn' to keep these values."
        )
        if on_low_censoring == "error":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        assumptions.append(
            f"Horizons {listed} have G(t) below censoring_floor={censoring_floor:g}; "
            "their IPCW estimates are unstable.")

    return SurvivalEvaluationResult(
        family=type(model._family).__name__,
        response=response,
        data_role=data_role,
        data_fingerprint=data_fingerprint(data),
        horizons=tuple(float(t) for t in horizon_values),
        n_observations=n,
        total_weight=float(evaluation_weights.sum()),
        censoring_floor=censoring_floor,
        on_low_censoring=on_low_censoring,
        time=time,
        event=event,
        weights=evaluation_weights,
        linear_predictor=linpred,
        survival=survival,
        horizon_rows=tuple(rows),
        assumptions=tuple(assumptions),
    )
