"""Stable execution-report value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from mboostrs._output import json_safe, output_frame


@dataclass(frozen=True)
class PathRow:
    """One learner update in a fitted boosting path.

    Mirrors the core's `mboostrs_core::report::PathRow` field-for-field, and
    adds the two resolved names the core row cannot carry (it holds indices,
    not the name tables) plus `scaled_coefficient_norm`, which needs `nu` from
    the control object. `parameter_index` and `parameter` are deliberately
    separate fields: the index is the schema, the name is the enrichment.
    R's `path_frame()` emits the same columns in the same order.
    """

    iteration: int
    cycle: int | None
    parameter_index: int | None
    parameter: str | None
    learner_index: int
    learner: str
    risk: float | None
    risk_change: float | None
    coefficient_norm: float
    scaled_coefficient_norm: float
    nuisance: tuple[float, ...] = ()

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a row mapping suitable for JSON or a data frame.

        ``include_arrays`` is accepted for the uniform export signature; a
        path row holds no arrays.
        """
        data = json_safe(asdict(self))
        data["nuisance"] = list(self.nuisance)
        return data


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def standard_path_rows(
    *,
    selected,
    coefficients,
    risk,
    nu: float,
    learner_names,
    nuisance_path=None,
) -> list[PathRow]:
    """Derive standard-model rows from retained path state."""
    rows = []
    risk = np.asarray(risk, dtype=np.float64)
    nuisance_path = nuisance_path or []
    for step, (learner_index, coefficient) in enumerate(
        zip(selected, coefficients, strict=False), start=1
    ):
        current_risk = _finite_or_none(risk[step]) if step < len(risk) else None
        previous_risk = _finite_or_none(risk[step - 1]) if step - 1 < len(risk) else None
        coefficient_norm = float(np.linalg.norm(np.asarray(coefficient, dtype=np.float64)))
        nuisance = (
            tuple(float(value) for value in nuisance_path[step])
            if step < len(nuisance_path)
            else ()
        )
        rows.append(
            PathRow(
                iteration=step,
                cycle=None,
                parameter_index=None,
                parameter=None,
                learner_index=int(learner_index),
                learner=str(learner_names[int(learner_index)]),
                risk=current_risk,
                risk_change=(
                    previous_risk - current_risk
                    if previous_risk is not None and current_risk is not None
                    else None
                ),
                coefficient_norm=coefficient_norm,
                scaled_coefficient_norm=float(nu) * coefficient_norm,
                nuisance=nuisance,
            )
        )
    return rows


def path_rows_output(rows: list[PathRow], output: str):
    """Convert path rows to records, pandas, or polars output."""
    return output_frame([row.to_dict() for row in rows], output=output)


def lss_path_rows(
    *, params, param_names, selection_order, risk, nu: float, noncyclic: bool,
    nu_per_param=None, labels=None,
) -> list[PathRow]:
    """Derive globally ordered LSS parameter-update rows.

    ``labels`` maps a derived learner name (a ``%X%`` pair's ``__ix0``) to
    the term the user wrote.
    """
    labels = labels or {}
    cursors = [0] * len(params)
    risk = np.asarray(risk, dtype=np.float64)
    rows = []
    for step, parameter_index in enumerate(selection_order, start=1):
        parameter_index = int(parameter_index)
        local_step = cursors[parameter_index]
        cursors[parameter_index] += 1
        cycle = local_step + 1
        cycle_ends = step == len(selection_order) or int(selection_order[step]) <= parameter_index
        risk_index = step if noncyclic else cycle if cycle_ends else None
        current_risk = (
            _finite_or_none(risk[risk_index])
            if risk_index is not None and risk_index < len(risk)
            else None
        )
        previous_risk = (
            _finite_or_none(risk[risk_index - 1])
            if risk_index is not None and risk_index > 0 and risk_index - 1 < len(risk)
            else None
        )
        param = params[parameter_index]
        learner_index = int(param["selected"][local_step])
        learner_names = [labels.get(name, name) for name in param["predictor_names"]]
        # Which shape the stored path has, not which learner type it used. A
        # `glmboostLSS` parameter carries `coefs_flat` — one scalar per step —
        # and an intercept learner that is not in `predictor_names`; a
        # `gamboostLSS` parameter carries `coefs`, a coefficient vector per
        # step, with every learner already named.
        #
        # This used to branch on `param.get("kind") != "bbs"`, testing a value
        # from the learner-spec vocabulary against a dict whose `kind` is
        # `"gam"` or absent. No `gamboostLSS` parameter ever matched, so every
        # `gamboostLSS` fit took the `glmboostLSS` branch and `path_frame()`
        # raised `KeyError: 'coefs_flat'`.
        if "coefs_flat" in param:
            learner_names = ["(Intercept)", *learner_names]
            coefficient = np.asarray([param["coefs_flat"][local_step]], dtype=np.float64)
        else:
            coefficient = np.asarray(param["coefs"][local_step], dtype=np.float64)
        coefficient_norm = float(np.linalg.norm(coefficient))
        rows.append(
            PathRow(
                iteration=step,
                cycle=None if noncyclic else cycle,
                parameter_index=parameter_index,
                parameter=str(param_names[parameter_index]),
                learner_index=learner_index,
                learner=str(learner_names[learner_index]),
                risk=current_risk,
                risk_change=(
                    previous_risk - current_risk
                    if previous_risk is not None and current_risk is not None
                    else None
                ),
                coefficient_norm=coefficient_norm,
                scaled_coefficient_norm=float(
                    nu if nu_per_param is None else nu_per_param[parameter_index]
                ) * coefficient_norm,
            )
        )
    return rows


@dataclass(frozen=True)
class ExecutionReport:
    """Versioned summary of the numerical route used by a fitted model.

    Timing fields are deliberately excluded from equality comparisons so two
    otherwise identical fits compare equal across machines and repeated runs.
    ``total_ns`` measures the Rust core fit. ``basis_ns`` and
    ``factorization_ns`` split learner preparation, while ``binding_ns``
    measures the remaining frontend conversion; these intervals are exclusive.
    Route-specific phases that do not run are ``None``.
    """

    schema_version: int
    instrumentation: str
    route: str
    n_observations: int
    n_learners: int
    total_basis_columns: int
    requested_mstop: int
    completed_mstop: int
    effective_threads: int
    gram_eligible: bool
    fallback_reasons: tuple[str, ...] = ()
    scalar_learner_count: int = 0
    block_learner_count: int = 0
    tree_learner_count: int = 0
    cholesky_solver_count: int = 0
    lu_solver_count: int = 0
    regularized_solver_count: int = 0
    max_solver_jitter: float | None = None
    factor_cache_bytes: int = 0
    retained_learner_bytes: int = 0
    gram_cache_bytes: int = 0
    calibration_count: int = 0
    tie_selection_count: int = 0
    component_subsample_iterations: int = 0
    calibration_converged_count: int = 0
    calibration_limit_count: int = 0
    calibration_failure_count: int = 0
    nonfinite_candidate_fallbacks: int = 0
    setup_ns: int | None = field(default=None, compare=False)
    basis_ns: int | None = field(default=None, compare=False)
    factorization_ns: int | None = field(default=None, compare=False)
    cross_gram_ns: int | None = field(default=None, compare=False)
    iteration_ns: int | None = field(default=None, compare=False)
    reconstruction_ns: int | None = field(default=None, compare=False)
    binding_ns: int | None = field(default=None, compare=False)
    total_ns: int | None = field(default=None, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExecutionReport:
        data = dict(value)
        data["fallback_reasons"] = tuple(data.get("fallback_reasons", ()))
        return cls(**data)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable dictionary.

        ``include_arrays`` is accepted for the uniform export signature; the
        report holds counts and timings only.
        """
        data = json_safe(asdict(self))
        data["fallback_reasons"] = list(self.fallback_reasons)
        return data

    def for_mstop(self, completed_mstop: int) -> ExecutionReport:
        """Return report metadata adjusted for a sliced model."""
        return replace(self, completed_mstop=completed_mstop)

    def format_text(self) -> str:
        """Render a compact, human-readable view of the stable report fields."""
        fallbacks = ",".join(self.fallback_reasons) or "none"
        lines = [
            (
                "mboostrs execution: "
                f"route={self.route} iterations={self.completed_mstop}/"
                f"{self.requested_mstop} learners={self.n_learners} "
                f"basis_columns={self.total_basis_columns} "
                f"threads={self.effective_threads}"
            ),
            (
                f"  gram_eligible={str(self.gram_eligible).lower()} "
                f"fallbacks={fallbacks} "
                f"nonfinite_fallbacks={self.nonfinite_candidate_fallbacks}"
            ),
            (
                "  learners: "
                f"scalar={self.scalar_learner_count} block={self.block_learner_count} "
                f"tree={self.tree_learner_count}; solvers: "
                f"cholesky={self.cholesky_solver_count} lu={self.lu_solver_count} "
                f"regularized={self.regularized_solver_count} "
                f"max_jitter={self.max_solver_jitter or 0.0:.3e}; "
                f"cache_kib: learner_data={self.retained_learner_bytes / 1024:.1f} "
                f"factors={self.factor_cache_bytes / 1024:.1f} "
                f"gram={self.gram_cache_bytes / 1024:.1f}; "
                f"calibrations={self.calibration_count} "
                f"converged={self.calibration_converged_count} "
                f"limit={self.calibration_limit_count} "
                f"failed={self.calibration_failure_count}; "
                f"tie_selections={self.tie_selection_count}"
            ),
        ]
        if self.total_ns is not None:
            setup_ms = (self.setup_ns or 0) / 1_000_000
            basis_ms = (self.basis_ns or 0) / 1_000_000
            factorization_ms = (self.factorization_ns or 0) / 1_000_000
            iteration_ms = (self.iteration_ns or 0) / 1_000_000
            cross_gram_ms = (self.cross_gram_ns or 0) / 1_000_000
            reconstruction_ms = (self.reconstruction_ns or 0) / 1_000_000
            binding_ms = (self.binding_ns or 0) / 1_000_000
            total_ms = self.total_ns / 1_000_000
            lines.append(
                f"  timing_ms: setup={setup_ms:.3f} basis={basis_ms:.3f} "
                f"factorization={factorization_ms:.3f} cross_gram={cross_gram_ms:.3f} "
                f"iterations={iteration_ms:.3f} reconstruction={reconstruction_ms:.3f} "
                f"binding={binding_ms:.3f} total={total_ms:.3f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ExecutionReport(route={self.route!r}, "
            f"completed_mstop={self.completed_mstop}/{self.requested_mstop}, "
            f"n_learners={self.n_learners})"
        )

    def __str__(self) -> str:
        return self.format_text()
