"""Simultaneous multi-quantile boosting via shared Rust learner blueprints."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from mboostrs import _rust
from mboostrs._comparison_identity import fit_identity
from mboostrs._control import boost_control
from mboostrs._frame import column_names, read_column
from mboostrs._weights import normalize_weights
from mboostrs.cvrisk import _with_model_weights, cv_weights
from mboostrs.families import Quantile
from mboostrs.folds import FoldScheme
from mboostrs.formula import apply_na_action, expand_formula, parse_gamboost_formula
from mboostrs.gamboost import (
    GAMBoostModel,
    LearnerLayout,
    _bols_form_for,
    _display_names,
    _engine_kind_for_spec,
    _extract_column,
    _intercept_only,
)
from mboostrs.gamlss import learner_spec
from mboostrs.interactions import expand_by_factors, expand_interactions
from mboostrs.interactions import factor_columns as _factor_columns_of
from mboostrs.learners import Btree, Buser, apply_global_bins, resolve_auto_bins
from mboostrs.rearrange import RearrangedQuantiles, rearrange_quantiles
from mboostrs.report import ExecutionReport
from mboostrs.transforms import (
    TransformRecipe,
    apply_recipe,
    encode_factors,
    extract_transforms,
    factor_term_labels,
    fit_recipe,
    impute_missing,
    lower_categorical_bols,
    switched_term_labels,
)

__all__ = ["MultiQuantileModel", "quantile_boost"]


def _interp_quantile(taus: np.ndarray, q_mat: np.ndarray, target_tau: float) -> np.ndarray:
    """Vectorized linear interpolation across sorted quantile columns ``q_mat`` (n, Q)."""
    if q_mat.shape[1] == 1:
        return q_mat[:, 0].copy()
    t = float(target_tau)
    if t <= float(taus[0]):
        return q_mat[:, 0].copy()
    if t >= float(taus[-1]):
        return q_mat[:, -1].copy()
    idx = int(np.searchsorted(taus, t, side="right"))
    lo_idx = max(0, idx - 1)
    hi_idx = min(len(taus) - 1, idx)
    t_lo = float(taus[lo_idx])
    t_hi = float(taus[hi_idx])
    if abs(t_hi - t_lo) <= 1e-15:
        return q_mat[:, lo_idx].copy()
    alpha = (t - t_lo) / (t_hi - t_lo)
    return (1.0 - alpha) * q_mat[:, lo_idx] + alpha * q_mat[:, hi_idx]


def _validate_quantiles(quantiles: Any) -> tuple[float, ...]:
    """Validate and sort distinct quantile levels strictly in (0, 1)."""
    if isinstance(quantiles, (str, bytes)):
        raise ValueError("quantiles must be a sequence of probabilities in (0, 1)")
    try:
        vals = [float(q) for q in quantiles]
    except TypeError as exc:
        raise ValueError("quantiles must be a non-empty 1D sequence of probabilities in (0, 1)") from exc
    if not vals:
        raise ValueError("quantiles must not be empty")
    for q in vals:
        if not (math.isfinite(q) and 0.0 < q < 1.0):
            raise ValueError(f"every quantile must lie strictly in (0, 1), got {q!r}")
    sorted_vals = sorted(vals)
    for a, b in zip(sorted_vals[:-1], sorted_vals[1:], strict=True):
        if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"quantiles must be distinct, got duplicate around {a!r}")
    return tuple(sorted_vals)


def _resolve_response_scale_opt(response_scale: Any) -> float | None:
    """Map ``response_scale`` to the float/None convention of Rust multi-quantile fit."""
    if response_scale is None or response_scale == "none":
        return None
    if response_scale == "auto":
        return 0.0
    val = float(response_scale)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(
            "response_scale must be 'auto', 'none' (or None), or a non-negative finite float; "
            f"got {response_scale!r}"
        )
    return val


def _resolve_fold_matrix(folds: Any, n: int, seed: int, w: np.ndarray, has_weights: bool) -> np.ndarray | None:
    """Resolve `folds` to an `(n, B)` contiguous float64 weight matrix or `None`."""
    if folds is None:
        return None
    if isinstance(folds, bool):
        raise ValueError("folds cannot be a bool")
    if isinstance(folds, (int, np.integer)):
        k = int(folds)
        if k == 0:
            return None
        if k < 2:
            raise ValueError(f"folds must be >= 2 when an integer (or None/0 for no CV), got {k}")
        mat = np.asarray(
            cv_weights(n, type="kfold", B=k, seed=int(seed), weights=(w if has_weights else None)),
            dtype=np.float64,
        )
        return np.ascontiguousarray(mat, dtype=np.float64)
    if isinstance(folds, FoldScheme):
        mat = np.asarray(folds.weights, dtype=np.float64)
    else:
        mat = np.asarray(folds, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != n or mat.shape[1] < 1:
        raise ValueError(f"folds must have shape ({n}, B) with B >= 1, got {mat.shape}")
    if has_weights:
        mat = _with_model_weights(mat, w)
    return np.ascontiguousarray(mat, dtype=np.float64)


class MultiQuantileModel:
    """Simultaneously fitted conditional quantile models sharing one learner basis."""

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...],
        models: dict[float, GAMBoostModel],
        optimal_mstop: dict[float, int],
        cv_risk: dict[float, np.ndarray],
        formula: str,
        response: str,
    ) -> None:
        self.quantiles = tuple(float(q) for q in quantiles)
        self.models = dict(models)
        self.optimal_mstop = {float(k): int(v) for k, v in optimal_mstop.items()}
        self.cv_risk = {float(k): np.asarray(v, dtype=np.float64) for k, v in cv_risk.items()}
        self.formula = str(formula)
        self.response = str(response)

    def _raw_matrix(self, newdata: Any = None, *, unseen_levels: str = "average") -> np.ndarray:
        cols = [
            np.asarray(
                self.models[q].predict(newdata, type="link", unseen_levels=unseen_levels),
                dtype=np.float64,
            ).ravel()
            for q in self.quantiles
        ]
        return np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)

    def _rearranged_matrix(self, newdata: Any = None, *, unseen_levels: str = "average") -> np.ndarray:
        matrix = self._raw_matrix(newdata, unseen_levels=unseen_levels)
        if matrix.shape[1] < 2:
            return matrix
        sorted_m, *_ = _rust.rearrange_levels(matrix)
        return np.asarray(sorted_m, dtype=np.float64)

    def predict(
        self,
        newdata: Any = None,
        *,
        rearrange: bool = True,
        unseen_levels: str = "average",
        output: str = "auto",
    ) -> RearrangedQuantiles | np.ndarray:
        """Predict all quantile levels on ``newdata``, rearranging crossings by default."""
        if rearrange and len(self.quantiles) >= 2:
            return rearrange_quantiles(
                [self.models[q] for q in self.quantiles],
                newdata,
                output=output,
                unseen_levels=unseen_levels,
            )
        return self._raw_matrix(newdata, unseen_levels=unseen_levels)

    def predict_mean(self, newdata: Any = None, *, unseen_levels: str = "average") -> np.ndarray:
        """Estimate conditional mean via probability-weighted integration over rearranged quantiles."""
        q_mat = self._rearranged_matrix(newdata, unseen_levels=unseen_levels)
        taus = np.asarray(self.quantiles, dtype=np.float64)
        if taus.size == 1:
            return q_mat[:, 0].copy()
        weights = np.empty(taus.size, dtype=np.float64)
        weights[0] = taus[0] + 0.5 * (taus[1] - taus[0])
        if taus.size > 2:
            weights[1:-1] = 0.5 * (taus[2:] - taus[:-2])
        weights[-1] = (1.0 - taus[-1]) + 0.5 * (taus[-1] - taus[-2])
        return np.asarray(q_mat @ weights, dtype=np.float64)

    def predict_interval(
        self,
        newdata: Any,
        *,
        level: float = 0.90,
        calibration: Any = None,
        unseen_levels: str = "average",
    ) -> dict[str, Any]:
        """Predict central quantile intervals, optionally calibrated via CQR (Romano et al., 2019)."""
        lvl = float(level)
        if not (math.isfinite(lvl) and 0.0 < lvl < 1.0):
            raise ValueError(f"level must be strictly in (0, 1), got {level!r}")
        alpha = 1.0 - lvl
        tau_lo = alpha / 2.0
        tau_hi = 1.0 - alpha / 2.0
        taus = np.asarray(self.quantiles, dtype=np.float64)

        q_mat = self._rearranged_matrix(newdata, unseen_levels=unseen_levels)
        lo = _interp_quantile(taus, q_mat, tau_lo)
        hi = _interp_quantile(taus, q_mat, tau_hi)
        median_pred = _interp_quantile(taus, q_mat, 0.5)

        q_pad = 0.0
        method = "quantile"
        if calibration is not None:
            y_cal = np.asarray(read_column(calibration, self.response).values, dtype=np.float64).ravel()
            if y_cal.size == 0 or not np.all(np.isfinite(y_cal)):
                raise ValueError("calibration response must be non-empty and finite")
            cal_mat = self._rearranged_matrix(calibration, unseen_levels=unseen_levels)
            lo_cal = _interp_quantile(taus, cal_mat, tau_lo)
            hi_cal = _interp_quantile(taus, cal_mat, tau_hi)
            scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
            q_pad = float(_rust.conformal_quantile(np.ascontiguousarray(scores, dtype=np.float64), lvl))
            method = "cqr"

        return {
            "lower": lo - q_pad,
            "upper": hi + q_pad,
            "predicted": median_pred,
            "level": lvl,
            "cqr_adjustment": float(q_pad),
            "method": method,
        }

    def crps(
        self,
        newdata: Any,
        y_true: Any = None,
        *,
        unseen_levels: str = "average",
    ) -> float:
        """Mean pinball-integrated CRPS via trapezoidal rule across the fitted quantile grid."""
        if y_true is None:
            y = np.asarray(read_column(newdata, self.response).values, dtype=np.float64).ravel()
        else:
            y = np.asarray(y_true, dtype=np.float64).ravel()
        q_mat = self._rearranged_matrix(newdata, unseen_levels=unseen_levels)
        if y.shape[0] != q_mat.shape[0]:
            raise ValueError(f"y_true length {y.shape[0]} does not match predictions length {q_mat.shape[0]}")
        taus = np.asarray(self.quantiles, dtype=np.float64)
        err = y[:, None] - q_mat
        pinball = np.where(err >= 0.0, taus[None, :] * err, (taus[None, :] - 1.0) * err)
        if taus.size == 1:
            return float(np.mean(2.0 * pinball[:, 0]))
        trap = getattr(np, "trapezoid", None) or np.trapz
        return float(np.mean(2.0 * trap(pinball, taus, axis=1)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary dictionary of the multi-quantile model."""
        return {
            "formula": self.formula,
            "response": self.response,
            "quantiles": list(self.quantiles),
            "optimal_mstop": {str(k): int(v) for k, v in self.optimal_mstop.items()},
            "cv_risk": {
                str(k): [float(x) for x in v.tolist()]
                for k, v in self.cv_risk.items()
            },
        }

    def __repr__(self) -> str:
        qs = ", ".join(f"{q:g}" for q in self.quantiles)
        return (
            f"MultiQuantileModel(quantiles=({qs}), response={self.response!r}, "
            f"mstop={self.optimal_mstop})"
        )

    def digest(self) -> str:
        """Return a concise, deterministic summary card of this MultiQuantileModel."""
        qs = ", ".join(f"{q:g}" for q in self.quantiles)
        first_model = next(iter(self.models.values()), None)
        n_obs = len(first_model._fitted) if first_model is not None else 0
        q_mid = min(self.quantiles, key=lambda q: abs(q - 0.5)) if self.quantiles else None
        cv_parts = []
        for q in self.quantiles:
            arr = self.cv_risk.get(q)
            if arr is not None and len(arr):
                idx = min(int(self.optimal_mstop.get(q, len(arr) - 1)), len(arr) - 1)
                cv_parts.append(f"q{q:g}={float(arr[idx]):.4g}")
        cv_text = ", ".join(cv_parts[:5]) if cv_parts else "none"
        if len(cv_parts) > 5:
            cv_text += f", (+{len(cv_parts) - 5} more)"
        lines = [
            f"[mboostrs MultiQuantileModel] response={self.response}, n={n_obs}, "
            f"quantiles=({qs}), formula={self.formula}",
            f"stopping: optimal_mstop={self.optimal_mstop}, oof_pinball=({cv_text})",
        ]
        if q_mid is not None and q_mid in self.models:
            m_mid = self.models[q_mid]
            counts: dict[str, int] = {}
            for idx in m_mid._selected:
                nm = m_mid._learner_names[int(idx)]
                counts[nm] = counts.get(nm, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = ", ".join(f"{nm}({c})" for nm, c in ranked[:8]) or "none"
            if len(ranked) > 8:
                shown += f", (+{len(ranked) - 8} more)"
            inactive = max(0, len(m_mid._learner_names) - len(counts))
            lines.append(f"active_terms[q={q_mid:g}]: {shown}; inactive={inactive}")
        lines.append("diagnostics: rearrangement=isotonic_available")
        return "\n".join(lines)


def quantile_boost(
    formula: str,
    data: Any,
    quantiles: Any = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    *,
    weights: Any = None,
    folds: Any = 5,
    max_mstop: int = 500,
    nu: float = 0.3,
    bins: int | str | None = "auto",
    response_scale: str | float | None = "auto",
    seed: int = 42,
    na_action: str = "impute",
) -> MultiQuantileModel:
    """Simultaneously CV-tune and fit multiple conditional quantile models in one Rust pass."""
    taus = _validate_quantiles(quantiles)
    max_mstop_int = int(max_mstop)
    if max_mstop_int < 1:
        raise ValueError(f"max_mstop must be >= 1, got {max_mstop!r}")
    nu_float = float(nu)
    if not (math.isfinite(nu_float) and 0.0 < nu_float <= 1.0):
        raise ValueError(f"nu must be in (0, 1], got {nu!r}")
    resp_scale_opt = _resolve_response_scale_opt(response_scale)

    if na_action not in ("error", "omit", "impute", "impute_model", "impute_model_split"):
        raise ValueError(f"invalid na_action={na_action!r}")

    _na_dropped: list[int] = []
    if na_action == "error":
        apply_na_action(formula, data, na_action)
    else:
        data, _na_dropped = apply_na_action(formula, data, na_action)
        if _na_dropped and weights is not None:
            w_raw = np.asarray(weights, dtype=np.float64)
            keep_mask = np.ones(len(w_raw), dtype=bool)
            keep_mask[_na_dropped] = False
            weights = w_raw[keep_mask]

    env: dict[str, Any] | None = None
    _raw_formula = formula
    _raw_data = data

    _model_imputed = None
    if na_action in ("impute_model", "impute_model_split"):
        formula, _model_imputed = impute_missing(
            expand_formula(formula, data), data,
            learner="bols", model_based=True,
            split_filled=(na_action == "impute_model_split"),
        )
        data = apply_recipe(_model_imputed, data)

    _interactions: list[Any] = []
    if "%X%" in formula:
        formula, _ix_env, _interactions = expand_interactions(
            formula, data,
            na_action="impute" if na_action in ("impute_model", "impute_model_split") else na_action,
        )
        if _interactions:
            data = dict(data) if isinstance(data, dict) else {c: np.asarray(data[c]) for c in data.columns}
            env = {}
            _ix_rows = next(m.shape[0] for m in _ix_env.values() if m is not None)
            for name, matrix in _ix_env.items():
                if matrix is None:
                    data[name] = np.zeros(_ix_rows)
                else:
                    env[name] = matrix

    offset_only = _intercept_only(formula, data)
    formula = expand_formula(formula, data)
    imputed = _model_imputed
    if na_action == "impute":
        formula, imputed = impute_missing(formula, data, learner="bols")
    formula, _recipe = extract_transforms(formula, data)
    if imputed is not None:
        _recipe = TransformRecipe(imputed.steps + _recipe.steps)
        data = apply_recipe(imputed, data)

    factor_labels = factor_term_labels(formula, data)
    formula = lower_categorical_bols(formula, data)
    _factor_columns = _factor_columns_of(data)
    _recipe = TransformRecipe(_recipe.steps + encode_factors(formula, data, factor_bols=True).steps)
    _response_transform = None
    if _recipe:
        _recipe = fit_recipe(_recipe, data)
        data = apply_recipe(_recipe, data)
        from mboostrs.formula import parse_response

        _resp = parse_response(formula)
        if isinstance(_resp, str):
            resp_steps = [s for s in _recipe.steps if s.name == _resp]
            if resp_steps:
                _response_transform = resp_steps[0]
                _recipe = TransformRecipe(tuple(s for s in _recipe.steps if s.name != _resp))

    response_name, formula_learners = parse_gamboost_formula(formula, env=env, data=data)
    if offset_only:
        formula_learners = []
    learners = formula_learners

    _factor_levels = {step.name: len(step.params[0]) for step in _recipe.steps if step.kind == "factor"}
    for learner in learners:
        if isinstance(learner, Btree):
            learner.categorical = [_factor_levels.get(name, 0) for name in learner.names]

    learners, _by_recipes, _by_display = expand_by_factors(learners, data, _factor_columns)
    if _by_recipes:
        _interactions = list(_interactions) + _by_recipes

    y = np.ascontiguousarray(_extract_column(data, response_name), dtype=np.float64)
    n = int(len(y))

    _bin_decisions: list[dict] = []
    if bins == "auto":
        if n >= 512:
            learners, _ = apply_global_bins(learners, 256)
    elif bins is not None:
        learners, _ = apply_global_bins(learners, int(bins))

    if any(getattr(learner, "bins", None) == "auto" for learner in learners):
        available = set(column_names(data))
        distinct: dict[str, int] = {}
        n_rows_seen = 0
        for learner in learners:
            name = getattr(learner, "name", None)
            if name is None or name in distinct or name not in available:
                continue
            col = np.asarray(_extract_column(data, name), dtype=np.float64)
            n_rows_seen = max(n_rows_seen, int(col.size))
            distinct[name] = int(np.unique(col[np.isfinite(col)]).size)
        learners, _bin_decisions = resolve_auto_bins(learners, distinct, max(n_rows_seen, n))

    has_weights = weights is not None
    if has_weights:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape != (n,):
            raise ValueError(f"weights has shape {w_arr.shape}; expected ({n},)")
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("Fitting weights must be finite")
    normalized = normalize_weights(weights, n, policy="upstream")
    w = np.ascontiguousarray(normalized.fit, dtype=np.float64)

    fold_matrix = _resolve_fold_matrix(folds, n, int(seed), w, has_weights)

    raw_learner_names = [
        learner.names[0] if isinstance(learner, Btree) else learner.name
        for learner in learners
    ]
    learner_names = _display_names(_recipe, raw_learner_names)
    learner_names = [
        factor_labels.get(raw, name)
        for raw, name in zip(raw_learner_names, learner_names, strict=True)
    ]
    _term_display = dict(_by_display)
    _term_display.update({
        inter.derived: inter.name
        for inter in _interactions
        if inter.derived not in _term_display
    })
    learner_names = [_term_display.get(name, name) for name in learner_names]
    _switch_labels = switched_term_labels(learners, _recipe)
    learner_names = [_switch_labels.get(pos, name) for pos, name in enumerate(learner_names)]
    learner_types = [_engine_kind_for_spec(learner) for learner in learners]

    learner_layout = LearnerLayout.from_mapping(range(len(learners)))
    rust_idx_to_formula_idx = list(learner_layout.rust_idx_to_formula_idx)
    formula_idx_to_rust_idx = dict(learner_layout.formula_idx_to_rust_idx)

    bols_form = _bols_form_for(learners, vector_offset=False, kronecker=False, family=Quantile(taus[0]))
    specs = [
        learner_spec(learner, data, bols_form=bols_form)
        for learner in learners
    ]

    buser_centering_means: dict[str, list[float]] = {}
    for spec in learners:
        if isinstance(spec, Buser):
            if spec.center:
                sw = float(np.sum(w))
                means = np.sum(spec.X * w[:, None], axis=0) / sw
                buser_centering_means[spec.name] = means.tolist()
            else:
                buser_centering_means[spec.name] = [0.0] * spec.X.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _rust.cv_and_fit_multi_quantile_specs(
            specs,
            y,
            w,
            list(taus),
            fold_matrix,
            max_mstop_int,
            nu_float,
            resp_scale_opt,
        )

    models: dict[float, GAMBoostModel] = {}
    optimal_mstop: dict[float, int] = {}
    cv_risk: dict[float, np.ndarray] = {}

    for item in results:
        tau = float(item["tau"])
        opt_m = int(item["opt_mstop"])
        cv_r = np.asarray(item["cv_risk"], dtype=np.float64)
        (
            offset_val,
            selected,
            coefs_vecs,
            risk,
            fitted,
            nuisance_vec,
            nuisance_vec_path,
            learner_store,
            execution_report,
            residual,
        ) = item["fit"]

        fam = Quantile(tau)
        ctrl = boost_control(opt_m, nu_float)
        comparison_identity = fit_identity(
            data=data,
            response_name=response_name,
            response_transform=_response_transform,
            weights=w,
            weight_policy=normalized.policy,
            dropped=_na_dropped,
            family=fam,
        )
        m = GAMBoostModel(
            formula=formula,
            family=fam,
            control=ctrl,
            response_name=response_name,
            learner_names=list(learner_names),
            learner_types=list(learner_types),
            fitted=np.asarray(fitted, dtype=np.float64),
            risk=np.asarray(risk, dtype=np.float64),
            selected=[learner_layout.rust_to_formula(s) for s in selected],
            coefficients=[np.array(c, dtype=np.float64) for c in coefs_vecs],
            offset=float(offset_val),
            nu=nu_float,
            data=data,
            weights=w,
            learners_spec=learners,
            rust_idx_to_formula_idx=rust_idx_to_formula_idx,
            formula_idx_to_rust_idx=formula_idx_to_rust_idx,
            buser_centering_means=buser_centering_means if buser_centering_means else None,
            family_sigma=nuisance_vec if nuisance_vec else None,
            family_sigma_path=nuisance_vec_path if nuisance_vec_path else None,
            learner_store=learner_store,
            execution_report=ExecutionReport.from_mapping(execution_report),
            residual=residual,
        )
        m._transform_recipe = _recipe if _recipe else None
        m._response_transform = _response_transform
        m._interactions = _interactions or None
        m._na_dropped = _na_dropped
        m._comparison_identity = comparison_identity
        m._bin_decisions = _bin_decisions
        m._set_weight_policy(normalized)
        m._refit_bundle = {
            "formula": formula,
            "data": data,
            "family": fam,
            "weights": weights,
            "offset": None,
            "env": env,
            "bins": None if bins == "auto" else bins,
            "recipe": None,
            "interactions": _interactions or None,
            "subsample": None,
            "subsample_seed": None,
            "na_action": na_action,
            "weight_policy": "upstream",
            "raw_formula": _raw_formula,
            "raw_data": _raw_data,
        }
        models[tau] = m
        optimal_mstop[tau] = opt_m
        cv_risk[tau] = cv_r

    return MultiQuantileModel(
        quantiles=taus,
        models=models,
        optimal_mstop=optimal_mstop,
        cv_risk=cv_risk,
        formula=_raw_formula,
        response=str(response_name),
    )
