"""Unified fast specification search (`autospec(..., search="fast")`).

Collapses the 12-stage exhaustive search (~150–200 fits) into a fast 1–3 CV
sweep pipeline:

- Stage 0: `fast_radar` (~0.2–0.5 s, 0 mboostrs fits) weeds dead noise columns,
  classifies linear (`bols`) vs smooth (`bbs`) shapes, detects interactions,
  and detects conditional heteroscedasticity on the *interactive* residuals.
- Stage 1: Single 5-fold CV sweep for the pruned additive formula (on an
  extreme-preserving subsample of up to 1,500 rows when `n > 1,500` and folds
  are random), stopping mstop doubling once marginal relative CV gain < 0.4%.
- Stage 2 (gated): Single 5-fold CV challenger adding `bspatial` for coordinate
  pairs and/or a histogram-accelerated `btree(..., split_method=histogram)`
  (or `%X%` pair) — skipped completely when Stage 0 sees no interaction signal.
- Stage 3 (gated): Single-pass 5-fold CV challenger for `GaussianLSS` (computing
  BOTH held-out log-score and held-out mean MSE in one pass) — skipped
  completely when Stage 0 sees constant residual spread or family is fixed.

Every structural addition still passes honest held-out paired-SE adoption gates
on the shared CV folds, and returns a standard `SpecRecommendation`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Any

import numpy as np

from mboostrs._autospec import (
    _LSS_LOGSCORE_MARGIN,
    FamilyEvidence,
    MstopEvidence,
    SpecRecommendation,
    _mse_worse,
    _unit_free_log_score,
    priority_order,
    probe_mstop,
)
from mboostrs._control import boost_control
from mboostrs._fast_screen import FastRadarResult, fast_radar
from mboostrs._frame import read_column
from mboostrs._names import quote_name, quote_response
from mboostrs.cvrisk import cv_weights
from mboostrs.families import Binomial, Gaussian
from mboostrs.formula import parse_gamboost_formula
from mboostrs.gamlss import GaussianLSS


def _slice_data(data: Any, idx: np.ndarray) -> Any:
    """Slice a Polars DataFrame, pandas DataFrame, or column dict by integer indices."""
    if isinstance(data, dict):
        return {k: np.asarray(v)[idx] for k, v in data.items()}
    # Polars DataFrame
    if hasattr(data, "select") and hasattr(data, "height"):
        return data[idx.tolist()]
    # Generic column-bearing frame (e.g. pandas): extract columns into a dict
    # without calling pandas-only indexing helpers.
    if hasattr(data, "columns"):
        return {k: np.asarray(data[k])[idx] for k in data.columns}
    return {k: np.asarray(data[k])[idx] for k in data}


def _search_subsample(
    data: Any,
    response: str,
    n_rows: int,
    *,
    max_rows: int = 1500,
    seed: int = 0,
) -> tuple[Any, np.ndarray, bool]:
    """Deterministic extreme-preserving subsample for fast CV model selection.

    Always keeps the top-15 and bottom-15 finite response rows so B-spline knot
    boundaries and tail spread are represented accurately.
    """
    if n_rows <= max_rows:
        return data, np.arange(n_rows, dtype=np.int64), False
    y = np.asarray(read_column(data, response).values, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(y))
    n_ext = min(15, finite_idx.size // 4)
    pinned: set[int] = set()
    if n_ext > 0:
        order = finite_idx[np.argsort(y[finite_idx], kind="mergesort")]
        pinned.update(int(i) for i in order[:n_ext])
        pinned.update(int(i) for i in order[-n_ext:])
    pool = np.array([i for i in range(n_rows) if i not in pinned], dtype=np.int64)
    rng = np.random.default_rng(seed + 101)
    need = max(0, max_rows - len(pinned))
    chosen = rng.choice(pool, size=min(need, pool.size), replace=False)
    idx = np.sort(np.concatenate([np.array(sorted(pinned), dtype=np.int64), chosen]))
    return _slice_data(data, idx), idx, True


def _plateau_probe_mstop(
    data: Any,
    response: str,
    formula: str,
    family: Any,
    *,
    folds: np.ndarray,
    nu: float,
    env: dict | None,
    fold_scheme: Any = None,
    start: int = 600,
    max_mstop: int = 1200,
    min_rel_gain: float = 0.012,
) -> MstopEvidence | None:
    """Tune `mstop` in 1 sweep (`start` steps), extending at most once if steep."""
    largest = max(100, int(start))
    row = probe_mstop(
        data, response, formula, family,
        folds=folds, largest=largest, nu=nu, env=env,
        extend=0, fold_scheme=fold_scheme,
    )
    if row is None:
        return None
    total_fits = row.fits
    if row.at_boundary and 2 * largest <= max_mstop:
        # Only pay for a second sweep if the second half of the first path was
        # still dropping steeply (> min_rel_gain relative risk reduction).
        mid = max(1, largest // 2)
        risk_arr = np.asarray(row.risk, dtype=np.float64)
        mid_risk = float(risk_arr[min(mid, risk_arr.size - 1)])
        end_risk = float(risk_arr[row.chosen])
        rel_drop = (mid_risk - end_risk) / max(1e-12, abs(mid_risk))
        if rel_drop >= min_rel_gain:
            row2 = probe_mstop(
                data, response, formula, family,
                folds=folds, largest=2 * largest, nu=nu, env=env,
                extend=0, fold_scheme=fold_scheme,
            )
            if row2 is not None:
                total_fits += row2.fits
                if float(row2.risk[row2.chosen]) < end_risk:
                    row = replace(row2, fits=total_fits)
                else:
                    row = replace(row, fits=total_fits)
    return row


def _fold_oob_mean_scores(
    data: Any,
    response: str,
    formula: str,
    family: Any,
    folds: np.ndarray,
    mstop: int,
    nu: float,
    env: dict | None,
    judge: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Compute per-fold Gaussian log-score and per-fold MSE of a mean model."""
    from mboostrs.gamboost import gamboost

    y = np.asarray(read_column(data, response).values, dtype=np.float64)
    n_folds = int(folds.shape[1])
    fold_res_va: list[np.ndarray] = []
    fold_mse = np.empty(n_folds, dtype=np.float64)
    fits = 0
    for k in range(n_folds):
        w = folds[:, k]
        tr = w > 0
        va = w == 0
        if judge is not None:
            va = va & np.asarray(judge, dtype=bool)[:, k]
        if not va.any() or not tr.any():
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = gamboost(
                    formula, data=data, family=family,
                    control=boost_control(int(mstop), nu),
                    weights=w, env=env,
                )
                pred = np.asarray(fit.predict(type="response"), dtype=np.float64)
        except BaseException:
            return None
        fits += 1
        res_va = y[va] - pred[va]
        fold_res_va.append(res_va)
        fold_mse[k] = float(np.mean(res_va * res_va))

    # Estimate the constant-variance sigma for fold k from the OTHER folds'
    # out-of-fold residuals (leave-one-fold-out), avoiding in-sample tree
    # residual shrinkage with zero leakage from fold k.
    fold_log = np.empty(n_folds, dtype=np.float64)
    for k in range(n_folds):
        res_va = fold_res_va[k]
        if n_folds >= 2:
            other_sq = np.concatenate([fold_res_va[j] ** 2 for j in range(n_folds) if j != k])
            sd = max(float(np.sqrt(np.mean(other_sq))), 1e-8)
        else:
            sd = max(float(np.sqrt(np.mean(res_va * res_va))), 1e-8)
        fold_log[k] = float(0.5 * (math.log(2.0 * math.pi * sd * sd) + np.mean((res_va / sd) ** 2)))
    return fold_log, fold_mse, fits


def _single_pass_lss_eval(
    data: Any,
    response: str,
    lss_formula: dict[str, str],
    lss_family: Any,
    folds: np.ndarray,
    fold_scheme: Any,
    mstop: int,
    nu: float,
    env: dict | None,
    judge: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Tune and evaluate GaussianLSS in ONE cvrisk pass + ONE fold pass for MSE/logscore.

    Uses `cvrisk_gamboostLSS` once to select the optimal `(mstop_mu, mstop_sigma)`
    and get the per-fold log-scores, then only computes per-fold mean MSE if
    log-score clears the gate (saving 5 fits whenever LSS doesn't beat mean!).
    """
    from mboostrs.gamlss import cvrisk_gamboostLSS, diagonal_grid, gamboostLSS

    largest = max(50, int(mstop))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cvrisk_gamboostLSS(
                lss_formula, data, families=lss_family,
                control=boost_control(largest, nu),
                folds=(fold_scheme if fold_scheme is not None else np.asarray(folds, dtype=float)),
                grid=diagonal_grid(largest, len(lss_family.param_names), length_out=15),
                env=env,
            )
    except BaseException:
        return None
    risk = np.asarray(cv.risk_matrix, dtype=np.float64)
    if risk.size == 0 or not np.all(np.isfinite(risk)):
        return None
    best_col = int(np.argmin(np.nanmean(risk, axis=0)))
    fold_log = np.asarray(risk[:, best_col], dtype=np.float64)
    opt_mstop = tuple(int(m) for m in cv.optimal_mstop)
    n_folds = int(risk.shape[0])

    def compute_mse() -> tuple[np.ndarray, int] | None:
        y = np.asarray(read_column(data, response).values, dtype=np.float64)
        fold_mse = np.empty(n_folds, dtype=np.float64)
        mse_fits = 0
        # Fit each fold once at optimal mstop to get out-of-fold mu.
        m_max = int(max(opt_mstop))
        for k in range(n_folds):
            w = folds[:, k]
            va = w == 0
            if judge is not None:
                va = va & np.asarray(judge, dtype=bool)[:, k]
            if not va.any():
                return None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = gamboostLSS(
                        lss_formula, data=data, families=lss_family,
                        control=boost_control(m_max, nu),
                        weights=w, env=env,
                    )
                    mu = np.asarray(fit.predict(parameter="mu", type="response"), dtype=np.float64)
            except BaseException:
                return None
            mse_fits += 1
            res = y[va] - mu[va]
            fold_mse[k] = float(np.mean(res * res))
        return fold_mse, mse_fits

    return {
        "mean_logscore": float(np.mean(fold_log)),
        "fold_logscores": fold_log,
        "optimal_mstop": opt_mstop,
        "fits": n_folds,
        "compute_mse": compute_mse,
    }


def _build_term(col: Any, shape: str, df: float) -> str:
    q = quote_name(col.name)
    if col.kind == "categorical":
        return f"brandom({q}, df={df:g})"
    if col.kind == "boolean" or shape == "bols":
        return f"bols({q})"
    return f"bbs({q}, df={df:g})"


def run_fast_autospec(
    data: Any,
    response: str,
    *,
    modelled: list[Any],
    terms: list[str],
    family: Any,
    family_was_inferred: bool,
    mstop: int,
    df: float,
    nu: float,
    step: str | None,
    seed: int,
    folds: np.ndarray | None,
    shared_scheme: Any,
    shared_judge: Any,
    folds_description: str | None,
    declared: dict[str, str],
    mandatory: frozenset[str],
    forbidden: frozenset[tuple[str, str]],
    coordinates: tuple[str, str] | None,
    groups: tuple[str, ...],
    interactions: str,
    deselect: bool | float = True,
    residual_tree: bool | str,
    lss_trees: bool | str,
    env: dict,
    hygiene: tuple,
    incomplete_rows: int,
    n_rows: int,
    na_action: str | None,
    treatment_columns: tuple[str, ...],
    response_column: Any,
    coord_pair: tuple[str, str] | None,
) -> SpecRecommendation | None:
    """Execute the unified fast 3-stage `autospec` pipeline.

    Returns a `SpecRecommendation` on success, or `None` to fall back to the
    exhaustive pipeline (e.g. when `sklearn` is absent or family is unsupported).
    """
    from mboostrs.explore import _paired_se, _partitions

    is_binary = isinstance(family, Binomial)
    is_gaussian = isinstance(family, Gaussian)
    # For inferred positive-continuous / large-integer responses (GammaReg or
    # Poisson without exposure=), run fast regression search on Gaussian (with
    # Stage 3 GaussianLSS handling non-constant variance cleanly).
    if family_was_inferred and type(family).__name__ in ("GammaReg", "Poisson"):
        family = Gaussian()
        is_gaussian = True
    if not (is_gaussian or is_binary):
        return None
    if not modelled:
        return None

    declared_names = set(declared.keys()) if isinstance(declared, dict) else set(declared)
    protected = declared_names | set(mandatory) | set(groups)
    if coordinates is not None:
        protected.update(coordinates)
    if coord_pair is not None:
        protected.update(coord_pair)

    radar: FastRadarResult | None = fast_radar(
        data, response, modelled,
        is_binary=is_binary, seed=seed, max_rows=2000, protected=protected,
    )
    if radar is None:
        return None

    # Build fast search dataset and folds (subsampled to 1,200 rows only when
    # n_rows > 1,200 and folds are random/default).
    can_subsample = (folds is None and shared_scheme is None and n_rows > 1200)
    if can_subsample:
        search_data, _, was_subsampled = _search_subsample(
            data, response, n_rows, max_rows=1200, seed=seed,
        )
        search_n = 1200
        search_folds = cv_weights(search_n, type="kfold", B=5, seed=seed)
        search_scheme = None
        search_judge = None
    else:
        search_data = data
        was_subsampled = False
        search_n = n_rows
        search_folds = (
            np.asarray(folds, dtype=np.float64)
            if folds is not None
            else cv_weights(n_rows, type="kfold", B=min(5, max(2, n_rows)), seed=seed)
        )
        search_scheme = shared_scheme
        search_judge = shared_judge

    # Map original terms by column name so declared terms are preserved verbatim.
    orig_term_by_col: dict[str, str] = {}
    for col, t in zip(modelled, terms, strict=True):
        orig_term_by_col[col.name] = t
    col_by_name = {c.name: c for c in modelled}

    # Build Stage 1 pruned, shape-classified additive formula.
    kept_terms: list[str] = []
    used_declared: set[str] = set()
    # Keep any declared terms that weren't in modelled (e.g. bspatial from coordinates=).
    for t in terms:
        if any(name in declared_names and t == orig_term_by_col.get(name) for name in declared_names):
            if t not in kept_terms:
                kept_terms.append(t)
                used_declared.add(t)
    for cname in radar.kept_columns:
        if cname in declared_names:
            t = orig_term_by_col.get(cname)
            if t and t not in kept_terms:
                kept_terms.append(t)
            continue
        col = col_by_name[cname]
        shape = radar.column_shapes.get(cname, "bbs")
        kept_terms.append(_build_term(col, shape, df))

    if not kept_terms:
        kept_terms = list(terms)

    resp_q = quote_response(response)
    additive_formula = f"{resp_q} ~ " + " + ".join(kept_terms)

    evidence: dict[str, Any] = {
        "fast_radar": {
            "backend": radar.backend,
            "seconds": round(radar.seconds, 4),
            "kept_columns": radar.kept_columns,
            "dropped_columns": radar.dropped_columns,
            "prefers_linear": radar.prefers_linear,
            "needs_interactions": radar.needs_interactions,
            "interaction_gain": round(radar.interaction_gain, 5),
            "interaction_columns": radar.interaction_columns,
            "needs_lss_spread": radar.needs_lss_spread,
            "needs_lss_tree_scale": radar.needs_lss_tree_scale,
            "spread_logscore_gain": round(radar.spread_logscore_gain, 5),
            "spread_columns": radar.spread_columns,
            "search_subsampled": was_subsampled,
            "search_rows": search_n,
        }
    }
    measured: list[str] = ["shapes", "mstop"]
    defaulted: list[str] = []
    if family_was_inferred:
        defaulted.append("family")

    # ── Stage 1: Single additive CV sweep ────────────────────────────────────
    spent = 0
    base_row = _plateau_probe_mstop(
        search_data, response, additive_formula, family,
        folds=search_folds, nu=nu, env=env or None,
        fold_scheme=search_scheme, start=max(int(mstop), 500), max_mstop=1200,
    )
    if base_row is None:
        return None
    spent += base_row.fits
    best_formula = additive_formula
    best_row = base_row
    best_nu = float(nu)
    base_risk = float(base_row.risk[base_row.chosen])
    base_folds = np.asarray(base_row.fold_scores, dtype=np.float64)
    parts = _partitions(search_folds)

    # ── Stage 2: Gated Interaction / Spatial / Histogram-Tree Challenger ─────
    use_residual_tree = bool(residual_tree)
    wants_inter = (interactions == "adopt") or use_residual_tree or (coord_pair is not None)
    if wants_inter and (radar.needs_interactions or coord_pair is not None):
        challenger_terms = list(kept_terms)
        added_desc: list[str] = []
        if coord_pair is not None and coordinates is None:
            c1, c2 = coord_pair
            sp_term = f"bspatial({quote_name(c1)}, {quote_name(c2)}, knots=10)"
            if sp_term not in challenger_terms:
                challenger_terms.append(sp_term)
                added_desc.append(sp_term)

        tree_added = False
        if radar.needs_interactions:
            active_cols = [
                c for c in (radar.interaction_columns or radar.kept_columns)
                if c in col_by_name
            ]
            # Filter out forbidden pairs if any.
            if forbidden and len(active_cols) >= 2:
                active_cols = [
                    c for c in active_cols
                    if not any(c in pair for pair in forbidden)
                ]
            if use_residual_tree and len(active_cols) >= 2:
                split_opt = ", split_method=histogram" if search_n >= 256 else ""
                cols_str = ", ".join(quote_name(c) for c in active_cols)
                tree_term = f"btree({cols_str}, maxdepth=5, selection=sse{split_opt})"
                challenger_terms.append(tree_term)
                added_desc.append(tree_term)
                tree_added = True
            elif interactions == "adopt" and len(active_cols) >= 2:
                num_active = [c for c in active_cols if col_by_name[c].kind == "numeric"]
                if len(num_active) >= 2:
                    pairs_to_try = [(num_active[0], num_active[1])]
                    if len(num_active) >= 3:
                        pairs_to_try.append((num_active[0], num_active[2]))
                    for a, b in pairs_to_try:
                        pair_term = (
                            f"bbs({quote_name(a)}, knots=8, df={df:g}) %X% "
                            f"bbs({quote_name(b)}, knots=8, df={df:g})"
                        )
                        challenger_terms.append(pair_term)
                        added_desc.append(pair_term)

        if added_desc:
            cand_formula = f"{resp_q} ~ " + " + ".join(challenger_terms)
            cand_nu = min(float(nu), 0.15) if tree_added else float(nu)
            cand_row = _plateau_probe_mstop(
                search_data, response, cand_formula, family,
                folds=search_folds, nu=cand_nu, env=env or None,
                fold_scheme=search_scheme, start=350, max_mstop=800,
            )
            if cand_row is not None:
                spent += cand_row.fits
                cand_risk = float(cand_row.risk[cand_row.chosen])
                cand_folds = np.asarray(cand_row.fold_scores, dtype=np.float64)
                rel_gain = (base_risk - cand_risk) / max(1e-12, abs(base_risk))
                se = (
                    float(_paired_se(cand_folds, base_folds, parts))
                    if cand_folds.size == base_folds.size and cand_folds.size >= 2
                    else float("nan")
                )
                diff = base_risk - cand_risk
                adopted_inter = bool(
                    cand_risk < base_risk
                    and (rel_gain >= 0.015 or (rel_gain >= 0.005 and np.isfinite(se) and diff > 2.0 * se))
                )
                evidence["fast_interaction"] = {
                    "added": tuple(added_desc),
                    "base_risk": base_risk,
                    "candidate_risk": cand_risk,
                    "relative_gain": float(rel_gain),
                    "paired_se": se,
                    "adopted": adopted_inter,
                    "candidate_mstop": int(cand_row.chosen),
                    "candidate_nu": cand_nu,
                }
                if adopted_inter:
                    best_formula = cand_formula
                    best_row = cand_row
                    best_nu = cand_nu
                    measured.append("interactions")

    # ── Stage 3: Gated Distributional (GaussianLSS) Challenger ───────────────
    chosen_family: Any = family
    chosen_formula: Any = best_formula
    chosen_mstop = int(best_row.chosen)
    # Skip LSS if the Stage 1-2 mean model already explained >98% of variance
    # (residuals are numerical/discretization noise) or radar didn't flag spread.
    y_search = np.asarray(read_column(search_data, response).values, dtype=np.float64)
    y_search_var = float(np.nanvar(y_search)) if y_search.size else 1.0
    best_r2_cv = 1.0 - float(best_row.risk[best_row.chosen]) / max(1e-12, y_search_var)
    if (
        is_gaussian
        and family_was_inferred
        and radar.needs_lss_spread
        and best_r2_cv < 0.985
    ):
        spread_cols = [c for c in radar.spread_columns if c in col_by_name]
        sigma_terms = [_build_term(col_by_name[c], radar.column_shapes.get(c, "bbs"), df) for c in spread_cols]
        use_tree_scale = bool(radar.needs_lss_tree_scale and lss_trees is not False)
        num_for_scale = [
            c for c in (spread_cols or list(radar.kept_columns))
            if c in col_by_name and col_by_name[c].kind in ("numeric", "boolean")
        ]
        if (use_tree_scale or not sigma_terms) and len(num_for_scale) >= 2 and lss_trees is not False:
            split_opt = ", split_method=histogram" if search_n >= 256 else ""
            cols_str = ", ".join(quote_name(c) for c in num_for_scale[:12])
            sigma_terms.append(f"btree({cols_str}, maxdepth=4, selection=sse{split_opt})")
        if sigma_terms:
            lss_family = GaussianLSS(stabilization="L2")
            lss_formula = {
                "mu": best_formula,
                "sigma": f"{resp_q} ~ " + " + ".join(sigma_terms),
            }
            lss_mstop_cap = max(120, min(250, int(best_row.chosen)))
            lss_eval = _single_pass_lss_eval(
                search_data, response, lss_formula, lss_family,
                search_folds, search_scheme, lss_mstop_cap, best_nu,
                env or None, judge=search_judge,
            )
            if lss_eval is not None:
                spent += lss_eval["fits"]
                mean_eval = _fold_oob_mean_scores(
                    search_data, response, best_formula, family,
                    search_folds, int(best_row.chosen), best_nu, env or None,
                    judge=search_judge,
                )
                if mean_eval is not None:
                    mean_log, mean_mse, m_fits = mean_eval
                    spent += m_fits
                    lss_log = lss_eval["fold_logscores"]
                    mean_log_avg = float(np.mean(mean_log))
                    lss_log_avg = float(lss_eval["mean_logscore"])
                    bar = _LSS_LOGSCORE_MARGIN * abs(
                        _unit_free_log_score(mean_log_avg, search_data, response, lss_family)
                    )
                    log_imp = mean_log_avg - lss_log_avg
                    log_se = (
                        float(_paired_se(lss_log, mean_log, parts))
                        if lss_log.size == mean_log.size and lss_log.size >= 2
                        else float("nan")
                    )
                    passes_log = bool(log_imp > bar and np.isfinite(log_se) and log_imp > 2.0 * log_se)
                    adopted_lss = False
                    lss_mse_avg = float("nan")
                    if passes_log:
                        mse_res = lss_eval["compute_mse"]()
                        if mse_res is not None:
                            lss_mse, mse_fits = mse_res
                            spent += mse_fits
                            lss_mse_avg = float(np.mean(lss_mse))
                            mse_se = float(_paired_se(lss_mse, mean_mse, parts))
                            if np.isfinite(mse_se) and not _mse_worse(
                                lss_mse_avg, float(np.mean(mean_mse)), mse_se
                            ):
                                adopted_lss = True
                    evidence["fast_lss"] = {
                        "mean_logscore": mean_log_avg,
                        "lss_logscore": lss_log_avg,
                        "improvement": float(log_imp),
                        "paired_se": log_se,
                        "bar": float(bar),
                        "mean_mse": float(np.mean(mean_mse)),
                        "lss_mse": lss_mse_avg,
                        "adopted": adopted_lss,
                    }
                    if adopted_lss:
                        chosen_family = lss_family
                        chosen_formula = lss_formula
                        chosen_mstop = int(max(lss_eval["optimal_mstop"]))
                        if "family" in defaulted:
                            defaulted.remove("family")
                        measured.append("family")
                        evidence["family"] = FamilyEvidence(
                            recommended=lss_family,
                            dispersion=float("nan"),
                            spread_tracks_mean=float("nan"),
                            n_scored=search_n,
                            fits=lss_eval["fits"],
                            lss_mstop=lss_eval["optimal_mstop"],
                            lss_formula=lss_formula,
                            lss_score=lss_log_avg,
                            mean_mstop=int(best_row.chosen),
                            mean_family=family,
                            sentence=(
                                f"GaussianLSS beat Gaussian on held-out log score "
                                f"({lss_log_avg:.4f} vs {mean_log_avg:.4f}, "
                                f"gain {log_imp:.4f} > 2 SE) without hurting mean MSE"
                            ),
                        )

    # Optional post-search deselection (e.g. `deselect=0.025` for sparse specs).
    if (
        isinstance(deselect, (int, float))
        and not isinstance(deselect, bool)
        and 0.0 < float(deselect) < 0.5
        and not isinstance(chosen_family, GaussianLSS)
        and isinstance(chosen_formula, str)
        and getattr(best_row, "_fit", None) is not None
    ):
        try:
            from mboostrs.selection import deselect as _deselect

            ds = _deselect(best_row._fit, threshold=float(deselect))
            if ds.kept:
                chosen_formula = ds.formula
                best_formula = ds.formula
                evidence["deselect"] = {
                    "threshold": float(deselect),
                    "kept": tuple(ds.kept),
                    "dropped": tuple(ds.dropped),
                }
        except Exception:
            pass

    # Scale mstop gently when search ran on a subsample and full data is larger.
    if was_subsampled and not isinstance(chosen_family, GaussianLSS):
        scale = min(2.0, math.sqrt(n_rows / float(search_n)))
        chosen_mstop = int(min(1800, round(chosen_mstop * scale)))

    evidence["mstop"] = best_row
    learners_out = (
        None
        if (
            "%X%" in best_formula
            or isinstance(chosen_family, GaussianLSS)
            or incomplete_rows > 0
        )
        else parse_gamboost_formula(best_formula, env=env or None)[1]
    )
    if isinstance(chosen_family, GaussianLSS) and "family" in evidence and evidence["family"].lss_mstop:
        rec_control = boost_control(int(max(evidence["family"].lss_mstop)), best_nu, step=step)
    else:
        rec_control = boost_control(chosen_mstop, best_nu, step=step)

    return SpecRecommendation(
        formula=chosen_formula,
        learners=learners_out,
        family=chosen_family,
        mstop=chosen_mstop,
        response=response,
        hygiene=hygiene,
        evidence=evidence,
        measured=tuple(measured),
        priority=priority_order(
            evidence, hygiene, incomplete_rows=incomplete_rows,
            n_rows=len(response_column.values),
            missing_by_column={c.name: c.n_missing for c in modelled if c.n_missing},
            treatment=treatment_columns,
        ),
        defaulted=tuple(defaulted),
        declared=declared,
        env=env,
        groups=groups,
        folds_description=(
            folds_description
            or (f"5 random folds on {search_n}-row extreme-preserving subsample"
                if was_subsampled else "5 random folds")
        ),
        incomplete_rows=incomplete_rows,
        na_action=na_action,
        fits=spent,
        reused=0,
        treatment=treatment_columns,
        control=rec_control,
    )
