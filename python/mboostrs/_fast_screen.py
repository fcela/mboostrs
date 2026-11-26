"""Fast single-pass variable weeding and structure radar for `autospec`.

Uses a lightweight 3-way held-out comparison on an 80/20 validation slice
(capped at 2,000 rows) to screen out pure-noise columns, classify linear vs.
smooth numeric effects, detect whether interactions/trees are worth testing,
and detect whether conditional variance moves with the predictors — all in
roughly 0.1–0.2 seconds before `mboostrs` runs any cross-validated fits.

Every adoption decision still goes through `mboostrs`'s own held-out CV gates;
the radar only proposes the compact candidate space and skips stages that have
no held-out signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FastRadarResult:
    """Summary of the fast 80/20 screening pass."""

    kept_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    column_shapes: dict[str, str]
    column_scores: dict[str, float]
    val_loss_null: float
    val_loss_linear: float
    val_loss_additive: float
    val_loss_interactive: float
    prefers_linear: bool
    interaction_gain: float
    needs_interactions: bool
    interaction_columns: tuple[str, ...]
    needs_lss_spread: bool
    needs_lss_tree_scale: bool
    spread_logscore_gain: float
    spread_columns: tuple[str, ...]
    backend: str
    seconds: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def _encode_matrix(data: Any, columns: list[Any]) -> tuple[np.ndarray, list[bool]]:
    """Build a finite float64 matrix plus a boolean categorical mask."""
    from mboostrs._frame import read_column

    n = len(read_column(data, columns[0].name).values)
    X = np.empty((n, len(columns)), dtype=np.float64)
    cat_mask: list[bool] = []
    for j, col in enumerate(columns):
        raw = read_column(data, col.name)
        vals = np.asarray(raw.values, dtype=np.float64)
        is_cat = col.kind == "categorical"
        cat_mask.append(is_cat)
        miss = ~np.isfinite(vals)
        if miss.any():
            if is_cat:
                vals = np.where(miss, -1.0, vals)
            else:
                present = vals[~miss]
                fill = float(np.median(present)) if present.size else 0.0
                vals = np.where(miss, fill, vals)
        X[:, j] = vals
    return X, cat_mask


def _fast_ridge(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
                cat_mask: list[bool]) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form standardized ridge regression + approximate marginal |t|-stats."""
    num_idx = [j for j, is_cat in enumerate(cat_mask) if not is_cat]
    p_total = X_tr.shape[1]
    tstats = np.zeros(p_total, dtype=np.float64)
    y_mean = float(np.mean(y_tr))
    if not num_idx:
        pred = np.full(y_va.shape[0], y_mean, dtype=np.float64)
        return float(np.mean((y_va - pred) ** 2)), tstats, pred

    Xn_tr = X_tr[:, num_idx]
    Xn_va = X_va[:, num_idx]
    mu = Xn_tr.mean(axis=0)
    sd = Xn_tr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z_tr = (Xn_tr - mu) / sd
    Z_va = (Xn_va - mu) / sd
    yc = y_tr - y_mean

    p = Z_tr.shape[1]
    n = Z_tr.shape[0]
    gram = Z_tr.T @ Z_tr
    rhs = Z_tr.T @ yc
    best_mse = float("inf")
    best_pred = np.full(y_va.shape[0], y_mean, dtype=np.float64)
    best_beta = np.zeros(p, dtype=np.float64)
    for alpha in (1e-2, 1.0, 25.0):
        A = gram + (alpha * max(1.0, n / 100.0)) * np.eye(p)
        try:
            beta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        pred = y_mean + Z_va @ beta
        mse = float(np.mean((y_va - pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_pred = pred
            best_beta = beta

    # Marginal |t| statistic for each numeric predictor (captures pure linear main effects).
    res = yc - Z_tr @ best_beta
    s2 = float(np.mean(res * res)) + 1e-12
    diag = np.maximum(np.diag(gram), 1e-12)
    se = np.sqrt(s2 / diag)
    for local_i, col_j in enumerate(num_idx):
        tstats[col_j] = abs(float(best_beta[local_i]) / float(se[local_i]))
    return best_mse, tstats, best_pred


def _detect_column_shapes(model_add: Any, X_tr: np.ndarray, columns: list[Any],
                          cat_mask: list[bool]) -> dict[str, str]:
    """Classify each numeric column as linear ('bols') or nonlinear ('bbs').

    Evaluates the strictly additive HGB model along a 15-point quantile grid
    for each numeric column (holding all other columns at their training
    medians); because the model is strictly additive, the 1D curve is exact up
    to an additive constant.
    """
    shapes: dict[str, str] = {}
    if X_tr.shape[0] < 20:
        return {c.name: ("brandom" if is_cat else "bbs") for c, is_cat in zip(columns, cat_mask, strict=True)}
    med = np.median(X_tr, axis=0)
    qs = np.linspace(0.05, 0.95, 15)
    for j, (col, is_cat) in enumerate(zip(columns, cat_mask, strict=True)):
        if is_cat:
            shapes[col.name] = "brandom"
            continue
        if col.kind == "boolean":
            shapes[col.name] = "bols"
            continue
        grid = np.unique(np.quantile(X_tr[:, j], qs))
        if grid.size < 4:
            shapes[col.name] = "bols"
            continue
        probe = np.tile(med, (grid.size, 1))
        probe[:, j] = grid
        f = np.asarray(model_add.predict(probe), dtype=np.float64).ravel()
        f_var = float(np.var(f))
        if f_var <= 1e-12:
            shapes[col.name] = "bols"
            continue
        # R^2 of a straight line fitted to the 1D partial effect curve.
        g_c = grid - grid.mean()
        denom = float(np.dot(g_c, g_c))
        if denom <= 1e-12:
            shapes[col.name] = "bols"
            continue
        slope = float(np.dot(g_c, f - f.mean())) / denom
        line_pred = f.mean() + slope * g_c
        r2_line = 1.0 - float(np.mean((f - line_pred) ** 2)) / f_var
        shapes[col.name] = "bols" if r2_line >= 0.985 else "bbs"
    return shapes


def _numpy_fallback_radar(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    columns: list[Any],
    cat_mask: list[bool],
    *,
    is_binary: bool,
    protected: set[str] | frozenset[str],
    t0: float,
) -> FastRadarResult:
    """Zero-dependency pure-NumPy screening fallback when sklearn is absent."""
    import time

    p = len(columns)
    val_null = float(np.mean((y_va - float(np.mean(y_tr))) ** 2))
    val_lin, tstats, lin_va = _fast_ridge(X_tr, y_tr, X_va, y_va, cat_mask)
    col_scores = {c.name: float(tstats[j]) for j, c in enumerate(columns)}
    shapes = {
        c.name: ("brandom" if is_cat else ("bols" if c.kind == "boolean" else "bbs"))
        for c, is_cat in zip(columns, cat_mask, strict=True)
    }
    # Keep all columns safely when no nonlinear tree screen is installed
    # (only weed extreme high-dimensional pure-noise columns with |t| < 1.0 when p > 20).
    if p <= 20:
        kept = [c.name for c in columns]
        dropped: list[str] = []
    else:
        kept = [
            c.name for j, c in enumerate(columns)
            if c.name in protected or cat_mask[j] or tstats[j] >= 1.5
        ]
        if len(kept) < min(p, 10):
            ranked = sorted(range(p), key=lambda j: tstats[j], reverse=True)
            kset = set(kept)
            for j in ranked:
                if len(kset) >= min(p, 10):
                    break
                kset.add(columns[j].name)
            kept = [c.name for c in columns if c.name in kset]
        dropped = [c.name for c in columns if c.name not in set(kept)]

    # Check linear log(res^2) correlation for heteroscedasticity.
    needs_spread = False
    spread_cols: tuple[str, ...] = ()
    if not is_binary:
        res_va = y_va - lin_va
        sd_va = max(float(np.std(res_va)), 1e-8)
        lr2 = np.log(np.maximum(res_va * res_va, (1e-3 * sd_va) ** 2))
        lr2_c = lr2 - lr2.mean()
        lr2_norm = float(np.linalg.norm(lr2_c))
        if lr2_norm > 1e-12:
            corrs = []
            for j, c in enumerate(columns):
                if cat_mask[j]:
                    continue
                xc = X_va[:, j] - X_va[:, j].mean()
                xn = float(np.linalg.norm(xc))
                if xn > 1e-12:
                    r = abs(float(np.dot(xc, lr2_c)) / (xn * lr2_norm))
                    if r >= 0.18:
                        corrs.append((r, c.name))
            corrs.sort(reverse=True)
            if corrs:
                needs_spread = True
                spread_cols = tuple(n for _, n in corrs[:6])

    return FastRadarResult(
        kept_columns=tuple(kept),
        dropped_columns=tuple(dropped),
        column_shapes=shapes,
        column_scores=col_scores,
        val_loss_null=val_null,
        val_loss_linear=val_lin,
        val_loss_additive=val_lin,
        val_loss_interactive=val_lin,
        prefers_linear=False,
        interaction_gain=0.0,
        needs_interactions=True,
        interaction_columns=tuple(kept[:16]),
        needs_lss_spread=needs_spread,
        needs_lss_tree_scale=False,
        spread_logscore_gain=0.0,
        spread_columns=spread_cols,
        backend="numpy.ridge_fallback",
        seconds=float(time.perf_counter() - t0),
    )


def fast_radar(
    data: Any,
    response: str,
    columns: list[Any],
    *,
    is_binary: bool = False,
    seed: int = 0,
    max_rows: int = 2_000,
    protected: set[str] | frozenset[str] = frozenset(),
) -> FastRadarResult | None:
    """Run the fast 80/20 screening pass (uses sklearn HGB if installed, else NumPy)."""
    import time

    t0 = time.perf_counter()
    from mboostrs._frame import read_column

    if not columns:
        return None

    y_full = np.asarray(read_column(data, response).values, dtype=np.float64)
    X_full, cat_mask = _encode_matrix(data, columns)
    finite = np.isfinite(y_full)
    if finite.sum() < 40:
        return None
    X_all = X_full[finite]
    y_all = y_full[finite]
    n_all, p = X_all.shape

    rng = np.random.default_rng(seed)
    if n_all > max_rows:
        idx = np.sort(rng.choice(n_all, size=max_rows, replace=False))
        X_sub, y_sub = X_all[idx], y_all[idx]
    else:
        X_sub, y_sub = X_all, y_all

    n = X_sub.shape[0]
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr, va = perm[:cut], perm[cut:]
    X_tr, y_tr = X_sub[tr], y_sub[tr]
    X_va, y_va = X_sub[va], y_sub[va]

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    except ImportError:
        return _numpy_fallback_radar(
            X_tr, y_tr, X_va, y_va, columns, cat_mask,
            is_binary=is_binary, protected=protected, t0=t0,
        )

    cat_arg = cat_mask if any(cat_mask) else None
    add_cst = [[j] for j in range(p)]

    if is_binary:
        from sklearn.linear_model import LogisticRegression

        def _logloss(yt, prob):
            p_clip = np.clip(prob, 1e-7, 1.0 - 1e-7)
            return float(-np.mean(yt * np.log(p_clip) + (1.0 - yt) * np.log(1.0 - p_clip)))

        p0 = float(np.clip(np.mean(y_tr), 1e-6, 1.0 - 1e-6))
        val_null = _logloss(y_va, np.full_like(y_va, p0))
        lin_tr_pred = np.full_like(y_tr, p0)
        lin_va_pred = np.full_like(y_va, p0)
        try:
            lr = LogisticRegression(max_iter=200)
            num_idx = [j for j, c in enumerate(cat_mask) if not c]
            if num_idx:
                mu = X_tr[:, num_idx].mean(axis=0)
                sd = np.where(X_tr[:, num_idx].std(axis=0) > 1e-12, X_tr[:, num_idx].std(axis=0), 1.0)
                lr.fit((X_tr[:, num_idx] - mu) / sd, y_tr.astype(int))
                lin_tr_pred = lr.predict_proba((X_tr[:, num_idx] - mu) / sd)[:, 1]
                lin_va_pred = lr.predict_proba((X_va[:, num_idx] - mu) / sd)[:, 1]
                val_lin = _logloss(y_va, lin_va_pred)
                tstats = np.zeros(p, dtype=np.float64)
                for li, cj in enumerate(num_idx):
                    tstats[cj] = abs(float(lr.coef_[0, li])) * 3.0
            else:
                val_lin = val_null
                tstats = np.zeros(p, dtype=np.float64)
        except Exception:
            val_lin = val_null
            tstats = np.zeros(p, dtype=np.float64)

        m_add = HistGradientBoostingClassifier(
            max_iter=120, max_depth=3, learning_rate=0.12,
            min_samples_leaf=20, l2_regularization=1.0,
            interaction_cst=add_cst, categorical_features=cat_arg,
            early_stopping=False, random_state=seed,
        )
        m_int = HistGradientBoostingClassifier(
            max_iter=80, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, l2_regularization=1.0,
            categorical_features=cat_arg, early_stopping=False, random_state=seed,
        )
        m_add.fit(X_tr, y_tr.astype(int))
        m_int.fit(X_tr, y_tr.astype(int))
        val_add = _logloss(y_va, m_add.predict_proba(X_va)[:, 1])
        val_int = _logloss(y_va, m_int.predict_proba(X_va)[:, 1])

        def _score_model(m, Xm):
            return _logloss(y_va, m.predict_proba(Xm)[:, 1])
    else:
        val_null = float(np.mean((y_va - float(np.mean(y_tr))) ** 2))
        val_lin, tstats, lin_va_pred = _fast_ridge(X_tr, y_tr, X_va, y_va, cat_mask)
        # Reconstruct training predictions from the same ridge fit for residual screen if needed.
        num_idx = [j for j, c in enumerate(cat_mask) if not c]
        if num_idx:
            Xn_tr = X_tr[:, num_idx]
            mu = Xn_tr.mean(axis=0)
            sd = np.where(Xn_tr.std(axis=0) > 1e-12, Xn_tr.std(axis=0), 1.0)
            Z_tr = (Xn_tr - mu) / sd
            yc = y_tr - float(np.mean(y_tr))
            try:
                beta_lin = np.linalg.solve(
                    Z_tr.T @ Z_tr + max(1.0, len(y_tr) / 100.0) * np.eye(len(num_idx)),
                    Z_tr.T @ yc,
                )
                lin_tr_pred = float(np.mean(y_tr)) + Z_tr @ beta_lin
            except np.linalg.LinAlgError:
                lin_tr_pred = np.full_like(y_tr, float(np.mean(y_tr)))
        else:
            lin_tr_pred = np.full_like(y_tr, float(np.mean(y_tr)))
        m_add = HistGradientBoostingRegressor(
            max_iter=120, max_depth=3, learning_rate=0.12,
            min_samples_leaf=20, l2_regularization=1.0,
            interaction_cst=add_cst, categorical_features=cat_arg,
            early_stopping=False, random_state=seed,
        )
        m_int = HistGradientBoostingRegressor(
            max_iter=80, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, l2_regularization=1.0,
            categorical_features=cat_arg, early_stopping=False, random_state=seed,
        )
        m_add.fit(X_tr, y_tr)
        m_int.fit(X_tr, y_tr)
        val_add = float(np.mean((y_va - m_add.predict(X_va)) ** 2))
        val_int = float(np.mean((y_va - m_int.predict(X_va)) ** 2))

        def _score_model(m, Xm):
            return float(np.mean((y_va - m.predict(Xm)) ** 2))

    # Fast 1-pass held-out permutation importance on X_va (n_va <= 400).
    signal_add = max(1e-12, val_null - val_add)
    signal_int = max(1e-12, val_null - val_int)
    best_signal = max(signal_add, signal_int)
    perm_add = np.zeros(p, dtype=np.float64)
    perm_int = np.zeros(p, dtype=np.float64)
    X_buf = X_va.copy()
    perm_rng = np.random.default_rng(seed + 17)
    for j in range(p):
        orig_col = X_buf[:, j].copy()
        X_buf[:, j] = perm_rng.permutation(orig_col)
        perm_add[j] = (_score_model(m_add, X_buf) - val_add) / max(1e-12, val_null)
        perm_int[j] = (_score_model(m_int, X_buf) - val_int) / max(1e-12, val_null)
        X_buf[:, j] = orig_col

    col_scores = {
        col.name: float(max(perm_add[j], perm_int[j]))
        for j, col in enumerate(columns)
    }

    # Variable weeding: only drop clear noise on wider tables (p > 8).
    kept: list[str] = []
    dropped: list[str] = []
    has_signal = (val_null - min(val_lin, val_add, val_int)) / max(1e-12, val_null) >= 0.015
    if p <= 8:
        kept = [c.name for c in columns]
    else:
        for j, col in enumerate(columns):
            if (
                col.name in protected
                or perm_int[j] >= 0.002
                or perm_add[j] >= 0.002
                or (tstats[j] >= 2.8 and best_signal > 0.01 * val_null)
            ):
                kept.append(col.name)
            else:
                dropped.append(col.name)
        # Safety floor so downstream mboostrs always has candidates to confirm.
        min_keep = min(p, 5 if has_signal else 4)
        if len(kept) < min_keep:
            ranked = sorted(
                range(p),
                key=lambda j: (max(perm_int[j], perm_add[j]), tstats[j]),
                reverse=True,
            )
            kept_set = set(kept)
            for j in ranked:
                if len(kept_set) >= min_keep:
                    break
                kept_set.add(columns[j].name)
            kept = [c.name for c in columns if c.name in kept_set]
            dropped = [c.name for c in columns if c.name not in kept_set]

    # Interaction radar: compare held-out depth-4 HGB against the BETTER of
    # linear ridge and strictly additive HGB (so linear-dominated tables like
    # abalone don't falsely trigger interactions because trees lag ridge).
    best_add = min(val_lin, val_add)
    prefers_linear = bool(val_lin <= val_add * 1.005 and val_lin <= val_int * 1.005)
    if is_binary:
        inter_gain = float((best_add - val_int) / max(1e-12, val_null))
        needs_inter = bool(val_int < best_add and (best_add - val_int) >= 0.006)
    else:
        r2_add = 1.0 - best_add / max(1e-12, val_null)
        r2_int = 1.0 - val_int / max(1e-12, val_null)
        inter_gain = float(r2_int - r2_add)
        rel_drop = float((best_add - val_int) / max(1e-12, best_add))
        needs_inter = bool(inter_gain >= 0.015 or (r2_add >= 0.5 and rel_drop >= 0.06))

    # Identify active interacting columns (those carrying the interactive model's signal).
    inter_cols: tuple[str, ...] = ()
    if needs_inter:
        ranked_int = sorted(
            [j for j in range(p) if columns[j].name in set(kept)],
            key=lambda j: perm_int[j],
            reverse=True,
        )
        pos = [columns[j].name for j in ranked_int if perm_int[j] >= 0.003]
        if len(pos) < 2:
            pos = [columns[j].name for j in ranked_int[: min(len(ranked_int), 8)]]
        inter_cols = tuple(pos[:16])

    # Column shapes (linear vs smooth) for kept numeric columns.
    shapes = _detect_column_shapes(m_add, X_tr, columns, cat_mask) if not is_binary else {
        c.name: ("brandom" if is_cat else ("bols" if c.kind == "boolean" else "bbs"))
        for c, is_cat in zip(columns, cat_mask, strict=True)
    }

    # Heteroscedasticity / spread radar (regression only, < 30 ms).
    # Uses profile Gaussian NLL on validation residuals so in-sample residual
    # shrinkage never biases the constant-vs-varying variance comparison.
    needs_spread = False
    needs_tree_scale = False
    spread_gain = 0.0
    spread_cols: tuple[str, ...] = ()
    if not is_binary:
        def _profile_nll(res: np.ndarray, log_v: np.ndarray) -> float:
            lv = np.clip(log_v, -12.0, 12.0)
            r2 = res * res
            c_hat = max(float(np.mean(r2 / np.exp(lv))), 1e-16)
            return float(0.5 * (1.0 + np.log(2.0 * np.pi) + np.log(c_hat) + float(np.mean(lv))))

        if needs_inter:
            tr_pred, va_pred = m_int.predict(X_tr), m_int.predict(X_va)
        elif val_lin < val_add:
            tr_pred, va_pred = lin_tr_pred, lin_va_pred
        else:
            tr_pred, va_pred = m_add.predict(X_tr), m_add.predict(X_va)
        res_tr = y_tr - tr_pred
        res_va = y_va - va_pred
        sd_tr = max(float(np.std(res_tr)), 1e-8)
        nll_const = _profile_nll(res_va, np.zeros_like(res_va))
        log_r2_tr = np.log(np.maximum(res_tr * res_tr, (1e-3 * sd_tr) ** 2))
        try:
            s_add = HistGradientBoostingRegressor(
                max_iter=45, max_depth=2, learning_rate=0.1,
                min_samples_leaf=20, l2_regularization=1.0,
                interaction_cst=add_cst, categorical_features=cat_arg,
                early_stopping=False, random_state=seed,
            )
            s_add.fit(X_tr, log_r2_tr)
            nll_add = _profile_nll(res_va, s_add.predict(X_va))

            s_tree = HistGradientBoostingRegressor(
                max_iter=55, max_depth=3, learning_rate=0.1,
                min_samples_leaf=20, l2_regularization=1.0,
                categorical_features=cat_arg, early_stopping=False, random_state=seed,
            )
            s_tree.fit(X_tr, log_r2_tr)
            nll_tree = _profile_nll(res_va, s_tree.predict(X_va))

            best_nll = min(nll_add, nll_tree)
            spread_gain = float(nll_const - best_nll)
            needs_spread = bool(spread_gain >= 0.012)
            needs_tree_scale = bool(needs_spread and (nll_add - nll_tree) >= 0.008)
            if needs_spread:
                s_win = s_tree if needs_tree_scale else s_add
                s_base = nll_tree if needs_tree_scale else nll_add
                X_sbuf = X_va.copy()
                sp_scores: list[tuple[float, str]] = []
                for j, col in enumerate(columns):
                    orig = X_sbuf[:, j].copy()
                    X_sbuf[:, j] = perm_rng.permutation(orig)
                    delta = _profile_nll(res_va, s_win.predict(X_sbuf)) - s_base
                    X_sbuf[:, j] = orig
                    if delta >= 0.005:
                        sp_scores.append((float(delta), col.name))
                sp_scores.sort(reverse=True)
                spread_cols = tuple(name for _, name in sp_scores[:10])
                # Never weed out a column that drives conditional variance.
                if spread_cols:
                    kept_set = set(kept) | set(spread_cols)
                    kept = [c.name for c in columns if c.name in kept_set]
                    dropped = [c.name for c in columns if c.name not in kept_set]
        except Exception:
            pass

    return FastRadarResult(
        kept_columns=tuple(kept),
        dropped_columns=tuple(dropped),
        column_shapes=shapes,
        column_scores=col_scores,
        val_loss_null=val_null,
        val_loss_linear=val_lin,
        val_loss_additive=val_add,
        val_loss_interactive=val_int,
        prefers_linear=prefers_linear,
        interaction_gain=inter_gain,
        needs_interactions=needs_inter,
        interaction_columns=inter_cols,
        needs_lss_spread=needs_spread,
        needs_lss_tree_scale=needs_tree_scale,
        spread_logscore_gain=spread_gain,
        spread_columns=spread_cols,
        backend="sklearn.HistGradientBoosting",
        seconds=float(time.perf_counter() - t0),
    )
