#!/usr/bin/env python
"""Regenerate every run B deliverable from the data directory.

For each dataset (main, small, messy):
  * additive penalized spline mean model (P-spline, Huber IRLS), smoothing
    parameter chosen by GCV;
  * mean effects of x1..x20 by Freedman-Lane score tests (Holm-adjusted);
  * pairwise interactions for all 190 pairs of x1..x20 by the same tests;
  * final mean model = all main effects + selected pairs;
  * variance drivers from a penalized spline model of log(squared residual);
  * 10-fold out-of-fold residuals give standardized residual quantiles, which
    turn the mean and scale predictions into 90% predictive intervals;
  * effect size = average over training rows of the change in predicted y
    when one input moves from its 25th to its 75th training percentile,
    with a 90% percentile bootstrap interval (300 row resamples).

Libraries: numpy, scipy (not used for fitting), pandas, scikit-learn
(SplineTransformer, IterativeImputer). No mboostrs.

Usage: python run_all.py   (writes into the directory containing this file)
"""
import os
import time
from itertools import combinations
import json

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from scipy import stats
from sklearn.preprocessing import SplineTransformer

SEED = 20261015
N_PERM = 499
N_BOOT = 300
RUN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(RUN_DIR), "data")
XC = [f"x{i}" for i in range(1, 21)]
MAIN_DF = 11          # spline coefficients per main effect (12 bases, last dropped)
PAIR_DF = 5           # free spline coefficients per pair margin (6 bases, sum-to-zero)
LAM_GRID = np.logspace(-2, 5, 15)
DATASETS = ["main", "small", "messy"]


def _sum_zero_basis(k):
    """k x (k-1) matrix whose columns span the coefficient vectors with zero sum.
    Each B-spline basis sums to 1 at every x, so a coefficient vector with zero
    sum removes the constant function from the block; the intercept then
    carries the level alone."""
    _, _, vt = np.linalg.svd(np.ones((1, k)))
    return vt[1:].T


ZC_MAIN = _sum_zero_basis(12)      # 12 bases -> 11 free coefficients
ZC_PAIR = _sum_zero_basis(6)       # 6 bases  -> 5 free coefficients
D12 = np.diff(np.eye(12), n=2, axis=0)
PM_MAIN = ZC_MAIN.T @ D12.T @ D12 @ ZC_MAIN   # second-difference penalty, reduced space


class Basis:
    """Spline bases fixed on the training inputs."""

    def __init__(self, tr):
        self.main = {c: SplineTransformer(n_knots=10, degree=3, extrapolation="constant")
                     .fit(tr[[c]].to_numpy()) for c in XC}
        self.pair = {c: SplineTransformer(n_knots=4, degree=3, extrapolation="constant")
                     .fit(tr[[c]].to_numpy()) for c in XC}
        self.regions = sorted(tr["region"].unique()) if "region" in tr else []

    def main_block(self, df, c):
        return self.main[c].transform(df[[c]].to_numpy()) @ ZC_MAIN

    def pair_block(self, df, a, b):
        Ba = self.pair[a].transform(df[[a]].to_numpy()) @ ZC_PAIR
        Bb = self.pair[b].transform(df[[b]].to_numpy()) @ ZC_PAIR
        return (Ba[:, :, None] * Bb[:, None, :]).reshape(len(df), -1)


def design(basis, df, pairs):
    parts = [np.ones((len(df), 1))]
    parts += [basis.main_block(df, c) for c in XC]
    if basis.regions:
        parts.append(np.column_stack(
            [(df["region"].to_numpy() == lv).astype(float) for lv in basis.regions[1:]]))
    parts += [basis.pair_block(df, a, b) for a, b in pairs]
    return np.hstack(parts)


def penalty(nreg, npairs, lam, lam_pair):
    """Block penalty: second-difference penalty on each main effect,
    ridge on region dummies and pair blocks, tiny ridge elsewhere."""
    p = 1 + MAIN_DF * len(XC) + nreg + PAIR_DF ** 2 * npairs
    P = np.zeros((p, p))
    for k in range(len(XC)):
        s = 1 + MAIN_DF * k
        P[s:s + MAIN_DF, s:s + MAIN_DF] = lam * PM_MAIN
    s = 1 + MAIN_DF * len(XC)
    P[s:s + nreg, s:s + nreg] = 1e-6 * np.eye(nreg)
    s += nreg
    for _ in range(npairs):
        m = PAIR_DF ** 2
        P[s:s + m, s:s + m] = lam_pair * np.eye(m)
        s += m
    return P + 1e-8 * np.eye(p)


def huber(X, y, P, beta0=None, maxit=50):
    """Penalized Huber M-estimate by IRLS (tuning constant 1.345, MAD scale)."""
    n = len(y)
    if beta0 is None:
        w = np.ones(n)
        beta = np.linalg.solve(X.T @ (X * w[:, None]) + P, X.T @ (w * y))
    else:
        beta = beta0
    for _ in range(maxit):
        r = y - X @ beta
        s = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1e-12)
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-300))
        bn = np.linalg.solve(X.T @ (X * w[:, None]) + P, X.T @ (w * y))
        conv = np.max(np.abs(bn - beta)) <= 1e-8 * (1 + np.max(np.abs(beta)))
        beta = bn
        if conv:
            break
    r = y - X @ beta
    s = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1e-12)
    w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-300))
    return beta, w


def gcv_lambda(X, y, w, nreg, npairs):
    n = len(y)
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (w * y)
    best = None
    for lam in LAM_GRID:
        P = penalty(nreg, npairs, lam, lam)
        A = XtWX + P
        tr_h = np.trace(np.linalg.solve(A, XtWX))
        beta = np.linalg.solve(A, XtWy)
        rss = np.sum(w * (y - X @ beta) ** 2)
        g = n * rss / (n - tr_h) ** 2
        if best is None or g < best[0]:
            best = (g, lam)
    return best[1]


def score_test(Xr, Pr, Z, w, resid, perms, Ainv=None):
    """Score test that the columns Z add nothing to the reduced model (Xr, Pr).
    The null distribution comes from permuting the reduced-model residuals."""
    if Ainv is None:
        Ainv = np.linalg.inv(Xr.T @ (Xr * w[:, None]) + Pr)
    G = Ainv @ (Xr.T @ (Z * w[:, None]))
    Zr = Z - Xr @ G
    Zw = Zr * w[:, None]
    V = Zr.T @ Zw
    Vi = np.linalg.pinv(V, rcond=1e-10)
    s_obs = Zw.T @ resid
    T_obs = s_obs @ Vi @ s_obs
    S = Zw.T @ resid[perms].T
    Tb = np.einsum("kb,kl,lb->b", S, Vi, S)
    p_perm = (1 + np.sum(Tb >= T_obs)) / (len(perms) + 1)
    # Gamma approximation matched to the permutation mean and variance. The
    # permutation p-value cannot fall below 1/(B+1), so it is used only as a check.
    m, v = Tb.mean(), Tb.var(ddof=1)
    p_gamma = stats.gamma.sf(T_obs, a=m * m / v, scale=v / m)
    return p_gamma, T_obs, p_perm


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def add_log(rows, dataset, library, call, n_rows, t0):
    rows.append({"dataset": dataset, "library": library, "call": call,
                 "n_rows": int(n_rows), "seconds": round(time.perf_counter() - t0, 4)})


def impute(tr, te):
    imp = IterativeImputer(max_iter=25, random_state=SEED)
    tr = tr.copy()
    te = te.copy()
    tr[XC] = imp.fit_transform(tr[XC])
    te[XC] = imp.transform(te[XC])
    return tr, te


def analyse(name, rows):
    tr = pd.read_csv(os.path.join(DATA_DIR, f"{name}_train.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, f"{name}_test_features.csv"))
    y = tr["y"].to_numpy(float)
    n = len(y)
    diag = {}

    miss = tr[XC].isna()
    if miss.to_numpy().any():
        anym = miss.any(axis=1).to_numpy()
        diag["train_missing_cell_rate"] = round(float(miss.to_numpy().mean()), 4)
        diag["train_rows_with_any_missing"] = int(anym.sum())
        diag["y_mean_rows_with_missing"] = round(float(y[anym].mean()), 3)
        diag["y_mean_complete_rows"] = round(float(y[~anym].mean()), 3)
        t0 = time.perf_counter()
        tr, te = impute(tr, te)
        add_log(rows, name, "sklearn IterativeImputer", "fit_transform train x1-x20, transform test", n, t0)

    basis = Basis(tr)
    nreg = len(basis.regions) - 1 if basis.regions else 0
    ones = np.ones(n)
    perm_rng = np.random.default_rng(SEED)
    perms = np.array([perm_rng.permutation(n) for _ in range(N_PERM)])

    # Additive mean model: GCV for lambda, Huber weights, GCV again.
    t0 = time.perf_counter()
    X0 = design(basis, tr, [])
    p0 = X0.shape[1]
    lam = gcv_lambda(X0, y, ones, nreg, 0)
    _, w0 = huber(X0, y, penalty(nreg, 0, lam, lam))
    lam = gcv_lambda(X0, y, w0, nreg, 0)
    P0 = penalty(nreg, 0, lam, lam)
    beta0, w0 = huber(X0, y, P0)
    r0 = y - X0 @ beta0
    diag["lambda_mean"] = float(lam)
    add_log(rows, name, "numpy penalized spline + Huber IRLS",
            "additive model: GCV lambda (2 passes) + Huber fit", n, t0)

    # Mean effects of x1..x20: block score tests in the additive model.
    t0 = time.perf_counter()
    main_p = {}
    for k, c in enumerate(XC):
        blk = slice(1 + MAIN_DF * k, 1 + MAIN_DF * (k + 1))
        keep = np.array([i for i in range(p0) if not (blk.start <= i < blk.stop)])
        Xr, Pr = X0[:, keep], P0[np.ix_(keep, keep)]
        br, wr = huber(Xr, y, Pr)
        main_p[c], _, _ = score_test(Xr, Pr, X0[:, blk], wr, y - Xr @ br, perms)
    reg_mean_p = None
    if nreg:
        blk = slice(1 + MAIN_DF * len(XC), 1 + MAIN_DF * len(XC) + nreg)
        keep = np.array([i for i in range(p0) if not (blk.start <= i < blk.stop)])
        Xr, Pr = X0[:, keep], P0[np.ix_(keep, keep)]
        br, wr = huber(Xr, y, Pr)
        reg_mean_p, _, _ = score_test(Xr, Pr, X0[:, blk], wr, y - Xr @ br, perms)
        diag["region_mean_p"] = round(float(reg_mean_p), 4)
    main_adj = holm([main_p[c] for c in XC])
    add_log(rows, name, "numpy score test (Freedman-Lane, 499 permutations)",
            "21 reduced Huber fits for main effects", n, t0)

    # Pairwise interactions: all 190 pairs, tested in the additive model.
    t0 = time.perf_counter()
    Ainv0 = np.linalg.inv(X0.T @ (X0 * w0[:, None]) + P0)
    pair_p = {}
    pair_chi = {}
    for a, b in combinations(XC, 2):
        Z = basis.pair_block(tr, a, b)
        pair_p[f"{a}:{b}"], _, pair_chi[f"{a}:{b}"] = score_test(X0, P0, Z, w0, r0, perms, Ainv=Ainv0)
    pair_names = list(pair_p)
    pair_adj = holm([pair_p[k] for k in pair_names])
    diag["pairs_gamma_p_below_005_raw"] = int(sum(v < 0.05 for v in pair_p.values()))
    pairs_sel = [tuple(k.split(":")) for k, a in zip(pair_names, pair_adj) if a < 0.05]
    add_log(rows, name, "numpy score test (Freedman-Lane, 499 permutations)",
            "190 pair tensor-block tests", n, t0)

    # Final mean model: all main effects plus selected pairs.
    t0 = time.perf_counter()
    Xf = design(basis, tr, pairs_sel)
    Pf = penalty(nreg, len(pairs_sel), lam, lam)
    bf, wf = huber(Xf, y, Pf)
    rf = y - Xf @ bf
    add_log(rows, name, "numpy penalized spline + Huber IRLS",
            "final mean model with selected pairs", n, t0)

    # Scale model: penalized spline on log squared residual of the final mean model.
    t0 = time.perf_counter()
    t_log = np.log(rf ** 2 + 1e-12)
    lam_s = gcv_lambda(X0, t_log, ones, nreg, 0)
    Ps = penalty(nreg, 0, lam_s, lam_s)
    bs = np.linalg.solve(X0.T @ X0 + Ps, X0.T @ t_log)
    diag["lambda_scale"] = float(lam_s)
    var_p = {}
    for k, c in enumerate(XC):
        blk = slice(1 + MAIN_DF * k, 1 + MAIN_DF * (k + 1))
        keep = np.array([i for i in range(p0) if not (blk.start <= i < blk.stop)])
        Xr, Pr = X0[:, keep], Ps[np.ix_(keep, keep)]
        br = np.linalg.solve(Xr.T @ Xr + Pr, Xr.T @ t_log)
        var_p[c], _, _ = score_test(Xr, Pr, X0[:, blk], ones, t_log - Xr @ br, perms)
    reg_var_p = None
    if nreg:
        blk = slice(1 + MAIN_DF * len(XC), 1 + MAIN_DF * len(XC) + nreg)
        keep = np.array([i for i in range(p0) if not (blk.start <= i < blk.stop)])
        Xr, Pr = X0[:, keep], Ps[np.ix_(keep, keep)]
        br = np.linalg.solve(Xr.T @ Xr + Pr, Xr.T @ t_log)
        reg_var_p, _, _ = score_test(Xr, Pr, X0[:, blk], ones, t_log - Xr @ br, perms)
        diag["region_variance_p"] = round(float(reg_var_p), 4)
    var_names = XC + (["region"] if nreg else [])
    var_pvals = [var_p[c] for c in XC] + ([reg_var_p] if nreg else [])
    var_adj = holm(var_pvals)
    variance_drivers = [c for c, a in zip(var_names, var_adj) if a < 0.05]
    add_log(rows, name, "numpy penalized spline (log squared residual) + score tests",
            "scale model GCV + 21 variance tests", n, t0)

    # Standardized out-of-fold residuals (10 folds, lambda fixed at full-data values).
    fold_rng = np.random.default_rng(SEED + 2)
    fold = fold_rng.permutation(n) % 10
    oof = np.empty(n)
    oof_add = np.empty(n)
    for k in range(10):
        t0 = time.perf_counter()
        tri = fold != k
        tei = ~tri
        bk, _ = huber(Xf[tri], y[tri], Pf)
        oof[tei] = Xf[tei] @ bk
        ba, _ = huber(X0[tri], y[tri], P0)
        oof_add[tei] = X0[tei] @ ba
        add_log(rows, name, "numpy penalized spline + Huber IRLS",
                f"10-fold OOF fold {k + 1}: final and additive fits", int(tri.sum()), t0)
    e = y - oof
    sig_tr = np.exp(0.5 * (X0 @ bs))
    z = e / sig_tr
    q_lo, q_hi = np.quantile(z, [0.05, 0.95])
    diag["oof_rmse_final"] = round(float(np.sqrt(np.mean(e ** 2))), 4)
    diag["oof_rmse_additive"] = round(float(np.sqrt(np.mean((y - oof_add) ** 2))), 4)
    diag["oof_rmse_constant"] = round(float(np.std(y)), 4)
    diag["oof_coverage_90_check"] = round(float(np.mean(
        (y >= oof + sig_tr * q_lo) & (y <= oof + sig_tr * q_hi))), 4)
    diag["z_quantiles_05_95"] = [round(float(q_lo), 4), round(float(q_hi), 4)]
    diag["sigma_min_max_train"] = [round(float(sig_tr.min()), 4), round(float(sig_tr.max()), 4)]

    # Predictions for the test rows.
    Xt = design(basis, te, pairs_sel)
    mu_t = Xt @ bf
    sig_t = np.exp(0.5 * (design(basis, te, []) @ bs))
    pred = pd.DataFrame({"row_id": te["row_id"].to_numpy(),
                         "lower90": mu_t + sig_t * q_lo,
                         "median": mu_t,
                         "upper90": mu_t + sig_t * q_hi})
    pred.to_csv(os.path.join(RUN_DIR, f"predictions_{name}.csv"), index=False, float_format="%.6f")

    # Effects: average change in predicted y, 25th to 75th percentile, over training rows.
    t0 = time.perf_counter()
    M = np.zeros((len(XC), Xf.shape[1]))
    qs = {}
    for j, c in enumerate(XC):
        q25, q75 = np.quantile(tr[c].to_numpy(float), [0.25, 0.75])
        qs[c] = (float(q25), float(q75))
        d_lo = tr.copy()
        d_lo[c] = q25
        d_hi = tr.copy()
        d_hi[c] = q75
        M[j] = (design(basis, d_hi, pairs_sel) - design(basis, d_lo, pairs_sel)).mean(axis=0)
    est = M @ bf
    boot_rng = np.random.default_rng(SEED + 1)
    eff_b = np.empty((N_BOOT, len(XC)))
    for b in range(N_BOOT):
        idx = boot_rng.integers(0, n, n)
        bb, _ = huber(Xf[idx], y[idx], Pf, beta0=bf, maxit=8)
        eff_b[b] = M @ bb
    lo = np.quantile(eff_b, 0.05, axis=0)
    hi = np.quantile(eff_b, 0.95, axis=0)
    add_log(rows, name, "numpy bootstrap (row resampling, warm-start Huber)",
            f"{N_BOOT} bootstrap refits for effect intervals", n, t0)

    effects = {}
    for j, c in enumerate(XC):
        effects[c] = {"selected": bool(main_adj[j] < 0.05),
                      "estimate": round(float(est[j]), 6),
                      "lower90": round(float(lo[j]), 6),
                      "upper90": round(float(hi[j]), 6),
                      "p_holm": round(float(main_adj[j]), 6),
                      "q25_q75": [round(qs[c][0], 4), round(qs[c][1], 4)]}
    perm_adj = holm([pair_chi[k] for k in pair_names])
    pairs = {k: {"selected": bool(a < 0.05), "p_holm": float(f"{a:.3g}"),
                 "p_gamma": float(f"{pair_p[k]:.3g}"),
                 "p_perm_holm": round(float(pa), 6)}
             for k, a, pa in zip(pair_names, pair_adj, perm_adj)}

    findings = {"dataset": name,
                "effects": effects,
                "pairs": pairs,
                "variance_drivers": variance_drivers,
                "variance_p_holm": {c: round(float(a), 6) for c, a in zip(var_names, var_adj)},
                "diagnostics": {**diag, "pairs_selected": [f"{a}:{b}" for a, b in pairs_sel],
                                "n_train": n, "n_test": len(te),
                                "n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED}}
    with open(os.path.join(RUN_DIR, f"findings_{name}.json"), "w") as fh:
        json.dump(findings, fh, indent=2)

    print(f"== {name}: n={n}, test={len(te)}")
    print("  diagnostics:", json.dumps(diag))
    print("  pairs selected:", findings["diagnostics"]["pairs_selected"])
    print("  top pairs:", sorted(((v["p_holm"], k) for k, v in pairs.items()))[:4])
    print("  variance drivers:", variance_drivers)
    for c in XC:
        e_ = effects[c]
        print(f"  {c:4s} sel={str(e_['selected']):5s} est={e_['estimate']:+.4f} "
              f"[{e_['lower90']:+.4f}, {e_['upper90']:+.4f}] p_holm={e_['p_holm']:.4f}")


def main():
    rows = []
    for name in DATASETS:
        analyse(name, rows)
    pd.DataFrame(rows, columns=["dataset", "library", "call", "n_rows", "seconds"]).to_csv(
        os.path.join(RUN_DIR, "fit_log.csv"), index=False)


if __name__ == "__main__":
    main()
