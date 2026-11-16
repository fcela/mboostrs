#!/usr/bin/env python
"""Regenerate the run A deliverables from the data directory.

Usage: python run_all.py [DATA_DIR]
Writes findings_<D>.json, predictions_<D>.csv, pairs_tests_<D>.csv and fit_log.csv
next to this script. Uses mboostrs only (plus numpy and polars for data handling).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import mboostrs as mb

SEED = 20261015
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/private/tmp/claude-501/-Users-fcela-src-public-github-mboostrs/"
    "21184ce7-2071-464b-b1fa-972cc27ee18c/scratchpad/study/data")
XS = [f"x{j}" for j in range(1, 21)]
LEVEL = 0.90
ALPHA = 0.05
N_PAIR_SCREEN = 3

# Per-dataset choices. messy has a categorical region column (a fixed effect here,
# 3 levels, not a grouping) and missing inputs, filled by the model with an
# indicator term per missing column (na_action="impute").
CONFIG = {
    "main": dict(extra=[], na="error", conformal="distributional"),
    "small": dict(extra=[], na="error", conformal="cv+"),
    "messy": dict(extra=["bols(region)"], na="impute", conformal="distributional"),
}

FIT_LOG = []


def timed(dataset, call, n_rows, fn):
    t0 = time.perf_counter()
    out = fn()
    FIT_LOG.append({"dataset": dataset, "library": "mboostrs", "call": call,
                    "n_rows": n_rows,
                    "seconds": round(time.perf_counter() - t0, 4)})
    return out


def learner_name(term):
    """'bbs(x1, df=4)' -> 'x1'; 'bols(region)' -> 'region'."""
    inner = term[term.index("(") + 1:]
    return inner.split(",")[0].split(")")[0].strip()


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def analyse(name):
    cfg = CONFIG[name]
    np.random.seed(SEED)
    train = pl.read_csv(DATA / f"{name}_train.csv", null_values=["", "NaN"])
    test = pl.read_csv(DATA / f"{name}_test_features.csv", null_values=["", "NaN"])
    n = train.height
    nkw = dict(na_action=cfg["na"])
    terms = [f"bbs({x}, df=4)" for x in XS] + cfg["extra"]
    formula = "y ~ " + " + ".join(terms)

    # Step 1: tune the stopping iteration by 5-fold CV, then fit at that value.
    tuned = timed(name, "tune_mstop(folds=5)", n, lambda: mb.tune_mstop(
        formula, train, folds=5, start=100, max_mstop=2000, seed=SEED, **nkw))
    mstop = int(tuned.mstop)
    print(f"[{name}] mstop={mstop} at_boundary={tuned.at_boundary}")
    model = timed(name, "gamboost", n, lambda: mb.gamboost(
        formula, train, control=mb.boost_control(mstop=mstop), **nkw))

    # Step 2: stability selection decides which inputs have a mean effect.
    stab = timed(name, "stabsel(q=5,PFER=1,B=50)", n, lambda: mb.stabsel(
        model, q=5, PFER=1, B=50, verbose=False))
    sel = set(stab.selected_names)
    sel_terms = [t for t in terms if learner_name(t) in sel]
    print(f"[{name}] stabsel selected: {sorted(sel)}")

    # Step 3: pairwise interactions, screened on the main-effects residual and
    # tested as 1-df product terms bols(a, by=b) added to the selected main effects.
    # p-values come from a REML refit; Holm adjustment across the tested pairs.
    resid = train["y"].to_numpy().astype(float) - np.asarray(
        model.predict(newdata=train, type="response"), dtype=float)
    scr = timed(name, "screen_interactions", n, lambda: mb.screen_interactions(
        train.select(XS), resid, na_action=cfg["na"]))
    scr_df = scr.to_frame() if hasattr(scr, "to_frame") else pl.DataFrame(scr)
    print(f"[{name}] screen head:\n{scr_df.head(6)}")
    cand = []
    for row in scr_df.iter_rows(named=True):
        pair = tuple(sorted([row["left"], row["right"]]))
        if pair != ("x6", "x7") and pair not in cand:
            cand.append(pair)
        if len(cand) == N_PAIR_SCREEN:
            break
    tested = [("x6", "x7")] + cand
    pvals, rows = {}, []
    for a, b in tested:
        ptr = f"bols({a}, by={b})"
        margins = [t for t in (f"bbs({a}, df=4)", f"bbs({b}, df=4)") if t not in sel_terms]
        fit2 = timed(name, f"gamboost(main + {ptr})", n, lambda: mb.gamboost(
            formula + " + " + ptr, train, control=mb.boost_control(mstop=mstop), **nkw))
        rf2 = timed(name, f"gam_refit(pair {a}:{b})", n, lambda: mb.gam_refit(
            fit2, terms=sel_terms + margins + [ptr]))
        keys = [k for k in rf2.tests if ptr in k]
        pvals[f"{a}:{b}"] = float(rf2.tests[keys[0]]["p_value"])
    adj = holm(pvals)
    pairs = {k: {"selected": bool(adj[k] < ALPHA)} for k in pvals}
    sel_pairs = [tuple(k.split(":")) for k in pvals if pairs[k]["selected"]]

    # Sensitivity: the x6:x7 surface as a tensor instead of a product term.
    tens = "bbs(x6) %X% bbs(x7)"
    fit_t = timed(name, "gamboost(main + tensor x6,x7)", n, lambda: mb.gamboost(
        formula + " + " + tens, train, control=mb.boost_control(mstop=mstop), **nkw))
    rf_t = timed(name, "gam_refit(tensor x6,x7)", n, lambda: mb.gam_refit(
        fit_t, terms=sel_terms + [t for t in ("bbs(x6, df=4)", "bbs(x7, df=4)") if t not in sel_terms] + [tens]))
    t_key = [k for k in rf_t.tests if "%X%" in k][0]
    tensor_p = float(rf_t.tests[t_key]["p_value"])
    pl.DataFrame([{"pair": k, "test": "product bols(a, by=b), refit Wald", "p_value": pvals[k],
                   "p_holm": adj[k], "selected": pairs[k]["selected"]} for k in pvals]
                 + [{"pair": "x6:x7", "test": "tensor bbs(x6) %X% bbs(x7), sensitivity",
                     "p_value": tensor_p, "p_holm": None, "selected": None}]
                 ).write_csv(HERE / f"pairs_tests_{name}.csv")
    print(f"[{name}] pairs {pairs}; p {pvals}; tensor p {tensor_p:.3g}")

    # Step 4: final mean model = main terms + selected pair terms. Re-tune the
    # stopping iteration on this formula, re-run stability selection, then refit.
    ptrs = [f"bols({a}, by={b})" for a, b in sel_pairs]
    margins = [t for a, b in sel_pairs for t in (f"bbs({a}, df=4)", f"bbs({b}, df=4)")]
    final_formula = formula + "".join(f" + {p}" for p in ptrs)
    if ptrs:
        tuned_f = timed(name, "tune_mstop(final, folds=5)", n, lambda: mb.tune_mstop(
            final_formula, train, folds=5, start=100, max_mstop=8000, seed=SEED, **nkw))
        mstop_f = int(tuned_f.mstop)
        print(f"[{name}] final mstop={mstop_f} at_boundary={tuned_f.at_boundary}")
    else:
        mstop_f = mstop
    final = timed(name, "gamboost(final)", n, lambda: mb.gamboost(
        final_formula, train, control=mb.boost_control(mstop=mstop_f), **nkw))
    stab_f = timed(name, "stabsel(final,q=5,PFER=1,B=50)", n, lambda: mb.stabsel(
        final, q=5, PFER=1, B=50, verbose=False))
    sel_f = set(stab_f.selected_names)
    main_sel_f = [t for t in terms if learner_name(t) in sel_f]
    eff_terms = main_sel_f + [t for t in margins if t not in main_sel_f] + ptrs
    eff_inputs = {learner_name(t) for t in main_sel_f} | {v for p in sel_pairs for v in p}
    print(f"[{name}] final stabsel: {sorted(sel_f)}; pairs in mean model: {sel_pairs}")
    rf = timed(name, "gam_refit(final selected terms)", n, lambda: mb.gam_refit(
        final, terms=eff_terms))

    # Effect of each input: average change in the refit's prediction when x moves
    # from its training 25th to 75th percentile, other inputs at their own values.
    effects = {}
    for x in XS:
        if x not in eff_inputs:
            effects[x] = {"selected": False, "estimate": 0.0,
                          "lower90": 0.0, "upper90": 0.0}
            continue
        v = train[x].to_numpy().astype(float)
        v = v[~np.isnan(v)]
        q25, q75 = np.quantile(v, [0.25, 0.75])
        c = rf.contrast(x, float(q25), float(q75), data=train, level=LEVEL)
        print(f"[{name}] {x}: q25={q25:.3f} q75={q75:.3f} estimate={c.estimate:.4f} "
              f"[{c.lower:.4f}, {c.upper:.4f}]")
        effects[x] = {"selected": True, "estimate": float(c.estimate),
                      "lower90": float(c.lower), "upper90": float(c.upper)}

    # Step 5: which inputs change the noise level (residual spread vs fitted value,
    # per input). Thresholds from the package's interpretation guide: |t| > 3.5 on
    # the input and a spread ratio above 1.3 in either direction.
    # Step 5: which inputs change the noise level. heteroscedasticity regresses
    # log(residual^2) on each input; its t and spread ratio are read against the
    # package's thresholds: |t| > 3.5 on the input and a spread ratio above 1.3 in
    # either direction. With missing inputs the test refuses rows with any gap in
    # the fitted columns, so for each input the rows where that input is observed are
    # used, with the other inputs median-filled (the model's own imputation fills
    # them with the median too, plus an indicator this check does not carry).
    def median_filled(frame):
        return frame.with_columns([
            pl.col(c).fill_nan(None).fill_null(pl.col(c).fill_nan(None).median())
            for c in XS])

    het_rows = {}
    if cfg["na"] == "error":
        het = timed(name, "heteroscedasticity", n, lambda: mb.heteroscedasticity(final, train))
        het_rows = het["by_column"]
    else:
        # The test reads the model's own fitted rows, so each input gets a mean model
        # refitted (same formula, same stopping iteration) on the rows where it is
        # observed, with the other inputs median-filled.
        for x in XS:
            sub = median_filled(train.filter(pl.col(x).is_not_null() & pl.col(x).is_not_nan()))
            mk = timed(name, f"gamboost(final, {x} observed rows, others median-filled)", sub.height,
                       lambda: mb.gamboost(final_formula, sub,
                                           control=mb.boost_control(mstop=mstop_f)))
            h = timed(name, f"heteroscedasticity({x})", sub.height,
                      lambda: mb.heteroscedasticity(mk, sub, columns=[x]))
            if x in h["by_column"]:
                het_rows[x] = h["by_column"][x]
    drivers = []
    for col in XS:
        if col not in het_rows:
            print(f"[{name}] {col}: not tested")
            continue
        r = het_rows[col]
        print(f"[{name}] het {col}: t={r['t']:.2f} spread_ratio={r['spread_ratio']:.3f}")
        if abs(r["t"]) > 3.5 and (r["spread_ratio"] > 1.3 or r["spread_ratio"] < 1 / 1.3):
            drivers.append(col)
    print(f"[{name}] variance drivers: {drivers}")

    # Step 6: 90% predictive intervals for new rows. A Gaussian location-scale model
    # carries the noise driver in its scale: mu = final mean formula, sigma = bbs
    # of each driver. The median is its mu prediction; the interval is the fitted
    # normal band widened by a split-conformal quantile (method="distributional"),
    # which holds its coverage on average over training-like rows even if the
    # Gaussian family is wrong. Outlying responses in messy widen this band.
    if not drivers:
        raise RuntimeError(f"{name}: no variance driver; a scale model is required")
    if cfg["conformal"] == "distributional":
        sig_formula = "y ~ " + " + ".join(f"bbs({x}, df=4)" for x in drivers)
        lss_p = timed(name, "gamboostLSS(mu=final, sigma=drivers)", n, lambda: mb.gamboostLSS(
            {"mu": final_formula, "sigma": sig_formula}, data=train, families=mb.GaussianLSS(),
            control=mb.boost_control(mstop=mstop_f), na_action=cfg["na"]))
        ci = timed(name, "conformal_interval(distributional)", n, lambda: mb.conformal_interval(
            lss_p, newdata=test, level=LEVEL, method="distributional", seed=SEED))
        median = np.asarray(lss_p.predict(newdata=test, parameter="mu"), dtype=float)
    else:
        # Small data (n=200): split-conformal on the mean model with cross-fitted
        # residuals (cv+). Its width varies by row through the fold models; the scale
        # model is too unstable at this n (its sigma spline extrapolates to widths above 900).
        ci = timed(name, "conformal_interval(cv+)", n, lambda: mb.conformal_interval(
            final, newdata=test, level=LEVEL, method="cv+", seed=SEED))
        median = np.asarray(final.predict(newdata=test), dtype=float)
    pred = pl.DataFrame({"row_id": test["row_id"].to_numpy(),
                         "lower90": np.asarray(ci.lower, dtype=float),
                         "median": median,
                         "upper90": np.asarray(ci.upper, dtype=float)})
    pred.write_csv(HERE / f"predictions_{name}.csv")

    findings = {"dataset": name, "effects": effects, "pairs": pairs,
                "variance_drivers": drivers}
    (HERE / f"findings_{name}.json").write_text(json.dumps(findings, indent=2))


def main():
    for name in ["main", "small", "messy"]:
        analyse(name)
    pl.DataFrame(FIT_LOG).write_csv(HERE / "fit_log.csv")


if __name__ == "__main__":
    main()
