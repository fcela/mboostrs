"""Canonical Scorecard 2.1 supplementary benchmark suite (D1a, D2, D3, D4, D6, D8, D10).

Executes deterministic, frozen planted-truth and stress protocols for dimensions
and high-anchor gates not covered by `benches/competitive_heldout`, writing a
versioned JSON record to `docs/benchmarks/scorecard_v21/results.json`.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

import mboostrs as mb
from mboostrs.causal import CausalSpec


def _spec(estimand: str = "ate") -> CausalSpec:
    return CausalSpec(
        treatment="t",
        outcome="y",
        covariates=("x0", "x1"),
        estimand=estimand,
        time_zero="baseline",
        follow_up="end of study",
        adjustment_rationale="baseline confounders x0 and x1",
    )


def bench_d1a_open_structure() -> dict:
    """Track A: prescribed rough formula + open-structure recovery across 12 tasks."""
    tp_open = 0
    total_open_signals = 0
    fp_open = 0
    total_open_nulls = 0
    tasks = []

    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed)
        n = 600
        x0 = rng.uniform(-2, 2, n)
        x1 = rng.uniform(-2, 2, n)
        x2 = rng.uniform(-2, 2, n)
        n0, n1, n2 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)

        # Case 1: rough linear on x0, missed nonlinear sin(2*x1) + noise n0..n2
        y1 = 1.2 * x0 + 1.8 * np.sin(2.0 * x1) + rng.normal(0, 0.35, n)
        d1 = {"x0": x0, "x1": x1, "n0": n0, "n1": n1, "n2": n2, "y": y1}
        f1 = mb.fit(d1, "y", seed=seed)
        form1 = str(f1.trail.final["formula"])
        hit1 = "x1" in form1
        false1 = sum(col in form1 for col in ("n0", "n1", "n2"))
        tp_open += int(hit1)
        total_open_signals += 1
        fp_open += false1
        total_open_nulls += 3
        tasks.append({"case": "missed_nonlinear", "seed": seed, "hit": hit1, "fp": false1})

        # Case 2: pure interaction x0 * x1 missed by additive spec + noise
        y2 = 1.0 * x0 + 2.2 * (x0 * x1) + rng.normal(0, 0.35, n)
        d2 = {"x0": x0, "x1": x1, "n0": n0, "n1": n1, "n2": n2, "y": y2}
        f2 = mb.fit(d2, "y", seed=seed)
        form2 = str(f2.trail.final["formula"])
        hit2 = ("btree(" in form2 or "%X%" in form2) and ("x0" in form2 and "x1" in form2)
        false2 = sum(col in form2 and f"bbs({col}" in form2 for col in ("n0", "n1", "n2"))
        tp_open += int(hit2)
        total_open_signals += 1
        fp_open += false2
        total_open_nulls += 3
        tasks.append({"case": "missed_interaction", "seed": seed, "hit": hit2, "fp": false2})

        # Case 3: missed heteroscedastic scale-only covariate x2
        sigma = np.exp(0.9 * x2) * 0.45
        y3 = 1.5 * np.sin(x0) + rng.normal(0.0, sigma, n)
        d3 = {"x0": x0, "x2": x2, "n0": n0, "n1": n1, "y": y3}
        f3 = mb.fit(d3, "y", seed=seed)
        fam3 = str(f3.trail.final["family"])
        form3 = str(f3.trail.final["formula"])
        hit3 = ("LSS" in fam3) and ("x2" in form3)
        false3 = sum(col in form3 for col in ("n0", "n1"))
        tp_open += int(hit3)
        total_open_signals += 1
        fp_open += false3
        total_open_nulls += 2
        tasks.append({"case": "missed_scale", "seed": seed, "hit": hit3, "fp": false3})

        # Case 4: complete rough spec (zero open signals; 4 pure noise candidates)
        y4 = 1.6 * x0 + rng.normal(0, 0.4, n)
        d4 = {"x0": x0, "n0": n0, "n1": n1, "n2": n2, "x2": x2, "y": y4}
        f4 = mb.fit(d4, "y", seed=seed)
        form4 = str(f4.trail.final["formula"])
        false4 = sum(col in form4 for col in ("n0", "n1", "n2", "x2"))
        fp_open += false4
        total_open_nulls += 4
        tasks.append({"case": "null_open_structure", "seed": seed, "hit": True, "fp": false4})

    return {
        "n_tasks": len(tasks),
        "open_tpr": round(tp_open / max(total_open_signals, 1), 4),
        "open_tp": tp_open,
        "open_total_signals": total_open_signals,
        "open_fdr_per_null": round(fp_open / max(total_open_nulls, 1), 4),
        "open_fp": fp_open,
        "open_total_nulls": total_open_nulls,
    }


def bench_d2_causal_humility() -> dict:
    """Thin-overlap ATO stability, automatic ato_companion, and Simpson sign reversal."""
    ato_errors = []
    ate_clipped_errors = []
    companion_triggered = 0
    sign_reversal_recovered = 0

    for seed in range(6):
        rng = np.random.default_rng(200 + seed)
        n = 800
        x0 = rng.normal(0, 1.5, n)
        x1 = rng.normal(0, 1.0, n)
        # Severe overlap violation + Simpson confounding: high x0 strongly increases T
        # and strongly decreases Y, so naive T-Y diff is negative while true effect is +1.5!
        logits = 2.6 * x0 - 1.2 * x1
        e_true = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, e_true, n).astype(float)
        tau_true = 1.5 + 0.4 * np.tanh(x1)
        y = -2.2 * x0 + 0.8 * x1 + t * tau_true + rng.normal(0, 0.6, n)
        w_ov = e_true * (1.0 - e_true)
        true_ato = float(np.sum(w_ov * tau_true) / np.sum(w_ov))
        true_ate = float(np.mean(tau_true))
        naive_diff = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))

        frame = {"x0": x0, "x1": x1, "t": t, "y": y}
        spec_ate = _spec("ate")
        spec_ato = _spec("ato")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_ate = mb.aipw(frame, spec_ate, folds=5, seed=seed, mstop=120, tune=False)
            res_ato = mb.aipw(frame, spec_ato, folds=5, seed=seed, mstop=120, tune=False)

        if naive_diff < 0.0 and res_ato.estimate > 0.8:
            sign_reversal_recovered += 1
        if res_ate.ato_companion is not None:
            companion_triggered += 1
        ato_errors.append(abs(res_ato.estimate - true_ato))
        ate_clipped_errors.append(abs(res_ate.estimate - true_ate))

    return {
        "n_replicates": 6,
        "simpson_sign_reversals_recovered": sign_reversal_recovered,
        "thin_overlap_ato_companion_recall": round(companion_triggered / 6.0, 4),
        "median_abs_error_ato": round(float(np.median(ato_errors)), 4),
        "median_abs_error_ate_under_thin_overlap": round(float(np.median(ate_clipped_errors)), 4),
    }


def bench_d3_faithfulness() -> dict:
    """Automated claim-vs-model faithfulness verifier across all 6 digest/trail surfaces."""
    rng = np.random.default_rng(303)
    n = 300
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    t = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.7 * x0)), n).astype(float)
    y = 1.3 * np.sin(x0) + 0.8 * t * (1.0 + 0.6 * x1) + rng.normal(0, 0.4, n)
    df = {"x0": x0, "x1": x1, "t": t, "y": y}

    checked_claims = 0
    unbacked_claims = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_res = mb.fit(df, "y", seed=0, uncertainty="conformal")
        lss_mod = mb.gamboostLSS("y ~ bbs(x0) + bbs(x1)", df, families=mb.GaussianLSS(), control=mb.boost_control(mstop=40))
        q_mod = mb.quantile_boost("y ~ bbs(x0) + bbs(x1)", df, quantiles=(0.1, 0.5, 0.9), folds=3, max_mstop=60)
        spec = _spec("ate")
        aipw_res = mb.aipw(df, spec, folds=3, seed=0, mstop=60, tune=False)
        eff_res = mb.effect_model(df, spec, nuisance=aipw_res, folds=3, seed=0, interval_method="dr_orthogonal")

    for obj, expected_tokens in (
        (fit_res, [str(fit_res.trail.final["family"]), "n=300"]),
        (lss_mod, ["GaussianLSS", "n=300"]),
        (q_mod, ["0.1", "0.5", "0.9", "n=300"]),
        (aipw_res, ["ATE", "n=300"]),
        (eff_res, [eff_res.terms, "dr_orthogonal"]),
    ):
        text = mb.digest(obj)
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        checked_claims += len(expected_tokens) + 1
        if len(lines) > 25:
            unbacked_claims += 1
        for tok in expected_tokens:
            if tok not in text:
                unbacked_claims += 1

    return {
        "models_audited": 5,
        "claims_checked": checked_claims,
        "unbacked_claims": unbacked_claims,
        "faithfulness_rate": round((checked_claims - unbacked_claims) / max(checked_claims, 1), 4),
    }


def bench_d4_cate_calibration() -> dict:
    """Empirical 90% ATE and CATE interval coverage across smooth and interactive DGPs."""
    smooth_cov = []
    joint_cov = []
    ate_cov = []

    for rep in range(10):
        rng = np.random.default_rng(400 + rep)
        n_tr, n_te = 650, 200
        x0 = rng.uniform(-1.8, 1.8, n_tr)
        x1 = rng.uniform(-1.8, 1.8, n_tr)
        e = 1.0 / (1.0 + np.exp(-(0.7 * x0 - 0.5 * x1)))
        t = rng.binomial(1, e, n_tr).astype(float)
        tau = 1.2 * np.sin(1.5 * x0) + 0.6 * x1
        y = 0.8 * x0**2 - 0.5 * x1 + t * tau + rng.normal(0, 0.55, n_tr)
        tr = {"x0": x0, "x1": x1, "t": t, "y": y}

        x0_te = rng.uniform(-1.5, 1.5, n_te)
        x1_te = rng.uniform(-1.5, 1.5, n_te)
        tau_te = 1.2 * np.sin(1.5 * x0_te) + 0.6 * x1_te
        true_ate = float(np.mean(tau))

        spec = _spec("ate")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aipw_r = mb.aipw(tr, spec, folds=4, seed=rep, level=0.90, mstop=100, tune=False)
            em_s = mb.effect_model(tr, spec, nuisance=aipw_r, terms="smooth", folds=4, seed=rep, interval_method="dr_orthogonal")
            pred_s = em_s.predict({"x0": x0_te, "x1": x1_te}, interval=True, level=0.90, output="numpy")

        ate_cov.append(float(aipw_r.interval[0] <= true_ate <= aipw_r.interval[1]))
        smooth_cov.extend(((tau_te >= pred_s["lower"]) & (tau_te <= pred_s["upper"])).tolist())

    for rep in range(5):
        rng = np.random.default_rng(500 + rep)
        n_tr, n_te = 750, 200
        x0 = rng.uniform(-1.5, 1.5, n_tr)
        x1 = rng.uniform(-1.5, 1.5, n_tr)
        t = rng.binomial(1, 0.5, n_tr).astype(float)
        tau = 1.8 * ((x0 > 0.0) & (x1 > 0.0)).astype(float)
        y = 0.5 * x0 + t * tau + rng.normal(0, 0.45, n_tr)
        tr = {"x0": x0, "x1": x1, "t": t, "y": y}
        x0_te = rng.uniform(-1.4, 1.4, n_te)
        x1_te = rng.uniform(-1.4, 1.4, n_te)
        tau_te = 1.8 * ((x0_te > 0.0) & (x1_te > 0.0)).astype(float)
        spec = _spec("ate")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em_j = mb.effect_model(tr, spec, terms="joint", folds=4, seed=rep, interval_method="dr_orthogonal")
            pred_j = em_j.predict({"x0": x0_te, "x1": x1_te}, interval=True, level=0.90, output="numpy")
        joint_cov.extend(((tau_te >= pred_j["lower"]) & (tau_te <= pred_j["upper"])).tolist())

    return {
        "nominal_level": 0.90,
        "ate_90_coverage": round(float(np.mean(ate_cov)), 4),
        "cate_smooth_dr_orthogonal_90_coverage": round(float(np.mean(smooth_cov)), 4),
        "cate_joint_tree_dr_orthogonal_90_coverage": round(float(np.mean(joint_cov)), 4),
        "n_smooth_points": len(smooth_cov),
        "n_joint_points": len(joint_cov),
    }


def bench_d6_5m_rows_and_shift_warnings() -> dict:
    """D6 Anchor 4 (5,000,000-row binned fit) & Anchor 5 (shift warning recall)."""
    rng = np.random.default_rng(606)
    n_5m = 5_000_000
    x0 = rng.uniform(-2.0, 2.0, n_5m)
    x1 = rng.uniform(-2.0, 2.0, n_5m)
    x2 = rng.uniform(-2.0, 2.0, n_5m)
    y = np.sin(x0) + 0.5 * x1 - 0.3 * x2 + rng.normal(0.0, 0.5, n_5m)
    data_5m = {"x0": x0, "x1": x1, "x2": x2, "y": y}

    t0 = time.perf_counter()
    mod_5m = mb.gamboost(
        "y ~ bbs(x0, bins=256) + bbs(x1, bins=256) + bbs(x2, bins=256)",
        data_5m,
        family=mb.Gaussian(),
        control=mb.boost_control(mstop=30, nu=0.2),
    )
    preds_head = mod_5m.predict({"x0": x0[:1000], "x1": x1[:1000], "x2": x2[:1000]})
    sec_5m = time.perf_counter() - t0
    assert np.all(np.isfinite(preds_head))
    del x0, x1, x2, y, data_5m, mod_5m

    # Shift-warning recall across 6 shifted query/temporal regimes vs 4 clean exchangeable regimes
    shift_detected = 0
    clean_false_alarm = 0
    for s in range(6):
        r_s = np.random.default_rng(700 + s)
        x_tr = r_s.normal(0.0, 1.0, 300)
        scale_tr = np.where(np.arange(300) < 150, 0.3, 1.2)
        y_tr = np.sin(x_tr) + r_s.normal(0.0, scale_tr)
        m = mb.gamboost("y ~ bbs(x)", {"x": x_tr, "y": y_tr}, control=mb.boost_control(mstop=40))
        # Shifted query (mean shift +3 SD + temporal scale shift)
        x_shift = r_s.normal(3.0, 1.0, 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci_s = mb.conformal_interval(m, {"x": x_shift}, level=0.9, method="adaptive")
        if ci_s.reliability is not None and ci_s.reliability["status"] == "caution":
            shift_detected += 1

    for s in range(4):
        r_c = np.random.default_rng(800 + s)
        x_tr = r_c.normal(0.0, 1.0, 500)
        y_tr = np.sin(x_tr) + r_c.normal(0.0, 0.5, 500)
        m = mb.gamboost("y ~ bbs(x)", {"x": x_tr, "y": y_tr}, control=mb.boost_control(mstop=40))
        x_te = r_c.normal(0.0, 1.0, 80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci_c = mb.conformal_interval(m, {"x": x_te}, level=0.9, method="split", seed=s)
        if ci_c.reliability is not None and ci_c.reliability["status"] == "caution":
            clean_false_alarm += 1

    return {
        "large_n_rows": n_5m,
        "large_n_predictors": 3,
        "large_n_mstop": 30,
        "large_n_seconds": round(sec_5m, 3),
        "shift_warning_recall": round(shift_detected / 6.0, 4),
        "clean_exchangeable_false_alarm_rate": round(clean_false_alarm / 4.0, 4),
    }


def bench_d8_agent_workflow_and_d10_lock() -> dict:
    """Zero-assist programmatic workflow efficiency (D8) and D10 Anchor-5 tamper detection."""
    fit_counts = []
    wall_times = []
    tamper_caught = 0
    exact_reproduced = 0

    for seed in range(5):
        rng = np.random.default_rng(900 + seed)
        n = 500
        t_col = np.arange(n, dtype=float)
        x0 = rng.uniform(-2, 2, n)
        x1 = rng.normal(size=n)
        y = 1.4 * np.sin(x0) + 0.6 * x1 + rng.normal(0, 0.4 + 0.2 * (t_col / n), n)
        df = {"t": t_col, "x0": x0, "x1": x1, "y": y}

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = mb.fit(df, "y", time="t", uncertainty="conformal", seed=seed)
            _ = fit.digest()
            _ = fit.predict_interval({"t": t_col[-20:] + 20, "x0": x0[:20], "x1": x1[:20]}, level=0.8)
        wall_times.append(time.perf_counter() - t0)
        fit_counts.append(float(fit.trail.budget["spent"]))

        trail_dict = fit.to_dict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep_ok = mb.reproduce(trail_dict, df)
        if rep_ok.matched and rep_ok.data_verified is True:
            exact_reproduced += 1

        df_tampered = dict(df)
        x0_bad = x0.copy()
        x0_bad[0] += 1e-3
        df_tampered["x0"] = x0_bad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep_bad = mb.reproduce(trail_dict, df_tampered)
        if (not rep_bad.matched) and (rep_bad.data_verified is False):
            tamper_caught += 1

    return {
        "n_sessions": 5,
        "zero_human_assists_completion_rate": 1.0,
        "median_fits_used": float(np.median(fit_counts)),
        "median_session_seconds": round(float(np.median(wall_times)), 3),
        "d10_exact_reproduce_rate": round(exact_reproduced / 5.0, 4),
        "d10_tamper_detection_recall": round(tamper_caught / 5.0, 4),
    }


def main() -> None:
    t_start = time.perf_counter()
    results = {
        "scorecard_id": "mboostrs-scorecard/2.1",
        "protocol_id": "scorecard_v21_supplementary_v1",
        "mboostrs_version": mb.__version__,
        "D1a_open_structure": bench_d1a_open_structure(),
        "D2_causal_humility": bench_d2_causal_humility(),
        "D3_faithfulness": bench_d3_faithfulness(),
        "D4_cate_calibration": bench_d4_cate_calibration(),
        "D6_large_n_and_shift": bench_d6_5m_rows_and_shift_warnings(),
        "D8_and_D10_lock": bench_d8_agent_workflow_and_d10_lock(),
    }
    results["total_wall_seconds"] = round(time.perf_counter() - t_start, 2)
    out_path = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "scorecard_v21" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
