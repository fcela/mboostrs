# Findings report — real_train.csv (Arm C: mboostrs, no skill files)

**Analyst:** benchmark agent (mboostrs only; docstrings/`help()`, no skill files)
**Wall clock:** 2026-10-12 03:43:30 UTC → 03:52:48 UTC (~9 min)
**Data:** 354 rows × 11 cols (`age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`, `y`);
all numeric, features pre-standardized (mean≈0, sd≈0.047), no missing values.
`y` continuous, mean 153.6, sd 77.4, range 25–346 → Gaussian family.
All claims below are associations, not causal effects. Analysis used only the
CSV; no package-source inspection, no `mboostrs.datasets`, no truth files.

## 1. Which predictors matter

Final model: `y ~ bols(age)+bols(sex)+bols(bmi)+bols(bp)+bols(s1)+bols(s2)+bols(s3)+bols(s4)+bols(s5)+bols(s6)`, Gaussian, mstop=293 (CV-tuned).

| Predictor | Verdict | Evidence |
|---|---|---|
| **s5** | Strong signal | Fitted slope +602 (largest); marginal r=0.573 |
| **bmi** | Strong signal | Slope +486; marginal r=0.570 |
| **bp** | Signal | Slope +363; marginal r=0.439 |
| **s3** | Signal (negative) | Slope −258; marginal r=−0.401 |
| **s1** | Signal (negative) | Slope −172; marginal r=0.196 (sign flips jointly — suppressor pattern among correlated serum measures) |
| **s6** | Weak signal | Slope +51; marginal r=0.352 |
| **sex** | Weak signal | Slope −328 on a ±0.05-scale variable (≈−31 across its range); marginal r≈−0.003 (pure suppressor: nothing alone, contributes jointly) |
| **age** | Weak signal | Slope +43; marginal r=0.168 |
| **s2** | ~Noise | Slope +8; nothing survives jointly |
| **s4** | ~Noise | Slope +2; marginal r=0.414 fully absorbed by s3/s5 jointly |

(Slopes are per-unit on the standardized features; multiply by each
feature's ~0.1 range for effect sizes: s5 ≈ +60, bmi ≈ +48, bp ≈ +36 per
0.1 feature units.)

## 2. Functional forms and magnitudes

All effects are linear — no curvature, kinks, jumps, or interactions
survived testing (see §3). Final fitted equation (response scale):

y ≈ 153.55 + 43.2·age − 327.5·sex + 486.5·bmi + 362.8·bp − 172.1·s1
+ 8.1·s2 − 258.4·s3 + 2.4·s4 + 602.1·s5 + 51.4·s6

In-sample RMSE 52.6; repeated 5-fold CV RMSE 54.3 (see §6).

## 3. Tested and ruled out

- **Nonlinearity:** `bbs(x, df=4)` for the six strongest predictors vs
  all-linear — 5-fold CV MSE 3019 vs 2953. Smooth terms lose; also triggered
  fold-extrapolation warnings on s4/s5. Ruled out.
- **Interactions:** `screen_interactions` on linear-fit residuals — all 45
  pairs ≤ +9.8 risk units against MSE ≈ 2953 (≤0.3%); top pair age×s5
  added as `btree(age, s5, maxdepth=2)` made CV *worse* (3537 vs 2953).
  Ruled out.
- **Discontinuities:** no jumps — `screen_discontinuities` not indicated;
  residual structure is smooth noise. (Not run: nothing in the marginal or
  residual screens suggested a regime break.)
- **Dropping weak predictors:** no-sex spec CV MSE 3153 vs 2953 full —
  worse at the shared mstop (path sensitivity); boosting's shrinkage already
  handles noise variables, so all 10 stay.
- **Heteroscedasticity:** `heteroscedasticity` check t=3.26, spread ratio
  1.68 — mild evidence residual spread varies. Not pursued: under Gaussian
  scoring of point predictions, a sigma model does not change mean
  predictions. Noted as a limitation for interval claims.
- **mstop sensitivity:** CV risk flat 2952–2955 for mstop 293→2000 —
  converged; 293 (tuned, not at boundary) is the honest choice.
- **OLS sanity check:** plain OLS 5-fold MSE 2904 vs mboostrs 2947 —
  OLS is ~1.5% better; the boosting selection path costs a little bias
  here (n=354, p=10 is OLS-friendly). Kept the mboostrs fit per the
  toolkit rule; the gap is reported honestly.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs (Python bindings, Rust core) only, via
  `/home/hatch/workspace/mboostrs-venv/bin/python`; pandas/numpy for
  handling; sklearn used *only* for the OLS sanity baseline (no modeling
  decisions rest on it).
- **Procedure:** marginal screen → `tune_mstop` (all-linear: mstop=293) →
  `cv_risk` spec comparisons (linear vs smooth, ±interaction, ±sex) →
  `screen_interactions`, `heteroscedasticity` → repeated 5-fold CV
  (seeds 11/22/33) of the final spec → refit on all 354 rows.
- **Seeds:** 11 (tuning, screens, primary CV), 22, 33 (CV repeats).
- **Model fits:** ≈ 60 — tune_mstop 5, cv_risk calls 3+3+3+3 folds ×2
  (spec + repeats) ≈ 40, final refit 1, screens fit nothing (native).
- **Compliance:** CSV only; no source inspection, no truth keys, no skill
  files, no prior hypotheses.

## 5. Bottom line

Essentially linear problem: s5, bmi, bp positive; s3, s1 negative;
s6/sex/age weak; s2/s4 noise. No curvature, interactions, or jumps.
Final all-linear boosted fit, CV RMSE ≈ 54.3.

## 6. OUT-OF-SAMPLE HONESTY

- **CV estimate:** repeated 5-fold CV (seeds 11/22/33), same folds per
  seed across specs: MSE 2953 / 2960 / 2929 → mean 2947 (sd 13) →
  **RMSE 54.3 ± ~0.4**. In-sample RMSE 52.6, so optimism ≈ 1.7 points.
- **How validated:** `cv_risk` with fixed mstop=293 (tuned once, seed 11,
  not re-tuned per repeat — slight selection optimism remains, estimated
  small since risk is flat in mstop beyond 293).
- **What could make held-out worse:** (a) the hidden test split may not
  be exchangeable rows — if it's a later time period or shifted
  population, all bets degrade; (b) mild heteroscedasticity (spread ratio
  1.68) means RMSE is an average — errors are larger in high-spread
  regions; (c) OLS beats this fit by ~1.5% in CV, so a pure-linear
  competitor may edge it; (d) n=354 is small — CV sd across seeds is
  ~13 MSE points, and a single held-out draw can easily land ±2 RMSE
  from the estimate.
- **Limitations:** linear-only model cannot capture any true nonlinearity
  the smooth screen lacked power to see (n=354); no missing-data or
  distribution-shift handling was built in (none present in train);
  uncertainty intervals were not produced — point predictions only, and
  the heteroscedasticity finding means any naive ±RMSE band is
  miscalibrated across the feature space.
- **Deliverables:** model pickle
  `reports/C-real-model.pkl` (dict: fitted GAMBoostModel + feature
  order; round-trip verified identical predictions); predict spec
  `reports/C-real-predict.txt`.
