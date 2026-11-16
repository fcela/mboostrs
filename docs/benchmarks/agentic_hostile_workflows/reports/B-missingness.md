# Findings report — missingness.csv (Arm B: scikit-learn + xgboost, benchmark v4)

**Analyst:** benchmark agent (arm B: sklearn/xgboost only; no mboostrs)
**Wall clock:** 2026-10-12 03:43:03 UTC → 03:54:48 UTC (~12 min)
**Data:** 600 rows × 7 cols (`x0, x1, x2, x3, x4, x5`, `y`); all numeric,
predictors on [0,1], y continuous (mean 1.99, sd 0.92). Missing values in
**x0 (65, 10.8%)** and **x2 (62, 10.3%)**; x1, x3, x4, x5, y complete.

All claims below are associations, not causal effects. Analysis used only the
CSV — no package-source inspection, no truth-key files, no prior hypotheses.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — kinked/hinge** | XGB perm. importance 0.64 (top); hinge fit: flat below ≈0.36, slope +2.4/unit above; hinge R²=0.31 vs linear 0.28 |
| **x2** | **Signal — inverted-U** | XGB perm. importance 0.48; quadratic peak ≈0.49, R²=0.28; edges ≈1.2, peak ≈2.6 |
| **x3** | **Signal — linear** | XGB perm. importance 0.28; OLS slope +1.05/unit, R²=0.11 |
| x1 | Noise | Flat binned means; +x1 to final model: CV R² 0.5702→0.5695 (nothing); MI with residuals ≈0.004 |
| x4 | Noise | Flat binned means; +x4: CV R² 0.5702→0.5706 (nothing); MI with residuals = 0.000. (Its XGB perm. importance of 0.09 was a mirage — trees spending splits on a noise column.) |
| x5 | Noise | Flat binned means; +x5: CV R² unchanged; MI with residuals ≈0.017 (noise floor) |

## 2. Functional forms and magnitudes

Final parametric summary (median-imputed, n=600):
**y ≈ −0.10 + 2.29·max(0, x0−0.36) + 6.22·x2 − 6.21·x2² + 1.06·x3 + ε**,
sd(ε)≈0.60, 5-fold CV R²=0.57 (0.66 on complete cases, n=526).

- **x0 → y: hinge (kink), not smooth.** Flat at ≈1.47 for x0 ≲ 0.36, then
  rising ≈+2.4 per unit of x0 above the kink. Kink location profile is
  flat-topped: 0.34–0.40 all within noise → report **0.36 ± 0.04**.
  Two-slope fit gives −0.18 below / +2.43 above — the below-slope is
  negligible, so the flat-below hinge stands. **No jump at the kink:**
  narrow-band (±0.05) means 1.39 below vs 1.42 above (diff 0.03) — a
  genuine kink, not a discontinuity.
- **x2 → y: symmetric inverted-U (hump).** Peak at **0.49** (R² profile
  max), spanning ≈1.4 outcome units (edges ≈1.2–1.3, peak ≈2.6).
  Quadratic and tent (piecewise-linear peak) forms tie (R² 0.2772 vs
  0.2779, symmetric tent slopes ±3.4) — reported as a smooth hump; the
  data cannot distinguish the two shapes.
- **x3 → y: linear, +1.05 per unit of x3** (effect range ≈1.05 over x3∈[0,1]).
- **x1, x4, x5:** no detectable association in any form tried.

The 4-term parametric model (CV R² 0.57) beats tuned XGBoost (CV R² 0.49) —
the structure is low-dimensional and smooth, exactly where a typed
parametric summary wins over a tree ensemble.

## 3. Tested and ruled out

- **x1, x4, x5 as predictors** (marginal binned means flat; CV ablation
  zero gain; zero mutual information with final-model residuals).
- **Interactions** — all pairs among {hinge-x0, x2, x3}: CV R² unchanged
  (0.5702 → 0.5664–0.5704). None.
- **Jump/discontinuity at the x0 kink** — narrow-band check shows
  continuity (diff 0.03); it is a kink in slope, not a step in level.
- **Jumps anywhere in x2** — best single split is at x2=0.1, which is just
  the hump's rising tail, not a discontinuity.
- **Heteroscedasticity** — residual sd by predictor quartile ranges
  0.53–0.70 with no systematic pattern; constant-variance assumption stands.
- **x0 smooth alternatives** — hinge beats plain linear (R² 0.309 vs
  0.276); no evidence for additional kinks.

## 4. Methods, libraries, seeds, fit count

- **Libraries:** scikit-learn 1.9.1, xgboost 1.7.6, pandas, numpy. No
  mboostrs anywhere in this run.
- **Procedure:** (1) missingness audit (counts, overlap, MCAR checks);
  (2) XGBoost (n=400, depth 4, lr 0.05, native NaN handling) → gain +
  permutation importance; (3) decile binned-means per predictor;
  (4) hinge scan for x0 (kink 0.15–0.85 step 0.01), quadratic/tent fits
  for x2, linear for x3; (5) narrow-band continuity check at the kink;
  (6) CV ablation for x1/x4/x5 and all pairwise interactions (5-fold,
  shared splits); (7) imputation-sensitivity check (complete-case vs
  median-imputed); (8) residual diagnostics (MI with residuals,
  quartile-sd homoscedasticity check).
- **Seeds:** 11 for all randomized procedures (CV splits, XGB, MI).
- **Model fits:** ≈ 11 gradient-boosting fits (incl. two 5-fold CVs) and
  ≈ 200 small OLS fits (scans, ablations, diagnostics).
- **Compliance:** analysis used only the CSV. No mboostrs import, no
  package-source inspection, no truth-key files, no prior hypotheses —
  every number above was computed from the CSV in this session.

## 5. Bottom line

Three true effects, three noise columns. **x0** kinks at ≈0.36 (flat
≈1.47 below, +2.4/unit above, no jump); **x2** is a symmetric hump
peaking at ≈0.49 (amplitude ≈1.4); **x3** is linear (+1.05/unit).
**x1, x4, x5 are noise** — x4's nonzero tree importance did not survive
any honest check. No interactions, no discontinuities, no
heteroscedasticity. Final honest CV R² ≈ 0.57. Associations only.

## 6. HOW YOU HANDLED THE MISSING VALUES

**What I found:** 65 NaN in x0 (10.8%), 62 in x2 (10.3%), 9 rows missing
both (≈ independent: expected 6.7 under independence). Missingness looks
**approximately MCAR**: missing-indicators correlate 0.05 (x0) and 0.03
(x2) with y; covariate means barely differ between missing/observed rows
(largest gap: x3 mean 0.563 vs 0.521, small vs its sd ≈0.29).

**What I did:** two-track strategy.
1. *Discovery* with XGBoost's **native NaN handling** — no imputation at
   all, so the importance ranking and functional-form exploration never
   saw an imputed value.
2. *Quantification* with **median imputation** + OLS for the parametric
   summary (needed complete rows for the closed-form fits).
3. *Sensitivity check:* kink location 0.37 (complete-case, n=535) vs 0.35
   (imputed, n=600); combined-model CV R² 0.66 vs 0.57. Same qualitative
   conclusions either way — imputation only attenuates, nothing flips.

**Why:** under MCAR, complete-case analysis is unbiased (just less
efficient) and median imputation is a mild, transparent approximation;
agreement between the two is the robustness check. XGBoost-native
handling avoids the imputation question entirely for discovery.

**Assumption it relies on:** missingness is MCAR, or at least MAR
independent of y given the observed predictors.

**What could go wrong:**
- If missingness is **MNAR** (e.g., high x0 values systematically more
  likely missing), the above-kink slope (+2.4) is biased — most likely
  attenuated, since the steep region would be under-observed. Nothing in
  the observed data can rule this out; it is untestable from the CSV alone.
- Single median imputation **understates uncertainty** (treated imputed
  values as known) and shrinks effects toward zero — hence the honest CV
  estimate (0.57) is reported on the imputed data, and the complete-case
  number (0.66) is shown alongside rather than instead.
- 9 doubly-missing rows contribute to neither the x0 nor the x2 fits;
  with only 600 rows this is a minor efficiency loss, not a bias under MCAR.
