# Findings report — sharp_smooth.csv (Arm B: sklearn + xgboost, benchmark v4)

**Analyst:** benchmark agent (arm B: scikit-learn + xgboost only; no mboostrs)
**Wall clock:** 2026-10-12 03:43:23 UTC → 03:55:46 UTC (~12 min)
**Data:** 800 rows × 5 cols (`x0, x1, x2, x3`, `y`); all predictors on [0,1],
no missing values. `y` continuous (mean 2.22, sd 1.09). Predictors mutually
uncorrelated (|r| < 0.06), so marginal views are clean.
**Fold claim:** 5-fold row-random CV (seed 11) throughout; numbers measure
prediction for new exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — steep sigmoid** | Pearson r=0.84; XGB gain share 0.74; permutation R² drop 1.46±0.04 |
| **x1** | **Signal — linear** | Pearson r=0.37; XGB gain share 0.17; permutation R² drop 0.31±0.01; slope +1.43±0.05 |
| x2 | Noise | Adding linear x2 to final model: CV MSE 0.1644 → 0.1644 (no gain); coef t=−1.37 |
| x3 | Noise | Adding linear x3: CV MSE 0.1644 → 0.1647 (worse); coef t=+0.06 |

## 2. Functional forms and magnitudes

Final model: **y = 0.08 + 2.94·σ((x0 − 0.51)/0.16) + 1.43·x1 + ε**,
σ = logistic, sd(ε) ≈ 0.41. In-sample R² = 0.862; honest 5-fold CV MSE =
0.166 (RMSE 0.408).

- **x0 → y: steep logistic sigmoid, total swing +2.94±0.14**, centered at
  x0 = 0.511±0.011, width w = 0.162±0.014. Max slope ≈ 4.5/unit at center.
  The transition occupies roughly x0 ∈ [0.2, 0.85] — steep, but spread over
  a wide band, not a cliff.
- **x1 → y: linear, +1.43 per unit of x1** (effect range 1.43 over x1∈[0,1]).
  Spline vs linear CV R²: 0.113 vs 0.125 — no curvature.
- **x2, x3:** no detectable association (see table).

## 3. Tested and ruled out

- **Genuine jump in x0** — rejected four independent ways (see §6).
- **Jump + smooth combined** — sigmoid+step CV MSE 0.1661 vs sigmoid-only
  0.1660; the fitted extra step is −0.016 at an arbitrary location.
- **Jumps anywhere else** — step scan over all columns × 181 candidate cuts
  on final residuals: nothing with |t| > 3.
- **Interactions** — XGBoost with interactions allowed: CV R² 0.828 vs
  additive-constrained 0.844 (interactions hurt); explicit x0·x1, x0·x2,
  x1·x2, x0·x3 product terms add ≤0.0001 to R².
- **Curvature in x1** — rejected (above). **Heteroscedasticity** —
  corr(|resid|, x0) = 0.016; constant variance stands.
- **x2/x3 effects** of any linear form — rejected (above). (XGBoost
  permutation importance showed tiny 0.024 blips for x2/x3 — investigated
  and confirmed as tree overfitting of noise, not real signal.)

## 4. Methods, libraries, seeds, fit count

- **Libraries:** scikit-learn 1.9.1, xgboost 1.7.6, pandas, numpy, scipy
  (curve_fit). No mboostrs. Python:
  /home/hatch/workspace/mboostrs-venv/bin/python.
- **Procedure:** (1) EDA — correlations, binned means (10/40/50 bins);
  (2) XGBoost (400 trees, depth 4) gain + permutation importance, manual
  partial dependence (sklearn's PDP rejected the XGB estimator object, so
  computed directly on a grid with other features at medians);
  (3) x1 form via linear-vs-spline CV; (4) residualized (y − linear x1) and
  fit step (181-cut grid) vs logistic sigmoid (curve_fit) with 5-fold CV;
  (5) sigmoid+step combined CV; (6) narrow-band discontinuity profile at
  four bandwidths; (7) interaction checks (constrained vs full XGB CV +
  explicit products); (8) residual step scan; (9) x2/x3 linear CV ablations.
- **Seeds:** 11 for all randomized procedures (CV splits, XGB, permutation).
- **Model fits:** ≈ 60 — ~15 substantive (XGB fits, curve_fit sigmoids,
  CV loops) plus ~45 trivial OLS evaluations inside the cut grids and scans.
- **Compliance:** analysis used only the CSV. No mboostrs source inspection,
  no truth-key files, no prior hypotheses about the data — every number
  above was computed from the CSV in this session.

## 5. Bottom line

Two true effects, two noise columns. y rises **linearly in x1 (+1.43/unit)**
and follows a **steep logistic sigmoid in x0 (swing +2.94, center 0.51,
width 0.16)**. x2 and x3 are noise. No interactions, no jumps,
homoscedastic. Associations only; no causal claims.

## 6. JUMP VS STEEP-SMOOTH — dedicated section

**Verdict: steep but continuous. There is no discontinuity in x0.**

Four independent lines of evidence, all pointing the same way:

1. **Step vs smooth CV shootout.** Best single step (cut at x0=0.505,
   jump +1.69): 5-fold CV MSE = 0.327. Logistic sigmoid: CV MSE = 0.166.
   The smooth model wins by 0.161 — the step's error is nearly *double*.
   A genuine jump would show the reverse (or a tie); here the data
   decisively prefer continuity.

2. **The sigmoid's width is 11.6 SE from zero.** Fitted w = 0.1624 ± 0.0140.
   A jump is the w → 0 limit of this family; w = 0 is rejected at
   overwhelming significance. The transition spans roughly ±2w ≈ ±0.32
   around the center — about a third of the predictor range.

3. **Sigmoid + step adds nothing.** Fitting a step on top of the sigmoid:
   CV MSE 0.1661 vs 0.1660 without it; the "extra step" comes out at
   −0.016 (0.5% of the sigmoid's swing) at an arbitrary location (0.71).
   There is no jump hiding on top of the smooth transition.

4. **Narrow-band discontinuity profile collapses.** Mean(y) just above vs
   just below x0 = 0.51 in shrinking bands:
   - h=0.100: +0.513 ± 0.065 (n=96/75)
   - h=0.050: +0.250 ± 0.094 (n=49/34)
   - h=0.025: +0.056 ± 0.120 (n=20/15)
   - h=0.0125: +0.089 ± 0.149 (n=15/9)
   
   A genuine jump of size J keeps the hi−lo difference ≈ J as h → 0.
   Here the difference shrinks toward zero with the bandwidth — the exact
   signature of a continuous transition viewed at increasing resolution.
   At h=0.025 the difference (0.056) is under half its standard error.

**Confidence:** high. The CV margin (step error 2× the sigmoid's), the
width estimate's distance from zero, and the collapsing narrow-band profile
are mutually reinforcing and each alone would be convincing. Softest
numbers: the exact sigmoid width (±0.014) and center (±0.011); the
qualitative verdict (continuous, not a jump) does not depend on them.
