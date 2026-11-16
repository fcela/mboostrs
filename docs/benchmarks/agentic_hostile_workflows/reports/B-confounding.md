# Findings report — confounding.csv (Arm B: scikit-learn + xgboost, benchmark v4)

**Analyst:** benchmark agent (arm B: sklearn/xgboost only; no mboostrs, no skill files)
**Wall clock:** 2026-10-12 03:43:12 UTC → 03:46:36 UTC (~3.5 min)
**Data:** 600 rows × 5 cols (`x0`, `g`, `x1`, `x2`, `y`); `g` is a 0/1 group
indicator (298 zeros, 302 ones); all predictors on [0,1]; no missing values.
`y` continuous (mean −0.25, sd 0.85).

All claims below are associations, not causal effects.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **g** | **Signal — large negative level shift** | Marginal r = −0.819 with y; group means 0.449 (g=0) vs −0.947 (g=1); OLS coef −2.04 conditional on x0; 5-fold CV R² 0.67 alone; XGBoost gain importance 0.91, permutation importance +2.04 |
| **x0** | **Signal — positive linear, CONFOUNDED by g** | Marginally *negative* (r = −0.423, OLS slope −1.35 alone) but **+1.58 per unit after conditioning on g**; within-group slopes +1.63 (g=0) / +1.53 (g=1); adding x0 to g lifts CV R² 0.67 → 0.77 |
| x1 | Noise | Marginal r = −0.075; OLS coef +0.01; binned means flat; no CV gain (0.770 → 0.770); XGB gain 0.016 |
| x2 | Noise | Marginal r = +0.082; OLS coef +0.02; binned means flat; no CV gain; XGB gain 0.017 |

## 2. Functional forms and magnitudes

Adopted model: `y ≈ −0.03 + 1.58·x0 − 2.04·g + ε`, sd(ε) ≈ 0.41,
5-fold CV R² = 0.77.

- **g → y: intercept shift, −2.04** (g=1 vs g=0), the dominant effect.
  Marginal mean difference is −1.40; the conditional estimate is larger
  because g correlates with x0 (see §6).
- **x0 → y: linear, +1.58 per unit of x0**, identical in both groups
  (no x0×g interaction: interaction coef −0.10, CV R² unchanged at 0.772).
  Binned means rise steadily within each group; quadratic terms are small
  and sign-inconsistent across groups (+0.89 / −0.77) with negligible R²
  gain — linear stands. No kink/threshold, no jump.
- **x1, x2:** no detectable association in any form tried (linear,
  12-bin means, XGBoost splits, residual correlation +0.007 / +0.015).

## 3. Tested and ruled out

- **x1/x2 effects** (linear, binned/nonlinear, tree splits, residual
  association after x0+g): all ≈ 0.
- **x0×g interaction**: coef −0.098, no CV gain; within-group slopes agree.
- **Curvature in x0**: quadratic adds nothing; binned profile linear.
- **Jumps / thresholds / discontinuities** in any predictor: none; x0's
  profile is smooth within groups.
- **Heteroscedasticity**: residual sd 0.396 (g=0) vs 0.417 (g=1);
  0.411 (x0<0.5) vs 0.402 (x0≥0.5). Constant variance stands.
- **The marginal x0 story**: a model on x0 alone (CV R² 0.17, negative
  slope) is decisively beaten by g+x0 (CV R² 0.77) — the marginal fit is
  a confounding artifact.

## 4. Methods, libraries, seeds, fit count

- **Libraries:** scikit-learn 1.9.1 (LinearRegression, KFold,
  permutation_importance), xgboost 1.7.6 (XGBRegressor, 300 trees,
  depth 3, lr 0.05), pandas/numpy. No mboostrs, no skill files.
- **Procedure:** (1) marginal correlations + inter-predictor correlations;
  (2) group means by g; (3) OLS marginal vs joint; (4) within-group
  y~x0 regressions; (5) 12-quantile binned means (within g for x0,
  overall for x1/x2); (6) interaction and quadratic probes;
  (7) 5-fold CV R² over nested specs (seed 11); (8) XGBoost fit +
  permutation importance (10 repeats, seed 11) + gain importance;
  (9) residual diagnostics on the adopted model.
- **Seeds:** 11 throughout.
- **Model fits:** ≈ 45 (OLS ~15 incl. 25 CV fold-fits across 5 nested
  specs, 1 XGBoost fit; permutation importance scores without refitting).
- **Compliance:** analysis used only the CSV. No package-source
  inspection, no truth-key files, no prior hypotheses — every number above
  was computed from the CSV in this session.

## 5. Bottom line

Two true components, two noise columns. **g** shifts y down by ≈2.0;
**x0** raises y by ≈1.6 per unit, linearly, in both groups — but appears
*negatively* related to y marginally (r = −0.423) because x0 is strongly
correlated with g (r = 0.766) and g lowers y. **x1 and x2 are noise.**
Associations only; no causal claims.

## 6. ASSOCIATION VS CAUSATION

| Effect | Marginal association | After conditioning | Supported language |
|---|---|---|---|
| g → y | −1.40 mean difference (r = −0.819) | −2.04 (OLS w/ x0) | g predicts y strongly in every specification. Whether g *causes* the shift cannot be determined from these data — g could mark a pre-existing group difference. No causal claim warranted. |
| x0 → y | **−1.35** per unit (r = −0.423) — negative | **+1.58** per unit (OLS w/ g) — positive | Sign reversal is fully explained by confounding: x0 and g correlate at 0.766, so marginally x0 proxies for g=1 (low y). Within each g-group the x0–y slope is positive and stable (+1.63/+1.53). The conditional association is the honest descriptive one; the marginal one is a textbook confounding artifact. Causal language still unwarranted — an unmeasured common cause of (x0, y) within groups cannot be ruled out observationally. |
| x1 → y | −0.075 (r) | +0.01 (OLS w/ rest) | No association in any specification. Noise. |
| x2 → y | +0.082 (r) | +0.02 (OLS w/ rest) | No association in any specification. Noise. |

**What the evidence does and doesn't support:** it supports reporting the
conditional estimates (+1.58 for x0, −2.04 for g) and explicitly warning
that the marginal x0 estimate has the wrong sign. It does not support any
statement of the form "changing x0 would change y" or "g causes lower y" —
this is a simulated observational cross-section with no intervention,
no temporal ordering, and no causal graph to license such claims.
