# Findings report — heteroscedastic.csv (Arm B: scikit-learn + xgboost, benchmark v4)

**Analyst:** benchmark agent (arm B: sklearn/xgboost only; no mboostrs)
**Wall clock:** 2026-10-12 03:43:05 UTC → 03:56:23 UTC (~13 min)
**Data:** 800 rows × 6 cols (`x0..x4`, `y`); all numeric, predictors on [0,1],
no missing values. `y` continuous (mean 1.96, sd 1.25).

All claims below are associations, not causal effects.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — mean effect, linear** | Pearson r = 0.459 with y; binned means rise monotonically 0.99 → 2.97 across x0; OLS slope +2.01 (se ≈ 0.12); permutation importance highest (0.57) |
| **x1** | **Signal — mean effect, inverted-U** | Pearson r = 0.050 (≈ 0) but binned means show a symmetric hump peaking ≈ 0.45–0.50 (1.13 → 2.68 → 1.36); well fit by 1.56·sin(π·x1); permutation importance 0.38 |
| **x2** | **Signal — variance effect only, no mean effect** | Binned mean of y flat across x2 (1.8–2.2); residual sd rises monotonically 0.30 → 1.60 across x2; Breusch–Pagan n·R² = 106.7 (χ²₁ crit 3.84), t = +11.1 |
| x3 | Noise | Marginal r = 0.048; binned residual means flat (max |dev| 0.13, ≈ noise); BP n·R² = 0.3 |
| x4 | Noise for the mean; **weak/suggestive variance signal** (see §6) | No mean association (binned residual means flat); variance association small, negative, p ≈ 0.01 uncorrected — not conclusive |

## 2. Functional forms and magnitudes

Final mean model (OLS, refit on all 800 rows):

**ŷ = −0.05 + 2.01·x0 + 1.56·sin(π·x1)**

- **x0 → y: linear, +2.01 per unit of x0** (effect range ≈ 2.0 over x0 ∈ [0,1]).
  Binned means are monotone with no visible kink or curvature.
- **x1 → y: symmetric inverted-U, peak at x1 ≈ 0.5, amplitude ≈ 1.56.**
  Zero linear correlation is expected for a symmetric hump, not evidence of
  no effect. sin(π·x1) and a quadratic in x1 fit identically well (R² = 0.352
  for the two-term mean model either way); I report the sine form but the
  honest claim is "symmetric hump, peak ≈ 0.5, swing ≈ 1.5".
- **x2 → y: no mean effect.** It acts on the spread (see §6).
- **x3, x4: no mean effect** in any form tried (linear, binned, tree splits).

Fit quality: 5-fold CV OOF RMSE = **1.011** for the parametric mean model,
vs 1.050 for the best XGBoost (depth-1 stumps) and 1.253 for the null (y sd).
The parametric form beats the flexible tree fit — the structure is simple
and the trees were mostly fitting noise.

## 3. Tested and ruled out

- **Interactions:** XGBoost max_depth 1 vs 3 vs 5 give OOF RMSE 1.050 / 1.058 /
  1.052 — identical within noise. No interaction improves on the additive model.
- **Jumps / discontinuities:** binned mean profiles of y vs x0 and x1 are
  smooth; no step-like transitions.
- **x1 asymmetry or shifted peak:** binned means symmetric about ≈ 0.5.
- **x0 nonlinearity:** monotone, near-linear binned profile; nothing suggests
  a kink.
- **x3/x4 mean effects:** linear correlations ≈ 0, binned residual means flat,
  permutation importance lowest (0.11 / 0.07 — the nonzero values are the
  usual tree-importance bleed, not real effects).
- **Constant variance:** decisively rejected (see §6).

## 4. Methods, libraries, seeds, fit count

- **Libraries:** scikit-learn 1.9.1 (LinearRegression, KFold,
  permutation_importance), xgboost 1.7.6 (XGBRegressor), pandas, numpy.
  No mboostrs anywhere in this run.
- **Procedure:** (1) EDA + marginal correlations; (2) decile-binned means and
  stds of y vs each predictor (mean-shape and variance screen in one pass);
  (3) XGBoost 5-fold CV at max_depth 1/3/5 → interaction check + baseline;
  (4) permutation importance; (5) OLS fits for x0 (linear), x1 (quadratic vs
  sine); (6) residual diagnostics vs x2–x4; (7) Breusch–Pagan-style
  resid²-on-x regressions + binned residual-sd profiles for the variance
  function; (8) stratified check of the weak x4 variance signal within x2
  halves; (9) parametric 5-fold CV for the final OOF number.
- **Seeds:** 0 (KFold splits, permutation_importance), 11 (XGBRegressor).
- **Model fits:** ≈ 24 — 15 XGBoost CV fits (3 configs × 5 folds), 1 full
  XGBoost fit (+50 permutation evaluations), ~8 OLS fits (instant).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no prior hypotheses — every
  number above was computed from the CSV in this session.

## 5. Bottom line

y = −0.05 + 2.01·x0 + 1.56·sin(π·x1) + ε, where **sd(ε) ≈ 0.22 + 1.42·x2**
— the noise variance grows ~5× across x2 while the mean does not depend on
x2 at all. x3 is pure noise; x4 has no mean effect and at most a weak
variance signal. No interactions, no jumps. Associations only.

## 6. VARIANCE ANALYSIS

**Is the variance constant? No — decisively not.** After fitting the mean
model above, residual sd by x2 octile:

| x2 bin center | 0.04 | 0.12 | 0.20 | 0.27 | 0.35 | 0.44 | 0.52 | 0.62 | 0.71 | 0.79 | 0.87 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resid sd | 0.29 | 0.39 | 0.52 | 0.59 | 0.71 | 0.91 | 0.84 | 1.10 | 1.24 | 1.29 | 1.60 | 1.53 |

Breusch–Pagan-style regression of resid² on x2: slope +2.64, t = **+11.1**,
n·R² = **106.7** (χ²₁ 5% critical value 3.84). The same test on x0, x1, x3
gives n·R² of 0.1–0.4: nothing.

**Estimated variance function:** sd(x2) ≈ **0.22 + 1.42·x2**
(R² = 0.978 on the 12 binned sds), i.e. Var(y|x) ≈ (0.22 + 1.42·x2)²,
independent of the other predictors. An exponential form
sd ≈ 0.35·exp(1.74·x2) fits nearly as well (R² = 0.950); the data do not
strongly discriminate linear-vs-exponential in sd, but both agree the
variance rises steeply and monotonically in x2.

**x4 caveat:** resid² on x4 gives t = −2.98 (n·R² = 8.8); the sign is
consistent within x2 strata (corr(x4, |resid|) = −0.08/−0.09 in each half)
and median |resid| drifts 0.54 → 0.46 across x4. This is a *suggestive*
secondary variance signal — small (≈ ±10% in typical |resid|, vs x2's 5×
in sd), p ≈ 0.01 uncorrected over 5 predictors tested. I report it as
not-conclusive: it could be a weak real effect or chance.

**What I did about it in the final model:** stated explicitly —
**my toolkit cannot model the variance function directly.** Neither plain
scikit-learn regressors nor standard XGBRegressor do distributional
(mean + variance) regression; there is no native GAMLSS-style joint fit and
no out-of-the-box prediction intervals that account for heteroscedasticity.
I estimated the variance function in a *second stage* (regressing
log(resid²) on x2), which is adequate for describing and reporting the
variance but is not a joint likelihood fit. Two consequences I want on the
record: (a) under squared-error loss the conditional mean is the optimal
point predictor *regardless* of heteroscedasticity, so my point predictions
are unaffected — the variance model matters for intervals and uncertainty,
not for the ŷ values; (b) the mean-model standard errors above assume
constant variance and are therefore approximate — with variance rising in
x2, efficiency could be gained by weighting, which this toolkit does not do
natively for this workflow.
