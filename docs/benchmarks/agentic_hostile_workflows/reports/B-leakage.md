# Findings report — leakage.csv (Arm B: sklearn/xgboost, benchmark v4)

**Analyst:** benchmark agent (arm B: scikit-learn + xgboost only; no mboostrs, no skill files)
**Wall clock:** 2026-10-12 03:43:27 UTC → 03:51:19 UTC (~8 min)
**Data:** 600 rows × 6 cols (`x0, x1, x2, x3, post`, `y`); all numeric, no
missing values. `y` continuous (mean 1.97, sd 0.91). All claims are
associations, not causal effects.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — linear** | Pearson r = 0.66 with y; OLS slope +2.08/unit (t ≈ 21.7); adding x0 alone gives CV R² = 0.43 |
| **x1** | **Signal — inverted-U** | Marginal r ≈ −0.02 (masks it), but adding x1² lifts CV R² from 0.43 → 0.66; quadratic coef −5.94 (t ≈ −18) |
| x2 | Noise | Perm importance 0.06; residual correlations < 0.05; binned residual means flat |
| x3 | Noise | Perm importance 0.06; residual correlations < 0.07; binned residual means flat |
| **post** | **LEAK — excluded** | See §6. Near-perfect copy of y (r = 0.9986); not a legitimate predictor |

## 2. Functional forms and magnitudes

Honest model (legitimate predictors only):

**y ≈ 2.42 + 2.08·(x0−0.5) + 0.08·(x1−0.5) − 5.94·(x1−0.5)² + ε**,
sd(ε) ≈ 0.52, 5-fold CV R² ≈ 0.66.

- **x0 → y: linear, positive, +2.08 per unit of x0** (effect range ≈ 2.08
  over x0 ∈ [0,1]). Binned means rise monotonically (0.95 → 2.85 across
  deciles); adding x0² gives no CV gain — no curvature.
- **x1 → y: symmetric inverted-U, peak at x1 ≈ 0.51, swing ≈ 1.48**
  (peak-to-edge). Linear term ≈ 0 (+0.08, ns) — hence the near-zero
  marginal correlation. The ∩ shape is visible in every x0 quartile
  (pivot table: y low at x1 Q1/Q4, high at Q2/Q3).
- **x2, x3:** no detectable effect — linear, quadratic, or binned; all
  residual correlations |r| < 0.07.

## 3. Tested and ruled out

- **x0 curvature** — x0² adds nothing on CV (0.6566 vs 0.6585 without it).
- **x0×x1 interaction** — OLS t = 2.35 (p = 0.019) in-sample, but CV R²
  does not improve (0.6582 vs 0.6585); judged spurious, not adopted.
- **x0×x1² and higher interactions** — no CV gain.
- **x2/x3 effects** in any form tried (linear, quadratic, binned).
- **Tree-based alternatives** — tuned XGBoost on legit features reaches
  only CV R² = 0.577 vs 0.659 for the smooth parametric form; the truth
  is smooth, and the parametric summary captures it better.
- **`post` as a predictor** — excluded (see §6). Including it gives CV
  R² = 0.995, which is the leakage trap, not real performance.

## 4. Methods, libraries, seeds, fit count

- **Libraries:** scikit-learn (LinearRegression, KFold, permutation_importance),
  xgboost (XGBRegressor), pandas, numpy. No mboostrs.
- **Procedure:** (1) descriptives + correlation screen — `post` immediately
  suspicious (r = 0.9986, identical moments to y); (2) OLS `post ~ y`
  (R² = 0.9972, resid sd 0.048) and `post ~ y + x's` (no gain) —
  confirmed downstream copy; (3) binned means for x0 form;
  (4) XGBoost CV with/without `post` (0.995 vs 0.577) — quantified the
  trap; permutation importance flagged x1 despite zero marginal
  correlation; (5) x0×x1 quartile pivot revealed the ∩ shape;
  (6) nested OLS CV comparison over polynomial terms selected
  x0 + x1 + x1²; (7) residual diagnostics vs x2, x3, x0², interactions.
- **Seeds:** 11 (all CV splits, XGB, permutation importance).
- **Model fits:** ≈ 60 (8 candidate OLS specs × 5-fold CV, 2 XGB CV runs,
  several XGB fits, permutation importance, OLS diagnostics).
- **Compliance:** analysis used only the CSV. No mboostrs source
  inspection, no `mboostrs.datasets`, no truth-key files, no prior
  hypotheses — every number above was computed from the CSV in this session.

## 5. Bottom line

Two true effects: **x0 linear (+2.08/unit)** and **x1 inverted-U (peak ≈
0.51, swing ≈ 1.48)**; x2 and x3 are noise. **`post` is a leaked
downstream copy of y and was excluded** — with it, CV R² = 0.995
(illusory); the honest model reaches CV R² ≈ 0.66. Associations only;
no causal claims.

## 6. LEAKAGE ASSESSMENT

| Column | Legitimate predictor? | Evidence | Action |
|---|---|---|---|
| x0 | Yes | Predicts y (r = 0.66); not a function of y | Used |
| x1 | Yes | Predicts y via ∩ shape; not a function of y | Used |
| x2 | Yes (but noise) | No relation to y in any form | Excluded as noise |
| x3 | Yes (but noise) | No relation to y in any form | Excluded as noise |
| **post** | **No — downstream proxy** | r = 0.9986 with y; OLS `post = 1.0009·y − 0.001`, R² = 0.9972, resid sd = 0.048; adding x0–x3 changes R² by 0.000005; `post − y` is pure noise (sd 0.048, |corr| < 0.04 with every column, 0.28% of y's variance) | **Excluded from all modeling** |

Why `post` cannot be legitimate: a predictor must be knowable at prediction
time, but `post` is y itself plus 0.3%-variance noise — it carries the
target's own realization, not ex-ante information. Any model using it reports
CV R² = 0.995 while learning nothing about the real structure (with `post`
in the model, XGBoost assigns ~zero importance to x0 and x1 — the leak
masks every true effect). The honest workflow is: drop `post` first, then
model. That is what this report does.
