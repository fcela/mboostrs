# Findings report — missingness.csv (Arm C: mboostrs, no skill files)

**Analyst:** benchmark agent (mboostrs only; docstrings/`help()`, no skill files)
**Wall clock:** 2026-10-12 03:43:08 UTC → 03:53:25 UTC (~10 min)
**Data:** 600 rows × 7 cols (`x0..x5`, `y`); all predictors numeric on [0,1],
`y` continuous (mean 1.99, sd 0.92). `x0` has 65 NaNs (10.8%), `x2` has
62 NaNs (10.3%); 482 complete cases. Gaussian family.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — kink at ≈0.40** | Flat below ~0.40, rises with slope +2.41 (se 0.12) above. Linear-x0 spec loses by 4.0 SE on 5-fold CV; hinge ties the smooth fit. |
| **x2** | **Signal — inverted-U** | Symmetric hump peaking at x2≈0.50, effect range ≈1.5 y-units. Zero linear correlation (−0.03); quadratic terms t=±18. |
| **x3** | **Signal — linear** | Slope +1.07 (se 0.08) per unit. Dropping it costs 4.9 SE on CV. |
| x1 | Noise | Corr 0.03; never selected by boosting; flat binned means. |
| x4 | Noise | Weak in-sample signal (OLS t=+2.4) but **no out-of-fold gain**: removing it changes 5-fold CV risk by +0.0023 (se 0.0035) = 0.66 SE in a boosting comparison, and −0.0027 (≈0.7 SE) in an OLS comparison. Binned means show no coherent pattern. |
| x5 | Noise | Corr −0.03; selected only in 5 late boosting iterations; flat profile. |

## 2. Functional forms and magnitudes

Final parametric summary (OLS on complete cases, n=482):

**y ≈ −0.50 + 2.41·max(0, x0−0.40) + 6.27·x2 − 6.28·x2² + 1.07·x3 + ε**,
sd(ε) ≈ 0.52, R² ≈ 0.68.

- **x0 → y: kink/threshold at ≈0.40 (±0.05).** Effect ≈ 0 below the kink
  (partial-dependence profile flat at ~2.0–2.2 for x0 < 0.4), then rises
  with slope ≈ +2.4 per unit of x0 above it. A 5-fold CV kink scan over
  0.15–0.65 minimizes at k=0.40 (0.35 and 0.45 within noise). The hinge form
  matches the smooth spline on CV (Δ = −0.00005 ± 0.00055) while the linear
  form is decisively worse (+0.0297 ± 0.0074).
- **x2 → y: inverted-U.** Rises from the edges to a peak at x2 ≈ 0.50,
  then falls symmetrically; total swing ≈ 1.5 y-units (PD: 0.88 → 2.37 →
  0.92). Pure curvature — marginal linear correlation is −0.03.
- **x3 → y: linear, positive.** +1.07 (se 0.08) per unit; effect range
  ≈ 1.0 over x3∈[0,1]. No detectable curvature (smooth vs linear tied).
- **x1, x4, x5:** no reliable association in any form tried.

## 3. Tested and ruled out

- **Linear x0** — rejected, +4.0 SE worse than smooth/hinge on CV.
- **x4 as a real effect** — rejected: two independent 5-fold CV ablations
  (boosting and OLS) show no predictive gain (≤0.7 SE); the in-sample
  t=+2.4 does not survive out-of-fold validation.
- **Interactions** — `screen_interactions` on model residuals: all 15
  predictor pairs ≤ −0.008 (noise floor); nothing left.
- **Discontinuities/jumps** — `screen_discontinuities` on residuals: all
  candidates |z| < 1.1 with negative eval gains; no jumps.
- **Heteroscedasticity** — mean |residual| below vs above each predictor's
  median all within 0.38–0.44; no variance driver. Constant-variance
  Gaussian stands.
- **x1/x5 effects** (linear, smooth, step) — all ≈ 0.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 only (Python bindings, Rust core), via
  /home/hatch/workspace/mboostrs-venv/bin/python; pandas/numpy for
  handling; OLS via numpy for quantification. No skill files consulted;
  API learned from docstrings/`help()`.
- **Procedure:** (1) missingness audit (counts, MCAR checks); (2)
  `column_screen` on complete cases; (3) binned-means eyeball of each
  predictor; (4) `tune_mstop` (mstop=157); (5) `gamboost` with
  bbs(x0)+bbs(x2)+linear others; (6) partial-dependence profiles;
  (7) `compare_specifications` 5-fold CV: smooth vs linear vs hinge x0,
  drop-x4, drop-x3 (seed 11, mstop 300); (8) CV kink-location scan;
  (9) `screen_interactions` and `screen_discontinuities` on residuals;
  (10) residual-spread check; (11) mean-imputation sensitivity refit
  (n=600) confirming identical qualitative conclusions.
- **Seeds:** 11 throughout.
- **Model fits:** ≈ 42 — tune_mstop 10, gamboost ~7, compare_specifications
  25; screens are native (no fits).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no prior hypotheses — every
  number above was computed from the CSV in this session.

## 5. How I handled the missing values

**What I found:** `x0` 10.8% NaN, `x2` 10.3% NaN (9 rows missing both),
all other columns complete. mboostrs refuses NaN outright — `gamboost`
raises `ValueError: Column 'x0' contains 65 NaN value(s)... Fitting and
prediction require finite values` — so handling was mandatory, not
optional.

**Missingness mechanism check:** y|missing vs y|observed = 2.13 vs 1.97
(x0), 2.06 vs 1.98 (x2) — small gaps consistent with sampling noise at
n≈65; means of the other predictors barely differ between missing and
observed rows (all within ~0.05). No evidence against MCAR, so
complete-case analysis is unbiased for the regression function, just
less efficient.

**What I did:** primary analysis on the 482 complete cases; sensitivity
check with mean-imputed x0/x2 on all 600 rows refit through the same
`gamboost` spec — identical qualitative conclusions (x0 curve range 1.58
vs 1.54, x2 hump 1.28 vs 1.49, x3 linear 0.97 vs 0.99, x4 weak 0.10 vs
0.18). I report the complete-case numbers as primary.

**Assumption and what could go wrong:** MCAR. If missingness were
actually MAR/MNAR (e.g., high-x0 values selectively missing in a way
correlated with y beyond x0), complete-case estimates of the x0 kink
location could shift and the imputation check would not catch it —
mean imputation is only valid under MCAR too, so the sensitivity check
shares the assumption rather than testing it. With ~11% missingness the
worst-case bias is bounded but not zero; a proper multiple-imputation
analysis would be the next step if the mechanism were in doubt.

## 6. Bottom line

Three true components, three noise columns. **x0** kinks at ≈0.40 (flat
below, +2.4/unit above), **x2** is a symmetric hump peaking at 0.50
(range ≈1.5), **x3** is linear (+1.07/unit). x1, x4, x5 are noise — x4's
weak in-sample signal (t=2.4) buys nothing out-of-fold. No interactions,
no jumps, no heteroscedasticity. Missing values (10.8% x0, 10.3% x2,
MCAR-plausible) handled by complete-case analysis with a mean-imputation
sensitivity check. Associations only; no causal claims.
