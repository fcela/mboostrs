# Findings report — leakage.csv (Arm C: mboostrs, no skill files)

**Analyst:** benchmark agent (mboostrs only; docstrings/`help()`, no skill files)
**Data:** 600 rows; predictors x0, x1, x2, x3, post (all numeric, no missing
values); response y (mean 1.97, sd 0.91).
**Wall clock:** 2026-10-12 03:43:02 UTC → 03:53:44 UTC (~11 min)

All claims below are associations, not causal effects.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — linear** | Pearson r = 0.664; `column_screen` hint "line", linear_r2 = 0.441; boosted slope +2.01/unit (t large) |
| **x1** | **Signal — curved (inverted-U)** | Pearson r = −0.024 (zero linear signal) but `column_screen` curve_r2 = 0.245, curvature = 0.244, hint "curve"; smooth beats linear by 0.186 OOF risk (se 0.020) = 9.4 SE |
| x2 | Noise | linear_r2 = 0.003; adding it + x3 changes OOF risk −0.0009 (se 0.0008) — nothing |
| x3 | Noise | linear_r2 = 0.001; same CV ablation — nothing |
| **post** | **NOT a legitimate predictor — target leakage** | See Leakage Assessment below. Excluded from all modeling. |

## 2. Functional forms and magnitudes

Final model: `y ~ bols(x0) + bbs(x1, df=4)`, Gaussian, mstop = 83 (tuned,
not at boundary). In-sample R² = 0.67, residual sd = 0.53.

- **x0 → y: linear, +2.01 per unit of x0** (effect range 2.01 over x0∈[0,1];
  binned means rise monotonically 0.95 → 2.84).
- **x1 → y: symmetric inverted-U, peak at x1 ≈ 0.50** (fitted peak value
  2.45, edges ≈ 1.04–1.27, total range ≈ 1.41). Zero linear correlation —
  invisible to any linear screen; found via the curve_r2 marginal and
  confirmed by the 9.4-SE smooth-vs-linear CV comparison.
- **x2, x3:** no detectable effect in any form tried.

## 3. Tested and ruled out

- **post as a predictor** — excluded as leakage (see §6). A model including
  it was fit once, purely to quantify the trap.
- **x2, x3 effects** (linear; CV ablation shows zero gain).
- **Linear x1** — decisively worse than the smooth (OOF risk +0.186, se
  0.020).
- **Jumps/regime changes** — `screen_discontinuities` on final residuals:
  top |z| < 1.0, all eval gains negative. Ruled out.
- **Interactions** — `screen_interactions` on final residuals: all 6 pairs
  ≤ 0 (noise floor). Ruled out.
- **Heteroscedasticity** — `mboostrs.heteroscedasticity`: overall t = 0.44;
  by-column |t| < 1.3; spread ratios ≈ 1. Constant-variance Gaussian stands.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 only (Python bindings, Rust core), via
  `/home/hatch/workspace/mboostrs-venv/bin/python`; pandas/numpy for
  handling. No skill files consulted; API learned from docstrings/`help()`.
- **Procedure:** (1) EDA + correlations — `post` immediately suspicious
  (r = 0.9986 with y, near-identical moments); (2) `column_screen` on
  x0–x3 — x0 "line", x1 "curve" despite r ≈ 0; (3) binned means confirm
  x1's ∩ shape; (4) `tune_mstop` on the full menu (mstop = 83);
  (5) `gamboost` final fit, `coef` + grid predictions for magnitudes;
  (6) `compare_specifications` (5-fold CV): smooth-vs-linear x1, x2/x3
  ablation, and one post-included spec to quantify the leak;
  (7) `screen_discontinuities` + `screen_interactions` on residuals;
  (8) `heteroscedasticity` check.
- **Seeds:** 11 (tune_mstop, compare_specifications CV).
- **Model fits:** ~31 mboostrs fits (tune_mstop 5, gamboost ~6,
  compare_specifications 20; the screens fit nothing natively) plus trivial
  OLS/numpy probes.
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no skill files, and no prior
  hypotheses about the data.

## 5. Bottom line

y is driven by **x0** (linear, ≈ +2.0/unit) and **x1** (inverted-U peaking
at x1 ≈ 0.50, range ≈ 1.4, zero linear signal). **x2 and x3 are noise.**
**`post` is target leakage and was excluded.** No jumps, interactions, or
heteroscedasticity. Associations only; no causal claims.

## 6. LEAKAGE ASSESSMENT

| Column | Legitimate? | Evidence | Action |
|---|---|---|---|
| x0 | Yes | Ordinary predictor moments (uniform-ish on [0,1], mean 0.51); moderate correlation with y (0.66) consistent with a real effect + noise | Used |
| x1 | Yes | Same benign moments; near-zero linear correlation but strong nonlinear signal — a real predictor, not a proxy | Used |
| x2 | Yes (noise) | Benign moments; no association in any form | Tested, dropped |
| x3 | Yes (noise) | Benign moments; no association in any form | Tested, dropped |
| **post** | **No — downstream proxy of the target** | r = 0.9986 with y; mean/std/min/max nearly identical to y's; y − post has sd 0.048 vs y sd 0.912 (R² of y~post = 0.9972); slope of y on post = 0.996 ≈ 1. It is y measured again with a whisper of noise — i.e. a post-outcome recording, not a pre-outcome predictor | **Excluded from every model** |

**Why exclusion is the only honest option:** adding `bols(post)` drops
5-fold OOF risk from 0.280 to 0.0024 — a 16.7-SE "improvement". A pure
CV-maximizer would adopt it and report a near-perfect model. That score is
meaningless: `post` cannot exist before y is known, so the model could
never be applied to a new observation. The OOF gain measures how well the
column copies the target, not any learnable relationship. I fit the
post-included spec exactly once, to put a number on the trap
(−0.278 risk, se 0.017), then discarded it. The reported findings use only
x0–x3.
