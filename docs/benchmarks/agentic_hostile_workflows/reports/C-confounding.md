# Findings report — confounding.csv (Arm C: mboostrs, no skill files)

**Analyst:** benchmark agent (arm C: mboostrs only; docstrings/`help()`, no skill files)
**Wall clock:** 2026-10-12 03:43:12 UTC → 04:02:33 UTC (~19 min)
**Data:** 600 rows × 5 cols (`x0, g, x1, x2`, `y`); `g` is a 0/1 group indicator
(302 ones, 298 zeros); all predictors on [0,1]; no missing values.
`y` continuous (mean −0.25, sd 0.85) → Gaussian family.
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction for
new exchangeable rows. All associations below; no causal claims.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — positive linear, +1.58/unit conditional on g** | Joint `glmboost("y ~ bols(x0)+bols(g)")`: coef +1.58. Within-group slopes +1.63 (g=0), +1.53 (g=1). `term_report` earns +0.074 (se 0.010) ≈ 7 SE; adding x0 to g cuts OOF risk 0.2427 → 0.1684 |
| **g** | **Signal — group level shift, −2.04 for g=1 conditional on x0** | Joint coef −2.04. Within-x0-quartile mean gaps (g1−g0): −1.96 to −1.98. `term_report` earns +0.432 (se 0.033) ≈ 13 SE; adding g to x0 cuts OOF risk 0.6016 → 0.1684 ≈ 12 SE |
| x1 | Noise | Marginal r=−0.07; partial r(x1,y|x0,g)=0.007; term_report −0.0003 (se 0.0001) |
| x2 | Noise | Marginal r=+0.08; partial r(x2,y|x0,g)=0.015; term_report −0.0002 (se 0.00008) |

## 2. Functional forms and magnitudes

Final model: `y ~ bols(x0) + bols(g)`, Gaussian. In-sample R² = 0.77,
residual sd = 0.41. OOF risk 0.1684.

- **x0 → y: linear, positive, +1.58 per unit of x0** (effect range 1.58 over
  x0∈[0,1]). Linearity checked two ways: `term_report` shape menu gives
  `bbs(x0)` only +0.00017 (se 0.0009) over `bols(x0)`; residuals are flat
  across x0 quintiles within each group (all |mean| ≤ 0.07).
- **g → y: level shift, −2.04 for g=1 vs g=0 at fixed x0** (binary predictor,
  so no shape question). This is the single largest effect (≈5× residual sd).
- **No x0×g interaction**: slopes +1.63 vs +1.53 across groups; interaction
  term t = −0.51, n.s. The x0 effect is the same in both groups.
- **x1, x2:** no detectable effect in any form tried.

## 3. The confounding pattern (marginal vs conditional)

x0 and g are strongly correlated (r = 0.766; mean x0 is 0.29 in g=0 vs 0.70
in g=1), and this produces a **Simpson reversal** on x0:

| Estimate | x0 coefficient | g coefficient |
|---|---|---|
| Marginal (`y ~ bols(x0)` alone) | **−1.352** | — |
| Marginal (`y ~ bols(g)` alone) | — | −1.395 |
| Joint (`y ~ bols(x0) + bols(g)`) | **+1.58** | −2.04 |

Marginally, x0 looks *negatively* related to y (−1.35) because high-x0 rows
are mostly g=1 rows, and g=1 rows have much lower y. Within either group,
x0's association is positive (+1.6). Conversely, g's marginal gap (−1.40)
*understates* its conditional gap (−2.04), because part of the raw group
difference is absorbed by the x0 imbalance working in the opposite
direction. The nonparametric cuts agree: within-x0-quartile g gaps are
−1.96/−1.98, and within-group x0 slopes are +1.63/+1.53.

## 4. Tested and ruled out

- **x1, x2 effects** (linear, smooth, interactions): all ≈ 0, in and out of
  the joint model.
- **x0 curvature**: rejected (shape menu + binned residuals, §2).
- **x0×g interaction**: rejected (t = −0.51; near-identical within-group slopes).
- **Jumps**: `screen_discontinuities` on final-model residuals — 0 of 25 rows
  typed as jumps, max |z| = 2.47 (noise range over 25 candidates). Note: the
  screen errors on a binary column in the background design ("smooth
  background is singular on the fit half"); reran on the continuous columns
  only, which is the supported use.
- **Leftover interactions**: `screen_interactions` on residuals — all 6 pairs
  ≤ 0 (noise floor).
- **Heteroscedasticity**: t = −0.41 overall; no column driver (|t| ≤ 1.8).
  Constant-variance Gaussian stands.
- **Anything left**: `reference_gap` on the final spec = −0.0124 (se 0.0028),
  `material=False` — the formula beats the flexible tree reference.

## 5. Methods, libraries, seeds, fit count

- **Library:** mboostrs only (Python bindings, Rust core), via
  `/home/hatch/workspace/mboostrs-venv/bin/python`; pandas/numpy for handling
  and descriptives. No skill files consulted; API learned from
  docstrings/`help()`.
- **Procedure:** (1) descriptives + `column_screen` (g step_r2=0.67, x0
  step_r2=0.23 "curve" hint — the hint was the confounding, not curvature);
  (2) predictor inter-correlations → spotted x0–g r=0.77; (3) within-group
  slopes and within-x0-band group gaps (nonparametric); (4) `glmboost`
  marginal vs joint coefficients → Simpson reversal; (5) `term_report`
  (shapes + earnings); (6) `compare_specifications` 5-spec shared-fold CV;
  (7) interaction / discontinuity / heteroscedasticity / reference-gap screens.
- **Seeds:** 11 for screens, term_report, compare_specifications,
  reference_gap.
- **Model fits:** ≈ 110 mboostrs fits — column_screen 1, glmboost ~6,
  term_report ~70 (14 specs × 5 folds), compare_specifications 25,
  reference_gap ~8; screens fit nothing (native).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no skill files, no prior
  hypotheses — every number above was computed from the CSV in this session.

## 6. ASSOCIATION VS CAUSATION

| Effect | Marginal association | After conditioning | Causal reading supported |
|---|---|---|---|
| x0 → y | −1.35/unit (negative) | +1.58/unit (positive) | **None.** Sign flips with conditioning; the data cannot say whether x0 causes y, g causes both, or an unmeasured variable drives all three. |
| g → y | −1.40 gap | −2.04 gap | **None.** g is an unlabeled 0/1 indicator; "group membership" is not a manipulable cause without knowing what g encodes. The gap is a predictive fact, not an effect estimate. |
| x1, x2 → y | ≈ 0 | ≈ 0 | None — no association to explain. |

The reversal is a warning about vocabulary, not a causal discovery: the
conditional associations (+1.58, −2.04) are the right *predictive*
description, but calling either "the effect of" its variable would require
assumptions (no unmeasured confounding, correct temporal order, g
manipulable) that this dataset cannot verify. Everything above is
association.

## 7. Bottom line

Two true predictors, two noise columns. **x0 raises y linearly (+1.58/unit)
and g=1 lowers y by −2.04 at fixed x0** — but marginally x0 looks *negatively*
related to y (−1.35) because x0 and g are correlated at r=0.77: a textbook
Simpson reversal. x1 and x2 are noise. No curvature, no interaction, no
jumps, no heteroscedasticity; the additive linear spec leaves nothing on the
table (reference gap closed). Associations only; no causal claims.
