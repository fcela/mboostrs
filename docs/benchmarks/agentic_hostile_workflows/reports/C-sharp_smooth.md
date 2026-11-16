# Findings report — sharp_smooth.csv (Arm C: mboostrs, no skill files)

**Analyst:** benchmark agent (arm C: mboostrs only; docstrings/`help()`, no skill files)
**Wall clock:** 2026-10-12 03:43:21 UTC → 04:03:26 UTC (~20 min)
**Data:** 800 rows × 5 cols (`x0, x1, x2, x3`, `y`); all predictors numeric on
[0,1], no missing values. `y` continuous (mean 2.22, sd 1.09) → Gaussian family.
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction for
new exchangeable rows.

All claims below are associations, not causal effects.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — steep smooth rise** | `column_screen` step_r2 = 0.606; `term_report` earns +0.859 (se 0.037) as `bbs(x0,df=6)`; marginal profile rises monotonically ~0.9 → ~3.6 |
| **x1** | **Signal — linear** | Pearson r = 0.37; monotone decile means 1.62 → 2.89; `term_report` earns +0.168 (se 0.010) as `bols(x1)`; `bols` beats all other shapes |
| x2 | Noise | Decile means flat ~2.1–2.5, r = −0.05; no CV gain in any form |
| x3 | Noise | Decile means flat ~2.1–2.4, r = 0.02; no CV gain in any form |

## 2. Functional forms and magnitudes

Final model: `y ~ bbs(x0,df=6) + bols(x1)`, Gaussian, mstop=500.
In-sample R² ≈ 0.86, residual sd ≈ 0.40, 5-fold OOF risk 0.171.

- **x0 → y: smooth, monotone increasing, steep in the middle.** Total swing
  ≈ 2.7 y-units over x0 ∈ [0,1]. Steepest local slope ≈ 5–6 y-units per
  x0-unit around x0 ≈ 0.45–0.65 (from 20-bin marginal means: e.g. 2.03 →
  2.31 across [0.44,0.54]). The fitted spline partial effect rises smoothly
  from −1.3 to +1.4 with no kink. `bbs` is the best shape; a jump indicator
  adds nothing (see §6).
- **x1 → y: linear, positive.** OLS slope ≈ +1.42 per unit of x1
  (effect range ≈ 1.4 over x1 ∈ [0,1]).
- **x2, x3:** no detectable association in any form tried (linear, smooth,
  step, interactions all ≈ 0).

## 3. Tested and ruled out

- **Genuine discontinuity in x0** — ruled out (see §6: screen clean on raw
  and residual passes, narrow-band steps ≤ 0.20 ± 0.14, jump indicator adds
  +0.00013 ± 0.00013 OOF risk, i.e. nothing).
- **Curvature in x1** — `term_report`: `bols` is the best shape; smooth
  alternatives earn nothing extra.
- **Interactions** — `screen_interactions` on final-model residuals: all 6
  pairs ≤ −0.0025 (noise floor); nothing left.
- **x2/x3 effects** (linear, smooth, step, interactions): all ≈ 0.
- **Heteroscedasticity** — t = 1.06 overall, no column driver (|t| < 1.34);
  constant-variance Gaussian stands; no distributional branch.
- **Missed structure** — `reference_gap` on the final spec: −0.022
  (se 0.004), `material=False`: the smooth formula *beats* the flexible
  tree reference. Nothing left on the table.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 only (Python bindings, Rust core), via
  `/home/hatch/workspace/mboostrs-venv/bin/python`; pandas/numpy/polars
  for handling. No skill files consulted; API learned from
  docstrings/`help()`.
- **Procedure:** (1) `column_screen` — marginal linear/curve/step shares;
  (2) `screen_discontinuities` on raw y; (3) smooth additive `gamboost`
  (`y ~ bbs(x0,df=4) + bols(x1) + bols(x2) + bols(x3)`) and
  `screen_discontinuities` on its response-scale residuals; (4) binned
  marginal profiles for all four predictors; (5) sorted fitted partial
  effect of x0; (6) narrow-band step scan (±0.02 around cuts 0.30–0.70,
  x1-adjusted); (7) `compare_specifications`: smooth vs smooth + best-cut
  jump indicator (cut 0.5027, 5-fold CV); (8) `term_report` shape
  comparisons; (9) `screen_interactions` on final residuals;
  (10) `heteroscedasticity`; (11) `reference_gap` vs tree reference.
- **Seeds:** 11 for screens/comparisons/`term_report`/`reference_gap`.
- **Model fits:** ≈ 75 mboostrs fits/screens — column_screen ×1,
  screen_discontinuities ×2, gamboost ×4, compare_specifications 10
  (2 specs × 5 folds), term_report ~45 (9 candidate specs × 5 folds),
  screen_interactions ×1, reference_gap ~10 (5-fold × formula + tree);
  the screens fit nothing (native Rust).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no skill files, and no prior
  hypotheses about the data.

## 5. Bottom line

Two true components, two noise columns. y rises **steeply but smoothly**
in x0 (swing ≈ 2.7, steepest ≈ 5–6/unit near x0 ≈ 0.5) and **linearly** in
x1 (slope ≈ +1.4/unit); x2 and x3 are noise. The steep x0 transition is a
steep continuous transition, **not** a genuine discontinuity (§6).
No interactions, no remaining jumps, no heteroscedasticity. Associations
only; no causal claims.

## 6. JUMP VS STEEP-SMOOTH — verdict: steep continuous transition, NOT a jump

Four independent lines of evidence, all pointing the same way:

1. **Discontinuity screen (raw y):** top candidates are weak — best
   axis-aligned hit `x0 > 0.5235` has jump +0.30 (se 0.084), z +3.54, gain
   +0.0048; diagonal candidates are degenerate lines with |z| ≤ 4.1 and
   gains ≤ 0.007. Nothing is typed as a jump (no type_z flag fires).
   For comparison, a genuine +2.0 diagonal jump in this n=800 regime
   scores z ≈ +19 on this screen.
2. **Discontinuity screen (smooth-model residuals):** all 25 candidates
   |z| < 2.1, all eval gains ≤ +0.0008 (mostly negative). After the smooth
   spline absorbs x0, nothing step-like remains.
3. **Narrow-band check:** within ±0.02 of cuts 0.30–0.70 (x1-adjusted),
   the below-vs-above step is at most +0.20 (se ≈ 0.12–0.19) — every cut
   within ~1.7 SE of zero, and the magnitudes match the local smooth
   slope (≈ 5/unit × 0.04 width ≈ 0.2), not a step. A genuine jump of any
   material size (≥ 0.5) would stand out at > 3 SE in these bands.
4. **Jump indicator vs smooth, head-to-head CV:** `bbs(x0,df=6)+bols(x1)`
   vs the same plus `I(x0 > 0.5027)` (best of 91 cuts): OOF risk change
   **+0.00013 (se 0.00013)** — the indicator adds literally nothing; the
   spline already captures the transition fully.

The steepness is real (local slope ≈ 5–6/unit vs residual sd ≈ 0.40, so a
0.05 move in x0 shifts the conditional mean by ~0.3, most of a residual
sd) — which is exactly why this case is a good trap for jump-hunting. But
every test that can separate "steep" from "discontinuous" says smooth.
