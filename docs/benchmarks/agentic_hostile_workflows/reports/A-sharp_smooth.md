# Findings report — sharp_smooth.csv (Arm A: mboostrs + agent skills, benchmark v4)

**Analyst:** benchmark agent (arm A: mboostrs + skills, incl. screen_discontinuities)
**Wall clock:** 2026-10-12 03:43:07 UTC → 04:03:27 UTC (~20 min)
**Data:** 800 rows × 5 cols (`x0, x1, x2, x3`, `y`); all numeric on [0,1]
(predictors), no missing values. `y` continuous (mean 2.22, sd 1.09) → Gaussian family.
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction for new
exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — steep, smooth, monotone increasing** | Marginal linear R²=0.710; `term_report` earns +0.888 (se 0.047), best shape `bbs`; effect range 2.77 response units (≈6.8× residual sd) |
| **x1** | **Signal — linear** | Marginal linear R²=0.137; `term_report` earns +0.170 (se 0.010), best shape `bols`; slope +1.41/unit |
| x2 | Noise | Marginal R²≈0.002; term_report −0.0008 (se 0.0009); all screens ≤ 0 |
| x3 | Noise | Marginal R²≈0.000; term_report −0.0021 (se 0.0013); all screens ≤ 0 |

## 2. Functional forms and magnitudes

Adopted model: `y ~ bbs(x0, df=4) + bols(x1)`, Gaussian, mstop=107 (tuned,
not at boundary). In-sample R² = 0.86, residual sd ≈ 0.41.

- **x0 → y: smooth monotone increase, range 2.77.** Genuinely curved:
  `bbs(x0,df=4)` beats `bols(x0)` by +0.0109 OOF risk (se 0.0041) = 2.7 SE.
  Steepest mid-range (max local slope +4.5 near x0≈0.50 vs mean slope +2.8),
  flattening toward both ends — an S-like shape, but see §6: it is
  continuous, not a jump. `bbs` beats `bbs(df=10)` (df=10 worse by ~10 SE)
  and beats `btree(x0,maxdepth=2)` (worse by 3.8 SE): the df=4 smooth already
  captures the shape; extra flexibility and tree structure both lose.
- **x1 → y: linear, +1.41 per unit of x1** (effect range 1.41 over x1∈[0,1];
  `bols` beats `bbs` by 2.0 SE on OOF risk).
- **x2, x3:** no detectable association in any form tried.

## 3. Tested and ruled out

- **A genuine jump in x0** — ruled out, see §6 (six independent lines of evidence).
- **Kink/hinge in x0** — the df=10 comparison covers localized extra
  flexibility and loses; fitted curve shows no curvature spike (max |2nd
  difference| = 0.046, negligible).
- **x2/x3 effects** (linear, smooth, tree, interactions): all ≈ 0.
- **Interactions:** `screen_interactions` on additive residuals — all 6 pairs
  ≤ −0.003. None.
- **Heteroscedasticity:** overall t = 1.42; by-column |t| < 1.3; spread ratios
  ≈ 1. Constant-variance Gaussian stands; no distributional branch.
- **Remaining jumps after the adopted model:** `screen_discontinuities` on
  response-scale residuals — all |z| < 2.1, all eval gains ≤ 0. Nothing left.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 (Python bindings, Rust core) from the PR
  checkout; pandas/numpy/polars/sklearn (KFold, spline/Ridge probes) for
  handling. No other modeling library.
- **Skills followed:** `analyzing-data-with-mboostrs` spine —
  column_screen → suggest_terms → gamboost → tune_mstop (mstop=107) →
  reference_gap (−0.0214, se 0.0051, material=False: formula beats the
  flexible tree reference, gap closed) → screen_discontinuities on raw y
  and on residuals → term_report → compare_specifications (two-SE rule) →
  heteroscedasticity; plus the new `example:disc-screening` worked example.
  `mboostrs-model-search` typed-candidate discipline: typed smooth kept
  over tree/step catch-alls on measured OOF evidence.
- **Seeds:** 11 for all randomized procedures.
- **Model fits:** ≈ 150 — tune_mstop 5, gamboost fits ~8, reference_gap ~20,
  compare_specifications 35 (3 rounds), term_report ~85 (17 specs × 5-fold);
  the screens fit nothing (native Rust core).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no prior hypotheses used —
  every number above was computed from the CSV in this session.

## 5. Bottom line

Two true components, two noise columns. x0 drives y through a steep but
smooth monotone curve (range 2.77, curved by 2.7-SE evidence against linear);
x1 is linear (+1.41/unit). x2 and x3 are noise. No jumps, no interactions,
no heteroscedasticity. Associations only; no causal claims.

## 6. JUMP VS STEEP-SMOOTH — verdict: steep continuous transition, NOT a jump

The trap in this case: x0's marginal `step_r2` is 0.606 — a single step
explains 61% of y's variance marginally, which screams "jump" to a naive
screen. Six independent checks say otherwise:

1. **Discontinuity screen typing (raw y):** `screen_discontinuities` top hits
   are degenerate out-of-range diagonal boundaries (e.g.
   `x1 > 21.38 + -39.76*x0`) with small jumps (+0.28–0.34), z ≈ 3–4, but
   `type_z` ≈ 0 — explicitly NOT typed as jumps. The only axis-aligned
   candidate (`x0 > 0.5235`: jump +0.299, z +3.54) carries eval gain +0.0048,
   negligible. Per the skill's step-5 rule, a flagged row with small
   `type_z` is a steep smooth crossing: prefer the smooth term.
2. **Screen on smooth-fit residuals:** all 25 rows |z| < 2.1, all eval gains
   ≤ 0 (most negative). A real jump would survive in the residuals; nothing does.
3. **Jump-indicator vs smooth, 5-fold CV:** `smooth(x0)+smooth(x1)` OOF RMSE
   0.4121 (se 0.0112) vs best single step indicator `I(x0>0.50)+smooth(x1)`
   0.5709 (se 0.0121) — the smooth wins by 0.159, ≈13 SE. Not close.
4. **Tree vs smooth:** `btree(x0,maxdepth=2)` (which can represent a jump)
   loses to `bbs(x0,df=4)` by +0.030 OOF risk (se 0.008) = 3.8 SE.
5. **Narrow-band checks:** band differences around candidate cuts are small
   (+0.14 to +0.38, se ≈ 0.1) and wander across cut locations — the signature
   of a steep gradient, not the large stable gap at one location a true
   discontinuity produces.
6. **Fitted-curve shape:** max local slope 4.5 vs mean slope 2.8 (1.6×, not
   an order of magnitude); no curvature spike anywhere. A df=10 spline —
   flexible enough to hug a near-jump — does *worse* than df=4.

The steep x0 transition is a genuine, curved, continuous rise. Claiming a
discontinuity here would be a false positive; the screen's `type_z` typing
is what keeps the agent from making it.
