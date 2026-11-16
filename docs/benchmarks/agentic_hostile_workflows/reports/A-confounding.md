# Findings report — confounding.csv (Arm A: mboostrs + agent skills, benchmark v4)

**Analyst:** benchmark agent (arm A: mboostrs + skills)
**Wall clock:** 2026-10-12 03:42:54 UTC → 2026-10-12 03:59:08 UTC (~16 min)
**Data:** 600 rows × 5 cols (`x0, g, x1, x2`, `y`); `g` is a 0/1 group indicator
(302 ones, 298 zeros); all predictors on [0,1], no missing values.
`y` continuous (mean −0.25, sd 0.85) → Gaussian family.
**Fold claim:** 5-fold row-random CV for all comparisons; numbers measure
prediction for new exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **g** | **Signal — group intercept shift** | `column_screen` step_r2 = 0.671 (dominant marginal signal); joint-model `bols(g)` effect −2.04 at fixed x0; group means of y: +0.449 (g=0) vs −0.947 (g=1) |
| **x0** | **Signal — positive linear, CONFOUNDED by g** | Marginally *negative* (r = −0.423, slope −1.35); within groups *positive* (+1.58/unit in both groups). The marginal sign flips after conditioning — textbook confounding/Simpson reversal |
| x1 | Noise | \|r\| = 0.075 marginal; within-group slopes −0.13/+0.16 (≈0); smooth range 0.17 vs 1.36 for x0 |
| x2 | Noise | \|r\| = 0.082 marginal; within-group slopes +0.07/+0.00; smooth range 0.08 |

## 2. Functional forms and magnitudes

Adopted model: `y ~ bols(x0) + bols(g)`, Gaussian. All magnitudes on the
response scale (residual sd ≈ 0.40).

- **x0 → y: linear, +1.58 per unit of x0 within each group** (mboostrs
  `gamboost`, prediction contrast 0→1 at fixed g; identical in g=0 and g=1).
  Binned quintile means rise monotonically within groups; no curvature
  detected (`bbs(x0)` adds nothing over `bols(x0)` in the joint fit).
- **g → y: intercept shift, −2.04** (g=1 vs g=0 at fixed x0; mboostrs
  prediction contrast). The raw group-mean gap (−1.40) understates it
  because x0 is higher in g=1 (0.701 vs 0.294), which offsets the shift.
- **x1, x2:** no detectable association in any form tried (linear, smooth,
  interactions).

The confounding mechanism, measured: x0's distribution is shifted by group
(x0 mean 0.294 in g=0 vs 0.701 in g=1) while g=1 has lower y. Marginally,
x0 looks harmful (slope −1.35); conditioned on g it is beneficial (+1.58).
Any analysis that omits g gets the *sign* of x0 wrong.

## 3. Tested and ruled out

- **x0 without g** (`y ~ bols(x0)`): slope −1.35 — the confounded answer;
  adding g flips x0 to +1.58. Omitted-variable sign reversal confirmed.
- **x0×g interaction** (slope differs by group): within-group slopes
  +1.64 vs +1.54, difference ≈ 0.10 ± ~0.19 — not distinguishable; single
  x0 slope stands.
- **Curvature in x0**: smooth vs linear not distinguishable; linear kept.
- **Jumps/regimes**: `screen_discontinuities` on tuned-fit residuals —
  25 rows, all |z| < 2.2, all eval gains negative. Jump ruled out.
  (Note: the screen errors on the binary `g` column itself — "smooth
  background is singular on the fit half" — so it was run on x0/x1/x2;
  g is already a step by construction.)
- **Interactions**: `screen_interactions` on residuals — all pairs ≤ −0.004.
- **x1/x2 effects** (linear, smooth, interactions): all ≈ 0.
- **Heteroscedasticity**: t = −0.49, spread_ratio 0.94; no column driver
  (|t| < 1.2). Constant-variance Gaussian stands.
- **`term_report`** (finished after the run budget; confirms the above):
  bols(g) earns +0.391 (se 0.038), bbs(x0, df=4) +0.077 (se 0.0098),
  bbs(x1) +0.0002 (se 0.0017) ≈ 0, bbs(x2) −0.0004 ≈ 0. No term wants
  another shape.
- **`reference_gap`** (finished after the run budget): gap −0.0117
  (se 0.0034), material=False — the formula beats the flexible tree
  reference; nothing missed.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs (Python bindings, Rust core) from the PR checkout;
  pandas/numpy for handling. No other modeling library.
- **Skills followed:** `analyzing-data-with-mboostrs` spine —
  column_screen → classify_columns/suggest_terms → gamboost → tune_mstop
  (mstop 435, not at boundary) → reference_gap → screen_discontinuities →
  term_report → heteroscedasticity; `mboostrs-model-search` typed-term
  discipline (linear kept over smooth where indistinguishable).
- **Seeds:** 11 for all randomized procedures.
- **Model fits:** tune_mstop runs ~6, gamboost fits ~10, screens native
  (no fits), term_report ~70 (14 candidate specs, 5-fold), reference_gap
  ~10. All skill-workflow diagnostics completed.
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no prior hypotheses — every
  number above was computed from the CSV in this session.

## 5. Bottom line

Two true components, two noise columns. y rises linearly in x0 (+1.58/unit
within groups) and drops by ~2.0 in group g=1 — but x0 is higher in the
low-y group, so its marginal association is *negative* (−1.35): omitting g
reverses the sign of x0. x1 and x2 are noise. No interactions, no jumps,
no heteroscedasticity. Associations only; no causal claims.

## 6. ASSOCIATION VS CAUSATION

| Effect | Marginal association | After conditioning | Causal language supported |
|---|---|---|---|
| x0 → y | −1.35/unit (r = −0.423): higher x0, lower y | **+1.58/unit** within each group: higher x0, higher y | None from this data. The sign reversal shows the marginal association is fully explained by confounding through g; the conditional +1.58 is the association holding group fixed. Whether x0 *causes* y cannot be determined without a causal design — g could itself be a proxy for an unmeasured common cause. |
| g → y | −1.40 (raw mean gap) | −2.04 at fixed x0 | None. g is a group label, not a manipulable treatment here; the −2.04 is a conditional mean difference, not an effect of intervening on g. |
| x1 → y | −0.21/unit (r = −0.075) | ≈ 0 within groups | None — no association after conditioning; noise. |
| x2 → y | +0.24/unit (r = +0.082) | ≈ 0 within groups | None — no association after conditioning; noise. |

The causal trap this case sets: reporting the marginal x0 slope (−1.35)
as "the effect of x0" gets the sign wrong. The conditional estimate
(+1.58) is the honest associational summary, and even it earns no causal
interpretation — a fit reports associations; a causal question needs a
causal design (per the skill's `causal` topic, not invoked here).
