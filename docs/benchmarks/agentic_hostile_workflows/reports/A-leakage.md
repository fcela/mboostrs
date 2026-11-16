# Findings report — leakage.csv (Arm A: mboostrs + agent skills, benchmark v4)

**Analyst:** benchmark agent (arm A: mboostrs + skills)
**Wall clock:** 2026-10-12 03:43:11 UTC → 04:01:59 UTC (~19 min)
**Data:** 600 rows × 6 cols (`x0, x1, x2, x3, post`, `y`); all numeric, no
missing values. `y` continuous (mean 1.97, sd 0.91) → Gaussian family.
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction
for new exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — linear** | Marginal r=0.664; `term_report` earns +0.357 (se 0.010) = 36 SE; best shape `bols` (linear beats smooth by +0.0036, se 0.001) |
| **x1** | **Signal — inverted-U curve** | Marginal r≈0 (zero linear signal); `column_screen` hint "curve" (curvature 0.244); `term_report` earns +0.196 (se 0.025) = 7.8 SE; best shape `bbs` |
| **post** | **LEAK — excluded** | post = −0.001 + 1.0009·y + noise, R²=0.9972, residual sd 0.048; residual uncorrelated with all x's (|r|<0.03). A near-deterministic function of the target; see §6 |
| x2 | Noise | Marginal r=−0.05; term_report −0.0016 (se 0.0013); residual corr −0.044 |
| x3 | Noise | Marginal r=+0.02; term_report −0.0010 (se 0.0018); residual corr +0.065 |

## 2. Functional forms and magnitudes

Honest model: `y ~ bols(x0, df=1) + bbs(x1, df=4)`, Gaussian, mstop=77
(tuned, not at boundary). In-sample R² = 0.67, OOF family risk 0.28.

- **x0 → y: linear, +2.10 per unit of x0** (effect range 2.10 over
  x0∈[0,1]). Quadratic adds nothing (resid sd 0.6820 vs 0.6817).
- **x1 → y: symmetric inverted-U, peak at x1 ≈ 0.50, swing ≈ 1.5**
  (quadratic fit: −6.03·x1² + 6.01·x1; binned means 1.57/2.11/2.48/2.24/1.44
  across quintiles). Zero linear correlation — the effect is invisible to
  any linear screen.
- **x2, x3:** no detectable association in any form tried.
- **Joint:** x0-linear + x1-quadratic explains R² = 0.669; residual sd 0.52.

## 3. Tested and ruled out

- **Jumps:** `screen_discontinuities` on honest residuals — top rows are
  degenerate (e.g. `x2 > 5.125 − 6.002·x1`, outside the data range),
  |z| ≤ 2.9, all `type_z` small (|type_z| ≤ 1.1, none typed as jump).
  Jump ruled out.
- **Interactions:** `screen_interactions` showed small positives
  (x0×x1 +0.105); `compare_specifications` of `+ btree(x0,x1,maxdepth=2)`
  vs base: OOF risk **worse** by +0.084 (se 0.020) = 4.1 SE worse.
  Interaction ruled out. (x0×x1 residual corr was 0.008.)
- **Curvature in x0:** rejected (linear wins).
- **Heteroscedasticity:** |resid| below/above median ratios 0.91–1.05
  across all columns; no driver. Constant-variance Gaussian stands.
- **Anything left:** `reference_gap` = −0.036 (se 0.010), material=False —
  the honest formula *beats* the flexible tree reference; nothing left on
  the table.
- **post as a predictor:** excluded (see §6).

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 (Python bindings, Rust core) from the PR
  checkout; pandas/numpy for handling. No other modeling library.
- **Skills followed:** `analyzing-data-with-mboostrs` spine —
  column_screen → suggest_terms → gamboost → tune_mstop → reference_gap →
  screen_discontinuities → term_report → compare_specifications; plus
  `mboostrs-model-search` typed-candidate discipline (typed smooth/linear
  terms preferred; btree catch-all tested and rejected on measured
  evidence). Worked example `example:disc-screening` consulted for the
  discontinuity screen.
- **Seeds:** 11 for all randomized procedures.
- **Model fits:** ≈ 150 — tune_mstop ~20, gamboost ~4, term_report ~100
  (17–21 candidate specs × 5 folds; the post-included run counted
  separately), reference_gap ~20, compare_specifications 10; the screens
  fit nothing (native).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no prior hypotheses used —
  every number above was computed from the CSV in this session.

## 5. Bottom line

Two true effects, two noise columns, one leak. y rises linearly in x0
(+2.10/unit) with a symmetric inverted-U in x1 (peak ≈ 0.50, swing ≈ 1.5);
x2 and x3 are noise. `post` is a downstream near-copy of y (R²=0.997) and
was excluded — the honest model reaches in-sample R² = 0.67 with nothing
left for a flexible reference to gain. Associations only; no causal claims.

## 6. LEAKAGE ASSESSMENT

| Column | Legitimate predictor? | Evidence |
|---|---|---|
| x0 | Yes | Ordinary predictor range [0,1], plausible marginal and joint effects |
| x1 | Yes | Same; zero marginal linear signal but strong conditional curve |
| x2 | Yes (but noise) | Same; no association found |
| x3 | Yes (but noise) | Same; no association found |
| **post** | **No — downstream proxy of the target** | (a) `post = −0.001 + 1.0009·y + ε`, R² = 0.9972, sd(ε) = 0.048 — post is y plus independent noise; (b) ε is uncorrelated with every x (|r| < 0.03), exactly the signature of a noisy copy of y rather than a cause of y (a genuine cause would share structure with the x's that drive y); (c) the name marks it as measured after the outcome; (d) `y ~ post` alone gives R² = 0.997 — no real prediction task has a legitimate predictor that good |

**What I did:** excluded `post` from every honest fit; all discoveries
above come from x0–x3 only. I additionally fit the dishonest spec
(`y ~ bols(post) + smooth(x0..x3)`) purely as a demonstration: `post`
earned +0.241 (se 0.014) while the *true* effects x0 and x1 went
**negative** (−0.038 and −0.026) — the leak doesn't just inflate the
score, it erases the real structure from the model. That is the cost of
using it.

**Why exclusion is right:** a predictor must be knowable at prediction
time. A variable that is a near-deterministic function of the target
cannot be available before the target is known — any model using it is
scoring itself against information it would not have in deployment. The
R² = 0.997 it buys is not performance; it is circularity.

**What could go wrong with my call:** the one alternative reading is that
y is *derived from* post (post measured first, y computed from it). That
reading makes the whole discovery task vacuous (nothing left to learn
from x0/x1) and contradicts the column name. If the data dictionary ever
showed post was recorded before y, the assessment would flip — but on the
evidence in the CSV, exclusion is the honest choice.
