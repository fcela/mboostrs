# Findings report — missingness.csv (Arm A: mboostrs + agent skills, benchmark v4)

**Analyst:** benchmark agent (arm A: mboostrs + skills)
**Wall clock:** 2026-10-12 03:43:02 UTC → 04:06:00 UTC (~23 min)
**Data:** 600 rows × 7 cols (`x0..x5`, `y`); all numeric on [0,1] (predictors),
no missing values in `y`. `y` continuous (mean 1.99, sd 0.92) → Gaussian family.
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction for
new exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — kinked/hinge** | `column_screen`: pearson 0.53, curve_r2 0.31, hint "curve"; `term_report` earns +0.233 (se 0.032); hinge scan peaks at kink ≈ 0.40 |
| **x2** | **Signal — inverted-U hump** | `column_screen`: curve_r2 0.27, pearson −0.03 (zero linear signal), hint "curve"; `term_report` earns +0.170 (se 0.029) |
| **x3** | **Signal — linear** | `column_screen`: pearson 0.33, hint "line"; `term_report` earns +0.094 (se 0.007); bols slope +0.98/unit |
| x1 | Noise | screen "none" (linear_r2 0.0007); never selected by boosting; term_report −0.0003 (se 0.0002) |
| x4 | Noise | screen "none"; term_report +0.0006 (se 0.0014); boosting gave it 8/181 late updates, range 0.08 |
| x5 | Noise | screen "none"; term_report −0.00004 (se 0.0002); 2/181 updates, range 0.01 |

## 2. Functional forms and magnitudes

Adopted model: `y ~ bbs(x0) + bbs(x2) + bols(x3)`, Gaussian, mstop=181
(tuned, not at boundary), `na_action="impute"`. All magnitudes on the
response scale.

- **x0 → y: hinge (kink), flat below ≈ 0.40, rising ≈ +2.4/unit above.**
  The smooth `bbs(x0)` fit is flat (−0.4 to −0.5) up to x0 ≈ 0.3–0.4, then
  rises monotonically to +1.07 at x0 = 1. A hinge scan (OLS, controlling
  for quadratic/cubic x2 and linear x3 on complete cases) peaks at kink
  c = 0.40 (R² profile reasonably peaked: 0.675 at 0.40 vs 0.669 at 0.30,
  0.673 at 0.45); slope above the kink +2.41/unit; slope below the kink
  −0.08 (≈ 0). Kink location uncertainty ≈ ±0.05.
- **x2 → y: symmetric inverted-U hump, peak at x2 ≈ 0.50, range ≈ 1.4.**
  Fitted effect: −0.90 at x2 = 0, +0.46 at x2 = 0.50, −0.92 at x2 = 1.0
  (imputed fit); complete-case fit gives −0.95 / +0.57 / −0.90 (range 1.5).
  Zero linear association (pearson −0.034) — the effect is purely curved,
  sine-like.
- **x3 → y: linear, +0.98 per unit of x3** (bols coefficient 0.979;
  effect range 0.98 over x3 ∈ [0,1]).
- **x1, x4, x5:** no detectable association in any form tried.

Final fit: in-sample R² ≈ 0.68, residual sd ≈ 0.52. `reference_gap` on the
final spec: −0.034 (se 0.012), `material=False` — the formula beats the
flexible tree reference; nothing left on the table.

## 3. Tested and ruled out

- **Jumps/discontinuities:** `screen_discontinuities` on response-scale
  residuals — top hit z = −0.39, `typed_jump=False`, gain negative. Jump
  ruled out.
- **Interactions:** `screen_interactions` on residuals — all 15 pairs ≤
  +0.005 (mostly negative). None.
- **x1/x4/x5 effects** (linear, smooth): term_report ≈ 0; excluded.
- **Heteroscedasticity:** t = 1.57, spread_ratio = 1.20; no column driver.
  Constant-variance Gaussian stands; no distributional branch.
- **x0 smooth-vs-hinge:** the hinge interpretation is read off the smooth
  fit plus the OLS kink scan (see §2); the smooth fit alone already
  captures it (reference gap closed).
- **Informative missingness:** the `x2_missing` / `x0_missing` indicators
  were selected, but their coefficients (−0.487 / +0.121) equal minus the
  fitted effect at each column's median fill point (+0.46 / −0.13), so the
  net prediction for a missing row ≈ the mean effect — exactly the
  median-fill bias correction under MCAR, not evidence that missingness
  itself predicts y. See §6.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 (Python bindings, Rust core) from the PR
  checkout; pandas/numpy for handling. No other modeling library.
- **Skills followed:** `analyzing-data-with-mboostrs` spine —
  column_screen → classify_columns/suggest_terms → gamboost →
  tune_mstop → reference_gap → screen_discontinuities →
  screen_interactions → term_report → heteroscedasticity; plus the
  `missing-data` topic (`na_action` policies) and `mboostrs-model-search`
  typed-candidate discipline.
- **Seeds:** 11 for all randomized procedures.
- **Model fits:** ≈ 60 mboostrs fits — tune_mstop 5, gamboost ~6,
  reference_gap ~20 (5-fold × formula + tree), term_report 25 candidate
  specs; screens/partial_effects/heteroscedasticity fit nothing (native).
  Plus cheap NumPy OLS probes (hinge scan, MCAR checks — not model fits).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key or generator files, no prior
  hypotheses — every number above was computed from the CSV in this session.

## 5. Bottom line

Three true components, three noise columns. x0 has a hinge at ≈ 0.40
(flat below, +2.4/unit above); x2 is a symmetric inverted-U peaking at
0.50 (range ≈ 1.4, no linear signal); x3 is linear (+0.98/unit). x1, x4,
x5 are noise. No jumps, no interactions, no heteroscedasticity, no
informative missingness. Associations only; no causal claims.

## 6. HOW YOU HANDLED THE MISSING VALUES

**What:** x0 had 65 NaNs (10.8%), x2 had 62 NaNs (10.3%); x1, x3, x4, x5,
y complete. I used `na_action="impute"` on every fit: numeric predictors
filled with their training median, plus a `bols(<x>_missing,
intercept=False)` indicator term per incomplete column, so the fit
estimates how missing-rows differ from median-filled rows. Per the
`missing-data` topic, both steps are part of the predictor recipe.

**Why:** it keeps all 600 rows (complete-case analysis would drop
~20%), and the indicator doubles as a built-in informative-missingness
test: if missingness predicted y beyond the fill, the indicator would
carry the extra signal.

**Assumption:** missingness is at most weakly related to the data
(MCAR-ish). Checked: y mean is 2.13 when x0 is missing vs 1.97 when
observed (≈ 1.4 SE, n=65 — sampling noise); 2.06 vs 1.98 for x2; other
columns' means barely shift between missing/observed rows. No strong
evidence against MCAR.

**What could go wrong:** median imputation biases nonlinear effects at
the fill point — and that is exactly what happened: with x2's hump
peaking near its median, median-filled rows got the peak effect (+0.46)
while their true (unobserved) x2 values spread over [0,1] average to a
much lower effect, so the `x2_missing` indicator (−0.487) was selected
to correct it. I verified this is a fill-bias correction, not real
signal: the indicator coefficient ≈ −(effect at median), netting missing
rows to ≈ the mean effect, and a complete-case refit (`na_action="omit"`)
reproduces the same x0/x2/x3 shapes (x2 range 1.5 vs 1.4 — the imputed
fit is slightly attenuated at the hump, as expected when 62 noisy rows
are pinned at the median). If missingness had been strongly informative
(MNAR), median imputation would have biased the effect estimates and the
complete-case sensitivity check would have diverged — it did not.
