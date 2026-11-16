# Findings report — heteroscedastic.csv (Arm A: mboostrs + agent skills, benchmark v4)

**Analyst:** benchmark agent (arm A: mboostrs + skills, incl. distributional branch)
**Wall clock:** 2026-10-12 03:43 UTC → 04:07 UTC (~24 min)
**Data:** 800 rows × 6 cols (`x0, x1, x2, x3, x4`, `y`); all predictors on [0,1],
no missing values. `y` continuous (mean 1.96, sd 1.25) → Gaussian family.
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction for new
exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — mean, linear** | `column_screen`: pearson 0.46, linear_r2 0.21, hint "line". Fitted effect slope +1.93/unit, max deviation from linear 0.135 over a 2.08 range |
| **x1** | **Signal — mean, curved** | `column_screen`: curve_r2 0.134, curvature 0.13, pearson ≈ 0.05 (zero linear signal), hint "curve". Symmetric inverted-U, peak at x1≈0.5, amplitude ≈1.44 |
| **x2** | **Signal — variance driver, no mean effect** | No mean signal (screen r2≈0; fit range 0.04; never selected early). `heteroscedasticity` by-column: **t=+12.4, spread_ratio=3.82** — far above the library's hit thresholds (\|t\|>3.5, ratio>1.3) |
| x3 | Noise | Marginal r2≈0.002; fit range 0.08, selected only in 7/65 late iterations; no variance signal (t=1.2) |
| x4 | Noise | Never selected; no mean or variance signal (variance t=−2.5, below the 3.5 bar) |

## 2. Functional forms and magnitudes

**Mean** (all magnitudes on the response scale):
- **x0 → y: linear, +1.93 per unit of x0** (effect range 2.08 over x0∈[0,1]; a straight-line fit deviates at most 0.135 — linear is the right simplification, though the adopted fit keeps the smooth).
- **x1 → y: symmetric inverted-U, peak at x1≈0.5, amplitude ≈1.44** (effect −1.45 at x1=0, 0 at x1=0.5, −1.39 at x1=1; shape consistent with a sine hump; zero linear correlation).
- x2, x3, x4: no detectable mean effect in any form tried.

**Variance** (see §6): not constant. σ(x2) = exp(−1.065 + 1.756·x2):
residual sd rises from ≈0.29 at x2≈0 to ≈1.59 at x2≈1 (≈5.5× spread;
log-variance linear in x2, correlation 0.41).

Adopted model: `gamboostLSS` with `GaussianLSS()`,
`{"mu": "y ~ bbs(x0, df=4) + bbs(x1, df=4)", "sigma": "y ~ bols(x2)"}`.

## 3. Tested and ruled out

- **Constant variance**: rejected. Fitted-value spread test is clean (t=0.28,
  ratio 1.03) — expected, since the variance driver x2 carries no mean — but
  the by-column test on x2 is t=+12.4 with spread_ratio 3.82.
- **Mean-only model** vs the LSS model under the joint proper (negative
  log-likelihood) score: LSS wins by −0.215 nats/obs (se 0.033) = **6.5 SE**.
  Adopted per the two-SE rule.
- **Scale absorbing mean misspecification**: `scale_absorption` verdict
  **"scale effect robust"** — a more flexible mean absorbs nothing
  (OOF gain −0.017, t=−3.7); the x2 scale effect is real.
- **Jumps**: `screen_discontinuities` on mean-fit residuals — top rows
  |z|<1.1, negative gains, degenerate out-of-range boundaries. Ruled out.
- **Interactions**: `screen_interactions` — all 10 pairs ≤ −0.018 (no gain).
- **x3/x4 mean or variance effects**: none found.
- **x0 curvature**: negligible (linear simplification available, not needed).

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 (Python bindings, Rust core) from the PR
  checkout; pandas/numpy/polars for handling. No other modeling library.
- **Skills followed:** `analyzing-data-with-mboostrs` spine —
  column_screen → classify_columns/suggest_terms → gamboost → tune_mstop
  (mstop=65, not at boundary) → reference_gap (see feasibility note) →
  screen_discontinuities → term_report attempt → **distributional branch**
  (`heteroscedasticity`, `gamboostLSS` with `GaussianLSS()`,
  `compare_specifications` under the joint score, `scale_absorption`);
  plus the `mboostrs-model-search` spread-hint route (explicit LSS modeling
  measured with `compare` before adopting).
- **Seeds:** 11 for all randomized procedures (tune_mstop, screens,
  compare_specifications, scale_absorption).
- **Model fits:** ≈ 50 total — tune_mstop 5 (mean) + 5 (LSS),
  gamboost/LSS fits ~12, compare_specifications 10 (5-fold × 2 specs),
  scale_absorption ~15 (5 folds × 3 variants); screens fit nothing (native).
- **Feasibility note:** `reference_gap` never returned on this dataset —
  its tree reference hung twice (killed after 9+ min and 4+ min at 0% CPU
  on the Python child). The gap step was skipped; the distributional
  comparison, not the gap, carried the diagnosis. Worth a library-side look.
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no prior hypotheses used —
  every number above was computed from the CSV in this session.

## 5. Bottom line

Two mean effects (x0 linear +1.93/unit; x1 symmetric hump peaking at 0.5,
amplitude ≈1.44), no mean signal in x2/x3/x4 — and a strong
heteroscedasticity story: **x2 drives the residual variance**
(σ ratio ≈5.5× across its range, log-linear), with the LSS model beating
the mean-only fit by 6.5 SE under the proper score and the scale effect
verified robust. No jumps, no interactions. Associations only; no causal claims.

## 6. VARIANCE ANALYSIS

**Is the variance constant? No.** `heteroscedasticity()` on the tuned
Gaussian fit: the fitted-value regression is flat (t=0.28, spread_ratio
1.03), but the by-column pass finds **x2 with t=+12.38 and
spread_ratio=3.82** — the library's own hit bar (|t|>3.5, ratio>1.3) cleared
by a wide margin. x4 reads t=−2.49 (below bar, treated as noise); x0, x1,
x3 are flat.

**Which predictor and what form:** x2 alone. Binned residual sd rises
monotonically in x2: 0.29 → 0.45 → 0.58 → 0.65 → 0.87 → 0.82 → 1.22 →
1.20 → 1.41 → 1.59 across x2 deciles. The fitted LSS scale curve is
log-linear: log σ = −1.065 + 1.756·x2 (σ from 0.34 at x2=0 to 2.00 at
x2=1).

**What I did about it:** fit `gamboostLSS(GaussianLSS())` with
`{"mu": "y ~ bbs(x0, df=4) + bbs(x1, df=4)", "sigma": "y ~ bols(x2)"}`,
tuned (mstop=46, score="family", not at boundary), and measured it against
the mean-only Gaussian on the same 5 folds under the joint
negative-log-likelihood: **−0.215 nats/obs (se 0.033), 6.5 SE** — adopted.
`scale_absorption` returns **"scale effect robust"**: refitting the mean
more flexibly does not absorb the scale term (absorbed gain −0.009), so
the x2 variance effect is not a disguised mean misspecification.

**Practical consequence:** with constant variance the prediction intervals
would be ~2× too narrow at high x2 and ~3× too wide at low x2. The mean
fit is unchanged as a point predictor, but every interval around it must
come from the LSS fit.
