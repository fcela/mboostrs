# Findings report — real_train.csv (Arm A: mboostrs + agent skills, benchmark v4)

**Analyst:** benchmark agent (arm A: mboostrs + skills)
**Wall clock:** 2026-10-12 03:43 UTC → 04:08 UTC (~25 min)
**Data:** 354 rows × 11 cols (`age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`, `y`).
Real data (sklearn diabetes set — no planted truth). No missing values.
`y`: mean 153.6, sd 77.4, range 25–346, integer-valued. Predictors look
standardized (mean ≈ 0, sd ≈ 0.047); `sex` is binary (2 distinct values).
**Family:** Gaussian, declared explicitly — `y` is integer-valued so the
default autospec wanted Poisson; a rounded continuous measurement needs
Gaussian (first `reference_gap` attempt failed on Poisson overflow, which
is how this surfaced).
**Fold claim:** 5-fold row-random CV throughout; numbers measure prediction
for new exchangeable rows.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **bmi** | Signal — linear, largest | term_report earns +314.6 (se 113.5); slope +460/unit of standardized bmi (≈120 over its observed range) |
| **s5** | Signal — **smooth, S-shaped**, 2nd largest | earns +255.5 (se 94.0); `bbs(s5)` beats `bols(s5)` by 64.1 (se 29.9) and by 71.7 (se 25.0) in the head-to-head spec comparison — adopted |
| **sex** | Signal — linear (binary shift) | earns +202.4 (se 75.7) **despite marginal r ≈ 0** — the association appears only conditional on the other predictors; slope −323/unit (≈−31 across its 0/1 range) |
| **bp** | Signal — linear | earns +192.5 (se 91.8); slope +367/unit (≈+88 over range) |
| **s1** | Small signal — linear | earns +14.2 (se 2.4); slope −162/unit (≈−45 over range). Note the sign: marginally +0.20, conditionally negative — collinearity with the other serum measures |
| **s3** | Borderline — linear | earns +8.8 (se 12.7); slope −237/unit (≈−67 over range) |
| age, s2, s4, s6 | No measurable signal | earns +2.5, −3.8, −6.2, −11.2 (all within noise of zero); dropped from the final spec |

The s5 smooth (centered): flat ≈ −49 at low s5, steep rise through the
middle (−34 → +4 → +38), plateau ≈ +47 at high s5, slight dip at the very
top edge (likely edge noise). Total swing ≈ 98 — the second-largest effect
after bmi.

## 2. Functional forms and magnitudes

Final model: `y ~ bols(sex) + bols(bmi) + bols(bp) + bols(s1) + bols(s3) + bbs(s5)`,
Gaussian, mstop=216 (tuned, not at boundary), fit on all 354 rows.
In-sample RMSE 51.8. All magnitudes on the response scale; slopes are per
unit of the standardized predictors, so the range-effects above are the
interpretable numbers. Associations only; no causal claims. The s1/sex
marginal-vs-conditional sign differences reflect collinearity among the
predictors — individual slopes are unstable, the joint prediction is what
was validated.

## 3. Tested and ruled out

- **Missed structure:** `reference_gap` on the linear spec: −624.1 (se 205.7),
  `material=False`, zero candidates — the additive formula already beats the
  flexible tree reference. Nothing left on the table.
- **Interactions:** `screen_interactions` on linear-spec residuals — all
  eval gains ≤ 0. None.
- **Jumps:** `screen_discontinuities` on residuals — top |z| < 0.6, all gains
  negative. No discontinuities.
- **Curvature elsewhere:** term_report shape checks — `bbs` never beats
  `bols` for any other predictor (all gains ≤ 0 or within noise).
- **Dropped terms:** age, s2, s4, s6 — zero/negative earns; the reduced spec
  scored nominally better (−17.8, se 17.9) than the full one, so the
  simplification was adopted per the within-noise rule.
- **Poisson family:** rejected — wrong for a rounded continuous response
  (and numerically infeasible here).

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 (PR branch build) only; pandas/numpy/polars
  for handling. No other modeling library.
- **Skills followed:** `analyzing-data-with-mboostrs` spine —
  column_screen → draft linear spec → gamboost → tune_mstop →
  reference_gap → term_report (shapes + earns) → screen_interactions /
  screen_discontinuities on residuals → compare_specifications (two-SE
  rule) → heteroscedasticity check → final refit on all rows.
- **Seeds:** 11 throughout.
- **Model fits:** ≈ 260 — tune_mstop ×3 (~30), gamboost fits ~8,
  reference_gap ~15, term_report 190, compare_specifications 20;
  the screens fit nothing (native).
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no `mboostrs.datasets`, no truth-key files, no use of outside knowledge
  about this dataset — every number above was computed from the CSV in
  this session. The sealed test split was not touched or reconstructed.

## 5. OUT-OF-SAMPLE HONESTY

- **CV estimate:** 5-fold row-random CV on the 354 training rows, shared
  folds, Gaussian MSE: final spec OOF MSE **2864 → RMSE ≈ 53.5**. The
  competing specs on the same folds: full-linear 2953, linear+bbs(s5)
  2881, reduced-linear 2929.
- **How validated:** 5-fold random split, seed 11, `mboostrs.compare_specifications`
  with the two-SE adoption rule; mstop tuned separately per spec
  (216 for the final spec).
- **What could make held-out worse:** (1) mstop was tuned on the same 354
  rows the CV evaluates — mild selection optimism, maybe a few %;
  (2) row-random folds assume exchangeable rows — if the sealed test comes
  from a different time period or population, the estimate is void;
  (3) n=354 with 6 terms is comfortable, but the s5 smooth's top-edge dip
  is estimated from few points and could wobble out of sample.
- **Limitations:** (1) **Heteroscedasticity:** residual spread grows with
  the fitted value (t=3.28, spread ratio 1.67, strongest along bmi/bp).
  This does not change the MSE-optimal point forecast, but prediction
  intervals from the constant-variance model are miscalibrated (too
  narrow at high y). A distributional (LSS) branch would fix intervals,
  not point RMSE — out of scope for the graded metric.
  (2) `y` is integer-valued; modeled as continuous — rounding predictions
  is a trivial post-step I did not apply.
  (3) The sex and s1 effects are purely conditional (marginals ≈ 0 and
  +0.20); in a shifted population their contributions could differ from
  what the training collinearity implies.
  (4) One agent, one run — no replication.

## 6. Deliverables

- Fitted model (pickle): `.../agentic-benchmark-v4/reports/A-real-model.pkl`
- Predict spec: `.../agentic-benchmark-v4/reports/A-real-predict.txt`
  (`pickle.load` + `.predict(newdata)` on a DataFrame with the same 10
  feature columns in the same order; verified extra columns are ignored
  and the pickle round-trips exactly)
