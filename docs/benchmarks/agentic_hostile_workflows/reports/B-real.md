# Findings report — real_train.csv (Arm B: sklearn + xgboost, benchmark v4)

**Analyst:** benchmark agent (arm B: sklearn/xgboost only)
**Wall clock:** 2026-10-12 03:43:10 UTC → 03:52:40 UTC (~9.5 min)
**Data:** 354 rows × 11 cols (`age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`, `y`).
Real data (diabetes progression), no planted truth, no missing values.
Features pre-standardized (mean ≈ 0, sd ≈ 0.047). `y`: mean 153.6, sd 77.4,
range 25–346.

## 1. Which predictors matter

Evidence: marginal correlations, permutation importance on the final model
(20 repeats), and coefficient effect ranges (coef × observed feature range).

| Predictor | Verdict | Evidence |
|---|---|---|
| **s5** | Strong signal | r = 0.57; perm. importance 0.55 (#2); effect range ≈ 227 |
| **bmi** | Strong signal | r = 0.57; perm. importance 0.16; effect range ≈ 124 |
| **bp** | Signal | r = 0.44; perm. importance 0.11; effect range ≈ 91 |
| **s1 / s2** | Joint signal (entangled) | r = 0.90 between them; coefs −921 / +605 (cancellation pattern — individually unstable, jointly strong); perm. importance 0.65 (#1) / 0.28 (#3) |
| **s3** | Weak/ambiguous | Marginal r = −0.40 but near-zero partial effect (coef +66, range ≈ 19; perm. importance ≈ 0) — its marginal signal is absorbed by correlated predictors |
| **s4** | Weak | r = 0.41 marginal; partial range ≈ 26; perm. importance ≈ 0.01 |
| **s6** | Weak | r = 0.35 marginal; partial range ≈ 13; perm. importance ≈ 0 |
| **sex** | Negligible | r ≈ 0; perm. importance 0.10 but tiny feature scale (range ±0.05) → effect range ≈ 33, likely noise picked up in-sample |
| **age** | Negligible | r = 0.17; partial range ≈ 10; perm. importance ≈ 0 |

Dominant story: `y` rises with s5, bmi, bp, and the s1/s2 pair; s3's strong
marginal negative correlation does not survive conditioning.

## 2. Functional forms and magnitudes

**Linear.** Every nonlinear or interaction-capable model lost to plain
regularized linear regression (see §3). Final model: RidgeCV,
alpha* = 0.001 (stable across CV seeds — essentially OLS with a whisper of
regularization), intercept 153.1. Direction: positive for s5, bmi, bp, s2;
the s1/s2 pair acts jointly (do not interpret the −921/+605 split
literally — it is an artifact of their 0.90 correlation). No kinks, jumps,
or interactions detected.

## 3. Tested and ruled out

- **Nonlinearity:** quadratic expansion + ridge → CV RMSE 56.25 vs 53.99
  linear. Ruled out.
- **Tree/GBM structure:** RandomForest 58.76, HistGradientBoosting 59.93,
  XGBoost 61.51 vs ridge 53.94. Interactions/nonlinearities, if present,
  are not exploitable at n = 354.
- **Outlier-driven fit:** HuberRegressor 54.05 vs ridge 53.99 — no
  meaningful outlier problem.
- **Regularization need:** alpha* = 0.001 stable; heavier shrinkage not
  supported by CV.
- **s3 as an independent negative driver:** marginal r = −0.40 collapses
  once s1/s2/s5/bmi/bp are in the model.

## 4. Methods, libraries, seeds, fit count

- **Libraries:** scikit-learn 1.9.1, xgboost 1.7.6, pandas, numpy. No
  mboostrs, no other modeling library.
- **Protocol:** repeated 5-fold CV (3 repeats, seed 11) comparing
  RidgeCV / LassoCV / ElasticNetCV / RandomForest / HistGradientBoosting /
  XGBoost; follow-up round (ridge alpha grids, poly2+ridge, Huber).
  Permutation importance (20 repeats, seed 11) on the final fit.
- **Model fits:** ≈ 155 (90 + 60 across the two CV rounds, plus final
  refit and screening fits); 20 permutation scoring passes.
- **Compliance:** analysis used only the CSV. No package-source inspection,
  no truth-key files, no prior hypotheses about this dataset used — every
  number above was computed from the CSV in this session. No attempt to
  reconstruct the sealed test split.

## 5. Bottom line

Honest linear model (RidgeCV, alpha = 0.001): `y` driven positively by s5,
bmi, bp, and the entangled s1/s2 pair; s3/s4/s6 weak-to-negligible;
age/sex negligible. CV RMSE ≈ 54.0, R² ≈ 0.50. Associations only.

## 6. OUT-OF-SAMPLE HONESTY

- **CV estimate:** repeated 5-fold (3×, seed 11) RMSE **53.99 ± 2.69**,
  R² 0.504 ± 0.063. This is my honest expectation for the sealed test set,
  assuming it comes from the same distribution via a random split.
- **Validation:** folds were row-random; one fixed seed (11) for the main
  comparison, alpha choice re-checked at seeds 22/33 (stable).
- **What could make held-out worse:**
  - Test-set sampling noise: with n ≈ 88 the test RMSE itself has
    sampling sd on the order of ±8 — a "bad" score may be luck, not model
    failure.
  - Mild selection optimism: I compared 6 model families on the same CV
    and reported the winner; the 53.99 is slightly optimistic for the
    *procedure*, though the follow-up round (tied results) suggests the
    optimism is small.
  - Non-random split: if the test set was carved non-randomly (e.g. by
    time or site), all bets are off — I cannot check this.
  - Mild heteroscedasticity: residual sd is 33 in the lowest fitted
    quartile vs ~55–59 elsewhere; the model does not capture this and a
    test set skewed toward high fitted values would score worse.
- **Limitations:** n = 354 with 10 predictors and heavy s1/s2
  collinearity — individual coefficients (especially s1/s2, sex) are
  unstable and should not be interpreted causally or in isolation. The
  model is a predictive summary, not a structural one.
