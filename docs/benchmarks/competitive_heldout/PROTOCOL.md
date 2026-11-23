# Competitive held-out study: protocol

- Scorecard: `mboostrs-scorecard/2.1`, `docs/scorecard/scorecard.md` SHA-256
  `12d0ea339b4894ca3821ba9a0a3394c89572f962d022420c36d485ee3aa3daec`.
- Mode: fully automated (each library is called once per task with no human
  or agent choice in between).
- Written 2026-11-21 before any run. Nothing below changes after results are
  read; deviations are recorded in DEVIATIONS.md.

## Libraries and calls

Each library gets its idiomatic default call. No library is tuned, and the
tuning effort for every library is zero user-set hyperparameters (fairness
rule 2). All run on one thread.

| Arm | Call |
|---|---|
| mboostrs | `mb.fit(train, "y", seed=s)` (budget `"standard"`), working tree; prediction `predict(type="response")`, or `model.predict(type="mean")` for distributional models |
| CatBoost | `CatBoostRegressor` / `CatBoostClassifier` defaults, `cat_features` = string columns, `thread_count=1`, `random_seed=s` |
| LightGBM | `LGBMRegressor` / `LGBMClassifier` defaults, string columns as pandas `category`, `n_jobs=1`, `random_state=s` |
| XGBoost | `XGBRegressor` / `XGBClassifier` defaults with `enable_categorical=True`, `tree_method="hist"`, `n_jobs=1`, `random_state=s` |
| EBM | `ExplainableBoostingRegressor` / `Classifier` defaults, `n_jobs=1`, `random_state=s` |
| Reference | scikit-learn pipeline: numeric median imputation with missing indicators and standardization, string one-hot with `handle_unknown="ignore"` and most-frequent imputation; `RidgeCV` / `LogisticRegression(max_iter=2000)` |

The reference is the prespecified simple baseline required by D11. It is not
an oracle.

## Tasks

**R: 95 real datasets** from the held-out study (`docs/benchmarks/fit_heldout`
manifests: 36 OpenML, 56 UCI, 3 `mboostrs.datasets`), seeds 5000, 5001,
5002. Test-use history: these datasets were used for four earlier studies of
`mb.fit` only (seeds 1000–4002), during which `mb.fit` was changed twice.
mboostrs therefore had development exposure to these tables and the
competitors had none. The seeds are new.

**S: planted-truth tasks**, 12 predictors uniform on (−1, 1) unless stated,
noise N(0, 0.5²), five independent draws each (generator seeds 0–4), 2,000
training rows and 2,000 test rows:

| Task | Response | True mean structure |
|---|---|---|
| S1 smooth | 2 sin(2x0) + 2(x1² − 1/3) + x2 | x0, x1 nonlinear; x2 linear |
| S2 interaction | 3 sign(x0 x1) + x2 | x0, x1 interaction only; x2 linear |
| S3 scale | x0 + sin(3x1), noise sd 0.25 + 1.5 u with u = (x2 + 1)/2 | x0 linear, x1 nonlinear; x2 scale only |
| S4 null | N(0, 1) | none |
| S5 small-n | as S1 with 200 training rows | as S1 |
| S6 proxies | as S1, with x3 = x0 + N(0, 0.05²) and x4 = x1 + N(0, 0.05²) | as S1; x3, x4 are proxies |
| S7 binary | Bernoulli(logit⁻¹(1.5 sin(2x0) + x1 − 0.5)) | x0 nonlinear, x1 linear |

## Measurements by dimension

**D11 predictive validity** (R and S1–S7): test MSE (regression) or log loss
(binary). Per task: excess over the reference `L_lib / L_ref − 1` and gap to
the best complete arm (an oracle comparison, labelled as such). Paired log
ratios mboostrs/competitor per task with a two-sided Wilcoxon test and a
bootstrap interval of the mean over tasks.

**D1b discovery from data alone** (Track B: no formula is supplied; every arm
receives data and response only). On S tasks, a predictor is *found* by an arm
when permuting it on the test rows (10 repeats) raises the test loss by more
than twice the standard error of the increase and by more than 0.5% of the
test loss. The same rule applies to every arm. Report true-positive rate over
true predictors and false-discovery rate over found predictors, by structure
type (nonlinear main, linear main, interaction, scale-only, proxy), and the
share of S4 runs with no finding. x2 in S3 and the proxies in S6 are reported
as their own types and excluded from FDR. For mboostrs the predictors in its
final formula are also reported as its native claim. On R, the 38 published
effect directions of the well-studied tier (`known_effects.json`) are scored
by the 10th-to-90th percentile perturbation used in the earlier study, for
every arm.

**D9 uncertainty honesty** (regression tasks of R and S): each arm's
idiomatic 90% prediction interval. mboostrs: `mb.conformal_interval(model,
test, level=0.9, method="split", seed=s)`. CatBoost: a second model with
`loss_function="MultiQuantile:alpha=0.05,0.95"`. LightGBM: two models with
`objective="quantile"`, alpha 0.05 and 0.95. XGBoost: one model with
`objective="reg:quantileerror"`, `quantile_alpha=[0.05, 0.95]`. Reference: the
OLS prediction interval of the ridge design. EBM: none offered, recorded as
such. Report empirical coverage and median width over the test-response
standard deviation; on S3, coverage by tercile of x2 (adaptivity anchor). On
binary tasks, expected calibration error of predicted probabilities over ten
equal-width bins.

**D5 iteration speed**: (a) fit-plus-predict seconds on every R and S run;
(b) the scorecard session: on 50,000 × 20 planted data (S1 structure plus 17
noise columns), ten follow-up fits — the base fit, five fits each dropping
one of x0–x4, one with x0·x1 added as a column, one on a 50% row subsample,
one with five more noise columns, one with the response thresholded at its
median (binary) — each followed by prediction on 2,000 rows and a 3-repeat
permutation importance on them. Report the median and p90 seconds per
follow-up.

**D6 robustness**: completion rate on R (messy real data: missing values,
mixed types, unseen levels at prediction); the excess over the reference on R
tasks with fewer than 500 rows and on S5. Large-n (5M rows) is unmeasured.

**D10 reproducibility**: ten R datasets (the first ten in manifest order that
every arm completed at seed 5000) are rerun with the same seed; predictions
must match to 1e-12. Record whether the arm writes a decision trail.

**Unmeasured here**: D1a (no prescribed-specification track), D2, D3, D4, D7,
D8. They stay `unmeasured` in the score sheet.

## Failures and limits

Per run limit 900 s. An error, timeout or non-finite loss is a failure of that
arm and stays in the completion denominator. Paired loss analyses use tasks
every compared arm completed on every seed.

## Scoring

Each measured dimension gets a 1–5 score per arm against the scorecard
anchors, with a written justification citing these measurements, after the
results are read. No aggregate is reported because six dimensions are
unmeasured.
