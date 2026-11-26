# Distributional prediction shootout: protocol

Written 2026-11-23, before the scored run. Smoke runs used to check the code
are listed under "Exposure before the run". Changes made after results are
read go in DEVIATIONS.md.

## Question

Each method receives development rows and returns, for every test row, a
predictive distribution of the response. How good are those distributions
on held-out rows, measured by a proper score, calibration and interval
coverage, and at what cost in time?

## Methods

Every arm finishes with a model fitted on all development rows. Arms that
choose a boosting step count on held-out rows use one seeded inner 80/20
split of the development rows (`arms.inner_split`), then refit on all
development rows with the chosen count. All arms run on one thread.

| Arm | What it fits |
|---|---|
| `mboostrs.fit` | `mb.fit(dev, "y", seed=s)`. Whatever model the automated search returns. A distributional (LSS) model is used as is; a mean-only model gets its family's distribution at the fitted mean (Gaussian with the development residual SD, or Poisson). |
| `mboostrs.GaussianLSS` | `tune_mstop` (5-fold CV) of a GaussianLSS model with one term per predictor in both parameters: `bbs(x)` for a numeric predictor with 12 or more distinct development values, `bols(x)` otherwise and for categoricals. |
| `mboostrs.LSS_select` | The same for GaussianLSS and StudentTLSS, and GammaLSS and LogNormalLSS when every development response is positive. The family with the lowest cross-validated risk (mean negative log-likelihood per row) is kept. |
| `mboostrs.quantiles` | One additive `Quantile(tau)` model per level in {0.01, 0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975, 0.99}, same terms, each `tune_mstop`-ed by 5-fold CV of the check loss; rows sorted. |
| `hessboost.normal` | hessboost 0.4.8 `dist:normal` at library defaults, rounds by inner early stopping (patience 50, at most 2000). |
| `hessboost.tuned` | hessboost families normal and student_t (plus gamma and lognormal for a positive response) crossed with max_depth {2, 4, 6} and min_child_weight {1, 20} at eta 0.05, rounds by inner early stopping on NLL; the configuration with the lowest inner-holdout CRPS is refitted. |
| `ngboost.normal` | NGBoost 0.6.1 Normal at defaults, rounds by inner early stopping. |
| `catboost.uncertainty` | CatBoost 1.2.10 `RMSEWithUncertainty` (Gaussian mean and variance), defaults, native categoricals, iterations by inner early stopping. |
| `lightgbm.quantiles` | LightGBM 4.7.0 `objective="quantile"` at the same 15 levels as `mboostrs.quantiles`, defaults, rounds by inner early stopping on the check loss; rows sorted. |
| `qrf` | quantile-forest 1.5.0 `RandomForestQuantileRegressor` defaults (Meinshausen's quantile regression forest). |
| `drforest.mmd`, `drforest.cart` | drforest 0.2.0 `DistributionalRandomForest` defaults with MMD or CART splitting. |
| `drforest.tuned` | drforest over criterion {mmd, cart} x min_samples_leaf {5, 20, 50}, chosen by inner-holdout CRPS. |
| `distreebution.crps_rf` | DisTreebution's CRPS regression forest (distreebu_rs 0.3.0, built from source at commit 08c73ba): 100 trees, each on a 60% subsample without replacement, max_depth 6, leave-one-out correction on. A test row's distribution pools the training responses of its leaf in every tree, each tree weighted equally and each response uniformly within its leaf, as in DisTreebution's `UQ` with its default `vr` aggregation. |
| `distreebution.crps_rf_tuned` | The same over max_depth {4, 6, 10}, chosen by inner-holdout CRPS. |
| `distreebution.pmqrf` | DisTreebution's pinball multi-quantile forest: the same forest with trees split on the summed pinball loss at the 99 scoring levels. |
| `reference.gaussian_linear` | Least squares on the design matrix with one residual SD for every row. |
| `reference.climatology` | The empirical distribution of the development responses, the same for every row. |

The six mboostrs arms are not all distinct claims. `mboostrs.LSS_select` is
the primary mboostrs arm: it is the distributional workflow a user of the
library is directed to. `mboostrs.fit` measures the automated entry point.
The primary arm of each competitor library is its tuned arm where one exists
(`hessboost.tuned`, `drforest.tuned`, `distreebution.crps_rf_tuned`).

Shared preprocessing (`arms.prepare`), identical for every arm: development
medians fill missing numeric values; missing categoricals become a level
"NA"; a categorical keeps its 50 most frequent development levels and maps
the rest, and unseen test levels, to "_other"; development-constant columns
are dropped. Matrix-input arms get numeric columns plus one-hot categoricals.
CatBoost and mboostrs get the categorical columns as strings. Every arm sees
the response divided by its development standard deviation, and its
forecast is multiplied back before scoring (the log density gets the
Jacobian term). Scores are on the original scale. The division keeps a
positive response positive, so the gamma and log-normal families stay
available.

Excluded: XGBoostLSS and LightGBMLSS (PyTorch dependency; hessboost covers
the same per-parameter Newton boosting of a parametric likelihood), PGBM
(requires Python below 3.12), R `drf` (drforest is the Python and Rust
implementation of the same estimator), R gamboostLSS (the mboostrs parity
study is a separate question).

## Tasks

**Planted truth** (`tasks.SYNTH`): 10 predictors uniform on (-1, 1), of which
at most 3 matter; development sizes 500 and 5,000; 5,000 test rows; seeds 0,
1, 2. 48 task instances.

| Task | Conditional distribution of y |
|---|---|
| S1 hetero_gauss | N(2 sin(2x0) + x1, exp(-0.7 + 0.8x2)²) |
| S2 student_t3 | 2 sin(2x0) + x1 + exp(-0.7 + 0.6x2) t₃ |
| S3 gamma | gamma with mean exp(0.5 + 0.6 sin(2x0) + 0.4x1) and shape exp(1 + 0.8x2) |
| S4 lognormal | log y ~ N(0.3 + sin(2x0) + 0.3x1, (0.25 + 0.25(x2 + 1))²) |
| S5 bimodal | equal-variance (0.4²) Gaussian modes at ±1.5 + 0.5x1; x0 sets the weight of the upper mode, logistic(3x0) |
| S6 skew_flip | 0.8x1 plus an error of mean 0 and variance 1 whose shape depends on the tercile of x0: Exp(1) − 1, N(0, 1), 1 − Exp(1) |
| S7 interaction | N(2x0x1 + x2, exp(−0.7 + 0.8x0x1)²) |
| S8 linear | N(x0 + 0.5x1 − 0.5x2, 0.5²) |

S1, S2, S3, S4 and S8 lie inside some mboostrs LSS family with additive
predictors. S5 and S6 lie outside every parametric family offered. S7 needs
an interaction that an additive model cannot represent.

**Real data**: the 35 datasets of OpenML-CTR23 (study 353), a curated
regression benchmark suite, with their default targets and no target
transformation. Seeds 0, 1, 2. Test rows: 20% of the dataset, at most 5,000.
Development rows: the rest, at most 10,000 (a seeded subsample beyond that).
105 task instances. Manifest with SHA-256 per file:
`benches/distributional_shootout/data/manifest.json` (its SHA-256
`80aa7466...3a01c6`).

## Scores

All scores are on the response scale, computed from the arm's quantiles at
the 99 levels 0.01, ..., 0.99, so every arm is scored the same way whatever
form its distribution takes.

- **CRPS (primary)**: the quantile decomposition, twice the mean pinball
  loss over the 99 levels. This is the continuous ranked probability score
  with the integral over levels replaced by a 99-point average; it is a
  proper score for the 99 quantiles. Arms that fit 15 levels are linearly
  interpolated in tau between them, which costs them a little.
- **Skill over climatology**: 1 − CRPS / CRPS(`reference.climatology`), per
  task instance. Scale-free; used to average across datasets.
- **Gap to best**: CRPS / (lowest CRPS of any arm on that task instance) − 1.
- **Excess over the truth** (planted tasks): CRPS / CRPS of the true
  quantiles − 1, and the mean absolute difference between predicted and true
  quantiles.
- **Calibration**: for each level tau, the share of test responses at or
  below the predicted tau-quantile; the mean and maximum absolute departure
  from tau over the 99 levels.
- **Central intervals** at 50, 80, 90 and 98%: coverage, mean width over the
  test-response SD, and interval score.
- **Adaptivity** (S1–S6): 90% coverage within terciles of the predictor that
  sets the spread or shape, and its largest departure from 0.9.
- **Log score**: mean negative log density, for parametric arms only (the
  forests and quantile arms have no density). Reported with the count of
  rows where the density is zero.
- **Point accuracy**: RMSE of the predictive mean, MAE of the median.
- **Cost**: wall-clock seconds of fit and predict for the arm, one thread.

## Analysis

Planted and real tasks are analysed separately. Seeds are averaged within a
task before tasks are compared.

1. Per task, arms ranked by mean CRPS; mean rank over tasks, on tasks every
   arm completed.
2. Mean skill over climatology and median gap to best, per arm.
3. Paired comparisons of `mboostrs.LSS_select` and `mboostrs.fit` against
   every other arm: per-task log CRPS ratio, its mean over tasks with a
   bootstrap 95% interval (tasks resampled, 10,000 draws), wins and losses
   counting a task as tied when the ratio is within 1%, and a two-sided
   Wilcoxon signed-rank test.
4. Planted tasks by generator: excess over the truth per arm and DGP, to show
   which conditional shapes favour which method.
5. Calibration, coverage, adaptivity, log score, point accuracy and seconds
   as tables per arm.

## Failures

A run over 3,600 seconds is killed. (The limit was raised from 1,800 after `mboostrs.quantiles` took 1,849 s on superconductivity in the timing probe.) An error, timeout or non-finite score is
a failure of that arm on that task instance. Completion rates are reported
per arm. Rankings and paired comparisons use tasks every compared arm
completed on every seed; the count of tasks dropped is reported.

## Exposure before the run

Code checks were run on S1 and S3 at 500 rows, seed 0, every arm, and on
superconductivity (OpenML 44964) seed 0, every arm, for timing. They led to
these changes before the protocol was fixed: LSS step counts are tuned with
the per-parameter formula mapping `tune_mstop` requires; hessboost at
defaults stopped after 1 to 9 rounds on those tasks, so the `hessboost.tuned`
arm was added; tuned arms for drforest and CRPS-RF were added so that every
focal competitor has one.

The response scaling was added after these probes, all at seed 0:

- On California housing (median value in dollars, SD about 115,000), a
  GaussianLSS fit's mean did not leave its offset. Test RMSE was 114,622 at
  200 and at 1,000 steps, with and without MAD stabilization, and the
  `tune_mstop` search ran to its 10,000-step limit. Test CRPS was 64,386,
  against 63,083 for climatology, 37,900 for least squares and 24,127 for QRF.
  With the response divided by 100,000 the same fit reached RMSE 65,000.
- On superconductivity (SD 34) the same search also ran to its limit.

The cause is that boosted LSS fitting is not scale-equivariant. MAD and L2
stabilization divide the gradient by its spread, so each step moves the
mean by about nu in response units, and a response measured in hundreds of
thousands needs on the order of a million steps. mboostrs matches R
gamboostLSS's `stabilize_ngradient` here, so R behaves the same way. The
library does not warn about it. This is reported as a finding about
mboostrs. `tune_mstop` also refuses `method="noncyclic"` for LSS
specifications (`NotImplementedError`), so the tuned LSS arms stay cyclic.

DisTreebution's PMQRF trees panic in distreebu_rs (`Option::unwrap()` on
`None`, `src/lib.rs:445`, an empty heap in `entropies_multi_quantiles`) on
superconductivity, California housing and geographical origin of music, with
99 and with 15 quantile levels. The arm is kept; its failures count.

No arm's configuration was changed in response to its test scores.

## Environment

Commit 81e8b12 (the compiled mboostrs extension is from 12c87a3; no Rust
source changed between them). Python 3.12.11 in a separate venv,
`benches/distributional_shootout/.venv`, because NGBoost's dependency
lifelines pins pandas below 3.0 and the repository venv has pandas 3.1.1.
Library versions: numpy 2.5.5, scipy 1.18.1, pandas 2.3.3, polars 2.1.0,
scikit-learn 1.9.1, hessboost 0.4.8, drforest 0.2.0, ngboost 0.6.1,
quantile-forest 1.5.0, catboost 1.2.10, lightgbm 4.7.0, distreebu_rs 0.3.0.
Apple M-series, 10 cores, 24 GB; 8 runs in parallel, each on one thread.
