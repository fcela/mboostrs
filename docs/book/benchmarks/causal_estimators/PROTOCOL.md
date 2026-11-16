# Causal estimator gap study: protocol

Written 2026-10-11, before any arm was fitted on the study draws. The sha256 of
this file and of the scripts are in `PREREG_HASHES.txt`.

Before this file was frozen, the API calls were checked on the first 2,500
training rows of study draw E01 (`run_arms.py --limit-rows 2500`, output
discarded). That check changed the arm list in one way: the `mboostrs.aipw`
`target` argument accepts only `"all"` and `"overlap"`, so the ATT arm was
removed. No estimate from that check is reported.

## Study objective

The causal battery in `docs/book/benchmarks/causal_discovery/` scored mboostrs
below EconML's LinearDML and below a hand-built statsmodels R-learner. The
analysts used different nuisance learners, second stages and defaults, so the
gap could not be assigned to a cause. This study fixes the analyst choices and
measures the gap arm by arm. It asks:

1. How far mboostrs' causal estimators are from EconML and DoubleML when the
   nuisance learners, folds, response weights and confounder set are the same.
2. Which of four sources accounts for the gap: the nuisance learners, the
   estimation framework, the second-stage form, or the defaults.
3. Whether mboostrs is faster than the other libraries at comparable settings.

## Draws

The generator is the job-training model in
`docs/book/benchmarks/causal_discovery/prepare.py`. This study imports its
functions read-only (`covariates`, `propensity`, `outcome_draws`,
`true_effects`, `make_rep`, `query_truths`, `load_population`). The applicant
covariates come from `AER::CPS1988` (`/tmp/cps1988.csv`, sha256
`ef5ee9cb...386251`, checked on load).

- Master seed `20261011`. Draw seeds are the first twelve values of
  `np.random.default_rng(20261011).integers(2**63)`, in order.
- Draws `E01` to `E10` carry the planted effects. Draws `Z01` and `Z02` have no
  effect (the generator's null model), and are used to check false rejections.
- Training cohort: 12,000 applicants drawn at random from the CPS population.
- Test cohort: 4,000 applicants drawn from the remaining population with the
  generator's later-intake weights (`1/sqrt(wage)`).
- Monte Carlo seed for the potential-outcome means: `5000 + k` for draw index
  `k` (the generator uses 400 draws per training row and 4,000 per test row).
- No replicate directory of the causal-discovery battery is read, and its
  `data/dev` files are not used. The draws are regenerated in memory by
  `draws.py`; the data files are not committed.

## Estimands and truth

- ATE: the mean over the 12,000 training applicants of the true individual
  effect `y1 - y0` (generator truth, `query_truths`).
- CATE: the true individual effect of each test applicant (`test_truth.cate`).
- Overlap pool: test applicants whose true enrolment probability `e(x)` lies in
  `[0.02, 0.98]`, the `DEFAULT_OVERLAP_BOUNDS` of mboostrs.
- ATT is not compared. `mboostrs.aipw` has no ATT target, and DoubleML's ATTE
  score refuses the `weights` argument this study needs.

## Response handling (every arm)

Earnings are missing for applicants who did not answer the follow-up survey.
Response depends on enrolment and on the confounders, so the responders are
not a random sample. Every arm uses the same inverse response weights:

- Response model: logistic regression of `responded` on enrolment, log prior
  earnings, education, months unemployed, experience, the three region
  indicators (northeast is the reference) and the metropolitan indicator. It is
  fitted on all 12,000 training rows, after standardising the columns, with
  `C = 1e6`.
- Weight `w = 1 / p_hat(responded = 1 | enrolment, covariates)` for each
  responder. Non-responders are dropped from every arm. Weights run from about
  1.05 to 1.65 in every draw, with a mean near 1.19 (see the erratum at the end).
- How each library receives `w`:
  - `mboostrs.aipw` and `mboostrs.effect_model`: `weights=w`. The library
    treats weights as frequencies (`_sample_weights`).
  - EconML `LinearDML` and `CausalForestDML`: `sample_weight=w`, passed to both
    nuisance fits and the final stage.
  - DoubleML `IRM`: `weights=w`. In DoubleML 0.11.4 these weights multiply the
    ATE score only; the nuisance learners are fitted on the responders without
    weights. The arm therefore carries a propensity fitted on a selected sample,
    and the study reports that as part of the arm.
  - Hand-built R-learner: `w` in both nuisance fits and the second stage.

## Covariates

- Confounder set `W`: `prior_earnings`, `education`, `months_unemployed`,
  `experience`, `region` (four levels) and `smsa`. This is the true confounder
  set of the generator's key. `age` is not used because `age = education +
  experience + 6` exactly, so `experience` carries the same information.
- Effect-modifier set for CATE: `W`. The effect-modifier columns of the key
  (`parttime`, `prior_earnings`, `education`, `months_unemployed`) are only
  partly inside `W`; `parttime` is outcome-only and is not in the design.
- Numeric design `X` for EconML, DoubleML and the hand-built arms: `prior_earnings`,
  `education`, `months_unemployed`, `experience`, indicators for midwest, south
  and west, and an indicator for metropolitan residence (8 columns). The
  mboostrs arms take the raw columns, with region and smsa as factors.
- Excluded: `ethnicity` (affects enrolment only), `distance_km` (instrument),
  `parttime`, `certificate`, `hours_post`, `responded`, `caseworker_note_len`.

## Folds and seeds

- Draw index `k` (0 to 11) sets the fold seed `101 + k`.
- Five folds, `KFold(shuffle=True, random_state=101 + k)`, over the responder
  rows. The same split is passed to EconML (`cv=`), to DoubleML
  (`set_sample_splitting`) and to the hand-built arms.
- The mboostrs arms take an integer `seed=101 + k` and `folds=5`. The library
  builds its own stratified partition from that seed, so its folds are not the
  same rows as the other arms. This is a difference of the library, not a
  setting, and it is reported with the results.
- The nuisance learners of EconML, DoubleML and the hand-built arms are seeded
  with `101 + k` where they accept a seed.

## Arms

Library versions: mboostrs from this repository, EconML 0.17.0, DoubleML 0.11.4,
scikit-learn 1.9.0, statsmodels 0.15.0. Threads: 4 (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
`RAYON_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `n_jobs=4` for forests).

Nuisance learners:

- HGB: scikit-learn `HistGradientBoostingRegressor` (outcome) and
  `HistGradientBoostingClassifier` (enrolment), default settings.
- GAM: `mboostrs.sklearn.GAMBoostRegressor` and `GAMBoostClassifier` with
  `formula = "y ~ bbs(f0, df=4) + bbs(f1, df=4) + bbs(f2, df=4) + bbs(f3, df=4)
  + bols(f4) + bols(f5) + bols(f6) + bols(f7)"`, which is the four continuous
  columns as P-splines and the four indicators as linear terms. `mstop = 100`,
  `nu = 0.1` (the adapter defaults). No tuning.
- Linear (hand-built only): statsmodels `WLS` for the outcome and `GLM`
  (binomial, `freq_weights = w`) for enrolment.

mboostrs arms (defaults unless listed):

| Arm | Call | Estimate |
|---|---|---|
| `mb_aipw_ate` | `aipw(dat, spec, weights=w, level=0.9, seed, folds=5)` | ATE, SE and 90% interval from the library |
| `mb_aipw_smooth` | as above with `smooth=True` | ATE |
| `mb_aipw_noclip` | as above with `propensity_clip=None` (default `(0.01, 0.99)`) | ATE |
| `mb_aipw_notune` | as above with `tune=False` (fixed `mstop=200`) | ATE |
| `mb_aipw_pooled` | as above with `outcome_model="pooled"` (default `"auto"` chooses `"two_part"` here) | ATE |
| `mb_aipw_proptrees` | as above with `propensity_learner="smooth+trees"` | ATE |
| `mb_eff_auto` | `effect_model(dat, spec, weights=w, seed, folds=5)`, `terms="auto"` | CATE; ATE is the mean CATE over the 12,000 training applicants. The library may refuse per-row intervals for a boosted second stage, in which case the arm has none. |
| `mb_eff_smooth` | as above with `terms="smooth"` | CATE with per-row 90% intervals; ATE as above |
| `mb_rank` | `rank_by_effect(mb_eff_auto model, test X, share=0.2)` | the rows the library selects |
| `mb_modifiers` | `effect_modifiers(mb_aipw_ate result, dat, covariates=W)` | the covariates it flags (BLP and GATES, Holm-adjusted) |

EconML arms:

| Arm | Call | Estimate |
|---|---|---|
| `ec_lin_hgb` | `LinearDML(model_y=HGB, model_t=HGB classifier, discrete_treatment=True, cv=KFold, X=X, W=None, sample_weight=w)` | ATE by `ate_inference` over the training applicants (90% interval); CATE by `effect` and `effect_interval(alpha=0.1)` |
| `ec_cf_hgb` | `CausalForestDML(same nuisances, n_estimators=100, n_jobs=4)` | CATE by `effect` and `effect_interval(alpha=0.1)`; ATE is the mean CATE over the training applicants (no SE) |
| `ec_lin_gam` | `LinearDML` with GAM nuisances | as `ec_lin_hgb` |
| `ec_cf_gam` | `CausalForestDML` with GAM nuisances | as `ec_cf_hgb` |

DoubleML arms (ATE only):

| Arm | Call |
|---|---|
| `dml_irm_hgb` | `DoubleMLIRM(ml_g=HGB regressor, ml_m=HGB classifier, n_folds=5, score="ATE", weights=w, trimming_threshold=0.01)` with the shared folds |
| `dml_irm_gam` | same with GAM nuisances |

Hand-built R-learner (statsmodels second stage):

1. Cross-fitted outcome model `m(x)` and enrolment model `e(x)` on the shared
   folds. The outcome model regresses earnings on `X` only.
2. Second stage: WLS (statsmodels, weights `w`, HC1 covariance) of
   `Y - m` on `[D - e, (D - e) * (X - mean_all(X))]`. The intercept-free
   coefficient on `D - e` is the ATE over the training population, because the
   centring is the mean of every training row.
3. CATE: `b0 + (X - mean_all(X)) . b`, with a pointwise 90% interval from the
   HC1 covariance.

| Arm | Outcome and enrolment models |
|---|---|
| `hand_lin` | statsmodels linear (WLS) and logistic (GLM) |
| `hand_hgb` | HGB |
| `hand_gam` | GAM |

## Measures

Per draw and arm, from `run_arms.py`:

- ATE: estimate, error (estimate minus truth), SE and 90% interval where the
  arm provides them, and whether the interval covers the truth.
- CATE, test applicants: PEHE (root mean squared error against the true CATE)
  over all 4,000 test rows and over the overlap pool; coverage of the per-row
  90% intervals over all rows and over the pool; mean interval width.
- Ranking: `k = round(0.2 * pool size)`. The oracle is the mean true CATE of
  the top `k` pool rows by true CATE. The chosen set is the top `k` pool rows
  by the arm's own estimate, or, for `mb_rank`, the rows that
  `rank_by_effect` selects (with `k` equal to their number). Regret in dollars
  is the oracle minus the mean true CATE of the chosen rows. The regret share
  is that regret divided by the oracle minus the pool mean.
- Null draws (`Z01`, `Z02`): the rate at which the 90% ATE interval excludes
  zero, and the mean absolute ATE estimate.
- Timing: wall time (`perf_counter`) and CPU time (`process_time`, all
  threads) for each arm's fit and prediction, plus the one-minute load average
  before and after. The machine is a 10-core Apple Silicon laptop shared with
  other sessions; its load average was 5.5 to 9 when the study started, so the
  timings are upper bounds that vary with load.

Across the ten effect draws, the summary gives the ATE bias (mean error), RMSE,
90% coverage and mean SE; the median PEHE, median regret share and median
timings with their interquartile ranges. The two null draws are reported
apart.

## Attribution

The gap is measured against `ec_lin_hgb`, the EconML arm with the default
nuisance learners. The contrasts below are fixed in `summarize.py`
(`CONTRASTS`), and each reports the first arm minus the second for ATE bias,
ATE RMSE, 90% coverage, median pool PEHE, median regret share, and median wall
and CPU time.

- (a) Nuisance learners, same framework: `ec_lin_gam - ec_lin_hgb`,
  `dml_irm_gam - dml_irm_hgb`, `hand_gam - hand_hgb`, `hand_gam - hand_lin`.
- (b) Estimation framework, same nuisance learners and second stage:
  `ec_lin_gam - hand_gam` (both linear, GAM nuisances), `ec_lin_hgb - hand_hgb`
  (both linear, HGB nuisances), `mb_aipw_ate - ec_lin_gam` and
  `mb_aipw_ate - dml_irm_gam` (mboostrs estimator against the other two with
  GAM nuisances).
- (c) Second-stage form: linear against forest at HGB nuisances
  (`ec_cf_hgb - ec_lin_hgb`) and at GAM nuisances (`ec_cf_gam - ec_lin_gam`);
  smooth against auto at mboostrs' internal nuisances (`mb_eff_smooth -
  mb_eff_auto`). A smooth second stage with the sklearn GAM nuisances has no
  public call in either library, so that cell is not measured.
- (d) Defaults, mboostrs AIPW against its own variants: no clipping, no tuning,
  pooled outcome model, smooth nuisances, smoothed propensity learner.

The report names the largest contrast in each group, by ATE RMSE and by median
regret share, and gives the size of each in the metric's units. The study
applies no significance test to the contrasts; with ten effect draws, a
difference is read as a size, not as a finding with a stated error.

## Outputs

- `per_draw_results.jsonl`: one record per draw and arm, as written by
  `run_arms.py`. `summarize.py` converts it to `per_draw_results.parquet`.
- `results.json`: arm measures and the contrasts from `summarize.py`.
- `README.md`: how to reproduce the run.

## Limitations, stated once

- The mboostrs and non-mboostrs arms use different partitions of the same
  rows (see Folds).
- DoubleML's response weights enter the score only (see Response handling).
- The per-row intervals describe the sampling variance of the second stage,
  given the nuisance fits. They do not cover the bias of the nuisances or of the
  penalised smooth, so their coverage is expected to fall below 90%.
- EconML's forest reports no standard error for its ATE in this design.
- Timings are measured on a shared machine. They compare arms within this run,
  not across machines.

## Erratum (2026-10-11, after the run)

The response-weight range in "Response handling" was written before the weights
were inspected. The observed range is 1.045 to 1.654 across the twelve draws, with
a mean near 1.19 (`w_min`, `w_max` and `w_mean` in `per_draw_results.jsonl`). The
sentence now states that range. No estimator, setting, fold, seed or draw changed.
The script hashes in `PREREG_HASHES.txt` refer to the text before this erratum;
the hash of this file after the erratum is listed there as well.
