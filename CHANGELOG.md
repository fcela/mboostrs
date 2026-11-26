# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `autospec(..., search="fast")`: unified 1–3 sweep specification search backed
  by a single-pass held-out screening radar (variable weeding, shape
  classification, interaction detection, and spread detection on interactive
  residuals; uses `sklearn.ensemble.HistGradientBoosting` when installed with a
  pure-NumPy fallback) while retaining `mboostrs`'s shared-fold paired-SE
  adoption gates, `SpecRecommendation.report()` audit trail, and interpretable
  GAM/GAMLSS formulas.
- Rust `btree` histogram-subtraction engine (`SplitMethod::Histogram`): skips
  $n \times p$ pre-sorting allocations, packs `(sum_w, sum_wu, count)` bins
  into a contiguous buffer, uses parent-minus-smaller-child histogram
  subtraction, and skips child histogram builds at maximum depth (~3× faster
  depth-5/6 trees). Exhaustive `autospec` also adopts `split_method=histogram`
  for multi-column trees when $n \ge 2{,}000$.

### Changed

- Distributional fits (`gamboostLSS`, `glmboostLSS`, their `cvrisk` and
  `stabsel`, `tune_mstop` on a distributional specification, `autospec` and
  `fit`) divide the response by a scale and map the fitted model back
  exactly to the original units. `response_scale="auto"` (the default) takes
  the weighted SD of the response each fit sees, so a cross-validation fold,
  a weighted refit and a fit on the same rows agree; a number fixes the
  scale; `None` (or `"none"`) fits the raw response, R gamboostLSS's path.
  Families without units (counts, proportions, compositions) are not
  scaled. gamboostLSS moves a location parameter by about `nu` per step in
  response units, so the same data in different units gave different fits:
  on California housing in dollars a GaussianLSS mean never left its offset
  (test RMSE 114,622, the response SD, after 1,000 steps; CRPS 62,111, worse
  than the unconditional distribution's 63,083), and the scaled fit reaches
  CRPS 33,047. Predictions, coefficients, risk and `logLik` are in the
  original units; the path at a given `mstop` is R's path on `y / s`, which
  differs from R's path on `y` when the SD is far from 1. The R package takes
  the same argument, defaulting to `getOption("mboostrs.response_scale",
  "auto")`. `parity_notes()` names the deviation on a scaled fit.
- `Quantile` and `Laplace` take `response_scale="auto"` too (their gradient,
  `tau - 1{y < f}` or `sign(y - f)`, does not grow with the response, so a
  fit moved by about `nu * tau` per step in response units). On California
  housing in dollars, a tuned median regression's pinball loss was 43,787
  and the 0.9-quantile's 53,700; on the scaled response 21,537 and 12,772.
  Under scaling the Laplace offset search runs on `y / s`.
- `Quantile` starts at the `tau`-quantile of the response (`qoffset`
  defaults to `tau`; R's `QuantReg` uses 0.5), and a fit given no `control`,
  like `tune_mstop` given no `nu`, takes steps of `nu = 0.3`. On 18 quantile
  fits (six OpenML-CTR23 data sets, `tau` 0.1, 0.5 and 0.9) the larger step
  cut the median time from 7.2 s to 2.7 s with the pinball loss 1.6% lower on
  average (worst task 2.3% higher). `Quantile(tau, qoffset=0.5,
  response_scale=None)` with `boost_control(nu=0.1)` is R's `QuantReg`.
  `tune_mstop(nu=None)` (the new default) uses a family's own `default_nu`,
  else 0.1.
- The GammaLSS offset search reaches a negative log-mean and a log shape
  below 0 when R's search interval cannot (a response with mean below 1, or
  more skewed than an exponential); every offset R's interval contains is
  unchanged.
- `tune_mstop` tunes a non-cyclic distributional specification
  (`method="noncyclic"`, #4): every fold's path is fitted once and resumed
  from one grid to the next, as for the cyclic method, and the grid counts
  total steps.
- A model returned by `fit` predicts a `brandom` or factor level the training
  rows did not contain as the average group, and the trail's `final`
  records `unseen_levels: "average"` (the trail schema gains the field).
  `predict` and the other prediction methods take `unseen_levels=None` by
  default, meaning the model's policy: `"error"` for a model fitted
  directly, as before. Before, a model `fit` built on OpenML
  video_transcoding refused every test split.
- `autospec`'s bar for adopting a distributional family (2% of the mean
  family's held-out log score, plus two paired standard errors) reads the
  log score on the response divided by its SD, so it no longer depends on
  the response's units.
- When the spread screen (`log(residual^2)` regressed on each column) finds
  no covariate, `autospec` offers GaussianLSS with a joint depth-3 tree over
  the numeric predictors as its scale equation, decided by the held-out log
  score with the usual gates, so a spread that depends on a product or a
  threshold of predictors is no longer missed.
- `autospec(lss_trees=...)` offers an adopted distributional model a joint
  tree (depths 3 and 6) in every parameter; `"auto"` (default) runs it when
  the remaining budget pays for it.
- `fit()` defaults `na_action` to `"impute"`, `interactions` to `"adopt"`,
  `residual_tree` to `"auto"`, and the `"standard"` fit budget to 64 fits.
  Above 3,000 rows (with no explicit `folds`, `groups`, or `time`), `fit()`
  runs specification search on a deterministic 3,000-row subsample (pinning
  response extremes) and fits the chosen specification on all rows.
  `Fit.trail.final["formula"]` reports active terms (>= 1% of risk reduction
  or LSS partial-effect variance, or intercept-only when CV risk does not
  improve over step 0 by 1%).
- `autospec` pre-exploration caps its screening table at 10,000 rows, skips
  1D shape probing on near-zero-contribution columns on wider tables (> 8
  predictors), retains observed-response rows during search under impute
  modes, reuses `MstopEvidence.fold_scores` in the residual-tree gate, and
  adds held-out residual-tree detection with sparse active-column scoping.
- `conformal_interval(..., unseen_levels=None)` inherits the model's unseen-level
  policy, and lowered factor `bols` and dummy-encoded categorical columns
  support `unseen_levels="average"` and `"zero_effect"`.
- The divergence errors of distributional fits name `stabilization="MAD"`,
  a smaller `nu` and a family for the response's shape as remedies.

### Fixed

- `predict(..., unseen_levels="average")` on a categorical column with a
  missing value (`None`) the fit never saw raised `TypeError` while listing
  the unseen levels.
- `impute_missing` adds a training-mode `impute_level` step for complete
  training categorical columns, keeps Polars `impute_level` columns as native
  `Utf8`/`String` rather than `pl.Object`, and drops post-omission constant
  predictors before probing.

### Documentation

- The book's R-parity chapter records every difference above, with the
  measurements, and the R-migration and family pages give the settings that
  reproduce R's fits.

## [0.2.3] - 2026-11-23

### Changed

- `fit(budget=...)` now gives the specification search a number of model
  fits: 16 for `"quick"`, 32 for `"standard"` (the default) and 512 for
  `"thorough"`. The count is checked before each stage starts, so a stage in
  progress can exceed it. Before, `fit` passed no budget to `autospec`, so the search
  measured no candidates and returned the dtype-driven default learner for
  each column. On nine competitive-study panels (three seeds each) the
  default specification's mean test loss was a median 7.8% and up to 63%
  above that of the best budget tried; on a tenth, a time-ordered panel, it
  kept a factor whose later levels never occur in training, so prediction
  stopped with the default `unseen_levels` setting. `"standard"` is within
  0.1% in median and 8.4% at worst on all ten, at a median 4.5 s per fit at
  one thread. The trail's `budget` records the limit (`limit`), the
  fits spent, and a degradation entry when the search stopped at the limit.
- `fit` limits the specification search's wall-clock time to 60, 120 and
  1200 seconds for `"quick"`, `"standard"` and `"thorough"`. When the time runs
  out, the search keeps what it measured if the probe in progress allows it,
  and otherwise falls back to the default specification `fit` returned
  before (one default learner per column, nothing measured). The trail's
  `budget` records `seconds_limit`, and `degradations` says which outcome
  occurred. Without the limit, a 1,025-column table (OpenML 44988,
  QSAR_TID_11) ran for more than 600 s; with it, `fit` returns in 121 s with
  the default specification. Inside an outer workflow budget, such as a
  `SearchSession`, that budget governs cost instead.

### Documentation

- The book has a page on `mb.fit` against CatBoost, LightGBM, XGBoost, EBM and
  a ridge or logistic reference on held-out data, scored per dimension under
  `mboostrs-scorecard/2.1`, with the comparison of `fit` against its previous
  version (`benchmarks/heldout_competition.ipynb`). The benchmark status page
  and `docs/scorecard/status.md` record the new D1b and D5 scores.
- A new appendix, "Benchmark records", consolidates the benchmark suite's
  protocols, validation and decision records, study reports, and agent-battery
  outcomes into nine structured scientific appendix chapters under
  `docs/book/appendix/records/` (with immutable pre-registrations and raw
  transcripts retained under `docs/benchmarks/` for audit), alongside a new
  canonical writing guide (`docs/WRITING_GUIDE.md`), a reconciled
  `docs/scorecard/` status summary, and a new blind-agent five-library
  heteroscedastic benchmark chapter (`benchmarks/blind_agent_five_libraries.md`)
  comparing `mboostrs` against CatBoost, LightGBM, InterpretML EBM, and NGBoost.

### Fixed

- `LSSModel.summary()` crashed with `TypeError` on non-cyclic or per-parameter
  dict `mstop` fits and with `KeyError`/`IndexError` on `glmboostLSS` fits.
- `conformal_interval()`, `LSSModel.predict_parameters()`, `LSSModel.cvrisk()`,
  and `discover()` now accept and propagate `unseen_levels=` (`"error"`,
  `"average"`, `"zero_effect"`).
- `fit()` and `autospec()` now accept `na_action=` and pass it through to model
  fitting, and `autospec` hygiene checks no longer misreport string/categorical
  or integer columns as `"discrete-under-float"`.
- Added the missing `[.mboostrs_gamboostLSS` S3 subsetting method in the R
  package (`mboostrs/R/mboostrs_gamlss.R`, exported in `NAMESPACE`).
- Fixed the `R parity evidence` GitHub Actions workflow to install `[dev,sklearn]`
  and `[plot]` extras alongside the built wheel, and restored the missing
  `mb.family_set("realplus")` reference in the agent quick-reference topic.

- `autospec` with a budget kept the fixed step for a Poisson, negative
  binomial or Gamma family whenever the line search did not reach a lower
  cross-validated risk, including when the budget ran out before the stopping
  depth was measured and neither run had a risk to compare. In that case no
  fit had checked the fixed step. On a 30-column Poisson grade response
  (OpenML 44967, mean 12) the unchecked step of 0.3 drove the fitted mean to
  0.18 and test MSE from 7.2 to 138.9. On another split of the same data the
  fixed-step run's family probe read a Pearson dispersion of 4631 off a
  diverged Poisson probe and adopted NBinomial (test MSE 20.0 against 7.7).
  `autospec` now takes the line-search run, family included, when the fixed
  step's risk was never measured, as it already did at budget 0, and records
  the reason in `rec.evidence["step"]`.

## [0.2.2] - 2026-11-18

### Fixed

- Rewrote the bundled agent guide around the installed API: runnable held-out
  prediction and save/reload examples, boosting versus REML routing, actual
  result fields, task-appropriate folds, and optional diagnostic workflows.
  Documented the convenience wrapper's implementation limits and tightened the guide's
  API check so version gates cannot hide references to nonexistent functions.

## [0.2.1] - 2026-11-18

### Fixed

- `gamlss_refit` returned a non-converged fit (`stop_reason="no_descent"`, the
  null smooth left unshrunk, REML criterion 389 instead of 168 on the
  regression test) on some x86-64 CPUs. The default smoothing-parameter pass
  stopped early on every platform, but the strict second pass that recovers the
  fit ran only when the final re-evaluation at that point failed, which happens
  on some CPUs and not others. The strict pass now runs whenever the default
  pass does not converge, and the converged (otherwise lower-criterion) result
  is kept.
- The README links to the documentation and to repository files are absolute,
  so they resolve on the PyPI project page.
- The smoothing-parameter searches of `gam`, `gam_refit`, `gamlss` and
  `gamlss_refit` stopped at different points on different CPUs. Near the
  optimum a step's predicted decrease in the REML/LAML criterion can be smaller
  than the rounding error in the computed criterion (1.4e-7 against up to
  6.6e-6 on a criterion of 1496 in the regression test), so the value test
  accepted the step on some CPUs and rejected it on others, and the rejected
  runs stopped on the `flat` rule with a gradient of 9e-4. When a trial point's
  criterion is within 1e-7 (relative) of the current value, the line search
  now decides by the slope along the step (the approximate Wolfe conditions of
  Hager and Zhang, 2005). On ten GitHub runners covering five CPU models, the
  direct and refit fits of that test now both stop on the gradient rule and
  agree to 3.1e-7 in the coefficients, where the EPYC 7763 runs differed by
  1.7e-5 before.
- Fits that use faer's parallel eigendecomposition and SVD (direct REML
  GAMs, refits and learner setup) gave results that differed in the last bits
  with the number of threads. faer's parallel kernels now always use 8 tasks,
  so results are bit-identical at any thread count. On 10 cores this costs
  about 2% of speed.

## [0.2.0] - 2026-11-16

Changes since the 0.1.3 release candidate (commit `16889c63`). The summary
lists affected public symbols and issue or PR references. Detailed notes
record the behavioral differences and their validation.

### Breaking

Measured against the public API of the 0.1.3 release candidate (commit
`16889c63`). Each entry was checked in the code; a function added after 0.1.3
cannot break a 0.1.3 caller, so changes to such functions are listed under
Changed.

Python:

- `glmboostLSS` and `gamboostLSS` (`python/mboostrs/gamlss.py`): every
  argument after `data` is keyword-only. `glmboostLSS(f, d, GaussianLSS())`
  raises `TypeError`; write `families=`.
- `GLMBoostModel.aic` and `GAMBoostModel.aic` (`glmboost.py`, `gamboost.py`,
  `_criteria.py:aic_contract`): the default `method="corrected"` raises
  `NotImplementedError` for every family other than `Gaussian`,
  `method="classical"` raises for `Gaussian` and `GaussReg`, and a model with
  no hat matrix raises instead of returning `None`. 0.1.3 returned a number or
  `None` in each of these cases.
- `logLik()` (`gamboost.py`, `glmboost.py`, `_likelihood.py`) raises
  `NotImplementedError` for `Laplace`, `Quantile`, `Expectile`, `Huber`,
  `AdaExp` and `HingeLoss`, where 0.1.3 returned `-risk`. `negative_risk()`
  returns the 0.1.3 value.
- `residuals(type=...)` (`_residuals.py`): `type="pearson"` raises
  `NotImplementedError` for `Laplace`, `Quantile`, `Expectile` and `Huber`
  (0.1.3 returned the working gradient), and the response residual raises for
  `CoxPH`, `Weibull`, `Loglog` and `Lognormal`.
- `Bmono` (`learners.py`): the default `type` changed from `"iterative"` to
  `"quad.prog"` (R's default), so every `bmono` term without an explicit
  `type` fits different numbers. Pass `type="iterative"` for the 0.1.3 fit.
  This change had no changelog entry; it came in commit `97078007`.
- Weights (`_weights.py`; R `.normalize_weights`): under the default
  `weight_policy="upstream"`, `glmboost`, `gamboost`, `glmboostLSS` and
  `gamboostLSS` rescale non-integer weights to sum to the number of positive
  weights, as mboost's `rescale_weights` does, which changes the fitted path.
  `weight_policy="raw"` gives the 0.1.3 fit. All-zero weights raise.
- `stabsel`, `stabsel_glmboostLSS`, `stabsel_gamboostLSS` and `StabSelResult`
  (`stabsel.py`): under `sampling_type="SS"` every path fits complementary
  pairs; `B` counts initial subsamples and the new `n_fits` counts fits;
  caller-supplied `folds` are initial subsamples whose complements are
  appended. Selection frequencies and error bounds differ from 0.1.3.
- `fitted_` and `risk_` (`_state.py:readonly`) return read-only views; an
  in-place write raises `ValueError`.
- `GAMBoostModel.predict` and `GLMBoostModel.predict` with `mstop` above the
  fitted step count raise `ValueError`; 0.1.3 returned the full-path
  prediction.
- Learner labels (`gamboost.py:_disambiguated_names`): learners that read the
  same column, such as `bols(x) + bbs(x)`, are named `"x (bols)"` and
  `"x (bbs)"` in `base_learner_names()`, and `coef()`, `partial_effects()` and
  `varimp()` use those keys. 0.1.3 named both `"x"`, and the later one
  overwrote the earlier in keyed results.
- Packaging (`pyproject.toml`): `altair` and `vl-convert-python` moved to the
  `plot` extra and `pyarrow` to the `arrow` extra. `pip install mboostrs` no
  longer installs them; plotting needs `pip install "mboostrs[plot]"`.
- `mboostrs.cvrisk` (`__init__.py`) is no longer bound as a package
  attribute, so `mb.cvrisk.cv_weights` raises `AttributeError`.
  `mb.cv_weights` and `from mboostrs.cvrisk import cv_weights` work.

Numerical results of 0.1.3 calls also change through fixes that move
mboostrs toward mboost; each is listed under Fixed with its measured
difference in the detailed notes, and they were not re-run for this list:
`bols(x, by=z)` and penalized `bols`, the `df` of `%X%` tensors, `df` to
`lambda` calibration (`r_zeroin`, singular values), `confint` bootstrap
weights, `Multinomial` `cvrisk` scoring, weighted `cvrisk`/`stabsel` fold
refits, `PropOdds` with four or more levels, `btree(x, by=z)`, and the risk
reported by packed Gaussian scalar fits.

R:

- `bols(x, df = ...)` raises in the R frontend (`mboostrs_gamboost.R`);
  0.1.3 fitted it with a penalty that did not match the requested `df`. Use
  `lambda =`.
- Weight rescaling and the stability-selection fold changes above apply to
  the R functions too.

Rust (`mboostrs-core`):

- `prepare_bmono` takes a new `center: bool` argument and
  `prepare_kronecker_bbs` a new `deriv: usize` argument.
- `stabsel_parameters` returns `Result<StabSelParams, MboostrsError>`; the
  error type was `String`.
- `MboostrsError` gains `BadFamilyParameter`, `StabSel` and `UnseenLevel`;
  `PreparedLearner` gains `Bhinge`, `BinnedBlockBols`, `BinnedScalarBols`,
  `BspatialStreamed` and `FactoredBlockBols`. Neither enum is
  `#[non_exhaustive]`, so an exhaustive `match` no longer compiles.
- `BoostControl` gains the public fields `component_fraction`,
  `instrumentation`, `nu_per_param`, `stabilization`, `step` and
  `subsample_seed`, and `BoostingPath` gains `nuisance_vec_path`, `report` and
  `residual`; a struct literal without `..` no longer compiles.
- `cvrisk_selected` and `cvrisk_selected_with` now apply their `weights`
  argument, which 0.1.3 ignored (`_weights`).

### Added

- `reference_measure(family)`; `gaic_table` refuses count vs continuous families; `scan_lss_families(na_action="omit")`; `compare_cv` refuses different scoring rules (#60).
- `fit(data, response, ...)`, `Fit`, `report(fit, format=)` and `DecisionTrail`: scaffolding of the automated modelling procedure; `mb.report` is now the trail renderer (#474).
- `rearrange_quantiles(models, newdata)`: sorts crossing quantile or expectile predictions and reports the crossings (Rust core, R too) (#103).
- `AIPWResult.predict_propensity(newdata)` (#487).
- `select_capacity(budget="tuned")` (default) tunes each rung's step count; `budget="shared"`, `max_mstop=` and sequence `tree_depth=` (#493, #479).
- `dump_result_json`, `load_result_json` and `ConformalInterval.to_json` (#665).
- `GamlssRefit.summary()` (#506).
- `DiscoveryResult.predictions` carries a `sigma` column when a spread model is adopted (#589).
- `api_card(obj)` and `python -m mboostrs sig NAME` (#505).
- Agent guide topic `golden-path`, the default of `agent_guide()` (#504, #474).
- `select_capacity(..., estimators=("boosting", "reml"))`.
- `screen_discontinuities(data, target)` and `DiscontinuityHit`.
- `iv_effect(outcome_missing="ipw")`; `ci_test` reports NaN for exact identities; warnings once per call (#484).
- `select_capacity(formula, data)`: ladder of base, tensor and tree rungs scored on shared folds.
- `AIPWResult.subgroup_effect(by)` and `AIPWResult.effect_surface(formula)` (#487, #714).
- `estimate_cost` covers `aipw`, `effect_model`, `ci_test`, `iv_effect`, `confounding_benchmark` and `discover`; `DiscoveryResult.stage_seconds` (#484, #487, #447, #713).
- A Polars `Enum` column is a factor in `gamboost`, `glmboost` and their predictions (#717).
- `discover(mean_step="fixed"|"line_search"|"auto")` and an actual-incumbent guard for interactions (#679, #721, #724).
- `reference_gap` and `ReferenceGapResult`: held-out gap between a specification and a flexible `btree` reference, with ranked candidate terms that may close it (#468).
- Agent guide worked examples ship in the wheel (`agent_guide("example:<name>")`, `install_skill`); fitting docstrings and warnings point to `agent_guide()` and `reference_gap` (#468).
- Agent guide topic `modeling-tactics`; `SKILL.md` step 5 asks for a `reference_gap` measurement before stopping (#468).
- `na_action="impute_model"`: observed and filled parts of a `bbs`/`bols` term on an imputed column (later made opt-in, see `impute_model_split`) (#457).
- `na_action="impute_model"`: model-based conditional-mean imputation of missing numeric predictors, fitted in the Rust core; `imputation_diagnostics(model)` (#457).
- `gam(formula, data, ...)` and `gamlss({...}, data, ...)`: penalized GAM / GAMLSS by REML without boosting, returning `GamRefit` / `GamlssRefit` with `contrast()` (#457).
- `GAMBoostModel.plot_btree_partition` and `btree_partition_table`: heatmap and table of a multi-column `btree` term's partition (#280).
- `ci_test(outcome_missing="ipw")`, `CITest.status="dependent"` for exact identities, `EffectModel.predict(output=)` and `collect_warnings()`; linear `aipw` baselines are written `bols(name)` (#444).
- Propensity learner and second-stage row bootstrap for `aipw` and `effect_model` (#423).
- `effect_modifiers(result, data, ...)`: best-linear-projection and group tests of effect heterogeneity from AIPW scores.
- Repeated-split and bootstrap intervals for `effect_model` and `aipw`; `interval_method="auto"` default (#423).
- `aipw(outcome_model="two_part")` for zero-inflated continuous outcomes (#423).
- `aipw(target="overlap")`: overlap-weighted ATE (#423).
- `aipw(outcome_model="auto")` (default) chooses the outcome model from a sparse-arm check (#423).
- `gam_refit(...).predict(newdata, interval=True, level=, corrected=, output=)` and other causal friction fixes (#416).
- Causal estimators accept missing outcomes, observation weights and sensitivity inputs (#416).
- `evaluate_specification` and `cv_risk` accept survival families and LSS fits, report failed folds, and add two fold designs (#72).
- `agent_guide("task-map")`: task-to-API map across Python, R and Rust (#79).
- `cluster_residual_check(model, cluster)` and other residual checks that name a revision (#73).
- `survival_at(times, newdata)` and `survfit()` for `CoxPH` fits, with the Breslow baseline computed in Rust (#78).
- `autospec(exposure=...)` and ordered-categorical response semantics (#127).
- `prediction_contrast`, `average_marginal_effect`, `distributional_contrast` take `cluster=` for cluster-level uncertainty (#76).
- `joint_support=` diagnostic for counterfactual rows in the three contrast functions (#76).
- `distributional_contrast(..., refit=gamlss_refit(...))`: refit-posterior interval (#76).
- `check_declarations(model_or_recommendation, data, ...)`: tests each declared sign or shape against the data (#121).
- `equivalence_sets(model, threshold=0.9)`: groups learners with overlapping design spaces (#116).
- `scale_absorption(model, ...)`: checks whether a `gamboostLSS` scale model absorbs mean misfit (#112).
- `specification_curve(source, data, response, ...)` and `SpecificationCurve` (#119).
- `heredity=`, `forbidden_pairs=` and `mandatory_terms=` on `SearchSession`, `discover` and `autospec` (#74).
- `autospec` compares `bspatial(x, y)` with `btree(x, y)` for a coordinate pair (#338).
- Comparison identity: every fit records a digest of its response, rows, weights and likelihood; ranking helpers refuse mismatched fits (#60).
- Comparison identity extended to `compare_summaries`, cross-validation curves and common samples (#60).
- `GAMBoostModel.predict_over_grid(variable, grid, ...)` (#296).
- `discover(data, response)` and `DiscoveryResult`: rule-driven draft, elimination, interaction and spread search (#357, #292).
- `split_refit(formula, data, ...)` and `SplitRefitResult`: select on one half, refit on the other (#117).
- `validate(model, newdata, ...)`: lists the issues `predict(newdata)` would meet (#293).
- `compare_specifications(..., max_mstop=)` and `cv_risk(..., max_mstop=)` set the inner tuning cap.
- `compare_specifications(mstop="cv")` rows report inner step counts and `inner_at_boundary`.
- `extrapolation_report(newdata)` on `GAMBoostModel`, `GLMBoostModel` and `LSSModel`.
- `provenance(data=None, seed=None)`: version, build and environment record (#357).
- `estimate_cost(call, *args, kwargs)`: fit count and time estimate before a call runs (#357).
- `sensitivity(model, data, parameter=...)` for distributional models (#335).
- `FAMILY_SETS` and `family_set(support)` exported at top level (#337).
- `agent_guide(topic)` and `install_skill()`: agent guide topics and an installable skill.
- `gamboostLSS` and `glmboostLSS` take `na_action=`.
- `insights`, `Finding` and `agent_guide()` are reachable from the installed package top level (#306, #291).
- `compare_models(sort=)` and `selected_learners`; `risk` withheld across incomparable losses.
- `brad(...)` radial learners work in Python and predict at new coordinates in Python and R.
- `NBinomial` fits expose their calibrated `size` (Python property, R `size(fit)`, Rust accessor).
- `cvrisk(preprocessing="refit")` in Python and R rebuilds preprocessing on each training fold; default `"fixed"` unchanged.
- `modeler_loop(..., family=)` accepts an explicit response family.
- `same_family_gap(model, cv)`: gap between a cross-validated linear boost and the same-family GLM.
- `autospec(preexploration="native"|"hessboost"|"off")`: one preliminary tree ensemble for priority, shapes and interaction screening.
- `column_screen(data, response)`: per-column line/curve/step diagnostics computed in the Rust core.
- `autospec` interaction screen adds pairs from a boosted-tree surrogate (Friedman's H).
- `ci_test(data, x, y, given=...)`: generalised covariance measure test of conditional independence.
- `autospec(residual_tree=True)`: tries one residual regression tree term and keeps it on held-out gain.
- `btree` takes categorical columns (two-set level splits).
- R `btree()` takes several columns including factors, and R `predict()` evaluates tree terms on new data.
- `effect_model(data, spec)`: conditional average treatment effect by an R-learner boosted additive model.
- `Multinomial` models take categorical predictors (`brandom(g)`, `bols(f)`) in Python and R.
- `autospec(deselect=0.025)`: deselection share as a number.
- `GamRefit.effects(newdata, centre=True)`.
- R `glmboost`, `gamboost`, `stabsel` and `predict` take a per-row offset.
- `autospec(...).fit(data, refit="reml")` and `gam_refit(unsupported="hold")`.
- `StabSelResult.variables`, `.selected_names`, `Deselection.variables`, `Probing.variables`, `formula.term_variables(term)`.
- `GamRefit.tests` and `GamRefit.summary()` (R `summary(gam_refit(...))`): per-term edf and approximate tests.
- `autospec(..., deselect=True)`: drops main effects under 1% of the risk reduction and re-tunes.
- `mboostrs_core::gam` (`try_gam_refit`): REML refit in Rust, replacing the NumPy/SciPy `gam_refit`.
- `gam_refit` optimizes Laplace-approximate REML for Binomial, Poisson, negative binomial and Cox refits.
- `gamlss_refit` (Python and R): joint penalized-likelihood refit of every distribution parameter.
- `GamRefit.loglik`, `.df`, `.aic()`; R `logLik()`, `AIC()` and `plot()` for refits.
- `GamlssRefit.draws(newdata, n_draws)` (R `gamlss_draws`): joint posterior draws.
- `GamRefit` and `GamlssRefit` pickle through JSON.
- `gam_refit` estimates the negative binomial `theta` by REML.
- `vcov(type="corrected")` and `effects(..., corrected=True)`: smoothing-parameter-corrected covariance.
- R `gam_refit(model, terms, which)` with `effects()`, `predict()`, `coef()`, `vcov()`, `print()`, `summary()`.
- `to_dict(*, include_arrays=False)` on every public result class.
- `deselect(model, threshold=0.01)` and `probe(formula, data)` (module `mboostrs.selection`).
- `gam_refit(model)`: REML refit of a Gaussian `gamboost` model's selected terms, returning `GamRefit`.
- `na_action="impute"` on `gamboost` and `glmboost`: median fill plus `<x>_missing` indicator term.
- `autospec(..., interactions="adopt")`; `SpecRecommendation.fit` expands `%X%` terms.
- `MstopTuning.model`: the tuned full-data fit from `tune_mstop`.
- `bols(f)` on a categorical column in `gamboost` fits one effect per level; `glmboost` expands factors to treatment contrasts.
- `gam_refit` accepts a numeric `by=` (varying coefficient).
- `autospec` fits a skewed strictly positive predictor as `bbs(log(x))`.
- `gam_refit` fits Binomial, Poisson and GammaReg models by PIRLS with REML smoothing.
- `ecdf(x)` formula transform, replayed on new data and saved with the model.
- `na_action="impute"` gives a categorical predictor's missing values a `"(missing)"` level.
- `predict(type="log_odds")` and `partial_effects(scale="log_odds")` for Binomial models.
- `glmboostLSS` and `gamboostLSS` accept string factor columns.
- `deselect` accepts a `method="noncyclic"` LSS fit.
- `confint(groups=)` on `glmboost` and `gamboost`: grouped bootstrap.
- `boost_control(step=, step_iterations=)`: `"fixed"` (default) or line-search step length.
- `compare_specifications` rows carry `search optimism` and `competitive set`.
- `compare_specifications(history=...)` extends optimism and competitive set across calls.
- `holdout(data, fraction=0.2, *, groups=None, time=None, seed=0)`.
- `cv_risk` and the assistants built on it warn when a group term is scored on folds that split its groups.
- `learner_design(term, data)` and `LearnerDesign`.
- The term `1` in an LSS parameter formula is the intercept (`"sigma": "y ~ 1"`).
- `compare_specifications` and `cv_risk` accept distributional candidates.
- `offset=` accepts one value per observation in `gamboost` and `glmboost`.
- `save_model(..., scoring_only=True)`.
- The R reader loads distributional model artifacts.
- `TermReport.fits`: exact model-fit count.
- `term_report` and `tune_mstop` accept distributional specifications.
- `gamboostLSS` accepts `%X%` tensor interactions.
- A formula may carry the response's transform (`log(rent) ~ ...`) in `gamboostLSS`.
- `unseen_levels=` reaches every LSS prediction surface.
- `grouped_folds(..., repeats=)`.
- `unseen_levels="average"|"zero_effect"|"error"` for `brandom` levels the fit never saw.
- `label_evidence`: whole-number codes above twelve values read as groups when the response says so.
- `autospec(..., groups=)` declares group columns with grouped folds.
- `autospec` data hygiene scan reports identifier, unique-integer and response-sibling columns.
- `tune_mstop` cross-validates the step count and grows the grid while the risk still falls.
- `pair_term_for` returns the interaction term two columns call for.
- `term_report` records `se`; `suggest_specifications(noise=2.0)`.
- `heteroscedasticity` tests the residual spread against each continuous predictor.
- `compare_specifications` rows carry `change se`.
- `screen_interactions(..., groups=)` scores a group column on its own levels.
- `suggest_specifications` proposes removing terms within `noise` standard errors of zero.
- `btree(selection="ctree")`: mboost's conditional-inference rule for choosing the split column. [flagged: the code default is `selection="sse"` since #265/#272; `"ctree"` is opt-in]
- `build_info()` (Python) and `mboostrs_build_info()` (R): identity of the loaded native library.
- `density_overlay_data` and distribution-profile tables and plots for LSS models.
- `BCTLSS` and `BCPELSS` families in Rust, Python and R.
- `bkernel(x, landmarks=m)` / `Bkernel(..., landmarks=m)`: Nyström landmark kernel ridge.
- Capability ledger carries an auto-derived `node_capabilities` list.
- `stabilization=` on `cvrisk_*LSS` and `stabsel_*LSS`; a fit-level override survives refits.
- `continue_to` warm-continues standard models from the retained fitted state where the engine keeps a residual.
- `CausalSpec` and `aipw`: ATE/ATT with cross-fitted boosting nuisance models.
- `confounding_sensitivity`: additive omitted-variable bias grid.
- Every LSS family constructor takes `stabilization=`; the fit-level argument defaults to `None` (use the family's).
- `LSSModel.__getitem__`: `fit[:k]` refits a distributional path to `k` steps.
- `model.risk` property on every model class.
- `mboostrs_core::try_fit_continue`: warm continuation of a boosting path.
- `model.unavailable_operations()`.
- `predict(newdata=, extrapolation=...)` and `model.range_report(newdata)`.
- `range_report()` returns one row per spline learner.
- Partial-effect plots shade stretches with thin data.
- `model.binning_report()`.
- `bins="auto"` on a learner or as the global argument.
- `check_binning()` reports `bytes_saved` and `scope`.
- `cv_risk`, `term_report` and `compare_specifications` take `weights=`, `folds=`, `na_action=`, `score=` and `mstop="cv"`.
- `ColumnKind` carries dtype, levels, order, missing count, date unit, `reason` and `alternatives`.
- A declared categorical keeps its level order; an ordered factor is suggested `bols`.
- `Suggestion.specification` and `Suggestion.evidence`.
- `heteroscedasticity(model)` and a distributional-response suggestion.
- `compare_specifications` rows record fold scheme, budget, `nu` and weights; `output=` returns a frame.
- Fold constructors returning a `FoldScheme` for non-exchangeable rows.
- `FoldScheme.check_leakage(groups=...)`.
- `CvRiskResult.standard_error`; CV plots draw it as a band.
- `CvRiskResult.mstop_1se` and `LssCvRiskResult.mstop_1se`.
- `CvRiskResult.compare_to(other)`: paired comparison on shared folds.
- `diagonal_grid(max_mstop, n_params)` and `LssCvRiskResult.grid_cost()`.
- `result.provenance` on every result.
- `model.parity_notes()`.
- `summary()` names the parity notes that apply.
- `GaussianKnownVarLSS` in the R frontend.
- `stabilization=` on `glmboostLSS` and `gamboostLSS`.
- `try_coef_glmboost`, `try_coef_gamboost` and `try_predict_glmboost` return `MboostrsError` instead of panicking.
- `mboostrs._refit.RefitSpec` with `subset(rows)` and `row_subset_blockers()`.
- `model.residual_kinds()`.
- `likelihood_contract()` on any model.
- `weight_policy={"upstream","raw"}` on `glmboost`, `gamboost`, `glmboostLSS` and `gamboostLSS`.
- `bins=` on `gamboostLSS`, `cvrisk_gamboostLSS` and `stabsel_gamboostLSS`.
- `n_fits` on `StabSelResult` and `LssStabSelResult`.
- Quantile-binned training storage (`bins=`) for `Bols`, `Bbs`, `Bspatial` and `Brandom`; `model.check_binning()`.
- `datasets.hip_cz()` and case study `cz_inequality.ipynb`.
- `datasets.usmacro()` and three deposit datasets.
- `llm_explain_prompt()` on `glmboost`, `gamboost` and LSS fits.
- `llm_describe()`: posts the prompt to an OpenAI-compatible endpoint.
- R `llm_explain_prompt()` and `llm_describe()`.
- A factor `by=` on `bols()` and `bbs()` in `gamboost`.
- `LogNormalLSS`, `WeibullLSS` and `LogLogLSS` accept a plain positive response (#333).

### Changed

- `prediction_profile`, `accumulated_local_effect` and the fallback path of
  `GAMBoostModel.predict_over_grid` (`python/mboostrs/_counterfactual.py`)
  stack their counterfactual frames row-wise and predict them in one
  `predict` call, in chunks of at most 2^20 rows, instead of one call per grid
  value or per ALE boundary. The fallback repeats a per-row `offset=` once per
  value. Factor encoding at prediction time (`transforms._encode_factor`) looks
  up each distinct label once. Signatures and return shapes are unchanged, and
  the results equal the per-value loop to 1e-12 relative. On 2,000 rows and 20
  values the median wall time fell from 2.2 to 1.8 ms (glmboost profile), 5.9
  to 3.5 ms (gamboostLSS profile), 9.1 to 5.1 ms (gamboost factor grid), 5.3
  to 1.3 ms (gamboost ALE) and 6.2 to 1.0 ms (gamboostLSS ALE); a `%X%` grid,
  whose time is basis evaluation, changed from 25.8 to 24.5 ms. Under
  `extrapolation="warn"` an out-of-range grid gives one warning for the
  stacked rows instead of one per value. Repeated calls reuse the model's
  learner store; a model restored from a pickle prepares it once (#296).

- R `lss_pdf`, `lss_cdf`, `lss_quantile`, `lss_mean` and
  `lss_deviance_increment` evaluate `mboostrs_core::lss_distribution` and
  `lss_quantile` instead of R code; the BCT and BCPE kernels exist only in
  the core. Results agree with the R implementations to 1e-8 relative.
  `lss_quantile` now also covers the families the core quantile supports
  (GammaLSS, BetaLSS, StudentTLSS and others), not only the ten R listed.
  `lss_mean` requires every parameter of the family. Invalid parameters are
  refused instead of giving `NaN`, and `GeometricLSS` mass is 0 off the
  non-negative integers. `lss_simulate` keeps R's random number generators
  (#848).

- `split_refit` summaries, `effect_model` candidate comparisons and the
  `autospec` log-variance screen use shared Rust kernels, with R bindings.
  The NumPy substitutes for missing `disc_scan`, `disc_step_z` and
  `iv_ratio_weighted` extension symbols are removed; source checkouts require
  a matching native build (`maturin develop --release`) (#821).

- Exact `gamboost` route: a block learner's sparse design keeps only the column-compressed layout (peak memory 3.57 GB to 3.01 GB at 10^6 rows; fits bit-identical) (#738, #323).
- Exact `gamboost` route: a block learner's column-compressed sparse design copy is built on first use rather than at preparation; the Gaussian Gram route never reads it, so exact fits no longer retain it (peak memory at 10^6 rows back to the pre-#323 profile; fits bit-identical) (#738, #323).
- `aipw` sparse-arm check sizes its propensity bins by weight (#420).
- `mboostrs.datasets.PLANTED_INSIGHTS` removed; the answer key moved to `benches/eval_insights/answers.py` (#295).
- `reference_gap` gives the tree reference every predictor column by default (`reference_columns=`, `exclude_columns=`) and ranks candidates on the family loss (#477).
- The penalty-drift warning is shown once per process per wording; `reset_shown_summaries()` shows it again (#420).
- Benchmarks: eleven saved artifacts regenerated and stamped with their commit (#304).
- Docs: `check_docs.py` lists benchmark artifacts with a missing or stale commit (`--strict-artifacts`) (#304).
- Docs: `docs/book/artifact_policy.json` classifies every benchmark artifact as `read_by_page` (a page, notebook or release claim reads a number from it) or `historical` (a log of a past run). `check_docs.py` fails on a stale `read_by_page` artifact, an unclassified artifact, or a historical artifact a notebook loads; stale historical artifacts stay warnings unless `--strict-artifacts` is set (#811).
- `mboostrs.eval_insights` removed from the installed package; it runs from `benches/eval_insights/` (#295).
- Causal error messages: `effect_model(level=)` names `predict`; `aipw`/`iv_effect` name the missing `CausalSpec` column (#487).
- Intercept removal (`- 1`, `+ 0`) raises `InterceptControlError` in every fitting function except `glmboost`; `gam` and `gamlss` refuse what they used to fit silently (#101).
- Tests: causal friction items from #484 confirmed against main.
- `discover` spread stage names the columns it did not test; `compare_specifications` rows note a dropped main effect spanned by a tensor (#447).
- `stabsel` missing-argument error suggests a `q`/`PFER` or `cutoff`/`PFER` pair (#506).
- Docs: capability ledger re-reviewed (`last_reviewed` 2026-10-21).
- Docs: `conformal_interval` `seed` docstring describes split and tie randomization (#646, #636, #650).
- `select_capacity` reads `tolerance` off the paired fold difference against the best rung; rung records carry `fold_scores` (#500).
- `autospec` pair-form picker uses one candidate path cap (`_candidate_path_cap`) (#536, #535).
- `conformal_interval(method="split"|"cv+")` warns about under-coverage along a fitted scale (#543).
- Poisson and negative-binomial `discover` compares its smooth draft with a linear backbone (`simplify_mean=False` keeps the draft) (#660).
- `discover` REML refits hold unsupported terms at their boosted contributions (#671).
- Split `conformal_interval` defaults to `score="auto"` (count-rank score for eligible Poisson/NBinomial fits) (#636, #650, #660).
- `conformal_interval(method="cqr", mstop="tune")` continues one fit through the doubling grid (#578).
- Docs: decisions recorded on #578, #563, #543, #506, #500 and #493; no default changed.
- `conformal_interval(method="cv+")` warns about covariate-third under-coverage (#543).
- `discover` drops a predictor that copies or monotonically transforms the response and warns (#590).
- `discover` warns when elimination leaves no mean term (#586).
- CI: `tests/test_guide_symbols_exist.py` checks that guide-mentioned symbols exist; guide examples call `grouped_folds` (#468).
- Benchmarks: paired estimation-discovery instrument `paired-ab/v1.1` (#511, #527).
- Benchmarks: stopping-rule ablation `stopping-ablation/2`; earlier conclusions withdrawn (#509).
- Docs: book benchmarks intro uses Knowledge Discovery Scorecard 2.1 (#511).
- Docs: Knowledge Discovery Scorecard 2.1 (D1 split into D1a and D1b) (#511, #509).
- Docs: book benchmarks intro documents Scorecard 2.0 (#507).
- Docs: appendix `benchmark_findings.ipynb` and the chooser-on-shootout probes (#493).
- `aipw` and `iv_effect` default to `smooth=True`; `aipw(outcome_model="cv")` added; `"auto"` pools small data (#487, #712, #484).
- `effect_modifiers` runs an R-learner gate by default (`gate=True`) (#484, #711, #487).
- Interaction nomination skips `brandom` x `brandom` pairs in `discover`, `suggest_specifications` and `probe_interactions` (#447, #709).
- `validate` lists source columns read through derived columns; `GamRefit.predict` documents its return types (#447, #710, #487).
- `autospec` fits distributional candidates with `stabilization="L2"` and checks mean squared error before adoption (#718).
- Benchmarks: varying-coefficient benchmark runs on raw-frame data; dated AIPW rerun (#304, #719, #725, #720).
- Agent guide `causal-roles`: the adjustment set follows the disjunctive cause criterion; do not drop a covariate because `ci_test` found no link to treatment (#478).
- `na_action="impute_model"` no longer splits a term on a filled column into observed and filled parts; the split is opt-in as `na_action="impute_model_split"` (#467, #457).
- `aipw(outcome_model="auto")` no longer selects `"two_part"`; the two-part model logs skewed baselines and holds covariates to the training range (#456, #434).
- `effect_model(terms="auto")` keeps a candidate only if it beats a constant effect by one SE with at least 5 steps; default `max_mstop` 1600 (was 10,000) (#456).
- Agent guide recommends EconML/DoubleML with `GAMBoostRegressor`/`GAMBoostClassifier` nuisances; scikit-learn adapters refuse string columns and no longer let a feature named `y` be overwritten (#451).
- One extrapolation policy on every mean path: `extrapolation=` added to `LSSModel.params`, `predict_parameters`, `predict_all` and `conformal_interval`; `GamRefitBootstrap.predict` defaults to `"warn"` (#415).
- `heteroscedasticity` tests every numeric predictor (`columns=`, `transforms=`); `discover` gains `max_mstop=` and `extrapolation=`; factor pairs use `bols(a, by=b)` (#447).
- Agent guide `SKILL.md` split into an entry point and topics under 9,000 characters each (#429).
- A categorical learner is named by the formula term as written (`bols(f)`, `brandom(f)`) at every level count; `coef()` keys change for 3+ level factors (#429).
- `DiscoveryResult.contrast` for a dropped term comes from a REML refit of the pre-elimination model instead of an exact 0 (#415).
- `effect_model(learner="r_winsorized")` (default) clips the R-learner pseudo-outcome; `learner="r"` keeps the unbounded form (#423, #422).
- `discover` interaction adoption bar and main-effect reference changed for recall; a null draw no longer takes 91 minutes (#415).
- `aipw` and `effect_model` nuisance handling: spec-driven `propensity_clip`/`outcome_missing`/`overlap_bounds`, warnings once per call (#429).
- Causal effect intervals, overlap diagnostics and speed (#416).
- `varimp()`, bootstrap intervals and LSS prediction intervals computed in the Rust core (#407).
- `aipw` sparse arms, outcome model, step cap, learner choice and ATT with missing outcomes (#420).
- `tune_mstop` resumes each fold's path across doubling grids for LSS specifications and response scores (#189).
- Conformal intervals (`conformal_interval`) computed in the Rust core (#407).
- Tests: R parity tests fail on an R error instead of skipping (`r_call` helper) (#405, #402).
- Tests: Poisson fixed-step risk path on large counts compared with R mboost 2.9.11 (#395).
- `evaluate_distribution` reports CRPS for every family with a cdf and a quantile (#78).
- `gamlss_refit` and its posterior draws run on the rayon pool (#388).
- `autospec` shape comparison keeps the incumbent when a candidate has not converged (#127).
- Docs: records which contrast intervals are not yet implemented.
- Sparse block learners keep a column-compressed copy for faster non-Gaussian and LSS steps (superseded in part by #738) (#323). [flagged: the row-compressed copy this added was removed in #738]
- `autospec` evidence records the residual tree's held-out gain and the interaction candidates (#124).
- Records that issue #319 did not reproduce.
- `GamRefit` average contrasts evaluated through the core on one averaged row.
- `extrapolation=` covers prediction contributions, Multinomial predictions, `bspatial` coordinates and `%X%` margins (#293).
- One boundary rule for `tune_mstop` and `compare_specifications(mstop="cv")` (#314).
- `fit_budget=` on `tune_mstop` and `compare_specifications` counts the worst case of the search.
- `gamboost("y ~ 1")` and `glmboost("y ~ 1")` fit an intercept-only model with `mstop` 0 (#357).
- Docs: case study `kdd_scorecard_benchmark.ipynb` listed as the 81st study.
- Docs: `screen_interactions` documents that a noise pair's score is negative in expectation.
- `ExtrapolationWarning` subclasses `Warning`, not `UserWarning`.
- `predict(extrapolation="clip")` also clamps the `by=` variable of a spline term.
- `risk` works as an attribute and as a call on `gamboost`, `glmboost` and `gamboostLSS` models (#357).
- An empty right-hand side (`"y ~ "`) error says to write `"y ~ 1"`.
- Missing-value and unseen-level errors name the fix (`na_action=`, dtype cast).
- `predict_parameters()` defaults to the response scale; pass `type="link"` for the previous output (#326, #229).
- Small non-negative integer count columns keep a measurement term instead of a categorical `brandom` in `autospec` typing (#313).
- Slicing a `gamboost` path (`model[:k]`, `with_mstop(k)`) rebuilds the fit with one design product per learner.
- Fixed-step profiled likelihood paths report material increases in their training loss.
- Docs: research report of 19 literature-verified enhancement avenues (maintainer notes).
- Benchmarks: harness rows record status and measurement identity.
- Benchmarks: `bench_drivers.py`, `bench_shapes.py`, `bench_pairs.py` and their book pages.
- Refit term designs, penalties and `%X%` tensors built in the Rust core (`mboostrs_core::gam::TermBasis`).
- Docs: book chapter `methodology/gamlss_reml`.
- Docs: book chapter `methodology/penalized_reml`.
- Docs: book benchmark pages against the leading library in each category.
- Benchmarks: `bench_cz_agent_search.py` repeats the commuting-zone search with held-out states.
- Docs: case study `munich_rents_distributional`.
- Docs: case study `distributional_comparison` (mboostrs vs gamboostLSS, mssm, NGBoost).
- Docs: case study `specification_search_validation`.
- Docs: notebook download bundle carries each page's declared evidence files.
- `autospec` returns a `SpecRecommendation` (still unpacks as `(formula, learners, family, mstop)`).
- Benchmarks: `bench_agent_search.py` on twenty simulated processes.
- Docs: every benchmark page opens with an annotated great_tables comparison.
- `autospec` is 5 to 15 times faster; its own step length is 0.3 (`gamboost` stays 0.1).
- `autospec` writes a Multinomial model's continuous column as `bols(x) + bbs(x, center=TRUE, df=1)`.
- `autospec` family probe checks Poisson/NBinomial against Gaussian on held-out squared error.
- `autospec` relative comparisons measure risk from the saturated model for Poisson, NBinomial and GammaReg.
- `autospec` fits a right-skewed non-negative column with zeros as `bbs(sqrt(x))`.
- `na_action="impute"` predicts a missing value in a column complete at fit time at the training median.
- Gaussian fits mixing `btree` with linear terms run on the Gram route.
- `autospec` deselection is checked on held-out data.
- `autospec` recommends Gaussian for an under-dispersed count response.
- A whole-number column with more than twelve values is read as group labels only if a permutation test finds no order.
- `autospec` reads a whole-number response with median above 1,000 as a continuous positive measurement.
- `y ~ .` leaves out a constant column, with a warning.
- `GAMBoostModel.predict`, `GLMBoostModel.predict` and `LSSModel.predict` accept `newdata` as the first positional argument.
- `autospec(nu=None)`: step length 0.1, lengthened to 0.5 when risk still falls at the cap.
- `autospec(interactions="adopt")` may adopt up to six screened pairs.
- `tune_mstop` cross-validates each doubling and fits the full path once at the end.
- `gam_refit` refits `bspatial(a, b)` and `a %X% b` tensor terms with one REML parameter per margin.
- `gam_refit` refits `CoxPH` models.
- `logLik()` is available for `GammaReg`.
- `PropOdds(levels=[...])` takes ordered categories; ordered pandas/Polars responses supply their own order.
- `df`-to-lambda calibration for large bases uses a blocked triangular solve.
- `NBinomial` fits about 5 times faster and `GammaReg` about 15 times faster.
- `gam_refit` refits `NBinomial` models and honours a per-row offset; `GamRefit.predict(offset=)`.
- `confint` warns once about inner cross-validation stopping at `mstop` (`n_at_mstop_cap`).
- Exact spline and block learners keep one copy of their design (memory).
- Screened pair terms use five inner knots per margin (`bbs(x, df=2.5, knots=5) %X% ...`).
- A `%X%` tensor term is named by the term as written in names, `coef`, `varimp` and `partial_effects`.
- Docs: case studies follow one teaching structure.
- `llm_describe` default model is `nvidia/nemotron-3-ultra-550b-a55b`.
- `llm_describe` prompt asks for a Markdown reply.
- Docs: case study `agent_specification_search` rewritten.
- `autospec` infers `Gaussian` rather than `GammaReg` for a low-dispersion positive response.
- `autospec` priority list names response-like columns first and summarises missing values in one line.
- `tune_mstop` treats a minimum in the last 2% of the grid as the boundary.
- `term_report` marks a group term on group-held-out folds `measurable=False`.
- `autospec` probes complete rows; `rec.fit` defaults to `na_action="omit"`.
- Docs: 391 book headings renamed to labels.
- Docs: `check_writing.py` removed; `check_docs.py` checks citation keys and Altair output.
- `Gehan` gradient and risk use an exact O(n log n) ranked form above n = 64 (results unchanged to rounding).
- Multinomial `cvrisk`, stability selection and fitting run through the compact learner-spec path with the Kronecker lift.
- Information-criterion results name their definition and normalization; invalid family/method combinations raise.
- Comparison tables check retained responses, effective weights and shared data columns.
- `CvRiskResult.compare_to` at selected optima is descriptive; intervals require an explicit stopping point and matching folds.
- `extrapolation="error"|"clip"` act on every value outside the spline interval; the 5% threshold affects warnings only.
- Specification scoring respects evaluation horizons of structured folds.
- Docs: the book's API reference is generated from the package objects and checked in CI.
- `boost_control()` returns a `BoostControl` named tuple instead of an anonymous 4-tuple.
- Every fitting function has the signature `fit(formula, data, *, ...)`; LSS options after `data` are keyword-only.
- `LSSModel.cvrisk()` needs no arguments.
- `families=` is the documented distributional family argument; `family=` stays as an alias.
- `import mboostrs` takes 50 ms instead of 559 ms; `mboostrs.plot` loads on first use.
- Importing mboostrs no longer changes Altair's process-wide renderer.
- `altair` and `vl-convert-python` moved to the `plot` extra and `pyarrow` to the `arrow` extra.
- `aic()` and `hatvalues()` raise with a reason instead of returning `None`.
- `cvrisk` warns when the risk is still falling at the last iteration evaluated.
- A `CvRiskResult` from a `FoldScheme` reports the scheme's name.
- `fitted_` and `risk_` return read-only views of the model's arrays.
- `control`, `offset` and `selected` are properties on every model class; the call forms warn (`FutureWarning`); `LSSModel.selected(...)` became `selected_which(...)`.
- `LSSModel.offset` is a `{parameter: value}` mapping.
- Every result object takes `to_frame(output=...)` and reads as a mapping.
- Model-kind dispatch goes through `mboostrs._dispatch`.
- Broad `except Exception` handlers name what they catch.
- `summary()`, `plot_diagnostics()` and `profile_mstop()` catch named exceptions only.
- CI: Ruff runs over `tests/`, `tools/` and `benches/`.
- Tests: `pytest.raises` checks name the exception type.
- R `bols(x, df=...)` is refused (no df calibration for block specs).
- `_rust.pyi` is generated from the module.
- Tests: `cargo test --doc` runs doc tests.
- Docs: public claims corrected (a bare numeric name in `gamboost` is `bbs(name)`).
- Docs: `continue_to` described as a refit from iteration zero.
- Docs: PFER described as a bound on the expected number of false selections.
- `llm_explain_prompt` carries a mandatory non-causal caveat.
- Capability contract records R `Bbs` new-data prediction as `partial`.
- Docs: `sp500_two_lanes.ipynb` withheld from the book.
- CV+ no longer refits the full model twice.
- CV+ processes target rows in chunks of about 64 MB.
- `residuals(type=...)` names each kind and refuses where it is undefined (`"pearson"` on `Laplace`, `Quantile`, `Expectile`, `Huber`).
- `CoxPH` refuses a response residual; `type="martingale"` and `type="deviance"` added.
- `Weibull`, `Loglog` and `Lognormal` refuse a response residual.
- `Gehan`, `Cindex` and `AUC` refuse a response residual.
- `logLik()` raises `NotImplementedError` for `Laplace`, `Quantile`, `Expectile`, `Huber`, `AdaExp`, `HingeLoss`, `AUC`, `Gehan`, `Cindex`; `negative_risk()` added.
- `aic()` and `gaic_table()` refuse robust and ranking losses and mismatched likelihood measures.
- Docs: the `Multinomial` risk omits reference-class observations.
- Docs: the AFT families' risk is the negative log-likelihood of `log(T)`.
- Docs: `CoxPH` risk is the Breslow negative log partial likelihood.
- Docs: `GammaReg` shape statement (superseded: `logLik()` now re-estimates the shape, see the `GammaReg` entry above). [flagged: contradicted by the code, where `GammaReg` re-estimates the shape and `logLik()` uses it]
- Fractional weights follow upstream's `rescale_weights`, which changes the fitted path of any fit with non-integer weights.
- CV and stability-selection folds go through the weight policy column by column.
- `model.weights` is the effective weight vector; the input vector is kept separately.
- `StabSelResult.B` counts initial subsamples (pairs under `"SS"`); `n_fits` counts fits.
- Caller-supplied stability-selection `folds` are `n x B` initial subsamples on every path.
- `stabsel()` first per-fold attempt asks for 1.5 times the full-data path length.
- Block base-learners no longer keep a duplicate `X'W` copy (memory).
- Docs: case study `sp500_two_lanes.ipynb` (later withheld from the book).
- Docs: case study on the S&P 500 regime break (later withheld from the book).
- Docs: `docs/book/check_writing.py` (later removed).
- Tools: `tools/curate_us_states.py` vendors the US states GeoJSON.
- `df_to_lambda` uses `families::r_zeroin`, a port of R's `uniroot`, on R's bracket.
- Docs: case study `deposit_elasticity.ipynb`.
- Tests: R parity fixtures for the deposit-elasticity designs.
- Tests: selection-sequence parity harness against R.
- Tests: R parity for `bmono(type="quad.prog")`.
- `base_learner_names()` disambiguates learners sharing a column (`"x (bols)"`, `"x (bbs)"`); `coef`, `partial_effects` and `varimp` keys follow.
- Docs: `check_writing.py` reports pages that cite without a reference list.
- Docs: `deposit_elasticity.ipynb` prose edits.
- Docs: `deposit_elasticity.ipynb` scope paragraph.
- Docs: writing guide rule 61.
- Docs: `deposit_elasticity.ipynb` ambiguous passages rewritten.
- Docs: `deposit_elasticity.ipynb` announcement sentences removed.
- Docs: `deposit_elasticity.ipynb` opening.
- Docs: `deposit_elasticity.ipynb` headings.
- Docs: `deposit_elasticity.ipynb` no longer presents the falling sensitivity as a finding.
- Docs: `check_docs.py --strip-outputs`.
- Docs: `deposit_elasticity.ipynb` rewritten for non-statisticians.
- Docs: `monotone_five.ipynb` and `causal_constraint.ipynb` quote results through `{eval}`.
- Docs: eleven case studies report results as paragraphs.
- `stabsel()` fits each subsample only as far as its result depends on.
- `gamboost()` prepares learners on the rayon pool with the GIL released.
- Linux and Windows wheels built for `x86-64-v3` (reverted, see the portable-baseline entry). [flagged: reverted; `.cargo/config.toml` and `release.yml` build the portable baseline]

### Fixed

- R `gamboost()` stores the centering constants the core used for `brandom`
  and `bmrf` learners fitted with `center = TRUE`. It kept zeros, so
  `predict(newdata =)` omitted the centering shift and returned values
  shifted by `sum_j c_j b_j` relative to `fitted()`. R `partial_effects()`
  for these learners uses the stored centering, the same vector `predict`
  subtracts.

- `mboostrs.sklearn` estimators compare the column names of a Polars frame
  at `predict` where scikit-learn cannot read them (scikit-learn before
  1.9). A reordered or renamed frame raises `ValueError` instead of being
  matched to the training columns by position.

- `autospec()` skips a step length in its bounded step-length search when
  the fixed-step fit at that length overflows the link (Poisson at
  `nu = 1.0`), instead of raising.

- R `partial_effects()` for `brandom` and `bmrf` subtracts the shared
  centering shift from every level, matching a centered indicator design
  instead of multiplying each coefficient by its own centering fraction
  (#848).

- R `lss_pdf(StudentTLSS(), ...)` uses the Rust log-likelihood instead of
  an overflowing gamma ratio; densities and deviance contributions remain
  finite at df above 340. Python uses the same kernel (#848).

- `split_refit` retains small normal-tail p-values and rejects invalid
  split p-values. The `autospec` variance screen declines numerically
  rank-deficient designs, including duplicate columns accepted through
  Cholesky roundoff (#821).

- `LSSModel.prediction_interval(unseen_levels="marginal", group_sd=...)` works (raised `AttributeError`) (#407).
- `conformal_interval(mstop="tune")` runs on a centred or scaled response (#663).
- `discover` tunes mean stopping at the learning rate of the returned fit (#659).
- `autospec` keeps a supplied single holdout for a residual tree over coordinates (#538).
- `cvrisk_gamboostLSS` and `cvrisk_glmboostLSS` accept an integer `folds=k` (#587).
- `discover` passes its requested stopping ceiling to the tuner (#574).
- `discover` completes binary varying-curve references for numeric and boolean inputs (#572).
- `discover` supplies a baseline curve for a two-level categorical varying smooth (#560).
- Effect reports resolve composed interaction and transform recipes to the original columns (#564).
- `autospec` replays an adopted tensor or joint-tree form at the stopping iteration that justified it (#535).
- Benchmarks: paired-ab accounting reports a `brandom` term's column (#511).
- `autospec` keeps `adoption_evidence` for adopted pair forms in `report()`.
- `select_capacity` tree rung reads factor columns; tensor pairs skip columns with two or fewer values (#493).
- `screen_discontinuities` review fixes: singular backgrounds skipped, jump typing by scale invariance, fit-half imputation.
- Benchmarks: insights harness scores an `LSSModel` adopted by `autospec` (#716).
- `gamlss_refit` gives each factor level its own column (it fitted a line in the integer codes); `factor_ridge=True` adds a REML ridge on level indicators (#457).
- `gamboostLSS` accepts `bols(f, by=z)` for a multi-level factor, and `gamboost` fits it per level; finite `GaussianLSS` held-out loss; infinite-loss folds reported as failed (#415).
- Distributional fixes: `tune_mstop` on a mapping returns its model; `conformal_interval(method="distributional")` re-estimates transforms; `cvrisk` accepts a flat LSS grid; `gamlss_refit` supports intercept-only parameters (#415).
- `gamboost(...).confint(method="bootstrap")` works with a string factor (default grid is each replicate's training rows) (#429).
- `discover` no longer crashes on a log-drafted predictor with missing values; interaction recall improved (#415).
- Causal benchmark rerun 3 fixes (items 5-10) (#429).
- Causal benchmark rerun 2 fixes (items 2, 3, 4, 6, 8, 9) (#423, #424).
- `confounding_benchmark` refits each estimate with the result's resolved outcome model, interactions and step caps.
- Causal benchmark rerun fixes (items 5-9) (#420).
- `compare_to_r(family="Binomial")` passes the response to R as a two-level factor (#402).
- Tests: R parity test for `gamboostLSS` trees uses matching `btree` selection settings (#401).
- Benchmark rerun defects (items 7-18), including `sensitivity` building `<col>_missing` itself (#388).
- `discover` handles missing values and skewed covariates (#388).
- `autospec(budget=0)` no longer recommends a Poisson step that diverges on large counts.
- `autospec` writes a coordinate pair in column order (`bspatial(x, y)`) (#338).
- `sensitivity(model, data, parameter=...)` holds the other parameters' columns at their means (#375).
- `gamlss_refit` stops a stalled inner Newton solve after three steps without gain.
- `gamlss_refit` converges on small samples with a null smooth in sigma (#374).
- `gamlss_refit(...).predict`, `.effects` and `.draws` read string factor labels.
- `confint(method="refit")` and `gam_refit` work with string `brandom`/`bols` factors.
- `LSSModel.prediction_interval(newdata=...)` no longer applies the factor recipe twice.
- `conformal_interval` works with string factors in every method.
- `predict` no longer averages every row when factor levels differ from the fitted ones only in dtype (#334).
- Restores three fixes reverted by the #344-#346 merge batch (#320, #322, #275, #319, #345).
- `sensitivity` reports a `bols(<string column>)` term under the source column name (#309).
- `pair_term_for` accepts `data=` and no longer proposes spline terms for string columns (#336).
- `autospec` no longer aborts when a candidate family's fit raises a Rust panic (#319).
- Restores btree histogram split finding and bspatial Gram-reuse work reverted by a stale file in the hinge learner commit (#341, #318, #287, #284, #288, #299).
- `autospec` no longer raises `cannot parse formula term` for `bols(log(x))` from its shape probe (#320).
- `gamboostLSS` reports a one-covariate `btree(x)` covariate as a vector in `partial_effects` (#324).
- `compare_specifications` reads a whole formula whose left-hand side is the response (#327).
- `compare_specifications` no longer crashes on a plain list of distributional candidates (#329).
- `scan_lss_families` refuses a bare right-hand side once before fitting and reads an integer `folds` as k-fold (#330).
- Without the `plot` extra, `hasattr(mboostrs, "plot")` is `False` and `inspect.getmembers(mboostrs)` works (#308).
- `insights()` accepts Polars frames and honours `family=` in its own fits (#305, #307, #309).
- GLM grid predictions without an intercept use the ordinary learner numbering; R GLM prediction replays `scale()` and other transforms.
- `GamRefit.predict(unseen_levels="zero_effect")` applies the policy to held boosted terms.
- `gam_refit` refits `bols(x) + bbs(x, center=TRUE)` as one smooth `bbs(x)`.
- `cvrisk`, `stabsel` and `deselect` on a weighted model multiply the model's weights into a 0/1 fold mask (fold refits were unweighted).
- R `cvrisk()` on a weighted `gamboost` model returns the cross-validated risk with weighted held-out rows (it refused).
- REML refits identify a main effect nested in a pair term (`bbs(x1) + bbs(x1) %X% bbs(x2)`).
- `gamboostLSS` names a `%X%` pair by the term as written, not `__ix0`.
- `plot_lss_selected` no longer shifts each bar's label by one.
- Regression tree fits allocate row buffers once per tree.
- Benchmarks: insight page computes top-5 stability per data set and method.
- `deselect` on a `gamboostLSS` fit with a `%X%` pair returns a model that predicts new rows.
- `gam_refit` and `gamlss_refit` accept a one-column term without an intercept.
- `autospec` measures quasi-Poisson for a non-negative non-count response with zeros.
- `autospec` uses the line-search step for Gamma, Poisson and NBinomial responses with large values.
- `varimp` for Multinomial credits each step with its change in the full negative log-likelihood.
- R `varimp` sums two learners of one name.
- `autospec` suppresses count-family warnings for class-valued families.
- `bbs(x, center=TRUE)` is honoured in Multinomial models.
- R `predict()` on new data for a Multinomial model with `bols(x)` lays out the design row by row.
- `autospec` builds an adopted pair over a log-scale column on the log scale.
- `gam_refit` accepts a `bols` term over a string column.
- `autospec(interactions="adopt")` proposes `brandom(g) %X% brandom(b)` for a group and a two-level label.
- The line-search step rule can take steps below 0.125 of the least-squares direction.
- The line-search step rule no longer scales a tree learner's encoded coefficient vector.
- `autospec` interaction screen floor is 1% of the residual variance.
- A `%X%` term over string factors predicts new rows.
- `predict(newdata, which=)` applies the transform recipe once.
- `autospec` needs at least 10 levels before a factor sets the fold design.
- `log(x)`/`sqrt(x)` terms hold a value below the training range at the training minimum, with an `ExtrapolationWarning`.
- `autospec` retries probes with `step="line_search"` when the risk becomes non-finite.
- `autospec` shape probe also switches for a gain of two paired fold standard errors.
- `cvrisk` and `stabsel` keep a per-row offset in every fold.
- `autospec` tunes depth before comparing families and shapes.
- `gam_refit` accepts a model fitted from `y ~ .`; a bare numeric name is `bbs`.
- R `predict(newdata=)` for block-form learners (`bols(x, lambda=)`, `brandom`) passes the design row-major.
- `PropOdds` with four or more levels fits (thresholds no longer cross).
- `cvrisk` for `Multinomial` scores held-out rows by the full negative log-likelihood.
- `tune_mstop` doubles its grid up to `max_mstop`.
- `btree(x, by=z)` grows its tree on the rows where `z` is 1, as mboost does.
- `nuisance()` returns the `NBinomial`/`GammaReg` dispersion or shape at `mstop`; slicing keeps it.
- `conformal_interval(method="cqr")` no longer inverts the band on skewed noise.
- Docs: the PDF book builds again (figure height limit).
- `gam_refit` REML search avoids singular penalized Gram matrices.
- `LSSModel.cvrisk()` on a `method="noncyclic"` fit cross-validates the total step count.
- `autospec` interaction screen forms its residual on the response scale.
- `predict(newdata=, which=)` for an unselected learner returns an array of the right length.
- `na_action="omit"` works on a pandas frame.
- `confint(newdata=)` works on a pandas frame.
- `cvrisk_gamboostLSS` and `cvrisk_glmboostLSS` default `grid` to the diagonal grid.
- Partial-effect plots and `partial_effects` label the link scale ("half log-odds" for Binomial).
- Docs: `extract('lambda')` returns the calibrated lambda for block learners.
- A text-label response column is refused with the column named on every fitting route.
- Docs: static HTML renders `display(chart)` Altair outputs as SVG.
- `plot_partial_effects` draws partial residuals.
- Distributional fits with per-parameter learning rates replay the path at those rates.
- `autospec(..., nu=...)` recommendations fit with the given rate.
- `datasets.munich_rent` docstring states euros.
- `term_report` measures a group term on group-held-out folds.
- Paired standard errors treat `repeated_kfold_folds` repeats correctly.
- `autospec` interaction lead for an integer group column is `brandom(g, by=x)`.
- `stabsel` on a model with a `%X%` interaction reports selection frequencies.
- `model[:k]` keeps the interaction recipe.
- `partial_effects` on a `%X%` model works on a grid and on `newdata=`.
- `grouped_folds` accepts an object array of Python strings.
- `heteroscedasticity` and `sensitivity` read both coordinates of a spatial learner.
- `gamboost(..., learners=..., na_action="omit")` drops rows over every column read.
- `autospec` with a fit budget handles a missing value.
- `compare_specifications`, `evaluate_specification` and `cv_risk` score `brandom` on grouped folds.
- `partial_effects` and `predict(which=)` include the step length for `buser` and exact `bkernel` learners.
- Docs: `continue_to` contract matches the implementation.
- R `cvrisk()` for `gamboost` refits folds on in-bag weights.
- `update()` and `model[:k]` on a distributional model keep stabilization and per-parameter rates.
- A weighted Multinomial fit charges each observation its own weight.
- Binary response encoding moved to the Rust `Family` contract; `AdaExp` and `HingeLoss` accept `{0, 1}` and `{-1, +1}`.
- Native diagnostic endpoints validate shapes and values and return `ValueError`.
- Invalid Dirichlet parameter counts return a typed error (`DirichletLSS::try_new`).
- Native B-spline replay helper validates inputs and accepts strided arrays.
- A learner specified by `df` no longer fits nothing when an outlier leaves a knot interval empty.
- `autospec` sends numeric 3-12-level columns to `brandom`, not `bols`.
- A `buser()` lattice design with an empty group no longer aborts.
- `deviance_profile` bins rows the same way in Python and R.
- `BCPELSS` scores stay finite when a response equals its fitted `mu`.
- The bmono constraint solvers no longer allocate per candidate solve.
- Conformal results report method-specific guarantees, bounds and assumptions.
- Refit specifications distinguish fitted-design replay from retraining preprocessing; conformal splits learn preprocessing on proper training rows.
- Docs: corrected book claims about cross-validation, stability selection, shape constraints and survival.
- Zero-adjusted Gamma density and CDF are zero below the support; PIT randomizes the atom at zero.
- Distributional mean calibration uses the family's response mean (zero-adjusted Gamma, Gumbel).
- AIC plots use one-based iterations.
- Interval coverage and censoring weights validate their inputs.
- `cv_risk(score="mse")` scores response-scale predictions (Brier score for Binomial).
- LSS `cvrisk()` and `stabsel()` refits honour the family's `stabilization`.
- Linux and Windows wheels use the portable x86-64 baseline again (the x86-64-v3 build could terminate with `SIGILL`).
- scikit-learn estimators compare column names, not positions; a reordered or renamed frame raises.
- `TreeBoostRegressor` and `TreeBoostClassifier` read the caller's column names.
- scikit-learn estimators set `feature_names_in_` only when the input had names.
- A scikit-learn regressor no longer carries `n_classes_ = None`.
- `confint()` draws bootstrap resamples from the model's weights, as `confint.mboost` does.
- `continue_to`, `confint` and stability-selection refits go through the model's `RefitSpec`.
- `refit_spec` no longer passes `learners=` and `env=` together.
- `cvrisk` and `stabsel` on a loaded artifact refuse by name.
- `classify_columns`, `suggest_terms`, `term_report`, `cv_risk` and `compare_specifications` read string, boolean and date columns.
- A boolean column is an indicator; `bols(flag)` fits.
- A date column is a continuous axis and fits with any learner.
- A two-level factor fits with `bols`.
- `cv_risk` scores with the family's own loss by default.
- `compare_specifications` accepts a plain list of right-hand sides.
- `term_report` shape search tries only shapes that fit.
- `gamboost().confint()` counts a replicate that selects no learner as a zero effect, as mboost does.
- `gamboost().confint(parm=...)` rejects an unknown learner name.
- `confint` counts and reports failed bootstrap replicates.
- R `coef()` on a distributional model applies the centering correction.
- `stabsel` on a Multinomial model refuses instead of crashing.
- R `predict(newdata=)` works for centered smooths (`bbs(center=TRUE)`, `bspatial`).
- R `r_bspatial_design` applies its `center` argument.
- R penalized `bols(x, lambda=)` matches mboost.
- Mixing `bols(x, lambda=)` and plain `bols(z)` in one formula matches mboost in both frontends.
- `llm_explain_prompt` no longer claims every effect is centred.
- `llm_explain_prompt` describes the AFT predictor as a location on the log-time scale.
- `llm_explain_prompt` rejects a `cv=` or `stabsel=` result from a different model.
- A formula term is parsed whole or rejected; unparsed text inside a learner call raises.
- Boolean learner arguments reject non-boolean text (`center=maybe`).
- A numeric learner argument that does not convert names the learner and argument.
- `glmboostLSS(env=...)` is rejected instead of ignored.
- Conformal calibration refits the model that was fitted (weights, `center`, subsampling kept).
- Conformal calibration warns when case weights break exchangeability.
- Conformal calibration refuses every fit it cannot validly subset (`Bmrf`, constraint matrices, interaction recipes, subsampling).
- `residuals()` and `residual_frame()` compute the same quantity.
- Docs: `PropOdds` documents its `{1, ..., K}` coding.
- Docs: `Multinomial` documented as `gamboost`-only.
- `family_diagnostics` applies the family's response encoding.
- Every tabular helper honours `output={"auto","polars","pandas","records"}`.
- `q_stats` has one implementation.
- `calibration_frame` works on a distributional model (`location_mean`).
- `LSSModel.worm_data()`.
- Invalid family parameters raise at construction (`Quantile(qoffset=2)`, `Huber(delta=-1)`, `PropOdds(n_levels=1)`, `Multinomial(n_classes=1)`, `Cindex(sigma=-1)`).
- Family parameter checks run in constructors, binding dispatch and artifact deserialization.
- `Multinomial(n_classes=3.5)` is refused instead of truncated.
- `Btree(names=...)` requires at least one variable and no repeats.
- Every scalar-name learner requires one non-empty string name.
- `Buser` `X`/`K` and `Bconstrained` `C` accept any 2-D array-like and are checked.
- `Bbs(boundary_knots=...)` must be two finite increasing values.
- R `Quantile()` takes `qoffset`.
- All-zero weights are rejected.
- The LSS weight-length error uses the shared message.
- `stabsel_glmboostLSS` and `stabsel_gamboostLSS` fit complementary pairs under `sampling_type="SS"`.
- R `stabsel()` refits on in-bag weights.
- Cyclic and non-cyclic `gamboostLSS` fits stop when the joint risk explodes.
- `ExecutionReport.retained_learner_bytes` reports on LSS paths.
- `gamboost(..., bins=)` keeps exact storage for terms that cannot be binned.
- `stabsel` "mstop too small" warning gives counts and remedies.
- Packed Gaussian scalar fits report risk without the ridge penalty term.
- Docs: CV risk chart in `cz_inequality.ipynb` renders.
- The Gaussian Gram fast path breaks exact selection ties by lowest index.
- R `partial_effects()` for `center = TRUE` splines.
- R `partial_effects(fit, which=)` returns the requested learner.
- `partial_effects()` for multi-column `btree` or `brad` learners.
- `bols(x, by=z)` at the default `intercept=True` fits `[z, x*z]` as R does; penalized `bols` uses the two-column form.
- `bols` penalizes every column, as R's `X_ols` does.
- A pickled model with transforms or `%X%` interactions predicts on new data.
- `%X%` honours marginal `df=`; tensor `df` is the product of the marginal `df`s.
- A `df` at the null-space dimension no longer diverges.
- `df`-to-`lambda` calibration reads the penalty spectrum through singular values.

### Detailed notes

The entries as written when each change was made, newest first. The
entries from before 11 October 2026 were dropped from this file by the
conflict resolution in merge `e7661f5b` (PR #479) and are restored here
from `e287d477^1`. Their `### Added`/`### Changed`/`### Fixed` headings
are folded into the summary above.

- Comparison identity: reference measure, scoring rule and a common sample
  for `scan_lss_families` (#60). Tests: `tests/test_comparison_identity.py`.
  - **Added:** `mboostrs._likelihood.reference_measure(family)` names the
    measure a likelihood is a density with respect to: `counting` (Poisson,
    Binomial, the count LSS families), `lebesgue` (continuous families),
    `zero_atom_lebesgue` (`ZeroAdjustedGammaLSS`) or `partial` (`CoxPH`).
    Fit identity records carry it, and a `risk_scale` field naming the scale
    of the risk.
  - **Changed:** `gaic_table` refuses a count family next to a continuous
    family of the same response (it ranked `PoissonLSS` against
    `GaussianLSS`); `compare_models` and `scan_lss_families` show those rows
    unordered with a `comparison` reason.
  - **Changed:** `compare_cv` and `CvRiskResult.compare_to` raise
    `ValueError` for two curves scored by different rules, such as Gaussian
    against Laplace or two quantile levels; they returned a paired difference
    of the two losses.
  - **Added:** `scan_lss_families(na_action="omit")` fits and scores every
    candidate family on the rows complete in every column any formula entry
    reads, and reports per row how many rows its own formulas would have
    kept. A fold matrix is restricted to those rows.

- Exact `gamboost` route: a block learner's sparse copy of its design now
  keeps only the column-compressed layout (#738). The row-compressed layout
  added in #323 duplicated the same entries at 52 bytes per row for a cubic
  `bbs` term; `S β` is now formed from the column layout, with the same
  floating-point operations in the same order, so fitted values are
  bit-identical. Peak memory footprint of ten `bbs(x, df=4)` terms at
  10^6 rows, 500 steps, one thread (release build, M2): 3.57 GB -> 3.01 GB
  (2.50 GB before #323). Probe:
  `cargo run --release --example mem_prepare -- 1000000 10` under
  `/usr/bin/time -l`: 3.25 GB -> 2.69 GB.

- Exact `gamboost` route: a block learner's sparse copy of its design is
  built on first use instead of at preparation (#738). The Gaussian Gram
  route never reads the copy — it works from the dense design and the
  cross-Grams — so every exact Gaussian fit retained about 46 MB per
  `bbs` term at 10^6 rows for an index it never touched. The copy is now
  held as a lazily built cache and materialises only when a general-path
  step or a prediction first needs it, built from the same stored
  design, so products through it are unchanged. Learners whose design is
  centred at preparation keep the eager build: their copy is taken from
  the pre-centring design, which is not retained. Peak memory footprint
  of ten `bbs(x, df=4)` terms at 10^6 rows, 500 steps, one thread
  (release build, Linux x86_64): 2,699 MB -> 2,252 MB, against 2,235 MB
  measured the same way at the post-G-12 commit `5c8236ef`. Probe:
  `cargo run --release --example mem_prepare -- 1000000 10` under a
  peak-RSS timer: 2,576 MB -> 2,118 MB, against 2,118 MB at `5c8236ef`.

- Automated modelling procedure, PR 1 scaffolding (issue #474). Tests: `tests/test_procedure_trail.py`.
  - **Added:** `docs/schemas/decision_trail.schema.json` (JSON Schema, draft 2020-12): the decision trail of one `fit` — provenance, data, folds, criterion, steps, final model, uncertainty and budget — exactly as section 5 of `docs/design/unified-procedure.md` specifies.
  - **Added:** `mb.fit(data, response, ...)` returns a `Fit` wrapping the chosen model, its `trail` and the uncertainty step. PR 1 is scaffolding: the specification is delegated to `autospec` and the trail records one `autospec_delegate` step plus the final formula, family and mstop. `budget` is validated against `quick`/`standard`/`thorough`; `uncertainty="stability"`/`"bootstrap"` are recognised and raise `NotImplementedError` until PR 10.
  - **Added:** `mb.report(fit, format="text"|"markdown"|"json")` renders the trail. It never fits anything.
  - **Added:** `mb.DecisionTrail` builds, serialises and (with `jsonschema`) validates the trail. Two runs with the same seed, data and version produce bit-identical trails — no timestamps.
  - **Added:** seeding behind the procedure: `mboostrs._procedure._seeding.streams(seed)` derives the folds/inner/btree/stability streams from `numpy.random.SeedSequence`.
  - **Changed:** `mb.report` is now the trail-rendering function. The `mboostrs.report` execution-report submodule stays importable and unchanged (`from mboostrs.report import ExecutionReport`); the top-level attribute follows the `insights` precedent.
  - **Unchanged:** `autospec`, `gamboost`, `glmboost` and the rest of the public API. The full search (steps 1–7b of the design note) lands in later PRs.

- `rearrange_quantiles(models, newdata)` sorts the predictions of several
  `Quantile` (or several `Expectile`) fits into increasing order in each
  row, the rearrangement of Chernozhukov, Fernández-Val and Galichon (2010),
  and reports the crossings before sorting: the share of rows with a
  crossing, the count for each adjacent pair of levels and the largest
  crossing (#103). Rows without a crossing are unchanged. The sort runs in
  the Rust core (`mboostrs_core::rearrange::rearrange_levels`), and R has
  the same function. The CQR interval in `conformal` now orders its two
  quantiles through the same core function.

- `aipw`'s sparse-arm check sizes its propensity bins by weight (#420).
  It uses one bin per 200 units of total weight, between 4 and 10 bins,
  so a 1,500-row design gets 7 bins instead of 10; each bin holds equal
  total weight. Arm counts, the 20-unit floor and the 5% target-share
  rule are weighted sums, so integer weights give the same flags as
  replicated rows. With 10 bins, many small designs had too few rows of
  one arm in the extreme bins and were flagged.

- The planted-insight answer key moved from `mboostrs.datasets` to
  `benches/eval_insights/answers.py`, so `mboostrs.datasets` no longer
  has `PLANTED_INSIGHTS` (#295). The generators `kinked_threshold`,
  `interaction_surface` and `spatial_regime` return the same data as
  before, and the eval harness scores are unchanged.

- `reference_gap` gives the tree reference every predictor column of the
  data by default, and with `time=` the lag and rolling features too
  (#477). `reference_columns="formula"` restores the earlier behaviour,
  a list of names selects columns, and `exclude_columns` keeps columns
  such as weights or ids out. The result lists `reference_columns` and
  `reference_extra_columns`. The top `top` candidates are measured
  whether or not the gap is material, and columns the formula omits are
  candidates. Candidates are ranked on the family loss: the screen runs
  on the negative gradient of the loss at the out-of-fold linear
  predictor, scaled by the square root of its curvature. For Gaussian
  fits this is `y - fitted`, so their ranking is unchanged.

- The penalty-drift warning (`bbs(x, df=4)` learners that boosting has
  undone much of the penalty of) is shown once per process for each
  wording, so a script that makes many top-level `gamboost` calls sees it
  once (#420). `collect_warnings()` still records every instance, and
  `reset_shown_summaries()` shows it again.

- `AIPWResult.predict_propensity(newdata)` scores new rows (#487 item
  10). It returns P(treatment | covariates) clipped to the fit's
  `propensity_clip`, from a propensity model `aipw` now refits on all
  rows with the formula, smooth flag and step count of its cross-fitted
  propensity fits. Its values on the training rows differ a little from
  the cross-fitted `propensity` field, which the estimate uses. The extra
  fit adds about 7% to a 5,000-row `aipw` call with default settings.

- Eleven saved benchmark artifacts that recorded no commit, or a commit
  absent from history, were regenerated on a release build and now record
  the commit they were produced at (#304): beta, gamma location-scale,
  shapes, spatial, survival, location-scale drivers, term choice, cap
  effect, insights, the desk-ops exceptions case and the two-standard-error
  zone search. The term-choice, insights and desk-ops pages changed: the
  tuned electricity score for `menu + tree pairs` rose from 0.8911 to
  0.9084, `autospec` on California housing adopted a distributional model
  on two of five splits, and the desk-ops search adopted
  `bbs(vendor_n, by=has_legacy_stack)` instead of
  `btree(complexity_score)`. Three benchmark generators were corrected to
  work with the installed scikit-learn and the raw-frame data sets. Six
  artifacts remain without a commit: the libraries comparison and its
  clipped copy and the interaction search and its second version each take
  more than 45 minutes, the varying-coefficient benchmark needs rewriting
  for raw-frame data sets, and the interaction-usage table has no generator.

- `docs/book/check_docs.py` lists each saved `bench_*.json` whose
  recorded commit is missing, is not in the history of HEAD, or is
  followed by commits that changed `python/mboostrs` or
  `crates/mboostrs-core`, with the book pages that load it (#304). The
  listing is a warning; `--strict-artifacts` makes it fail. The
  documentation CI job now fetches the full history so the commits can
  be found.

- The planted-insight eval harness moved from `mboostrs.eval_insights` to
  `benches/eval_insights/` and is no longer in the installed package
  (#295). Run it as `python -m eval_insights` from `benches/` in a
  source checkout.

- `LSSModel.prediction_interval(unseen_levels="marginal", group_sd=...)`
  raised an `AttributeError` because its helper was never written (#407).
  The helper now exists. Rows whose `brandom` group the fit never saw get
  the zero-effect location, which is the mean of the assumed group effect,
  and their scale widens from `sigma` to `sqrt(sigma^2 + group_sd^2)` for a
  `GaussianLSS` fit. Rows whose group the fit saw keep their conditional
  interval. `group_sd=None` still raises the `ValueError` that names the
  argument.

- Causal messages from #487. `effect_model(level=...)` raises a `TypeError`
  that names `level` as an argument of `predict` and shows the call
  (`predict(newdata, interval=True, level=0.9)`); other unknown keywords keep
  the plain message. `aipw` and `iv_effect` with a treatment, outcome or
  covariate that is not a column of `data` raise a `ValueError` naming the
  `CausalSpec` argument and the column (`covariates='nope'`). The extrapolation
  warning shows a `bbs(x, df=4, boundary.knots=c(lo, hi))` term as
  `bbs(x, df=4) fitted on [lo, hi]`; the fit is unchanged. The `aipw` docstring
  lists the keys of `diagnostics["overlap"]`, which is a plain dict. Not
  changed in this entry: `AIPWResult.propensity(newdata)` (the result keeps
  cross-fitted propensities only, with no model to predict new rows).

- Intercept removal (`- 1`, `+ 0`) is refused with a `ValueError` by every
  fitting function except `glmboost` (#101). `gamboost`, `gam`, `gamlss`,
  `gamboostLSS` and `glmboostLSS` name `glmboost` as the only function that
  removes the intercept, and say what they do with it instead. `gam` and
  `gamlss` used to fit silently with an intercept for `+ 0`; they now refuse
  it. The refusal is an `InterceptControlError`, a subclass of `ValueError`
  that is also a `NotImplementedError`, so existing `except
  NotImplementedError` clauses still match. `glmboost` is unchanged.

- Causal friction from #484, checked against current `main`. Items 1, 2, 3
  and 6 were already fixed by earlier changes (`iv_effect` accepts
  `outcome_missing`, `ci_test` names the column to drop and reports NaN for
  an exact dependence, warnings show the user's column name, and
  `causal_trap_report` does not block on an absent `post_treatment` column);
  their tests are in `tests/test_causal_484_friction.py`. Item 4 and item 5
  were confirmed with new end-to-end tests: `effect_model(terms="smooth")`
  reports `gain_over_constant` and `gain_se`, and each warning text is issued
  once per call. Item 9: the `aipw` docstring and the causal-average guide
  state that `level` defaults to 0.95. Items 7 and 8 are not addressed.

- `discover`'s spread stage now names the frame columns it did not test as
  spread candidates, with the reason for each (for example a categorical
  column), in its log entry for the stage. The spread test already covers every
  predictor of the data, not only the mean model's, so the report states which
  columns were left out instead of a mean-model limit (#447). The rest of
  #447 was already in place on main: the boundary message names
  `discover(max_mstop=...)`, and `discover` issues one extrapolation warning per
  call that names `discover(extrapolation=...)`. A `compare_specifications`
  row that drops a main effect spanned by a tensor carries a `note` (#447).

- `stabsel`'s missing-argument error now guides the caller instead of only
  refusing (#506). The two-of-`cutoff`/`q`/`PFER` requirement stays at
  mboost parity, as the issue's recorded decision keeps it; the error now
  names the model's effect count and a copy-pasteable mboost-style pair --
  `q` about `sqrt(0.8 * p)` with `PFER=1`, or the usual `cutoff=0.75,
  PFER=1` -- so the failed call and retry turn the D5 runs spent on this
  trap is no longer needed. A single-effect model, which has no `q` below
  the effect count to suggest, is guided to the cutoff pair alone.

- The capability ledger's `last_reviewed` moved to 2026-10-21 after
  re-review of the 2026-10-19 change (`c58b7ee`): the change sorts the
  exported-name lists and adds three genuine entries (`dump_result_json`,
  `load_result_json`, `imputation_diagnostics`), each verified against the
  live package; no status table moved. The derived `capabilities.ipynb`
  appendix is regenerated with the new review date, and the evidence check
  passes again.

- `conformal_interval`'s `seed` docstring now states the whole job the seed
  does: it keys the split or fold assignment and, when the default
  `score="auto"` resolves to the count-rank score, the score's tie
  randomization -- one uniform per calibration row and per query row from a
  stream keyed by the seed. An internal split and its equivalent held-out
  calibration agree exactly only when both carry the same seed; `seed=None`
  (the default) draws and reports the randomization seed in
  `randomization_seed`, so the call is not reproducible until a seed is
  given. The #646 agreement test predates the count-rank score (#636, #650)
  and compared the two routes across different seeds, which is why it failed
  on main; it now passes the same seed to both routes and also pins the
  resolved score's equality.

- `select_capacity` reads its `tolerance` off the **paired fold difference**
  against the best rung, not the best rung's marginal fold spread (#500).
  Every rung is scored on the same folds, so differencing two rungs' per-fold
  scores cancels what the folds share -- on rolling folds, the period each
  fold covers -- which is the same rule `coordinate_beats` applies to
  candidate forms. The old marginal reading let a few time-ordered folds'
  period heterogeneity set the bar: on the PM2.5 panel's five rolling-origin
  folds the default two-standard-error bar sat at about 14% of the held-out
  score, inert against real capacity gains, while the paired differences put
  it near 2% (measured on the panel: the base-vs-best paired standard errors
  are 0.0022-0.0031 against a marginal 0.017). The panel still does not adopt
  a larger rung because its rungs remain within the paired bar, and
  the default `tolerance=2.0` is unchanged. Each rung record now carries
  `fold_scores`, each fold's score at the step count the rung was scored at,
  and `cv_risk(mstop_grid=...)` rows carry the same per-fold values, so a
  reader can recompute every paired difference. When the folds cannot be
  paired -- one fold, or a failed fold on either side -- the marginal spread
  decides, as before.

- The autospec pair-form picker no longer caps its competing forms at twice
  the additive incumbent's own optimum (#536). The candidate stopping search
  now follows one policy, `_candidate_path_cap`: floored by the caller's
  requested `mstop` and the 200-step floor every other stopping search in
  `autospec` uses, grown from the incumbent when the incumbent itself needed
  a long path, and ceiled by `_MSTOP_EXTENSION_CAP`. A weak additive model
  converges early (the frozen joint_boundary draws stopped at 23-46 steps
  while the caller asked for 200), and the old `min(2 * chosen_mstop, cap)`
  judged every interaction candidate on a path it had not finished using --
  four of five frozen draws adopted a form whose risk was still falling at
  its cap. The cap sets path length only, so the picker's fit count is
  unchanged (one fit per candidate per fold) and the budget guard is
  untouched. Replayed on the five frozen joint_boundary draws (budget 32,
  one validation fold, mstop=200), each draw improves or ties: test MSE
  0.474/0.393/0.437/0.340/0.328 before against
  0.277/0.393/0.428/0.323/0.305 after, mean 0.394 -> 0.345, at the same 10
  fits per draw; seeds 3 and 4 still sit at the 200-step boundary, reported
  as such and not claimed converged. The #535 replay evidence and artifacts
  are unchanged.

- `conformal_interval(method="split")` and `"cv+"` report under-coverage
  along a **fitted scale**, not only along covariate thirds (#543). The
  constant-width bands look right on average while the high-scale rows
  undercover -- on the calibration study's heteroscedastic draw the top
  true-scale third covered 0.78 against a nominal 0.90 -- and when the scale
  is a combination of covariates (multiplicative noise, several weak drivers)
  no single covariate's thirds isolate it. The warning now also bins the
  calibration rows' coverage by a scale model fitted to them -- `log(residual^2)`
  regressed on the numeric covariates by direct REML -- and reports a third
  more than four binomial standard errors below nominal, in the same message
  as the covariate thirds. Measured on the study's draws: the heteroscedastic
  DGP fires 9/10 (as before), the homoscedastic and heavy-tailed draws stay
  silent 0/10, and a multiplicative DGP whose scale no covariate third sees
  fires through the new view alone. The default method and the intervals are
  unchanged.

- `select_capacity` searches each rung's step count out of fold before ranking
  it, closing the shared-budget gap the shootout battery traced (#493, item 3:
  joint_boundary +249% vs the hybrid's +65% with the risk still falling at
  the cap). The new default `budget="tuned"` scores every boosting rung at the
  step count its own held-out score turns at: the grid starts at `mstop`,
  doubles while the score is still falling at its largest point and never goes
  past `max_mstop` (new, four times `mstop` by default). `budget="shared"`
  keeps the original equal-step contract of #479. `tree_depth` accepts a
  sequence such as `(3, 6)` and scores one `+tree` rung per depth, keeping the
  smallest depth within `tolerance` of the best. Each rung record carries the
  `mstop` it was scored at, a rung still falling at the cap is warned about,
  and the selected model is refitted at its rung's tuned count. `cv_risk`'s
  `mstop_grid` route now scores the response losses (`mse`, `brier`,
  `log_loss`, `pinball`, `expectile`) along the grid with the same recoding a
  single-`mstop` call uses, not just the family loss.

- Ordinary Poisson and negative-binomial discovery compares its generated
  smooth mean draft with a linear backbone before dropping variables (#660).
  The simpler form is retained within two paired fold standard errors;
  `simplify_mean=False` retains the previous draft. Mandatory terms and
  transformations are preserved, and learning rates and Gaussian scale rules
  are unchanged. The complete frozen count confirmation and regression
  controls are reported in the count mean specification benchmark.

- Discovery's optional ordinary REML refits hold unsupported terms, including
  trees, at their boosted contributions when supported terms remain (#671).
  The log labels this conditional refit. Discovery contrasts then report the
  boosted estimate without an interval, because the refit estimates no
  uncertainty for the held contributions. All-tree and unsupported
  distributional refits are explicitly unavailable; boosted predictions and
  mean/scale selection are preserved.

- `dump_result_json` and `load_result_json` provide strict, reversible result
  JSON with explicit categories for positive/negative infinity and NaN,
  distinct from genuine missing values (#665). `ConformalInterval.to_json`
  exposes the same format. Existing dictionary exports retain their shapes
  and numerical floats for backward compatibility.

- Split `conformal_interval` now defaults to `score="auto"`. Eligible standard
  Poisson/NBinomial fits use a reproducible randomized count-rank score with a
  mean-preserving reference fitted on proper-training rows; a prespecified
  dispersion guard uses normalized scoring when needed (#636, #650). The
  Python and R results report score/reference/seed metadata and nonnegative
  integer cardinality, including empty sets. Explicit legacy scores retain
  their definitions. Two frozen confirmation cohorts meet the original-rate
  target; high-rate fixed-mean and zero-inflated count-size gaps remain (#660).

- `conformal_interval(method="cqr", mstop="tune")` continues one fit through the doubling grid and keeps each fold's fit and predictions between grid ends, instead of refitting every fold at every doubling (#578). The intervals are unchanged: the same grid ends and optimum are used. Measured at n = 10,000, the tuned CQR is about 1.3x faster where the search doubles, and unchanged where it stops at its first grid.

- `conformal_interval(mstop="tune")` runs on a centred or scaled response. The step count is cross-validated with the transform learned on the refit's rows, shared by every fold, instead of relearned per fold, which the cross-validation refused (#663). Unscaled responses are unchanged.

- `discover` now tunes mean stopping and estimates its tuning cost with the same learning rate used by the returned fit. Non-Gaussian models previously tuned at 0.1 and refitted at 0.02, so the selected stop described a different path (#659).

- Documented the decisions on #578 (CQR cost), #563 (discover's grid cap), #543 (the split default), #506 (stabsel defaults), #500 (select_capacity tolerance) and #493 (select_capacity budget). No default changed.

- `conformal_interval(method="cv+")` warns when its interval covers a covariate's
  middle or extreme third far below nominal, as split conformal already did (#543).
  The default method and its intervals are unchanged.

- `GamlssRefit` has a `summary()` method, listing each distribution parameter's terms with
  their effective degrees of freedom and REML smoothing parameters (#506). Other fit objects already had one.

- Autospec keeps a supplied single holdout when it validates a residual tree over
  coordinate predictors (#538). It no longer builds a grouped spatial scheme with
  one fold, which raised "B must be at least 2". Multi-fold designs are unchanged.

- Discovery reports the fitted scale next to the mean (#589). When the
  analysis adopts a spread model, `result.predictions` now carries a `sigma`
  column — the fitted scale of each `newdata` row, on the response scale —
  beside `mean` (the fitted mean), so a CRPS or log score reads the two
  distribution parameters instead of rebuilding the scale from the interval
  band as `(upper - lower) / (2 * z_level)`. The column is absent without a
  spread model and for families with no distributional counterpart, which
  `spread_formula` distinguishes. The `discover` topic in the agent guide
  states the same and stays under the 9,000-character one-read limit (#589).

- Discovery refuses a predictor that is a copy or a monotone transform of the
  response (#590). Before the draft, a numeric candidate whose absolute rank
  (Spearman) correlation with the response is at least 0.999 is dropped and
  named in `result.warnings`; a frame that still held `wages` when the response
  was `log(wages)` no longer selects `bbs(log(wages))` as a mean effect with no
  warning. Rank correlation is invariant to a monotone transform, so the check
  catches a duplicated column and a transformed one. A whole-number candidate
  with one distinct value per row is named in `result.warnings` as a possible
  row identifier but kept, because it may be an ordered quantity. A
  `mandatory_terms` entry that names a dropped copy raises `ValueError`.

- `cvrisk_gamboostLSS` and `cvrisk_glmboostLSS` accept an integer `folds=k`
  (#587). An integer builds k-fold in-bag weights with
  `mb.cv_weights(n, type='kfold', B=k)` (seed 42), as the non-LSS `cvrisk`
  does, so `cvrisk_gamboostLSS(model, folds=5)` no longer dies with
  `IndexError: tuple index out of range`; a `folds` below 2 raises a
  `ValueError` naming the `cv_weights` call. A fold matrix or `FoldScheme` is
  unchanged.

- Discovery warns when backward elimination leaves no mean term (#586). An
  intercept-only mean model (`formula == "1"`) is now named in
  `result.warnings`, with the family, instead of only being logged; previously
  a caller that did not read the log got a normal-looking result that predicts
  the response mean for every row.

- Discovery passes its requested stopping ceiling to the native-backed tuner
  and its preflight estimate (#574). Requests above 10,000 updates no longer
  silently search a smaller grid while reporting the larger one.

- Discovery completes binary varying-curve references for numeric and boolean
  inputs as well as categorical factors (#572). The original by values and
  adoption rule are retained; a missing continuous baseline is made available
  during the interaction comparison.

- CI: guide-mentioned symbols must exist on `mboostrs` (#468).
  - **Added:** `tests/test_guide_symbols_exist.py` reads the shipped agent guide
    (`SKILL.md`, every topic and worked example), extracts every backtick-coded
    callable it presents as mboostrs API, and asserts each one is importable. A
    small documented allowlist covers tokens that are not top-level exports --
    Python builtins, formula terms (`bbs`, `bols`, `btree`, `brandom`,
    `bspatial`), result methods and statistics/R prose -- and names mentioned
    only behind `hasattr(mb, "name")` are treated as version-gated. This closes
    the round-3 D5 hole where the guide named `reference_gap` before the
    installed wheel exported it.
  - **Fixed:** the `rent-analysis` and `readmissions-pitfalls` worked examples
    called `group_folds` / `mb.group_folds`, an R-style spelling the package
    does not export; they now call `grouped_folds`. Tests and guide text only,
    no Rust or Cargo change.

- Discovery supplies a baseline curve for a two-level categorical varying
  smooth when marginal elimination removed its continuous partner (#560).
  The interaction comparison records its reference dictionary and preserves
  numeric-indicator behavior and the existing adoption threshold.

- Effect reports resolve composed interaction and transform recipes to the
  original predictor columns (#564). `sensitivity()` and discovery no longer
  request internal names such as `log__x` for a factor-by smooth; malformed
  cyclic recipe metadata raises a descriptive error.

- Autospec replays an adopted tensor or joint-tree form at the stopping
  iteration that justified its adoption (#535). Its stopping evidence and
  cached full-data path now describe that same specification, using the
  existing cross-validation fits and unchanged adoption rule.

- Paired-ab instrument accounting fix (issue #511): `term_accounting` in `benches/paired_battery.py` no longer records the term string `brandom(g, df=4.0)` in `used_columns` — `_learner_columns` falls back to splitting the learner's display label, whose comma sits inside parens, so the whole term string came back as one pseudo-column. Random-effect learners now resolve through the spec object (`Brandom.name`, plus `by` when present). Recomputing the frozen baseline's linear_factors panel with the fix: structural recall 0.75 -> 1.00, FDP 0.25 -> 0.00 on all 8 replicates (accounting-only defect; predictive gaps unaffected). Regression test `test_brandom_term_reports_column_not_term_string` in `tests/test_paired_ab_battery.py` (15 tests).

- Paired estimation-discovery instrument `paired-ab/v1.1` (issue #511): ports measurement-validity pieces from the v2 design (PR #527, not merged) into the published battery. A-arm stopping now grows by doubling (`continue_to`) until the validation minimum is interior, ceiling 3200 — the fixed-cap-1000 argmin truncated 3/8 interaction replicates; `at_boundary`/`final_cap`/`growth_steps` recorded per replicate (B_discovery untouched). Partitions are four-way train/selection/calibration/test (60/15/15/10, n=1000), all arms paired; the calibration split is held out disjoint from selection, reserved for future interval calibration. New evaluator-only supplied-basis floors per panel (noiseless fits of supplied and partial formulas). New task `threshold_jump`: a discontinuous threshold the smooth basis cannot follow (partial floor ≈ 0.11 vs supplied ≈ 0.0025), the first task where discovery is expected to struggle. Deliberately not ported: v2's Task-dataclass redesign, `linear_sparse`/`mixed_factor_proxy`, v2's frozen baseline artifacts. `benches/paired_battery.py`, `benches/bench_paired_estimation_discovery.py` (+ `--floors`, report gains boundary rates and the precision contract), `tests/test_paired_ab_battery.py` (14 tests), `docs/benchmarks/paired_estimation_discovery_protocol.md`.

- `api_card(obj)` and `python -m mboostrs sig NAME` print a bounded card -- signature, one-line return type and a minimal example (≤20 lines) -- for any public name, resolved from the top level or as a dotted path (`GAMBoostModel.predict`). The guide index lists the key signatures of the high-churn targets, and `tune_mstop`, `compare_specifications`, `scan_lss_families`, `GamlssRefit`, `conformal_interval`, `GAMBoostModel.predict`, `save_model`/`load_model` and `stabsel` open their docstrings with a short usage block (#505).

- Stopping-rule ablation for the D5 predictive-gap diagnosis (#509),
  `stopping-ablation/2`: preserves the October 12 experiment as a superseded
  record and replaces its causal/universal conclusions with an artifact-derived
  audit. Adds a native fixed-preprocessing control for growth/patience, and
  fold-local fixed/growing/structured-validation comparisons for deployment.
  Forecasting uses whole-date rolling origins with explicit judge masks;
  grouped/spatial designs preserve whole units. Reports paired per-seed loss
  changes, medians, full ranges, denominators and worst regressions, including
  harmful configuration choices. Full fold-local risk curves and fold
  memberships are saved. The old claims that growth never materially worsens
  loss, patience never improves on growth and every configuration choice is
  benign are withdrawn. Original path diagnostics are present on main at
  `da27cd9`; the claim that they were unrecoverable from every branch is also
  withdrawn. Harness/tests: `benches/stopping_ablation_509.py`,
  `tests/test_stopping_ablation_509.py`; versioned evidence:
  `docs/benchmarks/stopping_ablation_509/`. Fold-local patience remains
  unsupported by the existing API. No boosting numerics or public default changes.

- Autospec pair-form adoption retains the existing held-out gain, paired-SE
  statistic and fold count in `adoption_evidence`, so adopted tensor and
  joint-tree terms appear with their supporting comparison in `report()`.
  This fixes the reporting omission in the competing-form route; fitting and
  adoption decisions are unchanged. Regression checks cover both forms and
  preserve the single residual screen when both candidates are offered.

- Agent guide: a new `golden-path` topic is the default of `agent_guide()` and
  the lead of `SKILL.md` (#504). It is the workflow on one page -- data,
  `fit()`, `report()`, then three common follow-ups -- with "open topic X only
  if Y" pointers, and it shows the current `gamboost` + `tune_mstop` +
  `reference_gap` path until the #474 `fit()`/`report()` entry points are
  importable (detected with `hasattr`); helpers stay public primitives. Docs
  and Python wiring only, no Rust or Cargo change.

- Book benchmarks intro updated to the Knowledge Discovery Scorecard 2.1 (`mboostrs-scorecard/2.1`, #511): the old eleven-dimension D1–D11 table becomes twelve dimensions (D1a/D1b plus D2–D11) with plain-prose per-dimension blocks, an information-contracts section, a what-goes-where mapping, a compact cheat-sheet and twelve weights; `tests/test_book_navigation.py` now checks 2.1 and the twelve IDs. Docs only.

- Knowledge Discovery Scorecard 2.1 (`mboostrs-scorecard/2.1`, #511, #509): splits the former D1 into D1a (relationship recovery under a prescribed specification) and D1b (relationship discovery from data alone), so a library is not credited for structure it was handed nor penalized for structure it was never asked to find. Documents the #511 information contracts (Track A / partial-knowledge / Track B; real-data prespecified ≠ oracle; paired excess/gap formulas), adds a `## Information contracts for discovery (#511)` section, touches D3/D4/D6/D7/D8/D9/D11 measures to name the contract, and updates the registry, report template, status page, migration crosswalk (label impact keeps `dim:D1` as a legacy alias; open questions for Fernando on D3-vs-D4 and D1-vs-D11) and `tools/check_scorecard.py`. Twelve dimensions; docs only.

- Book benchmarks intro documents the Knowledge Discovery Scorecard 2.0 (`mboostrs-scorecard/2.0`, #507): its D1–D11 names and measures, the mode weights, the aggregate fairness rule, and how the Part V notebooks freeze against it. Docs only.

- `select_capacity`: the tree rung reads factor columns and degenerate tensor pairs are guarded (part of #493).
  - **Cause:** the chooser-on-shootout rerun (`docs/benchmarks/chooser_shootout/`, probe `benches/probe_chooser_shootout.py`) traced two of the three remaining gaps to the chooser itself: the tree rung named only the numeric columns, so the bike panels' hour-by-working-day interaction -- a categorical -- was invisible to it, no rung paid, and the additive-level excess returned (+294.7% mean on `bike_iid` where the tuned hybrid sat at +11.3%); and a screened pair over a 0/1 column crashed the tensor construction (`boundary knots (0.0, 0.0) must be increasing`, `bike_future` seed 1).
  - **Changed:** the `+tree` rung's `btree(...)` now names every column the base formula reads, factors included; the pair screen skips columns with two or fewer distinct values, which cannot be spline tensor margins.
  - **Measured:** rerunning those panels with the fix, `bike_iid` selects `+tree` on all three splits and lands at +7.2%/+1.5%/+7.8% excess (mean +5.5%, ahead of the tuned hybrid's +11.3%); `bike_future` no longer crashes, wins seed 0 outright (-0.2%) and improves from +241.6% to a +54.3% mean -- its hardest seed remains a temporal-shift case that the open budget/depth item of #493 addresses. On planted panels (`tests/test_capacity.py`), a threshold whose location depends on a string factor is adopted through the tree rung, and the binary-column interaction no longer crashes and is excluded from the tensor pairs while the tree still covers it.

- Benchmark-findings catalog and the chooser-on-shootout probe (part of #493).
  - **Added:** the book's appendix gains *Benchmark findings and their fixes* (`appendix/benchmark_findings.ipynb`): one catalog of every defect, workflow gap and negative result the benchmark batteries produced -- the shootouts (capacity both ways, the `effect_model` representational gap, the teacher-transfer negative results), PM2.5 (the missing measure-what's-left step, log scales, extrapolation, the chooser's decision not to add capacity), the claim-cost battery (crashes, informative missingness, the rejected observed/filled split), the causal battery (overlap and extrapolation rounds), the diagonal-jump miss, the joint-regularization gate, and the planted-insight baseline -- each with the fix, its measured effect, and the open issue numbers.
  - **Added:** `benches/probe_chooser_shootout.py` and `benches/probe_pm25_chooser.py`, with the rerun artifacts in `docs/benchmarks/chooser_shootout/`. Measured: the chooser fixes the always-on-tree pathology (cancer +104.0% → +7.6%; smooth panels won) and traces three remaining gaps (#493); on PM2.5 it correctly declines capacity (0.6604 vs the 0.6615 submission) because that panel's gap was features (lags), not capacity.

- `screen_discontinuities` verification fixes (independent review findings).
  Tests: `tests/test_discontinuity_screen.py` (16 tests); Rust kernel in
  `crates/mboostrs-core/src/disc.rs`.
  - **Fixed:** a duplicated or near-collinear column made the diagonal
    unit's background singular, raising `MboostrsRuntimeError` and killing
    the whole screen. The kernel now skips that unit gracefully (returns no
    rows for it), matching the NumPy reference's behavior. Regression
    tests: exact-duplicate and 1e-9-noise near-duplicate columns.
  - **Fixed:** the jump-vs-smooth typing discriminator. The global
    `type_z` (step vs cubic background) typed steep smooth transitions as
    jumps: logistic sigmoid `1/(1+exp(-8u))` read type_z 23-27, ReLU kink
    read 4.48. The new local discriminator measures the level gap at two
    scales (nearest 15 vs 60 evaluation-half points per side), maximized
    over a threshold window for robustness to estimation error. A true
    jump is scale-invariant (gap ratio ~1); a steep smooth or kink has a
    gap that grows with scale (ratio ~2+). `typed_jump` now requires a
    significant small-scale gap (`type_gap_t > 4`) and a scale-invariant
    ratio (`type_ratio < 1.5`). The design was developed with GLM 5.3
    (which confirmed the band-exclusion alternative fails for sigmoids and
    recommended width-based discrimination); the scale-invariance
    formulation is the implementation. Resolution limit (documented): a
    smooth transition narrower than the small scale's span is
    indistinguishable from a jump.
  - **Fixed:** `na_action="impute"` used the full-column median, leaking
    eval-half information; now uses the fit-half median only.
  - **Fixed:** the near-zero-variance guard (`sd <= 0`) missed
    denormal-scale spreads; now a relative threshold
    (`sd <= 1e-12 * max(1, |mean|)`) that also catches non-finite spread.
  - **Fixed:** kernel dof check now skips when `n <= dim` (not just
    `dof == 0`), aligning with the reference's `dofF <= 0`.
  - **Doc precision:** "exactly standard-normal" -> t-distributed with
    `n_eval - (q + 1)` df (asymptotically normal); "rules a jump out" ->
    "no screened evidence for a jump"; "runs in the Rust core" ->
    "Rust-accelerated scan kernels". The `|type_z| >= 2` typing cutoff is
    documented as heuristic.
  - **Measured:** steep sigmoid and kink now type as smooth (not jumps);
    true +2.0 diagonal jump still types as jump (z 16.1, type_gap_t 4.6,
    ratio 1.40); held-out `sharp_smooth.csv` still types fully smooth.

- `select_capacity(..., estimators=("boosting", "reml"))` can compare Gaussian boosting and direct joint REML on the same folds, including an intercept-only baseline. Candidate records and `selected_estimator` identify the chosen fit. The existing boosting-only default is retained.

- `screen_discontinuities`: locate regime jumps a smooth fit absorbs, with selection-honest statistics. Tests: `tests/test_discontinuity_screen.py`; Rust kernel in `crates/mboostrs-core/src/disc.rs`.
  - **Cause:** an agentic benchmark found analysts using mboostrs systematically miss a sharp +2.0 jump across a diagonal boundary (`s1 = 0.45 + 0.35 s0`, n=800, noise sd 0.5): smooth fits absorb most of the jump, axis-aligned step candidates read "no gain" in cross-validation, and nothing in the workflow or the skill points at discontinuity search.
  - **Added:** `mb.screen_discontinuities(data, target)` scans two candidate families -- steps in one column and steps across a line in two standardized columns over 16 directions -- at interior quantile thresholds, all pairs while there are at most 20 columns and pairs among the best 10 axis columns beyond that, with a fit-half-only refinement of the best diagonal units. Each candidate is a step indicator against a smooth background, fitted by OLS in the Rust core (`disc_scan`: one sort and prefix-sum pass per direction). Selection ranks on the fit half; the reported `jump`, `se` and `z` come from the untouched evaluation half, so the winner's `z` is t-distributed with `n_eval - (q + 1)` degrees of freedom under no-jump (asymptotically standard normal) and the `|z| >= 2`-with-positive-gain gate (the row's `flagged`) fires on ~4.6% of no-step screens. `type_z` (the row's `typed_jump`) refits the step against a background augmented with smooth terms along the projection and separates a jump from a steep smooth crossing. Rows are `DiscontinuityHit` records with the boundary back-transformed to the original scale (`line`, `boundary_intercept`, `boundary_slope`). The kernel matches the numpy reference implementation from the design handoff to 1.6e-13 on every field.
  - **Measured:** on the benchmark panel the raw pass recovers the boundary at 99.2% row agreement (`s1 > 0.462 + 0.333 s0` against the truth `0.45 + 0.35`), jump 1.965 against 2.0, `z` 16.1, `type_z` 12.4. The residual pass after the additive baseline still flags (jump 1.006, `z` 9.2). The boundary's step indicator beats the additive baseline by 0.126 out-of-fold risk (5.2 SE) while `+ btree(pair)` loses (+0.008): the named boundary wins where the black box would have had to. Null panels stay quiet on the residual pass; pure-noise screens do not flag; a steep smooth rise (`sin(pi x)`) flags on the raw response but does not type as a jump.
  - **Guide:** SKILL.md step 5 now asks *where* a reference gap lives before adding terms, and the `modeling-tactics` topic carries the jump branch, including the negative case in which no flag rules a jump out. New worked example `example:disc-screening`.

- Causal friction from the rerun 5 analyst (part of #484). Tests: `tests/test_causal_484_friction.py`, `tests/test_causal_tools_423.py`, `tests/test_causal_rerun4_friction.py`, `tests/test_aipw_regression_456.py`.
  - **Added:** `iv_effect(..., outcome_missing="ipw")`, with the same meaning as in `aipw`. A response model for P(outcome observed | covariates, treatment, instrument) is cross-fitted on the same folds, the outcome mean is fitted on the observed rows weighted by 1/pi, and the DML-IV ratio takes the observed rows' inverse weights (`mboostrs_core::causal::iv_ratio_weighted`). `propensity_clip` bounds pi. The missing-outcome message now names an argument that `iv_effect` accepts.
  - **Changed:** `ci_test` when x or y is an exact linear combination of the conditioning set: `statistic`, `p_value` and `residual_correlation` are NaN, where they were 0, 1 and 0. The note names the columns of the identity and the one `ci_test` suggests dropping (the last conditioning column in it). Terms at rounding level are no longer named as part of the identity.
  - **Fixed:** `ci_test` warnings name the caller's conditioning columns (for example `bbs(w)`) instead of the internal `__ci_givenK` aliases.
  - **Fixed:** `effect_model(terms="smooth")` and other named terms report `gain_over_constant`, `gain_se`, `flat` and `eligible` for every candidate in `diagnostics["terms_selection"]`, as `terms="auto"` does. The rule is `"requested"`.
  - **Changed:** within one `aipw` or `effect_model` call, each distinct warning is issued once with a count of how many times it was raised (`once_per_call`).
  - **Fixed:** `causal_trap_report` does not report a `post_treatment` column that the frame lacks as a blocking finding. Absent covariates, treatment and outcome are still blocking.

- `select_capacity`: measure whether interaction capacity pays before adding it. Tests: `tests/test_capacity.py`.
  - **Added:** `mb.select_capacity(formula, data)` builds a ladder of nested specifications on top of the formula it is given -- the base, the base plus screened tensor pairs, and that plus one joint tree -- scores every rung out of fold on one shared fold partition at the same step budget, and keeps the *smallest* rung whose held-out score is within `tolerance` (default 2) standard errors of the best rung's. The tensor rung takes at most `pairs=` pairs from `screen_interactions` on the base fit's Pearson residual, each entering as `bbs(a, df=2.5) %X% bbs(b, df=2.5)`; the tree rung adds `btree(...)` over the numeric columns the formula reads. The result carries the per-rung table (score, fold SE, the rung's rhs, why it was built), the screened pairs, and the selected rung refitted on all rows.
  - **Cause:** the competitive study (`docs/benchmarks/competitive_shootout_report.md`) measured fixed capacity missing in both directions: the additive model sat a median 46.5% behind the best competing library per panel and the always-on joint tree recovered most of that (to 8.4%), but on the cancer panel the same tree turned a panel the additive model won by 11% into one it lost by 109%. The two probes that copied a teacher model's importances and shapes into a specification (`distillation_probe.json`, `shape_transfer_probe.json`) both scored worse than the plain tuned hybrid, so the capacity choice has to be measured on the data at hand, per data set.
  - **Measured:** on planted responses (n=500--800, `tests/test_capacity.py`), a smooth additive response keeps the base; a bilinear response keeps the tensor rung (base 4.92 vs 0.071 MSE, the tree rung 0.086); a smooth-times-step interaction `2 sin(x1) 1(x2 > 0)` keeps the tree rung, which the tensor cannot represent (0.19 vs 0.13); pure noise proposes no pair and keeps the base.

- `aipw` and `iv_effect` default to `smooth=True` (#487, PR #712). A linear
  propensity model can reach probabilities near 0 (4e-27 in one analyst
  run); pass `smooth=False` for the previous fit. `aipw(outcome_model="auto")`
  fits the pooled outcome model without treatment interactions for data of
  at most 1,500 rows; over 25 draws of 1,500 rows the mean absolute ATE
  error was 1,404 for that model against 2,487 for the previous rule
  (#484). Larger data use the sparse-arm rule as before. The new
  `aipw(outcome_model="cv")` fits `pooled`, `separate` and, for a
  zero-inflated continuous outcome, `two_part` on the cross-fitting folds,
  keeps the one with the lowest out-of-fold loss and reports every
  candidate's loss in `diagnostics["outcome_model_cv"]`. The causal usage
  page works through a design with few treated rows at high prior
  earnings, where the smooth default gives -3,169 against a truth of 4,946
  with overlap warnings and the estimate on the overlap rows is 4,572.

- `AIPWResult.subgroup_effect(by)` returns the mean of the augmented scores
  in each subgroup (a boolean mask, or one subgroup per value of a column
  of the training data) with its standard error and interval (#487, PR
  #714). The variance is the one the ATE uses, so a subgroup holding every
  row reproduces the ATE and its standard error. The computation is
  `mboostrs_core::causal::subgroup_scores`. `AIPWResult.effect_surface(formula)`
  fits the doubly robust second stage, a `gamboost` model of the augmented
  scores on the covariates, whose `predict` estimates the conditional
  effect.

- `effect_modifiers` runs an R-learner gate by default (`gate=True`, #484,
  PR #711). Before it reports any modifier it fits the smooth second stage
  of `effect_model` on the tested covariates, with the cross-fitted
  nuisances of the `AIPWResult`, and requires the out-of-fold gain over a
  constant effect to exceed its standard error. When the gate fails, no
  modifier is selected and `EffectModifiers.message` gives the gain and
  its standard error; `diagnostics["gate"]` records both. In 40 simulated
  draws with no effect the score tests alone claimed a modifier in 4 and
  the gated function in none; in 20 draws with an effect modifier the gate
  passed in all 20. The gate takes 0.28 s of CPU time on 2,000 rows.
  `gate=False` gives the previous behaviour. `covariates` accepts transform
  expressions such as `"log(prior_earnings)"`, using the transforms of the
  model formulas (#487).

- `estimate_cost` covers `aipw`, `effect_model`, `ci_test`, `iv_effect`,
  `confounding_benchmark` and `discover` (#484, #487, #447, PR #713). The
  counts follow the call's arguments and the signature defaults;
  `aipw(outcome_model="auto")` and `effect_model(terms="auto")` are upper
  bounds because the call chooses after fitting. For `discover`,
  `CostEstimate.stages` gives the first-pass fit count of each stage, and
  `DiscoveryResult.stage_seconds` holds the measured seconds of each stage.

- Interaction nomination skips a pair whose two members are both
  `brandom` terms, that is, a factor with more than ten levels paired with
  another factor (#447, PR #709). Such a pair is scored on folds that hold
  out part of each group, so its score rewards memorized group effects.
  The rule applies in `discover`, `suggest_specifications` and
  `probe_interactions`. `discover` logs each skipped pair with the terms
  to add by hand, for example `bols(g) + bols(b) + bols(g, by=b)`.
  Factor-by-numeric pairs are unchanged.

- `validate` lists a source column that a derived column is built from on
  its own line, naming the derived column it is read through
  (`read through derived columns: households (log__households)`). Only
  columns that no term reads, directly or through a derived column, are
  reported as not read by any term (#447, PR #710). `GamRefit.predict`
  documents its two return types: a NumPy array without `interval=True`,
  and a Polars frame with `estimate`, `se`, `lower`, `upper` and
  `outside_range` with it (#487).

- `autospec` fits its distributional candidates (BetaLSS, GammaLSS,
  GaussianLSS) with `stabilization="L2"`, on a step-count grid that is
  doubled while its optimum is the largest step count (#718). A candidate
  is adopted only if its held-out log score is better and the out-of-fold
  squared error of its mean is no worse than the cross-validated mean
  model's by more than 2% and two paired standard errors. `FamilyEvidence`
  records both checks. Before this, GammaLSS was fitted at `nu=0.3`
  without stabilization, became unstable after about 10 cycles, and was
  adopted at 10 steps on two California housing splits with test R^2 0.58
  and 0.62; with the change those two splits score 0.724 and 0.721, and
  the mean over the five splits is 0.723, against 0.714.

- The insights benchmark scores an `autospec` fit that adopts an
  `LSSModel` on its mean parameter `mu`, on the response scale, and
  records the family `autospec` chose on each split (#716). Such fits
  used to fail in the harness.

- A Polars `Enum` column is a factor in `gamboost`, `glmboost` and their
  predictions, encoded as a `Categorical` column is (#717). Its levels
  follow the declared order, so the first declared level is the reference
  level of a `bols` contrast. Before this, an `Enum` factor raised
  "holds values that are not numbers".

- `discover` adds an actual-incumbent guard (#679, PR #721): an
  interaction candidate is adopted only if it also has a lower paired
  cross-validated risk than the current mean model. The new
  `discover(mean_step="fixed" | "line_search" | "auto")` sets the step
  rule of every mean-model stage. `"fixed"` is the default, because on the
  reserved confirmation seeds the line-search default raised the mean
  error on other panels (upper 95% bound +131% on the low-count panel).
  `"auto"` resolves in the Rust core (`discovery_mean_step`): line search
  for NBinomial, fixed steps otherwise. The study's protocol, decisions
  and summaries are in `docs/book/benchmarks/count_incumbent_guard/`; the
  raw records are kept outside the repository (#724), and its README says
  where.

- Benchmarks (#304): the varying-coefficient benchmark encodes raw-frame
  data sets inside each fit, so it runs on cpu_act, credit_g and
  electricity (#719); the unreferenced `bench_interaction_usage.json` is
  removed. A dated AIPW rerun on the causal-estimator draws compares
  `smooth=True` with `smooth=False` and `outcome_model="auto"` with
  `"cv"` (#725). The causal estimator and causal discovery pages state
  that their recorded AIPW default arms were fitted with `smooth=False`
  (#720).

- Agent guide, `causal-roles`: the adjustment set follows the disjunctive cause criterion (adjust for every pre-treatment cause of the treatment or the outcome) and says not to drop a covariate because `ci_test` found no link to the treatment (part of #478).
  - **Cause:** in the causal benchmark's rerun 4 the analyst pruned covariates by test and left out `smsa`, a true confounder, on 4 of 8 replicates; the adjustment set was valid on 3 of 8 (EconML 8 of 8). The guide had said to adjust "only when" a variable affects both treatment and outcome.

- Reference gap: `mb.reference_gap(formula_or_model, data, ...)` measures how much a flexible tree model still gains over a specification on the same held-out rows (part of #468). Tests: `tests/test_reference_gap.py`.
  - **Added:** the reference is one `btree` term over every covariate the specification reads, with `maxdepth=4` by default and the same family and step length. Each model is tuned to its own best step count on the folds. The gap is the mean over folds of the specification's loss minus the reference's, with its standard error across folds.
  - **Added:** folds are a random k-fold split by default, rolling-origin blocks with `time=<column>` (each block is scored after the rows before it), or any `mboostrs.folds` scheme passed as `fold_scheme=`.
  - **Added:** when the gap is above two standard errors, the function ranks where it comes from. Interaction pairs are screened on the specification's out-of-fold residuals and become `bbs(a) %X% bbs(b)` (two numeric columns) or `bbs(x, by=f, df=6)` (a numeric column and a factor). With `time=`, float columns are lagged and rolling-meaned (`lags=(1, 3, 24)`) and screened the same way. The top `top` candidates are each refitted once, and `closed_share` is the share of the gap the candidates with a positive gain close together.
  - **Added:** `ReferenceGapResult` with `gap`, `gap_se`, `material`, `model_loss`, `reference_loss`, `candidates` (a Polars frame), `closed_share`, `notes`, `describe()` and `to_dict()`.
  - **Added:** the reference overfit check reports the reference's loss on its own training rows next to its held-out loss. A gap resting on fewer than three folds, and a best step count within 5% of `max_mstop`, are noted.
  - **Limitation:** candidates are screened on residual mean square, not on the family's loss, so for a quantile model the ranking follows squared residuals. The screened candidates are measured only when the gap is material. A structure that the reference cannot see, such as a lag of a column the formula does not name, is therefore missed when trees do not beat the specification.
  - **Unchanged:** `autospec`, `discover` and `screen_interactions`. The `extrapolation` argument is checked and stored but has no effect on the numbers returned, since every score is on rows the models were fitted on.

- `na_action="impute_model"` no longer splits a term on a filled column into observed and filled parts; the split (from #467) is opt-in as `na_action="impute_model_split"` (part of #457).
  - **Cause:** on the claim-cost benchmark (10 draws) the split lowered the coverage of the housing-age effect interval from 0.90 to 0.60 and left roof-class coverage at 0.80.

- The guide reaches an agent that has only the installed package (part of #468).
  - **Worked examples ship:** the analyses in `docs/skills/analyzing-data-with-mboostrs/examples/` are in the wheel and read with `mboostrs.agent_guide("example:<name>")`, listed by `agent_guide("index")` and copied by `install_skill`. New example `example:pm25-reference-gap`: a tuned additive model trailing CatBoost on hourly PM2.5, and the lags and interactions, chosen on the validation year, that closed 58% of the gap.
  - **Signposts:** the docstrings of `gamboost`, `glmboost`, `gam`, `gamlss`, `discover`, `gamboostLSS` and `glmboostLSS` name `agent_guide()` and `reference_gap`. The extrapolation, step-count boundary, penalty-drift and split-group-folds warnings end with the guide topic that explains them (`Guide: python -m mboostrs guide <topic>.`).
  - **Cause:** the benchmark analysts ran inside the repository. An agent with only the wheel had no worked examples, and the only pointer to the guide was the package docstring.

- Agent guide: new topic `modeling-tactics`, and step 5 of `SKILL.md` now tells an agent to measure the gap to a flexible reference (`reference_gap`) before it stops (part of #468).
  - **Cause:** in the PM2.5 forecast benchmark the mboostrs analyst stopped after tuning one additive formula and finished 7% behind CatBoost (2014 RMSE on log PM2.5 0.661 against 0.618). The guide presented steps 1–4 as the path and never asked how much a flexible model would still gain.
  - **Content:** the topic gathers the changes that closed measured gaps in the claims, causal and PM2.5 benchmarks: interactions and factor `by=` terms, lagged covariates on time-ordered rows, log scales for heavy-tailed covariates and responses, model-based imputation, basis size for narrow features, spread drivers outside the mean model, a second causal estimator as the reference, `ci_test` on exact identities, and the null and small-data checks. On PM2.5, lags and interactions chosen on 2013 brought the 2014 RMSE to 0.636 and beat CatBoost on the validation year (q50 pinball 21.21 against 21.27).

- Observed and filled parts of a smooth term on a model-filled column (part of #457). Tests: `tests/test_impute_model.py`.
  - **Added:** under `na_action="impute_model"`, a `bbs(x)` or `bols(x)` term on an imputed column is fitted as two terms: `[observed]`, multiplied by `1 - <x>_missing` (fitted on the observed rows), and `[filled]`, multiplied by `<x>_missing` (fitted on the filled rows). The formula still reads `bbs(x)`. The parts appear as `bbs(x) [observed]` and `bbs(x) [filled]` in `selected`.
  - **Added:** a contrast or effect query that sets `x` treats every row as observed, so only the observed part contributes.
  - **Changed:** `gam_refit` and `gamlss_refit` leave out the constant `<x>_missing` term when the split is present, because the observed parts sum to one minus that indicator. They emit a warning that names the term. Boosting keeps it. A tensor on the same column no longer produces a duplicate-term refusal in the refit.
  - **Not changed:** a categorical column keeps its `"(missing)"` level without a split. A `%X%` tensor over the column uses the fill without a split.
  - **Measured (10 draws, same seeds, current `main` code):** the split does not recover the age effect. Q2 coverage is 0.60 with the split, against 0.90 for `impute_model` without it and 0.90 for `impute`. Q2 bias is +0.013, against +0.008 and −0.003. Roof-class Q4 keeps its gain (coverage 0.80, bias +0.009; `impute` gives 0.30, +0.025).

- Model-based imputation of missing predictors: `na_action="impute_model"` (part of #457). Tests: `tests/test_impute_model.py`; Rust tests in `crates/mboostrs-core/src/impute.rs`.
  - **Cause:** on the claim-cost benchmark, `housing_median_age` is missing on 4–16% of rows, and `roof_class` carries information about that missing age. `na_action="impute"` fills the age with its median, so the roof-class coefficient absorbs part of the age effect.
  - **Added:** `na_action="impute_model"` for `gamboost`, `glmboost`, `gamboostLSS`, `glmboostLSS`, `discover`, `screen_interactions` and the refits. A numeric predictor with missing values is filled with its conditional mean given the other predictors of the formula. The mean is a penalized additive regression fitted in the Rust core on the rows where the predictor is observed: a cubic B-spline with 20 interior knots and a second-difference penalty for each numeric predictor, a level design with a ridge penalty for each factor predictor, and a 0/1 missingness column for each other predictor with missing values. The smoothing parameter is chosen by generalized cross-validation. The response is never a predictor. The `<x>_missing` indicator term is kept, as under `"impute"`.
  - **Added:** the fill is part of the fitted recipe. `predict` applies the stored fill and never refits it. A `%X%` tensor margin over an imputed column reads the model's fill at fit and at predict time.
  - **Added:** `mb.imputation_diagnostics(model)` returns the R-squared of each model-based fill, by column.
  - **Changed:** the remedy in the refusal for a missing column in `glmboost` and in the refit refusal names `na_action="impute_model"`, and the `predict` refusal names it when the other covariates predict the column, with that R-squared.
  - **Unchanged:** the default stays `na_action="impute"`. `screen_interactions` and the screens inside `discover` still fill with the median.
  - **Benchmark:** 10 fresh draws of the claim-cost design (own seeds), rerun2 final formula, GaussianLSS, mstop 4385. On the Q4 roof-class contrast the bias fell from +0.022 to +0.006 and the 90% interval coverage rose from 0.40 to 0.80. The age contrast Q2 got worse: bias rose from +0.014 to +0.029 and coverage fell from 0.70 to 0.20. The fills of the missing ages have a standard deviation of 9.2 years against 13.3 for the true ages, so the conditional-mean fill narrows the age effect. The imputation step adds about 0.05 s to a 3 s fit. Q6 (the coordinate contrast) is off by about −0.16 under both policies, so it is a formula question, not a missing-value one.

- Penalized GAM by REML from a formula, without boosting (part of #457). Tests: `tests/test_direct_reml_gam.py`.
  - **Added:** `mb.gam(formula, data, family=..., weights=, offset=, na_action=, method="REML")` fits every term of the formula, with one REML smoothing parameter per penalty, and returns a `GamRefit`. The designs and penalties are the ones `gam_refit` builds for a boosted model's terms. No boosting and no `mstop` take place. Families: Gaussian, Binomial, Poisson and Gamma.
  - **Added:** `mb.gamlss({"mu": ..., "sigma": ...}, data, family=mb.GaussianLSS())` fits a distributional model the same way and returns a `GamlssRefit`. `mboostrs.gamlss` is also the module, so `from mboostrs import gamlss` and `mb.gamlss.GaussianLSS` keep working; calling the module runs the fit.
  - **Added:** `contrast(variable, from_value, to_value, *, data, level=0.9)` on `GamRefit` and `GamlssRefit` (per parameter, with `parameter=`). It returns an `EffectContrast` with the mean change of the link-scale predictor over the rows of `data`, its standard error from the full covariance, and the interval. Several covariates can move together.
  - **Added:** `RefitResult::linear_contrast` and `LssRefitResult::parameter_contrasts` in `mboostrs-core`, bound as `GamRefitCore.contrast` and `GamlssRefitCore.parameter_contrasts`.
  - **Added:** `RefitControl::gram`, off by default. `mb.gam` sets it: the REML fit factors `X'WX + S` by Cholesky instead of a QR of the n-row design, and the post-fit concurvity uses the Gram too. `gam_refit` keeps the QR path, so its numbers are unchanged; the two paths agree to about 1e-7 in the coefficients (the Gram squares the condition number).
  - **Fixed:** the identifiability check for dependent unpenalized directions multiplied the n-row design by each null-space direction (one pass per direction). It now forms `D'(X'WX)D` once. The nested-term check computed its cross products with a single-threaded product over all rows and now uses the blocked weighted products.
  - **Changed:** the B-spline recursion in `bspline_basis` no longer allocates per row, and `row_tensor` no longer divides and takes remainders per element. The basis values are unchanged.

- **Behaviour change (#456):** `aipw(outcome_model="auto")` no longer selects `"two_part"`. On a zero-inflated outcome whose sparse-arm check flags a bin, `auto` uses the pooled model with treatment interactions again, as before #434. `outcome_model="two_part"` must be requested by name. The two-part model now enters each strictly positive baseline with skewness above 1 (such as prior earnings in dollars) on the log scale, holds its covariates to the training range, and `aipw` warns when a fitted outcome mean lies more than one observed outcome range beyond the outcome's observed range (`diagnostics["outcome_range"]`).
  - **Cause:** the two-part log model was linear in raw prior earnings in dollars, and its mean takes an exponential. A row far outside the training range got a log mean that drove the mean to about 5e8. On the 12 job-training draws, the old default had an ATE error above 100,000 dollars on 7 draws (RMSE 5.6 million).
  - **Now:** the default has RMSE 1,019 dollars, no error above 2,400 dollars, and a mean absolute error of 28 dollars on the two no-effect draws. Measured per model on the same 12 draws, `separate` RMSE 1,051, `pooled` 999, bounded `two_part` 685 (but 490 mean absolute error on the no-effect draws, against 28 for the default). Tests: `tests/test_aipw_regression_456.py`.

- **Behaviour change (#456):** `effect_model(terms="auto")` returns only a candidate that beats a constant effect by more than its standard error and whose tuned step count is at least 5. It returns `"smooth"` when the chosen candidate is within one standard error of it. `diagnostics["terms_selection"]` records the rule. The default `max_mstop` is now 1600 (was 10,000), the cap `aipw` uses.
  - **Cause:** the propensity nuisance reached the 10,000-step cap in every fold, and its inner cross-validation took most of the fit time. On the 12 job-training draws, `auto` returned stumps, a candidate that the constant-effect check flags, on 11 draws; it now returns stumps on 2 and smooth on 10. Mean ranking regret fell from 0.65 to 0.34 (median 0.87 to 0.22); mean PEHE on the effect draws rose from 1,034 to 1,280 because the smooth candidate has higher PEHE where stumps had lower PEHE.
  - **Speed:** one `auto` fit on 10,074 responder rows took 126 s before and 36 s after; one `terms="smooth"` fit took 108 s and 31 s (CPU seconds 128 and 52 for `auto`). The machine load was 60 to 90 during these runs.

- Categorical contrasts in the REML refit (part of #457). Tests: `tests/test_refit_contrast_coverage.py`.
  - **Fixed:** `gamlss_refit` read a factor that the distributional fit had encoded as integer codes as one numeric column, so `bols(f)` fitted a straight line in the codes. Every contrast between non-adjacent levels was wrong; on a planted effect with levels A=0, B=0.3, C=0.05, D=0.35, the D-A contrast came out 0.11 low. The refit now reads the model's recorded factors and gives each level its own column (the same design `gam_refit` uses).
  - **Added:** `gamlss_refit(..., factor_ridge=True)` puts a ridge penalty on the level indicators, with REML choosing its strength, so level effects shrink toward the reference level. The default, `False`, leaves factor levels unpenalized.
  - **Unchanged:** a factor term in the default refit was already unpenalized, and the refit's sum-to-zero centring does not change a contrast.
  - **Not changed:** the effect-query intervals still miss on smooth-term bias. The claim-cost query Q6 (a location move between two spatial bumps) covers 0 of 10 draws because the refit inherits the boosted tensor's five knots; ten knots lower its bias from -0.15 to -0.11 but do not reach coverage, and corrected covariances leave the width unchanged.

- Spread search on small samples with a factor `by=` in the mean model (part of #415). Tests: `tests/test_small_n_spread_search.py`.
  - **Fixed:** `gamboostLSS` refused `bols(f, by=z)` for a factor `f` with more than two levels ("Encoding it as numbers would make bols treat the levels as an ordered measurement"), so every candidate of a spread search that carried the term failed. `gamboost` accepted the same term but multiplied the integer codes of `f` by `z`, which fits `bols(code, by=z)` rather than one effect per level. Both now lower the term to the level design multiplied by `z`, as mboost does. A `gamboost` model with a factor `by=` changes its fit.
  - **Changed:** `bols(a, by=b)` with factors `a` and `b` is the cell design of the two factors, one coefficient per pair of levels, in `gamboost` and in the distributional fits. It was refused in the distributional fits and read as codes in `gamboost`.
  - **Changed:** `compare_specifications` with a fold count keeps every level of each factor that a candidate uses in every training fold, where a level has at least two rows in a combination. Each combination of levels is dealt across the folds. A comparison that uses a factor gets a different fold assignment from the random split, so its numbers change from earlier runs. Numeric-only comparisons keep the random folds.
  - **Fixed:** a `GaussianLSS` held-out loss was infinite for a row whose predictive density underflows to zero (about 38 or more standard deviations from the fitted mean), and the candidate was rejected with "fold_scores must be finite when no fold failed". The Gaussian held-out loss now uses the log density, which stays finite.
  - **Changed:** a fold whose held-out loss is infinite is reported as a failed fold. The candidate's row in `compare_specifications` has `error` naming the fold and the cause, and the other candidates are still compared. The error names `extrapolation="clip"` and a smaller spread model as remedies.

- Visualize multi-column `btree` partitions (issue #280). `GAMBoostModel.plot_btree_partition(term=None, grid=60, data=None, at=None)` draws a term's own fitted contribution as a 2-D Altair heatmap: rectangle marks over the two columns the tree splits on, the `btree_splits()` cutpoints as dashed rules, and the training points when `data` is given. A one-column tree falls back to a step plot, and a tree over more than two columns takes `at={column: value}` to fix the columns not drawn. `btree_partition_table` (a `GAMBoostModel` method and a `mboostrs.gamboost`/`mboostrs.plot` function) returns the same partition as a Polars frame, one row per region with `<column>_lower`, `<column>_upper` and the term's `value` at the region centre. A one-tree term is enumerated by walking that tree, so the rows are its actual leaves rather than the finer grid of its cutpoints; a term selected over several boosting steps uses the grid its recorded cutpoints induce. LSS models, which keep one btree term per parameter, raise a `ValueError` naming `gamboost()`.

- Causal estimation with EconML and DoubleML (issue #451). Tests: `tests/test_sklearn_causal_libraries.py`, `tests/test_agent_guide_topics.py`.
  - **Recommended route:** for double machine learning and instrumental-variable estimation, the agent guide now recommends EconML or DoubleML with `GAMBoostRegressor` / `GAMBoostClassifier` as the nuisance learners. The overview and four causal sub-topics (`causal-average`, `causal-outcome-models`, `causal-heterogeneity`, `causal-instruments`) have a runnable example. The native `aipw` and `effect_model` stay available, and the guide says what they add (no extra dependency, `outcome_missing="ipw"`, the overlap and sensitivity tools, and the two-part outcome model).
  - **Fixed:** a feature named `y` in a pandas frame was overwritten by the response in the data the formula is fitted on, so the fit silently used the response as a covariate. A column whose name the formula parser cannot read (a space or a parenthesis) also failed. Without `formula=`, such a column is now fitted under the name `f<j>`, and `y` is never a feature name. With `formula=`, a column the parser cannot read, or a column named like the response, is refused and the message names it. The same applies to `LSSRegressor`, `SurvivalBoost`, `TreeBoostRegressor` and `TreeBoostClassifier`.
  - **Changed:** a string or category column given to a scikit-learn adapter is refused with the column name and a suggestion to encode it as dummy columns. The error was numpy's `could not convert string to float`.
  - **Changed:** the agent guide test runs each python example as its own case. A fenced block can declare `requires=<package>`; the example is skipped when that package is not installed.

- Mean predictions outside the training range: one policy on every path, and an `extrapolation=` argument where a mean path lacked one (part of #415). Tests: `tests/test_small_n_prediction_blowup.py`.
  - **Cause:** a spline continues at its edge slope beyond its boundary knots. On an 800-row draw of the claim-cost data, a raw heavy-tailed covariate (`ave_occup`, whose training maximum was 6.3 and whose test rows reached 1,000+) gave one row a predicted mean of 338 while the other means lay in [-0.5, 1]. The default policy warned, and the warning was recorded, but the value was returned unclipped.
  - **Fixed:** `conformal_interval(...)` returned `fitted_mean` with the extrapolation warning suppressed. The full model's mean now carries the same policy and the same warning as the interval, in one summary.
  - **Changed:** `GamRefitBootstrap.predict` defaulted to `extrapolation="clip"` while `GamRefit.predict` and `GamlssRefit.predict` defaulted to `"warn"`. The bootstrap now defaults to `"warn"` like the other refit predictions.
  - **Added:** `extrapolation=` on `LSSModel.params`, `LSSModel.predict_parameters`, `LSSModel.predict_all`, `predict_all` (model method and module function), and `conformal_interval` (function and model method). The default is `"warn"`, as in `predict`; `"clip"` holds each spline covariate at its training edge and warns with the rows clipped.
  - **Unchanged:** the default of `predict` stays `"warn"`. `tests/test_extrapolation_safety.py` pins it because R mboost parity depends on the edge-slope continuation. Pass `extrapolation="clip"` to bound a mean that is extrapolated.

- Distributional-model fixes from the rerun 2 claim-cost friction log (part of #415). Tests: `tests/test_lss_rerun2_friction.py`.
  - **Fixed:** `tune_mstop` on a mapping (distributional) specification returned `MstopTuning.model = None`. It now refits the model at the chosen step count and returns it, as the single-parameter path does. The fit count is unchanged (`folds + 1`).
  - **Changed:** `gam_refit`'s warning for held unpenalized directions names the terms that share them. When a main effect's variable is a margin of a tensor in the same refit, the remedy is to drop the main effect; otherwise it names the terms and asks for one of them to be dropped. The previous text always advised dropping a duplicate column.
  - **Changed:** `conformal_interval(method="distributional")` no longer refuses a model with fitted transformations. The transforms (medians of `na_action="impute"`, log floors, centres) are estimated again from the proper-training rows of each split, so the calibration rows do not shape them. The refit needs the inputs before the transforms, which a distributional fit now keeps; a fit with an interaction (`%X%`) or an `env=` matrix keeps no such inputs and still refuses, naming the reason. `calibration_source` states which transforms were re-estimated. The refusal that names a fitted transform does not list an imputation median for a column that was complete at fit time.
  - **Fixed:** the prediction recipe for `na_action="impute"` wrote a trailing ` + ` into a formula whose only transforms were categorical fills. The formula is now returned unchanged.
  - **Changed:** `predict` on a model fitted with `na_action="impute"` refuses a missing value in a column the formula reads that the fit recorded no fill for, and says so (it no longer recommends `na_action="impute"`, which the model already has). Columns that had a fill (the training median, or a missing-value level) are still filled.
  - **Added:** `gamlss_refit` supports intercept-only parameters (`"y ~ 1"`, and `"y ~ 1 + bbs(x)"` where the `+ 1` is the intercept). `GamlssRefit.intercept_only` lists the parameters with no term, and `GamlssRefit.intercepts` gives each parameter's intercept with its standard error.
  - **Fixed:** `cvrisk` on a distributional model with `grid=[100, 250, ...]` (a flat list of step counts) raised `TypeError: 'int' object is not iterable`. A flat list is now a common grid for all parameters. A grid point with the wrong number of counts raises a `ValueError` that names `make_grid_lss` and shows the expected shape.

- Claims benchmark rerun 2 friction fixes (issue #447). Tests: `tests/test_discover_rerun2_friction.py`.
  - **Behaviour change:** `heteroscedasticity(model, data)` tests every numeric predictor of the data, not only the mean model's, so `by_column` can list a column the mean model does not use. New keyword arguments: `columns=` (the candidates; default every column `classify_columns` describes, minus the response) and `transforms=` (`{"households": "log"}` tests the slope on `log(households)`). The result has `tested_columns` and `untested_columns` (each with its reason); each `by_column` row has `transform`.
  - **Fixed:** `discover`'s spread stage nominates from every candidate column, with the drafted transform (a right-skewed column is tested on its log). On four fresh full draws of the claims battery, the adopted spread driver is `log(households)` in all four; on four fresh null draws, none is adopted. A mean model that is intercept-only (`y ~ 1`) no longer skips the spread tests, and a distributional model with no mean term is fitted as `y ~ 1`. The spread log entry lists the columns tested.
  - `discover(max_mstop=...)` caps the mean and distributional tuning grids (default: 400 steps for the mean model, 1600 for the distributional model). The boundary message names `discover(max_mstop=...)`.
  - `discover(extrapolation="warn" | "allow" | "clip" | "error")` is passed to the screens and the tuning. Extrapolation warnings from the stages are issued once per call and name `discover(extrapolation=...)`, not `compare_specifications(...)`. Repeated warnings of other kinds are grouped by `collect_warnings`.
  - **Changed:** factor pairs in `discover` (and `pair_term_for`) with at most ten levels each are `bols(a, by=b)`, one coefficient per pair of levels, instead of `brandom(a) %X% brandom(b)`. The grouped-folds caveat applies only to a `brandom` term with a factor of more than ten levels.
  - `compare_specifications` rows have a `note` key. A row that drops a main effect whose columns a `%X%` tensor in the same specification spans says so; when its change and its standard error are both 0, the note says dropping the term does not change the fit.
  - `validate` reports a raw column that a transform reads (`log__x` from `x`, `x_missing` from `x`) as `read_through_transform` (severity `info`) instead of listing it as not read. Only columns no term or transform reads are in `extra_column`.
  - `discover` logs an `estimate` entry (`candidate` is the stage, `estimated_seconds` the `estimate_cost` time) before each stage that refits, and checks each optional stage against `budget_seconds`. The final model is fitted regardless of the budget, and the log says when it went past it.

- Agent guide split into an entry point and topics (part of #429).
  - **Changed:** `SKILL.md` is now the entry point: what the library is, the first five steps of the workflow, the topic index and the rule to read a topic with `python -m mboostrs guide <topic>`. It is 5,866 characters (was 44,639). The rest of the workflow moved into topics: `screen-and-draft`, `fit-and-tune`, `interactions-and-terms`, `specification-search`, `distributional-branch`, `sensitivity-and-contrasts`, `explain`, `pitfalls`, `interpretation`, `reading-a-fit`, `closing-the-analysis`, `before-the-first-fit`, `quick-reference` (the runnable block), and `insights`. Three existing topics were over 9,000 characters (`discover` 12,509, `uncertainty` 10,925, `task-map` 9,146) and were split at a section heading: `discover` (`discover-limits`), `uncertainty` (`uncertainty-selection`) and `task-map` (`task-map-interpret-export`). The first part of each keeps its file name. `install_skill` copies the same files; the frontmatter (`name`, `description`) is unchanged.
  - **Added:** `tests/test_agent_guide_topics.py` fails when any guide file is over 9,000 characters or the entry point is 8,000 or more, and lists the topics that have no runnable example (`PROSE_TOPICS`), so a new topic without an example is a decision, not an omission.
  - **Changed:** `tests/test_skill_quickref.py`, `tests/test_family_set_export.py`, `tests/test_messages_416.py` and `tests/test_discoverability_79.py` read the moved text from its topic file. The examples themselves are unchanged.

- Factor learners are named by the formula term as written, at any number of levels (part of #429).
  - **Changed:** a categorical column's learner was named by its column when it had three or more levels (`ethnicity`) and by the term when it had two (`bols(ethnicity)`). It is now named by the term at every level count: `bols(ethnicity)`, `brandom(ethnicity)`, `brandom(ethnicity, lambda=0)`, or the bare name `ethnicity` when the formula writes the bare name. This applies to `base_learner_names()`, `coef()`, `selected`, `partial_effects` and the `gamboostLSS` / `glmboostLSS` learner names. Numeric learners are unchanged (`bbs(x)` is still `x`).
  - **Compatibility:** `which=` still accepts the column name for a factor learner (`which="ethnicity"`), so existing calls keep working. Dictionary keys from `coef()` change: a 3-level `bols` or `brandom` coefficient is under `"bols(ethnicity)"` or `"brandom(ethnicity)"`, not `"ethnicity"`.
  - Tests: `tests/test_factor_learner_names.py` (18 tests); the pins in six existing tests moved to the new labels.
  - Not changed: `glmboost` keeps its dummy-coded names (`twob`), and it refuses a 3-level factor under `bols`.

- Causal friction fixes from the rerun 4 benchmark (issue #444). Tests: `tests/test_causal_rerun4_friction.py`.
  - `ci_test(..., outcome_missing="ipw")` keeps rows with a missing `y` and weights each observed row by 1 / P(y observed | given, x) from a boosted response model, clipped to 0.01 to 0.99. The default `"error"` now names both remedies and why dropping rows can bias a role test. Missing values in x and the conditioning columns are still refused.
  - `CITest.status` is `"dependent"` and `skipped` is True when x or y is an exact linear combination of the conditioning set. The test is not run, and `describe()` and `repr()` lead with "UNDETERMINED". The p-value of 1 is a placeholder. The new fields `outcome_missing` and `n_missing_outcome` record the missing-outcome handling.
  - **Behaviour change:** a `bols` term means a linear term in gamboost, and a plain name means a spline. Linear baselines are now written `bols(name)` on every path, so `smooth=False` takes effect in the tuned fit and in the `"smooth+trees"` candidate, which previously fitted splines. Formula text in `diagnostics` changes from `x1 + x2` to `bols(x1) + bols(x2)`; the glmboost fit is the same. A formula with a categorical baseline is fitted by gamboost, so `aipw(smooth=False, tune=False)` runs with a string column.
  - `aipw(mstop=...)` accepts the mapping `diagnostics["nuisance_mstop_summary"]` (the median fold count per nuisance) with `tune=False`, which refits each nuisance at its own fixed step count. `diagnostics["nuisance_mstop"][name]["first_grid"]` now reports that nuisance's own count.
  - `EffectModel.predict(output=...)`: the default for `interval=False` is still the NumPy array. `output="numpy"` names the array form; `output="polars"`, `"pandas"` or `"records"` return a frame with an `estimate` column; `interval=True` returns a frame with `estimate` (with `output="numpy"`, a dict of arrays).
  - The constant-effect warning of `effect_model` says that the R-loss comparison has low power for detecting modifiers and names `effect_modifiers` for "who benefits more"; `EffectModifiers.describe()` says its tests concern the direction of the average effect along a covariate and not the R-learner fit.
  - `mb.collect_warnings()` is a context manager for scripts and loops: warnings with the same wording (numbers aside) raised in the block are issued once, with the count. Outside a block, the extrapolation summary of `ci_test` is shown once per process.

- Rerun 4 friction fixes (part of #429).
  - **Fixed:** `gamboost(...).confint(parm=..., method="bootstrap")` failed in every replicate for a string factor, with "column 'ethnicity' contains level(s) [0.0, 1.0] not present when the model was fitted". The default evaluation grid was the stored training frame, which holds the factor as its 0..L-1 codes, and each replicate re-applied the factor's levels to them. The default grid is now each replicate's own training rows (`newdata=None`); a `newdata=` frame is unchanged. Tests: `tests/test_rerun4_tools.py`.
  - **Added:** `GamRefit.coef(level=0.95, type="bayesian")` returns a Polars frame with one row per coefficient, in `vcov()` order: `term`, `coefficient`, `estimate`, `se`, `lower`, `upper`. `GamRefit.term_coefficients(term)` returns the rows of one formula term and lists the terms when the name is unknown. The coefficients are weights on the refit's sum-to-zero basis, so `coef()` names them by basis column (`bols(g)[1]`); level effects and their intervals are from `effects()`. The `vcov()` docstring states the index order.
  - **Changed:** `gam_refit` on a formula string says to fit a model with `mb.gamboost(formula, data)` first and then pass it to `gam_refit`.
  - **Changed:** the causal agent-guide topic is split. `topics/causal` is now an overview of the workflow in order with one example, and eight sub-topics hold the details: `causal-roles`, `causal-average`, `causal-outcome-models`, `causal-heterogeneity`, `causal-effect-model-checks`, `causal-effect-modifiers`, `causal-overlap-sensitivity`, `causal-instruments`. The overview is 4,964 characters (was 39,514 for the single topic); each sub-topic is under 9,000. Each sub-topic has a runnable example, and `SKILL.md` points at all of them. No other topic is over 15,000 characters (the largest is `discover`, 12,509).

- Propensity learner and second-stage row bootstrap for `aipw` and `effect_model` (part of issue #423, interval calibration, attempt 3).
  - `aipw(..., propensity_learner="smooth" | "auto" | "smooth+trees")` and the same argument on `effect_model`. The default is `"smooth"`, the propensity formula that the `smooth` argument builds (linear by default), so the default fits are unchanged. `"smooth+trees"` adds `btree(name, maxdepth=1)` for each numeric covariate with more than two distinct values, fitted by `gamboost`. `"auto"` fits both on the same folds and keeps the one with the lower out-of-fold binomial deviance. `diagnostics["propensity_learner"]` records the candidates, their deviances, the choice and each formula; `describe()` states the choice. With `propensity_terms=`, an explicit learner other than `"smooth"` is refused.
  - `"auto"` and `"smooth+trees"` are opt-in. Measured on 32 draws: the tree formula had the lower out-of-fold deviance on all of them, by 0.014 to 0.025, and against the true propensity on E11 to E16 the mean absolute error fell from 0.061 to 0.047 over all rows and from 0.031 to 0.007 above prior earnings of 40,000. The `aipw` ATE did not gain coverage: 0.88 with `"smooth"` and 0.81 with `"auto"` (28 and 26 of 32 draws), RMSE 1,236 and 1,250, and on E1 the `aipw` call took 16.6 s with `"smooth"` and 27.0 s with `"auto"`. The ATE misses above the truth come from the two-part outcome model, which the propensity formula does not change.
  - `effect_model(..., interval_method="second_stage_bootstrap")` resamples the fitting rows `bootstrap_draws` times (default 200), keeps the pseudo-outcome and nuisance predictions fixed, refits the REML second stage on each draw, and returns the full-sample estimate with the percentile interval of the draws. It accepts `nuisance=`. The draws live in `EffectModel.second_stage`; `diagnostics["interval"]` records the number of draws and how many converged.
  - **Behaviour change:** `interval_method` now defaults to `"auto"`, which is `"second_stage_bootstrap"` for a smooth second stage and `"refit"` for the boosted candidates. Smooth-fit intervals are wider. Per-row 90% coverage against the true effect, on the same 32 draws: `"refit"` 0.846 (mean width 4,718); `"second_stage_bootstrap"` percentile 0.907 (width 7,161); normal approximation from the draws 0.937 (width 7,667). Coverage above prior earnings of 60,000 was 0.25, 0.67 and 0.91 respectively. The estimate's mean error there was -1,496 and is the same for all three intervals, so the tail misses are bias, which the interval does not cover; `describe()` says so. Pass `interval_method="refit"` for the earlier intervals.
  - Cost: on one draw, `effect_model` took 35 s with `"refit"` and 56 s with the bootstrap, on a loaded machine; the 200 refits alone took 7.6 s.
  - `GamRefit.bootstrap(counts, response, weights=None)` returns a `GamRefitBootstrap` with one REML refit per draw of multiplicities. The Rust core has `gam::refit_row_bootstrap` and `gam::bootstrap_linear_predictor`, bound as `_rust.gam_refit_bootstrap_core`. `causal::binomial_deviance` is bound as `_rust.binomial_deviance`.
  - Measurement: 32 draws of the enrolment design from the causal benchmark generator, each with 12,000 training rows and 4,000 test rows (E1 to E30 with a planted effect, Z1 and Z2 without), 90% intervals, per-row coverage against each test row's true effect. Tests: `tests/test_interval_calibration_423.py`.

- Effect modifiers from AIPW scores: `effect_modifiers(result, data, covariates=None, method="blp", groups=4, level=0.9, overlap_bounds=(0.02, 0.98), adjust="holm")`.
  - For each covariate, a best linear predictor (BLP) slope of the `AIPWResult` scores on the covariate, standardized over the overlap rows, with HC1 standard errors (per level against the first level for a categorical covariate, with a joint test), and GATES: the mean score in quantile groups (levels for a categorical covariate) with the top-minus-bottom contrast and its standard error. `EffectModifiers.table()` gives both tests with intervals, Holm-adjusted p-values, direction and a decision; `method` selects the test that sets the decision.
  - `EffectModifiers.projection("all" | "passing" | names)` returns a `BLPProjection`, the joint regression on those covariates, and `mboostrs.rank_by_effect(projection, newdata, propensity=...)` ranks rows inside the overlap by it. The weighted least squares, HC1 covariance, Wald test, GATES contrasts and Holm adjustment are in `mboostrs-core` (`effect_modifiers`), bound in `mboostrs-py`.
  - Measured on ten effect draws and two null draws from the causal benchmark design (12,000 training rows, 4,000 test rows, level 0.9): the BLP-joint ranking claimed the four material modifiers with the correct direction in 10 of 40 pairs (0.25), against 3 of 40 for `effect_model`; it made no wrong-direction material claims (`effect_model`: 7), one claim on the two null draws (`effect_model`: 2), and an in-overlap regret of 0.648 (`effect_model`: 0.897). The standard errors treat the scores as fixed and omit nuisance error, and the regret against the best 20 percent of all rows cannot rank methods when those rows lie outside the overlap (about 92 percent of them here).
  - Joint regressions drop a covariate that is a linear combination of the others (for example age, education and experience in this design) with a warning, instead of failing.
  - This entry does not change `effect_model`; its defaults changed in the entry above.

- `discover` contrast of a dropped term (issue #415, small-sample coverage).
  - **Behaviour change:** `result.contrast(variable, ...)` for a variable whose mean term backward elimination dropped no longer returns an exact 0 with a zero-width interval. Its estimate comes from a REML refit of the model before elimination (the draft terms with the adopted interactions), its interval is that refit's interval widened to include 0, and its note begins "not selected". A variable that the draft never had, and any variable when the adopted model is distributional, still returns exactly 0.
  - Why: on the claims benchmark at 1,500 rows, backward elimination dropped both coordinates on about half the draws, and the spatial query then returned exactly 0. The widened interval is the rule of this change: it reports what the term would contribute without claiming it is in the model.
  - Measured on ten fresh draws of 1,500 rows (seeds 417200 to 417209) and two null draws of 1,500 rows (417300 and 417301), with the query contrast of each variable summed over the query's variables. Query coverage 0.90 before, 0.96 after (Q6 0.20 to 0.70; Q6 mean interval width 0.38 to 0.89). Mean-effect TPR 0.80 before and after; false mean terms 0.10 per draw before and after; no pair adopted before or after. Null draws: no term adopted, and query coverage 1.00 after (0.94 without the widening).
  - Measured on twenty fresh full-size draws (417000 to 417019): selection and estimates are unchanged, and query coverage is 0.82 before and after. The contrast change reads only dropped terms, and full-size draws kept both coordinates.
  - Considered and not adopted: a single joint `bspatial(latitude, longitude)` term in the mean draft. At 1,500 rows it raised Q6 coverage to 0.50 (0.20 before), but on the twenty full-size draws it raised adopted false pairs from 0.39 to 1.72 per draw on the eighteen draws that finished, and took about ten times as long per run. Two draws did not finish within 30 minutes.

- Repeated-split and bootstrap intervals for `effect_model` and `aipw` (issue #423, interval coverage).
  - `aipw(..., repeat_aggregation="average" | "median")`. `"average"` is the default and is the existing `repeats>1` behaviour: nuisance predictions are averaged over the partitions and one score is formed; its interval is the influence-function interval and does not include split-to-split variation. `"median"` computes an estimate and standard error on each partition and reports the median estimate, with variance equal to the median over partitions of `se_r^2 + (estimate_r - estimate)^2` (Chernozhukov et al., 2018). `diagnostics["repeat_estimates"]` and `diagnostics["repeat_standard_errors"]` list the partitions for any `repeats>1`.
  - `effect_model(..., interval_method="refit" | "repeats" | "bootstrap", interval_repeats=5, bootstrap_draws=200)`. `"refit"` is the default and is unchanged. `"repeats"` refits the nuisances and the smooth second stage on `interval_repeats` split seeds and combines the per-row estimates and standard errors by the median rule; `predict` then returns the median as the estimate. `"bootstrap"` refits the whole pipeline on `bootstrap_draws` row resamples and returns full-sample estimates with percentile intervals. Both need `terms="smooth"` (or `"auto"` resolving to smooth) and refuse `nuisance=`. The new fits are `EffectModel.interval_fits`; `diagnostics["interval"]` records the method.
  - The median rule (`median_repeat_combine`) and the percentile limits (`bootstrap_percentile_interval`, type-7 quantiles) are in `mboostrs-core` (`causal`), bound in `mboostrs-py`.
  - Measured on five fresh draws from the causal benchmark generator (own seeds 777000 onward; 12,000 training rows, 4,000 test rows, `terms="smooth"`, 90% intervals, per-row coverage against the true effect of each test row). Mean over draws: `"refit"` coverage 0.874 (range 0.777 to 0.957), mean width 4,953; `"repeats"` 0.874 (0.782 to 0.959), width 4,964; `"bootstrap"` with 30 resamples 0.867 (0.745 to 0.961), width 5,504. The default stays `"refit"`: neither alternative changed coverage, and the split term widened the repeats intervals by under 1%. The root mean squared error against the true effect was about 1.4 times the refit standard error. `aipw` ATE (truth: the training-sample ATE): coverage 4 of 5 draws for `repeats=1`, for `repeats=5` with `"average"` and for `repeats=5` with `"median"`; mean interval width 2,753, 2,721 and 2,770. Run time per draw on the shared machine, with load averages between 16 and 99 on 10 cores: one `effect_model` refit 91 s, `"repeats"` 475 s, `"bootstrap"` with 30 resamples 7,803 s; `aipw` 29 s, 155 s and 150 s. These times are inflated by the load and are reported for their ratios, not as absolute speeds.

- `effect_model` second stage, item 1 of issue #423. The R-learner pseudo-outcome `(y - m) / (t - e)` is unbounded where `|t - e|` is small. `effect_model` takes `learner="r_winsorized"` (the default), which clips the pseudo-outcome at its 1st and 99th percentiles over the rows with positive weight (`winsorize_pseudo_outcome` in the Rust core; bounds in `diagnostics["pseudo_bounds"]`), and `learner="r"` for the unbounded pseudo-outcome of earlier versions. **Behaviour change:** the default fit differs from earlier versions. It warns when the largest absolute fitted effect exceeds 10 times the interquartile range of the observed outcome (`diagnostics["effect_range"]`). Categorical covariates and modifiers are accepted, as in `aipw` (#422). `EffectModel.propensity` holds the cross-fitted propensities as before and is also callable: `fx.propensity(newdata)` returns `e(x)` for new rows, and `rank_by_effect` uses it. `AIPWResult.propensity(newdata)` is not added: the result stores no propensity model, and `aipw` is changed by a separate change.
  - The smooth second stage fits each modifier's `bbs` over its 1st to 99th percentile range of the fitting rows (`boundary.knots`), leaves the outside rows out of the fit, and holds predictions at the range edge (`extrapolation="clip"`, now the default of `EffectModel.predict` for `terms="smooth"`). A positive modifier whose 99th percentile is at least 3 times its median is entered on the log scale. The scales are in `diagnostics["smooth_scale"]`. The rule is `modifier_scale` in the Rust core. `gam_refit` now reads `boundary.knots=c(lo, hi)` in a `bbs` term, and the interaction-term parser no longer reads `knots` out of `boundary.knots` (it used to fail on that key).
  - Measured on five fresh draws from the causal benchmark generator (seeds 20260011 to 20260015; 12,000 training rows, `terms="smooth"`). Mean per-row RMSE against the true effects: training rows 312,623 before the change, 29,851 with the winsorized pseudo-outcome alone, and 5,816 with the range rule; held-out rows 104,897, 13,558 and 4,226. Draws with an effect beyond 10 times the outcome IQR: 5 of 5 before, 3 of 5 with the winsorized pseudo-outcome alone, 0 of 5 with the range rule. The 90% per-row coverage: training rows 0.80, 0.77 and 0.76; held-out rows 0.85, 0.82 and 0.83. Coverage of the training rows stays below nominal.

- Claims rerun: `discover` crash on a log-drafted predictor with missing values, and interaction recall (issue #415).
  - **Fix:** `discover` raised `log(x) produced n non-finite value(s)` when a predictor was drafted on `log(x)` and had missing values, because the transform read the value before the imputation step filled it. `gamboost`, `glmboost` and `gamboostLSS` now plan `na_action="impute"` before the transforms are extracted, and a replayed imputed column is written back before a transform reads it. The fit and `predict` with a missing value of such a predictor are finite. The claims benchmark data, whose `total_bedrooms` has missing values and is drafted on `log`, reproduced it.
  - `discover` interaction screen: each pair's score is the mean of `screen_interactions` over five row splits (seeds `seed` to `seed + 4`). With one split, the true age x income pair's rank among the 78 pairs ranged from 1st to 12th across ten draws and three split seeds.
  - `discover` interaction smooths take each column's main-term scale: a column drafted on `log(x)` gets its pair tensor on `log(x)` (`pair_term_for(..., transforms=)`). Before, the pair used the raw column, while its main term used the log.
  - **Behaviour change:** an interaction is adopted at `2 * sqrt(log(5))`, about 2.54, change se (was 2.0), the bar for five candidates. Mean terms and spread keep the two-standard-error rule.
  - Measured on ten fresh full-size draws of the claims benchmark generator (15,480 training rows, seeds 415000 to 415009) and two null draws (`discover`, end to end): age x income adopted on 4 of 10 draws before and 9 of 10 after; false interaction pairs 6 before and 3 after (0.60 and 0.30 per draw); no pair on the two null draws before or after. Mean-effect TPR was 1.00 in both. Mean query coverage (8 queries, 90% intervals) 0.80 before, 0.84 after.
  - `discover` at 1,500 rows is unchanged by these edits (ten fresh draws): mean-effect TPR 0.80, query coverage 0.80. Its coverage shortfall is the spatial query: backward elimination drops latitude and longitude on several draws, and the query then returns exactly 0.

- Claims rerun, attempt 2: `discover` interaction recall and a 91-minute null-draw run (issue #415).
  - **Fix (stall):** on claims-benchmark null draw 417101 the first mean-effects comparison fitted a `brandom(total_bedrooms)` term with 1,779 levels. The label test (`label_evidence`) read that whole-number column as labels on pure noise: its z-statistic was 5.8, and under permutation the statistic has a standard deviation of 1.4 to 1.9, not 1 (it flagged 3 of 180 permuted responses). A `brandom` term costs about n x levels^2 per refit, so each 200-step gamboost fit on 15,000 rows took 15 to 33 seconds with it, against 0.24 seconds without it. `label_evidence` now returns `None` for a column with more than `LABEL_MAX_LEVELS = 200` distinct values, so such a column is drafted as a smooth. The cap also applies to `autospec`, which calls the same test. Null draw 417101 now finishes in 150 seconds, against 142 to 200 seconds for the other null draws.
  - **Fix (adoption):** an interaction is adopted only if its change in out-of-fold risk is negative. A pair that the boosting never selects has change and `change se` both exactly 0, and `0 <= -bar * 0` held, so such pairs were adopted. Five of the nine false pairs in the 20 full draws were of this kind.
  - **Behaviour change (recall):** an interaction is tested against the mean model with both of its continuous main effects refitted at the tensor's margin smooth, `bbs(x, df=2.5, knots=5)`, in place of the draft's `df=4` main effects. The reference then carries the same smooths the tensor is built from, so the paired fold differences measure what the tensor adds. Measured on 20 full-size draws (15,480 training rows, seeds 417000 to 417019) and 4 null draws (seeds 417100 to 417103), with `discover` end to end: age x income adopted on 20 of 20 draws (was 12 of 20), 0.15 false pairs per draw (was 0.45), no adoption on the null draws (none before, on the three null draws that finished). The mean-effect TPR is 1.00 in both. Mean query coverage (8 queries, 90% intervals) is 0.87 (was 0.82), with the age query at 20 of 20 and the roof-class query at 2 of 20 in both.
  - The false pairs that remain are longitude x ave_occup on two draws and latitude x median_house_value on one, with changes of 0.00005 to 0.0004 in out-of-fold risk against 0.0019 to 0.0033 for the planted pair. The bar stays at `2 * sqrt(log(5))`, about 2.54: lowering it to 2.0 gives 0.20 false pairs per draw on the same runs, and no null draw supports a bar below about 1.8 (the lowest `change / change se` over 60 null tests was -1.75).
  - Not adopted, measured on the same draws: a candidate pool of ten pairs (12 of 20 at 0.35 false pairs), quantile boundary knots on the margins with five knots (12 of 20 at 0.05) or three knots (10 of 20 at 0.05), and quantile knots with the matched reference (20 of 20 at 0.35).
  - `discover` at 1,500 rows is unchanged by these edits (ten fresh draws): mean-effect TPR 0.80, query coverage 0.80. Its coverage shortfall is the spatial query: backward elimination drops latitude and longitude on several draws, and the query then returned exactly 0 (the entry above replaces that exact 0 with the dropped-term contrast).

- `aipw(outcome_model="two_part")` and the `"auto"` rule for zero-inflated outcomes (part of #423). A continuous outcome that is zero or positive with some zeros (for example earnings with zero for people without a job) is fitted as two models: a Bernoulli model for P(Y > 0 | x, a) and a Gaussian model for log Y among the positive rows, each with the treatment as a linear term. The mean is `P(Y > 0) * exp(m + s2 / 2)`. `"auto"` now picks it when the sparse-arm check flags a bin and the outcome has zeros; before, it picked the pooled model with treatment interactions in that case. **Behaviour change:** `aipw` with `outcome_model="auto"` on such an outcome gives a different estimate when a bin is flagged. Other outcomes, and unflagged fits, are unchanged. Simulation on ten effect and ten no-effect draws of 12,000 rows: all-rows ATE RMSE 908 against 2,463 for the previous auto choice on the effect draws, with 10 of 10 90% intervals covering the truth against 6 of 10. Outcomes with negative values, binary outcomes and outcomes with no zeros are refused with `"two_part"`, and `outcome_interactions=True` is refused with it. The core functions `log_residual_variance` and `two_part_mean` are in `mboostrs-core` and bound as `_rust` helpers.

- Causal benchmark rerun 3, items 5 to 10 (issue #429).
  - `ci_test` tests a categorical `x` with more than two levels jointly, as a factor: one 0/1 indicator per level after the first observed level, residualized on `given`, with a chi-squared test with one degree of freedom per indicator. A two-level categorical `x` or `y` is one indicator. A categorical `y` with more than two levels is refused, and the message names the per-level route (`data[y] == level` as a 0/1 column). `_reduce_conditioning` returns which x columns it kept. The result's `functions` counts the indicators.
  - `cv_risk(..., mstop_grid=[...])` returns a frame with one row per grid value (`mstop`, `risk`, `fold_se`). Each fold is fitted once to the largest grid value and its path is cut to each value, so the grid costs the fits of one call. Its `risk` equals `cv_risk(mstop=m)` at that value (checked to 1e-9). It supports `score="family"` on a standard single-response family with `na_action="error"`. `EvaluationResult.aggregate` is added: the same number as `score` and `float(result)`, which is what the float return of `cv_risk` equals. `GAMBoostModel.cvrisk(folds=int)` builds a k-fold split on `seed`, as the functional cvrisk does. `tune_mstop` fills `fold_scores` with each fold's out-of-fold risk at the chosen `mstop` (their mean is the curve's value there); it was empty before.
  - `rank_by_effect` reports ties at the selection boundary. `OverlapRanking.tied_rows` counts the rows that share the estimate at the cut when that estimate also falls on an unchosen row; `tie_break` names the rule. Rows with equal estimates are ordered by interval width (the span of `lower` to `upper`, or `se` when only that is given), narrower first, then by row position. Without an interval the order is row position. `describe()` states the count and the rule, and `table()` has a `tied_at_boundary` column. **Behaviour change:** rows that tie at the boundary can now fall in a different order from earlier versions, when an interval is available.
  - `confounding_benchmark` accepts `(result, data)` as well as `(data, result)`; the two are told apart by type. The canonical order is `(data, result)`. `confounding_sensitivity` keeps the result first, since its estimate can be given as numbers.
  - `gam_refit(...).effects(rows, level=)` is documented in the `uncertainty` agent topic as a method of the `GamRefit` object, with its return shape and options.
  - `iv_effect(data, spec, instrument, *, folds=5, seed=0, ...)` and `IVResult`: a cross-fitted partially linear IV estimate (DML-IV) for a binary treatment and a continuous or binary instrument. Out of fold, boosted models predict the outcome, the treatment and the instrument from the covariates; the estimate is `sum(z y) / sum(z d)` over the residuals. The variance is the influence-function sandwich, with a normal interval at `level`. The first stage gives the partial R2 and partial F; a partial F below `weak_threshold` (10) sets `weak_instrument` and warns. `describe()` states the estimand (a weighted average of effects among compliers under monotonicity) and the assumptions. The ratio, influence function and variance are `iv_ratio` in `mboostrs-core`, bound as `_rust.iv_ratio`.

- `aipw` and `effect_model` nuisance handling, items 1 to 4 of issue #429.
  - `aipw` and `effect_model` read `propensity_clip`, `outcome_missing` and `overlap_bounds` from the `CausalSpec` when the call does not pass them. Their defaults are now `FROM_SPEC`, so a spec's value applies. **Behaviour change:** a spec built with a non-default `propensity_clip`, `outcome_missing` or `overlap_bounds` now changes these calls; the default spec gives the same results as before. An explicit argument always wins, and `propensity_clip=None` still means no clipping.
  - Warnings from the nuisance fits are issued once per call, after all folds. Each names its nuisance model (propensity, response, pooled outcome, control-arm outcome, treated-arm outcome) and the argument that controls it. Extrapolation warnings name `extrapolation=`; the flat-curve warning names `flat_tolerance=`. The penalty-drift and other warnings from those fits are named the same way. **Behaviour change:** `aipw` warns once per nuisance per call, not once per prediction.
  - `aipw` and `effect_model` take `extrapolation=` (default `"warn"`), passed to the nuisance predictions and their tuning, and `flat_tolerance=` (default 0.01, the `tune_mstop` default), applied to the tuned nuisance curves. `flat_tolerance=None` turns the flat-curve check off. Nuisance tuning runs with `tune_mstop`'s own flat-curve check off, so each warning comes from this check and names its nuisance.
  - `aipw(propensity_terms=, outcome_terms=)` replace the formula that `smooth` builds for the propensity and outcome models. Each takes a right-hand side string, which is fitted by `gamboost`. With `outcome_model="pooled"` the treatment is added to `outcome_terms`. `outcome_interactions=True` is refused with `outcome_terms`; the interaction is written inside the string with `by=`. The formulas used are in `nuisance["terms"]`, and `nuisance["custom_terms"]` says which models got one. A string with `~`, an empty string, or the treatment as a bare term is refused.
  - `aipw(smooth=)` takes a dict with the keys `"propensity"`, `"outcome"` and `"response"`, for example `smooth={"propensity": False, "outcome": True}`. A key that is not given takes `False`. `nuisance["smooth_by_nuisance"]` holds the flags; `nuisance["smooth"]` keeps the value as passed.
  - An outcome column with `None` among its numbers (from `to_dict()`) is read as numeric, with `None` as a missing value, and the usual `outcome_missing` rule applies. An outcome whose present values are text is refused with a message that says missing values must be null or NaN in a numeric column.
  - `confounding_benchmark` refits with the result's `smooth`, `outcome_model` and other recorded options. A result fitted with `propensity_terms` or `outcome_terms` is refitted with those formulas minus every term that names the omitted covariate (issue #429); the rows report the reduced formulas in `propensity_terms` and `outcome_terms`. When no term is left, the call refuses before any refit and names the custom formula.

- Causal benchmark rerun 2, items 2, 3, 4, 6, 8 and 9 (issue #423).
  - `ci_test` drops a conditioning column that is an exact linear combination of the others (`experience = age - education - 6`). The result lists it in `dropped` and names it, with its coefficients, in `notes`. When `x` or `y` is itself a combination of the conditioning set, the result has `p_value` 1 and `statistic` 0, with a note. The call used to raise `MboostrsRuntimeError: the warm-start state does not describe this fit`.
  - **Core fix:** the warm-start check compared a resumed risk with a direct recomputation, with a tolerance of `1e-8` relative to the resume value. The retained risk is updated incrementally, and its rounding scales with the risk the path started from. On an almost exact fit the resume value is near zero, and a drift of `2e-8` was refused. The tolerance is now `1e-7` relative to the larger of the resume and start risks; a different response or weights still differs by the start risk's order and is refused. `GAMBoostModel.continue_to` and `tune_mstop` on such a fit no longer raise.
  - `ci_test` takes `extrapolation=` ("warn", "error", "allow", "clip"). Its internal extrapolation warnings are combined into one per call, and name the column of `given`, not the alias `__ci_givenN`. `given` accepts categorical and string columns, entered as factors with `bols`. A categorical `x` or `y` is refused with the remedy.
  - `ci_test` regressions are capped at 1,000 steps (`CI_MAX_MSTOP`). Before, the cap was `tune_mstop`'s 10,000, and a residual regression whose risk kept falling ran to it. On the benchmark's 12,000-row file, a call on `months_unemployed` took 34.7 s and on `education` 7.9 s; with the cap at 1,000 they took 7.2 s and 3.1 s. The statistics moved under 1.5%. A regression that reaches the cap is listed in `notes`. The six-column run (`folds=3`) took 47.5 s before and 17.9 s after, on a machine shared with other jobs.
  - `causal_trap_report(frame, spec, outcome_missing=)`: with `"ipw"` the missing-outcome finding is `outcome_missing_ipw`, which says the target is all rows under a missing-at-random assumption, in place of the subpopulation advisory. The default is still the subpopulation advisory. `mb.causal_trap_report` was already exported at the top level; the causal agent topic now documents it.
  - `CausalSpec` carries `propensity_clip`, `outcome_missing` and `overlap_bounds` (defaults `(0.01, 0.99)`, `"error"`, `(0.02, 0.98)`), validated at construction. `overlap_report` reads `overlap_bounds` when `bounds` is not passed, and `causal_trap_report` reads `outcome_missing`. `aipw` and `effect_model` do not read the spec yet: their call sites are unchanged, and they still take their own defaults.
  - `tune_mstop` warns when the chosen step count's cross-validated risk is within `flat_tolerance` (default 0.01, relative to the intercept-only risk at step 0) of that intercept-only risk. The message says the curve is flat and the fit is close to a constant. `flat_tolerance=None` turns it off. `ci_test`'s internal regressions run with the check off.
  - Warning count of a typical causal pipeline on the benchmark's 12,000-row training file (`aipw` with `smooth=True`, `outcome_missing="ipw"`, `overlap_report`, `ci_test` on six columns with `folds=3`, `effect_model` with `terms="smooth"`): 73 warnings in 13 distinct messages before this change, 23 in 11 after. The `ci_test` calls went from 56 warnings to 6 (one per call).
  - The sparse-arm warning already reads "of all rows" (issue #423 item 7, fixed in the #424 changelog entry); a test checks the wording.

- `aipw(..., target="all" | "overlap")` (issue #423). `target="all"` is the default and is the ATE over every row, unchanged. `target="overlap"` is the overlap-weighted ATE: each row is weighted by `e (1 - e)` with its cross-fitted propensity, so rows near propensity 0 or 1 count little. It does not divide by the propensity, so `propensity_clip` is not applied to it; the response clip still applies under `outcome_missing="ipw"`. Only `estimand="ate"` accepts it. The result, `describe()`, `repr()` and `diagnostics["target"]` name the target, and `describe()` of an all-rows result with more than 5% of rows below 0.05 or above 0.95 points to `target="overlap"`. The overlap report's `estimate_inside` is not reported for the overlap target, and `overlap_frame` weights rows by `1 - e` (treated) and `e` (control). `confounding_benchmark` refits with the result's target. The score is `augmented_scores(..., "overlap")` in `mboostrs-core`. On a reduced thin-overlap design (n = 3,000, 20 seeds, effect proportional to baseline), the all-rows ATE had a bias of -86 and coverage 1 of 20 of its 90% intervals; the overlap target had a bias of +8 and coverage 17 of 20 of its own truth. Diagnosis: the all-rows error is mostly the outcome model's extrapolation in the tail (the plug-in term) and the propensity clip, which removes part of the augmentation there.

- `aipw` chooses its outcome model from the data (issue #423). `outcome_model` takes `"auto"`, and `"auto"` is the default. It runs the sparse-arm check on the cross-fitted propensity: a flagged bin selects `"pooled"` with `outcome_interactions=True`, and no flag selects `"separate"`. The choice and its reason are in `diagnostics["outcome_model_choice"]`, and `describe()` states them. **Behaviour change:** a default `aipw` call on a design with a flagged bin now fits the pooled model with interactions and gives different estimates and intervals; a call with no flagged bin is unchanged. `outcome_model="separate"` and `"pooled"` keep their behaviour. `outcome_interactions=True` with `"separate"` is still refused, and with `"auto"` it selects the pooled model. The sparse-arm warning names the model that was fitted, and "of the all rows" now reads "of all rows". `modeler` causal results call `aipw` and change in the same way. On the simulated sparse design in the causal topic, the pooled model with interactions had a smaller bias than separate models over 10 seeds (-107 against -131 dollars for a true 1,700); no model's 90% interval covered the true effect there.

- `confounding_benchmark` refits each omitted-covariate estimate with the result's resolved `outcome_model`, its `outcome_interactions`, `mstop` and `max_mstop`. It previously passed no outcome model and no `max_mstop`. The nuisance record's `mstop_max` key, which held the first grid value, is removed; use `nuisance["mstop"]`.

- Causal benchmark rerun, items 5 to 9 (issue #420).
  - `gam_refit(...).predict` and `gamlss_refit(...).predict` take `extrapolation=` ("warn", "clip", "error", "allow"), with the boosted predict's meaning and default. The range check covers the covariates of `bbs`, `bspatial` and `%X%` B-spline terms against their training ranges; `bols` and `brandom` are not checked, as in the boosted predict. `interval=True` frames have an `outside_range` column. `EffectModel.predict` passes the policy through.
  - `effect_model` scores a constant effect (`tau` equal to the ATE, the R-loss minimizer on each fold's training rows) on the same folds. `EffectModel.constant_rloss` and `diagnostics["constant_effect"]` hold it, and `describe()` reports it beside the candidates. The call warns when the chosen candidate does not beat the constant by more than its standard error, and when a boosted candidate is tuned to fewer than 5 steps. `effect_model` issues its extrapolation warnings once per call.
  - `confounding_sensitivity` and `confounding_benchmark` document `output=` and return plain values with `output="records"`; the default is still a Polars frame.
  - The penalty-drift warning is counted within a routine (`routine_warnings` and `one_summary` routines, including `effect_model`) and reported once with the number of fits affected. Fits run inside mboostrs were already silent; a user loop of direct `gamboost` calls still warns once per call.
  - `rank_by_effect(effect, newdata, *, share=0.2, propensity=None, bounds=(0.02, 0.98))` ranks the rows inside the overlap region by estimated effect, returns the chosen positions, the gain of the top group with an interval for a smooth `EffectModel`, and the rows outside the overlap separately. Returns an `OverlapRanking`. Documented in the `causal` agent topic.

- Causal effect intervals, overlap and speed (issue #416, items 3 and 5).
  - `effect_model(..., terms="smooth")` fits `tau(x)` as a penalized additive model with REML smoothing (`gam_refit`) on the R-loss pseudo-outcome, weighted by `(t - e)^2`. `predict(newdata, interval=True, level=)` returns a frame with `estimate`, `se`, `lower`, `upper` and `outside_overlap`; `se` is the summed-prediction standard error from the full coefficient covariance. The interval covers the second stage's sampling variation given the cross-fitted nuisances, not nuisance error or the choice of modifiers. On simulated effects the 95% per-row intervals cover 0.97 and the 90% ones 0.93 to 0.94 of fresh rows, near nominal. `"stumps"` and `"joint"` remain boosted and refuse `interval=True`, naming `terms="smooth"`.
  - `effect_model` scores all candidates by their out-of-fold R-loss on the same folds, with one normalization. The smooth candidate no longer runs a boosted step search to `max_mstop`. On 10,000 rows with 8 covariates, where the effect depends on all of them, the time fell from 15.6 s to 6.4 s with the same rmse (0.323 to 0.324); on a second 10,000-row case with a one-modifier effect, from 7.2 s to 4.9 s (single runs, same machine). The remaining cost is the nested nuisance tuning inside `aipw`'s rule, which is not shared across folds.
  - `overlap_report(source, data=None, bounds=(0.02, 0.98), covariates=None)` reports the share of rows whose propensity lies outside the bounds, the covariate region where most rows are outside (numeric: the span of the flagged quantile bins; categorical: the flagged levels), and the estimate restricted to the rows inside the bounds next to the full estimate. It accepts a propensity array, an `AIPWResult` or an `EffectModel`. `aipw` and `effect_model` store it in `diagnostics["overlap"]`.
  - `EffectModel.predict` warns once with the count when new rows have an estimated propensity outside `overlap_bounds`, and flags them. A new row missing a covariate of the propensity model gets no flag and a warning.
  - `effect_model` and `aipw` take `overlap_bounds=`, default `(0.02, 0.98)`.

- Variable importance, bootstrap intervals and LSS prediction intervals moved to the Rust core (issue #407, items 2 and 3). `varimp()` for `glmboost`, `gamboost` and LSS fits attributes each update's risk change through `mboostrs_core::inference::varimp_attribution`; the reductions, selection frequencies and percentages equal the previous Python values to 1e-12 (checked on fitted models). The bootstrap in `confint()` (`glmboost` and `gamboost`, row and `groups=` resampling) keeps its refits in Python, and the count of failed, untuned and mstop-capped replicates, the percentile interval (`percentile_interval`, numpy's linear quantile with NaN skipped; the centred GAM band centres each replicate first) and the group draws move to the core. The row-bootstrap draws were already the core's `cv_weights` stream and are unchanged, so the row intervals equal the previous ones to 1e-12. **Changed: `groups=` bootstrap draws.** They came from `numpy.random.default_rng`, and they now come from a ChaCha8 stream seeded with the same `seed`. A given seed therefore gives different group-bootstrap intervals from earlier versions, reproducibly; the interval definitions are unchanged. Every LSS family's quantile is computed by `mboostrs_core::lss_quantile` for 23 families (`GaussianKnownVarLSS` and `BinomialLSS` take their extra per-row argument as before), and `prediction_interval` calls it directly. The Python `quantile` methods are thin wrappers, and the quantiles equal the previous values to 1e-12 for every family. Refusals are now `MboostrsValueError`, a `ValueError` subclass, with the same message. `prediction_interval` on `GaussianKnownVarLSS` raises `NotImplementedError` naming the reason; before, it raised a `TypeError` for the missing `s2` argument. R: `varimp()` for `mboostrs_glmboost` and `mboostrs_gamboost` calls the core through `varimp_attribution_r`, and an R-parity test checks the per-learner reductions against mboost's `varimp(percent = FALSE)`. Not moved: the R `confint()` and R `lss_quantile()`, which still use R numerics. `unseen_levels="marginal"` on `prediction_interval` still calls a method that does not exist (`_marginal_prediction_interval`); that path was never implemented.

- Causal benchmark friction, items 6-12 (issue #416). `gam_refit(...).predict(newdata, interval=True, level=, corrected=, output=)` returns `estimate`, `se`, `lower` and `upper` for the summed prediction: the intercept plus every term, with `se = sqrt(x' V x)` from the full coefficient covariance across terms (`mboostrs_core::gam::RefitResult::linear_predictor`). `gamlss_refit(...).predict(..., interval=True)` gives the same per parameter, from that parameter's own coefficients (`LssRefitResult::parameter_predictors`). `se` is on the link scale; `type="response"` maps the bounds through the inverse link. `tune_mstop`, `compare_specifications`, `term_report` and `cv_risk` take `extrapolation=` ("warn", "error", "allow", "clip") for their fold-refit predictions; the warning names the argument on the calling routine. `tune_mstop`'s default family path builds its folds on the full data, so the argument has no effect there. A formula string with no `~` (`tune_mstop("bbs(x)", ...)`, `gamboost("bbs(x)")`, `term_report` without `response=`) raises a `ValueError` naming the response and the `y ~ <rhs>` fix. `ci_test` accepts any conditioning column name: each column is copied under an internal alias before the formula is built. A bare categorical name in a `gamboost` formula is coded as `brandom(name, lambda=0)` (the fit `bols(name)` gives) and warns once, naming the column; the agent guide now says so. The Binomial parity note states what it changes in the summary (boosting steps about 1.44 times a natural-log gradient's at the same `nu`; the reported risk is the unscaled negative log-likelihood). The penalty-drift warning says it changes no fitted value.

- `aipw` sparse arms, outcome model, step cap, learner choice and ATT with missing outcomes (issue #420, items 1 to 4).
  - `aipw` counts treated and control rows in each decile of the cross-fitted propensity (`diagnostics["sparse_arms"]`). A bin is flagged for an arm when it holds more than 5% of the target rows and fewer than 20 rows of that arm. With `outcome_model="separate"` a flagged bin warns, naming its propensity range, its counts and the covariate ranges of its rows.
  - `aipw(outcome_model="pooled")` fits one outcome model on all observed rows with the treatment as a linear term. `outcome_interactions=True`, which needs the pooled model, multiplies each baseline term by the treatment. The default `"separate"` is unchanged.
  - `aipw(max_mstop=)`, default 1600 and never below `mstop`. **Changed:** with `tune=True`, `mstop` is now the first grid of the inner search, not its largest step count. The grid doubles while the cross-validated risk still falls, up to `max_mstop`. A nuisance still falling at `max_mstop` triggers a warning that names `max_mstop`. `diagnostics["nuisance_mstop"]` records each nuisance's step counts by fold.
  - With `smooth=True` a numeric baseline with fewer than ten distinct values enters linearly (`bols`), the rule `effect_model` uses. String, categorical and boolean baselines are accepted: a factor is coded with treatment contrasts, and `diagnostics["balance"]` has one row per level. `nuisance["terms"]` records the formula terms of each nuisance.
  - `outcome_missing="ipw"` now supports `estimand="att"`. Each residual term of the ATT numerator is weighted by its row's response propensity, and the target is all treated rows, observed or not. Its influence-function interval treats the cross-fitted nuisances as fixed, as the ATE interval does.
  - Removed refusals: `aipw` no longer raises `NotImplementedError` for categorical baselines or for the ATT with missing outcomes. `effect_model` still refuses categorical baselines.

- Causal estimators: missing outcomes, observation weights and sensitivity inputs (issue #416, items 1, 2 and 4).
  - `aipw` and `effect_model` take `outcome_missing="ipw"`, which treats the outcome as missing at random given the covariates and treatment. A response model P(observed | x, a) is cross-fitted like the propensity, and the augmented score divides the observed-row terms by the response propensity. The ATE then covers all rows. The result reports the response propensity range and its effective sample size, and keeps the observed indicator and response propensity per row. `outcome_missing="error"` (the default) still refuses a missing outcome, and its message names the column, the count and both remedies. The ATT has no missing-outcome form and raises `NotImplementedError`.
  - `aipw` and `effect_model` take `weights=`, observation weights read as frequencies. They enter the nuisance fits, the averaged score and the R-loss, and the uncertainty is the one for the weighted sample.
  - `confounding_sensitivity` takes `estimate=` and `se=` (or `scores=`, a per-row score vector) in place of an `AIPWResult`. The interval is normal at `level` when built from an estimate; an `AIPWResult` keeps its own interval.
  - **Behaviour change:** `aipw` clips propensities to `(0.01, 0.99)` by default; it previously did not clip. `effect_model` already clipped at these bounds, and the two estimators now agree. The reason is that without clipping a row with propensity 0.001 receives an inverse-probability weight of 1000 and can dominate the estimate; the bounds cap each inverse propensity at 100. The change moves `aipw` estimates only for rows whose propensity lies outside the bounds. Pass `propensity_clip=None` for the previous behaviour. `mboostrs.modeler` causal results call `aipw` and change in the same way.
  - The AIPW score, the response-weighted R-loss, the influence variance (frequency or cluster), the effective sample size and the mean with its IID standard error are in `mboostrs_core::causal`, bound to Python as `augmented_scores`, `rloss_transform_weighted`, `influence_variance`, `effective_sample_size` and `mean_and_iid_se`.
  - Known limitation: the learners centre their design by the unweighted mean, so a weighted nuisance fit follows a different boosting path from the fit on replicated rows. The two agree once boosting converges (1e-8 at 1000 steps on a 400-row example; 1e-3 at 100 steps).

- `tune_mstop` resumes each fold's path across the doubling grids for distributional (LSS) specifications and for response scores (issue #189). A response-score fold on the Gaussian Gram route (for example a `bbs` term with `score="mse"`) keeps no residual to resume from, so it is refitted from zero at each grid, as before; every other fold is continued from where the previous grid stopped, so its fit count is folds + 1. The doubling loop of the LSS and response routes runs in the Rust core (`mboostrs_core::grid_search::search_grid`); LSS folds resume through `mboostrs_core::lss::try_cvrisk_lss_search`, and a diagonal LSS path can be stopped and continued (`LssDiagonalPath`). Results match the refit-from-zero search to 1e-12 (`tests/test_tune_mstop_resume.py`). On a GaussianLSS tune with 5 folds, grid 50 to 100 to 200, n = 20,000, the wall time fell from 0.59 to 0.65 s to 0.22 to 0.23 s on the same machine. The Gram-route response tune takes the same time as before (0.36 to 0.39 s, same fit count).

- Conformal intervals run in the Rust core (issue #407, item 1). The calibration scores, the finite-sample conformal quantile, the CV+ order statistics and guarantee, the CQR and distributional bands with the inversion fix, and the tuned step grid for CQR quantile fits are in `mboostrs_core::conformal`, bound from Python as `mboostrs._rust.conformal_*`. The refits, the calibration split and fold assignment, and the family quantiles stay in Python. Intervals agree with the former Python numerics to 1e-12 for split, CV+, CQR, distributional and held-out calibration; `tests/test_conformal_core.py` keeps the former numerics as an oracle. The CV+ path no longer builds the n x n prediction pool; it passes the folds x target rows matrix in blocks. A NaN prediction or score is refused with a value error. `conformal_interval()` is added to the R package for glmboost and gamboost fits with `split`, `cv+` and `cqr`. The R version draws its split and folds with R's random numbers, so a seed selects different rows than the Python package does. The distributional method is Python-only.

- Evaluation results for survival, LSS and failed folds, and two new fold designs (issue #72). `evaluate_specification` and `cv_risk` accept a survival family (`CoxPH`, `Weibull`, `Loglog`, `Lognormal`, `Gehan`, `Cindex`) with a `Surv(time, status)` response or a `(time, status)` pair; the held-out loss is the family loss, and each record carries the `event`. Survival scoring needs `score="family"`; a Surv response with a non-survival family, or the reverse, is refused, and so is a `cbind` or (outcome, variance) LSS response, which `cv_risk` does not read yet. A fold whose training or evaluation side has no rows after `na_action="omit"` (`FoldError`) is kept as a record with `status="failed"`, its `error` and a NaN score (`result.failed_folds`, `fold_frame()`), so the result has no score and `comparison_issues` reports "failed folds"; the float return of `cv_risk` still raises. `known_group_folds(groups, B)` holds out rows whose group also trains each fold; the result's `evaluation_task` is `"known_group"`, distinct from `"new_group"` (`grouped_folds`). `rolling_folds(..., time=, embargo=)` orders rows by timestamp (numeric or `datetime64`) and drops training rows inside a gap before each judged block; `check_leakage` compares timestamps and reports embargo violations. `cvrisk_gamboostLSS` and `stabsel_gamboostLSS` take `na_action=` with the meaning it has in `gamboostLSS`. `family_held_out_score` accepts the `[time, status]` survival layout.

- Discoverability, items 2, 3 and 5 (issue #79). A task-to-API map, `agent_guide("task-map")`, gives the Python, R and Rust core entry point for fit, tune, evaluate, diagnose, compare, interpret, predict with uncertainty and export, and says where a frontend has no such function. Its names are checked against the package, the R NAMESPACE and the Rust source by `tests/test_discoverability_79.py`. Zero-row tables: `to_frame()` of `AicResult`, `AIPWResult`, `ContrastResult`, `MarginalEffectResult`, `ElasticityResult`, `DistributionEvaluationResult`, `EvaluationResult` (no retained records), `CvRiskResult`, `StabSelResult`, `LssCvRiskResult` and `LssStabSelResult` returns a frame with the same columns and no rows, in polars, pandas and records form; before, a zero-row result had no columns. `LSSRegressor.predict` returns the conditional mean (`mu` for gaussian, gamma and negativebinomial; the Student-t mean where `df > 1`). A Student-t row with `df <= 1` has no finite mean, so `predict` and `score` raise a `ValueError` that names the family and points to `predict_params` and `predict_interval`. `LSSRegressor` is tested in sklearn pipelines, grid search and sample-weight metadata routing.

- R parity tests fail on an R error (issue #405). About 40 `r_parity` sites caught any exception from an R call and skipped the test, which hid the Binomial response defect in issue #402. A new `r_call` helper in `tests/conftest.py` skips only when rpy2 or a named R package is missing; an error raised by the R call fails the test. `compare_to_r` converts the 0/1 response to a factor on the R side for `Binomial`, as mboost requires, so the Binomial parity check runs instead of skipping. `test_r_gamboostlss_trees_split_factor_levels_as_python_does` is marked `xfail(strict=True)`: the R gamboostLSS btree over a factor column diverges from the Python fit from the third boosting step (max absolute difference 0.53 at `mstop=60`).

- Residual checks that name a revision (issue #73, items 1, 3-5). `cluster_residual_check(model, cluster)` reports the one-way ANOVA intra-cluster correlation of the response residuals with a permutation p-value and suggests `brandom(cluster)`. `time_residual_check(model, time)` reports the Ljung-Box statistic of the residuals averaged by time value, with a permutation p-value and the lags outside the white-noise band, and suggests a time smooth. `influence_check(model)` reports Cook's distance, leave-one-out residuals and studentized deleted residuals from the hat-matrix diagonal, flags rows by a Bonferroni-adjusted Student-t reference, and refits without the top flagged rows to measure the exact change in fitted values. `revision_checks` runs the three. Each result is a `RevisionFinding` (or `InfluenceFinding`) with the statistic, its reference, the assumption challenged, the alternatives, the revision and the held-out check, and each reports the false-alarm rate simulated on data with no structure for the same design. The numerics are in `mboostrs_core::residual_structure`. The agent guide gains the `diagnostics` topic with the pattern-to-revision table. Not included: spatial residual structure and binned-residual uncertainty bands (issue #73, items 2 and the spatial part of item 1).

- Poisson fixed-step risk path on large counts (issue #395, tests only). The
  `gamboost` risk path with `nu=0.03` on counts near 50 agrees with R mboost
  2.9.11 to about 1e-9 relative through iteration 20. From about iteration 26
  the two paths drift apart by floating-point rounding that the fixed step
  amplifies; at iteration 100 they differ by about 2 percent on the test data,
  and R's own path changes by far more under a one-ulp change in its offset.
  No numerical code changed.

- Fixed-horizon survival for CoxPH models (issue #78, survival part). The
  Breslow baseline is computed in Rust (`mboostrs-core::survival`) and stored
  on each fitted model, so `survfit()` and the new `survival_at(times,
  newdata)` work after `save_model`/`load_model` without the training rows.
  `survival_at` returns S(t | x) at user-given horizons, one row per subject.
  `evaluate_survival(model, data, horizons=..., data_role=...)` returns a
  `SurvivalEvaluationResult` with an IPCW Brier score and a cumulative/dynamic
  IPCW AUC per horizon, the Kaplan-Meier censoring survival G(t) on the
  evaluated rows, and a censoring floor (default 0.05) that raises or warns.
  `survfit()` output is unchanged.

- Evaluation by response type, part of issue #78. `evaluate_distribution` reports CRPS for every family with a cdf and a quantile, not only GaussianLSS: the Gaussian form stays closed, count families (Poisson, negative binomial, geometric, binomial, zero-inflated and zero-adjusted Poisson) use the sum over integer thresholds of `(F(k) - 1{y <= k})^2`, and the other families, including the zero-adjusted gamma, use Simpson quadrature of the same integral split at the observation. CRPS is withheld, with the reason in the result's assumptions, for transformed responses and for count families given non-integer outcomes. The kernels are `crps_normal`, `crps_discrete_sum` and `crps_cdf_quadrature` in `mboostrs-core`. `EvaluationResult.auc_interval(level)` gives the weighted AUC with a DeLong standard error and a normal interval, and `EvaluationResult.auc_difference(other, level)` gives the paired AUC difference on the same records with the covariance the shared records induce; both refuse non-record independent units. The DeLong kernel is `auc_delong` in `mboostrs-core`. Not included: survival metrics, uncertainty for precision-recall curves, simultaneous multiclass calibration uncertainty, marginal predictive distributions for unseen groups and R counterparts.

- `compare_to_r` with `family="Binomial"` hands the 0/1 response to R as a two-level factor, which mboost's `Binomial()` requires. The R parity test `test_binomial_r_parity` previously hit R's "response is not a factor" error and skipped instead of checking the fit; it now asserts the fit (issue #402). Measured fitted max abs difference 1.1e-16.

- R parity test `test_r_gamboostlss_trees_split_factor_levels_as_python_does` (issue #401). It failed with a max difference of 0.53 because the test called R's `btree` with its default `selection="ctree"` and Python's `Btree` with its default `selection="sse"`, so it compared two split criteria. Both calls now name `selection="ctree"`; the predictions then agree exactly. The Python default is unchanged.

- `gamlss_refit` and its posterior draws use the rayon pool (issue #388, item 19). The joint cross-products `X'WX` are summed over 1,024-row blocks computed in parallel and added in block order; the per-row terms of the criterion gradient, the smoothing-parameter gradient components, the Hessian of the criterion and the draws (in chunks of 64) are computed in parallel. The draws' normal deviates are still drawn in one sequential stream, so a given seed uses the same deviates at any thread count. Results agree with the sequential computation to rounding (1e-12 relative in the unit test). On a 12,384-row GaussianLSS refit with eight mu terms (`benches/probe_refit_threads.py`), the refit's wall time at 4 threads fell from 136 s to 74-107 s on a machine under load; the single-thread CPU time rose by about 5-10%.

- Benchmark rerun defects, items 7-18 (issue #388). `sensitivity` builds the `<col>_missing` indicator of an `na_action="impute"` model itself, so `data` needs only the original column; a `glmboost` fit did not, because its formula path kept the indicator name. The indicator is `<col>_missing_` when the data already has `<col>_missing`, and this rule is in the `gamboost`, `glmboost` and `impute_missing` docstrings. `sensitivity(parameter=...)` on an LSS model names every column the model reads that the data lacks, as the term the caller wrote (`bols(roof_class)`). `turns` is true only when the curve reverses with a rise and a fall each larger than 5% of its range. `compare_specifications` records a candidate that cannot be scored on its own row (NaN risk, an `error` field naming the candidate) and raises only when every candidate fails; every row carries `specification` and `rhs`. `conformal_interval(method="distributional")` subsets a `%X%` tensor term's design with the training rows, so the refit runs on a tensor model. `ConformalInterval` gains `fitted_mean` (the full model's prediction at the interval rows), `to_frame()` gains `median` and `level`, and the `predicted` column is documented per method. Cross-validation routines (`cv_risk`, `tune_mstop`, `compare_specifications`, conformal) issue one `ExtrapolationWarning` per call, naming the number of fold predictions and learners affected. `validate` reports unused columns as one `extra_column` issue that lists them.

- `mb.discover` handles missing values and skewed covariates (issue #388, items 1-6). The spread stage passes `na_action` to its distributional fits, which previously refused NaN predictors. A NaN counts as missing, as a null does. Interaction terms `a %X% b` take `na_action`: under `"impute"` a numeric margin's missing values are filled with its training median, recorded in the term for prediction; under `"error"` the refusal names the column and the remedy. A refit refuses a numeric column with missing values and names it. The tune stage doubles its mstop grid (100, then 200, then 400 steps) while the risk is still falling, within the budget; if the risk has not turned, the stage is logged with a reason beginning `boundary`, the warning is added to `result.warnings`, and the full adopted term set is kept. An unordered categorical column with at most 10 levels is drafted as `bols`, not `brandom`. A positive predictor with sample skewness above 1 is drafted on `log(x)` (formula-level, replayed on `newdata`, and `contrast()` takes original-scale values); the log is recorded in the draft log entry and in each mean-effect record as `transform`. Sensitivity grids for such a column are geometric. `screen_interactions` gains `na_action` (`"error"` default, `"impute"`, `"omit"`); the error names the column, the first row and the remedy. `gam_refit` accepts `bbs(log(x))` and `bbs(sqrt(x))` terms. `gamboostLSS` cross-validation passes `na_action` to each fold's fit.

- `autospec` response semantics (issue #127, Python). `exposure="column"`
  declares counts per unit of exposure: `log(exposure)` is an offset in every
  probe fit and in `rec.fit()`, and the column is not a predictor. An ordered
  categorical response is inferred as `PropOdds` rather than `Multinomial`.
  `rec.report()` and `rec.evidence["response"]` state the response kind and
  whether the family was declared or inferred. A per-observation offset no
  longer breaks `summary(as_dict=True)`, `to_dict()` or the text summary.

- `autospec` shape comparison under non-convergence (issue #127, Python). A
  candidate whose held-out risk is still falling in the last step band of its
  path no longer takes part in a replacement: the incumbent stands and the
  report says that no shape comparison was made. `ShapeEvidence.comparable`
  records it.

- Cluster-level uncertainty for contrasts and marginal effects (issue #76). `prediction_contrast`, `average_marginal_effect` and `distributional_contrast` take `cluster=` (one label per row), `level=`, `n_draws=` and `seed=`. The weighted average's cluster-robust (sandwich) standard error and a cluster-bootstrap percentile interval are computed in `mboostrs_core::contrast::cluster_average`; clusters are the sampling units and the fitted model is held fixed. The result carries `dependence`, `interval` and `uncertainty="cluster_bootstrap"`. A contrast whose per-row values are constant has a zero-width interval: the band reflects the spread of the row contributions across clusters, not the coefficient uncertainty of the fit.

- Joint-support diagnostic for counterfactual rows (issue #76). The three contrast functions take `joint_support="report" | "warn" | "error" | "off"`, `joint_threshold=` (default 1.0 training standard deviations) and `support_columns=`. A counterfactual row is supported when the training data has a row in the same categorical cell whose numeric predictors lie within the threshold; `mboostrs_core::support::cell_nearest_distance` computes the distances. The share of supported rows is in `support["joint"]`. This catches an unobserved factor level or a nonlinear gap that the linear check of `prediction_profile` does not see.

- Refit-posterior interval for distributional contrasts (issue #76). `distributional_contrast(..., refit=mb.gamlss_refit(lss))` takes one contrast per coefficient draw from `GamlssRefit.draws` (the two settings share each draw's coefficients) and reports the percentile interval as `uncertainty="refit_draws"`. The point estimate stays with the boosted model, and the refit's own centre is reported beside it: the two can disagree, and the record says so. `cluster=` and `refit=` cannot be combined.

- Not in this change: refit-posterior intervals for `prediction_contrast` and `average_marginal_effect` (the GAM refit has no coefficient draws in the core yet), simultaneous bands for profiles and ALE, joint support for `average_elasticity`, `accumulated_local_effect` and `prediction_profile`, and the target of binned derivatives.

- Faster per-step cost of sparse block learners in non-Gaussian and LSS boosting (issue #323). The sparse copy of a `bbs` or factor design now keeps a column-compressed copy too, and `Xᵀ(w∘u)` is one gather-dot per column instead of a row-wise scatter that updated the same accumulators on every row. On a CPS1988 resample with 6 terms and 4 threads, the boosting loop of a Binomial `gamboost` at 128,000 rows took 9.8 ms per iteration before and 5.4 ms after (median of 6 runs each, on a machine under load); GaussianLSS took 14.3 ms and 9.4 ms. Fits are unchanged to 1e-10 relative. The sparse copies add memory: peak RSS rises about 3–5%.

- `autospec(data, response, budget=0)` on a whole-number response with a large mean (counts near 50 to 300) recommended the fixed step nu = 0.3 for Poisson, and `rec.fit(data)` raised "risk became non-finite". No fit runs at budget 0, so nothing had checked that step. The Poisson fixed step is not scale-invariant: its update is nu times a least-squares fit to `y - exp(f)`, which grows with the mean. With no budget, a Poisson recommendation now uses `step="line_search"`, and `rec.evidence["step"]` says why. With a budget, the probes and the cross-validated comparison decide as before. `gamboost`/`glmboost` defaults are unchanged, so R mboost parity holds; R mboost also stops on the same fixed-step path. The non-finite-risk error now names the iteration, the family, the step length and the remedy (`step="line_search"`, a smaller `nu`, or the response's scale).

- Residual-tree and interaction reporting (issue #124). `evidence["residual_tree"]` now records the held-out gain of the tree (`held_out_gain`, the fold mean of incumbent risk minus tree risk, with `held_out_se`), the number of candidates (`n_candidates`), and for an adopted tree `share`: `covariance_share` (the tree's exact part of the prediction's variance) and `absolute_share` (its part of the summed absolute deviations), both on training rows. The screen-stage lead is recorded apart from the adoption decision in `evidence["residual_tree_detection"]`: the share of response variance the additive fit leaves (`leftover_share`), a depth-3 tree's R^2 on that leftover (`tree_r2`) and their product. `rec.report()` states both lines separately and says how much of the fit the adopted tree carries. `evidence["interaction_candidates"]` counts the pairs each interaction screen ranked, and `evidence["interaction_structure"]` records for each two-column term whether it is a tensor surface, which of its columns have their main effects in the formula, and `purified: False` (no purification is implemented). `term_report` reports `candidates`, the number of specifications it compared; `insights` detected pairs carry `candidates`. The shares and paired statistics are computed in Rust (`mboostrs_core::contribution`) and bound as `mboostrs._rust.contribution_shares`, `paired_mean_se` and `leftover_share`. The screen-stage tree R^2 is now computed on the response scale, so for a non-Gaussian fit the offer rule reads a different residual than before.

- `check_declarations(model_or_recommendation, data, *, folds=5, seed=0)` tests
  each declared sign or shape against the data (issue #121). A `bmono` term
  (`monotone=` in `autospec`, or a formula term) is tested by refitting without
  its constraint: `positive`/`negative` on the average 10th-to-90th-percentile
  contrast with its out-of-fold stability; `increasing`/`decreasing`/`convex`/
  `concave` on the share of curve steps in the declared direction and the
  held-out risk change from removing the constraint. Each record has a verdict
  (`agrees`, `contradicts`, `inconclusive`) under a stated rule, the measured
  numbers, and an associational sentence. Structural declarations (treatment,
  groups, measurements, coordinates, adjacency, basis, interaction rules) are
  listed as not tested, with the reason. `SpecRecommendation.with_declaration_checks`
  stores the records and `report()` shows them; `SearchSession` has a
  `check_declarations` action and reports each declaration with its last check
  in `state()["declarations"]`. The numerics are in
  `mboostrs_core::declarations`.

- Selection reports for interchangeable predictors (issue #116, part). `mb.equivalence_sets(model, threshold=0.9)` groups the learners of a `gamboost` or `glmboost` fit whose design spaces overlap at or above the threshold, using the core's pairwise concurvity (`model.concurvity(pairwise=True)`); the overlap is the smaller of the two directional values. `mb.substitution_sets(stability)` reads the per-subsample selections that `stabsel` now keeps (`StabSelResult.fit_selected`) and reports pairs that are each selected often but rarely together, with the group's frequency next to each member's; pass `equivalence=` to require that the pair also overlap. `mb.selection_report(model, deselection=, stability=)` puts both next to a `deselect` result and lists each dropped learner that has a kept equivalent. Every result states its scope as predictive substitution, not a causal claim. Distributional (LSS) fits are refused by `equivalence_sets`, and LSS stability results have no per-subsample selections for `substitution_sets`. Near-tie recording during the fit and calibrated deselection thresholds are not included.

- `scale_absorption(model, *, folds=5, seed=0, flexible=None)` (issue #112,
  Python). For a cyclic `gamboostLSS` fit with a continuous location-scale
  family, it refits the model on one fold split with a flexible mean and with
  a slower scale rate, and reports how much of the fitted log-sigma spread
  each removes, the out-of-fold joint score of each, and the residual
  structure left in the mean. It returns a `ScaleAbsorption` with a verdict
  (`"scale effect robust"`, `"scale effect absorbs mean misspecification"` or
  `"inconclusive"`), the rule that decided it, and `to_dict()`.

- `mboostrs.specification_curve(source, data, response, *, budget=None, seed=0, output="auto")` fits the candidate specifications a search touched on one shared fold partition and returns a `SpecificationCurve`. The source is a `SearchSession`, its `export_transcript()` record, an `autospec` recommendation, or a `mb.discover` result. Each row has the candidate's out-of-fold score, its paired gap to the selected specification with the paired fold standard error, and a declared-or-raised origin for its family, shape, transform and fold scheme; a curve with any raised candidate is labelled post hoc. `joint_test` is a sign-flip test of whether the selected specification beats the curve median, computed by `mboostrs_core::selection::try_curve_sign_test`; the folds are reused, so it is descriptive. `budget` is a fit count, refused from `estimate_cost` before any scoring. `plot()` returns an Altair chart and `to_dict()` plain types. `export_transcript()` gains the keys that rebuild its fold partition and family (`fold_count`, `seed`, `nu`, `na_action`, `unseen_levels`, `weighted`, `family_id`, `opening_family`, `opening_family_id`). Issue #119.

- Search-wide constraints (#74). `SearchSession`, `mb.discover` and `autospec` take `heredity="none" | "weak" | "strong"`, `forbidden_pairs=` and `mandatory_terms=`; `mb.discover` also takes `candidates=`. Heredity applies to a joint term (a `%X%` tensor, a `by=` term or a multi-column `btree`): `"strong"` admits it only when every column it reads has a main-effect term in the same specification, `"weak"` when one does. Forbidden pairs are never nominated or accepted, and mandatory columns keep their main term through every drop rule. For `discover`, a bare value applies to the mean model and a mapping keyed by `"mu"` or `"sigma"` sets each distribution parameter's rule; `mandatory_terms={"sigma": [...]}` includes one log-scale term without the spread test. The default is `"none"`, which is the previous behaviour. The rule in force is recorded in `SearchSession.state()`, `checkpoint()` and `export_transcript()` (`constraints`), in every `autospec` report (`interaction rule:`), and in the `"rule"` entry of `discover`'s log. `discover` now screens interactions when backward elimination leaves an empty mean model, so a pure interaction is found under `"none"`; before, that stage was skipped. `SearchSession` has no per-parameter formulas, so the parameter mappings are `discover`-only.

- `autospec` compares `bspatial(x, y)` with `btree(x, y)` for a coordinate pair (names such as `x`/`y`, `lat`/`lon`, `easting`/`northing`) on the shared folds, against the additive pair. Candidates: the additive pair, the surface, a stump `btree(x, y)`, a depth-3 `btree(x, y, maxdepth=3)`, and the surface with the chosen tree. The tree replaces the surface by the 2%-or-two-SE rule, the combination by two paired SE, and the chosen surface replaces the additive pair by the 2%-or-two-SE rule. Recorded in `evidence["coordinate_surface"]` and in `rec.report()`; `joint_tree=False` turns it off. Declared `coordinates=` terms are not replaced. `insights()` emits a measured finding when a coordinate tree beats a `bspatial` surface on the pair by more than the paired fold standard error. `suggest_terms` and `classify_columns` name the tree alternative in the reason for each coordinate column; their terms are unchanged. Issue #338.

- `_coordinate_pair_names` returns the pair in column order, so the term reads `bspatial(x, y)` rather than `bspatial(y, x)`. Issue #338.

- Issue #319 did not reproduce: the Meuse 124-row split reproducer with `budget=200` completed on all 20 splits, with and without the coordinate pair.

- Comparison identity for ranking helpers (issue #60, Python). Every fit
  records a digest of its observed response, its kept rows against the input
  rows, its effective weights and weight policy, its response transform, its
  likelihood measure and its scoring rule; held-out evaluations record their
  evaluation weights and response transform. `gaic_table`, `compare_models`,
  `scan_lss_families` and `compare_specifications` run one check on these
  records. Candidates fitted to different rows, a permuted response, or
  different effective weights raise `ValueError` naming the difference. Rows
  that share their data but differ in response transform or weight policy are
  returned unsorted with a `comparison` reason column, and `sort=` on
  `compare_models` raises for them. Legitimate same-sample comparisons return
  the same output as before. Models fitted before this change compare on the
  retained data as before. The R package is not changed.

- Comparison identity extended to summaries, cross-validation curves and common samples (issue #60, Python). `compare_summaries` runs the same check as the ranking helpers: models on different rows, responses or weights raise `ValueError`, and models that differ only in response scale or weight policy are shown with a "Not ranked against each other" note. Its `cv=` results must share one fold matrix. `CvRiskResult.compare_to` and `compare_cv` refuse two curves whose response, rows or weights differ; two curves on different response scales (`log(y)` against `y`) return a result with `comparison` set, a NaN mean difference and no interval. `gaic_table` and `compare_models` take `common_sample=True`: every candidate is refitted on the rows all of them keep, and each row reports `common sample rows`, `common sample excluded rows` and `common sample excluded by` (which candidates dropped the rows); a warning names the removed rows. Default `common_sample=False` is unchanged. `compare_models(cv=...)` cannot be combined with `common_sample=True` when rows are excluded, because the cross-validation results were computed on the full sample. `scan_lss_families` does not take `common_sample=`: its candidates share one data set and no missing-value option. The R package is not changed.

- `sensitivity(model, data, parameter=...)` on a distributional model holds the columns of the other parameters' formulas at their means (or most frequent levels) on the grid, as it holds the measured parameter's other columns. Before, a formula that read a column the measured parameter did not (a factor, a `log(...)` transform, a second column) made the prediction fail with "prediction data is missing column". A transform term such as `bbs(log(b))` is also reported under its source column `b`, for a gamboost model as well. `mboostrs.discover` reads its shape, range and slope from the same grid instead of computing its own curve for distributional models. Issue #375.

- `GAMBoostModel.predict_over_grid(variable, grid, newdata=, ...)` predicts at each value of one variable, with every other column held at each row's own value. It is the sibling of `predict_over_mstop` (#296). A term that does not read the variable is evaluated once; a term that reads it is evaluated once per grid value, instead of replaying every term for every value. Its numerics are `mboostrs_core::scoring::predict_over_grid`. `extrapolation=` and `unseen_levels=` behave as in `predict`, and `offset=` takes the new rows' offset. Multinomial models, models with `Buser` terms or interactions that read the variable, and factor-valued grids fall back to one `predict` per value. `prediction_profile` uses it for gamboost models. On a 20,000-row model with six smooth terms and a 25-point grid, its median wall time fell from 277.6 ms to 32.6 ms, and the average and ICE values agree with the per-value result to 1e-15. LSS models keep the per-value path, since there is no LSS `predict_over_mstop` to model the method on.

- `mboostrs.discover(data, response)` runs one rule-driven analysis and returns a `DiscoveryResult`. It drafts one term per column, drops each mean term whose removal does not raise out-of-fold risk by more than two `change se`, adds up to five screened interactions that lower it by two `change se`, tests a log-scale driver for a Gaussian response, chooses the stopping iteration by cross-validation, refits by REML, and predicts `newdata` with intervals. The result carries `mean_effects` with shape labels, `interactions`, `spread_drivers`, `predictions`, `contrast(...)` intervals, the `warnings` raised, a `log` of every decision with its evidence, and a `to_dict()` with provenance. `budget_seconds` skips a stage whose `estimate_cost` would exceed it, and the log says so. A permuted response returns no effects, interactions or drivers. Issues #357 and #292.

- `split_refit(formula, data, *, family=None, select="deselect", repeats=1,
  fraction=0.5, level=0.9, seed=0, contrasts=None)` chooses terms on one random
  half of the rows (`deselect`, `stabsel`, or a callable) and refits the chosen
  terms by REML on the other half (`gam_refit` for a `gamboost` formula,
  `gamlss_refit` for a mapping of distribution parameters), so the refit's
  intervals and tests do not reuse the rows that chose the terms. With several
  repeats it reports selection frequencies, medians of the per-split bounds, and
  p-values aggregated by the Meinshausen, Meier and Bühlmann (2009) rule.
  `contrasts=[(variable, reference, alternative)]` reports the average
  predictor change over the inference rows with its interval. It returns a
  `SplitRefitResult` with `to_dict()` and `summary()`. Selective inference is
  not implemented. Issue #117 (items 1 and 2).

- `GamRefit`-based average contrasts: `mboostrs.selection._average_contrast`
  evaluates the average over rows of a term's effect difference through the
  core's effect call on one averaged raw row, so the interval is the exact
  standard error of the average. Used by `split_refit`; not exported.

- `gamlss_refit` no longer spends minutes on a single criterion evaluation
  when the inner Newton solve for the coefficients stalls near the optimum:
  the solve now stops after three consecutive steps that gain nothing beyond
  rounding, where it used to run to its 200-iteration cap and every
  line-search trial repeated that. On one replicate of the claims-discovery
  benchmark (a GaussianLSS fit with a spatial surface and a varying
  coefficient on 15,480 rows) the refit took 536 s; it now takes about
  100 s, and the other replicates' results are unchanged. The convergence
  report (`refit.convergence`) gains `stop_reason`, `evaluations`,
  `inner_iterations`, `restarted`, `restart_iterations` and `seconds`, and
  `MBOOSTRS_TRACE_REML=1` prints one line per outer iteration to stderr.

- `gamlss_refit` converges on small samples with a null smooth in sigma (#374).
  With `gamboostLSS` GaussianLSS fits of `mu ~ bbs(x)` and `sigma ~ bbs(z) +
  bbs(w)`, where `w` is unrelated to the response, the refit raised "could not
  evaluate its criterion at the start" or "the refit optimum cannot be
  re-evaluated" on 7 of 24 synthetic data sets (n = 80 and 150). Its first pass
  is unchanged. When that pass fails, the refit repeats the search from the
  first starting smoothing parameter (0, 2, 4, ... up to `rho_bound`) whose
  penalized likelihood has a finite maximum, and it counts an inner solve that
  reaches its iteration cap as infeasible. All 24 now converge. If no starting
  smoothing parameter is feasible, the refit raises an error that names each
  parameter's smooth terms and says to remove the term or lower its df.

- `mboostrs.validate(model, newdata, offset=None, output="auto")` (#293) lists the issues `predict(newdata)` would meet, without predicting. It covers gamboost, glmboost and gamboostLSS. Each record has a `kind` (`missing_column`, `unseen_level`, `na`, `extra_column`, `row_count`, `offset`, `extrapolation`, `not_range_checked`, `check_skipped`), a `severity` (`"error"` where predict refuses the rows, `"warning"` where it runs), the `parameter`, `term` and `column` concerned, a `count`, `n_rows`, `worst_distance` for extrapolation, the unseen `levels`, and a `suggested_policy`. `output="records"` returns a list of plain dicts.

- The `extrapolation=` policy now covers prediction contributions, Multinomial predictions, both coordinates of `bspatial`, and the B-spline margins of `%X%` terms (#293). `GAMBoostModel.prediction_contributions()` takes `extrapolation=` and defaults to `"warn"`, so it now warns for rows outside the training range; pass `extrapolation="allow"` to keep the old silent behaviour. `gamboostLSS.partial_effects()` takes `extrapolation=` and defaults to `"allow"`, as the GAM method does. `btree`, `bkernel`, `buser` and `brad` terms have no training-range boundary and stay unchecked; `validate()` lists them as `not_range_checked`.

- One boundary rule for `tune_mstop` and `compare_specifications(mstop="cv")` (#314). A search extends its grid while the optimum lies in the boundary band at the end of the grid, which is now 5% of the grid's length (`mboostrs.cvrisk.BOUNDARY_SHARE`) for every search. It used to extend only inside a 2% band, and warn in a 5% band, so an optimum at 97% of the grid warned without extending. A search warns only when it reaches its limit with the optimum still in the band, and the warning names the limit that applied (`max_mstop=` or `extensions=`).

- `compare_specifications(..., mstop="cv", max_mstop=...)` and `cv_risk(..., mstop="cv", max_mstop=...)` set the cap of the inner tuning. The default stays 500. The inner search starts at 500 steps, or at the cap when it is smaller, and grows up to the cap. Passing `max_mstop` with a fixed `mstop` is an error.

- `compare_specifications(mstop="cv")` rows report the inner step counts across outer folds as `inner_mstop_min`, `inner_mstop_median` and `inner_mstop_max`, and `inner_at_boundary` is true when a fold's inner search stopped at the cap with the risk still falling. The warning names `compare_specifications` (or `cv_risk`) and `max_mstop`, and no longer advises re-running `cvrisk`. Fold records of `cv_risk` carry `inner_mstop` and `inner_at_boundary`.

- `fit_budget=` on `tune_mstop` and `compare_specifications` counts the worst case of the search. Under the default family score, growing a grid resumes the folds and adds no fits, so the count is unchanged. Under any other score, with `fold_local=True`, or for a distributional specification, every grid refits the folds, and the count covers every grid up to `max_mstop`. A call whose count was previously within budget can now be refused.

- Intercept-only models. `gamboost("y ~ 1", ...)` and `glmboost("y ~ 1", ...)`
  fit no base learner: every prediction is the family's offset, the path has
  no selections, and `mstop` reports 0 whatever `control` asked for. Before,
  `gamboost` raised "intercept control ('+ 1') is not supported yet" and
  `glmboost` raised "has no predictors". Predict (with and without `newdata`,
  link and response), `fitted`, `risk`, `summary`, `varimp`, `coef`,
  `selected`, `cvrisk`, save and load, `prediction_contrast` (0),
  `conformal_interval` and `compare_specifications(["1", ...])` accept the
  model. `gam_refit` refuses it with an error that says it is intercept only.
  The engine accepts an empty learner list (`fit`, `cv`, `cvrisk`) as the
  offset-only path. Issue #357.

- Case study `case_studies/kdd_scorecard_benchmark.ipynb` is listed as the 81st study: the overview says "Eighty-one", the case-standard baseline records 81 studies with no open gaps, and the notebook's study card states its sampling unit. The notebook is stored without outputs, as the other book notebooks are.

- `terms.screen_interactions` documents that a pure-noise pair's score is negative in expectation, not zero-centred: a single score can exceed zero by chance. The test `test_screen_scores_pure_noise_below_zero_on_average` now asserts the mean over noise replicates is below zero and each replicate's maximum is below 0.03, in place of a strict zero bound on each maximum. The screen's computation is unchanged.

- `ExtrapolationWarning` is a direct subclass of `Warning`, not of
  `UserWarning`. A blanket `warnings.filterwarnings("ignore",
  category=UserWarning)` used to hide it; it is now filtered only by its own
  class. Code that caught it with `pytest.warns(UserWarning)` or
  `category=UserWarning` must name `ExtrapolationWarning` instead.

- `predict(extrapolation="clip")` clamps the `by=` variable of a spline
  term to its training range, as well as the covariate. An out-of-range
  `by=` value used to scale the term's effect without limit. The default
  `extrapolation="warn"` also reports a `by=` variable that is out of
  range, naming it as the covariate. This applies to `gamboost`,
  `gamboostLSS` and `which=` contributions.

- `extrapolation_report(newdata)` on `GAMBoostModel`, `GLMBoostModel` and
  `LSSModel` returns one row per term and covariate with rows outside
  their training range: `term`, `covariate`, `kind` (`"spline basis"` or
  `"linear"`), `parameter`, the rows below and above, the training range,
  the range of `newdata`, and `worst_distance` in training ranges. It is
  a Polars frame by default, with the `output=` options of the other
  reports. It lists the same excursions as `range_report()` and the warning.

- Provenance records (#357). `mboostrs.provenance(data=None, *, seed=None)`
  returns a dict with the mboostrs version, the `build_info()` identity, the
  Python version, the platform, and the NumPy, Polars and (when installed)
  pandas versions. With `data`, it adds the row count, the column names with
  dtypes, and a SHA-256 of a canonical byte layout of the columns, equal for a
  Polars frame, a pandas frame and a dict holding the same values and dtypes.
  `to_dict()` of `gamboost`, `glmboost`, `gamboostLSS` and `glmboostLSS`
  models, of `gam_refit` and `gamlss_refit` results, and of
  `ConformalInterval` now carries a `"provenance"` entry. The model records
  are the `summary(as_dict=True)` content with that entry added; the
  refit and model entries fingerprint the training frame, and the interval
  entry has no data fingerprint because it holds no training rows.

- Cost before a call runs (#357). `mboostrs.estimate_cost(call, *args,
  **kwargs)` counts the model fits of `gamboost`, `gamboostLSS`,
  `tune_mstop`, `compare_specifications`, `term_report`, `stabsel` or
  `conformal_interval` and times one ordinary fit, returning `fits`,
  `seconds_per_fit`, `estimated_seconds` and an explanation. `tune_mstop`,
  `term_report`, `stabsel`, `stabsel_glmboostLSS` and `stabsel_gamboostLSS`
  take `fit_budget=`, which refuses before any fit when the call would need
  more fits, as `compare_specifications` already did. `term_report` now counts
  its removal refits as well as its shape refits, which its docstring formula
  left out. Its results are unchanged.

- `risk` works as an attribute and as a call on `gamboost`, `glmboost` and
  `gamboostLSS` models (#357). `fit.risk` is still the read-only risk path,
  and `fit.risk()` returns the same values as a fresh writable ndarray. Before,
  the call raised numpy's `'numpy.ndarray' object is not callable`. The
  `gamboostLSS` call used to raise a `FutureWarning` that the call form was
  deprecated; that warning and its registry entry are removed, and
  `fit.risk(parameter=...)` still raises `NotImplementedError` pointing to
  `parameter_risk()`.

- `sensitivity(model, data, parameter=...)` measures a distributional model
  (`gamboostLSS` or `glmboostLSS`) one distribution parameter at a time. It
  grids the columns of that parameter's terms, holds the others at their
  means, and reports the parameter's linear predictor on its link scale. The
  default is the family's first parameter, and each row carries a
  `parameter` key naming it. `parameter=` is refused for a `gamboost` or
  `glmboost` model. The refusal for an object with no learner specs or
  formula now points at `sensitivity(..., parameter=)`, `prediction_contrast`
  and `distributional_contrast` (#335). Output for gamboost and glmboost
  models is unchanged.

- `FAMILY_SETS` and `family_set(support)` are exported at the top level.
  `FAMILY_SETS` maps each response support (`"realline"`, `"realplus"`,
  `"real0to1"`, `"counts"`, `"zeroplus"`) to its candidate LSS families, and
  `mb.family_set("realplus")` returns the names for one support. An unknown
  support raises a `ValueError` that lists the valid ones. Before, the table
  was importable only from `mboostrs.model_selection`; two agents in a
  simulation guessed `mb.FAMILY_SETS` and failed (#337).

- Agent guide topics and an installable skill. `mboostrs.agent_guide("index")`
  lists five topic sections — `"missing-data"`, `"uncertainty"`,
  `"categoricals"`, `"costs"`, `"unattended-pipelines"` — and
  `agent_guide("<topic>")` returns one, each with an example the test suite
  runs. `mboostrs.install_skill()` and `python -m mboostrs skill install`
  copy the skill and its topics into `.claude/skills/`, where coding agents
  that read skills load it. `python -m mboostrs guide [topic]` prints them.
  The guide's statement that no intervals or p-values exist was wrong: it
  now points at the REML refits, `conformal_interval` and LSS
  `prediction_interval`. Found by an agent benchmark in which a Haiku
  analyst learned mboostrs from the package alone and wrote its own
  imputation, unaware of `na_action="impute"`.

- `gamboostLSS` and `glmboostLSS` take `na_action` (`"error"`, `"omit"`,
  `"impute"`), as `gamboost` and `glmboost` do. `"impute"` adds the median
  fill and missing-value indicator term in every parameter formula that
  reads the column and replays them on new rows; `"omit"` drops the rows
  (and their weights) across all parameter formulas.

- `gamlss_refit(...).predict`, `.effects` and `.draws` read string factor
  labels. They used to compare the labels with the integer codes of the
  encoded training frame and refuse every row ("level(s) ['C', 'D'] not seen
  at fit time"), suggesting `unseen_levels='zero_effect'`, which would have
  zeroed the effect.

- `confint(method="refit")` works on a `gamboost` model with a string
  `brandom` or `bols` factor; it passed the encoded training frame to the
  refit and failed the same way. `gam_refit` and `confint(method="refit")`
  also work on a model fitted with `na_action="impute"` (the indicator terms
  are refitted, and new rows are filled before the refit reads them).

- `LSSModel.prediction_interval(newdata=...)` applied the factor recipe twice
  and failed on any frame with a string factor.

- `conformal_interval` works with string factors in every method (`"split"`,
  `"cv+"`, `"cqr"`, held-out `calibration=`, `"distributional"`): new rows
  are brought to the training frame's coding once. The distributional method
  no longer refuses a model whose only recipe steps are factor encodings or
  missing-value indicators; it still refuses value-fitted steps (scale,
  centre, median fill, log floor) and now names them.

- An empty right-hand side (`"y ~ "`, what a pipeline builds when its
  selection drops every term) now says to write `"y ~ 1"` for an
  intercept-only model. An agent's pipeline crashed on exactly this on a
  replicate with no signal.

- A missing predictor value's error names the fix: "pass na_action='impute'
  ... or na_action='omit'". An unseen-level error whose fitted levels are
  numbers and incoming ones labels (or the reverse) says the frame is
  probably coded differently, instead of suggesting `unseen_levels`.

- `predict` no longer silently averages every row when a factor column's
  levels differ from the fitted ones only in dtype (#334). Fitting
  `y ~ brandom(f) + bbs(x)` where `f` is a string column `'1', '2', '3'`
  and predicting on newdata where `f` is Int64 1, 2, 3 raised under the
  default `unseen_levels="error"` but, under `"average"` (or
  `"zero_effect"`), predicted every row at the average group without a
  word — the levels are unseen yet none is a new group. Every unseen-level
  check (the factor and dummy encodings of `transforms.apply_recipe`, and
  the interaction marginals) now detects the mismatch — every non-null
  incoming level unseen, every one equal to a fitted level after `str()`
  or a numeric cast — and refuses it under every policy with
  `MboostrsValueError(code="unseen_level")` naming the cast that fixes it,
  e.g. `pl.col('f').cast(pl.Utf8)`. Genuinely new levels keep the policy,
  and `gamboostLSS` predict is covered too.

- **Breaking:** `predict_parameters()` now defaults to the response scale
  (#326). It used to forward an omitted `type` straight to `predict()`, whose
  default is the link scale (R mboost parity), so an `NBinomialLSS` fit
  returned `sigma` as `log(size)` (≈1.03 where the size is ≈2.8) — and the
  call also fired #229's link-scale warning naming `predict()`, a method the
  caller never called. Both implementations (the LSS classes and the standard
  models' `{"f": ...}`) now read an omitted `type` as `type="response"` and
  emit no warning: each parameter comes back on the scale it is named for —
  an NB `sigma` is the size, a Gaussian `sigma` a standard deviation, a
  logit parameter a probability. Pass `type="link"` to keep the old linear
  predictors; any explicit `type` still passes through unchanged, and
  `predict()` itself still defaults to the link scale with its warning. Two
  warning-text fixes came with it: every `maybe_warn_link_scale` call site
  now passes the name of the method that actually ran, and the
  `link_scale_label` entries for the negative-binomial families no longer
  call a size a dispersion — `NBinomialLSS`'s `sigma` is the size (theta),
  now labelled "log-size (theta; larger = less overdispersion)", while the
  `NegativeBinomialLSS`/`ZINBLSS`/`PIGLSS` `sigma` really is a dispersion
  (gamlss NBI parameterization, variance `mu + sigma*mu^2`) and now says
  which way it points ("larger = more overdispersion"). Pre-1.0, so the
  default changes with no deprecation shim.

- `main` collects and runs again: the #344–#346 merge batch resolved
  conflicts against older sides and reverted three merged fixes. #320's
  transformed-term predicate (`_reads_transformed`, every learner, not only
  `bbs(log(...))`/`bbs(sqrt(...))`) was removed, leaving its regression test
  with an `ImportError` at collection; the three call sites are restored with
  #322's additions intact. #275's zero-inflation scan budget ladder
  (`_ZERO_SCAN_BUDGETS`, escalate while the leading family or the PoissonLSS
  baseline still falls at the cap, adopt only on interior minima) was
  replaced by the old fixed 300-step scan; the ladder is restored with #319's
  native-panic recovery (`except BaseException`, declined to the
  dtype-inferred Poisson) kept. #345 updated the docs copy of the analysis
  skill without syncing the shipped `python/mboostrs/agent_docs/SKILL.md`
  the   byte-identity test compares; the docs file is copied over.

- `sensitivity` reports a `bols(<string column>)` term under the source
  column the caller passed (`gender`), not the derived design column the fit
  builds internally (`__by0`), which the frame never contains and the user
  cannot supply. The same mapping applies to tensor interactions
  (`bbs(x) %X% bbs(z)` fits on `__ix0`): the schema names `x` and `z`. A
  two-level string column is gridded over its levels and held at its modal
  level for the other terms, like any unordered categorical (#309).

- Small non-negative integer count columns keep a measurement term instead of
  a categorical `brandom` (#313). `kid5` (children under five, 0..3), `k618`
  or `ment` (0..77) were typed as one shrunken effect per value, so
  `predict()` raised on any count the fit had not seen — a test row with
  `kid5 = 4` could not be scored. `classify_columns`, `suggest_terms` and
  `autospec` now read such a column as a count when its values are
  non-negative whole numbers from 0 or 1 with at least three of them,
  covering at least half of the range they span or named like a count
  (`n_`, `num`, `count`, `kid`, `child`, `ment`, `visits`, `age`): up to four
  distinct values get `bols(name)`, more get `bbs(name, df=min(df, levels-1))`
  — one effect that extends to unseen counts, not one level per value. The
  alternatives name `brandom(name, df=...)` for when the codes are labels.
  String, enum and declared-categorical columns are never counts, label
  codes that are neither 0/1-based nor contiguous (region 11, 27, 53, 98)
  keep `brandom`, and the reading wins over the `label_evidence` rule that
  could previously retype such a column.

- `pair_term_for` accepts `data=` and no longer proposes spline terms for
  string columns (#336). Without `levels=`, `pair_term_for("x", "b")` built
  `bbs(x, df=2.5, knots=5) %X% bbs(b, ...)` even when `b` was a string
  column, and fitting it failed with `could not convert string to float`.
  Pass the frame as `data=` and each margin not already described by
  `levels` is read from it: a two-level string column gives `bbs(x, by=b)`
  (a smooth that switches on with it), a many-level one is treated as a
  group and gives a random slope, `brandom(b, by=x)`. With `data=None` the
  behaviour on explicitly described columns is unchanged.

- `autospec` no longer aborts when a candidate family's fit hits a Rust panic
  (#319). On the Meuse log-zinc benchmark, 6 of 20 train/holdout splits
  stopped with a pyo3 `PanicException` (`digamma requires x > 0, got 0`).
  The exception escaped the GammaLSS family probe because it derives from
  `BaseException`. Two fixes:
  - Every native fit call (`call_fit`) now re-raises a panic as
    `MboostrsRuntimeError` with `code="native_panic"` and chains the
    original exception, so probes that catch `RuntimeError` record the
    family as failed and move on.
  - At the source, `digamma` returns `NaN` for a non-positive or `NaN`
    argument instead of asserting. When `exp(eta_sigma)` underflows in a
    diverging GammaLSS fit, the risk becomes non-finite, and the engine
    reports that as its usual "LSS risk became non-finite" solver error.

- `main` builds again (#341). The hinge learner commit (#318) carried a stale
  copy of `crates/mboostrs-core/src/learners.rs` and reverted the btree
  histogram split finding (#287) and the bspatial Gram-reuse/streamed work
  (#284, #288, #299), leaving `mboostrs-core` uncompilable. `learners.rs` is
  restored and only the hinge additions (and its Rust tests, now in
  `mod bhinge_tests`) are re-applied. `Hinge` is now exported from
  `mboostrs` like the other learner classes. `Finding`, `agent_guide` and
  `insights` are back in the capability manifest's export list, and the
  README counts thirteen base learners.

- `autospec` no longer raises `cannot parse formula term: 'bols(log(x))'`
  when its shape probe prefers a straight line on a log- or sqrt-transformed
  predictor (#320). The check that routes transformed terms around the strict
  formula parser recognised only `bbs(log(...))`/`bbs(sqrt(...))`; one
  predicate now covers every learner, at all three sites.

- `gamboostLSS`: a one-covariate `btree(x)` now reports its covariate as a
  vector in `partial_effects`, as `gamboost` does, instead of an `(n, 1)`
  column. The column crashed `llm_explain_prompt` (IndexError in `_condense`)
  and `plot_lss` (#324).

- `compare_specifications` reads a whole formula whose left-hand side is the
  `response` (`"y ~ bbs(x) + bols(z)"`) as its right-hand side, instead of
  failing with `cannot parse formula term`; a formula for a different response
  raises an error that names the right-hand side to pass (#327).

- `compare_specifications` no longer crashes with `unhashable type: 'dict'` on a
  plain list of distributional (`{"mu": ..., "sigma": ...}`) candidates; each
  row is named `"mu: ...; sigma: ..."` and keeps its mapping as `rhs` (#329).

- `scan_lss_families` refuses a bare right-hand side (`{"mu": "bbs(x)", ...}`)
  once, before fitting, instead of returning one identical error row per
  family; reads an integer `folds` as a seeded k-fold split, as
  `compare_specifications` does; and warns when every candidate family fails,
  since the first row is then an error, not a winner. The docstring now says a
  mapping holds whole formulas (#330).

- Agent entry points are reachable from the installed package (#306):
  `mboostrs.insights` and `mboostrs.Finding` are top-level (the function, not
  the submodule of the same name); `mboostrs.agent_guide()` returns the
  "analyzing data with mboostrs" skill, now shipped as package data
  (`mboostrs/agent_docs/SKILL.md`, kept byte-identical to
  `docs/skills/...` by a test); the package docstring and README point to
  both. The skill gains a runnable quick reference (one call per spine verb
  with the return fields agents guessed wrong: `ColumnKind.name`,
  `SpecRecommendation.formula`), executed by `tests/test_skill_quickref.py`,
  a Step 0 on reading `insights()` findings, and a caveat that `btree` has no
  `df` (#291).

- Without the `plot` extra, `hasattr(mboostrs, "plot")` is `False` and
  `inspect.getmembers(mboostrs)` works: the lazy `__getattr__` now raises an
  `AttributeError` carrying the install hint instead of the `ImportError`
  (#308). `from mboostrs import plot` and `import mboostrs.plot` still raise
  the `ImportError` with the hint.

- `insights()` accepts Polars frames (it read the response with pandas-only
  `.values`, #305) and honours `family=` in its own fit: `tune_mstop`,
  `gamboost`, the residuals, `screen_interactions` and `sensitivity` now use
  the family instead of always fitting least squares (#307). `family=None`
  infers one from the response (`{0, 1}` -> Binomial, non-negative integer
  counts -> Poisson, otherwise Gaussian) and every finding records the family
  in `evidence["family"]`. The "residual spread moves with the fitted value"
  finding is asked only under a Gaussian fit; Poisson fits get a Pearson
  overdispersion check (`basis="detected"`) instead, and Binomial fits get
  neither, so a binary or count response no longer top-ranks a spurious
  "fit gamboostLSS with a scale predictor" suggestion. Two-level columns enter
  as `bols()`, the internal folds' `ExtrapolationWarning`s are silenced, and a
  `sensitivity()` failure on `bols(<string column>)` (#309) leaves findings
  without a direction instead of failing the call.

- `compare_models()` takes `sort=` (best first; largest first for `logLik`,
  smallest first otherwise) and reports `selected_learners`, the number of
  distinct base learners each path updated. It withholds `risk` (`None`, with
  a `risk_unavailable` reason on each row) when the models do not share one
  loss, such as Gaussian and Binomial fits on the same response or two quantile
  levels, and refuses to sort by a withheld risk. `cv=` takes one
  `CvRiskResult` per model, adds `cv_mstop` and `cv_risk`, and requires all of
  them to carry the same fold fingerprint. `SummaryTable.rows` keep numeric
  values, so `Summary.as_dict()` returns numbers rather than formatted strings.

- `brad(...)` formulas now work in Python. Radial learners retain their training
  centres, Matérn range and whitening matrix, so models in Python and R predict
  directly at new coordinate rows. The basis construction and replay share the
  Rust core; saved models preserve the recipe and weighted centering.

- Slicing a `gamboost` path (`model[:k]`, `with_mstop(k)`) rebuilds the training
  fit with one design product per learner instead of one per step; tree steps
  are still replayed one by one. A 5,000-step `bspatial` + six-`bbs` path on
  12,524 rows slices in 0.03 s instead of 11.5 s, to 5e-15 of a cold fit.

- Fitted `NBinomial` models expose their calibrated `size` (Python property,
  R `size(fit)`, Rust family accessor). `nuisance()` keeps returning that size;
  statsmodels NB2 uses `alpha = 1 / size`. Unsupported fixed-size, `alpha` and
  `dispersion` constructor arguments now explain this parameterization.

- Fixed-step profiled likelihood paths now report material increases in their
  training loss, including a final excess greater than 0.1% of the absolute path
  minimum for Gamma, negative-binomial, hurdle, ordinal and parametric survival
  likelihoods. Adaptive Huber remains omitted. The default fixed step retains
  mboost's convention; a
  calibrated shape can make it overshoot. Line search uses a stable Gamma
  loss difference near convergence. R exposes `step="line_search"` and
  carries the rule through standard/distributional fitting, cross-validation,
  stability selection and bootstrap controls.

- Standard-model `cvrisk(preprocessing="refit")` in Python and R rebuilds
  preprocessing on each training fold. The default `"fixed"` mode retains
  mboost's fitted-design convention. Results record the mode; family scoring
  restores each fold's training-fitted nuisance path in Rust. Structured folds
  retain explicit held-out masks and complete fit/evaluation weights.

- GLM grid predictions without an intercept use the same learner numbering
  as ordinary predictions. R GLM prediction replays `scale()` and other
  supported formula transformations with their fitted parameters.

- `GamRefit.predict(unseen_levels="zero_effect")` now applies the policy to
  boosted terms retained with `unsupported="hold"`, including categorical
  trees. New groups no longer fail in that held portion after the refitted
  terms have accepted the policy.

- `modeler_loop(..., family=...)` accepts an explicit response family for
  discovery, comparison, final tuning and budget fallback. New book examples
  demonstrate budgeted interaction search, varying coefficients, measured
  domain judgments and an optional live LLM connection.

- Research report `.remember/FEYNMAN_ENHANCEMENTS.md` (autoresearch loop,
  2026-09-27): 19 literature-verified enhancement avenues in four themes —
  disciplined specification search (specification-curve reporting, model-X
  knockoff selection with FDR control, heredity-enforced interactions,
  concurvity diagnostics, falsifiable declarations), plausible causal
  structure (CAM/RESIT ordering with GCM pruning, nonlinear ICP confidence
  sets, Riesz regression, additive score matching), engine competitiveness
  (momentum-accelerated component-wise boosting, Hessian-bound step guards
  and penalization-drift warnings from the AISTATS 2025 BAM theory, balanced
  and natural-gradient step lengths) and distributional breadth
  (non-crossing quantile processes, copula components, extreme-value tails
  with covariates, robust LSS, additive index models, censored quantiles).
  Every avenue was checked against the codebase and the existing plans; a
  novelty accounting table separates new avenues from parked ideas (E9, E10,
  natural gradients, copula LSS) it extends.

- `same_family_gap(model, cv)` reports when the cross-validated step of a linear boost is still short of the same-family GLM. It gives the chosen `mstop`, the boost `logLik`, the comparison log-likelihood and the gap, and it does not move `mstop` or keep boosting. Gaussian, binomial, Poisson, gamma, negative binomial, proportional odds and a linear beta regression are fit in-process.

- `autospec(preexploration="native"|"hessboost"|"off")` reuses one preliminary
  tree ensemble for variable priority, Gaussian shape proposals and interaction
  screening. Permutation importance, partial dependence, imputation and screening
  statistics now run in one Rust API exposed by both bindings. Shape pruning
  requires marginal/partial agreement, retains incumbents and declared terms,
  and leaves ambiguous evidence open; held-out adoption rules are unchanged.
  The native screen is charged once, bounded to 32 partial-dependence columns,
  and recorded in the recommendation. The optional `hessboost` Cargo feature
  enables the pinned 0.1.1 engine; explicit requests fail clearly when absent.
  Pair adoption now reserves its held-out validation cost before extending the
  tuning path; unaffordable or unvalidated pairs remain suggestions.

- `column_screen(data, response)`: scatterplot diagnostics of the response
  against each column, computed in the Rust core (`screen::column_diagnostics`):
  the shares of variance a line, a curve (20 bin means, adjusted for their
  degrees of freedom) and one step explain, Pearson and Spearman
  correlations, skewness, distinct values, and a hint (`none`, `line`,
  `curve`, `step`, `levels`). `autospec` records it and states it in one
  report line.

- `autospec`'s interaction screen adds pairs from a boosted-tree surrogate:
  depth-4 trees over every column fitted on the same rows, columns ranked by
  permutation importance, and Friedman's H statistic with the interaction
  variance for each pair among the top eight (`screen::h_statistic`). A
  surrogate pair is proposed when at least 5% of its joint partial dependence
  is not additive and its interaction variance clears the screen's floor;
  like every lead, it is adopted only when held-out risk says so. It finds a
  pair with no main effects (`sin(3 x1 x2)`) in about 1.5 s.

- `ci_test(data, x, y, given=...)`: a test of conditional independence, the
  generalised covariance measure (Shah and Peters 2020). `x` and `y` are
  regressed on `given` with boosted additive models predicted out of fold,
  and the residual products are tested for mean zero. A numeric `x` is
  expanded into `x` and hinge functions at its quartiles and tested jointly
  (chi-squared), so a non-monotone dependence such as `y` on `x^2` is found.
  It separates a proxy from its driver (a proxy shows no association once its
  driver is conditioned on) and holds its level under the null (40 null data
  sets). The statistics are in the Rust core (`causal::gcm_test`,
  `gcm_test_multi`).

- `autospec(residual_tree=True)`: after the interaction step, try a depth-5
  regression tree over every numeric and categorical column as one more
  boosted term, and keep it only when it lowers the cross-validated risk by
  2% or two paired fold standard errors (the rule adopted pairs pass). It
  carries structure the main effects and pairs miss: interactions of three or
  more columns, and pairs the screen did not propose. The report states the
  gain and that the term is a prediction term, not an effect curve. On the
  training rows of seven real data sets, inner-validation R^2 rose from
  0.57 to 0.75 (kin8nm), 0.34 to 0.48 (white wine), 0.85 to 0.92 (airfoil),
  0.90 to 0.94 (grid stability) and 0.85 to 0.90 (Miami housing).

- `btree` takes categorical columns: a factor is coded by its training levels
  and a split divides the levels into two sets (the levels ordered by the
  mean residual, which finds the best two-set split for squared error). A
  level the fit never saw follows the child that held more training weight,
  under `predict(..., unseen_levels="zero_effect")`; without that argument it
  is refused as before. It had refused every factor with more than two
  levels.

- R `btree()` takes several columns, including factors (coded by their
  training levels and split into two level sets), and R `predict()` evaluates
  tree terms on new data by decoding the stored trees in Rust; it had taken
  one column and refused to predict any tree term. R and Python agree to
  4e-16 on a factor-and-numeric tree, including an unseen level.

- `effect_model(data, spec)`: the conditional average treatment effect
  tau(x) as a boosted additive model, fitted by the R-learner. The outcome
  mean and propensity are cross-fitted (or taken from an `aipw` result via
  `nuisance=`), and tau is fitted to the pseudo-outcome (y - m)/(t - e) with
  weights (t - e)^2 over either linear-plus-smooth or one-split-tree terms,
  whichever has the lower cross-validated R-loss. `fx.model` is an ordinary
  boosted model, so `partial_effects` shows how each covariate changes the
  effect. The pseudo-outcome transform and the R-loss are in the Rust core
  (`mboostrs_core::causal`). On the simulated designs of the causal
  benchmark the conditional-effect error (PEHE) over six seeds was 0.122 on
  the smooth design (varying-coefficient boosting 0.19, grf 0.20) and 0.216
  on the threshold design (0.35 and 0.165).

- `gam_refit` refits `bols(x) + bbs(x, center=TRUE)` as the one smooth
  `bbs(x)`; it had failed with a message about interaction marginals. The
  refit's difference penalty leaves the linear part unpenalized, so the
  results equal a refit of `bbs(x)` exactly.

- Multinomial models take categorical predictors: `brandom(g)`, and a `bols`
  over a factor (lowered to it), are wrapped per class over their level
  indicators, in Python and R alike (agreement 6e-16); a label the fit never
  saw predicts at the average level. Every categorical column had made a
  Multinomial fit fail. `autospec` for a Multinomial response no longer
  proposes `btree` terms, which that family cannot fit, and reports pairs as
  not screened.

- `autospec(deselect=0.025)`: the deselection share as a number. The default
  (1%) is set for prediction; a larger share gives a sparser specification
  for identifying drivers. Measured on development seeds of the drivers
  design (not the published ones): at 2.5%, a 1,000-row binary response kept
  0.6 non-drivers per fit instead of 4.1 with the same driver recall (0.98),
  and 500 Gaussian rows kept 0 instead of 0.4; on seven real data sets test
  R^2 fell by 0 to 0.010.

- `GamRefit.effects(newdata, centre=True)`: each effect centred over the rows
  of `newdata`, with the interval of that contrast computed from the
  centred design, for comparing an effect with a curve centred over the same
  rows.

- Benchmark harness (`benches/_jobs.py`): each cached row carries its status
  and the identity it was measured under (runner and imported modules,
  installed mboostrs build, package and R versions, threads, platform, time);
  a row is reused only under the same identity, rows are strict JSON, and
  `tools/check_benchmark_provenance.py` requires the identity of every row.

- R `glmboost`, `gamboost`, `stabsel` and `predict` take a per-row offset
  (`offset = log(exposure)`), as mboost does: it starts every fit and every
  cross-validation fold, and new rows supply their own through
  `predict(offset =)`. The R frontends refused any offset longer than one
  number. Fitted values and cross-validated risk match mboost 2.9 to 1e-8.

- `autospec(...).fit(data, refit="reml")`: fit the recommendation, drop terms
  under 1% of the risk reduction (`deselect`) and refit the rest with REML
  smoothing parameters (`gam_refit`), in one call. `gam_refit(unsupported=
  "hold")` keeps a term it cannot represent (`btree`, `bmono`) at its boosted
  contribution and refits the others around it (`GamRefit.held`).

- Selection results name the data columns they selected the same way:
  `StabSelResult.variables` and `.selected_names`, `Deselection.variables`,
  `Probing.variables`; `formula.term_variables(term)`. For a distributional
  model `Deselection.variables` is `{parameter: columns}`. `probe(model)` takes a
  fitted model, as `deselect` and `stabsel` do. Models expose `family` and
  `formula`; families print as `Quantile(tau=0.9, qoffset=0.5)`.

- `GamRefit.tests` and `GamRefit.summary()` (R: `summary(gam_refit(...))`):
  each term's effective degrees of freedom, reference degrees of freedom, F
  or chi-square statistic and approximate p-value for no effect (Wood 2013),
  the table mgcv's `summary.gam` prints. Computed in the Rust core; agrees
  with mgcv 1.9.3 within its own 2e-5 tail tolerance on the body-fat model.

- `autospec(..., deselect=True)`: after tuning `mstop`, the recommendation
  drops main-effect terms under 1% of the tuned fit's risk reduction and
  tunes `mstop` again, reported in `evidence["deselect"]`; declared columns
  (`monotone=`, `coordinates=`, `measurements=`, `groups=` and the rest) are
  never dropped. On a 30-column
  benchmark with 6 drivers it kept 0 noise columns at 2,000 rows where it had
  kept 14.

- Benchmarks `benches/bench_drivers.py`, `bench_shapes.py`, `bench_pairs.py`
  and their book pages: driver identification among proxies and noise, eight
  effect shapes, and interaction discovery, all without a prior specification.

- `mboostrs_core::gam`: the REML refit in Rust (`try_gam_refit`), replacing
  the NumPy/SciPy implementation of `gam_refit`. The REML criterion is
  minimized by Newton's method with exact gradient and Hessian in the log
  smoothing parameters (Wood 2011), instead of L-BFGS-B on a numerical
  gradient; a tensor term's two penalties are diagonalized together once, so
  `log|S|+` and its derivatives cost O(rank) per evaluation and stay accurate
  when the two smoothing parameters differ by many orders of magnitude.
  Results agree with the Python version to 1e-4 in edf where the smoothing
  parameter is interior, and are closer to mgcv on the two reference cases
  (bodyfat sigma 3.9991 against mgcv's 3.999; Cox edf 5.513 against 5.51).
  A default `bspatial` refit takes 0.22 s instead of 0.74 s; the Cox refit
  0.05 s instead of 2.9 s. `GamRefit` gains `reml_score` and `convergence`.
  The Python layer now only builds the term designs.

- `gam_refit` optimizes the Laplace-approximate REML directly for Binomial,
  Poisson, negative binomial and Cox refits (mgcv's default), after the
  performance iteration: Newton's method finds the penalized MLE at each
  smoothing parameter, BFGS moves the smoothing parameters on the exact
  gradient, and a smoothing parameter that ends at its bound is restarted
  from the interior so the lower of two local minima is kept. Against mgcv
  the edf now agree to 3e-4 (Binomial), 1e-3 (Poisson) and 2e-3 (negative
  binomial), and predictions to 1e-5. On the Cox fixture the refit reaches a
  lower REML score than mgcv's default search (mgcv's own score: 3742.642
  against 3742.703 at its reported optimum), with x3 at edf 1.0 instead of
  2.6; mgcv started from that point agrees. `convergence["method"]` reports
  `"outer"` or `"performance"`. GammaReg also uses the outer iteration, with
  the gamma shape estimated by the criterion (edf within 3e-3 of mgcv).

- `gamlss_refit` (Python `mb.gamlss_refit`, R `gamlss_refit`): refits
  every distribution parameter's terms of a `gamboostLSS`/`glmboostLSS`
  model jointly by penalized likelihood, with smoothing parameters from the
  Laplace-approximate REML (Wood, Pya and Saefken 2016), in the Rust core
  (`mboostrs_core::gam::try_gamlss_refit`). `effects()`, `predict(type=
  "link" | "response")`, `vcov(type=)`, `edf`/`smoothing` per parameter.
  Against mgcv's `gaulss` on a simulated location-scale sample the edf agree
  to 0.02 and the fitted means and standard deviations to 0.003; Python and
  R agree to 1e-6. `LssFamily::neg_hessian` gives each family's per-row
  information (closed form for GaussianLSS, differences of the analytic
  gradient otherwise), checked against the risk for 23 families.

- Term designs and penalties for refits and `%X%` interactions are built in
  the Rust core (`mboostrs_core::gam::TermBasis`, now with `bols` ridge and
  factor-`by` margins; `row_tensor`, `kron`), replacing the NumPy
  difference penalties, indicator codings and tensor products in
  `interactions.py` and `selection.py`. Results are unchanged.

- Log-likelihood and conditional AIC for `gam_refit`: `GamRefit.loglik`,
  `.df`, `.aic()`; R `logLik()` and `AIC()`, and an R `plot()` method that
  draws each effect with its band. The log-likelihood follows mgcv's
  `logLik.gam` (matches to 1e-5 on body fat); the degrees of freedom are the
  refit's edf plus one for an estimated scale or dispersion, where mgcv's
  `AIC` uses its larger smoothing-parameter-corrected `edf2`.

- Posterior draws for GAMLSS refits: `GamlssRefit.draws(newdata, n_draws)`
  (R: `gamlss_draws(refit, newdata, n_draws)`) samples all coefficients
  jointly from `N(coef, cov)` (or the corrected covariance) in Rust
  (`LssRefitResult::parameter_draws`) and returns every distribution
  parameter on its natural scale, so intervals for centiles and other
  combinations of parameters include the correlation between their
  estimates.

- `GamRefit` and `GamlssRefit` pickle: the Rust results round-trip through
  JSON (`RefitResult::to_json`/`from_json`, the same for
  `LssRefitResult`), which the PyO3 classes use for `__reduce__`.

- Book: `methodology/gamlss_reml` (the joint penalized likelihood, Newton
  with the cross-parameter information, why per-row information can be
  indefinite, the Laplace-approximate criterion and its gradient, relation to
  the RS/CG algorithms and to boosting, the motorcycle example against
  mgcv's `gaulss`); `usage/refit` gains a `gamlss_refit` section and
  `usage/gamboostlss` points to it.

- Book: `methodology/penalized_reml` (boosting as shrinkage, the penalized
  fit, sum-to-zero constraints, the mixed-model view, the REML derivation
  with a computed criterion curve, Newton's method and its exact
  derivatives, tensor terms, effective degrees of freedom, interval
  coverage with a 200-sample simulation, the corrected covariance, PIRLS,
  the Laplace-approximate criterion, Cox and negative binomial refits, the
  implementation) and `usage/refit` (the `gam_refit` workflow on body fat,
  Munich rents and polio, the family table, surfaces, the R equivalent,
  limitations). The inference chapter's refit section points to both; the
  notation appendix and the R migration table have the new symbols and
  mapping. The survival, structure, spatial, causal and insight benchmark
  rows that use the refit were rerun with the Rust implementation.

- `gam_refit` estimates the negative binomial `theta` by the same REML
  criterion (as mgcv's `nb()` does), starting from the boosted fit's value
  (2.2947 against mgcv's 2.2947 on the test fixture, where the boosted fit
  had 2.354); `GamRefit.theta` / `refit$theta` report it.

- `vcov(type="corrected")` and `effects(..., corrected=True)`: the
  covariance corrected, to first order, for the uncertainty of the
  estimated smoothing parameters (`Vp + J V_rho J'`, Wood, Pya and Saefken
  2016). Smoothing parameters at a bound are treated as known and mgcv's
  second-order term is not included, so the corrected standard errors are
  smaller than mgcv's `unconditional=TRUE` ones (bodyfat: 0.955 against
  1.052 at one row, where the uncorrected ones agree at 0.761).

- R: `gam_refit(model, terms, which)` with `effects()`, `predict()`,
  `coef()`, `vcov()`, `print()` and `summary()` methods, over the same Rust
  core. Term designs are built in Rust from basis recipes
  (`mboostrs_core::gam::TermBasis`: `bbs`, `bols`, `brandom`, and the tensor
  of two margins for `bspatial`), and effects at new rows are computed by
  `mboostrs_core::gam::block_effect`; the refit object holds plain R data and
  survives `saveRDS`. Python and R agree to 1e-8 on the same data
  (`tests/test_gam_refit_r_parity.py`); against mgcv
  (`mboostrs/tests/test_gam_refit.R`) edf agree to 0.01 and predictions to
  1e-3 for Gaussian, Poisson, negative binomial, random-effect and
  tensor-surface fits.

- Book benchmark pages against the leading library in each category:
  `benchmark_structure` (counts, clusters, monotone effects, correlated
  drivers), `benchmark_uncertainty` (prediction intervals),
  `benchmark_survival`, `benchmark_classes` (unordered and ordered classes)
  and `benchmark_spatial` (mgcv, GPBoost, EBM, tree boosting), each with its
  harness in `benches/` and results in `docs/book/cases/bench_*.json`.

- One export convention for result records: `to_dict(*, include_arrays=False)`
  on every public result class (29 classes, from `CvRiskResult` and
  `StabSelResult` to `AIPWResult`, `Deselection` and `GamRefit`). The flag is
  keyword-only, so `to_dict(True)` raises `TypeError` instead of changing the
  payload; the default record is accepted by `json.dumps` and leaves out
  per-observation and per-fold arrays, which `include_arrays=True` adds.
  `__repr__` is a single line that never prints an array.
  `tests/test_export_contract.py` checks the signature of every class and the
  JSON safety of real records.

- `mb.deselect(model, threshold=0.01)` and `mb.probe(formula, data)`
  (module `mboostrs.selection`) select terms after, or instead of,
  cross-validated stopping. Deselection (Strömer et al. 2022) drops the
  learners whose share of the in-bag risk reduction is below the threshold
  and refits the rest with `mstop` re-tuned; probing (Thomas et al. 2017)
  adds a permuted copy of every single-column term and keeps the terms
  selected before the first copy. On `benches/bench_insights.py`'s additive
  task (5 signal, 5 noise features) a CV-tuned `gamboost` kept 4.7 noise
  features on average; deselection kept 0 and probing 0.7, with lower
  held-out error. Results are `Deselection` and `Probing` records with
  `to_dict()`.

- `mb.gam_refit(model)` refits a Gaussian `gamboost` model's selected terms
  (or `deselect(...).kept`) as one penalized GAM on the learners' own designs
  and penalties, with each term's smoothing parameter chosen by REML, and
  returns `GamRefit` with effective degrees of freedom, sigma and pointwise
  Bayesian intervals (`effects(newdata, level=)`). On the benchmark's
  additive task its nominal 90% bands covered the true effects 90.8% of the
  time, against 62% for the bootstrap of the boosted fit, whose estimates
  are shrunk by early stopping.

- `na_action="impute"` on `gamboost` and `glmboost`: each numeric predictor
  with a missing value is filled with its training median and a
  `<x>_missing` indicator enters as its own term; both are replayed on new
  data, so the model predicts rows with missing values. With `"omit"` a
  10%-per-feature missingness left 321 of 1,000 training rows and predictions
  for 35% of test rows.

- `autospec(..., interactions="adopt")` adds the pair terms the interaction
  screen proposes (duplicate-axis pairs excepted) and re-tunes `mstop`;
  `SpecRecommendation.fit` expands the `%X%` terms. The default stays
  `"report"`.

- `MstopTuning.model`: the tuned full-data fit, so `tune_mstop(...).model`
  tunes and fits in one call.

- `bols(f)` on a categorical column in `gamboost` fits one effect per level
  (R's dummy-coded `bols(factor)`; the fitted path matches R to 1e-10), and
  a bare factor term or `y ~ .` does the same. `glmboost(y ~ x + f)` expands
  `f` into R's treatment-contrast columns (`fb`, `fc`), matching R's
  coefficients to 1e-8.

- `gam_refit` accepts a numeric `by=` on its terms (a varying coefficient):
  deselection then refit brought the conditional-treatment-effect error on
  `benches/bench_causal.py`'s smooth design from 0.176 to 0.109 (mgcv 0.105).

- `autospec` fits a strictly positive predictor with skewness above 2 as
  `bbs(log(x))` and reports it ("right-skewed predictor"); on California
  housing its specification reached held-out R^2 0.74 where `bbs` on the raw
  columns reached 0.21. It also reports a frame above 10^5 rows as "large
  frame" with `bins=256` as the remedy. `cvrisk`
  on `gamboost`/`glmboost` warns when one fold's held-out risk at the last
  step is more than ten times the median fold's, which lets that fold decide
  `mstop` (California housing: 11 of 2,000 steps chosen, held-out R^2 0.43
  against 0.71 at 2,000).

- `gam_refit` fits Binomial, Poisson and GammaReg models by penalized
  iteratively reweighted least squares with REML smoothing at each step
  (mgcv's performance iteration). Binomial effects are reported on the
  log-odds scale, Poisson and GammaReg on the log of the mean;
  `GamRefit.predict(newdata, type="response")` returns the mean.

- `ecdf(x)` formula transform: the training empirical distribution function,
  interpolated between at most 257 recorded quantiles and held at 0 and 1
  outside the training range, replayed on new data and saved with the model.
  `bbs(ecdf(x))` places the spline's knots at quantiles of `x` and predicts a
  flat effect beyond the training range. On California housing (three
  splits) `bbs(ecdf(x))` on every column reached held-out R^2 0.73 to 0.74,
  against -0.19 to 0.66 for `bbs(x)`, whose cross-validated `mstop` was
  decided by folds that held out extreme `AveOccup` rows.

- `na_action="impute"` gives a categorical predictor's missing values a level
  of their own, `"(missing)"`, which is never the reference level. With the
  default `na_action="error"` a missing label raises a `ValueError` naming
  the column (it used to fail while sorting levels).

- `predict(type="log_odds")` on `gamboost`/`glmboost` and
  `partial_effects(scale="log_odds")` on `gamboost` report a Binomial model's
  predictor and learner contributions on the log-odds scale (twice mboost's
  link), the scale other libraries report. Other families raise.

- `glmboostLSS` and `gamboostLSS` accept string factor columns: a bare
  factor term in `glmboostLSS` becomes treatment-contrast columns, and
  `bols(f)`/`brandom(f)` in `gamboostLSS` are encoded as in `gamboost`. A
  factor `by=` is refused.

- `mb.deselect` accepts a `method="noncyclic"` `glmboostLSS`/`gamboostLSS`
  fit: shares are computed over all distribution parameters, a parameter
  left with no term is refitted as `y ~ 1`, and the refit's total step count
  is cross-validated. `kept` and `dropped` entries carry the parameter name
  and `formula` is a per-parameter mapping. A cyclic fit is refused.

- `confint(groups=)` on `glmboost` and `gamboost` models draws whole groups
  in every bootstrap sample instead of single rows, and the inner `B_mstop`
  tuning folds draw groups too. `groups` is a column name or one label per
  row; a row's weight in a sample is its group's draw count times its model
  weight. Use it when rows are repeated measurements of a unit (the points of
  a curve, the visits of a patient): resampling such rows separately treats
  them as independent and gives intervals that are too narrow. The result
  records `resampled` ("rows" or "groups").

- `boost_control(step=, step_iterations=)` chooses how each boosting step's
  length is set once the winning learner is selected. The default `"fixed"`
  is the classic constant step `nu * h` that `mboost` takes.
  `step="line_search"` finds the multiple `rho` of the winner's direction
  that minimizes the training loss — a coarse grid bracket (0 … 4, doubling
  while the loss falls) plus golden-section refinement, a handful of `O(n)`
  loss evaluations per step — then takes `nu * rho * h`, so `nu` becomes a
  fraction of the loss-optimal step and stops meaning a different amount of
  progress for every response scale. Friedman's (2001) original gradient
  boosting multiplies each fitted learner by this `rho`; Bühlmann & Hothorn
  (2007) omit the search, as `mboost` does. Selection is unchanged; the
  squared-error families are bit-identical either way (the least-squares
  step is already optimal, `rho = 1`); a large-count Poisson fit that
  diverges under the fixed step now completes, and a rare-event binomial
  fit reaches the same risk in a fraction of the iterations. The rule
  travels with the model: `cvrisk`, `stabsel`, `update`, `continue_to`,
  `confint` and conformal refits all carry it, and artifacts store it. The
  non-cyclic LSS method refuses it, naming its reason. The record grew two
  defaulted fields (`step`, `step_iterations`); controls written as bare
  4-tuples still work and read as the fixed step.

- `compare_specifications` rows now carry `search optimism` and
  `competitive set`. `search optimism` is the Tibshirani & Tibshirani (2009)
  estimate of how optimistic the winning out-of-fold score is because it is
  the best of the candidates compared: the winner's fold score minus the
  lowest score any candidate reached on that fold, averaged over folds
  (within each partition for repeated schemes), in the score's own units.
  `competitive set` lists every candidate within one paired fold standard
  error of the winner, winner included — the candidates the folds cannot
  tell apart. Both fields describe the comparison, so they are the same on
  every row. A repeated candidate lists its name once.

- `compare_specifications(history=...)` extends both fields across the calls
  of one search: pass the rows of earlier `include_evaluations=True`
  comparisons on the same data, folds and score, and the optimism and
  competitive set cover every candidate tried, in the order tried. Each row
  also carries `search candidates` and `search fits`, the counts across every
  call, history first. Rows without their retained evaluation are refused,
  as are rows whose data, folds or score do not pair with the current call.

- New `holdout(data, fraction=0.2, *, groups=None, time=None, seed=0)`
  returns `(search, test)` before any fitting starts, as the same kind of
  frame as `data`: rows dealt at random by default, whole groups dealt to
  one side under `groups=`, the most recent rows under `time=` — each split
  matching the task its test score describes. The test part is remembered by
  content: the evaluation assistants (`evaluate_specification`,
  `compare_specifications`, `term_report`, `tune_mstop`) warn, from the
  second specification pointed at those exact rows, how many have now used
  them, since refitting test rows spends them. Hand-written
  predict-and-score loops are not intercepted; score the test part once.

- `cv_risk` — and every assistant built on it: `evaluate_specification`,
  `compare_specifications`, `term_report`, `tune_mstop` — warns when a group
  term (`brandom`, `bmrf`) is scored on folds that split its groups: rows of
  a group held out while other rows of the same group train the fold. That
  fold's score partly rewards memorized level effects; the warning names the
  column, counts the affected folds, and points to folds that keep groups on
  one side (`mboostrs.folds.grouped_folds`), which measure prediction in new
  groups instead. Folds that already keep groups together stay silent.

- `learner_design(term, data)` and `LearnerDesign(term, data)` expose the fitted
  design/penalty recipe used by
  `bbs`, `bols` or `brandom`. Its `LearnerDesign` result carries the fitting
  `design` and `penalty` matrices for `buser` and evaluates the same fitted
  basis on new rows through `transform(newdata)`. The `buser` case no longer
  imports private interaction helpers.

- `benches/bench_cz_agent_search.py` runs the specification search of
  `cases/cz_agent_search` five times, setting aside each fifth of the states in
  turn, so every commuting zone is predicted once by a model whose search never
  read its state. `--simplification-se` sets how many paired standard errors a
  removal may cost and still be accepted, and each split's result is written as
  it finishes. Outputs `docs/book/cases/bench_cz_agent_search.json` and, for the
  rule the case rejects, `bench_cz_agent_search_two_se.json`.

- `cases/munich_rents_distributional` models log rent as `GaussianLSS`: smooth
  terms for floor area and construction year in the location, and either the
  same smooth terms or a constant in the scale. Validation holds whole
  districts out; inside each outer training portion, grouped folds select the
  stopping point of each scale specification, and the inner winner supplies
  the outer predictions. Scores are negative log density on the euro scale
  (the log-rent density plus the Jacobian term `log y`), with the central 90%
  interval's coverage and width in euros per month. District bootstrap
  intervals and a probability-integral-transform check describe evaluation
  uncertainty and calibration, and the page closes by saving the selected
  model and scoring a reloaded copy on the same rows.

- `cases/distributional_comparison` compares mboostrs with R's `gamboostLSS`,
  `mssm` and `NGBoost` on two Gaussian generating processes — one additive
  mean, one with an `x1*x2` interaction in the mean and the log scale — under
  two tracks (additive predictors, or one explicit interaction term) and
  three configurations per library, each configuration selected on the same
  three inner folds. Five independent training/test draws per setting are
  scored by one `gaussian_scores` (log score, CRPS, 90% coverage and width);
  paired Monte Carlo standard errors compare each competitor with mboostrs
  draw by draw, and parameter-recovery MSE compares each fit with the known
  mean and log-scale functions. `benches/bench_distributional.py` writes
  `docs/book/cases/bench_distributional.json`, which the page reads so a book
  build needs none of the comparison libraries. In the recorded run mboostrs
  and `gamboostLSS` return identical scores at equal cycle counts and step
  length, `mssm`'s REML smoothing attains the lowest mean log score on both
  processes (0.016 to 0.025 below mboostrs), and `NGBoost` is highest on every
  track.

- `cases/specification_search_validation` repeats the documented specification
  search on independent data: four known processes, twelve new training
  samples each, and a fresh 2,000-row test sample that no probe or search sees.
  The study measures the change in test error from the initial `autospec` fit
  to the final search fit, the noise columns and interactions the final
  formula carries, which generating main effects it retains, and keeps every
  search transcript. `benches/bench_specification_validation.py` writes
  `docs/book/cases/bench_specification_validation.json`; the page reports
  unfavourable changes too, because a smaller formula can predict worse.

- The notebook download bundle carries a page's declared evidence. A notebook
  lists the local files it reads — its recorded-study JSON — in
  `metadata.mboostrs_artifacts`; `docs/book/generate_notebooks.py` includes
  those files beside the notebook, and `tools/check_surfaces_agree.py` checks
  their presence and content, so a downloaded recorded study runs without
  fetching inputs the site never disclosed.

- The term `1` in an LSS parameter formula is the intercept, as in R's
  `y ~ 1`. `gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}, data)` fits
  one estimated constant for the scale: the parser emits one `bols` on an
  implicit column of ones (`intercept=False`, because that column *is* the
  constant), prediction rows supply it, and cross-validation runs unchanged —
  no fabricated `one=1.0` column, no `bols(one, intercept=False)`
  workaround, whose exact numbers this spelling reproduces to zero
  difference. `cases/munich_rents_distributional` uses it. A frame that
  actually holds a column called `1` keeps referring to the column, the
  convention `expand_formula` documents for `0`/`1` terms, and `gamboost`
  still refuses intercept control with its existing message, because its
  constant lives in the offset.

- `compare_specifications` and `cv_risk` accept distributional candidates.
  With `family=GaussianLSS()`, a candidate may be a mapping of parameter name
  to right-hand side — `{"mu": "bbs(x)", "sigma": "1"}` — or a plain string
  shared by every parameter, which is what a single-formula `gamboostLSS`
  call means. Each candidate is fitted as one distributional model per fold
  and scored by the family's joint proper score, the negative log predictive
  density from the family's own `pdf`, so a spread diagnosis from
  `heteroscedasticity` can be compared in the same assistant that proposed
  it. Parameter names are checked against the family (`{"mu", "scale"}` is
  refused at the call), the single-quantity scores are refused — they would
  drop the fitted scale and rank by another target — and `mstop="cv"` tunes
  inside each outer fold with the same planned-fit accounting. On 300 rows
  whose scale grows with `x`, the constant-scale candidate scored 1.40
  against 1.09 for the varying scale, a change of −0.31 with paired
  standard error 0.045.

- `offset=` accepts one known contribution per observation in `gamboost` and
  `glmboost`, R's `offset=` semantics: `log(exposure)` makes a Poisson fit a
  rate model, and boosting never re-estimates the offset. The vector is the
  frozen start of the linear predictor; the intercept column is fitted on top
  of it, which is what R does — mboostrs matches R's `glmboost(offset=)` to
  8.6e-9 and `gamboost(offset=)` with `bbs` to 1e-13 and with `bols` to 6e-15
  on the link scale (the `bols` path switches to the two-column block form,
  because the centered form borrows its constant from the model offset and a
  per-observation offset is not one number). The vector flows through
  cross-validation, training-row prediction and `save_model`/`load_model`
  unchanged; predicting `newdata` requires the new rows' own `offset=`
  (refused otherwise, with the remedy in the message). A zero-exposure row
  gives `log(0) = -inf` and is refused with the two remedies — drop the row
  or weight it zero — instead of entering the first risk evaluation;
  Multinomial and survival responses refuse a vector with the reason (one
  offset per observation names no unique vector for them). `tests/
  test_observation_offsets.py` carries the contract, including the two R
  parity tests.

- `save_model(..., scoring_only=True)` keeps what prediction reads and drops
  the rest of the training state. The response column — the transform step's
  *source* when the response was fitted as `log(rent)` — leaves the retained
  frame, and the formula and control a refit needs become `None`, so a
  loaded scoring-only model predicts on new rows and on its stored fitted
  path, keeps its recipes, tensor designs and response transform, and refuses
  `update`/`with_mstop`/`continue_to` with the retention error instead of
  quietly carrying a full checkpoint. Nothing about the fitted model
  changes: the same fit saved both ways predicts identically (pinned by
  tests on transformed-response and `%X%` models).

- The R reader loads distributional artifacts. `mboostrs/R/mboostrs_load.R`
  now decodes the `lss_family` tag (which carries `param_names` for readers
  without the Python class), the per-parameter learner blocks, a response
  transform step and an interaction recipe, and reports family, per-parameter
  step counts, offsets, learner names and the scoring-only marker — while
  still scoring linear `glmboost` artifacts directly and refusing to predict
  from what needs the Python reader's spline machinery.
  `cases/munich_rents_distributional` states both facts in its saving section.

- `TermReport` carries `fits`, the exact number of model fits it spent: the
  baseline refit plus one per removal and per shape candidate, each on every
  fold — counted from the same evaluation results that produce its scores,
  on both the standard and the distributional paths. With this, every piece
  of evidence a search reads states its own cost (`SpecRecommendation.fits`,
  `TermReport.fits`, the comparison rows' fit counts, `MstopTuning.fits`),
  and the documented search loop's transcript now totals them across the
  whole procedure — autospec's `budget=` caps only its probes, while the
  term reports, per-round refits, comparisons and the stopping sweep all
  spend outside it, which the transcript states instead of implying the one
  budget covered everything.

- `term_report` and `tune_mstop` accept distributional specifications — the
  existing functions, extended, with no parallel `term_reportLSS` /
  `tune_mstopLSS` names. `term_report({"mu": "bbs(x, df=4)",
  "sigma": "1 + bbs(z, df=4)"}, data, "y", family=GaussianLSS())` reports what
  each term of each parameter earns on the joint proper score, with paired
  fold standard errors, and whether another shape of it earns more; a
  parameter that loses its last term becomes the estimated constant `1` in
  the removal candidate, and an intercept that is a parameter's only term is
  not a candidate. `tune_mstop({"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ 1"},
  data, family=GaussianLSS())` runs the same doubling grid, read off the LSS
  diagonal, and its optimum is exactly what the fitted model's own `cvrisk`
  reports — one fit per fold per grid, boundary detection on the sampled
  step counts. On a frame whose mean is smooth in x with noise z in the
  scale, the report separates the three cases: the signal term earns more
  than two paired SEs, the noise term's removal lowers the score, and the
  explicit `1` beside a spline earns exactly nothing.

- `gamboostLSS` accepts `%X%` tensor interactions:
  `gamboostLSS({"mu": "y ~ bbs(x, df=4) %X% bbs(z, df=4)", "sigma": "y ~ 1"})`
  expands each tensor through the same recipe `gamboost` uses — a derived
  `buser` term whose design rides in `env` — where it previously failed inside
  the formula parser with "got df= more than once". The model keeps the
  interaction recipes: `predict(newdata=)` rebuilds the design at the new
  rows, `with_mstop`/`update`, `cvrisk`, `stabsel` and `save_model`/`load_model`
  all carry it, and the module-level `cvrisk_gamboostLSS`/`stabsel_gamboostLSS`
  entries accept a caller's raw `%X%` formulas. `glmboostLSS` keeps its
  one-linear-term-per-predictor contract and refuses `%X%` with the reason
  and the bilinear alternative. On a genuinely curved surface — sin(x)(z²+1/2),
  which no bilinear form spans — the tensor's held-out mean error was 4.5×
  lower than the additive-plus-`bols(x, by=z)` alternative. The R comparison
  found an upstream boundary: R 2.9-11's own `gamboostLSS` cannot fit
  `bbs(x) %X% bbs(z)` (it fails in `knotf` with "boundary.knots must be a
  vector of length 2" while the same term fits in R `gamboost`), so mboostrs
  supports a specification upstream cannot; the recorded oracle in
  `tests/test_lss_tensor_interactions.py` recomputes predictions from the
  rebuilt design by hand instead.

- A formula may carry the response's transform: `gamboostLSS({"mu":
  "log(rent) ~ bbs(area, df=4)", "sigma": "log(rent) ~ 1"})` fits the Gaussian
  to log(rent) and the model remembers the step, so no `log_rent` column is
  fabricated and nothing is inverted by hand. `prediction_interval` (and the
  `predint`/`PI` aliases) returns euro-scale bounds, exactly the inverse
  transform of the log-scale quantiles; `pit` and `quantile_residuals` compare
  the observed rent with the fitted distribution on the scale it was fitted
  to; `evaluate_distribution` scores the original response, carrying the
  change-of-variables factor (`+ log(rent)`) that used to be added by hand,
  and says so in its assumptions. The response whitelist is the invertible
  single-column transforms — `log`, `log1p`, `sqrt`, `exp` and the affine
  `scale`/`center` (fitted on the training response) — because quantiles and
  densities must be able to come back; `pmin`/`pmax` and `I()` on a response
  are refused with the reason. The step survives `with_mstop`/`update`,
  slicing, `continue_to`, and `save_model`/`load_model`; the LSS `cvrisk` and
  `stabsel` entries accept a caller's raw or rewritten formulas;
  transformed predictors work in LSS formulas and replay for new rows. On
  lognormal rent the interval matches `scipy.stats.lognorm.ppf` to 1e-12 and
  the evaluated mean log score matches the direct lognormal negative log
  density exactly; `cases/munich_rents_distributional` now writes the
  declarative formula and lets the evaluation carry the Jacobian.

- `unseen_levels=`, the point policy for a `brandom` level the fit never
  saw, reaches every LSS prediction surface instead of only `predict`:
  `params`, `predict_all`, `prediction_interval` (and the `predint`/`PI`
  aliases), `pit`, `quantile_residuals`, `residual_frame`, and
  `evaluate_distribution` all take it, default to refusing as `predict`
  does, and thread it to the same underlying call — so a grouped LSS fit can
  be scored on held-out groups anywhere, consistently.
  `evaluate_distribution` records a non-refusing policy in its result's
  assumptions when the model carries a group term ("rows whose group level
  the fit never saw are scored as a member of an average group... every
  score conditions on that point prediction and none integrates new-group
  uncertainty"), and refuses a policy a model's `predict` cannot apply.

- `grouped_folds(..., repeats=)` deals the groups several times and stacks the
  deals, as `repeated_kfold_folds` stacks row partitions; every group is held
  out once in each deal. With few groups, which groups share a fold moves a
  comparison by as much as the difference between two specifications, and
  several deals average that out. `FoldScheme.repeats` records the number of
  partitions a scheme holds.

- `autospec` returns a `SpecRecommendation` instead of a bare tuple, and can
  measure the guesses it used to only assume. It still unpacks as
  `(formula, learners, family, mstop)`, so existing callers are unaffected,
  and `budget=` — the number of model fits the probes may spend, zero by
  default — decides how much is measured rather than defaulted:
  - **data hygiene**, at no fit: constant and near-constant columns (dropped
    from the specification, since a term on a constant fits the intercept
    twice), heavy missingness with the rows a fit would drop, a float column
    holding two or three distinct values, and response shape — zero inflation,
    a few-valued float response, a right-skewed positive one.
  - **family**, from held-out spread rather than dtype: counts are scored by
    Pearson dispersion, so an overdispersed count moves from `Poisson` to
    `NBinomial`; a positive response whose spread does not grow with its mean
    moves from `GammaReg` to `Gaussian`. Both are cases the dtype rule cannot
    see.
  - **learner shapes**, by held-out risk on shared folds: a threshold effect
    moves from `bbs` to `btree`. A shape that wins by less than 2% leaves the
    default standing, and the report carries both numbers.
  - **interactions**, screened from the main-effect residual and *reported,
    not added*: the screen is a training-residual lead, not held-out evidence,
    so the recommendation stays additive and hands over the term to test. A
    group column (a `brandom` term) is paired as a random slope,
    `brandom(g, by=x)`, and a two-valued column as a smooth it switches on,
    `bbs(x, by=b)`, rather than gridded through `%X%`; the pairing follows the
    term the recommendation chose rather than the dtype.
  - **mstop**, read off one cross-validated path so the whole grid costs one
    sweep; an optimum at the last iteration is reported as a floor, not an
    optimum.
  - **a prior**: pass `prior="y ~ bbs(x) + bbs(z)"` (or a mapping with
    `family`/`mstop`) and it becomes the baseline every probe measures
    against, each of its terms the incumbent shape.
  - `report()` prints the reasoning with the findings ordered by what they
    cost to get wrong, and marks every answer as measured or defaulted.

- `unseen_levels=` decides how a `brandom` term predicts a level the fit never
  saw, such as a group held out of a grouped cross-validation fold.
  `"average"` predicts the row as a member of an average group: the level's
  coefficient is the mean of the term's level coefficients. `"zero_effect"`
  gives it a zero coefficient, which is what `model.cvrisk` computes for a
  held-out group. The two agree closely for a random intercept and differ for a
  random slope (`brandom(g, by=x)`), which absorbs part of the average slope
  during boosting: on 30 groups, new-group squared error was 1.87 with
  `"average"` and 2.07 with `"zero_effect"`. The default, `"error"`, refuses
  the row, and the error names both policies. The keyword is on `predict`,
  `predict_over_mstop`, `partial_effects` and `prediction_contributions` of
  `gamboost` models and on `predict` and `partial_effects` of `gamboostLSS`
  models, and it covers string-labelled group columns. `cv_risk`,
  `evaluate_specification` and `compare_specifications` default to
  `"average"`, and each fold record counts the rows it applied to as
  `n_unseen_level_rows`.

- Whole-number group codes above twelve values are read as groups when the
  response says so. `label_evidence` compares the mean response at each value:
  a grouping code has level means that vary beyond sampling noise (a one-way
  analysis-of-variance F at least 4 standard deviations above its no-signal
  value) while adjacent values differ as much as arbitrary pairs.
  `classify_columns(..., response=)` and `autospec` then fit the column with
  `brandom`. On the packaged data, India's 404 district codes and Munich's 290
  district codes read as groups; child age in months and apartment area read
  as measurements.

- `autospec(..., groups=)` declares group columns: each gets a `brandom` term,
  and the probes share grouped folds that hold whole groups out. The report's
  first line after the formula says which folds the probes used.

- `autospec`'s data hygiene scan reports columns that are not predictors: a
  text `identifier` (at least 90% distinct values; left out), a
  `unique-integer` column (identifier or count; kept), a `response-sibling`
  (a column named like the response, with its correlation), a
  `duplicate-encoding` of a grouping another column records (left out), and
  `integer-labels`.

- `tune_mstop` cross-validates the step count and doubles the grid while the
  risk is still falling at its end; `autospec`'s step-count probe uses it
  within its fit budget and reports the extension.

- `pair_term_for` returns the interaction term two columns call for: a tensor
  spline for two continuous columns, `bbs(x, by=b)` for a two-valued column,
  and a random slope `brandom(g, by=x)` for a group column.
  `suggest_specifications` and `autospec` use it.

- `term_report` records `se`, the standard error of each gain's paired fold
  difference. `suggest_specifications(noise=2.0)` leaves out shape changes
  whose gain is within two standard errors of zero, and offers the terms that
  earn nothing as one joint drop as well as one at a time.

- `heteroscedasticity` also tests the residual spread against each continuous
  predictor (`by_column`, `strongest`), and the distributional suggestion names
  the predictor the spread follows.

- `compare_specifications` rows carry `change se`, the standard error of the
  paired fold differences behind `change`.

- `screen_interactions(..., groups=)` scores a group column on its own levels.
  Quantile bins of a group code put unrelated groups in one cell, which hid a
  random slope.

- `suggest_specifications` also proposes removing a term whose gain is less
  than `noise` standard errors above zero, since the folds cannot tell it from
  a term that earns nothing, and offers all such terms as one joint removal.

- `benches/bench_agent_search.py` runs the documented specification search on
  twenty simulated processes and records, for each, the true terms, interactions
  and groups the final specification carries, the noise columns it keeps, and
  its held-out squared error against the true mean function's.

- Every benchmark page opens with an annotated great_tables comparison of all
  methods (title, subtitle with the replication, spanners, the best value per
  column in bold, source notes that define each column and its uncertainty),
  and the pages' result tables use the same form. `docs/book/cases/bench_tables.py`
  builds them. The PDF build converts each table's HTML: `patch_tex.py` now
  restores the header cells a two-row stub drops, sets short columns at a
  fixed width that fits their values and wraps long ones (`tabularx`, or a
  `longtable` across pages), and narrows spanners to their columns.

- `btree` chooses the column to split as mboost's `btree` does, by partykit's
  conditional inference rule (`selection="ctree"`, the new default): the
  column with the largest quadratic association statistic with the gradient,
  then its best split. A multi-column tree now matches `mboost::btree` to
  1e-15 (stumps and depth 2); the squared-error rule it used before differed
  by 0.12 on a three-column example, and stays available as
  `selection="sse"`. `selection="ctree_pvalue"` compares the statistic's
  chi-squared p-values, which removes the rule's preference for factors with
  many levels. A one-column tree is unchanged. `autospec`'s residual tree
  uses `selection="sse"`, the rule it was evaluated with.

- `autospec` is 5 to 15 times faster. Its own default step length is 0.3
  (`gamboost`'s stays 0.1): the probes walk each candidate's path to its
  cross-validated optimum, and at 0.1 those paths ran to 10,000 steps, while
  held-out R^2 on development data matched within 0.001. A log-link family
  guessed from the dtype (whole numbers as counts) is no longer fitted
  before the family probe judges it: the depth sweep uses Gaussian, and only
  a fit that overflows switches to the line search, instead of the whole
  search restarting with it. On cpu_activity (4,000 rows, 21 columns)
  `autospec(interactions="adopt")` went from 1,673 to 107 seconds with the
  same held-out R^2 (0.9824 against 0.9825).

- `cvrisk`, `stabsel` and `deselect` on a weighted model multiply the
  model's weights into a 0/1 fold mask such as `cv_weights(n, "kfold", 5)`.
  The fold refits had used the mask itself as their weights, so they were
  unweighted while the held-out risk was weighted; mboost behaves the same way
  and documents `cv(model.weights(fit))` as the way to avoid it. On a
  pseudo-outcome regression with weights between 0.0001 and 1 the old
  behaviour stopped boosting after 10 steps instead of 25 to 300. A matrix
  that already carries the weights (`cv_weights(..., weights=w)`), a bootstrap
  count matrix and a fractional matrix are used as given. The R `glmboost`
  `cvrisk` applies the same rule, and its default folds now carry the model's
  weights.

- R `cvrisk()` on a weighted `gamboost` model returns the cross-validated
  risk; it had refused, because the engine scored every held-out row with
  weight one. The engine takes held-out risk weights
  (`engine::try_fit_with_oob_weights`), and each fold's held-out rows are
  scored with the model's weights and divided by their total, as mboost
  does. It agrees with `mboost::cvrisk` to 1e-13 and with the Python
  `cvrisk` to 7e-13.

- `autospec` writes each continuous column of a Multinomial model as mboost's
  decomposition, `bols(x) + bbs(x, center=TRUE, df=1)`, instead of
  `bbs(x, df=4)`. On development seeds of four three-class designs (1,500
  rows) the excess test log-loss over the true model fell from 0.020 to 0.011
  (mixed effects), 0.023 to 0.006 (linear), 0.019 to 0.016 (strongly
  nonlinear) and 0.024 to 0.014 (mixed, 14 noise columns). Deselection treats
  a column's two terms as one.

- `autospec`'s family probe checks a count family (Poisson, NBinomial) against
  Gaussian on held-out squared error, as it already did for GammaReg, and
  the rule for both is now: replaced when worse by 2% of Gaussian's error and
  by two paired standard errors, or by half again whatever the spread. A
  Gamma fit that ran away in one fold had been kept at ten times Gaussian's
  error because its standard error was large (energy_efficiency: test R^2
  -0.10 on one split, now 0.989); hours worked per week, whole numbers with a
  mass at zero, move from NBinomial to Gaussian (health_insurance R^2 0.31 ->
  0.37). A candidate family whose fit diverges loses the comparison instead
  of stopping `autospec`.

- Relative comparisons inside `autospec` (the shape gain, the held-out check
  of adopted pairs and of deselection) measure the risk from the saturated
  model's risk for Poisson, NBinomial and GammaReg, whose risks are negative
  log-likelihoods that go below zero; near zero the ratio had turned
  noise-level differences into tens of percent.

- A right-skewed non-negative column that holds zeros is fitted as
  `bbs(sqrt(x))` (a zero cannot be logged), and the shape probe now compares
  a transformed column's alternatives on its transformed scale; it had
  skipped every log- or square-root-transformed column.

- `na_action="impute"` predicts a new row missing a value in a column that
  had none missing in training, at the training median.

- Gaussian fits and cross-validation with `btree` terms next to linear terms
  (`bbs`, `bols`, `brandom`) run on the Gram route: the linear terms keep
  their projection updates and each tree is fitted to an explicit residual
  kept beside them, instead of the whole fit falling back to the general
  route. Results agree with the general route to 1e-8 (same selection
  sequence); a five-fold sweep of 2,000 steps with 19 `bbs` and 2 `btree`
  terms on 5,000 rows took 3.3 s instead of 11.4 s. `autospec`'s shape probe
  scores a `btree` candidate for every column, so it gains the most.

- `autospec` deselection is checked on held-out data: the terms under 1% of
  the risk reduction are dropped only as far as the cross-validated risk, on
  the same folds, rises by at most 1%; when dropping all of them rises more,
  the smallest are dropped and the rest kept (a bisection over the terms
  ordered by share, one sweep per step), and the report says which and why.
  Many small real effects can together carry the fit: on cpu_activity the
  unchecked rule dropped 16 of 21 terms and raised the held-out risk by 67%
  (test R^2 0.984 -> 0.972); the check drops the 7 near 0% and keeps the
  rest (R^2 0.983). `rec.fit(refit="reml")` no longer deselects a second
  time, which bypassed the check and overrode `deselect=False`.

- `autospec`'s family probe recommends Gaussian for a count response whose
  held-out Pearson dispersion is clearly below 2/3: the values vary less
  than a Poisson mean allows (percentages, ratings, bounded scores stored as
  whole numbers). CPU utilisation (cpu_activity, dispersion 0.09) had been
  kept on Poisson, which diverged at the fixed step and ran every probe with
  the line search.

- A whole-number column with more than twelve values is read as group labels
  only if its values show no order: a permutation test pools consecutive
  values into about 20 blocks and compares the spread of the block means with
  199 shuffled orders (`LabelEvidence.order_p`). The adjacency ratio alone
  called a skewed measurement with many two-row levels a grouping (disk reads
  per second in cpu_activity, ratio 0.61), fitted it with `brandom` and made
  every probe hold whole "groups" of it out.

- `autospec` reads a whole-number response with median above 1,000 as a
  continuous positive measurement (GammaReg, or Gaussian after the family
  probe) instead of Poisson counts, with a warning. Poisson's gradient
  `y - mu` is on the response's scale, so on house values near 2e5 every
  fit diverged in the first step (california_housing failed on all splits);
  at that size rounding and counting cannot be told apart, and a Poisson
  standard deviation is under 3% of the mean.

- `y ~ .` leaves out a column holding a single value in the rows, with a
  warning. A random split made `area_largest` constant in solar_flare's
  training rows and the spline over it failed on its boundary knots.

- `GAMBoostModel.predict`, `GLMBoostModel.predict` and `LSSModel.predict` take `newdata` as the
  first positional argument (`model.predict(test)`), as `GamRefit.predict`,
  R's `predict(object, newdata)` and scikit-learn do; the keyword form is
  unchanged.

- `autospec(nu=None)`: the default step length is 0.1, lengthened to 0.5
  when the cross-validated risk is still falling at the 10,000-step cap
  (strongly collinear columns), and the report says so; an explicit `nu` is
  used as given.

- `autospec(interactions="adopt")` may adopt up to six screened pairs (the
  report lists three); pairs the tuned fit leaves unused are dropped by the
  deselection step, and a pair is kept only when removing it raises the
  cross-validated risk by 2% or by two paired fold standard errors (the
  screen reads the training residual and proposes chance pairs when there
  are many columns and few rows; the report lists the removed ones). On
  grid stability it adopts all four reaction-time x price-elasticity pairs
  (test R^2 0.883 -> 0.915).

- `tune_mstop` cross-validates each doubling of its grid from a one-step
  model and fits the full path once at the end, instead of refitting the
  full-data model at every doubling (1.8 s to 0.4 s on a 12-term, 1,000-row
  fit). `autospec`'s shape probe compares candidates one column at a time
  with the rest of the model held fixed (one-term fits), which with the
  former took `autospec(budget=200)` from 57 s to 6 s on the same data.

- `gam_refit` refits `bspatial(a, b)` and `a %X% b` tensor terms, with one
  REML smoothing parameter per margin (as mgcv's `te()`); `smoothing` reports
  the pair. The REML criterion takes the log pseudo-determinant of the
  weighted sum of a term's penalties.

- `gam_refit` refits `CoxPH` models: it alternates the Breslow baseline
  hazard with a Poisson penalized-least-squares step (the Poisson form of the
  Cox likelihood), with REML smoothing and pointwise intervals on the
  log-hazard scale. On a simulated proportional-hazards sample its effective
  degrees of freedom match mgcv's `cox.ph` (5.51 for the curved term in both)
  and its effect error equals mgcv's (0.083 over eight seeds).

- `logLik()` (and the information criteria built on it) is available for
  `GammaReg`: its registry entry said the shape was fixed at 1 and refused,
  but the shape is re-estimated at every step, as in mboost. The value is the
  Gamma log-density at the fitted mean and the shape `nuisance()` reports
  (checked against SciPy to 1e-8).

- `PropOdds(levels=[...])` takes the categories in order, lowest first, and
  a pandas ordered `Categorical` or polars `Enum` response supplies its own
  order; either is recoded to 1..K once at fit entry, so folds and refits that
  lack a class keep the same coding. A label column with no declared order is
  refused with that advice.

- Calibrating a smooth term's smoothing parameter from `df` is faster for
  large bases: the inverse Cholesky factor is now computed by a blocked
  triangular solve instead of an element-wise loop. A 576-column
  `bspatial` term's setup fell from 0.29 s to 0.17 s, which every
  cross-validation fold pays.

- `NBinomial` fits are about 5 times faster (a 2,000-row, six-term fit with
  cross-validation: 25 s to 5 s) and `GammaReg` fits about 15 times faster:
  the per-step dispersion or shape search still follows R's `optimize`, but
  each evaluation reads precomputed sums instead of three special functions
  per row. The chosen values are unchanged (R parity tests pass).

- `gam_refit` refits `NBinomial` models (with the dispersion the boosted fit
  estimated) and honours a per-row offset such as log exposure;
  `GamRefit.predict` takes `offset=` and, like `effects`,
  `unseen_levels="zero_effect"` to predict a group the refit never saw at the
  average group.

- `confint` no longer repeats `cvrisk`'s "still falling" warning for every
  bootstrap replicate whose inner cross-validation stops at the model's
  `mstop`. It counts them (`n_at_mstop_cap` in the result) and warns once
  when they are 10% of the replicates or more.

- Exact spline and block learners keep one copy of their design. A
  `BlockBols` learner held the design three times (as an ndarray, as a faer
  matrix and as `X'W`), the Gaussian fast path cloned every design again, and
  `df -> lambda` calibration and the fit's learner templates added two more.
  Peak memory for ten `bbs` terms fell from 2.12 GB to 0.48 GB at 200,000
  rows and from 6.66 GB to 2.4 GB at 1,000,000 rows, where the
  single-threaded fit took 4.3 s instead of 9.4 s. Fitting benchmarks are 10-53% faster; learner
  setup at 1,000 rows is 4-5% slower. Fitted values change only in the last
  digits (R parity unchanged).

- The pair terms the interaction screen proposes (and
  `autospec(interactions="adopt")` adds) are written with five inner knots
  per margin, `bbs(x, df=2.5, knots=5) %X% bbs(z, df=2.5, knots=5)`: an 8 x 8
  = 64-column tensor rather than the 24 x 24 = 576 columns of the default 20
  knots. On `benches/bench_insights.py`'s interaction task the fit took 0.22 s
  against 5.4 s, with excess error 0.0295 against 0.0314.

- A `%X%` tensor term is named by the term as written, such as
  `bols(z, intercept=False) %X% bbs(t, df=4)`, in the model repr,
  `base_learner_names`, `summary`, `varimp`, `selection_counts`, `coef`,
  `extract` and `partial_effects`, as a factor `by=` term already was. It
  used to appear as the derived column `__ix0`, which a reader could not map
  back to the formula. `varimp`'s `variable_names` and `selection_counts`'
  `column` list the two columns the tensor reads. `which=`, `parm=` and
  `learner_index` still accept `__ix0`. Refits (`update`, `confint`,
  `stabsel`, conformal) now name `%X%` and factor `by=` terms the same way as
  the fit they come from.

- The case-study collection now follows one teaching structure: each opening
  names the task and reusable package capabilities, the main fit-tune-interpret
  workflow precedes optional extensions, and advanced searches and benchmarks
  are labelled as such in the overview. Public case code cells contain one
  analytical step, stay within 35 lines, and substantial cells explain an
  analytical choice in a comment; `check_cells.py` enforces those mechanical
  limits. The contributor writing guide is tracked rather than left under an
  ignored working-memory directory.

- `llm_describe`'s default model is now `nvidia/nemotron-3-ultra-550b-a55b`,
  the largest Nemotron the default endpoint serves. The previous default,
  `nvidia/llama-3.1-nemotron-nano-8b-v1`, is retired: the
  endpoint answers requests for it with HTTP 410, so every call that did not
  pass `llm=` failed, and `cases/llm_explains_a_model` published that failure
  instead of an explanation. `llm=` still overrides it, and
  `GET {base_url}/models` lists what an endpoint serves.

- `llm_describe`'s prompt now asks the model to reply in Markdown — short bold
  labels or headings per effect, bullets, and small tables where they help,
  without a code fence — and `llm_describe`'s documentation says to render the
  reply with a Markdown renderer rather than printing it as plain text. The R
  mirror (`mboostrs_explain.R` and `man/llm_describe.Rd`) carries the same
  instruction. `cases/llm_explains_a_model` shows the reply rendered as
  Markdown, recorded again against the new prompt (artifact hash
  `d33c4e133f6855d9`); the live call is commented out in the notebook because it
  needs only the private `NVIDIA_API_KEY` to reach a paid endpoint, and the
  page instead replays the recorded artifact.

- `cases/agent_specification_search` is rewritten: it runs the search loop
  step by step on one simulated frame, grades it against the generating
  process, reports the twenty processes of `bench_agent_search.py`, and lists
  the library changes the study led to. `cases/cz_agent_search` is new: the same
  loop on the commuting-zone life-expectancy data, checked against the
  hand-built `cases/cz_inequality`. `cases/autospec_bodyfat` is rewritten
  around the Gaussian specification `autospec` now returns with a budget, and
  `usage/choosing_terms` documents the label reading above twelve values,
  `groups=` and the new hygiene findings.

- `autospec` infers `Gaussian` rather than `GammaReg` for a positive response
  whose standard deviation is below 10% of its mean, where a Gamma's variance
  cannot be told from constant variance and a boosted Gamma diverges. With a
  budget, the family probe reads the residual spread against a Gaussian mean,
  and when that spread points at `GammaReg` it compares `GammaReg` with
  `Gaussian` on held-out squared error, each at its own cross-validated step
  count, keeping `GammaReg` only if it is not worse by more than two paired
  standard errors. On `bodyfat` the spread points at a Gamma and the boosted
  Gamma's squared error is 26.8 against 10.6, so `autospec` now returns the
  Gaussian specification that the body-fat case study previously reached with
  a `family=` override. `GammaReg` multiplies its gradient by a shape parameter
  it re-estimates at every step, and a fixed step length overshoots as the fit
  improves.

- `autospec`'s priority list names columns that look like the response first,
  and summarises missing values as one line about the rows left out, when
  they are a tenth of the frame or more, instead of one line per column.

- `tune_mstop` treats a minimum in the last 2% of the grid as the boundary,
  and says when the risk is lowest at the first step: every step then makes
  held-out predictions worse.

- On folds that hold a group term's groups out, `term_report` reports the term
  with `measurable=False`: every held-out row belongs to a group the fit never
  saw, so the term's gain is zero by construction. `suggest_specifications`
  never proposes dropping such a term.

- `autospec` probes a frame with missing values on its complete rows,
  `rec.fit` defaults to `na_action="omit"`, and the report counts the rows left
  out. The scan's missing-value finding says so.

- Book headings are labels naming the method, function, argument or data set
  a section covers. 391 headings that were questions, clauses opening with
  What/Where/When/Why, or sentences stating a result were renamed, for example
  "The split the probes share" to "Cross-validation folds: `folds=` and
  `seed=`". The one link to a renamed anchor and the tests that asserted old
  heading text were updated.

- `docs/book/check_writing.py` is removed and `check_docs.py` no longer matches
  phrases in prose. `check_docs.py` took over the two structural checks it
  lacked: citation keys that `references.bib` does not define, and Altair charts
  without a title. The typed author-year citation check was dropped with the
  phrase rules, and CI no longer runs `check_writing.py`.

- REML refits (`gam_refit`, `gamlss_refit`, and R's `gam_refit`) of a model
  with a pair term and a main effect nested in it, such as `bbs(x1) +
  bbs(x1) %X% bbs(x2)`, were not identified: cubic B-splines on any knots
  reproduce cubic polynomials, so the tensor basis repeats the main effect's
  `x1`, `x1^2` and `x1^3` columns. The Gaussian refit returned an EDF of
  -9.8 for `bbs(x1)` and did not converge; the distributional refit failed
  to evaluate its criterion in 12 of 15 fits of the distributional study's
  `autospec` pairs route. The model builder in the Rust core now applies
  mgcv's `gam.side` rule: each refit term carries the data columns it reads
  (`RefitTerm::variables`, filled by both bindings), a term is compared only
  with the terms whose columns it contains, and a pivoted Cholesky
  factorization of its residual cross-product removes one column per
  direction those terms already span. Columns dependent within a term (a
  spline column with no data in its support) stay, as in mgcv, so fits
  without nested terms are unchanged; the bodyfat refit still matches mgcv.
  The result reports the coefficients at full width with zeros in the
  removed positions, so prediction and serialization are unchanged.

- `gamboostLSS` models named a `%X%` pair by its derived design (`__ix0`)
  in `base_learner_names`, `selected_which`, `coef`, `path_frame` and
  `partial_effects`, and `deselect` reported the pair as `buser(__ix0,
  X=__ix0_X, K=__ix0_K, ...)` in `kept`, `dropped`, `shares` and `formula`,
  with `variables` listing `__ix0_X` and `__ix0_K` as data columns. They now
  use the term as written, as `gamboost` does; the derived name is still
  accepted where a learner is looked up.

- The LSS selection-frequency chart (`plot_lss_selected`) prepended an
  intercept to every `gamboostLSS` parameter's learners, so each bar carried
  its neighbour's label.

- A regression tree fit allocated each child's sorted row lists at every
  node (163 allocations per boosting step in the allocation test). The row
  buffers are now allocated once per tree and partitioned in place (9).

- The insight benchmark page computed its "top-5 stability" column per row
  instead of per data set and method: under Polars 1.41 `map_elements` in a
  group aggregation runs once per element, so the Jaccard overlap compared
  the characters of feature names. It is now the mean Jaccard overlap of
  the top-five feature sets between pairs of splits.

- `deselect` on a distributional (`gamboostLSS`) fit with a `%X%` pair term
  returned a model that could not predict new rows ("new-data prediction for
  Buser learner '__ix0' requires buser_newdata"): the reduced model was built
  from the rewritten formulas without the tensor recipes. They are now
  carried over, and `gamlss_refit` maps the derived `buser(__ix0, ...)` term
  back to the `%X%` pair it stands for, which it can refit.

- `gam_refit` and `gamlss_refit` refused a one-column term without an
  intercept, such as the varying slope `bols(x1, by=x2, intercept=False)`,
  with "has one column, which the sum-to-zero constraint removes". A single
  non-constant column is not confounded with the intercept, so it now takes
  no sum-to-zero constraint and is estimated; a constant column is still
  refused.

- `autospec` never measured the family of a non-negative response with zeros
  that is not a count (an occupation count scaled to a rate, a spend): it
  kept Gaussian from the dtype. It now also fits quasi-Poisson (a log link,
  variance proportional to the mean) and keeps it only when its held-out
  squared error is lower by 2% and two paired standard errors. On the socmob
  data, inner-validation R^2 rose from 0.698 to 0.872. A candidate family
  whose fixed-step fit diverges in that comparison is refitted with the line
  search before it counts as a loss.

- `autospec` fitted a Gamma, Poisson or negative binomial response with the
  fixed step unless that step overflowed. On a response in the hundreds of
  thousands (Miami house prices) the fixed-step Gamma fit stopped after 23
  steps; a line search went on to 8,773 and raised inner-validation R^2 from
  0.859 to 0.889. For these families `autospec` now also runs the search
  with `step="line_search"` and keeps the recommendation with the lower
  cross-validated risk on the same folds; the report says which. An explicit
  `step=` is honoured as before.

- `varimp` for a Multinomial model credited each step with its change in the
  in-bag risk, mboost's loss, which leaves the reference class out and can
  rise on a step that lowers the full negative log-likelihood. True
  predictors received negative shares (x3 at -12% on one seed of the classes
  benchmark), and `autospec`'s deselection dropped them. Each step is now
  credited with its change in the full likelihood, replayed in Rust along the
  path, in Python and R alike (agreement to 1e-6); every share is
  non-negative.

- R `varimp` on a model with two learners of one name (`bols(x) + bbs(x)`)
  failed with "factor level is duplicated"; the learners' shares are summed
  under the name, as in Python.

- `autospec` warned that a declared `Multinomial` or `PropOdds` response had
  "values exactly zero" suited to a count family and "few distinct values in
  a float response"; these warnings are suppressed for every family that
  models class labels, not only an inferred `Binomial`.

- `bbs(x, center=TRUE)` was ignored in a Multinomial model: the per-class
  Kronecker term always used the raw basis, so a "centred" smooth fitted the
  same model as an uncentred one. The term is now centred in both bindings:
  it is fitted on the deviation from the penalty's null space (the
  difference-matrix transform `D'(DD')^-1`, as in mboost), and prediction
  replays the transform. Python and R predictions agree to 2e-13.

- R `predict()` on new data for a Multinomial model with a `bols(x)` term
  laid the intercept-plus-slope design out column by column where the engine
  reads it row by row, so predicted probabilities differed from the fitted
  ones by up to 0.37. The design is now built row by row.

- A pair adopted by `autospec` over a column fitted on the log scale was built
  on the raw column, so a new row far beyond the training range (average
  occupancy 600 where training reached 64) extrapolated the tensor to an
  absurd value (test R^2 0.01 on one California split, 0.75 on the others).
  `%X%` now accepts `bbs(log(x))` and `bbs(sqrt(x))` margins (new values
  below the training minimum held at it), `gam_refit` refits them, and
  adopted pairs use the scale of their columns' main terms.

- `gam_refit` failed on a `bols` term over a string column (`could not convert
  string to float`); it now uses the treatment-coded level design mboost's
  `bols(f)` fits. Five of the 24 real-data sets had failed this way.

- `autospec(interactions="adopt")` proposed `brandom(g, by=b)` for a group
  and a two-level label column, which the fit refuses; it now proposes
  `brandom(g) %X% brandom(b)`.

- The line-search step rule could not take a step below 0.125 of the
  least-squares direction: when every point of its grid overflowed the link
  (Poisson on counts near 1e5 wants about 1e-5), it returned an overflowing
  step and the fit stopped as non-finite. The search now halves its lower end
  until the loss falls, and never returns a section point worse than its
  best grid point.

- The line-search step rule multiplied a tree learner's whole coefficient
  vector by the step scale, although that vector encodes the tree's node
  count and splits, so the stored tree decoded to no nodes and prediction
  panicked. Only the leaf means are scaled now (standard and LSS engines).

- `autospec`'s interaction screen required a pair to remove 0.01 of mean
  squared error in the response's units, so on a response measured in small
  units no pair was ever reported; the floor is now 1% of the residual's
  variance.

- A `%X%` term over string factors could not predict new rows: prediction
  encoded the labels before rebuilding the tensor design, which the fit had
  built from the raw labels, so every level read as unseen. Prediction now
  replays the fit's order, and a per-term prediction of the training rows
  uses the fitted designs.

- `predict(newdata, which=)` applied the transform recipe twice, so a
  per-learner prediction of a `brandom` term over a string column refused
  every level.

- `autospec` read any factor with 3 to 12 levels as a grouping unit and held
  whole levels out of each fold, which measures prediction for unseen
  categories; a factor needs at least 10 levels to set the fold design
  (declare `groups=` otherwise), and deselection is skipped when the tuned
  step count is below 10.

- A `log(x)` or `sqrt(x)` term refused to predict a new value below the
  training range (`log(0)`); the value is held at the training minimum, with
  an `ExtrapolationWarning`.

- `autospec` stopped with "risk became non-finite" on a Poisson, negative
  binomial or gamma response in the hundreds; it now retries its probes with
  `step="line_search"` and the recommendation carries that step rule
  (`autospec(step=)`, `tune_mstop(step=)`).

- `autospec`'s shape probe switched learners only for a gain above 2% of the
  held-out risk, which misses a clearly better shape when noise dominates the
  risk; it also switches for a gain of two paired fold standard errors when
  the column carries at least 1% of the tuned fit's risk reduction (on noise
  columns that rule passes by chance).

- `cvrisk` (gamboost, glmboost) and `stabsel` dropped a per-row offset:
  every fold started from a fitted constant instead of the known offset, so
  `tune_mstop`, `autospec` and stability selection on an exposure model chose
  against the wrong risk (a Gaussian check with offset sd 5: held-out risk
  25.8 instead of 0.22). The CV path now matches mboost to 1e-6.

- `autospec` compared families and shapes at 100 boosting steps, before any
  candidate had fitted the signal, and stopped its `mstop` search after three
  doublings. It now tunes the depth first, compares shapes one column at a time
  with the rest of the model held fixed at that depth, and lets the search run
  to 10,000 steps. On an eight-shape benchmark its signal RMSE went from 0.413
  to 0.351 (equal to `tune_mstop("y ~ .")`).

- `gam_refit` refused a model fitted from `y ~ .`; a bare numeric name is
  refitted as the `bbs` gamboost fitted (the `parse_gamboost_formula`
  docstring said a bare name becomes `bols`; it becomes `bbs`, as in R).

- R `predict(newdata =)` was wrong for a block-form learner (`bols(x, lambda
  =)`, a formula mixing `bols` forms, `brandom`): the design was passed
  column-major to a reader expecting row-major (error 1.14 on the link scale
  in the regression test).

- `PropOdds` with four or more levels did not fit: its thresholds were
  searched one at a time and could cross, after which a middle class's
  probability was negative and the first threshold grew without bound (test
  log-loss 5.9 where 1.28 is attainable). The thresholds are now parameterized
  as R's `d2s` does, which keeps them increasing, and start from R's values; a
  four-level linear fit matches mboost's thresholds and coefficients to 3e-4.
  The threshold search is also about 7 times faster per step (each row reads
  only its two neighbouring thresholds from precomputed exponentials).

- `cvrisk` for `Multinomial` minimized mboost's loss, which leaves the
  reference class out of the negative log-likelihood, and stopped too early
  (257 steps where held-out log-loss was lowest at about 1,000). Held-out rows
  are now scored with the full negative log-likelihood (`Family::held_out_risk`);
  the in-bag risk path is unchanged and still matches mboost. This is a
  deliberate deviation from mboost's `cvrisk`.

- `tune_mstop` stopped doubling its grid after three extensions (800 steps
  from the default start), without a warning, although `max_mstop` is 10,000;
  a model with a 50-level `brandom` term needs about 4,000. It now doubles
  until the risk turns or `max_mstop` is reached (`extensions=None`) and warns
  when the search ends with the risk still falling.

- `btree(x, by=z)` grew its tree on every row and multiplied the prediction by
  `z`; mboost grows it on the rows where `z` is 1 (`weights * by`). The effect
  of a binary `z` now matches mboost to three decimals (it was -0.49 and 0.50
  where mboost gives -0.22 and 0.75).

- `nuisance()` returned None for `NBinomial` and `GammaReg` fits and did not
  follow the step count; it now returns the dispersion or shape at the model's
  `mstop`, as R's `nuisance()` does. Slicing a model (`fit[:k]`) dropped the
  nuisance path, so a sliced `PropOdds` model returned link values for
  `type="response"`.

- `conformal_interval(method="cqr")` raised "every conformal lower endpoint
  must be <= its upper endpoint" on skewed or heavy-tailed noise (5 of 24
  benchmark runs): a negative calibration correction inverted the band where
  the two quantile fits were close. Crossed fits are ordered and an inverted
  band is returned as its midpoint. Its quantile fits also reused the Gaussian
  model's step count; each quantile's step count is now cross-validated on the
  training half, which lowered the interval score by 2-3%.

- The PDF book did not build since 13 September: a 549 x 3,390-pixel faceted
  chart set only a width, and pdfLaTeX shipped out pages until it ran out of
  link destinations at page 498,589. Every figure now also gets a height limit
  of 85% of the text height (`patch_tex.cap_figure_heights`).

- `gam_refit`'s REML search could stop at a smoothing parameter where the
  penalized Gram matrix was singular, whose log-determinant `slogdet` reports
  as minus infinity. On the body-fat data the refit gave hipcirc 1.91
  effective degrees of freedom against mgcv's 1.75; it now gives 1.71.

- `LSSModel.cvrisk()` on a `method="noncyclic"` fit refit every fold as a
  cyclic fit and tuned per-parameter budgets, so its optimum could not be
  passed back to `with_mstop`. It now cross-validates the total step count:
  each fold is fitted once to the largest total and every smaller total is
  read off that path; the grid points and `optimal_mstop` are one-element
  lists, which `with_mstop` reads as the total.

- `autospec`'s interaction screen formed its residual on the link scale
  (`observed - fitted_`), so for Binomial, Poisson and NBinomial fits it
  reported pair "leads" that were curvature of the link; it now uses the
  response scale. The polio and respiratory-infection case pages, whose
  leads came from this, were re-executed and rewritten.

- `predict(newdata=, which=)` for a learner the fit never selected returned
  an array sized by the number of columns of a dict `newdata`.

- `na_action="omit"` failed on a pandas frame (its `.filter` selects
  columns, and the row mask took the Polars branch).

- `confint(newdata=)` failed on a pandas frame.

- `cvrisk_gamboostLSS` and `cvrisk_glmboostLSS` default `grid` to the
  diagonal grid `LSSModel.cvrisk()` uses instead of refusing the call.

- Partial-effect plots label the y-axis with the link scale ("half log-odds"
  for Binomial), and `partial_effects` says so: mboost's Binomial predictor
  is half the log-odds.

- Removed the parity note saying `extract('lambda')` returns None for block
  learners. Since the calibrated lambda is carried through the learner
  boundary, `extract('lambda')` returns it for `bbs`, `bns`, `bspatial`,
  `brandom`, `bmrf`, `bkernel`, `buser` and `%X%` terms, and every model
  summary had been repeating the stale note.

- A response column of text labels is refused with the column named, a sample
  of its values, and the two ways out, on every route: `gamboost`,
  `glmboost`, `gamboostLSS`, `glmboostLSS` and the specification assistants.
  The previous error was numpy's `could not convert string to float:
  'high'`, which named one value and neither the column nor the fix.
  `Multinomial` still encodes text class labels itself; the message says so.
  The same replacement covers a non-numeric column reaching the numeric
  extraction path anywhere.

- Static HTML now renders Altair charts shown with `display(chart)` as SVGs.
  Those outputs have notebook type `display_data`; the postprocessor handled
  only `execute_result`, leaving the `alt.LayerChart(...)` plain-text fallback
  or a permanent loading state on some pages. Both forms now inline referenced
  SVG content and discard the competing plain representation.

- `plot_partial_effects` now draws partial residuals as documented. Its point
  layer previously repeated each observation's fitted term contribution, so
  every point lay on the fitted line and the apparent residual diagnostic
  contained no residuals. Points now add the family's working residual to the
  term contribution; the fitted line and predictor-density rug are unchanged.

- Distributional fits with per-parameter learning rates now replay the fitted
  path at those same rates. Previously, `predict(newdata=...)`, `centiles`,
  `coef`, partial effects and path-frame scaled coefficient norms used the
  scalar rate in `boost_control`; a model fitted with, for example,
  `nu={"mu": 1.0, "sigma": 0.01}` could therefore return new-data centiles
  that disagreed with its fitted centiles. Additive learner stores retain the
  resolved rate for their parameter, linear replay and reports use that rate,
  and pickle/JSON loading preserves it when rebuilding stores. Tests cover
  linear and additive, cyclic and noncyclic models, including serialization.

- `autospec(..., nu=0.03, mstop=15).fit(data)` fitted with the default rate
  0.1. The recommendation rebuilt its control from the step count alone, so
  the replayed fit differed from the explicit rate-0.03 fit — by up to 0.94
  on the 80-row Gaussian probe that found this. The recommendation now
  carries the control it was asked for, the step count it settled on with the
  caller's step length; `rec.fit()` replays it, an explicit `control=`
  overrides it, and `rec.report()` names a non-default step length. Tested at
  budget zero and with probes running.

- `munich_rent`'s docstring said rent was in DM. The upstream
  `gamlss.data::rent99` documentation states euros; the docstring and the
  dataset's data card now say euros, and the card records the units of every
  column. No numeric data changed.

- `term_report` said a group term's gain on folds that hold its groups out was
  "zero by construction", and `suggest_specifications` never offered to remove
  one. The term's own prediction for a held-out group is the average group, but
  removing it refits every other term, which does change the error: on the
  commuting-zone frame of `cases/cz_agent_search`, removing the state term
  raises out-of-fold error by 0.054 (se 0.021). The report now prints that
  measured change with a note on what it means, and a group term whose removal
  lowers the error by more than the noise threshold is offered for removal; one
  the folds cannot separate from zero is still never offered.

- The paired standard errors of `compare_specifications`, `term_report` and
  `autospec`'s family probe treated every fold of `repeated_kfold_folds` as a
  separate draw. Folds of different repeats reuse the same rows, so the
  standard error shrank with the number of repeats: on simulated data with six
  repeats of 5-fold it was 0.0064 where the change varied by 0.028 across data
  sets, and a two-standard-error rule accepted changes the data could not
  support. The variance is now computed within each partition and averaged.

- `autospec`'s interaction lead for an integer-coded group column was
  `bbs(x, by=g)`, which multiplies the smooth by the code's numeric value. It
  is now the random slope `brandom(g, by=x)`: on 30 groups with random slopes,
  held-out squared error 0.39 against 1.89. A two-valued numeric column was
  paired through `%X%`, a degenerate tensor, and is now `bbs(x, by=b)`.

- `stabsel` on a model with a `%X%` interaction reported a selection frequency
  of zero for every learner. Each fold refit subset the data rows while the
  interaction's tensor design kept every row, every refit failed, and the
  failures were only warned about. A fold whose row subset would misalign
  something built against all training rows (a tensor design, a user design
  matrix, a neighbourhood graph) now refits on the full data with the
  out-of-bag rows at zero weight, which is how `stabs::stabsel` refits an
  `mboost` model, and `stabsel` raises when every refit fails.

- A model sliced with `model[:k]` lost its interaction recipe, so
  `predict(newdata=)` on a sliced `%X%` model raised, and so did
  `cv_risk(..., mstop="cv")`, which predicts through a sliced model. Slicing
  now carries the same attributes as `continue_to`.

- `partial_effects` on a model with a `%X%` interaction raised whenever it
  evaluated anywhere but the training rows: on a grid (`n_points=`, which
  `plot_partial_effects` uses) and on `newdata=`, the tensor term was given its
  training design beside rows of another length. On `newdata=` it also skipped
  the label encoding `predict` applies, so a `brandom` term on text labels
  raised, and `predict_over_mstop(newdata=)` raised the same way for such a
  term. Both now prepare new rows as `predict` does, and partial effects on
  new rows sum to the prediction.

- `grouped_folds` raised `AttributeError` on an object array of Python strings,
  which is what a polars string column's `to_numpy()` returns.

- `heteroscedasticity` and `sensitivity` read a spatial learner's name,
  `"lon,lat"`, as a column name; they now read both coordinates, and every
  column of a tree learner.

- `gamboost(..., learners=..., na_action="omit")` ignored `na_action` and
  refused the first missing value. Rows are now dropped over every column the
  formula and the learners read; a drop that would misalign a supplied `Buser`
  design matrix is refused with a message.

- `autospec` with a fit budget failed in its first probe on a frame with a
  missing value.

- `compare_specifications`, `evaluate_specification` and `cv_risk` could not
  score a specification containing `brandom` on grouped folds. Every held-out
  group is a level its fold's fit never saw, and the prediction step raised.
  Those rows are now predicted as an average group; with
  `unseen_levels="zero_effect"` the grouped `cv_risk` score of a
  `bols(x) + brandom(g)` specification equals `model.cvrisk` on the same folds
  to a relative 1e-9.

- `partial_effects` and `predict(which=...)` left out the step length ν for
  `buser` learners, which include `%X%` tensor terms, and for exact `bkernel`
  learners. Those per-learner contributions were 1/ν times too large, 10 times
  at the default ν = 0.1, and a model's contributions did not add up to its
  prediction. They now add up to 1e-8. Partial effects of a `%X%` term also
  evaluate on new data, and those of a plain `buser` term refuse new data
  because its design cannot be rebuilt from the columns.

- The documented contract for `continue_to` described an implementation the
  package no longer has. The engine's warm continuation is wired through to
  `gamboost`: a fit whose family calibrates as a pure function of `(y, f, w)`
  keeps the gradient it stopped on and resumes from it, matching a cold fit
  bitwise on risk, fitted values and selection. The docstrings, the global
  parity note and the cross-validation boundary warning all still said the
  method refits from iteration zero. They now name the routes that resume and
  the ones that do not: the Gaussian Gram route keeps per-learner projections
  rather than a length-n residual, and `PropOdds`, `Multinomial`, the survival
  families, binned learners, `glmboost` and the distributional models refit.

- R's `cvrisk()` for `gamboost` models computed a different quantity from the
  one it reported. It refit each fold on a *row subset* rather than on the
  fold's in-bag weights, which moves knots and centering onto the subsample;
  it scored held-out rows with a per-family loss table written in R instead of
  the fitted family's risk, where squared error stood in for every survival
  loss and for Multinomial compared class codes with link predictions; it
  averaged per-row losses instead of dividing by the out-of-bag weight total;
  and its grid started at mstop 1 rather than the offset state. Each fold now
  refits on its in-bag weights and reads the engine's out-of-bag risk path, so
  the result matches `mboost::cvrisk` on shared folds (7.2e-13 for a Gaussian
  `bbs + bols` fit, 3.1e-12 for Poisson). The returned matrix has `mstop + 1`
  columns. A model fitted with non-constant weights now refuses rather than
  returning a number: the engine's out-of-bag risk weights every held-out row
  by one.

- A distributional model rebuilt by `update()` — which is what `model[:k]`
  does, since a cyclic path is not prefix-consistent — dropped the fit's
  stabilization and its per-parameter learning rates. Slicing a fit made with
  `stabilization="MAD"` returned exactly the unstabilized fit, differing from a
  direct MAD fit at the same mstop by 0.52 on the link scale, and
  `nu={"mu": 0.1, "sigma": 0.02}` became the control's scalar rate. Both
  settings now travel with the refit, as `cvrisk` and `stabsel` already did.

- A weighted Multinomial fit charged each observation another observation's
  weight. The response is stored flat, `K-1` entries per observation, with
  that observation's weight repeated over its block; the reported risk read
  those weights by observation index, so observation `i` was charged the
  weight of observation `i / (K-1)`. An integer-weighted three-class fit
  reported risk 98.4402 where the weighted likelihood is 88.3595, and the
  iris example reported 220.2557 against `mboost`'s 236.4396. Cross-validation
  and stability selection read each fold's in-bag weight column the same way,
  so every fold trained on a shifted subset and agreed with a correct fit only
  at the offset step. The fitted coefficient path was already right: the
  Multinomial gradient does not read weights, and the learners already
  received one weight per observation. The native family-diagnostic endpoint
  now states that contract — one weight per observation, expanded internally
  for the flat Multinomial layout — instead of requiring a length it then
  misread.

- Binary response encoding now belongs to the Rust `Family` contract instead
  of three PyO3 fit entry points. `AdaExp` and `HingeLoss` consequently accept
  both `{0, 1}` and `{-1, +1}` in direct Rust and frontend fits, while
  rejecting intermediate labels; Binomial no longer admits negative values
  other than the `-1` class.

- The native standard- and distributional-family diagnostic endpoints now
  validate response/eta/weight shapes, finite values, and weight domains and
  accept strided NumPy views. Malformed direct calls return `ValueError`
  instead of reaching contiguous-slice or length assertions in numerical code.

- Invalid Dirichlet parameter counts at Python and R native boundaries now use
  the checked `DirichletLSS::try_new` constructor and return a typed error
  instead of triggering the compatibility constructor's assertion.

- The native raw B-spline replay helper now accepts strided covariate arrays
  and validates its data, knot count, and boundary interval before basis
  construction, preventing prediction-time panics on sliced new data.

- A learner specified by `df` no longer fits nothing when an outlying value
  leaves a knot interval empty and the data are large or heavily weighted.
  The empty B-spline column makes `X'WX` singular in one direction, which puts
  a value near `1e9` into the penalty spectrum that `df` is calibrated from.
  Beside it, the dense SVD rounded the small penalized values to zero, so the
  df curve appeared to level off at the target, and the calibration returned
  the lambda cap. The term was then only the penalty's null space, a straight
  line for `bbs`, with no warning. On 200,000 N(0, 1) rows with a minimum of
  −5.35, `bbs(x, df = 4)` calibrated lambda to 1e14 where R's `df2lambda`
  gives 96,531. It now gives 96,531.34, and the fitted risk equals R's. Binned
  fits (`bins=`) failed from about 90,000 rows. When the spectrum is resolved
  that coarsely, the calibrated lambda is now checked against the same
  spectrum computed from the pencil `(K, X'WX + (1 + eps)·K)`, whose values
  lie in `[0, 1)`. It is recomputed from that spectrum only when its df is
  off. Well-resolved spectra are not re-derived, so results that were correct
  before are unchanged. Every learner calibrated by `df` goes through this
  function.

  The same precision loss also caused a milder error with no empty column
  involved. On one 16,000-row N(0, 1) column, `bbs(df = 4)` was calibrated to
  lambda 13,659, where R gives 12,431.804. That lambda is 10% too large and
  gives df 3.92 on the accurate spectrum. It now gives 12,431.804. Binned
  learners are affected the same way, because bin counts act as weights. The
  n = 16,000 rows of `benches/binned_selection_grid.json` changed accordingly
  and were regenerated.

- `autospec` no longer sends numeric columns with three to five integer levels
  to `bols`. That learner fits one slope across numeric codes, not separate
  group means; all numeric 3–12-level columns now use `brandom`, matching the
  corrected `classify_columns` recommendation. Declared ordered categoricals
  still use the directly valid label-based group learner, but now raise
  `OrderedCategoryWarning` explaining that the order was not modeled and how
  to supply explicit scores for linear, smooth, or monotone alternatives.

- A `buser()` design whose columns form an interaction lattice no longer
  aborts when one of its groups holds no observations. `center = FALSE` (the
  default) routes the design through the Kronecker-sum factorization, which
  reads the group count off the column layout; the per-group solvers were
  counted from the data instead, so an unoccupied residue class left the
  solver vector short and the fit raised a Rust panic through the binding
  rather than fitting. The unoccupied group's Gram is now the shared penalty
  block, which contributes a zero effect, matching the materialised learner
  it replaces bit for bit — before and after a change of weights.

- `deviance_profile` bins the same rows together in Python and in R. Both now
  give the first `n %% bins` intervals one observation more than the rest.
  R used `cut(seq_len(n), breaks = bins)`, which spreads the remainder over
  different intervals, so the two frontends reported different bin
  membership, and different per-bin summaries, for the same model whenever
  `n` was not a multiple of `bins`.

- `BCPELSS` scores stay finite when a response falls exactly on its fitted
  `mu`. The kernel factor `|z/c|^(tau-2)` is infinite there for `tau < 2` and
  `ln|z/c|` is infinite for every `tau`, each against a factor of zero, so the
  mu, nu and tau gradients returned NaN. Every one of those products tends to
  zero as `z` approaches the origin, and the limit is now used. The mu offset
  is the weighted mean of the response, so an integer-valued response can put
  an observation exactly on it at the first iteration.

- The bmono constraint solvers no longer allocate per candidate solve.
  `cho_solve_inplace` documented itself as factoring into the caller's buffer
  and allocating nothing, but passed the matrix to faer by shared reference,
  which reads it and allocates a fresh factor. It now calls faer's in-place
  LLᵀ with caller-owned working memory, sized once when the learner is
  prepared. Measured steady-state allocations per candidate solve fall from
  about ten to zero for QuadProg and to the cross-thread noise floor for the
  iterative solver. Results are unchanged.

- Conformal results report method-specific guarantees, bounds, training/fold
  counts and assumptions. Split/CQR report their nominal marginal bound; balanced
  CV+ reports its generally lower finite-sample bound. CV+ is no longer described
  as constant-width. Endpoint arrays are validated, owned and read-only, and
  weighted fits now receive a precise sampling-assumption warning.

- Refit specifications distinguish fitted-design replay from training a new
  preprocessing pipeline. Conformal splits now learn scaling, centering,
  encoding and formula interaction state from proper training rows. Fitted
  prediction recipes remain unchanged; non-rebuildable external matrices are
  still refused with their row-alignment reason.

- Corrected book claims about cross-validation, stability selection, shape
  constraints and survival. Test risk need not be U-shaped, bootstrap OOB
  weights are explicit rather than a subtraction, selection frequency is not
  causal truth or monotone effect size, nested training classes do not order
  held-out error, and `bbs(observed_time, by=x)` is not a time-dependent Cox
  effect. The latter now points to required start-stop risk-set support.

- `build_info()` in Python and `mboostrs_build_info()` in R report identity
  embedded in the loaded native library: Rust/Cargo source digest, Git revision
  and dirty state at build time, compiler, target, target and crate features,
  profile, ABI schema, and a composite build ID. Package version remains a
  separate release identifier. New benchmark metadata records this native
  identity, so a source checkout or ambient `RUSTFLAGS` can no longer be
  mistaken for evidence about an already-built extension.

- Distribution-profile tables and plots for LSS models. `density_overlay_data`
  averages row-specific fitted densities or masses instead of evaluating a
  distribution at average parameters; zero-adjusted mixtures show their atom
  separately from the continuous density. `centile_fan_data` retains row and
  optional split identity for nested predictive bands, while
  `deviance_profile` reconstructs joint deviance in equal-count covariate bins
  without presenting it as a parameter-specific attribution. Corresponding
  Python model methods and R S3/base-graphics methods render the three tables.
  The existing worm plot remains the detrended transformed QQ view, and
  `plot_lss` remains the parameter-term profile. R source tarballs now retain
  the maintained R tests so `R CMD check` runs the published contracts.

- `BCTLSS` and `BCPELSS` in the Rust engine and both Python and R frontends.
  The positive-response families fit location, scale, Box-Cox power, and tail
  shape with the links and starting values used by `gamboostLSS`. Density,
  CDF, quantile, inverse-CDF simulation, deviance, PIT, centile, and prediction-
  interval workflows are available. Distribution functions match
  `gamlss.dist`; complete fitted paths match `gamboostLSS`, including BCT's
  upstream forward-difference score for the tail parameter. The response mean
  helper refuses both families because their `mu` parameter is a location, not
  generally the truncated distribution's mean.

- The `bkernel` learner gained a Nyström/landmark alternative. A `landmarks`
  argument (formula `bkernel(x, landmarks = 30)` or `Bkernel("x",
  landmarks=30)`) builds the kernel ridge against `m` evenly spaced quantile
  landmarks instead of the full n×n matrix: an `n×m` design via `K_nm L^-T`
  with `K_mm = L L'`, dropping kernel-ridge memory from O(n²) to O(n·m) and
  exact at `m` below the number of distinct `x`. Both frontends reach it —
  Python `gamboost`/`gamboostLSS` (fit, predict, partial effects, `cvrisk`,
  pickle) and R `gamboostLSS` (replay + partial effects). Restricted to the
  Gaussian kernel: compact-support kernels make the landmark matrix singular.

- The capability ledger now carries an auto-derived `node_capabilities` list: one
  flat record per `frontend × family/learner/method × operation` cross, with the
  operation status, the validation ids that back it, and
  `model_class`, `family`, `learner`, `operation`, `weight_kind`, `oracle_version`
  and `last_success_artifact` fields. The generator derives it from the existing
  family/learner/method tables, so it cannot drift from the statuses that row
  matrix advertises; the validator refuses unknown validation ids and invalid
  weight kinds. (Per-validation `weight_kind` and `oracle_version` attribution

- The Gehan family now computes its gradient and risk via an exact `O(n log n)`
  ranked form (single sort + prefix sums) above n = 64, matching the AUC ranked
  kernels. Results are unchanged to arithmetic rounding; R parity preserved.
  Measured ~41x faster at n = 400 and ~3700x at n = 12,000 on the gradient step
  (`bench_ranking`, correctness-gated).
  remains a person-scale pass over each fixture; those fields are `"unknown"` /
  `""` pending it.)

- LSS gradient stabilization now survives refits: a fit-level `stabilization=`
  override is stored on the model and propagated through `cvrisk`/`stabsel`
  refits (the `cvrisk_*LSS`/`stabsel_*LSS` functions gained a `stabilization=`
  parameter), so a distributional model does not silently drop its
  stabilization when cross-validated or stably selected.

- `continue_to` warm-continues standard-models from the retained fitted state
  where the engine keeps a residual (the general route and packed-scalar
  Gaussian), resuming via a new `fit_gamboost_continue_specs` endpoint rather
  than refitting from zero; the result equals a cold fit bit-for-bit. The
  Gaussian Gram route (block/bbs Gaussian) still cold-refits because it keeps
  per-learner projections rather than a length-n residual, and Multinomial/
  PropOdds/survival/binned fits do too.

- Multinomial `cvrisk`, stability selection, and fitting now run through the
  compact learner-spec path with the Kronecker lift, matching R's out-of-bag
  Negative Multinomial Likelihood at the observation-level normalisation. The
  fit path gained a `KroneckerBbs` blueprint (`bbs(x) %O% bols(class)` raw-basis
  + `I+K1` recipe); the legacy ~90-argument array boundary is now unreachable
  and scheduled for removal. Per-parameter LSS
  learning rates are exposed as the fit-level `nu=` argument on the Python
  `glmboostLSS`/`gamboostLSS` fit, `cvrisk`, and `stabsel` entry points,
  following gamboostLSS's `boost_control(nu = c(...))` contract: a scalar
  recycles, a list is family order, and a `{param: value}` dict names every
  parameter, validated in the engine.

- `CausalSpec` records a binary-treatment intervention, estimand, target
  population, timing, baseline adjustment set and identification assumptions.
  `aipw` estimates ATE or ATT with cross-fitted boosting nuisance models and
  optional inner tuning, returning reconstructible row scores, influence
  values, fold identities and overlap diagnostics. The first supported
  inferential scope is numeric baseline covariates, continuous outcomes and
  IID standard errors; unsupported causal designs fail explicitly.
  Results include standardized balance before/after weighting. Grouped
  cross-fitting returns no interval until a cluster-robust variance is implemented.

- `confounding_sensitivity` evaluates an explicit additive omitted-variable
  bias grid in user-specified outcome-effect and imbalance units. It shifts
  estimates and intervals without presenting the grid as an identified or
  calibrated bound. Causal specifications require time zero, follow-up, and
  an adjustment rationale.

- Zero-adjusted Gamma density and CDF are now zero below the response support.
  PIT uses the left CDF limit and randomizes the atom at zero while leaving
  continuous observations unchanged. R applies the same rule and now handles
  the zero-mass left limit for every supported lattice family.

- Distributional mean calibration uses the family's response mean. In
  particular, zero-adjusted Gamma uses `(1-nu)*mu` and Gumbel includes its
  scale-dependent mean shift; undefined means are refused. Logistic density
  evaluation is stable in both extreme tails, and quantile probabilities must
  be finite.

- Information-criterion results identify their definition, normalization and
  conditional hat-trace df. Invalid family/method combinations and ignored
  penalties are rejected. Gaussian comparison tables use `mboost_corrected`
  instead of the misleading `GAIC(2)` column; LSS tables label their update-count
  penalty as `heuristic_GAIC(k)`. Unlike criteria cannot be ranked together.
  Table metrics now refer to the same selected iteration, and likelihood
  profiles call `logLik()` instead of incorrectly reversing corrected AIC.

- Comparison tables check retained responses, effective weights and shared
  data columns. R fits retain the risk kind for criterion validation. LSS
  likelihood metadata no longer treats update counts as effective df.

- AIC plots use one-based iterations, aligning the curve with the optimum
  marker, and show the criterion's name and normalization.

- Interval coverage rejects mismatched shapes, non-finite observations and
  invalid bounds while retaining unbounded intervals. Censoring weights
  reject non-vector/empty inputs and non-positive or non-finite clipping limits.

- CV comparisons at selected optima are now always descriptive, including
  matching optima. Approximate intervals require an explicit stopping point
  and recorded matching fold identities. Default comparisons respect the
  reported optimum instead of accidentally selecting the offset-only model.

- Extrapolation `error` and `clip` policies now act on every value outside
  the checked spline interval. The 5% materiality threshold controls warning
  presentation only; it no longer permits small excursions or skips clipping.

- Specification scoring now respects the evaluation horizons in structured
  folds, including rolling and sliding windows. Structured folds retain base
  observation weights and use their training weights once; bare matrices
  continue to supply resampling multipliers, including bootstrap counts.
  Conflicting weights, invalid masks, and empty effective folds raise errors.

- `cv_risk(score="mse")` now scores response-scale predictions. For Binomial
  models this is the Brier score, including conversion of signed class labels
  to 0/1. Previously it compared observations with link-scale predictions.

- LSS `cvrisk()` and `stabsel()` refits now honor the family's
  `stabilization` setting; both previously dropped the gradient
  stabilization the fit used.

- Every LSS family constructor takes `stabilization=` (`"none"`, `"MAD"`,
  `"L2"`), upstream gamboostLSS's spelling since 2.2-0 deprecated the global
  option. The fit-level `stabilization=` argument remains and now defaults to
  `None`, meaning "use the family's"; passing a value overrides the family
  for that fit. Case variants (`"mad"`, `"l2"`) are accepted.

- The book's API reference is generated from the package's own objects
  (`tools/generate_api_reference.py`), and CI fails when it is stale. It was
  one hand-written markdown cell documenting a few dozen of 146 public names,
  and what it did document had drifted: it showed `glmboost(formula, *, data,
  …)` after `data` became positional, listed `control()` and `offset()` after
  both became properties, and gave `risk(parameter=None)` for a method whose
  arguments were removed. Every signature, default and keyword-only marker now
  comes from `inspect.signature`, every property-or-method marking from
  `inspect.getattr_static`, and each family and learner carries the
  verification status its capability row records.
  The page also carries a **model-surface table**: every public member of the
  three fitted-model classes and its form on each, so a member that is a
  property on one and a method on another is visible at a glance rather than
  discovered by a caller.

- `BoostControl`: `boost_control()` returns a named record instead of an
  anonymous 4-tuple. `help(mb.boost_control)` was empty, `type(c)` was `tuple`,
  and reading a field meant knowing that `nu` is index 1. It is a `NamedTuple`,
  so it still **is** a tuple — the bindings, the artifact format and every
  `control[0]` keep working — and `c.nu` now reads as what it is. Validation
  still happens in the Rust constructor, so the rules are stated once.

- `LSSModel.__getitem__`: `fit[:k]` truncates a distributional path with the
  same spelling the standard classes use. It refits rather than slices,
  because a cyclic path is not prefix-consistent, and says so;
  `fit[:{"mu": 40, "sigma": 10}]` gives the parameters different budgets.

- `model.risk` on every class, as a property. It existed only on the
  distributional class, as a method whose two arguments were both refusals.

- Every fitting function is `fit(formula, data, *, ...)`. The distributional
  entry points were positional all the way down; the standard ones required
  `data=`. One rule now, and `tests/test_api_consistency.py` runs the same
  task on all four classes and compares the *form* of the answer — property
  against method, argument shape, return shape, and the exception class on the
  same mistake.

- `LSSModel.cvrisk()` needs no arguments. It required both `folds` and `grid`,
  arguing that the fold scheme has to be shared across the whole grid — which
  is a reason to build *one* and share it, not a reason to have none. Reaching
  a first cross-validated result meant assembling two matrices by hand. The
  default grid is the diagonal, which the engine reads off one fitted path per
  fold.

- `families=` is the documented distributional family argument, matching
  `gamboostLSS` in R and the R frontend here — and used 254 times in this
  codebase against 11 for the alias. The docstring said the reverse.
  `family=` is still accepted.

- `import mboostrs` costs **50 ms** instead of **559 ms**. 509 of those were
  `mboostrs.plot`, and 453 of *those* were Altair, which pulls `jsonschema`
  and through it `rfc3987_syntax`, whose module body alone took 349 ms
  (measured with `python -X importtime`). Every caller paid that to fit a
  model. `mboostrs.plot` now loads on first use through a module `__getattr__`,
  so `mb.plot.plot_risk(fit)`, `from mboostrs import plot` and every
  `model.plot*()` method are unchanged — only the moment of the import moves.

- Importing no longer changes Altair's **process-wide** renderer. It flipped
  it from `default` to `svg`, which affects every Altair chart in the process,
  including ones from libraries with nothing to do with mboostrs.
  `mboostrs.plot.use_svg_renderer()` does it explicitly, and setting
  `MBOOSTRS_SVG_RENDERER=1` makes importing the module do it — which is what
  the documentation book's build scripts now set, next to the two Polars
  formatting variables already there.

- `altair` and `vl-convert-python` moved to a `plot` extra, and `pyarrow` to
  an `arrow` extra. No package code imports `pyarrow`, and current Polars
  converts to and from pandas without it, so it was a hard dependency for a
  path that does not exist. The base install is `numpy` and `polars`; fitting,
  prediction, cross-validation and every table work without the extras, and
  `import mboostrs.plot` without Altair says which extra to install rather
  than only naming the missing module.

- Generally tagged Linux and Windows wheels use Rust's portable x86-64
  baseline again. The former x86-64-v3 configuration could install on an older
  CPU and terminate with `SIGILL`; the Python preflight checked only AVX2 even
  though v3 requires a larger feature set, and it also rejected a compatible
  generic source build because the installed extension carried no marker of
  how it was compiled. CPU-specific instructions remain available for local
  benchmarks through `RUSTFLAGS="-C target-cpu=native"`, never for release or
  CI builds.

- The scikit-learn estimators compare **column names**, not positions.
  `check_array` returns a bare ndarray, so a frame's names were gone before
  anything compared them with the ones the fit saw — and the training names
  were then attached to the new columns by position. Measured on a fit over
  `a` and `b`: `predict(X[["b", "a"]])` returned −2.119 where `predict(X)`
  returned 1.039, having multiplied `b`'s values by `a`'s coefficient, and
  `predict(X.rename(columns={"a": "z"}))` returned the original numbers as
  though nothing had changed. Only a wrong column *count* was caught. Every
  estimator now routes validation through scikit-learn's `validate_data`, so a
  reordered or renamed frame raises the same message any other estimator would
  give. A compatibility shim covers both APIs: the `BaseEstimator` method up to
  1.5, the free function from 1.6 (the method is gone in 1.9), and the package
  supports ≥ 1.3.

- `TreeBoostRegressor` and `TreeBoostClassifier` read the caller's column
  names. They captured them *after* validation had already replaced the frame
  with an array, so every tree fit was built over `f0, f1, …` whatever the
  columns were called — which made an explicit `formula="y ~ btree(a)"` fail
  on a frame with a column named `a`.

- `feature_names_in_` is set only when the input carried names. It was set to
  `["f0", "f1"]` for a bare array, so a fitted estimator claimed feature names
  it had never been given.

- A regressor no longer carries `n_classes_ = None`. A fitted attribute that
  does not apply should be absent; `None` reads as "fitted with classes, found
  none".

- **Warm continuation in the engine.** `mboostrs_core::try_fit_continue`
  resumes a boosting path from the state a shorter fit left, rather than
  replaying it. `continue_to(N)` on a fit of `m` iterations ran all `N` again:
  the first `m` are deterministic given the data, so the answer was right and
  cost twice what it needed to. A continuation is **bit-for-bit** equal to a
  cold path — the tests compare raw bit patterns of the risk path and the
  fitted values, over `Gaussian`, `Poisson`, `Huber`, `NBinomial` and
  `PropOdds` — and runs only the iterations it adds (71 gradient evaluations
  against 101, continuing 30→100).
  Two things were needed for exactness, and neither was obvious:
  - `PropOdds` calibrates its thresholds by coordinate descent **seeded from
    the current thresholds**, so its state is path-dependent and
    recalibrating on resume would run five sweeps the original path never ran.
    A resumed fit restores them from the retained path instead.
    `Family::calibration_is_path_dependent` marks it; every other family
    calibrates by a pure function of `(y, f, w)` and is reproduced exactly by
    recalibrating.
  - `BoostingPath` retains the negative gradient the loop left
    (`BoostingPath.residual`). For a squared-error family the loop maintains
    it incrementally, and the rounding accumulated in that subtraction is not
    what a fresh evaluation at the same fitted values gives: recomputing it on
    resume put the continued path one unit in the last place away at the
    resume step, and the difference compounded. The vector is *moved* into the
    path — it is allocated once per fit and dead after the loop — so retaining
    it costs nothing, which matters because cross-validation and stability
    selection go through the same code.

  The Python `continue_to` still refits cold: reaching the engine's warm start
  from Python needs the retained residual passed back across the FFI boundary
  that P3.1 is about to replace, so it is wired up there rather than twice.

- `confint()` draws its bootstrap resamples from the model's own weights. It
  drew them as `binomial(1, 0.5, n)`, ignoring those weights, where `mboost`'s
  `confint.mboost` uses `cv(model.weights(object), B = B)` — multinomial
  counts with probabilities proportional to the weights, so a row the fit gave
  no weight is never drawn. Measured on a 300-row fit whose first half has
  slope +2, second half −2, and second half carries weight zero: the fit
  estimates **+2.02** and the interval came back **[−0.508, 0.241]**, centred
  on −0.13. It did not contain the estimate, because the replicates had fitted
  all 300 rows, where the halves cancel. It is now [1.972, 2.081].

- `continue_to`, `confint` and the stability-selection fold refits go through
  the model's `RefitSpec`, so every hyperparameter the fit accepted reaches
  the refit. Each rebuilt the call by hand from five or six attributes, so a
  fit with `bins=`, `center=`, explicit `learners=`, a non-default
  `weight_policy` or `na_action="omit"` was refitted as a plainer model under
  the same name.

- `refit_spec` no longer passes `learners=` and `env=` together, which
  `gamboost` refuses. Any fit that rewrote its formula — an interaction lowers
  to `buser(__ix0, X=__ix0_X, ...)` — hit this, so `conformal_interval` on an
  interaction fit raised. The spec keeps both, because `row_subset_blockers()`
  reads the learners to see what a row subset would misalign; only the call
  chooses.

- `aic()` and `hatvalues()` refuse with a reason instead of returning `None`.
  A `None` meant either "this artifact carries no learner store" or "a
  selected learner has no hat matrix" — a `btree` prediction is not a linear
  map of the response, so there is no trace to take — and the caller could not
  tell which.

- `cvrisk` and `stabsel` on a loaded artifact refuse by name. `stabsel`
  reached `_extract_column` on `None` and reported `Cannot extract column 'y'
  from <class 'NoneType'>`.

- `model.unavailable_operations()`: what a model cannot do, and why. Empty for
  a model fitted in this session; for one loaded from a portable artifact or
  unpickled, every operation that refits — with the reason stated once rather
  than found five times at five call sites.

- `predict(newdata=, extrapolation="warn"|"error"|"allow"|"clip")` and
  `model.range_report(newdata)`. The old warning was `mboost`'s wording —
  "Some 'x' values are beyond 'boundary.knots'" — where `'x'` is the literal
  string, not the column, so a model over `area` and `rooms` was told about
  `'x'`. It fired once per call however many learners were involved, so a
  second out-of-range learner was silent, and it said nothing about how far
  outside. The message now names each offending learner, how many rows fall
  each side, and the worst distance **in training ranges**, which is what
  separates a boundary case from a different question. `"clip"` evaluates the
  basis at the nearest training value, holding the effect flat outside the
  range instead of continuing at the edge slope; neither continuation is more
  correct, and the policy makes the choice visible. Measured on a fit over `x`
  in `[-2.40, 2.00]` with residual sd 0.3, the two continuations differ by 3.5
  at `x = ±6`.

- `range_report()` returns one row per spline learner: the interval its basis
  was built over, the range it is being asked about, the rows each side, and
  the distance in training ranges.

- Partial-effect plots shade stretches where the data is thin. A fitted curve
  is drawn at the same weight everywhere, so a stretch supported by three
  observations looks exactly like one supported by three hundred; the rug
  carried the density for a reader who counted ticks.

- `model.binning_report()`: requested bins, distinct values, **effective**
  bins, ties collapsed, and which learners `bins=` did not reach. `bins=50` on
  a column taking seven distinct values bins nothing, and nothing said so.

- `bins="auto"`, on a learner or as the global argument. It sizes the bin
  count against four things and records which one decided: four bins per basis
  column (a binned design with as many rows as columns is exactly determined
  and leaves the penalty nothing to shrink), the column's distinct values, a
  memory budget, and a floor of four rows per bin below which the exact design
  is already about as small. On a 20000-row `bbs(x, df=4)` it chooses 92 bins,
  collapsing 19908 ties and cutting retained storage from 23.7 MB to under a
  tenth of that; on a column with seven distinct values it declines and says
  why.

- `check_binning()` reports `bytes_saved` and a `scope` line stating that
  agreement is evidence about that fit on that subsample, not a property of
  the bin count.

- The specification assistants read the frames people actually have. Every one
  of `classify_columns`, `suggest_terms`, `term_report`, `cv_risk` and
  `compare_specifications` cast each column with `np.asarray(v, dtype=float)`,
  so a frame with one string column raised `could not convert string to float:
  'b'` — nothing in the frame could be classified, scored or reported on, not
  even the numeric columns beside it. Two of the five also assumed a mapping
  and raised `'DataFrame' object has no attribute 'items'` on a Polars frame.
  A single adapter (`mboostrs._frame`) now reads mappings, Polars and pandas,
  and all five give identical answers on all three.

- A **boolean** column is an indicator, not an unordered factor. `bols(flag)`
  was refused with "encoding it as numbers would make bols treat the levels as
  an ordered measurement" — which is the correct reading for a boolean, since
  `False < True` is the coding and not an arbitrary label order. The identical
  column cast to float fitted and recovered the effect.

- A **date** column is a continuous axis and fits with any learner. It was
  read as an unordered factor and refused — except under Polars, whose dtype
  name fell outside the categorical set, so the same data fitted in one
  backend and was refused in the other two. Dates are encoded as days and
  timestamps as seconds, and the unit is reported, so a slope is per day
  rather than per nanosecond.

- A **two-level** factor fits with `bols`. With two levels every coding gives
  the same fit up to the sign of one coefficient, so the ordering objection
  does not apply; it still applies, and is still raised, for three or more.

- `cv_risk` scores with the family's own loss by default. It computed squared
  error of the response for every family, so a `Binomial` fit was ranked by
  comparing a 0/1 outcome with an unbounded log-odds — not the loss the model
  is fitted to minimise. `score="mse"` keeps the old behaviour.

- `compare_specifications` accepts a plain list of right-hand sides. It
  required a mapping and raised `'list' object has no attribute 'items'`.

- The shape search in `term_report` only tries shapes that fit. It offered
  `bols` on an unordered factor, which the fit path refuses for the same
  reason a spline over label codes is meaningless, so the report raised
  partway through.

- `cv_risk`, `term_report` and `compare_specifications` take `weights=`,
  `folds=` (a count, a matrix, or a `FoldScheme`, so grouped and time-ordered
  validation reach them), `na_action=` and `score=`. `mstop="cv"` tunes the
  stopping point *inside* each outer fold, so specifications are compared each
  at its own best stop rather than at one budget that favours whichever suits
  it.

- `ColumnKind` carries the source dtype, the level labels in code order,
  whether the order is declared, the missing count, the date unit, a `reason`
  for the learner it suggests, and `alternatives` — the other learners that
  fit the same column and what each one *assumes*. `explain()` prints it. A
  recommendation with no stated alternative is an instruction.

- A declared categorical (`pandas.Categorical`, Polars `Enum`) keeps its own
  level order rather than being re-sorted alphabetically, and an ordered
  factor is suggested `bols` — its codes are a scale — where an unordered one
  gets `brandom`.

- `Suggestion` carries `specification`, the full right-hand side the change
  produces, ready to pass to `compare_specifications` or `gamboost`, and
  `evidence`, the numbers behind the ranking.

- `heteroscedasticity(model)` and a distributional-response suggestion. The
  test is the slope of `log(residual²)` on the fitted value; under constant
  variance `log(residual²)` is `log` of a chi-square with one degree of
  freedom, whose variance is `π²/2` whatever the scale of the data, so the
  slope gets a standard error without estimating one. Measured on a fit with
  `sd ∝ exp(0.5x)`: `t = 7.6` and the spread grows 2.8-fold across the fitted
  range; on the same design with constant variance, `t = -0.13`.

- `compare_specifications` rows carry what the score was computed under — the
  fold scheme's own name, the budget, `nu`, whether weights were used — so two
  comparisons run days apart can be told apart, and `output=` returns a frame.

- Fold constructors for data whose rows are not exchangeable, all returning a
  `FoldScheme` that `cvrisk` and `stabsel` take wherever a matrix would:
  - `stratified_folds(y, B)` preserves class proportions. A plain 10-fold
    split of 200 rows carrying 12 positives leaves two of the ten folds
    holding **no** positive case at all (measured on the default
    `cv_weights` folds); the stratified split gives every fold one or two.
    The out-of-fold risk of a fold with no positive case estimates nothing.
  - `grouped_folds(groups, B)` keeps each cluster whole, for repeated
    measurements and clustered sampling. A random split leaves a row's own
    cluster-mates in the training data, so the held-out row is partly
    predicted by itself.
  - `blocked_folds(n, B)` holds out contiguous stretches, for local
    dependence. It still trains on rows after the block it is judged on, and
    says so rather than pretending otherwise — for a spatial transect that is
    not a leak.
  - `rolling_folds(n, B, horizon=, initial=, expanding=)` trains on the past
    only, expanding or sliding. The only scheme here that measures
    forecasting.
  - `repeated_kfold_folds(n, k, repeats)` averages over several partitions,
    for when the choice of partition moves the answer more than the models do.

- `FoldScheme.check_leakage(groups=...)` reports rows a fold both trains on
  and is judged by — including for a grouping the scheme knows nothing about,
  which is the case that matters: a plain k-fold split of repeated
  measurements leaks by subject and the matrix cannot say so.
  `mboostrs.folds.warn_if_leaky(matrix, groups)` does the same for a bare
  matrix.

- `CvRiskResult.standard_error`: the across-fold spread of the mean risk. The
  CV plots now draw it as a band; a curve whose whole dip is a fraction of one
  standard error has chosen its `mstop` from the fold split.

- `CvRiskResult.mstop_1se` (and `LssCvRiskResult.mstop_1se`): the
  one-standard-error rule — the earliest stop the folds cannot distinguish
  from the best one. Drawn on the plot when it differs from the minimiser.

- `CvRiskResult.compare_to(other)`: a paired comparison on the folds two
  models share, refusing to pair curves from different folds (checked by a
  fingerprint of the fold matrix, since the matrix itself is not kept). It
  reports the mean difference, an interval, and **both** standard errors: the
  naive across-fold one and the Nadeau–Bengio correction for the fact that CV
  folds share training rows. For 10-fold the correction inflates the error by
  1.45; for the default bootstrap folds, whose held-out share is measured at
  0.369 rather than assumed to be `1/B`, by nearly four.

- `diagonal_grid(max_mstop, n_params)` and `LssCvRiskResult.grid_cost()`. The
  engine already read every diagonal grid point off one fitted path per fold,
  and nothing surfaced it: ten diagonal points cost one path, a 10×10 product
  grid costs a hundred. `grid_cost()` reports the split, and the docstrings
  say why nothing off the diagonal can be reused — a cyclic LSS path is not
  prefix-consistent, since freezing one parameter changes the later updates of
  the others.

- `cvrisk` warns when the cross-validated risk is still falling at the last
  iteration evaluated: the reported `optimal_mstop` is then where the path
  stopped, not where the risk turns, and it reads exactly like a minimum the
  data chose. The message names the refit to run. The LSS grid gets the same
  warning when the best point is the largest budget searched.

- A `CvRiskResult` from a `FoldScheme` reports the scheme's own name —
  "grouped 5-fold" — where it used to say "5 supplied folds".

- `gamboost().confint()` counts a replicate that never selects a learner as a
  **zero** effect, which is what an unselected base-learner contributes and
  what `mboost`'s `confint` bootstraps (its `predict(which=)` returns zeros).
  Those replicates were recorded as `NaN` and dropped by `np.nanquantile`,
  which removes exactly the replicates where the effect is zero and so pushes
  the interval away from it. Measured on a 150-row fit where 39 of 300
  replicates never selected `z`: the interval covered zero at 37% of grid
  points before, and at 100% after.

- `gamboost().confint(parm=...)` rejects a learner name the model does not
  have. An unknown name used to reach the bootstrap loop, match nothing, and
  come back as an all-`NaN` interval — a typo reported as a result.

- A bootstrap replicate that fails is counted and reported. Both `confint`
  implementations caught every exception and left the row `NaN`, so an
  interval built from 40 surviving replicates was returned in the same shape,
  with the same `level`, as one built from 1000. Now: all replicates failing
  raises and quotes the first exception; some failing warns with the count and
  the reason; a failed inner `mstop` cross-validation warns separately,
  because those replicates did contribute, at the outer model's `mstop`. The
  result carries `B`, `n_used`, `n_failed`, `n_untuned` and `seed`.

- Reading a fit's stored state no longer hands out the model's own array.
  `fitted_` and `risk_` returned `self._fitted` and `self._risk`, so
  `f = fit.fitted_; f -= f.mean()` changed what `fit.residuals()` reported
  afterwards (measured: the first three response residuals moved from
  `-0.5211, -0.4967, -1.4113` to `-0.4969, -0.4725, -1.3871`) while
  `fit.predict()` recomputed from the coefficients and returned the original
  values. They are read-only views now: reading costs nothing, a write raises
  `ValueError: assignment destination is read-only`, and `.copy()` still gives
  a writable array. Methods that compute a quantity — `fitted()`,
  `model_weights()` — return fresh arrays as before.

- `control`, `offset` and `selected` are properties on every model class. The
  three classes disagreed: `control` and `offset` were methods on `glmboost`
  and `gamboost` and properties on the distributional classes, and `selected`
  was the other way round, so the same expression returned a value on one fit
  and a bound method on another — and `if fit.selected:` was true for every
  distributional fit whether or not anything had been selected. The old call
  forms still work and emit a `FutureWarning` naming the release that removes
  them (`mboostrs._compat.DEPRECATIONS` holds the schedule).
  `fit.selected(parameter=..., merge=...)` on a distributional model is now
  `fit.selected_which(parameter=..., merge=...)`, which gained `merge=`.

- A distributional fit's `offset` is the `{parameter: value}` mapping that
  `offset_` already returned. It was a list in `param_names` order, so the
  same quantity under two names indexed differently.

- Every result object takes `to_frame(output=...)` and reads as a mapping.
  `LssCvRiskResult` and `LssStabSelResult` took no `output=`; `CvRiskResult`,
  `StabSelResult` and `AicResult` were not subscriptable, so generic code
  written against `result["aggregate"]` worked for a distributional fit and
  raised `TypeError` for a standard one. `AicResult` gained `to_frame()`.

- Every result carries `result.provenance`: the model kind, family, row count,
  base-learner count, fitted budget, weight policy, the **effective** weight
  total, and — where the operation resamples — the seed, the fold scheme and
  the fit count. `result.describe()` prints it. A curve saved to disk or read
  back later could not previously say what it was computed on.

- Runtime branching on model kind goes through `mboostrs._dispatch`
  (`isinstance` against the real classes) everywhere. `summary`, `explain`,
  `plot` and `model_selection` still asked `hasattr(model, "gaic")`,
  `hasattr(model, "partial_effects")` and `not hasattr(model, "_selected")`,
  each of which silently inverts the day the other class grows that name.
  `mboostrs._protocols` states the shared surface for type checking, and the
  three classes are tested against it.

- The remaining broad `except Exception` handlers name what they catch: the
  Altair renderer and font registration at import, the partial-effect rug
  variable, the new-data coordinate column, and the three `profile_mstop` AIC
  branches. Two stay broad on purpose and say so — the rows in
  `profile_mstop` and `compare_to_r` *are* the typed missing value, carrying
  `status="error"` and the message.

- Broad `except Exception` handlers in `summary()`, `plot_diagnostics()` and
  `profile_mstop()` now name the exceptions they mean. A refusal that is a
  fact about the model — `partial_effects()` on a tree, `residuals()` on a
  ranking family, `gaic()` on a family with no likelihood — leaves the column
  or panel out, as before. Everything else propagates: an arithmetic or
  indexing error inside those calls used to be reported as a blank cell or a
  missing panel, which reads as "this model does not support it".

- Ruff runs over `tests/`, `tools/` and `benches/` in CI, not just `python/`.
  It was not run in CI at all, so 144 findings outside `python/` had
  accumulated unseen: unused imports and locals, `f`-strings with no fields,
  and assignment-in-`except` shadowing. All are fixed; the line-length and
  import-order rules are relaxed per-directory in `pyproject.toml` for test
  and tool files rather than reformatted.

- Tests assert the exception the public contract promises. Eleven
  `pytest.raises(Exception)` checks passed on any failure at all, including an
  `AttributeError` from a typo in the test itself; each now names the type and
  a fragment of the message (`ValueError, match="mstop must be > 0"`,
  `KeyError, match="is required by the model"`).

- `model.parity_notes()`: where a fit differs from what its names refer to.
  A user reaching for `Gehan`, `Bkernel` or an AFT `logLik` has a meaning in
  mind, taken from `mboost` or from the literature; sometimes mboostrs
  computes something else, and sometimes `mboost` does too and mboostrs
  reproduces it, so the object differs from its *name* rather than from
  upstream. Each note says four things: a short citable name, the mechanism,
  the consequence for a reader who assumed otherwise, and **whether upstream
  shares the behaviour** — the last being what decides what to do about it,
  since a deviation mboostrs invented might be a bug to report while one
  shared with upstream is a property of the method as implemented everywhere.
  Notes carrying a measurement cite the test that made it.

- `summary()` names the notes that apply, so a reader meets them where they
  are reading rather than in a generated appendix they would have to think to
  look for.

- R's `coef()` on a distributional model applies the centering correction it
  was missing. `glmboostLSS` centers its predictors, so a slope fitted on the
  centered column implies an intercept shift of `-sum(beta_j * mean_j)` when
  the coefficients are reported on the original scale. Without it the R
  frontend reported the intercept as 0 whenever the intercept learner was
  never selected — the usual case, since the offset already carries the level
  — where `gamboostLSS` and the mboostrs Python frontend both report
  `-mean(x) * slope`. Measured: -0.036374 against a reported 0. Predictions
  were unaffected, which is why the existing parity tests passed: they
  compared `predict()` and not `coef()`.

- `GaussianKnownVarLSS` in the R frontend. It was the only Python-only
  distributional family, and the asymmetry had no reason behind it: the core
  family and the `cbind(outcome, s2)` response path both existed, and the R
  dispatch simply had no entry. R and Python now agree on offset, risk and
  fitted values to 1e-6 on a shared fixture. The family has no `gamboostLSS`
  counterpart, so the capability note says plainly that the cross-frontend
  check establishes the two frontends assemble the same call, not that the
  family is right — there is no independent implementation to compare against.

- `stabsel` on a Multinomial model refuses instead of crashing. A Multinomial
  fit stores `n * (K - 1)` fitted values, so the row count `stabsel` took from
  them was `K - 1` times too large and the fold matrix failed to broadcast:
  `operands could not be broadcast together with shapes (400,) (200,)`, which
  names neither the family nor the missing feature. It now refuses for the
  same stated reason `cvrisk` does — the Kronecker learner expansion has no
  counterpart in the selection-path engine — and names the alternatives.

- R `predict(newdata=)` works for centered smooths. `bbs(center=TRUE)` removes
  the penalty's null space, so the fitted design has `p - d` columns where the
  raw basis has `p`; the R bridge rebuilt the raw basis, found the counts
  disagreed, and refused. That reached the default `bbs(center=TRUE)` and
  every `bspatial`, whose `center` defaults to TRUE — so new-data prediction
  was unavailable for the most common smooth in the R frontend. The bridge now
  rebuilds through the same transform the fit used, and predicting on the
  training rows reproduces `fitted()` to 1e-15.

  Two different transforms are in play, and using the wrong one *runs*: a
  one-dimensional `bbs` centers by the difference matrix `D'(DD')⁻¹`, while
  `bspatial` centers by the spectral decomposition of the tensor penalty.
  Rebuilding a `bbs` with the spectral transform gave an in-sample replay off
  by 0.80 — a number that reads as a poor fit rather than an error. The tests
  therefore check that prediction reproduces `fitted()`, not that it succeeds.

- `r_bspatial_design` applies its `center` argument, which it accepted and
  ignored.

- `stabilization={"none","MAD","L2"}` on `glmboostLSS` and `gamboostLSS`,
  matching `gamboostLSS:::stabilize_ngradient`. The parameters of a GAMLSS
  live on different scales — a location in the response's units, a log-scale
  in dimensionless ones — so their partial gradients have different magnitudes
  and one `nu` takes very different-sized steps on each. `"MAD"` divides each
  gradient by the weighted median absolute deviation from its weighted median;
  `"L2"` by the root weighted mean square. Both carry upstream's floor of
  `1e-4`, and `"L2"` its ceiling of `1e4`. The default is `"none"`, which is
  `gamboostLSS`'s default too, so an unchanged call is unchanged. Coefficients
  match `gamboostLSS::glmboostLSS` with `GaussianLSS(stabilization=...)` to
  1e-5 on all three settings.

  This is the upstream option, not the divergence guard added earlier: the
  guard stops a path whose risk has already exploded, and stabilization is
  what keeps a well-behaved path from taking mismatched steps in the first
  place.

- A penalized `bols` matches `mboost` in the R frontend. `bols(x, lambda=1)`
  differed: R's `fitted()[1]` was -0.2348387 against `mboost`'s -0.2317270 on
  a ten-row Gaussian fixture, with the gap opening at the first iteration and
  growing with `mstop`. `bols(x, intercept=TRUE)` has two engine forms — a
  centered single column taking its constant from the model offset, and the
  two-column block `[1, x]` carrying its own — and a penalty makes them
  different models, because R shrinks the learner's own intercept along with
  its slope and a constant supplied by the offset is not shrinkable. The R
  parser sent the centered form regardless; it now sends the block form, and
  the ten cases in `mboostrs/tests/test_bols_form_parity.R` agree with
  `mboost` to 3.6e-15.

- Mixing the two `bols` forms in one formula matches `mboost`, in **both**
  frontends. `bols(x, lambda=2) + bols(z)` put a block learner beside a
  centered one, so the constant was represented twice and the two
  representations competed along the path: fitted values differed from
  `mboost` by 1.1e-3 where an all-centered or all-block menu agreed to 3e-15.
  The form is now a property of the whole menu — one penalized or `by=` term
  puts every `bols` in the formula into the block form. The selection sequence
  was identical throughout, so a parity test comparing only `selected()` would
  have passed the whole time.

- `bols(x, df=...)` is refused by the R frontend rather than fitted with the
  wrong penalty. The block spec carries a penalty matrix and no `df`, and the
  R binding has no `df -> lambda` calibration for it; the refusal points at
  `lambda=` and at the Python frontend, which does calibrate.

- `_rust.pyi` describes the module it is a stub for. It had drifted both ways —
  22 of 36 registered symbols had no stub, and 8 stub entries named symbols the
  module does not export — because nothing compared the two. It is generated
  now, by `tools/generate_rust_stub.py`, from the extension's own
  `__text_signature__`; CI runs `--check` and fails on drift. Argument types
  are `Any`, because PyO3 does not surface them, so what the stub pins is the
  set of symbols and each one's argument names, arity and defaults — which is
  the part a caller gets wrong.

- `cargo test --doc` runs tests. It ran zero. The fallible path helpers carry
  runnable examples, and one of them caught a wrong assumption while being
  written: `CoefResult` pairs learner indices with values rather than indexing
  densely by learner.

- `try_coef_glmboost`, `try_coef_gamboost` and `try_predict_glmboost` return
  `MboostrsError` instead of panicking on a mismatch. A boosting path's
  selection sequence and its coefficient array are two halves of one record,
  and nothing checked they had the same length: a caller that built one from a
  truncated path and the other from the full one got `index out of bounds`
  with no mention of which two things disagreed. A design whose length is not
  `n * p` is reported with both numbers. The panicking forms remain for the
  internal callers that construct both arrays together.

- Public claims that were mechanically disprovable. `gamboost`'s docstring
  said "a bare name is a linear term"; a bare numeric name becomes
  `bbs(name)`, a P-spline at `df=4`, matching R — so `"y ~ x"` fits a smooth,
  not a line. It also said "every learner is given the same degrees of
  freedom, which is what makes the comparison between them fair"; the defaults
  differ and callers set them independently, so equal df is something the
  caller arranges, and `"y ~ bbs(x) + bols(z)"` is not an equal-df menu.

- `continue_to` is described as what it does. Two book pages said it "extends
  without restarting"; it refits from iteration zero and saves the
  re-specification, not the computation.

- PFER is described as an upper bound on the expected *number* of falsely
  selected learners — a count, not a probability and not a rate — and the
  per-learner figure as the fraction of subsamples the learner was selected
  in, not "the stability selection probability". Stability selection makes no
  statement about the probability that a learner belongs in the model. PFER
  also bounds *learners*, so a `bols(x) + bbs(x)` menu spends it on two
  candidates for one source variable.

- The LLM description no longer claims every effect is centered. Measured: a
  plain `bbs` effect had mean 2.7 on a fit whose offset was supposed to carry
  the level. Where an effect sits depends on its learner; what is always
  meaningful is the difference between two points *within* one effect, and the
  text now says that.

- The LLM description no longer calls the AFT predictor "the log of the
  expected survival time". It is a location on the log-time scale: `exp(eta)`
  is the fitted distribution's scale, and the mean is that times a
  family-specific factor the predictor does not carry. The exact statement —
  that a contribution of +1 multiplies every survival quantile by `e` — is
  what it says now.

- The LLM description carries a mandatory non-causal caveat, asks for the
  covariates the model *uses to predict* rather than those that "drive" the
  response, and no longer asks the model to name the most likely mechanism
  behind a surprising effect: the fit contains no evidence separating a real
  mechanism from confounding or a data artefact, so it asks what observation
  would distinguish them instead.

- `llm_explain_prompt` rejects a `cv=` or `stabsel=` result belonging to a
  different model. Both were rendered into the prompt as if they were this
  model's, with nothing for the reader to check them against.

- The capability contract records R `Bbs` new-data prediction as `partial`,
  not `verified`. `predict.mboostrs_gamboost` refuses a selected
  `bbs(center=TRUE)` term — measured: "coefficient length 22 does not match
  rebuilt design columns 24" — which reaches `bspatial` too, whose `center`
  defaults to TRUE. A `partial` status was added to the contract's vocabulary
  for exactly this shape: some inputs work and some are refused by a
  limitation the implementation reports.

- `sp500_two_lanes.ipynb` is withheld from the book. Its title asserts that
  the break *raised* returns, and its prose calls baseline variables
  confounders, post-break variables mediators, and fitted contrasts effects,
  none of which its design identifies: the break date is chosen after seeing
  the data, the panel holds current index members so departed firms are absent
  from the pre-break window they belonged to, there is no comparison group and
  no parallel-trends argument, and row-wise half-samples ignore that returns
  cluster within a stock and correlate across stocks on a day. The page keeps
  a banner saying what would make it publishable. The case-study count in the
  overview was already stale at eighteen against a prose "Seventeen"; removing
  the row makes the two agree.

- A formula term is understood whole or rejected. Each learner parser looped a
  keyword regex over the text inside its parentheses, and a regex reports what
  it matched and nothing about what it did not, so three things passed
  silently. **Extra positional variables**: `bbs(x, y, z)` matched with `x` as
  the variable and `"y, z"` as the keyword string, where no keyword matched —
  the learner was `bbs(x)` and the other two columns vanished, so a caller
  writing what they took for a multi-column term got a single-column fit with
  nothing to indicate it. **Duplicate keywords**: `df=3, df=4` took the last
  one quietly. **Trailing text** inside the parentheses, for the same reason
  as the first. The parsers now prove the keyword matches tile the text, and
  report anything left over against the term the caller wrote.

- A boolean argument rejects anything that is not one. The test was
  `value in ("True", "TRUE", "true")`, so `bbs(x, center=maybe)` became
  `center=False` — a typo that turned centering off and changed the fit.
  `T`/`F` are now accepted alongside the Python and R spellings.

- A numeric argument that will not convert names the learner and the argument.
  It raised a bare `could not convert string to float: 'notanumber'`, which
  named neither.

- `glmboostLSS(env=...)` is rejected rather than accepted and ignored. `env`
  resolves symbols inside learner calls; a linear formula has none, so there
  was nothing for it to resolve and accepting it read as support for something
  that did not happen. The message points at `gamboostLSS`.

- Conformal calibration refits the model that was actually fitted. `_refit`
  rebuilt the call from the formula, data, family and control alone, dropping
  the weights, `center`, the row and component subsampling settings,
  explicitly supplied learners, `env` and `bins`. A conformal interval on a
  weighted, centred, binned fit was therefore calibrated against an
  unweighted, uncentred, exact one, and reported a coverage guarantee for a
  model the caller had not fitted.

- Conformal calibration says when its guarantee does not apply. Split
  conformal's coverage rests on the calibration scores being exchangeable with
  the test point's, which case weights break: an observation weighted 3 counts
  three times in the fit and once in the calibration. Weighted conformal
  prediction is a different construction and mboostrs does not implement it,
  so a weighted fit now warns that the level is nominal rather than
  guaranteed, instead of applying the unweighted guarantee silently.

- Conformal calibration refuses every fit it cannot validly subset, not only
  `Buser`. `Bmrf`, an externally supplied constraint matrix, an interaction
  recipe built against the full frame, and row subsampling have the same
  difficulty, and each is now named in the refusal.

- CV+ no longer refits the full model. It did so twice — once to learn the
  prediction width and once for the point estimate — repeating the whole
  boosting path for a model already in hand.

- CV+ holds a bounded working set. Its endpoints are order statistics over the
  leave-fold-out predictions at each target row, so the target rows are
  independent; they are now processed in chunks sized to about 64 MB, where
  the whole `n x n_new` pool was materialised. The default
  `newdata=training_data` makes that pool `n x n`, or 800 MB at n = 10,000.

- `mboostrs._refit.RefitSpec`: one description of how a model was fitted, with
  `subset(rows)` that takes every row-aligned object together and
  `row_subset_blockers()` that names what stops a row subset being valid.
  Continuation, bootstrap inference, conformal calibration, cross-validation
  and stability selection each used to reconstruct the call from a different
  subset of the model's attributes.

- `residuals(type=...)` names each kind for the quantity it is, and refuses
  where that quantity is undefined instead of returning a different one.
  `type="pearson"` on `Laplace`, `Quantile`, `Expectile` and `Huber` returned
  the *working gradient*: `Laplace`'s is `sign(y - mu)`, which takes two
  distinct values across a whole sample. Those families declare no variance
  function, so Pearson is refused and the refusal names `type="working"` —
  the new kind, available for every family, that this quantity always was.

- `CoxPH` refuses a response residual, which was `time - exp(eta)`:
  subtracting an observed time from a relative risk. `type="martingale"` and
  `type="deviance"` are the residuals a partial-likelihood fit has, and are
  now implemented with Breslow's baseline cumulative hazard.

- The AFT families (`Weibull`, `Loglog`, `Lognormal`) refuse a response
  residual for the same reason: `eta` is a location on the log-time scale and
  the response is censored, so the subtraction is neither on a common scale
  nor meaningful for a censored row. The refusal points at `type="working"`
  and at `quantile_residuals()`, which handles censoring. A censored residual
  derived for AFT is not implemented, and the message says so rather than
  offering the subtraction.

- `Gehan`, `Cindex` and `AUC` refuse a response residual: their fitted value
  is a ranking score, identified only up to a monotone transformation, so the
  difference from an observation has no units and no zero.

- `residuals()` and `residual_frame()` compute the same thing. For a Gaussian
  fit the frame divided by the residual RMS and the method did not, and both
  called the result Pearson. Neither does now: `V(mu) = 1` for a Gaussian, so
  its Pearson residual is the raw residual, as in R's
  `residuals.glm(type="pearson")`. Dividing by an estimated dispersion gives a
  *standardised* residual, which is a different quantity.

- `model.residual_kinds()` lists what the family defines. The list comes from
  the same family table the implementation dispatches on, so the declaration
  and the code cannot drift apart.

- `PropOdds` documents the `{1, ..., K}` coding it accepts. The class said
  `{0, ..., K-1}` while the fit required the other, so a caller following the
  docstring got "y must be integers in 1..K, got 0". The docstring also said
  `n_levels` was inferred from the response; it is not, and omitting it leaves
  the count at 2, so a three-category response is refused.

- `Multinomial` documents that it is available on `gamboost` only.
  `glmboost` rejects any response outside `{0, 1}` whatever `n_classes` says,
  because the Kronecker design the family needs is built on the `gamboost`
  path.

- `family_diagnostics` applies the family's own response encoding, as `fit`
  and `cvrisk` do. Without it a `Binomial` response given as `{0, 1}` reached
  a loss written for `{-1, +1}`: the offset, gradient and risk were then
  self-consistent but described a different function from the one the fit
  minimises, so the book's finite-difference audit passed while checking the
  wrong thing. The risk from this entry point now matches the fitted risk
  exactly.

- Every tabular helper takes `output={"auto","polars","pandas","records"}` and
  honours it, on standard and distributional models alike. `coef_all` accepted
  an `output` argument and then built the frame without it, so all four values
  produced the same result; its old default `"dataframe"` is not a value the
  output policy defines and is now mapped to `"auto"` rather than rejected.
  `predict_all`, `residual_frame`, `q_stats`, `worm_data` and `outlier_frame`
  did not take the argument at all in their method forms.

- `q_stats` has one implementation instead of two that disagreed. The module
  function used equal-width bins and a population variance; the model method
  used equal-count bins and a sample variance, so the same fit gave different
  variances under identical column names. Both now use equal-count bins — so
  every bin's moments come from a comparable number of observations, which
  equal-width bins on a skewed variable do not give — the unbiased variance,
  and the same square root of it as the scale for skewness and excess
  kurtosis. Both report the binning variable in an `xvar` column.

- `calibration_frame` works on a distributional model, where it previously did
  not exist. It reports the location parameter's mean as `location_mean`
  rather than `fitted_mean`, because the two coincide only when the family's
  first parameter is the mean — true for `GaussianLSS`, false for the survival
  families whose `mu` is a location on the log-time scale. It also switched to
  equal-count bins, and refuses a multi-class fit, whose prediction is a
  probability matrix with no single column to bin.

- `LSSModel` gained `worm_data()`, which the module-level function already
  supported.

- `logLik()` returns a log-likelihood or refuses, instead of returning a
  negated loss for every family. `Binomial`, `Poisson`, `NBinomial`,
  `PropOdds`, the AFT families and every distributional family minimise the
  exact negative log-likelihood, so `-risk` is the answer and the value is
  unchanged. `Gaussian` reconstructs it from `sigma^2 = RSS/n`, as before. For
  `Laplace`, `Quantile`, `Expectile`, `Huber`, `AdaExp`, `HingeLoss`, `AUC`,
  `Gehan` and `Cindex` there is no probability model at all, and `logLik()`
  now raises `NotImplementedError` naming the quantity the fit does estimate
  and what to use instead, rather than returning a number that compares like a
  log-likelihood without being one. `mboost` returns `-risk` throughout; the
  new `negative_risk()` is the accessor for that quantity, under a name that
  says what it is.

- `aic()` and `gaic_table()` refuse the same objectives. An AIC over a robust
  or ranking loss is not an AIC, and `gaic_table()` additionally refuses to
  rank log-likelihoods taken with respect to different quantities — a Gaussian
  fit of `time` against an AFT fit of the same column, for instance, where
  both are proper likelihoods but one is a density of `T` and the other of
  `log(T)`, so the difference between their AICs is not a likelihood ratio.
  The refusal names which model broke the comparison.

- `likelihood_contract()` on any model returns the family's declared row:
  whether the loss comes from a probability model, whether `-risk` is the
  complete log-likelihood, what measure it is taken with respect to, which
  residual kinds are defined, and the variance function where one exists.

- The `Multinomial` risk omits every observation in the reference class. Its
  loss is summed over the K-1 non-reference columns, where such an observation
  is zero in all of them, so it contributes nothing; the omitted
  `-log(p_reference)` moves with the fit, so this is a missing part of the
  objective rather than a constant that cancels. `mboost::Multinomial` has the
  same loss, so the behaviour is kept for parity, and the gradient is correct
  for those observations so the *fit* is unaffected. What the omission reaches
  is every risk-valued output: the reported risk, the risk path, the
  out-of-bag risk `cvrisk` selects `mstop` on, and any information criterion.
  `logLik()` now refuses for `Multinomial` and says so.

- The AFT families' risk is the negative log-likelihood of `log(T)`, not of
  `T`: the `1/t` Jacobian is absent. This is a proper likelihood, and it is
  what makes an AFT AIC incomparable with a Gaussian AIC of the same column.

- `CoxPH`'s risk is the Breslow negative log *partial* likelihood, which ranks
  Cox models against each other and is not a density of the data.

- `GammaReg` fixes the shape parameter at 1 and never estimates it, so the
  log-likelihood of the fitted model is not determined by the risk.

- Invalid family and learner parameters are rejected at construction instead
  of reaching a numerical routine. `Quantile(qoffset=2)`, `Huber(delta=-1)`,
  `PropOdds(n_levels=1)`, `Multinomial(n_classes=1)`, and `Cindex(sigma=-1)`
  all used to construct. A non-positive Huber `delta` makes the loss decrease
  with the residual and reports a negative value as risk; a non-positive
  C-index `sigma` inverts the smoothing sigmoid so the loss rewards discordant
  pairs; a one-category count describes a response that does not exist. The
  binding repaired the last two silently with `.max(2)` and `.max(3)`, fitting
  a different model than the caller asked for; those repairs are gone and the
  values are refused by name.

- The same checks run at three places, because a family reaches the engine by
  three routes: a Python or R constructor, the binding dispatch used by
  cross-validation and stability selection, and artifact deserialization. A
  value rejected when typed is now rejected when loaded from a file.

- A category count must be a whole number. `Multinomial(n_classes=3.5)` was
  truncated; it is now refused.

- `Btree(names=...)` requires at least one variable and no repeats: an empty
  list gives a tree with nothing to split on, and a repeated name makes the
  same column compete with itself.

- Every scalar-name learner requires one non-empty string. `Bols` already
  rejected a list; `Bbs`, `Bmono`, `Bmrf`, `Brandom`, `Bns`, `Bkernel`,
  `Bconstrained`, `Buser`, and both `Bspatial` names accepted one, and a
  non-string besides. Those reached the fit path and failed there as
  `unhashable type: 'list'` when the name was used as a dict key, or as a
  column lookup that could not match, neither of which names the argument.

- `Buser`'s `X`/`K` and `Bconstrained`'s `C` accept any 2-D array-like (a
  nested list included) rather than requiring a numpy array, are converted
  once so downstream code has an array, and are checked for finiteness and
  conformability. A non-finite entry is named by position.

- `Bbs(boundary_knots=...)` must hold two finite, strictly increasing values.
  A degenerate or reversed pair produced a basis with no support or one whose
  knots run backwards -- a silently wrong design rather than an error.

- R's `Quantile()` gained the `qoffset` argument it was missing, matching the
  Python constructor and the value the engine already read.

- Fractional weights now follow upstream's `rescale_weights`, which changes
  the fitted path of every weighted fit whose weights are not integers.
  Integer weights are frequencies -- weight 3 stands for three identical rows
  -- and keep the scale they were given; fractional weights are analytic,
  where only the ratios carry meaning, and are rescaled to sum to the number
  of weight-positive rows before fitting. `mboostrs` sent the raw vector to
  the engine, and the weight scale is not cosmetic: it enters `X'WX`, the
  `df -> lambda` calibration, and the risk. On a ten-row Gaussian `gamboost`
  fit with `bols(x, lambda=1)` and weights summing to 8.4, every fitted value
  and every risk value differed from R; both now agree to 1e-6. Integer-weight
  fixtures could not see it, because integers are exactly the case upstream
  leaves alone.

- `weight_policy={"upstream","raw"}` on `glmboost`, `gamboost`, `glmboostLSS`,
  and `gamboostLSS` in both frontends names the rule rather than leaving it
  implicit. `"raw"` keeps the caller's scale, for weights whose sum is itself
  meaningful, and is the previous behaviour under a name.

- Cross-validation and stability-selection folds go through the same policy
  column by column, because upstream refits each fold through `mboost_fit`.
  Fold matrices drawn from unit weights are 0/1 and unaffected; it matters
  when the model itself carries fractional weights and each fold inherits
  them. Complementary pairing is a property of the columns before that step,
  matching `stabs:::check_folds` building the pair and `mboost_fit` rescaling
  each column afterwards.

- All-zero weights are rejected rather than fitted. The object they produced
  reported a completed path over an empty sample: every learner's normal
  equations were empty, the risk was identically zero at every iteration, and
  the fitted values were all zero. Those are finite numbers that mean nothing,
  so the failure moved to the call.

- Models distinguish the weights they were given from the weights they used.
  `model.weights` (R: the stored `weights`) is the effective vector the engine
  saw -- the same convention as `model.weights()` in `mboost`, and what CV and
  stability-selection folds should be drawn from -- while `input_weights()`
  returns the caller's own vector and `weights_note()` says in one line how
  the scale was settled. Drawing folds from the input vector under a rescaled
  fit now fails with that instruction instead of producing negative
  complementary weights.

- The LSS weight-length error message is the shared one
  (`weights length 3 != n=10`) rather than a second spelling.

- Stability selection under the default `sampling_type="SS"` now really uses
  complementary pairs on every entry point. `stabsel_glmboostLSS` and
  `stabsel_gamboostLSS` drew `2 * B` independent half-samples and then handed
  `B` to the Shah-Samworth bound as if adjacent columns were complements; no
  adjacent pair summed to the full-sample weights, so the reported error
  control was computed for a sampling scheme the run had not performed. All
  paths — standard Python, LSS Python, and R — now build folds through one
  constructor that draws `B` half-samples and appends their exact complements,
  so column `b` and column `B + b` sum elementwise to the model's weights.
  Existing LSS parity tests only covered `sampling_type="MB"`, which left the
  default outside the gate.

- Stability-selection results now report the subsample count and the fit count
  separately and consistently. `B` is the number of initial subsamples the
  bound is computed on — complementary pairs under `"SS"` — and `n_fits` is
  the number of models fitted, `2 * B` under `"SS"` and `B` under `"MB"`. The
  standard Python result previously set `B` to the fit count; the R `stabsel`
  result reported the requested `B` while passing `B / 2` to the bound; and
  the LSS results carried no `B` at all. Both counts now appear in `print()`
  output and in the "mstop too small" warning.

- Caller-supplied `folds` means the same thing everywhere: an `n x B` matrix
  of *initial* subsamples, matching `stabs:::check_folds`. Under `"SS"` the
  complements are appended, so `B` supplied columns produce `2 * B` fits.
  Pairing is never inferred from an even column count. The R LSS entry points
  previously treated supplied folds as the final fit matrix and halved the
  column count for the bound, so the same matrix meant different things in
  Python and R.

- R's `stabsel()` refits on in-bag *weights* rather than on a row-subset data
  frame, so every fold sees the same rows in the same order and a subsample
  and its complement correspond row for row. It also uses the same
  q-truncation and `phat` accumulation as the LSS and Python paths, where it
  previously stopped a fold's accumulation early instead of charging later
  selections to the first learner, and it gained `sampling_type=` and
  `folds=` arguments instead of hardcoding `"SS"` while sampling
  independently.

- Cyclic and non-cyclic `gamboostLSS` fits no longer run to completion when
  the joint risk explodes. Component-wise steps are proportional to the
  gradient, and GaussianLSS updates `mu` along `(y - mu)/sigma^2`, so a
  response whose standard deviation is far below 1 (decimal-fraction returns,
  for example) multiplies every cycle's step by `1/sigma^2`: coefficients
  overshoot orders of magnitude past the least-squares solution and the risk
  grows geometrically (observed as `1e8 -> 1e213` within tens of cycles).
  Both LSS engines now stop with a typed error once the risk exceeds its
  running minimum by more than six orders of magnitude — measured as an
  absolute increment against `|min| + 1`, because LSS risks are signed — or
  turns non-finite, naming the response-scale remedy. Brief risk excursions
  at aggressive step lengths (`nu` up to 0.4) recover within a few cycles and
  are left alone, matching R's behaviour; the guard fires only on paths whose
  output is meaningless regardless.

- `ExecutionReport.retained_learner_bytes` now reports on the `gamboostLSS`
  and `glmboostLSS` paths, where it read `0` whether or not binning was on.
  It was the one diagnostic that would have told a caller their `bins=` took
  effect, and on the path where binning saves the most it said nothing. The
  LSS figure sums the parameters, because an LSS fit prepares the candidate
  set once per distribution parameter and holds all of them at once.

- `gamboost(..., bins=)` no longer refuses a whole formula because one term
  cannot be binned. A `by=` on any learner, an intercept-carrying `bols`, and
  a shape-constrained `bbs` have no binned design; they now keep their exact
  storage and are named in a warning, and every other term is binned. Before,
  the global argument raised `ValueError: global bins requires
  Bols(..., intercept=False)` on any formula written the ordinary way, so the
  argument
  that existed was the one that did not work on first contact.

- The "mstop too small in some subsampling replicates" warning now says how
  many folds fell short, how many distinct learners the best of them reached,
  and what to do when raising `mstop` will not help. A non-cyclic path that
  keeps re-selecting the same few terms can fail to reach `q` at any budget --
  on the S&P case study's scale-dominated specification no half-sample reached
  eight distinct learners at `mstop=600` -- and the remedy there is a smaller
  `q`, not a larger `mstop`. The warning matters for cost as well as for the
  error bound: a fold that never reaches `q` never stops early, so it spends
  the whole budget.

- `stabsel()`'s first per-fold attempt asks for half again what the full-data
  path needed to reach `q` distinct learners, where it asked for twice. The
  factor is what decides how much of the saving the shortcut keeps, and 2 was
  chosen without measuring: over 120 model shapes spanning row count, learner
  count, `q`, `nu`, predictor correlation, and seed -- restricted to shapes
  where the multiplier rather than the 32-iteration floor decides the budget --
  the most a fold ever needed was 1.41 times the full-data path, the 95th
  percentile 1.23, and the median 0.98. A half-sample usually reaches its
  `q`-th distinct learner slightly sooner than the full sample does. Folds that
  fall outside the first attempt are still refitted with `mstop` quadrupled, so
  the result is unchanged either way. On a 20,000-row twelve-learner Huber fit
  at `mstop=2000`, stability selection went from 3.8 to 4.6 times faster than
  fitting every fold to `mstop`.

- Removed a duplicate `p x n` copy of `X'W` from every block base-learner
  (`bbs`, `bspatial`, block `bols`). The array was held in both faer and
  ndarray layouts; every fit-time matrix-vector product read the faer one, and
  only the Gaussian Gram route wanted the ndarray, which it now materializes
  once at setup, as `bmono` already did. Retained learner data falls by a
  quarter — a 300,000-row, six-spline `gamboostLSS` fit reports 2,131 MB
  instead of 2,822 MB, and its peak resident set is 4,381 MB instead of
  5,418 MB. Stability selection prepares one such candidate set per fold, so
  the same quarter comes off every fold in flight.

- A case study on the two-lane hypothesis for S&P 500 constituent returns,
  `cases/sp500_two_lanes.ipynb`. A `gamboostLSS` fit on a 933,000-row daily
  panel asks whether the AI break moved location and scale in the same tiers,
  and finds it did not: both AI tiers stepped up in return, only the AI core
  stepped up in dispersion. Stability selection over fifty half-samples,
  a Student-t refit, and randomised quantile residuals on a held-out block
  check the scale result.

- `bins=` on `gamboostLSS`, `cvrisk_gamboostLSS`, and `stabsel_gamboostLSS`,
  matching the `gamboost` argument: a global quantile-bin count applied to
  every `bols`/`bbs`/`bspatial`/`brandom` term that does not carry its own.
  A model fitted with it carries the value into `LSSModel.stabsel()`,
  `LSSModel.cvrisk()`, and `update()`, so a refit does not rebuild the exact
  design the fit deliberately avoided. Before, the only route to a binned LSS
  fit was writing `bins=` inside every `bbs(...)` in every parameter's
  formula. On a 300,000-row, six-spline, two-parameter model the fit takes
  0.44 s at `bins=100` against 2.9 s exact, and stability selection over ten
  subsample fits takes 1.0 s against 82 s, on the same selected set. The
  linear frontends take no `bins=`: `glmboost` and `glmboostLSS` have no
  binned storage at all, so the argument would name a path that does not
  exist.

- `n_fits` on `StabSelResult` and `LssStabSelResult`, the number of models the
  run actually fitted. Under the default `sampling_type="SS"` that is `2 * B`,
  because complementary-pairs sampling needs each half-sample and its
  complement for the Shah-Samworth bound; the doubling is required by the
  maths but the parameter name reads as a fit count, so a caller budgeting for
  `B=25` was paying for 50 fits with nothing in the result saying so. The `B`
  argument is now documented as counting pairs on `stabsel`,
  `stabsel_glmboostLSS`, and `stabsel_gamboostLSS`.

- Opt-in quantile-binned training storage for `Bols`, `Bbs`, and `Bspatial`,
  with exact discrete storage for `Brandom`, a global `gamboost(..., bins=)`
  default, fold-shared bin indices in `cvrisk`/`stabsel`, and
  `model.check_binning()` for a matched subsample refit with binning disabled.
  At `n=100,000`, ten 24-column spline learners retain 4.2 MB at `K=100`
  instead of 720 MB and fit 6.0 times faster on the recorded Apple M4
  benchmark. `ExecutionReport.retained_learner_bytes` exposes the retained
  learner-data footprint separately from solver and Gram caches. The
  executable implementation chapter `implementation/binned_learners.ipynb`
  derives the compact solve, and `cases/million_rows_binned.ipynb` applies it
  to a one-million-row model with a matched exact-subsample check. A direct
  100,000-row exact-versus-binned comparison gives the same test RMSE and
  selected learner set; the compact fit is about 1.4x faster and retains about
  105x less learner data on the executed development run.

- A case study on the AI regime break in S&P 500 constituent returns
  (`docs/book/cases/sp500_two_lanes.ipynb`). One `gamboostLSS` fit on roughly
  930,000 stock-day rows estimates a location lane and a scale lane at once,
  and the two disagree: both AI tiers gain against the rest of the index at the
  break, while only the AI core's dispersion steps up, and under `StudentTLSS`
  only the AI core's tail index falls. The page is built around two
  specifications rather than one, because the natural control is a mediator —
  trailing volatility measured after November 2022 is a consequence of the
  break, so conditioning on it removes most of the AI core's return step. The
  estimation model therefore controls for each stock's volatility and momentum
  as of the last pre-break trading day, and the contemporaneous versions are
  used only where prediction is the job, for the stopping rule and the
  calibration check. Every base learner enters at two degrees of freedom, each
  continuous control split into a linear term and a centred `df=2` spline, so
  the selection comparison is not decided by flexibility
  (Hofner et al. 2011). Scale calibration is read from randomised quantile
  residuals rather than from a Gaussian-consistent scale estimator, which on a
  response with excess kurtosis near 60 reads about 18 per cent below the
  residuals' own standard deviation and would make a well-calibrated scale
  model look uniformly over-confident.

- A case study on area characteristics and life expectancy
  (`docs/book/cases/cz_inequality.ipynb`), with the `datasets.hip_cz()`
  commuting-zone file behind it. It re-asks the bivariate question of
  Chetty et al. (2016) under stability selection: 39 area covariates on 437
  commuting zones, each entering as a competing `bols`/`bbs` pair at one degree
  of freedom, against a `bspatial` surface on the zone centroids. Three
  learners clear the PFER-1 cutoff for bottom-quartile men — the surface at
  1.00, smoking at 0.99, Medicare spending per enrollee at 0.88 — and the
  covariates with the largest bivariate correlations are not among them. Every
  covariate that fails selection despite a correlation above 0.3 correlates
  above 0.4 with the fitted surface, which takes its regional variation first.
  The fitted surface is drawn on a map of the lower 48 states, and every number
  in the prose is an `{eval}` expression evaluated against the notebook's
  kernel at build time rather than a typed literal.

- `docs/book/check_writing.py`, which checks book prose against the
  mechanically checkable rules in `.remember/WRITING_STYLE_GUIDE.md`:
  typed numbers that should be `{eval}` expressions, author-year citations
  typed instead of `{cite}` keys, competing nouns for one concept, verdict
  labels, verdict verbs, question headings, stored outputs and untitled charts.
  It covers the rules `check_docs.py` does not, so neither duplicates the
  other's patterns, and is advisory unless `--strict` is passed.

- `tools/curate_us_states.py`, which vendors `docs/book/data/us-states.geo.json`
  — the 48 contiguous states and DC, decoded from the `us-atlas` TopoJSON
  distribution of the US Census cartographic boundary files and thinned to a
  vertex every 0.05 degrees. The commuting-zone case study draws its spatial
  surface on it. The outline is vendored rather than fetched by the chart so
  that `myst build --execute` resolves no remote URL.

- `families::r_zeroin`, a statement-by-statement port of the algorithm behind
  R's `uniroot` (`R_zeroin2`), and `df_to_lambda` now uses it on R's own
  bracket `[0, options("mboost_lambdaMax")]` with R's precedence rules from
  `mboost:::df2lambda`. The previous code solved the same equation with
  scipy's `brentq` convergence test on a conditioning-limited bracket. The
  conditioning limit is still applied, as a cap on the value returned rather
  than as the search interval, so the returned `lambda` is unchanged wherever
  the cap used to bind. Fixture-backed by
  `crates/mboostrs-core/tests/df_lambda_r_parity.rs` and
  `tests/fixtures/generate_df_lambda_fixtures.R`, which record R's `lambda`
  for six designs across 19 `df` values and pin agreement in three regimes:
  3.1e-7 relative where `df` sits at least 0.01 above the rank of the
  penalty's null space, a conditioning-deflated 1.7e-5 between the rank and
  that margin, and, at or below the rank, only that both implementations
  shrink the penalized directions away. `lambda` is not identified at the
  floor: `df(lambda)` never reaches the null-space rank, so `lambda` behaves
  like `c / (df - rank)` and the 1e-11 gap between R's LAPACK spectrum and
  faer's is amplified without bound. That is the whole of the `bmono`
  divergence reported yesterday; the constrained solver itself agrees. Two
  further tests record what the floor does to a fit rather than to `lambda`:
  at `df` equal to the rank the fitted values lie in the penalty's null space
  to within 1.3e-10, and half a degree of freedom above it at least 1e-3 of
  the fit lies outside. A learner at the floor is not fitting a curve badly,
  it is confined to straight lines — or, for a two-margin tensor at the
  customary `df = 2` per margin, to the bilinear span {1, a, b, ab}.

- A case study on deposit-supply elasticities
  (`docs/book/cases/deposit_elasticity.ipynb`). It estimates
  `-d log(balances) / d(rate)` as a smooth function of the state on two US
  samples — quarterly M1 against the T-bill rate, 1959-1993, and monthly bank
  deposits against the policy-rate-to-deposit-rate spread, 2009-2026 — and
  reports the marginal cost of funds each implies. It reproduces an earlier
  R `mboost` pass on the historical design to four decimals, then prices the
  three things that pass left out: cross-validation re-run inside each
  bootstrap replicate (the interval widens 45% and covers zero), the
  estimator's bias under no curvature (+0.003), and the design's power at the
  estimated curvature (0.41). The elasticity falls with the rate level in the
  historical sample and rises with the spread in the modern one.

- Four packaged datasets behind that study, curated by
  `tools/curate_deposit_data.py` and recorded with their vintage:
  `datasets.usmacro()` (quarterly US macro aggregates, the frozen 2009
  statsmodels extract), `datasets.deposit_market()` (monthly deposits, policy
  rate, FDIC national deposit rates spliced across the discontinued FRED
  series, and the Fed balance sheet), `datasets.bank_funding()` and
  `datasets.bank_funding_annual()` (Call Report deposits and interest expense
  aggregated over every insured US institution).

- R parity fixtures and tests for the designs that study fits
  (`tests/test_deposit_elasticity_parity.py`,
  `tests/fixtures/generate_deposit_elasticity_fixtures.R`): centered P-splines
  at df=1, a `by=` varying-coefficient interaction, and the modern spread
  design match mboost's selection path, risk path and fitted values exactly.
  Two divergences are pinned as strict expected failures: a design asking for
  `df` at the rank of the penalty's null space, where `lambda` is ill-
  conditioned and R's spectrum and ours part company (see below), and `bbs`
  ignoring `center=` when `constraint=` is set, which leaves a constrained
  df=1 spline able to fit only a straight line.

- A selection-sequence parity harness (`tests/test_selection_sequence_parity.py`,
  `--run-r-parity`): for gaussian, exact-tie, and binomial menus it asserts the
  index of the winning base learner at every iteration matches R exactly — the
  comparison that localises a divergence to one iteration — with risk
  trajectories at 1e-10 where the per-step arithmetic permits (gaussian) and a
  justified 1e-5 for binomial, whose Cholesky-vs-QR solve differences are
  amplified by the link (iteration 1 is bit-identical; measured 5.5e-7 by
  iteration 60).

- R parity for the `bmono(type="quad.prog")` active-set solver
  (`tests/test_bmono_quadprog_parity.py`, committed fixture from
  `tests/fixtures/generate_bmono_quadprog_fixtures.R`, runs without R). Both
  implementations run the same primal active-set QP through different Cholesky
  code, so the fitted values are compared at every iteration of the path with
  the measured accumulation as the tolerance: 1.5e-11 for `increasing` at df=8,
  9.4e-14 at df=12, 1.2e-10 for `convex` at mstop=60 and 6.7e-10 at mstop=300
  (1.8e-10 of the fitted span). The design is chosen so the constraint binds
  partially — an `increasing` constraint on rising data activates nothing and
  the QP collapses to the unconstrained solve — and three assertions keep it
  that way: the share of curve steps the constraint pins has a floor, the gap
  must accumulate rather than jump between neighbouring iterations, and the
  constrained fit must differ from the unconstrained one.

- `base_learner_names()` distinguishes learners that share a column. The
  decomposition `bols(x) + bbs(x, df=1, center=True)` stores both halves under
  the name `x`; the list now reads `["x (bols)", "x (bbs)"]`, and
  `learner_index()` accepts that form so the name leads back to an index and to
  `which=`. `disambiguate=False` returns the stored column names, which remain
  the keys of `coef()`, `selected`, `varimp` and `which=` — those are R's names
  for the same objects. A name that legitimately contains parentheses, such as
  `I(x^2)`, is left alone.

- `llm_explain_prompt()` on `glmboost`, `gamboost` and LSS fits, and as
  `mboostrs.llm_explain_prompt(model)`. It renders a fit as one prompt for a
  language model: how component-wise boosting produced it, the summary and
  selection tables, each learner's share of the in-bag risk reduction, and one
  serialized partial effect per learner — a smooth effect on an equispaced grid,
  a factor by level label, a two-covariate effect as a grid over both — each with
  the covariate's observed range, the learner's basis and penalty, and a stated
  shape (monotone, or where it turns). Effect values are rounded to the
  precision the effect's own span supports. The closing section states what the
  fit cannot answer: no standard errors, importance shared between correlated
  covariates, shapes valid only over the observed range, everything conditional
  on `mstop`. `cv=` folds in a `cvrisk` result, `stabsel=` a selection
  probability per learner, and `dataset_description=`/`response_description=`
  the context a fit cannot carry. `output="messages"` returns the system/user
  pair a chat API takes; `output="turns"` returns a three-turn sequence —
  describe the effects, name what is surprising and what could produce it, then
  compress both into `num_sentences` — which is `t2ebm`'s chain-of-thought
  structure, and exists because a model asked for a short summary first tends to
  produce a fluent one that skips the anomalies. Nothing is sent anywhere; the
  function returns text. Modelled on `t2ebm`, which does this for explainable
  boosting machines. The book chapter is `usage/llm_explanations.ipynb`.

- `llm_describe()` on the same fits, and as `mboostrs.llm_describe(model)`: the
  one function in the module that makes a network request. It posts the prompt to
  an OpenAI-compatible chat-completions endpoint and returns the reply. The
  endpoint defaults to NVIDIA's hosted inference API
  (`https://integrate.api.nvidia.com/v1`) and the model to `z-ai/glm-5.2`;
  `base_url=` points it at anything else speaking that protocol (a local NIM
  container, vLLM, llama.cpp, Ollama, OpenAI), `client=` uses a client object you
  already have, and the key is read from `MBOOSTRS_LLM_API_KEY`,
  `NVIDIA_API_KEY`, or `OPENAI_API_KEY`. `reasoning=True` runs the three turns in
  one conversation and returns the closing summary. The request goes out over
  `urllib`, so this adds no dependency. A reply that spent its whole budget in
  `reasoning_content` and returned no answer raises rather than returning an
  empty string.

- `llm_explain_prompt()` and `llm_describe()` in the R package, for
  `mboostrs_glmboost` and `mboostrs_gamboost`. The fixed sections are worded
  identically to the Python frontend, so a prompt built in either language reads
  the same to the model receiving it. R posts through the **curl** package when
  it is installed and the `curl` command-line tool otherwise, so it adds no hard
  dependency either. There is no LSS form in R, because R's `partial_effects`
  covers the standard models only.

- A factor `by=` on `bols()` and `bbs()` in `gamboost`. R stacks the learner's
  design once per non-reference level of the factor and penalizes it
  block-diagonally, which is the row-wise tensor of the treatment-contrast
  indicators with the design; those terms lower to a `buser` learner over that
  stacked design and predict on new data through the same recipe an `%X%`
  interaction uses. `df` applies to the whole stacked design and must be at
  least `differences * (levels - 1)`, as in R. Previously any factor `by=`
  raised `could not convert string to float` — or, when another term in the
  formula had already encoded the same column, silently multiplied the design
  by the integer level codes. `glmboost` and the LSS frontends, which cannot
  build the stacked design, now refuse a factor `by=` instead of doing that.

- Packed Gaussian scalar fits no longer include the ridge penalty in
  `sum(weights * prediction**2)` when updating risk incrementally. The fitted
  values and selection path were already correct, but reported risk was too
  large for penalized `bols`; exact-vs-binned path tests now pin selection,
  coefficients, fitted values, and risk together.

- The cross-validation-risk chart in `docs/book/cases/cz_inequality.ipynb`
  rendered empty. Its x axis is logarithmic and `cvrisk`'s grid starts at
  iteration 0, so every point collapsed onto x = 0 and the "optimal mstop" rule
  landed there too. Iteration 0, the intercept-only fit, is now dropped before
  plotting.

- The Gaussian Gram fast path broke exact selection ties between identical
  learners. Its incremental gradient-projection update read the lower-triangle
  cross-Gram blocks in the transposed memory layout `.t().to_owned()` leaves,
  and the matrix-vector kernels accumulate in a different order there, so two
  learners over bit-identical columns drifted ~1e-14 apart and the tie R's
  `which.min` resolves to the lowest index was decided by rounding noise from
  iteration 5 on. The lower-triangle blocks are now materialised in standard
  row-major layout, matching the upper triangle; twin learners stay
  bit-identical and the lowest index wins every iteration, as in R. Regression
  test `gaussian_gram_twin_learners_stay_tied` plus the GAUSS_TIE config of
  the new selection-sequence parity harness pin it end to end against R.

- R's `partial_effects()` returned a curve unrelated to the fit for any
  `center = TRUE` spline. The spectral centering transform leaves `p - d`
  coefficients while the design rebuilt for the new x has all `p` basis columns,
  so accumulating them recycled the shorter vector: measured on
  `bbs(x, df = 4, center = TRUE)`, a curve spanning 0.29 against the model's own
  1.88. R has no helper that re-applies the centering transform on new x — the
  same gap that makes `predict.mboostrs_gamboost` skip such a block — so the
  learner is now refused with a warning naming the cause instead of returning a
  fabricated curve. Covers the bbs/bmono, bns and brandom/bmrf branches.

- R's `partial_effects(fit, which = "x2")` returned another learner's effect.
  Filtering renumbered the kept learners from 1 while `object$selected` holds
  positions in the unfiltered list, so the loop aggregated whichever learner sat
  at the new index and multiplied those coefficients by the requested learner's
  design. Measured on a two-learner gamboost: the filtered call gave a range of
  [-0.374, 0.364] against the correct [-0.813, 0.972]. The loop now keeps the
  original indices, and `which=` also accepts them.

- `partial_effects()` raised `KeyError: column 'x1,x3'` for a `btree` or `brad`
  learner over more than one column. Such a learner is named for all of them
  joined by commas, and the effect coordinate was looked up as if that joined
  name were a column. It now reports against the learner's first column, which
  is what `bspatial` already did. `plot_partial_effects()` inherits the fix.

- `bols(x, by=z)` at the default `intercept=True` fitted a different model from
  R. R multiplies the whole design by `by`, giving the two columns `[z, x*z]`,
  so the term carries the `z` main effect; `gamboost` mapped `intercept=True`
  onto a centered single column spanning `{1, x*z}`, which cannot represent it.
  Measured on 300 points at mstop 100: max fitted difference 0.626, 46% of the
  fitted range. The same reasoning applies to any penalized `bols` — R shrinks
  the learner's own intercept, and a constant coming from the model offset
  cannot be shrunk — so `bols(x, df=)` and `bols(x, lambda=)` now take the
  two-column form too, and agree with R to 1e-16 where they were 6e-3 off.

- `bols` penalized every column but its intercept, where R's `X_ols` builds
  `K = diag(ncol(X))`. `lambda` was calibrated against a different matrix than
  it was applied to.

- A restored model can predict on new data again: `__getstate__` carried
  neither the transform recipe nor the interaction recipes, so a pickled model
  with a `log(x)` term or a `%X%` interaction came back unable to rebuild the
  columns and designs those terms need. Model artifacts (`save_model`) carried
  the transform recipe but not the interaction recipes; both now round-trip.

- `%X%` now honors the `df=` written on its marginals. It parsed them for knot
  placement and then assigned the tensor `df = ncol // 4` — 144 degrees of
  freedom for two default `bbs` margins — so `bbs(a, df=2) %X% bbs(b, df=2)`
  and `bbs(a, df=8) %X% bbs(b, df=8)` produced identical fits, and the term won
  every component-wise selection step against `df=4` main effects for reasons
  unrelated to signal. The tensor `df` is now the product of the marginal `df`s
  and the tensor `lambda` is 1 when both margins fix their own, which is R
  `mboost`'s rule; a margin that fixes `lambda` carries it into its block of the
  Kronecker sum, and an unadorned `bols` margin (`lambda = 0`) contributes no
  penalty.

- A `df` at the dimension of the penalty's null space no longer diverges. No
  finite `lambda` attains it, so the calibration clamped at 1e15, where the
  Gram no longer factorizes: the solver fell back to LU, the hat matrix came
  back with eigenvalues outside `[0, 1]`, and a boosting step through a negative
  one moved the fit away from the residual. The training risk rose instead of
  falling — to 1e34 with eight tensor `buser` terms and 1e171 with 30
  `bspatial(x1, x2, df=4)` terms. `lambda` is now capped at the largest value
  whose Gram still factorizes with a margin. Both `bbs` and `bspatial` warn at
  that boundary, as R does, and so does `%X%` when the marginal `df`s multiply
  into it.

- `df`-to-`lambda` calibration reads the penalty spectrum through singular
  values rather than a self-adjoint eigendecomposition, which resolved the
  small values of a rank-deficient design's spectrum to about three digits.
  `lambda` was off by 1% for a two-margin tensor and the fitted values by 1e-3;
  they now agree with R to 1e-7. `bspatial` R-parity tightens from 2e-4 to 1e-5.

- `check_writing.py` reports a page that cites sources but renders no
  reference list, which is guide rule 58: a `{cite}` key with no
  `bibliography` directive resolves to an author-year string pointing at a list
  the reader of that page alone does not have. `cases/cz_inequality.ipynb` was
  the one page in the book with that defect and now carries the directive.

- Three sentences in `cases/deposit_elasticity.ipynb` that suspended a subject
  for a dozen words of definition before reaching its verb. The scope paragraph
  is four short sentences instead of one wedged one, and says plainly that the
  number a bank needs is how much new money one extra point brings in, that the
  study does not produce it, and why: market rates move for reasons that move
  deposits directly, and one bank raising its own rate does none of those
  things. `co-movement` is gone from the page. Guide rule 37 carries the pattern
  — a mid-sentence aside longer than about five words is a sentence — together
  with the observation that a passage rewritten twice and still unreadable
  should be said out loud and written down fresh, since later drafts swap words
  while keeping the shapes that caused the load.

- The scope paragraph of `cases/deposit_elasticity.ipynb` states its point
  instead of posing it. "It makes no causal claim, and the missing one is the
  one a bank would want" made the reader resolve two descriptions and match
  them to recover a plain fact; it now says that pricing a deposit rate turns on
  what a bank's own balances would do if it raised its own rate, and that this
  data cannot give that. Guide rule 37 carries the general case: a subject named
  by a description the reader must resolve — *the missing one*, *the one that
  matters* — where the name was available.

- Guide rule 61, a quantity the prose uses is defined in the prose and not in
  the code: what goes into a derived quantity, which way round, and in what
  units, before the prose uses it. Rules 37 and 40 gained the two failures that
  came with it — a comparative separated from its verb by a subordinate clause,
  which binds to the wrong word on first reading, and the two cases where
  attaching a definition is the wrong repair: a term used once, where deleting
  it costs the same clause and needs no term, and a mechanism named a section
  away from where it is described. `check_writing.py` gained the
  p-spline/penalised spline synonym pair.

- Four passages in `cases/deposit_elasticity.ipynb` that a reader could not
  resolve. The forecast claim read "one specification predicts quarters it was
  not fitted to better than another does", which misparses at "fitted to better"
  and uses "specification" three sections before anything defines it; it now
  says which of two models predicts better, and that each is fitted to part of
  the record and scored on the part it never saw. `out-of-bag risk` appeared
  once, unexplained, and is gone in favour of saying what the zero-weighted rows
  do. The block bootstrap was described in the historical section and named only
  in the modern one, 35 cells later. And the modern spread named only one of its
  two legs: the deposit rate was identified and the fed funds rate standing in
  for the depositor's alternative was not, so the direction of the spread was
  left for the reader to infer from the code.

- Seven sentences whose only job was to announce the next one are gone from
  `cases/deposit_elasticity.ipynb` — "The arithmetic is worth doing once",
  "Three things come out of the table", "Both can be simulated", "Splitting the
  sample is the sharpest of the three" among them. Guide rule 57 already banned
  the construction; it now carries the table of instances and the test that
  finds them, which is that the sentence would still be true if the section it
  introduces were deleted.

- The opening of `cases/deposit_elasticity.ipynb` builds its situation before
  turning on it. It began "A deposit rate applies to every dollar in the
  account, not only to the dollars that a rate increase attracts", which
  presupposes three things it has not said: that someone is raising a rate, that
  raising it brings money in, and that a marginal cost is something the reader
  already distinguishes from a posted rate. It now says what deposits are and
  what the posted rate looks like first, and only then that another dollar costs
  more than that. Guide rule 42 carries the general case: an opening cannot
  correct a picture the reader has not been shown.

- Every heading in `cases/deposit_elasticity.ipynb` states the section's result
  rather than its topic, which is guide rule 60. A heading is the most-read line
  in its section and the one most likely to be quoted alone, and a table of
  contents made of topics carries no information. `## What the fit says a
  deposit cost` became `## Deposits posting nothing cost tens of points at the
  margin`, `## How much of the fall is signal` became `## The fall does not
  survive changing the model`, and the chapter title became `# A deposit costs
  tens of points more than its posted rate`. The one section that has no result
  to report keeps a descriptive heading: `## Five candidates compete at every
  step` states what is true of the method. `check_writing.py` flags a heading
  opening on What, How, Whether, Where or Why, since each names a question the
  section answers. The other case studies still carry topic headings and are
  candidates for the same change.

- `cases/deposit_elasticity.ipynb` no longer presents its falling deposit
  sensitivity as a finding. The page had checked the estimate against sampling
  noise — a block bootstrap, cross-validation re-run inside each replicate, and
  a power study — and never against the specification. Three checks it now runs
  change the reading. Letting the time trend bend rather than holding it linear
  returns a flat sensitivity of 0.0239 and a fall of exactly zero, at a
  one-year-ahead error of 0.0925 against 0.0866, so cross-validation prefers the
  linear trend but the rate term was holding curvature a competing trend term
  can take instead. The half of the sample carrying every quarter with a rate
  above 12% returns a flat 0.0077. And simulating from the two textbook
  money-demand specifications fitted to the same quarters shows the estimate is
  not a discovery in either direction: a constant elasticity (log-log) implies a
  fall of 0.0503, three times the 0.0164 estimated, and returns something above
  the observed value in 90% of runs, while a constant sensitivity returns
  something below it in 94%. The section heading became
  `## What the fit says a deposit cost`, the marginal cost of funds is reported
  as the part that holds across specifications, and the conclusions separate it
  from the contested slope. The modern sample gets the same treatment: removing
  the balance-sheet control turns its rise of +0.0270 into a fall of -0.0533, at
  a six-month-ahead error of 0.0658 against 0.0329.

- `check_docs.py --strip-outputs` exists. Guide rule 31 documented the flag and
  the script had no argument parsing at all, so the command in the guide failed
  as an unrecognised path. It clears stored outputs and execution counts from
  every notebook under `docs/book`, which is what an editor writes back
  whenever a notebook is run locally.

- `cases/deposit_elasticity.ipynb` is rewritten to be readable on its own by
  someone who is not a statistician. The opening does the repricing arithmetic
  on round numbers before any notation, so `MC = r_d + 1/theta` arrives as a
  consequence of paying the new rate on the whole book rather than as a
  definition; the estimated quantity is called *the sensitivity* in prose and
  `theta` only in the mathematics. Every section states its finding in its first
  sentence, the results are paragraphs rather than the
  claim/kind/evidence/confidence table guide rule 38 rules out, and the terms a
  reader would otherwise have to fetch from elsewhere in the book — degrees of
  freedom, the null space `center=True` removes, rolling-origin
  cross-validation, power, the varying-coefficient term, Huber loss — are
  explained where they are first used. The page carries its own
  `{bibliography}`, and names the three packaged datasets and the install line,
  so it can be sent to someone without the rest of the documentation.

  131 of the numbers in its prose are `{eval}` expressions evaluated against
  the notebook's kernel; the 56 literals left are the grid points named in the
  cell beside the sentence, the interval level, and the illustrative arithmetic
  in the opening. Two charts are new: the marginal cost of funds drawn beside
  the sensitivity it is the reciprocal of, and the five specifications'
  one-year-ahead errors as a sorted bar chart with the best two-parameter model
  drawn as a rule. Chart titles state the finding rather than the variables,
  and the modern sensitivity curve draws the zero line the fitted curve crosses.

- `cases/monotone_five.ipynb` and `cases/causal_constraint.ipynb` quote their
  results through `{eval}` expressions evaluated against the notebook's kernel
  rather than as typed literals — 58 and 81 of them, covering every fitted
  quantity the prose states. What stays a literal is what guide rule 33
  exempts: parameters of the simulated data-generating processes, grid points
  named in the cell beside the sentence, and section numbers. Sentences that
  quoted a quantity computed further down the page were moved below the cell
  that computes it, since an inline expression is evaluated in document order.

- Eleven case studies report their results as paragraphs rather than as a
  `**Supported.** / **Not supported.**` list, which is what guide rule 38 asks
  for: the label repeats the sentence, the two-bucket sort is not the shape of
  the findings, and an empty bucket attracts filler. `Out of scope` labels
  stay, since a scope limit is a fact about the sample rather than a verdict on
  the evidence.

- `stabsel()` fits each subsample only as far as its result depends on. The
  q-truncation rewrites every selection from the (q+1)-th distinct learner
  onward, and `phat` records each learner's first appearance, so a fold's path
  is fully determined once its q-th distinct learner has entered; the
  iterations after that were computed and discarded. Folds now stop there,
  starting from twice what the full-data path needed and quadrupling for any
  fold that has not reached q distinct learners. `phat`, `max_`, and the
  selected set are bit-identical to fitting every fold to `mstop`. `n_jobs`
  also defaults to `-1` rather than `1`, since the per-fold refits now release
  the GIL throughout. On the commuting-zone case study — 80 learners, 437 rows,
  `mstop = 2455`, `B = 100` (200 fits) — this is 190 s to 13 s.

- `gamboost()` prepares its learners on the rayon pool with the GIL released.
  Preparation (design matrix, penalty factorization, `df` -> `lambda` solve) is
  independent per learner and touches no Python object. It dominates a short
  fit's wall time on a wide model — 0.14 s of the 0.16 s a 40-iteration fit of
  80 learners on 218 rows takes, most of it one `bspatial` tensor — and it is
  paid once per fold by every per-fold caller.

- Linux and Windows wheels now compile with `target-cpu=x86-64-v3`
  (AVX2/FMA/BMI2), a 10-20% speedup on the hot GEMV/Gram-solve kernels on
  2013-or-newer x86_64 CPUs (Intel Haswell / AMD Excavator and later).
  **Breaking for older hardware**: installing the prebuilt wheel on a
  pre-2013 x86_64 CPU will crash with an illegal-instruction error; build
  from source instead (`pip install --no-binary mboostrs mboostrs`) on
  such machines. macOS wheels are unaffected.

- `LogNormalLSS`, `WeibullLSS` and `LogLogLSS` accept a plain positive
  response (zinc concentrations, costs), not only `Surv(time, status)`. A plain
  response is treated as uncensored, every observation an event, and its risk
  is the negative log-likelihood of the response on its own scale, the scale
  of `GammaLSS` and the other `realplus` families. The censored `Surv` risk is
  unchanged. A zero, negative or missing value is refused with the remedy.
  `FAMILY_SETS["realplus"]` now includes the three families, so
  `scan_lss_families(..., support="realplus")` ranks a log-normal (#333).

## [Unreleased]

## [0.1.3] - 2026-06-25

### Added

- 13 plotting functions (`plot_risk`, `plot_coef_path`, `plot_partial_effects`,
  `plot_cvrisk`, `plot_aic`, `plot_varimp`, `plot_residuals`, `plot_confint`,
  `plot_selected`, `plot_lss`, `plot_lss_risk`, `plot_multinomial`,
  `plot_lss_cvrisk`) — all return Altair charts (`alt.Chart` /
  `alt.LayerChart` / `alt.ConcatChart`) composable with `alt.hconcat` /
  `alt.vconcat`
- Model convenience methods: `.plot_varimp()`, `.plot_selected()`,
  `.plot_residuals()` on both GLMBoostModel and GAMBoostModel
- `mboostrs` Altair theme with bundled IBM Plex Sans / IBM Plex Mono fonts
  (SIL OFL 1.1, shipped in the wheel under `mboostrs/fonts/`): IBM Carbon
  palette, Plex Sans titles/labels, Plex Mono tick numerals. The fonts are
  registered with vl-convert at import, so static export
  (`chart.save("fig.png")` / `.svg`) renders identically on any machine;
  interactive renderers fall back through the CSS font stack. Charts embed
  with the SVG renderer (text re-paints when webfonts load; crisp at any
  zoom), and the docs book serves the Plex faces via `@font-face` (latin
  woff2 subsets inlined in `custom.css`). Theme registration uses the
  `alt.theme` API (no deprecation warning on Altair ≥ 5.5), with the legacy
  registry as fallback.
- Three-layer Jupyter Book documentation (34 pages: usage, methodology,
  implementation) with PDF and HTML export
- Expanded methodology chapters: functional gradient boosting, families,
  survival, base-learners, model selection, GAMLSS, stability selection
- Expanded `plotting_serialization.md` with full examples for all 13 functions

- Parallel fold loops for LSS cross-validation and stability selection
  (`cvrisk` over the per-parameter mstop grid, `stabsel` for glmboostLSS and
  gamboostLSS): folds run via Rayon with order-preserving collection, so
  results are identical to single-threaded runs. Combined with dropping
  unused per-step increment storage in the LSS drivers (no binding consumed
  it), measured 3.1–3.4× on 4 cores (n=2000, 4 bbs learners, mstop=200)

### Fixed

- Centralized GAMBoost Rust/formula learner index mapping in `LearnerLayout`,
  preventing drift between fit, predict, slicing, AIC/hatvalue, and external
  learner paths
- Applied the missing `nu` multiplier for Python-side Bkernel/Buser newdata
  prediction
- Replaced silent `LearnerStore` prediction metadata placeholders with explicit
  unsupported states and clear errors
- Updated LSS CV comments to describe the actual per-grid refit semantics
- Corrected LSS stability-selection parity coverage to compare against R's
  returned cutoff/q values instead of a stale hard-coded cutoff
- LSS drivers (`fit_lss`, `fit_lss_noncyclic`) record an explicit no-op step
  when every base-learner fit yields a non-finite SSE (degenerate gradient),
  instead of silently re-applying the previous iteration's update buffers
- `Laplace` offset reverted from the closed-form weighted median to R's exact
  `optimize()` Brent search over `range(y)`: the median deviated from R at
  Brent-tolerance scale on every fit (not only on even-n plateaus) and broke
  R-parity at 1e-10
- Unicode arrows (→ ↔) in doc source that broke LaTeX PDF rendering
- Replaced `\mathbb{1}` / `\mathbb{R}` with LaTeX-safe alternatives
- Removed `{func}`/`{meth}`/`{class}` Sphinx roles (unsupported in MyST 2.x)

## [0.1.0] - 2026-06-07

### Added

#### Core engine (Rust)

- Component-wise functional gradient boosting: `fit()`, `fit_with()` in
  `mboostrs-core`
- 13 loss families: Gaussian, GaussReg, Binomial, Poisson, Laplace, GammaReg,
  Quantile, Expectile, Huber, AdaExp, CoxPH, Weibull, Loglog, Multinomial
- Gaussian covariance/Gram-update fast path for single-fit inbag risk with
  all-linear learners
- Incremental risk update for Gaussian in general engine path
- Monomorphized cvrisk via `with_concrete_family!` dispatch
- Cached design matrices in `LearnerStore` for fast predict
- Learner-store reuse across CV folds via `reweight()`
- Streaming OOB risk in cvrisk (no per-fold increment storage)
- Parallel bootstrap refit loops via `n_jobs` (GIL-released fits)
- GEMV-ified engine inner column loop (3× speedup on large-n)
- SSE-by-identity selection, Binomial clamp, center_spectral eigenvalue fix
- Bounds-check elimination in engine hot loops
- Zero-alloc solve/SSE for gamboost performance
- O(n log n) bootstrap fold draws + eliminated per-fold y copy
- R's type-7 quantile in `families.rs` (deliberately matching R, not numpy)

#### Base learners

- `bols` (linear), `bbs` (P-spline), `bmono` (monotone P-spline), `btree`
  (regression tree), `brandom` (random effects), `bmrf` (Markov random fields),
  `bspatial` (spatial), `bns` (B-spline), `bkernel` (kernel)
- `by=` varying-coefficient support through full Python stack
- Kronecker-product learners for Multinomial (`bbs %O% bols`)

#### Python API

- `glmboost()` / `GLMBoostModel` — linear boosting
- `gamboost()` / `GAMBoostModel` — additive boosting with mixed base-learners
- `glmboostLSS()` / `gamboostLSS()` / `LSSModel` — GAMLSS boosting (cyclic and
  non-cyclic)
- 12 LSS families: GaussianLSS, GammaLSS, StudentTLSS, BetaLSS, NBinomialLSS,
  ZIPoLSS, ZINBLSS, DirichletLSS, LogNormalLSS, LogLogLSS, WeibullLSS
- `cvrisk()` with bootstrap, subsampling, and k-fold CV (per-model and LSS
  2-D grid)
- `stabsel()` for stability selection (GLM, GAM, and glmboostLSS)
- Inference: `coef()`, `confint()`, `varimp()`, `extract()`, `residuals()`
- Model utilities: `logLik()`, `summary()`, `update()`, `fitted_`,
  `coef_path()`, `aic()`
- `predict()` with `newdata`, `mstop`, and `type` arguments
- `boost_control(mstop=, nu=, risk=, trace=)` matching R's `boost_control`
- `cv()` and `mstop()` helpers for CV risk objects
- scikit-learn compatible estimators (`mboostrs.sklearn`)
- Pickle serialization for GLMBoostModel and GAMBoostModel

#### R binding (extendr)

- `gamboost()`, `cvrisk()`, `stabsel()` via extendr
- `glmboostLSS` + `gamboostLSS` R bindings
- Smooth (bbs) Multinomial in R binding
- `stabsel_glmboostLSS` R binding
- gamboost bbs designs built via the core basis (matching mboost)

#### Validation

- R parity tests (via rpy2, `--run-r-parity` flag) validating numerical
  agreement to ~1e-6 tolerance
- Parity tests for: Gaussian bols/bbs/bmono/btree/brandom/bmrf, Binomial,
  Poisson, CoxPH, survival families, all LSS families, Multinomial
- Property-based tests (NaN/Inf input validation)
- Smoke tests (R-independent)

#### CI / infrastructure

- Release pipeline + PyPI publish via maturin-action
- crates.io publish for `mboostrs-core`
- ruff lint gate
- `panic = "unwind"` for PyO3 (panics → Python exceptions)
- macOS cargo link config for dynamic_lookup

#### Documentation

- README with installation, quickstart, and scikit-learn guide
- Jupyter Book (34 pages) with PDF and HTML export
- R parity matrix appendix

### Fixed

- Multinomial softmax denominator (reference-class term) — risk/fitted now
  match R
- ScalarBols hat matrix intercept component
- brandom centering-means cumulative-offset indexing in predict_gamboost
- brandom center=True predict: column-mean centering instead of spectral
- GAMBoost `coef()`: warn on `off2int`, include offset in result
- `risk='none'` to match R semantics (inbag risk values)
- Binomial inverse link
- faer matmul β=0 silently writes nothing — zero buf + β=1 instead
- Gram path falls back to general path on degenerate (singular) learners
- Correctness bugs from engineering audit (2026-06-07)
