# Final message of the ebm analyst, verbatim

The analyst's report-file write was refused by the agent harness, so its report came back in this message. The REPORT.md (and, where marked, FRICTION.md) in runs/ebm/ is extracted from it unchanged.

EBM analyst report (interpret 0.7.8, ExplainableBoostingRegressor). Run directory: docs/book/benchmarks/claims_discovery/runs/ebm/

DELIVERABLES WRITTEN
- run.py (pipeline). Canonical command run from the run directory: `python run.py --train <dev train> --test <dev test> --out dev_out --seed 1` exited 0 in 130 s on the loaded machine (n_jobs=4).
- dev_out/findings.json (brief schema, all Q1-Q8 present).
- dev_out/predictions.parquet (row_id, mean, lower90, upper90; 5,160 rows, no nulls).
- dev_out/tuning_log.csv (27 rows: 5 tuning configs, 13 drop-one refits, 2 feature-set choices, 6 pair candidates, 1 spread model; budget 40).
- dev_out/diagnostics.json (drop-one, pair and spread test tables, coverage by subgroup, marginal Spearman correlations, bootstrap SDs).
- dev_out/final_config.json (features, pairs, tuned parameters, seed).
- refit.py. `fit_predict(train, test, drop)` reads dev_out/final_config.json. It ran in about 3.8 s with no drop and with drop=['ave_occup']. With no drop its output matches dev_out/predictions.parquet `mean` exactly (max abs difference 0.0).
- REPORT.md and FRICTION.md are NOT written as files. The tool refused the REPORT.md write under the subagent rule for reports, and I removed the FRICTION.md file I had created earlier. Their content follows.

FINAL MODEL
ExplainableBoostingRegressor, objective rmse, 8 outer bags, max_leaves=3, min_samples_leaf=20, learning rate 0.04 (default), early stopping on the default 15% validation split. Covariates: median_income, roof_class (nominal), housing_median_age, longitude, latitude, ave_occup. Pairs: median_income x housing_median_age, roof_class x housing_median_age. 5-fold out-of-fold RMSE 0.469 (all 13 covariates: 0.471). Spread model: EBM on log(residual^2 + 1e-6), all covariates. Intervals: mean +/- q*sigma with q = 3.35 (weighted conformal on out-of-fold standardized residuals; density-ratio weights, classifier AUC 0.61).

REPORT (dev data, seed 1)

1. Associations (log scale; other covariates at median or mode; association = Holm p < 0.01 and worse in at least 4 of 5 folds when dropped):
- median_income, decreasing, convex: +0.55 at $10k, +0.05 at $30k, -0.22 at $50k, -0.50 at $100k, -0.65 at $150k. Dropping it raises out-of-fold MSE 6.8%.
- roof_class, categorical: A -0.12, B -0.11, C +0.06, D +0.27. Dropping raises MSE 6.6%.
- housing_median_age, threshold increase: flat near -0.13 for ages 5-25, rising between 30 and 40 to about +0.22, flat above 40. Dropping raises MSE 5.6%.
- ave_occup, increasing: -0.24 at 1.5, +0.12 at 4, +0.22 at 7. Dropping raises MSE 0.6% (weakest; Holm p 0.007).
- longitude and latitude, spatial: dropping raises MSE 1.8% and 0.8%. The longitude-latitude pair did not pass (p = 0.18).
- Not associated given the others: total_rooms, total_bedrooms, population, households, ave_rooms, ave_bedrooms, median_house_value.

2. Interactions (both involve housing_median_age):
- With median_income: MSE 0.5% lower (5 of 5 folds, p = 3e-5). Rise in y from age 20 to 50 is +0.51 at $20k income and +0.31 at $100k.
- With roof_class: MSE 0.3% lower (4 of 5 folds, p = 0.003). Weak; no clear pattern found.
- median_income x roof_class: p = 0.008 but gain 0.15%, below the 0.2% size rule.

3. Spread: households, population, total_rooms and total_bedrooms. Smaller block groups have larger residual spread. Permutation raises log-squared-residual error by 0.4-0.8% (Holm p below 1e-4, 5 of 5 folds). Direction is clear; the size of the change in residual scale is not computed in the deliverables. Plausible reading (to test): y averages over households, so noise falls with count.

4. Surcharge proposal: not supported. median_house_value is not in the prediction model; dropping it from a model with income lowers out-of-fold MSE 0.2% (0 of 5 folds). Marginal Spearman with y is -0.31: higher house values go with lower y, and income carries most of that information.

5. Predictions: out-of-fold coverage 0.900 overall, 0.897-0.904 by roof class and by income tercile. Median interval width 1.30 (10th-90th percentile 1.01-1.75). Test coverage is unknown (no test y). Train and test differ (classifier AUC 0.61; effective sample size 12,292 of 15,480).

6. Effect queries (estimate, 90% bootstrap interval, 30 Bayesian-bootstrap refits of the prediction model):
Q1 -0.506 (-0.537, -0.488); Q2 +0.405 (0.368, 0.438); Q3 +0.259 (0.237, 0.299); Q4 +0.388 (0.364, 0.422); Q5, Q7, Q8 = 0 with zero-width intervals (covariates not in the prediction model); Q6 +0.301 (0.242, 0.319).
A version that kept all covariates was tried and dropped: on a permuted-outcome draw its location query interval excluded zero.

7. Confidence: strong for median_income, roof_class, housing_median_age and location. Seed 2 (run before the alpha change to 0.01) gave the same covariates, pairs and spread drivers; housing age was labelled inverted_u instead of threshold_increase. Weak for ave_occup and both pairs. Query intervals are bootstrap intervals for a fixed model and leave out selection and tuning uncertainty, so they are likely too narrow. Permuted-outcome draw: no mean effect, interaction or spread driver. 800-row draw: out-of-fold coverage 0.894 overall, 0.87-0.95 across roof classes; no interactions and no spread drivers.

8. Cannot tell: causal effects (the outcome is simulated); whether location stands in for unobserved factors; behavior under the test shift beyond what the covariates show; whether missingness in housing_median_age (7.6% train, 7.0% test) carries information; anything about interactions among pairs not tested (only six candidates).

9. Next checks: spatial cross-validation by region; a spline or threshold fit for housing age (the 30-40 rise); a missing-indicator test; the spread effect in residual-scale units; coverage on the test book once outcomes are available.

FRICTION (InterpretML 0.7.8; quotes verbatim)
1. Missing values warning at fit: "UserWarning: Missing values detected. Our visualizations do not currently display missing values. To retain the glassbox nature of the model you need to either set the missing values to an extreme value like -1000 that will be visible on the graphs, or manually examine the missing value score in ebm.term_scores_[term_index][0]". Gives a fix, but partly; does not say the default missing="separate" already gives a separate bin. run.py suppresses it and records counts in findings.json.
2. Zero sample weights rejected: "ValueError: illegal sample_weight value of zero". No fix given. Used exponential (Bayesian bootstrap) weights.
3. Default interactions="5x" silently adds 65 pair terms (eval_terms returns 78 columns for 13 covariates). No message. Set explicitly.
4. term_scores_ bins vs cut points: term_scores_ has more entries than bins_ cut points (the missing bin is at index 0 per the docstring). Mapping by index raised "IndexError: index 719 is out of bounds for axis 0 with size 719". Fix: evaluate eval_terms on a grid frame instead.
5. validation_size=0: a 12,000-row fit took 165 s and reported best_iteration_ 650000 with default max_rounds 50000. No message.
6. measure_interactions without init_score ranks pairs on y, so main effects dominate the ranking. The docstring says only "either a model that can generate scores or per-sample initialization score". Passing out-of-fold main-effect predictions as init_score gave a different ranking. Nothing says which quantity is ranked.
7. No predictive intervals for regression: objectives are rmse, poisson/tweedie/gamma deviance, pseudo_huber, rmse_log; no quantile objective. predict_with_uncertainty gives the SD of the bagged mean, not a predictive SD. Not stated in the docs. Built the interval separately.
8. No standard errors for term or contrast estimates: standard_deviations_ is the spread of bagged term scores. No association test (term_importances has no null distribution). All tests are my own.
9. Timing: n_jobs=4 starts loky workers; on the shared machine, one timing call exceeded the 120 s command limit. Not a library message.

NOT COMPLETED
- REPORT.md and FRICTION.md as files (content above instead).
- Test-book coverage (no test labels).
- Spread effect size in residual-scale units.
- Query intervals exclude selection uncertainty; Q5, Q7, Q8 are structural zeros.
