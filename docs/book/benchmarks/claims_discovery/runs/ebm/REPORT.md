<!-- Saved by the coordinator from the analyst's final message, verbatim: the agent harness refused this analyst's write of REPORT.md ("Subagents should return findings as text, not write report files"). -->

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
