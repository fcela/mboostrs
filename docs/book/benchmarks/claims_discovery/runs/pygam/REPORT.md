<!-- Saved by the coordinator from the analyst's final message, verbatim: the agent harness refused this analyst's write of REPORT.md ("Subagents should return findings as text, not write report files"). -->

# Claim-cost index: what the dev data supports (pyGAM 0.12.0)

## Summary
Four covariates are associated with the expected log claim-cost index: median income (decreasing), housing median age (rises above about 35 years), average occupancy per household (increasing) and roof class (higher for D and C). Location has a smooth spatial effect. Median house value is not a driver once income is in the model. The spread of y is wider in small block groups.

## 1. Covariates and shape of association
- Median income: decreasing and convex. The expected value falls from about +0.7 at income 1.2 to about -0.2 at income 7.9 (tens of thousands of dollars). The steepest part is the lower half of the range.
- Housing median age: flat at about -0.13 below age 30, rising through 35 to about +0.35 at age 40. The curve turns down above 45. The 871 block groups at exactly 52 are the census cap, so the turn rests on a thin, unusual group. The "inverted_u" label comes from this turn. The evidence supports a threshold-like rise, not a peak.
- Average occupancy (persons per household): increasing; the log-scale effect spans about 0.38 over the 2nd to 98th percentile.
- Roof class relative to A: B +0.01, C +0.17, D +0.38.
- Location: joint smooth of longitude and latitude, effect range about 0.86 across the training area.
- Not associated: households, average rooms, average bedrooms and median house value were removed by drop-one without a held-out error rise beyond noise. Total rooms, total bedrooms and population are not candidates because they are exact products of households and the per-household averages. Median house value has a raw rank correlation of -0.31 with y, but that runs through income.

## 2. Interaction
Housing median age and median income interact. The income slope is steeper for older block groups. In an exploratory fit (not in the pipeline output), the total income effect between income 1.5 and 7 was about 0.7 at age 10 and about 1.0 at age 50. The pair screen gave a held-out error change of -0.00074 (SE 0.00012); the joint refit lowered held-out MSE by 0.00093. No other reduced-model pair passed, and roof-by-covariate pairs did not pass.

## 3. Spread
Spread depends on households. Mean absolute residual is about 0.95 for block groups with fewer than 20 households and about 0.20 above 1,200. Adding log households to the log|residual| model lowers held-out error by 0.040 (SE 0.0025); no other covariate passed. The scale ratio between the 2nd and 98th percentile of households is about 2.8. Residuals are heavy-tailed (excess kurtosis about 8), so intervals use empirical residual quantiles scaled by fitted spread, not a normal distribution.

## 4. Median house value as a driver
The data do not support median house value as a claim-cost driver. With income and occupancy in the model, adding it does not reduce held-out error (drop-one change -0.0001, SE 0.0001). The pricing surcharge rests on the marginal correlation, which income explains. Q8 (house value 150k to 350k) has estimate 0.

## Effect queries (average change in expected y over training rows; 90% bootstrap interval)
| Query | Estimate | 90% interval |
|---|---|---|
| Q1 income 2.5 to 6.0 | -0.48 | -0.51 to -0.45 |
| Q2 housing age 20 to 45 | +0.40 | +0.35 to +0.45 |
| Q3 ave_occup 2 to 4 | +0.23 | +0.20 to +0.25 |
| Q4 roof A to D | +0.38 | +0.36 to +0.41 |
| Q5 ave_rooms 4 to 7 | 0 | 0 to 0 (not in model) |
| Q6 location SF to LA | +0.35 | +0.32 to +0.37 |
| Q7 households 300 to 1200 | 0 | 0 to 0 (not a mean driver) |
| Q8 house value 150k to 350k | 0 | 0 to 0 (not in model) |
Intervals are percentile bootstrap intervals over 100 refits with lam fixed at 0.1. They reflect estimation variance only, not model choice.

## Predictions
Predictions for 5,160 test rows are in predictions.parquet. The 90% interval is the fitted mean plus the empirical 5% and 95% quantiles of out-of-fold residual divided by fitted spread. On training rows, coverage is 0.900 overall, 0.887 to 0.910 across households quintiles, and 0.897 to 0.903 across roof classes. Five test values lay outside the training range (for example one ave_occup of 1,243) and were clipped before prediction.

## Confidence
- Income and roof effects: high. Held-out error rises sharply when they are removed.
- Average occupancy: moderate to high. Smaller effect than income; its shape depends on a few very large values.
- Housing-age rise near 35: moderate. The turn at the top is not reliable.
- Location: moderate. Real in CV; fine structure is not interpretable.
- Housing age by income interaction: moderate. Screen and refit agree; the size is small.
- Households and spread: moderate to high on direction, lower on exact shape.

## What the data cannot tell
- The test rows are a later book of business with a different covariate mix (more extreme occupancy, different roof mix). There are no test outcomes to check the intervals.
- Population, total rooms and households are exact products of households and averages. The model uses the averages and households, so an effect carried by total population alone is not separately identified.
- The census cap at 52 and sparse ages above 45 limit the housing-age shape.
- Missing housing age (7.6% of train, 7.0% of test) was filled from location and income. Missingness is associated with location, so the fill is a model-based estimate.

## What I would check next
- Refit the housing-age effect with a monotone constraint and compare held-out error with the unconstrained curve.
- Test whether households is the only driver of scale, using a location-scale model outside pyGAM.
- Check interval coverage on a holdout of spatial blocks, which resembles the test shift more than random folds.
- Check whether the age-by-income interaction holds within roof classes.
