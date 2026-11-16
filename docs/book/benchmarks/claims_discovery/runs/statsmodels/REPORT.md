<!-- Saved by the coordinator from the analyst's final message, verbatim: the agent harness refused this analyst's write of REPORT.md ("Subagents should return findings as text, not write report files"). -->

Claim-cost index: drivers, spread and predictions

Source: dev training data (15,480 block groups), statsmodels 0.15.0. y is a log claim-cost index, so a change of 0.10 in y is about a 10% change in expected claim cost.

Answers in brief
1. Four covariates and a location surface are associated with the expected index: median income (down), housing median age (up, with a threshold), average occupancy (up, then flat), roof class (up, D highest) and location.
2. Yes, one pair: the effect of income depends on housing age.
3. Yes. The spread of y is larger in small block groups. Households carries it.
4. No. After adjusting for income, age, roof class and location, median house value shows no detectable association. The surcharge proposal is not supported by these data.
5. Predictions and 90% prediction intervals are in predictions.parquet (5,160 rows).

Question 1: mean associations. Each value is the average fitted y over training rows, other covariates at observed values; ranges are 1st to 99th percentile.
- Median income (tens of thousands of dollars): decreasing, steepest at low incomes. Average fitted y is +0.86 at income 1 ($10k), +0.36 at 3, +0.07 at 5, -0.17 at 9. Drop-one F = 160 on 20 df (with its interaction), p < 1e-300.
- Housing median age (years): threshold increase. Flat to about 25 years (within 0.01 of its value at 5 years), a rise of about 0.32 between 30 and 45 years, then flat above about 45. The threshold position is read from the curve and has no interval. p = 2e-222.
- Average occupancy (persons per household): increasing, then levelling off. Fitted y is +0.19 at 2 persons, +0.43 at 4, about +0.44 above 5. p = 1e-93.
- Roof class: categorical. Average fitted y is +0.19 for A, +0.21 for B, +0.37 for C, +0.58 for D. p = 2e-211.
- Location (longitude and latitude, one joint surface): not additive in the two coordinates. p = 3e-89.
Not associated after adjustment: median house value, total rooms, population, households, average rooms, average bedrooms. Population, households, total rooms and average bedrooms each have p between 0.004 and 0.043 when added alone to the final model (average rooms 0.12), but none passes selection together with the others. These counts are strongly correlated with average occupancy, and the data cannot say which of them carries any effect.

Question 2: interaction. Income and housing age interact (Holm-adjusted p = 2e-18). Moving income from its 10th to 90th percentile (about $18k to $58k) changes average fitted y by -0.52 where housing age is at its 10th percentile (13 years) and by -0.78 where housing age is at its 90th percentile (46 years). The income gradient is steeper in older housing areas. A roof-by-income interaction was significant in the pair screen (Holm p = 2e-5) but was removed by the second elimination, so it is not in the final model; treat it as a candidate. Occupancy-by-income (Holm p = 0.03) was also not kept.

Question 3: spread. The spread of y falls as block groups get larger. Predicted standard deviation is 0.85 for a block group with 32 households (1st percentile) and 0.22 for one with 1,920 households (99th percentile), so the 90% interval is about 3.8 times wider in the smallest block groups. Households is the only term in the spread model. The 800-row subsample picked population instead, which is expected given the correlation between the two.

Question 4: median house value. The marginal association is negative (correlation -0.29): expensive areas in this file have lower claim cost. Holding income, housing age, roof class and location fixed, the linear association is near zero. Moving from $150k to $350k changes average fitted y by -0.01, 90% interval -0.03 to +0.01 (y is in log units, so at most about 3% either way under the linear form). Adding house value to the final model gives F = 1.9 on 4 df, p = 0.12. The data do not support a surcharge based on house value.

Effect queries: average change in the fitted mean over training rows; 90% bootstrap intervals (300 refits, selected terms fixed).
| Query | Change | Estimate | 90% interval |
| Q1 | median income 2.5 to 6.0 | -0.467 | -0.488 to -0.445 |
| Q2 | housing age 20 to 45 | +0.372 | +0.352 to +0.394 |
| Q3 | occupancy 2.0 to 4.0 | +0.244 | +0.227 to +0.264 |
| Q4 | roof A to D | +0.386 | +0.363 to +0.408 |
| Q5 | rooms 4.0 to 7.0 | 0 | not in the mean model |
| Q6 | (lat, lon) 37.8,-122.3 to 34.2,-117.4 | +0.272 | +0.238 to +0.302 |
| Q7 | households 300 to 1,200 | 0 | not in the mean model |
| Q8 | house value 150k to 350k | 0 | not in the mean model |
Q5, Q7 and Q8 are zero because those variables are not in the selected mean model, not because a zero was measured. Households affects spread, not the mean.

Prediction intervals: the interval is the mean plus the 5% and 95% standardised-residual quantiles, scaled by sigma(households). On training rows it covers 90.0% of observations; coverage is 0.89 to 0.91 across household quintiles, roof classes and rows with and without missing housing age. These are in-sample checks; test outcomes are not available. The test book is richer: its median income is 0.44 standard deviations above the training income, so the mean prediction is lower (0.20 against 0.31). The income effect is linear beyond the outer knots, so predictions at the highest incomes are extrapolations.

Limits
- The associations are conditional correlations in one data set, not the effects of changing a block group's properties.
- The selection step chooses one set of terms. Effect-query intervals do not include uncertainty from that choice or from tuning.
- Missing housing age (7.6% of training rows, 7.0% of test rows) is imputed by regression. Rows with missing age sit 0.06 above their imputed prediction (90% interval 0.03 to 0.08). The cause is not known.
- The location surface has no mechanism attached and is used for prediction only within the observed area.

Next steps
1. Score the test predictions and intervals when outcomes arrive, with coverage by income band.
2. Fit population, households and total rooms as one pre-specified block with and without occupancy to resolve the count question.
3. Establish how the housing-age missingness arose, and check it against the source.
4. Retest the roof-by-income interaction on a larger sample before using it for pricing.
