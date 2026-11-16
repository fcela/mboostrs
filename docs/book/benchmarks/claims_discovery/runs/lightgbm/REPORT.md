# Claim-cost index: LightGBM analysis

Data: 15,480 training and 5,160 test block groups. Model: LightGBM 4.7.0 boosting, tuned by 5-fold CV. Results are from the dev run.

## 1. Covariates associated with the expected value

A covariate passes when refitting without it raises held-out squared error by at least 0.5% (paired CV, Bonferroni over 13 groups).

| Covariate | Shape | Partial effect on E[y] | MSE change |
|---|---|---|---|
| `median_income` | decreasing | 0.76 at 1.2 to -0.07 at 7.9 | 6.6% |
| `housing_median_age` | increasing | 0.18 at 5 years to 0.57 at 52 | 5.8% |
| `roof_class` | categorical | mean A 0.21, B 0.21, C 0.36, D 0.56 | 6.7% |
| `longitude`, `latitude` | spatial (tested jointly) | 0.24 across the longitude range, 0.14 across latitude | 3.0% |
| `ave_occup` | increasing | 0.15 at 1.6 persons to 0.45 at 4.9 | 0.8% |

`ave_occup` is real on the dev data (p = 1e-4) but small, and it is close to the cutoff.

Not associated: `total_rooms`, `total_bedrooms`, `population`, `households` (tested as a collinear cluster), `ave_rooms`, `ave_bedrooms`, `median_house_value`.

## 2. Interactions

An unrestricted model beats an additive model by 2.1% of held-out MSE. With y permuted, the same comparison gives gains within 0.1%.

- `housing_median_age` x `median_income`: 1.5% of additive MSE (p = 4e-7). This is the largest pair.
- Nine further pairs, each 0.5% to 0.9% of MSE, with p between 1e-7 and 1e-3. They involve `ave_occup`, `roof_class`, `median_income`, `housing_median_age` and location.

The pairs share variables, so the test does not identify a single interacting pair.

## 3. Spread of y

The out-of-fold residual SD falls as block groups get larger: 0.66 in the lowest total-rooms quintile, 0.33 in the highest. Four correlated counts (total rooms, total bedrooms, households, population) carry this pattern. Tested as one cluster, they raise the log-residual loss by 3.0% (p = 5e-9).

The data cannot identify which count drives the spread, so all four are listed with the same direction.

## 4. The surcharge proposal

The data do not support `median_house_value` as a driver of claim cost. Its marginal Spearman correlation with y is -0.31: expensive areas have lower claim cost here. Given the other covariates, removing it changes held-out MSE by 0.02% (p = 0.39), so it adds no predictive information beyond income.

The effect query for 150,000 to 350,000 is -0.06 (90% interval -0.08 to -0.04). This is a model contrast that moves house value away from the other covariates it correlates with (log house value and income have r = 0.65), so it is extrapolation. A surcharge based on high house value is not supported.

## 5. Test predictions

`predictions.parquet` holds a mean and a 90% interval per test row. The interval is conformalized quantile regression (CQR) calibrated on out-of-fold predictions.

- Out-of-fold coverage is 0.900 overall, 0.896 to 0.903 by roof class, and 0.892 to 0.905 by predicted-mean tercile.
- Mean width is 1.41.
- Test covariates are shifted from training: standardized mean difference +0.54 for `median_income` and +0.33 for `median_house_value`. Coverage on the test book has not been checked.

## Effect queries

Average change in E[y] over the training rows. The 90% interval comes from 60 bootstrap refits.

| Query | Change | Estimate | 90% interval |
|---|---|---|---|
| Q1 | median_income 2.5 to 6.0 | -0.457 | -0.489 to -0.416 |
| Q2 | housing_median_age 20 to 45 | 0.379 | 0.359 to 0.402 |
| Q3 | ave_occup 2.0 to 4.0 | 0.216 | 0.167 to 0.238 |
| Q4 | roof_class A to D | 0.355 | 0.329 to 0.369 |
| Q5 | ave_rooms 4.0 to 7.0 | -0.008 | -0.030 to 0.013 |
| Q6 | (lat, lon) to (34.2, -117.4) from (37.8, -122.3) | 0.253 | 0.206 to 0.267 |
| Q7 | households 300 to 1,200 | -0.003 | -0.010 to 0.015 |
| Q8 | median_house_value 150,000 to 350,000 | -0.063 | -0.081 to -0.041 |

## How sure I am

- Income, housing age, roof class and location: strong. Each has p below 1e-6 on the dev data.
- `ave_occup`: moderate. It passes the rule, but the effect is small.
- Interactions: the overall interaction is clear; the individual pairs are weak evidence.
- Spread: the size pattern is strong; the driving variable is not identified.
- Intervals: coverage matches 90% on the training folds. The test shift is the main risk.

Stress checks: with y permuted, the run reports no findings and effect queries within 0.005 of zero. On an 800-row subsample, it recovers housing age, income, roof class and location, reports no interactions, and flags `roof_class` for spread (p = 0.002), which the full dev data does not show.

## What the data cannot tell

These are associations with the simulated outcome, not causal effects. The p-values come from overlapping CV folds and are approximate. Roof class is unordered, so A < B < C < D is a label.

## Next checks

- Interval coverage on the later book, once outcomes arrive.
- The four size variables against a single size index.
- Interaction stability across more seeds.
