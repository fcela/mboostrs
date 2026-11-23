# Claim-cost index: what drives it

Source: dev training data (15,480 block groups). Run: seed 1, dev_out/. Library: mboostrs 0.1.3.

## Summary

- The expected claim-cost index rises with housing-stock age from about 30 to about 46 years, with average occupancy, and with the insurer's roof class (highest for class D). It falls with median income and varies by location.
- Residual spread is largest for small block groups and falls as the number of households rises.
- Median house value has no detectable association with claim cost once income and location are in the model. The data do not support the surcharge.

## Method

A term is kept only when removing it costs more than two standard errors of out-of-fold score. The spread is modelled jointly with the mean (Gaussian location-scale model), and every covariate was tested as a spread driver. Effect intervals come from a restricted-maximum-likelihood refit of the chosen terms. The effects are associations, not interventions.

## Question 1: associations and shapes

| Covariate | Shape | Evidence |
|---|---|---|
| Median income | Decreasing, flattening above about \$50,000 | Mean changes by -0.93 from the 2nd to 98th percentile |
| Housing median age | Flat up to about 28 years, rising to about 46, then flat | Rises by +0.33 across the range |
| Average occupancy | Increasing, flattening at high values | +0.38 across the range; removal costs 0.020 (SE 0.001) |
| Roof class | Categorical: B close to A, C +0.17, D +0.39 relative to A | Removal costs 0.040 (SE 0.002) |
| Longitude and latitude | Spatial surface (not additive) | Removal costs 0.003 (SE 0.001) |
| Housing age missing | Missing raises the mean by +0.07 | Removal costs 0.001 (SE 0.0002) |

Not associated with the mean once the terms above are in: median house value, average rooms, households, population, total rooms, total bedrooms.

## Question 2: interactions

- Longitude and latitude: the location effect is not additive; the north-south pattern differs by longitude. Score gain 0.003 (SE 0.001).
- Housing age and median income: the age effect differs by income. Score gain 0.003 (SE 0.001).

A roof-class by missing-age interaction proposed by the search was rejected by the check; it made the out-of-sample score worse by 0.06.

## Question 3: spread

Block-group size drives the spread. Log households lowers the out-of-fold joint score by 0.065 (SE 0.003). Residual standard deviation by household quartile, smallest to largest: 0.64, 0.45, 0.40, 0.32. The spread at the 95th percentile of households is about 45% of the spread at the 5th.

Population, total rooms and total bedrooms give almost the same gain (within 0.005), so the data cannot say which size measure is the cause. After households, none of the next three candidates adds to the spread model.

Confidence: high that spread falls with block-group size; moderate on the specific measure.

## Question 4: median house value

No support. Its raw correlation with the outcome is negative (Pearson -0.28). The association disappears once income and location are in the model. Moving house value from 150,000 to 350,000 changes the expected index by -0.015 (90% interval -0.031 to +0.001).

## Effect queries

Average change in the expected value over the 15,480 training rows, other columns held at observed values. 90% intervals from posterior draws of the refit.

| Query | Change | Estimate | 90% interval |
|---|---|---|---|
| Q1 | median income 2.5 to 6.0 | -0.47 | -0.49 to -0.46 |
| Q2 | housing age 20 to 45 | +0.41 | +0.38 to +0.43 |
| Q3 | average occupancy 2 to 4 | +0.24 | +0.22 to +0.25 |
| Q4 | roof class A to D | +0.38 | +0.37 to +0.40 |
| Q5 | average rooms 4 to 7 | +0.02 | 0.00 to +0.04 (not selected) |
| Q6 | (37.8, -122.3) to (34.2, -117.4) | +0.30 | +0.27 to +0.32 |
| Q7 | households 300 to 1200 | -0.003 | -0.016 to +0.010 (not selected) |
| Q8 | median house value 150k to 350k | -0.015 | -0.031 to +0.001 (not selected) |

Q5, Q7 and Q8 are not selected by the rule; their estimates come from a refit with the covariate added, and the intervals are widened to include zero. Q5 should not be read as an effect.

## Prediction intervals

90% intervals for the 5,160 test rows from distributional conformal prediction. Mean width 1.40 (median 1.33; range 0.57 to 4.05), wider for small block groups. Five-fold cross-fitting on the training rows gives held-out coverage of 0.898 overall, and 0.89 to 0.91 across roof class, household tertile, latitude tertile and income tertile.

The test rows are a later book of business. Average rooms are higher (5.7 against 5.3 in training), and one row each exceeds the training range for occupancy and for households. Coverage on the test rows is not known.

## Confidence

- High: income, housing age, occupancy, roof class, and the size dependence of the spread.
- Moderate: the two interactions (3 to 4 standard errors) and the shape of the location surface.
- Low: average rooms, households in the mean, median house value.

## What the data cannot tell

- The effects are associations. They do not show what a pricing action would change.
- Housing age is missing for 7.6% of rows, and that missingness is associated with location (correlation 0.18 with latitude). Its +0.07 effect on the mean may reflect location.
- The spread finding does not establish a mechanism.
- On an 800-row subsample the rule keeps fewer terms and the stopping iteration falls to about 15. Results on small data are noisier and should not be quoted as established.

## Next checks

1. Compare the size measures for the spread on a new sample.
2. Plot the location surface and the age-income interaction before quoting values from them.
3. Test whether missing housing age relates to claim cost beyond location.
4. Check interval coverage on the later book of business, especially for large block groups and high-occupancy rows.
