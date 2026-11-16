# Claim-cost index: what drives it, and how sure we are

Dev data: 15,480 training and 5,160 test block groups. Model: Gaussian location-scale boosting
(mboostrs 0.1.3). The location is the expected value of `y`; the scale is its spread. Intervals are
90%. These are associations, not causal effects.

## 1. Covariates that move the expected value

| Covariate | Shape | Change, 5th to 95th percentile | Evidence |
|---|---|---|---|
| Median income | decreasing | −0.73 (−0.76 to −0.69) | Dropping it raises out-of-fold NLL by 0.033 (SE 0.001) |
| Housing median age | threshold increase | +0.37 (+0.33 to +0.40) | Rise between about 25 and 45 years; drop costs 0.037 (SE 0.002) |
| Average occupancy | increasing | +0.29 (+0.27 to +0.31) | Drop costs 0.015 (SE 0.001) |
| Roof class | categorical | D vs A +0.38; C vs A +0.16; B vs A +0.01 | Drop costs 0.037 (SE 0.001) |
| Longitude and latitude | spatial | Joint surface; point range across the 5×5 box 0.49 | Beats separate smooths by 0.003 (SE 0.001) |

Not associated: median house value, average rooms, population, total rooms, average bedrooms,
households (as a mean term), and the missing-value indicators. A term stayed only if dropping it
raised out-of-fold NLL by more than two standard errors.

Confidence is high for income, roof class, age and the spatial surface. Occupancy is clear, but its
shape above 3 persons per household rests on few rows. The age threshold location is approximate.

## 2. Interactions

The effect of housing age depends on median income. The 5th-to-95th percentile change in age is
+0.51 (+0.45 to +0.55) at the 10th income percentile and +0.24 (+0.19 to +0.30) at the 90th. The
improvement over the additive model is 0.0029 out-of-fold NLL (SE 0.0005). Confidence: moderate.

The joint longitude-latitude surface also beats the additive terms (0.0031, SE 0.0007). Its draw
intervals are wide at the corners of the box, where training rows are sparse.

## 3. Spread

The spread falls as the block group gets larger. The standard deviation at the 95th percentile of
households (1,151) is 0.44 times its value at the 5th percentile (124); the 90% interval of the log
ratio is −0.84 to −0.79. Against a constant spread, the out-of-fold NLL gain is 0.083 (SE 0.004).
Population and income added nothing to the scale in exploration. Confidence: high.

## 4. Does median house value drive claim cost?

Not beyond income. Its raw correlation with `y` is negative (−0.28), but that comes through income.
With income, age, occupancy, location and roof class in the model, dropping house value lowers
out-of-fold NLL by 0.0005 (SE 0.0002). The house-value query (Q8) is zero under the final model.
The data do not support a surcharge based on house value. Confidence: moderate to high.

## 5. Predictions and intervals

`predictions.parquet` gives the mean and 90% interval for each test row. The interval is
mean + q₀.₀₅·σ to mean + q₀.₉₅·σ, where σ is the fitted spread and the quantiles (−1.55 and +1.55)
come from out-of-fold standardized residuals. Out-of-fold coverage on training rows is 0.90 overall,
and between 0.88 and 0.91 in every household quintile, roof class and missing-value group. The mean
width is 1.38.

The test rows are a different book. Median income is 4.04 against 3.38 in training, with significant
shifts in house values and room counts. The intervals assume the same relation between covariates
and `y` on the test rows, so their coverage there is unknown. One test row has average occupancy
above the training maximum; its prediction extrapolates the spline by the edge slope.

## 6. Effect queries (average change in expected value over training rows)

| Query | Change | Estimate | 90% interval |
|---|---|---|---|
| Q1 | income 2.5 → 6.0 | −0.47 | −0.49 to −0.45 |
| Q2 | housing age 20 → 45 | +0.39 | +0.36 to +0.42 |
| Q3 | occupancy 2.0 → 4.0 | +0.24 | +0.22 to +0.25 |
| Q4 | roof class A → D | +0.38 | +0.35 to +0.40 |
| Q5 | average rooms 4 → 7 | 0 | 0 to 0 |
| Q6 | (37.8, −122.3) → (34.2, −117.4) | +0.38 | +0.35 to +0.42 |
| Q7 | households 300 → 1200 | 0 | 0 to 0 |
| Q8 | house value 150k → 350k | 0 | 0 to 0 |

Intervals come from REML parameter draws. Q5, Q7 and Q8 are zero because those covariates are not
in the selected location model; their zero-width intervals do not include the chance that the
selection was wrong. Boosted point estimates agree within 0.01.

## 7. Limits

- The intervals treat the selected terms as known, so they are narrower than the selection
  uncertainty.
- The test shift is large; the selection rules and calibration may not hold on the later book.
- The surface is estimated only where block groups exist.
- Out-of-fold NLL was flat from 1,200 to 1,600 boosting steps (0.574897 and 0.574876); 1,600 is the
  top of the search grid.

## 8. Next checks

1. When later outcomes exist, score the model on them and recalibrate if coverage falls.
2. Refit with occupancy clipped at the 99.5th percentile to see whether the extreme row moves any estimate.
