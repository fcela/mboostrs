# Claim-cost index: what the dev data support

Source: dev run (`dev_out/`), mboostrs 0.1.3. Effects are associations with the expected log claim-cost index `y`, other covariates held at observed values. They are not causal effects.

## 1. Associations and their shapes

- **Median income**: decreasing. The expected `y` falls by 0.48 (90% interval -0.50 to -0.46) when income moves from 2.5 to 6.0 (tens of thousands of dollars), and the fall is steepest at low incomes.
- **Housing median age**: flat below about 30 years, rising between about 30 and 45 years, roughly flat after. Moving from 20 to 45 years adds 0.39 (0.37 to 0.43). Ages are top-coded at 52 (5.6% of training rows), so the curve above 45 years rests on a coarse group.
- **Roof class**: categorical. Class D is 0.39 above class A (0.36 to 0.40). Classes B and C fall between them, in the order A < B < C < D.
- **Average occupancy** (persons per household): increasing, convex on the log scale. Moving from 2 to 4 persons adds 0.24 (0.22 to 0.26). The raw variable has heavy outliers (skewness 78), so the model uses its log.
- **Location** (longitude and latitude, one joint surface): moving from (37.8, -122.3) to (34.2, -117.4) adds 0.36 (0.33 to 0.40).

No association with the expected value once the terms above are in: total rooms, total bedrooms, population, households, average rooms, average bedrooms, median house value.

## 2. Interaction

Housing median age and median income interact. The increase from age 20 to 45 is 0.43 at the 25th percentile of income (2.5) and 0.35 at the 75th (4.5). The out-of-fold change for the product term is -0.0023 (standard error 0.0006). No other pair passed.

## 3. Spread

The spread of `y` is not constant. It falls as block groups get larger. The predicted standard deviation is 0.56 at the 10th percentile of households (183), 0.41 at the median (410) and 0.33 at the 90th (882); the out-of-fold improvement from households is 0.080 (standard error 0.003). Population and total rooms track block size and help alone, but add nothing once households is in, so households is the one reported driver. No other covariate changed the spread.

## 4. Median house value (pricing proposal)

The data do not support median house value as a driver of claim cost. The raw association is negative (rank correlation -0.30), but house value and median income are correlated (rank correlation 0.65). Once income is in the model, house value adds nothing out of fold; removing it slightly lowers the risk (-0.0005, standard error 0.0002). The data show that low-income areas cost more, and that high house value does not add to that. The surcharge rests on a correlation that income explains.

## 5. Predictions for the test rows

Each test row has a mean and a 90% interval: mean plus or minus q times the predicted standard deviation, with q calibrated on 2,322 held-out training rows. Coverage on 2,322 other training rows is 0.91 overall and 0.89 to 0.93 within household quartiles, roof classes and income terciles. Mean interval width is 1.39.

The test book has higher incomes: mean median income is 4.54 against 3.65 in training (standardized difference 0.54). The intervals are not reweighted for this shift. A few test rows lie just outside the training range of occupancy or households and are predicted with the edge value held flat.

## 6. How sure I am

- **Income, roof class, housing age, household-count spread**: strong. Each out-of-fold change is well above three standard errors. Shapes are less certain than existence.
- **Occupancy**: moderate. Change 0.0098 (standard error 0.0008); its shape depends on the log scale.
- **Location**: clearly needed (change 0.0067, standard error 0.0005); its detailed shape is not identified.
- **Housing age by income interaction**: moderate; it did not appear on an 800-row subsample, so treat it as a lead.
- **Effect-query intervals**: about plus or minus 0.02. They condition on the selected terms and omit selection uncertainty, so they probably understate the true uncertainty.

## 7. What the data cannot tell

- Causation. Every number is an association.
- Shapes above the top-coded values: housing age is capped at 52 and house value at 500,001.
- Whether missing housing age (7.6% of training, 7.0% of test rows) is unrelated to claim cost; it is filled with the median plus an indicator term.
- The local form of the location surface, and whether the roof effect differs by region.

## 8. Next checks

1. Run the pipeline on another draw and compare the selected terms and effect queries.
2. Refit the age-by-income interaction without the top-coded rows.
3. Test a power-law block-size scale for the spread, to confirm the functional form.
4. Recompute effect-query intervals by refitting on resampled rows, which includes selection.
5. Check coverage on a higher-income subsample, the shift the test book shows.
