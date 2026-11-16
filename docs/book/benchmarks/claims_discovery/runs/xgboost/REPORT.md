<!-- Saved by the coordinator from the analyst's final message, verbatim: the agent harness refused this analyst's write of REPORT.md ("Subagents should return findings as text, not write report files"). -->

# Claim-cost index: XGBoost findings (dev data, seed 1)

## Model
XGBoost 3.4.0 with depth-2 trees, 350 rounds, min_child_weight 50, learning rate 0.05, chosen by 5-fold CV. Out-of-fold RMSE 0.470 (R2 0.40; SD of y 0.605). Depth 2 had the lowest CV error (0.4703); depth 3 gave 0.4715 and depth 6 gave 0.4758.

Partial dependence (PD) is the average predicted y when one covariate is set to a fixed value for every training row, with the other covariates at their observed values. A change in PD is a change in expected y.

## 1. Covariates associated with the expected value
- **Median income: decreasing.** PD falls from 0.49 at income 2.5 to 0.00 near 6.4 and to -0.09 at 7.9 (tens of thousands of dollars). Strongest association: permuting it raises held-out squared error by 0.101.
- **Housing median age: threshold increase.** PD is flat near 0.18 up to age 24, then rises: 0.25 at 32, 0.38 at 35, 0.51 at 39, 0.56 at 46. The rise is centred at 32 to 36 years.
- **Average occupancy: increasing, levelling off.** PD is 0.15 at 1.6 persons per household, 0.26 at 2.5 and 0.45 at 4.3.
- **Roof class: categorical.** PD is 0.20 for A, 0.21 for B, 0.36 for C, 0.56 for D.
- **Location: spatial.** Permuting longitude and latitude together raises held-out error by 0.011. Moving along the 5th-to-95th percentile diagonal changes expected y by 0.40.
- **Median house value: small decrease.** PD falls from 0.33 to 0.27 across the range (0.06, or 0.10 SD of y), at the lower edge of the practical threshold used here.

Not associated after correction: total rooms, total bedrooms, population, households, average rooms, average bedrooms.

## 2. Interactions
One pair: housing age with median income. The pair term lowers held-out error by 0.0017 (0.8% of the additive model's error; Bonferroni p = 0.003). The global test that the full model beats the additive model gives p = 0.0006. The direction and size of the age-by-income difference were not characterised in this run.
Six pairs were tested, chosen by interaction strength: longitude by latitude, occupancy by roof class, income by roof class, income by occupancy, age by house value. None reached significance. Other pairs were not tested.

## 3. Spread of y
The spread is not constant. It falls as household count rises: the log squared residual drops by 0.85 between the 5th and 95th percentile of households, a residual SD ratio of about 1.5. Total bedrooms and total rooms pass the statistical test, but their spread curves change by less than the practical threshold. These counts move with households, so block-group size is the likely signal; households is the strongest of them. No other covariate changes the spread materially.

## 4. Median house value and the surcharge
The data do not support house value as a driver of higher claim cost.
- The raw correlation of house value with y is -0.28: pricier block groups have lower y on average.
- House value correlates 0.66 with income, and income is the stronger driver.
- With other covariates held, moving house value from 150,000 to 350,000 changes expected y by -0.06 (90% interval -0.08 to -0.04). The sign is opposite to the surcharge rationale. A surcharge would charge more where the model expects lower claims.

## 5. Predictions
Test predictions are in predictions.parquet. The 90% interval is weighted split-conformal: out-of-fold residuals scaled by a fitted spread model, with calibration weights for a mild train-test shift (adversarial AUC 0.60). Median width 1.30 (10th to 90th percentile 1.09 to 1.64). Out-of-fold coverage is 0.90 overall, and 0.88 to 0.91 across roof class, income thirds and household thirds (lowest household third 0.88). Test coverage cannot be checked; the test rows have no y.

## Effect queries
Estimate: mean over training rows of the change in predicted y. Interval: 5th to 95th percentile of 30 bootstrap refits.

| Query | Change | Estimate | 90% interval |
|---|---|---|---|
| Q1 | income 2.5 to 6.0 | -0.480 | -0.565 to -0.445 |
| Q2 | housing age 20 to 45 | 0.384 | 0.338 to 0.402 |
| Q3 | occupancy 2.0 to 4.0 | 0.220 | 0.191 to 0.235 |
| Q4 | roof class A to D | 0.362 | 0.346 to 0.381 |
| Q5 | average rooms 4.0 to 7.0 | 0.006 | -0.018 to 0.017 |
| Q6 | location SF to LA | 0.264 | 0.226 to 0.281 |
| Q7 | households 300 to 1200 | 0.000 | -0.007 to 0.010 |
| Q8 | house value 150k to 350k | -0.060 | -0.077 to -0.039 |

## How sure
- High: income, housing age, roof class, occupancy (held-out t above 10 across 10 folds). Location is strong on the joint test, but its shape is known only along one diagonal.
- Moderate: age-by-income interaction (significant, small).
- Low to moderate: house value decrease (Bonferroni p = 0.002, at the threshold).
- Moderate: household count as the spread driver. The signal is strong; which count variable is the true driver is not settled.
- Query intervals reflect refit variability on this training set. They exclude sampling noise and model-form error, so they are likely too narrow.

## What the data cannot tell
- These are associations within this book, not causal effects. Location and house value may proxy for unobserved risk.
- Shapes come from a depth-2 boosted model; thresholds and steep smooth curves are hard to separate.
- Correlated covariates share effects; permutation tests credit whichever one the model uses.
- Changes in how y depends on x in the test book cannot be detected without test y.

## Next checks
1. Fit a penalised-spline model to confirm the age threshold and income shape.
2. Plot the age effect within income bands.
3. Test households against population and total bedrooms for spread, conditioning on each other.
4. Refit the house value effect without income, and on a later sample if one exists.
5. Use repeated-split intervals for the effect queries.

Robustness: on an 800-row draw, the age, income, occupancy and roof effects are found; location and house value are not. With y permuted, no mean effect, interaction or spread driver is flagged, and all effect-query estimates are within 0.01 of zero.
