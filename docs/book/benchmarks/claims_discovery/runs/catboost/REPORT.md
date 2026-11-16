<!-- Saved by the coordinator from the analyst's final message, verbatim: the agent harness refused this analyst's write of REPORT.md ("Subagents should return findings as text, not write report files"). -->

# Claim-cost index: what drives it

Model: CatBoost regression trees (depth 4, L2 10, 170 trees, learning rate 0.1) on 15,480 training rows. Five-fold out-of-fold RMSE 0.470; SD of y 0.605. Source: dev_out/ from run.py --seed 1.

## 1. Covariates associated with the expected value

Four covariates and one spatial term pass the association test (held-out permutation, z at least 4). Shapes come from partial-dependence curves of the final model.

- Housing median age: threshold increase. Flat at 0.19 to 0.21 up to about 29 years; rises from 0.23 at 32 to 0.54 at 42; levels off near 0.56 above 45. The cut point lies between 32 and 36 years.
- Median income: decreasing, steepest at low incomes. 0.76 at 1.2 (tens of thousands of dollars), 0.13 at 4.7, -0.09 at 7.9.
- Average occupancy: increasing. 0.15 at 1.6 persons, 0.37 at 3.6, 0.45 at 4.9.
- Roof class: categorical. A 0.20, B 0.21, C 0.37, D 0.59. A and B are indistinguishable.
- Longitude and latitude: spatial. Joint z = 13.7. Higher in the east (0.47 at longitude -117.7, 0.18 at -122.8) and in the north above 38 degrees (0.40 at 38.6, 0.32 at 35 to 37).

Not associated: total rooms, total bedrooms, population, households, average rooms, average bedrooms.

## 2. Interactions

None established. The strongest candidate, housing median age with median income, has pair test t = 2.9 (cut 3.0). The test fits a depth-4 model to the residuals of an additive model. Longitude with latitude shows no further interaction (t = -1.7). Power for weak interactions is unknown, so a weak interaction could be missed.

## 3. Spread of y

Spread is largest in small block groups. The predicted mean absolute residual is 0.48 at 60 households and 0.28 at 1,574. Households are a variance driver (z = 7.7): spread decreases as households increase. Total rooms (z = 3.1), total bedrooms (z = 2.8) and population (z = 2.3) fall below the cut. They correlate with households (0.93 for log population against log households), so the household effect may stand in for block size. Income, age, house value, occupancy and roof class are not called (z below 2.1).

## 4. Median house value as a driver

The data do not support house value as a driver.

- The raw correlation with y is -0.28. House value correlates 0.66 with income, which accounts for most of the raw association.
- With the other covariates in the model, house value reduces squared error by 0.0015 (z = 3.5), about 0.7% of residual mean squared error. Below the cut of z = 4, so not called.
- Q8 (150,000 to 350,000) is -0.07, interval -0.10 to -0.05. The sign is opposite to the surcharge rationale. Q8 comes from a model that keeps house value as an input, so its interval describes that model and does not show a driver.

A surcharge on high-value areas is not supported. Testing a value effect needs a later book with house value and income both recorded.

## 5. Effect queries

Each estimate is the average change in the prediction over the 15,480 training rows. Intervals are the 5th and 95th percentiles of 40 bootstrap refits with fixed settings.

Q1 income 2.5 to 6.0: -0.488 (-0.566, -0.420)
Q2 housing age 20 to 45: 0.366 (0.336, 0.391)
Q3 occupancy 2.0 to 4.0: 0.214 (0.181, 0.239)
Q4 roof A to D: 0.392 (0.369, 0.406)
Q5 average rooms 4 to 7: 0.007 (-0.018, 0.026)
Q6 (37.8, -122.3) to (34.2, -117.4): 0.245 (0.215, 0.282)
Q7 households 300 to 1,200: -0.004 (-0.018, 0.007)
Q8 house value 150k to 350k: -0.071 (-0.100, -0.049)

The intervals cover refit variability only, not tuning, model choice or shape bias. Q5 and Q7 include zero.

## 6. Prediction intervals

Each test row gets mean +/- q x s(x). s(x) is the predicted absolute residual from a second CatBoost model. q = 2.10 is the 90% quantile of |residual| / s(x) over out-of-fold training residuals (scaled split-conformal).

Out-of-fold coverage: 0.900 overall; 0.895 to 0.908 by roof class; 0.899 to 0.903 by spread tercile. Mean test interval width 1.38. Test coverage cannot be computed (no y). Test rows have higher income (standardised difference 0.54) and house value (0.33), so coverage on the later book may differ.

## 7. How sure I am

- Age, income, occupancy, roof class, spatial term: strong (z 10 to 35). Income and occupancy shapes were stable across runs. The age cut point is approximate.
- Households and spread: moderate (z = 7.7).
- Interactions: weak; age with income is borderline (t 2.9 final, 3.1 earlier) and not called.
- Permuted-outcome draw (no structure): no associations, interactions or spread drivers; the model stopped at 3 trees.
- 800-row subsample: age, income and roof class found; longitude, latitude and occupancy missed.

## 8. What the data cannot tell, and next checks

The data cannot establish causal effects, and tree models may approximate the outcome poorly. The test book has higher incomes and house values than training, so effect queries describe the training population. Next: fit a smooth additive model to check the age cut point and income curve; repeat the interaction and spread tests over several seeds and fold splits; check test coverage once outcomes exist; map the spatial term against geography.
