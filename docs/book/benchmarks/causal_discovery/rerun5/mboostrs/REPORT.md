# Job-training program: effect on one-year follow-up earnings

Dev run: `python run.py --train train.parquet --test test.parquet --out dev_out --seed 1`. Dollars per applicant; intervals are 90%.

Model: cross-fitted augmented IPW (`mboostrs.aipw`) with the propensity formula chosen by out-of-fold deviance (spline plus depth-one trees won), an outcome model pooled with treatment interactions (chosen by the sparse-arm rule), and inverse weighting on response for missing follow-up earnings. Three cross-fitting partitions. Per-applicant effects: smooth R-learner on age, prior earnings and education.

## 1. Average effects

- ATE, all 12,000 applicants: 5,575 (4,318 to 6,831).
- ATT, the 3,690 who enrolled: 3,696 (3,228 to 4,164).

The ATT is lower because enrolled applicants had low prior earnings (mean 17,872 against 37,801), and the effect grows with prior earnings (section 3).

Confidence: moderate in the size, low in the exact value. Other nuisance models gave 5,755 to 6,414. Restricting to applicants with propensity between 0.02 and 0.98 gives 3,986 (standard error 450). An instrumental estimate (distance_km) gives a complier effect of 4,789 (-1,952 to 11,530): consistent with the ATE, too wide to separate.

## 2. Channels

- Employment: enrolment raised the probability of positive earnings by 0.105 (0.089 to 0.121).
- Earnings if employed: log earnings among the employed are 0.153 higher (0.112 to 0.195), about 16%. Enrolment changes who is employed, so this is descriptive. It is claimed because the total effect is distinguishable from zero.

Both channels carry part of the effect; these data cannot split it exactly.

## 3. Who benefits

No covariate passes the claim rule (joint p below 0.05 in the primary fit, median below 0.05 across four nuisance fits, one sign). No modifier is claimed.

Leading candidates (median joint p; coefficient per standard deviation, four fits):

- Age: p = 0.051, -684 to -1,830 (younger larger).
- Prior earnings: p = 0.16, +606 to +5,864 (higher larger).
- Education: p = 0.19, +127 to +949 (more education larger).

Subgroup averages:

- Prior earnings at least 41,330: 11,298 (6,833 to 15,763). At most 15,910: 3,459 (2,943 to 3,975). The largest contrast, but the top group is mostly outside overlap (section 5).
- Age under 25: 4,439 (3,502 to 5,376).
- Education under 12: 7,010 (1,637 to 12,382); 16 or more: 5,464 (3,094 to 7,834). Not established.

An R-learner on the three candidates does not beat a constant by one standard error (gain 153,000, SE 169,000).

## 4. Roles and adjustment

Roles come from conditional independence tests at level 0.05, against enrolment and against follow-up earnings. The instrument is removed from the other columns' conditioning sets, because distance_km proxies smsa.

| Column | Role | Adjusted |
|---|---|---|
| age, education | confounder | yes |
| experience | confounder | no (age minus education minus 6) |
| smsa, prior_earnings, months_unemployed | confounder | yes |
| ethnicity, region | treatment_only | yes |
| parttime | outcome_only | yes |
| distance_km | instrument | no |
| caseworker_note_len | irrelevant | no |
| certificate, hours_post | mediator | no |
| responded | post_treatment_selection | no (response weights only) |
| enrolled; earnings_post | treatment; outcome | |

Dropping prior earnings moves the refit estimate from 6,056 to -5,919 (refit.py).

## 5. Where the data cannot say

- Prior earnings above 45,230: 90% of the 2,400 training applicants there lie outside propensity 0.02 to 0.98. Their effects come from the outcome model.
- No unemployed months: 52% of 1,469 are outside the bounds.
- Non-responders (1,928): the estimates assume response is unrelated to earnings given covariates and enrolment. Response depends on enrolment and prior earnings, so this is the assumption most likely to fail; no data here test it.
- Test applicants have lower prior earnings (mean 25,884 against 31,673); 6.9% are outside the bounds.

## 6. Ranking for next year

cate_predictions.parquet holds cate, lower90 and upper90 for 4,000 test applicants. The ranking is the cate order.

- The top 800 by cate have mean cate 7,896; 267 of them lie outside overlap, so their values are extrapolated.
- The top 800 inside overlap have mean cate 6,593 (4,910 to 8,277).

If the agency will not rely on extrapolation, use the overlap-restricted list. Confidence in either is low to moderate.

## 7. Sensitivity to an unmeasured confounder

The grid assumes an omitted confounder with an effect of 2,629, 6,572 or 13,143 dollars per standard deviation and an enrolled-minus-non-enrolled difference of 0.1 to 0.5 standard deviations. The ATE changes sign only at 13,143 with 0.5 (-967).

Reversing the 5,575 estimate needs effect times difference of 5,575: for example 13,143 per standard deviation with a 0.42 difference. Dropping prior earnings moves the refit by about 12,000, so a hidden confounder of that strength would reverse the sign.

## 8. Checks

- No-effect draw (enrolment permuted within covariate cells): ATE -1,551 (-3,629 to 526); the interval covers zero; no modifier or channel is claimed.
- Dev run: about 3 to 5 minutes on a shared machine.

## Next steps

1. Get the eligibility and caseworker steering rules; compare the propensity model with them.
2. Enrol a small random sample above 40,000 to measure their effect directly.
3. Compare responders and non-responders on pre-programme columns; collect a second earnings measure if possible.
4. Look for a second instrument to check the distance instrument.
