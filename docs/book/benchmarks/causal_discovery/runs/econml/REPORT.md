# Job-training program: effect on follow-up earnings (EconML, dev data)

Library: econml 0.17.0. Final model: `LinearDML` (double machine learning, linear in the covariates), fitted on the 10,072 applicants who answered the follow-up survey, each weighted by the inverse of their estimated response probability. Numbers from `dev_out/findings.json` (seed 1).

## 1. Average effect of enrolling

- ATE, all 12,000 applicants: **5,247 dollars** (90% interval 4,071 to 6,424).
- ATT, the 3,690 enrolled applicants: **3,760 dollars** (3,283 to 4,237).

The ATT is lower because the effect rises with prior earnings (question 3) and enrolled applicants earn less before the program. The ATE includes 1,015 training applicants (8%) who almost never enrolled; their effects are extrapolated (question 5). The size of the ATE depends on the final model: a quadratic model considered during dev tuning gave 9,839 and was rejected (see tuning_log.csv).

## 2. Employment and earnings channels

- Employment: enrolling raises the probability of working in the follow-up year by **5.5 percentage points** (3.0 to 8.0).
- Earnings of those in work: among employed applicants, enrolled applicants earn **4,219 dollars more** (3,325 to 5,113). This compares employed people only; enrolment can change who is employed, so this is not a clean causal effect on earnings.

Both channels carry part of the effect. The employment channel is the better identified.

## 3. Effect modifiers

No characteristic met the claim rule: the linear model's coefficient must be significant at 95% and agree in sign with a weighted outcome regression with enrolment-by-covariate interactions.

- Education: +794 dollars per SD in the linear model (40 to 1,549); -2,041 (SE 462) in the regression. The signs disagree.
- Prior earnings: +2,601 per SD (-45 to 5,247, not significant); +2,806 (SE 1,186) in the regression.
- Months unemployed: +112 (-609 to 834, not significant); +1,204 (SE 383) in the regression.

Subgroup averages from the linear model: education under 12 years 2,883 (1,831 to 3,935) against 16 or more years 7,576 (5,460 to 9,692); bottom quarter of prior earnings 3,023 against top quarter 8,572 (5,104 to 12,039). The education contrast rests on the term that the regression contradicts, so it is unsettled.

## 4. Roles and adjustment

- Adjust for: age, education, prior earnings, months unemployed, ethnicity, smsa, region. Each affects both enrolment and earnings.
- Experience equals age - education - 6, so age and education cover it; including it makes the model singular.
- Parttime affects earnings but not enrolment; it is included to reduce variance.
- Do not adjust for: certificate and hours_post (mediators, measured after enrolment), responded (measured after enrolment; used only as weights), distance_km (an instrument), caseworker_note_len (no association with enrolment or earnings).

Distance predicts enrolment (first-stage t = -8.4 given covariates) and has no detectable direct link to earnings (t = -0.42). An instrumental-variable estimate using it gives 6,107 (-731 to 12,944): consistent with the adjusted estimate, far less precise.

## 5. Where the data cannot say what the program did

- Prior earnings above about 40,000 dollars: 1,015 training applicants have an estimated enrolment probability below 0.02 (median prior earnings 73,560). The estimates for them rest on extrapolation. 205 new applicants fall in this group.
- Non-response: 16% gave no follow-up earnings, and response depends on enrolment (logit coefficient 0.45). The estimates assume that, given covariates and enrolment, response is unrelated to earnings. The data cannot test this.

## 6. Ranking the new applicants

`cate_predictions.parquet` gives each of the 4,000 new applicants a predicted effect with a 90% interval. Mean predicted effect 4,636 dollars (5th to 95th percentile 1,191 to 8,934). Enrolling the top 800 (20%) gives an expected total gain of about 6.5 million dollars, against about 3.7 million for a random 20%.

Caveat: the ranking follows prior earnings closely (correlation 0.85). Of the top 800, 572 earn more than 40,000 dollars, where the ranking depends on extrapolating the linear trend.

## 7. Sensitivity to an unmeasured characteristic

The same model refit with one confounder removed from the adjustment set:

- Without prior earnings: -4,375 (change -9,622; sign reverses).
- Without months unemployed: 4,875 (change -372).
- Without education: 5,078 (change -169).

An unmeasured characteristic as strong as prior earnings would reverse the conclusion that the program raised average earnings.

## How sure I am

- Employment effect (5.5 points): reasonably sure in sign.
- Average earnings effect (5,247): moderately sure in sign, not in size. The interval reflects sampling error under the model only, not model choice, response, or extrapolation.
- Education, prior earnings and unemployment as modifiers: not sure.
- Ranking: follows the model; not reliable for high earners.

## What I would check next

1. Estimate the ATE on the region with overlap (prior earnings below 40,000) beside the full-population figure.
2. Bound the ATE for worst-case non-response, and model response with the caseworker note and distance as auxiliary information.
3. Test the education interaction with a pre-specified test on held-out data.
