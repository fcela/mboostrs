# Job-training program: effects, who gains, and whom to enrol

Dev data, seed 1. Dollars are follow-up earnings one year after application. Intervals are 90% unless stated.

## 1. Average effects

| Query | Estimate | 90% interval |
|---|---|---|
| ATE, all 12,000 applicants | 4,256 | 3,026 to 5,485 |
| ATT, enrolled applicants | 3,714 | 3,248 to 4,180 |
| Employment probability, ATE | +0.087 | +0.070 to +0.104 |

Enrolment raised follow-up earnings. The ATT is smaller than the ATE: enrolled applicants gained less than the average applicant.

The estimator is an augmented inverse-probability score. Propensity and response propensity come from mboostrs.aipw. Earnings are modelled with one boosted baseline that does not contain enrolment. The arm-specific aipw estimate is 6,421 (5,084 to 7,758). I did not use it: on a no-effect draw I built from the control rows it gave -2,107, with an interval excluding zero, because its top prior-earnings stratum holds about eight enrolled rows.

Confidence is moderate. Two no-effect draws gave -509 and 367, both with intervals covering zero. A 1,500-row subsample gave 2,453 (249 to 4,656). The intervals ignore nuisance-model error.

## 2. Employment and earnings

The effect works mainly through employment: employment rises 8.7 percentage points. Among people in work, mean earnings are 3,528 higher when enrolled (95% interval 2,597 to 4,571).

A decomposition gives an employment part of about 2,130 and an earnings-among-workers part of about 2,300. The second part conditions on employment, which the program affects, so it is descriptive. Both channels are claimed because their 95% intervals exclude zero.

## 3. Who gains more

No characteristic passes the Bonferroni-corrected modifier test, so effect_modifiers is empty. Two patterns are not claimed:

- Prior earnings: the fitted effect is about 2,560 in the lowest tenth and about 2,250 in the highest (p = 0.021; threshold 0.0014). The shape is non-monotone.
- Education: under-12s averaged 6,555 (580 to 12,531), 16-plus averaged 1,712 (166 to 3,259). The full-covariate model was not preferred to prior earnings alone, so this contrast is unexplained by the model and needs more data.

## 4. Roles and adjustment set

| Column | Role | Reason |
|---|---|---|
| age, education, experience | confounder | predict enrolment and earnings |
| prior_earnings | confounder | strongest predictor of both |
| months_unemployed | confounder | predicts both |
| region | treatment_only | predicts enrolment (z 2.7), weak earnings link |
| ethnicity | treatment_only | predicts enrolment (z 4.5), no earnings link |
| smsa, parttime | outcome_only | predict earnings only |
| distance_km | instrument | predicts enrolment (z 5.5), no earnings link |
| caseworker_note_len | irrelevant | neither |
| certificate, hours_post | mediator | measured during or after the program |
| responded | post_treatment_selection | measured after enrolment |

The adjustment set is every column except the instrument, the irrelevant column and the post-enrolment columns. Treatment-only columns are included: adjusting for a pre-enrolment column that affects only enrolment does not bias the estimate, and it guards against a confounder the tests miss in small samples. Dropping them changes the dev ATE from 4,256 to 4,184.

Roles rest on z-tests at a two-SE threshold. Calling distance an instrument needs an exclusion restriction the data cannot test.

## 5. Where the data cannot tell

- Prior earnings of 57,050 or more: 1,200 training rows, 88% outside the propensity bounds 0.02 to 0.98. Enrolment was rare there, so effects rest on very few enrolled rows.
- Overall 9.6% of training rows are outside the bounds. Inside them the ATE is 4,600 (SE 382).
- 114 of 4,000 new applicants have propensity outside the bounds. Their effects are extrapolations, and their covariates are held at the fitted edge.
- Non-respondents (16% of all applicants) have no follow-up earnings. The estimates assume response is missing at random given covariates and enrolment. If response depends on earnings, the estimates are biased, and these data cannot detect it.
- New applicants have lower prior earnings (mean 25,884 against 31,673) and longer unemployment spells.

## 6. Whom to enrol

Each applicant gets an effect and 90% interval in cate_predictions.parquet. Enrolling the top 800 (20%) by effect gives an expected total gain of about 4.69 million dollars, a mean of 5,866 each. Random choice of 800 would give about 3.05 million. None of the top 800 is outside the overlap bounds.

Per-applicant intervals cover the effect model's sampling error only. They do not include model uncertainty, and they are wider near the edge of the overlap region.

## 7. Sensitivity to an unmeasured factor

Dropping prior earnings from the adjustment moves the ATE by -12,172. Dropping any other measured covariate moves it by less than 500. An omitted factor of that size would flip the sign.

The grid expresses the bias as outcome effect times imbalance between enrolled and not enrolled. Cancelling the estimate needs a bias of about 4,300. Among the cells computed, the sign flips only at an outcome effect of 20,000 with an imbalance of 0.5. An unmeasured factor would have to be strong and unevenly spread between arms to change the conclusion.

## 8. What I would check next

1. Run more no-effect draws to measure the coverage of the ATE interval directly.
2. Raise the nuisance step cap above 800, which the propensity and baseline models reached.
3. Test whether the education contrast survives a model with education alone.
4. Ask how survey response was collected, since that decides whether missing-at-random is plausible.
5. Check whether distance is plausibly exogenous before using it as an instrument.
