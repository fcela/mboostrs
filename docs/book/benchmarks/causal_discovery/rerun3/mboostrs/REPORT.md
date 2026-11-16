# Job-training program: effect on follow-up earnings

Source: `run.py --seed 1`, dev files. Intervals are 90%. Dollars are earnings one year after the program year.

## 1. Average effects

| Query | Estimate | 90% interval |
|---|---|---|
| ATE, all applicants | +4,930 | 3,433 to 6,427 |
| ATT, enrolled applicants | +3,774 | 3,213 to 4,334 |
| EMP_ATE, change in probability of any earnings | +0.098 | 0.077 to 0.120 |

Method: cross-fitted augmented inverse-probability weighting on the pre-application covariates in section 4, with propensities clipped to 0.01 to 0.99. Non-respondents (16%) are weighted by an inverse response model.

The ATE depends on specification: a spline baseline gives 5,859, a regression-only estimate 3,043, and the overlap-weighted ATE 4,128. The ATT stays between 3,740 and 3,780. I would report the ATE as roughly 3,000 to 6,000, centred near 4,900.

## 2. Channels

Employment: yes. The employment interval excludes zero.

Earnings of people in work: yes, under an assumption. Employment alone accounts for 0.098 times 21,900 dollars, the earnings of employed controls who resemble the enrolled, or about 2,150. The remainder is 2,774 (1,660 to 3,889). It assumes the people moved into work earn what similar employed controls earn. If they earn more, part of the remainder is an artefact.

## 3. Effect modifiers

Education: the effect is larger for higher education. Inside the region where enrolment happened, the effect is about 2,800 under 12 years, 3,800 for 12 to 15 years, and 6,000 for 16 or more. Univariate test p = 0.0024, against a Bonferroni threshold of 0.0063 over eight candidates.

Prior earnings (p = 0.014) and months unemployed (p = 0.040) move the effect alone, but not once education is in the model (joint p = 0.065 and 0.7). Age (p = 0.096) is not claimed.

The all-applicant means point the other way for under-12s (7,214 against 3,782 for 16 or more). 132 of the 1,878 under-12s have prior earnings above 39,900, where almost nobody enrolled, so their scores are extrapolated. I rely on the overlap-region gradient.

## 4. Roles and adjustment

| Column | Role |
|---|---|
| age, education, experience, prior_earnings, months_unemployed | confounder |
| ethnicity, region | treatment_only |
| smsa, parttime | outcome_only |
| distance_km | instrument |
| caseworker_note_len | irrelevant |
| certificate, hours_post | mediator |
| responded | post_treatment_selection |

Adjust for the nine pre-treatment columns other than distance_km. Do not adjust for certificate, hours_post or responded, which follow enrolment. Enrolment changes response (85.5% against 83.2%). Distance predicts enrolment and has no link to earnings given enrolment (p = 0.73), so it is an instrument and is left out. Experience is an exact function of age and education. The instrument and treatment_only labels are a convention: the data cannot show whether either is independent of unmeasured causes.

## 5. What the data cannot tell

- High prior earnings. Above 39,900, 32 of 3,238 applicants enrolled, and 83% of them have an enrolment probability below 0.02, so their effects are extrapolated. They are 18% of next year's applicants.
- Non-response. The estimates assume survey answers are unrelated to earnings given covariates and enrolment. The data cannot test this.
- Mediators. Hours and certificate channels cannot be separated from the earnings effect.
- Compliers. Distance could support an instrumental-variable estimate; mboostrs has no such estimator, so none is reported.

## 6. Top 20% of next year's applicants

The ranking uses the education effect model, on applicants whose enrolment probability is between 0.02 and 0.98. The 800 chosen all have 16 or more years of schooling; ties at 16 years were broken by row order. Estimated mean gain is 5,110 dollars per chosen applicant, about 4.09 million dollars in total, with no interval on the total. Per-applicant 90% intervals are in `cate_predictions.parquet` and reflect second-stage sampling only. 640 next-year applicants lie outside the overlap region and cannot be ranked.

## 7. Sensitivity to an unmeasured confounder

Omitting prior earnings moves the ATE by −2,342 (3.1 standard errors). Omitting education moves it by 7 dollars and omitting months unemployed by 140. A confounder as strong as prior earnings is a serious threat. The grid in `run_details.json` varies the outcome effect per standard deviation (1,000 to 5,000 dollars) and the imbalance (0.1 to 0.5 standard deviations). The sign never changes on it. Reversing the sign needs a product near 4,900 dollars, for example 5,000 dollars per standard deviation with a one-standard-deviation imbalance.

## How sure I am

- ATT and employment: fairly sure; stable across seeds and variants.
- ATE: fairly sure it is positive; the size is uncertain.
- Education modifier: moderately sure (p = 0.002 to 0.003 across pseudo-outcomes).
- Earnings channel: depends on the section 2 assumption.
- Top-20% gain: the gradient is clear; the total has no interval.

## What I would check next

1. Shift non-respondents' earnings by fixed amounts and track the ATE.
2. Estimate an instrumental-variable effect with distance and compare it with the ATT.
3. Report the overlap-region estimate as the main figure for the high-prior group, noting its different target.
4. Re-run the role tests and the eligibility cut on resamples to check that the labels are stable.
5. Score the top-20% policy on a held-out split of the program-year data.
