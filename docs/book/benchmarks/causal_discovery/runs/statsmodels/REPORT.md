# Job-training program: effects on one-year earnings

Source: `dev_out/` (`run.py --seed 1`, dev files). Intervals are 90% bootstrap percentile intervals over 200 replicates of the whole procedure. Dollars are per applicant per year.

## 1. Average effects

Enrolling raised follow-up earnings. ATE over all 12,000 applicants: $4,670 (3,692 to 5,388). ATT over the 3,690 who enrolled: $3,515 (3,181 to 4,227). I am reasonably sure of the ATT. The ATE depends on the effect assumed above $40,000 of prior earnings, where enrolment almost never occurred (section 5); without the cap it is $5,278. A statsmodels `TreatmentEffect` cross-check (IPW on responders, no response weights) gives $3,552 (SE 1,506), just below this interval; its regression-adjustment and AIPW estimates, which extrapolate without a cap, give $8,300 to $8,900.

## 2. Channels

The program worked mainly through employment: +7.4 percentage points (5.6 to 9.7). Among workers I found no detectable change in earnings: the naive log-wage difference is -0.006, and Lee-type bounds are -0.12 to +0.11. I do not claim an earnings channel; the bounds are wide, so this is not established, not absent. The employment gain falls as prior earnings rise, by 0.59 points per $1,000 (-0.75 to -0.38).

## 3. Who gained more

- Metro residence (`smsa`): larger by $2,346 (1,611 to 4,338). Apart from distance, which is the instrument, this is the only modifier that passes a Benjamini-Hochberg correction across the 11 candidates (q = 0.10) and its bootstrap interval.
- Prior earnings: the dollar effect rises $114 per $1,000 (88 to 198). Its p-value (0.04) does not pass the correction, so it is not claimed. It nets a falling employment gain against rising earnings among workers.
- Education: effects average $3,999 (1,704 to 4,268) below 12 years and $5,346 (3,942 to 7,441) at 16 or more, but education drops out of the effect model once prior earnings is included, so it is not claimed.
- Distance to the training centre: the dollar effect rises $1,764 per log-km (659 to 2,550). Distance is the instrument (section 4), so I do not claim it.
- Age below 25: $3,343 (3,184 to 4,988), not different from the average.

## 4. Roles of the columns

- Treatment: `enrolled`. Outcome: `earnings_post`.
- Confounders: `age`, `prior_earnings`, `months_unemployed`. `region` also predicts earnings weakly (p = 0.07) and is adjusted for.
- `experience` equals `age - education - 6` exactly. It is redundant and dropped.
- Treatment-only: `ethnicity`, `region`. Outcome-only: `education`, `smsa`, `parttime`. Adjusting for them is safe and adds precision.
- Instrument: `distance_km`. It predicts enrolment (first-stage F = 184) and has no average association with earnings (p = 0.83). Its exclusion is doubtful because it changes the dollar effect (section 3).
- Irrelevant: `caseworker_note_len`. No association with enrolment or earnings, including interactions (all p > 0.4).
- Mediators, not adjusted for: `certificate` and `hours_post`, measured during or after the program.
- Post-treatment selection: `responded`. Enrolment raises the response rate (logit coefficient 0.45, p < 0.001); I correct with inverse response weights, not adjustment.

## 5. Where the data cannot tell what the program did

Pooled enrolment above $40,000 of prior earnings is under 2%, against about 22% in the band just below. These applicants are 27% of the training file and 18% of the test file. The data say nothing about what enrolment would have done for them. I hold the effect at its value at the edge of the enrolled range ($39,790 of prior earnings), so effects above that point are assumptions. Two further gaps: the data cannot show whether non-response depends on unrecorded earnings, and the new applicants are poorer than the program-year applicants (mean prior earnings $25,884 against $31,673).

## 6. Ranking for next year

Test applicants are ordered by the predicted effect `cate`, and the top 800 (20%) have a mean of $6,779, against $4,271 for all test applicants. However, 57% of that group has prior earnings at or above $40,000, where the effect is the edge value. Those applicants are rarely eligible in practice. If enrolment is limited to applicants below $40,000, the ranking should be applied within that group; I did not score that restricted ranking. Among test applicants, 86% have a 90% interval entirely above zero.

## 7. Sensitivity

- Unmeasured confounding: an unmeasured confounder would need to explain about 7.7% of the residual variance of both enrolment and earnings to push the linear estimate to zero (6.1% for the interval to include zero).
- Instrument: 2SLS on distance gives $4,310 (-1,146 to 9,766). The interval is too wide to use, and the exclusion restriction is in doubt.
- Dropping prior earnings from the models turns the ATE into -$4,860. The estimate depends on it.
- Modifier claims use a multiplicity correction, but in one permuted null draw the rule still claimed a modifier; treat the list as a shortlist.

## What I would check next

1. Earnings in the year before application as a placebo outcome, to test whether the prior-earnings adjustment removes the selection.
2. The content of the caseworker note, which may carry the unmeasured factor behind steering.
3. Non-response: compare responders and non-responders on the variables that predict earnings, by enrolment group.
4. The $40,000 boundary: confirm the eligibility rule and collect enrolment decisions for applicants near it.
5. A randomised encouragement arm below $40,000, to test the employment effect and the metro modifier directly.
