# Job-training program: effect of enrolment on one-year earnings

Data: program-year training file (12,000 applicants) and new applicants file (4,000). Dev run: `python run.py --seed 1`. Intervals are 90% unless stated.

## Summary

Enrolling raised follow-up earnings by about 5,700 dollars on average over all applicants [4,400; 6,900], and by about 3,800 [3,400; 4,300] among the enrolled. Enrolment raised the chance of work by about 10 points. The effect is larger for applicants with more schooling. The data say almost nothing about applicants above the 40,000 prior-earnings limit, and the average depends on that part.

## Method

Doubly robust estimation (`aipw`), five-fold cross-fitting, boosted spline nuisance models. The propensity model includes a tree term for the eligibility step. The outcome model is the two-part model, chosen from three by out-of-fold error. Non-responses (16%) are inverse-weighted by a response model that includes enrolment.

## 1. Average and subgroup effects

| Query | Estimate | 90% interval |
|---|---|---|
| ATE, all applicants | 5,658 | [4,433; 6,882] |
| ATT, enrolled | 3,836 | [3,379; 4,293] |
| Education under 12 | 6,661 | [662; 12,659] |
| Education 16 or more | 6,310 | [4,399; 8,221] |
| Prior earnings at or below 15,910 | 3,620 | [3,121; 4,119] |
| Prior earnings at or above 41,330 | 11,455 | [6,992; 15,918] |
| Age under 25 | 4,548 | [3,640; 5,455] |
| Employment probability, ATE | +0.103 | [0.087; 0.120] |

Confidence: the employment effect and the ATT are clear. The ATE depends on the outcome model (5,658 to 6,084 across the three candidates) and on the top quarter of prior earnings (section 5).

## 2. Work, earnings, or both

Both. Employment rose by 10.3 points. Among responding applicants in work, enrolment raised log earnings by 0.147 [0.126; 0.168], about 16% higher. That contrast conditions on employment, which enrolment changes. Splitting the ATE into the two parts is not identified; a rough split puts 2,000 to 4,100 dollars through jobs.

## 3. Who gains more

Education: the effect is larger at higher schooling, by about 1,160 dollars per standard deviation (p = 0.002; Holm-adjusted p = 0.02, inside the overlap region). Nothing else passed the adjusted test.

Prior earnings: the average effect is about 3,600 at low prior earnings and about 11,500 at the top quartile. The top-quartile figure rests on extrapolation. Inside the overlap region the relation is weak (joint p about 0.08), so I do not claim it.

Age: the average effect under 25 is 4,548 against 5,658 overall. The age test does not survive multiple-comparison adjustment, so I do not claim it.

## 4. Roles

| Column | Role | Adjusted for |
|---|---|---|
| age, education, experience | confounder | yes (experience is fixed by age and education) |
| prior_earnings, months_unemployed | confounder | yes |
| smsa, parttime | confounder (outcome link detected) | yes |
| ethnicity, region | treatment_only | yes |
| caseworker_note_len | irrelevant (no link detected) | yes (leaving it out changes the ATE by about 20 dollars) |
| distance_km | instrument | no |
| certificate, hours_post | mediator | no |
| responded | post-treatment selection | no; enrolment changes response (p < 0.001) |
| enrolled, earnings_post | treatment, outcome | |

Only distance_km is left out. Columns are not dropped for lack of a detected link: at 1,500 rows the tests missed real confounders.

## 5. Where the data cannot say

- Applicants with prior earnings above about 42,000: 2,472 training rows (21%), 17 enrolled. Their enrolment probability is below 0.02 or above 0.98, so their effect comes from the outcome model. They contribute 2,302 of the 5,658 ATE; the overlap rows alone give 4,227.
- Applicants who did not answer the survey (16%). The analysis assumes their earnings are missing at random given covariates and enrolment. That cannot be checked with these data.

## 6. Ranking for next year

The top 20% (800 applicants) by estimated effect, chosen from the 3,460 test applicants inside the overlap region. The 540 applicants outside that region are not ranked. The chosen group has an average estimated effect of 6,031 dollars per enrolled applicant. Test applicants above 40,000 in prior earnings are 18% of the file, against 27% in training.

Per-applicant effects are weakly supported: the gain from letting the effect vary with prior earnings is 3.0 thousand dollars, standard error 7.2 thousand. The median 90% interval is 3,700 dollars wide and covers sampling error only.

## 7. Sensitivity to an unmeasured factor

Dropping prior earnings from the adjustment set moves the ATE by -10,715, more than the estimate itself. Dropping any other measured covariate moves it by 200 dollars or less. A hidden factor with outcome effect d dollars per standard deviation and treated-minus-control imbalance b standard deviations moves the estimate by d times b. It removes the ATE at d times b = 5,658: for example d = 13,143 (half a standard deviation of earnings) with b = 0.43. A hidden factor as strong as prior earnings would overturn the conclusion.

## Checks

- Instrumental-variable estimate (distance_km): 5,104 [-1,679; 11,888], first-stage F = 105.
- Placebo draw (enrolment simulated from the propensity model, no effect): ATE -1,101 [-2,470; 267], covering zero; no channel or modifier claimed.
- Refits dropping one or two covariates: ATE 5,440 to 5,904.

## What I would check next

1. A randomized or lottery-based comparison above the 40,000 limit.
2. Administrative earnings records for non-respondents, to test the missing-at-random assumption.
3. The rule that caseworkers used for steering, and whether distance_km is unrelated to unobserved motivation.
