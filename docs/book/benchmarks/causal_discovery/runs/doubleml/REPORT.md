# Job-training program: effect on follow-up earnings (dev data, DoubleML 0.11.4)

## Summary

Enrolling raised follow-up earnings by 2,896 dollars on average over all program-year applicants (90% interval 1,263 to 4,529). Over the applicants who enrolled, the average effect is 3,783 dollars (3,321 to 4,245). The main channel is employment: the share of applicants with positive earnings rose by 9.3 percentage points (7.6 to 11.1). Earnings among those who worked did not change by a detectable amount. Applicants with prior earnings above 40,000 dollars were almost never enrolled, so the data do not describe their effect.

## Method

DoubleMLSSM with random forests (200 trees) for the outcome, for enrolment and for response, with the learners chosen on out-of-fold scores. Non-response is treated as random given enrolment and the adjustment set. Five cross-fitting folds; headline values are medians over two sample splits. The ATT uses a doubly robust score built from the same nuisance predictions. Averaging the ATE score over enrolled applicants gives about twice that value and was not used.

## Answers

### 1. Average and attendee effects

ATE 2,896 (1,263 to 4,529). ATT 3,783 (3,321 to 4,245). The raw gap between enrolled and non-enrolled responders, about -13,300 dollars, reflects the low prior earnings of enrolees.

### 2. Work or earnings

Employment rose by 9.3 points (7.6 to 11.1). Among employed applicants, hours rose by 121 (103 to 138), but their earnings rose by 225 dollars (-1,393 to 1,844), which is not distinguishable from zero. The effect works mainly by getting people into work. The earnings-if-employed channel is not established. Comparisons among the employed condition on a post-treatment variable and describe that group only.

### 3. Who gains more

Within prior earnings up to 40,000 dollars, the effect rises with education: 2,231 dollars below 12 years, 4,193 for 12 to 15 years, and 5,922 for 16 years or more. The raw p-value is 0.008. Across eight candidate characteristics the Benjamini-Hochberg adjusted value is 0.065, so education is not claimed as a modifier at the 5% level. Over the full file the pattern reverses (6,219; 4,491; -2,914), because applicants above 40,000 dollars carry very large weights and noisy pseudo-outcomes. Age and months unemployed show no modifier. Prior earnings cannot be tested within the overlap region, because the top band is empty there.

### 4. Roles and adjustment

| Column | Role |
|---|---|
| age, prior_earnings, months_unemployed | confounder |
| education, smsa, parttime | outcome_only |
| ethnicity, region | treatment_only |
| experience | confounder (age - education - 6; not entered separately) |
| distance_km | instrument |
| caseworker_note_len | irrelevant |
| enrolled | treatment |
| certificate, hours_post | mediator |
| responded | post_treatment_selection |
| earnings_post | outcome |

Adjust for age, education, ethnicity, smsa, region, parttime, prior_earnings and months_unemployed. Do not adjust for distance_km (instrument), caseworker_note_len (irrelevant), certificate and hours_post (mediators), or responded (selection). Pre-treatment roles come from a statistical screen at p < 0.01; an instrument must also have a first-stage chi-square of at least 64 and not be explained by the other columns.

### 5. What the data cannot tell

1. Applicants with prior earnings above 40,000 dollars are 27% of the program year and 18% of new applicants, and 0.9% of them enrolled. Their effect is not estimated.
2. Non-response was 16%. The estimates assume response is random given enrolment and the adjustment set. Whether response depends on earnings itself cannot be tested.
3. The certificate's share of the effect sits inside the total and is not separated.
4. Individual effects for applicants with prior earnings of 30,000 to 40,000 have wide intervals.

### 6. Ranking for next year

Of 4,000 new applicants, the top 800 by predicted effect have a mean predicted effect of 6,214 dollars, about 5.0 million dollars in total. A random 800 at the average predicted effect gives about 2.5 million. All 800 have 90% lower bounds above zero, and none has prior earnings above 40,000. The predictions come from a best linear predictor in education, months unemployed, age and prior earnings, fitted to applicants below 40,000 of prior earnings. Earlier basis choices changed which applicants fell in the top group, so membership near the cut-off is uncertain.

### 7. Sensitivity to an unmeasured factor

DoubleML's sensitivity analysis is not available for the selection model, so I refitted without each adjustment column. Dropping prior earnings moves the ATE from 2,896 to -5,431 dollars. Dropping months unemployed moves it by -587, age by -526, and education by +59. A factor that affects enrolment and earnings as strongly as prior earnings would reverse the sign of the ATE. The conclusion depends on prior earnings being fully measured.

## How sure

Nuisance learner choice moved the ATE between 2,152 and 3,704 on single splits. The headline value moves about 300 dollars across split seeds. On a 1,500-applicant subsample the ATE is -916 dollars (-4,029 to 2,198). The earnings-if-employed channel crosses zero on that subsample, so that channel call is fragile.

## What I would check next

1. Formal sensitivity bounds for the selection model.
2. Bounds on the ATE under non-random response.
3. The ATE and modifiers for prior earnings up to 40,000, reported separately, with more data above that value if available.
4. An IV check using distance, conditional on metropolitan residence. An exploratory 2SLS on responders gave about 4,800 dollars.
5. An employment and earnings decomposition that does not condition on employment.
