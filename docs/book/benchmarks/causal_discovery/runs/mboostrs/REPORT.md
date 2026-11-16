# Job-training program: effects on follow-up earnings

Library: mboostrs 0.1.3. Dev data, `run.py --seed 1`, outputs in `dev_out/`. Dollar figures are per applicant; intervals are 90 %.

## 1. Average effect (ATE, ATT)

| Query | Estimate | 90 % interval |
|---|---|---|
| ATE, all applicants | +5,611 | 4,333 to 6,890 |
| ATT, enrolled | +3,786 | 3,333 to 4,239 |
| EMP_ATE (employment probability) | +0.111 | 0.093 to 0.129 |

The program raised earnings. The ATT is below the ATE because enrolled applicants have low prior earnings, and the effect rises with prior earnings (section 3).

## 2. Channels (question 2)

Both channels carry effect. Enrolment raised the employment rate by about 11 percentage points. Among employed respondents, adjusted for the same covariates, enrolled applicants have log earnings about 0.15 higher (about 16 %). That comparison conditions on employment, which the program changes, so it is evidence of a difference among workers, not an identified effect.

## 3. Who gains more (question 3)

Prior earnings is the only modifier found. The effect is larger at higher prior earnings: the centred effect is -1,592 at the 10th percentile and +4,195 at the 90th. Above 40,000 the curve rests on 29 enrolled applicants.

The second stage adopted prior earnings and education. Adding age, months unemployed, smsa or part-time work did not lower out-of-fold risk by two standard errors. Education does not pass the modifier test: its refit p-value is above 0.001 at some winsorization level (the largest is 0.47). Subgroup differences by education and age reflect their correlation with prior earnings.

| Subgroup | Estimate | 90 % interval |
|---|---|---|
| Education below 12 | +6,506 | 1,521 to 11,491 |
| Education 16 or more | +6,436 | 4,002 to 8,870 |
| Prior earnings, bottom quartile | +3,581 | 3,067 to 4,096 |
| Prior earnings, top quartile | +11,709 | 7,118 to 16,299 |
| Age below 25 | +4,647 | 3,731 to 5,563 |

## 4. Roles and adjustment (question 4)

- Confounders: age, education, prior earnings, months unemployed. `experience` equals age - education - 6, so it adds nothing beyond them.
- Outcome only: part-time work.
- Treatment only: ethnicity, region (they predict enrolment, not earnings, given the confounders).
- Instrument: distance to the training centre (predicts enrolment; independent of the confounders given the other covariates).
- Irrelevant: smsa, caseworker note length.
- Mediators: certificate, hours_post. Post-treatment selection: responded. Treatment: enrolled. Outcome: earnings_post.

Adjust for the confounders and part-time work. Do not adjust for distance, ethnicity, region, certificate, hours or response.

## 5. What the data cannot tell (question 5)

- Prior earnings of 40,000 or more: 3,225 training applicants, 29 enrolled (0.9 %). Their effects are extrapolated from lower-earning enrolled applicants.
- 675 of 4,000 new applicants (17 %) have an enrolment probability below 0.02 under the training model. All have prior earnings of 40,000 or more.
- Earnings are missing for 16 %. The estimates assume response depends only on enrolment and covariates; this cannot be tested.
- Top prior-earnings quartile: the augmented score gives +11,709 (7,118 to 16,299); the library's R-learner (`effect_model` on respondents) gives +4,994. The score interval excludes the R-learner value. I did not resolve this.

## 6. Ranking for next year (question 6)

`cate_predictions.parquet` gives an effect and 90 % interval for each new applicant. Enrolling the top 800 (20 %) has a predicted total gain of 7.39 million; a random 800 has an expected 3.88 million. Of the top 800, 91 % have prior earnings of 40,000 or more (18 % of new applicants). All 800 have positive lower bounds.

The ranking depends on the effect above 40,000. If that effect equals the lower-group average (+3,820), the model's top 800 gain 3.27 million, a random 800 gain 3.06 million, and the ranking built on that assumption gains 4.35 million. I would enrol high-earning applicants only after the program has enrolled some and measured their outcomes.

## 7. Unmeasured confounding (question 7)

Dropping prior earnings from the adjustment set moves the ATE by -11,536 and reverses its sign (to -5,925). Dropping age, education, months unemployed or part-time work moves it by less than 300.

With an unmeasured variable that raises earnings by gamma dollars per unit and differs by delta units between enrolled and other applicants, the adjusted ATE is 5,611 - gamma x delta:

| gamma | delta = 0.25 | delta = 0.5 | delta = 1.0 |
|---|---|---|---|
| 2,000 | 5,111 | 4,611 | 3,611 |
| 5,000 | 4,361 | 3,111 | 611 |
| 10,000 | 3,111 | 611 | -4,389 |

A hidden factor about as strong as prior earnings would remove the estimated effect.

## 8. How sure

- Intervals are conditional on the fitted nuisance models and one cross-fitting split. Split-to-split variance is not included.
- Four simulated draws with no effect, built from the dev covariates, were run. The ATE interval covered zero in three. In the fourth (-2,683 to -391), the same draw claimed an employment effect. No draw claimed an effect modifier or an earnings channel. Of the 20 subgroup intervals in those draws, three excluded zero.
- An earlier design, with separate outcome models for each arm, gave -2,142 on the first no-effect draw. The pooled design gave -408 on that draw.
- Per-applicant intervals are conservative (per-term standard errors are added) and wide (median width 3,442).
- There are no true effects for these data, so no estimate is checked against truth.

## 9. Next checks

1. Repeat over several split seeds and add the split variance to the ATE interval.
2. Report the ATE and ATT inside the overlap region next to the full-population values.
3. Measure the effect above the eligibility limit with a small enrolment trial.
4. Bound the effect for non-respondents under plausible earnings differences.
5. Compare the augmented score and the R-learner on the high-earnings group in a direct check.
