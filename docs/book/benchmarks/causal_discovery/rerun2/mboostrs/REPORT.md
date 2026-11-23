# Job-training program: effect on follow-up earnings

Dev data, mboostrs 0.1.3, `run.py --seed 1`. Intervals are 90%. Dollar amounts are differences in follow-up earnings.

## 1. Average effects
- ATE +\$6,477 [5,675; 7,278] over all 12,000 applicants.
- ATT +\$3,714 [3,254; 4,173] over the 3,690 applicants who enrolled.

Earnings are missing for 1,928 applicants who did not answer the survey. Both estimates assume the outcome is missing at random given the covariates and enrolment. The data cannot test this. Confidence: moderate. The propensity model reached its tuning cap (800 steps) with risk still falling. A tuned run gave +\$6,015 [5,176; 6,855].

## 2. Work or earnings
Both. Enrolment raised the probability of employment in the follow-up year by 9.8 percentage points [8.3; 11.4]. Among applicants employed in the data, follow-up earnings were \$4,807 higher [3,669; 5,945]. This contrast conditions on employment, which enrolment can change, so it is a within-employed comparison, not a causal share of the effect.

## 3. Who gains more
Claimed, at a 5% family-wise threshold over 11 covariates:
- Prior earnings, larger when higher: \$9,419 above \$27,230 against \$3,535 below (z 6.1).
- Months unemployed, larger when lower: \$8,642 at two months or fewer against \$3,924 above (z -4.9).

Not claimed: age (under 25: \$4,710 [3,647; 5,772]; halves z 2.5); education (under 12: \$5,734 [2,651; 8,817]; 16 or more: \$8,025 [6,329; 9,722], intervals overlap); ethnicity (halves z -2.7).

## 4. Roles
- Confounders: age, prior_earnings, months_unemployed. experience is an exact function of age and education (age - education - 6), so it is not in the design and takes the role of its components.
- Outcome only (predict earnings, not enrolment): education, smsa, parttime.
- Instrument (predict enrolment, not earnings given the others): ethnicity, region, distance_km.
- Irrelevant: caseworker_note_len.
- Treatment: enrolled. Outcome: earnings_post.
- Mediators: certificate, hours_post (measured during or after the program).
- Post-treatment selection: responded.

Adjustment set: age, education, smsa, parttime, prior_earnings, months_unemployed, caseworker_note_len. The instrument-role columns are left out. Adjusting for them would not bias the estimate but would reduce overlap. Post-treatment columns are never adjusted.

## 5. Where the data cannot tell what the program did
- Applicants with high prior earnings. Above about \$70,000, 83% of applicants have an enrolment probability outside [0.02, 0.98]. Across all training rows the share is 4.6%. The estimate for applicants at or above the 75th percentile of prior earnings is \$14,243, with no interval, because one arm has fewer than 30 applicants.
- Non-respondents. Response depends on age, education and prior earnings. If non-response also depends on unmeasured earnings-related factors, the estimates are biased. No bound was computed.
- The certificate pathway. The total effect is reported. The share that runs through certificate completion is not identified.

## 6. Ranking next year's applicants
Of the 4,000 new applicants, 3,904 have an enrolment probability inside [0.02, 0.98]. The 800 with the largest predicted effect (the top 20%) are the first to enrol. Their mean predicted effect is \$8,266, against \$4,506 for all 3,904 inside applicants. The 96 applicants outside the overlap region have no cate and are not ranked. The per-applicant model passes an out-of-fold check (gain \$5.9 million, standard error \$1.8 million). Cross-validation chose only 4 boosting steps, so predicted effects vary little (interquartile range \$3,120 to \$4,820). Treat the ranking as weakly supported.

## 7. Sensitivity to an unmeasured confounder
Omitting one adjusted covariate moves the ATE by at most \$203, except prior earnings. Omitting prior earnings moves it by -\$12,645 and reverses the sign. The conclusion rests on adjusting for prior earnings. A confounder as strong as prior earnings would overturn the sign. The adjusted ATE reaches zero when the outcome effect per unit of the omitted confounder times its treated-minus-control difference equals \$6,477. For example, an outcome effect of \$2,000 per unit needs an imbalance of 3.2 units.

## Checks and limits
- No-effect data (earnings generated from control outcomes plus noise): ATE -\$637 [-1,494; 221]; the ATT and employment effects cover zero; no modifiers or channels were claimed. One of seven subgroup intervals excluded zero (education under 12: -\$2,806 [-5,230; -381]), about the rate expected from 90% intervals.
- The R-learner (effect_model, smooth terms) was unusable here: per-row effects on the training rows ranged from -\$11.5 million to +\$19.2 million, and its loss was worse than a constant effect. It was not used.

## Next checks
- Simulated draws, with and without effect, to measure interval coverage.
- A propensity search above 800 steps, and the ATE under that setting.
- A bound for non-response.
- A per-applicant model whose intervals include nuisance uncertainty.
