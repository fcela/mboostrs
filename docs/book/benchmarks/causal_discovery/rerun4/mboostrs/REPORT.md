# Job-training program: effect on follow-up earnings

Source: mboostrs 0.1.3, run.py --seed 1 on the dev files. Intervals are 90%.

## Summary

Enrolling raised follow-up earnings. Averaged over all 12,000 program-year applicants the effect is +\$5,841 (\$4,664 to \$7,017); among enrollees it is +\$3,714 (\$3,247 to \$4,182). Employment rose by 10.5 percentage points (8.9 to 12.1), and earnings of those in work rose by about 16%. The effect is larger for applicants with more schooling. The data cannot say what the program does for applicants with prior earnings above about \$40,000, because almost none of them were enrolled.

## 1. Average effects

Estimator: cross-fitted augmented inverse-probability weighting (aipw) with boosted models. Non-respondents (16%) enter through a response model.

- ATE: +\$5,841 [4,664; 7,017].
- ATT: +\$3,714 [3,247; 4,182]. Smaller than the ATE because enrollees had lower prior earnings, where the effect is smaller (section 3).
- Employment (EMP_ATE): +0.105 [0.089; 0.121], a difference in probability.

Subgroup effects in dollars: education under 12 years, 6,893 [1,190; 12,597]; 16 years or more, 6,233 [4,297; 8,169]; prior earnings at or below \$15,910, 3,636 [3,133; 4,140]; at or above \$41,330, 12,101 [7,839; 16,363], an extrapolated value (section 5); age under 25, 4,551 [3,641; 5,460].

## 2. Channels

Both. Enrolment raises employment (section 1). Among employed respondents, log earnings rise by 0.145 [0.124; 0.166], about 16%. That comparison conditions on employment, which enrolment changes, so it compares partly different people. The split of the total effect between the two channels is not estimated.

## 3. Who gains more

- Schooling: larger at higher education. The BLP slope is +\$1,111 per standard deviation of education (2.9 years), Holm-adjusted p = 0.03, on overlap applicants. The full-sample subgroup means above do not show it, because they include the extrapolated high-earnings applicants. Confidence: moderate.
- Prior earnings: BLP slope +\$5,241 per standard deviation, adjusted p = 0.24. Not claimed.
- Age and months unemployed: not claimed (unadjusted p = 0.07 and 0.09).

## 4. Roles and adjustment

- Confounders, adjust: age, education, prior_earnings, months_unemployed. experience is an exact function of age and education, so it adds nothing.
- Treatment-only, adjust (not needed for bias): ethnicity, region.
- Outcome-only, adjust (precision): smsa, parttime.
- Instrument, do not adjust: distance_km. It predicts enrolment (p < 1e-50) and not earnings given enrolment (p = 0.50).
- Irrelevant: caseworker_note_len (p = 0.25 for enrolment, 0.93 for earnings).
- Mediators, do not adjust: certificate, hours_post (both measured during or after the program).
- Post-treatment selection, do not condition on: responded. It is used only to weight non-respondents.
- Treatment: enrolled. Outcome: earnings_post.

Roles come from conditional-independence tests at p < 0.05. On 1,500 rows they are less reliable: months_unemployed was labelled treatment_only.

## 5. Where the data cannot tell

- Prior earnings above about \$40,000. Enrolment in this group was 0.9%. About 19% of training applicants have an enrolment probability outside 0.02 to 0.98, mostly from this group. Their effects are model extrapolations, and the CATE_prior_q4 interval does not include that uncertainty.
- Non-respondents. Their earnings are unobserved. The estimates assume that response is unrelated to earnings once covariates and enrolment are known. If an unmeasured factor drives both, the bias cannot be signed from these data.
- New applicants have lower prior earnings: 18% are above \$40,000, against 27% in the program year.

## 6. Ranking for next year

Individual effects come from a smooth model with education and prior earnings as modifiers, fitted on the training file. Ranking uses only applicants inside the overlap region. The top 20% is 800 of 4,000 applicants, with expected gain \$6,076 per enrollee [\$4,724; \$7,428], about \$4.9 million in total (\$3.8 to \$5.9 million). Inside the overlap region the average model effect is \$3,974, so the ranking adds about \$2,100 per enrollee. The R-learner comparison does not show that the modifiers beat a constant effect by more than their standard error, so confidence in this addition is low. The 535 applicants outside the overlap region (13%) are ranked last.

## 7. Sensitivity

Let an unmeasured factor shift earnings by D dollars per standard deviation, with an adjusted treated-minus-control difference of S standard deviations. The bias is D x S. It would take \$5,841 to remove the average effect, for example \$8,000 per SD with S = 0.73. At \$8,000 per SD and S = 0.5, the adjusted ATE is \$1,841 and its interval excludes zero. The 90% lower bound reaches zero at a bias of about \$4,664.

## How sure I am

- ATE positive: fairly high, given the adjustment set and the response assumption. With enrolment reassigned at random the ATE is \$201 [-429; 830]. On a 1,500-row subsample it is \$3,956 [2,201; 5,711].
- Employment effect: fairly high. Earnings among the employed: moderate.
- Education modifier: moderate to low. Ranking gain beyond the average effect: low.

## Next checks

1. Instrumental-variable estimate using distance_km (iv_effect).
2. Bounds on non-respondents' earnings, or a response model with more predictors.
3. A randomized pilot for applicants with prior earnings above \$40,000.
4. A pre-specified test of the education modifier on new data.
