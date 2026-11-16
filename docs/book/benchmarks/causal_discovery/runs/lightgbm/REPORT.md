# Job-training program: effect on follow-up earnings (LightGBM analysis)

Source: `python run.py --train data/dev/train.parquet --test data/dev/test.parquet --out dev_out --seed 1`. Intervals are 90% normal intervals (estimate plus or minus 1.645 bootstrap standard deviations) from 18 bootstrap replicates; the target was 30 and the time budget stopped the run early.

## 1. Average effects (dollars of follow-up earnings)

- ATE, all applicants: **3,102** (90% interval 1,358 to 4,846).
- ATT, enrolled applicants: **3,589** (3,238 to 3,941).
- Employment (ATE in probability): **+0.089** (0.073 to 0.106).

The ATE interval is wide because enrolment is almost absent above about 36,000 of prior earnings (see 5). Trimming the propensity at 0.05 instead of 0.02 gives 2,799; at 0.01 it gives 3,617.

## 2. Channels

Employment: yes. The employment channel of the decomposition has an interval that excludes zero, and employment rises by 0.089 probability.

Earnings of those in work: no detectable effect. The interval for the wage channel includes zero. The wage part conditions on employment, which the program changes, so this is a model-based comparison, not a randomized one.

## 3. Effect modifiers

No modifier is claimed. Education below 12 shows a larger point estimate (4,243; 1,607 to 6,880) than education 16 or more (-666; -3,660 to 2,329). The difference is suggestive and its interval is too wide to claim. Median-split contrasts for age, prior earnings and months unemployed do not separate from zero at 95%. A claim at the 90% level needs more replicates.

## 4. Roles and adjustment

- Confounders: prior_earnings, months_unemployed.
- Outcome-only (predict untreated earnings, weak enrolment link): age, education, experience. experience is an exact function of age and education.
- Instrument: distance_km (strong enrolment link; no link to untreated earnings or employment).
- Treatment: enrolled. Outcome: earnings_post.
- Mediators (do not adjust): certificate, hours_post.
- Post-treatment selection (do not adjust): responded.
- Irrelevant: caseworker_note_len, parttime, smsa. ethnicity and region predict enrolment but not earnings; they sit near the screen threshold and changed label between runs. They are excluded from the final models.

Adjustment set: age, education, experience, prior_earnings, months_unemployed.

## 5. Where the data cannot tell

- Prior earnings of about 36,000 and above: enrolled share 0.009 among 3,204 training applicants. Effects there rest on about 24 enrolled responders. The 723 test applicants (18%) in this region get extrapolated effects.
- Nonresponders (16%): earnings are observed only for responders. The analysis assumes response is unrelated to earnings once covariates and enrolment are known. This cannot be checked here.

## 6. Ranking for next year

Ranking by the `cate` column gives a sum of 7.7 million dollars over the top 800 test applicants (20%). The reliability of this is low: 41% of the top 800 are in the non-overlap region (prior earnings above about 36,000), where effects are extrapolated. Ranking should be restricted to applicants inside the overlap region, or the high-prior predictions should be checked against other data, before budget is allocated.

## 7. Sensitivity to an unmeasured confounder

- A bias of 1,358 dollars would move the ATE's 90% lower bound to zero; a bias of 3,102 would move the point estimate to zero. An unmeasured factor that raised earnings by 5,000 dollars and raised enrolment probability by 0.10 would lower the ATE to about 2,600.
- An instrumental-variable estimate using distance gives 10,006 (7,389 to 12,623), with a first-stage partial correlation of -0.157. It is about three times the DR ATE, and the intervals do not overlap. Possible causes: effects for applicants whose enrolment responds to distance are larger; the exclusion restriction fails; or effects vary strongly. This is unresolved and is the main open question.

Conclusion for the agency: enrolment raised employment and earnings on average, by about 3,000 dollars, but the size is uncertain (roughly 1,400 to 4,800), and the program's effect for high prior earners is not identified.

## Next checks

1. Run more bootstrap replicates (30 or more) and compute the effect-modifier contrasts again.
2. Compare the distance-based instrument estimate with the DR estimate within prior-earnings strata.
3. Restrict the ranking to the overlap region and compare the two rankings on held-out data.
4. Test whether response depends on earnings, using any auxiliary survey data.
