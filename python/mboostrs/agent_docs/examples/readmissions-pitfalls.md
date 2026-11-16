# Example 3: Hospital readmissions — unseen levels and the tensor floor

**Dataset:** 6,100 discharges. Response: `readmit_30d` (binary).
Columns: `age`, `comorbidity_score`, `hospital` (22 levels),
`discharge_month`, `length_of_stay`, `discharge_id`.

## Before the first fit

`discharge_id` excluded. `hospital` is the group column — and here it
matters twice: for grouped folds *and* for the unseen-level policy,
because the holdout contains two hospitals absent from training.

**The unseen-level decision, made up front.** `unseen_levels=` controls
what a categorical term does with a level the fit never saw. Options:
`"error"` (refuse), `"average"` (predict as an average group),
`"zero_effect"` (zero coefficient — what `cvrisk` uses for held-out
groups). For deployment to new hospitals, `"error"` would refuse every
such prediction; `"average"` is the honest choice for a random
intercept — a new hospital is drawn from the same population. I set
`unseen_levels="average"` on the `brandom(hospital)` term and write
down why. (gamboost.py: the `unseen_levels` parameter docstring —
verify the current line numbers; the semantics above are what matter.)

## Steps 1-4 — screen, draft, fit, tune

`column_screen`: `comorbidity_score` (0.26, "curve"), `age` (0.18,
"curve"), `length_of_stay` (0.12, "line"), `hospital` (0.09,
"levels"), `discharge_month` (0.03, "none"). Draft:
`readmit_30d ~ bbs(comorbidity_score) + bbs(age) +
bols(length_of_stay) + brandom(hospital)`, Binomial family.
Tune with `grouped_folds(train["hospital"])` — mstop 2000 → 640.

## Step 5 — the tensor trap I almost walked into

`suggest_specifications` proposes `bbs(age, comorbidity_score,
kind="tensor")` — a smooth surface for the two strongest predictors.
Tempting. But the tensor df floor (terms.py: `TENSOR_MINIMUM_DF = 4.0`):
a tensor term at or below 4 total df is at maximum shrinkage and can
only produce a *bilinear* surface — the two-margin null space is
constants and straight lines in each margin, 2×2. The suggestion came
with marginal df 2.5 per margin (the `TENSOR_MARGINAL_DF` default),
which squares to 6.25 total — clears the floor, but barely, and the
surface would be heavily shrunk toward bilinear anyway.

I check what the interaction screen says first: `screen_interactions`
on the residual shows (`age`, `comorbidity_score`) barely above the
1% floor. The data does not want this surface. I decline the tensor
and note why: weak screen signal *plus* the floor means I would be
fitting an expensive, heavily-shrunk near-bilinear term for nothing.
A `btree(age, comorbidity_score, maxdepth=2)` alternative also fails
the two-SE rule in step 7. The additive model stands.

The tensor candidate has weak interaction-screen support and a small
effective dimension, so I reject it. A depth-two tree alternative also
fails the two-SE rule in step 7; the additive model remains the selected
specification.

## Steps 8-9 — sensitivity with a turn

`sensitivity` over `comorbidity_score`: the partial effect rises
steeply to score 3, then *flattens and slightly declines* past 5.
A turn. Worth reporting: the highest-score patients are not the
highest-risk — likely a selection effect (the sickest get different
discharge planning). I do not over-interpret; I report the turn and
flag it as a question for the clinician, not a finding.

`prediction_contrast` hospital exceedance: P(readmit) across hospitals
at median covariates ranges 0.11–0.23 — a real spread, and the two
unseen hospitals predict at 0.16/0.17 under `"average"`, inside the
observed range. The unseen-level policy behaved as designed.

## Step 10 — explain

The reader gets: the additive spec, the declined tensor (with the
floor reasoning), the comorbidity turn flagged as a question, and
the hospital spread with the unseen-hospital behavior documented.

## Results to carry forward

1. `TENSOR_MINIMUM_DF = 4.0` constrains low-df tensor terms to
   bilinear surfaces. Check the interaction screen before accepting a
   tensor suggestion.
2. Unseen-level policy is a deployment decision, not a tuning
   parameter — make it before the first fit, from the deployment
   story.
3. The sensitivity curve turns above comorbidity score 5. Report the
   change and identify selection in discharge planning as one possible
   explanation for clinician review.
