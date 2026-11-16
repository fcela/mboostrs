# Example 2: Insurance claims — the distributional branch

**Dataset:** 9,600 policy-years. Response: `n_claims` (count, 0-7).
Columns: `age_policyholder`, `vehicle_age`, `region` (8 levels),
`bonus_malus` (numeric score), `coverage` (3 levels), `policy_id`.

## Before the first fit

`policy_id` is an ID — excluded up front. `region` is a grouping.
Counts: the response is a non-negative integer, so the family decision
is live from the start. Holdout: 20% of policy-years, stratified by
`coverage`. `na_action='omit'` (1.5% missing `vehicle_age`).

## Steps 1-2 — screen and draft

`column_screen`: `bonus_malus` (0.31, "curve"), `age_policyholder`
(0.22, "curve"), `vehicle_age` (0.14, "line"), `region` (0.11,
"levels"), `coverage` (0.06, "levels"). Draft:
`n_claims ~ bbs(bonus_malus) + bbs(age_policyholder) +
bols(vehicle_age) + brandom(region) + bols(coverage)`.

## Steps 3-4 — fit and tune

Poisson family first (the default for counts), mstop tuned with
grouped folds on `region` — lesson from Example 1 applied up front.
mstop 2000 → 890.

## Step 5 — the screen that changes the plan

`term_report` looks normal. But the residual check inside step 5 shows
the Poisson deviance residuals fanning out: variance growing faster
than the mean. This is the distributional branch calling — I do not
push through steps 6-7 on the mean model first.

## Step 8 — scan_lss_families (taken early)

```
fam = mb.scan_lss_families(fit, train)
```

The scan compares Poisson against NegativeBinomial and
ZeroInflatedPoisson on held-out log-score. NegativeBinomial wins by
0.04 nats with SE 0.011 — clears two SE. The estimated dispersion is
not constant: I refit as `gamboostLSS` with a `bols`+`bbs` spec on the
dispersion parameter too, letting `bonus_malus` move the spread.

The spread ratio check on
the LSS fit: residual dispersion in the top bonus-malus quartile is
2.3× the bottom quartile. In this fit, drivers in the top bonus-malus
quartile have greater residual dispersion as well as a higher expected
claim count.

## Steps 6-7 — back to the mean spec, then contrasts

With the NegativeBinomial LSS fit in place, `suggest_specifications`
offers an interaction (`age_policyholder` × `vehicle_age`); it fails
the two-SE rule and I drop it. The final model is the additive LSS.

`distributional_contrast`: the exceedance shift — P(n_claims ≥ 2) for
a high bonus-malus driver vs a low one, at median covariates: 0.31 vs
0.09. Under the Poisson fit this contrast was 0.24 vs 0.11 — the
Poisson understated the tail for the risky group because it could not
let the variance move. The comparison shows how allowing dispersion to vary changes the tail
probabilities.

## Step 10 — explain

The reader gets: the mean spec, the dispersion spec, the 2.3× spread
ratio, and the exceedance contrast under both families — with the
explicit note that the Poisson number was wrong in the risky tail.

## Results to carry forward

1. The spine is not rigid: step 8 can preempt steps 6-7 when the
   residuals demand it. "Deciding each next step from measured
   evidence" outranks the numbering.
2. For counts, always check dispersion before trusting the mean
   model's contrasts — tail probabilities affect the expected cost.
3. The interesting finding was in the variance, not the mean.
