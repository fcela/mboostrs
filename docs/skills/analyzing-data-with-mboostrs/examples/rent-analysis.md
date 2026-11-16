# Example 1: Apartment rents — grouped tuning and varying slopes

**Dataset:** 4,200 apartment listings. Response: monthly rent (EUR).
Columns: `area` (m²), `rooms`, `district` (12 levels), `year_built`,
`floor`, `has_balcony` (bool), `distance_center` (km), `listing_id`.

## Before the first fit

Roles first. `listing_id` is a whole-number column with 4,200 distinct
values — an ID, not a measurement. I write it down as "exclude" before
anything else, because step 2's label logic reads dtypes and a numeric ID
looks like a measurement until the response proves otherwise. `district`
is a grouping (neighborhoods), not a measurement. No leakage: all columns
are known at listing time. Missing data: 3% of `year_built` missing;
`na_action='omit'` for the whole path, decided once.

Holdout: reserve 20% of listings (stratified by district) as the
evaluation set. Everything below touches training rows only.

## Step 1 — column_screen

```
screen = mb.column_screen(train, 'rent', output='records')
```

Sorted by `curve_r2`: `area` (0.61, hint "curve"), `district` (0.44,
hint "levels"), `distance_center` (0.28, hint "curve"),
`year_built` (0.19, hint "curve"), `rooms` (0.17, hint "line"),
`floor` (0.04, hint "none"), `has_balcony` (0.03, hint "none").

`area` dominates, as expected. `floor` and `has_balcony` score near zero
marginally — I do not drop them; the screen is marginal and a masked
effect can hide here. They stay out of the first draft but on the watch
list for step 5.

## Step 2 — classify_columns / suggest_terms

```
kinds = mb.classify_columns(train, 'rent')
terms = mb.suggest_terms(kinds)
```

`area` → `bbs(area)`, `distance_center` → `bbs(distance_center)`,
`year_built` → `bbs(year_built)`, `rooms` → `bols(rooms)`,
`district` → `brandom(district)`. `listing_id` correctly excluded —
the label logic saw 4,200 levels with level means indistinguishable from
noise and refused it. Good: the step-1 discipline paid off, but the
library would have caught it anyway.

Draft: `rent ~ bbs(area) + bbs(distance_center) + bbs(year_built) +
bols(rooms) + brandom(district)`.

## Step 3 — fit

```
fit = mb.gamboost(draft, train, family=mb.Gaussian(),
                  control=mb.boost_control(mstop=2000, nu=0.1))
```

## Step 4 — tune_mstop

```
tuned = mb.tune_mstop(fit, train, folds=mb.folds(train, k=5))
```

mstop 2000 → 1132. Refit at 1132.

Listings within a district share neighborhood information. Random folds
place listings from the same district in both training and validation sets,
so they estimate risk for new listings from known districts. The deployment
task includes new districts, so I tune with grouped folds
(`folds=mb.grouped_folds(train["district"])`): mstop 1132, and the
holdout-tune gap closed. This fold design estimates performance on districts
omitted from each training set.

## Step 5 — screen_interactions + term_report

```
pairs = mb.screen_interactions(train, fit.resid(), response='rent')
report = mb.term_report(fit)
```

`term_report` shows `bbs(area)` earning the most risk reduction,
`brandom(district)` second. `screen_interactions` top hit:
(`area`, `district`) with held-out MSE removed well above the 1% floor —
a varying slope: the price per m² differs by neighborhood. Runner-up
pairs score near zero (noise). `floor` and `has_balcony` still show
nothing on the residual — they stay out.

## Step 6 — suggest_specifications

The suggestion list offers `bbs(area, by=district)`-style varying
coefficient and a `btree` alternative for `distance_center` (the screen
had a mild step hint I downweighted). I take the varying-coefficient
candidate and one `bbs → btree` swap for `distance_center` as a
sensitivity check, nothing else — two candidates, not ten.

## Step 7 — compare_specifications

```
comp = mb.compare_specifications([current, varying, tree_dist], train,
                                 folds=grouped_folds)
```

The varying-coefficient spec clears the two-SE rule (change 0.021,
SE 0.008). The `btree(distance_center)` swap does not (change 0.004,
SE 0.006) — I drop it, noting the curve was fine. Adopt the varying
coefficient, refit, re-screen: the (`area`, `district`) pair no longer
tops the interaction screen. The loop closes.

## Step 8 — heteroscedasticity

```
het = mb.heteroscedasticity(fit2, train)
```

Spread ratio (top vs bottom residual-variance quartile by fitted value):
1.6. Below the "worth modeling" threshold — the Gaussian homoscedastic
fit stands. I skip `scan_lss_families` and say why: no evidence the
variance moves enough to matter.

## Step 9 — sensitivity + contrasts

`sensitivity` over `area`: the partial effect is monotone increasing
with a visible flattening past 120 m² — no turns, no surprises worth
reporting. `prediction_contrast` for district: the exceedance shift —
P(rent > 1500) ranges from 0.08 to 0.61 across districts at median
covariates. That spread *is* the interesting finding: neighborhood
matters more than any within-unit feature.

## Step 10 — llm_explain_prompt

Render for the reader: what was fit, what each term earns, the
district-varying price per m², the 1.6 spread ratio (not modeled), and
the exceedance spread across districts. Associations, not causal
claims — stated explicitly.

## Results to carry forward

1. Grouped folds are not optional when a group column exists — random
   folds lie in the optimistic direction.
2. The interaction screen earns its keep: the varying slope was not in
   the draft and the screen found it.
3. "Not interesting" is a result: the 1.6 spread ratio and the
   no-change `btree` swap both got written down as negatives.
