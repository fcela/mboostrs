# Modeling tactics

Each entry links a modeling situation to a diagnostic and a revision, with
evidence from the corresponding benchmark. Run the checks before interpreting
a model that may leave structure unexplained.

## The gap to a flexible reference

After tuning, run `reference_gap(formula, data)`. It fits a `btree` booster
on the same held-out rows and loss as your formula and reports the loss
difference with its standard error (SE).

The reference sees every predictor column of `data` and, with `time=`, lagged
and rolling-mean copies of the numeric columns; `reference_extra_columns`
lists those the formula does not use. `reference_columns="formula"` limits it
to the formula's columns; `exclude_columns=[...]` removes weight, offset or
identifier columns.

- **Gap within two SE.** The formula captures what a flexible model finds.
  Report that the check was run.
- **Gap above two SE.** Structure is missing. Add the best candidate, refit,
  and run it again until the gap is within two SE or the candidates stop
  paying.

The top `top` candidates (default 5) are refitted and measured whatever the
gap; a `measured_gain` above two `gain_se` is evidence for the term. They are
omitted columns, lags and interactions, ranked on the family loss: the
negative loss gradient at out-of-fold predictions, scaled by the root of the
curvature (the ordinary residual for Gaussian, Pearson for Poisson).

In the PM2.5 benchmark, lags, tensor interactions and `by=` terms chosen on
2013 cut the 2014 RMSE on log PM2.5 from 0.661 to 0.636 (CatBoost 0.618).

**Check the reference before trusting its gap.** Random folds of
time-ordered or grouped rows favour the reference, because a tree memorizes
neighbours of each held-out row. Use `time=` or grouped folds, and resample
whole blocks when you put an interval on the gap.

## Sources of a gap

- **Jumps.** A gap none of the candidates below closes may be a regime jump:
  `screen_discontinuities(data, residual, response=...)` on the tuned fit's
  response-scale residual scans axis-aligned and diagonal boundaries and
  reports each with a selection-honest `z` (the candidate was chosen on one
  half of the rows and scored on the other, so `|z| >= 2` with a positive
  `gain` is an honest signal, not the winner's curse) and a `type_z` that
  separates a jump from a steep smooth crossing. A `typed_jump` row
  proposes a `btree(pair)` term or the boundary's step indicator, which the
  comparison step scores like any other candidate; a screen that does not
  flag means no screened evidence for a jump in these columns, and the gap belongs to a smooth
  or interaction term.
- **Interactions.** `screen_interactions` on the out-of-fold residuals
  ranks pairs. A numeric pair becomes `bbs(a) %X% bbs(b)` at df at least
  6.25 (the tensor df floor is 4, see `pitfalls`). A numeric covariate whose
  effect changes with a factor becomes `bbs(x, by=f, df=6)`; a factor `by=`
  needs df at least 6.
- **Lags.** On time-ordered rows, the current value of a covariate misses
  its recent history: a front passing through, a storm building. Add the
  covariate at 1, 3 and 24 steps back and a 24-step rolling mean. Never add
  a lag of the response unless it is known at prediction time.
- **Transforms.** A positive covariate with a long right tail (largest
  value many times the median) belongs on the log scale. On a 1,500-row
  claims replicate, `bbs(ave_occup)` on the raw scale met a test row at
  1,243 against a training maximum of 6.57, and the spline's straight-line
  extension predicted 338 for a truth near 0. A heavy-tailed positive
  response belongs on the log scale too: on PM2.5, `log1p` cut the pinball
  loss from 21.2 to 14.0.
- **Unfinished tuning.** A stopping iteration at the end of the search
  grid is not a choice. Raise `max_mstop` or `nu` and tune again.

## New rows outside the training range

`predict` warns when a row is outside the range a term was fitted on, and
the warning gives how many training ranges out it is. A spline continues by
its edge slope, which is an assumption, not an estimate. Look at the worst
row before reporting: a log-scale term, or `extrapolation="clip"` to hold the
effect flat, bounds the damage.

## Missing covariates

When the missing column can be predicted from the others, the median fill
lets the other columns absorb its effect. In the claims benchmark, roof
class was assigned partly from housing age, which was missing on 4–16% of
rows. Roof-class interval coverage was 0.30 under the median fill and 0.80
under `na_action="impute_model"`, which fills each gap from the other
columns and keeps the missingness indicator. See `missing-data`.

## Categorical columns

Pass a categorical column as strings or a categorical dtype, never as
integer codes: `bols(code)` fits one straight line through the codes, which
ties every level's effect to its position. Check that a refit reports one
coefficient per level.

## Narrow features and smoothing bias

A penalized smooth flattens a narrow peak, and its interval is centred on
the flattened value. In the claims benchmark every library missed the
location effect: their 90% intervals covered the truth on at most 1 of 8
replicates, and a 12×12 spatial basis cannot resolve the planted bump (a
standard deviation of about 0.4 degrees). When an
effect lives on a small region, raise the basis size (`knots=`) and compare
held-out loss; `mb.gam` refits fast enough to try several.

## Spread

Test every covariate as a driver of the residual spread, not only the ones
in the mean model. In one claims run the spread search saw only the mean
model's columns and missed households, the largest driver (out-of-fold
gain 0.065, SE 0.003), until a manual check of residual SD by quartile found
it.

## Causal analyses

- **A second estimator is the reference.** Estimate the average effect
  with `aipw` and with one other route on the same rows, for example
  EconML's `LinearDML` with mboostrs nuisance models through the
  scikit-learn adapters (topic `causal-outcome-models`). A difference larger than
  both intervals points at a nuisance model or an overlap problem, not at
  the effect.
- **Role tests on deterministic columns.** `ci_test` returns p = 1 when a
  column is an exact linear combination of the conditioning set (age equals
  education plus experience plus 6 in the job-training data). That is not
  evidence of independence. Read `test.notes`, and drop one column of an
  exact identity before testing roles.
- **Outcome models on wide-range covariates.** A model fitted on the log
  of an outcome and mapped back with `exp` amplifies any extrapolation. In
  the gap study, a log-earnings model linear in raw prior earnings predicted
  means near 5×10⁸ dollars for the highest earners. Put skewed positive
  covariates on the log scale and check the range of the predicted means.

## Checks before the report

- **No signal.** Permute the response and run the pipeline again. It must
  claim nothing.
- **Small data.** Rerun on a subsample of a few hundred rows. A claim that
  disappears is not established at the full size either; say so.
- **Speed for the loop.** A follow-up refit should take seconds. `mb.gam`
  refits a chosen formula by REML in under a second on 15,000 rows; use it
  for drop-one checks and effect queries instead of a full boosting run.

## Estimator choice

A capacity comparison can also compare fitting methods. For Gaussian responses,
pass `estimators=("boosting", "reml")` to `select_capacity` to compare the
boosted base and tensor forms with direct joint REML fits on the same folds.
The tree form uses boosting. The menu also includes an intercept-only baseline.

```python
import mboostrs as mb

data = mb.datasets.bodyfat()
pick = mb.select_capacity(
    "DEXfat ~ .", data, folds=5, mstop=500,
    estimators=("boosting", "reml"),
)
pick.selected, pick.selected_estimator
pick.rungs
```

Each candidate record names its form and estimator, and the step count it was
scored at. The smallest form within `tolerance` standard errors of the best
score is kept, read off the paired fold difference against the best form,
which cancels what the folds share; within the same form,
the estimator order supplied by the caller breaks ties. This is a predictive
selection heuristic, not a significance test. The selected model is refitted
with the selected estimator on all training rows. A REML fit has no boosting
stopping iteration; its exported `mstop` is `None`.

This opt-in comparison supports Gaussian family loss and MSE. REML fits learn
smoothing parameters and preprocessing on each fold's training rows. Each
boosted candidate is searched for the step count its held-out score turns at,
starting at the requested `mstop` and doubling while the score still falls,
up to `max_mstop` (four times `mstop` by default); `budget="shared"` compares
every form at the same step count instead.
