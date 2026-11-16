# `discover`: constraints, contrasts, budgets and limits

## Constraints

Four optional arguments change which terms the stages may use. Each applies
to the mean model; the spread stage has only one term, a log-scale term for a
single column.

- `heredity="none"` (the default), `"weak"` or `"strong"` sets when an
  interaction may be nominated. `"strong"` needs a main-effect term for both
  of the pair's columns in the mean model at the time of the test; `"weak"`
  needs one. Backward elimination runs first, so the main effects a pair
  needs are the ones that survived it. With `"strong"`, a pure interaction
  such as `sin(3 * x1 * x2)` is not found, because x1 and x2 have no main
  effect to keep. With `"none"` it is found. The log records each pair the
  rule bars.
- `forbidden_pairs=[("x1", "x2")]` lists column pairs that are never
  nominated or adopted.
- `mandatory_terms=["x4"]` lists columns whose mean term backward
  elimination never drops, whatever its out-of-fold change.
- `candidates={"mu": [...]}` limits the predictors each parameter's search
  may use. Mandatory columns are always included.

A bare value applies to the mean model. A mapping keyed by `"mu"` or
`"sigma"` gives each parameter its own value; `mandatory_terms={"sigma": ["z"]}`
puts a log-scale term on `z` without the spread test. Any other key raises
`ValueError`. The rule in force is the `"rule"` entry of `result.log` and the
`constraints` field of the result.

## Contrasts

`result.contrast(variable, reference, alternative)` is the mean over the
training rows of the change in the prediction when `variable` moves from
`reference` to `alternative`. A variable whose term backward elimination
dropped returns a contrast whose note says "not selected". Its estimate comes
from a REML refit of the model before elimination (the draft terms with the
adopted interactions), and its interval is that refit's interval widened to
include 0, because elimination kept the term out. The number shows what the
term would contribute; it does not show that the term is in the model. With a
distributional refit, or for a variable the draft never had, the contrast is
exactly 0 with a note that it was not selected.

With a GAM refit the interval is conservative: it is the sum of the
pointwise half-widths of the affected effects, because the refit keeps no
posterior draws. With a distributional refit the interval is taken from
posterior draws of the requested parameter. A varying-coefficient term is
centred over the training rows, so a contrast that depends only on such a
term averages to exactly zero, and its interval has zero width.

## Null data, small data and budgets

On a response with no structure the mean model is usually `y ~ 1` and the
result has no interactions or spread drivers. `result.warnings` then says the
mean model is intercept-only and names the family; every prediction is the
response mean. Each run keeps the candidates it rejected in `log`, so the
rejections can be inspected.

Data with fewer than 30 complete rows raise `ValueError`. When the data hold
fewer than 50 rows per fold, the fold count is lowered to one fold per 50 rows,
with a minimum of three, and the log records the change.

A numeric predictor that is a copy or a monotone transform of the response
(absolute rank correlation at least 0.999) is dropped before the draft and
named in `result.warnings`. Pass it as the response, or leave it out. A
whole-number predictor with one distinct value per row is named in
`result.warnings` as a possible row identifier but kept; a `mandatory_terms`
entry that names a dropped response copy makes the call raise `ValueError`.

Every stage that refits models has its expected time from `estimate_cost`
written to `log` before it runs. The entries have `stage: "estimate"` and
`candidate` set to the stage they estimate, with the seconds in
`estimated_seconds`.

`mb.estimate_cost("discover", data, response, ...)` gives the fit count of each
stage on its first pass before the run, in `stages`, and the time of one fit.
After the run, `result.stage_seconds` holds the measured seconds of each stage
(`"mean effects"`, `"interactions"`, `"spread"`, `"tune"`, `"refit"`, and
`"predict"` when `newdata` was given); `repr(result)` shows their sum.

`budget_seconds` sets a limit on the stages. A stage that would take the run
past the limit is skipped, and the log says which stage and by how much.
A skipped mean stage keeps the full draft, so the result is then not reduced.
The final model is fitted whatever the budget; when that fit goes past the
limit, the log says so.

`extrapolation` (default `"warn"`) sets how the fold fits of the screens and
the tuning treat held-out rows outside a term's training range. The warning is
issued once per call and names `discover(extrapolation=...)`. The final
`newdata` predictions always use `"clip"`.

## The candidates a result touched

`mboostrs.specification_curve(result, data, response)` fits every candidate
the run compared, on the fold partition the run used, and reports each one's
out-of-fold score next to the adopted model's. Each row carries `gap vs
selected` (its score minus the adopted model's) and `gap se`, the paired fold
standard error of that gap. Each row also says whether its choices were written
down before any result or raised by one. The draft and its first
leave-one-out set are written down before any result. The later elimination
rounds, the interaction candidates and the log-scale candidates are raised by
the results, and a curve with any raised candidate is labelled post hoc.

`joint_test` gives a one-sided sign-flip p-value for whether the adopted model
beats the median candidate on the shared folds. The folds are the ones the run
used, so the p-value describes this run. It does not confirm a finding on new
data. The same call accepts an `autospec` recommendation or a `SearchSession`.

## Limits

- Intervals are conditional on the adopted terms. The selection that chose
  them is not in the interval, so a term adopted on the same data has an
  interval that is too narrow.
- The spread stage tests a Gaussian log scale only. Other families report
  `spread` as not run in the log.
- Screens and comparisons use the rows that have every predictor and the
  response. `na_action` applies to the fits. A NaN counts as missing, the same
  as a null. Under `"impute"`, a numeric interaction partner is filled with its
  training median in the tensor term, without an indicator term.
- With `na_action="omit"`, a `newdata` row with a missing predictor cannot be
  predicted, and the call raises. The default `"impute"` fills such values with
  the training median or the missing level, as `missing-data` describes.
