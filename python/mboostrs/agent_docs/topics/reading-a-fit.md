# Reading a fit, its limits, and what a first reading misses

## Effect interpretation

The fit is additive: the prediction is an offset plus one effect per
selected base learner, summed on the link scale (one scale per distribution
parameter, where the fit has several). Component-wise boosting built it by
repeating one step: fit every candidate learner to the current negative
gradient, keep the one that reduces the loss most, add `nu` times it. A
learner picked often accumulated a large effect; a learner never picked
has coefficient zero and is absent from the model.

Effects are **not** all centered on zero; where one sits vertically depends
on its learner (`bbs(..., center=True)` sits near zero; `bols` carries its
own intercept; a plain `bbs` keeps the level it accumulated). Read the
*difference between two points within one effect* — the change in the
linear predictor between those covariate values — and compare within an
effect, never across effects and never against zero.

## Limits of the fitted model

- The boosted fit itself carries no standard errors: each learner was
  selected and shrunk on the same data, so the usual interval formulas do
  not apply, and a bootstrap of the boosted fit under-covers. Intervals and
  tests come from refitting the selected terms by REML (`gam_refit`,
  `gamlss_refit`, `confint(method="refit")`), and prediction intervals from
  `conformal_interval` or an LSS fit's `prediction_interval`; all of them
  are conditional on the terms boosting selected, which the report must
  say. `agent_guide("uncertainty")` shows each call. The out-of-fold
  comparisons in steps 4-7 remain the evidence for which terms belong.
- Importance is a share of in-bag risk reduction. Two correlated covariates
  split the credit, so a small share can mean redundancy, and shares depend
  on which other learners were offered.
- Effect shapes hold over the covariate range they were printed with.
  Outside it the learner extrapolates by construction, not from data.
- Everything is conditional on the stopping iteration: the same data stopped
  elsewhere gives different effects, and a longer path gives larger ones.
- These are associations, not causal effects. A covariate can carry a large
  effect because it causes the response, shares a cause with it, stands in
  for something absent, or because of how the data were collected — the fit
  does not distinguish, and neither does any contrast you derived from it.
  Say "moves with", not "drives", unless the analyst supplied a design that
  identifies an effect.

## Patterns to investigate

Before you write the report, examine the patterns you are about to report
and the fit behind them, and surface what a first reading misses: regions
surprising or counterintuitive for the subject matter; effects that look
like data artefacts rather than mechanisms; importance shares or update
counts that could mislead — credit split between correlated covariates, many
updates buying little risk reduction, a learner absorbing the signal of a
variable absent from the model. For each: which numbers make it notable,
which explanations could produce it (a real mechanism; a data artefact such
as a sparse covariate range; a covariate standing in for another;
confounding by something not in the model; the path having stopped where it
did), which you consider most likely on subject-matter grounds, and what
observation, analysis or experiment would settle it. The fit alone cannot
settle it — commit to a judgment instead of listing possibilities. This is
the middle turn of `llm_explain_prompt(output="turns")`; run it as your own
check even when no language model is in the loop.
