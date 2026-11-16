# Reporting thresholds

## Thresholds by diagnostic

The brief asks for concrete thresholds; these are the ones this library's
own decisions are made with.

- **Turns in sensitivity.** `turns=True` is always reportable — the effect
  cannot be summarized by one number, and where it turns is the finding.
  For magnitude, compare the effect's range against the response's own
  spread: a range under ~5% of the response's standard deviation is a
  flatline to report as "no practical sensitivity"; anything you would act
  on is interesting.
- **Spread ratios in heteroscedasticity.** `spread_ratio` (or its
  reciprocal — read the larger direction) with `|t| > 3` on the fitted
  value, or `|t| > 3.5` on a driver column, and ratio > 1.3: report it and
  name the driver. A ratio of 2 means the best-fitted decile carries twice
  the residual spread of the worst-fitted — intervals there are wrong by
  that much. Between 1.1 and 1.3, whatever the t, is a small pattern: note
  it, do not build a distributional model for it.
- **Exceedance shifts in distributional contrasts.** Report when the shift
  crosses a line someone acts on: give both probabilities and the relative
  change (a 0.08 → 0.19 shift is 2.4× and decision-grade; 0.52 → 0.54 is
  not). Pair every exceedance claim with the mean contrast — a scale-only
  change that moves a tail while the mean stands still is the signature
  finding of a distributional model, and the reason it was adopted.
- **Interaction screen hits.** `removed` is an MSE per held-out row, on the
  same scale as out-of-fold risk differences; rank it against the total
  residual variance, and only pairs whose `removed` clears zero by more
  than the neighbouring scores' spread are leads. One large `removed` with
  the next scores at zero is a lead; `removed` values that all cluster
  together near some small positive number are the screen's noise floor.
- **Term earnings.** `earns` beyond two SE out of fold: the term is doing
  work. Within two SE: on these folds it is indistinguishable from nothing —
  try removing it (step 6 offers it) and let step 7 decide, rather than
  defending it with the point estimate.
- **`suggest_specifications` order.** Measured before detected, always.
  A detected suggestion with a larger expected gain than a measured one is
  the trap the sort exists to prevent: structure evidence is a hypothesis
  until a specification carrying it is scored.
