# Diagnostics that lead to a revision

A fitted mean model leaves residuals behind. Three patterns in them point to
a specific change of the model: correlation within clusters, correlation over
time, and single rows that move the fit. The three checks in
`mboostrs.residual_checks` measure each pattern, judge it against a stated
reference, and name the change to try next.

- `cluster_residual_check(model, cluster)` returns a `RevisionFinding`.
- `time_residual_check(model, time)` returns a `RevisionFinding`.
- `influence_check(model)` returns an `InfluenceFinding`.
- `revision_checks(model, cluster=..., time=...)` runs the three and returns
  a `RevisionReport` with `.findings`, `.flagged` and `.revisions`.

Each result has the same fields. `flagged` is the decision at `alpha`
(0.05 by default). `statistic`, `value` and `units` give the number the
decision rests on. `p_value` and `reference` say how that number is judged.
`false_alarm_rate` is the share of simulated datasets with no structure, with
the same design, in which the same rule fires. `assumption`, `alternatives`,
`revision` and `held_out_check` carry the decision guide below. `summary`
prints all of it as text.

## Decision guide

Read a row from left to right. The pattern is what the check measured. The
assumption is what the pattern challenges. Keep every alternative in view:
a residual check does not choose between them. The revision is the first
change to try, and the held-out check is the comparison that would show the
change helped.

| Pattern in the residuals | Assumption it challenges | Alternatives to keep open | Revision to try | Held-out check |
|---|---|---|---|---|
| Intra-cluster correlation above zero (`cluster`) | Rows are independent given the predictors | A cluster effect the model lacks; a predictor that varies by cluster and is missing; batch effects aligned with the labels | `brandom(cluster)` in the formula, then recheck at a larger `mstop` | cvrisk with `grouped_folds(cluster)`, with and without the term |
| Ljung-Box flag over time (`time`) | Residuals are uncorrelated over time | A smooth trend the model lacks; an autoregressive error; a level shift at a known date | `bbs(time)` or `bols(time)`; if the time term is already in the formula, the dependence is in the error and intervals are too narrow | cvrisk with `blocked_folds` ordered by time; `rolling_folds` for forecasting |
| Row with a large Cook's distance (`influence`) | No few rows drive the fit | A data-entry error; a true extreme value a heavier-tailed family would fit; an extrapolated predictor value | Inspect the rows; correct or drop errors and refit; for real extremes, refit without the rows and report both fits | Held-out loss of the full fit and of each deletion refit on rows neither used |
| Nothing flagged | None of the above is indicated at `alpha` | Structure below the detection level of this sample | Keep the formula; a flag-free result is not evidence of a correct specification | Not needed for this check alone |

A flag is evidence for a revision, not a selection of one. A cluster flag
does not establish that the cluster is the cause; a time flag does not
establish a trend. Each revision is a candidate to compare on held-out data.

## Cluster structure: `cluster_residual_check`

The statistic is the one-way ANOVA intra-cluster correlation of the response
residuals: the share of residual variance that lies between clusters, after
allowing for unequal cluster sizes. It is near zero when residuals of one
cluster are no more alike than residuals of different clusters. It is near
one when a shared cluster departure accounts for most of the residual.

The reference is a permutation distribution. The residuals are shuffled
across rows, which keeps the cluster sizes, and the ICC is recomputed on each
shuffle. The p-value is the share of shuffles whose ICC is at least the
observed one, counting the observed value once. The F-test p-value from the
ANOVA table is also in `evidence`. A flag needs a positive ICC and a p-value
below `alpha`.

The `cluster` argument takes a column name of the model's data or one label
per row. Labels are coded by their sorted values, so text labels work. A check needs
at least two clusters and at least one cluster with two or more rows. Under
five clusters a random intercept is weakly identified; the check still runs,
and the revision text says so.

## Time structure: `time_residual_check`

The residuals are averaged within each distinct `time` value and ordered by
time. The statistic is the Ljung-Box sum of squared sample autocorrelations
over lags `1..max_lag`, with `max_lag` defaulting to `floor(10 * log10(m))`,
where `m` is the number of distinct time values. Lag `k` is `k` steps between
distinct time values, so gaps in time are not filled in. `time=` must be
numeric: an index, a year, or a day count. A text time raises an error with
the conversion to make.

The reference is a permutation distribution of the Ljung-Box statistic over
shuffles of the series. The chi-squared p-value with `max_lag` degrees of
freedom is in `evidence`. `evidence["autocorrelation"]` holds the
autocorrelation at each lag, and `evidence["lags_outside_band"]` lists the
lags whose autocorrelation is outside the white-noise band `1.96 / sqrt(m)`.

## Influence: `influence_check`

The measures come from the hat-matrix diagonal, so the first pass needs no
refit. For each row the check computes Cook's distance, the leave-one-out
residual `r / (1 - h)` and the externally studentized deleted residual. The
flag uses the deleted residual, referred to a Student-t distribution with
`n - df - 1` degrees of freedom and Bonferroni-adjusted over the rows. That
reference is exact for a linear smoother with normal errors. A boosting fit
is not one, so the simulated false-alarm rate is the number to read.

The top `refit_top` flagged rows (five by default) are then removed one at a
time and the model is refitted with the same arguments. Each record in
`refits` gives the largest and mean change in fitted values on the rows that
were kept. That change is exact, not an approximation, and it is the number
to compare with the size of the effect being estimated.

The check refuses families other than Gaussian and GaussReg, and it refuses
case weights. Its leverage formulas assume normal errors and unit weights.

## False-alarm rates

Each check simulates datasets with no structure and applies its own rule to
each. The cluster simulation draws independent normal residuals on the same
cluster layout. The time simulation draws independent normal residuals at
the same time points. The influence simulation draws residuals with variance
`sigma^2 (1 - h_i)` and treats them as independent, because only the
hat-matrix diagonal is available; the full residual covariance is not used.
`draws` sets the number of simulated datasets, and `false_alarm_rate` is the
share of them in which the rule fires.

The cost is about `draws * permutations` evaluations of the statistic, each
linear in the number of rows. With the defaults (`draws=200`,
`permutations=999`), one check on 300 rows takes about one second. Lower
`permutations` for a quick look and raise it for a final report.

## Limits

- Residuals are the response residuals (`y - fitted`). For a binomial or a
  count family they are a rough screen, because their spread depends on the
  mean.
- Spatial residual structure and binned-residual uncertainty bands are not
  covered here.
- The time check reads the order it is given. A series with several rows per
  time value uses their average, so a panel is checked one cluster at a time,
  or after averaging.
- None of the checks estimates a mechanism. Each one names a change that a
  held-out comparison can accept or reject.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 200
site = np.repeat(np.arange(10), 20)
x = rng.uniform(-2, 2, n)
y = np.sin(x) + rng.normal(0, 1.0, 10)[site] + rng.normal(0, 0.5, n)
df = pl.DataFrame({"x": x, "y": y, "site": site, "t": np.arange(n, dtype=float)})
model = mb.gamboost("y ~ bbs(x, df=4)", df, control=mb.boost_control(mstop=200, nu=0.1))

report = mb.revision_checks(model, cluster="site", time="t",
                            permutations=99, draws=50, seed=1)
for finding in report.findings:
    print(finding.check, finding.flagged, round(finding.value, 3), finding.false_alarm_rate)
print(report.revisions)
assert report.findings[0].flagged          # the planted site effect is found
assert "brandom(site)" in report.findings[0].revision
```
