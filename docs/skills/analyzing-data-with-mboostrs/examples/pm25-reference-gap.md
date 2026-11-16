# Example 4: Hourly PM2.5 — closing the gap to a tree ensemble

**Dataset:** hourly PM2.5 at one Beijing station, 2010–2014. Response:
`pm2.5` (µg/m³, positive, 99th percentile about 400 against a median near
70). Columns: `hour`, `month`, `TEMP`, `PRES`, `DEWP` (dew point), `Iws`
(cumulated wind speed), `cbwd` (wind direction, 4 levels), `Is`, `Ir`
(hours of snow and rain). Task: the 10th, 50th and 90th percentiles of
PM2.5 for every hour of 2014, trained on 2010–2013.

The numbers below come from the PM2.5 forecast benchmark, where an agent
analyst stopped at step 4 and a later investigation, with each change
chosen on 2013 only, recovered most of the gap to CatBoost.

## Before the first fit

Rows are in time order, so random folds would let the model see the hours
around each held-out hour. Validation is time-ordered: fit 2010–2012,
score 2013, then refit 2010–2013 at the chosen step count. The response
is heavy-tailed, so the model is fitted to `log1p(pm2.5)`; quantiles are
unchanged by a monotone transform, so `expm1` of a fitted quantile is the
quantile of PM2.5. On 2013 the log scale cut the mean pinball loss of the
three quantiles from 21.2 to 14.0.

## Steps 3–4: the first model

`Quantile(0.5)` with `bbs` on hour, month, day of year, TEMP, PRES, DEWP,
Iws and their 24-hour rolling means, `bols` on `cbwd`, `Is` and `Ir`, all
at df=4. The step count is chosen on the 2013 path. 2013 pinball for the
median: 21.55. The analyst stopped here: 25 configurations of this one
formula, varying df, nu and the transform. Its 2014 RMSE on log PM2.5
was 0.661 against CatBoost's 0.618.

## Step 5: the gap to a flexible reference

In the benchmark the reference was CatBoost: its 2013 median pinball was
21.27 against the additive model's 21.55, a gap of 1.3% and the signal that
structure is missing. `reference_gap(formula, train, time="No")` runs a
comparison with a booster of mboostrs `btree` learners on time-ordered
folds over 2010–2012. On these data it reported no material gap (0.1%,
within its SE): its tree sees only the formula's columns, so it cannot show
that lags are missing, and the gap is small inside the training years. Name
candidate terms with `candidates=` to have them measured anyway.

Before trusting a gap, check the reference. A tree scored on random folds
of hourly data looks better than it is. With time-ordered folds, and the
gap resampled by whole weeks, CatBoost's lead on 2014 held: 0.043 in RMSE,
95% interval 0.019 to 0.067, better in 10 of 12 months. The gap is real,
not overfitting.

## Candidate revisions

Each candidate was added, the step count re-chosen on 2013, and the 2013
median pinball recorded:

| Change | 2013 pinball | 2014 RMSE (log) |
|---|---|---|
| first model | 21.55 | 0.661 |
| + weather 1, 3 and 24 hours earlier (TEMP, DEWP, PRES, Iws) | 21.26 | 0.649 |
| + tensors DEWP×Iws, DEWP×PRES, hour×month | 21.24 | 0.643 |
| + `bbs(DEWP, by=cbwd, df=6)`, `bbs(Iws, by=cbwd, df=6)`, TEMP×hour, DEWP×DEWP 3 h earlier | 21.21 | 0.636 |
| CatBoost | 21.27 | 0.618 |

- **Lags** gave the largest single gain. A front passing through shows as a change in
  dew point and pressure over a few hours; the current value alone cannot
  show it.
- **Interactions.** The pairs tested were the ones where weather plausibly
  acts jointly: stagnant air traps pollution under some humidity and
  pressure conditions, and wind direction decides where the air comes from. A factor `by=` term needs df of at least 6.
- **A single deep tree term did not help.** `btree(DEWP, Iws, PRES, TEMP,
  cbwd, hour, month, maxdepth=4)` added to the formula raised the 2013
  pinball to 22.48, and its best step count fell to 750.

The final model beats CatBoost on the year used for choosing (21.21
against 21.27) and closes 58% of the 2014 gap (0.043 to 0.018). One
validation year chose these changes, so the remaining 2014 gap is not
resolved by this analysis.

## Held-out results

The 2014 forecast comes from an additive quantile model with lagged
weather and seven interaction terms. A flexible tree booster reaches 0.618
against the model's 0.636 in 2014 RMSE on the log scale; the model's terms
show which weather conditions drive PM2.5 and by how much, which the tree
does not. Report the coverage of the 10%–90% band on the validation year
by season: in the benchmark run it covered 74% of 2014 hours against the
nominal 80%, and 68% in winter.
