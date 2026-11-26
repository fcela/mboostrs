# Categorical predictors

Pass string columns as they are; the fit encodes them and records the
levels. Two learners take a categorical column:

- `bols(g)` — one unpenalized coefficient per level after the first
  (treatment coding). Its degrees of freedom are the number of levels minus
  one, so in a selection race against `bbs(x, df=4)` a many-levelled `bols`
  is favoured for its flexibility; give the race equal footing with
  `brandom(g, df=4)` or compare the two out of fold.
- `brandom(g, df=...)` — a ridge-penalized effect per level, shrunk toward
  the average group. Use it for many levels, for groups (subjects, sites),
  and whenever the selection must be fair against smooth terms.

A bare categorical name in a formula (`y ~ g`, with no learner) is coded as
`brandom(g, lambda=0)`: one unpenalized effect per level, the same fit as
`bols(g)`. `gamboost` warns once, naming the column. Write `bols(g)` to state
that coding in the formula, or `brandom(g, df=...)` for a ridge-penalized
effect.

Prediction, refits and intervals take the labels as written: pass a frame
with the same string labels to `predict`, `gam_refit(...).effects`,
`gamlss_refit(...).draws`, `confint(method="refit")` and
`conformal_interval`. A label the fit never saw raises unless you choose a
policy with `unseen_levels=`: `"average"` predicts the row as an average
group, `"zero_effect"` gives a `brandom` term zero effect. A model returned
by `mb.fit` predicts such a label as the average group without being told,
and its decision trail records `unseen_levels: "average"` in `final`; pass
`unseen_levels="error"` to its `predict` to refuse instead. A column whose
values match the fitted labels only after a type cast (fitted as strings
`"1"`, `"2"`, arriving as integers 1, 2) is refused under every policy:
cast it back. An error that lists numbers where you passed labels (or the
reverse) means one frame was encoded and the other was not; pass the raw
frame, not an encoded copy.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 400
df = pl.DataFrame({"x": rng.uniform(0, 1, n), "roof": rng.choice(list("ABCD"), n)})
df = df.with_columns(y=df["x"] + 0.4 * (df["roof"] == "D").to_numpy()
                       + rng.normal(0, 0.3, n))
fit = mb.gamboost("y ~ bbs(x, df=4) + brandom(roof, df=4)", df,
                  control=mb.boost_control(mstop=300))
grid = pl.DataFrame({"x": [0.5] * 4, "roof": list("ABCD")})
print(fit.predict(newdata=grid))
print(mb.gam_refit(fit).predict(grid))
print(fit.predict(newdata=pl.DataFrame({"x": [0.5], "roof": ["E"]}),
                  unseen_levels="average"))
```

To report a categorical effect, order the levels by their effect and say
which levels cannot be told apart (overlapping refit intervals), instead
of reporting a slope.
