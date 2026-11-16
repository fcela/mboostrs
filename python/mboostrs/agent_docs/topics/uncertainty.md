# Uncertainty: effect intervals, term tests, contrasts and prediction intervals

A boosted fit has no standard errors: its terms were selected and shrunk on
the same data, so the usual formulas do not apply, and a bootstrap of the
boosted fit reproduces the shrinkage (its 90% bands cover about 62–74% on
the package's benchmarks). The library gets intervals by refitting the
terms boosting selected as one penalized model with REML smoothing
parameters, the construction of mgcv's `gam(method="REML")`. Every interval
below is conditional on the selected terms; say so in the report.

| Question | Call | Returns |
|---|---|---|
| Interval for each term's effect | `mb.gam_refit(model).effects(rows, level=0.9)` or `model.confint(method="refit", level=0.9)` | `{term: (estimate, lower, upper)}` per row |
| Is a term's effect zero? | `mb.gam_refit(model).tests` / `.summary()` | edf, statistic, p-value per term (mgcv's `summary.gam` test) |
| Same, distributional model | `mb.gamlss_refit(lss).effects(rows, parameter="mu")` | as above, per parameter |
| Interval for an average contrast | `mb.gamlss_refit(lss).draws(rows, n_draws=...)` on two copies of the rows | posterior draws of every parameter at every row |
| Interval for a new observation | `mb.conformal_interval(model, rows, level=0.9)`; for an LSS fit also `lss.prediction_interval(level, newdata)` | lower, upper per row |
| Several `Quantile` fits as one distribution | `mb.rearrange_quantiles([f10, f50, f90], rows)` | rows sorted across levels, crossings |
| Interval for a standardized contrast when rows share clusters | `mb.prediction_contrast(model, rows, var, ref, alt, cluster=labels)`; the same `cluster=` works on `average_marginal_effect` and `distributional_contrast` | `.interval` (cluster-bootstrap percentile), `.dependence` (cluster-robust SE, cluster count); the fitted model is held fixed |
| Interval for a distributional contrast from a refit | `mb.distributional_contrast(lss, rows, "x", 0.2, 0.8, quantity="quantile", value=0.9, refit=mb.gamlss_refit(lss))` | `.interval` from the refit's coefficient draws; `.dependence["center"]` is the refit's own centre, which can differ from the point estimate |
| Interval when the terms were chosen on the same rows | `mb.split_refit(formula, data, contrasts=[(variable, reference, alternative)], repeats=...)` | per-split and aggregated contrasts, selection frequencies, p-values |
| Which terms are real? | `mb.stabsel(model, q=..., PFER=...)` (recommended); `mb.compare_specifications` with the two-SE rule | stable set with an expected-false-selections budget (PFER); out-of-fold risk changes |
| Does a declared sign or shape agree with the fit? | `mb.check_declarations(rec_or_model, data, folds=5)`, or `rec.with_declaration_checks(data)` then `rec.report()` | one record per declaration: `verdict` (agrees, contradicts, inconclusive), the measured shares or contrasts, held-out risk change from removing the constraint, and the rule applied |

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 600
df = pl.DataFrame({"x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n),
                   "g": rng.choice(["a", "b", "c"], n)})
df = df.with_columns(
    y=np.sin(3 * df["x"].to_numpy()) + 0.4 * (df["g"] == "c").to_numpy()
      + rng.normal(0, 0.2 + 0.5 * df["z"].to_numpy(), n))

# Term intervals and tests on a mean model.
model = mb.gamboost("y ~ bbs(x) + bbs(z) + brandom(g)", df,
                    control=mb.boost_control(mstop=300))
refit = mb.gam_refit(model)
print(refit.tests["bbs(z)"]["p_value"])          # z has no mean effect here
est, lo, hi = refit.effects(df.head(5), level=0.9)["bbs(x)"]

# Average contrast with an interval, from a distributional model.
lss = mb.gamboostLSS({"mu": "y ~ bbs(x) + brandom(g)", "sigma": "y ~ bbs(z)"},
                     data=df, families=mb.GaussianLSS(),
                     control=mb.boost_control(mstop=300))
lrefit = mb.gamlss_refit(lss)
at_lo = lrefit.draws(df.with_columns(pl.lit(0.2).alias("x")), n_draws=400, seed=1)["mu"]
at_hi = lrefit.draws(df.with_columns(pl.lit(0.8).alias("x")), n_draws=400, seed=1)["mu"]
change = (at_hi - at_lo).mean(axis=1)            # one average contrast per draw
print(change.mean(), np.quantile(change, [0.05, 0.95]))

# Prediction intervals for new rows.
print(lss.prediction_interval(level=0.9, newdata=df.head(3)))
print(mb.conformal_interval(lss, df.head(3), level=0.9, seed=0).to_frame())
```

Reading the results:

- A cluster interval treats the clusters as the sampling units, so it needs at
  least two clusters, and fewer than ten makes its percentile interval
  unreliable (the call warns). It does not include the fit's own variability:
  a contrast that is the same on every row, such as a linear effect, has a
  zero-width cluster interval.
- A contrast with `joint_support="report"` (the default) records the share of
  rows whose counterfactual combination has no nearby training row in the same
  categorical cell. Read a low share as the contrast resting on extrapolation.
- Use the same seed for both `draws` calls, so the two sets of draws pair
  up and the contrast's interval reflects the contrast, not the noise of
  two independent samples.
- A variable boosting never selected has no term in the refit, so its
  contrast is exactly zero with no interval. Report it as "not selected",
  and give the evidence for leaving it out (a `compare_specifications` row,
  a `stabsel` probability), instead of a zero-width interval.
- `prediction_interval` trusts the fitted family. `conformal_interval` holds
  its coverage whatever the family (`method="distributional"` for an LSS fit,
  `"cqr"` for heteroscedastic noise with a mean model, `"cv+"` for small
  samples), on average over rows like the training rows. Neither corrects
  for a test set whose covariates differ from training; check coverage
  within subgroups (by the spread driver, by region) on held-out folds.

## Stability selection after fitting

`mb.stabsel(model, q=..., PFER=...)` is the recommended post-fit selector and
the only one with an error-rate guarantee (Meinshausen and Bühlmann, 2010):
`PFER` bounds the expected number of falsely selected learners. `mb.deselect`
and `mb.probe` are cheaper heuristics with no such guarantee. `stabsel` picks
*which* learners; `cvrisk` picks *how long* to fit them, so refit the stable
set's formula and tune `mstop` there.

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(0)
n = 300
cols = [f"x{i}" for i in range(10)]
X = {c: rng.normal(size=n) for c in cols}
y = 2 * X["x0"] - X["x1"] + 0.5 * X["x2"]**2 + rng.normal(size=n)
data = dict(X); data["y"] = y

model = mb.gamboost("y ~ " + " + ".join(f"bbs({c})" for c in cols), data,
                    control=mb.boost_control(mstop=300))

# B=20 here for speed (40 fits); use B=50+ in practice.
stab = mb.stabsel(model, q=6, PFER=1, B=20, verbose=False)
print(stab)                   # stable set, cutoff, q, PFER, fit count
print(stab.selected_names)    # ['x0', 'x1', 'x2'] — exactly the signals
print(stab.to_frame())        # per-learner max frequency and selected flag
```

## Term effect intervals: gam_refit(...).effects

`mb.gam_refit(model)` returns a `GamRefit` object, which refits the terms the
`gamboost` fit selected. Its `effects` is a method of that object, called with
the rows to evaluate:

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 600
df = pl.DataFrame({"x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n)})
df = df.with_columns(y=np.sin(3 * df["x"].to_numpy()) + rng.normal(0, 0.2, n))
model = mb.gamboost("y ~ bbs(x) + bbs(z)", df, control=mb.boost_control(mstop=200))

refit = mb.gam_refit(model)                    # GamRefit; the model is its only required argument
effects = refit.effects(df.head(5), level=0.9) # method call; a dict keyed by term
est, lo, hi = effects["bbs(x)"]                # three arrays, one value per row
```

Each term maps to `(estimate, lower, upper)`, where the arrays have one value
per row of `newdata`. Each effect is centred over the fitting rows, so the
estimate is the term's contribution to the linear predictor on the link scale,
not the response. The interval is pointwise at `level`.

- `centre=True` centres each effect over the rows of `newdata` instead. Use it
  to compare an effect with a curve that is centred over the same rows.
- `unseen_levels="zero_effect"` or `"average"` lets a `brandom` term read a group
  the refit did not see. The default `"error"` refuses such a row.
- `corrected=True` widens each interval for the uncertainty in the estimated
  smoothing parameters.
- A distributional model has the same call with a parameter name:
  `mb.gamlss_refit(lss).effects(rows, parameter="mu")` returns the
  `{term: (estimate, lower, upper)}` dict of that parameter only; without
  `parameter` it returns one such dict per parameter.
