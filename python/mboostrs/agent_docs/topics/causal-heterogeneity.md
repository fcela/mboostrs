# Heterogeneous effects with effect_model and rank_by_effect

Step 5 of the `causal` workflow: effects at each covariate pattern, with intervals, and ranking rows by effect.

## Heterogeneous effects with effect_model

`effect_model` estimates `tau(x)`, the effect at each covariate pattern, with
the R-learner. Pass `terms="smooth"` to get per-row intervals.

```
fx = mb.effect_model(data, spec, terms="smooth")
frame = fx.predict(new_rows, interval=True, level=0.95)
```

With `interval=False`, `predict` returns a NumPy array; `output="polars"` gives a frame with `estimate`. With `interval=True` it returns a frame with `estimate`, `se`, `lower`, `upper`, `outside_overlap` and `outside_range`, or a dict of arrays with `output="numpy"`. `outside_range` is true where a modifier with a spline basis
lies outside its training range. `predict(..., extrapolation=)` takes
`"warn"` (the default: predict and warn once), `"clip"` (hold each modifier at
its range edge), `"error"` or `"allow"`. The interval is the second stage's
pointwise interval given cross-fitted nuisance predictions. It covers the
sampling variation of that stage. It does not cover error in the
nuisance models, the choice of modifiers, or the bias of a smooth where the
true effect bends sharply.

By default (`interval_method="auto"`), a smooth second stage gets
`"second_stage_bootstrap"`. The pseudo-outcome and the nuisance predictions are
held fixed; the fitting rows are resampled with replacement `bootstrap_draws`
times (default 200), and each resample is refitted with the REML second stage
(its smoothing is chosen again on each draw). The interval is the 5% to 95%
quantile range of the draws, and the estimate is the full-sample fit. The
interval is wider than the pointwise covariance interval (see the table below).
The interval does not refit the nuisance models, so it accepts `nuisance=`. The
200 resamples took 7.6 s on one draw of the enrolment design; the whole
`effect_model` call took 56 s with this default and 35 s with
`interval_method="refit"`. Both times were taken under heavy machine load, so
the difference is noisy. `interval_method="refit"` keeps the covariance interval.

On 32 draws of the enrolment design (30 with a planted effect, two without),
per-row 90% coverage against the true effect was 0.846 for `"refit"` (mean width
4,718) and 0.907 for the percentile interval of `"second_stage_bootstrap"` (mean
width 7,161). The normal interval from the same draws, the mean estimate plus or
minus 1.645 times the draws' standard deviation, covered 0.937 at width 7,667.
Coverage by prior earnings, from above 60,000 down:

| Interval | up to 20k | 20k to 40k | 40k to 60k | above 60k |
|---|---|---|---|---|
| `"refit"` | 0.949 | 0.883 | 0.605 | 0.249 |
| percentile | 0.936 | 0.922 | 0.852 | 0.669 |

The estimate's mean error in the 40k to 60k band was -490 and above 60k it was
-1,496, the same for every interval method, so the misses above 40,000 are a
bias of the estimate rather than a narrow interval. The interval does not cover
it. Changing the propensity formula from `"smooth"` to `"auto"` moved the
`"refit"` per-row coverage by 0.004 overall and by at most 0.012 in a band, and
moved the mean error by at most 116 in a band.

`interval_method=` chooses a wider interval that also covers the nuisance
models' split-to-split variation. `"repeats"` refits the nuisance models and
the second stage on `interval_repeats` split seeds (default 5, the first being
the fit itself). At each row it takes the estimate and standard error from each
fit and combines them by the median rule of Chernozhukov et al. (2018): the
estimate is the median over fits, and the variance is the median of
`se^2 + (estimate_r - estimate)^2`. With `"repeats"`, `predict` returns that
median as the estimate, with or without `interval=True`. `"bootstrap"` resamples
the rows with replacement `bootstrap_draws` times (default 200), refits the
whole pipeline on each resample, and reports the full-sample estimate with the
5% and 95% quantiles of the refitted effects. Both need the smooth second stage
(`terms="smooth"`, or `"auto"` when it is chosen), and both refuse `nuisance=`,
because they refit the nuisance models. Each refit is a full `effect_model` fit,
so the call costs about `interval_repeats` (or `bootstrap_draws`) times the
default fit. The refits run with their warnings suppressed; the primary fit
reports them. `fx.diagnostics["interval"]` records the method and the number of
refits. Neither method corrects the bias of the nuisance models or of the
penalized smooth. On five simulated draws of the causal benchmark design, the
per-row 90% coverage was 0.874 for `"refit"`, 0.874 for `"repeats"` and 0.867
for `"bootstrap"` (30 resamples), which did not favour either over `"refit"`
at that time. The
split-to-split term widened the `"repeats"` intervals by under 1%, so the
per-row miscoverage is not caused by the split randomness; the root mean squared
error against the true effect was about 1.4 times the standard error.

## Ranking rows by effect with rank_by_effect

Ranking rows by their estimated effect and taking the top fifth is only sound
inside the overlap region. Outside it, the effect comes from extrapolation.
`rank_by_effect` ranks only the rows whose propensity lies inside the bounds,
and lists the other rows separately so they never fill the top group.

```
ranking = mb.rank_by_effect(fx, new_rows, share=0.2, bounds=(0.02, 0.98))
ranking.chosen          # positions in new_rows, best first
ranking.outside         # positions outside the bounds, not ranked
ranking.gain            # mean estimated effect of the chosen rows
ranking.gain_interval   # (lower, upper) at 0.95, or None
print(ranking.describe())
```

`fx` is an `EffectModel`, or a frame of estimates with an `estimate` column
and a `propensity=` argument. The top group holds `round(share * n)` rows,
where `n` counts all rows of `new_rows`. If the overlap region holds fewer
rows than that, the group is the inside rows and `ranking.shortfall` says how
many places are unfilled. `gain_interval` comes from the full coefficient
covariance, so it is available only when `fx` was fitted with
`terms="smooth"`. `ranking.table(output="polars")` gives one row per input
row with its estimate, propensity, overlap flag and rank.

Rows with exactly equal estimates are common when the modifier is discrete,
because every row with the same level gets the same estimate. When the cut
falls inside such a group, `ranking.tied_rows` gives the number of rows that
share the estimate at the cut, and `ranking.tie_break` names the rule. Tied
rows are ordered by interval width, narrower first, and then by row position;
without an interval the order is row position alone. `describe()` states both,
and `table()` marks the tied rows in a `tied_at_boundary` column.

## Conditional effects with CausalForestDML

`CausalForestDML` (EconML) returns the effect at each row of `X`, from a forest
fitted to the cross-fitted outcome and treatment residuals.

```python requires=econml
import numpy as np
from econml.dml import CausalForestDML
from mboostrs.sklearn import GAMBoostClassifier, GAMBoostRegressor

rng = np.random.default_rng(3)
n = 1500
x = rng.normal(size=(n, 3))
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x[:, 0]))).astype(int)
y = 2 * x[:, 0] + (1.5 + 0.5 * x[:, 1]) * a + np.sin(x[:, 2]) + rng.normal(size=n)

forest = CausalForestDML(model_y=GAMBoostRegressor(), model_t=GAMBoostClassifier(),
                         discrete_treatment=True, cv=4, n_estimators=200, random_state=3)
forest.fit(y, a, X=x[:, [1]], W=x[:, [0, 2]])
# Effects at x1 = -2, -1, 0, 1 and 2; the true effect there is 0.5 to 2.5.
grid = np.linspace(-2, 2, 5)[:, None]
print(np.round(forest.effect(grid), 2).tolist())
```

## Runnable example

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(1)
n = 800
x1 = rng.normal(size=n)
x2 = rng.uniform(-1, 1, n)
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1))).astype(float)
y = (1.5 + 0.8 * x1) * a + 2 * x1 + x2 + rng.normal(size=n)   # the effect rises with x1
data = {"a": a, "y": y, "x1": x1, "x2": x2}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 and x2 are measured before treatment and affect both treatment and outcome",
)
fx = mb.effect_model(data, spec, terms="smooth", modifiers=("x1",), folds=3, seed=1)
frame = fx.predict({"x1": [-1.0, 0.0, 1.0], "x2": [0.0, 0.0, 0.0]},
                   interval=True, level=0.95)
print(frame["estimate"].to_list())

# Rank new rows by estimated effect, using only rows inside the overlap bounds.
new_rows = {"x1": x1[:200], "x2": x2[:200]}
ranking = mb.rank_by_effect(fx, new_rows, share=0.2, bounds=(0.02, 0.98))
print(ranking.describe())
```
