# Checks on effect_model candidates

Part of step 5 of the `causal` workflow: which candidate `effect_model` chose and whether it beats a constant effect.

## Other candidates and the constant-effect comparison

The other candidates, `terms="stumps"` and
`terms="joint"`, are boosted fits with no coefficient covariance, and
`predict(interval=True)` refuses them. `fx.ate_plugin` is the unweighted mean of `tau`
over the fitting rows (each row counts once, with no overlap weighting) and has
no interval; use `aipw` for an average effect with an interval.

`effect_model` also fits a constant effect, a single number for `tau`, on the
same folds and scores it with the same out-of-fold R-loss. That number is the
R-loss constant, an average of the pseudo-outcome weighted by `e(x)(1 - e(x))`.
It is not the ATE, and it differs from `fx.ate_plugin` when overlap is uneven. `fx.constant_rloss`
holds that loss, and `fx.describe()` prints it with the difference between the
chosen candidate and the constant, and its standard error. The call warns when
the chosen candidate does not beat the constant by more than its standard
error, which means the modifiers add no reliable heterogeneity at this sample
size; report the constant effect from `aipw` in that case. The call also warns
when the chosen boosted candidate was tuned to fewer than 5 boosting steps,
which means `tau(x)` is close to a constant.

The second stage fits the R-learner pseudo-outcome `(y - m) / (t - e)`, which
grows as the overlap `|t - e|` shrinks. By default, `effect_model` clips that
pseudo-outcome at its 1st and 99th percentiles over the rows with positive
weight (`learner="r_winsorized"`); the bounds are in
`fx.diagnostics["pseudo_bounds"]`. `learner="r"` keeps the unbounded
pseudo-outcome. Clipping shrinks the effects of the tail rows when the tails
are light, and the per-row 90% coverage then falls by one to two points.

The smooth candidate (`terms="smooth"`) fits each modifier's spline over the
1st to 99th percentile range of the fitting rows. Rows outside that range are
left out of the fit, and `predict` holds a modifier at the edge of its range
(`extrapolation="clip"`, the default for smooth fits), so the effect is flat
beyond the range. A positive modifier whose 99th percentile is at least 3
times its median is entered as its logarithm. The chosen scales are in
`fx.diagnostics["smooth_scale"]`, and `fx.diagnostics["smooth_rows_outside_range"]`
counts the fitting rows left out. The boosted candidates use the raw columns.

The call warns when the largest absolute fitted effect exceeds 10 times the
interquartile range of the observed outcome; when it does, inspect the rows
with the largest `|tau(x)|` and their propensity, and report the constant
effect from `aipw`. The automatic candidate choice (`terms="auto"`, the
default) returns a candidate only when it beats a constant effect by more than
its standard error and its tuned step count is at least 5. Among the candidates
that pass, it keeps the lowest cross-validated R-loss, and it returns
`"smooth"` when that candidate beats `"smooth"` by no more than one standard
error. `fx.diagnostics["terms_selection"]` gives the gains, the flags and the
rule that decided the choice. The gains and standard errors are reported for
every candidate, including a named one (`terms="smooth"`), with the rule
`"requested"`, so a named choice can be checked against the constant effect
the same way.

The cross-fitted nuisance models (the propensity, and the outcome model that
the R-learner's `m(x)` is built from) take the same `extrapolation=` and
`flat_tolerance=` arguments as `aipw`, and their warnings are issued once per
call under the same rules. `effect_model` reads `propensity_clip`,
`outcome_missing` and `overlap_bounds` from the spec when the call does not pass
them. It has no `propensity_terms` or `outcome_terms`; use `aipw` for a custom
nuisance formula.

Categorical covariates and modifiers are accepted. They are read as factors,
as the boosting formulas read them, and new rows must carry the same labels.

`fx.propensity` holds the cross-fitted propensities of the fitting rows as an
array. Calling it with new rows, `fx.propensity(new_rows)`, returns `e(x)` for
each row from the propensity model fitted on all rows. `rank_by_effect` uses
the same call when `propensity=` is not given.

## Runnable example

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(1)
n = 800
x1 = rng.normal(size=n)
x2 = rng.uniform(-1, 1, n)
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1))).astype(float)
y = (1.5 + 0.8 * x1) * a + 2 * x1 + x2 + rng.normal(size=n)
data = {"a": a, "y": y, "x1": x1, "x2": x2}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 and x2 are measured before treatment and affect both treatment and outcome",
)
# terms="auto" compares the candidates on the same folds and keeps the best.
fx = mb.effect_model(data, spec, terms="auto", folds=3, seed=1)
print(fx.constant_rloss)
print(fx.describe())
```
