# Effect modifiers

Part of step 5 of the `causal` workflow: tests of which covariates change the effect.

## Effect modifier tests with `effect_modifiers`

`effect_modifiers` tests whether the treatment effect changes along each
covariate. It takes an `AIPWResult` from `aipw`, because its input is the
per-row augmented scores of that result: each score has the effect at that row's
covariates as its conditional mean, so a regression of the scores on a covariate
measures how the effect changes along it.

```
tests = mb.effect_modifiers(result, data, method="blp", level=0.9)
tests.table(output="polars")      # one row per covariate
tests.selected()                  # covariates whose decision is True
tests.diagnostics["gate"]         # the R-learner gate: gain, gain_se, passed
tests.message                     # why no modifier is claimed, when the gate failed
proj = tests.projection("passing")
ranking = mb.rank_by_effect(proj, new_rows, share=0.2, propensity=e_new)
```

For each covariate, the function fits two tests on the rows whose propensity
lies inside `overlap_bounds` (default `(0.02, 0.98)`):

- The BLP slope: the regression of the score on the covariate. A numeric
  covariate is standardized over those rows, so the slope is the change in the
  effect, in outcome units, per standard deviation of the covariate. A
  categorical covariate gets one coefficient per level, relative to the first
  level in sorted order, and a joint test over the levels.
- GATES: the mean score in `groups` quantile groups (default 4) of a numeric
  covariate, or in each level of a categorical one. The contrast is the top group
  minus the bottom group, and its standard error comes from the group variances.

Both tests use heteroskedasticity-robust (HC1) standard errors and are reported
in the same table. `method` chooses the test that sets `decision` and `direction`:
`"blp"` (the default) or `"gates"`. A decision is True when the p-value, adjusted
across the covariates by Holm's step-down method (`adjust="holm"`), is below
`1 - level`. The table also gives the intervals at `level`.

## The R-learner gate

The score regressions treat the AIPW scores as fixed numbers, so their standard
errors omit the error of the nuisance fits, and on data with no effect
modification they can report a modifier. Before it claims any modifier,
`effect_modifiers` therefore runs a gate (`gate=True`, the default). The gate
fits the smooth second stage of `effect_model` on the tested covariates, using the
cross-fitted nuisances of `result`, and compares its out-of-fold R-loss with the
R-loss of a constant effect on the same folds. It passes when the gain over the
constant exceeds its standard error.

When the gate fails, every `decision` is False, `tests.selected()` is empty and
`message` (also the first sentence of `describe()`) gives the gain and standard
error. The score tests stay in the table, and `test_decision` keeps what they
alone would have claimed. `diagnostics["gate"]` holds `passed`, `gain`,
`gain_se`, the modifiers used, the seed (`gate_seed`) and the seconds taken.
`gate=False` decides on the score tests alone. The gate needs the treatment and
outcome columns in `data`, and covariates that are in `result.spec.covariates`.
Cost on a 2,000-row simulation with five covariates, after `aipw` with its defaults: the gate took 0.28 s of CPU time (median of five draws; about 1 s of wall time on a loaded machine), against 0.34 s for the `aipw` call and 0.001 s for the score tests. The gate reuses the nuisances of `result` and fits one penalized second stage on five folds.
The gate shares the low power of the R-loss comparison, so a real but small
modifier can fail it. In 40 simulated draws with no effect (2,000 rows, five covariates) the score tests alone claimed a modifier in 4 and the gate in none; in 20 draws with an effect that rises with one covariate, the gate passed in all 20 and the covariate was still claimed.

## Covariates written as transforms

`covariates` accepts expressions with the transforms of the model formulas:
`log`, `log1p`, `sqrt`, `exp`, `scale`, `center` and `ecdf` of one column,
`pmin` and `pmax` of a column and a bound, and `I` for arithmetic on columns,
written as in a formula (`log(x)`). A name that is a column of
`data` is read as that column. The table, `describe()` and the
projection name the expression as written.

```
tests = mb.effect_modifiers(result, data, covariates=["log(prior_earnings)", "education"])
```

The slope of a numeric covariate is per standard deviation of the transformed
values, so a covariate whose effect is linear on the log scale gives a stronger
test as `log(x)` than as `x`. A fitted transform (`scale`, `center`, `ecdf`) keeps
the parameters of the fitting rows when `projection(...).predict` scores new
rows. An expression with another function is an error that lists the supported
transforms. With the gate on, an expression is gated through the data columns it
reads, which must be in `result.spec.covariates`.

`projection("all")` fits the regression of the scores on all covariates jointly,
and `projection("passing")` refits it on the covariates with a True decision.
`rank_by_effect` accepts either projection and ranks the rows inside the overlap
region by the projected effect, as it does for an `EffectModel`. A projection has
no propensity model of its own, so `propensity=` is required. `effect_model` is
unchanged and remains the default route for predicting an effect at new rows.

Measured limits, from a benchmark with 12,000 training rows, 4,000 test rows, ten
draws with an effect that depends on the covariates, and two draws with no effect:

- Power is low. At level 0.9 the BLP test gave the four material modifiers (prior
  earnings, education, months unemployed, part-time status) the correct direction
  in 10 of 40 covariate-draw pairs, a true positive rate of 0.25. GATES gave 9 of
  40.
- Before the gate, with `method="blp"` at level 0.9, the two null draws produced one claim (a
  region covariate on one draw). The
  standard errors treat the scores as fixed and omit the error of the nuisance
  fits. Where that error correlates with a covariate, the test can claim an
  effect that is absent.
- Regret against the best 20 percent of all rows cannot rank methods on this
  design. The best 20 percent lie mostly outside the overlap region (about 92
  percent of them in the benchmark), so every ranking that stays inside the
  overlap region scores near 1 on that measure. Compare rankings with the best
  rows inside the overlap region instead, with the same group size as the
  ranking chooses. On that measure the BLP projection gave a regret of 0.648
  against 0.897 for `effect_model` on the same draws (0 is the best group,
  1 a random group inside the overlap region).
- Against `effect_model` on the same draws, the BLP claims had no wrong-direction
  material claims (0), where `effect_model` selected covariates with the wrong
  sign seven times.

The R-loss comparison that `effect_model` reports has low power for detecting effect modifiers. For who benefits more, use `effect_modifiers`; its tests concern the direction of the average effect along each covariate.

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
res = mb.aipw(data, spec, folds=3, mstop=60, tune=False, seed=1)
tests = mb.effect_modifiers(res, data, method="blp", level=0.9)
print(tests.table(output="polars"))
print(tests.selected())
```
