# Average effects

Step 4 of the `causal` workflow: the average treatment effect and how its nuisance models are set.

## Average effects with aipw

`aipw` estimates the ATE or ATT with cross-fitted nuisance models for the
propensity and the outcome. Its interval is an influence-function interval at
`level`, which defaults to 0.95 (a 95% interval); pass `level=0.9` for a 90%
interval.
`result.describe()` gives the estimate, the interval, the propensity range and
the identification assumptions.

`repeats=R` draws R partitions of the rows and cross-fits on each. The default,
`repeat_aggregation="average"`, averages the out-of-fold nuisance predictions
over the partitions and computes one score; its interval does not include the
variation between partitions. `repeat_aggregation="median"` computes an
estimate and a standard error on each partition, reports the median estimate,
and sets the variance to the median over partitions of
`se_r^2 + (estimate_r - estimate)^2` (the median rule of Chernozhukov et al.,
2018). The spread between partitions then enters the interval. The per-partition
estimates and standard errors are in `diagnostics["repeat_estimates"]` and
`diagnostics["repeat_standard_errors"]`.

When some outcomes are missing, pass `outcome_missing="ipw"`. The function
then fits a response model for the probability that the outcome is observed,
given the covariates and the treatment, and weights the observed rows by its
inverse. This assumes the outcome is missing at random given those covariates
and the treatment. A missing outcome that depends on the outcome itself is not
corrected by the model, and no option in the function can detect that case.
Without `outcome_missing="ipw"`, any missing outcome value is refused. With
`outcome_missing="ipw"` the ATT is also available. Its target is all treated
rows, observed or not. Dropping the treated rows with a missing outcome gives a
different estimate, because it changes the target to the treated rows that
have an outcome.

Observation weights are passed as `weights=`. They are read as frequencies.

An outcome column may hold `None` among its numbers, as a column from
`to_dict()` does. Each `None` is read as a missing outcome, and the usual rule
for missing outcomes applies. An outcome whose present values are text is
refused, and the message says that missing values must be null or NaN in a
numeric column.

Baselines may be numeric, boolean, string or categorical. The default `smooth=False`
fits linear terms only. With `smooth=True`, a numeric baseline with at least ten
distinct values gets a penalized spline (`bbs`). A baseline with fewer distinct values, and every factor, gets a linear
`bols` term, with treatment contrasts for a factor. The right-hand side of each
fitted nuisance formula is in `result.nuisance["terms"]`.

`smooth` can be set per nuisance model: `smooth={"propensity": False,
"outcome": True}`. The keys are `"propensity"`, `"outcome"` and `"response"`.
A model the dict does not name takes `False`. `result.nuisance["smooth_by_nuisance"]`
holds the flag used for each model.

The defaults are `smooth=False` and `outcome_model="cv"`. Over 10 simulated
draws at n=12,000 with 90% intervals, the median and largest absolute ATE error,
coverage and median seconds were: `smooth=False` with `"cv"` 434 and 2,322, 0.9,
2.7 s; `smooth=True` with `"cv"` 550 and 1,706, 0.9, 8.0 s; `smooth=False` with
`"auto"` 790 and 2,322, 0.6, 1.5 s; `smooth=True` with `"auto"` 1,137 and 8,205,
0.6, 5.9 s. `iv_effect` has the same `smooth=False` default.

`propensity_terms=` and `outcome_terms=` replace the formula that `smooth`
builds for the propensity model and the outcome model. Each takes a
right-hand side as a string, with no response and no `~`:

```
aipw(data, spec, propensity_terms="bbs(age) + btree(prior_earnings, maxdepth=1)")
```

A tree term fits a step, such as a propensity that jumps at an eligibility
limit, which a spline does not fit well. A custom right-hand side is fitted by
`gamboost`. The response model keeps the formula that `smooth` builds.
`result.nuisance["custom_terms"]` says which models got a custom string, and
`result.nuisance["terms"]` gives the right-hand side each model used. With
`outcome_model="pooled"`, aipw adds the treatment as a linear term to
`outcome_terms`. With `outcome_terms`, `outcome_interactions=True` is refused:
write the interaction inside the string with `by=` set to the treatment column,
for example `"bbs(age, df=4, by=treated)"`. A custom term that names the
treatment as a bare term is refused. A custom `outcome_terms` with
`outcome_model="auto"` and a flagged bin uses the pooled model without
interactions, and the flag is recorded without a warning.

`propensity_learner=` chooses the propensity formula when `propensity_terms` is
not given. The default, `"smooth"`, is the formula that `smooth` builds: linear
terms when `smooth` is off, `bbs` terms when it is on. The fit is
the same as without the argument. `"smooth+trees"` adds `btree(name, maxdepth=1)` for each
numeric baseline with more than two distinct values and is fitted by `gamboost`.
A depth-one tree makes one split, so it can place a step such as an eligibility
cutoff that a spline spreads over its basis. `"auto"` fits both formulas on the
same folds and keeps the one with the lower out-of-fold binomial deviance.
`result.diagnostics["propensity_learner"]` holds the two deviances and the
choice, and `result.describe()` states them. When the tree formula wins under
`"auto"`, the propensities, and so the estimate, change. `effect_model` takes the
same argument for its propensity model, with the same default. With
`propensity_terms`, an explicit `propensity_learner` other than `"smooth"` is
refused.

`"auto"` and `"smooth+trees"` are opt-in. On 32 simulated draws of the enrolment
design (30 with a planted effect and two without), the tree formula had the
lower out-of-fold deviance on every draw, by 0.014 to 0.025. Against the true
propensity on six of the draws, the mean absolute error fell from 0.061 to 0.047
over all rows, and from 0.031 to 0.007 for the rows with prior earnings above
40,000, where the true propensity is about 0.01 and the smooth formula
overstated it. The ATE interval covered the truth on 26 of 32 draws with
`"auto"`, against 28 of 32 with `"smooth"`; the root mean squared error was 1250,
against 1236, and the mean interval width rose 10%. The fit took about 1.6 times
as long on one draw (27 s against 17 s). The ATE misses on the upside came from
the two-part outcome model that `"auto"` selected in that run; `"auto"` no longer
selects it, and the propensity formula does not change it. The default stays
`"smooth"` for that reason.

To refit at the step counts of a tuned run without a second tuned pass, pass `mstop=diagnostics["nuisance_mstop_summary"]` with `tune=False`.

## Recommended route: EconML or DoubleML with mboostrs nuisance learners

`DoubleMLIRM` estimates the average effect of a binary treatment with the
same doubly robust score as `aipw`. Its outcome model is `ml_g` and its
propensity model is `ml_m`, both cross-fitted over `n_folds` folds. Its
output is `coef`, `se` and `confint()`; it has no overlap report, so use
`aipw` when `overlap_report` is needed.

```python requires=doubleml
import numpy as np
import doubleml as dml
from mboostrs.sklearn import GAMBoostClassifier, GAMBoostRegressor

rng = np.random.default_rng(1)
n = 1500
x = rng.normal(size=(n, 3))
d = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x[:, 0]))).astype(float)
y = 2 * x[:, 0] + 1.5 * d + np.sin(x[:, 2]) + rng.normal(size=n)

data = dml.DoubleMLData.from_arrays(x, y, d)
irm = dml.DoubleMLIRM(data, ml_g=GAMBoostRegressor(), ml_m=GAMBoostClassifier(), n_folds=5)
irm.fit()
print("average effect", round(float(irm.coef[0]), 3), "se", round(float(irm.se[0]), 3))
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
y = 1.5 * a + 2 * x1 + x2 + rng.normal(size=n)
y = np.where(rng.uniform(size=n) > 0.15, y, np.nan)   # 15% of outcomes missing
data = {"a": a, "y": y, "x1": x1, "x2": x2}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 and x2 are measured before treatment and affect both treatment and outcome",
)
# Missing outcomes need outcome_missing="ipw"; without it they are refused.
res = mb.aipw(data, spec, folds=4, mstop=150, tune=False, seed=1,
              outcome_missing="ipw")
print(res.estimate, res.interval)
print(res.describe())
```
