# Overlap and sensitivity to unmeasured confounding

Step 6 of the `causal` workflow: whether treated and control rows exist at each covariate pattern, and how far an omitted confounder could move the estimate.

## Overlap

Overlap asks whether treated and control rows exist at each covariate pattern.
Where they do not, the propensity is near 0 or 1, and the estimate there rests
on the model rather than on comparable rows. `overlap_report` gives the share
of rows with propensity outside the bounds, the covariate region where most
rows are outside, and the estimate restricted to the rows inside the bounds.

```
report = mb.overlap_report(res, data=data, bounds=(0.02, 0.98))
print(report.share_outside, report.estimate, report.estimate_inside)
```

The bounds `(0.02, 0.98)` are a convention. State the bounds used. When the
result carries a spec, `overlap_report` uses the spec's `overlap_bounds` unless
`bounds=` is passed. Report the
full-population estimate and the inside-bounds estimate side by side, and say
that the second one is the effect for the population with overlap, which can
differ from the target population. `aipw` and `effect_model` store the same
report in `diagnostics["overlap"]`.

The book page `usage/causal`, section "Worked case: a thin treated tail", shows a
simulated design where the default estimate is -3,169 against a true 4,946 and the
inside-bounds estimate is 4,572.

For a new row whose estimated propensity lies outside the bounds,
`fx.predict(..., interval=True)` sets `outside_overlap` to true and emits one
warning with the count. Treat those effects as extrapolations.

## Overlap-weighted target with aipw

`target` chooses the population the estimate describes. `target="all"` (the
default) is the ATE over every row. `target="overlap"` is the overlap-weighted
ATE: each row counts with weight e(1 - e), where e is its cross-fitted
propensity, so rows with a propensity near 0 or 1 count little. The weights are
at most 1/4 and the propensity is not divided, so `propensity_clip` does not
apply to this target. Use it when `result.describe()` reports that many rows
have a propensity below 0.05 or above 0.95. In those rows the all-rows score
divides a residual by a propensity near 0, the outcome model extrapolates to
where almost no rows are treated, and clipping the propensity changes the mean
of the score. On a reduced simulated design (n = 3,000, propensity near 0.004
for 21% of rows, an effect proportional to the baseline, 20 seeds), the
all-rows ATE had a bias of -86 dollars and root mean squared error of 90, and
its 90% interval covered its truth in 1 of 20 seeds. With `target="overlap"` the
bias was +8 dollars, the root mean squared error 15, and the interval covered
the overlap truth in 17 of 20 seeds. The overlap target estimates a different
quantity from the ATE: the two agree only when the propensity is one half
everywhere. `describe()`, `repr()` and `diagnostics["target"]` state the target.
Only `estimand="ate"` accepts `target="overlap"`.

## Sensitivity to unmeasured confounding

No function here shows that unmeasured confounding is absent.
`confounding_sensitivity` shows how far the estimate would move under stated
assumptions. Give the change in the outcome per unit of an omitted confounder
(`outcome_effects`) and the confounder's treated-minus-control difference
after adjustment (`imbalances`), in the confounder's units. Their product is
the assumed bias, and the grid reports the adjusted estimate and whether its
sign changes.

```
grid = mb.confounding_sensitivity(res, outcome_effects=[0.5, 1.0],
                                  imbalances=[-0.1, 0.1])
```

`confounding_benchmark` reruns the analysis with one measured covariate
removed at a time. Each shift is a scale for choosing the grid, not a bound on
the omitted confounder. The canonical call is `mb.confounding_benchmark(data,
res)`, data first. `mb.confounding_benchmark(res, data)` gives the same rows,
because the two arguments are told apart by type. `confounding_sensitivity`
takes the result first, since its estimate can also be given as plain numbers.

A result fitted with `propensity_terms=` or `outcome_terms=` is refitted with
those formulas minus every term that names the omitted covariate. The rows
report the reduced formulas. If a custom formula has no term left once the
covariate is removed, the call refuses before any refit and names the formula.

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
data = {"a": a, "y": y, "x1": x1, "x2": x2}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 and x2 are measured before treatment and affect both treatment and outcome",
)
res = mb.aipw(data, spec, folds=3, mstop=60, tune=False, seed=1)
report = mb.overlap_report(res, data=data, bounds=(0.02, 0.98))
print(report.share_outside, report.estimate, report.estimate_inside)

# The overlap-weighted target counts rows near propensity 0 or 1 less.
res_overlap = mb.aipw(data, spec, folds=3, mstop=60, tune=False, seed=1,
                      target="overlap")
print(res_overlap.estimate)

grid = mb.confounding_sensitivity(res, outcome_effects=[0.5, 1.0],
                                  imbalances=[-0.1, 0.1])
print(grid.shape)
```
