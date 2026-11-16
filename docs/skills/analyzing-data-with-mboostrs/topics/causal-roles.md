# Causal question, design checks and the adjustment set

Step 1 to 3 of the `causal` workflow: the spec, the checks on it, and the variables it adjusts for.

## Declaring the question

`CausalSpec` names the treatment column, the outcome column, the baseline
covariates, the estimand and the timing. A spec with no time zero, no follow-up
window or no adjustment rationale is refused, because each of those changes
what the number means.

- `treatment` and `outcome` are column names. `treatment_contrast=(control,
  treated)` gives the two labels of a binary treatment; the default is `(0, 1)`.
- `estimand="ate"` is the average effect over the target population. `"att"`
  is the average effect among the treated.
- `outcome_type="continuous"` gives a difference in the outcome's units.
  `"binary"` gives a risk difference.
- `time_zero` says when treatment is assigned. `follow_up` says how long the
  outcome is measured after it.
- `adjustment_rationale` says in words why the listed covariates are enough.
- `propensity_clip` (default `(0.01, 0.99)`, or `None` for no clipping),
  `outcome_missing` (default `"error"`) and `overlap_bounds` (default
  `(0.02, 0.98)`) are the spec's handling choices. `aipw`, `effect_model`,
  `overlap_report` and `causal_trap_report` read them when the call does not
  pass its own argument. A value passed to a call always wins, and
  `propensity_clip=None` means no clipping.

The effect scale is part of the result. A continuous outcome in dollars gives
a difference in dollars. A binary outcome gives a difference in probabilities,
which is reported as percentage points. A relative effect is not an output of
these functions. To report one, model the log of a positive outcome as the
outcome column and read the estimate as a difference in log units; the
relative change is `exp(estimate) - 1`.

## Checking the design with causal_trap_report

`mb.causal_trap_report(frame, spec)` checks the spec against the data before
an estimate is computed. It returns a `TrapReport` whose `findings` have a
`name`, a `severity` (`"blocking"` or `"advisory"`) and a `message`. A
blocking finding means the estimand is not identified on these rows, for
example a column the spec names is absent or one arm is empty. Columns in
`post_treatment` are not checked for presence: a frame may omit them on purpose
(a training file holds them, a test file does not), and the design does not
adjust for them.

A missing outcome is reported according to `outcome_missing`. With the
default `"error"`, the report says the estimate would cover only the rows with
an observed outcome, a subpopulation of the spec's target. With
`outcome_missing="ipw"`, the finding is `outcome_missing_ipw` and says the
target is all rows, observed or not, under a missing-at-random assumption. The
argument `outcome_missing=` overrides the spec's value.

## Choosing the adjustment set

Adjust for every variable measured before treatment that is a cause of the
treatment, of the outcome, or of both (the disjunctive cause criterion,
VanderWeele 2019). A variable that affects both is a baseline confounder;
including it removes the difference in outcomes that comes from the
treatment groups having different values of that variable. A variable that
affects only the outcome costs nothing in bias and narrows the interval.

Do not drop a pre-treatment variable because `ci_test` found no link to the
treatment. A large p-value means no association was detected at this sample
size, which is not evidence of independence: a weak confounder fails that
test and is then left unadjusted. In the job-training benchmark, an analyst
who pruned by test left out `smsa`, a true confounder, on 4 of 8
replicates, and those adjustment sets were invalid. Use `ci_test` to find
variables to exclude for a reason (an instrument: strongly predicts
treatment and is independent of the outcome given treatment and the other
covariates), not to prune confounders. When a test's `status` is
`"dependent"`, nothing was tested; decide that variable from the study
design.

Do not adjust for a variable measured after treatment starts. There are two
common kinds.

- A mediator lies on the path from treatment to outcome. Adjusting for it
  removes the part of the effect that runs through it, and the remaining
  estimate answers a different question.
- A post-treatment selection variable decides whether the outcome is seen at
  all. Survey response after a program is one example. Treatment can change
  who responds, and the people who respond also share unmeasured causes of the
  outcome. Conditioning on response, or analysing responders only, opens a
  path between treatment and outcome that the design did not intend.

`CausalSpec(post_treatment=...)` refuses a listed variable in the adjustment
set. The list is the analyst's statement of which variables are downstream of
treatment, so it must be written from the study's timeline, not from which
variables improve the fit.

Predictive importance is not an identification argument. A variable
measured after treatment does not become admissible because it predicts the
outcome well.

## Conditional independence with ci_test

`ci_test(data, x, y, given=...)` tests whether `x` and `y` are independent
given the columns in `given`, with the generalised covariance measure. `x` may
be numeric (a 0/1 column is fine) or categorical. A categorical `x` is tested
jointly as a factor: each level after the first observed level is one 0/1
indicator, residualized on `given` like a numeric `x`, and the indicators are
tested together with a chi-squared test with one degree of freedom per
indicator. A categorical `y` is accepted when it has two observed levels. With
more levels, test one level at a time by passing `data["y"] == level` as a 0/1
column for each level of interest. `given` takes numeric, boolean, string or
categorical columns; a factor is coded with treatment contrasts.

```
test = mb.ci_test(data, x="income", y="enrolled", given=["age", "region"], folds=3)
print(test.describe())
```

Before fitting, the conditioning set is checked for exact linear dependence.
A column that is an exact combination of the others (for example
`experience = age - education - 6`) is removed and listed in `test.dropped`.
When `x` or `y` is itself such a combination, no test is run: `status` is
`"dependent"`, and `statistic`, `p_value` and `residual_correlation` are NaN.
The note names the identity and the columns in it (for example `experience =
-6 * intercept + age - education` with `age`, `education` and `experience`),
and says which one `ci_test` suggests dropping: the last conditioning column in
the identity. Dropping that column leaves the rest of the set, so the test runs.
Each such decision, and each regression that reached the step cap, is one line
of `test.notes`.

Each regression is a boosted fit tuned by inner cross-validation, capped at
1,000 steps (`mboostrs.causal.CI_MAX_MSTOP`). A regression whose out-of-fold
risk is still falling at the cap says so in `test.notes`. The residuals then
come from a fit that has not converged, so read the statistic as approximate
when that note appears. Extrapolation to held-out rows outside a conditioning
column's training range follows `extrapolation=` (`"warn"`, `"error"`,
`"allow"` or `"clip"`). Under `"warn"` a call issues one warning, naming the
column of `given`.

If `y` has missing values, pass `outcome_missing="ipw"`. Read `result.status` before `p_value`: `"dependent"` means `x` or `y` is an exact combination of the conditioning set and nothing was tested.

## Runnable example

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(1)
n = 800
x1 = rng.normal(size=n)                    # baseline confounder
x2 = rng.uniform(-1, 1, n)                 # baseline covariate, affects the outcome only
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1))).astype(float)
y = 1.5 * a + 2 * x1 + x2 + rng.normal(size=n)
data = {"a": a, "y": y, "x1": x1, "x2": x2}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 and x2 are measured before treatment",
)
report = mb.causal_trap_report(data, spec)
print([(finding.name, finding.severity) for finding in report.findings])

# Check whether x2 is associated with treatment after conditioning on x1.
test = mb.ci_test(data, x="x2", y="a", given=["x1"], folds=3, seed=1)
print(test.describe())
```
