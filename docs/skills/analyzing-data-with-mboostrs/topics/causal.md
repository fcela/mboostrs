# Treatment effects from observational data

A causal estimate answers a question about what would happen under a different
treatment, and the data alone do not supply that answer. The functions here
take the question as an explicit object, a `CausalSpec`, and attach the
assumptions to every result. This topic is the overview: the steps in order,
the sub-topics that hold the details, and one worked example.

## Workflow

Run the steps in this order. Each step names the sub-topic that gives its
details.

1. Declare the question with `CausalSpec`: the treatment and outcome columns,
   the baseline covariates, the estimand, the time zero and the follow-up.
   Read `causal-roles`.
2. Check the spec against the data with `causal_trap_report(frame, spec)`. A
   blocking finding means the estimand cannot be identified on those rows.
   Read `causal-roles`.
3. Choose the adjustment set: measured baseline confounders only. A variable
   measured after treatment, such as a mediator or a post-treatment selection
   variable, is excluded. Test a candidate with `ci_test`. Read `causal-roles`.
4. Estimate the average effect with `aipw`. Read `causal-average`. The
   nuisance warnings and the choice of outcome model are in
   `causal-outcome-models`.
5. Estimate how the effect varies along the covariates with `effect_model`,
   `rank_by_effect` and `effect_modifiers`. Read `causal-heterogeneity`,
   `causal-effect-model-checks` and `causal-effect-modifiers`.
6. Check overlap with `overlap_report`, and test sensitivity to unmeasured
   confounding with `confounding_sensitivity` and `confounding_benchmark`.
   Read `causal-overlap-sensitivity`. When no measured adjustment set removes
   the confounding and the design has an instrument, `iv_effect` estimates the
   effect among compliers under four stated assumptions. Read
   `causal-instruments`.
7. Report the result in the order given under Reporting.

## Reporting

A causal result states, in this order: the treatment contrast, the estimand
and target population, the effect scale and its units, the adjustment set and
the reason for it, the variables excluded as post-treatment and the reason,
how missing outcomes were handled, the overlap bounds with the full and
inside-bounds estimates, and the sensitivity grid with the units of its
parameters.

## Recommended route: EconML or DoubleML with mboostrs nuisance learners

For double machine learning, doubly robust and instrumental-variable
estimators, use EconML or DoubleML as the estimator and mboostrs models as its
nuisance learners. A nuisance learner is the model that predicts one input of
the estimator (the outcome, the treatment or the instrument) from the
covariates. The library cross-fits those models, builds the orthogonal score
and computes the standard error. mboostrs supplies the models: `GAMBoostRegressor`
and `GAMBoostClassifier`, or the linear `GLMBoostRegressor` and
`GLMBoostClassifier`, from `mboostrs.sklearn`.

The argument names are these. EconML `LinearDML` and `CausalForestDML` take
`model_y` (outcome) and `model_t` (treatment; a classifier when
`discrete_treatment=True`). EconML `DRLearner` takes `model_regression` and
`model_propensity`. DoubleML `DoubleMLPLR` takes `ml_l` (outcome) and `ml_m`
(treatment), `DoubleMLIRM` takes `ml_g` (outcome) and `ml_m` (propensity), and
`DoubleMLPLIV` takes `ml_l` (outcome), `ml_m` (instrument) and `ml_r`
(treatment).

In EconML, `X` holds the effect modifiers and `W` the other covariates. The
nuisance model sees the two stacked in that order. With a NumPy array its
columns are named `f0, f1, ...`, so a `formula=` on the nuisance model refers
to those names. Pass numeric columns only. EconML and DoubleML convert their
inputs to numbers before the nuisance model runs, so a text or category
covariate fails in the library, not in mboostrs: encode it as dummy columns
first.

The native estimators (`aipw`, `effect_model`, `ci_test`, `iv_effect`) remain
useful in these cases. They need no extra dependency. `outcome_missing="ipw"`
handles missing outcomes, which the external estimators do not do here.
`aipw` offers a two-part outcome model for a continuous outcome with many zeros,
such as earnings with non-earners, when it is requested with
`outcome_model="two_part"` (see `causal-outcome-models`). The overlap and sensitivity tools
(`overlap_report`, `confounding_sensitivity`) take an `aipw` result, so they
apply to the native route only. The R package does not expose these
estimators. None of these points makes the native estimators more accurate
than EconML or DoubleML.

```python requires=econml
import numpy as np
from econml.dml import LinearDML
from mboostrs.sklearn import GAMBoostClassifier, GAMBoostRegressor

rng = np.random.default_rng(1)
n = 1500
# Column 0 confounds treatment and outcome, column 1 changes the effect, and
# column 2 affects only the outcome. The true average effect is about 1.5.
x = rng.normal(size=(n, 3))
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x[:, 0]))).astype(int)
y = 2 * x[:, 0] + (1.5 + 0.5 * x[:, 1]) * a + np.sin(x[:, 2]) + rng.normal(size=n)

est = LinearDML(model_y=GAMBoostRegressor(), model_t=GAMBoostClassifier(),
                discrete_treatment=True, cv=4, random_state=1)
est.fit(y, a, X=x[:, [1]], W=x[:, [0, 2]])

print("average effect", round(float(est.ate(X=x[:, [1]])), 3))
print("95% interval", np.round(est.ate_inference(X=x[:, [1]]).conf_int_mean(), 3))
```

## Sub-topics

Each name is a topic of `mboostrs.agent_guide(name)`.

- `causal-roles`: the spec and its effect scale, `causal_trap_report`, the
  adjustment set, post-treatment variables, and `ci_test`.
- `causal-average`: `aipw` with its influence interval, repeats, missing
  outcomes, baseline smoothing, custom nuisance formulas and the propensity
  learner.
- `causal-outcome-models`: `aipw` nuisance steps and warnings, the choice of
  outcome model (separate, pooled, two-part), and the sparse-arm check.
- `causal-heterogeneity`: `effect_model` with per-row intervals and the
  interval methods, and `rank_by_effect`.
- `causal-effect-model-checks`: the candidate choice of `effect_model`, the
  comparison with a constant effect, and the pseudo-outcome clipping and
  effect-size warnings.
- `causal-effect-modifiers`: `effect_modifiers`, the BLP and GATES tests of
  which covariates change the effect, and projections.
- `causal-overlap-sensitivity`: `overlap_report`, the overlap-weighted target,
  `confounding_sensitivity` and `confounding_benchmark`.
- `causal-instruments`: `iv_effect`, its assumptions and the first-stage check.

## Runnable example

The example simulates a study with a baseline confounder and a known effect of
1.5 in the outcome's units. Some outcomes are missing at random.

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(1)
n = 800
x1 = rng.normal(size=n)
x2 = rng.uniform(-1, 1, n)
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1))).astype(float)
y = 1.5 * a + 2 * x1 + x2 + rng.normal(size=n)
y = np.where(rng.uniform(size=n) > 0.15, y, np.nan)
data = {"a": a, "y": y, "x1": x1, "x2": x2}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1", "x2"), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 and x2 are measured before treatment and affect both treatment and outcome",
)
res = mb.aipw(data, spec, folds=4, mstop=150, tune=False, seed=1,
              outcome_missing="ipw")
print(res.estimate, res.interval)

report = mb.overlap_report(res, data=data, bounds=(0.02, 0.98))
print(report.share_outside, report.estimate, report.estimate_inside)

fx = mb.effect_model(data, spec, terms="smooth", modifiers=("x1",),
                     outcome_missing="ipw", seed=1)
frame = fx.predict({"x1": [-1.0, 0.0, 1.0], "x2": [0.0, 0.0, 0.0]}, interval=True)
print(frame["estimate"].to_list())

grid = mb.confounding_sensitivity(res, outcome_effects=[0.5, 1.0],
                                  imbalances=[-0.1, 0.1])
print(grid.shape)
```
