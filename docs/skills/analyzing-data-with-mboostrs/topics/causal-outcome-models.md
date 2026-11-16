# Outcome models and nuisance steps in aipw

Part of step 4 of the `causal` workflow: how the outcome model is chosen, how the nuisance steps are set, and the warnings that name them.

## Nuisance warnings, step counts and outcome models

Nuisance warnings are issued once per call, after all folds have run. Each one
names its nuisance model and the argument that controls it:

- A prediction outside a spline's training range. The text names the model
  (propensity, response, pooled outcome, control-arm outcome or treated-arm
  outcome) and `aipw(extrapolation=...)`. `extrapolation="warn"` (the default)
  predicts and warns; `"clip"` holds the prediction at the range edge;
  `"allow"` is silent; `"error"` refuses.
- A tuned step count on a flat cross-validated curve. The chosen count's risk
  lies within `flat_tolerance` (default 0.01, or 1%) of the risk at step 0, so
  the nuisance is close to a constant. The text names `aipw(flat_tolerance=...)`;
  `flat_tolerance=None` turns the check off.

`mstop` is the first step grid of each nuisance model. `max_mstop` (default
1600, and never below `mstop`) is the largest step count the search reaches.
With the default `tune=True`, the grid doubles while the cross-validated risk
still falls. `result.diagnostics["nuisance_mstop"]` lists each nuisance's
chosen step count per fold and whether any fold reached `max_mstop` with the
risk still falling. In that case aipw warns and names `max_mstop`; raise it to
search further. With `tune=False` every nuisance runs `mstop` steps.

`outcome_model="cv"` (the default) fits each candidate on the cross-fitting folds already in
use and keeps the one with the lowest out-of-fold loss on the observed outcomes
(weighted mean binomial deviance for a binary outcome, weighted mean squared
error for a continuous one, from the cross-fitted `mu0` and `mu1` at each row's
own arm). The candidates are `"pooled"`, `"separate"`, `"two_part"` (a continuous
outcome that is zero or positive with a zero) and `"pooled_interactions"` (when
`"auto"` would choose it). Each loss and the chosen label are in
`result.diagnostics["outcome_model_cv"]`. The outcome step runs once per
candidate. The choice and its reason are in
`result.diagnostics["outcome_model_choice"]`, and `result.describe()` states
them.

`outcome_model="auto"` fits `"pooled"` without interactions when the data have
at most 1,500 rows, the size of the small job-training draws described below.
With more rows it runs the sparse-arm check below on the cross-fitted
propensity. A flagged bin gives `"pooled"` with `outcome_interactions=True`; no
flagged bin gives `"separate"`. `"auto"` never selects `"two_part"`, which you
request by name.

`outcome_model="separate"` fits one outcome model per arm, each on that arm's
rows. `outcome_model="pooled"` fits one outcome model on all observed rows, with
the treatment as a linear term. The treated arm then uses the control arm's
baseline shape plus one estimated shift. `outcome_interactions=True` needs the
pooled model (with `"auto"` it selects it) and multiplies each baseline term by
the treatment, so the baseline shape can differ by arm. `outcome_interactions=True`
with `"separate"` or `"two_part"` raises an error.

`outcome_model="two_part"` is for a continuous outcome that is zero or positive
with at least one zero, such as earnings that are zero for a person without a
job. It fits two models on the same rows, each with the treatment as a linear
term. A Bernoulli model gives the probability P(Y > 0 | x, a) on every observed
row. A Gaussian model gives the mean m of log Y on the positive rows. The
outcome mean is `P(Y > 0 | x, a) * exp(m + s2 / 2)`, where `s2` is the residual
variance of log Y. Each strictly positive baseline whose skewness exceeds 1, such
as earnings in dollars, enters the log model as its logarithm, because a linear
term in dollars lets the log mean grow with the raw value and the exponential
then amplifies it. The employment model keeps the raw baselines. Both models
hold each numeric covariate to its training range at prediction time, whatever
`extrapolation` says. A negative outcome, a binary outcome, or an outcome with
no zeros is refused, with the reason named. The formula of both parts is in
`result.nuisance["terms"]["outcome"]`, and a custom `outcome_terms` is used as
written, with no logged copy. The two-part model has no treatment interactions:
a log model with interactions gave means in the millions on some draws.

`result.diagnostics["outcome_range"]` counts the fitted means `mu0` and `mu1`
that lie more than one width of the observed outcome range beyond either end of
that range. Those rows are warned about; the estimate is not changed.

On 12 draws of 12,000 rows from the job-training design (10 with an effect, 2
without), `"auto"` had an ATE RMSE of 1,116 dollars on the effect
draws and a mean absolute error of 28 dollars on the two no-effect draws. The
bounded two-part model had 714 and 490. The default was kept because the two-part
model is worse on the no-effect draws.

`result.diagnostics["sparse_arms"]` counts treated and control weight in each
propensity bin of the cross-fitted propensity, with the propensity range and the
covariate ranges of each bin. Bins hold equal total weight: one bin per 200
units, between 4 and 10 bins. A bin is flagged for an arm when it holds more than 5% of the
target weight (all rows for the ATE, treated rows for the ATT) and fewer than 20
weighted rows of that arm. A flagged bin warns whenever the fit uses separate
models or interactions, and the warning names the model that was fitted.

At n = 1,500 on the job-training design the flag fired in all 25 draws of a
check. The rule that applies above 1,500 rows had a mean absolute ATE error of
2,487 dollars, against 1,404 for `outcome_model="pooled"`; 1,500 is the largest
row count those draws cover, and it is the `"auto"` threshold.

On a simulated design with n = 12,000, a tail region holding about 16% of rows
and 0.1 to 0.2% of all rows treated, and an effect equal to half the baseline,
the flag fired. Over 10 seeds the pooled model with interactions had a bias of
-107 dollars against -131 for separate models, for a true effect of 1,700
dollars. With no
effect the pooled model with interactions had a bias of 0.6 dollars and its
intervals covered zero in 8 of 10 seeds; separate models had a bias of -94
dollars and covered zero in none. On a balanced design with no flag, separate
models and the pooled model with interactions had the same bias (0.0035) and
RMSE (0.021) in the ATE. The `"auto"` rule therefore keeps separate models where
nothing is flagged.

## Recommended route: outcome models in EconML or DoubleML

The outcome model of an external estimator is the nuisance learner `model_y`
(EconML) or `ml_l` and `ml_g` (DoubleML). Any mboostrs estimator can take that
place, including one with its own `formula=`. The formula's column names are
`f0, f1, ...` when the input is a NumPy array, in the order EconML stacks them:
`X` first, then `W`. In the example, `f0` is the effect modifier, `f1` the
confounder and `f2` the other covariate. The formula's left side is a placeholder;
the outcome is supplied by `fit`.

```python requires=econml
import numpy as np
from econml.dml import LinearDML
from mboostrs.sklearn import GAMBoostClassifier, GAMBoostRegressor

rng = np.random.default_rng(2)
n = 1500
x = rng.normal(size=(n, 3))
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x[:, 0]))).astype(int)
y = 2 * x[:, 0] + (1.5 + 0.5 * x[:, 1]) * a + np.sin(x[:, 2]) + rng.normal(size=n)

# A smooth term for the modifier, a linear term for the confounder and a
# smooth term for the other covariate.
outcome = GAMBoostRegressor(formula="y ~ bbs(f0, df=5) + bols(f1) + bbs(f2, df=4)")
est = LinearDML(model_y=outcome, model_t=GAMBoostClassifier(),
                discrete_treatment=True, cv=4, random_state=2)
est.fit(y, a, X=x[:, [1]], W=x[:, [0, 2]])
print("average effect", round(float(est.ate(X=x[:, [1]])), 3))
```

## Runnable example

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(3)
n = 800
x1 = rng.normal(size=n)
a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1))).astype(float)
# Earnings are zero for 30% of rows, so the outcome has zeros and positive values.
earn = np.where(rng.uniform(size=n) < 0.3, 0.0,
                np.exp(9 + 0.5 * x1 + 0.3 * a + rng.normal(scale=0.4, size=n)))
data = {"a": a, "earn": earn, "x1": x1}

spec = mb.CausalSpec(
    treatment="a", outcome="earn", covariates=("x1",), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="treatment assignment", follow_up="one period",
    adjustment_rationale="x1 is measured before treatment and affects both treatment and earnings",
)
res = mb.aipw(data, spec, folds=3, mstop=60, tune=False, seed=1,
              outcome_model="two_part")
print(res.diagnostics["outcome_model_choice"])
print(res.estimate, res.interval)
```
