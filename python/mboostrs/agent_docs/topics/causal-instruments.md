# Instruments

Step 6 of the `causal` workflow, for a design with an instrument.

## Instrumental variables with iv_effect

When no set of measured covariates removes the confounding of a binary
treatment, an instrument can still identify an effect. `iv_effect(data, spec,
instrument)` estimates one for a binary treatment and a continuous or binary
instrument. The instrument has to satisfy four assumptions, which the data do
not check: relevance (it shifts the treatment after the covariates),
independence (it shares no unmeasured cause with the outcome), exclusion (it
reaches the outcome only through the treatment) and monotonicity (no unit is
moved out of treatment by it).

```
iv = mb.iv_effect(data, spec, "offer", folds=5, seed=0)
print(iv.describe())
iv.estimate, iv.interval, iv.first_stage_f
```

The estimator is the partially linear IV model with cross-fitting (DML-IV).
Out of fold, boosted models predict the outcome, the treatment and the
instrument from the covariates. Each variable is then replaced by its residual
(`y`, `d` and `z` below), and the estimate is the ratio `sum(z * y) / sum(z * d)`.
The estimand is a weighted average of the treatment effect among compliers, the
units whose treatment the instrument switches on. With one effect shared by
every unit, that effect is the estimand.

- `iv.first_stage_f` is the partial F of the instrument on the treatment after
  the covariates, and `iv.first_stage_r2` its partial R2. Below 10
  (`weak_threshold`), the result is marked `iv.weak_instrument` and a warning
  is issued. A weak instrument biases the ratio toward the confounded
  comparison and makes the interval too narrow.
- `iv.interval` is a normal interval at `level`, from the influence-function
  variance. It holds the nuisance predictions fixed and does not include
  variation from the choice of folds.
- `spec` supplies the treatment, outcome and covariates. Its `estimand` field
  is not used, because the estimand here is the complier effect.
- `iv.to_frame()` gives the residuals and the influence value of each row.
- A missing outcome is refused by default, with a message that names
  `outcome_missing="ipw"`. Pass `outcome_missing="ipw"` to keep every row: a
  response model for P(outcome observed | covariates, treatment, instrument) is
  cross-fitted, each observed row is weighted by its inverse, and
  `iv.nuisance["response_propensity_min"]` and `..._max` give the range of the
  fitted probabilities. The result assumes the outcome is missing at random given
  those columns. `propensity_clip` (default from the spec, 0.01 to 0.99) bounds
  the probabilities. With no missing outcome the argument changes nothing.

## Recommended route: DoubleMLPLIV with mboostrs nuisance learners

For a continuous treatment `d` with an instrument `z`, `DoubleMLPLIV`
estimates a linear effect of `d` on `y`. Its three nuisance models are
`ml_l` for E[y | X], `ml_m` for E[z | X] and `ml_r` for E[d | X], each
cross-fitted. The instrument assumptions stated for `iv_effect` in this topic
apply; the library does not test them.

```python requires=doubleml
import numpy as np
import doubleml as dml
from mboostrs.sklearn import GAMBoostRegressor

rng = np.random.default_rng(4)
n = 1500
x = rng.normal(size=(n, 3))
z = rng.normal(size=n)
u = rng.normal(size=n)  # unmeasured confounder of d and y
d = 0.8 * z + 0.5 * x[:, 0] + u + 0.5 * rng.normal(size=n)
y = 1.5 * d + x[:, 0] + 2 * u + rng.normal(size=n)

data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
pliv = dml.DoubleMLPLIV(data, ml_l=GAMBoostRegressor(), ml_m=GAMBoostRegressor(),
                        ml_r=GAMBoostRegressor(), n_folds=5)
pliv.fit()
print("effect of d", round(float(pliv.coef[0]), 3), "se", round(float(pliv.se[0]), 3))
```

## Runnable example

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(2)
n = 1000
x1 = rng.normal(size=n)
u = rng.normal(size=n)                        # unmeasured; drives treatment and outcome
z = rng.binomial(1, 0.5, n).astype(float)     # the instrument: an offer
a = (0.8 * z + 0.5 * x1 + u + rng.normal(size=n) > 0.5).astype(float)
y = 1.5 * a + 2 * x1 + u + rng.normal(size=n)
data = {"a": a, "y": y, "x1": x1, "z": z}

spec = mb.CausalSpec(
    treatment="a", outcome="y", covariates=("x1",), estimand="ate",
    target_population="simulated sample", unit="person",
    time_zero="offer", follow_up="one period",
    adjustment_rationale="x1 is measured before the offer",
)
iv = mb.iv_effect(data, spec, "z", folds=3, seed=0)
print(iv.describe())
print(iv.estimate, iv.interval, iv.first_stage_f)
```
