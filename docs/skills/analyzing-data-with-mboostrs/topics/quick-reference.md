# Calls, formulas, and returned objects

Use `mb.api_card("name")` or `python -m mboostrs sig NAME` before an
unfamiliar call. These conventions differ from sklearn and from R's mboost.

| Operation | Call / returned value |
|---|---|
| Input data | Polars/Pandas DataFrame or dict of equal-length columns; new data supplies the predictors with training-compatible types |
| Smooth boosting | `mb.gamboost(formula, data, family=..., control=mb.boost_control(mstop=...))` returns `GAMBoostModel` |
| CV stopping search | `mb.tune_mstop(formula, data, family=..., folds=..., seed=...)` returns `MstopTuning`: `.mstop`, `.at_boundary`, `.risk`, `.model` |
| Direct REML | `mb.gam(formula, data, family="gaussian")` returns `GamRefit`; inspect `.convergence` and `.summary()` |
| Draft specification | `mb.autospec(data, "y")`: `.formula`, `.family`, `.report()` |
| Classify columns | `mb.classify_columns(data, "y")` returns a list; column name is `.name`, not `.column` |
| Screen columns | `mb.column_screen(data, "y")` returns a Polars frame; screening is exploratory, not a test of adjusted effects |
| Predict means/probabilities | `model.predict(newdata=rows, type="response")` for boosted models returns a NumPy array |
| Term contribution | `model.predict(newdata=rows, which=learner_name, type="link")`; choose a name from `model.base_learner_names()` |
| Persist boosted models | `mb.save_model(model, path)` returns a path; `mb.load_model(path)` returns a fitted model |

## Family starting points

| Response | Boosting family | Response prediction |
|---|---|---|
| Real-valued continuous | `mb.Gaussian()` | conditional mean |
| Binary, encoded 0/1 | `mb.Binomial()` | probability of 1 |
| Nonnegative integer counts | `mb.Poisson()` | expected count; check dispersion |
| Strictly positive continuous | `mb.GammaReg()` | conditional mean; zeros need another model |

Support narrows the candidates; it does not establish the correct distribution.
Read "distributional-branch" when variance, tails, zero mass, or multiple
distribution parameters matter. Do not pass class labels as an arbitrary
numeric regression response. Check the signature for `family=` versus
`families=` on the chosen fitting function.

## Formula rules

Write learners inside a string; do not import `bbs` or `bols` from Python.
`"y ~ bols(x)"` is linear, `"y ~ bbs(x, df=4)"` smooth,
`"y ~ bols(category)"` an unpenalized categorical effect, and
`"y ~ brandom(site, df=4)"` a penalized group effect. A bare numeric name
in `gamboost` is a smooth; in `glmboost` it is a linear learner. Be explicit.
`"bbs(x, df=4) %X% bbs(z, df=4)"` builds a tensor interaction; read
"pitfalls" about its df floor and cost before comparing it to main effects.
In boosting, matching learner df helps comparisons; in direct REML,
smoothing parameters are estimated and `knots=` controls basis capacity.

## Binary prediction and saving a model

This is an API smoke example with a deliberately fixed stop count, not a
tuned performance estimate. Use "golden-path" to tune and evaluate a model.

```python
import tempfile
from pathlib import Path
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(3)
x = rng.normal(size=200)
y = rng.binomial(1, 1 / (1 + np.exp(-2 * x)))
model = mb.gamboost("y ~ bols(x)", {"x": x, "y": y},
                    family=mb.Binomial(), control=mb.boost_control(mstop=40))
new = {"x": np.array([-1.0, 0.0, 1.0])}
probability = model.predict(newdata=new, type="response")
assert probability.shape == (3,)
assert np.all((probability >= 0) & (probability <= 1))
with tempfile.TemporaryDirectory() as directory:
    path = mb.save_model(model, Path(directory) / "model.json")
    restored = mb.load_model(path)
    np.testing.assert_allclose(restored.predict(newdata=new, type="response"), probability)
print(probability)
```

For distributional models use parameter-specific prediction; read
"distributional-branch" and the model's signature. For direct REML intervals,
read "direct-reml-gam": an effect interval, a mean-response interval, and a
new-observation interval are different outputs. Do not assume every model
class accepts the same prediction or persistence options.
