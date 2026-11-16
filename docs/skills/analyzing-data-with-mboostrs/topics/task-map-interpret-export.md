# Task map: interpret, predict with uncertainty and export

## Interpret

| Python | R | Rust core | Notes |
|---|---|---|---|
| `GLMBoostModel.coef` | `coef` | `path.rs::coef_glmboost` | Coefficients of a linear model at a chosen step. |
| `GAMBoostModel.partial_effects` | `partial_effects` | `path.rs::coef_gamboost` | Smooth term curves on a grid of the covariate. |
| `stabsel` | `stabsel` | `stabsel.rs::stabsel_parameters` | Stability selection: how often each term is selected across subsamples. The recommended path for deciding which inputs carry signal (the only selector with an error-rate guarantee, PFER). |
| `average_marginal_effect` | — | `contrast.rs::cluster_average` | Python only. Average change in the prediction for a small change in one variable. The cluster averaging is the core's `cluster_average`. |
| `prediction_contrast` | — | `contrast.rs::cluster_average` | Python only. Average change in the prediction between two values of one variable. |
| `accumulated_local_effect` | — | — | Python only. Centered effect curve of one variable. |
| `prediction_profile` | — | — | Python only. Average prediction over a grid of one variable, with individual curves. |
| `average_elasticity` | — | — | Python only. Average percentage change in the prediction for a percentage change in one variable. |
| `sensitivity` | — | — | Python only. How far the fitted function moves across each variable's range. |
| `term_report` | — | — | Python only. Refits without each term and reports what each term contributes. |
| `llm_explain_prompt` | `llm_explain_prompt` | — | Renders a fitted model as text for a language model. No numerics. |
| `llm_describe` | `llm_describe` | — | Sends that text to a language model. No numerics. |

## Predict with uncertainty

| Python | R | Rust core | Notes |
|---|---|---|---|
| `GLMBoostModel.predict` | `predict` | `path.rs::predict_glmboost` | Point predictions at a chosen step, for new rows. |
| `GLMBoostModel.confint` | `confint` | — | Intervals for coefficients or effects. No core entry point is listed for this row. |
| `LSSModel.prediction_interval` | `prediction_interval` | — | Predictive interval from the fitted distribution. The quantiles come from each family object. |
| `LSSModel.centiles` | `centiles` | — | Quantiles of the fitted distribution at chosen probabilities. |
| `conformal_interval` | — | — | Python only. Distribution-free intervals from a calibration split. The calibration is not in the Rust core. |

## Export

| Python | R | Rust core | Notes |
|---|---|---|---|
| `save_model` | — | `artifact.rs::Artifact` | Writes a JSON artifact. R cannot write one; it reads Python artifacts with `load_model`. |
| `load_model` | `load_model` | `artifact.rs::Artifact` | Reads a JSON artifact written by `save_model`. |
| `GLMBoostModel.summary` | `summary` | — | Model summary. Python returns it as a record with `summary(as_dict=True)`. |
| `GLMBoostModel.path_frame` | `path_frame` | `path.rs::path_rows` | One row per step with the risk and the selected term. |
| `GLMBoostModel.to_dict` | — | — | Python only. JSON-safe record of the fitted model. |

## Worked example

Fit a linear model, choose the step count by cross-validation, refit at that
step and predict.

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(0)
x = rng.normal(size=200)
data = {"y": 2 * x + rng.normal(size=200), "x": x}

fit = mb.glmboost("y ~ x", data=data, control=mb.boost_control(100, 0.1))
cv = fit.cvrisk(type="kfold", B=5, seed=1, parallel=False)
refit = mb.glmboost("y ~ x", data=data,
                    control=mb.boost_control(cv.optimal_mstop, 0.1))
print(refit.predict({"x": np.array([0.0, 1.0])}))
```
