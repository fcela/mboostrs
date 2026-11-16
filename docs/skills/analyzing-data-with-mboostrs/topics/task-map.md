# Task map

This page names the function that performs each analysis task in three
places: the Python package (`mboostrs`), the R package (`mboostrs`, exports
in its NAMESPACE) and the Rust core (`crates/mboostrs-core/src/`). The Rust
column gives the file and the function or type. Python names are attributes
of the package (`glmboost`) or of a class (`GLMBoostModel.cvrisk`). R names
are the exported function or the generic the method is registered under.

A cell with `—` means the frontend has no function for that task. The Notes
column then says what the frontend does instead, or that the row has no core
entry point. Python has the widest set of functions. The R package covers
fitting, tuning, the distributional diagnostics, coefficient and effect
extraction, stability selection and loading artifacts. It has no held-out
evaluation helper, no conformal intervals and no model comparison table.

## Fit

| Python | R | Rust core | Notes |
|---|---|---|---|
| `glmboost` | `glmboost` | `engine.rs::try_fit` | Linear base-learners. The boosting loop is shared by all models. |
| `gamboost` | `gamboost` | `engine.rs::try_fit` | Smooth base-learners such as `bbs`, built in `learners.rs`. |
| `glmboostLSS` | `glmboostLSS` | `lss.rs::try_fit_lss` | Distributional fit: one boosting loop per distribution parameter. |
| `gamboostLSS` | `gamboostLSS` | `lss.rs::try_fit_lss` | Smooth terms for every distribution parameter. |
| `boost_control` | `boost_control` | `control.rs::BoostControl` | Step count, step length and the other fitting controls. |
| `discover` | — | — | Python only. Runs screening, fitting, tuning, refit and prediction under fixed rules and logs each decision. |

## Tune

| Python | R | Rust core | Notes |
|---|---|---|---|
| `GLMBoostModel.cvrisk` | `cvrisk` | `cv.rs::try_cvrisk` | Cross-validated risk at each step for a linear model. |
| `GAMBoostModel.cvrisk` | `cvrisk` | `cv.rs::try_cvrisk` | Cross-validated risk at each step for a smooth model. |
| `cvrisk_glmboostLSS` | `cvrisk_glmboostLSS` | `lss.rs::try_cvrisk_lss` | Cross-validated risk for a distributional linear fit, one optimum per parameter. |
| `tune_mstop` | `mstop` | `cv.rs::try_cvrisk` | Python searches the step count and doubles the grid while the risk still falls. R reads the optimal step from a `cvrisk` result. |
| `cv_risk` | — | `scoring.rs::response_score_path` | Python only. Out-of-fold risk of a formula for any family, with the held-out predictions retained. |
| `profile_mstop` | — | — | Python only. Refits or re-predicts at several step counts. In R, call `predict()` at each step count. |

`tune_mstop` also warns when the chosen step count's cross-validated risk is
within 1% of the intercept-only risk (step 0). The curve is then flat and the
fit is close to a constant. `flat_tolerance=` sets the gap, and `None` turns the
check off.

## Evaluate

| Python | R | Rust core | Notes |
|---|---|---|---|
| `evaluate_specification` | — | `scoring.rs::response_score` | Python only. Held-out score of one specification on a fixed fold set. |
| `evaluate_distribution` | — | `lss.rs::make_family` | Python only. Density, PIT and CRPS of a fitted distribution on rows the fit did not use. The core provides the family definitions used for fitting; these scores are computed in Python. |
| `holdout` | — | — | Python only. Splits a frame into a search part and a test part. |
| `GLMBoostModel.risk` | `risk` | `engine.rs::try_fit` | Training risk at each step. This is not held-out risk; use `cvrisk` for that. |

## Diagnose

| Python | R | Rust core | Notes |
|---|---|---|---|
| `residual_frame` | `residual_frame` | — | Residuals of a fitted distributional model, as a frame. |
| `pit` | `pit` | — | Probability integral transform values of a distributional model. |
| `q_stats` | `q_stats` | — | Quantile-residual summaries of a distributional model. |
| `calibration_frame` | — | — | Python only. Observed against predicted frequencies by bin. |
| `GAMBoostModel.concurvity` | `concurvity` | `gam/concurvity.rs::concurvity` | Concurvity between smooth terms. |
| `GLMBoostModel.support_report` | `support_report` | `support.rs::joint_support` | Share of new rows outside the range of the training data. |
| `same_family_gap` | — | — | Python only. Compares a linear boost at its cross-validated step with the same-family GLM. |
| `column_screen` | — | `screen.rs::column_diagnostics` | Python only. Scatter diagnostics of the response against each column. |
| `GLMBoostModel.varimp` | `varimp` | — | Variable importance. No core entry point is listed for this row. |

## Compare

| Python | R | Rust core | Notes |
|---|---|---|---|
| `gaic_table` | `AIC` | `engine.rs::try_compute_aic` | Information criterion for fits on the same sample. R compares fits one at a time with `AIC()`. |
| `GLMBoostModel.aic` | `AIC` | `engine.rs::try_compute_aic_trace` | Criterion at each step of one fit. |
| `compare_models` | — | `scoring.rs::response_score` | Python only. Several metrics for several models. |
| `compare_specifications` | — | `selection.rs::paired_gain` | Python only. Scores several formulas on the same folds and reports paired differences. |
| `compare_summaries` | — | — | Python only. Puts several fitted models side by side. |
