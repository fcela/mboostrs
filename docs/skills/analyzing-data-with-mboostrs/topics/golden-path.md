# First analysis: tune, predict, evaluate

This example runs on the installed package with only its required dependencies.
It uses independent synthetic rows and Gaussian regression. For your data,
choose the response, allowed predictors, response family, missing-data policy,
and evaluation unit first. A formula is not a sklearn estimator: pass the
formula and training data together.

```python
import numpy as np
import polars as pl
import mboostrs as mb

rng = np.random.default_rng(7)
x = rng.uniform(-2, 2, 400)
data = pl.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.3, x.size)})
search, test = mb.holdout(data, fraction=0.2, seed=7)
formula = "y ~ bbs(x, df=4)"
family = mb.Gaussian()

tuned = mb.tune_mstop(
    formula, search, family=family, folds=5, seed=7,
    start=100, max_mstop=1600,
)
print("mstop:", tuned.mstop, "at_boundary:", tuned.at_boundary)
if tuned.at_boundary:
    raise RuntimeError("Stopping search hit its budget; extend max_mstop before evaluation")
model = tuned.model  # already fitted on search rows at the selected mstop

# Use the test response only after the specification and tuning are fixed.
prediction = model.predict(newdata=test, type="response")
y_test = test["y"].to_numpy()
assert prediction.shape == y_test.shape
assert np.isfinite(prediction).all()
mse = float(np.mean((y_test - prediction) ** 2))
baseline = float(np.mean((y_test - search["y"].mean()) ** 2))
print({"test_mse": mse, "intercept_only_test_mse": baseline})
assert mse < baseline  # a sanity check for this synthetic signal, not all data
```

`tuned` is an `MstopTuning`, not a fitted model. Its `.model` is the fitted
model at `.mstop`; `.risk` is the CV risk path and `.at_boundary` says whether
the search observed the minimum turn. Reusing `.model` avoids an extra refit.
If you fit directly with `mb.gamboost`, use
`control=mb.boost_control(mstop=chosen_count)`; its default count is not CV tuning.

`mb.fit(data, "y")` is also available, with `mb.report(result)` and
`result.model`. It currently delegates to `autospec`; its budget labels do
not change compute, and time handling and selection-aware uncertainty options
are not implemented. The explicit path above exposes the folds, stopping
search, and evaluation rather than implying those optional stages ran.

## Adapt the example before running on real data

- **Binary response:** encode 0/1, pass `family=mb.Binomial()`, and predict
  with `type="response"` for probabilities. Choose a binary evaluation score;
  a Gaussian MSE baseline does not specify a classification analysis.
- **New groups:** use `mb.holdout(data, groups="site")`, then create
  `mb.grouped_folds(search["site"].to_numpy(), B=5)` for `folds=`. Decide
  how to predict levels absent from training; set `unseen_levels=` consistently
  in tuning and prediction. To predict new rows in known groups, read
  `mb.known_group_folds` instead. Group membership itself does not dictate the
  deployment target.
- **Future observations:** use a time holdout and `mb.rolling_folds` within
  the search data. Random or ordinary blocked folds do not measure forecasting.
- **Missing data:** the default refuses missing formula values. Read
  "missing-data" before choosing `na_action="omit"` or `"impute"`.
- **Effect intervals without boosting selection:** start with `mb.gam` and
  "direct-reml-gam". For intervals after selecting terms, read
  "uncertainty-selection". For treatment effects, read "causal".

## Follow up when the task needs it

Open "fit-and-tune" only if the stopping search needs attention or a different
fold/score contract. Open "diagnostics" for residual patterns, and
"modeling-tactics" for a weak validation score or a `reference_gap` worth
investigating. Open "uncertainty" before interpreting an interval or a test.
Use "task-map-interpret-export" to save and reload the model.

For model revision, work on search rows with shared validation folds;
`mb.term_report` and `mb.compare_specifications` can measure candidate gains.
Do not repeatedly inspect the final test response to choose revisions.
No model is required to beat a flexible reference or select a nonempty set of
terms: a null result can be the right result. Stop at the requested scope and
state unresolved diagnostics or budget limits.
