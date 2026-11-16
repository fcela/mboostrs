# scikit-learn interface

`mboostrs.sklearn` provides estimators that follow the scikit-learn protocol
(`fit(X, y)` / `predict(X)`, `get_params` / `set_params`), so the models work
inside `Pipeline`, `cross_val_score`, and `GridSearchCV`. It requires
scikit-learn: `pip install mboostrs[sklearn]`.

```python
from mboostrs.sklearn import GLMBoostRegressor, GAMBoostRegressor
from mboostrs.sklearn import GLMBoostClassifier, GAMBoostClassifier
```

| Estimator | Engine | Default per-column learner | Family |
|-----------|--------|----------------------------|--------|
| `GLMBoostRegressor` | `glmboost` | `bols` (linear) | gaussian / poisson / gamma / laplace / huber |
| `GAMBoostRegressor` | `gamboost` | `bbs(df=...)` (P-spline) | same |
| `GLMBoostClassifier` | `glmboost` | `bols` | Binomial |
| `GAMBoostClassifier` | `gamboost` | `bbs(df=...)` | Binomial |

Common parameters: `mstop` (iterations), `nu` (learning rate); regressors take
`family`; the `GAM*` estimators take `df` and `formula`. All are constructor
parameters, so `GridSearchCV` can tune them.

## Reconciling the formula and matrix interfaces

The native `mboostrs` API is formula-based (`"y ~ bbs(x1) + bols(x2)"` over named
data). scikit-learn is matrix-based (`fit(X, y)` over an unnamed array). The
estimators bridge the two in two modes.

### 1. Default: one learner per column

Without a formula, each column of `X` becomes one base learner of the
estimator's type. The estimator class encodes the choice the formula would
otherwise carry — `GLM*` use `bols` (linear), `GAM*` use `bbs` (smooth):

```python
GAMBoostRegressor(df=4).fit(X, y)     # y ~ bbs(f0,df=4) + bbs(f1,df=4) + ...
```

This matrix interface works with scikit-learn estimators and model-selection
tools. All columns use the same learner; use an explicit formula when terms
need different learners.

### 2. Explicit formula (constructor parameter)

The `GAM*` estimators accept a full `mboostrs` formula for per-term control
(mixed `bols`/`bbs`/`btree`/..., per-feature `df`, interactions, `by=`):

```python
GAMBoostRegressor(formula="y ~ bbs(age, df=5) + bols(income) + btree(z)")
```

- The formula references column names. Pass a pandas/polars **DataFrame** and
  its real column names are used (and exposed as `feature_names_in_`); pass a
  bare array and columns are named `f0, f1, ...`.
- The left-hand side of `~` is a placeholder — the response is supplied by `y`
  in `fit`.

### Formula configuration

In scikit-learn, *all configuration lives in constructor parameters*.
`get_params`/`set_params` read them, `clone()` reconstructs an estimator from
them, and `GridSearchCV` tunes them. A value passed to `fit(X, y, formula=...)`
is a *fit parameter*: it is not part of `get_params`, so `clone()` drops it —
and because `cross_val_score`/`GridSearchCV` clone the estimator per fold, the
formula would silently vanish during cross-validation. Fit parameters are meant
for per-call *data* (`sample_weight`, `groups`), not model configuration.

Putting the formula in the constructor also makes it tunable:

```python
GridSearchCV(GAMBoostRegressor(), {"formula": [f1, f2], "mstop": [100, 300]})
```

## Classification

`GLMBoostClassifier` / `GAMBoostClassifier` handle binary targets. `fit`
records `classes_`, encodes the positive class as 1, and:

```python
clf = GAMBoostClassifier(df=4, mstop=200).fit(X, y)
clf.predict_proba(X)   # (n, 2): columns are P(class0), P(class1)
clf.predict(X)         # original class labels
clf.score(X, y)        # accuracy (from ClassifierMixin)
```

## Example: pipeline + grid search

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from mboostrs.sklearn import GLMBoostRegressor

pipe = make_pipeline(StandardScaler(), GLMBoostRegressor(nu=0.1))
gs = GridSearchCV(pipe, {"glmboostregressor__mstop": [50, 100, 200]}, cv=5)
gs.fit(X, y)
```
