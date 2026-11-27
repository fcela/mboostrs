# `Hinge`

Defined in `mboostrs.learners`; exported as `mboostrs.Hinge`.


*Class*

```python
Hinge(name, *, df=4.0, n_candidates=20, lambda_value=None, center=True, by=None)
```

Adaptive hinge base-learner that estimates a piecewise-linear effect.

One boosting selection adds exactly one listable hinge term. At each
step the prepared learner scans `n_candidates` interior-quantile
knots of the raw predictor in both directions -- `max(0, x - t)`
(right-facing) and `max(0, t - x)` (left-facing) -- fits every
candidate by ridge-weighted least squares against the current
residual, and keeps the one with the smallest residual SSE.

The fitted knot marks a threshold, regime change or other abrupt change
in slope. The final model is additive, and each selected hinge reports
its knot, direction and coefficient.

Args:
    name: predictor name in the data frame.
    df: degrees of freedom the ridge on each selected hinge is
        calibrated to (default 4). The selected hinge has about two
        effective parameters (knot plus coefficient), so `df`
        governs shrinkage of the coefficient, not the knot choice;
        the knot is selected by SSE, not penalized. Calibrated
        through the same `df -> lambda` machinery as the other
        learners unless `lambda_value` is supplied.
    n_candidates: interior-quantile knots scanned per selection
        (default 20). The grid is built once from the raw column,
        before any weights, so cross-validation folds and replayed
        predictions see the same candidates the fit did.
    lambda_value: penalty weight used directly instead of calibrating
        from `df`. Takes precedence over `df` when supplied.
    center: weight-center each selected hinge column before fitting
        (default True), so the hinge carries no constant and the
        model offset owns the mean. The centering travels with the
        selection and is re-applied on new data.
    by: optional varying-coefficient variable name. Multiplies the
        hinge column by that column, fitting a kink whose height
        varies with it (the `bbs(by=)` convention).

Example:
    A planted kink at `x = 0.5`; the hinge learns the bend while a
    P-spline smooths across it:

```python
>>> import numpy as np, mboostrs as mb
>>> from mboostrs.learners import Hinge
>>> rng = np.random.default_rng(1)
>>> n = 600
>>> x = rng.uniform(-2, 2, n)
>>> y = np.maximum(0.0, x - 0.5) + 0.1 * rng.normal(size=n)
>>> data = {"x": x, "y": y}
>>> kw = dict(family=mb.Gaussian(), control=mb.boost_control(200, 0.1))
>>> hinge = mb.gamboost("y ~ hinge(x, df=4)", data, **kw)
>>> smooth = mb.gamboost("y ~ bbs(x, df=4)", data, **kw)
>>> grid = {"x": np.linspace(-2, 2, 9)}
>>> truth = np.maximum(0.0, grid["x"] - 0.5)
>>> rmse = lambda m: float(np.sqrt(np.mean(
...     (np.asarray(m.predict(grid)).ravel() - truth) ** 2)))
>>> rmse(hinge) < rmse(smooth)
True
```

Source: [`python/mboostrs/learners.py` lines 1198-1269](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L1198-L1269)
