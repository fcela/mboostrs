# `evaluate_distribution`

Defined in `mboostrs.distribution_evaluation`; exported as `mboostrs.evaluate_distribution`.


*Function*

```python
evaluate_distribution(
    model,
    data,
    response=None,
    data_role,
    weights=None,
    levels=(0.5, 0.8, 0.95),
    thresholds=(),
    random_state=0,
    mstop=None,
    unseen_levels='error'
)
```

Evaluate a fitted scalar-response LSS distribution on supplied data.

`data_role` must be `"external"` or `"training"`. External means
the outcomes were not used for fitting, tuning, specification search, or
preprocessing choices; this is a caller declaration retained in the result,
not something an already-fitted object can prove. Training results diagnose
fit but do not estimate future predictive performance.

The family must implement density/mass, CDF, quantile, and deviance
increments. PIT randomizes uniformly between `F(y-)` and `F(y)`, so
discrete and mixed atoms are handled without subtracting an assumed unit.
Central interval metrics use the proper interval score. Strict exceedance
probabilities use `1-F(threshold)`. CRPS, the integral of the squared
gap between the predictive cdf and the step at the outcome, is reported
for every family with a cdf and a quantile: GaussianLSS by its closed
form, count families (Poisson, negative binomial, geometric, binomial,
zero-inflated and zero-adjusted Poisson) as a sum over integer thresholds,
and every other family, including the zero-adjusted gamma, by Simpson
quadrature. CRPS is withheld for transformed responses, because it is not
invariant to the transform; the result's assumptions name the reason.

`unseen_levels` is the point policy of `model.predict`: how a row
whose `brandom` level the fit never saw is scored — `"average"` as a
member of an average group, `"zero_effect"` with a zero coefficient,
or `"error"` (the default) refused. A non-refusing policy is recorded
in the result's assumptions, because the scores then condition on a point
prediction for a group the fit never saw and do not integrate new-group
uncertainty.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/distribution_evaluation.py` lines 246-446](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/distribution_evaluation.py#L246-L446)
