# `evaluate_specification`

Defined in `mboostrs.explore`; exported as `mboostrs.evaluate_specification`.


*Function*

```python
evaluate_specification(rhs, data, response, **kwargs)
```

Evaluate one specification and retain its held-out evidence.

Arguments are those of `cv_risk`. The returned
`mboostrs.evaluation.EvaluationResult` records the exact score,
prediction scale, equal-fold aggregation, row identifiers, exclusions,
fold membership, weights and a data fingerprint. Set
`retain_arrays=False` to omit row memberships and predictions.

Source: [`python/mboostrs/explore.py` lines 1174-1184](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1174-L1184)
