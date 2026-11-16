# `Multinomial`

Defined in `mboostrs.families`; exported as `mboostrs.Multinomial`.


*Class*

```python
Multinomial(n_classes=None)
```

Multinomial logistic likelihood for an unordered K-class response.

Generalises `Binomial` to more than two classes using a
Kronecker-structured design: one linear predictor per class, sharing the
base learners. `predict(type="response")` returns an `(n, K)`
probability matrix whose rows sum to one.

Available on `mboostrs.gamboost` only. `mboostrs.glmboost`
rejects any response outside `{0, 1}` whatever `n_classes` says,
because the Kronecker-structured design this family needs is built only on
the `gamboost` path. Write `gamboost("y ~ bols(x)")` for a linear
multi-class fit.

**Parameters**

- `n_classes` (int, optional): Number of classes, at least 3. Inferred from the response when
  omitted, on the `gamboost` path.

Source: [`python/mboostrs/families.py` lines 312-337](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L312-L337)
