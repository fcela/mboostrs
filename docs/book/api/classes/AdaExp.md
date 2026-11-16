# `AdaExp`

Defined in `mboostrs.families`; exported as `mboostrs.AdaExp`.


*Class*

```python
AdaExp()
```

Exponential loss, `exp(-(2y - 1) f)` — the AdaBoost loss.

Classification with a two-class response coded `{0, 1}`. It penalises
confident mistakes far more sharply than `Binomial` does, which
makes it more sensitive to mislabelled observations.

Source: [`python/mboostrs/families.py` lines 77-83](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L77-L83)
