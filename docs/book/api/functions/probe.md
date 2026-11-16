# `probe`

Defined in `mboostrs.selection`; exported as `mboostrs.probe`.


*Function*

```python
probe(formula, data=None, *, family=None, control=None, seed=0, folds=5, **fit_kwargs)
```

Select terms by the first entry of a permuted shadow, then refit them.

`formula` is a formula string with `data`, or a fitted `gamboost` /
`glmboost` model, whose formula, data, family and fitting arguments are
reused (`probe(model)`, as `deselect(model)` and `stabsel(model)`).

Every term that reads one column (`x`, `bbs(x, df=4)`, `bols(x)`)
gets a shadow: the same term over a permuted copy of the column. The
model with all terms and shadows is boosted up to `control.mstop`
(default 1000) iterations, and the terms selected before the first shadow
are kept. Terms over several columns (`%X%`, `bspatial`) get no shadow
and are kept only if selected before the first shadow.

`seed` seeds the generator that draws one row permutation per shadow
column. A fixed seed reproduces the same shadows and the same refit folds.

**Returns**

Probing
    `selected`, the iteration `stop` at which the first shadow entered,
    its name, the formula of the selected terms and `model`, that formula
    refitted without shadows with `mstop` by cross-validation.

Probing needs no tuning, but it has no error-rate guarantee: a shadow can
enter early by chance, which stops selection early, and a true term with
a weak effect can enter after a shadow.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/selection.py` lines 455-564](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L455-L564)
