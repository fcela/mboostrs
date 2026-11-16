# `confounding_benchmark`

Defined in `mboostrs.causal`; exported as `mboostrs.confounding_benchmark`.


*Function*

```python
confounding_benchmark(data, result=None, *, variables=None, folds=None, output='auto')
```

Benchmark sensitivity by omitting one measured covariate at a time.

Each row reruns the same AIPW configuration after deleting one declared
baseline covariate. The observed estimate shift gives a scale for choosing
sensitivity scenarios: for example, an assumed unmeasured bias twice the
largest measured-covariate shift. It is not a bound. Omission can change
overlap and the entire nuisance fit, measured covariates need not resemble
an unmeasured one, and correlated measured covariates can mask each other's
contribution.

The canonical call is `confounding_benchmark(data, result)`: the data
frame first, then the `AIPWResult`. The reversed order,
`confounding_benchmark(result, data)`, is also accepted; the two are told
apart by type, so either order gives the same rows. (The companion
`confounding_sensitivity` takes the result first, because its
estimate can instead be given as plain numbers.)

Integer folds are reconstructed from the retained count and seed. If the
original estimate used an explicit grouped or ordered `FoldScheme`, pass
that same object as `folds` so its sampling-unit semantics are preserved.
Each refit uses the result's outcome model (the one `"auto"` resolved to),
its treatment interactions, and its `mstop` and `max_mstop`.

A result fitted with `propensity_terms` or `outcome_terms` is refitted
with the same formulas minus every term that names the omitted covariate;
the rows report the reduced formulas in `propensity_terms` and
`outcome_terms` (`None` where the model used its default formula). When
every term of a custom formula names the omitted covariate, the call refuses
before any refit, naming the custom formula, because a refit with no
formula is not a model.

`output` chooses the return type, as for the other frames of the
package: `"auto"` (the default) is a Polars frame when Polars is
installed and a pandas frame otherwise, `"polars"` and `"pandas"` force
one, and `"records"` returns a list of dicts holding plain Python
numbers, booleans and `None`, which `json.dumps` accepts. One row is
returned per omitted covariate. A Polars frame's `to_dict()` returns
Series; use `output="records"` for plain values.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `ValueError`

Source: [`python/mboostrs/causal.py` lines 5620-5742](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5620-L5742)
