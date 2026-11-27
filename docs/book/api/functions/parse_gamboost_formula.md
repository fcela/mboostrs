# `parse_gamboost_formula`

Defined in `mboostrs.formula`; exported as `mboostrs.parse_gamboost_formula`.


*Function*

```python
parse_gamboost_formula(
    formula: str,
    env: dict | None = None,
    data=None
) -> tuple[str | tuple[str, str], list[mboostrs.learners.Bbs | mboostrs.learners.Bkernel | mboostrs.learners.Bmono | mboostrs.learners.Bns | mboostrs.learners.Bols | mboostrs.learners.Brad | mboostrs.learners.Btree | mboostrs.learners.Bmrf | mboostrs.learners.Bspatial | mboostrs.learners.Brandom | mboostrs.learners.Bconstrained | mboostrs.learners.Buser]]
```

Parse an additive formula into its response and learner objects.

A column name that is not a plain identifier (it has spaces or operators,
starts with a digit, or is a reserved word such as `bbs` or `log`) is
written in backticks, as in R: ``"y ~ bbs(`house age`) + bols(`price/sqft`)"``.
Inside backticks a backslash escapes a backtick or a backslash, so a column
named ``a`b`` is written ```a\`b```. Every formula that mboostrs prints or
generates uses the same spelling, so it parses back to the same model.

**Parameters**

- `formula` (str): `"y ~ bbs(x, df=4) + bols(z)"`. A bare name becomes a `Bbs` with
  default arguments, as in R's `gamboost`.
- `env` (dict, optional): Names resolved inside learner arguments, for the arguments a formula
  cannot express literally — `bmrf(district, K=K_matrix)` needs
  `env={"K_matrix": K}`.

**Returns**

tuple
    `(response, learners)`, where `learners` holds the constructed
    learner objects in formula order. They can be passed straight to
    `mboostrs.gamboost.gamboost` as `learners=`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Examples**

```python
>>> response, learners = parse_gamboost_formula("y ~ bbs(a, df=4) + bols(b)")
>>> response, [type(l).__name__ for l in learners]
('y', ['Bbs', 'Bols'])
```

Source: [`python/mboostrs/formula.py` lines 1112-1290](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/formula.py#L1112-L1290)
