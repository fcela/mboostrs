# `expand_pairs`

Defined in `mboostrs.terms`; exported as `mboostrs.expand_pairs`.


*Function*

```python
expand_pairs(
    data,
    response=None,
    pairs=None,
    df=4.0,
    marginal_df=2.5,
    knots=5,
    kind='tensor',
    maxdepth=4
)
```

Interaction terms for `pairs`, or for every pair when none are given.

`pairs` is a sequence of column-name pairs. The cost of the default is
quadratic in the number of columns — a 21-column frame has 210 pairs, each
of which the selection step scores at every iteration — so pass the output
of `screen_interactions` rather than the default on anything wide.

Source: [`python/mboostrs/terms.py` lines 732-749](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L732-L749)
