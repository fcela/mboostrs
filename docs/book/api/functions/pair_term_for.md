# `pair_term_for`

Defined in `mboostrs.explore`; exported as `mboostrs.pair_term_for`.


*Function*

```python
pair_term_for(
    left,
    right,
    levels=None,
    groups=(),
    categorical=(),
    transforms=None,
    data=None
)
```

The learner that carries an interaction between two columns.

Two continuous columns get a tensor-product spline surface. A two-valued
column is an indicator, so a continuous partner gets a smooth that switches
on with it, `bbs(x, by=b)`, and the term fits how the effect of `x`
differs between the two values. A group column (a `brandom` term, or any
column in `groups`) gets a random slope, `brandom(g, by=x)`: one
shrunken slope on `x` per group. `bbs(x, by=g)` is not used for a
multi-level group: on a numeric group code, `by=` multiplies the smooth
by the code's value, which is an arbitrary label. (On a column of
categorical dtype, `by=` instead expands into one smooth per level; the
random slope is the term that means the same thing for both encodings.)
Two label columns with at most `FACTOR_TENSOR_MAX_LEVELS` levels each get
`bols(a, by=b)`: one coefficient per combination of their levels, fitted
as fixed effects (the cell design of the pair). A group column with more levels is a
`brandom` term instead; a group paired with a two-valued column of labels
(`categorical`) gets `brandom(g) %X% brandom(b)`, since a label column is
not a number a slope could multiply. `transforms` maps a column to the scale its main term
uses (`{"x": "log"}`); its smooth in the pair term uses the same scale,
so the pair does not extrapolate on a raw scale the main term avoided.

`levels`, `groups` and `categorical` describe the two columns. Columns
omitted from these arguments are assumed continuous. Pass the frame as
`data=` to infer level counts and identify string, enum, categorical
and boolean columns as labels. Columns with more than two levels are
also classified as groups. With `data=None`, the supplied descriptions
determine the term.

Source: [`python/mboostrs/explore.py` lines 2493-2580](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2493-L2580)
