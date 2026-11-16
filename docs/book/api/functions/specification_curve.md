# `specification_curve`

Defined in `mboostrs.specification_curve`; exported as `mboostrs.specification_curve`.


*Function*

```python
specification_curve(source, data, response, *, budget=None, seed=0, output='auto')
```

Score the specifications a search touched on one shared fold partition.

**Parameters**

- `source` (SearchSession, dict, SpecRecommendation or DiscoveryResult): The record to read the candidates from: a live
  `mboostrs.SearchSession`, the dict from its
  `export_transcript()`, the recommendation returned by
  `mboostrs.autospec`, or the result of `mboostrs.discover`.
  See the module notes for what each source contributes.
- `data` (polars or pandas DataFrame, or a mapping of column to values): The training rows the source was built on. A session or transcript
  refuses data whose fingerprint differs from the one it recorded.
- `response` (str): Name of the response column.
- `budget` (int, optional): Largest number of model fits the curve may perform. The count is
  taken from `mboostrs.estimate_cost` before any specification is
  scored, and a curve over the budget is refused with the count it would
  need. `None` means no limit.
- `seed` (int, default 0): Seed of the fold partition for a source that records none (an
  `autospec` recommendation), and of the sign-flip draws when they are
  sampled.
- `output` ({"auto", "polars", "pandas", "records"}, default "auto"): The type of `SpecificationCurve.rows`. `"auto"` gives a polars
  frame when polars is installed, which it always is.

**Returns**

SpecificationCurve
    One record per specification, the joint test, the post hoc label and
    the cost. `to_dict()` gives plain types and `plot()` an Altair chart.

**Raises**

- `ValueError`: When the data do not match the source, the source has no selected
specification, its record lacks what is needed to rebuild its folds, or
the curve needs more fits than `budget`.
- `TypeError`: When `source` is none of the four kinds above.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `RuntimeError`

**Notes**

The fold partition is shared by every candidate, so the paired gaps are
paired fold by fold. `gap se` is the standard error that
`mboostrs.compare_specifications` reports as `change se`, so the
same caution applies: it understates the variation across data sets.
Candidates are not tuned: a candidate scores at the source's stopping
iteration. Response transforms (`log` or `log1p` of the response) are
not in the curve, because they change the target that every candidate is
scored on; `mboostrs.SearchSession.run` with `response_probe`
measures them separately.

**Examples**

Scoring the candidates of an `autospec` recommendation:

>>> import numpy as np, polars as pl, mboostrs as mb
>>> rng = np.random.default_rng(0)
>>> x = rng.uniform(-2, 2, 200)
>>> data = pl.DataFrame({"x": x, "junk": rng.normal(size=200),
...                      "y": np.sin(x) + rng.normal(0, 0.3, 200)})
>>> rec = mb.autospec(data, "y", budget=150)
>>> curve = mb.specification_curve(rec, data, "y", budget=200)
>>> sum(row["role"] == "selected" for row in curve.records)
1

Source: [`python/mboostrs/specification_curve.py` lines 263-381](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/specification_curve.py#L263-L381)
