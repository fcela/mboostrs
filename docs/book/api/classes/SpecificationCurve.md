# `SpecificationCurve`

Defined in `mboostrs.specification_curve`; exported as `mboostrs.SpecificationCurve`.


*Class*

```python
SpecificationCurve(
    source: str,
    response: str,
    score: str,
    score_parameter: object,
    mstop: int,
    nu: float,
    fold_scheme: str,
    partitions: int,
    folds_per_partition: int,
    selected: str,
    records: tuple,
    joint_test: dict,
    post_hoc: bool,
    label: str,
    fits: int,
    estimated_seconds: float,
    seconds: float,
    budget: int | None,
    excluded_rows: int,
    notes: tuple,
    output: str = 'auto'
) -> None
```

The candidate specifications of a search, scored on one fold partition.

`records` holds one dict per specification, sorted by out-of-fold score
(lowest first). Its keys are `specification` (a name), `role`
(`"selected"` or `"candidate"`), `family`, `rhs` (the right-hand
side; `"mu: ...; sigma: ..."` for a distributional family),
`transform` (the terms that read a transform, separated by `"; "`),
`family origin`, `shape origin`, `transform origin` and
`fold scheme origin` (each `"declared"`, `"raised"` or `"none"`),
`origin` (`"declared"` only when every choice is declared), ``fold
scheme`, `out-of-fold score`` (mean loss over the folds; lower is
better), `gap vs selected` (this score minus the selected one's), ``gap
se` (the paired fold standard error of that gap), `within two se``
(whether the gap is at most two of those standard errors; a gap below zero,
a candidate that scores better, is within by that rule) and `rank`
(position by score, from 1). The selected row has no standard error.

Source: [`python/mboostrs/specification_curve.py` lines 187-260](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/specification_curve.py#L187-L260)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`rows`](#specificationcurve-rows) | Property | The records as a frame, or a list of dicts for ``output="records"``. |
| [`to_dict`](#specificationcurve-to_dict) | Method | The whole result as plain Python types, for JSON. |
| [`plot`](#specificationcurve-plot) | Method | The curve as an Altair chart: each gap with its +/- 2 paired SE interval. |

(specificationcurve-rows)=
## `SpecificationCurve.rows`

*Property* of `SpecificationCurve`

```python
SpecificationCurve.rows
```

The records as a frame, or a list of dicts for `output="records"`.

`output="auto"` gives polars, `"polars"` a polars frame and
`"pandas"` a pandas frame.

Source: [`python/mboostrs/specification_curve.py` lines 229-236](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/specification_curve.py#L229-L236)

(specificationcurve-to_dict)=
## `SpecificationCurve.to_dict`

*Method* of `SpecificationCurve`

```python
SpecificationCurve.to_dict(*, include_arrays=False)
```

The whole result as plain Python types, for JSON.

`include_arrays` is accepted for the package's export convention. A
curve holds no arrays, so the payload is the same either way.

Source: [`python/mboostrs/specification_curve.py` lines 238-254](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/specification_curve.py#L238-L254)

(specificationcurve-plot)=
## `SpecificationCurve.plot`

*Method* of `SpecificationCurve`

```python
SpecificationCurve.plot()
```

The curve as an Altair chart: each gap with its +/- 2 paired SE interval.

Source: [`python/mboostrs/specification_curve.py` lines 256-260](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/specification_curve.py#L256-L260)
