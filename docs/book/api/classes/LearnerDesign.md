# `LearnerDesign`

Defined in `mboostrs.interactions`; exported as `mboostrs.LearnerDesign`.


*Class*

```python
LearnerDesign(term: str, data)
```

A fitted design/penalty recipe for `buser` and inspection.

`design` and `penalty` are the matrices built on the fitting data.
`transform` evaluates the same fitted recipe on new rows, preserving
spline boundary knots and factor levels. The object currently supports the
marginal learners accepted by `%X%`: `bbs`, `bols` and `brandom`.

Source: [`python/mboostrs/interactions.py` lines 93-113](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/interactions.py#L93-L113)

## Members

| Member | Kind | Summary |
|---|---|---|
| `transform` | Method | Evaluate the fitted design on ``data`` without refitting its basis. |

(learnerdesign-transform)=
## `LearnerDesign.transform`

*Method* of `LearnerDesign`

```python
LearnerDesign.transform(data, *, unseen: str = 'error') -> ndarray
```

Evaluate the fitted design on `data` without refitting its basis.

`unseen="zero"` gives rows with a factor level the fit never saw
all-zero indicators instead of raising.

Source: [`python/mboostrs/interactions.py` lines 107-113](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/interactions.py#L107-L113)
