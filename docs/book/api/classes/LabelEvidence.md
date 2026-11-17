# `LabelEvidence`

Defined in `mboostrs.terms`; exported as `mboostrs.LabelEvidence`.


*Class*

```python
LabelEvidence(
    column: str,
    levels: int,
    signal_ratio: float,
    signal_z: float,
    adjacency_ratio: float,
    labels: bool,
    order_p: float = nan
) -> None
```

Whether a whole-number column's values act as labels or as a measurement.

A column of 40 integer codes and a column of ages in whole years look the
same to a dtype rule. The response tells them apart without fitting a
model. Take the mean response at each value. For a grouping code the means
differ from level to level and neighbouring codes are no more alike than
any two codes; for a measurement with a smooth effect, neighbouring values
have similar means. Both statistics are corrected for the noise in a mean
of `n` rows, using the pooled within-level variance.

Source: [`python/mboostrs/terms.py` lines 124-178](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L124-L178)

## Members

| Member | Kind | Summary |
|---|---|---|
| `sentence` | Property |  |

(labelevidence-sentence)=
## `LabelEvidence.sentence`

*Property* of `LabelEvidence`

```python
LabelEvidence.sentence  # -> str
```

*No description in the docstring.*

Source: [`python/mboostrs/terms.py` lines 159-178](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L159-L178)
