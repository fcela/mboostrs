# `FoldScheme`

Defined in `mboostrs.folds`; exported as `mboostrs.FoldScheme`.


*Class*

```python
FoldScheme(
    weights: numpy.ndarray,
    held_out: numpy.ndarray,
    scheme: str,
    seed: int | None = None,
    groups: numpy.ndarray | None = None,
    strata: numpy.ndarray | None = None,
    ordered: bool = False,
    judge: numpy.ndarray | None = None,
    _note: str = '',
    base_weights: numpy.ndarray | None = None,
    repeats: int = 1,
    known_groups: numpy.ndarray | None = None,
    times: numpy.ndarray | None = None,
    embargo: float | numpy.timedelta64 | None = None
) -> None
```

An `(n, B)` in-bag weight matrix, and what it means.

Passing one where a matrix is expected works: `np.asarray(scheme)`
returns the weights, so `model.cvrisk(folds=scheme)` needs no
unwrapping.

Build a scheme with a constructor such as `kfold_folds` or
`grouped_folds` rather than by hand. Every array is copied and made
read-only at construction, and construction raises `ValueError` when a
shape disagrees with `weights`, a fold trains on a row it also holds out,
a fold trains on or scores no rows, or `repeats` does not divide the
number of folds.

**Attributes**

- `weights` (ndarray, shape (n, B)): In-bag weights: the model's weights on the rows a fold trains on, zero
  on the rows it holds out.
- `held_out` (ndarray of bool, shape (n, B)): Whether the row is held out in that fold. A row whose model weight is
  zero is out of the sample and is held out nowhere.
- `scheme` (str): The scheme's name, for example `"stratified 10-fold"`.
- `seed` (int or None): The seed, when the scheme draws one.
- `groups` (ndarray or None): The grouping the scheme kept whole, when it was built around one.
- `strata` (ndarray or None): The strata whose proportions the scheme preserved.
- `ordered` (bool): Whether the scheme orders folds in time, so that the check for
  training on the future applies.
- `judge` (ndarray of bool, shape (n, B), or None): The rows whose out-of-fold risk is scored, when that is not every
  zero-weight row (rolling-origin folds). It must be a subset of
  `held_out`.
- `base_weights` (ndarray, shape (n,), or None): Observation weights before folds were assigned, used to weight
  evaluation losses. `None` means unit weights.
- `repeats` (int): Number of stacked partitions; `B // repeats` folds each.
- `known_groups` (ndarray or None): Grouping spread across folds on purpose (`known_group_folds`).
- `times` (ndarray or None): Timestamps the order of a `rolling_folds` scheme was taken from.
- `embargo` (float, numpy.timedelta64 or None): Minimum gap, in the units of `times`, between the last training time
  and the first judged time of a fold.

Source: [`python/mboostrs/folds.py` lines 79-546](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L79-L546)

## Members

| Member | Kind | Summary |
|---|---|---|
| `shape` | Property |  |
| `n_folds` | Property |  |
| `to_dict` | Method | A JSON-safe record with the weight evidence only on request. |
| `fold` | Method | The row indices fold ``b`` holds out. |
| `fold_sizes` | Method | How many rows each fold holds out. |
| `take` | Method | The same scheme restricted to `rows` (positions or boolean mask). |
| `check_leakage` | Method | Report rows that a fold both trains on and is judged by. |
| `describe` | Method | A short text summary: scheme name, fold count, held-out sizes. |
| `to_frame` | Method | One row per (row, fold): whether the row is in-bag and its weight. |

(foldscheme-shape)=
## `FoldScheme.shape`

*Property* of `FoldScheme`

```python
FoldScheme.shape  # -> tuple[int, int]
```

*No description in the docstring.*

Source: [`python/mboostrs/folds.py` lines 265-267](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L265-L267)

(foldscheme-n_folds)=
## `FoldScheme.n_folds`

*Property* of `FoldScheme`

```python
FoldScheme.n_folds  # -> int
```

*No description in the docstring.*

Source: [`python/mboostrs/folds.py` lines 269-271](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L269-L271)

(foldscheme-to_dict)=
## `FoldScheme.to_dict`

*Method* of `FoldScheme`

```python
FoldScheme.to_dict(*, include_arrays=False)
```

A JSON-safe record with the weight evidence only on request.

**Parameters**

- `include_arrays` (bool, default False): Include `weights`, `held_out`, `groups`, `strata`,
  `known_groups` and `times` as nested lists. Without it the
  record holds scalars plus `n_folds` and `n_rows`.

**Returns**

dict
    JSON-serialisable record.

Source: [`python/mboostrs/folds.py` lines 281-301](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L281-L301)

(foldscheme-fold)=
## `FoldScheme.fold`

*Method* of `FoldScheme`

```python
FoldScheme.fold(b: int) -> ndarray
```

The row indices fold `b` holds out.

**Parameters**

- `b` (int): Zero-based fold (column) index.

**Returns**

ndarray of int
    Sorted row positions with `held_out[:, b]` true.

Source: [`python/mboostrs/folds.py` lines 303-316](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L303-L316)

(foldscheme-fold_sizes)=
## `FoldScheme.fold_sizes`

*Method* of `FoldScheme`

```python
FoldScheme.fold_sizes() -> ndarray
```

How many rows each fold holds out.

**Returns**

ndarray of int, shape (B,)
    Count of held-out rows per fold.

Source: [`python/mboostrs/folds.py` lines 318-326](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L318-L326)

(foldscheme-take)=
## `FoldScheme.take`

*Method* of `FoldScheme`

```python
FoldScheme.take(rows)
```

The same scheme restricted to `rows` (positions or boolean mask).

A model fitted with `na_action='omit'` estimates on the complete
cases, so a scheme drawn over the full frame must drop the same rows
before its weights can drive that model's cross-validation. Every
per-row field -- weights, held-out masks, the judge set, group and
stratum labels, base weights -- is subset together, so fold
membership and judge semantics are preserved exactly.

**Parameters**

- `rows` (array_like of int or bool): Row positions to keep, or a boolean mask of length `n`.

**Returns**

FoldScheme
    A new scheme over the selected rows, with the same fold count.

**Raises**

- `ValueError`: If the restricted scheme leaves a fold with no training or no
scored rows (raised by the constructor).

Source: [`python/mboostrs/folds.py` lines 328-377](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L328-L377)

(foldscheme-check_leakage)=
## `FoldScheme.check_leakage`

*Method* of `FoldScheme`

```python
FoldScheme.check_leakage(groups=None, *, raise_on_leak: bool = True) -> dict
```

Report rows that a fold both trains on and is judged by.

With no argument this checks the grouping the scheme was built around
-- for a grouped scheme, that no cluster straddles a split; for a time
scheme, that no fold trains on rows later than the ones it holds out.

Pass `groups` to ask about a grouping the scheme knows nothing
about, which is the useful case: a plain k-fold split of repeated
measurements leaks by subject, and the matrix cannot say so.

**Parameters**

- `groups` (array_like, shape (n,), optional): Group label per row. Defaults to the grouping the scheme was built
  around (`self.groups`).
- `raise_on_leak` (bool, default True): Raise when a leak is found. With `False` the report is returned
  instead.

**Returns**

dict
    `{"scheme": str, "leaks": list of dict, "clean": bool}`. Each
    leak has `fold`, `kind` (`"group"`, `"time"` or
    `"embargo"`) and the groups or row positions involved.

**Raises**

- `ValueError`: If `groups` does not have one entry per row, or if a leak is
found and `raise_on_leak` is true.

Source: [`python/mboostrs/folds.py` lines 379-479](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L379-L479)

(foldscheme-describe)=
## `FoldScheme.describe`

*Method* of `FoldScheme`

```python
FoldScheme.describe() -> str
```

A short text summary: scheme name, fold count, held-out sizes.

**Returns**

str
    Multi-line description including the seed and the number of
    groups kept whole when the scheme has them.

Source: [`python/mboostrs/folds.py` lines 481-517](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L481-L517)

(foldscheme-to_frame)=
## `FoldScheme.to_frame`

*Method* of `FoldScheme`

```python
FoldScheme.to_frame(output='auto')
```

One row per (row, fold): whether the row is in-bag and its weight.

**Parameters**

- `output` ({"auto", "pandas", "polars", "records"}, default "auto"): Container for the result; `"auto"` is Polars when installed,
  otherwise pandas, otherwise records.

**Returns**

frame or list of dict
    Columns `row`, `fold` (zero-based), `weight` and `held_out`;
    `n * B` rows.

Source: [`python/mboostrs/folds.py` lines 519-546](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L519-L546)
