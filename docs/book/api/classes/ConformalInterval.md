# `ConformalInterval`

Defined in `mboostrs.conformal`; exported as `mboostrs.ConformalInterval`.


*Class*

```python
ConformalInterval(
    lower: numpy.ndarray,
    upper: numpy.ndarray,
    predicted: numpy.ndarray,
    level: float,
    method: str,
    n_calibration: int,
    quantile: float | None = None,
    guaranteed_coverage: float | None = None,
    guarantee: str = '',
    assumptions: tuple[str, ...] = (),
    n_training: int | None = None,
    n_folds: int | None = None,
    mstop_source: str = '',
    calibration_source: str = '',
    n_set_aside: int = 0,
    fitted_mean: numpy.ndarray | None = None,
    score: str | None = None,
    requested_score: str | None = None,
    support: str = 'real',
    reference_distribution: str | None = None,
    zero_probability: float | None = None,
    reference_dispersion: float | None = None,
    randomization_seed: int | None = None,
    reliability: dict | None = None,
    _temporal_scale_ratio: float | None = None
) -> None
```

Prediction intervals with their calibration evidence attached.

Source: [`python/mboostrs/conformal.py` lines 81-249](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L81-L249)

## Members

| Member | Kind | Summary |
|---|---|---|
| `width` | Property |  |
| `count_size` | Property | Number of nonnegative integers covered, including zero for empty sets. |
| `to_dict` | Method | A plain-Python record; lower, upper and predicted arrays only on request. |
| `to_json` | Method | Strict JSON preserving infinities, NaN and genuine missing values. |
| `contains` | Method |  |
| `empirical_coverage` | Method |  |
| `to_frame` | Method | The interval as a table with one row per interval row. |

(conformalinterval-width)=
## `ConformalInterval.width`

*Property* of `ConformalInterval`

```python
ConformalInterval.width  # -> np.ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/conformal.py` lines 157-159](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L157-L159)

(conformalinterval-count_size)=
## `ConformalInterval.count_size`

*Property* of `ConformalInterval`

```python
ConformalInterval.count_size  # -> np.ndarray | None
```

Number of nonnegative integers covered, including zero for empty sets.

Source: [`python/mboostrs/conformal.py` lines 161-166](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L161-L166)

(conformalinterval-to_dict)=
## `ConformalInterval.to_dict`

*Method* of `ConformalInterval`

```python
ConformalInterval.to_dict(*, include_arrays=False)
```

A plain-Python record; lower, upper and predicted arrays only on request.

Nonfinite numerical values are preserved for backward compatibility;
use `to_json()` for strict JSON that distinguishes them from nulls. The
`"provenance"` entry carries the library and environment; it has no
data fingerprint because the interval holds no training rows.

Source: [`python/mboostrs/conformal.py` lines 168-184](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L168-L184)

(conformalinterval-to_json)=
## `ConformalInterval.to_json`

*Method* of `ConformalInterval`

```python
ConformalInterval.to_json(*, include_arrays=False, indent=None)
```

Strict JSON preserving infinities, NaN and genuine missing values.

Decode with `mboostrs.load_result_json`. The versioned envelope
records nonfinite categories separately from its null placeholders.
`include_arrays=True` includes row endpoints and predictions.

Source: [`python/mboostrs/conformal.py` lines 186-195](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L186-L195)

(conformalinterval-contains)=
## `ConformalInterval.contains`

*Method* of `ConformalInterval`

```python
ConformalInterval.contains(observed) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/conformal.py` lines 197-207](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L197-L207)

(conformalinterval-empirical_coverage)=
## `ConformalInterval.empirical_coverage`

*Method* of `ConformalInterval`

```python
ConformalInterval.empirical_coverage(observed) -> float
```

*No description in the docstring.*

Source: [`python/mboostrs/conformal.py` lines 209-210](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L209-L210)

(conformalinterval-to_frame)=
## `ConformalInterval.to_frame`

*Method* of `ConformalInterval`

```python
ConformalInterval.to_frame(output='auto')
```

The interval as a table with one row per interval row.

Columns: `predicted` (the interval centre), `lower`, `upper`,
`width` (`upper - lower`), `median`, `level` (the coverage
level, repeated on each row), and `fitted_mean` when the interval
carries it. `median` repeats `predicted` under the name that
`prediction_interval` gives its centre column. For
`method="distributional"` that centre is the fitted distribution's
median. For split and CV+ it is the point function used for calibration;
for CQR it is the midpoint of the calibrated band. Inverting a response
transform does not in general produce a conditional mean or median.

Source: [`python/mboostrs/conformal.py` lines 212-241](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L212-L241)
