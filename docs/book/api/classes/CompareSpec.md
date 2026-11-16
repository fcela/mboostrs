# `CompareSpec`

Defined in `mboostrs.compare`; exported as `mboostrs.CompareSpec`.


*Class*

```python
CompareSpec(
    model_type: str = 'glmboost',
    formula: str = 'y ~ x1 + x2',
    family: str = 'Gaussian',
    mstop: int = 50,
    nu: float = 0.1,
    n: int = 200,
    seed: int = 42,
    risk: str = 'inbag',
    weights: list[float] | None = None
)
```

Specification for a single model comparison.

**Parameters**

- `model_type` (str): One of "glmboost", "gamboost", "glmboostLSS", "gamboostLSS".
- `formula` (str): Model formula.
- `family` (str): Family name (e.g. "Gaussian", "Binomial", "GaussianLSS").
- `mstop` (int): Number of boosting iterations.
- `nu` (float): Step length.
- `n` (int): Number of observations for synthetic data.
- `seed` (int): Random seed for reproducible synthetic data.
- `risk` (str): Risk type ("inbag" or "oobag").
- `weights` (optional): Observation weights (if None, equal weights).

Source: [`python/mboostrs/compare.py` lines 39-107](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L39-L107)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#comparespec-to_dict) | Method | The specification as a JSON-safe mapping (it holds no arrays). |
| [`from_dict`](#comparespec-from_dict) | Method |  |
| [`to_json`](#comparespec-to_json) | Method |  |

(comparespec-to_dict)=
## `CompareSpec.to_dict`

*Method* of `CompareSpec`

```python
CompareSpec.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

The specification as a JSON-safe mapping (it holds no arrays).

Source: [`python/mboostrs/compare.py` lines 88-100](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L88-L100)

(comparespec-from_dict)=
## `CompareSpec.from_dict`

*Method* of `CompareSpec`

```python
CompareSpec.from_dict(d: dict[str, typing.Any]) -> CompareSpec
```

*No description in the docstring.*

Source: [`python/mboostrs/compare.py` lines 102-104](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L102-L104)

(comparespec-to_json)=
## `CompareSpec.to_json`

*Method* of `CompareSpec`

```python
CompareSpec.to_json() -> str
```

*No description in the docstring.*

Source: [`python/mboostrs/compare.py` lines 106-107](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/compare.py#L106-L107)
