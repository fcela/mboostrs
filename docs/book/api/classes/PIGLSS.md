# `PIGLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.PIGLSS`.


*Class*

```python
PIGLSS(stabilization: str = 'none')
```

Poisson-inverse-Gaussian family: `mu` (log link, mean), `sigma`
(log link, dispersion). Matches R `gamboostLSS::PIGLSS()` (wrapping
gamlss.dist `PIG`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 2362-2382](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2362-L2382)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `mean` | Method | Conditional count mean ``mu``. |

(piglss-response)=
## `PIGLSS.response`

*Method* of `PIGLSS`

```python
PIGLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2374-2378](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2374-L2378)

(piglss-mean)=
## `PIGLSS.mean`

*Method* of `PIGLSS`

```python
PIGLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 2380-2382](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2380-L2382)
