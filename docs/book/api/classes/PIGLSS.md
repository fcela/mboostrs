# `PIGLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.PIGLSS`.


*Class*

```python
PIGLSS(stabilization: str = 'none')
```

Poisson-inverse-Gaussian family: `mu` (log link, mean), `sigma`
(log link, dispersion). Matches R `gamboostLSS::PIGLSS()` (wrapping
gamlss.dist `PIG`). Requires non-negative integer counts.

Source: [`python/mboostrs/gamlss.py` lines 2315-2335](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2315-L2335)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`response`](#piglss-response) | Method |  |
| [`mean`](#piglss-mean) | Method | Conditional count mean ``mu``. |

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

Source: [`python/mboostrs/gamlss.py` lines 2327-2331](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2327-L2331)

(piglss-mean)=
## `PIGLSS.mean`

*Method* of `PIGLSS`

```python
PIGLSS.mean(params) -> ndarray
```

Conditional count mean `mu`.

Source: [`python/mboostrs/gamlss.py` lines 2333-2335](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2333-L2335)
