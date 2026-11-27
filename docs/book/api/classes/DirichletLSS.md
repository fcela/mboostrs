# `DirichletLSS`

Defined in `mboostrs.gamlss`; exported as `mboostrs.DirichletLSS`.


*Class*

```python
DirichletLSS(K: int, stabilization: str = 'none')
```

Dirichlet family for compositional responses: `K` concentration
parameters `alpha_k = exp(eta_k)` (log links). The response is a `K`-column
composition specified as `cbind(y1, ..., yK) ~ ...` (each row sums to 1).
Matches R `gamboostLSS::DirichletLSS(K)`. Requires `0 < y < 1`.

Source: [`python/mboostrs/gamlss.py` lines 2749-2795](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2749-L2795)

## Members

| Member | Kind | Summary |
|---|---|---|
| `response` | Method |  |
| `pdf` | Method | Dirichlet density for a K-column compositional response. |
| `simulate` | Method | Draw Dirichlet samples with concentration ``alpha_k``. |
| `deviance_increment` | Method |  |

(dirichletlss-response)=
## `DirichletLSS.response`

*Method* of `DirichletLSS`

```python
DirichletLSS.response(k: int, eta: numpy.ndarray) -> ndarray
```

*No description in the docstring.*

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `IndexError`

Source: [`python/mboostrs/gamlss.py` lines 2767-2770](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2767-L2770)

(dirichletlss-pdf)=
## `DirichletLSS.pdf`

*Method* of `DirichletLSS`

```python
DirichletLSS.pdf(y, params) -> ndarray
```

Dirichlet density for a K-column compositional response.

Source: [`python/mboostrs/gamlss.py` lines 2772-2783](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2772-L2783)

(dirichletlss-simulate)=
## `DirichletLSS.simulate`

*Method* of `DirichletLSS`

```python
DirichletLSS.simulate(params, rng=None, size=None) -> ndarray
```

Draw Dirichlet samples with concentration `alpha_k`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/gamlss.py` lines 2785-2791](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2785-L2791)

(dirichletlss-deviance_increment)=
## `DirichletLSS.deviance_increment`

*Method* of `DirichletLSS`

```python
DirichletLSS.deviance_increment(y, params) -> ndarray
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 2793-2795](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L2793-L2795)
