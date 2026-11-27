# `adaptive_conformal_stream`

Defined in `mboostrs.conformal`; exported as `mboostrs.adaptive_conformal_stream`.


*Function*

```python
adaptive_conformal_stream(
    predicted,
    observed,
    half_width,
    level: float = 0.9,
    lr: float = 0.02
) -> AdaptiveConformalStream
```

Online Adaptive Conformal Inference (Gibbs & Candès, 2021).

Sequentially updates the effective miscoverage target `alpha_t` after each
observation `y_t` arrives and scales the next step's baseline `half_width`
by the Gaussian tail ratio `z(alpha_t) / z(1 - level)` *before* observing
`y_t`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/conformal.py` lines 1832-1908](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L1832-L1908)
