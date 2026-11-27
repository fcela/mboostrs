# `AdaptiveConformalStream`

Defined in `mboostrs.conformal`; exported as `mboostrs.AdaptiveConformalStream`.


*Class*

```python
AdaptiveConformalStream(
    lower: numpy.ndarray,
    upper: numpy.ndarray,
    coverage: float,
    alpha_path: numpy.ndarray,
    multipliers: numpy.ndarray,
    level: float,
    lr: float
) -> None
```

Online Adaptive Conformal Inference stream (Gibbs & Candès, 2021).

Source: [`python/mboostrs/conformal.py` lines 1811-1829](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L1811-L1829)
