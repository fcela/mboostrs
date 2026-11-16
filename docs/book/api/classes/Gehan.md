# `Gehan`

Defined in `mboostrs.families`; exported as `mboostrs.Gehan`.


*Class*

```python
Gehan()
```

Gehan rank loss for AFT survival models (mboost::Gehan()).

Response must be a `Surv(time, event)` pair (see `CoxPH`).
Only unit case weights are supported.

The Gehan loss depends only on pairwise residual differences, so a
constant offset cancels and the risk is flat in it -- the offset is
unidentifiable. mboostrs therefore defines `offset() == 0.0` rather
than reproducing R's arbitrary `optimize()`-on-a-flat-function value;
fitted values consequently differ from R's by an additive constant
(gradient, coefficient increments, `selected`, and risk all match R
exactly). See `crates/mboostrs-core/src/families.rs` (`Gehan`) for
the full derivation.

Source: [`python/mboostrs/families.py` lines 351-367](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L351-L367)
