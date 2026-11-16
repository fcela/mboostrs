# `AUC`

Defined in `mboostrs.families`; exported as `mboostrs.AUC`.


*Class*

```python
AUC()
```

Smoothed 1 - AUC ranking loss (mboost::AUC()).

Response must be a two-level (binary) column; any two distinct raw
values are accepted -- the Rust core recodes the larger to +1 and the
smaller to -1, mirroring R's factor level ordering.

Source: [`python/mboostrs/families.py` lines 340-348](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L340-L348)
