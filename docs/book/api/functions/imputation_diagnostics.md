# `imputation_diagnostics`

Defined in `mboostrs.transforms`; exported as `mboostrs.imputation_diagnostics`.


*Function*

```python
imputation_diagnostics(model) -> dict
```

Report the R-squared of each model-based predictor fill, by column.

Works on any fit that takes `na_action="impute_model"`. An empty dict
means the model fills no column with a model. See `imputation_r2`.

Source: [`python/mboostrs/transforms.py` lines 1184-1190](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/transforms.py#L1184-L1190)
