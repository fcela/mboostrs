# `save_model`

Defined in `mboostrs.serialize`; exported as `mboostrs.save_model`.


*Function*

```python
save_model(model, path, *, scoring_only=False, indent: int | None = None) -> Path
```

Write `model` to `path` as a schema-versioned JSON artifact.
Usage:

```python
import mboostrs as mb
mb.save_model(fit, "fit.json")
mb.load_model("fit.json").predict(newdata)
```

Accepts `GLMBoostModel`, `GAMBoostModel`, and `LSSModel`. A distributional
artifact carries one learner block per distribution parameter and the
family's own state; it is readable by `load_model` here and by the R
reader for inspection.

A full artifact also stores the fit's call after preprocessing (the
rewritten formula, the training frame, the weights, the offset and the
fit arguments), so a loaded model can `continue_to`, `update`, `cvrisk`
and run the other refits. That record sits in a top-level
`refit_bundle` key, which the Rust and R readers do not read. When it
holds a value the format cannot encode (an object in `env`), the
artifact is written without it and the loaded model refuses refits.

`scoring_only=True` keeps what prediction needs and drops the rest of
the training state: the response column leaves every stored column set,
and the formula and control a refit would need are omitted, so a loaded
scoring-only model predicts (on new rows and on its stored fitted path)
and refuses `update`/`with_mstop` with the retention error rather than
silently carrying a full training checkpoint. Nothing about the fitted
model changes; the same fit saved both ways predicts identically.

A scoring-only artifact is not free of training rows. It keeps the
training predictor columns that `gamboost` and distributional models
rebuild their learners from, the observation weights, and the in-sample
fitted values, which are a smooth function of the response. Treat it as
containing row-level training data when deciding where it may be shared.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`

Source: [`python/mboostrs/serialize.py` lines 409-505](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/serialize.py#L409-L505)
