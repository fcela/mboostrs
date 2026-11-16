# `load_model`

Defined in `mboostrs.serialize`; exported as `mboostrs.load_model`.


*Function*

```python
load_model(path)
```

Read a model artifact written by `save_model`.
Usage:

```python
import mboostrs as mb
fit = mb.load_model("fit.json")
fit.predict(newdata)
```

The inverse of `save_model`, restoring the recorded model class with
all of its state, including a distributional artifact's per-parameter
learners and any response transform.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/serialize.py` lines 519-569](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/serialize.py#L519-L569)
