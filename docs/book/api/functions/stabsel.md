# `stabsel`

Defined in `mboostrs.stabsel`; exported as `mboostrs.stabsel`.


*Function*

```python
stabsel(
    model,
    cutoff=None,
    q=None,
    PFER=None,
    B=None,
    folds=None,
    assumption='unimodal',
    sampling_type='SS',
    grid=None,
    verbose=True,
    eval=True,
    n_jobs=-1,
    fit_budget=None
)
```

Stability selection for mboostrs models.
Usage:

```python
import mboostrs as mb
model = mb.glmboost("y ~ x + z", data, control=mb.boost_control(50))
result = mb.stabsel(model, q=5, PFER=1)
result.selected_names
```

Specify two of `cutoff`, `q` and `PFER`. The third is computed
from the theoretical error bounds. For example,
`stabsel(model, cutoff=0.75, PFER=1)` specifies the cutoff and the
expected-false-learners bound.

`fit_budget` is the largest number of model fits the call may perform.
The count is `stabsel_fit_count` of the initial subsample count (`2 * B`
under `sampling_type="SS"`, `B` under `"MB"`), and the call refuses
before any subsample is drawn when the count is larger. `None` (the
default) does not check.

`PFER` bounds the expected *number* of falsely selected learners from
above. It is a count, not a probability and not a rate, and it bounds
*learners*: a menu such as `bols(x) + bbs(x)` gives one source variable
two candidates, and a PFER of 1 permits one false learner among them, not
one false variable.

The reported `phat` is the fraction of subsamples in which a learner was
selected -- an empirical frequency. It is not the probability that the
learner belongs in the model, and stability selection makes no such
statement.

`B` is the number of initial subsamples or complementary pairs used in
the error bound. Under `sampling_type="SS"`, each pair fits both halves,
for `2 * B` fits; `sampling_type="MB"` fits `B` subsamples. The
result reports the actual number of fits as `n_fits`.

`folds` is an `n x B` in-bag weight matrix of *initial* subsamples,
the same convention as `stabs::stabsel`. Under `"SS"` the complements
are constructed from the model's weights and appended, so supplying `B`
columns produces `2 * B` fits; pairing is never inferred from an even
column count.

`n_jobs` sets the worker threads for the per-fold refits (-1 = all CPUs,
1 = sequential). The refits release the GIL for both learner preparation
and boosting, so threads give real parallelism; the selection paths are
computed per fold and do not depend on `n_jobs`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/stabsel.py` lines 428-661](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L428-L661)
