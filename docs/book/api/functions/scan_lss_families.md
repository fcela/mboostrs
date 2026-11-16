# `scan_lss_families`

Defined in `mboostrs.model_selection`; exported as `mboostrs.scan_lss_families`.


*Function*

```python
scan_lss_families(
    formula,
    data,
    families: list | None = None,
    control=None,
    metric: str = 'GAIC',
    fit_func='glmboostLSS',
    k: float = 2.0,
    support: str | None = None,
    folds=None,
    na_action: str = 'error'
) -> list[dict[str, Any]]
```

Scan candidate LSS families and rank by a metric.
Usage:

```python
import mboostrs as mb
rows = mb.scan_lss_families("y ~ bbs(x)", df, support="realplus")
rows[0]  # best candidate first, or the first error if all failed
```

The counterpart of `gamlss::chooseDist`: choosing the response
distribution is usually the least-informed decision in a distributional
model, and fitting the alternatives is a more direct way to make it than
reasoning about the response's shape.

**Parameters**

- `formula` (str or mapping): The model specification, shared by every candidate: a whole formula
  (`"y ~ bbs(x)"`), or a mapping with one whole formula per
  distribution parameter (`{"mu": "y ~ bbs(x)", "sigma": "y ~ 1"}`).
  Families differ in how many parameters they have, so a mapping is
  matched by position and any parameter beyond its length reuses the
  last entry. A bare right-hand side is refused before anything is fitted.
- `data` (dict or DataFrame)
- `families` (list, optional): LSS family instances or names to try (e.g. `[GaussianLSS(), "GammaLSS"]`).
  Defaults to the candidate set for `support`.
- `support` (str, optional): The response support whose candidate set is used when `families` is
  not given: `"realline"`, `"realplus"`, `"real0to1"`, `"counts"`
  or `"zeroplus"`. The sets are `FAMILY_SETS` in
  `mboostrs.model_selection` (also exported as `mb.FAMILY_SETS`);
  `mb.family_set(support)` returns the family names for one support.
- `folds` (int or array, optional): Fold weights for `metric="cvrisk"`: an `(n, B)` weight matrix, or a
  number of folds for a seeded k-fold split (`cv_weights`). Defaults to
  5-fold.
- `control` (boost_control tuple, optional): Defaults to `boost_control(100, 0.1)`.
- `metric` ({"cvrisk", "GAIC", "risk", "logLik"}): What to rank by. Lower is better for every one but `logLik`.

  `"cvrisk"` is out-of-bag risk from cross-validation, and is the
  metric to prefer. `"GAIC"` matches `chooseDist` but see the warning
  below.
- `fit_func` ({"glmboostLSS", "gamboostLSS"}): Which LSS fit function to use.
- `k` (float): Penalty for GAIC.
- `na_action` ({"error", "omit"}): `"error"` (the default) refuses a missing value in any column a
  formula reads. `"omit"` fixes one sample before any candidate is
  fitted: the rows complete in every column that any entry of
  `formula` reads. A one-parameter family reads only the first entry
  of a mapping and would keep rows a three-parameter family drops; every
  family is fitted and scored on the common rows instead, so no
  candidate scores better by having dropped rows. `folds` given as a
  matrix has one row per input row and is restricted to the kept rows.
  A `UserWarning` names the number of rows removed, and each row
  reports `common sample rows` and `common sample excluded rows`
  (how many rows its own formulas would have kept that the common
  sample removed).

**Returns**

list of dict
    Sorted by metric (best first). Failed fits have `error` set and sort
    last, so a candidate that could not fit is recorded rather than
    silently missing. When every candidate fails, a `RuntimeWarning`
    names the first error: the first row is then not a winner. With `metric="cvrisk"` each row also carries
    `cvrisk_mstop`, the grid step where the out-of-bag minimum landed;
    it equals the budget when the risk curve was still falling at the cap,
    which makes the ranking provisional. When the fitted candidates differ in
    response transform, weight policy or reference measure (a count family's
    probability mass next to a continuous family's density), the rows keep
    candidate order, each fitted row carries a `comparison` reason, and
    `metric` does not order them. Candidates fitted to different rows or
    weights raise `ValueError`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

Any comparison across families rests on every family's risk being the exact
negative log-likelihood of the same data: one that dropped a constant term
would appear better or worse by that constant.
**Legacy GAIC score.** Its penalty counts boosting
updates plus one; it is not effective df and is no longer reported as
likelihood df. Under cyclic fitting every parameter is
updated once per iteration, so that count is `mstop * n_params + 1`: it
measures the *boosting budget*, not how much structure the model absorbed.
A two-parameter family at `mstop=60` scores df 121 against a one-parameter
family's 61 whatever either of them fitted, so GAIC ranks largely by
parameter count.

Source: [`python/mboostrs/model_selection.py` lines 649-888](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/model_selection.py#L649-L888)
