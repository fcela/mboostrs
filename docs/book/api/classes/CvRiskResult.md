# `CvRiskResult`

Defined in `mboostrs.cvrisk`; exported as `mboostrs.CvRiskResult`.


*Class*

```python
CvRiskResult(
    risk_matrix,
    aggregate,
    grid,
    optimal_mstop,
    fold_type,
    provenance=None,
    fold_fingerprint='',
    data_fingerprint='',
    folds=None,
    held_out_fraction=None,
    design=None,
    preprocessing='fixed',
    warn_boundary=True,
    comparison_identity=None
)
```

Cross-validated risk along the boosting path.

Returned by a model's `cvrisk()`. Slice the model to
`optimal_mstop` to get the tuned fit.

**Attributes**

- `risk_matrix` (ndarray, shape (n_folds, n_grid)): Out-of-fold risk for every fold at every grid point. Two models
  evaluated on the same folds can be compared using their paired
  per-fold differences.
- `aggregate` (ndarray, shape (n_grid,)): Mean risk across folds at each grid point.
- `grid` (ndarray): The iteration counts evaluated.
- `optimal_mstop` (int): The grid point minimising `aggregate`.
- `fold_type` (str): How the folds were produced — "10-fold kfold" for folds this call
  generated, the scheme's own name for a `FoldScheme`, and "10
  supplied folds" for a bare matrix, whose scheme this object cannot
  know.
- `provenance` (Provenance or None): The model, family, row count, budget and weight policy this curve was
  computed under. `result.describe()` prints it.
- `held_out_fraction` (float): The average share of the sample a fold holds out: `1/k` for k-fold,
  about 0.37 for a bootstrap draw. It scales the Nadeau-Bengio
  correction in `compare_to`, which is why it is measured rather
  than assumed.
- `data_fingerprint` (str): Digest of the response and weights the folds were fitted to
  (`data_fingerprint`); empty for a result built without one.
- `data_identity` (str): SHA-256 digest of the observation IDs, observed response and effective
  weights of the model the curve was computed for, read from its
  comparison identity; empty when the model recorded none.
  `compare_to` refuses two curves whose digests differ.
- `fold_fingerprint` (str): A digest of training fold weights, evaluation masks and model
  observation weights. Matching digests are necessary for pairing;
  callers must also use the same observations in the same order and
  comparable scoring targets. The digest does not identify the data.
- `folds` (FoldScheme or None): The scheme, when one was passed. A generated or bare matrix is not
  retained -- it is `n * B` floats and the caller already has it.
- `design` (str): The fold design this curve was computed under: the scheme's own name
  for a `FoldScheme`, "N supplied folds" for a bare partition matrix,
  and "N bootstrap folds" when the matrix carries the multiplicity
  weights a bootstrap draw leaves behind. `compare_to` refuses an
  interval for the designs the Nadeau-Bengio correction does not cover.
- `preprocessing` ({"fixed", "refit"}): Whether the full-data recipe was reused or preprocessing was learned
  within each training fold.

Source: [`python/mboostrs/cvrisk.py` lines 97-415](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L97-L415)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`data_identity`](#cvriskresult-data_identity) | Property | SHA-256 digest of the observation IDs, response and weights of the |
| [`to_dict`](#cvriskresult-to_dict) | Method | A JSON-safe record; the raw fold-risk matrix only on request. |
| [`standard_error`](#cvriskresult-standard_error) | Property | Standard error of the mean risk at each grid point. |
| [`mstop_1se`](#cvriskresult-mstop_1se) | Property | The one-standard-error rule: the smallest `mstop` whose risk is |
| [`at_boundary`](#cvriskresult-at_boundary) | Method | Whether the curve is still falling at the last grid point. |
| [`report`](#cvriskresult-report) | Method | Fold design, grid, optimum, held-out score and boundary flag as a |
| [`compare_to`](#cvriskresult-compare_to) | Method | Compare two models fold by fold, on the folds they share. |
| [`plot`](#cvriskresult-plot) | Method | Plot cross-validated risk. |
| [`to_frame`](#cvriskresult-to_frame) | Method | Long-format table: one row per (fold, mstop) with its risk. |

(cvriskresult-data_identity)=
## `CvRiskResult.data_identity`

*Property* of `CvRiskResult`

```python
CvRiskResult.data_identity  # -> str
```

SHA-256 digest of the observation IDs, response and weights of the
model this curve was computed for (`_comparison_identity.data_identity`),
or `""` when the model recorded none.

Source: [`python/mboostrs/cvrisk.py` lines 221-228](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L221-L228)

(cvriskresult-to_dict)=
## `CvRiskResult.to_dict`

*Method* of `CvRiskResult`

```python
CvRiskResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the raw fold-risk matrix only on request.

**Parameters**

- `include_arrays` (bool, default False): Include `risk_matrix`, the `(n_folds, n_grid)` out-of-fold
  risks. The aggregate, grid and standard error are always included.

**Returns**

dict
    JSON-serialisable record of the optimum, aggregate curve, fold
    design and fingerprints.

Source: [`python/mboostrs/cvrisk.py` lines 230-264](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L230-L264)

(cvriskresult-standard_error)=
## `CvRiskResult.standard_error`

*Property* of `CvRiskResult`

```python
CvRiskResult.standard_error  # -> np.ndarray
```

Standard error of the mean risk at each grid point.

The across-fold standard deviation divided by the square root of the
fold count. The estimate describes variation across this set of folds;
the folds share training rows, so it is not a confidence bound for
variation across datasets. `compare_to` applies the
Nadeau-Bengio correction when comparing two models under its supported
fold assumptions.

Source: [`python/mboostrs/cvrisk.py` lines 268-282](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L268-L282)

(cvriskresult-mstop_1se)=
## `CvRiskResult.mstop_1se`

*Property* of `CvRiskResult`

```python
CvRiskResult.mstop_1se  # -> int
```

The one-standard-error rule: the smallest `mstop` whose risk is
within one standard error of the minimum.

The rule adds the standard error at the minimum to the minimum risk,
then returns the smallest positive grid point whose risk is no greater
than that threshold. For boosting, the smallest grid point is the
earliest stopping iteration within the rule's threshold.

Source: [`python/mboostrs/cvrisk.py` lines 284-299](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L284-L299)

(cvriskresult-at_boundary)=
## `CvRiskResult.at_boundary`

*Method* of `CvRiskResult`

```python
CvRiskResult.at_boundary() -> bool
```

Whether the curve is still falling at the last grid point.

The minimum lies at the last evaluated grid point. The selected step
count may therefore depend on the grid limit; extend the path to check
whether risk declines further.

Source: [`python/mboostrs/cvrisk.py` lines 301-313](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L301-L313)

(cvriskresult-report)=
## `CvRiskResult.report`

*Method* of `CvRiskResult`

```python
CvRiskResult.report() -> CvReport
```

Fold design, grid, optimum, held-out score and boundary flag as a
`CvReport`, the shape an LSS result reports too.

Source: [`python/mboostrs/cvrisk.py` lines 315-328](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L315-L328)

(cvriskresult-compare_to)=
## `CvRiskResult.compare_to`

*Method* of `CvRiskResult`

```python
CvRiskResult.compare_to(other, *, mstop=None, level=0.95)
```

Compare two models fold by fold, on the folds they share.

Pairing uses the per-fold differences to account for variation shared
by the two models on the same folds. The folds must match: the
fingerprints are checked and a mismatch raises an error.

`mstop` selects an explicit grid point (default: each model at its
reported optimum). Selected optima are always compared descriptively,
even when they coincide. An explicit point enables an approximate
interval under the random-fold assumptions; choosing that point after
inspecting the curves still incurs selection bias.

**Parameters**

- `other` (CvRiskResult): The second curve. The difference reported is `self - other`, so
  a negative mean favours `self` (risk is a loss).
- `mstop` (int, optional): Grid point to compare both models at. Supplying it enables the
  approximate Nadeau-Bengio interval; omitting it compares each
  model at its own `optimal_mstop` and leaves `interval` None.
- `level` (float, default 0.95): Confidence level of the interval, using Student-t quantiles on
  `n_folds - 1` degrees of freedom.

**Returns**

CvComparison
    Mean paired difference, naive and corrected standard errors, the
    t statistic and (when `mstop` was given) the interval. Curves on
    different response scales return NaN differences with a notice.

**Raises**

- `ValueError`: If the two curves have different fold fingerprints or fold
counts, fewer than two folds, or a fold design the
Nadeau-Bengio correction does not cover.

Source: [`python/mboostrs/cvrisk.py` lines 330-370](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L330-L370)

(cvriskresult-plot)=
## `CvRiskResult.plot`

*Method* of `CvRiskResult`

```python
CvRiskResult.plot()
```

Plot cross-validated risk.

**Returns**

alt.LayerChart

Source: [`python/mboostrs/cvrisk.py` lines 372-380](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L372-L380)

(cvriskresult-to_frame)=
## `CvRiskResult.to_frame`

*Method* of `CvRiskResult`

```python
CvRiskResult.to_frame(output='auto')
```

Long-format table: one row per (fold, mstop) with its risk.

Columns: `fold` (1-based), `mstop`, `risk`, `aggregate_risk`
(the mean risk at that `mstop` across folds), and `is_optimal`
(whether that `mstop` is `self.optimal_mstop`).

**Parameters**

- `output` ({"auto", "pandas", "polars", "records"}, default "auto"): Container for the result.

**Returns**

frame or list of dict
    `n_folds * n_grid` rows.

Source: [`python/mboostrs/cvrisk.py` lines 382-415](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L382-L415)
