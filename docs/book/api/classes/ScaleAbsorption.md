# `ScaleAbsorption`

Defined in `mboostrs.absorption`; exported as `mboostrs.ScaleAbsorption`.


*Class*

```python
ScaleAbsorption(
    verdict: str,
    rule: str,
    absorbed: dict,
    scale: dict,
    oof: dict,
    residual_structure: dict,
    specifications: dict,
    folds: int,
    seed: int,
    notes: tuple
) -> None
```

Result of `scale_absorption`.

`verdict` is one of `"scale effect robust"`,
`"scale effect absorbs mean misspecification"` or `"inconclusive"`.
`rule` states the thresholds that decided it.

`scale` maps each variant (`"given"`, `"flexible_mean"`,
`"slower_scale"`) to `sd_log_sigma`, the standard deviation of the
fitted linear predictor of sigma over the training rows, and
`term_ranges`, the range (max minus min over the training rows) of each
sigma term's fitted effect.

`absorbed` gives, for the flexible mean and the slower scale rate, the
share of the given model's `sd_log_sigma` that the variant removed:
`1 - sd_variant / sd_given`. Negative values mean the variant's spread is
larger.

`oof` maps each variant to `mean_nll` (out-of-fold joint negative
log-likelihood per observation, averaged over the folds) and `fold_nll`.
The two alternatives also carry `gain` (given minus variant, so a
positive value means the variant predicts held-out rows better), `se`
(standard error of that paired gain across folds) and `t` (gain over
se).

`residual_structure` maps each variant to the standardized residual
`(y - mu) / sigma` screened against the model's predictors:
`columns` holds the curvature of the residual on each predictor
(`curve_r2 - linear_r2`) and `pairs` the out-of-sample explained
variation of each predictor pair. These diagnostics are reported for
context; the verdict uses the scale and held-out score criteria above.

`specifications` records the mean formula of the flexible variant and
the per-parameter learning rates of the slower variant. `notes` lists
conditions the reader should know about.

Source: [`python/mboostrs/absorption.py` lines 55-122](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/absorption.py#L55-L122)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | The result as plain dicts, lists and floats, for JSON or a report. |

(scaleabsorption-to_dict)=
## `ScaleAbsorption.to_dict`

*Method* of `ScaleAbsorption`

```python
ScaleAbsorption.to_dict(*, include_arrays=False) -> dict
```

The result as plain dicts, lists and floats, for JSON or a report.

`include_arrays` is accepted for the export convention every result
record follows. This result holds no per-row arrays, so the flag adds
nothing to the payload.

Source: [`python/mboostrs/absorption.py` lines 104-122](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/absorption.py#L104-L122)
