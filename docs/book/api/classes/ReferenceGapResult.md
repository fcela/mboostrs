# `ReferenceGapResult`

Defined in `mboostrs.reference_gap`; exported as `mboostrs.ReferenceGapResult`.


*Class*

```python
ReferenceGapResult(
    gap: float,
    gap_se: float,
    material: bool,
    relative_gain: float,
    reference_loss: float,
    model_loss: float,
    model_formula: str,
    reference_formula: str,
    model_mstop: int,
    reference_mstop: int,
    model_at_cap: bool,
    reference_at_cap: bool,
    reference_in_sample_loss: float,
    candidates: polars.dataframe.frame.DataFrame,
    closed_share: float | None,
    combined_formula: str | None,
    combined_gain: float | None,
    combined_gain_se: float | None,
    folds: int,
    family: str,
    notes: list[str] = <factory>,
    data: polars.dataframe.frame.DataFrame | None = None,
    fold_losses: dict = <factory>,
    reference_columns: list = <factory>,
    reference_extra_columns: list = <factory>,
    screen_residual: str = 'none'
) -> None
```

Result of comparing a specification with a tree reference, including candidate revisions.

Losses are the family's risk per held-out row, on the scale the family
fits (the link scale for the log1p target of a quantile model, for example).
A positive `gap` means the reference has the lower loss.

**Attributes**

- `gap` (float): Mean over folds of `model loss - reference loss` at each model's own
  best step count. Positive when the reference is better.
- `gap_se` (float): Standard error of `gap` across folds.
- `material` (bool): Whether `gap > 2 * gap_se`.
- `relative_gain` (float): `gap` as a share of the specification's loss above the saturated
  floor (see `_risk_floor`). Zero for families whose floor is zero.
- `reference_loss, model_loss` (float): Fold-aggregated held-out loss of the reference and the specification.
- `model_formula, reference_formula` (str): The specification, and the reference `btree` formula.
- `model_mstop, reference_mstop` (int): Best step count of each model on the folds.
- `model_at_cap, reference_at_cap` (bool): Whether the best step count sits within 5% of `max_mstop`. Such a
  value is unfinished tuning: the loss was still falling at the cap.
- `reference_in_sample_loss` (float): Loss of the reference fitted at `reference_mstop` on all analysis
  rows and scored on the same rows. Compare with `reference_loss`.
- `candidates` (polars.DataFrame): The top candidate terms in rank order, one row each. Columns:
  `rank`, `kind` (`"interaction"`, `"column"` for a column the
  formula omits, `"lag"` or `"user"`), `term`,
  `source`, `screen_share` (out-of-sample share of the mean square of
  the screening residual that the term removes in the screen; the
  residual is described under `screen_residual`), `measured_gain`
  and `gain_se` (held-out gain of the specification plus this term,
  over the specification, with its fold standard error),
  `relative_gain`, `closed_share` (`measured_gain / gap`),
  `mstop` and `at_cap`. Candidates are measured whether or not the
  gap is material; `closed_share` is meaningful only when it is.
- `closed_share` (float or None): Share of the gap closed by adding every candidate with a positive
  measured gain together, in one refit. `None` when the gap is not
  material or no candidate was measured. `combined_gain` is still
  reported when the gap is not material.
- `combined_formula` (str or None): The specification plus those candidates.
- `combined_gain, combined_gain_se` (float or None): Held-out gain of `combined_formula` over the specification.
- `folds` (int): Number of fold columns the gap rests on.
- `family` (str): Name of the family the models were fitted with.
- `notes` (list of str): Plain statements about the checks that need attention, such as an
  overfit reference, a gap resting on fewer than three folds, or a
  candidate pair that cannot be turned into a term.
- `data` (polars.DataFrame): The rows the models were fitted on: time-ordered when `time=` was
  given, with the lag and rolling columns the candidates refer to.
- `fold_losses` (dict): `"model"` and `"reference"`: the held-out loss of each fold, at
  each model's best step count. `gap` is the mean of their difference.
- `reference_columns` (list of str): The columns the reference `btree` was given, including lag and
  rolling features built with `time=`.
- `reference_extra_columns` (list of str): The members of `reference_columns` that the formula does not use.
  A gap that appears with these columns and not without them points at
  information the specification leaves out.
- `screen_residual` (str): `"family"` when the candidate screen ran on the scaled negative
  gradient of the family loss, `"response"` when it ran on
  `y - mean` (Gaussian fits, and families whose loss couples rows), or
  `"none"` when no screen ran.

Source: [`python/mboostrs/reference_gap.py` lines 48-270](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/reference_gap.py#L48-L270)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`describe`](#referencegapresult-describe) | Method | Plain-text summary: the gap, its source and what to try next. |
| [`to_dict`](#referencegapresult-to_dict) | Method | The result as JSON-ready data. The fitted models are not included. |

(referencegapresult-describe)=
## `ReferenceGapResult.describe`

*Method* of `ReferenceGapResult`

```python
ReferenceGapResult.describe() -> str
```

Plain-text summary: the gap, its source and what to try next.

Source: [`python/mboostrs/reference_gap.py` lines 163-231](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/reference_gap.py#L163-L231)

(referencegapresult-to_dict)=
## `ReferenceGapResult.to_dict`

*Method* of `ReferenceGapResult`

```python
ReferenceGapResult.to_dict(*, include_arrays: bool = False) -> dict
```

The result as JSON-ready data. The fitted models are not included.

`include_arrays=True` adds `"fold_losses"`: the per-fold held-out
loss of the specification and of the reference.

Source: [`python/mboostrs/reference_gap.py` lines 233-270](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/reference_gap.py#L233-L270)
