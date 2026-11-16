# `insights`

Defined in `mboostrs.insights`; exported as `mboostrs.insights`.


*Function*

```python
insights(data, response, *, top_mains=8, top_pairs=5, seed=0, family=None)
```

Run screen -> fit -> diagnose -> propose; return ranked findings.

**Parameters**

- `data` (frame with the response column and candidate predictors.)
- `response` (name of the response column.)
- `top_mains` (how many screened columns enter the additive fit.)
- `top_pairs` (how many screened pairs are reported as detections.)
- `seed` (random seed threaded through the pipeline.)
- `family` (model family for the fit. ``None`` infers one from the): response: only values `{0, 1}` -> `mboostrs.families.Binomial`,
  non-negative integers with more than two distinct values ->
  `mboostrs.families.Poisson`, anything else the Gaussian
  default. The family used is recorded in every finding's evidence
  under `"family"`.

**Returns**

list of `Finding`, ranked measured-first, deduplicated across the
screen and propose stages.

Source: [`python/mboostrs/insights.py` lines 243-453](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/insights.py#L243-L453)
