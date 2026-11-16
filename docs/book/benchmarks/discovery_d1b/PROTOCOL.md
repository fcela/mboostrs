# Relationship discovery from data alone (scorecard D1b)

Measures scorecard dimension D1b: finding true structure, and not reporting
false structure, when the analyst supplies only the data and the response name.
Scorecard `mboostrs-scorecard/2.1`, content SHA256
`ccfd33a0b7f9c2cbadc910912551273e66046aaf82c36440d2c286df0cc43fad`.

## Data

Eight candidate predictors `x1` to `x8`, uniform on [-1, 1], 500 rows per draw,
10 seeds per process. The response is:

| process | planted predictors | response |
|---|---|---|
| `additive_nonlinear` | x1, x2 | sin(2 pi x1) + x2^2 + N(0, 0.3^2) |
| `interaction` | x3, x4 | 2 x3 x4 + N(0, 0.3^2) |
| `pure_noise` | none | N(0, 1) |

No method receives the formula, the family, or the planted set.

## Methods and selection rules (fixed before running)

- `mboostrs_discover`: the predictors named in the formula returned by
  `mb.discover(data, "y", seed=seed)`, with its default settings.
- `catboost_importance`: CatBoost (300 iterations, depth 4). Selected are the
  predictors whose importance exceeds the 95th percentile of importances from
  the same model fitted to 10 permuted responses (80 pooled null values).
- `lightgbm_importance`: LightGBM (300 trees, 15 leaves, gain importance), with
  the same permutation rule.

## Metrics

- True positive rate (TPR): share of planted predictors selected, pooled over
  draws. Reported per process.
- False discovery rate (FDR): share of selected predictors that are not planted,
  pooled over draws. Undefined when nothing is selected.
- On `pure_noise`, the share of draws with any selection.

The best competitor for D1b is the method with the lowest FDR at its TPR. For
the additive process this is the LightGBM rule (FDR 0, TPR 0.85).

## Run history

- Run 1 used one permutation for the null threshold. Its noise false-selection
  rates (80% CatBoost, 70% LightGBM) came from that weak null, not from the
  libraries. Corrected to 10 permutations before the reported run.
- Reported run: `results.json`, scores in `scores.json`.

## Command

    .venv/bin/python docs/book/benchmarks/discovery_d1b/run.py \
        docs/book/benchmarks/discovery_d1b/results.json
    .venv/bin/python docs/book/benchmarks/discovery_d1b/score.py

## Limits

Three processes with 10 seeds each. Interaction and nonlinear structure are
planted in a few predictors only; no threshold, categorical, or missing-value
structure is covered. The permutation null is calibrated for these tree models
only; it is not a valid test for mboostrs.
