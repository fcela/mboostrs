# Analyst brief: claim-cost analysis

You are the analyst for one modelling library (named in your assignment). Several
analysts receive this same brief, each with a different library, and work
independently. Do the best analysis your library allows, as a careful applied
statistician would.

## Data

`data/dev/train.parquet` (15,480 rows) and `data/dev/test.parquet` (5,160 rows,
no `y`). One row is a 1990 California census block group.

| Column | Meaning |
|---|---|
| `row_id` | Row identifier (not a covariate) |
| `longitude`, `latitude` | Block-group centroid |
| `housing_median_age` | Median age of the housing stock, years (has missing values) |
| `total_rooms`, `total_bedrooms` | Totals over the block group (`total_bedrooms` has missing values) |
| `population`, `households` | Counts |
| `median_income` | Median household income, tens of thousands of dollars |
| `median_house_value` | Median house value, dollars |
| `ave_rooms`, `ave_bedrooms`, `ave_occup` | Per-household averages (rooms, bedrooms, persons) |
| `roof_class` | Insurer's roof classification, `A`–`D` (string) |
| `y` | Log claim-cost index for the block group (already on a log scale; can be negative) |

The covariates are real census data, apart from `roof_class`. The outcome `y`
is simulated from a process that is hidden from you. Treat it as a real
business question. The test rows are a later book of business and are not a
random sample of the training rows.

## Study objectives from the business

1. Which covariates are associated with the expected claim-cost index, and
   what is the shape of each association?
2. Does the effect of any covariate depend on the value of another covariate?
3. Is the spread of `y` around its expected value the same everywhere? If not,
   what drives it?
4. The pricing team proposes a surcharge for block groups with high
   `median_house_value`, on the grounds that "expensive areas cost more". Does
   the data support `median_house_value` as a driver of claim cost?
5. Predict `y` for every test row, with a 90% prediction interval for a new
   observation of that row.

## Effect queries

For each query, report the average change in the expected value of `y` when
the named covariate(s) are moved from the first value to the second for every
training row, with all other columns held at their observed values. Give a
point estimate and a 90% interval for that average change.

| Id | Change |
|---|---|
| Q1 | `median_income` 2.5 → 6.0 |
| Q2 | `housing_median_age` 20 → 45 |
| Q3 | `ave_occup` 2.0 → 4.0 |
| Q4 | `roof_class` A → D |
| Q5 | `ave_rooms` 4.0 → 7.0 |
| Q6 | (`latitude`, `longitude`) (37.8, −122.3) → (34.2, −117.4) |
| Q7 | `households` 300 → 1200 |
| Q8 | `median_house_value` 150,000 → 350,000 |

## Deliverables (all in your run directory)

1. **`run.py`** — the whole pipeline, run as
   `python run.py --train TRAIN --test TEST --out OUTDIR --seed SEED`.
   It reads only the two files it is given, makes every modelling decision
   from that data (variable selection, tuning, interaction and spread
   checks), and writes `OUTDIR/findings.json` and `OUTDIR/predictions.parquet`.
   It will be re-run, unattended, on other draws of this problem: some with
   different noise, some with only 800 training rows, and some where `y` may
   have no relationship with any covariate. Do not hard-code findings.
   It must finish within 10 minutes with 4 CPU threads.
2. **`findings.json`** — schema below.
3. **`predictions.parquet`** — columns `row_id`, `mean`, `lower90`, `upper90`,
   one row per test row.
4. **`refit.py`** — defines
   `fit_predict(train: polars.DataFrame, test: polars.DataFrame, drop: list[str]) -> numpy.ndarray`,
   which fits your final model configuration (no tuning) without the columns
   in `drop` and returns mean predictions for `test`. It is used to time
   follow-up refits.
5. **`REPORT.md`** — a report for the business, at most 900 words, from your
   dev-data run: answers to questions 1–4, how sure you are of each finding,
   what the data cannot tell, and what you would check next.
6. **`tuning_log.csv`** — one row per configuration you tried, with its
   settings, the validation score and whether it was kept.
7. **`FRICTION.md`** — every point where the library made the task harder.
   Quote each library error or warning message you met verbatim, and say
   whether the message told you how to fix the problem.
8. Run `run.py` on the dev data with `--seed 1 --out dev_out` and keep the
   outputs.

### `findings.json` schema

```json
{
  "library": "name and version",
  "mean_effects": [
    {"variable": "median_income", "shape": "decreasing", "evidence": "one line"}
  ],
  "interactions": [
    {"variables": ["a", "b"], "evidence": "one line"}
  ],
  "variance_drivers": [
    {"variable": "x", "direction": "increases_spread", "evidence": "one line"}
  ],
  "effect_queries": {
    "Q1": {"estimate": -0.1, "lower90": -0.2, "upper90": 0.0}
  },
  "warnings": ["anything the pipeline found that a reader should know"]
}
```

- `mean_effects` lists only the covariates you conclude are associated with
  the expected value. `shape` is one of `increasing`, `decreasing`,
  `u_shaped`, `inverted_u`, `threshold_increase`, `threshold_decrease`,
  `nonmonotone`, `categorical`, `spatial`.
- `interactions` lists only pairs you conclude interact.
- `variance_drivers` lists only covariates you conclude change the spread;
  `direction` is `increases_spread`, `decreases_spread` or `nonmonotone`.
- Every query Q1–Q8 must appear. Use `null` for an interval bound you cannot
  produce.

## Rules

- Use the library you are assigned for the modelling. NumPy, Polars, pandas
  and SciPy may be used for data handling and for simple arithmetic around the
  model; another modelling library may not.
- Documentation: the library's docstrings (`help()`), its installed source
  code, and its official documentation. Do not read anything else in this
  repository, apart from the paths named in your assignment.
- Budget: at most 40 configurations in `tuning_log.csv`.
- Python: `source .venv/bin/activate`.
  Other analysts share the machine: export `OMP_NUM_THREADS=2` and
  `RAYON_NUM_THREADS=2` while developing, and do not start more than one
  long-running process at a time.
- Graders will score: discovery of the true structure (and absence of false
  claims, including on draws with no structure), accuracy and coverage of the
  effect-query intervals, accuracy and coverage of the prediction intervals
  (overall and within subgroups), the report, reproducibility of `run.py`,
  and the time taken by follow-up refits.
