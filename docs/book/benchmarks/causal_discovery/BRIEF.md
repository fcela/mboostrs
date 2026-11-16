# Analyst brief: job-training effects on earnings

You are the analyst for one modelling library (named in your assignment).
Several analysts receive this same brief, each with a different library, and
work independently. Do the best causal analysis your library allows, as a
careful applied statistician would.

## Setting

A state workforce agency ran a voluntary job-training program. Applicants met a
caseworker, and each decided whether to enrol. Enrolment was not randomized.
Caseworkers steered applicants with low prior earnings towards the program, and
applicants earning more than 40,000 dollars a year were rarely eligible. A
follow-up survey one year later recorded earnings; not every applicant answered
it. The agency wants to know what the program did and whom to enrol next year,
when the budget covers 20% of the new applicants.

## Data

`data/dev/train.parquet` (12,000 applicants from the program year) and
`data/dev/test.parquet` (4,000 new applicants for next year, pre-enrolment
columns only). The applicant characteristics come from real survey records of
US men; the program, the survey response and the follow-up outcomes are
simulated from a process that is hidden from you. Treat it as a real
evaluation. The new applicants are not a random sample of the program-year
applicants.

| Column | Measured | Meaning |
|---|---|---|
| `row_id` | — | Row identifier (not a covariate) |
| `age` | at application | Years |
| `education` | at application | Years of schooling |
| `experience` | at application | Years of potential work experience (age − education − 6) |
| `ethnicity` | at application | `cauc` or `afam` |
| `smsa` | at application | Lives in a metropolitan area (`yes`/`no`) |
| `region` | at application | `northeast`, `midwest`, `south`, `west` |
| `parttime` | at application | Worked part time in the year before applying (`yes`/`no`) |
| `prior_earnings` | at application | Earnings in the year before applying, dollars |
| `months_unemployed` | at application | Months without work in the two years before applying |
| `distance_km` | at application | Distance from home to the nearest training centre, km |
| `caseworker_note_len` | at application | Length of the caseworker's intake note, characters |
| `enrolled` | at enrolment | 1 if the applicant enrolled in the program (train only) |
| `certificate` | during the program | 1 if the applicant earned the program certificate (train only) |
| `responded` | at follow-up | 1 if the applicant answered the follow-up survey (train only) |
| `hours_post` | at follow-up | Hours worked in the follow-up year; missing if no response (train only) |
| `earnings_post` | at follow-up | Earnings in the follow-up year, dollars; 0 if not employed; missing if no response (train only) |

## Study objectives from the agency

1. What was the average effect of enrolling on follow-up earnings, in dollars,
   over all program-year applicants (ATE) and over those who enrolled (ATT)?
2. Did the program work by getting people into work, by raising the earnings
   of those in work, or both?
3. For whom was the effect larger or smaller? Name the characteristics that
   change the size of the effect, and describe how.
4. Which columns must an analysis adjust for, and which must it not adjust
   for? Give a role for every column from `age` to `earnings_post`.
5. For which applicants can these data not tell what the program did?
6. Rank next year's applicants so that enrolling the top 20% gives the largest
   total gain in earnings.
7. How sensitive are your conclusions to a characteristic that affects both
   enrolment and earnings but is not in the data?

## Effect queries

Each query is an average over the 12,000 program-year applicants in the
training file (all of them, whether or not they answered the survey), of the
difference between expected follow-up earnings if enrolled and if not
enrolled. Give a point estimate and a 90% interval.

| Id | Population |
|---|---|
| ATE | All applicants |
| ATT | Applicants who enrolled |
| CATE_educ_lt12 | `education` < 12 |
| CATE_educ_ge16 | `education` ≥ 16 |
| CATE_prior_q1 | `prior_earnings` at or below its 25th percentile in the training file |
| CATE_prior_q4 | `prior_earnings` at or above its 75th percentile in the training file |
| CATE_age_lt25 | `age` < 25 |
| EMP_ATE | All applicants; the outcome is employment (`earnings_post` > 0), so the effect is a difference in probability |

## Deliverables (all in your run directory)

1. **`run.py`**, the whole pipeline, run as
   `python run.py --train TRAIN --test TEST --out OUTDIR --seed SEED`.
   It reads only the two files it is given, makes every modelling decision
   from that data, and writes `OUTDIR/findings.json` and
   `OUTDIR/cate_predictions.parquet`. It will be re-run, unattended, on other
   draws of this problem: some with different noise, some with only 1,500
   training rows, and some where enrolment has no effect at all. Do not
   hard-code findings. It must finish within 10 minutes with 4 CPU threads.
2. **`findings.json`**, schema below.
3. **`cate_predictions.parquet`**: columns `row_id`, `cate`, `lower90`,
   `upper90`, one row per test applicant: the expected effect of enrolling
   that applicant on follow-up earnings, in dollars, with a 90% interval.
   The ranking for question 6 is the order of `cate`.
4. **`refit.py`**, defining
   `fit_ate(train: polars.DataFrame, drop: list[str]) -> float`, which fits
   your final configuration (no tuning) without the columns in `drop` and
   returns the ATE estimate. It is used to time follow-up analyses.
5. **`REPORT.md`**, a report for the agency, at most 900 words, from your
   dev-data run: answers to questions 1–7, how sure you are of each, what the
   data cannot tell, and what you would check next.
6. **`tuning_log.csv`**, one row per configuration you tried, with its
   settings, the validation score and whether it was kept.
7. **`FRICTION.md`**: every point where the library made the task harder.
   Quote each library error or warning message you met verbatim, and say
   whether the message told you how to fix the problem.
8. Run `run.py` on the dev data with `--seed 1 --out dev_out` and keep the
   outputs.

### `findings.json` schema

```json
{
  "library": "name and version",
  "effect_queries": {
    "ATE": {"estimate": 1000.0, "lower90": 500.0, "upper90": 1500.0}
  },
  "roles": {"age": "confounder"},
  "adjustment_set": ["age"],
  "effect_modifiers": [
    {"variable": "education", "direction": "larger_when_lower", "evidence": "one line"}
  ],
  "channels": {"employment": true, "earnings_if_employed": true},
  "limited_overlap": [
    {"description": "who the data cannot speak for", "variable": "prior_earnings"}
  ],
  "warnings": ["anything the pipeline found that a reader should know"]
}
```

- Every query must appear. Use `null` for an interval bound you cannot produce.
- `roles` has one entry for every column from `age` to `earnings_post`, each
  one of `confounder`, `instrument`, `treatment_only`, `outcome_only`,
  `mediator`, `post_treatment_selection`, `treatment`, `outcome`,
  `irrelevant`.
- `effect_modifiers` lists only characteristics you conclude change the
  effect in dollars; `direction` is `larger_when_higher`,
  `larger_when_lower` or `nonmonotone`.
- `channels` says which channels you conclude carry a non-zero part of the
  effect.

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
- Graders will score: the effect estimates and the coverage of their
  intervals (including on draws where the program has no effect), the
  per-applicant effects and their intervals, the value of your top-20%
  ranking, the variable roles, the effect modifiers and channels you claim
  (and the absence of false claims), the report, reproducibility of `run.py`,
  and the time taken by follow-up analyses.
