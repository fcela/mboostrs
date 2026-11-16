# Causal estimator benchmark

This benchmark compares mboostrs' causal estimators (`aipw`, `effect_model`,
`rank_by_effect`, `effect_modifiers`) with EconML (`LinearDML`,
`CausalForestDML`), DoubleML (`IRM`) and a hand-built R-learner on the
job-training design of `../causal_discovery/`. The analyst choices are fixed
across libraries, so the differences come from the libraries. The question,
draws, arms, measures and planned contrasts are in `PROTOCOL.md`.

## Files

| File | Content |
|---|---|
| `PROTOCOL.md` | Question, draws, response weights, arms with exact settings, measures, attribution rules |
| `PREREG_HASHES.txt` | sha256 of the protocol and scripts, recorded before the run |
| `draws.py` | Generates the study draws from the generator in `../causal_discovery/prepare.py` |
| `run_arms.py` | Fits every arm on every draw and grades it; writes `per_draw_results.jsonl` |
| `summarize.py` | Summarises the per-draw records into `results.json` and `per_draw_results.parquet` |
| `per_draw_results.jsonl` | One record per draw and arm, as written by `run_arms.py` |
| `per_draw_results.parquet` | The same records as a table |
| `results.json` | Per-arm measures and the pre-specified contrasts |

The per-row CATE file (`per_row.parquet`, one row per test applicant, draw and
arm) is written to `/tmp` by default and is not committed; pass `--rows-out` to
keep it elsewhere.

## Requirements

- mboostrs built from this repository, with the compiled extension in
  `python/mboostrs/` (`maturin develop --release`, or the copy step in
  `.claude/agent_common.md`).
- EconML 0.17.0, DoubleML 0.11.4, scikit-learn 1.9.0, statsmodels 0.15.0,
  polars, pandas.
- The AER::CPS1988 file `CPS1988.csv`. `draws.py` checks its sha256 against
  `ef5ee9cbf3674f9cf242549517cf7801bffc5e98c1006055c9fdefebaf386251`.

## Reproduce

From the repository root, with the environment above active:

```bash
python docs/book/benchmarks/causal_estimators/run_arms.py --src /path/to/CPS1988.csv
python docs/book/benchmarks/causal_estimators/summarize.py
```

`run_arms.py` runs about 20 arms on each of 12 draws (10 with effects, 2
without). It uses 4 threads and writes `per_draw_results.jsonl` in this
directory. A full run takes about 40 minutes on an Apple Silicon laptop; the
timings in the output depend on the load on the machine.

To check the protocol hashes against these files:

```bash
shasum -a 256 docs/book/benchmarks/causal_estimators/{PROTOCOL.md,draws.py,run_arms.py,summarize.py}
```

The output should match `PREREG_HASHES.txt`. `PROTOCOL.md` has an erratum
section added after the run, so its current hash is the second entry for that
file in `PREREG_HASHES.txt`. Any other edit to a script or to the protocol after
that file was recorded changes the hash, and the change has to be listed in the
report.

`run_arms.py --limit-rows N` runs the arms on the first N rows of each draw. It
is for checking the setup and does not produce study results.
