# Deviations from PROTOCOL.md

1. The NaN percentile fix in run_one.py (`nanquantile`) was made before the
   first study run; no run used the earlier code.
2. Analysis: the reference arm's median interval width over sd was NaN because
   some OLS intervals had non-finite width. The median is taken over finite
   widths, and the number of runs with a non-finite width is reported
   (`nonfinite_width_runs`). Coverage is unaffected.
3. Analysis: "R tasks with fewer than 500 rows" is implemented as tasks whose
   seed-5000 training split has fewer than 400 rows (80% of 500).
4. 2026-11-26 refresh of the `mboostrs` arm: all 320 `mboostrs` runs (285 real
   runs on seeds 5000–5002 and 35 planted draws), the 50,000 × 20 D5 analyst
   session, and the 10-task D10 reproducibility check were rerun on the same
   Apple M4 machine and single-thread harness against the unchanged competitor
   records (`catboost`, `lightgbm`, `xgboost`, `ebm`, `reference`). Because the
   library changes evaluated in this refresh were developed after inspecting
   the 2026-11-21 results on the same 95 real tables, the real-data comparison
   carries additional development exposure for `mboostrs`.
