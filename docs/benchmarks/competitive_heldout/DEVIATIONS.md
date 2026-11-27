# Deviations from PROTOCOL.md

1. The NaN percentile fix in run_one.py (`nanquantile`) was made before the
   first study run; no run used the earlier code.
2. Analysis: the reference arm's median interval width over sd was NaN because
   some OLS intervals had non-finite width. The median is taken over finite
   widths, and the number of runs with a non-finite width is reported
   (`nonfinite_width_runs`). Coverage is unaffected.
3. Analysis: "R tasks with fewer than 500 rows" is implemented as tasks whose
   seed-5000 training split has fewer than 400 rows (80% of 500).
4. 2026-11-26 refresh of the `mboostrs` arm (v0.2.3): all 320 `mboostrs` runs (285 real
   runs on seeds 5000–5002 and 35 planted draws), the 50,000 × 20 D5 analyst
   session, and the 10-task D10 reproducibility check were rerun on the same
   Apple M4 machine and single-thread harness against the unchanged competitor
   records (`catboost`, `lightgbm`, `xgboost`, `ebm`, `reference`). Because the
   library changes evaluated in this refresh were developed after inspecting
   the 2026-11-21 results on the same 95 real tables, the real-data comparison
   carries additional development exposure for `mboostrs`.
5. 2026-11-26 second refresh of the `mboostrs` arm (v0.2.4): all 320 `mboostrs`
   runs, the 50,000 × 20 D5 session, and the 10-task D10 check were rerun on the
   same Apple M4 harness after enabling `search="fast"` by default in `mb.fit`,
   fixing the remaining 16 D6 failure modes, adding $O(n+L)$ diagonal `brandom`
   and mixed numeric+categorical histogram `btree`, restricting undeclared
   grouped-fold auto-detection to integer-coded unit IDs, and defaulting
   `Fit.predict` to `extrapolation="clip"`. Because these upgrades were also
   informed by diagnostics on the same 95 real tables, development exposure
   remains elevated relative to the five untouched competitors; fresh held-out
   splits (seeds 6000–6001) were run separately as a generalization check.
