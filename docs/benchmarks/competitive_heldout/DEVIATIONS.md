# Deviations from PROTOCOL.md

1. The NaN percentile fix in run_one.py (`nanquantile`) was made before the
   first study run; no run used the earlier code.
2. Analysis: the reference arm's median interval width over sd was NaN because
   some OLS intervals had non-finite width. The median is taken over finite
   widths, and the number of runs with a non-finite width is reported
   (`nonfinite_width_runs`). Coverage is unaffected.
3. Analysis: "R tasks with fewer than 500 rows" is implemented as tasks whose
   seed-5000 training split has fewer than 400 rows (80% of 500).
