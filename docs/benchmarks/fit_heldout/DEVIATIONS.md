# Deviations from the pre-registrations

Deviation note (added before the study runs, after one smoke run at seed 999): dataset 44968 has y = c3 + 1.18 exactly, giving log ratios near -68. The primary analysis is unchanged; a sensitivity analysis using the median of r and the analysis without 44968 is added and labelled as such.

Deviation 2 (found while the study ran, before any analysis): when the search
selects a distributional (LSS) model, `predict(type="response")` returns a dict
of distribution parameters, and the runner's `float()` cast failed. The runner
now scores `model.predict(type="mean")`, the expected response, for such
models. Only the runs that failed with that TypeError are rerun; every other
run is kept as first recorded.
