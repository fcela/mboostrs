# Deviations from the protocol

Both deviations were made after the results were read. They change only the
analysis; no arm was rerun or changed. The tables in `tables.md` use them.

1. **Ranking set.** The protocol ranks arms on the tasks every arm completed.
   `distreebution.pmqrf` completed 10 of 105 real-data runs (distreebu_rs
   panics, see PROTOCOL.md), which left 2 of 35 real datasets for the
   ranking. Mean rank, gap to best and best-on counts now use the arms that
   completed at least 90% of their runs (every arm except PMQRF on real data),
   on the tasks all of those arms completed: 16 of 16 planted tasks, 32 of
   35 real datasets. PMQRF's paired comparisons are still reported, on the
   tasks it completed.
2. **Median skill.** The protocol reports the mean skill over climatology.
   Two arms have single forecasts so far off that the mean is meaningless
   (`hessboost.normal`, whose mean skill is about −1.7e8 on real data, and
   `mboostrs.GaussianLSS`, about −21). The median skill is reported beside
   the mean, and a mean below −10 is printed as "< −10".
