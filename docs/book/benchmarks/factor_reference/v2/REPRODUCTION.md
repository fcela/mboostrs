# Versioned control runners

The final runner is `benches/bench_factor_reference_v2.py`. The original
`bench_factor_reference.py` remains byte-identical to the schema-repair
runner, so its published failed/successful replay remains reproducible after
the dictionary repair is merged. The report is `benches/report_factor_reference.py`.

The first version-2 capture used the new generator in the original module.
Its two raw phases, gate summary and HTML are retained in `initial_capture/`
at source revision f290194. The runner was then given its own module without
changing generators, seeds, calls, policy, thresholds or loss. Both phases
were repeated under the distinct runner identity; the report verifies exact
input/prediction/MSE agreement against that first capture on every draw.
This is a repetition for version isolation, not another independent confirmation.
The frozen analytical protocol is unchanged.

Use the schema-repaired parent checkout for the before import and the full
fixed checkout for the after import, with the repository Python environment:

```bash
PYTHONPATH=<schema-parent>/python python -m benches.bench_factor_reference_v2 --phase before
PYTHONPATH=<dictionary-fix>/python python -m benches.bench_factor_reference_v2 --phase after
python -m benches.report_factor_reference
```
