# Competitive held-out study

Runner for [docs/benchmarks/2026-11-21-competitive-heldout.md](../../docs/benchmarks/2026-11-21-competitive-heldout.md),
scored against `mboostrs-scorecard/2.1`. It needs the real datasets that
[../fit_heldout](../fit_heldout/README.md) prepares in `../fit_heldout/data`,
and `../fit_heldout/known_effects.json`. The scripts are the ones that produced
the recorded results. Two kinds of edit were made in the copy: the sibling
directory path (`../proof` became `../fit_heldout`), and lint fixes in
`d10.py`.

From this directory, with the repository venv and scikit-learn, catboost,
lightgbm, xgboost and interpret installed:

```sh
PY=../../.venv/bin/python
$PY -I prep_synth.py                          # planted tasks S1-S7 -> data/
$PY -I run_all.py $PY 8 results.jsonl         # 1,920 runs, about 2 hours on 8 cores
for a in mboostrs catboost lightgbm xgboost ebm reference; do
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -I session.py $a >> sessions.jsonl
done                                           # D5 analyst session
$PY -I d10.py $PY                             # D10 reruns
$PY -I analyze.py                             # analysis.json
```
