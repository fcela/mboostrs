# Held-out study of `mb.fit`

Runner for the study recorded in
[docs/benchmarks/2026-11-21-fit-heldout.md](../../docs/benchmarks/2026-11-21-fit-heldout.md).
It compares `mb.fit(train, "y")` at commit 12c87a3 (no budget passed to
autospec) with the working tree on 95 datasets, scoring held-out loss and the
direction of 38 published effects.

The scripts are the ones that produced the recorded results, with lint-only
edits (unused imports removed, one lambda written as a function) and one
later fix: `run_one.py` takes the effect percentiles with `nanquantile`,
because some files store missing values as NaN. The recorded results are
unchanged by it: every known-effect column with NaN belongs to a dataset on
which both arms failed every run. They read and
write next to themselves: `data/` (parquet files and manifests), `results*.jsonl`,
`analysis*.json` and `old_fit.py`, all gitignored.

From this directory, with the repository venv:

```sh
PY=../../.venv/bin/python
git show 12c87a3:python/mboostrs/_procedure/_fit.py > old_fit.py
$PY -I prep_openml.py data          # OpenML, through scikit-learn's fetch_openml cache
rm data/45402.parquet data/197.parquet   # duplicates of 44985 and 44978
$PY -I prep_uci.py                  # UCI, through ucimlrepo (uci/metadata.json lists candidates)
$PY prep_mb.py                      # three mboostrs.datasets tables
cp ../../docs/benchmarks/fit_heldout/known_effects.json .
$PY -I scan.py                      # onceover scan -> onceover_scan.json
$PY -I run_all.py $PY 8 4000,4001,4002 results_v4.jsonl
$PY -I analyze.py results_v4.jsonl 4000,4001,4002
```

`prep_openml.py` reads every dataset in the local scikit-learn OpenML cache,
so the set it finds depends on that cache; the recorded manifest
(`docs/benchmarks/fit_heldout/manifest_openml.json`) lists the 36 used.
`prep_uci.py` drops UCI tasks that list only classification but have numeric
class codes (Primary Tumor, Absenteeism); the recorded manifest lists the 56
used. `analyze.py` writes `analysis_v2.json` whichever seeds it is given.
