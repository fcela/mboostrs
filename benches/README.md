# Benchmarks

Every maintained benchmark study is registered in [registry.json](registry.json)
and driven through the single entry point [run.py](run.py).

## Quick start

```bash
# List all registered studies, their tiers, refresh modes, and book pages
.venv/bin/python benches/run.py list
.venv/bin/python benches/run.py list --tier headline

# Check artifact freshness against HEAD
.venv/bin/python benches/run.py status

# Preview the exact commands for a study without running it
.venv/bin/python benches/run.py run distributional_comparison --dry-run

# Rerun a study, run its post-analysis/checkers, and re-execute its book page(s)
.venv/bin/python benches/run.py run distributional_comparison --update-book
.venv/bin/python benches/run.py run realdata_autospec --workers 8 --update-book

# Re-execute only the book page(s) after manually editing an artifact
.venv/bin/python benches/run.py update-book distributional_shootout
```

## Study tiers

Studies are grouped into five tiers that mirror Part V of the book
(`docs/book/myst.yml`):

- **`headline`**: Canonical cross-library studies that determine where mboostrs
  stands overall (`distributional_shootout`, `heldout_competition`,
  `fit_heldout`, `competitive_shootout`, `realdata_autospec`).
- **`domain`**: Focused cross-library suites by task type
  (`distributional_comparison`, `library_comparison`, `public_suites`).
- **`workflow`**: End-to-end discovery, causal, and agent studies.
- **`calibration`**: Uncertainty, response-scale, and conformal coverage audits.
- **`ablation`**: Internal component ablations and engineering studies
  (`interaction_capacity`, `stopping_rule_ablation`, `predictive_tracks`,
  `encoding_tracks`, etc.).

## Refresh modes

- **`full`**: Reruns every arm (mboostrs and competitors) and regenerates
  analysis outputs.
- **`mboostrs_only`**: Remeasures only the mboostrs rows on identical draws and
  merges them into the stored artifact while retaining competitor rows verbatim
  (fast turnaround when iterating on mboostrs).
- **`recorded_external`**: Historical artifacts without an in-repo runner;
  dependent book pages validate and render stored outputs.

## Adding or updating a benchmark

1. Add an entry to [registry.json](registry.json) with `id`, `title`, `tier`,
   `refresh_mode`, `runner_script`, `command`, `post_commands`, `artifacts`,
   and `pages`.
2. Add the page to the matching subsection under `Part V — Benchmarks` in
   `docs/book/myst.yml`.
3. Verify registry integrity with `pytest tests/test_benchmark_registry.py` and
   book integrity with `.venv/bin/python docs/book/check_docs.py`.
