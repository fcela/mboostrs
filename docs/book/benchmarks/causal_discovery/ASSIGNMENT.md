# Analyst assignment

Each analyst is a Claude Haiku agent launched once, in the background, with
the message below and no further help. Only the library name and version,
the run directory, the extra text and the documentation source change
between analysts. All analysts also receive `BRIEF.md` word for word.

## Message template

> Assignment: you are the analyst for the library **{LIBRARY}** (Python package `{PACKAGE}`, version {VERSION}{EXTRA}).
>
> Read the brief at docs/book/benchmarks/causal_discovery/BRIEF.md and carry it out completely. Your run directory is docs/book/benchmarks/causal_discovery/{RUN_DIR}/ — write every deliverable there. The data is in docs/book/benchmarks/causal_discovery/data/dev/.
>
> Documentation you may use for {LIBRARY}: {DOCS}. Do not read anything else in the repository, and do not read any other analyst's run directory.
>
> Work on your own, without asking questions; nobody will answer. Deliver all eight items in the brief. Before you finish, confirm that `python run.py --train <dev train> --test <dev test> --out dev_out --seed 1` runs from your run directory and writes findings.json and cate_predictions.parquet, and that refit.py imports and runs. Your final message: a short list of the deliverables written, the final model, and anything you could not complete. Then put the full text of REPORT.md and FRICTION.md in your final message as well, each between lines `----- REPORT.md -----` / `----- end REPORT.md -----` (and likewise for FRICTION.md), because the file tool may refuse to write report files.

## Per-library values

| Library | Package, version | Extra text | Documentation allowed |
|---|---|---|---|
| mboostrs | `mboostrs` 0.1.3 | "; it ships an agent guide: `python -m mboostrs guide` and `python -m mboostrs guide <topic>`" | `help()`, the agent guide, installed source (`python/mboostrs/`), and the book at `docs/book/` (usage/, methodology/, appendix/, case_studies/); docs/book/benchmarks/ and docs/book/cases/ excluded |
| EconML | `econml` 0.17.0 | "; scikit-learn estimators may be used as nuisance models" | `help()`, installed source, https://econml.azurewebsites.net/ |
| DoubleML | `doubleml` 0.11.4 | "; scikit-learn estimators may be used as nuisance learners" | `help()`, installed source, https://docs.doubleml.org/ |
| statsmodels | `statsmodels` 0.15.0 | "; any of its regression, GLM, GAM, discrete-choice or treatment-effect tools, with patsy formulas if you like" | `help()`, installed source, https://www.statsmodels.org/stable/ |
| LightGBM | `lightgbm` 4.7.0 | "; build any meta-learner you need from LightGBM models" | `help()`, installed source, https://lightgbm.readthedocs.io/ |

Run directories: `runs/<library>/` for the first run and `rerun/mboostrs/`
for the mboostrs rerun after the fixes.
