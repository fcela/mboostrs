# Analyst assignment

Each analyst was a Claude Haiku agent launched once, in the background, with
the message below and no further help. The only parts that changed between
analysts were the library name and version, the run directory, and the
documentation source shown in the table. All seven also received `BRIEF.md`
word for word.

## Message template

> Assignment: you are the analyst for the library **{LIBRARY}** (Python package `{PACKAGE}`, version {VERSION}{EXTRA}).
>
> Read the brief at docs/book/benchmarks/claims_discovery/BRIEF.md and carry it out completely. Your run directory is docs/book/benchmarks/claims_discovery/runs/{RUN_DIR}/ — write every deliverable there. The data is in docs/book/benchmarks/claims_discovery/data/dev/.
>
> Documentation you may use for {LIBRARY}: {DOCS}. Do not read anything else in the repository, and do not read any other analyst's run directory.
>
> Work on your own, without asking questions; nobody will answer. Deliver all eight items in the brief. Before you finish, confirm that `python run.py --train <dev train> --test <dev test> --out dev_out --seed 1` runs from your run directory and writes findings.json and predictions.parquet, and that refit.py imports and runs. Your final message: a short list of the deliverables written, the final model, and anything you could not complete.

## Per-library values

| Library | Package, version | Extra text | Documentation allowed |
|---|---|---|---|
| mboostrs | `mboostrs` 0.1.3 | — | `help()`, installed source (`python/mboostrs/`), and the book at `docs/book/` (usage/, methodology/, appendix/, case_studies/); docs/book/benchmarks/ and docs/book/cases/ excluded |
| XGBoost | `xgboost` 3.4.0 | — | `help()`, installed source, https://xgboost.readthedocs.io/ |
| LightGBM | `lightgbm` 4.7.0 | — | `help()`, installed source, https://lightgbm.readthedocs.io/ |
| CatBoost | `catboost` 1.2.10 | — | `help()`, installed source, https://catboost.ai/docs/ |
| InterpretML EBM | `interpret` 0.7.8 | "; use `interpret.glassbox.ExplainableBoostingRegressor` and related tools in that package" | `help()`, installed source, https://interpret.ml/docs/ |
| pyGAM | `pygam` 0.12.0 | — | `help()`, installed source, https://pygam.readthedocs.io/ |
| statsmodels | `statsmodels` 0.15.0 | "; any of its regression, GLM, GAM or quantile-regression tools, with patsy formulas if you like" | `help()`, installed source, https://www.statsmodels.org/stable/ |

The mboostrs wording differs in one place: its documentation line names the
book's directories and excludes `docs/book/benchmarks/` and `docs/book/cases/`
instead of saying "do not read anything else in the repository", because the
library's documentation lives in this repository.
