# Writing Guide for `mboostrs` Documentation

This guide governs all public documentation for `mboostrs`: the MyST book (`docs/book/`), `README.md`, standalone examples (`examples/`), and public Python/R/Rust docstrings.

## 1. Publication vs. Lab Archive

The book is a scientific monograph and user manual, not a development log.

- **Public book (`docs/book/`):** Written for statisticians, data scientists, and engineers using a released version of `mboostrs`. Describe what the library does, why the method works or fails, and what comparative studies measured.
- **Lab archive (`docs/benchmarks/`, git history, issue tracker):** Retains immutable pre-registrations, SHA-256 protocol digests, raw JSON results, and verbatim agent transcripts for reproducibility. Link these repository files for audit; do not compile raw engineering logs, PR validations, or superseded drafts into the book or PDF.
- **No process narration in published prose:** Never include issue or pull-request numbers (`#123`, `PR 123`), private repository names (`mboostrs-dev`), git commit hashes, branch names, local filesystem paths (`/workspace/...`, `/private/tmp/...`), friction logs, or "before/after fix" development timelines. Report each empirical result with the release version it was measured on.

## 2. Scientific Reporting for Benchmarks and Case Studies

Write empirical chapters in the register of a peer-reviewed statistical software paper (such as *Journal of Statistical Software* or *JMLR*).

1. **Lead with the statistical question.** Open with the modeling problem and empirical question in plain terms before introducing any internal rubric code.
2. **State the design completely.** Specify the data source (or exact data-generating process), sample size, sampling/dependence unit, train/validation/test split, information given to each method (e.g., prescribed formula vs. data alone), fitting and tuning budgets, and comparator versions and settings.
3. **Define metrics unambiguously.** Give units and direction (lower or higher is better). Prefer paired per-task or per-split differences with uncertainty (standard error or interval) over bare point averages.
4. **Report complete outcomes.** Keep timeouts, crashes, and non-convergent runs in the denominator alongside successful fits. State explicitly when datasets were used during method development rather than strictly held out.
5. **Separate evidence from open questions.** State scope and limitations clearly once, in a dedicated closing section. Do not attach repetitive defensive caveats to every paragraph.
6. **Use scorecards sparingly.** Define the scorecard rubric once (`docs/book/benchmarks/overview.ipynb`). In individual study chapters, lead with concrete statistical measurements (loss, coverage, TPR/FDR, wall-clock time) and summarize grades in a compact table rather than substituting rubric codes (`D1b`, `D11`) for plain language.

## 3. Prose Craft and Avoiding AI Slop

- **Direct, concrete prose.** Use active verbs, short sentences, and exact numbers. State the result first, then explain the mechanism.
- **Avoid LLM filler and hype.** Do not use throat-clearing or promotional language (*delve, tapestry, comprehensive, seamless, pivotal, crucial, leverage, underscore, showcase, notably, importantly, it is worth noting, in conclusion, state-of-the-art, groundbreaking*) or repetitive rhetorical templates (*"Not only X, but also Y"*, *"It is not X; it is Y"*).
- **Avoid defensive agent-ese.** Do not narrate script executions or test suites in book prose, do not repeat parenthetical definitions of basic terms on every page (define once per chapter or link the glossary), and do not stamp identical boilerplate dropdown forms across chapters. Integrate estimand, units, dependence, and validation naturally into the chapter's design description.
- **Make tables and figures earn their space.** Every figure and table must be interpreted in the text. Use `display(...)` rather than `print(...)` for DataFrames so both HTML and LaTeX/PDF render cleanly.

## 4. Technical Accuracy and Single Source of Truth

- **Verify against source code.** Every equation, default argument, parameter name, return type, and R-vs-Python difference must match `crates/mboostrs-core/`, `python/mboostrs/`, and `mboostrs/`.
- **Do not duplicate generated facts.** Link to the generated capability reference (`appendix/capabilities.ipynb`) and API manuals (`api/`, `appendix/r_reference.ipynb`) instead of hand-maintaining prose counts of pages, families, or learners that drift.
- **Notebook output policy:**
  - `docs/book/**/*.ipynb`: Book notebooks are executed by `docs/book/build-html.sh` (which merges outputs for HTML and PDF). Keep source notebooks clean and verified by `docs/book/check_cells.py` and `docs/book/check_docs.py`.
  - `examples/python/*.ipynb`: Keep committed without stored outputs; CI executes all standalone examples via `docs/book/execute_examples.sh` and checks them with `tools/check_examples.py`.

## 5. Automated Checks vs. Editorial Judgment

Automated checkers enforce **objective mechanical invariants**: syntax, executable code blocks, valid links/anchors/citations, symbol existence against `mb.__all__` and Rust declarations, generated-surface sync, and zero leaked tracker/commit/path artifacts (`docs/book/check_register.py`). Prose clarity, scientific rigor, and pedagogical flow are enforced by human and editorial review—not by keyword-matching regexes.
