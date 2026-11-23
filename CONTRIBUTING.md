# Contributing to mboostrs

mboostrs is a statistical modeling library with a Rust engine and Python and
R interfaces. This guide covers contributions to its code, tests and
documentation.

## Quick start

```bash
git clone https://github.com/fcela/mboostrs.git
cd mboostrs

# Python development install
pip install maturin
maturin develop --release

# Run tests (no R needed)
cargo test --workspace --exclude mboostrs-r
pytest -m "not r_parity"

# Lint
cargo fmt --check --all
cargo clippy --workspace --exclude mboostrs-r --all-targets -- -D warnings
ruff check python/ tests/ tools/ benches/
```

## Project structure

```
crates/
  mboostrs-core/src/    # Pure Rust engine — the hot loop, families, learners
  mboostrs-py/src/      # PyO3 bindings — numpy ↔ Rust conversion
  mboostrs-r/           # R bindings (extendr), built as a static library for the R package
mboostrs/               # R package: R sources, DESCRIPTION, NAMESPACE, man pages
python/mboostrs/        # Python layer — model classes, formula parser, plotting
tests/                  # pytest test suite
docs/book/              # MyST book (website and PDF)
examples/python/        # Interactive notebooks
```

## Development workflow

### Rust changes

The core engine is in `crates/mboostrs-core/src/`. Key files:

| File | What it does |
|------|-------------|
| `engine.rs` | Boosting loop, `fit_with()`, `try_fit_gaussian_gram` |
| `families.rs` | `Family` trait and standard-family implementations |
| `learners.rs` | `PreparedLearner`, base learner preparation and fitting |
| `cv.rs` | Cross-validation: `cv()`, `cvrisk_with()` |
| `linalg.rs` | `CachedCholesky`, Gram solver |
| `lss.rs` | LSS (multi-parameter) engine |

PyO3 converts a Rust panic into a Python exception, which requires
`panic = "unwind"` in `Cargo.toml`. Do not change it to `abort`: with
`panic = "abort"`, any panic terminates the Python interpreter.

### Python changes

The Python layer is in `python/mboostrs/`. Key files:

| File | What it does |
|------|-------------|
| `glmboost.py` | `GLMBoostModel` class |
| `gamboost.py` | `GAMBoostModel` class |
| `families.py` | Family wrappers (thin — maps Python names to Rust) |
| `learners.py` | Learner classes (`Bbs`, `Bols`, etc.) |
| `formula.py` | Formula parser (regex-based) |
| `plot.py` | Plotting functions using Altair |
| `sklearn.py` | scikit-learn estimator adapters |

### Adding a new family

1. Implement the `Family` trait in `crates/mboostrs-core/src/families.rs`
2. Map the Rust name in `with_concrete_family!` macro (`crates/mboostrs-py/src/lib.rs`)
3. Add the Python wrapper class in `python/mboostrs/families.py`
4. Export from `python/mboostrs/__init__.py`
5. Add smoke tests + R parity tests

### Adding a new base learner

1. Define the `PreparedLearner` variant in `crates/mboostrs-core/src/learners.rs`
2. Implement `fit_step_ref` and `predict_into`
3. Wire the formula parser in `python/mboostrs/formula.py`
4. Add the learner class in `python/mboostrs/learners.py`

## Testing

```bash
# Rust unit tests
cargo test --workspace --exclude mboostrs-r

# Python tests without R
pytest -m "not r_parity"

# Full R parity suite (needs R + rpy2)
pytest --run-r-parity
```

New features must include smoke tests (`test_smoke.py`) and, where applicable,
R-parity tests. R-parity tests compare mboostrs output with R, either by running
R through rpy2 or against pre-computed R fixtures under `tests/fixtures/`.

## Documentation

All public documentation follows [docs/WRITING_GUIDE.md](docs/WRITING_GUIDE.md).
The book is a MyST project in `docs/book/`. It needs `mystmd`
(`npm install -g mystmd`), the Python dependencies in `requirements.txt`, and
an installed `mboostrs` built from the same source. To build the website
locally:

```bash
cd docs/book
pip install -r requirements.txt
bash build-html.sh   # -> _build/html/index.html
```

`build-html.sh` executes the notebook cells, regenerates derived figures and
also builds the PDF. `docs/book/README.md` describes the preview and
source-check commands.

Usage pages go in `docs/book/usage/`, methodology in `docs/book/methodology/`,
case studies in `docs/book/cases/` or `docs/book/case_studies/`,
implementation details in `docs/book/implementation/`, and benchmarks in
`docs/book/benchmarks/`.

## Notebooks

- Standalone examples in `examples/python/` are committed **without** stored
  cell outputs; CI executes them via `docs/book/execute_examples.sh` and
  `tools/check_examples.py` rejects stale stored outputs.
- Book notebooks under `docs/book/` are executed during the book build
  (`docs/book/build-html.sh`) and checked by `docs/book/check_cells.py` and
  `docs/book/check_docs.py`.

## CI

The CI pipeline runs:

- `cargo fmt --check --all`
- `cargo clippy --workspace --exclude mboostrs-r --all-targets -- -D warnings`
- `cargo test --workspace --exclude mboostrs-r`
- `cargo check -p mboostrs-r` (needs R headers)
- `cargo test --doc -p mboostrs-core`
- `RUSTDOCFLAGS="-D warnings" cargo doc -p mboostrs-core --no-deps`
- `ruff check python/ tests/ tools/ benches/`
- `pytest -m "not r_parity"`
- `python tools/generate_capabilities.py --check`, with the matching `--check`
  runs of `generate_data_cards.py`, `generate_api_reference.py` and
  `generate_rust_stub.py`
- `python tools/check_front_door.py` and the other `tools/check_*.py` scripts
  listed in `.github/workflows/ci.yml`
- `bash docs/book/build-html.sh` and the R package check

Run these locally and fix any failures before you open a pull request.

## Experiment records in issues

Each experiment conducted on an issue is recorded in that issue as:

- **Hypothesis**: what was believed, and why it mattered.
- **Method**: what was done, in enough detail to reproduce it.
- **Measurements**: the numbers, each with a link to the artifact it came from
  (result files, notebooks, fitted models).
- **Conclusion**: what the measurements establish, and what they rule out.

Rules:

- A conclusion that overturns an earlier one must say explicitly what it
  replaces.
- A benchmark number must name the commit, the dependency versions, and
  the artifacts it came from.
- A rejected hypothesis stays in the issue; it is not deleted or edited away.

## License

MIT
