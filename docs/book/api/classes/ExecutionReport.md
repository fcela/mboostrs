# `ExecutionReport`

Defined in `mboostrs.report`; exported as `mboostrs.ExecutionReport`.


*Class*

```python
ExecutionReport(
    schema_version: int,
    instrumentation: str,
    route: str,
    n_observations: int,
    n_learners: int,
    total_basis_columns: int,
    requested_mstop: int,
    completed_mstop: int,
    effective_threads: int,
    gram_eligible: bool,
    fallback_reasons: tuple[str, ...] = (),
    scalar_learner_count: int = 0,
    block_learner_count: int = 0,
    tree_learner_count: int = 0,
    cholesky_solver_count: int = 0,
    lu_solver_count: int = 0,
    regularized_solver_count: int = 0,
    max_solver_jitter: float | None = None,
    factor_cache_bytes: int = 0,
    retained_learner_bytes: int = 0,
    gram_cache_bytes: int = 0,
    calibration_count: int = 0,
    tie_selection_count: int = 0,
    component_subsample_iterations: int = 0,
    calibration_converged_count: int = 0,
    calibration_limit_count: int = 0,
    calibration_failure_count: int = 0,
    nonfinite_candidate_fallbacks: int = 0,
    setup_ns: int | None = None,
    basis_ns: int | None = None,
    factorization_ns: int | None = None,
    cross_gram_ns: int | None = None,
    iteration_ns: int | None = None,
    reconstruction_ns: int | None = None,
    binding_ns: int | None = None,
    total_ns: int | None = None
) -> None
```

Versioned summary of the numerical route used by a fitted model.

Timing fields are deliberately excluded from equality comparisons so two
otherwise identical fits compare equal across machines and repeated runs.
`total_ns` measures the Rust core fit. `basis_ns` and
`factorization_ns` split learner preparation, while `binding_ns`
measures the remaining frontend conversion; these intervals are exclusive.
Route-specific phases that do not run are `None`.

Source: [`python/mboostrs/report.py` lines 180-307](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L180-L307)

## Members

| Member | Kind | Summary |
|---|---|---|
| `from_mapping` | Method |  |
| `to_dict` | Method | Return a JSON-serializable dictionary. |
| `for_mstop` | Method | Return report metadata adjusted for a sliced model. |
| `format_text` | Method | Render a compact, human-readable view of the stable report fields. |

(executionreport-from_mapping)=
## `ExecutionReport.from_mapping`

*Method* of `ExecutionReport`

```python
ExecutionReport.from_mapping(
    value: collections.abc.Mapping[str, typing.Any]
) -> ExecutionReport
```

*No description in the docstring.*

Source: [`python/mboostrs/report.py` lines 229-233](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L229-L233)

(executionreport-to_dict)=
## `ExecutionReport.to_dict`

*Method* of `ExecutionReport`

```python
ExecutionReport.to_dict(*, include_arrays: bool = False) -> dict[str, Any]
```

Return a JSON-serializable dictionary.

`include_arrays` is accepted for the uniform export signature; the
report holds counts and timings only.

Source: [`python/mboostrs/report.py` lines 235-243](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L235-L243)

(executionreport-for_mstop)=
## `ExecutionReport.for_mstop`

*Method* of `ExecutionReport`

```python
ExecutionReport.for_mstop(completed_mstop: int) -> ExecutionReport
```

Return report metadata adjusted for a sliced model.

Source: [`python/mboostrs/report.py` lines 245-247](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L245-L247)

(executionreport-format_text)=
## `ExecutionReport.format_text`

*Method* of `ExecutionReport`

```python
ExecutionReport.format_text() -> str
```

Render a compact, human-readable view of the stable report fields.

Source: [`python/mboostrs/report.py` lines 249-297](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/report.py#L249-L297)
