"""Internal output-formatting and alias-normalization utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_EXTRACT_ALIAS_MAP: dict[str, str] = {
    "coef": "coefficients",
    "coefs": "coefficients",
    "coefficient": "coefficients",
    "x": "design",
    "model_matrix": "design",
    "designs": "design",
    "selection": "selected",
    "w": "weights",
    "residual": "residuals",
    "offsets": "offset",
    "penalties": "penalty",
    "lambdas": "lambda",
    "dfs": "df",
}


def json_safe(value: Any) -> Any:
    """Convert a value to plain JSON-serializable Python types.

    NumPy scalars become their Python equivalents, arrays become lists,
    tuples become lists, and mappings become dicts, recursively. This is the
    single export convention result ``to_dict()`` methods build on: the
    returned record contains no NumPy containers. IEEE infinities and NaN
    remain numerical floats for backward compatibility. Strict JSON excludes
    those constants: use ``mboostrs.dump_result_json(record)`` and
    ``mboostrs.load_result_json`` to distinguish unbounded/undefined floats
    from genuine missing values in a reversible versioned envelope.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def export_result(result: Any, *, include_arrays: bool = False,
                  arrays: tuple[str, ...] = (),
                  extras: dict[str, Any] | None = None,
                  with_provenance: bool = False,
                  training_data: Any = None) -> dict[str, Any]:
    """The shared ``to_dict()`` implementation for dataclass results.

    Copies every ``repr`` field, applying :func:`json_safe` so the record is
    JSON-serializable. Fields named in ``arrays`` are omitted unless
    ``include_arrays`` is true — the convention's gate for large raw arrays
    (fold scores, row evidence, per-observation vectors). Nested values that
    expose their own ``to_dict()`` are folded in as that record. ``extras``
    supplies derived keys that are not dataclass fields (lengths, computed
    optima); the class passes them explicitly.

    ``with_provenance=True`` adds a ``"provenance"`` entry (see
    :func:`mboostrs.provenance`). ``training_data`` is the frame the result
    was computed on; when it is given, the entry includes its fingerprint.
    """
    from dataclasses import fields

    data: dict[str, Any] = {}
    for field in fields(result):
        if not field.repr:
            continue
        if field.name in arrays and not include_arrays:
            continue
        value = getattr(result, field.name)
        if callable(getattr(value, "to_dict", None)):
            value = value.to_dict()
        data[field.name] = json_safe(value)
    if extras:
        data.update(extras)
    if with_provenance:
        data["provenance"] = _provenance_record(training_data)
    return data


def export_model(model: Any, *, training_data: Any = None) -> dict[str, Any]:
    """``to_dict()`` for a fitted model: its ``summary(as_dict=True)`` record,
    plus the ``"provenance"`` entry for the training data it holds."""
    record = json_safe(model.summary(as_dict=True))
    record["provenance"] = _provenance_record(training_data)
    return record


def _provenance_record(training_data: Any) -> dict[str, Any]:
    from mboostrs._provenance import provenance

    return provenance(training_data)


def _polars_frame(records: list[dict[str, Any]]) -> Any:
    """Build a Polars frame without leaking NumPy object scalar types.

    Polars requires one physical type per column. Some inspection tables use a
    scalar for an offset and a coefficient vector for a structured learner; an
    Object column preserves that deliberately heterogeneous public value.
    """
    import polars as pl

    normalized = [
        {key: json_safe(value) for key, value in record.items()}
        for record in records
    ]
    schema_overrides: dict[str, Any] = {}
    keys = list(dict.fromkeys(key for record in normalized for key in record))
    for key in keys:
        values = [record[key] for record in normalized if record.get(key) is not None]
        contains_sequence = [isinstance(value, list) for value in values]
        if any(contains_sequence) and not all(contains_sequence):
            schema_overrides[key] = pl.Object
    columns = {
        key: [record.get(key) for record in normalized]
        for key in keys
    }
    return pl.DataFrame(
        columns,
        schema_overrides=schema_overrides,
        strict=False,
    )


def _normalize_extract_what(what: str) -> str:
    return _EXTRACT_ALIAS_MAP.get(what, what)


def _empty_frame(columns: tuple[str, ...], output: str) -> Any:
    """A frame with the named columns and no rows.

    A list of dicts has no keys to name columns from, so an empty result
    would otherwise come back with no columns at all.
    """
    if output in ("polars", "auto"):
        try:
            import polars as pl
        except ImportError:
            if output == "polars":
                raise
        else:
            return pl.DataFrame(schema={name: pl.Null for name in columns})
    if output in ("pandas", "auto"):
        try:
            import pandas as pd
        except ImportError:
            if output == "pandas":
                raise
            return []
        return pd.DataFrame(columns=list(columns))
    return []


def output_frame(
    records: list[dict[str, Any]],
    output: str = "auto",
    *,
    columns: tuple[str, ...] | None = None,
) -> Any:
    """Convert a list of dicts to the requested output format.

    Parameters
    ----------
    records : list of dict
        Row-oriented data.
    output : {"auto", "pandas", "polars", "records"}
        - ``"auto"``: polars if available, else pandas, else raw list.
        - ``"pandas"``: :class:`pandas.DataFrame`; raises on import failure.
        - ``"polars"``: :class:`polars.DataFrame`; raises on import failure.
        - ``"records"``: always return the raw ``list[dict]``.
    columns : tuple of str, optional
        Column names for an empty ``records``, so the frame still has its
        columns. Ignored when there are records.

    Returns
    -------
    DataFrame or list of dict
    """
    if output == "records":
        return records
    if not records and columns is not None and output in ("auto", "polars", "pandas"):
        return _empty_frame(columns, output)
    if output == "auto":
        try:
            import polars  # noqa: F401

            return _polars_frame(records)
        except ImportError:
            try:
                import pandas as pd

                return pd.DataFrame(records)
            except ImportError:
                return records
    if output == "pandas":
        import pandas as pd

        return pd.DataFrame(records)
    if output == "polars":
        return _polars_frame(records)
    raise ValueError(
        f"output must be 'auto', 'pandas', 'polars', or 'records', got {output!r}"
    )


def _normalize_output(output: str | None, as_frame: bool | None) -> str:
    """Merge ``output=`` and ``as_frame=`` into a single output string.

    * ``as_frame=False`` is equivalent to ``output="records"``.
    * ``as_frame=True`` defaults to ``output="auto"``.
    * If both are supplied, ``output=`` wins, but a deprecation-style preference
      is recorded for ``output=``.
    """
    if output is not None:
        return output
    if as_frame is False:
        return "records"
    return "auto"
