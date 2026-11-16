"""The provenance record: which library, environment and data produced a result.

`provenance()` returns a plain dict with no timestamps, so two calls on the same
installation and the same data return equal records. A `to_dict()` record
carries one of these under `"provenance"`, and the data part of the record is a
SHA-256 digest of the training columns.

The digest reads the columns through a canonical byte layout, so a Polars
frame, a pandas frame and a dict of arrays holding the same values and dtypes
hash to the same value. The layout is fixed per column: its name, its dtype
label, its row count, then its values. Floats are hashed as little-endian
IEEE-754 bytes with NaN and negative zero normalized. Text, and object columns
whose non-missing values are all text, are hashed value by value with a
missing-value flag, so a missing value in a text column is distinct from the
text "None". Missing numeric values hash as NaN. A pandas categorical column is
hashed as its text values, labelled "string".
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import operator
import platform
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from mboostrs._output import json_safe

__all__ = ["provenance"]

_DIGEST_TAG = b"mboostrs-data-fingerprint-v1\n"


def provenance(data: Any = None, *, seed: int | None = None) -> dict[str, Any]:
    """Return build, environment and optional data identity metadata for a result.

    Parameters
    ----------
    data : DataFrame, dict of arrays, optional
        The rows the result was computed on, as a Polars frame, a pandas frame
        or a mapping from column name to array. When given, the record carries
        a ``"data"`` entry with the row count, the column names with their
        dtypes, and a SHA-256 digest of the values. Equal content gives an
        equal digest whichever container holds it, and a changed value gives a
        different digest.
    seed : int, optional
        The random seed the computation used, recorded as given.

    Returns
    -------
    dict
        ``mboostrs_version``, ``build`` (the mapping from :func:`build_info`),
        ``python_version``, ``platform``, ``packages`` (the NumPy and Polars
        versions, and pandas when it is installed), ``seed`` when a seed was
        given, and ``data`` when data was given. The record has no timestamp.

    Notes
    -----
    The digest reads every value once in Python for text columns, so a frame
    with millions of text cells takes seconds to hash. Numeric columns are
    hashed as one byte string each.
    """
    import polars as pl

    import mboostrs
    from mboostrs.build import build_info

    packages: dict[str, Any] = {"numpy": np.__version__, "polars": pl.__version__}
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        packages["pandas"] = pd.__version__

    record: dict[str, Any] = {
        "mboostrs_version": mboostrs.__version__,
        "build": json_safe(dict(build_info())),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    if seed is not None:
        record["seed"] = operator.index(seed)
    if data is not None:
        record["data"] = fingerprint(data)
    return record


def fingerprint(data: Any) -> dict[str, Any]:
    """Row count, column dtypes and SHA-256 digest of a table's columns.

    Accepts a Polars frame, a pandas frame or a mapping from column name to
    array-like values. See :func:`provenance` for the canonical layout.
    """
    columns = _columns(data)
    names = [name for name, _ in columns]
    if len(set(names)) != len(names):
        raise ValueError("data has duplicate column names; rename them so each column "
                         "has a unique name before recording provenance")
    n_rows = None
    blocks = []
    described = []
    for name, values in columns:
        label, body, length = _canonical_column(values)
        if n_rows is None:
            n_rows = length
        elif length != n_rows:
            raise ValueError(f"column {name!r} has {length} rows, expected {n_rows}")
        blocks.append(_pack_text(name) + _pack_text(label) + struct.pack("<Q", length) + body)
        described.append({"name": name, "dtype": label})

    digest = hashlib.sha256()
    digest.update(_DIGEST_TAG)
    digest.update(struct.pack("<QQ", n_rows or 0, len(blocks)))
    for block in blocks:
        digest.update(block)
    return {"n_rows": int(n_rows or 0), "columns": described, "sha256": digest.hexdigest()}


def _columns(data: Any) -> list[tuple[str, Any]]:
    if hasattr(data, "get_columns") and hasattr(data, "schema"):
        return [(series.name, series) for series in data.get_columns()]
    if hasattr(data, "dtypes") and hasattr(data, "columns"):
        return [(str(name), data[name]) for name in data.columns]
    if isinstance(data, Mapping):
        return [(str(name), values) for name, values in data.items()]
    raise TypeError("data must be a Polars frame, a pandas frame or a mapping from "
                    f"column name to array; got {type(data).__name__}")


def _pack_text(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _canonical_column(values: Any) -> tuple[str, bytes, int]:
    """(dtype label, payload bytes, row count) for one column."""
    raw = values.to_numpy() if hasattr(values, "to_numpy") else values
    array = np.asarray(raw)
    if array.ndim != 1:
        raise ValueError(f"each column must be one-dimensional; got shape {array.shape}")
    if array.dtype.kind == "O":
        array = _typed_object_column(array.tolist())
        if array is None:
            items = np.asarray(raw, dtype=object).tolist()
            return "string", _text_payload(items), len(items)
    kind = array.dtype.kind
    if kind == "U":
        items = array.tolist()
        return "string", _text_payload(items), len(items)
    if kind == "f":
        array = np.where(np.isnan(array), np.nan, array) + 0.0
    if kind in "mM":
        payload = np.ascontiguousarray(array.view("<i8")).tobytes()
        return array.dtype.name, payload, len(array)
    if kind not in "biuf":
        raise TypeError(f"unsupported column dtype {array.dtype}")
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array.dtype.name, np.ascontiguousarray(little).tobytes(), len(array)


def _typed_object_column(items: list[Any]) -> np.ndarray | None:
    """The numeric or boolean array an object column's values form, or None
    when the column is text (or holds only missing values).

    Booleans stay boolean only when no value is missing. Numbers stay integer
    only when no value is missing and none is a float; otherwise they are
    float with NaN, which is how Polars and pandas both report a nullable
    integer column converted to NumPy.
    """
    present = [value for value in items if not _is_missing(value)]
    if all(isinstance(value, str) for value in present):
        return None
    if not all(_is_number(value) for value in present):
        raise TypeError("object column mixes text, numbers or booleans; cast it to one "
                        "dtype before recording provenance")
    complete = len(present) == len(items)
    if complete and all(isinstance(value, (bool, np.bool_)) for value in items):
        return np.asarray(items, dtype=bool)
    if complete and all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                        for value in items):
        return np.asarray(items, dtype=np.int64)
    return np.asarray([np.nan if _is_missing(value) else float(value) for value in items],
                      dtype=np.float64)


def _is_number(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value)
    return type(value).__name__ == "NAType"


def _text_payload(values: list[Any]) -> bytes:
    parts = []
    for value in values:
        if _is_missing(value):
            parts.append(b"\x01")
        else:
            encoded = str(value).encode("utf-8")
            parts.append(b"\x00" + struct.pack("<Q", len(encoded)) + encoded)
    return b"".join(parts)
