"""One reading of a data frame, whatever the frame is made of.

The specification assistants each cast every column with
``np.asarray(v, dtype=float)``. That works for a frame of floats and raises
``ValueError: could not convert string to float: 'b'`` for anything else --
which is to say, for most real data. A frame with one string column could not
be classified, could not be scored, and could not be reported on, even for the
numeric columns beside it.

:func:`frame_columns` reads a mapping, a Polars frame or a pandas frame and
returns a :class:`Column` per predictor: the numeric encoding the engine needs,
*and* what the column actually was, because the two decisions that follow --
which learner suits it, and what its levels mean -- cannot be made from the
numbers alone.

Encoding rules, and why each one:

* **Numeric** passes through. Missing values stay `NaN` and are counted.
* **Boolean** becomes 0/1 and is marked ordered. `False < True` is not an
  arbitrary choice of coding, so a boolean is an indicator, not an unordered
  factor -- which is why `bols` on one is right and refusing it is wrong.
* **String / object** becomes factor codes over the sorted distinct values.
  The labels are kept, in code order, so a coefficient can be reported against
  the level it belongs to instead of against an integer.
* **A declared categorical** (`pandas.Categorical`, Polars `Enum` or
  `Categorical`) keeps *its* level order, not the sorted one. Declaring the
  order is the user saying it matters, and re-sorting would discard it.
  ``ordered`` follows the dtype.
* **Date and datetime** become a float axis -- days for a date, seconds for a
  timestamp -- and are marked ordered with ``kind="datetime"``. A time axis is
  a continuous measurement, so a smooth over it is meaningful; the unit is
  recorded so a reported slope can be given per day rather than per
  nanosecond.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Column", "column_names", "frame_columns", "read_column", "take_rows"]


@dataclass(frozen=True)
class Column:
    """One column, as numbers and as what it was."""

    name: str
    #: ``"numeric"``, ``"boolean"``, ``"categorical"`` or ``"datetime"``.
    kind: str
    #: The numeric encoding: float64, `NaN` where the value is missing.
    values: np.ndarray
    #: Level labels in code order, for a categorical column; `None` otherwise.
    levels: tuple | None = None
    #: Whether the values carry an order the model may rely on. True for
    #: numeric, boolean, datetime, and a categorical declared ordered.
    ordered: bool = True
    #: How many values are missing.
    n_missing: int = 0
    #: The source dtype, verbatim, for messages.
    origin: str = ""
    #: For a datetime column, the unit of `values`: ``"days"`` or
    #: ``"seconds"``.
    unit: str | None = None

    @property
    def n_levels(self) -> int:
        """Distinct non-missing values."""
        if self.levels is not None:
            return len(self.levels)
        present = self.values[~np.isnan(self.values)]
        return int(np.unique(present).size)

    def label(self, code) -> object:
        """The level a code stands for, or the code itself for a numeric
        column."""
        if self.levels is None or not np.isfinite(code):
            return code
        index = int(code)
        return self.levels[index] if 0 <= index < len(self.levels) else code


def column_names(data) -> list[str]:
    """Column names, in order, from a mapping or a frame."""
    if hasattr(data, "columns") and not isinstance(data, dict):
        return list(data.columns)
    return list(data)


def _raw(data, name):
    """The column's values, as whatever object the frame holds."""
    if hasattr(data, "columns") and not isinstance(data, dict):
        return data[name]
    return data[name]


def _polars_dtype_name(values) -> str:
    dtype = getattr(values, "dtype", None)
    return "" if dtype is None else str(dtype)


def _categorical_levels(values):
    """Declared levels and orderedness, or ``None`` when there is no dtype
    saying so.

    A declared order is the user stating that the levels mean something in
    that sequence; discarding it and sorting alphabetically would silently
    reverse ``"low" < "medium" < "high"``.
    """
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return None
    name = type(dtype).__name__

    # A Polars `Enum` fixes its levels and their order at construction, which
    # is the same statement pandas makes with `ordered=True`.
    if name == "Enum":
        return tuple(dtype.categories.to_list()), True
    # A Polars `Categorical` fixes no order -- its `categories` are whatever
    # the data happened to contain, in encounter order, and indexing them
    # raises when the column is short. Its levels are read off the data below,
    # sorted, like any other string column.
    if name == "Categorical":
        return None

    # pandas: `dtype.categories` is an Index, and `ordered` says whether the
    # sequence is meant to be read as one.
    categories = getattr(dtype, "categories", None)
    if categories is None:
        return None
    return tuple(categories.tolist()), bool(getattr(dtype, "ordered", False))


def read_column(data, name: str) -> Column:
    """Read one column into its numeric encoding and its description."""
    values = _raw(data, name)
    origin = _polars_dtype_name(values) or type(values).__name__
    declared = _categorical_levels(values)

    if hasattr(values, "to_numpy"):
        try:
            array = values.to_numpy()
        except (TypeError, ValueError):
            array = np.asarray(values.to_list()
                               if hasattr(values, "to_list") else list(values))
    else:
        array = np.asarray(values)

    if declared is not None:
        levels, ordered = declared
        codes = _codes_for(array, levels)
        return Column(name=name, kind="categorical", values=codes,
                      levels=tuple(levels), ordered=ordered,
                      n_missing=int(np.isnan(codes).sum()), origin=origin)

    kind = array.dtype.kind

    if kind == "b":
        # `False < True` is not an arbitrary coding, so a boolean is an
        # indicator rather than an unordered factor.
        numeric = array.astype(np.float64)
        return Column(name=name, kind="boolean", values=numeric,
                      levels=(False, True), ordered=True, origin=origin)

    if kind == "M":
        return _datetime_column(name, array, origin)

    if kind in "iuf":
        numeric = array.astype(np.float64)
        return Column(name=name, kind="numeric", values=numeric,
                      n_missing=int(np.isnan(numeric).sum()), origin=origin)

    if kind in "OUS":
        # An object column of dates, which is what a mapping of
        # `datetime.date` gives before any frame has typed it.
        as_dates = _try_dates(array)
        if as_dates is not None:
            return _datetime_column(name, as_dates, origin)
        present = [v for v in array.tolist() if v is not None
                   and not (isinstance(v, float) and np.isnan(v))]
        levels = tuple(sorted(set(present), key=_sort_key))
        codes = _codes_for(array, levels)
        return Column(name=name, kind="categorical", values=codes,
                      levels=levels, ordered=False,
                      n_missing=int(np.isnan(codes).sum()), origin=origin)

    raise TypeError(
        f"column {name!r} has dtype {array.dtype!r}, which mboostrs cannot "
        f"read as numbers, levels or times. Convert it before passing the "
        f"frame."
    )


def _sort_key(value):
    """Sort levels within a type, so a mixed column does not raise."""
    return (type(value).__name__, str(value))


def _codes_for(array, levels) -> np.ndarray:
    index = {level: float(position) for position, level in enumerate(levels)}
    return np.array(
        [index.get(value, np.nan) for value in array.tolist()],
        dtype=np.float64,
    )


def _try_dates(array):
    """``array`` as datetime64, or ``None`` if it does not hold dates."""
    sample = next((v for v in array.tolist() if v is not None), None)
    if sample is None:
        return None
    import datetime

    if not isinstance(sample, (datetime.date, datetime.datetime, np.datetime64)):
        return None
    try:
        return np.asarray(array.tolist(), dtype="datetime64[ns]")
    except (TypeError, ValueError):
        return None


def _datetime_column(name, array, origin) -> Column:
    """A time axis as a float, with the unit recorded.

    Days for a date, seconds for a timestamp. A slope reported per nanosecond
    is arithmetically correct and useless to read.
    """
    as_days = array.astype("datetime64[D]")
    whole_days = np.all(array == as_days.astype(array.dtype))
    if whole_days:
        numeric = as_days.astype("float64")
        unit = "days"
    else:
        numeric = array.astype("datetime64[s]").astype("float64")
        unit = "seconds"
    numeric = np.where(np.isnat(array), np.nan, numeric)
    return Column(name=name, kind="datetime", values=numeric, ordered=True,
                  n_missing=int(np.isnan(numeric).sum()), origin=origin,
                  unit=unit)


def frame_columns(data, response=None, *, exclude=()) -> list[Column]:
    """Every predictor column, read into :class:`Column`."""
    skip = {response, *exclude} - {None}
    return [read_column(data, name) for name in column_names(data)
            if name not in skip]


def take_rows(data, rows):
    """The rows `rows` selects, in the frame's own type.

    `rows` is a boolean mask or integer positions. A `dict` stays a dict, a
    Polars frame stays a Polars frame, a pandas frame stays a pandas frame.
    Casting everything to a dict of floats -- which is what the specification
    assistants did to subset for a fold -- loses the string columns entirely,
    so the subset could not be refitted.

    Dispatch order matters: Polars is checked first because it is what this
    package returns, and its row selection is `[]`, not `.iloc`. Reaching for
    `.iloc` first raised `AttributeError: 'DataFrame' object has no attribute
    'iloc'` for every Polars frame that took the Python refit path.
    """
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    if isinstance(data, dict):
        return {name: np.asarray(values)[rows] for name, values in data.items()}
    if hasattr(data, "filter") and not hasattr(data, "iloc"):   # polars
        return data[list(rows)]
    return data.iloc[rows]                                      # pandas
