"""One validation table for every public parameter, checked before dispatch.

A family or learner constructor is the last place a bad value can be named
against the argument the caller wrote. Past that point the value reaches a
numerical routine, where it becomes a wrong number rather than an error:
``Huber(delta=-1)`` makes the loss decrease with the residual and reports a
negative value as risk; ``Cindex(sigma=-1)`` inverts the smoothing sigmoid;
``PropOdds(n_levels=1)`` and ``Multinomial(n_classes=1)`` describe a
categorical response with one category, which binding code used to repair
silently with ``.max(2)`` and ``.max(3)``.

Each checker names the parameter, the rule it failed, and the value it got,
so the message is actionable without reading the source. They are used by the
Python constructors, the binding dispatch, and artifact deserialization, so a
value that is rejected when typed is also rejected when loaded from a file.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _describe(value) -> str:
    """A short, faithful rendering of a rejected value."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return repr(value)


def check_finite_number(name: str, value, *, allow_none: bool = False):
    """A real, finite scalar, or ``None`` when that is allowed."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a finite number, got None")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"{name} must be a finite number, got {type(value).__name__} "
            f"{_describe(value)}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {_describe(out)}")
    return out


def check_unit_interval(name: str, value, *, allow_none: bool = False):
    """A probability-like scalar strictly inside ``(0, 1)``."""
    out = check_finite_number(name, value, allow_none=allow_none)
    if out is None:
        return None
    if not 0.0 < out < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {_describe(out)}")
    return out


def check_positive(name: str, value, *, allow_none: bool = False):
    """A finite scalar strictly greater than zero."""
    out = check_finite_number(name, value, allow_none=allow_none)
    if out is None:
        return None
    if not out > 0.0:
        raise ValueError(f"{name} must be positive, got {_describe(out)}")
    return out


def check_int_at_least(name: str, value, minimum: int, *, allow_none: bool = False):
    """A whole number no smaller than ``minimum``.

    A float that is exactly integral is accepted and narrowed, because a value
    read back from JSON or from R arrives as a double. A float with a
    fractional part is rejected rather than truncated.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be an integer >= {minimum}, got None")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)) or float(value) != int(float(value)):
            raise ValueError(
                f"{name} must be a whole number >= {minimum}, got {_describe(value)}"
            )
        value = int(value)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer >= {minimum}, got "
            f"{type(value).__name__} {_describe(value)}"
        )
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {out}")
    return out


def check_whole_int(name: str, value, minimum: int) -> int:
    """An integer (Python or NumPy, not bool) no smaller than ``minimum``.

    A float is rejected even when it is integral: ``TypeError`` for any
    non-integer type, ``ValueError`` for an integer below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer >= {minimum}, got "
            f"{type(value).__name__} {_describe(value)}")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {out}")
    return out


def check_name_list(name: str, value) -> list[str]:
    """A non-empty list of distinct, non-empty variable names.

    A bare string is accepted as a one-element list, which is how a caller
    naturally writes a single variable.
    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(
            f"{name} must be a variable name or a list of them, got "
            f"{type(value).__name__}"
        )
    names = list(value)
    if not names:
        raise ValueError(f"{name} must name at least one variable, got an empty list")
    for entry in names:
        if not isinstance(entry, str):
            raise TypeError(
                f"{name} must contain variable names, got "
                f"{type(entry).__name__} {_describe(entry)}"
            )
        if not entry.strip():
            raise ValueError(f"{name} must not contain an empty variable name")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"{name} must be distinct; {', '.join(repr(d) for d in duplicates)} "
            f"appears more than once"
        )
    return names


def check_matrix(name: str, value, *, rows: int | None = None,
                 cols: int | None = None, square: bool = False) -> np.ndarray:
    """A finite two-dimensional array of the required shape.

    Any array-like the caller can write is accepted -- a nested list, a numpy
    array, anything ``np.asarray`` turns into a numeric block -- and is
    converted once here so downstream code has an array, not a maybe-array.
    """
    try:
        out = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a numeric 2-D array-like, got "
            f"{type(value).__name__}"
        ) from exc
    if out.ndim != 2:
        raise ValueError(
            f"{name} must be 2-dimensional, got a {out.ndim}-dimensional array "
            f"with shape {out.shape}"
        )
    if out.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        bad = int(np.argmax(~np.isfinite(out.ravel())))
        row, col = divmod(bad, out.shape[1])
        raise ValueError(
            f"{name} must be finite; {name}[{row}, {col}] is {out.ravel()[bad]}"
        )
    if square and out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be square, got shape {out.shape}")
    if rows is not None and out.shape[0] != rows:
        raise ValueError(
            f"{name} must have {rows} rows, got {out.shape[0]}"
        )
    if cols is not None and out.shape[1] != cols:
        raise ValueError(
            f"{name} must have {cols} columns, got {out.shape[1]}"
        )
    return out


def check_boundary_knots(name: str, value, *, allow_none: bool = True):
    """Two finite, strictly increasing boundary knots.

    A degenerate pair gives a basis with no support, and a reversed pair gives
    one whose knots run backwards; both produce a design that is silently
    wrong rather than an error at fit time.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be two increasing numbers, got None")
    knots = np.asarray(value, dtype=np.float64).ravel()
    if knots.size != 2:
        raise ValueError(
            f"{name} must hold exactly two values (lower, upper), got "
            f"{knots.size}"
        )
    lower, upper = float(knots[0]), float(knots[1])
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f"{name} must be finite, got ({lower}, {upper})")
    if not lower < upper:
        raise ValueError(
            f"{name} must be strictly increasing, got lower={lower!r} and "
            f"upper={upper!r}"
        )
    return (lower, upper)


def check_fit_budget(needed: int, fit_budget, what: str, *, detail: str = "",
                     remedy: str) -> None:
    """Refuse a call whose planned number of model fits exceeds ``fit_budget``.

    ``needed`` is computed before any fit runs, so a refusal costs nothing.
    ``None`` disables the check. The message names the planned count, the
    budget, and the change that makes the call fit.
    """
    if fit_budget is None:
        return
    budget = check_int_at_least("fit_budget", fit_budget, 1, allow_none=False)
    if needed > budget:
        raise ValueError(
            f"{what} needs {needed} fits{detail}, exceeding fit_budget={budget}; "
            f"raise fit_budget to at least {needed}, or {remedy}")


def _model_unseen_levels(model, value):
    """The unseen-level policy of a prediction call: ``value``, or when it is
    ``None`` (the default) the model's own policy.

    A model's policy is ``"error"`` -- refuse a ``brandom``/factor level the
    fit never saw -- unless the model records another: one built by
    ``mb.fit`` predicts an unseen level as the average group (``"average"``)
    and records that in its trail.
    """
    if value is None:
        return getattr(model, "_default_unseen_levels", "error")
    return value


def _model_extrapolation(model, value):
    """The spline extrapolation policy of a prediction call: ``value``, or when
    it is ``None`` (the default) the model's own policy.

    A direct ``gamboost`` / ``gamboostLSS`` model defaults to ``"warn"``
    (preserving R ``mboost`` linear continuation with an ``ExtrapolationWarning``).
    A model returned by ``mb.fit`` defaults to ``"clip"`` (holding spline terms
    flat at their training boundary knots while ``bols`` linear terms still
    continue linearly), preventing single out-of-range rows from blowing up
    automated predictions. Callers can always override by passing an explicit
    ``extrapolation=`` value.
    """
    if value is None:
        return getattr(model, "_default_extrapolation", "warn")
    return value


