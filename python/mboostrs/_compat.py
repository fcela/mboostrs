"""Transitional shims, and the schedule that says when each one goes.

A shim with no removal date is not a migration, it is a second API. Every
entry in :data:`DEPRECATIONS` names the release that introduced it and the
release that removes it, and :func:`warn_deprecated` puts both in the message
the caller sees, so "when does this stop working?" is answered where the
warning is read rather than in a changelog the caller has to go find.

The warnings are ``FutureWarning``, not ``DeprecationWarning``: Python hides
``DeprecationWarning`` outside ``__main__``, and the callers here are library
users writing analysis code, who should be told the first time rather than
after they upgrade and the call disappears.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

__all__ = ["DEPRECATIONS", "Deprecation", "still_callable", "warn_deprecated"]


@dataclass(frozen=True)
class Deprecation:
    """One transitional form, and when it stops working."""

    #: What the caller wrote, e.g. ``"fit.control()"``.
    old: str
    #: What to write instead, e.g. ``"fit.control"``.
    new: str
    #: Why the two forms existed, in one clause.
    reason: str
    #: Release that introduced the shim.
    since: str
    #: Release that removes it.
    removed_in: str

    def message(self) -> str:
        return (
            f"{self.old} is deprecated since mboostrs {self.since} and will be "
            f"removed in {self.removed_in}. Write {self.new}. {self.reason}"
        )


_ACCESSOR_REASON = (
    "The three model classes disagreed about which of these were methods and "
    "which were properties, so the same expression worked on one fit and "
    "returned a bound method on another. They are all properties now: they "
    "read stored state and take no arguments."
)

DEPRECATIONS: dict[str, Deprecation] = {
    "control-call": Deprecation(
        old="fit.control()",
        new="`fit.control`",
        reason=_ACCESSOR_REASON,
        since="0.2.0",
        removed_in="0.3.0",
    ),
    "offset-call": Deprecation(
        old="fit.offset()",
        new="`fit.offset`",
        reason=_ACCESSOR_REASON,
        since="0.2.0",
        removed_in="0.3.0",
    ),
    "lss-selected-call": Deprecation(
        old="fit.selected(...) on a distributional model",
        new="`fit.selected` for the whole mapping, or "
            "`fit.selected_which(parameter=...)` to filter it",
        reason=_ACCESSOR_REASON,
        since="0.2.0",
        removed_in="0.3.0",
    ),
}


def warn_deprecated(key: str, stacklevel: int = 3) -> None:
    """Emit the ``FutureWarning`` registered under ``key``."""
    warnings.warn(DEPRECATIONS[key].message(), FutureWarning,
                  stacklevel=stacklevel)


_CALLABLE_SUBCLASSES: dict[type, type] = {}


def _callable_subclass(base: type) -> type:
    """A subclass of ``base`` that is also callable.

    Subclassing keeps the value's own type working -- ``isinstance(x, tuple)``,
    tuple unpacking, ``==`` against a plain tuple, ``dict(x)`` -- which a proxy
    object built out of ``__getattr__`` would not.
    """
    existing = _CALLABLE_SUBCLASSES.get(base)
    if existing is not None:
        return existing

    def __call__(self, *args, **kwargs):
        warn_deprecated(self._mboostrs_deprecation)
        return self._mboostrs_call(*args, **kwargs)

    def __reduce_ex__(self, protocol):
        """Serialize the public value, not the transitional call shim.

        The call target is usually a lambda closing over the fitted model and
        is deliberately not durable state.  Returning the underlying public
        value makes ``pickle.dumps(fit.control)`` behave like pickling a
        ``BoostControl`` and avoids exposing a dynamically-created class that
        another process cannot import.
        """
        del protocol
        import numpy as np

        if isinstance(self, np.ndarray):
            return (_restore_array, (np.asarray(self).copy(),
                                     bool(self.flags.writeable)))
        if issubclass(base, tuple) and hasattr(base, "_fields"):
            value = base(*tuple(self))
        else:
            value = base(self)
        return (_identity, (value,))

    # No `__slots__`: a subclass of `tuple` or `float` that declares it cannot
    # take instance attributes, and one of `ndarray` cannot either, so the
    # call target would have nowhere to live. The per-instance dict costs
    # nothing at the scale these accessors are used.
    cls = type(
        f"Callable{base.__name__[:1].upper()}{base.__name__[1:]}",
        (base,),
        {
            "__call__": __call__,
            "__reduce_ex__": __reduce_ex__,
            "_mboostrs_call": None,
            "_mboostrs_deprecation": "",
            "__doc__": (
                f"A {base.__name__} that is still callable, for one release, "
                f"because the accessor that returns it used to be a method."
            ),
        },
    )
    _CALLABLE_SUBCLASSES[base] = cls
    return cls


def still_callable(value, *, call, deprecation: str):
    """``value``, plus a ``__call__`` that warns and delegates to ``call``.

    Used where an accessor became a property but the method form has to keep
    working for a release. ``value`` keeps its own type, so everything a
    caller did with the property form is unaffected; only calling it warns.
    """
    if deprecation not in DEPRECATIONS:
        raise KeyError(f"no deprecation registered under {deprecation!r}")
    cls = _callable_subclass(type(value))
    return _bind(_construct(cls, value), call, deprecation)


def _construct(cls, value):
    """Build a `cls` holding `value`, however that type is built.

    Most types take their own value: `tuple(t)`, `dict(d)`, `float(x)`. A
    NumPy array does not -- `ndarray(arr)` reads its argument as a *shape* --
    so a subclass of one is made with `.view`, which is NumPy's own idiom for
    it and keeps the buffer rather than copying it (including its read-only
    flag).
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.view(cls)
    if isinstance(value, tuple) and hasattr(cls, "_fields"):
        return cls(*value)
    return cls(value)


def _bind(wrapped, call, deprecation: str):
    """Attach the call target to ``wrapped``."""
    wrapped._mboostrs_call = call
    wrapped._mboostrs_deprecation = deprecation
    return wrapped


def _identity(value):
    """Pickle reconstruction helper for an ordinary public value."""
    return value


def _restore_array(value, writeable: bool):
    """Restore an ndarray compatibility value without the callable shim."""
    value = value.copy()
    value.flags.writeable = writeable
    return value
