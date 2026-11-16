"""mboostrs exception hierarchy with machine-readable error attributes.

mboostrs has relied on bare ``ValueError`` and ``KeyError`` instances with
exact message text, which is fragile for callers and opaque for tooling. These
classes subclass the built-in type they replace, so every existing ``except``
clause keeps working, and add attributes a handler can read instead of parsing
the message:

* ``code``    — a short stable identifier of the failure kind
  (``"shape"``, ``"learner_setup"``, ``"bad_response"``, ``"solver"``,
  ``"control"``, ``"bad_family_parameter"``, ``"stabsel"``,
  ``"unseen_level"``, ``"unknown_parameter"``, ``"missing_column"``,
  ``"missing_value"``)
* ``subject`` — the learner, family, parameter, or column the error is about
* ``details`` — a longer machine-oriented description of the failure
* ``rust_variant`` — the name of the Rust ``MboostrsError`` variant that
  produced the error, when it originated in the Rust engine (``None`` for
  errors raised in Python)

This module is a leaf: it imports nothing from the ``mboostrs`` package, so
the Rust extension and the rest of the package can import it without an import
cycle.
"""

from __future__ import annotations


class MboostrsError(Exception):
    """Base class for every mboostrs error, carrying error attributes.

    Subclasses add the built-in exception they replace as one of their bases.
    ``MboostrsError`` alone is never raised; use a subclass so ``except
    ValueError`` and friends keep matching.
    """

    code: str | None = None
    subject: str | None = None
    details: str | None = None
    rust_variant: str | None = None

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        subject: str | None = None,
        details: str | None = None,
        rust_variant: str | None = None,
    ):
        super().__init__(message)
        for name, value in (
            ("code", code),
            ("subject", subject),
            ("details", details),
            ("rust_variant", rust_variant),
        ):
            if value is not None:
                setattr(self, name, value)


class MboostrsValueError(MboostrsError, ValueError):
    """A failed validation: a bad argument, shape, parameter, or response."""


class MboostrsRuntimeError(MboostrsError, RuntimeError):
    """A numerical or solver failure in the Rust engine."""


class MboostrsKeyError(MboostrsError, KeyError):
    """A missing map key, such as a column the model needs but the data lacks."""