"""Collect the warnings a script raises, so that a repeated one is shown once.

A pipeline that calls the same routine in a loop gets the same warning again
on every call, with only the counts changed (the fold count, the number of
rows). :func:`collect_warnings` is a context manager that records the warnings
raised in its block, groups the ones with the same wording (every number
replaced by ``#``), and re-issues each group once when the block ends, with the
number of times it was raised.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections import OrderedDict
from contextlib import contextmanager

__all__ = ["collect_warnings"]

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: How many :func:`collect_warnings` blocks are open.
_active = 0


def warning_key(text) -> str:
    """``text`` with every number replaced by ``#``.

    Two warnings with the same key differ only in their numbers, such as a fold
    count or a row count."""
    return _NUMBER.sub("#", str(text))


def collecting_active() -> bool:
    """Whether a :func:`collect_warnings` block is open."""
    return _active > 0


@contextmanager
def collect_warnings():
    """Show each repeated warning once, with a count, when the block ends.

    Use it around a loop or a script that calls mboostrs many times::

        with mb.collect_warnings():
            for seed in range(50):
                result = mb.ci_test(data, "x", "y", given=("z",), seed=seed)

    Every warning raised inside the block is recorded, from mboostrs or from
    any other package. At exit, warnings with the same wording (numbers aside)
    are grouped. A warning that occurred once is issued as it was, with its
    original location. A group of several is issued once, with the wording of
    its first occurrence (so the numbers are those of the first call) and a
    note of how many times it occurred. Outside a block, mboostrs warnings are
    issued as they occur, except that the extrapolation summary of
    :func:`ci_test` is shown once per process.
    """
    global _active
    _active += 1
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        _active -= 1
    _reissue(caught)


def _with_text(warning, text):
    """A copy of ``warning`` (same class and attributes) with new text."""
    copy = warning.__class__(text)
    for name, value in vars(warning).items():
        setattr(copy, name, value)
    return copy


def _reissue(caught, scope="the collect_warnings block", stacklevel=4, keep_location=True):
    """Issue the recorded warnings: one of each wording, with a count when it
    was raised more than once. ``scope`` names the span the count refers to in
    the text of a grouped warning. ``stacklevel`` is the frame a grouped warning
    points at, counted from this function's caller. A warning raised once keeps
    the file and line it was raised at when ``keep_location`` is set, and
    otherwise points at the frame ``stacklevel`` names."""
    groups: OrderedDict = OrderedDict()
    for item in caught:
        groups.setdefault((item.category, warning_key(item.message)), []).append(item)
    for items in groups.values():
        if len(items) == 1:
            item = items[0]
            if keep_location:
                warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
            else:
                warnings.warn(item.message, item.category, stacklevel=stacklevel)
            continue
        first = items[0].message
        text = (f"{first} This warning was raised {len(items)} times in {scope}; "
                "the numbers above are from the first occurrence.")
        warnings.warn(_with_text(first, text), items[0].category, stacklevel=stacklevel)


def once_per_call(fn):
    """Decorate a routine so that each distinct warning raised inside one call is
    issued once for that call, with the number of times it was raised.

    The routine's fits can raise the same boundary warning once per nuisance
    model, repeat and fold. The wording of a warning is compared with its numbers
    replaced, as in :func:`collect_warnings`, so two copies that differ only in a
    count are one warning. The wrapped function keeps its name, docstring and
    signature. A warning from a nested routine is grouped again by the outer
    call, which is harmless."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                return fn(*args, **kwargs)
        finally:
            # stacklevel 3: this wrapper, then the caller of the routine. The
            # location of a warning raised once is the caller's line too, since the
            # recorded location would be this wrapper's.
            _reissue(caught, scope="this call", stacklevel=3, keep_location=False)

    return wrapper
