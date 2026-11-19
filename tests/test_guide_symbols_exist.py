"""Every symbol the agent guide names must exist on the shipped ``mboostrs`` (#468).

Round-3 D5 read the guide against an older wheel and hit ``AttributeError`` on
``reference_gap``: the guide named a callable the installed release did not
export, because the guide and the package ship on different clocks. The topics'
runnable blocks are executed by ``test_agent_guide_topics.py``, but a name that
appears only in prose -- or anywhere in a worked example -- was never checked.

This module reads the shipped guide (``SKILL.md``, every topic and every worked
example), extracts the callables it presents as mboostrs API, and asserts each
one is importable. A name reachable only through prose is exactly the failure
that motivated the check, so prose counts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs.agent_docs import EXAMPLES, TOPICS
from mboostrs.families import Gaussian

#: An inline code span; a callable written as ``name(...)`` is read from here.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
#: A fenced code block, whatever its language tag.
_FENCED_CODE = re.compile(r"```[a-zA-Z0-9_+.-]*\n(.*?)```", re.S)
#: A call, with an optional ``mb.`` / ``mboostrs.`` receiver. The lookbehind
#: keeps ``result.describe(`` from reading as a top-level ``describe``.
_CALL = re.compile(r"(?<![\w.])(?:(mb|mboostrs)\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\(")
#: The same call, restricted to an explicit module receiver.
_PREFIXED_CALL = re.compile(r"(?<![\w.])(?:mb|mboostrs)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
#: Python builtins the guide happens to spell as calls.
PYTHON_BUILTINS = frozenset({"hasattr", "help", "repr", "round", "sum"})
#: Math and transform spellings used inside formulas or expressions.
MATH_SPELLINGS = frozenset({"exp", "log", "log10", "log1p", "sin", "sqrt", "floor"})
#: Formula term constructors: read from the formula string, never imported.
FORMULA_TERMS = frozenset({"bbs", "bols", "btree", "brandom", "bspatial"})
#: Methods or attributes of a fit / result object, written without a receiver.
RESULT_METHODS = frozenset({
    "btree_partition_table", "btree_splits", "confint", "contrast", "describe",
    "predict", "projection", "select", "table", "to_dict", "to_frame", "varimp",
})
#: Statistics and R-only spellings that are prose, not mboostrs exports.
PROSE_WORDS = frozenset({"AIC", "P", "Surv", "e", "m", "p", "tau"})

#: Tokens the check would otherwise read as missing API. Kept small and split
#: by why each is not a top-level name: a new entry is a claim that the token
#: is genuinely not part of the shipped surface, never a way to silence drift.
_ALLOWED = PYTHON_BUILTINS | MATH_SPELLINGS | FORMULA_TERMS | RESULT_METHODS | PROSE_WORDS


def _guide_files() -> list[Path]:
    """The shipped guide entry point, topics and worked examples."""
    files = [Path(mb.agent_guide(path_only=True))]
    files += [Path(mb.agent_guide(topic, path_only=True)) for topic in TOPICS]
    files += [Path(mb.agent_guide(f"example:{name}", path_only=True))
              for name in EXAMPLES]
    return files


def _referenced(text: str) -> set[tuple[bool, str]]:
    """``(is_prefixed, name)`` for each call the text presents as mboostrs API.

    Inline code spans are read whole: a bare ``name(`` there is the guide's own
    claim that ``name`` is public. Fenced blocks contribute only the
    unambiguous ``mb.name(`` / ``mboostrs.name(`` calls, because a bare name
    inside a block is usually a local variable.
    """
    refs: set[tuple[bool, str]] = set()
    for span in _INLINE_CODE.findall(text):
        for module, name in _CALL.findall(span):
            refs.add((bool(module), name))
    for block in _FENCED_CODE.findall(text):
        refs.update((True, name) for name in _PREFIXED_CALL.findall(block))
    return refs


def _missing(texts: dict[str, str]) -> dict[str, list[str]]:
    """Missing names -> the guide files that name them, empty when in step."""
    missing: dict[str, set[str]] = {}
    for filename, text in texts.items():
        for prefixed, name in _referenced(text):
            if not prefixed and name in _ALLOWED:
                continue
            if not hasattr(mb, name):
                missing.setdefault(name, set()).add(filename)
    return {name: sorted(files) for name, files in sorted(missing.items())}


def test_every_guide_called_symbol_exists_on_the_shipped_package():
    texts = {path.name: path.read_text(encoding="utf-8") for path in _guide_files()}
    missing = _missing(texts)
    assert missing == {}, (
        "the guide names symbols the installed mboostrs does not export: "
        + "; ".join(f"{name} ({', '.join(files)})" for name, files in missing.items())
        + " -- fix the guide to the shipped spelling or record a non-API token "
        "in the documented allowlist."
    )


def test_the_check_would_catch_a_guide_name_that_is_not_shipped():
    """A guard that cannot fail is not a guard."""
    missing = _missing({"guide.md": "Then run `definitely_not_an_export(data)`."})
    assert list(missing) == ["definitely_not_an_export"]


def test_future_api_gates_do_not_hide_unshipped_calls():
    text = ('Run `mb.not_a_shipped_api(data)` when `hasattr(mb, "not_a_shipped_api")`; otherwise use '
            '`grouped_folds(groups)` and `bbs(x, df=4)`.')
    assert _missing({"guide.md": text}) == {"not_a_shipped_api": ["guide.md"]}


def test_reference_gap_is_importable_and_callable_on_a_tiny_frame():
    """The #468 reproduction: the guide's guide-reach symbol must exist here."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.uniform(-2, 2, n)
    g = rng.choice(["a", "b", "c"], n)
    y = np.sin(2 * x1) + 0.8 * (g == "a") + rng.normal(0, 0.3, n)
    data = pl.DataFrame({"y": y, "x1": x1, "g": g})

    result = mb.reference_gap("y ~ bbs(x1, df=4) + bols(g)", data,
                              family=Gaussian(), folds=2, max_mstop=100, top=0)
    assert isinstance(result, mb.ReferenceGapResult)
    assert np.isfinite(result.gap)
