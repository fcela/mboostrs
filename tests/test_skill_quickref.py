"""The skill's runnable quick reference runs (#291 comment, #306).

The block moved from SKILL.md to topics/quick-reference.md when SKILL.md became
an entry point under 8,000 characters, so the test reads it from there.
"""

import re
import warnings
from pathlib import Path

import pytest

QUICKREF = (Path(__file__).resolve().parents[1] / "docs" / "skills"
            / "analyzing-data-with-mboostrs" / "topics" / "quick-reference.md")


def _quickref_block():
    text = QUICKREF.read_text(encoding="utf-8")
    match = re.search(r"```python\n(.*?)```", text, re.S)
    assert match, "no python block in topics/quick-reference.md"
    return match.group(1)


def test_quickref_block_runs():
    code = _quickref_block()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            exec(compile(code, "SKILL.md:quickref", "exec"), {})
        except AttributeError as exc:
            # insights() on a Polars frame: fixed separately (#305 / PR #310).
            if "'values'" in str(exc):
                pytest.xfail("#305: insights() on Polars")
            raise
