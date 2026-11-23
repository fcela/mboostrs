"""Verify standalone `examples/python/` notebooks and book notebooks match the public API."""

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "python"
sys.path.insert(0, str(ROOT / "tools"))

import mboostrs as mb  # noqa: E402
from check_examples import check  # noqa: E402


def notebooks():
    return sorted(EXAMPLES.glob("*.ipynb"))


def prose(path):
    notebook = json.loads(path.read_text())
    return "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "markdown")


def code(path):
    notebook = json.loads(path.read_text())
    return "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code")


def test_there_are_examples_to_check():
    assert len(notebooks()) == 8


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_each_example_passes_every_check(path):
    assert check(path) == []


def test_every_mb_name_used_anywhere_resolves():
    public = set(mb.__all__)
    unresolved = {}
    for path in notebooks():
        names = set(re.findall(r"\bmb\.(\w+)", code(path) + "\n" + prose(path)))
        missing = sorted(n for n in names if n not in public and not n.startswith("_"))
        if missing:
            unresolved[path.name] = missing
    assert unresolved == {}, unresolved


def test_the_plot_functions_are_reached_where_they_live():
    from mboostrs import plot

    for name in ("plot_cvrisk", "plot_aic", "plot_stabsel"):
        assert not hasattr(mb, name), f"mb.{name} is not top-level"
        assert hasattr(plot, name), f"mb.plot.{name} should exist"


def test_trees_reject_nan_input():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    x[:3] = np.nan
    data = {"x": x, "y": rng.normal(size=60)}
    with pytest.raises(ValueError, match="NaN"):
        mb.gamboost("y ~ btree(x)", data, control=mb.boost_control(5, 0.1))


def test_no_example_stores_outputs():
    offenders = {}
    for path in notebooks():
        notebook = json.loads(path.read_text())
        stored = sum(1 for c in notebook["cells"]
                     if c["cell_type"] == "code" and c.get("outputs"))
        if stored:
            offenders[path.name] = stored
    assert offenders == {}, offenders


def test_the_checker_catches_an_unresolved_name(tmp_path):
    import check_examples

    page = tmp_path / "x.ipynb"
    page.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["mb.plot_cvrisk(c)\n"], "outputs": []},
    ]}))
    problems = check_examples.check(page)
    assert any("not a public name" in p for p in problems)


def test_the_checker_catches_stored_outputs():
    import check_examples

    page = EXAMPLES / "_tmp_stored.ipynb"
    page.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["1\n"], "outputs": [{"a": 1}]},
    ]}))
    try:
        assert any("stored outputs" in p for p in check_examples.check(page))
    finally:
        page.unlink()


def test_stored_outputs_in_a_book_page_are_check_docs_business(tmp_path):
    import check_examples

    page = tmp_path / "book_page.ipynb"
    page.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["1\n"], "outputs": [{"a": 1}]},
    ]}))
    assert not any("stored outputs" in p for p in check_examples.check(page))


def test_the_book_pages_are_checked_not_only_the_examples():
    import check_examples

    paths = [str(p) for p in check_examples.notebooks()]
    assert any("docs/book/methodology" in p for p in paths)
    assert any("examples/python" in p for p in paths)
    assert len(paths) > 80


def test_the_migration_table_calls_names_that_resolve():
    import json as _json

    import mboostrs as _mb

    assert not hasattr(_mb, "diagnostics")
    page = ROOT / "docs" / "book" / "usage" / "r_migration.ipynb"
    text = "\n".join("".join(c["source"])
                      for c in _json.loads(page.read_text())["cells"])
    assert "mb.diagnostics." not in text
    assert "mb.quantile_residuals(model)" in text


def test_ci_runs_the_example_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "tools/check_examples.py" in workflow


def test_the_checker_passes_on_the_committed_examples():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_examples.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stdout
