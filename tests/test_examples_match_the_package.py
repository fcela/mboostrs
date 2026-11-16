"""The public examples say what the package does.

Eight notebooks in `examples/python/` sat outside every factual check the book
pages get. Measured against the package they document, four statements were
false and one call did not resolve:

* `mb.plot_cvrisk`, `mb.plot_aic` and `mb.plot_stabsel` were offered as
  top-level functions. They live in `mb.plot`, and every result object carries
  `.plot()`.
* `bns` was described as having "no penalty" — twice in a table and once in
  prose. It is ridge-penalised; upstream's is unpenalised, which is a recorded
  parity deviation rather than a shared property.
* `bkernel` was called "Kernel smoothing (Nadaraya-Watson)". It builds an
  explicit kernel design and fits it with a ridge penalty.
* Trees were said to "handle non-linearities, interactions, and missing data
  naturally". Every learner requires finite input; a `NaN` raises
  `Column 'x' contains 4 NaN value(s)`.

Three of those four were in markdown cells, so a checker that read only code
would have found none of them.

Seven of the eight also carried stored outputs from an earlier revision, which
nothing regenerated and nothing compared with the current package.
`execute_examples.sh` runs all eight against the built wheel; that run is the
evidence, not a saved cell.
"""

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
from check_examples import CONTRADICTED, check  # noqa: E402


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


# ─── the specific corrections ──────────────────────────────────────


def test_every_mb_name_used_anywhere_resolves():
    """Prose counts: three of the four errors were in markdown."""
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


def test_bns_is_not_called_unpenalised():
    """It is ridge-penalised, and the package records the deviation."""
    from mboostrs._parity_notes import LEARNER_NOTES

    names = [n.name for n in LEARNER_NOTES["Bns"]]
    assert "bns is ridge-penalized, not unpenalized" in names
    for path in notebooks():
        assert not re.search(r"[Bb]ns\b[^.]{0,90}\bno (difference )?penalt",
                             prose(path)), path.name


def test_bkernel_is_not_called_nadaraya_watson():
    from mboostrs._parity_notes import LEARNER_NOTES

    assert LEARNER_NOTES["Bkernel"], "the deviation should be recorded"
    for path in notebooks():
        assert not re.search(r"[Bb]kernel\b[^.]{0,90}Nadaraya", prose(path)), path.name


def test_trees_are_not_said_to_handle_missing_data():
    """Measured: a NaN raises rather than being handled."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    x[:3] = np.nan
    data = {"x": x, "y": rng.normal(size=60)}
    with pytest.raises(ValueError, match="NaN"):
        mb.gamboost("y ~ btree(x)", data, control=mb.boost_control(5, 0.1))

    for path in notebooks():
        assert not re.search(r"[Tt]rees?\b[^.]{0,90}missing (data|values?)\s+naturally",
                             prose(path)), path.name


# ─── stored outputs ────────────────────────────────────────────────


def test_no_example_stores_outputs():
    offenders = {}
    for path in notebooks():
        notebook = json.loads(path.read_text())
        stored = sum(1 for c in notebook["cells"]
                     if c["cell_type"] == "code" and c.get("outputs"))
        if stored:
            offenders[path.name] = stored
    assert offenders == {}, offenders


# ─── the checker can fail ──────────────────────────────────────────


def test_the_checker_catches_an_unresolved_name(tmp_path, monkeypatch):
    import check_examples

    page = tmp_path / "x.ipynb"
    page.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["mb.plot_cvrisk(c)\n"], "outputs": []},
    ]}))
    problems = check_examples.check(page)
    assert any("not a public name" in p for p in problems)


def test_the_checker_catches_a_contradicted_claim(tmp_path):
    import check_examples

    page = tmp_path / "x.ipynb"
    page.write_text(json.dumps({"cells": [
        {"cell_type": "markdown",
         "source": ["`bns` fits a spline with no penalty at all.\n"]},
    ]}))
    problems = check_examples.check(page)
    assert any("ridge-penalised" in p for p in problems)


def test_the_checker_catches_stored_outputs():
    """Scoped to `examples/python`: the book's stored outputs are the build's
    business, and `check_docs.py` already rules on those."""
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
    """Two checkers ruling on one thing would disagree eventually."""
    import check_examples

    page = tmp_path / "book_page.ipynb"
    page.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["1\n"], "outputs": [{"a": 1}]},
    ]}))
    assert not any("stored outputs" in p for p in check_examples.check(page))


def test_every_contradiction_states_what_the_package_does():
    """A rule that says only "do not write this" leaves the author guessing
    which way to correct it."""
    for pattern, reason in CONTRADICTED:
        assert pattern and len(reason) > 40, pattern


# ─── the book is checked too ───────────────────────────────────────


def test_the_book_pages_are_checked_not_only_the_examples():
    """Restricting this to `examples/python` left
    `methodology/base_learners.ipynb` heading a section "`bns` — unpenalised
    B-spline" directly above a body saying "with an identity (ridge)
    penalty": a reader scanning headings got the opposite of the truth."""
    import check_examples

    paths = [str(p) for p in check_examples.notebooks()]
    assert any("docs/book/methodology" in p for p in paths)
    assert any("examples/python" in p for p in paths)
    assert len(paths) > 80


def test_the_bns_heading_states_the_penalty_it_has():
    import json

    page = ROOT / "docs" / "book" / "methodology" / "base_learners.ipynb"
    text = "\n".join("".join(c["source"])
                      for c in json.loads(page.read_text())["cells"]
                      if c["cell_type"] == "markdown")
    assert "`bns` — unpenalised B-spline" not in text
    assert "`bns` — ridge-penalised B-spline" in text


def test_explaining_a_deviation_is_allowed():
    """A rule that forbids the word makes the correct sentence unwritable.
    Naming upstream in the same breath is the deviation being explained."""
    import check_examples

    page = ROOT / "tests" / "_tmp_explained.ipynb"
    page.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": [
        "Upstream `mboost`'s `bns` is unpenalised; this one is not.\n"]}]}))
    try:
        assert check_examples.check(page) == []
    finally:
        page.unlink()


def test_claiming_it_of_our_own_bns_is_still_caught():
    import check_examples

    page = ROOT / "tests" / "_tmp_claimed.ipynb"
    page.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": [
        "`bns` fits a spline with no penalty at all.\n"]}]}))
    try:
        assert any("ridge-penalised" in p for p in check_examples.check(page))
    finally:
        page.unlink()


def test_a_mathematical_condition_is_not_a_support_claim():
    """"all base learners have identical df" is a condition on a derivation.
    The first version of the support rule flagged it in a gamlss exercise."""
    import check_examples

    page = ROOT / "tests" / "_tmp_maths.ipynb"
    page.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": [
        "when the parameters are orthogonal and all base learners have "
        "identical $df$, the schedules agree.\n"]}]}))
    try:
        assert check_examples.check(page) == []
    finally:
        page.unlink()


def test_a_support_claim_is_still_caught():
    import check_examples

    page = ROOT / "tests" / "_tmp_support.ipynb"
    page.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": [
        "mboostrs supports all base learners from upstream.\n"]}]}))
    try:
        assert any("blackboost" in p for p in check_examples.check(page))
    finally:
        page.unlink()


def test_the_migration_table_calls_names_that_resolve():
    """It offered `mb.diagnostics.quantile_residuals(model)`. The submodule is
    popped from the package namespace on purpose, so an R user following the
    table got an `AttributeError`."""
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
