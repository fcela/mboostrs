"""The first thing a new user copies has to work.

D8 item 4 asks for a script that executes every displayed front-door call and
validates its return type and shape, prose-only calls included. Nothing ran
the README's Python, and two of its nine blocks did not work:

* the scikit-learn example called `.fit(X, y)` with no `X` defined anywhere in
  the README — `NameError: name 'X' is not defined`;
* the classification example fitted `Binomial` against the continuous `y` from
  the opening block — `family 'Binomial' rejected response: y must be in
  {-1, +1}, {0, 1}, or [0, 1] proportions, got -1.33`.

Both are the reader's first ten minutes.

Prose-only calls are included because that is where the equivalent error hid
in the examples: `mb.plot_cvrisk` was offered in a sentence, not a fenced
block, so a checker reading only code found nothing wrong with it for months.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mboostrs as mb  # noqa: E402
from check_front_door import (  # noqa: E402
    describe,
    prose_api_names,
    python_blocks,
    skip_reason,
)

README = (ROOT / "README.md").read_text()


def test_the_readme_still_shows_python():
    assert len(python_blocks(README)) >= 8


def test_every_readme_block_executes():
    """The whole point: run them, do not read them."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_front_door.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "block(s) executed" in result.stdout


def test_every_prose_api_name_resolves():
    public = set(mb.__all__)
    unresolved = sorted(n for n in prose_api_names(README)
                        if n not in public and not n.startswith("_"))
    assert unresolved == []


def test_the_sklearn_block_defines_its_own_x():
    """It called `.fit(X, y)` with no `X` anywhere in the README."""
    blocks = [s for _, s in python_blocks(README) if "GridSearchCV" in s]
    assert blocks, "the scikit-learn example should still be there"
    assert re.search(r"^X = ", blocks[0], re.M), blocks[0]


def test_the_classification_block_uses_a_binary_response():
    """`Binomial` rejects a continuous response by name, which is what the
    README asked it to fit."""
    blocks = [s for _, s in python_blocks(README) if "mb.Binomial()" in s]
    assert blocks
    assert "y ~ x + z" not in blocks[0], "that fits Binomial against continuous y"


def test_binomial_still_rejects_a_continuous_response():
    """The refusal the block used to hit, pinned: if this ever stops raising,
    the README block was not wrong and this test should say so."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    n = 120
    frame = pl.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        mb.glmboost("y ~ x", frame, family=mb.Binomial(),
                    control=mb.boost_control(10, 0.1))


# ─── the checker itself ────────────────────────────────────────────


def test_a_block_can_declare_a_skip_with_a_reason():
    assert skip_reason("# check-front-door: skip — needs R\nlibrary(x)") == "needs R"
    assert skip_reason("fit = mb.glmboost(...)") is None


def test_describe_reports_type_and_shape():
    """A reader needs to predict the return; the check reports what came
    back so a claim can be compared with it."""
    import numpy as np

    assert describe(np.zeros((3, 2))) == "ndarray(3, 2)"
    assert describe([1, 2, 3]) == "list[3]"
    assert describe(7) == "int"


def test_the_checker_fails_on_a_broken_block(tmp_path, monkeypatch):
    """A check that cannot fail is not a check."""
    import check_front_door

    readme = tmp_path / "README.md"
    readme.write_text("```python\nraise RuntimeError('boom')\n```\n")
    monkeypatch.setattr(check_front_door, "README", readme)
    problems = check_front_door.run_blocks(readme.read_text(), show=False)
    assert problems and "block failed" in problems[0]


def test_the_checker_fails_on_a_prose_only_missing_name(tmp_path):
    import check_front_door

    text = "Call `mb.plot_cvrisk(cvr)` to see the folds.\n"
    problems = check_front_door.check_prose_names(text)
    assert any("plot_cvrisk" in p for p in problems)


def test_running_the_readme_leaves_no_files_behind():
    """The serialization example writes `model.json`; the reader's session is
    theirs to litter, this one is not."""
    before = {p.name for p in ROOT.iterdir()}
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_front_door.py")],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert {p.name for p in ROOT.iterdir()} == before


def test_ci_runs_the_front_door_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "tools/check_front_door.py" in workflow
