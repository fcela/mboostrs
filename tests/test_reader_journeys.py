"""The two D8 reader journeys, and the migration table's deviations.

D8 item 1 says a new Python user must be able to install, fit, tune, truncate,
diagnose, and predict from the README and quickstart alone. Item 2 says a
statistically experienced user must be able to choose among mean, quantile,
survival, ranking, and distributional models and find the prediction scale and
the unavailable diagnostics without opening the source. Item 3 says an R user
must find a migration table whose rows either reproduce upstream or name the
deviation at the call.

All three are claims about reachability, and all three were unchecked. The
first journey ran four of its six operations. The migration table mapped seven
objects that compute something other than their upstream namesake without any
row saying so, omitted four more entirely, and documented `coef()` with a key
that raises `KeyError`.

These tests hold the checkers to their contract rather than restating it: a
checker that cannot fail is a checker that says nothing, so each one is shown
failing on a document with the defect reintroduced.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
WALKTHROUGH = ROOT / "tools" / "check_walkthrough.py"
MIGRATION = ROOT / "tools" / "check_migration_table.py"


def run(script, *arguments):
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.fixture(scope="module")
def walkthrough():
    sys.path.insert(0, str(ROOT))
    from tools import check_walkthrough
    return check_walkthrough


@pytest.fixture(scope="module")
def migration():
    sys.path.insert(0, str(ROOT))
    from tools import check_migration_table
    return check_migration_table


# --- D8 item 1 -------------------------------------------------------------

def test_the_six_operations_are_the_ones_d8_names(walkthrough):
    assert [op.name for op in walkthrough.OPERATIONS] == [
        "install", "fit", "tune", "truncate", "diagnose", "predict",
    ]


def test_every_operation_demands_both_a_call_and_an_interpretation(walkthrough):
    """A call the reader can copy is half of it. `fit.residuals()` in a list
    of five one-liners is a call with no way to read the answer."""
    for operation in walkthrough.OPERATIONS:
        assert operation.call and operation.interpretation
        assert operation.conveys, operation.name
        re.compile(operation.call)
        re.compile(operation.interpretation)


def test_the_quickstart_diagnoses_and_shows_what_wrong_looks_like():
    text = "".join(
        "".join(cell["source"])
        for cell in json.loads(
            (BOOK / "usage" / "quickstart.ipynb").read_text())["cells"]
    )
    assert "residual_frame" in text
    assert "bols(z)" in text, "the deliberately wrong fit"
    assert "unavailable_operations" in text


def test_the_new_user_journey_passes():
    result = run(WALKTHROUGH, "--journey", "new-user")
    assert result.returncode == 0, result.stderr


def test_removing_the_diagnosis_fails_the_journey(walkthrough, monkeypatch):
    """The checker earns its place only if it fails when the walkthrough
    stops diagnosing."""
    without = tuple(op for op in walkthrough.OPERATIONS
                    if op.name != "diagnose")
    diagnose = next(op for op in walkthrough.OPERATIONS
                    if op.name == "diagnose")
    broken = type(diagnose)(
        name="diagnose", call=r"\.no_such_diagnostic\(",
        interpretation=diagnose.interpretation, conveys=diagnose.conveys,
    )
    monkeypatch.setattr(walkthrough, "OPERATIONS", without + (broken,))
    problems = walkthrough.check_new_user(show=False)
    assert any("cannot diagnose" in problem for problem in problems)


# --- D8 item 2 -------------------------------------------------------------

def test_the_five_model_kinds_are_the_ones_d8_names(walkthrough):
    assert set(walkthrough.MODEL_KINDS) == {
        "mean", "quantile", "survival", "ranking", "distributional",
    }


def test_each_kind_names_a_family_the_package_actually_has(walkthrough):
    import mboostrs as mb
    known = {row["name"] for row in mb.family_capabilities(as_frame=False)}
    for kind, family in walkthrough.MODEL_KINDS.items():
        assert family in known, f"{kind} -> {family}"


def test_the_experienced_user_journey_passes():
    result = run(WALKTHROUGH, "--journey", "experienced-user")
    assert result.returncode == 0, result.stderr


def test_a_missing_decision_page_fails_the_journey(walkthrough, monkeypatch):
    monkeypatch.setattr(walkthrough, "DECISION_PAGES",
                        walkthrough.DECISION_PAGES + ("usage/absent.ipynb",))
    problems = walkthrough.check_experienced_user(show=False)
    assert any("usage/absent.ipynb" in problem for problem in problems)


# --- D8 item 3 -------------------------------------------------------------

def test_the_migration_check_passes():
    result = run(MIGRATION)
    assert result.returncode == 0, result.stderr


def test_it_reads_the_parity_registry_rather_than_a_hand_list(migration):
    """The deviations come from `_parity_notes`, so a note added there is
    checked without anyone remembering to update the page's checker."""
    from mboostrs._parity_notes import FAMILY_NOTES, LEARNER_NOTES

    found = {key for key, _, _ in migration.deviations()}
    expected = {
        key
        for registry in (FAMILY_NOTES, LEARNER_NOTES)
        for key, notes in registry.items()
        if any(not note.shared_with_upstream for note in notes)
    }
    assert found == expected


def test_a_new_unshared_deviation_fails_until_someone_decides(migration,
                                                              monkeypatch):
    """The keyword map cannot rot quietly: an unmapped deviation is a
    failure, so adding one forces a decision about the migration page."""
    from mboostrs._parity_notes import ParityNote

    note = ParityNote(name="a brand new difference", mechanism="m",
                      consequence="c", shared_with_upstream=False)
    monkeypatch.setattr(migration, "FAMILY_NOTES",
                        {**migration.FAMILY_NOTES, "Gaussian": (note,)})
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'tools');"
         "import check_migration_table as c;"
         "from mboostrs._parity_notes import ParityNote;"
         "c.FAMILY_NOTES['Zzz'] = (ParityNote('a brand new difference',"
         " 'm', 'c', False),);"
         "sys.argv = ['x']; sys.exit(c.main())"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 1
    assert "no keyword for it" in result.stderr


def test_shared_deviations_are_exempt(migration):
    """A difference upstream shares is not a migration hazard: the ported
    script behaves the same in both packages."""
    from mboostrs._parity_notes import FAMILY_NOTES

    shared = [note for note in FAMILY_NOTES["CoxPH"]
              if note.shared_with_upstream]
    assert shared, "CoxPH's partial likelihood is shared with upstream"
    assert all(note.name not in migration.KEYWORDS for note in shared)


@pytest.mark.parametrize("call, key", [
    ("gamboost", "x"),
    ("glmboost", "x"),
])
def test_the_documented_coef_key_is_the_real_one(call, key):
    """The page used to say `fit.coef()["bols(x)"]`, which raises."""
    import numpy as np

    import mboostrs as mb

    rng = np.random.default_rng(0)
    n = 80
    data = {"x": rng.normal(size=n)}
    data["y"] = data["x"] + rng.normal(scale=0.3, size=n)
    formula = "y ~ bols(x)" if call == "gamboost" else "y ~ x"
    fit = getattr(mb, call)(formula, data, control=mb.boost_control(20, 0.1))
    assert key in fit.coef()
    with pytest.raises(KeyError):
        fit.coef()["bols(x)"]


def test_the_concise_table_is_at_the_top_of_the_page():
    """D8 item 3 asks for a *concise* table. The page is 17k characters; a
    reader who needs only the mapping should not have to scroll it."""
    cells = json.loads(
        (BOOK / "usage" / "r_migration.ipynb").read_text())["cells"]
    opening = "".join(cells[0]["source"])
    assert "## A complete script in R and Python" in opening
    rows = [line for line in opening.splitlines() if line.startswith("| ")]
    assert len(rows) >= 9, "the whole workflow, one row per step"
    for step in ("fit", "tune", "truncate", "predict"):
        assert any(f"| {step}" in row for row in rows), step


# --- D0: how much of the namespace the canonical path reaches ----------

FRONT_DOOR = (
    "README.md",
    "docs/book/usage/quickstart.ipynb",
    "docs/book/usage/five_decisions.ipynb",
    "docs/book/usage/interface_map.ipynb",
)

#: How many of the public names the front door reaches today. D0 asks for one
#: canonical workflow that is "simpler to discover"; the namespace is 146 flat
#: names and every one of them is documented somewhere, so the question is not
#: whether a name is written down but whether the path a reader takes reaches
#: it. This is a ratchet: the front door may cover more, never less.
FRONT_DOOR_NAMES = 83


def front_door_text():
    import json

    parts = []
    for name in FRONT_DOOR:
        path = ROOT / name
        if path.suffix == ".ipynb":
            parts.append("\n".join(
                "".join(cell["source"])
                for cell in json.loads(path.read_text())["cells"]))
        else:
            parts.append(path.read_text())
    return "\n".join(parts)


def test_the_canonical_path_reaches_most_of_the_namespace():
    import mboostrs as mb

    text = front_door_text()
    reached = [name for name in sorted(mb.__all__)
               if re.search(rf"\b{re.escape(name)}\b", text)]
    assert len(reached) >= FRONT_DOOR_NAMES, (
        f"the front door reaches {len(reached)} of {len(mb.__all__)} public "
        f"names, down from {FRONT_DOOR_NAMES}. A name that leaves the "
        f"canonical path is reachable only by knowing it is there."
    )


def test_the_ratchet_is_not_stale():
    """A ratchet that has drifted far below the real number stops catching
    anything. If the front door has grown, raise it."""
    import mboostrs as mb

    reached = sum(1 for name in sorted(mb.__all__)
                  if re.search(rf"\b{re.escape(name)}\b", front_door_text()))
    assert reached <= FRONT_DOOR_NAMES + 12, (
        f"the front door now reaches {reached} names; raise FRONT_DOOR_NAMES")


def test_install_and_quickstart_come_before_the_interface_map():
    """D0's other measurable half. The page after install was a matrix of
    every entry point; it is now one worked fit, then the five decisions,
    then the map."""
    import yaml

    config = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    pages = walk(config["project"]["toc"], [])
    order = [pages.index(p) for p in ("usage/install.ipynb",
                                      "usage/quickstart.ipynb",
                                      "usage/five_decisions.ipynb",
                                      "usage/interface_map.ipynb")]
    assert order == sorted(order), pages[:6]
