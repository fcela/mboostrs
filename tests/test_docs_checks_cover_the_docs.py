"""The documentation checks catch structural defects, and only those.

`docs/book/check_docs.py` checks the book's structure: ToC pages,
one H1 per notebook, links and anchors, citation keys, chart titles,
figure captions, names the code must provide, and orphan notebooks. It does not
match patterns in prose. `check_writing.py`, which did, was removed on
2026-09-14: a writer fixing its findings produced near-variants of the flagged
sentences rather than plain ones.
"""

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
sys.path.insert(0, str(BOOK))

from check_docs import (  # noqa: E402
    CITE_KEY_RE,
    H1_RE,
    bibliography_keys,
    markdown_cells,
    prose_without_fences,
    unresolved_citation_keys,
    untitled_chart_cells,
)


def toc_notebooks() -> list[str]:
    config = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    return [p for p in walk(config["project"]["toc"], []) if p.endswith(".ipynb")]


def notebook(page: str) -> dict:
    return json.loads((BOOK / page).read_text())


def test_the_book_has_more_pages_than_cases():
    pages = toc_notebooks()
    assert len(pages) > 80
    assert sum(1 for p in pages if not p.startswith("cases/")) > 50


# ─── one H1, with code fences ignored ──────────────────────────────


def test_a_comment_in_a_fence_is_not_a_heading():
    """`usage/install.ipynb` once counted six H1s: five were shell comments."""
    text = "# Real heading\n\n```bash\n# not a heading\npip install x\n```\n"
    assert len(H1_RE.findall(prose_without_fences(text))) == 1


def test_a_tilde_fence_is_stripped_too():
    text = "# H\n\n~~~python\n# comment\n~~~\n"
    assert len(H1_RE.findall(prose_without_fences(text))) == 1


def test_every_toc_page_has_exactly_one_h1():
    for page in toc_notebooks():
        prose = prose_without_fences("\n".join(markdown_cells(notebook(page))))
        assert len(H1_RE.findall(prose)) == 1, page


# ─── citations ─────────────────────────────────────────────────────


def test_the_bibliography_keys_are_read_from_the_file():
    keys = bibliography_keys()
    assert "buhlmann2003boosting" in keys
    assert "chetty2016inequality" in keys
    assert len(keys) > 20


def test_the_cite_role_yields_its_keys():
    assert CITE_KEY_RE.findall("see {cite:t}`buhlmann2003boosting` and {cite}`a,b`") == [
        "buhlmann2003boosting", "a,b",
    ]


def test_an_unresolved_citation_key_is_reported():
    assert unresolved_citation_keys("{cite}`buhlmann2003boosting,not_a_key`") == ["not_a_key"]


def test_every_citation_in_the_book_resolves():
    for page in toc_notebooks():
        markdown = "\n".join(markdown_cells(notebook(page)))
        assert unresolved_citation_keys(markdown) == [], page


# ─── chart titles ──────────────────────────────────────────────────


def test_a_chart_without_a_title_is_reported():
    nb = {"cells": [
        {"cell_type": "code", "source": ["alt.Chart(df).mark_point()"]},
        {"cell_type": "code", "source": ["alt.Chart(df).properties(title='x')"]},
        {"cell_type": "code", "source": ["base = alt.Chart(df).mark_point()"]},
        {"cell_type": "code", "source": ["chart = alt.Chart(df)\ndisplay(chart)"]},
        {"cell_type": "markdown", "source": ["alt.Chart in prose"]},
    ]}
    assert untitled_chart_cells(nb) == [1, 4]


def test_every_chart_in_the_book_has_a_title():
    for page in toc_notebooks():
        assert untitled_chart_cells(notebook(page)) == [], page


def test_ci_runs_the_code_cell_readability_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "python docs/book/check_cells.py" in workflow


# ─── structure only ────────────────────────────────────────────────


def test_no_checker_matches_phrases_in_prose():
    assert not (BOOK / "check_writing.py").exists()
    source = (BOOK / "check_docs.py").read_text()
    assert "STYLE_ERRORS" not in source and "STYLE_ADVISORIES" not in source


def test_ci_runs_the_structural_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "python docs/book/check_docs.py" in workflow
    assert "check_writing" not in workflow


# ─── orphan notebooks ──────────────────────────────────────────────


def test_no_backup_named_notebook_sits_beside_the_book():
    """`causal_constraint_bak.ipynb` and `_bak2.ipynb` sat next to the
    `causal_constraint.ipynb` the ToC lists: two more answers to the same
    question, with nothing saying which is current."""
    from check_docs import BACKUP_MARKERS

    offenders = [
        p.name for p in (BOOK / "cases").rglob("*.ipynb")
        if any(marker in p.name.lower() for marker in BACKUP_MARKERS)
    ]
    assert offenders == [], offenders


def test_the_orphan_check_accepts_current_draft_names():
    """Retained draft results do not make a notebook a backup copy."""
    from check_docs import orphan_notebook_errors, toc_pages

    config = yaml.safe_load((BOOK / "myst.yml").read_text())
    pages = toc_pages(config["project"]["toc"])
    assert orphan_notebook_errors(pages) == []


def test_the_orphan_check_catches_a_backup_name(tmp_path, monkeypatch):
    """A check that cannot fail is not a check."""
    import check_docs

    fake_book = tmp_path / "book"
    (fake_book / "cases").mkdir(parents=True)
    (fake_book / "cases" / "page_bak.ipynb").write_text(json.dumps({"cells": []}))
    monkeypatch.setattr(check_docs, "BOOK", fake_book)
    problems = check_docs.orphan_notebook_errors([])
    assert any("backup-named" in p for p in problems)


def test_orphan_draft_outputs_are_allowed_and_unchanged(tmp_path, monkeypatch):
    import check_docs

    fake_book = tmp_path / "book"
    (fake_book / "cases").mkdir(parents=True)
    page = fake_book / "cases" / "draft.ipynb"
    page.write_text(json.dumps({
        "cells": [{"cell_type": "code", "source": [], "execution_count": 1, "outputs": [{"x": 1}]}]
    }))
    before = page.read_bytes()
    monkeypatch.setattr(check_docs, "BOOK", fake_book)
    problems = check_docs.orphan_notebook_errors([])
    assert problems == []
    assert page.read_bytes() == before


def test_a_clean_orphan_draft_is_allowed(tmp_path, monkeypatch):
    """A draft may be outside the ToC when its name distinguishes it from live pages."""
    import check_docs

    fake_book = tmp_path / "book"
    (fake_book / "cases").mkdir(parents=True)
    (fake_book / "cases" / "new_study.ipynb").write_text(json.dumps({
        "cells": [{"cell_type": "code", "source": [], "outputs": []}]
    }))
    monkeypatch.setattr(check_docs, "BOOK", fake_book)
    assert check_docs.orphan_notebook_errors([]) == []
