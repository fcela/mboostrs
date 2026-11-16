"""Book validation accepts retained results and never rewrites notebooks."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/book"))
import check_docs  # noqa: E402


def notebook():
    return {"cells": [
        {"cell_type": "markdown", "source": ["# Results\n"]},
        {"cell_type": "code", "source": [], "execution_count": 1,
         "outputs": [{"output_type": "display_data", "data": {"text/html": "<table>μ result</table>"}}]},
    ]}


@pytest.fixture
def book(tmp_path, monkeypatch):
    root, book = tmp_path, tmp_path / "book"
    monkeypatch.setattr(check_docs, "BOOK", book)
    monkeypatch.setattr(check_docs, "ROOT", root)
    manifest = book / "mermaid_diagrams.json"
    monkeypatch.setattr(check_docs, "MERMAID_MANIFEST", manifest)
    pages = [book / name for name in ("benchmarks/live.ipynb", "usage/live.ipynb", "cases/draft.ipynb")]
    for page in pages:
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({"diagrams": {}}))
    # The fixture book has no benchmark artifacts, so the release policy lists none.
    (book / "artifact_policy.json").write_text(json.dumps({"read_by_page": {}, "historical": {}}))
    (book / "myst.yml").write_text(yaml.safe_dump({"project": {"toc": [
        {"file": "benchmarks/live.ipynb"}, {"file": "usage/live.ipynb"}]}}))
    for name, text in {
        "Cargo.toml": 'version = "0.1.3"\n',
        "pyproject.toml": 'version = "0.1.3"\n',
        "python/mboostrs/__init__.py": '__version__ = "0.1.3"\n',
        "mboostrs/DESCRIPTION": "Version: 0.1.3\n",
    }.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    real_run = subprocess.run

    def generated_check(command, **kwargs):
        # Only the two unrelated generated-file checks are stubbed; git calls
        # (artifact staleness) run for real, and the notebook/ToC/heading/link/
        # orphan validation below runs unchanged.
        if command[0] == "git":
            return real_run(command, **kwargs)
        assert Path(command[1]).name in {"generate_capabilities.py", "generate_custom_css.py", "generate_api_manual.py"}
        assert "--check" in command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(check_docs.subprocess, "run", generated_check)
    return pages


def test_checker_accepts_stored_results_and_preserves_all_notebook_bytes(book):
    before = {page: page.read_bytes() for page in book}
    assert check_docs.main() == 0
    assert {page: page.read_bytes() for page in book} == before


@pytest.mark.parametrize("heading", ["No heading", "# First\n# Second\n"])
def test_stored_results_do_not_exempt_pages_from_heading_validation(book, heading, capsys):
    page = book[1]
    content = notebook()
    content["cells"][0]["source"] = [heading]
    page.write_text(json.dumps(content))
    before = {path: path.read_bytes() for path in book}
    assert check_docs.main() == 1
    assert "must contain exactly one H1" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in book} == before


def test_missing_page_is_reported_without_stripping_other_pages(book, capsys):
    page = book[1]
    page.unlink()
    before = {path: path.read_bytes() for path in book if path.exists()}
    assert check_docs.main() == 1
    assert "missing ToC page: usage/live.ipynb" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in before} == before
