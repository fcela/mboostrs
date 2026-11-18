"""Benchmark artifacts are classified by the commit they record and by the release policy.

The fixture is a three-commit git repository with one artifact per outcome:
current, older than the code it describes, recorded against a commit that is
not an object of the repository (it predates the public history), and
recorded with no commit at all. Each test writes its own
`artifact_policy.json`: a stale `read_by_page` artifact fails the check, a
stale `historical` artifact is a warning unless `--strict-artifacts` is set.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/book"))
import check_docs  # noqa: E402

GIT_IDENTITY = ("-c", "user.name=test", "-c", "user.email=test@example.com")


def _git(root, *args):
    return subprocess.run(["git", *GIT_IDENTITY, *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(root, message):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact(commit):
    """A minimal artifact in the shape `benches/_jobs.py` writes, with the commit set."""
    return json.dumps({"provenance": {"commit": commit}, "results": []}) + "\n"


def _notebook(text):
    return json.dumps({"cells": [{"cell_type": "markdown", "metadata": {}, "source": [text]}],
                       "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path
    book = root / "docs" / "book"
    _git(root, "init", "-q")
    _write(book / "myst.yml", yaml.safe_dump({"project": {"toc": [{"file": "cases/page.ipynb"}]}}))
    _write(book / "mermaid_diagrams.json", json.dumps({"diagrams": {}}))
    _write(root / "Cargo.toml", 'version = "0.1.3"\n')
    _write(root / "pyproject.toml", 'version = "0.1.3"\n')
    _write(root / "python/mboostrs/__init__.py", '__version__ = "0.1.3"\n')
    _write(root / "mboostrs/DESCRIPTION", "Version: 0.1.3\n")
    _write(root / "python/mboostrs/core.py", "v1\n")
    sha1 = _commit(root, "code v1")
    _write(root / "python/mboostrs/core.py", "v2\n")
    _commit(root, "code v2")
    _write(root / "python/mboostrs/core.py", "v3\n")
    _write(book / "cases/bench_old.json", _artifact(sha1))
    _write(book / "cases/bench_none.json", _artifact(None))
    _write(book / "cases/bench_missing.json", _artifact("a" * 40))
    _write(book / "cases/page.ipynb", _notebook(
        "# Page\n\nLoads bench_old.json and bench_none.json.\n"))
    sha3 = _commit(root, "code v3 and artifacts recorded before the refresh")
    _write(book / "cases/bench_current.json", _artifact(sha3))
    _commit(root, "docs-only: record a current artifact")

    monkeypatch.setattr(check_docs, "ROOT", root)
    monkeypatch.setattr(check_docs, "BOOK", book)
    monkeypatch.setattr(check_docs, "MERMAID_MANIFEST", book / "mermaid_diagrams.json")

    real_run = subprocess.run

    def run(command, **kwargs):
        # The two generated-file checks are unrelated; every git call runs for real.
        if len(command) > 1 and Path(str(command[1])).name in {
                "generate_capabilities.py", "generate_custom_css.py", "generate_api_manual.py"}:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr(check_docs.subprocess, "run", run)
    return root


def _policy(root, read_by_page, historical):
    """Write `artifact_policy.json` with each named artifact under cases/."""
    _write(root / "docs/book/artifact_policy.json", json.dumps({
        "read_by_page": {f"cases/{name}": "reason" for name in read_by_page},
        "historical": {f"cases/{name}": "reason" for name in historical},
    }))


ALL_HISTORICAL = ("bench_old.json", "bench_none.json", "bench_missing.json")


def test_each_artifact_is_classified_by_its_commit(repo):
    lines, counts = check_docs.stale_artifact_report()
    assert counts == {
        "up to date": 1,
        "older than code changes": 1,
        "commit not in history": 0,
        "predates public history": 1,
        "no commit recorded": 1,
    }
    assert sorted(key for key, _ in lines) == [
        "cases/bench_missing.json", "cases/bench_none.json", "cases/bench_old.json"]
    listed = "\n".join(line for _, line in lines)
    assert "bench_current.json" not in listed
    assert "bench_old.json: older than code changes (2 commit(s) changed" in listed
    assert "bench_missing.json: predates public history" in listed
    assert "bench_none.json: no commit recorded" in listed


def test_listed_artifact_names_the_pages_that_load_it(repo):
    lines, _ = check_docs.stale_artifact_report()
    old = next(line for _, line in lines if "bench_old.json" in line)
    missing = next(line for _, line in lines if "bench_missing.json" in line)
    assert old.endswith("loaded by: cases/page.ipynb")
    assert missing.endswith("loaded by: no page")


def test_stale_historical_artifacts_are_warnings(repo, capsys):
    _policy(repo, ["bench_current.json"], ALL_HISTORICAL)
    assert check_docs.main([]) == 0
    err = capsys.readouterr().err
    assert "Historical benchmark artifacts, allowed to be stale" in err
    assert "bench_old.json" in err
    assert "bench_current.json" not in err


def test_strict_flag_makes_stale_historical_artifacts_fail(repo, capsys):
    _policy(repo, ["bench_current.json"], ALL_HISTORICAL)
    assert check_docs.main(["--strict-artifacts"]) == 1
    err = capsys.readouterr().err
    assert "2 historical benchmark artifact(s) are stale" in err
    assert "1 benchmark artifact(s) predate the public history" in err


def test_artifact_from_an_unmerged_branch_is_not_in_history(repo):
    """A commit the repository holds but HEAD does not contain is reported as
    not in history, distinct from one the repository has never seen."""
    _git(repo, "checkout", "-q", "-b", "side")
    _write(repo / "python/mboostrs/side.py", "side\n")
    side = _commit(repo, "side branch work")
    _git(repo, "checkout", "-q", "-")
    _write(repo / "docs/book/cases/bench_side.json", _artifact(side))
    _commit(repo, "record an artifact from the side branch")
    lines, counts = check_docs.stale_artifact_report()
    assert counts["commit not in history"] == 1
    assert counts["predates public history"] == 1
    listed = "\n".join(line for _, line in lines)
    assert "bench_side.json: commit not in history" in listed


def test_predating_artifacts_warn_even_when_a_page_reads_them(repo, capsys):
    """Their staleness cannot be measured, so a read_by_page artifact whose
    commit predates the history is a warning, not a failure; the policy still
    classifies it, and --strict-artifacts fails it."""
    _policy(repo, ["bench_current.json", "bench_missing.json"],
            ["bench_old.json", "bench_none.json"])
    assert check_docs.main([]) == 0
    err = capsys.readouterr().err
    assert "bench_missing.json: predates public history" in err
    assert "is not current" not in err
    assert check_docs.main(["--strict-artifacts"]) == 1
    assert "1 benchmark artifact(s) predate the public history" in capsys.readouterr().err


def test_stale_read_by_page_artifact_fails_and_is_named(repo, capsys):
    _policy(repo, ["bench_current.json", "bench_old.json", "bench_none.json"],
            ["bench_missing.json"])
    assert check_docs.main([]) == 1
    err = capsys.readouterr().err
    failures = [line for line in err.splitlines()
                if line.startswith("- benchmark artifact read by a page is not current")]
    assert len(failures) == 2
    assert any("cases/bench_old.json: older than code changes" in line for line in failures)
    assert any("cases/bench_none.json: no commit recorded" in line for line in failures)
    assert "bench_current.json" not in err


def test_missing_policy_fails(repo, capsys):
    assert check_docs.main([]) == 1
    assert "invalid benchmark artifact policy artifact_policy.json" in capsys.readouterr().err


def test_unclassified_and_unknown_artifacts_fail(repo, capsys):
    _write(repo / "docs/book/artifact_policy.json", json.dumps({
        "read_by_page": {"cases/bench_current.json": "reason", "cases/bench_gone.json": "reason"},
        "historical": {f"cases/{name}": "reason" for name in ("bench_old.json", "bench_none.json")},
    }))
    assert check_docs.main([]) == 1
    err = capsys.readouterr().err
    assert "cases/bench_missing.json is not classified in artifact_policy.json" in err
    assert "lists cases/bench_gone.json, which git does not track" in err


def test_artifact_in_both_lists_fails(repo, capsys):
    _write(repo / "docs/book/artifact_policy.json", json.dumps({
        "read_by_page": {"cases/bench_current.json": "reason"},
        "historical": {f"cases/{name}": "reason"
                       for name in (*ALL_HISTORICAL, "bench_current.json")},
    }))
    assert check_docs.main([]) == 1
    assert ("cases/bench_current.json is listed as both read_by_page and historical"
            in capsys.readouterr().err)


def test_historical_artifact_loaded_by_a_code_cell_fails(repo, capsys):
    _policy(repo, ["bench_current.json"], ALL_HISTORICAL)
    notebook = json.loads((repo / "docs/book/cases/page.ipynb").read_text())
    notebook["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [],
                              "execution_count": None,
                              "source": ["json.loads(Path('bench_missing.json').read_text())\n"]})
    _write(repo / "docs/book/cases/page.ipynb", json.dumps(notebook))
    _git(repo, "add", "-A")
    assert check_docs.main([]) == 1
    err = capsys.readouterr().err
    assert ("cases/bench_missing.json is classified historical in artifact_policy.json, "
            "but cases/page.ipynb loads it") in err
    # The page's Markdown cell names bench_old.json and bench_none.json; prose is not a load.
    assert "bench_old.json is classified historical" not in err


def test_committed_policy_classifies_every_book_artifact():
    """The repository's own policy lists each tracked artifact exactly once."""
    real_root = Path(__file__).resolve().parents[1]
    policy = json.loads((real_root / "docs/book/artifact_policy.json").read_text())
    tracked = subprocess.run(["git", "ls-files", "--", "docs/book/**/bench_*.json"],
                             cwd=real_root, check=True, capture_output=True,
                             text=True).stdout.split()
    tracked = {path.removeprefix("docs/book/") for path in tracked}
    assert not set(policy["read_by_page"]) & set(policy["historical"])
    assert set(policy["read_by_page"]) | set(policy["historical"]) == tracked
    assert all(reason.strip() for reason in
               [*policy["read_by_page"].values(), *policy["historical"].values()])


def _artifact_with_rows(commit, components):
    """An artifact whose results carry the given cache-identity components."""
    rows = [{"score": 1.0, "_identity": {"fingerprint": "f", "components": c}}
            if c is not None else {"score": 1.0} for c in components]
    return json.dumps({"provenance": {"commit": {"sha": commit, "dirty": False}},
                       "results": rows}) + "\n"


def test_artifact_measuring_no_mboostrs_code_stays_current(repo):
    """A competitor-only baseline records identities without an `mboostrs`
    component; commits to the mboostrs sources since its run do not make it
    stale. One row that depends on mboostrs, or that records no identity,
    makes the code-change check apply again."""
    old = _git(repo, "rev-list", "--max-parents=0", "HEAD")
    competitor = {"runner_hash": "r", "environment": {}}
    with_mboostrs = {**competitor, "mboostrs": {"version": "0.1.3"}}
    cases = repo / "docs/book/cases"
    _write(cases / "bench_competitor.json", _artifact_with_rows(old, [competitor, competitor]))
    _write(cases / "bench_mixed.json", _artifact_with_rows(old, [competitor, with_mboostrs]))
    _write(cases / "bench_unrecorded.json", _artifact_with_rows(old, [competitor, None]))
    _commit(repo, "record artifacts against the first commit")
    listed = "\n".join(line for _, line in check_docs.stale_artifact_report()[0])
    assert "bench_competitor.json" not in listed
    assert "bench_mixed.json: older than code changes" in listed
    assert "bench_unrecorded.json: older than code changes" in listed
