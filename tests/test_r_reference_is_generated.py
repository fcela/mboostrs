"""The R half of the reference, generated from the R package's own files.

D1's remaining item: "linking each entry to a minimal copyable example, and
the R/Rust halves of the reference. The Python surface is generated ... and
the R NAMESPACE/Rd surface is checked by the capability contract but not
rendered into the book."

So an R user reading the book had a migration table and no reference: to see
what `mboostrs::stabsel` takes, they had to install the package and type `?`.
The information existed — `NAMESPACE` says what is exported and `man/*.Rd`
carries the usage and the title — and nothing brought the two into the book.

The page is generated from both, so it cannot drift: a renamed argument
changes the `\\usage` block, which changes the page, and CI fails until it is
regenerated. That is the same contract the Python reference has, and the
reason both exist rather than a hand-written page that was accurate once.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_r_reference.py"
PAGE = ROOT / "docs" / "book" / "appendix" / "r_reference.ipynb"


@pytest.fixture(scope="module")
def reference():
    sys.path.insert(0, str(ROOT))
    from tools import generate_r_reference
    return generate_r_reference


@pytest.fixture(scope="module")
def text():
    return "".join(json.loads(PAGE.read_text())["cells"][0]["source"])


def test_the_committed_page_is_current():
    result = subprocess.run([sys.executable, str(GENERATOR), "--check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_every_exported_name_with_a_help_page_appears(reference, text):
    missing = [entry["name"] for entry in reference.help_pages()
               if entry["name"] not in text]
    assert missing == []


def test_internal_helpers_stay_out(text):
    """A page marked `\\keyword{internal}` is documented for a maintainer,
    not offered to a user."""
    for name in (".matern32", ".parse_brad", ".check_unit_interval"):
        assert name not in text, name


def test_signatures_keep_their_nested_defaults(text):
    """An Rd usage block wraps long calls, and matching to the first `)`
    truncated every default that is itself a call: `gamboost(formula, data,
    family = Gaussian()` lost the rest of its arguments."""
    line = next(line for line in text.splitlines()
                if line.startswith("| `gamboost("))
    assert "family = Gaussian()" in line
    assert "control = boost_control()" in line
    assert line.count("(") == line.count(")")


def test_the_signature_comes_from_the_help_file(reference):
    """Not from a hand-written copy. A drift here is the failure the whole
    generated-surface pattern exists to prevent."""
    entry = next(e for e in reference.help_pages() if e["name"] == "glmboost")
    rd = (ROOT / "mboostrs" / "man" / "glmboost.Rd").read_text()
    for argument in ("formula", "data", "family", "control", "weights"):
        assert argument in entry["usage"], argument
        assert argument in rd, argument


def test_nothing_lands_in_an_ungrouped_bucket(reference, text):
    """The grouping has a fallback, and the fallback being used means either
    the grouping or the export needs a decision. Eighteen entries were in it
    on the first run — the LSS distribution functions, the LLM helpers, and
    four fit accessors — and each got a group rather than a bucket."""
    assert "Everything else" not in text
    groups = {reference.group_of(entry["name"])
              for entry in reference.help_pages()}
    assert "Everything else" not in groups


def test_the_s3_generics_are_listed_once_each(reference, text):
    """A user calls `predict(fit)`, not `predict.mboostrs_gamboost(fit)`.
    Listing methods individually would bury the calls a reader needs."""
    methods = reference.s3_methods()
    assert len(methods) < sum(len(classes) for classes in methods.values())
    for generic in methods:
        assert text.count(f"| `{generic}` |") == 1
    assert "predict" in methods
    assert len(methods["predict"]) > 1, "predict dispatches on several classes"
    assert "| `predict` |" in text


def test_it_points_at_the_narrower_surface(text):
    """An R user arriving here from the Python documentation needs to know
    which four things this frontend does not read."""
    for gap in ("%X%", "by=", "bins=", "inline transforms"):
        assert gap in text, gap
    assert "r_migration" in text


def test_the_page_is_in_the_toc():
    import yaml

    config = yaml.safe_load(
        (ROOT / "docs" / "book" / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    pages = walk(config["project"]["toc"], [])
    assert "appendix/r_reference.ipynb" in pages
    assert pages.index("appendix/r_reference.ipynb") > \
        pages.index("appendix/api_reference.ipynb")


def test_the_check_fails_when_the_page_is_stale(tmp_path, monkeypatch,
                                                reference):
    stale = tmp_path / "r_reference.ipynb"
    stale.write_text("{}")
    monkeypatch.setattr(reference, "OUTPUT", stale)
    monkeypatch.setattr(sys, "argv", ["generate_r_reference.py", "--check"])
    assert reference.main() == 1


def test_ci_runs_the_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "generate_r_reference.py --check" in workflow


@pytest.mark.parametrize("entry", ["glmboost", "gamboostLSS", "cvrisk",
                                   "stabsel", "load_model", "base_learners"])
def test_the_entries_a_migrating_reader_needs_are_present(entry, text):
    assert re.search(rf"`{re.escape(entry)}\(|`{re.escape(entry)}`", text), entry


def test_nested_rd_title_preserves_the_complete_description(reference):
    rd = r"\title{Weibull model with \code{Surv(time, status)} and right censoring}"
    title = reference.clean(reference.rd_block(rd, "title"))
    assert title == "Weibull model with `Surv(time, status)` and right censoring"


def test_nested_rd_link_is_flattened_without_dropping_the_tail(reference):
    rd = r"\title{Default endpoint for \code{\link{llm_describe}} requests}"
    title = reference.clean(reference.rd_block(rd, "title"))
    assert title == "Default endpoint for `llm_describe` requests"
