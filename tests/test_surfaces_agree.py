"""The shipped surfaces describe the same package.

D8 item 6 lists ten surfaces that must agree at one commit. Most already have
a check: the README's calls are executed, the landing page's counts come from
`myst.yml`, the quickstart is walked end to end, the API reference and the
capability tables are generated with a `--check`, and the examples are
measured against the package.

Three were left, and each was wrong. The notebook ZIP a reader downloads was
missing a page the ToC lists, still carried one the ToC had dropped, and 46 of
its 86 pages differed from their sources. R help had no page for five exported
objects. The crate README named 18 standard families where the crate defines
21.

A downloadable ZIP and a set of `.Rd` files are exactly the surfaces nothing
re-reads, which is why they drift and why the check has to run rather than be
remembered.
"""

import json
import pathlib
import re
import subprocess
import sys
import zipfile

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
CHECK = ROOT / "tools" / "check_surfaces_agree.py"


@pytest.fixture(scope="module")
def surfaces():
    sys.path.insert(0, str(ROOT))
    from tools import check_surfaces_agree
    return check_surfaces_agree


def test_the_surfaces_agree():
    result = subprocess.run(
        [sys.executable, str(CHECK)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_the_zip_contains_the_toc_and_declared_evidence(surfaces, tmp_path):
    config = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    # Markdown ToC entries (the generated API manual) are not notebooks and
    # are not in the notebook bundle.
    pages = {p for p in walk(config["project"]["toc"], []) if p.endswith(".ipynb")}
    bundle = surfaces.build_notebook_zip(tmp_path / "notebooks.zip")
    with zipfile.ZipFile(bundle) as archive:
        artifacts = set()
        for page in pages:
            notebook = json.loads((BOOK / page).read_text())
            for name in notebook.get("metadata", {}).get("mboostrs_artifacts", []):
                # Artifacts are declared relative to the page; the bundle
                # ships them relative to the book, so `../data/x` from a
                # page in `cases/` is stored as `data/x`.
                artifacts.add(
                    str(((BOOK / page).parent / name).resolve().relative_to(BOOK.resolve())))
        assert set(archive.namelist()) == pages | artifacts
        for artifact in artifacts:
            assert archive.read(artifact) == (BOOK / artifact).read_bytes()


def test_a_stale_zip_page_is_caught(surfaces, tmp_path):
    """The failure that actually happened: a page edited after the bundle was
    built, so the download is a different book from the site. The bundle is a
    build output now, so the check builds a current one; this pins that it
    still reports a bundle handed to it that has drifted."""
    fresh = surfaces.build_notebook_zip(tmp_path / "fresh.zip")
    stale = tmp_path / "stale.zip"
    with zipfile.ZipFile(fresh) as source, \
            zipfile.ZipFile(stale, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "usage/quickstart.ipynb":
                data = data + b" "
            target.writestr(name, data)
    problems = surfaces.check_notebook_zip(show=False, zip_path=stale)
    assert any("differ from their source" in problem for problem in problems)


def test_every_r_export_has_a_help_page(surfaces):
    """`?name` is how an R user reads an API; a missing page is an error at
    the prompt for something the package exports."""
    assert surfaces.r_exports() <= surfaces.r_documented()


def test_the_learners_stay_documented_as_formula_syntax():
    """The other direction is deliberate. `bols` and `bbs` are not exported,
    because they are formula syntax rather than callable functions, and
    `base_learners.Rd` says so and names the difference from upstream, where
    they are exported functions."""
    page = (ROOT / "mboostrs" / "man" / "base_learners.Rd").read_text()
    for learner in ("bols", "bbs", "bmono", "btree", "buser"):
        assert f"\\alias{{{learner}}}" in page, learner
    assert "formula syntax" in page
    assert "mboost" in page


@pytest.mark.parametrize("page, subject", [
    ("GaussianKnownVarLSS", "known sampling variance"),
    ("ZINBLSS", "zero"),
    ("load_model", "artifact"),
    ("Quantile", "quantile"),
    ("GaussianLSS", "location-scale"),
])
def test_each_repaired_page_describes_its_own_object(page, subject):
    """Four pages documented the wrong object, because a roxygen block
    separated from its function by an inserted definition is filed under
    whatever comes next, silently. `GaussianKnownVarLSS.Rd` held ZINBLSS's
    text; `load_model`'s page was filed under `.PORTABLE_MODEL_CLASSES`;
    `Quantile` and `GaussianLSS` had blocks stranded above a section comment
    and no page of their own at all.

    Nothing about the result looked broken -- each page rendered, and a reader
    of `?GaussianKnownVarLSS` got a coherent description of a different
    family.
    """
    text = (ROOT / "mboostrs" / "man" / f"{page}.Rd").read_text(encoding="utf-8")
    assert f"\\alias{{{page}}}" in text
    title = re.search(r"\\title\{(.*?)\n?\}", text, re.S).group(1)
    assert subject.lower() in title.lower() or subject.lower() in text.lower()


def test_the_crate_readme_names_every_family(surfaces):
    text = surfaces.CRATE_README.read_text(encoding="utf-8")
    for family in surfaces.crate_families():
        assert f"`{family}`" in text, family


def test_the_crate_family_count_is_the_real_one(surfaces):
    text = surfaces.CRATE_README.read_text(encoding="utf-8")
    stated = re.search(r"The (\d+) standard families", text)
    assert stated is not None
    assert int(stated.group(1)) == len(surfaces.crate_families())
