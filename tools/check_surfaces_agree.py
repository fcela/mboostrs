#!/usr/bin/env python3
"""Check that the shipped surfaces describe the same package.

D8 item 6: "The README, landing page, quickstart, API reference, examples, R
help, crate README, capability tables, notebook ZIP, and PDF must agree at one
commit."

Most of that list already has a check: the README's calls are executed
(`check_front_door.py`), the landing page's counts come from `myst.yml`
(`test_book_navigation.py`), the quickstart is walked end to end
(`check_walkthrough.py`), the API reference and the capability tables are
generated with a `--check`, and the examples are measured against the package
(`check_examples.py`).

Three were left, and each was wrong:

* the **notebook ZIP** a reader downloads is written from the source pages, so
  it goes stale silently. It was missing a page the ToC lists, still carried
  one the ToC had dropped, and 46 of its 86 pages differed from the sources;
* **R help** documented five exported objects nowhere, so `?aic_path` failed
  for something the package exports;
* the **crate README** named 18 standard families where the crate defines 21,
  and its base-learner list had drifted from the `prepare_*` entry points.

The PDF is not checked here. It is a build artifact rather than a committed
file, produced by `docs/book/build-pdf.sh` from the same ToC these checks
already cover.

The other direction of the R-help comparison -- documented but not exported --
is deliberate and stays out: `base_learners.Rd` documents `bols`, `bbs` and
the rest as *formula syntax*, saying in as many words that calling them at the
prompt is an error and that upstream `mboost` exports them as functions. That
page is the fix for a real confusion, not a defect.

Run from the repository root:

    python tools/check_surfaces_agree.py
    python tools/check_surfaces_agree.py --show
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
import tempfile
import zipfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
NAMESPACE = ROOT / "mboostrs" / "NAMESPACE"
MAN = ROOT / "mboostrs" / "man"
CRATE_README = ROOT / "crates" / "mboostrs-core" / "README.md"
FAMILIES_RS = ROOT / "crates" / "mboostrs-core" / "src" / "families.rs"

#: Rust family structs the crate README does not have to list, with why. An
#: entry is a statement about the type, not permission to leave the list
#: short.
CRATE_README_EXEMPT = {
    "Family": "the trait, not a family",
}


def toc_pages() -> list[str]:
    config = yaml.safe_load((BOOK / "myst.yml").read_text(encoding="utf-8"))

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    return walk(config["project"]["toc"], [])


def build_notebook_zip(dest: pathlib.Path) -> pathlib.Path:
    """Build the download bundle at `dest` with the book's own generator."""
    spec = importlib.util.spec_from_file_location(
        "generate_notebooks", BOOK / "generate_notebooks.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    return generator.build(dest)


def check_notebook_zip(show: bool,
                       zip_path: pathlib.Path | None = None) -> list[str]:
    """The download must be the book, not a snapshot of an older one.

    The bundle is a build output that `build-html.sh` writes beside the
    rendered site, so with no `zip_path` this builds one from the current
    sources and checks the ToC and declared evidence files. Pass a path to check a
    bundle that already exists -- the one a local `build-html.sh` left in
    `exports/`, or one a workflow is about to publish.
    """
    # Markdown ToC entries (for example the generated API manual) carry no
    # notebook metadata and are not part of the notebook bundle.
    pages = [p for p in toc_pages() if p.endswith(".ipynb")]
    artifacts = []
    for page in pages:
        notebook = json.loads((BOOK / page).read_text())
        for relative in notebook.get("metadata", {}).get("mboostrs_artifacts", []):
            # The same resolution the bundle generator applies: an artifact
            # declared as `../data/x.parquet` from a page in `cases/` is
            # shipped as `data/x.parquet`, not as a literal `cases/../`
            # path, so the expected name must be normalized the same way.
            artifact = ((BOOK / page).parent / relative).resolve()
            if not artifact.is_relative_to(BOOK.resolve()) or not artifact.is_file():
                raise ValueError(f"invalid notebook artifact {relative!r} in {page}")
            artifacts.append(str(artifact.relative_to(BOOK.resolve())))
    expected = set(pages) | set(artifacts)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = (
            build_notebook_zip(pathlib.Path(tmp) / "notebooks.zip")
            if zip_path is None else zip_path
        )
        with zipfile.ZipFile(archive_path) as archive:
            shipped = set(archive.namelist())
            problems = []
            for page in sorted(expected - shipped):
                problems.append(
                    f"the notebook ZIP is missing {page}, required by the ToC "
                    f"or notebook artifact metadata")
            for page in sorted(shipped - expected):
                problems.append(
                    f"the notebook ZIP ships {page}, which is not declared "
                    f"by the ToC or notebook artifact metadata")
            stale = [
                page for page in sorted(expected)
                if page in shipped
                and archive.read(page) != (BOOK / page).read_bytes()
            ]
    if stale:
        problems.append(
            f"{len(stale)} file(s) in the notebook ZIP differ from their "
            f"source, starting with {', '.join(stale[:3])}. Regenerate it "
            f"with docs/book/generate_notebooks.py"
        )
    if show and not problems:
        print(f"  notebook ZIP: {len(pages)} pages, all current")
    return problems


def r_exports() -> set[str]:
    text = NAMESPACE.read_text(encoding="utf-8")
    return {name.strip().strip('"')
            for name in re.findall(r"^export\(([^)]+)\)", text, re.M)}


def r_documented() -> set[str]:
    documented = set()
    for page in MAN.glob("*.Rd"):
        documented |= set(
            re.findall(r"\\alias\{([^}]+)\}", page.read_text(encoding="utf-8")))
    return {name.strip().strip('"') for name in documented}


def check_r_help(show: bool) -> list[str]:
    """Every exported R object needs a help page: `?name` is how an R user
    reads an API, and a missing page is an error at the prompt."""
    missing = sorted(r_exports() - r_documented())
    if show and not missing:
        print(f"  R help: {len(r_exports())} export(s), each documented")
    return [
        f"`{name}` is exported from the R package with no \\alias in "
        f"mboostrs/man, so `?{name}` fails"
        for name in missing
    ]


def crate_families() -> set[str]:
    """The family structs the crate actually defines."""
    text = FAMILIES_RS.read_text(encoding="utf-8")
    implemented = set(re.findall(r"^impl Family for (\w+)", text, re.M))
    return implemented - set(CRATE_README_EXEMPT)


def check_crate_readme(show: bool) -> list[str]:
    text = CRATE_README.read_text(encoding="utf-8")
    problems = []

    defined = crate_families()
    named = {name for name in re.findall(r"`(\w+)`", text) if name in defined}
    for family in sorted(defined - named):
        problems.append(
            f"crates/mboostrs-core/README.md does not name `{family}`, which "
            f"the crate implements `Family` for")

    stated = re.search(r"The (\d+) standard families", text)
    if stated is None:
        problems.append(
            "crates/mboostrs-core/README.md no longer states how many "
            "standard families the crate has")
    elif int(stated.group(1)) != len(defined):
        problems.append(
            f"crates/mboostrs-core/README.md says {stated.group(1)} standard "
            f"families; the crate implements `Family` for {len(defined)}")
    elif show:
        print(f"  crate README: {len(defined)} families, all named")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="print each surface as it passes")
    arguments = parser.parse_args()

    problems = (check_notebook_zip(arguments.show)
                + check_r_help(arguments.show)
                + check_crate_readme(arguments.show))

    if problems:
        print("Surfaces disagree:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Surfaces agree: notebook ZIP, R help and crate README match the "
          "sources they describe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
