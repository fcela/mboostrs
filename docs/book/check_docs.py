#!/usr/bin/env python3
"""Fail fast on documentation defects that do not require rendering.

The MyST build remains the authority for rendering and cross-references. This
checker catches cheaper source-level failures first: missing or duplicate ToC
pages, notebooks without exactly one H1, broken local
links and anchors, citation keys that `references.bib` does not define, Altair
charts without a title, figures without a caption, Rust items and `mb.` names
the code does not provide, source citations past the end of a file, stale
generated files, and package-version drift.

Benchmark artifacts (`bench_*.json` under docs/book) are checked for staleness
against the commit they record. An artifact whose commit is missing, not an
ancestor of HEAD, or followed by commits that changed `python/mboostrs` or
`crates/mboostrs-core` is listed with the pages that load it.
An artifact whose recorded commit is not an object of this repository at all
was produced before the public history began (the public repository starts
from a single snapshot commit); its staleness cannot be measured here, so it
is listed as a warning, whatever its policy class, until its benchmark is rerun
and records a commit of this history. `--strict-artifacts` turns that warning
into a failure.
`artifact_policy.json` classifies every artifact. A stale `read_by_page`
artifact (a page, notebook or release claim reads a number from it) is a
failure. A stale `historical` artifact (a log of a past run that nothing reads)
is a warning; `--strict-artifacts` turns those warnings into failures. An
unclassified artifact, or a historical one that a notebook loads, is a failure.

Stored notebook outputs are allowed. Validation reads source notebooks without
rewriting their execution counts or saved results.

It checks structure only. Nothing here matches patterns in prose: phrase lists
were removed on 2026-09-14 because a writer fixing their findings produced
near-variants of the flagged sentences rather than plain ones.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")
H1_RE = re.compile(r"^# (?!#)", re.MULTILINE)
GENERATED_TARGETS = {BOOK / "exports/mboostrs.pdf", BOOK / "exports/notebooks.zip"}
MERMAID_MANIFEST = BOOK / "mermaid_diagrams.json"
MERMAID_ID_RE = re.compile(r"mermaid-([0-9a-f]{12})\.svg$")
MERMAID_FONT_MARKER = "--mboostrs-mermaid-font-v1"



INTRO_FAMILY_ROW_RE = re.compile(
    r"\* - Standard families\n\s*- (.*?)(?=\n\* - )", re.DOTALL
)
#: Families the intro's standard-family row deliberately leaves to the row
#: below it, which covers ranking and survival-rank losses.
INTRO_FAMILY_ROW_EXEMPT = {"AUC", "Gehan", "Cindex"}


def _family_shaped(listed: set[str], declared: set[str]) -> set[str]:
    """Words in the row that are meant to be family names.

    The row is prose, so it also contains ordinary capitalised words. Treat a
    word as a family name if it is declared, or if it is a near-miss of one —
    same first four characters — which catches `Gamma` for `GammaReg` without
    flagging `All` or `Python`.
    """
    prefixes = {name[:4] for name in declared}
    return {name for name in listed if name in declared or name[:4] in prefixes}


def intro_family_row_errors(markdown: str) -> list[str]:
    """Check the landing page's standard-family list against `families.py`.

    This row is hand-maintained prose summarising a contract the generated
    appendix already gets right, and it had drifted in both directions: it
    named `Gamma`, which does not exist (the class is `GammaReg`, and in R a
    bare `Gamma` resolves to `stats::Gamma` and fails two layers down with
    "argument is of length zero"), and it omitted `GaussReg` and `HingeLoss`,
    which are public in both frontends. A reader decides here whether the
    package covers their case.
    """
    match = INTRO_FAMILY_ROW_RE.search(markdown)
    if not match:
        return ["intro.ipynb: could not find the 'Standard families' table row to check"]
    # Only the enumeration itself, which ends at the first sentence break. The
    # prose after it may name a family in order to say it does not exist —
    # the row explains that `Gamma` is `GammaReg` — and that is not a claim
    # that the name is available.
    enumeration = match.group(1).split(". ", 1)[0]
    try:
        from mboostrs import families
    except Exception:
        return []
    declared = {
        name for name in getattr(families, "__all__", []) if name[:1].isupper()
    } - INTRO_FAMILY_ROW_EXEMPT
    listed = set(re.findall(r"\b[A-Z][A-Za-z]+\b", enumeration))
    errors = []
    for name in sorted(_family_shaped(listed, declared) - declared):
        errors.append(
            f"intro.ipynb standard-family row names `{name}`, which "
            f"mboostrs.families does not export"
        )
    missing = sorted(declared - listed)
    if missing:
        errors.append(
            "intro.ipynb standard-family row omits "
            + ", ".join(f"`{name}`" for name in missing)
        )
    return errors



SOURCE_CITATION_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:rs|py|R)):(\d+)(?:-(\d+))?`")
#: Where a cited source file may live. A citation is usually a bare basename
#: (`lss.rs:280`), so resolution is by basename across these roots.
SOURCE_ROOTS = ("crates", "python", "mboostrs", "tools", "benches", "tests")


def _source_line_counts() -> dict[str, int]:
    """Longest file with each basename, and each repo-relative path.

    Some basenames are ambiguous — `lib.rs` exists in both binding crates — so
    a bare citation is checked against the longest candidate. That is
    permissive by design: it cannot pin a citation to the right file, but it
    still catches a line number past the end of every file it could mean,
    which is what a citation left behind by a refactor looks like.
    """
    counts: dict[str, int] = {}
    for root in SOURCE_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".rs", ".py", ".R") or not path.is_file():
                continue
            if "target" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            except OSError:
                continue
            relative = str(path.relative_to(ROOT))
            counts[relative] = lines
            counts[path.name] = max(counts.get(path.name, 0), lines)
    return counts


def source_citation_errors(markdown: str, counts: dict[str, int]) -> list[str]:
    """Check `file.rs:line` citations resolve to a real file and a real line.

    Fourteen chapters have no executable content, so nothing in them runs and
    nothing catches drift; the Rust-item and `mb.*` guards are all the
    verification they get. A source citation is the other thing those chapters
    are full of, and a refactor moves line numbers without touching the prose.
    """
    errors = []
    for match in SOURCE_CITATION_RE.finditer(markdown):
        name, start, end = match.group(1), int(match.group(2)), match.group(3)
        limit = counts.get(name)
        if limit is None:
            errors.append(f"cites `{name}`, which is not a source file in this repo")
            continue
        highest = max(start, int(end)) if end else start
        if highest > limit:
            errors.append(
                f"cites `{match.group(0).strip('`')}` but the longest `{name}` "
                f"has {limit} lines"
            )
    return errors


def toc_pages(nodes: list[object]) -> list[Path]:
    pages: list[Path] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if file_name := node.get("file"):
            pages.append(BOOK / str(file_name))
        if pattern := node.get("pattern"):
            pages.extend(sorted(BOOK.glob(str(pattern))))
        children = node.get("children", [])
        if isinstance(children, list):
            pages.extend(toc_pages(children))
    return pages


def markdown_cells(notebook: dict[str, object]) -> list[str]:
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        return []
    result: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        result.append("".join(source) if isinstance(source, list) else str(source))
    return result


def prose_without_fences(markdown: str) -> str:
    """Remove fenced directives/code before checking Markdown headings."""
    lines: list[str] = []
    fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if not fence:
            lines.append(line)
    return "\n".join(lines)


RUST_FENCE_RE = re.compile(r"```rust\n(.*?)```", re.DOTALL)
RUST_ITEM_RE = re.compile(r"\b(?:fn|struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)")
# Names that are illustrative rather than claims about this codebase: standard
# library and dependency types a snippet may legitimately mention.
RUST_ITEM_ALLOWLIST = {"Mat", "MatRef", "Result", "Option", "Vec", "Cholesky", "Lu"}


def rust_items_in_docs(markdown: str) -> set[str]:
    """Rust item names a page presents as belonging to this codebase."""
    names: set[str] = set()
    for block in RUST_FENCE_RE.findall(markdown):
        names.update(RUST_ITEM_RE.findall(block))
    return names - RUST_ITEM_ALLOWLIST


def rust_items_in_source() -> set[str]:
    """Every fn/struct/enum/trait actually declared in the Rust crates."""
    names: set[str] = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        names.update(RUST_ITEM_RE.findall(path.read_text(encoding="utf-8")))
    return names


PY_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
PY_ATTR_RE = re.compile(r"\b(?:mb|mboostrs)\.([A-Za-z_][A-Za-z0-9_]*)")


def python_attrs_in_docs(markdown: str) -> set[str]:
    """`mb.NAME` / `mboostrs.NAME` used inside NON-executed ```python blocks.

    Executed code cells need no check: the docs job runs every one, so a stale
    API call there already fails the build. Fenced ```python blocks inside
    markdown are never executed, which is where a renamed or removed API can sit
    unnoticed — the same gap the Rust item check closes.

    Prose is deliberately exempt. A sentence may name an API in order to say it
    does not exist, as methodology/survival does for `mb.Surv(...)`.
    """
    names: set[str] = set()
    for block in PY_FENCE_RE.findall(markdown):
        names.update(PY_ATTR_RE.findall(block))
    return names


def unknown_python_attrs(names: set[str]) -> set[str]:
    """Filter to those the installed package does not provide.

    Returns nothing when mboostrs is not importable, so a source-only checkout
    can still run the rest of this script.
    """
    try:
        import importlib

        import mboostrs
    except Exception:
        return set()
    unknown = set()
    for name in names:
        if hasattr(mboostrs, name):
            continue
        try:
            importlib.import_module(f"mboostrs.{name}")
        except Exception:
            unknown.add(name)
    return unknown


#: `{cite}`key``, `{cite:p}`a,b``. The keys are what must resolve.
CITE_KEY_RE = re.compile(r"\{cite[:a-z]*\}`([^`]*)`")
_BIB_KEYS: set[str] | None = None


def bibliography_keys() -> set[str]:
    """Every key `references.bib` defines. `myst.yml` declares it as the
    project bibliography, so a key resolves book-wide without a per-page
    `{bibliography}` directive; a key missing from the file renders as a
    dangling reference."""
    global _BIB_KEYS
    if _BIB_KEYS is None:
        text = (BOOK / "references.bib").read_text(encoding="utf-8")
        _BIB_KEYS = set(re.findall(r"^@\w+\s*\{\s*([^,\s]+)", text, re.M))
    return _BIB_KEYS


def unresolved_citation_keys(markdown: str) -> list[str]:
    keys = []
    for group in CITE_KEY_RE.findall(markdown):
        for key in group.split(","):
            if key.strip() and key.strip() not in bibliography_keys():
                keys.append(key.strip())
    return keys


def untitled_chart_cells(notebook: dict[str, object]) -> list[int]:
    """1-based indices of emitted Altair charts with no ``title=``.

    Intermediate layers are often assigned in one short cell and composed in
    the next. An assignment emits no figure and therefore needs no title of its
    own; the final expression or ``display(chart)`` call does.
    """
    found = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        emits_expression = bool(tree.body and isinstance(tree.body[-1], ast.Expr))
        emits_display = "display(" in text
        if ("alt.Chart" in text and (emits_expression or emits_display)
                and "title=" not in text):
            found.append(index)
    return found


def uncaptioned_figures(markdown: str) -> list[str]:
    """Figure directives with no visible caption.

    `:alt:` is written for a screen reader and is invisible to everyone else,
    so a figure carrying only alt text leaves a sighted reader to infer what it
    shows from the surrounding prose. Sixteen figures were in that state.
    """
    missing: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        opening = lines[index].strip()
        if not opening.startswith("```{figure}"):
            index += 1
            continue
        path = opening.removeprefix("```{figure}").strip()
        index += 1
        has_caption = False
        while index < len(lines) and lines[index].strip() != "```":
            line = lines[index]
            if not line.startswith(":") and line.strip():
                has_caption = True
            index += 1
        if not has_caption:
            missing.append(path)
        index += 1
    return missing


def mermaid_figures(markdown: str) -> list[tuple[str, dict[str, str], str]]:
    """Return Mermaid figure paths, directive options, and visible captions."""
    figures: list[tuple[str, dict[str, str], str]] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        opening = lines[index].strip()
        if not opening.startswith("```{figure}") or "mermaid-" not in opening:
            index += 1
            continue
        path = opening.removeprefix("```{figure}").strip()
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != "```":
            body.append(lines[index])
            index += 1
        options: dict[str, str] = {}
        caption_lines: list[str] = []
        for line in body:
            if line.startswith(":") and ":" in line[1:]:
                key, value = line[1:].split(":", 1)
                options[key.strip()] = value.strip()
            elif line.strip() or caption_lines:
                caption_lines.append(line.strip())
        figures.append((path, options, " ".join(caption_lines).strip()))
        index += 1
    return figures


def local_link_target(page: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:", "data:")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    # A leading slash is relative to the site root, which serves the book
    # directory; MyST adds the deployment's base path to such links.
    base = BOOK if path.startswith("/") else page.parent
    resolved = (base / path.lstrip("/")).resolve()
    if not resolved.suffix and not resolved.exists():
        notebook_target = resolved.with_suffix(".ipynb")
        if notebook_target.exists():
            return notebook_target
    return resolved


def heading_slugs(page: Path) -> set[str]:
    """The anchors a link on this page can target.

    Two kinds. MyST slugifies heading text — lower-cased, non-alphanumerics
    collapsed to single hyphens, ends trimmed — so ``### Radial basis surfaces
    with `brad``` is ``#radial-basis-surfaces-with-brad``. And an explicit
    ``(name)=`` line before a heading defines ``#name``.

    Prefer the explicit form for a link from another page: a bare heading slug
    resolves in the HTML build but not the LaTeX one, which reports it as an
    undefined reference and prints ``??`` in the PDF.
    """
    if page.suffix == ".ipynb":
        markdown = "\n\n".join(markdown_cells(json.loads(page.read_text(encoding="utf-8"))))
    else:
        markdown = page.read_text(encoding="utf-8")
    slugs = set()
    for line in prose_without_fences(markdown).splitlines():
        explicit = re.match(r"^\(([\w-]+)\)=\s*$", line.strip())
        if explicit:
            slugs.add(explicit.group(1))
            continue
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*$", line)
        if not match:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", match.group(1).lower()).strip("-")
        if slug:
            slugs.add(slug)
    return slugs


def read_version(path: Path, pattern: str) -> str | None:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


#: Name fragments that mark a notebook as a copy of another one. A book with
#: `x.ipynb` and `x_bak.ipynb` beside it has two answers to the same question
#: and no way to tell which is current; the reader who opens the wrong one is
#: not told.
BACKUP_MARKERS = ("_bak", "_take2", "delete__", " copy", "_copy", "_old")


def orphan_notebook_errors(toc_pages: list[Path]) -> list[str]:
    """Source notebooks beside the book that the ToC does not list.

    Drafts may retain outputs, like registered pages. A backup name such as
    `causal_constraint_bak.ipynb` beside `causal_constraint.ipynb` still leaves
    the reader unable to identify the current page, so those names are refused.
    """
    listed = {page.resolve() for page in toc_pages}
    problems: list[str] = []
    for path in sorted((BOOK / "cases").rglob("*.ipynb")):
        if path.resolve() in listed:
            continue
        relative = path.relative_to(BOOK)
        lowered = path.name.lower()
        if any(marker in lowered for marker in BACKUP_MARKERS):
            problems.append(
                f"backup-named notebook beside the book: {relative}. Delete it "
                f"or give it a name that is not a copy of a live page."
            )
    return problems


#: Source trees whose changes can make a saved benchmark stale.
ARTIFACT_CODE_PATHS = ("python/mboostrs", "crates/mboostrs-core")
ARTIFACT_GLOB = "docs/book/**/bench_*.json"
#: Which artifacts a release needs current (`read_by_page`) and which may be stale (`historical`).
ARTIFACT_POLICY_NAME = "artifact_policy.json"
PAGE_GLOBS = ("docs/book/**/*.ipynb", "docs/book/**/*.md")
ARTIFACT_CLASSES = (
    "older than code changes",
    "commit not in history",
    "predates public history",
    "no commit recorded",
)


def tracked_files(*patterns: str) -> list[str]:
    """Repo-relative paths git tracks under the given pathspecs."""
    result = subprocess.run(
        ["git", "ls-files", "--", *patterns],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


def artifact_commit(path: Path) -> str | None:
    """The commit a saved benchmark records, or None when it records none.

    Two shapes occur. `benches/_jobs.py` writes `provenance.commit` as a
    mapping with `sha` beside `dirty` and `status_digest`; older artifacts
    store the bare sha string, and some record `"sha": null`.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    record = document.get("provenance")
    if not isinstance(record, dict):
        record = document
    commit = record.get("commit")
    if isinstance(commit, dict):
        commit = commit.get("sha")
    if isinstance(commit, str) and commit.strip():
        return commit.strip()
    return None


def artifact_uses_mboostrs(path: Path) -> bool:
    """Whether any result in a saved benchmark depends on the mboostrs build.

    `benches/_jobs.py` records each job's cache identity in `_identity`, and
    adds an `mboostrs` component only for jobs that import mboostrs (the
    runner's `uses_mboostrs`). An artifact whose every result carries an
    identity without that component, such as a competitor-only baseline,
    measures no mboostrs code, so mboostrs commits cannot make it stale. A
    result without a recorded identity counts as depending on mboostrs.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    results = document.get("results") if isinstance(document, dict) else None
    if not isinstance(results, list) or not results:
        return True
    for row in results:
        identity = row.get("_identity") if isinstance(row, dict) else None
        components = identity.get("components") if isinstance(identity, dict) else None
        if not isinstance(components, dict) or "mboostrs" in components:
            return True
    return False


def artifact_status(sha: str | None, uses_mboostrs: bool = True) -> tuple[str, str] | None:
    """`(class, detail)` for an artifact that is stale, None when it is current.

    Each git call runs once per artifact. A commit that is not an ancestor of
    HEAD is reported as such, not as stale, because the numbers may come from
    a branch that was never merged here. With `uses_mboostrs` false, changes
    to the mboostrs sources do not make the artifact stale.
    """
    if sha is None:
        return ("no commit recorded", "")
    if not re.fullmatch(r"[0-9a-f]{4,40}", sha):
        return ("commit not in history", f"malformed commit {sha!r}")
    known = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if known.returncode != 0:
        return (
            "predates public history",
            f"{sha[:12]} is not a commit of this repository; rerun the benchmark "
            "to record one",
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if ancestor.returncode != 0:
        return ("commit not in history", f"{sha[:12]} is not an ancestor of HEAD")
    if not uses_mboostrs:
        return None
    log = subprocess.run(
        ["git", "log", "--oneline", f"{sha}..HEAD", "--", *ARTIFACT_CODE_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    changes = [line for line in log.stdout.splitlines() if line.strip()]
    if not changes:
        return None
    return (
        "older than code changes",
        f"{len(changes)} commit(s) changed {' and '.join(ARTIFACT_CODE_PATHS)} "
        f"since {sha[:12]}",
    )


def stale_artifact_report() -> tuple[list[tuple[str, str]], dict[str, int]]:
    """`(artifact, line)` for each stale or unprovenanced artifact, and per-class counts.

    `artifact` is the path relative to docs/book, the key `artifact_policy.json`
    uses. Each line names the book pages that load the artifact, found by
    searching the tracked notebooks and Markdown pages for its file name.
    """
    counts = {label: 0 for label in ("up to date", *ARTIFACT_CLASSES)}
    pages = tracked_files(*PAGE_GLOBS)
    page_text: dict[str, str] = {}
    lines: list[tuple[str, str]] = []
    for relative in tracked_files(ARTIFACT_GLOB):
        path = ROOT / relative
        status = artifact_status(artifact_commit(path), artifact_uses_mboostrs(path))
        if status is None:
            counts["up to date"] += 1
            continue
        label, detail = status
        counts[label] += 1
        loaded_by = []
        for page in pages:
            if page not in page_text:
                page_text[page] = (ROOT / page).read_text(encoding="utf-8", errors="replace")
            if path.name in page_text[page]:
                loaded_by.append(str(Path(page).relative_to(BOOK.relative_to(ROOT))))
        described = f"{label} ({detail})" if detail else label
        key = str(path.relative_to(BOOK))
        lines.append((
            key,
            f"{key}: {described}; loaded by: "
            + (", ".join(loaded_by) if loaded_by else "no page"),
        ))
    return lines, counts


def load_artifact_policy() -> tuple[dict[str, str], dict[str, str], list[str]]:
    """The `read_by_page` and `historical` maps of `artifact_policy.json`, and its defects.

    A defect is a missing or malformed file, an artifact listed in both maps, a
    listed artifact git does not track, or a tracked artifact listed in neither.
    """
    policy_path = BOOK / ARTIFACT_POLICY_NAME
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        read_by_page = dict(policy["read_by_page"])
        historical = dict(policy["historical"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {}, {}, [f"invalid benchmark artifact policy {ARTIFACT_POLICY_NAME}: {exc}"]
    problems: list[str] = []
    tracked = {str((ROOT / relative).relative_to(BOOK)) for relative in tracked_files(ARTIFACT_GLOB)}
    for name in sorted(set(read_by_page) & set(historical)):
        problems.append(f"{name} is listed as both read_by_page and historical in {ARTIFACT_POLICY_NAME}")
    for name in sorted((set(read_by_page) | set(historical)) - tracked):
        problems.append(f"{ARTIFACT_POLICY_NAME} lists {name}, which git does not track under docs/book")
    for name in sorted(tracked - set(read_by_page) - set(historical)):
        problems.append(
            f"{name} is not classified in {ARTIFACT_POLICY_NAME}; add it to read_by_page "
            f"if a page, notebook or release claim reads a number from it, else to historical"
        )
    return read_by_page, historical, problems


def historical_artifacts_read_by_notebooks(historical: dict[str, str]) -> list[str]:
    """Historical artifacts that a notebook code cell names or its metadata declares.

    A notebook that loads an artifact reads numbers from it, so the artifact
    belongs in `read_by_page`. Prose mentions are not counted: a page may name
    the record of a superseded run without reading it.
    """
    problems: list[str] = []
    names = {Path(key).name: key for key in historical}
    for page in tracked_files("docs/book/**/*.ipynb"):
        try:
            notebook = json.loads((ROOT / page).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        declared = " ".join(str(item) for item in notebook.get("metadata", {}).get("mboostrs_artifacts", []))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        for name, key in sorted(names.items()):
            if name in code or name in declared:
                problems.append(
                    f"{key} is classified historical in {ARTIFACT_POLICY_NAME}, but "
                    f"{Path(page).relative_to(BOOK.relative_to(ROOT))} loads it; move it to read_by_page"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="also fail when a historical benchmark artifact is stale or records no "
             "commit (stale read_by_page artifacts always fail)",
    )
    arguments = parser.parse_args([] if argv is None else argv)
    errors: list[str] = []
    declared_rust_items = rust_items_in_source()
    source_line_counts = _source_line_counts()
    try:
        mermaid_manifest = json.loads(MERMAID_MANIFEST.read_text(encoding="utf-8"))
        mermaid_diagrams = mermaid_manifest["diagrams"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(f"invalid Mermaid manifest: {exc}")
        mermaid_diagrams = {}
    referenced_mermaid: set[str] = set()
    capability_check = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_capabilities.py"),
            "--check",
            "--skip-node-validation",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if capability_check.returncode:
        errors.append("generated capability contract is stale: " + capability_check.stderr.strip())
    manual_check = subprocess.run(
        [sys.executable, str(ROOT / "tools/generate_api_manual.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if manual_check.returncode:
        errors.append("generated Python API manual is stale: "
                      + (manual_check.stdout + manual_check.stderr).strip())
    css_check = subprocess.run(
        [sys.executable, str(BOOK / "generate_custom_css.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if css_check.returncode:
        errors.append("custom.css is stale: " + css_check.stderr.strip())
    config = yaml.safe_load((BOOK / "myst.yml").read_text(encoding="utf-8"))
    pages = toc_pages(config.get("project", {}).get("toc", []))
    if not pages:
        errors.append("myst.yml contains no notebook pages")
    duplicates = sorted({str(page.relative_to(BOOK)) for page in pages if pages.count(page) > 1})
    if duplicates:
        errors.append(f"duplicate ToC pages: {', '.join(duplicates)}")

    for page in pages:
        relative = page.relative_to(BOOK)
        if not page.is_file():
            errors.append(f"missing ToC page: {relative}")
            continue
        try:
            text = page.read_text(encoding="utf-8")
            # Markdown pages (the generated API manual) are checked as a
            # notebook with one markdown cell.
            notebook = (
                {"cells": [{"cell_type": "markdown", "source": text}]}
                if page.suffix == ".md" else json.loads(text)
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid notebook {relative}: {exc}")
            continue

        markdown = "\n".join(markdown_cells(notebook))

        # A Rust snippet that names an item this codebase does not declare is
        # describing something that does not exist. linear_algebra.ipynb carried
        # a `try_factorize` calling `g.llt()` and a `struct GramSolver` with a
        # `recompute` method, none of which were ever real.
        for item in sorted(rust_items_in_docs(markdown) - declared_rust_items):
            errors.append(
                f"{relative} shows Rust item `{item}`, which is not declared anywhere in crates/"
            )

        for attr in sorted(unknown_python_attrs(python_attrs_in_docs(markdown))):
            errors.append(
                f"{relative} shows `mb.{attr}` in a python block, "
                f"which the installed mboostrs does not provide"
            )

        if relative.as_posix() == "intro.ipynb":
            errors.extend(intro_family_row_errors(markdown))

        for problem in source_citation_errors(markdown, source_line_counts):
            errors.append(f"{relative} {problem}")

        for key in unresolved_citation_keys(markdown):
            errors.append(f"{relative} cites `{key}`, which references.bib does not define")

        for cell_index in untitled_chart_cells(notebook):
            errors.append(f"chart without a title in {relative}, cell {cell_index}")

        if "```{mermaid}" in markdown or "```mermaid" in markdown:
            errors.append(
                f"unconverted Mermaid block in {relative}; run render_mermaid.py"
            )
        for figure_path, options, caption in mermaid_figures(markdown):
            match = MERMAID_ID_RE.search(figure_path)
            if match is None:
                errors.append(f"malformed Mermaid figure path in {relative}: {figure_path}")
                continue
            diagram_id = match.group(1)
            referenced_mermaid.add(diagram_id)
            diagram = mermaid_diagrams.get(diagram_id)
            if not isinstance(diagram, dict):
                errors.append(f"Mermaid figure {diagram_id} in {relative} is absent from the manifest")
                continue
            expected_alt = diagram.get("alt")
            expected_caption = diagram.get("caption")
            if options.get("alt") != expected_alt:
                errors.append(f"Mermaid alt text differs from manifest in {relative}: {diagram_id}")
            if caption != expected_caption:
                errors.append(f"Mermaid caption differs from manifest in {relative}: {diagram_id}")
            if caption.startswith(("flowchart", "sequenceDiagram")):
                errors.append(f"raw Mermaid source used as caption in {relative}: {diagram_id}")
        for path in uncaptioned_figures(markdown):
            errors.append(
                f"figure without a caption in {relative}: {path} — `:alt:` is "
                f"the screen-reader description, not the caption"
            )
        prose = prose_without_fences(markdown)
        h1_count = len(H1_RE.findall(prose))
        if h1_count != 1:
            errors.append(f"{relative} must contain exactly one H1; found {h1_count}")
        for raw_target in LINK_RE.findall(markdown):
            target = local_link_target(page, raw_target)
            if target is None:
                continue
            if target not in GENERATED_TARGETS and not target.exists():
                errors.append(f"broken local link in {relative}: {raw_target}")
                continue
            fragment = urlsplit(raw_target.strip().strip("<>")).fragment
            if fragment and target.exists() and target.suffix in (".ipynb", ".md"):
                if fragment not in heading_slugs(target):
                    errors.append(
                        f"link in {relative} points at a heading that does not "
                        f"exist: {raw_target}"
                    )

    unreferenced = sorted(set(mermaid_diagrams) - referenced_mermaid)
    if unreferenced:
        errors.append("unreferenced Mermaid manifest entries: " + ", ".join(unreferenced))
    for diagram_id, diagram in sorted(mermaid_diagrams.items()):
        if not isinstance(diagram, dict) or any(
            not isinstance(diagram.get(field), str) or not diagram[field].strip()
            for field in ("source", "caption", "alt")
        ):
            errors.append(f"incomplete Mermaid manifest entry: {diagram_id}")
            continue
        svg = BOOK / "figures" / f"mermaid-{diagram_id}.svg"
        png = svg.with_suffix(".png")
        if not svg.is_file() or MERMAID_FONT_MARKER not in svg.read_text(encoding="utf-8"):
            errors.append(f"Mermaid SVG missing embedded DM Sans font: {svg.relative_to(BOOK)}")
        if not png.is_file():
            errors.append(f"missing Mermaid PNG for PDF output: {png.relative_to(BOOK)}")

    errors.extend(orphan_notebook_errors(pages))

    versions = {
        "Cargo.toml": read_version(ROOT / "Cargo.toml", r'^version\s*=\s*"([^"]+)"'),
        "python/mboostrs/__init__.py": read_version(
            ROOT / "python/mboostrs/__init__.py", r'^__version__\s*=\s*"([^"]+)"'
        ),
        "mboostrs/DESCRIPTION": read_version(ROOT / "mboostrs/DESCRIPTION", r"^Version:\s*(\S+)"),
        "pyproject.toml": read_version(ROOT / "pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
    }
    if None in versions.values() or len(set(versions.values())) != 1:
        errors.append("version mismatch: " + ", ".join(f"{name}={version}" for name, version in versions.items()))

    artifact_lines, artifact_counts = stale_artifact_report()
    read_by_page, historical, policy_problems = load_artifact_policy()
    errors.extend(policy_problems)
    errors.extend(historical_artifacts_read_by_notebooks(historical))
    pre_public = [line for _, line in artifact_lines if "predates public history" in line.split(";")[0]]
    pre_public_keys = {key for key, line in artifact_lines if line in pre_public}
    stale_read = [line for key, line in artifact_lines
                  if key not in historical and key not in pre_public_keys]
    stale_historical = [line for key, line in artifact_lines
                        if key in historical and key not in pre_public_keys]
    if pre_public:
        print(
            f"{len(pre_public)} benchmark artifact(s) record a commit from before the public "
            "history, so their staleness cannot be measured here; rerun them to record a "
            "commit of this history (--strict-artifacts makes them fail):",
            file=sys.stderr,
        )
        for line in pre_public:
            print(f"- {line}", file=sys.stderr)
    if stale_historical:
        print(
            f"Historical benchmark artifacts, allowed to be stale ({ARTIFACT_POLICY_NAME} lists "
            "them as logs of past runs that no page reads; --strict-artifacts makes them fail):",
            file=sys.stderr,
        )
        for line in stale_historical:
            print(f"- {line}", file=sys.stderr)
    for line in stale_read:
        errors.append(
            f"benchmark artifact read by a page is not current (rerun its benchmark): {line}"
        )
    if stale_historical and arguments.strict_artifacts:
        errors.append(
            f"{len(stale_historical)} historical benchmark artifact(s) are stale or record no "
            "commit (--strict-artifacts); see the list above"
        )
    if pre_public and arguments.strict_artifacts:
        errors.append(
            f"{len(pre_public)} benchmark artifact(s) predate the public history "
            "(--strict-artifacts); see the list above"
        )

    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Documentation sources valid: {len(pages)} ToC notebooks; "
        f"version {next(iter(versions.values()))}; capability contract current; "
        f"{len(declared_rust_items)} Rust items available to cite"
    )
    if artifact_counts:
        print(
            "Benchmark artifacts: "
            + ", ".join(f"{count} {label}" for label, count in artifact_counts.items()),
            file=sys.stderr if artifact_lines else sys.stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
