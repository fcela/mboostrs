"""Bundle book notebooks and their declared local evidence for download.

Notebooks declare supporting files in `metadata.mboostrs_artifacts`, relative
to the notebook. Those files accompany the page so recorded studies can run
from the download without fetching undisclosed inputs.

The bundle is a build output: `build-html.sh` writes it next to the rendered
site, and the documentation workflow publishes both. It is not tracked, so
anything that needs one builds it.

Usage:
    python3 docs/book/generate_notebooks.py [--output PATH]
"""

import zipfile
import json
from pathlib import Path

BOOK = Path(__file__).parent
EXPORTS = BOOK / "exports"
ZIP_PATH = EXPORTS / "notebooks.zip"


def _gather_pages() -> list[Path]:
    """Return all .ipynb files listed in the book's myst.yml ToC."""
    import yaml

    config = yaml.safe_load((BOOK / "myst.yml").read_text())
    pages: list[Path] = []
    toc = config["project"]["toc"]

    def walk(entries):
        for entry in entries:
            if "file" in entry and entry["file"].endswith(".ipynb"):
                pages.append(BOOK / entry["file"])
            if "children" in entry:
                walk(entry["children"])

    walk(toc)
    return pages


def bundle_files() -> list[Path]:
    """ToC notebooks and the local evidence files their metadata declares."""
    pages = _gather_pages()
    artifacts = set()
    for page in pages:
        notebook = json.loads(page.read_text())
        for relative in notebook.get("metadata", {}).get("mboostrs_artifacts", []):
            artifact = (page.parent / relative).resolve()
            if not artifact.is_relative_to(BOOK.resolve()) or not artifact.is_file():
                raise ValueError(f"invalid notebook artifact {relative!r} in {page}")
            artifacts.add(artifact)
    return [*pages, *sorted(artifacts)]


def build(dest: Path | None = None) -> Path:
    """Write the bundle to `dest`, defaulting to `exports/notebooks.zip`."""
    target = ZIP_PATH if dest is None else Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in bundle_files():
            zf.write(src, src.relative_to(BOOK))
    return target


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bundle the book notebooks.")
    parser.add_argument(
        "--output", type=Path, default=ZIP_PATH,
        help="where to write the bundle (default docs/book/exports/notebooks.zip)")
    args = parser.parse_args()

    pages = _gather_pages()
    print(f"Bundling {len(pages)} notebooks...")
    target = build(args.output)
    size_mb = target.stat().st_size / 1_024 / 1_024
    print(f"Done: {target.name} ({len(pages)} notebooks, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
