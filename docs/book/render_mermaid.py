#!/usr/bin/env python3
"""Pre-render all {mermaid} blocks to SVG for PDF builds.

MyST's LaTeX export cannot convert ``{mermaid}`` directives — they are silently
dropped from the PDF.  This script extracts every mermaid block from the
notebooks, renders it to SVG via mmdc (``@mermaid-js/mermaid-cli``), saves the
SVG to ``figures/``, and replaces the block with a ``{figure}`` image reference.

Usage
-----
    python3 render_mermaid.py            # render new/changed diagrams
    python3 render_mermaid.py --force    # re-render all diagrams

The script is idempotent — rerunning re-renders only changed diagrams (unless
``--force``).  Generated SVGs should be committed to the repo so CI builds
don't need mmdc or Chrome.

Requires ``@mermaid-js/mermaid-cli`` installed locally and a compatible
chrome-headless-shell in the puppeteer cache.  Install with::

    npm install @mermaid-js/mermaid-cli
    npx @puppeteer/browsers install chrome-headless-shell@<version>
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).parent
ROOT = BOOK.parent.parent
FIGURES = BOOK / "figures"
MANIFEST = BOOK / "mermaid_diagrams.json"
FONT_DIR = ROOT / "python" / "mboostrs" / "fonts"
FONT_FAMILY = "DM Sans"
FONT_MARKER = "--mboostrs-mermaid-font-v1"


def notebook_paths():
    """Yield notebooks only from source directories (skip build artifacts)."""
    for nb in sorted(BOOK.rglob("*.ipynb")):
        rel = nb.relative_to(BOOK)
        if rel.parts[0] in ("_build", "_site", "exports", "node_modules"):
            continue
        yield nb


def find_mermaid_blocks(notebook: Path):
    """Yield dicts with cell, mermaid source, fence syntax, and line ranges."""
    with open(notebook) as f:
        nb = json.load(f)
    cell_idx = -1
    for cell in nb.get("cells", []):
        cell_idx += 1
        source = cell.get("source", [])
        in_mermaid = False
        mermaid_lines = []
        fence_type = None
        fence_start = None
        for line_idx, line in enumerate(source):
            stripped = line.strip()
            if not in_mermaid:
                if stripped in ("```{mermaid}", "```mermaid"):
                    in_mermaid = True
                    fence_type = stripped
                    mermaid_lines = []
                    fence_start = line_idx
            else:
                if stripped == "```":
                    in_mermaid = False
                    if mermaid_lines:
                        yield {
                            "cell": cell_idx,
                            "source": "".join(mermaid_lines),
                            "fence": fence_type,
                            "fence_start": fence_start,
                            "fence_end": line_idx,
                            "notebook": notebook,
                        }
                    mermaid_lines = []
                    fence_type = None
                    fence_start = None
                else:
                    mermaid_lines.append(line)


def hash_source(source):
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def svg_path(diagram_id):
    return FIGURES / f"mermaid-{diagram_id}.svg"


def load_manifest():
    """Load the sources and prose that make generated diagrams reproducible."""
    if not MANIFEST.is_file():
        return {"schema_version": 1, "font_family": FONT_FAMILY, "diagrams": {}}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported Mermaid manifest schema: {MANIFEST}")
    return manifest


def save_manifest(manifest):
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def diagram_prose(source):
    """Extract required caption and alt-text comments from a new Mermaid block."""
    caption = None
    alt = None
    diagram_lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("%% caption:"):
            caption = stripped.removeprefix("%% caption:").strip()
        elif stripped.startswith("%% alt:"):
            alt = stripped.removeprefix("%% alt:").strip()
        else:
            diagram_lines.append(line)
    if not caption or not alt:
        raise ValueError(
            "Mermaid blocks require one-line '%% caption:' and '%% alt:' comments"
        )
    return "".join(diagram_lines), caption, alt


def _font_data_uri(filename):
    payload = base64.b64encode((FONT_DIR / filename).read_bytes()).decode("ascii")
    return f"data:font/woff2;base64,{payload}"


def mermaid_css():
    """Embed the book font because an SVG image cannot inherit page CSS."""
    regular = _font_data_uri("DMSans-400.woff2")
    bold = _font_data_uri("DMSans-700.woff2")
    return f'''@font-face {{
  font-family: "{FONT_FAMILY}";
  src: url("{regular}") format("woff2");
  font-style: normal;
  font-weight: 400;
}}
@font-face {{
  font-family: "{FONT_FAMILY}";
  src: url("{bold}") format("woff2");
  font-style: normal;
  font-weight: 700;
}}
svg {{ {FONT_MARKER}: 1; }}
svg, text, tspan, .label, .label div, .label p, .nodeLabel,
.edgeLabel, .messageText, .loopText, .actor, .actor-box,
.actor-man line, .actor-man circle, foreignObject div, foreignObject span,
foreignObject p {{
  font-family: "{FONT_FAMILY}", sans-serif !important;
}}
'''


# ---------------------------------------------------------------------------
# Rendering backends
# ---------------------------------------------------------------------------

def _mmdc_path():
    """Locate the mmdc binary."""
    rc = subprocess.run(["which", "mmdc"], capture_output=True, text=True)
    if rc.returncode == 0:
        return rc.stdout.strip()
    candidates = [
        BOOK.parent.parent / "node_modules" / ".bin" / "mmdc",
        Path.cwd() / "node_modules" / ".bin" / "mmdc",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def render_via_mmdc(source, out_file, extra_args=None):
    """Render with a browser after loading the embedded book font."""
    renderer = BOOK / "render_mermaid.mjs"
    if not renderer.is_file() or _mmdc_path() is None:
        print(
            "  Mermaid dependencies not found (run npm install)",
            file=sys.stderr,
        )
        return False

    mmd_file = out_file.with_suffix(".mmd")
    mmd_file.write_text(source)
    config_file = out_file.with_suffix(".json")
    config_file.write_text(json.dumps({
        "theme": "default",
        "fontFamily": FONT_FAMILY,
        "themeVariables": {"fontFamily": FONT_FAMILY},
    }))
    css_file = out_file.with_suffix(".css")
    css_file.write_text(mermaid_css(), encoding="utf-8")
    env = os.environ.copy()
    env.pop("PUPPETEER_CACHE_DIR", None)
    scale = "1"
    extra_args = extra_args or []
    if "-s" in extra_args:
        scale = extra_args[extra_args.index("-s") + 1]
    cmd = [
        "node",
        str(renderer),
        str(mmd_file),
        str(out_file),
        str(config_file),
        str(css_file),
        scale,
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=True, env=env,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  mmdc FAILED: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print(f"  stderr: {exc.stderr[:500]}", file=sys.stderr)
        return False
    finally:
        mmd_file.unlink(missing_ok=True)
        config_file.unlink(missing_ok=True)
        css_file.unlink(missing_ok=True)
    return True


def render_one(source, svg_file, force=False):
    """Render the SVG (used by the HTML/site build) and a same-stem PNG.

    Mermaid's SVG node labels are HTML <foreignObject> content; rsvg-convert
    (used by build-pdf.sh to rasterize figures for LaTeX) silently drops
    foreignObject, so every diagram would render as empty boxes in the PDF.
    The PNG is rendered by the same real-browser (puppeteer) pipeline, so
    it keeps the text, and build-pdf.sh embeds it directly for mermaid
    figures instead of converting the SVG.
    """
    ok = True
    svg_has_font = (
        svg_file.is_file()
        and FONT_MARKER in svg_file.read_text(encoding="utf-8")
    )
    # The PNG is the file that reaches the PDF, and it cannot carry a CSS
    # marker, so it used to be invalidated only by not existing: a font change
    # re-rendered the SVG for the HTML build and silently left the PDF on the
    # old typeface until someone ran --force. Tying the PNG's staleness to the
    # SVG's covers both outputs with one check.
    png_file = svg_file.with_suffix(".png")
    if force or not svg_has_font:
        ok = render_via_mmdc(source, svg_file) and ok
        ok = render_via_mmdc(source, png_file, extra_args=["-s", "3"]) and ok
    elif not png_file.exists():
        ok = render_via_mmdc(source, png_file, extra_args=["-s", "3"]) and ok
    return ok


def inject_figure_reference(
    notebook, cell_idx, svg_rel, alt, caption, fence_start, fence_end
):
    with open(notebook) as f:
        nb = json.load(f)
    cell = nb["cells"][cell_idx]
    cell["cell_type"] = "markdown"
    source = cell["source"]
    figure_lines = [
        "```{figure} " + svg_rel + "\n",
        f":name: {Path(svg_rel).stem}\n",
        f":alt: {alt}\n",
        "\n",
        f"{caption}\n",
        "```\n",
    ]
    source[fence_start:fence_end + 1] = figure_lines
    cell["source"] = source
    cell.pop("execution_count", None)
    cell.pop("outputs", None)
    with open(notebook, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  replaced mermaid block with {{figure}} -> {svg_rel}")


def main():
    force = "--force" in sys.argv
    FIGURES.mkdir(exist_ok=True)
    manifest = load_manifest()
    diagrams = manifest.setdefault("diagrams", {})
    converted = 0
    for nb_path in notebook_paths():
        blocks = list(find_mermaid_blocks(nb_path))
        if not blocks:
            continue
        # Process blocks bottom-to-top within each cell so that replacing
        # a block at a higher line index doesn't shift indices for blocks
        # at lower line indices.
        blocks.sort(key=lambda b: (b["cell"], b["fence_start"]), reverse=True)
        for block in blocks:
            source, caption, alt = diagram_prose(block["source"])
            did = hash_source(source)
            svg = svg_path(did)
            rel = nb_path.relative_to(BOOK.parent.parent)
            converted += 1
            diagrams[did] = {"source": source, "caption": caption, "alt": alt}
            print(f"[new {converted}] {rel}:{block['cell']} -> {svg.name}")
            ok = render_one(source, svg, force=force)
            if ok:
                svg_rel = os.path.relpath(svg, nb_path.parent)
                inject_figure_reference(nb_path, block["cell"], svg_rel, alt, caption,
                                        block["fence_start"], block["fence_end"])

    if converted:
        save_manifest(manifest)

    rendered = 0
    for did, diagram in sorted(diagrams.items()):
        rendered += 1
        print(f"[render {rendered}/{len(diagrams)}] mermaid-{did}")
        if not render_one(diagram["source"], svg_path(did), force=force):
            raise RuntimeError(f"failed to render Mermaid diagram {did}")

    # Outputs are content-addressed by a hash of their source, which makes
    # rendering correctly incremental — but nothing removed the superseded
    # pair, and the docstring says to commit the outputs, so every diagram edit
    # left a tracked orphan behind. Report rather than delete: the files are
    # committed, so removing them is the author's call, not this script's.
    live = {f"mermaid-{did}" for did in diagrams}
    orphans = sorted(
        path.name
        for path in FIGURES.glob("mermaid-*")
        if path.stem not in live
    )
    if orphans:
        total = sum((FIGURES / name).stat().st_size for name in orphans)
        print(
            f"\n{len(orphans)} orphaned Mermaid render(s), {total} bytes, "
            f"superseded by a source edit and not referenced by any chapter:"
        )
        for name in orphans:
            print(f"  {name}")
        print("  remove with: git rm docs/book/figures/{" + ",".join(orphans) + "}")

    print(
        f"\n{rendered} Mermaid diagram(s) rendered; "
        f"{converted} new block(s) converted."
    )
    return rendered


if __name__ == "__main__":
    main()
