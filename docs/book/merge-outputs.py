#!/usr/bin/env python3
"""Post-process MyST HTML build: inline SVG file references and strip text/plain.

MyST's static HTML build stores Altair chart outputs in two ways inside the
``window.__remixContext`` hydration blob:

1. **Inline SVG** — ``{"content": "<svg>...</svg>", "content_type": "image/svg+xml"}``
   Renders correctly without any client-side fetch.

2. **Hash/path reference** — ``{"content_type": "image/svg+xml", "hash": "...",
   "path": "/build/XXX.svg"}`` — the client renderer must fetch the external
   file.  In the static build this fetch fails, so the output area shows
   "Loading..." indefinitely.

Additionally, when both ``image/svg+xml`` and ``text/plain`` are present in a
MIME bundle, the client renderer may fall back to the text/plain content
(``alt.Chart(...)``) instead of rendering the SVG.

This script walks every HTML page's ``__remixContext``, replaces hash/path
SVG references with inline content (read from the build directory), and
removes ``text/plain`` from MIME bundles that already have ``image/svg+xml``.
"""

import json
import re
from pathlib import Path

BUILD_DIR = Path("_build")


def inline_svg_refs(node, html_dir):
    """Inline SVGs and remove their plain-text fallback from notebook output.

    A chart returned as the cell's last expression is an ``execute_result``;
    ``display(chart)`` produces ``display_data``. Both carry the same MIME
    bundle and need the same rewrite. Handling only the first form made the
    result depend on how the author displayed an otherwise identical chart.
    """
    if isinstance(node, dict):
        jd = node.get("jupyter_data")
        if (isinstance(jd, dict)
                and jd.get("output_type") in {"execute_result", "display_data"}):
            data = jd.get("data", {})
            svg = data.get("image/svg+xml")
            if isinstance(svg, dict) and "path" in svg and "content" not in svg:
                svg_path = html_dir / svg["path"].lstrip("/")
                if svg_path.exists():
                    content = svg_path.read_text(encoding="utf-8")
                    data["image/svg+xml"] = {
                        "content": content,
                        "content_type": "image/svg+xml",
                    }
                else:
                    print(f"    WARNING: SVG file missing: {svg_path}")
            if "image/svg+xml" in data and "text/plain" in data:
                del data["text/plain"]
        for value in node.values():
            if isinstance(value, (dict, list)):
                inline_svg_refs(value, html_dir)
    elif isinstance(node, list):
        for item in node:
            inline_svg_refs(item, html_dir)



def script_safe_json(value):
    """Serialize `value` for embedding between `<script>` tags.

    `json.dumps` escapes nothing HTML-aware, so an inlined SVG carrying
    `</script>` — in a title, a tooltip, or an embedded font comment — would
    terminate the script element early and destroy the page's hydration data.
    Escaping `<` as `\u003c` is what MyST's own serializer does; the built tree
    shows both conventions side by side because this rewrite did not.
    """
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def unresolved_chart_outputs(node, location="root"):
    """Paths of chart fallbacks or external SVG references left in ``node``."""
    problems = []
    if isinstance(node, dict):
        jd = node.get("jupyter_data")
        if isinstance(jd, dict):
            data = jd.get("data", {})
            plain = data.get("text/plain")
            if isinstance(plain, dict):
                plain = plain.get("content", "")
            if isinstance(plain, list):
                plain = "".join(plain)
            if str(plain or "").startswith("alt."):
                problems.append(f"{location}: Altair text fallback remains")
            svg = data.get("image/svg+xml")
            if isinstance(svg, dict) and "path" in svg and "content" not in svg:
                problems.append(f"{location}: external SVG reference remains")
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                problems.extend(unresolved_chart_outputs(value, f"{location}.{key}"))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            problems.extend(unresolved_chart_outputs(item, f"{location}[{index}]"))
    return problems


def patch_html():
    """Patch __remixContext in all HTML pages: inline SVG refs + strip text/plain."""
    html_dir = BUILD_DIR / "html"
    n_patched = 0
    unresolved = []
    # rglob, not glob("*/index.html"): the latter is exactly one level deep and
    # misses _build/html/index.html, the book's landing page and the one page a
    # first-time reader sees. An unpatched SVG reference renders as "Loading..."
    # forever, which is the failure this script exists to prevent.
    for html_path in sorted(html_dir.rglob("index.html")):
        html = html_path.read_text(encoding="utf-8")
        m = re.search(
            r"<script>window\.__remixContext\s*=\s*(\{.+?\});\s*</script>",
            html,
            re.DOTALL,
        )
        if not m:
            continue
        try:
            ctx = json.loads(m.group(1))
        except json.JSONDecodeError:
            print(f"    WARNING: could not parse __remixContext in {html_path}")
            continue
        before = json.dumps(ctx, ensure_ascii=False)
        inline_svg_refs(ctx, html_dir)
        unresolved.extend(
            f"{html_path}: {problem}"
            for problem in unresolved_chart_outputs(ctx)
        )
        after = json.dumps(ctx, ensure_ascii=False)
        if before != after:
            new_json = script_safe_json(ctx)
            new_html = html[: m.start(1)] + new_json + html[m.end(1) :]
            html_path.write_text(new_html, encoding="utf-8")
            n_patched += 1
            slug = html_path.parent.name if html_path.parent != html_dir else "(landing page)"
            print(f"    {slug}")
    if n_patched:
        print(f"  Patched {n_patched} HTML pages")
    if unresolved:
        raise RuntimeError(
            "chart outputs remain unresolved after HTML patching:\n- "
            + "\n- ".join(unresolved)
        )


if __name__ == "__main__":
    patch_html()
