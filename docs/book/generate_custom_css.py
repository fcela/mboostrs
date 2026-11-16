#!/usr/bin/env python3
"""Build `custom.css` from `custom.css.in` and the packaged woff2 fonts.

`custom.css` is 276 KB across 54 lines, with single lines up to 91,636
characters, because the four font faces are base64 data URIs. That is
load-bearing rather than an oversight: MyST content-hashes the stylesheet to
`_build/html/build/custom-<hash>.css` and does not stage assets referenced from
inside it, so a relative `url(fonts/...)` resolves to a path nothing writes,
and an absolute one breaks under the `BASE_URL` the gh-pages deploy uses. There
are zero woff2 files anywhere in a built tree; inlining is what makes the fonts
resolve.

What the inlining cost was reviewability: a font swap and a rule change were
indistinguishable in a diff, and the bytes were a second copy of fonts the
package already ships. Generating the file fixes both — `custom.css.in` holds
the rules and is 54 readable lines, and the payloads come from
`python/mboostrs/fonts/`, which AGENTS.md gotcha 33 already names as the single
source of truth for the font family names.
"""

import base64
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]
FONT_DIR = ROOT / "python/mboostrs/fonts"
TEMPLATE = BOOK / "custom.css.in"
OUTPUT = BOOK / "custom.css"
PLACEHOLDER = "{{FONT}}"

#: One entry per `{{FONT}}` in the template, in order. Must match the
#: `@font-face` blocks' family and weight.
FONT_ORDER = (
    "DMSans-400.woff2",
    "DMSans-700.woff2",
    "CommitMono-Regular.woff2",
    "CommitMono-Bold.woff2",
)


def render() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != len(FONT_ORDER):
        raise SystemExit(
            f"{TEMPLATE.name} has {template.count(PLACEHOLDER)} {PLACEHOLDER} "
            f"placeholders but FONT_ORDER lists {len(FONT_ORDER)} fonts"
        )
    for name in FONT_ORDER:
        path = FONT_DIR / name
        if not path.is_file():
            raise SystemExit(f"missing font: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        template = template.replace(
            PLACEHOLDER, f"data:font/woff2;base64,{encoded}", 1
        )
    return template


def main() -> int:
    rendered = render()
    if "--check" in sys.argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != rendered:
            print(
                "custom.css is stale; run python3 docs/book/generate_custom_css.py",
                file=sys.stderr,
            )
            return 1
        print(f"custom.css current ({len(rendered)} bytes, {len(FONT_ORDER)} fonts)")
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote custom.css ({len(rendered)} bytes, {len(FONT_ORDER)} fonts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
