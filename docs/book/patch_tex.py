"""Patch generated LaTeX before pdflatex run.

1. Replace .svg} with .pdf} in all chapter .tex files
   (rsvg-convert produces .pdf from .svg) -- except mermaid-*.svg, which
   are repointed at the sibling .png instead. Mermaid's SVG node labels
   are HTML <foreignObject> content that rsvg-convert silently drops
   (see render_mermaid.py); the PNG is rendered by the same puppeteer
   pipeline and keeps the text.
2. Convert verbatim environments to minted with language detection.
3. Repair generated tables: MyST converts Markdown and DataFrame HTML output
   to a `tabular` with every column forced to an identical
   `p{0.NNN\\linewidth}` fraction, regardless of content. A column whose
   cells are all short (e.g. a single digit) still gets the same width as
   a column of long family names, so the long column overflows its box and
   visually collides with its neighbor (LaTeX doesn't insert a gap for
   `p`-column overflow). Re-tag short tables (equal-width `p{}` columns,
   4+ of them, every cell short) as natural-width `l` columns, which size
   to content instead. Convert tall tables to `longtable` while preserving
   their wrapping column widths; an ordinary `tabular` cannot split across
   pages and can send pdfTeX into an unbounded empty-page output loop.
4. Reject legacy Mermaid captions that contain graph source. Source notebooks
   must supply the explanatory captions recorded in mermaid_diagrams.json;
   silently replacing malformed prose would conceal the documentation defect.
"""
import re
import sys
from pathlib import Path


def detect_language(code: str) -> str:
    """Detect code language from content heuristics."""
    lines = code.strip().split("\n")[:20]
    if not lines:
        return "text"

    all_lines = code.strip().split("\n")
    nonempty = [line for line in all_lines if line.strip()]

    # Rust
    rust_score = sum(
        2
        for line in lines
        if line.strip().startswith(("fn ", "impl ", "pub ", "let ", "unsafe ", "struct ", "enum ", "trait ", "macro_rules!"))
    )
    rust_score += sum(1 for line in lines if "::<" in line)
    # `-> ` in Rust looks like `->` in R; only count it when Rust signals are present
    if rust_score >= 2:
        rust_score += sum(1 for line in nonempty if "-> " in line)

    # R
    r_score = sum(
        2
        for line in lines
        if line.strip().startswith(("library(", "require(", "install.packages("))
    )
    r_score += sum(1 for line in nonempty if " <- " in line)
    r_score += sum(1 for line in lines if re.search(r"function\s*\(", line) and "{" in line)

    # bash
    bash_cmds = [
        "curl ", "cargo ", "git ", "cd ", "ls ", "export ", "source ", "make ",
        "maturin ", "pip install", "npm ", "rustup ", " which ", "chmod ",
        "rm ", "mv ", "cp ", "mkdir ", "touch ", "echo ", "cat ",
    ]
    bash_score = 0
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(c) for c in bash_cmds) or stripped.startswith(
            "$ "
        ):
            bash_score += 2

    # Python
    python_score = sum(
        2
        for line in lines
        if line.strip().startswith(
            (
                "import ", "from ", "def ", "class ", ">>>", "print(",
                "for ", "while ", "if ", "elif ", "else:", "try:", "except",
                "with ", "return ", "raise ", "yield ", "lambda ",
            )
        )
    )
    python_score += sum(
        1
        for line in lines
        if any(pattern in line for pattern in ["= mb.", "mb.", "np.", "plt.", "pd."])
    )
    python_score += sum(1 for line in lines if line.strip().startswith("@"))  # decorators

    # Tiebreak: if mixed signals, prefer Python (dominant language in this book)
    if rust_score >= 2 and rust_score > python_score:
        return "rust"
    if r_score >= 2 and r_score > bash_score and r_score >= python_score:
        return "r"
    if bash_score >= 2 and bash_score > python_score:
        return "bash"
    if python_score >= 2:
        return "python"

    # Weak signals: if any bash command found, guess bash
    if bash_score > 0 or r_score > 0:
        return "bash" if bash_score >= r_score else "r"

    return "text"


def patch_verbatim_to_minted(content: str) -> str:
    """Convert \\begin{verbatim}...\\end{verbatim} to minted blocks.

    Handles nested braces and escaped characters.
    """
    # Match verbatim blocks with their content
    pattern = re.compile(
        r"\\begin\{verbatim\}\n?(.*?)\n?\\end\{verbatim\}",
        re.DOTALL,
    )

    def replace_block(m):
        code = m.group(1)
        lang = detect_language(code)
        return f"\\begin{{minted}}[frame=single]{{{lang}}}\n{code}\n\\end{{minted}}"

    return pattern.sub(replace_block, content)


# MyST omits the numeric fraction for a single full-width column.
_EQUAL_WIDTH_COL = re.compile(
    r"p\{\\dimexpr (?:0\.\d+|1\.0+)?\\linewidth-2\\tabcolsep\}"
)
_TABULAR = re.compile(
    r"\\begin\{tabular\}\{((?:" + _EQUAL_WIDTH_COL.pattern + r")+)\}"
    r"(.*?)\\end\{tabular\}",
    re.DOTALL,
)
_MAX_SHORT_CELL = 25
_MAX_TABULAR_ROWS = 24
# Characters a full-width line holds. The book's tables are typewriter-heavy,
# which sets fewer characters per line than prose at the same width, so a
# p{} column of fraction F wraps at about F * _CHARS_PER_LINE characters.
_CHARS_PER_LINE = 60
# Body lines on an a4 page at 11pt. A tabular taller than this fits on no
# page at all, and that is the case pdfTeX answers with empty pages forever.
_PAGE_LINES = 42
# \texttt{, \textbf{, \toprule, the braces they close with, and the quote
# ligatures: markup that occupies no width on the page and must not be
# counted when estimating how many lines a cell wraps to.
_MARKUP = re.compile(r"\\[a-zA-Z@]+\s*\{?|[{}]|``|''")


_MULTICOLUMN = re.compile(r"\\multicolumn\{(\d+)\}")


def _row_span(row):
    """Columns a tabular row covers, counting each multicolumn's span."""
    cells = row.split("&")
    spans = [int(m.group(1)) if (m := _MULTICOLUMN.search(c)) else 1 for c in cells]
    return sum(spans)


def pad_short_header_rows(body, n_cols):
    """Restore the cells a two-row header cell leaves out of the row below.

    A great_tables header whose stub (row-label) cell, or a column label
    outside every spanner, spans two rows loses its second row in MyST's
    HTML-to-LaTeX conversion, so the labels under the spanners start too far
    left. Below a row that has a spanner, the missing cells sit where that
    row has single cells; anywhere else a short row is padded on the left.
    """
    rows = body.split(r"\\")
    out = []
    single_positions = None
    for row in rows:
        stripped = row.strip()
        if stripped and not re.fullmatch(
            r"(?:\\(?:toprule|midrule|hline|bottomrule)\s*)+", stripped
        ):
            # A row can begin with a rule command left over from the previous line.
            end = re.match(r"\s*(?:\\(?:hline|toprule|midrule)\s*)*", row).end()
            lead, rest = row[:end], row[end:]
            missing = n_cols - _row_span(rest)
            if missing > 0:
                cells = rest.split("&")
                if single_positions is not None and len(single_positions) == missing:
                    for position in single_positions:
                        cells.insert(position, " ")
                    rest = "&".join(cells)
                else:
                    rest = " & " * missing + rest
                row = lead + rest
            # Columns of this row's single cells, when it also has a spanner.
            spans = [int(m.group(1)) if (m := _MULTICOLUMN.search(c)) else 1
                     for c in rest.split("&")]
            if any(1 < k < n_cols for k in spans):
                starts = [sum(spans[:i]) for i in range(len(spans))]
                single_positions = [st for st, k in zip(starts, spans) if k == 1]
            else:
                single_positions = None
        out.append(row)
    return r"\\".join(out)


def _estimated_lines(body, fractions):
    """Lines the wrapped cells of a tabular body will occupy on the page."""
    total = 0
    for row in body.split(r"\\"):
        if not row.strip():
            continue
        if _MULTICOLUMN.search(row) and "&" not in row:
            # A title or note across the whole table wraps at the full width.
            text = _MARKUP.sub("", row).strip()
            total += max(1, -(-len(text) // _CHARS_PER_LINE))
            continue
        height = 1
        for cell, fraction in zip(row.split("&"), fractions):
            text = _MARKUP.sub("", cell).strip()
            per_line = max(1, int(float(fraction) * _CHARS_PER_LINE))
            height = max(height, -(-len(text) // per_line))
        total += height
    return total


# MyST gives every column the same share of the line whatever it holds, so a
# column of "1e-08" is as wide as a column of prose and the prose column wraps
# to twenty lines. Width follows content instead: a column's share is the mean
# length of its cells, floored so a narrow column still fits its own values,
# and a long unbreakable identifier counts for at most _LONG_TOKEN_CAP so one
# test path does not take the whole line.
_MIN_COLUMN_SHARE = 0.06
_LONG_TOKEN_CAP = 12


def _column_shares(body, n_cols):
    """The fraction of the line each column gets, proportional to its content."""
    totals = [0.0] * n_cols
    counts = [0] * n_cols
    widest_token = [0] * n_cols
    header_token = [0] * n_cols
    header_end = _header_end(body)
    position = 0
    for row in body.split(r"\\"):
        position += len(row) + 2
        if not row.strip():
            continue
        j = 0
        for cell in row.split("&"):
            if j >= n_cols:
                break
            span = _MULTICOLUMN.search(cell)
            if span:
                # Ignore the spanner itself, but retain ordinary headings
                # beside it at their actual column positions.
                j += int(span.group(1))
                continue
            text = _MARKUP.sub("", cell).strip()
            totals[j] += len(text)
            counts[j] += 1
            for token in text.split():
                widest_token[j] = max(widest_token[j], len(token))
                if position <= header_end:
                    header_token[j] = max(header_token[j], len(token))
            j += 1
    weights = [
        max(totals[j] / counts[j] if counts[j] else 1.0,
            min(widest_token[j], _LONG_TOKEN_CAP),
            # Wide proportional tables need room for heading words even
            # when most values below them are single-digit counts.
            (2 if n_cols >= 8 else 1) * min(header_token[j], _LONG_TOKEN_CAP), 1.0)
        for j in range(n_cols)
    ]
    total = sum(weights)
    shares = [w / total for w in weights]
    floored = {j for j, s in enumerate(shares) if s < _MIN_COLUMN_SHARE}
    if floored and len(floored) < n_cols:
        # Hold the floored columns at the floor and divide what is left among
        # the rest; renormalising everything would undo the floor.
        remaining = 1.0 - _MIN_COLUMN_SHARE * len(floored)
        rest_total = sum(shares[j] for j in range(n_cols) if j not in floored)
        shares = [
            _MIN_COLUMN_SHARE if j in floored else shares[j] * remaining / rest_total
            for j in range(n_cols)
        ]
    return shares


def _colspec(shares):
    return "".join(
        f"p{{\\dimexpr {s:.3f}\\linewidth-2\\tabcolsep}}" for s in shares
    )


# A column whose body cells are all at most this many characters is a
# short column: numbers and short labels, set in a fixed width that fits its
# longest value so it never runs into the next column. Its heading may wrap.
_NATURAL_CELL = 16
# Text width of the book's a4, 11pt page in ems of the body font; footnote
# size sets 11/9 as many ems on the same line.
_LINE_EMS = 32.0


def _column_lengths(body, n_cols, start):
    """Longest body cell per column, from ``start`` on, skipping full-width rows."""
    longest = [0] * n_cols
    for row in body[start:].split(r"\\"):
        if not row.strip() or _MULTICOLUMN.search(row):
            continue
        for j, cell in enumerate(row.split("&")[:n_cols]):
            text = _MARKUP.sub("", cell).strip()
            longest[j] = max(longest[j], len(text))
    return longest


def _mixed_layout(body, n_cols):
    """Fixed-width short columns and wrapping long columns for one table.

    Returns None unless some but not all columns are short. Otherwise
    returns ``(widths, small)``: each short column's width in ems (None for a
    long column) and whether the table is set in footnote size, which it is
    when the short columns would leave the long ones less than a third of
    the line. Fall back to proportional wrapping if even footnote size
    leaves less than six ems for the long columns after cell padding.
    """
    longest = _column_lengths(body, n_cols, _header_end(body))
    short = [n <= _NATURAL_CELL for n in longest]
    if not any(short) or all(short):
        return None
    # About half an em per character of figures and short words, and room
    # for a heading word such as "C-index" to sit on one line.
    widths = [max(0.55 * longest[j] + 0.5, 3.6) if short[j] else None
              for j in range(n_cols)]
    # Each column also carries 2 \tabcolsep, 12pt, about 1.1 em.
    used = sum(w for w in widths if w) + 1.1 * n_cols
    small = used > _LINE_EMS * 2 / 3
    line_ems = _LINE_EMS * (11 / 9 if small else 1.0)
    left = line_ems - sum(w for w in widths if w) - _padding_ems(n_cols, small)
    if left < 6.0:
        # An X column or the longtable's remaining-width p{} column would
        # otherwise be too narrow, or negative, even after reducing the font.
        return None
    return widths, small


def _padding_ems(n_cols, small):
    """Ems of \\tabcolsep padding: 12pt of 11pt type, or 8pt of 9pt type."""
    return (0.9 if small else 1.1) * n_cols


def _wrap_spanners(body, widths):
    """Let a spanner over fixed-width columns wrap within their total width.

    A centred spanner wider than the columns under it widens the last of
    them, which pushes the table past the margin.
    """
    out = []
    for row in body.split(r"\\"):
        cells = row.split("&")
        spans = [int(m.group(1)) if (m := _MULTICOLUMN.search(c)) else 1 for c in cells]
        if not any(1 < k < len(widths) for k in spans):
            out.append(row)
            continue
        start = 0
        for i, (cell, k) in enumerate(zip(cells, spans)):
            covered = widths[start:start + k]
            if k > 1 and all(covered):
                width = sum(covered)
                cells[i] = cell.replace(
                    f"\\multicolumn{{{k}}}{{c}}",
                    f"\\multicolumn{{{k}}}{{>{{\\centering\\arraybackslash}}"
                    f"p{{\\dimexpr {width:.1f}em+{2 * (k - 1)}\\tabcolsep\\relax}}}}", 1)
            start += k
        out.append("&".join(cells))
    return r"\\".join(out)


def _long_column_spec(widths, fraction):
    """A p{} column taking ``fraction`` of what the short columns leave."""
    fixed = sum(w for w in widths if w)
    return (f"p{{\\dimexpr ({fraction:.3f}\\linewidth - {fraction:.3f}\\dimexpr {fixed:.1f}em\\relax"
            f" - {2 * len(widths) * fraction:.3f}\\tabcolsep)\\relax}}")


def _mixed_lines(body, n_cols, widths, small):
    """Lines a mixed-layout table occupies, counting the long columns' wrapping."""
    line_ems = _LINE_EMS * (11 / 9 if small else 1.0)
    left = line_ems - sum(w for w in widths if w) - _padding_ems(n_cols, small)
    n_long = sum(1 for w in widths if w is None)
    per_line = max(1, int(left / n_long / 0.5))
    total = 0
    for row in body.split(r"\\"):
        if not row.strip():
            continue
        if _MULTICOLUMN.search(row) and "&" not in row:
            text = _MARKUP.sub("", row).strip()
            total += max(1, -(-len(text) // int(line_ems / 0.5)))
            continue
        height = 1
        for j, cell in enumerate(row.split("&")[:n_cols]):
            if widths[j] is None:
                text = _MARKUP.sub("", cell).strip()
                height = max(height, -(-len(text) // per_line))
        total += height
    return total


def _header_end(body):
    """Index just after the rule closing the column headings.

    A great_tables header puts its title, subtitle and spanner rows (all
    multicolumn) above the column labels; the rule after the first row that
    is not a multicolumn row closes the heading.
    """
    position = 0
    for m in re.finditer(r"\\hline", body):
        before = body[position:m.start()]
        rows = [r for r in before.split(r"\\") if r.strip()]
        position = m.end()
        if rows and not _MULTICOLUMN.search(rows[-1]):
            return m.end()
    first = body.find(r"\hline")
    return first + len(r"\hline") if first >= 0 else 0


def repair_generated_tables(content: str) -> str:
    """Make generated equal-width tables readable and safe to paginate."""

    def replace_table(m):
        colspec, body = m.group(1), m.group(2)
        n_cols = len(_EQUAL_WIDTH_COL.findall(colspec))
        body = pad_short_header_rows(body, n_cols)
        # A spanner over some of the columns keeps MyST's equal-width guess,
        # which no longer matches the resized columns; centre it instead.
        body = re.sub(
            r"\\multicolumn\{(\d+)\}\{p\{\\dimexpr 0\.\d+\\linewidth-2\\tabcolsep\}\}",
            lambda m: f"\\multicolumn{{{m.group(1)}}}{{c}}", body)
        shares = _column_shares(body, n_cols)
        sized = _colspec(shares)
        rows = [
            row for row in body.split(r"\\")
            if row.strip() and row.strip() not in (r"\toprule", r"\hline", r"\bottomrule")
        ]
        cells = [c.strip() for row in rows for c in row.split("&")]
        mixed = _mixed_layout(body, n_cols) if n_cols >= 4 else None
        if mixed:
            # Short columns get a fixed width that fits their values; the
            # long ones (a method name, say) wrap in what is left of the
            # line: tabularx's X columns on one page, p{} columns sized from
            # the same arithmetic in a longtable across pages.
            widths, small = mixed
            body = _wrap_spanners(body, widths)
            n_long = sum(1 for w in widths if w is None)
            fixed = [
                f">{{\\raggedleft\\arraybackslash}}p{{{w:.1f}em}}" if w else None
                for w in widths
            ]
            opening, closing = (
                ("{\\footnotesize\\setlength{\\tabcolsep}{4pt}\n", "}") if small else ("", ""))
            page_lines = _PAGE_LINES * (11 / 9 if small else 1.0)
            if len(rows) > _MAX_TABULAR_ROWS or _mixed_lines(body, n_cols, widths, small) > page_lines:
                spec = "".join(f or _long_column_spec(widths, 1 / n_long) for f in fixed)
                end = _header_end(body)
                paged_body = body[:end] + "\n\\endhead" + body[end:]
                return (f"{opening}\\begin{{longtable}}{{{spec}}}"
                        f"{paged_body}\\end{{longtable}}{closing}")
            spec = "".join(f or ">{\\raggedright\\arraybackslash}X" for f in fixed)
            return (f"{opening}\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}"
                    f"{body}\\end{{tabularx}}{closing}")
        # Wide tables that cannot keep fixed numeric columns need more room
        # for both their headings and the gaps between wrapped values.
        if n_cols >= 8:
            opening, closing = "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n", "}"
        elif any(_estimated_lines(row, shares) > _PAGE_LINES * 0.75 for row in rows):
            # Longtable can split between rows only. Leave a margin for
            # word wrapping in long API signatures, which the character
            # estimate understates, so no single row exceeds a whole page.
            opening, closing = "{\\footnotesize\n", "}"
        else:
            opening, closing = "", ""
        # Row count alone misses the table that broke this build: thirteen
        # rows of three columns, several cells holding a paragraph of prose
        # in a third of the text width, so the box stood taller than a page.
        too_tall = (
            len(rows) > _MAX_TABULAR_ROWS
            or _estimated_lines(body, shares) > _PAGE_LINES
        )
        if too_tall:
            # Repeat the column headings after each page break.
            end = _header_end(body)
            paged_body = body[:end] + "\n\\endhead" + body[end:]
            return (
                f"{opening}\\begin{{longtable}}{{{sized}}}"
                f"{paged_body}\\end{{longtable}}{closing}"
            )
        if n_cols < 4 or any(len(c) > _MAX_SHORT_CELL for c in cells):
            return f"{opening}\\begin{{tabular}}{{{sized}}}{body}\\end{{tabular}}{closing}"
        return f"\\begin{{tabular}}{{{'l' * n_cols}}}{body}\\end{{tabular}}"

    return _TABULAR.sub(replace_table, content)


_MERMAID_INCLUDEGRAPHICS = re.compile(
    r"\\includegraphics\[width=[\d.]+\\linewidth\]\{(files/mermaid-[^}]+)\}"
)


def resize_mermaid_figures(content: str) -> str:
    """Apply one fixed size rule to every Mermaid figure instead of a bare
    width.

    MyST emits ``\\includegraphics[width=0.7\\linewidth]`` for every
    ``{figure}`` directive that doesn't set an explicit ``:width:``, with no
    height limit. Mermaid diagrams span a wide range of aspect ratios (tall
    stacked flowcharts vs. wide single-row ones), so a fixed width alone
    makes tall diagrams enormous -- tall enough to overflow the page and
    push the caption below the printable area, where it silently
    disappears -- while wide diagrams shrink to an unreadable strip.
    Cap both dimensions and let ``keepaspectratio`` pick the binding one:
    diagrams use the full text width when their height allows it, and
    shrink to fit within 45% of the text height otherwise, which always
    leaves room on the page for the caption below it.
    """
    return _MERMAID_INCLUDEGRAPHICS.sub(
        r"\\includegraphics[width=\\linewidth,height=0.45\\textheight,"
        r"keepaspectratio]{\1}",
        content,
    )


_FIGURE_INCLUDEGRAPHICS = re.compile(
    r"\\includegraphics\[width=([\d.]+)\\linewidth\]\{((?!files/mermaid-)[^}]+)\}"
)


def cap_figure_heights(content: str) -> str:
    """Give every other figure a height limit as well as its width.

    A chart with many stacked panels (a faceted effect plot is 549 x 3,390
    pixels) is about 1,700 pt tall at 70% of the line width, twice the page.
    pdfLaTeX then shipped out pages without end until it ran out of
    destination names at page 498,589 and produced no PDF. Capping the height
    at 85% of the text height with ``keepaspectratio`` shrinks only such
    figures; the rest keep their width.
    """
    return _FIGURE_INCLUDEGRAPHICS.sub(
        r"\\includegraphics[width=\1\\linewidth,height=0.85\\textheight,"
        r"keepaspectratio]{\2}",
        content,
    )


def reject_legacy_mermaid_captions(content: str) -> None:
    """Fail rather than publishing raw graph syntax as explanatory prose."""
    markers = (r"\caption[]{flowchart", r"\caption[]{sequenceDiagram")
    for marker in markers:
        if marker in content:
            raise ValueError(
                "raw Mermaid source reached a LaTeX caption; "
                "repair the notebook figure directive and mermaid_diagrams.json"
            )


_MINTED_BLOCK = re.compile(r"(\\begin\{minted\}.*?\\end\{minted\})", re.DOTALL)
_TEXT_ARROW = re.compile(r"(?<!\\ensuremath\{)\\(?:right|left)arrow")


def ensure_text_arrows(content: str) -> str:
    """Make MyST's Unicode arrows valid in text and math-mode table cells.

    MyST emits a bare ``\\rightarrow`` for some Unicode arrows inside HTML
    tables, while prose arrows get ``$\\rightarrow$``. ``\\ensuremath`` works
    in either context. Minted blocks are excluded so source code stays literal.
    """
    chunks = _MINTED_BLOCK.split(content)
    for index in range(0, len(chunks), 2):
        chunks[index] = _TEXT_ARROW.sub(
            lambda match: r"\ensuremath{" + match.group(0) + "}", chunks[index]
        )
    return "".join(chunks)


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_tex.py <tex_dir>")
        sys.exit(1)

    tex_dir = Path(sys.argv[1])
    if not tex_dir.is_dir():
        print(f"ERROR: {tex_dir} is not a directory")
        sys.exit(1)

    svg_count = 0
    minted_count = 0
    table_count = 0
    resize_count = 0
    arrow_count = 0
    svg_ref = re.compile(r"([\w./-]+)\.svg\}")

    def repoint(m):
        path = m.group(1)
        png = tex_dir / f"{path}.png"
        ext = "png" if "mermaid-" in path and png.is_file() else "pdf"
        return f"{path}.{ext}}}"

    for f in sorted(tex_dir.glob("mboostrs-*.tex")):
        content = f.read_text("utf-8")

        new = svg_ref.sub(repoint, content)
        if new != content:
            svg_count += 1

        new2 = patch_verbatim_to_minted(new)
        if new2 != new:
            minted_count += 1

        new3 = repair_generated_tables(new2)
        if new3 != new2:
            table_count += 1

        new4 = cap_figure_heights(resize_mermaid_figures(new3))
        if new4 != new3:
            resize_count += 1

        new5 = ensure_text_arrows(new4)
        if new5 != new4:
            arrow_count += 1
        reject_legacy_mermaid_captions(new5)

        if new5 != content:
            f.write_text(new5, "utf-8")

    print(f"  patched {svg_count} chapter files (.svg -> .pdf)")
    print(f"  repaired tables in {table_count} chapter files (width and pagination)")
    print(f"  converted {minted_count} chapter files (verbatim -> minted)")
    print(f"  resized figures in {resize_count} chapter files")
    print(f"  repaired text arrows in {arrow_count} chapter files")


if __name__ == "__main__":
    main()
