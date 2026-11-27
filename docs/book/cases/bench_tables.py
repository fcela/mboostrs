"""Annotated great_tables tables for the benchmark pages.

The PDF build converts a table's HTML into a LaTeX tabular and keeps text
and bold but not cell fills, so the best value in a column is marked in bold
inside the cell and nothing a reader needs is carried by colour alone.
"""

import polars as pl
from great_tables import GT, loc, style


def bold_best(gt, frame, better, decimals=3, formats=None):
    """Format ``better``'s columns to ``decimals`` places, the best in bold.

    ``better`` maps a column name to ``"max"``, ``"min"`` or a number, the
    target a value is best closest to; ``formats`` may give a column its own
    number of decimals. Ties at the displayed precision are all bold.
    """
    for column, direction in better.items():
        values = frame[column].drop_nulls()
        if values.is_empty():
            continue
        best = _best(values, direction)
        text = f"{{:.{(formats or {}).get(column, decimals)}f}}"

        def cell(value, best=best, text=text):
            if value is None:
                return "–"
            shown = text.format(value)
            return f"<strong>{shown}</strong>" if shown == text.format(best) else shown

        gt = gt.fmt(cell, columns=column)
    return gt


def _best(values, direction):
    """The best of ``values``: largest, smallest, or closest to a target."""
    if direction == "max":
        return values.max()
    if direction == "min":
        return values.min()
    return min(values.to_list(), key=lambda v: abs(v - float(direction)))


def shade_mboostrs(gt, column="method"):
    """Shade mboostrs rows on the site (the PDF shows no fills)."""
    return gt.tab_style(style.fill(color="#e8f0fb"),
                        loc.body(rows=pl.col(column).str.starts_with("mboostrs")))


def table(frame, *, title, subtitle, stub="method", stubhead="Method", labels=None,
          spanners=None, better=None, better_row=None, better_row_cols=None,
          decimals=3, formats=None, notes=(), group=None):
    """An annotated comparison table.

    ``labels`` renames columns; ``spanners`` maps a spanner label to its
    columns; ``better`` maps each scored column to ``"max"`` or ``"min"``
    and puts that column's best value in bold (per ``group`` when given, so
    each block of rows has its own best); ``better_row`` (``"min"`` or
    ``"max"``) puts the best value in each row in bold across
    ``better_row_cols`` (or all non-stub, non-group columns); ``formats`` maps
    a column to its number of decimals for columns that are not scored;
    ``notes`` are source notes, one per paragraph.
    """
    better = better or {}
    row_cols = (
        list(better_row_cols) if better_row_cols is not None
        else [c for c in frame.columns if c not in (stub, group)]
    ) if better_row is not None else []
    gt = GT(frame, rowname_col=stub, groupname_col=group).tab_header(
        title=title, subtitle=subtitle).tab_stubhead(stubhead)
    for label, columns in (spanners or {}).items():
        gt = gt.tab_spanner(label, columns)
    if labels:
        gt = gt.cols_label(labels)
    formats = dict(formats or {})
    # Every other float column that is not scored gets the table's decimals.
    for column, dtype in frame.schema.items():
        if dtype.is_float() and column not in better and column not in row_cols and column not in formats:
            formats[column] = decimals
    for column, digits in formats.items():
        if column not in better and column not in row_cols:
            gt = gt.fmt_number(columns=column, decimals=digits)
    if group is None:
        gt = bold_best(gt, frame, better, decimals, formats)
    else:
        gt = _bold_best_by_group(gt, frame, better, decimals, group, formats)
    if better_row is not None and row_cols:
        gt = _bold_best_by_row(gt, frame, better_row, decimals, formats, row_cols)
    gt = gt.sub_missing(missing_text="–")
    if stub in frame.columns and frame[stub].dtype == pl.String:
        gt = shade_mboostrs(gt, stub)
    for note in notes:
        gt = gt.tab_source_note(note)
    return gt


def _as_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _bold_best_by_row(gt, frame, direction, decimals, formats, cols):
    """Format ``cols`` and put each row's best numeric value in bold."""
    for i in range(frame.height):
        nums = []
        raw_vals = {}
        for col in cols:
            val = frame[col][i]
            raw_vals[col] = val
            num = _as_float(val)
            if num is not None:
                nums.append(num)
        best = _best(pl.Series(nums), direction) if nums else None
        for col in cols:
            val = raw_vals[col]
            dtype = frame.schema[col]
            if val is None:
                shown = "–"
            elif dtype.is_float():
                text = f"{{:.{formats.get(col, decimals)}f}}"
                s = text.format(val)
                shown = f"<strong>{s}</strong>" if best is not None and s == text.format(best) else s
            else:
                s = str(val)
                num = _as_float(val)
                is_best = best is not None and num is not None and abs(num - best) <= 1e-12
                shown = f"<strong>{s}</strong>" if is_best else s
            gt = gt.fmt(lambda _v, shown=shown: shown, columns=col, rows=[i])
    return gt


def _bold_best_by_group(gt, frame, better, decimals, group, formats):
    """``bold_best`` with the best taken within each ``group`` of rows."""
    for column, direction in better.items():
        text = f"{{:.{formats.get(column, decimals)}f}}"
        best = {g: _best(part[column].drop_nulls(), direction)
                for (g,), part in frame.group_by(group)
                if part[column].drop_nulls().len() > 0}
        marked = [
            None if v is None
            else (f"<strong>{text.format(v)}</strong>"
                  if best.get(g) is not None and text.format(v) == text.format(best[g])
                  else text.format(v))
            for g, v in zip(frame[group], frame[column], strict=True)
        ]
        # fmt sees one value at a time and not its row, so format by position.
        for i, shown in enumerate(marked):
            gt = gt.fmt(lambda _v, shown=shown: "–" if shown is None else shown,
                        columns=column, rows=[i])
    return gt
