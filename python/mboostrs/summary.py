"""Model summaries in text and HTML.

A boosted fit is described by more than its coefficients: the family and link
that define the loss, the number of iterations and the step length that stopped
it there, how far the risk moved from the offset-only model, and which base
learners the path actually spent its updates on. :func:`build_summary`
assembles those into a :class:`Summary`, which renders as fixed-width text in a
terminal and as a table in a notebook.

The layout follows `statsmodels`' results summaries: a two-column header of
model and fit facts, then one or more tables, then notes. The contents differ
because the quantities differ — a boosted model has no standard errors on its
coefficients, and the honest substitutes are how often a learner was selected
and when it first entered.
"""

from __future__ import annotations

import html
import textwrap
from datetime import UTC, datetime

import numpy as np

__all__ = ["Summary", "SummaryTable", "build_summary", "compare_summaries"]

TEXT_WIDTH = 78


def _fmt(value, decimals=4):
    """Format one cell for display.

    Integers print without a decimal point, floats to a fixed width, and
    anything non-numeric passes through as its string form.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):d}"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return str(v)
        if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-4):
            return f"{v:.{decimals}g}"
        return f"{v:.{decimals}f}"
    return str(value)


def _plain(value):
    """A cell value as a JSON-native Python object.

    NumPy scalars become ``int``, ``float`` or ``bool``. Any other type that
    JSON cannot encode is written as its string form.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


class SummaryTable:
    """A titled table of rows, rendered as text or HTML.

    `align` gives one of "l" or "r" per column; numeric columns read better
    right-aligned and names left-aligned. Cells keep their values in `rows`;
    they are formatted only when the table is rendered, so `as_dict()` holds
    numbers.
    """

    def __init__(self, title, headers, rows, align=None, note=None):
        self.title = title
        self.headers = list(headers)
        self.rows = [list(row) for row in rows]
        self.align = list(align) if align else ["l"] + ["r"] * (len(self.headers) - 1)
        self.note = note

    def _cells(self):
        return [[_fmt(c) for c in row] for row in self.rows]

    def widths(self):
        cols = [len(h) for h in self.headers]
        for row in self._cells():
            for i, cell in enumerate(row):
                cols[i] = max(cols[i], len(cell))
        return cols

    def as_text(self):
        widths = self.widths()
        def line(cells):
            out = []
            for cell, w, a in zip(cells, widths, self.align, strict=False):
                out.append(cell.rjust(w) if a == "r" else cell.ljust(w))
            return "  ".join(out).rstrip()

        parts = []
        if self.title:
            parts.append(self.title)
        parts.append(line(self.headers))
        parts.append("-" * min(TEXT_WIDTH, max(len(line(self.headers)), 20)))
        parts.extend(line(r) for r in self._cells())
        if self.note:
            parts.append("")
            parts.extend(textwrap.wrap(self.note, width=TEXT_WIDTH))
        return "\n".join(parts)

    def as_html(self):
        head = "".join(
            f'<th style="text-align:{"right" if a == "r" else "left"};'
            f'padding:2px 10px;border-bottom:1px solid currentColor">{html.escape(h)}</th>'
            for h, a in zip(self.headers, self.align, strict=False)
        )
        body = []
        for row in self._cells():
            cells = "".join(
                f'<td style="text-align:{"right" if a == "r" else "left"};'
                f'padding:2px 10px">{html.escape(c)}</td>'
                for c, a in zip(row, self.align, strict=False)
            )
            body.append(f"<tr>{cells}</tr>")
        title = (
            f'<caption style="text-align:left;font-weight:600;padding:6px 0">'
            f"{html.escape(self.title)}</caption>"
            if self.title
            else ""
        )
        note = (
            f'<div style="font-size:0.85em;opacity:0.75;padding:4px 0">'
            f"{html.escape(self.note)}</div>"
            if self.note
            else ""
        )
        return (
            '<table style="border-collapse:collapse;font-variant-numeric:tabular-nums;'
            f'margin:8px 0">{title}<thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table>{note}"
        )

    def as_dict(self):
        rows = [[_plain(c) for c in row] for row in self.rows]
        return {"title": self.title, "headers": self.headers, "rows": rows}


class Summary:
    """A model summary that prints as text and displays as HTML.

    Returned by every model's ``summary()``. Printing it, or evaluating it at a
    prompt, gives the text form; a notebook renders the HTML form.
    """

    def __init__(self, title, header_pairs, tables=None, notes=None):
        self.title = title
        self.header_pairs = list(header_pairs)
        self.tables = list(tables or [])
        self.notes = list(notes or [])

    def as_text(self):
        lines = [self.title.center(TEXT_WIDTH).rstrip(), "=" * TEXT_WIDTH]

        # Two key/value pairs per line, statsmodels-style. A value too long to
        # sit in a half-width column takes a line of its own; pairing it would
        # widen every other row to match it.
        rendered = [(k, _fmt(v)) for k, v in self.header_pairs]
        key_w = max((len(k) for k, _ in rendered), default=0) + 1
        # Two columns of "key: value" plus a three-space gutter must fit the
        # width, which fixes the widest value a paired row can hold.
        wide_value = max(8, (TEXT_WIDTH - 3 - 2 * (key_w + 1)) // 2)
        pairs = [(k, v) for k, v in rendered if len(v) <= wide_value]
        singles = [(k, v) for k, v in rendered if len(v) > wide_value]

        half = (len(pairs) + 1) // 2
        left, right = pairs[:half], pairs[half:]
        val_w = max((len(v) for _, v in pairs), default=0)
        for i in range(half):
            lk, lv = left[i]
            cell = f"{lk + ':':<{key_w}} {lv:>{val_w}}"
            if i < len(right):
                rk, rv = right[i]
                cell = f"{cell}   {rk + ':':<{key_w}} {rv:>{val_w}}"
            lines.append(cell.rstrip())
        for k, v in singles:
            prefix = f"{k + ':':<{key_w}} "
            wrapped = textwrap.wrap(v, width=TEXT_WIDTH - len(prefix)) or [""]
            lines.append(prefix + wrapped[0])
            lines.extend(" " * len(prefix) + w for w in wrapped[1:])
        lines.append("=" * TEXT_WIDTH)

        for table in self.tables:
            lines.append(table.as_text())
            lines.append("=" * TEXT_WIDTH)
        for note in self.notes:
            lines.extend(textwrap.wrap(note, width=TEXT_WIDTH))
        return "\n".join(lines)

    def as_html(self):
        pairs = "".join(
            f'<tr><td style="padding:1px 12px 1px 0;opacity:0.75">{html.escape(str(k))}</td>'
            f'<td style="padding:1px 24px 1px 0;text-align:right;'
            f'font-variant-numeric:tabular-nums">{html.escape(_fmt(v))}</td></tr>'
            for k, v in self.header_pairs
        )
        notes = "".join(
            f'<div style="font-size:0.85em;opacity:0.75;padding:2px 0">{html.escape(n)}</div>'
            for n in self.notes
        )
        return (
            '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
            'font-size:0.88em;line-height:1.45">'
            f'<div style="font-weight:600;padding-bottom:6px">{html.escape(self.title)}</div>'
            f'<table style="border-collapse:collapse">{pairs}</table>'
            + "".join(t.as_html() for t in self.tables)
            + notes
            + "</div>"
        )

    def as_dict(self):
        return {
            "title": self.title,
            "header": dict(self.header_pairs),
            "tables": [t.as_dict() for t in self.tables],
            "notes": list(self.notes),
        }

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return self.as_text()

    def _repr_html_(self):
        return self.as_html()


# ---------------------------------------------------------------------------
# Shared quantities
# ---------------------------------------------------------------------------

def _risk_facts(model):
    """Offset-only risk, final risk, and the fraction of the gap closed.

    ``risk[0]`` is the risk of the offset-only model, before any learner is
    added, so it is the baseline any fit has to beat.
    """
    risk = np.asarray(model._risk, dtype=float)
    if risk.size == 0:
        return None, None, None
    start, end = float(risk[0]), float(risk[min(model.mstop, risk.size - 1)])
    reduction = (start - end) / abs(start) if start != 0 else float("nan")
    return start, end, reduction


def _selection_counts(model, mstop=None):
    """Times each learner was updated, and the iteration it first entered."""
    stop = model.mstop if mstop is None else mstop
    names = model.base_learner_names()
    counts, first = {}, {}
    for iteration, index in enumerate(model._selected[:stop], start=1):
        name = names[index]
        counts[name] = counts.get(name, 0) + 1
        first.setdefault(name, iteration)
    return counts, first


def _offset_text(offset):
    """The offset for the summary table: a number for a scalar offset, and the
    row count and mean of a per-observation offset (log exposure)."""
    if isinstance(offset, np.ndarray):
        return f"per observation (n={offset.size}, mean {float(np.mean(offset)):.4g})"
    return float(offset)


def _family_name(family):
    return type(family).__name__


def _predict_scale_row(model):
    """A header row naming the default ``predict()`` scale (#229).

    The default prediction is link-scale; on a non-identity-link family that
    is not the response, so the summary names the scale next to the family
    instead of leaving the reader to remember it.
    """
    from mboostrs.families import prediction_scale_text

    return ("Default predict scale", prediction_scale_text(model._family))


def _cv_table(model, cv):
    """A cross-validation block for a fit, given the `cvrisk` result for it.

    The comparison worth making is between the iteration the model stopped at
    and the iteration cross-validation prefers. A model carries no record of
    how it was tuned, so `cv` has to be supplied; nothing here infers it.
    """
    from mboostrs.cvrisk import _optimum_index

    aggregate = np.asarray(cv.aggregate, dtype=float)
    matrix = np.asarray(cv.risk_matrix, dtype=float)
    grid = np.asarray(cv.grid)
    best = _optimum_index(aggregate, grid)
    n_folds = matrix.shape[0]

    at_best = matrix[:, best]
    standard_error = float(at_best.std(ddof=1) / np.sqrt(n_folds)) if n_folds > 1 else float("nan")

    # Where the model actually sits on the cross-validated curve.
    position = int(np.argmin(np.abs(grid - model.mstop)))
    risk_here = float(aggregate[position])
    risk_best = float(aggregate[best])
    excess = risk_here - risk_best

    rows = [
        ["Folds", f"{n_folds} ({cv.fold_type})"],
        ["Optimal mstop", int(cv.optimal_mstop)],
        ["CV risk at optimum", risk_best],
        ["Standard error across folds", standard_error],
        ["Model mstop", int(model.mstop)],
        ["CV risk at model mstop", risk_here],
    ]
    note = None
    if int(cv.optimal_mstop) != int(model.mstop):
        direction = "past" if model.mstop > cv.optimal_mstop else "short of"
        in_se = (
            " — within one standard error of the optimum"
            if standard_error == standard_error and excess <= standard_error
            else ""
        )
        note = (
            f"The model stops {direction} the cross-validated optimum "
            f"({model.mstop} against {int(cv.optimal_mstop)}), costing "
            f"{excess:+.4g} in cross-validated risk{in_se}."
        )
    return SummaryTable("Cross-validation", ["quantity", "value"], rows,
                        align=["l", "r"], note=note)


def _common_header(model, kind, n_learners, n_selected):
    start, end, reduction = _risk_facts(model)
    weights = getattr(model, "_weights", None)
    n = len(model._fitted) if getattr(model, "_fitted", None) is not None else None
    pairs = [
        ("Model", kind),
        ("Family", _family_name(model._family)),
        _predict_scale_row(model),
        ("No. observations", n),
        ("Base learners", n_learners),
        ("Learners selected", n_selected),
        ("Iterations (mstop)", model.mstop),
        ("Step length (nu)", float(model._nu)),
        ("Offset", _offset_text(getattr(model, "_offset", float("nan")))),
        ("Risk at mstop", end),
        ("Risk at offset", start),
        ("Risk reduction", f"{reduction:.1%}" if reduction == reduction else ""),
    ]
    omitted = len(getattr(model, "_na_dropped", []) or [])
    if omitted:
        pairs.append(("Rows dropped (NA)", omitted))
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.size and not np.allclose(w, w.flat[0]):
            pairs.append(("Weights", "non-constant"))
    pairs.append(("Date", datetime.now(UTC).strftime("%Y-%m-%d")))
    return pairs


def _selection_table(model, mstop=None, title="Base learners"):
    stop = model.mstop if mstop is None else mstop
    names = list(model._learner_names)
    types = getattr(model, "_learner_types", None)
    counts = np.bincount(np.asarray(model._selected[:stop], dtype=int),
                         minlength=len(names))
    first: dict = {}
    for iteration, index in enumerate(model._selected[:stop], start=1):
        first.setdefault(index, iteration)

    # One row per learner, not per name: `bbs(x) + bmono(x)` must not have its
    # updates merged into a single row carrying one of the two kinds.
    ordered = sorted(
        (i for i in range(len(names)) if counts[i] > 0),
        key=lambda i: (-counts[i], first.get(i, stop + 1), i),
    )
    repeated = {name for name in names if names.count(name) > 1}
    specs = getattr(model, "_learners_spec", None)
    labels = model.base_learner_names() if specs is not None else None
    headers = ["learner", "kind", "updates", "% of path", "first iter"]
    rows = []
    for i in ordered:
        label = names[i]
        kind = types[i] if types is not None and i < len(types) else ""
        if names[i] in repeated:
            tag = kind if kind else f"#{i}"
            if labels is not None:
                # Keep by= and ordinal suffixes from the same labels that
                # identify coefficient, effect and extraction entries.
                prefix = f"{names[i]} ("
                if labels[i].startswith(prefix):
                    tag, _, ordinal = labels[i][len(prefix):].rpartition(")")
                    label = f"{names[i]} [{tag}]{ordinal}"
                else:
                    label = labels[i]
            else:
                label = f"{names[i]} [{tag}]"
        rows.append([label, kind, int(counts[i]), f"{100.0 * counts[i] / stop:.1f}%", first[i]])
    if types is None:
        headers.pop(1)
        rows = [[r[0], *r[2:]] for r in rows]
    never = [n for n in dict.fromkeys(names)
             if all(counts[i] == 0 for i, x in enumerate(names) if x == n)
             and n != "(Intercept)"]
    note = None
    if never:
        shown = ", ".join(never[:6]) + (" ..." if len(never) > 6 else "")
        note = f"Never selected ({len(never)}): {shown}"
    return SummaryTable(title, headers, rows, note=note)


# ---------------------------------------------------------------------------
# Per-model-type builders
# ---------------------------------------------------------------------------

def _glm_summary(model, cv=None):
    coefs = model.coef()
    counts, _ = _selection_counts(model)
    n_selected = len(counts)
    header = _common_header(model, "glmboost", len(model._learner_names), n_selected)
    header.insert(2, ("Response", getattr(model, "_response_name", "") or ""))

    _, first = _selection_counts(model)
    nonzero = {k: v for k, v in coefs.items() if abs(v) > 1e-12}
    rows = []
    for name, value in sorted(nonzero.items(), key=lambda kv: -abs(kv[1])):
        count = counts.get(name)
        rows.append([
            name,
            float(value),
            count if count is not None else "",
            f"{100.0 * count / model.mstop:.1f}%" if count else "",
            first.get(name, ""),
        ])
    never = [n for n in model._learner_names
             if n not in counts and n != "(Intercept)"]
    note = None
    if never:
        shown = ", ".join(never[:8]) + (" ..." if len(never) > 8 else "")
        note = f"Never selected ({len(never)}): {shown}"
    tables = [
        SummaryTable(
            "Coefficients",
            ["term", "coef", "updates", "% of path", "first iter"],
            rows, note=note,
        ),
    ]
    if cv is not None:
        tables.append(_cv_table(model, cv))
    notes = [
        "Coefficients are the sum of the shrunken updates the path applied to "
        "each term; they carry no standard error, because the term entered the "
        "model by the same data that estimated it.",
        *_parity_note_lines(model),
    ]
    return Summary(f"{_family_name(model._family)} boosting results", header, tables, notes)


def _parity_note_lines(model):
    """One summary line per deviation, so a reader meets them where they read.

    A deviation recorded only in a generated appendix is one a user has to
    think to look for, and the point of these is that the user does not know
    to. The full statement -- mechanism, consequence, whether upstream shares
    it -- is `model.parity_notes()`; this is the pointer.
    """
    try:
        from mboostrs._parity_notes import notes_for
    except ImportError:
        return []
    notes = notes_for(model)
    if not notes:
        return []
    # A note with an `effect` says in one sentence what it changes in this
    # fit's output, so it is printed in full; the rest are named and pointed to.
    lines = [f"{note.name}: {note.effect}. parity_notes() has the detail."
             for note in notes if note.effect]
    pointed = [note for note in notes if not note.effect]
    if pointed:
        named = "; ".join(note.name for note in pointed)
        lines.append(
            f"This fit carries {len(pointed)} documented parity note(s): {named}. "
            f"Call parity_notes() for what each one does and what follows from it.")
    return lines


def _gam_summary(model, cv=None):
    counts, _ = _selection_counts(model)
    n_selected = len(counts)
    header = _common_header(model, "gamboost", len(model._learner_names), n_selected)
    header.insert(2, ("Response", getattr(model, "_response_name", "") or ""))

    tables = [_selection_table(model)]

    # A smooth learner has no single coefficient; the range of its fitted
    # effect is the comparable scalar.
    #
    # `partial_effects` refuses when a learner's effect is not a curve over one
    # covariate -- a tree, a user-supplied design -- and that refusal is a fact
    # about the model, so the table simply omits the column. Anything else is
    # a defect, and swallowing it here turned a broken `partial_effects` into a
    # silently shorter summary.
    try:
        effects = model.partial_effects()
    except (NotImplementedError, ValueError, KeyError):
        effects = {}
    if effects:
        rows = []
        for name, (_x, effect) in effects.items():
            effect = np.asarray(effect, dtype=float)
            rows.append([
                name,
                float(effect.min()),
                float(effect.max()),
                float(effect.max() - effect.min()),
            ])
        rows.sort(key=lambda r: -r[3])
        tables.insert(0, SummaryTable(
            "Fitted effects (on the training data)",
            ["learner", "min", "max", "range"], rows,
            note="Range is the span of the learner's contribution to the "
                 "linear predictor, the comparable scalar when a learner has "
                 "no single coefficient.",
        ))
    if cv is not None:
        tables.append(_cv_table(model, cv))
    notes = [
        "Effects are fitted values, not estimates with standard errors; "
        "confint() bootstraps them and plot_confint() draws the bands.",
        *_parity_note_lines(model),
    ]
    return Summary(f"{_family_name(model._family)} boosting results", header, tables, notes)


def _lss_summary(model, cv=None):
    families = getattr(model, "_families", None) or getattr(model, "_family", None)
    risk = np.asarray(model._risk, dtype=float)
    start = float(risk[0]) if risk.size else float("nan")
    mstop_val = model.mstop
    if isinstance(mstop_val, dict):
        step_idx = (
            sum(int(v) for v in mstop_val.values())
            if getattr(model, "_method", None) == "noncyclic"
            else max((int(v) for v in mstop_val.values()), default=0)
        )
        mstop_display = ", ".join(f"{k}={v}" for k, v in mstop_val.items())
    else:
        step_idx = int(mstop_val)
        mstop_display = mstop_val
    end = float(risk[min(step_idx, risk.size - 1)]) if risk.size else float("nan")
    reduction = (start - end) / abs(start) if risk.size and start != 0 else float("nan")
    nu_per_param = getattr(model, "_nu_per_param", None)
    if nu_per_param is not None and len(set(nu_per_param)) > 1:
        nu_display = ", ".join(
            f"{name}={float(rate):.4g}"
            for name, rate in zip(model.param_names, nu_per_param, strict=False)
        )
    else:
        nu_display = float(model._nu)

    header = [
        ("Model", getattr(model, "_model_kind", "gamboostLSS")),
        ("Family", _family_name(families)),
        _predict_scale_row(model),
        ("Parameters", ", ".join(model.param_names)),
        ("Offsets", ", ".join(
            f"{name}={float(model._params[k]['offset']):.4g}"
            for k, name in enumerate(model.param_names)
        )),
        # `_lss_summary` is only reached for an LSS fit, which always has
        # `fitted()`; the guard here read as though it might not.
        ("No. observations",
         len(np.asarray(model.fitted()[model.param_names[0]]))),
        ("Iterations (mstop)", mstop_display),
        ("Step length (nu)", nu_display),
        ("Risk at mstop", end),
        ("Risk at offset", start),
        ("Risk reduction", f"{reduction:.1%}" if reduction == reduction else ""),
        ("Date", datetime.now(UTC).strftime("%Y-%m-%d")),
    ]

    tables = []
    for k, pname in enumerate(model.param_names):
        params = model._params[k]
        if "learners" in params:
            names = [ln.name for ln in params["learners"]]
        else:
            names = ["(Intercept)", *params.get("predictor_names", ())]
        counts, first = {}, {}
        for iteration, index in enumerate(params["selected"], start=1):
            name = names[index]
            counts[name] = counts.get(name, 0) + 1
            first.setdefault(name, iteration)
        n_updates = len(params["selected"])
        denom = n_updates or 1
        rows = [
            [name, count, f"{100.0 * count / denom:.1f}%", first[name]]
            for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
        never = [n for n in names if n not in counts]
        note = (
            f"Never selected for {pname}: " + ", ".join(never[:6])
            if never else None
        )
        tables.append(SummaryTable(
            f"Parameter {pname} — {n_updates} updates",
            ["learner", "updates", "% of parameter's path", "first iter"],
            rows, note=note,
        ))

    if cv is not None:
        tables.append(_cv_table(model, cv))
    notes = [
        "Each distribution parameter has its own selection path; the model "
        "cycles between them, so an update spent on one is not spent on "
        "another.",
        *_parity_note_lines(model),
    ]
    return Summary(f"{_family_name(families)} distributional boosting results",
                   header, tables, notes)


def build_summary(model, cv=None):
    """Build the :class:`Summary` appropriate to a fitted model.

    `cv` is an optional :class:`~mboostrs.cvrisk.CvRiskResult` for this model;
    supplying it adds the cross-validation block, which compares the iteration
    the model stopped at with the one cross-validation prefers.
    """
    from mboostrs._dispatch import is_lss

    if is_lss(model):
        return _lss_summary(model, cv=cv)
    if getattr(model, "_learner_types", None) is not None:
        return _gam_summary(model, cv=cv)
    return _glm_summary(model, cv=cv)


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def _model_kind(model):
    from mboostrs._dispatch import is_lss

    if is_lss(model):
        return "gamboostLSS"
    if getattr(model, "_learner_types", None) is not None:
        return "gamboost"
    return "glmboost"


def _comparison_facts(model, cv=None):
    """The scalar facts about one fit, keyed by row label."""
    kind = _model_kind(model)
    facts = {"Model": kind}

    if kind == "gamboostLSS":
        families = getattr(model, "_families", None) or model._family
        facts["Family"] = _family_name(families)
        facts["Parameters"] = ", ".join(model.param_names)
        risk = np.asarray(model._risk, dtype=float)
        start = float(risk[0]) if risk.size else float("nan")
        end = float(risk[min(model.mstop, risk.size - 1)]) if risk.size else float("nan")
    else:
        facts["Family"] = _family_name(model._family)
        counts, _ = _selection_counts(model)
        facts["Base learners"] = len(model._learner_names)
        facts["Learners selected"] = len(counts)
        start, end, _ = _risk_facts(model)
        facts["Observations"] = len(model._fitted)

    facts["Iterations (mstop)"] = model.mstop
    facts["Step length (nu)"] = float(model._nu)
    facts["Risk at mstop"] = end
    if start is not None and start == start and start != 0:
        facts["Risk reduction"] = f"{(start - end) / abs(start):.1%}"
    if cv is not None:
        aggregate = np.asarray(cv.aggregate, dtype=float)
        facts["CV risk"] = float(aggregate.min())
        facts["CV optimal mstop"] = int(cv.optimal_mstop)
    return facts


def compare_summaries(*models, labels=None, cv=None):
    """Put several fitted models side by side as one :class:`Summary`.

    Each model becomes a column: the first table holds the scalar facts, the
    second how many updates each base learner received under each model. A
    learner absent from a model shows an empty cell, so the rows line up
    across models fitted on different formulas.

    Parameters
    ----------
    *models
        Fitted models. They need not be the same kind; rows that do not apply
        to a model are left blank.
    labels : list of str, optional
        Column headings. Defaults to "model 1", "model 2", ...
    cv : list, optional
        One :class:`~mboostrs.cvrisk.CvRiskResult` per model, or None for a
        model that has none. Adds the cross-validated risk and optimal
        iteration to the facts table.

    Returns
    -------
    Summary

    Notes
    -----
    The models must be fitted to the same rows, response and effective
    weights, or this raises ``ValueError``: their risks and criteria would
    describe different data. Models that share their data but differ in
    response scale (``log(y)`` against ``y``) or weight policy are shown
    side by side with a note that they are not ranked against each other.
    The risks are comparable across models only when the family and the data
    are the same; a Gaussian risk and a Binomial risk are different losses.
    ``cv`` results must all come from the same folds.
    """
    from mboostrs._comparison_identity import check_comparison, check_cv_folds

    if not models:
        raise ValueError("compare_summaries needs at least one model")
    if labels is None:
        labels = [f"model {i}" for i in range(1, len(models) + 1)]
    if len(labels) != len(models):
        raise ValueError(
            f"got {len(labels)} labels for {len(models)} models"
        )
    if cv is None:
        cv = [None] * len(models)
    elif len(cv) != len(models):
        raise ValueError(f"got {len(cv)} cv results for {len(models)} models")
    comparison = check_comparison(models, helper="compare_summaries")
    if any(c is not None for c in cv):
        check_cv_folds(cv, helper="compare_summaries")

    facts = [_comparison_facts(m, c) for m, c in zip(models, cv, strict=False)]

    # Row order follows the first model, then any row only later ones have.
    row_labels = []
    for f in facts:
        for key in f:
            if key not in row_labels:
                row_labels.append(key)

    fact_rows = [
        [label] + [f.get(label, "") for f in facts] for label in row_labels
    ]
    tables = [
        SummaryTable("Fit", ["", *labels], fact_rows,
                     align=["l"] + ["r"] * len(labels))
    ]

    # Update counts per learner, for the models that have named learners.
    per_model = []
    for model in models:
        if _model_kind(model) == "gamboostLSS":
            counts = {}
            for k in range(len(model.param_names)):
                params = model._params[k]
                names = [ln.name for ln in params["learners"]]
                for index in params["selected"]:
                    counts[names[index]] = counts.get(names[index], 0) + 1
        else:
            counts, _ = _selection_counts(model)
        per_model.append(counts)

    learner_names = []
    for counts in per_model:
        for name in counts:
            if name not in learner_names:
                learner_names.append(name)
    if learner_names:
        # Most-used first, summed across models, so the rows a reader cares
        # about are at the top.
        learner_names.sort(key=lambda n: -sum(c.get(n, 0) for c in per_model))
        rows = [
            [name] + [c.get(name, "") for c in per_model] for name in learner_names
        ]
        tables.append(SummaryTable(
            "Updates per base learner", ["learner", *labels], rows,
            align=["l"] + ["r"] * len(labels),
            note="Blank means the learner is not in that model's formula, or "
                 "was never selected.",
        ))

    header = [("Models compared", len(models))]
    families = {f.get("Family") for f in facts}
    header.append(("Family", families.pop() if len(families) == 1 else "mixed"))
    observations = {f.get("Observations") for f in facts if "Observations" in f}
    if len(observations) == 1:
        header.append(("Observations", observations.pop()))
    header.append(("Date", datetime.now(UTC).strftime("%Y-%m-%d")))

    notes = []
    if len(families) > 1 or "mixed" in dict(header).values():
        notes.append(
            "The models do not share a family, so their risks are different "
            "losses and are not comparable with each other."
        )
    if comparison.verdict == "segregate":
        notes.append("Not ranked against each other: " + "; ".join(comparison.reasons) + ".")
    return Summary("Model comparison", header, tables, notes)
