"""Column names inside formula text: quoting, unquoting and masking.

A formula names a data column either as a plain identifier (``x``, ``house.age``)
or, for any other name, in backticks as R does: ``bbs(`house age`)``,
``bols(`price/sqft`)``. Inside backticks a backslash escapes the next
character, so a name containing a backtick is written ``\\```, and a name
containing a backslash is written ``\\\\``; ``\\n`` and ``\\t`` stand for a
newline and a tab. This is the rule of R's parser and of ``deparse``.
"""

from __future__ import annotations

import re

#: Regex source matching one column name in formula text: a plain identifier or
#: a backticked name with backslash escapes.
NAME_SRC = r"(?:[\w.]+|`(?:[^`\\]|\\.)+`)"

#: Regex source for a name that starts like an identifier (not a digit), as used
#: by the patterns that scan text for columns.
IDENT_SRC = r"(?:[A-Za-z_][\w.]*|`(?:[^`\\]|\\.)+`)"

_SYNTACTIC = re.compile(r"^(?:[A-Za-z_]|\.(?![0-9]))[\w.]*$")

#: Words that a bare name cannot be, because the formula language gives them a
#: meaning: the learner constructors, the transforms, R's response wrappers and
#: R's reserved words and constants.
RESERVED = frozenset({
    "bols", "bbs", "bmono", "btree", "bmrf", "bspatial", "brandom", "bns",
    "bkernel", "bconstrained", "buser", "brad", "hinge",
    "I", "log", "log1p", "sqrt", "exp", "scale", "center", "ecdf", "pmin",
    "pmax", "Surv", "cbind",
    "if", "else", "repeat", "while", "function", "for", "next", "break",
    "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "in",
    "True", "False", "None",
})

_ESCAPES = {"n": "\n", "t": "\t"}


def needs_quoting(name: str) -> bool:
    """Whether ``name`` must be written in backticks in formula text."""
    return not _SYNTACTIC.match(name) or name in RESERVED


def quote_name(name: str) -> str:
    """``name`` as it is written in formula text: bare when it is a plain
    identifier, otherwise in backticks with ``\\``, backtick, newline and tab
    escaped."""
    name = str(name)
    if not needs_quoting(name):
        return name
    body = (name.replace("\\", "\\\\").replace("`", "\\`")
            .replace("\n", "\\n").replace("\t", "\\t"))
    return f"`{body}`"


def unquote_name(token: str) -> str:
    """The column name a formula token stands for: a backticked token loses its
    backticks and its escapes; any other token is returned stripped."""
    token = token.strip()
    if len(token) >= 2 and token[0] == "`" and token[-1] == "`":
        out, i, body = [], 0, token[1:-1]
        while i < len(body):
            ch = body[i]
            if ch == "\\" and i + 1 < len(body):
                nxt = body[i + 1]
                out.append(_ESCAPES.get(nxt, nxt))
                i += 2
            else:
                out.append(ch)
                i += 1
        return "".join(out)
    return token


def mask_names(text: str) -> str:
    """``text`` with the inside of every backticked name (and the backticks)
    replaced by ``x``, so the result has the same length and no operator,
    bracket or comma that belonged to a name. Positions found in the masked
    text are valid in the original."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "`":
            j = i + 1
            while j < n and text[j] != "`":
                j += 2 if text[j] == "\\" else 1
            end = min(j, n - 1)
            out.append("x" * (end - i + 1))
            i = end + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def split_top_level(text: str, seps: str) -> list[tuple[str, str]]:
    """Split ``text`` at bracket depth 0 on any character of ``seps`` outside
    backticks. Returns ``(separator, piece)`` pairs; the first separator is
    ``""``."""
    masked = mask_names(text)
    out, depth, start, sep = [], 0, 0, ""
    for i, ch in enumerate(masked):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch in seps and depth == 0:
            out.append((sep, text[start:i]))
            sep, start = ch, i + 1
    out.append((sep, text[start:]))
    return out


def backtick_hint(fragment: str) -> str | None:
    """A backticked spelling of ``fragment`` when it looks like a column name
    written without backticks (spaces, operators, a leading digit, a reserved
    word), else ``None``."""
    frag = fragment.strip()
    if not frag or "`" in frag or "=" in frag:
        return None
    if re.fullmatch(r"[\w.]+", frag) and not needs_quoting(frag):
        return None
    return quote_name(frag) if needs_quoting(frag) else f"`{frag}`"


def unparsed_term_message(term: str) -> str:
    """The error for a formula term that did not parse, naming the backticked
    spelling when the term reads like an unescaped column name."""
    base = f"cannot parse formula term: {term!r}"
    stripped = term.strip()
    m = re.match(r"^(\w+)\s*\((.*)\)$", stripped, re.DOTALL)
    candidate = None
    if m:
        inner = split_top_level(m.group(2), ",")[0][1]
        hint = backtick_hint(inner)
        if hint is not None:
            candidate = (inner.strip(), f"{m.group(1)}({hint}"
                         + (m.group(2)[len(inner):] if len(m.group(2)) > len(inner) else "")
                         + ")")
    elif backtick_hint(stripped) is not None:
        candidate = (stripped, quote_name(stripped))
    if candidate is None:
        return base
    raw, fixed = candidate
    return (f"{base}. If {raw!r} is a column name, write it in backticks "
            f"({fixed}): names with spaces, operators, a leading digit or a "
            f"reserved word need backticks.")


def partition_tilde(formula: str) -> tuple[str, str, str]:
    """``str.partition("~")`` that ignores a ``~`` inside a backticked name."""
    i = mask_names(formula).find("~")
    if i < 0:
        return formula, "", ""
    return formula[:i], "~", formula[i + 1:]


_EXPRESSION_RESPONSE = re.compile(
    r"^\s*(?:Surv|cbind|log|log1p|sqrt|exp|scale|center)\s*\(.*\)\s*$", re.DOTALL)


def quote_response(response: str) -> str:
    """The response side of a generated formula: a column name is quoted when it
    needs it; ``Surv(...)``, ``cbind(...)`` and transform calls are returned as
    written."""
    text = str(response)
    if _EXPRESSION_RESPONSE.match(text):
        return text
    return quote_name(text)


def split_terms(rhs: str) -> list[str]:
    """The terms of a right-hand side, split on ``+`` outside brackets and
    backticked names."""
    return [piece.strip() for _, piece in split_top_level(rhs, "+") if piece.strip()]
