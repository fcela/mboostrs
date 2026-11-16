"""Search-wide constraints on the specifications a search may build.

Three constraints apply to the interaction and term choices of
:func:`~mboostrs.autospec`, :class:`~mboostrs.SearchSession` and
:func:`~mboostrs.discover`:

``heredity``
    When a term that reads two or more columns jointly may stand in a
    specification. Such a term is an interaction: a ``%X%`` tensor, a
    ``by=`` varying coefficient, or a ``btree`` over several columns.
    ``"strong"`` requires a main-effect term for every column it reads, in
    the same specification. ``"weak"`` requires one. ``"none"`` requires
    nothing.
``forbidden_pairs``
    Column pairs that no term may read together, in any specification.
``mandatory_terms``
    Columns whose main-effect term every specification keeps.

The functions here decide admission from the terms of a specification and
hold no fitted state. Each search reads them when it opens and records the
rule in force in its own report.
"""

from __future__ import annotations

import re

from mboostrs.explore import split_terms
from mboostrs.formula import term_variables

__all__ = ["HEREDITY_RULES", "DEFAULT_RULE"]

#: The values ``heredity`` accepts.
HEREDITY_RULES = ("none", "weak", "strong")

#: The rule a search applies when no constraint is given.
DEFAULT_RULE = {"heredity": "none", "forbidden_pairs": [], "mandatory_terms": []}

_BY_ARGUMENT = re.compile(r"\bby\s*=")


def joint_columns(term):
    """The columns a term reads jointly, or ``None`` for a term that reads one.

    A term is joint when it multiplies or conditions one column on another:
    a ``%X%`` tensor, a ``by=`` argument, or a ``btree`` over two or more
    columns. ``bspatial(lon, lat)`` and ``brandom(group)`` are not joint:
    a two-coordinate smooth is a main effect of its coordinates.
    """
    text = str(term)
    names = term_variables(text)
    if len(names) < 2:
        return None
    if "%X%" in text or _BY_ARGUMENT.search(text) or text.lstrip().startswith("btree("):
        return frozenset(names)
    return None


def main_columns(terms):
    """The columns with a main-effect term among ``terms``.

    Every column a non-joint term reads counts as having its main effect.
    """
    out = set()
    for term in terms:
        if joint_columns(term) is None:
            out.update(term_variables(term))
    return out


def admits(columns, mains, heredity):
    """Whether the rule lets a joint term over ``columns`` stand.

    ``mains`` is the set of columns with a main-effect term in the same
    specification.
    """
    if heredity == "strong":
        return frozenset(columns) <= mains
    if heredity == "weak":
        return bool(frozenset(columns) & mains)
    return True


def heredity_violations(terms, heredity):
    """The joint terms among ``terms`` that the heredity rule does not admit."""
    if heredity == "none":
        return []
    mains = main_columns(terms)
    out = []
    for term in terms:
        columns = joint_columns(term)
        if columns is not None and not admits(columns, mains, heredity):
            out.append(term)
    return out


def forbidden_hits(terms, forbidden_pairs):
    """``(term, pair)`` for each term that reads both columns of a forbidden pair."""
    hits = []
    for term in terms:
        names = set(term_variables(term))
        for pair in forbidden_pairs:
            if pair <= names:
                hits.append((term, pair))
    return hits


def pair_label(pair):
    """A forbidden pair as ``"a x b"``, columns in sorted order."""
    return " x ".join(sorted(pair))


def check_heredity(value, name="heredity"):
    """The validated heredity rule."""
    if value not in HEREDITY_RULES:
        raise ValueError(f'{name} must be "none", "weak" or "strong", got {value!r}')
    return value


def check_forbidden_pairs(pairs, columns, response, name="forbidden_pairs"):
    """Validated pairs as a list of frozensets, in the order given, without repeats."""
    out = []
    for item in pairs or ():
        try:
            left, right = item
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} entries must be two column names, such as ('x1', 'x2'); "
                f"got {item!r}") from None
        left, right = str(left), str(right)
        if left == right:
            raise ValueError(f"{name} entry {item!r} names one column twice; "
                             "a pair needs two different columns")
        for column in (left, right):
            if column == response:
                raise ValueError(f"{name} names the response {response!r}; "
                                 "a pair must join two predictors")
            if column not in columns:
                raise ValueError(f"{name} names {column!r}, which is not a column of the "
                                 "data; pass the name of a predictor column")
        pair = frozenset((left, right))
        if pair not in out:
            out.append(pair)
    return out


def check_mandatory(names, columns, response, name="mandatory_terms"):
    """Validated column names as a sorted tuple, without repeats."""
    if isinstance(names, str):
        names = (names,)
    out = set()
    for column in names or ():
        column = str(column)
        if column == response:
            raise ValueError(f"{name} names the response {response!r}; "
                             "name a predictor column")
        if column not in columns:
            raise ValueError(f"{name} names {column!r}, which is not a column of the "
                             "data; pass the name of a predictor column")
        out.add(column)
    return tuple(sorted(out))


def rule_record(heredity, forbidden_pairs, mandatory_terms):
    """The rule in force as plain data, for state, transcripts and checkpoints."""
    return {"heredity": heredity,
            "forbidden_pairs": sorted(sorted(pair) for pair in forbidden_pairs),
            "mandatory_terms": sorted(mandatory_terms)}


def missing_mains(terms, mandatory_terms):
    """The mandatory columns that ``terms`` gives no main-effect term."""
    return sorted(set(mandatory_terms) - main_columns(terms))


def rhs_terms(rhs):
    """The top-level terms of a right-hand side string."""
    return split_terms(rhs or "")
