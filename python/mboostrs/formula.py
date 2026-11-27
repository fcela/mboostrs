import difflib
import re
from typing import NoReturn

from mboostrs._errors import MboostrsValueError
from mboostrs._names import (
    NAME_SRC as N,
)
from mboostrs._names import (
    mask_names,
    partition_tilde,
    quote_name,
    split_top_level,
    unparsed_term_message,
    unquote_name,
)
from mboostrs.learners import (
    Bbs,
    Bconstrained,
    Bkernel,
    Bmono,
    Bmrf,
    Bns,
    Bols,
    Brad,
    Brandom,
    Bspatial,
    Btree,
    Buser,
    Hinge,
)

_LEARNER_CALL_RE = re.compile(
    r"^(bols|bbs|bmono|btree|bmrf|bspatial|brandom|bns|bkernel|bconstrained|buser|brad|hinge)"
    rf"\s*\(\s*({N})"
)

#: The only learner a linear formula can name. `glmboost` fits one linear term
#: per predictor, so `bols(x)` is a synonym for a bare `x` and is accepted as a
#: migration affordance for readers coming from `mboost`. Every other learner
#: describes a basis expansion `glmboost` cannot fit.
_LINEAR_LEARNER = "bols"


def _extract_varname(term: str) -> str:
    """The column a term refers to, for any learner.

    Used by `.` expansion, which needs to know which columns a formula already
    names regardless of how they are being fitted, so it accepts every learner.
    `parse_formula` applies the linear-only restriction on top of this.
    """
    m = _LEARNER_CALL_RE.match(term)
    if m:
        return unquote_name(m.group(2))
    if re.match(rf"^{N}$", term):
        return unquote_name(term)
    raise ValueError(unparsed_term_message(term))


def _linear_varname(term: str) -> str:
    """The column a term refers to, rejecting learners `glmboost` cannot fit.

    A linear formula reduced `bbs(x, df=4)` — or `bkernel(x)`, or `btree(x)` —
    to the bare name `x` and fitted a linear term, so
    `glmboost("y ~ bbs(x, df=4)")` returned a fit identical to
    `glmboost("y ~ x")` with nothing to say the spline had been discarded.
    Upstream `mboost::glmboost` rejects every learner call, `bols` included,
    because it passes the formula to `model.frame`; mboostrs keeps `bols` as a
    migration affordance because it means exactly what a bare name means.
    """
    m = _LEARNER_CALL_RE.match(term)
    if m and m.group(1) != _LINEAR_LEARNER:
        learner, varname = m.group(1), unquote_name(m.group(2))
        raise ValueError(
            f"{learner}({varname}) expands {varname} into a basis; glmboost and "
            f"glmboostLSS fit one linear term per predictor and cannot use it. "
            f"Use gamboost or gamboostLSS for {learner}() terms, or "
            f"bols({varname}) / a bare {varname} for a linear term."
        )
    return _extract_varname(term)


def no_response_message(formula: str) -> str:
    """The message for a formula string that has no ``~``, so no response.

    The remedy is the same formula with a response column on the left of
    the ``~``: a right-hand side such as ``"bbs(x) + bols(z)"`` becomes
    ``"y ~ bbs(x) + bols(z)"``.
    """
    rhs = formula.strip()
    return (f"formula must contain '~': {formula!r} has no response, because "
            f"nothing is named on the left of a '~'. Give the response column "
            f"first, as 'y ~ {rhs}' (y is the name of the response column)")


def require_response(formula, caller: str | None = None) -> None:
    """Raise ``ValueError`` when a formula string has no response (no ``~``).

    A dict (a distributional specification) and a non-string are left to the
    caller. `caller` is named in the message when given.
    """
    if not isinstance(formula, str) or "~" in formula:
        return
    prefix = f"{caller}: " if caller else ""
    raise ValueError(prefix + no_response_message(formula))


def _split_formula(formula: str) -> tuple[str, list[str]]:
    tilde = mask_names(formula).find("~")
    if tilde < 0:
        raise ValueError(no_response_message(formula))
    parts = [formula[:tilde], formula[tilde + 1:]]
    response = parts[0].strip()
    if not response:
        raise ValueError("formula must have a response variable before '~'")
    rhs = parts[1].strip()
    terms = [piece.strip() for _, piece in split_top_level(rhs, "+") if piece.strip()]
    if not terms:
        raise ValueError(
            "formula must have at least one predictor after '~'; for a model "
            "with the intercept only (every term dropped by a selection step, "
            "say) write 'y ~ 1'. gamboost() and glmboost() fit it as an "
            "offset-only model, and gamboostLSS() and glmboostLSS() fit the "
            "intercept as a parameter learner")
    return response, terms


_SURV_RE = re.compile(rf"^Surv\s*\(\s*({N})\s*,\s*({N})\s*\)$")
_CBIND_RE = re.compile(rf"^cbind\s*\(\s*({N}(?:\s*,\s*{N})+)\s*\)$")


def _parse_response(response: str) -> str | tuple[str, ...]:
    """Parse a formula response: a plain name, ``Surv(time, status)`` (2-tuple),
    or ``cbind(y1, y2, ...)`` (K-tuple, for multivariate families)."""
    m_surv = _SURV_RE.match(response)
    if m_surv:
        return (unquote_name(m_surv.group(1)), unquote_name(m_surv.group(2)))
    m_cbind = _CBIND_RE.match(response)
    if m_cbind:
        return tuple(unquote_name(piece) for _, piece in split_top_level(m_cbind.group(1), ","))
    return unquote_name(response) if response.startswith("`") else response


def parse_response(formula: str) -> str | tuple[str, ...]:
    """The response side alone: a name, a ``Surv(time, status)`` pair, or a
    ``cbind(...)`` tuple.

    `parse_formula` also reads the right-hand side, and its linear-learner
    rules refuse `bbs()`/`brandom()` terms; a caller that only needs the
    response (naming the column, matching a recipe step) must not pay that
    check.
    """
    response, _terms = _split_formula(formula)
    return _parse_response(response)


def parse_formula(formula: str) -> tuple[str | tuple[str, ...], list[str]]:
    """Split a linear formula into its response and predictor names.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"``. A ``Surv(time, status)`` response returns as a tuple
        of the two column names.

    Returns
    -------
    tuple
        ``(response, predictors)``, where `response` is a column name or a
        tuple for a survival response, and `predictors` is a list of names.

    Examples
    --------
    >>> parse_formula("y ~ a + b")
    ('y', ['a', 'b'])
    """
    response, terms = _split_formula(formula)
    response = _parse_response(response)
    predictor_names = [_linear_varname(t) for t in terms]
    return response, predictor_names


# --------------------------------------------------------------------------
# `.` expansion and term removal
# --------------------------------------------------------------------------


_IDENTIFIER = re.compile(r"[A-Za-z_][\w.]*")
_NOT_COLUMNS = frozenset({"TRUE", "FALSE", "True", "False", "None", "NULL", "Inf", "NA"})


def term_variables(term: str) -> tuple[str, ...]:
    """The data columns a formula term reads, in order of first appearance.

    ``bbs(x1, df=4)`` reads ``x1``; ``bols(x2, by=z)`` reads ``x2`` and ``z``;
    ``bbs(log(x5))`` reads ``x5``; ``bbs(a) %X% bbs(b)`` reads ``a`` and ``b``.
    A name followed by ``(`` (a learner or a transform) or by ``=`` (an
    argument name) is not a column, and neither are quoted strings.
    """
    ticked = {}

    def _stash(match):
        ticked[f"__tick{len(ticked)}__"] = unquote_name(match.group(0))
        return f" __tick{len(ticked) - 1}__ "

    text = re.sub(r"`(?:[^`\\]|\\.)+`", _stash, str(term))
    text = re.sub(r"(\"[^\"]*\"|'[^']*'|%\w+%)", " ", text)
    # These arguments take enum tokens, not column names. A real column
    # called `sse` still counts in `bols(sse)`; only `selection=sse` is skipped.
    text = re.sub(
        r"\b(?:selection|constraint|split_method|knot_placement)\s*=\s*[A-Za-z_]\w*",
        " ",
        text,
    )
    out = []
    for match in _IDENTIFIER.finditer(text):
        name = ticked.get(match.group(0), match.group(0))
        rest = text[match.end():].lstrip()
        if rest.startswith("(") or (rest.startswith("=") and not rest.startswith("==")):
            continue
        if (name in _NOT_COLUMNS and match.group(0) not in ticked) or name in out:
            continue
        out.append(name)
    return tuple(out)


def _signed_terms(rhs: str) -> list[tuple[str, str]]:
    """Split a right-hand side into ``(sign, term)`` pairs at bracket depth 0."""
    terms: list[tuple[str, str]] = []
    sign = "+"
    for separator, piece in split_top_level(rhs, "+-"):
        term = piece.strip()
        if separator:
            sign = separator
        if term:
            terms.append((sign, term))
            sign = "+"
    return terms


def _intercept_only(formula: str, data) -> bool:
    """Whether the right-hand side is the intercept ``1`` and nothing else.

    ``y ~ 1`` is an offset-only model: every row is predicted by the family's
    offset and no base learner is fitted. A column actually named ``1`` is a
    predictor, not the intercept.
    """
    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        return False
    return _signed_terms(rhs) == [("+", "1")] and "1" not in data_columns(data)


def split_intercept(formula: str, data) -> tuple[str, bool]:
    """Remove R's intercept terms from a formula: ``(formula, intercept)``.

    ``- 1``, ``+ 0`` and a leading ``0 +`` drop the intercept, as R's
    ``model.matrix`` does; ``+ 1`` keeps it. A column actually named ``0``
    or ``1`` stays a column. The returned formula has the terms removed, so
    the rest of the parser never sees them, except for the intercept-only
    form ``y ~ 1`` (the intercept alone), which is returned as ``y ~ 1``.
    """
    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        return formula, True
    columns = set(data_columns(data))
    kept, intercept = [], True
    if not _signed_terms(rhs):
        return formula, True
    for sign, term in _signed_terms(rhs):
        if term in ("0", "1") and term not in columns:
            if (sign, term) in (("-", "1"), ("+", "0")):
                intercept = False
            elif (sign, term) == ("-", "0"):
                intercept = True
            continue
        kept.append((sign, term))
    if not kept and _intercept_only(formula, data):
        return f"{lhs.strip()} ~ 1", True
    if not kept:
        raise ValueError(f"formula {formula!r} has no predictors once the intercept term is removed")
    body = " ".join(f"{sign} {term}" for sign, term in kept).lstrip("+ ").strip()
    return f"{lhs.strip()} ~ {body}", intercept


#: Intercept removal: ``- 1`` and ``+ 0``. Only glmboost honours these.
_INTERCEPT_REMOVAL = (("-", "1"), ("+", "0"))

#: Terms that request the intercept a function fits anyway: ``+ 1`` and ``- 0``.
_INTERCEPT_REDUNDANT = (("+", "1"), ("-", "0"))

#: Every bare ``0`` / ``1`` term that `expand_formula` refuses, ``+ 1`` and
#: ``- 0`` included.
_INTERCEPT_ANY = (("-", "1"), ("+", "0"), ("+", "1"), ("-", "0"))

#: What each fitting function does with the intercept and how to fit without
#: one, for the refusal message.
_INTERCEPT_INSTEAD = {
    "gamboost": "gamboost fits the mean of y as an offset and each base-learner carries "
                "its own intercept, so there is no intercept term to remove. To fit "
                "without an intercept, use bols() terms: gamboost('y ~ bols(x1) + "
                "bols(x2)') reproduces R's glmboost(y ~ x1 + x2 - 1). "
                "bols(x, intercept=False) is a different operation: it drops the "
                "intercept column from inside that one term's design and changes the fit",
    "gam": "gam always fits an intercept as part of the penalized fit, and it is not a "
           "formula term. To fit without an intercept, use glmboost()",
    "gamlss": "gamlss fits an intercept for every distribution parameter, and a parameter "
              "with no formula has an intercept only. To fit without an intercept, use "
              "glmboost()",
    "gamboostLSS": "gamboostLSS fits an intercept for every distribution parameter",
    "glmboostLSS": "glmboostLSS fits an intercept for every distribution parameter",
}


class InterceptControlError(MboostrsValueError, NotImplementedError):
    """Intercept removal (``- 1``, ``+ 0``) passed to a fitting function that does not honour it.

    It is a ``ValueError``, because the formula is the argument to change, and a
    ``NotImplementedError``, so code that caught the earlier refusal still does.
    """


def refuse_intercept_control(formula: str, data, frontend: str, *,
                             forms=_INTERCEPT_REMOVAL) -> None:
    """Raise `InterceptControlError` if `formula` has a term from `forms`.

    The shared check for every fitting function that cannot remove the
    intercept. ``frontend`` is the function's name, used in the message. A
    column actually named ``0`` or ``1`` is a predictor and is not checked.
    ``+ 1`` and ``- 0`` request the intercept the function fits anyway; they
    are refused with a message that says to delete the term.
    """
    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        return
    columns = set(data_columns(data))
    for sign, term in _signed_terms(rhs):
        if term not in ("0", "1") or term in columns or (sign, term) not in forms:
            continue
        if (sign, term) in _INTERCEPT_REMOVAL:
            instead = _INTERCEPT_INSTEAD.get(frontend, f"{frontend} fits an intercept")
            message = (f"intercept control ('{sign} {term}') in {formula!r} is supported "
                       f"by glmboost only. {instead}.")
        else:
            message = (f"'{sign} {term}' in {formula!r} requests an intercept, which "
                       f"{frontend} always fits; delete the term.")
        raise InterceptControlError(message, code="intercept_control", subject=frontend)


def data_columns(data) -> list[str]:
    """Column names of a dict / pandas / polars frame, in order."""
    if isinstance(data, dict):
        return list(data.keys())
    columns = getattr(data, "columns", None)
    if columns is None:
        raise TypeError(f"cannot list columns of {type(data)}")
    return [str(c) for c in columns]


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][\w.]*")


def _formula_columns(formula: str, data) -> list[str]:
    """Data columns referenced anywhere in `formula` (response and predictors).

    Any identifier that is also a column name counts; learner names like `bbs`
    that are not columns are ignored. Used to scope the missing-value check to
    the variables the model actually reads.
    """
    present = set(data_columns(data))
    seen: list[str] = []
    tokens = [unquote_name(t) for t in re.findall(r"`(?:[^`\\]|\\.)+`", formula)]
    tokens += _IDENTIFIER_RE.findall(re.sub(r"`(?:[^`\\]|\\.)+`", " ", formula))
    for token in tokens:
        if token in present and token not in seen:
            seen.append(token)
    return seen


def _row_is_missing(values) -> "list[bool]":
    """Per-row missingness for one column: NaN/Inf for numeric, null for object."""
    import numpy as np

    array = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    if array.dtype.kind in "fc":
        return ~np.isfinite(array)
    if array.dtype.kind in "iu":
        return np.zeros(array.shape[0], dtype=bool)
    # object / string: treat None and NaN floats as missing
    return np.array(
        [v is None or (isinstance(v, float) and v != v) for v in array.tolist()],
        dtype=bool,
    )


def apply_na_action(formula: str, data, na_action: str):
    """Handle missing values in the columns `formula` uses.

    * ``"error"`` (default): leave `data` unchanged; the downstream finite check
      raises, naming the column and row.
    * ``"omit"``: drop every row with a missing value in a referenced column and
      return ``(filtered_data, dropped_row_indices)``.

    Returns ``(data, dropped_indices)``; ``dropped_indices`` is empty for
    ``"error"`` and for a clean ``"omit"``.
    """
    import numpy as np

    if na_action not in ("error", "omit", "impute", "impute_model", "impute_model_split"):
        raise ValueError(
            "na_action must be 'error', 'omit', 'impute', 'impute_model' or 'impute_model_split', "
            f"got {na_action!r}")
    columns = _formula_columns(formula, data)
    if na_action == "error":
        # The finite check downstream covers numeric columns; a label column
        # would otherwise fail while its levels are sorted, naming no column.
        from mboostrs.transforms import _is_categorical

        # A categorical response's missing label is the family's to refuse.
        lhs = partition_tilde(formula)[0]
        for name in columns:
            if re.search(rf"\b{re.escape(name)}\b", lhs):
                continue
            if _is_categorical(data, name):
                count = int(_row_is_missing(data[name]).sum())
                if count:
                    raise ValueError(
                        f"column {name!r} has {count} missing value(s). Pass "
                        f"na_action='impute' to give them a level of their "
                        f"own, '(missing)', or na_action='omit' to drop those "
                        f"rows")
        return data, []

    if na_action in ("impute", "impute_model", "impute_model_split"):
        # Predictors are imputed by the transform recipe; only a missing
        # response drops its row.
        lhs = partition_tilde(formula)[0]
        columns = [c for c in columns if re.search(rf"\b{re.escape(c)}\b", lhs)]
    if not columns:
        return data, []

    n = None
    missing = None
    for name in columns:
        col_missing = _row_is_missing(
            data[name] if not isinstance(data, dict) else data[name]
        )
        if missing is None:
            n = col_missing.shape[0]
            missing = np.zeros(n, dtype=bool)
        missing |= col_missing

    dropped = np.flatnonzero(missing).tolist()
    if not dropped:
        return data, []
    if len(dropped) == n:
        raise ValueError(
            "na_action='omit' would drop every row: every observation has a "
            "missing value in at least one model variable"
        )

    keep = ~missing
    if isinstance(data, dict):
        filtered = {
            k: (np.asarray(v)[keep] if np.ndim(v) else v) for k, v in data.items()
        }
    elif hasattr(data, "filter") and not hasattr(data, "iloc"):  # polars
        # pandas also has `.filter`, which selects columns by label; the
        # `iloc` check keeps a pandas frame out of this branch.
        import polars as pl

        filtered = data.filter(pl.Series(keep))
    else:  # pandas
        filtered = data.loc[keep].reset_index(drop=True)
    return filtered, dropped


_INNER_DOT_RE = re.compile(r"^(\w+)\s*\(\s*\.\s*(,\s*.*)?\)$", re.DOTALL)


def expand_formula(formula: str, data) -> str:
    """Resolve ``.`` and ``- term`` against the data, returning a plain formula.

    * ``y ~ .`` becomes every column except the response.
    * ``y ~ . - z`` drops ``z`` from that expansion.
    * ``y ~ . + bbs(z, df=4)`` expands ``.`` to the remaining columns and keeps
      the explicit term, so a single predictor can be given a different learner
      without naming all the others.

    Formulas containing neither ``.`` nor a ``-`` are returned unchanged, so
    this is a no-op for existing code.
    """
    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        raise ValueError(no_response_message(formula))
    if _intercept_only(formula, data):
        return formula

    signed = _signed_terms(rhs)
    columns = set(data_columns(data))
    # `0`/`1` mean the intercept only when no column is actually named that. A
    # frame with a column "0" (sklearn's compliance checks build exactly that)
    # must keep referring to the column.
    refuse_intercept_control(formula, data, "gamboost", forms=_INTERCEPT_ANY)
    if (
        not any(term == "." for _, term in signed)
        and not any(_INNER_DOT_RE.match(term) for _, term in signed)
        and all(sign == "+" for sign, _ in signed)
    ):
        return formula

    response = _parse_response(lhs.strip())
    response_names = {response} if isinstance(response, str) else set(response)

    kept: list[str] = []
    removed: set[str] = set()
    dot_positions: list[int] = []
    explicit_vars: set[str] = set()

    inner_dot: list[int] = []
    for sign, term in signed:
        if term == ".":
            if sign == "-":
                raise ValueError("'- .' is not a valid term")
            dot_positions.append(len(kept))
            kept.append(".")
            continue
        # `btree(.)` — a dot inside a learner call means every remaining
        # predictor as arguments to that one learner, not one learner each.
        # It is how a tree over all predictors is written without naming them:
        # `y ~ btree(.)` is R's `blackboost(y ~ ., data)`.
        if _INNER_DOT_RE.match(term):
            if sign == "-":
                raise ValueError(f"'- {term}' is not a valid term")
            inner_dot.append(len(kept))
            kept.append(term)
            continue
        if sign != "-" and re.match(r"^brad\s*\(", term):
            explicit_vars.update(name for name in term_variables(term) if name in columns)
        name = _extract_varname(term)
        if sign == "-":
            removed.add(name)
        else:
            explicit_vars.add(name)
            kept.append(term)

    if dot_positions or inner_dot:
        expansion = [
            c
            for c in data_columns(data)
            if c not in response_names and c not in explicit_vars and c not in removed
        ]
        if not expansion:
            raise ValueError(
                "'.' expanded to no predictors; the data has no columns beyond "
                "the response and the terms already named"
            )
        # A column holding one value has no effect a learner could fit, and a
        # spline over it has no range to place knots in. `.` passes over it;
        # naming it explicitly still reaches the learner and its error.
        constant = [c for c in expansion if _is_constant(data, c)]
        if constant and len(constant) < len(expansion):
            import warnings

            warnings.warn(
                f"'.' leaves out {', '.join(map(repr, constant))}: "
                f"{'it holds' if len(constant) == 1 else 'each holds'} a single "
                "value in these rows, so no learner can fit an effect",
                UserWarning, stacklevel=3)
            expansion = [c for c in expansion if c not in constant]
        # Splice the expansion in where the dot appeared.
        out: list[str] = []
        for term in kept:
            if term == ".":
                out.extend(quote_name(c) for c in expansion)
                continue
            match = _INNER_DOT_RE.match(term)
            if match:
                rest = match.group(2) or ""
                out.append(f"{match.group(1)}({', '.join(quote_name(c) for c in expansion)}{rest})")
                continue
            out.append(term)
        kept = out
    else:
        kept = [t for t in kept if _extract_varname(t) not in removed]

    if not kept:
        raise ValueError("formula has no predictors left after removals")

    return f"{lhs.strip()} ~ " + " + ".join(kept)


def _is_constant(data, name: str) -> bool:
    """Whether a column's non-missing values are all one value."""
    import numpy as np

    from mboostrs._frame import read_column

    try:
        values = np.asarray(read_column(data, name).values)
        if values.dtype.kind == "f":
            values = values[~np.isnan(values)]
        return values.size > 0 and np.unique(values).size == 1
    except (TypeError, ValueError, KeyError):
        return False


_BBS_CALL_RE = re.compile(rf"^bbs\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BOLS_CALL_RE = re.compile(rf"^bols\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BMONO_CALL_RE = re.compile(rf"^bmono\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BTREE_CALL_RE = re.compile(rf"^btree\s*\(\s*({N}(?:\s*[+,]\s*{N})*)\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_HINGE_CALL_RE = re.compile(rf"^hinge\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BRAD_CALL_RE = re.compile(rf"^brad\s*\(\s*({N}(?:\s*,\s*{N})*)\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BMRF_CALL_RE = re.compile(rf"^bmrf\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BRANDOM_CALL_RE = re.compile(rf"^brandom\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BSPATIAL_CALL_RE = re.compile(
    rf"^bspatial\s*\(\s*({N})\s*,\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL
)
_BNS_CALL_RE = re.compile(rf"^bns\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BKERNEL_CALL_RE = re.compile(rf"^bkernel\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BCONSTRAINED_CALL_RE = re.compile(rf"^bconstrained\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)
_BUSER_CALL_RE = re.compile(rf"^buser\s*\(\s*({N})\s*(?:,\s*(.*))?\s*\)$", re.DOTALL)

_KWARG_RE = re.compile(
    r"""([\w.]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(c\s*\([^)]*\))|([a-zA-Z]\w*\s*\([^)]*\))"""
    r"""|(`(?:[^`\\]|\\.)+`|[^,)]+))""")


def _kwarg_value(m) -> str:
    if m.group(2) is not None:
        return m.group(2)
    if m.group(3) is not None:
        return m.group(3)
    if m.group(4) is not None:
        return m.group(4).strip()
    if m.group(5) is not None:
        return m.group(5).strip()
    return unquote_name(m.group(6))


def _is_bare_identifier(m) -> bool:
    """True if the kwarg value was an unquoted symbol (not a number, string, or c())."""
    return m.group(6) is not None


def _resolve_env_symbol(
    symbol: str, param: str, learner: str, env: dict | None
):
    """Resolve *symbol* from *env* for the given *param* in *learner*.

    Raises ``ValueError`` with a diagnostic message if *env* is ``None`` or
    *symbol* is not found.
    """
    if env is None:
        raise ValueError(
            f"{learner}(..., {param}={symbol}) requires an env= argument to resolve "
            f"'{symbol}'. Pass env={{{symbol!r}: ...}} or use "
            f"learners=[{learner.capitalize()}('...', {param}={symbol})] instead."
        )
    if symbol not in env:
        raise ValueError(
            f"{learner}(..., {param}={symbol}) could not resolve the symbol "
            f"'{symbol}' in env. Pass env={{{symbol!r}: <array>}} or construct "
            f"the learner explicitly."
        )
    return env[symbol]


def _kwarg_matches(kwargs_str: str, learner: str):
    """The keyword assignments in ``kwargs_str``, proving nothing was ignored.

    Each learner parser used to loop ``_KWARG_RE.finditer`` directly, which
    reports what it matched and says nothing about what it did not. Three
    things slipped through, all silently:

    * **Extra positional variables.** ``bbs(x, y, z)`` matched the term regex
      with ``x`` as the variable and ``"y, z"`` as the keyword string, where no
      keyword matched -- so the learner was ``bbs(x)`` and ``y`` and ``z``
      vanished. A caller writing a multi-column term got a single-column fit
      with nothing to indicate it.
    * **Duplicate keywords.** ``bbs(x, df=3, df=4)`` took the last, quietly.
    * **Trailing text inside the parentheses**, for the same reason as the
      first.

    This checks that the matches tile the string, so anything the regex did
    not consume is reported against the term the caller wrote.
    """
    matches = list(_KWARG_RE.finditer(kwargs_str))

    seen: dict[str, int] = {}
    for match in matches:
        key = match.group(1).strip()
        if key in seen:
            raise ValueError(
                f"{learner}(...) got {key}= more than once. Remove the "
                f"duplicate; mboostrs does not pick one silently."
            )
        seen[key] = match.start()

    position = 0
    leftovers = []
    for match in matches:
        leftovers.append(kwargs_str[position:match.start()])
        position = match.end()
    leftovers.append(kwargs_str[position:])

    for fragment in leftovers:
        residue = fragment.strip().strip(",").strip()
        if residue:
            raise ValueError(
                f"{learner}(...) has text mboostrs did not understand: "
                f"{residue!r}. Each argument after the variable must be "
                f"name=value. A learner over several columns is written as one "
                f"term per column, or as buser() with an explicit design."
            )
    return matches


#: Spellings accepted for a boolean argument, covering Python's and R's.
_TRUE_LITERALS = frozenset({"True", "TRUE", "true", "T"})
_FALSE_LITERALS = frozenset({"False", "FALSE", "false", "F"})


def _kwarg_bool(value: str, key: str, learner: str) -> bool:
    """A boolean argument, or a refusal naming the spellings that work.

    ``val in ("True", "TRUE", "true")`` mapped every other string to ``False``,
    so ``bbs(x, center=maybe)`` turned centering off without a word -- and
    centering changes the fit.
    """
    text = value.strip()
    if text in _TRUE_LITERALS:
        return True
    if text in _FALSE_LITERALS:
        return False
    raise ValueError(
        f"{learner}(..., {key}={text}) is not a boolean. Write True or False "
        f"(TRUE/FALSE are accepted for R users)."
    )


_LEARNER_PARAMETERS: dict[str, tuple[str, ...]] = {
    "bbs": (
        "bins", "knots", "degree", "differences", "df", "lambda",
        "lambda_value", "center", "cyclic", "boundary.knots", "constraint",
        "deriv", "by", "knot_placement",
    ),
    "bols": ("df", "bins", "lambda", "lambda_value", "intercept", "by"),
    "bmono": (
        "knots", "degree", "differences", "niter", "df", "lambda2",
        "constraint", "type", "by",
    ),
    "btree": (
        "minbucket", "minsplit", "maxdepth", "by", "selection",
        "split_method", "lambda_leaf", "colsample", "colsample_seed",
    ),
    "hinge": ("df", "n_candidates", "lambda", "lambda_value", "center", "by"),
    "bmrf": ("K", "df", "lambda", "lambda_value", "center", "by"),
    "brandom": ("df", "bins", "lambda", "lambda_value", "center", "by"),
    "bspatial": (
        "bins", "knots", "degree", "differences", "df", "lambda",
        "lambda_value", "center", "by",
    ),
    "bns": ("knots", "degree", "df", "center", "by"),
    "bkernel": ("bandwidth", "df", "kernel", "landmarks", "center", "by"),
    "bconstrained": (
        "C", "knots", "degree", "differences", "niter", "df", "lambda2",
        "by",
    ),
    "buser": ("X", "K", "df", "lambda", "lambda_value", "center", "by"),
    "brad": ("knots", "df", "lambda", "lambda_value", "theta", "center", "by"),
}


def _raise_unknown_parameter(learner: str, key: str) -> NoReturn:
    """Reject an unrecognized ``learner(...)`` keyword with a suggestion.

    Raises :class:`~mboostrs.MboostrsValueError` (a ``ValueError`` subclass),
    so existing ``except ValueError`` handlers keep catching it, and carries
    ``code="unknown_parameter"``, ``subject=<learner>`` and ``details=<key>``
    for handlers that want the offending keyword without parsing the message.
    """
    valid = _LEARNER_PARAMETERS[learner]
    best = difflib.get_close_matches(key, valid, n=1, cutoff=0.6)
    message = f"unknown {learner} parameter {key!r}"
    if best:
        message += f"; did you mean {best[0]!r}?"
    raise MboostrsValueError(
        message,
        code="unknown_parameter",
        subject=learner,
        details=key,
    )


def _kwarg_number(value: str, key: str, learner: str, cast=float):
    """A numeric argument, named against the term when it will not convert.

    The bare ``could not convert string to float: 'notanumber'`` this replaces
    named neither the learner nor the argument.
    """
    text = value.strip()
    try:
        return cast(text)
    except (TypeError, ValueError) as exc:
        wanted = "an integer" if cast is int else "a number"
        raise ValueError(
            f"{learner}(..., {key}={text}) is not {wanted}."
        ) from exc


def _kwarg_bins(value: str, learner: str):
    """`bins=`, which is a count or the string ``"auto"``.

    A formula carries `bins='auto'` or `bins="auto"` with the quotes, so they
    are stripped before the literal is compared.
    """
    text = value.strip().strip("'\"")
    if text == "auto":
        return "auto"
    return _kwarg_number(value, "bins", learner, int)


def _parse_bbs_kwargs(kwargs_str: str) -> dict:
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bbs"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "bins":
            kwargs[key] = _kwarg_bins(val, "bbs")
        elif key in ("knots", "degree", "differences"):
            kwargs[key] = _kwarg_number(val, key, "bbs", int)
        elif key == "df":
            kwargs[key] = _kwarg_number(val, key, "bbs")
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "bbs")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "bbs")
        elif key == "cyclic":
            kwargs[key] = _kwarg_bool(val, key, "bbs")
        elif key == "boundary.knots":
            parts = val.strip("c()")
            lo, hi = [float(x.strip()) for x in parts.split(",")]
            kwargs["boundary_knots"] = (lo, hi)
        elif key == "constraint":
            kwargs[key] = val
        elif key in ("deriv",):
            kwargs[key] = _kwarg_number(val, key, "bbs", int)
        elif key == "by":
            kwargs[key] = val
        elif key == "knot_placement":
            kwargs[key] = val.strip().strip("'\"")
        else:
            _raise_unknown_parameter('bbs', key)
    return kwargs


def _reject_extra_positional(term_name: str, tail: str) -> None:
    """Refuse a learner call whose extra arguments are bare column names.

    R's `bols(x1, x2)` builds ONE learner over both columns, which selects the
    two together rather than competing them. The kwargs parser here matches
    `key=value` pairs only, so a bare second name matched nothing and was
    dropped: `bols(x1, x2)` fitted x1 alone and lost x2 with no warning.
    """
    if tail is None:
        return
    pieces = [piece for _, piece in split_top_level(tail, ",")]

    for piece in pieces:
        piece = piece.strip()
        if piece and "=" not in piece and re.match(rf"^{N}$", piece):
            raise ValueError(
                f"{term_name}({piece!r}) names a second column. A single "
                f"learner over several columns — R's {term_name}(a, b), which "
                "selects them jointly — is not supported. Write one term per "
                f"column ({term_name}(a) + {term_name}(b)) to select them "
                "separately, or use Buser with an explicit design matrix to "
                "keep them in one learner."
            )


def _parse_bols_kwargs(kwargs_str: str) -> dict:
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bols"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "df":
            kwargs[key] = _kwarg_number(val, key, "bols")
        elif key == "bins":
            kwargs[key] = _kwarg_bins(val, "bols")
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "bols")
        elif key == "intercept":
            kwargs[key] = _kwarg_bool(val, key, "bols")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bols', key)
    return kwargs


def _parse_bmono_kwargs(kwargs_str: str) -> dict:
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bmono"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key in ("knots", "degree", "differences", "niter"):
            kwargs[key] = _kwarg_number(val, key, "bmono", int)
        elif key in ("df", "lambda2"):
            kwargs[key] = _kwarg_number(val, key, "bmono")
        elif key in ("constraint", "type"):
            kwargs[key] = val
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bmono', key)
    return kwargs


def _parse_btree_kwargs(kwargs_str: str) -> dict:
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "btree"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key in ("minbucket", "minsplit", "maxdepth", "colsample_seed"):
            kwargs[key] = _kwarg_number(val, key, "btree", int)
        elif key in ("lambda_leaf", "colsample"):
            kwargs[key] = _kwarg_number(val, key, "btree")
        elif key == "by":
            kwargs[key] = val
        elif key in ("selection", "split_method"):
            kwargs[key] = val.strip().strip("'\"")
        else:
            _raise_unknown_parameter('btree', key)
    return kwargs


def _parse_bmrf_kwargs(kwargs_str: str, *, env: dict | None = None) -> dict:
    """Parse bmrf kwargs from a formula string.

    When ``K=<symbol>`` is present, the symbol is resolved from *env*.
    If *env* is ``None`` and ``K=`` is used, a diagnostic ``ValueError``
    is raised.
    """
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bmrf"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "K" and _is_bare_identifier(m):
            kwargs["K"] = _resolve_env_symbol(val, "K", "bmrf", env)
        elif key == "df":
            kwargs[key] = _kwarg_number(val, key, "bmrf")
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "bmrf")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "bmrf")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bmrf', key)
    return kwargs


def _parse_brandom_kwargs(kwargs_str: str) -> dict:
    """Parse brandom kwargs from a formula string."""
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "brandom"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "df":
            kwargs[key] = _kwarg_number(val, key, "brandom")
        elif key == "bins":
            kwargs[key] = _kwarg_bins(val, "brandom")
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "brandom")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "brandom")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('brandom', key)
    return kwargs


def _parse_bspatial_kwargs(kwargs_str: str) -> dict:
    """Parse bspatial kwargs from a formula string."""
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bspatial"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "bins":
            kwargs[key] = _kwarg_bins(val, "bspatial")
        elif key in ("knots", "degree", "differences"):
            kwargs[key] = _kwarg_number(val, key, "bspatial", int)
        elif key == "df":
            kwargs[key] = _kwarg_number(val, key, "bspatial")
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "bspatial")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "bspatial")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bspatial', key)
    return kwargs


def _parse_bns_kwargs(kwargs_str: str) -> dict:
    """Parse bns kwargs from a formula string."""
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bns"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key in ("knots", "degree"):
            kwargs[key] = _kwarg_number(val, key, "bns", int)
        elif key == "df":
            kwargs[key] = _kwarg_number(val, key, "bns")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "bns")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bns', key)
    return kwargs


def _parse_bkernel_kwargs(kwargs_str: str) -> dict:
    """Parse bkernel kwargs from a formula string."""
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bkernel"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "bandwidth":
            kwargs[key] = _kwarg_number(val, key, "bkernel")
        elif key == "df":
            kwargs[key] = _kwarg_number(val, key, "bkernel")
        elif key in ("kernel",):
            kwargs[key] = val
        elif key == "landmarks":
            kwargs[key] = _kwarg_number(val, key, "bkernel", cast=int)
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "bkernel")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bkernel', key)
    return kwargs


def _parse_bconstrained_kwargs(kwargs_str: str, *, env: dict | None = None) -> dict:
    """Parse bconstrained kwargs from a formula string.

    When ``C=<symbol>`` is present, the symbol is resolved from *env*.
    If *env* is ``None`` and ``C=`` is used, a diagnostic ``ValueError``
    is raised.
    """
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "bconstrained"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "C" and _is_bare_identifier(m):
            kwargs["C"] = _resolve_env_symbol(val, "C", "bconstrained", env)
        elif key in ("knots", "degree", "differences", "niter"):
            kwargs[key] = _kwarg_number(val, key, "bconstrained", int)
        elif key in ("df", "lambda2"):
            kwargs[key] = _kwarg_number(val, key, "bconstrained")
        elif key == "type":
            kwargs[key] = val.strip("\"'")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('bconstrained', key)
    return kwargs


def _parse_buser_kwargs(kwargs_str: str, *, env: dict | None = None) -> dict:
    """Parse buser kwargs from a formula string.

    ``X=<symbol>`` and ``K=<symbol>`` are required and resolved from *env*.
    """
    kwargs = {}
    has_x = False
    has_k = False
    for m in _kwarg_matches(kwargs_str, "buser"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "X" and _is_bare_identifier(m):
            kwargs["X"] = _resolve_env_symbol(val, "X", "buser", env)
            has_x = True
        elif key == "K" and _is_bare_identifier(m):
            kwargs["K"] = _resolve_env_symbol(val, "K", "buser", env)
            has_k = True
        elif key == "df":
            kwargs[key] = _kwarg_number(val, key, "buser")
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "buser")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "buser")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter('buser', key)
    if not has_x:
        raise ValueError(
            "buser() requires X=<symbol>, e.g. "
            "buser(custom, X=X_custom, K=K_custom, df=4)"
        )
    if not has_k:
        raise ValueError(
            "buser() requires K=<symbol>, e.g. "
            "buser(custom, X=X_custom, K=K_custom, df=4)"
        )
    return kwargs


def parse_gamboost_formula(
    formula: str,
    *,
    env: dict | None = None,
    data=None,
) -> tuple[
    str | tuple[str, str],
    list[Bbs | Bkernel | Bmono | Bns | Bols | Brad | Btree | Bmrf | Bspatial | Brandom | Bconstrained | Buser],
]:
    """Parse an additive formula into its response and learner objects.

    A column name that is not a plain identifier (it has spaces or operators,
    starts with a digit, or is a reserved word such as ``bbs`` or ``log``) is
    written in backticks, as in R: ``"y ~ bbs(`house age`) + bols(`price/sqft`)"``.
    Inside backticks a backslash escapes a backtick or a backslash, so a column
    named ``a`b`` is written ```a\\`b```. Every formula that mboostrs prints or
    generates uses the same spelling, so it parses back to the same model.

    Parameters
    ----------
    formula : str
        ``"y ~ bbs(x, df=4) + bols(z)"``. A bare name becomes a ``Bbs`` with
        default arguments, as in R's ``gamboost``.
    env : dict, optional
        Names resolved inside learner arguments, for the arguments a formula
        cannot express literally — ``bmrf(district, K=K_matrix)`` needs
        ``env={"K_matrix": K}``.

    Returns
    -------
    tuple
        ``(response, learners)``, where `learners` holds the constructed
        learner objects in formula order. They can be passed straight to
        :func:`~mboostrs.gamboost.gamboost` as ``learners=``.

    Examples
    --------
    >>> response, learners = parse_gamboost_formula("y ~ bbs(a, df=4) + bols(b)")
    >>> response, [type(l).__name__ for l in learners]
    ('y', ['Bbs', 'Bols'])
    """
    # `- 1` and `+ 0` remove the global intercept candidate; gamboost has none
    # to remove, so the learners are those of the formula without the term.
    # `data` decides whether a bare `0` or `1` is a column of that name.
    formula, _ = split_intercept(formula, {} if data is None else data)
    response, terms = _split_formula(formula)
    response = _parse_response(response)
    learners = []
    for term in terms:
        # `y ~ 1` names the intercept, as it does in R: the term is not a
        # column but the estimated constant itself, one coefficient on a
        # column of ones. `intercept=False` because the ones column *is* the
        # constant; a second one would double it.
        if term == "1":
            learners.append(Bols("1", intercept=False))
            continue
        m_brad = _BRAD_CALL_RE.match(term)
        if m_brad:
            names = [unquote_name(piece) for _, piece in split_top_level(m_brad.group(1), ",")]
            kwargs = {}
            for match in _kwarg_matches(m_brad.group(2) or "", "brad"):
                key, value = match.group(1), _kwarg_value(match)
                if key == "knots":
                    kwargs[key] = _kwarg_number(value, key, "brad", int)
                elif key in ("df", "theta"):
                    kwargs[key] = _kwarg_number(value, key, "brad")
                elif key in ("lambda", "lambda_value"):
                    kwargs["lambda_value"] = _kwarg_number(value, key, "brad")
                elif key == "center":
                    kwargs[key] = _kwarg_bool(value, key, "brad")
                elif key == "by":
                    kwargs[key] = value
                else:
                    _raise_unknown_parameter("brad", key)
            learners.append(Brad(names, **kwargs))
            continue
        m_bspatial = _BSPATIAL_CALL_RE.match(term)
        if m_bspatial:
            name1 = unquote_name(m_bspatial.group(1))
            name2 = unquote_name(m_bspatial.group(2))
            kwargs = (
                _parse_bspatial_kwargs(m_bspatial.group(3))
                if m_bspatial.group(3)
                else {}
            )
            learners.append(Bspatial(name1, name2, **kwargs))
            continue
        m_bmrf = _BMRF_CALL_RE.match(term)
        if m_bmrf:
            name = unquote_name(m_bmrf.group(1))
            kwargs = (
                _parse_bmrf_kwargs(m_bmrf.group(2), env=env)
                if m_bmrf.group(2)
                else {}
            )
            if "K" not in kwargs:
                kwargs["K"] = None
            learners.append(Bmrf(name, **kwargs))
            continue
        m_brandom = _BRANDOM_CALL_RE.match(term)
        if m_brandom:
            name = unquote_name(m_brandom.group(1))
            kwargs = _parse_brandom_kwargs(m_brandom.group(2)) if m_brandom.group(2) else {}
            learners.append(Brandom(name, **kwargs))
            continue
        m_bkernel = _BKERNEL_CALL_RE.match(term)
        if m_bkernel:
            name = unquote_name(m_bkernel.group(1))
            kwargs = _parse_bkernel_kwargs(m_bkernel.group(2)) if m_bkernel.group(2) else {}
            learners.append(Bkernel(name, **kwargs))
            continue
        m_bns = _BNS_CALL_RE.match(term)
        if m_bns:
            name = unquote_name(m_bns.group(1))
            kwargs = _parse_bns_kwargs(m_bns.group(2)) if m_bns.group(2) else {}
            learners.append(Bns(name, **kwargs))
            continue
        m_bconstrained = _BCONSTRAINED_CALL_RE.match(term)
        if m_bconstrained:
            name = unquote_name(m_bconstrained.group(1))
            kwargs = (
                _parse_bconstrained_kwargs(m_bconstrained.group(2), env=env)
                if m_bconstrained.group(2)
                else {}
            )
            if "C" not in kwargs:
                kwargs["C"] = None
            learners.append(Bconstrained(name, **kwargs))
            continue
        m_buser = _BUSER_CALL_RE.match(term)
        if m_buser:
            name = unquote_name(m_buser.group(1))
            kwargs = _parse_buser_kwargs(m_buser.group(2), env=env) if m_buser.group(2) else {}
            learners.append(Buser(name, **kwargs))
            continue
        m_bmono = _BMONO_CALL_RE.match(term)
        if m_bmono:
            name = unquote_name(m_bmono.group(1))
            kwargs = _parse_bmono_kwargs(m_bmono.group(2)) if m_bmono.group(2) else {}
            learners.append(Bmono(name, **kwargs))
            continue
        m_btree = _BTREE_CALL_RE.match(term)
        if m_btree:
            names_str = m_btree.group(1)
            clean_names = []
            for _, part in split_top_level(names_str, "+,"):
                part = part.strip()
                if "=" in mask_names(part):
                    break
                if part:
                    clean_names.append(unquote_name(part))
            kwargs = _parse_btree_kwargs(m_btree.group(2)) if m_btree.group(2) else {}
            learners.append(Btree(clean_names, **kwargs))
            continue
        m_hinge = _HINGE_CALL_RE.match(term)
        if m_hinge:
            name = unquote_name(m_hinge.group(1))
            _reject_extra_positional("hinge", m_hinge.group(2))
            kwargs = _parse_hinge_kwargs(m_hinge.group(2)) if m_hinge.group(2) else {}
            learners.append(Hinge(name, **kwargs))
            continue
        m_bols = _BOLS_CALL_RE.match(term)
        if m_bols:
            name = unquote_name(m_bols.group(1))
            _reject_extra_positional("bols", m_bols.group(2))
            kwargs = _parse_bols_kwargs(m_bols.group(2)) if m_bols.group(2) else {}
            learners.append(Bols(name, **kwargs))
            continue
        m_bbs = _BBS_CALL_RE.match(term)
        if m_bbs:
            name = unquote_name(m_bbs.group(1))
            kwargs = _parse_bbs_kwargs(m_bbs.group(2)) if m_bbs.group(2) else {}
            learners.append(Bbs(name, **kwargs))
            continue
        if re.match(rf"^{N}$", term):
            learners.append(Bbs(unquote_name(term)))
            continue
        raise ValueError(unparsed_term_message(term))
    return response, learners


__all__ = ["parse_formula", "parse_gamboost_formula", "parse_response"]


def _parse_hinge_kwargs(kwargs_str: str) -> dict:
    kwargs = {}
    for m in _kwarg_matches(kwargs_str, "hinge"):
        key = m.group(1).strip()
        val = _kwarg_value(m)
        if key == "df":
            kwargs[key] = _kwarg_number(val, key, "hinge")
        elif key == "n_candidates":
            kwargs[key] = _kwarg_number(val, key, "hinge", int)
        elif key in ("lambda", "lambda_value"):
            kwargs["lambda_value"] = _kwarg_number(val, key, "hinge")
        elif key == "center":
            kwargs[key] = _kwarg_bool(val, key, "hinge")
        elif key == "by":
            kwargs[key] = val
        else:
            _raise_unknown_parameter("hinge", key)
    return kwargs
