"""Inline formula transforms and the recipe that replays them.

A formula may name a transformed predictor directly::

    y ~ log(income) + bbs(scale(age), df=4)

Fitting materializes each transform into a derived column and rewrites the
formula to reference it. The parameters of any *fitted* transform are recorded
in a recipe stored on the model, and `predict(newdata=...)` replays that recipe
before extracting columns.

The recipe is the point of this module. `scale(x)` centers and scales by the
training mean and standard deviation; recomputing them on `newdata` would
silently produce a different mapping and therefore wrong predictions — the
classic error that pre-transforming by hand invites. Recording the fitted
parameters removes the possibility.

Transforms are whitelisted rather than `eval`-ed, so a formula stays safe to
accept from a config file and every transform stays replayable.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from mboostrs._errors import MboostrsValueError
from mboostrs._names import IDENT_SRC as IDENT
from mboostrs._names import NAME_SRC as N
from mboostrs._names import partition_tilde, quote_name, unquote_name

__all__ = [
    "TransformRecipe",
    "apply_recipe",
    "display_names",
    "encode_factors",
    "expand_factor_dummies",
    "impute_missing",
    "lower_categorical_bols",
    "extract_transforms",
    "response_forward",
    "response_inverse",
    "response_jacobian",
]

# fname(column) — a whitelisted transform applied to a single bare column.
_TRANSFORM_RE = re.compile(rf"\b(log|log1p|sqrt|exp|scale|center|ecdf)\s*\(\s*({N})\s*\)")

#: Quantile grid size for `ecdf(x)`: the recorded map interpolates linearly
#: between at most this many training quantiles.
_ECDF_POINTS = 257

# fname(column, bound) — the two-argument clamps, matching R's pmin/pmax.
_CLAMP_RE = re.compile(
    rf"\b(pmin|pmax)\s*\(\s*({N})\s*,\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*\)"
)

# I(<arithmetic over data columns>) — e.g. I(x**2), I(x * 2 + 1), I(x * z).
_IDENTITY_RE = re.compile(r"\bI\s*\(\s*([^()]+?)\s*\)")

_STATELESS = {
    "log": np.log,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "exp": np.exp,
}

#: Response transforms with an inverse and a Jacobian in the Rust core.
_INVERTIBLE = frozenset({"log", "log1p", "sqrt", "exp", "scale", "center"})

_CLAMPS = {"pmin": np.minimum, "pmax": np.maximum}

#: The transforms a *response* may carry. Every one is invertible, so
#: original-scale quantiles, medians and densities can be recovered from the
#: fit; the clamps and ``I()`` are not invertible and are refused for a
#: response. `scale`/`center` are fitted on the training response, so the
#: step carries ``(mean, sd)`` params (sd=1 for center) like it does for
#: predictors.
_RESPONSE_KINDS = ("log", "log1p", "sqrt", "exp", "scale", "center")

_RESPONSE_RE = re.compile(
    rf"^\s*(log|log1p|sqrt|exp|scale|center)\s*\(\s*({N})\s*\)\s*$"
)


@dataclass(frozen=True)
class _Step:
    """One derived column: how to build it, and any fitted parameters."""

    name: str            # derived column name
    kind: str            # "log" | "scale" | "center" | "I" | "pmin" | ...
    sources: tuple       # source column names, in first-appearance order
    expression: str      # original formula text; also the display name
    params: tuple = ()   # fitted parameters, e.g. (mean, sd) for scale

    @property
    def source(self) -> str:
        """The first source column.

        Every transform but `I()` reads exactly one column, and the artifact
        format wrote a single `source` before `I()` could take more than one.
        """
        return self.sources[0]


@dataclass(frozen=True)
class TransformRecipe:
    """An ordered set of derived columns with their fitted parameters."""

    steps: tuple[_Step, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.steps)

    @property
    def derived_names(self) -> list[str]:
        return [s.name for s in self.steps]

    @property
    def source_names(self) -> list[str]:
        names: list[str] = []
        for step in self.steps:
            for source in step.sources:
                if source not in names:
                    names.append(source)
        return names

    @property
    def display_names(self) -> dict[str, str]:
        """Derived column name -> the expression the user wrote.

        The derived name has to be formula-safe, so it cannot be `log(x)`.
        Coefficient tables and selection paths report the expression through
        this mapping, which is what R shows.
        """
        return {step.name: step.expression for step in self.steps}

    def describe(self) -> list[dict]:
        return [
            {
                "derived": s.name,
                "expression": s.expression,
                "source": s.source,
                "sources": list(s.sources),
                "kind": s.kind,
                "params": list(s.params),
            }
            for s in self.steps
        ]


def display_names(recipe, names):
    """Replace derived column names with the expressions the user wrote.

    `log(x)` becomes the column `log__x` so the rewritten formula stays
    parseable, but that name is an implementation detail: R reports the term as
    `log(x)`, and so should coefficient tables, selection paths and plots. A
    name that is not a derived column passes through unchanged.

    A `factor` step encodes its column in place, so its derived name *is* the
    user's name and it is skipped.
    """
    if not recipe:
        return list(names)
    mapping = {
        step.name: step.expression
        for step in recipe.steps
        if step.kind not in ("factor", "datetime")
    }
    return [mapping.get(name, name) for name in names]


def _derived_name(expression: str, taken: dict[str, str]) -> str:
    """A formula-safe identifier for a derived column.

    Formula parsing accepts `[\\w.]+`, so the encoded name must avoid brackets
    and operators. The name is built from the whole expression, not from the
    transform's kind and source column: those two are not unique. `I(x**2)` and
    `I(x**3)` both reduced to `I__x`, and because a repeated derived name is
    treated as the same column, `y ~ I(x**2) + I(x**3)` silently fitted a
    one-term model — no error, no warning, a cubic term simply absent.

    Sanitizing can still map two expressions together (`I(x**2)` and `I(x*2)`
    both give `I__x_2`), so `taken` maps each name already issued to the
    expression that produced it and a suffix is added when they differ. Nothing
    depends on the exact spelling: users see `expression`, and the derived name
    is internal.
    """
    opened = expression.replace("(", "__", 1)
    base = re.sub(r"_+$", "", re.sub(r"[^\w.]", "_", opened)) or "derived"
    if taken.get(base, expression) == expression:
        return base
    suffix = 2
    while taken.get(f"{base}_{suffix}", expression) != expression:
        suffix += 1
    return f"{base}_{suffix}"


def _column(data, name: str) -> np.ndarray:
    values = data[name]
    return np.asarray(
        values.to_numpy() if hasattr(values, "to_numpy") else values, dtype=np.float64
    ).ravel()


def _ordered_names(expression: str) -> list[str]:
    """Identifiers in `expression`, in first-appearance order, without repeats.

    A backticked name is returned as written, backticks included.
    """
    names: list[str] = []
    for name in re.findall(rf"{IDENT}", expression):
        if name not in names:
            names.append(name)
    return names


def _sub_outside_names(pattern, function, text: str) -> str:
    """``pattern.sub(function, text)`` that ignores text inside backticked names."""
    from mboostrs._names import mask_names

    masked = mask_names(text)
    out, last = [], 0
    for found in pattern.finditer(masked):
        start, end = found.span()
        out.append(text[last:start])
        out.append(function(pattern.fullmatch(text, start, end)))
        last = end
    out.append(text[last:])
    return "".join(out)


def _has_column(data, name: str) -> bool:
    if isinstance(data, dict):
        return name in data
    return name in getattr(data, "columns", ())


def extract_transforms(formula: str, data) -> tuple[str, TransformRecipe]:
    """Rewrite inline transforms into derived column names.

    Returns the rewritten formula and the recipe needed to build (and later
    replay) the derived columns. A formula with no transforms returns unchanged
    with an empty recipe.
    """
    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        from mboostrs.formula import no_response_message

        raise ValueError(no_response_message(formula))

    steps: list[_Step] = []
    seen: dict[str, _Step] = {}
    # Derived name -> the expression that produced it, so a name is reused for
    # a repeat of the same transform and disambiguated for a different one.
    taken: dict[str, str] = {}

    def _register(kind: str, sources, expression: str, params=()) -> str:
        name = _derived_name(expression, taken)
        taken[name] = expression
        if name not in seen:
            if _has_column(data, name):
                raise ValueError(
                    f"transform {expression!r} would derive a column named {name!r}, "
                    f"which already exists in the data; rename that column or the "
                    f"transform's source to avoid the collision"
                )
            step = _Step(
                name=name,
                kind=kind,
                sources=tuple(sources),
                expression=expression,
                params=params,
            )
            seen[name] = step
            steps.append(step)
        return name

    def _require_column(source: str, expression: str) -> None:
        if not _has_column(data, source):
            raise ValueError(
                f"transform {expression!r} references column {source!r}, "
                f"which is not in the data"
            )

    def _sub_transform(match: re.Match) -> str:
        kind, source = match.group(1), unquote_name(match.group(2))
        _require_column(source, match.group(0))
        return _register(kind, (source,), match.group(0))

    def _sub_clamp(match: re.Match) -> str:
        kind, source, bound = match.group(1), unquote_name(match.group(2)), match.group(3)
        _require_column(source, match.group(0))
        return _register(kind, (source,), match.group(0), params=(float(bound),))

    def _sub_identity(match: re.Match) -> str:
        written = match.group(1)
        # R spells exponentiation `^`, Python `**`. Accept both, but evaluate
        # the Python form and report back the form the user wrote, so the same
        # formula gives the same coefficient label in either frontend.
        expression = written.replace("^", "**")
        tokens = _ordered_names(expression)
        names = [unquote_name(t) for t in tokens]
        # A backticked name cannot be evaluated; the stored expression refers to
        # it by an alias, the position of the column among the sources.
        for position, token in enumerate(tokens):
            if token.startswith("`"):
                expression = expression.replace(token, f"__c{position}__")
        columns = [n for n in names if _has_column(data, n)]
        unknown = [n for n in names if n not in columns]
        if unknown:
            raise ValueError(
                f"I({written}) references unknown name(s) {unknown}; only "
                f"arithmetic on data columns is supported"
            )
        if not columns:
            raise ValueError(f"I({written}) must reference at least one data column")
        if not re.fullmatch(r"[\w.\s+\-*/%()0-9]+", expression):
            raise ValueError(f"I({written}) contains unsupported syntax")
        return _register("I", columns, f"I({written})", params=(expression,))

    # A transformed response: `log(rent) ~ ...` fits the model to
    # log(rent), and the step recorded here is what inverts quantiles and
    # carries the Jacobian back to the original scale. The whitelist is the
    # invertible single-column transforms; a clamp or an `I()` on the
    # response would make the original scale unreachable, so it is refused
    # with the reason rather than fitted silently on a scale nobody can
    # leave again.
    lhs_stripped = lhs.strip()
    if _RESPONSE_RE.match(lhs_stripped):
        kind, source = _RESPONSE_RE.match(lhs_stripped).groups()
        source = unquote_name(source)
        _require_column(source, lhs_stripped)
        lhs = _register(kind, (source,), lhs_stripped)
    elif _CLAMP_RE.fullmatch(lhs_stripped) or _IDENTITY_RE.fullmatch(lhs_stripped):
        raise ValueError(
            f"response transform {lhs_stripped!r} is not invertible, so the "
            f"original response scale could not be recovered from the fit; "
            f"use one of {', '.join(_RESPONSE_KINDS)}(column) instead"
        )

    # Identity expressions first: they may contain names that also match a
    # transform pattern.
    new_rhs = _sub_outside_names(_IDENTITY_RE, _sub_identity, rhs)
    new_rhs = _sub_outside_names(_CLAMP_RE, _sub_clamp, new_rhs)
    new_rhs = _sub_outside_names(_TRANSFORM_RE, _sub_transform, new_rhs)

    if not steps:
        return formula, TransformRecipe()
    return f"{lhs.strip()} ~ {new_rhs.strip()}", TransformRecipe(tuple(steps))


def fit_recipe(recipe: TransformRecipe, data) -> TransformRecipe:
    """Compute any fitted parameters from the training data.

    `scale` and `center` record the training mean and standard deviation;
    `ecdf` records training quantiles. The rest are pure functions and are
    returned unchanged.
    """
    if not recipe:
        return recipe
    fitted: list[_Step] = []
    for step in recipe.steps:
        if step.kind in ("scale", "center"):
            from mboostrs import _rust

            values = _column(data, step.source)
            mean, sd = _rust.transform_mean_sd(_flat_float(values))
            if step.kind != "scale":
                sd = 1.0
            if step.kind == "scale" and not sd > 0.0:
                raise ValueError(
                    f"scale({step.source}) has zero variance in the training "
                    f"data, so it cannot be standardized"
                )
            fitted.append(
                _Step(step.name, step.kind, step.sources, step.expression, (mean, sd))
            )
        elif step.kind == "ecdf":
            fitted.append(_Step(step.name, step.kind, step.sources, step.expression,
                                _ecdf_params(_column(data, step.source), step.source)))
        elif step.kind in ("log", "sqrt"):
            # The smallest training value: a new value below it, where the
            # transform may be undefined (log(0)), is held there.
            values = _column(data, step.source)
            finite = values[np.isfinite(values)]
            lower = float(finite.min()) if finite.size else float("nan")
            fitted.append(_Step(step.name, step.kind, step.sources, step.expression,
                                (lower,) if lower > 0 or step.kind == "sqrt" and lower >= 0
                                else ()))
        else:
            fitted.append(step)
    return TransformRecipe(tuple(fitted))


def _ecdf_params(values: np.ndarray, source: str) -> tuple:
    """Training quantiles and their probabilities for `ecdf(x)`.

    `ecdf(x)` maps a value to the share of training rows at or below it,
    interpolated linearly between at most `_ECDF_POINTS` quantiles and held
    at 0 and 1 outside the training range. A B-spline over `ecdf(x)` with
    equally spaced knots has its knots at quantiles of `x`, so every knot
    interval holds about the same number of rows, and a new value beyond the
    training range gets the prediction at the nearest training extreme.
    Tied values share one point at their average probability.
    """
    from mboostrs import _rust

    try:
        knots, probabilities = _rust.transform_ecdf_params(_flat_float(values))
    except ValueError:
        raise ValueError(
            f"ecdf({source}) needs at least two distinct finite training values") from None
    return (tuple(knots), tuple(probabilities))


def _median(values) -> float:
    """Median of a column (the Rust core's `np.median`)."""
    from mboostrs import _rust

    return float(_rust.transform_median(_flat_float(values)))


def _flat_float(values) -> np.ndarray:
    """A contiguous one-dimensional float64 copy for the Rust transform kernels."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())


def _apply_kernel(function: str, kind: str, params, values) -> np.ndarray:
    """Run a Rust transform kernel on an array of any shape."""
    from mboostrs import _rust

    array = np.asarray(values, dtype=np.float64)
    out = getattr(_rust, function)(kind, [float(p) for p in params], _flat_float(array))
    return np.asarray(out, dtype=np.float64).reshape(array.shape)


def _evaluate(step: _Step, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Build one derived column from its source columns, keyed by name."""
    if step.kind in _STATELESS:
        values = columns[step.source]
        if step.params and step.kind in ("log", "sqrt"):
            (lower,) = step.params
            below = np.asarray(values, dtype=np.float64) < lower
            if below.any():
                from mboostrs._extrapolation import ExtrapolationWarning

                warnings.warn(
                    f"{int(below.sum())} value(s) of {step.source!r} lie below its "
                    f"smallest training value {lower:g}; {step.expression} holds them "
                    f"there, so their effect is the one at the training minimum",
                    ExtrapolationWarning, stacklevel=4)
                values = np.maximum(values, lower)
        out = _apply_kernel("transform_forward", step.kind, (), values)
        if not np.all(np.isfinite(out)):
            bad = int(np.sum(~np.isfinite(out)))
            raise ValueError(
                f"{step.expression} produced {bad} non-finite value(s); check "
                f"the domain of {step.source!r} (log and sqrt need positive "
                f"input)"
            )
        return out
    if step.kind in _CLAMPS:
        (bound,) = step.params
        return _CLAMPS[step.kind](columns[step.source], bound)
    if step.kind in ("scale", "center"):
        mean, sd = step.params
        return _apply_kernel("transform_forward", step.kind, (mean, sd), columns[step.source])
    if step.kind == "ecdf":
        values, probabilities = step.params
        return _apply_kernel("transform_forward", "ecdf", (*values, *probabilities),
                             columns[step.source])
    if step.kind in ("factor", "datetime"):
        raise ValueError(
            f"{step.kind} steps are encoded from the source frame"
        )
    if step.kind == "I":
        (expression,) = step.params
        # Validated to arithmetic over data column names, with no builtins in
        # scope, so this stays a whitelist rather than an eval escape hatch.
        namespace = {name: columns[name] for name in step.sources}
        namespace.update({f"__c{i}__": columns[name] for i, name in enumerate(step.sources)})
        namespace["__builtins__"] = {}
        return np.asarray(eval(expression, namespace), dtype=np.float64)  # noqa: S307
    raise ValueError(f"unknown transform kind {step.kind!r}")


def response_inverse(step, values):
    """The original response scale from transformed values: ``g^-1``.

    `step` is the response's recipe step (``log(rent)`` and its fitted
    params for `scale`/`center`); `values` are on the transformed scale the
    fit works on. Quantiles commute with this inverse, so lower/median/upper
    bounds transformed through it are exact original-scale quantiles.
    """
    kind = step.kind
    if kind not in _INVERTIBLE:
        raise ValueError(
            f"{step.expression!r} is not an invertible response transform")
    return _apply_kernel("transform_inverse", kind, step.params, values)


def response_forward(step, values):
    """``g(y)``: the original response on the transformed scale the fit
    works on. A PIT or a residual needs the observed outcome on the scale
    the family was fitted to, without changing the caller's data."""
    return _evaluate(step, {step.source: np.asarray(values, dtype=np.float64)})


def response_jacobian(step, original):
    """``|dg/dy|`` at original response values, for densities of y.

    A density on the transformed scale `g(y)` becomes a density of `y` after
    multiplying by this factor: f_y(y) = f_g(g(y)) * |g'(y)|. For a log
    response this is the `+ log(rent)` term of the euro-scale negative log
    density that the rent study had to add by hand.
    """
    kind = step.kind
    if kind not in _INVERTIBLE:
        raise ValueError(
            f"{step.expression!r} is not an invertible response transform")
    return _apply_kernel("transform_jacobian", kind, step.params, original)


def refuse_unfilled_missing(model, recipe, data) -> None:
    """Refuse a missing value that a model fitted with ``na_action="impute"`` cannot fill.

    The fit records a fill (a median, or a level for a label) for every column
    of its formula that had a value to fill from, so a new row missing one of
    those columns is predicted. A column the formula reads and the recipe
    never fills is one the fit could not fill either; a missing value in it
    is refused here, naming the column, rather than with the generic message
    that recommends ``na_action="impute"``, which the model already has.
    Only a model whose recipe has imputation steps is checked.
    """
    if not recipe or not any(step.kind in ("impute", "impute_model", "impute_level",
                                           "missing_indicator")
                             for step in recipe.steps):
        return
    formula = getattr(model, "_formula", None)
    if formula is None:
        return
    formulas = list(formula.values()) if isinstance(formula, dict) else [formula]
    read = set()
    for text in formulas:
        if isinstance(text, str):
            read.update(unquote_name(t) for t in re.findall(rf"{IDENT}", partition_tilde(text)[2]))
    filled = {source for step in recipe.steps for source in step.sources}
    for name in sorted(read - filled):
        if not _has_column(data, name):
            continue
        values = data[name]
        values = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
        if values.dtype.kind != "f":
            continue
        missing = np.isnan(values)
        if missing.any():
            first = int(np.flatnonzero(missing)[0])
            raise ValueError(
                f"Column {name!r} contains {int(missing.sum())} NaN value(s) out of "
                f"{values.size} (first at row {first}). The model was fitted with "
                f"na_action='impute', but it records no fill for {name!r}, so the value "
                f"cannot be predicted. Drop those rows, or impute them yourself before "
                f"predicting.")


def apply_recipe(recipe: TransformRecipe, data, *, unseen_level_columns=frozenset(),
                 unseen_levels="error"):
    """Return `data` plus one derived column per recipe step.

    The input is not mutated; a dict copy (or frame `with_columns`) is returned.
    A factor column named in ``unseen_level_columns`` encodes a label the fit
    never saw as ``-1``, a code no fitted level carries, which the ``brandom``
    replay then predicts under the ``unseen_levels`` policy. Every other
    factor column refuses such a label.
    """
    if not recipe:
        return data

    # Model-based fills are computed from the columns as they were passed in,
    # before any step writes a filled column back: a column imputed from
    # another imputed column reads that column's missing values, as the fit did.
    model_fills = {}
    for step in recipe.steps:
        if step.kind == "impute_model":
            missing = [name for name in step.sources if not _has_column(data, name)]
            if missing:
                named = ", ".join(repr(name) for name in missing)
                raise ValueError(
                    f"prediction data is missing column {named}, needed by "
                    f"{step.expression}"
                )
            model_fills[step.name] = _impute_model_fill(step, data)

    derived = {}
    for step in recipe.steps:
        missing = [name for name in step.sources if not _has_column(data, name)]
        if missing:
            named = ", ".join(repr(name) for name in missing)
            raise ValueError(
                f"prediction data is missing column {named}, needed by "
                f"{step.expression}"
            )
        if step.kind == "impute_model":
            data = _with_columns(data, {step.name: model_fills[step.name]})
        elif step.kind == "factor":
            derived[step.name] = _encode_factor(
                step, data, allow_unseen=step.name in unseen_level_columns)
        elif step.kind == "dummy":
            derived[step.name] = _encode_dummy(
                step, data,
                unseen_levels=(unseen_levels if step.source in unseen_level_columns else "error"))
        elif step.kind == "missing_indicator":
            if _has_column(data, step.name):
                # Already built: some prediction routes replay the recipe on
                # data it has transformed once, where the source column is
                # imputed and would read as never missing.
                continue
            raw = np.asarray(_column(data, step.source), dtype=float)
            derived[step.name] = np.isnan(raw).astype(float)
        elif step.kind == "observed_indicator":
            if _has_column(data, step.name):
                continue
            raw = np.asarray(_column(data, step.source), dtype=float)
            derived[step.name] = (~np.isnan(raw)).astype(float)
        elif step.kind == "impute":
            raw = np.asarray(_column(data, step.source), dtype=float)
            derived[step.name] = np.where(np.isnan(raw), step.params[0], raw)
            # A transform step after this one (log(x) under an imputed x) reads
            # the imputed values, so they are written back to `data` now.
            data = _with_columns(data, {step.name: derived[step.name]})
        elif step.kind == "impute_level":
            raw = data[step.source]
            raw = np.asarray(raw.to_numpy() if hasattr(raw, "to_numpy") else raw,
                             dtype=object)
            miss = _missing_labels(raw)
            if hasattr(data, "with_columns"):
                # Keep Polars columns as native Utf8/String rather than pl.Object.
                replaced = [
                    step.params[0] if m else v
                    for m, v in zip(miss.tolist(), raw.tolist(), strict=True)
                ]
            else:
                replaced = np.where(miss, step.params[0], raw)
            # Replaced in `data` at once: the factor steps after this one read
            # the same column and must see the missing-value level.
            data = _with_columns(data, {step.name: replaced})
        elif step.kind == "datetime":
            derived[step.name] = _encode_datetime(step, data)
        else:
            derived[step.name] = _evaluate(
                step, {name: _column(data, name) for name in step.sources}
            )

    return _with_columns(data, derived)


def _with_columns(data, derived):
    """`data` with the columns in `derived` added or replaced."""
    if not derived:
        return data
    if isinstance(data, dict):
        return {**data, **derived}

    # polars
    if hasattr(data, "with_columns"):
        import polars as pl

        return data.with_columns(
            [pl.Series(name, values) for name, values in derived.items()]
        )
    # pandas
    out = data.copy()
    for name, values in derived.items():
        out[name] = values
    return out


# --------------------------------------------------------------------------
# categorical predictors
# --------------------------------------------------------------------------

# Learners whose design treats the column as a set of unordered levels. Any
# other learner would read the encoded codes as a continuous measurement and
# silently impose an ordering on the categories, so encoding is refused there.
_CATEGORICAL_LEARNERS = frozenset({"brandom", "bmrf"})

_TERM_LEARNER_RE = re.compile(rf"^\s*([A-Za-z_]\w*)\s*\(\s*({N})")


def _is_categorical(data, name: str) -> bool:
    """Whether a column's values are labels rather than measurements.

    A **boolean** is not one of them, though it was counted as one until the
    refusal it produced was measured: `bols(flag)` on a boolean column raised
    "encoding it as numbers would make bols treat the levels as an ordered
    measurement", while the identical column cast to float fitted and
    recovered the effect. `False < True` is the coding, not an arbitrary order
    over labels, so an indicator is exactly what a boolean is and `bols` is
    exactly the learner for it.
    """
    if isinstance(data, dict):
        values = np.asarray(data[name])
        return values.dtype.kind in "OUS"
    column = data[name]
    dtype = getattr(column, "dtype", None)
    if dtype is None:
        return False
    if hasattr(dtype, "kind"):                      # pandas / numpy
        return dtype.kind in "OUS"
    # A Polars `Enum` prints as ``Enum(categories=[...])``, so the name before
    # the parenthesis is what identifies it.
    return str(dtype).split("(")[0] in ("String", "Categorical", "Enum", "Utf8", "Object")


def _is_datetime(data, name: str) -> bool:
    """Whether a column holds dates or timestamps.

    A date is an ordered measurement, so every learner may use it: a smooth
    over time is a trend, a stump over time is a regime break. It was read as
    an unordered factor and refused for every learner but `brandom` -- except
    under Polars, where the dtype name is `Date` rather than anything in the
    categorical set, so the same frame fitted in one backend and was refused
    in the other two.
    """
    values = data[name] if not isinstance(data, dict) else data[name]
    dtype = getattr(values, "dtype", None)
    if dtype is not None and getattr(dtype, "kind", "") == "M":
        return True
    if dtype is not None and str(dtype).split("(")[0] in ("Date", "Datetime"):
        return True
    array = np.asarray(values.to_numpy() if hasattr(values, "to_numpy")
                       else values)
    if array.dtype.kind == "M":
        return True
    if array.dtype.kind != "O":
        return False
    import datetime

    sample = next((v for v in array.tolist() if v is not None), None)
    return isinstance(sample, (datetime.date, datetime.datetime))


#: The level `na_action="impute"` gives a categorical column's missing values.
_MISSING_LEVEL = "(missing)"


def _levels(data, name: str) -> tuple:
    """Levels of a factor column, with the missing-value level last so it is
    never the reference level of a treatment contrast.

    A Polars `Enum` column lists its levels in the declared order, restricted
    to the values the column holds, so the fit's level order is the Enum's
    sequence. Every other factor (a String, a Polars `Categorical`) is sorted.
    """
    values = data[name]
    dtype = getattr(values, "dtype", None)
    values = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    present = {v for v in values.tolist()}
    declared = []
    if dtype is not None and str(dtype).split("(")[0] == "Enum":
        declared = [level for level in dtype.categories.to_list() if level in present]
    rest = present.difference(declared)
    return tuple(declared) + tuple(sorted(
        rest, key=lambda v: (isinstance(v, str) and v.startswith(_MISSING_LEVEL), v)))


_BARE_BOLS_RE = re.compile(rf"^\s*bols\s*\(\s*({N})\s*\)\s*$")


def lower_categorical_bols(formula: str, data) -> str:
    """Rewrite an unpenalized `bols` over a categorical column as its level design.

    mboost's ``bols(f)`` for a factor ``f`` is least squares on the dummy
    design ``[1, d_2, ..., d_K]``. A zero-penalty ``brandom(f, lambda=0)`` fits
    least squares on ``[d_1, ..., d_K]``, which spans the same columns, so every
    boosting step, and therefore the fitted path, is the same; only the
    coefficients are parametrized by level rather than as contrasts. Both a
    ``bols(f)`` with no other argument and a bare ``f`` term (what ``y ~ .``
    produces) are rewritten; a two-level column is left alone because its
    integer code is already an indicator.
    """
    from mboostrs.formula import _signed_terms

    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        return formula
    rewritten = []
    bare = []
    changed = False
    for sign, term in _signed_terms(rhs):
        match = _BARE_BOLS_RE.match(term)
        column = (unquote_name(match.group(1)) if match
                  else (unquote_name(term) if re.fullmatch(rf"{N}", term.strip()) else None))
        if (column is not None and _has_column(data, column)
                and _is_categorical(data, column) and len(_levels(data, column)) > 2):
            if not match:
                bare.append((column, len(_levels(data, column))))
            term = f"brandom({quote_name(column)}, lambda=0)"
            changed = True
        rewritten.append((sign, term))
    if not changed:
        return formula
    if bare:
        _warn_bare_categorical(bare)
    body = " ".join(f"{sign} {term}" for sign, term in rewritten)
    body = body[2:] if body.startswith("+ ") else body
    return f"{lhs.rstrip()} ~ {body}"


_LABEL_TERM_RE = re.compile(rf"^\s*(bols|brandom)\s*\(\s*({N})\s*(,[^()]*)?\)\s*$")


def factor_term_labels(formula: str, data) -> dict[str, str]:
    """The reported name of each categorical column's learner, as the formula writes it.

    A categorical column is stored under its own name, so ``bols(f)`` with
    three levels and a bare ``f`` would both be reported as ``"f"``, while the
    same term with two levels was reported as ``"bols(f)"``. This returns
    ``{column: term}`` for every categorical column that one ``bols(...)``,
    ``brandom(...)`` or bare term in `formula` names, so the learner is
    labelled with the term the user wrote: ``"bols(f)"``, ``"brandom(f, lambda=0)"``
    or ``"f"`` for a bare name. A column that appears in more than one term,
    or under another learner (``bmrf(f)``, ``btree(f)``), is left out and keeps
    its column name, because one written term cannot label two learners.
    """
    from mboostrs.formula import _signed_terms

    _, sep, rhs = partition_tilde(formula)
    if not sep:
        return {}
    terms_by_column: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    for _, term in _signed_terms(rhs):
        text = " ".join(term.split())
        match = _LABEL_TERM_RE.match(text)
        if match:
            column, label = unquote_name(match.group(2)), text
        elif re.fullmatch(rf"{N}", text):
            column, label = unquote_name(text), text
        else:
            learner = _TERM_LEARNER_RE.match(text)
            if learner is None:
                continue
            column, label = unquote_name(learner.group(2)), None
        terms_by_column.setdefault(column, []).append(label)
    for column, written in terms_by_column.items():
        if len(written) != 1 or written[0] is None:
            continue
        if _has_column(data, column) and _is_categorical(data, column):
            labels[column] = written[0]
    return labels


def _warn_bare_categorical(bare) -> None:
    """One warning naming the bare categorical names a formula gave, the
    learner they were coded with, and the spelling that states the choice."""
    import warnings

    names = ", ".join(f"{column!r} ({levels} levels)" for column, levels in bare)
    first = bare[0][0]
    warnings.warn(
        f"the formula names categorical column(s) {names} bare, with no learner. "
        "A bare categorical name is coded as brandom(name, lambda=0): one "
        "unpenalized effect per level, the same fit as bols(name). Write "
        f"bols({quote_name(first)}) to state that coding in the formula, or brandom({quote_name(first)}) "
        "for a ridge-penalized random effect.",
        UserWarning, stacklevel=4)


def impute_missing(formula: str, data, *, learner: str = "bols",
                   model_based: bool = False, split_filled: bool = False,
                   predictors=None) -> tuple[str, TransformRecipe]:
    """Median imputation with a missing-value indicator, for ``na_action="impute"``.

    With ``model_based=True`` (``na_action="impute_model"``) a numeric
    predictor with missing values is filled from a penalized additive
    regression of that column on the other predictors instead of its median;
    the missingness indicator is unchanged. See `_fit_imputation_model`.

    With ``split_filled=True`` as well (``na_action="impute_model_split"``), a
    smooth or linear term on a filled column is fitted as two terms: one on
    the rows where the column was observed and one on the filled rows. On the
    claim-cost benchmark (10 draws) the split lowered the coverage of the
    housing-age effect interval from 0.90 to 0.60 and left the roof-class
    coverage at 0.80, so it is opt-in.

    A categorical predictor with a missing value gets one recipe step that
    replaces the missing entries with a level of their own, ``"(missing)"``,
    so the factor's encoding gives missing rows their own effect. Every
    numeric predictor in `formula` that has a missing value in `data`
    gets two recipe steps: ``<x>_missing`` (1 where ``x`` is missing, 0
    otherwise) and ``x`` itself with its missing values replaced by the
    training median. The indicator enters the formula as its own linear term,
    so the fit estimates how rows with a missing ``x`` differ from rows at the
    median. Both steps are replayed on new data, so a model fitted this way
    predicts rows with missing values. A numeric predictor with no missing
    value in `data` gets the median step alone: a new row missing it is
    predicted at the training median, with no indicator effect, since none
    could be estimated. This is the missing-indicator method:
    unbiased for prediction when missingness is unrelated to the unobserved
    value, and a working approximation otherwise.

    The indicator's name is ``<x>_missing``. If `data` already has a column
    with that name, the indicator takes a trailing underscore
    (``<x>_missing_``, and another one for each further clash), so the
    caller's column is never overwritten. The indicator is a derived column:
    prediction and `sensitivity` build it from `<x>`, and a caller supplies
    only `<x>`.
    """
    from mboostrs.formula import _formula_columns

    lhs, sep, _ = partition_tilde(formula)
    if not sep:
        return formula, TransformRecipe()
    response = {unquote_name(t) for t in re.findall(rf"{IDENT}", lhs)}
    # The first argument of a buser term names a placeholder column of an
    # expanded %X% tensor; its design comes from X=, so it is not a predictor.
    placeholders = {unquote_name(t) for t in re.findall(rf"\bbuser\(\s*({IDENT})", formula)}
    taken = set(_data_column_names(data))
    steps: list[_Step] = []
    extra_terms = []
    # Predictors of the imputation models: the formula's columns unless the
    # caller names them (a distributional fit passes every parameter's columns).
    predictor_columns = list(predictors) if predictors is not None else None
    # (column, missing indicator, observed indicator) of each model-based fill.
    switched: list[tuple[str, str, str]] = []
    for column in _formula_columns(formula, data):
        # `__observed_*` is a derived switch indicator, not a covariate.
        if column in response or column in placeholders or column.startswith("__"):
            continue
        values = data[column]
        values = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
        if _is_categorical(data, column):
            missing = _missing_labels(values)
            if not missing.any():
                # Complete in training: no missing-level effect can be estimated,
                # so a new row missing this label gets the training mode,
                # mirroring the median step on complete numeric columns.
                non_null = [v for v in values.tolist() if v is not None]
                if non_null:
                    counts: dict = {}
                    for v in non_null:
                        counts[v] = counts.get(v, 0) + 1
                    mode_label = max(counts, key=lambda k: (counts[k], str(k)))
                    steps.append(_Step(name=column, kind="impute_level", sources=(column,),
                                       expression=column, params=(mode_label,)))
                continue
            # A categorical column gets a level of its own for a missing
            # value; the level's effect is the indicator's.
            observed = {str(v) for v in values[~missing].tolist()}
            label = _MISSING_LEVEL
            while label in observed:
                label += "_"
            steps.append(_Step(name=column, kind="impute_level", sources=(column,),
                               expression=column, params=(label,)))
            continue
        if values.dtype.kind not in "fiu":
            continue
        if values.dtype.kind != "f" or not np.isnan(values).any():
            # Complete in training: no indicator effect can be estimated, but
            # a new row missing this value still gets the training median.
            if values.size:
                steps.append(_Step(name=column, kind="impute", sources=(column,),
                                   expression=column,
                                   params=(_median(values),)))
            continue
        present = values[~np.isnan(values)]
        if present.size == 0:
            raise ValueError(f"column {column!r} has no observed value to impute from")
        indicator = f"{column}_missing"
        while indicator in taken:
            indicator += "_"
        taken.add(indicator)
        steps.append(_Step(name=indicator, kind="missing_indicator", sources=(column,),
                           expression=indicator, params=()))
        if model_based and split_filled:
            # The observed-row indicator (1 - missing) is the multiplier of the
            # observed part of a switched smooth term; see _switch_model_terms.
            observed = f"__observed_{column}"
            while observed in taken:
                observed += "_"
            taken.add(observed)
            steps.append(_Step(name=observed, kind="observed_indicator", sources=(column,),
                               expression=observed, params=()))
            switched.append((column, indicator, observed))
        if model_based:
            if predictor_columns is None:
                predictor_columns = [c for c in _formula_columns(formula, data)
                                     if c not in response and c not in placeholders]
            steps.append(_fit_imputation_model(column, data, predictor_columns))
        else:
            steps.append(_Step(name=column, kind="impute", sources=(column,),
                               expression=column, params=(_median(present),)))
        # Without an intercept the term is zero for an observed value, so its
        # coefficient is the difference between a missing row and a row at the
        # median.
        extra_terms.append(f"{learner}({quote_name(indicator)}, intercept=False)" if learner
                           else quote_name(indicator))
    if not steps:
        return formula, TransformRecipe()
    if switched:
        formula = _switch_model_terms(formula, switched, learner)
    if not extra_terms:
        return formula, TransformRecipe(tuple(steps))
    return formula.rstrip() + " + " + " + ".join(extra_terms), TransformRecipe(tuple(steps))


def _split_top(rhs: str) -> list[str]:
    """The `+`-separated terms of a right-hand side, outside parentheses."""
    from mboostrs._names import split_top_level

    return [piece.strip() for _, piece in split_top_level(rhs, "+") if piece.strip()]


_SWITCHABLE_HEAD = re.compile(rf"^(bbs|bols)\(\s*({IDENT})\s*([,)])")


def _switch_model_terms(formula: str, switched, learner: str) -> str:
    """Split each smooth or linear term on a model-filled column into two switched terms.

    ``bbs(x, args)`` becomes ``bbs(x, args, by=__observed_x)`` (fitted on the
    rows where ``x`` was observed) plus ``bbs(x, args, by=x_missing)`` (fitted
    on the filled rows, where the design is multiplied by the indicator). A
    bare ``x`` term is ``bbs(x)`` under ``gamboost`` and ``bols(x)`` under
    ``glmboost``. A term that already has ``by=``, or that is part of an
    interaction, is left as written. The constant ``x_missing`` term that
    `impute_missing` adds is unchanged.
    """
    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        return formula
    by_column = {column: (indicator, observed) for column, indicator, observed in switched}
    out = []
    for term in _split_top(rhs):
        if unquote_name(term) in by_column and re.fullmatch(rf"{N}", term):
            kind, column, args = ("bbs" if learner == "bols" else "bols"), unquote_name(term), ""
            written = term
        else:
            match = _SWITCHABLE_HEAD.match(term)
            if not match or unquote_name(match.group(2)) not in by_column:
                out.append(term)
                continue
            kind, column = match.group(1), unquote_name(match.group(2))
            written = match.group(2)
            if not _closes_at_end(term, match.start(0) + len(match.group(1))):
                out.append(term)
                continue
            rest = term[term.index(written, len(kind)) + len(written):-1].lstrip()
            args = rest[1:].strip() if rest.startswith(",") else ""
            if "by=" in args:
                out.append(term)
                continue
        indicator, observed = by_column[column]
        tail = f", {args}" if args else ""
        out.append(f"{kind}({written}{tail}, by={quote_name(observed)})")
        out.append(f"{kind}({written}{tail}, by={quote_name(indicator)})")
    return f"{lhs.strip()} ~ {' + '.join(out)}"


def _closes_at_end(term: str, open_at: int) -> bool:
    """Whether the parenthesis opened at ``term[open_at]`` closes on the last character."""
    depth = 0
    for i in range(open_at, len(term)):
        if term[i] == "(":
            depth += 1
        elif term[i] == ")":
            depth -= 1
            if depth == 0:
                return i == len(term) - 1
    return False


def _missing_labels(values) -> np.ndarray:
    """Which entries of a label column are missing (None or NaN)."""
    return np.array([v is None or (isinstance(v, float) and np.isnan(v))
                     for v in np.asarray(values, dtype=object).tolist()], dtype=bool)


def _label_array(data, name: str) -> np.ndarray:
    values = data[name]
    return np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values,
                      dtype=object)


def _fit_imputation_model(target: str, data, predictors) -> _Step:
    """The ``impute_model`` recipe step for a numeric column with missing values.

    The conditional mean of `target` given the predictors is fitted in Rust on
    the rows where `target` is observed (`mboostrs._rust.impute_model_fit`).
    A numeric predictor enters as its values, with NaN for a missing value. A
    categorical predictor enters as the codes of its training levels, NaN for a
    missing or unseen label. The step records the model's parameters, the
    levels of each categorical predictor (``()`` for a numeric one) and the
    model's R-squared on the observed rows, which `imputation_r2` reports.
    The response is never a predictor.
    """
    from mboostrs import _rust

    y = _column(data, target)
    cols, counts, pred_levels, names = [], [], [], []
    for name in predictors:
        # `__ix*`/`__by*` are the derived designs of %X% and by= terms, not
        # covariates; the design they stand for is built from its own columns.
        if name == target or name.startswith("__") or not _has_column(data, name) \
                or _is_datetime(data, name):
            continue
        if _is_categorical(data, name):
            labels = _label_array(data, name)
            missing = _missing_labels(labels)
            levels = tuple(sorted({v for v in labels[~missing].tolist()}, key=str))
            if not levels:
                continue
            index = {v: i for i, v in enumerate(levels)}
            cols.append(_label_codes(labels, missing, index))
            counts.append(len(levels))
            pred_levels.append(levels)
        else:
            try:
                cols.append(_column(data, name))
            except (TypeError, ValueError):
                continue
            counts.append(0)
            pred_levels.append(())
        names.append(name)
    design = np.column_stack(cols) if cols else np.empty((y.size, 0))
    flat, r2 = _rust.impute_model_fit(np.ascontiguousarray(y), np.ascontiguousarray(design), counts)
    return _Step(name=target, kind="impute_model", sources=(target, *names),
                 expression=target,
                 params=(tuple(float(v) for v in flat), tuple(pred_levels), float(r2)))


def _label_codes(labels, missing, index) -> np.ndarray:
    return np.array([np.nan if m else float(index.get(v, np.nan))
                     for v, m in zip(labels.tolist(), missing.tolist(), strict=True)],
                    dtype=float)


def _impute_model_fill(step: _Step, data) -> np.ndarray:
    """The target column with its missing entries replaced by the model's prediction."""
    from mboostrs import _rust

    target = _column(data, step.source)
    missing = np.isnan(target)
    if not missing.any():
        return target
    names, pred_levels = step.sources[1:], step.params[1]
    cols = []
    for name, levels in zip(names, pred_levels, strict=True):
        if levels:
            labels = _label_array(data, name)
            index = {v: i for i, v in enumerate(levels)}
            cols.append(_label_codes(labels, _missing_labels(labels), index))
        else:
            cols.append(_column(data, name))
    design = np.column_stack(cols) if cols else np.empty((target.size, 0))
    out = target.copy()
    predicted = _rust.impute_model_predict(
        [float(v) for v in step.params[0]], np.ascontiguousarray(design[missing]))
    out[missing] = np.asarray(predicted, dtype=float)
    return out


def imputation_r2(recipe) -> dict:
    """R-squared of each model-based imputation in `recipe`, by column name.

    A column imputed by its median has no model and does not appear. An
    R-squared near 0 means the other covariates explain little of the column,
    so the model-based fill is close to a constant.
    """
    return {step.name: float(step.params[2])
            for step in (recipe.steps if recipe else ()) if step.kind == "impute_model"}


def imputation_diagnostics(model) -> dict:
    """Report the R-squared of each model-based predictor fill, by column.

    Works on any fit that takes ``na_action="impute_model"``. An empty dict
    means the model fills no column with a model. See `imputation_r2`.
    """
    return imputation_r2(getattr(model, "_transform_recipe", None))


def expand_factor_dummies(formula: str, data) -> tuple[str, TransformRecipe]:
    """Expand each bare categorical term into R's treatment-contrast columns.

    ``glmboost(y ~ x + f)`` in R builds its design through ``model.matrix``:
    a factor ``f`` with levels ``a < b < c`` contributes the 0/1 columns
    ``fb`` and ``fc`` (the first level is the reference), each its own
    component. This returns the formula with ``f`` replaced by those columns
    and one ``dummy`` step per column, which `apply_recipe` rebuilds on new
    data and which refuses a level the fit never saw. Columns with two levels
    are expanded too, so the coefficient carries R's name (``fb``).
    """
    from mboostrs.formula import _signed_terms

    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        return formula, TransformRecipe()
    taken = set(_data_column_names(data))
    steps: list[_Step] = []
    rewritten = []
    for sign, term in _signed_terms(rhs):
        column = unquote_name(term) if re.fullmatch(rf"{N}", term.strip()) else term.strip()
        if (re.fullmatch(rf"{N}", term.strip()) and _has_column(data, column)
                and _is_categorical(data, column)):
            levels = _levels(data, column)
            names = []
            for level in levels[1:]:
                # `region(missing)` becomes `region_missing`, not
                # `region_missing_`: a trailing separator carries nothing.
                derived = re.sub(r"\W", "_", f"{column}{level}").rstrip("_") or column
                while derived in taken:
                    derived += "_"
                taken.add(derived)
                names.append(derived)
                steps.append(_Step(name=derived, kind="dummy", sources=(column,),
                                   expression=derived, params=(level, levels)))
            for derived in names:
                rewritten.append((sign, derived))
            continue
        rewritten.append((sign, term))
    if not steps:
        return formula, TransformRecipe()
    body = " ".join(f"{sign} {term}" for sign, term in rewritten)
    body = body[2:] if body.startswith("+ ") else body
    return f"{lhs.rstrip()} ~ {body}", TransformRecipe(tuple(steps))


def _data_column_names(data):
    if isinstance(data, dict):
        return list(data)
    return list(getattr(data, "columns", []))


def _encode_dummy(step: _Step, data, *, unseen_levels="error"):
    level, levels = step.params[0], tuple(step.params[1])
    column = data[step.source]
    values = column.to_numpy() if hasattr(column, "to_numpy") else np.asarray(column)
    listed = values.tolist()
    mismatch = _unseen_dtype_mismatch(step.source, levels, listed, column)
    if mismatch is not None:
        raise mismatch
    unseen = sorted({v for v in listed if v not in levels},
                    key=lambda v: (v is None, str(v)))
    if unseen and unseen_levels not in ("average", "zero_effect"):
        raise ValueError(
            f"column {step.source!r} contains level(s) {unseen} not present "
            f"when the model was fitted (levels: {list(levels)}). A level the "
            f"model never saw has no estimated effect."
        )
    if unseen and unseen_levels == "average" and levels:
        known = set(levels)
        avg = 1.0 / len(levels)
        return np.array([avg if v not in known else (1.0 if v == level else 0.0)
                         for v in listed])
    return np.array([1.0 if v == level else 0.0 for v in listed])


def encode_factors(formula: str, data, *, factor_bols: bool = False) -> TransformRecipe:
    """Record an integer encoding for each categorical predictor in `formula`.

    Only learners in `_CATEGORICAL_LEARNERS` accept a categorical column, and
    ``bols`` when `factor_bols` is set: `gamboost` then lowers ``bols(f, ...)``
    to its treatment-contrast design (`interactions.expand_by_factors`). For
    any other learner the column is refused with an explanation, because the
    encoded codes would be read as a continuous measurement — a silent
    statistical error rather than a visible failure.
    """
    from mboostrs.formula import _signed_terms

    _, sep, rhs = partition_tilde(formula)
    if not sep:
        return TransformRecipe()

    steps: list[_Step] = []
    seen: set[str] = set()
    for _, term in _signed_terms(rhs):
        tree = re.match(r"^\s*btree\s*\((.*)\)\s*$", term)
        if tree:
            # A tree splits a factor column into two sets of levels, so every
            # categorical column of the term is coded 0..L-1, not refused.
            from mboostrs._names import mask_names, split_top_level

            for part in (piece for _, piece in split_top_level(tree.group(1), ",")):
                if "=" in mask_names(part):
                    continue
                column = unquote_name(part)
                if (column in seen or not _has_column(data, column)
                        or not _is_categorical(data, column)):
                    continue
                seen.add(column)
                steps.append(_Step(name=column, kind="factor", sources=(column,),
                                   expression=f"btree({quote_name(column)})",
                                   params=(_levels(data, column),)))
            continue
        match = _TERM_LEARNER_RE.match(term)
        if match:
            learner, column = match.group(1), unquote_name(match.group(2))
        elif re.fullmatch(rf"{N}", term.strip()):
            learner, column = "", unquote_name(term)
        else:
            continue

        if not _has_column(data, column):
            continue

        if _is_datetime(data, column):
            if column in seen:
                continue
            seen.add(column)
            steps.append(
                _Step(
                    name=column,         # encoded in place
                    kind="datetime",
                    sources=(column,),
                    expression=f"{learner}({column})" if learner else column,
                    params=(),
                )
            )
            continue

        if not _is_categorical(data, column):
            continue
        levels = _levels(data, column)
        # Two levels are an indicator, whatever their labels: any coding gives
        # the same fitted values up to the sign of one coefficient, so there
        # is no ordering to get wrong and the refusal below does not apply.
        accepted = learner in _CATEGORICAL_LEARNERS or (factor_bols and learner == "bols")
        if not accepted and len(levels) > 2:
            shown = f"{learner}({column})" if learner else column
            raise ValueError(
                f"{shown} is a categorical column with {len(levels)} levels. "
                f"Encoding it as numbers would make "
                f"{learner or 'the default learner'} treat the levels as an "
                f"ordered measurement. In gamboost, bols({quote_name(column)}) fits one "
                f"effect per level (with intercept=, df= or lambda= as in "
                f"mboost); brandom({quote_name(column)}) fits them with a ridge penalty; "
                f"otherwise expand the column into indicator columns yourself."
            )
        if column in seen:
            continue
        seen.add(column)
        steps.append(
            _Step(
                name=column,             # encoded in place
                kind="factor",
                sources=(column,),
                expression=f"{learner}({column})",
                params=(levels,),
            )
        )
    return TransformRecipe(tuple(steps))


def _encode_datetime(step: _Step, data):
    """A date column as the float axis the engine reads.

    Days for a date, seconds for a timestamp -- the same encoding
    `mboostrs._frame` reports, so a coefficient's units match what
    `classify_columns` said they would be.
    """
    from mboostrs._frame import read_column

    return read_column({step.source: data[step.source]}, step.source).values


def _unseen_dtype_mismatch(source: str, levels, listed, column):
    """The refusal for a factor column whose levels differ from the fitted
    ones only in dtype, or ``None`` when that is not what the data show.

    Every non-null incoming level unseen, yet every one equal to a fitted
    level after ``str()`` (or after casting the fitted strings to numbers),
    is a dtype change, not a new group: fitted as strings '1', '2', '3',
    arriving as Int64 1, 2, 3. ``unseen_levels="average"`` would then
    silently predict every row at the average group — the policy is for a
    level the fit never saw, and these are the fitted levels in another
    dtype — so the mismatch is refused under every policy, naming the cast
    that fixes it. Any level the fit saw, or a genuinely new level among the
    incoming ones, returns ``None`` and keeps the caller's own behaviour.
    """
    non_null = [v for v in listed
                if v is not None and not (isinstance(v, float) and np.isnan(v))]
    fitted = set(levels)
    if not non_null or any(v in fitted for v in non_null):
        return None
    if all(isinstance(level, str) for level in levels):
        fitted_kind, target = "strings", "pl.Utf8"
    elif all(isinstance(level, (int, np.integer)) and not isinstance(level, bool)
             for level in levels):
        fitted_kind, target = "integers", "pl.Int64"
    elif all(isinstance(level, (float, np.floating)) for level in levels):
        fitted_kind, target = "floats", "pl.Float64"
    else:
        return None
    fitted_strs = {str(level) for level in levels}
    fitted_numbers = set()
    for level in levels:
        try:
            fitted_numbers.add(float(level))
        except (TypeError, ValueError):
            pass
    for value in non_null:
        if isinstance(value, bool):
            return None
        if str(value) in fitted_strs:
            continue
        # A string that does not match is a new label; a number that does
        # not is a new value. Only a number matching a fitted string (or a
        # string matching a fitted number) is the dtype change.
        if isinstance(value, str) or _not_a_fitted_number(value, fitted_numbers):
            return None
    shown = _shown_levels(non_null)
    arriving = str(getattr(column, "dtype", "") or "")
    if arriving[:1].isupper():                      # polars: Int64, Utf8
        arriving_kind = arriving
    elif all(isinstance(v, str) for v in shown):
        arriving_kind = "strings"
    elif all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
             for v in shown):
        arriving_kind = "integers"
    else:
        arriving_kind = "floats"
    message = (
        f"column {source!r} was fitted as {fitted_kind} {list(levels)} but "
        f"arrives as {arriving_kind} {shown}; cast it, e.g. "
        f"pl.col({source!r}).cast({target}), before predicting. These are "
        f"the fitted levels in a different dtype, not new levels, so "
        f"unseen_levels='average' would silently predict every row at the "
        f"average group."
    )
    return MboostrsValueError(
        message,
        code="unseen_level",
        subject=source,
        details=f"levels differ only by dtype: fitted {fitted_kind} "
                f"{list(levels)}, arrived as {arriving_kind} {shown}",
    )


def _not_a_fitted_number(value, fitted_numbers) -> bool:
    """Whether an incoming number equals no fitted level numerically — the
    match ``str()`` alone misses: an Int64 column with a null reaches this
    check as the float 1.0, whose ``str()`` is ``'1.0'``, not the fitted
    ``'1'``."""
    try:
        return float(value) not in fitted_numbers
    except (TypeError, ValueError):
        return True


def _shown_levels(non_null) -> list:
    """The distinct incoming levels, sorted, for an error message."""
    try:
        return sorted(set(non_null))
    except TypeError:
        return sorted(set(non_null), key=str)


def _encode_factor(step: _Step, data, *, allow_unseen=False):
    levels = step.params[0]
    column = data[step.source]
    values = column.to_numpy() if hasattr(column, "to_numpy") else np.asarray(column)
    index = {level: i for i, level in enumerate(levels)}
    # The checks and the lookup only need each distinct value once; a stacked
    # counterfactual frame repeats a few labels over many rows. Values that
    # cannot be sorted together (an object column with None or mixed types)
    # take the row-by-row path.
    try:
        distinct, inverse = np.unique(values, return_inverse=True)
    except TypeError:
        distinct = None
    if distinct is not None and distinct.ndim == 1 and inverse.shape == values.shape:
        listed = distinct.tolist()
    else:
        distinct = None
        listed = values.tolist()
    mismatch = _unseen_dtype_mismatch(step.source, levels, listed, column)
    if mismatch is not None:
        raise mismatch
    # A missing value (None) is a level like any other here, and does not
    # compare with the labels; it is listed last.
    unseen = sorted({v for v in listed if v not in index},
                    key=lambda v: (v is None, str(v)))
    if unseen and not allow_unseen:
        remedy = (
            ' Pass unseen_levels="average" to predict them as an average group, '
            'or unseen_levels="zero_effect" for a zero coefficient.'
            if step.expression.startswith("brandom(") else ""
        )
        raise ValueError(
            f"column {step.source!r} contains level(s) {unseen} not present "
            f"when the model was fitted (levels: {list(levels)}). A level the "
            f"model never saw has no estimated effect.{remedy}"
        )
    codes = np.array([float(index.get(v, -1)) for v in listed], dtype=np.float64)
    return codes if distinct is None else codes[inverse]


def switched_term_labels(learners, recipe) -> dict:
    """Labels of the two parts of each switched smooth or linear term, by position.

    `impute_model` writes a smooth term on a filled column as ``bbs(x, by=...)``
    twice: once on the observed rows and once on the filled rows. Both learners
    carry the column name `x`, so the report labels them ``bbs(x) [observed]``
    and ``bbs(x) [filled]``. The result maps the position of each such learner
    in `learners` to its label; other positions are absent.
    """
    steps = recipe.steps if recipe else ()
    observed = {s.name: s.source for s in steps if s.kind == "observed_indicator"}
    filled = {s.name: s.source for s in steps
              if s.kind == "missing_indicator" and s.source in observed.values()}
    labels = {}
    for position, learner in enumerate(learners):
        by = getattr(learner, "by", None)
        if not isinstance(by, str):
            continue
        if by in observed:
            part, column = "observed", observed[by]
        elif by in filled:
            part, column = "filled", filled[by]
        else:
            continue
        labels[position] = f"{type(learner).__name__.lower()}({column}) [{part}]"
    return labels
