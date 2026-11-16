"""``validate(model, newdata)``: the problems ``predict`` would meet, listed first.

``predict`` stops on some of them (an unseen factor level under the default
``unseen_levels="error"``, a missing column, a per-observation offset that was
not supplied) and continues silently on others (a covariate outside its
training range under ``extrapolation="allow"``, a missing value that a fit
without imputation turns into a NaN prediction). ``validate`` runs the same
replay and range checks that predict runs, without predicting, and returns one
record per problem. Nothing in it parses a message: each record names its
``kind``, its ``severity`` and the column, term and counts it refers to.

Issue kinds
-----------

``missing_column`` (error)
    A column a term reads is not in ``newdata``.
``unseen_level`` (error)
    A factor column holds a level the fit never saw. ``column`` names it,
    ``levels`` lists the unseen labels and ``count`` the rows that carry one.
``na`` (error, or warning when the fit imputes the column)
    A column a term reads has missing values. ``count`` is the number of
    missing rows.
``extra_column`` (warning)
    Columns no term reads, and that are not the response. predict ignores
    them. All of them are one issue: the message lists the names and ``count``
    gives how many there are. ``column`` names the column when there is only
    one.
``read_through_transform`` (info)
    Columns that no term reads directly, but that a derived column is built from:
    ``x`` is read through ``log__x`` (a transform recorded in the fit's recipe)
    or through ``x_missing`` (the missing-value indicator). A column built by
    hand, such as ``households`` read through ``households__log``, is matched by
    name. These columns are not ignored. ``count`` gives how many; the message
    names the derived column each one is read through, on one line.
``row_count`` (error)
    ``newdata`` has no rows, or its columns have different lengths.
``offset`` (error)
    The model was fitted with a per-observation offset and ``offset=`` was not
    given (or has the wrong length), or an offset was given to a
    ``gamboostLSS`` model, whose ``predict`` takes none.
``extrapolation`` (warning)
    A spline, ``bspatial`` or ``%X%`` margin, or a linear term, has rows
    outside its training range. ``worst_distance`` is the farthest such row in
    training ranges. ``parameter`` names the distribution parameter for
    ``gamboostLSS``.
``not_range_checked`` (warning)
    A ``btree``, ``bkernel``, ``buser`` or ``brad`` term. Its predictions have
    no training-range check at all, so no ``extrapolation=`` policy applies.
``check_skipped`` (warning)
    A check that could not run, with the reason in ``message``.

Each record also carries ``n_rows``, the number of rows in ``newdata``, and
``suggested_policy``, the argument or action that addresses the issue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mboostrs._frame import column_names, read_column
from mboostrs._names import quote_name
from mboostrs._output import json_safe, output_frame

__all__ = ["validate", "RECORD_COLUMNS", "ISSUE_KINDS"]

#: Columns of the records, in the order a frame carries them.
RECORD_COLUMNS = (
    "kind", "severity", "parameter", "term", "column", "count", "n_rows",
    "worst_distance", "levels", "message", "suggested_policy",
)

#: The issue kinds ``validate`` can return.
ISSUE_KINDS = (
    "missing_column", "unseen_level", "na", "extra_column", "read_through_transform",
    "row_count",
    "offset", "extrapolation", "not_range_checked", "check_skipped",
)


@dataclass(frozen=True)
class _Term:
    """One term the model predicts with: its distribution parameter (LSS only),
    its label as the user wrote it, its learner (``None`` for a linear
    ``bols``), and the data columns it reads."""

    parameter: str | None
    label: str
    spec: object | None
    columns: tuple


@dataclass(frozen=True)
class _Recipe:
    """What the fit's transform recipe and ``%X%`` terms add to the raw columns."""

    #: derived ``%X%`` column -> the two marginal columns it reads
    tensor_inputs: dict
    #: derived ``%X%`` column -> the term as written, ``bbs(x) %X% bbs(z)``
    tensor_names: dict
    #: derived transform column -> the raw columns it reads
    sources: dict
    #: raw columns the fit imputes (``na_action='impute'``)
    imputed: frozenset
    #: the recipe's factor steps
    factors: tuple


def validate(model, newdata, *, offset=None, output="auto"):
    """The issues that ``predict(newdata)`` would meet, before it runs.

    Parameters
    ----------
    model : GAMBoostModel, GLMBoostModel or LSSModel
        A fitted model.
    newdata : dict or DataFrame
        The rows that will be predicted, with raw columns as ``predict`` takes
        them.
    offset : number or array-like, optional
        The offset ``predict`` would receive. Checked against the rows of
        ``newdata`` by the rules ``predict(offset=)`` uses.
    output : {"auto", "pandas", "polars", "records"}, default "auto"
        ``"records"`` returns a list of dicts. The other values return a frame
        with one row per issue, following the package's ``output=`` convention.

    Returns
    -------
    list of dict, or DataFrame
        One record per issue, with the keys in ``RECORD_COLUMNS``. An empty
        result means the checks listed in the module documentation found
        nothing. Every value is a plain Python type, so the records can be
        passed to ``json.dumps``.

    Notes
    -----
    ``validate`` replays the transforms, ``%X%`` designs and factor encodings
    that ``predict`` replays, so a model whose data cannot be replayed raises
    here rather than returning issues. It never predicts and never modifies
    ``newdata``.

    The range checks of ``predict`` do not cover ``btree``, ``bkernel``,
    ``buser`` and ``brad`` terms, because those have no training-range
    boundary. ``validate`` lists each of them as ``not_range_checked``.
    """
    if newdata is None:
        raise TypeError(
            "validate() needs newdata: the rows you intend to predict. "
            "Pass the frame or mapping of new observations."
        )
    model_family, terms, parameters = _model_terms(model)
    recipe = _recipe(model)
    response = _response_names(model)
    specs = [term.spec for term in terms if term.spec is not None]

    present = column_names(newdata)
    present_set = set(present)

    readers = {}
    for term in terms:
        for column in term.columns:
            readers.setdefault(column, []).append(term)
    required = list(readers)
    missing = [column for column in required if column not in present_set]
    readable = [column for column in required if column in present_set]

    lengths = {column: len(read_column(newdata, column).values) for column in readable}
    if not lengths and present:
        lengths = {present[0]: len(read_column(newdata, present[0]).values)}
    n_rows = int(next(iter(lengths.values()))) if lengths else 0

    issues = []

    def add(kind, severity, **fields):
        issues.append(_record(kind, severity, n_rows=n_rows, **fields))

    if not present:
        add("row_count", "error", count=0,
            message="newdata has no columns, so it has no rows to predict",
            suggested_policy="pass a frame or mapping with the columns the model reads")
    elif len(set(lengths.values())) > 1:
        described = ", ".join(f"{name!r} has {length}" for name, length in lengths.items())
        add("row_count", "error",
            message=f"the columns of newdata have different lengths: {described}",
            suggested_policy="pass columns of equal length, one value per row")
    elif n_rows == 0:
        add("row_count", "error", count=0, message="newdata has no rows",
            suggested_policy="pass at least one row to predict")

    for column in missing:
        labels = ", ".join(term.label for term in _unique_terms(readers[column]))
        add("missing_column", "error", column=column, term=labels,
            message=f"column {column!r} is read by {labels} but is not in newdata",
            suggested_policy=f"add column {column!r} to newdata, with the values the model "
                             "was fitted on")

    missing_values = []
    for column in readable:
        n_missing = int(read_column(newdata, column).n_missing)
        if not n_missing:
            continue
        labels = ", ".join(term.label for term in _unique_terms(readers[column]))
        if column in recipe.imputed:
            add("na", "warning", column=column, term=labels, count=n_missing,
                message=f"column {column!r} has {n_missing} missing value(s) in {n_rows} rows; "
                        "the fit imputes this column",
                suggested_policy="none needed: predict applies the fitted imputation to these rows")
        else:
            missing_values.append(column)
            add("na", "error", column=column, term=labels, count=n_missing,
                message=f"column {column!r} has {n_missing} missing value(s) in {n_rows} rows, "
                        "which predict refuses",
                suggested_policy=_refused_column_policy(model, newdata, column))

    _unseen_level_issues(newdata, recipe.factors, specs, present_set, add)

    if missing:
        add("check_skipped", "warning", column=None,
            message="extrapolation check not run: "
                    + ", ".join(repr(column) for column in missing) + " missing from newdata",
            suggested_policy="add the missing columns, then validate again")
    elif missing_values:
        add("check_skipped", "warning", column=None,
            message="extrapolation check not run: predict cannot evaluate the basis at "
                    "missing values in " + ", ".join(repr(column) for column in missing_values),
            suggested_policy="resolve the na issues above, then validate again")
    elif n_rows:
        _extrapolation_issues(model, newdata, add)

    for term in terms:
        if term.spec is not None and _is_unchecked(term.spec, recipe):
            add("not_range_checked", "warning", parameter=term.parameter, term=term.label,
                column=term.columns[0] if term.columns else None,
                message=f"{term.label} has no training-range boundary, so predict does not "
                        "check newdata against its training data for this term",
                suggested_policy="no extrapolation= policy applies; compare its covariates "
                                 "with the training data before trusting the prediction")

    issues.extend(_extra_column_issues(present, required, response, recipe, n_rows))
    issues.extend(_offset_issues(model, model_family, offset, n_rows, parameters))

    return output_frame([json_safe(issue) for issue in issues], output,
                        columns=RECORD_COLUMNS)


# ──────────────────────────────────────────────────────────────────────────
# Model description
# ──────────────────────────────────────────────────────────────────────────


def _model_terms(model):
    """``(family, terms, parameters)`` for a GAM, GLM or LSS model.

    ``parameters`` lists the distribution-parameter names of an LSS model and
    is ``None`` otherwise.
    """
    from mboostrs.gamboost import GAMBoostModel
    from mboostrs.gamlss import LSSModel
    from mboostrs.glmboost import GLMBoostModel

    recipe = _recipe(model)
    if isinstance(model, GAMBoostModel):
        terms = [
            _Term(None, _spec_label(spec, recipe), spec, _spec_columns(spec, recipe))
            for spec in model._learners_spec or ()
        ]
        return model._family, terms, None
    if isinstance(model, GLMBoostModel):
        terms = [_Term(None, f"bols({quote_name(name)})", None, _resolve(name, recipe))
                 for name in model._predictor_names or ()]
        return model._family, terms, None
    if isinstance(model, LSSModel):
        terms = []
        for k, parameter in enumerate(model.param_names):
            p = model._params[k]
            if p.get("kind") == "gam":
                labels = model._learner_labels(k)
                for label, spec in zip(labels, p["learners"], strict=False):
                    terms.append(_Term(parameter, label, spec, _spec_columns(spec, recipe)))
            else:
                for name in p.get("predictor_names") or ():
                    terms.append(_Term(parameter, f"bols({quote_name(name)})", None,
                                       _resolve(name, recipe)))
        return model._family, terms, list(model.param_names)
    raise TypeError(
        "validate() takes a fitted gamboost, glmboost or gamboostLSS model, got "
        f"{type(model).__name__}"
    )


def _recipe(model):
    """The transform recipe and ``%X%`` maps of a fitted model, as one record."""
    tensor_inputs, tensor_names = {}, {}
    for interaction in getattr(model, "_interactions", None) or ():
        tensor_inputs[interaction.derived] = (interaction.left.column, interaction.right.column)
        tensor_names[interaction.derived] = interaction.name
    recipe = getattr(model, "_transform_recipe", None)
    steps = recipe.steps if recipe else ()
    sources, imputed = {}, set()
    for step in steps:
        sources.setdefault(step.name, tuple(step.sources))
        if step.kind in ("impute", "impute_model", "impute_level"):
            imputed.add(step.source)
    factors = tuple(step for step in steps if step.kind == "factor")
    return _Recipe(tensor_inputs, tensor_names, sources, frozenset(imputed), factors)


def _resolve(name, recipe):
    """The raw data columns that a learner's column name stands for.

    A ``%X%`` derived column reads its two marginal columns, and a transform
    column (``log(x)``) reads its source; any other name is a raw column.
    """
    if name in recipe.tensor_inputs:
        return tuple(recipe.tensor_inputs[name])
    if name in recipe.sources:
        return recipe.sources[name]
    return (name,)


def _spec_columns(spec, recipe):
    """The raw columns a learner reads, in the order its arguments name them.

    A ``buser`` is a design the caller supplies (``buser_newdata``), so its
    name is not a data column; a ``%X%`` design is the exception, its derived
    name stands for two marginal columns. A radial ``brad`` reads the columns
    its basis was built on.
    """
    from mboostrs.learners import Buser

    names = []
    radial = getattr(spec, "_radial_basis", None)
    own_names = getattr(spec, "names", None)
    if radial and radial.get("names"):
        names.extend(radial["names"])
    elif isinstance(own_names, (list, tuple)):
        # btree and brad take a list of columns; their `name` property joins it.
        names.extend(name for name in own_names if isinstance(name, str))
    elif not isinstance(spec, Buser) or getattr(spec, "name", None) in recipe.tensor_inputs:
        # A bspatial has two coordinates; its `name` property joins them.
        coordinates = [getattr(spec, "name1", None), getattr(spec, "name2", None)]
        if all(isinstance(value, str) for value in coordinates):
            names.extend(coordinates)
        elif isinstance(getattr(spec, "name", None), str):
            names.append(spec.name)
    by = getattr(spec, "by", None)
    if isinstance(by, str):
        names.append(by)
    columns = []
    for name in names:
        for column in _resolve(name, recipe):
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _spec_label(spec, recipe):
    """The term as the user wrote it: ``bbs(x)``, ``bbs(x) %X% bbs(z)``."""
    name = getattr(spec, "name", None)
    if name in recipe.tensor_names:
        return recipe.tensor_names[name]
    radial = getattr(spec, "_radial_basis", None)
    if radial and radial.get("names"):
        return f"brad({', '.join(quote_name(_c) for _c in radial['names'])})"
    kind = type(spec).__name__.lower()
    if kind == "btree":
        return f"btree({', '.join(quote_name(_c) for _c in spec.names)})"
    label = f"{kind}({name})"
    by = getattr(spec, "by", None)
    return f"{label}, by={quote_name(by)}" if by is not None else label


def _response_names(model):
    """The response column names. A model reads them but no term does."""
    response = getattr(model, "_response_name", None)
    if response is None:
        return set()
    if isinstance(response, tuple):
        return set(response)
    return {response}


def _unique_terms(terms):
    """``terms`` without repeats, in order."""
    seen, out = set(), []
    for term in terms:
        key = (term.parameter, term.label)
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def _is_unchecked(spec, recipe):
    """Whether predict has no training-range check for ``spec``.

    A ``%X%`` design is a ``buser`` too, but its marginals are checked, so it
    is not listed.
    """
    from mboostrs.learners import Bkernel, Btree, Buser

    if isinstance(spec, Buser) and getattr(spec, "name", None) in recipe.tensor_names:
        return False
    return isinstance(spec, (Btree, Bkernel, Buser))


# ──────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────


def _unseen_level_issues(newdata, factor_steps, specs, present, add):
    """Unseen factor levels per column, by the encoding predict uses.

    Each factor step is encoded by the function predict's replay calls, with
    unseen labels let through as code ``-1``. Missing labels are not counted
    as unseen; they are reported under ``na``.
    """
    from mboostrs.glmboost import _brandom_only_columns
    from mboostrs.transforms import _encode_factor

    brandom_only = _brandom_only_columns(specs)
    for step in factor_steps:
        column = step.source
        if column not in present:
            continue
        try:
            codes = _encode_factor(step, newdata, allow_unseen=True)
        except ValueError as exc:
            add("unseen_level", "error", column=column, message=str(exc),
                suggested_policy="convert the column back to the dtype the model was fitted "
                                 "with; no unseen_levels policy repairs a change of type")
            continue
        raw = newdata[column]
        listed = (raw.to_numpy() if hasattr(raw, "to_numpy") else np.asarray(raw)).tolist()
        unseen = np.asarray(codes) == -1
        unseen &= np.array([not _is_missing(value) for value in listed], dtype=bool)
        if not unseen.any():
            continue
        levels = sorted({json_safe(listed[i]) for i in np.flatnonzero(unseen)}, key=str)
        if column in brandom_only:
            policy = ("unseen_levels='average' predicts these rows as an average group; "
                      "unseen_levels='zero_effect' gives them a zero effect")
        else:
            policy = ("no unseen_levels policy applies to this column, because only brandom "
                      "and btree terms accept unseen levels; drop these rows, or refit with "
                      "training data that contains the levels")
        add("unseen_level", "error", column=column, count=int(unseen.sum()), levels=levels,
            message=f"column {column!r} holds {len(levels)} level(s) the fit never saw: "
                    + ", ".join(str(level) for level in levels),
            suggested_policy=policy)


def _is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def _refused_column_policy(model, newdata, column):
    """The remedy named for a column that predict refuses because it has missing values.

    ``na_action='impute'`` is always named. When the training data is kept on
    the model and the other covariates predict the column (the R-squared of
    the ``impute_model`` fit on the training rows is at least 0.05), the
    model-based fill is named too, with that R-squared.
    """
    policy = ("fit with na_action='impute' (median fill and an indicator) or "
              "na_action='omit', or fill the values before predicting")
    r2 = _predictability(model, column)
    if r2 is not None and r2 >= 0.05:
        policy = ("fit with na_action='impute_model' (model-based fill from the other "
                  f"covariates, whose R-squared for {column!r} is {r2:.2f}), or "
                  "na_action='impute' (median fill and an indicator), or na_action='omit'; "
                  "or fill the values before predicting")
    return policy


def _predictability(model, column):
    """R-squared of the conditional-mean imputation model for `column`, or None.

    Uses the training data kept on the model. None when no training data is kept
    or the column is not numeric.
    """
    from mboostrs.transforms import _fit_imputation_model, _is_categorical

    data = (getattr(model, "_refit_bundle", None) or {}).get("raw_data")
    if data is None:
        return None
    try:
        if column not in (data.columns if hasattr(data, "columns") else data.keys()):
            return None
        if _is_categorical(data, column):
            return None
        others = [c for c in (data.columns if hasattr(data, "columns") else data.keys())
                  if c != column]
        step = _fit_imputation_model(column, data, others)
    except (TypeError, ValueError):
        return None
    return float(step.params[2])


def _extrapolation_issues(model, newdata, add):
    """One record per term and covariate with rows outside its training range.

    The records are those of ``extrapolation_report``, which is the table the
    ``predict`` range policy acts on.
    """
    try:
        records = model.extrapolation_report(newdata, output="records")
    except (ValueError, KeyError, TypeError) as exc:
        add("check_skipped", "warning", message=f"extrapolation check not run: {exc}",
            suggested_policy="the extrapolation check reads the training data the model "
                             "stores; refit the model to restore it")
        return
    for record in records:
        if record["kind"] == "linear":
            policy = ("no extrapolation= policy applies to a linear term, which continues the "
                      "fitted line; check worst_distance before trusting the prediction there")
        else:
            policy = ("predict(extrapolation='clip') holds the effect at the training edge; "
                      "extrapolation='error' refuses these rows; extrapolation='allow' keeps "
                      "the edge-slope continuation")
        add("extrapolation", "warning", parameter=record["parameter"], term=record["term"],
            column=record["covariate"], count=int(record["n_outside"]),
            worst_distance=float(record["worst_distance"]),
            message=f"{record['term']}, covariate {record['covariate']}: "
                    f"{record['n_outside']} row(s) outside the training range "
                    f"[{record['train_low']:.6g}, {record['train_high']:.6g}], farthest "
                    f"{record['worst_distance']:.2g} training ranges out",
            suggested_policy=policy)


def _extra_column_issues(present, required, response, recipe, n_rows):
    """Columns of ``newdata`` that no term reads and that are not the response.

    The notices are one issue, whatever the number of columns: a frame that
    carries many unused columns gets one record that lists them in its message
    and gives their count, not one record per column.
    """
    used = set(required) | response | set(recipe.tensor_inputs) | set(recipe.sources)
    # A raw column that a derived column is built from (log__x from x, x_missing
    # from x, a %X% tensor's margins) is read through that derived column, not
    # unread. The recorded transform recipe gives the mapping. Without a recipe
    # entry (a derived column the caller built, such as households__log), the
    # name is the only link, so a column is matched by the suffix rule below.
    # A %X% tensor's margins are not replaced by a derived column: the tensor
    # term reads them, and they are in `used`.
    through: dict[str, list[str]] = {}
    for derived, sources in recipe.sources.items():
        for raw in sources:
            if isinstance(raw, str) and raw != derived:
                through.setdefault(raw, []).append(derived)
    for column in present:
        if column in through:
            continue
        for name in required:
            if _is_derived_name(name, column):
                through.setdefault(column, []).append(name)
    issues = []
    # A source is read through its derived column even when the term resolves to
    # the source too: the recipe puts the source in `required`, so it is not
    # tested against `used` here.
    read = [column for column in present if column in through]
    if read:
        described = ", ".join(
            f"{column} ({', '.join(dict.fromkeys(through[column]))})" for column in read)
        issues.append(_record(
            "read_through_transform", "info", column=read[0] if len(read) == 1 else None,
            count=len(read), n_rows=n_rows,
            message=f"read through derived columns: {described}",
            suggested_policy="none needed; the derived column named in parentheses is "
                             "what the term reads"))
    extra = [column for column in present if column not in used and column not in through]
    if not extra:
        return issues
    names = ", ".join(repr(column) for column in extra)
    verb = "is not read" if len(extra) == 1 else "are not read"
    noun = "column" if len(extra) == 1 else "columns"
    issues.append(_record(
        "extra_column", "warning", column=extra[0] if len(extra) == 1 else None,
        count=len(extra), n_rows=n_rows,
        message=f"{len(extra)} {noun} {verb} by any term; predict ignores "
                f"{'it' if len(extra) == 1 else 'them'}: {names}",
        suggested_policy="none needed; drop the column(s) to silence this issue"))
    return issues


def _is_derived_name(name, raw):
    """Whether the column ``name`` is a derived column built from the column ``raw``.

    The suffix rule: ``raw__<transform>`` (households__log), ``<transform>__raw``
    (log__households, the name `impute_missing` and the transform recipe use),
    and ``raw_missing`` or ``raw_missing_`` (the missing-value indicator).
    """
    return (name.startswith(f"{raw}__") or name.endswith(f"__{raw}")
            or name in (f"{raw}_missing", f"{raw}_missing_"))


def _offset_issues(model, family, offset, n_rows, parameters):
    """Offset checks, by the rules ``predict`` applies to its ``offset=``."""
    from mboostrs.glmboost import _resolve_prediction_offset

    if parameters is not None:
        if offset is None:
            return []
        return [_record(
            "offset", "error", n_rows=n_rows,
            message="gamboostLSS predict() takes no offset=, so the offset given here "
                    "would be ignored",
            suggested_policy="drop offset=; the fitted parameter offsets are used")]
    try:
        _resolve_prediction_offset(getattr(model, "_offset", None), offset, family,
                                   n_rows, requires_explicit=True)
    except (ValueError, TypeError) as exc:
        return [_record(
            "offset", "error", n_rows=n_rows, message=str(exc),
            suggested_policy="pass offset= with one value per row of newdata "
                             "(log(exposure) of the new rows for a rate model)")]
    return []


def _record(kind, severity, *, parameter=None, term=None, column=None, count=None,
            n_rows=None, worst_distance=None, levels=None, message="",
            suggested_policy=""):
    """One issue record, with every key of :data:`RECORD_COLUMNS` present."""
    return {
        "kind": kind,
        "severity": severity,
        "parameter": parameter,
        "term": term,
        "column": column,
        "count": count,
        "n_rows": n_rows,
        "worst_distance": worst_distance,
        "levels": levels,
        "message": message,
        "suggested_policy": suggested_policy,
    }
