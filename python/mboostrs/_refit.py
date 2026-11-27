"""One description of how a model was fitted, so it can be fitted again.

Several operations refit a model: continuation to a larger `mstop`, bootstrap
confidence intervals, conformal calibration, cross-validation, and stability
selection. Each was reconstructing the call from whatever attributes it
happened to know about, and each knew a different subset. `conformal.py`'s
version passed the formula, the data, the family and the control, and dropped
the weights, `center`, the row and component subsampling settings, explicitly
supplied learners, `env`, and `bins` -- so a conformal interval computed on a
weighted, centred, binned fit was calibrated against an unweighted, uncentred,
exact one, and reported a coverage guarantee for a model the caller had not
fitted.

A `RefitSpec` holds both the fitted prediction recipe and the raw effective
training call. Prediction replay reuses fitted transformations. Validation
subsets call ``subset(refit_preprocessing=True)`` and refit stateful transforms
and formula interactions on their proper training rows.

**Row subsetting is a property of the spec, not of the caller.** A refit on a
subset of rows has to subset every row-aligned object together -- the data and
the weights at least, and a `Buser` design matrix if there is one -- and leave
every other argument alone. `subset()` does that; a caller cannot forget the
weights because a caller no longer touches them.

**What cannot be subset is named once.** A learner can carry state that is
tied to the training rows or to the levels present in them: a `Buser` design
matrix has one row per observation, a `Bmrf` penalty has one row per region,
and an interaction recipe was built against a particular set of columns.
`row_subset_blockers()` lists those, so an operation that needs a row subset
can refuse with the specific reason instead of a blanket "not supported" or,
worse, a silent misalignment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

#: Arguments that are aligned with the training rows and must be subset
#: together with the data. Everything else is a hyperparameter of the fit and
#: is carried through a subset unchanged.
#: Arguments with one entry per row; a scalar offset is not one of them.
ROW_ALIGNED_ARGUMENTS = ("weights", "offset")

#: The design of a `%X%` tensor term, as `expand_interactions` names it in
#: `env` (`__ix0_X` for the first term). It has one row per training row, as
#: the data does, so a row subset takes it with the data. Its penalty
#: (`__ix0_K`) has a coefficient axis and is kept whole.
_TENSOR_DESIGN_KEY = re.compile(r"^__ix\d+_X$")


#: Learner classes whose external `K` / `C` matrices are indexed by basis
#: coefficients, never by observations. Their row-indexed design, when they
#: have one (`Buser.X`), is checked separately.
_COEFFICIENT_AXIS_ATTRIBUTES = frozenset({"Bconstrained", "Buser", "Bmono", "Bbs", "Bns"})


@dataclass(frozen=True)
class RefitSpec:
    """Everything needed to reproduce a fit, and to reproduce it on a subset.

    ``kind`` selects the entry point; ``arguments`` holds the call it takes,
    less ``data``, which is passed separately because it is the thing a subset
    replaces.
    """

    kind: str
    formula: Any
    data: Any
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Recipes applied to ``data`` before fitting, carried so a rebuilt model
    #: predicts through the same derived columns.
    transform_recipe: Any = None
    #: The response's own transform step (`log(rent) ~ ...`), carried beside
    #: the predictor recipe for the same reason the recipe is: the rewritten
    #: formula does not re-extract on a refit.
    response_transform: Any = None
    interactions: Any = None
    #: Original pre-transformation call, retained for training a pipeline on a
    #: row subset. Prediction replay continues to use ``formula``/``data`` and
    #: ``transform_recipe`` above.
    raw_formula: Any = None
    raw_data: Any = None
    raw_arguments: dict[str, Any] | None = None
    refits_preprocessing: bool = False

    def entry_point(self):
        from mboostrs.gamboost import gamboost
        from mboostrs.gamlss import gamboostLSS, glmboostLSS
        from mboostrs.glmboost import glmboost

        return {
            "glmboost": glmboost,
            "gamboost": gamboost,
            "glmboostLSS": glmboostLSS,
            "gamboostLSS": gamboostLSS,
        }[self.kind]

    def build(self, **overrides):
        """Fit the model this spec describes, with ``overrides`` applied."""
        arguments = {**self.arguments, **overrides}
        arguments = {k: v for k, v in arguments.items() if v is not None
                     or k in ("weights", "offset")}
        # `gamboost` refuses `learners=` and `env=` together. When the fit
        # rewrote the formula -- an interaction lowers to
        # `buser(__ix0, X=__ix0_X, ...)` -- `env` is what resolves those names,
        # so the formula-plus-env pair is what reproduces the fit. The spec
        # keeps both because `row_subset_blockers()` reads the learners to see
        # what a row subset would misalign; only the *call* has to choose.
        if arguments.get("env") is not None:
            arguments.pop("learners", None)
        model = self.entry_point()(
            self.formula, data=self.data, **arguments
        )
        if self.transform_recipe is not None and not self.refits_preprocessing:
            model._transform_recipe = self.transform_recipe
        # The response step rides along with the recipe for the same reason:
        # the rewritten formula triggers no re-extraction on a refit, so the
        # step would otherwise be lost on a continued or sliced model.
        if self.response_transform is not None and not self.refits_preprocessing:
            model._response_transform = self.response_transform
        if self.interactions is not None and not self.refits_preprocessing:
            model._interactions = self.interactions
            # The refit reads the rewritten formula, whose learners carry the
            # derived columns (`__ix0`, `__by0`); name them by the terms the
            # caller wrote, as the original fit does, so `which=` and every
            # table agree between a model and its refits.
            labels = {
                getattr(item, "derived", None): getattr(item, "name", None)
                for item in self.interactions
            }
            if hasattr(model, "_learner_names"):
                model._learner_names = [
                    labels.get(name) or name for name in model._learner_names
                ]
        return model

    def subset(self, rows, *, refit_preprocessing: bool = False) -> RefitSpec:
        """The same fit, restricted to ``rows``.

        Every row-aligned argument is subset with the data, so the two cannot
        come apart. With ``refit_preprocessing=True``, use raw effective inputs
        and refit stateful transformations and formula interactions inside the
        subset; the default preserves the fitted design for upstream-compatible
        replay. Call :meth:`row_subset_blockers` first: this does not
        check, because a caller that has already decided to proceed should not
        pay for the check twice.
        """
        rows = np.asarray(rows)
        if refit_preprocessing:
            if self.raw_data is None or self.raw_formula is None:
                raise ValueError(
                    "this model did not retain raw pre-transformation inputs, "
                    "so preprocessing cannot be fitted inside the training subset"
                )
            data = self.raw_data
            formula = self.raw_formula
            arguments = dict(self.raw_arguments or self.arguments)
        else:
            data = self.data
            formula = self.formula
            arguments = dict(self.arguments)
        for name in ROW_ALIGNED_ARGUMENTS:
            value = arguments.get(name)
            if value is not None and np.ndim(value) > 0:
                arguments[name] = np.asarray(value)[rows]
        if arguments.get("env") is not None:
            arguments["env"] = _subset_tensor_designs(
                arguments["env"], rows, _row_count(data))
        sub_data = _subset_rows(data, rows)
        formula, arguments = _guard_constant_spline_subset(formula, sub_data, arguments)
        return replace(
            self, formula=formula, data=sub_data, arguments=arguments,
            transform_recipe=None if refit_preprocessing else self.transform_recipe,
            response_transform=None if refit_preprocessing else self.response_transform,
            interactions=None if refit_preprocessing else self.interactions,
            refits_preprocessing=refit_preprocessing,
        )

    def row_subset_blockers(self, *, refit_preprocessing: bool = False) -> list[str]:
        """Reasons a refit on a subset of the rows would be wrong.

        Empty when a row subset is safe. Each entry names the object and what
        about it does not survive the subset, so the caller's error message can
        say which learner to remove or replace.
        """
        blockers = []
        data = self.raw_data if refit_preprocessing and self.raw_data is not None else self.data
        arguments = (self.raw_arguments if refit_preprocessing and self.raw_arguments is not None
                     else self.arguments)
        n = _row_count(data)

        # The processed learner specifications are also the authoritative
        # record of external matrices resolved from a formula's env. Raw call
        # arguments may contain only env=, so inspecting them alone misses a
        # row-aligned Buser X or a Bmrf graph.
        learners = self.arguments.get("learners") or arguments.get("learners") or []
        rebuilt_interactions = (
            {getattr(item, "derived", None) for item in (self.interactions or [])}
            if refit_preprocessing else set()
        )
        for learner in learners:
            name = type(learner).__name__
            design = getattr(learner, "X", None)
            if (design is not None and np.asarray(design).shape[0] == n
                    and getattr(learner, "name", None) not in rebuilt_interactions):
                blockers.append(
                    f"{name}({learner.name!r}) carries a user-supplied design "
                    f"matrix with one row per observation; subsetting the data "
                    f"without subsetting it would misalign the two"
                )
            elif name == "Bmrf":
                blockers.append(
                    f"Bmrf({learner.name!r}) carries a neighbourhood penalty "
                    f"indexed by the regions present in the training data; a "
                    f"row subset that drops a region leaves the penalty "
                    f"describing a graph the design no longer has"
                )
            elif name in _COEFFICIENT_AXIS_ATTRIBUTES:
                # The library's own learners declare what their external
                # matrices are indexed by. `Bconstrained.C` is (constraints,
                # coefficients) and `Buser.K` is a (coefficients, coefficients)
                # penalty: neither has a row axis, so a row subset cannot
                # misalign them, even when the number of coefficients happens
                # to equal the row count (24 rows and the default 24-column
                # spline basis used to be refused for that reason alone).
                continue
            else:
                # An object the library does not know: an external matrix
                # with a dimension equal to the row count may be indexed by
                # rows, so refuse rather than guess.
                for attribute in ("K", "C"):
                    matrix = getattr(learner, attribute, None)
                    if matrix is None:
                        continue
                    shape = np.asarray(matrix).shape
                    if n in shape:
                        blockers.append(
                            f"{name}({learner.name!r}) carries an external "
                            f"`{attribute}` of shape {shape}, which has a "
                            f"dimension of length {n} -- the training row "
                            f"count -- so a row subset would misalign it"
                        )

        if arguments.get("subsample") is not None:
            blockers.append(
                "the fit used row subsampling (subsample=), so it is already a "
                "fit of a random subset; nesting another subset would make the "
                "refit's sampling distribution differ from the original's"
            )

        return blockers

    def describe_weights(self) -> str:
        """One line on whether this refit carries weights, for a guarantee."""
        weights = self.arguments.get("weights")
        if weights is None:
            return "unweighted"
        return f"weighted (policy {self.arguments.get('weight_policy', 'upstream')!r})"


def _row_count(data) -> int:
    if hasattr(data, "height"):
        return int(data.height)
    if hasattr(data, "shape") and not isinstance(data, dict):
        return int(data.shape[0])
    for value in dict(data).values():
        return len(np.asarray(value).ravel())
    return 0


def _subset_tensor_designs(env, rows, n: int):
    """`env` with each `%X%` tensor design taken to ``rows``.

    The tensor design is evaluated at every training row with the basis fixed
    on the full training frame, so a row subset keeps the fitted basis and
    takes the matching rows, exactly as a `bbs` term's design does. A design
    whose first axis is not the row count is left alone.
    """
    out = dict(env)
    for key, value in env.items():
        if (_TENSOR_DESIGN_KEY.match(key) and value is not None
                and np.ndim(value) == 2 and np.shape(value)[0] == n):
            out[key] = np.asarray(value)[rows]
    return out


def _subset_rows(data, rows):
    """Take ``rows`` from a mapping, a polars frame, or a pandas frame.

    Delegates to the one row-subsetting implementation in `stabsel`, which
    dispatches on frame type. A second copy here would be a second place for
    the polars/pandas order to be got wrong.
    """
    from mboostrs.stabsel import _take_rows

    return _take_rows(data, np.asarray(rows))


def _column_constant_value(data, col: str) -> float | None:
    """Return the unique finite scalar value if numeric `col` is constant on `data`."""
    try:
        if isinstance(data, dict):
            if col not in data:
                return None
            raw = data[col]
        elif hasattr(data, "columns"):
            if col not in data.columns:
                return None
            raw = data[col]
        else:
            return None
        arr = raw.to_numpy() if hasattr(raw, "to_numpy") else np.asarray(raw)
        vals = np.asarray(arr, dtype=float)
    except (TypeError, ValueError, KeyError):
        return None
    present = vals[np.isfinite(vals)]
    if present.size == 0:
        return None
    lo, hi = float(np.min(present)), float(np.max(present))
    return lo if lo == hi else None


def _guard_formula_constant_bbs(formula: str, sub_data) -> str:
    """Inject a leak-free boundary.knots pair when a row subset collapses a bbs column."""
    import re

    from mboostrs._names import partition_tilde, split_terms, unquote_name

    lhs, sep, rhs = partition_tilde(formula)
    if not sep or "bbs(" not in rhs:
        return formula
    terms = split_terms(rhs)
    changed = False
    new_terms = []
    bbs_re = re.compile(r"^\s*bbs\(\s*(?:(log|sqrt)\(\s*([^)]+?)\s*\)|([^,)]+?))(?:\s*,\s*(.*))?\)\s*$")
    for term in terms:
        if "%X%" in term or "boundary.knots" in term or "boundary_knots" in term:
            new_terms.append(term)
            continue
        m = bbs_re.match(term)
        if not m:
            new_terms.append(term)
            continue
        transform, tcol, rawcol, rest = m.groups()
        col = unquote_name((tcol or rawcol).strip())
        cval = _column_constant_value(sub_data, col)
        if cval is None:
            new_terms.append(term)
            continue
        if transform == "log" and cval > 0:
            cval = float(np.log(cval))
        elif transform == "sqrt" and cval >= 0:
            cval = float(np.sqrt(cval))
        bk = f"boundary.knots=c({cval!r}, {cval + 1.0!r})"
        inner = f"{transform}({tcol.strip()})" if transform else rawcol.strip()
        new_term = f"bbs({inner}, {rest.strip()}, {bk})" if (rest and rest.strip()) else f"bbs({inner}, {bk})"
        new_terms.append(new_term)
        changed = True
    return f"{lhs.strip()} ~ " + " + ".join(new_terms) if changed else formula


def _guard_constant_spline_subset(formula, sub_data, arguments: dict) -> tuple[Any, dict]:
    """Keep row-subset refits (e.g. split conformal) valid when a rare-event spline column becomes constant."""
    from copy import copy

    learners = arguments.get("learners")
    if learners:
        new_learners = None
        for idx, learner in enumerate(learners):
            if type(learner).__name__ == "Bbs" and getattr(learner, "boundary_knots", None) is None:
                cval = _column_constant_value(sub_data, learner.name)
                if cval is not None:
                    if new_learners is None:
                        new_learners = list(learners)
                    clone = copy(learner)
                    clone.boundary_knots = (cval, cval + 1.0)
                    new_learners[idx] = clone
        if new_learners is not None:
            arguments = {**arguments, "learners": new_learners}
            return formula, arguments

    if isinstance(formula, str):
        return _guard_formula_constant_bbs(formula, sub_data), arguments
    if isinstance(formula, dict):
        return {k: _guard_formula_constant_bbs(v, sub_data) for k, v in formula.items()}, arguments
    return formula, arguments


def refit_spec(model) -> RefitSpec:
    """The refit specification for a fitted model.

    Built from the bundle the fit stored, so a hyperparameter the fit accepted
    is carried rather than re-derived. A model that did not retain one -- an
    unpickled one, say -- raises here rather than at the point a silently
    incomplete refit would have gone wrong.
    """
    kind = _model_kind(model)

    if kind in ("glmboostLSS", "gamboostLSS"):
        arguments = dict(model._refit_arguments())
        formula = arguments.pop("formula")
        data = arguments.pop("data")
        arguments.setdefault("weights", getattr(model, "_input_weights", None))
        # The rewritten formula names each tensor term's `buser` symbol, and
        # the refit reads the interaction records to replay those terms on new
        # rows. The records are carried as the gamboost branch below does.
        # The inputs before the transforms, when the fit kept them: a refit
        # with `refit_preprocessing=True` estimates the transforms again on
        # its own rows from these.
        raw = getattr(model, "_raw_inputs", None) or {}
        # Raw-input refits rebuild any `%X%` interaction matrices from `raw_data`
        # and `raw_formula`, so the full fit's generated `env` must be omitted.
        raw_args = (
            {**{k: v for k, v in arguments.items() if k != "env"}, "na_action": raw["na_action"]}
            if raw else None
        )
        return RefitSpec(kind=kind, formula=formula, data=data,
                         arguments=arguments,
                         interactions=getattr(model, "_interactions", None),
                         raw_formula=raw.get("formula"), raw_data=raw.get("data"),
                         raw_arguments=raw_args)

    bundle = getattr(model, "_refit_bundle", None)
    if bundle is None:
        raise ValueError(
            "this model did not retain the inputs needed to refit it (it is "
            "a scoring-only artifact, was saved or pickled by a release that "
            "did not keep them, or was built without them). Refit it from the "
            "original call."
        )

    # A sliced model (`fit[:k]`, or one truncated to a cross-validated
    # optimum) keeps the control it was fitted with, whose mstop is the
    # original path length. Refits reproduce the model as it stands, as
    # mboost's resampling does, so the control carries the current mstop;
    # callers that want another budget (continue_to, stability selection's
    # override) replace it explicitly.
    from mboostrs._control import as_control

    control = as_control(model._control)
    current = getattr(model, "mstop", None)
    if isinstance(current, (int, np.integer)) and int(current) >= 1:
        control = control._replace(mstop=int(current))
    arguments = {
        "family": bundle.get("family"),
        "control": control,
        "weights": bundle.get("weights"),
        "offset": bundle.get("offset"),
        "weight_policy": bundle.get("weight_policy", "upstream"),
    }
    if kind == "gamboost":
        arguments["env"] = bundle.get("env")
        arguments["bins"] = bundle.get("bins")
        arguments["learners"] = getattr(model, "_learners_spec", None)
    else:
        arguments["center"] = bundle.get("center")

    for name in ("subsample", "subsample_seed", "na_action"):
        if name in bundle:
            arguments[name] = bundle[name]

    # `center=False` and `subsample_seed=0` are meaningful values, so only
    # genuinely absent arguments are dropped.
    return RefitSpec(
        kind=kind,
        formula=bundle.get("formula", model._formula),
        data=bundle.get("data", model._data),
        arguments={k: v for k, v in arguments.items() if v is not None},
        transform_recipe=getattr(model, "_transform_recipe", None),
        response_transform=getattr(model, "_response_transform", None),
        interactions=getattr(model, "_interactions", None),
        raw_formula=bundle.get("raw_formula"),
        raw_data=bundle.get("raw_data"),
        # The raw call arguments carry the control the model was fitted with;
        # the current mstop applies to the raw-pipeline refit as well.
        raw_arguments=(
            {**bundle["raw_arguments"], "control": control}
            if bundle.get("raw_arguments") is not None
            and "control" in bundle["raw_arguments"] else bundle.get("raw_arguments")
        ),
    )


#: The operations that refit the model, and what each one needs the fit's own
#: specification for. A scoring-only artifact carries the fitted path but not
#: the call that produced it, so all of these are unavailable after loading
#: one -- and saying which, once, beats five different failures at five call
#: sites.
REFIT_DEPENDENT_OPERATIONS = (
    ("continue_to", "grows the path by refitting to a larger mstop"),
    ("cvrisk", "refits on each fold's in-bag weights"),
    ("stabsel", "refits on each subsample"),
    ("confint", "refits on bootstrap resamples"),
    ("conformal_interval", "refits on a calibration split"),
    # Slicing to a smaller `mstop` works on any model; only growing needs a
    # refit, so this one is conditionally unavailable and says so.
    ("update", "refits when asked for an mstop larger than the fitted one; "
               "slicing to a smaller one still works"),
)


def refit_available(model) -> tuple[bool, str]:
    """Whether this model can be refitted, and why not when it cannot."""
    if getattr(model, "_refit_bundle", None) is not None:
        return True, ""
    if getattr(model, "_refit_arguments", None) is not None:
        try:
            model._refit_arguments()
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        else:
            return True, ""
    return False, (
        "this model carries no record of the call that produced it. A "
        "scoring-only portable artifact stores the fitted path -- "
        "coefficients, selections, risk -- which is enough to predict and to "
        "read the fit, and not the data, weights or learner arguments a refit "
        "would need. Artifacts and pickles written by earlier releases drop "
        "them too."
    )


def unavailable_operations(model) -> list[dict]:
    """The operations this model cannot perform, and why.

    Empty for a model fitted in this session, and for one unpickled or loaded
    from a full artifact, which keep the fit's call. For a scoring-only
    artifact, every entry of :data:`REFIT_DEPENDENT_OPERATIONS`, so a caller
    can ask once instead of discovering it five times.
    """
    available, reason = refit_available(model)
    if available:
        return []
    return [{"operation": name, "needs": needs, "reason": reason}
            for name, needs in REFIT_DEPENDENT_OPERATIONS]


def require_refit(model, operation: str) -> RefitSpec:
    """The spec, or a refusal naming the operation and what is missing."""
    available, reason = refit_available(model)
    if not available:
        raise ValueError(f"{operation} is unavailable: {reason}")
    return refit_spec(model)


#: The fit arguments `update()` accepts as keywords, by entry point. The
#: formula, the data, `learners`, `env` and `na_action` are absent: the
#: stored spec holds the formula and data after preprocessing (transforms
#: applied, interactions rewritten, rows dropped), so a replacement for any of
#: them would be combined with state built from the old one.
_UPDATE_FIT_ARGUMENTS = {
    "glmboost": ("family", "weights", "offset", "center", "subsample",
                 "subsample_seed", "weight_policy"),
    "gamboost": ("family", "weights", "offset", "bins", "subsample",
                 "subsample_seed", "weight_policy"),
}


def refit_with_updates(model, mstop, updates: dict):
    """`model` fitted again with the keyword arguments of `update()` applied.

    A keyword is either a `BoostControl` field (`nu`, `risk`, `trace`, `step`,
    `step_iterations`), a whole `control=` record, or one of the fit arguments
    in `_UPDATE_FIT_ARGUMENTS`. Every argument not named keeps the value the
    fit used, and the control's `mstop` is the model's current one unless
    `mstop` (or `control=`) sets another. An unsupported name raises
    `TypeError` before anything is fitted.
    """
    from mboostrs._control import BoostControl, as_control, boost_control

    kind = _model_kind(model)
    fit_names = _UPDATE_FIT_ARGUMENTS.get(kind, ())
    control_names = tuple(f for f in BoostControl._fields if f != "mstop")
    accepted = set(fit_names) | set(control_names) | {"control"}
    unknown = sorted(set(updates) - accepted)
    if unknown:
        raise TypeError(
            f"update() got unsupported keyword argument(s) "
            f"{', '.join(repr(name) for name in unknown)}. It accepts mstop, "
            f"control, the control fields {', '.join(control_names)}, and the "
            f"fit arguments {', '.join(fit_names)}. To change the formula, the "
            f"data or the learners, call {kind}() again.")

    spec = require_refit(model, "update")
    control = as_control(updates.get("control") or spec.arguments["control"])
    control = control._replace(**{name: updates[name] for name in control_names
                                  if name in updates})
    if mstop is not None:
        control = control._replace(mstop=int(mstop))
    # Through the validating constructor, so `update(nu=2)` is refused with
    # the same message as `boost_control(nu=2)`.
    control = boost_control(*control)
    overrides = {name: updates[name] for name in fit_names if name in updates}
    return spec.build(control=control, **overrides)


def _model_kind(model) -> str:
    """Which entry point fitted this model.

    Through the shared predicates rather than a class-name test, so a subclass
    is dispatched on what it is rather than on what it is called.
    """
    from mboostrs._dispatch import is_glmboost, is_lss

    if is_lss(model):
        return getattr(model, "_model_kind", "glmboostLSS")
    return "glmboost" if is_glmboost(model) else "gamboost"


def require_row_subsettable(model, operation: str, *, refit_preprocessing: bool = False) -> RefitSpec:
    """The spec, or a refusal naming what stops a row subset being valid."""
    spec = refit_spec(model)
    blockers = spec.row_subset_blockers(refit_preprocessing=refit_preprocessing)
    if blockers:
        reasons = "; ".join(blockers)
        raise NotImplementedError(
            f"{operation} refits the model on a subset of the rows, which this "
            f"fit does not support: {reasons}. Fit on an explicit training "
            f"split and calibrate manually."
        )
    return spec
