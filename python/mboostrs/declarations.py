"""Declared signs and shapes, tested against the fitted association.

A declaration constrains one column's association with the response. In a
specification it is a ``bmono`` term: ``monotone=`` in `autospec` writes one,
and a formula can carry the same term. `check_declarations` tests each such
term against the data.

For each constraint the function fits the specification with that constraint
removed, and measures the association the data support in the unconstrained
model. The constrained model is not the evidence: a monotone constraint forces
its own contrast and its own steps to the declared sign, so it agrees with the
declaration by construction. Held-out folds show whether the measurement holds
up on rows the fit did not see.

The numbers are computed in the Rust core (`mboostrs_core::declarations`); this
module fits models, predicts on the declared variable's grid and passes the
arrays across.

Declarations that name no sign or shape (treatment, groups, measurements,
coordinates, adjacency, basis, interaction rules, locks) are listed with the
reason no test applies. Their verdict is ``inconclusive``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

from mboostrs._control import boost_control
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs._frame import read_column, take_rows
from mboostrs._names import NAME_SRC as N
from mboostrs._names import partition_tilde, unquote_name
from mboostrs._names import split_terms as _split_plus
from mboostrs.formula import term_variables

__all__ = ["DeclarationCheck", "check_declarations"]

#: Declared words that test the sign of the average contrast, with the sign
#: each one declares.
SIGN_CONSTRAINTS = {"positive": 1.0, "negative": -1.0}

#: Declared words that test a curve's steps: the sign each declares, and
#: whether the steps are second differences (curvature) or first differences.
SHAPE_CONSTRAINTS = {"increasing": (1.0, False), "decreasing": (-1.0, False),
                     "convex": (1.0, True), "concave": (-1.0, True)}

#: The contrast runs from the 10th to the 90th percentile of the variable.
CONTRAST_PROBS = (0.1, 0.9)

#: Grid points for a shape test, spanning the 1st to the 99th percentile.
GRID_POINTS = 41
GRID_PROBS = tuple(0.01 + 0.98 * k / (GRID_POINTS - 1) for k in range(GRID_POINTS))

#: Bootstrap refits of the unconstrained model behind each sign verdict, and the
#: two-sided level of the percentile interval they give.
SIGN_BOOTSTRAP_REPLICATES = 200
SIGN_INTERVAL_LEVEL = 0.95

#: Arguments a ``bmono`` term may carry that a ``bbs`` term accepts too. The
#: rest (``constraint``, the solver options) are dropped for the unconstrained
#: refit.
_SHARED_KWARGS = ("df", "knots", "degree", "differences", "center", "by", "lambda")

_BMONO = re.compile(r"bmono\([^()]*\)")
_BSPATIAL = re.compile(rf"bspatial\(\s*({N})\s*,\s*({N})")
_BMRF = re.compile(rf"bmrf\(\s*({N})")
_BUSER = re.compile(rf"buser\(\s*({N})")

_STRUCTURAL_REASONS = {
    "treatment": ("a causal question; an association in the fit does not test it. "
                  "The causal workflow is mboostrs.causal (CausalSpec, aipw, "
                  "negative_control)"),
    "groups": ("a grouping structure fitted as random effects; it declares no sign "
               "or shape"),
    "measurements": ("an ordered quantity fitted as a linear slope; the declaration "
                     "gives no sign to test"),
    "coordinates": ("a two-dimensional surface; no one-dimensional sign or shape is "
                    "declared"),
    "adjacency": "a neighbourhood penalty over a graph; it declares no sign or shape",
    "basis": "a user-supplied basis; it declares no sign or shape",
    "forbid": "an interaction rule that removes a pair from the search; the fit does "
              "not test it",
    "mandatory": "a term the search keeps in place; the fit does not test it",
    "lock": "a predictor protected from later edits; protection states no shape",
}


@dataclass(frozen=True)
class DeclarationCheck:
    """The test of one declaration against the fit.

    `verdict` is ``"agrees"``, ``"contradicts"`` or ``"inconclusive"``. The
    rule that maps measurements to a verdict is in `evidence["rule"]`.
    `agreement` is one sentence, phrased as agreement or disagreement with the
    fitted association. `evidence` holds the measured numbers.
    """

    declaration: str
    kind: str
    variable: str
    #: ``"sign"``, ``"shape"`` or ``None`` when the declaration names no sign or
    #: shape and no test ran.
    test: str | None
    verdict: str
    agreement: str
    evidence: dict = field(default_factory=dict)
    #: Why no measurement decides the verdict. Empty when a test ran.
    reason: str = ""

    def to_dict(self, *, include_arrays=False) -> dict:
        """A JSON-safe record. The average curve of a shape test is an array;
        it is left out unless `include_arrays=True`."""
        record = asdict(self)
        if not include_arrays:
            record["evidence"].pop("curve", None)
        return record

    def __str__(self) -> str:
        return f"{self.declaration}: {self.verdict} -- {self.agreement}"


@dataclass(frozen=True)
class _Spec:
    """What a specification declares, whatever object it came from."""

    formula: str
    family: object
    control: object
    response: str
    env: dict | None = None
    treatment: tuple = ()
    groups: tuple = ()
    measurements: tuple = ()
    forbid: tuple = ()
    mandatory: tuple = ()
    locks: tuple = ()


@dataclass(frozen=True)
class _Item:
    """One declaration, before any fit. `term` is set for a constraint that is
    tested; `reason` is set for one that is not."""

    kind: str
    variable: str
    term: str | None = None
    reason: str = ""

    @property
    def declaration(self) -> str:
        return f"{self.kind}({self.variable})"


def check_declarations(model_or_recommendation, data, *, folds=5, seed=0):
    """Test each declaration of a specification against the data.

    Parameters
    ----------
    model_or_recommendation : SpecRecommendation or GAMBoostModel
        The specification. A recommendation from `autospec` carries its
        declarations; a fitted `gamboost` model carries the ``bmono`` terms of
        its formula.
    data : dict, Polars frame or pandas frame
        The rows the specification was built on. Rows with a missing response
        or a missing formula column are left out.
    folds : int, default 5
        Held-out folds for the out-of-fold measurements. Rows are assigned to
        folds at random, or by group when the recommendation declares
        ``groups=``, so that whole groups are held out.
    seed : int, default 0
        Seed for the fold assignment.

    Returns
    -------
    list of DeclarationCheck
        One record per declaration, in the order the specification lists them.

    Notes
    -----
    A sign declaration (``positive``, ``negative``) is tested on the contrast of
    the variable between its 10th and 90th percentile, computed on the link scale.
    The full-data contrast is compared with its 95% percentile interval over 200
    row-bootstrap refits of the unconstrained model (rows resampled with
    replacement, each refit evaluated on the original rows). The verdict is
    ``agrees`` when the interval excludes zero in the declared direction and at
    least 80% of the five held-out folds (four of five) share the sign of the
    full-data contrast; ``contradicts`` when the interval excludes zero in the
    opposite direction with the same fold agreement; otherwise ``inconclusive``.

    On 30 simulated samples of 300 rows with no association, ``positive`` gave a
    decided verdict (agrees or contradicts) in 3 samples and was inconclusive in
    27. The bootstrap costs 200 refits per sign declaration. The earlier rule,
    which used the Nadeau-Bengio corrected standard error of the fold contrasts,
    decided 8 of the 30 null samples and was replaced.

    A shape declaration (``increasing``, ``decreasing``, ``convex``,
    ``concave``) is tested on the unconstrained curve over a grid spanning the
    1st to the 99th percentile, and on the held-out risk of the model with and
    without the constraint. The verdict is ``agrees`` when at least 90% of the
    steps go the declared way and removing the constraint does not lower
    held-out risk by more than 1%; ``contradicts`` when at most 10% go the
    declared way and removing the constraint lowers held-out risk by more than
    1%; otherwise ``inconclusive``.

    Held-out rows whose other covariates lie outside the training range of a
    fold are clipped to that range, as ``predict(extrapolation="clip")`` does.

    The measurements are associations in the data. They do not estimate what
    would happen under an intervention on the variable.
    """
    if int(folds) < 2:
        raise ValueError("folds must be at least 2; pass folds=5 or another count")
    spec = _resolve(model_or_recommendation)
    return _check(spec, data, folds=int(folds), seed=int(seed))


def _resolve(obj) -> _Spec:
    """The declared specification of a recommendation, a fitted model or a
    `_Spec`."""
    from mboostrs._autospec import SpecRecommendation
    from mboostrs._control import as_control
    from mboostrs.gamboost import GAMBoostModel

    if isinstance(obj, _Spec):
        return obj
    if isinstance(obj, SpecRecommendation):
        if isinstance(obj.formula, dict):
            raise NotImplementedError(
                "check_declarations tests one response distribution at a time; this "
                "recommendation holds one formula per distribution parameter. Test "
                "each parameter's formula through gamboost(...) and check_declarations "
                "on that fit.")
        rule = obj.evidence.get("interaction_rule") or {}
        return _Spec(
            formula=obj.formula, family=obj.family, response=obj.response,
            control=obj.control or boost_control(obj.mstop),
            env=dict(obj.env) if obj.env else None,
            treatment=tuple(obj.treatment), groups=tuple(obj.groups),
            measurements=tuple(obj.measurements),
            forbid=tuple(rule.get("forbid", ())),
            mandatory=tuple(rule.get("mandatory", ())))
    if isinstance(obj, GAMBoostModel):
        formula = obj._formula
        return _Spec(formula=formula, family=obj._family,
                     response=partition_tilde(formula)[0].strip(),
                     control=as_control(obj._control))
    raise TypeError(
        "check_declarations takes a SpecRecommendation from autospec or a fitted "
        f"gamboost model; got {type(obj).__name__}")


def _items(spec: _Spec) -> list[_Item]:
    """Every declaration the specification states, in formula order."""
    rhs = partition_tilde(spec.formula)[2]
    items = []
    for match in _BMONO.finditer(rhs):
        args = [a.strip() for a in match.group(0)[len("bmono("):-1].split(",")
                if a.strip()]
        constraint = next((a.split("=", 1)[1].strip() for a in args[1:]
                           if a.replace(" ", "").startswith("constraint=")),
                          "increasing")
        if constraint not in SIGN_CONSTRAINTS and constraint not in SHAPE_CONSTRAINTS:
            raise ValueError(
                f"bmono constraint {constraint!r} is not one of "
                f"{sorted({*SIGN_CONSTRAINTS, *SHAPE_CONSTRAINTS})}")
        items.append(_Item(kind=constraint, variable=args[0], term=match.group(0)))
    for column in spec.treatment:
        items.append(_Item("treatment", column, reason=_STRUCTURAL_REASONS["treatment"]))
    for column in spec.groups:
        items.append(_Item("groups", column, reason=_STRUCTURAL_REASONS["groups"]))
    for column in spec.measurements:
        items.append(_Item("measurements", column,
                           reason=_STRUCTURAL_REASONS["measurements"]))
    for first, second in _BSPATIAL.findall(rhs):
        items.append(_Item("coordinates", f"{unquote_name(first)}, {unquote_name(second)}",
                           reason=_STRUCTURAL_REASONS["coordinates"]))
    for name in _BMRF.findall(rhs):
        items.append(_Item("adjacency", unquote_name(name), reason=_STRUCTURAL_REASONS["adjacency"]))
    for name in _BUSER.findall(rhs):
        items.append(_Item("basis", unquote_name(name), reason=_STRUCTURAL_REASONS["basis"]))
    for pair in spec.forbid:
        items.append(_Item("forbid", pair, reason=_STRUCTURAL_REASONS["forbid"]))
    for name in spec.mandatory:
        items.append(_Item("mandatory", name, reason=_STRUCTURAL_REASONS["mandatory"]))
    for name in spec.locks:
        items.append(_Item("lock", name, reason=_STRUCTURAL_REASONS["lock"]))
    return items


def declaration_items(obj) -> list[dict]:
    """The declarations of a recommendation, model or search, with no fit.

    Each entry gives the declaration, its kind, the column it names, and the
    reason no test applies when no sign or shape is tested.
    """
    return [{"declaration": item.declaration, "kind": item.kind,
             "variable": item.variable, "tested": item.term is not None,
             "reason": item.reason}
            for item in _items(_resolve(obj))]


def _formula_columns(spec: _Spec) -> tuple[str, ...]:
    """Every column the fitted formula reads, the response first."""
    names = [spec.response]
    for term in _split_plus(partition_tilde(spec.formula)[2]):
        for name in term_variables(term.strip()):
            if name not in names:
                names.append(name)
    for name in spec.groups:
        if name not in names:
            names.append(name)
    return tuple(names)


def _complete_rows(data, names):
    """The rows where every named column is present, and how many were dropped."""
    mask = None
    for name in names:
        present = ~np.isnan(np.asarray(read_column(data, name).values, dtype=float))
        mask = present if mask is None else mask & present
    if mask.all():
        return data, 0
    return take_rows(data, mask), int(mask.size - mask.sum())


def _fit(formula: str, data, spec: _Spec):
    from mboostrs.gamboost import gamboost

    kwargs = {"env": spec.env} if spec.env else {}
    return gamboost(formula, data=data, family=spec.family, control=spec.control,
                    **kwargs)


def _unconstrained(formula: str, term: str) -> str:
    """The formula with one ``bmono`` term replaced by a ``bbs`` term of the same
    variable and degrees of freedom."""
    args = [a.strip() for a in term[len("bmono("):-1].split(",") if a.strip()]
    shared = [a for a in args[1:] if a.split("=", 1)[0].strip() in _SHARED_KWARGS]
    return formula.replace(term, "bbs(" + ", ".join([args[0], *shared]) + ")", 1)


def _check(spec: _Spec, data, *, folds: int, seed: int) -> list[DeclarationCheck]:
    from mboostrs.cvrisk import cv_weights
    from mboostrs.folds import grouped_folds

    items = _items(spec)
    names = _formula_columns(spec)
    data, dropped = _complete_rows(data, names)
    response = np.asarray(read_column(data, spec.response).values, dtype=float)
    n = int(response.shape[0])
    if folds > n:
        raise ValueError(f"folds={folds} exceeds the {n} complete rows")
    if spec.groups:
        from mboostrs.autospec import _group_labels

        held = np.asarray(grouped_folds(_group_labels(data, spec.groups), B=folds,
                                        seed=seed).held_out, dtype=bool)
    else:
        held = np.asarray(cv_weights(n, type="kfold", B=folds, seed=seed)) == 0
    records = []
    for item in items:
        if item.term is None:
            records.append(DeclarationCheck(
                declaration=item.declaration, kind=item.kind, variable=item.variable,
                test=None, verdict="inconclusive",
                agreement=f"{item.declaration} is not tested against the fit: "
                          f"{item.reason}.",
                evidence={"rows": n, "rows_dropped": dropped}, reason=item.reason))
            continue
        records.append(_test_constraint(spec, item, data, response, held, n, dropped, seed))
    return records


def _bootstrap_contrasts(spec, data, unconstrained, variable, low, high, seed):
    """The 10th-to-90th-percentile contrast of the unconstrained model, refitted
    on each of `SIGN_BOOTSTRAP_REPLICATES` resamples of the rows (drawn with
    replacement). Each refit's contrast is evaluated on the original rows."""
    n = int(np.asarray(read_column(data, spec.response).values).shape[0])
    rng = np.random.default_rng(10_000 + seed)
    out = np.empty(SIGN_BOOTSTRAP_REPLICATES)
    for b in range(SIGN_BOOTSTRAP_REPLICATES):
        fit = _fit(unconstrained, take_rows(data, rng.integers(0, n, n)), spec)
        pair = np.asarray(fit.predict_over_grid(
            variable, [low, high], newdata=data, type="link", output="numpy",
            extrapolation="clip", unseen_levels="average"), dtype=float)
        out[b] = float(np.mean(pair[1] - pair[0]))
    return np.ascontiguousarray(out)


def _test_constraint(spec, item, data, response, held, n, dropped, seed) -> DeclarationCheck:
    from mboostrs._rust import declaration_quantiles, declaration_shape_check, declaration_sign_check
    from mboostrs.explore import _score

    variable = item.variable
    column = read_column(data, variable)
    if column.kind == "categorical":
        raise ValueError(
            f"{item.declaration}: {variable!r} is categorical; a sign or shape "
            "declaration needs a numeric column, so recode it or drop the declaration")
    values = np.ascontiguousarray(column.values, dtype=float)
    quantiles = declaration_quantiles(values, [*CONTRAST_PROBS, *GRID_PROBS])
    low, high, grid_values = quantiles[0], quantiles[1], quantiles[2:]
    unconstrained = _unconstrained(spec.formula, item.term)
    shape = item.kind in SHAPE_CONSTRAINTS
    if shape:
        sign, second = SHAPE_CONSTRAINTS[item.kind]
    else:
        sign, second = SIGN_CONSTRAINTS[item.kind], False

    full = _fit(unconstrained, data, spec)
    stacked = np.asarray(full.predict_over_grid(
        variable, [low, high, *grid_values], newdata=data, type="link", output="numpy",
        extrapolation="clip", unseen_levels="average"), dtype=float)
    hi, lo = np.ascontiguousarray(stacked[1]), np.ascontiguousarray(stacked[0])
    grid = np.ascontiguousarray(stacked[2:])
    fits = 1

    # Held-out rows can lie outside the training range of another covariate. Those
    # values are clipped to the training range, as predict(extrapolation="clip")
    # does; the warning is suppressed here because it would repeat for every fold.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        fold_hi, fold_lo, risk_constrained, risk_unconstrained, fold_fits = _fold_fits(
            held, data, response, spec, unconstrained, spec.formula, variable, low, high,
            shape, _score)
    fits += fold_fits

    boot = None
    if not shape:
        boot = _bootstrap_contrasts(spec, data, unconstrained, variable, low, high, seed)
        fits += SIGN_BOOTSTRAP_REPLICATES
    base = {"rows": n, "rows_dropped": dropped, "folds": held.shape[1],
            "fits": fits, "variable": variable, "constraint": item.kind}
    if not shape:
        (contrast, fold_contrasts, lower, upper, folds_agreeing,
         verdict) = declaration_sign_check(sign, hi, lo, fold_hi, fold_lo, boot)
        evidence = {**base, "test": "sign", "contrast_percentiles": list(CONTRAST_PROBS),
                    "contrast": float(contrast),
                    "fold_contrasts": [float(c) for c in fold_contrasts],
                    "interval_lower": float(lower), "interval_upper": float(upper),
                    "interval_level": SIGN_INTERVAL_LEVEL,
                    "bootstrap_replicates": SIGN_BOOTSTRAP_REPLICATES,
                    "folds_agreeing": int(folds_agreeing),
                    "rule": ("the full-data contrast is compared with the percentile interval of "
                             f"{SIGN_BOOTSTRAP_REPLICATES} row-bootstrap refits of the unconstrained "
                             "model at 95%; agrees if the interval excludes zero in the declared "
                             "direction and at least 80% of folds (four of five) share the sign of "
                             "the full-data contrast; contradicts if the interval excludes zero in "
                             "the opposite direction with the same fold agreement; otherwise "
                             "inconclusive")}
        return DeclarationCheck(
            declaration=item.declaration, kind=item.kind, variable=variable, test="sign",
            verdict=verdict,
            agreement=_sign_sentence(item, verdict, contrast, lower, upper,
                                     folds_agreeing, held.shape[1]),
            evidence=evidence)

    (curve, steps, share, risk_c, risk_u, change, verdict) = declaration_shape_check(
        sign, second, grid, risk_constrained, risk_unconstrained)
    evidence = {**base, "test": "shape", "grid_points": GRID_POINTS,
                "grid_range": [float(quantiles[2]), float(quantiles[-1])],
                "curve": [float(v) for v in curve], "steps": int(steps),
                "share_in_declared_direction": None if share is None else float(share),
                "risk_constrained": float(risk_c), "risk_unconstrained": float(risk_u),
                "risk_change_without_constraint": float(change),
                "rule": ("agrees if at least 90% of the steps go the declared way and "
                         "removing the constraint does not lower held-out risk by more than "
                         "1%; contradicts if at most 10% go the declared way and removing the "
                         "constraint lowers held-out risk by more than 1%; otherwise "
                         "inconclusive")}
    return DeclarationCheck(
        declaration=item.declaration, kind=item.kind, variable=variable, test="shape",
        verdict=verdict, agreement=_shape_sentence(item, verdict, share, change),
        evidence=evidence)


def _fold_fits(held, data, response, spec, unconstrained, constrained, variable, low,
               high, shape, score):
    """Per fold: the unconstrained contrast on held-out rows, and for a shape
    declaration the held-out risk of the constrained and unconstrained models.
    Returns the arrays the Rust checks take and the number of fits made."""
    fold_hi, fold_lo, risk_constrained, risk_unconstrained = [], [], [], []
    fits = 0
    for k in range(held.shape[1]):
        test_rows = held[:, k]
        train = take_rows(data, ~test_rows)
        test = take_rows(data, test_rows)
        fit_u = _fit(unconstrained, train, spec)
        fits += 1
        pair = np.asarray(fit_u.predict_over_grid(
            variable, [low, high], newdata=test, type="link", output="numpy",
            extrapolation="clip", unseen_levels="average"), dtype=float)
        fold_lo.append(np.ascontiguousarray(pair[0]))
        fold_hi.append(np.ascontiguousarray(pair[1]))
        if not shape:
            continue
        fit_c = _fit(constrained, train, spec)
        fits += 1
        y_test = response[test_rows]
        ones = np.ones(y_test.shape[0])
        eta_u = fit_u.predict(newdata=test, type="link", unseen_levels="average",
                              extrapolation="clip")
        eta_c = fit_c.predict(newdata=test, type="link", unseen_levels="average",
                              extrapolation="clip")
        risk_unconstrained.append(score(spec.family, y_test, eta_u, ones))
        risk_constrained.append(score(spec.family, y_test, eta_c, ones))
    return fold_hi, fold_lo, risk_constrained, risk_unconstrained, fits


def _sign_sentence(item, verdict, contrast, lower, upper, folds_agreeing, folds) -> str:
    decl = item.declaration
    variable = item.variable
    size = (f"{contrast:+.3g} link units, with a 95% bootstrap interval of "
            f"({lower:+.3g}, {upper:+.3g}), and {folds_agreeing} of {folds} folds sharing its sign")
    if verdict == "agrees":
        return (f"{decl} agrees with the fitted association: the contrast of {variable} between "
                f"its 10th and 90th percentiles is {size}.")
    if verdict == "contradicts":
        return (f"{decl} contradicts the fitted association: the contrast of {variable} between "
                f"its 10th and 90th percentiles is {size}, against the declared sign.")
    return (f"{decl} is inconclusive: the contrast of {variable} between its 10th and 90th "
            f"percentiles is {size}; a verdict needs the interval to exclude zero and at least "
            "80% of folds to share the sign of the contrast.")


def _shape_sentence(item, verdict, share, change) -> str:
    decl = item.declaration
    if share is None:
        return (f"{decl} is inconclusive: the unconstrained curve of {item.variable} is "
                "flat between its 1st and 99th percentiles.")
    share_text = f"{share:.0%} of the steps of the unconstrained curve go the declared way"
    change_text = f"removing the constraint changes held-out risk by {change:+.1%}"
    if verdict == "agrees":
        return f"{decl} agrees with the fitted association: {share_text}, and {change_text}."
    if verdict == "contradicts":
        return (f"{decl} contradicts the fitted association: only {share_text}, and "
                f"{change_text}.")
    return f"{decl} is inconclusive: {share_text}, and {change_text}; neither rule is met."
