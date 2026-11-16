"""Variable selection after, or instead of, cross-validated stopping.

Cross-validated early stopping chooses the step count that predicts best, and
that step count is usually later than the one that selects well: a noise
term whose effect is tiny barely changes the held-out risk, so it stays in.
On `benches/bench_insights.py`'s additive task a CV-tuned `gamboost` kept
4.7 of 5 pure-noise features on average. Two remedies from the boosting
literature are provided here.

* :func:`deselect` (Strömer, Staerk, Klein, Weinhold, Titze and Mayr, 2022,
  *Statistical Methods in Medical Research*): fit with the tuned `mstop`,
  drop every base learner whose share of the total risk reduction (what
  `varimp` reports) is below a threshold, then refit the remaining terms and
  tune `mstop` again.
* :func:`probe` (Thomas, Hepp, Mayr and Bischl, 2017, *Computational and
  Mathematical Methods in Medicine*): add a permuted copy ("shadow") of every
  single-column term, fit once, and keep the terms selected before the first
  iteration that selects a shadow. A shadow has the same distribution as its
  column and no association with the response, so the first shadow selection
  marks the point where the fit starts to take up noise. No cross-validation
  decides the selection.

Neither gives an error-rate guarantee; stability selection (`mb.stabsel`)
bounds the expected number of false selections under its assumptions.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from mboostrs import _rust
from mboostrs._names import IDENT_SRC as IDENT
from mboostrs._names import NAME_SRC as N
from mboostrs._names import partition_tilde, quote_name, unquote_name
from mboostrs._names import split_terms as _split_plus
from mboostrs._output import export_result, json_safe
from mboostrs._workflow import call_fit, fit_checkpoint

__all__ = ["Deselection", "EffectContrast", "GamRefit", "Probing", "deselect", "gam_refit",
           "probe"]


@dataclass(frozen=True)
class Deselection:
    """The terms :func:`deselect` kept and dropped, and the refitted model."""

    #: Formula terms kept, in formula order. For a distributional model each
    #: entry is a ``(parameter, term)`` pair.
    kept: tuple
    #: Formula terms dropped, with their share of the risk reduction. For a
    #: distributional model each entry is ``(parameter, term, share)``.
    dropped: tuple
    #: Share of the total risk reduction per learner in the original fit.
    shares: dict
    #: The share below which a learner was dropped.
    threshold: float
    #: The reduced formula the refit used; a ``{parameter: formula}`` mapping
    #: for a distributional model.
    formula: str | dict
    #: The refitted model, at the step count its own cross-validation chose.
    model: object = field(repr=False, compare=False)

    @property
    def variables(self):
        """The data columns the kept terms read, in formula order; for a
        distributional model, ``{parameter: columns}``."""
        if self.kept and isinstance(self.kept[0], tuple):
            by_parameter = {}
            for parameter, term in self.kept:
                by_parameter.setdefault(parameter, []).append(term)
            return {parameter: _variables(terms) for parameter, terms in by_parameter.items()}
        return _variables(self.kept)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record (the model itself is not exported)."""
        mstop = self.model.mstop
        mstop = ({str(k): int(v) for k, v in mstop.items()} if isinstance(mstop, dict)
                 else int(mstop))
        variables = self.variables
        variables = ({k: list(v) for k, v in variables.items()} if isinstance(variables, dict)
                     else list(variables))
        return export_result(self, include_arrays=include_arrays,
                             extras={"mstop": mstop, "variables": variables})


@dataclass(frozen=True)
class Probing:
    """Result of :func:`probe`: selected terms, the first shadow, and its entry iteration."""

    #: Formula terms selected before the first shadow, in formula order.
    selected: tuple
    #: The iteration (1-based) that first selected a shadow; ``None`` if no
    #: shadow was selected within the budget.
    stop: int | None
    #: The shadow that entered first.
    first_shadow: str | None
    #: The formula of the selected terms.
    formula: str
    #: The selected terms refitted without shadows, `mstop` by cross-validation.
    model: object = field(repr=False, compare=False)

    @property
    def variables(self):
        """The data columns the selected terms read, in formula order."""
        return _variables(self.selected)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record (the model itself is not exported)."""
        return export_result(self, include_arrays=include_arrays,
                             extras={"mstop": int(self.model.mstop),
                                     "variables": list(self.variables)})


# ------------------------------------------------------------------ helpers
def _term_columns(term):
    """The data columns a formula term reads."""
    from mboostrs.formula import term_variables

    return term_variables(term)


def _variables(terms):
    from mboostrs.formula import term_variables

    out = []
    for term in terms:
        out.extend(v for v in term_variables(term) if v not in out)
    return tuple(out)


def _entry(model):
    """The fitting function that produced `model`."""
    import mboostrs as mb

    kind = type(model).__name__
    if kind == "GAMBoostModel":
        return mb.gamboost
    if kind == "GLMBoostModel":
        return mb.glmboost
    raise TypeError(
        f"selection works on gamboost and glmboost models; got {kind}. For a "
        f"distributional model, deselect() takes a method='noncyclic' fit and "
        f"stabsel_gamboostLSS selects per parameter")


def _bundle(model):
    bundle = getattr(model, "_refit_bundle", None)
    if not bundle or bundle.get("raw_formula") is None or bundle.get("raw_data") is None:
        raise ValueError(
            "selection refits the model from its original formula and data, "
            "which this model does not carry (a scoring-only or loaded artifact)")
    return bundle


def _terms(formula, data):
    """(response side, [(sign, term)]) of the formula with `.` expanded.

    An intercept written as ``+ 1`` (or ``+ 0``) is dropped first: an
    intercept-only parameter keeps its intercept in every refit, so it is not
    a term to expand or to refit.
    """
    from mboostrs.formula import _signed_terms, data_columns, expand_formula

    lhs, _, rhs = partition_tilde(formula)
    columns = set(data_columns(data))
    kept = [(sign, term) for sign, term in _signed_terms(rhs)
            if not (term in ("0", "1") and term not in columns and sign == "+")]
    rebuilt = f"{lhs} ~ " + " ".join(f"{sign} {term}" for sign, term in kept) if kept \
        else f"{lhs} ~ 1"
    expanded = expand_formula(rebuilt, data)
    lhs, _, rhs = partition_tilde(expanded)
    return lhs.strip(), _signed_terms(rhs)


def _switched_constant_names(model) -> set:
    """Names of the constant missing-value learners of a model's switched terms.

    A column whose smooth term was split into observed and filled parts (the
    recipe holds an ``observed_indicator`` for it) has a constant
    ``<x>_missing`` learner, which the refit leaves out.
    """
    recipe = getattr(model, "_transform_recipe", None)
    steps = recipe.steps if recipe else ()
    observed = {s.source for s in steps if s.kind == "observed_indicator"}
    return {s.name for s in steps if s.kind == "missing_indicator" and s.source in observed}


def _term_learners(model, terms):
    """For each `(sign, term)`, the learner names it produced.

    A `gamboost` formula produces one learner per `+` term, in order. A
    `glmboost` formula names each learner by its column, and a factor term
    produces one learner per contrast column (`catb`, `catc`), which the
    transform recipe records against its source column.
    """
    names = list(model._learner_names)
    additive = [(sign, t) for sign, t in terms if sign == "+" and t.strip() not in ("1", "0")]
    if type(model).__name__ == "GAMBoostModel":
        # The refit leaves out the constant <x>_missing learner of a switched
        # term (see _with_imputation_terms), so it has no term of its own.
        constant = _switched_constant_names(model)
        names = [name for name in names if name not in constant]
        if len(additive) != len(names):
            raise NotImplementedError(
                f"the formula's {len(additive)} terms do not map one to one onto "
                f"the model's {len(names)} learners (a factor by= or another "
                f"term that expands into several learners); write the reduced "
                f"formula by hand from `varimp()`")
        return {t: [name] for (_, t), name in zip(additive, names, strict=True)}
    recipe = getattr(model, "_transform_recipe", None)
    source_of = {s.name: s.source for s in (recipe.steps if recipe else ())}
    mapping = {t: [] for _, t in additive}
    for name in names:
        if name == "(Intercept)":
            continue
        term = name if name in mapping else source_of.get(name)
        if term in mapping:
            mapping[term].append(name)
    return mapping


def _refit(model, formula, *, folds, seed, mstop):
    """Fit `formula` with the original model's arguments and tune `mstop`."""
    import mboostrs as mb

    bundle = _bundle(model)
    arguments = {k: v for k, v in (bundle.get("raw_arguments") or {}).items()
                 if v is not None}
    control = arguments.pop("control", None) or mb.boost_control()
    control = control._replace(mstop=int(mstop or max(control.mstop, model.mstop) * 2))
    arguments.pop("learners", None)
    fit = _entry(model)(formula, data=bundle["raw_data"], control=control, **arguments)
    n = len(np.asarray(fit.predict()).ravel())
    fold_matrix = mb.cv_weights(n, "kfold", folds, seed=seed) if isinstance(folds, int) else folds
    return fit.with_mstop(max(fit.cvrisk(folds=fold_matrix).optimal_mstop, 1))


# ---------------------------------------------------------------- deselect
def deselect(model, threshold=0.01, *, folds=5, seed=0, mstop=None):
    """Drop base learners with a small share of the risk reduction, then refit.

    Parameters
    ----------
    model : GAMBoostModel, GLMBoostModel or LSSModel
        A fit at its tuned step count (for example ``tune_mstop(...).model``).
        A distributional model must be fitted with ``method="noncyclic"``,
        the setting Strömer et al. (2022) propose deselection for: each step
        updates one learner of one parameter, so each step's risk reduction
        belongs to one learner. Shares are computed over all parameters
        together, a parameter left with no term is refitted as intercept
        only (``y ~ 1``), and the refit's total step count is tuned by
        ``LSSModel.cvrisk``.
    threshold : float
        A learner whose share of the total in-bag risk reduction is below this
        is dropped. Strömer et al. (2022) use 0.01.
    folds : int or fold matrix
        Cross-validation for the refit's `mstop`: an int is random k-fold on
        `seed`; pass ``grouped_folds(...)`` for grouped data.
    mstop : int, optional
        The refit's largest step count; defaults to twice the larger of the
        model's control `mstop` and its current `mstop`.

    Returns
    -------
    Deselection
        ``kept``, ``dropped``, the per-learner ``shares``, the reduced
        ``formula`` and the refitted ``model``.

    A learner's share is its accumulated in-bag risk reduction, so a term that
    moved the fit a little in many steps can fall below the threshold even
    when the true effect is non-zero but small; the threshold trades those
    losses against the noise terms removed.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in (0, 1)")
    if type(model).__name__ == "LSSModel":
        return _deselect_lss(model, threshold, folds=folds, seed=seed, mstop=mstop)
    _entry(model)
    bundle = _bundle(model)
    lhs, terms = _terms(bundle["raw_formula"], bundle["raw_data"])
    shares = {name: float(v) for name, v in model.varimp(percent=True)["reduction"].items()}
    mapping = _term_learners(model, terms)

    # Each additive term's share is the sum over the learners it produced;
    # the core sums and applies the threshold.
    additive = [term for sign, term in terms if sign == "+" and term.strip() not in ("1", "0")]
    values, lengths = [], []
    for term in additive:
        names = mapping.get(term, [])
        lengths.append(len(names))
        values.extend(shares.get(name, 0.0) for name in names)
    term_shares = _rust.deselect_group_sums(values, lengths)
    keep_flags = iter(_rust.deselect_keep(term_shares, threshold))
    share_of = iter(term_shares)
    kept, dropped = [], []
    for sign, term in terms:
        if sign != "+" or term.strip() in ("1", "0"):
            kept.append((sign, term))
            continue
        share, keep = next(share_of), next(keep_flags)
        if keep:
            kept.append((sign, term))
        else:
            dropped.append((term, share))
    kept_terms = [t for s, t in kept if s == "+" and t.strip() not in ("1", "0")]
    if not kept_terms:
        raise ValueError(
            f"no learner reaches {threshold:.0%} of the risk reduction; nothing "
            f"to refit")
    formula = f"{lhs} ~ " + " ".join(
        f"{'+' if i and s == '+' else ('-' if s == '-' else '')} {t}".strip()
        for i, (s, t) in enumerate(kept)).replace("~ +", "~")
    refit = _refit(model, formula, folds=folds, seed=seed, mstop=mstop)
    return Deselection(
        kept=tuple(kept_terms),
        dropped=tuple((t, round(share, 6)) for t, share in dropped),
        shares=shares, threshold=threshold, formula=formula, model=refit)


def _deselect_lss(model, threshold, *, folds, seed, mstop):
    """`deselect` for a non-cyclic `glmboostLSS`/`gamboostLSS` fit."""
    import mboostrs as mb
    from mboostrs.formula import _signed_terms

    if model._method != "noncyclic":
        raise NotImplementedError(
            "deselection needs the risk reduction of each learner, and a cyclic "
            "LSS fit records risk once per cycle, after every parameter has "
            "been updated, so the reduction cannot be divided between "
            "learners. Refit with method='noncyclic', the setting deselection "
            "was proposed for (Strömer et al., 2022).")
    arguments = model._refit_arguments()
    base = arguments.pop("formula")
    names = [str(name) for name in model.param_names]
    formulas = base if isinstance(base, dict) else {name: base for name in names}

    # Shares by (parameter, learner index): two terms over one column carry
    # the same learner name, so the name alone cannot tell them apart.
    parameter_index = {name: i for i, name in enumerate(names)}
    rows = list(model.path_frame(output="records"))
    shares_by_pair = {
        (names[p], index): share
        for p, index, share in _rust.deselect_lss_shares(
            [parameter_index[row["parameter"]] for row in rows],
            [int(row["learner_index"]) for row in rows],
            [float(row.get("risk_change") or 0.0) for row in rows])}
    learner_names = model.base_learner_names()
    glm = model._model_kind != "gamboostLSS"

    kept, dropped, shares, reduced = [], [], {}, {}
    for name in names:
        lhs, _, rhs = partition_tilde(formulas[name])
        terms = [(sign, term.strip()) for sign, term in _signed_terms(rhs)]
        learners = learner_names[name]
        if glm:
            # One learner per column, named by it, after the intercept.
            index_of = {learner: i for i, learner in enumerate(learners)}
        else:
            # One learner per `+` term in order; an explicit `1` is a learner.
            added = [term for sign, term in terms if sign == "+" and term != "0"]
            if len(added) != len(learners):
                raise NotImplementedError(
                    f"the {name} formula's {len(added)} terms do not map one to "
                    f"one onto the fit's {len(learners)} learners (a term that "
                    f"expands into several learners); write the reduced "
                    f"formulas by hand from varimp()")
            index_of = {term: i for i, term in enumerate(added)}
        keep, fixed = [], []
        for sign, term in terms:
            if sign != "+" or term in ("1", "0"):
                fixed.append(f"- {term}" if sign == "-" else f"+ {term}")
                continue
            if term not in index_of:
                raise NotImplementedError(
                    f"the {name} formula's term {term!r} does not name one "
                    f"learner of the fit; write the reduced formulas by hand "
                    f"from varimp()")
            share = shares_by_pair.get((name, index_of[term]), 0.0)
            shares[f"{term}.{name}"] = share
            if _rust.deselect_keep([share], threshold)[0]:
                keep.append(term)
                kept.append((name, term))
            else:
                dropped.append((name, term, round(share, 6)))
        # A parameter with no term left keeps its intercept alone.
        rhs_new = " + ".join(keep) if keep else "1"
        extras = [f for f in fixed if not (f == "+ 1" and not keep)]
        reduced[name] = " ".join([f"{lhs.strip()} ~ {rhs_new}", *extras])
    if not kept:
        raise ValueError(
            f"no learner reaches {threshold:.0%} of the risk reduction; nothing "
            f"to refit")

    if getattr(model, "_method", None) == "noncyclic":
        # The total number of updates; `mstop` collapses equal per-parameter
        # counts to one integer, which is half the total for two parameters.
        current = model._budget()
    else:
        budget = model.mstop
        current = sum(budget.values()) if isinstance(budget, dict) else int(budget)
    control = arguments.pop("control")
    control = control._replace(mstop=int(mstop or max(control.mstop, current) * 2))
    entry = mb.gamboostLSS if model._model_kind == "gamboostLSS" else mb.glmboostLSS
    fit = entry(reduced, **arguments, control=control, method="noncyclic",
                weights=model._weights)
    # The reduced formulas are the rewritten ones, read against the processed
    # frame: the recipes that rebuild derived columns (a `%X%` tensor, a
    # transformed column or response) on new rows belong to the original fit
    # and are carried over, as the LSS refit does.
    fit._transform_recipe = getattr(model, "_transform_recipe", None)
    fit._response_transform = getattr(model, "_response_transform", None)
    fit._interactions = getattr(model, "_interactions", None)
    fit._rewritten_formulas = dict(reduced)
    n = len(model.model_weights())
    fold_matrix = (mb.cv_weights(n, "kfold", folds, seed=seed,
                                 weights=model.model_weights())
                   if isinstance(folds, int) else folds)
    cv = fit.cvrisk(folds=fold_matrix)
    refit = fit.with_mstop(max(cv.optimal_mstop[0], 1))
    # The fit reads `%X%` pairs as derived designs, `buser(__ix0, ...)`; the
    # record names them by the term the user wrote.
    shown = _written_term_labeler(model)
    return Deselection(
        kept=tuple((name, shown(term)) for name, term in kept),
        dropped=tuple((name, shown(term), share) for name, term, share in dropped),
        shares={f"{shown(key.rsplit('.', 1)[0])}.{key.rsplit('.', 1)[1]}": value
                for key, value in shares.items()},
        threshold=threshold,
        formula={name: shown(text) for name, text in reduced.items()},
        model=refit)


def _written_term_labeler(model):
    """A function that rewrites each derived `%X%` term in a formula or term
    (`buser(__ix0, X=__ix0_X, K=__ix0_K, df=6.25)`) as the term the user wrote."""
    labels = {getattr(ix, "derived", None): ix.name
              for ix in getattr(model, "_interactions", None) or ()
              if str(getattr(ix, "derived", "")).startswith("__ix")}
    pattern = re.compile(r"buser\(\s*(__ix\d+)\s*,[^()]*(?:\([^()]*\)[^()]*)*\)")

    def shown(text):
        return pattern.sub(lambda m: labels.get(m.group(1), m.group(0)), text)

    return shown


# ------------------------------------------------------------------- probe
_SINGLE_COLUMN_TERM = re.compile(rf"^\s*(?:(b\w+)\s*\(\s*)?({IDENT})\s*(,.*)?\)?\s*$", re.DOTALL)


def probe(formula, data=None, *, family=None, control=None, seed=0, folds=5, **fit_kwargs):
    """Select terms by the first entry of a permuted shadow, then refit them.

    ``formula`` is a formula string with ``data``, or a fitted ``gamboost`` /
    ``glmboost`` model, whose formula, data, family and fitting arguments are
    reused (``probe(model)``, as ``deselect(model)`` and ``stabsel(model)``).

    Every term that reads one column (``x``, ``bbs(x, df=4)``, ``bols(x)``)
    gets a shadow: the same term over a permuted copy of the column. The
    model with all terms and shadows is boosted up to ``control.mstop``
    (default 1000) iterations, and the terms selected before the first shadow
    are kept. Terms over several columns (``%X%``, ``bspatial``) get no shadow
    and are kept only if selected before the first shadow.

    ``seed`` seeds the generator that draws one row permutation per shadow
    column. A fixed seed reproduces the same shadows and the same refit folds.

    Returns
    -------
    Probing
        ``selected``, the iteration ``stop`` at which the first shadow entered,
        its name, the formula of the selected terms and ``model``, that formula
        refitted without shadows with `mstop` by cross-validation.

    Probing needs no tuning, but it has no error-rate guarantee: a shadow can
    enter early by chance, which stops selection early, and a true term with
    a weak effect can enter after a shadow.
    """
    import mboostrs as mb
    from mboostrs._frame import column_names
    from mboostrs.formula import _signed_terms, expand_formula

    if not isinstance(formula, str):
        model = formula
        _entry(model)
        if data is not None:
            raise TypeError("probe(model) reuses the model's data; do not pass data=")
        bundle = _bundle(model)
        arguments = {k: v for k, v in (bundle.get("raw_arguments") or {}).items()
                     if v is not None and k not in ("learners", "control")}
        family = family or arguments.pop("family", None)
        arguments.pop("family", None)
        fit_kwargs = {**arguments, **fit_kwargs}
        formula, data = bundle["raw_formula"], bundle["raw_data"]
    elif data is None:
        raise TypeError("probe(formula, data): data is required with a formula")
    family = family or mb.Gaussian()
    control = control or mb.boost_control(mstop=1000, nu=0.1)
    expanded = expand_formula(formula, data)
    lhs, _, rhs = partition_tilde(expanded)
    terms = [t for s, t in _signed_terms(rhs) if s == "+" and t.strip() not in ("1", "0")]

    columns = {name: np.asarray(_column(data, name)) for name in column_names(data)}
    shadow_sources, shadow_terms = {}, []
    for term in terms:
        match = _SINGLE_COLUMN_TERM.match(term)
        if not match or "%X%" in term:
            continue
        column = unquote_name(match.group(2))
        if column not in columns:
            continue
        shadow = f"shadow__{column}"
        shadow_sources.setdefault(shadow, column)
        shadow_terms.append(term.replace(match.group(2), quote_name(shadow), 1))
    # One seeded permutation of the row indices per shadow column, drawn by
    # the core; a shadow is its source column in that order.
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    n_rows = len(next(iter(columns.values()))) if columns else 0
    orders = _rust.probe_permutations(n_rows, len(shadow_sources), int(seed))
    shadows = {shadow: columns[column][np.asarray(order, dtype=np.intp)]
               for (shadow, column), order in zip(shadow_sources.items(), orders, strict=True)}
    if not shadow_terms:
        raise ValueError("probe needs at least one term over a single column")

    augmented = {**columns, **shadows}
    entry = mb.glmboost if all(re.fullmatch(rf"{N}", t.strip()) for t in terms) else mb.gamboost
    wide = entry(f"{lhs.strip()} ~ " + " + ".join(terms + shadow_terms), data=augmented,
                 family=family, control=control, **fit_kwargs)

    names = list(wide._learner_names)
    order = list(wide._selected)
    # The core finds the first iteration that selects a shadow and the real
    # learners chosen before it.
    stop, first_index, before = _rust.probe_stop(
        [int(i) for i in order], ["shadow__" in name for name in names])
    first_shadow = None if first_index is None else names[first_index]
    selected_learners = []
    for index in before:
        if names[index] not in selected_learners:
            selected_learners.append(names[index])

    mapping = {}
    for term, name in zip(terms, [n for n in names if not n.startswith("shadow__")][:len(terms)], strict=True):
        mapping[term] = name
    if entry is mb.glmboost:
        mapping = {t: unquote_name(t) for t in terms}
    chosen = [t for t in terms if mapping.get(t) in selected_learners]
    if not chosen:
        raise ValueError(
            f"a shadow ({first_shadow}) entered before any real term; nothing "
            f"is selected")
    reduced = f"{lhs.strip()} ~ " + " + ".join(chosen)
    refit = entry(reduced, data=data, family=family,
                  control=control._replace(mstop=max(control.mstop, 1)), **fit_kwargs)
    n = len(np.asarray(refit.predict()).ravel())
    fold_matrix = mb.cv_weights(n, "kfold", folds, seed=seed) if isinstance(folds, int) else folds
    refit = refit.with_mstop(max(refit.cvrisk(folds=fold_matrix).optimal_mstop, 1))
    return Probing(selected=tuple(chosen), stop=stop, first_shadow=first_shadow,
                   formula=reduced, model=refit)


def _column(data, name):
    values = data[name]
    return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)


def _breslow(time, status, eta, w):
    """Breslow cumulative baseline hazard at each row's own time."""
    order = np.argsort(time, kind="stable")
    t, d, risk = time[order], (w * status)[order], (w * np.exp(eta))[order]
    at_risk = np.cumsum(risk[::-1])[::-1]
    # Ties share the risk set of their first member.
    first = np.searchsorted(t, t, side="left")
    increments = np.where(d > 0, d / at_risk[first], 0.0)
    cumulative = np.cumsum(increments)
    last = np.searchsorted(t, t, side="right") - 1
    hazard = np.empty_like(t)
    hazard[order] = cumulative[last]
    return hazard


# --------------------------------------------------------------- gam_refit
def _spline_variables(term):
    """The columns of a spline term whose training range bounds its basis.

    A ``bbs`` or ``bspatial`` term, and the B-spline margins of a ``%X%`` term,
    read a basis defined between its boundary knots. Linear ``bols`` and
    ``brandom`` terms are not range-checked, as in the boosted predict.
    """
    from mboostrs.formula import term_variables

    if "%X%" in term:
        return tuple(name for side in term.split("%X%") if "bbs(" in side
                     for name in term_variables(side))
    if re.match(r"\s*(bbs|bspatial)\s*\(", term):
        return term_variables(term)
    return ()


def _training_ranges(terms, data):
    """``{column: (low, high, term)}`` for the spline covariates of `terms`,
    each column's minimum and maximum over the training rows."""
    out = {}
    for term in terms:
        for name in _spline_variables(term):
            if name in out:
                continue
            try:
                values = np.asarray(_column(data, name), dtype=float)
            except (TypeError, ValueError, KeyError):
                continue  # a label column: no numeric range to check
            finite = values[np.isfinite(values)]
            if finite.size:
                out[name] = (float(finite.min()), float(finite.max()), term)
    return out


def _range_rows(ranges, newdata):
    """A :class:`~mboostrs._extrapolation.RangeRow` per range covariate present
    in `newdata`, with the raw values it measured, as ``[(row, values)]``."""
    from mboostrs._extrapolation import row_for
    from mboostrs.transforms import _has_column

    out = []
    for name, (low, high, term) in ranges.items():
        if not _has_column(newdata, name):
            continue
        values = np.asarray(_column(newdata, name), dtype=float)
        out.append((row_for(term, name, values, low, high), values))
    return out


def _outside_mask(pairs, n=None):
    """Per row, whether any covariate in `pairs` lies outside its training range.

    Returns ``None`` when there is no covariate to check and `n` is not given.
    """
    if not pairs:
        return None if n is None else np.zeros(n, dtype=bool)
    mask = np.zeros(len(pairs[0][1]), dtype=bool)
    for row, values in pairs:
        with np.errstate(invalid="ignore"):
            mask |= (values < row.train_low) | (values > row.train_high)
    return mask


def _apply_range_policy(pairs, newdata, extrapolation):
    """Act on ``extrapolation=`` for the range-checked covariates of `newdata`.

    ``"warn"`` predicts and issues one :class:`ExtrapolationWarning`;
    ``"error"`` refuses; ``"allow"`` predicts silently; ``"clip"`` returns a
    copy of `newdata` whose out-of-range covariates are held at the nearest
    training value, which is the boosted predict's policy.
    """
    from mboostrs._extrapolation import POLICIES, apply_policy

    if extrapolation not in POLICIES:
        raise ValueError(f"extrapolation must be one of {POLICIES}, got {extrapolation!r}")
    if not pairs:
        return newdata
    rows = [row for row, _ in pairs]
    values = {index: raw for index, (_, raw) in enumerate(pairs)}
    updated = apply_policy(rows, values, extrapolation, stacklevel=4)
    if extrapolation == "clip":
        for index, row in enumerate(rows):
            if row.n_outside:
                newdata = _with_values(newdata, row.variable, updated[index])
    return newdata


def _with_values(data, name, values):
    """A copy of `data` whose column `name` holds `values`, row by row."""
    if isinstance(data, dict):
        out = dict(data)
        out[name] = np.asarray(values)
        return out
    if hasattr(data, "with_columns"):                       # polars
        import polars as pl

        return data.with_columns(pl.Series(name, np.asarray(values)))
    return data.assign(**{name: np.asarray(values)})        # pandas


def _frame_rows(data) -> int:
    """The number of rows of a dict, polars or pandas frame."""
    if isinstance(data, dict):
        return len(np.asarray(next(iter(data.values()))))
    if hasattr(data, "height"):
        return data.height
    return len(data)


def _contrast_values(variable, from_value, to_value):
    """``(names, from, to)`` tuples for one covariate or several moved together."""
    if isinstance(variable, str):
        names, lo, hi = (variable,), (from_value,), (to_value,)
    else:
        names, lo, hi = tuple(variable), tuple(from_value), tuple(to_value)
        if not (len(names) == len(lo) == len(hi)):
            raise ValueError(
                "variable, from and to name several covariates, so from and to must be "
                f"sequences of the same length; got {len(names)}, {len(lo)} and {len(hi)}")
    if not names:
        raise ValueError("contrast needs at least one covariate to move")
    return names, lo, hi


def _moved(frame, names, values, n):
    """A copy of `frame` whose covariates `names` hold `values` in every row."""
    for name, value in zip(names, values, strict=True):
        frame = _with_values(frame, name, np.full(n, value))
    return frame


def _effect_contrast(estimate, se, level, scale):
    from statistics import NormalDist

    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    z = NormalDist().inv_cdf(0.5 + float(level) / 2.0)
    return EffectContrast(estimate=estimate, se=se, lower=estimate - z * se,
                          upper=estimate + z * se, level=float(level), scale=scale)


@dataclass(frozen=True)
class EffectContrast:
    """The average change in a linear predictor when covariates move, with its interval.

    ``estimate`` is the mean, over the rows of the data given to ``contrast``,
    of the change in the predictor when the covariates move from their
    ``from`` values to their ``to`` values with every other column at the
    row's own value. ``se`` is its standard error from the coefficient
    covariance, and ``lower`` and ``upper`` are ``estimate -/+ z * se`` at
    ``level``. ``scale`` is the link scale the change is on (``"link"``).
    """

    estimate: float
    se: float
    lower: float
    upper: float
    level: float
    scale: str = "link"


@dataclass(frozen=True)
class GamRefit:
    """A jointly penalized refit of a set of terms, with Bayesian intervals.

    Produced by :func:`gam_refit`. ``edf`` and ``smoothing`` give each term's
    effective degrees of freedom and REML smoothing parameter, ``sigma`` the
    residual standard deviation. :meth:`effects` returns each term's
    effect at new rows with a pointwise interval.
    """

    terms: tuple
    edf: dict
    smoothing: dict
    sigma: float
    #: The formula terms were chosen on the same data the refit uses; the
    #: intervals are conditional on that choice.
    conditional_on_selection: bool
    #: Scale of `effects` and `predict`: "identity", "logit" (log-odds) or "log".
    link: str = "identity"
    #: -2 x restricted log-likelihood at the optimum, up to a constant.
    reml_score: float = float("nan")
    #: Newton and PIRLS iteration counts, the final gradient norm, and
    #: whether both converged.
    convergence: dict = field(default_factory=dict)
    #: NBinomial: the dispersion the refit used, estimated by the same REML
    #: criterion starting from the boosted fit's value; None otherwise.
    theta: float | None = None
    #: Terms the refit could not represent (``btree``, ``bmono``, ...) held at
    #: their boosted contribution, as a known offset, by ``unsupported="hold"``.
    held: tuple = ()
    _blocks: tuple = field(repr=False, compare=False, default=())
    _core: object = field(repr=False, compare=False, default=None)
    #: The boosted model and learner names that produce the held terms'
    #: contribution at new rows, and the factor from its link to the refit's.
    _held_model: object = field(repr=False, compare=False, default=None)
    _held_learners: tuple = field(repr=False, compare=False, default=())
    _held_scale: float = field(repr=False, compare=False, default=1.0)
    #: The ``na_action="impute"`` steps of the boosted model (median fills,
    #: missing-value indicators, a ``"(missing)"`` level), replayed on new
    #: rows before any refit design reads them.
    _impute: object = field(repr=False, compare=False, default=None)
    #: The boosted model's training frame, held by reference for the
    #: provenance entry of :meth:`to_dict`; not copied.
    _training_data: object = field(repr=False, compare=False, default=None)
    #: ``{column: (low, high, term)}``: the training range of each spline
    #: covariate, which :meth:`predict` checks new rows against.
    _ranges: dict = field(repr=False, compare=False, default_factory=dict)
    #: The prior weights of the fitting rows (ones when none were given); the
    #: default weights of :meth:`bootstrap`.
    _weights: object = field(repr=False, compare=False, default=None)

    def _imputed(self, newdata):
        if self._impute is None or newdata is None:
            return newdata
        from mboostrs.transforms import apply_recipe

        return apply_recipe(self._impute, newdata)

    def __repr__(self):
        edf = ", ".join(f"{k}: {v:.2f}" for k, v in self.edf.items())
        held = f", held={self.held}" if self.held else ""
        return (f"GamRefit(link={self.link!r}, edf={{{edf}}}, sigma={self.sigma:.4g}"
                f"{held})")

    @property
    def tests(self):
        """``{term: {"edf", "ref_df", "statistic", "p_value"}}``: each term's
        approximate test of ``f = 0`` (Wood 2013), the numbers mgcv's
        ``summary.gam`` prints. The statistic is a chi-square for a known
        scale (Binomial, Poisson, negative binomial, Cox) and an F statistic
        (``statistic / ref_df``) when the scale is estimated. See
        :meth:`summary` for how to read the p-values."""
        out = {}
        f_test = self._core.residual_df is not None
        for block in self._core.blocks():
            if "p_value" not in block:
                continue
            statistic = float(block["statistic"])
            ref_df = float(block["ref_df"])
            out[block["name"]] = {
                "edf": float(block["edf"]),
                "ref_df": ref_df,
                "statistic": statistic / ref_df if f_test and ref_df > 0 else statistic,
                "p_value": float(block["p_value"]),
            }
        return out

    def summary(self):
        """A table of the terms, as mgcv's ``summary.gam`` gives it.

        Each row gives the term's effective degrees of freedom (``edf``), the
        rank its test uses (``ref_df``), the test statistic (F when the scale
        is estimated, chi-square otherwise) and an approximate p-value for
        the hypothesis that the term's effect is zero everywhere (Wood 2013).
        The p-values treat the smoothing parameters as known, as mgcv's do.
        When the terms were selected on the same data
        (``conditional_on_selection``), they are smaller than the p-values
        of a model fixed in advance, and a small p-value says the kept term's
        shape is well determined rather than that selection was justified.
        """
        from mboostrs.summary import Summary, SummaryTable

        f_test = self._core.residual_df is not None
        # Below machine precision a tail probability is reported as R does.
        rows = [[term, t["edf"], t["ref_df"], t["statistic"],
                 "< 2e-16" if t["p_value"] < 2.2e-16 else t["p_value"]]
                for term, t in self.tests.items()]
        table = SummaryTable(None, ["term", "edf", "ref_df",
                                    "F" if f_test else "chi_sq", "p_value"], rows)
        header = [("Link", self.link), ("Observations", int(self._core.n)),
                  ("Scale" if f_test else "Scale (fixed)", float(self._core.scale)),
                  ("REML score", self.reml_score), ("Degrees of freedom", self.df),
                  ("Log-likelihood", self.loglik)]
        if self.theta is not None:
            header.append(("Theta", float(self.theta)))
        notes = []
        if self.held:
            notes.append(f"Held at their boosted contribution: {', '.join(self.held)}.")
        if self.conditional_on_selection:
            notes.append("The terms were selected on the same data, so the p-values are "
                         "conditional on that selection and smaller than those of a "
                         "model fixed in advance.")
        return Summary("Penalized refit (REML)", header, [table], notes)

    def concurvity(self, output="auto"):
        """mgcv's concurvity of each term against the rest of the model.

        Concurvity is the additive-model form of collinearity: how far a
        term's fitted function could be reproduced by the other terms. When it
        is high the partial effects are not separately identified -- the
        same fit can be redistributed between the terms -- so their shapes,
        intervals and the selection that chose them are unstable. Each row
        gives mgcv's three measures (``concurvity(full = TRUE)``; Wood,
        *Generalized Additive Models*, 2nd ed., section 5.6.2), each between
        0 (the term is orthogonal to the others) and 1 (it lies in their
        span): ``worst`` over every coefficient vector of the term,
        ``observed`` at the fitted coefficients, and ``estimate``, an average
        over coefficient directions. They are computed in the core from the
        refit's model matrix, the intercept included among the other terms.
        """
        from mboostrs._output import output_frame

        rows = []
        for block in self._core.blocks():
            if "concurvity_worst" not in block:
                continue
            rows.append({"term": block["name"],
                         "worst": float(block["concurvity_worst"]),
                         "observed": float(block["concurvity_observed"]),
                         "estimate": float(block["concurvity_estimate"])})
        return output_frame(rows, output)

    def _held_contribution(self, newdata, unseen_levels):
        if not self._held_learners:
            return 0.0
        return self._held_scale * sum(
            np.asarray(self._held_model.predict(newdata=newdata, which=name,
                                                unseen_levels=unseen_levels),
                       dtype=float).ravel()
            for name in self._held_learners)

    @property
    def _coef(self):
        return np.asarray(self._core.coef)

    @property
    def _cov(self):
        return np.asarray(self._core.cov)

    @property
    def _intercept(self):
        return float(self._core.intercept)

    @property
    def loglik(self):
        """Log-likelihood of the refit, as mgcv's ``logLik.gam`` computes it:
        Gaussian with the maximum-likelihood variance RSS/n, Cox the partial
        likelihood. Its degrees of freedom are :attr:`df`."""
        return float(self._core.loglik)

    @property
    def df(self):
        """Effective degrees of freedom of the fit, plus one for an estimated
        scale, negative binomial theta or gamma shape."""
        return float(self._core.df)

    def aic(self):
        """Conditional AIC, ``-2 loglik + 2 df``. mgcv's ``AIC(gam)`` counts
        the degrees of freedom corrected for smoothing-parameter uncertainty
        (its ``edf2``), which are larger, so mgcv's AIC is larger by about
        twice that difference (0.9 on the body-fat model). It compares
        refits of the same response, not boosted models."""
        return -2.0 * self.loglik + 2.0 * self.df

    def vcov(self, type="bayesian"):
        """Covariance matrix of the coefficients, as a square NumPy array.

        Row and column ``i`` refer to coefficient ``i`` of the vector that
        :meth:`coef` lists, in the same order. Index 0 is the intercept. After
        it come the terms in the order of :attr:`terms`, each with its own
        block of consecutive indices, one per column of the term's constrained
        basis: a factor ``bols`` has one fewer than its number of levels, a
        numeric ``bols`` one, and a ``bbs`` term one fewer than its basis
        columns. The last index is therefore the last column of the last
        term, which is not a treatment effect unless the treatment is that
        term, and even then it is a constrained-basis weight, not a contrast.
        Use :meth:`coef` or :meth:`term_coefficients` to find an index by
        name, and :meth:`effects` for an effect.
        ``type="bayesian"`` is conditional on the smoothing parameters;
        ``type="corrected"`` adds the first-order correction for their
        uncertainty (Wood, Pya and Saefken 2016), mgcv's
        ``vcov(unconditional=TRUE)`` without its second-order term."""
        if type == "bayesian":
            return np.asarray(self._core.cov)
        if type == "corrected":
            return np.asarray(self._core.cov_corrected)
        raise ValueError('type must be "bayesian" or "corrected"')

    def _coefficient_labels(self):
        """``(term, coefficient)`` for each entry of the coefficient vector.

        The intercept comes first, then each term's block in :attr:`terms`
        order. A coefficient is named by its term and the 1-based position of
        its column in that term's constrained basis, ``bbs(x, df=4)[3]``.
        Those columns are a sum-to-zero reparametrization of the term's design,
        not its raw indicator or spline columns, so a coefficient is not a
        level difference or a slope; the effect at each row is in
        :meth:`effects`.
        """
        labels = [("(Intercept)", "(Intercept)")]
        blocks = list(self._core.blocks())
        if len(blocks) != len(self._blocks):
            raise RuntimeError("the refit's coefficient blocks do not match its terms")
        for block, (term, _design, _by) in zip(blocks, self._blocks, strict=True):
            width = int(block["end"]) - int(block["start"])
            labels.extend((term, f"{term}[{j}]") for j in range(1, width + 1))
        if len(labels) != self._coef.size:
            raise RuntimeError("the refit's coefficient labels do not match its coefficients")
        return labels

    def coef(self, level=0.95, *, type="bayesian"):
        """The refit's coefficients as a table, with pointwise intervals.

        Returns a Polars DataFrame with one row per coefficient, in the order
        of :meth:`vcov`, and the columns:

        ``term``
            The formula term the coefficient belongs to; ``"(Intercept)"`` for
            the intercept.
        ``coefficient``
            The term's formula text with the 1-based position of the column in
            its constrained basis, such as ``bbs(x, df=4)[3]`` or
            ``bols(g)[2]``. The basis is the refit's sum-to-zero
            reparametrization of the term's design, so one coefficient is not
            a level difference or a slope: a factor's coefficients do not
            compare the levels one by one. Read the effect at each row with
            :meth:`effects`, and its interval there.
        ``estimate``
            The coefficient.
        ``se``
            Its standard error, the square root of the diagonal of
            ``vcov(type)``.
        ``lower``, ``upper``
            The interval ``estimate -/+ z * se`` at ``level``, with ``z`` the
            standard normal quantile (a normal approximation).

        ``type`` selects the covariance as in :meth:`vcov`. The intervals are
        for the coefficients, not for an effect: an effect at a row combines
        several coefficients, so :meth:`effects` gives that interval.
        """
        from statistics import NormalDist

        import polars as pl

        if not 0.0 < float(level) < 1.0:
            raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
        estimate = np.asarray(self._coef, dtype=float)
        se = np.sqrt(np.clip(np.diag(self.vcov(type)), 0.0, None))
        z = NormalDist().inv_cdf(0.5 + float(level) / 2.0)
        labels = self._coefficient_labels()
        return pl.DataFrame({
            "term": [term for term, _ in labels],
            "coefficient": [name for _, name in labels],
            "estimate": estimate,
            "se": se,
            "lower": estimate - z * se,
            "upper": estimate + z * se,
        })

    def term_coefficients(self, term, level=0.95, *, type="bayesian"):
        """The rows of :meth:`coef` that belong to one formula term.

        ``term`` is a term as written in :attr:`terms`, such as
        ``"bols(ethnicity)"`` or ``"bbs(x, df=4)"``. Raises a ValueError that
        lists the terms when the name is not one of them. Use ``"(Intercept)"``
        for the intercept.
        """
        table = self.coef(level=level, type=type)
        known = list(dict.fromkeys(table["term"].to_list()))
        wanted = term.strip()
        if wanted not in known:
            raise ValueError(
                f"{term!r} is not a term of this refit; its terms are {known}")
        return table.filter(table["term"] == wanted)

    def effects(self, newdata, level=0.9, *, unseen_levels="error", corrected=False,
                centre=False):
        """``{term: (estimate, lower, upper)}`` at the rows of `newdata`.

        ``unseen_levels="zero_effect"`` gives a ``brandom`` term zero effect
        (with zero standard error) at a group the refit never saw, the
        average over groups, which is the prediction for a new group that
        lme4 (``allow.new.levels``) and mgcv (``exclude=``) make. Its interval
        is that of the average; a new group's own effect varies around it by
        the random-effect standard deviation, which the interval leaves out.
        ``unseen_levels="average"`` gives every term that reads the column
        (a ``brandom``, a ``bols`` over a factor, a factor ``by=``, a
        ``%X%`` factor margin) the mean of its level effects at such a row,
        as `GAMBoostModel.predict` does.

        Ordinary smooths are centred over the weighted fitting rows. A
        nonconstant numeric ``by=`` term retains its constant-in-x direction,
        which becomes the effect of the by variable. Its product with the
        by variable need not have mean zero. The interval is the Bayesian
        pointwise interval ``estimate +/- z * se`` from the coefficient
        covariance (Wahba 1983; Nychka 1988), which averages close to nominal
        coverage over the effect's range when the smoothing parameter is
        estimated by REML. ``corrected=True`` widens it by the uncertainty of
        the estimated smoothing parameters (see :meth:`vcov`).

        ``centre=True`` centres each effect over the rows of `newdata`
        instead: the estimate is ``f(x_i)`` minus its mean over those rows,
        and the interval is that contrast's, from the covariance of the
        centred design (the design row minus the column means over
        `newdata`), not the uncentred interval shifted. Use it to compare an
        effect with a curve centred over the same rows, such as a known truth
        or another package's estimate.
        """
        from statistics import NormalDist

        if unseen_levels not in ("error", "zero_effect", "average"):
            raise ValueError("unseen_levels must be 'error', 'zero_effect' or 'average'")
        z = NormalDist().inv_cdf(0.5 + level / 2)
        unseen = {"zero_effect": "zero", "average": "average"}.get(unseen_levels, "error")
        newdata = self._imputed(newdata)
        out = _effects_of(self._core.effect, self._blocks, newdata, z, unseen, corrected,
                          centre=centre)
        return out

    def contrast(self, variable, from_value, to_value, *, data, level=0.9):
        """The average change in the link-scale prediction when covariates move.

        Every row of `data` is taken with `variable` set to `from_value` in one
        copy and to `to_value` in another, all other columns at the row's own
        values. The result is the mean over the rows of the difference of the
        two link-scale predictions, with its standard error and the
        ``level`` interval, returned as an :class:`EffectContrast`. The
        standard error uses the full covariance, so the terms a covariate
        enters through (a smooth and a tensor, say) are combined with their
        covariance rather than summed as separate intervals.

        ``variable`` is one column name, with a scalar ``from_value`` and
        ``to_value`` (a factor's level labels work too), or a sequence of names
        moved together, such as ``("latitude", "longitude")``, with sequences of
        values. A covariate that no term reads has a change of exactly zero with
        zero standard error. The spline covariates are checked against their
        training ranges as :meth:`predict` checks them, with a warning.
        """
        names, lo_values, hi_values = _contrast_values(variable, from_value, to_value)
        base = self._imputed(data)
        n = _frame_rows(base)
        lo = _moved(base, names, lo_values, n)
        hi = _moved(base, names, hi_values, n)
        for frame in (lo, hi):
            pairs = _range_rows(self._ranges, frame)
            _apply_range_policy(pairs, frame, "warn")
        raws_hi, bys_hi, zeros = self._design_rows(hi, "error")
        raws_lo, bys_lo, _ = self._design_rows(lo, "error")
        estimate, se = self._core.contrast(raws_hi, bys_hi, zeros, raws_lo, bys_lo, zeros,
                                           corrected=False)
        return _effect_contrast(float(estimate), float(se), level, "link")

    def predict(self, newdata, type="link", *, unseen_levels="error", offset=None,
                interval=False, level=0.9, corrected=False, output="auto",
                extrapolation="warn"):
        """The refit's prediction at the rows of `newdata`, on the link scale
        (``type="link"``) or the response scale (``type="response"``).
        ``unseen_levels`` is as in :meth:`effects`. ``offset`` is the new
        rows' offset (log exposure) when the model was fitted with one; it is
        added to the linear predictor.

        ``extrapolation`` is as for :meth:`GAMBoostModel.predict`, applied to
        the covariates of the spline terms (``bbs``, ``bspatial`` and the
        B-spline margins of ``%X%``) against their training ranges. A basis
        continues linearly beyond its boundary knots, so the default
        ``"warn"`` predicts and issues one warning naming the covariate and
        the rows outside; ``"error"`` refuses; ``"allow"`` predicts silently;
        ``"clip"`` holds each covariate at its nearest training value.

        ``interval=True`` returns a frame with ``estimate``, ``se``, ``lower``
        and ``upper`` for each row instead of the bare array, and
        ``outside_range``, true where a range-checked covariate of that row
        lies outside its training range (computed on the values as given,
        before any clipping). The estimate is the sum of the intercept and
        every term's effect, and ``se`` is
        ``sqrt(x' V x)`` from the full covariance ``V`` of the intercept and
        all terms, so it includes the covariance between terms and the
        intercept's uncertainty; a term's own interval (:meth:`effects`) omits
        the intercept. ``se`` is always on the link scale. The bounds are
        ``estimate +/- z * se`` on the link scale, and mapped through the
        inverse link when ``type="response"``. ``level`` is the coverage;
        ``corrected=True`` uses the covariance corrected for the uncertainty of
        the smoothing parameters; ``output`` is as for other frames. The
        interval is conditional on the held terms (``held``) and on the
        chosen terms, as :meth:`effects` is.

        Returns
        -------
        numpy.ndarray or polars.DataFrame
            Without ``interval=True``, a NumPy array with one value per row of
            `newdata`, on the scale named by `type`. With ``interval=True``, a
            Polars frame with one row per row of `newdata` and the columns
            ``estimate``, ``se``, ``lower``, ``upper`` and ``outside_range``
            (``output="auto"``, the default, gives Polars when it is installed;
            ``output="records"`` gives a list of dicts with the same keys).
        """
        if type not in ("link", "response"):
            raise ValueError('type must be "link" or "response"')
        pairs = _range_rows(self._ranges, newdata)
        outside = _outside_mask(pairs)
        newdata = _apply_range_policy(pairs, newdata, extrapolation)
        if interval:
            return self._predict_interval(newdata, type, unseen_levels, offset, level,
                                          corrected, output, outside)
        total = self._intercept if offset is None else self._intercept + np.asarray(offset, float)
        total = total + self._held_contribution(newdata, unseen_levels)
        for estimate, _, _ in self.effects(newdata, unseen_levels=unseen_levels).values():
            total = total + estimate
        eta = np.asarray(total, dtype=float)
        return eta if type == "link" else self._to_response(eta)

    def _to_response(self, eta):
        if self.link == "logit":
            return 1 / (1 + np.exp(-eta))
        if self.link == "log":
            return np.exp(eta)
        return eta

    def _design_rows(self, newdata, unseen_levels):
        """Each term's raw design, its `by=` column and its zero-row flag at
        the rows of `newdata`, in the order of ``self._blocks``."""
        unseen = {"zero_effect": "zero", "average": "average"}.get(unseen_levels, "error")
        raws, bys, zeros = [], [], []
        for term, design, by in self._blocks:
            raws.append(np.ascontiguousarray(design.transform(newdata, unseen=unseen),
                                             dtype=float))
            bys.append(None if by is None
                       else np.ascontiguousarray(_column(newdata, by), dtype=float))
            zeros.append(term.lstrip().startswith("brandom"))
        return raws, bys, zeros

    def _mean_interval(self, newdata, level=0.95, corrected=False):
        """``(estimate, lower, upper)`` of the mean over the rows of `newdata`
        of the link-scale prediction, with its interval.

        The prediction is linear in each term's raw design row, so the mean
        over rows of the predictions is the prediction at the mean design row
        of each term (a `by=` column multiplied in first). One call to the
        core on those averaged rows gives the mean and its standard error
        from the full covariance, including the covariance between terms.
        """
        from statistics import NormalDist

        raws, bys, _ = self._design_rows(self._imputed(newdata), "error")
        averaged = []
        for raw, by in zip(raws, bys, strict=True):
            if by is not None:
                raw = raw * by[:, None]
            averaged.append(np.ascontiguousarray(raw.mean(axis=0, keepdims=True)))
        eta, se = self._core.prediction(averaged, [None] * len(averaged),
                                        [False] * len(averaged), corrected=bool(corrected))
        z = NormalDist().inv_cdf(0.5 + level / 2)
        estimate = float(np.asarray(eta, dtype=float).ravel()[0])
        spread = z * float(np.asarray(se, dtype=float).ravel()[0])
        return estimate, estimate - spread, estimate + spread

    def _predict_interval(self, newdata, type, unseen_levels, offset, level, corrected,
                          output, outside=None):
        from statistics import NormalDist

        from mboostrs._output import output_frame

        if unseen_levels not in ("error", "zero_effect", "average"):
            raise ValueError("unseen_levels must be 'error', 'zero_effect' or 'average'")
        if not 0 < level < 1:
            raise ValueError(f"level must be between 0 and 1; got {level!r}")
        held = self._held_contribution(newdata, unseen_levels)
        raws, bys, zeros = self._design_rows(self._imputed(newdata), unseen_levels)
        eta, se = self._core.prediction(raws, bys, zeros, corrected=bool(corrected))
        eta = np.asarray(eta, dtype=float) + held
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=float)
        se = np.asarray(se, dtype=float)
        z = NormalDist().inv_cdf(0.5 + level / 2)
        lower, upper = eta - z * se, eta + z * se
        if type == "response":
            eta, lower, upper = (self._to_response(a) for a in (eta, lower, upper))
        flags = np.zeros(len(eta), dtype=bool) if outside is None else outside
        rows = [{"estimate": float(e), "se": float(s), "lower": float(lo), "upper": float(hi),
                 "outside_range": bool(flag)}
                for e, s, lo, hi, flag in zip(eta, se, lower, upper, flags, strict=True)]
        return output_frame(rows, output,
                            columns=("estimate", "se", "lower", "upper", "outside_range"))

    def bootstrap(self, counts, response, weights=None):
        """Refit this Gaussian refit on each bootstrap draw of its rows.

        ``counts`` is a ``B x n`` array of row multiplicities, one row per draw
        (a multinomial resample gives each row a count and the counts of a draw
        sum to ``n``). ``response`` is the response of the ``n`` fitting rows, in
        the order the refit used, and ``weights`` their prior weights; the
        weights default to those the refit was fitted with. Draw ``b`` is fitted
        with prior weights ``weights * counts[b]``. The designs, penalties and
        spline knots are the full sample's, and each draw re-estimates its
        smoothing parameters by REML. Returns a :class:`GamRefitBootstrap`.
        """
        from mboostrs import _rust

        if self.link != "identity":
            raise NotImplementedError(
                "the row bootstrap refits the Gaussian (identity-link) second stage; "
                f"this refit has link {self.link!r}")
        if self.held:
            raise NotImplementedError(
                f"the row bootstrap cannot refit the held terms {list(self.held)}; "
                "refit without them")
        y = np.ascontiguousarray(response, dtype=float).ravel()
        counts = np.ascontiguousarray(counts, dtype=float)
        if counts.ndim != 2 or counts.shape[1] != y.size:
            raise ValueError(
                f"counts must have shape (draws, {y.size}), one column per fitting row; "
                f"got {counts.shape}")
        base = self._weights if weights is None else weights
        rows_w = (np.ones(y.size) if base is None
                  else np.ascontiguousarray(base, dtype=float).ravel())
        if rows_w.size != y.size:
            raise ValueError(
                f"weights have {rows_w.size} entries for {y.size} fitting rows")
        _, names, designs, penalties, bys = _refit_parts(list(self.terms), self._training_data)
        core = call_fit(_rust.gam_refit_bootstrap_core,
            names, designs, penalties, bys, y, counts, weights=rows_w,
            variables=[list(_term_columns(name)) for name in names])
        return GamRefitBootstrap(core=core, refit=self)

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the terms, their edf, smoothing parameters,
        sigma and term tests (``tests``); ``include_arrays=True`` adds the
        refit's coefficient vector."""
        extras = {"tests": self.tests}
        if include_arrays and self._coef is not None:
            extras["coefficients"] = json_safe(self._coef)
        return export_result(self, include_arrays=include_arrays, extras=extras,
                             with_provenance=True, training_data=self._training_data)


#: Family -> (link, whether the scale is known). The refit uses the canonical
#: parametrization: log-odds for Binomial (twice mboostrs' half log-odds),
#: log mean for Poisson and GammaReg.
_REFIT_FAMILIES = {"Gaussian": ("identity", False), "Binomial": ("logit", True),
                   "Poisson": ("log", True), "GammaReg": ("log", False),
                   "NBinomial": ("log", True), "CoxPH": ("log", True)}

#: One column, bare or under a log/sqrt transform, with the term's options.
_REFIT_TERM = re.compile(
    rf"^\s*(bbs|bols|brandom)\s*\(\s*(?:(?:log|sqrt)\(\s*{IDENT}\s*\)"
    rf"|{IDENT})\s*(,.*)?\)\s*$",
    re.DOTALL)
_REFIT_SPATIAL = re.compile(
    rf"^\s*bspatial\s*\(\s*({IDENT})\s*,\s*({IDENT})\s*(,.*)?\)\s*$", re.DOTALL)


class _TensorDesign:
    """A row-tensor term for `gam_refit`: `bspatial(a, b)` or `a %X% b`.

    Its penalty is the Kronecker sum `K_a (x) I + I (x) K_b`, and the refit
    gives each margin its own smoothing parameter, as mgcv's `te()` does.
    """

    def __init__(self, left, right, data):
        from mboostrs.interactions import _marginal_design, _tensor

        self._left, self._right = left, right
        Xl, Kl = _marginal_design(left, data)
        Xr, Kr = _marginal_design(right, data)
        self.design = _tensor(Xl, Xr)
        from mboostrs import _rust

        pairs = (_rust.kron(np.ascontiguousarray(Kl, dtype=float), np.eye(Kr.shape[0])),
                 _rust.kron(np.eye(Kl.shape[0]), np.ascontiguousarray(Kr, dtype=float)))
        self.penalties = [np.asarray(p) for p in pairs if np.any(np.asarray(p))]

    def transform(self, data, *, unseen="error"):
        from mboostrs.interactions import _marginal_design, _tensor

        return _tensor(_marginal_design(self._left, data, unseen=unseen)[0],
                       _marginal_design(self._right, data, unseen=unseen)[0])


class _LevelDesign:
    """`bols(f)` over a categorical column, for `gam_refit`: mboost's
    treatment-coded factor, an intercept column and one indicator per level
    after the first, with no penalty (the refit's centring removes the
    intercept, as it does for a numeric `bols(x)`). The labels are read as they are in the data (strings included),
    and a label the fit never saw raises unless `unseen="zero"`, which gives
    the reference level's row, or `unseen="average"`, which gives the mean of
    the level rows."""

    def __init__(self, column, data, *, ridge=False):
        from mboostrs.interactions import _raw_column

        self._column = column
        self._levels = tuple(sorted({v for v in _raw_column(data, column).tolist()}, key=str))
        self.design = self.transform(data)
        width = self.design.shape[1]
        # Default: the level effects are unpenalized, so each effect is the
        # level's own least-squares contrast. With ridge=True the indicator
        # columns (every level but the reference) get an identity penalty, so
        # REML shrinks each level's effect toward the reference level.
        penalty = np.zeros((width, width))
        if ridge and width > 1:
            penalty[1:, 1:] = np.eye(width - 1)
        self.penalties = [penalty]

    def transform(self, data, *, unseen="error"):
        from mboostrs.interactions import _raw_column

        raw = _raw_column(data, self._column).tolist()
        index = {level: j for j, level in enumerate(self._levels)}
        new = sorted({v for v in raw if v not in index}, key=str)
        if new and unseen not in ("zero", "average"):
            from mboostrs.interactions import _level_kind_hint

            hint = _level_kind_hint(self._levels, new)
            raise ValueError(
                f"column {self._column!r} has level(s) {new} not seen at fit time "
                f"(levels: {list(self._levels)})" + (f"; {hint}" if hint else ""))
        design = np.zeros((len(raw), len(self._levels)))
        design[:, 0] = 1.0
        for row, value in enumerate(raw):
            j = index.get(value)
            if j is None and unseen == "average":
                # The mean of the level rows: every level after the first
                # carries 1/L, so the row predicts the mean level effect.
                design[row, 1:] = 1.0 / len(self._levels)
            elif j:
                design[row, j] = 1.0
        return design


def _refit_design(term, data, factor_ridge=False, encoded=frozenset()):
    """The design and penalties of one refit term.

    `encoded` names the columns the model's training frame holds as integer
    codes of their labels (a factor the fit encoded). Those are factors even
    though the frame reads them as numbers, so `bols` on them is the level
    design and not a linear term in the codes.
    """
    from mboostrs.interactions import _parse_marginal, learner_design
    from mboostrs.transforms import _is_categorical

    bare = re.fullmatch(rf"\s*bols\(\s*({IDENT})\s*\)\s*", term)
    if bare and (unquote_name(bare.group(1)) in encoded
                 or _is_categorical(data, unquote_name(bare.group(1)))):
        return _LevelDesign(unquote_name(bare.group(1)), data, ridge=factor_ridge)

    if "%X%" in term:
        left, right = term.split("%X%")
        return _TensorDesign(_parse_marginal(left, data), _parse_marginal(right, data), data)
    spatial = _REFIT_SPATIAL.match(term)
    if spatial:
        # bspatial's basis is the tensor of two bbs margins with its own knots,
        # degree and differences.
        args = {}
        for part in (spatial.group(3) or "").lstrip(",").split(","):
            if "=" in part:
                key, value = (piece.strip() for piece in part.split("=", 1))
                args[key] = value
        shared = ", ".join(f"{k}={args[k]}" for k in ("knots", "degree", "differences")
                           if k in args)
        margin = lambda column: _parse_marginal(                            # noqa: E731
            f"bbs({quote_name(column)}{', ' + shared if shared else ''})", data)  # already quoted text
        return _TensorDesign(margin(spatial.group(1)), margin(spatial.group(2)), data)
    design = learner_design(term, data)
    design.penalties = [np.asarray(design.penalty, dtype=float)]
    return design


def _check_refit_columns(terms, data) -> None:
    """Refuse a numeric column with a missing value before the Rust refit does.

    The Rust design only says that a term's variable is non-finite. This names
    the column, the term and the first row, and the remedy.
    """
    from mboostrs._errors import MboostrsValueError
    from mboostrs.formula import term_variables
    from mboostrs.transforms import _column

    for term in terms:
        for name in dict.fromkeys(term_variables(term)):
            try:
                values = _column(data, name)
            except (TypeError, ValueError):
                continue  # a label column: its missing level is handled by the design
            bad = ~np.isfinite(values)
            if bad.any():
                first = int(np.flatnonzero(bad)[0])
                raise MboostrsValueError(
                    f"the refit term {term!r} reads column {name!r}, which has "
                    f"{int(bad.sum())} missing value(s), first at row {first}. Fit with "
                    "na_action='impute' (median fill and an indicator term), or "
                    "na_action='impute_model' when the other covariates predict the column "
                    "(model-based fill and the same indicator term), or "
                    "na_action='omit' (drop those rows), or fill the column before fitting")


def _refit_parts(terms, data, factor_ridge=False, encoded=frozenset()):
    """Each term's raw design, penalties and `by=` column, for the Rust refit.

    A numeric `by=` multiplies the term's function by that column. With a
    nonconstant by variable, Rust retains the full basis: its constant-in-x
    direction represents the by variable's main effect. Actual overlap with
    the intercept or explicit main effects is removed by side constraints.
    `factor_ridge` is passed to each categorical `bols` term (see `_LevelDesign`).
    """
    from mboostrs.interactions import _by_factor_marginal, _parse_marginal
    from mboostrs.transforms import _is_categorical

    _check_refit_columns(terms, data)
    blocks, names, designs, penalties, bys = [], [], [], [], []
    for term in terms:
        by_match = re.search(rf",\s*by\s*=\s*({IDENT})", term)
        by = unquote_name(by_match.group(1)) if by_match else None
        base = re.sub(rf",\s*by\s*=\s*{IDENT}", "", term)
        if by is not None and _is_categorical(data, by):
            # A factor by= is the term's basis times the treatment-coded level
            # indicators, the tensor gamboost fits: one copy of the term per
            # level after the first, each the difference from that level.
            design = _TensorDesign(_parse_marginal(base, data),
                                   _by_factor_marginal(by, data), data)
            by = None
        else:
            design = _refit_design(base, data, factor_ridge, encoded)
        blocks.append((term, design, by))
        names.append(term)
        designs.append(np.ascontiguousarray(design.design, dtype=float))
        penalties.append([np.ascontiguousarray(k, dtype=float) for k in design.penalties])
        bys.append(None if by is None
                   else np.ascontiguousarray(_column(data, by), dtype=float))
    return blocks, names, designs, penalties, bys


def _effects_of(core_effect, blocks, newdata, z, unseen, corrected, centre=False):
    """`{term: (estimate, lower, upper)}` from a core effect evaluator.

    The evaluator maps a raw design row linearly to the effect and its
    standard error, so centring the raw design over the rows (with a `by=`
    column multiplied in first) gives the centred effect and the standard
    error of that contrast."""
    out = {}
    for index, (term, design, by) in enumerate(blocks):
        raw = np.ascontiguousarray(design.transform(newdata, unseen=unseen), dtype=float)
        by_values = (None if by is None
                     else np.ascontiguousarray(_column(newdata, by), dtype=float))
        if centre:
            if by_values is not None:
                raw = raw * by_values[:, None]
                by_values = np.ones(raw.shape[0])
            raw = np.ascontiguousarray(raw - raw.mean(axis=0, keepdims=True))
        # The sum-to-zero reparametrization turns a zero indicator row into
        # a nonzero row; an unseen group's effect is zero, not that row.
        estimate, se = core_effect(
            index, raw, by_values,
            zero_rows_have_no_effect=term.lstrip().startswith("brandom"),
            corrected=corrected)
        estimate, se = np.asarray(estimate), np.asarray(se)
        out[term] = (estimate, estimate - z * se, estimate + z * se)
    return out


def _treat_as_observed(data, impute, name):
    """`data` with the imputation indicators of `name` set to a row where it was observed.

    A contrast sets `name` to a value in every row, so the missing-value
    indicator is 0 and the observed-row indicator is 1 (the fill is not used).
    """
    if impute is None:
        return data
    for step in impute.steps:
        if step.source != name or step.kind not in ("missing_indicator", "observed_indicator"):
            continue
        data = _with_column(data, step.name, 0.0 if step.kind == "missing_indicator" else 1.0)
    return data


def _with_column(data, name, value):
    """A copy of `data` whose column `name` is `value` in every row."""
    if isinstance(data, dict):
        n = len(np.asarray(next(iter(data.values()))))
        out = dict(data)
        out[name] = np.full(n, value)
        return out
    if hasattr(data, "with_columns"):                       # polars
        import polars as pl

        return data.with_columns(pl.lit(value).alias(name))
    return data.assign(**{name: value})                     # pandas


def _average_contrast(refit, data, variable, reference, alternative, level):
    """`(estimate, lower, upper)` of the average over the rows of `data` of the
    refit's change in the link-scale predictor when `variable` moves from
    `reference` to `alternative`, every other column at each row's value.

    A term's effect is linear in the coefficients, so the average over rows of
    its raw-design rows at `alternative` minus those at `reference` is a single
    raw row. The core evaluates that row as an effect with its standard error,
    which is the standard error of the average. A variable no term reads has a
    contrast of exactly zero. A variable read by two or more terms would need
    the covariance between those terms' averages, which the per-term core call
    does not give, and is refused by name.
    """
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + level / 2)
    # A switched term's filled part is multiplied by the missing indicator, which
    # a contrast sets to 0 (the value is observed), so it contributes nothing.
    filled_by = {step.name for step in (refit._impute.steps if refit._impute else ())
                 if step.kind == "missing_indicator" and step.source == variable}
    readers = [i for i, (term, _, by) in enumerate(refit._blocks)
               if variable in _term_columns(term) and by not in filled_by]
    if not readers:
        return 0.0, 0.0, 0.0
    if len(readers) > 1:
        names = [refit._blocks[i][0] for i in readers]
        raise NotImplementedError(
            f"the contrast of {variable!r} is read by {len(names)} refit terms "
            f"({', '.join(names)}), and an interval for their sum needs the covariance "
            f"between them, which is not computed here. Write those terms as one term, "
            f"or refit without a contrast on {variable!r}")
    index = readers[0]
    term, design, by = refit._blocks[index]
    base = refit._imputed(data)
    rows = []
    for value in (alternative, reference):
        frame = _treat_as_observed(_with_column(base, variable, value), refit._impute, variable)
        raw = np.asarray(design.transform(frame), dtype=float)
        if by is not None:
            raw = raw * _column(frame, by).astype(float)[:, None]
        rows.append(raw)
    average = np.ascontiguousarray((rows[0] - rows[1]).mean(axis=0, keepdims=True))
    # The rows above already carry the term's by values, so the averaged row
    # is multiplied by one (the core needs a by value for each row it is given).
    by_one = np.ones(1) if by is not None else None
    estimate, se = refit._core.effect(index, average, by_one, zero_rows_have_no_effect=False,
                                      corrected=False)
    estimate = float(np.asarray(estimate).ravel()[0])
    se = float(np.asarray(se).ravel()[0])
    return estimate, estimate - z * se, estimate + z * se


_CENTER_ARG = re.compile(r",\s*center\s*=\s*(TRUE|True|true|1)\b")
_BBS_TERM = re.compile(rf"\s*bbs\(\s*({IDENT})\s*(,.*)?\)\s*$")
_BOLS_TERM = re.compile(rf"\s*bols\(\s*({IDENT})\s*(,\s*by\s*=\s*{N}\s*)?\)\s*$")


def _merge_centred_smooths(terms):
    """Refit ``bols(x) + bbs(x, center=TRUE)`` as the one smooth ``bbs(x)``.

    Boosting splits a smooth into its linear part (``bols``) and its
    deviation from linear (the centred ``bbs``) so that the two can be
    selected separately. A penalized refit estimates both at once: the
    difference penalty of ``bbs(x)`` leaves linear functions unpenalized
    (for ``differences >= 2``), so the full smooth contains the linear term
    and REML decides how far to bend it. The centred term becomes ``bbs(x)``
    and a ``bols`` over the same column and ``by=`` is dropped.
    """
    def by_of(args):
        found = re.search(rf"by\s*=\s*({N})", args or "")
        return found.group(1) if found else None

    out = []
    smooths = set()
    for term in terms:
        match = _BBS_TERM.match(term)
        if match and _CENTER_ARG.search(term):
            # REML chooses the smoothing, so the boosting df goes too.
            term = re.sub(rf",\s*(df|lambda)\s*=\s*{N}", "", _CENTER_ARG.sub("", term))
            match = _BBS_TERM.match(term)
        if match:
            args = match.group(2) or ""
            differences = re.search(r"differences\s*=\s*(\d+)", args)
            if differences is None or int(differences.group(1)) >= 2:
                smooths.add((match.group(1), by_of(args)))
        if term not in out:
            out.append(term)
    kept = []
    for term in out:
        match = _BOLS_TERM.match(term)
        if match and (match.group(1), by_of(match.group(2))) in smooths:
            continue
        kept.append(term)
    return kept


@fit_checkpoint
def gam_refit(model, terms=None, *, which="selected", unsupported="error"):
    """Refit a boosted model's terms as one penalized GAM with REML smoothing.

    Boosting shrinks every effect and couples selection with smoothing, so a
    bootstrap of the boosted fit reproduces the shrinkage and its bands
    under-cover (62% for nominal 90% on `benches/bench_insights.py`'s additive
    task). This refits the chosen terms jointly on the same designs and
    penalties the learners use: a penalized least-squares fit with one
    smoothing parameter per term estimated by REML, the construction of
    mgcv's `gam(method="REML")` (Wood 2011), and pointwise Bayesian intervals
    from the coefficient covariance.

    Parameters
    ----------
    model : GAMBoostModel
        A `gamboost` fit with the Gaussian, Binomial, Poisson, NBinomial,
        GammaReg or CoxPH family. Effects are on the canonical link scale:
        log-odds for Binomial (mboostrs' own Binomial predictor is half the
        log-odds), the log of the mean for Poisson, NBinomial and GammaReg,
        and the log hazard ratio for CoxPH (fitted through the Poisson form of
        the Cox likelihood, alternating with a Breslow baseline hazard).
    terms : sequence of str, optional
        Formula terms to refit. By default the terms of the model's formula
        whose learners were selected (``which="selected"``) or all of them
        (``which="all"``); pass the output of `deselect(...).kept` to refit a
        deselected model.
    unsupported : {"error", "hold"}
        What to do with a term the refit cannot represent (``btree``,
        ``bmono``, ``bkernel`` ...). ``"error"`` refuses it by name;
        ``"hold"`` keeps it at its boosted contribution, entered as a known
        offset, and refits the other terms around it. Held terms are listed in
        ``GamRefit.held``, included in ``predict`` and absent from
        ``effects``, and the intervals are conditional on them.

    Returns
    -------
    GamRefit

    Supported terms are ``bbs(x, ...)``, ``bols(x)`` and ``brandom(g, ...)``
    over one column, each optionally with a numeric ``by=`` column (a
    varying coefficient), and ``bspatial``/``%X%`` tensor terms; any other
    term is refused by name. A centred smooth ``bbs(x, center=TRUE)`` is
    refitted as ``bbs(x)``, which absorbs a ``bols(x)`` term over the same
    column: the refit's penalty leaves the linear part free. Non-Gaussian families are fitted by penalized
    iteratively reweighted least squares, and their smoothing parameters
    minimize the Laplace-approximate REML criterion, as mgcv does; the
    negative binomial theta and the gamma shape are estimated by the same
    criterion. The scale is fixed at 1 for Binomial, Poisson and Cox. The intervals are
    conditional on the chosen terms: when the terms were selected on the same
    data, they do not account for that selection.
    """
    if type(model).__name__ != "GAMBoostModel":
        raise TypeError(
            "gam_refit works on gamboost models, got "
            f"{type(model).__name__}. Fit a model with mb.gamboost(formula, data) "
            "first, then pass it to gam_refit.")
    family = type(model._family).__name__
    if family not in _REFIT_FAMILIES:
        raise NotImplementedError(
            f"gam_refit supports {', '.join(sorted(_REFIT_FAMILIES))}; this model is "
            f"{family}")
    bundle = _bundle(model)
    data = bundle["raw_data"]
    raw_formula = bundle["raw_formula"]
    impute = None
    if bundle.get("na_action") in ("impute", "impute_model", "impute_model_split"):
        # The raw frame still has its missing values; the imputation steps
        # of the model's recipe fill them as the fit did, and the indicator
        # terms the fit added join the formula.
        from mboostrs.transforms import TransformRecipe, apply_recipe

        recipe = getattr(model, "_transform_recipe", None)
        steps = tuple(s for s in (recipe.steps if recipe else ())
                      if s.kind in ("impute", "impute_model", "missing_indicator",
                                    "observed_indicator", "impute_level"))
        if steps:
            impute = TransformRecipe(steps)
            data = apply_recipe(impute, data)
        raw_formula = _with_imputation_terms({"": raw_formula}, {"": bundle["formula"]},
                                             [""])[""]
    lhs, all_terms = _terms(raw_formula, data)
    # Refitting every term of the formula is refitting a prespecified model;
    # any other set was chosen, here or by the caller, from these data.
    conditional = not (terms is None and which == "all")
    if terms is None:
        mapping = _term_learners(model, all_terms)
        selected = set(model.selected)
        terms = [t for s, t in all_terms if s == "+" and t.strip() not in ("1", "0")
                 and (which == "all" or any(n in selected for n in mapping.get(t, [])))]
    # gamboost fits a bare numeric name as a default `bbs` (as R's gamboost
    # does) and a bare categorical name as its level design.
    from mboostrs.transforms import _is_categorical

    terms = [(t.strip() if not _LSS_BARE.match(t) else
              f"brandom({t.strip()})" if _is_categorical(data, t.strip()) else f"bbs({t.strip()})")
             for t in terms]
    terms = _merge_centred_smooths(terms)
    if not terms:
        if not any(s == "+" and t.strip() not in ("1", "0") for s, t in all_terms):
            raise ValueError(
                "gam_refit has no term to refit: the formula is intercept only "
                f"({raw_formula!r}), so the model is its offset and has no smooth "
                "or linear effect to estimate. Refit with a formula that has terms.")
        raise ValueError(
            "no terms to refit: boosting selected none of the formula's terms. "
            "Pass terms= to name the terms to refit.")
    if unsupported not in ("error", "hold"):
        raise ValueError('unsupported must be "error" or "hold"')
    refused = [t for t in terms if not (_REFIT_TERM.match(t) or _REFIT_SPATIAL.match(t)
                                        or "%X%" in t)]
    if refused and unsupported == "error":
        raise NotImplementedError(
            f"gam_refit supports bbs, bols and brandom over one column, bspatial and "
            f"%X% tensor terms; cannot refit {refused}. Pass unsupported='hold' to "
            f"keep them at their boosted contribution and refit the rest")
    held_learners = ()
    if refused:
        mapping = _term_learners(model, all_terms)
        held_learners = tuple(name for t in refused for name in mapping.get(t, []))
        terms = [t for t in terms if t not in refused]
        if not terms:
            raise ValueError(f"no term the refit can represent; all of {refused} are held")

    surv = re.fullmatch(rf"\s*Surv\(\s*({N})\s*,\s*({N})\s*\)\s*", lhs)
    if family == "CoxPH":
        if not surv:
            raise ValueError("a CoxPH refit needs a Surv(time, status) response")
        event_time = np.asarray(_column(data, unquote_name(surv.group(1))), dtype=float)
        y = np.asarray(_column(data, unquote_name(surv.group(2))), dtype=float)
    else:
        y = np.asarray(_column(data, unquote_name(lhs)), dtype=float)
    n = y.size
    weights = bundle.get("weights")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    # A per-row offset (log exposure) is part of the linear predictor and is
    # not estimated; a scalar offset is the boosting start and is absorbed by
    # the refit's intercept.
    raw_offset = (bundle.get("raw_arguments") or {}).get("offset")
    offset = (np.asarray(raw_offset, dtype=float).ravel()
              if raw_offset is not None and np.ndim(raw_offset) > 0 else np.zeros(n))
    # mboostrs' Binomial predictor is half the log-odds; the refit's is the
    # log-odds, so a held term's boosted contribution is doubled.
    held_scale = 2.0 if family == "Binomial" else 1.0
    if held_learners:
        offset = offset + held_scale * sum(
            np.asarray(model.predict(newdata=data, which=name), dtype=float).ravel()
            for name in held_learners)
    if family == "NBinomial":
        # The dispersion is taken from the boosted fit, as mgcv's `nb()`
        # would estimate it; the refit's working weights use it as known.
        theta = float(np.asarray(model.nuisance()).ravel()[0])

    # One block per term: the learner's raw design and penalties at the
    # fitting rows. The constraint, the REML fit and the intervals are
    # computed in Rust (`mboostrs_core::gam`).
    from mboostrs import _rust

    blocks, names, designs, penalties, bys = _refit_parts(terms, data)
    core = call_fit(_rust.gam_refit_core,
        names, designs, penalties, bys, np.ascontiguousarray(y, dtype=float),
        weights=np.ascontiguousarray(w, dtype=float),
        offset=np.ascontiguousarray(offset, dtype=float), family=family,
        theta=theta if family == "NBinomial" else None,
        time=np.ascontiguousarray(event_time, dtype=float) if family == "CoxPH" else None,
        variables=[list(_term_columns(name)) for name in names])
    _warn_refit_convergence(dict(core.convergence()), terms=terms)
    edf, smoothing = {}, {}
    for block in core.blocks():
        edf[block["name"]] = float(block["edf"])
        values = [float(v) for v in block["lambda"]]
        # One value per margin for a tensor term, as mgcv reports te() terms.
        smoothing[block["name"]] = (values[0] if len(values) == 1
                                    else tuple(values) if values else 0.0)
    return GamRefit(terms=tuple(terms), edf=edf, smoothing=smoothing,
                    sigma=float(np.sqrt(core.scale)), conditional_on_selection=conditional,
                    link=core.link, reml_score=float(core.reml_score),
                    convergence=dict(core.convergence()),
                    theta=core.theta, held=tuple(refused),
                    _blocks=tuple(blocks), _core=core,
                    _held_model=model if held_learners else None,
                    _held_learners=held_learners, _held_scale=held_scale,
                    _impute=impute, _training_data=data, _weights=w,
                    _ranges=_training_ranges(terms, data))


class GamRefitBootstrap:
    """The REML refits of a row bootstrap, one per draw, of a :class:`GamRefit`.

    Made by :meth:`GamRefit.bootstrap`. :meth:`predict` gives each draw's
    link-scale prediction at new rows, so the spread of the draws at a row is
    the bootstrap standard error of the refit's prediction there.
    """

    def __init__(self, core, refit):
        self._core = core
        self._refit = refit

    @property
    def n_draws(self):
        """The number of bootstrap draws."""
        return int(self._core.n_draws)

    def converged(self):
        """Whether each draw's smoothing-parameter search converged, as a list
        of booleans in draw order."""
        return [bool(v) for v in self._core.converged()]

    def predict(self, newdata, extrapolation="warn"):
        """The link-scale prediction of every draw at the rows of ``newdata``,
        as a ``draws x rows`` array.

        ``extrapolation`` is applied to the spline covariates as in
        :meth:`GamRefit.predict`; the default ``"warn"`` is the same default
        and keeps the edge-slope continuation, with one warning naming the rows.
        ``"clip"`` holds each covariate at its training range edge.
        """
        refit = self._refit
        pairs = _range_rows(refit._ranges, newdata)
        newdata = _apply_range_policy(pairs, newdata, extrapolation)
        raws, bys, zeros = refit._design_rows(refit._imputed(newdata), "error")
        return np.asarray(self._core.linear_predictor(raws, bys, zeros), dtype=float)


def _term_margins(term):
    """The data columns a refit term reads, without its ``by=`` column."""
    from mboostrs.formula import term_variables

    base = re.sub(rf",\s*by\s*=\s*{IDENT}", "", term)
    return term_variables(base)


def _null_sharing_message(dropped, shared, terms):
    """The RuntimeWarning text for unpenalized directions held at zero."""
    head = (f"gam_refit held {dropped} unpenalized direction(s) at zero: the "
            "data cannot separate the unpenalized parts of the terms named "
            "below, so their sum is estimated and the split between them is not. ")
    if not shared:
        return head + ("Two or more terms have the same linear part on the "
                       "fitting rows; drop one of them, or drop a duplicate column.")
    named = ", ".join(repr(t) for t in shared)
    margins = {t: _term_margins(t) for t in shared}
    remedies = []
    for main in shared:
        main_vars = margins[main]
        if len(main_vars) != 1:
            continue
        for tensor in shared:
            tensor_vars = margins[tensor]
            if tensor is main or len(tensor_vars) < 2 or main_vars[0] not in tensor_vars:
                continue
            remedies.append(
                f"drop the main effect {main!r}, because the tensor {tensor!r} "
                f"already contains its linear part on the fitting rows")
    if remedies:
        remedy = "; or ".join(remedies)
    elif any(len(margins[t]) > 1 for t in shared):
        remedy = ("drop one of the named terms, or refit the tensor without the "
                  "shared margin, since a tensor's margins carry main-effect "
                  "directions")
    else:
        remedy = ("drop one of the named terms, or drop a duplicate column, "
                  "since they have the same linear part on the fitting rows")
    return head + f"Terms sharing the directions: {named}. Remedy: {remedy}."


def _warn_refit_convergence(convergence, terms=()):
    """Warn when the REML iteration did not converge, saying why."""
    import warnings

    dropped = int(convergence.get("unidentified_directions", 0) or 0)
    if dropped:
        # The core names the blocks whose unpenalized directions take part in
        # the dependency; keep their formula order.
        named = set(convergence.get("unidentified_terms", ()))
        shared = tuple(t for t in terms if t in named)
        warnings.warn(_null_sharing_message(dropped, shared, terms),
                      RuntimeWarning, stacklevel=3)
    if convergence.get("converged", True):
        return
    reason = convergence.get("stop_reason", "unrecorded")
    gradient = float(convergence.get("gradient_norm", float("nan")))
    if reason == "no_descent":
        detail = ("no step along the Newton direction lowered the REML "
                  "criterion although its gradient is not small. This happens "
                  "when several terms are nearly the same function of the data "
                  "and are left almost unpenalized, so the criterion cannot be "
                  "computed accurately there. Remove near-duplicate predictors "
                  "or refit fewer terms")
    elif reason in ("iteration_limit", "pirls_iteration_limit", "outer_iteration_limit"):
        detail = (f"the iteration stopped at its limit ({reason}); raise the "
                  "corresponding limit or check the terms for redundancy")
    else:
        detail = f"stop reason {reason!r}"
    warnings.warn(
        f"gam_refit's smoothing-parameter iteration did not converge (largest "
        f"gradient component {gradient:.3g}): {detail}. Predictions can still be "
        "accurate, but smoothing parameters, edf and intervals are not at a "
        "REML optimum; `refit.convergence` has the details.",
        RuntimeWarning, stacklevel=3)


# ------------------------------------------------------------- gamlss_refit
@dataclass(frozen=True)
class GamlssRefit:
    """A penalized refit of a distributional model's terms, all parameters
    estimated jointly, with REML smoothing parameters.
    Usage::

        import mboostrs as mb
        refit = mb.gamlss_refit(model)  # model: a gamboostLSS/glmboostLSS fit
        refit.effects(newdata, parameter="mu")
        refit.predict(newdata)

    Produced by :func:`gamlss_refit`. ``edf`` and ``smoothing`` map each
    distribution parameter to ``{term: value}``. :meth:`effects` returns each
    term's centred effect with a pointwise interval; :meth:`predict` returns
    each parameter on the link or the natural scale.
    """

    family: str
    terms: dict
    edf: dict
    smoothing: dict
    reml_score: float
    convergence: dict
    #: The formula terms were chosen on the same data the refit uses; the
    #: intervals are conditional on that choice.
    conditional_on_selection: bool
    _blocks: dict = field(repr=False, compare=False, default_factory=dict)
    _core: object = field(repr=False, compare=False, default=None)
    #: The boosted model's predictor-recipe replay (factor codes, derived
    #: columns). The refit reads the model's training frame, where factor
    #: labels are already integer codes, so new rows pass through the same
    #: recipe before any design reads them.
    _replay: object = field(repr=False, compare=False, default=None)
    #: The distributional model's training frame, held by reference for the
    #: provenance entry of :meth:`to_dict`; not copied.
    _training_data: object = field(repr=False, compare=False, default=None)
    #: ``{parameter: {column: (low, high, term)}}``: the training range of each
    #: spline covariate of each parameter's terms.
    _ranges: dict = field(repr=False, compare=False, default_factory=dict)
    #: Each parameter's intercept: ``{parameter: {"estimate": ..., "se": ...}}``
    #: on the link scale, where ``se`` is the square root of the intercept's
    #: Bayesian variance. Every parameter has one, including a parameter whose
    #: formula has no term (see `intercept_only`).
    intercepts: dict = field(default_factory=dict)
    #: The parameters with no smooth or linear term in the refit. Their whole
    #: predictor is the intercept, so `intercepts` is their coefficient and
    #: `vcov()` its variance.
    intercept_only: tuple = ()

    def _prepared(self, newdata, unseen_levels):
        if self._replay is None or newdata is None:
            return newdata
        return self._replay(newdata, unseen_levels=unseen_levels)

    @property
    def parameters(self):
        """The distribution parameters, in family order."""
        return tuple(self.terms)

    def _index(self, parameter):
        if parameter not in self.terms:
            raise ValueError(f"parameter must be one of {self.parameters}; got {parameter!r}")
        return self.parameters.index(parameter)

    def vcov(self, type="bayesian"):
        """Covariance of the joint coefficients (each parameter's intercept,
        then its terms, parameter by parameter); ``type="corrected"`` adds the
        first-order uncertainty of the smoothing parameters."""
        if type == "bayesian":
            return np.asarray(self._core.cov)
        if type == "corrected":
            return np.asarray(self._core.cov_corrected)
        raise ValueError('type must be "bayesian" or "corrected"')

    def effects(self, newdata, parameter=None, level=0.9, *, unseen_levels="error",
                corrected=False):
        """``{parameter: {term: (estimate, lower, upper)}}`` at the rows of
        `newdata`, on each parameter's link scale; with `parameter`, only that
        parameter's ``{term: ...}``. Each effect is centred over the fitting
        rows. ``unseen_levels`` and ``corrected`` are as in
        :meth:`GamRefit.effects`."""

        if unseen_levels not in ("error", "zero_effect", "average"):
            raise ValueError("unseen_levels must be 'error', 'zero_effect' or 'average'")
        return self._effects_prepared(self._prepared(newdata, unseen_levels), parameter,
                                      level, unseen_levels, corrected)

    def _effects_prepared(self, newdata, parameter, level, unseen_levels, corrected):
        from statistics import NormalDist

        z = NormalDist().inv_cdf(0.5 + level / 2)
        unseen = {"zero_effect": "zero", "average": "average"}.get(unseen_levels, "error")
        wanted = self.parameters if parameter is None else (parameter,)
        out = {}
        for name in wanted:
            q = self._index(name)
            out[name] = _effects_of(
                lambda b, raw, by, *, zero_rows_have_no_effect, corrected, q=q:
                    self._core.effect(q, b, raw, by, zero_rows_have_no_effect, corrected),
                self._blocks[name], newdata, z, unseen, corrected)
        return out if parameter is None else out[parameter]

    def _union_ranges(self):
        """The training range of every spline covariate of any parameter."""
        union = {}
        for ranges in self._ranges.values():
            for name, bounds in ranges.items():
                union.setdefault(name, bounds)
        return union

    def predict(self, newdata, parameter=None, type="link", *, unseen_levels="error",
                interval=False, level=0.9, corrected=False, output="auto",
                extrapolation="warn"):
        """Each parameter's linear predictor (``type="link"``) or its value on
        the natural scale through the family's inverse link
        (``type="response"``), as ``{parameter: array}``; with `parameter`,
        that parameter's array.

        ``interval=True`` returns, for each parameter, a frame with
        ``estimate``, ``se``, ``lower`` and ``upper`` per row (a frame for
        `parameter` alone, else a dict of frames). The estimate is the
        parameter's intercept plus its terms, and ``se`` is
        ``sqrt(x' V x)`` over that parameter's own coefficients (its
        intercept and terms), with the covariance between them. ``se`` is on
        the link scale; with ``type="response"`` the bounds are mapped through
        the inverse link. ``outside_range`` is true for a row where a spline
        covariate of that parameter's terms lies outside its training range.
        ``level``, ``corrected``, ``output`` and ``extrapolation`` are as in
        :meth:`GamRefit.predict`; the policy is applied to each covariate
        once, across all parameters that read it."""
        if type not in ("link", "response"):
            raise ValueError('type must be "link" or "response"')
        if unseen_levels not in ("error", "zero_effect", "average"):
            raise ValueError("unseen_levels must be 'error', 'zero_effect' or 'average'")
        pairs = _range_rows(self._union_ranges(), newdata)
        newdata = _apply_range_policy(pairs, newdata, extrapolation)
        if interval:
            return self._predict_interval(newdata, parameter, type, unseen_levels, level,
                                          corrected, output, pairs)
        newdata = self._prepared(newdata, unseen_levels)
        n = len(np.asarray(_column(newdata, next(iter(newdata)) if isinstance(newdata, dict)
                                   else newdata.columns[0])))
        wanted = self.parameters if parameter is None else (parameter,)
        effects = self._effects_prepared(newdata, None, 0.9, unseen_levels, False)
        out = {}
        for name in wanted:
            q = self._index(name)
            eta = np.full(n, self._core.intercept(q))
            for estimate, _, _ in effects[name].values():
                eta = eta + estimate
            out[name] = eta if type == "link" else np.asarray(
                self._core.response(q, np.ascontiguousarray(eta)))
        return out if parameter is None else out[parameter]

    def contrast(self, variable, from_value, to_value, *, data, parameter=None, level=0.9):
        """The average change in each parameter's linear predictor when covariates move.

        Each row of `data` is taken with the covariates at `from_value` in one
        copy and at `to_value` in another, every other column at the row's own
        value, and the mean of the difference of each parameter's link-scale
        predictor is returned with its standard error and ``level`` interval.
        Arguments are as for :meth:`GamRefit.contrast`. The result is a dict
        ``{parameter: EffectContrast}``, or one :class:`EffectContrast` when
        `parameter` is given. Each parameter's change is on its own link scale:
        the identity for ``mu`` and the log for a scale parameter.
        """
        names, lo_values, hi_values = _contrast_values(variable, from_value, to_value)
        base = self._prepared(data, "error")
        n = _frame_rows(base)
        lo_raws, lo_bys, lo_zeros = self._term_rows(_moved(base, names, lo_values, n), "error")
        hi_raws, hi_bys, hi_zeros = self._term_rows(_moved(base, names, hi_values, n), "error")
        pairs = self._core.parameter_contrasts(hi_raws, hi_bys, hi_zeros,
                                               lo_raws, lo_bys, lo_zeros, corrected=False)
        out = {name: _effect_contrast(float(est), float(se), level, "link")
               for name, (est, se) in zip(self.parameters, pairs, strict=True)}
        if parameter is None:
            return out
        self._index(parameter)  # raises for an unknown parameter name
        return out[parameter]

    def draws(self, newdata, n_draws=1000, *, seed=0, corrected=False, unseen_levels="error"):
        """Draws of every distribution parameter, on its natural scale, at the
        rows of `newdata`, from the approximate posterior of the joint
        coefficients: ``{parameter: array of shape (n_draws, n_rows)}``.

        A quantity that combines parameters, such as a centile of the
        response distribution, is computed draw by draw from these arrays,
        and its interval taken from the quantiles over the draws; the draws
        keep the correlation between the parameters' estimates.
        ``corrected=True`` draws from the covariance corrected for
        smoothing-parameter uncertainty."""
        if unseen_levels not in ("error", "zero_effect", "average"):
            raise ValueError("unseen_levels must be 'error', 'zero_effect' or 'average'")
        raws, bys, zeros = self._parameter_rows(newdata, unseen_levels)
        if not any(raws):
            raise ValueError("every parameter is intercept-only; there are no rows to draw at")
        arrays = self._core.parameter_draws(raws, bys, zeros, n_draws=int(n_draws),
                                            seed=int(seed), corrected=bool(corrected))
        return {name: np.asarray(a) for name, a in zip(self.parameters, arrays, strict=True)}

    def _parameter_rows(self, newdata, unseen_levels):
        """Each parameter's terms at the rows of `newdata`: raw designs, `by=`
        columns and zero-row flags, as ``[parameter][term]`` lists."""
        newdata = self._prepared(newdata, unseen_levels)
        return self._term_rows(newdata, unseen_levels)

    def _term_rows(self, newdata, unseen_levels):
        """`_parameter_rows` for data already passed through the predictor replay."""
        unseen = {"zero_effect": "zero", "average": "average"}.get(unseen_levels, "error")
        raws, bys, zeros = [], [], []
        for name in self.parameters:
            q_raw, q_by, q_zero = [], [], []
            for term, design, by in self._blocks[name]:
                q_raw.append(np.ascontiguousarray(design.transform(newdata, unseen=unseen),
                                                  dtype=float))
                q_by.append(None if by is None
                            else np.ascontiguousarray(_column(newdata, by), dtype=float))
                q_zero.append(term.lstrip().startswith("brandom"))
            raws.append(q_raw)
            bys.append(q_by)
            zeros.append(q_zero)
        return raws, bys, zeros

    def _predict_interval(self, newdata, parameter, type, unseen_levels, level, corrected,
                          output, pairs=()):
        from statistics import NormalDist

        from mboostrs._output import output_frame

        if not 0 < level < 1:
            raise ValueError(f"level must be between 0 and 1; got {level!r}")
        raws, bys, zeros = self._parameter_rows(newdata, unseen_levels)
        if not any(raws):
            raise ValueError("every parameter is intercept-only; there are no rows to predict at")
        predicted = self._core.parameter_predictions(raws, bys, zeros, corrected=bool(corrected))
        z = NormalDist().inv_cdf(0.5 + level / 2)
        if parameter is not None:
            self._index(parameter)  # raises for an unknown parameter name
        wanted = self.parameters if parameter is None else (parameter,)
        out = {}
        for name, (eta, se) in zip(self.parameters, predicted, strict=True):
            if name not in wanted:
                continue
            eta = np.asarray(eta, dtype=float)
            se = np.asarray(se, dtype=float)
            lower, upper = eta - z * se, eta + z * se
            # Only the covariates this parameter's own terms read decide its flags.
            own = [(row, values) for row, values in pairs
                   if row.variable in self._ranges.get(name, {})]
            flags = _outside_mask(own, n=eta.size)
            if type == "response":
                q = self._index(name)
                eta, lower, upper = (np.asarray(self._core.response(
                    q, np.ascontiguousarray(a))) for a in (eta, lower, upper))
            rows = [{"estimate": float(e), "se": float(s), "lower": float(lo), "upper": float(hi),
                     "outside_range": bool(flag)}
                    for e, s, lo, hi, flag in zip(eta, se, lower, upper, flags, strict=True)]
            out[name] = output_frame(rows, output,
                                     columns=("estimate", "se", "lower", "upper",
                                              "outside_range"))
        return out[parameter] if parameter is not None else out

    def summary(self):
        """Each distribution parameter's terms, with their effective degrees of freedom.

        One row per parameter and term: ``edf`` is the effective degrees of
        freedom the penalized refit gives the term and ``smoothing`` its REML
        smoothing parameter. The refit carries no term tests, so the table
        reports the fitted quantities only. :meth:`effects` gives a term's
        shape with a pointwise interval.
        """
        from mboostrs.summary import Summary, SummaryTable

        rows = [[parameter, term, float(self.edf[parameter][term]),
                 float(self.smoothing[parameter][term])]
                for parameter in self.parameters
                for term in self.edf.get(parameter, {})]
        table = SummaryTable(None, ["parameter", "term", "edf", "smoothing"], rows)
        header = [("Parameters", ", ".join(self.parameters))]
        return Summary("Distributional penalized refit (REML)", header, [table],
                       ["Effective degrees of freedom and REML smoothing parameters "
                        "as estimated by the joint refit."])

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the terms, edf and smoothing parameters per
        distribution parameter; ``include_arrays=True`` adds the coefficients."""
        extras = ({"coefficients": json_safe(np.asarray(self._core.coef))}
                  if include_arrays and self._core is not None else None)
        return export_result(self, include_arrays=include_arrays, extras=extras,
                             with_provenance=True, training_data=self._training_data)


_LSS_BARE = re.compile(rf"^\s*{IDENT}\s*$")


def _switched_pairs(rhs: str) -> dict:
    """{(kind, column): [term, term]} for the terms `impute_model` switched in `rhs`."""
    from mboostrs.transforms import _split_top

    pairs: dict = {}
    switched = set()
    terms = _split_top(rhs)
    for term in terms:
        m = re.match(rf"^\s*(bbs|bols)\(\s*({IDENT})", term)
        if m and "by=__observed_" in term:
            switched.add(m.group(2))
    for term in terms:
        m = re.match(rf"^\s*(bbs|bols)\(\s*({IDENT})", term)
        if m and m.group(2) in switched and "by=" in term:
            pairs.setdefault((m.group(1), m.group(2)), []).append(term)
    return pairs


def _replace_switched(written_rhs: str, pairs: dict) -> list:
    """The written terms with each plain term on a switched column replaced by its pair."""
    from mboostrs.transforms import _split_top

    out = []
    for term in _split_top(written_rhs):
        m = re.match(rf"^\s*(bbs|bols)\(\s*({IDENT})\s*([,)])", term)
        bare = term.strip()
        if m and "by=" not in term and "%X%" not in term and (m.group(1), m.group(2)) in pairs:
            out.extend(pairs[(m.group(1), m.group(2))])
        elif not m and any(column == bare for (_, column) in pairs):
            kind = next(k for (k, column) in pairs if column == bare)
            out.extend(pairs[(kind, bare)])
        else:
            out.append(term)
    return out


def _with_imputation_terms(formulas, rewritten, params):
    """Add the missing-value indicator terms ``na_action="impute"`` wrote.

    The model keeps the formula as written, and its training frame already
    carries the imputed columns and their ``<x>_missing`` indicators; the
    indicator terms exist only in the rewritten formula, so they are copied
    over to keep the refit's terms in step with the learners.
    """
    if rewritten is None:
        return formulas
    rewritten = rewritten if isinstance(rewritten, dict) else {p: rewritten for p in params}
    out = {}
    for name in params:
        written = formulas[name]
        extra = [t for t in _split_plus(partition_tilde(rewritten[name])[2])
                 if re.match(rf"^\s*bols\(\s*{IDENT}_missing_*\s*,\s*intercept\s*=\s*False\s*\)\s*$", t)
                 and t.strip() not in written]
        pairs = _switched_pairs(partition_tilde(rewritten[name])[2])
        if pairs:
            # The observed parts sum to 1 - <x>_missing, so with the intercept the
            # constant <x>_missing term lies in the span of the other terms, and a
            # refit with it is not identified. The refit leaves it out; the fitted
            # function space does not change. Boosting keeps the term.
            indicators = {b for term in (t for group in pairs.values() for t in group)
                          for b in re.findall(rf"by=({N})", term)
                          if not b.startswith("__observed_")}
            dropped = [t for t in extra
                       if any(re.search(rf"\b{re.escape(b)}\b", t) for b in indicators)]
            if dropped:
                warnings.warn(
                    f"the refit leaves out the constant term {', '.join(dropped)}: the "
                    "observed and filled parts of the switched term already contain it "
                    "(they sum to one, with the intercept)", UserWarning, stacklevel=3)
            extra = [t for t in extra if t not in dropped]
            # A column the fit switched into observed and filled terms is refitted
            # with the same two terms, in place of the plain term.
            lhs = partition_tilde(written)[0].strip()
            written = f"{lhs} ~ " + " + ".join(
                _replace_switched(partition_tilde(written)[2], pairs))
        out[name] = written + "".join(f" + {t}" for t in extra)
    return out


@fit_checkpoint
def gamlss_refit(model, terms=None, *, which="selected", factor_ridge=False):
    """Refit a distributional model's terms as one penalized likelihood with
    REML smoothing parameters.

    Every distribution parameter keeps the terms its boosting formula gave it
    (by default those selected at least once, ``which="selected"``; all of
    them with ``which="all"``; or ``terms={"mu": [...], "sigma": [...]}``).
    All coefficients are estimated jointly by penalized maximum likelihood,
    and the smoothing parameters minimize the Laplace-approximate restricted
    likelihood (Wood, Pya and Saefken 2016). A parameter with no terms keeps
    an intercept. Linear terms of a ``glmboostLSS`` formula are refitted as
    ``bols``.

    Parameters
    ----------
    model : LSSModel
        A ``gamboostLSS`` or ``glmboostLSS`` fit.
    terms : dict, optional
        Terms per distribution parameter.
    which : {"selected", "all"}
    factor_ridge : bool, default False
        Categorical ``bols`` terms are unpenalized: each level's effect is
        estimated from that level's rows alone, relative to the reference
        level. ``True`` puts an identity (ridge) penalty on the level
        indicators, with its strength chosen by REML, which shrinks every
        level's effect toward the reference level.

    Returns
    -------
    GamlssRefit

    The intervals are conditional on the chosen terms. Survival and
    multi-column responses are not supported.
    """
    from mboostrs import _rust

    if type(model).__name__ != "LSSModel":
        raise TypeError("gamlss_refit works on gamboostLSS and glmboostLSS models")
    family = model._family
    params = list(family.param_names)
    formula = model._formula
    if formula is None or model._data is None:
        raise ValueError("gamlss_refit needs the model's formula and data, which this "
                         "model does not carry (a loaded artifact)")
    formulas = formula if isinstance(formula, dict) else {p: formula for p in params}
    formulas = _with_imputation_terms(formulas, getattr(model, "_rewritten_formulas", None),
                                      params)
    data = model._data
    selected = model.selected
    conditional = not (terms is None and which == "all")
    chosen = {}
    for q, name in enumerate(params):
        lhs, all_terms = _terms(formulas[name], data)
        additive = [t.strip() for s, t in all_terms if s == "+" and t.strip() not in ("1", "0")]
        if terms is not None:
            wanted = [t.strip() for t in terms.get(name, [])]
        elif which == "all":
            wanted = additive
        else:
            # Named as `selected` names them (a `%X%` pair by its written term).
            # The intercept learner (`+ 1`) is not a term: the refit keeps
            # every parameter's intercept whether or not a term is chosen.
            learners = [name for name in model._learner_labels(q) if name != "1"]
            if len(learners) != len(additive):
                raise NotImplementedError(
                    f"the {name} formula's {len(additive)} terms do not map one to one "
                    f"onto its {len(learners)} learners; pass terms= explicitly")
            used = set(selected.get(name, []))
            wanted = [t for t, learner in zip(additive, learners, strict=True) if learner in used]
        chosen[name] = [f"bols({quote_name(t)})" if _LSS_BARE.match(t) else t for t in wanted]
    # A model refitted from rewritten formulas (deselect's reduced model)
    # names each `%X%` pair by its derived design, `buser(__ix0, ...)`; the
    # tensor recipe records the pair it stands for.
    tensors = {getattr(r, "derived", None): r.name
               for r in (getattr(model, "_interactions", None) or ())
               if getattr(r, "derived", None) and "%X%" in getattr(r, "name", "")}
    for name, ts in chosen.items():
        mapped = []
        for t in ts:
            derived = re.match(r"\s*buser\(\s*(__ix\d+)\s*,", t)
            mapped.append(tensors.get(derived.group(1), t) if derived else t)
        chosen[name] = mapped
    unsupported = [t for ts in chosen.values() for t in ts
                   if not (_REFIT_TERM.match(t) or _REFIT_SPATIAL.match(t) or "%X%" in t)]
    if unsupported:
        raise NotImplementedError(
            f"gamlss_refit supports bbs, bols and brandom over one column, bspatial and "
            f"%X% tensor terms; cannot refit {unsupported}")

    response = model._response_name or lhs.strip()
    y = np.ascontiguousarray(_column(data, response), dtype=float)
    if y.ndim != 1:
        raise NotImplementedError("gamlss_refit supports single-column responses")
    weights = model._weights
    # The training frame holds each factor the fit encoded as integer codes;
    # the model's transform recipe names those columns.
    encoded = frozenset(s.name for s in getattr(getattr(model, "_transform_recipe", None),
                                                "steps", ()) or ()
                        if getattr(s, "kind", None) == "factor")
    parts = {name: _refit_parts(chosen[name], data, factor_ridge, encoded) for name in params}
    start = model.predict(type="link")
    core = call_fit(_rust.gamlss_refit_core,
        family._rust_name, params,
        [parts[p][1] for p in params], [parts[p][2] for p in params],
        [parts[p][3] for p in params], [parts[p][4] for p in params], y,
        weights=None if weights is None else np.ascontiguousarray(weights, dtype=float),
        start_etas=[np.ascontiguousarray(start[p], dtype=float) for p in params],
        # The columns each term reads: a term is checked for columns repeated
        # by the terms nested in it (mgcv's gam.side).
        variables=[[list(_term_columns(t)) for t in parts[p][1]] for p in params])
    edf, smoothing, intercepts = {}, {}, {}
    coef = np.asarray(core.coef, dtype=float)
    variance = np.diag(np.asarray(core.cov, dtype=float))
    for info in core.params():
        edf[info["name"]] = {b["name"]: float(b["edf"]) for b in info["blocks"]}
        smoothing[info["name"]] = {
            b["name"]: (float(b["lambda"][0]) if len(b["lambda"]) == 1
                        else tuple(float(v) for v in b["lambda"]) if b["lambda"] else 0.0)
            for b in info["blocks"]}
        start = int(info["start"])
        intercepts[info["name"]] = {"estimate": float(coef[start]),
                                    "se": float(np.sqrt(max(variance[start], 0.0)))}
    return GamlssRefit(
        family=type(family).__name__, terms={p: tuple(chosen[p]) for p in params},
        edf=edf, smoothing=smoothing, reml_score=float(core.reml_score),
        convergence=dict(core.convergence()), conditional_on_selection=conditional,
        intercepts=intercepts,
        intercept_only=tuple(p for p in params if not chosen[p]),
        _blocks={p: tuple(parts[p][0]) for p in params}, _core=core,
        _replay=getattr(model, "_replay_newdata", None), _training_data=model._data,
        _ranges={p: _training_ranges(chosen[p], data) for p in params})
