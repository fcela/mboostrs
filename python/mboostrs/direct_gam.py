"""Penalized GAMs fitted by REML from a formula, without boosting.

`gam` and `gamlss` take a formula, build each term's design and penalty the
way `gam_refit` and `gamlss_refit` build them for a boosted model, and fit the
joint penalized likelihood in the Rust core, with one smoothing parameter per
penalty chosen by REML (Wood 2011). Every formula term is in the fit: no
base-learner selection and no boosting iteration count (`mstop`) take place.
The results are the `GamRefit` and `GamlssRefit` objects of the refit
functions, so `predict`, `effects`, `coef`, `vcov`, `draws` and `contrast`
behave as they do for a refit.
"""

from __future__ import annotations

import re

import numpy as np

from mboostrs import _rust
from mboostrs._names import IDENT_SRC as IDENT
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._workflow import call_fit, fit_checkpoint
from mboostrs.formula import refuse_intercept_control
from mboostrs.selection import (
    _LSS_BARE,
    _REFIT_SPATIAL,
    _REFIT_TERM,
    GamlssRefit,
    GamRefit,
    _column,
    _frame_rows,
    _refit_parts,
    _term_columns,
    _terms,
    _training_ranges,
    _warn_refit_convergence,
)

__all__ = ["gam", "gamlss"]

#: Accepted ``family=`` strings and the core's family names.
_GAM_FAMILIES = {"gaussian": "Gaussian", "binomial": "Binomial", "poisson": "Poisson",
                 "gamma": "GammaReg"}
#: Family objects from `mboostrs.families` and their core names.
_GAM_FAMILY_OBJECTS = {"Gaussian": "Gaussian", "GaussReg": "Gaussian",
                       "Binomial": "Binomial", "Poisson": "Poisson", "GammaReg": "GammaReg"}
_NA_ACTIONS = ("error", "omit", "impute")
_INDICATOR = re.compile(rf"bols\(\s*({IDENT}_missing_*)\s*,\s*intercept\s*=\s*False\s*\)")


def _family_name(family) -> str:
    if isinstance(family, str):
        key = family.strip().lower()
        if key not in _GAM_FAMILIES:
            raise ValueError(
                f"family must be one of {sorted(_GAM_FAMILIES)} or a family object such as "
                f"mb.Binomial(); got {family!r}")
        return _GAM_FAMILIES[key]
    name = type(family).__name__
    if name not in _GAM_FAMILY_OBJECTS:
        raise NotImplementedError(
            f"gam fits {', '.join(sorted(_GAM_FAMILIES))}; family {name} is not one of them. "
            "Fit it with mb.gamboost(...) and refit with gam_refit, or use mb.gamlss for a "
            "distributional family")
    return _GAM_FAMILY_OBJECTS[name]


def _check_na_action(na_action) -> None:
    if na_action not in _NA_ACTIONS:
        raise ValueError(f"na_action must be one of {_NA_ACTIONS}; got {na_action!r}")


def _check_method(method) -> None:
    if method != "REML":
        raise NotImplementedError(
            f"method={method!r} is not available: the penalized fit chooses its smoothing "
            "parameters by REML only. Pass method='REML'")


def _as_formula(formula) -> str:
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError(f"formula must be a string such as 'y ~ bbs(x)'; got {formula!r}")
    return formula


def _formula_terms(formula: str, data):
    """``(response, terms)`` of one formula, the terms as `gam_refit` reads them.

    A bare numeric name is ``bbs(name)``, a smooth as in `gamboost`, and a bare
    categorical name is ``bols(name)``, an unpenalized factor with one
    coefficient per level after the first. ``brandom(name)`` is a penalized
    random effect. The intercept belongs to the fit and is not a term.
    """
    from mboostrs.transforms import _is_categorical

    lhs, all_terms = _terms(formula, data)
    terms = []
    for sign, term in all_terms:
        text = term.strip()
        if sign != "+":
            raise NotImplementedError(
                f"gam takes additive terms only; '{sign} {text}' in {formula!r} is not "
                "supported. Remove it from the formula")
        if text in ("1", "0"):
            continue
        if _LSS_BARE.match(text):
            text = f"bols({quote_name(text)})" if _is_categorical(data, text) else f"bbs({quote_name(text)})"
        terms.append(text)
    refused = [t for t in terms if not (_REFIT_TERM.match(t) or _REFIT_SPATIAL.match(t)
                                        or "%X%" in t)]
    if refused:
        raise NotImplementedError(
            f"gam fits bbs, bols and brandom over one column, bspatial and %X% tensor terms; "
            f"cannot fit {refused}. Use mb.gamboost(...) with gam_refit(..., "
            "unsupported='hold') for those terms")
    return lhs.strip(), terms


def _formula_columns(formula: str) -> list[str]:
    """Every data column a formula reads, its response included."""
    from mboostrs.formula import term_variables

    lhs, _, rhs = partition_tilde(formula)
    return list(dict.fromkeys(term_variables(lhs.strip()) + term_variables(rhs)))


def _prepare_rows(formulas, data, weights, offset, na_action):
    """Apply ``na_action`` to the frame, the weights and the offset together.

    Returns ``(data, weights, offset, formulas, recipe)``. ``"impute"`` fills
    each missing numeric value with the training median and adds a
    missing-value indicator term to every formula that reads that column
    (`impute_missing`); the recipe replays the fill on new rows. ``"omit"``
    drops the rows with a missing value in a column any formula reads.
    ``"error"`` refuses, naming the column and its first missing row.
    """
    from mboostrs._frame import read_column, take_rows
    from mboostrs.transforms import apply_recipe, impute_missing

    names = list(dict.fromkeys(c for f in formulas for c in _formula_columns(f)))
    recipe = None
    if na_action == "impute":
        # One recipe for all formulas, so a column read by two parameters gets
        # the same fill and the same indicator in both.
        lhs = partition_tilde(formulas[0])[0]
        combined = lhs + "~ " + " + ".join(partition_tilde(f)[2].strip() for f in formulas)
        rewritten, recipe = impute_missing(combined, data, learner="bols")
        data = apply_recipe(recipe, data)
        indicators = _INDICATOR.findall(rewritten)
        out = []
        for f in formulas:
            read = _formula_columns(f)
            extra = [f"bols({quote_name(ind)}, intercept=False)" for ind in indicators
                     if ind[: ind.rfind("_missing")] in read and ind not in f]
            out.append(f + "".join(f" + {t}" for t in extra))
        formulas = out
    elif na_action == "omit":
        n = _frame_rows(data)
        mask = np.ones(n, dtype=bool)
        for name in names:
            mask &= ~np.isnan(np.asarray(read_column(data, name).values, dtype=float))
        if not mask.all():
            if not mask.any():
                raise ValueError("na_action='omit' would drop every row: each row has a "
                                 "missing value in a column the formula reads")
            data = take_rows(data, mask)
            weights = None if weights is None else np.asarray(weights, float)[mask]
            offset = None if offset is None else np.asarray(offset, float)[mask]
    else:
        for name in names:
            values = np.asarray(read_column(data, name).values, dtype=float)
            bad = np.isnan(values)
            if bad.any():
                first = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    f"the formula reads column {name!r}, which has {int(bad.sum())} missing "
                    f"value(s), first at row {first}. Pass na_action='impute' (median fill "
                    "and a missing-value indicator), na_action='omit' (drop those rows), or "
                    "fill the column before fitting")
    return data, weights, offset, formulas, recipe


class _Replay:
    """Replays an imputation recipe on new rows, as `GamRefit` does."""

    def __init__(self, recipe):
        self._recipe = recipe

    def __call__(self, newdata, *, unseen_levels="error"):
        from mboostrs.transforms import apply_recipe

        if self._recipe is None or newdata is None:
            return newdata
        return apply_recipe(self._recipe, newdata)


@fit_checkpoint
def gam(formula, data, *, family="gaussian", weights=None, offset=None,
        na_action="error", method="REML"):
    """Fit a penalized GAM by REML from a formula, with no boosting.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    Every term of the formula is in the fit, and each term's smoothing
    parameter is chosen by REML: the criterion of mgcv's
    ``gam(method="REML")`` (Wood 2011), maximized by Newton's method in the
    Rust core. The terms are built as :func:`gam_refit` builds them, so the
    fit uses the same designs and penalties as a refit of a boosted model whose
    selected terms are the same.

    Parameters
    ----------
    formula : str
        ``"y ~ terms"``. Supported terms: ``bbs(x, ...)``, a P-spline smooth (a
        bare numeric name means ``bbs(name)``); ``bols(x)``, a linear term (a
        bare categorical name means ``bols(name)``, an unpenalized factor with
        one coefficient per level after the first); ``brandom(g)``, a penalized
        random effect; ``bspatial(x, y)``, a smooth surface; and ``a %X% b``, a
        tensor product of two smooths. A numeric ``by=`` column makes a
        varying coefficient. A transform such as ``log(x)`` may be written
        inside a term.
    data : polars DataFrame, pandas DataFrame or dict of columns
    family : {"gaussian", "binomial", "poisson", "gamma"} or a family object
        ``mb.Gaussian()``, ``mb.Binomial()``, ``mb.Poisson()`` or ``mb.GammaReg()``.
        The link is the canonical one: identity, logit, log and log.
    weights : array, optional
        Prior weights, one per row.
    offset : array, optional
        A per-row offset on the link scale, such as log exposure. It is added
        to the linear predictor and not estimated.
    na_action : {"error", "omit", "impute"}
        ``"error"`` refuses a formula column with a missing value. ``"omit"``
        drops the rows that have one. ``"impute"`` fills a numeric column with
        its training median and adds an indicator term for the rows that were
        missing; the fill is replayed on new rows by :meth:`GamRefit.predict`.
    method : {"REML"}
        Only REML is available.

    Returns
    -------
    GamRefit
        With ``predict`` (``interval=True`` and ``extrapolation=``),
        ``effects``, ``contrast``, ``coef``, ``vcov``, ``summary``, and the
        per-term ``edf`` and ``smoothing``.

    Notes
    -----
    The intervals are Bayesian pointwise intervals conditional on the
    smoothing parameters, as :meth:`GamRefit.effects` describes. The
    smoothing parameters are estimated from the same rows the intervals use.
    """
    _check_method(method)
    _check_na_action(na_action)
    family_name = _family_name(family)
    formula = _as_formula(formula)
    refuse_intercept_control(formula, data, "gam")
    data, weights, offset, (prepared,), recipe = _prepare_rows(
        [formula], data, weights, offset, na_action)
    lhs, terms = _formula_terms(prepared, data)
    if not terms:
        raise ValueError(
            f"gam has no term to fit: {formula!r} has only an intercept. Add a term, for "
            "example a bbs(x) smooth or a bols(x) linear term")
    n = _frame_rows(data)
    y = np.ascontiguousarray(np.asarray(_column(data, lhs), dtype=float))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError(f"weights have {w.size} entries for {n} rows")
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    if off.size != n:
        raise ValueError(f"offset has {off.size} entries for {n} rows")
    blocks, names, designs, penalties, bys = _refit_parts(terms, data)
    core = call_fit(
        _rust.gam_refit_core, names, designs, penalties, bys, y,
        weights=np.ascontiguousarray(w, dtype=float),
        offset=np.ascontiguousarray(off, dtype=float), family=family_name,
        theta=None, time=None, variables=[list(_term_columns(name)) for name in names],
        gram=True)
    _warn_refit_convergence(dict(core.convergence()), terms=terms)
    edf, smoothing = {}, {}
    for block in core.blocks():
        edf[block["name"]] = float(block["edf"])
        values = [float(v) for v in block["lambda"]]
        smoothing[block["name"]] = (values[0] if len(values) == 1
                                    else tuple(values) if values else 0.0)
    return GamRefit(
        terms=tuple(terms), edf=edf, smoothing=smoothing,
        sigma=float(np.sqrt(core.scale)), conditional_on_selection=False,
        link=core.link, reml_score=float(core.reml_score),
        convergence=dict(core.convergence()), theta=core.theta, held=(),
        _blocks=tuple(blocks), _core=core, _held_model=None, _held_learners=(),
        _held_scale=1.0, _impute=recipe, _training_data=data, _weights=w,
        _ranges=_training_ranges(terms, data))


@fit_checkpoint
def gamlss(formulas, data, *, family=None, weights=None, na_action="error",
           method="REML"):
    """Fit a distributional regression by REML from one formula per parameter.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    Each distribution parameter (``mu`` and ``sigma`` of ``GaussianLSS``, for
    example) takes its own formula. All coefficients are estimated jointly by
    penalized maximum likelihood, and each smooth has its own smoothing
    parameter chosen by REML, as :func:`gamlss_refit` does for a boosted model.

    Parameters
    ----------
    formulas : dict
        ``{parameter: "y ~ terms"}``, with the parameter names of the family. A
        parameter left out has an intercept only. Every formula must name the
        same response.
    data : polars DataFrame, pandas DataFrame or dict of columns
    family : LSS family object, default ``mb.GaussianLSS()``
    weights : array, optional
        Prior weights, one per row.
    na_action : {"error", "omit", "impute"}
        As in :func:`gam`. The imputation is shared by all parameters.
    method : {"REML"}
        Only REML is available.

    Returns
    -------
    GamlssRefit
        With ``predict(parameter=..., interval=True)``, ``effects``,
        ``contrast``, ``draws`` and ``vcov``, per parameter. Each parameter is
        on its own link scale, identity for ``mu`` and log for ``sigma``.
    """
    from mboostrs.gamlss import GaussianLSS

    _check_method(method)
    _check_na_action(na_action)
    if family is None:
        family = GaussianLSS()
    if not isinstance(formulas, dict) or not formulas:
        raise ValueError(
            "formulas must be a dict such as {'mu': 'y ~ bbs(x)', 'sigma': 'y ~ bols(z)'}; "
            f"got {type(formulas).__name__}")
    params = tuple(family.param_names)
    unknown = [k for k in formulas if k not in params]
    if unknown:
        raise ValueError(f"{type(family).__name__} has parameters {params}; the formulas "
                         f"name {unknown}, which it does not have")
    for formula in formulas.values():
        refuse_intercept_control(formula, data, "gamlss")
    given = {k: _as_formula(formulas[k]) for k in params if k in formulas}
    responses = {k: partition_tilde(f)[0].strip() for k, f in given.items()}
    if len(set(responses.values())) > 1:
        raise ValueError(f"every formula must name the same response; got {responses}")
    response = next(iter(responses.values()))
    text = [given.get(k, f"{quote_response(response)} ~ 1") for k in params]
    data, weights, _, prepared, recipe = _prepare_rows(text, data, weights, None, na_action)
    n = _frame_rows(data)
    y = np.ascontiguousarray(np.asarray(_column(data, response), dtype=float))
    if weights is not None:
        weights = np.ascontiguousarray(np.asarray(weights, dtype=float).ravel())
        if weights.size != n:
            raise ValueError(f"weights have {weights.size} entries for {n} rows")
    chosen, parts = {}, {}
    for p, formula in zip(params, prepared, strict=True):
        _, chosen[p] = _formula_terms(formula, data)
        parts[p] = _refit_parts(chosen[p], data)
    core = call_fit(
        _rust.gamlss_refit_core, family._rust_name, list(params),
        [parts[p][1] for p in params], [parts[p][2] for p in params],
        [parts[p][3] for p in params], [parts[p][4] for p in params], y,
        weights=weights, start_etas=None,
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
        convergence=dict(core.convergence()), conditional_on_selection=False,
        intercepts=intercepts,
        intercept_only=tuple(p for p in params if not chosen[p]),
        _blocks={p: tuple(parts[p][0]) for p in params}, _core=core,
        _replay=_Replay(recipe), _training_data=data,
        _ranges={p: _training_ranges(chosen[p], data) for p in params})
