"""Effect modifiers from AIPW scores: best linear predictor slopes and GATES.

The AIPW score of a row has conditional mean equal to the effect at that row's
covariates. Regressing the scores on a covariate therefore measures how the
effect changes along that covariate, and this module runs that regression for
each covariate and for all covariates jointly. The numerics (weighted least
squares with HC1 standard errors, group means, contrasts, the Wald test and
Holm adjustment) are in the Rust core; this module builds the design matrices
and assembles the results.

Before it claims any modifier the function runs a gate: a cross-fitted
R-learner effect model (:func:`mboostrs.effect_model`) on the tested covariates
must beat a constant effect out of fold. The score regressions treat the scores
as fixed numbers and omit the error of the nuisance fits, so on data with no
effect modification they can report a modifier; the gate does not share that
error source.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from mboostrs import _rust
from mboostrs._output import export_result, output_frame
from mboostrs.causal import AIPWResult, _check_bounds, _modifier_values
from mboostrs.transforms import (
    _CLAMP_RE,
    _IDENTITY_RE,
    _TRANSFORM_RE,
    _column,
    _has_column,
    apply_recipe,
    extract_transforms,
    fit_recipe,
)

__all__ = ["BLPProjection", "EffectModifiers", "effect_modifiers"]

_METHODS = ("blp", "gates")
_ADJUST = ("holm", "none")
#: Relative tolerance under which a design column counts as a combination of
#: the columns before it (see ``linear_dependence`` in the Rust core).
_DEPENDENCE_TOL = 1e-10


#: What a covariate expression may be, as the formula parser reads it.
_SUPPORTED_TRANSFORMS = (
    "log(x), log1p(x), sqrt(x), exp(x), scale(x), center(x), ecdf(x), "
    "pmin(x, bound), pmax(x, bound) and I(arithmetic on columns)")
_PLAIN_NAME = re.compile(r"[A-Za-z_][\w.]*")
#: The gate passes when the out-of-fold gain over a constant effect exceeds its
#: standard error, the rule effect_model uses to call a candidate eligible.
_GATE_RULE = "out-of-fold R-loss gain over a constant effect exceeds its standard error"


def _expression_recipe(data, expression):
    """The fitted transform recipe of a covariate written as a formula transform,
    e.g. ``log(prior_earnings)``, and the derived column's name.

    The expression is read by the formula parser's own transform code
    (``mboostrs.transforms``), so the transforms are the ones a model formula
    accepts, and a fitted one (``scale``, ``center``, ``ecdf``) keeps the
    parameters of the fitting rows for new rows."""
    text = expression.strip()
    if not any(r.fullmatch(text) for r in (_TRANSFORM_RE, _CLAMP_RE, _IDENTITY_RE)):
        raise ValueError(
            f"covariate {expression!r} is not a column of the data and is not a supported "
            f"transform. The supported transforms are {_SUPPORTED_TRANSFORMS}, each applied "
            "to data columns, as in a model formula")
    try:
        _, recipe = extract_transforms(f"__covariate ~ {text}", data)
    except ValueError as error:
        raise ValueError(f"covariate {expression!r}: {error}") from error
    recipe = fit_recipe(recipe, data)
    return recipe, recipe.steps[-1].name


def _covariate_values(data, names, recipes=None):
    """Values of each covariate under the name as written, the data columns each
    one reads, and the transform recipe of each expression.

    A name that is a column of ``data`` is read as that column; any other name
    is a transform expression. ``recipes`` replays the recipes of an earlier
    call, so new rows are transformed with the fitted parameters."""
    recipes = dict(recipes or {})
    values, sources = {}, {}
    plain = [name for name in names if name not in recipes and _has_column(data, name)]
    if plain:
        values.update(_modifier_values(data, plain))
        sources.update({name: (name,) for name in plain})
    for name in names:
        if name in values:
            continue
        if name not in recipes:
            if _PLAIN_NAME.fullmatch(name):
                known = sorted(getattr(data, "columns", None) or list(data))
                raise ValueError(
                    f"covariates names {name!r}, which is not a column of the data; "
                    f"the columns are {known}")
            recipes[name] = _expression_recipe(data, name)[0]
        recipe = recipes[name]
        derived = apply_recipe(recipe, data)
        values[name] = _column(derived, recipe.steps[-1].name)
        sources[name] = tuple(recipe.source_names)
    return ({name: values[name] for name in names}, sources,
            {name: recipes[name] for name in names if name in recipes})


def _plan(name, values):
    """How a covariate enters the design: standardized numeric or level dummies.

    ``values`` are the covariate's values on the rows inside the overlap
    region. A numeric covariate is centred and scaled by its values there, so
    its coefficient is the change per standard deviation. A categorical one
    (an object array of labels) gets one dummy per level after the first."""
    if values.dtype == object:
        levels = tuple(sorted(set(values.tolist())))
        if len(levels) < 2:
            raise ValueError(
                f"covariate {name!r} has one level inside the overlap region, so it cannot "
                "modify the effect; drop it from covariates= or widen overlap_bounds=")
        return {"name": name, "kind": "categorical", "levels": levels}
    x = np.asarray(values, dtype=np.float64)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(
            f"covariate {name!r} is constant inside the overlap region, so it cannot modify "
            "the effect; drop it from covariates= or widen overlap_bounds=")
    return {"name": name, "kind": "numeric", "mean": float(x.mean()), "sd": sd}


def _block(plan, values):
    """The design columns of one covariate, as an ``(n, k)`` array with ``k >= 1``."""
    if plan["kind"] == "numeric":
        x = np.asarray(values, dtype=np.float64)
        return ((x - plan["mean"]) / plan["sd"])[:, None]
    labels = np.asarray(values, dtype=object)
    unknown = sorted({str(v) for v in labels.tolist()} - set(plan["levels"]))
    if unknown:
        raise ValueError(
            f"covariate {plan['name']!r} has levels {unknown} that the projection was not fit "
            f"on (fitted levels: {list(plan['levels'])}); recode them to a fitted level first")
    return np.column_stack([(labels == level).astype(np.float64)
                            for level in plan["levels"][1:]])


def _block_labels(plan):
    if plan["kind"] == "numeric":
        return [plan["name"]]
    return [f"{plan['name']}[{level}]" for level in plan["levels"][1:]]


def _design(plans, values):
    """Intercept, then the blocks of ``plans``.

    Returns the ``(n, p)`` design, the design columns of each covariate, and a
    label for every column."""
    n = len(values[plans[0]["name"]])
    columns = [np.ones((n, 1))]
    labels = ["intercept"]
    spans = {}
    width = 1
    for plan in plans:
        block = _block(plan, values[plan["name"]])
        spans[plan["name"]] = list(range(width, width + block.shape[1]))
        width += block.shape[1]
        columns.append(block)
        labels.extend(_block_labels(plan))
    design = np.ascontiguousarray(np.hstack(columns), dtype=np.float64)
    return design, spans, labels


def _independent(design, labels):
    """Columns to keep, and the dropped columns with the kept column they repeat.

    Two covariates can be exactly collinear in a dataset: in the CPS extract
    potential experience is age minus education minus six, so the three are
    linearly dependent and a regression on all of them is singular. The
    dependent column is dropped with a warning, and the fit uses the rest."""
    p = design.shape[1]
    flat = np.asfortranarray(design).ravel(order="F")
    hits = _rust.linear_dependence(flat, design.shape[0], p, _DEPENDENCE_TOL)
    dropped = {int(pos): int(source[0][0]) for pos, source in hits if source}
    kept = [j for j in range(p) if j not in dropped]
    notes = [(labels[pos], labels[src]) for pos, src in dropped.items()]
    if notes:
        described = "; ".join(f"{a} is a combination of {b}" for a, b in notes)
        warnings.warn(
            f"the joint regression drops {len(notes)} collinear column(s): {described}. "
            "Remove one of the collinear covariates from covariates= to keep its coefficient.",
            UserWarning, stacklevel=3)
    return kept, dropped


def _fit(psi, weights, design):
    """Weighted least squares with HC1 covariance, from the Rust core."""
    p = design.shape[1]
    coef, se, cov, n_used = _rust.effect_robust_wls(psi, design.ravel(), p, weights)
    return np.asarray(coef), np.asarray(se), np.asarray(cov).reshape(p, p), n_used


def _wald(coef, cov, indices):
    """Joint chi-squared test and p-value of the coefficients at ``indices``."""
    if not indices:
        return float("nan"), float("nan")
    chi2, pval = _rust.effect_wald_test(coef, cov.ravel(), coef.size, list(indices))
    return float(chi2), float(pval)


def _gates_codes(plan, values, groups):
    """Group codes: quantile groups for a numeric covariate, level codes otherwise."""
    if plan["kind"] == "categorical":
        index = {level: i for i, level in enumerate(plan["levels"])}
        codes = [index[v] for v in np.asarray(values, dtype=object).tolist()]
        return codes, len(plan["levels"])
    x = np.asarray(values, dtype=np.float64)
    probs = [k / groups for k in range(1, groups)]
    cuts = np.asarray(_rust.declaration_quantiles(x, probs), dtype=np.float64)
    return np.searchsorted(cuts, x, side="right").astype(int).tolist(), groups


def _group_name(plan, n_groups, index):
    """The label of a GATES group: its level for a categorical covariate, else its
    quantile group counted from the bottom."""
    if plan["kind"] == "categorical":
        return plan["levels"][index]
    return f"group {index + 1} of {n_groups}"


def _run_gate(result, data, names, sources, seed):
    """Whether a cross-fitted R-learner on the tested covariates beats a constant
    effect out of fold.

    The second stage is the smooth candidate of :func:`mboostrs.effect_model`, fit
    on the pseudo-outcome built from the nuisances already cross-fitted in
    ``result``. The tested columns are the data columns the covariates read."""
    from mboostrs.causal import effect_model

    spec = result.spec
    columns = tuple(dict.fromkeys(c for name in names for c in sources[name]))
    outside = [c for c in columns if c not in spec.covariates]
    if outside:
        raise ValueError(
            f"the R-learner gate fits the effect on columns of result.spec.covariates, and "
            f"{outside} are not among them. Pass covariates= from {list(spec.covariates)}, "
            "or gate=False to test them on the score regressions alone")
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = effect_model(
            data, spec, nuisance=result, modifiers=columns, terms="smooth", seed=seed,
            interval_method="refit",  # the gate reads the out-of-fold gain, not intervals
            outcome_missing=result.nuisance.get("outcome_missing", "error"))
    seconds = time.perf_counter() - start
    for w in caught:
        # The constant-effect warning is the gate's own result, recorded below.
        if "does not beat a constant effect" not in str(w.message):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    selection = model.diagnostics["terms_selection"]
    gain = float(selection["gain_over_constant"]["smooth"])
    gain_se = float(selection["gain_se"]["smooth"])
    passed = bool(np.isfinite(gain) and np.isfinite(gain_se) and gain > gain_se)
    record = {"run": True, "passed": passed, "gain": gain, "gain_se": gain_se,
              "rule": _GATE_RULE, "terms": "smooth", "modifiers": columns,
              "nuisance": "the aipw result", "seed": seed, "seconds": seconds}
    message = ""
    if not passed:
        message = (
            f"No modifier is claimed: a cross-fitted R-learner smooth effect on "
            f"{list(columns)} does not beat a constant effect out of fold (gain {gain:.6g}, "
            f"standard error {gain_se:.6g}; the gate requires the gain to exceed its "
            "standard error). The score tests are reported in the table but none is a claim. "
            "Pass gate=False to decide on the score tests alone.")
    return record, message


def _critical(level):
    return NormalDist().inv_cdf(1 - (1 - level) / 2)


@dataclass(frozen=True)
class BLPProjection:
    """A linear projection of the effect on covariates, usable as a ranking source.

    The projection is the best linear predictor of the AIPW score on the
    covariates it was fit with: ``predict(newdata)`` returns the fitted value
    ``a + sum_j b_j z_j(x)``, where ``z_j`` is a covariate standardized over the
    overlap region of the fit (or the dummy of a categorical level). Its values
    are estimated effects in outcome units at each row. Passing it to
    :func:`mboostrs.rank_by_effect` ranks the rows inside the overlap region by
    this estimate.
    """

    covariates: tuple
    plans: tuple = field(repr=False)
    coefficients: tuple = field(repr=False)
    #: Design columns used by the fit, as positions in the full design.
    kept: tuple = field(repr=False)
    labels: tuple
    n: int
    level: float
    #: Transform recipes of the covariates written as expressions, by name.
    recipes: dict = field(default_factory=dict, repr=False)

    def predict(self, newdata):
        """The projected effect of each row of ``newdata``, one value per row.

        A covariate written as a transform (``"log(prior_earnings)"``) is
        computed from the columns of ``newdata`` with the parameters of the
        fitting rows."""
        values, _, _ = _covariate_values(newdata, self.covariates, self.recipes)
        design, _, _ = _design(self.plans, values)
        design = np.ascontiguousarray(design[:, list(self.kept)])
        coef = np.asarray(self.coefficients, dtype=np.float64)
        return np.asarray(_rust.effect_linear_predict(
            design.ravel(), design.shape[1], coef), dtype=np.float64)

    def __repr__(self) -> str:
        return f"BLPProjection(covariates={list(self.covariates)}, n={self.n})"


@dataclass(frozen=True)
class EffectModifiers:
    """Per-covariate tests of effect modification, with a joint projection.

    ``records`` holds one entry per covariate, in the order given:

    * ``blp_slope``, ``blp_se``, ``blp_lower``, ``blp_upper``: the coefficient
      of the covariate in a regression of the AIPW score on that covariate
      alone, with HC1 standard errors. For a numeric covariate it is the change
      in the effect, in outcome units, per standard deviation of the covariate;
      for a categorical covariate it is ``None`` and ``blp_p`` is the joint
      test over the level dummies.
    * ``gates_contrast``, ``gates_se``, ``gates_lower``, ``gates_upper``: the
      difference between the mean score of the top and bottom groups. For a
      numeric covariate the groups are quantile groups in increasing order, so
      the contrast is top quantile group minus bottom quantile group; for a
      categorical covariate the contrast is the largest group mean minus the
      smallest, and ``gates_p`` is the chi-squared test that all level means
      are equal.
    * ``blp_levels``: for a categorical covariate, the BLP coefficient of each
      level relative to the first level (in sorted order), with its standard
      error; empty for a numeric covariate.
    * ``blp_p``, ``gates_p``: two-sided p-values of the two tests, and
      ``blp_p_adjusted``, ``gates_p_adjusted``: the same p-values adjusted across
      covariates with the method given to :func:`effect_modifiers`.
    * ``estimate``: the point estimate of the method's test (the BLP slope or
      the GATES contrast); ``direction`` is its sign (``"increases"`` or
      ``"decreases"`` along a numeric covariate, ``"differs by level"`` for a
      categorical one). ``gates_top`` and ``gates_bottom`` name the groups of
      the contrast.
    * ``test_decision``: True when the adjusted p-value of the method's test is
      below ``1 - level``.
    * ``decision``: ``test_decision`` when the gate passed or was not run, and False
      for every covariate when the gate failed.

    ``diagnostics["gate"]`` records the gate: whether it ran, ``passed``, the
    out-of-fold ``gain`` over a constant effect and its standard error
    ``gain_se`` (in units of the weighted squared R-loss), the modifiers it
    used, the seconds it took and the rule. ``message`` states the reason when
    it failed.
    """

    method: str
    level: float
    adjust: str
    bounds: tuple
    n: int
    n_inside: int
    groups: int
    records: tuple = field(repr=False)
    joint: tuple = field(repr=False)
    _core: dict = field(repr=False, compare=False)
    #: The gate record (see :func:`effect_modifiers`) and any other diagnostics.
    diagnostics: dict = field(default_factory=dict, repr=False)
    #: A sentence stating why no modifier is claimed when the gate failed; empty otherwise.
    message: str = ""

    def table(self, output="auto"):
        """One row per covariate; see the ``records`` description."""
        return output_frame(list(self.records), output)

    def joint_table(self, output="auto"):
        """One row per design column of the joint regression on all covariates.

        ``block_p`` is the joint test of the covariate's columns: the same as
        ``p`` for a numeric covariate and the test over all level dummies for a
        categorical one. A column dropped as collinear has NaN coefficients and
        ``collinear`` set to True."""
        return output_frame(list(self.joint), output)

    def selected(self):
        """The covariates whose decision is True, in the order given."""
        return tuple(r["covariate"] for r in self.records if r["decision"])

    def projection(self, covariates="all"):
        """The joint projection on ``"all"`` covariates, on the ``"passing"`` ones
        (those with a True decision), or on an explicit list of names.

        A projection needs at least one covariate; if none passes, pass
        ``covariates="all"`` or name the covariates explicitly."""
        if isinstance(covariates, str):
            if covariates == "all":
                names = tuple(r["covariate"] for r in self.records)
            elif covariates == "passing":
                names = self.selected()
                if not names:
                    raise ValueError(
                        "no covariate has a True decision, so a projection on the passing "
                        "covariates would be constant; use projection('all') or name "
                        "covariates explicitly")
            else:
                raise ValueError(
                    "covariates must be 'all', 'passing' or a list of names; "
                    f"got {covariates!r}")
        else:
            names = tuple(covariates)
            known = {r["covariate"] for r in self.records}
            missing = [c for c in names if c not in known]
            if missing or not names:
                raise ValueError(
                    f"covariates {missing or 'given as an empty list'} were not tested by this "
                    f"result; pass one or more names from {sorted(known)}")
        core = self._core
        plans = tuple(core["plans"][name] for name in names)
        values = {name: core["values"][name] for name in names}
        design, _, labels = _design(plans, values)
        kept, _ = _independent(design, labels)
        coef, _, _, _ = _fit(core["psi"], core["weights"], design[:, kept])
        return BLPProjection(
            covariates=names, plans=plans, coefficients=tuple(float(c) for c in coef),
            kept=tuple(kept), labels=tuple(labels[j] for j in kept),
            n=int(core["psi"].size), level=self.level,
            recipes={name: core["recipes"][name] for name in names if name in core["recipes"]})

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the tests and the joint regression."""
        return export_result(self, include_arrays=include_arrays, arrays=("records", "joint"),
                             extras={"selected": list(self.selected())})

    def __repr__(self) -> str:
        return (f"EffectModifiers(method={self.method!r}, n_inside={self.n_inside}, "
                f"selected={list(self.selected())})")

    def describe(self) -> str:
        """Each covariate's point estimate, adjusted p-value and decision."""
        lower, upper = self.bounds
        gate = self.diagnostics.get("gate", {})
        text = (f"{self.message} " if self.message else "") + (
            f"Effect modifier tests on {self.n_inside} of {self.n} rows with propensity in "
                f"[{lower:g}, {upper:g}]; decisions by the {self.method} test at "
                f"{self.level:.0%} level, {self.adjust} adjustment across covariates. "
                "Each test asks whether the average effect changes along a covariate "
                "(the direction of the change, from the BLP slope or the top-minus-bottom "
                "GATES contrast). ")
        if gate.get("run"):
            text += (f"R-learner gate {'passed' if gate['passed'] else 'failed'}: gain over a "
                     f"constant effect {gate['gain']:.6g}, standard error {gate['gain_se']:.6g}. ")
        else:
            text += "No R-learner gate was run, so the decisions rest on the score tests alone. "
        for r in self.records:
            if r["kind"] == "numeric":
                estimate = (f"BLP slope {r['blp_slope']:.6g} per SD, "
                            f"GATES {r['gates_contrast']:.6g}")
            else:
                estimate = f"level dummies joint p {r['blp_p']:.3g}"
            text += (f"{r['covariate']}: {estimate}, adjusted p {r['adjusted_p']:.3g}, "
                     f"{'claimed' if r['decision'] else 'not claimed'}. ")
        return text.rstrip()


def effect_modifiers(result, data, *, covariates=None, method="blp", groups=4, level=0.9,
                     overlap_bounds=(0.02, 0.98), adjust="holm", gate=True, gate_seed=42):
    """Test whether the effect changes along each covariate, using the AIPW scores.

    Rows are kept when their cross-fitted propensity lies inside
    ``overlap_bounds``. For each covariate the function fits two tests on those
    rows: a best linear predictor (BLP) slope, the regression of the score on
    the covariate (standardized for a numeric covariate), and GATES, the mean
    score in ``groups`` quantile groups of the covariate with the top-minus-bottom
    contrast. Both use heteroskedasticity-robust (HC1) standard errors. A joint
    regression on all covariates is also fit; its projection ranks rows (see
    :meth:`EffectModifiers.projection` and :func:`mboostrs.rank_by_effect`).

    Parameters
    ----------
    result : AIPWResult
        The result of :func:`mboostrs.aipw`. Its per-row scores, cross-fitted
        propensities and observation weights (when present) are used.
    data : mapping or data frame
        The rows the result was fit on, with the covariates as columns.
    covariates : sequence of str, optional
        The covariates to test. Defaults to the covariates of ``result.spec``. A
        name that is not a column of ``data`` is read as a transform expression
        with the transforms of the model formulas: ``log(x)``, ``log1p(x)``,
        ``sqrt(x)``, ``exp(x)``, ``scale(x)``, ``center(x)``, ``ecdf(x)``,
        ``pmin(x, bound)``, ``pmax(x, bound)`` and ``I(...)`` arithmetic on
        columns. ``covariates=["log(prior_earnings)"]`` tests the effect along
        log prior earnings. Reports name the expression as written. A fitted
        transform (``scale``, ``center``, ``ecdf``) keeps the parameters of
        the fitting rows when a projection scores new rows. An unsupported
        expression is an error that lists the supported transforms.
    method : {"blp", "gates"}
        Which test decides ``decision``, ``direction`` and ``estimate``. Both
        tests are always reported.
    groups : int
        Number of quantile groups for GATES on a numeric covariate.
    level : float
        Confidence level of the intervals. The decision threshold is
        ``1 - level`` on the adjusted p-value.
    overlap_bounds : tuple
        ``(lower, upper)`` propensity bounds of the rows used.
    adjust : {"holm", "none"}
        Multiplicity adjustment across the covariates tested by the call. Holm
        controls the family-wise error rate over that set.
    gate : bool
        With ``True`` (the default) the function first runs an R-learner gate: it
        fits the smooth second stage of :func:`mboostrs.effect_model` on the
        tested covariates, using the cross-fitted nuisances of ``result`` and
        folds stratified by treatment, and compares its out-of-fold R-loss with
        that of a constant effect. The gate passes when the gain exceeds its
        standard error. When it fails, every ``decision`` is False,
        ``selected()`` is empty and ``message`` states the reason; the score
        tests are still reported, with their own result in ``test_decision``.
        The gate's gain, standard error and outcome are in
        ``diagnostics["gate"]``. The gate needs the treatment and outcome
        columns in ``data`` and covariates that are in ``result.spec.covariates``
        (the columns an expression reads). ``gate=False`` skips it and decides
        on the score tests alone. The score regressions
        treat the scores as fixed and so omit the error of the nuisance fits; on
        data with no effect modification they can report a modifier, and the
        gate removes that claim. Cost: on a 2,000-row simulation with five
        covariates, after :func:`mboostrs.aipw` with its defaults, the
        gate took 0.28 s of CPU time (median of five draws; about 1 s of wall time on a
        loaded machine), against 0.34 s for the ``aipw`` call and 0.001 s for the score
        tests. It reuses the nuisances of ``result`` and fits one penalized second stage
        on five folds, without the second-stage bootstrap that ``effect_model`` runs for
        intervals. On 40 simulated draws with no effect (2,000 rows, five covariates) the
        score tests alone claimed a modifier in 4 and the gate in none; on 20 draws with an effect that
        rises with one covariate the gate passed in all 20 and kept that covariate.
        The gate has the low power of the R-loss comparison, so a real but
        small modifier can fail it.
    gate_seed : int
        Seed of the gate's folds.

    Returns
    -------
    EffectModifiers

    Notes
    -----
    The regressions use the observation weights of the result, as the estimate
    is a weighted mean of the scores. The standard errors treat the scores as
    fixed: they do not include the variability of the nuisance fits. The
    covariates are not chosen from these results, so a covariate selected after
    looking at them does not carry the nominal error rate.
    """
    if not isinstance(result, AIPWResult):
        raise TypeError("effect_modifiers needs an AIPWResult from mboostrs.aipw")
    if method not in _METHODS:
        raise ValueError(f"method must be 'blp' or 'gates'; got {method!r}")
    if adjust not in _ADJUST:
        raise ValueError(f"adjust must be 'holm' or 'none'; got {adjust!r}")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 2:
        raise ValueError(f"groups must be an integer of at least 2; got {groups!r}")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1); got {level!r}")
    lower, upper = _check_bounds(overlap_bounds)
    names = tuple(result.spec.covariates if covariates is None else covariates)
    if not names:
        raise ValueError("effect_modifiers needs at least one covariate")
    if len(set(names)) != len(names):
        raise ValueError("covariates must not repeat a name")
    all_values, sources, recipes = _covariate_values(data, names)
    n = int(np.asarray(result.scores).size)
    for name, column in all_values.items():
        if len(column) != n:
            raise ValueError(
                f"covariate {name!r} has {len(column)} rows but the result has {n}; pass the "
                "data the result was fit on")
    propensity = np.asarray(result.propensity, dtype=np.float64)
    inside = np.isfinite(propensity) & (propensity >= lower) & (propensity <= upper)
    n_inside = int(inside.sum())
    if n_inside < 2 * (len(names) + 1):
        raise ValueError(
            f"only {n_inside} rows lie inside overlap_bounds={overlap_bounds}; too few for "
            "the regression. Widen overlap_bounds or check the propensity model")
    psi = np.asarray(result.scores, dtype=np.float64)[inside]
    weights = (np.ones(n_inside) if result.weights is None
               else np.asarray(result.weights, dtype=np.float64)[inside])
    values = {name: col[inside] for name, col in all_values.items()}
    plans = {name: _plan(name, values[name]) for name in names}

    # One covariate at a time: BLP slope (or joint dummy test) and GATES.
    single = []
    for name in names:
        plan = plans[name]
        design, spans, _ = _design([plan], values)
        coef, se, cov, _ = _fit(psi, weights, design)
        idx = spans[name]
        blp_chi2, blp_p = _wald(coef, cov, idx)
        if plan["kind"] == "numeric":
            slope, slope_se = float(coef[idx[0]]), float(se[idx[0]])
            level_effects = ()
        else:
            slope, slope_se = None, None
            level_effects = tuple(
                {"level": level, "coefficient": float(coef[col]), "se": float(se[col])}
                for level, col in zip(plan["levels"][1:], idx, strict=True))
        codes, n_groups = _gates_codes(plan, values[name], groups)
        means, _, counts, ordered_c, extreme_c, omnibus = _rust.effect_gates(
            psi, weights, codes, n_groups)
        if plan["kind"] == "numeric":
            contrast, contrast_se, gates_p = ordered_c[0], ordered_c[1], ordered_c[2]
            top, bottom = ordered_c[3], ordered_c[4]
        else:
            contrast, contrast_se, gates_p = extreme_c[0], extreme_c[1], omnibus[2]
            top, bottom = extreme_c[3], extreme_c[4]
        single.append({
            "name": name, "categorical": plan["kind"] == "categorical",
            "levels": plan.get("levels"), "n_groups": n_groups,
            "blp_slope": slope, "blp_se": slope_se, "blp_p": blp_p,
            "blp_levels": level_effects,
            "gates_contrast": float(contrast), "gates_se": float(contrast_se),
            "gates_p": float(gates_p),
            "gates_top": _group_name(plan, n_groups, top),
            "gates_bottom": _group_name(plan, n_groups, bottom),
            "gates_means": tuple(float(m) for m in means),
            "gates_counts": tuple(int(c) for c in counts),
        })
    p_blp = np.array([s["blp_p"] for s in single])
    p_gates = np.array([s["gates_p"] for s in single])
    if adjust == "holm":
        adj_blp = np.asarray(_rust.effect_holm_adjust(p_blp))
        adj_gates = np.asarray(_rust.effect_holm_adjust(p_gates))
    else:
        adj_blp, adj_gates = p_blp.copy(), p_gates.copy()
    alpha = 1 - level
    z = _critical(level)

    # Joint regression on all covariates: the joint table and the projection.
    joint_plans = [plans[name] for name in names]
    design, spans, labels = _design(joint_plans, values)
    kept, dropped = _independent(design, labels)
    coef_k, se_k, cov_k, _ = _fit(psi, weights, design[:, kept])
    position = {col: i for i, col in enumerate(kept)}
    joint_rows = []
    for plan in joint_plans:
        name = plan["name"]
        block_kept = [position[c] for c in spans[name] if c in position]
        block_p = _wald(coef_k, cov_k, block_kept)[1]
        for col in spans[name]:
            if col in position:
                i = position[col]
                col_p = _wald(coef_k, cov_k, [i])[1]
                row = {"covariate": name, "column": labels[col],
                       "coefficient": float(coef_k[i]), "se": float(se_k[i]),
                       "p": col_p, "block_p": block_p, "collinear": False}
            else:
                row = {"covariate": name, "column": labels[col],
                       "coefficient": float("nan"), "se": float("nan"),
                       "p": float("nan"), "block_p": block_p, "collinear": True}
            joint_rows.append(row)

    records = []
    for s, pb, pg, ab, ag in zip(single, p_blp, p_gates, adj_blp, adj_gates, strict=True):
        has_blp = s["blp_slope"] is not None
        if method == "blp":
            point, decision_p = s["blp_slope"], ab
        else:
            point, decision_p = s["gates_contrast"], ag
        if s["categorical"]:
            direction = "differs by level"
        elif point == 0:
            direction = "no change"
        else:
            direction = "increases" if point > 0 else "decreases"
        records.append({
            "covariate": s["name"],
            "kind": "categorical" if s["categorical"] else "numeric",
            "method": method,
            "blp_slope": s["blp_slope"],
            "blp_se": s["blp_se"],
            "blp_lower": s["blp_slope"] - z * s["blp_se"] if has_blp else None,
            "blp_upper": s["blp_slope"] + z * s["blp_se"] if has_blp else None,
            "blp_levels": s["blp_levels"],
            "blp_p": float(pb),
            "blp_p_adjusted": float(ab),
            "gates_contrast": s["gates_contrast"],
            "gates_se": s["gates_se"],
            "gates_lower": s["gates_contrast"] - z * s["gates_se"],
            "gates_upper": s["gates_contrast"] + z * s["gates_se"],
            "gates_p": float(pg),
            "gates_p_adjusted": float(ag),
            "gates_top": s["gates_top"],
            "gates_bottom": s["gates_bottom"],
            "gates_means": s["gates_means"],
            "gates_counts": s["gates_counts"],
            "estimate": float(point) if point is not None else None,
            "direction": direction,
            "p_value": float(pb if method == "blp" else pg),
            "adjusted_p": float(decision_p),
            "decision": bool(decision_p < alpha),
        })
    for r in records:
        r["test_decision"] = r["decision"]
    if gate:
        gate_record, message = _run_gate(result, data, names, sources, gate_seed)
        if not gate_record["passed"]:
            for r in records:
                r["decision"] = False
    else:
        gate_record, message = {"run": False}, ""
    return EffectModifiers(
        method=method, level=level, adjust=adjust, bounds=(lower, upper), n=n,
        n_inside=n_inside, groups=groups, records=tuple(records), joint=tuple(joint_rows),
        _core={"psi": psi, "weights": weights, "values": values, "plans": plans,
               "recipes": recipes},
        diagnostics={"gate": gate_record}, message=message,
    )
