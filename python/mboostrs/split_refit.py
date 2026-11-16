"""Sample-split selection followed by a REML refit on the other half of the rows.

After boosting has chosen terms, an interval for a chosen effect that is
computed on the same rows is conditional on that choice, and it is too
narrow. :func:`split_refit` removes the dependence in the simplest valid way:
the rows are split at random, one half chooses the terms, and the other half,
which played no part in the choice, estimates the chosen terms by REML
(:func:`mboostrs.gam_refit` for a ``gamboost`` formula,
:func:`mboostrs.gamlss_refit` for a mapping of distribution parameters). The
intervals and tests of the refit are then valid for the chosen terms, for a
fixed split. The price is that each refit sees only ``fraction`` of the rows.

Each repeat draws its own split, so one split's choice is a single draw. With
``repeats > 1`` the per-split results are aggregated by
:func:`meinshausen_buhlmann_p` and by medians (see :func:`split_refit`).

Not covered: selective inference, which conditions the sampling distribution on
the selection event itself (the polyhedral approach of Lee et al., 2016, applied
to component-wise L2 boosting). That is a separate method, and this module does
not implement it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mboostrs._output import json_safe

__all__ = ["SplitContrast", "SplitFit", "SplitRefitResult", "SplitTerm", "meinshausen_buhlmann_p",
           "split_refit"]

#: Folds of the cross-validation that tunes the boosted fit on the selection half.
_FOLDS = 5
#: Error control of `stabsel` when `select="stabsel"`: a selection frequency
#: cutoff of 0.75, and the expected number of false learners held to 1. At a
#: cutoff of 0.6 with PFER = 1, a formula with two learners admits no
#: selection size q under the unimodal bound, and `stabsel` refuses the call.
_STABSEL_CUTOFF = 0.75
_STABSEL_PFER = 1.0
#: Step budget of the distributional boosted fit before its cross-validation.
_LSS_GRID = 400
#: Coefficient draws behind each distributional contrast interval.
_DRAWS = 1000
_SELECT_NAMES = ("deselect", "stabsel")


@dataclass(frozen=True)
class SplitContrast:
    """The average contrast of one variable in one split, with its interval."""

    variable: str
    reference: object
    alternative: object
    #: False when no term the split's refit kept reads the variable; the
    #: contrast is then exactly zero with a zero-width interval.
    selected: bool
    estimate: float
    lower: float
    upper: float
    #: Two-sided p-value of "contrast = 0" in this split alone.
    p_value: float


@dataclass(frozen=True)
class SplitFit:
    """One repeat: the rows, the terms the selection half chose, and the
    inference half's refit results."""

    repeat: int
    n_selection: int
    n_inference: int
    #: The chosen terms, labelled by the refit (``"bbs(x1, df=4)"``, or
    #: ``"mu: bbs(x1, df=4)"`` for a distributional model).
    selected: tuple
    #: Each chosen term's p-value for "its effect is zero" (Wood, 2013),
    #: keyed by label; empty for a distributional refit, which reports no term tests.
    term_p_values: dict
    contrasts: tuple


@dataclass(frozen=True)
class SplitTerm:
    """One term across repeats."""

    term: str
    #: Fraction of the repeats in which the selection half chose the term.
    selection_frequency: float
    #: One repeat: that split's p-value. Several repeats: the Meinshausen,
    #: Meier and Buhlmann (2009) aggregate of the p-values, with a split that
    #: did not choose the term counted as p = 1. None when the refit reports
    #: no term tests.
    p_value: float | None


@dataclass(frozen=True)
class ContrastSummary:
    """One requested contrast across repeats.

    The estimate and bounds are medians over repeats of the per-split values.
    The p-value is the per-split one for a single repeat, and the
    Meinshausen, Meier and Buhlmann (2009) aggregate otherwise.
    """

    variable: str
    reference: object
    alternative: object
    #: Fraction of the repeats in which the variable was read by a kept term.
    selection_frequency: float
    estimate: float
    lower: float
    upper: float
    p_value: float


@dataclass(frozen=True)
class SplitRefitResult:
    """The output of :func:`split_refit`: the aggregate, and the per-repeat records.

    ``terms`` lists every term chosen in at least one repeat. ``contrasts``
    holds one :class:`ContrastSummary` per requested contrast. ``splits``
    holds one :class:`SplitFit` per repeat.
    """

    formula: object
    family: str
    select: str
    repeats: int
    fraction: float
    level: float
    seed: int
    #: "link" for a ``gamboost`` refit; for a distributional refit, the
    #: response-scale mean of the first distribution parameter.
    scale: str
    terms: tuple
    contrasts: tuple
    splits: tuple = field(repr=False, default=())

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record. ``include_arrays=True`` adds one entry per repeat
        under ``"splits"`` (the rows used, the terms chosen, the term p-values and
        each contrast's per-split interval); by default only the aggregate is
        returned."""
        record = {
            "formula": json_safe(self.formula),
            "family": self.family,
            "select": self.select,
            "repeats": self.repeats,
            "fraction": self.fraction,
            "level": self.level,
            "seed": self.seed,
            "scale": self.scale,
            "terms": [json_safe(t.__dict__) for t in self.terms],
            "contrasts": [json_safe(c.__dict__) for c in self.contrasts],
        }
        if include_arrays:
            record["splits"] = [
                {"repeat": s.repeat, "n_selection": s.n_selection, "n_inference": s.n_inference,
                 "selected": list(s.selected), "term_p_values": json_safe(s.term_p_values),
                 "contrasts": [json_safe(c.__dict__) for c in s.contrasts]}
                for s in self.splits]
        return record

    def summary(self):
        """Selection frequencies and contrast intervals, one line each."""
        pct = round(100 * self.level)
        lines = [
            f"Sample-split refit of {self.formula!r}: {self.family}, select={self.select!r}, "
            f"{self.repeats} repeat(s), {round(100 * self.fraction)}% of the rows for "
            f"selection, {pct}% intervals on the {self.scale} scale.",
        ]
        if self.terms:
            lines.append("Terms (selection frequency, p-value):")
            for t in self.terms:
                p = "n/a" if t.p_value is None else f"{t.p_value:.3g}"
                lines.append(f"  {t.term}: {t.selection_frequency:.2f}, p = {p}")
        if self.contrasts:
            lines.append("Average contrasts (median over repeats):")
            for c in self.contrasts:
                lines.append(
                    f"  {c.variable}: {c.alternative} vs {c.reference} = {c.estimate:.4g} "
                    f"[{c.lower:.4g}, {c.upper:.4g}], read by a kept term in "
                    f"{c.selection_frequency:.2f} of repeats, p = {c.p_value:.3g}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"SplitRefitResult(select={self.select!r}, repeats={self.repeats}, "
                f"terms={len(self.terms)}, contrasts={len(self.contrasts)})")


def meinshausen_buhlmann_p(p_values, gamma=0.5):
    """Aggregate per-split p-values by the quantile rule of Meinshausen, Meier
    and Bühlmann (2009, *JRSS-B* 71, "p-values for high-dimensional
    regression").

    The p-values from the splits are divided by ``gamma``, their empirical
    ``gamma``-quantile is taken, and the result is capped at 1. For the
    default ``gamma=0.5`` this is twice the median p-value, capped at 1. A
    single repeat is not aggregated: :func:`split_refit` reports its p-value
    as it is.
    """
    from mboostrs import _rust

    return _rust.aggregate_split_p(np.asarray(p_values, dtype=float).ravel(), gamma)


def _split_rows(n, fraction, rng):
    """Positions of the selection half and the inference half, from one random
    permutation: disjoint, and together every row."""
    order = rng.permutation(n)
    n_selection = int(round(fraction * n))
    return np.sort(order[:n_selection]), np.sort(order[n_selection:])


def _two_sided_p(estimate, lower, upper, level):
    """Normal p-value of "effect = 0" from an interval at `level`."""
    from mboostrs import _rust

    return _rust.normal_interval_p(estimate, lower, upper, level)


def _draw_p(draws):
    """Bootstrap-style two-sided p-value of "effect = 0" from posterior draws."""
    from mboostrs import _rust

    return _rust.draw_p(np.asarray(draws, dtype=float).ravel())


def _term_names(formula, data):
    """The additive terms of a formula, as written (no `1`/`0`)."""
    from mboostrs.selection import _terms

    _, all_terms = _terms(formula, data)
    return [t.strip() for s, t in all_terms if s == "+" and t.strip() not in ("1", "0")]


def _gam_selection(formula, sel_data, family, select, seed):
    """The formula terms the selection half keeps, in formula order."""
    import mboostrs as mb
    from mboostrs.selection import _term_learners, _terms

    if callable(select):
        return _validated(select(sel_data), _term_names(formula, sel_data), "select")
    fit = mb.tune_mstop(formula, sel_data, family, folds=_FOLDS, seed=seed).model
    if select == "deselect":
        try:
            return list(mb.deselect(fit, folds=_FOLDS, seed=seed).kept)
        except ValueError as error:
            if "no learner reaches" in str(error):
                return []
            raise
    _, all_terms = _terms(formula, sel_data)
    mapping = _term_learners(fit, all_terms)
    result = mb.stabsel(fit, cutoff=_STABSEL_CUTOFF, PFER=_STABSEL_PFER, verbose=False)
    chosen = {result.learner_names[i] for i in result.selected}
    return [term for term, names in mapping.items() if chosen.intersection(names)]


def _lss_selection(mapping, sel_data, family, select, seed):
    """`{parameter: [terms]}` the selection half keeps."""
    import mboostrs as mb
    from mboostrs._refit import _row_count

    params = list(family.param_names)
    allowed = {p: _term_names(mapping[p], sel_data) for p in params}
    if callable(select):
        out = select(sel_data)
        pairs = (out.items() if isinstance(out, dict)
                 else [(p, t) for p, t in out])
        chosen = {p: [] for p in params}
        for parameter, term in pairs:
            if parameter not in chosen:
                raise ValueError(f"select returned parameter {parameter!r}; the family's "
                                 f"parameters are {params}")
            chosen[parameter].append(term)
        return {p: _validated(chosen[p], allowed[p], f"select ({p})") for p in params}
    fit = mb.gamboostLSS(mapping, sel_data, family=family, method="noncyclic", mstop=_LSS_GRID)
    cv = fit.cvrisk(folds=mb.cv_weights(_row_count(sel_data), "kfold", _FOLDS, seed=seed))
    total = max(int(cv.optimal_mstop[0]), 1)
    try:
        kept = mb.deselect(fit.with_mstop(total), folds=_FOLDS, seed=seed).kept
    except ValueError as error:
        if "no learner reaches" in str(error):
            return {p: [] for p in params}
        raise
    out = {p: [] for p in params}
    for parameter, term in kept:
        out[parameter].append(term)
    return out


def _validated(terms, allowed, what):
    terms = [str(t).strip() for t in terms]
    unknown = [t for t in terms if t not in allowed]
    if unknown:
        raise ValueError(f"{what} returned {unknown}, which are not terms of the formula "
                         f"({allowed}); return formula terms as written")
    return [t for t in allowed if t in terms]


def _refit_gam(formula, inf_data, family, chosen):
    """A gam_refit of the chosen terms on the inference half.

    The model is a one-step boosted fit: it carries the inference rows and the
    formula, which `gam_refit` reads. Its boosted coefficients are not used.
    """
    import mboostrs as mb

    model = mb.gamboost(formula, inf_data, family=family, control=mb.boost_control(mstop=1))
    return mb.gam_refit(model, terms=chosen)


def _refit_lss(mapping, inf_data, family, chosen):
    """A gamlss_refit of the chosen terms on the inference half (see `_refit_gam`)."""
    import mboostrs as mb

    model = mb.gamboostLSS(mapping, inf_data, family=family, mstop=1)
    return mb.gamlss_refit(model, terms=chosen)


def _gam_split(refit, inf_data, contrasts, level):
    from mboostrs.formula import term_variables
    from mboostrs.selection import _average_contrast

    out = []
    for variable, reference, alternative in contrasts:
        read = any(variable in term_variables(t) for t in refit.terms)
        estimate, lower, upper = _average_contrast(refit, inf_data, variable, reference,
                                                   alternative, level)
        out.append(SplitContrast(variable, reference, alternative, read, estimate, lower, upper,
                                 _two_sided_p(estimate, lower, upper, level)))
    return tuple(out)


def _lss_split(refit, inf_data, contrasts, level, seed, chosen_location):
    from mboostrs import _rust
    from mboostrs.formula import term_variables
    from mboostrs.selection import _with_column

    location = refit.parameters[0]
    out = []
    for variable, reference, alternative in contrasts:
        read = any(variable in term_variables(t) for t in chosen_location)
        if not read:
            out.append(SplitContrast(variable, reference, alternative, False, 0.0, 0.0, 0.0, 1.0))
            continue
        alt_data = _with_column(inf_data, variable, alternative)
        ref_data = _with_column(inf_data, variable, reference)
        # The same seed gives both calls the same coefficient draws, so the
        # difference of the two draw sets is the draw of the contrast.
        draws_alt = refit.draws(alt_data, n_draws=_DRAWS, seed=seed)[location]
        draws_ref = refit.draws(ref_data, n_draws=_DRAWS, seed=seed)[location]
        estimate, lower, upper, p_value = _rust.draw_contrast(
            np.asarray(refit.predict(alt_data, parameter=location, type="response"), dtype=float),
            np.asarray(refit.predict(ref_data, parameter=location, type="response"), dtype=float),
            np.asarray(draws_alt, dtype=float), np.asarray(draws_ref, dtype=float), level)
        out.append(SplitContrast(variable, reference, alternative, True, estimate, lower, upper,
                                 p_value))
    return tuple(out)


def _check_arguments(formula, select, repeats, fraction, level, contrasts, names):
    if not (callable(select) or select in _SELECT_NAMES):
        raise ValueError(f"select must be 'deselect', 'stabsel' or a callable; got {select!r}")
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError(f"repeats must be a positive integer; got {repeats!r}")
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must lie strictly between 0 and 1; got {fraction!r}")
    if not 0 < level < 1:
        raise ValueError(f"level must lie strictly between 0 and 1; got {level!r}")
    for spec in contrasts or ():
        if len(spec) != 3:
            raise ValueError("each contrast is (variable, reference, alternative)")
        variable, reference, alternative = spec
        if variable not in names:
            raise ValueError(f"contrast variable {variable!r} is not a column of data")
        if reference == alternative:
            raise ValueError(f"contrast of {variable!r} has equal reference and alternative")
    if isinstance(formula, dict) and select == "stabsel":
        raise NotImplementedError(
            "select='stabsel' is implemented for a gamboost formula; for a mapping of "
            "distribution parameters use select='deselect' or a callable")


def split_refit(formula, data, *, family=None, select="deselect", repeats=1, fraction=0.5,
                level=0.9, seed=0, contrasts=None):
    """Choose terms on one random half of the rows and estimate them by REML on
    the other half, optionally over several random splits.

    Parameters
    ----------
    formula : str or dict
        A ``gamboost`` formula, ``"y ~ bbs(x1, df=4) + bbs(x2, df=4)"``, or a
        mapping of distribution parameter names to formulas for a
        distributional family (``family=`` then names one, such as
        ``mb.GaussianLSS()``).
    data
        A dict of columns, a Polars or a pandas frame.
    family : optional
        The family, as for ``gamboost`` or ``gamboostLSS``.
    select : {"deselect", "stabsel"} or callable
        How the selection half chooses terms. ``"deselect"`` tunes the boosted
        fit on that half by 5-fold cross-validation and keeps the terms whose
        share of the risk reduction is at least 1% (:func:`mboostrs.deselect`).
        ``"stabsel"`` keeps the terms whose selection frequency over subsamples
        passes ``cutoff=0.75`` under an expected-false-learners bound of
        ``PFER=1``, with ``stabsel``'s default subsampling
        (:func:`mboostrs.stabsel`); it needs a ``gamboost`` formula. A callable
        ``select(data)`` receives the selection half and returns the formula
        terms to keep, as strings, or for a mapping ``{parameter: [terms]}`` or
        a list of ``(parameter, term)`` pairs. Returned terms must be terms of
        the formula, written as they are there.
    repeats : int
        Number of random splits. Each repeat draws its own split and seed.
    fraction : float
        Share of the rows in the selection half; the rest are the inference
        half. The refit sees only the inference half.
    level : float
        Coverage of the intervals.
    seed : int
        Seeds the splits, the fold assignments of the selection, and the
        coefficient draws. The same seed reproduces the same result.
    contrasts : list of (variable, reference, alternative), optional
        Average contrasts to report. For each repeat the contrast is the mean,
        over the inference rows, of the refit's change in the predictor when
        ``variable`` changes from ``reference`` to ``alternative`` and every
        other column keeps its value in that row. For a ``gamboost`` refit the
        predictor is on the link scale, and the interval comes from the refit's
        coefficient covariance. For a distributional refit it is the
        response-scale mean of the first distribution parameter, and the
        interval is the quantiles of posterior draws. A variable that no term
        the split kept reads contributes exactly zero in that split.

    Returns
    -------
    SplitRefitResult
        ``terms`` and ``contrasts`` hold the aggregate over repeats; ``splits``
        holds one record per repeat, with the rows used, the terms chosen, the
        term p-values and each contrast's per-split interval.

    Aggregation over repeats (``repeats > 1``)
        Selection frequency is the share of repeats that chose a term or read a
        contrast variable. Each contrast's estimate and interval bounds are the
        medians over repeats of the per-split values. Each p-value is aggregated
        by :func:`meinshausen_buhlmann_p` with ``gamma = 0.5``: twice the median
        of the per-split p-values, capped at 1. A repeat that did not choose a
        term counts as p = 1 for that term. With one repeat the per-split
        p-value is reported unchanged, and the intervals are those of the split.

    Limitations
    -----------
    The intervals are valid for a fixed split. Selective inference, which
    conditions on the selection event of the whole boosting path (the
    polyhedral method of Lee et al., 2016), is not implemented here. Each
    refit uses ``fraction`` of the rows, so a split interval is wider than one
    from a refit on all rows. A term that `gam_refit` cannot represent (``btree``,
    ``bmono``, ``bkernel``) makes the refit raise, naming the term. A contrast
    variable read by two or more refit terms raises, because the interval of
    their sum needs the covariance between those terms.

    The coverage of the procedure is a property of the whole procedure, the
    split, the selection rule and the refit, and it is checked by simulation in
    ``tests/test_split_refit.py``, not by the intervals alone.
    """
    from mboostrs._frame import column_names, take_rows
    from mboostrs._refit import _row_count

    lss = isinstance(formula, dict)
    names = column_names(data)
    _check_arguments(formula, select, repeats, fraction, level, contrasts, names)
    contrasts = [tuple(c) for c in (contrasts or ())]
    if lss:
        if family is None or not hasattr(family, "param_names"):
            raise ValueError("a mapping formula needs a distributional family, such as "
                             "family=mb.GaussianLSS()")
        params = list(family.param_names)
        missing = [p for p in params if p not in formula]
        if missing:
            raise ValueError(f"the mapping formula has no formula for {missing}; give one "
                             f"for each parameter of the family, {params}")
        family_name = type(family).__name__
    else:
        family_name = type(family).__name__ if family is not None else "Gaussian"

    n = _row_count(data)
    rng = np.random.default_rng(seed)
    splits = []
    for repeat in range(int(repeats)):
        sel_rows, inf_rows = _split_rows(n, fraction, rng)
        split_seed = int(rng.integers(0, 2**31 - 1))
        sel_data = take_rows(data, sel_rows)
        inf_data = take_rows(data, inf_rows)
        if lss:
            chosen = _lss_selection(formula, sel_data, family, select, split_seed)
            refit = _refit_lss(formula, inf_data, family, chosen) if any(chosen.values()) else None
            # A distributional refit labels each term "parameter: term", so one
            # term written for two parameters stays two terms.
            selected = tuple(f"{p}: {t}" for p in params for t in chosen[p])
            term_p = {}
            contrast_records = (_lss_split(refit, inf_data, contrasts, level, split_seed,
                                           chosen[params[0]])
                                if refit is not None else _zero_contrasts(contrasts))
        else:
            chosen = _gam_selection(formula, sel_data, family, select, split_seed)
            refit = _refit_gam(formula, inf_data, family, chosen) if chosen else None
            selected = tuple(refit.terms) if refit is not None else ()
            term_p = ({k: float(v["p_value"]) for k, v in refit.tests.items()}
                      if refit is not None else {})
            contrast_records = (_gam_split(refit, inf_data, contrasts, level)
                                if refit is not None else _zero_contrasts(contrasts))
        splits.append(SplitFit(repeat, len(sel_rows), len(inf_rows), selected, term_p,
                               contrast_records))

    select_label = select if isinstance(select, str) else "callable"
    scale = f"response of {params[0]}" if lss else "link"
    return SplitRefitResult(
        formula=formula, family=family_name, select=select_label, repeats=int(repeats),
        fraction=float(fraction), level=float(level), seed=int(seed), scale=scale,
        terms=tuple(_aggregate_terms(splits, repeats, lss)),
        contrasts=tuple(_aggregate_contrasts(splits, contrasts, repeats)),
        splits=tuple(splits))


def _zero_contrasts(contrasts):
    return tuple(SplitContrast(v, r, a, False, 0.0, 0.0, 0.0, 1.0) for v, r, a in contrasts)


def _aggregate_terms(splits, repeats, lss):
    seen = []
    for split in splits:
        for label in split.selected:
            if label not in seen:
                seen.append(label)
    out = []
    for label in seen:
        chosen = [label in s.selected for s in splits]
        frequency = sum(chosen) / repeats
        if lss:
            out.append(SplitTerm(label, frequency, None))
            continue
        per_split = [s.term_p_values.get(label, 1.0) if label in s.selected else 1.0
                     for s in splits]
        p = (per_split[0] if repeats == 1
             else meinshausen_buhlmann_p(per_split))
        out.append(SplitTerm(label, frequency, p))
    return out


def _aggregate_contrasts(splits, contrasts, repeats):
    from mboostrs import _rust

    out = []
    for index, (variable, reference, alternative) in enumerate(contrasts):
        records = [s.contrasts[index] for s in splits]
        p_values = [r.p_value for r in records]
        p = p_values[0] if repeats == 1 else meinshausen_buhlmann_p(p_values)
        out.append(ContrastSummary(
            variable, reference, alternative,
            selection_frequency=sum(r.selected for r in records) / repeats,
            estimate=_rust.transform_median(np.array([r.estimate for r in records], dtype=float)),
            lower=_rust.transform_median(np.array([r.lower for r in records], dtype=float)),
            upper=_rust.transform_median(np.array([r.upper for r in records], dtype=float)),
            p_value=p))
    return out
