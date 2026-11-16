"""The ``autospec`` "I'm feeling lucky" specification builder.

``autospec(data, response)`` guesses an entire boosting specification from the
data alone and returns it as ``(formula, learners, family, mstop)``, the four
objects :func:`~mboostrs.gamboost.gamboost` needs. No tuning, no cross
validation, no prior knowledge: the guess is made from the column types (for
the learners) and from the response (for the family), the two sources of
information that do not depend on the model being specified.

The returned tuple is a *specification*, not a fitted model. The whole point
of returning it half-architected is that every choice can be inspected and
overridden before fitting: read the formula, edit a ``df``, change the family,
raise ``mstop``, and hand the result to ``gamboost``. The guesses are the
default R ``mboost`` choices where one exists, and the codebase's own
classifiers where a data-driven default is better:

* **Learners** follow :func:`~mboostrs.terms.classify_columns`. A categorical
  column with more than two levels, including one stored as a few numeric
  codes, uses ``brandom`` -- the group learner fits one shrunken effect per
  level. ``bols`` on a numeric column always means one slope across its values;
  it is used only for a two-valued indicator unless the caller chooses that
  ordinal-scale interpretation explicitly.
* **Family** is inferred from the response column:
  a boolean or two-level framing is ``Binomial``, a multi-class factor is
  ``Multinomial``, a count is ``Poisson``, a positive-continuous response is
  ``GammaReg``, and everything else is ``Gaussian``. Whole numbers with a
  median above 1,000 are read as a positive measurement, not counts.
* **mstop** defaults to 100, matching :func:`~mboostrs.boost_control`.

One response type is refused with a reason rather than guessed: a two-level
*string* factor. ``Binomial`` needs a numeric ``{0, 1}`` or boolean response
(the engine codes internally as ``{-1, +1}``), so a label column has to be
recoded before ``autospec`` can name a family for it. A multi-level string
response is fine -- ``Multinomial`` encodes labels itself.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import replace

import numpy as np

from mboostrs._autospec import (
    _EXPOSURE,
    _MSTOP_EXTENSION_CAP,
    _SPREAD_TRACKS,
    _STEP,
    DeselectEvidence,
    SpecRecommendation,
    _FitCache,
    _lss_mu_scan,
    _risk_floor,
    compare_finished_routes,
    hygiene_findings,
    interaction_structure,
    priority_order,
    probe_family,
    probe_heteroscedasticity,
    probe_interactions,
    probe_mstop,
    probe_shapes,
)
from mboostrs._control import boost_control
from mboostrs._errors import MboostrsRuntimeError
from mboostrs._frame import read_column
from mboostrs._names import NAME_SRC as N
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._names import split_terms as _split_plus
from mboostrs._validate import check_whole_int
from mboostrs.cvrisk import cv_weights
from mboostrs.families import Binomial, GammaReg, Gaussian, Multinomial, Poisson, PropOdds
from mboostrs.folds import grouped_folds, rolling_folds
from mboostrs.formula import parse_gamboost_formula
from mboostrs.terms import CATEGORICAL_MAX_LEVELS, count_like, label_evidence

#: Doublings of the mstop grid a probe may make: 200 doubled 6 times passes
#: the 10,000-step cap.
_MAX_EXTENSIONS = 6
#: Fits set aside, in units of the fold count, to compare a finished
#: distributional route with the finished mean route: the distributional
#: model's log score and the squared error of its mean, and the mean model's
#: out-of-fold prediction and log score, one pass over the folds each.
_COMPARISON_FOLD_FITS = 5
#: Fewest remaining fits, in units of the fold count and beyond the
#: comparison's, with which the two routes are both run. Below it the family
#: probe's checks decide, as they did before the routes were compared.
_TWO_ROUTE_MIN_FOLDS = 8


def _candidate_path_cap(requested, incumbent):
    """The largest step count a candidate form's stopping search may reach.

    The additive incumbent's own optimum can only raise the cap; it cannot
    truncate a candidate below the caller's requested budget or the
    200-step floor every other stopping search in this file uses. A weak
    additive model converges early -- on the frozen joint_boundary draws
    (#536) it stopped at 23 steps while the caller asked for 200 -- and
    capping the interaction candidates at twice that, with no floor, judged
    every form on a path it had not finished using. `_MSTOP_EXTENSION_CAP`
    stays the hard ceiling. The cap sets path length, not fit count: each
    candidate costs one fit per fold whatever the cap is.
    """
    return min(_MSTOP_EXTENSION_CAP, max(2 * int(incumbent), int(requested), 200))

__all__ = ["OrderedCategoryWarning", "autospec"]

#: A term whose column is read through a transform: any learner call with
#: `log(` or `sqrt(` as its argument, not only `bbs(log(...))` -- the shape
#: probe can also propose `bols(log(x))` (#320).
_TRANSFORMED_TERM = re.compile(r"\b[A-Za-z_]\w*\(\s*(?:log|sqrt)\(")


def _reads_transformed(term: str) -> bool:
    return bool(_TRANSFORMED_TERM.search(term))


class OrderedCategoryWarning(UserWarning):
    """An ordered categorical predictor was fitted as unordered groups."""



def _scheme_rows(scheme, rows):
    """The `FoldScheme` restricted to (or reordered as) ``rows``: row ``i``
    of the result is row ``rows[i]`` of ``scheme``, in every per-row array."""
    import dataclasses

    rows = np.asarray(rows)

    def take(value):
        return None if value is None else np.asarray(value)[rows]

    return dataclasses.replace(
        scheme, weights=np.asarray(scheme.weights)[rows],
        held_out=np.asarray(scheme.held_out)[rows], judge=take(scheme.judge),
        groups=take(scheme.groups), strata=take(scheme.strata),
        base_weights=take(scheme.base_weights))

def autospec(data, response, *, family=None, mstop=100, df=4.0, smooth="bbs",
              budget=0, folds=None, seed=0, nu=None, prior=None,
              monotone=None, coordinates=None, adjacency=None, basis=None,
              groups=None, measurements=None, treatment=None, interactions="report",
              deselect=True, step=None, residual_tree=False, preexploration="native",
              heredity="none", forbid=None, mandatory=None, joint_tree=True,
              exposure=None):
    """Guess a complete boosting specification for ``data``.

    Parameters
    ----------
    data : dict or DataFrame
        Polars or pandas frame, or a mapping of column name to array.
    response : str
        Column holding the response.
    family : Family, optional
        Use this family instead of inferring one from the response. Whole-number
        values are otherwise read as counts, including values stored as
        floating point; that ambiguous case emits a warning and is recorded in
        ``rec.report()``. Declare the meaning of the response:

        * ``Gaussian()`` for a rounded continuous measurement (values such as
          house prices or ages recorded to the nearest unit);
        * ``Poisson()`` or ``NBinomial()`` for counts;
        * ``PropOdds()`` for ordered classes coded as integers ``1..K``
          (``PropOdds(n_levels=K)``) or as an ordered categorical column.
          Ordered categorical responses are inferred as ``PropOdds`` even
          without this argument; unordered labels are ``Multinomial``.
    exposure : str, optional
        Column of positive exposures (time at risk, area, number of trials)
        for a count response. The response is then modelled as a rate per
        unit of exposure: ``log(exposure)`` is an offset in every fit, the
        column is not a predictor, and ``rec.fit`` passes the offset. The
        family is ``Poisson()`` unless ``family=NBinomial()`` is given.
        Missing or non-positive exposures raise an error; predictions on new
        rows need their exposures as ``offset=`` (see ``rec.fit``).
    mstop : int
        The step count in the returned spec. Default 100, matching
        :func:`~mboostrs.boost_control`.
    df : float
        Degrees of freedom for the penalised learners. Default 4, matching
        ``bbs`` and R ``mboost``.
    smooth : {"bbs", "btree"}
        The learner for a continuous column. ``"bbs"`` is a P-spline
        (default); ``"btree"`` fits thresholds instead.
    budget : int
        How many model fits the probes may spend. ``0``, the default, runs
        none of them: the specification is the dtype-driven one this function
        has always returned, at the same cost, plus the hygiene findings,
        which need no fit. Give a budget to have the choices measured --
        the family probe costs ``1 + n_folds`` fits per candidate family --
        and ``rec.report()`` then distinguishes what was measured from what
        fell back to a default. A measurement that lands within sampling
        noise of a decision line declines to choose and says so, and fits
        two probes need in common are made once and counted once. The first
        measurement tunes `mstop` for the dtype-driven specification, and the
        family and shape probes compare their candidates at that depth: at
        a default 100 steps a slowly learned signal is still mostly unfitted
        and every candidate scores about the same. The shape probe changes
        one column's learner at a time with the rest of the model held fixed
        at its fitted value, which costs one-term fits. The family probe also
        offers the distributional family the response's support names —
        ``BetaLSS`` for a response inside (0, 1), ``GammaLSS`` for a spread
        that grows with the level — and scores it against the surviving mean
        family on held-out log score, adopting it only when it clearly wins;
        a search that adopts one stops at the family, and ``rec.fit()`` then
        builds the distributional model.
        For Gaussian responses with coordinate-like predictors (such as
        ``lon`` and ``lat``), the final shape comparison also offers one
        joint CART tree over the undeclared continuous predictors, at depths
        3 and 6. Each specification is tuned on the shared folds; replacement
        requires at least 2% less held-out risk. Declared and mandatory terms
        remain unchanged. The comparison costs 3 * (folds + 2) fits; once the
        family is settled those fits are set aside from the earlier probes
        when the budget can also pay for one more sweep, and
        ``evidence["joint_tree"]["reserved"]`` records the amount.
        ``evidence["joint_tree"]`` records the candidates,
        their scores and the decision, including a skipped comparison when
        the budget or interaction constraints prevent it. Selection on these
        folds needs a separate test set to measure generalisation.
        When the search has finished, the specification it settled on is
        compared as a whole with two simple models fixed in advance, on the
        same folds and by the same held-out risk:

        * ``"linear"``: the starting specification with every one-column
          smooth written as a linear term (``bbs(x, df=4)``, ``btree(x)``
          and ``bns(x)`` become ``bols(x)``; a ``log``/``sqrt`` transform of
          the input is kept). Factor terms (``bols``, ``brandom``), the
          caller's declared terms and terms over two or more columns stay as
          written.
        * ``"additive"``: the starting specification itself, the one
          ``budget=0`` returns: a ``bbs(x, df=df)`` smooth per continuous
          column, ``bols``/``brandom`` per factor, with the declarations
          applied; with `prior`, the prior's formula.

        Each rival is fitted at the search's step length and judged at its
        own best step count. A rival replaces the searched specification
        only when its held-out risk is lower by more than one standard error
        of the paired fold difference; on a single holdout that standard
        error is undefined and the searched specification stands. The
        comparison costs one sweep (``n_folds`` fits) for the searched
        specification and one per rival that differs from it, and runs only
        when the remaining budget pays for it; the search itself does not set
        fits aside for it. ``evidence["whole_model"]`` records each rival's
        formula, held-out risk (``risk``) and step count (``mstop``), the
        paired difference to the searched specification (``difference``,
        rival minus searched, negative when the rival does better) with its
        standard error (``difference_se``), whether the rival replaced the
        searched specification (``replaced``, ``winner``), or why the
        comparison did not run (``reason``). A distributional
        recommendation carries the reason only: its held-out score is a log
        score, not the mean family's risk, and
        ``evidence["finished_routes"]`` compares it with the finished mean
        model.
    folds : ndarray or FoldScheme, optional
        Fold weights the probes share, so every comparison is made on the
        same split. When omitted, the frame's own structure names them: a
        date or datetime column gives rolling-origin folds ordered by it
        (the past predicts the future), and columns the specification fits
        as groups -- ``brandom`` or ``bmrf`` terms, declared or read from
        the dtype -- give grouped folds that hold whole groups out;
        otherwise 5 random folds on `seed`. A supplied `folds` overrides the
        detection; `groups=` adds its columns to the detected set.
        A design with a single fold (one supplied holdout) carries no
        across-fold variation, so the paired fold tests cannot fire.
        On such a design the probes that would otherwise decide on an
        aggregate margin use the per-row evidence of the holdout
        itself: a shape replacement, a screened pair's form, an
        interaction pair or the residual tree is taken only when its
        margin also clears the paired per-row z bar. The z recorded in
        the evidence is the one the decision used.
        Families whose loss couples rows (CoxPH, AUC, Cindex, Gehan)
        have no per-row losses; their single-holdout decisions keep the
        margin rule alone.
        The distributional family's adoption checks are paired fold
        tests with no per-row form, so on a single fold they cannot
        pass: a distributional family is not adopted on a supplied
        holdout, and the mean family stands.
    seed : int
        Seed for the default and grouped folds.
    nu : float, optional
        Step length of every probe and of the recommendation. ``None``, the
        default, is 0.3 (``gamboost``'s own default is mboost's 0.1): the
        probes walk each candidate's path to its cross-validated optimum, and
        at 0.1 those paths ran to 10,000 steps, while held-out accuracy on
        development data matched within 0.001 of R^2. It is lengthened to 0.5
        when the cross-validated risk is still falling at the 10,000-step cap
        (strongly collinear columns make component-wise boosting slow); the
        report says so. A value you give is used as given. When you give
        none, the finished specification is also read at a bounded grid of
        other step lengths (0.1, 0.2, 0.5, 0.7 and 1.0, in addition to the
        step length in force) once the search settles, at one fold path per
        extra value charged to the same fit budget, and the recommendation
        takes the step length whose sweep reads the lowest cross-validated
        risk; ``evidence["nu_search"]`` records the grid, each value's
        chosen step count and risk, and the fits spent.
    preexploration : {"native", "neural", "dual", "hessboost", "off"}
        Preliminary boosted-tree exploration for budgeted searches. One
        ensemble supplies permutation importance, partial-dependence shape
        diagnostics and interaction proposals; its fit is charged once.
        For Gaussian responses, agreement between marginal and partial
        diagnostics narrows shape comparisons, ordered by importance.
        Incumbents and declared terms are retained; ambiguous shapes stay
        open. Other families use the pair proposals without shape pruning.
        Every replacement still requires the usual held-out comparison.
        ``"neural"`` fits a seeded Rust MLP (two 32-unit tanh layers),
        using one-hot categories and training-fitted numeric scaling.
        ``"dual"`` fits both native trees and the net, charges two fits,
        and merges their pair proposals with source labels. Neural subset
        diagnostics extend greedily to at most five inputs; they describe
        total nonadditivity, including lower-order interactions, and do
        not automatically add higher-order terms. Only tree evidence may
        narrow shape comparisons. The default remains ``"native"``.
        ``"off"`` uses the full shape catalogue and residual pair screen.
        ``"hessboost"`` requires a build with the optional Cargo feature of
        that name; otherwise it raises an error. Native is the default.
        Screening uses the supplied training data; risk curves after this
        adaptive search are descriptive, not unbiased performance estimates.
        Reserve separate test data for evaluation of the whole procedure.
    prior : str or mapping, optional
        A specification you already believe in, as a formula string or a
        mapping with ``formula`` and optionally ``family`` and ``mstop``. It
        becomes the baseline: the probes measure against it rather than
        against the dtype guess, each term it names is the incumbent shape,
        and the report says which probe disagreed with it and by how much. A
        probe that finds nothing better leaves your specification alone.
    monotone : dict, optional
        Mapping of column name to a ``Bmono`` constraint (``"increasing"``,
        ``"decreasing"``, ``"convex"``, ``"concave"``, ``"positive"``,
        ``"negative"``). Those columns get ``bmono`` terms carrying the
        stated constraint. Declared terms are fixed: the shape probe scores
        the alternatives against them and reports what the constraint costs,
        but never replaces them.
    coordinates : tuple of str, optional
        Two column names that are coordinates. They get one
        ``bspatial(col1, col2)`` term in place of their two separate terms.
    adjacency : dict, optional
        Mapping of column name to a neighbourhood penalty matrix ``K`` in
        graph-Laplacian form (see :class:`~mboostrs.learners.Bmrf`). Those
        columns get ``bmrf`` terms carrying ``K``.
    basis : dict, optional
        Mapping of column name to an ``(X, K)`` pair. Those columns get
        ``buser`` terms built from your design and penalty.
    groups : str or sequence of str, optional
        Columns that identify groups: clinics, districts, subjects. Each gets
        a ``brandom`` term, and the probes share grouped folds that hold
        whole groups out, so every measurement scores prediction for a
        group the fit has not seen. Columns the dtype rule already fits as
        groups are detected on their own; this declares the ones it reads
        as measurements. Without `folds`, both sets are held out whole.
    measurements : str or sequence of str, optional
        Columns you know are ordered quantities, whatever the dtype rule
        would do with them: a seasonal coordinate sampled at three
        values, a code whose levels are equally spaced scores. Each gets
        one slope (``bols``), the shape probe prices the smooth and the
        threshold against it, and the columns leave the group-detection
        set. This is the one-word answer to the report's note that a
        few-valued column "is read as groups, not as a measured
        quantity".
    interactions : {"report", "adopt"}
        What to do with the pairs the interaction screen finds (it runs when
        `budget` allows). ``"report"`` lists them under
        ``evidence["interactions"]`` and leaves the formula additive;
        ``"adopt"`` also adds each proposed pair term (a ``%X%`` tensor, a
        varying slope, or a switch-on smooth), skipping pairs that record one
        axis twice, and re-tunes `mstop` for the larger formula when budget
        remains.
    residual_tree : bool or "auto"
        After the interaction step, try one more term: a regression tree of
        depth 5 over every numeric and categorical column (``btree(x1, ...,
        xp, maxdepth=5)``; a factor splits into two sets of its levels),
        boosted alongside the other terms. It fits structure
        that the main effects and the adopted pairs do not: interactions of
        three or more columns, and pairs the screen did not propose. The term
        is kept only when it lowers the cross-validated risk by 2% or by two
        paired fold standard errors, the rule adopted pairs must pass, and
        `mstop` is tuned again with it. ``evidence["residual_tree"]`` records
        the comparison: ``held_out_gain`` is the fold mean of the incumbent's
        risk minus the tree's risk (positive when the tree lowers it), with
        ``held_out_se`` its standard error and ``z`` the paired statistic the
        rule read; when the standard error is undefined (one fold, or a fold
        without a score) ``held_out_se`` is NaN and ``held_out_reason`` says
        why. A distributional family records the same fields for its `mu`
        predictor, in the family's held-out loss (negative log-likelihood
        per row). On the mean route ``share`` gives the part of
        the adopted prediction the tree carries on training rows, as
        ``covariance_share`` and ``absolute_share``. The other terms keep
        their curves. The screen-stage lead is recorded apart from the
        decision, in ``evidence["residual_tree_detection"]``: the share of the
        response variance the additive fit leaves, and a depth-3 tree's R^2 on
        that leftover. When the predictors include a coordinates-like pair and
        the folds are not
        already grouped, the adoption comparison uses grouped folds over
        spatial grid cells: random row folds let the tree validate on
        rows neighbouring its training rows, so the gain need not transfer
        to unseen regions. ``evidence["residual_tree"]["validation"]`` names
        the scheme that decided. The
        term is not an effect curve to read: ``rec.fit(data, refit="reml")``
        holds it at its boosted contribution while the other terms are
        refitted. ``"auto"`` offers the tree only when the training-only
        rule fires: the predictors include a coordinates-like pair,
        or adopted pairs leave residual structure a shallow tree captures.
        The default ``False`` never offers; ``True`` always offers.
    step : {None, "fixed", "line_search"}
        The step rule of every probe and of the recommendation (see
        :func:`~mboostrs.boost_control`). ``None``, the default, fits with the
        fixed step and, when a fit diverges (a Poisson, negative binomial or
        gamma response in the hundreds overflows its inverse link in the
        first fixed steps), runs again with ``"line_search"``; the
        recommendation's control then carries the line search, so
        ``rec.fit`` replays it, and the report says why. With ``budget=0`` no
        fit runs to detect a divergence, so a Poisson response takes the line
        search directly, because its fixed step grows with the mean.
    heredity : {"none", "weak", "strong"}, default "none"
        The marginality rule for adopted interactions (Nelder; Chipman 1996;
        Bien, Taylor & Tibshirani 2013). ``"strong"`` adopts a pair only when
        both of its main effects are in the model and keeps both through
        deselection; ``"weak"`` requires one and keeps at least one.
        Adoption is tested while every modelled column still has its main
        term, and deselection runs afterwards, so a pair whose parents are
        both modelled is admitted under ``"strong"`` too. A parent that is not
        modelled (a constant column, or one the hygiene checks drop) bars
        the pair. ``"none"`` (the default) adopts on held-out evidence alone.
        The rule in force is stated in ``rec.report()`` and in
        ``evidence["interaction_rule"]``.
    forbid : iterable of (str, str), optional
        Column pairs the interaction screen never lists or adopts.
    mandatory : iterable of str, optional
        Columns whose terms deselection never drops.
    joint_tree : bool, default True
        Offer the coordinate comparisons when undeclared continuous predictors
        include a coordinate-like pair: ``bspatial(x, y)`` against
        ``btree(x, y)`` on the shared folds (``evidence["coordinate_surface"]``),
        then the final Gaussian joint-tree comparison
        (``evidence["joint_tree"]``). ``False`` keeps the search to its
        per-column shapes and explicitly requested pair/residual-tree probes,
        for example when a caller needs an additive specification. Joint
        replacement also skips non-default heredity rules and forbidden pairs
        among its inputs.
    deselect : bool or float
        After `mstop` is tuned (when `budget` allows), drop each term whose
        share of the tuned fit's in-bag risk reduction is below 1% (or below
        the share given as a number, such as ``deselect=0.025``) and tune
        `mstop` again for the smaller formula (Strömer et al. 2022; the rule
        :func:`~mboostrs.deselect` applies to a fitted model). A
        cross-validated boosting fit keeps every term selected even once, so
        most noise columns survive it; this removes them. The smaller
        formula is kept only if its cross-validated risk, on the same folds,
        is at most 1% above the full formula's; many small terms can together
        carry real signal. Declared terms
        (`monotone=`, `coordinates=`, `groups=` and the rest) and terms of a
        `prior` are never dropped. ``evidence["deselect"]`` lists the dropped
        terms, their shares and the held-out loss. The default share is set
        for prediction. A larger one gives a sparser specification for
        identifying drivers: at 2.5%, on a simulated design with six drivers,
        three correlated proxies and 21 noise columns, a 1,000-row binary
        response kept 0.6 non-drivers per fit instead of 4.1 with the same
        driver recall, while test R^2 on seven real data sets fell by up to
        0.01.
    treatment : str or CausalSpec, optional
        The column a causal question is about. It is left out of the
        predictive specification -- scoring a treatment by its predictive
        association answers a different question from estimating its
        effect -- and the report points at the causal workflow that does
        answer it: ``mboostrs.causal`` (``CausalSpec``, ``aipw``,
        ``negative_control``). A full `CausalSpec` must name this call's
        response as its outcome, and its post-treatment columns are left
        out with the treatment.

    Returns
    -------
    SpecRecommendation
        It unpacks as the tuple this function has always returned --
        ``(formula, learners, family, mstop)`` -- and additionally carries the
        data-hygiene findings and, once probes run, the evidence behind each
        choice. ``rec.report()`` prints the reasoning and ``rec.fit(data)``
        fits the specification. The tuple is: a formula string naming the
        terms, the matching :class:`~mboostrs.learners.Bbs`/
        :class:`~mboostrs.learners.Bols`/:class:`~mboostrs.learners.Brandom`
        objects, the response family, and the step count. The first three feed
        ``gamboost(formula, data, learners=..., family=...)`` directly.

    Raises
    ------
    ValueError
        If `family` is omitted and the response is a two-level string factor,
        which ``Binomial`` cannot read until the labels are recoded as 0/1.
        Also if `prior` is combined with any declaration keyword, or a
        declaration names a column the frame does not hold, a column is
        declared both a group and a measurement, or `treatment`
        names the response, a column the frame does not hold, or a
        `CausalSpec` whose outcome is not this call's response.
    """
    mstop = check_whole_int("mstop", mstop, 1)
    budget = check_whole_int("budget", budget, 0)
    if not isinstance(deselect, bool) and not (
            isinstance(deselect, (int, float)) and 0 < float(deselect) < 0.5):
        raise ValueError(
            f"deselect must be True, False or a share of the risk reduction in "
            f"(0, 0.5), such as 0.025; got {deselect!r}")
    if preexploration not in ("native", "neural", "dual", "hessboost", "off"):
        raise ValueError('preexploration must be "native", "neural", "dual", "hessboost" or "off"')
    if not isinstance(joint_tree, bool):
        raise ValueError("joint_tree must be True or False")
    auto_nu = nu is None
    # autospec's own step length. On the training rows of development data
    # sets, 0.3 matched 0.1 in held-out R^2 (cpu_activity 0.9819 against
    # 0.9825, airfoil 0.5584 against 0.5592) at a third of the time: the probes
    # walk each path to its cross-validated optimum, and at 0.1 those paths ran
    # to 10,000 steps. gamboost's own default stays mboost's 0.1.
    nu = _AUTOSPEC_NU if nu is None else nu
    arguments = {k: v for k, v in locals().items() if k not in ("step", "auto_nu")}
    if step is None:
        from mboostrs._errors import MboostrsRuntimeError

        try:
            rec = autospec(**{**arguments, "nu": None if auto_nu else nu}, step="fixed")
        except MboostrsRuntimeError as exc:
            if "non-finite" not in str(exc):
                raise
            rec = autospec(**{**arguments, "nu": None if auto_nu else nu}, step="line_search")
            rec.evidence["step"] = (
                "line_search — the fixed step made the risk non-finite (the "
                "inverse link overflowed in the first steps), so every probe "
                "and rec.fit() use a line search for the step length")
            return rec
        if type(rec.family).__name__ in _COUNT_SCALE_NAMES and not budget:
            # With no budget no fit runs, so nothing checks the fixed step
            # before rec.fit(). The Poisson fixed step is not scale-invariant:
            # its update is nu times a least-squares fit to y - mu, which grows
            # with the mean. On counts near 50 the nu = 0.3 path overflowed in
            # its second step, and on counts near 3 its held-out deviance was
            # about 40% above the line search's (1.65 against 1.17 per row). The
            # line search needs no fit to be safe, so the recommendation takes it.
            searched = autospec(**{**arguments, "nu": None if auto_nu else nu},
                                step="line_search")
            searched.evidence["step"] = (
                f"line_search — for {type(rec.family).__name__} the fixed step "
                "grows with the fitted mean, and with no budget no fit checks it, "
                "so every probe and rec.fit() use the line search for the step length")
            return searched
        if type(rec.family).__name__ in _LOG_LINK_NAMES and budget:
            # A log-link fit of a large response can stop after a few fixed
            # steps, long before it fits the signal (a Gamma model of house
            # prices turned after 23 steps, where a line search went on to
            # 8,773 and a better held-out fit). Run the search again with the
            # line search and keep the recommendation whose cross-validated
            # risk is lower, on the same folds.
            searched = autospec(**{**arguments, "nu": None if auto_nu else nu},
                                step="line_search")
            fixed_risk, searched_risk = _cv_minimum(rec), _cv_minimum(searched)
            if (type(searched.family) is type(rec.family)
                    and searched_risk < fixed_risk):
                searched.evidence["step"] = (
                    f"line_search — for {type(rec.family).__name__} the line "
                    f"search reached a lower cross-validated risk "
                    f"({searched_risk:.6g} against {fixed_risk:.6g} with the "
                    f"fixed step), so every probe and rec.fit() use it")
                return replace(searched, fits=searched.fits + rec.fits)
            return replace(rec, fits=rec.fits + searched.fits)
        return rec
    if step not in ("fixed", "line_search"):
        raise ValueError('step must be None, "fixed" or "line_search"')
    if exposure is not None and not isinstance(exposure, str):
        raise ValueError(f"exposure must be the name of a column, got {exposure!r}")
    token = _STEP.set(step)
    exposure_token = _EXPOSURE.set(None)
    try:
        return _autospec_body(**arguments, step=step, auto_nu=auto_nu)
    finally:
        _EXPOSURE.reset(exposure_token)
        _STEP.reset(token)


def _autospec_body(data, response, *, family, mstop, df, smooth, budget, folds, seed, nu,
                   prior, monotone, coordinates, adjacency, basis, groups, measurements,
                   treatment, interactions, deselect, step, residual_tree=False,
                   auto_nu=False, preexploration="native", heredity="none", forbid=None,
                   mandatory=None, joint_tree=True, exposure=None):
    """The body of :func:`autospec` at one step rule."""
    if heredity not in ("none", "weak", "strong"):
        raise ValueError(f'heredity must be "none", "weak" or "strong", got {heredity!r}')
    if residual_tree not in (False, True, "auto"):
        raise ValueError(f'residual_tree must be False, True or "auto", got {residual_tree!r}')
    forbidden = {frozenset((str(a), str(b))) for a, b in (forbid or ())}
    mandatory = tuple(str(c) for c in (mandatory or ()))
    pair_parents = {}
    from mboostrs.formula import data_columns as _columns

    missing_mandatory = [c for c in mandatory if c not in set(_columns(data))]
    if missing_mandatory:
        raise ValueError(f"mandatory names columns the data do not hold: {missing_mandatory}")
    if response in mandatory:
        raise ValueError(f"mandatory names the response {response!r}; name a predictor column")
    if smooth not in ("bbs", "btree"):
        raise ValueError(f"smooth must be 'bbs' or 'btree', got {smooth!r}")

    columns = list(data.columns) if hasattr(data, "columns") else list(data)
    if response not in columns:
        raise ValueError(
            f"response column {response!r} is not in the data "
            f"(columns: {', '.join(map(repr, columns))})"
        )
    family_was_inferred = family is None
    if family is None:
        family = _infer_family(data, response)
    exposure_values = None
    if exposure is not None:
        exposure_values = _read_exposure(data, response, exposure)
        if family is None:
            family = Poisson()
        elif type(family).__name__ not in ("Poisson", "NBinomial"):
            raise ValueError(
                f"exposure= applies to count families, but family is "
                f"{type(family).__name__}; pass Poisson() or NBinomial(), or drop "
                f"exposure= to model the response without an offset")
        family_was_inferred = False

    predictors = [read_column(data, name) for name in columns if name != response]
    ordered = [column for column in predictors
               if column.kind == "categorical" and column.ordered
               and len(column.levels or ()) > 2]
    if ordered:
        names = ", ".join(repr(column.name) for column in ordered)
        warnings.warn(
            f"ordered categorical predictor(s) {names} are fitted with brandom, "
            "which estimates shrunken level effects but does not use their declared "
            "order. To model an ordinal trend, add an explicitly scored numeric "
            "column and choose bols (one equally spaced slope), bbs (a smooth over "
            "scores), or bmono (a stated monotone direction).",
            OrderedCategoryWarning,
            stacklevel=2,
        )
    if groups is None:
        groups = ()
    elif isinstance(groups, str):
        groups = (groups,)
    else:
        groups = tuple(groups)
    for name in groups:
        if name == response or name not in columns:
            raise ValueError(
                f"groups= names {name!r}, which is not a predictor in the data")
    if measurements is None:
        measurements = ()
    elif isinstance(measurements, str):
        measurements = (measurements,)
    else:
        measurements = tuple(measurements)
    for name in measurements:
        if name == response or name not in columns:
            raise ValueError(
                f"measurements= names {name!r}, which is not a predictor "
                f"in the data")
        if name in groups:
            raise ValueError(
                f"pass groups= or measurements= for {name!r}, not both: a "
                f"column is either a grouping or an ordered quantity")
    treatment_columns = _read_treatment(treatment, response, columns)
    response_column = read_column(data, response)
    response_values = (response_column.values
                       if response_column.kind in ("numeric", "boolean") else None)
    all_columns = [read_column(data, name) for name in columns]
    hygiene = hygiene_findings(data, response, all_columns, groups=groups,
                               caller_folds=folds is not None,
                               treatment=treatment_columns,
                               measurements=measurements,
                               # A class label's coding is the family's, not
                               # a warning sign, whether declared or inferred.
                               binary_response=type(family).__name__ in (
                                   "Binomial", "Multinomial", "PropOdds",
                                   "AdaExp", "HingeLoss", "AUC"))
    dropped = {finding.column for finding in hygiene
               if finding.kind in ("constant", "empty", "identifier",
                                   "duplicate-encoding")}
    modelled = [column for column in predictors
                if column.name not in dropped
                and column.name not in treatment_columns
                and column.name != exposure]
    if not modelled:
        raise ValueError(
            "no predictor survives the hygiene checks: "
            + "; ".join(str(finding) for finding in hygiene))

    if prior is not None:
        if monotone or coordinates or adjacency or basis or measurements:
            raise ValueError(
                "pass prior= or the declaration keywords (monotone=, "
                "coordinates=, adjacency=, basis=, measurements=), not "
                "both: a prior already states its own terms.")
        prior_formula, prior_family, prior_mstop = _read_prior(prior, response)
        formula = prior_formula
        _, learners = parse_gamboost_formula(formula)
        terms = _split_plus(partition_tilde(formula)[2])
        named = _prior_columns(terms)
        modelled = [column for column in modelled if column.name in named]
        if prior_family is not None:
            family = prior_family
            family_was_inferred = False
        if prior_mstop is not None:
            mstop = prior_mstop
        declared = tuple(sorted(named))
        env = {}
    else:
        terms = [
            f"brandom({quote_name(column.name)}, df={df})" if column.name in groups
            else f"bols({quote_name(column.name)})" if column.name in measurements
            else _term_for(column, df, smooth, response_values)
            for column in modelled
        ]
        env, declared = _apply_declarations(
            modelled, terms, df=df, monotone=monotone,
            coordinates=coordinates, adjacency=adjacency, basis=basis)
        if type(family).__name__ == "Multinomial":
            terms = [_decomposed(term) for term in terms]
        formula = f"{quote_response(response)} ~ " + " + ".join(terms)
        if any((_reads_transformed(term) or " + " in term)
               for term in terms):
            # A transform term is expanded when gamboost fits the formula.
            learners = None
        else:
            _, learners = parse_gamboost_formula(formula, env=env or None)
    # The starting specification (the dtype-driven terms with the caller's
    # declarations, or the prior) is one of the simple rivals the finished
    # search is compared with at the end (`_whole_model_step`).
    start_formula = formula
    evidence: dict = {}
    evidence["response"] = _response_record(
        family, response_column, declared=not family_was_inferred, exposure=exposure)
    # The rule in force is recorded on every run, the default included, so the
    # report and the evidence state what was applied.
    evidence["interaction_rule"] = {
        "heredity": heredity,
        "forbid": tuple(sorted(" x ".join(sorted(pair)) for pair in forbidden)),
        "mandatory": mandatory}
    measured: list[str] = []
    defaulted = (["family"] if family_was_inferred else []) + ["shapes", "interactions", "mstop"]
    spent = 0

    # The probes fit on complete rows. A missing value in the response or a
    # modelled column makes every fit refuse the row, so the rows are left out
    # once, here, and the report and `rec.fit` say so.
    complete = np.isfinite(response_column.values)
    for column in modelled:
        complete &= np.isfinite(column.values)
    incomplete_rows = int((~complete).sum())
    if incomplete_rows and not complete.any():
        raise ValueError("no row has the response and every modelled column present")
    if incomplete_rows:
        from mboostrs._frame import take_rows

        data = take_rows(data, np.flatnonzero(complete))
    if exposure_values is not None:
        # The offset every probe fit carries, on the same rows as the data.
        _EXPOSURE.set(np.log(exposure_values[complete]))

    n_rows = int(complete.sum())
    folds_description = ""
    shared_scheme = None      # the FoldScheme, when the design carries a judge set
    shared_judge = None        # its judge mask, if it has one
    if folds is not None:
        folds_description = (f"supplied by the caller ({folds.scheme})"
                              if hasattr(folds, "scheme") else "supplied by the caller")
        if hasattr(folds, "scheme"):
            # The scheme itself is subset, so the matrix, its judge mask and
            # every refit that reads the scheme describe the same rows.
            shared_scheme = (_scheme_rows(folds, np.flatnonzero(complete))
                             if incomplete_rows else folds)
            shared_judge = shared_scheme.judge
            shared_folds = np.asarray(shared_scheme, dtype=float)
        else:
            shared_folds = np.asarray(folds, dtype=float)
            if incomplete_rows:
                shared_folds = shared_folds[complete]
    else:
        # The frame's own structure names the honest folds when the caller
        # left the design to the tool. Columns the specification fits as
        # groups -- brandom or bmrf terms, declared with groups= or read
        # off the dtype -- say rows within a group share more than the
        # model explains. The folds hold out ONE grouping, the one with the
        # most levels: combining several group columns into one label makes
        # each (child, exam-round) pair its own group, which collapses the
        # repeated-measures grouping it was meant to protect. A declared
        # groups= is the caller's authority and is used as given; the
        # other group-fitted columns are then split, and the report says so.
        detected_time = _time_column(all_columns, response, dropped)
        declared_groups = list(groups)
        group_terms = sorted(
            {column.name for column, term in zip(modelled, terms, strict=True)
             if term.startswith(("brandom(", "bmrf("))})
        shared_folds = None
        if detected_time is not None:
            try:
                scheme = rolling_folds(n_rows, B=5)
            except ValueError:
                if budget > 0:
                    folds_description = (
                        f"5 random folds ({n_rows} rows are too few for a "
                        f"rolling origin)")
            else:
                # rolling_folds numbers rows in time order; row order[k] is the
                # k-th earliest, so the scheme is moved to row order as a whole.
                # Moving only the matrix left the refits that read the scheme
                # (cvrisk, the mstop probe) training on later rows than they
                # scored whenever the data were not sorted by time (#61).
                order = np.argsort(read_column(data, detected_time).values,
                                   kind="stable")
                shared_scheme = _scheme_rows(scheme, np.argsort(order, kind="stable"))
                shared_judge = shared_scheme.judge
                shared_folds = np.asarray(shared_scheme, dtype=float)
                folds_description = (
                    f"rolling-origin 5-fold ordered by {detected_time}: the "
                    f"past predicts the future, and only rows after the first "
                    f"half are judged (detected from the column's dates; pass "
                    f"folds= for another design)")
        elif declared_groups:
            labels = _group_labels(data, declared_groups)
            distinct = int(np.unique(labels).size)
            if distinct < 2:
                if budget > 0:
                    folds_description = (
                        f"5 random folds ({', '.join(declared_groups)} holds "
                        f"every row, so there are no groups to hold out)")
            else:
                scheme = grouped_folds(labels, B=min(5, distinct), seed=seed)
                shared_folds = np.asarray(scheme, dtype=float)
                folds_description = (
                    f"{shared_folds.shape[1]} grouped folds holding whole "
                    f"groups of {', '.join(declared_groups)} out (declare "
                    f"folds= for another design)")
        elif group_terms and max(int(np.unique(read_column(data, name).values).size)
                                 for name in group_terms) >= _MIN_UNIT_LEVELS:
            unit = max(group_terms,
                       key=lambda name: int(np.unique(
                           read_column(data, name).values).size))
            labels = read_column(data, unit).values
            distinct = int(np.unique(labels).size)
            if distinct < 2:
                if budget > 0:
                    folds_description = (
                        f"5 random folds ({unit} holds every row, so there "
                        f"are no groups to hold out)")
            else:
                scheme = grouped_folds(labels, B=min(5, distinct), seed=seed)
                shared_folds = np.asarray(scheme, dtype=float)
                split = [name for name in group_terms if name != unit]
                note = (f"; {', '.join(split)} are also fitted as groups "
                        f"but their rows are split, not held out whole"
                        if split else "")
                folds_description = (
                    f"{shared_folds.shape[1]} grouped folds holding whole "
                    f"groups of {unit} out (detected: the terms fit it as a "
                    f"grouping{note}; declare folds= or groups= to control "
                    f"the design)")
        elif budget > 0:
            folds_description = "5 random folds (rows, not groups, are held out)"
            if group_terms:
                folds_description += (
                    f"; {', '.join(group_terms)} have fewer than {_MIN_UNIT_LEVELS} "
                    f"levels, so they are fitted as categories whose levels all "
                    f"appear in training; declare groups= to predict unseen groups")

    def fold_matrix():
        if shared_folds is not None:
            return shared_folds
        return cv_weights(n_rows, type="kfold", B=5, seed=seed)

    cache = _FitCache()
    screen_reused = 0
    # Marginal scatterplot diagnostics of the response against each modelled
    # column (no fit; the Rust core's screen). They name what the shape probe
    # then tests held out.
    try:
        from mboostrs.terms import column_screen

        evidence["column_screen"] = tuple(column_screen(
            data, response, [c.name for c in modelled], output="records"))
    except Exception:
        pass

    # The probes compare specifications that are fitted, not ones stopped
    # after a default 100 steps: first find how many steps the dtype guess
    # needs, and judge the family and shape candidates at that depth.
    probe_depth = min(int(mstop), 100)
    depth_row, depth_formula, depth_family = None, None, None
    if budget > 0 and ("family" in defaulted or "shapes" in defaulted):
        fold_weights = fold_matrix()
        n_folds = int(np.asarray(fold_weights).shape[1])
        if budget >= 3 * n_folds:
            # A log-link family guessed from the dtype (whole numbers read as
            # counts) is not fitted before the family probe has judged it: its
            # fixed steps overflow on a wide response, and the line search that
            # rescues them walks a long path at several risk evaluations per
            # step (790 of 1,195 s on cpu_activity). The depth comes from a
            # Gaussian sweep, and the family probe decides next.
            sweep_family = (Gaussian() if family_was_inferred
                            and type(family).__name__ in _LOG_LINK_NAMES else family)

            def depth_sweep():
                return probe_mstop(
                    data, response, formula, sweep_family, folds=fold_weights,
                    largest=max(int(mstop), 200), nu=nu, env=env or None,
                    extend=min(_MAX_EXTENSIONS, (budget // 3 - n_folds) // n_folds),
                    fold_scheme=shared_scheme)
            try:
                depth_row = depth_sweep()
            except MboostrsRuntimeError as exc:
                # The dtype guess (a count family for whole numbers) can
                # overflow under the fixed step before the family probe has
                # judged it. Only this sweep switches to the line search; the
                # family probe decides next, and a Gaussian verdict goes back
                # to the fixed step. Rerunning the whole search with the line
                # search took three times as long on cpu_activity.
                if "non-finite" not in str(exc) or _STEP.get() != "fixed":
                    raise
                token = _STEP.set("line_search")
                try:
                    depth_row = depth_sweep()
                finally:
                    _STEP.reset(token)
                evidence["depth_step"] = (
                    f"the {type(family).__name__} depth sweep overflowed under the "
                    f"fixed step and was repeated with the line search")
            if depth_row is not None:
                spent += depth_row.fits
                # A path still falling at the step cap has not fitted the
                # signal (strongly collinear columns make component-wise
                # boosting slow), and every probe at that depth would compare
                # unfitted models. Longer steps reach the fit in fewer of them.
                if (depth_row.at_boundary and depth_row.chosen >= _MSTOP_EXTENSION_CAP
                        and auto_nu and nu < _FAST_NU and budget - spent >= 3 * n_folds):
                    faster = probe_mstop(
                        data, response, formula, sweep_family, folds=fold_weights,
                        largest=max(int(mstop), 200), nu=_FAST_NU, env=env or None,
                        extend=min(_MAX_EXTENSIONS, (budget // 3 - n_folds) // n_folds),
                        fold_scheme=shared_scheme)
                    if faster is not None:
                        spent += faster.fits
                        evidence["nu"] = (
                            f"nu={_FAST_NU:g} — at nu={nu:g} the cross-validated risk "
                            f"was still falling at {depth_row.chosen} steps, so the "
                            f"steps were lengthened; every probe and rec.fit() use it")
                        nu = _FAST_NU
                        depth_row = faster
                probe_depth = max(int(depth_row.chosen), 50)
                depth_formula, depth_family = formula, sweep_family

    # A distributional candidate that passes the family probe's checks is not
    # adopted yet. Its shape, pair and residual-tree probes run on its first
    # parameter (`mu`) and are scored by the family's own held-out loss; the
    # mean family's probes then run in full; and the two finished models are
    # scored on the same folds (`compare_finished_routes`). `pending` holds
    # the distributional route while the mean route runs.
    pending = None
    # The step length the simple rivals of the whole-model check are fitted
    # with: the search's own, before a tree step shortens it.
    rival_nu = nu

    def run_lss_route(lss_family, family_evidence, fit_formula, lss_budget,
                      adopt_family):
        # The probes work on copies of the evidence, the measured list and the
        # defaulted list: the mean route writes the same keys, and only the
        # route that wins keeps its record.
        nonlocal spent
        route_evidence = dict(evidence)
        route_measured, route_defaulted = list(measured), list(defaulted)
        if adopt_family:
            route_evidence["family"] = family_evidence
            if "family" in route_defaulted:
                route_defaulted.remove("family")
            if "family" not in route_measured:
                route_measured.append("family")
        route_nu = nu
        if lss_budget > spent and modelled:
            family_evidence, fit_formula, route_nu, spent = _search_lss_mu(
                data, response, lss_family=lss_family,
                family_evidence=family_evidence, formula=formula,
                modelled=modelled, terms=terms, folds=fold_matrix(),
                scheme=shared_scheme, nu=nu, env=env or None, df=df,
                declared=declared, interactions=interactions,
                heredity=heredity, forbidden=forbidden,
                residual_tree=residual_tree, budget=lss_budget, spent=spent,
                mean_mstop=family_evidence.mean_mstop or probe_depth,
                seed=seed, n_rows=n_rows, defaulted=route_defaulted,
                measured=route_measured, evidence=route_evidence,
                judge=shared_judge)
            route_evidence["family"] = family_evidence
        return {"family": lss_family, "family_evidence": family_evidence,
                "formula": formula if fit_formula is None else fit_formula,
                "evidence": route_evidence, "measured": route_measured,
                "defaulted": route_defaulted, "nu": route_nu}

    def lss_recommendation(route, message):
        # The recommendation's fit() builds the distributional model at the
        # family's own step counts (#228, #322).
        route["evidence"]["search_stopped_at_family"] = message
        route["evidence"]["whole_model"] = {
            "compared": False,
            "reason": _WHOLE_MODEL_LSS_REASON}
        family_evidence = route["family_evidence"]
        lss_mstop = (max(family_evidence.lss_mstop)
                     if family_evidence.lss_mstop else mstop)
        route_evidence = route["evidence"]
        return SpecRecommendation(
            formula=route["formula"],
            learners=learners,
            family=route["family"],
            mstop=int(lss_mstop),
            response=response,
            hygiene=hygiene,
            evidence=route_evidence,
            measured=tuple(route["measured"]),
            priority=priority_order(
                route_evidence, hygiene, incomplete_rows=incomplete_rows,
                n_rows=len(response_column.values),
                missing_by_column={
                    column.name: column.n_missing
                    for column in modelled if column.n_missing},
                treatment=treatment_columns),
            defaulted=tuple(route["defaulted"]),
            declared=declared,
            env=env,
            groups=groups,
            folds_description=folds_description,
            incomplete_rows=incomplete_rows,
            fits=spent,
            reused=cache.reused,
            treatment=treatment_columns,
            control=boost_control(int(lss_mstop), route["nu"], step=step),
        )

    def start_lss(lss_family, family_evidence, fit_formula, mean_family,
                  adopt_family, message):
        """Start the distributional route. Returns the finished
        recommendation when the budget cannot pay for the comparison with the
        finished mean route (the candidate's checks stand), else `None`: the
        route is held in `pending` and the mean route runs next."""
        nonlocal pending, budget
        n_route_folds = int(np.asarray(fold_matrix()).shape[1])
        reserve = _COMPARISON_FOLD_FITS * n_route_folds
        remaining = budget - spent
        if remaining < reserve + _TWO_ROUTE_MIN_FOLDS * n_route_folds:
            route = run_lss_route(lss_family, family_evidence, fit_formula,
                                  budget, adopt_family)
            route["evidence"]["finished_routes"] = {
                "compared": False,
                "reason": (
                    f"{remaining} fits remained after the family probe, fewer "
                    f"than the {reserve + _TWO_ROUTE_MIN_FOLDS * n_route_folds} "
                    f"a comparison of the two finished routes needs "
                    f"({reserve} for the comparison, "
                    f"{_TWO_ROUTE_MIN_FOLDS * n_route_folds} for the probes), "
                    f"so the family probe's checks decided")}
            return lss_recommendation(route, message)
        # The comparison's fits are set aside; the rest is split evenly. The
        # distributional route spends first and what it leaves goes to the
        # mean route.
        share = (remaining - reserve) // 2
        before = spent
        route = run_lss_route(lss_family, family_evidence, fit_formula,
                              spent + share, adopt_family)
        pending = {"route": route, "family_evidence": family_evidence,
                   "mean_family": mean_family, "message": message,
                   "budget": budget, "reserve": reserve, "share": share,
                   "remaining": remaining, "lss_fits": spent - before,
                   "spent_before_mean": spent}
        budget -= reserve
        return None

    if budget > spent and family_was_inferred:
        fold_weights = fold_matrix()
        n_folds = int(np.asarray(fold_weights).shape[1])
        if budget - spent >= n_folds:
            evidence_row = probe_family(
                data, response, formula, family,
                folds=fold_weights, mstop=probe_depth, nu=nu,
                env=env or None, validate=budget >= 6 * n_folds,
                cache=cache, judge=shared_judge, fold_scheme=shared_scheme)
            if evidence_row is not None:
                family = evidence_row.recommended
                evidence["family"] = evidence_row
                measured.append("family")
                defaulted.remove("family")
                spent += evidence_row.fits
                if getattr(family, "param_names", None) is not None:
                    # The probe adopted a distributional family (#228).
                    finished = start_lss(
                        family, evidence_row, formula, evidence_row.mean_family
                        or Gaussian(), False,
                        f"the family probe adopted {type(family).__name__} on "
                        "its held-out log score, and the search stopped at the "
                        "family: the shape, pair and residual-tree probes ran on "
                        "its mu predictor, scored by its held-out log score; "
                        "deselection does not apply to a multi-parameter model. "
                        "rec.fit() fits the distributional model")
                    if finished is not None:
                        return finished
                    family = pending["mean_family"]
                    evidence["family"] = replace(evidence_row, recommended=family)

    # A Gaussian whose spread moves with a covariate is GaussianLSS: the
    # family probe assumes one spread for every row, so this second question
    # -- does the scale itself depend on a predictor -- is asked on its own
    # (#322). It runs only when the family is still Gaussian (an adopted
    # GammaLSS/BetaLSS or a measured GammaReg already answers it) and the
    # response's spread did not track the mean (that is Gamma's shape, not an
    # additive scale).
    if (budget > spent and family_was_inferred
            and type(family).__name__ == "Gaussian" and modelled):
        family_evidence = evidence.get("family")
        tracks = getattr(family_evidence, "spread_tracks_mean", float("nan"))
        if not (np.isfinite(tracks) and tracks > _SPREAD_TRACKS):
            fold_weights = fold_matrix()
            n_folds = int(np.asarray(fold_weights).shape[1])
            # Two sweeps are needed: one for this probe's out-of-fold mean and
            # one for the mstop sweep that still has to run. Spending the last
            # sweep here would leave mstop unmeasured.
            if budget - spent >= 2 * n_folds:
                het = probe_heteroscedasticity(
                    data, response, formula,
                    [column.name for column in modelled], terms,
                    folds=fold_weights, mstop=probe_depth, nu=nu,
                    env=env or None, validate=budget - spent >= 2 * n_folds,
                    cache=cache, judge=shared_judge, fold_scheme=shared_scheme)
                if het is not None:
                    evidence["heteroscedasticity"] = het
                    measured.append("heteroscedasticity")
                    spent += het.fits
                    if getattr(het.recommended, "param_names", None) is not None:
                        finished = start_lss(
                            het.recommended, het, het.lss_formula, family, True,
                            "the residual screen found the spread moves with a "
                            "covariate, and GaussianLSS's held-out log score "
                            "beat the mean-only Gaussian, and the search stopped "
                            "at the family: the shape, pair and residual-tree "
                            "probes ran on its mu predictor, scored by its "
                            "held-out log score; deselection does not apply to "
                            "a multi-parameter model. rec.fit() fits the "
                            "distributional model")
                        if finished is not None:
                            return finished

    # The final joint-tree comparison (below) runs last and costs a fixed
    # 3 * (folds + 2) fits. Without a reservation the earlier probes spend
    # the budget first and the step is skipped for lack of fits (#807: in
    # the joint-tree gate study the step was reached with more than 23 of
    # 32 fits spent on every bike_iid draw). Once the family is settled,
    # set its fits aside from every later budget draw and release them just
    # before the step. The reservation is taken only when the step's entry
    # checks can already pass and one mstop sweep's worth of fits remains
    # for the earlier steps; otherwise the step keeps its own skip rule.
    # The coordinate-surface step runs after this reservation is taken and
    # before it is released, and arbitrates against it at its call site
    # (#864): the reservation can be released to that step under the
    # conditions stated there.
    joint_reserve = 0
    if (joint_tree and budget > spent
            and _joint_tree_may_run(modelled, terms, family, declared=declared,
                                    mandatory=mandatory, forbidden=forbidden,
                                    heredity=heredity)):
        # Before the shape probe ``terms`` holds the dtype-driven terms (or
        # the prior's); the call site reads the same set from the final terms.
        reserve_folds = int(np.asarray(fold_matrix()).shape[1])
        cost = _joint_tree_cost(reserve_folds)
        if budget - spent >= cost + reserve_folds:
            joint_reserve = cost
            budget -= joint_reserve

    # Every relative comparison below measures risk from the saturated
    # model's, which is not zero for the log-link families.
    risk_floor = _risk_floor(family, data, response)

    # One preliminary ensemble supplies shapes, priority and pairs. It is
    # charged once and reused by the later interaction probe.
    exploration = None
    exploration_fits = 2 if preexploration == "dual" else 1
    if (preexploration != "off" and budget > spent
            and budget - spent >= exploration_fits + 3 * fold_matrix().shape[1]
            and n_rows >= 20 and modelled and exposure is None
            and type(family).__name__ not in ("Multinomial", "PropOdds")):
        # Preexploration fits its trees and nets without the exposure offset,
        # so under exposure= it proposes nothing; the shape and pair probes
        # then use the full catalogue.
        from mboostrs._autospec import _preexploration

        try:
            exploration = _preexploration(data, response, modelled, terms,
                                          seed=seed, engine=preexploration)
        except (ValueError, RuntimeError) as exc:
            if preexploration == "hessboost":
                raise
            evidence["preexploration_skipped"] = str(exc)
        if exploration is not None:
            exploration["shape_pruning"] = (preexploration != "neural"
                                            and type(family).__name__ in ("Gaussian", "GaussReg"))
            evidence["preexploration"] = exploration
            spent += exploration["fits"]
    elif preexploration != "off":
        evidence["preexploration_skipped"] = "insufficient budget/rows or unsupported response"

    # Reserve fits to pick a screened pair's form (tensor vs joint tree). The
    # screen is cheap; the held-out comparison costs n_folds per candidate,
    # and the post-shape mstop probe also draws from the budget. Without a
    # reservation, a tight budget spends everything before the pair can be
    # confirmed and screened pairs die "unvalidated" (measured on the
    # product-interaction panel at budget 32: the pair screened at
    # removed=0.055 but was discarded for lack of fits).
    n_folds = int(np.asarray(fold_matrix()).shape[1])
    pair_reserve = 4 * n_folds  # three picker candidates + the post-shape mstop probe
    # The form picker scores three candidates on the shared folds (the
    # additive incumbent plus the two pair forms), so its cost is exactly
    # 3*n_folds fits -- bounded and stated. That amount is earmarked here and
    # every later budget draw must leave it alone: the reserve the shape
    # probe leaves is otherwise spent by the post-shape mstop probe and the
    # deselection sweeps, and the picker starves (measured on the
    # product-interaction panel at budget 32: the screen ranked x0×x1 first
    # and the joint-tree lead was proposed, but the picker gate failed on
    # 4/8 replicates). In the worst case (adopt mode, no competing pair) up
    # to 3*n_folds fits go unspent -- the same posture as the shape probe's
    # unconditional carve-out above.
    picker_need = 3 * n_folds if interactions == "adopt" else 0

    if budget > spent and "shapes" in defaulted:
        fold_weights = fold_matrix()
        shape_evidence, shape_fits = probe_shapes(
            data, response, modelled, terms, family,
            folds=fold_weights, mstop=min(probe_depth, _MSTOP_EXTENSION_CAP),
            nu=nu, df=df, max_fits=max(n_folds, budget - spent - pair_reserve),
            fixed_names=declared,
            env=env or None, cache=cache, judge=shared_judge,
            fold_scheme=shared_scheme, depth=probe_depth,
            term_shares=(_term_shares(depth_row._fit, probe_depth, terms)
                         if depth_row is not None and depth_row._fit is not None
                         and formula == depth_formula else None),
            risk_floor=risk_floor,
            # A squared-error surrogate diagnoses response-scale shapes.
            # For other links it still proposes pairs, but cannot prune
            # the link-scale shape catalogue.
            preexploration=exploration if exploration and exploration["shape_pruning"] else None)
        if shape_evidence:
            chosen = {row.column: row for row in shape_evidence}
            rebuilt = []
            for column, term in zip(modelled, terms, strict=True):
                row = chosen.get(column.name)
                if row is None or row.chosen == row.incumbent:
                    rebuilt.append(term)
                else:
                    rebuilt.append(next(s.term for s in row.scores
                                        if s.shape == row.chosen))
            formula = f"{quote_response(response)} ~ " + " + ".join(rebuilt)
            learners = (None if any(_reads_transformed(term) for term in rebuilt)
                        else parse_gamboost_formula(formula)[1])
            terms = rebuilt
            evidence["shapes"] = tuple(shape_evidence)
            measured.append("shapes")
            defaulted.remove("shapes")
            spent += shape_fits

    chosen_mstop = int(mstop)
    main_fit = None
    if (depth_row is not None and formula == depth_formula and family is depth_family
            and "mstop" in defaulted
            and (not depth_row.at_boundary or depth_row.chosen >= _MSTOP_EXTENSION_CAP)):
        # The shape and family probes left the specification the depth sweep
        # tuned unchanged: its sweep, on the same folds, is the mstop evidence
        # (also when it ended at the cap, which a second sweep cannot pass).
        chosen_mstop = depth_row.chosen
        evidence["mstop"] = depth_row
        measured.append("mstop")
        defaulted.remove("mstop")
        main_fit = depth_row._fit
    if budget > spent and "mstop" in defaulted:
        fold_weights = fold_matrix()
        n_folds = int(np.asarray(fold_weights).shape[1])
        if budget - spent - picker_need >= n_folds:
            extensions = min(_MAX_EXTENSIONS,
                             (budget - spent - picker_need - n_folds) // n_folds)
            mstop_row = probe_mstop(
                data, response, formula, family,
                folds=fold_weights, largest=max(int(mstop), 200, probe_depth), nu=nu,
                env=env or None, extend=extensions, fold_scheme=shared_scheme)
            if mstop_row is not None:
                chosen_mstop = mstop_row.chosen
                evidence["mstop"] = mstop_row
                measured.append("mstop")
                defaulted.remove("mstop")
                spent += mstop_row.fits
                # The sweep already fitted the main-effect model on the full
                # data; the interaction screen truncates that path instead of
                # fitting it again.
                main_fit = mstop_row._fit

    if interactions not in ("report", "adopt"):
        raise ValueError('interactions must be "report" or "adopt"')
    # The screen reads a one-column residual; a Multinomial fit has one per
    # class, and no Kronecker wrapping for a pair term, so it is not screened.
    if type(family).__name__ == "Multinomial" and "interactions" in defaulted:
        defaulted.remove("interactions")
        evidence["interactions_skipped"] = (
            "not screened: a Multinomial fit has one residual per class and no "
            "pair terms")
    if (budget > spent and "interactions" in defaulted and len(modelled) > 1
            and type(family).__name__ != "Multinomial"):
        screen_counts = {}
        leads, screen_fits, screen_reused = probe_interactions(
            data, response, formula, family, modelled, terms,
            mstop=chosen_mstop, nu=nu, env=env or None, main_fit=main_fit,
            # Adopted pairs the tuned fit leaves unused are dropped by the
            # deselection step, so adoption can take more leads than a
            # report lists.
            top=_ADOPT_LEADS if interactions == "adopt" else 3,
            surrogate=exploration if exploration is not None else False, seed=seed,
            counts=screen_counts)
        evidence["interaction_candidates"] = screen_counts
        if forbidden:
            # A forbidden pair is neither listed nor adopted.
            kept_leads = [lead for lead in leads
                          if frozenset((lead.left, lead.right)) not in forbidden]
            if len(kept_leads) < len(leads):
                evidence["forbidden_interactions"] = tuple(
                    f"{lead.left} x {lead.right}" for lead in leads if lead not in kept_leads)
            leads = kept_leads
        evidence["interactions"] = leads
        measured.append("interactions")
        defaulted.remove("interactions")
        spent += screen_fits
        in_model = {column.name for column in modelled}
        pair_parents = {lead.term: (lead.left, lead.right) for lead in leads if lead.term}

        def heredity_allows(lead):
            # Strong heredity (Nelder's marginality): both main effects are in
            # the model; weak: at least one is.
            present = (lead.left in in_model) + (lead.right in in_model)
            return {"none": True, "weak": present >= 1, "strong": present == 2}[heredity]

        # Pairs with competing tensor/joint-tree candidates are set aside:
        # the two model the same joint effect, and the form is picked after
        # deselection (below), where the budget allows a held-out comparison.
        competing_pairs = set()
        if interactions == "adopt":
            by_kind = {}
            for lead in leads:
                if lead.kind in ("tensor", "joint_tree") and lead.term \
                        and heredity_allows(lead):
                    key = frozenset((lead.left, lead.right))
                    by_kind.setdefault(key, set()).add(lead.kind)
            competing_pairs = {key for key, kinds in by_kind.items()
                               if kinds == {"tensor", "joint_tree"}}
            if competing_pairs:
                leads = tuple(lead for lead in leads
                              if frozenset((lead.left, lead.right))
                              not in competing_pairs
                              or lead.kind not in ("tensor", "joint_tree"))
                pair_parents = {lead.term: (lead.left, lead.right)
                                for lead in leads if lead.term}
                evidence["competing_pairs"] = tuple(
                    "_x_".join(sorted(key)) for key in competing_pairs)

        adopted = [lead.term for lead in leads
                   if interactions == "adopt" and lead.kind != "duplicate-axis" and lead.term
                   and heredity_allows(lead)]
        if interactions == "adopt" and heredity != "none":
            barred = [lead.term for lead in leads
                      if lead.kind != "duplicate-axis" and lead.term and not heredity_allows(lead)]
            if barred:
                evidence["heredity_barred_interactions"] = tuple(barred)
        # O(n/constant) cap on offered pairs: with few rows the screen
        # passes chance pairs, and each pair term costs effective rows the
        # small sample cannot spare. The screen's top-6 cap binds first at
        # ordinary sizes.
        pair_cap = max(1, n_rows // _PAIRS_PER_ROW)
        if len(adopted) > pair_cap:
            evidence["pair_cap"] = (pair_cap, n_rows,
                                    tuple(t for t in adopted[pair_cap:]))
            adopted = adopted[:pair_cap]
        adopt_folds = fold_matrix()
        adopt_n_folds = int(np.asarray(adopt_folds).shape[1])
        validation_fits = adopt_n_folds * (1 + len(adopted))
        if adopted and budget - spent - picker_need < adopt_n_folds + validation_fits:
            evidence["unvalidated_interactions"] = tuple(adopted)
            adopted = []
        if adopted:
            before_adoption = (formula, learners, chosen_mstop, evidence.get("mstop"),
                               main_fit)
            formula = formula + " + " + " + ".join(adopted)
            evidence["adopted_interactions"] = tuple(adopted)
            learners = None       # the %X% terms are expanded when the fit parses the formula
            if budget - spent - picker_need >= adopt_n_folds:
                mstop_row = probe_mstop(
                    data, response, formula, family,
                    folds=adopt_folds, largest=max(int(mstop), 200, chosen_mstop), nu=nu,
                    env=env or None,
                    extend=max(0, min(_MAX_EXTENSIONS,
                                      (budget - spent - picker_need - validation_fits
                                       - adopt_n_folds) // adopt_n_folds)),
                    fold_scheme=shared_scheme)
                if mstop_row is not None:
                    chosen_mstop = mstop_row.chosen
                    evidence["mstop"] = mstop_row
                    spent += mstop_row.fits
                    main_fit = mstop_row._fit
                else:
                    main_fit = None
            # The screen measured each pair on the training residual. Keep an
            # adopted pair only if held-out risk says the model needs it:
            # removing it must cost 2% of the risk or clear the paired-SE
            # bar, which grows as 2*sqrt(log(candidates)) so a many-column
            # screen does not pass chance pairs on a fixed bar. With many
            # columns and few rows the screen passes chance pairs, which
            # the tuned fit then uses.
            if (main_fit is not None
                    and budget - spent - picker_need >= adopt_n_folds * (1 + len(adopted))):
                kept, rejected, adoption_evidence, spent = _held_out_pairs(
                    data, response, formula, adopted, family, adopt_folds,
                    min(2 * chosen_mstop, _MSTOP_EXTENSION_CAP), nu, env or None,
                    shared_scheme, spent, risk_floor=risk_floor,
                    n_candidates=len(adopted))
                evidence["adoption_evidence"] = adoption_evidence
                if rejected:
                    formula = f"{quote_response(response)} ~ " + " + ".join(terms + kept)
                    evidence["adopted_interactions"] = tuple(kept)
                    evidence["rejected_interactions"] = tuple(rejected)
                    if kept and budget - spent - picker_need >= adopt_n_folds:
                        mstop_row = probe_mstop(
                            data, response, formula, family, folds=adopt_folds,
                            largest=max(int(mstop), 200, chosen_mstop), nu=nu,
                            env=env or None,
                            extend=max(0, min(_MAX_EXTENSIONS,
                                              (budget - spent - picker_need
                                               - adopt_n_folds) // adopt_n_folds)),
                            fold_scheme=shared_scheme)
                        if mstop_row is not None:
                            chosen_mstop = mstop_row.chosen
                            evidence["mstop"] = mstop_row
                            spent += mstop_row.fits
                            main_fit = mstop_row._fit
                    elif not kept:
                        # Back to the additive specification and its sweep.
                        formula, learners, chosen_mstop, additive_row, main_fit = \
                            before_adoption
                        if additive_row is not None:
                            evidence["mstop"] = additive_row
            else:
                # A failed sweep or unexpectedly exhausted budget is not
                # evidence for a pair. Restore the last validated specification.
                evidence["unvalidated_interactions"] = tuple(adopted)
                evidence.pop("adopted_interactions", None)
                formula, learners, chosen_mstop, additive_row, main_fit = before_adoption
                if additive_row is not None:
                    evidence["mstop"] = additive_row
                else:
                    evidence.pop("mstop", None)

    offer_tree = residual_tree is True
    detection = None
    if residual_tree in (True, "auto") and type(family).__name__ != "Multinomial":
        # The screen-stage lead, recorded apart from the adoption decision.
        detection = _residual_tree_detection(data, response, modelled, main_fit)
        if detection is not None:
            evidence["residual_tree_detection"] = detection
    if residual_tree == "auto" and type(family).__name__ != "Multinomial":
        offer_tree, offer_reason = _should_offer_residual_tree(
            modelled, evidence, main_fit, data, response, detection=detection)
        if offer_tree:
            evidence["residual_tree_offer"] = offer_reason
    if offer_tree and type(family).__name__ != "Multinomial":
        formula, chosen_mstop, main_fit, spent = _residual_tree_step(
            data, response, formula, family, modelled, evidence, fold_matrix(),
            shared_scheme, chosen_mstop, main_fit, nu, env or None, budget, spent,
            risk_floor, int(mstop))
        if "adopted_tree" in evidence:
            measured.append("residual_tree")
            # The tree term is parsed from the formula when the fit runs; the
            # learner list built before it would drop the tree from rec.fit().
            learners = None
            if nu != evidence["residual_tree"]["nu"]:
                nu = evidence["residual_tree"]["nu"]
                evidence["nu"] = (
                    f"nu={nu:g} — the residual tree needs a smaller step than the "
                    f"additive terms; the recommendation and rec.fit() use it")

    # Retain eligible tree inputs before marginal deselection: a coordinate
    # with no main effect can still carry signal jointly with its neighbour.
    joint_modelled = _joint_tree_columns(modelled, terms)
    # The coordinate comparison reads the columns before marginal deselection:
    # a coordinate whose own main effect is small can still carry signal
    # through a surface over its pair (the same reason joint_modelled keeps it).
    pair_modelled, pair_terms = list(modelled), list(terms)

    # Shares of a fit stopped after a handful of steps say which terms moved
    # first, not which matter; deselection needs a tuned path of some length.
    deselect_share = (_DESELECT_THRESHOLD if deselect is True
                      else float(deselect) if deselect else 0.0)
    if (deselect_share and "mstop" in measured and main_fit is not None and budget > spent
            and len(terms) > 1 and chosen_mstop >= _DESELECT_MIN_MSTOP):
        adopted_terms = list(evidence.get("adopted_interactions", ()))
        if "adopted_tree" in evidence:
            adopted_terms.append(evidence["adopted_tree"])
        shares = _term_shares(main_fit, chosen_mstop, terms + adopted_terms)
        if shares is not None:
            # Every term but a declared one is a candidate: a main effect, and an
            # adopted pair, which the screen admitted on the training residual
            # and which the tuned fit may leave unused (a pair that shares a
            # column with a real one picks up a fraction of a percent). An
            # adopted tensor of two `bbs` margins spans both margins' main
            # effects, so a column that acts only through its pair loses
            # nothing when its main term goes.
            # Declared columns are the caller's statements about the data:
            # mechanism keywords, measurements= and groups= alike.
            kept_by_caller = (set(declared) | set(measurements or ()) | set(groups or ())
                              | set(mandatory))
            kept_main = set(kept_by_caller)
            if heredity == "strong":
                # Both parents of every adopted pair stay in the model; the
                # pair itself can still be dropped.
                for term in adopted_terms:
                    kept_main.update(pair_parents.get(term, ()))
            dropped = [(term, share) for term, share in shares.items()
                       if share < deselect_share
                       and not _is_declared(term, kept_by_caller if term in adopted_terms
                                            else kept_main)]
            if heredity == "weak" and adopted_terms:
                # Keep at least one parent of every adopted pair: when both
                # would go, the one with the larger share stays.
                column_term = {column.name: term for column, term
                               in zip(modelled, terms, strict=True)}
                for term in adopted_terms:
                    parents = [column_term[c] for c in pair_parents.get(term, ())
                               if c in column_term]
                    going = [(t, sh) for t, sh in dropped if t in parents]
                    if parents and len(going) == len(parents):
                        keep = max(going, key=lambda item: item[1])
                        dropped = [item for item in dropped if item != keep]
            if dropped and len(dropped) < len(shares):
                full_row = evidence.get("mstop")
                ordered = sorted(dropped, key=lambda item: item[1])
                fold_weights = fold_matrix()
                n_folds = int(np.asarray(fold_weights).shape[1])
                start = max(int(mstop), 200, chosen_mstop)

                def reduced(count):
                    gone = {term for term, _ in ordered[:count]}
                    pairs = [(column, term) for column, term
                             in zip(modelled, terms, strict=True) if term not in gone]
                    r_adopted = [term for term in adopted_terms if term not in gone]
                    r_terms = [term for _, term in pairs]
                    r_formula = f"{quote_response(response)} ~ " + " + ".join(r_terms + r_adopted)
                    r_learners = (
                        None if r_adopted or any(_reads_transformed(term) or "ecdf(" in term
                                                 for term in r_terms)
                        else (parse_gamboost_formula(r_formula, env=env or None)[1]
                              if learners is not None else None))
                    return (r_formula, r_learners, r_terms,
                            [column for column, _ in pairs], r_adopted)

                tries = {}

                def attempt(count):
                    # One cross-validated sweep of the formula without the
                    # `count` smallest terms; None when the budget is spent.
                    nonlocal spent
                    spec = reduced(count)
                    row = None
                    if budget - spent - picker_need >= n_folds:
                        row = probe_mstop(
                            data, response, spec[0], family, folds=fold_weights,
                            largest=start, nu=nu, env=env or None,
                            extend=max(0, min(_MAX_EXTENSIONS,
                                              (budget - spent - picker_need - n_folds)
                                              // n_folds)),
                            fold_scheme=shared_scheme)
                    if row is not None:
                        spent += row.fits
                    loss = None if row is None else _held_out_loss(full_row, row, risk_floor)
                    tries[count] = (spec, row, loss)
                    return loss

                # Drop every small term if the held-out risk allows it;
                # otherwise bisect for the most small terms that can go.
                accepted, refused = len(ordered), None
                loss_all = attempt(len(ordered))
                if loss_all is not None and loss_all > _DESELECT_HELD_OUT_LOSS:
                    accepted, refused = 0, len(ordered)
                    while refused - accepted > 1:
                        middle = (accepted + refused) // 2
                        loss = attempt(middle)
                        if loss is None:
                            break
                        if loss <= _DESELECT_HELD_OUT_LOSS:
                            accepted = middle
                        else:
                            refused = middle

                def listing(items):
                    return ", ".join(f"{term} {share:.2%}" for term, share in items)

                kept_note = ""
                if refused is not None:
                    kept_note = (
                        f"; {len(ordered) - accepted} more under "
                        f"{deselect_share:.0%} kept ({listing(ordered[accepted:])}): "
                        f"dropping them too raised the held-out risk by "
                        f"{tries[refused][2]:.1%}, more than "
                        f"{_DESELECT_HELD_OUT_LOSS:.0%}")
                if accepted == 0:
                    evidence["deselect"] = DeselectEvidence(
                        dropped=(), threshold=deselect_share,
                        held_out_loss=tries[refused][2],
                        sentence=(f"{len(ordered)} of {len(shares)} terms each carry "
                                  f"less than {deselect_share:.0%} of the risk "
                                  f"reduction ({listing(ordered)}), but dropping the "
                                  f"{refused} smallest raises the held-out risk by "
                                  f"{tries[refused][2]:.1%}, more than "
                                  f"{_DESELECT_HELD_OUT_LOSS:.0%}: all kept"))
                else:
                    (formula, learners, terms, modelled, adopted_terms), row, loss = \
                        tries[accepted]
                    if "adopted_interactions" in evidence:
                        evidence["adopted_interactions"] = tuple(adopted_terms)
                    if row is not None:
                        chosen_mstop = row.chosen
                        evidence["mstop"] = row
                        main_fit = row._fit
                    checked = ("" if loss is None else
                               f"; dropping them changed the held-out risk by "
                               f"{round(loss, 3) + 0.0:+.1%}")
                    evidence["deselect"] = DeselectEvidence(
                        dropped=tuple((term, round(float(share), 6))
                                      for term, share in ordered[:accepted]),
                        threshold=deselect_share, held_out_loss=loss,
                        sentence=(f"{accepted} of {len(shares)} terms dropped for "
                                  f"less than {deselect_share:.0%} of the risk "
                                  f"reduction ({listing(ordered[:accepted])}){checked}"
                                  f"{kept_note}; mstop tuned again for the rest"))

    # A coordinate pair gets its own measured choice between a tensor surface,
    # a tree, or both, against the additive pair. It runs before the joint
    # tree, which then compares its candidates with the incumbent it leaves.

    # Screened pairs with competing tensor/joint-tree candidates (#267):
    # pick the better form on shared-fold held-out risk by the shape rule,
    # and adopt it only if it beats the additive incumbent by the same rule.
    # The screen measures the pair on the training residual, so its
    # validation is not held out; without the incumbent comparison the picker
    # adopts chance pairs the screen passes on additive truth (measured:
    # 6/8 false-positive replicates on the smooth-additive panel). Runs
    # after deselection so the incumbent is the clean additive specification.
    # On a single supplied holdout the adoption below also spends two fold
    # fits on a per-row audit of the winning form; reserve them up front,
    # as the residual-tree step does.
    from mboostrs._autospec import _single_fold_design as _picker_single_design

    picker_audit_reserve = (2 if interactions == "adopt"
                            and _picker_single_design(
                                fold_matrix(), shared_scheme, family)
                            else 0)
    if (interactions == "adopt" and evidence.get("competing_pairs")
            and budget - spent >= picker_need + picker_audit_reserve):
        from mboostrs._autospec import _SHAPE_SE, _held_out_pair_z, coordinate_beats, score_coordinate_candidates
        all_leads = evidence.get("interactions", ())
        for pair_key in evidence["competing_pairs"]:
            key = frozenset(pair_key.split("_x_"))
            tensor_term = next(
                (lead.term for lead in all_leads
                 if frozenset((lead.left, lead.right)) == key
                 and lead.kind == "tensor"), None)
            tree_term = next(
                (lead.term for lead in all_leads
                 if frozenset((lead.left, lead.right)) == key
                 and lead.kind == "joint_tree"), None)
            if not tensor_term or not tree_term:
                continue
            candidates = {
                "additive": formula,
                "tensor": f"{formula} + {tensor_term}",
                "joint_tree": f"{formula} + {tree_term}",
            }
            # The competing forms are searched on the candidate-path policy:
            # floored by the requested budget and the house 200, never only
            # by the additive incumbent's early optimum (#536).
            picker_cap = _candidate_path_cap(mstop, chosen_mstop)
            scores, form_fits = score_coordinate_candidates(
                data, response, candidates, family, fold_matrix(),
                picker_cap, nu,
                env=env or None, fold_scheme=shared_scheme)
            spent += form_fits
            tree_beats_tensor, _, _ = coordinate_beats(
                scores["tensor"], scores["joint_tree"], risk_floor)
            tensor_beats_tree, _, _ = coordinate_beats(
                scores["joint_tree"], scores["tensor"], risk_floor)
            if tree_beats_tensor and not tensor_beats_tree:
                pair_kind = "joint_tree"
            else:
                pair_kind = "tensor"
            pair_term = (tree_term if pair_kind == "joint_tree"
                         else tensor_term)
            warranted, held_out_gain, paired_z = coordinate_beats(
                scores["additive"], scores[pair_kind], risk_floor)
            if (warranted and picker_audit_reserve
                    and scores["additive"].stopping is not None
                    and scores[pair_kind].stopping is not None
                    and scores["additive"].stopping.chosen >= 1
                    and scores[pair_kind].stopping.chosen >= 1):
                # One fold: the paired fold z above is inert, so the
                # aggregate margin alone would decide the adoption --
                # the bare-margin route the residual-tree step guards
                # against. Read the per-row differences on the same
                # held-out rows instead, each form at its own selected
                # step, and require the bar the fold route would have
                # used; the evidence then reports the per-row z in
                # place of the fold one. Families whose loss couples
                # rows have no per-row losses and keep the margin rule.
                row_z, _row_gain, _row_se, audit_fits = _held_out_pair_z(
                    data, response, candidates["additive"],
                    scores["additive"].stopping.chosen, picker_cap, nu,
                    candidates[pair_kind],
                    scores[pair_kind].stopping.chosen, picker_cap, nu,
                    family, fold_matrix(), env=env or None)
                spent += audit_fits
                if row_z is not None:
                    paired_z = row_z
                    warranted = row_z >= _SHAPE_SE
            evidence[f"pair_form_{pair_key}"] = {
                "additive_risk": scores["additive"].risk,
                "tensor_risk": scores["tensor"].risk,
                "joint_tree_risk": scores["joint_tree"].risk,
                "winner": pair_kind if warranted else "neither",
                "picker_fits": form_fits,
                "held_out_gain": held_out_gain,
                "paired_z": paired_z,
                "folds": len(scores[pair_kind].fold_risk),
            }
            if not warranted:
                continue
            # Adoption was justified at this form's own selected step. Replay
            # that exact path rather than the previous additive model's step
            # and evidence; no additional CV fits are needed (#535).
            row = scores[pair_kind].stopping
            chosen_mstop = row.chosen
            evidence["mstop"] = row
            main_fit = row._fit
            evidence[f"pair_form_{pair_key}"].update(
                mstop=row.chosen, search_cap=row.grid[-1], at_boundary=row.at_boundary)
            if "mstop" not in measured:
                measured.append("mstop")
            if "mstop" in defaulted:
                defaulted.remove("mstop")
            formula = candidates[pair_kind]
            evidence["adopted_interactions"] = (
                evidence.get("adopted_interactions", ()) + (pair_term,))
            # The form picker already validated the term against the
            # incumbent. Preserve that comparison for the same report and
            # result accessors used by the earlier adoption route.
            evidence.setdefault("adoption_evidence", {})[pair_term] = (
                len(scores[pair_kind].fold_risk), held_out_gain, paired_z, _SHAPE_SE)
            learners = None
            measured.append("pair_form")
            # Only the top competing pair; the screen caps leads at 6 and the
            # joint-tree companion is only added for the top-1 tensor lead.
            break

    if (joint_tree and budget > spent
            and type(family).__name__ not in ("Multinomial", "PropOdds")):
        # Arbitration between this step and the joint-tree reservation
        # (#864). The reservation was taken before the earlier steps spent,
        # without regard to this step's claim on the same fits: at budgets
        # 60 and 70 on the coordinate fixture it left this step unpaid
        # while the joint tree -- the weaker of the two specifications on
        # that data -- was adopted. The reservation is therefore released
        # to this step when three things hold: this step cannot be paid
        # from the unreserved remainder, it can be paid once the
        # reservation is released, and the budget outside the reservation
        # is itself at least this step's full cost. The third condition
        # keeps the reservation intact where the surface step was never
        # affordable outside it (at budget 40 on the same fixture the
        # unreserved budget is 19 against this step's cost of 32, and the
        # reservation is the only route to the fixture's interaction).
        # When the reservation is released and spent, the joint-tree step
        # keeps its own budget check and skips if the fits are gone; the
        # release is recorded in this step's evidence.
        surface_lent = 0
        if joint_reserve:
            from mboostrs._autospec import _single_fold_design as _surface_single_design

            surface_folds = int(np.asarray(fold_matrix()).shape[1])
            surface_audit = (2 if _surface_single_design(
                fold_matrix(), shared_scheme, family) else 0)
            surface_cost = _coordinate_surface_cost(surface_folds, surface_audit)
            if (budget - spent < surface_cost
                    and budget + joint_reserve - spent >= surface_cost
                    and budget >= surface_cost):
                surface_lent = joint_reserve
                budget += joint_reserve
                joint_reserve = 0
        spent_before_surface = spent
        formula, surface_row, spent = _coordinate_surface_step(
            data, response, formula, family, pair_modelled, pair_terms, evidence,
            fold_matrix(), shared_scheme, chosen_mstop, nu, main_fit, env or None,
            budget, spent, df=df, declared=declared, mandatory=mandatory,
            forbidden=forbidden, heredity=heredity, risk_floor=risk_floor)
        if surface_lent:
            if spent == spent_before_surface:
                # The step stood without spending (no coordinate pair, a
                # declared pair, a constraint): the reservation was not
                # used, so it stands again for the joint-tree step.
                budget -= surface_lent
                joint_reserve = surface_lent
            elif evidence.get("coordinate_surface") is not None:
                evidence["coordinate_surface"]["joint_reserve_released"] = surface_lent
        if evidence.get("coordinate_surface", {}).get("candidates"):
            measured.append("coordinate_surface")
        if surface_row is not None:
            learners = None      # the spatial/tree terms are parsed when the fit runs
            chosen_mstop = surface_row.chosen
            evidence["mstop"] = surface_row
            main_fit = surface_row._fit

    # Release the joint-tree reservation: the step is the only remaining
    # spender on the mean route.
    budget += joint_reserve
    if (joint_tree and budget > 0 and type(family).__name__ in ("Gaussian", "GaussReg")
            and _has_coordinate_pair(joint_modelled)):
        formula, learners, chosen_mstop, nu, main_fit, spent = _joint_tree_step(
            data, response, formula, learners, family, joint_modelled, evidence,
            fold_matrix(), shared_scheme, chosen_mstop, nu, main_fit, env or None,
            budget, spent, declared=declared, mandatory=mandatory,
            forbidden=forbidden, heredity=heredity, risk_floor=risk_floor)
        if joint_reserve:
            evidence["joint_tree"]["reserved"] = joint_reserve
        if evidence.get("joint_tree", {}).get("candidates"):
            measured.append("joint_tree")
        if evidence.get("joint_tree", {}).get("adopted"):
            if "shapes" not in measured:
                measured.append("shapes")
            if "shapes" in defaulted:
                defaulted.remove("shapes")
            if "mstop" not in measured:
                measured.append("mstop")
            if "mstop" in defaulted:
                defaulted.remove("mstop")

    if pending is not None:
        # Both routes are finished: the mean route above, the distributional
        # route held since the family check. The distributional model is
        # adopted when it passes the log-score check and the mean check
        # against the finished mean model on the same folds.
        budget = pending["budget"]
        route = pending["route"]
        route_family = route["family_evidence"]
        comparison = compare_finished_routes(
            data, response, mean_family=family, mean_formula=formula,
            mean_mstop=chosen_mstop, mean_nu=nu, lss_family=route["family"],
            lss_formula=route["formula"], lss_nu=route["nu"],
            lss_largest=max(int(route_family.lss_largest or 0), 50),
            folds=fold_matrix(), fold_scheme=shared_scheme,
            env=env or None, cache=cache, judge=shared_judge)
        mean_name = type(family).__name__
        lss_name = type(route["family"]).__name__
        record = {
            "compared": comparison is not None,
            "mean_route_family": mean_name,
            "mean_route_formula": formula,
            "mean_route_mstop": int(chosen_mstop),
            "lss_route_formula": route["formula"],
            "budget": {
                "total": int(pending["budget"]),
                "remaining_at_family_check": int(pending["remaining"]),
                "set_aside_for_comparison": int(pending["reserve"]),
                "distributional_route_share": int(pending["share"]),
                "distributional_route_fits": int(pending["lss_fits"]),
                "mean_route_fits": int(spent - pending["spent_before_mean"]),
                "comparison_fits": (int(comparison["fits"])
                                    if comparison is not None else 0)}}
        if comparison is not None:
            spent += comparison["fits"]
            record.update({key: comparison[key] for key in (
                "lss_score", "mean_score", "score_se", "lss_mse", "mean_mse",
                "mse_se", "log_score_check", "mean_check", "lss_mstop")})
            record["winner"] = ("distributional" if comparison["winner"] == "lss"
                                else "mean")
            text = (
                f"after the shape, pair and residual-tree probes ran on both "
                f"routes, {lss_name} scores {comparison['lss_score']:.3f} "
                f"nats/row against {comparison['mean_score']:.3f} for "
                f"{mean_name} (standard error of the difference "
                f"{comparison['score_se']:.2g}), and the squared error of its "
                f"mean is {comparison['lss_mse']:.3g} against "
                f"{comparison['mean_mse']:.3g} (standard error "
                f"{comparison['mse_se']:.2g}); log-score check "
                f"{comparison['log_score_check']}, mean check "
                f"{comparison['mean_check']}, so "
                + (lss_name if comparison["winner"] == "lss" else mean_name))
        else:
            record["winner"] = "mean"
            text = (f"the two finished routes could not be scored on the same "
                    f"folds, so {mean_name} stands")
        record["sentence"] = text
        if record["winner"] == "distributional":
            route_family = replace(
                route_family, lss_mstop=comparison["lss_mstop"],
                lss_at_boundary=comparison["lss_at_boundary"],
                lss_largest=comparison["lss_largest"],
                sentence=route_family.sentence + "; " + text)
            route["family_evidence"] = route_family
            route["evidence"]["family"] = route_family
            route["evidence"]["finished_routes"] = record
            return lss_recommendation(route, pending["message"])
        evidence["finished_routes"] = record
        if "family" in evidence:
            evidence["family"] = replace(
                evidence["family"],
                sentence=evidence["family"].sentence + "; " + text)

    if budget > 0:
        formula, chosen_mstop, nu, replaced_by, spent = _whole_model_step(
            data, response, formula, family, evidence, fold_matrix(),
            shared_scheme, chosen_mstop, nu, start_formula, rival_nu,
            env or None, budget, spent, int(mstop))
        if evidence["whole_model"]["compared"]:
            measured.append("whole_model")
        if replaced_by is not None:
            # The search's records stay as its history; the residual tree is
            # no longer a term of the returned formula. A transform or a
            # `%X%` term is expanded when the fit parses the formula.
            rival_terms = _split_plus(partition_tilde(formula)[2])
            learners = (None if any(_reads_transformed(term) or "%" in term
                                    for term in rival_terms)
                        else parse_gamboost_formula(formula, env=env or None)[1])
            evidence.pop("adopted_tree", None)
            if "mstop" not in measured:
                measured.append("mstop")
            if "mstop" in defaulted:
                defaulted.remove("mstop")

    # Bounded step-length search (#900): every probe above fits at the one
    # step length, so the finished specification is re-read at a small grid
    # of other step lengths and the cross-validated best is kept. Each grid
    # value costs one path per fold -- the unit the budget already counts --
    # and the search runs only on fits the budget still holds, so its cost
    # is charged to `spent` like every other probe's. A step length the
    # caller gave is used as given and never searched; a distributional
    # family is tuned by its own route and is not searched here either;
    # and a step length a tree step set for the finished specification
    # (the residual tree's or the joint tree's, both of which choose a
    # small step deliberately because a deep tree term needs one) stands:
    # the search tunes the route's own step length, it does not override
    # a step length chosen for the specification's structure.
    if (budget > spent and auto_nu and step == "fixed"
            and getattr(family, "param_names", None) is None
            and "adopted_tree" not in evidence
            and not (isinstance(evidence.get("joint_tree"), dict)
                     and evidence["joint_tree"].get("adopted"))
            and "mstop" in measured and evidence.get("mstop") is not None):
        from mboostrs._autospec import probe_nu_search

        nu_row, nu_best, nu_fits, nu_records = probe_nu_search(
            data, response, formula, family, folds=fold_matrix(),
            incumbent=evidence["mstop"], nu=nu, env=env or None,
            fold_scheme=shared_scheme, max_fits=budget - spent)
        if nu_fits:
            spent += nu_fits
            evidence["nu_search"] = {
                "grid": tuple(record["nu"] for record in nu_records),
                "candidates": nu_records,
                "fits": int(nu_fits),
                "incumbent_nu": float(nu_records[0]["nu"]),
                "chosen_nu": float(nu_best),
                "adopted": bool(nu_best != nu),
                "sentence": (
                    f"step lengths {', '.join(format(r['nu'], 'g') for r in nu_records)} "
                    f"were probed on the finished specification at "
                    f"{nu_fits} fits; nu={nu_best:g} read the lowest "
                    f"cross-validated risk "
                    f"({float(nu_row.risk[nu_row.chosen]):.6g} at "
                    f"{nu_row.chosen} steps)")}
            if nu_best != nu:
                evidence["nu"] = (
                    f"nu={nu_best:g} — the bounded step-length search read a "
                    f"lower cross-validated risk there than at "
                    f"nu={nu:g} ({float(nu_records[0]['risk']):.6g} at "
                    f"{nu_records[0]['mstop']} steps); every earlier probe "
                    f"ran at nu={nu:g}, and rec.fit() uses nu={nu_best:g}")
                nu = nu_best
                chosen_mstop = nu_row.chosen
                evidence["mstop"] = nu_row

    reused = cache.reused + screen_reused
    # Whether each two-column term is a tensor surface, and whether its main
    # effects are in the formula; recorded for the report and the export.
    evidence["interaction_structure"] = interaction_structure(formula)
    return SpecRecommendation(
        formula=formula,
        learners=learners,
        family=family,
        mstop=chosen_mstop,
        response=response,
        hygiene=hygiene,
        evidence=evidence,
        exposure=exposure,
        measured=tuple(measured),
        priority=priority_order(
            evidence, hygiene, incomplete_rows=incomplete_rows,
            n_rows=len(response_column.values),
            missing_by_column={column.name: column.n_missing for column in modelled
                               if column.n_missing},
            treatment=treatment_columns),
        defaulted=tuple(defaulted),
        declared=declared,
        env=env,
        groups=groups,
        measurements=tuple(measurements),
        folds_description=folds_description,
        incomplete_rows=incomplete_rows,
        fits=spent,
        reused=reused,
        treatment=treatment_columns,
        control=boost_control(chosen_mstop, nu, step=step),
    )


#: A term under this share of the risk reduction is dropped (Strömer et al.
#: 2022 use 1%).
_DESELECT_THRESHOLD = 0.01

#: ... unless dropping the small terms together raises the minimum of the
#: cross-validated risk, on the same folds, by more than this fraction.
#: Twenty terms of 0.5% each carry a tenth of the fit: on cpu_activity
#: dropping 16 of 21 such terms raised the held-out risk by 67% and took
#: test R^2 from 0.984 to 0.972. Measured against the risk rather than the
#: risk reduction, which on a response the model explains almost entirely
#: makes that loss look like 1%.
_DESELECT_HELD_OUT_LOSS = 0.01


def _joint_tree_cost(n_folds):
    """Fits the joint-tree comparison spends: three sweeps (the incumbent and
    two tree depths), each of ``n_folds`` fold paths plus the preparation fit
    and the final full-data fit that Gaussian tune_mstop also makes."""
    return 3 * (n_folds + 2)


def _coordinate_surface_cost(n_folds, audit_reserve=0):
    """Fits the coordinate-surface comparison needs before it can run: five
    candidate sweeps and the final sweep of the adopted formula, stated as
    ``6 * n_folds + 2``, plus the single-holdout per-row audit's reserve
    when the fold design calls for it."""
    return 6 * n_folds + 2 + audit_reserve


def _joint_tree_columns(modelled, terms):
    """The columns the final joint-tree step reads, in ``modelled`` order.

    A column is kept when it is numeric and its term in ``terms`` (aligned
    with ``modelled``) is a ``bbs``, ``bols``, ``btree`` or ``bns`` term that
    reads that column alone. A numeric column coded as groups (``brandom``),
    held by a shape-constrained or spatial learner, or read together with
    another column (a ``by=`` argument) keeps its term and is not offered to
    the joint tree. The reservation check and the call site both read this
    set, so the reservation is taken only for a step the call site enters.
    """
    from mboostrs.formula import term_variables

    return [c for c, t in zip(modelled, terms, strict=True)
            if c.kind == "numeric" and term_variables(t) == (c.name,)
            and t.startswith(("bbs(", "bols(", "btree(", "bns("))]


def _joint_tree_may_run(modelled, terms, family, *, declared, mandatory, forbidden,
                        heredity):
    """Whether the final joint-tree step's entry checks can pass.

    The checks read what is known before the search spends: the family, the
    interaction constraints, and a coordinate pair among the columns
    ``_joint_tree_columns`` keeps from ``modelled`` and ``terms`` that the
    caller did not declare or make mandatory. The step repeats its own
    checks on the specification it is handed, which can still refuse when
    the shape probe, run after the reservation, gives a coordinate a term
    outside ``_joint_tree_columns``; its reserved fits then go unspent.
    """
    if type(family).__name__ not in ("Gaussian", "GaussReg") or heredity != "none":
        return False
    protected = set(declared) | set(mandatory)
    names = {c.name for c in _joint_tree_columns(modelled, terms)
             if c.name not in protected}
    if any(pair <= names for pair in forbidden):
        return False
    return _has_coordinate_pair([c for c in modelled if c.name in names])


def _joint_tree_step(data, response, formula, learners, family, modelled, evidence,
                     folds, scheme, chosen_mstop, nu, main_fit, env, budget, spent,
                     *, declared, mandatory, forbidden, heredity, risk_floor):
    """Compare the current specification with a joint tree, preserving declarations.

    This final comparison replaces the free main effects and interactions
    together, rather than adding a residual learner to an additive fit.
    All fitting and CV are delegated to the native tuning route.
    """
    from mboostrs._autospec import _relative_gain
    from mboostrs.explore import split_terms
    from mboostrs.formula import term_variables

    unchanged = (formula, learners, chosen_mstop, nu, main_fit)
    report = {"adopted": False, "candidates": [], "incumbent": formula}
    evidence["joint_tree"] = report
    protected = set(declared) | set(mandatory)
    incumbent_terms = split_terms(partition_tilde(formula)[2])
    # A multi-column declared term protects all its inputs, including the
    # other margin of a declared interaction.
    while True:
        before = set(protected)
        for term in incumbent_terms:
            used = set(term_variables(term))
            if used & protected:
                protected.update(used)
        if protected == before:
            break
    columns = [c for c in modelled if c.kind == "numeric" and c.name not in protected]
    names = [c.name for c in columns]
    if not _has_coordinate_pair(columns):
        report["reason"] = "the coordinate pair is declared or unavailable for replacement"
        return (*unchanged, spent)
    if heredity != "none" or any(pair <= set(names) for pair in forbidden):
        report["reason"] = "a joint replacement would bypass the interaction constraints"
        return (*unchanged, spent)
    n_folds = int(np.asarray(folds).shape[1])
    per_sweep = n_folds + 2
    # The incumbent keeps its own step length; deep trees use a smaller one.
    # Compare at the same maximum grid, with each path choosing its optimum.
    # Reserve the whole comparison before making any fit.
    if budget - spent < _joint_tree_cost(n_folds):
        report["reason"] = "insufficient budget for the incumbent and two joint-tree depths"
        return (*unchanged, spent)
    tree_nu = min(nu, _RESIDUAL_TREE_NU)
    previous = evidence.get("mstop")
    largest = min(_MSTOP_EXTENSION_CAP,
                  max(200, chosen_mstop, len(previous.risk) - 1 if previous else 0))
    base = probe_mstop(data, response, formula, family, folds=folds, largest=largest,
                       nu=nu, env=env, fold_scheme=scheme)
    # A failed probe may have fitted part of its folds; charge the reservation.
    spent += per_sweep if base is None else base.fits + 2
    if base is None:
        report["reason"] = "the incumbent's sweep failed"
        return (*unchanged, spent)
    report["incumbent_risk"] = base.risk[base.chosen]
    report["nu"] = tree_nu
    report["incumbent_nu"] = nu
    report["largest"] = largest
    kept = [t for t in incumbent_terms if not term_variables(t)
            or not set(term_variables(t)) <= set(names)]
    best = None
    for depth in (3, 6):
        term = f"btree({', '.join(quote_name(n) for n in names)}, maxdepth={depth}, selection=sse)"
        candidate = f"{quote_response(response)} ~ " + " + ".join(kept + [term])
        row = probe_mstop(data, response, candidate, family, folds=folds,
                          largest=largest, nu=tree_nu, env=env, fold_scheme=scheme)
        spent += per_sweep if row is None else row.fits + 2
        gain = (None if row is None else
                _relative_gain(base.risk[base.chosen], row.risk[row.chosen], risk_floor))
        report["candidates"].append({"formula": candidate, "term": term,
                                     "risk": None if row is None else row.risk[row.chosen],
                                     "mstop": None if row is None else row.chosen,
                                     "relative_gain": gain,
                                     "at_boundary": None if row is None else row.at_boundary})
        if gain is not None and gain >= _SHAPE_MARGIN_PAIR:
            if best is None or gain > best[0]:
                best = (gain, candidate, row)
    if best is None:
        report["reason"] = "neither joint tree lowers the shared-fold risk by at least 2%"
        return (*unchanged, spent)
    gain, candidate, row = best
    report.update(adopted=True, formula=candidate, relative_gain=gain)
    evidence["mstop"] = row
    evidence["nu"] = f"nu={tree_nu:g} — the shared-fold joint-tree comparison's step length"
    return candidate, None, row.chosen, tree_nu, row._fit, spent


def _coordinate_surface_step(data, response, formula, family, modelled, terms, evidence,
                             folds, scheme, chosen_mstop, nu, main_fit, env, budget, spent,
                             *, df, declared, mandatory, forbidden, heredity, risk_floor):
    """Compare ``bspatial(x, y)`` with ``btree(x, y)`` for a coordinate pair.

    The pair's single-column terms are the incumbent's main effects; a term
    that marginal deselection already dropped is absent from the incumbent.
    Candidates are cross-validated on the shared folds: the additive pair (the
    incumbent), a tensor surface ``bspatial(x, y)``, a stump ``btree(x, y)``, a
    depth-3 tree ``btree(x, y, maxdepth=3)``, and the tensor with the chosen
    tree. The deeper tree replaces the stump, and the tree replaces the tensor,
    by the shape rule (`coordinate_beats`); the combination replaces the
    surface only by the two-standard-error rule; the surface replaces the
    additive pair when it beats the pair by the shape rule. Returns
    ``(formula, row, spent)``: ``row`` is the step length's `MstopEvidence` of
    the adopted formula, or None when the additive pair stands. The decision
    and every candidate's risk go to ``evidence["coordinate_surface"]``.
    On a single supplied holdout the paired fold route cannot fire (one
    fold), so the surface's adoption additionally requires the per-row
    paired z of the held-out losses to clear the same 2.0 bar, when the
    family's loss admits per-row losses.
    """
    from mboostrs._autospec import (
        _SHAPE_SE,
        _held_out_pair_z,
        _single_fold_design,
        coordinate_beats,
        score_coordinate_candidates,
    )
    from mboostrs.explore import split_terms
    from mboostrs.formula import term_variables

    names = _coordinate_pair_names(modelled)
    if names is None:
        return formula, None, spent
    x, y = names
    report = {"pair": (x, y), "candidates": [], "adopted": False, "reason": ""}
    evidence["coordinate_surface"] = report

    def stand(reason):
        report["reason"] = reason
        return formula, None, spent

    if {x, y} & (set(declared) | set(mandatory)):
        return stand("the pair is declared with coordinates= or mandatory=, so the "
                     "caller's terms stand")
    if heredity != "none":
        return stand("a coordinate surface would bypass the heredity rule")
    if frozenset((x, y)) in forbidden:
        return stand("the pair is forbidden as an interaction, and a surface over it is one")
    n_folds = int(np.asarray(folds).shape[1])
    # Five candidate sweeps (additive, tensor, two trees, their combination),
    # then the final sweep of the adopted formula. A single supplied
    # holdout also spends two fold fits on the per-row audit of the
    # adoption (below); reserve them up front, as the residual-tree
    # step does.
    audit_reserve = 2 if _single_fold_design(folds, scheme, family) else 0
    if budget - spent < _coordinate_surface_cost(n_folds, audit_reserve):
        return stand("insufficient budget for five candidate sweeps and the final sweep")

    parts = split_terms(partition_tilde(formula)[2])
    # A coordinate whose main effect marginal deselection dropped has no term
    # to replace; the surface then competes with the additive model without it.
    mains = {}
    for name in (x, y):
        matches = [term for column, term in zip(modelled, terms, strict=True)
                   if column.name == name and term in parts
                   and term_variables(term) == (name,)]
        if matches:
            mains[name] = matches[0]
    rest = [term for term in parts if term not in mains.values()]
    if any({x, y} <= set(term_variables(term)) for term in rest):
        return stand("the pair already carries an interaction term, so a surface would "
                     "duplicate it")

    spatial_df = max(float(df), 6.0)
    surface_term = f"bspatial({quote_name(x)}, {quote_name(y)}, df={spatial_df})"
    # Stumps (the default btree) add one split per step, so their sum is
    # additive in x and y; a depth-3 tree can carry a rectangle's corner. The
    # stump stands unless the deeper tree beats it by the shape rule.
    stump_term = f"btree({quote_name(x)}, {quote_name(y)})"
    deep_term = f"btree({quote_name(x)}, {quote_name(y)}, maxdepth=3)"

    def with_pair(extra):
        return f"{quote_response(response)} ~ " + " + ".join(rest + extra)

    candidates = {
        "additive": formula,
        "bspatial": with_pair([surface_term]),
        "btree": with_pair([stump_term]),
        "btree3": with_pair([deep_term]),
    }
    labels = {"additive": "the additive pair", "bspatial": surface_term,
              "btree": stump_term, "btree3": deep_term}
    largest = min(_MSTOP_EXTENSION_CAP, max(200, int(chosen_mstop)))
    scores, fits = score_coordinate_candidates(
        data, response, candidates, family, folds, largest, nu,
        env=env, fold_scheme=scheme)
    spent += fits
    floor = risk_floor
    additive = scores["additive"]

    deeper_wins, _, _ = coordinate_beats(scores["btree"], scores["btree3"], floor)
    tree = scores["btree3"] if deeper_wins else scores["btree"]
    tree_term = labels[tree.name]
    # The combination of the tensor and the chosen tree: one more sweep.
    both_name, both_term = "both", f"{surface_term} + {tree_term}"
    candidates = {both_name: with_pair([surface_term, tree_term])}
    both_scores, fits = score_coordinate_candidates(
        data, response, candidates, family, folds, largest, nu,
        env=env, fold_scheme=scheme)
    spent += fits
    scores[both_name] = both_scores[both_name]
    labels[both_name] = both_term
    for name in ("additive", "bspatial", "btree", "btree3", both_name):
        _, gain, z = coordinate_beats(additive, scores[name], floor)
        report["candidates"].append({
            "name": name, "term": labels[name], "risk": scores[name].risk,
            "relative_gain": gain, "z": z})

    # The tree replaces the tensor surface by the shape rule; the two together
    # replace the chosen surface only when they beat it by two paired SE.
    tree_beats_surface, _, _ = coordinate_beats(scores["bspatial"], tree, floor)
    surface = tree if tree_beats_surface else scores["bspatial"]
    both_beats, _, _ = coordinate_beats(surface, scores[both_name], floor, margin=False)
    chosen = scores[both_name] if both_beats else surface
    report.update(surface=surface.name, both_beats_surface=both_beats,
                  tree=tree.name, tree_beats_surface=tree_beats_surface, chosen=chosen.name)
    replaces, gain, z = coordinate_beats(additive, chosen, floor)
    audited_z = None
    if (replaces and audit_reserve
            and additive.stopping is not None
            and chosen.stopping is not None
            and additive.stopping.chosen >= 1
            and chosen.stopping.chosen >= 1):
        # One fold: the paired fold z above is inert, so the aggregate
        # margin alone would decide the adoption -- the bare-margin
        # route the sibling steps guard against. Read the per-row
        # differences on the same held-out rows instead, each side at
        # its own selected step, and require the bar the fold route
        # would have used; the report then records the per-row z in
        # place of the fold one. Families whose loss couples rows have
        # no per-row losses and keep the margin rule.
        row_z, _row_gain, _row_se, audit_fits = _held_out_pair_z(
            data, response, additive.formula, additive.stopping.chosen,
            largest, nu, chosen.formula, chosen.stopping.chosen, largest,
            nu, family, folds, env=env)
        spent += audit_fits
        if row_z is not None:
            audited_z = row_z
            z = row_z
            replaces = bool(row_z >= _SHAPE_SE)
    report.update(chosen_relative_gain=gain, chosen_z=z)
    if not replaces:
        if audited_z is not None:
            return stand(f"{labels[chosen.name]} clears the margin on the "
                         f"single holdout but its per-row paired z "
                         f"({audited_z:.2f}) is under the {_SHAPE_SE:.1f} bar "
                         f"the fold route would have used")
        return stand(f"{labels[chosen.name]} does not beat the additive pair by 2% of "
                     f"the held-out risk or two paired fold standard errors")

    adopted_formula = chosen.formula
    row = probe_mstop(data, response, adopted_formula, family, folds=folds,
                      largest=largest, nu=nu, env=env, fold_scheme=scheme)
    if row is None:
        return stand(f"the sweep for {labels[chosen.name]} failed")
    spent += row.fits
    report.update(adopted=True, formula=adopted_formula, mstop=row.chosen)
    return adopted_formula, row, spent


def _held_out_loss(full, reduced, floor=0.0):
    """Relative rise of the minimum cross-validated risk from the full to the
    reduced specification (negative when the reduced one does better); None
    when either curve is missing."""
    if full is None or reduced is None or not full.risk or not reduced.risk:
        return None
    best_full = float(np.nanmin(np.asarray(full.risk, dtype=float)))
    best_reduced = float(np.nanmin(np.asarray(reduced.risk, dtype=float)))
    # Measured from the saturated risk (`_risk_floor`), so a negative
    # log-likelihood risk reads on the same scale as squared error.
    best_full, best_reduced = best_full - floor, best_reduced - floor
    if not (np.isfinite(best_full) and np.isfinite(best_reduced)) or best_full <= 0:
        return None
    return best_reduced / best_full - 1.0

#: Depth of the residual tree term. On the training rows of seven
#: development data sets, depth 5 gained most of what depths 3 to 7 did, and
#: depth 3 needed up to 40,000 boosting steps.
_RESIDUAL_TREE_DEPTH = 5

#: The step length the residual tree candidate is fitted with.
_RESIDUAL_TREE_NU = 0.1


def _paired_held_out_gain(base_folds, tree_folds):
    """`(held_out_gain, held_out_se, z, reason)` for a residual tree.

    The held-out incremental value of the tree: the fold mean of (incumbent
    risk - tree risk) on the same folds, in the risk units of the score the
    folds hold, with its standard error (`paired_mean_se` in the Rust core)
    and their ratio ``z``. Positive means the tree lowers the held-out risk.
    The mean route and the distributional route both record these numbers.
    ``reason`` is None when the standard error is defined, else a sentence
    saying why it is not; ``z`` is then 0.
    """
    from mboostrs._rust import paired_mean_se

    base_folds = np.asarray(base_folds, float).ravel()
    tree_folds = np.asarray(tree_folds, float).ravel()
    if (base_folds.size == 0 or base_folds.size != tree_folds.size
            or not (np.isfinite(base_folds).all() and np.isfinite(tree_folds).all())):
        # A failed fold has no score to compare; the tree is not adopted on it.
        return (float("nan"), float("nan"), 0.0,
                "a fold's held-out score is missing, so the folds cannot be paired")
    held_out_gain, held_out_se = paired_mean_se(base_folds, tree_folds)
    if base_folds.size < 2:
        return (float(held_out_gain), float("nan"), 0.0,
                "one fold: a standard error of the fold differences needs at "
                "least two folds")
    z = float(held_out_gain / held_out_se) if held_out_se > 0 else 0.0
    return float(held_out_gain), float(held_out_se), z, None


def _residual_tree_step(data, response, formula, family, modelled, evidence, folds,
                        scheme, chosen_mstop, main_fit, nu, env, budget, spent,
                        risk_floor, mstop):
    """Add ``btree(<numeric columns>, maxdepth=5)`` when held-out risk says so.

    The incumbent and the candidate are each judged at their own best step
    count; the tree needs more steps than the additive terms, so the
    candidate's sweep may extend its grid. Returns the (possibly unchanged)
    formula, step count, full-data fit and fits spent. On a single supplied
    holdout the paired fold route cannot fire (one fold), so adoption
    additionally requires the per-row paired z of the held-out losses to
    clear the same 2.0 bar, when the family's loss admits per-row losses."""
    from mboostrs._autospec import (
        _held_out_pair_z,
        _path_held_out_evidence,
        _path_held_out_risk,
        _relative_gain,
        _single_fold_design,
    )

    # Factor columns enter as level codes, which the tree splits into two
    # sets of levels.
    columns = [c.name for c in modelled if c.kind in ("numeric", "boolean", "categorical")]
    n_folds = int(np.asarray(folds).shape[1])
    # A single supplied holdout also spends two fold fits on the per-row
    # audit of the adoption (below); reserve them up front.
    audit_reserve = 2 if _single_fold_design(folds, scheme, family) else 0
    if len(columns) < 2 or budget - spent < 4 * n_folds + audit_reserve:
        return formula, chosen_mstop, main_fit, spent
    term = (f"btree({', '.join(quote_name(_c) for _c in columns)}, maxdepth={_RESIDUAL_TREE_DEPTH}, "
            f"selection=sse)")
    candidate = formula + " + " + term
    # A deep tree term needs a small step: at 0.3 it overfits within a few
    # steps and was declined where at 0.1 it lowered held-out risk
    # (grid_stability R^2 0.904 against 0.937, white wine 0.449 against 0.478
    # on development training rows). The candidate is fitted at `tree_nu` and
    # compared with the incumbent at the incumbent's own step length, each at
    # its best step count.
    tree_nu = min(nu, _RESIDUAL_TREE_NU)
    # With coordinates present, judge the adoption on spatial-cell grouped
    # folds (#266): random row folds let the tree validate on rows
    # neighbouring its training rows, so the gain does not predict transfer
    # to unseen regions. `evidence["residual_tree"]["validation"]` records
    # which scheme decided.
    spatial, n_cells = _spatial_cell_scheme(data, modelled, folds, scheme)
    judge_folds = np.asarray(spatial) if spatial is not None else folds
    judge_scheme = spatial if spatial is not None else scheme
    shared_label = scheme.scheme if scheme is not None else "row"
    validation = (f"spatial-cell grouped folds ({n_cells} cells)"
                  if spatial is not None else f"shared {shared_label} folds")
    row = probe_mstop(data, response, candidate, family, folds=judge_folds,
                      largest=max(mstop, 200, chosen_mstop), nu=tree_nu, env=env,
                      extend=max(0, min(_MAX_EXTENSIONS,
                                        (budget - spent - 3 * n_folds) // n_folds)),
                      fold_scheme=judge_scheme)
    if row is None:
        evidence["residual_tree"] = {"term": term, "adopted": False,
                                     "reason": "the candidate's sweep failed",
                                     "n_candidates": 1, "validation": validation}
        return formula, chosen_mstop, main_fit, spent
    spent += row.fits
    largest_base = min(2 * chosen_mstop, _MSTOP_EXTENSION_CAP)
    single = _single_fold_design(judge_folds, judge_scheme, family)
    base_step = 0
    if single:
        base_evidence, base_fold_tuple = _path_held_out_evidence(
            data, response, formula, family, judge_folds, largest_base, nu,
            env=env, fold_scheme=judge_scheme)
        if base_evidence is None:
            base_risk, base_folds = float("nan"), ()
            spent += int(np.asarray(judge_folds).shape[1])
        else:
            base_risk = base_evidence.risk[base_evidence.chosen]
            base_step = base_evidence.chosen
            base_folds = base_fold_tuple
            spent += base_evidence.fits
    else:
        base_risk, fits, base_folds = _path_held_out_risk(
            data, response, formula, family, judge_folds, largest_base, nu, env=env,
            fold_scheme=judge_scheme)
        spent += fits
    tree_risk, fits, tree_folds = _path_held_out_risk(
        data, response, candidate, family, judge_folds, len(row.risk) - 1, tree_nu, env=env,
        fold_scheme=judge_scheme)
    spent += fits
    base_folds = np.asarray(base_folds, float).ravel()
    tree_folds = np.asarray(tree_folds, float).ravel()
    gain = _relative_gain(base_risk, tree_risk, risk_floor)
    held_out_gain, held_out_se, z, held_out_reason = _paired_held_out_gain(
        base_folds, tree_folds)
    row_z = None
    if (single and np.isfinite(gain) and gain >= _SHAPE_MARGIN_PAIR
            and base_step >= 1 and row.chosen >= 1):
        # One fold: the paired fold route above is inert (its standard
        # error is undefined on a single fold risk), so the margin alone
        # would decide. Read the per-row differences on the same
        # held-out rows instead, and require the bar the fold route
        # would have used; the evidence then reports the per-row gain,
        # its standard error and its z in place of the fold ones.
        row_z, row_gain, row_se, audit_fits = _held_out_pair_z(
            data, response, formula, base_step, largest_base, nu,
            candidate, row.chosen, len(row.risk) - 1, tree_nu,
            family, judge_folds, env=env)
        spent += audit_fits
        if row_z is not None:
            held_out_gain, held_out_se, z = row_gain, row_se, row_z
            held_out_reason = None
    if row_z is not None:
        adopted = bool(np.isfinite(gain) and gain >= _SHAPE_MARGIN_PAIR
                       and z >= 2.0)
    else:
        adopted = bool(np.isfinite(gain) and (gain >= _SHAPE_MARGIN_PAIR
                                              or (base_folds.size > 1 and held_out_gain > 0
                                                  and z >= 2.0)))
    evidence["residual_tree"] = {"term": term, "adopted": adopted,
                                 "relative_gain": round(float(gain), 6),
                                 "held_out_gain": round(float(held_out_gain), 6),
                                 "held_out_se": round(float(held_out_se), 6),
                                 "z": round(z, 2), "folds": int(base_folds.size),
                                 "n_candidates": 1, "mstop": row.chosen,
                                 "validation": validation}
    if held_out_reason is not None:
        evidence["residual_tree"]["held_out_reason"] = held_out_reason
    if not adopted:
        return formula, chosen_mstop, main_fit, spent
    evidence["adopted_tree"] = term
    evidence["mstop"] = row
    evidence["residual_tree"]["nu"] = tree_nu
    # How much of the adopted prediction the tree carries, on training rows,
    # at the step count the tree was judged at.
    evidence["residual_tree"]["share"] = _tree_contribution_share(
        row._fit[:row.chosen])
    return candidate, row.chosen, row._fit, spent


#: A simple rival replaces the searched specification when its held-out risk
#: is lower by more than this many standard errors of the paired fold
#: difference.
_WHOLE_MODEL_SE = 1.0

#: Why a distributional recommendation carries no whole-model comparison.
_WHOLE_MODEL_LSS_REASON = (
    "the simple rivals are mean models scored by the mean family's held-out "
    "risk, and a distributional model is scored by its held-out log score per "
    "row, so the two are not on one scale; the comparison of the "
    "distributional model with the finished mean model is "
    "evidence['finished_routes']")

_SMOOTH_LEARNER = re.compile(r"^(bbs|btree|bns)\((.*)\)$", re.DOTALL)


def _first_argument(arguments):
    """The text before the first top-level comma of a learner's arguments."""
    depth = 0
    for index, char in enumerate(arguments):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return arguments[:index].strip()
    return arguments.strip()


def _linear_rival(formula):
    """The formula with each one-column smooth term written as ``bols``.

    ``bbs(x, df=4)``, ``btree(x)`` and ``bns(x)`` become ``bols(x)``; a
    transformed input keeps its transform (``bbs(log(x), df=4)`` becomes
    ``bols(log(x))``). Every other term is kept as written: ``bols`` and
    ``brandom`` terms, the caller's declared terms (``bmono``, ``bspatial``,
    ``bmrf``, ``buser``) and terms over two or more columns. A term that
    becomes a duplicate (the two parts of a Multinomial decomposition) is
    written once.
    """
    from mboostrs.formula import term_variables

    lhs, _, rhs = partition_tilde(formula)
    out = []
    for term in _split_plus(rhs):
        term = term.strip()
        match = _SMOOTH_LEARNER.match(term)
        if match and len(term_variables(term)) == 1:
            term = f"bols({_first_argument(match.group(2))})"
        if term not in out:
            out.append(term)
    return f"{lhs.strip()} ~ " + " + ".join(out)


def _whole_model_step(data, response, formula, family, evidence, folds, scheme,
                      chosen_mstop, nu, start_formula, rival_nu, env, budget,
                      spent, mstop):
    """Compare the searched specification with two prespecified simple models.

    The rivals are fixed: ``"linear"``, the starting specification with every
    one-column smooth written as ``bols`` (`_linear_rival`), and
    ``"additive"``, the starting specification itself (the dtype-driven terms
    with the caller's declarations, or the `prior`). Each is fitted at the
    search's step length `rival_nu` and judged at its own best step on the
    same folds as the searched specification. The paired fold differences
    and their standard error come from `_paired_held_out_gain`.

    A rival replaces the searched specification when its held-out risk is
    lower by more than one standard error of the paired difference. On one
    fold the standard error is undefined and nothing is replaced. Returns
    ``(formula, mstop, nu, replaced_by, spent)``; ``replaced_by`` is the
    rival's name or None. The record is ``evidence["whole_model"]``.
    """
    n_folds = int(np.asarray(folds).shape[1])
    rivals = {"linear": _linear_rival(start_formula), "additive": start_formula}
    to_fit = {name: rival for name, rival in rivals.items() if rival != formula}
    distinct = len(set(to_fit.values()))
    need = n_folds * (1 + distinct) if distinct else 0
    record = {"compared": False, "chosen_formula": formula,
              "rivals": dict(rivals), "rule": (
                  f"a rival replaces the searched specification when its "
                  f"held-out risk is lower by more than {_WHOLE_MODEL_SE:g} "
                  f"standard error of the paired fold difference; never on a "
                  f"single holdout")}
    evidence["whole_model"] = record

    def stand(reason):
        record["reason"] = reason
        record["sentence"] = reason
        return formula, chosen_mstop, nu, None, spent

    if not to_fit:
        return stand("the searched specification is the linear and the additive "
                     "rival, so there is nothing to compare")
    if budget - spent < need:
        return stand(f"{budget - spent} fits remained after the search, fewer than "
                     f"the {need} the comparison needs ({n_folds} per fold path: "
                     f"the searched specification and {distinct} rival(s)); "
                     f"raise budget= to run it")
    row = evidence.get("mstop")
    chosen_largest = min(_MSTOP_EXTENSION_CAP, max(
        int(chosen_mstop), 1, len(row.risk) - 1 if row is not None else 0))
    rival_largest = min(_MSTOP_EXTENSION_CAP,
                        max(int(mstop), 200, 2 * int(chosen_mstop), chosen_largest))

    def score(rival_formula, largest, step_nu):
        from mboostrs._autospec import _path_held_out_evidence

        try:
            stopping, per_fold = _path_held_out_evidence(
                data, response, rival_formula, family, folds, largest, step_nu,
                env=env, fold_scheme=scheme)
        except (MboostrsRuntimeError, ValueError):
            stopping, per_fold = None, ()
        return stopping, np.asarray(per_fold, float)

    chosen, chosen_folds = score(formula, chosen_largest, nu)
    spent += n_folds
    if chosen is None:
        return stand("the searched specification's fold paths failed, so it "
                     "could not be scored")
    record.update(compared=True, folds=n_folds,
                  chosen_risk=float(chosen.risk[chosen.chosen]),
                  chosen_mstop=int(chosen.chosen))
    scored = {}
    results = {}
    for name, rival_formula in rivals.items():
        entry = {"formula": rival_formula}
        if rival_formula == formula:
            entry.update(same_as_chosen=True, risk=record["chosen_risk"],
                         mstop=record["chosen_mstop"], difference=0.0,
                         difference_se=0.0, z=0.0, beats_chosen=False)
            results[name] = entry
            continue
        if rival_formula not in scored:
            scored[rival_formula] = score(rival_formula, rival_largest, rival_nu)
            spent += n_folds
        stopping, rival_folds = scored[rival_formula]
        if stopping is None:
            entry.update(risk=float("nan"), beats_chosen=False,
                         reason="the rival's fold paths failed")
            results[name] = entry
            continue
        # `_paired_held_out_gain` is the fold mean of (chosen - rival):
        # positive when the rival has less held-out risk.
        gain, se, z, reason = _paired_held_out_gain(chosen_folds, rival_folds)
        beats = bool(reason is None and np.isfinite(se) and se > 0
                     and gain > _WHOLE_MODEL_SE * se)
        entry.update(risk=float(stopping.risk[stopping.chosen]),
                     mstop=int(stopping.chosen),
                     at_boundary=bool(stopping.at_boundary),
                     difference=-float(gain), difference_se=float(se),
                     z=round(float(z), 2), beats_chosen=beats,
                     _stopping=stopping)
        if reason is not None:
            entry["reason"] = reason
        results[name] = entry
    record["rivals"] = results
    record["fits"] = n_folds * (1 + len(scored))
    winners = [name for name, entry in results.items() if entry["beats_chosen"]]

    def describe(name):
        entry = results[name]
        if entry.get("same_as_chosen"):
            return f"{name} rival is the searched specification"
        if not np.isfinite(entry["risk"]):
            return f"{name} rival failed"
        return (f"{name} rival risk {entry['risk']:.4g} (difference "
                f"{entry['difference']:+.3g}, standard error "
                f"{entry['difference_se']:.2g})")

    summary = (f"searched specification risk {record['chosen_risk']:.4g} on "
               f"{n_folds} folds; " + "; ".join(describe(name) for name in results))
    if not winners:
        single = n_folds < 2
        record["replaced"] = False
        record["winner"] = "searched"
        record["sentence"] = summary + (
            "; one fold gives no standard error, so the searched specification "
            "stands" if single else
            f"; no rival is lower by more than {_WHOLE_MODEL_SE:g} standard "
            f"error of the paired difference, so the searched specification "
            f"stands")
        for entry in results.values():
            entry.pop("_stopping", None)
        return formula, chosen_mstop, nu, None, spent
    best = min(winners, key=lambda name: results[name]["risk"])
    stopping = results[best]["_stopping"]
    for entry in results.values():
        entry.pop("_stopping", None)
    record["replaced"] = True
    record["winner"] = best
    record["sentence"] = summary + (
        f"; the {best} rival is lower by more than {_WHOLE_MODEL_SE:g} "
        f"standard error of the paired difference and replaces the searched "
        f"specification {formula}")
    evidence["mstop"] = replace(
        stopping, sentence=(f"mstop={stopping.chosen} — the {best} rival's "
                            f"shared-fold path (cap {stopping.grid[-1]}), adopted "
                            f"by the whole-model comparison"))
    if rival_nu != nu or "nu" in evidence:
        evidence["nu"] = (f"nu={rival_nu:g} — the step length of the {best} "
                          f"rival the whole-model comparison adopted")
    return results[best]["formula"], int(stopping.chosen), rival_nu, best, spent


#: Families with a log link, whose fixed-step fits of a large response can
#: stop early; `autospec` also tries the line search for them.
_LOG_LINK_NAMES = ("GammaReg", "Poisson", "NBinomial")

#: Families whose fixed-step update grows with the fitted mean, so that a
#: large count response overflows the inverse link at the autospec step length.
#: Poisson's negative gradient is ``y - exp(f)``, which is on the count scale;
#: NBinomial's is bounded by its dispersion and does not overflow.
_COUNT_SCALE_NAMES = ("Poisson",)


def _cv_minimum(rec):
    """The lowest cross-validated risk of the recommendation's final sweep."""
    row = rec.evidence.get("mstop")
    if row is None or not len(row.risk):
        return float("inf")
    risk = np.asarray(row.risk, dtype=float)
    return float(np.nanmin(risk)) if np.isfinite(risk).any() else float("inf")


#: Interaction leads `interactions="adopt"` may add; the report lists three.
_ADOPT_LEADS = 6

#: The step length `autospec` uses when the caller gives none.
_AUTOSPEC_NU = 0.3

#: The step length `autospec` switches to when the default one leaves the
#: cross-validated risk still falling at the step cap.
_FAST_NU = 0.5

#: Deselection reads shares off a tuned path; below this many steps it is
#: skipped.
_DESELECT_MIN_MSTOP = 10

#: A column fitted as groups sets the fold design (whole groups held out)
#: only with at least this many levels: a clustering unit such as a clinic or
#: a child. A factor with a few levels, a product type or a region, is a
#: category every training fold sees, and its rows are split at random.
_MIN_UNIT_LEVELS = 10


#: Column-name pairs that read as coordinates. When the predictors include
#: one, the default smooth-pair machinery is the wrong tool for the
#: geography: California house prices change sharply at city boundaries,
#: which a latitude-longitude tensor smooths over. The residual-tree offer
#: rule of #244 keys on these.
_COORDINATE_PAIRS = frozenset([
    frozenset(("latitude", "longitude")),
    frozenset(("lat", "lon")),
    frozenset(("lat", "lng")),
    frozenset(("lat", "long")),
    frozenset(("x", "y")),
    frozenset(("easting", "northing")),
])


def _has_coordinate_pair(modelled):
    """Whether two modelled columns read as a coordinates pair."""
    names = {c.name.lower() for c in modelled}
    return any(pair <= names for pair in _COORDINATE_PAIRS)


def _coordinate_pair_names(modelled):
    """The matched coordinate column names in modelled case, or None.

    The names come back in the order the columns appear in ``modelled``, so
    the term is written ``bspatial(x, y)`` rather than ``bspatial(y, x)``
    whatever order the set of known pairs happens to iterate in.
    """
    lowered = {c.name.lower() for c in modelled}
    for pair in _COORDINATE_PAIRS:
        if pair <= lowered:
            return tuple(c.name for c in modelled if c.name.lower() in pair)
    return None


def _spatial_cell_codes(data, modelled):
    """Integer grid-cell codes over the coordinate pair, or None.

    Discretizes the coordinates-like pair into quantile bins (~10 rows per
    cell). Returns None when no pair is present or a coordinate is
    degenerate (missing values or fewer than 3 distinct values).
    """
    pair = _coordinate_pair_names(modelled)
    if pair is None:
        return None
    vals = []
    for name in pair:
        col = read_column(data, name)
        v = np.asarray(col.values, dtype=float)
        if col.n_missing or np.unique(v).size < 3:
            return None
        vals.append(v)
    n = len(vals[0])
    per = max(2, min(50, int(round((n / 10) ** 0.5))))
    codes = np.zeros(n, dtype=np.int64)
    stride = 1
    for v in vals:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, per + 1)))
        if edges.size < 3:
            return None
        b = np.clip(np.digitize(v, edges[1:-1]), 0, edges.size - 2)
        codes += b * stride
        stride *= edges.size - 1
    return codes


def _spatial_cell_scheme(data, modelled, folds, scheme):
    """A grouped-fold scheme over spatial grid cells, or None (#266).

    The residual-tree adoption decision compares held-out risk of the
    incumbent against the incumbent plus a joint tree. With random row folds
    and coordinates present, the tree validates on rows neighbouring its
    training rows, so a gain measured in-sample need not transfer to unseen
    regions (California housing: adopted at relative gain 0.21, z=7.5,
    then scored worse than the smooth baseline on the spatial holdout).
    Grouped folds over grid cells make the adoption statistic measure
    transfer to unseen regions instead. Returns ``(scheme, n_cells)``; the
    scheme is None when there are no coordinates, the shared scheme is
    already grouped, the caller supplied one holdout, or too few cells form
    -- the caller then keeps the shared scheme. A single holdout cannot be
    replaced by grouped k-fold validation, which requires at least two folds.
    """
    if np.asarray(folds).shape[1] < 2 or getattr(scheme, "groups", None) is not None:
        return None, 0
    codes = _spatial_cell_codes(data, modelled)
    if codes is None:
        return None, 0
    n_folds = int(np.asarray(folds).shape[1])
    n_cells = len(np.unique(codes))
    if n_cells < 2 * n_folds:
        return None, 0
    seed = scheme.seed if getattr(scheme, "seed", None) is not None else 0
    return grouped_folds(codes, B=n_folds, seed=seed,
                         weights=getattr(scheme, "base_weights", None)), n_cells


def _residual_tree_detection(data, response, modelled, main_fit):
    """The screen-stage lead for a residual tree, or None when it cannot be measured.

    Detection is what the additive fit leaves behind, measured on training
    rows: ``leftover_share`` is the share of the response variance the fit
    does not explain, and ``tree_r2`` is the in-sample R^2 of a depth-3 tree
    on that leftover over the numeric columns. ``product`` is their product,
    the share of the total variance a shallow tree reads from the leftover.
    The numbers are a lead, not a held-out measurement: adoption is decided
    by the held-out comparison in ``_residual_tree_step`` (#124).
    """
    from sklearn.tree import DecisionTreeRegressor

    from mboostrs._rust import leftover_share

    if main_fit is None:
        return None
    y = np.asarray(read_column(data, response).values, dtype=float)
    # The response scale: a link-scale predictor is not comparable with y for
    # a Binomial or Poisson fit.
    fitted = np.asarray(main_fit.predict(type="response"), dtype=float).ravel()
    if fitted.shape[0] != y.shape[0]:
        return None
    numeric = [c for c in modelled if c.kind == "numeric"]
    if len(numeric) < 2:
        return None
    X = np.column_stack([np.asarray(read_column(data, c.name).values, dtype=float)
                         for c in numeric])
    # Rows with a missing value in the response, the fit or a predictor carry
    # no leftover to read.
    usable = np.isfinite(y) & np.isfinite(fitted) & np.isfinite(X).all(axis=1)
    if usable.sum() < 10:
        return None
    y, fitted, X = y[usable], fitted[usable], X[usable]
    share = float(leftover_share(y, fitted))
    residual = y - fitted
    tree_r2 = 0.0
    if np.isfinite(share) and share > 0:
        tree = DecisionTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X, residual)
        tree_r2 = float(tree.score(X, residual))
    return {"leftover_share": share, "tree_r2": tree_r2,
            "product": share * tree_r2, "candidates": 1,
            "basis": "training rows, depth-3 tree on the additive leftover"}


#: The screen-stage tree R^2 that offers a residual tree after pairs were
#: adopted: a shallow tree taking a tenth of the leftover on training rows is
#: the signature of structure the smooth pairs left behind.
_RESIDUAL_TREE_OFFER_R2 = 0.10


def _should_offer_residual_tree(modelled, evidence, main_fit, data, response,
                                detection=None):
    """The training-only offer rule for a residual tree (#244).

    Offers when (b) the predictors include a coordinates-like pair, or (a)
    adopted pairs left a large residual-scale reduction: the pairs were
    adopted, so interaction structure is present, and a shallow regression
    tree fitted on the training residual still explains a substantial share
    of it (``detection["tree_r2"]``), which is what sharp-boundary structure
    looks like to a smooth model. The offer is training-only; whether the
    tree is ADOPTED is decided by the held-out incremental-value statistic
    in ``_residual_tree_step``.
    """
    if _has_coordinate_pair(modelled):
        return True, "predictors include a coordinates-like pair"
    adopted = evidence.get("adopted_interactions", ())
    if adopted and detection is not None:
        r2 = detection["tree_r2"]
        if r2 >= _RESIDUAL_TREE_OFFER_R2:
            return True, (f"adopted pairs leave residual structure: a "
                          f"depth-3 tree explains {r2:.0%} of the "
                          f"training residual variance")
    return False, ""


def _tree_learner_indices(fit):
    """The positions of the tree base-learners in a fitted model."""
    from mboostrs.learners import Btree

    specs = getattr(fit, "_learners_spec", None) or ()
    return [i for i, spec in enumerate(specs) if isinstance(spec, Btree)]


def _tree_contribution_share(fit):
    """How much of a fitted model's linear predictor its tree learners carry.

    Returns ``{"covariance_share", "absolute_share", "rows"}`` on training
    rows, or None when the model has no tree learner. The covariance share
    is ``cov(tree, eta) / var(eta)`` for the link-scale predictor ``eta``; the
    shares of all learners sum to one, so it is the tree's exact part of the
    prediction's variance. The absolute share is the tree's part of the summed
    absolute deviations of the tree and the rest of the prediction (see
    ``mboostrs._rust.contribution_shares``).
    """
    from mboostrs._rust import contribution_shares

    indices = _tree_learner_indices(fit)
    if not indices:
        return None
    eta = np.asarray(fit.predict(type="link"), dtype=float).ravel()
    tree = np.zeros(eta.shape[0])
    for index in indices:
        for effect in fit.partial_effects(which=index).values():
            values = effect[1] if isinstance(effect, tuple) else effect
            tree += np.asarray(values, dtype=float).ravel()
    usable = np.isfinite(eta) & np.isfinite(tree)
    if usable.sum() < 2:
        return None
    covariance_share, absolute_share = contribution_shares(
        tree[usable], eta[usable] - tree[usable])
    return {"covariance_share": float(covariance_share),
            "absolute_share": float(absolute_share), "rows": int(usable.sum())}


def _search_lss_mu(data, response, *, lss_family, family_evidence, formula, modelled,
                   terms, folds, scheme, nu, env, df, declared, interactions,
                   heredity, forbidden, residual_tree, budget, spent, mean_mstop,
                   seed, n_rows, defaulted, measured, evidence, judge=None):
    """Run the shape, pair and residual-tree probes on a distributional
    family's first parameter (`mu`) and return what they adopted.

    Each probe varies the `mu` formula and scores the variant by the family's
    own held-out loss (`_LSSMu`), on the same folds and with the rules the
    mean-model probes use: a shape or pair or tree term is adopted when it
    lowers the held-out loss by 2% or by two paired fold standard errors.
    The other parameter's formula stays as the family probe chose it. The
    pair screen reads the residual of a Gaussian fit of the `mu` formula
    (the screen only proposes leads; the held-out loss of the distributional
    model decides). Deselection is not run.

    A change that passes the held-out loss must also keep the mean: the
    log-link density weights a miss by 1 / mu^2, so a change can lower the
    family's loss while the squared error of mu rises (a large mu on a
    high response costs little). Each adopted change is checked with the rule
    the family choice uses for its mean check (`_mse_worse`: worse by 2% and
    two paired standard errors, each model at its own step count) and is
    refused when the mean is worse.

    Returns `(family_evidence, fit_formula, nu, spent)`; `fit_formula` is a
    mapping from parameter name to formula when `mu` changed, else the
    formula the caller passed. Probe results are written to `evidence`.
    """
    from mboostrs._autospec import (
        _SHAPE_SE,
        _lss_fold_mean_mse,
        _LSSMu,
        _mse_worse,
        coordinate_beats,
        score_coordinate_candidates,
    )
    from mboostrs.explore import _paired_se, _partitions

    names = tuple(lss_family.param_names)
    given = family_evidence.lss_formula
    start_mu = given[names[0]] if isinstance(given, dict) else formula
    mu_formula = start_mu
    other_formula = given[names[1]] if isinstance(given, dict) else formula
    view = _LSSMu(lss_family, other_formula)
    folds = np.asarray(folds, dtype=float)
    n_folds = int(folds.shape[1])
    steps0 = (max(family_evidence.lss_mstop) if family_evidence.lss_mstop
              else int(family_evidence.lss_largest or 50))
    # Candidates are cross-validated along a path four times the family's own
    # optimum (never longer than the path the family check needed); a
    # candidate whose optimum is at the end of that path doubles it.
    largest = int(min(max(4 * steps0, 50),
                      max(int(family_evidence.lss_largest or 0), 50)))
    base_risk = float(family_evidence.lss_score)
    # A relative gain presumes a risk bounded below by zero; a density can
    # have a negative negative log-likelihood, so the floor puts the scale at
    # the risk's magnitude.
    floor = base_risk - abs(base_risk) if np.isfinite(base_risk) else 0.0
    changed = {}
    refused = []
    pair_reserve = 4 * n_folds
    mean_cache, mean_fits = {}, [0]

    def mean_errors(mu_f, nu_f):
        key = (mu_f, float(nu_f))
        if key not in mean_cache:
            steps = view.optimal.get(key, (None,))[0]
            result = _lss_fold_mean_mse(
                data, response, view.formula(mu_f), lss_family, folds,
                max(steps) if steps else steps0, nu_f, env, judge=judge)
            if result is not None:
                mean_fits[0] += result[1]
            mean_cache[key] = None if result is None else result[0]
        return mean_cache[key]

    def mean_holds(candidate_f, candidate_nu, current_f, current_nu):
        """`(holds, (candidate mse, current mse, se))`."""
        new, cur = mean_errors(candidate_f, candidate_nu), mean_errors(current_f, current_nu)
        if new is None or cur is None or new.size != cur.size or new.size == 0:
            return False, (float("nan"), float("nan"), float("nan"))
        se = _paired_se(new, cur, _partitions(folds))
        numbers = (float(new.mean()), float(cur.mean()), float(se))
        return (not _mse_worse(numbers[0], numbers[1], se)), numbers

    if budget > spent and "shapes" in defaulted and modelled:
        shape_evidence, shape_fits = probe_shapes(
            data, response, modelled, terms, view, folds=folds, mstop=largest,
            nu=nu, df=df, max_fits=max(n_folds, budget - spent - pair_reserve),
            fixed_names=declared, env=env, fold_scheme=scheme, depth=largest,
            risk_floor=floor)
        spent += shape_fits
        if shape_evidence:
            chosen = {row.column: row for row in shape_evidence}
            rebuilt = []
            for column, term in zip(modelled, terms, strict=True):
                row = chosen.get(column.name)
                if row is None or row.chosen == row.incumbent:
                    rebuilt.append(term)
                else:
                    rebuilt.append(next(s.term for s in row.scores
                                        if s.shape == row.chosen))
            current_terms, by_column = list(terms), {row.column: row for row in shape_evidence}
            for index, (column, old) in enumerate(zip(modelled, terms, strict=True)):
                if rebuilt[index] == old:
                    continue
                trial = list(current_terms)
                trial[index] = rebuilt[index]
                ok, (new_mse, cur_mse, se) = mean_holds(
                    f"{response} ~ " + " + ".join(trial), nu,
                    f"{response} ~ " + " + ".join(current_terms), nu)
                if ok:
                    current_terms = trial
                    continue
                row = by_column[column.name]
                by_column[column.name] = replace(
                    row, chosen=row.incumbent, sentence=(
                        f"{row.sentence}; but the squared error of mu is "
                        f"{new_mse:.3g} against {cur_mse:.3g} (standard error of "
                        f"the difference {se:.2g}), so {row.incumbent} stands"))
                refused.append(rebuilt[index])
                rebuilt[index] = old
            shape_evidence = [by_column[row.column] for row in shape_evidence]
            changed["shapes"] = [t for t, old in zip(current_terms, terms, strict=True)
                                 if t != old]
            terms = current_terms
            mu_formula = f"{response} ~ " + " + ".join(terms)
            evidence["shapes"] = tuple(shape_evidence)
            measured.append("shapes")
            defaulted.remove("shapes")

    if budget > spent and "interactions" in defaulted and len(modelled) > 1:
        counts = {}
        try:
            leads, screen_fits, _ = probe_interactions(
                data, response, mu_formula, Gaussian(), modelled, terms,
                mstop=int(max(mean_mstop, 50)), nu=nu, env=env, main_fit=None,
                top=_ADOPT_LEADS if interactions == "adopt" else 3,
                surrogate=False, seed=seed, counts=counts)
        except (MboostrsRuntimeError, ValueError) as exc:
            leads, screen_fits = (), 0
            evidence["interactions_skipped"] = f"the pair screen failed: {exc}"
        if forbidden:
            leads = tuple(lead for lead in leads
                          if frozenset((lead.left, lead.right)) not in forbidden)
        evidence["interaction_candidates"] = counts
        evidence["interactions"] = leads
        measured.append("interactions")
        defaulted.remove("interactions")
        spent += screen_fits
        in_model = {column.name for column in modelled}

        def heredity_allows(lead):
            present = (lead.left in in_model) + (lead.right in in_model)
            return {"none": True, "weak": present >= 1, "strong": present == 2}[heredity]

        groups = {}
        if interactions == "adopt":
            for lead in leads:
                if lead.kind != "duplicate-axis" and lead.term and heredity_allows(lead):
                    groups.setdefault(frozenset((lead.left, lead.right)),
                                      {})[lead.kind] = lead.term
        pair_cap = max(1, n_rows // _PAIRS_PER_ROW)
        kept, rejected, adoption, unvalidated = [], [], {}, []
        current = None
        for kinds in groups.values():
            if len(kept) >= pair_cap or budget - spent < n_folds * (2 + len(kinds)):
                unvalidated.extend(kinds.values())
                continue
            candidates = {kind: f"{mu_formula} + {term}" for kind, term in kinds.items()}
            if current is None:
                base, fits = score_coordinate_candidates(
                    data, response, {"current": mu_formula}, view, folds, largest,
                    nu, env=env, fold_scheme=scheme)
                current, spent = base["current"], spent + fits
            scores, fits = score_coordinate_candidates(
                data, response, candidates, view, folds, largest, nu, env=env,
                fold_scheme=scheme)
            spent += fits
            winners = [(scores[kind].risk, kind) for kind in scores
                       if coordinate_beats(current, scores[kind], floor)[0]]
            kind = min(winners)[1] if winners else min(
                scores, key=lambda k: scores[k].risk if np.isfinite(scores[k].risk)
                else float("inf"))
            _, gain, z = coordinate_beats(current, scores[kind], floor)
            term = kinds[kind]
            adoption[term] = (len(scores[kind].fold_risk), round(gain, 6),
                              round(z, 2), _SHAPE_SE)
            holds = mean_holds(candidates[kind], nu, mu_formula, nu)[0] if winners else False
            if winners and holds:
                kept.append(term)
                mu_formula = candidates[kind]
                current = scores[kind]
            else:
                if winners:
                    refused.append(term)
                rejected.append((term, round(gain, 6), round(z, 2)))
        if interactions == "adopt":
            evidence["adopted_interactions"] = tuple(kept)
            if rejected:
                evidence["rejected_interactions"] = tuple(rejected)
            if unvalidated:
                evidence["unvalidated_interactions"] = tuple(unvalidated)
            evidence["adoption_evidence"] = adoption
            changed["pairs"] = list(kept)

    offer_tree = residual_tree is True or (
        residual_tree == "auto" and _has_coordinate_pair(modelled))
    columns = [c.name for c in modelled
               if c.kind in ("numeric", "boolean", "categorical")]
    if offer_tree and len(columns) >= 2 and budget - spent >= 4 * n_folds:
        term = (f"btree({', '.join(columns)}, maxdepth={_RESIDUAL_TREE_DEPTH}, "
                f"selection=sse)")
        tree_nu = min(nu, _RESIDUAL_TREE_NU)
        spatial, n_cells = _spatial_cell_scheme(data, modelled, folds, scheme)
        judge_folds = np.asarray(spatial) if spatial is not None else folds
        judge_scheme = spatial if spatial is not None else scheme
        validation = (f"spatial-cell grouped folds ({n_cells} cells)"
                      if spatial is not None else "shared folds")
        candidate = f"{mu_formula} + {term}"
        base, fits = score_coordinate_candidates(
            data, response, {"base": mu_formula}, view, judge_folds, largest, nu,
            env=env, fold_scheme=judge_scheme)
        spent += fits
        tree, fits = score_coordinate_candidates(
            data, response, {"tree": candidate}, view, judge_folds, largest,
            tree_nu, env=env, fold_scheme=judge_scheme)
        spent += fits
        adopted, gain, z = coordinate_beats(base["base"], tree["tree"], floor)
        if adopted and not mean_holds(candidate, tree_nu, mu_formula, nu)[0]:
            adopted = False
            refused.append(term)
        # The same held-out gain and standard error the mean route records,
        # here in the family's held-out loss (negative log-likelihood per
        # row). ``z`` stays the statistic the adoption rule read.
        held_out_gain, held_out_se, _, held_out_reason = _paired_held_out_gain(
            base["base"].fold_risk, tree["tree"].fold_risk)
        if held_out_reason is not None and np.isfinite(held_out_gain):
            # A finite gain without a standard error is the one-fold case.
            # There the mean route replaces the fold statistic by per-row paired
            # losses on a single holdout; those read the engine's per-row
            # risk, which is not the distributional held-out score, so the
            # distributional route has no per-row reading.
            held_out_reason += ("; the per-row audit the mean route uses on a "
                                "single holdout reads the engine's per-row risk, "
                                "which is not the distributional held-out score")
        evidence["residual_tree"] = {
            "term": term, "adopted": bool(adopted),
            "relative_gain": round(float(gain), 6),
            "held_out_gain": round(float(held_out_gain), 6),
            "held_out_se": round(float(held_out_se), 6),
            "z": round(float(z), 2),
            "folds": len(tree["tree"].fold_risk) or n_folds, "n_candidates": 1,
            "validation": validation}
        if held_out_reason is not None:
            evidence["residual_tree"]["held_out_reason"] = held_out_reason
        measured.append("residual_tree")
        if adopted:
            evidence["adopted_tree"] = term
            evidence["residual_tree"]["nu"] = tree_nu
            mu_formula, nu = candidate, tree_nu
            changed["tree"] = term

    spent += mean_fits[0]
    if refused:
        evidence["lss_mu_mean_refused"] = tuple(refused)
    if mu_formula == start_mu:
        return family_evidence, given if isinstance(given, dict) else None, nu, spent
    full = view.formula(mu_formula)
    optimal = view.optimal.get((mu_formula, float(nu)))
    if optimal is None:
        _, fits, _, _ = _lss_mu_scan(view, data, response, mu_formula, folds,
                                     largest, nu, env, scheme)
        spent += fits
        optimal = view.optimal.get((mu_formula, float(nu)))
    updated = replace(
        family_evidence, lss_formula=full,
        lss_mstop=optimal[0] if optimal else family_evidence.lss_mstop)
    evidence["lss_mu_search"] = {key: value for key, value in changed.items() if value}
    return updated, full, nu, spent


def _held_out_pairs(data, response, formula, adopted, family, folds, largest, nu, env,
                    scheme, spent, risk_floor=0.0, n_candidates=None):
    """`(kept, rejected, evidence, spent)`: the adopted pair terms whose
    removal raises the cross-validated risk by 2% or by the scaled paired
    fold-SE bar, the others with their measured cost, and per-term adoption
    evidence ``{term: (n_folds, rel, z, z_bar)}``.

    ``n_candidates`` is the number of pairs the screen offered: the SE bar
    grows as ``2 * sqrt(log(n_candidates))`` so a many-column screen does
    not pass chance pairs on a fixed 2-SE bar.
    """
    from mboostrs._autospec import _path_held_out_risk, _relative_gain

    with_risk, fits, with_folds = _path_held_out_risk(
        data, response, formula, family, folds, largest, nu, env=env, fold_scheme=scheme)
    spent += fits
    kept, rejected = [], []
    adoption_evidence = {}
    z_bar = _adoption_z_bar(len(adopted) if n_candidates is None else n_candidates)
    lhs, rhs = (lambda _p: (_p[0], _p[2]))(partition_tilde(formula))
    base_terms = _split_plus(rhs)
    n_folds = int(np.asarray(with_folds).size) if with_folds is not None else 0
    for term in adopted:
        without = f"{lhs.strip()} ~ " + " + ".join(t for t in base_terms if t != term)
        risk, fits, folds_without = _path_held_out_risk(
            data, response, without, family, folds, largest, nu, env=env, fold_scheme=scheme)
        spent += fits
        d = np.asarray(folds_without, float) - np.asarray(with_folds, float)
        rel = _relative_gain(risk, with_risk, risk_floor)
        z = (float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
             if d.size > 1 and d.std(ddof=1) > 0 else 0.0)
        adoption_evidence[term] = (n_folds, round(float(rel), 6), round(z, 2),
                                   round(z_bar, 2))
        if rel >= _SHAPE_MARGIN_PAIR or (d.size > 1 and d.mean() > 0 and z >= z_bar):
            kept.append(term)
        else:
            rejected.append((term, round(float(rel), 6), round(z, 2)))
    return kept, rejected, adoption_evidence, spent


#: An adopted pair whose removal costs this share of the held-out risk is
#: kept whatever the fold spread.
_SHAPE_MARGIN_PAIR = 0.02


def _adoption_z_bar(n_candidates):
    """The paired-SE bar a pair must clear, scaled by the candidate count.

    The 2-SE rule is a fixed bar: with many candidate pairs, one clears it
    by chance. Scaling the bar as ``2 * sqrt(log(n_candidates))`` is the
    multiple-comparison correction the issue asks for -- the bar grows
    slowly with the number of pairs the screen offered, and equals 2.0 when
    a single pair is on trial, so the G-35 rule is scaled, not replaced.
    """
    n_candidates = max(int(n_candidates), 1)
    return max(2.0, 2.0 * math.sqrt(math.log(n_candidates)))


#: The most pairs the adoption route offers, as a share of the sample size:
#: a pair term costs roughly this many effective rows, so offering more
#: pairs than n // _PAIRS_PER_ROW invites the tuned fit to spend them on
#: noise. O(n/constant), per the issue; the screen's top-6 cap still binds
#: first at ordinary sizes.
_PAIRS_PER_ROW = 50


_SMOOTH_TERM = re.compile(rf"^bbs\(((?:log|sqrt)\({N}\)|{N}), df=[\d.]+\)$")


def _decomposed(term):
    """A column's smooth as mboost's decomposition, ``bols(x) + bbs(x,
    center=TRUE, df=1)``: a line, and a smooth deviation from it with one
    degree of freedom. For a Multinomial response, where each term is fitted
    once per class, it halved the excess test log-loss of ``bbs(x, df=4)`` on
    three of four simulated three-class designs (linear, mixed, mixed with
    noise columns) and lowered it on the fourth (strongly nonlinear effects).
    Other terms are returned unchanged."""
    match = _SMOOTH_TERM.match(term)
    if not match:
        return term
    column = match.group(1)
    return f"bols({column}) + bbs({column}, center=TRUE, df=1)"


def _term_shares(fit, mstop, terms):
    """Each term's share of the in-bag risk reduction at `mstop`, or None when
    the fit's learners do not map onto the terms. A compound term (``bols(x) +
    bbs(x, center=TRUE, df=1)``, one column's decomposition) owns as many
    learners as it has parts and gets their summed share."""
    names = list(fit._learner_names)
    parts = [_split_plus(term) for term in terms]
    if sum(len(p) for p in parts) != len(names):
        return None
    at = fit.with_mstop(int(mstop)) if int(mstop) < fit.mstop else fit
    reduction = at.varimp(percent=True)["reduction"]
    shares, position = {}, 0
    for term, pieces in zip(terms, parts, strict=True):
        owned = names[position:position + len(pieces)]
        position += len(pieces)
        shares[term] = float(sum(reduction.get(name, 0.0) for name in owned))
    return shares


def _is_declared(term, declared):
    from mboostrs.formula import term_variables

    return any(name in declared for name in term_variables(term))


def _group_labels(data, groups):
    """One label per row for the declared group columns, combined."""
    columns = [read_column(data, name) for name in groups]
    if len(columns) == 1:
        return columns[0].values
    return np.array(["|".join(str(column.values[row]) for column in columns)
                     for row in range(len(columns[0].values))])


def _read_treatment(treatment, response, columns):
    """The columns a treatment declaration leaves out of the battery.

    `treatment` is a column name, or a `CausalSpec` -- in which case the
    spec's outcome must be this call's response, and its post-treatment
    columns are left out with the treatment: adjusted-for-baseline modelling
    is the question the spec declares, and the battery does not quietly
    re-admit the columns it excludes.
    """
    from mboostrs.causal import CausalSpec

    if treatment is None:
        return ()
    if isinstance(treatment, CausalSpec):
        if treatment.outcome != response:
            raise ValueError(
                f"the CausalSpec models outcome {treatment.outcome!r} but "
                f"autospec was called with response {response!r}")
        named = (treatment.treatment, *treatment.post_treatment)
    else:
        named = (treatment,)
    for name in named:
        if name == response:
            raise ValueError("treatment must not name the response column")
        if name not in columns:
            raise ValueError(
                f"treatment names {name!r}, which is not a column in the "
                f"data (columns: {', '.join(map(repr, columns))})")
    return tuple(dict.fromkeys(named))


def _time_column(columns, response, dropped):
    """The frame's time axis: the first date or datetime column other than
    the response that hygiene did not drop.

    A column of timestamps orders the rows whatever the specification does
    with it -- modelled as a smooth, named by a prior, or left out -- and
    random folds train on the future to judge the past."""
    for column in columns:
        if column.name == response or column.name in dropped:
            continue
        if column.kind == "datetime":
            return column.name
    return None


def _apply_declarations(modelled, terms, *, df, monotone, coordinates,
                        adjacency, basis):
    """Rewrite ``terms`` for the columns the caller declared knowledge for.

    Mechanism knowledge cannot be guessed from column shape, so it arrives as
    keyword arguments instead: `monotone` writes ``bmono`` with the stated
    constraint, `coordinates` merges two columns into one ``bspatial`` term,
    and `adjacency`/`basis`` attach matrices through the formula parser's
    ``env`` resolution. Returns ``(env, declared_names)``; ``terms`` is
    modified in place.
    """
    import re as _re

    from mboostrs.learners import Bmono

    names = [column.name for column in modelled]
    env: dict = {}
    declared: list[str] = []

    def _require(name, keyword):
        if name not in names:
            raise ValueError(
                f"{keyword} names column {name!r}, which is not a predictor "
                f"in the data (predictors: {', '.join(map(repr, names))}).")

    def _symbol(kind, name):
        return f"_autospec_{kind}_{_re.sub(r'[^A-Za-z0-9_]', '_', name)}"

    if monotone:
        if not isinstance(monotone, dict):
            raise ValueError(
                f"monotone must map column name to constraint; got {monotone!r}")
        for name, direction in monotone.items():
            _require(name, "monotone=")
            if direction not in Bmono.VALID_CONSTRAINTS:
                raise ValueError(
                    f"monotone[{name!r}] must be one of "
                    f"{Bmono.VALID_CONSTRAINTS}, got {direction!r}.")
            terms[names.index(name)] = (
                f"bmono({quote_name(name)}, constraint={direction}, df={df})")
            declared.append(name)

    if coordinates is not None:
        if len(coordinates) != 2 or coordinates[0] == coordinates[1]:
            raise ValueError(
                f"coordinates must name two distinct columns; got {coordinates!r}.")
        for name in coordinates:
            _require(name, "coordinates=")
        first, second = coordinates
        # A tensor surface has a null space of dimension differences**2 = 4,
        # so df=4 (the default the caller passes for marginal smooths) is the
        # null space itself. `Bspatial`'s own default (6, matching R) is the
        # floor; a caller who raised df for the smooths keeps theirs.
        spatial_df = max(float(df), 6.0)
        terms[names.index(first)] = (
            f"bspatial({quote_name(first)}, {quote_name(second)}, df={spatial_df})")
        terms[names.index(second)] = ""
        declared.extend(coordinates)

    if adjacency:
        if not isinstance(adjacency, dict):
            raise ValueError(
                f"adjacency must map column name to a penalty matrix K; "
                f"got {adjacency!r}")
        for name, K in adjacency.items():
            _require(name, "adjacency=")
            K = np.asarray(K, dtype=float)
            if K.ndim != 2 or K.shape[0] != K.shape[1]:
                raise ValueError(
                    f"adjacency[{name!r}] must be a square matrix; got "
                    f"shape {K.shape}.")
            env[_symbol("K", name)] = K
            terms[names.index(name)] = f"bmrf({quote_name(name)}, K={_symbol('K', name)}, df={df})"
            declared.append(name)

    if basis:
        if not isinstance(basis, dict):
            raise ValueError(
                f"basis must map column name to an (X, K) pair; got {basis!r}")
        for name, pair in basis.items():
            _require(name, "basis=")
            try:
                X, K = pair
            except (TypeError, ValueError):
                raise ValueError(
                    f"basis[{name!r}] must be an (X, K) pair; got {pair!r}"
                    ) from None
            env[_symbol("X", name)] = np.asarray(X, dtype=float)
            env[_symbol("K", name)] = np.asarray(K, dtype=float)
            terms[names.index(name)] = (
                f"buser({quote_name(name)}, X={_symbol('X', name)}, "
                f"K={_symbol('K', name)}, df={df})")
            declared.append(name)

    # ``coordinates`` blanks one slot per pair; drop it from terms *and* from
    # modelled so the two stay index-aligned for the probes.
    for i in range(len(terms) - 1, -1, -1):
        if terms[i] == "":
            del terms[i]
            del modelled[i]

    return env, tuple(declared)


def _infer_family(data, response):
    """The family the response calls for, or a reason it cannot be guessed."""
    column = read_column(data, response)
    present = column.values[~np.isnan(column.values)]

    if column.kind == "boolean":
        return Binomial()

    if column.kind == "categorical":
        levels = column.levels or ()
        n_levels = len(levels)
        if n_levels == 2:
            raise ValueError(
                f"response column {response!r} has two levels, which would be "
                f"Binomial, but its values are labels {list(levels)} "
                f"and Binomial reads a numeric {{0, 1}} or boolean response. "
                f"Recode the column first. A three-or-more-level label column "
                f"would infer Multinomial."
            )
        if column.ordered:
            # An ordered categorical (an Enum or an ordered Categorical) is
            # an ordinal response: the declared order is the model's, and an
            # unordered Multinomial would discard it.
            return PropOdds(levels=levels)
        return Multinomial(n_levels)

    # Numeric response: classify by what the numbers are.
    if present.size == 0:
        return Gaussian()
    if np.all((present == 0) | (present == 1)):
        return Binomial()
    if (present.min() >= 0 and np.all(present == np.round(present))
            and not _large_whole_numbers(present)):
        # Whole-number floating measurements and floating count columns have
        # the same values. Retain the established count default, but make the
        # assumption visible so a caller can declare the response meaning.
        source_dtype = getattr(data[response], "dtype", None)
        if source_dtype is not None and "float" in str(source_dtype).lower():
            warnings.warn(
                f"response {response!r} has whole-number values stored as "
                "floating point; autospec assumes counts and selects Poisson. "
                "Pass family=Gaussian() for a rounded continuous measurement "
                "or declare another known family.",
                UserWarning, stacklevel=3)
        return Poisson()
    if _large_whole_numbers(present):
        warnings.warn(
            f"response {response!r} has whole-number values with median "
            f"{float(np.median(present)):,.0f}; at that magnitude a count and a "
            "rounded measurement are indistinguishable, so autospec treats it as "
            "a continuous positive measurement (GammaReg or Gaussian) rather "
            "than Poisson counts. Pass family=Poisson() or NBinomial() to model "
            "it as counts.",
            UserWarning, stacklevel=3)
    if present.min() > 0 and present.max() <= 1:
        warnings.warn(
            f"response {response!r} lies in (0, 1). GammaReg's support "
            "continues past 1, so autospec starts from Gaussian. A proportion "
            "is a BetaLSS mean and precision; give autospec a budget and the "
            "family probe scores BetaLSS against Gaussian on held-out log "
            "score, or pass family=BetaLSS() to glmboostLSS or gamboostLSS.",
            UserWarning, stacklevel=3)
        return Gaussian()
    if present.min() > 0 and not _narrow_positive(present):
        return GammaReg()
    return Gaussian()


#: Whole numbers whose median exceeds this are read as a continuous
#: measurement, not counts.
COUNT_MEDIAN_LIMIT = 1000.0


def _read_exposure(data, response, exposure):
    """The exposure column as floats, after checking it can be an offset.

    `log(exposure)` enters every row's linear predictor, so each row needs a
    finite, positive exposure. A missing or zero value is an error here, not
    a silently dropped row, because dropping rows would change the response
    the caller asked about.
    """
    if exposure == response:
        raise ValueError(
            f"exposure= names the response column {response!r}; name a separate "
            "column of exposures (time at risk, area, number of trials)")
    columns = list(data.columns) if hasattr(data, "columns") else list(data)
    if exposure not in columns:
        raise ValueError(
            f"exposure={exposure!r} is not a column in the data "
            f"(columns: {', '.join(map(repr, columns))})")
    column = read_column(data, exposure)
    if column.kind != "numeric":
        raise ValueError(
            f"exposure column {exposure!r} must be numeric, not {column.kind}")
    values = np.asarray(column.values, dtype=float)
    bad = ~np.isfinite(values) | (values <= 0)
    if bad.any():
        raise ValueError(
            f"exposure column {exposure!r} has {int(bad.sum())} missing or "
            "non-positive values; log(exposure) is the offset of every row, so "
            "fill those exposures or drop those rows before calling autospec")
    return values


#: Whole-number responses with at most this many distinct values are named as
#: possible ordered classes in the report.
_ORDERED_CLASS_MAX = 12


def _response_record(family, column, *, declared, exposure=None):
    """What the response is taken to be, with its source and one report
    sentence.

    The record says whether the family was declared by the caller or read
    from the values, so a rounded measurement that looks like counts, or an
    ordered response read as unordered, is visible in ``rec.report()`` and in
    ``evidence["response"]``.
    """
    name = type(family).__name__
    source = "declared" if declared else "inferred"
    numeric = column.kind in ("numeric", "boolean")
    values = np.asarray(column.values, dtype=float) if numeric else np.array([])
    present = values[~np.isnan(values)]
    whole = bool(present.size) and bool(np.all(present == np.round(present)))
    distinct = int(np.unique(present).size) if present.size else 0
    if exposure is not None:
        kind = "count per exposure"
        sentence = (f"{name} rate model (declared): the counts are modelled per "
                    f"unit of {exposure!r} through log({exposure!r}) as an offset")
    elif name == "Gaussian" and declared and whole:
        kind = "rounded continuous"
        sentence = ("Gaussian (declared): the whole-number values are a rounded "
                    "continuous measurement, not counts")
    elif name == "Gaussian":
        kind = "continuous"
        sentence = f"Gaussian ({source})"
    elif name in ("Poisson", "NBinomial"):
        kind = "counts"
        if declared:
            sentence = f"{name} counts (declared)"
        elif distinct <= _ORDERED_CLASS_MAX:
            sentence = (f"{name} counts (inferred from {distinct} whole-number values; "
                        "if they are rounded measurements pass family=Gaussian(), "
                        "and if they are ordered classes pass "
                        f"family=PropOdds(n_levels={distinct}))")
        else:
            sentence = (f"{name} counts (inferred from whole-number values, {distinct} "
                        "distinct; if they are rounded measurements pass "
                        "family=Gaussian())")
    elif name == "PropOdds":
        kind = "ordered classes"
        n_classes = getattr(family, "n_levels", None)
        sentence = (f"PropOdds ({source}): ordered classes"
                    + (f", {n_classes} levels" if n_classes else "")
                    + "; a single linear predictor with ordered thresholds")
    elif name == "Multinomial":
        kind = "unordered classes"
        sentence = f"Multinomial ({source}): unordered classes"
    elif name == "Binomial":
        kind = "binary"
        sentence = f"Binomial ({source})"
    elif name == "GammaReg":
        kind = "positive continuous"
        sentence = f"GammaReg ({source}): a positive response whose spread grows with its level"
    else:
        kind = name
        sentence = f"{name} ({source})"
    return {"kind": kind, "family": name, "source": source,
            "exposure": exposure, "whole_numbers": whole,
            "distinct_values": distinct, "sentence": sentence}


def _large_whole_numbers(present):
    """Whether whole-number values are too large to be told from a rounded
    measurement.

    Above a median of a thousand, rounding to integers changes a value by
    less than 0.05%, and a Poisson standard deviation (the square root of
    the mean) is under 3% of the mean -- far below the spread real data of
    that size carry, so the Poisson family is never the right model and the
    choice left is the one between constant and growing spread. Poisson's
    negative gradient ``y - mu`` is also on the response's scale, so a fixed
    step on counts near 1e5 overflows the log link in the first iteration
    (house values in California, median 180,000). GammaReg's gradient
    ``y/mu - 1`` is scale-free.
    """
    return bool(np.all(present == np.round(present))
                and float(np.median(present)) > COUNT_MEDIAN_LIMIT)


#: A positive response whose standard deviation is below this fraction of its
#: mean is modelled as Gaussian rather than Gamma.
NARROW_POSITIVE_CV = 0.1


def _narrow_positive(present):
    """Whether a positive response varies too little, relative to its level,
    for a Gamma to differ from a Gaussian.

    A Gamma's variance grows with the square of its mean. Over a range of means
    that differ by a few percent, that growth is a few percent too, which the
    data cannot tell from constant variance. The Gamma's shape is then in the
    thousands, its gradient on the log scale is correspondingly large, and
    boosting it with a fixed step length diverges from the first step: life
    expectancy at 40 across US commuting zones (mean 76, standard deviation
    1.3 years) is such a response.
    """
    mean = float(np.mean(present))
    return mean > 0 and float(np.std(present)) < NARROW_POSITIVE_CV * mean


def _term_for(column, df, smooth, response_values=None):
    """One column's learner, guaranteed to be one the fit path can build."""
    name = column.name
    if column.kind == "boolean":
        return f"bols({quote_name(name)})"

    if column.kind == "categorical":
        n_levels = len(column.levels) if column.levels is not None else 0
        if n_levels <= 2:
            # Two levels are an indicator, whatever the labels: encode_factors
            # reads no ordering into a binary coding.
            return f"bols({quote_name(name)})"
        # A multi-level factor cannot be coded as a measurement. brandom
        # shrinks the level effects and is the fit path's accepted learner.
        return f"brandom({quote_name(name)}, df={df})"

    if column.kind == "datetime":
        return f"bbs({quote_name(name)}, df={df})"

    present = column.values[~np.isnan(column.values)]
    levels = int(np.unique(present).size) if present.size else 0

    # A whole-number column that reads as a count is a measurement, not a
    # set of labels: `brandom` has no coefficient for a count the fit never
    # saw, so `predict()` raises on one (issue #313). The same rule
    # `classify_columns` applies decides first.
    if count_like(column, levels) is not None:
        if levels <= 4:
            return f"bols({quote_name(name)})"
        return f"bbs({quote_name(name)}, df={min(df, levels - 1)})"

    # A numeric column of whole numbers with 3+ distinct values is tentatively
    # read as labels. `bols` would be one slope across the codes, not separate
    # level effects, so the group-mean learner is required at every cardinality.
    discrete = bool(present.size) and bool(np.all(present == np.round(present)))
    if discrete and 3 <= levels <= CATEGORICAL_MAX_LEVELS:
        return f"brandom({quote_name(name)}, df={df})"
    # Above the cutoff a whole-number column may still be a grouping code; the
    # response decides, by the same rule `classify_columns` applies.
    if discrete and response_values is not None:
        evidence = label_evidence(name, column.values, response_values)
        if evidence is not None and evidence.labels:
            return f"brandom({quote_name(name)}, df={df})"

    if levels <= 2:
        return f"bols({quote_name(name)})"
    if smooth == "bbs":
        from mboostrs._autospec import skew_transform

        skewed = skew_transform(present)
        if skewed is not None:
            # The "right-skewed predictor" finding: on the raw scale one
            # cross-validation fold can decide mstop.
            return f"bbs({skewed[0]}({quote_name(name)}), df={df})"
        return f"bbs({quote_name(name)}, df={df})"
    return f"btree({quote_name(name)})"


def _read_prior(prior, response):
    """The formula, family and mstop a caller's prior states."""
    if isinstance(prior, str):
        formula, family, mstop = prior, None, None
    else:
        try:
            formula = prior["formula"]
        except (TypeError, KeyError):
            raise ValueError(
                "prior must be a formula string, or a mapping with a "
                f"'formula' key; got {prior!r}") from None
        family = prior.get("family")
        mstop = prior.get("mstop")
    if "~" not in formula:
        raise ValueError(
            f"prior formula must name the response, as in "
            f"'{quote_response(response)} ~ bbs(x)'; got {formula!r}")
    stated = partition_tilde(formula)[0].strip()
    if stated != response:
        raise ValueError(
            f"prior formula models {stated!r} but the response is {response!r}")
    return formula, family, (int(mstop) if mstop is not None else None)


def _prior_columns(terms):
    """Column names the prior's terms reference."""
    names = set()
    for term in terms:
        inside = term[term.find("(") + 1:term.rfind(")")] if "(" in term else term
        first = inside.split(",")[0].strip()
        if first:
            names.add(first)
    return names
