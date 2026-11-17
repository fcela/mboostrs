"""The recommendation `autospec` returns, and the checks that cost no fit.

`autospec` used to return a four-element tuple guessed from column dtypes.
That is a lookup table: it answers "what learner does a float column get"
and none of the questions a modeller actually has -- whether the family is
right, whether a term needs a different shape, whether an interaction is
carried, how deep to run, whether the data is fit to model at all.

This module holds the result those answers are reported through, and the
first tier of evidence: the checks that read the data and fit nothing. A
recommendation still unpacks as the old tuple, so existing callers keep
working while the object carries what the tuple could not.
"""

from __future__ import annotations

import contextvars
import re
import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from mboostrs._control import BoostControl
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._workflow import BudgetExhausted, call_fit

#: A column whose spread is at or below this fraction of its own scale is
#: constant for modelling purposes: a learner fitted on it has nothing to
#: separate, and the fit spends a term on noise.
_NEAR_CONSTANT_TOLERANCE = 1e-12

#: Missingness at or above this fraction is reported as a finding rather than
#: a note: the rows a fit silently drops are then most of the data.
_HIGH_MISSING_FRACTION = 0.5

#: A float column with no more distinct values than this is a category wearing
#: a continuous dtype, and a spline over it interpolates between labels.
_DISCRETE_UNDER_FLOAT_LEVELS = 3

#: A text column with at least this share of distinct values names rows rather
#: than groups: almost every level holds one row, so no effect can be shared.
_IDENTIFIER_UNIQUE_SHARE = 0.9

#: A whole-number column this close to one value per row is an identifier or a
#: count, and only the analyst knows which.
_UNIQUE_INTEGER_SHARE = 0.99

#: Two columns record one grouping when their distinct value pairs number at
#: most this multiple of the larger column's distinct values ...
_DUPLICATE_PAIR_SLACK = 1.02

#: ... and the smaller column has at least this share of the larger one's
#: values, which excludes a coarser grouping nested in a finer one (states in
#: census divisions).
_DUPLICATE_LEVEL_SHARE = 0.9


@dataclass(frozen=True)
class HygieneFinding:
    """One thing the data says before any model is fitted.

    `column` is the column it concerns (the response included), `detail`
    states what was measured, and `consequence` says what it does to the
    recommendation -- which is the part a modeller acts on.
    """

    column: str
    kind: str
    detail: str
    consequence: str

    def __str__(self) -> str:
        return f"{self.column}: {self.detail} -- {self.consequence}"


@dataclass(frozen=True)
class SpecRecommendation:
    """A specification to start from, with the evidence behind it.

    Unpacks as the four-element tuple `autospec` has always returned --
    ``formula, learners, family, mstop`` -- so a caller written against the
    tuple keeps working:

        formula, learners, family, mstop = mb.autospec(df, "y")

    while a caller that wants the reasoning reads the attributes or calls
    :meth:`report`.
    """

    formula: str
    learners: list
    family: object
    mstop: int
    response: str
    hygiene: tuple[HygieneFinding, ...] = ()
    #: Probe name -> what it measured. Empty when no probe was run.
    evidence: dict = field(default_factory=dict)
    #: Probes that ran, against those that fell back to a dtype default.
    measured: tuple[str, ...] = ()
    defaulted: tuple[str, ...] = ()
    #: Columns whose terms came from the caller — a declaration keyword
    #: (monotone=, coordinates=, adjacency=, basis=) or a prior — rather
    #: than a dtype guess. Terms outside the shape probe's catalogue are
    #: fixed incumbents: alternatives are priced against them, never
    #: substituted.
    declared: tuple[str, ...] = ()
    #: Symbols the returned formula references (matrix declarations), for
    #: every route that re-parses it. Empty unless adjacency= or basis= was
    #: used.
    env: dict = field(default_factory=dict)
    #: Columns declared as groups (``groups=``): fitted with ``brandom`` and
    #: held out whole by the folds the probes share.
    groups: tuple[str, ...] = ()
    #: How the probes' folds were built, in words, for the report.
    folds_description: str = ""
    #: Rows left out of every probe because the response or a modelled column
    #: is missing there. `fit` drops the same rows.
    incomplete_rows: int = 0
    #: Model fits spent producing this recommendation.
    fits: int = 0
    #: Fits a later probe did not make because an earlier one had already
    #: made them: the fold fits shared between the family and shape probes,
    #: and the interaction screen's main-effect path taken from the `mstop`
    #: sweep's full-data fit. `fits` counts each fit once, and `reused` says
    #: how much of the work was shared rather than repeated.
    reused: int = 0
    #: Columns declared as treatment (`treatment=`): left out of the
    #: predictive specification, because scoring a treatment by its
    #: predictive association answers a different question from estimating
    #: its effect. `mboostrs.causal` is the workflow for that one.
    treatment: tuple[str, ...] = ()
    #: Columns declared as ordered quantities (``measurements=``): fitted as a
    #: linear slope, with no sign stated.
    measurements: tuple[str, ...] = ()
    #: Results of `check_declarations` for this recommendation, one record per
    #: declaration. Empty until `with_declaration_checks` is called.
    declaration_checks: tuple = ()
    #: Findings most-consequential first, one line each.
    priority: tuple[str, ...] = ()
    #: The control `fit` replays: the step count the recommendation settled
    #: on with the step length the caller asked for. `None` on a
    #: recommendation built by hand, in which case `fit` uses the default
    #: step length with `mstop`. A `control=` passed to `fit` overrides it.
    control: BoostControl | None = None
    #: The column of exposures the response is a rate per unit of
    #: (``autospec(exposure=)``). `fit` passes its log as the offset.
    exposure: str | None = None

    # -- tuple compatibility -------------------------------------------
    def __iter__(self):
        return iter((self.formula, self.learners, self.family, self.mstop))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return (self.formula, self.learners, self.family, self.mstop)[index]

    def _declarations_lines(self) -> list[str]:
        """The declarations section of :meth:`report`: each declaration with its
        check when one has been run, and otherwise with the reason it is not
        tested."""
        from mboostrs.declarations import declaration_items

        if isinstance(self.formula, dict):
            return []
        items = declaration_items(self)
        if not items:
            return []
        checks = {check.declaration: check for check in self.declaration_checks}
        heading = ("declarations, tested against the data" if checks else
                   "declarations, not tested (call with_declaration_checks(data) to test)")
        lines = [heading]
        for item in items:
            check = checks.get(item["declaration"])
            if check is not None:
                lines.append(f"  {check.agreement}")
            elif item["tested"]:
                lines.append(f"  {item['declaration']}: not tested")
            else:
                lines.append(f"  {item['declaration']}: not tested -- {item['reason']}.")
        lines.append("")
        return lines

    # -- checking declarations -----------------------------------------
    def with_declaration_checks(self, data, *, folds=5, seed=0):
        """A copy of this recommendation with each declaration tested against
        `data`.

        Runs `mboostrs.check_declarations` and stores the records in
        ``declaration_checks``, where :meth:`report` shows them. Costs one
        fit per fold plus one full-data fit per declared sign or shape.
        """
        from mboostrs.declarations import check_declarations

        checks = check_declarations(self, data, folds=folds, seed=seed)
        return replace(self, declaration_checks=tuple(checks))

    # -- using it ------------------------------------------------------
    def fit(self, data, refit=None, **kwargs):
        """Fit the recommended specification, returning a `GAMBoostModel`, or
        an `LSSModel` for a distributional family.

        The recommendation's recorded control -- the step count it settled on
        with the step length `autospec` was called with -- is what replays, so
        ``autospec(df, "y", nu=0.03).fit(df)`` fits at rate 0.03. An explicit
        ``control=`` overrides it, and any other keyword argument reaches
        `gamboost`, so the recommendation can be taken as a starting point and
        adjusted in the same call.

        ``refit="reml"`` finishes with a penalized refit: the boosted fit's
        terms are refitted jointly with REML
        smoothing parameters (`gam_refit`), which removes the shrinkage early
        stopping leaves on every effect and gives intervals with close to
        nominal coverage. Terms the refit cannot represent (`btree`, `bmono`)
        stay at their boosted contribution. It returns a `GamRefit`, whose
        ``held`` lists those terms; the boosted model is the refit's input and
        is not returned, unless the refit's smoothing-parameter iteration did
        not converge: the boosted model is then returned with a warning. When
        `autospec` had no budget to tune `mstop`, and
        so did not run its own deselection step, the boosted fit is
        deselected first (`deselect`, terms under 1% of the risk reduction
        dropped).

        A distributional recommendation (a family with several parameters,
        such as `GaussianLSS`) fits an `LSSModel`, and ``refit="reml"``
        refits it with `gamlss_refit`: every parameter keeps the terms
        boosting selected for it, all coefficients are estimated jointly and
        the smoothing parameters by REML. It returns a `GamlssRefit`. That
        refit has no ``held`` terms: when a parameter's formula carries a
        term it cannot represent (`btree`, `bmono`), or the refit does not
        converge, the boosted model is returned with a warning. Deselection
        runs only on a non-cyclic fit, the one that records each learner's
        share of the risk reduction.
        """
        if refit is not None:
            if refit != "reml":
                raise ValueError('refit must be None or "reml"')
            from mboostrs.selection import deselect, gam_refit

            boosted = self.fit(data, **kwargs)
            if type(boosted).__name__ == "LSSModel":
                return self._refit_distributional(boosted)
            reduced = boosted
            if "mstop" not in self.measured and len(boosted._learner_names) > 1:
                try:
                    reduced = deselect(boosted).model
                except ValueError:
                    reduced = boosted
            try:
                refitted = gam_refit(reduced, unsupported="hold")
            except ValueError as exc:
                if "no term the refit can represent" not in str(exc):
                    raise
                warnings.warn(
                    f"refit='reml' found no term to refit ({exc}); returning the "
                    f"boosted model", RuntimeWarning, stacklevel=2)
                return reduced
            # A refit whose smoothing parameters are not at a REML optimum has
            # no claim to the near-nominal intervals it is adopted for, so the
            # recommendation keeps the boosted model (issue #70). `gam_refit`
            # has already warned with the stop reason.
            convergence = getattr(refitted, "convergence", None) or {}
            if not convergence.get("converged", True):
                warnings.warn(
                    "refit='reml' did not converge (stop reason "
                    f"{convergence.get('stop_reason', 'unrecorded')!r}); returning "
                    "the boosted model. Call gam_refit(model) directly to inspect "
                    "the unconverged refit.", RuntimeWarning, stacklevel=2)
                return reduced
            return refitted
        from mboostrs import boost_control, gamboost, gamboostLSS

        control = kwargs.pop("control", None) or self.control \
            or boost_control(self.mstop)
        if self.exposure is not None and "offset" not in kwargs:
            # The rate model's offset: log(exposure) of every row of `data`.
            # gamboost drops the rows with missing values together with the
            # response, as it does for the offset of any fit.
            from mboostrs._frame import read_column

            exposure = np.asarray(read_column(data, self.exposure).values, dtype=float)
            kwargs["offset"] = np.log(exposure)
        if self.incomplete_rows:
            kwargs.setdefault("na_action", "omit")
        if getattr(self.family, "param_names", None) is not None:
            # A distributional recommendation from the family probe (#228):
            # one formula, one predictor per distribution parameter.
            if self.env and "env" not in kwargs:
                kwargs["env"] = self.env
            return gamboostLSS(self.formula, data=data, families=self.family,
                               control=control, **kwargs)
        if self.learners is None and "learners" not in kwargs:
            # Adopted `%X%` terms: gamboost expands them from the formula.
            if self.env and "env" not in kwargs:
                kwargs["env"] = self.env
            return gamboost(self.formula, data=data, family=self.family,
                            control=control, **kwargs)
        if self.env and "learners" not in kwargs and "env" not in kwargs:
            # gamboost re-parses the formula even when learners are supplied,
            # and the parse needs the symbols the declared terms reference.
            # Rebuilding the learners from the same parse is equivalent.
            kwargs["env"] = self.env
            return gamboost(self.formula, data=data, family=self.family,
                            control=control, **kwargs)
        return gamboost(self.formula, data=data, family=self.family,
                        learners=self.learners, control=control, **kwargs)

    def _refit_distributional(self, boosted):
        """``fit(refit="reml")`` for a distributional recommendation: the
        `gamlss_refit` of the boosted `LSSModel`, or the boosted model with a
        warning when that refit cannot represent a term or does not converge."""
        from mboostrs.selection import deselect, gamlss_refit

        reduced = boosted
        if ("mstop" not in self.measured and boosted._method == "noncyclic"
                and sum(len(terms) for terms in boosted.selected.values()) > 1):
            try:
                reduced = deselect(boosted).model
            except ValueError:
                reduced = boosted
        try:
            refitted = gamlss_refit(reduced)
        except NotImplementedError as exc:
            # gamlss_refit has no counterpart of gam_refit's
            # unsupported="hold": a btree or bmono term in any parameter's
            # formula leaves no refit to adopt.
            warnings.warn(
                f"refit='reml' cannot refit this distributional model ({exc}); "
                f"returning the boosted model", RuntimeWarning, stacklevel=3)
            return reduced
        convergence = refitted.convergence or {}
        if not convergence.get("converged", True):
            warnings.warn(
                "refit='reml' did not converge (stop reason "
                f"{convergence.get('stop_reason', 'unrecorded')!r}); returning "
                "the boosted model. Call gamlss_refit(model) directly to "
                "inspect the unconverged refit.", RuntimeWarning, stacklevel=3)
            return reduced
        return refitted

    def report(self) -> str:
        """The reasoning: what was measured, what was assumed, what to do."""
        head = (self.formula if not isinstance(self.formula, dict)
                else "; ".join(f"{name}: {formula}"
                               for name, formula in self.formula.items()))
        lines = [f"autospec: {head}", ""]
        if self.folds_description:
            rows = (f"; {self.incomplete_rows} incomplete rows left out"
                    if self.incomplete_rows else "")
            lines.extend([f"folds: {self.folds_description}{rows}", ""])

        response = self.evidence.get("response")
        if response is not None:
            lines.extend([f"response: {response['sentence']}", ""])

        lines.append("data hygiene (no fit -- scans only)")
        if self.hygiene:
            lines.extend(f"  {finding}" for finding in self.hygiene)
        else:
            lines.append("  no constant, heavily missing or mistyped column")
        lines.append("")

        family = self.evidence.get("family")
        if family is None:
            if "family" in self.defaulted:
                lines.append(f"family: {type(self.family).__name__} — from the "
                             f"response's dtype; not measured")
            else:
                lines.append(f"family: {type(self.family).__name__} — "
                             "supplied by the caller; no family comparison")
        else:
            lines.append(f"family: {type(self.family).__name__} — "
                         f"{family.sentence} ({family.n_scored} held-out rows)")
        if "search_stopped_at_family" in self.evidence:
            lines.append(f"search: {self.evidence['search_stopped_at_family']}")
        lines.append("")

        spread = self.evidence.get("heteroscedasticity")
        if spread is not None and spread is not self.evidence.get("family"):
            lines.append(f"spread: {spread.sentence}")
            lines.append("")

        shapes = self.evidence.get("shapes", ())
        if shapes:
            lines.append("shapes (held-out risk, shared folds)")
            lines.extend(f"  {row.column}: {row.sentence}" for row in shapes)
        elif self.declared:
            lines.append("shapes: not measured; caller-declared terms and "
                         "dtype defaults stand as written")
        else:
            lines.append("shapes: not measured; each column keeps the learner "
                         "its dtype implies")
        lines.append("")

        if "interactions" in self.measured:
            leads = self.evidence.get("interactions", ())
            count = _candidate_sentence(self.evidence.get("interaction_candidates"))
            if leads:
                lines.append("structure the additive terms miss "
                             "(screened on the training residual, so a lead "
                             "rather than a finding)" + count)
                lines.extend(f"  {lead}" for lead in leads)
            else:
                lines.append("structure: no pair removes enough residual to "
                             "report" + count + "; if you suspect one, fit it explicitly")
            adopted = self.evidence.get("adopted_interactions", ())
            adoption_evidence = self.evidence.get("adoption_evidence", {})
            if adopted and adoption_evidence:
                lines.append("  adopted with held-out evidence: " + ", ".join(
                    f"{term} ({rel:.1%}, {z:.1f} paired SE over a "
                    f"{z_bar:.1f}-SE bar, {n} folds)"
                    for term in adopted
                    for (n, rel, z, z_bar) in [adoption_evidence.get(term, (0, 0.0, 0.0, 2.0))]))
            rejected = self.evidence.get("rejected_interactions", ())
            if rejected:
                lines.append("  adopted, then removed because held-out risk did not "
                             "need them: " + ", ".join(
                                 f"{term} ({rel:.1%}, {z:.1f} paired SE)"
                                 for term, rel, z in rejected))
            for entry in self.evidence.get("interaction_structure", ()):
                if entry["term"] in adopted:
                    lines.append(f"  {_structure_sentence(entry)}")
        else:
            lines.append("structure: " + self.evidence.get("interactions_skipped",
                                                           "not screened"))
        rule = self.evidence.get("interaction_rule")
        if rule:
            parts = [f"heredity {rule['heredity']}"]
            if rule["forbid"]:
                parts.append("forbidden pairs " + ", ".join(rule["forbid"]))
            if rule["mandatory"]:
                parts.append("mandatory " + ", ".join(rule["mandatory"]))
            lines.append("  interaction rule: " + "; ".join(parts))
        for key, label in (("forbidden_interactions", "not listed (forbidden)"),
                           ("heredity_barred_interactions",
                            "not adopted (heredity: a parent main effect is absent)")):
            if self.evidence.get(key):
                lines.append(f"  {label}: " + ", ".join(self.evidence[key]))
        unvalidated = self.evidence.get("unvalidated_interactions")
        if unvalidated:
            lines.append("  suggested only: " + ", ".join(unvalidated)
                         + " — insufficient budget or failed sweep; held-out validation is required")
        screen = self.evidence.get("column_screen")
        if screen:
            by_hint = {}
            for row in screen:
                by_hint.setdefault(row["hint"], []).append(row["column"])
            parts = [f"{hint}: {', '.join(cols)}" for hint in
                     ("curve", "step", "line", "levels", "none") if (cols := by_hint.get(hint))]
            lines.append(f"column screen (marginal, no fit; {len(screen)} candidate "
                         f"columns): " + "; ".join(parts))
        exploration = self.evidence.get("preexploration")
        if exploration is not None:
            narrowed = [f"{row['column']}: {'/'.join(row['shapes'])} ({row['reason']})"
                        for row in exploration["columns"] if row["reason"] != "ambiguous"]
            detail = ("shape hints: " + "; ".join(narrowed) if narrowed
                      else "shape comparisons remain open")
            if not exploration["shape_pruning"]:
                detail = "pair proposals only; shapes are not pruned for this probe/family"
            proposed = len(exploration.get("pairs", ()))
            lines.append(f"pre-exploration ({exploration['engine']}, {exploration['fits']} probe fit(s); "
                         f"{proposed} candidate pairs): "
                         + detail
                         + "; proposals still require held-out comparison")
            if exploration.get("neural_failure"):
                lines.append("  neural probe unavailable; tree proposals retained: "
                             + exploration["neural_failure"])
            for subset in [s for s in exploration.get("subsets", []) if s.get("nominated")][:4]:
                lines.append("  subset lead (unvalidated): " + ", ".join(subset["columns"])
                             + "; total nonadditivity may include lower-order interactions")
        if "preexploration_skipped" in self.evidence:
            lines.append("pre-exploration: " + self.evidence["preexploration_skipped"])
        detection = self.evidence.get("residual_tree_detection")
        if detection is not None:
            lines.append(
                f"residual tree screen (a lead, on training rows): the additive fit "
                f"leaves {detection['leftover_share']:.1%} of the response variance; "
                f"a depth-3 tree reads {detection['tree_r2']:.1%} of that leftover, "
                f"product {detection['product']:.4f}; {detection['candidates']} candidate tree")
        tree = self.evidence.get("residual_tree")
        if tree is not None:
            if "relative_gain" not in tree:
                lines.append(f"residual tree: not assessed — {tree.get('reason', '')}")
            elif tree["adopted"]:
                share = tree.get("share")
                carries = (f"; on training rows it carries {share['covariance_share']:.1%} "
                           f"of the variance of the prediction and "
                           f"{share['absolute_share']:.1%} of the absolute contribution"
                           if share else "")
                lines.append(
                    f"residual tree: {tree['term']} added — {_held_out_change(tree)} "
                    f"({tree['relative_gain']:.1%} relative, {tree['z']:.1f} paired SE "
                    f"over {tree['folds']} folds){carries}; it is a prediction term, "
                    f"not an effect curve")
            else:
                lines.append(
                    f"residual tree: not added — {tree['term']} {_held_out_change(tree)} "
                    f"({tree['z']:.1f} paired SE), under the 2% or two-SE rule; the "
                    f"tree carries none of the fit")
        lines.append("")

        lines.extend(self._declarations_lines())

        deselected = self.evidence.get("deselect")
        if deselected is not None:
            lines.append(f"terms: {deselected.sentence}")
            lines.append("")

        surface = self.evidence.get("coordinate_surface")
        if surface is not None:
            x, y = surface["pair"]
            lines.append(f"coordinate pair {x}, {y} (held-out risk, shared folds; "
                         f"{len(surface['candidates'])} candidates)")
            for candidate in surface["candidates"]:
                risk = candidate["risk"]
                if not np.isfinite(risk):
                    detail = "sweep failed"
                else:
                    detail = (f"risk {risk:.4f}, {candidate['relative_gain']:+.1%} against "
                              f"the additive pair ({candidate['z']:.1f} paired SE)")
                lines.append(f"  {candidate['term']}: {detail}")
            if surface["adopted"]:
                lines.append(f"  adopted: {surface['formula']} (mstop {surface['mstop']})")
                for entry in self.evidence.get("interaction_structure", ()):
                    if entry["term"].startswith("bspatial("):
                        lines.append(f"  {_structure_sentence(entry)}")
            elif surface["candidates"]:
                lines.append(f"  additive pair stands: {surface['reason']}")
            else:
                lines.append(f"  not compared: {surface['reason']}")
            lines.append("")

        joint = self.evidence.get("joint_tree")
        if joint is not None:
            lines.append(f"joint tree (held-out risk, shared folds; "
                         f"{len(joint['candidates'])} candidates)")
            if "incumbent_risk" in joint:
                lines.append(f"  incumbent risk {joint['incumbent_risk']:.4f}, "
                             f"nu {joint['incumbent_nu']:g}; joint-tree nu {joint['nu']:g}; "
                             f"grid through {joint['largest']} steps")
            for candidate in joint["candidates"]:
                risk = candidate["risk"]
                detail = ("sweep failed" if risk is None else
                          f"risk {risk:.4f}, mstop {candidate['mstop']}"
                          + (" (grid boundary)" if candidate["at_boundary"] else ""))
                lines.append(f"  {candidate['term']}: {detail}")
            if joint["adopted"]:
                lines.append(f"  replaces {joint['incumbent']} with {joint['formula']}: "
                             f"{joint['relative_gain']:.1%} less risk, over the 2% rule")
            else:
                lines.append(f"  incumbent stands: {joint['reason']}")
            lines.append("  tuning and selection use these folds; measure generalisation "
                         "on separate test data")
            lines.append("")

        whole = self.evidence.get("whole_model")
        if whole is not None:
            if whole.get("compared"):
                lines.append("whole model (held-out risk, shared folds): "
                             + whole["sentence"])
            else:
                lines.append(f"whole model: not compared — {whole['reason']}")
            lines.append("")

        mstop = self.evidence.get("mstop")
        if mstop is None:
            lines.append(f"mstop: {self.mstop} — the default; not measured")
        else:
            lines.append(f"mstop: {self.mstop} — {mstop.sentence}")
        if "nu" in self.evidence:
            lines.append(f"step length: {self.evidence['nu']}")
        elif self.control is not None and self.control.nu != _default_autospec_nu():
            lines.append(f"step length: nu={self.control.nu:g} — the rate the "
                         "caller asked for; rec.fit() replays it")
        if "step" in self.evidence:
            lines.append(f"step rule: {self.evidence['step']}")
        lines.append("")

        lines.append("priority (most consequential first)")
        for index, line in enumerate(self.priority, start=1):
            lines.append(f"  {index}. {line}")
        lines.append("")

        if self.defaulted:
            lines.append("not measured: " + ", ".join(self.defaulted)
                         + " — raise `budget=` to measure them")
        spent = f"model fits spent: {self.fits}"
        if self.reused:
            spent += f" ({self.reused} of them reused from an earlier probe)"
        lines.append(spent)
        return "\n".join(lines)


@dataclass(frozen=True)
class DeselectEvidence:
    """Terms dropped for a small share of the tuned fit's risk reduction."""

    #: ``(term, share)`` for each dropped term, share as a fraction; empty
    #: when the held-out check kept them.
    dropped: tuple
    threshold: float
    sentence: str
    #: Relative rise of the minimum cross-validated risk when the terms are
    #: dropped (negative when the reduced formula does better); None when it
    #: was not measured.
    held_out_loss: float | None = None


def interaction_structure(formula):
    """One record per two-column term in the formula: its kind, its columns,
    and whether each column's main effect is also a term of the formula.

    A two-column term is a tensor surface (``bbs(x) %X% bbs(y)`` or
    ``bspatial(x, y)``), a varying-coefficient term (``bbs(x, by=b)``,
    ``bols(x, by=b)``) or a factor product (``brandom(g) %X% brandom(b)``). A
    tensor surface spans the marginal shape of each of its columns, so part of
    its fitted surface is the main effect of each column, whether or not that
    main effect is also in the formula. Nothing here purifies the surface: the
    record says so with ``purified: False``. ``spans_marginals`` is decided for
    tensor surfaces only and is None for the other kinds, which are not
    classified.
    """
    from mboostrs.explore import split_terms
    from mboostrs.formula import term_variables

    if not isinstance(formula, str) or "~" not in formula:
        return ()
    terms = [t.strip() for t in split_terms(partition_tilde(formula)[2])]
    mains = {term_variables(t)[0] for t in terms
             if len(term_variables(t)) == 1 and "%X%" not in t}
    records = []
    for term in terms:
        if len(term_variables(term)) != 2 or term.startswith("btree("):
            continue
        columns = tuple(term_variables(term))
        if term.startswith("bspatial(") or (term.startswith("bbs(") and "%X%" in term):
            kind, spans = "tensor", True
        elif "%X%" in term:
            kind, spans = "factor-by-factor", None
        else:
            kind, spans = "varying", None
        records.append({"term": term, "kind": kind, "columns": columns,
                        "marginal_in_formula": {c: c in mains for c in columns},
                        "spans_marginals": spans, "purified": False})
    return tuple(records)


def _candidate_sentence(counts):
    """The candidate count a screen reports, as a parenthesised clause."""
    if not counts:
        return ""
    parts = [f"{counts['residual_pairs']} pairs screened on the residual"]
    if counts.get("surrogate_pairs"):
        parts.append(f"{counts['surrogate_pairs']} proposed by the surrogate")
    return " (" + "; ".join(parts) + ")"


def _structure_sentence(entry):
    """One line on a pair term's kind and its main effects, for the report."""
    present = [c for c, yes in entry["marginal_in_formula"].items() if yes]
    absent = [c for c, yes in entry["marginal_in_formula"].items() if not yes]
    main = ("main effects in the formula: " + (", ".join(present) or "none")
            + ("; not in the formula: " + ", ".join(absent) if absent else ""))
    text = f"{entry['term']}: {entry['kind']}; {main}."
    if entry["spans_marginals"]:
        text += (" The surface spans the marginal shape of each column, so it "
                 "includes their main effects; it is not purified against them.")
    else:
        text += " Not purified."
    return text


def _held_out_change(tree):
    """How a residual tree's held-out risk moved, from its recorded gain."""
    gain = tree.get("held_out_gain")
    if gain is None or not np.isfinite(gain):
        return "changes the held-out risk by an unmeasured amount"
    if gain >= 0:
        return f"lowers the held-out risk by {gain:.4g} (fold mean)"
    return f"raises the held-out risk by {-gain:.4g} (fold mean)"


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


#: A positive predictor whose sample skewness exceeds this is fitted on the
#: log scale. With an even knot grid over the raw range, most basis
#: functions of such a column sit over a few extreme rows; a
#: cross-validation fold that holds those rows out extends the spline across
#: the empty stretch, its held-out risk explodes, and the tuned `mstop`
#: follows that one fold (California housing's `AveOccup`, skewness near 40,
#: stopped a 5-fold search at 11 of 2000 steps).
_PREDICTOR_SKEW_FOR_LOG = 2.0

#: Above this many rows the report recommends the binned route.
_BINNING_ROWS = 100_000


def right_skew(values):
    """Sample skewness of a strictly positive numeric column, else ``None``."""
    values = np.asarray(values, dtype=float)
    present = values[np.isfinite(values)]
    if len(present) <= 3 or not np.all(present > 0):
        return None
    scale = float(np.std(present))
    if scale == 0:
        return None
    return float(np.mean(((present - np.mean(present)) / scale) ** 3))


def skew_transform(values):
    """`(transform, skewness)` for a right-skewed predictor, else None.

    A strictly positive column with skewness above
    `_PREDICTOR_SKEW_FOR_LOG` is fitted on the log scale. A non-negative
    column that holds zeros cannot be logged; its square root is defined at
    zero and pulls in the long right tail without a constant that would
    depend on the column's units (socmob's occupation counts, 0 to 746 with
    skewness 5, left a raw-scale spline that extrapolated to -790).
    """
    values = np.asarray(values, dtype=float)
    present = values[np.isfinite(values)]
    if len(present) <= 3 or np.any(present < 0):
        return None
    scale = float(np.std(present))
    if scale == 0:
        return None
    skew = float(np.mean(((present - np.mean(present)) / scale) ** 3))
    if skew <= _PREDICTOR_SKEW_FOR_LOG:
        return None
    return ("log" if np.all(present > 0) else "sqrt"), skew


def hygiene_findings(data, response: str, columns, groups=(), *,
                     caller_folds: bool = False,
                     treatment=(), measurements=(),
                     binary_response: bool = False
                     ) -> tuple[HygieneFinding, ...]:
    """What the data says about itself, before anything is fitted.

    `columns` are the `read_column` records for every column, response
    included. Each check states what it measured and what the recommendation
    does about it, because a finding a modeller cannot act on is noise.
    `groups` are the columns the caller declared as groups;
    `caller_folds` says the caller supplied `folds=`, in which case the
    fold design is theirs and the findings describe it as such; `treatment`
    names the columns the caller declared as treatment (or, with a
    `CausalSpec`, its post-treatment columns), which the recommendation
    leaves out of the predictive specification; `measurements` names the
    columns the caller declared as ordered quantities, whose terms and
    findings follow the declaration; `binary_response` says the inferred
    family is already `Binomial`, in which case a response of 0s and 1s
    is the family's own coding and the zero-mass and few-valued findings
    are not raised: they would warn about a model that fits.
    """
    from mboostrs.terms import count_like, label_evidence

    findings: list[HygieneFinding] = []
    n_rows = max((len(column.values) for column in columns), default=0)
    response_column = next((column for column in columns if column.name == response), None)
    response_values = (response_column.values
                       if response_column is not None
                       and response_column.kind in ("numeric", "boolean") else None)

    for name in treatment:
        if name == response:
            continue
        findings.append(HygieneFinding(
            name, "treatment", "declared as treatment, not as a predictor",
            "left out of the predictive specification. Predictive "
            "association is not an identification argument: for the "
            "effect of a treatment, use `mboostrs.causal` (`CausalSpec`, "
            "`aipw`, `negative_control`) rather than this fit"))

    for column in columns:
        values = np.asarray(column.values)
        present = _finite(values) if values.dtype.kind in "fiu" else values
        missing = n_rows - len(present)

        if len(present) == 0:
            findings.append(HygieneFinding(
                column.name, "empty", "every value is missing",
                "no learner is produced for it"))
            continue

        if values.dtype.kind in "fiu":
            spread = float(np.max(present) - np.min(present))
            scale = max(abs(float(np.mean(present))), 1.0)
            if spread <= _NEAR_CONSTANT_TOLERANCE * scale:
                findings.append(HygieneFinding(
                    column.name, "constant",
                    f"one value ({float(present[0]):g}) in {len(present)} rows",
                    "no learner is produced for it: a term on a constant "
                    "fits the intercept twice"))
                continue

        if missing and n_rows:
            fraction = missing / n_rows
            if fraction >= _HIGH_MISSING_FRACTION:
                findings.append(HygieneFinding(
                    column.name, "missing",
                    f"{missing} of {n_rows} rows missing ({fraction:.0%})",
                    "a fit drops those rows for every term, so the model is "
                    "trained on the remainder -- decide whether that "
                    "remainder is the population you mean"))
            else:
                findings.append(HygieneFinding(
                    column.name, "missing",
                    f"{missing} of {n_rows} rows missing ({fraction:.0%})",
                    "the probes use the complete rows, and `rec.fit` leaves "
                    "the incomplete ones out (`na_action=\"omit\"`)"))

        if (column.name != response and values.dtype.kind == "f"
                and len(np.unique(present)) <= _DISCRETE_UNDER_FLOAT_LEVELS):
            levels = np.unique(present)
            if column.name in measurements:
                consequence = (
                    "declared as measurements: fitted with one slope (bols), "
                    "and the shape probe prices the smooth and the threshold "
                    "against it")
            elif count_like(column, len(levels)) is not None:
                consequence = (
                    "read as a count: fitted with one effect that extends to "
                    "counts the fit never saw, not one level per value. "
                    f"Declare groups=\"{column.name}\" if the values are "
                    "labels rather than counts")
            else:
                consequence = (
                    "read as groups, not as a measured quantity: a spline "
                    "would interpolate between labels. Declare "
                    f"measurements=\"{column.name}\" if the values are "
                    "ordered quantities")
            findings.append(HygieneFinding(
                column.name, "discrete-under-float",
                f"{len(levels)} distinct values ({', '.join(f'{v:g}' for v in levels)}) "
                "in a float column",
                consequence))

        labelled = False
        if (column.name != response and column.kind == "numeric"
                and response_values is not None
                and count_like(column, len(np.unique(present))) is None):
            evidence = label_evidence(column.name, values, response_values)
            labelled = evidence is not None and evidence.labels
            if evidence is not None and evidence.labels:
                if column.name in measurements:
                    consequence = (
                        "declared as measurements, so fitted with one slope "
                        "rather than group effects")
                elif column.name in groups:
                    consequence = ("declared as groups: fitted with brandom, and "
                                   "the probes hold whole groups out")
                elif caller_folds:
                    consequence = (
                        "fitted with brandom rather than a smooth. The folds "
                        "you supplied decide how these groups are held out")
                else:
                    consequence = (
                        "fitted with brandom rather than a smooth, and "
                        "detected as a grouping: the probes' folds hold "
                        "whole groups out, so every measurement scores rows "
                        "whose group the fit has not seen. Declare folds= to "
                        "score a different population")
                findings.append(HygieneFinding(
                    column.name, "integer-labels", evidence.sentence, consequence))

        if (column.name != response and values.dtype.kind in "fiu"
                and column.kind == "numeric"
                and column.name not in measurements and column.name not in groups
                and not labelled
                and count_like(column, len(np.unique(present))) is None):
            skewed = skew_transform(present)
            if (skewed is not None
                    and len(np.unique(present)) > _DISCRETE_UNDER_FLOAT_LEVELS):
                kind, skew = skewed
                findings.append(HygieneFinding(
                    column.name, "right-skewed predictor",
                    f"sample skewness {skew:.1f}, range {float(np.min(present)):g} "
                    f"to {float(np.max(present)):g}",
                    f"fitted as bbs({kind}({column.name})): on the raw scale an even knot "
                    f"grid puts most of the spline over a few extreme rows, and a "
                    f"cross-validation fold that holds them out extends the curve "
                    f"across the empty range, so one fold can decide mstop"))

    findings.extend(_frame_findings(response, columns))
    if n_rows > _BINNING_ROWS:
        # G-12 in the benchmark record: the exact route keeps three copies of
        # every spline design (about 6.6 KB per row for ten terms at 10^6
        # rows); the binned route stores the basis at the bins.
        findings.append(HygieneFinding(
            response, "large frame",
            f"{n_rows:,} rows",
            "pass bins=256 to the fit (rec.fit(data, bins=256)): the binned "
            "route stores each spline at 256 values instead of every row, "
            "about a tenth of the exact route's memory at 10^6 rows, and "
            "check_binning() measures what it changes"))
    findings.extend(_response_findings(data, response, columns,
                                       binary_response=binary_response))
    return tuple(findings)


def _name_tokens(name: str) -> list[str]:
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", str(name)) if token]


def _frame_findings(response: str, columns) -> list[HygieneFinding]:
    """Columns that are not predictors: identifiers, the response's siblings,
    and a second encoding of a grouping another column already records."""
    findings: list[HygieneFinding] = []
    predictors = [column for column in columns if column.name != response]
    response_column = next((column for column in columns if column.name == response), None)

    for column in predictors:
        present = column.values[np.isfinite(column.values)]
        if present.size < 20:
            continue
        distinct = int(np.unique(present).size)
        share = distinct / present.size
        if column.kind == "categorical" and share >= _IDENTIFIER_UNIQUE_SHARE:
            findings.append(HygieneFinding(
                column.name, "identifier",
                f"{distinct} distinct labels in {present.size} rows",
                "no learner is produced for it: a label that names almost every "
                "row has no rows across which to share an effect"))
        elif (column.kind == "numeric" and share >= _UNIQUE_INTEGER_SHARE
                and np.all(present == np.round(present))):
            findings.append(HygieneFinding(
                column.name, "unique-integer",
                f"{distinct} distinct whole numbers in {present.size} rows",
                "fitted as a measurement: right if the numbers are quantities "
                "(a population, a count); leave it out if they are codes that "
                "identify rows"))

    if response_column is not None:
        target = _name_tokens(response)
        y = response_column.values
        for column in predictors:
            tokens = _name_tokens(column.name)
            if len(target) < 2 or not tokens:
                continue
            extra = [token for token in tokens if token not in target]
            missing = [token for token in target if token not in tokens]
            contains = not missing and bool(extra)
            parallel = (len(target) >= 3 and len(tokens) == len(target)
                        and len(extra) == 1 and len(missing) == 1)
            if not (contains or parallel):
                continue
            both = np.isfinite(column.values) & np.isfinite(y)
            correlation = ""
            if (both.sum() > 3 and np.std(column.values[both]) > 0
                    and np.std(y[both]) > 0):
                r = float(np.corrcoef(column.values[both], y[both])[0, 1])
                correlation = f", correlation {r:+.2f} with the response"
            how = (f"contains the response's name with {', '.join(extra)} added"
                   if contains else
                   f"is the response's name with {missing[0]} replaced by {extra[0]}")
            findings.append(HygieneFinding(
                column.name, "response-sibling",
                f"its name {how}{correlation}",
                "a parallel outcome, or a statistic computed from the response, "
                "is not a predictor: keep it only if its value is known before "
                "the response is, and otherwise leave it out"))

    # Only label-like columns can be two encodings of one grouping: text
    # categories, and whole-number codes. A numeric measurement that takes one
    # value per group (a region's income) maps one-to-one onto the group code
    # too, but it has an order and a size a new group can be predicted from.
    def label_like(column):
        if column.kind == "categorical":
            return True
        if column.kind != "numeric":
            return False
        present = column.values[np.isfinite(column.values)]
        return bool(present.size) and bool(np.all(present == np.round(present)))

    candidates = [column for column in predictors
                  if label_like(column)
                  and 2 < column.n_levels <= len(column.values) / 2]
    dropped: set[str] = set()
    for index, first in enumerate(candidates):
        if first.name in dropped:
            continue
        for second in candidates[index + 1:]:
            if second.name in dropped:
                continue
            both = np.isfinite(first.values) & np.isfinite(second.values)
            if both.sum() < 20:
                continue
            _, a = np.unique(first.values[both], return_inverse=True)
            _, b = np.unique(second.values[both], return_inverse=True)
            ka, kb = int(a.max()) + 1, int(b.max()) + 1
            pairs = int(np.unique(a.astype(np.int64) * kb + b).size)
            larger, smaller = max(ka, kb), min(ka, kb)
            if (pairs <= _DUPLICATE_PAIR_SLACK * larger
                    and smaller >= _DUPLICATE_LEVEL_SHARE * larger):
                dropped.add(second.name)
                findings.append(HygieneFinding(
                    second.name, "duplicate-encoding",
                    f"records the same grouping as {first.name}: {pairs} distinct "
                    f"value pairs for {ka} and {kb} values",
                    f"no learner is produced for it; {first.name} carries the "
                    f"grouping, and fitting both would count it twice"))
    return findings


def _response_findings(data, response: str, columns, *,
                       binary_response: bool = False) -> list[HygieneFinding]:
    """What the response's own shape says about the family that fits it.

    `binary_response` suppresses the zero-mass and few-valued findings: when
    the family models class labels (`Binomial`, `Multinomial`, `PropOdds`
    ...), a response of a few integer codes is the family's coding, not a
    warning sign, and findings about it would send the modeller hunting for
    a defect that is not there.
    """
    column = next((c for c in columns if c.name == response), None)
    if column is None:
        return []
    values = np.asarray(column.values)
    if values.dtype.kind not in "fiu":
        return []
    present = _finite(values)
    if len(present) == 0:
        return []

    findings: list[HygieneFinding] = []
    zeros = int(np.sum(present == 0.0))
    if (not binary_response and zeros and np.all(present >= 0)):
        fraction = zeros / len(present)
        if fraction >= 0.3:
            findings.append(HygieneFinding(
                response, "zero-inflated",
                f"{zeros} of {len(present)} values are exactly zero ({fraction:.0%})",
                "a count or positive family will fit the mass at zero through "
                "the mean. scan_lss_families(..., support='counts') ranks "
                "ZIPoLSS and ZeroAdjustedPoissonLSS; autospec's family probe "
                "does not try them"))

    distinct = np.unique(present)
    if (not binary_response and len(distinct) <= _DISCRETE_UNDER_FLOAT_LEVELS
            and values.dtype.kind == "f"):
        findings.append(HygieneFinding(
            response, "few-valued",
            f"{len(distinct)} distinct values in a float response",
            "a continuous family is being asked to model what looks like a "
            "label; check the coding before trusting the fit"))

    if len(present) > 3 and np.all(present > 0):
        centred = present - np.mean(present)
        scale = np.std(present)
        if scale > 0:
            skew = float(np.mean((centred / scale) ** 3))
            if skew > 2.0:
                findings.append(HygieneFinding(
                    response, "right-skewed",
                    f"sample skewness {skew:.1f}, minimum {float(np.min(present)):g}",
                    "a Gaussian mean fits the bulk and misses the tail; the "
                    "family probe compares Gamma and the count families"))
    return findings


# ── Q1: which family does the response come from ──────────────────────────
#
# The dtype rule puts `Poisson` on any non-negative integer column, and an
# overdispersed count looks exactly like a Poisson until the fit overfits its
# noise. What separates the two is not the mean -- both fit nearly the same
# one -- but the variance, so that is what the probe measures, on held-out
# rows where an overfitted mean cannot hide the spread.


@dataclass(frozen=True)
class FamilyEvidence:
    """What the held-out residuals say about the response's distribution."""

    recommended: object
    #: Pearson dispersion, mean((y - mu)^2 / V(mu)), on out-of-fold rows. One
    #: is what the assumed family predicts; above one is overdispersion.
    dispersion: float
    #: Spearman-style association between |residual| and the fitted mean. A
    #: Gamma's spread grows with the mean; a Gaussian's does not.
    spread_tracks_mean: float
    n_scored: int
    fits: int
    sentence: str
    #: True when the measurement fell too close to the decision line to
    #: distinguish the candidates, so the probe kept the dtype-inferred
    #: family and reported both sides rather than choosing. `recommended`
    #: is then the inferred family, not a measured verdict.
    declined: bool = False
    #: Held-out squared error of GammaReg and Gaussian, each at its own
    #: cross-validated step count, and the paired standard error of their
    #: difference. Set when the spread pointed at GammaReg and the budget
    #: allowed the check.
    gamma_mse: float = float("nan")
    gaussian_mse: float = float("nan")
    mse_difference_se: float = float("nan")
    #: Held-out squared error of the non-Gaussian candidate the check
    #: compared (GammaReg, Poisson or NBinomial).
    candidate_mse: float = float("nan")
    #: Held-out log score in nats per row of the distributional family the
    #: support pointed at (BetaLSS inside (0, 1), GammaLSS for a growing
    #: spread) and of the mean family it was measured against, plus the
    #: paired standard error of their difference. Set when the budget
    #: allowed the check (#228).
    lss_score: float = float("nan")
    mean_log_score: float = float("nan")
    score_difference_se: float = float("nan")
    #: The per-parameter step count the distributional family's own
    #: cross-validation chose, in family-parameter order, when the LSS
    #: candidate was adopted.
    lss_mstop: tuple = ()
    #: The multi-parameter formula a non-support probe adopted, a mapping
    #: from family parameter to formula (``{"mu": ..., "sigma": ...}``) when
    #: the response's spread moved with a covariate and GaussianLSS carried
    #: it (#322). `None` when the LSS candidate, if any, fitted the mean's
    #: single formula for every parameter.
    lss_formula: object = None
    #: The step count the mean family was tuned to by cross-validation (0
    #: when no tuning was run), so a later check scores the same mean model.
    mean_mstop: int = 0
    #: Gradient stabilization the distributional candidate was fitted with
    #: ("none", "MAD" or "L2"), and whether its cross-validated score was
    #: still improving at the largest step count tried (`lss_at_boundary`),
    #: with that largest step count (`lss_largest`).
    lss_stabilization: str = ""
    lss_at_boundary: bool = False
    lss_largest: int = 0
    #: The distributional candidate's mean check: held-out squared error of
    #: its mean (mu on the response scale) and of the mean family at its own
    #: tuned step count, and the paired standard error of their difference.
    #: `lss_mean_check` is "passed", "failed" or "" (not run).
    lss_mse: float = float("nan")
    mean_family_mse: float = float("nan")
    lss_mse_se: float = float("nan")
    lss_mean_check: str = ""
    #: The mean family the distributional candidate was adopted over (the
    #: family the single-predictor probes continue with), set when the
    #: candidate passed the preliminary checks. `None` otherwise.
    mean_family: object = None


#: Dispersion this far above one is more than sampling noise at the sizes
#: these probes run on, and says the count family is too tight.
_OVERDISPERSION = 1.5
#: Dispersion this far below one says the counts vary less than a Poisson
#: mean allows -- bounded scores, percentages, ratings -- and a
#: constant-variance Gaussian describes them better.
_UNDERDISPERSION = 1.0 / _OVERDISPERSION
#: Correlation between |residual| and the mean above this says the spread
#: grows with the level, which is Gamma's shape, not Gaussian's.
_SPREAD_TRACKS = 0.25
#: How many standard errors a measurement must clear the decision line by
#: before it chooses. A Pearson dispersion has variance about 2/n under the
#: Poisson it is testing, and a correlation about 1/n under independence;
#: a measurement inside the band is sampling noise around the line, and the
#: probe declines to choose and says so, rather than reporting a verdict
#: the same data could have produced on the other side.
_FAMILY_DECISION_SES = 2.0


#: The step rule every probe fits with. `autospec` sets it to
#: "line_search" when a fixed-step fit of the specification diverges.
_STEP = contextvars.ContextVar("autospec_step", default="fixed")

#: The per-row log-exposure offset every probe fit carries, for a call that
#: declared `exposure=`. `autospec` sets it on the complete rows; None means
#: the response is not a rate.
_EXPOSURE = contextvars.ContextVar("autospec_exposure", default=None)


def _offset_kwargs():
    """`{"offset": ...}` for a probe fit under a declared exposure, else `{}`."""
    offset = _EXPOSURE.get()
    return {} if offset is None else {"offset": offset}


def _default_autospec_nu():
    from mboostrs.autospec import _AUTOSPEC_NU

    return _AUTOSPEC_NU


def _control(mstop, nu):
    """The probes' control record, at the step rule of the current call."""
    from mboostrs import boost_control

    return boost_control(int(mstop), nu, step=_STEP.get())


class _FitCache:
    """The fits the probes share within one `autospec` call.

    Every probe fits specifications on the same fold matrix, and some of
    those specifications repeat: the family probe's mean-model folds are the
    shape probe's incumbent when the family stands. The cache keys a fold
    fit on everything that distinguishes it -- formula, family, step count,
    step length, fold column -- and a call's `env` and folds are constant,
    so a hit is the same fit. It exists only for one `autospec` call;
    nothing outside the call can reach it.
    """

    def __init__(self):
        self._models: dict = {}
        #: Fits a later probe took from the cache rather than making.
        self.reused = 0

    def fold_fit(self, data, formula, family, folds, column, mstop, nu, env):
        """The fit for `(formula, family, mstop, nu)` on fold `column`.

        Returns `(model, was_cached)`.
        """
        from mboostrs import gamboost
        from mboostrs.families import _family_params

        key = (formula, type(family).__name__,
               repr(_family_params(family)), int(mstop), float(nu),
               int(column))
        hit = self._models.get(key)
        if hit is not None:
            self.reused += 1
            return hit, True
        fit = gamboost(formula, data=data, family=family, env=env,
                       control=_control(mstop, nu),
                       weights=np.asarray(folds, dtype=float)[:, column],
                       **_offset_kwargs())
        self._models[key] = fit
        return fit, False


def _out_of_fold_mean_or_search(data, response, formula, family, folds, mstop, nu,
                                env=None, cache=None, judge=None):
    """`_out_of_fold_mean`, repeated with the line search when the fixed step
    makes a log-link fit overflow; only these fits change step rule."""
    from mboostrs._errors import MboostrsRuntimeError

    try:
        return _out_of_fold_mean(data, response, formula, family, folds, mstop, nu,
                                 env=env, cache=cache, judge=judge)
    except MboostrsRuntimeError as exc:
        if "non-finite" not in str(exc) or _STEP.get() != "fixed":
            raise
        token = _STEP.set("line_search")
        try:
            return _out_of_fold_mean(data, response, formula, family, folds, mstop, nu,
                                     env=env, cache=None, judge=judge)
        finally:
            _STEP.reset(token)


def _out_of_fold_mean(data, response, formula, family, folds, mstop, nu,
                      env=None, cache=None, judge=None):
    """Out-of-fold response predictions, and the rows they belong to.

    `cache`, when the probes share one within an `autospec` call, remembers
    the fit for each fold so a later probe with the same specification
    reuses it instead of refitting. `judge`, when the fold design carries
    one (a rolling-origin scheme judges only the horizon block), names the
    rows a fold is judged on; without it every zero-weight row counts,
    which is the right judge set for k-fold and grouped splits.
    """

    folds = np.asarray(folds, dtype=float)
    n_rows, n_folds = folds.shape
    predicted = np.full(n_rows, np.nan)
    fits = 0
    for column in range(n_folds):
        in_bag = folds[:, column]
        out = in_bag == 0
        if judge is not None:
            out &= np.asarray(judge, dtype=bool)[:, column]
        held_out = np.flatnonzero(out)
        if held_out.size == 0:
            continue
        fit, hit = _shared_fold_fit(
            cache, data, formula, family, folds, column, mstop, nu, env)
        if not hit:
            fits += 1
        predicted[held_out] = np.asarray(
            fit.predict(type="response"), dtype=float)[held_out]
    return predicted, fits


def _shared_fold_fit(cache, data, formula, family, folds, column, mstop, nu,
                    env):
    """The fold-column fit for one specification, from the cache if a probe
    already made it. Returns `(model, was_cached)`."""
    if cache is not None:
        return cache.fold_fit(data, formula, family, folds, column, mstop,
                              nu, env)
    from mboostrs import gamboost

    fit = gamboost(formula, data=data, family=family, env=env,
                   control=_control(mstop, nu),
                   weights=np.asarray(folds, dtype=float)[:, column],
                   **_offset_kwargs())
    return fit, False


#: Budget ladder for the zero-inflation count scan (#202): a candidate that
#: leaves its risk curve falling at the cap has not been compared at all, so
#: the ladder has to reach well past a single 300-step rung. The top rung costs
#: about a minute of scanning on a 1,500-row linear specification; the
#: ladder only climbs while the ranking has not turned.
_ZERO_SCAN_BUDGETS = (300, 1200, 4800, 19200)


#: LSS families whose scan win means the zeros need their own component (#202).
_ZERO_CAPABLE_LSS = frozenset(
    {"ZIPoLSS", "ZINBLSS", "ZeroAdjustedPoissonLSS"})
#: LSS scan names that map back to the single-parameter NBinomial: the counts
#: are overdispersed but the zeros need no separate component, so the
#: glmboost path (and every downstream probe) keeps working unchanged.
_NB_LSS_NAMES = frozenset(
    {"NBinomialLSS", "NegativeBinomialLSS", "GeometricLSS"})


#: Smooth learners the family scan linearizes: glmboostLSS fits one linear
#: term per predictor, and the scan compares families, not shapes.
_LEARNER_RE = re.compile(r"\b(bbs|btree|bns)\(")


def _linearize_formula(formula):
    """Map smooth terms to ``bols`` for the family scan.

    ``glmboostLSS`` fits one linear term per predictor, so a ``bbs`` formula
    fails every candidate. The scan compares families, not shapes, so smooth
    terms become linear here; the recommended shape is untouched. Anything
    unrecognized is left verbatim — the scan records per-family errors rather
    than raising, and the caller falls back when the Poisson baseline itself
    cannot fit.
    """
    lhs, _, rhs = partition_tilde(formula)
    out = []
    i = 0
    while i < len(rhs):
        m = _LEARNER_RE.match(rhs, i)
        if not m:
            out.append(rhs[i])
            i += 1
            continue
        # Consume the argument list, splitting top-level commas so nested
        # calls (bbs(log(x), df=4)) keep their first argument intact.
        j = m.end()
        depth = 1
        args, cur = [], []
        while j < len(rhs) and depth > 0:
            c = rhs[j]
            if c == "(":
                depth += 1
                cur.append(c)
            elif c == ")":
                depth -= 1
                if depth > 0:
                    cur.append(c)
            elif c == "," and depth == 1:
                args.append("".join(cur))
                cur = []
            else:
                cur.append(c)
            j += 1
        args.append("".join(cur))
        first = args[0].strip() if args and args[0].strip() else "x"
        out.append(f"bols({quote_name(first)})")
        i = j
    return f"{lhs}~{''.join(out)}"




def _route_zero_inflated(data, response, formula, folds, observed, zero_share):
    """Run the count-family scan when exact zeros dominate the response (#202).

    Called from :func:`probe_family` when more than half the response is
    exactly zero. Runs ``scan_lss_families(..., support="counts")`` with
    cross-validated risk, escalating the step budget while the leading
    family or the ``PoissonLSS`` baseline still has its out-of-bag risk
    falling at the cap, and adopts a zero-capable winner only once both
    curves have an interior minimum: a comparison whose risk curves are
    still falling has not actually turned, whatever the provisional ranking
    says. When even the largest budget leaves a curve at the cap the probe
    declines: Poisson is reported as the dtype-inferred family, not as a
    measured verdict, and the sentence names the budget that ran out.

    Returns a :class:`FamilyEvidence`, or ``None`` when the scan agrees with
    Poisson (or could not run) so the caller falls back to the dispersion
    rule. Never raises: a failed scan must not crash autospec, and a native
    panic from the scan's LSS fits surfaces here as a `BaseException` that
    declines to the dtype-inferred Poisson rather than aborting (#319).
    """
    from mboostrs import NBinomial, Poisson, boost_control
    from mboostrs.model_selection import scan_lss_families

    n = int(observed.size)
    n_folds = int(np.asarray(folds).shape[1]) if folds is not None else 5

    def _ok(rows):
        return [r for r in rows
                if r.get("status") == "ok" and r.get("cvrisk") is not None]

    def _interior(row, budget):
        step = row.get("cvrisk_mstop")
        return step is not None and step < budget

    rows, budget, spent = None, _ZERO_SCAN_BUDGETS[0], 0
    for step_budget in _ZERO_SCAN_BUDGETS:
        try:
            scanned = scan_lss_families(
                _linearize_formula(formula), data, support="counts",
                metric="cvrisk", fit_func="glmboostLSS",
                control=boost_control(step_budget, 0.1), folds=folds)
        except BaseException as exc:
            # The scan fits LSS candidates, so a native panic (#319) surfaces
            # here as a `BaseException`; a failed scan declines to a Poisson
            # verdict rather than aborting autospec.
            return FamilyEvidence(
                recommended=Poisson(), dispersion=float("nan"),
                spread_tracks_mean=float("nan"), n_scored=n, fits=0,
                sentence=(
                    f"{zero_share:.0%} exact zeros: the zero-inflation count "
                    f"scan could not run ({exc}); Poisson stands as the "
                    "dtype-inferred family"),
                declined=False)
        ok = _ok(scanned)
        base = next((r for r in ok if r["family"] == "PoissonLSS"), None)
        if not ok or base is None:
            return None
        rows, budget = scanned, step_budget
        spent += len(ok) * (1 + n_folds)
        if _interior(ok[0], budget) and _interior(base, budget):
            # The ranking has turned: both curves the adoption decision
            # rests on have left the cap, so a larger budget cannot
            # reorder them.
            break
    ok = _ok(rows)
    base = next((r for r in ok if r["family"] == "PoissonLSS"), None)
    best = ok[0]  # scan_lss_families sorts best-first
    winner = best["family"]
    if winner == "PoissonLSS" or best["cvrisk"] >= base["cvrisk"]:
        # The scan agrees with the dtype guess; let the dispersion rule and
        # its standard sentence speak.
        return None
    if winner not in _NB_LSS_NAMES | _ZERO_CAPABLE_LSS:
        return None
    if not (_interior(best, budget) and _interior(base, budget)):
        # Still at the cap at the largest budget: no ranking among the
        # count families has turned. Report Poisson as the dtype-inferred
        # family rather than adopting a provisional winner, and say what
        # ran out (#202: the tuned ZIP fit needed ~3,200 updates per
        # parameter, past every fixed budget a probe may assume).
        return FamilyEvidence(
            recommended=Poisson(), dispersion=float("nan"),
            spread_tracks_mean=float("nan"), n_scored=n, fits=spent,
            sentence=(
                f"{zero_share:.0%} exact zeros: the count scan's "
                f"cross-validated risk was still falling at the {budget}-step "
                f"cap ({winner} first at {best['cvrisk']:.3f} vs PoissonLSS "
                f"{base['cvrisk']:.3f}, a provisional ranking), so the probe "
                "declines to choose: Poisson stands as the dtype-inferred "
                "family, and scan_lss_families with a larger boost_control "
                "is the way to settle it"),
            declined=True)
    # The zeros need their own component. autospec's downstream probes and
    # fit() are gamboost machinery, so the recommendation stays
    # single-parameter: NBinomial is materially better than Poisson on
    # zero-inflated counts, and the sentence names the scan's actual winner
    # with its margin for a caller who fits it directly with glmboostLSS.
    # (Adopting the LSS winner itself needs LSS-aware mstop/shape/interaction
    # probes — a separate enhancement.)
    return FamilyEvidence(
        recommended=NBinomial(), dispersion=float("nan"),
        spread_tracks_mean=float("nan"), n_scored=n, fits=spent,
        sentence=(
            f"{zero_share:.0%} exact zeros: the count scan (cvrisk, {budget} "
            f"steps) ranks {winner} first at {best['cvrisk']:.3f} vs "
            f"PoissonLSS {base['cvrisk']:.3f}, its held-out risk minimum "
            f"interior at {best['cvrisk_mstop']} steps, so the zeros get "
            "their own component; recommending NBinomial, the best "
            "single-parameter family — fit "
            f"{winner} directly with glmboostLSS for the scan winner"),
        declined=False)


def probe_family(data, response, formula, inferred, *, folds, mstop, nu,
                 env=None, validate=False, cache=None, judge=None,
                 fold_scheme=None):
    """Measure whether the response's spread matches the assumed family.

    Returns `None` when the response shape leaves nothing to choose between
    (a 0/1 response has one candidate), so the report can say the choice was
    not measured rather than implying a comparison happened.

    A measurement within sampling noise of the decision line does not
    choose: the probe declines, keeps the dtype-inferred family, and states
    both sides of the line, because the same data could have produced the
    other verdict. The standard errors are the ones the estimators carry at
    that sample size -- sqrt(2/n) for a Pearson dispersion under the Poisson
    it tests, 1/sqrt(n) for the correlation under independence -- times
    `_FAMILY_DECISION_SES`.

    A count response whose held-out dispersion is clearly below
    `_UNDERDISPERSION` varies less than a Poisson mean allows (a percentage
    or a rating stored as whole numbers), and Gaussian is recommended.

    When the spread of a positive response grows with its level, GammaReg is
    the family its variance calls for, but boosting it can still predict
    worse. Its gradient is multiplied by the shape parameter, which is
    recalibrated at every step and grows as the fit improves, so a fixed step
    length overshoots and the held-out Gamma risk stops being a guide to where
    to stop. With `validate`, GammaReg and Gaussian are compared on held-out
    squared error, each at its own cross-validated step count, and GammaReg is
    replaced when its squared error is worse by 2% and by two paired
    standard errors, or by half again. A count family (Poisson or NBinomial)
    is checked the same way.
    """
    from mboostrs import GammaReg, Gaussian, NBinomial, Poisson
    from mboostrs._frame import read_column

    observed = _finite(np.asarray(read_column(data, response).values))
    if observed.size == 0:
        return None
    unique = np.unique(observed)
    is_binary = unique.size <= 2 and set(unique.tolist()) <= {0.0, 1.0}
    if is_binary:
        return None
    from mboostrs.autospec import _large_whole_numbers

    is_count = bool(np.all(observed >= 0) and np.all(observed == np.round(observed))
                    and not _large_whole_numbers(observed))
    is_positive = bool(np.all(observed > 0))
    if not (is_count or is_positive):
        if validate and np.all(observed >= 0) and np.mean(observed == 0) < 0.5:
            # A non-negative measurement with zeros (an occupation count
            # scaled to a rate, a spend) is neither a count nor positive. A
            # log link fits effects that multiply; quasi-Poisson has that link
            # and a variance proportional to the mean, and needs no integers.
            evidence = FamilyEvidence(
                recommended=Poisson(), dispersion=float("nan"),
                spread_tracks_mean=float("nan"), n_scored=int(observed.size), fits=0,
                sentence=(
                    f"non-negative with {np.mean(observed == 0):.0%} zeros and "
                    f"not whole numbers: a log link (quasi-Poisson) was tried "
                    f"against Gaussian"))
            return _validate_against_gaussian(
                Poisson(), data, response, formula, folds, nu, env, evidence,
                cache=cache, judge=judge, fold_scheme=fold_scheme,
                require_better=True)
        return None

    if is_positive and not is_count:
        from mboostrs.autospec import NARROW_POSITIVE_CV, _narrow_positive

        if np.all(observed <= 1):
            evidence = FamilyEvidence(
                recommended=Gaussian(), dispersion=float("nan"),
                spread_tracks_mean=float("nan"), n_scored=int(observed.size),
                fits=0,
                sentence=(
                    "every response lies in (0, 1). GammaReg's support "
                    "continues past 1, so the mean candidate is Gaussian"
                    + (" and BetaLSS, the proportion family, is scored "
                       "against it on held-out log score"
                       if validate else
                       ". BetaLSS is the proportion family and is not in "
                       "this search")))
            if validate:
                return _validate_lss(
                    "BetaLSS", data, response, formula, folds, mstop, nu,
                    env, evidence, mean_mstop=mstop, cache=cache,
                    judge=judge, fold_scheme=fold_scheme)
            return evidence
        if _narrow_positive(observed):
            cv = float(np.std(observed) / np.mean(observed))
            return FamilyEvidence(
                recommended=Gaussian(), dispersion=float("nan"),
                spread_tracks_mean=float("nan"), n_scored=int(observed.size), fits=0,
                sentence=(
                    f"the response's standard deviation is {cv:.1%} of its mean, "
                    f"below {NARROW_POSITIVE_CV:.0%}: a Gamma's spread would change "
                    f"by as little across the range of means, which the data cannot "
                    f"tell from constant variance, and boosting a Gamma that narrow "
                    f"diverges, so Gaussian"))
        # The spread is read against a Gaussian mean. A Gamma mean fit on a
        # response it does not suit can diverge, and residuals from a diverging
        # fit grow with its fitted value whatever the data's variance is.
        mean_family = Gaussian()
    else:
        mean_family = inferred

    predicted, fits = _out_of_fold_mean_or_search(
        data, response, formula, mean_family, folds, mstop, nu, env=env,
        cache=cache, judge=judge)
    scored = np.isfinite(predicted)
    if scored.sum() < 8:
        return None
    y = np.asarray(read_column(data, response).values, dtype=float)[scored]
    mu = predicted[scored]
    residual = y - mu
    n_scored = int(scored.sum())

    declined = False
    if is_count:
        zero_share = float(np.mean(observed == 0.0))
        if zero_share > 0.5:
            # More exact zeros than a Poisson mean explains: run the count
            # scan and let a zero model compete for the recommendation,
            # rather than reporting Poisson from the dispersion rule (#202).
            routed = _route_zero_inflated(
                data, response, formula, folds, observed, zero_share)
            if routed is not None:
                return routed
        # A held-out mean near zero would make one row's term enormous (hours
        # worked, mostly zero, gave a dispersion of 1e11): floor it at a
        # thousandth of the mean response.
        safe = np.maximum(mu, max(1e-3 * float(np.mean(y)), 1e-8))
        dispersion = float(np.mean(residual ** 2 / safe))
        se = float(np.sqrt(2.0 / n_scored))
        declined = (abs(dispersion - _OVERDISPERSION)
                    < _FAMILY_DECISION_SES * se)
        under = dispersion + _FAMILY_DECISION_SES * se < _UNDERDISPERSION
        if under:
            recommended = Gaussian()
            sentence = (
                f"held-out Pearson dispersion {dispersion:.2f}, below the "
                f"{_UNDERDISPERSION:.2f} line by more than sampling noise "
                f"(standard error {se:.2f} on {n_scored} scored rows): the "
                "counts vary far less than a Poisson mean allows, as bounded "
                "scores and percentages do, so a constant-variance Gaussian")
        elif declined:
            recommended = inferred
            sentence = (
                f"held-out Pearson dispersion {dispersion:.2f} is within "
                f"sampling noise of the {_OVERDISPERSION:g} line that "
                f"separates Poisson from NBinomial (standard error {se:.2f} "
                f"on {n_scored} scored rows), so the probe declines to "
                f"choose: {type(inferred).__name__} stands as the "
                "dtype-inferred family, and `cvrisk` on both is the way to "
                "separate them")
        else:
            recommended = NBinomial() if dispersion > _OVERDISPERSION else Poisson()
            sentence = (
                f"held-out Pearson dispersion {dispersion:.2f}: "
                + ("the counts are more spread than a Poisson mean allows, so "
                   "NBinomial carries the extra variance"
                   if dispersion > _OVERDISPERSION else
                   "consistent with Poisson, whose variance equals its mean"))
        tracks = float("nan")
    else:
        variance = float(np.var(residual)) or 1.0
        dispersion = float(np.mean(residual ** 2) / variance)
        absolute = np.abs(residual)
        if absolute.std() > 0 and mu.std() > 0:
            tracks = float(np.corrcoef(absolute, mu)[0, 1])
        else:
            tracks = 0.0
        se = float(1.0 / np.sqrt(n_scored))
        declined = abs(tracks - _SPREAD_TRACKS) < _FAMILY_DECISION_SES * se
        # Choose, rather than defer to the dtype guess: a positive response
        # is inferred as GammaReg, so "keep what was inferred" would leave a
        # constant-variance response on a family that models a constant
        # coefficient of variation.
        grows = tracks > _SPREAD_TRACKS
        if declined:
            recommended = inferred
            sentence = (
                f"spread against level: correlation {tracks:+.2f} between the "
                f"absolute residual and the fitted mean is within sampling "
                f"noise of the ±{_SPREAD_TRACKS:g} line that separates a "
                f"constant-variance Gaussian from a spread-that-grows Gamma "
                f"(standard error {se:.2f} on {n_scored} scored rows), so the "
                f"probe declines to choose: {type(inferred).__name__} stands "
                "as the dtype-inferred family, and `cvrisk` on both is the "
                "way to separate them")
        elif grows:
            recommended = GammaReg()
            sentence = (
                f"spread against level: correlation {tracks:+.2f} between the "
                f"absolute residual and the fitted mean — "
                "the spread grows with the level, which is what a Gamma "
                "models and a Gaussian does not")
        else:
            recommended = Gaussian()
            sentence = (
                f"spread against level: correlation {tracks:+.2f} between the "
                f"absolute residual and the fitted mean — "
                "the spread does not grow with the level, so a constant-"
                "variance Gaussian fits it and Gamma's growing spread does not")

    evidence = FamilyEvidence(
        recommended=recommended, dispersion=dispersion,
        spread_tracks_mean=tracks, n_scored=n_scored,
        fits=fits, sentence=sentence, declined=declined)
    if validate and not is_count and type(recommended).__name__ == "GammaReg":
        evidence = _validate_gamma(data, response, formula, folds, nu, env,
                                   evidence, cache=cache, judge=judge,
                                   fold_scheme=fold_scheme)
        # The spread grows with the level; whether the shape itself moves
        # with the covariates is what a second predictor would add (#228).
        # Asked against whichever mean family survived the squared-error
        # check: a boosted GammaReg whose fixed step overshoots and loses
        # to Gaussian is the pathology GammaLSS's stabilized gradient
        # exists for, so the distributional question is still live.
        evidence = _validate_lss(
            "GammaLSS", data, response, formula, folds, mstop, nu, env,
            evidence, mean_mstop=mstop, cache=cache, judge=judge,
            fold_scheme=fold_scheme)
    elif validate and is_count and type(recommended).__name__ in ("Poisson", "NBinomial"):
        evidence = _validate_against_gaussian(
            recommended, data, response, formula, folds, nu, env, evidence,
            cache=cache, judge=judge, fold_scheme=fold_scheme)
    return evidence


#: A non-Gaussian candidate is replaced by Gaussian when its held-out squared
#: error is worse by this share and by two paired standard errors ...
_GAMMA_MSE_MARGIN = 0.02
#: ... or by this ratio, whatever the standard error.
_RUNAWAY_MSE_RATIO = 1.5


def _tuned_fold_mse(data, response, formula, family, folds, nu, env, *,
                    is_candidate=False, cache=None, judge=None, fold_scheme=None):
    """Per-fold held-out squared error of a mean family at its own
    cross-validated step count.

    Returns `(per-fold errors, step count, fits, status)`; `status` is
    `None` on success, `"invalid"` when the tuning rejected the inputs, and
    `"diverged"` when a candidate's risk went non-finite under every step
    rule.

    A candidate whose fixed-step fit diverges is refitted with the line
    search before it is judged: a log link on a response with a wide range
    overflows in the first fixed steps, which says nothing about how well
    the family predicts.
    """
    from mboostrs._errors import MboostrsRuntimeError
    from mboostrs._frame import read_column
    from mboostrs.explore import tune_mstop

    y = np.asarray(read_column(data, response).values, dtype=float)
    steps = ((_STEP.get(), "line_search")
             if is_candidate and _STEP.get() == "fixed" else (_STEP.get(),))
    tuning, step_used, fits = None, steps[0], 0
    for step_used in steps:
        try:
            with warnings.catch_warnings():
                # A search cut short here is weighed by this probe itself.
                warnings.filterwarnings("ignore", message=_SEARCH_ENDED_FALLING)
                tuning = tune_mstop(
                    formula, data, family,
                    folds=fold_scheme if fold_scheme is not None else folds,
                    start=200, nu=nu, extensions=1, env=env, step=step_used)
            break
        except MboostrsRuntimeError as exc:
            if not is_candidate or "non-finite" not in str(exc):
                raise
            tuning = None
        except ValueError:
            return np.asarray([]), 0, fits, "invalid"
    if tuning is None:
        return np.asarray([]), 0, fits, "diverged"
    fits += tuning.fits
    token = _STEP.set(step_used)
    try:
        predicted, used = _out_of_fold_mean(
            data, response, formula, family, folds,
            max(1, tuning.mstop), nu, env=env,
            cache=cache if step_used == steps[0] else None, judge=judge)
    finally:
        _STEP.reset(token)
    fits += used
    return (_per_fold_mse(y, predicted, folds), int(max(1, tuning.mstop)),
            fits, None)


def _per_fold_mse(y, predicted, folds):
    """Mean squared error of `predicted` on each fold's held-out rows (the
    rows with a finite prediction), one value per fold that has any."""
    per_fold = []
    for column in range(folds.shape[1]):
        ok = (folds[:, column] == 0) & np.isfinite(predicted)
        if ok.any():
            per_fold.append(float(np.mean((y[ok] - predicted[ok]) ** 2)))
    return np.asarray(per_fold)


def _mse_worse(candidate_mse, base_mse, se):
    """True when `candidate_mse` is worse than `base_mse` by
    `_GAMMA_MSE_MARGIN` and two paired standard errors, or by
    `_RUNAWAY_MSE_RATIO`.

    A fit that runs away in one fold makes the standard error large enough
    to excuse anything (energy_efficiency: 26 against 2.66, standard error
    22), while a genuine Poisson count can trail Gaussian by 6% at one
    standard error from noise alone.
    """
    excess = candidate_mse - base_mse
    return bool((excess > _GAMMA_MSE_MARGIN * base_mse
                 and (not np.isfinite(se) or excess > 2.0 * se))
                or candidate_mse > _RUNAWAY_MSE_RATIO * base_mse)


def _validate_gamma(data, response, formula, folds, nu, env, evidence,
                    cache=None, judge=None, fold_scheme=None):
    """Keep GammaReg only if it predicts the response about as well as Gaussian."""
    from mboostrs import GammaReg

    return _validate_against_gaussian(
        GammaReg(), data, response, formula, folds, nu, env, evidence,
        cache=cache, judge=judge, fold_scheme=fold_scheme)


def _validate_against_gaussian(candidate, data, response, formula, folds, nu, env,
                               evidence, cache=None, judge=None, fold_scheme=None,
                               require_better=False):
    """Replace `candidate` (GammaReg, Poisson or NBinomial) by Gaussian when
    its held-out squared error, each at its own cross-validated step count,
    is worse by `_GAMMA_MSE_MARGIN` and two paired standard errors, or by
    `_RUNAWAY_MSE_RATIO`.

    The spread measurements pick the family whose variance matches the data;
    this asks whether the boosted fit of that family predicts the mean as
    well. A count family on a whole-number measurement with a mass at zero
    (hours worked per week) and a Gamma whose recalibrated shape makes the
    fixed step overshoot both fit the variance and predict the mean worse.
    """
    from dataclasses import replace

    from mboostrs import Gaussian
    from mboostrs.explore import _paired_se, _partitions

    label = type(candidate).__name__
    folds = np.asarray(folds, dtype=float)
    fits = evidence.fits
    fold_mse, tuned = {}, {}
    for name, family in (("candidate", candidate), ("gaussian", Gaussian())):
        per_fold, tuned_mstop, used, status = _tuned_fold_mse(
            data, response, formula, family, folds, nu, env,
            is_candidate=name == "candidate", cache=cache, judge=judge,
            fold_scheme=fold_scheme)
        fits += used
        if status == "invalid":
            return evidence
        if status == "diverged":
            # A fit that diverges under every step rule has lost the comparison.
            return replace(
                evidence, recommended=Gaussian(), fits=fits,
                sentence=evidence.sentence + f"; but boosted, {label} diverged "
                "(the risk became non-finite along the path), so Gaussian")
        fold_mse[name] = per_fold
        tuned[name] = tuned_mstop
    candidate_mse = float(np.mean(fold_mse["candidate"]))
    gaussian_mse = float(np.mean(fold_mse["gaussian"]))
    se = _paired_se(fold_mse["candidate"], fold_mse["gaussian"], _partitions(folds))
    excess = candidate_mse - gaussian_mse
    worse = _mse_worse(candidate_mse, gaussian_mse, se)
    if require_better:
        # A family the response's shape does not call for (a log link for a
        # non-negative measurement with zeros) must earn its place: it is
        # kept only when it predicts better by the same margin.
        better = (-excess > _GAMMA_MSE_MARGIN * gaussian_mse
                  and np.isfinite(se) and -excess > 2.0 * se)
        if not better:
            return replace(
                evidence, recommended=Gaussian(), fits=fits,
                candidate_mse=candidate_mse, gaussian_mse=gaussian_mse,
                mse_difference_se=se, gamma_mse=float("nan"),
                sentence=(
                    f"{evidence.sentence}; boosted, {label} does not predict "
                    f"clearly better (held-out squared error {candidate_mse:.3g} "
                    f"against {gaussian_mse:.3g}, standard error of the "
                    f"difference {se:.2g}), so Gaussian"))
        return replace(
            evidence, fits=fits, candidate_mse=candidate_mse,
            gaussian_mse=gaussian_mse, mse_difference_se=se, gamma_mse=float("nan"),
            sentence=(
                f"{evidence.sentence}; boosted, {label} predicts better: held-out "
                f"squared error {candidate_mse:.3g} against {gaussian_mse:.3g} "
                f"for Gaussian (standard error of the difference {se:.2g})"))
    if worse:
        why = (". GammaReg's gradient is scaled by its shape parameter, which "
               "grows as the fit improves, and a fixed step length overshoots"
               if label == "GammaReg" else "")
        sentence = (
            evidence.sentence + f"; but boosted, {label} predicts worse: held-out "
            f"squared error {candidate_mse:.3g} against {gaussian_mse:.3g} for "
            f"Gaussian (standard error of the difference {se:.2g}), each at its "
            f"own cross-validated step count, so Gaussian{why}")
        recommended = Gaussian()
    else:
        sentence = (
            evidence.sentence + f"; boosted, {label} predicts as well as Gaussian "
            f"within sampling noise (held-out squared error {candidate_mse:.3g} "
            f"against {gaussian_mse:.3g}, standard error of the difference "
            f"{se:.2g})")
        recommended = evidence.recommended
    gamma = candidate_mse if label == "GammaReg" else float("nan")
    kept = "candidate" if type(recommended).__name__ == label else "gaussian"
    return replace(evidence, recommended=recommended, sentence=sentence, fits=fits,
                   candidate_mse=candidate_mse, gamma_mse=gamma,
                   gaussian_mse=gaussian_mse, mse_difference_se=se,
                   mean_mstop=tuned[kept])


# ── the distributional candidates the support points at (#228) ────────────

#: An LSS family replaces the mean family only when its held-out log score
#: is better than the mean family's by this share and by two paired
#: standard errors — the same margins `_validate_against_gaussian` uses on
#: squared error, on the one scale both families share: nats per row.
_LSS_LOGSCORE_MARGIN = _GAMMA_MSE_MARGIN


def _gaussian_fold_nll(fit, y, held, weights):
    """Held-out Gaussian log score at a sigma the fold's own rows estimate."""
    mu = np.asarray(fit.predict(type="response"), dtype=float)
    in_bag = weights > 0
    sigma2 = float(np.average((y[in_bag] - mu[in_bag]) ** 2,
                              weights=weights[in_bag])) or 1.0
    rows = held & np.isfinite(mu)
    return float(np.mean(
        0.5 * np.log(2.0 * np.pi * sigma2)
        + (y[rows] - mu[rows]) ** 2 / (2.0 * sigma2)))


def _family_fold_logscores(data, response, formula, family, folds, mstop, nu,
                           env=None, cache=None, judge=None):
    """Per-fold held-out log score, in nats per row, of a mean family.

    A Gaussian is scored at a sigma each fold estimates on its own training
    rows; a family with a nuisance parameter (GammaReg's shape) at the fold
    fit's own nuisance value, the same rule `family_held_out_score` applies.
    """
    from mboostrs._frame import read_column
    from mboostrs.explore import _fitted_family_score

    folds = np.asarray(folds, dtype=float)
    y = np.asarray(read_column(data, response).values, dtype=float)
    scores, folds_used, fits = [], [], 0
    for column in range(folds.shape[1]):
        weights = folds[:, column]
        held = weights == 0
        if judge is not None:
            held &= np.asarray(judge, dtype=bool)[:, column]
        if not held.any():
            continue
        with warnings.catch_warnings():
            # A fold fit whose step length is too large for the loss's
            # curvature warns about its own path; this probe weighs the
            # scores, not the paths.
            warnings.simplefilter("ignore")
            fit, hit = _shared_fold_fit(
                cache, data, formula, family, folds, column, mstop, nu, env)
        if not hit:
            fits += 1
        if type(family).__name__ == "Gaussian":
            score = _gaussian_fold_nll(fit, y, held, weights)
        else:
            mu = np.asarray(fit.predict(type="response"), dtype=float)
            rows = held & np.isfinite(mu)
            score = _fitted_family_score(
                fit, family, y[rows], mu[rows], np.ones(int(rows.sum())))
        scores.append(score)
        folds_used.append(column)
    return np.asarray(scores, dtype=float), folds_used, fits


class _LSSMu:
    """A distributional family with every parameter but the first fixed.

    The shape, pair and residual-tree probes vary the formula of one
    predictor and score each variant by held-out risk. For a distributional
    family that predictor is `mu`; this object carries the family and the
    formula of the other parameter, which the probes leave as the family
    probe chose it, so a probe passes a `mu` formula and the scorers here
    cross-validate the family's own held-out loss (`_lss_fold_logscores`).
    `optimal` records, for each `mu` formula scored, the per-parameter step
    counts at its optimum and the step length it was scored at.
    """

    def __init__(self, family, other_formula):
        self.family = family
        self.other_formula = other_formula
        self.param_names = tuple(family.param_names)
        self.optimal = {}

    def formula(self, mu_formula):
        """The multi-parameter formula with `mu_formula` as the first entry."""
        return {self.param_names[0]: mu_formula,
                self.param_names[1]: self.other_formula}


def _lss_mu_scan(view, data, response, mu_formula, folds, largest, nu, env,
                 fold_scheme):
    """`(risk, fits, per-fold risks, at_cap)` of a `mu` formula under the
    distributional family's held-out loss (nats per row for the log score the
    family probe reports), the shape `_path_held_out_scan` returns."""
    n_folds = int(np.asarray(folds).shape[1])
    out = _lss_fold_logscores(
        data, response, view.formula(mu_formula), view.family, folds,
        fold_scheme, largest, nu, env)
    if out is None:
        return float("nan"), n_folds, (), False
    mean, per_fold, _, optimal, fits, at_boundary, used = out
    view.optimal[(mu_formula, float(nu))] = (tuple(int(m) for m in optimal), int(used))
    return float(mean), int(fits), tuple(float(v) for v in per_fold), bool(at_boundary)


#: Gradient stabilization every distributional candidate is fitted with.
#: The cyclic LSS fit updates the location and the shape in turn; the
#: unstabilized gradient of the shape grows with the fitted spread, and at
#: the step length autospec uses (nu = 0.3) the path becomes unstable after
#: about ten cycles on a positive response with a wide range.
_LSS_STABILIZATION = "L2"
#: Points on the diagonal step-count grid. The cyclic engine reads every
#: point off one fitted path per fold, so a longer grid costs no fits.
_LSS_GRID_POINTS = 10
#: Times the largest step count is doubled while the optimum sits at the top
#: of the grid. Each doubling refits one path per fold.
_LSS_EXTENSIONS = 3


def _lss_candidate(lss_name):
    """The distributional family `lss_name`, with `_LSS_STABILIZATION`."""
    from importlib import import_module

    return getattr(import_module("mboostrs"), lss_name)(
        stabilization=_LSS_STABILIZATION)


def _lss_fold_logscores(data, response, formula, lss_family, folds,
                        fold_scheme, mstop, nu, env):
    """Per-fold held-out log score of a distributional family, nats/row.

    The cyclic engine reads a whole diagonal grid off one fitted path per
    fold, so this costs one fit per fold whatever the grid length. When the
    optimum is the largest step count, that count is doubled (at most
    `_LSS_EXTENSIONS` times, and not past `_MSTOP_EXTENSION_CAP`) and the
    grid is read again, so the reported optimum is an interior minimum
    unless the cap is reached. Returns `(mean score, per-fold scores, fold
    columns, per-parameter mstop at the optimum, fits, at_boundary,
    largest)` or `None` when the family cannot be fitted on these folds at
    all. `at_boundary` is True when the score was still improving at the
    largest step count tried, `largest`.
    """
    from mboostrs import boost_control
    from mboostrs.gamlss import cvrisk_gamboostLSS, diagonal_grid

    kept, largest = None, max(1, int(mstop))
    fits = 0
    for _ in range(_LSS_EXTENSIONS + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = cvrisk_gamboostLSS(
                    formula, data, families=lss_family,
                    control=boost_control(largest, nu),
                    folds=(fold_scheme if fold_scheme is not None
                           else np.asarray(folds, dtype=float)),
                    grid=diagonal_grid(largest, len(lss_family.param_names),
                                       length_out=_LSS_GRID_POINTS),
                    env=env,
                )
        except BaseException:
            # #319: a failed candidate must never abort the whole search. The
            # native layer reports some failures as a PyO3 `PanicException`,
            # which derives from `BaseException`, not `Exception` (e.g.
            # `digamma(0)` on the GammaLSS path); catching only `Exception`
            # let those escape and killed autospec instead of declining the
            # family.
            result = None
        risk = (np.asarray(result.risk_matrix, dtype=float)
                if result is not None else np.empty(0))
        if risk.size == 0 or not np.all(np.isfinite(risk)):
            # A longer path that breaks down leaves the shorter, finite one
            # as the answer, flagged as not having reached its minimum.
            if kept is not None:
                kept = (*kept[:5], True, kept[6])
            return kept
        best = int(np.argmin(np.nanmean(risk, axis=0)))
        n_folds = int(risk.shape[0])
        at_edge = best == risk.shape[1] - 1
        fits += n_folds
        kept = (float(np.nanmean(risk[:, best])), risk[:, best],
                list(range(n_folds)), list(result.optimal_mstop), fits,
                at_edge, largest)
        if not at_edge or 2 * largest > _MSTOP_EXTENSION_CAP:
            break
        largest *= 2
    return kept


def _lss_fold_mean_mse(data, response, formula, lss_family, folds, mstop,
                       nu, env, judge=None):
    """Per-fold held-out squared error of a distributional family's mean
    (mu on the response scale), each fold fitted at `mstop` steps per
    parameter. Returns `(errors, fits)`, or `None` when a fit fails."""
    from mboostrs import boost_control, gamboostLSS
    from mboostrs._frame import read_column

    folds = np.asarray(folds, dtype=float)
    y = np.asarray(read_column(data, response).values, dtype=float)
    predicted = np.full(len(y), np.nan)
    fits = 0
    for column in range(folds.shape[1]):
        weights = folds[:, column]
        held = weights == 0
        if judge is not None:
            held &= np.asarray(judge, dtype=bool)[:, column]
        if not held.any():
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = gamboostLSS(
                    formula, data=data, families=lss_family,
                    control=boost_control(int(mstop), nu), weights=weights,
                    env=env)
                mu = np.asarray(
                    fit.predict(parameter="mu", type="response"), dtype=float)
        except BaseException:
            return None
        fits += 1
        predicted[held] = mu[held]
    return _per_fold_mse(y, predicted, folds), fits


def _validate_lss(lss_name, data, response, formula, folds, mstop, nu, env,
                  evidence, mean_mstop, cache=None, judge=None,
                  fold_scheme=None, lss_formula=None):
    """Offer the distributional family the response's support points at, and
    adopt it only when it passes two held-out checks against the mean family.

    The log-score check: its held-out log score is better than the mean
    family's by `_LSS_LOGSCORE_MARGIN` and two paired standard errors. The
    mean check: its held-out squared error of the mean (mu on the response
    scale) is no worse than the mean family's by `_GAMMA_MSE_MARGIN` and two
    paired standard errors (the rule `_validate_against_gaussian` applies).
    Both checks are adoption gates, so a paired standard error that cannot
    be computed from the folds given -- one fold, or a non-finite fold
    difference -- fails the check it belongs to: an uncomputable proof is a
    refusal, never a waiver.
    The mean family is tuned by cross-validation and both checks score it at
    that step count; `mean_mstop` is the fallback when it cannot be tuned.
    The distributional candidate is fitted with `_LSS_STABILIZATION` on a
    step-count grid that is extended while its optimum sits at the top.

    `lss_formula` is the multi-parameter formula to score; it defaults to the
    mean's single `formula` (BetaLSS and GammaLSS fit both parameters from the
    same right-hand side). The heteroscedasticity probe passes a mapping that
    gives `sigma` its own covariate terms (#322).

    The support says a response inside (0, 1) is beta-shaped and a positive
    response whose spread grows is gamma-shaped; what this comparison
    measures is whether modelling the second parameter buys anything, and
    whether it does so without giving up accuracy of the mean. A
    constant-shape truth leaves the mean family within the margin, and the
    mean family stands: its one nuisance parameter is estimated while
    boosting, and a second predictor that says nothing new should not be
    adopted (#228). A log score can be won by the spread model while the
    mean is fitted worse than the mean family fits it; the mean check
    refuses that. Without `validate` the probe never reaches here, and
    the support sentence alone names the family it did not search.
    """
    from dataclasses import replace

    from mboostrs.explore import _paired_se, _partitions

    lss_family = _lss_candidate(lss_name)
    mean_family = evidence.recommended
    mean_label = type(mean_family).__name__
    folds = np.asarray(folds, dtype=float)
    lss = _lss_fold_logscores(
        data, response,
        lss_formula if lss_formula is not None else formula,
        lss_family, folds, fold_scheme, mstop, nu, env)
    if lss is None:
        return replace(
            evidence, sentence=(
                evidence.sentence + f"; boosted, {lss_name} could not be "
                f"fitted on these folds, so {mean_label} stands"))
    lss_mean, lss_scores, lss_folds, optimal, lss_fits, at_boundary, largest = lss
    stab = lss_family.stabilization
    boundary = (f"; its held-out score was still improving at {largest} steps, "
                "the largest tried" if at_boundary else "")

    # The mean family at its own cross-validated step count, for both checks.
    from mboostrs import Gaussian
    from mboostrs._frame import read_column

    y = np.asarray(read_column(data, response).values, dtype=float)
    fits = evidence.fits + lss_fits
    tuned_mstop = (evidence.mean_mstop
                   if evidence.mean_mstop and isinstance(mean_family, Gaussian)
                   else 0)
    if tuned_mstop:
        predicted, used = _out_of_fold_mean(
            data, response, formula, mean_family, folds, tuned_mstop, nu,
            env=env, cache=cache, judge=judge)
        mean_mse, fits = _per_fold_mse(y, predicted, folds), fits + used
    else:
        mean_mse, tuned_mstop, used, status = _tuned_fold_mse(
            data, response, formula, mean_family, folds, nu, env,
            is_candidate=not isinstance(mean_family, Gaussian), cache=cache,
            judge=judge, fold_scheme=fold_scheme)
        fits += used
        if status is not None:
            mean_mse, tuned_mstop = None, max(1, int(mean_mstop))
    mean_scores, mean_folds, mean_fits = _family_fold_logscores(
        data, response, formula, mean_family, folds, tuned_mstop, nu,
        env=env, cache=cache, judge=judge)
    fits += mean_fits
    paired = [(score, column) for score, column in zip(lss_scores, lss_folds,
                                                      strict=True)
              if column in mean_folds and np.isfinite(score)]
    mean_by_fold = dict(zip(mean_folds, mean_scores, strict=True))
    firsts = np.asarray([score for score, _ in paired], dtype=float)
    seconds = np.asarray([mean_by_fold[column] for _, column in paired])
    se = (_paired_se(firsts, seconds, _partitions(folds))
          if paired else float("nan"))
    improvement = float(np.mean(seconds) - lss_mean) if paired else 0.0
    # Adoption needs the paired proof itself: a standard error that cannot
    # be computed from these folds (one fold, or a non-finite fold
    # difference) is a refusal, never a waiver -- the rule
    # `_validate_against_gaussian`'s `require_better` path already applies.
    better = bool(
        paired
        and improvement > _LSS_LOGSCORE_MARGIN * abs(float(np.mean(seconds)))
        and np.isfinite(se) and improvement > 2.0 * se)
    mean_score = float(np.mean(seconds)) if paired else float("nan")
    score_text = (
        f"{lss_name}'s held-out log score is {lss_mean:.3f} nats/row against "
        f"{mean_score:.3f} for {mean_label}"
        + (f" (standard error of the difference {se:.2g})"
           if np.isfinite(se) else ""))
    common = dict(
        fits=fits, lss_score=lss_mean, mean_log_score=mean_score,
        score_difference_se=se, mean_mstop=tuned_mstop,
        lss_stabilization=stab, lss_at_boundary=at_boundary,
        lss_largest=largest)
    if not better:
        return replace(
            evidence, **common,
            sentence=(
                evidence.sentence + f"; boosted with {stab} stabilization, "
                + score_text + boundary
                + f", not clearly better, so {mean_label}"))

    # The mean check: a better log score does not say the mean is fitted as
    # well as the mean family fits it.
    lss_mse = None
    if mean_mse is not None:
        lss_mse = _lss_fold_mean_mse(
            data, response,
            lss_formula if lss_formula is not None else formula,
            lss_family, folds, max(optimal), nu, env, judge=judge)
    if lss_mse is None or mean_mse is None or lss_mse[0].size != mean_mse.size:
        return replace(
            evidence, **common, lss_mean_check="failed",
            sentence=(
                evidence.sentence + f"; boosted with {stab} stabilization, "
                + score_text + boundary + f"; but the squared error of "
                f"{lss_name}'s mean could not be compared with {mean_label}'s, "
                f"so {mean_label}"))
    lss_mse_folds, mse_fits = lss_mse
    fits += mse_fits
    common["fits"] = fits
    mse_se = _paired_se(lss_mse_folds, mean_mse, _partitions(folds))
    lss_error, mean_error = float(np.mean(lss_mse_folds)), float(np.mean(mean_mse))
    mse_text = (
        f"the squared error of its mean is {lss_error:.3g} against "
        f"{mean_error:.3g} for {mean_label} at its tuned {tuned_mstop} steps"
        + (f" (standard error of the difference {mse_se:.2g})"
           if np.isfinite(mse_se) else ""))
    common.update(lss_mse=lss_error, mean_family_mse=mean_error,
                  lss_mse_se=mse_se)
    if not np.isfinite(mse_se):
        # The mean check is an adoption gate, so it follows the adoption
        # rule, not the veto rule: with no computable standard error it has
        # no proof and cannot pass. (`_mse_worse`'s own non-finite clause
        # serves its veto callers, where waiving can only refuse a
        # candidate, never adopt one.)
        return replace(
            evidence, **common, lss_mean_check="failed",
            sentence=(
                evidence.sentence + f"; boosted with {stab} stabilization, "
                + score_text + boundary + f"; but {mse_text}, and the "
                f"standard error of that difference cannot be computed "
                f"from these folds, so the mean check has no proof and "
                f"{mean_label} stands"))
    if _mse_worse(lss_error, mean_error, mse_se):
        return replace(
            evidence, **common, lss_mean_check="failed",
            sentence=(
                evidence.sentence + f"; boosted with {stab} stabilization, "
                + score_text + boundary + f"; but {mse_text}, a worse mean "
                f"by more than {_GAMMA_MSE_MARGIN:.0%} and two standard "
                f"errors, so {mean_label}"))
    return replace(
        evidence, recommended=lss_family, **common, lss_mean_check="passed",
        mean_family=mean_family,
        lss_mstop=tuple(int(m) for m in optimal), lss_formula=lss_formula,
        sentence=(
            evidence.sentence + f"; boosted with {stab} stabilization, "
            + score_text + boundary + f"; {mse_text}, not worse by more "
            f"than {_GAMMA_MSE_MARGIN:.0%} and two standard errors, "
            f"so {lss_name} carries both parameters"))


def compare_finished_routes(data, response, *, mean_family, mean_formula,
                            mean_mstop, mean_nu, lss_family, lss_formula,
                            lss_nu, lss_largest, folds, fold_scheme, env,
                            cache=None, judge=None):
    """Score a finished mean route and a finished distributional route on the
    same folds and decide between them.

    The mean route is `mean_family` fitted with `mean_formula` for `mean_mstop`
    steps at step length `mean_nu`: the model after its shape, pair and
    residual-tree probes. The distributional route is `lss_family` fitted with
    the multi-parameter `lss_formula` (its `mu` formula after the same
    probes), its step counts chosen by cross-validation up to `lss_largest`.
    Both are scored on the held-out log score (the mean family at the sigma
    or nuisance value each fold fit estimates on its own training rows) and on
    the held-out squared error of the mean, over the same `folds`. The
    distributional model is chosen only when its log score is better by
    `_LSS_LOGSCORE_MARGIN` and two paired standard errors and its squared
    error is not worse by the rule `_mse_worse` applies. Both checks are
    adoption gates, so a paired standard error that cannot be computed
    from these folds -- one fold, or a non-finite fold difference --
    fails the check it belongs to: an uncomputable proof is a refusal,
    never a waiver.

    Returns a dict with both routes' scores and the verdict, or `None` when
    either route cannot be scored on these folds. `fits` counts the model fits
    the comparison made.
    """
    from mboostrs._frame import read_column
    from mboostrs.explore import _paired_se, _partitions

    folds = np.asarray(folds, dtype=float)
    y = np.asarray(read_column(data, response).values, dtype=float)
    try:
        lss = _lss_fold_logscores(data, response, lss_formula, lss_family,
                                  folds, fold_scheme, lss_largest, lss_nu, env)
        if lss is None:
            return None
        lss_mean, lss_scores, lss_folds, optimal, fits, at_boundary, largest = lss
        lss_mse = _lss_fold_mean_mse(data, response, lss_formula, lss_family,
                                     folds, max(optimal), lss_nu, env,
                                     judge=judge)
        if lss_mse is None:
            return None
        fits += lss_mse[1]
        predicted, used = _out_of_fold_mean(
            data, response, mean_formula, mean_family, folds, mean_mstop,
            mean_nu, env=env, cache=cache, judge=judge)
        fits += used
        mean_mse = _per_fold_mse(y, predicted, folds)
        mean_scores, mean_folds, used = _family_fold_logscores(
            data, response, mean_formula, mean_family, folds, mean_mstop,
            mean_nu, env=env, cache=cache, judge=judge)
        fits += used
    except BaseException:
        return None
    lss_mse_folds = lss_mse[0]
    if lss_mse_folds.size != mean_mse.size or lss_mse_folds.size == 0:
        return None
    paired = [(score, column) for score, column in zip(lss_scores, lss_folds,
                                                      strict=True)
              if column in mean_folds and np.isfinite(score)]
    if not paired:
        return None
    by_fold = dict(zip(mean_folds, mean_scores, strict=True))
    firsts = np.asarray([score for score, _ in paired], dtype=float)
    seconds = np.asarray([by_fold[column] for _, column in paired])
    partitions = _partitions(folds)
    score_se = _paired_se(firsts, seconds, partitions)
    mean_score = float(np.mean(seconds))
    improvement = mean_score - float(np.mean(firsts))
    # Both checks are adoption gates: a standard error that cannot be
    # computed from these folds (one fold, or a non-finite fold
    # difference) fails the check it belongs to -- a refusal, never a
    # waiver, as in `_validate_lss`.
    score_ok = bool(
        improvement > _LSS_LOGSCORE_MARGIN * abs(mean_score)
        and np.isfinite(score_se) and improvement > 2.0 * score_se)
    mse_se = _paired_se(lss_mse_folds, mean_mse, partitions)
    lss_error, mean_error = float(np.mean(lss_mse_folds)), float(np.mean(mean_mse))
    mean_ok = bool(np.isfinite(mse_se)
                   and not _mse_worse(lss_error, mean_error, mse_se))
    return {
        "lss_score": float(np.mean(firsts)), "mean_score": mean_score,
        "score_se": float(score_se), "lss_mse": lss_error,
        "mean_mse": mean_error, "mse_se": float(mse_se),
        "lss_mstop": tuple(int(m) for m in optimal),
        "lss_at_boundary": bool(at_boundary), "lss_largest": int(largest),
        "log_score_check": "passed" if score_ok else "failed",
        "mean_check": "passed" if mean_ok else "failed",
        "winner": "lss" if (score_ok and mean_ok) else "mean",
        "fits": int(fits)}


# ── Q1b: does the response's spread move with a covariate ─────────────────
#
# The family probe above asks which distribution the response comes from,
# assuming one spread for every row. A Gaussian whose variance changes with a
# covariate is still Gaussian -- but its scale is a function of that
# covariate, which is GaussianLSS. The classical test regresses the log of
# the squared residual on the covariate design: under a constant variance,
# log(resid^2) = log(sigma^2) + log(chi^2_1), whose noise has the known
# variance pi^2/2, so the OLS t-statistics need no estimated residual
# variance (Harvey, 1976; the identity `explore.heteroscedasticity` uses). A
# covariate whose |t| clears a Bonferroni bar is offered to GaussianLSS's
# sigma equation, and the held-out log score decides whether the second
# predictor earns its place (#322).
#
# The screen is deliberately stricter than `explore.heteroscedasticity`, which
# skips every column with two or fewer levels: the motivating CPS1988 case is
# a binary part-time indicator that the old screen never saw. Every modelled
# column enters -- a two-level column as one indicator, a category as its
# indicators -- and the decision is made by the held-out log score, not the
# screen alone.

#: The floor under the per-covariate |t| bar, whatever the family-wise bar.
_HET_T_MIN = 3.0
#: The family-wise false-positive rate the per-covariate bar controls: with
#: `p` design columns the bar is the two-sided normal quantile at `alpha/p`.
_HET_ALPHA = 0.01


def _heteroscedasticity_bar(n_design_columns):
    """The two-sided |t| a log-residual^2 slope must clear to be offered."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1.0 - _HET_ALPHA / (2.0 * max(int(n_design_columns), 1)))
    return max(_HET_T_MIN, float(z))


def _heteroscedasticity_screen(residual, columns, scored):
    """Slopes of log(residual^2) on the modelled columns, with the t-test the
    known log-chi^2 variance allows.

    `residual` holds one value per `scored` row, and each `Column` is read at
    those same rows (a judged rolling-origin fold scores only its horizon
    block). A numeric, boolean or datetime column enters as one centred
    regressor, a categorical as centred indicators for all but its first
    level. Returns `(slopes, t_stats, owners, r_squared, n_used)` -- `owners`
    names the column each design column stands for -- or `None` when there is
    too little residual mass to regress.
    """
    from mboostrs import _rust

    design: list[np.ndarray] = []
    owners: list[str] = []
    for column in columns:
        values = np.asarray(column.values, dtype=float)[scored]
        if column.kind in ("numeric", "boolean", "datetime"):
            if not np.isfinite(values).all() or float(np.std(values)) == 0.0:
                continue
            design.append(values)
            owners.append(column.name)
        else:
            present = np.unique(values[np.isfinite(values)])
            if present.size < 2 or present.size > 12:
                continue
            for level in present[1:]:
                indicator = (values == level).astype(float)
                indicator[~np.isfinite(values)] = np.nan
                if np.std(indicator) == 0.0:
                    continue
                design.append(indicator)
                owners.append(column.name)
    if not design:
        return None
    result = _rust.log_variance_screen(
        np.asarray(residual, dtype=float), np.column_stack(design))
    if result is None:
        return None
    slopes, t_stats, r_squared, n_used = result
    return np.asarray(slopes), np.asarray(t_stats), owners, r_squared, n_used


def probe_heteroscedasticity(data, response, formula, column_names, terms, *,
                             folds, mstop, nu, env=None, validate=False,
                             cache=None, judge=None, fold_scheme=None):
    """Test whether the response's variance moves with a covariate, and -- when
    it does and the budget allows -- offer GaussianLSS a scale equation and
    score it held out (#322).

    The Gaussian mean is fitted out of fold (reusing the family probe's folds
    from `cache` when it already made them), log(residual^2) is regressed on
    the covariate design, and a covariate whose |t| clears the Bonferroni bar
    becomes a term in the scale equation. With `validate` the GaussianLSS
    carrying that equation is compared, on held-out log score, against the
    mean-only Gaussian by the same `_validate_lss` margin the support probes
    use. Returns the `FamilyEvidence` (whose `recommended` is GaussianLSS only
    when it won), or `None` when there is nothing to measure.
    """
    from dataclasses import replace

    from mboostrs import Gaussian
    from mboostrs._frame import read_column

    y = np.asarray(read_column(data, response).values, dtype=float)
    predicted, fits = _out_of_fold_mean_or_search(
        data, response, formula, Gaussian(), folds, mstop, nu, env=env,
        cache=cache, judge=judge)
    scored = np.isfinite(predicted) & np.isfinite(y)
    if int(scored.sum()) < 20:
        return None
    residual = y[scored] - predicted[scored]
    # Read the predictors from the same (row-subset) `data` the mean was
    # fitted on, so a value here is the residual's own row.
    columns = [read_column(data, name) for name in column_names]
    screen = _heteroscedasticity_screen(residual, columns, scored)
    if screen is None:
        return None
    _, t_stats, owners, r_squared, n_used = screen
    bar = _heteroscedasticity_bar(len(owners))
    # One term per column, however many design columns it contributed.
    significant = list(dict.fromkeys(
        name for name, t_stat in zip(owners, t_stats, strict=True)
        if abs(t_stat) >= bar))
    sentence = (
        f"spread against the covariates: log(residual^2) on {len(owners)} "
        f"design columns over {n_used} rows has R^2 {r_squared:.2f}; "
        + (f"{', '.join(significant)} "
           + ("clears" if len(significant) == 1 else "clear")
           + f" the |t| >= {bar:.1f} bar, so the spread moves with "
           + ("it" if len(significant) == 1 else "them")
           if significant else
           f"no covariate's |t| clears the {bar:.1f} bar, so the spread is "
           f"constant in them"))
    evidence = FamilyEvidence(
        recommended=Gaussian(), dispersion=float("nan"),
        spread_tracks_mean=float("nan"), n_scored=int(scored.sum()),
        fits=fits, sentence=sentence)
    if not significant:
        return evidence
    term_by_column = {column.name: term
                      for column, term in zip(columns, terms, strict=True)}
    scale_formula = (f"{quote_response(response)} ~ "
                     + " + ".join(term_by_column[name] for name in significant))
    lss_formula = {"mu": formula, "sigma": scale_formula}
    evidence = replace(evidence, lss_formula=lss_formula)
    if not validate:
        return evidence
    return _validate_lss(
        "GaussianLSS", data, response, formula, folds, mstop, nu, env,
        evidence, mean_mstop=mstop, cache=cache, judge=judge,
        fold_scheme=fold_scheme, lss_formula=lss_formula)


# ── Q2: does the learner shape fit the column ─────────────────────────────
#
# `bbs` for a continuous column is a default, not a measurement. A threshold
# effect is fitted badly by a smooth curve and well by a tree; a straight line
# is fitted by a spline at the cost of the wiggle it can afford. The probe
# refits the same specification with one term's shape changed and compares
# held-out risk. The family is fixed across the variants, so its own loss is
# comparable between them.


@dataclass(frozen=True)
class ShapeScore:
    """One shape for one column, scored on the shared folds."""

    column: str
    shape: str
    term: str
    risk: float
    #: The risk on each fold at the step count that minimizes `risk`, for
    #: paired comparisons between shapes.
    fold_risk: tuple = ()
    #: True when the lowest risk is at the last step of the path, so the
    #: candidate was still improving when the path stopped.
    at_cap: bool = False


@dataclass(frozen=True)
class ShapeEvidence:
    """What the folds said about one column's learner shape.

    `comparable` is False when at least one candidate's path was still
    falling at its last step. The shapes then were not compared at a common
    level of convergence, so `chosen` is the incumbent and `sentence` says
    which path stopped early.
    """

    column: str
    chosen: str
    incumbent: str
    scores: tuple[ShapeScore, ...]
    fits: int
    sentence: str
    comparable: bool = True


#: A shape replaces the incumbent when its mean held-out gain over the folds
#: exceeds this many paired fold standard errors, the rule the specification
#: search uses for every change.
_SHAPE_SE = 2.0

#: The paired rule is used for a column with at least this share of the
#: tuned fit's risk reduction.
_SIGNAL_SHARE = 0.01

#: Without per-fold risks, a shape must beat the incumbent by this fraction
#: of its risk to replace it.
_SHAPE_MARGIN = 0.02


def _paired_gain(incumbent, candidate):
    """`(mean, standard error)` of the incumbent's minus the candidate's
    per-fold risk, or None when either lacks per-fold risks."""
    a, b = np.asarray(incumbent.fold_risk, float), np.asarray(candidate.fold_risk, float)
    if a.size < 2 or a.size != b.size or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))


def _shape_variants(column, df: float, incumbent_term: str | None = None) -> dict:
    """The learner shapes worth comparing for this column, by name.

    A column fitted on a transformed scale (``bbs(log(x))`` or
    ``bbs(sqrt(x))``, the skewed-predictor rule) is compared on that scale:
    the spline and the straight line in ``log(x)``, and the tree, which a
    monotone transform does not change.
    """
    name = column.name
    if column.kind in ("categorical", "boolean"):
        return {}
    levels = getattr(column, "levels", None)
    if levels is not None and len(levels) <= 2:
        return {}
    for transform in ("log", "sqrt"):
        scaled = f"{transform}({quote_name(name)})"
        if incumbent_term == f"bbs({scaled}, df={df})":
            return {
                "bbs": incumbent_term,
                "bols": f"bols({scaled})",
                "btree": f"btree({quote_name(name)})",
            }
    return {
        "bbs": f"bbs({quote_name(name)}, df={df})",
        "bols": f"bols({quote_name(name)})",
        "btree": f"btree({quote_name(name)})",
    }


#: Families whose risk is a negative log-likelihood with no lower bound at
#: zero: the risk of a good fit can be negative, so a share "of the risk"
#: needs the saturated model's risk subtracted first.
_LOG_LINK_FAMILIES = ("poisson", "nbinomial", "gammasreg")


def _risk_floor(family, data, response) -> float:
    """Mean risk per row of the saturated model (fitted value equal to the
    observation), the floor any held-out risk sits above.

    A relative gain ``(a - b) / a`` presumes a risk bounded below by zero,
    as squared error and the binomial loss are (their saturated risk is 0).
    Poisson, negative binomial and Gamma risks are negative log-likelihoods
    that go below zero, and near zero the ratio inflates a noise-level
    difference into tens of percent. Measured from this floor, the risk is
    half the deviance, and the shares mean what they mean for squared error.
    The family's own risk (with its default nuisance parameter) is evaluated
    at ``f = log(y)``; the floor is 0 for every other family.
    """
    from mboostrs._frame import read_column
    from mboostrs._rust import family_diagnostics
    from mboostrs.families import _family_params, _family_rust_name

    name = _family_rust_name(family)
    if name not in _LOG_LINK_FAMILIES:
        return 0.0
    y = np.asarray(read_column(data, response).values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    f = np.log(np.maximum(y, 1e-10))
    _, _, risk = family_diagnostics(name, _family_params(family),
                                    np.ascontiguousarray(y), f, np.ones(y.size))
    return float(risk) / y.size


def _relative_gain(worse, better, floor=0.0):
    """`(worse - better)` as a share of `worse` measured from `floor`."""
    scale = worse - floor
    return (worse - better) / scale if scale > 0 else 0.0


def _held_out_risk(data, response, formula, family, folds, mstop, nu,
                   env=None, cache=None, judge=None):
    """`(mean risk per held-out row, fits spent)` for one specification.

    Uses the engine's risk rather than a hand-written loss, so the number the
    probe compares is the one the fit minimises. `cache` is the probes'
    shared fit cache within one `autospec` call; `judge` is the fold
    design's judge mask, when it carries one (see `_out_of_fold_mean`).
    """
    from mboostrs._frame import read_column
    from mboostrs._rust import family_diagnostics
    from mboostrs.explore import _fitted_nuisance
    from mboostrs.families import _family_params, _family_rust_name

    folds = np.asarray(folds, dtype=float)
    y = np.asarray(read_column(data, response).values, dtype=float)
    name = _family_rust_name(family)
    params = _family_params(family)

    total, scored, fits = 0.0, 0, 0
    for column in range(folds.shape[1]):
        in_bag = folds[:, column]
        out = in_bag == 0
        if judge is not None:
            out &= np.asarray(judge, dtype=bool)[:, column]
        held_out = np.flatnonzero(out)
        if held_out.size == 0:
            continue
        fit, hit = _shared_fold_fit(
            cache, data, formula, family, folds, column, mstop, nu, env)
        if not hit:
            fits += 1
        eta = np.asarray(fit.predict(type="link"), dtype=float)[held_out]
        # Score at the fold fit's nuisance estimate (NBinomial dispersion,
        # GammaReg shape), as the core cvrisk does; the constructor default
        # would bias a family comparison against the calibrated families.
        scalar, vector = (_fitted_nuisance(fit)
                          if hasattr(fit, "nuisance") else (None, None))
        _, _, risk = family_diagnostics(
            name, params, np.ascontiguousarray(y[held_out]),
            np.ascontiguousarray(eta), np.ones(held_out.size), scalar, vector)
        total += float(risk)
        scored += held_out.size
    if scored == 0:
        return float("nan"), fits
    return total / scored, fits


def _single_fold_design(folds, fold_scheme, family):
    """Whether the design is one supplied holdout with no judge set.

    On a single fold the across-fold standard error is zero by
    construction, so every paired route in the probes is inert and a
    bare margin on the aggregate curve is the only test a candidate
    faces. The probes then require the per-row paired evidence of
    `_fold_row_losses` as well. Schemes are excluded (they carry their
    own judge semantics), as are the distributional probes, whose
    held-out score is not the engine risk the row losses read.
    """
    if fold_scheme is not None or isinstance(family, _LSSMu):
        return False
    try:
        return int(np.asarray(folds).shape[1]) == 1
    except (IndexError, TypeError, ValueError):
        return False


def _fold_row_losses(data, response, formula, family, folds, largest, step,
                     nu, env=None):
    """`(per-row held-out losses at `step`, fits spent)`.

    Fits the fold's in-bag rows once, at `largest`, and reads the
    engine's per-row loss on the rows the fold holds out, at the fit's
    own nuisance value -- the row-level version of `_held_out_risk`.
    `(None, fits)` when the family's loss couples rows, so no per-row
    loss exists (CoxPH, AUC, Cindex, Gehan).
    """
    folds = np.asarray(folds, dtype=float)
    weights = folds[:, 0]
    held = np.flatnonzero(weights == 0)
    if held.size == 0:
        return None, 0
    from mboostrs import gamboost

    fit = gamboost(formula, data=data, family=family, env=env,
                   control=_control(int(largest), nu), weights=weights,
                   **_offset_kwargs())
    return _row_losses_of_fit(fit, data, response, family, held, step), 1


def _row_losses_of_fit(fit, data, response, family, held, step):
    """Per-row held-out losses of an existing fold fit, read at `step`.

    The row-level companion of `_held_out_risk`'s scoring: the engine's
    loss at the fit's own nuisance value. None when the family's loss
    couples rows (CoxPH, AUC, Cindex, Gehan).
    """
    from mboostrs._frame import read_column
    from mboostrs._rust import family_held_out_rows
    from mboostrs.explore import _fitted_nuisance
    from mboostrs.families import _family_params, _family_rust_name

    sliced = (fit.with_mstop(int(step))
              if int(step) < int(fit.mstop) else fit)
    eta = np.asarray(sliced.predict(type="link"), dtype=float)[held]
    scalar, vector = (_fitted_nuisance(sliced)
                      if hasattr(sliced, "nuisance") else (None, None))
    y = np.asarray(read_column(data, response).values, dtype=float)
    rows = family_held_out_rows(
        _family_rust_name(family), _family_params(family),
        np.ascontiguousarray(y[held]), np.ascontiguousarray(eta),
        scalar, vector)
    if rows is None:
        return None
    return np.asarray(rows, dtype=float)


def _paired_row_z(losses_without, losses_with):
    """`(z, mean difference, standard error)` of per-row loss differences.

    Positive means the `with` specification loses less per held-out row.
    `(None, nan, nan)` when either side has no per-row losses, so the
    caller falls back to the rule it would have used without them.
    """
    if losses_without is None or losses_with is None:
        return None, float("nan"), float("nan")
    diff = (np.asarray(losses_without, dtype=float)
            - np.asarray(losses_with, dtype=float))
    if diff.size < 3 or not np.isfinite(diff).all():
        return None, float("nan"), float("nan")
    mean = float(diff.mean())
    se = float(diff.std(ddof=1)) / float(np.sqrt(diff.size))
    if se <= 0.0:
        return (float("inf") if mean > 0 else 0.0), mean, se
    return mean / se, mean, se


def _held_out_pair_z(data, response, formula_without, step_without,
                     largest_without, nu_without, formula_with, step_with,
                     largest_with, nu_with, family, folds, env=None):
    """`(z, mean difference, standard error, fits spent)` for one adoption.

    The two specifications' fold fits are read at their own chosen steps
    (the steps the path comparison judged them at), and the difference
    is taken row by row on the held-out rows, so the standard error is
    the one the aggregate curve hides: two paths can post minima far
    apart while the row differences that separate them are noise. The
    z is `(None, ...)` when either family's loss couples rows.
    """
    # A failing fit propagates: scoring the pair as "no reading" would let the
    # adoption proceed without the row-level check. BudgetExhausted is not
    # caught here either; the search session that opened the budget stops
    # the search on it.
    losses_without, fits_without = _fold_row_losses(
        data, response, formula_without, family, folds, largest_without,
        step_without, nu_without, env)
    losses_with, fits_with = _fold_row_losses(
        data, response, formula_with, family, folds, largest_with,
        step_with, nu_with, env)
    z, mean, se = _paired_row_z(losses_without, losses_with)
    return z, mean, se, fits_without + fits_with


def _shape_row_z(data, response, terms, term_index, incumbent_term, best_term,
                 family, folds, mstop, nu, env, backfit):
    """`(per-row paired z, evaluations spent)` for one shape replacement
    on a single holdout, or `(None, evaluations)` when no per-row reading
    exists (a row-coupled loss, a failed path).

    The z is read from the comparison that produced the margin: with a
    fold backfit, each term's one-term path at its own chosen step, the
    held-out rows' losses taken row by row (`_FoldBackfit.risk_rows`);
    without one, the two full formulas at their path-chosen steps
    (`_held_out_pair_z`). An evaluation is one scored path, charged by
    the caller at the probe's per-variant rate.
    """
    if backfit is not None:
        from mboostrs.formula import term_variables

        _, _, losses_without = backfit.risk_rows(
            term_index, incumbent_term, term_variables(incumbent_term))
        _, _, losses_with = backfit.risk_rows(
            term_index, best_term, term_variables(best_term))
        z, _mean, _se = _paired_row_z(losses_without, losses_with)
        return z, 2
    base = f"{quote_response(response)} ~ " + " + ".join(terms)
    swapped = [best_term if i == term_index else term
               for i, term in enumerate(terms)]
    candidate = f"{quote_response(response)} ~ " + " + ".join(swapped)
    ev_without, _ = _path_held_out_evidence(
        data, response, base, family, folds, mstop, nu, env=env)
    ev_with, _ = _path_held_out_evidence(
        data, response, candidate, family, folds, mstop, nu, env=env)
    if ev_without is None or ev_with is None:
        return None, 2
    z, _mean, _se, pair_fits = _held_out_pair_z(
        data, response, base, ev_without.chosen, mstop, nu,
        candidate, ev_with.chosen, mstop, nu, family, folds, env=env)
    return z, 2 + pair_fits



def _path_held_out_risk(data, response, formula, family, folds, largest, nu,
                        env=None, fold_scheme=None):
    """`(lowest cross-validated risk along the path, fits spent, per-fold
    risks at that step)`.

    Each candidate is judged at its own best step count up to `largest`, read
    off one path per fold. Judging every candidate at one fixed, small step
    count compares them before any of them has fitted the signal, and favours
    whichever learner moves fastest in the first steps (a tree or a line)
    over the one that fits the data best.
    """
    risk, fits, fold_risk, _ = _path_held_out_scan(
        data, response, formula, family, folds, largest, nu, env=env,
        fold_scheme=fold_scheme)
    return risk, fits, fold_risk


def _path_held_out_scan(data, response, formula, family, folds, largest, nu,
                        env=None, fold_scheme=None):
    """`_path_held_out_risk` with one more field: whether the lowest risk sits
    at the last step of the path (`largest`), so the path was still falling
    when it stopped and its minimum was not reached.

    Returns `(risk, fits, per-fold risks, at_cap)`.
    """
    if isinstance(family, _LSSMu):
        return _lss_mu_scan(family, data, response, formula, folds, largest,
                            nu, env, fold_scheme)
    stopping, per_fold = _path_held_out_evidence(
        data, response, formula, family, folds, largest, nu, env=env,
        fold_scheme=fold_scheme)
    n_folds = int(np.asarray(folds).shape[1])
    if stopping is None:
        return float("nan"), n_folds, (), False
    return (stopping.risk[stopping.chosen], n_folds, per_fold,
            _still_falling(stopping.risk[1:]))


def _path_held_out_evidence(data, response, formula, family, folds, largest, nu,
                            env=None, fold_scheme=None):
    """The existing fold paths with their selected step and full-data replay.

    Keeping this evidence does not fit another model. The legacy risk/scan
    helpers retain their return shapes and fold-path budget accounting.
    """
    from mboostrs import gamboost
    from mboostrs.cvrisk import is_at_boundary

    fit = gamboost(formula, data=data, family=family, env=env,
                   control=_control(int(largest), nu), **_offset_kwargs())
    cv = fit.cvrisk(folds=fold_scheme if fold_scheme is not None
                    else np.asarray(folds, dtype=float), _inner=True)
    risk = tuple(float(v) for v in cv.aggregate)
    if not np.any(np.isfinite(risk[1:])):
        return None, ()
    chosen = int(cv.optimal_mstop)
    grid = tuple(int(v) for v in cv.grid)
    index = grid.index(chosen)
    per_fold = tuple(float(v) for v in np.asarray(cv.risk_matrix)[:, index])
    stopping = MstopEvidence(
        chosen=chosen, grid=grid, risk=risk,
        at_boundary=is_at_boundary(chosen, grid[-1]),
        fits=int(np.asarray(folds).shape[1]),
        sentence=f"mstop={chosen} — the adopted form's shared-fold path (cap {grid[-1]})",
        _fit=fit)
    return stopping, per_fold


def _still_falling(curve):
    """Whether a held-out risk curve over steps 1..n is still falling at its end.

    The minimum must lie in the boundary band that `mboostrs.cvrisk` uses for
    the step search, and the risk just before that band must exceed the
    minimum by more than the shape margin. The second test keeps a flat tail
    (a linear signal fitted in a few steps, whose risk moves in the fourth
    significant figure after that) from counting as unconverged.
    """
    from mboostrs.cvrisk import boundary_tolerance

    curve = np.asarray(curve, dtype=float)
    if curve.size < 3 or not np.any(np.isfinite(curve)):
        return False
    last = curve.size - 1
    best = int(np.nanargmin(curve))
    band = boundary_tolerance(last)
    if best < last - band or last - band - 1 < 0:
        return False
    before = curve[last - band - 1]
    return bool(np.isfinite(before)
                and before - curve[best] > _SHAPE_MARGIN * abs(curve[best]))


@dataclass(frozen=True)
class CoordinateScore:
    """One candidate learner set for a coordinate pair, scored on shared folds.

    ``risk`` is the lowest mean held-out risk along the candidate's path (its
    own best step count), and ``fold_risk`` the per-fold risks at that step,
    which the paired comparison reads. ``risk`` is NaN when the path failed.
    """

    name: str
    formula: str
    risk: float
    fold_risk: tuple[float, ...]
    stopping: MstopEvidence | None = field(default=None, repr=False, compare=False)


def score_coordinate_candidates(data, response, candidates, family, folds, largest, nu,
                                *, env=None, fold_scheme=None):
    """``({name: CoordinateScore}, fits)`` for each formula in ``candidates``.

    Every candidate runs the same cross-validated path (`_path_held_out_risk`)
    on the same folds, so the per-fold differences are paired. A candidate
    whose path raises is scored NaN and cannot win a comparison.
    """
    from mboostrs._errors import MboostrsRuntimeError

    scores, fits = {}, 0
    for name, formula in candidates.items():
        if isinstance(family, _LSSMu):
            risk, n_fits, fold_risk, _ = _lss_mu_scan(
                family, data, response, formula, folds, largest, nu, env,
                fold_scheme)
            fits += n_fits
            scores[name] = CoordinateScore(
                name, formula, float(risk), tuple(fold_risk), None)
            continue
        try:
            stopping, fold_risk = _path_held_out_evidence(
                data, response, formula, family, folds, largest, nu,
                env=env, fold_scheme=fold_scheme)
            risk = float("nan") if stopping is None else stopping.risk[stopping.chosen]
        except (MboostrsRuntimeError, ValueError):
            risk, stopping, fold_risk = float("nan"), None, ()
        n_fits = int(np.asarray(folds).shape[1])
        fits += n_fits
        scores[name] = CoordinateScore(name, formula, float(risk), tuple(fold_risk), stopping)
    return scores, fits


def coordinate_beats(current, challenger, floor=0.0, *, margin=True):
    """``(beats, gain, z)``: whether ``challenger`` has less held-out risk than ``current``.

    ``gain`` is the relative reduction of the mean held-out risk, measured from
    ``floor`` (`_relative_gain`), and ``z`` the mean paired fold difference over
    its standard error (`_paired_gain`). The challenger beats the current
    learner when its paired difference is positive and at least two standard
    errors, or, when ``margin`` is True, when the reduction is at least 2% of
    the held-out risk. This is the rule `probe_shapes` applies to every shape
    alternative.
    """
    if not (np.isfinite(current.risk) and np.isfinite(challenger.risk)):
        return False, 0.0, 0.0
    gain = _relative_gain(current.risk, challenger.risk, floor)
    paired = _paired_gain(current, challenger)
    if paired is None:
        return bool(margin and gain >= _SHAPE_MARGIN), float(gain), 0.0
    mean, se = paired
    z = mean / se if se > 0 else (float("inf") if mean > 0 else 0.0)
    consistent = mean > 0 and z >= _SHAPE_SE
    return bool(consistent or (margin and gain >= _SHAPE_MARGIN)), float(gain), float(z)


class _FoldBackfit:
    """Per-fold fits of the incumbent specification, for one-term comparisons.

    For fold k and column j, the other terms' contribution
    ``eta_k - f_jk`` is held fixed as an offset and only the candidate term
    for column j is boosted on the fold's in-bag rows; its held-out risk is
    read along the path and the candidate is scored at its lowest point.
    This is one backfitting step per candidate: it compares shapes for one
    column given the rest of the model at the fitted depth, and each
    candidate costs one-term fits instead of full-model fits.
    """

    def __init__(self, data, response, formula, family, folds, depth, mstop, nu,
                 env, cache=None):
        self.folds = np.asarray(folds, dtype=float)
        self.response = response
        self.family = family
        self.mstop = int(mstop)
        self.nu = nu
        self.env = env
        self.data = data
        self.fits = []
        #: Incumbent fold fits made here rather than taken from the cache.
        self.made = 0
        for column in range(self.folds.shape[1]):
            fit, hit = _shared_fold_fit(cache, data, formula, family, self.folds,
                                        column, int(depth), nu, env)
            self.made += 0 if hit else 1
            eta = np.asarray(fit.predict(type="link"), dtype=float).ravel()
            self.fits.append((fit, eta))

    def risk(self, term_index, term, columns):
        """`(lowest mean held-out risk, per-fold risks at that step)` of `term`
        in place of term `term_index`."""
        risk, fold_risk, _ = self.risk_scan(term_index, term, columns)
        return risk, fold_risk

    def risk_scan(self, term_index, term, columns):
        """`risk`, with a third field: whether the lowest mean risk sits at the
        last step of the one-term path, so that path had not reached its
        minimum."""
        from mboostrs import gamboost
        from mboostrs._frame import read_column
        from mboostrs.explore import cvrisk_at

        sub = {self.response: np.asarray(read_column(self.data, self.response).values)}
        for name in columns:
            sub[name] = np.asarray(read_column(self.data, name).values)
        # A term refitted alone on its partial offset needs about as many
        # steps as the full model spent on it, not the full model's depth:
        # three times the incumbent's own step count, at least 200.
        spent = max(int(np.sum(np.asarray(fit._selected) == term_index))
                    for fit, _ in self.fits)
        length = int(min(self.mstop, max(200, 3 * spent)))
        paths = []
        for column, (fit, eta) in enumerate(self.fits):
            learner = fit._learner_names[term_index]
            own = np.asarray(fit.predict(newdata=self.data, which=learner),
                             dtype=float).ravel()
            # Fitted at one step: only its cross-validation path is used, and
            # that runs to `length` steps.
            one = gamboost(f"{quote_response(self.response)} ~ {term}", data=sub, family=self.family,
                           env=self.env, offset=eta - own,
                           control=_control(1, self.nu))
            cv = cvrisk_at(one, self.folds[:, [column]], length)
            paths.append(np.asarray(cv.aggregate, dtype=float))
        stacked = np.vstack(paths)[:, 1:]
        mean = np.nanmean(stacked, axis=0)
        if not np.any(np.isfinite(mean)):
            return float("nan"), (), False
        best = int(np.nanargmin(mean))
        return (float(mean[best]), tuple(float(v) for v in stacked[:, best]),
                _still_falling(mean))

    def risk_rows(self, term_index, term, columns):
        """`risk_scan`'s chosen step with the per-row held-out losses.

        The one-term path of `risk_scan` on the (single) fold, the step
        its lowest risk sits at, and the engine's per-row losses on the
        held-out rows of a one-term fold fit refitted at that step --
        the row-level reading of the comparison `risk_scan` scores in
        aggregate. `(risk, step, losses)`; losses is None when the
        family's loss couples rows, so no per-row loss exists.
        """
        from mboostrs import gamboost
        from mboostrs._frame import read_column
        from mboostrs.explore import cvrisk_at

        fit, eta = self.fits[0]
        learner = fit._learner_names[term_index]
        own = np.asarray(fit.predict(newdata=self.data, which=learner),
                         dtype=float).ravel()
        sub = {self.response: np.asarray(read_column(self.data, self.response).values)}
        for name in columns:
            sub[name] = np.asarray(read_column(self.data, name).values)
        spent = max(int(np.sum(np.asarray(f._selected) == term_index))
                    for f, _ in self.fits)
        length = int(min(self.mstop, max(200, 3 * spent)))
        one = gamboost(f"{quote_response(self.response)} ~ {term}", data=sub,
                       family=self.family, env=self.env, offset=eta - own,
                       control=_control(1, self.nu))
        cv = cvrisk_at(one, self.folds[:, [0]], length)
        path = np.asarray(cv.aggregate, dtype=float)[1:]
        if not np.any(np.isfinite(path)):
            return float("nan"), 0, None
        step = int(np.nanargmin(path)) + 1
        weights = self.folds[:, 0]
        held = np.flatnonzero(weights == 0)
        refit = gamboost(f"{quote_response(self.response)} ~ {term}",
                         data=sub, family=self.family, env=self.env,
                         offset=eta - own, weights=weights,
                         control=_control(step, self.nu))
        losses = _row_losses_of_fit(refit, sub, self.response, self.family,
                                    held, step)
        return float(path[step - 1]), step, losses


def probe_shapes(data, response, columns, terms, family, *, folds, mstop, nu,
                 df, max_fits, fixed_names=(), env=None, cache=None,
                 judge=None, fold_scheme=None, depth=None, term_shares=None,
                 risk_floor=0.0, preexploration=None):
    """Score each continuous column's shape against its alternatives.

    Every candidate formula is cross-validated along a path of `mstop` steps
    and scored at its lowest held-out risk on that path, so each shape is
    compared at the step count that suits it.

    Runs columns in formula order and stops when `max_fits` is exhausted, so
    a caller's budget bounds the cost. Columns that were not reached are
    absent from the result rather than reported as having no alternative.

    ``fixed_names`` lists columns whose terms came from a declaration or a
    prior and sit outside the shape catalogue (``bmono``, ``buser``, ...).
    Those are scored as fixed incumbents: the alternatives are priced
    against them and the report states what the declared term costs, but no
    alternative ever replaces one. A non-catalogue term on a column *not* in
    ``fixed_names`` (``brandom`` on a numeric few-levels column, say) is a
    dtype default outside the comparison and is skipped, as before.

    On a single supplied holdout (one fold, no scheme, no judge set), the
    paired fold standard error is undefined, so a replacement would rest
    on the aggregate margin alone. There a replacement must also clear
    the paired per-row z of the comparison that produced the margin,
    read on the holdout's own rows at the steps that comparison judged
    (`_shape_row_z`) -- the same bar the fold route applies when folds
    exist. Families whose loss couples rows have no per-row losses and
    keep the margin rule.
    """
    folds = np.asarray(folds, dtype=float)
    per_variant = int(folds.shape[1])
    fixed_names = frozenset(fixed_names)
    evidence: list[ShapeEvidence] = []
    spent = 0
    # One-term backfitting comparisons need a scalar-per-row offset and a
    # plain fold matrix; a multi-parameter family or a rolling-origin judge
    # set falls back to cross-validating each full candidate formula.
    backfit = None
    one_term = (judge is None and type(family).__name__ not in ("Multinomial", "PropOdds")
                and not isinstance(family, _LSSMu) and max_fits >= per_variant)
    if one_term:
        base = f"{quote_response(response)} ~ " + " + ".join(terms)
        try:
            backfit = _FoldBackfit(data, response, base, family, folds,
                                   depth or mstop, mstop, nu, env, cache=cache)
            spent += backfit.made
        except BudgetExhausted:
            raise
        except Exception:
            backfit = None
    # A one-term fit costs about 1/len(terms) of a full fit.
    per_one_term = per_variant / max(len(terms), 1)
    # On a single supplied holdout the paired fold comparison below
    # cannot fire (one fold), so a replacement would rest on the
    # aggregate margin alone; such a replacement is audited against the
    # holdout's per-row differences before it is made (see the decision
    # below). A judge set scores a subset of the held-out rows the audit
    # does not read, so judged designs keep the margin rule.
    single = _single_fold_design(folds, fold_scheme, family) and judge is None

    proposals = ({row["column"]: row for row in preexploration["columns"]}
                 if preexploration is not None else {})
    candidates = list(zip(columns, terms, strict=True))
    if preexploration is not None:
        priority = {name: i for i, name in enumerate(preexploration["priority"])}
        candidates.sort(key=lambda pair: priority.get(pair[0].name, len(priority)))
    for column, incumbent_term in candidates:
        variants = _shape_variants(column, df, incumbent_term)
        if type(family).__name__ == "Multinomial":
            # The Multinomial fit wraps bols, bbs and brandom per class; a
            # tree has no such wrapping.
            variants.pop("btree", None)
        if len(variants) < 2:
            continue
        catalog_incumbent = next(
            (shape for shape, term in variants.items()
             if term == incumbent_term), None)
        if catalog_incumbent is None and column.name not in fixed_names:
            continue
        fixed = catalog_incumbent is None
        incumbent = catalog_incumbent if catalog_incumbent is not None \
            else incumbent_term
        proposal = proposals.get(column.name)
        if proposal is not None and column.name not in fixed_names:
            # Screening proposes alternatives. Always score the incumbent;
            # declared terms retain their complete constraint-cost comparison.
            allowed = set(proposal["shapes"]) | {incumbent}
            variants = {shape: term for shape, term in variants.items() if shape in allowed}
        n_scores = len(variants) + (1 if fixed else 0)
        cost = per_one_term if backfit is not None else per_variant
        if spent + cost * n_scores > max_fits:
            break
        term_index = terms.index(incumbent_term)

        scores: list[ShapeScore] = []
        column_fits = 0
        to_score = list(variants.items())
        if fixed:
            to_score.append((incumbent_term, incumbent_term))
        for shape, term in to_score:
            candidate_terms = [term if other is incumbent_term else other
                               for other in terms]
            formula = f"{quote_response(response)} ~ " + " + ".join(candidate_terms)
            risk, fold_risk, at_cap = None, (), False
            if backfit is not None:
                from mboostrs.formula import term_variables

                try:
                    risk, fold_risk, at_cap = backfit.risk_scan(
                        term_index, term, term_variables(term))
                    fits = cost
                except BudgetExhausted:
                    raise
                except Exception:
                    risk = None
            if risk is None:
                risk, fits, fold_risk, at_cap = _path_held_out_scan(
                    data, response, formula, family, folds, mstop, nu, env=env,
                    fold_scheme=fold_scheme)
            spent += fits
            column_fits += fits
            if np.isfinite(risk):
                scores.append(ShapeScore(column.name, shape, term, risk, fold_risk,
                                         bool(at_cap)))
        if not scores:
            continue

        still_falling = [score.shape for score in scores if score.at_cap]
        if still_falling:
            # A candidate whose held-out risk is lowest at its last step has not
            # reached a minimum, so the shapes are not at a common level of
            # convergence. A tree that fits the signal in a few steps then
            # looks better than a smooth that needs more steps. No replacement
            # is made on that comparison.
            names = " and ".join(still_falling)
            evidence.append(ShapeEvidence(
                column=column.name, chosen=incumbent, incumbent=incumbent,
                scores=tuple(sorted(scores, key=lambda score: score.risk)),
                fits=int(round(column_fits)),
                sentence=(f"{incumbent} stands; no shape comparison is made: "
                          f"{names} still improved at the end of its path (up to "
                          f"{int(mstop)} steps), so the held-out minimum was not "
                          f"reached and the shapes are not at a common level of "
                          f"convergence. Raise mstop or nu to compare them."),
                comparable=False))
            continue

        ranked = sorted(scores, key=lambda score: score.risk)
        best = ranked[0]
        current = next(s for s in scores if s.shape == incumbent)
        gain = _relative_gain(current.risk, best.risk, risk_floor)
        table = ", ".join(f"{s.shape} {s.risk:.4f}" for s in ranked)
        # The gain is judged against the fold-to-fold spread of the paired
        # differences: a fixed share of the total held-out risk hides a shape
        # that is clearly better when the noise variance dominates the risk
        # (a jump fitted by a tree instead of a smooth, at 5,000 rows, gains
        # about 1% of a risk that is mostly noise, in every fold).
        # A change passes when either measure says so: a gain of 2% of the
        # held-out risk, or a gain of two paired fold standard errors. The
        # second catches a small, consistent gain the first misses; the first
        # catches a large gain one noisy fold hides from the second.
        paired = _paired_gain(current, best)
        z = (paired[0] / paired[1] if paired is not None and paired[1] > 0
             else float("inf") if paired is not None and paired[0] > 0 else 0.0)
        # The paired rule applies to a column that carries signal (at least
        # the deselection threshold's share of the tuned fit's risk
        # reduction). Across tens of columns and two alternatives each, a
        # noise column passes two standard errors by chance, and a tree on it
        # then fits chance splits.
        carries_signal = (term_shares is None
                          or term_shares.get(incumbent_term, 0.0) >= _SIGNAL_SHARE)
        consistent = (carries_signal and paired is not None and paired[0] > 0
                      and z >= _SHAPE_SE)
        better = gain >= _SHAPE_MARGIN or consistent
        # On a single supplied holdout the paired fold route above is
        # inert (one fold), so `better` can rest on the aggregate margin
        # alone -- the bare-margin adoption the residual-tree step
        # guards against. A replacement then also needs the paired
        # per-row z of the comparison that produced the margin, read on
        # the holdout's own rows, to clear the same bar (`_shape_row_z`).
        # Families whose loss couples rows have no per-row losses; their
        # replacements keep the margin rule, as do declared terms (never
        # replaced), multi-fold designs and schemes.
        audit_z = None
        if (single and not fixed and best.shape != incumbent and better
                and not consistent):
            audit_z, audit_evals = _shape_row_z(
                data, response, terms, term_index, current.term,
                best.term, family, folds, mstop, nu, env, backfit)
            audit_spent = audit_evals * cost
            spent += audit_spent
            column_fits += audit_spent
            if audit_z is not None and audit_z < _SHAPE_SE:
                better = False
        evidence_text = (f"{gain:.1%} less held-out risk"
                         + (f", {z:.1f} paired fold standard errors" if paired is not None else ""))
        stands_text = (f"the best alternative gains less than {_SHAPE_MARGIN:.0%} of the "
                       f"held-out risk" + (f" and {z:.1f} < {_SHAPE_SE:g} paired fold standard "
                                           f"errors" if paired is not None else ""))
        if fixed:
            # A declared term is never replaced; its sentence reports what the
            # constraint costs, and a cost below 2% of the risk is not one.
            if best.shape == incumbent:
                chosen, sentence = incumbent, (
                    f"{incumbent} stands (declared): it also fits best of "
                    f"{table} — the constraint costs nothing measurable")
            elif gain < _SHAPE_MARGIN:
                chosen, sentence = incumbent, (
                    f"{incumbent} stands (declared): {table} — no alternative beats "
                    f"it by {_SHAPE_MARGIN:.0%} of the held-out risk")
            else:
                chosen, sentence = incumbent, (
                    f"{incumbent} stands (declared): {table} — the "
                    f"constraint costs {evidence_text} against {best.shape}")
        elif best.shape == incumbent or not better:
            if audit_z is not None:
                chosen, sentence = incumbent, (
                    f"{incumbent} stands: {table} — {best.shape} gains "
                    f"{gain:.1%} of the held-out risk, but the design is "
                    f"a single holdout, where that margin is the only "
                    f"aggregate evidence; the paired per-row difference "
                    f"is {audit_z:.1f} standard errors, under the "
                    f"{_SHAPE_SE:g} a replacement requires")
            else:
                chosen, sentence = incumbent, (
                    f"{incumbent} stands: {table} — {stands_text}")
        else:
            if audit_z is not None:
                chosen, sentence = best.shape, (
                    f"{best.shape} over {incumbent}: {table} — "
                    f"{evidence_text}; on the single holdout the paired "
                    f"per-row difference is {audit_z:.1f} standard errors")
            else:
                chosen, sentence = best.shape, (
                    f"{best.shape} over {incumbent}: {table} — {evidence_text}")
        evidence.append(ShapeEvidence(
            column=column.name, chosen=chosen, incumbent=incumbent,
            scores=tuple(ranked), fits=int(round(column_fits)), sentence=sentence))

    return evidence, int(np.ceil(spent))


# ── Q4: how deep does the boosting need to run ────────────────────────────
#
# `mstop=100` is a default carried from `boost_control`, not a measurement.
# The sweep cross-validates the chosen specification and reports where the
# held-out risk turns. An optimum at the end of the grid is not an optimum:
# it is the largest value tried, and the report says so.


@dataclass(frozen=True)
class MstopEvidence:
    """Where the cross-validated risk turned, and whether it turned at all."""

    chosen: int
    grid: tuple[int, ...]
    risk: tuple[float, ...]
    at_boundary: bool
    fits: int
    sentence: str
    #: The full-data fit of the last grid the sweep walked, at `largest`
    #: iterations -- the same specification, family and step length the
    #: recommendation settled on. The interaction screen truncates this path
    #: instead of fitting the main-effect model again. Carried privately: it
    #: is an optimisation input, not part of the evidence.
    _fit: object = field(default=None, repr=False, compare=False)


#: The mstop grid is never extended past this many iterations.
#: `tune_mstop`'s warning when its search ends with the risk still falling.
_SEARCH_ENDED_FALLING = "the cross-validated risk was still falling"
_MSTOP_EXTENSION_CAP = 10_000


def probe_mstop(data, response, formula, family, *, folds, largest, nu,
                env=None, extend=0, fold_scheme=None):
    """Cross-validate one path and read the optimum off the curve.

    One fit per fold covers the whole grid: the cross-validated risk is
    recorded at every iteration, so sweeping candidate values costs nothing
    beyond the single path each fold already walks. When the minimum is the
    last iteration, the grid is doubled and swept again, at most `extend`
    times; `tune_mstop` does the work, and a caller outside `autospec` can use
    it directly. The sweep's full-data fit of the last grid travels on the
    evidence (as `_fit`) so the interaction screen can reuse its path.
    ``fold_scheme`` is the `FoldScheme` object the folds came from, when they
    came from one, so the sweep is judged on the scheme's judge set rather
    than on every zero-weight row.
    """
    from mboostrs.explore import tune_mstop

    try:
        with warnings.catch_warnings():
            # `MstopEvidence.at_boundary` carries this to the report instead.
            warnings.filterwarnings("ignore", message=_SEARCH_ENDED_FALLING)
            tuning = tune_mstop(
                formula, data, family,
                folds=fold_scheme if fold_scheme is not None
                else np.asarray(folds, dtype=float),
                start=largest, nu=nu, extensions=extend,
                max_mstop=_MSTOP_EXTENSION_CAP, env=env, step=_STEP.get(),
                **_offset_kwargs())
    except ValueError:
        return None
    return MstopEvidence(
        chosen=tuning.mstop, grid=tuple(range(len(tuning.risk))), risk=tuning.risk,
        at_boundary=tuning.at_boundary, fits=tuning.fits, sentence=tuning.sentence,
        _fit=getattr(tuning, "_fit", None))


#: Step lengths the bounded step-length search probes, in addition to the
#: route's own. The grid brackets the default 0.3 on both sides and reaches
#: the top of the range the step length is used at in practice; five extra
#: values keep the search's cost at five fold paths (see `probe_nu_search`).
_NU_SEARCH_GRID = (0.1, 0.2, 0.5, 0.7, 1.0)

#: A probed step length replaces the route's own only when its risk is
#: lower by more than this margin, scaled by the incumbent's risk. Along a
#: flat risk curve the step count compensates for the step length almost
#: exactly, so the grid's readings differ in the seventh decimal place;
#: differences that small are the same path read at another step count,
#: not a better step length.
_NU_SEARCH_MIN_GAIN = 1e-4


def probe_nu_search(data, response, formula, family, *, folds, incumbent, nu,
                    env=None, fold_scheme=None, max_fits=None):
    """Probe a bounded grid of step lengths for one specification.

    `incumbent` is the specification's existing `MstopEvidence` at step
    length `nu`: its path is already paid for, so it enters the comparison
    at no cost. Every other grid value costs one `probe_mstop` sweep --
    one path per fold, the unit the fit budget already counts -- read at
    the incumbent sweep's grid end, with no grid extensions, so each
    candidate costs exactly the fold count of fits. Candidates are probed
    in grid order and skipped once `max_fits` (when given) cannot pay for
    another sweep.

    Returns `(best_evidence, best_nu, fits_spent, records)`, where
    `records` holds one dict per step length considered -- the incumbent
    first, with `fits` 0 -- giving each candidate's chosen step count and
    the cross-validated risk there. The best candidate is the lowest risk
    at its chosen step count, and it replaces the incumbent only when it
    clears `_NU_SEARCH_MIN_GAIN` against the incumbent's risk; otherwise
    the incumbent is returned, so the search changes the step length only
    on a measured gain, not on a flat-curve tie.
    """
    from mboostrs._errors import MboostrsRuntimeError

    n_folds = int(np.asarray(folds).shape[1])
    largest = len(incumbent.risk) - 1
    records = [{"nu": float(nu), "mstop": int(incumbent.chosen),
                "risk": float(incumbent.risk[incumbent.chosen]),
                "fits": 0, "incumbent": True}]
    incumbent_risk = records[0]["risk"]
    margin = _NU_SEARCH_MIN_GAIN * max(1.0, abs(incumbent_risk))
    best, best_nu = incumbent, float(nu)
    best_risk = incumbent_risk
    spent = 0
    for candidate in _NU_SEARCH_GRID:
        if candidate == float(nu):
            continue
        if max_fits is not None and spent + n_folds > max_fits:
            break
        try:
            row = probe_mstop(data, response, formula, family, folds=folds,
                              largest=largest, nu=candidate, env=env, extend=0,
                              fold_scheme=fold_scheme)
        except MboostrsRuntimeError:
            # A long fixed step can push the linear predictor far enough
            # that the inverse link overflows (Poisson at nu = 1.0 does).
            # That step length cannot be fitted, so it is not a candidate;
            # the sweep it aborted is charged as a full sweep of folds.
            spent += n_folds
            continue
        if row is None:
            continue
        spent += row.fits
        risk = float(row.risk[row.chosen])
        records.append({"nu": float(candidate), "mstop": int(row.chosen),
                        "risk": risk, "fits": int(row.fits),
                        "incumbent": False})
        if risk < best_risk:
            best, best_nu, best_risk = row, float(candidate), risk
    if best_nu != float(nu) and best_risk >= incumbent_risk - margin:
        return incumbent, float(nu), spent, tuple(records)
    return best, best_nu, spent, tuple(records)


# ── Q3: is there structure the additive terms miss ────────────────────────
#
# The screen reads the main-effect model's residual: what the additive terms
# could not explain. A pair that removes residual variance is a lead, not a
# finding -- it is measured on the training residual, not held out -- so the
# probe reports the pair and the term that would carry it, and leaves the
# specification additive. Adding a term on this evidence alone would put
# something in the model that nothing out of fold has justified.


@dataclass(frozen=True)
class InteractionLead:
    """A pair the residual points at, and the term that would carry it."""

    left: str
    right: str
    removed: float
    term: str
    kind: str
    #: Rank correlation between the margins, for the duplicate-axis kind.
    correlation: float = float("nan")
    #: Sources joined by '+': residual, tree/surrogate, and/or neural.
    source: str = "residual"
    #: Friedman's H^2 of the pair in the surrogate, when it proposed it.
    h2: float = float("nan")

    def __str__(self) -> str:
        if self.source != "residual":
            return (f"{self.left} x {self.right}: screening variance {self.removed:.3g} "
                    f"({self.source}; H^2 {self.h2:.2f}) — {self.term}")
        if self.kind == "duplicate-axis":
            return (f"{self.left} x {self.right}: removes {self.removed:.3f} "
                    f"of the residual, but the margins record one axis twice "
                    f"(rank correlation {self.correlation:.2f}) — a term on "
                    f"the pair re-enters what the main effects already "
                    f"carry, so it is reported, not proposed")
        return (f"{self.left} x {self.right}: removes {self.removed:.3f} of "
                f"the residual — {self.term}")


#: A pair that removes less than this share of the residual's variance
#: leaves the residual essentially where it found it.
_INTERACTION_FLOOR = 0.01

#: Rank correlation at or above this says the two margins are one axis
#: measured twice -- a calendar year against a date, a baseline age against
#: the current age. The lead is then reported as a duplicate rather than
#: proposed as structure: no pair term can add what each margin already
#: carries, and the screen's job is leads, not arithmetic.
_DUPLICATE_AXIS_RANK = 0.95


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of `values`, tied values taking the mean of their positions."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start
        while (stop + 1 < len(values)
               and sorted_values[stop + 1] == sorted_values[start]):
            stop += 1
        ranks[start:stop + 1] = np.arange(start + 1, stop + 2).mean()
        start = stop + 1
    out = np.empty(len(values), dtype=float)
    out[order] = ranks
    return out


def _pair_rank_correlation(columns, left: str, right: str):
    """Spearman rank correlation of two columns, or `None` when either is
    not an ordered quantity (a categorical's codes carry no rank worth
    reading) or a column is constant."""
    by_name = {column.name: column for column in columns}
    first, second = by_name.get(left), by_name.get(right)
    if (first is None or second is None
            or first.kind not in ("numeric", "datetime", "boolean")
            or second.kind not in ("numeric", "datetime", "boolean")):
        return None
    a = np.asarray(first.values, dtype=float)
    b = np.asarray(second.values, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return None
    ra = _average_ranks(a[keep])
    rb = _average_ranks(b[keep])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def probe_interactions(data, response, formula, family, columns, terms, *,
                       mstop, nu, top=3, env=None, main_fit=None, surrogate=None, seed=0,
                       exact_fit=False, counts=None):
    """Pairs whose joint structure the additive terms miss, best first.

    A column is treated as a group when the recommendation fits it with
    `brandom`, not when its dtype looks categorical: an integer-coded group
    is a numeric column, and pairing it with `%X%` would grid its labels as
    if they were measurements.

    A pair whose margins are one axis measured twice -- rank correlation at
    or above `_DUPLICATE_AXIS_RANK` -- is reported with kind
    ``"duplicate-axis"`` and no proposed term, after the actionable leads:
    the pair removes residual, but a term on it re-enters what the main
    effects already carry.

    ``main_fit`` is the `mstop` sweep's full-data fit of the same
    specification, family and step length, when it ran; the screen reads the
    residual of that path truncated to ``max(mstop, 50)`` iterations instead
    of fitting the main-effect model again. Returns
    ``(leads, fits, reused)``, where ``reused`` is 1 when the path was
    reused and 0 when the screen fitted its own. When a dict is passed as
    ``counts``, it is filled with the candidates each screen generated:
    ``residual_pairs`` (every pair of the columns, ranked on the residual) and
    ``surrogate_pairs`` (the pairs the boosted-tree surrogate proposed).
    """
    from mboostrs import gamboost, screen_interactions
    from mboostrs._frame import read_column

    steps = max(int(mstop), 50)
    if exact_fit and main_fit is not None:
        # A policy stage screens the fitted incumbent at its own selected
        # stopping iteration, including when that iteration is below 50.
        # No deeper, different model is fitted merely to obtain residuals.
        fit = main_fit
        screen_fits, reused = 0, 1
    elif main_fit is not None and getattr(main_fit, "mstop", 0) >= steps:
        fit = main_fit[:steps]
        screen_fits, reused = 0, 1
    else:
        fit = gamboost(formula, data=data, family=family, env=env,
                       control=_control(steps, nu), **_offset_kwargs())
        screen_fits, reused = 1, 0
    observed = np.asarray(read_column(data, response).values, dtype=float)
    # On the response scale. `fitted_` is the additive predictor, which for a
    # non-identity link is not comparable with the observations: an additive
    # logit model leaves 0/1-minus-logit residuals that every pair of signal
    # columns appears to explain, and the screen reported false leads.
    #
    # For a non-Gaussian family the raw response-scale residual still has a
    # variance that depends on the fitted mean: a Binomial `y - p` has
    # variance p(1-p), largest at p = 0.5, so rows near the middle of the
    # probability scale dominate the screen's MSE and the ranking measures
    # variance rather than structure. The Pearson residual divides that out
    # and puts every row on a common variance scale (#315). The change is in
    # what `probe_interactions` hands `screen_interactions`, not in
    # `screen_interactions` itself.
    on_response = np.asarray(fit.predict(type="response"), dtype=float)
    if on_response.ndim > 1 and on_response.shape[1] != 1:
        # A multi-class fit has no single response-scale prediction per row.
        on_response = np.asarray(fit.fitted_, dtype=float)
    from mboostrs.families import Gaussian, GaussReg
    from mboostrs.terms import pearson_residual
    if isinstance(family, (Gaussian, GaussReg)):
        residual = observed - on_response.ravel()
    else:
        residual = pearson_residual(fit, data, response)

    # Only the columns the recommendation actually models. A column hygiene
    # dropped -- a constant, say -- carries no information, and a pair naming
    # it is an artefact of screening the frame rather than the model.
    names = [column.name for column in columns]
    pairs = [(left, right)
             for index, left in enumerate(names)
             for right in names[index + 1:]]
    if counts is not None:
        counts["residual_pairs"] = len(pairs)
        counts["surrogate_pairs"] = 0
    if not pairs:
        return (), 0, 0
    from mboostrs.explore import pair_term_for

    groups = {column.name for column, term in zip(columns, terms, strict=True)
              if term.startswith(("brandom(", "bmrf("))}
    groups.update(column.name for column in columns
                  if column.kind in ("categorical", "boolean"))
    levels = {column.name: column.n_levels for column in columns}
    categorical = {column.name for column in columns if column.kind == "categorical"}
    # A column fitted on the log or square-root scale keeps that scale in its
    # pair terms.
    transforms = {column.name: kind for column, term in zip(columns, terms, strict=True)
                  for kind in ("log", "sqrt") if term.startswith(f"bbs({kind}(")}
    try:
        screened = screen_interactions(
            data, residual, response=response, pairs=pairs,
            groups=[name for name in groups if levels.get(name, 0) > 2])
    except Exception:
        return (), 0, 0

    leads: list[InteractionLead] = []
    duplicates: list[InteractionLead] = []
    # `removed` is a mean squared error in the response's units, so the floor
    # is a share of the residual's variance: an absolute floor never passes
    # on a response measured in small units (grid stability, variance 1e-3).
    finite = np.asarray(residual, dtype=float)
    finite = finite[np.isfinite(finite)]
    floor = _INTERACTION_FLOOR * (float(np.var(finite)) if finite.size else 1.0)
    for pair in screened:
        if pair.removed <= floor:
            continue
        rho = _pair_rank_correlation(columns, pair.left, pair.right)
        if rho is not None and abs(rho) >= _DUPLICATE_AXIS_RANK:
            duplicates.append(InteractionLead(
                left=pair.left, right=pair.right,
                removed=float(pair.removed), term="",
                kind="duplicate-axis", correlation=rho))
            continue
        # One rule for every pair term, shared with suggest_specifications: a
        # two-valued column switches a smooth on, a group gets a random slope,
        # two continuous columns get a tensor surface.
        term = pair_term_for(pair.left, pair.right, levels=levels, groups=groups,
                             categorical=categorical, transforms=transforms,
                             data=data)
        if "%X%" in term and term.startswith("bbs("):
            kind = "tensor"
        elif "%X%" in term:
            kind = "factor-by-factor"
        else:
            kind = "varying"
        leads.append(InteractionLead(
            left=pair.left, right=pair.right, removed=float(pair.removed),
            term=term, kind=kind))
    leads.sort(key=lambda lead: lead.removed, reverse=True)
    leads = leads[:top]
    # A boosted-tree surrogate proposes pairs the residual screen can miss:
    # its partial dependence carries interactions whatever the main effects
    # look like. Its pairs join the leads; adoption still needs held-out risk.
    known = {frozenset((lead.left, lead.right)) for lead in leads + duplicates}
    source_map = {}
    default_source = "surrogate"
    if surrogate is None or surrogate is True:
        try:
            surrogate = _surrogate_pairs(data, response, columns, seed=seed)
            screen_fits += 1
        except Exception:
            surrogate = []
    elif surrogate is False:
        surrogate = []
    else:
        default_source = "neural" if surrogate["engine"] == "neural" else "surrogate"
        source_map = {frozenset(row["columns"]): "+".join(row["sources"])
                      for row in surrogate.get("pair_sources", []) if row["sources"]}
        surrogate = surrogate["pairs"]
    # A cached pre-scan may include columns absent from the accepted model.
    # Filter before applying the lead limit so invalid high-rank pairs do not
    # crowd out valid lower-rank proposals.
    if counts is not None:
        counts["surrogate_pairs"] = len(surrogate)
    eligible = [row for row in surrogate if row[0] in names and row[1] in names]
    for left, right, h2, variance in eligible[:top]:
        key = frozenset((left, right))
        if variance <= floor or h2 < _SURROGATE_MIN_H2:
            continue
        source = source_map.get(key, default_source)
        if key in known:
            leads = [replace(lead, source=f"{lead.source}+{source}", h2=float(h2))
                     if frozenset((lead.left, lead.right)) == key else lead for lead in leads]
            continue
        rho = _pair_rank_correlation(columns, left, right)
        if rho is not None and abs(rho) >= _DUPLICATE_AXIS_RANK:
            continue
        term = pair_term_for(left, right, levels=levels, groups=groups,
                             categorical=categorical, transforms=transforms,
                             data=data)
        kind = ("tensor" if "%X%" in term and term.startswith("bbs(")
                else "factor-by-factor" if "%X%" in term else "varying")
        leads.append(InteractionLead(left=left, right=right, removed=float(variance),
                                     term=term, kind=kind, source=source, h2=float(h2)))
    # A joint tree is the high-capacity alternative to the tensor surface for
    # the strongest screen signal (#267: on California housing a joint tree
    # beat the tensor decisively; the multi-column btree() already parses).
    # Offer it only for the single top-ranked tensor lead: each joint-tree
    # candidate costs validation fits, and the validation bar (not the count)
    # is what controls false discoveries. The tensor pipeline already covers
    # additional pairs.
    for lead in leads:
        if lead.kind == "tensor" and lead.removed > floor:
            leads.append(InteractionLead(
                left=lead.left, right=lead.right, removed=lead.removed,
                term=(f"btree({quote_name(lead.left)}, {quote_name(lead.right)}, "
                      f"maxdepth=3, selection=sse)"),
                kind="joint_tree", source=lead.source, h2=lead.h2))
            break
    return tuple(leads) + tuple(duplicates), screen_fits, reused


#: A surrogate pair is a lead only when at least this share of its joint
#: partial dependence is not additive: trees approximate a straight line with
#: steps, and on purely additive data those steps leave H^2 near 0.015.
_SURROGATE_MIN_H2 = 0.05

def _preexploration(data, response, columns, terms=(), *, seed=0, engine="native", mstop=300):
    """Convert columns for the Rust exploration and restore their public names."""
    import json

    from mboostrs._frame import read_column
    from mboostrs._rust import preexplore_json

    names = [c.name for c in columns
             if c.kind in ("numeric", "boolean", "categorical", "datetime")]
    if not names:
        return None
    raw = [read_column(data, name) for name in names]
    categories = [len(c.levels) if c.kind == "categorical" else 0 for c in raw]
    transforms = [1 if any(f"log({quote_name(name)})" in t for t in terms)
                  else 2 if any(f"sqrt({quote_name(name)})" in t for t in terms) else 0
                  for name in names]
    result = json.loads(call_fit(preexplore_json,
        np.ascontiguousarray(np.column_stack([c.values for c in raw]), dtype=float),
        np.ascontiguousarray(read_column(data, response).values, dtype=float),
        categories, transforms, engine=engine, seed=seed, rounds=mstop,
        pairs=len(names) > 1, _fit_count=2 if engine == "dual" else 1))
    for row in result["columns"]:
        row["column"] = names[row["index"]]
    result["priority"] = [names[i] for i in result["priority"]]
    result["pairs"] = [(names[j], names[k], h2, variance)
                       for j, k, h2, variance in result["pairs"]]
    for row in result.get("pair_sources", []):
        row["columns"] = [names[i] for i in row["indices"]]
    for row in result.get("subsets", []):
        row["columns"] = [names[i] for i in row["indices"]]
    return result


def _surrogate_pairs(data, response, columns, *, seed=0, mstop=300):
    """Pair proposals from the shared Rust pre-exploration pipeline."""
    result = _preexploration(data, response, columns, seed=seed, mstop=mstop)
    return result["pairs"] if result is not None else []


# ── Q6: which finding matters most ────────────────────────────────────────
#
# Five findings in a report are five things to read and no guidance about
# which to act on. The ordering is by what changes the model most if it is
# wrong: the family sets the loss every later choice is measured under, a
# missed interaction is structure no amount of tuning recovers, a shape is a
# local improvement, and `mstop` is the last dial to turn.


#: Rows left out for missing values are a priority when they are at least this
#: share of the frame; below it they are a note in the hygiene list.
_PRIORITY_INCOMPLETE_SHARE = 0.10


def priority_order(evidence: dict, hygiene, *, incomplete_rows=0, n_rows=0,
                   missing_by_column=None, treatment=()) -> tuple[str, ...]:
    """The findings, most consequential first, one line each.

    A column named like the response comes first: fitted as a predictor, a
    parallel outcome or a statistic of the response makes every later
    measurement flattering. A declared treatment comes before every
    measurement, because the report's job there is to say the battery did
    not answer the question it looks closest to. Missing values are one line
    about the rows left out, when those rows are a tenth of the frame or
    more, rather than one line per column.
    """
    lines: list[str] = []

    for name in treatment:
        lines.append(
            f"treatment: {name} is declared as the treatment, so this "
            f"recommendation leaves it out and measures no effect of it. "
            f"Predictive association is not an identification argument: "
            f"for the effect, use `mboostrs.causal` (`CausalSpec`, `aipw`, "
            f"`negative_control`)")

    siblings = [f for f in hygiene if f.kind == "response-sibling"]
    if siblings:
        names = ", ".join(f.column for f in siblings)
        lines.append(
            f"data: {names} {'is' if len(siblings) == 1 else 'are'} named like "
            f"the response; a parallel outcome or a statistic of the response "
            f"is not a predictor, so leave {'it' if len(siblings) == 1 else 'them'} "
            f"out unless known before the response is")

    blocking = [f for f in hygiene if f.kind in ("constant", "empty")]
    for finding in blocking[:2]:
        lines.append(f"data: {finding}")

    if n_rows and incomplete_rows / n_rows >= _PRIORITY_INCOMPLETE_SHARE:
        worst = sorted(((count, name) for name, count in (missing_by_column or {}).items()
                        if count / n_rows >= 0.05), reverse=True)
        named = (", mostly " + ", ".join(f"{name} ({count})" for count, name in worst[:3])
                 if worst else "")
        lines.append(
            f"data: {incomplete_rows} of {n_rows} rows "
            f"({incomplete_rows / n_rows:.0%}) are left out because a modelled "
            f"column is missing there{named}; decide whether the complete rows "
            f"are the population you mean, or leave the worst columns out")

    family = evidence.get("family")
    if family is not None:
        lines.append(f"family: {family.sentence}")

    spread = evidence.get("heteroscedasticity")
    if spread is not None and spread is not family \
            and getattr(spread.recommended, "param_names", None) is not None:
        lines.append(
            f"spread: {spread.sentence}; GaussianLSS models the scale as a "
            f"function of the covariate, which the mean-only Gaussian cannot")

    leads = evidence.get("interactions", ())
    actionable = [lead for lead in leads if lead.kind != "duplicate-axis"]
    if actionable:
        top = actionable[0]
        lines.append(
            f"structure: {top.left} x {top.right} removes {top.removed:.3f} of "
            f"the residual the additive terms leave — additive terms cannot "
            f"recover it at any mstop; add {top.term} to test it held out")

    shapes = evidence.get("shapes", ())
    changed = [row for row in shapes if row.chosen != row.incumbent]
    for row in changed[:2]:
        lines.append(f"shape: {row.column}: {row.sentence}")

    joint = evidence.get("joint_tree", {})
    if joint.get("adopted"):
        lines.append(f"shape: joint tree replaces the free terms "
                     f"({joint['relative_gain']:.1%} less shared-fold risk)")

    mstop = evidence.get("mstop")
    if mstop is not None and mstop.at_boundary:
        lines.append(f"mstop: {mstop.sentence}")

    if not lines:
        lines.append(
            "nothing measured contradicts the dtype-driven specification")
    return tuple(lines)
