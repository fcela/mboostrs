"""One-call discovery: screen -> fit -> diagnose -> propose.

``insights(data, response)`` runs the discovery pipeline in the right order and
returns a single ranked, deduplicated list of findings. It is thin glue over the
existing pieces in :mod:`mboostrs.terms` and :mod:`mboostrs.explore`; it fits no
new statistics of its own.

Pipeline
--------
1. **Screen**: :func:`mboostrs.terms.column_screen` ranks columns by the variance
   their binned means explain; :func:`mboostrs.terms.screen_interactions` ranks
   pairs by held-out residual MSE removed. Both are detections, not fits.
2. **Fit**: an additive specification over the screened mains -- ``bbs`` for
   a column with more than two distinct values, ``bols`` for a two-level one
   -- with ``mstop`` chosen by :func:`mboostrs.explore.tune_mstop`. The
   ``family`` argument is threaded through the fit; ``None`` infers one from
   the response (``{0, 1}`` -> ``Binomial``, non-negative integer counts ->
   ``Poisson``, otherwise the Gaussian default).
3. **Diagnose**: :func:`mboostrs.explore.term_report` (what each term earns,
   shape alternatives), :func:`mboostrs.explore.screen_interactions` on the
   fitted residuals (what is left), :func:`mboostrs.explore.heteroscedasticity`
   (does the spread move with the fit -- asked only under the Gaussian
   contract; a count or binary response gets the overdispersion check or
   nothing instead), :func:`mboostrs.explore.sensitivity`
   (effect directions and magnitudes).
4. **Propose**: :func:`mboostrs.explore.suggest_specifications` turns the report
   into ranked, measured suggestions; screen-stage detections are merged in and
   deduplicated, measured evidence winning over detected.

A finding's ``basis`` follows the :class:`mboostrs.explore.Suggestion`
convention: ``"measured"`` means the effect was fitted and scored on held-out
folds; ``"detected"`` means structure was found in a residual without any model
of it fitted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs._frame import read_column
from mboostrs._names import NAME_SRC as N
from mboostrs._names import quote_name, quote_response, unquote_name
from mboostrs._output import export_result


@dataclass(frozen=True)
class Finding:
    """One thing the discovery pipeline found, and what to do about it.

    ``basis`` says what kind of number ``effect_size`` is, following the
    :class:`~mboostrs.explore.Suggestion` convention: ``"measured"`` means the
    effect was fitted and scored on held-out folds; ``"detected"`` means
    structure was found in a residual without any model of it fitted. A
    detected signal says something is there; it does not say which learner
    will capture it.
    """

    finding: str
    effect_size: float
    uncertainty: float
    basis: str = "measured"
    evidence: dict = field(default_factory=dict)
    next_action: str = ""

    def __repr__(self):  # pragma: no cover - display only
        return (f"{self.finding}: {self.effect_size:+.4g} "
                f"(±{self.uncertainty:.2g}) [{self.basis}] -> {self.next_action}")

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record of the finding and its evidence."""
        return export_result(self, include_arrays=include_arrays)


def _screen_mains(data, response, top_k=8):
    """Top columns by binned-mean variance explained (detection)."""
    from mboostrs.terms import column_screen

    screen = column_screen(data, response)
    rows = []
    for row in screen.head(top_k).iter_rows(named=True):
        curve_r2 = float(row.get("curve_r2", 0.0) or 0.0)
        if curve_r2 <= 0.01:
            continue
        rows.append({
            "column": str(row.get("column", "")),
            "curve_r2": curve_r2,
            "linear_r2": float(row.get("linear_r2", 0.0) or 0.0),
            "hint": str(row.get("hint", "")),
        })
    return rows


def _infer_family(y):
    """The family a response suggests: {0, 1} is binary, integer counts are
    Poisson, anything else gets the Gaussian default."""
    from mboostrs.families import Binomial, Gaussian, Poisson

    present = y[~np.isnan(y)]
    if present.size == 0:
        return Gaussian()
    distinct = np.unique(present)
    if set(distinct.tolist()) == {0.0, 1.0}:
        return Binomial()
    integers = bool(np.all(distinct == np.round(distinct)))
    if integers and distinct.size > 2 and bool(np.all(distinct >= 0)):
        return Poisson()
    return Gaussian()


def _family_name(family) -> str:
    """The family's name, whether it arrived as a class or an instance."""
    return getattr(family, "__name__", None) or type(family).__name__


def _is_gaussian(family) -> bool:
    """Whether the raw residual spread is a heteroscedasticity test for this
    family -- the same contract :func:`mboostrs.explore.heteroscedasticity`
    holds itself to."""
    from mboostrs.families import Gaussian, GaussReg

    return isinstance(family, (Gaussian, GaussReg))


def _fit_additive(data, response, columns, family=None, seed=0):
    """Fit the screened mains -- ``bols`` on a column with at most two
    distinct values, ``bbs`` on the rest -- and return (model, mstop,
    formula)."""
    from mboostrs._control import boost_control
    from mboostrs.explore import tune_mstop
    from mboostrs.gamboost import gamboost

    terms = " + ".join(
        f"{'bols' if read_column(data, c).n_levels <= 2 else 'bbs'}({c})"
        for c in columns)
    formula = f"{quote_response(response)} ~ {terms}"
    tuned = tune_mstop(formula, data, family=family, seed=seed)
    mstop = int(tuned.mstop)
    model = gamboost(formula, data, family=family,
                     control=boost_control(mstop=mstop))
    return model, mstop, formula


def _coordinate_surface_finding(data, response, columns, family, mstop, seed):
    """A measured finding when a coordinate tree beats ``bspatial`` for a pair.

    When the screened columns include a coordinate pair (``x`` and ``y``, or
    ``lat`` and ``lon``, and the other names `autospec` reads as coordinates),
    the learners are fitted with the other screened columns on the same five
    folds: ``bspatial(x, y)``, and a tree on the pair, ``btree(x, y)`` or
    ``btree(x, y, maxdepth=3)`` as `autospec` picks between them. The finding
    is emitted when the tree's held-out risk is lower than the surface's by
    more than the standard error of the paired fold differences. Returns None
    when there is no pair, or when the surface is not beaten by that margin.
    """
    from types import SimpleNamespace

    from mboostrs._autospec import (
        _paired_gain,
        _relative_gain,
        _risk_floor,
        coordinate_beats,
        score_coordinate_candidates,
    )
    from mboostrs.autospec import _coordinate_pair_names
    from mboostrs.cvrisk import cv_weights

    pair = _coordinate_pair_names([SimpleNamespace(name=c) for c in columns])
    if pair is None:
        return None
    x, y = pair
    others = [c for c in columns if c not in pair]
    rest = [f"{'bols' if read_column(data, c).n_levels <= 2 else 'bbs'}({c})"
            for c in others]

    def with_pair(term):
        return f"{quote_response(response)} ~ " + " + ".join(rest + [term])

    stump_term = f"btree({quote_name(x)}, {quote_name(y)})"
    deep_term = f"btree({quote_name(x)}, {quote_name(y)}, maxdepth=3)"
    surface_term = f"bspatial({quote_name(x)}, {quote_name(y)}, df=6.0)"
    candidates = {"bspatial": with_pair(surface_term),
                  "btree": with_pair(stump_term),
                  "btree3": with_pair(deep_term)}
    n_rows = int(np.asarray(read_column(data, response).values).shape[0])
    folds = cv_weights(n_rows, type="kfold", B=5, seed=seed)
    nu = _boost_nu(mstop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        scores, _ = score_coordinate_candidates(
            data, response, candidates, family, folds, max(200, int(mstop)), nu)
    surface = scores["bspatial"]
    # The tree is the stump unless the depth-3 tree beats it by the shape
    # rule, as in autospec's coordinate comparison.
    deeper, _, _ = coordinate_beats(scores["btree"], scores["btree3"])
    tree = scores["btree3"] if deeper else scores["btree"]
    tree_term = deep_term if deeper else stump_term
    if not (np.isfinite(surface.risk) and np.isfinite(tree.risk)):
        return None
    paired = _paired_gain(surface, tree)
    if paired is None:
        return None
    mean, se = paired
    if not mean > se:
        return None
    floor = _risk_floor(family, data, response)
    share = _relative_gain(surface.risk, tree.risk, floor)
    scale = max(surface.risk - floor, 1e-12)
    return Finding(
        finding=f"{tree_term} beats {surface_term}",
        effect_size=float(share),
        uncertainty=float(se / scale),
        basis="measured",
        evidence={"columns": [x, y], "term": tree_term,
                  "surface_term": surface_term,
                  "surface_risk": float(surface.risk), "tree_risk": float(tree.risk),
                  "paired_difference": float(mean), "paired_se": float(se),
                  "folds": int(folds.shape[1]), "mstop": int(mstop)},
        next_action=(f"use {tree_term} for the location effect; keep {surface_term} "
                     "only if the surface is needed, and check both on held-out "
                     "data"),
    )


def _boost_nu(mstop):
    """The step length `insights` fits its additive model with."""
    from mboostrs._control import boost_control

    return float(boost_control(mstop=int(mstop)).nu)


def _dedupe_key(finding):
    """Identity of a finding for dedup: the columns it concerns, sorted."""
    cols = finding.evidence.get("columns")
    if cols:
        return ("cols", tuple(sorted(cols)))
    return ("text", finding.finding)


def insights(data, response, *, top_mains=8, top_pairs=5, seed=0,
             family=None):
    """Run screen -> fit -> diagnose -> propose; return ranked findings.

    Parameters
    ----------
    data : frame with the response column and candidate predictors.
    response : name of the response column.
    top_mains : how many screened columns enter the additive fit.
    top_pairs : how many screened pairs are reported as detections.
    seed : random seed threaded through the pipeline.
    family : model family for the fit. ``None`` infers one from the
        response: only values ``{0, 1}`` -> :class:`mboostrs.families.Binomial`,
        non-negative integers with more than two distinct values ->
        :class:`mboostrs.families.Poisson`, anything else the Gaussian
        default. The family used is recorded in every finding's evidence
        under ``"family"``.

    Returns
    -------
    list of :class:`Finding`, ranked measured-first, deduplicated across the
    screen and propose stages.
    """
    from mboostrs.explore import (
        heteroscedasticity,
        sensitivity,
        suggest_specifications,
        term_report,
    )
    from mboostrs.families import Poisson
    from mboostrs.terms import screen_interactions

    y = np.asarray(read_column(data, response).values, dtype=float)

    if family is None:
        family = _infer_family(y)
    family_used = _family_name(family)

    # 1. Screen ---------------------------------------------------------
    mains = _screen_mains(data, response, top_k=top_mains)
    if not mains:
        return []
    columns = [m["column"] for m in mains]

    # 2. Fit, 3. Diagnose -------------------------------------------------
    # The fold fits inside tune_mstop and term_report predict on rows a
    # fold's spline basis never saw, which is what their own extrapolation
    # policy is for -- not something to report to the caller of insights().
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        model, mstop, formula = _fit_additive(data, response, columns,
                                              family=family, seed=seed)
        resid = y - np.asarray(model.predict(type="response"),
                               dtype=float).ravel()

        report = term_report(formula, data, response=response, family=family,
                             mstop=mstop, seed=seed)
        screened_pairs = screen_interactions(data, resid, response=response,
                                             seed=seed)
        try:
            sens_rows = sensitivity(model, data)
        except ValueError:
            # sensitivity() cannot yet map a bols(<string column>) term's
            # derived __by<k> column back to its source (#309); the findings
            # then carry no direction rather than the whole call failing.
            sens_rows = []
    sens_by_var = {}
    for row in sens_rows:
        var = row.get("variable", "")
        slope = next((v for k, v in row.items()
                      if k.startswith("slope")), float("nan"))
        direction = ""
        if isinstance(slope, (int, float)) and np.isfinite(slope):
            direction = "increasing" if slope > 0 else "decreasing"
        sens_by_var[var] = direction

    findings = []

    # Measured: what each fitted term earns (from term_report).
    for contrib in getattr(report, "contributions", []):
        term = contrib.get("term", "") if isinstance(contrib, dict) else getattr(
            contrib, "term", "")
        earned = contrib.get("earns", 0.0) if isinstance(contrib, dict) else getattr(
            contrib, "earns", 0.0)
        se = contrib.get("se", 0.0) if isinstance(contrib, dict) else getattr(
            contrib, "se", 0.0)
        cols = [c for c in columns if c in term]
        direction = ""
        for var, d in sens_by_var.items():
            if var and var in term:
                direction = d
                break
        findings.append(Finding(
            finding=f"effect of {term}" + (f" ({direction})" if direction else ""),
            effect_size=float(earned),
            uncertainty=float(se),
            basis="measured",
            evidence={"columns": cols, "term": term, "mstop": mstop,
                      "out_of_fold_gain": float(earned)},
            next_action=(f"keep {term}" if earned > 0
                         else f"consider dropping {term}; refit and compare"),
        ))

    # Measured: a coordinate tree that beats the smooth surface on the pair.
    coordinate = _coordinate_surface_finding(data, response, columns, family,
                                             mstop, seed)
    if coordinate is not None:
        findings.append(coordinate)

    # Measured: residual spread -> distributional suggestion. Only a Gaussian
    # contract makes the raw residual spread a test; a Binomial or a Poisson
    # fit has spread that legitimately moves with the mean, so the spread
    # finding would fire on a perfectly correct fit.
    if _is_gaussian(family):
        het = heteroscedasticity(model)
        slope = het.get("slope", 0.0)
        t_stat = het.get("t", 0.0)
        if abs(t_stat) > 2.0:
            findings.append(Finding(
                finding="residual spread moves with the fitted value",
                effect_size=float(het.get("spread_ratio", 1.0)),
                uncertainty=float(1.0 / max(abs(t_stat), 1e-9)),
                basis="measured",
                evidence={"columns": [], "slope": float(slope),
                          "t": float(t_stat),
                          "spread_ratio": float(het.get("spread_ratio", 1.0))},
                next_action=("fit a distributional model (gamboostLSS) with a "
                             "scale predictor"),
            ))
    elif isinstance(family, Poisson):
        # For counts the question is not whether the spread moves with the
        # mean (Poisson says it does) but whether it moves *more* than the
        # mean: the Pearson dispersion, which is 1 for a true Poisson.
        mu = np.asarray(model.predict(type="response"), dtype=float).ravel()
        keep = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
        n_kept = int(keep.sum())
        if n_kept > 1:
            dispersion = float(
                np.sum((y[keep] - mu[keep]) ** 2 / mu[keep]) / (n_kept - 1))
            if dispersion > 1.5:
                findings.append(Finding(
                    finding="counts are overdispersed relative to Poisson",
                    effect_size=dispersion,
                    uncertainty=float("nan"),
                    # In-sample residual structure with no model of it
                    # fitted: a detection, not a held-out measurement.
                    basis="detected",
                    evidence={"columns": [], "dispersion": dispersion},
                    next_action=("compare NegBinomial (or a Poisson-scale "
                                 "LSS) against Poisson on held-out risk"),
                ))

    # Detected: screened pairs on the residual (no model fitted).
    for pair in list(screened_pairs)[:top_pairs]:
        left = pair.left if hasattr(pair, "left") else pair[0]
        right = pair.right if hasattr(pair, "right") else pair[1]
        removed = pair.removed if hasattr(pair, "removed") else pair[2]
        if removed <= 0:
            continue
        findings.append(Finding(
            finding=f"interaction between {left} and {right}",
            effect_size=float(removed),
            uncertainty=float("nan"),
            basis="detected",
            evidence={"columns": [left, right], "removed": float(removed),
                      # Every pair the screen ranked, not only the reported ones.
                      "candidates": len(screened_pairs)},
            next_action=(f"add an interaction term for {left} x {right}, "
                         "refit, and score it before believing it"),
        ))

    # Measured: suggestions from the report (one change at a time).
    import re
    for sugg in suggest_specifications(report, screened=screened_pairs):
        # Columns named in the suggestion term: 'bbs(x2) %X% bbs(x3)' or
        # the screened-pair form 'x2 x x3'.
        cols = sorted({unquote_name(c) for c in re.findall(rf"bbs\(({N})\)", sugg.term or "")})
        if not cols:
            m = re.fullmatch(r"\s*(\w+)\s*x\s*(\w+)\s*", sugg.term or "")
            cols = sorted(m.groups()) if m else []
        if not cols:
            cols = [c for c in columns if c in (sugg.term or "")]
        findings.append(Finding(
            finding=sugg.reason,
            effect_size=float(sugg.expected_gain),
            uncertainty=float("nan"),
            basis=sugg.basis,
            evidence={"columns": cols, "term": sugg.term,
                      "specification": sugg.specification,
                      **dict(sugg.evidence)},
            next_action=sugg.action,
        ))

    # 4. Deduplicate: measured wins over detected for the same columns ----
    for f in findings:
        f.evidence["family"] = family_used
    best = {}
    for f in findings:
        key = _dedupe_key(f)
        prev = best.get(key)
        if prev is None:
            best[key] = f
        elif prev.basis == "detected" and f.basis == "measured":
            best[key] = f
    findings = list(best.values())

    # Rank: measured first, then by |effect_size| descending.
    findings.sort(key=lambda f: (0 if f.basis == "measured" else 1,
                                 -abs(f.effect_size)
                                 if np.isfinite(f.effect_size) else 0.0))
    return findings
