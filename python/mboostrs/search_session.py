"""An agent-directed specification search over public, measured operations.

The session is an orchestrator, not an estimator. It retains the training
boundary, folds and workflow ledger while a human or external agent chooses
one next action. Statistical fits and scores use the existing native-backed
entry points. No provider connection or prescribed action order is required.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from mboostrs._constraints import (
    DEFAULT_RULE,
    admits,
    check_forbidden_pairs,
    check_heredity,
    check_mandatory,
    forbidden_hits,
    heredity_violations,
    joint_columns,
    main_columns,
    missing_mains,
    pair_label,
    rhs_terms,
    rule_record,
)
from mboostrs._frame import column_names, read_column
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._output import json_safe
from mboostrs._workflow import _CURRENT, BudgetExhausted, Workflow
from mboostrs.autospec import autospec
from mboostrs.explore import compare_specifications, cv_risk, split_terms, term_report, tune_mstop
from mboostrs.formula import term_variables
from mboostrs.gamboost import gamboost
from mboostrs.policy import CostBudget, _data_fingerprint, _fingerprint, _policy_folds
from mboostrs.terms import classify_columns, column_screen, screen_interactions

__all__ = ["SearchEvidence", "SearchSession", "open_search"]


@dataclass(frozen=True)
class SearchEvidence:
    """One action's result and its position in the search's audit trail."""

    action: str
    status: str
    evidence_id: str
    state_id: str
    fit_charges: int
    seconds: float
    summary: dict
    value: object = field(repr=False, compare=False, default=None)

    def to_dict(self, *, include_arrays=False):
        """Audit metadata; the arbitrary action value is retrieved separately."""
        return json_safe({
            "action": self.action, "status": self.status,
            "evidence_id": self.evidence_id, "state_id": self.state_id,
            "fit_charges": self.fit_charges, "seconds": self.seconds,
            "summary": self.summary,
        })


#: Quantiles whose values become candidate cutpoints in structure screening.
_CUTPOINT_GRID = (0.25, 0.5, 0.75)


def _screen_r2(basis, target) -> float:
    """Squared correlation of one basis column with a target vector.

    The single-column screening statistic: the share of the target's variance
    a typed basis column (a hinge, an indicator, a log slope, their product)
    lines up with, computed on the rows where both are finite.
    """
    b = np.asarray(basis, dtype=float)
    t = np.asarray(target, dtype=float)
    keep = np.isfinite(b) & np.isfinite(t)
    if keep.sum() < 10:
        return 0.0
    b = b[keep] - b[keep].mean()
    t = t[keep] - t[keep].mean()
    denominator = float(b @ b)
    if denominator <= 0.0:
        return 0.0
    total = float(t @ t)
    if total <= 0.0:
        return 0.0
    return float((b @ t) ** 2 / (denominator * total))


def _residualize(values, orthonormal):
    """`values` minus their projection onto the columns of `orthonormal`.

    Screening residuals of a fitted incumbent punishes basis columns that
    share content with the incumbent's own terms — a hinge on `w` screens
    weakly after `bols(w)` absorbed its linear part, and the knot ranking
    distorts toward extreme quantiles. Projecting both the basis column and
    the target out of the incumbent's design span first restores the
    ranking: the screen then reads what the basis adds beyond the fitted
    terms, not what survives by being unfamiliar to them.
    """
    v = np.asarray(values, dtype=float)
    if orthonormal is None:
        return v
    v = v - v.mean()
    return v - orthonormal @ (orthonormal.T @ v)


def _lead_pair(lead):
    """The two columns a pair lead joins, or ``None`` when it names no pair.

    Interaction leads carry ``left`` and ``right``. A lead that carries only its
    ``term`` is read from the term's columns instead.
    """
    left, right = getattr(lead, "left", None), getattr(lead, "right", None)
    if left is not None and right is not None:
        return left, right
    names = term_variables(str(getattr(lead, "term", "")))
    return (names[0], names[1]) if len(names) == 2 else None


def _centered(values):
    """A numeric column with its mean removed, for product screening."""
    column = np.asarray(values, dtype=float)
    return column - np.nanmean(column)


class SearchSession:
    """A persistent training-only search whose next diagnostic is caller-chosen.

    ``run`` performs exactly the named action. ``state`` lists the operations
    currently legal and their preconditions. The data/fold/metric boundary is
    fixed when the session opens. Test rows are accepted only by the one-time
    ``final_evaluate`` action after ``freeze``. The Python caller must keep its
    own test data out of training and domain-context arguments.
    """

    def __init__(self, data, response, *, family=None, score="mse", folds=5,
                 seed=0, weights=None, groups=(), budget=None,
                 na_action="error", unseen_levels="average", score_parameter=None,
                 nu=0.1, mstop=100, locks=(), causal=None, forbidden_pairs=(),
                 mandatory_terms=(), heredity="none"):
        if response not in column_names(data):
            raise ValueError(f"response {response!r} is absent from the training data")
        if budget is not None and not isinstance(budget, CostBudget):
            raise TypeError("budget must be a CostBudget")
        if score not in ("family", "mse", "brier", "log_loss", "pinball", "expectile"):
            raise ValueError(f"unsupported score {score!r}")
        if score != "family":
            from mboostrs import _rust

            _rust.response_score(np.array([0.]), np.array([[.5]]), np.ones(1),
                                 score, score_parameter)
        elif score_parameter is not None:
            raise ValueError("score_parameter is only used by pinball or expectile")
        groups = (groups,) if isinstance(groups, str) else tuple(groups)
        locks = (locks,) if isinstance(locks, str) else tuple(locks)
        if causal is not None:
            from mboostrs.causal import CausalSpec

            if not isinstance(causal, CausalSpec) or causal.outcome != response:
                raise ValueError("causal must be a CausalSpec for this response")
            locks = tuple(dict.fromkeys((*locks, *causal.covariates)))
        unknown = set(locks) - (set(column_names(data)) - {response})
        if unknown:
            raise ValueError(f"locks name absent or response columns: {sorted(unknown)}")
        self._data = data
        self.causal = causal
        self._forbidden = (set() if causal is None else
                           {causal.treatment, *causal.post_treatment})
        # Search-wide constraints. They are fixed when the session opens, so a
        # checkpoint restores the rule the evidence was measured under.
        self.heredity = check_heredity(heredity)
        self.forbidden_pairs = tuple(check_forbidden_pairs(
            forbidden_pairs, column_names(data), response))
        self.mandatory_terms = check_mandatory(
            mandatory_terms, column_names(data), response)
        if set(self.mandatory_terms) & self._forbidden:
            raise ValueError("mandatory_terms names causal treatment or post-treatment "
                             "columns, which the search never uses")
        self.response = response
        self.family = family
        self.score = score
        self.score_parameter = score_parameter
        self.seed = int(seed)
        self.weights = weights
        self.groups = groups
        self.locks = locks
        self._opening_locks = locks
        self.nu = float(nu)
        from mboostrs._validate import check_int_at_least

        self.mstop = check_int_at_least("mstop", mstop, 1)
        self._opening_mstop = self.mstop
        self.na_action = na_action
        self.unseen_levels = unseen_levels
        self.folds = _policy_folds(data, response, folds, groups, seed, weights)
        self.data_id = _data_fingerprint(data)
        self.fold_id = _fingerprint(self.folds)
        self._tracker = Workflow(budget or CostBudget())
        self._opening_charges = 0
        self._family_id = _fingerprint(family)
        # The family and fold count the session opened with, which the
        # transcript records so a curve can rebuild the same partition.
        self._opening_family = family
        self._opening_family_id = self._family_id
        self._fold_count = (folds if isinstance(folds, int) and not isinstance(folds, bool)
                            else None)
        self.spec = None
        self.model = None
        self._fitted_by_spec = {}
        self.frozen = False
        self.evaluated = False
        self.hypotheses = []
        self.evidence: list[SearchEvidence] = []
        self._declaration_checks = None
        opening = {"training": self.data_id, "folds": self.fold_id,
                   "response": response, "score": score,
                   "build": self._build_id(), "seed": seed,
                   "score_parameter": score_parameter,
                   "budget": self._tracker.budget.to_dict(),
                   "nu": self.nu, "initial_mstop": self._opening_mstop,
                   "na_action": na_action,
                   "unseen_levels": unseen_levels,
                   "locks": locks, "groups": groups,
                   "causal_id": _fingerprint(causal),
                   "input_weights": _fingerprint(weights)}
        # A session with no constraint keeps the identifier it had before the
        # constraints existed, so earlier checkpoints still replay.
        if self.constraints != DEFAULT_RULE:
            opening["constraints"] = self.constraints
        self._state_id = self._hash(opening)

    @staticmethod
    def _hash(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

    @property
    def constraints(self):
        """The search-wide rule in force: heredity, forbidden pairs, mandatory terms."""
        return rule_record(self.heredity, self.forbidden_pairs, self.mandatory_terms)

    def _pair_barred(self, left, right, rhs=None):
        """Why a pair of columns may not be nominated in ``rhs``, or ``None``.

        ``rhs`` defaults to the current specification. The heredity test reads
        the main effects that specification already has.
        """
        columns = frozenset((left, right))
        if any(columns == pair for pair in self.forbidden_pairs):
            return f"forbidden pair {pair_label(columns)}"
        mains = main_columns(rhs_terms(self.spec if rhs is None else rhs))
        if not admits(columns, mains, self.heredity):
            return (f"heredity {self.heredity}: the specification lacks a main effect "
                    f"of {', '.join(sorted(columns - mains)) or 'either column'}")
        return None

    @staticmethod
    def _provider_record(judgment):
        calls = getattr(judgment, "calls", ())
        if not calls:
            return None
        record = calls[-1]
        return {key: record.get(key) for key in
                ("llm", "endpoint", "prompt_sha256", "reply_sha256", "usage", "status")}

    def _check_training(self):
        if _data_fingerprint(self._data) != self.data_id:
            raise ValueError("training data changed after the search session opened")
        if _fingerprint(self.folds) != self.fold_id:
            raise ValueError("validation folds changed after the search session opened")
        if _fingerprint(self.family) != self._family_id:
            raise ValueError("family parameters changed without a recorded set_family action")

    def _numeric_diagnostics(self, left, right, data=None):
        """Native numeric diagnostics on complete rows of two named columns."""
        from mboostrs import _rust

        source = self._data if data is None else data
        try:
            x = np.asarray(read_column(source, left).values, dtype=float)
            y = (np.asarray(right, dtype=float) if not isinstance(right, str)
                 else np.asarray(read_column(source, right).values, dtype=float))
        except (TypeError, ValueError):
            return None
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 10:
            return None
        return dict(_rust.column_diagnostics(np.ascontiguousarray(x[keep]),
                                             np.ascontiguousarray(y[keep])))

    def _model_frame(self, model):
        """Discovery data restricted to the rows `model` was fitted on.

        A fit with na_action='omit' estimates on the complete cases of its
        formula, so residual-length diagnostics must read those rows; the
        fitted model records the dropped positions in `_na_dropped`.
        """
        data = self._discovery_data()
        dropped = getattr(model, "_na_dropped", None) or []
        if not len(dropped):
            return data
        from mboostrs._frame import take_rows

        names = column_names(data)
        n = len(read_column(data, names[0]).values)
        keep = np.ones(n, dtype=bool)
        keep[np.asarray(dropped, dtype=int)] = False
        return take_rows(data, keep)

    def _dispersion_attribution(self):
        """Mean-versus-dispersion attribution for a fitted count model.

        A covariate the mean has already absorbed leaves both screens quiet.
        A covariate the mean is missing lines up with the *signed* Pearson
        residual. A covariate that drives only the dispersion leaves the
        signed residual alone but moves ``log`` of its square — the split
        ``heteroscedasticity`` cannot make for counts, whose spread
        legitimately grows with the mean. Screening only: the rows say which
        compare to run, nothing switches family here.
        """
        model = self.model
        frame = self._model_frame(model)
        mu = np.asarray(model.predict(type="response"), dtype=float).ravel()
        y = np.asarray(read_column(frame, self.response).values, dtype=float)
        variance = mu
        nuisance = model.nuisance()
        theta = None
        if type(self.family).__name__ == "NBinomial" and nuisance is not None:
            values = np.ravel(np.asarray(nuisance, dtype=float))
            if values.size:
                theta = float(values[0])
                variance = mu + mu * mu / theta
        scaled = (y - mu) / np.sqrt(np.maximum(variance, 1e-12))
        log_square = np.log(np.maximum(scaled * scaled, 1e-12))
        in_mean = set(term_variables(self.spec or ""))
        rows = []
        for name in column_names(frame):
            if name == self.response:
                continue
            column = read_column(frame, name)
            if column.kind != "numeric":
                continue
            values = np.asarray(column.values, dtype=float)
            signed_r2 = _screen_r2(values, scaled)
            log_square_r2 = _screen_r2(values, log_square)
            if signed_r2 >= .01:
                attribution = "ambiguous" if log_square_r2 >= .01 else "mean"
            elif log_square_r2 >= .01:
                attribution = "dispersion"
            else:
                continue
            rows.append({"column": name, "in_mean": name in in_mean,
                         "signed_residual_r2": signed_r2,
                         "log_square_r2": log_square_r2,
                         "attribution": attribution})
        rows.sort(key=lambda row: max(row["signed_residual_r2"],
                                      row["log_square_r2"]), reverse=True)
        dispersion_columns = [row["column"] for row in rows
                              if row["attribution"] == "dispersion"]
        return {"rows": rows[:8], "bar": "1% of residual-variance share",
                "theta": theta,
                "dispersion_columns": dispersion_columns,
                "recipe": ("compare a distributional alternative before adding "
                           "these to the mean: gamboostLSS with the current mean "
                           "terms on mu and the dispersion columns on sigma "
                           "(PoissonLSS or NBinomialLSS); measure, then adopt"),
                "interpretation": ("screening attribution; a dispersion row is a "
                                   "sigma candidate, not evidence for another "
                                   "mean term")}

    def _proxy_notes(self, contributions):
        """Advisory correlation with a locked sibling, never an automatic drop."""
        notes = []
        for row in contributions:
            variable, se = row.get("variable"), row.get("se")
            if variable is None or se is None or row["earns"] > 2 * se:
                continue
            if set(term_variables(row["term"])) & set(self.mandatory_terms):
                continue
            for locked in self.locks:
                if variable == locked:
                    continue
                diagnostics = self._numeric_diagnostics(variable, locked)
                if diagnostics is None or abs(diagnostics["pearson"]) < .8:
                    continue
                nested = " + ".join(term for term in split_terms(self.spec)
                                    if term != row["term"])
                notes.append({"term": row["term"], "locked_sibling": locked,
                              "correlation": diagnostics["pearson"],
                              "conditional_earn": row["earns"], "earn_se": se,
                              "drop_candidate": nested,
                              "interpretation": "possible proxy; compare the nested drop before deciding"})
        return notes

    #: mgcv's "worst" concurvity at or above which a term is reported: the
    #: term is then nearly a function of the others in the model, so its
    #: partial effect and the others' are not separately identified.
    CONCURVITY_THRESHOLD = 0.8

    #: How many of the most-concurved pairs the term report names.
    CONCURVITY_PAIR_LIMIT = 5

    def _concurvity_notes(self):
        """Terms that are near function-space duplicates of the rest, and
        the most-concurved pairs of terms.

        The current specification is fitted at the session's mstop and
        refitted by REML (`gam_refit`, terms it cannot represent held), and
        mgcv's three concurvity measures are read per term (Wood, Generalized
        Additive Models, 2nd ed., section 5.6.2). The boosted fit's pairwise
        measures then name which two terms are duplicates of each other,
        which the per-term rows cannot. A finding, never a drop: concurvity
        says the effects cannot be told apart, not which to keep.

        Returns ``(notes, pairs)``: ``notes`` holds one record per term
        whose worst-case concurvity reaches ``CONCURVITY_THRESHOLD``;
        ``pairs`` holds at most ``CONCURVITY_PAIR_LIMIT`` records for the
        most-concurved unordered pairs of terms, each at or above the same
        threshold.
        """
        from mboostrs import boost_control
        from mboostrs.gamboost import gamboost
        from mboostrs.selection import gam_refit

        model = gamboost(f"{quote_response(self.response)} ~ {self.spec}", self._data, family=self.family,
                         control=boost_control(self.mstop, self.nu), weights=self.weights,
                         na_action=self.na_action)
        try:
            refit = gam_refit(model, unsupported="hold")
        except ValueError:
            refit = None
        notes = []
        if refit is not None:
            for row in refit.concurvity(output="records"):
                if row["worst"] >= self.CONCURVITY_THRESHOLD:
                    notes.append({**row,
                                  "interpretation": (
                                      "nearly a function of the other terms; its partial "
                                      "effect and theirs are not separately identified. "
                                      "Compare dropping it or merging the duplicates "
                                      "before reading either effect")})
        return notes, self._concurvity_pairs(model)

    def _concurvity_pairs(self, model):
        """The most-concurved unordered pairs of terms, from the boosted fit.

        The pairwise measures come from the fitted model itself
        (`concurvity(pairwise=True)`, computed by the core from the boosting
        path), so no refit is needed. Each pair is reported once, at the
        worse of its two ordered values.
        """
        if not hasattr(model, "concurvity"):
            return []
        worst = {}
        for row in model.concurvity(pairwise=True, output="records"):
            pair = tuple(sorted((row["term"], row["against"])))
            current = worst.get(pair)
            if current is None or row["worst"] > current["worst"]:
                worst[pair] = row
        pairs = []
        for (left, right), row in sorted(worst.items(),
                                         key=lambda item: -item[1]["worst"]):
            if row["worst"] < self.CONCURVITY_THRESHOLD or len(pairs) >= self.CONCURVITY_PAIR_LIMIT:
                break
            pairs.append({"pair": [left, right],
                          "worst": row["worst"], "observed": row["observed"],
                          "estimate": row["estimate"],
                          "interpretation": (
                              f"{left} and {right} reach into each other's "
                              "function space; whichever is read, its shape can "
                              "be carried by the other. Compare the nested drops "
                              "before keeping both")})
        return pairs

    def _threshold_sibling(self, left, right, residual, data=None):
        """Nominate a one-split varying effect when a margin looks step-like."""
        for threshold, slope in ((left, right), (right, left)):
            diagnostics = self._numeric_diagnostics(threshold, residual, data=data)
            if diagnostics is None:
                continue
            distinct = diagnostics["distinct"]
            step = (distinct <= 2 or
                    (diagnostics["integer_like"] and distinct <= 12) or
                    (diagnostics["step_share"] >= .7 and diagnostics["step_r2"] >= .02))
            if not step:
                continue
            term = (f"bols({quote_name(slope)}, by={quote_name(threshold)})" if distinct <= 2 else
                    f"btree({quote_name(threshold)}, by={quote_name(slope)}, maxdepth=1)")
            return {"thresholded_sibling_suggested": True,
                    "threshold_column": threshold, "sibling_term": term,
                    "next_action": "transform_candidates"}
        return {"thresholded_sibling_suggested": False,
                "threshold_column": None, "sibling_term": None,
                "next_action": None}

    def available_actions(self):
        """Operations available now, with a short statement of their purpose."""
        actions = {
            "classify": "read column types and proposed main-effect learners; no fit",
            "column_screen": "inspect marginal response associations; screening only",
            "autospec": "request a bounded family/shape discovery report; no adoption",
            "pre_scan": "run a native tree, neural or dual nomination probe",
            "transform_candidates": "nominate log1p and one-split interaction siblings",
            "structure_candidates": "nominate hinge, cutpoint and basis-interaction terms",
            "hypothesis": "record a variable role, proposed explanation and falsifier",
            "set_family": "declare the fitting family before comparing candidates",
        }
        if self.causal is not None:
            actions["causal_traps"] = "check the declared estimand and adjustment design"
        if self.family is not None:
            actions["set_spec"] = "set an editable right-hand side; unmeasured until compared"
        if self.spec is not None:
            actions["lock"] = "protect an already included predictor from later edits"
        if self.spec is not None and self.family is not None:
            actions.update({
                "compare": "score named candidates on the shared training folds",
                "evaluate": "score the current specification at a stated stopping iteration",
                "term_report": "measure term removal and shape alternatives",
                "tune": "choose mstop and retain the resulting full-data fit",
                "check_declarations": "test each declared sign or shape of the current "
                                      "specification against the training folds",
                "fit": "fit the current specification at the stated mstop",
            })
            if type(self.family).__name__ == "Gaussian" and self.score == "mse":
                actions["response_probe"] = "compare identity, log or log1p response fits on original-scale MSE"
        if self.model is not None:
            actions["interaction_screen"] = "screen residual pairs of the current fitted model"
            actions["partial_effects"] = "inspect fitted component effects at stated rows"
            actions["freeze"] = "lock the specification before final evaluation"
        if self.spec is not None and self.family is not None:
            actions["pair_candidates"] = "nominate pairs from an accepted fitted model's residuals"
            actions["tree_candidate"] = "nominate one tree over the accepted columns"
        if any(row.action == "compare" and row.status == "measured"
               for row in self.evidence):
            actions["adopt"] = "adopt a candidate that clears the recorded search rule"
        if self.frozen and not self.evaluated:
            actions = {"final_evaluate": "score one untouched test sample once"}
        if self.evaluated:
            actions = {}
        costs = {"classify": "0", "column_screen": "0", "hypothesis": "0",
                 "set_family": "0", "set_spec": "0", "lock": "0",
                 "pre_scan": "1 or 2",
                 "transform_candidates": "1 or 2; reused report costs 0",
                 "structure_candidates": "1; reused report costs 0",
                 "causal_traps": "0",
                 "autospec": "up to the requested probe allowance",
                 "interaction_screen": "0", "partial_effects": "0",
                 "pair_candidates": "0 with a retained fit", "tree_candidate": "0",
                 "adopt": "0", "freeze": "0", "final_evaluate": "0",
                 "fit": "1", "compare": "at least one per candidate and fold",
                 "response_probe": "one fit per response candidate and fold",
                 "evaluate": "at least one per fold",
                 "term_report": "at least one per term and fold",
                 "check_declarations": "one fit per fold and one full-data fit per "
                                       "declared sign or shape; 0 when none is declared",
                 "tune": "at least one per fold plus a full-data fit"}
        return {name: {"purpose": purpose, "fit_charges": costs[name]}
                for name, purpose in actions.items()}

    def _session_spec(self):
        """What the current specification declares, in the form
        `mboostrs.declarations` reads."""
        from mboostrs import boost_control
        from mboostrs.declarations import _Spec

        return _Spec(formula=f"{quote_response(self.response)} ~ {self.spec}", family=self.family,
                     control=boost_control(self.mstop, nu=self.nu),
                     response=self.response, groups=tuple(self.groups),
                     locks=tuple(self.locks),
                     treatment=() if self.causal is None else (self.causal.treatment,))

    def _declaration_key(self):
        """Identifies the specification a stored declaration check was run on."""
        return (self.spec, self.mstop, self._family_id)

    def _declaration_view(self):
        """Each declaration of the current specification, with its last check when
        that check was run on the same specification and mstop."""
        from mboostrs.declarations import declaration_items

        if self.spec is None:
            return []
        checks = {}
        if self._declaration_checks is not None and \
                self._declaration_checks[0] == self._declaration_key():
            checks = {row.declaration: row.to_dict()
                      for row in self._declaration_checks[1]}
        return [{**item, "check": checks.get(item["declaration"])}
                for item in declaration_items(self._session_spec())]

    def state(self):
        """Bounded, JSON-safe view for an analyst or an external agent."""
        budget = self._tracker.report()
        operations = budget.pop("operations")
        budget["operation_count"] = len(operations)
        budget["recent_operations"] = operations[-8:]
        return json_safe({
            "state_id": self._state_id, "data_id": self.data_id,
            "fold_id": self.fold_id, "response": self.response,
            "family": None if self.family is None else type(self.family).__name__,
            "score": self.score, "score_parameter": self.score_parameter,
            "fold_scheme": self.folds.scheme, "groups": self.groups,
            "locks": self.locks, "spec": self.spec, "mstop": self.mstop,
            "constraints": self.constraints,
            "forbidden_columns": sorted({self.response, *self._forbidden}),
            "causal_design": None if self.causal is None else self.causal.describe(),
            "frozen": self.frozen, "evaluated": self.evaluated,
            "available_actions": self.available_actions(),
            "budget": budget,
            "hypothesis_count": len(self.hypotheses),
            "recent_hypotheses": self.hypotheses[-8:],
            "evidence_count": len(self.evidence),
            "recent_evidence_ids": [row.evidence_id for row in self.evidence[-12:]],
            "declarations": self._declaration_view(),
        })

    def checkpoint(self):
        """Export a training-only search for replay in another Python process.

        Fitted native objects are not serialized. Resume restores every
        comparison and charge, but asks for one explicit ``fit`` or ``tune``
        action before screening residuals or freezing a model. The returned
        record does not include training or test rows; pass the same training
        frame to :meth:`resume` and its fingerprint is checked.
        """
        if self.frozen:
            raise ValueError("checkpoint before freeze; a native fitted model is not serialized")
        rows = []
        for evidence in self.evidence:
            record = evidence.to_dict()
            if evidence.action == "compare" and evidence.status == "measured":
                record["comparison_rows"] = json_safe(evidence.value)
            rows.append(record)
        return json_safe({
            "version": 1, "data_id": self.data_id, "fold_id": self.fold_id,
            "build_id": self._build_id(),
            "response": self.response, "family_id": _fingerprint(self.family),
            "score": self.score, "score_parameter": self.score_parameter,
            "folds": {**self.folds.to_dict(include_arrays=True),
                      "base_weights": self.folds.base_weights},
            "seed": self.seed, "groups": self.groups, "locks": self.locks,
            "initial_locks": self._opening_locks,
            "constraints": self.constraints,
            "causal": None if self.causal is None else asdict(self.causal),
            "input_weights": self.weights,
            "nu": self.nu, "mstop": self.mstop,
            "initial_mstop": self._opening_mstop, "spec": self.spec,
            "na_action": self.na_action, "unseen_levels": self.unseen_levels,
            "state_id": self._state_id, "hypotheses": self.hypotheses,
            "evidence": rows, "budget_report": self._tracker.report(),
            "opening_fit_charges": self._opening_charges,
            "model_replay": ("fit or tune required after resume" if self.model is not None
                             else "no fitted model retained"),
        })

    def export_transcript(self):
        """Export a JSON-safe decision record, including after freeze.

        This record contains no training rows, fold arrays or fitted model. It
        documents what was measured and the final test score, if available;
        it cannot be resumed for fitting. Use :meth:`checkpoint` before freeze
        when a resumable training session is needed.
        """
        final = next((row.summary for row in reversed(self.evidence)
                      if row.action == "final_evaluate" and row.status == "measured"), None)
        return json_safe({
            "version": 1, "kind": "search_transcript",
            "response": self.response, "family": (None if self.family is None
                                                  else type(self.family).__name__),
            "score": self.score, "score_parameter": self.score_parameter,
            "spec": self.spec, "mstop": self.mstop, "locks": self.locks,
            "groups": self.groups, "data_id": self.data_id, "fold_id": self.fold_id,
            "fold_count": self._fold_count, "seed": self.seed,
            "nu": self.nu, "na_action": self.na_action,
            "unseen_levels": self.unseen_levels, "weighted": self.weights is not None,
            "family_id": self._family_id,
            "opening_family": (None if self._opening_family is None
                               else type(self._opening_family).__name__),
            "opening_family_id": self._opening_family_id,
            "build_id": self._build_id(), "state_id": self._state_id,
            "constraints": self.constraints,
            "frozen": self.frozen, "evaluated": self.evaluated,
            "final_evaluation": final, "hypotheses": self.hypotheses,
            "evidence": [row.to_dict() for row in self.evidence],
            "budget_report": self._tracker.report(),
        })

    @classmethod
    def resume(cls, data, record, *, family=None):
        """Restore the exact fold/evidence boundary, requiring the same data."""
        from mboostrs.causal import CausalSpec
        from mboostrs.folds import FoldScheme

        if record.get("kind") == "search_transcript":
            raise ValueError("a metadata-only transcript cannot resume fitting; use a pre-freeze checkpoint")
        if record.get("version") != 1:
            raise ValueError("unsupported search checkpoint version")
        if _data_fingerprint(data) != record["data_id"]:
            raise ValueError("checkpoint training data do not match")
        if cls._build_id() != record["build_id"]:
            raise ValueError("checkpoint native build does not match")
        if _fingerprint(family) != record["family_id"]:
            raise ValueError("checkpoint family or its parameters do not match")
        stored = record["folds"]
        folds = FoldScheme(
            np.asarray(stored["weights"], dtype=float),
            np.asarray(stored["held_out"], dtype=bool), stored["scheme"],
            seed=stored["seed"],
            groups=None if stored["groups"] is None else np.asarray(stored["groups"]),
            strata=None if stored["strata"] is None else np.asarray(stored["strata"]),
            ordered=stored["ordered"],
            judge=None if stored["judge"] is None else np.asarray(stored["judge"], dtype=bool),
            base_weights=(None if stored["base_weights"] is None else
                          np.asarray(stored["base_weights"], dtype=float)),
            repeats=stored["repeats"],
        )
        if _fingerprint(folds) != record["fold_id"]:
            raise ValueError("checkpoint fold design does not match its fingerprint")
        report = record["budget_report"]
        causal = record["causal"]
        if causal is not None:
            causal = dict(causal)
            for key in ("covariates", "post_treatment", "treatment_contrast", "assumptions"):
                causal[key] = tuple(causal[key])
            causal = CausalSpec(**causal)
        # Checkpoints written before the constraints existed carry none: they
        # were measured with the default rule.
        rule = record.get("constraints") or DEFAULT_RULE
        session = cls(
            data, record["response"], family=family, score=record["score"],
            score_parameter=record["score_parameter"], folds=folds,
            seed=record["seed"], groups=record["groups"],
            weights=record["input_weights"],
            budget=CostBudget(**report["budget"]), na_action=record["na_action"],
            unseen_levels=record["unseen_levels"], nu=record["nu"],
            mstop=record["initial_mstop"], locks=record["initial_locks"],
            causal=causal, heredity=rule["heredity"],
            forbidden_pairs=[tuple(pair) for pair in rule["forbidden_pairs"]],
            mandatory_terms=tuple(rule["mandatory_terms"]),
        )
        if session.fold_id != record["fold_id"]:
            raise ValueError("restored fold design differs after validation")
        initial_state = session._state_id
        session.spec = record["spec"]
        session.locks = tuple(record["locks"])
        session.mstop = record["mstop"]
        session.hypotheses = list(record["hypotheses"])
        session._state_id = record["state_id"]
        session._tracker.charges = report["fit_charges"]
        session._opening_charges = record["opening_fit_charges"]
        session._tracker.events = list(report["operations"])
        session._tracker.stopped = report["stopped"]
        session._tracker.blocked_request = report["blocked_request"]
        session._tracker.started = time.perf_counter() - report["elapsed_seconds"]
        previous = initial_state
        for index, item in enumerate(record["evidence"]):
            expected_id = cls._hash({
                "previous": previous, "index": index, "action": item["action"],
                "status": item["status"], "summary": item["summary"],
                "fit_charges": item["fit_charges"],
            })
            expected_state = cls._hash({"previous": previous,
                                        "evidence_id": expected_id})
            if item["evidence_id"] != expected_id or item["state_id"] != expected_state:
                raise ValueError(f"checkpoint evidence {index} fails its identity chain")
            if item["action"] == "compare" and item["status"] == "measured":
                compact = [
                    {key: row[key] for key in ("specification", "rhs", "out-of-fold risk",
                                                "change", "change se")}
                    for row in item["comparison_rows"]
                ]
                if compact != item["summary"]["rows"]:
                    raise ValueError(f"checkpoint comparison {index} differs from its evidence")
            session.evidence.append(SearchEvidence(
                item["action"], item["status"], item["evidence_id"],
                item["state_id"], item["fit_charges"], item["seconds"],
                item["summary"], item.get("comparison_rows"),
            ))
            previous = expected_state
        if previous != record["state_id"]:
            raise ValueError("checkpoint terminal state does not match its evidence chain")
        if (sum(item["fit_charges"] for item in record["evidence"])
                + record["opening_fit_charges"] != report["fit_charges"]):
            raise ValueError("checkpoint fit charges differ from the evidence ledger")
        return session

    @staticmethod
    def _build_id():
        from mboostrs.build import build_info

        return build_info()["build_id"]

    def _record(self, action, status, summary, value, before, started):
        charges = self._tracker.charges - before
        index = len(self.evidence)
        payload = {"previous": self._state_id, "index": index,
                   "action": action, "status": status, "summary": summary,
                   "fit_charges": charges}
        evidence_id = self._hash(payload)
        self._state_id = self._hash({"previous": self._state_id,
                                     "evidence_id": evidence_id})
        result = SearchEvidence(action, status, evidence_id, self._state_id,
                                charges, time.perf_counter() - started,
                                json_safe(summary), value)
        self.evidence.append(result)
        return result

    def drive(self, judgment, *, max_actions=4):
        """Let a bounded judge choose successive public actions.

        ``judgment`` receives only ``state()`` and a small reply schema. It
        may be a human-written callable or :class:`mboostrs.LLMJudge` with a
        configured client. This method never passes test rows or runs final
        evaluation. A failed or malformed reply is logged and stops the
        driver; the caller can continue manually from the retained state.
        """
        from mboostrs._validate import check_int_at_least

        limit = check_int_at_least("max_actions", max_actions, 1)
        for _ in range(limit):
            state = self.state()
            legal = {name: value for name, value in state["available_actions"].items()
                     if name not in ("final_evaluate",)}
            if not legal:
                return "no_available_actions"
            digest = {"purpose": "search_action", "state": {**state,
                      "available_actions": legal},
                      "reply_schema": {"action": "one available action or stop",
                                       "arguments": "object of action arguments",
                                       "state_id": state["state_id"],
                                       "reasoning": "why this action answers the current question"}}
            before = self._tracker.charges
            started = time.perf_counter()
            try:
                reply = judgment(digest)
                if not isinstance(reply, Mapping) or set(reply) != {
                        "action", "arguments", "state_id", "reasoning"}:
                    raise ValueError("judge reply must match the action schema exactly")
                if reply["state_id"] != state["state_id"]:
                    raise ValueError("judge reply cites a stale state_id")
                if not isinstance(reply["arguments"], dict) or not isinstance(reply["reasoning"], str):
                    raise ValueError("judge arguments must be an object and reasoning a string")
                action = reply["action"]
                if action != "stop" and action not in legal:
                    raise ValueError(f"judge requested unavailable action {action!r}")
            except Exception as exc:
                summary = {"reason": str(exc)[:300]}
                if provider := self._provider_record(judgment):
                    summary["provider_call"] = provider
                self._record("judge", "failed", summary,
                             None, before, started)
                return "judge_failed"
            summary = {"chosen_action": action, "reasoning": reply["reasoning"],
                       "cited_state_id": reply["state_id"]}
            if provider := self._provider_record(judgment):
                summary["provider_call"] = provider
            self._record("judge", "unmeasured", summary, None, before, started)
            if action == "stop":
                return "judge_stopped"
            try:
                result = self.run(action, **reply["arguments"])
            except (TypeError, ValueError, RuntimeError):
                return "action_failed"
            if result.status in ("refused", "failed"):
                return result.status
        return "action_limit"

    def _validate_rhs(self, rhs):
        from mboostrs.modeler import _candidate_problem
        from mboostrs.transforms import extract_transforms

        if not isinstance(rhs, str) or not rhs.strip():
            raise ValueError("specification must be a non-empty right-hand side")
        source_columns = set(term_variables(rhs))
        available = set(column_names(self._data))
        forbidden = {self.response, *self._forbidden}
        if source_columns & forbidden:
            raise ValueError(f"names forbidden column(s) {sorted(source_columns & forbidden)}")
        if source_columns - available:
            raise ValueError(f"names unknown column(s) {sorted(source_columns - available)}")
        if set(self.locks) - source_columns:
            raise ValueError(f"would drop locked column(s) {sorted(set(self.locks) - source_columns)}")
        terms = rhs_terms(rhs)
        hits = forbidden_hits(terms, self.forbidden_pairs)
        if hits:
            term, pair = hits[0]
            raise ValueError(f"term {term!r} reads forbidden pair {pair_label(pair)}; "
                             "forbidden pairs are never nominated or adopted")
        dropped = missing_mains(terms, self.mandatory_terms)
        if dropped:
            raise ValueError(f"would drop mandatory column(s) {dropped}: keep a main-effect "
                             "term for each mandatory column")
        bad = heredity_violations(terms, self.heredity)
        if bad:
            columns = sorted(joint_columns(bad[0]))
            raise ValueError(f"heredity {self.heredity}: interaction {bad[0]!r} needs a "
                             f"main-effect term for {columns} in the same specification")
        rewritten, recipe = extract_transforms(f"{quote_response(self.response)} ~ {rhs}", self._data)
        _, _, parsed_rhs = partition_tilde(rewritten)
        problem = _candidate_problem(
            parsed_rhs, response=self.response, locks=(), forbidden=set(),
            columns=available | {step.name for step in recipe.steps})
        if problem:
            raise ValueError(problem)
        return rhs.strip()

    def run(self, action, **options):
        """Run one chosen operation and return its evidence and native value."""
        if action not in self.available_actions():
            raise ValueError(f"action {action!r} is unavailable; call state() for legal actions")
        self._check_training()
        before = self._tracker.charges
        started = time.perf_counter()
        status = "measured"
        try:
            if action in ("freeze", "final_evaluate"):
                value, summary = self._execute(action, options)
            else:
                active = _CURRENT.get()
                if active is None:
                    with self._tracker:
                        value, summary = self._execute(action, options)
                elif active is self._tracker:
                    value, summary = self._execute(action, options)
                else:
                    if self.evidence or self._tracker.charges:
                        raise ValueError("cannot switch workflow ledgers after search actions")
                    if self._tracker.budget != active.budget:
                        raise ValueError("session budget must match the enclosing workflow")
                    self._tracker = active
                    self._opening_charges = active.charges
                    before = active.charges
                    value, summary = self._execute(action, options)
            if action in ("classify", "column_screen", "hypothesis", "pre_scan",
                          "transform_candidates",
                          "structure_candidates",
                          "set_spec", "set_family", "lock", "interaction_screen",
                          "pair_candidates", "tree_candidate", "causal_traps",
                          "adopt", "fit", "freeze"):
                status = ("screening_only" if action in
                          ("classify", "column_screen", "hypothesis", "pre_scan",
                           "transform_candidates",
                           "structure_candidates",
                           "interaction_screen", "pair_candidates", "tree_candidate",
                           "causal_traps") else "unmeasured")
            if action == "autospec" and value.fits == 0:
                status = "screening_only"
        except BudgetExhausted as exc:
            return self._record(action, "refused", {"reason": str(exc),
                         "blocked_request": exc.report["blocked_request"]},
                         None, before, started)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._record(action, "failed", {"reason": str(exc)[:500]},
                         None, before, started)
            raise
        return self._record(action, status, summary, value, before, started)

    def _execute(self, action, options):
        accepted = {
            "classify": set(), "column_screen": {"columns"},
            "autospec": {"budget", "preexploration", "mstop", "df",
                         "interactions", "residual_tree", "joint_tree", "nu", "step"},
            "pre_scan": {"engine", "rounds"},
            "transform_candidates": {"engine", "rounds", "report"},
            "structure_candidates": {"engine", "rounds", "report", "columns"},
            "hypothesis": {"statement", "variable", "role", "would_count_against"},
            "set_family": {"family"}, "set_spec": {"rhs"}, "lock": {"columns"},
            "compare": {"candidates", "mstop"},
            "response_probe": {"transforms"},
            "evaluate": {"mstop", "fit_options"},
            "adopt": {"comparison_id", "candidate", "simplification_se"},
            "term_report": {"mstop", "shapes", "proxy_hygiene", "concurvity"},
            "check_declarations": {"folds", "seed"},
            "tune": {"start", "fold_local", "fit_options"},
            "fit": {"fit_options"},
            "interaction_screen": {"pairs", "groups", "seed"},
            "pair_candidates": {"rhs", "model", "mstop", "surrogate"},
            "tree_candidate": {"rhs"},
            "partial_effects": {"which", "newdata"},
            "causal_traps": set(),
            "freeze": set(), "final_evaluate": {"data", "weights"},
        }
        unexpected = set(options) - accepted[action]
        if unexpected:
            raise ValueError(f"{action} does not accept {sorted(unexpected)}")
        if action == "classify":
            kinds = classify_columns(self._discovery_data(), response=self.response)
            return kinds, {"columns": [{"name": k.name, "kind": k.kind,
                                        "suggested": getattr(k, "suggested", None)} for k in kinds]}
        if action == "column_screen":
            names = options.get("columns")
            if names is not None and set(names) & self._forbidden:
                raise ValueError("causal treatment or post-treatment columns cannot be screened")
            rows = column_screen(self._discovery_data(), self.response, names, output="records")
            return rows, {"rows": rows[:12], "total_columns": len(rows)}
        if action == "autospec":
            allowance = options.get("budget", 0)
            rec = autospec(self._discovery_data(), self.response, family=self.family,
                           budget=allowance, folds=self.folds, seed=self.seed,
                           groups=self.groups, preexploration=options.get("preexploration", "native"),
                           mstop=options.get("mstop", self.mstop), df=options.get("df", 4.0),
                           interactions=options.get("interactions", "report"),
                           residual_tree=options.get("residual_tree", False),
                           joint_tree=options.get("joint_tree", True),
                           nu=options.get("nu", self.nu), step=options.get("step", "fixed"),
                           heredity=self.heredity,
                           forbid=[tuple(sorted(pair)) for pair in self.forbidden_pairs],
                           mandatory=self.mandatory_terms)
            return rec, {"formula": rec.formula, "family": type(rec.family).__name__,
                         "constraints": self.constraints,
                         "mstop": getattr(rec, "mstop", self.mstop),
                         "fits_reported": rec.fits,
                         "probe_budget": allowance,
                         "preexploration": options.get("preexploration", "native"),
                         "requested_mstop": options.get("mstop", self.mstop),
                         "measured": getattr(rec, "measured", ()),
                         "defaulted": getattr(rec, "defaulted", ())}
        if action == "pre_scan":
            from mboostrs._autospec import _preexploration

            screen_data = self._discovery_data()
            columns = [read_column(screen_data, name) for name in column_names(screen_data)
                       if name != self.response]
            terms = [] if self.spec is None else split_terms(self.spec)
            report = _preexploration(screen_data, self.response, columns, terms,
                                     seed=self.seed, engine=options.get("engine", "native"),
                                     mstop=options.get("rounds", 300))
            return report, {"engine": report["engine"], "priority": report["priority"],
                            "pair_sources": report.get("pair_sources", [])[:8],
                            "interpretation": "screening proposal, not adoption"}
        if action == "transform_candidates":
            from mboostrs._autospec import _preexploration

            report = options.get("report")
            if report is None:
                screen_data = self._discovery_data()
                columns = [read_column(screen_data, name) for name in column_names(screen_data)
                           if name != self.response]
                report = _preexploration(
                    screen_data, self.response, columns,
                    [] if self.spec is None else split_terms(self.spec),
                    seed=self.seed, engine=options.get("engine", "native"),
                    mstop=options.get("rounds", 100))
            if not isinstance(report, Mapping) or "columns" not in report or "pairs" not in report:
                raise ValueError("report must be a pre_scan result with columns and pairs")
            allowed = set(column_names(self._discovery_data())) - {self.response}
            if any(item.get("column") not in allowed for item in report["columns"]):
                raise ValueError("pre_scan report names a forbidden or unknown column")
            leads = []
            for item in report["columns"]:
                name = item["column"]
                if read_column(self._data, name).kind != "numeric":
                    continue
                diagnostics = self._numeric_diagnostics(name, self.response)
                if diagnostics is None or diagnostics["skewness"] <= 1.0:
                    continue
                values = np.asarray(read_column(self._data, name).values, dtype=float)
                if np.any(~np.isfinite(values)) or np.any(values < 0):
                    continue
                raw_term = next((term for term in split_terms(self.spec or "")
                                 if term.startswith((f"bols({quote_name(name)}", f"bbs({quote_name(name)}"))), None)
                term = (raw_term.replace(f"({quote_name(name)}", f"(log1p({quote_name(name)})", 1)
                        if raw_term else f"bols(log1p({quote_name(name)}))")
                rhs = (self.spec.replace(raw_term, term, 1) if raw_term else
                       f"{self.spec} + {term}" if self.spec else term)
                leads.append({"kind": "log1p_predictor", "column": name,
                              "term": term, "candidate_rhs": rhs,
                              "screen_skewness": diagnostics["skewness"],
                              "interpretation": "domain-safe shape lead; compare before adoption"})
            by_name = {item["column"]: item for item in report["columns"]}

            def threshold_margin(name):
                if by_name[name].get("partial", {}).get("step_share", 0) >= .7:
                    return True
                diagnostics = self._numeric_diagnostics(name, self.response)
                return bool(diagnostics and diagnostics["integer_like"]
                            and diagnostics["distinct"] <= 12)

            barred = {}
            for left, right, h2, variance in report["pairs"][:8]:
                if left not in by_name or right not in by_name:
                    continue
                threshold = next((name for name in (left, right)
                                  if threshold_margin(name)), None)
                if threshold is None:
                    continue
                slope = right if threshold == left else left
                if read_column(self._data, slope).kind != "numeric":
                    continue
                reason = self._pair_barred(threshold, slope)
                if reason is not None:
                    barred[f"{threshold} x {slope}"] = reason
                    continue
                term = f"btree({quote_name(threshold)}, by={quote_name(slope)}, maxdepth=1)"
                rhs = f"{self.spec} + {term}" if self.spec else term
                leads.append({"kind": "single_cutpoint_interaction",
                              "column": threshold, "partner": slope,
                              "term": term, "candidate_rhs": rhs,
                              "screen_h2": h2, "screen_variance": variance,
                              "interpretation": "one-split tree learns a cutpoint; compare before adoption"})
                if sum(item["kind"] == "single_cutpoint_interaction" for item in leads) >= 3:
                    break
            response_diagnostics = self._numeric_diagnostics(self.response, self.response)
            response_hint = None
            if response_diagnostics is not None:
                observed = np.asarray(read_column(self._data, self.response).values, dtype=float)
                if np.all(observed >= 0) and response_diagnostics["skewness"] > 1.0:
                    response_hint = "positive skew: compare identity and log1p response on original-scale folds"
            spread_hint = None
            dispersion_hint = None
            if self.model is not None:
                family_name = type(self.family).__name__
                if family_name in ("Poisson", "NBinomial"):
                    dispersion_hint = self._dispersion_attribution()
                else:
                    from mboostrs import heteroscedasticity

                    try:
                        spread = heteroscedasticity(self.model,
                                                    self._model_frame(self.model))
                    except ValueError:
                        spread = {}
                    if abs(spread.get("t", 0.0)) >= 2:
                        spread_hint = {"strongest": spread.get("strongest"),
                                       "spread_ratio": spread.get("spread_ratio"),
                                       "suggestion": "consider explicit LSS scale modeling; no family switch"}
            return leads, {"leads": leads[:8], "response_hint": response_hint,
                           "constraint_barred": barred, "constraints": self.constraints,
                           "spread_hint": spread_hint, "dispersion_hint": dispersion_hint,
                           "source_engine": report.get("engine"),
                           "interpretation": "screening only; measure candidate_rhs with compare"}
        if action == "structure_candidates":
            from mboostrs._autospec import _preexploration

            report = options.get("report")
            if report is None:
                screen_data = self._discovery_data()
                columns = [read_column(screen_data, name) for name in column_names(screen_data)
                           if name != self.response]
                report = _preexploration(
                    screen_data, self.response, columns,
                    [] if self.spec is None else split_terms(self.spec),
                    seed=self.seed, engine=options.get("engine", "native"),
                    mstop=options.get("rounds", 100))
            if not isinstance(report, Mapping) or "priority" not in report:
                raise ValueError("report must be a pre_scan result with a priority list")
            allowed = set(column_names(self._discovery_data())) - {self.response}
            priority_order = [name for name in report.get("priority", []) if name in allowed]
            named = set(priority_order)
            priority_order += [name for name in column_names(self._data)
                               if name in allowed and name not in named]
            if options.get("columns"):
                wanted = set(options["columns"])
                priority_order = [name for name in priority_order if name in wanted]
            numeric = [name for name in priority_order
                       if read_column(self._data, name).kind == "numeric"][:5]
            span = None
            if self.model is not None:
                frame = self._model_frame(self.model)
                target = np.asarray(self.model.residuals(type="response"), dtype=float)
                try:
                    blocks = self.model.extract(what="design")
                    design = np.column_stack([
                        np.asarray(block, dtype=float).reshape(len(target), -1)
                        for block in blocks.values()]) if blocks else None
                except Exception:
                    design = None
                if design is not None and design.shape[1]:
                    basis_with_intercept = np.column_stack(
                        [np.ones(len(target)), design - design.mean(axis=0)])
                    span, _ = np.linalg.qr(basis_with_intercept)
                target_note = ("residuals of the current fitted specification, "
                               "residualized on its design span")
            else:
                frame = self._discovery_data()
                observed = np.asarray(read_column(self._data, self.response).values,
                                      dtype=float)
                target = observed - np.nanmean(observed)
                target_note = "the demeaned response (no model fitted yet)"
            if numeric and len(target) != len(read_column(frame, numeric[0]).values):
                raise ValueError("screening frame and target lengths disagree")
            target = _residualize(target, span)

            def candidate_r2(basis):
                return _screen_r2(_residualize(basis, span), target)

            # Cutpoints a fitted btree actually used, when the incumbent
            # carries one: empirical knots from the soak itself, ranked by how
            # many boosting steps chose them, joined to the quantile grid.
            fitted_cuts = {}
            if self.model is not None:
                try:
                    for info in self.model.btree_splits().values():
                        for split in info["splits"]:
                            fitted_cuts.setdefault(split["variable"], []).append(
                                (split["steps"], split["cutpoint"]))
                except Exception:
                    fitted_cuts = {}

            leads = []
            per_column = {}
            for name in numeric:
                values = np.asarray(read_column(frame, name).values, dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size < 20:
                    continue
                entries = []
                centred = finite - finite.mean()
                scale = float(np.sqrt(np.mean(centred ** 2)))
                skew = (float(np.mean(centred ** 3)) / scale ** 3
                        if scale > 0 else 0.0)
                if np.all(finite >= 0) and skew > 1.0:
                    entries.append({"basis": "log1p", "kind": "log1p_slope",
                                    "values": np.log1p(np.maximum(values, 0.0))})
                best_hinge = best_indicator = None
                spread = finite.max() - finite.min()
                cuts = {float(np.quantile(finite, share))
                        for share in _CUTPOINT_GRID}
                for _steps, cut in sorted(fitted_cuts.get(name, ()),
                                          reverse=True):
                    if len(cuts) >= 6:
                        break
                    if any(abs(existing - cut) < .02 * spread for existing in cuts):
                        continue
                    cuts.add(cut)
                for cut in sorted(cuts):
                    if spread <= 0 or (cut - finite.min()) / spread < .05:
                        continue
                    hinge = {"basis": "hinge", "kind": "hinge_term", "cutpoint": cut,
                             "values": np.maximum(values - cut, 0.0)}
                    indicator = {"basis": "indicator", "kind": "cutpoint_indicator",
                                 "cutpoint": cut, "values": (values >= cut).astype(float)}
                    if best_hinge is None or candidate_r2(hinge["values"]) > best_hinge[1]:
                        best_hinge = (hinge, candidate_r2(hinge["values"]))
                    if (best_indicator is None
                            or candidate_r2(indicator["values"]) > best_indicator[1]):
                        best_indicator = (indicator, candidate_r2(indicator["values"]))
                # A column whose every cutpoint sits against the edge of its
                # range contributes no hinge or indicator; both may be None.
                screened = [pair[0] for pair in (best_hinge, best_indicator)
                            if pair is not None]
                for entry in list(entries) + screened:
                    entry = dict(entry)
                    entry["r2"] = candidate_r2(entry["values"])
                    per_column.setdefault(name, []).append(entry)

            # Marginal leads: the typed term itself, ready for compare.
            for name, entries in per_column.items():
                for entry in entries:
                    if entry["kind"] == "log1p_slope":
                        term = f"bols(log1p({quote_name(name)}))"
                        lead = {"kind": "log1p_slope", "column": name}
                    elif entry["kind"] == "hinge_term":
                        term = f"bols(pmax({name}, {entry['cutpoint']:.6g}))"
                        lead = {"kind": "hinge_term", "column": name,
                                "cutpoint": entry["cutpoint"]}
                    else:
                        term = f"btree({quote_name(name)}, maxdepth=1)"
                        lead = {"kind": "cutpoint_indicator", "column": name,
                                "cutpoint": entry["cutpoint"]}
                    lead.update({"term": term,
                                 "candidate_rhs": (f"{self.spec} + {term}" if self.spec
                                                   else term),
                                 "screen_r2": entry["r2"],
                                 "interpretation": "typed structure lead; compare before adoption"})
                    leads.append(lead)

            # Product leads: a transformed slope that changes across a
            # threshold on another column. Screened as centered products so a
            # main effect alone cannot light the product up.
            barred = {}
            for slope_name in numeric:
                slope_entries = [e for e in per_column.get(slope_name, [])
                                 if e["kind"] == "log1p_slope"]
                if not slope_entries:
                    continue
                slope = _centered(slope_entries[0]["values"])
                for threshold_name in numeric:
                    if threshold_name == slope_name:
                        continue
                    indicator = next((e for e in per_column.get(threshold_name, [])
                                      if e["kind"] == "cutpoint_indicator"), None)
                    if indicator is None:
                        continue
                    reason = self._pair_barred(threshold_name, slope_name)
                    if reason is not None:
                        barred[f"{threshold_name} x {slope_name}"] = reason
                        continue
                    product = slope * _centered(indicator["values"])
                    r2 = candidate_r2(product)
                    term = f"btree({quote_name(threshold_name)}, by=log1p({quote_name(slope_name)}), maxdepth=1)"
                    leads.append({"kind": "basis_interaction",
                                  "slope_column": slope_name, "slope_transform": "log1p",
                                  "threshold_column": threshold_name,
                                  "cutpoint": indicator["cutpoint"],
                                  "term": term,
                                  "candidate_rhs": (f"{self.spec} + {term}" if self.spec
                                                    else term),
                                  "screen_r2": r2,
                                  "interpretation": ("typed basis interaction; the tree "
                                                     "learns the cutpoint; compare before "
                                                     "adoption")})
            leads.sort(key=lambda row: row["screen_r2"], reverse=True)
            for lead in leads:
                lead["screens_positive"] = bool(lead["screen_r2"] >= .01)
            return leads, {"leads": leads[:8], "constraint_barred": barred,
                           "constraints": self.constraints,
                           "screened_columns": numeric,
                           "target": target_note,
                           "bar": "screen_r2 >= 1% of target variance",
                           "screen_positive": sum(lead["screens_positive"]
                                                  for lead in leads[:8]),
                           "source_engine": report.get("engine"),
                           "interpretation": "screening only; measure candidate_rhs with compare"}
        if action == "hypothesis":
            text = options.get("statement", "").strip()
            if not text or not options.get("would_count_against", "").strip():
                raise ValueError("hypothesis needs a statement and what would count against it")
            record = {"statement": text, "variable": options.get("variable"),
                      "role": options.get("role"),
                      "would_count_against": options["would_count_against"],
                      "status": "analyst-stated; not established by data"}
            self.hypotheses.append(record)
            return record, record
        if action == "set_family":
            self.family = options["family"]
            self._family_id = _fingerprint(self.family)
            self.model = None
            self._fitted_by_spec.clear()
            return self.family, {"family": type(self.family).__name__,
                                 "family_id": self._family_id}
        if action == "set_spec":
            self.spec = self._validate_rhs(options["rhs"])
            self.model = None
            return self.spec, {"spec": self.spec, "status": "unmeasured"}
        if action == "lock":
            requested = options.get("columns", ())
            requested = (requested,) if isinstance(requested, str) else tuple(requested)
            used = set(term_variables(self.spec))
            invalid = set(requested) - used
            if invalid:
                raise ValueError(f"only existing specification columns can be locked: {sorted(invalid)}")
            self.locks = tuple(dict.fromkeys((*self.locks, *requested)))
            return self.locks, {"locked_columns": self.locks}
        if action == "compare":
            candidates = options.get("candidates", {})
            if not isinstance(candidates, dict) or not candidates or "current" in candidates:
                raise ValueError("compare needs named candidates other than 'current'")
            candidates = {name: self._validate_rhs(rhs) for name, rhs in candidates.items()}
            rows = compare_specifications(
                {"current": self.spec, **candidates}, self._data, self.response,
                family=self.family, folds=self.folds,
                mstop=options.get("mstop", self.mstop), nu=self.nu, seed=self.seed,
                weights=self.weights, score=self.score,
                score_parameter=self.score_parameter, na_action=self.na_action,
                unseen_levels=self.unseen_levels, include_evaluations=True)
            baseline_terms = set(split_terms(self.spec))
            contenders = []
            soft_removals = []
            for row in rows[1:]:
                terms = set(split_terms(row["rhs"]))
                candidate = {"candidate": row["specification"], "rhs": row["rhs"],
                             "change": row["change"], "change_se": row["change se"]}
                if row["change"] < -2 * row["change se"]:
                    contenders.append(candidate)
                elif terms < baseline_terms and row["change"] <= 2 * row["change se"]:
                    soft_removals.append(candidate)
            def order(row):
                return row["change"], len(split_terms(row["rhs"])), row["candidate"]
            contenders.sort(key=order)
            soft_removals.sort(key=order)
            # The prefer-interpretable decision needs one more number than the
            # rows carry: how far each other clearing candidate trails the
            # best, in paired SE of their difference on the shared folds.
            from mboostrs.explore import _paired_se, _partitions

            winner_gaps = []
            if len(contenders) > 1:
                by_name = {row["specification"]: row for row in rows}
                best_name = contenders[0]["candidate"]
                best_scores = by_name[best_name]["evaluation"].fold_scores
                for candidate in contenders[1:]:
                    scores = by_name[candidate["candidate"]]["evaluation"].fold_scores
                    gap = candidate["change"] - contenders[0]["change"]
                    paired = float(_paired_se(scores, best_scores,
                                              _partitions(self.folds)))
                    winner_gaps.append({"candidate": candidate["candidate"],
                                        "gap_vs_best": gap,
                                        "paired_se": paired,
                                        "within_two_se": bool(gap <= 2 * paired)})
            # The evaluations were the working material for the gaps; the
            # evidence value and checkpoint carry plain rows only.
            rows = [{key: value for key, value in row.items()
                     if key != "evaluation"} for row in rows]
            summary = {"baseline": self.spec,
                       "family": type(self.family).__name__,
                       "score": self.score, "fold_id": self.fold_id,
                       "two_se_candidates": contenders,
                       "also_cleared": contenders[1:],
                       "winner_gaps": winner_gaps,
                       "soft_removal_candidates": soft_removals,
                       "rows": [
                {key: row[key] for key in ("specification", "rhs", "out-of-fold risk",
                                            "change", "change se")}
                       for row in rows], "rule": "two paired SE; explicit caller adoption; descriptive heuristic"}
            # The evidence ID is assigned by _record after this call returns.
            return rows, summary
        if action == "response_probe":
            from mboostrs import _rust

            transforms = tuple(options.get("transforms", ("identity", "log1p")))
            if (not transforms or transforms[0] != "identity"
                    or len(set(transforms)) != len(transforms)
                    or any(kind not in ("identity", "log", "log1p") for kind in transforms)):
                raise ValueError("transforms must start with identity and contain unique log/log1p choices")
            rows = []
            baseline = None
            for kind in transforms:
                evaluation = cv_risk(
                    self.spec, self._data, self.response, family=self.family,
                    folds=self.folds, mstop=self.mstop, nu=self.nu, seed=self.seed,
                    weights=self.weights, score="mse", na_action=self.na_action,
                    unseen_levels=self.unseen_levels, return_result=True,
                    retain_arrays=False,
                    response_transform=None if kind == "identity" else kind)
                if baseline is None:
                    baseline = evaluation
                    paired = None
                else:
                    paired = dict(_rust.paired_gain(
                        np.ascontiguousarray(baseline.fold_scores, dtype=float),
                        np.ascontiguousarray(evaluation.fold_scores, dtype=float),
                        partitions=getattr(self.folds, "repeats", 1)))
                rows.append({"transform": kind, "formula": evaluation.formula,
                             "mse": evaluation.score, "fold_scores": evaluation.fold_scores,
                             "gain_vs_identity": None if paired is None else paired["gain"],
                             "paired_se": None if paired is None else paired["se"],
                             "clears_two_se": False if paired is None else paired["passed"]})
            return rows, {"rows": rows, "metric": "original-response MSE",
                          "family": type(self.family).__name__,
                          "decision": "measured only; no response or family switch",
                          "limitation": ("inverse of mean log is a back-transformed point "
                                         "prediction, not a bias-corrected conditional mean")}
        if action == "evaluate":
            fit_options = dict(options.get("fit_options") or {})
            if "weights" in fit_options or "nu" in fit_options:
                raise ValueError("weights and nu belong to the session, not fit_options")
            evaluated = cv_risk(
                self.spec, self._data, self.response, family=self.family,
                folds=self.folds, mstop=options.get("mstop", self.mstop),
                nu=self.nu, seed=self.seed, weights=self.weights,
                score=self.score, score_parameter=self.score_parameter,
                na_action=self.na_action, unseen_levels=self.unseen_levels,
                return_result=True, **fit_options)
            return evaluated, {"score": evaluated.score,
                               "fold_scores": evaluated.fold_scores,
                               "metric": self.score, "mstop": options.get("mstop", self.mstop),
                               "fits_reported": evaluated.fit_count}
        if action == "adopt":
            evidence_id = options.get("comparison_id")
            evidence = next((row for row in self.evidence if row.evidence_id == evidence_id
                             and row.action == "compare" and row.status == "measured"), None)
            if evidence is None:
                raise ValueError("adopt requires a measured comparison_id from this session")
            if (evidence.summary["baseline"] != self.spec
                    or evidence.summary["family"] != type(self.family).__name__
                    or evidence.summary["score"] != self.score
                    or evidence.summary["fold_id"] != self.fold_id):
                raise ValueError("comparison is stale for the current specification or design")
            name = options.get("candidate")
            row = next((item for item in evidence.summary["rows"]
                        if item["specification"] == name), None)
            if row is None or name == "current":
                raise ValueError("candidate is absent from the cited comparison")
            smaller = set(split_terms(row["rhs"])) < set(split_terms(self.spec))
            tolerance = float(options.get("simplification_se", 0.0))
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError("simplification_se must be finite and non-negative")
            passed = (row["change"] < -2 * row["change se"] or
                      (smaller and row["change"] <= tolerance * row["change se"]))
            if not passed:
                raise ValueError("candidate does not clear the stated adoption rule")
            self.spec = row["rhs"]
            self.model = None
            return self.spec, {"spec": self.spec, "comparison_id": evidence_id,
                               "candidate": name, "change": row["change"],
                               "change_se": row["change se"],
                               "simplification_se": tolerance}
        if action == "check_declarations":
            if self.weights is not None:
                raise ValueError("check_declarations does not take observation weights; "
                                 "open the session without weights to test declarations")
            if self.spec is None:
                raise ValueError("check_declarations needs a specification; call set_spec first")
            from mboostrs.declarations import check_declarations

            records = check_declarations(self._session_spec(), self._data,
                                         folds=int(options.get("folds", 5)),
                                         seed=int(options.get("seed", self.seed)))
            self._declaration_checks = (self._declaration_key(), records)
            return records, {"declarations": [row.to_dict() for row in records],
                             "tested": sum(row.test is not None for row in records)}
        if action == "term_report":
            if self.score not in ("family", "mse"):
                raise ValueError("term_report currently supports family risk or MSE")
            report = term_report(f"{quote_response(self.response)} ~ {self.spec}", self._data,
                                 family=self.family, folds=self.folds,
                                 mstop=options.get("mstop", self.mstop), nu=self.nu,
                                 seed=self.seed, score=self.score,
                                 na_action=self.na_action,
                                 shapes=options.get("shapes", False))
            notes = (self._proxy_notes(report.contributions)
                     if options.get("proxy_hygiene", False) else [])
            concurvity, concurvity_pairs = (self._concurvity_notes()
                                            if options.get("concurvity", False)
                                            else ([], []))
            return report, {"baseline": report.baseline,
                            "terms": report.contributions[:12],
                            "proxy_notes": notes,
                            "concurvity_notes": concurvity,
                            "concurvity_pairs": concurvity_pairs,
                            "fits_reported": report.fits}
        if action == "tune":
            fit_options = dict(options.get("fit_options") or {})
            if "weights" in fit_options or "nu" in fit_options:
                raise ValueError("weights and nu belong to the session, not fit_options")
            tuning = tune_mstop(f"{quote_response(self.response)} ~ {self.spec}", self._data,
                                self.family, folds=self.folds, seed=self.seed,
                                start=options.get("start", self.mstop), nu=self.nu,
                                weights=self.weights,
                                score=self.score, score_parameter=self.score_parameter,
                                unseen_levels=self.unseen_levels,
                                fold_local=options.get("fold_local", self.score != "family"),
                                na_action=self.na_action, **fit_options)
            self.mstop = tuning.mstop
            self.model = tuning.model
            if self.model is not None:
                self._fitted_by_spec[self.spec] = self.model
            return tuning, {"mstop": tuning.mstop, "score": self.score,
                            "fits_reported": getattr(tuning, "total_fits", tuning.fits)}
        if action == "fit":
            from mboostrs._control import boost_control

            fit_options = dict(options.get("fit_options") or {})
            if "weights" in fit_options or "nu" in fit_options:
                raise ValueError("weights and nu belong to the session, not fit_options")
            step = fit_options.pop("step", "fixed")
            self.model = gamboost(f"{quote_response(self.response)} ~ {self.spec}", self._data,
                                  family=self.family, na_action=self.na_action,
                                  control=boost_control(self.mstop, self.nu, step=step),
                                  weights=self.weights, **fit_options)
            self._fitted_by_spec[self.spec] = self.model
            return self.model, {"spec": self.spec, "mstop": self.mstop}
        if action == "interaction_screen":
            residual = self.model.residuals(type="response")
            screen_data = self._model_frame(self.model)
            screen_groups = tuple(dict.fromkeys((*self.groups, *options.get("groups", ()))))
            leads = screen_interactions(screen_data, residual, response=self.response,
                                        groups=screen_groups, seed=options.get("seed", self.seed),
                                        pairs=options.get("pairs"))
            barred = {}
            admitted = []
            for row in leads:
                reason = self._pair_barred(row.left, row.right)
                if reason is None:
                    admitted.append(row)
                else:
                    barred[f"{row.left} x {row.right}"] = reason
            leads = admitted
            top_pairs = []
            for row in leads[:6]:
                top_pairs.append({"left": row.left, "right": row.right,
                                  "removed": row.removed,
                                  **(self._threshold_sibling(row.left, row.right, residual,
                                                              data=screen_data)
                                     if row.removed > 0 else
                                     {"thresholded_sibling_suggested": False,
                                      "threshold_column": None, "sibling_term": None,
                                      "next_action": None})})
            return leads, {"top_pairs": top_pairs, "constraint_barred": barred,
                           "constraints": self.constraints,
                           "interpretation": ("removed is residual-screen MSE, not a paired "
                                              "two-SE comparison; compare before adoption")}
        if action == "pair_candidates":
            from mboostrs.policy import _incumbent_pair_leads

            rhs = options.get("rhs", self.spec)
            fitted = options.get("model") or self._fitted_by_spec.get(rhs)
            if fitted is None:
                raise ValueError("pair_candidates requires a retained fitted model for rhs")
            leads, fits = _incumbent_pair_leads(
                self._model_frame(fitted), self.response, self.family, rhs,
                fitted, options.get("mstop", getattr(fitted, "mstop", self.mstop)),
                self.groups, self.seed, options.get("surrogate", False))
            barred = {}
            admitted = []
            for lead in leads:
                pair = _lead_pair(lead)
                reason = None if pair is None else self._pair_barred(*pair, rhs)
                if reason is None:
                    admitted.append(lead)
                else:
                    barred[f"{pair[0]} x {pair[1]}"] = reason
            leads = admitted
            return leads, {"rhs": rhs, "leads": [
                {"term": lead.term, "source": lead.source,
                 "kind": lead.kind} for lead in leads],
                "constraint_barred": barred, "constraints": self.constraints,
                "fits_reported": fits, "interpretation": "nomination, not adoption"}
        if action == "tree_candidate":
            from mboostrs.policy import _incumbent_tree_term

            rhs = options.get("rhs", self.spec)
            term = (None if type(self.family).__name__ == "Multinomial" else
                    _incumbent_tree_term(self._discovery_data(), self.response, rhs))
            barred = None
            if term is not None and forbidden_hits([term], self.forbidden_pairs):
                _, pair = forbidden_hits([term], self.forbidden_pairs)[0]
                barred = f"forbidden pair {pair_label(pair)}"
                term = None
            return term, {"rhs": rhs, "term": term, "constraint_barred": barred,
                          "constraints": self.constraints,
                          "interpretation": "nomination, not adoption"}
        if action == "partial_effects":
            which = options["which"]
            values = self.model.partial_effects(which=which,
                                                newdata=options.get("newdata"))
            return values, {"which": which, "scale": "component link scale",
                            "interpretation": "fitted effect, not a causal contrast"}
        if action == "causal_traps":
            from mboostrs.causal import causal_trap_report

            report = causal_trap_report(self._data, self.causal)
            return report, {"blocking": [str(f) for f in report.blocking],
                            "advisory": [str(f) for f in report.advisory],
                            "interpretation": "design check, not effect identification"}
        if action == "freeze":
            self.frozen = True
            return self.model, {"spec": self.spec, "mstop": self.mstop,
                                "test_status": "not_evaluated"}
        if action == "final_evaluate":
            from mboostrs import _rust

            data = options["data"]
            if self.response not in column_names(data):
                raise ValueError("test data must contain the response column")
            y = np.asarray(read_column(data, self.response).values, dtype=float)
            weights = (np.ones(len(y), dtype=float) if options.get("weights") is None
                       else np.asarray(options["weights"], dtype=float))
            if self.score == "family":
                from mboostrs.families import _family_params, _family_rust_name

                family_name = type(self.family).__name__
                if hasattr(self.family, "param_names"):
                    raise NotImplementedError(
                        "family scoring of a distributional (LSS) holdout is not "
                        "supported; final_evaluate needs a response-scale score")
                if family_name == "Multinomial":
                    raise NotImplementedError(
                        "family scoring of a multinomial holdout is not supported; "
                        "use score='log_loss' or 'brier'")
                eta = np.asarray(self.model.predict(
                    data, type="link", unseen_levels=self.unseen_levels),
                    dtype=float).ravel()
                nuisance = self.model.nuisance()
                nuisance_scalar, nuisance_vector = None, None
                if nuisance is not None:
                    values = np.ravel(np.asarray(nuisance, dtype=float))
                    if values.size > 1:
                        nuisance_vector = values.tolist()
                    elif values.size == 1:
                        nuisance_scalar = float(values[0])
                score = _rust.family_held_out_score(
                    _family_rust_name(self.family), _family_params(self.family),
                    np.ascontiguousarray(y, dtype=float),
                    np.ascontiguousarray(eta, dtype=float),
                    np.ascontiguousarray(weights, dtype=float),
                    nuisance_scalar, nuisance_vector)
                self.evaluated = True
                return score, {"score": score, "score_name": "family",
                               "n_test": len(y),
                               "task": "one-time untouched-test evaluation",
                               "interpretation": ("weighted mean family loss, the "
                                                  "quantity cvrisk reports for a fold")}
            predictions = self.model.predict(data, type="response",
                                             unseen_levels=self.unseen_levels)
            values = np.asarray(predictions, dtype=float).reshape(len(y), -1)
            score, _ = _rust.response_score(y, values, weights, self.score,
                                            self.score_parameter)
            self.evaluated = True
            return score, {"score": score, "score_name": self.score,
                           "n_test": len(y), "task": "one-time untouched-test evaluation"}
        raise ValueError(f"unsupported action {action!r}")

    def _discovery_data(self):
        if not self._forbidden:
            return self._data
        return {name: self._data[name] for name in column_names(self._data)
                if name not in self._forbidden}


def open_search(data, response, **kwargs):
    """Open a training-only session for human or external-agent tool choice."""
    return SearchSession(data, response, **kwargs)
