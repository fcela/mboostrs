"""``specification_curve``: the specifications a search touched, scored on one fold partition.

A specification search compares right-hand sides and families and then keeps
one. The record of that search says which candidates were compared, but not
how each of them scores against the selected one on the same folds. This
module collects the candidates a source records, fits every one of them on one
shared fold partition, and reports each one's out-of-fold score and its paired
gap to the selected specification, with the standard error of that gap.

Sources
-------
``SearchSession``
    Candidates are the right-hand sides of its ``set_spec`` and ``adopt``
    actions, the rows of its ``compare`` actions, and the ``candidate_rhs``
    leads of its screening actions. Families come from ``set_family``. The
    partition is the session's own fold scheme.
exported transcript (``SearchSession.export_transcript()``)
    The same candidates, read from the record. The record holds no training
    rows, so the transcript must be passed with the training data it was made
    from; the data are checked against its fingerprint. The partition is
    rebuilt from the recorded fold count and seed and checked against the
    recorded fold fingerprint. A family is rebuilt from its class name with
    default arguments, and refused unless its fingerprint matches the recorded
    one.
``autospec`` recommendation
    The recommended formula, its shape probes (each column's incumbent term and
    the alternatives it was scored against), the rejected interactions, the
    dropped terms, and the family alternative the family probe measured.
    No partition is recorded, so one 5-fold partition is built from ``seed``.
``mb.discover`` result
    The draft, every leave-one-out set of the backward elimination, the
    interaction candidates and the log-scale candidates, replayed from the
    result's decision log. The partition is the one ``discover`` used (its
    fold count and seed), built on the rows with every predictor present.

What is scored
--------------
Every candidate is fitted with :func:`~mboostrs.compare_specifications` at the
source's stopping iteration and step length. The stopping iteration is not
tuned per candidate, so a candidate's score is its score at the step count the
search settled on. Rows missing a value in a column that any candidate uses
are removed before any fit, and the number removed is reported, so every
candidate is scored on the same rows.

A curve whose candidates use more than one family cannot use a family's own
loss, because the losses are in different units. Such a curve is scored by
the mean squared error on the response scale, and the result says so.

Declared and raised choices
---------------------------
Each candidate records whether it was written down before any observed result
(a screen, a comparison, a probe) or was raised by one. Under a
``SearchSession`` a candidate is declared when it appears before the first
measured or screening row of the transcript. Under ``discover`` the draft and
its first leave-one-out set are declared; the later elimination rounds, the
interaction candidates and the log-scale candidates are raised. Under
``autospec`` the default shapes and the family are declared when no family
probe ran; the probes' alternatives are raised. A curve with any raised
candidate is labelled post hoc. Each row carries an origin for its family,
its shape (the right-hand side) and its transform terms, and the fold scheme,
which is always declared.

A declared candidate stays declared when the search later selected it. The
selection is the observed result the joint test asks about, so the curve does
not relabel a declared candidate as raised because it won. A search that picks
one of several declared candidates therefore shows a post hoc selection on a
declared curve, and the ``post hoc`` label marks only the candidates that were
raised.

Joint test
----------
``joint_test`` asks whether the selected specification beats the curve median
by more than the signs of the paired fold differences allow. On each fold the
difference is the median score over all rows minus the selected row's score.
The statistic is the mean difference, and the p-value is the one-sided share
of random sign flips of the paired differences (exhaustive when there are at
most 16 flip units, otherwise 10,000 seeded draws). With stacked partitions the
flip unit is a partition. The selected folds are reused: the curve and the
selection were both computed on them. The test is descriptive. It describes
this curve on these folds and is not a confirmatory test for new data.

Example
-------
>>> import numpy as np, polars as pl, mboostrs as mb
>>> rng = np.random.default_rng(1)
>>> x = rng.uniform(-2, 2, 200)
>>> data = pl.DataFrame({"x": x, "z": rng.normal(size=200),
...                      "y": np.sin(x) + rng.normal(0, 0.3, 200)})
>>> session = mb.SearchSession(data, "y", family=mb.Gaussian(), score="mse", folds=5)
>>> session.run("set_spec", rhs="bbs(x, df=4)")  # doctest: +SKIP
>>> session.run("compare", candidates={"add z": "bbs(x, df=4) + bols(z)"})  # doctest: +SKIP
>>> curve = mb.specification_curve(session, data, "y")  # doctest: +SKIP
>>> curve.records[0]["specification"]  # doctest: +SKIP
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from mboostrs._names import partition_tilde
from mboostrs._output import json_safe, output_frame
from mboostrs._validate import check_fit_budget, check_int_at_least
from mboostrs.cost import estimate_cost
from mboostrs.explore import _paired_se, _partitions, compare_specifications, split_terms
from mboostrs.formula import term_variables
from mboostrs.policy import _data_fingerprint, _fingerprint, _policy_folds

__all__ = ["SpecificationCurve", "specification_curve"]

#: Evaluation statuses that count as an observed result.
_OBSERVED = ("measured", "screening_only")

#: A term that reads its column through a transform call.
_TRANSFORM = re.compile(r"\b(?:log1p|log|sqrt)\(")

#: Sign patterns drawn when the flip units are too many to enumerate.
_SIGN_DRAWS = 10_000

_OUTPUTS = ("auto", "polars", "pandas", "records")


def _family_key(family):
    """A string that is equal for two families with the same class and parameters.

    `_fingerprint` cannot hash every family (a distributional family has no
    cache form), so those fall back to the class name and attributes.
    """
    if family is None:
        return "None"
    try:
        return _fingerprint(family)
    except TypeError:
        attributes = json.dumps(sorted((k, repr(v)) for k, v in vars(family).items()))
        return f"{type(family).__module__}.{type(family).__qualname__}:{attributes}"


@dataclass
class _Candidate:
    """One right-hand side under one family, as the source wrote it down."""

    family: object
    rhs: object
    #: Written down before any observed result.
    declared: bool

    def key(self):
        return (_family_key(self.family), json.dumps(self.rhs, sort_keys=True))

    def terms(self):
        parts = self.rhs.values() if isinstance(self.rhs, Mapping) else (self.rhs,)
        return [term for part in parts for term in split_terms(part)]

    def label(self):
        if isinstance(self.rhs, Mapping):
            return "; ".join(f"{name}: {part}" for name, part in self.rhs.items())
        return self.rhs


@dataclass
class _Plan:
    """What a source contributes before any fit: candidates and the settings."""

    kind: str
    response: str
    candidates: list
    selected: _Candidate
    mstop: int
    nu: float
    unseen_levels: str = "average"
    score: str | None = None
    score_parameter: object = None
    weights: object = None
    #: A fold scheme over the full data, or None to build one from the count.
    folds: object = None
    fold_count: int | None = None
    groups: tuple = ()
    seed: int = 0
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class SpecificationCurve:
    """The candidate specifications of a search, scored on one fold partition.

    ``records`` holds one dict per specification, sorted by out-of-fold score
    (lowest first). Its keys are ``specification`` (a name), ``role``
    (``"selected"`` or ``"candidate"``), ``family``, ``rhs`` (the right-hand
    side; ``"mu: ...; sigma: ..."`` for a distributional family),
    ``transform`` (the terms that read a transform, separated by ``"; "``),
    ``family origin``, ``shape origin``, ``transform origin`` and
    ``fold scheme origin`` (each ``"declared"``, ``"raised"`` or ``"none"``),
    ``origin`` (``"declared"`` only when every choice is declared), ``fold
    scheme``, ``out-of-fold score`` (mean loss over the folds; lower is
    better), ``gap vs selected`` (this score minus the selected one's), ``gap
    se`` (the paired fold standard error of that gap), ``within two se``
    (whether the gap is at most two of those standard errors; a gap below zero,
    a candidate that scores better, is within by that rule) and ``rank``
    (position by score, from 1). The selected row has no standard error.
    """

    source: str
    response: str
    score: str
    score_parameter: object
    mstop: int
    nu: float
    fold_scheme: str
    partitions: int
    folds_per_partition: int
    selected: str
    records: tuple
    joint_test: dict
    post_hoc: bool
    label: str
    fits: int
    estimated_seconds: float
    seconds: float
    budget: int | None
    excluded_rows: int
    notes: tuple
    output: str = "auto"

    @property
    def rows(self):
        """The records as a frame, or a list of dicts for ``output="records"``.

        ``output="auto"`` gives polars, ``"polars"`` a polars frame and
        ``"pandas"`` a pandas frame.
        """
        return output_frame([dict(row) for row in self.records], self.output)

    def to_dict(self, *, include_arrays=False):
        """The whole result as plain Python types, for JSON.

        ``include_arrays`` is accepted for the package's export convention. A
        curve holds no arrays, so the payload is the same either way.
        """
        return json_safe({
            "source": self.source, "response": self.response, "score": self.score,
            "score_parameter": self.score_parameter, "mstop": self.mstop, "nu": self.nu,
            "fold_scheme": self.fold_scheme, "partitions": self.partitions,
            "folds_per_partition": self.folds_per_partition,
            "selected": self.selected, "rows": [dict(row) for row in self.records],
            "joint_test": self.joint_test, "post_hoc": self.post_hoc, "label": self.label,
            "fits": self.fits, "estimated_seconds": self.estimated_seconds,
            "seconds": self.seconds, "budget": self.budget,
            "excluded_rows": self.excluded_rows, "notes": list(self.notes),
        })

    def plot(self):
        """The curve as an Altair chart: each gap with its +/- 2 paired SE interval."""
        from mboostrs.plot import plot_specification_curve

        return plot_specification_curve(self)


def specification_curve(source, data, response, *, budget=None, seed=0, output="auto"):
    """Score the specifications a search touched on one shared fold partition.

    Parameters
    ----------
    source : SearchSession, dict, SpecRecommendation or DiscoveryResult
        The record to read the candidates from: a live
        :class:`~mboostrs.SearchSession`, the dict from its
        ``export_transcript()``, the recommendation returned by
        :func:`~mboostrs.autospec`, or the result of :func:`~mboostrs.discover`.
        See the module notes for what each source contributes.
    data : polars or pandas DataFrame, or a mapping of column to values
        The training rows the source was built on. A session or transcript
        refuses data whose fingerprint differs from the one it recorded.
    response : str
        Name of the response column.
    budget : int, optional
        Largest number of model fits the curve may perform. The count is
        taken from :func:`~mboostrs.estimate_cost` before any specification is
        scored, and a curve over the budget is refused with the count it would
        need. ``None`` means no limit.
    seed : int, default 0
        Seed of the fold partition for a source that records none (an
        ``autospec`` recommendation), and of the sign-flip draws when they are
        sampled.
    output : {"auto", "polars", "pandas", "records"}, default "auto"
        The type of ``SpecificationCurve.rows``. ``"auto"`` gives a polars
        frame when polars is installed, which it always is.

    Returns
    -------
    SpecificationCurve
        One record per specification, the joint test, the post hoc label and
        the cost. ``to_dict()`` gives plain types and ``plot()`` an Altair chart.

    Raises
    ------
    ValueError
        When the data do not match the source, the source has no selected
        specification, its record lacks what is needed to rebuild its folds, or
        the curve needs more fits than ``budget``.
    TypeError
        When ``source`` is none of the four kinds above.

    Notes
    -----
    The fold partition is shared by every candidate, so the paired gaps are
    paired fold by fold. ``gap se`` is the standard error that
    :func:`~mboostrs.compare_specifications` reports as ``change se``, so the
    same caution applies: it understates the variation across data sets.
    Candidates are not tuned: a candidate scores at the source's stopping
    iteration. Response transforms (``log`` or ``log1p`` of the response) are
    not in the curve, because they change the target that every candidate is
    scored on; :meth:`~mboostrs.SearchSession.run` with ``response_probe``
    measures them separately.

    Examples
    --------
    Scoring the candidates of an ``autospec`` recommendation:

    >>> import numpy as np, polars as pl, mboostrs as mb
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 200)
    >>> data = pl.DataFrame({"x": x, "junk": rng.normal(size=200),
    ...                      "y": np.sin(x) + rng.normal(0, 0.3, 200)})
    >>> rec = mb.autospec(data, "y", budget=150)
    >>> curve = mb.specification_curve(rec, data, "y", budget=200)
    >>> sum(row["role"] == "selected" for row in curve.records)
    1
    """
    started = time.perf_counter()
    budget = check_int_at_least("budget", budget, 1, allow_none=True)
    seed = check_int_at_least("seed", seed, 0)
    if output not in _OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(_OUTPUTS)}, not {output!r}")
    plan = _read_source(source, data, response, seed)
    frame = _as_polars(data)
    candidates, selected_index = _unique_candidates(plan)
    variables = {response}
    for candidate in candidates:
        for part in (candidate.rhs.values() if isinstance(candidate.rhs, Mapping)
                     else (candidate.rhs,)):
            variables.update(term_variables(part))
    frame, weights, folds, excluded = _common_rows(frame, plan, variables, response)
    score, parameter = _choose_score(plan, candidates)
    if plan.score is not None and score != plan.score:
        plan.notes.append(
            f"the curve spans {len({c.key()[0] for c in candidates})} families whose "
            f"losses are in different units, so every specification is scored by the "
            f"mean squared error on the response scale instead of {plan.score!r}")

    groups = _family_groups(candidates)
    fits, estimated = _check_budget(plan, groups, candidates, frame, response, folds,
                                    weights, score, parameter, budget, seed)

    fold_scores, risks = _score_groups(groups, candidates, frame, response, folds, weights,
                                       plan, score, parameter, budget, seed)
    partitions = _partitions(folds)
    if len({fold_scores[i].size for i in range(len(candidates))}) != 1:
        raise RuntimeError("the specifications were not scored on one set of folds")
    fold_matrix = np.vstack([fold_scores[i] for i in range(len(candidates))])
    records = _records(candidates, selected_index, risks, fold_matrix, partitions, folds)
    joint = _joint_test(fold_matrix, selected_index, partitions, seed, risks,
                        candidates)
    raised = sum(1 for row in records if row["origin"] == "raised")
    post_hoc = raised > 0
    label = (f"post hoc: {raised} of {len(records)} specifications were raised by observed "
             "results" if post_hoc else
             f"declared before scoring: all {len(records)} specifications were written "
             "down before any observed result")
    return SpecificationCurve(
        source=plan.kind, response=response, score=score, score_parameter=parameter,
        mstop=plan.mstop, nu=plan.nu, fold_scheme=str(folds.scheme),
        partitions=partitions, folds_per_partition=fold_matrix.shape[1] // partitions,
        selected=next(row["specification"] for row in records if row["role"] == "selected"),
        records=tuple(records), joint_test=joint, post_hoc=post_hoc, label=label,
        fits=fits, estimated_seconds=estimated,
        seconds=time.perf_counter() - started, budget=budget, excluded_rows=excluded,
        notes=tuple(plan.notes), output=output)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _read_source(source, data, response, seed):
    from mboostrs._autospec import SpecRecommendation
    from mboostrs.discover import DiscoveryResult
    from mboostrs.search_session import SearchSession

    if isinstance(source, SearchSession):
        return _read_session(source, data, response)
    if isinstance(source, Mapping) and source.get("kind") == "search_transcript":
        return _read_transcript(source, data, response)
    if isinstance(source, SpecRecommendation):
        return _read_autospec(source, data, response, seed)
    if isinstance(source, DiscoveryResult):
        return _read_discover(source, data, response)
    raise TypeError(
        "source must be a SearchSession, the dict from its export_transcript(), an "
        "autospec recommendation or a mb.discover result, not "
        f"{type(source).__name__}")


def _read_session(session, data, response):
    if response != session.response:
        raise ValueError(f"the session searched response {session.response!r}, not {response!r}")
    if _data_fingerprint(data) != session.data_id:
        raise ValueError(
            "the data differ from the rows the session searched; pass the training frame "
            "the session was opened on")
    if session.spec is None:
        raise ValueError(
            "the session has no selected specification; set one with set_spec, or adopt a "
            "candidate, before calling specification_curve")
    rows = [(evidence.action, evidence.status, evidence.summary,
             evidence.value if evidence.action == "set_family" else None)
            for evidence in session.evidence]
    candidates = _walk(rows, session._opening_family)
    notes = []
    if session.evidence and any(e.action == "autospec" for e in session.evidence):
        notes.append("autospec actions run inside the session are not enumerated; pass the "
                     "autospec recommendation as the source to score its candidates")
    return _Plan(
        kind="search_session", response=response, candidates=candidates,
        selected=_Candidate(session.family, session.spec, declared=False),
        mstop=int(session.mstop), nu=float(session.nu),
        unseen_levels=session.unseen_levels, score=session.score,
        score_parameter=session.score_parameter, weights=session.weights,
        folds=session.folds, groups=tuple(session.groups), seed=int(session.seed),
        notes=notes)


def _read_transcript(record, data, response):
    if record.get("version") != 1:
        raise ValueError("unsupported search transcript version")
    if record["response"] != response:
        raise ValueError(f"the transcript searched response {record['response']!r}, "
                         f"not {response!r}")
    if _data_fingerprint(data) != record["data_id"]:
        raise ValueError("the data differ from the rows the transcript was made from; pass "
                         "the training frame it was made from")
    if record.get("spec") is None:
        raise ValueError("the transcript has no selected specification")
    if "fold_count" not in record or "seed" not in record or "family_id" not in record:
        raise ValueError(
            "this transcript predates the fields that rebuild its fold partition and family "
            "(fold_count, seed, family_id); export it again from the SearchSession")
    if record.get("weighted"):
        raise ValueError("a weighted transcript does not record its weights; pass the "
                         "SearchSession, or a checkpoint with its weights")
    if record["fold_count"] is None:
        raise ValueError("the session was opened with a fold scheme object, which a "
                         "transcript cannot rebuild; pass the SearchSession")
    rows = []
    for row in record["evidence"]:
        summary = row["summary"]
        family = (_family_from_record(summary.get("family"), summary.get("family_id"))
                  if row["action"] == "set_family" else None)
        rows.append((row["action"], row["status"], summary, family))
    opening = _family_from_record(record.get("opening_family"),
                                  record.get("opening_family_id"))
    candidates = _walk(rows, opening)
    folds = _policy_folds(data, response, int(record["fold_count"]),
                          tuple(record["groups"]), int(record["seed"]), None)
    if _fingerprint(folds) != record["fold_id"]:
        raise ValueError("the partition rebuilt from the transcript's fold count and seed "
                         "does not match its fold fingerprint; pass the SearchSession")
    selected_family = _family_from_record(record.get("family"), record.get("family_id"))
    return _Plan(
        kind="search_transcript", response=response, candidates=candidates,
        selected=_Candidate(selected_family, record["spec"], declared=False),
        mstop=int(record["mstop"]), nu=float(record["nu"]),
        unseen_levels=record["unseen_levels"], score=record["score"],
        score_parameter=record["score_parameter"], folds=folds,
        groups=tuple(record["groups"]), seed=int(record["seed"]),
        notes=["families are rebuilt from the transcript's class names with default "
               "arguments; the fingerprint check confirms each one"])


def _read_autospec(rec, data, response, seed):
    from mboostrs.families import GammaReg, Gaussian

    if getattr(rec.family, "param_names", None) is not None:
        raise ValueError(
            "specification_curve does not score a distributional autospec recommendation "
            "yet; pass the SearchSession or the discover result that produced it")
    evidence = rec.evidence
    selected_terms = split_terms(partition_tilde(rec.formula)[2])
    probed = "family" in evidence
    family = rec.family
    candidates = []

    shape_rows = tuple(evidence.get("shapes", ()))
    chosen_terms = {}
    incumbent_terms = {}
    for row in shape_rows:
        scores = {s.shape: s.term for s in row.scores}
        chosen_terms[row.column] = scores.get(row.chosen)
        incumbent_terms[row.column] = scores.get(row.incumbent)

    def replace(terms, old, new):
        return [new if term == old else term for term in terms]

    if not probed:
        default_terms = list(selected_terms)
        for column in chosen_terms:
            if chosen_terms[column] and incumbent_terms[column]:
                default_terms = replace(default_terms, chosen_terms[column],
                                        incumbent_terms[column])
        candidates.append(_Candidate(family, _join(default_terms), declared=True))
    for row in shape_rows:
        for score in row.scores:
            chosen = chosen_terms.get(row.column)
            if chosen is None or score.term == chosen or chosen not in selected_terms:
                continue
            candidates.append(_Candidate(
                family, _join(replace(selected_terms, chosen, score.term)), declared=False))
    deselect = evidence.get("deselect")
    if deselect is not None and getattr(deselect, "dropped", ()):
        dropped = {term for term, _share in deselect.dropped}
        candidates.append(_Candidate(
            family, _join([t for t in selected_terms if t not in dropped]), declared=False))
    for rejected in evidence.get("rejected_interactions", ()):
        term = rejected[0]
        candidates.append(_Candidate(family, _join(selected_terms + [term]), declared=False))
    family_row = evidence.get("family")
    if family_row is not None and np.isfinite(getattr(family_row, "gamma_mse", np.nan)):
        other = (Gaussian() if isinstance(family, GammaReg) else
                 GammaReg() if isinstance(family, Gaussian) else None)
        if other is not None:
            candidates.append(_Candidate(other, _join(selected_terms), declared=False))
    notes = [f"the recommendation records no fold partition: the curve builds one 5-fold "
             f"partition from seed={seed}"]
    if probed:
        notes.append("the family probe ran, so no default specification is declared: the "
                     "incumbent family is not recorded")
    return _Plan(
        kind="autospec", response=response, candidates=candidates,
        selected=_Candidate(family, _join(selected_terms), declared=False),
        mstop=int(rec.mstop), nu=float(getattr(rec.control, "nu", 0.1) or 0.1),
        fold_count=5, seed=int(seed), notes=notes)


def _read_discover(result, data, response):
    from mboostrs.discover import NU
    from mboostrs.families import Gaussian
    from mboostrs.gamboost import GAMBoostModel
    from mboostrs.gamlss import GaussianLSS

    if not result.log:
        raise ValueError("the discover result has no decision log to replay")
    mean_family = (result.model.family if isinstance(result.model, GAMBoostModel)
                   else Gaussian())
    log = result.log
    draft = next(entry["candidate"] for entry in log if entry["stage"] == "draft")
    current = split_terms(draft)
    candidates = [_Candidate(mean_family, draft, declared=True)]

    rounds = 0

    def emit_round(terms):
        nonlocal rounds
        declared = rounds == 0
        rounds += 1
        candidates.append(_Candidate(mean_family, _join(terms), declared=declared))
        for index in range(len(terms)):
            candidates.append(_Candidate(
                mean_family, _join(terms[:index] + terms[index + 1:]), declared=declared))

    kept_emitted = False
    for entry in log:
        if entry["stage"] != "mean effects" or entry["candidate"] in (None, "1"):
            continue
        if entry["adopted"]:
            if not kept_emitted:
                emit_round(current)
                kept_emitted = True
            continue
        emit_round(current)
        current.remove(entry["candidate"])

    # A comparison entry says "adopted:" or "rejected:" in its reason. Other
    # entries that name a pair (not nominated, failed) were never scored.
    base = list(current)
    for entry in log:
        if entry["stage"] != "interactions" or entry["candidate"] is None:
            continue
        if not entry["reason"].startswith(("adopted:", "rejected:")):
            continue
        term = entry["candidate"]
        candidates.append(_Candidate(mean_family, _join(base + [term]), declared=False))
        if entry["adopted"]:
            base.append(term)
    mu = _join(base)
    lss_family = GaussianLSS()
    spread_entries = [e for e in log if e["stage"] == "spread" and e["candidate"]
                      and e["candidate"].startswith("sigma: ")]
    mandatory = [e for e in spread_entries if e["reason"].startswith("adopted: mandatory")]
    compared = [e["candidate"][len("sigma: "):] for e in spread_entries
                if e["reason"].startswith(("adopted:", "rejected:"))]
    if mandatory:
        # A mandatory log-scale term is included without a comparison, so the
        # only distributional candidate is the declared one.
        sigma = mandatory[0]["candidate"][len("sigma: "):]
        candidates.append(_Candidate(lss_family, {"mu": mu, "sigma": sigma}, declared=True))
    elif compared:
        candidates.append(_Candidate(lss_family, {"mu": mu, "sigma": "1"}, declared=False))
        for sigma in compared:
            candidates.append(_Candidate(lss_family, {"mu": mu, "sigma": sigma},
                                         declared=False))
    if result.spread_formula is None:
        selected = _Candidate(mean_family, result.formula, declared=False)
    else:
        selected = _Candidate(lss_family, {"mu": result.formula,
                                           "sigma": result.spread_formula}, declared=False)
    return _Plan(
        kind="discover", response=response, candidates=candidates, selected=selected,
        mstop=int(result.mstop), nu=float(NU), folds=None, fold_count=int(result.folds),
        seed=int(result.seed),
        notes=["the partition is the discover run's: its fold count and seed, on the rows "
               "with every column the candidates use"])


def _walk(rows, opening_family):
    """Candidates in the order the search wrote them down.

    ``rows`` holds ``(action, status, summary, family)`` per evidence row, with
    the family already resolved for a ``set_family`` row. A candidate is declared
    when no observed result precedes the row that wrote it down; a candidate a
    screen proposes is raised, because the screen is itself an observed result.
    """
    candidates = []
    family = opening_family
    observed = False
    for action, status, summary, value in rows:
        before = observed
        if action == "set_family":
            family = value
        elif action == "set_spec":
            candidates.append(_Candidate(family, summary["spec"], declared=not before))
        elif action == "compare" and status == "measured":
            for row in summary.get("rows", ()):
                candidates.append(_Candidate(family, row["rhs"], declared=not before))
        elif action == "adopt":
            candidates.append(_Candidate(family, summary["spec"], declared=not before))
        if status == "screening_only":
            for lead in summary.get("leads", ()):
                if "candidate_rhs" in lead:
                    candidates.append(_Candidate(family, lead["candidate_rhs"], declared=False))
        if status in _OBSERVED:
            observed = True
    return candidates


def _family_from_record(name, family_id):
    """A family rebuilt from its class name, accepted only if its fingerprint matches."""
    if name is None:
        return None
    from mboostrs import families, gamlss

    cls = getattr(families, name, None) or getattr(gamlss, name, None)
    if not isinstance(cls, type):
        raise ValueError(f"the transcript names family {name!r}, which specification_curve "
                         "cannot construct; pass the SearchSession")
    try:
        family = cls()
    except TypeError as exc:
        raise ValueError(f"family {name} needs arguments the transcript does not record; "
                         "pass the SearchSession") from exc
    if _fingerprint(family) != family_id:
        raise ValueError(f"family {name} was set with arguments the transcript does not "
                         "record; pass the SearchSession")
    return family


# ---------------------------------------------------------------------------
# Rows, folds and scoring
# ---------------------------------------------------------------------------


def _join(terms):
    return " + ".join(terms) if terms else "1"


def _as_polars(data):
    from mboostrs.discover import _as_frame

    return _as_frame(data)


def _is_distributional(family):
    return hasattr(family, "param_names") and hasattr(family, "pdf")


def _harmonize(plan):
    """Score a curve that mixes a distributional family with a Gaussian mean on one joint score.

    `compare_specifications` scores a distributional family only by its joint
    proper score, so a Gaussian mean candidate cannot share a curve with a
    `GaussianLSS` candidate on the mse. Each such candidate is refitted as
    `GaussianLSS` with a constant scale (``sigma: 1``), the comparator
    `discover` itself uses for its log-scale stage. Any other family cannot be
    placed on that score, and the curve refuses it.
    """
    from mboostrs.gamlss import GaussianLSS

    everything = [*plan.candidates, plan.selected]
    if not any(_is_distributional(c.family) for c in everything):
        return
    converted = []
    for candidate in everything:
        if _is_distributional(candidate.family):
            converted.append(candidate)
            continue
        if type(candidate.family).__name__ != "Gaussian":
            raise ValueError(
                f"the curve mixes a distributional family with {type(candidate.family).__name__}; "
                "a distributional curve can hold only Gaussian mean candidates, so drop the "
                f"{type(candidate.family).__name__} candidates from the source")
        converted.append(_Candidate(GaussianLSS(), {"mu": candidate.rhs, "sigma": "1"},
                                    candidate.declared))
    plan.candidates = converted[:-1]
    plan.selected = converted[-1]
    plan.notes.append(
        "the curve mixes a distributional family with Gaussian mean candidates, so each "
        "Gaussian candidate is refitted as GaussianLSS with a constant scale and every "
        "specification is scored by the joint score of GaussianLSS")


def _unique_candidates(plan):
    """Candidates without repeats, first occurrence kept, and the selected index."""
    _harmonize(plan)
    seen = {}
    unique = []
    for candidate in plan.candidates:
        key = candidate.key()
        if key not in seen:
            seen[key] = len(unique)
            unique.append(candidate)
    selected_key = plan.selected.key()
    if selected_key not in seen:
        seen[selected_key] = len(unique)
        unique.append(_Candidate(plan.selected.family, plan.selected.rhs, declared=False))
    return unique, seen[selected_key]


def _common_rows(frame, plan, variables, response):
    """Rows complete in every column a candidate uses, with the folds and weights cut to them."""
    columns = [name for name in frame.columns if name in variables]
    missing = variables - set(columns)
    if missing:
        raise ValueError(f"the data have no column(s) {sorted(missing)} that the candidates use")
    mask = frame.select(pl.all_horizontal(
        [pl.col(name).is_not_null() for name in columns])).to_series()
    keep = mask.to_numpy()
    excluded = int(frame.height - keep.sum())
    if excluded:
        frame = frame.filter(mask)
    weights = plan.weights
    if weights is not None and np.ndim(weights) > 0:
        weights = np.asarray(weights, dtype=float)[keep]
    if plan.folds is not None:
        from mboostrs.autospec import _scheme_rows

        folds = _scheme_rows(plan.folds, np.flatnonzero(keep)) if excluded else plan.folds
    else:
        folds = _policy_folds(frame, response, plan.fold_count, plan.groups, plan.seed,
                              weights)
    return frame, weights, folds, excluded


def _choose_score(plan, candidates):
    families = {candidate.key()[0] for candidate in candidates}
    if plan.score is None:
        return ("family" if len(families) == 1 else "mse"), None
    if len(families) > 1 and plan.score == "family":
        return "mse", None
    parameter = plan.score_parameter if plan.score in ("pinball", "expectile") else None
    return plan.score, parameter


def _family_groups(candidates):
    """Candidate indices by family, in first-appearance order."""
    groups = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(candidate.key()[0], []).append(index)
    return groups


def _group_call(indices, candidates, plan, folds, weights, score, parameter, seed):
    """The specifications of one family and the keyword arguments they share."""
    specifications = {f"c{index}": candidates[index].rhs for index in indices}
    kwargs = dict(family=candidates[indices[0]].family, folds=folds, mstop=plan.mstop,
                  nu=plan.nu, seed=seed, weights=weights, score=score,
                  score_parameter=parameter, na_action="error",
                  unseen_levels=plan.unseen_levels)
    return specifications, kwargs


def _check_budget(plan, groups, candidates, frame, response, folds, weights, score,
                  parameter, budget, seed):
    """Fit count and seconds of the whole curve, read from estimate_cost before any scoring."""
    fits = 0
    estimated = 0.0
    for indices in groups.values():
        specifications, kwargs = _group_call(indices, candidates, plan, folds, weights,
                                             score, parameter, seed)
        cost = estimate_cost("compare_specifications", specifications, frame, response,
                             **kwargs)
        fits += cost.fits
        estimated += cost.estimated_seconds
    check_fit_budget(
        fits, budget, "specification_curve",
        detail=f" for {len(candidates)} specifications in {len(groups)} famil"
               f"{'y' if len(groups) == 1 else 'ies'}",
        remedy="raise budget to at least that count, or use a source with fewer specifications")
    return fits, estimated


def _score_groups(groups, candidates, frame, response, folds, weights, plan, score,
                  parameter, budget, seed):
    """Fold scores and out-of-fold risks, by candidate index.

    One compare_specifications call per family, all on the same folds. Each
    row's evaluation keeps the fold scores that the paired gaps are computed from.
    """
    fold_scores, risks = {}, {}
    for indices in groups.values():
        specifications, kwargs = _group_call(indices, candidates, plan, folds, weights,
                                             score, parameter, seed)
        rows = compare_specifications(specifications, frame, response,
                                      include_evaluations=True, fit_budget=budget, **kwargs)
        for row in rows:
            index = int(row["specification"][1:])
            fold_scores[index] = np.asarray(row["evaluation"].fold_scores, dtype=float)
            risks[index] = float(row["out-of-fold risk"])
    return fold_scores, risks


def _records(candidates, selected_index, risks, fold_matrix, partitions, folds):
    order = sorted(range(len(candidates)), key=lambda i: (risks[i], i))
    family_declared = {}
    term_declared = {}
    for candidate in candidates:
        family_declared.setdefault(candidate.key()[0], candidate.declared)
        for term in candidate.terms():
            term_declared.setdefault(term, candidate.declared)
    selected_scores = fold_matrix[selected_index]
    records = []
    for rank, index in enumerate(order, start=1):
        candidate = candidates[index]
        transforms = [term for term in candidate.terms() if _TRANSFORM.search(term)]
        if transforms:
            transform_origin = ("declared" if all(term_declared[t] for t in transforms)
                                else "raised")
        else:
            transform_origin = "none"
        family_origin = "declared" if family_declared[candidate.key()[0]] else "raised"
        shape_origin = "declared" if candidate.declared else "raised"
        origin = ("declared" if family_origin == "declared" and shape_origin == "declared"
                  and transform_origin in ("declared", "none") else "raised")
        is_selected = index == selected_index
        if is_selected:
            gap, se, within = 0.0, None, None
        else:
            gap = risks[index] - risks[selected_index]
            se = float(_paired_se(fold_matrix[index], selected_scores, partitions))
            if not np.isfinite(se):
                se = None
            within = None if se is None else bool(gap <= 2 * se)
        records.append({
            "specification": f"s{index + 1:02d}",
            "role": "selected" if is_selected else "candidate",
            "family": type(candidate.family).__name__ if candidate.family is not None else "None",
            "rhs": candidate.label(),
            "transform": "; ".join(transforms),
            "family origin": family_origin,
            "shape origin": shape_origin,
            "transform origin": transform_origin,
            "fold scheme origin": "declared",
            "origin": origin,
            "out-of-fold score": risks[index],
            "gap vs selected": gap,
            "gap se": se,
            "within two se": within,
            "fold scheme": str(folds.scheme),
            "rank": rank,
        })
    return records


def _joint_test(fold_matrix, selected_index, partitions, seed, risks, candidates):
    from mboostrs import _rust

    median_risk = float(np.median([risks[i] for i in range(len(candidates))]))
    base = {"selected_risk": risks[selected_index], "curve_median_risk": median_risk,
            "alternative": "the selected specification's out-of-fold score is below the curve "
                           "median on the shared folds",
            "folds_reused": True, "confirmatory": False}
    if fold_matrix.shape[0] < 2:
        return {**base, "available": False, "statistic": None, "p_value": None,
                "units": None, "exact": None, "draws": None,
                "reason": "one specification: there is no curve to compare against"}
    test = _rust.specification_curve_sign_test(
        np.ascontiguousarray(fold_matrix, dtype=float), selected_index, partitions,
        _SIGN_DRAWS, int(seed))
    return {**base, "available": True, "statistic": float(test["statistic"]),
            "p_value": float(test["p_value"]), "units": int(test["units"]),
            "exact": bool(test["exact"]), "draws": int(test["draws"]), "reason": None}
