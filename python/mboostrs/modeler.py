"""The modeler loop: a measured specification search with optional judgment.

The loop is the modeler's workflow as a function: fit a discovering
specification (:func:`~mboostrs.autospec`), report what each term earns
(:func:`~mboostrs.term_report`), screen the residual for joint structure
(:func:`~mboostrs.screen_interactions`), propose changes
(:func:`~mboostrs.suggest_specifications`), and let paired fold evidence
decide adoption (:func:`~mboostrs.compare_specifications`, two paired
standard errors; a removal the folds cannot distinguish from no change
keeps the smaller specification), then set the final step count
(:func:`~mboostrs.tune_mstop`). Run with ``judgment=None`` it is fully
deterministic and makes no network request; it is the documented search
the agent benchmarks record.

Two seams change what the loop is without changing what decides:

* ``judgment=`` — a callable that receives a digest of the measured
  evidence each round and returns judgment calls a human modeler would
  make (keep, lock, drop, add, constraints, leakage flags, causal
  assumptions). :class:`LLMJudge` implements it against any
  OpenAI-compatible chat endpoint, NVIDIA NIM by default. The judge
  proposes; the two-standard-error rule still disposes. Judge failures
  are logged and the loop continues deterministically.

* ``causal=`` — a :class:`~mboostrs.CausalSpec`. Causes are not in the
  data: no fold, term gain or screened pair identifies an effect, so the
  causal lane fixes the design before the search starts. The treatment
  and declared post-treatment columns never enter the nuisance
  specification, the adjustment covariates become locked terms no edit
  may remove, every causal statement carries a provenance tag in an
  :class:`AssumptionLedger` (``modeler-stated``,
  ``llm-proposed-confirmed``, ``llm-proposed-unconfirmed``,
  ``unresolved``), and the loop closes with the cross-fitted
  :func:`~mboostrs.aipw` estimate of the declared estimand. A causal
  reading of a predictive quantity — importance, selection frequency,
  monotonicity — is refused; those rules are stated to the judge and
  enforced on its edits.

The search may spend measured evidence on any column; the locks decide
membership, evidence decides form. Every step, judgment and refusal is
recorded in the transcript with cumulative fit counts.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from mboostrs._frame import column_names
from mboostrs._names import partition_tilde, quote_name, quote_response
from mboostrs._validate import check_int_at_least
from mboostrs.autospec import autospec
from mboostrs.causal import CausalSpec, aipw, causal_trap_report
from mboostrs.explain import (
    _API_KEY_VARS,
    _BASE_URL_VARS,
    DEFAULT_LLM,
    _client_chat,
    _from_env,
    _post_chat,
    _reply_text,
)
from mboostrs.explore import (
    split_terms,
    suggest_specifications,
)
from mboostrs.folds import grouped_folds
from mboostrs.formula import parse_gamboost_formula, term_variables
from mboostrs.gamboost import gamboost
from mboostrs.terms import classify_columns

__all__ = [
    "AssumptionLedger",
    "Assumption",
    "CausalDraft",
    "LLMJudge",
    "ModelerResult",
    "SpecificationJudgment",
    "Transcript",
    "elicit_causal_spec",
    "modeler_loop",
]


@dataclass
class Transcript:
    """One recorded step per line of the search, in order."""

    steps: list = field(default_factory=list)

    def log(self, step, **details):
        self.steps.append({"step": step, **details})


_PROVENANCE = (
    "modeler-stated",
    "llm-proposed-confirmed",
    "llm-proposed-unconfirmed",
    "unresolved",
)


@dataclass
class Assumption:
    """One causal statement with its provenance, never inferred from data."""

    statement: str
    provenance: str
    source: str = ""
    round: int | None = None

    def __post_init__(self):
        if self.provenance not in _PROVENANCE:
            raise ValueError(f"provenance must be one of {_PROVENANCE}")
        if not self.statement.strip():
            raise ValueError("an assumption statement must be non-empty")


@dataclass
class AssumptionLedger:
    """The causal statements a run depends on, tagged by who vouched for them.

    An LLM-proposed statement is recorded ``llm-proposed-unconfirmed`` and
    stays that way until a person confirms it; the loop never upgrades
    provenance on its own. Unresolved items belong in the ledger as
    ``unresolved`` so outputs can say what was never settled.
    """

    entries: list = field(default_factory=list)

    def add(self, statement, provenance, *, source="", round=None):
        entry = Assumption(str(statement), provenance, source=source, round=round)
        self.entries.append(entry)
        return entry

    def confirm(self, statement):
        """Mark the first matching unconfirmed LLM proposal as confirmed."""
        for entry in self.entries:
            if entry.statement == statement and entry.provenance == "llm-proposed-unconfirmed":
                entry.provenance = "llm-proposed-confirmed"
                return entry
        raise KeyError(f"no unconfirmed proposal matches {statement!r}")

    def to_list(self):
        return [
            {"statement": e.statement, "provenance": e.provenance,
             "source": e.source, "round": e.round}
            for e in self.entries
        ]


#: The refusal rules stated to the judge and enforced on its edits. A causal
#: reading of a predictive quantity is wrong no matter how confident the
#: model sounds, so the rules are mechanical where they can be.
_REFUSAL_RULES = (
    "Predictive quantities (term gains, importance, selection frequency, "
    "monotonicity, screened pairs) are evidence about prediction, never about "
    "causes. Do not state or imply a causal role from them.",
    "Locked columns are fixed by the declared design. Edits that remove them "
    "are refused mechanically.",
    "The treatment and post-treatment columns never enter the outcome model.",
    "Every causal statement you make is recorded as llm-proposed-unconfirmed "
    "until a person confirms it; state it as a proposal, not a fact.",
)

_JUDGE_SCHEMA = {
    "reasoning": "one short paragraph",
    "drop": ["column or term names to remove"],
    "add": ["terms to add, e.g. 'bbs(x1) %X% bbs(x2)' or a column name"],
    "replace": {"existing term": "replacement term or sum of terms"},
    "lock": ["columns to keep regardless of measured gain"],
    "constraints": {"monotone": {"column": "increasing|decreasing"}},
    "flags": [{"kind": "leakage|support|causal", "term": "name", "note": "why"}],
    "assumptions": [{"statement": "a causal claim", "confidence": "low|medium|high"}],
    "estimand_note": "one sentence on the question being asked",
    "evidence_id": "copy the evidence_id of the digest being judged",
}


@dataclass(frozen=True)
class SpecificationJudgment:
    """Validated editable proposal; predictive evidence decides adoption.

    Replacements name existing terms exactly. Monotone proposals require an
    explicit direction; interactions involving that column must be resolved
    before a univariate constraint can describe its complete effect.
    """

    drop: tuple[str, ...] = ()
    add: tuple[str, ...] = ()
    lock: tuple[str, ...] = ()
    replace: dict = field(default_factory=dict)
    monotone: dict = field(default_factory=dict)
    flags: tuple = ()
    assumptions: tuple = ()
    reasoning: str = ""
    estimand_note: str = ""
    evidence_id: str | None = None

    def __post_init__(self):
        for key in ("drop", "add", "lock"):
            values = getattr(self, key)
            if not isinstance(values, (tuple, list)) or any(
                    not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{key} must contain non-empty strings")
        if not isinstance(self.replace, dict) or any(
                not isinstance(k, str) or not k.strip()
                or not isinstance(v, str) or not v.strip() for k, v in self.replace.items()):
            raise ValueError("replace must map existing terms to replacement terms")
        if not isinstance(self.monotone, dict) or any(
                not isinstance(k, str) or not k.strip() or v not in ("increasing", "decreasing")
                for k, v in self.monotone.items()):
            raise ValueError("monotone must map columns to increasing or decreasing")
        for key in ("flags", "assumptions"):
            values = getattr(self, key)
            if not isinstance(values, (tuple, list)) or any(not isinstance(v, dict) for v in values):
                raise ValueError(f"{key} must contain objects")
        if not isinstance(self.reasoning, str) or not isinstance(self.estimand_note, str):
            raise ValueError("reasoning and estimand_note must be strings")
        if self.evidence_id is not None and not isinstance(self.evidence_id, str):
            raise ValueError("evidence_id must be a string")

    def to_dict(self, *, include_arrays=False):
        """The editable JSON proposal accepted by the judgment interface."""
        proposal = {
            "drop": list(self.drop), "add": list(self.add), "lock": list(self.lock),
            "replace": dict(self.replace), "constraints": {"monotone": dict(self.monotone)},
            "flags": list(self.flags), "assumptions": list(self.assumptions),
            "reasoning": self.reasoning, "estimand_note": self.estimand_note,
        }
        if self.evidence_id is not None:
            proposal["evidence_id"] = self.evidence_id
        return proposal

    @classmethod
    def from_mapping(cls, proposal):
        if not isinstance(proposal, Mapping):
            raise ValueError("judgment must be an object")
        unknown = set(proposal) - set(_JUDGE_SCHEMA)
        if unknown:
            raise ValueError(f"unsupported judgment fields: {sorted(unknown)}")

        def names(key):
            values = proposal.get(key, [])
            if not isinstance(values, list) or any(
                    not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{key} must be a list of non-empty strings")
            return tuple(dict.fromkeys(v.strip() for v in values))

        replacements = proposal.get("replace", {})
        if not isinstance(replacements, dict) or any(
                not isinstance(k, str) or not k.strip()
                or not isinstance(v, str) or not v.strip()
                for k, v in replacements.items()):
            raise ValueError("replace must map existing terms to replacement terms")
        constraints = proposal.get("constraints", {})
        if not isinstance(constraints, dict) or set(constraints) - {"monotone"}:
            raise ValueError("constraints supports only a monotone mapping")
        monotone = constraints.get("monotone", {})
        if monotone == []:
            monotone = {}
        if not isinstance(monotone, dict) or any(
                not isinstance(k, str) or not k.strip()
                or v not in ("increasing", "decreasing") for k, v in monotone.items()):
            raise ValueError("monotone must map columns to increasing or decreasing")
        for key in ("flags", "assumptions"):
            if not isinstance(proposal.get(key, []), list) or any(
                    not isinstance(v, dict) for v in proposal.get(key, [])):
                raise ValueError(f"{key} must be a list of objects")
        for key in ("reasoning", "estimand_note", "evidence_id"):
            if key in proposal and not isinstance(proposal[key], str):
                raise ValueError(f"{key} must be a string")
        return cls(drop=names("drop"), add=names("add"), lock=names("lock"),
                   replace=dict(replacements), monotone=dict(monotone),
                   flags=tuple(proposal.get("flags", [])),
                   assumptions=tuple(proposal.get("assumptions", [])),
                   reasoning=proposal.get("reasoning", ""),
                   estimand_note=proposal.get("estimand_note", ""),
                   evidence_id=proposal.get("evidence_id"))


def _digest(round_index, *, response, spec, family, mstop, report, screened,
            suggestions, groups, fits, locks, causal, domain_context, search_id=None,
            preexploration=None):
    """The measured evidence a judge sees: bounded, JSON-safe, predictive only."""
    digest = {
        "round": round_index,
        "search_id": search_id,
        "cumulative_fits": fits,
        "response": response,
        "family": type(family).__name__,
        "mstop": mstop,
        "current_spec": spec,
        "term_gains": [
            {"term": row["term"], "earns": round(float(row["earns"]), 6),
             "se": float(row["se"]) if "se" in row else None,
             "measurable": row.get("measurable", True)}
            for row in report.contributions[:12]
        ],
        "screened_pairs": [
            {"left": p.left, "right": p.right, "removed": round(float(p.removed), 6)}
            for p in screened[:6]
        ],
        "suggestions": [
            {"action": s.action, "term": s.term} for s in suggestions[:8]
        ],
        "groups": list(groups),
        "locked_columns": list(locks),
        "treatment": causal.treatment if causal else None,
        "post_treatment": list(causal.post_treatment) if causal else [],
        "estimand": causal.describe() if causal else None,
        "domain_context": domain_context,
        "rules": list(_REFUSAL_RULES),
        "reply_schema": _JUDGE_SCHEMA,
    }
    if preexploration is not None:
        digest["pre_scan"] = {
            "engine": preexploration["engine"], "fits": preexploration["fits"],
            "neural_failure": preexploration.get("neural_failure"),
            "pair_sources": preexploration.get("pair_sources", [])[:12],
            "subsets": [row for row in preexploration.get("subsets", [])
                        if row.get("nominated")][:6],
            "interpretation": "screening proposals only; fold comparison decides adoption",
        }
    from mboostrs._workflow import summary as budget_summary

    resources = budget_summary()
    if resources is not None:
        digest["workflow_budget"] = resources
        digest["cumulative_fits"] = resources["fit_charges"]
    digest["evidence_id"] = hashlib.sha256(
        json.dumps(digest, sort_keys=True, default=str).encode()).hexdigest()
    return digest


_JUDGE_SYSTEM = (
    "You are the judgment layer of a measured statistical specification "
    "search. Every adoption decision is made by cross-validated evidence you "
    "never see; your value is domain reasoning about variables, interactions "
    "and constraints. Causes are not in the data. Obey the rules in the "
    "user message exactly. Reply with one JSON object and nothing else, "
    "matching the reply_schema: keys reasoning (string), drop (list of "
    "strings), add (list of strings), lock (list of strings), constraints "
    "(object mapping monotone columns to increasing or decreasing), "
    "replace (mapping of existing terms to replacements), "
    "evidence_id (copy from the digest), flags (list of objects with kind, term, "
    "note), assumptions (list of objects with statement, confidence), "
    "estimand_note (string). Use empty lists for keys you have nothing for."
)

_ACTION_SYSTEM = (
    "You choose one next action in a measured statistical specification search. "
    "The state lists the only legal actions and their costs. The user message "
    "contains training-only evidence, never test rows. Return one JSON object "
    "with action (one available action or 'stop'), arguments (object), "
    "state_id (copy exactly from the state), and reasoning (string). "
    "A screen is only a proposal; adoption requires a measured paired-fold "
    "comparison. A predictive association does not establish a causal effect. "
    "Do not request final_evaluate: only the analyst supplies untouched test "
    "data after freeze. If evidence or budget is insufficient, choose stop."
)


class LLMJudge:
    """A judge backed by any OpenAI-compatible chat endpoint.

    Parameters mirror :func:`~mboostrs.explain.llm_describe`: the default
    endpoint is NVIDIA's hosted inference API and the model id is
    configuration, so GLM, Nemotron or any served model works; a
    ``client=`` with ``chat.completions.create`` (the OpenAI SDK, LiteLLM,
    a local vLLM container) replaces the endpoint entirely. Each call is
    appended to :attr:`calls` with the prompt and reply SHA-256 digests,
    so a run's judgments are auditable without retaining their text
    (``retain_text=True`` keeps the text too).

    ``max_calls`` limits provider requests, including failed requests.
    ``max_total_tokens`` is checked between requests using provider-reported
    usage, so the last request can exceed it. Missing usage prevents another
    request when a token budget is set; it is never treated as zero cost.
    """

    def __init__(self, *, llm=DEFAULT_LLM, base_url=None, api_key=None,
                 client=None, temperature=0.0, max_tokens=2000, timeout=180,
                 retain_text=False, max_calls=None, max_total_tokens=None):
        self.llm = llm
        self.base_url = base_url
        self.api_key = api_key
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retain_text = retain_text
        self.calls: list = []
        self.max_calls = check_int_at_least("max_calls", max_calls, 1, allow_none=True)
        self.max_total_tokens = check_int_at_least("max_total_tokens", max_total_tokens, 1,
                                                  allow_none=True)

    @property
    def tokens_used(self):
        """Provider-reported tokens, or None when usage is unavailable."""
        total = 0
        for call in self.calls:
            count = (call.get("usage") or {}).get("total_tokens")
            if not isinstance(count, int) or count < 0:
                return None
            total += count
        return total

    def _send(self, messages):
        if self.client is not None:
            payload = _client_chat(
                self.client, messages, llm=self.llm,
                max_tokens=self.max_tokens, temperature=self.temperature,
                return_payload=True,
            )
            return payload, "client"
        api_key = self.api_key
        if api_key is None:
            api_key, _ = _from_env(_API_KEY_VARS)
        if not api_key:
            raise ValueError(
                "LLMJudge needs an API key: pass api_key=, or set one of "
                + ", ".join(_API_KEY_VARS)
                + ", or pass client= with its own configuration."
            )
        base_url = self.base_url
        if base_url is None:
            base_url, _ = _from_env(_BASE_URL_VARS)
        base_url = base_url or "https://integrate.api.nvidia.com/v1"
        payload = _post_chat(
            messages, llm=self.llm, base_url=base_url, api_key=api_key,
            max_tokens=self.max_tokens, temperature=self.temperature,
            timeout=self.timeout,
        )
        return payload, base_url

    def __call__(self, digest):
        """One judgment: send the digest, return the parsed JSON calls."""
        if self.max_calls is not None and len(self.calls) >= self.max_calls:
            raise RuntimeError("judgment call budget exhausted")
        if self.max_total_tokens is not None and (
                self.tokens_used is None or self.tokens_used >= self.max_total_tokens):
            raise RuntimeError("judgment token budget exhausted or usage unavailable")
        user = json.dumps(digest, indent=1, default=str)
        messages = [
            {"role": "system", "content": (
                _ELICIT_SYSTEM if digest.get("purpose") == "causal_design"
                else _ACTION_SYSTEM if digest.get("purpose") == "search_action"
                else _JUDGE_SYSTEM)},
            {"role": "user", "content": user},
        ]
        record = {
            "llm": self.llm,
            "prompt_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            "evidence_id": digest.get("evidence_id"),
            "usage": None,
            "status": "pending",
        }
        if self.retain_text:
            record["prompt"] = user
        start = time.perf_counter()
        self.calls.append(record)
        try:
            payload, endpoint = self._send(messages)
            record.update(usage=payload.get("usage"), endpoint=endpoint)
            reply = _reply_text(payload)
            record.update(reply_sha256=hashlib.sha256(reply.encode()).hexdigest(),
                          reply_chars=len(reply))
            if self.retain_text:
                record["reply"] = reply
            judgment = _extract_json(reply)
            record["status"] = "received"
            return judgment
        except Exception as exc:
            record.update(status="failed", error=str(exc)[:300])
            raise
        finally:
            record["seconds"] = time.perf_counter() - start


def _extract_json(text):
    """The first JSON object in a reply, tolerating fences and prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("the reply contains no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(obj, dict):
        raise ValueError("the reply's first JSON value is not an object")
    return obj


def _columns_of(spec):
    """The predictor columns a specification's right-hand side mentions."""
    return set(term_variables(spec))


#: What the elicitation interview asks for. Every field the model returns
#: is a proposal a person confirms; nothing here is inferred from the data.
_ELICIT_SCHEMA = {
    "treatment": "column name of the binary treatment/exposure",
    "outcome": "column name of the outcome measured after treatment",
    "time_zero": "when follow-up starts, in the study's own words",
    "follow_up": "the follow-up window that defines the outcome",
    "covariates": ["baseline covariate column names, measured before treatment"],
    "post_treatment": ["columns measured after treatment started"],
    "adjustment_rationale": "why these covariates suffice for the estimand; "
                            "predictive importance is not an argument",
    "questions": ["what a person must answer before this design is trusted"],
    "unresolved": ["design points the data cannot settle"],
}

_ELICIT_SYSTEM = (
    "You are eliciting a causal analysis design from a person through an "
    "interviewer. You see a variable digest: names, kinds, missingness. "
    "The digest is prompts, not evidence: causes are not in the data, so "
    "every role you assign is a proposal the person must confirm, and you "
    "must say what you could not decide (questions, unresolved). Never "
    "infer a causal role from an association or a column name alone; use "
    "the domain context if given, and otherwise ask. Reply with one JSON "
    "object matching the schema, nothing else."
)


@dataclass
class CausalDraft:
    """A proposed causal design: a spec a person has not yet confirmed.

    ``spec`` is ``None`` when the proposal failed the mechanical checks in
    ``problems``; ``questions`` and ``unresolved`` travel with the draft so
    the confirming person sees what the elicitation could not settle. The
    draft's ledger entry is ``llm-proposed-unconfirmed`` until
    :meth:`AssumptionLedger.confirm` marks it confirmed.
    """

    spec: object
    ledger: AssumptionLedger
    questions: list
    unresolved: list
    problems: list

    def __repr__(self):  # pragma: no cover - display only
        return (f"CausalDraft(spec={'set' if self.spec else 'none'}, "
                f"problems={len(self.problems)}, "
                f"questions={len(self.questions)}, "
                f"unresolved={len(self.unresolved)})")


def _variable_digest(frame):
    """Column names, kinds and missingness shares: prompts, not evidence."""
    kinds = classify_columns(frame)
    digest = []
    for kind in kinds:
        values = np.asarray(frame[kind.name]) if kind.name in frame else None
        missing = None
        if values is not None and values.dtype.kind in "fiub":
            missing = round(float(np.mean(~np.isfinite(values.astype(float)))), 4)
        digest.append({"column": kind.name, "kind": kind.kind,
                       "levels": getattr(kind, "levels", None),
                       "missing": missing})
    return digest


def elicit_causal_spec(frame, judgment, *, domain_context=None):
    """Interview a judgment layer for a causal design and validate it.

    The judge receives the variable digest and the schema above and returns
    a proposed treatment, outcome, timing, adjustment set and rationale.
    Mechanical checks then run against the frame: every named column must
    exist, the treatment must be binary for the supported estimators, the
    proposal must state timing and a rationale, and
    :class:`~mboostrs.CausalSpec`'s own validation must pass. A proposal
    that fails any check returns ``CausalDraft(spec=None, problems=[...])``
    — it is never silently repaired.

    Returns a :class:`CausalDraft`. Confirming the design is a separate
    human act: ``draft.ledger.confirm(draft.ledger.entries[0].statement)``
    upgrades the draft's entry to ``llm-proposed-confirmed``, and the
    person's own additions are recorded as ``modeler-stated``.

    Parameters
    ----------
    frame : mapping of column name to array
    judgment : callable
        ``judgment(digest) -> dict``; see :class:`LLMJudge`.
    domain_context : str, optional
        What the columns and the study mean, sent with the digest.
    """
    ledger = AssumptionLedger()
    digest = {
        "purpose": "causal_design",
        "variables": _variable_digest(frame),
        "domain_context": domain_context,
        "schema": _ELICIT_SCHEMA,
        "rules": list(_REFUSAL_RULES),
    }
    try:
        proposal = judgment(digest)
    except Exception as exc:
        return CausalDraft(None, ledger, [f"the judge failed: {str(exc)[:200]}"],
                           [], [f"judge error: {str(exc)[:200]}"])
    problems, questions, unresolved = [], [], []
    for key in ("treatment", "outcome", "time_zero", "follow_up",
                "adjustment_rationale"):
        if not str(proposal.get(key, "")).strip():
            problems.append(f"the proposal leaves {key!r} empty")
    columns = set(frame)
    for key in ("treatment", "outcome"):
        name = str(proposal.get(key, "")).strip()
        if name and name not in columns:
            problems.append(f"named {key} {name!r} is not a column of the frame")
    covariates = [str(c) for c in (proposal.get("covariates") or [])]
    missing = [c for c in covariates if c not in columns]
    if missing:
        problems.append(f"proposed covariates absent from the frame: {missing}")
    post = [str(c) for c in (proposal.get("post_treatment") or [])]
    treatment = str(proposal.get("treatment", "")).strip()
    if treatment in columns:
        values = np.asarray(frame[treatment])
        if values.dtype.kind in "fiub":
            values = values[np.isfinite(values.astype(float))]
        distinct = np.unique(values)
        if len(distinct) != 2:
            problems.append(
                f"treatment {treatment!r} has {len(distinct)} distinct values; "
                "the supported estimators need a binary treatment")
    spec = None
    if not problems:
        try:
            spec = CausalSpec(
                treatment=treatment,
                outcome=str(proposal["outcome"]).strip(),
                covariates=tuple(dict.fromkeys(covariates)),
                time_zero=str(proposal["time_zero"]).strip(),
                follow_up=str(proposal["follow_up"]).strip(),
                post_treatment=tuple(dict.fromkeys(post)),
                adjustment_rationale=str(proposal["adjustment_rationale"]).strip(),
            )
        except ValueError as exc:
            problems.append(f"the proposal fails CausalSpec validation: {exc}")
    questions = [str(q) for q in (proposal.get("questions") or [])]
    unresolved = [str(u) for u in (proposal.get("unresolved") or [])]
    ledger.add(
        spec.describe() if spec is not None
        else "an invalid causal design proposal (see problems)",
        "llm-proposed-unconfirmed", source="elicit_causal_spec")
    return CausalDraft(spec, ledger, questions, unresolved, problems)


def _numeric(values):
    return np.asarray(values).dtype.kind in "fiub"


def _complete(frame):
    keep = np.all([np.isfinite(np.asarray(v, dtype=float)) for v in frame.values()
                   if _numeric(v)], axis=0)
    return {c: np.asarray(v)[keep] for c, v in frame.items()}


def _folds(data, groups, seed):
    if groups:
        return grouped_folds(np.asarray(data[groups[0]], dtype=object), B=5, seed=seed)
    return 5


@dataclass
class ModelerResult:
    """Model and specification selected by the loop, with the decision record."""

    spec: str
    family: object
    mstop: int
    groups: tuple
    transcript: Transcript
    ledger: AssumptionLedger
    refusals: list
    fits: int
    causal_result: object = None
    trap_report: object = None
    judgment_calls: int = 0
    model: object = field(default=None, repr=False)
    stop_reason: str = "round_limit"
    locked_columns: tuple = ()
    budget_report: dict | None = None
    action_evidence: tuple = ()

    def to_dict(self, *, include_arrays=False):
        """Editable specification and audit trail, excluding model internals."""
        from mboostrs._output import json_safe

        return json_safe({
            "spec": self.spec, "family": type(self.family).__name__,
            "mstop": self.mstop, "groups": self.groups, "fits": self.fits,
            "stop_reason": self.stop_reason, "locked_columns": self.locked_columns,
            "judgment_calls": self.judgment_calls,
            "budget_report": self.budget_report,
            "action_evidence": [row.to_dict() for row in self.action_evidence],
            "transcript": self.transcript.steps, "refusals": self.refusals,
            "assumptions": self.ledger.to_list(),
            "causal_result": (self.causal_result.to_dict(include_arrays=include_arrays)
                              if self.causal_result is not None else None),
            "trap_report": (self.trap_report.to_dict(include_arrays=include_arrays)
                            if self.trap_report is not None else None),
        })

    def __repr__(self):  # pragma: no cover - display only
        return (f"ModelerResult({self.spec!r}, family={type(self.family).__name__}, "
                f"mstop={self.mstop}, groups={list(self.groups)}, fits={self.fits}, "
                f"judgment_calls={self.judgment_calls}, refusals={len(self.refusals)})")


def _candidate_problem(rhs, *, response, locks, forbidden, columns):
    touched = _columns_of(rhs)
    if touched & forbidden:
        return f"names forbidden column(s) {sorted(touched & forbidden)}"
    if columns is not None and touched - set(columns):
        return f"names unknown column(s) {sorted(touched - set(columns))}"
    if set(locks) - touched:
        return f"would drop locked column(s) {sorted(set(locks) - touched)}"
    try:
        if "~" in rhs:
            raise ValueError("a candidate must be a right-hand side")
        for term in split_terms(rhs):
            margins = term.split("%X%")
            if len(margins) > 2:
                raise ValueError("tensor terms require two margins")
            for marginal in margins:
                parse_gamboost_formula(f"{quote_response(response)} ~ {marginal.strip()}")
    except (ValueError, TypeError) as exc:
        return f"does not parse: {str(exc)[:160]}"
    return None


def _apply_judgment(judgment, *, spec, response, locks, forbidden, refusals,
                    ledger, round_index, t, columns=None):
    """Translate a validated proposal into editable, measurable candidates."""
    proposal = (judgment if isinstance(judgment, SpecificationJudgment)
                else SpecificationJudgment.from_mapping(judgment))
    candidates = {}
    terms = split_terms(spec)

    def refuse(change, reason):
        refusals.append({"round": round_index, "change": change, "reason": reason})

    def candidate(label, pieces):
        rhs = " + ".join(dict.fromkeys(pieces)) or "1"
        problem = _candidate_problem(rhs, response=response, locks=locks,
                                     forbidden=forbidden, columns=columns)
        if problem:
            refuse(label.removeprefix("judge "), problem)
        elif rhs != spec:
            candidates[label] = rhs

    for name in proposal.drop:
        removed = [term for term in terms if term == name or name in _columns_of(term)]
        if not removed:
            refuse(f"drop {name}", "no term of the specification names it")
        else:
            candidate(f"judge drop {name}", [term for term in terms if term not in removed])
    for term in proposal.add:
        candidate(f"judge add {term}", [*terms, term])
    for old, new in proposal.replace.items():
        if old not in terms:
            refuse(f"replace {old}", "replacement must name an existing term exactly")
        else:
            candidate(f"judge replace {old}",
                      [piece for term in terms for piece in (split_terms(new) if term == old else [term])])
    for name, direction in proposal.monotone.items():
        involved = [term for term in terms if name in _columns_of(term)]
        if not involved:
            refuse(f"constrain {name}", "column has no existing effect to constrain")
        elif any(_columns_of(term) != {name} or "%X%" in term or "by=" in term.replace(" ", "")
                 for term in involved):
            refuse(f"constrain {name}",
                   "resolve interactions or varying coefficients before imposing a whole-effect constraint")
        else:
            constrained = f'bmono({quote_name(name)}, constraint="{direction}")'
            candidate(f"judge constrain {name}",
                      [term for term in terms if term not in involved] + [constrained])

    extra_locks = []
    for name in proposal.lock:
        if name in locks:
            continue
        if name in forbidden or name not in _columns_of(spec):
            refuse(f"lock {name}", "only an existing permitted predictor can be locked")
            continue
        extra_locks.append(name)
        ledger.add(f"column {name} should not be removable from the specification",
                   "llm-proposed-unconfirmed", source="judgment lock request", round=round_index)
    for flag in proposal.flags:
        if flag.get("term"):
            t.log("judgment-flag", round=round_index,
                  kind=str(flag.get("kind", "note")), term=str(flag["term"]),
                  note=str(flag.get("note", ""))[:300])
    for item in proposal.assumptions:
        if str(item.get("statement", "")).strip():
            ledger.add(str(item["statement"]), "llm-proposed-unconfirmed",
                       source=f"judgment (confidence: {item.get('confidence', 'unstated')})",
                       round=round_index)
    return candidates, extra_locks


def _modeler_loop(frame, response="y", *, family=None, transcript=None, judgment=None,
                 causal=None, domain_context=None, locks=(), budget=250,
                 max_rounds=4, mstop=300, seed=0, tune_start=None,
                 preexploration="native", checkpoint=None):
    """Run the modeler's workflow: discover, digest, judge, edit, measure.

    With ``judgment=None`` and ``causal=None`` this is the documented
    deterministic search: classify columns, run a budgeted
    :func:`~mboostrs.autospec`, then up to ``max_rounds`` rounds of term
    reports, residual interaction screening and specification comparison
    under the two-paired-standard-error rule, then a final
    :func:`~mboostrs.tune_mstop`. The transcript records every step with
    cumulative fit counts, so the whole cost of the search is stated, not
    only autospec's probes.

    Parameters
    ----------
    frame : mapping of column name to array
    response : str
        Outcome column. With ``causal=`` this must equal the spec's outcome
        and is ignored in favour of it.
    transcript : Transcript, optional
        An existing transcript to append to.
    judgment : callable, optional
        ``judgment(digest) -> dict``; see :class:`LLMJudge`. Proposals are
        translated into candidates, refusals and ledger entries; paired
        fold evidence still decides adoption. A judge that raises or
        returns unparseable output is logged and the loop continues.
    causal : CausalSpec, optional
        Turns on the causal lane: treatment and declared post-treatment
        columns are excluded from the nuisance search, the adjustment
        covariates (plus ``locks=``) become locked terms no edit may
        remove, an :class:`AssumptionLedger` seeded from the spec records
        provenance, and the loop closes with the cross-fitted
        :func:`~mboostrs.aipw` estimate of the declared estimand.
    domain_context : str, optional
        What the columns mean, for the judge. Sent to the model, never
        used as evidence.
    locks : iterable of str, optional
        Columns no edit may remove, in addition to the causal adjustment set.
    budget, max_rounds, mstop, seed : search settings
    tune_start : int, optional
        Starting grid point for the final mstop sweep; default
        ``max(mstop, 200)``.
    preexploration : {"native", "neural", "dual", "hessboost", "off"}
        Discovery probe; ``"dual"`` shares tree and neural leads with the
        judge. These are proposals, and still require measured adoption.

    Returns
    -------
    ModelerResult
    """
    # Preserve each backend's column dtype while using one mapping protocol
    # throughout the search; no fitted transformation is materialized here.
    frame = {name: frame[name] for name in column_names(frame)}
    if causal is not None:
        response = causal.outcome
    max_rounds = check_int_at_least("max_rounds", max_rounds, 0, allow_none=False)
    seed = check_int_at_least("seed", seed, 0, allow_none=False)
    t = transcript if transcript is not None else Transcript()
    ledger = AssumptionLedger()
    refusals: list = []
    fits = 0

    model_frame = frame
    forbidden = set()
    base_locks = tuple(dict.fromkeys(locks))
    if causal is not None:
        forbidden = {causal.treatment, *causal.post_treatment}
        model_frame = {c: v for c, v in frame.items() if c not in forbidden}
        base_locks = tuple(dict.fromkeys((*causal.covariates, *base_locks)))
        missing = [c for c in causal.covariates if c not in model_frame]
        if missing:
            raise ValueError(f"adjustment covariates absent from the frame: {missing}")
        ledger.add(causal.describe(), "modeler-stated", source="CausalSpec")
        ledger.add(causal.adjustment_rationale, "modeler-stated",
                   source="adjustment_rationale")
        for name in causal.assumptions:
            ledger.add(name, "modeler-stated", source="CausalSpec.assumptions")
        t.log("causal-design", estimand=causal.describe(),
              locked=list(base_locks), excluded=sorted(forbidden))

    invalid_locks = set(base_locks) - (set(column_names(model_frame)) - {response})
    if invalid_locks:
        raise ValueError(f"locks name absent or forbidden predictors: {sorted(invalid_locks)}")
    if checkpoint is not None:
        checkpoint(ledger=ledger, refusals=refusals, locked_columns=base_locks)

    kinds = classify_columns(model_frame, response=response)
    rec = autospec(model_frame, response, family=family, budget=budget, mstop=mstop, seed=seed,
                   preexploration=preexploration)
    fits += rec.fits
    labelled = [f.column for f in rec.hygiene if f.kind == "integer-labels"]
    text_groups = [c.name for c in kinds if c.kind == "categorical" and c.levels > 2
                   and c.dtype == "categorical"]
    groups = tuple(dict.fromkeys(labelled + text_groups))
    if groups:
        rec = autospec(model_frame, response, family=family, budget=budget, mstop=mstop, groups=groups, seed=seed,
                       preexploration=preexploration)
        fits += rec.fits
    t.log("autospec", formula=rec.formula, family=type(rec.family).__name__,
          mstop=int(rec.mstop), groups=list(groups),
          nu=rec.control.nu if rec.control is not None else 0.1,
          folds=rec.folds_description, priority=list(rec.priority), fits=fits)

    family, mstop = rec.family, int(rec.mstop)
    nu = rec.control.nu if rec.control is not None else 0.1
    spec = partition_tilde(rec.formula)[2].strip()
    if base_locks:
        absent = [c for c in base_locks if c not in _columns_of(spec)]
        if absent:
            categorical = {k.name for k in kinds if k.kind in ("categorical", "label")}
            spec = " + ".join([spec, *(f"brandom({quote_name(c)})" if c in categorical else f"bols({quote_name(c)})"
                                      for c in absent)])
            t.log("causal-locks-added" if causal is not None else "locks-added", columns=absent,
                  reason="the adjustment set is fixed by the design")
    # Missing values: the search runs on the complete rows, as autospec's probes did.
    data = _complete(model_frame) if rec.incomplete_rows else model_frame
    folds = _folds(data, groups, seed)
    from mboostrs._workflow import _CURRENT
    from mboostrs.search_session import open_search

    active_workflow = _CURRENT.get()
    action_session = open_search(
        data, response, family=family, score="family", folds=folds, seed=seed,
        groups=groups, nu=nu, mstop=mstop, locks=base_locks,
        budget=None if active_workflow is None else active_workflow.budget)
    action_session.run("set_spec", rhs=spec)
    from mboostrs.build import build_info
    from mboostrs.evaluation import data_fingerprint

    search_id = hashlib.sha256(json.dumps({
        "data": data_fingerprint(data), "spec": spec, "seed": seed,
        "family": type(family).__name__, "build": build_info()["build_id"],
    }, sort_keys=True).encode()).hexdigest()
    candidates = {f"add {lead.term}": f"{spec} + {lead.term}"
                  for lead in rec.evidence.get("interactions", ())[:2]}

    judgment_calls = 0
    stop_reason = "round_limit"
    for round_index in range(max_rounds):
        fit = action_session.run("fit").value
        fits += 1
        if checkpoint is not None:
            checkpoint(spec=spec, family=family, mstop=mstop, model=fit, groups=groups,
                       judgment_calls=judgment_calls, locked_columns=base_locks)
        report = action_session.run("term_report", shapes=(round_index == 0)).value
        fits += report.fits
        screened = action_session.run(
            "interaction_screen", groups=report.groups, seed=0).value
        gaussian = type(family).__name__ in ("Gaussian", "GaussReg")
        suggestions = suggest_specifications(report, screened=screened[:3],
                                             model=fit if gaussian else None)
        # The joint removal is always scored; it can rank below single
        # removals because its terms' gains, summed, can be near zero.
        joint = [s for s in suggestions if s.action == "drop" and s.specification
                 and s.term.endswith("terms that earn nothing")]
        for s in (*joint, *suggestions[:6]):
            if s.action in ("refit", "drop", "add") and s.specification:
                # A continuous x continuous pair offers three learners (tensor
                # and both varying-coefficient directions) under one `term`, so
                # key an add by its full specification to score all three.
                # Drop/refit keep their short labels; the selection below
                # reads `drop ...` prefixes and `terms that earn nothing`.
                label = (s.specification if s.action == "add"
                         else f"{s.action} {s.term}")
                candidates.setdefault(label, s.specification)
        if any(s.action == "model" for s in suggestions):
            t.log("distributional-flag", round=round_index)
        if judgment is not None:
            from mboostrs._workflow import checkpoint as budget_checkpoint

            # Any proposed edit needs at least the incumbent and one
            # candidate on the same folds. Do not spend a provider call
            # when even that smallest comparison cannot be admitted.
            n_folds = int(getattr(folds, "n_folds", folds))
            budget_checkpoint("judgment_followup_comparison", requested=2 * n_folds)
            digest = _digest(
                round_index, response=response, spec=spec, family=family,
                mstop=mstop, report=report, screened=screened,
                suggestions=suggestions, groups=groups, fits=fits,
                locks=base_locks, causal=causal, domain_context=domain_context,
                search_id=search_id, preexploration=rec.evidence.get("preexploration"))
            t.log("evidence", round=round_index, evidence=digest)
            try:
                judgment_calls += 1
                record = judgment(digest)
                proposal = (record if isinstance(record, SpecificationJudgment)
                            else SpecificationJudgment.from_mapping(record))
                record = proposal.to_dict()
                if proposal.evidence_id is not None and proposal.evidence_id != digest["evidence_id"]:
                    raise ValueError("judgment cites a different evidence_id")
                judged, extra = _apply_judgment(
                    proposal, spec=spec, response=response, locks=base_locks,
                    forbidden=forbidden | {response}, refusals=refusals,
                    ledger=ledger, round_index=round_index, t=t,
                    columns=column_names(data))
            except Exception as exc:  # a failed judge must not stop the search
                t.log("judgment-failed", round=round_index,
                      evidence_id=digest["evidence_id"], error=str(exc)[:300])
            else:
                candidates.update(judged)
                base_locks = tuple(dict.fromkeys((*base_locks, *extra)))
                if extra:
                    action_session.run("lock", columns=extra)
                t.log("judgment", round=round_index,
                      drop=len(record.get("drop") or []),
                      add=len(record.get("add") or []),
                      flags=len(record.get("flags") or []),
                      assumptions=len(record.get("assumptions") or []),
                      evidence_id=digest["evidence_id"],
                      proposal=record,
                      candidates_added=len(judged),
                      judgment_sha256=hashlib.sha256(
                          json.dumps(record, sort_keys=True,
                                     default=str).encode()).hexdigest())
            if checkpoint is not None:
                checkpoint(judgment_calls=judgment_calls, locked_columns=base_locks)
        if not candidates:
            stop_reason = "no_candidates"
            break
        enforced = {}
        for label, rhs in candidates.items():
            problem = _candidate_problem(rhs, response=response, locks=base_locks,
                                         forbidden=forbidden | {response}, columns=column_names(data))
            if problem:
                refusals.append({"round": round_index, "change": label,
                                 "reason": problem})
            else:
                enforced[label] = rhs
        if not enforced:
            stop_reason = "no_permitted_candidates"
            break
        try:
            compared = action_session.run("compare", candidates=enforced)
            rows = compared.value
        except (ValueError, RuntimeError) as exc:
            stop_reason = "comparison_failed"
            t.log("comparison-failed", round=round_index, candidates=enforced,
                  error=str(exc)[:300], note="incumbent retained; failed fit cost unavailable")
            break
        fits += rows[-1]["cumulative fit count"]
        scored = [{"change": row["specification"], "delta": row["change"],
                   "se": row["change se"], "rhs": row["rhs"]} for row in rows[1:]]
        comparison_id = hashlib.sha256(json.dumps(
            {"search_id": search_id, "round": round_index, "scores": scored},
            sort_keys=True, default=str).encode()).hexdigest()
        t.log("comparison", round=round_index, comparison_id=comparison_id,
              baseline=spec, candidates=scored, fits=fits,
              rule="two paired SE search heuristic; not a hypothesis test")
        better = [s for s in scored if s["delta"] < -2 * s["se"]]
        # A removal the folds cannot distinguish from no change keeps the
        # smaller specification; the joint removal is preferred when it passes.
        removals = [s for s in scored if s["change"].removeprefix("judge ").startswith("drop")
                    and (s["delta"] <= 0 or s["delta"] < 2 * s["se"])]
        joint = [s for s in removals if "terms that earn" in s["change"]]
        if better:
            pick = min(better, key=lambda s: s["delta"])
        elif joint:
            pick = joint[0]
        elif removals:
            pick = min(removals, key=lambda s: s["delta"])
        else:
            pick = None
        if pick is None:
            stop_reason = "no_supported_change"
            t.log("stop", round=round_index, reason=stop_reason, comparison_id=comparison_id)
            break
        spec = pick["rhs"]
        action_session.run("adopt", comparison_id=compared.evidence_id,
                           candidate=pick["change"], simplification_se=2.0)
        t.log("accept", round=round_index, change=pick["change"],
              delta=round(pick["delta"], 5), se=round(pick["se"], 5), fits=fits,
              spec=spec, comparison_id=comparison_id)
        candidates = {}

    tuning = action_session.run(
        "tune", start=tune_start if tune_start is not None
        else max(mstop, 200), fold_local=False).value
    fits += tuning.fits + 2  # tuning's skeleton and final full-data fit
    if checkpoint is not None:
        checkpoint(spec=spec, family=family, mstop=tuning.mstop, model=tuning.model,
                   groups=groups, judgment_calls=judgment_calls, locked_columns=base_locks)
    t.log("final", spec=spec, mstop=tuning.mstop, sentence=tuning.sentence,
          fits=fits, stop_reason=stop_reason, search_id=search_id,
          nu=nu, seed=seed)

    causal_result = None
    trap_report = None
    if causal is not None:
        # The trap battery runs before the estimate: a design whose estimand
        # is unidentified on these rows is refused, not estimated with a wide
        # interval. Advisory findings travel with the result.
        # `frame`, not `model_frame`: the design check needs the treatment and
        # outcome, and the model frame excludes both by design.
        trap_report = causal_trap_report(frame, causal)
        t.log("causal-traps", blocking=[f.name for f in trap_report.blocking],
              advisory=[f.name for f in trap_report.advisory],
              findings=[str(f) for f in trap_report.findings])
        if trap_report.blocking:
            for finding in trap_report.blocking:
                refusals.append({"change": f"estimate {causal.outcome} ~ {causal.treatment}",
                                 "reason": finding.name, "detail": finding.message,
                                 "severity": "blocking"})
        else:
            causal_result = aipw(frame, causal, seed=seed)
            t.log("causal-result", estimate=round(float(causal_result.estimate), 6),
                  standard_error=(None if causal_result.standard_error is None
                                  else round(float(causal_result.standard_error), 6)),
                  interval=(None if causal_result.interval is None
                            else [round(float(v), 6) for v in causal_result.interval]),
                  nuisance={k: (round(float(v), 6) if isinstance(v, (int, float))
                                else v)
                            for k, v in (causal_result.nuisance or {}).items()
                            if isinstance(v, (int, float, str, bool))},
                  assumptions=ledger.to_list())
    return ModelerResult(spec=spec, family=family, mstop=tuning.mstop,
                         groups=groups, transcript=t, ledger=ledger,
                         refusals=refusals, fits=fits,
                         causal_result=causal_result, trap_report=trap_report,
                         judgment_calls=judgment_calls, model=tuning.model,
                         action_evidence=tuple(action_session.evidence),
                         stop_reason=stop_reason, locked_columns=base_locks)


def modeler_loop(frame, response="y", *, family=None, transcript=None, judgment=None,
                 causal=None, domain_context=None, locks=(), budget=250,
                 max_rounds=4, mstop=300, seed=0, tune_start=None,
                 preexploration="native", workflow_budget=None, progress=None):
    """Run a measured specification search with optional LLM proposals.

    ``workflow_budget=CostBudget(...)`` limits the entire call, including
    discovery, diagnostic refits, comparison, final tuning and causal
    nuisance fits. A simple linear/group model is fitted first as a usable
    fallback. Exhaustion returns the most recent completed incumbent with
    ``stop_reason='budget_<limit>'`` and an operation ledger in
    ``budget_report``. If no initial fit can be admitted, a structured
    ``workflow_budget`` runtime error reports that no model is available.

    Fit charges are hard admission limits. Native CV batches reserve all
    their folds before dispatch; failed batches retain that charge because
    their actual completed-fit count is unavailable. Time and memory are
    checked between native calls, so a running call or its preparation can
    overshoot them. Memory is the absolute process-lifetime high-water RSS
    in MiB, not the incremental allocation of this search. ``progress`` is
    called with one operation record after each admitted native call.

    ``budget`` remains the autospec probe allowance. Without
    ``workflow_budget``, the original deterministic/LLM search is used.
    ``family`` fixes the response family throughout discovery, comparison,
    final tuning and the budget fallback; when omitted, autospec chooses it.
    See :class:`SpecificationJudgment` for edits and :class:`LLMJudge` for
    provider budgets; only measured candidates can be adopted.
    """
    arguments = dict(response=response, family=family, transcript=transcript, judgment=judgment,
                     causal=causal, domain_context=domain_context, locks=locks, budget=budget,
                     max_rounds=max_rounds, mstop=mstop, seed=seed, tune_start=tune_start,
                     preexploration=preexploration)
    if workflow_budget is None:
        if progress is not None:
            raise ValueError("progress requires workflow_budget (use CostBudget() for no limits)")
        return _modeler_loop(frame, **arguments)

    from mboostrs._workflow import Workflow
    from mboostrs.policy import CostBudget, _linear_rhs

    if not isinstance(workflow_budget, CostBudget):
        raise TypeError("workflow_budget must be a CostBudget")
    max_rounds = check_int_at_least("max_rounds", max_rounds, 0, allow_none=False)
    mstop = check_int_at_least("mstop", mstop, 1, allow_none=False)
    seed = check_int_at_least("seed", seed, 0, allow_none=False)
    response = causal.outcome if causal is not None else response
    frame = {name: frame[name] for name in column_names(frame)}
    forbidden = {causal.treatment, *causal.post_treatment} if causal is not None else set()
    initial_data = {name: values for name, values in frame.items() if name not in forbidden}
    required = tuple(dict.fromkeys((*locks, *(causal.covariates if causal is not None else ()))))
    invalid = set(required) - (set(initial_data) - {response})
    if invalid:
        raise ValueError(f"locks name absent or forbidden predictors: {sorted(invalid)}")
    transcript = transcript if transcript is not None else Transcript()
    arguments.update(response=response, transcript=transcript)
    tracker = Workflow(workflow_budget, progress)
    with tracker:
        # No fit is spent choosing the fallback's family and row policy.
        initial = autospec(initial_data, response, family=family, budget=0, mstop=mstop, seed=seed)
        rhs = _linear_rhs(initial_data, response)
        fit = gamboost(f"{quote_response(response)} ~ {rhs}", initial_data, family=initial.family,
                       control=initial.control, na_action="omit")
        best = ModelerResult(spec=rhs, family=initial.family, mstop=mstop, groups=(),
                             transcript=transcript, ledger=AssumptionLedger(), refusals=[],
                             fits=tracker.charges, model=fit, locked_columns=required)
        transcript.log("budget-fallback", spec=rhs, mstop=mstop,
                       note="simple fitted candidate; not yet cross-validated")

        def checkpoint(**state):
            for name, value in state.items():
                setattr(best, name, value)
            tracker.checkpoint()

        try:
            result = _modeler_loop(frame, **arguments, checkpoint=checkpoint)
            best = result
            tracker.checkpoint()
        except Exception:
            if tracker.stopped is None:
                raise
            result = best
            result.stop_reason = f"budget_{tracker.stopped}"
            transcript.log("budget-stop", reason=tracker.stopped, spec=best.spec,
                           mstop=best.mstop,
                           note="most recent completed incumbent; later edits/tuning may be unfinished")
        result.fits = tracker.charges
        result.budget_report = tracker.report()
        transcript.log("workflow-budget", report=result.budget_report)
        return result
