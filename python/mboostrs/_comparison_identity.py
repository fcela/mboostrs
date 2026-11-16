"""Comparison identity: the rows, response, weights and scale a fit or evaluation used.

A ranking of candidate models is meaningful only when every candidate was
computed on the same observations in the same order, with the same effective
weights and on the same response scale. A fit records those facts once, as a
small JSON-safe dict in ``model._comparison_identity``, and the comparison
helpers read the dicts through :func:`check_comparison`, which returns one of
three outcomes:

``"rank"``
    Every row was computed on the same target. The helper sorts as usual.
``"segregate"``
    The rows share their data but not their scale, such as a model fitted to
    ``log(y)`` next to one fitted to ``y``, or a count family's probability
    mass next to a continuous family's density of the same counts. The helper
    shows every row with a ``comparison`` column giving the reason and does
    not order them.
``"refuse"``
    The rows were computed on different data: different kept rows, a
    different response or row order, or different effective weights. Two
    cross-validation curves are also refused when their risks come from
    different scoring rules (squared against absolute error). The check
    raises ``ValueError`` naming what differs.

The record holds digests of the data, not the data. Digests come from
:func:`mboostrs._provenance.fingerprint`, so a column hashes the same whichever
container holds it. The record's ``data_identity`` field is one SHA-256 digest
of the observation IDs in row order, the observed response values and the
effective weights (:func:`data_identity`). Two records whose ``data_identity``
values are equal describe the same data; two whose values differ are refused.
A record written before the field existed is checked through its separate
response, sample and weight digests.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Comparison", "CommonSample", "check_comparison", "check_curves", "check_cv_folds",
    "check_evaluations", "common_sample_plan", "data_identity", "family_scale",
    "fit_identity", "record_data_identity", "refit_on_common_sample", "risk_scale",
]

#: Layout version of the identity record. Bump when a field changes meaning.
#: Adding a field does not change the meaning of the others.
IDENTITY_VERSION = 1

#: Prefix of every `data_identity` digest; change it if the byte layout changes.
_DATA_IDENTITY_TAG = b"mboostrs-data-identity-v1\0"


def _update_float64(digest, label: bytes, values) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest.update(label + b"\0" + str(array.shape).encode("ascii") + b"\0")
    digest.update(array.view(np.uint8))


def data_identity(response, weights, row_ids, *, n_input: int | None = None) -> str:
    """SHA-256 hex digest of the observation IDs, response values and weights.

    Parameters
    ----------
    response
        The observed response, one array, or a sequence of equal-length arrays
        for a response of several columns (survival time and status, a cbind
        response). Values are hashed as little-endian float64.
    weights
        The effective weights, one per row, hashed as little-endian float64.
    row_ids
        The ID of each row, in row order. Integer IDs are hashed as
        little-endian int64; other IDs by the UTF-8 text of their ``repr``.
    n_input
        The number of input rows the kept rows were taken from, when rows were
        dropped. It separates "all 120 rows" from "rows 0-119 of 121".

    Two digests are equal only when the IDs, their order, the response value
    and the weight of every row are equal. A response permuted among rows of
    the same length therefore has another digest, as have the same response
    and IDs under other weights.
    """
    digest = hashlib.sha256(_DATA_IDENTITY_TAG)
    ids = np.asarray(row_ids)
    if ids.dtype.kind in "iub":
        digest.update(b"ids:int64\0" + str(ids.shape).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(ids, dtype="<i8").view(np.uint8))
    else:
        digest.update(b"ids:repr\0" + str(ids.shape).encode("ascii") + b"\0")
        for value in ids.ravel():
            item = value.item() if isinstance(value, np.generic) else value
            digest.update(repr(item).encode("utf-8") + b"\0")
    digest.update(b"n_input\0" + str(-1 if n_input is None else int(n_input)).encode("ascii"))
    columns = (list(response) if isinstance(response, (list, tuple))
               else [response])
    for index, column in enumerate(columns):
        _update_float64(digest, f"response{index}".encode("ascii"), column)
    _update_float64(digest, b"weights", weights)
    return digest.hexdigest()


def record_data_identity(record) -> str:
    """The ``data_identity`` of an identity record, or ``""`` when it has none.

    A model loaded from a file written before the field existed, or a fit
    whose response could not be digested, has none.
    """
    if not isinstance(record, dict):
        return ""
    return str(record.get("data_identity") or "")

_REMEDY = (
    "Fit every candidate on one shared sample with one response and one weight "
    "policy: drop the rows any candidate's formula drops (a missing value in a "
    "column it reads) from the data before fitting, and keep the response and "
    "weights the same across candidates."
)


def family_scale(family) -> tuple:
    """The identity of the loss a family's risk is measured in.

    Two risks are on one scale when the family class and every constructor
    setting match: ``Quantile(tau=0.1)`` and ``Quantile(tau=0.5)`` are different
    losses. Settings come from the constructor signature, so nuisance values a
    fit estimates (a dispersion, a threshold) do not split one family in two.
    """
    if family is None:
        return ()
    signature = inspect.signature(type(family).__init__)
    settings = tuple(
        (name, repr(getattr(family, name, None)))
        for name in signature.parameters
        if name != "self"
    )
    return (type(family).__name__, settings)


def _reference_measure(family) -> str | None:
    from mboostrs._likelihood import reference_measure

    return None if family is None else reference_measure(family)


def risk_scale(family) -> str:
    """The scale a fit's risk and its cross-validated risk are measured on.

    A distributional (LSS) family's risk is its complete negative
    log-likelihood, so two LSS families share a scale when their likelihoods
    describe the same quantity under the same reference measure: a
    ``GaussianLSS`` and a ``GammaLSS`` risk of one response can be compared, a
    ``GaussianLSS`` and a ``PoissonLSS`` risk cannot. Any other family's risk
    is its own loss, on the scale of that family with its settings
    (:func:`family_scale`).
    """
    from mboostrs._likelihood import family_likelihood

    if family is not None and hasattr(family, "n_params") and hasattr(family, "param_names"):
        contract = family_likelihood(family)
        if contract is not None and contract.loglik_source == "negative_risk":
            return f"lss_nll:{contract.loglik_measure}:{_reference_measure(family)}"
    return repr(family_scale(family))


def _as_names(response_name) -> tuple[str, ...]:
    return (response_name,) if isinstance(response_name, str) else tuple(response_name)


def fit_identity(*, data, response_name, response_transform, weights, weight_policy,
                 dropped, family) -> dict[str, Any] | None:
    """The identity record of one fit, or None when a column cannot be digested.

    Parameters
    ----------
    data
        The processed frame the fit read. It still holds the observed response
        columns, because a transform adds its derived column beside them.
    response_name
        The response as the fit reads it: a column name, or a tuple of names for
        survival, cbind and known-variance responses.
    response_transform
        The recipe step of a transformed response (``log(y)``), or None.
    weights
        The effective fit weights, one per kept row.
    weight_policy
        The weight policy name the fit used.
    dropped
        Indices of the input rows removed by ``na_action``. The input row count
        is the kept rows plus these.
    family
        The family object, for the scoring rule.

    Returns None when a column is of a type the digest cannot read. The
    comparison then falls back to comparing the retained data directly.
    """
    from mboostrs._frame import read_column
    from mboostrs._provenance import fingerprint

    w = np.asarray(weights, dtype=np.float64).ravel()
    n_kept = int(w.size)
    if response_transform is not None:
        # The observed column is the transform's source. Digesting it, rather
        # than the derived column, makes `log(y)` and `y` fits of one sample
        # carry the same response digest and differ only in their transform.
        observed_names = tuple(response_transform.sources)
        transform_label = response_transform.expression
    else:
        observed_names = _as_names(response_name)
        transform_label = "identity"
    try:
        observed = {name: read_column(data, name).values for name in observed_names}
        for name, values in observed.items():
            if len(values) != n_kept:
                raise ValueError(f"response column {name!r} has {len(values)} rows, "
                                 f"the fit kept {n_kept}")
        dropped_index = np.asarray(sorted(int(i) for i in dropped), dtype=np.intp)
        n_input = n_kept + int(dropped_index.size)
        included = np.ones(n_input, dtype=bool)
        included[dropped_index] = False
        try:
            # Observation IDs are the kept rows' positions in the input.
            identity = data_identity([observed[name] for name in observed_names], w,
                                     np.flatnonzero(included), n_input=n_input)
        except (TypeError, ValueError):
            identity = None
        return {
            "version": IDENTITY_VERSION,
            "data_identity": identity,
            "response": {
                "columns": list(observed_names),
                "n_rows": n_kept,
                "sha256": fingerprint(observed)["sha256"],
            },
            "sample": {
                "n_input": n_input,
                "n_rows": n_kept,
                "sha256": fingerprint({"included": included})["sha256"],
            },
            "weights": {
                "policy": weight_policy or "upstream",
                "n_rows": n_kept,
                "sha256": fingerprint({"weight": w})["sha256"],
            },
            "response_transform": transform_label,
            "likelihood_measure": ("response" if response_transform is None
                                   else "transformed_response"),
            "scoring_rule": repr(family_scale(family)),
            "reference_measure": _reference_measure(family),
            "risk_scale": risk_scale(family),
        }
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Comparison:
    """The outcome of :func:`check_comparison` for one table of models."""

    verdict: str
    reasons: tuple[str, ...] = ()

    @property
    def notice(self) -> str:
        """The text a segregated row carries in its ``comparison`` column."""
        return "not ranked: " + "; ".join(self.reasons)


def _data_refusals(records, noun: str = "model") -> list[str]:
    """Differences in the rows or the response between each record and the first.

    When both records carry a ``data_identity``, it decides: equal digests
    mean the same data, and unequal ones are refused, named by the component
    digest that differs. A record without one (written before the field
    existed) is checked through the component digests alone.
    """
    ref = records[0]
    found = []
    for j, rec in enumerate(records[1:], start=2):
        mine, theirs = record_data_identity(rec), record_data_identity(ref)
        if mine and theirs and mine == theirs:
            continue
        if rec["sample"]["sha256"] != ref["sample"]["sha256"]:
            if rec["sample"]["n_input"] != ref["sample"]["n_input"]:
                what = (f"it was fitted to {rec['sample']['n_input']} input rows, "
                        f"{noun} 1 to {ref['sample']['n_input']}")
            else:
                what = (f"it kept {rec['sample']['n_rows']} of "
                        f"{rec['sample']['n_input']} rows and {noun} 1 kept "
                        f"{ref['sample']['n_rows']}; the kept rows are not the same rows")
            found.append(f"{noun} {j} is on a different sample: {what}")
        elif rec["response"]["sha256"] != ref["response"]["sha256"]:
            found.append(f"{noun} {j} has different response values or row order "
                         f"from {noun} 1")
        elif rec["weights"]["sha256"] != ref["weights"]["sha256"]:
            found.append(f"{noun} {j} has different effective weights from {noun} 1 "
                         f"(weight policy {rec['weights']['policy']!r} against "
                         f"{ref['weights']['policy']!r})")
        elif mine and theirs:
            found.append(f"{noun} {j} has a different data identity from {noun} 1 "
                         f"({mine[:12]} against {theirs[:12]}): the observation IDs, "
                         f"response values or effective weights differ")
    return found


def _score_refusals(records, noun: str = "model") -> list[str]:
    """Differences in the scoring rule each record's risk is measured with.

    A record written before ``risk_scale`` existed is not checked here.
    """
    ref = records[0]
    found = []
    for j, rec in enumerate(records[1:], start=2):
        mine, theirs = rec.get("risk_scale"), ref.get("risk_scale")
        if mine is not None and theirs is not None and mine != theirs:
            found.append(f"{noun} {j} is scored with {mine} and {noun} 1 with {theirs}; "
                         f"risks measured by different scoring rules have no paired "
                         f"difference")
    return found


def _scale_segregations(records, noun: str = "model") -> list[str]:
    """Differences in response scale or likelihood measure, which keep rows unranked."""
    ref = records[0]
    found = []
    for j, rec in enumerate(records[1:], start=2):
        if rec["response_transform"] != ref["response_transform"]:
            found.append(
                f"{noun} {j} uses response transform {rec['response_transform']!r} and "
                f"{noun} 1 uses {ref['response_transform']!r}; their likelihoods are "
                f"over different response scales, so their risks and information "
                f"criteria are not on one scale")
        elif rec["likelihood_measure"] != ref["likelihood_measure"]:
            found.append(f"{noun} {j} takes its likelihood over "
                         f"{rec['likelihood_measure']!r} and {noun} 1 over "
                         f"{ref['likelihood_measure']!r}")
        elif (rec.get("reference_measure") and ref.get("reference_measure")
              and rec["reference_measure"] != ref["reference_measure"]):
            found.append(
                f"{noun} {j}'s likelihood is taken with respect to the "
                f"{rec['reference_measure']!r} measure and {noun} 1's with respect to "
                f"{ref['reference_measure']!r}; a probability mass and a density of "
                f"the same observations are not on one scale")
        if rec["weights"]["policy"] != ref["weights"]["policy"]:
            # Weights that match after the policy has run would give equal
            # risks, but the policies name different weight contracts, and a
            # ranking should not rest on that coincidence.
            found.append(f"{noun} {j} was fitted with weight policy "
                         f"{rec['weights']['policy']!r} and {noun} 1 with "
                         f"{ref['weights']['policy']!r}; the two policies are different "
                         f"weight contracts even where the effective weights coincide, "
                         f"so the rows are not ranked")
    return found


def _decide(records, *, shared: str | None, helper: str) -> Comparison:
    """The rank, segregate or refuse decision for records that have identities."""
    refusals = _data_refusals(records)
    if shared is not None:
        refusals.append(shared)
    if refusals:
        raise ValueError(
            f"{helper}() cannot rank these models: ranking needs the same observed "
            f"responses, row order, kept rows and effective weights, and these differ: "
            + "; ".join(refusals) + ". " + _REMEDY)
    reasons = _scale_segregations(records)
    if reasons:
        return Comparison("segregate", tuple(reasons))
    return Comparison("rank")


def check_comparison(models, *, helper: str) -> Comparison:
    """Decide whether the models in one table can be ranked against each other.

    Returns ``Comparison("rank")`` or ``Comparison("segregate", reasons)``, and
    raises ``ValueError`` on a refusal, naming each difference and the remedy.
    A model without an identity record (fitted before the record existed, or
    by a path that does not write one) is compared through its retained data,
    which is the check the helpers used before identity records existed.
    """
    from mboostrs.model_selection import _require_same_response_and_weights, _shared_column_mismatch

    models = list(models)
    if len(models) < 2:
        return Comparison("rank")
    records = [getattr(model, "_comparison_identity", None) for model in models]
    if any(record is None for record in records):
        _require_same_response_and_weights(models)
        return Comparison("rank")
    return _decide(records, shared=_shared_column_mismatch(models), helper=helper)


def check_curves(records, *, helper: str = "compare_cv") -> Comparison:
    """The same decision for two cross-validation curves, read from their records.

    A curve's record is the identity of the model it was computed for. Two
    curves on different rows, response values or weights are refused, since a
    paired difference of their risks would compare the data as much as the
    models. A pair that differs only in response scale (``log(y)`` against
    ``y``) is segregated: its risks are on different scales, so there is no
    difference to report.
    """
    if len(records) < 2:
        return Comparison("rank")
    refusals = _data_refusals(records, noun="curve")
    if refusals:
        raise ValueError(
            f"{helper}() cannot pair these curves: they were computed on different "
            f"folds, because their response, rows or weights differ: "
            + "; ".join(refusals) + ". Build one fold matrix -- with "
            "mboostrs.cv_weights or one of the fold constructors -- and fit both "
            "models on the same data.")
    score_refusals = _score_refusals(records, noun="curve")
    if score_refusals:
        raise ValueError(
            f"{helper}() cannot pair these curves: their out-of-fold risks are "
            f"measured by different scoring rules: " + "; ".join(score_refusals)
            + ". Score both models with one response score on the same folds, "
            "for example mboostrs.cv_risk(..., score='mse') for each family.")
    reasons = _scale_segregations(records, noun="curve")
    if reasons:
        return Comparison("segregate", tuple(reasons))
    return Comparison("rank")


def check_cv_folds(results, *, helper: str) -> None:
    """Refuse a table of cross-validation results that do not share one fold matrix.

    ``None`` entries (a model without a cross-validation result) are skipped.
    Every present result needs a non-empty fold fingerprint, and all of them
    must be equal, so every model is scored on the same folds.
    """
    fingerprints = {c.fold_fingerprint for c in results if c is not None}
    if not fingerprints or "" in fingerprints or len(fingerprints) > 1:
        raise ValueError(
            f"{helper}() cv results must come from the same folds: every supplied "
            "result needs a non-empty fold_fingerprint, and all of them must be equal")


@dataclass(frozen=True)
class CommonSample:
    """The rows every model in a table can be scored on.

    ``keep`` holds the input row positions in every candidate's sample, in
    input order. ``excluded`` and ``excluded_by`` are per model, in the order
    the models were given: how many rows that model kept and the common sample
    removes, and which other candidates dropped those rows, with counts.
    """

    keep: np.ndarray
    n_input: int
    excluded: tuple[int, ...]
    excluded_by: tuple[str, ...]

    @property
    def changed(self) -> bool:
        """True when at least one model loses rows to the common sample."""
        return any(self.excluded)

    @property
    def n_rows(self) -> int:
        return int(self.keep.size)


def _rows(count: int) -> str:
    return f"{count} row" if count == 1 else f"{count} rows"


def common_sample_plan(models, labels) -> CommonSample:
    """The rows eligible for every model, and which candidate excludes each row.

    A model's eligible rows are the input rows it did not drop (its
    ``na_action`` removed the rest). The common sample is the intersection of
    those sets. Every model must have been fitted to the same input, which
    its identity record shows by the input row count, and each model's record
    must agree with its dropped rows, so that the row positions can be
    placed. Nothing is scored here.
    """
    models = list(models)
    records = [getattr(model, "_comparison_identity", None) for model in models]
    if any(record is None for record in records):
        raise ValueError(
            "common_sample=True needs the comparison identity of every model, which "
            "records the rows each one dropped; refit the models with this release "
            "rather than using ones loaded from an older file.")
    sizes = {record["sample"]["n_input"] for record in records}
    if len(sizes) > 1:
        raise ValueError(
            f"common_sample=True needs one input data set: the models were fitted to "
            f"{sorted(sizes)} input rows. Fit every candidate to the same data frame, "
            f"then pass common_sample=True.")
    n_input = sizes.pop()
    dropped = [np.asarray(sorted({int(i) for i in (getattr(model, "_na_dropped", None) or ())}),
                          dtype=np.intp) for model in models]
    for j, (record, gone) in enumerate(zip(records, dropped, strict=True), start=1):
        if record["sample"]["n_rows"] + gone.size != n_input:
            raise ValueError(
                f"model {j} kept {record['sample']['n_rows']} rows and dropped "
                f"{gone.size} of {n_input} input rows, which does not add up; the "
                f"model cannot be placed on the common sample. Refit it.")
    union = np.unique(np.concatenate(dropped)) if dropped else np.empty(0, dtype=np.intp)
    keep = np.setdiff1d(np.arange(n_input), union)
    excluded, excluded_by = [], []
    for j, gone in enumerate(dropped):
        # Rows model j keeps that some other candidate dropped.
        removed = np.setdiff1d(union, gone)
        excluded.append(int(removed.size))
        causes = []
        for k, other in enumerate(dropped):
            if k == j:
                continue
            count = int(np.isin(removed, other).sum())
            if count:
                causes.append(f"{labels[k]} ({_rows(count)})")
        excluded_by.append(", ".join(causes))
    return CommonSample(keep=keep, n_input=n_input, excluded=tuple(excluded),
                        excluded_by=tuple(excluded_by))


def refit_on_common_sample(model, plan: CommonSample):
    """The same fit as ``model``, refitted on the common sample's rows.

    The row positions are taken in the model's own retained rows: every
    common row is one the model kept, so the positions exist. A standard
    model re-runs its preprocessing on the common rows, as a fit on those rows
    alone would. A distributional model keeps its processed frame and subsets
    it, since it retains no raw frame. A fit whose matrices or subsampling
    cannot be restricted to a subset raises with the reason from
    :func:`mboostrs._refit.require_row_subsettable`.
    """
    from mboostrs._dispatch import is_lss
    from mboostrs._refit import require_row_subsettable

    gone = np.asarray(sorted({int(i) for i in (getattr(model, "_na_dropped", None) or ())}),
                      dtype=np.intp)
    kept_input = np.setdiff1d(np.arange(plan.n_input), gone)
    positions = np.searchsorted(kept_input, plan.keep)
    if is_lss(model):
        spec = require_row_subsettable(model, "common_sample=")
        return spec.subset(positions).build()
    spec = require_row_subsettable(model, "common_sample=", refit_preprocessing=True)
    return spec.subset(positions, refit_preprocessing=True).build()


def check_evaluations(evaluations, *, helper: str) -> Comparison:
    """The same check for held-out evaluation results, which never segregate.

    A comparison of held-out scores reports a paired fold standard error for
    each difference, which has no meaning across different targets. So every
    pair must share its target (data fingerprint, folds, score and evaluation
    weights) or the check raises. Returns ``Comparison("rank")`` otherwise.
    """
    evaluations = list(evaluations)
    if len(evaluations) < 2:
        return Comparison("rank")
    reference = evaluations[0]
    found = []
    for index, other in enumerate(evaluations[1:], start=2):
        issues = reference.comparison_issues(other)
        if issues:
            found.append(f"result {index} differs from result 1 in " + ", ".join(issues))
    if found:
        raise ValueError(
            f"{helper}() evaluation results are not comparable: " + "; ".join(found) + ". "
            "Score every candidate on the same data, folds, score and weights; "
            "rerun any earlier rows under those settings.")
    return Comparison("rank")
