"""Selection reports: learners the data cannot tell apart, and what replaces them.

When two predictors carry nearly the same information (a variable and a
near-copy of it, two encodings of one axis), a boosted fit picks one of them by
a small margin, and that pick can change from one resample to the next. The
functions here name the learners that could stand in for one another, in two
ways:

- :func:`equivalence_sets` groups learners whose design spaces overlap on the
  fitted rows. The overlap is the largest fraction of one learner's function
  space that the other learner can reproduce, from the core's pairwise
  concurvity (`model.concurvity(pairwise=True)`). Two learners at overlap 1
  give the same fitted values whatever the coefficients.
- :func:`substitution_sets` reads the per-subsample selections of a stability
  selection run and finds learners that are often selected, but rarely in the
  same subsample. Each such group reports how often any member was selected,
  next to the frequency of each member.

:func:`selection_report` puts both next to a :func:`~mboostrs.deselect`
result, and names the dropped learners that have a kept equivalent.

Every output here describes predictive substitution in the rows and resamples
the fit used. A group does not say which member causes the response, and
selecting one member over another is not evidence about the mechanism.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations

#: Attached to every report so the scope travels with the numbers.
SCOPE = (
    "Predictive substitution in the rows and resamples used. A group names "
    "learners that carry the same information there; it does not identify "
    "which member is causal, and it is not a causal claim."
)


def _finite(value):
    """A float, or None for NaN, so the record stays JSON-safe."""
    value = float(value)
    return None if value != value else value


@dataclass(frozen=True)
class EquivalenceSets:
    """Groups of learners whose design spaces overlap at or above a threshold.

    ``groups`` holds the connected components of the pairs whose overlap is at
    least ``threshold``, each with two or more learners. Components can chain:
    if A overlaps B and B overlaps C, the three form one group even when A and
    C fall below the threshold, so read ``overlap`` for the pair values.
    """

    #: Each group as a tuple of learner names, in the model's learner order.
    groups: tuple
    #: Overlap of each measured pair, keyed by ``(a, b)`` in learner order. The
    #: value is the smaller of the two directional concurvity values, so it is
    #: symmetric. None when the core could not measure the pair (NaN).
    overlap: dict = field(repr=False)
    #: The overlap at or above which two learners were grouped.
    threshold: float
    #: Learners whose design was measured, in learner order. Trees and
    #: ``%O%`` Kronecker learners have no design in the core's concurvity
    #: measure and are not listed.
    measured: tuple = ()
    scope: str = SCOPE

    def group_of(self, name):
        """The group containing ``name``, or None when it has no equivalent."""
        for group in self.groups:
            if name in group:
                return group
        return None

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record. ``include_arrays`` is accepted for the shared
        export interface; these results hold no arrays, so it changes nothing."""
        return {
            "groups": [list(group) for group in self.groups],
            "overlap": [
                {"a": a, "b": b, "overlap": _finite(value) if value is not None else None}
                for (a, b), value in self.overlap.items()
            ],
            "threshold": self.threshold,
            "measured": list(self.measured),
            "scope": self.scope,
        }


@dataclass(frozen=True)
class SubstitutionGroup:
    """Learners that replace one another across subsamples."""

    #: Member learner names, in the run's learner order.
    members: tuple
    #: Share of subsamples in which at least one member was selected. This is
    #: the frequency of the decision the group stands for.
    group_frequency: float
    #: Each member's share of subsamples in which it was selected.
    member_frequency: dict


@dataclass(frozen=True)
class SubstitutionSets:
    """Substitute pairs and the groups they form, from a stability selection run.

    A pair is a substitute pair when both learners reach ``min_share`` of the
    subsamples, and they are selected together in at most ``max_joint`` times
    the smaller of their two frequencies. Groups are the connected components of
    the substitute pairs, so a group can chain as described for
    :class:`EquivalenceSets`.
    """

    groups: tuple
    #: Substitute pairs as dicts with ``a``, ``b``, ``frequency_a``,
    #: ``frequency_b``, ``frequency_both`` (both selected), ``frequency_either``
    #: (at least one selected), and ``overlap`` (None when no equivalence sets
    #: were given), sorted by the joint frequency.
    pairs: tuple
    #: Subsamples the frequencies were counted over: the fits ``stabsel`` ran.
    subsamples: int
    min_share: float
    max_joint: float
    #: The overlap a pair needed, or None when the pairs were not screened by
    #: overlap.
    min_overlap: float | None = None
    scope: str = SCOPE

    def group_of(self, name):
        """The substitution group containing ``name``, or None."""
        for group in self.groups:
            if name in group.members:
                return group
        return None

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record. ``include_arrays`` is accepted for the shared
        export interface; these results hold no arrays, so it changes nothing."""
        return {
            "groups": [
                {"members": list(g.members), "group_frequency": g.group_frequency,
                 "member_frequency": dict(g.member_frequency)}
                for g in self.groups
            ],
            "pairs": [dict(p) for p in self.pairs],
            "subsamples": self.subsamples,
            "min_share": self.min_share,
            "max_joint": self.max_joint,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class SelectionReport:
    """Equivalence sets, substitution sets, and the deselected learners they explain.

    ``dropped_with_substitute`` lists each learner that :func:`~mboostrs.deselect`
    dropped while a learner in its equivalence group was kept. Such a drop
    removes one of two interchangeable predictors, not the information they
    share. A dropped learner with no kept equivalent is listed in
    ``dropped_alone``.
    """

    equivalence: EquivalenceSets
    substitution: SubstitutionSets | None
    dropped_with_substitute: tuple
    dropped_alone: tuple
    scope: str = SCOPE

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record. ``include_arrays`` is accepted for the shared
        export interface; these results hold no arrays, so it changes nothing."""
        return {
            "equivalence": self.equivalence.to_dict(),
            "substitution": None if self.substitution is None else self.substitution.to_dict(),
            "dropped_with_substitute": [dict(row) for row in self.dropped_with_substitute],
            "dropped_alone": [dict(row) for row in self.dropped_alone],
            "scope": self.scope,
        }


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _components(n, edges):
    """Connected components of ``edges`` over ``range(n)``, as sorted index lists."""
    parent = list(range(n))
    for a, b in edges:
        parent[_find(parent, a)] = _find(parent, b)
    buckets = {}
    for i in range(n):
        buckets.setdefault(_find(parent, i), []).append(i)
    return [sorted(members) for members in buckets.values() if len(members) >= 2]


#: A basis wider than this share of the rows raises the overlap of unrelated
#: learners; see `_warn_if_bases_are_wide`.
WIDE_BASIS_SHARE = 0.1


def _warn_if_bases_are_wide(model, names):
    """Warn when a learner's basis has more than `WIDE_BASIS_SHARE` of the rows as columns.

    The overlap is the largest fraction of one learner's function space that
    another can reproduce on the rows. With a basis that takes a large share of
    the rows, two learners reproduce each other's functions by chance. On a
    check with three independent normal predictors and `bbs(df=4)` (about 24
    columns), the largest unrelated overlap was 0.99 on 71 rows, and 0.57 on
    300 rows.
    """
    store = getattr(model, "_learner_store", None)
    if store is None or not hasattr(store, "extract_design"):
        store = model._replay_store()
    if store is None or not hasattr(store, "extract_design"):
        return
    n = len(model._fitted)
    widest = None
    for rust_idx in range(len(names)):
        design = store.extract_design(rust_idx)
        if design is None or not design:
            continue
        columns = len(design[0])
        if widest is None or columns > widest[0]:
            widest = (columns, names[model._rust_to_formula(rust_idx)])
    if widest is None or n == 0 or widest[0] / n <= WIDE_BASIS_SHARE:
        return
    columns, name = widest
    warnings.warn(
        f"the widest basis is {name!r} with {columns} columns on {n} rows "
        f"({columns / n:.2f} of the rows, above {WIDE_BASIS_SHARE}). Learners "
        "with a basis this wide can overlap each other by chance: on a check "
        "with independent normal predictors, a 24-column bbs overlapped 0.99 on "
        "71 rows and at most 0.57 on 300 rows. Read the pair values in "
        "`overlap` before treating a group as interchangeable, or use fewer "
        "basis columns (a smaller `df=`) or more rows.",
        stacklevel=3,
    )


def equivalence_sets(model, *, threshold=0.9):
    """Learners whose design spaces overlap at or above ``threshold``.

    Parameters
    ----------
    model : GAMBoostModel or GLMBoostModel
        A fitted model. The overlap is measured on the rows the fit used, over
        every learner with a design, whether or not the boosting path selected
        it.
    threshold : float
        Overlap at or above which two learners are grouped, in (0, 1]. The
        core's concurvity measure reports 1 for a learner that another learner
        reproduces exactly. On a simulated fit with 300 rows, unrelated terms
        measured 0.15 to 0.3 and a near-copy of a variable about 0.99. On few
        rows, unrelated terms can also reach the threshold; a warning names
        the wide basis when that is possible.

    Returns
    -------
    EquivalenceSets

    Notes
    -----
    The overlap is a pessimistic bound: it is the largest fraction of one
    learner's function space that the other can reproduce over every
    coefficient vector, so a rich basis such as a P-spline can score high
    against a linear term that lies in its span. Read the pair values, not
    only the groups.
    """
    from mboostrs._model_api import _learner_label, concurvity_rows

    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must lie in (0, 1], got {threshold}")
    if type(model).__name__ not in ("GAMBoostModel", "GLMBoostModel"):
        raise TypeError(
            "equivalence_sets measures one design per learner of a GAMBoostModel or "
            "GLMBoostModel. A distributional (LSS) fit has one design per parameter "
            "and is not supported; fit the parameter's model on its own to compare "
            "its learners."
        )
    names = list(model._learner_names)
    label_to_name = {_learner_label(model, i): names[i] for i in range(len(names))}
    rows = concurvity_rows(model, pairwise=True)
    _warn_if_bases_are_wide(model, names)
    directional = {}
    for row in rows:
        term = label_to_name.get(row["term"], row["term"])
        against = label_to_name.get(row["against"], row["against"])
        directional[(term, against)] = row["worst"]
    measured = tuple(
        name for name in names
        if any(term == name for term, _ in directional)
    )
    index = {name: i for i, name in enumerate(measured)}
    overlap = {}
    edges = []
    for a, b in combinations(measured, 2):
        ab = directional.get((a, b))
        ba = directional.get((b, a))
        if ab is None or ba is None:
            continue
        value = min(ab, ba)
        value = None if value != value else float(value)
        overlap[(a, b)] = value
        if value is not None and value >= threshold:
            edges.append((index[a], index[b]))
    groups = tuple(
        tuple(measured[i] for i in members)
        for members in _components(len(measured), edges)
    )
    return EquivalenceSets(groups=groups, overlap=overlap, threshold=float(threshold),
                           measured=measured)


def substitution_sets(stability, *, min_share=0.1, max_joint=0.2,
                      equivalence=None, min_overlap=0.5):
    """Learners that stability selection picks in turn rather than together.

    Parameters
    ----------
    stability : StabSelResult
        A result of :func:`~mboostrs.stabsel` on a ``GAMBoostModel`` or
        ``GLMBoostModel``. Its ``fit_selected`` holds each subsample's
        selection.
    min_share : float
        A learner enters the search when it is selected in at least this share
        of the subsamples.
    max_joint : float
        A pair is a substitute pair when the share of subsamples selecting both
        is at most ``max_joint`` times the smaller of the two individual shares.
        Two independent learners selected in shares ``f`` and ``g`` are selected
        together in about ``f * g`` of the subsamples, which is ``max(f, g)``
        times ``min(f, g)``. A ``max_joint`` well below the larger share
        therefore separates alternation from chance co-selection.
    equivalence : EquivalenceSets, optional
        When given, a pair also has to overlap at least ``min_overlap`` in the
        fit's design spaces. Without this, two unrelated learners that the path
        rarely selects together can join a group only because one of them is
        rarely selected at all, so a chain of unrelated learners can form.
    min_overlap : float
        The overlap a pair needs when ``equivalence`` is given. Unrelated
        learners measured 0.16 to 0.3 in the simulation behind this module, and
        a correlated proxy (correlation about 0.89) measured 0.79 to 0.82.

    Returns
    -------
    SubstitutionSets

    Raises
    ------
    ValueError
        When the result carries no per-subsample selections, as LSS stability
        results do not.
    """
    subsample_sets = stability.fit_selected
    if not subsample_sets:
        raise ValueError(
            "substitution_sets needs the per-subsample selections that stabsel "
            "keeps, and this result has none. Run stabsel on a GAMBoostModel or "
            "GLMBoostModel; distributional (LSS) stability results do not keep "
            "them."
        )
    if not (0 < min_share <= 1) or not (0 <= max_joint <= 1):
        raise ValueError("min_share must lie in (0, 1] and max_joint in [0, 1]")
    names = list(stability.learner_names)
    B = len(subsample_sets)
    selected = [set(s) for s in subsample_sets]
    freq = [sum(i in s for s in selected) / B for i in range(len(names))]
    candidates = [i for i in range(len(names)) if freq[i] >= min_share]

    pairs = []
    edges = []
    for a, b in combinations(candidates, 2):
        both = sum(a in s and b in s for s in selected) / B
        either = sum(a in s or b in s for s in selected) / B
        if both > max_joint * min(freq[a], freq[b]):
            continue
        overlap = None
        if equivalence is not None:
            overlap = equivalence.overlap.get((names[a], names[b]),
                                              equivalence.overlap.get((names[b], names[a])))
            if overlap is None or overlap < min_overlap:
                continue
        pairs.append({
            "a": names[a], "b": names[b],
            "frequency_a": freq[a], "frequency_b": freq[b],
            "frequency_both": both, "frequency_either": either,
            "overlap": overlap,
        })
        edges.append((a, b))
    pairs.sort(key=lambda row: (row["frequency_both"], row["a"], row["b"]))

    groups = []
    for members in _components(len(names), edges):
        group_frequency = sum(any(i in s for i in members) for s in selected) / B
        groups.append(SubstitutionGroup(
            members=tuple(names[i] for i in members),
            group_frequency=group_frequency,
            member_frequency={names[i]: freq[i] for i in members},
        ))
    return SubstitutionSets(
        groups=tuple(groups), pairs=tuple(pairs), subsamples=B,
        min_share=float(min_share), max_joint=float(max_joint),
        min_overlap=None if equivalence is None else float(min_overlap))


def selection_report(model, *, deselection=None, stability=None,
                     equivalence_threshold=0.9, min_share=0.1, max_joint=0.2,
                     min_overlap=0.5):
    """Equivalence sets, substitution sets and deselection, read together.

    Parameters
    ----------
    model : GAMBoostModel or GLMBoostModel
        The fit the learners belong to.
    deselection : Deselection, optional
        A result of :func:`~mboostrs.deselect` on ``model``. Its dropped
        learners are classified against the equivalence sets.
    stability : StabSelResult, optional
        A stability selection run on ``model``, for the substitution sets. The
        substitution pairs are screened by the overlap in the equivalence sets.
    equivalence_threshold, min_share, max_joint, min_overlap
        Passed to :func:`equivalence_sets` and :func:`substitution_sets`.

    Returns
    -------
    SelectionReport
    """
    from mboostrs.selection import _bundle, _term_learners, _terms

    equivalence = equivalence_sets(model, threshold=equivalence_threshold)
    substitution = None
    if stability is not None:
        substitution = substitution_sets(stability, min_share=min_share,
                                         max_joint=max_joint, equivalence=equivalence,
                                         min_overlap=min_overlap)
    with_substitute, alone = [], []
    if deselection is not None:
        bundle = _bundle(model)
        _, terms = _terms(bundle["raw_formula"], bundle["raw_data"])
        mapping = _term_learners(model, terms)
        kept_names = {name for term in deselection.kept for name in mapping.get(term, [])}
        for term, share in deselection.dropped:
            for name in mapping.get(term, []):
                group = equivalence.group_of(name)
                kept_equivalent = sorted(
                    member for member in (group or ()) if member != name and member in kept_names
                )
                if kept_equivalent:
                    with_substitute.append({"dropped": term, "share": float(share),
                                            "kept_equivalent": kept_equivalent})
                else:
                    alone.append({"dropped": term, "share": float(share)})
    return SelectionReport(
        equivalence=equivalence,
        substitution=substitution,
        dropped_with_substitute=tuple(with_substitute),
        dropped_alone=tuple(alone),
    )
