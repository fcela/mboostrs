"""Choosing base learners from the data.

A `gamboost` formula names one base learner per term, and which learner suits a
column depends on what the column is. A two-valued indicator has nothing for a
spline to smooth; a variable with a threshold in it is fitted better by a tree
than by a P-spline; a high-cardinality grouping wants a random effect. Written
by hand for a wide frame this is a long string that is easy to get wrong, and
getting it wrong is not cosmetic — a tensor product of two spline bases on two
indicator columns is both degenerate and expensive.

`suggest_terms` applies that reasoning and returns the right-hand side.
`expand_pairs` does the same for two-way interactions. `screen_interactions`
ranks candidate pairs before any model is fitted, so a formula can carry the
few that look worth carrying rather than all p(p-1)/2 of them.

Nothing here fits a model or makes a decision that cannot be overridden: each
function returns strings or a ranking, which the caller is free to edit before
passing to `gamboost`.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace

import numpy as np

from mboostrs._names import quote_name

#: A column with at most this many distinct values is treated as categorical
#: rather than continuous.
CATEGORICAL_MAX_LEVELS = 12

#: Historical threshold retained for compatibility. Multi-level factors now
#: always use `brandom`: the current `bols` path does not indicator-code them,
#: and on numeric codes it fits one slope rather than one mean per level.
RANDOM_EFFECT_MIN_LEVELS = 6

DEFAULT_DF = 4.0

#: Dimension of the null space of a two-margin second-difference tensor
#: penalty: constants and straight lines in each margin, so 2 x 2. A tensor
#: term given this many degrees of freedom or fewer is at maximum shrinkage and
#: can only produce a bilinear surface.
TENSOR_MINIMUM_DF = 4.0

#: Marginal df for a tensor term. Squared, it must clear TENSOR_MINIMUM_DF.
TENSOR_MARGINAL_DF = 2.5

#: A whole-number column with more levels than CATEGORICAL_MAX_LEVELS is read
#: as labels when its level means vary more than sampling noise explains, by at
#: least this many standard deviations of the ratio under no signal ...
LABEL_SIGNAL_Z = 4.0

#: ... and when codes next to each other differ, on average, at least this
#: much relative to two arbitrary codes. The ratio is about 1 for labels,
#: whose order carries no information, and about 0 for a measurement whose
#: effect changes smoothly with its value.
LABEL_ADJACENCY_RATIO = 0.5

#: ... and when the values' order is not detectable: for label codes the
#: observed order is one random order of the level means, so a test against
#: shuffled orders rejects it at this rate. The adjacency ratio alone cannot
#: carry the decision -- its spread for labels depends on the number of
#: levels and the signal (1% quantile above 0.7 at 100+ levels, near 0.2 at
#: 20 levels and a weak signal) -- and a skewed measurement whose levels
#: mostly hold two or three rows has noisy neighbours and a ratio near 0.6.
LABEL_ORDER_ALPHA = 0.01
LABEL_ORDER_PERMUTATIONS = 199

#: A whole-number column with more distinct values than this is never read as
#: labels. A `brandom` term has one column per level, so each refit costs about
#: n x levels^2: a column of 1,779 distinct counts (the claims benchmark's
#: total_bedrooms) made each 200-step gamboost fit on 15,000 rows take 15 to 33
#: seconds, against 0.24 seconds with that term left out. The label
#: test also flagged that column on 3 of 180 permuted responses, and its
#: z-statistic under permutation had a standard deviation of 1.4 to 1.9, not 1.
LABEL_MAX_LEVELS = 200

#: The share of rows whose value occurs at least twice, below which a
#: whole-number column is too close to unique for level means to be compared.
LABEL_MIN_COVERAGE = 0.8

#: Column-name fragments that say a column counts something (`kid5`,
#: `n_visits`, `age` in years) even when its values are sparse over the
#: range they span. Matched case-insensitively, anywhere in the name.
COUNT_NAME_FRAGMENTS = ("n_", "num", "count", "kid", "child", "ment",
                        "visits", "age")

#: Column-name fragments that say a whole-number column is a physical,
#: temporal, or clinical measurement rather than nominal group codes.
MEASUREMENT_NAME_FRAGMENTS = (
    "age", "year", "yr", "month", "day", "hour", "minute", "week",
    "score", "rate", "temp", "weight", "height", "bmi", "bp",
    "glucose", "insulin", "chol", "duration", "tenure", "income",
    "price", "cost", "salary", "distance", "length", "width",
    "depth", "area", "volume", "speed", "pressure", "humidity",
    "wind", "pm25", "pm10",
)

_EXPLICIT_CODE_RE = re.compile(
    r"(?:^|_)(?:id|code|zip|fips|district|cluster|group|site|ward|region)(?:$|_)"
)


def _is_explicit_code_name(name: str) -> bool:
    """True when ``name`` carries an explicit identifier, code, or group token."""
    return bool(_EXPLICIT_CODE_RE.search(name.lower()))


def count_like(column, levels):
    """(min, max) when a whole-number column reads as a count or bounded measurement, else None.

    A count column -- children under five, outpatient visits, an age in
    whole years -- is a number: ``bols`` and ``bbs`` extend its effect to a
    count the fit never saw, while ``brandom`` fits one coefficient per
    value seen in training and ``predict()`` raises on a new one (issue
    #313: a test row with ``kid5 = 4`` could not be scored). The reading is
    structural, so it comes before both the level-count reading and
    `label_evidence`: every value a non-negative whole number, at least three
    distinct values, and either:

    - minimum 0 or 1, with values filling at least half of the integer range
      or matching `COUNT_NAME_FRAGMENTS`, or
    - minimum > 1 (non-zero-origin bounded integer measurement such as age
      16..90 or calendar year), not named as an explicit code/ID token, and
      either matching `COUNT_NAME_FRAGMENTS`/`MEASUREMENT_NAME_FRAGMENTS` or
      spanning a dense contiguous range (span <= 200, > `CATEGORICAL_MAX_LEVELS`
      levels, and >= 50% integer fill).

    A string, enum or declared-categorical column is never a count: its codes
    are labels whatever they look like.

    ``column`` is a `mboostrs._frame.Column` and ``levels`` its count of
    distinct non-missing values.
    """
    if column.kind != "numeric" or levels < 3:
        return None
    present = column.values[~np.isnan(column.values)]
    if not present.size:
        return None
    if not np.all(present == np.round(present)):
        return None
    low, high = float(np.min(present)), float(np.max(present))
    if low < 0:
        return None
    lowered = column.name.lower()
    if low <= 1:
        if levels < 0.5 * (high - low + 1):
            if not any(fragment in lowered for fragment in COUNT_NAME_FRAGMENTS):
                return None
        return low, high
    if _is_explicit_code_name(column.name):
        return None
    all_fragments = (*COUNT_NAME_FRAGMENTS, *MEASUREMENT_NAME_FRAGMENTS)
    if any(fragment in lowered for fragment in all_fragments):
        return low, high
    if (
        levels > CATEGORICAL_MAX_LEVELS
        and (high - low) <= 200.0
        and levels >= 0.50 * (high - low + 1.0)
    ):
        return low, high
    return None


@dataclass(frozen=True)
class LabelEvidence:
    """Whether a whole-number column's values act as labels or as a measurement.

    A column of 40 integer codes and a column of ages in whole years look the
    same to a dtype rule. The response tells them apart without fitting a
    model. Take the mean response at each value. For a grouping code the means
    differ from level to level and neighbouring codes are no more alike than
    any two codes; for a measurement with a smooth effect, neighbouring values
    have similar means. Both statistics are corrected for the noise in a mean
    of ``n`` rows, using the pooled within-level variance.
    """

    column: str
    levels: int
    #: Variance of the level means over the variance sampling noise alone
    #: gives them (the one-way analysis-of-variance F). Near 1 when the column
    #: carries no signal.
    signal_ratio: float
    #: How far `signal_ratio` is above 1 in standard deviations of its
    #: no-signal distribution, ``(F - 1) * sqrt((levels - 1) / 2)``. With many
    #: levels a small excess is decisive: 400 district means that vary 1.5
    #: times what noise explains are about 7 standard deviations out.
    signal_z: float
    #: Mean squared difference between the means of adjacent codes, noise
    #: removed, over twice the between-level variance, noise removed. About 1
    #: for labels, about 0 for a smooth measurement; NaN without signal.
    adjacency_ratio: float
    labels: bool
    #: Share of shuffled level orders whose pooled level means spread at
    #: least as much as the observed order's (the observed order counted as
    #: one of them); see `_order_p_value`. Small when neighbouring values
    #: have similar means: the values carry an order. NaN without signal.
    order_p: float = float("nan")

    @property
    def sentence(self) -> str:
        if self.labels:
            return (f"{self.levels} whole-number values behave as labels: the "
                    f"mean response varies between values "
                    f"{self.signal_ratio:.1f} times as much as noise explains "
                    f"({self.signal_z:.1f} standard deviations above no "
                    f"signal), and adjacent values differ as much as arbitrary "
                    f"pairs (adjacency ratio {self.adjacency_ratio:.2f})")
        if (np.isfinite(self.order_p) and self.adjacency_ratio >= LABEL_ADJACENCY_RATIO
                and self.order_p < LABEL_ORDER_ALPHA and self.signal_z >= LABEL_SIGNAL_Z):
            return (f"{self.levels} whole-number values do not behave as labels: "
                    f"adjacent values have closer mean responses (adjacency ratio "
                    f"{self.adjacency_ratio:.2f}) than in all but "
                    f"{self.order_p:.1%} of shuffled orders, so the values carry "
                    f"an order and are fitted as a measurement")
        return (f"{self.levels} whole-number values do not behave as labels "
                f"(signal ratio {self.signal_ratio:.1f}, "
                f"{self.signal_z:.1f} standard deviations; adjacency ratio "
                f"{self.adjacency_ratio:.2f})")


def _order_p_value(means, counts):
    """Share of shuffled level orders whose pooled means spread at least as
    much as the observed order's.

    Consecutive levels are pooled into about 20 blocks of equal row count and
    the row-weighted variance of the block means is the statistic. Pooling
    label codes averages their effects, so block means vary little in any
    order; pooling a measurement's neighbouring values keeps its trend, so
    the observed order spreads the block means more than a shuffle does.
    Neighbouring single levels are too noisy for this when most levels hold
    two or three rows. The generator is fixed, so the same frame always reads
    the same way.
    """
    k = means.size
    n_blocks = max(2, min(20, k // 3))
    total = float(counts.sum())

    def spread(order):
        c = counts[order]
        block = np.minimum((np.cumsum(c) - c / 2.0) * n_blocks // total,
                           n_blocks - 1).astype(int)
        rows = np.bincount(block, weights=c, minlength=n_blocks)
        sums = np.bincount(block, weights=c * means[order], minlength=n_blocks)
        used = rows > 0
        block_means = sums[used] / rows[used]
        grand = sums.sum() / total
        return float(np.sum(rows[used] * (block_means - grand) ** 2))

    observed = spread(np.arange(k))
    shuffles = np.random.default_rng(0)
    at_least = sum(spread(shuffles.permutation(k)) >= observed
                   for _ in range(LABEL_ORDER_PERMUTATIONS))
    return (at_least + 1) / (LABEL_ORDER_PERMUTATIONS + 1)


def label_evidence(name, values, response):
    """Read a whole-number column against the response; see `LabelEvidence`.

    Returns ``None`` when the question does not arise: a column that is not
    whole numbers, has at most `CATEGORICAL_MAX_LEVELS` values (those are
    already read as levels) or more than `LABEL_MAX_LEVELS` (too many levels
    to fit one column each), or has too few repeated values to compare level
    means. The statistics use the values seen at least twice, and only when
    those values hold at least `LABEL_MIN_COVERAGE` of the rows and average
    two or more rows each: a district code where a few districts appear once
    is still read, while a column of mostly unique integers is not.
    """
    values = np.asarray(values, dtype=float)
    response = np.asarray(response, dtype=float)
    if values.shape != response.shape:
        return None
    keep = np.isfinite(values) & np.isfinite(response)
    x, y = values[keep], response[keep]
    if x.size == 0 or not np.all(x == np.round(x)):
        return None
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if counts.size <= CATEGORICAL_MAX_LEVELS or counts.size > LABEL_MAX_LEVELS:
        return None
    repeated = counts[inverse] >= 2
    if repeated.mean() < LABEL_MIN_COVERAGE:
        return None
    x, y = x[repeated], y[repeated]
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    k = counts.size
    if k <= CATEGORICAL_MAX_LEVELS or x.size - k < k:
        return None
    means = np.bincount(inverse, weights=y) / counts
    within = float(np.sum((y - means[inverse]) ** 2) / (x.size - k))
    noise_of_means = within * float(np.mean(1.0 / counts))
    between = float(np.var(means, ddof=1))
    if noise_of_means <= 0:
        signal_ratio = np.inf if between > 0 else 1.0
    else:
        signal_ratio = between / noise_of_means
    signal = between - noise_of_means
    order_p = float("nan")
    if signal <= 0:
        adjacency = float("nan")
    else:
        noise_of_differences = within * float(np.mean(1.0 / counts[:-1] + 1.0 / counts[1:]))
        adjacency = float((np.mean(np.diff(means) ** 2) - noise_of_differences)
                          / (2.0 * signal))
        order_p = _order_p_value(means, counts)
    signal_z = float((signal_ratio - 1.0) * np.sqrt((k - 1) / 2.0))
    labels = bool(signal_z >= LABEL_SIGNAL_Z and np.isfinite(adjacency)
                  and adjacency >= LABEL_ADJACENCY_RATIO
                  and order_p >= LABEL_ORDER_ALPHA)
    low_x, high_x = float(np.min(x)), float(np.max(x))
    lowered_name = str(name).lower()
    all_fragments = (*COUNT_NAME_FRAGMENTS, *MEASUREMENT_NAME_FRAGMENTS)
    if (
        low_x >= 2.0
        and not _is_explicit_code_name(str(name))
        and (
            any(fragment in lowered_name for fragment in all_fragments)
            or (high_x - low_x <= 200.0 and k >= 0.50 * (high_x - low_x + 1.0))
        )
    ):
        labels = False
    return LabelEvidence(column=name, levels=int(k), signal_ratio=float(signal_ratio),
                         signal_z=signal_z, adjacency_ratio=adjacency, labels=labels,
                         order_p=order_p)


def _columns(data, response=None):
    """(name, values) for every predictor column, as float arrays.

    The encoding comes from `mboostrs._frame`, so a string, boolean, declared
    categorical or date column arrives as numbers here instead of raising
    ``could not convert string to float``.
    """
    from mboostrs._frame import frame_columns

    return [(column.name, column.values)
            for column in frame_columns(data, response)]


@dataclass(frozen=True)
class ColumnKind:
    """What a column looks like, the learner that follows, and what else it
    could be.

    ``term`` is a recommendation, and a recommendation with no stated
    alternative is an instruction. ``alternatives`` names the other learners
    that fit the same column and says what each one *assumes* -- the estimand
    it targets and what it gives up -- so the choice can be made rather than
    accepted. Everything here is a string the caller may edit before it
    reaches `gamboost`.
    """

    name: str
    levels: int
    kind: str          # "binary", "categorical", "continuous", "datetime"
    term: str
    #: What the column is made of: ``"numeric"``, ``"boolean"``,
    #: ``"categorical"`` or ``"datetime"``.
    dtype: str = "numeric"
    #: Level labels in code order, for a categorical column. A coefficient can
    #: then be reported against the level it belongs to rather than a number.
    level_names: tuple | None = None
    #: Whether the values carry an order a model may rely on.
    ordered: bool = True
    #: Missing values in the column.
    n_missing: int = 0
    #: For a date column, the unit the numeric axis is in.
    unit: str | None = None
    #: Why ``term`` was chosen, in one sentence.
    reason: str = ""
    #: ``(term, what it assumes)`` for each learner that also fits this
    #: column, in the order worth trying them.
    alternatives: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:      # pragma: no cover - display only
        return f"{self.name}: {self.kind} ({self.levels} levels) -> {self.term}"

    def explain(self) -> str:
        """The recommendation, its reason, and the alternatives, as text."""
        lines = [f"{self.name} ({self.dtype}, {self.levels} levels"
                 + (f", {self.n_missing} missing" if self.n_missing else "")
                 + ")",
                 f"  suggested: {self.term}",
                 f"    {self.reason}"]
        if self.level_names is not None:
            shown = ", ".join(repr(level) for level in self.level_names[:6])
            more = "" if len(self.level_names) <= 6 else ", ..."
            order = "declared order" if self.ordered else "no order"
            lines.append(f"  levels ({order}): {shown}{more}")
        for term, assumption in self.alternatives:
            lines.append(f"  or {term}")
            lines.append(f"    {assumption}")
        return "\n".join(lines)


def hinge(name, df=DEFAULT_DF, n_candidates=20):
    """The ``hinge(...)`` term for a formula, as a string.

    One boosting selection adds one hinge at a knot the fit places where
    the data bend.
    Use the term inside a ``gamboost`` formula (``"y ~ hinge(x, df=4)"``),
    or build it programmatically.

    Parameters
    ----------
    name : str
        The predictor column.
    df : float
        Degrees of freedom the ridge on each selected hinge is calibrated
        to (default 4).
    n_candidates : int
        Interior-quantile knots scanned per selection (default 20).

    Returns
    -------
    str

    Examples
    --------
    >>> hinge("dew_point")
    'hinge(dew_point, df=4.0, n_candidates=20)'
    """
    return f"hinge({name}, df={df}, n_candidates={n_candidates})"




def classify_columns(data, response=None, df=DEFAULT_DF, smooth="bbs"):
    """Describe each predictor and name the learner suggested for it.

    Parameters
    ----------
    data : dict or DataFrame
    response : str, optional
        Column to leave out. When it is numeric, it is also used to read a
        whole-number column with more than `CATEGORICAL_MAX_LEVELS` values:
        see `label_evidence`. A column whose values behave as labels gets
        ``brandom``; one that behaves as a measurement keeps the smooth.
        A column that reads as a count (non-negative whole numbers from 0
        or 1, at least three of them; see `count_like`) keeps a measurement
        term whatever the response says, because its effect has to extend
        to a count the fit never saw.
    df : float
        Degrees of freedom for the penalised learners.
    smooth : {"bbs", "btree", "hinge", "menu"}
        The learner for a continuous column. ``"bbs"`` is a P-spline;
        ``"btree"`` is a stump, which fits a threshold where a spline would
        smooth across it; ``"hinge"`` places a kink where the data bends,
        keeping both sides straight; ``"menu"`` offers ``bols``, ``bbs`` and
        ``btree`` for the column and leaves the choice to boosting's
        selection step.

    Returns
    -------
    list of ColumnKind
    """
    if smooth not in ("bbs", "btree", "hinge", "menu"):
        raise ValueError(
            f"smooth must be 'bbs', 'btree', 'hinge' or 'menu', got {smooth!r}"
        )
    from mboostrs._frame import frame_columns

    response_values = _numeric_response(data, response)
    kinds = [_classify_one(column, df, smooth, response_values)
             for column in frame_columns(data, response)]
    return _with_coordinate_reasons(kinds)


def _with_coordinate_reasons(kinds):
    """Name the joint tree alternative on a coordinate pair's two columns.

    The draft still fits each coordinate on its own, as a smooth. When two
    continuous columns read as coordinates (`autospec`'s pair names), each
    one's reason says what the joint alternatives are: ``bspatial(x, y)``, a
    smooth surface, and ``btree(x, y)``, a tree over the pair, which fits a
    patchy or bounded location effect better.
    """
    from mboostrs.autospec import _coordinate_pair_names

    pair = _coordinate_pair_names([k for k in kinds if k.kind == "continuous"])
    if pair is None:
        return kinds
    x, y = pair
    note = (f"; {x} and {y} read as coordinates, so each is fitted alone here. "
            f"For a location effect that is patchy or has boundaries, the joint "
            f"term btree({x}, {y}) is the tree alternative to the smooth surface "
            f"bspatial({x}, {y}); autospec with a budget measures both")
    return [replace(k, reason=k.reason + note) if k.name in pair else k
            for k in kinds]


def _numeric_response(data, response):
    """The response as floats, or None when it is absent or not numeric."""
    if response is None:
        return None
    from mboostrs._frame import column_names, read_column

    if response not in column_names(data):
        return None
    column = read_column(data, response)
    if column.kind not in ("numeric", "boolean"):
        return None
    return column.values


def _classify_one(column, df, smooth, response_values=None) -> ColumnKind:
    """One column's learner, its reason, and its alternatives."""
    name = column.name
    present = column.values[~np.isnan(column.values)]
    levels = int(np.unique(present).size)
    common = {
        "name": name, "levels": levels, "dtype": column.kind,
        "level_names": column.levels, "ordered": column.ordered,
        "n_missing": column.n_missing, "unit": column.unit,
    }

    if column.kind == "datetime":
        return ColumnKind(
            kind="datetime", term=f"bbs({quote_name(name)}, df={df})",
            reason=(f"a date is a continuous axis (the numbers are "
                    f"{column.unit}), so a smooth over it is a trend that may "
                    f"bend"),
            alternatives=(
                (f"bols({quote_name(name)})",
                 "one slope: a constant rate of change over the whole period, "
                 "which is what to fit when the question is 'how fast', not "
                 "'what shape'"),
                (f"btree({quote_name(name)})",
                 "thresholds: a level shift at a date, for a policy change or "
                 "a regime break rather than a gradual drift"),
            ),
            **common)

    if column.kind == "categorical":
        shown = levels
        if column.ordered:
            score = f"{name}_score"
            return ColumnKind(
                kind="categorical", term=f"brandom({quote_name(name)}, df={df})",
                reason=("the fit path can model the declared labels only as "
                        "shrunken level effects; this valid default does not "
                        "use their declared order"),
                alternatives=(
                    (f"bols({quote_name(score)})",
                     "after explicitly adding numeric scores: one slope that "
                     "assumes adjacent levels are equally spaced"),
                    (f"bbs({quote_name(score)}, df={df})",
                     "after explicitly adding numeric scores: a curve over "
                     "the declared order, still assuming meaningful spacing"),
                    (f"bmono({score}, df={df})",
                     "after explicitly adding numeric scores: a monotone curve; "
                     "choose and state increasing or decreasing direction"),
                ),
                **common)
        return ColumnKind(
            kind="categorical", term=f"brandom({quote_name(name)}, df={df})",
            reason=(f"{shown} unordered levels: the group learner estimates "
                    "one effect per level and shrinks thinly observed levels "
                    "toward the overall mean"),
            alternatives=(), **common)

    if column.kind == "boolean" or levels <= 2:
        return ColumnKind(
            kind="binary", term=f"bols({quote_name(name)})",
            reason=("two values: the effect is one difference between them, "
                    "and there is nothing for a spline to smooth"),
            alternatives=(), **common)

    # A whole-number column that reads as a count is a measurement, not a set
    # of labels: its effect has to extend to a count the fit never saw, and
    # `brandom` has no coefficient for one (#313).
    count = count_like(column, levels)
    if count is not None:
        low, high = count
        smooth_df = min(df, levels - 1)
        reason = (f"{levels} whole-number values from {low:g} to {high:g} that "
                  "read as a count: one effect that extends to unseen "
                  "counts, not one level per value")
        if levels <= 4:
            return ColumnKind(
                kind="continuous", term=f"bols({quote_name(name)})", reason=reason,
                alternatives=(
                    (f"bbs({quote_name(name)}, df={smooth_df})",
                     "a curve over the counts: right when the effect bends or "
                     "saturates as the count grows, wrong when each extra "
                     "count adds the same difference"),
                    (f"brandom({quote_name(name)}, df={df})",
                     "if the codes are labels rather than counts: one shrunken "
                     "effect per value; predict() fails on a value not seen "
                     "in training"),
                ),
                **common)
        return ColumnKind(
            kind="continuous", term=f"bbs({quote_name(name)}, df={smooth_df})",
            reason=reason,
            alternatives=(
                (f"bols({quote_name(name)})",
                 "one slope across the counts: the cheapest shape, right when "
                 "each extra count adds the same difference"),
                (f"brandom({quote_name(name)}, df={df})",
                 "if the codes are labels rather than counts: one shrunken "
                 "effect per value; predict() fails on a value not seen in "
                 "training"),
            ),
            **common)

    # A numeric column is only treated as categorical if its values are whole
    # numbers. A continuous measurement that happens to take few distinct
    # values in a small sample is still continuous, and coding it as levels
    # would throw away the ordering.
    discrete = bool(np.all(present == np.round(present)))
    if discrete and levels <= CATEGORICAL_MAX_LEVELS:
        return ColumnKind(
            kind="categorical", term=f"brandom({quote_name(name)}, df={df})",
            reason=(f"{levels} whole-number values: read as levels rather than "
                    "a measurement, so the model needs one shrunken effect per "
                    "value rather than one slope across the codes"),
            alternatives=(
                (f"bols({quote_name(name)})",
                 "treat the codes as an equally spaced ordinal/count scale and "
                 "fit one linear slope, not separate level means"),
                (f"bbs({quote_name(name)}, df={df})",
                 "treat the codes as a measurement and fit a smooth over "
                 "them: right when they are counts or a graded scale, wrong "
                 "when they are labels that happen to be numbers"),
            ),
            **common)

    if discrete and response_values is not None:
        evidence = label_evidence(name, column.values, response_values)
        if evidence is not None and evidence.labels:
            return ColumnKind(
                kind="categorical", term=f"brandom({quote_name(name)}, df={df})",
                reason=(evidence.sentence + ", so the model needs one "
                        "shrunken effect per value; evaluate it on folds "
                        "that hold whole groups out"),
                alternatives=(
                    (f"bbs({quote_name(name)}, df={df})",
                     "treat the values as a measurement and fit a smooth "
                     "over them: right for counts or ages, wrong for labels"),
                ),
                **common)

    if smooth == "bbs":
        term = f"bbs({quote_name(name)}, df={df})"
        reason = ("a continuous measurement: a P-spline fits a curve and is "
                  "penalised toward a straight line, so it costs little when "
                  "the effect is in fact straight")
    elif smooth == "btree":
        term = f"btree({quote_name(name)})"
        reason = ("a continuous measurement, fitted as thresholds: a stump "
                  "finds a cut point where a spline would smooth across it")
    else:
        term = f"bols({quote_name(name)}) + bbs({quote_name(name)}, df={df}) + btree({quote_name(name)})"
        reason = ("all three shapes offered at once, so boosting's selection "
                  "step picks per iteration rather than the specification "
                  "deciding in advance. Give them equal df or the comparison "
                  "is biased toward the flexible one")
    alternatives = tuple(
        (candidate, assumption)
        for candidate, assumption in (
            (f"bols({quote_name(name)})",
             "one slope: the effect is assumed straight, which is the "
             "cheapest thing to select and the easiest to report"),
            (f"bbs({quote_name(name)}, df={df})",
             "a curve that may bend, penalised toward a straight line"),
            (f"btree({quote_name(name)})",
             "axis-aligned thresholds: fits a jump, and cannot express a "
             "gradual change without many steps"),
        )
        if candidate not in term
    )
    return ColumnKind(kind="continuous", term=term, reason=reason,
                      alternatives=alternatives, **common)


def suggest_terms(data, response=None, df=DEFAULT_DF, smooth="bbs"):
    """The main-effect right-hand side suggested for ``data``.

    Examples
    --------
    >>> suggest_terms({"age": [30, 41, 52], "sex": [0, 1, 0], "y": [1.0, 2.0, 3.0]},
    ...               response="y")
    'bbs(age, df=4.0) + bols(sex)'
    """
    return " + ".join(c.term for c in classify_columns(data, response, df, smooth))


def pair_term(left, right, levels, df=DEFAULT_DF, marginal_df=TENSOR_MARGINAL_DF,
              knots=5, kind="tensor", maxdepth=4):
    """The learner for one two-way interaction.

    ``kind="tensor"`` builds a spline surface for two continuous margins, and
    falls back to a varying coefficient or a product where a margin is
    categorical, because a spline basis on a two-valued column is degenerate.
    ``kind="tree"`` builds a two-variable tree, which fits axis-aligned
    thresholds rather than a smooth surface and costs no dense design matrix.

    Marginal degrees of freedom multiply, so two ``df=2`` margins give a term
    carrying df=4. That number is not free to choose: a second-difference
    penalty leaves constants and straight lines unpenalised, so each margin has
    a two-dimensional null space and their tensor has a four-dimensional one.
    A term asked for df=4 is asked for exactly its null space, lambda is not
    identified there, and the fit collapses onto that null space — a bilinear
    surface a + bx + cy + dxy, whatever the data look like. Measured on a
    sin(1.5x)cos(1.5y) surface over a uniform square, n=900: a df=4 term
    explains R-squared 0.0001 of it and lies in the null space to 1e-10,
    boosted or in a single solve.

    How fast a term escapes the null space above the floor depends on both the
    design and the path, so the figures below bound nothing in general. On that
    same design a boosted fit (600 steps, nu=0.1) explains 0.59 at df=4.2 and
    0.92 at df=9, where a single penalised solve at the same lambda explains
    0.02 and 0.50 — boosting accumulates curvature over steps that one solve
    does not. On a clustered design the escape is far slower again. The floor
    itself is exact and design-independent; everything above it is not.

    So `TENSOR_MINIMUM_DF` is a hard floor, not a guideline, and the default
    sits above it. The consequence worth knowing is that a tensor surface
    cannot be given the same degrees of freedom as a ``df=4`` main effect and
    still be a surface; if unbiased selection against df=4 learners matters
    more than curvature, use ``kind="tree"`` or a varying coefficient instead.
    """
    if kind == "tensor" and levels[left] > 2 and levels[right] > 2:
        if marginal_df * marginal_df <= TENSOR_MINIMUM_DF:
            warnings.warn(
                f"tensor term {left} %X% {right} asks for "
                f"df={marginal_df * marginal_df:g}, at or below the "
                f"{TENSOR_MINIMUM_DF:g}-dimensional null space of its penalty. "
                f"lambda is not identified there and the fit collapses to a "
                f"bilinear surface. Use marginal_df above "
                f"{np.sqrt(TENSOR_MINIMUM_DF):g}.",
                UserWarning, stacklevel=2)
    if kind == "tree":
        return f"btree({quote_name(left)}, {quote_name(right)}, maxdepth={maxdepth})"
    if kind != "tensor":
        raise ValueError(f"kind must be 'tensor' or 'tree', got {kind!r}")
    left_continuous = levels[left] > 2
    right_continuous = levels[right] > 2
    if left_continuous and right_continuous:
        return (f"bbs({quote_name(left)}, df={marginal_df}, knots={knots})"
                f" %X% bbs({quote_name(right)}, df={marginal_df}, knots={knots})")
    if left_continuous:
        return f"bbs({quote_name(left)}, by={quote_name(right)}, df={df})"
    if right_continuous:
        return f"bbs({quote_name(right)}, by={quote_name(left)}, df={df})"
    return f"bols({quote_name(left)}, by={quote_name(right)})"


def pair_terms(left, right, levels, df=DEFAULT_DF, marginal_df=TENSOR_MARGINAL_DF,
               knots=5, kind="tensor", maxdepth=4):
    """Every learner worth trying for one two-way interaction, cheapest check
    first included.

    For two continuous columns and ``kind="tensor"`` this returns three
    candidates in order:

    1. the tensor-product spline surface ``pair_term`` already builds;
    2. ``bols(left, by=right)``, a varying coefficient: the effect of ``left``
       varies linearly with ``right``;
    3. ``bols(right, by=left)``, the other direction.

    The tensor is a 6.25-df learner at the default ``marginal_df=2.5``. A
    bilinear signal — a product term, the interaction a varying coefficient
    represents with one degree of freedom — can be a genuine effect and still
    sit only one standard error under that surface, so a screen that proposes
    the tensor alone never offers the cheap check that would find it. Issue
    #315: the published ``dist x educ`` interaction in the Bangladesh wells
    data is that signal, and offering all three candidates lets the
    selection/scoring step (`compare_specifications`) decide which learner
    earns its degrees of freedom.

    For every other combination of column types, and for ``kind="tree"``, the
    answer is a single-element list holding exactly `pair_term`'s output:
    those shapes have no cheap bilinear rival to offer, and delegating keeps
    the two functions from drifting apart. `pair_term` itself is unchanged.
    """
    tensor = pair_term(left, right, levels, df, marginal_df, knots, kind, maxdepth)
    if (kind == "tensor" and levels.get(left, 0) > 2 and levels.get(right, 0) > 2):
        return [tensor,
                f"bols({quote_name(left)}, by={quote_name(right)})",
                f"bols({quote_name(right)}, by={quote_name(left)})"]
    return [tensor]


def expand_pairs(data, response=None, pairs=None, df=DEFAULT_DF,
                 marginal_df=TENSOR_MARGINAL_DF, knots=5, kind="tensor",
                 maxdepth=4):
    """Interaction terms for ``pairs``, or for every pair when none are given.

    ``pairs`` is a sequence of column-name pairs. The cost of the default is
    quadratic in the number of columns — a 21-column frame has 210 pairs, each
    of which the selection step scores at every iteration — so pass the output
    of `screen_interactions` rather than the default on anything wide.
    """
    columns = _columns(data, response)
    levels = {name: int(len(np.unique(values[~np.isnan(values)])))
              for name, values in columns}
    if pairs is None:
        names = [name for name, _ in columns]
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    return [pair_term(a, b, levels, df, marginal_df, knots, kind, maxdepth)
            for a, b in pairs]


@dataclass(frozen=True)
class ScreenedPair:
    """One candidate interaction and the mean squared error it removed."""

    left: str
    right: str
    removed: float

    def __repr__(self) -> str:      # pragma: no cover - display only
        return f"{self.left} x {self.right}: {self.removed:+.4g}"


def pearson_residual(fit, data, response):
    """The Pearson residual of ``fit`` on ``data``, one float per row.

    A response-scale residual ``y - predict(type="response")`` has a variance
    that depends on the fitted mean for every non-Gaussian family, so the
    rows are not on a common scale: for ``Binomial`` the variance of ``y - p``
    is ``p(1-p)``, largest at ``p = 0.5``, and a screen that sums squared
    residuals over rows is dominated by those rows rather than by structure.
    The Pearson residual divides the raw residual by its standard deviation,
    putting every row on unit variance so the screen measures structure.
    The formulas, with ``p`` and ``mu`` the fitted response-scale means:

    - ``Binomial``: ``(y - p) / sqrt(p (1 - p))``, ``p`` clipped to
      ``[1e-6, 1 - 1e-6]``.
    - ``Poisson``: ``(y - mu) / sqrt(mu)``, ``mu`` clipped below at ``1e-6``.
    - ``Gaussian`` / ``GaussReg``: the ordinary residual ``y - predict()``
      (identity link, so it is also the response-scale residual).
    - any other family: a `UserWarning` that Pearson residuals are not defined
      for it, falling back to ``y - predict(type="response")``.

    Returns a 1-D ``float`` numpy array.
    """
    from mboostrs._frame import read_column
    from mboostrs.families import Binomial, Gaussian, GaussReg, Poisson

    y = np.asarray(read_column(data, response).values, dtype=float).ravel()
    family = fit.family
    if isinstance(family, Binomial):
        p = np.clip(np.asarray(fit.predict(type="response"), dtype=float).ravel(),
                    1e-6, 1 - 1e-6)
        return (y - p) / np.sqrt(p * (1 - p))
    if isinstance(family, Poisson):
        mu = np.clip(np.asarray(fit.predict(type="response"), dtype=float).ravel(),
                     1e-6, None)
        return (y - mu) / np.sqrt(mu)
    if isinstance(family, (Gaussian, GaussReg)):
        return y - np.asarray(fit.predict(), dtype=float).ravel()
    warnings.warn(
        f"Pearson residuals are not defined for "
        f"{type(family).__name__}; falling back to the response-scale "
        f"residual y - predict(type='response').",
        UserWarning, stacklevel=2)
    return y - np.asarray(fit.predict(type="response"), dtype=float).ravel()


def screen_interactions(data, residual, response=None, bins=4, seed=0, pairs=None,
                        groups=(), na_action="error"):
    """Rank candidate pairs by the residual variation they explain out of sample.

    Both columns of a pair are binned into ``bins`` quantile cells on half the
    rows, the cell means of ``residual`` are taken there, and the mean squared
    error those means remove is measured on the other half. A pair that fits
    only noise removes nothing out of sample, so its expected score is below
    zero (about minus the cell count over the fit rows) and the ranking is not
    a count of flexibility. A single score can exceed zero by chance. No model
    is fitted.

    The half-split is a seeded permutation of the rows. A fixed seed gives
    the same scores across runs and thread counts.

    ``residual`` is what the additive fit left behind, on the response
    scale: ``y - model.predict(type="response")``. For a Gaussian fit that is
    ``y - model.predict()``. For a family with a non-identity link (Binomial,
    Poisson, Gamma) the default ``predict()`` is the link-scale predictor, and
    subtracting it from the response leaves curvature that every pair of
    signal columns appears to explain.

    ``removed`` is a mean squared error, per held-out observation, so it is on
    the same scale as the difference between two out-of-fold risks and can be
    ranked alongside them.

    A column named in ``groups`` is not binned: each of its levels is its own
    cell. Quantile bins of a grouping code put unrelated groups in one cell, so
    a slope that differs from group to group averages out and a random slope
    goes undetected.

    Missing values (NaN, or infinity) are handled by ``na_action``, with the
    meanings the fit functions give it:

    * ``"error"`` (default): raise, naming the column, the number of missing
      values, the first row that has one, and the remedy.
    * ``"impute"``: a numeric column's missing values become its median. A
      grouped column's missing level is a group of its own, as the fit
      functions give a categorical predictor's missing level.
    * ``"omit"``: rows with a missing value in any screened column, or a
      non-finite ``residual``, are dropped before the screen. The residual
      must then be the one for the same rows as ``data``.

    ``residual`` must be finite under ``"error"`` and ``"impute"``.

    Returns
    -------
    list of ScreenedPair
        Sorted by ``removed``, descending; equal scores tie by the pair's
        column order. Callers choose the cutoff; a score at or below zero is
        evidence against the pair, not merely a weak ranking.
    """
    if na_action not in ("error", "omit", "impute", "impute_model", "impute_model_split"):
        raise ValueError(
            "na_action must be 'error', 'omit', 'impute', 'impute_model' or 'impute_model_split', "
            f"got {na_action!r}")
    columns = _columns(data, response)
    names = [name for name, _ in columns]
    values = np.column_stack([v for _, v in columns]).astype(float)
    residual = np.asarray(residual, dtype=float).ravel()
    if len(residual) != values.shape[0]:
        raise ValueError(
            f"residual has {len(residual)} rows, data has {values.shape[0]}")

    group_names = set(groups or ())
    values, residual = _screen_missing(values, residual, names, group_names, na_action)

    # A grouped column crosses into Rust as level codes, so a missing level
    # keeps its own cell exactly as the numpy pass's np.unique coding gave
    # it (NaN sorts last, so it becomes the highest code).
    factor_levels = [0] * values.shape[1]
    for j, name in enumerate(names):
        if name in group_names:
            _, codes = np.unique(values[:, j], return_inverse=True)
            values[:, j] = codes
            factor_levels[j] = int(codes.max()) + 1

    index = {name: j for j, name in enumerate(names)}
    if pairs is None:
        wanted = [(index[a], index[b])
                  for i, a in enumerate(names) for b in names[i + 1:]]
    else:
        wanted = [(index[left], index[right]) for left, right in pairs]
    if not wanted:
        return []

    from mboostrs._rust import screen_pairs
    rows = screen_pairs(
        np.ascontiguousarray(values, dtype=float),
        np.ascontiguousarray(residual),
        bins=bins, seed=int(seed), factor_levels=factor_levels,
        pairs=wanted, top_k=None)
    return [ScreenedPair(left=names[a], right=names[b], removed=float(removed))
            for a, b, removed in rows]


# Typing discriminator tuning. _TYPE_GAP_K1 and _TYPE_GAP_K2 are the two
# scales (nearest-point counts) for the scale-invariance test. A true jump
# shows the same level gap at both scales (ratio ~1); a steep smooth
# transition or kink shows a gap that grows with scale (ratio ~2+), because
# wider-spaced points span more of the slope. _TYPE_GAP_W is the half-width
# of the threshold window searched for robustness to threshold error.
# _TYPE_RATIO_CRIT is the ratio threshold.
_TYPE_GAP_K1 = 15
_TYPE_GAP_K2 = 60
_TYPE_GAP_W = 0.2
_TYPE_RATIO_CRIT = 1.5
_TYPE_GAP_T_CRIT = 4.0


@dataclass(frozen=True)
class DiscontinuityHit:
    """One evaluated boundary from :func:`screen_discontinuities`.

    ``kind`` is ``"axis"`` (a step in one column, ``column > threshold_x``)
    or ``"diagonal"`` (a step across a line in two columns,
    ``right > boundary_intercept + boundary_slope * left``, with
    ``direction_deg`` the direction of the projection the boundary is a
    threshold on). ``threshold`` is that threshold on the standardized
    projection; ``threshold_x``, ``boundary_intercept`` and
    ``boundary_slope`` are the same boundary on the original scale, with the
    fields a purely vertical or horizontal boundary cannot express left as
    ``NaN`` and named in ``line``.

    ``jump_fit`` and ``t_fit`` are the fit-half step estimate and its
    t-statistic; selection ranked candidates on ``t_fit`` alone, so these are
    the in-sample numbers. ``jump``, ``se`` and ``z`` are the same
    coefficient estimated on the other half of the rows, which selection
    never looked at, so ``z`` follows a t distribution with
    ``n_eval - (q + 1)`` degrees of freedom under the null of no
    jump anywhere in the candidate set (asymptotically standard normal). ``gain`` is the held-out
    mean-squared-error reduction of the fit-half step model against the
    fit-half base model, per held-out observation, on the same scale as the
    difference between two out-of-fold risks. ``type_z`` refits the step
    against a background augmented with smooth terms along the projection:
    large when the step survives (a jump), near zero when a smooth crossing
    explains it. ``type_ratio`` is the local discriminator: the ratio of the
    level gap at two scales (nearest 60 vs nearest 15 evaluation-half points
    per side), maximized over a window of thresholds around the fit-half
    estimate. ``type_gap_t`` is the t-statistic of the small-scale gap. A
    true jump shows a significant gap that persists across scales (large
    ``type_gap_t``, ratio ~1); a steep smooth transition or kink shows a gap
    that grows with scale (ratio ~2+), and a gentle smooth shows no
    significant gap at all -- neither is typed as a jump even though the
    global ``type_z`` may be large. ``n_above`` counts the rows on the far
    side of the boundary.
    """

    kind: str
    columns: tuple
    direction_deg: float
    threshold: float
    threshold_x: float
    boundary_intercept: float
    boundary_slope: float
    line: str
    n_above: int
    jump_fit: float
    t_fit: float
    jump: float
    se: float
    z: float
    gain: float
    type_z: float
    type_ratio: float
    type_gap_t: float

    @property
    def flagged(self) -> bool:
        """The screen's decision rule: ``|z| >= 2`` with a positive gain.

        Under the null of no jump anywhere this fires on about 4.6% of
        screens; a screen that does not flag is a first-class result that
        finds no screened evidence for a jump in these columns."""
        return abs(self.z) >= 2.0 and self.gain > 0.0

    @property
    def typed_jump(self) -> bool:
        """A flagged row with a genuine discontinuity: ``flagged``, a
        significant small-scale gap (``type_gap_t``), and a scale-invariant
        gap (``type_ratio`` below threshold). The gap t-test takes the
        largest discontinuity over a window around the fit-half threshold;
        a true jump shows a large gap that persists across scales (ratio
        ~1), while a steep smooth transition or kink shows a gap that grows
        with scale (ratio ~2+), and a gentle smooth shows no significant gap
        at all. Note the resolution limit: a smooth transition narrower
        than the small scale's span is indistinguishable from a jump (see
        the ``type_ratio`` discussion in :func:`screen_discontinuities`)."""
        return (self.flagged and np.isfinite(self.type_gap_t)
                and self.type_gap_t > _TYPE_GAP_T_CRIT
                and np.isfinite(self.type_ratio)
                and self.type_ratio < _TYPE_RATIO_CRIT)

    def __repr__(self) -> str:      # pragma: no cover - display only
        typed = f", type_z {self.type_z:+.2f}" if np.isfinite(self.type_z) else ""
        typed += f", type_gap_t {self.type_gap_t:+.1f}" if np.isfinite(self.type_gap_t) else ""
        typed += f", ratio {self.type_ratio:.2f}" if np.isfinite(self.type_ratio) else ""
        return (f"{self.line}: jump {self.jump:+.3f} (se {self.se:.3f}), "
                f"z {self.z:+.2f}, gain {self.gain:+.4g}{typed}")


def _quantile_cuts(values, cells):
    """Interior equal-count cutpoints: the k/K empirical quantiles, k=1..K-1
    (numpy 'linear' interpolation), deduplicated."""
    probs = np.arange(1, cells) / cells
    cuts = np.quantile(values, probs)
    return np.unique(cuts)


def screen_discontinuities(data, target, *, response=None, columns=None,
                           cells=20, directions=16, seed=0, pairs=None,
                           top_k=25, all_pairs_max_p=20,
                           fallback_top_columns=10, min_side=10,
                           refine_units=3, refine_factor=8, type_top=5,
                           na_action="error"):
    """Screen for axis-aligned and diagonal step discontinuities.

    The screen identifies the location and orientation of a candidate jump.
    It complements :func:`~mboostrs.reference_gap`, which measures how much
    structure a smooth model leaves unexplained.

    Two candidate families are scanned, all fitted as the step indicator
    ``1{u > c}`` against a smooth background by ordinary least squares:

    * ``axis``: steps in one column, ``x_j > c`` at ``cells`` interior
      quantile cutpoints, background ``[1, u, u^2]``;
    * ``diagonal``: steps across a line in two columns,
      ``cos(theta) z_j + sin(theta) z_k > c`` on standardized columns over
      ``directions`` evenly spread directions, background ``[1, z_j, z_k]``.
      All pairs are scanned while the column count is at most
      ``all_pairs_max_p``; beyond that, pairs among the
      ``fallback_top_columns`` columns whose axis scan scored best.

    The rows are split once (``seed``) into a fit half and an evaluation
    half. Every candidate is scored on the fit half only -- ``jump_fit`` and
    its t-statistic ``t_fit`` -- and selection ranks on that; the same step
    is then estimated on the untouched evaluation half, giving ``jump``,
    ``se`` and ``z``. Under the null of no jump anywhere in the candidate
    set the winning row's ``z`` follows a t distribution with
    ``n_eval - (q + 1)`` degrees of freedom (asymptotically standard normal),
    so the decision rule ``|z| >= 2`` with a positive held-out ``gain`` (the row's
    ``flagged`` property) fires on about 4.6% of null screens regardless of
    how many candidates were searched. ``t_fit`` and ``z`` are not the same
    quantity: ranking by the naive t-statistic of the winning boundary
    overstates, which is why selection and reporting use different halves.
    The scan uses Rust-accelerated kernels; one unit costs a sort and a
    prefix sum.

    ``type_z`` types the top rows globally: the step is refitted against a background
    augmented with the smooth terms along the projection (a quadratic and a
    cubic in it). ``type_ratio`` is the local discriminator that drives the
    ``typed_jump`` decision: the ratio of the level gap measured at two
    scales (nearest 60 vs nearest 15 evaluation-half points per side),
    maximized over a window around the fit-half threshold. A true jump is
    scale-invariant (ratio ~1); a steep smooth transition or kink has a gap
    that grows with scale (ratio ~2+), so the row is not typed as a jump.
    Resolution limit: the small scale sets the resolution -- a transition
    narrower than its span types as a jump, honestly, because the data
    cannot resolve it. A row with ``typed_jump`` true is a jump; a flagged
    row with a large ``type_ratio`` asks for a smooth or tensor candidate
    instead of a step indicator.

    Parameters
    ----------
    data : polars or pandas DataFrame, or a mapping of column to values
        The rows to screen. Numeric, boolean and categorical columns are read
        through their encodings, as every screen in this module does.
    target : array-like
        What to scan: the response, or -- the recommended second pass -- the
        response-scale residual of a fitted smooth model, which removes the
        smooth structure so the jump stands alone. Pass residuals on the
        response scale (``y - predict(type="response")``); a link-scale
        residual leaves curvature every column appears to explain.
    response : str, optional
        The response column, excluded from the predictors.
    columns : sequence of str, optional
        Screen only these columns; all non-response columns by default.
    cells : int
        Threshold cutpoints per direction, as interior quantiles of the fit
        half. Lowered automatically so small samples keep cells on both
        sides.
    directions : int
        Directions per column pair, evenly spread over the half circle.
    seed : int
        Seed of the fit/evaluation half split and of the tie order.
    pairs : sequence of (str, str), optional
        Screen only these column pairs (the axis family is unchanged).
    top_k : int
        Rows returned, best first. ``0`` returns all.
    all_pairs_max_p, fallback_top_columns : int
        The pair-scan switch: all pairs while there are at most
        ``all_pairs_max_p`` columns, otherwise pairs among the best
        ``fallback_top_columns`` axis columns.
    min_side : int
        Fewest rows a side of the boundary must hold on each half; a
        candidate that splits off fewer rows is skipped.
    refine_units, refine_factor : int
        The best ``refine_units`` diagonal units are re-scanned on a
        threshold grid twice as fine and sub-directions ``refine_factor``
        times as fine around the winner. All refinement reads fit-half
        quantities only, so the evaluation-half calibration is untouched.
    type_top : int
        How many of the best rows get the ``type_z`` typing refit.
    na_action : {"error", "omit", "impute"}
        Missing values in a screened column or the target: raise naming the
        column and row (default), drop the rows, or fill a column with its
        fit-half median (so the fill never reads eval-half rows) -- the
        target rows cannot be filled, pass a residual that is finite, or
        drop those rows.

    Returns
    -------
    list of DiscontinuityHit
        Best first by ``|t_fit|``. Read the winner's ``flagged`` and
        ``typed_jump``, and treat an unflagged screen as a result: it finds no
        screened evidence for a jump in these columns, and the missing structure a
        ``reference_gap`` still reports belongs to a smooth or interaction
        term, not a step.

    Examples
    --------
    A diagonal jump that no single-column step can express:

    >>> import numpy as np, polars as pl, mboostrs as mb
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> s0, s1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    >>> data = pl.DataFrame({
    ...     "s0": s0, "s1": s1, "noise": rng.uniform(0, 1, n),
    ...     "y": 2.0 * (s1 > 0.45 + 0.35 * s0) + 1.5 * s0
    ...          + 0.5 * rng.normal(size=n)})
    >>> hits = mb.screen_discontinuities(data, data["y"].to_numpy(), seed=0)
    >>> hits[0].typed_jump, hits[0].columns
    (True, ('s0', 's1'))
    """
    from mboostrs._rust import disc_scan, disc_step_z

    encoded = dict(_columns(data, response))
    names = list(encoded)
    if columns is not None:
        missing = [c for c in columns if c not in encoded]
        if missing:
            raise ValueError(f"columns {missing} are not in the data")
        names = list(columns)
    if not names:
        raise ValueError("the screen needs at least one predictor column")
    values = np.column_stack([encoded[name] for name in names]).astype(float)
    t = np.asarray(target, dtype=float).ravel()
    if len(t) != values.shape[0]:
        raise ValueError(
            f"target has {len(t)} rows, data has {values.shape[0]}")

    if na_action not in ("error", "omit", "impute"):
        raise ValueError(
            f"na_action must be 'error', 'omit' or 'impute', got {na_action!r}")
    bad = ~np.isfinite(values)
    bad_target = ~np.isfinite(t)
    if na_action == "omit":
        keep = ~(bad.any(axis=1) | bad_target)
        if not keep.any():
            raise ValueError(
                "na_action='omit' would drop every row of the discontinuity screen: "
                "every row has a missing value in a screened column or the target")
        values, t = values[keep], t[keep]
        bad = ~np.isfinite(values)
    elif bad_target.any():
        first = int(np.flatnonzero(bad_target)[0])
        raise ValueError(
            f"the target has {int(bad_target.sum())} non-finite value(s), first at "
            f"row {first}, and na_action={na_action!r} cannot fill a target. Pass "
            "na_action='omit' to drop those rows, or a residual that is finite on "
            "every row")

    n, p = values.shape
    if n < 80:
        raise ValueError(f"the screen needs at least 80 rows, got {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fit_rows, eval_rows = perm[:n // 2], perm[n // 2:]
    fit = np.zeros(n, dtype=np.uint8)
    fit[fit_rows] = 1

    # Imputation reads the fit half only: the fill value must not be a
    # function of eval-half rows, or the eval-half null calibration breaks.
    for j, name in enumerate(names):
        column_bad = bad[:, j]
        if not column_bad.any():
            continue
        if na_action == "error":
            first = int(np.flatnonzero(column_bad)[0])
            raise ValueError(
                f"column {name!r} has {int(column_bad.sum())} missing value(s) in the "
                f"discontinuity screen, first at row {first}. Pass "
                "na_action='impute' to fill them with the fit-half column median, "
                "na_action='omit' to drop those rows, or fill the column yourself "
                "before screening")
        fit_bad = column_bad[fit_rows]
        present = values[fit_rows[~fit_bad], j]
        if present.size == 0:
            raise ValueError(
                f"column {name!r} has no observed value on the fit half to impute "
                f"from in the discontinuity screen; drop the column or pass "
                f"na_action='omit'")
        values[column_bad, j] = float(np.median(present))

    mean = values[fit_rows].mean(axis=0)
    sd = values[fit_rows].std(axis=0, ddof=1)
    # A column with no usable variation cannot be standardized: exactly
    # constant, constant to machine precision relative to its own scale, or
    # non-finite spread (huge values whose variance overflows). The relative
    # threshold keeps a legitimately small-scaled column working while the
    # standardized values stay finite.
    scale = np.maximum(1.0, np.abs(mean))
    bad_sd = (sd <= 1e-12 * scale) | ~np.isfinite(sd)
    if np.any(bad_sd):
        raise ValueError(
            f"constant column(s) on the fit half: "
            f"{[names[j] for j in np.where(bad_sd)[0]]}. A column with no usable "
            f"variation cannot be standardized; drop it, rescale it, or pass "
            f"columns= to screen a subset")
    Z = (values - mean) / sd

    K = int(min(cells, max(4, n // 25)))
    D = int(directions)
    thetas = np.pi * np.arange(D) / D

    def unit_rows(u, background, cuts):
        return disc_scan(np.ascontiguousarray(u, dtype=float),
                         np.ascontiguousarray(background, dtype=float),
                         np.ascontiguousarray(t, dtype=float),
                         fit, np.ascontiguousarray(cuts, dtype=float),
                         int(min_side))

    def axis_unit(j, n_cells):
        u = Z[:, j]
        background = np.column_stack([np.ones(n), u, u * u])
        cuts = _quantile_cuts(u[fit_rows], n_cells)
        rows = []
        for (threshold, n_above, jump_fit, t_fit, jump, se, z, gain) in unit_rows(u, background, cuts):
            threshold_x = float(mean[j] + sd[j] * threshold)
            rows.append({
                "kind": "axis", "columns": (names[j],),
                "direction_deg": 0.0, "threshold": float(threshold),
                "n_above": int(n_above), "jump_fit": float(jump_fit),
                "t_fit": float(t_fit), "jump": float(jump), "se": float(se),
                "z": float(z), "gain": float(gain),
                "threshold_x": threshold_x,
                "boundary_intercept": float("nan"),
                "boundary_slope": float("nan"),
                "line": f"{names[j]} > {threshold_x:.4g}",
            })
        return rows

    rows = []
    for j in range(p):
        rows.extend(axis_unit(j, K))

    col_best = {}
    for r in rows:
        name = r["columns"][0]
        if name not in col_best or abs(r["t_fit"]) > abs(col_best[name]["t_fit"]):
            col_best[name] = r
    top_cols = sorted(col_best, key=lambda c: -abs(col_best[c]["t_fit"]))
    top_cols = top_cols[:min(fallback_top_columns, p)]

    if pairs is not None:
        wanted = [tuple(sorted(pr)) for pr in pairs]
    else:
        all_pairs = [(names[a], names[b])
                     for a in range(p) for b in range(a + 1, p)]
        if p <= all_pairs_max_p:
            wanted = all_pairs
        else:
            wanted = [(a, b) for ii, a in enumerate(top_cols)
                      for b in top_cols[:ii]]
            index = {c: i for i, c in enumerate(names)}
            wanted = [tuple(sorted(pr, key=lambda c: index[c])) for pr in wanted]

    index = {c: i for i, c in enumerate(names)}

    def diagonal_unit(ca, cb, theta, n_cells, label_deg=None):
        j, k = index[ca], index[cb]
        zj, zk = Z[:, j], Z[:, k]
        background = np.column_stack([np.ones(n), zj, zk])
        u = np.cos(theta) * zj + np.sin(theta) * zk
        cuts = _quantile_cuts(u[fit_rows], n_cells)
        out = []
        for (threshold, n_above, jump_fit, t_fit, jump, se, z, gain) in unit_rows(u, background, cuts):
            deg = label_deg if label_deg is not None else theta * 180 / np.pi
            ct, st = np.cos(theta), np.sin(theta)
            if abs(st) < 1e-12:
                thr_x, a_, b_ = float("nan"), float("nan"), float("nan")
                line = f"{ca} > {mean[j] + sd[j] * threshold:.4g}"
            elif abs(ct) < 1e-12:
                thr_x, a_, b_ = float("nan"), float("nan"), float("nan")
                line = f"{cb} > {mean[k] + sd[k] * threshold:.4g}"
            else:
                b_ = -(sd[k] * ct) / (sd[j] * st)
                a_ = (mean[k] + sd[k] * threshold / st
                      + sd[k] * ct * mean[j] / (sd[j] * st))
                thr_x = float("nan")
                line = f"{cb} > {a_:.4g} + {b_:.4g}*{ca}"
            out.append({
                "kind": "diagonal", "columns": (ca, cb),
                "direction_deg": float(deg), "threshold": float(threshold),
                "n_above": int(n_above), "jump_fit": float(jump_fit),
                "t_fit": float(t_fit), "jump": float(jump), "se": float(se),
                "z": float(z), "gain": float(gain),
                "threshold_x": thr_x, "boundary_intercept": float(a_),
                "boundary_slope": float(b_), "line": line,
            })
        return out

    for (ca, cb) in wanted:
        for d in range(D):
            rows.extend(diagonal_unit(ca, cb, thetas[d], K,
                                     label_deg=180.0 * d / D))

    # Fit-half-only refinement of the best diagonal units: a finer threshold
    # grid and sub-directions around each winner. The evaluation half stays
    # untouched, so the winner's z keeps its exact null calibration.
    if refine_units > 0 and D > 0:
        pool = sorted(rows, key=lambda r: -abs(r["t_fit"]))
        seen_units, to_refine = set(), []
        for r in pool:
            if r["kind"] != "diagonal":
                continue
            unit = (r["columns"], round(r["direction_deg"], 6))
            if unit in seen_units:
                continue
            seen_units.add(unit)
            to_refine.append(r)
            if len(to_refine) >= refine_units:
                break
        K_ref = int(min(2 * K, 40))
        sub = [(2 * m + 1) * np.pi / (D * refine_factor)
               for m in range(-(refine_factor // 2), (refine_factor + 1) // 2)]
        for r in to_refine:
            ca, cb = r["columns"]
            theta0 = r["direction_deg"] * np.pi / 180
            keep = [x for x in rows
                    if not (x["kind"] == "diagonal"
                            and x["columns"] == (ca, cb)
                            and abs(x["direction_deg"] - r["direction_deg"]) < 1e-9)]
            refined = list(diagonal_unit(ca, cb, theta0, K_ref))
            for off in sub:
                refined.extend(diagonal_unit(ca, cb, theta0 + off, K_ref))
            rows = keep + refined

    rows.sort(key=lambda r: (-abs(r["t_fit"]), r["kind"], r["columns"],
                             r["direction_deg"], r["threshold"]))
    seen, unique = set(), []
    for r in rows:
        key = (r["kind"], r["columns"], round(r["direction_deg"], 9),
               round(r["threshold"], 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Typing: the step against a smooth background augmented along the
    # projection. The refit reads the evaluation half, which selection never
    # touched, so type_z carries the same exact null calibration as z.
    #
    # The global type_z cannot separate a true jump from a steep smooth
    # transition: a sigmoid or kink concentrated near the boundary is not in
    # the cubic background's span, so the step soaks it up and type_z stays
    # large. The local discriminator (type_gap) is a discontinuity t-test:
    # it compares the nearest evaluation-half points on each side of the
    # threshold. A true jump shows a level gap; a steep smooth transition or
    # kink is continuous there, so the gap vanishes. The number of nearest
    # points sets the resolution limit: a transition narrower than their
    # span is indistinguishable from a jump.
    eval_mask = np.zeros(n, dtype=np.uint8)
    eval_mask[eval_rows] = 1
    for r in unique:
        r["type_z"] = float("nan")
        r["type_ratio"] = float("nan")
        r["type_gap_t"] = float("nan")
    for r in unique[:type_top]:
        if r["kind"] == "axis":
            j = index[r["columns"][0]]
            u = Z[:, j]
            background = np.column_stack([np.ones(n), u, u * u, u ** 3])
        else:
            j, k = index[r["columns"][0]], index[r["columns"][1]]
            theta = r["direction_deg"] * np.pi / 180
            u = np.cos(theta) * Z[:, j] + np.sin(theta) * Z[:, k]
            background = np.column_stack([np.ones(n), Z[:, j], Z[:, k],
                                          u * u, u ** 3])
        step = (u > r["threshold"]).astype(float)
        typed = disc_step_z(np.ascontiguousarray(background, dtype=float),
                            np.ascontiguousarray(step, dtype=float),
                            np.ascontiguousarray(t, dtype=float), eval_mask)
        if typed is not None:
            r["type_z"] = float(typed)
        # Local discontinuity discriminator: scale-invariance of the gap.
        # The level gap is measured at two scales (nearest K1 vs K2 points
        # per side), maximized over a window of thresholds around the
        # fit-half estimate for robustness to threshold error. A true jump
        # shows the same gap at both scales (ratio ~1); a steep smooth
        # transition or kink shows a gap that grows with scale (ratio ~2+).
        ue, te = u[eval_rows], t[eval_rows]
        c0 = r["threshold"]
        best1, best2, best_t1 = 0.0, 0.0, 0.0
        for c in np.linspace(c0 - _TYPE_GAP_W, c0 + _TYPE_GAP_W, 11):
            below = ue < c
            above = ue > c
            if below.sum() < _TYPE_GAP_K2 or above.sum() < _TYPE_GAP_K2:
                continue
            db = c - ue[below]
            da = ue[above] - c
            ob = np.argsort(db)
            oa = np.argsort(da)
            yb1 = te[below][ob[:_TYPE_GAP_K1]]
            ya1 = te[above][oa[:_TYPE_GAP_K1]]
            yb2 = te[below][ob[:_TYPE_GAP_K2]]
            ya2 = te[above][oa[:_TYPE_GAP_K2]]
            g1 = abs(ya1.mean() - yb1.mean())
            g2 = abs(ya2.mean() - yb2.mean())
            v1 = yb1.var(ddof=1) / _TYPE_GAP_K1 + ya1.var(ddof=1) / _TYPE_GAP_K1
            t1 = g1 / np.sqrt(v1) if v1 > 0 else 0.0
            if g1 > best1:
                best1 = g1
            if g2 > best2:
                best2 = g2
            if t1 > best_t1:
                best_t1 = t1
        if best1 > 0:
            r["type_ratio"] = float(best2 / best1)
            r["type_gap_t"] = float(best_t1)
        else:
            r["type_ratio"] = float("nan")
            r["type_gap_t"] = float("nan")

    hits = [DiscontinuityHit(
        kind=r["kind"], columns=r["columns"], direction_deg=r["direction_deg"],
        threshold=r["threshold"], threshold_x=r["threshold_x"],
        boundary_intercept=r["boundary_intercept"],
        boundary_slope=r["boundary_slope"], line=r["line"],
        n_above=r["n_above"], jump_fit=r["jump_fit"], t_fit=r["t_fit"],
        jump=r["jump"], se=r["se"], z=r["z"], gain=r["gain"],
        type_z=r["type_z"], type_ratio=r["type_ratio"],
        type_gap_t=r["type_gap_t"]) for r in unique]
    return hits[:top_k] if top_k else hits


def _screen_missing(values, residual, names, group_names, na_action):
    """The screened values and residual with their missing entries handled.

    Returns ``(values, residual)`` with every value finite, except a grouped
    column's missing level under ``"impute"``, which stays NaN so the grouping
    codes it as its own level. See :func:`screen_interactions`.
    """
    missing = ~np.isfinite(values)
    bad_residual = ~np.isfinite(residual)
    if na_action == "omit":
        keep = ~(missing.any(axis=1) | bad_residual)
        if not keep.any():
            raise ValueError(
                "na_action='omit' would drop every row of the interaction screen: "
                "every row has a missing value in a screened column or the residual")
        return values[keep], residual[keep]
    if bad_residual.any():
        first = int(np.flatnonzero(bad_residual)[0])
        raise ValueError(
            f"the residual has {int(bad_residual.sum())} non-finite value(s), first at "
            f"row {first}, and na_action={na_action!r} cannot fill a residual. Pass "
            "na_action='omit' to drop those rows from the screen, or a residual that "
            "is finite on every row")
    for j, name in enumerate(names):
        column_missing = missing[:, j]
        if not column_missing.any():
            continue
        if name in group_names and na_action in ("impute", "impute_model", "impute_model_split"):
            continue
        if na_action == "error":
            first = int(np.flatnonzero(column_missing)[0])
            raise ValueError(
                f"column {name!r} has {int(column_missing.sum())} missing value(s) in the "
                f"interaction screen, first at row {first}. Pass na_action='impute' to fill "
                "them with the column median (a grouped column keeps them as a level of "
                "their own), na_action='omit' to drop those rows, or fill the column "
                "yourself before screening")
        present = values[~column_missing, j]
        if present.size == 0:
            raise ValueError(
                f"column {name!r} has no observed value to impute from in the interaction "
                "screen; drop the column or pass na_action='omit'")
        values[column_missing, j] = float(np.median(present))
    return values, residual


def column_screen(data, response, columns=None, *, output="auto"):
    """Scatterplot diagnostics of the response against each column.

    For each numeric, boolean or categorical column (a categorical column is
    read through its level codes), the Rust core measures how much of the
    response's variance a straight line, a curve (means in up to 20 bins of
    equal counts, adjusted for the bins' degrees of freedom) and the best
    single step each explain, the Pearson and
    Spearman correlations, the column's skewness and its distinct values.
    These are marginal: a column correlated with another carries some of its
    effect, which a model fitted to all columns separates.

    Columns
    -------
    ``linear_r2``, ``curve_r2``, ``step_r2``
        Shares of the response variance explained.
    ``curvature``
        ``curve_r2 - linear_r2``: evidence for a smooth over a line.
    ``step_share``
        What a single step adds over the line, relative to ``curvature``:
        near one or above when a jump carries the curve.
    ``hint``
        ``"none"`` (the curve explains under 1%), ``"levels"`` for a factor,
        ``"line"``, ``"step"`` or ``"curve"``, from the shares above; a hint
        for which shapes to test, not a decision.

    Returns a data frame (see ``output``) sorted by ``curve_r2``.
    """
    from mboostrs._frame import read_column
    from mboostrs._output import output_frame
    from mboostrs._rust import column_diagnostics

    y = np.asarray(read_column(data, response).values, dtype=float)
    names = columns
    if names is None:
        names = [c for c in (data.columns if hasattr(data, "columns") else data)
                 if c != response]
    rows = []
    for name in names:
        column = read_column(data, name)
        if column.kind not in ("numeric", "boolean", "categorical"):
            continue
        x = np.asarray(column.values, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        d = dict(column_diagnostics(np.ascontiguousarray(x[ok]), np.ascontiguousarray(y[ok])))
        if d["curve_r2"] < 0.01:
            hint = "none"
        elif column.kind == "categorical":
            hint = "levels"
        elif d["curvature"] < 0.1 * d["curve_r2"]:
            hint = "line"
        elif d["step_share"] >= 0.8:
            hint = "step"
        else:
            hint = "curve"
        rows.append({"column": name, "kind": column.kind, **d, "hint": hint})
    rows.sort(key=lambda row: row["curve_r2"], reverse=True)
    return output_frame(rows, output)
