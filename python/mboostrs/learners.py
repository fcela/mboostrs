import warnings

import numpy as np

from mboostrs._control import boost_control
from mboostrs._validate import (
    check_boundary_knots,
    check_matrix,
    check_name_list,
)


def _count(name, value, minimum=0):
    """Validate an integer count argument (`knots`, `degree`, `differences`,
    `deriv`, `niter`, ...) at or above `minimum`.

    Upstream `mboost` rejects `bbs(x, knots = -3)`, `bbs(x, degree = -1)`, and
    `bbs(x, differences = -1)` at construction and accepts `0` for all three.
    Without this check a negative value reaches PyO3's `i32 -> usize`
    conversion and surfaces as `OverflowError: can't convert negative int to
    unsigned`, which names neither the parameter nor the constraint.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


#: `bins="auto"` before it is resolved against the data.
AUTO_BINS = "auto"

#: Fewest bins `bins="auto"` will choose. Below this the binned design has too
#: few distinct rows to carry a spline basis of any useful dimension.
MIN_AUTO_BINS = 64

#: Bins per basis column. A binned design with as many rows as it has columns
#: is exactly determined and cannot represent the shrinkage a penalty applies;
#: four rows per column leaves the penalty something to work on.
AUTO_BINS_PER_COLUMN = 4

#: Rows per bin below which binning is not worth doing: the exact design is
#: already about the size the binned one would be.
AUTO_MIN_ROWS_PER_BIN = 4

#: Memory `bins="auto"` aims to keep one learner's retained design under.
AUTO_MEMORY_BUDGET_BYTES = 8 * 1024 * 1024


def _bins(name, value):
    """Validate `bins=`, which is a count or the string ``"auto"``."""
    if isinstance(value, str):
        if value != AUTO_BINS:
            raise ValueError(
                f"{name} must be a positive integer or {AUTO_BINS!r}, "
                f"got {value!r}"
            )
        return AUTO_BINS
    return _count(name, value, minimum=1)


def auto_bins(n_rows, n_distinct, basis_columns, *,
              memory_budget=AUTO_MEMORY_BUDGET_BYTES):
    """The bin count `bins="auto"` chooses, or `None` for "do not bin".

    Four things decide it, and the first three are the ones the caller would
    otherwise have to reason about by hand:

    * **The basis dimension.** A binned design has one row per bin, so a
      learner with `k` basis columns needs more than `k` bins to be
      determined at all, and comfortably more than that for the penalty to
      have anything to shrink. Hence `AUTO_BINS_PER_COLUMN` per column, with
      `MIN_AUTO_BINS` as a floor.
    * **The distinct values in the column.** More bins than distinct values
      collapses nothing: the effective bin count is capped by the data.
    * **A memory budget.** The retained design is `bins x columns x 8` bytes,
      so the budget caps the bin count directly.
    * **The row count.** Below `AUTO_MIN_ROWS_PER_BIN` rows per bin the exact
      design is already about as small as the binned one, and binning only
      buys an approximation.

    Returns `None` when binning would not help, which is a decision worth
    reporting rather than a silent no-op -- `binning_report()` says so.
    """
    basis_columns = max(int(basis_columns), 1)
    target = max(MIN_AUTO_BINS, AUTO_BINS_PER_COLUMN * basis_columns)
    cap = max(basis_columns + 1, int(memory_budget // (8 * basis_columns)))
    target = min(target, cap)
    if n_distinct <= target:
        return None
    if n_rows < AUTO_MIN_ROWS_PER_BIN * target:
        return None
    return int(target)


def _positive(name, value):
    """Validate a strictly positive, finite float argument (`df`, `lambda2`,
    `bandwidth`)."""
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v


def _nonnegative(name, value):
    """Validate a non-negative, finite float argument (`lambda_value`).

    A penalty weight below zero rewards roughness instead of penalizing it.
    Upstream accepts it silently; mboostrs refuses.
    """
    v = float(value)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return v


def _mrf_penalty(K):
    """Validate a `bmrf` neighbourhood penalty and return it as a float array.

    The matrix must be a graph Laplacian: symmetric, with every row summing
    to zero. The mistake this catches is passing an adjacency matrix (zero
    diagonal, `+1` for neighbours) instead. It has the same shape and the
    same sparsity pattern, so nothing downstream would object, but it is
    indefinite — `beta' K beta` can be negative — so it rewards the
    divergence between neighbouring effects that the penalty exists to
    suppress. R's `mboost` refuses it with the same `sum == 0` test.
    """
    if K is None:
        # A missing K is reported at fit time, where the message can name the
        # learner and the formula it came from. Nothing to validate here.
        return None

    arr = np.asarray(K, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"K must be a square matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("K must be finite; it contains NaN or infinity")

    scale = np.abs(arr).max()
    if scale == 0:
        return arr
    tol = 1e-8 * scale * arr.shape[0]

    if not np.allclose(arr, arr.T, atol=tol, rtol=0):
        raise ValueError("K must be symmetric: K[i, j] and K[j, i] differ")

    row_sums = arr.sum(axis=1)
    worst = int(np.argmax(np.abs(row_sums)))
    if abs(row_sums[worst]) > tol:
        looks_like_adjacency = (
            np.allclose(np.diag(arr), 0.0, atol=tol, rtol=0) and arr.min() >= 0
        )
        hint = (
            "\n\nThis looks like an adjacency matrix. Convert it with "
            "K = numpy.diag(A.sum(axis=1)) - A."
            if looks_like_adjacency
            else ""
        )
        raise ValueError(
            f"K must be a graph Laplacian, but row {worst} sums to "
            f"{row_sums[worst]:.6g} rather than 0. Set K[i, i] to the number "
            "of neighbours of region i and K[i, j] = -1 when i and j are "
            "adjacent (the format of R's BayesX::bnd2gra)." + hint
        )
    return arr


def _df_against_null_space(df, null_rank, center, *, supports_center=True):
    """Reject a `df` below the rank of the penalty's null space, warn at it.

    A P-spline penalty of difference order `d` leaves a `d`-dimensional
    unpenalized subspace, so `df(lambda)` cannot fall below `d` no matter how
    large `lambda` grows; asking for less silently pins `lambda` at its maximum
    and fits a different model from the one requested. `center=True` removes
    the null space, and then any positive `df` is attainable. This is the check
    `mboost::bbs` performs, with its message.

    `df` equal to the rank is the boundary: it is approached as `lambda` grows
    but never reached, so it also pins `lambda`. R warns there rather than
    failing, and so does this — the fit is the maximally shrunk one, which is a
    defensible thing to ask for, but it is not a `df` the argument controls.

    When `supports_center=False` (e.g. for `Bmono`, which has no `center`
    parameter), the "or set center=True" suggestion is omitted from messages.
    """
    if df is None or center or null_rank == 0:
        return
    center_hint = " or set center=True" if supports_center else ""
    if float(df) < null_rank:
        raise ValueError(
            "not possible to specify 'df' smaller than the rank of the null "
            "space (unpenalized part of P-spline). Use a larger value for "
            f"'df' (>= {null_rank}){center_hint}. Got df={df!r}."
        )
    if float(df) == null_rank:
        center_hint_w = (
            ", or center=True to remove the null space"
            if supports_center else ""
        )
        warnings.warn(
            f"df={df!r} equals the rank of the penalty's null space (the "
            "unpenalized part of the P-spline), which no finite lambda "
            "attains: the learner is fitted at maximum shrinkage and df= no "
            f"longer sets its flexibility. Use df > {null_rank}"
            f"{center_hint_w}.",
            UserWarning,
            stacklevel=3,
        )


#: Shape constraints a P-spline learner can impose on its fitted curve.
#: `Bbs` applies one by projecting each boosting step's coefficients; `Bmono`
#: enforces it through the constrained solver. Defined once so the two cannot
#: drift apart.
_SHAPE_CONSTRAINTS = ("increasing", "decreasing", "convex", "concave",
                      "positive", "negative")


def _single_column_name(learner: str, name):
    """Require exactly one non-empty column name.

    A list reached the fit path unchecked and failed there with
    "unhashable type: 'list'" when the name was used as a dict key, which
    names neither the argument nor the constraint. A non-string reached it
    and became a column lookup that could not match anything.
    """
    if isinstance(name, (list, tuple, set)):
        joined = ", ".join(f"{learner}({n!r})" for n in list(name)[:3])
        raise TypeError(
            f"{learner} takes one column name, got {type(name).__name__} "
            f"{list(name)!r}. A single learner over several columns is not "
            f"supported; write one per column ({joined}) to select them "
            "separately, or use Buser with an explicit design matrix to keep "
            "them in one learner."
        )
    if not isinstance(name, str):
        raise TypeError(
            f"{learner} takes a column name, got {type(name).__name__} {name!r}"
        )
    if not name.strip():
        raise ValueError(f"{learner} column name must not be empty")
    return name


class Bols:
    """Ordinary least-squares base-learner (R `bols`).

    Fits the predictor linearly: the design is the column itself (plus an
    intercept column unless `intercept=False`), and each boosting step solves
    an unpenalized least-squares problem on it. A categorical column expands to
    one indicator per level. This is the only learner `glmboost` uses, and a
    bare column name in a `glmboost` formula expands to `bols(name)`.

    Args:
        name: predictor name in the data frame.
        intercept: include a per-learner intercept column (default True).
        df: degrees of freedom. Supplying it adds a ridge penalty calibrated
            so the learner's hat matrix has this trace; omitted (the default),
            the learner is unpenalized.
        lambda_value: ridge penalty, used directly instead of calibrating from
            `df`. Takes precedence over `df` when supplied.
        by: optional varying-coefficient variable name. `bols(x, by=z)`
            multiplies the whole design by `z`, so the learner fits
            `z * (b0 + b1 * x)` and carries the `z` main effect with it. A
            factor `by` has no single column to multiply by: the design is
            stacked once per non-reference level, so the term fits each
            level's difference from the first. `gamboost` builds that;
            `glmboost` and the LSS frontends refuse it.
        bins: optional quantile-bin count for compact training storage.
            Currently requires ``intercept=False`` and no ``by=`` term.
    """

    def __init__(self, name, *, intercept=True, df=None, lambda_value=None, by=None,
                 bins=None):
        self.name = _single_column_name("Bols", name)
        self.intercept = intercept
        if df is not None:
            _positive("df", df)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        self.df = df
        self.lambda_value = lambda_value
        self.by = by
        self.bins = None if bins is None else _bins("bins", bins)
        if self.bins is not None and self.intercept:
            raise ValueError("binned Bols currently requires intercept=False")
        if self.bins is not None and self.by is not None:
            raise ValueError("binned Bols with by= is not implemented; omit bins to use the exact learner")


class Bbs:
    """Penalized B-spline (P-spline) base-learner (R `bbs`).

    Expands the predictor into a B-spline basis over `knots` equally spaced
    inner knots and penalizes the `differences`-order differences between
    adjacent coefficients. The penalty weight is calibrated so the learner's
    hat matrix has trace `df`. This is the default expansion of a bare column
    name in a `gamboost` formula.

    Args:
        name: predictor name in the data frame.
        knots: number of inner knots (default 20). The basis has
            `knots + degree + 1` columns.
        degree: B-spline degree (default 3, cubic).
        differences: order of the difference penalty (default 2). Its null
            space — the polynomial part the penalty cannot shrink — has
            dimension `differences`, which is the floor on `df`.
        df: degrees of freedom (default 4). Must be at least `differences`
            unless `center=True`.
        lambda_value: penalty weight, used directly instead of calibrating
            from `df`. Takes precedence over `df` when supplied.
        center: project the penalty's null space out of the design (default
            False), absorbing the polynomial part into the model intercept.
        by: optional varying-coefficient variable name. Multiplies the basis
            by that column, fitting a curve whose height varies with it. A
            factor `by` stacks the basis once per non-reference level instead,
            so the term fits each level's difference from the first, and needs
            `df` at least `differences * (levels - 1)`. `gamboost` builds that;
            `glmboost` and the LSS frontends refuse it.
        cyclic: wrap the basis at the boundary knots so the fitted curve and
            its derivatives match at both ends (default False).
        boundary_knots: `(lo, hi)` pair bounding the basis. Defaults to the
            observed range of the predictor.
        constraint: optional shape constraint, one of `VALID_CONSTRAINTS`.
        deriv: order of derivative to evaluate the basis at (default 0, the
            function itself).
        bins: optional quantile-bin count. At least one bin per distinct value
            reproduces the exact design; smaller values approximate it.
        knot_placement: how inner knots are placed, ``"equidistant"`` (default,
            matching R ``mboost::bbs`` with a finite-difference penalty) or
            ``"quantile"`` (an O-spline with deduplicated empirical-quantile
            inner knots and an exact integrated second-derivative curvature
            penalty ``integral B''(x) B''(x)^T dx`` that preserves raw-scale
            linearity in the penalty null space).
    """

    #: Training range a Multinomial (Kronecker) fit places its knots over,
    #: set by that fit so a loaded model predicts without its data.
    kronecker_boundary = None

    VALID_CONSTRAINTS = _SHAPE_CONSTRAINTS
    VALID_KNOT_PLACEMENTS = ("equidistant", "quantile")

    def __init__(self, name, *, knots=20, degree=3, differences=2, df=4,
                 lambda_value=None, center=False, by=None,
                 cyclic=False, boundary_knots=None, constraint=None,
                 deriv=0, bins=None, knot_placement="equidistant"):
        # Numeric-argument validation, mirroring the checks `mboost::bbs`
        # performs at construction.
        knots = _count("knots", knots)
        degree = _count("degree", degree)
        differences = _count("differences", differences)
        deriv = _count("deriv", deriv)
        if knot_placement not in self.VALID_KNOT_PLACEMENTS:
            raise ValueError(
                f"knot_placement must be one of {self.VALID_KNOT_PLACEMENTS}, "
                f"got {knot_placement!r}"
            )
        if knot_placement == "quantile":
            if degree < 2:
                raise ValueError(
                    f"knot_placement='quantile' (O-spline) requires degree >= 2, got {degree}"
                )
            if cyclic:
                raise ValueError("knot_placement='quantile' cannot be combined with cyclic=True")
            if constraint is not None:
                raise ValueError("knot_placement='quantile' does not yet support shape constraints")
        if df is not None:
            _positive("df", df)
            _df_against_null_space(df, 2 if knot_placement == "quantile" else differences, center)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        self.name = _single_column_name("Bbs", name)
        self.knots = knots
        self.degree = degree
        self.differences = differences
        self.df = df
        self.lambda_value = lambda_value
        self.center = center
        self.by = by
        self.cyclic = cyclic
        # A degenerate or reversed pair gives a basis with no support or
        # one whose knots run backwards -- a design that is silently
        # wrong rather than an error at fit time.
        self.boundary_knots = check_boundary_knots("boundary_knots", boundary_knots)
        self.deriv = deriv
        self.bins = None if bins is None else _bins("bins", bins)
        self.knot_placement = knot_placement
        if self.bins is not None and by is not None:
            raise ValueError("binned Bbs with by= is not implemented; omit bins to use the exact learner")
        if self.bins is not None and constraint is not None:
            raise ValueError("binned Bbs does not yet support shape constraints")
        if constraint is not None and constraint not in self.VALID_CONSTRAINTS:
            raise ValueError(
                f"constraint must be one of {self.VALID_CONSTRAINTS} or None, got {constraint!r}"
            )
        self.constraint = constraint


class Bmono:
    """Monotone-constrained P-spline learner (R `bmono`).

    At each boosting step solves the constrained QP
    `min ½β'Aβ - b'β  s.t.  Dβ ≥ 0` via one of two solvers:

    - `type="quad.prog"` (default, and R's default): primal active-set QP.
      Returns the exact minimiser, so the fitted coefficients satisfy `Dβ ≥ 0`
      at every boosting step and the cumulative fit satisfies the constraint
      at every iteration count. Matches R to ~5e-10 on the parity fixtures for
      every constraint shape.
    - `type="iterative"`: R's `type = "iterative"` — violation-penalty fixed
      point with `λ₂ = 1e6`. About 35x faster per step (measured: 8 ms against
      282 ms for 1000 iterations on n=600), and feasible only approximately.

    The per-step violation left by `"iterative"` is small but carries a
    consistent sign, and boosting sums the steps, so the violation in the
    cumulative fit grows with the iteration count instead of averaging out.
    Fitting `constraint="increasing"` to a falling relation, the fitted curve
    drops 0.0037 in total by iteration 10 and 0.675 by iteration 2000, where
    the constraint permits no drop at all. `"quad.prog"` holds at exactly zero
    for every iteration count, which is why it is the default. Choose
    `"iterative"` only where the speed matters more than exact feasibility.

    Args:
        name: predictor name in the data frame.
        constraint: one of "increasing", "decreasing", "convex", "concave",
            "positive", "negative".
        knots, degree, differences, df: same as `Bbs`.
        type: "quad.prog" (default) or "iterative". Aliases for quad.prog:
            "quadprog".
        lambda2: iterative-only; violation-penalty coefficient (default
            1e6 — matches R). Ignored when type="quad.prog".
        niter: iterative-only; max inner fixed-point iterations (default 10).
            Ignored when type="quad.prog".
    """

    VALID_CONSTRAINTS = _SHAPE_CONSTRAINTS
    VALID_TYPES = ("iterative", "quad.prog", "quadprog")

    def __init__(self, name, *, constraint="increasing", knots=20, degree=3,
                 differences=2, df=4, type="quad.prog", lambda2=1e6, niter=10,
                 by=None, center=False):
        if constraint not in self.VALID_CONSTRAINTS:
            raise ValueError(
                f"constraint must be one of {self.VALID_CONSTRAINTS}, got {constraint!r}"
            )
        if type not in self.VALID_TYPES:
            raise ValueError(
                f"type must be one of {self.VALID_TYPES}, got {type!r}"
            )
        knots = _count("knots", knots)
        degree = _count("degree", degree)
        differences = _count("differences", differences)
        niter = _count("niter", niter, minimum=1)
        if df is not None:
            _positive("df", df)
            # `bmono` shares `bbs`'s difference penalty, so it has the same
            # null space.
            _df_against_null_space(df, differences, center,
                                   supports_center=True)
        _positive("lambda2", lambda2)
        self.name = _single_column_name("Bmono", name)
        self.constraint = constraint
        self.knots = knots
        self.degree = degree
        self.differences = differences
        self.df = df
        self.type = type
        self.lambda2 = lambda2
        self.niter = niter
        self.by = by
        self.center = center


class Bmrf:
    """Markov Random Field base-learner (R `bmrf`).

    Fits spatially-smooth effects for categorical/factor variables with
    a neighborhood penalty matrix K. At each boosting step, the indicator
    design (one column per level) is Ridge-penalized by K.

    Args:
        name: predictor name in the data frame (must be a factor/column
              whose integer codes give the region levels, 1-based).
        K: neighbourhood penalty matrix (n_levels x n_levels), in graph
           Laplacian form: ``K[i, i]`` is the number of neighbours of region
           ``i``, ``K[i, j] = -1`` when ``i`` and ``j`` are adjacent, and 0
           otherwise. Every row sums to zero. This is what
           ``BayesX::bnd2gra`` produces and what ``mboost::bmrf`` requires.
           Passing an adjacency matrix instead (zero diagonal, ``+1`` for
           neighbours) is rejected: it has the same sparsity pattern but is
           indefinite, so it would penalise the wrong directions.
        df: degrees of freedom (default 4). Used to compute lambda when
            lambda is not supplied.
        lambda_value: optional smoothing parameter lambda. Takes precedence
                      over df when supplied.
        center: centre the design by projecting out the null space of K
                (default False). When True, the constant effect is
                absorbed into the intercept.
        by: optional varying-coefficient variable name.
        bins: optional quantile-bin count per coordinate axis. Only occupied
            two-dimensional cells are stored.
    """

    def __init__(self, name, *, K, df=4, lambda_value=None, center=False, by=None):
        if df is not None:
            _positive("df", df)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        self.name = _single_column_name("Bmrf", name)
        self.K = _mrf_penalty(K)
        self.df = df
        self.lambda_value = lambda_value
        self.center = center
        self.by = by


class Bspatial:
    """Tensor-product P-spline base-learner (R `bspatial`).

    Fits a smooth 2-D surface via the Kronecker product of two marginal
    B-spline bases. The penalty is the additive Kronecker penalty
    K1 ⊗ I2 + I1 ⊗ K2.

    In R, `bspatial(x1, x2, ...)` is a thin wrapper around `bbs` that
    detects two positional arguments and switches to tensor-product mode.
    Here we make the two-coordinate interface explicit.

    Args:
        name1: name of the first coordinate column in the data frame.
        name2: name of the second coordinate column in the data frame.
        knots: number of inner knots for each marginal (default 20).
        degree: B-spline degree for each marginal (default 3).
        differences: order of the difference penalty for each marginal
                     (default 2).
        df: degrees of freedom (default 6, matching R `bspatial`).
        lambda_value: optional smoothing parameter lambda. Takes precedence
                      over df when supplied.
        center: centre the design by spectral projection (default False).
        by: optional varying-coefficient variable name.
        bins: enable exact compact factor storage. Must be at least the number
            of observed levels; factor levels are never merged.
    """

    def __init__(self, name1, name2, *, knots=20, degree=3, differences=2,
                 df=6, lambda_value=None, center=False, by=None, bins=None):
        knots = _count("knots", knots)
        degree = _count("degree", degree)
        differences = _count("differences", differences)
        if df is not None:
            _positive("df", df)
            # The additive Kronecker penalty K1 (x) I + I (x) K2 has a null
            # space of dimension `differences ** 2` — the tensor product of the
            # two marginal null spaces. Verified against `mboost::bspatial`,
            # which accepts df = 4 and rejects df = 3.99 at the default
            # differences = 2.
            _df_against_null_space(df, differences**2, center)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        self.name1 = _single_column_name("Bspatial name1", name1)
        self.name2 = _single_column_name("Bspatial name2", name2)
        self.knots = knots
        self.degree = degree
        self.differences = differences
        self.df = df
        self.lambda_value = lambda_value
        self.center = center
        self.by = by
        self.bins = None if bins is None else _bins("bins", bins)
        if self.bins is not None and by is not None:
            raise ValueError("binned Bspatial with by= is not implemented; omit bins to use the exact learner")

    @property
    def name(self):
        return f"{self.name1},{self.name2}"


class Brandom:
    """Random-effects base-learner (R `brandom`).

    Fits a factor as random effects: expands the factor into an
    indicator design matrix (one column per level, no reference
    dropped — equivalent to R's ``contrasts.arg = "contr.dummy"``),
    then applies a ridge penalty. This is identical to ``bols`` on
    the indicator matrix, so internally it maps to the BlockBols path.

    Args:
        name: predictor name in the data frame (must be a factor or
            integer column whose unique values are the levels).
        df: degrees of freedom (default 4). Used to compute lambda
            when lambda is not supplied.
        lambda_value: optional smoothing parameter lambda. Takes
            precedence over df when supplied.
        center: centre the design by removing the weighted column
            means (default False). When True, the constant effect
            is absorbed into the intercept.
        by: optional varying-coefficient variable name.
    """

    def __init__(self, name, *, df=4, lambda_value=None, center=False, by=None,
                 bins=None):
        if df is not None:
            # `brandom` is a ridge penalty on the indicator design, so it has
            # no null space and any positive `df` is attainable.
            _positive("df", df)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        self.name = _single_column_name("Brandom", name)
        self.df = df
        self.lambda_value = lambda_value
        self.center = center
        self.by = by
        self.bins = None if bins is None else _bins("bins", bins)
        if self.bins is not None and by is not None:
            raise ValueError("binned Brandom with by= is not implemented; omit bins to use the exact learner")


__all__ = ["Bols", "Bbs", "Bmono", "Btree", "Bmrf", "Bspatial", "Brandom",
           "Bns", "Bkernel", "Bconstrained", "Buser", "Brad", "Hinge", "boost_control"]


class Btree:
    """Tree base-learner (R `btree`). Fits decision stumps by default.

    At each boosting step, finds the best binary split across all input
    columns and returns the node means as the fitted values. This is the
    base-learner used by R's `blackboost()`.

    In a formula, ``btree(lon, lat, income, maxdepth=6, selection=sse)``
    makes one tree that can split on all three columns. ``btree(.)`` uses
    all predictors. Separate terms ``btree(lon) + btree(lat)`` fit an
    additive model of two trees. A joint tree of depth greater than one can
    represent interactions through successive splits on different columns.

    Args:
        names: list of predictor names this tree may split on.
        minbucket: minimum weighted observations in a terminal node (default 4).
        minsplit: minimum weighted observations to attempt a split (default 10).
        maxdepth: maximum tree depth. 1 = stumps (the R `btree` default).
        by: optional varying-coefficient variable z. The term fits f(t) * z
            by least squares (a tree grown on u / z with weights w * z^2).
            For a 0/1 z this is R's btree(by=): a tree grown on the rows
            where z is 1, its prediction multiplied by z.
        selection: how a node chooses the column it splits. ``"sse"`` (the
            default) takes the split over every column that most reduces the
            squared error (CART): it favours columns with many split points
            and can find a column whose effect is not monotone at the first
            split, where ctree's linear association statistic is near zero
            (for example, on California housing a joint tree scores 0.2638 with
            ``"sse"`` vs 0.2705 with ``"ctree"`` at the same depth/budget).
            ``"ctree"`` is partykit's conditional inference tree as mboost's
            ``btree`` runs it (fitted values agree with mboost to 1e-15): the
            column with the largest association
            statistic with the gradient (``n r^2`` for a number, the
            between-level sum of squares over the variance for a factor), then
            its best split. ``"ctree_pvalue"`` compares that statistic's
            chi-squared p-values instead, so a factor with many levels has no
            advantage over a number. Pass ``selection="ctree"`` explicitly for
            R-parity with mboost's ``btree``.

    Example: where the two criteria disagree.

    ``"sse"`` evaluates every split; ``"ctree"`` first asks which column is
    linearly associated with the gradient. When the true effect is sharp but
    non-monotone, that association can be near zero at the first split, and
    ``"ctree"`` spends the split on a decoy column:

    >>> import numpy as np, mboostrs as mb
    >>> from mboostrs.learners import Btree
    >>> rng = np.random.default_rng(1)
    >>> n = 400
    >>> x1 = rng.uniform(-2, 2, n)   # the real signal: a "hole", |x1| > 1
    >>> x2 = rng.uniform(-2, 2, n)   # weak linear decoy
    >>> y = (np.abs(x1) > 1).astype(float) + 0.08 * x2 + rng.normal(0, 0.1, n)
    >>> data = {"x1": x1, "x2": x2, "y": y}
    >>> kw = dict(family=mb.Gaussian(), control=mb.boost_control(1))
    >>> sse = mb.gamboost("y ~ btree(x1, x2, maxdepth=1)", data, **kw,
    ...                   learners=[Btree(["x1", "x2"], maxdepth=1,
    ...                                   selection="sse")])
    >>> ctree = mb.gamboost("y ~ btree(x1, x2, maxdepth=1)", data, **kw,
    ...                     learners=[Btree(["x1", "x2"], maxdepth=1,
    ...                                     selection="ctree")])
    >>> grid = {"x1": np.array([-1.5, 1.5]), "x2": np.zeros(2)}
    >>> np.round(np.asarray(sse.predict(grid)), 3)    # split x1: found it
    array([0.482, 0.553])
    >>> np.round(np.asarray(ctree.predict(grid)), 3)  # split x2: missed it
    array([0.484, 0.484])

    When to prefer which: keep the ``"sse"`` default when the predictors are
    mostly numeric and the response may jump -- spatial discontinuities,
    thresholds, regime changes. On California housing, a joint depth-6
    tree reaches 0.2638 log-RMSE with ``"sse"`` against 0.2705 with
    ``"ctree"`` at the same depth and budget. Reach for ``"ctree"`` when the
    column set is a noisy mix and you want the association test to refuse
    weak splits: ``"sse"`` always finds *some* split that reduces in-sample
    SSE, while ``"ctree"`` only spends a split where the association clears
    the test. Note the test itself is not neutral -- under ``"ctree"`` a
    factor with many levels has an edge over a number; ``"ctree_pvalue"``
    compares p-values instead and removes it.
    """

    #: Per input, the number of factor levels when the column is a factor
    #: code (0 for a numeric input); set by the fit.
    categorical = None

    def __init__(self, names, *, minbucket=4, minsplit=10, maxdepth=1, by=None,
                 selection="sse", split_method="exact",
                 lambda_leaf: float = 0.0, colsample: float = 1.0,
                 colsample_seed: int = 0):
        if selection not in ("sse", "ctree", "ctree_pvalue"):
            raise ValueError("selection must be 'sse', 'ctree' or 'ctree_pvalue'")
        if split_method not in ("exact", "histogram"):
            raise ValueError("split_method must be 'exact' or 'histogram'")
        self.selection = selection
        self.split_method = split_method
        for _n, _v in (("minbucket", minbucket), ("minsplit", minsplit),
                       ("maxdepth", maxdepth)):
            _count(_n, _v, minimum=1)
        lambda_leaf = _nonnegative("lambda_leaf", lambda_leaf)
        colsample_val = float(colsample)
        if not np.isfinite(colsample_val) or colsample_val <= 0.0 or colsample_val > 1.0:
            raise ValueError(f"colsample must lie in (0, 1], got {colsample!r}")
        colsample_seed = _count("colsample_seed", colsample_seed, minimum=0)
        # A tree splits on the columns it is given, so an empty list is a
        # learner with nothing to split on and a repeated name makes the same
        # column compete with itself.
        self.names = check_name_list("Btree names", names)
        self.minbucket = minbucket
        self.minsplit = minsplit
        self.maxdepth = maxdepth
        self.by = by
        self.lambda_leaf = lambda_leaf
        self.colsample = colsample_val
        self.colsample_seed = colsample_seed

    @property
    def name(self):
        return ",".join(self.names)

    def __repr__(self):
        parts = [repr(self.names)]
        if self.minbucket != 4:
            parts.append(f"minbucket={self.minbucket!r}")
        if self.minsplit != 10:
            parts.append(f"minsplit={self.minsplit!r}")
        if self.maxdepth != 1:
            parts.append(f"maxdepth={self.maxdepth!r}")
        if self.by is not None:
            parts.append(f"by={self.by!r}")
        if self.selection != "sse":
            parts.append(f"selection={self.selection!r}")
        if self.split_method != "exact":
            parts.append(f"split_method={self.split_method!r}")
        if self.lambda_leaf != 0.0:
            parts.append(f"lambda_leaf={self.lambda_leaf!r}")
        if self.colsample != 1.0:
            parts.append(f"colsample={self.colsample!r}")
        if self.colsample_seed != 0:
            parts.append(f"colsample_seed={self.colsample_seed!r}")
        return f"Btree({', '.join(parts)})"

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.names == other.names
            and self.minbucket == other.minbucket
            and self.minsplit == other.minsplit
            and self.maxdepth == other.maxdepth
            and self.by == other.by
            and self.selection == other.selection
            and self.split_method == other.split_method
            and self.lambda_leaf == other.lambda_leaf
            and self.colsample == other.colsample
            and self.colsample_seed == other.colsample_seed
        )


class Bns:
    """Ridge-penalized B-spline base-learner (mboostrs extension).

    Constructs a B-spline basis from the predictor and fits it with an
    identity (ridge) penalty. Because the basis is not penalized by a
    difference operator, bns uses more degrees of freedom than bbs and
    produces a less smooth fit.

    Current upstream ``mboost::bns()`` reports that the learner was removed
    and recommends ``bbs``. This class retains the historical name for
    compatibility but is not an upstream-equivalent natural spline.

    Args:
        name: predictor name in the data frame.
        knots: number of inner knots (default 20).
        degree: B-spline degree (default 3, cubic).
        df: degrees of freedom (default = ncol(basis), i.e. effectively
            unpenalized). Reducing df adds ridge regularization.
        center: centre the design by spectral projection (default False).
        by: optional varying-coefficient variable name.
    """

    def __init__(self, name, *, knots=20, degree=3, df=None, center=False, by=None):
        knots = _count("knots", knots)
        degree = _count("degree", degree)
        if df is not None:
            # `bns` substitutes an identity ridge for the difference penalty,
            # so like `brandom` it has no null space.
            _positive("df", df)
        self.name = _single_column_name("Bns", name)
        self.knots = knots
        self.degree = degree
        self.df = df
        self.center = center
        self.by = by


class Bkernel:
    """Kernel base-learner (R `bkernel`).

    Constructs an n×n kernel matrix K[i,j] = kernel(|x_i - x_j| / h),
    then fits a kernel ridge regression on it. Because the design is n×n,
    this is O(n²) in memory — avoid for large n.

    Pass `landmarks=m` to build the Nyström approximation instead: an n×m
    design against `m` evenly-spaced quantile landmarks of x, with the same
    ridge as the exact path. Memory is O(n·m), and the approximation is exact
    when `m` equals the number of distinct x values. Coefficients then index
    the landmarks, so they number `m`, not `n`.

    Args:
        name: predictor name in the data frame.
        bandwidth: kernel bandwidth h (default 1.0).
        df: degrees of freedom (default 4). Used to compute lambda.
        kernel: kernel type, one of "gaussian" (default),
            "epanechnikov", or "uniform".
        landmarks: number of Nyström landmarks (default None = exact n×n).
        center: centre the design by spectral projection (default False).
        by: optional varying-coefficient variable name.
    """

    VALID_KERNELS = ("gaussian", "epanechnikov", "uniform")

    def __init__(self, name, *, bandwidth=1.0, df=4, kernel="gaussian",
                 landmarks=None, center=False, by=None):
        if kernel not in self.VALID_KERNELS:
            raise ValueError(
                f"kernel must be one of {self.VALID_KERNELS}, got {kernel!r}"
            )
        _positive("bandwidth", bandwidth)
        if df is not None:
            _positive("df", df)
        if landmarks is not None:
            if isinstance(landmarks, bool) or not isinstance(landmarks, (int, np.integer)):
                raise ValueError(
                    f"landmarks must be a positive integer or None, got {landmarks!r}"
                )
            landmarks = int(landmarks)
            if landmarks < 1:
                raise ValueError(
                    f"landmarks must be at least 1, got {landmarks}"
                )
            if kernel != "gaussian":
                raise ValueError(
                    f"landmarks (Nyström approximation) is only supported for "
                    f"kernel='gaussian'; compact-support kernels (epanechnikov, "
                    f"uniform) produce singular landmark matrices. Got kernel={kernel!r}"
                )
        self.name = _single_column_name("Bkernel", name)
        self.bandwidth = bandwidth
        self.df = df
        self.kernel = kernel
        self.landmarks = landmarks
        self.center = center
        self.by = by


class Bconstrained:
    """Constrained P-spline base-learner (R `bconstrained`).

    generalizes bmono to arbitrary linear constraints Cβ ≥ 0. The
    constraint matrix C is supplied by the caller. For monotone
    constraints, C = D (first-difference matrix); for convexity, C = D².

    Each step solves the constrained least-squares problem exactly with the
    primal active-set QP bmono uses by default (`type="quad.prog"`), so every
    step's coefficients satisfy ``Cβ >= 0``. `type="iterative"` selects the
    violation-penalty fixed point instead, which is cheaper but leaves a
    small same-signed violation each step that boosting accumulates (3e-6
    below zero after one step on ``C = I``).

    Args:
        name: predictor name in the data frame.
        C: constraint matrix (m x p) such that Cβ ≥ 0.
        knots: number of inner knots (default 20).
        degree: B-spline degree (default 3).
        differences: order of the difference penalty (default 2).
        df: degrees of freedom (default 4).
        lambda_value: optional smoothing parameter. Takes precedence
            over df when supplied.
        type: "quad.prog" (default) or "iterative"; alias "quadprog".
        lambda2: violation-penalty coefficient (default 1e6). Used by
            type="iterative" only.
        niter: max inner fixed-point iterations (default 10). Used by
            type="iterative" only.
        by: optional varying-coefficient variable name.
    """

    VALID_TYPES = ("iterative", "quad.prog", "quadprog")

    def __init__(self, name, *, C, knots=20, degree=3, differences=2,
                 df=4, lambda_value=None, type="quad.prog", lambda2=1e6, niter=10,
                 by=None):
        if type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}, got {type!r}")
        if C is not None:
            C = check_matrix("Bconstrained C", C)
        knots = _count("knots", knots)
        degree = _count("degree", degree)
        differences = _count("differences", differences)
        niter = _count("niter", niter, minimum=1)
        if df is not None:
            _positive("df", df)
            _df_against_null_space(df, differences, False, supports_center=False)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        _positive("lambda2", lambda2)
        self.name = _single_column_name("Bconstrained", name)
        self.C = C
        self.knots = knots
        self.degree = degree
        self.differences = differences
        self.df = df
        self.lambda_value = lambda_value
        self.type = type
        self.lambda2 = lambda2
        self.niter = niter
        self.by = by


class Buser:
    """User-supplied base-learner (R `buser`).

    Takes an arbitrary design matrix X (n × p) and penalty matrix K (p × p),
    then wraps them as a penalized linear base-learner. This is the most
    flexible base-learner — users can define custom learners without writing
    Rust code.

    Args:
        name: base-learner name.
        X: design matrix, shape (n_obs, n_params).
        K: penalty matrix, shape (n_params, n_params).
        df: degrees of freedom (default 4). Used to compute lambda when
            lambda_value is not supplied.
        lambda_value: optional smoothing parameter lambda. Takes precedence
            over df when supplied.
        center: centre the design by removing weighted column means
            (default False).
        by: optional varying-coefficient variable name.
    """

    _radial_basis = None

    def __init__(self, name, *, X, K, df=4, lambda_value=None,
                 center=False, by=None):
        # Any 2-D array-like is accepted and converted here, so downstream
        # code has an array rather than a maybe-array, and a non-finite entry
        # is named at its position instead of reaching the normal equations.
        X = check_matrix("Buser X", X)
        K = check_matrix("Buser K", K, square=True, rows=X.shape[1])
        if df <= 0:
            raise ValueError(f"Buser df must be positive, got {df}")
        if lambda_value is not None and lambda_value < 0:
            raise ValueError(f"Buser lambda_value must be non-negative, got {lambda_value}")
        self.name = _single_column_name("Buser", name)
        self.X = X
        self.K = K
        self.df = df
        self.lambda_value = lambda_value
        self.center = center
        self.by = by


# ── Radial basis (R `brad`) ──────────────────────────────────────────────────
#
# `brad` is a penalized linear smoother, which `Buser` already expresses, so
# this is a construction rather than a new engine variant: it builds the design
# and penalty and hands them to `Buser`. The engine, the FFI boundary, and the
# prediction path replays the retained radial geometry at new locations.


def _matern_32(distance, theta):
    """Matérn 3/2 correlations, evaluated by the shared Rust core."""
    from mboostrs import _rust

    d = np.asarray(distance, dtype=np.float64)
    return np.asarray(_rust.brad_covariance(d.ravel().tolist(), float(theta))).reshape(d.shape)


def _effective_range(x, eps=1e-3):
    """Matérn range at the training diameter, evaluated by the Rust core."""
    from mboostrs import _rust

    return _rust.brad_effective_range(np.atleast_2d(np.asarray(x, dtype=np.float64)), float(eps))


def _radial_design(data, recipe):
    """Convert named coordinates and replay the stored basis in Rust."""
    from mboostrs import _rust

    columns = []
    for name in recipe["names"]:
        values = data[name]
        columns.append(np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values,
                                  dtype=np.float64).ravel())
    return _rust.brad_design(np.ascontiguousarray(np.column_stack(columns)),
                             recipe["centres"], recipe["whitening"], recipe["theta"])


class Brad:
    """Radial-basis base-learner (R `brad`).

    Places `knots` centres over the predictor space, evaluates a Matérn
    covariance from every observation to every centre, and whitens the result
    by the centres' own covariance so a plain ridge penalty applies:

        X = C(x, knots) @ C(knots, knots)^(-1/2),    K = I

    That is the construction `mboost::brad` uses. Being a penalized linear
    smoother, it lowers to :class:`Buser` — no new engine variant — so it works
    everywhere `buser` does, including cross-validation and stability
    selection. The centres, range and whitening matrix are retained for
    ``design(newdata)`` and automatic model prediction at new locations.

    Use it for smooth surfaces over two or more coordinates where a
    tensor-product spline (`bspatial`) is awkward: scattered spatial data,
    or three-plus dimensional smooths where the tensor grid would explode.

    Args:
        names: predictor column name, or a list of them for a multi-dimensional
            smooth.
        data: mapping or frame the columns are read from. `brad` has to build
            its design from the fitting data. Omit data for a deferred formula learner.
        knots: number of basis centres (default 100, capped at the number of
            distinct rows).
        df: degrees of freedom (default 4).
        lambda_value: penalty weight, used instead of calibrating from `df`.
        theta: Matérn range. Defaults to the range making correlation at the
            training data diameter equal to 1e-3.
        center: centre the design (default False).
        by: optional varying-coefficient variable name.

    Note this will not reproduce R's `brad` to machine precision: R selects
    centres with `fields::cover.design` and this uses greedy farthest-point
    selection. Both are space-filling; neither choice is canonical.
    """

    def __init__(self, names, data=None, *, knots=100, df=4, lambda_value=None,
                 theta=None, center=False, by=None):
        if isinstance(names, str):
            names = [names]
        self.names = list(names)
        if not self.names:
            raise ValueError("brad requires at least one predictor")
        self.knots = _count("knots", knots, minimum=1)
        _positive("df", df)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        if theta is not None:
            _positive("theta", theta)
        self.df, self.lambda_value, self.center, self.by = df, lambda_value, center, by
        self._theta_requested = theta
        self.theta = theta
        # The artifact decoder restores fitted arrays after this lazy constructor.
        self.data = None
        self.centres = self.whitening = self.X = self.K = None
        if data is not None:
            self._build(data)

    def _coordinates(self, data):
        columns = []
        for name in self.names:
            values = data[name]
            columns.append(np.asarray(values.to_numpy() if hasattr(values, "to_numpy")
                                      else values, dtype=np.float64).ravel())
        return np.ascontiguousarray(np.column_stack(columns))

    def _build(self, data):
        from mboostrs import _rust

        self.X, self.K, self.centres, self.whitening, self.theta = _rust.brad_basis(
            self._coordinates(data), self.knots, self._theta_requested)

    def design(self, newdata):
        """Whitened design at new locations, using the retained training recipe.

        Returns an ``(n_new, n_centres)`` NumPy matrix. Fitting applies ``by``
        and weighted centering after this construction, as for a user design.
        """
        if self.centres is None:
            raise ValueError("Brad.design requires a training design; supply data when constructing Brad")
        return _radial_design(newdata, {"names": self.names, "centres": self.centres,
                                      "whitening": self.whitening, "theta": self.theta})

    @property
    def name(self):
        return ",".join(self.names)

    def as_buser(self, data=None):
        """Equivalent user design with a retained radial prediction recipe."""
        if data is not None:
            self._build(data)
        if self.X is None:
            raise ValueError("Brad requires training data before lowering to Buser")
        learner = Buser(self.name, X=self.X, K=self.K, df=self.df,
                        lambda_value=self.lambda_value, center=self.center, by=self.by)
        learner._radial_basis = {"names": self.names, "centres": self.centres,
                                 "whitening": self.whitening, "theta": self.theta}
        return learner


def basis_columns_of(learner) -> int:
    """How many design columns a learner's basis has.

    The number `bins="auto"` sizes against: a binned design with fewer rows
    than this cannot carry the basis.
    """
    if isinstance(learner, Bbs):
        return int(learner.knots) + int(learner.degree)
    if isinstance(learner, Bspatial):
        marginal = int(learner.knots) + int(learner.degree)
        return marginal * marginal
    if isinstance(learner, Bols):
        return 1 if not learner.intercept else 2
    if isinstance(learner, Brandom):
        return 1
    return 1


def resolve_auto_bins(learners, distinct_counts, n_rows,
                      memory_budget=AUTO_MEMORY_BUDGET_BYTES):
    """Replace every ``bins="auto"`` with a number, or with `None`.

    ``distinct_counts`` maps a learner's column name to how many distinct
    values it holds. Returns ``(learners, decisions)``, where each decision is
    a dict recording what was chosen and why -- `binning_report()` shows it,
    because "auto" that leaves no trace is a number the reader cannot check.
    """
    import copy

    out, decisions = [], []
    for learner in learners:
        if getattr(learner, "bins", None) != AUTO_BINS:
            out.append(learner)
            continue
        columns = basis_columns_of(learner)
        distinct = int(distinct_counts.get(learner.name, n_rows))
        chosen = auto_bins(n_rows, distinct, columns,
                           memory_budget=memory_budget)
        resolved = copy.copy(learner)
        resolved.bins = chosen
        out.append(resolved)
        if chosen is None:
            reason = (
                f"{distinct} distinct values is at or below the "
                f"{max(MIN_AUTO_BINS, AUTO_BINS_PER_COLUMN * columns)} bins "
                f"the basis would need, so binning would collapse nothing"
                if distinct <= max(MIN_AUTO_BINS, AUTO_BINS_PER_COLUMN * columns)
                else f"{n_rows} rows is fewer than "
                     f"{AUTO_MIN_ROWS_PER_BIN} per bin, so the exact design is "
                     f"already about as small"
            )
        else:
            reason = (f"{AUTO_BINS_PER_COLUMN} bins per basis column "
                      f"({columns} columns), capped by {distinct} distinct "
                      f"values and an {memory_budget // (1024 * 1024)} MB "
                      f"budget")
        decisions.append({
            "learner": learner.name,
            "basis_columns": columns,
            "distinct_values": distinct,
            "chosen_bins": chosen,
            "reason": reason,
        })
    return out, decisions


def apply_global_bins(learners, bins):
    """Set `bins` on every learner in `learners` that can carry it.

    A learner with its own explicit `bins=` keeps that value. A learner whose
    class supports binning but whose configuration does not — `by=` on any of
    them, `intercept=True` on `Bols`, a shape constraint on `Bbs` — is returned
    unchanged and named in the second return value, so the caller can say which
    terms stayed on the exact path. Classes with no binned storage at all
    (`Btree`, `Brad`, `Buser`, ...) pass through without comment.

    Returns `(learners, skipped)`, where `skipped` is a list of
    `(learner_name, reason)` pairs.
    """
    import copy

    bins = _bins("bins", bins)
    out, skipped = [], []
    for learner in learners:
        if not isinstance(learner, Bols | Bbs | Bspatial | Brandom):
            out.append(learner)
            continue
        if learner.bins is not None:
            out.append(learner)
            continue
        if learner.by is not None:
            skipped.append((learner.name, "by= terms have no binned design"))
            out.append(learner)
            continue
        if isinstance(learner, Bols) and learner.intercept:
            skipped.append(
                (learner.name, "binned bols needs intercept=False")
            )
            out.append(learner)
            continue
        if isinstance(learner, Bbs) and learner.constraint is not None:
            skipped.append(
                (learner.name, "shape-constrained bbs has no binned design")
            )
            out.append(learner)
            continue
        binned = copy.copy(learner)
        binned.bins = bins
        out.append(binned)
    return out, skipped


def warn_skipped_bins(skipped, stacklevel=3):
    """Report the learners `apply_global_bins` left on the exact path."""
    if not skipped:
        return
    detail = "; ".join(f"{name} ({reason})" for name, reason in skipped)
    warnings.warn(
        f"bins= was not applied to {len(skipped)} learner(s), which keep "
        f"their exact training storage: {detail}",
        stacklevel=stacklevel,
    )


class Hinge:
    """Adaptive hinge base-learner that estimates a piecewise-linear effect.

    One boosting selection adds exactly one listable hinge term. At each
    step the prepared learner scans ``n_candidates`` interior-quantile
    knots of the raw predictor in both directions -- ``max(0, x - t)``
    (right-facing) and ``max(0, t - x)`` (left-facing) -- fits every
    candidate by ridge-weighted least squares against the current
    residual, and keeps the one with the smallest residual SSE.

    The fitted knot marks a threshold, regime change or other abrupt change
    in slope. The final model is additive, and each selected hinge reports
    its knot, direction and coefficient.

    Args:
        name: predictor name in the data frame.
        df: degrees of freedom the ridge on each selected hinge is
            calibrated to (default 4). The selected hinge has about two
            effective parameters (knot plus coefficient), so ``df``
            governs shrinkage of the coefficient, not the knot choice;
            the knot is selected by SSE, not penalized. Calibrated
            through the same ``df -> lambda`` machinery as the other
            learners unless ``lambda_value`` is supplied.
        n_candidates: interior-quantile knots scanned per selection
            (default 20). The grid is built once from the raw column,
            before any weights, so cross-validation folds and replayed
            predictions see the same candidates the fit did.
        lambda_value: penalty weight used directly instead of calibrating
            from ``df``. Takes precedence over ``df`` when supplied.
        center: weight-center each selected hinge column before fitting
            (default True), so the hinge carries no constant and the
            model offset owns the mean. The centering travels with the
            selection and is re-applied on new data.
        by: optional varying-coefficient variable name. Multiplies the
            hinge column by that column, fitting a kink whose height
            varies with it (the ``bbs(by=)`` convention).

    Example:
        A planted kink at ``x = 0.5``; the hinge learns the bend while a
        P-spline smooths across it:

        >>> import numpy as np, mboostrs as mb
        >>> from mboostrs.learners import Hinge
        >>> rng = np.random.default_rng(1)
        >>> n = 600
        >>> x = rng.uniform(-2, 2, n)
        >>> y = np.maximum(0.0, x - 0.5) + 0.1 * rng.normal(size=n)
        >>> data = {"x": x, "y": y}
        >>> kw = dict(family=mb.Gaussian(), control=mb.boost_control(200, 0.1))
        >>> hinge = mb.gamboost("y ~ hinge(x, df=4)", data, **kw)
        >>> smooth = mb.gamboost("y ~ bbs(x, df=4)", data, **kw)
        >>> grid = {"x": np.linspace(-2, 2, 9)}
        >>> truth = np.maximum(0.0, grid["x"] - 0.5)
        >>> rmse = lambda m: float(np.sqrt(np.mean(
        ...     (np.asarray(m.predict(grid)).ravel() - truth) ** 2)))
        >>> rmse(hinge) < rmse(smooth)
        True
    """

    def __init__(self, name, *, df=4.0, n_candidates=20, lambda_value=None,
                 center=True, by=None):
        n_candidates = _count("n_candidates", n_candidates, minimum=1)
        if df is not None:
            _positive("df", df)
        if lambda_value is not None:
            _nonnegative("lambda_value", lambda_value)
        self.name = _single_column_name("Hinge", name)
        self.df = df
        self.n_candidates = n_candidates
        self.lambda_value = lambda_value
        self.center = center
        self.by = by
