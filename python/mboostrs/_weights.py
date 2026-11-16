"""One weight contract for every fitting entry point.

A boosting fit uses weights in three places at once: the base-learner normal
equations (``X'WX``), the ``df -> lambda`` calibration that reads the hat
matrix of those same equations, and the empirical risk. All three scale with
the weight vector, so the *scale* of the weights, not only their ratios,
changes the fitted path.

Upstream ``mboost`` settles that scale in ``mboost_fit`` before anything else
runs::

    rescale_weights <- function(w) {
        if (max(abs(w - floor(w))) < sqrt(.Machine$double.eps)) return(w)
        return(w / sum(w) * sum(w > 0))
    }

Integer weights are read as frequencies -- an observation with weight 3 stands
for three identical rows -- and are left alone, because their scale already
carries meaning. Anything fractional is read as an analytic weight, where only
the ratios are meaningful, and is rescaled to sum to the number of
weight-positive observations, which is the scale an unweighted fit of those
rows would have had.

``mboostrs`` sent raw fractional weights to the engine. On a ten-row Gaussian
``gamboost`` fit with ``bols(x, lambda=1)`` and weights summing to 8.4 that
moved every fitted value and every risk value away from R. Integer-weight
fixtures cannot see it, because integers are the case upstream leaves
unchanged.
"""
from __future__ import annotations

import math

import numpy as np

#: Named weight policies. ``"upstream"`` reproduces ``mboost``'s
#: ``rescale_weights``; ``"raw"`` passes the caller's vector through unchanged,
#: which is an mboostrs extension for callers who mean the weight scale
#: literally (survey weights whose sum is the population total, for example).
WEIGHT_POLICIES = ("upstream", "raw")

# `sqrt(.Machine$double.eps)`, the tolerance upstream uses to decide whether a
# weight vector is integral.
_INTEGER_TOL = math.sqrt(np.finfo(np.float64).eps)


def is_integral(w: np.ndarray) -> bool:
    """True when every weight is an integer to upstream's tolerance."""
    if w.size == 0:
        return True
    return bool(np.max(np.abs(w - np.floor(w))) < _INTEGER_TOL)


def rescale_weights(w: np.ndarray) -> np.ndarray:
    """``mboost:::rescale_weights``: frequencies pass, analytic weights rescale.

    Integer vectors are returned unchanged. Anything else is scaled to sum to
    the number of positive weights.
    """
    w = np.asarray(w, dtype=np.float64)
    if is_integral(w):
        return w
    total = float(w.sum())
    if total <= 0.0:
        # Unreachable for non-negative weights: a vector summing to zero has
        # every entry zero, which is integral and returned above.
        return w
    return w / total * float(np.count_nonzero(w > 0))


class NormalizedWeights:
    """What the caller passed, what the fit uses, and why they differ.

    ``fit`` is the vector the engine sees, and is what ``model.weights``
    returns -- matching ``model.weights()`` in ``mboost``, which returns the
    rescaled vector, so cross-validation and stability-selection folds drawn
    from it are drawn on the same scale R draws them on. ``given`` is the
    caller's own vector, kept so a model can report the input it was handed
    rather than only the vector it derived.
    """

    __slots__ = ("given", "fit", "policy", "rescaled")

    def __init__(self, given, fit, policy: str, rescaled: bool):
        self.given = given
        self.fit = fit
        self.policy = policy
        self.rescaled = rescaled

    @property
    def is_unit(self) -> bool:
        return self.given is None

    def describe(self) -> str:
        """One line naming the policy and what it did, for reports."""
        if self.is_unit:
            return "unit weights"
        if self.policy == "raw":
            return "user weights, policy='raw' (no rescaling)"
        if self.rescaled:
            return (
                f"user weights rescaled to sum to {self.fit.sum():.6g} "
                f"(policy='upstream'; fractional weights are analytic)"
            )
        return "user weights unchanged (policy='upstream'; integer frequencies)"

    def __repr__(self) -> str:
        return f"NormalizedWeights({self.describe()})"


def rescale_fold_weights(folds, policy: str = "upstream") -> np.ndarray:
    """Apply the weight policy to each column of a resampling weight matrix.

    A fold is a weight vector, and upstream runs every fold through the same
    ``mboost_fit`` that rescales the model's own weights, so a fold whose
    entries are fractional is rescaled to sum to the number of rows it keeps.
    Fold matrices drawn from unit weights are 0/1 and integral, so this is a
    no-op there; it matters when the model itself carries fractional weights,
    where each fold inherits them.

    Complementary pairing is a property of the *unrescaled* columns: the pair
    is built from the model's weights, and each column is then rescaled for
    its own fit, exactly as ``stabs:::check_folds`` builds the pair and
    ``mboost_fit`` rescales each column afterwards.
    """
    folds = np.asarray(folds, dtype=np.float64)
    if policy != "upstream" or folds.size == 0:
        return folds
    if is_integral(folds):
        return folds
    out = np.array(folds, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        out[:, j] = rescale_weights(out[:, j])
    return np.ascontiguousarray(out)


class WeightPolicyMixin:
    """What the caller gave, what the fit used, and which policy connected them.

    ``self._weights`` holds the *effective* vector the engine saw, which is
    what ``weights`` returns and what cross-validation and stability-selection
    folds are drawn from -- the same convention as ``model.weights()`` in
    ``mboost``, which returns the rescaled vector. The caller's own vector is
    kept separately so a model can report the input it was handed rather than
    only the vector it derived.
    """

    #: The vector the caller passed, before any scale policy was applied.
    #: ``None`` when the fit was unweighted.
    _input_weights = None
    #: Which named policy settled the fit scale: ``"upstream"`` or ``"raw"``.
    _weight_policy = "upstream"
    #: Whether the policy actually changed the vector.
    _weights_rescaled = False

    def _set_weight_policy(self, normalized: NormalizedWeights) -> None:
        self._input_weights = normalized.given
        self._weight_policy = normalized.policy
        self._weights_rescaled = normalized.rescaled

    def _weight_summary(self) -> dict:
        """What the weights were, for `summary()`.

        Four facts rather than one: which policy settled the scale, whether it
        moved, what the caller passed, and what the fit ran on. The last two
        differ whenever fractional weights met the upstream policy, and `n`
        says nothing about either.
        """
        import numpy as _np

        given = getattr(self, "_input_weights", None)
        effective = self.weights
        return {
            "policy": getattr(self, "_weight_policy", None) or "upstream",
            "rescaled": bool(getattr(self, "_weights_rescaled", False)),
            "input_total": (None if given is None else
                            float(_np.sum(_np.asarray(given, dtype=float)))),
            "effective_total": (None if effective is None else
                                float(_np.sum(_np.asarray(effective,
                                                          dtype=float)))),
            "note": self.weights_note(),
        }

    @property
    def weight_policy(self) -> str:
        """The named policy that settled the weight scale for this fit."""
        return self._weight_policy

    @property
    def weights(self):
        """The effective weight vector the fit ran on.

        This is what cross-validation and stability-selection folds should be
        drawn from, matching ``model.weights()`` in ``mboost``. It differs
        from ``input_weights()`` when fractional weights were rescaled under
        the ``"upstream"`` policy.
        """
        stored = getattr(self, "_weights", None)
        if stored is None:
            fitted = getattr(self, "_fitted", None)
            n = 0 if fitted is None else len(np.asarray(fitted))
            return np.ones(n, dtype=np.float64)
        return np.asarray(stored, dtype=np.float64).copy()

    def input_weights(self):
        """The weights as the caller passed them, or ``None`` if unweighted.

        ``weights`` returns the effective vector the fit used, which differs
        from this one when fractional weights were rescaled under the
        ``"upstream"`` policy.
        """
        if self._input_weights is None:
            return None
        return np.asarray(self._input_weights).copy()

    def weights_note(self) -> str:
        """One line saying how the weight scale was settled."""
        effective = getattr(self, "_weights", None)
        if effective is None:
            effective = np.ones(1)
        return NormalizedWeights(
            self._input_weights,
            np.asarray(effective),
            self._weight_policy,
            self._weights_rescaled,
        ).describe()


def normalize_weights(
    weights,
    n: int,
    *,
    policy: str = "upstream",
    allow_all_zero: bool = False,
) -> NormalizedWeights:
    """Validate ``weights`` and apply the named scale policy.

    ``weights=None`` means unit weights. Otherwise the vector must be
    length ``n``, finite, and non-negative. Zero weights are kept in place:
    an excluded row keeps its position, so every row-aligned object stays
    aligned with the design and no row is silently dropped.

    Raises ``ValueError`` for a wrong length, a non-finite entry, a negative
    entry, an unknown ``policy``, or an all-zero vector unless
    ``allow_all_zero`` is set. An all-zero vector is rejected because the fit
    it produces is not a degenerate fit of the data -- it is a fit of nothing:
    every learner's normal equations are empty, the risk is identically zero
    at every iteration, and the fitted values are all zero, so the object
    reports a completed path over an empty sample.
    """
    if policy not in WEIGHT_POLICIES:
        raise ValueError(
            f"policy must be one of {WEIGHT_POLICIES}, got {policy!r}"
        )
    if weights is None:
        return NormalizedWeights(None, np.ones(n, dtype=np.float64), policy, False)

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != n:
        raise ValueError(f"weights length {w.shape[0]} != n={n}")
    if not np.all(np.isfinite(w)):
        bad = int(np.argmax(~np.isfinite(w)))
        raise ValueError(
            f"weights must be finite; weights[{bad}] is {w[bad]}"
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    positive = int(np.count_nonzero(w > 0))
    if positive == 0 and not allow_all_zero:
        raise ValueError(
            "weights are all zero, so the fit would have no observations: "
            "every learner's normal equations are empty and the risk is zero "
            "at every iteration. Pass positive weights on the rows to fit."
        )

    given = np.ascontiguousarray(w.copy())
    if policy == "raw":
        return NormalizedWeights(given, np.ascontiguousarray(w), policy, False)

    rescaled = not is_integral(w)
    fit = rescale_weights(w)
    return NormalizedWeights(
        given, np.ascontiguousarray(fit, dtype=np.float64), policy, rescaled
    )
