"""What a bootstrap run managed to do, reported instead of discarded.

`confint()` fits `B` models on resampled weights and takes quantiles of the
coefficients. A replicate can fail for reasons that are part of the method --
a resample that drops every row carrying a level, a design that goes singular
-- so one failure must not end the run. The handling was `except Exception:
return None`, and the failed rows stayed `NaN` in the coefficient matrix,
which `np.nanquantile` then skipped. An interval computed from 40 surviving
replicates was reported in the same shape, with the same `level`, as one
computed from 1000, and nothing distinguished them.

Each replicate records its outcome in a flag bit. The core counts the bits
(`mboostrs_core::inference::bootstrap_counts`); :func:`bootstrap_outcome` turns
the counts into a statement:

* every replicate failed -- raise, quoting the first exception. There is no
  interval, and a dictionary of `NaN` is not one.
* some failed -- warn with the count and the first message, and put
  `n_failed` in the result so a caller can check it rather than read stderr.
* the inner `mstop` cross-validation failed on some replicates -- warn
  separately. Those replicates did produce coefficients, but at the outer
  model's `mstop` rather than their own, so they estimate something slightly
  different; that is a different fact from a missing replicate and gets its
  own count.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "AT_CAP",
    "FAILED",
    "UNTUNED",
    "BootstrapTally",
    "bootstrap_outcome",
    "group_bootstrap_counts",
]

#: Flag bit: the replicate produced no estimate.
FAILED = 1
#: Flag bit: the inner `mstop` cross-validation failed; the replicate was
#: fitted at the outer model's `mstop` instead.
UNTUNED = 2
#: Flag bit: the inner cross-validation chose the largest `mstop` allowed,
#: where the risk was still falling.
AT_CAP = 4


@dataclass
class BootstrapTally:
    """The outcome of each of ``B`` replicates, recorded while they run.

    Replicates run on worker threads. Replicate ``b`` only writes its own
    slot of :attr:`flags`, so the writes never collide; the exception lists
    are appended to with ``list.append``, which is atomic under CPython.
    """

    #: Number of replicates.
    B: int
    #: One bit set per outcome, indexed by replicate.
    flags: list[int] = field(init=False)
    #: Exceptions of failed replicates, in the order they were recorded.
    failed_errors: list[BaseException] = field(init=False, default_factory=list)
    #: Exceptions of replicates whose inner `mstop` selection failed.
    untuned_errors: list[BaseException] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.flags = [0] * int(self.B)

    def mark(self, b: int, bit: int, exc: BaseException | None = None) -> None:
        """Set ``bit`` for replicate ``b``, keeping ``exc`` for the message."""
        self.flags[b] |= bit
        if exc is not None:
            if bit == FAILED:
                self.failed_errors.append(exc)
            elif bit == UNTUNED:
                self.untuned_errors.append(exc)


#: Share of replicates at the `mstop` cap above which `bootstrap_outcome`
#: warns. A few replicates at the cap are expected when the model's own
#: optimum is well inside it; each inner CV used to warn separately.
_AT_CAP_SHARE = 0.1


def bootstrap_outcome(tally: BootstrapTally, *, B: int, what: str) -> dict:
    """Raise, warn, or stay quiet; return the counts to put in the result.

    ``what`` names the quantity for the message, e.g. ``"coefficient"``.
    """
    from mboostrs import _rust

    n_failed, n_untuned, n_at_cap = _rust.bootstrap_counts(tally.flags)
    if n_failed >= B:
        first = tally.failed_errors[0]
        raise RuntimeError(
            f"every one of the {B} bootstrap replicates failed, so there is no "
            f"{what} interval to report. The first failure was "
            f"{type(first).__name__}: {first}"
        ) from first
    if n_failed:
        first = tally.failed_errors[0]
        warnings.warn(
            f"{n_failed} of {B} bootstrap replicates failed and were dropped; "
            f"the {what} interval is a quantile of the {B - n_failed} that "
            f"finished, not of {B}. The first failure was "
            f"{type(first).__name__}: {first}",
            RuntimeWarning,
            stacklevel=3,
        )
    if n_untuned:
        first = tally.untuned_errors[0]
        warnings.warn(
            f"the inner mstop cross-validation failed on {n_untuned} "
            f"of {B} bootstrap replicates; those replicates used the outer "
            f"model's mstop, so they estimate the {what} at a stopping point "
            f"chosen on the full sample rather than on their own resample. The "
            f"first failure was {type(first).__name__}: {first}",
            RuntimeWarning,
            stacklevel=3,
        )
    if n_at_cap and n_at_cap >= _AT_CAP_SHARE * B:
        warnings.warn(
            f"the inner cross-validation of {n_at_cap} of {B} bootstrap "
            f"replicates chose the largest mstop the model allows, with the "
            f"risk still falling; those replicates may be under-fitted, which "
            f"narrows the {what} interval. Refit the model with a larger "
            f"mstop before computing the interval.",
            RuntimeWarning,
            stacklevel=3,
        )
    return {
        "B": int(B),
        "n_failed": int(n_failed),
        "n_untuned": int(n_untuned),
        "n_at_mstop_cap": int(n_at_cap),
        "n_used": int(B) - int(n_failed),
    }


def group_bootstrap_counts(groups, eligible, B, seed):
    """Row multiplicities for ``B`` bootstrap samples drawn over whole groups.

    Every group with at least one eligible row is a unit. Each sample draws as
    many units as there are, with replacement, and each eligible row of a unit
    drawn ``k`` times gets multiplicity ``k``; an ineligible row gets 0. The
    rows of one group therefore enter or leave a sample together, which is
    what a bootstrap of repeated measurements (the points of one curve, the
    visits of one patient) needs: resampling the rows separately treats them
    as independent and gives intervals that are too narrow.

    The draws are made by the core (ChaCha8, seeded with ``seed``).

    Parameters
    ----------
    groups : array-like, one value per row
        Group labels.
    eligible : array-like of bool, one value per row
        Rows that may be drawn (positive weight).
    B : int
        Number of samples.
    seed : int
        Seed of the resampling stream.

    Returns
    -------
    ndarray of shape (n_rows, B)
    """
    from mboostrs import _rust

    groups = np.asarray(groups)
    eligible = np.asarray(eligible, dtype=bool)
    if groups.ndim != 1 or groups.shape != eligible.shape:
        raise ValueError(
            f"groups= needs one label per row: got {groups.shape[0] if groups.ndim else 0} "
            f"labels for {eligible.shape[0]} rows")
    _, codes = np.unique(groups, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64).ravel()
    return np.asarray(_rust.group_bootstrap_counts(
        codes.tolist(), eligible.tolist(), int(B), int(seed) % (2**64)))
