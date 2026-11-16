"""The procedure's random streams, and the BLAS thread count it ran under.

One ``seed`` produces every stream through ``numpy.random.SeedSequence(seed)``
so that a trail records one number and a run can be reproduced (design note
section 3.6). The streams are spawned in a fixed order -- ``folds``, ``inner``,
``btree``, ``stability`` -- because the order is part of the seed contract: two
runs that spawn in different orders would draw different numbers from the same
seed.

``blas_threads`` is best effort. It is recorded for provenance only: it changes
between machines, which is why section 3.6 promises bit-identical trails on one
platform rather than across all of them.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["STREAMS", "blas_threads", "streams"]

#: The stream names, in the order they are spawned. Fixed: see the module
#: docstring.
STREAMS = ("folds", "inner", "btree", "stability")

_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def streams(seed: int) -> dict[str, np.random.Generator]:
    """The four independent generators one seed produces.

    ``SeedSequence(seed).spawn(4)`` hands each stream its own independent child
    sequence, so drawing from ``folds`` cannot shift what ``btree`` draws. The
    dictionary order matches :data:`STREAMS`.
    """
    children = np.random.SeedSequence(int(seed)).spawn(len(STREAMS))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAMS, children, strict=True)
    }


def blas_threads() -> int | None:
    """The BLAS thread count in force, or ``None`` when it cannot be read.

    Reads ``threadpoolctl`` when it is installed, which reports the limits the
    loaded numerical libraries actually use. Otherwise it falls back to the
    usual environment variables -- the value a process would run with, not the
    library's own default -- and returns ``None`` when neither says.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        pass
    else:
        try:
            counts = [
                int(pool.get("num_threads", 0))
                for pool in threadpool_info()
                if int(pool.get("num_threads", 0)) > 0
            ]
        except Exception:  # noqa: BLE001 - provenance must never fail a fit
            counts = []
        if counts:
            return max(counts)
    for name in _BLAS_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None
