"""Name-keyed front end to the core's variable-importance attribution.

`varimp()` credits each boosting update's risk change to the learner it
selected. The arithmetic (the group sums, the selection shares and the
normalization to percentages) is `mboostrs_core::inference::varimp_attribution`;
this module only maps learner names to group indices and back.
"""

from __future__ import annotations

import numpy as np

__all__ = ["attribute_by_name"]


def attribute_by_name(changes, update_names, all_names, *, percent):
    """Reduction and selection frequency keyed by name.

    Parameters
    ----------
    changes : sequence of float
        Risk reduction credited to each update, already divided by the number
        of observations.
    update_names : sequence of str
        The name of the learner (or of the learner-parameter pair, for an LSS
        fit) that each update selected; one entry per update.
    all_names : sequence of str
        Every name to report, in report order. A name no update selected is
        reported with a reduction and frequency of zero. A name repeated in
        `all_names` is reported once.
    percent : bool
        When true, each reduction is divided by the sum of the absolute
        reductions, so the reductions are shares.

    Returns
    -------
    (reduction, selfreq) : two dicts keyed by name, in `all_names` order.
    """
    from mboostrs import _rust

    keys = list(dict.fromkeys(all_names))
    position = {name: i for i, name in enumerate(keys)}
    groups = [position[name] for name in update_names]
    reduction, selfreq = _rust.varimp_attribution(
        [float(c) for c in changes], groups, len(keys), bool(percent))
    return (dict(zip(keys, (float(v) for v in reduction), strict=True)),
            dict(zip(keys, (float(v) for v in selfreq), strict=True)))


def risk_changes(risk, n):
    """Per-update risk reduction, ``-diff(risk) / n``, as a float array."""
    return -1.0 * np.diff(np.asarray(risk, dtype=np.float64)) / n
