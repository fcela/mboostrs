"""The unified modelling procedure behind ``mboostrs.fit`` and ``report``.

Private scaffolding for issue #474. PR 1 lands the container, the trail schema
and the seed discipline only; the search logic (column typing, family, shape,
interactions, scale, pruning, gaps, stability) lands in the later PRs named in
``docs/design/unified-procedure.md``. Nothing here is public: the top-level
names are bound in :mod:`mboostrs`.
"""

from __future__ import annotations

from mboostrs._procedure._fit import Fit, fit
from mboostrs._procedure._report import report
from mboostrs._procedure._reproduce import ReproductionResult, reproduce
from mboostrs._procedure._trail import DecisionTrail

__all__ = ["DecisionTrail", "Fit", "ReproductionResult", "fit", "report", "reproduce"]
