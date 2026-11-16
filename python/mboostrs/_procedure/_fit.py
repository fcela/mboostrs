"""Model fitting and the `Fit` result container."""

from __future__ import annotations

from typing import Any

from mboostrs._procedure._stages import inputs_step
from mboostrs._procedure._trail import DecisionTrail

__all__ = ["Fit", "fit"]

_BUDGETS = ("quick", "standard", "thorough")
_UNCERTAINTIES = ("stability", "bootstrap", "none")


class Fit:
    """A fitted model wrapped with the trail that produced it.

    ``model`` is the underlying fitted model (a ``GAMBoostModel`` or
    ``LSSModel``), ``trail`` is the :class:`~mboostrs._procedure._trail.DecisionTrail`
    explaining how it was chosen, and ``stability`` is currently ``None``
    because selection-aware uncertainty is not yet supported.
    ``predict`` delegates to the model, so a ``Fit`` can stand in wherever the
    model itself was used.
    """

    def __init__(self, model: Any, trail: DecisionTrail, stability: Any = None):
        self.model = model
        self.trail = trail
        self.stability = stability

    def predict(self, newdata: Any, **kwargs: Any) -> Any:
        """Predict with the wrapped model. See ``model.predict``."""
        return self.model.predict(newdata, **kwargs)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """The wrapping record: the decision trail as a JSON-safe dict."""
        del include_arrays  # accepted for the shared to_dict() convention
        return self.trail.to_dict()

    def __repr__(self) -> str:
        final = self.trail.final
        family = final.get("family")
        mstop = final.get("mstop")
        return f"Fit(family={family!r}, mstop={mstop!r}, steps={len(self.trail.steps)})"


def _formula_text(formula: Any) -> str:
    if isinstance(formula, str):
        return formula
    if isinstance(formula, dict):
        return "; ".join(f"{name}: {value}" for name, value in formula.items())
    return str(formula)


def fit(
    data: Any,
    response: str,
    *,
    family: Any = None,
    folds: Any = None,
    groups: Any = None,
    time: Any = None,
    seed: int = 0,
    budget: str = "standard",
    uncertainty: str = "none",
    **kwargs: Any,
) -> Fit:
    """Fit one model and return it with the trail of how it was chosen.

    ``fit`` uses ``autospec`` to choose the specification and returns the
    fitted model with a trail containing the input declarations, specification
    decision, final formula, family and stopping iteration.

    Parameters
    ----------
    data : DataFrame or dict
        Polars/pandas frame or mapping of column name to array.
    response : str
        The response column.
    family : Family, optional
        Pinned family, passed to ``autospec``.
    folds : optional
        Fold scheme or weight matrix, passed to ``autospec``.
    groups, time : optional
        Declared dependence structure. ``groups`` is passed to ``autospec``;
        ``time`` currently raises ``NotImplementedError``.
    seed : int, default 0
        The single seed for every random stream.
    budget : {"quick", "standard", "thorough"}
        Accepted values are ``"quick"``, ``"standard"`` and ``"thorough"``.
        The setting is validated but does not currently change the compute
        budget.
    uncertainty : {"none", "stability", "bootstrap"}
        ``"none"`` is currently supported. ``"stability"`` and
        ``"bootstrap"`` raise ``NotImplementedError``.
    **kwargs
        Passed through to ``autospec``.
    """
    if budget not in _BUDGETS:
        raise ValueError(f"budget must be one of {_BUDGETS}, got {budget!r}")
    if uncertainty not in _UNCERTAINTIES:
        raise ValueError(
            f"uncertainty must be one of {_UNCERTAINTIES}, got {uncertainty!r}"
        )
    if uncertainty != "none":
        raise NotImplementedError(
            f"uncertainty={uncertainty!r} is not implemented until #474 PR 10 "
            "(selection-aware stability over the whole search)"
        )
    if time is not None:
        raise NotImplementedError(
            "time= (rolling folds ordered by a time column) is not implemented "
            "in PR 1; it arrives with the fold-construction PR"
        )

    from mboostrs.autospec import autospec

    spec = autospec(data, response, family=family, seed=seed, folds=folds,
                    groups=groups, **kwargs)
    model = spec.fit(data)

    trail = DecisionTrail()
    trail.start(data, response, seed, folds=folds)
    formula = _formula_text(spec.formula)
    trail.steps.append(inputs_step(spec))
    # Stages 1-12 still record only through autospec's evidence; each moves
    # to its own step in the order of docs/design/procedure-stages.md.
    trail.add_step(
        "1-12", "autospec_delegate",
        incumbent=None,
        candidates=[{
            "spec": formula,
            "score": None,
            "d": None,
            "se": None,
            "status": "ok",
            "reason": "delegated specification from autospec (PR 1 scaffolding)",
        }],
        threshold=None,
        adopted=formula,
        rule=None,
        reason=(
            "PR 1 scaffolding: the specification comes from autospec. The "
            "full search (steps 1-7b of the design note) lands in later PRs."
        ),
    )
    trail.set_final(formula, spec.family, spec.mstop)
    trail.set_uncertainty("none")
    trail.set_budget(budget, spent=0.0, degradations=[])
    return Fit(model=model, trail=trail)
