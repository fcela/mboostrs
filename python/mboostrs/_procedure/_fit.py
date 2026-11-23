"""Model fitting and the `Fit` result container."""

from __future__ import annotations

from typing import Any

from mboostrs._procedure._stages import inputs_step
from mboostrs._procedure._trail import DecisionTrail

__all__ = ["Fit", "fit"]

_BUDGETS = ("quick", "standard", "thorough")
# Model fits each setting lets autospec's probes spend. Measured on ten
# competitive-shootout panels, three seeds, one thread (#913 item 2): mean test
# loss relative to the best budget had median 3.1% / worst 44.6% at 16 fits,
# 0.1% / 8.4% at 32 and 0.0% / 15.8% at 512, with median wall-clock 0.45 s,
# 4.5 s and 24 s. autospec stopped on its own at 161 fits or fewer, so 512
# leaves room for wider tables rather than buying more search on these.
_BUDGET_FITS = {"quick": 16, "standard": 32, "thorough": 512}
# Wall-clock seconds each setting gives the search. On 36 held-out OpenML
# datasets (#913) the slowest completed "standard" search took 51 s and the
# slowest "quick" search outside one 1,025-column table took 33 s; that table
# exceeded 600 s at both settings. The limits are about twice the slowest
# completed run, so they bind on outliers rather than on typical tables.
_BUDGET_SECONDS = {"quick": 60.0, "standard": 120.0, "thorough": 1200.0}
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


def _timed_search(data: Any, response: str, limit: int, seconds: float,
                  **kwargs: Any) -> tuple[Any, str | None]:
    """Run autospec with a fit budget and a wall-clock limit.

    Returns the recommendation and ``None``, ``"partial"`` (the time ran out
    inside a probe that keeps its best result so far) or ``"fallback"`` (the
    time ran out where autospec cannot recover, so the recommendation is
    autospec's measured-nothing default, the budget-0 specification). The
    limit is checked whenever a probe asks to start a model fit, so a fit
    already running finishes first. Inside an outer workflow budget (for
    example a ``SearchSession``), that budget governs cost and no limit is
    added here.
    """
    from mboostrs._workflow import _CURRENT, BudgetExhausted, Workflow
    from mboostrs.autospec import autospec
    from mboostrs.policy import CostBudget

    if _CURRENT.get() is not None:
        return autospec(data, response, budget=limit, **kwargs), None
    tracker = Workflow(CostBudget(seconds=seconds))
    try:
        with tracker:
            spec = autospec(data, response, budget=limit, **kwargs)
    except BudgetExhausted:
        return autospec(data, response, budget=0, **kwargs), "fallback"
    return spec, ("partial" if tracker.stopped else None)


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
    na_action: str | None = None,
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
        The number of model fits and the wall-clock time the specification
        search is given: 16 fits and 60 s, 32 fits and 120 s, 512 fits and
        1200 s. Each fit is one cross-validated or held-out evaluation of a
        candidate family, learner, interaction or stopping depth. With fewer
        fits more decisions fall back to the dtype-driven default learner for
        each column. ``"standard"`` took a median of 4.5 s on benchmark
        panels of up to 4,000 rows at one thread; ``"quick"`` took 0.45 s and
        lost up to 45% in test loss on some of them.

        Neither limit is exact. autospec checks the fit count before each
        stage starts, so a stage in progress can exceed it (on 36 held-out
        OpenML datasets the search spent up to 60 fits at 32 and 45 at 16).
        The time is checked before each model fit starts, so the search can
        overrun by the length of one fit. When the time runs out the search
        keeps what it measured if the probe in progress allows it, and
        otherwise returns the default specification (one default learner per
        column, nothing measured). The trail's ``budget`` records both limits
        and the fits spent, and ``degradations`` says when a limit stopped the
        search and which of those outcomes occurred.
    uncertainty : {"none", "stability", "bootstrap"}
        ``"none"`` is currently supported. ``"stability"`` and
        ``"bootstrap"`` raise ``NotImplementedError``.
    na_action : {"impute", "impute_model", "impute_model_split", "omit", "error"}, optional
        Missing-value policy passed through to ``autospec`` and the final fit.
        ``None`` (default) drops incomplete rows when missing values are present.
    **kwargs
        Passed through to ``autospec``.
    """
    if budget not in _BUDGETS:
        raise ValueError(
            f"budget must be one of {_BUDGETS}, got {budget!r}; for an explicit "
            "number of model fits call autospec(..., budget=n) directly")
    if uncertainty not in _UNCERTAINTIES:
        raise ValueError(
            f"uncertainty must be one of {_UNCERTAINTIES}, got {uncertainty!r}"
        )
    if uncertainty != "none":
        raise NotImplementedError(
            f"uncertainty={uncertainty!r} is not yet supported by fit(); "
            "pass uncertainty='none' and compute intervals with "
            "conformal_interval() or gam_refit()"
        )
    if time is not None:
        raise NotImplementedError(
            "time= (rolling folds ordered by a time column) is not yet "
            "supported by fit(); pass explicit rolling folds via folds="
        )

    limit, seconds = _BUDGET_FITS[budget], _BUDGET_SECONDS[budget]
    spec, stopped = _timed_search(data, response, limit, seconds, family=family,
                                  seed=seed, folds=folds, groups=groups,
                                  na_action=na_action, **kwargs)
    model = spec.fit(data)

    trail = DecisionTrail()
    trail.start(data, response, seed, folds=folds)
    formula = _formula_text(spec.formula)
    trail.steps.append(inputs_step(spec))
    # Stages 1-12 currently record through autospec's evidence.
    trail.add_step(
        "1-12", "autospec_delegate",
        incumbent=None,
        candidates=[{
            "spec": formula,
            "score": None,
            "d": None,
            "se": None,
            "status": "ok",
            "reason": f"specification chosen by autospec within {limit} model fits",
        }],
        threshold=None,
        adopted=formula,
        rule=None,
        fits=int(spec.fits),
        reason=(
            f"autospec ran its stages 1-12 with a budget of {limit} model fits "
            f"and spent {int(spec.fits)}."
        ),
    )
    trail.set_final(formula, spec.family, spec.mstop)
    trail.set_uncertainty("none")
    degradations = []
    if stopped == "fallback":
        degradations.append(
            f"search exceeded its {seconds:g} s limit and could not keep a "
            "partial result; the specification is autospec's default with "
            "nothing measured (budget 0)")
    elif stopped == "partial":
        degradations.append(
            f"search reached its {seconds:g} s limit; decisions after that "
            "point kept their default learner")
    if stopped != "fallback" and spec.fits >= limit:
        degradations.append(
            f"search stopped at the budget of {limit} model fits; decisions it "
            "could not pay for kept their default learner (see rec.report())")
    trail.set_budget(budget, spent=float(spec.fits), degradations=degradations,
                     limit=limit, seconds_limit=seconds)
    return Fit(model=model, trail=trail)
