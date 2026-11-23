"""``report`` renders a :class:`~mboostrs._procedure._fit.Fit`'s decision
trail. It only reads: never fits anything.

``"json"`` returns the trail dictionary itself; ``"text"`` and ``"markdown"``
give a human-readable rendering of the same facts -- final formula, family and
mstop, provenance, one line per step, the uncertainty section ("not computed"
when there is none) and the budget.
"""

from __future__ import annotations

from typing import Any

from mboostrs._procedure._trail import DecisionTrail

__all__ = ["report"]

_FORMATS = ("text", "markdown", "json")


def _as_trail(fit_obj: Any) -> DecisionTrail:
    if isinstance(fit_obj, DecisionTrail):
        return fit_obj
    trail = getattr(fit_obj, "trail", None)
    if isinstance(trail, DecisionTrail):
        return trail
    raise TypeError("report() expects a Fit or DecisionTrail, "
                    f"got {type(fit_obj).__name__}")


def _provenance_line(trail: DecisionTrail) -> str:
    p = trail.provenance
    return (f"provenance: mboostrs={p.get('mboostrs')}, "
            f"git_sha={p.get('git_sha')}, seed={p.get('seed')}, "
            f"blas_threads={p.get('blas_threads')}")


def _step_line(step: dict[str, Any], *, bullet: str = "  -") -> str:
    number = step.get("step", "?")
    decision = step.get("decision", "?")
    adopted = step.get("adopted")
    verdict = step.get("verdict")
    reason = step.get("reason")
    pieces = [f"{bullet} step {number}: {decision}"]
    if adopted is not None:
        pieces.append(f"adopted: {adopted}")
    if verdict is not None:
        pieces.append(f"verdict: {verdict}")
    if reason:
        pieces.append(f"reason: {reason}")
    return " ".join(pieces)


def _uncertainty_section(trail: DecisionTrail) -> str:
    if trail.uncertainty.get("method") == "none":
        return "uncertainty: not computed"
    method = trail.uncertainty.get("method", "?")
    B = trail.uncertainty.get("B")
    bound = trail.uncertainty.get("expected_false_selections_bound")
    text = [f"uncertainty: {method}"]
    if B is not None:
        text.append(f"B={B}")
    if bound is not None:
        text.append(f"expected false selections bound={bound}")
    return ", ".join(text)


def _render_text(trail: DecisionTrail) -> str:
    final = trail.final
    data = trail.data
    lines = [
        "mboostrs fit: " + (final.get("formula") or "?"),
        f"family: {final.get('family')}",
        f"mstop: {final.get('mstop')}",
        f"data: n={data.get('n')}, response={data.get('response')}, "
        f"columns={', '.join(data.get('columns', {}))}",
        f"criterion: {trail.criterion}",
        _provenance_line(trail),
        f"folds: scheme={trail.folds.get('scheme')}, K={trail.folds.get('K')}, "
        f"R={trail.folds.get('R')}, hash={trail.folds.get('hash')}",
    ]
    lines.append(f"steps: {len(trail.steps)}")
    lines.extend(_step_line(step) for step in trail.steps)
    if trail.steps:
        lines.append("")
    lines.append(_uncertainty_section(trail))
    budget = trail.budget
    degradations = budget.get("degradations", [])
    lines.append(f"budget: setting={budget.get('setting')}, "
                 f"limit={budget.get('limit')}, "
                 f"seconds_limit={budget.get('seconds_limit')}, "
                 f"spent={budget.get('spent')}, "
                 f"degradations={len(degradations)}")
    return "\n".join(lines)


def _render_markdown(trail: DecisionTrail) -> str:
    final = trail.final
    data = trail.data
    lines = [
        "# mboostrs fit",
        "",
        f"- **formula:** `{final.get('formula')}`",
        f"- **family:** `{final.get('family')}`",
        f"- **mstop:** `{final.get('mstop')}`",
        f"- **data:** n={data.get('n')}, response=`{data.get('response')}`, "
        f"columns: {', '.join(data.get('columns', {}))}",
        f"- **criterion:** `{trail.criterion}`",
        f"- **provenance:** mboostrs `{trail.provenance.get('mboostrs')}`, "
        f"git_sha `{trail.provenance.get('git_sha')}`, "
        f"seed `{trail.provenance.get('seed')}`, "
        f"blas_threads `{trail.provenance.get('blas_threads')}`",
        f"- **folds:** scheme `{trail.folds.get('scheme')}`, "
        f"K={trail.folds.get('K')}, R={trail.folds.get('R')}, "
        f"hash `{trail.folds.get('hash')}`",
        "",
        "## Steps",
        "",
    ]
    if trail.steps:
        lines.extend(_step_line(step, bullet="-") for step in trail.steps)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Uncertainty")
    lines.append("")
    lines.append(f"- {_uncertainty_section(trail)}")
    budget = trail.budget
    lines.extend([
        "",
        "## Budget",
        "",
        f"- setting: `{budget.get('setting')}`",
        f"- model fits allowed: `{budget.get('limit')}`",
        f"- seconds allowed: `{budget.get('seconds_limit')}`",
        f"- spent: `{budget.get('spent')}`",
        f"- degradations: {len(budget.get('degradations', []))}",
    ])
    return "\n".join(lines)


def report(fit_obj: Any, format: str = "text") -> Any:
    """Render a ``Fit`` (or ``DecisionTrail``) as text, markdown or JSON.

    Returns a ``str`` for ``"text"``/``"markdown"`` and the trail ``dict`` for
    ``"json"``. Purely a reader of the trail -- it never fits anything.
    """
    if format not in _FORMATS:
        raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
    trail = _as_trail(fit_obj)
    if format == "json":
        return trail.to_dict()
    if format == "markdown":
        return _render_markdown(trail)
    return _render_text(trail)