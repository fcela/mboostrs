"""Named stages of the search behind ``fit`` (issue #474).

``STAGES`` lists autospec's stages in run order. A stage function returns one
trail step: its index, name, the schema's adoption fields, the fits it spent
and an ``evidence`` mapping. Plan: ``docs/design/procedure-stages.md``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["STAGES", "inputs_step"]

STAGES = (
    "inputs", "screen", "depth", "family", "scale", "shapes", "interactions",
    "residual_tree", "deselect", "pair_form", "coordinate_surface",
    "joint_tree", "routes", "gaps", "uncertainty",
)


def inputs_step(spec: Any) -> dict[str, Any]:
    """Stage 0: what the response is, what the data checks found, and what
    the caller declared. No model is fitted in this stage."""
    response = dict(spec.evidence.get("response", {}))
    return {
        "step": STAGES.index("inputs"),
        "stage": "inputs",
        "decision": "inputs",
        "incumbent": None,
        "candidates": [],
        "adopted": response.get("family"),
        "rule": response.get("source"),
        "reason": response.get("sentence"),
        "fits": 0,
        "evidence": {
            "response": response,
            "hygiene": [
                {"column": f.column, "kind": f.kind, "detail": f.detail,
                 "consequence": f.consequence}
                for f in spec.hygiene
            ],
            "incomplete_rows": int(spec.incomplete_rows),
            "declared": list(spec.declared),
            "groups": list(spec.groups),
            "measurements": list(spec.measurements),
            "treatment": list(spec.treatment),
            "interaction_rule": dict(spec.evidence.get("interaction_rule", {})),
        },
    }
