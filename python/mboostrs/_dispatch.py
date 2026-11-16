"""One place that answers "what kind of fitted model is this?".

The Python layer needs to branch on model kind in several modules — the
diagnostics builders, the conformal refit helpers, the serializer, and the
plotting entry points. Those branches were written four different ways
(`type(model).__name__ == "LSSModel"`, `not hasattr(model, "_learners_spec")`,
a tuple of class-name strings, and an instance attribute), three of which
compare class *names* as strings and so break under subclassing, while the
`hasattr` form silently inverts if the other class ever grows that attribute.

The predicates here are `isinstance` checks against the real classes. They
import those classes inside the function body: `glmboost`, `gamboost`, and
`gamlss` all import from modules that import this one, so a module-level
import would be circular.
"""

from __future__ import annotations

__all__ = ["is_lss", "is_glmboost", "is_gamboost", "is_standard",
           "model_kind_name"]


def is_lss(model) -> bool:
    """True for a distributional (`gamboostLSS`/`glmboostLSS`) fit."""
    from mboostrs.gamlss import LSSModel

    return isinstance(model, LSSModel)


def is_glmboost(model) -> bool:
    """True for a linear (`glmboost`) fit."""
    from mboostrs.glmboost import GLMBoostModel

    return isinstance(model, GLMBoostModel)


def is_gamboost(model) -> bool:
    """True for an additive (`gamboost`) fit."""
    from mboostrs.gamboost import GAMBoostModel

    return isinstance(model, GAMBoostModel)


def is_standard(model) -> bool:
    """True for either single-parameter model class.

    These two share 53 public members and are the pair the portable artifact
    format supports.
    """
    from mboostrs.gamboost import GAMBoostModel
    from mboostrs.glmboost import GLMBoostModel

    return isinstance(model, (GLMBoostModel, GAMBoostModel))


def model_kind_name(model) -> str:
    """The model's kind as a short string, for error messages."""
    if is_lss(model):
        return "LSSModel"
    if is_glmboost(model):
        return "GLMBoostModel"
    if is_gamboost(model):
        return "GAMBoostModel"
    return type(model).__name__
