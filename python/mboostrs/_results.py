"""What every result object carries, and how it is read.

A result used to be whatever the operation that built it happened to keep.
`CvRiskResult` knew its fold count and a `fold_type` string; `StabSelResult`
knew `B` and `n_fits`; `AicResult` knew nothing beyond the numbers. So a
cross-validation curve saved to disk, or pasted into a notebook a week later,
could not answer how many rows it was computed on, which family, what the
fitted budget was, whether weights were rescaled, or what seed produced the
folds -- the questions that decide whether two results may be compared.

:class:`Provenance` answers them, and hangs off every result as
``result.provenance``. It records nothing that has to be recomputed: each
field is already known at the point the result is built.

:class:`MappingResultMixin` gives every result the same reading surface. The
distributional results grew ``result["aggregate"]`` when they replaced the
R-style lists that `cvrisk_gamboostLSS` returned, and the standard ones never
had it, so generic code written against one raised ``TypeError: 'CvRiskResult'
object is not subscriptable`` on the other. Both answer now, and both take
``to_frame(output=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from mboostrs._output import export_result

__all__ = ["AicResult", "MappingResultMixin", "Provenance"]

#: Columns of ``AicResult.to_frame``, kept for the zero-row frame.
AIC_COLUMNS = (
    "mstop", "aic", "df", "is_optimal", "criterion", "normalization",
    "df_convention", "penalty",
)


@dataclass(frozen=True)
class Provenance:
    """What produced a result.

    Every field is optional because the operations that build results know
    different things: a stability-selection run knows its fit count, an AIC
    path does not resample at all. A field left `None` means "this operation
    has no such quantity", not "unknown".
    """

    #: ``"cvrisk"``, ``"stabsel"``, ``"aic"``, ``"confint"``.
    operation: str
    #: ``"glmboost"``, ``"gamboost"``, ``"glmboostLSS"``, ``"gamboostLSS"``.
    model: str | None = None
    #: The family's class name, e.g. ``"Gaussian"``, ``"GaussianLSS"``.
    family: str | None = None
    #: Rows the operation ran on.
    n_obs: int | None = None
    #: Candidate base-learners.
    n_learners: int | None = None
    #: The fitted budget: an int, or one per parameter for a distributional
    #: fit.
    mstop: object = None
    #: ``"upstream"`` or ``"raw"`` -- which weight rule the fit followed.
    weight_policy: str | None = None
    #: Sum of the effective weights the fit used. Under the upstream policy
    #: fractional weights are rescaled, so this is not the sum of what the
    #: caller passed, and two results with different values here were not
    #: computed on the same sample.
    effective_weight_total: float | None = None
    #: Seed for the resampling scheme, when this operation drew one.
    seed: int | None = None
    #: How the folds or subsamples were produced, in words.
    folds: str | None = None
    #: Models actually fitted, which for complementary-pairs stability
    #: selection is twice the subsample count.
    n_fits: int | None = None
    #: Warnings raised while producing the result, kept so they survive being
    #: pickled, saved, or read in a session that did not see stderr.
    warnings: tuple[str, ...] = ()
    #: The package version that produced it.
    version: str = ""

    def __post_init__(self):
        if not self.version:
            from mboostrs import __version__

            object.__setattr__(self, "version", __version__)

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        """A JSON-safe record of every field. ``include_arrays`` is accepted
        for the uniform export signature; provenance holds no arrays."""
        from mboostrs._output import json_safe

        return {f.name: json_safe(getattr(self, f.name)) for f in fields(self)}

    def describe(self) -> str:
        """The provenance as lines, skipping the fields this operation has no
        value for."""
        lines = [f"{self.operation} (mboostrs {self.version})"]
        labels = {
            "model": "model", "family": "family", "n_obs": "observations",
            "n_learners": "base-learners", "mstop": "mstop",
            "weight_policy": "weight policy",
            "effective_weight_total": "effective weight",
            "seed": "seed", "folds": "folds", "n_fits": "fits",
        }
        for name, label in labels.items():
            value = getattr(self, name)
            if value is not None:
                lines.append(f"  {label}: {value}")
        for message in self.warnings:
            lines.append(f"  warning: {message}")
        return "\n".join(lines)


def provenance_from_model(model, operation: str, **extra) -> Provenance:
    """Read a fitted model's own record of itself.

    Used by the operations that start from a model. The distributional
    `cvrisk`/`stabsel` entry points start from a formula and data instead, and
    build a :class:`Provenance` directly.
    """
    from mboostrs._dispatch import model_kind_name

    kind = {
        "GLMBoostModel": "glmboost",
        "GAMBoostModel": "gamboost",
        "LSSModel": getattr(model, "_model_kind", "gamboostLSS"),
    }.get(model_kind_name(model), model_kind_name(model))

    family = getattr(model, "_family", None) or getattr(model, "_families", None)
    weights = model.weights
    defaults = {
        "model": kind,
        "family": type(family).__name__ if family is not None else None,
        "n_obs": int(len(weights)),
        "n_learners": len(getattr(model, "_learner_names", []) or []) or None,
        "mstop": model.mstop,
        "weight_policy": getattr(model, "weight_policy", None),
        "effective_weight_total": float(weights.sum()),
    }
    defaults.update(extra)
    return Provenance(operation=operation, **defaults)


class MappingResultMixin:
    """``result["field"]``, ``dict(result)`` and ``to_frame(output=...)``.

    A subclass lists the keys it answers to in ``_MAPPING_KEYS`` and, where a
    key is not the attribute's own name, maps it in ``_MAPPING_ALIASES``. The
    aliases exist because the distributional results replaced R-style lists
    whose names (``"risk"``, ``"mstop"``, ``"max"``, ``"names"``) callers had
    already written against.
    """

    _MAPPING_KEYS: tuple[str, ...] = ()
    _MAPPING_ALIASES: dict[str, str] = {}

    def keys(self) -> tuple[str, ...]:
        return self._MAPPING_KEYS

    def __getitem__(self, key):
        if key not in self._MAPPING_KEYS:
            raise KeyError(
                f"{type(self).__name__} has no {key!r}; it answers to "
                f"{list(self._MAPPING_KEYS)}"
            )
        return getattr(self, self._MAPPING_ALIASES.get(key, key))

    def __contains__(self, key) -> bool:
        return key in self._MAPPING_KEYS

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def describe(self) -> str:
        """Where this result came from, as text. See :class:`Provenance`."""
        record = getattr(self, "provenance", None)
        if record is None:
            return f"{type(self).__name__}: no provenance recorded"
        return record.describe()


@dataclass
class AicResult(MappingResultMixin):
    """An information-criterion path: the value at each iteration, the
    iteration that minimises it, and the degrees of freedom behind it."""

    aic: object
    mstop: int
    df: object
    provenance: Provenance | None = field(default=None)

    _MAPPING_KEYS = ("aic", "mstop", "df")

    criterion: str = "unspecified"
    normalization: str = "unspecified"
    df_convention: str = "unspecified"
    penalty: float | None = None

    def __repr__(self):
        return (f"AicResult(criterion={self.criterion!r}, mstop={self.mstop}, "
                f"aic_min={self.aic[self.mstop - 1]:.4f})")

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the criterion and df paths only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("aic", "df"),
            extras={"n": len(self.aic)},
        )

    def __len__(self):
        return len(self.aic)

    def plot(self):
        from mboostrs.plot import plot_aic
        return plot_aic(self)

    def to_frame(self, output="auto"):
        """One row per iteration: the criterion and its degrees of freedom."""
        from mboostrs._output import output_frame

        rows = [
            {
                "mstop": index + 1,
                "aic": float(value),
                "df": float(self.df[index]),
                "is_optimal": bool(index + 1 == self.mstop),
                "criterion": self.criterion,
                "normalization": self.normalization,
                "df_convention": self.df_convention,
                "penalty": self.penalty,
            }
            for index, value in enumerate(self.aic)
        ]
        return output_frame(rows, output, columns=AIC_COLUMNS)
