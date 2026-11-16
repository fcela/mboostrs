"""Tensor-product interaction learners via the ``%X%`` operator.

A formula term ``bbs(x) %X% bols(g)`` fits the interaction of two base
learners: a smooth curve whose shape varies with the group ``g``, penalized
jointly. ``bbs(x) %X% bbs(z)`` fits an interaction surface with independent
marginal penalties — more general than ``bspatial``, which ties the two
smoothness levels together.

The construction reuses tested machinery rather than adding an engine variant.
An interaction of two marginals with designs ``X1`` (n×p1) and ``X2`` (n×p2)
is the row-wise tensor (Khatri-Rao) product ``T`` with ``T[i] =
outer(X1[i], X2[i]).ravel()`` (n×p1p2), penalized by the Kronecker sum
``K1 ⊗ I + I ⊗ K2``. That is exactly what the already-parity-tested ``buser``
learner fits: a caller-supplied design and penalty.

The tensor carries one ``df``, and it is the product of the two marginal
``df``s — 16 for two ``bbs`` margins at the default 4, R `mboost`'s rule for the
same operator. A margin that fixes ``lambda`` instead has it multiplied into its
block of the Kronecker sum; when both do, the tensor takes ``lambda = 1`` and no
``df``, since there is nothing left to calibrate. ``bols`` fixes ``lambda = 0``
unless told otherwise, so an unadorned linear margin contributes no penalty and
counts as 1 in the product.

Because the Kronecker sum's null space is the tensor product of the marginal
null spaces, the tensor ``df`` has a floor: two second-difference penalties
leave the bilinear functions unpenalized, four directions, and no ``lambda``
shrinks the term below ``df = 4``. Marginal ``df``s whose product lands at or
under that floor warn — the fit is the maximally shrunk one, but ``df=`` has
stopped setting the flexibility.

Prediction rebuilds each marginal design at the new points — smooth marginals
through the engine's own ``bbs_raw_design`` (training boundary knots), linear
and factor marginals from their recorded encoding — then forms the tensor
again. The marginal recipes are stored on the fitted model so a reloaded or
sliced model predicts identically.

Supported marginals: ``bbs``, ``bols``, ``brandom``. Each side may be either.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from mboostrs._errors import MboostrsValueError
from mboostrs._names import NAME_SRC as N
from mboostrs._names import partition_tilde, quote_name, unquote_name

__all__ = [
    "expand_interactions",
    "learner_design",
    "LearnerDesign",
    "TensorInteraction",
]

# `<learner>(col[, args])` — one marginal of an interaction.
_MARGINAL_RE = re.compile(rf"^\s*(bbs|bols|brandom)\s*\(\s*({N})\s*(?:,\s*(.*))?\)\s*$")
#: A spline or linear margin on a transformed column, ``bbs(log(x), ...)``.
_TRANSFORMED_MARGINAL_RE = re.compile(
    rf"^\s*(bbs|bols)\s*\(\s*(log|sqrt)\s*\(\s*({N})\s*\)\s*(?:,\s*(.*))?\)\s*$")
_OP = "%X%"


@dataclass(frozen=True)
class _Marginal:
    kind: str            # "bbs" | "bols" | "brandom"
    column: str
    # bbs: boundary_knots, n_knots, degree; bols: degree; brandom: levels
    params: dict


@dataclass(frozen=True)
class TensorInteraction:
    """A stored interaction: its two marginals and the derived buser column."""

    name: str
    derived: str
    left: _Marginal
    right: _Marginal

    def describe(self) -> dict:
        return {
            "name": self.name,
            "derived": self.derived,
            "left": {"kind": self.left.kind, "column": self.left.column},
            "right": {"kind": self.right.kind, "column": self.right.column},
        }


class LearnerDesign:
    """A fitted design/penalty recipe for ``buser`` and inspection.

    ``design`` and ``penalty`` are the matrices built on the fitting data.
    :meth:`transform` evaluates the same fitted recipe on new rows, preserving
    spline boundary knots and factor levels. The object currently supports the
    marginal learners accepted by ``%X%``: ``bbs``, ``bols`` and ``brandom``.
    """

    def __init__(self, term: str, data):
        self.term = term.strip()
        self._recipe = _parse_marginal(term, data)
        self.design, self.penalty = _marginal_design(self._recipe, data)

    def transform(self, data, *, unseen: str = "error") -> np.ndarray:
        """Evaluate the fitted design on ``data`` without refitting its basis.

        ``unseen="zero"`` gives rows with a factor level the fit never saw
        all-zero indicators instead of raising.
        """
        return _marginal_design(self._recipe, data, unseen=unseen)[0]


# --------------------------------------------------------------------------
# marginal designs
# --------------------------------------------------------------------------


#: `df` and `lambda` each marginal carries when the formula does not say. These
#: are R `mboost`'s own defaults: `bbs` and `brandom` are penalized and
#: calibrated through `df=4`, `bols` is unpenalized (`lambda=0`) and has no `df`.
#: Writing one of the two blanks the other, as `bbs()`, `bols()` and `brandom()`
#: do — `df` and `lambda` are two spellings of one quantity, and `df2lambda`
#: refuses to receive both.
_MARGINAL_HYPER_DEFAULTS = {
    "bbs": {"df": 4.0, "lambda_value": None},
    "brandom": {"df": 4.0, "lambda_value": None},
    "bols": {"df": None, "lambda_value": 0.0},
}


#: What each marginal reads out of its own call. Everything here reaches the
#: composed tensor: `df`/`lambda` through `_marginal_hyper`, the basis shape
#: through `_bbs_defaults`, `intercept` through the `bols` branch. A keyword
#: outside the set would be parsed and then dropped, so the fit would differ
#: from what the formula says and nothing would report it.
_MARGINAL_ARGUMENTS = {
    "bbs": ("knots", "degree", "differences", "df", "lambda", "boundary.knots", "lambda_value"),
    "bols": ("intercept", "df", "lambda", "lambda_value"),
    "brandom": ("df", "lambda", "lambda_value"),
}


_NAMED_ARG_RE = re.compile(r"([\w.]+)\s*=\s*(c\([^()]*\)|[\w.+-]+)")


def _boundary_knots_arg(args: str | None):
    """The `boundary.knots=c(lo, hi)` of a bbs call as a pair of floats, or None."""
    value = _parse_args(args).get("boundary.knots")
    if value is None:
        return None
    inner = value.strip()[2:-1] if value.strip().startswith("c(") else value
    lo, hi = (float(piece) for piece in re.findall(r"[^,\s]+", inner))
    if not lo < hi:
        raise ValueError(f"boundary.knots must be c(lo, hi) with lo < hi; got {value}")
    return lo, hi


def _parse_args(args: str | None, kind: str | None = None) -> dict:
    """`key=value` pairs of a marginal call, as strings.

    With `kind`, a keyword the marginal does not carry into the tensor is an
    error rather than a silent drop: `bbs(x, center=True)` would otherwise fit
    an uncentered basis under a name that says otherwise.
    """
    if not args:
        return {}
    # A value may be a c(...) vector, as in boundary.knots=c(0, 10); the key is
    # matched whole so that boundary.knots is not read as knots.
    parsed = dict(_NAMED_ARG_RE.findall(args))
    if kind is not None:
        accepted = _MARGINAL_ARGUMENTS[kind]
        unknown = [key for key in parsed if key not in accepted]
        if unknown:
            raise ValueError(
                f"{kind}() marginal of an interaction does not carry "
                f"{', '.join(repr(key) for key in sorted(unknown))} into the "
                f"tensor; it accepts {', '.join(accepted[:-1])}"
            )
    return parsed


def _marginal_hyper(kind: str, args: str | None) -> dict:
    """The `df` and `lambda` this marginal contributes to the tensor."""
    hyper = dict(_MARGINAL_HYPER_DEFAULTS[kind])
    parsed = _parse_args(args)
    if "df" in parsed:
        hyper["df"] = float(parsed["df"])
        hyper["lambda_value"] = None
    if "lambda" in parsed or "lambda_value" in parsed:
        hyper["lambda_value"] = float(parsed.get("lambda", parsed.get("lambda_value")))
        hyper["df"] = None
    return hyper


def _bbs_defaults(args: str | None) -> dict:
    knots, degree, differences = 20, 3, 2
    parsed = _parse_args(args)
    if "knots" in parsed:
        knots = int(parsed["knots"])
    if "degree" in parsed:
        degree = int(parsed["degree"])
    if "differences" in parsed:
        differences = int(parsed["differences"])
    return {"n_knots": knots, "degree": degree, "differences": differences}


def _raw_column(data, column):
    values = data[column]
    return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)


def _fit_marginal(kind: str, column: str, args: str | None, data,
                  transform: str | None = None) -> _Marginal:
    """Record what is needed to rebuild this marginal's design at new points.

    `transform` (``"log"`` or ``"sqrt"``) fits the margin on the transformed
    column; the training minimum is recorded so new values are transformed
    the same way."""
    _parse_args(args, kind)  # refuse a keyword the tensor would not carry
    hyper = _marginal_hyper(kind, args)
    if kind == "brandom":
        raw = _raw_column(data, column)
        levels = tuple(sorted({v for v in raw.tolist()}))
        return _Marginal(kind, column, {"levels": levels, **hyper})

    from mboostrs.transforms import _column

    x = _column(data, column)
    scale = None
    if transform is not None:
        raw = np.asarray(x, dtype=float)
        lower = float(np.min(raw[np.isfinite(raw)]))
        if transform == "log" and lower <= 0:
            raise ValueError(f"log({quote_name(column)}) needs positive values; the minimum is {lower:g}")
        if transform == "sqrt" and lower < 0:
            raise ValueError(f"sqrt({quote_name(column)}) needs non-negative values; the minimum is {lower:g}")
        scale = (transform, lower)
        x = _transformed(raw, scale)
        hyper = {**hyper, "transform": scale}
    if kind == "bbs":
        cfg = _bbs_defaults(args)
        # The knots span the observed values; a missing value is filled later
        # (na_action="impute") or refused, and must not reach the boundary.
        observed = np.asarray(x, dtype=float)
        observed = observed[np.isfinite(observed)]
        # A declared boundary.knots pair is the basis's range, as in the boosted
        # fit; the refit must rebuild the same basis.
        boundary = _boundary_knots_arg(args) or (float(np.min(observed)), float(np.max(observed)))
        return _Marginal(
            kind, column,
            {
                "boundary_knots": boundary,
                "n_knots": cfg["n_knots"],
                "degree": cfg["degree"],
                "differences": cfg["differences"],
                **hyper,
            },
        )
    if kind == "bols":
        parsed = _parse_args(args)
        intercept = parsed.get("intercept", "True") not in ("False", "FALSE", "false")
        return _Marginal(kind, column, {**hyper, "intercept": intercept})
    raise ValueError(f"unsupported interaction marginal {kind!r}")


def _with_training_fill(marginal: _Marginal, data, na_action: str) -> _Marginal:
    """The marginal with the training median recorded as its fill value.

    Under ``na_action="impute"`` a numeric margin's missing values are read
    as this median, in the fit and in every prediction, so the stored
    recipe carries the fill. The other na_action values need no fill: "omit"
    has dropped the rows before the tensor is built, and "error" refuses them
    in :func:`_margin_values`.
    """
    if na_action not in ("impute", "impute_model", "impute_model_split") or marginal.kind not in ("bbs", "bols"):
        return marginal
    from mboostrs.transforms import _median

    values = np.asarray(_raw_column(data, marginal.column), dtype=float)
    present = values[np.isfinite(values)]
    if present.size == values.size or present.size == 0:
        return marginal
    return _Marginal(marginal.kind, marginal.column,
                     {**marginal.params, "impute_value": _median(present)})


def _margin_values(marginal: _Marginal, data) -> np.ndarray:
    """The margin's raw column as floats, with its training fill applied.

    A missing value without a fill is refused here, before the Rust basis is
    built, with the column and the two remedies named.
    """
    from mboostrs.transforms import _column

    values = np.asarray(_column(data, marginal.column), dtype=float)
    missing = ~np.isfinite(values)
    if not missing.any():
        return values
    fill = marginal.params.get("impute_value")
    if fill is None:
        first = int(np.flatnonzero(missing)[0])
        raise MboostrsValueError(
            f"the interaction term's margin {marginal.column!r} has "
            f"{int(missing.sum())} missing value(s), first at row {first}. Pass "
            f"na_action='impute' to fill them with the training median, or "
            f"na_action='omit' to drop those rows, or fill the column yourself "
            f"before fitting.")
    return np.where(missing, fill, values)


def _marginal_design(marginal: _Marginal, data, *, unseen: str = "error"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(design, penalty)`` for a marginal at the rows in `data`.

    A marginal that fixes its own ``lambda`` has it multiplied into the penalty
    here, which is how a per-margin smoothing level survives into a tensor that
    carries a single ``lambda`` of its own. ``bols`` fixes ``lambda = 0`` unless
    the call says otherwise, so its default block is zero and the interaction is
    penalized along the other margin only.
    """
    design, penalty = _raw_marginal_design(marginal, data, unseen=unseen)
    lambda_value = marginal.params.get("lambda_value")
    if lambda_value is not None:
        penalty = lambda_value * penalty
    return design, penalty


def _raw_marginal_design(marginal: _Marginal, data, *, unseen: str = "error") -> tuple[np.ndarray, np.ndarray]:
    """The marginal's design and its unscaled penalty, built by the Rust core
    from the marginal's basis recipe (`mboostrs_core::gam::TermBasis`).

    - `bbs`: B-splines with the recorded boundary knots, extrapolated
      linearly outside them, and a difference penalty; cyclic when the
      marginal says so (wrapped basis, circulant penalty).
    - `bols`: `[1, x]` or `x`, with a ridge penalty over every column, as R
      `bols` builds it. It is scaled by the marginal's lambda, which is 0 by
      default: an unadorned `bols` margin contributes no penalty at all.
    - `byfactor`: R codes a factor `by` with treatment contrasts,
      `model.matrix(~ g)[, -1]`, so the first level is the reference and
      gets no block; the term fits the difference from that level. The
      ridge penalty on this margin reduces the Kronecker sum to
      `kron(I, K_learner)`, the block-diagonal penalty R builds.
    - `brandom`: full dummy coding, one column per level, as R `brandom`
      passes `contrasts.arg = "contr.dummy"`.
    - `contrasts`: R `bols(f)` for a factor `f`, the treatment-contrast
      design `model.matrix(~ f)` (`[1, d_2, ..., d_K]`, or `[d_2, ..., d_K]`
      with `intercept = FALSE`) with a ridge penalty over every column.
    - `unit`: the constant column, unpenalized. A tensor with it is the other
      margin unchanged, which is how a `contrasts` design rides in a
      `TensorInteraction` recipe.
    """
    from mboostrs import _rust

    p = marginal.params
    if marginal.kind == "bbs":
        recipe = {"kind": "bbs", "boundary": tuple(float(v) for v in p["boundary_knots"]),
                  "n_knots": int(p["n_knots"]), "degree": int(p["degree"]),
                  "differences": int(p.get("differences", 2)),
                  "cyclic": bool(p.get("cyclic", False))}
        values = np.ascontiguousarray(
            _transformed(_margin_values(marginal, data), p.get("transform")), dtype=float)
    elif marginal.kind == "bols":
        recipe = {"kind": "bols", "intercept": bool(p.get("intercept", True)), "ridge": True}
        values = np.ascontiguousarray(
            _transformed(_margin_values(marginal, data), p.get("transform")), dtype=float)
    elif marginal.kind in ("byfactor", "brandom"):
        recipe = {"kind": marginal.kind,
                  "levels": [float(j + 1) for j in range(len(p["levels"]))]}
        values = _level_codes(marginal, data, unseen=unseen)
        design, penalties = _rust.term_basis_design(recipe, [values], unseen_zero=True)
        design = _average_unseen_rows(np.asarray(design), values, len(p["levels"]), unseen)
        return design, np.asarray(penalties[0])
    elif marginal.kind == "contrasts":
        recipe = {"kind": "byfactor",
                  "levels": [float(j + 1) for j in range(len(p["levels"]))]}
        values = _level_codes(marginal, data, unseen=unseen)
        design = np.asarray(_rust.term_basis_design(recipe, [values], unseen_zero=True)[0])
        design = _average_unseen_rows(design, values, len(p["levels"]), unseen)
        if p.get("intercept", True):
            design = np.column_stack([np.ones(design.shape[0]), design])
        return design, np.eye(design.shape[1])
    elif marginal.kind == "unit":
        return np.ones((len(_raw_column(data, marginal.column)), 1)), np.zeros((1, 1))
    else:
        raise ValueError(f"unsupported marginal {marginal.kind!r}")
    design, penalties = _rust.term_basis_design(recipe, [values], unseen_zero=True)
    return np.asarray(design), np.asarray(penalties[0])


def _average_unseen_rows(design, codes, n_levels, unseen):
    """Under ``unseen="average"``, give each unseen-level row (code 0) the
    mean of the level indicator rows: ``1/L`` in every level's column. A
    treatment-coded margin has no column for its reference level, whose
    effect is zero, so the ``L - 1`` columns still get ``1/L`` each. The term
    then predicts such a row at the mean of its ``L`` level effects, the
    "average group" `brandom` uses."""
    if unseen != "average":
        return design
    rows = np.asarray(codes) == 0
    if np.any(rows):
        design = np.array(design, dtype=float, copy=True)
        design[rows, :] = 1.0 / n_levels
    return design


def _level_kind_hint(levels, new_levels) -> str | None:
    """Why unseen levels may be the fitted ones under another coding.

    Labels (strings) arriving where the fit recorded numbers, or numbers
    where it recorded labels, are usually one frame encoded and the other
    not, not new groups: ``unseen_levels`` would then average or zero every
    row. Returns the sentence to say so, or ``None`` when the kinds agree.
    """
    def kind(values):
        if all(isinstance(v, str) for v in values):
            return "labels"
        if all(isinstance(v, (int, float, np.integer, np.floating))
               and not isinstance(v, bool) for v in values):
            return "numbers"
        return None

    fitted, incoming = kind(list(levels)), kind(list(new_levels))
    if fitted is None or incoming is None or fitted == incoming:
        return None
    return (f"the fit recorded {fitted} and these are {incoming}, so the "
            f"frame is probably coded differently from the one the fit saw "
            f"(factor codes on one side, labels on the other); pass the "
            f"frame as the fit read it rather than setting unseen_levels, "
            f"which would give every such row the average group")


def _level_codes(marginal: _Marginal, data, *, unseen: str = "error") -> np.ndarray:
    """Each row's level as its 1-based position among the recorded levels;
    0 for a level the fit never saw.

    ``unseen="zero"`` lets such rows through: their indicator row is all zero,
    so a random effect contributes nothing there, the population prediction
    for a new group. ``unseen="average"`` lets them through too, and the
    caller replaces the row by the mean of the level rows. Otherwise they
    raise. A level that differs from a fitted one only in dtype raises
    under every policy instead (`transforms._unseen_dtype_mismatch`): it is
    not a new group, and the policies would silently average or zero it.
    """
    from mboostrs.transforms import _unseen_dtype_mismatch

    raw = _raw_column(data, marginal.column)
    levels = marginal.params["levels"]
    listed = raw.tolist()
    mismatch = _unseen_dtype_mismatch(marginal.column, levels, listed, raw)
    if mismatch is not None:
        raise mismatch
    index = {level: j + 1 for j, level in enumerate(levels)}
    new_levels = sorted({v for v in listed if v not in index}, key=str)
    if new_levels and unseen not in ("zero", "average"):
        raise ValueError(
            f"column {marginal.column!r} has level(s) {new_levels} not seen at "
            f"fit time (levels: {list(levels)}); "
            + (_level_kind_hint(levels, new_levels)
               or "a random effect (`brandom`) can predict them at its average "
                  "with unseen_levels='zero_effect'")
        )
    return np.asarray([float(index.get(v, 0)) for v in listed])


def _factor_indicators(marginal: _Marginal, data, *, unseen: str = "error") -> np.ndarray:
    """One indicator column per recorded level, in level order (an all-zero
    row for an unseen level with ``unseen="zero"``)."""
    from mboostrs import _rust

    recipe = {"kind": "brandom",
              "levels": [float(j + 1) for j in range(len(marginal.params["levels"]))]}
    codes = _level_codes(marginal, data, unseen=unseen)
    return np.asarray(_rust.term_basis_design(recipe, [codes], unseen_zero=True)[0])


def _tensor(design_l: np.ndarray, design_r: np.ndarray) -> np.ndarray:
    """Row-wise (Khatri-Rao) tensor product, computed by the Rust core."""
    from mboostrs import _rust

    return np.asarray(_rust.row_tensor(np.ascontiguousarray(design_l, dtype=float),
                                       np.ascontiguousarray(design_r, dtype=float)))


def _marginal_null_dim(marginal: _Marginal) -> int:
    """How many directions of this marginal its penalty leaves free.

    A zero penalty — what `bols` carries at its default `lambda = 0`, and what
    a factor `by=` margin always carries — leaves all of them. A difference
    penalty of order `d` leaves the polynomials below degree `d`. A ridge
    penalty leaves none.
    """
    if marginal.params.get("lambda_value") == 0.0:
        if marginal.kind == "bols":
            return 2 if marginal.params.get("intercept", True) else 1
        if marginal.kind == "byfactor":
            return len(marginal.params["levels"]) - 1
        return len(marginal.params["levels"])
    if marginal.kind == "bbs":
        return int(marginal.params.get("differences", 2))
    return 0


def _warn_if_df_is_at_the_floor(name: str, df: float, left: _Marginal, right: _Marginal) -> None:
    """Warn when the tensor `df` is one no finite lambda reaches.

    The null space of the Kronecker sum is the tensor product of the two
    marginal null spaces, so its dimension is their product, and `df(lambda)`
    falls to that number and stops. Below it there is nothing left to shrink;
    at it, the shrinkage is approached but never reached. Either way `lambda`
    pins at its maximum and the `df=` on the marginals stops controlling
    anything, which is worth saying out loud — R `mboost` warns at the same
    boundary.
    """
    floor = _marginal_null_dim(left) * _marginal_null_dim(right)
    if floor == 0 or df > floor:
        return
    warnings.warn(
        f"interaction {name}: the marginal df= multiply to {df}, at or below "
        f"the {floor}-dimensional unpenalized part of the tensor penalty. No "
        f"finite lambda reaches a df that low, so the term is fitted at "
        f"maximum shrinkage and df= no longer sets its flexibility. Raise a "
        f"marginal df so their product exceeds {floor}.",
        UserWarning,
        stacklevel=4,
    )


def _tensor_hyper(left: _Marginal, right: _Marginal) -> tuple[float | None, float | None]:
    """The ``(df, lambda)`` of the tensor, from the two marginals'.

    R `mboost`'s ``%X%`` rule. When both margins fix a ``lambda`` there is
    nothing left to calibrate — each penalty block already carries its own
    smoothing level — so the tensor takes ``lambda = 1`` and no ``df``.
    Otherwise the tensor is calibrated through a ``df`` that is the *product* of
    the marginal ``df``s, a margin without one counting as 1. The product is
    what the construction gives: for two smoothers applied along independent
    coordinates the tensor hat matrix is their Kronecker product, whose trace is
    the product of the marginal traces. Two `bbs` margins at the default
    ``df = 4`` therefore give the interaction 16 degrees of freedom.
    """
    lambda_l = left.params.get("lambda_value")
    lambda_r = right.params.get("lambda_value")
    if lambda_l is not None and lambda_r is not None:
        return None, 1.0
    df_l = left.params.get("df")
    df_r = right.params.get("df")
    return (1.0 if df_l is None else df_l) * (1.0 if df_r is None else df_r), None


def _kron_sum_penalty(pen_l: np.ndarray, pen_r: np.ndarray) -> np.ndarray:
    """`K_l (x) I + I (x) K_r`, with the Kronecker products from the Rust core."""
    from mboostrs import _rust

    pl, pr = pen_l.shape[0], pen_r.shape[0]
    left = _rust.kron(np.ascontiguousarray(pen_l, dtype=float), np.eye(pr))
    right = _rust.kron(np.eye(pl), np.ascontiguousarray(pen_r, dtype=float))
    return np.asarray(left) + np.asarray(right)


def build_tensor(interaction: TensorInteraction, data) -> tuple[np.ndarray, np.ndarray]:
    """Design and Kronecker-sum penalty for an interaction on `data`."""
    Xl, Kl = _marginal_design(interaction.left, data)
    Xr, Kr = _marginal_design(interaction.right, data)
    return _tensor(Xl, Xr), _kron_sum_penalty(Kl, Kr)


def build_tensor_design(interaction: TensorInteraction, data, *, unseen="error") -> np.ndarray:
    """Just the design — used at prediction time (penalty is fixed).

    ``unseen="zero"`` lets a ``brandom``/``byfactor`` marginal level the fit
    never saw through, coding it so the interaction contributes nothing for
    that row — the tensor equivalent of ``unseen_levels='zero_effect'``.
    """
    Xl, _ = _marginal_design(interaction.left, data, unseen=unseen)
    Xr, _ = _marginal_design(interaction.right, data, unseen=unseen)
    return _tensor(Xl, Xr)


# --------------------------------------------------------------------------
# formula integration
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# a categorical `by=`
# --------------------------------------------------------------------------


def factor_columns(data) -> set:
    """The columns holding factors rather than numbers.

    Taken before `encode_factors` replaces a factor with its integer codes,
    since after that step nothing in the frame distinguishes a level code from
    a measurement — which is exactly the confusion a `by=` must not make.
    """
    from mboostrs.transforms import _is_categorical

    names = list(data.keys()) if isinstance(data, dict) else list(data.columns)
    return {name for name in names if _is_categorical(data, name)}


def reject_factor_by(formula: str, data, frontend: str) -> None:
    """Refuse a factor ``by=`` in a frontend that cannot expand it.

    A factor `by=` needs one copy of the design per level (see
    :func:`expand_by_factors`), which `gamboost` builds and the linear and LSS
    frontends do not. Left alone the column is encoded to integer codes and
    multiplied in as if the levels were a measurement, which is a wrong answer
    rather than a failure.
    """
    factors = factor_columns(data)
    named = {unquote_name(match.group(1))
             for match in re.finditer(rf"by\s*=\s*({N})", formula)}
    offending = sorted(named & factors)
    if not offending:
        return
    columns = ", ".join(repr(name) for name in offending)
    raise NotImplementedError(
        f"{frontend} cannot fit a factor by= ({columns}): it needs one copy of "
        f"the design per level of the factor. Use gamboost(), which builds "
        f"that, or pass a numeric by=."
    )


def _by_factor_marginal(column: str, data) -> _Marginal:
    raw = _raw_column(data, column)
    levels = tuple(sorted({v for v in raw.tolist()}, key=str))
    if len(levels) < 2:
        raise ValueError(
            f"by={column!r} has a single level {levels[0]!r}; a factor by= "
            "codes the difference from its first level, so two are needed"
        )
    return _Marginal("byfactor", column, {"levels": levels, "df": None,
                                          "lambda_value": 0.0})


def _learner_marginal(learner, data) -> _Marginal:
    """The learner's own design and penalty, as an interaction marginal."""
    from mboostrs.learners import Bbs, Bols

    if isinstance(learner, Bols):
        hyper = {
            "df": learner.df,
            "lambda_value": learner.lambda_value,
            "intercept": bool(learner.intercept),
        }
        if hyper["df"] is None and hyper["lambda_value"] is None:
            hyper["lambda_value"] = 0.0     # R `bols`: unpenalized by default
        return _Marginal("bols", learner.name, hyper)

    if isinstance(learner, Bbs):
        unsupported = [
            name
            for name, unsupported_value in (
                ("center", bool(learner.center)),
                ("constraint", learner.constraint is not None),
                ("deriv", int(learner.deriv) != 0),
            )
            if unsupported_value
        ]
        if unsupported:
            raise NotImplementedError(
                f"bbs({quote_name(learner.name)}, by={quote_name(learner.by)}) with "
                f"{', '.join(unsupported)}: a factor by= builds the per-level "
                "design here, and that construction does not carry "
                f"{' or '.join(unsupported)}. Drop it, or use a numeric by=."
            )
        from mboostrs.transforms import _column

        x = _column(data, learner.name)
        boundary = (
            tuple(float(v) for v in learner.boundary_knots)
            if learner.boundary_knots is not None
            else (float(np.min(x)), float(np.max(x)))
        )
        return _Marginal(
            "bbs",
            learner.name,
            {
                "boundary_knots": boundary,
                "n_knots": int(learner.knots),
                "degree": int(learner.degree),
                "differences": int(learner.differences),
                "df": learner.df,
                "lambda_value": learner.lambda_value,
                # Carried into the Rust TermBasis::Bbs recipe by
                # _raw_marginal_design: a cyclic bbs with a factor by=
                # builds one wrapped basis per level.
                "cyclic": bool(learner.cyclic),
            },
        )

    raise NotImplementedError(
        f"{learner.__class__.__name__}(by=...) with a factor by= is not "
        "supported; bols() and bbs() are. A numeric by= works for every "
        "learner, and bbs(x) %X% brandom(g) fits a related per-group surface."
    )


def expand_by_factors(learners, data, factors=(), start=0):
    """Lower learners carrying a factor ``by=`` into tensor ``buser`` terms.

    R multiplies a learner's design by its ``by`` column. When that column is a
    factor there is no single column to multiply by: R codes it with treatment
    contrasts and stacks one copy of the design per non-reference level,
    penalized block-diagonally (`X_bbs`, `X_ols`). That is the row-wise tensor
    of the indicator columns with the learner's design, penalized by the
    Kronecker sum of a zero penalty and the learner's — which is what
    :func:`build_tensor` already builds and what ``buser`` already fits, so the
    learner lowers to a ``buser`` term over the stacked design and predicts on
    new data through the same recipe an ``%X%`` interaction uses.

    The learner's own ``df`` applies to the whole stacked design, as in R. The
    null space grows with the number of levels — a second-difference penalty
    over `k - 1` blocks leaves `2(k - 1)` directions free — so a `df` under
    that is rejected, again as in R.

    `factors` names the columns that were factors before any integer encoding
    was applied to the frame, since a `by=` reading encoded codes as a
    measurement is the silent error this exists to prevent.

    Returns ``(learners, recipes, display_names)``: the rewritten learner list,
    the recipes to store on the model for prediction, and a mapping from each
    derived name to the term the user wrote. Derived names are ``__by<k>``
    numbered from `start`, so several calls (one per LSS parameter) can share
    one model without colliding.
    """
    from mboostrs.transforms import _has_column, _is_categorical

    def is_factor(column):
        # A learner can name something that is not a data column (an
        # intercept term `1`, a derived `__ix` design).
        return column in factors or (_has_column(data, column)
                                     and _is_categorical(data, column))

    from mboostrs.learners import Bols, Buser

    def is_factor_bols(learner):
        # A bols over a factor lowers to its level design, whatever its by=:
        # none, numeric (times the column), or a factor (one cell per pair).
        return isinstance(learner, Bols) and is_factor(learner.name)

    if not any(
        (getattr(learner, "by", None) is not None and is_factor(learner.by))
        or is_factor_bols(learner)
        for learner in learners
    ):
        return list(learners), [], {}

    out, recipes, display = [], [], {}
    for learner in learners:
        if is_factor_bols(learner):
            lowered, recipe = _lower_factor_bols(learner, data,
                                                 f"__by{start + len(recipes)}",
                                                 is_factor)
            out.append(lowered)
            recipes.append(recipe)
            display[recipe.derived] = recipe.name
            continue
        by = getattr(learner, "by", None)
        if by is None or not is_factor(by):
            out.append(learner)
            continue

        left = _by_factor_marginal(by, data)
        right = _learner_marginal(learner, data)
        derived = f"__by{start + len(recipes)}"
        recipe = TensorInteraction(
            name=f"{right.kind}({quote_name(learner.name)}, by={quote_name(by)})",
            derived=derived, left=left, right=right,
        )
        design, penalty = build_tensor(recipe, data)

        df, lambda_value = right.params.get("df"), right.params.get("lambda_value")
        if df is not None:
            floor = _marginal_null_dim(left) * _marginal_null_dim(right)
            if floor and float(df) < floor:
                raise ValueError(
                    f"{recipe.name}: df={df} is below the {floor}-dimensional "
                    f"unpenalized part of the penalty, which no lambda can "
                    f"shrink — a factor by= stacks one block per non-reference "
                    f"level, and the null space stacks with it. Use "
                    f"df >= {floor}."
                )
            _warn_if_df_is_at_the_floor(recipe.name, float(df), left, right)

        out.append(
            Buser(derived, X=design, K=penalty,
                  df=4.0 if df is None else float(df),
                  lambda_value=None if lambda_value is None else float(lambda_value))
        )
        recipes.append(recipe)
        display[derived] = recipe.name
    return out, recipes, display


def _lower_factor_bols(learner, data, derived, is_factor):
    """``bols(f, ...)`` for a factor `f`, as a ``buser`` term and its recipe.

    A plain ``bols(f)`` is rewritten earlier (`lower_categorical_bols`) to the
    unpenalized level design, which spans the same columns. With
    ``intercept=False``, ``df=``, ``lambda=`` or a ``by=``, the term is built as
    mboost builds it: the treatment-contrast design ``[1, d_2, ..., d_K]``
    (``[d_2, ..., d_K]`` without the intercept, the first level being the zero
    row), multiplied by the ``by=`` margin, with the ridge penalty ``lambda * I``
    over every column, ``lambda`` taken as given or calibrated from ``df``.
    `is_factor` says which columns were factors before encoding.
    """
    from mboostrs.learners import Buser

    # The column holds the factor's integer codes by now, so numeric order is
    # level order and code 0, the first level, is the reference.
    raw = _raw_column(data, learner.name)
    levels = tuple(sorted({v for v in raw.tolist()}))
    intercept = bool(learner.intercept)
    name = (f"bols({quote_name(learner.name)}"
            + ("" if learner.by is None else f", by={quote_name(learner.by)}")
            + ("" if intercept else ", intercept=False")
            + ("" if learner.df is None else f", df={learner.df:g}")
            + ("" if learner.lambda_value is None else f", lambda={learner.lambda_value:g}")
            + ")")
    # The right margin is what the level design is multiplied by, row by row.
    # Without a by= it is a unit column. A numeric by= is the column itself, as
    # mboost's `X * by`. A factor by= is the other factor's full level design
    # (intercept included), so the tensor has one coefficient per pair of levels:
    # the cell design of the two factors.
    if learner.by is None:
        right = _Marginal("unit", learner.name, {"df": None, "lambda_value": None})
    elif is_factor(learner.by):
        by_raw = _raw_column(data, learner.by)
        by_levels = tuple(sorted({v for v in by_raw.tolist()}))
        right = _Marginal("contrasts", learner.by,
                          {"levels": by_levels, "intercept": True, "df": None,
                           "lambda_value": None})
    else:
        right = _fit_marginal("bols", learner.by, "intercept=False", data)
    recipe = TensorInteraction(
        name=name, derived=derived,
        left=_Marginal("contrasts", learner.name,
                       {"levels": levels, "intercept": intercept, "df": None,
                        "lambda_value": None}),
        right=right,
    )
    design, penalty = build_tensor(recipe, data)
    if learner.df is None and learner.lambda_value is None:
        lambda_value = 0.0
    else:
        lambda_value = learner.lambda_value
    lowered = Buser(derived, X=design, K=penalty,
                    df=4.0 if learner.df is None else float(learner.df),
                    lambda_value=None if lambda_value is None else float(lambda_value))
    return lowered, recipe


def _parse_marginal(text: str, data) -> _Marginal:
    transformed = _TRANSFORMED_MARGINAL_RE.match(text)
    if transformed:
        kind, transform, column, args = transformed.groups()
        column = unquote_name(column)
        return _fit_marginal(kind, column, args, data, transform=transform)
    match = _MARGINAL_RE.match(text)
    if not match:
        raise ValueError(
            f"interaction operand {text.strip()!r} must be one of "
            f"bbs(col), bols(col), brandom(col), or bbs/bols of log(col) or sqrt(col)"
        )
    kind, column, args = match.group(1), unquote_name(match.group(2)), match.group(3)
    return _fit_marginal(kind, column, args, data)


def _transformed(values, transform):
    """`values` on the margin's scale. `transform` is ``(kind, lower)``: a
    value below the training minimum `lower` is held at it, as the formula's
    ``log``/``sqrt`` steps hold new values, so a new zero cannot become
    ``-inf``."""
    if not transform:
        return values
    kind, lower = transform
    clipped = np.maximum(np.asarray(values, dtype=float), lower)
    return np.log(clipped) if kind == "log" else np.sqrt(clipped)


def learner_design(term: str, data) -> LearnerDesign:
    """Build a reusable design and penalty for one base-learner term.

    Parameters
    ----------
    term : str
        One ``bbs(...)``, ``bols(...)`` or ``brandom(...)`` term. Its fitted
        basis information is retained for :meth:`LearnerDesign.transform`.
    data : frame or mapping
        Rows used to fit the basis and establish spline boundaries or factor
        levels.

    Returns
    -------
    LearnerDesign
        ``design`` and ``penalty`` can be passed to ``buser``; ``transform``
        constructs the conformable design for new rows.
    """
    return LearnerDesign(term, data)


def expand_interactions(formula: str, data, *, na_action: str = "error"):
    """Rewrite ``a %X% b`` terms into buser references.

    Returns ``(rewritten_formula, env, interactions)`` where ``env`` maps the
    injected design/penalty matrix names, and ``interactions`` is the list of
    :class:`TensorInteraction` recipes to store for prediction. A formula with
    no ``%X%`` returns unchanged with an empty env and list.

    ``na_action="impute"`` fills a numeric margin's missing values with its
    training median, recorded in the recipe so new rows are filled the same
    way. Unlike a main effect, a tensor margin gets no ``<x>_missing``
    indicator. With the default ``"error"``, a missing margin value raises,
    naming the column.
    """
    from mboostrs.formula import _signed_terms

    lhs, sep, rhs = partition_tilde(formula)
    if not sep:
        from mboostrs.formula import no_response_message

        raise ValueError(no_response_message(formula))
    if _OP not in rhs:
        return formula, {}, []

    env: dict = {}
    interactions: list[TensorInteraction] = []
    out_terms: list[str] = []

    for sign, term in _signed_terms(rhs):
        if sign == "-" and _OP in term:
            raise ValueError("removing an interaction term ('- a %X% b') is not supported")
        if _OP not in term:
            out_terms.append(term)
            continue

        parts = term.split(_OP)
        if len(parts) != 2:
            raise ValueError(
                f"interaction term {term.strip()!r} must be exactly 'a %X% b'"
            )
        left = _with_training_fill(_parse_marginal(parts[0], data), data, na_action)
        right = _with_training_fill(_parse_marginal(parts[1], data), data, na_action)

        k = len(interactions)
        derived = f"__ix{k}"
        x_name, k_name = f"__ix{k}_X", f"__ix{k}_K"
        interaction = TensorInteraction(
            name=f"{parts[0].strip()} %X% {parts[1].strip()}",
            derived=derived, left=left, right=right,
        )
        design, penalty = build_tensor(interaction, data)
        env[x_name] = design
        env[k_name] = penalty
        df, lambda_value = _tensor_hyper(left, right)
        if df is not None:
            _warn_if_df_is_at_the_floor(interaction.name, df, left, right)
        hyper = f"df={df}" if df is not None else f"lambda={lambda_value}"
        out_terms.append(f"buser({quote_name(derived)}, X={x_name}, K={k_name}, {hyper})")
        interactions.append(interaction)
        # buser needs the derived column to exist in the data frame; it is a
        # placeholder (the real design comes from X=), so any aligned column
        # works. Reuse the left marginal's column values.
        env[derived] = None

    rewritten = f"{lhs.strip()} ~ " + " + ".join(out_terms)
    return rewritten, env, interactions
