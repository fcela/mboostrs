"""GAMBoost model class and gamboost() function."""

from __future__ import annotations

import numbers
import os
import re
import sys
import warnings
from dataclasses import dataclass

import numpy as np

from mboostrs._comparison_identity import fit_identity
from mboostrs._control import as_control, step_kwargs
from mboostrs._model_api import ModelAPIMixin, capture_fit_warnings
from mboostrs._names import partition_tilde, quote_name
from mboostrs._results import AicResult, provenance_from_model
from mboostrs._validate import (
    _model_extrapolation,
    _model_unseen_levels,
)
from mboostrs._weights import normalize_weights
from mboostrs._workflow import call_fit, fit_checkpoint
from mboostrs.cvrisk import (
    CvRiskResult,
    _as_fold_matrix,
    _fold_design,
    data_fingerprint,
    fold_fingerprint,
    fold_label_for,
    held_out_fraction,
    warn_if_at_boundary,
    warn_if_one_fold_dominates,
)
from mboostrs.families import (
    _TYPE_DEFAULT,
    CoxPH,
    Gaussian,
    GaussReg,
    Multinomial,
    PropOdds,
    _family_params,
    _family_rust_name,
    _inverse_link,
    maybe_warn_link_scale,
    prediction_scale_text,
)
from mboostrs.families import Loglog as LoglogFam
from mboostrs.families import Lognormal as LognormalFam
from mboostrs.families import (
    Weibull as WeibullFam,
)
from mboostrs.folds import FoldScheme
from mboostrs.formula import _intercept_only, expand_formula, parse_gamboost_formula
from mboostrs.glmboost import (
    _binomial_cbind_response,
    _breslow_baseline_of,
    _confint_groups,
    _drop_aligned,
    _extract_column,
    _extract_response,
    _frame_length,
    _is_binomial_cbind,
    _is_vector_offset,
    _merge_part,
    _nonfinite_message,
    _part_of,
    _prepare_offset,
    _replay_interactions,
    _replay_transforms,
    _resolve_prediction_offset,
    _tensor_margin_clamp,
    _tensor_margin_rows,
    np_all_finite,
)
from mboostrs.interactions import factor_columns as _factor_columns_of
from mboostrs.learners import (
    Bbs,
    Bconstrained,
    Bkernel,
    Bmono,
    Bmrf,
    Bns,
    Bols,
    Brad,
    Brandom,
    Bspatial,
    Btree,
    Buser,
    Hinge,
)
from mboostrs.report import ExecutionReport, path_rows_output, standard_path_rows
from mboostrs.transforms import (
    TransformRecipe,
    apply_recipe,
    encode_factors,
    extract_transforms,
    factor_term_labels,
    fit_recipe,
    impute_missing,
    lower_categorical_bols,
)
from mboostrs.transforms import display_names as _display_names

# Kernel functions for Bkernel predict with newdata (Python-side computation)
_KERNEL_FUNCS = {}
_BKERNEL_WARN_CELLS = int(os.environ.get("MBOOSTRS_BKERNEL_WARN_CELLS", "10000000"))


def _top_level_parts(name: str) -> list:
    """The comma-separated variables in a learner name.

    A learner over several columns is named for all of them — `bspatial` is
    `"x1,x2"` — and `varimp` reports that list sorted, as R does. Commas inside
    a term the user wrote, as in the `"bbs(x1, by=g)"` a factor `by=` reports
    under, are not separators, so only depth-zero commas split.
    """
    parts, current, depth = [], [], 0
    for character in name:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        if character == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    parts.append("".join(current))
    return [part.strip() for part in parts]


def _effect_coordinate_name(spec) -> str:
    """The column a learner's partial effect is reported against.

    A learner reading several columns has no single coordinate, so the first one
    stands in: `bspatial(x1, x2)` reports against `x1`, and a `btree` or `brad`
    over several columns against the first column it may split on. Without this
    such a learner would look up its own joined name — `btree(x1, x3)` is named
    `"x1,x3"` — as if it were a column, and fail.
    """
    if isinstance(spec, Bspatial):
        return spec.name1
    names = getattr(spec, "names", None)
    if names:
        return names[0]
    return spec.name


def _warn_bkernel_scale(name: str, n: int) -> None:
    cells = int(n) * int(n)
    if cells >= _BKERNEL_WARN_CELLS:
        approx_mib = cells * 8.0 / (1024.0 * 1024.0)
        warnings.warn(
            f"Bkernel learner {name!r} builds a dense {n} x {n} kernel matrix "
            f"(about {approx_mib:.1f} MiB before solver overhead). Consider a "
            "smaller sample, a different base learner, or an approximate kernel "
            "backend when available.",
            ResourceWarning,
            stacklevel=3,
        )


def _register_kernel(name):
    def decorator(fn):
        _KERNEL_FUNCS[name] = fn
        return fn

    return decorator


@_register_kernel("gaussian")
def _kernel_gaussian(d: float, bandwidth: float) -> float:
    h = bandwidth
    return np.exp(-0.5 * (d / h) ** 2) if h > 0 else 0.0


@_register_kernel("epanechnikov")
def _kernel_epanechnikov(d: float, bandwidth: float) -> float:
    h = bandwidth
    if h <= 0 or d >= h:
        return 0.0
    return 0.75 * (1.0 - (d / h) ** 2)


@_register_kernel("uniform")
def _kernel_uniform(d: float, bandwidth: float) -> float:
    h = bandwidth
    return 1.0 if h > 0 and d < h else 0.0


def _multinomial_response(model) -> np.ndarray:
    """Rebuild the flattened Multinomial indicator the fit was run on.

    The fit encodes the raw categorical labels as a one-hot indicator over the
    sorted classes, dropping the last (reference) column, and flattens it to
    ``n_obs * (K-1)``. ``cvrisk`` must hand the engine the same vector, so this
    reproduces that encoding from the model's retained data, response column,
    and already-settled ``n_classes``.
    """
    y_raw = np.asarray(model._data[model._response_name])
    classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    K = model._family.n_classes
    y_codes = np.array([class_to_idx[yi] for yi in y_raw], dtype=np.int64)
    indicator = np.eye(K, dtype=np.float64)[y_codes][:, :-1]
    return np.ascontiguousarray(indicator.ravel(), dtype=np.float64)


try:
    from mboostrs._rust import (
        boost_control as _boost_control,
    )
    from mboostrs._rust import (
        cv_weights as _cv_weights,
    )
    from mboostrs._rust import (
        cvrisk_gamboost_specs as _cvrisk_gamboost_specs,
    )
    from mboostrs._rust import (
        cvrisk_gamboost_specs_extending as _cvrisk_gamboost_specs_extending,
    )
    from mboostrs._rust import (
        fit_gamboost_continue_specs as _fit_gamboost_continue_specs,
    )
    from mboostrs._rust import (
        fit_gamboost_specs as _fit_gamboost_specs,
    )
    from mboostrs._rust import (
        fit_glmboost as _fit_glmboost,
    )
    from mboostrs._rust import (
        prepare_learner_store as _prepare_learner_store,
    )

    _HAS_RUST_ENGINE = True
except ImportError:
    _HAS_RUST_ENGINE = False
    _boost_control = None
    _fit_gamboost_specs = None
    _fit_gamboost_continue_specs = None
    _fit_glmboost = None
    _prepare_learner_store = None
    _cv_weights = None
    _cvrisk_gamboost_specs = None
    _cvrisk_gamboost_specs_extending = None


_RUST_LEARNER_ORDER = (
    "scalar",
    "bbs",
    "bmono",
    "btree",
    "bmrf",
    "bspatial",
    "bkernel",
    "brandom",
    "buser",
)


@dataclass(frozen=True)
class _LearnerBlock:
    kind: str
    start: int
    formula_indices: tuple[int, ...]

    @property
    def stop(self) -> int:
        return self.start + len(self.formula_indices)

    def contains(self, rust_idx: int) -> bool:
        return self.start <= rust_idx < self.stop

    def local_index(self, rust_idx: int) -> int:
        if not self.contains(rust_idx):
            raise IndexError(f"Rust learner index {rust_idx} is outside {self.kind!r} block")
        return rust_idx - self.start


@dataclass(frozen=True)
class LearnerLayout:
    """Rust/formula learner ordering for GAMBoost models.

    Rust stores GAM learners in engine order, not formula order. Keeping that
    mapping in one immutable object prevents selected-index, coefficient, and
    prediction-filtering paths from each rebuilding the same offsets by hand.
    """

    blocks: tuple[_LearnerBlock, ...]
    rust_idx_to_formula_idx: tuple[int, ...]
    formula_idx_to_rust_idx: dict[int, int]

    @classmethod
    def from_blocks(cls, block_formula_indices: dict[str, list[int]]) -> LearnerLayout:
        blocks = []
        rust_to_formula: list[int] = []
        start = 0
        for kind in _RUST_LEARNER_ORDER:
            formula_indices = tuple(block_formula_indices.get(kind, ()))
            blocks.append(_LearnerBlock(kind, start, formula_indices))
            rust_to_formula.extend(formula_indices)
            start += len(formula_indices)
        formula_to_rust = {formula_idx: rust_idx for rust_idx, formula_idx in enumerate(rust_to_formula)}
        if len(formula_to_rust) != len(rust_to_formula):
            raise ValueError("duplicate formula learner indices in GAM learner layout")
        return cls(tuple(blocks), tuple(rust_to_formula), formula_to_rust)

    @classmethod
    def from_specs(cls, learners_spec: list) -> LearnerLayout:
        block_formula_indices = {kind: [] for kind in _RUST_LEARNER_ORDER}
        for formula_idx, spec in enumerate(learners_spec):
            block_formula_indices[_engine_kind_for_spec(spec)].append(formula_idx)
        return cls.from_blocks(block_formula_indices)

    @classmethod
    def from_mapping(cls, rust_idx_to_formula_idx: list[int] | tuple[int, ...]) -> LearnerLayout:
        rust_to_formula = tuple(int(i) for i in rust_idx_to_formula_idx)
        block = _LearnerBlock("unknown", 0, rust_to_formula)
        formula_to_rust = {formula_idx: rust_idx for rust_idx, formula_idx in enumerate(rust_to_formula)}
        return cls((block,), rust_to_formula, formula_to_rust)

    def rust_to_formula(self, rust_idx: int) -> int:
        return self.rust_idx_to_formula_idx[rust_idx]

    def formula_to_rust(self, formula_idx: int) -> int:
        return self.formula_idx_to_rust_idx[formula_idx]

    def block_for_rust_idx(self, rust_idx: int) -> _LearnerBlock:
        for block in self.blocks:
            if block.contains(rust_idx):
                return block
        raise IndexError(f"Rust learner index {rust_idx} is outside layout of {len(self)} learners")

    def rust_indices_for_kind(self, kind: str) -> range:
        for block in self.blocks:
            if block.kind == kind:
                return range(block.start, block.stop)
        raise KeyError(f"Unknown learner block kind {kind!r}")

    def split_selected_for_external_kinds(
        self,
        selected: list[int],
        coefs: list[list[float]],
        external_kinds: set[str],
    ) -> tuple[list[int], list[list[float]], dict[str, list[tuple[int, list[float]]]]]:
        filtered_selected: list[int] = []
        filtered_coefs: list[list[float]] = []
        external_steps = {kind: [] for kind in external_kinds}
        for rust_idx, coef in zip(selected, coefs, strict=False):
            block = self.block_for_rust_idx(rust_idx)
            if block.kind in external_kinds:
                external_steps[block.kind].append((block.local_index(rust_idx), coef))
            else:
                filtered_selected.append(rust_idx)
                filtered_coefs.append(coef)
        return filtered_selected, filtered_coefs, external_steps

    def __len__(self) -> int:
        return len(self.rust_idx_to_formula_idx)


def _decode_btree_nodes(vec):
    """Decode one boosting step's btree coefficient vector into node records.

    The layout is `FittedTree::to_coef_vec`'s: a node count, then per node a
    tag — ``0`` splits on a number (``var, cutpoint, left, right``), ``1`` is
    a leaf (``mean``), ``2`` splits a column into level sets
    (``var, m, level_1..level_m, left, right``). Two trees do not add, so the
    records are per step; nothing about them may be summed.
    """
    values = np.asarray(vec, dtype=np.float64).ravel()
    if not values.size:
        return []
    index = 0
    count = int(values[index])
    index += 1
    nodes = []
    for _ in range(count):
        tag = values[index]
        index += 1
        if tag == 0.0:
            nodes.append({"kind": "split", "variable": int(values[index]),
                          "cutpoint": float(values[index + 1]),
                          "left": int(values[index + 2]),
                          "right": int(values[index + 3])})
            index += 4
        elif tag == 1.0:
            nodes.append({"kind": "leaf", "mean": float(values[index])})
            index += 1
        elif tag == 2.0:
            variable = int(values[index])
            level_count = int(values[index + 1])
            levels = [int(v) for v in values[index + 2:index + 2 + level_count]]
            nodes.append({"kind": "level_split", "variable": variable,
                          "left_levels": levels,
                          "left": int(values[index + 2 + level_count]),
                          "right": int(values[index + 3 + level_count])})
            index += 4 + level_count
        else:
            raise ValueError(f"unknown btree node tag {tag!r}")
    return nodes


def _resolve_btree_learner(model, term=None):
    """The formula index, disambiguated name and spec of one btree term.

    ``term`` is a base-learner name (as :meth:`base_learner_names` reports) or
    a formula index; ``None`` picks the only btree when the model has exactly
    one. A name that is not a btree, a model with no btree, and an ambiguous
    ``term`` each raise a `ValueError` naming the tree terms the model does
    have.
    """
    from mboostrs.learners import Btree

    if not (hasattr(model, "_btree_indices") and hasattr(model, "btree_splits")):
        kind = type(model).__name__
        if "LSS" in kind:
            raise ValueError(
                "btree partitions are read from the gamboost() "
                "btree_splits(), which an LSS model does not expose. LSS keeps "
                "one btree term per distribution parameter, so its partition "
                "cannot be read yet; refit with gamboost() for a partition "
                "plot."
            )
        raise ValueError(
            f"{kind} does not expose btree_splits(), so it has no btree "
            "partition to read; fit the tree with gamboost(), e.g. "
            "`gamboost('y ~ btree(x, z, maxdepth=2)', ...)`"
        )
    tree_indices = sorted(model._btree_indices())
    if not tree_indices:
        raise ValueError(
            "this model has no btree base-learner to read a partition from; "
            "fit a term such as `btree(x, z, maxdepth=2)`"
        )
    names = model.base_learner_names()
    tree_names = [names[i] for i in tree_indices]
    if term is None:
        if len(tree_indices) != 1:
            raise ValueError(
                f"this model has {len(tree_indices)} btree terms {tree_names}; "
                "pass term=<one of them>"
            )
        index = tree_indices[0]
    else:
        try:
            found = model._which_indices(term)
        except (TypeError, ValueError):
            raise ValueError(
                f"unknown base-learner {term!r}; this model's btree terms are "
                f"{tree_names}"
            ) from None
        tree_set = set(tree_indices)
        matched = [i for i in found if i in tree_set]
        if not matched:
            raise ValueError(
                f"{term!r} is not a btree base-learner; this model's btree "
                f"terms are {tree_names}"
            )
        if len(matched) > 1:
            raise ValueError(
                f"{term!r} names more than one learner; pick one of "
                f"{[names[i] for i in matched]}"
            )
        index = matched[0]
    spec = model._learners_spec[index]
    if not isinstance(spec, Btree):
        raise ValueError(
            f"{names[index]!r} is not a btree base-learner"
        )
    return index, names[index], spec


def _btree_term_effect(model, index, varying):
    """One btree term's fitted contribution at the rows in ``varying``.

    ``varying`` maps a column the term reads to a 1-D array; every other
    column the model's learners read is held at its training median so the
    replay carries the same row count for each learner. This is the term's own
    partial prediction, not the full model's, and it bypasses the predict-time
    transform recipe because the columns are already on the fitted scale.

    Repeated boosting steps of the term are summed, which is what the fitted
    contribution is; a single step's tree is not added to another's encoding.
    """
    from mboostrs.learners import Buser

    varying = {
        name: np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
        for name, values in varying.items()
    }
    if not varying:
        raise ValueError("a btree contribution needs at least one varying column")
    n = len(next(iter(varying.values())))
    source = {}
    for column in model._learner_column_names():
        if column in varying:
            source[column] = varying[column]
        else:
            training = np.asarray(model._data_array_for(column), dtype=np.float64)
            source[column] = np.full(n, float(np.median(training)))
    buser_newdata = {}
    for spec in model._learners_spec or ():
        if isinstance(spec, Buser) and getattr(spec, "_radial_basis", None) is None:
            width = int(np.asarray(spec.X).shape[1])
            buser_newdata[spec.name] = np.zeros((n, width), dtype=np.float64)
    store = model._replay_store()
    if store is None:
        raise ValueError(
            "cannot evaluate the btree term: this model carries no learner "
            "store and none could be rebuilt from its specifications"
        )
    step_indices = [i for i, sel in enumerate(model._selected) if sel == index]
    if not step_indices:
        return np.zeros(n, dtype=np.float64)
    rust_index = model._formula_to_rust(index)
    selected = [rust_index] * len(step_indices)
    coefs = [np.ascontiguousarray(model._coefficients[i], dtype=np.float64)
             for i in step_indices]
    x_data, by_data = model._replay_slices(source, buser_newdata or None)
    effect = store.predict_newdata_grouped(
        0.0, selected, coefs, x_data, None, by_data, unseen_levels="error",
    )
    return np.asarray(effect, dtype=np.float64).ravel()


def _btree_partition_cutpoints(model, name):
    """The numeric cutpoints :meth:`GAMBoostModel.btree_splits` recorded.

    Returns ``{column: sorted unique cutpoints}`` for the term keyed by
    ``name``; an empty mapping when the term has no numeric split.
    """
    info = model.btree_splits().get(name) or {}
    cuts = {}
    for row in info.get("splits", ()):
        column = row.get("variable")
        if column is not None:
            cuts.setdefault(column, set()).add(float(row["cutpoint"]))
    return {column: sorted(points) for column, points in cuts.items()}


def _btree_tree_leaves(nodes, columns, bounds):
    """The axis-aligned leaf boxes of one decoded tree.

    Walks the pre-order node list from the root (index ``0``), narrowing each
    column's ``[low, high]`` along every ``x <= cutpoint`` split (``left`` when
    the value is ``<=`` the cutpoint, matching the fitted tree's own descent).
    Returns a list of ``(low, high)`` dictionaries keyed by column. A
    level-set split has no numeric bound, so the caller rejects those trees.
    """
    leaves = []

    def walk(node_idx, low, high):
        node = nodes[node_idx]
        kind = node["kind"]
        if kind == "leaf":
            leaves.append((dict(low), dict(high)))
            return
        if kind == "level_split":
            raise ValueError("a level-set split has no numeric partition bound")
        column = columns[node["variable"]]
        cut = float(node["cutpoint"])
        left_high = dict(high)
        left_high[column] = min(left_high[column], cut)
        walk(node["left"], low, left_high)
        right_low = dict(low)
        right_low[column] = max(right_low[column], cut)
        walk(node["right"], right_low, high)

    start_low = {column: bounds[column][0] for column in columns}
    start_high = {column: bounds[column][1] for column in columns}
    walk(0, start_low, start_high)
    return leaves


def _btree_partition_boxes(model, name, columns, bounds):
    """The regions of a btree term that used more than one boosting step.

    Each step fits its own tree, so the summed term is piecewise constant on
    the common refinement of the trees' cutpoints: the grid of the per-column
    union of cuts recorded by :meth:`GAMBoostModel.btree_splits`. A one-tree
    term is handled by :func:`_btree_tree_leaves`, which keeps the tree's
    actual leaves rather than the finer grid.
    """
    from itertools import product

    cutpoints = _btree_partition_cutpoints(model, name)
    bands = {}
    for column in columns:
        low, high = bounds[column]
        interior = [point for point in cutpoints.get(column, ())
                    if low < point < high]
        edges = [low, *interior, high]
        bands[column] = list(zip(edges[:-1], edges[1:], strict=True))
    cells = 1
    for column in columns:
        cells *= len(bands[column])
    if cells > 100_000:
        raise ValueError(
            f"btree term {name!r} was selected in too many boosting steps for "
            f"its cutpoint grid to be listed ({cells} regions); refit with "
            "fewer steps, or read the cuts from btree_splits()"
        )
    boxes = []
    for cell in product(*(bands[column] for column in columns)):
        low = {column: cell[k][0] for k, column in enumerate(columns)}
        high = {column: cell[k][1] for k, column in enumerate(columns)}
        boxes.append((low, high))
    return boxes


def btree_partition_table(model, term=None):
    """The fitted partition of a btree term, one row per region.

    Each row bounds a region of the term's own fitted partition with
    ``<column>_lower`` and ``<column>_upper`` and gives the term's fitted
    ``value`` at the region's centre: a one-column tree lists its intervals, a
    two-column tree its leaves (up to four for ``maxdepth=2``), and so on. A
    one-tree term is enumerated by walking that tree, so the regions are its
    actual leaves rather than the coarser grid of its cutpoints. A term that
    used several boosting steps is a sum of trees whose common refinement the
    cutpoints of :meth:`GAMBoostModel.btree_splits` describe; the regions are
    then that grid.

    Parameters
    ----------
    model : GLMBoostModel or GAMBoostModel
        A fit whose ``btree_splits()`` records a tree.
    term : str or int, optional
        Which btree term to read. The only btree term is picked when there is
        exactly one; a model with several needs the name or index.

    Returns
    -------
    polars.DataFrame
        One row per region with per-column bounds and the fitted value.
    """
    import polars as pl

    from mboostrs.learners import Btree

    index, name, spec = _resolve_btree_learner(model, term)
    if not isinstance(spec, Btree):  # pragma: no cover - guarded by resolver
        raise ValueError(f"{name!r} is not a btree base-learner")
    columns = list(spec.names)
    info = model.btree_splits().get(name) or {}
    level_variables = info.get("level_split_variables") or []
    if level_variables:
        raise ValueError(
            f"btree term {name!r} splits the categorical column(s) "
            f"{level_variables} into level sets; a level-set region has no "
            "numeric lower/upper bound, so no partition table is built"
        )
    step_indices = [i for i, sel in enumerate(model._selected) if sel == index]
    if not step_indices:
        raise ValueError(
            f"btree term {name!r} was never selected by the fit, so it has no "
            "fitted partition to read"
        )
    bounds = {}
    for column in columns:
        training = np.asarray(model._data_array_for(column), dtype=np.float64)
        bounds[column] = (float(np.min(training)), float(np.max(training)))
    if len(step_indices) == 1:
        nodes = _decode_btree_nodes(model._coefficients[step_indices[0]])
        boxes = _btree_tree_leaves(nodes, columns, bounds)
    else:
        boxes = _btree_partition_boxes(model, name, columns, bounds)

    rows = []
    for low, high in boxes:
        varying = {}
        row = {}
        for column in columns:
            lo, hi = low[column], high[column]
            centre = lo if hi <= lo else 0.5 * (lo + hi)
            varying[column] = np.array([centre], dtype=np.float64)
            row[f"{column}_lower"] = lo
            row[f"{column}_upper"] = hi
        row["value"] = float(_btree_term_effect(model, index, varying)[0])
        rows.append(row)
    return pl.DataFrame(rows)


def _bols_form_for(learner_specs, *, vector_offset=False, kronecker=False,
                   family=None) -> str:
    """`"block"` when any `bols` in the menu needs it, `"centered"` otherwise.

    The centered single-column form takes its constant from the model offset;
    the block form `[1, x]` carries its own. Both are correct alone, and
    mixing them represents the constant twice, so the two compete along the
    path. A penalized or `by=` term forces the block form for itself, and this
    propagates that choice to the rest of the menu.

    The centered form is also wrong when another learner cannot keep the
    residual mean at zero. Every basis that contains the constant (B-splines,
    trees, level indicators) refits the mean each time it is selected; a
    `by = z` learner's columns are scaled by z, so its span lacks the constant
    and the residual mean drifts. mboost's `bols(x)` re-absorbs that drift
    through its own intercept column; a centered column cannot. Measured on
    `bbs(x, by=z) + bols(w)`: 0.015 from mboost with the centered form,
    2.7e-14 with the block form.

    `kronecker=True` (a Multinomial fit) skips the penalized-`brandom` and
    `bkernel` rule: the Kronecker lift rebuilds a centered `bols` as the
    `[1, x]` design itself, so its centered form is already the block form.
    """
    from mboostrs.learners import Bkernel, Bols, Brandom, Buser

    if vector_offset:
        return "block"
    # The centered form relies on the residual mean staying at zero, which a
    # squared-error loss keeps (the offset is the mean, and every centered
    # update has mean zero). Any other loss moves the mean of its negative
    # gradient as the fit changes, and mboost's bols(x) re-fits it through its
    # own intercept column: bbs(x) + bols(z) sat 6.4e-4 from mboost for
    # Binomial, 6.8e-4 for Poisson and 0.053 for Laplace with the centered
    # form, 1e-12 with the block form.
    if (family is not None and not kronecker
            and not isinstance(family, (Gaussian, GaussReg))):
        return "block"

    for spec in learner_specs or []:
        if getattr(spec, "by", None) is not None:
            return "block"
        # A user design need not contain the constant, and a penalty may
        # shrink it; the block form is mboost's own and right either way
        # (0.22 from mboost for `buser(cbind(x, x^2)) + bols(w)` centered).
        # This covers the `%X%`, factor `by=`, factor `bols` and `brad` terms,
        # which are lowered to `Buser`.
        if isinstance(spec, Buser):
            return "block"
        # A ridge on level indicators (or a kernel ridge) shrinks the constant
        # direction too, so the residual mean drifts as it is selected:
        # `brandom(g) + bols(w)` sat 7e-4 from mboost with the centered form,
        # 3e-16 with the block form. An unpenalized brandom (what a plain
        # `bols(factor)` lowers to) keeps the mean.
        if not kronecker and (
                isinstance(spec, Bkernel)
                or (isinstance(spec, Brandom) and spec.lambda_value != 0.0)):
            return "block"
        if not isinstance(spec, Bols):
            continue
        # A penalized bols shrinks its own intercept; a bols without one is
        # an uncentered column through the origin, which moves the residual
        # mean like a `by=` learner (0.013 from mboost beside a centered bols).
        if (spec.df is not None or spec.lambda_value is not None
                or not spec.intercept):
            return "block"
    return "centered"


def _engine_kind_for_spec(spec) -> str:
    if isinstance(spec, Bols):
        return "scalar"
    if isinstance(spec, Bns):
        return "bbs"
    if isinstance(spec, Bkernel):
        return "bkernel"
    if isinstance(spec, Buser):
        return "buser"
    if isinstance(spec, Bbs):
        return "bmono" if spec.constraint is not None else "bbs"
    if isinstance(spec, (Bmono, Bconstrained)):
        return "bmono"
    if isinstance(spec, Btree):
        return "btree"
    if isinstance(spec, Hinge):
        return "bhinge"
    if isinstance(spec, Bmrf):
        return "bmrf"
    if isinstance(spec, Bspatial):
        return "bspatial"
    if isinstance(spec, Brandom):
        return "brandom"
    raise TypeError(f"Unsupported GAM learner spec: {spec!r}")


def _make_indicator(col, n_levels):
    """Build an (n × L) indicator matrix from integer-coded column.

    Auto-detects 0-based vs 1-based coding: if min(col) == 0, treats codes
    as 0-based (Python native); if min(col) >= 1, treats as 1-based (R style).
    """
    col_min = int(col.min())
    offset = 0 if col_min == 0 else 1  # subtract 1 if 1-based
    n = len(col)
    ind = np.zeros((n, n_levels), dtype=np.float64)
    for i in range(n):
        lev = int(col[i]) - offset
        if lev < 0 or lev >= n_levels:
            raise ValueError(
                f"Level code {col[i]} (0-based: {lev}) out of range [0, {n_levels - 1}] at observation {i}"
            )
        ind[i, lev] = 1.0
    return ind


def uses_spec_cv_boundary(family) -> bool:
    """Whether cross-validation describes its learners as compact specs.

    Every family does. Multinomial joined the spec boundary once the Kronecker
    lift was taught to the spec CV path (`cvrisk_gamboost_specs` now takes
    `multinomial_n_classes`), so the 90-argument array CV boundary is dead.
    Kept as a single named predicate so the reader sees where the exception
    *used* to be taken.
    """
    return True


_UNSEEN_LEVEL_POLICIES = ("error", "zero_effect", "average")


def _grid_frame_columns(data):
    """Copies of a frame's columns as arrays, so one can be replaced per grid value."""
    from mboostrs._frame import column_names

    columns = {}
    for name in column_names(data):
        values = data[name]
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        columns[name] = np.array(values, copy=True)
    return columns


def _grid_label(variable, value):
    """The output column name for one grid value, such as ``x=0.5``."""
    if isinstance(value, (float, np.floating)):
        return f"{variable}={float(value):g}"
    return f"{variable}={value}"


def _check_unseen_levels(value):
    """The ``unseen_levels=`` policy, refused unless the replay knows it.

    `'marginal'` is a distribution policy, not a point policy: it spreads the
    predictive distribution of an unseen group over an assumed group-effect
    distribution, so it is accepted by the interval/scoring surfaces and
    refused by `predict`/`params`, which return points.
    """
    if value not in (*_UNSEEN_LEVEL_POLICIES, "marginal"):
        raise ValueError(
            f"unseen_levels must be 'error', 'zero_effect', 'average' or "
            f"'marginal' (the distribution surfaces only), got {value!r}"
        )
    return value


def _disambiguated_names(names, specs):
    """Learner labels that are unique wherever the learners differ.

    A learner is stored under the column it reads, so ``bols(x) + bbs(x)``
    both come back as ``"x"`` and so do ``bbs(hour) + bbs(hour, by=z)``. A
    repeated name gets the learner type appended (``"x (bols)"``); where the
    type is shared too, the ``by=`` variable is added (``"hour (bbs, by=z)"``),
    and any learners still tied are numbered in formula order. Every
    accessor keyed by these labels -- ``partial_effects``, ``coef``,
    ``varimp``, ``extract`` and the summary -- then keeps one entry per
    learner instead of letting the later one overwrite the earlier.
    """
    def kind(index):
        return type(specs[index]).__name__.lower()

    labels = []
    for index, name in enumerate(names):
        if names.count(name) == 1:
            labels.append(name)
            continue
        same_kind = [j for j, other in enumerate(names)
                     if other == name and kind(j) == kind(index)]
        by = getattr(specs[index], "by", None)
        if len(same_kind) > 1 and by is not None:
            labels.append(f"{name} ({kind(index)}, by={quote_name(by)})")
        else:
            labels.append(f"{name} ({kind(index)})")
    first = list(labels)
    for index, label in enumerate(first):
        tied = [j for j, other in enumerate(first) if other == label]
        if len(tied) > 1:
            labels[index] = f"{label} [{tied.index(index) + 1}]"
    return labels


class GAMBoostModel(ModelAPIMixin):
    """Fitted generalized additive boosting model."""

    def __init__(
        self,
        formula: str,
        family,
        control: tuple,
        response_name: str,
        learner_names: list[str],
        learner_types: list[str],
        fitted: np.ndarray,
        risk: np.ndarray,
        selected: list[int],
        coefficients: list[np.ndarray],
        offset: float,
        nu: float,
        data=None,
        weights: np.ndarray | None = None,
        learners_spec: list | None = None,
        rust_idx_to_formula_idx: list[int] | None = None,
        formula_idx_to_rust_idx: dict | None = None,
        buser_centering_means: dict | None = None,
        family_sigma: list[float] | None = None,
        family_sigma_path: list[list[float]] | None = None,
        learner_store=None,  # PyO3 LearnerStore object
        execution_report: dict | ExecutionReport | None = None,
        residual: np.ndarray | None = None,
    ):

        self._formula = formula
        self._family = family
        self._control = control
        self._response_name = response_name
        self._learner_names = learner_names
        self._learner_types = learner_types
        self._fitted = fitted
        self._risk = risk
        self._selected = selected
        self._coefficients = coefficients
        self._offset = offset
        self._nu = nu
        self._data = data
        self._weights = weights
        self._learners_spec = learners_spec
        self._residual = None if residual is None else np.asarray(residual, dtype=np.float64)
        if learners_spec is not None and rust_idx_to_formula_idx is not None:
            # Two fit boundaries produce two orderings: the spec boundary takes
            # learners in formula order, the Kronecker (Multinomial) array
            # boundary groups them by kind. The mapping the fit returned is
            # authoritative for which one this model has; all it must satisfy
            # is that every learner appears exactly once.
            self._learner_layout = LearnerLayout.from_mapping(rust_idx_to_formula_idx)
            if sorted(self._learner_layout.rust_idx_to_formula_idx) != list(range(len(learners_spec))):
                raise ValueError("stored GAM learner layout does not match learner specs")
            self._rust_idx_to_formula_idx = list(self._learner_layout.rust_idx_to_formula_idx)
            self._formula_idx_to_rust_idx = dict(self._learner_layout.formula_idx_to_rust_idx)
        elif learners_spec is not None:
            self._learner_layout = LearnerLayout.from_specs(learners_spec)
            self._rust_idx_to_formula_idx = list(self._learner_layout.rust_idx_to_formula_idx)
            self._formula_idx_to_rust_idx = dict(self._learner_layout.formula_idx_to_rust_idx)
        elif rust_idx_to_formula_idx is not None:
            self._learner_layout = LearnerLayout.from_mapping(rust_idx_to_formula_idx)
            self._rust_idx_to_formula_idx = list(self._learner_layout.rust_idx_to_formula_idx)
            self._formula_idx_to_rust_idx = dict(self._learner_layout.formula_idx_to_rust_idx)
        else:
            self._learner_layout = None
            self._rust_idx_to_formula_idx = None
            self._formula_idx_to_rust_idx = formula_idx_to_rust_idx
        self._replay_columns = None
        self._buser_centering_means = buser_centering_means
        self._family_sigma = family_sigma
        self._family_sigma_path = list(family_sigma_path) if family_sigma_path is not None else []
        self._learner_store = learner_store
        self._execution_report = (
            execution_report
            if isinstance(execution_report, ExecutionReport)
            else ExecutionReport.from_mapping(execution_report)
            if execution_report is not None
            else None
        )


    def path_frame(self, output="auto"):
        """Return one structured row per boosting iteration."""
        rows = standard_path_rows(
            selected=self._selected,
            coefficients=self._coefficients,
            risk=self._risk,
            nu=self._nu,
            learner_names=self._learner_names,
            nuisance_path=self._family_sigma_path,
        )
        return path_rows_output(rows, output)

    def _coefs_rust(self):
        """Rust-facing contiguous-array view of the per-step coefficients, cached.

        The boundary accepts NumPy buffers directly, so the cached value is one
        contiguous `float64` array per step rather than a list-of-lists that
        pyo3 would unbox element by element on every predict/risk call. The
        coefficient list is immutable after fit (truncation builds a new model),
        so the cache is keyed on the list object's identity.
        """
        cache = self.__dict__.get("_coefs_rust_cache")
        if cache is None or cache[0] is not self._coefficients:
            cache = (self._coefficients, [
                np.ascontiguousarray(c, dtype=np.float64) for c in self._coefficients
            ])
            self.__dict__["_coefs_rust_cache"] = cache
        return cache[1]

    def _step_coefs(self) -> list[list[float]]:
        """The per-step coefficient vectors as plain lists, for the learner
        store's hat-matrix calls: a `btree` step's hat matrix is built from
        the partition encoded in its coefficients, so the trace and diagonal
        calls receive the fitted path alongside the selection."""
        return [np.asarray(c, dtype=np.float64).tolist() for c in self._coefficients]

    def _selected_rust(self) -> list[int]:
        if self._learner_layout is not None:
            return [self._learner_layout.formula_to_rust(s) for s in self._selected]
        return [self._formula_idx_to_rust_idx[s] for s in self._selected]

    def _formula_to_rust(self, formula_idx: int) -> int:
        if self._learner_layout is not None:
            return self._learner_layout.formula_to_rust(formula_idx)
        return self._formula_idx_to_rust_idx[formula_idx]

    def _rust_to_formula(self, rust_idx: int) -> int:
        if self._learner_layout is not None:
            return self._learner_layout.rust_to_formula(rust_idx)
        return self._rust_idx_to_formula_idx[rust_idx]




    def penalty_drift(self, output="auto"):
        """Each penalized learner's degrees of freedom along the fitted path.

        A learner's ``df=`` (or ``lambda=``) sets the flexibility of one
        boosting step. Selected ``m`` times at step ``nu``, its accumulated
        contribution is the operator ``I - (I - nu S)^m`` applied on its basis,
        whose df ``sum_k 1 - (1 - nu s_k)^m`` (over the eigenvalues ``s_k``
        of the smoother ``S``) grows with ``m`` towards the rank of the basis,
        the df of the unpenalized fit (Schulte and Rügamer, 2025). Columns:
        ``learner``, ``selections``, ``single_step_df`` (``tr(S)``),
        ``boosted_df``, ``unpenalized_df`` and ``drift_share``, the part of
        the gap from ``single_step_df`` to ``unpenalized_df`` that boosting
        has used. Each learner is measured on its own basis; overlap between
        learners is ignored. Trees and monotone splines are omitted.
        """
        from mboostrs._model_api import penalty_drift_rows
        from mboostrs._output import output_frame

        return output_frame(penalty_drift_rows(self), output)

    def concurvity(self, pairwise=False, output="auto"):
        """Concurvity of the fitted terms, as mgcv's ``concurvity`` measures it.

        Concurvity is the additive-model form of collinearity: how far one
        term's function space reaches into another's, which is when partial
        effects stop being separately identified -- the same fit can be
        redistributed between the terms, so their shapes, intervals and the
        selection that chose them are unstable (Wood, *Generalized Additive
        Models*, 2nd ed., section 5.6.2). Each row of the default table
        measures one term against the rest of the model (mgcv's
        ``concurvity(full = TRUE)``) and gives three values from 0
        (orthogonal) to 1 (the term lies in the others' span), where
        near-collinear terms can overshoot 1 slightly from the numerics:
        ``worst``, the largest fraction of a term function's squared norm
        the other terms can reproduce over every coefficient vector;
        ``observed``, the same fraction at the fitted coefficients (NaN for
        a term that was never selected); ``estimate``, an average over
        coefficient directions. ``worst`` is a pessimistic bound -- mgcv
        reports high values between unrelated rich bases -- so compare
        magnitudes against the other rows rather than against 0.

        With ``pairwise=True`` the rows give mgcv's
        ``concurvity(full = FALSE)``: one row per ordered pair, where
        ``term`` is measured against ``against`` alone, no intercept
        involved. Terms are reduced by the constant direction before
        measuring, as mgcv measures its constrained smooths, so a constant
        term reports NaN. Trees and ``%O%`` Kronecker learners are omitted.

        Parameters
        ----------
        pairwise : bool
            Term-against-the-rest rows (the default) or one row per
            ordered pair of terms.
        output : {"auto", "records", "polars", "pandas", "frame"}
            The return format; see :func:`mboostrs.gamboost` for the
            convention.
        """
        from mboostrs._model_api import concurvity_rows
        from mboostrs._output import output_frame

        return output_frame(concurvity_rows(self, pairwise=pairwise), output)

    @property
    def family(self):
        """The response family the model was fitted with."""
        return self._family

    @property
    def formula(self) -> str:
        """The formula as it was written (``y ~ .`` stays ``y ~ .``).
        ``variable_names`` lists the columns it expanded to."""
        bundle = getattr(self, "_refit_bundle", None) or {}
        return bundle.get("raw_formula") or self._formula

    @property
    def fitted_(self) -> np.ndarray:
        """The fitted linear predictor.

        Read-only: this is the fit's own array, not a copy. See
        :mod:`mboostrs._state`.
        """
        from mboostrs._state import readonly

        # Multinomial: return the (n, K-1) linear predictor matrix (matching
        # predict(type="link")); other families return the flat fitted vector.
        if isinstance(self._family, Multinomial):
            nr = self._family.n_classes - 1
            return readonly(
                np.asarray(self._fitted, np.float64).reshape(-1, nr)
            )
        return readonly(self._fitted)

    def fitted(self, mstop=None) -> np.ndarray:
        """Fitted values, optionally at a given mstop.

        mstop: int or None. If None, returns fitted values at the full mstop.
               If given, recomputes fitted values at iteration `mstop`.
        """
        if mstop is None or mstop >= self.mstop:
            return self.fitted_.copy()
        return self._predict_training(mstop)



    def aic(self, method="corrected", k=2.0):
        """Compute AIC for each boosting iteration.

        Returns an AicResult; refuses if a required hat matrix is unavailable.
        ``'corrected'`` (default) is the upstream per-observation Gaussian
        stopping criterion, not a total GAIC; ``'gmdl'`` is Gaussian-only.
        ``'classical'`` uses total likelihood risk and refuses Gaussian
        squared-error loss. Positive ``k`` applies only to the classical
        method. Results identify the criterion, normalization and conditional
        hat-trace df, which excludes model-search df. For a `btree` term the
        trace is likewise conditional on each step's fitted partition, so it
        excludes split-search df as well.
        """
        if isinstance(self._family, Multinomial):
            # R does not provide AIC for the Kronecker-product Multinomial model
            # either: AIC.mboost needs hatvalues, which are "not implemented for
            # models fitted via mboost" (the %O% operator is mboost-only). The df
            # trace would also need the n*(K-1) Kronecker expansion. Match R's
            # stance with a clear error instead of a misleading number.
            raise NotImplementedError(
                "aic() is not available for the Multinomial family (Kronecker-"
                "product base-learners have no hat matrix; R's AIC.mboost errors "
                "the same way)."
            )
        if isinstance(self._family, (CoxPH, WeibullFam, LoglogFam, LognormalFam)):
            raise NotImplementedError(
                "aic() is not available for survival families "
                "(response is a (n, 2) vector; a single residual SS is meaningless)."
            )
        # AIC is `-2 * loglik + k * df`. When the risk is not a
        # log-likelihood the first term is a negated robust or ranking loss, so
        # the result is not an information criterion and differences between
        # two of them are not on the AIC scale. Refuse rather than return a
        # number that compares like one.
        from mboostrs._likelihood import loglik_refusal

        refusal = loglik_refusal(self._family)
        if refusal:
            raise NotImplementedError(
                refusal.replace("logLik is not available", "aic() is not available", 1)
                + " Choose mstop by cross-validated risk (cvrisk) instead, "
                "which needs no likelihood."
            )
        from mboostrs._criteria import aic_contract, weighted_hat
        from mboostrs._refit import refit_available

        contract = aic_contract(self._family, method, k, self._control[2])
        store = self._replay_store()
        if store is None:
            _, reason = refit_available(self)
            raise ValueError(
                "aic() needs the fitted learners' Gram factorizations to take "
                f"the hat-matrix trace, and {reason}"
            )
        w = self._weights
        sumw = float(np.sum(w)) if w is not None else float(len(self._fitted))
        frame = self._replay_frame()
        if frame is None:
            _, reason = refit_available(self)
            raise ValueError(
                f"aic() needs the training response to form the risk, and "
                f"{reason}"
            )
        y = _extract_response(frame, self._response_name, self._family)
        # Match mboost's gMDL signal term, sum(response^2), even with weights.
        y_ss = float(np.sum(y**2))
        selected_rust = self._selected_rust()
        coefs = self._step_coefs()
        weighted = weighted_hat(self, store, selected_rust, coefs)
        result = store.compute_aic(
            selected_rust,
            self._risk.tolist(),
            sumw,
            y_ss,
            method,
            k,
            df=None if weighted is None else weighted[0],
            coefs=coefs,
        )
        if result is None:
            raise NotImplementedError(
                "hatvalues() needs a hat matrix for every selected learner, "
                "and at least one of them has none: `bmono` solves a "
                "constrained problem whose solution is not a linear map of "
                "the response. A `btree` learner is not a reason for this "
                "refusal: conditional on the partition it was fitted with, a "
                "tree step's prediction is a weighted average of the working "
                "response over each leaf, and its hat matrix is built from "
                "that averaging; the search that chose the splits is not "
                "counted, the convention a spline learner's data-chosen knots "
                "already get. The learners in this fit are "
                f"{sorted(set(self.selected))}. Without a hat matrix there is "
                "no trace to take, so there are no leverages and no "
                "hat-matrix degrees of freedom; choose mstop by "
                "cross-validated risk (cvrisk) instead."
            )
        aic_vals, mstop, df = result
        return AicResult(np.array(aic_vals), mstop, np.array(df),
                         provenance_from_model(self, "aic"), **contract)

    def hatvalues(self):
        """Per-observation leverage: the hat-matrix diagonal at the current
        mstop, as a 1-d array of length n.

        A deserialized model rebuilds its learner store from the stored
        specifications, so this works after a pickle round trip. It refuses
        when there is no hat matrix to take a diagonal of -- a `bmono`
        learner among the selected ones -- rather than returning `None` and
        leaving the caller to guess why. A `btree` learner among the
        selected ones does not refuse: each tree step contributes the
        averaging matrix of the partition it was fitted with, conditional
        on that partition, and the diagonal does not count the search that
        chose the splits -- the convention the conditional hat trace
        already applies to a spline learner's data-chosen knots and to
        learner selection itself.
        """
        from mboostrs._criteria import weighted_hat
        from mboostrs._refit import refit_available

        if isinstance(self._family, Multinomial):
            raise NotImplementedError("hatvalues() is not available for Multinomial family models.")
        store = self._replay_store()
        if store is None:
            _, reason = refit_available(self)
            raise ValueError(f"hatvalues() is unavailable: {reason}")
        selected_rust = self._selected_rust()
        coefs = self._step_coefs()
        weighted = weighted_hat(self, store, selected_rust, coefs)
        diag = (store.compute_hat_diag_py(selected_rust, coefs) if weighted is None
                else weighted[1])
        if diag is None:
            raise NotImplementedError(
                "hatvalues() needs a hat matrix for every selected learner, "
                "and at least one of them has none: `bmono` solves a "
                "constrained problem whose solution is not a linear map of "
                "the response. A `btree` learner is not a reason for this "
                "refusal: conditional on the partition it was fitted with, a "
                "tree step's prediction is a weighted average of the working "
                "response over each leaf, and its hat matrix is built from "
                "that averaging; the search that chose the splits is not "
                "counted, the convention a spline learner's data-chosen knots "
                "already get. The learners in this fit are "
                f"{sorted(set(self.selected))}. Without a hat matrix there is "
                "no trace to take, so there are no leverages and no "
                "hat-matrix degrees of freedom; choose mstop by "
                "cross-validated risk (cvrisk) instead."
            )
        return np.array(diag)

    def _data_fingerprint(self):
        """`data_fingerprint` of the response and weights `cvrisk` fits."""
        if self._data is None:
            return ""
        is_multinomial = isinstance(self._family, Multinomial)
        y = (_multinomial_response(self) if is_multinomial
             else _extract_response(self._data, self._response_name, self._family))
        n_resp = max(1, (getattr(self._family, "n_classes", 2) or 2) - 1)
        n_obs = (self._fitted.shape[0] // n_resp if is_multinomial else len(self._fitted))
        w = self._weights if self._weights is not None else np.ones(n_obs)
        return data_fingerprint(y, w)

    def _replay_slices(self, source, buser_newdata=None):
        """The raw per-learner data the core's replay recipes expect.

        One entry per learner, in formula order — the same order the fit
        boundary received them in. Most learners take one value per
        observation; `Bspatial` packs its two coordinates as `[x1, x2]`, and
        `Btree` and `Buser` take a row-major `(n, n_features)` design.

        `source` is the frame to read columns from: the training data for an
        in-sample replay, `newdata` for `predict(newdata=)`. `None` falls back
        to the cached training columns.

        Returns `(x_data, by_data)`; `by_data` is `None` where a learner has no
        `by=` term, which leaves the recipe's own training-length values in
        place. Both are lists of contiguous `float64` arrays: the boundary
        takes buffers directly, so building lists here would only be unboxed
        again on the Rust side.
        """

        # One numeric conversion of the frame per call, then slice columns
        # from it: the per-column `_extract_column` walk repeats the frame's
        # dtype checks and copies for every learner. Frames that will not cast
        # wholesale (a factor still stored as labels, duplicate names) fall
        # back to the per-column extractor, which raises the same errors it
        # always did.
        frame = None
        positions = {}
        if (source is not None and hasattr(source, "to_numpy")
                and hasattr(source, "columns")):
            try:
                names = list(source.columns)
            except TypeError:
                names = None
            if names is not None and len(set(names)) == len(names):
                try:
                    frame = np.ascontiguousarray(source.to_numpy(dtype=np.float64))
                    positions = {name: index for index, name in enumerate(names)}
                except (TypeError, ValueError):
                    frame = None

        def column(name):
            if source is None:
                return np.asarray(self._data_array_for(name), dtype=np.float64)
            if frame is not None and name in positions:
                return frame[:, positions[name]]
            return np.asarray(_extract_column(source, name), dtype=np.float64)

        x_data, by_data = [], []
        for spec in self._learners_spec:
            if isinstance(spec, Bspatial):
                values = np.concatenate([column(spec.name1), column(spec.name2)])
            elif isinstance(spec, Btree):
                values = np.column_stack([column(name) for name in spec.names]).ravel()
            elif isinstance(spec, Buser):
                if buser_newdata is not None and spec.name in buser_newdata:
                    design = buser_newdata[spec.name]
                elif spec._radial_basis is not None and source is not None:
                    from mboostrs.learners import _radial_design

                    design = _radial_design(source, spec._radial_basis)
                else:
                    design = spec.X
                values = np.asarray(design, dtype=np.float64).ravel()
            else:
                values = column(spec.name)
            x_data.append(np.ascontiguousarray(values, dtype=np.float64))
            by_data.append(
                np.ascontiguousarray(column(spec.by), dtype=np.float64)
                if spec.by is not None else None
            )
        return x_data, (by_data if any(b is not None for b in by_data) else None)

    def _range_rows(self, x_data, by_data=None):
        """Training range of each spline learner's covariate, and of its `by=`.

        Returns `(rows, index_of)`. A spline term contributes its basis times
        its `by=` variable, so an out-of-range `by=` value is reported and
        clipped alongside the covariate. `index_of` maps each row's position
        to `("x", learner position)` or `("by", learner position)`, which
        tells the policy which array the row measured.

        Only the basis learners are listed. A `bols` term is linear
        everywhere, so predicting outside the training range continues the
        same line the fit estimated rather than a basis the fit never
        constrained -- a different kind of claim, and not the one the boundary
        knots are about.
        """
        from mboostrs._extrapolation import row_for

        rows, index_of = [], {}
        for position, (spec, values) in enumerate(
                zip(self._learners_spec, x_data, strict=False)):
            if not isinstance(spec, (Bbs, Bns, Bmono, Bconstrained, Bspatial)):
                continue
            label = f"{type(spec).__name__.lower()}({spec.name})"
            if spec.by is not None:
                label = f"{label}, by={quote_name(spec.by)}"
            if isinstance(spec, Bspatial):
                # The two coordinates are packed one after the other; each
                # margin is checked against its own column's training range.
                half = len(values) // 2
                margins = [
                    (spec.name1, slice(0, half), self._column_range(spec.name1)),
                    (spec.name2, slice(half, None), self._column_range(spec.name2)),
                ]
            else:
                margins = [(spec.name, None, self._training_range(spec))]
            for variable, part, (low, high) in margins:
                index_of[len(rows)] = ("x", position, part)
                rows.append(row_for(label, variable,
                                    values if part is None else values[part], low, high))
            if by_data is not None and by_data[position] is not None:
                by_range = self._by_training_range(spec.by)
                if by_range is not None:
                    index_of[len(rows)] = ("by", position, None)
                    rows.append(row_for(label, spec.by, by_data[position], *by_range))
        return rows, index_of

    def _column_range(self, name):
        """Training minimum and maximum of one column, as floats."""
        column = np.asarray(self._data_array_for(name), dtype=np.float64)
        return float(np.min(column)), float(np.max(column))

    def _apply_extrapolation_policy(self, x_data, policy, by_data=None):
        """Act on `extrapolation=`, returning the values to evaluate at.

        Returns `(x_data, by_data)`, with the covariate and `by=` arrays clamped
        to their training ranges under ``"clip"`` and unchanged otherwise.
        """
        from mboostrs._extrapolation import POLICIES, apply_policy

        rows, index_of = self._range_rows(x_data, by_data)
        if not rows:
            if policy not in POLICIES:
                raise ValueError(
                    f"extrapolation must be one of {POLICIES}, got {policy!r}"
                )
            return x_data, by_data
        values = {row_index: _part_of(x_data if kind == "x" else by_data, position, part)
                  for row_index, (kind, position, part) in index_of.items()}
        updated = apply_policy(rows, values, policy, stacklevel=4)
        if updated is values:
            return x_data, by_data
        out_x = list(x_data)
        out_by = list(by_data) if by_data is not None else None
        for row_index, (kind, position, part) in index_of.items():
            target = out_x if kind == "x" else out_by
            target[position] = _merge_part(target[position], part, updated[row_index])
        return out_x, out_by

    def _newdata_under_policy(self, newdata, policy):
        """Apply `extrapolation=` to raw `newdata` for the paths that rebuild
        designs themselves (contributions, partial effects).

        Errors and warnings are as in :meth:`predict`; under ``"clip"`` the
        returned frame holds the clipped values in each spline learner's
        covariate column and in its `by=` column. A spline on a derived column
        (``log(x)``) cannot be clipped in the raw frame, so that combination is
        refused.
        """
        from mboostrs._extrapolation import POLICIES, apply_policy

        if policy not in POLICIES:
            raise ValueError(f"extrapolation must be one of {POLICIES}, got {policy!r}")
        if newdata is None or policy == "allow":
            return newdata
        replayed, buser = _replay_interactions(
            self, newdata, None, stage="raw", unseen_levels="average")
        replayed = _replay_transforms(self, replayed, "average")
        replayed, buser = _replay_interactions(
            self, replayed, buser, stage="encoded", unseen_levels="average")
        x_data, by_data = self._replay_slices(replayed, buser)
        rows, index_of = self._range_rows(x_data, by_data)
        if not rows:
            return newdata
        values = {row_index: np.asarray(
                      _part_of(x_data if kind == "x" else by_data, position, part),
                      dtype=np.float64)
                  for row_index, (kind, position, part) in index_of.items()}
        updated = apply_policy(rows, values, policy, stacklevel=5)
        if updated is values:
            return newdata
        from mboostrs._frame import column_names

        present = set(column_names(newdata))
        changes = {}
        for row_index in index_of:
            if np.array_equal(updated[row_index], values[row_index]):
                continue
            name = rows[row_index].variable
            if name not in present:
                raise NotImplementedError(
                    f"extrapolation='clip' cannot clip {name!r}: the spline is "
                    "fitted on a derived column the new data does not carry. "
                    "Use 'warn', 'error' or 'allow'.")
            value = updated[row_index]
            if name in changes:
                # The same column is also a covariate or `by=` of another
                # learner: keep the clamp of both.
                row = rows[row_index]
                value = np.clip(changes[name], row.train_low, row.train_high)
            changes[name] = value
        if isinstance(newdata, dict):
            return {**newdata, **changes}
        from mboostrs.transforms import _with_columns

        return _with_columns(newdata, changes)

    def _range_pairs(self, newdata):
        """`(kind, RangeRow)` for every spline and linear term against `newdata`.

        `newdata=None` reads the training data. The raw frame goes through the
        fit's transforms and interaction recipes first, as in predict.
        """
        buser_newdata = None
        raw = self._data if newdata is None else newdata
        tensor_rows = _tensor_margin_rows(self, raw)[0]
        if newdata is not None:
            # Raw new data goes through the fit's transforms and interaction
            # recipes first, as in predict; read straight from the frame, a
            # categorical column failed as "not numbers".
            newdata, buser_newdata = _replay_interactions(
                self, newdata, None, stage="raw", unseen_levels="average")
            newdata = _replay_transforms(self, newdata, "average")
            newdata, buser_newdata = _replay_interactions(
                self, newdata, buser_newdata, stage="encoded", unseen_levels="average")
        source = self._data if newdata is None else newdata
        x_data, by_data = self._replay_slices(source, buser_newdata)
        rows, _index = self._range_rows(x_data, by_data)
        pairs = [("spline basis", row) for row in rows + tensor_rows]
        pairs.extend(("linear", row) for row in self._linear_range_rows(source))
        return pairs

    def range_report(self, newdata=None, output="auto"):
        """Each learner's training range against `newdata`.

        One row per spline learner and per `by=` variable of a spline learner,
        and per linear (`bols`) term on a numeric column: the interval it was
        fitted over, the range it is being asked about, how many rows fall each
        side, and the worst distance in training ranges -- the number that
        separates a boundary case from a different question. `term` says which
        kind of row it is. A spline basis is extrapolated by its edge slope,
        which the fit does not constrain, and `predict(extrapolation=)` acts on
        those rows. A linear term continues the line the fit estimated, so it is
        reported here but not warned about or clipped; the distance still says
        how far the prediction relies on that line holding. With no `newdata`
        the report reads off the ranges of the training data.

        `gamboostLSS` prediction and R `predict()` apply the same
        `extrapolation=` policy to spline learners.
        """
        from mboostrs._output import output_frame

        records = [{**row.as_dict(), "term": kind}
                   for kind, row in self._range_pairs(newdata)]
        return output_frame(records, output)

    def extrapolation_report(self, newdata, output="auto"):
        """Terms and covariates with rows outside their training range.

        One row per term and covariate that has at least one row of `newdata`
        outside its training range. The columns are `parameter` (always
        ``"f"`` for this model), `term` (the learner, such as ``bbs(x)`` or
        ``bbs(x, by=z)``), `covariate` (the column whose values left the
        range), `kind` (``"spline basis"`` or ``"linear"``), `n_below` and
        `n_above` (rows below and above the training range), `n_outside`,
        `train_low` and `train_high` (the training range), `new_low` and
        `new_high` (the range of `newdata` on that covariate), and
        `worst_distance` (the farthest outside row, in training ranges; 1
        means a whole training range beyond the edge).

        A covariate that stays in range is not listed, so an empty frame means
        `newdata` lies inside every training range. `predict(extrapolation=)`
        acts on the spline rows: ``"clip"`` clamps the covariate and `by=`
        values listed here.
        """
        from mboostrs._extrapolation import REPORT_COLUMNS, report_records
        from mboostrs._model_api import STANDARD_PREDICTOR
        from mboostrs._output import output_frame

        records = report_records(
            (STANDARD_PREDICTOR, kind, row) for kind, row in self._range_pairs(newdata))
        return output_frame(records, output, columns=REPORT_COLUMNS)

    def _linear_range_rows(self, source):
        """Training range against `source` for each `bols` on a numeric column.

        A `bols` over a categorical column (lowered to level indicators) or a
        derived column the prediction data does not carry has no range to
        report and is skipped.
        """
        from mboostrs._extrapolation import row_for

        rows = []
        for spec in self._learners_spec or ():
            if not isinstance(spec, Bols):
                continue
            try:
                training = np.asarray(self._data_array_for(spec.name), dtype=np.float64)
                values = np.asarray(_extract_column(source, spec.name), dtype=np.float64)
            except (KeyError, ValueError, TypeError):
                continue
            if training.size == 0 or not np.all(np.isfinite(training)):
                continue
            rows.append(row_for(f"bols({quote_name(spec.name)})", spec.name, values,
                                float(np.min(training)), float(np.max(training))))
        return rows

    def _training_range(self, spec):
        """The interval a spline learner's basis was built over.

        Outside it the basis extrapolates linearly, which is what
        `predict(newdata=)` warns about. An explicit `boundary_knots=` wins;
        otherwise the range is the training column's own.
        """
        given = getattr(spec, "boundary_knots", None)
        if given is not None:
            return float(given[0]), float(given[1])
        column = np.asarray(self._data_array_for(spec.name), dtype=np.float64)
        return float(np.min(column)), float(np.max(column))

    def _by_training_range(self, name):
        """The training range of a `by=` variable: its own column's range.

        A `by=` value multiplies the basis, so the range that matters is the
        one the variable took in training; `boundary_knots=` does not apply.
        Returns `None` when the variable has no numeric training column (a
        categorical or derived one): it has no range to check, so the term
        predicts as it always did.
        """
        try:
            column = np.asarray(self._data_array_for(name), dtype=np.float64)
            return float(np.min(column)), float(np.max(column))
        except (KeyError, ValueError, TypeError):
            return None

    def _fit_learner_specs(self, data, *, vector_offset=False):
        """The learner specs the fit sent to the core, rebuilt on `data`.

        Refits (cross-validation folds, stability-selection subsamples) must
        prepare the same learners the fit did. For Multinomial that includes
        the Kronecker form of a `bbs` (penalty `K + I`, `df` calibrated over
        the class blocks); without `kronecker_n_resp` the spec is an ordinary
        `bbs`, whose single-class lambda the Kronecker lift then reuses, and
        the fold refits fitted a different smoother from the model's.
        """
        from mboostrs.gamlss import learner_spec

        n_classes = getattr(self._family, "n_classes", None)
        kronecker_n_resp = (n_classes - 1 if isinstance(self._family, Multinomial)
                            and n_classes else None)
        bols_form = _bols_form_for(self._learners_spec, vector_offset=vector_offset,
                                   kronecker=kronecker_n_resp is not None,
                                   family=self._family)
        return [learner_spec(spec, data, bols_form=bols_form,
                             kronecker_n_resp=kronecker_n_resp)
                for spec in self._learners_spec]

    def _replay_store(self):
        """The learner store to predict through, rebuilding it if needed.

        A store owns Gram factorizations and work buffers and cannot be
        pickled, so a deserialized model arrives without one. Rebuilding it
        from the stored learner specifications and training weights gives back
        a store identical to the one the fit produced, which is what lets a
        restored model predict — and compute AIC and hat values — the same way.

        Returns `None` when the model has no specifications to rebuild from, or
        when it was fitted through the Kronecker (Multinomial) path, whose
        learner list does not correspond one-to-one with the formula terms.
        """
        store = self._learner_store
        if store is not None and store.basis_info_len() == len(self._learners_spec or ()):
            return store
        if (
            self._learners_spec is None
            or _prepare_learner_store is None
            or isinstance(self._family, Multinomial)
        ):
            return None
        from mboostrs.gamlss import learner_spec

        weights = self._weights
        if weights is None:
            return None
        source = self._replay_frame()
        if source is None:
            return None
        # The same `bols` form the fit chose: a per-observation offset makes
        # every `bols` carry its own intercept column, and a store rebuilt in
        # the centered form reads those coefficients against the wrong design.
        bols_form = _bols_form_for(
            self._learners_spec,
            vector_offset=_is_vector_offset(self._offset),
            kronecker=isinstance(self._family, Multinomial),
            family=self._family)
        try:
            specs = [learner_spec(spec, source, bols_form=bols_form)
                     for spec in self._learners_spec]
        except (KeyError, ValueError):
            return None
        self._learner_store = _prepare_learner_store(
            specs, np.asarray(weights, dtype=np.float64), self._nu
        )
        return self._learner_store

    def _learner_column_names(self) -> list[str]:
        """Every training column the learners read, in formula order.

        `Buser` is absent: its design travels on the learner itself rather than
        being read from the data frame.
        """
        names: list[str] = []
        for spec in self._learners_spec or ():
            if isinstance(spec, Bspatial):
                candidates = [spec.name1, spec.name2]
            elif isinstance(spec, Btree):
                candidates = list(spec.names)
            elif isinstance(spec, Buser):
                candidates = list(spec._radial_basis["names"]) if spec._radial_basis is not None else []
            else:
                candidates = [spec.name]
            if spec.by is not None:
                candidates.append(spec.by)
            for candidate in candidates:
                if candidate not in names:
                    names.append(candidate)
        return names

    def _replay_column_cache(self) -> dict | None:
        """The learner columns, materialized for storage.

        The response goes in too, in its own dtype: `aic()` needs it for the
        residual sum of squares, and it is the only thing that stopped a
        deserialized model from computing one.
        """
        if self._learners_spec is None:
            return self._replay_columns
        columns = {}
        for name in self._learner_column_names():
            try:
                columns[name] = np.asarray(self._data_array_for(name), dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                return self._replay_columns
        response = self._response_name
        for name in (response,) if isinstance(response, str) else (response or ()):
            if name in columns or self._data is None:
                continue
            try:
                columns[name] = np.asarray(_extract_column(self._data, name))
            except (KeyError, TypeError, ValueError):
                pass
        return columns

    def _replay_frame(self):
        """A column mapping the learners can be rebuilt from.

        The training frame when the model still holds one, otherwise the
        columns cached at fit time — which is what a deserialized model has.
        """
        if self._data is not None:
            return self._data
        return self._replay_columns

    def _data_array_for(self, name: str) -> np.ndarray:
        """A training column, by name.

        `_replay_columns` holds every column the learners read, which is what a
        deserialized model has instead of the training frame. Six per-kind
        caches (`scalar_x_data`, `bbs_x_data`, ...) used to be consulted in turn
        here; they held the same columns, keyed by learner kind because the
        prediction path needed them grouped that way.
        """
        if self._replay_columns is not None and name in self._replay_columns:
            return self._replay_columns[name]
        return _extract_column(self._data, name)

    # ── Milestone 8: extraction / inspection ──────────────────────

    def variable_names(self):
        """Names of all variables used in the model (response + predictors)."""
        names = []
        if self._response_name is not None:
            resp = self._response_name
            if isinstance(resp, str):
                names.append(resp)
            else:
                names.extend(resp)
        if self._learners_spec is not None:
            for spec in self._learners_spec:
                vn = getattr(spec, "name", None)
                if vn and vn not in names:
                    names.append(vn)
        return names

    def base_learner_names(self, disambiguate=True):
        """Names of all base learners.

        A learner is stored under the column it reads, and two learners can read
        one column: ``bols(x) + bbs(x, df=1, center=True)`` is the standard way
        to split a term into a linear part and a deviation from it, and both
        halves are named ``"x"``. Where a name repeats, the learner type is
        appended — ``"x (bols)"``, ``"x (bbs)"`` — so the list identifies the
        learners rather than the columns. :meth:`learner_index` accepts that
        form, which is how such a name reaches ``which=``.

        Parameters
        ----------
        disambiguate : bool
            Append the learner type to names that more than one learner shares.
            :meth:`coef`, :meth:`partial_effects` and :meth:`varimp` are keyed
            by these names. ``False`` returns the stored column names, which
            :attr:`selected` reports.

        Returns
        -------
        list of str
        """
        names = list(self._learner_names)
        specs = getattr(self, "_learners_spec", None)
        if not disambiguate or specs is None:
            return names
        return _disambiguated_names(names, specs)

    def _interaction_labels(self):
        """Each ``%X%`` term's derived column name mapped to the term as written."""
        return {
            interaction.derived: interaction.name
            for interaction in getattr(self, "_interactions", None) or ()
            if interaction.derived.startswith("__ix")
        }

    def _learner_alias(self, name):
        """The label of a ``%X%`` term given its derived column name.

        Learners are named by the term the user wrote. Derived interaction
        names such as ``__ix0`` are accepted as aliases.
        """
        return self._interaction_labels().get(name, name)

    def _which_indices(self, which):
        """Learner indices one ``which=`` entry names.

        A name two learners share (``bbs(x) + bols(x)`` both read ``x``)
        names both; the disambiguated ``"x (bols)"`` from
        :meth:`base_learner_names` names one; an integer is an index.
        """
        if isinstance(which, (int, np.integer)):
            return [int(which)]
        if not isinstance(which, str):
            raise TypeError(f"Invalid which element: {which!r}")
        name = self._learner_alias(which)
        matches = [i for i, n in enumerate(self._learner_names) if n == name]
        if matches:
            return matches
        try:
            return [self.learner_index(which)]
        except ValueError:
            pass
        # A categorical column's learner is labelled with the term as written
        # (`brandom(g)`), not with the column; the column name still selects it.
        specs = getattr(self, "_learners_spec", None) or []
        by_column = [
            i for i in range(len(self._learner_names))
            if name in self._learner_columns(i)
            or (len(specs) == len(self._learner_names)
                and getattr(specs[i], "name", None) == name)
        ]
        if by_column:
            return by_column
        raise ValueError(f"Unknown base-learner: {which!r}")

    def _learner_columns(self, index):
        """The data columns learner ``index`` reads, sorted.

        A ``%X%`` tensor is labelled with the term the user wrote, so its
        columns come from the stored interaction's two margins rather than
        from the label. A joint ``btree`` is labelled by its first column
        only, so its candidate columns come from the spec's ``names`` --
        every column the tree may split on -- rather than from the label.
        """
        specs = getattr(self, "_learners_spec", None)
        spec = specs[index] if specs is not None else None
        derived = getattr(spec, "name", None)
        if isinstance(derived, str) and derived.startswith("__ix"):
            for interaction in getattr(self, "_interactions", None) or ():
                if interaction.derived == derived:
                    return sorted({interaction.left.column, interaction.right.column})
        if isinstance(spec, Btree):
            return sorted(set(spec.names))
        return sorted(_top_level_parts(self._learner_names[index]))

    def selection_counts(self, output="auto"):
        """One row per base learner with the boosting updates it received.

        :attr:`selected` is keyed by the column a learner reads, and two
        learners can read one column -- ``bols(x) + bbs(x, df=1,
        center=True)`` is the standard linear-plus-deviation split, and
        counting ``selected`` merges the halves. This table counts per
        learner, names each row with the disambiguated form of
        :meth:`base_learner_names`, and includes the learners that were
        never selected with zero updates, so they remain distinguishable from
        learners that are absent from the model.

        Parameters
        ----------
        output : {"auto", "pandas", "polars", "records"}
            Frame backend for the table, or ``"records"`` for the raw rows.

        Returns
        -------
        DataFrame or list of dict
            One row per learner, most updates first: ``learner``
            (disambiguated name), ``column`` (the column it reads),
            ``updates`` (selections received), ``share`` (updates over all
            boosting steps).
        """
        from collections import Counter

        from mboostrs._output import output_frame

        counts = Counter(self._selected)
        total = max(len(self._selected), 1)
        records = []
        for index, name in enumerate(self.base_learner_names()):
            updates = int(counts.get(index, 0))
            records.append({
                "learner": name,
                "column": ",".join(self._learner_columns(index)),
                "updates": updates,
                "share": updates / total,
            })
        records.sort(key=lambda row: row["updates"], reverse=True)
        return output_frame(records, output)



    def model_matrix(self, which=None):
        """Return the design matrix for the specified base-learner(s).

        Parameters
        ----------
        which : str, int, or list thereof, optional
            Base-learner name(s) or index. None = all selected learners.
        """
        return self.extract(what="design", which=which)



    def nuisance(self):
        """Return nuisance parameters (family-specific).

        The family's nuisance parameter at the model's step count, as R's
        ``nuisance()`` returns it: the size of ``NBinomial`` (``Var = mu + mu**2 / sigma``), the shape
        of ``GammaReg`` and the parametric survival families, the thresholds
        of ``PropOdds``. None for a family that has none.
        """
        if self._family_sigma_path and self.mstop < len(self._family_sigma_path):
            return np.asarray(self._family_sigma_path[self.mstop], dtype=np.float64).copy()
        if self._family_sigma is not None and len(self._family_sigma):
            return np.asarray(self._family_sigma, dtype=np.float64).copy()
        return None



    def predict(self, newdata=None, *, mstop=None, type=_TYPE_DEFAULT, which=None,
                aggregate="sum", buser_newdata=None, extrapolation="warn",
                unseen_levels=None, offset=None):
        """Predicted values, optionally decomposed by learner.
        Usage::

            import mboostrs as mb
            fit = mb.gamboost("y ~ bbs(x, df=4)", data)
            fit.predict(newdata=data, type="response")
            fit.predict(data, mstop=100, which="x (bbs)")  # one learner's contribution

        Parameters
        ----------
        newdata : dict or DataFrame, optional
            New observations. If None, predicts on the training data.
        offset : number or array-like, optional
            The offset these rows start from. A model fitted with a
            per-observation offset needs the offset of every *new* row, so
            predicting ``newdata`` requires this argument; the training rows
            reuse the fitted vector.
        mstop : int, optional
            Truncate the boosting path at this iteration.
        type : {"link", "response", "class", "log_odds"}
            Link-scale or response-scale predictions. ``"log_odds"`` is the
            link scale times 2 for ``Binomial``, whose link is half the
            log-odds; with ``which=`` it gives each learner's contribution to
            the log-odds, the scale other libraries report.
            ``"class"`` returns 0/1 class codes for Binomial, AdaExp,
            HingeLoss and AUC. A score on the decision boundary returns 0.
        which : int, str, or list thereof, optional
            If given, return only the contribution of the specified learner(s).
            Strings are learner names.
        aggregate : {"sum", "cumsum", "none"}
            "sum" returns the aggregated contribution.
            "cumsum" returns cumulative contributions at each step.
            "none" returns per-step contributions.
        buser_newdata : dict, optional
            Passed through for Buser learners (see main predict).
        extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
            How to treat rows that leave a spline learner's training range.
            The check covers the covariate and the ``by=`` variable of each
            spline term, since the term's effect is the basis times the
            ``by=`` value. The warning names the terms, the covariate, the rows
            each side, and the worst distance in training ranges. Excursions
            below 5% of the span do not warn, but ``error`` rejects and
            ``clip`` clamps every value strictly outside the interval.
            ``extrapolation_report(newdata)`` lists the same excursions as a
            table. The check covers univariate spline learners, both coordinates
            of a ``bspatial`` term, and the B-spline marginals of a ``%X%``
            term, in full-model predictions, in ``which=`` contributions and in
            Multinomial predictions. ``btree``, ``bkernel``, ``buser`` and
            ``brad`` terms have no training-range boundary to check, so this
            policy does not apply to them (``validate()`` lists them).
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            What a term over a categorical column does with a level in
            ``newdata`` that the fit never saw, such as a group held out of a
            grouped cross-validation fold. ``"error"`` refuses the prediction.
            ``"average"`` predicts the row as a member of an average group:
            each term that reads the column contributes the mean of its
            contributions over the fitted levels. For ``brandom`` that is the
            mean level coefficient -- near zero for a random intercept, and for
            a random slope (``brandom(g, by=x)``) the part of the average slope
            the group term absorbed during boosting, which a new group's
            prediction needs. A ``bols`` over a factor, a factor ``by=``, a
            ``%X%`` factor margin and a ``btree`` split on the column average
            the same way (a tree over the levels on each side of each split).
            ``"zero_effect"`` gives the level a zero coefficient, the value
            the ridge penalty shrinks every level towards (a ``btree`` sends it
            to the branch that held more training weight); it is the
            prediction ``cvrisk`` makes for a group its fold holds out, where
            that group's rows enter the fit with zero weight.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        extrapolation = _model_extrapolation(self, extrapolation)
        if type_was_default := type is _TYPE_DEFAULT:
            type = "link"
        if isinstance(mstop, numbers.Integral) and mstop > self.mstop:
            raise ValueError(
                f"predict(mstop={int(mstop)}) is beyond the model's fitted {self.mstop} "
                "steps. The prediction would silently use the last fitted step. Refit the "
                f"model with mstop={int(mstop)}, or pass mstop <= {self.mstop}."
            )
        if type == "class":
            from mboostrs.families import _class_prediction

            if which is not None or aggregate != "sum":
                raise ValueError("type='class' requires the whole model and aggregate='sum'")
            return _class_prediction(self.predict(
                newdata=newdata, mstop=mstop, type="link",
                buser_newdata=buser_newdata, extrapolation=extrapolation,
                unseen_levels=unseen_levels, offset=offset), self._family)
        if type == "log_odds":
            from mboostrs.families import log_odds_factor, scale_output

            factor = log_odds_factor(self._family)
            return scale_output(self.predict(
                newdata=newdata, mstop=mstop, type="link", which=which,
                aggregate=aggregate, buser_newdata=buser_newdata,
                extrapolation=extrapolation, unseen_levels=unseen_levels,
                offset=offset), factor)
        unseen_levels = _check_unseen_levels(unseen_levels)
        # The fit built each %X% design from the raw columns, ran the recipe,
        # then built the factor by= designs from the encoded columns;
        # prediction replays them in the same order.
        clamp = _tensor_margin_clamp(self, newdata, extrapolation)
        newdata, buser_newdata = _replay_interactions(
            self, newdata, buser_newdata, stage="raw", unseen_levels=unseen_levels,
            clamp=clamp)
        newdata = _replay_transforms(self, newdata, unseen_levels)
        newdata, buser_newdata = _replay_interactions(
            self, newdata, buser_newdata, stage="encoded", unseen_levels=unseen_levels,
            clamp=clamp)
        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link' or 'response', got {type!r}")
        maybe_warn_link_scale(self, type_was_default, method="predict")
        if aggregate not in ("sum", "cumsum", "none"):
            raise ValueError(f"aggregate must be 'sum', 'cumsum', or 'none', got {aggregate!r}")

        if which is not None or aggregate != "sum":
            return self._prediction_contributions(
                newdata=self._newdata_under_policy(newdata, extrapolation),
                mstop=mstop,
                type=type,
                which=which,
                aggregate=aggregate,
                buser_newdata=buser_newdata,
                unseen_levels=unseen_levels,
            )
        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link' or 'response', got {type!r}")
        if newdata is not None and isinstance(self._family, Multinomial):
            newdata = self._newdata_under_policy(newdata, extrapolation)
            if self._learner_store is not None and self._learners_spec is not None:
                nr = self._family.n_classes - 1
                # No by= handling here is intentional: Multinomial rejects by=
                # at fit time (the Kronecker path cannot thread it), so a fitted
                # Multinomial model never carries a by= learner to replay.
                x_data = [
                    np.ascontiguousarray(_extract_column(newdata, spec.name), dtype=np.float64)
                    for spec in self._learners_spec
                ]
                selected_rust = self._selected_rust()
                coefs_rust = self._coefs_rust()
                eta = np.asarray(
                    self._learner_store.predict_newdata_kronecker(
                        0.0,
                        selected_rust,
                        coefs_rust,
                        x_data,
                        nr,
                        mstop=mstop,
                    ),
                    dtype=np.float64,
                ).reshape(-1, nr)
            else:
                eta = self._multinomial_newdata_eta(newdata, mstop)
            if type == "response":
                return self._multinomial_predict(eta.ravel(), "response")
            return eta
        if newdata is None:
            k = mstop if mstop is not None else self.mstop
            if k >= self.mstop:
                eta = self._fitted.copy()
            else:
                eta = self._predict_training(k)
        else:
            if self._learners_spec is None:
                raise ValueError("Cannot predict on new data: model was fitted without storing learner specs")
            missing_buser = any(isinstance(s, Buser) and s._radial_basis is None
                                and (buser_newdata is None or s.name not in buser_newdata)
                                for s in self._learners_spec)
            if missing_buser:
                raise ValueError(
                    "predict(newdata=...) with Buser learners requires "
                    "buser_newdata: dict mapping Buser name -> X_new matrix "
                    "(n_new, p). Pass the new design matrices for each "
                    "Buser learner."
                )
            store = self._replay_store()
            if store is None:
                raise ValueError(
                    "Cannot predict on new data: this model carries no learner "
                    "store and none could be rebuilt from its specifications"
                )
            x_data, by_data = self._replay_slices(newdata, buser_newdata)
            selected_rust = self._selected_rust()
            coefs_rust = self._coefs_rust()

            x_data, by_data = self._apply_extrapolation_policy(
                x_data, extrapolation, by_data)
            # An offset-only model has no learner column to count rows from.
            n_rows = _frame_length(newdata, "newdata")
            effective_offset = _resolve_prediction_offset(
                self._offset, offset, self._family, len(x_data[0]) if x_data else n_rows,
                requires_explicit=True)

            eta = np.asarray(
                store.predict_newdata_grouped(
                    effective_offset, selected_rust, coefs_rust, x_data, mstop, by_data,
                    unseen_levels=unseen_levels, n_rows=n_rows,
                ),
                dtype=np.float64,
            )
        if isinstance(self._family, Multinomial):
            return self._multinomial_predict(eta, type)
        if isinstance(self._family, PropOdds) and type == "response":
            from mboostrs._rust import propodds_response as _propodds_response

            k_effective = mstop if mstop is not None else self.mstop
            if self._family_sigma_path and k_effective >= 0 and k_effective < len(self._family_sigma_path):
                sigma = self._family_sigma_path[k_effective]
            else:
                sigma = self._family_sigma
            if sigma:
                k_cat = len(sigma) + 1
                flat = _propodds_response(np.asarray(eta, dtype=np.float64).tolist(), list(sigma))
                return np.asarray(flat, dtype=np.float64).reshape(len(eta), k_cat)
            return eta
        if type == "response":
            return _inverse_link(eta, self._family)
        return eta

    def _prediction_contributions(
        self, *, newdata=None, mstop=None, type="link", which=None, aggregate="sum",
        buser_newdata=None, unseen_levels="error",
    ):
        """Decompose predictions into per-learner contributions.

        For aggregate="sum" with which, reuses partial_effects logic.
        For aggregate="none"/"cumsum", computes per-step contributions.
        """
        k = mstop if mstop is not None else self.mstop
        source = newdata if newdata is not None else self._data

        # Resolve which learners to include.
        if which is not None:
            if isinstance(which, int | str):
                which = [which]
            allowed_names = set()
            for w in which:
                if isinstance(w, int):
                    allowed_names.add(self._learner_names[w])
                else:
                    allowed_names.add(w)
        else:
            allowed_names = None

        if aggregate == "sum":
            if allowed_names is not None:
                # Use partial_effects for per-learner aggregated contributions.
                # New rows ran through the recipe and the %X% designs in
                # `predict`; the training rows are evaluated on the fitted
                # designs (newdata=None).
                pe = self.partial_effects(which=list(allowed_names), newdata=newdata,
                                          unseen_levels=unseen_levels, _prepared=True,
                                          _buser=buser_newdata, mstop=mstop)
                parts = []
                for name in pe:
                    _, effect = pe[name]
                    parts.append(effect)
                if not parts:
                    # A learner the fit never selected contributes zero on
                    # every row. `len(source)` counted a dict's columns, not
                    # its rows.
                    n = (len(self._fitted) if newdata is None
                         else self._source_n_rows(source))
                    return np.zeros(n, dtype=np.float64)
                result = np.sum(parts, axis=0)
            else:
                # aggregate="sum", which=None → normal predict
                return self.predict(newdata=newdata, mstop=mstop, type="link",
                                    buser_newdata=buser_newdata,
                                    unseen_levels=unseen_levels)
            if type == "response":
                return _inverse_link(result, self._family)
            return result

        # aggregate="none" or "cumsum": compute per-step contributions.
        if self._learner_store is None and self._learners_spec is None:
            raise ValueError("Cannot compute step-level contributions without learner state")

        # Rows filtered out by `which` become zero vectors. Their length must
        # match the prediction rows, which is the newdata row count, not the
        # training length — so defer the zero-fill until a real contribution
        # has revealed `n`.
        step_contribs = []
        n = None
        for s in range(k):
            sel = self._selected[s]
            sel_name = self._learner_names[sel]
            if allowed_names is not None and sel_name not in allowed_names:
                step_contribs.append(None)
                continue
            # Single-step prediction: offset=0, only this step's learner.
            step_contrib = self._predict_single_step(
                s,
                newdata=source,
                buser_newdata=buser_newdata,
                unseen_levels=unseen_levels,
            )
            if n is None:
                n = len(step_contrib)
            step_contribs.append(step_contrib)

        if n is None:
            # Every selected step was filtered out by `which`; fall back to the
            # source row count so the all-zero result still has the right shape.
            n = len(self._fitted) if newdata is None else self._source_n_rows(source)

        step_matrix = np.array(
            [c if c is not None else np.zeros(n, dtype=np.float64) for c in step_contribs],
            dtype=np.float64,
        )

        if aggregate == "cumsum":
            result = np.cumsum(step_matrix, axis=0)
            if allowed_names is None:
                result = result + self._offset
        else:
            result = step_matrix

        if type == "response":
            if aggregate == "none":
                raise ValueError(
                    "type='response' with aggregate='none' is not supported; "
                    "contributions in response space are not additive"
                )
            if isinstance(self._family, PropOdds):
                raise ValueError("type='response' with contributions not supported for PropOdds")
            if aggregate == "cumsum" and allowed_names is not None:
                raise ValueError("type='response' with aggregate='cumsum' requires which=None (full model)")
            result = _inverse_link(result, self._family)
        return result

    def _source_n_rows(self, source):
        """Best-effort row count of a prediction `source` (dict/DataFrame).

        Used only for the rare all-steps-filtered zero-result shape. Tries each
        learner's column name(s) until one resolves; falls back to the training
        length if none do (e.g. only external-design learners).
        """
        for spec in self._learners_spec:
            for attr in ("name", "name1"):
                nm = getattr(spec, attr, None)
                if isinstance(nm, str):
                    try:
                        return len(_extract_column(source, nm))
                    except (KeyError, TypeError, ValueError):
                        pass
            names = getattr(spec, "names", None)
            if names:
                try:
                    return len(_extract_column(source, names[0]))
                except (KeyError, TypeError, ValueError):
                    pass
        return len(self._fitted)

    def _predict_single_step(self, step_idx, *, newdata=None, buser_newdata=None,
                             unseen_levels="error"):
        """Compute the prediction contribution of a single boosting step.

        Returns a 1-D array of length n (training data or newdata).
        The contribution is ``nu * B_j(x) @ coef_s`` where j is the selected
        learner at step s.
        """
        sel = self._selected[step_idx]
        coef = self._coefficients[step_idx]
        source = newdata if newdata is not None else self._data

        # One step, no offset, through the core's replay. Bkernel and Buser
        # used to be computed here in Python because the store could not
        # replay them; the recipe the fit returns now covers every kind, so
        # they take this path like everything else.
        store = self._replay_store()
        if store is None:
            raise ValueError(
                "cannot replay a single boosting step: this model carries no "
                "learner store and none could be rebuilt from its specifications"
            )
        x_data, by_data = self._replay_slices(source, buser_newdata)
        single_coef = np.ascontiguousarray(coef, dtype=np.float64)
        return np.asarray(
            store.predict_newdata_grouped(
                0.0, [self._formula_to_rust(sel)], [single_coef], x_data, 1, by_data,
                unseen_levels=unseen_levels,
            ),
            dtype=np.float64,
        )

    def _response_score_path(self, observed, weights, newdata, score, parameter=None,
                              unseen_levels="average", offset=None, extrapolation="warn"):
        """Stream scalar response scores through the shared native path scorer."""
        newdata, user = _replay_interactions(self, newdata, None, stage="raw", unseen_levels=unseen_levels)
        newdata = _replay_transforms(self, newdata, unseen_levels)
        newdata, user = _replay_interactions(self, newdata, user, stage="encoded", unseen_levels=unseen_levels)
        store = self._replay_store()
        if store is None or isinstance(self._family, (Multinomial, PropOdds)):
            raise NotImplementedError("scalar path replay is unavailable for this model")
        x, by = self._replay_slices(newdata, user)
        x, by = self._apply_extrapolation_policy(x, extrapolation, by)
        initial = _resolve_prediction_offset(self._offset, offset, self._family,
                                             len(observed), requires_explicit=True)
        return store.score_newdata_path(
            initial, self._selected_rust(), self._coefs_rust(), x,
            np.ascontiguousarray(observed, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            _family_rust_name(self._family), _family_params(self._family), score, parameter,
            by, unseen_levels=unseen_levels)

    def predict_over_mstop(self, mstops, *, newdata=None, type="link",
                           extrapolation="warn", output="auto",
                           unseen_levels=None):
        """Predictions at several path positions, in one pass over the basis.

        `predict(newdata=, mstop=k)` builds every learner's new-data basis, so
        calling it once per grid point rebuilds them each time. Measured on
        20,000 rows and eight `bbs` learners, six positions cost 52.1 ms
        against 8.7 for one -- six sets of B-spline evaluations to change
        nothing but how many path steps were summed.

        This walks the path once, carrying the coefficient sums forward from
        one position to the next, and evaluates each basis once for the grid.

        Parameters
        ----------
        mstops : sequence of int
            Path positions, ascending. The grid is walked once, so each
            position continues the previous one's sums.
        newdata : frame or dict, optional
            Where to predict. Defaults to the training data.
        type : {"link", "response"}
            As in :meth:`predict`.
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            As in :meth:`predict`.

        Returns
        -------
        A `(len(mstops), n)` array under ``output="numpy"``, otherwise a frame
        with one column per position named ``mstop_<k>``.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        extrapolation = _model_extrapolation(self, extrapolation)
        unseen_levels = _check_unseen_levels(unseen_levels)
        positions = [int(value) for value in mstops]
        if not positions:
            raise ValueError("mstops must name at least one position")
        if positions != sorted(positions):
            raise ValueError(
                f"mstops must be ascending, got {positions}: the grid is "
                f"walked once and each position continues the previous one"
            )
        if any(position < 0 for position in positions):
            raise ValueError(f"mstop must be non-negative, got {positions}")

        store = self._replay_store() if self._learners_spec is not None else None
        can_batch = (
            store is not None
            and not isinstance(self._family, Multinomial)
            and not any(isinstance(spec, Buser) for spec in self._learners_spec)
        )
        if not can_batch:
            # Kronecker replay, `Buser` (which needs its own new-data
            # matrices), and a restored model with no store take the ordinary
            # path once per position: correct, and no faster.
            rows = [np.asarray(self.predict(newdata=newdata, type=type,
                                            mstop=position,
                                            extrapolation=extrapolation,
                                            unseen_levels=unseen_levels),
                               dtype=np.float64)
                    for position in positions]
        else:
            # The per-position path above prepares new rows inside predict;
            # this one replays directly, so it encodes labels here.
            x_data, by_data = self._replay_slices(
                _replay_transforms(self, newdata, unseen_levels))
            x_data, by_data = self._apply_extrapolation_policy(
                x_data, extrapolation, by_data)
            etas = store.predict_newdata_grid(
                self._offset, self._selected_rust(), self._coefs_rust(),
                x_data, positions, by_data,
                unseen_levels=unseen_levels,
            )
            rows = [np.asarray(eta, dtype=np.float64) for eta in etas]
            if type == "response":
                rows = [np.asarray(_inverse_link(eta, self._family),
                                   dtype=np.float64)
                        for eta in rows]

        stacked = np.vstack(rows)
        if output == "numpy":
            return stacked
        from mboostrs._output import output_frame

        return output_frame(
            [{"row": index,
              **{f"mstop_{position}": float(stacked[grid, index])
                 for grid, position in enumerate(positions)}}
             for index in range(stacked.shape[1])],
            output,
        )

    def predict_over_grid(self, variable, grid, *, newdata=None, mstop=None,
                          type="link", extrapolation="warn", unseen_levels=None,
                          offset=None, output="auto"):
        """Predictions at each value of one variable, other columns held per row.

        For each grid value, ``variable`` is set to that value in every row of
        ``newdata``, and the model predicts those rows. Every other column keeps
        each row's own value. The result equals one :meth:`predict` call per grid
        value on the replaced frame. The difference is the work: a term that does
        not read ``variable`` is evaluated once, and a term that reads it is
        evaluated once per grid value, not once per boosting step.

        Parameters
        ----------
        variable : str
            Column whose value is replaced. It may be a covariate of a spline or
            linear term, or the ``by=`` variable of one.
        grid : sequence
            Values to set. Numeric values are used as given; a factor column
            takes its level labels.
        newdata : frame or dict, optional
            Rows to predict. Defaults to the training data.
        mstop : int, optional
            Truncate the boosting path at this iteration, as in :meth:`predict`.
        type : {"link", "response"}
            As in :meth:`predict`.
        extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
            Applied to the replaced values exactly as :meth:`predict` applies it:
            a grid value outside a spline term's training range warns, errors,
            is kept, or is clamped.
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            As in :meth:`predict`.
        offset : number or array-like, optional
            As in :meth:`predict`.
        output : str, default "auto"
            ``"numpy"`` returns the array described below. Other values return a
            frame with one column per grid value, named ``variable=value``.

        Returns
        -------
        ndarray
            Under ``output="numpy"``, a ``(len(grid), n)`` array: row ``k`` holds
            the prediction of every row with ``variable`` set to ``grid[k]``.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        extrapolation = _model_extrapolation(self, extrapolation)
        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link' or 'response', got {type!r}")
        unseen_levels = _check_unseen_levels(unseen_levels)
        grid_values = list(grid)
        if not grid_values:
            raise ValueError("grid must name at least one value")
        if newdata is None:
            newdata = self._replay_frame()
            if newdata is None:
                raise ValueError(
                    "predict_over_grid needs newdata: this model holds no training "
                    "frame to predict from. Pass newdata=."
                )
        from mboostrs._frame import column_names

        if variable not in set(column_names(newdata)):
            raise ValueError(
                f"variable {variable!r} is not a column of newdata; pass a frame "
                "that carries every predictor the model uses"
            )
        n = _frame_length(newdata, "newdata")
        plan = self._grid_plan(variable, grid_values, type)
        if plan is not None:
            stacked = self._grid_predictions_one_pass(
                plan, variable, grid_values, newdata, n, mstop=mstop,
                type=type, extrapolation=extrapolation,
                unseen_levels=unseen_levels, offset=offset)
        else:
            # The fallback for models whose grid path is not implemented
            # (Multinomial, Buser, transforms or interactions that read the
            # variable, multi-column terms, factor-valued grids): the replaced
            # frames are stacked row-wise and go through predict together.
            from mboostrs._counterfactual import predict_frames, replaced_frames, tiled_offset

            def predict_chunk(frame, copies):
                predicted = np.asarray(self.predict(
                    newdata=frame, type=type, mstop=mstop,
                    extrapolation=extrapolation, unseen_levels=unseen_levels,
                    offset=tiled_offset(offset, n, copies)), dtype=np.float64)
                if predicted.ndim != 1:
                    raise ValueError(
                        f"predict_over_grid returns one prediction per row, but predict "
                        f"returned shape {(n, *predicted.shape[1:])} for this model; "
                        "call predict once per value instead"
                    )
                return predicted

            columns = _grid_frame_columns(newdata)
            stacked = np.vstack(predict_frames(
                predict_chunk, replaced_frames(columns, variable, grid_values, n)))
        if output == "numpy":
            return stacked
        from mboostrs._output import output_frame

        labels = [_grid_label(variable, value) for value in grid_values]
        return output_frame(
            [{"row": index,
              **{label: float(stacked[k, index]) for k, label in enumerate(labels)}}
             for index in range(stacked.shape[1])],
            output,
        )

    def _grid_plan(self, variable, grid_values, type):
        """Which learners read ``variable``, when the one-pass grid path applies.

        Returns ``(x_learners, by_learners)``: the learner positions whose
        covariate or ``by=`` is ``variable``. Returns ``None`` when the grid path
        does not apply, and the caller predicts the replaced frames, stacked
        row-wise, through :meth:`predict`.
        The path needs a store and a single linear predictor, so it is refused
        for Multinomial, Buser (including interactions, which create Buser terms),
        PropOdds on the response scale, and a ``variable`` that a transform or
        interaction derives from. It is also refused when a term that reads
        ``variable`` is not a univariate spline, linear or random-effect term.
        """
        if self._learners_spec is None or isinstance(self._family, Multinomial):
            return None
        if any(isinstance(spec, Buser) for spec in self._learners_spec):
            return None
        if type == "response" and isinstance(self._family, PropOdds):
            return None
        recipe = getattr(self, "_transform_recipe", None)
        if recipe and (variable in recipe.source_names or variable in recipe.derived_names):
            return None
        for interaction in getattr(self, "_interactions", None) or ():
            if variable in (interaction.left.column, interaction.right.column):
                return None
        numeric = all(isinstance(value, (int, float, np.integer, np.floating))
                      and not isinstance(value, (bool, np.bool_))
                      for value in grid_values)
        if not numeric:
            return None
        x_learners, by_learners = [], []
        for position, spec in enumerate(self._learners_spec):
            spanned = [getattr(spec, "name1", None), getattr(spec, "name2", None)]
            spanned.extend(getattr(spec, "names", None) or ())
            if variable in spanned:
                return None
            reads_x = getattr(spec, "name", None) == variable
            reads_by = getattr(spec, "by", None) == variable
            if not (reads_x or reads_by):
                continue
            if not isinstance(spec, (Bbs, Bns, Bmono, Bconstrained, Bols, Brandom)):
                return None
            if reads_x:
                x_learners.append(position)
            if reads_by:
                by_learners.append(position)
        if self._replay_store() is None:
            return None
        return x_learners, by_learners

    def _grid_predictions_one_pass(self, plan, variable, grid_values, newdata, n, *,
                                   mstop, type, extrapolation, unseen_levels, offset):
        """The `(len(grid), n)` predictions of the one-pass grid path.

        The base frame is replayed once. The grid values replace ``variable`` in
        the covariate and ``by=`` inputs of the learners that read it. The
        extrapolation policy is applied to the base rows and to the grid values,
        which is what :meth:`predict` applies to each replaced frame.
        """
        x_learners, by_learners = plan
        store = self._replay_store()
        x_base, by_base = self._replay_slices(
            _replay_transforms(self, newdata, unseen_levels))
        grid = np.asarray(grid_values, dtype=np.float64)
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"grid values for {variable!r} must be finite")
        length = len(x_base)
        x_mixed = list(x_base)
        by_present = by_base is not None or bool(by_learners)
        by_mixed = list(by_base) if by_base is not None else [None] * length
        for position in x_learners:
            x_mixed[position] = grid.copy()
        for position in by_learners:
            by_mixed[position] = grid.copy()
        x_policy, by_policy = self._apply_extrapolation_policy(
            x_mixed, extrapolation, by_mixed if by_present else None)
        x_pass = [x_base[j] if j in x_learners else x_policy[j] for j in range(length)]
        by_pass = None
        if by_present:
            by_pass = [by_base[j] if j in by_learners else by_policy[j]
                       for j in range(length)]
        grid_x = [(j, np.ascontiguousarray(x_policy[j], dtype=np.float64))
                  for j in x_learners]
        grid_by = [(j, np.ascontiguousarray(by_policy[j], dtype=np.float64))
                   for j in by_learners]
        effective_offset = _resolve_prediction_offset(
            self._offset, offset, self._family, n, requires_explicit=True)
        rows = store.predict_over_grid(
            effective_offset, self._selected_rust(), self._coefs_rust(),
            x_pass, by_pass, len(grid_values), grid_x, grid_by,
            mstop, unseen_levels, n,
        )
        stacked = np.asarray(rows, dtype=np.float64).reshape(len(grid_values), n)
        if type == "response":
            stacked = np.asarray(_inverse_link(stacked, self._family), dtype=np.float64)
        return stacked

    def prediction_contributions(self, *, newdata=None, mstop=None, which=None, aggregate="sum",
                                 buser_newdata=None, unseen_levels=None,
                                 extrapolation="warn"):
        """Per-learner prediction contributions.

        Parameters
        ----------
        newdata : dict or DataFrame, optional
        mstop : int, optional
        which : int, str, or list thereof, optional
            Restrict to specific learner(s).
        aggregate : {"sum", "cumsum", "none"}
        buser_newdata : dict, optional
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            As in :meth:`predict`.
        extrapolation : {"warn", "error", "allow", "clip"}, default "warn"
            As in :meth:`predict`: the contributions of spline, ``bspatial``
            and ``%X%`` margin terms are checked against their training ranges
            before they are evaluated.

        Returns
        -------
        ndarray
            For ``aggregate="sum"``: 1-D array of length n.
            For ``aggregate="cumsum"``: 2-D array of shape (mstop, n).
            For ``aggregate="none"``: 2-D array of shape (mstop, n).
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        # predict() takes the contribution path for every aggregate except a
        # whole-model "sum", so it is the one place the range policy runs.
        return self.predict(
            newdata=newdata,
            mstop=mstop,
            type="link",
            which=which,
            aggregate=aggregate,
            buser_newdata=buser_newdata,
            unseen_levels=unseen_levels,
            extrapolation=extrapolation,
        )

    def _multinomial_predict(self, eta, type):
        """Reshape the flat (row-major) Multinomial predictor into a matrix.

        ``type="link"`` returns the ``(n, K-1)`` linear predictor; ``type="response"``
        returns the ``(n, K)`` softmax probability matrix whose last column is the
        reference class, matching R's ``predict(type="response")`` layout.
        """
        K = self._family.n_classes
        nr = K - 1
        eta = np.asarray(eta, dtype=np.float64)
        n = eta.size // nr
        f = eta.reshape(n, nr)  # row-major: row i is observation i's K-1 logits
        if type == "link":
            return f
        # response: clamp like R (pmin(abs(f), 36) * sign(f)), then softmax with
        # the reference class (implicit logit 0) as the last column.
        fc = np.clip(f, -36.0, 36.0)
        ef = np.exp(fc)
        denom = 1.0 + ef.sum(axis=1, keepdims=True)
        probs = np.empty((n, K), dtype=np.float64)
        probs[:, :nr] = ef / denom
        probs[:, nr] = 1.0 / denom[:, 0]
        return probs

    def _multinomial_newdata_eta(self, newdata, mstop):
        """Linear predictor (n_new, K-1) for new data, Multinomial family.

        Each selected base-learner contributes ``nu * (X1_new @ B_l^T)`` where the
        stored Kronecker coefficient is reshaped to ``B_l`` of shape (K-1, p1)
        (column-major class blocks) and ``X1_new`` is the learner's design at the
        new x: ``[1, x]`` / ``[x]`` for bols, and the raw B-spline basis (built with
        the *training* boundary knots) for bbs. The Multinomial offset is 0.
        """
        from mboostrs._rust import bbs_raw_design

        if self._learners_spec is None:
            raise ValueError("model was fitted without learner specs; cannot predict on new data")
        nr = self._family.n_classes - 1
        mmax = self.mstop if mstop is None else min(mstop, len(self._selected))

        design_cache: dict[int, np.ndarray] = {}

        def x1_for(li: int) -> np.ndarray:
            if li in design_cache:
                return design_cache[li]
            spec = self._learners_spec[li]
            x = np.ascontiguousarray(_extract_column(newdata, spec.name), dtype=np.float64)
            if isinstance(spec, Bbs):
                if getattr(spec, "kronecker_boundary", None) is not None:
                    bk = tuple(float(v) for v in spec.kronecker_boundary)
                else:
                    xtr = np.asarray(_extract_column(self._data, spec.name),
                                     dtype=np.float64)
                    bk = (float(xtr.min()), float(xtr.max()))
                flat, p = bbs_raw_design(x, bk, int(spec.knots), int(spec.degree))
                x1 = np.asarray(flat, dtype=np.float64).reshape(len(x), p)
            elif isinstance(spec, Bols):
                x1 = np.column_stack([np.ones(len(x)), x]) if getattr(spec, "intercept", True) else x.reshape(-1, 1)
            else:
                raise NotImplementedError(
                    f"Multinomial newdata prediction supports only bols/bbs learners; got {spec.__class__.__name__}"
                )
            design_cache[li] = x1
            return x1

        first = self._learners_spec[self._selected[0]] if mmax else self._learners_spec[0]
        n_new = len(_extract_column(newdata, first.name))
        eta = np.zeros((n_new, nr), dtype=np.float64)
        # The predictor is linear in the coefficients, so aggregate each
        # learner's coefs over its selected steps first, then do one matmul per
        # distinct learner: O(L·n·p·K + mstop·p·K) instead of O(mstop·n·p·K).
        agg: dict[int, np.ndarray] = {}
        for m in range(mmax):
            li = self._selected[m]
            beta = np.asarray(self._coefficients[m], dtype=np.float64)
            if li in agg:
                agg[li] = agg[li] + beta
            else:
                agg[li] = beta.copy()
        for li, beta in agg.items():
            x1 = x1_for(li)
            p1 = x1.shape[1]
            b = beta.reshape(nr, p1)  # column-major class blocks: row k = class k coefs
            eta += self._nu * (x1 @ b.T)
        return eta

    def _predict_training(self, k: int) -> np.ndarray:
        if self._learner_store is not None:
            selected_rust = self._selected_rust()
            coefs_rust = self._coefs_rust()
            return np.asarray(
                self._learner_store.predict(
                    self._offset,
                    selected_rust,
                    coefs_rust,
                    mstop=k,
                ),
                dtype=np.float64,
            )
        if self._learners_spec is None:
            return self._fitted.copy()
        # No live store (a deserialized model): rebuild one from the stored
        # specifications and replay the path through it.
        store = self._replay_store()
        if store is None:
            return self._fitted.copy()
        x_data, by_data = self._replay_slices(self._data)
        return np.asarray(
            store.predict_newdata_grouped(
                self._offset, self._selected_rust(), self._coefs_rust(), x_data, k, by_data
            ),
            dtype=np.float64,
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.start is not None and key.start != 0:
                raise ValueError("Only slicing from the start (model[:k]) is supported")
            k = key.stop
            if k is None:
                k = self.mstop
            if k < 0:
                k = max(0, self.mstop + k)
            k = min(k, self.mstop)
        elif isinstance(key, int):
            if key < 0:
                key = self.mstop + key
            k = key
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

        if k <= 0:
            raise ValueError("Cannot slice to 0 or negative iterations")

        new_selected = self._selected[:k]
        new_coefficients = self._coefficients[:k]
        new_risk = self._risk[: k + 1]

        if k >= self.mstop:
            new_fitted = self._fitted.copy()
        else:
            new_fitted = self._predict_training(k)

        _sliced = GAMBoostModel(
            formula=self._formula,
            family=self._family,
            control=self._control,
            response_name=self._response_name,
            learner_names=self._learner_names,
            learner_types=self._learner_types,
            fitted=new_fitted,
            risk=new_risk,
            selected=new_selected,
            coefficients=new_coefficients,
            offset=self._offset,
            nu=self._nu,
            data=self._data,
            weights=self._weights,
            learners_spec=self._learners_spec,
            rust_idx_to_formula_idx=self._rust_idx_to_formula_idx,
            formula_idx_to_rust_idx=self._formula_idx_to_rust_idx,
            learner_store=self._learner_store,
            execution_report=(self._execution_report.for_mstop(k) if self._execution_report is not None else None),
            # The nuisance parameter (NBinomial's dispersion, PropOdds'
            # thresholds) at step k: without it a sliced PropOdds model
            # returned link values for `type="response"`.
            family_sigma=(self._family_sigma_path[k] if k < len(self._family_sigma_path)
                          else self._family_sigma),
            family_sigma_path=self._family_sigma_path[: k + 1],
        )
        # A sliced model predicts through the same derived columns, including
        # the tensor designs of `%X%` terms, which are rebuilt from
        # `_interactions` on new data; without them a sliced interaction model
        # refused `predict(newdata=)`, and so did `cv_risk(mstop="cv")`, which
        # predicts through a sliced model.
        _sliced._transform_recipe = getattr(self, "_transform_recipe", None)
        _sliced._response_transform = getattr(self, "_response_transform", None)
        _sliced._interactions = getattr(self, "_interactions", None)
        # ... and refits from the same inputs. Without this, `confint`,
        # `cvrisk` and the conformal refits refuse on a sliced model -- the
        # ordinary way to reach the tuned fit -- while working on the model it
        # was sliced from. `continue_to` carries the same attributes.
        _sliced._refit_bundle = getattr(self, "_refit_bundle", None)
        _sliced._na_dropped = getattr(self, "_na_dropped", None)
        _sliced._comparison_identity = getattr(self, "_comparison_identity", None)
        _sliced._bin_decisions = getattr(self, "_bin_decisions", None)
        _sliced._input_weights = getattr(self, "_input_weights", None)
        _sliced._weight_policy = getattr(self, "_weight_policy", "upstream")
        # The baseline depends on the linear predictor at `mstop`; see the
        # matching block in GLMBoostModel.__getitem__.
        if _sliced._data is not None:
            _sliced._breslow_baseline = _breslow_baseline_of(_sliced)
        elif k >= self.mstop:
            _sliced._breslow_baseline = getattr(self, "_breslow_baseline", None)
        return _sliced


    def binning_report(self, output="auto"):
        """What binning each learner actually got, and what it cost.

        `bins=k` is a request. What a learner ends up with is the smaller of
        `k` and the column's distinct values, so `bins=50` on a column taking
        seven values bins nothing and says nothing -- this reports the
        effective count, the ties collapsed, and the learners `bins=` did not
        reach at all.

        One row per learner. The retained storage the fit used is on
        `execution_report().retained_learner_bytes`; the *saving* against
        exact storage is not a subtraction this can do -- it would need the
        exact fit -- so `check_binning()` reports both, on a subsample, along
        with how far the two fits diverged.
        """
        from mboostrs._output import output_frame

        decisions = {row["learner"]: row
                     for row in getattr(self, "_bin_decisions", []) or []}
        rows = []
        for spec in self._learners_spec:
            name = getattr(spec, "name", None)
            if name is None:
                continue
            requested = getattr(spec, "bins", None)
            distinct = None
            if self._data is not None:
                try:
                    column = np.asarray(self._data_array_for(name),
                                        dtype=np.float64)
                    distinct = int(np.unique(column[np.isfinite(column)]).size)
                except (KeyError, ValueError, TypeError):
                    distinct = None
            effective = (None if requested is None
                         else requested if distinct is None
                         else min(int(requested), distinct))
            decision = decisions.get(name, {})
            rows.append({
                "learner": f"{type(spec).__name__.lower()}({name})",
                "requested_bins": requested,
                "distinct_values": distinct,
                "effective_bins": effective,
                "ties_collapsed": (None if effective is None or distinct is None
                                   else max(distinct - effective, 0)),
                "binned": requested is not None,
                "auto_reason": decision.get("reason"),
            })
        return output_frame(rows, output)

    def check_binning(self, n=2000, seed=42, mstop=None):
        """Refit one subsample with binning on and off.

        Returns the maximum fitted-value discrepancy relative to the exact
        refit's residual standard deviation, plus the first selection-path
        divergence. This answers whether the chosen bin count is adequate for
        this model rather than relying on a generic recommendation.

        The comparison applies to this model and the selected subsample.
        Agreement supports the bin count for this fit; other data or model
        specifications can produce a different discrepancy.
        """
        import copy

        from mboostrs.stabsel import _take_rows

        binned_terms = [
            learner for learner in self._learners_spec
            if getattr(learner, "bins", None) is not None
        ]
        if not binned_terms:
            raise ValueError("the model has no approximately binned learner to check")
        total = len(self._fitted)
        sample_n = min(int(n), total)
        if sample_n < 2:
            raise ValueError("n must select at least two observations")
        rows = np.random.default_rng(seed).choice(total, size=sample_n, replace=False)
        rows.sort()
        sample = _take_rows(self._data, rows)
        sample_weights = None if self._weights is None else np.asarray(self._weights)[rows]
        budget = self.mstop if mstop is None else min(int(mstop), self.mstop)
        control = as_control(self._control)._replace(mstop=budget, nu=self._nu, trace=False)
        compact_learners = [copy.copy(learner) for learner in self._learners_spec]
        exact_learners = [copy.copy(learner) for learner in self._learners_spec]
        for learner in exact_learners:
            if hasattr(learner, "bins"):
                learner.bins = None
        compact = gamboost(
            self._formula, data=sample, family=self._family, control=control,
            weights=sample_weights, learners=compact_learners,
        )
        exact = gamboost(
            self._formula, data=sample, family=self._family, control=control,
            weights=sample_weights, learners=exact_learners,
        )
        fitted_difference = np.asarray(compact.fitted_) - np.asarray(exact.fitted_)
        response = _extract_response(sample, self._response_name, self._family)
        response = np.asarray(response, dtype=np.float64)[:sample_n]
        residual_sd = float(np.std(response - np.asarray(exact.fitted_), ddof=1))
        if residual_sd <= np.finfo(float).eps:
            residual_sd = float(np.std(response, ddof=1))
        first_divergence = next(
            (index + 1 for index, (left, right) in enumerate(
             zip(compact._selected, exact._selected, strict=True)) if left != right),
            None,
        )
        maximum = float(np.max(np.abs(fitted_difference)))
        compact_bytes = compact.execution_report().retained_learner_bytes
        exact_bytes = exact.execution_report().retained_learner_bytes
        return {
            "n": sample_n,
            "mstop": budget,
            "max_abs_fitted_difference": maximum,
            "residual_sd": residual_sd,
            "relative_to_residual_sd": maximum / residual_sd if residual_sd > 0 else float("inf"),
            "first_selection_divergence": first_divergence,
            "selection_sequence_identical": first_divergence is None,
            "binned_retained_learner_bytes": compact_bytes,
            "exact_retained_learner_bytes": exact_bytes,
            "bytes_saved": exact_bytes - compact_bytes,
            "scope": (
                f"measured on {sample_n} of {total} rows at mstop={budget}; "
                f"evidence about this fit, not a property of the bin count"
            ),
        }

    def cvrisk(self, folds=None, type="bootstrap", B=25, prob=0.5, seed=42, parallel=True,
               at_mstop=None, *, preprocessing="fixed", _inner=False, _extend=None,
               _path_cache=None):
        """Cross-validate the fitted model's risk path.

        `at_mstop=k` judges the path truncated at `k` steps: the returned
        curve, fold rows and grid end there, and the boundary warning is
        skipped because the step count is the caller's choice, not a
        search's output -- fixing the step answers the question the
        warning asks. Pass it when comparing designs or specifications at
        a prespecified step count; leave it unset when tuning.

        ``preprocessing="fixed"`` reuses the full-data transformations and
        spline knots, matching mboost. ``"refit"`` learns transformations,
        factor encodings and spline boundaries on each fold's training rows.
        Held-out rows use that fitted recipe, with zero effects for unseen
        group levels. Refit folds run sequentially through the native fit,
        prediction and family-score APIs. The result records the mode.

        `_extend` (internal, used by `tune_mstop`) is a mapping of
        ``max_mstop``, ``patience``, ``boundary_share`` and
        ``max_extensions``: the grid starts at the model's ``mstop`` and grows
        by resuming each fold's path while the held-out risk is still falling.
        The call then returns ``(result, grid_ends, at_boundary)``.

        Parameters
        ----------
        folds : int, FoldScheme or array_like of shape (n, B), optional
            An integer is a k-fold split with that many folds (at least 2),
            drawn with ``seed``. A :class:`FoldScheme` or an ``(n, B)`` matrix
            of in-bag weights is used as given; a scheme also carries its
            judge mask. ``None`` generates ``B`` folds of resampling type
            ``type``.
        type : {"bootstrap", "kfold", "subsampling"}, default "bootstrap"
            Resampling used when ``folds`` is ``None``.
        B : int, default 25
            Number of generated folds when ``folds`` is ``None``.
        prob : float, default 0.5
            Share of observations drawn per fold for ``"subsampling"``.
        seed : int, default 42
            Random seed for generated folds.
        parallel : bool, default True
            Run folds in parallel in the Rust engine.
        at_mstop : int, optional
            Judge the path truncated at this many steps; the curve and grid end
            there and the boundary warning is skipped.
        preprocessing : {"fixed", "refit"}, default "fixed"
            Reuse the full-data recipe, or learn it within each training fold.

        Returns
        -------
        CvRiskResult
            Out-of-fold risk matrix of shape ``(n_folds, n_grid)``, the
            aggregate curve over the grid, and ``optimal_mstop``.

        Raises
        ------
        ValueError
            If ``folds`` is an integer below 2, the fold matrix does not match
            the number of observations, or the model keeps no data to refit
            from.
        RuntimeError
            If the Rust engine is not available.
        """
        from mboostrs._cv_preprocessing import check_preprocessing, refitted_cv_path

        preprocessing = check_preprocessing(preprocessing)
        if preprocessing == "refit" and _extend is not None:
            raise NotImplementedError(
                "Automatic CV grid extension is not yet supported with "
                "preprocessing='refit'; fit with a larger mstop or use "
                "preprocessing='fixed'"
            )
        if not _HAS_RUST_ENGINE:
            raise RuntimeError("Rust engine not available")
        is_multinomial = isinstance(self._family, Multinomial)
        n_resp = (self._family.n_classes - 1) if is_multinomial else 1
        n_obs = (
            len(self._weights)
            if self._weights is not None
            else (self._fitted.shape[0] // n_resp if is_multinomial else len(self._fitted))
        )
        if folds is None:
            w = self._weights if self._weights is not None else np.ones(n_obs)
            folds_matrix = _cv_weights(
                n_obs, type, B, prob, seed, np.asarray(w, dtype=np.float64)
            )
            fold_label = f"{B}-fold {type}"
        elif isinstance(folds, (int, np.integer)) and not isinstance(folds, bool):
            # A fold count is a k-fold split on `seed`, built as cv_weights and
            # the functional cvrisk build it: the same rows in each fold.
            count = int(folds)
            if count < 2:
                raise ValueError(f"folds must be at least 2 for k-fold cross-validation; got {count}")
            w = self._weights if self._weights is not None else np.ones(n_obs)
            folds_matrix = _cv_weights(
                n_obs, "kfold", count, prob, seed, np.asarray(w, dtype=np.float64)
            )
            fold_label = f"{count}-fold kfold"
        else:
            folds_matrix = _as_fold_matrix(folds, n_obs, self._weight_policy, self._weights)
            fold_label = fold_label_for(folds, folds_matrix)
        judge_mask = ((folds.judge if folds.judge is not None else folds.held_out)
                      if isinstance(folds, FoldScheme) else None)
        if self._data is None:
            from mboostrs._refit import refit_available

            _, reason = refit_available(self)
            raise ValueError(f"cvrisk is unavailable: {reason}")

        y = (
            _multinomial_response(self)
            if is_multinomial
            else _extract_response(self._data, self._response_name, self._family)
        )
        w = self._weights if self._weights is not None else np.ones(n_obs)
        if isinstance(folds, FoldScheme) and folds.base_weights is not None:
            w = np.asarray(folds.base_weights, dtype=np.float64)
        mstop_val, nu, risk_str, trace_bool = (
            self._control.mstop, self._control.nu, self._control.risk, self._control.trace)


        multinomial_n_classes = self._family.n_classes if is_multinomial else None
        binding = _cvrisk_gamboost_specs if _extend is None else _cvrisk_gamboost_specs_extending
        extend_kwargs = {} if _extend is None else dict(_extend)
        if preprocessing == "refit":
            out = refitted_cv_path(self, folds_matrix, judge_mask, w, cache=_path_cache)
        else:
            out = call_fit(binding,
                self._fit_learner_specs(
                    self._data, vector_offset=_is_vector_offset(self._offset)),
                y,
                w,
                mstop_val,
                nu,
                _family_rust_name(self._family),
                _family_params(self._family),
                folds_matrix,
                judge_mask,
                # A per-row offset (log exposure) is part of every fold's linear
                # predictor, as it is of the full fit; a fitted scalar offset is
                # re-estimated on each fold's in-bag rows, as R's cvrisk does.
                self._offset if _is_vector_offset(self._offset) else None,
                risk_str,
                trace_bool,
                parallel,
                fold_label,
                multinomial_n_classes,
                _fit_count=folds_matrix.shape[1],
                **step_kwargs(self._control),
                **extend_kwargs,
            )
        if _extend is None:
            risk_rows, aggregate, grid, optimum = out
        else:
            (risk_rows, aggregate, grid, optimum), grid_ends, at_boundary = out
        result = CvRiskResult(
            np.asarray(risk_rows, dtype=np.float64),
            np.asarray(aggregate, dtype=np.float64),
            grid,
            optimum,
            fold_label,
            provenance_from_model(
                self,
                "cvrisk",
                folds=fold_label,
                n_fits=int(folds_matrix.shape[1]),
            ),
            fold_fingerprint=fold_fingerprint(
                folds_matrix, judge=judge_mask, weights=w, response=y
            ),
            data_fingerprint=data_fingerprint(y, w),
            comparison_identity=getattr(self, "_comparison_identity", None),
            held_out_fraction=held_out_fraction(folds_matrix, judge_mask),
            folds=folds if isinstance(folds, FoldScheme) else None,
            design=_fold_design(folds, folds_matrix),
            preprocessing=preprocessing,
            warn_boundary=_extend is None,
        )
        if at_mstop is None:
            if not _inner:
                warn_if_at_boundary(result)
                warn_if_one_fold_dominates(result)
        else:
            from mboostrs.cvrisk import truncated_at_mstop

            result = truncated_at_mstop(result, at_mstop)
        if _extend is not None:
            return result, tuple(int(end) for end in grid_ends), bool(at_boundary)
        return result
    def coef(self, which=None, off2int=False, aggregate="sum"):
        """Extract coefficients per base-learner.

        which: base-learner name(s) or index/indices. None = all used.
        off2int: ignored with warning for gamboost (matching R's behavior;
            R warns "Arguments off2int unknown" for gamboost).
        aggregate: "sum" (default) sums per-step coefs; "none" returns raw per-step lists.
        Returns dict mapping learner name -> coefficient value or array.
        The offset is always included as result["(offset)"].
        """
        if off2int:
            warnings.warn(
                "Argument off2int is unknown for gamboost (ignored), "
                "matching R's behavior. Use extract('offset') to get the offset.",
                UserWarning,
                stacklevel=2,
            )
        names = self._learner_names
        if which is not None:
            if isinstance(which, str):
                which = [which]
            elif isinstance(which, int):
                which = [names[which]]
            indices = []
            for w in which:
                for index in self._which_indices(w):
                    if index not in indices:
                        indices.append(index)
        else:
            indices = list(range(len(names)))

        if aggregate == "sum":
            # Btree coefficient vectors encode a fitted tree's structure, and
            # two trees do not add: summing them is meaningless and the
            # encodings need not even share a length. Tree learners get their
            # per-step decoded records instead; every other learner is linear
            # in its coefficients and sums as before.
            tree_indices = self._btree_indices()
            acc = {idx: None for idx in indices if idx not in tree_indices}
            tree_steps = {idx: [] for idx in indices if idx in tree_indices}
            for step, (sel, coef) in enumerate(zip(self._selected,
                                                   self._coefficients,
                                                   strict=False)):
                if sel in acc:
                    c = self._nu * coef
                    if acc[sel] is None:
                        acc[sel] = c.copy()
                    else:
                        acc[sel] = acc[sel] + c
                elif sel in tree_steps:
                    tree_steps[sel].append(
                        {"step": step, "nodes": _decode_btree_nodes(coef)})
            # Two learners can read one column ("bols(x) + bbs(x)" both name
            # "x"), and a dict keyed by the raw name kept only one of them.
            # Every entry is keyed by `base_learner_names()`, which is the raw
            # name unless two learners share it ("x (bols)", "x (bbs)").
            distinct = self.base_learner_names()
            result = {}
            for idx in indices:
                if idx in tree_steps:
                    result[distinct[idx]] = tree_steps[idx]
                    continue
                name = distinct[idx]
                v = acc[idx]
                centering = self._bols_intercept_centering(idx)
                if centering is not None:
                    # A centered bols fits b * (x - c). mboost's bols(x) fits
                    # [1, x] and reports both coefficients, so report the
                    # intercept -b * c the centered column implies.
                    slope = 0.0 if v is None else float(np.ravel(v)[0])
                    result[name] = np.array([-slope * centering, slope])
                elif v is None:
                    result[name] = 0.0
                else:
                    result[name] = v.item() if v.ndim == 0 or v.size == 1 else v
        else:
            tree_indices = self._btree_indices()
            distinct = self.base_learner_names()
            result = {distinct[idx]: [] for idx in indices}
            for sel, coef in zip(self._selected, self._coefficients,
                                 strict=False):
                if sel in indices:
                    # A tree learner's per-step entry is its decoded node
                    # records; the raw encoding is an implementation detail.
                    key = distinct[sel]
                    result[key].append(
                        _decode_btree_nodes(coef) if sel in tree_indices
                        else self._nu * coef)
        result["(offset)"] = self._offset
        return result

    def _bols_intercept_centering(self, index):
        """The centering value of a `bols(x)` with an intercept fitted in the
        centered single-column form, or None for any other learner (the
        block form `[1, x]` already reports its intercept)."""
        from mboostrs.learners import Bols

        specs = self._learners_spec
        if specs is None or len(specs) != len(self._learner_names):
            return None
        spec = specs[index]
        if not isinstance(spec, Bols) or not getattr(spec, "intercept", True):
            return None
        store = self._replay_store()
        rust_index = self._formula_to_rust(index)
        if store is None or not hasattr(store, "scalar_centering"):
            return None
        sizes = {np.size(coef) for sel, coef in zip(self._selected, self._coefficients,
                                                    strict=False) if sel == index}
        if sizes and sizes != {1}:
            return None
        return store.scalar_centering(int(rust_index))

    def _btree_indices(self):
        """Indices of fitted btree learners, empty when specs are absent."""
        if (self._learners_spec is None
                or len(self._learners_spec) != len(self._learner_names)):
            return set()
        return {index for index, spec in enumerate(self._learners_spec)
                if isinstance(spec, Btree)}

    def btree_splits(self):
        """The cutpoints the fitted btree learners actually used.

        One entry per btree learner that was selected at least once. Numeric
        splits name the variable and the cutpoint with the number of boosting
        steps that used it; categorical splits name the variable and the
        level sets. The variables follow the tree's own input order, so index
        ``i`` in a step's decoded nodes is ``variables[i]``. A tree's splits
        are empirical cutpoints learned by the fit. Use them to propose
        hinge, threshold or other simpler candidate terms when examining a tree.
        """
        result = {}
        tree_indices = self._btree_indices()
        if not tree_indices:
            return result
        for index in sorted(tree_indices):
            spec = self._learners_spec[index]
            variables = list(getattr(spec, "names", ()) or ())
            numeric: dict[tuple[int, float], int] = {}
            level_sets: dict[int, int] = {}
            for sel, coef in zip(self._selected, self._coefficients,
                                 strict=False):
                if sel != index:
                    continue
                for node in _decode_btree_nodes(coef):
                    if node["kind"] == "split":
                        key = (node["variable"], node["cutpoint"])
                        numeric[key] = numeric.get(key, 0) + 1
                    elif node["kind"] == "level_split":
                        level_sets[node["variable"]] = (
                            level_sets.get(node["variable"], 0) + 1)
            if not numeric and not level_sets:
                continue
            result[self.base_learner_names()[index]] = {
                "variables": variables,
                "splits": [
                    {"variable": (variables[var] if var < len(variables)
                                  else f"column_{var}"),
                     "cutpoint": point, "steps": steps}
                    for (var, point), steps in sorted(numeric.items())],
                "level_split_variables": [
                    (variables[var] if var < len(variables) else f"column_{var}")
                    for var in sorted(level_sets)],
            }
        return result

    def btree_partition_table(self, term=None):
        """The fitted btree partition as a Polars frame, one row per region.

        The companion to :meth:`btree_splits`: the cutpoints there name the
        boundaries, and here they become the ``<column>_lower`` and
        ``<column>_upper`` bounds of each region with the term's fitted
        ``value`` at the region centre.

        Parameters
        ----------
        term : str or int, optional
            Which btree term to read; the only one when there is exactly one.

        Returns
        -------
        polars.DataFrame
        """
        return btree_partition_table(self, term=term)

    def plot_btree_partition(self, term=None, grid=60, data=None, at=None):
        """Plot a btree term's fitted partition.

        A heatmap over the two columns the term splits on, or a step plot for a
        one-column term; see :func:`mboostrs.plot.plot_btree_partition`.

        Returns
        -------
        alt.Chart
        """
        from mboostrs.plot import plot_btree_partition

        return plot_btree_partition(self, term=term, grid=grid, data=data,
                                    at=at)

    def residuals(self, type="response", newdata=None, mstop=None):
        """Residuals of the named kind, or a refusal saying why there are none.

        Parameters
        ----------
        type : {"response", "pearson", "working", "deviance", "martingale"}
            ``"response"`` is ``y - mu`` with ``mu`` on the response's own
            scale. ``"pearson"`` is ``(y - mu) / sqrt(V(mu))``, available only
            for families that declare a variance function. ``"working"`` is the
            fitted family's negative gradient -- what the next boosting step
            regresses on -- and is available for every family. ``"deviance"``
            is the signed square root of each observation's deviance
            contribution. ``"martingale"`` is for `CoxPH`.

        If a residual kind is undefined for the family, the method raises
        `NotImplementedError` and names the available kinds. Use
        `residual_kinds()` to inspect them.

        See :meth:`residual_kinds` for what this family offers, and
        :func:`mboostrs.quantile_residuals` for a kind that is comparable
        across families because it is uniform under a correct fit.
        """
        from mboostrs._residuals import residuals as _residuals

        return _residuals(self, kind=type, newdata=newdata, mstop=mstop)

    def residual_kinds(self):
        """The residual kinds defined for this model's family."""
        from mboostrs._residuals import available_kinds

        return available_kinds(self)

    def extract(self, what, which=None):
        """Extract model components, matching R's extract.mboost/glmboost.

        what: one of "design", "penalty", "lambda", "df", "coefficients",
              "residuals", "offset", "weights".
              Aliases accepted: "coef", "coefs", "coefficient" -> "coefficients";
              "x", "model_matrix" -> "design"; "w" -> "weights";
              "penalties" -> "penalty"; "lambdas" -> "lambda"; "dfs" -> "df".
        which: base-learner name(s) or index. None = all used learners.
        Returns dict mapping learner name -> extracted value.

        "design" is the design on every training row, which is mboost's
        ``extract(..., expand = TRUE)``; mboost's default reports a factor
        learner's design on its distinct rows. A ``bols(x)`` reports the
        block ``[1, x]`` and "penalty" reports K without lambda, as mboost
        does.
        """
        from mboostrs._output import _normalize_extract_what

        what = _normalize_extract_what(what)
        valid = ("design", "penalty", "lambda", "df", "coefficients", "residuals", "offset", "weights", "selected")
        if what not in valid:
            raise ValueError(f"what must be one of {valid}, got {what!r}")

        if what == "coefficients":
            return self.coef(which=which)
        if what == "residuals":
            return self.residuals()
        if what == "offset":
            return self._offset
        if what == "weights":
            return self._weights.copy() if self._weights is not None else np.ones(len(self._fitted))
        if what == "selected":
            return self.selected

        if self._learner_store is None:
            raise ValueError("extract() requires the learner store (model built with default gamboost())")

        names = self._learner_store.learner_names()
        # The store names a `%X%` term by its derived column; the caller and
        # the result use the term as written.
        labels = self._interaction_labels()
        derived_of = {label: derived for derived, label in labels.items()}
        # Results are keyed like `partial_effects()` and `coef()`: two learners
        # on one column (`bbs(x) + bbs(x, by=z)`) would otherwise share the
        # store's column name and the second would overwrite the first.
        distinct = self.base_learner_names()
        if which is not None:
            if isinstance(which, str):
                which = [which]
            elif isinstance(which, int):
                which = [which]
            indices = []
            for w in which:
                if isinstance(w, int):
                    indices.append(w)
                elif isinstance(w, str):
                    if distinct.count(w) == 1 and w not in names:
                        # A disambiguated label, "hour (bbs, by=z)".
                        indices.append(self._formula_to_rust(distinct.index(w)))
                        continue
                    try:
                        indices.append(names.index(derived_of.get(w, w)))
                    except ValueError:
                        raise ValueError(f"Unknown base-learner: {w!r}") from None
                else:
                    raise TypeError(f"Invalid which element: {w!r}")
        else:
            indices = list(set(self._selected))
            indices = list({self._formula_to_rust(s) for s in indices})

        # Map each rust-store learner index back to its formula spec so the
        # The "df" branch reports `tr(2H - HᵀH)`, which is the quantity R's
        # `extract(what="df")` returns and the one its `df2lambda` inverts.
        # Computing it reproduces R in both directions: a learner asked for
        # `df=4` comes back 4.00000, and one given `lambda=5` comes back
        # 9.26474 against R's 9.264741 on the same 140 rows.
        #
        # It used to report the *spec's* df instead. That is right when the
        # caller asked for a df and wrong when they asked for a lambda, since
        # `df` keeps its default of 4: `bbs(x, lambda_value=5)` reported 4.0
        # for a smoother whose df is 9.26. The nominal value is still the
        # fallback for learners with no closed-form hat matrix (bmono, btree,
        # Kronecker), where neither trace exists.
        rust_to_formula = {r: self._rust_to_formula(r) for r in range(len(self._rust_idx_to_formula_idx))}

        result = {}
        for idx in indices:
            name = labels.get(names[idx], names[idx])
            fidx_label = rust_to_formula.get(idx)
            if (fidx_label is not None and 0 <= fidx_label < len(distinct)
                    and name not in labels.values()):
                name = distinct[fidx_label]
            # A centered bols(x) is stored as one column x - c; mboost's
            # bols(x) is the block [1, x] with an unpenalized ridge, and
            # extract() reports that form.
            fidx = rust_to_formula.get(idx)
            centering = (self._bols_intercept_centering(fidx)
                         if fidx is not None else None)
            if what == "design":
                mat = self._learner_store.extract_design(idx)
                mat = np.array(mat) if mat is not None else None
                if mat is not None and centering is not None:
                    mat = np.column_stack([np.ones(len(mat)), mat[:, 0] + centering])
                result[name] = mat
            elif what == "penalty":
                # The store holds lambda * K; mboost reports K, and the
                # identity for an unpenalized (lambda = 0) ridge.
                mat = self._learner_store.extract_penalty(idx)
                mat = np.array(mat) if mat is not None else None
                lam = self._learner_store.extract_lambda(idx)
                if mat is not None and lam is not None:
                    if centering is not None:
                        mat = np.eye(2)
                    elif lam > 0:
                        mat = mat / lam
                    else:
                        mat = np.eye(mat.shape[0])
                result[name] = mat
            elif what == "lambda":
                result[name] = self._learner_store.extract_lambda(idx)
            elif what == "df":
                value = self._learner_store.extract_df_mboost(idx)
                if value is None:
                    fidx = rust_to_formula.get(idx)
                    if fidx is not None and self._learners_spec is not None:
                        nominal = getattr(self._learners_spec[fidx], "df",
                                          None)
                        value = (float(nominal) if nominal is not None
                                 else self._learner_store.extract_df(idx))
                    else:
                        value = self._learner_store.extract_df(idx)
                result[name] = value
        return result

    def _multinomial_nll_path(self):
        """Full negative log-likelihood on the training rows after each step."""
        n_resp = self._family.n_classes - 1
        y = np.ascontiguousarray(_multinomial_response(self), dtype=np.float64)
        w = (np.ones(len(y) // n_resp) if self._weights is None
             else np.asarray(self._weights, dtype=np.float64))
        store = self._learner_store if self._learner_store is not None else self._replay_store()
        if store is None:
            return np.asarray(self._risk, dtype=np.float64)
        return np.asarray(store.staged_held_out_risk(
            self._offset, self._selected_rust(), self._coefs_rust(),
            _family_rust_name(self._family), _family_params(self._family),
            y.tolist(), np.repeat(w, n_resp).tolist()), dtype=np.float64)

    def varimp(self, percent=True):
        """Variable importance based on in-bag risk reduction per base-learner.

        Returns a dict with keys:
          reduction: dict of learner_name -> risk reduction
          selfreq: dict of learner_name -> selection frequency
          variable_names: dict of learner_name -> variable name
          split_columns: dict of learner_name -> columns actually used in splits
        With ``percent=True`` (the default) each reduction is divided by the
        sum of their absolute values, so the reductions are shares summing
        to 1. ``percent=False`` returns the in-bag risk reduction per
        observation, which is what mboost's ``varimp()`` returns.

        Each base-learner has its own entry, as in mboost. Learners that
        share a column (`bbs(x) + bmono(x)`) are keyed by
        :meth:`base_learner_names` (``"x (bbs)"``, ``"x (bmono)"``); sum the
        entries whose ``variable_names`` agree for a variable's total.

        ``variable_names`` lists every candidate input column a learner may
        read -- for a joint ``btree`` that is the whole column list, not the
        single column the label is named for. ``split_columns`` narrows that
        to the columns the fitted trees actually split on (from
        :meth:`btree_splits`); for non-tree learners the two agree. Counting
        inputs a model retains should use ``split_columns``: a joint tree
        that never split on a column does not use it.
        """
        risk = self._risk
        if len(risk) < 2:
            return {"reduction": {}, "selfreq": {}, "variable_names": {}, "split_columns": {}}
        n = len(self._fitted)
        if isinstance(self._family, Multinomial):
            # The in-bag risk path is mboost's loss, which leaves the reference
            # class out; a step that improves the full likelihood can raise it,
            # which gave true predictors negative shares. Attribute each
            # step's change in the full negative log-likelihood instead.
            risk = self._multinomial_nll_path()
            n = n // (self._family.n_classes - 1)

        from mboostrs._attribution import attribute_by_name, risk_changes

        # One entry per base-learner, as mboost's varimp reports. Learners
        # that share a column (`bbs(x) + bols(x)`) are keyed "x (bbs)",
        # "x (bols)" (`base_learner_names()`); `variable_names` maps each key
        # to its columns for aggregating by variable.
        learner_names = self.base_learner_names()
        changes = risk_changes(risk, n)
        reduction, selfreq = attribute_by_name(
            changes, [learner_names[j] for j in self._selected], learner_names,
            percent=percent)

        variable_names = {}
        split_columns = {}
        splits = self.btree_splits()
        for index, name in enumerate(learner_names):
            variable_names[name] = ",".join(self._learner_columns(index))
            entry = splits.get(name)
            if entry is None:
                # Not a tree learner, or a tree with no recorded splits:
                # what it reads is what it uses.
                split_columns[name] = ",".join(self._learner_columns(index))
            else:
                used = {s["variable"] for s in entry["splits"]}
                used.update(entry["level_split_variables"])
                split_columns[name] = ",".join(sorted(used))

        return {"reduction": reduction, "selfreq": selfreq, "variable_names": variable_names,
                "split_columns": split_columns}

    def learner_index(self, name, kind=None):
        """Index of a base learner, disambiguating learners that share a name.

        ``bols(x) + bbs(x, df=1, center=True)`` puts two learners on column
        ``x``, so the name alone does not identify one. `kind` selects between
        them by learner type — ``"bols"``, ``"bbs"``, ``"btree"`` and so on.

        Parameters
        ----------
        name : str
            Column the learner is built on. The disambiguated form
            :meth:`base_learner_names` returns for a shared name — ``"x (bols)"``
            — is also accepted, and carries the kind with it.
        kind : str, optional
            Learner type, matched case-insensitively against the class name.
            Required when more than one learner shares the name.

        Returns
        -------
        int
            Index usable as ``partial_effects(which=...)``.
        """
        # `base_learner_names()` hands back "x (bols)" for a name two learners
        # share, and that name has to lead somewhere. Split it only when the
        # whole string is not itself a learner name — `I(x^2)` and a factor
        # `by=` term both put parentheses in a legitimate name.
        name = self._learner_alias(name)
        if name not in self._learner_names:
            # A label `base_learner_names()` handed back -- "x (bbs)", or
            # "x (bbs, by=z)" when kind alone does not separate the learners.
            labels = self.base_learner_names()
            if labels.count(name) == 1:
                index = labels.index(name)
                actual_kind = type(self._learners_spec[index]).__name__.lower()
                if kind is not None and kind.lower() != actual_kind:
                    raise ValueError(
                        f"learner {name!r} has kind {actual_kind!r}; requested kind={kind!r}")
                return index
        if name not in self._learner_names:
            suffix = re.fullmatch(r"(?P<name>.+?) \((?P<kind>[a-z]+)\)", name)
            if suffix and suffix.group("name") in self._learner_names:
                name = suffix.group("name")
                kind = kind or suffix.group("kind")
        matches = [
            index for index, learner_name in enumerate(self._learner_names)
            if learner_name == name
        ]
        if not matches:
            raise ValueError(
                f"no base learner named {name!r}; the model has "
                f"{sorted(set(self._learner_names))}"
            )
        if kind is not None:
            matches = [
                index for index in matches
                if type(self._learners_spec[index]).__name__.lower() == kind.lower()
            ]
            if not matches:
                kinds = sorted({
                    type(self._learners_spec[i]).__name__.lower()
                    for i, n in enumerate(self._learner_names) if n == name
                })
                raise ValueError(
                    f"no {kind!r} learner named {name!r}; the learners on that "
                    f"column are {kinds}"
                )
        if len(matches) > 1:
            kinds = sorted({type(self._learners_spec[i]).__name__.lower()
                            for i in matches})
            raise ValueError(
                f"{len(matches)} learners are named {name!r} ({kinds}); pass "
                "kind= to choose one"
            )
        return matches[0]

    def partial_effects(self, which=None, n_points=None, newdata=None,
                        unseen_levels=None, scale="link", extrapolation="allow",
                        *, _prepared=False, _buser=None, mstop=None):
        """Compute per-learner partial effects (contribution to the additive predictor).

        Parameters
        ----------
        which : str, int, or list of those, optional
            Base-learner(s) to compute effects for. If None, all selected
            base-learners.

            A name two learners share (the decomposition idiom
            ``bols(x) + bbs(x, df=1, center=True)``) selects both, keyed
            ``"x (bols)"`` and ``"x (bbs)"``; that form, or the learner's index,
            selects one.
        n_points : int, optional
            For smooth (bbs/bmono/bspatial) effects, evaluate on an equispaced
            grid of this many points. If None, evaluate on training/newdata.
        newdata : dict or DataFrame, optional
            Data to evaluate effects on. If None, uses training data.
        unseen_levels : {"error", "average", "zero_effect"}, optional
            ``None`` (default) uses the model's policy: ``"error"``, except
            for a model returned by ``mb.fit``, which predicts an unseen
            level as the average group.
            As in :meth:`predict`.
        extrapolation : {"allow", "warn", "error", "clip"}, default "allow"
            As in :meth:`predict`, for `newdata` outside a spline learner's
            training range. The default stays silent, because an effect curve
            is often drawn on purpose beyond the data; ``predict(which=...)``
            passes its own policy (default ``"warn"``). The ``%X%`` margins
            are not checked here, because this method takes the ``%X%`` designs
            it is given rather than building them from ``newdata``.
        scale : {"link", "log_odds"}, default "link"
            ``"link"`` reports effects on the family's link scale, as mboost
            does: for ``Binomial`` that is half the log-odds (the response is
            ``1 / (1 + exp(-2 * eta))``). ``"log_odds"`` multiplies a
            ``Binomial`` effect by 2, the scale EBM, XGBoost, mgcv and other
            libraries report and the one whose exponential is an odds ratio.
        mstop : int, optional
            Truncate the boosting path at this iteration: only steps before
            it contribute to each effect, so the result equals the effect of
            a model refit with ``mstop`` steps. If None, the full fitted
            path contributes.

        Returns
        -------
        dict
            Mapping base-learner name -> (x_values, effect_values) pairs.
        """
        unseen_levels = _model_unseen_levels(self, unseen_levels)
        if not _prepared:
            newdata = self._newdata_under_policy(newdata, extrapolation)
        if scale not in ("link", "log_odds"):
            raise ValueError(f"scale must be 'link' or 'log_odds', got {scale!r}")
        if scale == "log_odds":
            from mboostrs.families import log_odds_factor

            factor = log_odds_factor(self._family)
            effects = self.partial_effects(which=which, n_points=n_points,
                                           newdata=newdata,
                                           unseen_levels=unseen_levels,
                                           _prepared=_prepared, _buser=_buser,
                                           mstop=mstop)
            return {name: (x, np.asarray(effect, dtype=np.float64) * factor)
                    for name, (x, effect) in effects.items()}
        unseen_levels = _check_unseen_levels(unseen_levels)
        if mstop is None:
            step_limit = len(self._selected)
        elif int(mstop) < 0:
            raise ValueError(f"mstop must be non-negative, got {mstop!r}")
        else:
            step_limit = min(int(mstop), len(self._selected))
        if self._learners_spec is None:
            raise ValueError("Cannot compute partial effects: model was fitted without storing learner specs")
        # New rows take the preparation predict(newdata=) gives them: labels
        # encoded by the fit's recipe, and each %X% tensor design rebuilt at
        # the new rows.
        buser_newdata = None
        if newdata is not None:
            # `predict(newdata, which=)` has already run the recipe on these
            # rows; a second pass would read encoded labels as new ones.
            if _prepared:
                # The caller built the %X% designs from the raw rows.
                buser_newdata = _buser
            else:
                newdata, buser_newdata = _replay_interactions(
                    self, newdata, None, stage="raw", unseen_levels=unseen_levels)
                newdata = _replay_transforms(self, newdata, unseen_levels)
                newdata, buser_newdata = _replay_interactions(
                    self, newdata, buser_newdata, stage="encoded",
                    unseen_levels=unseen_levels)

        if which is None:
            # Indices, not names: two learners on one column share a name,
            # and resolving the name found only the first of them.
            which = list(dict.fromkeys(int(i) for i in self._selected[:step_limit]))
        elif isinstance(which, (str, int, np.integer)):
            which = [which]

        names = self._learner_names
        distinct = self.base_learner_names()
        targets = []
        for entry in which:
            for formula_idx in self._which_indices(entry):
                if not 0 <= formula_idx < len(names):
                    raise ValueError(
                        f"learner index {formula_idx} is out of range; the "
                        f"model has {len(names)} learners"
                    )
                if formula_idx not in targets:
                    targets.append(formula_idx)
        result = {}
        for formula_idx in targets:
            # Keyed by `base_learner_names()`: the raw name unless two
            # learners share it ("x (bols)", "x (bbs)").
            name = distinct[formula_idx]

            step_indices = [i for i, s in enumerate(self._selected[:step_limit])
                            if s == formula_idx]
            if not step_indices:
                continue

            spec = self._learners_spec[formula_idx]

            if isinstance(spec, Bkernel):
                if spec.landmarks is not None:
                    # Nyström coefficients index the landmarks, not the
                    # training rows, so the effect is the store's replay of
                    # the same cross-kernel + L^-1 the fit used, evaluated on
                    # the requested rows. The exact n×n branch below stays the
                    # training-point kernel sum it has always been.
                    source = newdata if newdata is not None else self._data
                    step_indices = [i for i, s in enumerate(self._selected[:step_limit])
                                    if s == formula_idx]
                    filtered_selected = [self._formula_to_rust(formula_idx)] * len(step_indices)
                    filtered_coefs = [
                        np.ascontiguousarray(self._coefficients[i], dtype=np.float64)
                        for i in step_indices
                    ]
                    store = self._replay_store()
                    if store is None:
                        raise ValueError(
                            "cannot compute a Bkernel Nyström partial effect: "
                            "this model carries no learner store and none could "
                            "be rebuilt from its specifications"
                        )
                    x_data, by_data = self._replay_slices(source, buser_newdata)
                    effect = np.asarray(
                        store.predict_newdata_grouped(
                            0.0, filtered_selected, filtered_coefs, x_data,
                            None, by_data, unseen_levels=unseen_levels,
                        ),
                        dtype=np.float64,
                    )
                    var_name = _effect_coordinate_name(spec)
                    try:
                        x_vals = _extract_column(source, var_name)
                    except (KeyError, ValueError, TypeError):
                        x_vals = self._data_array_for(var_name)
                    result[name] = (x_vals, effect)
                    continue
                x_train = _extract_column(self._data, spec.name)
                n_tr = len(x_train)
                kernel_fn = _KERNEL_FUNCS.get(spec.kernel, _kernel_gaussian)
                bw = spec.bandwidth
                total_coef = np.zeros(n_tr, dtype=np.float64)
                for si in step_indices:
                    total_coef += np.asarray(self._coefficients[si], dtype=np.float64)
                # As in the Buser branch: `_coefficients` are unshrunken
                # per-step fits, so the hand-summed replay needs the step
                # length the predictor applies.
                nu = self._control.nu
                effect = np.zeros(n_tr, dtype=np.float64)
                for i in range(n_tr):
                    s = 0.0
                    for j in range(n_tr):
                        s += kernel_fn(abs(float(x_train[i]) - float(x_train[j])), bw) * total_coef[j]
                    effect[i] = nu * s
                x_vals = x_train
                result[name] = (x_vals, effect)
                continue

            if isinstance(spec, Buser):
                if spec._radial_basis is not None:
                    store = self._replay_store()
                    source = newdata if newdata is not None else self._data
                    x_data, by_data = self._replay_slices(source, buser_newdata)
                    selected_rust = [self._formula_to_rust(formula_idx)] * len(step_indices)
                    coefs = [
                        np.ascontiguousarray(self._coefficients[i], dtype=np.float64)
                        for i in step_indices
                    ]
                    effect = np.asarray(store.predict_newdata(0.0, selected_rust, coefs,
                                                             x_data, None, by_data), dtype=float)
                    coordinate = spec._radial_basis["names"][0]
                    result[name] = (_extract_column(source, coordinate), effect)
                    continue
                nu = self._control.nu
                if newdata is None:
                    X = np.asarray(spec.X, dtype=np.float64)
                else:
                    # An interaction-derived buser replays its tensor design
                    # at the new rows; a plain buser needs the caller's
                    # design, which partial effects do not take.
                    buser_designs = buser_newdata or {}
                    if spec.name not in buser_designs:
                        raise ValueError(
                            f"partial effects for the buser learner {spec.name!r} "
                            "cannot be evaluated on new data: a user-supplied "
                            "design is not rebuilt automatically. Evaluate on "
                            "the training rows, or express the term as a %X% "
                            "interaction, whose design replays automatically."
                        )
                    X = np.asarray(buser_designs[spec.name], dtype=np.float64)
                if spec.center and self._buser_centering_means is not None:
                    # Stored as a list (it is model state that serializes as
                    # plain data), so it is an array again before broadcasting.
                    means = np.asarray(self._buser_centering_means[spec.name],
                                       dtype=np.float64)
                    X = X - means[np.newaxis, :]
                total_coef = np.zeros(X.shape[1], dtype=np.float64)
                for si in step_indices:
                    total_coef += np.asarray(self._coefficients[si], dtype=np.float64)
                # `_coefficients` stores each step's unshrunken fit; the
                # predictor scales every step by nu, and this hand-summed
                # branch must do the same (the store's replay path does it
                # internally).
                effect = nu * (X @ total_coef)
                if spec.by is not None:
                    by_col = _extract_column(
                        newdata if newdata is not None else self._data, spec.by)
                    effect = effect * np.asarray(by_col, dtype=np.float64)
                x_vals = np.arange(len(effect))
                result[name] = (x_vals, effect)
                continue

            filtered_selected = [self._formula_to_rust(formula_idx)] * len(step_indices)
            filtered_coefs = [
                np.ascontiguousarray(self._coefficients[i], dtype=np.float64)
                for i in step_indices
            ]

            source = newdata if newdata is not None else self._data
            is_smooth = isinstance(spec, Bbs | Bmono | Bconstrained | Bspatial | Bns | Bkernel)

            if is_smooth and n_points is not None and newdata is None:
                var_name = _effect_coordinate_name(spec)

                x_raw = self._data_array_for(var_name)
                x_grid = np.linspace(float(np.min(x_raw)), float(np.max(x_raw)), n_points)

                # The grid only needs columns a learner actually reads. Building
                # it from the full data frame would force a string column (a
                # factor stored as labels, not yet encoded) through a float
                # cast, which has nothing to do with the smooth effect under
                # isolation. Iterate the learner specs instead and hold each
                # referenced numeric column at its mean; the replay only pulls
                # these.
                grid_columns = {var_name}
                for s in self._learners_spec:
                    if isinstance(s, Bspatial):
                        grid_columns.add(s.name1)
                        grid_columns.add(s.name2)
                    elif isinstance(s, Btree):
                        grid_columns.update(s.names)
                    elif isinstance(s, Buser):
                        continue
                    else:
                        grid_columns.add(s.name)
                    if s.by is not None:
                        grid_columns.add(s.by)

                grid_data = {}

                def get(k):
                    if isinstance(self._data, dict):
                        return np.asarray(self._data[k], dtype=np.float64).ravel()
                    return _extract_column(self._data, k)

                for col in grid_columns:
                    if col == var_name:
                        grid_data[col] = x_grid
                    elif col == self._response_name:
                        grid_data[col] = np.zeros(n_points)
                    else:
                        grid_data[col] = np.full(
                            n_points, float(np.mean(get(col))),
                        )

                replay_source = grid_data
                # A user or tensor design has no column to hold at its mean.
                # It takes zero rows of its own width: the replay checks every
                # learner's row count and reads only this learner's rows.
                replay_designs = {
                    s.name: np.zeros((n_points, np.asarray(s.X).shape[1]))
                    for s in self._learners_spec if isinstance(s, Buser)
                }
                x_vals = x_grid
            else:
                replay_source = source
                replay_designs = buser_newdata
                var_name = _effect_coordinate_name(spec)
                try:
                    x_vals = _extract_column(source, var_name)
                except (KeyError, ValueError, TypeError):
                    # `source` may be new data that does not carry the
                    # learner's coordinate column; fall back to training.
                    x_vals = self._data_array_for(var_name)

            store = self._replay_store()
            if store is None:
                raise ValueError(
                    "cannot compute partial effects: this model carries no "
                    "learner store and none could be rebuilt from its specifications"
                )
            x_data, by_data = self._replay_slices(replay_source, replay_designs)
            effect = np.asarray(
                store.predict_newdata_grouped(
                    0.0, filtered_selected, filtered_coefs, x_data, None, by_data,
                    unseen_levels=unseen_levels,
                ),
                dtype=np.float64,
            )

            result[name] = (x_vals, effect)

        return result

    def plot_partial_effects(self, which=None, n_points=100):
        """Plot partial effects.

        Parameters
        ----------
        which : list of str, optional
            Which base-learners to plot.
        n_points : int
            Number of grid points for smooth effects.

        Returns
        -------
        alt.Chart | alt.ConcatChart
        """
        from mboostrs.plot import plot_partial_effects

        return plot_partial_effects(self, which=which, n_points=n_points)

    def plot(self, which=None, n_points=100):
        """Plot partial effects.

        Parameters
        ----------
        which : list of str, optional
            Which base-learners to plot.
        n_points : int
            Number of grid points for smooth effects.

        Returns
        -------
        alt.Chart | alt.ConcatChart
        """
        return self.plot_partial_effects(which=which, n_points=n_points)





    def _confint_refit(self, parm, level, newdata):
        """REML-refit intervals backing ``confint(method="refit")``.

        Refits the selected terms jointly by REML (:func:`gam_refit`,
        ``unsupported="hold"``) and returns its Bayesian pointwise bands,
        which average close to nominal when the smoothing parameters are
        REML-estimated. ``parm`` names are resolved to their formula terms;
        a family or term the refit cannot represent is refused by name by
        :func:`gam_refit` itself.
        """
        from mboostrs.selection import _term_learners, gam_refit

        terms = None
        if parm is not None:
            entries = [parm] if isinstance(parm, (str, int, np.integer)) else list(parm)
            wanted = set()
            for entry in entries:
                try:
                    found = self._which_indices(entry)
                except ValueError:
                    raise ValueError(
                        f"unknown base-learner {entry!r}; this model's learners "
                        f"are {self.base_learner_names()}"
                    ) from None
                wanted.update(self.base_learner_names()[i] for i in found)
            # One learner per `+` term on a gamboost model, so the mapping
            # inverts; a term that expands into several learners (a factor
            # `by=`) cannot be named precisely -- refit everything and let
            # the caller read the term it wants.
            try:
                from mboostrs.selection import _terms as _split_terms
                lhs_terms = _split_terms(self._formula, self._data)[1]
                mapping = _term_learners(self, lhs_terms)
                inv = {}
                for t, names in mapping.items():
                    for nm in names:
                        inv.setdefault(nm, []).append(t)
                terms = sorted({t for nm in wanted for t in inv.get(nm, [])})
            except NotImplementedError:
                terms = None
            if not terms:
                terms = None
        refit = gam_refit(self, terms=terms, unsupported="hold")
        # The refit reads frames as the caller writes them (factor labels,
        # missing values); `self._data` holds the encoded training frame, so
        # the default grid is the raw training data the refit was fitted on.
        from mboostrs.selection import _bundle

        nd = newdata if newdata is not None else _bundle(self).get("raw_data", self._data)
        fx = refit.effects(nd, level=level)
        return {
            "lower": {t: np.asarray(lo, dtype=float) for t, (_, lo, _) in fx.items()},
            "upper": {t: np.asarray(hi, dtype=float) for t, (_, _, hi) in fx.items()},
            "estimate": {t: np.asarray(e, dtype=float) for t, (e, _, _) in fx.items()},
            "level": level,
            "method": "refit",
            "held": refit.held,
            "edf": refit.edf,
            "conditional_on_selection": refit.conditional_on_selection,
        }

    def confint(self, parm=None, level=0.95, B=1000, B_mstop=25, seed=None, newdata=None, n_jobs=1,
                groups=None, centre=False, method="bootstrap"):
        """Confidence intervals for partial effects.

        method="bootstrap" (default): bootstrap intervals for partial effects.
        These under-cover on a boosted fit (74% empirical coverage at nominal
        90% on the book's additive benchmark): resampling reproduces the
        shrinkage that early stopping induces instead of correcting it, so
        every call on this path warns and points at method="refit".
        method="refit": refit the selected terms jointly by REML
        (:func:`mboostrs.selection.gam_refit`) and return its Bayesian
        pointwise bands, which average close to nominal. Terms the refit
        cannot represent are held at their boosted contribution
        (``unsupported="hold"``) and listed in the result's ``held`` entry;
        a family the refit does not support is refused by name.

        parm: base-learner name(s) or index. None = all used. For
        method="refit", names are resolved to their formula terms.
        level: confidence level (default 0.95).
        B: number of bootstrap replications (bootstrap method only).
        B_mstop: number of inner CV folds for mstop selection (0 = skip).
        seed: random seed.
        newdata: optional data for prediction grid.
        n_jobs: bootstrap worker threads (1 = sequential, -1 = all CPUs). Results
            are independent of n_jobs (each replicate has its own seeded RNG).
            Most effective with B_mstop=0; when B_mstop>0 the inner CV is already
            parallel (rayon), so n_jobs>1 mainly contends and adds little.
        groups: a column name or one label per row. When given, every
            bootstrap sample (and every inner fold of the `B_mstop` tuning)
            draws whole groups with replacement instead of single rows, and a
            row's weight in a sample is its group's draw count times its
            model weight. Use it when rows are repeated measurements of a
            unit -- the points of one curve, the visits of one patient --
            since resampling such rows separately treats them as independent
            and gives intervals that are too narrow.
        centre: when True, each replicate's effect is centred over the rows
            of `newdata` (its mean there subtracted) before the quantiles are
            taken, so the interval is that of the centred effect
            ``f(x) - mean f``. A partial effect's level is not identified: the
            offset and the other terms can absorb a constant, and a ``bbs``
            term (``center=False``, the default) carries its own. Use
            ``centre=True`` to compare an effect with a curve known up to a
            constant, such as a simulation's true component centred over the
            same rows; comparing the default interval with such a curve
            mixes in an arbitrary shift. On a three-term additive simulation
            (400 rows, ten seeds, nominal 0.80) the bands covered the true
            components 0.38 of the time when compared with the components as
            written, whose means over the range are not zero, and 0.89 when
            both were centred over the grid. ``boot_pred`` stays uncentred.
        Returns a dict with 'lower', 'upper', 'level', 'centred', 'boot_pred', and the
        counts describing how many replicates the interval was actually built
        from ('B', 'n_used', 'n_failed', 'n_untuned', and 'n_at_mstop_cap',
        the replicates whose inner cross-validation chose the model's largest
        `mstop` with the risk still falling; 10% or more of them warns), and
        'resampled' ('rows' or 'groups').

        A replicate that does not select a learner contributes effect zero
        for that learner; these zero values are included in the interval.

        With method="refit" the returned dict instead carries 'lower',
        'upper', 'estimate' (each keyed by formula term), 'level', 'method'
        ('"refit"'), the refit's 'edf' per term, 'held' (terms the refit could
        not represent, kept at their boosted contribution), and
        'conditional_on_selection'.
        """
        if method not in ("bootstrap", "refit"):
            raise ValueError(f'method must be "bootstrap" or "refit", got {method!r}')
        if method == "refit":
            return self._confint_refit(parm=parm, level=level, newdata=newdata)
        warnings.warn(
            "Bootstrap intervals on a boosted fit under-cover (74% empirical "
            "coverage at nominal 90% on the book's additive benchmark): "
            "resampling reproduces the shrinkage that early stopping induces "
            "instead of correcting it. For calibrated bands, use "
            "confint(method='refit'), which refits the selected terms by REML "
            "(see mboostrs.selection.gam_refit).",
            UserWarning, stacklevel=2,
        )
        from mboostrs._bootstrap import AT_CAP, FAILED, UNTUNED, BootstrapTally, bootstrap_outcome
        from mboostrs._parallel import parallel_map
        from mboostrs._refit import refit_spec as _refit_spec

        def mstop_fn(cvr):
            return cvr.optimal_mstop

        if not isinstance(centre, (bool, np.bool_)):
            raise TypeError(f"centre must be True or False, got {centre!r}")
        # Learner indices, keyed by `base_learner_names()`: a name two
        # learners share (`bols(x) + bbs(x, center=True)`) names both, and
        # "x (bbs)" names one. Resolving names directly kept only the first.
        if parm is None:
            which_indices = list(dict.fromkeys(int(i) for i in self._selected))
        else:
            entries = [parm] if isinstance(parm, (str, int, np.integer)) else list(parm)
            which_indices = []
            # An unknown name used to reach the bootstrap loop, miss every
            # replicate's `partial_effects` output, and come back as an
            # all-NaN interval -- a typo reported as a result.
            for entry in entries:
                try:
                    found = self._which_indices(entry)
                except ValueError:
                    raise ValueError(
                        f"unknown base-learner {entry!r}; this model's learners "
                        f"are {self.base_learner_names()}"
                    ) from None
                which_indices.extend(i for i in found if i not in which_indices)
        distinct = self.base_learner_names()
        which_names = [distinct[i] for i in which_indices]

        n = len(self._fitted)
        model_weights = self.model_weights()
        # R's `cv(model.weights(object), B = B)`: multinomial counts drawn with
        # probabilities proportional to the model's own weights, so a row the
        # fit gave no weight is never drawn.
        group_labels = _confint_groups(self, groups, n)
        if group_labels is None:
            outer = _cv_weights(n, "bootstrap", B, prob=0.5,
                                seed=42 if seed is None else int(seed),
                                weights=model_weights)
        else:
            from mboostrs._bootstrap import group_bootstrap_counts

            weights_now = np.asarray(model_weights, dtype=np.float64)
            outer = group_bootstrap_counts(
                group_labels, weights_now > 0, B,
                42 if seed is None else int(seed),
            ) * weights_now[:, None]
        seeds = np.random.SeedSequence(seed if seed is not None else 42).spawn(B)
        spec = _refit_spec(self)
        # newdata=None evaluates each replicate on its own training design.
        # The stored training frame is not usable here: `self._data` holds
        # the encoded factors (0..L-1 codes), and a replicate re-applies the
        # fit's recipe to any frame passed as newdata, so the codes were read
        # as unseen levels in every replicate. The raw frame avoids that but
        # fails on missing values under na_action="omit", so the default
        # grid is left to the replicate itself.
        nd = newdata

        # Effects are evaluated on `nd`, so a replicate that fails — or that
        # never selects the learner — must pad to `nd`'s length, not to the
        # training length. Padding with `n` makes the per-replicate vectors
        # ragged whenever `newdata` has a different row count, and np.array
        # then fails on the stack instead of producing an interval.
        if nd is None:                         # each replicate's training rows
            n_eval = n
        elif hasattr(nd, "height"):           # polars
            n_eval = nd.height
        elif isinstance(nd, dict):
            n_eval = len(next(iter(nd.values())))
        else:                                  # pandas: `.values` is an array
            n_eval = len(nd)

        tally = BootstrapTally(B=B)
        _REPLICATE_FAILURES = (ValueError, RuntimeError, KeyError,
                               np.linalg.LinAlgError)

        def _one(b):
            # Independent RNG per replicate -> thread-safe and n_jobs-invariant.
            child = np.random.default_rng(seeds[b])
            w_boot = np.ascontiguousarray(outer[:, b], dtype=np.float64)
            try:
                # Through the refit spec: `weight_policy`, `bins`, `center`,
                # `na_action`, `subsample` and the transform recipe all reach
                # the replicate. The hand-built call carried the learners and
                # nothing else.
                mod_b = spec.build(weights=w_boot, control=self._control)
                if B_mstop > 0:
                    if group_labels is None:
                        folds_b = _cv_weights(n, "bootstrap", B_mstop, 0.5, int(child.integers(0, 2**31)), w_boot)
                    else:
                        # Counts only: cvrisk multiplies the folds by the
                        # replicate's own weights.
                        from mboostrs._bootstrap import group_bootstrap_counts

                        folds_b = group_bootstrap_counts(
                            group_labels, w_boot > 0, B_mstop, int(child.integers(0, 2**31)))
                    try:
                        # Inner CV runs sequentially when the outer bootstrap loop
                        # is threaded, so the two axes never oversubscribe cores.
                        cvr = mod_b.cvrisk(folds=folds_b, parallel=(n_jobs in (None, 1)),
                                           _inner=True)
                        if cvr.at_boundary():
                            tally.mark(b, AT_CAP)
                        mod_b = mod_b[: mstop_fn(cvr)]
                    except _REPLICATE_FAILURES as exc:
                        # The replicate still contributes, at the outer mstop.
                        tally.mark(b, UNTUNED, exc)
                pe = mod_b.partial_effects(which=which_indices, newdata=nd)
                out = {}
                for w_name in which_names:
                    if w_name in pe:
                        # partial_effects returns (x_values, effect_values)
                        _, eff = pe[w_name]
                        out[w_name] = np.asarray(eff, dtype=float)
                    else:
                        # Not selected in this replicate: its contribution is
                        # zero, the same value `mboost` bootstraps.
                        out[w_name] = np.zeros(n_eval)
                return out
            except _REPLICATE_FAILURES as exc:
                tally.mark(b, FAILED, exc)
                return None

        rows = [row for row in parallel_map(_one, range(B), n_jobs)
                if row is not None]
        counts = bootstrap_outcome(tally, B=B, what="partial-effect")
        counts["weights"] = spec.describe_weights()
        predictions = {w_name: [r[w_name] for r in rows] for w_name in which_names}

        result = {"level": level, "boot_pred": predictions,
                  "centred": bool(centre),
                  "seed": seed, "B_mstop": B_mstop,
                  "resampled": "rows" if group_labels is None else "groups",
                  **counts}
        from mboostrs import _rust

        lower = {}
        upper = {}
        for w_name, preds in predictions.items():
            pred_matrix = np.asarray(preds, dtype=np.float64)
            lo, hi = _rust.percentile_interval(pred_matrix, float(level), bool(centre))
            lower[w_name] = np.asarray(lo, dtype=np.float64)
            upper[w_name] = np.asarray(hi, dtype=np.float64)
        result["lower"] = lower
        result["upper"] = upper
        return result

    def __getstate__(self):
        return {
            "formula": self._formula,
            "family": self._family,
            "control": self._control,
            "response_name": self._response_name,
            "learner_names": self._learner_names,
            "learner_types": self._learner_types,
            "fitted": self._fitted,
            "risk": self._risk,
            "selected": self._selected,
            "coefficients": self._coefficients,
            "offset": self._offset,
            "nu": self._nu,
            "weights": self._weights,
            "learners_spec": self._learners_spec,
            # One column per variable the learners read, which is what a
            # restored model rebuilds its learner store from. This replaces the
            # thirty per-kind cache keys that used to be needed to rebuild
            # designs Python-side.
            "replay_columns": self._replay_column_cache(),
            "rust_idx_to_formula_idx": self._rust_idx_to_formula_idx,
            "formula_idx_to_rust_idx": self._formula_idx_to_rust_idx,
            "family_sigma": self._family_sigma,
            "family_sigma_path": self._family_sigma_path,
            "execution_report": self._execution_report,
            # Both are needed to predict on new data: the recipe rebuilds
            # derived columns, and the interaction recipes rebuild the tensor
            # designs of `%X%` and factor-`by` terms. Without them a restored
            # model can still report its fit but cannot see new rows.
            "transform_recipe": getattr(self, "_transform_recipe", None),
            # The response's own transform step, when the formula was written
            # `log(rent) ~ ...`: the thing that inverts quantiles to the
            # original scale and carries the Jacobian in densities. The
            # predictor recipe above never contains it, so prediction data
            # does not need the response column.
            "response_transform": getattr(self, "_response_transform", None),
            "interactions": getattr(self, "_interactions", None),
            "comparison_identity": getattr(self, "_comparison_identity", None),
            # The weight contract has to survive the round trip: without
            # these, a restored model reported the default policy and no
            # input vector, so `input_weights()` came back None and
            # `weight_policy` said "upstream" for a fit made with "raw".
            "input_weights": getattr(self, "_input_weights", None),
            "weight_policy": getattr(self, "_weight_policy", None),
            "weights_rescaled": getattr(self, "_weights_rescaled", None),
            "breslow_baseline": getattr(self, "_breslow_baseline", None),
            # The fit's own call after preprocessing, which `continue_to`,
            # `update(**kwargs)`, `cvrisk` and the other refits rebuild the
            # model from, and which `formula` reads the written formula from.
            # A restored model without it refused every refit.
            "refit_bundle": getattr(self, "_refit_bundle", None),
            # The training-row means a centered `buser` design is shifted by
            # in `partial_effects`, and the residual `continue_to` resumes
            # from.
            "buser_centering_means": getattr(self, "_buser_centering_means", None),
            "residual": getattr(self, "_residual", None),
        }

    def __setstate__(self, state):
        # Restored explicitly rather than left to the mixin's defaults: a
        # missing key means state written before the weight contract
        # existed, and "upstream" with no input vector is the right reading
        # of that, but only because nothing else was recorded.
        self._refit_bundle = state.get("refit_bundle")
        self._buser_centering_means = state.get("buser_centering_means")
        residual = state.get("residual")
        self._residual = (None if residual is None
                          else np.asarray(residual, dtype=np.float64))
        self._input_weights = state.get("input_weights")
        self._weight_policy = state.get("weight_policy") or "upstream"
        self._weights_rescaled = state.get("weights_rescaled")
        self._breslow_baseline = state.get("breslow_baseline")
        self._formula = state["formula"]
        self._family = state["family"]
        # An artifact decodes the control record as a plain tuple; the
        # record restores `control.nu` and the other named fields the
        # refit and partial-effect paths read.
        control = state["control"]
        self._control = None if control is None else as_control(control)
        self._response_name = state["response_name"]
        self._learner_names = state["learner_names"]
        self._learner_types = state["learner_types"]
        self._fitted = state["fitted"]
        self._risk = state["risk"]
        self._selected = state["selected"]
        self._coefficients = state["coefficients"]
        self._offset = state["offset"]
        self._nu = state["nu"]
        self._data = None
        self._weights = state.get("weights")
        self._learners_spec = state.get("learners_spec")
        rust_idx_to_formula_idx = state.get("rust_idx_to_formula_idx")
        if self._learners_spec is not None and rust_idx_to_formula_idx is not None:
            self._learner_layout = LearnerLayout.from_mapping(rust_idx_to_formula_idx)
            if sorted(self._learner_layout.rust_idx_to_formula_idx) != list(
                range(len(self._learners_spec))
            ):
                raise ValueError("stored GAM learner layout does not match learner specs")
            self._rust_idx_to_formula_idx = list(self._learner_layout.rust_idx_to_formula_idx)
            self._formula_idx_to_rust_idx = dict(self._learner_layout.formula_idx_to_rust_idx)
        elif self._learners_spec is not None:
            self._learner_layout = LearnerLayout.from_specs(self._learners_spec)
            self._rust_idx_to_formula_idx = list(self._learner_layout.rust_idx_to_formula_idx)
            self._formula_idx_to_rust_idx = dict(self._learner_layout.formula_idx_to_rust_idx)
        elif rust_idx_to_formula_idx is not None:
            self._learner_layout = LearnerLayout.from_mapping(rust_idx_to_formula_idx)
            self._rust_idx_to_formula_idx = list(self._learner_layout.rust_idx_to_formula_idx)
            self._formula_idx_to_rust_idx = dict(self._learner_layout.formula_idx_to_rust_idx)
        else:
            self._learner_layout = None
            self._rust_idx_to_formula_idx = None
            self._formula_idx_to_rust_idx = state.get("formula_idx_to_rust_idx")
        self._replay_columns = state.get("replay_columns")
        self._bbs_deriv = state.get("bbs_deriv", [])
        self._family_sigma = state.get("family_sigma")
        self._family_sigma_path = state.get("family_sigma_path", [])
        self._execution_report = state.get("execution_report")
        self._transform_recipe = state.get("transform_recipe")
        self._response_transform = state.get("response_transform")
        self._comparison_identity = state.get("comparison_identity")
        self._interactions = state.get("interactions")
        self._learner_store = None

    def logLik(self):
        """The fitted model's log-likelihood, or a refusal naming the reason.

        Available only where the fitted loss is a log-likelihood. `Binomial`,
        `Poisson`, `NBinomial`, `PropOdds` and the AFT families minimise the
        exact negative log-likelihood, so this is `-risk`. `Gaussian` minimises the
        residual sum of squares, which is not the Gaussian NLL -- no variance,
        no `2*pi`, and a factor of two out -- so the log-likelihood is
        reconstructed with `sigma^2 = RSS/n`, the MLE the loss implies but
        does not carry.

        For a robust or ranking loss -- `Laplace`, `Quantile`, `Expectile`,
        `Huber`, `AdaExp`, `HingeLoss`, `AUC`, `Gehan`, `Cindex` -- there is no
        probability model, so this method raises `NotImplementedError` for
        them. `GammaReg` does not refuse. Its risk omits `log(y)` on each row,
        and `logLik` adds that term back, so the number is the gamma density
        at the fitted mean and the `nuisance()` shape. `negative_risk()`
        leaves those terms out. `Multinomial` refuses, because its reported
        risk omits the reference class; see :meth:`likelihood_contract`.

        Unlike `mboost`, this method reports a log-likelihood only when the
        family defines one. The Gaussian value includes the normalizing
        terms implied by its fitted variance. Use :meth:`negative_risk` for
        the negated empirical risk convention used by `mboost`.
        """
        from mboostrs._likelihood import normalized_loglik

        return normalized_loglik(self)

    def negative_risk(self):
        """The negated empirical risk at the current mstop.

        This matches `mboost`'s `logLik` convention, and for most families is
        the quantity the fit minimised with its sign flipped. `Multinomial` is
        the exception: its reported risk omits the reference class, so
        `negative_risk()` is the mboost-comparable *reported* value while the
        fit actually descends the full categorical NLL (see
        :meth:`likelihood_contract`). It is a log-likelihood
        only for the families where the risk happens to be one; for the rest it
        is a negated robust or ranking loss, and differences between two of
        them are not likelihood ratios. Use it to compare two fits of the *same
        family* on the *same data*, and :meth:`logLik` when a log-likelihood is
        what the calculation needs.
        """
        return float(-self._risk[self.mstop])

    def likelihood_contract(self):
        """What this family's risk is: a likelihood, a loss, or neither.

        Returns the :class:`~mboostrs._likelihood.FamilyLikelihood` row, whose
        fields say whether the loss comes from a probability model, whether
        `-risk` is the complete log-likelihood, what measure that likelihood is
        taken with respect to, and which residual kinds are defined.
        """
        from mboostrs._likelihood import family_likelihood

        return family_likelihood(self._family)

    def to_dict(self, *, include_arrays=False):
        """The ``summary(as_dict=True)`` record plus a ``"provenance"`` entry:
        the library build and environment, and the fingerprint of the
        training data. ``include_arrays`` is accepted for the uniform export
        signature; the summary holds no per-row arrays."""
        from mboostrs._output import export_model

        return export_model(self, training_data=self._data)

    def summary(self, as_dict=False, cv=None):
        """Summarise the fitted model.

        Returns a :class:`~mboostrs.summary.Summary`, which prints as
        fixed-width text and renders as a table in a notebook. Use
        ``as_dict=True`` for the machine-readable form.

        Parameters
        ----------
        as_dict : bool
            If True, return a JSON-serializable dict instead of the Summary.
        cv : CvRiskResult, optional
            The cross-validation result for this model. Supplying it adds a
            block comparing the iteration the model stopped at with the one
            cross-validation prefers. A model keeps no record of how it was
            tuned, so this is not inferred.
        """
        ms = self.mstop
        family_name = self._family.__class__.__name__

        selected = self._selected[:ms]
        freq = {}
        for s in selected:
            name = self._learner_names[s]
            freq[name] = freq.get(name, 0) + 1

        coefs = self.coef()
        nonzero = {}
        for k, v in coefs.items():
            # A tree learner's entry is its decoded per-step node records,
            # not coefficients; its presence is the selection frequency above.
            if isinstance(v, (list, dict)):
                continue
            arr = np.atleast_1d(v)
            if np.any(np.abs(arr) > 1e-12):
                nonzero[k] = arr.tolist()

        info = {
            "model_class": "GAMBoostModel",
            "family": family_name,
            "prediction_scale": prediction_scale_text(self._family),
            "formula": self._formula,
            "mstop": ms,
            "nu": float(self._nu),
            # A per-observation offset (log exposure of a rate model) is a
            # vector, so the record gives its length and mean, not one number.
            "offset": ({"per_observation": int(np.asarray(self._offset).size),
                        "mean": float(np.mean(self._offset))}
                       if _is_vector_offset(self._offset)
                       else float(self._offset)),
            "risk_at_mstop": float(self._risk[ms]),
            "n": len(self._fitted),
            "n_omitted": len(getattr(self, "_na_dropped", []) or []),
            "selection_frequency": {
                name: {"count": count, "pct": 100.0 * count / ms}
                for name, count in sorted(freq.items(), key=lambda x: -x[1])
            },
            "nonzero_coefficients": nonzero,
            # `n` alone does not say what a row counted for. A fit made with
            # fractional weights under the upstream policy ran on a rescaled
            # vector, and a summary that reported only the row count described
            # a sample nobody fitted.
            "weights": self._weight_summary(),
        }
        if as_dict:
            return info
        from mboostrs.summary import build_summary  # noqa: PLC0415

        return build_summary(self, cv=cv)

    def update(self, mstop=None, **kwargs):
        """Update the model by changing mstop or other parameters.

        With only ``mstop``, returns the model sliced to it, or grown to it
        through :meth:`continue_to` when it is larger than the current one.
        With keyword arguments, fits the model again with those arguments
        replaced and every other argument as the original fit had it.

        Parameters
        ----------
        mstop : int, optional
            New stopping iteration. Without keyword arguments it slices or
            grows the existing model; with them it is the refit's ``mstop``,
            which otherwise stays at the current one.
        kwargs : dict
            Arguments to change in the refit: the control fields ``nu``,
            ``risk``, ``trace``, ``step`` and ``step_iterations``, a whole
            ``control=`` record, or the fit arguments ``family``, ``weights``,
            ``offset``, ``bins``, ``subsample``, ``subsample_seed`` and
            ``weight_policy``. Any other name raises ``TypeError``.

        Returns
        -------
        GAMBoostModel
            Updated model.
        """
        if kwargs:
            from mboostrs._refit import refit_with_updates

            return refit_with_updates(self, mstop, kwargs)
        if mstop is not None:
            if mstop > self.mstop:
                return self.continue_to(mstop)
            return self[:mstop]
        return self

    def continue_to(self, mstop: int) -> GAMBoostModel:
        """Return the model grown to a larger ``mstop``.

        ``fit`` fixes ``mstop`` up front, but the interactive loop is fit,
        inspect the risk path, discover the optimum is at the boundary, extend.
        ``continue_to(N)`` does that without re-specifying the model.

        The returned model is the one a cold ``fit`` with ``mstop=N`` would
        produce, to machine precision — boosting is deterministic given the
        data, so the first ``self.mstop`` iterations are identical either way.
        Slicing it back reproduces the current model: ``fit.continue_to(N)[:m]``
        equals ``fit`` for ``m == fit.mstop``.

        Growing to ``N <= self.mstop`` slices instead. Growing below 1 is an
        error.

        Note
        ----
        Where the fit kept the gradient it left off on, this **resumes** it and
        does not recompute the first ``self.mstop`` iterations. That covers
        every family whose calibration is a pure function of ``(y, f, w)``.
        A fit that took the Gaussian Gram route keeps per-learner projections
        rather than a length-n residual, and ``PropOdds``, ``Multinomial``, the
        survival families and binned learners have no resume path, so those
        refit from iteration zero. The equivalence guarantee above holds on
        both routes; only the cost differs.
        """
        if mstop <= self.mstop:
            if mstop < 1:
                raise ValueError(f"mstop must be >= 1, got {mstop}")
            return self[:mstop]

        from mboostrs._refit import refit_spec

        # A scoring-only artifact keeps what prediction needs and drops the
        # refit bundle; growing such a model is a refit and says so, rather
        # than dereferencing the bundle it does not hold.
        if getattr(self, "_refit_bundle", None) is None:
            raise ValueError(
                "this model did not retain the inputs needed to refit it; a "
                "scoring-only artifact predicts but does not grow, and neither "
                "does an artifact or pickle written by a release that did not "
                "keep them. Fit it again to continue.")
        # Warm continuation: resume the retained fitted state rather than
        # refit from zero. Only where the residual was kept and the family's
        # calibration is a pure function of (y, f, w) -- PropOdds is
        # path-dependent, Multinomial is Kronecker-flat and not wired here, and
        # the survival families concatenate [time, event] so their residual
        # index does not match `fitted`. Everything else falls back to the cold
        # refit, which the equivalence guarantee below covers.
        warm = self._try_warm_continue(mstop)
        if warm is not None:
            return warm

        # Through the refit spec, so every hyperparameter the fit accepted --
        # `learners`, `subsample`, `na_action` among them -- reaches the grown
        # model. The hand-written call listed six of them and silently dropped
        # the rest, so continuing a subsampled or explicitly-learner-specified
        # fit produced a different model under the same name. The spec also
        # carries the transform and interaction recipes forward, which the
        # rewritten formula no longer triggers on a refit.
        spec = refit_spec(self)
        control = as_control(self._control)._replace(mstop=mstop, nu=self._nu)
        return spec.build(control=control)

    def _try_warm_continue(self, mstop: int):
        """The grown model via `try_fit_continue`, or `None` when that route is
        unavailable for this fit."""
        from mboostrs.families import (
            CoxPH,
            Loglog,
            Lognormal,
            Multinomial,
            PropOdds,
            Weibull,
            _family_params,
            _family_rust_name,
        )
        _residual = getattr(self, "_residual", None)
        # A per-observation offset is the frozen start of the path, and the
        # resume binding takes the start as one number; the cold refit
        # carries the vector through the fit's own `offset=` argument.
        if (
            _residual is None
            or len(_residual) == 0
            or _fit_gamboost_continue_specs is None
            or _is_vector_offset(self._offset)
            or isinstance(self._family, (Multinomial, PropOdds,
                                         CoxPH, Weibull, Loglog, Lognormal))
        ):
            return None
        if any(getattr(spec, "bins", None) is not None for spec in self._learners_spec):
            return None

        from mboostrs.gamlss import learner_spec

        # A restored model holds no training frame; the learner columns and
        # the response it stored for the replay store serve the same purpose.
        data = self._replay_frame()
        if data is None:
            return None
        try:
            y = np.ascontiguousarray(
                _extract_response(data, self._response_name, self._family),
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            return None
        w = (
            self._weights
            if self._weights is not None
            else np.ones(len(self._fitted), dtype=np.float64)
        )
        nu = self._control.nu
        bols_form = _bols_form_for(
            self._learners_spec, vector_offset=_is_vector_offset(self._offset),
            kronecker=isinstance(self._family, Multinomial), family=self._family)
        try:
            specs = [learner_spec(learner, data, bols_form=bols_form)
                     for learner in self._learners_spec]
        except (KeyError, ValueError):
            return None
        (
            offset_val,
            selected_rust,
            coefs_vecs,
            risk,
            fitted,
            nuisance_vec,
            nuisance_vec_path,
            learner_store,
            execution_report,
            residual,
        ) = call_fit(_fit_gamboost_continue_specs,
            specs,
            y,
            np.ascontiguousarray(w, dtype=np.float64),
            int(mstop),
            nu,
            _family_rust_name(self._family),
            _family_params(self._family),
            np.ascontiguousarray(self._fitted, dtype=np.float64),
            float(self._offset),
            list(self._risk),
            list(self._selected),
            list(self._coefficients),
            np.ascontiguousarray(_residual, dtype=np.float64),
            warm_nuisance=self._family_sigma_path,
            **step_kwargs(self._control),
        )

        rust_to_formula = getattr(self, "_rust_idx_to_formula_idx", None)
        selected = (
            [rust_to_formula[s] for s in selected_rust]
            if rust_to_formula is not None
            else list(selected_rust)
        )

        grown = GAMBoostModel(
            formula=self._formula,
            family=self._family,
            control=as_control(self._control)._replace(mstop=mstop, nu=nu),
            response_name=self._response_name,
            learner_names=self._learner_names,
            learner_types=self._learner_types,
            fitted=np.asarray(fitted, dtype=np.float64),
            risk=np.asarray(risk, dtype=np.float64),
            selected=selected,
            coefficients=[np.array(c, dtype=np.float64) for c in coefs_vecs],
            offset=offset_val,
            nu=nu,
            data=self._data,
            weights=self._weights,
            learners_spec=self._learners_spec,
            rust_idx_to_formula_idx=self._rust_idx_to_formula_idx,
            formula_idx_to_rust_idx=self._formula_idx_to_rust_idx,
            buser_centering_means=getattr(self, "_buser_centering_means", None),
            family_sigma=nuisance_vec if nuisance_vec else None,
            family_sigma_path=nuisance_vec_path if nuisance_vec_path else None,
            learner_store=learner_store,
            execution_report=execution_report,
            residual=residual,
        )
        grown._transform_recipe = getattr(self, "_transform_recipe", None)
        grown._response_transform = getattr(self, "_response_transform", None)
        grown._interactions = getattr(self, "_interactions", None)
        grown._refit_bundle = getattr(self, "_refit_bundle", None)
        grown._na_dropped = getattr(self, "_na_dropped", None)
        grown._comparison_identity = getattr(self, "_comparison_identity", None)
        grown._bin_decisions = getattr(self, "_bin_decisions", None)
        grown._input_weights = getattr(self, "_input_weights", None)
        grown._weight_policy = getattr(self, "_weight_policy", "upstream")
        grown._breslow_baseline = _breslow_baseline_of(grown)
        return grown

    def __repr__(self) -> str:
        family_name = self._family.__class__.__name__
        # `_formula` is the rewritten formula the engine fitted (a `%X%` term
        # becomes a derived `buser`); show the one the caller wrote.
        bundle = getattr(self, "_refit_bundle", None) or {}
        formula = bundle.get("raw_formula") or self._formula
        return f"GAMBoostModel(formula={formula!r}, family={family_name}, mstop={self.mstop})"



def _omit_for_learners(formula, data, learners, na_action):
    """`na_action` for a call that supplies its own learner objects.

    The rows checked are those of every column the formula or a learner reads:
    a learner's variable, both coordinates of a spatial term, every column of a
    tree, and any `by=` column. A `Buser` learner carries a design matrix with
    one row per observation of the data as given, and dropping rows from the
    data would leave that matrix misaligned, so a drop that would be needed is
    refused instead of made.
    """
    from mboostrs._frame import column_names
    from mboostrs.formula import apply_na_action

    available = set(column_names(data))
    names = []
    for spec in learners:
        for attribute in ("name", "name1", "name2", "by"):
            value = getattr(spec, attribute, None)
            if isinstance(value, str) and value in available:
                names.append(value)
        names.extend(name for name in (getattr(spec, "names", None) or ())
                     if name in available)
    response = partition_tilde(formula)[0].strip()
    columns = list(dict.fromkeys([response, *names]))
    columns = [name for name in columns if name in available]
    if not columns:
        return data, []
    filtered, dropped = apply_na_action(
        f"{columns[0]} ~ " + " + ".join(columns[1:] or columns[:1]), data, na_action)
    if dropped and any(isinstance(spec, Buser) for spec in learners):
        raise ValueError(
            f"na_action={na_action!r} would drop {len(dropped)} rows, but a Buser "
            "learner carries a design matrix with one row per observation of the "
            "data as given, which dropping rows would misalign. Drop the rows from "
            "the data and the matrix together before fitting.")
    return filtered, dropped

@capture_fit_warnings
@fit_checkpoint
def gamboost(
    formula: str,
    data,
    *,
    family=None,
    control=None,
    weights=None,
    offset=None,
    learners: list | None = None,
    env: dict | None = None,
    subsample: float | None = None,
    subsample_seed: int = 0,
    na_action: str = "error",
    bins: int | None = None,
    weight_policy: str = "upstream",
) -> GAMBoostModel:
    """Fit a component-wise boosted additive model.

    Workflow, pitfalls and worked analyses: ``mboostrs.agent_guide()``, or
    ``python -m mboostrs guide index`` from a shell. After a fit, run
    ``mboostrs.reference_gap`` to measure what a flexible model still gains.

    Generalises :func:`glmboost` to base learners that are not single
    coefficients: P-splines, monotone splines, random effects, spatial
    surfaces, trees and user-supplied designs. One learner is selected and
    updated per iteration, so the fit stays additive and each term's
    contribution can be read off separately.

    Selection is fair between learners only when they have equal degrees of
    freedom, and that is something the caller arranges rather than something
    this function does. A spline with more `df` than a linear term wins
    selection partly for being more flexible, not only for fitting better.
    `bbs` defaults to `df=4`; `bols` is unpenalized by default and carries the
    df its design implies, so `"y ~ bbs(x) + bols(z)"` is *not* an equal-df
    menu. Write the `df` you want on every term when the comparison between
    them is the point.

    Parameters
    ----------
    formula : str
        Formula whose terms name learners, ``"y ~ bbs(x, df=4) + bols(z)"``.
        A bare numeric name becomes ``bbs(name)`` — a P-spline at the default
        ``df=4``, matching R's ``gamboost`` — so ``"y ~ x"`` fits a smooth, not
        a line. Write ``bols(x)`` for a linear term. A bare *categorical* name
        is coded as ``brandom(name, lambda=0)``, one unpenalized effect per
        level (the fit ``bols(name)`` gives), and a ``UserWarning`` names the
        column; its levels are never numbered, because a smooth over numbered
        levels would fit an order that is not there. ``"y ~ ."`` expands to
        every column except the response.
    data : frame or mapping
        Polars or pandas frame, or a mapping of column name to array.
    family : Family, optional
        The loss to minimise. Defaults to :class:`~mboostrs.families.Gaussian`.
    control : boost_control or tuple, optional
        ``boost_control(mstop, nu)``. Defaults to 100 iterations at ``nu=0.1``.
    weights : array, optional
        Non-negative observation weights, one per row. Integer weights are
        read as frequencies -- weight 3 stands for three identical rows -- and
        are used at the scale given. Fractional weights are read as analytic
        weights, where only the ratios carry meaning, and are rescaled to sum
        to the number of weight-positive rows before fitting; see
        ``weight_policy``. A zero weight excludes a row while keeping its
        position, so every row-aligned object stays aligned. An all-zero
        vector is rejected: the resulting object would report a completed path
        over an empty sample.
    offset : float, optional
        A single starting value for the linear predictor. Per-observation
        offsets are not supported; a constant array is read as its one value.
    learners : list, optional
        Learner objects (``Bbs``, ``Bols``, ``Bmrf``, ...) given directly
        instead of being parsed from the formula. Use with ``"y ~ ."`` when a
        learner needs an argument awkward to write in a formula, such as a
        penalty matrix.
    env : dict, optional
        Names resolved inside formula learner arguments, for example the
        ``K`` matrix of ``bmrf(district, K=K_matrix)``.
    subsample : float, optional
        Fit on this fraction of rows, drawn once before fitting.
    subsample_seed : int
        Seed for that draw.
    na_action : {"error", "omit", "impute", "impute_model", "impute_model_split"}
        What to do with rows holding a missing value: refuse them, drop them,
        or (``"impute"``) fill each missing predictor value with the training
        median and add a ``<x>_missing`` indicator term, which lets the model
        predict rows with missing values; a missing response still drops its
        row. ``"impute_model"`` does the same with a model-based fill: each
        numeric predictor with missing values is filled from a penalized
        additive regression on the other predictors, fitted on the training
        rows where it is observed. ``"impute_model_split"`` also fits a smooth or linear term on a filled
        column as two terms, one on the rows where it was observed and one on the
        filled rows. It is opt-in: on the claim-cost benchmark it lowered the
        coverage of one effect interval from 0.90 to 0.60. If the data already has a column named
        ``<x>_missing``, the indicator is named ``<x>_missing_`` instead. The indicator is built
        from ``<x>`` at prediction time, so prediction data and
        ``sensitivity`` need only ``<x>``.
    bins : int, optional
        Global quantile-bin count for supported continuous ``bols``/``bbs``
        learners and per-axis count for ``bspatial``. A learner's explicit
        ``bins=`` value takes precedence. Random-intercept learners are always
        stored through their exact discrete bins. Terms with no binned design
        -- a ``by=`` on any learner, an intercept-carrying ``bols``, a
        shape-constrained ``bbs``, and every learner class outside those four
        -- keep their exact storage; the first three are named in a warning,
        because those are the ones a caller writing ``bins=`` expects to be
        binned.
    weight_policy : {"upstream", "raw"}
        How the scale of ``weights`` is settled before fitting. ``"upstream"``
        (the default) reproduces ``mboost``'s ``rescale_weights``: integer
        weights pass through and fractional weights are rescaled to sum to the
        number of positive weights. ``"raw"`` passes the vector through
        unchanged, for a caller whose weight scale is itself meaningful (survey
        weights summing to a population total, for example). The scale is not
        cosmetic -- it enters ``X'WX``, the ``df -> lambda`` calibration, and
        the risk -- so ``"raw"`` gives a different path from R's on the same
        fractional weights. The chosen policy is on the model as
        ``weight_policy`` and in ``summary()``.

    Returns
    -------
    GAMBoostModel
        A spline learner has no single coefficient, so read its fit with
        ``partial_effects()`` or ``mboostrs.plot.plot_partial_effects``
        rather than ``coef()``.

    See Also
    --------
    glmboost : linear terms only, one coefficient each.
    gamboostLSS : one additive predictor per distribution parameter.

    Examples
    --------
    >>> import mboostrs as mb
    >>> data = mb.datasets.bodyfat()
    >>> fit = mb.gamboost("DEXfat ~ bbs(age, df=4) + bbs(hipcirc, df=4)",
    ...                   data=data, family=mb.Gaussian(),
    ...                   control=mb.boost_control(200, 0.1))
    >>> effects = fit.partial_effects()
    """
    if not _HAS_RUST_ENGINE:
        raise RuntimeError("Rust engine not available. Run `maturin develop --release` first.")
    if family is None:
        family = Gaussian()
    from mboostrs.families import require_standard_family

    require_standard_family(family, "gamboost")
    if control is None:
        # A family may set its own default step length (Quantile: 0.3).
        control = (100, getattr(family, "default_nu", 0.1), "inbag", False)
    if len(control) == 2:
        control = (control[0], control[1], "inbag", False)

    control = as_control(control)
    mstop, nu, risk_str, trace_bool = control.mstop, control.nu, control.risk, control.trace

    # Missing-value policy: "omit" drops rows with a missing value in a model
    # variable before any other processing (weights/offset are aligned below).
    _na_dropped: list[int] = []
    if na_action == "error" and learners is None:
        from mboostrs.formula import apply_na_action

        # Names a label column with missing values; changes nothing else.
        apply_na_action(formula, data, na_action)
    elif na_action != "error":
        if learners is None:
            from mboostrs.formula import apply_na_action

            data, _na_dropped = apply_na_action(formula, data, na_action)
        else:
            data, _na_dropped = _omit_for_learners(formula, data, learners, na_action)
        if _na_dropped:
            weights, offset = _drop_aligned(weights, offset, _na_dropped)

    if learners is not None and env is not None:
        raise ValueError(
            "Cannot specify both `learners` and `env`. "
            "Use `learners=[...]` for programmatic learner construction "
            "or `env={{...}}` for formula-based matrix resolution."
        )

    # The call before transforms/interactions are materialized. Prediction
    # replay uses the fitted recipes below; validation refits rebuild them on
    # each proper training subset.
    # `y ~ x - 1` / `y ~ 0 + x`: gamboost has no global intercept candidate to
    # drop. The offset is the family's offset, and each learner carries its own
    # null space, so removing the term leaves the learner list unchanged. A
    # redundant `+ 1` / `- 0` stays refused.
    from mboostrs.formula import (
        _INTERCEPT_REDUNDANT,
        refuse_intercept_control,
        split_intercept,
    )

    if not _intercept_only(formula, data):
        refuse_intercept_control(formula, data, "gamboost", forms=_INTERCEPT_REDUNDANT)
    formula, _global_intercept = split_intercept(formula, data)
    _raw_formula = formula
    _raw_data = data
    # An ordered response (levels= or an ordered categorical) becomes
    # PropOdds' 1..K codes here, once, so every later read of the response
    # (cross-validation, refits, diagnostics) sees the same coding.
    from mboostrs.families import ordinal_response
    from mboostrs.formula import parse_response

    data, family = ordinal_response(data, parse_response(formula), family)
    _raw_learners = learners
    _raw_env = env

    # Interaction terms (a %X% b) rewrite into buser learners with injected
    # tensor designs. Do this first, before factor encoding, so a factor
    # marginal inside an interaction is consumed here rather than double-handled.
    # A model-based fill is planned and applied before the tensor terms are
    # expanded, so a tensor margin over an imputed column reads the model's
    # values instead of its median.
    _model_imputed = None
    if na_action in ("impute_model", "impute_model_split"):
        formula, _model_imputed = impute_missing(expand_formula(formula, data), data,
                                                 learner="bols", model_based=True,
                                                 split_filled=na_action == "impute_model_split")
        data = apply_recipe(_model_imputed, data)
    _interactions = []
    if "%X%" in formula:
        from mboostrs.interactions import expand_interactions

        formula, _ix_env, _interactions = expand_interactions(
            formula, data, na_action="impute" if na_action in ("impute_model", "impute_model_split") else na_action)
        if _interactions:
            data = dict(data) if isinstance(data, dict) else {
                c: np.asarray(data[c]) for c in data.columns
            }
            env = dict(env) if env else {}
            _ix_rows = next(
                m.shape[0] for m in _ix_env.values() if m is not None
            )
            for name, matrix in _ix_env.items():
                if matrix is None:
                    # placeholder column the buser term names; the real design
                    # comes from its X= matrix, so any aligned column serves.
                    data[name] = np.zeros(_ix_rows)
                else:
                    env[name] = matrix

    # `y ~ 1` is offset-only: no base learner is fitted (see below).
    offset_only = _intercept_only(formula, data)
    formula = expand_formula(formula, data)
    # Imputation is planned on the formula before its transforms are
    # extracted, so a transform such as log(x) names x and its imputation
    # steps run before the transform reads x (a log of a missing value is
    # not finite).
    imputed = _model_imputed
    if na_action == "impute":
        # Median imputation plus a missing-value indicator term per
        # predictor with a missing value; replayed on new data.
        formula, imputed = impute_missing(formula, data, learner='bols')
    formula, _recipe = extract_transforms(formula, data)
    if imputed is not None:
        _recipe = TransformRecipe(imputed.steps + _recipe.steps)
        # The factor encoding below reads its levels from `data`, so a
        # categorical column's missing-value level must already be in place.
        data = apply_recipe(imputed, data)
    # `bols(f)` over a categorical column is mboost's dummy-coded learner;
    # it becomes the equivalent unpenalized level design before the factor
    # encoding below would refuse it. The term as the user wrote it is kept
    # first, because the learner it lowers to is named by its column alone.
    factor_labels = factor_term_labels(formula, data)
    formula = lower_categorical_bols(formula, data)
    # Which columns are factors, recorded before the encoding below turns them
    # into integer codes. A `by=` has to know: multiplying a design by level
    # codes reads them as a measurement, and nothing downstream can tell that
    # from a genuine numeric `by`.
    _factor_columns = _factor_columns_of(data)
    _recipe = TransformRecipe(
        _recipe.steps + encode_factors(formula, data, factor_bols=True).steps
    )
    _response_transform = None
    if _recipe:
        _recipe = fit_recipe(_recipe, data)
        data = apply_recipe(_recipe, data)
        # A transformed response rides in the recipe as the derived column
        # the fit reads; prediction data never carries it. Split it out: the
        # replay recipe keeps predictor steps only, and the model keeps the
        # step that inverts quantiles and carries the Jacobian.
        from mboostrs.formula import parse_response

        _response_name = parse_response(formula)
        if isinstance(_response_name, str):
            response_steps = [s for s in _recipe.steps
                              if s.name == _response_name]
            if response_steps:
                _response_transform = response_steps[0]
                _recipe = TransformRecipe(tuple(
                    s for s in _recipe.steps if s.name != _response_name))

    # Everything above is one-time preprocessing (formula rewriting, transform
    # materialization, interaction tensors). Capture the fully-processed inputs
    # so continue_to(N) can refit to a larger mstop without re-running any of
    # it — which would otherwise double-apply transforms or need the interaction
    # env rebuilt. See GAMBoostModel.continue_to.
    _refit_bundle = {
        "formula": formula,
        "data": data,
        "family": family,
        "weights": weights,
        "offset": offset,
        "env": env,
        "bins": bins,
        "recipe": None,          # already applied to `data`
        "interactions": _interactions or None,
        "subsample": subsample,
        "subsample_seed": subsample_seed,
        "na_action": na_action,
        "weight_policy": weight_policy,
    }
    _refit_bundle["raw_formula"] = _raw_formula
    _refit_bundle["raw_data"] = _raw_data
    _refit_bundle["raw_arguments"] = {
        "family": family, "control": control, "weights": weights,
        "offset": offset, "env": _raw_env, "learners": _raw_learners,
        "bins": bins, "subsample": subsample,
        "subsample_seed": subsample_seed, "na_action": na_action,
        "weight_policy": weight_policy,
    }

    response_name, formula_learners = parse_gamboost_formula(
        formula, env=env, data=data)
    if offset_only:
        # `Bols("1")` is the intercept as a fitted coefficient, which is what
        # gamboostLSS fits. gamboost() fits no learner for `y ~ 1`.
        formula_learners = []
    if learners is None:
        learners = formula_learners

    # `Brad` is a construction over `Buser` — a Matérn design against
    # space-filling centres, whitened so a plain ridge penalty applies. Lower it
    # here, once, so every `isinstance(spec, Buser)` branch below and every
    # downstream path (predict, cvrisk, stabsel, serialization) sees a learner
    # it already knows. That is the whole reason `brad` needed no engine work.
    learners = [
        learner.as_buser(data) if isinstance(learner, Brad) else learner
        for learner in learners
    ]
    # A tree splits a factor column into two sets of its levels; the recipe
    # has coded it 0..L-1 with the training vocabulary, and the learner needs
    # each column's level count.
    _factor_levels = {step.name: len(step.params[0]) for step in _recipe.steps
                      if step.kind == "factor"}
    for learner in learners:
        if isinstance(learner, Btree):
            learner.categorical = [_factor_levels.get(name, 0) for name in learner.names]

    # A factor `by=` has no single column to multiply the design by: R stacks
    # one copy of the design per non-reference level. That is a tensor with the
    # indicator columns, so those learners lower to buser terms here, next to
    # the `%X%` ones and carrying the same kind of recipe — which is what lets
    # `predict(newdata=)` rebuild their designs.
    from mboostrs.interactions import expand_by_factors

    learners, _by_recipes, _by_display = expand_by_factors(
        learners, data, _factor_columns
    )
    if _by_recipes:
        _interactions = list(_interactions) + _by_recipes
        _refit_bundle["interactions"] = _interactions

    _bin_decisions: list[dict] = []
    if bins is not None:
        from mboostrs.learners import apply_global_bins, warn_skipped_bins

        learners, skipped = apply_global_bins(learners, bins)
        warn_skipped_bins(skipped)

    if any(getattr(learner, "bins", None) == "auto" for learner in learners):
        from mboostrs._frame import column_names
        from mboostrs.learners import resolve_auto_bins

        available = set(column_names(data))
        distinct, n_rows = {}, 0
        for learner in learners:
            name = getattr(learner, "name", None)
            if name is None or name in distinct or name not in available:
                continue
            column = np.asarray(_extract_column(data, name), dtype=np.float64)
            n_rows = max(n_rows, int(column.size))
            distinct[name] = int(np.unique(column[np.isfinite(column)]).size)
        learners, _bin_decisions = resolve_auto_bins(learners, distinct, n_rows)

    is_survival = isinstance(family, CoxPH | WeibullFam | LoglogFam | LognormalFam)
    multinomial_n_classes = None

    if is_survival:
        if isinstance(response_name, str):
            raise ValueError(
                f"Survival family {family.__class__.__name__} requires a Surv(time, event) response, "
                f"got single variable {response_name!r}"
            )
        time_name, event_name = response_name
        time_arr = _extract_column(data, time_name).copy()
        event_arr = _extract_column(data, event_name)
        # R's Weibull/Loglog replace time==0 with sort(unique(time))[2]/10
        if isinstance(family, WeibullFam | LoglogFam) and np.any(time_arr == 0):
            nonzero = np.sort(np.unique(time_arr[time_arr > 0]))
            if len(nonzero) > 0:
                time_arr[time_arr == 0] = nonzero[0] / 10.0
        y = np.concatenate([time_arr, event_arr])
        n = len(time_arr)
    else:
        multinomial_n_classes = None
        is_binomial_cbind = _is_binomial_cbind(family, response_name)
        if isinstance(family, Multinomial):
            # For Multinomial, y can be categorical (strings/factors)
            y_raw = np.asarray(data[response_name])
            n = len(y_raw)
            # A missing label has no class to encode. Without this the value
            # reached `class_to_idx[nan]` and came back as `KeyError:
            # np.float64(nan)` -- an internal dictionary lookup escaping to
            # the caller, where every other family reports the column and the
            # row.
            if y_raw.dtype.kind in "fc":
                nonfinite = ~np.isfinite(y_raw.astype(np.float64))
                if nonfinite.any():
                    raise ValueError(_nonfinite_message(
                        f"Column '{response_name}'",
                        y_raw.astype(np.float64)))
            elif y_raw.dtype.kind == "O":
                missing = np.array([value is None or value != value
                                    for value in y_raw])
                if missing.any():
                    raise ValueError(
                        f"Column '{response_name}' contains "
                        f"{int(missing.sum())} missing label(s) out of "
                        f"{y_raw.size} (first at row "
                        f"{int(np.flatnonzero(missing)[0])}). A class label "
                        f"has to name a class; drop those rows first."
                    )
            classes = np.unique(y_raw)
            K = len(classes)
            if K < 3 and family.n_classes is None:
                raise ValueError(
                    f"Column '{response_name}' holds {K} distinct label(s) "
                    f"({', '.join(map(str, classes))}); Multinomial needs at "
                    f"least three classes. Two classes are a binary response: "
                    f"code them as 0/1 and fit family=Binomial()."
                )
            if family.n_classes is not None:
                if family.n_classes < K:
                    raise ValueError(
                        f"n_classes={family.n_classes} but column "
                        f"'{response_name}' holds {K} distinct labels "
                        f"({', '.join(map(str, classes[:6]))}"
                        f"{', ...' if K > 6 else ''}). Raise n_classes or "
                        f"check the response."
                    )
                K = family.n_classes
            else:
                # The count is inferred and kept on the *model's* family, not
                # written back to the caller's. Assigning to `family` here
                # meant a `Multinomial()` reused on a second dataset carried
                # the first one's class count, and a fourth label came back as
                # `IndexError: index 3 is out of bounds for axis 0 with size
                # 3` from inside NumPy.
                family = Multinomial(n_classes=K)
            multinomial_n_classes = K
            # The Kronecker bbs takes its knots from the training range; keep
            # that range on the spec, where new-data prediction reads it once
            # the training data is gone (a loaded artifact keeps no data).
            for learner in learners:
                if isinstance(learner, Bbs) and learner.boundary_knots is not None:
                    raise NotImplementedError(
                        f"bbs({quote_name(learner.name)}, boundary_knots=...) under Multinomial: "
                        "the Kronecker smooth places its knots over the training "
                        "range and would ignore the boundary. Drop boundary_knots.")
                if isinstance(learner, Bbs):
                    xtr = np.asarray(_extract_column(data, learner.name), dtype=np.float64)
                    learner.kronecker_boundary = (float(xtr.min()), float(xtr.max()))
            # Map y to 0..K-1 integer codes, sorted order
            class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
            y_codes = np.array([class_to_idx[yi] for yi in y_raw], dtype=np.int64)
            # Indicator matrix (n, K), drop last column (reference class = highest)
            indicator = np.eye(K, dtype=np.float64)[y_codes][:, :-1]
            y = np.ascontiguousarray(indicator.ravel(), dtype=np.float64)
            # bols and bbs base-learners are Kronecker-wrapped for the Multinomial
            # engine (bbs via the raw-basis / I+K1-penalty recipe matching R's
            # bbs %O% bols(class)). Other learners (btree/bmono/...) are not
            # supported and would silently mismatch R, so reject them up front.
            # A categorical predictor reaches here as `brandom` (a `bols` over
            # a factor is lowered to `brandom(f, lambda=0)`), which is wrapped
            # like `bols` over its level indicators.
            bad = sorted({ln.__class__.__name__ for ln in learners
                          if not isinstance(ln, (Bols, Bbs, Brandom))})
            if bad:
                raise NotImplementedError(
                    "Multinomial currently supports bols(), bbs() and brandom() "
                    f"base-learners; got {bad}. (R supports more via %O%, but the "
                    "Kronecker wrapping for those is not implemented here.)"
                )
            # The Kronecker expansion does not thread by=, so a by= term would
            # be silently dropped at both fit and prediction (verified: a by=
            # Multinomial fit is bit-identical to the no-by fit). Reject it
            # rather than mis-fit. Removing this requires by= support in the
            # Kronecker fit AND predict_newdata_kronecker paths.
            by_learners = sorted(
                {ln.name for ln in learners if getattr(ln, "by", None) is not None}
            )
            if by_learners:
                raise NotImplementedError(
                    "Multinomial does not support by= (varying-coefficient) "
                    f"learners; got by= on {by_learners}. The Kronecker "
                    "expansion silently ignores by=, so it is rejected here "
                    "rather than producing an incorrect fit."
                )
        elif is_binomial_cbind:
            y, total = _binomial_cbind_response(data, response_name)
            n = len(y)
            if weights is None:
                weights = total
            else:
                weights = np.asarray(weights, dtype=np.float64) * total
        else:
            y = _extract_column(data, response_name)
            n = len(y)

    offset = _prepare_offset(offset, n, family)

    if weights is not None and not np_all_finite(np.asarray(weights)):
        raise ValueError(
            _nonfinite_message("Fitting weights", np.asarray(weights, dtype=np.float64).ravel())
        )
    normalized = normalize_weights(weights, n, policy=weight_policy)
    w = normalized.fit

    family_name = _family_rust_name(family)
    family_params = _family_params(family)

    # `learner_names` and `learner_types` label the terms in every summary,
    # plot, and coefficient mapping, so they are derived directly from the
    # learners rather than accumulated by the array-building block below.
    # What the user sees: coefficient keys, `selected`, `which=`, plot labels.
    # A term written `bbs(log(x))` reports as `log(x)`, the expression, not as
    # the derived column `log__x` it became. R does the same. The learners
    # themselves keep the derived name, because that is the column they read.
    raw_learner_names = [
        learner.names[0] if isinstance(learner, Btree) else learner.name
        for learner in learners
    ]
    learner_names = _display_names(_recipe, raw_learner_names)
    # A categorical column's learner is labelled with the term as written
    # (`bols(f)`, `brandom(f)`, `f`), whatever the number of levels.
    learner_names = [factor_labels.get(raw, name)
                     for raw, name in zip(raw_learner_names, learner_names, strict=True)]
    # A lowered factor-`by` learner and a `%X%` tensor report as the term the
    # user wrote rather than as the derived buser column they became.
    _term_display = dict(_by_display)
    _term_display.update({
        interaction.derived: interaction.name
        for interaction in _interactions
        if interaction.derived not in _term_display
    })
    learner_names = [_term_display.get(name, name) for name in learner_names]
    # The observed and filled parts of a switched impute_model term share a
    # column name; they are labelled by part, e.g. "bbs(x) [observed]".
    from mboostrs.transforms import switched_term_labels

    _switch_labels = switched_term_labels(learners, _recipe)
    learner_names = [_switch_labels.get(position, name)
                     for position, name in enumerate(learner_names)]
    learner_types = [_engine_kind_for_spec(learner) for learner in learners]

    # A bkernel learner builds a dense n x n kernel matrix. Warn about the
    # memory before fitting, whichever boundary the fit takes. A Nyström
    # fit (landmarks set) builds n x m instead, so no warning.
    for learner in learners:
        if isinstance(learner, Bkernel) and learner.landmarks is None:
            _warn_bkernel_scale(learner.name, n)

    # Every fit crosses one compact, self-describing learner-spec boundary.
    # Multinomial bbs terms lower to KroneckerBbs blueprints so their raw-basis
    # and I+K1 penalty recipe remains identical to the former array path.
    from mboostrs.gamlss import learner_spec

    kronecker_n_resp = (
        (multinomial_n_classes - 1) if multinomial_n_classes is not None else None
    )
    learner_layout = LearnerLayout.from_mapping(range(len(learners)))
    rust_idx_to_formula_idx = list(learner_layout.rust_idx_to_formula_idx)
    (
        offset_val,
        selected,
        coefs_vecs,
        risk,
        fitted,
        nuisance_vec,
        nuisance_vec_path,
        learner_store,
        execution_report,
        residual,
    ) = call_fit(_fit_gamboost_specs,
        [
            learner_spec(
                learner,
                data,
                # A per-observation offset freezes the constant the centered
                # form borrows from the model offset, so every bols carries
                # its own: R's design, where the intercept column is fitted
                # on top of the offset the caller supplied.
                bols_form=_bols_form_for(
                    learners, vector_offset=_is_vector_offset(offset),
                    kronecker=kronecker_n_resp is not None, family=family),
                kronecker_n_resp=kronecker_n_resp,
            )
            for learner in learners
        ],
        y,
        w,
        mstop,
        nu,
        family_name,
        family_params,
        offset,
        multinomial_n_classes,
        risk_str,
        trace_bool,
        subsample,
        subsample_seed,
        **step_kwargs(control),
    )
    # A per-observation fit reports the scalar start it never used; the model
    # keeps the caller's vector, which every prediction and refit must supply.
    if _is_vector_offset(offset):
        offset_val = offset

    execution_report = ExecutionReport.from_mapping(execution_report)
    if trace_bool:
        print(execution_report.format_text(), file=sys.stderr)

    selected_formula = [learner_layout.rust_to_formula(s) for s in selected]
    formula_idx_to_rust_idx = dict(learner_layout.formula_idx_to_rust_idx)
    coefficients = [np.array(c, dtype=np.float64) for c in coefs_vecs]

    # Collect brandom levels data for prediction
    brandom_levels_data = {}
    for spec in learners:
        if isinstance(spec, Brandom):
            col = _extract_column(data, spec.name)
            brandom_levels_data[spec.name] = col.copy()

    # Collect Buser x data and centering means for prediction
    buser_x_data = {}
    buser_centering_means = {}
    for spec in learners:
        if isinstance(spec, Buser):
            buser_x_data[spec.name] = spec.X
            if spec.center:
                sw = float(np.sum(w)) if w is not None else float(spec.X.shape[0])
                means = np.sum(spec.X * (w[:, None] if w is not None else 1.0), axis=0) / sw
                buser_centering_means[spec.name] = means.tolist()
            else:
                buser_centering_means[spec.name] = [0.0] * spec.X.shape[1]

    comparison_identity = fit_identity(
        data=data, response_name=response_name, response_transform=_response_transform,
        weights=w, weight_policy=normalized.policy, dropped=_na_dropped, family=family)
    _model = GAMBoostModel(
        formula=formula,
        family=family,
        control=control,
        response_name=response_name,
        learner_names=learner_names,
        learner_types=learner_types,
        fitted=np.asarray(fitted, dtype=np.float64),
        risk=np.asarray(risk, dtype=np.float64),
        selected=selected_formula,
        coefficients=coefficients,
        offset=offset_val,
        nu=nu,
        data=data,
        weights=w,
        learners_spec=learners,
        rust_idx_to_formula_idx=rust_idx_to_formula_idx,
        formula_idx_to_rust_idx=formula_idx_to_rust_idx,
        buser_centering_means=buser_centering_means if buser_centering_means else None,
        family_sigma=nuisance_vec if nuisance_vec else None,
        family_sigma_path=nuisance_vec_path if nuisance_vec_path else None,
        learner_store=learner_store,
        execution_report=execution_report,
        residual=residual,
    )
    _model._transform_recipe = _recipe if _recipe else None
    _model._response_transform = _response_transform
    _model._interactions = _interactions or None
    _model._refit_bundle = _refit_bundle
    _model._na_dropped = _na_dropped
    _model._comparison_identity = comparison_identity
    _model._bin_decisions = _bin_decisions
    _model._set_weight_policy(normalized)
    _model._breslow_baseline = _breslow_baseline_of(_model)
    from mboostrs._model_api import warn_if_penalty_drifted, warn_if_risk_rose

    warn_if_risk_rose(_model)
    warn_if_penalty_drifted(_model)
    _model._global_intercept = _global_intercept
    return _model
