"""Stability selection for mboostrs models.

Reimplements the stabsel algorithm from R's stabs/mboost packages:
1. Generate B half-sample subsampling folds
2. For each fold, refit the model and record selected() at each mstop
3. Compute selection probabilities (phat) per base-learner per mstop
4. Apply cutoff to determine stable selections

Supports both gamboost and glmboost models.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mboostrs._control import as_control, step_kwargs
from mboostrs._output import export_result
from mboostrs._results import MappingResultMixin, provenance_from_model
from mboostrs._validate import check_fit_budget
from mboostrs.cvrisk import _as_fold_matrix

try:
    from mboostrs._rust import (
        cvrisk_selected_gamboost_specs as _cvrisk_selected_gamboost_specs,
    )
    from mboostrs._rust import cvrisk_selected_glmboost as _cvrisk_selected_glmboost
    _HAS_SELECTED_ENGINE = True
except ImportError:
    _HAS_SELECTED_ENGINE = False
    _cvrisk_selected_gamboost_specs = None

try:
    from mboostrs._rust import (
        stabsel_d_rc as _rust_d_rc,
    )
    from mboostrs._rust import (
        stabsel_max_q as _rust_max_q,
    )
    from mboostrs._rust import (
        stabsel_min_d as _rust_min_d,
    )
    from mboostrs._rust import (
        stabsel_params as _rust_stabsel_parameters,
    )
    from mboostrs._rust import (
        stabsel_um_const as _rust_um_const,
    )
    _HAS_RUST_STABSEL = True
except ImportError:
    _HAS_RUST_STABSEL = False

if TYPE_CHECKING:
    pass


def _require_rust_stabsel() -> None:
    """The stabsel parameter math lives in the Rust core (mboostrs._rust).

    These functions used to carry pure-Python/scipy fallbacks, but the Rust
    implementations are the single source of truth (verified bit-exact to R's
    stabs package). The compiled extension is part of the package, so it is
    always present in a real install; if it somehow is not, fail clearly rather
    than silently diverge (and avoid pulling in scipy, which is not a dependency).
    """
    if not _HAS_RUST_STABSEL:
        raise RuntimeError(
            "stability-selection parameter functions require the compiled "
            "mboostrs Rust engine (mboostrs._rust), which is unavailable."
        )


def _maxQ(p: int, B: int) -> int:
    _require_rust_stabsel()
    return int(_rust_max_q(p, B))


def _um_const(cutoff: float, B: int, theta: float) -> float:
    _require_rust_stabsel()
    return _rust_um_const(cutoff, B, theta)


def _D_rc(theta: float, which: int, B: int, r: float) -> float:
    """R's stabs::D function for the r-concave tail bound (Rust core).

    Worst-case P(V >= which) over r-concave discrete distributions on
    {0,...,B} with E[V] = theta * B. ``r`` is the r-concavity order
    (negative, e.g. -0.5 or -0.25).
    """
    _require_rust_stabsel()
    return _rust_d_rc(theta, which, B, r)


def _minD(q: int, p: int, cutoff: float, B: int) -> float:
    """R's stabs::minD r-concave PFER bound (Rust core)."""
    _require_rust_stabsel()
    return _rust_min_d(q, p, cutoff, B)


def _stabsel_parameters(p, cutoff=None, q=None, PFER=None, B=50,
                        assumption="unimodal", sampling_type="SS",
                        verbose=False):
    if _HAS_RUST_STABSEL and cutoff is not None and q is not None:
        res = _rust_stabsel_parameters(p=p, cutoff=cutoff, q=q, b=B,
                                       assumption=assumption,
                                       sampling_type=sampling_type)
        return {"cutoff": float(res["cutoff"]),
                "q": int(res["q"]),
                "PFER": float(res["PFER"]),
                "p": p, "B": B,
                "sampling_type": sampling_type,
                "assumption": res["assumption"]}
    if _HAS_RUST_STABSEL and cutoff is not None and PFER is not None:
        res = _rust_stabsel_parameters(p=p, cutoff=cutoff, pfer=PFER, b=B,
                                       assumption=assumption,
                                       sampling_type=sampling_type)
        return {"cutoff": float(res["cutoff"]),
                "q": int(res["q"]),
                "PFER": float(res["PFER"]),
                "p": p, "B": B,
                "sampling_type": sampling_type,
                "assumption": res["assumption"]}
    if _HAS_RUST_STABSEL and q is not None and PFER is not None:
        res = _rust_stabsel_parameters(p=p, q=q, pfer=PFER, b=B,
                                       assumption=assumption,
                                       sampling_type=sampling_type)
        return {"cutoff": float(res["cutoff"]),
                "q": int(res["q"]),
                "PFER": float(res["PFER"]),
                "p": p, "B": B,
                "sampling_type": sampling_type,
                "assumption": res["assumption"]}
    n_miss = sum(x is None for x in [cutoff, q, PFER])
    if n_miss != 1:
        if n_miss > 1:
            # The two-of-three requirement stays at mboost parity (#506's
            # recorded decision); the message guides the caller to a pair
            # instead of leaving the trap that cost a failed call and a
            # retry turn in the D5 runs.
            mboost_q = max(1, min(int(round((0.8 * p) ** 0.5)), p - 1))
            suggestion = (f"the mboost-style starting point is q={mboost_q} "
                          f"(about sqrt(0.8 * p) for p={p}) with PFER=1, "
                          f"e.g. stabsel(model, q={mboost_q}, PFER=1); "
                          if p >= 2 else "")
            raise ValueError(
                "Two of cutoff, q, and PFER must be specified; the third is "
                f"computed from the pair. This model has {p} candidate "
                f"effects; {suggestion}"
                "the usual alternative is stabsel(model, cutoff=0.75, "
                "PFER=1). No value defaults: this matches mboost.")
        if n_miss < 1:
            raise ValueError("Only two of cutoff, q, and PFER can be specified at the same time")

    if PFER is not None and PFER < 0:
        raise ValueError("PFER must be greater than 0")
    if cutoff is not None and assumption != "r-concave" and (cutoff < 0.5 or cutoff > 1):
        raise ValueError("cutoff must be between 0.5 and 1")
    if cutoff is not None and assumption == "r-concave" and (cutoff < 0 or cutoff > 1):
        raise ValueError("cutoff must be between 0 and 1")
    if q is not None:
        if p < q:
            raise ValueError(f"q ({q}) must be smaller than the number of effects ({p})")
        if q < 0:
            raise ValueError("q must be greater than 0")

    # Every supported (cutoff, q, PFER) pair is computed by the Rust core in the
    # fast-paths above, and invalid argument counts/ranges were rejected by the
    # validation above. Reaching here means the compiled engine is unavailable.
    _require_rust_stabsel()
    raise AssertionError("unreachable: stabsel_parameters fast-paths exhausted")


@dataclass
class StabSelResult(MappingResultMixin):
    """Stability-selection frequencies, selected learners, and error bound.

    ``B`` is the number of initial subsamples (or complementary pairs under
    ``sampling_type="SS"``); ``n_fits`` is the number of models actually fit.
    ``PFER`` is an upper bound on the expected number of falsely selected
    learners, not a probability attached to any one learner.
    """

    #: Selection frequency per learner and iteration: the fraction of
    #: subsamples in which the learner had been selected by that iteration.
    #: An empirical frequency, not the probability that the learner belongs
    #: in the model -- stability selection makes no such statement.
    phat: np.ndarray
    selected: list[int]
    #: Each learner's largest ``phat`` along the path.
    max_: np.ndarray
    cutoff: float
    q: int
    #: Upper bound on the expected *number* of falsely selected learners. A
    #: count, not a probability and not a rate, despite "error rate" in the
    #: name it carries in the literature. It bounds learners, so two learners
    #: on the same source variable count twice.
    PFER: float
    p: int
    #: Initial subsamples the error bound was computed on: complementary
    #: *pairs* under ``sampling_type="SS"``, independent half-samples under
    #: ``"MB"``. This is the caller's ``B``, not the number of fits.
    B: int
    sampling_type: str
    assumption: str
    learner_names: list[str]
    #: Models actually fitted: ``2 * B`` under ``sampling_type="SS"``, because
    #: complementary-pairs sampling needs each half-sample and its complement
    #: for the Shah-Samworth error bound, and ``B`` under ``"MB"``.
    n_fits: int = 0
    #: The model, family, row count, budget and weight policy the run was
    #: performed under. ``result.describe()`` prints it.
    provenance: object = None
    #: Digest of the response and weights the subsample fits used, as
    #: `CvRiskResult.data_fingerprint`; empty when the model kept no data.
    data_fingerprint: str = ""
    #: The learner indices each subsample fit selected along its path, one
    #: sorted tuple per fit, after the q-truncation that ``phat`` uses. Empty
    #: when the run did not keep them. `substitution_sets` reads these.
    fit_selected: tuple = ()

    _MAPPING_KEYS = ("phat", "selected", "max_", "cutoff", "q", "PFER", "p",
                     "B", "sampling_type", "assumption", "learner_names",
                     "n_fits", "max", "names")
    # `max` is a property here, not an alias: it pairs the values with the
    # learner names, which `max_` alone does not.
    _MAPPING_ALIASES = {"names": "learner_names"}

    @property
    def max(self):
        """Each learner's largest ``phat``, keyed by name."""
        return {name: float(value)
                for name, value in zip(self.learner_names, self.max_,
                                       strict=False)}

    @property
    def names(self):
        return self.learner_names

    @property
    def selected_names(self):
        """The selected learners by name (``selected`` holds their indices)."""
        return [self.learner_names[i] for i in self.selected]

    @property
    def variables(self):
        """The data columns the selected learners read, in learner order."""
        from mboostrs.selection import _variables

        return _variables(self.selected_names)

    def __repr__(self):
        names = [self.learner_names[i] for i in self.selected]
        sel_str = ", ".join(names) if names else "(none)"
        return (f"StabSelResult(selected=[{sel_str}], "
                f"cutoff={self.cutoff:.2f}, q={self.q}, PFER={self.PFER:.3f}, "
                f"n_fits={self.n_fits})")

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; the selection-probability paths only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("phat", "max_"),
            extras={"max": self.max, "selected_names": self.selected_names,
                    "variables": list(self.variables)},
        )

    def plot(self):
        from mboostrs.plot import plot_stabsel
        return plot_stabsel(self)

    def to_frame(self, output="auto"):
        """One row per candidate learner: name, max selection probability
        over the subsampling path, and whether it was selected.
        """
        from mboostrs.gamlss import _records_frame

        selected_set = set(self.selected)
        rows = [
            {
                "learner": self.learner_names[i],
                "max_phat": float(self.max_[i]),
                "selected": i in selected_set,
            }
            for i in range(self.p)
        ]
        return _records_frame(rows, output,
                              columns=("learner", "max_phat", "selected"))


def _subsample_folds(n: int, B: int, weights=None, seed: int = 42) -> np.ndarray:
    """``B`` independent half-sample in-bag weight columns, ``n x B``.

    Each column carries the model's weights on a random half of the rows and
    zero on the rest. This is the raw draw; complementary pairing is applied
    separately by :func:`_append_complements`.
    """
    rng = np.random.RandomState(seed)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    k = n // 2
    folds = np.zeros((n, B), dtype=np.float64)
    for b in range(B):
        idx = np.zeros(n, dtype=np.float64)
        idx[:k] = 1.0
        rng.shuffle(idx)
        folds[:, b] = idx * w
    return folds


def _append_complements(folds: np.ndarray, weights=None) -> np.ndarray:
    """Append each column's exact complement, giving ``n x 2B``.

    Column ``b`` and column ``B + b`` sum elementwise to the full weight
    vector, which is what complementary-pairs stability selection and the
    Shah-Samworth bound are defined on. Matches
    ``folds <- cbind(folds, weights - folds)`` in ``stabs::stabsel``.
    """
    folds = np.asarray(folds, dtype=np.float64)
    n = folds.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape[0] != n:
        raise ValueError(
            f"weights has length {w.shape[0]} but folds has {n} rows"
        )
    complement = w.reshape(-1, 1) - folds
    if np.any(complement < -1e-9):
        row, col = np.unravel_index(int(np.argmin(complement)), complement.shape)
        raise ValueError(
            "folds hold more weight than the model does, so the complement "
            f"would be negative: fold column {col} puts {folds[row, col]:g} on "
            f"row {row}, where the model's weight is {w[row]:g}. Build the "
            "folds from the model's effective weights (`model.weights`), which "
            "are the rescaled vector the fit used, not from the vector passed "
            "to `weights=`."
        )
    return np.hstack([folds, np.maximum(complement, 0.0)])


def make_folds(
    n: int,
    B: int,
    sampling_type: str = "SS",
    weights=None,
    seed: int = 42,
) -> np.ndarray:
    """The in-bag weight matrix a stability-selection run actually fits.

    ``sampling_type="SS"`` draws ``B`` half-samples and appends their exact
    complements, so the result has ``2 * B`` columns and column ``b`` pairs
    with column ``B + b``. ``sampling_type="MB"`` draws ``B`` independent
    half-samples and returns them unchanged.

    ``B`` counts initial subsamples — pairs under ``"SS"`` — and is the value
    the error bound is computed on. The column count is the number of model
    fits the run performs.

    This is the one fold constructor for every stability-selection entry
    point in the package: standard models, LSS models, and the R frontend.
    """
    if sampling_type not in ("SS", "MB"):
        raise ValueError(
            f"sampling_type must be 'SS' or 'MB', got {sampling_type!r}"
        )
    if B < 1:
        raise ValueError(f"B must be at least 1, got {B}")
    folds = _subsample_folds(n, B, weights=weights, seed=seed)
    if sampling_type == "SS":
        folds = _append_complements(folds, weights)
    return folds


def stabsel_fit_count(B: int, sampling_type: str = "SS") -> int:
    """Model fits a stability-selection run performs for ``B`` initial subsamples.

    ``"SS"`` fits each complementary pair, so ``2 * B``; ``"MB"`` fits each
    subsample once, so ``B``. This is the column count `make_folds` returns.
    """
    if sampling_type not in ("SS", "MB"):
        raise ValueError(
            f"sampling_type must be 'SS' or 'MB', got {sampling_type!r}"
        )
    return 2 * int(B) if sampling_type == "SS" else int(B)


def stabsel_planned_fits(B, folds, sampling_type: str = "SS") -> int:
    """Fits a stability-selection call will perform, from its arguments alone.

    With caller-supplied ``folds`` the initial subsample count is the matrix's
    column count, whatever ``B`` says; without them it is ``B``, and when
    ``B`` is None the default of `stabsel` (50 under ``"SS"``, 100 under
    ``"MB"``).
    """
    if folds is not None:
        initial = int(np.asarray(folds).shape[1])
    elif B is None:
        initial = 100 if sampling_type == "MB" else 50
    else:
        initial = int(B)
    return stabsel_fit_count(initial, sampling_type)


def _pair_count_phrase(B: int, n_fits: int, sampling_type: str) -> str:
    """``B`` and ``n_fits`` in one clause, so a warning names both counts."""
    if sampling_type == "SS":
        return f"B={B} complementary pairs, {n_fits} fits"
    return f"B={B} subsamples, {n_fits} fits"


def _fit_matrix(folds, sampling_type: str, weights=None) -> np.ndarray:
    """Turn caller-supplied ``folds`` into the matrix that will be fitted.

    Caller-supplied ``folds`` are always the *initial* subsamples: ``B``
    columns, one per subsample under ``"MB"`` and one per pair under ``"SS"``.
    Under ``"SS"`` the complements are appended here, so a caller passing
    ``B`` columns gets ``2 * B`` fits. Pairing is never inferred from an even
    column count.
    """
    folds = np.ascontiguousarray(np.asarray(folds, dtype=np.float64))
    if folds.ndim != 2:
        raise ValueError("folds must be a 2-dimensional n x B weight matrix")
    if sampling_type == "SS":
        folds = np.ascontiguousarray(_append_complements(folds, weights))
    return folds


def stabsel(
    model,
    cutoff=None,
    q=None,
    PFER=None,
    B=None,
    folds=None,
    assumption="unimodal",
    sampling_type="SS",
    grid=None,
    verbose=True,
    eval=True,
    n_jobs=-1,
    fit_budget=None,
):
    """Stability selection for mboostrs models.
    Usage::

        import mboostrs as mb
        model = mb.glmboost("y ~ x + z", data, control=mb.boost_control(50))
        result = mb.stabsel(model, q=5, PFER=1)
        result.selected_names

    Specify two of ``cutoff``, ``q`` and ``PFER``. The third is computed
    from the theoretical error bounds. For example,
    ``stabsel(model, cutoff=0.75, PFER=1)`` specifies the cutoff and the
    expected-false-learners bound.

    ``fit_budget`` is the largest number of model fits the call may perform.
    The count is `stabsel_fit_count` of the initial subsample count (``2 * B``
    under ``sampling_type="SS"``, ``B`` under ``"MB"``), and the call refuses
    before any subsample is drawn when the count is larger. ``None`` (the
    default) does not check.

    ``PFER`` bounds the expected *number* of falsely selected learners from
    above. It is a count, not a probability and not a rate, and it bounds
    *learners*: a menu such as ``bols(x) + bbs(x)`` gives one source variable
    two candidates, and a PFER of 1 permits one false learner among them, not
    one false variable.

    The reported ``phat`` is the fraction of subsamples in which a learner was
    selected -- an empirical frequency. It is not the probability that the
    learner belongs in the model, and stability selection makes no such
    statement.

    ``B`` is the number of initial subsamples or complementary pairs used in
    the error bound. Under ``sampling_type="SS"``, each pair fits both halves,
    for ``2 * B`` fits; ``sampling_type="MB"`` fits ``B`` subsamples. The
    result reports the actual number of fits as ``n_fits``.

    ``folds`` is an ``n x B`` in-bag weight matrix of *initial* subsamples,
    the same convention as ``stabs::stabsel``. Under ``"SS"`` the complements
    are constructed from the model's weights and appended, so supplying ``B``
    columns produces ``2 * B`` fits; pairing is never inferred from an even
    column count.

    ``n_jobs`` sets the worker threads for the per-fold refits (-1 = all CPUs,
    1 = sequential). The refits release the GIL for both learner preparation
    and boosting, so threads give real parallelism; the selection paths are
    computed per fold and do not depend on ``n_jobs``.
    """
    from mboostrs._refit import refit_available
    from mboostrs.families import Multinomial
    from mboostrs.glmboost import GLMBoostModel

    can_refit, reason = refit_available(model)
    if not can_refit:
        # Every fold is a refit, so a model with no record of its own call
        # cannot do this. It used to reach `_extract_column` on `None` and
        # report `Cannot extract column 'y' from <class 'NoneType'>`.
        raise ValueError(f"stabsel is unavailable: {reason}")

    is_glmboost = isinstance(model, GLMBoostModel)

    p = len(model._learner_names)
    learner_names = model._learner_names

    if sampling_type == "MB":
        assumption = "none"
    else:
        if assumption not in ("unimodal", "r-concave", "none"):
            raise ValueError("assumption must be one of 'unimodal', 'r-concave', 'none'")

    if B is None and folds is None:
        B = 100 if sampling_type == "MB" else 50
    elif folds is not None:
        if B is not None and B != folds.shape[1]:
            warnings.warn(
                "B should equal number of folds; B set to ncol(folds)", stacklevel=2
            )
        B = folds.shape[1]

    pars = _stabsel_parameters(
        p=p, cutoff=cutoff, q=q, PFER=PFER, B=B,
        assumption=assumption, sampling_type=sampling_type,
        verbose=verbose,
    )

    if not eval:
        return pars
    planned = stabsel_planned_fits(B, folds, sampling_type)
    check_fit_budget(
        planned, fit_budget, "stabsel",
        detail=f" ({_pair_count_phrase(planned // 2 if sampling_type == 'SS' else planned, planned, sampling_type)})",
        remedy="lower B, or pass fewer folds")

    cutoff_val = pars["cutoff"]
    q_val = pars["q"]
    PFER_val = pars["PFER"]

    is_multinomial = isinstance(getattr(model, "_family", None), Multinomial)
    n = (
        len(model._weights)
        if model._weights is not None
        else (
            len(model._fitted) // (model._family.n_classes - 1)
            if is_multinomial
            else len(model._fitted)
        )
    )
    w_full = model._weights if model._weights is not None else np.ones(n)

    if folds is None:
        fold_seed = 42
        folds = make_folds(n, B, sampling_type=sampling_type, weights=w_full,
                           seed=fold_seed)
        fold_scheme = (f"{B} half-sample pairs" if sampling_type == "SS"
                       else f"{B} half-samples")
    else:
        # A caller's matrix: its scheme is the caller's business and this
        # function cannot name it.
        fold_seed = None
        folds = _fit_matrix(folds, sampling_type, weights=w_full)
        fold_scheme = f"{B} supplied subsamples"
    n_fits = folds.shape[1]

    # Grid of mstop values
    if grid is None:
        m = model.mstop
        grid = list(range(0, m + 1))
    else:
        if grid != list(range(0, max(grid) + 1)):
            raise ValueError("grid must be of the form 0:m")
        m = max(grid)

    # Get selected sequences for each fold
    all_selected = _selected_paths(
        model, folds, n_fits, is_glmboost, n_jobs, q_val, m
    )

    # Apply q-truncation: after more than q unique learners appear, replace
    # all remaining selections with the first learner.
    # Matches R's: qq <- sapply(1:length(xs), function(x) length(unique(xs[1:x])))
    #              xs[qq > q] <- xs[1]
    all_selected_trunc = []
    for sel in all_selected:
        if not sel:
            all_selected_trunc.append(sel)
            continue
        truncated = []
        seen = set()
        threshold_reached = False
        for idx in sel:
            if threshold_reached:
                truncated.append(sel[0])
            else:
                if idx not in seen:
                    seen.add(idx)
                    if len(seen) > q_val:
                        threshold_reached = True
                        truncated.append(sel[0])
                    else:
                        truncated.append(idx)
                else:
                    truncated.append(idx)
        all_selected_trunc.append(truncated)

    # Build phat matrix (p x m) — matches R's stabsel.mboost logic
    phat = np.zeros((p, m))
    for sel in all_selected_trunc:
        if not sel:
            continue
        for learner_idx in range(p):
            try:
                first_sel = sel.index(learner_idx)
            except ValueError:
                first_sel = m
            phat[learner_idx, first_sel:] += 1
    phat /= len(all_selected_trunc)

    max_phat = phat.max(axis=1)
    selected_indices = [i for i in range(p) if max_phat[i] >= cutoff_val]

    if verbose:
        uniques = [
            len(set(sel[:m])) if len(sel) >= m else len(set(sel))
            for sel in all_selected_trunc
        ]
        short = sum(1 for n_unique in uniques if n_unique < q_val)
        if short:
            warnings.warn(
                "mstop too small in some subsampling replicates to select "
                f"q={q_val} base-learners; Increase mstop before applying stabsel. "
                f"{short} of {len(uniques)} fits fell short "
                f"({_pair_count_phrase(B, n_fits, sampling_type)}) and the best "
                f"reached {max(uniques, default=0)} distinct learners. A model "
                f"whose path keeps re-selecting the same few terms may not "
                f"reach q={q_val} at any mstop, in which case lower q instead: "
                "a fold that never reaches q also never stops early, so the "
                "run costs the full mstop.",
                stacklevel=2,
            )

    return StabSelResult(
        phat=phat,
        selected=selected_indices,
        max_=max_phat,
        cutoff=cutoff_val,
        q=q_val,
        PFER=PFER_val,
        p=p,
        B=B,
        sampling_type=sampling_type,
        assumption=assumption,
        learner_names=learner_names,
        n_fits=n_fits,
        provenance=provenance_from_model(
            model, "stabsel", seed=fold_seed, folds=fold_scheme,
            n_fits=n_fits,
        ),
        data_fingerprint=(model._data_fingerprint()
                          if hasattr(model, "_data_fingerprint") else ""),
        fit_selected=tuple(tuple(sorted(set(sel))) for sel in all_selected_trunc),
    )


def _iters_to_q(selected, q, default):
    """Iterations of ``selected`` up to and including the q-th distinct entry.

    ``default`` when q distinct entries never appear.
    """
    seen = set()
    for i, learner_idx in enumerate(selected):
        seen.add(learner_idx)
        if len(seen) >= q:
            return i + 1
    return default


def _selected_paths(model, folds, n_fits, is_glmboost, n_jobs, q_val, m):
    """The per-fold selection paths stabsel needs, computed no further than that.

    Only the first ``q`` distinct learners of a fold's path reach ``phat``: the
    q-truncation rewrites every entry from the (q+1)-th distinct one onward, and
    ``phat`` records each learner's *first* appearance, so a repeat changes
    nothing either. A fold's path can therefore stop at the iteration where its
    q-th distinct learner enters, which on a wide model is a small fraction of
    ``mstop`` (roughly 60 of 2455 iterations for 80 learners at nu = 0.1).

    That iteration is not known in advance. The first attempt uses half again
    what the full-data path needed, which is the closest available estimate of
    what a half-sample needs, and folds that have not reached q distinct
    learners are refitted with mstop quadrupled until they do or the model's
    mstop is reached. Selection paths are a deterministic function of mstop --
    iteration k picks the same learner whatever the stopping iteration is -- so
    the result is identical to fitting every fold to the full mstop, and the
    factor only trades an occasional extra refit against a smaller first
    attempt.

    The 1.5 comes from measuring it. Over 120 model shapes spanning n, learner
    count, q, nu, predictor correlation, and seed -- restricted to the shapes
    where the multiplier rather than the 32 floor decides the budget -- the
    largest a fold needed was 1.41 times the full-data path, the 95th
    percentile was 1.23, and the median was 0.98: a half-sample typically
    reaches its q-th distinct learner slightly *sooner* than the full sample
    does. The factor was 2, which no observed shape came close to needing.

    How much this saves depends entirely on how far ``mstop`` sits above what
    the folds need. On a 20,000-row, twelve-learner Huber fit needing 129
    iterations to reach q = 6, stability selection at ``mstop=2000`` runs 3.8
    times faster than fitting every fold to mstop; at ``mstop=300`` it runs at
    the same speed, because the first attempt is already most of the path.
    """
    paths = [None] * n_fits
    pending = list(range(n_fits))
    full_data_iters = _iters_to_q(list(model._selected), q_val, m)
    m_try = min(m, max(int(np.ceil(1.5 * full_data_iters)), 32))
    while pending:
        fold_block = folds[:, pending]
        if is_glmboost and _HAS_SELECTED_ENGINE:
            block = _run_glmboost_selected(model, fold_block, m_try)
        elif _HAS_SELECTED_ENGINE and any(
            getattr(spec, "bins", None) is not None
            for spec in getattr(model, "_learners_spec", [])
        ):
            block = _run_gamboost_specs_selected(model, fold_block, m_try)
        else:
            block = _run_refit_selected(model, fold_block, is_glmboost, n_jobs, m_try)
        short = []
        for k, b_idx in enumerate(pending):
            paths[b_idx] = block[k]
            if len(set(block[k])) < q_val:
                short.append(b_idx)
        if not short:
            break
        if m_try >= m:
            break
        pending = short
        m_try = min(m, m_try * 4)
    return paths


def _run_gamboost_specs_selected(model, folds, mstop_override):
    """Selection-only fold paths through the compact blueprint boundary."""
    from mboostrs.families import Binomial, Multinomial, _family_params, _family_rust_name
    from mboostrs.gamboost import _multinomial_response
    from mboostrs.glmboost import _cbind_proportion

    is_multinomial = isinstance(model._family, Multinomial)
    n_classes = model._family.n_classes if is_multinomial else None
    n = (
        len(model._weights)
        if model._weights is not None
        else (
            len(model._fitted) // (n_classes - 1)
            if is_multinomial
            else len(model._fitted)
        )
    )
    if is_multinomial:
        y = _multinomial_response(model)
    elif isinstance(model._family, Binomial) and isinstance(model._response_name, tuple) \
            and len(model._response_name) == 2:
        # A cbind fit's weights already carry the trial counts, so the
        # proportion alone rebuilds the response the fit was run on.
        y = _cbind_proportion(model._data, model._response_name)
    elif isinstance(model._response_name, tuple):
        first = _extract_col(model._data, model._response_name[0])
        second = _extract_col(model._data, model._response_name[1])
        y = np.concatenate([first, second])
    else:
        y = _extract_col(model._data, model._response_name)
    weights = model._weights if model._weights is not None else np.ones(n)
    nu = as_control(model._control).nu
    from mboostrs.glmboost import _is_vector_offset

    vector_offset = _is_vector_offset(model._offset)
    selected = _cvrisk_selected_gamboost_specs(
        model._fit_learner_specs(model._data, vector_offset=vector_offset),
        np.asarray(y, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        int(mstop_override),
        nu,
        _family_rust_name(model._family),
        _family_params(model._family),
        _as_fold_matrix(folds, n, model._weight_policy, model._weights),
        # A per-row offset (log exposure) starts every subsample's fit, as it
        # started the full fit; a fitted scalar is re-estimated per subsample.
        model._offset if vector_offset else None,
        n_classes,
        **step_kwargs(model._control),
    )
    return np.asarray(selected, dtype=np.intp).tolist()


def _run_glmboost_selected(model, folds, mstop_override=None):
    """Use Rust engine to get selected() per fold for glmboost."""
    from mboostrs.families import Binomial, _family_params, _family_rust_name
    from mboostrs.glmboost import _cbind_proportion

    n = len(model._fitted)
    # Survival families (CoxPH/Gehan/Cindex/Weibull/...) store a
    # (time_col, event_col) tuple response name; concatenate to the [time,
    # event] layout the Rust engine expects, matching
    # GLMBoostModel.cvrisk's identical handling in glmboost.py. A Binomial
    # cbind(success, failure) response is also a tuple, but the fit read it
    # as the proportion and folded the trial count into the weights, so
    # that same proportion is the response here.
    if isinstance(model._family, Binomial) and isinstance(model._response_name, tuple) \
            and len(model._response_name) == 2:
        y = _cbind_proportion(model._data, model._response_name)
    elif isinstance(model._response_name, tuple):
        time_arr = _extract_col(model._data, model._response_name[0])
        event_arr = _extract_col(model._data, model._response_name[1])
        y = np.concatenate([time_arr, event_arr])
    else:
        y = _extract_col(model._data, model._response_name)
    w = model._weights if model._weights is not None else np.ones(n)
    family_name = _family_rust_name(model._family)
    family_params = _family_params(model._family)
    mstop_val, nu = as_control(model._control).mstop, as_control(model._control).nu
    if mstop_override is not None:
        mstop_val = mstop_override

    # Build X matrix with centered columns (matching glmboost.py logic)
    X = np.column_stack([
        _extract_col(model._data, name) for name in model._predictor_names
    ])
    # The fit's own centering: the column means, or none for
    # glmboost(center=False), whose folds are refitted on the raw columns.
    centered = bool(np.any(np.asarray(model._centering_means or [0.0]) != 0.0))
    X_centered = X - X.mean(axis=0) if centered else X

    folds_matrix = _as_fold_matrix(folds, n, model._weight_policy, model._weights)

    sel_matrix = _cvrisk_selected_glmboost(
        X_centered.astype(np.float64),
        y.astype(np.float64),
        w.astype(np.float64),
        mstop_val,
        nu,
        family_name,
        family_params,
        model._learner_names,
        folds_matrix,
        # offset=None: each subsample refits its own offset (e.g. Gaussian uses
        # the in-bag mean of y), matching R's stabsel, which refits glmboost per
        # subsample via update(weights=fold). Passing the full-data offset here
        # froze every fold's offset and perturbed the per-fold selection path
        # (the selection-probability/phat trajectory diverged from R even though
        # the final selected set matched). The design stays full-data-centered,
        # which is correct: R centers once at model creation, not per fold.
        # A per-row offset (log exposure) is not an estimate and is passed.
        model._offset if isinstance(model._offset, np.ndarray) else None,
        **step_kwargs(model._control),
        intercept=getattr(model, "_intercept", True),
        center=centered,
    )
    return np.asarray(sel_matrix, dtype=np.intp).tolist()


def _take_rows(data, rows):
    """The subset of `data` at integer positions `rows`, keeping its type.

    The implementation is `mboostrs._frame.take_rows`; this name is kept
    because several modules already import it from here.
    """
    from mboostrs._frame import take_rows

    return take_rows(data, rows)


def _run_refit_selected(model, folds, is_glmboost, n_jobs=1, mstop_override=None):
    """Refit model on each subsample fold (pure Python fallback).

    The fold refits are independent; ``n_jobs`` (>1 or -1) runs them on a thread
    pool (the Rust fit releases the GIL, so this parallelizes).
    """
    from mboostrs._parallel import parallel_map
    from mboostrs._refit import refit_spec

    del is_glmboost  # the spec knows which entry point produced the model
    # Through the refit spec, so a fold refit carries `center`, `bins`,
    # `learners`, `na_action` and the rest. The hand-built call passed five
    # arguments, so stability selection on a centred or binned fit selected
    # from a different model than the one it was asked about.
    spec = refit_spec(model)
    control = as_control(model._control)
    if mstop_override is not None:
        control = control._replace(mstop=int(mstop_override))
    # A row subset is the fast refit, but it is wrong when the fit carries
    # anything built against every training row: the tensor design of a `%X%`
    # term, a user-supplied design matrix, a neighbourhood graph. Those folds
    # refit on the full data with the out-of-bag rows at zero weight, which is
    # how `stabs::stabsel` refits every `mboost` model (`update(weights=)`).
    # Subsetting them failed on every fold, and the run reported a selection
    # frequency of zero for every learner.
    full_data_refit = bool(spec.row_subset_blockers())

    def _one(b_idx):
        fold_weights = folds[:, b_idx]
        inbag = np.where(fold_weights > 0)[0]
        if len(inbag) < 2:
            return []
        try:
            if full_data_refit:
                sub_model = spec.build(weights=fold_weights, control=control)
            else:
                sub_model = spec.subset(inbag).build(
                    weights=fold_weights[inbag], control=control,
                )
            return list(sub_model._selected)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"Fold {b_idx} refit failed ({type(exc).__name__}: {exc}); "
                "treating as empty selection",
                stacklevel=2,
            )
            return _FailedRefit(f"{type(exc).__name__}: {exc}")

    results = parallel_map(_one, range(folds.shape[1]), n_jobs)
    failures = [result for result in results if isinstance(result, _FailedRefit)]
    if failures and len(failures) == len(results):
        raise ValueError(
            f"every one of the {len(results)} stability-selection refits failed, "
            f"so there are no selections to count; the first failure was "
            f"{failures[0].message}")
    return [[] if isinstance(result, _FailedRefit) else result for result in results]


class _FailedRefit:
    """A fold refit that raised, kept apart from a refit that selected nothing."""

    def __init__(self, message):
        self.message = message


def _extract_col(data, name):
    from mboostrs.glmboost import _extract_column
    return _extract_column(data, name)
