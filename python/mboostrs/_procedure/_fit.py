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
_BUDGET_FITS = {"quick": 16, "standard": 64, "thorough": 512}
# Wall-clock seconds each setting gives the search. On 36 held-out OpenML
# datasets (#913) the slowest completed "standard" search took 51 s and the
# slowest "quick" search outside one 1,025-column table took 33 s; that table
# exceeded 600 s at both settings. The limits are about twice the slowest
# completed run, so they bind on outliers rather than on typical tables.
_BUDGET_SECONDS = {"quick": 60.0, "standard": 120.0, "thorough": 1200.0}
_UNCERTAINTIES = ("stability", "bootstrap", "conformal", "none")


#: How a model returned by `fit` predicts a ``brandom`` or factor level the
#: training rows did not contain (see `GAMBoostModel.predict`).
UNSEEN_LEVELS = "average"
#: Default missing-value policy for `fit()` when `na_action` is omitted.
DEFAULT_NA_ACTION = "impute"
#: Maximum rows used by `autospec` specification search inside `fit()`. Above
#: this count, search runs on a deterministic subsample (pinning min/max response
#: rows to preserve response support) and the final model is fitted on all rows.
_SEARCH_MAX_ROWS = 3_000
#: Minimum share of risk reduction (or effect variance for LSS) a
#: term must carry to remain in the final reported formula.
_ACTIVE_TERM_SHARE = 0.01


class Fit:
    """A fitted model wrapped with the trail that produced it.

    ``model`` is the underlying fitted model (a ``GAMBoostModel`` or
    ``LSSModel``), ``trail`` is the :class:`~mboostrs._procedure._trail.DecisionTrail`
    explaining how it was chosen, and optional uncertainty summaries are
    attached to ``stability``, ``bootstrap``, or ``conformal`` depending on
    ``uncertainty=``. ``predict`` delegates to the model and ``predict_interval``
    computes conformal prediction intervals (defaulting to ``"adaptive"`` when
    fitted with ``time=``).
    """

    def __init__(
        self,
        model: Any,
        trail: DecisionTrail,
        stability: Any = None,
        bootstrap: Any = None,
        conformal: Any = None,
        *,
        time: str | None = None,
    ):
        self.model = model
        self.trail = trail
        self.stability = stability
        self.bootstrap = bootstrap
        self.conformal = conformal
        self._time = time

    def predict(self, newdata: Any, **kwargs: Any) -> Any:
        """Predict with the wrapped model. See ``model.predict``.

        Defaults ``extrapolation="clip"`` so spline terms are held flat outside
        their training boundary knots on automated fits (linear ``bols`` terms
        still extrapolate linearly). Pass an explicit ``extrapolation=`` value
        (e.g. ``"warn"``, ``"allow"``, ``"error"``) to override.
        """
        kwargs.setdefault("extrapolation", getattr(self.model, "_default_extrapolation", "clip"))
        return self.model.predict(newdata, **kwargs)

    def predict_interval(
        self,
        newdata: Any = None,
        *,
        level: float = 0.9,
        method: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compute conformal prediction intervals on ``newdata``.

        Defaults ``method="adaptive"`` when ``fit(..., time=...)`` was used
        and ``method="split"`` otherwise.
        """
        from mboostrs.conformal import conformal_interval

        chosen_method = (
            method
            if method is not None
            else ("adaptive" if self._time is not None else "split")
        )
        seed = (
            self.trail.provenance.get("seed")
            if isinstance(self.trail.provenance, dict)
            else None
        )
        kwargs.setdefault("seed", seed)
        if chosen_method == "adaptive" and self._time is not None and "time" not in kwargs:
            kwargs["time"] = self._time
        return conformal_interval(
            self.model, newdata, level=level, method=chosen_method, **kwargs
        )

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """The wrapping record: the decision trail as a JSON-safe dict."""
        del include_arrays  # accepted for the shared to_dict() convention
        return self.trail.to_dict()

    def reproduce(self, data: Any, **kwargs: Any) -> Any:
        """Replay this fit's decision trail on ``data`` and verify agreement."""
        from mboostrs._procedure._reproduce import reproduce as _reproduce

        return _reproduce(self, data, **kwargs)

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


def _row_count(data: Any) -> int:
    if isinstance(data, dict):
        return len(next(iter(data.values()))) if data else 0
    return int(getattr(data, "height", len(data)))


def _search_subsample(data: Any, response: str, seed: int, max_rows: int) -> Any:
    """Deterministic subsample for specification search on large tables."""
    import numpy as np

    from mboostrs._frame import read_column, take_rows

    n = _row_count(data)
    if n <= max_rows:
        return data
    rng = np.random.default_rng(seed)
    y = np.asarray(read_column(data, response).values)
    finite = np.isfinite(y) if y.dtype.kind == "f" else np.ones(n, dtype=bool)
    uniq = np.unique(y[finite]) if finite.any() else np.array([])
    if 2 <= uniq.size <= 10:
        # Stratified subsample preserving class proportions.
        chosen = []
        for cls in uniq:
            idx = np.flatnonzero(finite & (y == cls))
            k = max(1, int(round(max_rows * idx.size / max(1, int(finite.sum())))))
            k = min(k, idx.size)
            chosen.append(rng.choice(idx, size=k, replace=False))
        rows = np.sort(np.concatenate(chosen))[:max_rows]
    else:
        rows = rng.choice(n, size=max_rows, replace=False)
        # Pin response extremes (min/max finite y) so positive-support families
        # (GammaReg, Poisson, BetaLSS) never get inferred from a subsample that
        # missed a non-positive or boundary response row.
        if y.dtype.kind in "fiu" and finite.any():
            f_idx = np.flatnonzero(finite)
            extremes = {int(f_idx[int(np.argmin(y[f_idx]))]), int(f_idx[int(np.argmax(y[f_idx]))])}
            row_set = set(rows.tolist())
            missing_ext = [e for e in extremes if e not in row_set]
            if missing_ext:
                rows = np.concatenate([rows[: max_rows - len(missing_ext)], np.array(missing_ext, dtype=rows.dtype)])
        rows = np.sort(rows)
    # Pin domain-boundary rows for numeric predictors whose full-data minimum
    # crosses 0 (min <= 0 in full data while subsample has min > 0, or min < 0
    # while subsample has min >= 0), so autospec never chooses log(x)/sqrt(x)
    # on a subsample that missed a zero or negative training observation.
    from mboostrs._frame import column_names

    row_set = set(rows.tolist())
    pin_rows: list[int] = []
    for col_name in column_names(data):
        if col_name == response:
            continue
        col = read_column(data, col_name)
        if col.kind != "numeric":
            continue
        vals = np.asarray(col.values, dtype=float)
        fin = np.isfinite(vals)
        if not fin.any():
            continue
        f_idx = np.flatnonzero(fin)
        min_pos = int(f_idx[int(np.argmin(vals[f_idx]))])
        full_min = float(vals[min_pos])
        if full_min <= 0.0 and min_pos not in row_set:
            sub_vals = vals[rows]
            sub_fin = sub_vals[np.isfinite(sub_vals)]
            if sub_fin.size > 0:
                sub_min = float(np.min(sub_fin))
                if (full_min <= 0.0 < sub_min) or (full_min < 0.0 <= sub_min):
                    pin_rows.append(min_pos)
                    row_set.add(min_pos)
    if pin_rows:
        # Keep the pinned rows by replacing an equal number of non-pinned interior rows.
        pin_set = set(pin_rows)
        keep_n = max(0, max_rows - len(pin_rows))
        kept = [r for r in rows.tolist() if r not in pin_set][:keep_n]
        rows = np.sort(np.asarray(kept + pin_rows, dtype=rows.dtype))
    return take_rows(data, rows)


def _sanitize_formula_domains(formula: Any, data: Any) -> Any:
    """Downgrade any log(col)/sqrt(col) whose full-data domain violates positivity."""
    import re

    import numpy as np

    from mboostrs._frame import read_column
    from mboostrs._names import unquote_name

    cache: dict[str, float | None] = {}

    def _col_min(name: str) -> float | None:
        if name not in cache:
            try:
                vals = np.asarray(read_column(data, name).values, dtype=float)
                fin = vals[np.isfinite(vals)]
                cache[name] = float(np.min(fin)) if fin.size else None
            except Exception:
                cache[name] = None
        return cache[name]

    pattern = re.compile(r"\b(log|sqrt)\(\s*([^()]+?)\s*\)")

    def _fix_one(expr: str) -> str:
        if "log(" not in expr and "sqrt(" not in expr:
            return expr

        def _repl(match: re.Match[str]) -> str:
            kind, raw_col = match.group(1), match.group(2).strip()
            col = unquote_name(raw_col)
            lo = _col_min(col)
            if lo is None:
                return match.group(0)
            if kind == "log":
                if lo > 0.0:
                    return match.group(0)
                if lo >= 0.0:
                    return f"sqrt({raw_col})"
                return raw_col
            if lo >= 0.0:
                return match.group(0)
            return raw_col

        return pattern.sub(_repl, expr)

    if isinstance(formula, str):
        return _fix_one(formula)
    if isinstance(formula, dict):
        return {k: _fix_one(v) for k, v in formula.items()}
    return formula


def _active_terms_for_gam(model: Any, formula: str, response: str, spec: Any) -> str:
    """Prune unselected or negligible (<1% risk reduction) terms from formula."""
    from mboostrs._names import partition_tilde, quote_response, split_terms
    from mboostrs.formula import term_variables

    lhs = quote_response(response)
    null_formula = f"{lhs} ~ 1"
    mstop_ev = (getattr(spec, "evidence", None) or {}).get("mstop")
    if mstop_ev is not None and getattr(mstop_ev, "risk", None):
        r0 = float(mstop_ev.risk[0])
        idx = min(int(mstop_ev.chosen), len(mstop_ev.risk) - 1)
        r_opt = float(mstop_ev.risk[idx])
        if r0 > 0 and (r_opt >= r0 or (r0 - r_opt) / abs(r0) < 0.01):
            return null_formula

    _, sep, rhs = partition_tilde(formula)
    if not sep:
        return formula
    terms = split_terms(rhs)
    if not terms:
        return null_formula

    if not hasattr(model, "varimp"):
        return null_formula if getattr(model, "mstop", 1) == 0 else formula
    vi = model.varimp(percent=False)
    reductions = vi.get("reduction") or {}
    var_names = vi.get("variable_names") or {}
    total_drop = float(sum(max(0.0, float(v)) for v in reductions.values()))
    if total_drop <= 0.0:
        return null_formula

    declared = set(getattr(spec, "declared", ()) or ())
    # Track which variable sets appear in both a tree term and a non-tree term
    # so we only disambiguate by "(btree)" when a variable set is duplicated.
    varset_counts: dict[frozenset[str], int] = {}
    for term in terms:
        vs = frozenset(term_variables(term))
        varset_counts[vs] = varset_counts.get(vs, 0) + 1
    # Compute per-term risk shares (both overall and among non-tree additive
    # terms, so a genuine weak main effect like x2 on S2 is not pruned solely
    # because a dominant interaction/tree absorbed >90% of total risk drop).
    term_shares: list[tuple[str, frozenset[str], bool, float]] = []
    additive_drop = 0.0
    for term in terms:
        tvars = frozenset(term_variables(term))
        is_tree = term.lstrip().startswith("btree(")
        disambiguate = varset_counts.get(tvars, 0) > 1
        drop = 0.0
        for key, red in reductions.items():
            vstr = str(var_names.get(key, key))
            clean_vstr = vstr.removesuffix(" (btree)").strip()
            if "(" in clean_vstr:
                lvars = frozenset(term_variables(clean_vstr))
            else:
                lvars = frozenset(v.strip() for v in clean_vstr.split(",") if v.strip())
            if lvars != tvars:
                continue
            if disambiguate and ("(btree)" in str(key)) != is_tree:
                continue
            drop += max(0.0, float(red))
        if not is_tree:
            additive_drop += drop
        term_shares.append((term, tvars, is_tree, drop))

    kept = []
    for term, tvars, is_tree, drop in term_shares:
        if tvars & declared:
            kept.append(term)
            continue
        share = drop / total_drop
        if share >= _ACTIVE_TERM_SHARE:
            kept.append(term)
        elif (
            not is_tree
            and additive_drop > 0.0
            and share >= 0.003
            and (drop / additive_drop) >= 0.03
        ):
            kept.append(term)
    return f"{lhs} ~ " + " + ".join(kept) if kept else null_formula


def _active_formula(model: Any, spec: Any, response: str) -> str:
    """Formula of terms that meaningfully contribute in the fitted model."""
    import numpy as np

    from mboostrs._names import partition_tilde, quote_response, split_terms
    from mboostrs.formula import term_variables

    raw_formula = spec.formula
    if isinstance(raw_formula, dict):
        lhs = quote_response(response)
        declared = set(getattr(spec, "declared", ()) or ())
        out = {}
        for pname, pformula in raw_formula.items():
            _, sep, rhs = partition_tilde(pformula)
            terms = split_terms(rhs) if sep else []
            if not terms:
                out[pname] = f"{lhs} ~ 1"
                continue
            # Prefer partial-effect variance shares on training rows: works for
            # both cyclic and noncyclic LSS fits and cleanly drops 1-step noise
            # selections whose effect variance is negligible (< 1%).
            effect_vars: dict[frozenset[str], float] = {}
            total_var = 0.0
            try:
                pe = model.partial_effects(parameter=pname)
                for key, val in pe.items():
                    arr = np.asarray(val[1] if isinstance(val, tuple) else val, dtype=float).ravel()
                    v = float(np.nanvar(arr)) if arr.size else 0.0
                    if np.isfinite(v) and v > 0:
                        kvars = frozenset(term_variables(str(key))) if "(" in str(key) else frozenset(
                            s.strip() for s in str(key).split(",") if s.strip()
                        )
                        effect_vars[kvars] = effect_vars.get(kvars, 0.0) + v
                        total_var += v
            except Exception:
                total_var = 0.0
            if total_var <= 0.0:
                out[pname] = f"{lhs} ~ 1"
                continue
            kept = []
            for term in terms:
                tvars = frozenset(term_variables(term))
                if tvars & declared:
                    kept.append(term)
                    continue
                if effect_vars.get(tvars, 0.0) / total_var >= _ACTIVE_TERM_SHARE:
                    kept.append(term)
            out[pname] = f"{lhs} ~ " + " + ".join(kept) if kept else f"{lhs} ~ 1"
        return _formula_text(out)

    if isinstance(raw_formula, str):
        try:
            return _active_terms_for_gam(model, raw_formula, response, spec)
        except Exception:
            return raw_formula
    return _formula_text(raw_formula)


def _compute_stability(model: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    """Compute stability selection for ``uncertainty='stability'``."""
    import warnings

    import numpy as np

    from mboostrs.stabsel import make_folds, stabsel

    is_lss = type(model).__name__ == "LSSModel" or hasattr(model, "param_names")
    learner_names = list(getattr(model, "_learner_names", ()) or ())
    p = len(learner_names)
    b_val = 50

    if is_lss:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stab = model.stabsel(cutoff=0.75, PFER=1.0, B=b_val, verbose=False)
                terms = [
                    {"term": str(name), "pi": float(val)}
                    for name, val in zip(stab.learner_names, stab.max_, strict=False)
                ]
                return stab, {
                    "B": int(stab.B),
                    "pi_thr": float(stab.cutoff),
                    "terms": terms,
                    "expected_false_selections_bound": float(stab.PFER),
                }
            except Exception as exc:  # noqa: BLE001
                terms = []
                for pname in getattr(model, "param_names", ()):
                    try:
                        pe = model.partial_effects(parameter=pname)
                        for k in pe:
                            terms.append({"term": f"{pname}:{k}", "pi": 1.0})
                    except Exception:  # noqa: BLE001
                        pass
                return {"kind": "lss_selection_summary", "terms": terms, "note": str(exc)}, {
                    "B": 0,
                    "pi_thr": None,
                    "terms": terms,
                    "expected_false_selections_bound": None,
                    "note": str(exc),
                }

    if p < 2:
        terms = [{"term": learner_names[0], "pi": 1.0 if getattr(model, "mstop", 0) > 0 else 0.0}] if p == 1 else []
        return {"kind": "single_learner_summary", "terms": terms}, {
            "B": 0,
            "pi_thr": None,
            "terms": terms,
            "expected_false_selections_bound": None,
        }

    n = (
        len(model._weights)
        if getattr(model, "_weights", None) is not None
        else len(model._fitted)
    )
    w_full = model._weights if getattr(model, "_weights", None) is not None else np.ones(n, dtype=float)
    init_folds = make_folds(n, b_val, sampling_type="MB", weights=w_full, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stab = stabsel(model, cutoff=0.75, PFER=1.0, B=b_val, folds=init_folds, verbose=False)
        except ValueError:
            stab = stabsel(model, cutoff=0.75, q=max(1, p // 2), B=b_val, folds=init_folds, verbose=False)
    terms = [
        {"term": str(name), "pi": float(val)}
        for name, val in zip(stab.learner_names, stab.max_, strict=False)
    ]
    return stab, {
        "B": int(stab.B),
        "pi_thr": float(stab.cutoff),
        "terms": terms,
        "expected_false_selections_bound": float(stab.PFER),
    }


def _compute_bootstrap(model: Any, data: Any, response: str, seed: int) -> tuple[Any, dict[str, Any]]:
    """Compute REML refit (with bootstrap draws when Gaussian identity) or row bootstrap fallback."""
    import warnings

    import numpy as np

    from mboostrs._frame import read_column
    from mboostrs._refit import refit_spec
    from mboostrs.selection import gam_refit

    b_draws = 25
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            refit = gam_refit(model, unsupported="hold")
            b_used: int | None = None
            if refit.link == "identity" and not refit.held:
                y = np.asarray(read_column(refit._training_data, response).values, dtype=float).ravel()
                n = y.size
                rng = np.random.default_rng(seed)
                counts = rng.multinomial(n, np.full(n, 1.0 / n), size=b_draws)
                boot_draws = refit.bootstrap(counts, response=y)
                object.__setattr__(refit, "_bootstrap_draws", boot_draws)
                b_used = b_draws
            terms = [{"term": str(t), "pi": None} for t in refit.terms]
            return refit, {"B": b_used, "terms": terms}
        except Exception:  # noqa: BLE001
            pass

    # Fallback row-bootstrap draw summary for trees, GLMBoostModel, LSSModel, etc.
    spec = refit_spec(model)
    n = _row_count(data)
    rng = np.random.default_rng(seed)
    preds: list[np.ndarray] = []
    b_fb = 20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(b_fb):
            idx = rng.integers(0, n, size=n)
            try:
                sub = spec.subset(idx, refit_preprocessing=True).build()
                p = sub.predict(data, type="response")
                if isinstance(p, dict):
                    p = np.asarray(p.get("mu", next(iter(p.values()))), dtype=float).ravel()
                else:
                    p = np.asarray(p, dtype=float).ravel()
                if p.shape[0] == n and np.all(np.isfinite(p)):
                    preds.append(p)
            except Exception:  # noqa: BLE001
                continue
    pred_sd = (
        float(np.mean(np.std(np.vstack(preds), axis=0)))
        if len(preds) >= 2
        else 0.0
    )
    summary = {
        "kind": "row_bootstrap",
        "B": len(preds),
        "mean_pred_sd": pred_sd,
    }
    return summary, {"B": len(preds), "terms": [], "status": "ok" if preds else "unavailable"}


def _compute_conformal(
    model: Any,
    data: Any,
    response: str,
    seed: int,
    time: str | None,
) -> tuple[Any, dict[str, Any]]:
    """Compute conformal calibration (regression/count/LSS) or probability ECE (classification)."""
    import warnings

    import numpy as np

    from mboostrs._frame import read_column
    from mboostrs.conformal import conformal_interval

    fam_name = type(getattr(model, "_family", None)).__name__
    if fam_name in ("Binomial", "Multinomial", "PropOdds"):
        y_raw = np.asarray(read_column(data, response).values, dtype=float).ravel()
        probs = np.clip(
            np.asarray(model.predict(data, type="response"), dtype=float).ravel(),
            1e-12,
            1.0 - 1e-12,
        )
        n = int(y_raw.size)
        if fam_name == "Binomial" and probs.size == n:
            brier = float(np.mean((probs - y_raw) ** 2))
            bins = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for i in range(10):
                mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < 9 else probs <= bins[i + 1])
                if np.any(mask):
                    ece += (float(np.sum(mask)) / max(1, n)) * abs(
                        float(np.mean(y_raw[mask])) - float(np.mean(probs[mask]))
                    )
            status = "reliable" if ece < 0.1 else "caution"
            summary = {
                "kind": "classification_calibration",
                "ece": ece,
                "brier": brier,
                "n": n,
                "status": status,
            }
            return summary, {
                "conformal_method": "probability_calibration",
                "ece": ece,
                "n_calibration": n,
                "status": status,
            }
        summary = {"kind": "classification_calibration", "n": n, "status": "reliable"}
        return summary, {
            "conformal_method": "probability_calibration",
            "n_calibration": n,
            "status": "reliable",
        }

    conf_method = "adaptive" if time is not None else "split"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ci = conformal_interval(
                model,
                data,
                level=0.9,
                method=conf_method,
                seed=seed,
                time=time,
            )
            return ci, {
                "conformal_method": str(ci.method),
                "level": float(ci.level),
                "n_calibration": int(ci.n_calibration),
                "quantile": (float(ci.quantile) if ci.quantile is not None else None),
                "mean_width": float(np.mean(ci.width)),
                "status": (ci.reliability or {}).get("status"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"kind": "conformal_unavailable", "error": str(exc)}, {
                "conformal_method": conf_method,
                "status": "unavailable",
                "note": str(exc),
            }


def fit(
    data: Any,
    response: str,
    *,
    family: Any = None,
    folds: Any = None,
    groups: Any = None,
    time: str | None = None,
    seed: int = 0,
    budget: str = "standard",
    uncertainty: str = "none",
    na_action: str | None = None,
    **kwargs: Any,
) -> Fit:
    """Fit one model and return it with the trail of how it was chosen.

    ``fit`` uses ``autospec`` (with guarded interaction adoption and
    ``residual_tree="auto"`` by default) to choose the specification and
    returns the fitted model with a trail containing the input declarations,
    specification decision, active final formula (terms carrying at least 1%
    of risk reduction or effect variance, or intercept-only when the CV curve
    is flat), family, and stopping iteration.

    The returned model predicts unseen categorical or ``brandom`` levels as
    the average group by default (override per call with
    ``unseen_levels="error"``). On tables with more than 3,000 rows and no
    explicit ``folds`` or ``groups``, specification search runs on a
    deterministic 3,000-row subsample (pinning the response extremes to
    preserve support) and the final model is fitted on all rows.

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
    groups : optional
        Declared grouping column(s), passed to ``autospec``.
    time : str, optional
        Time column name ordering the observations. When ``folds`` is omitted,
        constructs rolling-origin folds via :func:`~mboostrs.folds.rolling_folds`
        and defaults ``fit.predict_interval(...)`` to ``method="adaptive"``.
    seed : int, default 0
        The single seed for every random stream.
    budget : {"quick", "standard", "thorough"}
        The number of model fits and the wall-clock time the specification
        search is given: 16 fits and 60 s, 64 fits and 120 s, 512 fits and
        1200 s. Each fit is one cross-validated or held-out evaluation of a
        candidate family, learner, interaction or stopping depth. With fewer
        fits more decisions fall back to the dtype-driven default learner for
        each column.

        Neither limit is exact. autospec checks the fit count before each
        stage starts, so a stage in progress can exceed it. The time is
        checked before each model fit starts, so the search can overrun by the
        length of one fit. When the time runs out the search keeps what it
        measured if the probe in progress allows it, and otherwise returns the
        default specification (one default learner per column, nothing
        measured). The trail's ``budget`` records both limits and the fits
        spent, and ``degradations`` says when a limit stopped the search and
        which of those outcomes occurred.
    uncertainty : {"none", "stability", "bootstrap", "conformal"}
        Post-fit uncertainty quantification computed and attached to the
        returned :class:`Fit` (``fit.stability``, ``fit.bootstrap``, or
        ``fit.conformal``) and recorded in ``trail.uncertainty``.
    na_action : {"impute", "impute_model", "impute_model_split", "omit", "error"}, optional
        Missing-value policy passed through to ``autospec`` and the final fit.
        ``None`` (default) uses ``"impute"`` so incomplete predictors are
        median/mode-filled with missingness indicators and prediction handles
        missing values cleanly; pass ``"omit"`` or ``"error"`` to override.
    **kwargs
        Passed through to ``autospec``. ``interactions`` defaults to
        ``"adopt"`` and ``residual_tree`` defaults to ``"auto"``.
    """
    if budget not in _BUDGETS:
        raise ValueError(
            f"budget must be one of {_BUDGETS}, got {budget!r}; for an explicit "
            "number of model fits call autospec(..., budget=n) directly")
    if uncertainty not in _UNCERTAINTIES:
        raise ValueError(
            f"uncertainty must be one of {_UNCERTAINTIES}, got {uncertainty!r}"
        )

    effective_folds = folds
    time_values = None
    if time is not None:
        if not isinstance(time, str):
            raise ValueError(
                f"time must be a column name string or None, got {type(time).__name__}"
            )
        if time == response:
            raise ValueError("time must not name the response column")
        from mboostrs._frame import column_names, read_column
        from mboostrs.folds import rolling_folds

        if time not in column_names(data):
            raise ValueError(f"time column {time!r} not found in data")
        time_values = read_column(data, time).values
        if effective_folds is None:
            n_total = _row_count(data)
            n_splits = min(5, max(2, n_total // 2)) if n_total < 10 else 5
            effective_folds = rolling_folds(n_total, B=n_splits, time=time_values)

    limit, seconds = _BUDGET_FITS[budget], _BUDGET_SECONDS[budget]
    effective_na = DEFAULT_NA_ACTION if na_action is None else na_action
    search_kwargs = dict(kwargs)
    search_kwargs.setdefault("interactions", "adopt")
    search_kwargs.setdefault("residual_tree", "auto")
    default_search = "exhaustive" if budget == "thorough" else "fast"
    search_kwargs.setdefault("search", default_search)
    if search_kwargs.get("search") == "exhaustive" and budget != "thorough":
        search_kwargs.setdefault("lss_trees", False)
    else:
        search_kwargs.setdefault("lss_trees", "auto")
    search_data = data
    search_folds = effective_folds
    subsampled_for_search = False
    if (
        search_kwargs.get("search") == "exhaustive"
        and folds is None
        and groups is None
        and _row_count(data) > _SEARCH_MAX_ROWS
    ):
        if time is not None and time_values is not None:
            import numpy as np

            from mboostrs._frame import read_column, take_rows
            from mboostrs.folds import rolling_folds

            recent_idx = np.argsort(np.asarray(time_values), kind="stable")[-_SEARCH_MAX_ROWS:]
            search_data = take_rows(data, recent_idx)
            search_folds = rolling_folds(
                _row_count(search_data),
                B=5,
                time=read_column(search_data, time).values,
            )
        else:
            search_data = _search_subsample(data, response, seed, _SEARCH_MAX_ROWS)
        subsampled_for_search = True
        # Skip the 25-fit step-length grid sweep on large subsampled tables;
        # the residual tree still lowers nu to 0.1 automatically if adopted.
        search_kwargs.setdefault("nu", 0.3)
    spec, stopped = _timed_search(search_data, response, limit, seconds, family=family,
                                  seed=seed, folds=search_folds, groups=groups,
                                  na_action=effective_na, **search_kwargs)
    sanitized = _sanitize_formula_domains(spec.formula, data)
    if sanitized != spec.formula:
        from dataclasses import replace as _dc_replace
        spec = _dc_replace(spec, formula=sanitized, learners=None)
    from mboostrs.families import Gaussian as _Gaussian

    fit_extra: dict[str, Any] = {}
    if not isinstance(spec.family, _Gaussian) and _row_count(data) >= 1500:
        fit_extra["bins"] = 256
    if fit_extra:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="bins= was not applied")
            model = spec.fit(data, **fit_extra)
    else:
        model = spec.fit(data)
    # A group code the training rows never saw is predicted as the average
    # group rather than refused: a model fitted automatically meets new codes
    # (a new store, a new source video) as routinely as new values. A caller
    # who wants the refusal passes `unseen_levels="error"` to `predict`.
    model._default_unseen_levels = UNSEEN_LEVELS
    # Hold spline terms flat at their training boundary knots by default on
    # automated fits (linear `bols` terms still extrapolate linearly), so a
    # single extreme out-of-range predictor value cannot blow up predictions.
    model._default_extrapolation = "clip"

    trail = DecisionTrail()
    trail.start(data, response, seed, folds=effective_folds)
    candidate_formula = _formula_text(spec.formula)
    final_formula = (
        candidate_formula if stopped == "fallback"
        else _active_formula(model, spec, response)
    )
    trail.steps.append(inputs_step(spec))
    # Stages 1-12 currently record through autospec's evidence.
    radar_ev = (getattr(spec, "evidence", None) or {}).get("fast_radar") or {}
    fast_subsampled = bool(radar_ev.get("search_subsampled"))
    sub_n = int(radar_ev.get("search_rows") or _SEARCH_MAX_ROWS) if fast_subsampled else _SEARCH_MAX_ROWS
    search_note = (
        f" (specification searched on {sub_n:,}-row subsample; final fit on all {_row_count(data):,} rows)"
        if (subsampled_for_search or fast_subsampled) else ""
    )
    trail.add_step(
        "1-12", "autospec_delegate",
        incumbent=None,
        candidates=[{
            "spec": candidate_formula,
            "score": None,
            "d": None,
            "se": None,
            "status": "ok",
            "reason": f"specification chosen by autospec within {limit} model fits{search_note}",
        }],
        threshold=None,
        adopted=final_formula,
        rule=None,
        fits=int(spec.fits),
        reason=(
            f"autospec ran its stages 1-12 with a budget of {limit} model fits "
            f"and spent {int(spec.fits)}{search_note}."
        ),
    )
    trail.set_final(final_formula, spec.family, spec.mstop, unseen_levels=UNSEEN_LEVELS)

    stability_val = None
    bootstrap_val = None
    conformal_val = None
    if uncertainty == "none":
        trail.set_uncertainty("none")
    elif uncertainty == "stability":
        stability_val, unc_fields = _compute_stability(model, seed)
        trail.set_uncertainty("stability", **unc_fields)
    elif uncertainty == "bootstrap":
        bootstrap_val, unc_fields = _compute_bootstrap(model, data, response, seed)
        trail.set_uncertainty("bootstrap", **unc_fields)
    elif uncertainty == "conformal":
        conformal_val, unc_fields = _compute_conformal(model, data, response, seed, time)
        trail.set_uncertainty("conformal", **unc_fields)

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
    return Fit(
        model=model,
        trail=trail,
        stability=stability_val,
        bootstrap=bootstrap_val,
        conformal=conformal_val,
        time=time,
    )
