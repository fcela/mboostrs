"""deselect and probe on the core equal the earlier NumPy implementation,
which is frozen below (``_old_*``), except for the probe permutation, which
the core draws from its own seeded generator."""

import re
import warnings  # noqa: F401

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs.selection import (
    Deselection, Probing, _bundle, _column, _entry, _refit, _term_learners, _terms,
    _written_term_labeler, deselect, probe,
)

# ---------------------------------------------------------------- frozen
def _old_deselect(model, threshold=0.01, *, folds=5, seed=0, mstop=None):
    """Drop base learners with a small share of the risk reduction, then refit.

    Parameters
    ----------
    model : GAMBoostModel, GLMBoostModel or LSSModel
        A fit at its tuned step count (for example ``tune_mstop(...).model``).
        A distributional model must be fitted with ``method="noncyclic"``,
        the setting Strömer et al. (2022) propose deselection for: each step
        updates one learner of one parameter, so each step's risk reduction
        belongs to one learner. Shares are computed over all parameters
        together, a parameter left with no term is refitted as intercept
        only (``y ~ 1``), and the refit's total step count is tuned by
        ``LSSModel.cvrisk``.
    threshold : float
        A learner whose share of the total in-bag risk reduction is below this
        is dropped. Strömer et al. (2022) use 0.01.
    folds : int or fold matrix
        Cross-validation for the refit's `mstop`: an int is random k-fold on
        `seed`; pass ``grouped_folds(...)`` for grouped data.
    mstop : int, optional
        The refit's largest step count; defaults to twice the larger of the
        model's control `mstop` and its current `mstop`.

    Returns
    -------
    Deselection
        ``kept``, ``dropped``, the per-learner ``shares``, the reduced
        ``formula`` and the refitted ``model``.

    A learner's share is its accumulated in-bag risk reduction, so a term that
    moved the fit a little in many steps can fall below the threshold even
    when the true effect is non-zero but small; the threshold trades those
    losses against the noise terms removed.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in (0, 1)")
    if type(model).__name__ == "LSSModel":
        return _old_deselect_lss(model, threshold, folds=folds, seed=seed, mstop=mstop)
    _entry(model)
    bundle = _bundle(model)
    lhs, terms = _terms(bundle["raw_formula"], bundle["raw_data"])
    shares = {name: float(v) for name, v in model.varimp(percent=True)["reduction"].items()}
    mapping = _term_learners(model, terms)

    kept, dropped = [], []
    for sign, term in terms:
        if sign != "+" or term.strip() in ("1", "0"):
            kept.append((sign, term))
            continue
        share = sum(shares.get(name, 0.0) for name in mapping.get(term, []))
        (kept if share >= threshold else dropped).append((sign, term) if share >= threshold
                                                          else (term, share))
    kept_terms = [t for s, t in kept if s == "+" and t.strip() not in ("1", "0")]
    if not kept_terms:
        raise ValueError(
            f"no learner reaches {threshold:.0%} of the risk reduction; nothing "
            f"to refit")
    formula = f"{lhs} ~ " + " ".join(
        f"{'+' if i and s == '+' else ('-' if s == '-' else '')} {t}".strip()
        for i, (s, t) in enumerate(kept)).replace("~ +", "~")
    refit = _refit(model, formula, folds=folds, seed=seed, mstop=mstop)
    return Deselection(
        kept=tuple(kept_terms),
        dropped=tuple((t, round(share, 6)) for t, share in dropped),
        shares=shares, threshold=threshold, formula=formula, model=refit)


def _old_deselect_lss(model, threshold, *, folds, seed, mstop):
    """`deselect` for a non-cyclic `glmboostLSS`/`gamboostLSS` fit."""
    import mboostrs as mb
    from mboostrs.formula import _signed_terms

    if model._method != "noncyclic":
        raise NotImplementedError(
            "deselection needs the risk reduction of each learner, and a cyclic "
            "LSS fit records risk once per cycle, after every parameter has "
            "been updated, so the reduction cannot be divided between "
            "learners. Refit with method='noncyclic', the setting deselection "
            "was proposed for (Strömer et al., 2022).")
    arguments = model._refit_arguments()
    base = arguments.pop("formula")
    names = [str(name) for name in model.param_names]
    formulas = base if isinstance(base, dict) else {name: base for name in names}

    # Shares by (parameter, learner index): two terms over one column carry
    # the same learner name, so the name alone cannot tell them apart.
    reduction = {}
    for row in model.path_frame(output="records"):
        key = (row["parameter"], int(row["learner_index"]))
        reduction[key] = reduction.get(key, 0.0) + float(row.get("risk_change") or 0.0)
    total = sum(abs(v) for v in reduction.values()) or 1.0
    learner_names = model.base_learner_names()
    glm = model._model_kind != "gamboostLSS"

    kept, dropped, shares, reduced = [], [], {}, {}
    for name in names:
        lhs, _, rhs = formulas[name].partition("~")
        terms = [(sign, term.strip()) for sign, term in _signed_terms(rhs)]
        learners = learner_names[name]
        if glm:
            # One learner per column, named by it, after the intercept.
            index_of = {learner: i for i, learner in enumerate(learners)}
        else:
            # One learner per `+` term in order; an explicit `1` is a learner.
            added = [term for sign, term in terms if sign == "+" and term != "0"]
            if len(added) != len(learners):
                raise NotImplementedError(
                    f"the {name} formula's {len(added)} terms do not map one to "
                    f"one onto the fit's {len(learners)} learners (a term that "
                    f"expands into several learners); write the reduced "
                    f"formulas by hand from varimp()")
            index_of = {term: i for i, term in enumerate(added)}
        keep, fixed = [], []
        for sign, term in terms:
            if sign != "+" or term in ("1", "0"):
                fixed.append(f"- {term}" if sign == "-" else f"+ {term}")
                continue
            if term not in index_of:
                raise NotImplementedError(
                    f"the {name} formula's term {term!r} does not name one "
                    f"learner of the fit; write the reduced formulas by hand "
                    f"from varimp()")
            share = abs(reduction.get((name, index_of[term]), 0.0)) / total
            shares[f"{term}.{name}"] = share
            if share >= threshold:
                keep.append(term)
                kept.append((name, term))
            else:
                dropped.append((name, term, round(share, 6)))
        # A parameter with no term left keeps its intercept alone.
        rhs_new = " + ".join(keep) if keep else "1"
        extras = [f for f in fixed if not (f == "+ 1" and not keep)]
        reduced[name] = " ".join([f"{lhs.strip()} ~ {rhs_new}", *extras])
    if not kept:
        raise ValueError(
            f"no learner reaches {threshold:.0%} of the risk reduction; nothing "
            f"to refit")

    if getattr(model, "_method", None) == "noncyclic":
        # The total number of updates; `mstop` collapses equal per-parameter
        # counts to one integer, which is half the total for two parameters.
        current = model._budget()
    else:
        budget = model.mstop
        current = sum(budget.values()) if isinstance(budget, dict) else int(budget)
    control = arguments.pop("control")
    control = control._replace(mstop=int(mstop or max(control.mstop, current) * 2))
    entry = mb.gamboostLSS if model._model_kind == "gamboostLSS" else mb.glmboostLSS
    fit = entry(reduced, **arguments, control=control, method="noncyclic",
                weights=model._weights)
    # The reduced formulas are the rewritten ones, read against the processed
    # frame: the recipes that rebuild derived columns (a `%X%` tensor, a
    # transformed column or response) on new rows belong to the original fit
    # and are carried over, as the LSS refit does.
    fit._transform_recipe = getattr(model, "_transform_recipe", None)
    fit._response_transform = getattr(model, "_response_transform", None)
    fit._interactions = getattr(model, "_interactions", None)
    fit._rewritten_formulas = dict(reduced)
    n = len(model.model_weights())
    fold_matrix = (mb.cv_weights(n, "kfold", folds, seed=seed,
                                 weights=model.model_weights())
                   if isinstance(folds, int) else folds)
    cv = fit.cvrisk(folds=fold_matrix)
    refit = fit.with_mstop(max(cv.optimal_mstop[0], 1))
    # The fit reads `%X%` pairs as derived designs, `buser(__ix0, ...)`; the
    # record names them by the term the user wrote.
    shown = _written_term_labeler(model)
    return Deselection(
        kept=tuple((name, shown(term)) for name, term in kept),
        dropped=tuple((name, shown(term), share) for name, term, share in dropped),
        shares={f"{shown(key.rsplit('.', 1)[0])}.{key.rsplit('.', 1)[1]}": value
                for key, value in shares.items()},
        threshold=threshold,
        formula={name: shown(text) for name, text in reduced.items()},
        model=refit)




_OLD_SINGLE_COLUMN_TERM = re.compile(r"^\s*(?:(b\w+)\s*\(\s*)?([A-Za-z_][\w.]*)\s*(,.*)?\)?\s*$", re.DOTALL)


def _old_probe(formula, data=None, *, family=None, control=None, seed=0, folds=5, **fit_kwargs):
    """Select terms by the first entry of a permuted shadow, then refit them.

    ``formula`` is a formula string with ``data``, or a fitted ``gamboost`` /
    ``glmboost`` model, whose formula, data, family and fitting arguments are
    reused (``probe(model)``, as ``deselect(model)`` and ``stabsel(model)``).

    Every term that reads one column (``x``, ``bbs(x, df=4)``, ``bols(x)``)
    gets a shadow: the same term over a permuted copy of the column. The
    model with all terms and shadows is boosted up to ``control.mstop``
    (default 1000) iterations, and the terms selected before the first shadow
    are kept. Terms over several columns (``%X%``, ``bspatial``) get no shadow
    and are kept only if selected before the first shadow.

    Returns
    -------
    Probing
        ``selected``, the iteration ``stop`` at which the first shadow entered,
        its name, the formula of the selected terms and ``model``, that formula
        refitted without shadows with `mstop` by cross-validation.

    Probing needs no tuning, but it has no error-rate guarantee: a shadow can
    enter early by chance, which stops selection early, and a true term with
    a weak effect can enter after a shadow.
    """
    import mboostrs as mb
    from mboostrs._frame import column_names
    from mboostrs.formula import _signed_terms, expand_formula

    if not isinstance(formula, str):
        model = formula
        _entry(model)
        if data is not None:
            raise TypeError("probe(model) reuses the model's data; do not pass data=")
        bundle = _bundle(model)
        arguments = {k: v for k, v in (bundle.get("raw_arguments") or {}).items()
                     if v is not None and k not in ("learners", "control")}
        family = family or arguments.pop("family", None)
        arguments.pop("family", None)
        fit_kwargs = {**arguments, **fit_kwargs}
        formula, data = bundle["raw_formula"], bundle["raw_data"]
    elif data is None:
        raise TypeError("probe(formula, data): data is required with a formula")
    family = family or mb.Gaussian()
    control = control or mb.boost_control(mstop=1000, nu=0.1)
    expanded = expand_formula(formula, data)
    lhs, _, rhs = expanded.partition("~")
    terms = [t for s, t in _signed_terms(rhs) if s == "+" and t.strip() not in ("1", "0")]

    rng = np.random.default_rng(seed)
    columns = {name: np.asarray(_column(data, name)) for name in column_names(data)}
    shadows, shadow_terms = {}, []
    for term in terms:
        match = _OLD_SINGLE_COLUMN_TERM.match(term)
        if not match or "%X%" in term:
            continue
        column = match.group(2)
        if column not in columns:
            continue
        shadow = f"shadow__{column}"
        if shadow not in shadows:
            shadows[shadow] = rng.permutation(columns[column])
        shadow_terms.append(re.sub(rf"\b{re.escape(column)}\b", shadow, term, count=1))
    if not shadow_terms:
        raise ValueError("probe needs at least one term over a single column")

    augmented = {**columns, **shadows}
    entry = mb.glmboost if all(re.fullmatch(r"[\w.]+", t.strip()) for t in terms) else mb.gamboost
    wide = entry(f"{lhs.strip()} ~ " + " + ".join(terms + shadow_terms), data=augmented,
                 family=family, control=control, **fit_kwargs)

    names = list(wide._learner_names)
    order = list(wide._selected)
    selected_learners, stop, first_shadow = [], None, None
    for iteration, index in enumerate(order, start=1):
        name = names[index]
        if name.startswith("shadow__") or "shadow__" in name:
            stop, first_shadow = iteration, name
            break
        if name not in selected_learners:
            selected_learners.append(name)

    mapping = {}
    for term, name in zip(terms, [n for n in names if not n.startswith("shadow__")][:len(terms)], strict=True):
        mapping[term] = name
    if entry is mb.glmboost:
        mapping = {t: t.strip() for t in terms}
    chosen = [t for t in terms if mapping.get(t) in selected_learners]
    if not chosen:
        raise ValueError(
            f"a shadow ({first_shadow}) entered before any real term; nothing "
            f"is selected")
    reduced = f"{lhs.strip()} ~ " + " + ".join(chosen)
    refit = entry(reduced, data=data, family=family,
                  control=control._replace(mstop=max(control.mstop, 1)), **fit_kwargs)
    n = len(np.asarray(refit.predict()).ravel())
    fold_matrix = mb.cv_weights(n, "kfold", folds, seed=seed) if isinstance(folds, int) else folds
    refit = refit.with_mstop(max(refit.cvrisk(folds=fold_matrix).optimal_mstop, 1))
    return Probing(selected=tuple(chosen), stop=stop, first_shadow=first_shadow,
                   formula=reduced, model=refit)


# ----------------------------------------------------------------- tests
import mboostrs.selection as selection  # noqa: E402


def _additive(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 6))
    y = 0.6 * X[:, 0] + np.sin(1.5 * X[:, 1]) + 0.4 * X[:, 2] ** 2 + rng.normal(0, 0.5, n)
    data = {f"x{i + 1}": X[:, i] for i in range(6)}
    data["y"] = y
    return data


def _lss_data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (5, n))
    y = np.sin(2 * x[0]) + rng.normal(size=n) * np.exp(0.5 * x[1])
    return {"y": y, **{f"x{i}": x[i] for i in range(5)}}


def _same_deselection(new, old):
    assert new.kept == old.kept
    assert [d[:-1] for d in new.dropped] == [d[:-1] for d in old.dropped]
    assert max([abs(a[-1] - b[-1]) for a, b in zip(new.dropped, old.dropped, strict=True)] or [0]) <= 1e-12
    assert new.shares.keys() == old.shares.keys()
    assert max(abs(new.shares[k] - old.shares[k]) for k in new.shares) <= 1e-12
    assert new.formula == old.formula
    assert new.model.mstop == old.model.mstop


@pytest.mark.parametrize("threshold", [0.01, 0.05])
def test_deselect_gamboost_matches_frozen(threshold):
    data = _additive()
    rhs = " + ".join(f"bbs(x{i}, df=4)" for i in range(1, 7))
    tuned = mb.tune_mstop(f"y ~ {rhs}", data).model
    _same_deselection(deselect(tuned, threshold), _old_deselect(tuned, threshold))


def test_deselect_glmboost_matches_frozen():
    data = _additive(seed=3)
    tuned = mb.tune_mstop("y ~ x1 + x2 + x3 + x4 + x5 + x6", data).model
    _same_deselection(deselect(tuned, 0.02), _old_deselect(tuned, 0.02))


@pytest.mark.parametrize("threshold", [0.01, 0.2])
def test_deselect_lss_matches_frozen(threshold):
    data = _lss_data()
    fit = mb.gamboostLSS({"mu": "y ~ bbs(x0) + bbs(x2)", "sigma": "y ~ bols(x1) + bols(x3)"},
                         data=data, families=mb.GaussianLSS(),
                         control=mb.boost_control(200, 0.1), method="noncyclic")
    _same_deselection(deselect(fit, threshold), _old_deselect(fit, threshold))


def test_deselect_threshold_is_checked():
    tuned = mb.gamboost("y ~ bbs(x1)", data=_additive())
    for bad in (0, 1, -0.1):
        with pytest.raises(ValueError, match="threshold"):
            deselect(tuned, bad)


def _numpy_permutations(n, count, seed):
    rng = np.random.default_rng(seed)
    return [rng.permutation(np.arange(n)) for _ in range(count)]


@pytest.mark.parametrize("seed", [0, 5])
def test_probe_matches_frozen_when_given_the_numpy_permutations(seed, monkeypatch):
    """Everything but the generator: with NumPy's permutations supplied to the
    core, the stopping index, selected terms and refit equal the old result."""
    data = _additive(seed=2)
    rhs = " + ".join(f"bbs(x{i}, df=4)" for i in range(1, 7))
    old = _old_probe(f"y ~ {rhs}", data, seed=seed)
    monkeypatch.setattr(selection._rust, "probe_permutations", _numpy_permutations)
    new = probe(f"y ~ {rhs}", data, seed=seed)
    assert new.selected == old.selected
    assert new.stop == old.stop
    assert new.first_shadow == old.first_shadow
    assert new.formula == old.formula
    assert new.model.mstop == old.model.mstop


def test_probe_with_the_core_generator_is_reproducible_and_selects_signal():
    data = _additive(seed=2)
    rhs = " + ".join(f"bbs(x{i}, df=4)" for i in range(1, 7))
    first = probe(f"y ~ {rhs}", data, seed=4)
    again = probe(f"y ~ {rhs}", data, seed=4)
    assert (first.selected, first.stop, first.first_shadow) == (
        again.selected, again.stop, again.first_shadow)
    assert first.first_shadow is None or first.first_shadow.startswith("shadow__")
    assert {"bbs(x1, df=4)", "bbs(x2, df=4)"} <= set(first.selected)


def test_probe_permutations_are_seeded_permutations():
    n, count = 200, 4
    a = _rust.probe_permutations(n, count, 3)
    assert a == _rust.probe_permutations(n, count, 3)
    assert a != _rust.probe_permutations(n, count, 4)
    for order in a:
        assert sorted(order) == list(range(n))
    assert len({tuple(o) for o in a}) == count
    # The position of row 0 is close to uniform over many draws.
    draws = _rust.probe_permutations(10, 5000, 0)
    counts = np.bincount([o.index(0) for o in draws], minlength=10)
    assert counts.min() > 400 and counts.max() < 600


def test_probe_stop_is_the_first_shadow_in_the_selected_sequence():
    is_shadow = [False, False, True, True]
    stop, first, before = _rust.probe_stop([1, 0, 1, 3, 2, 0], is_shadow)
    assert (stop, first, before) == (4, 3, [1, 0])
    assert _rust.probe_stop([0, 1, 0], is_shadow) == (None, None, [0, 1])
