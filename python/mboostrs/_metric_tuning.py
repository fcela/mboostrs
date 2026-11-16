"""Fold-local orchestration for response-score stopping paths.

Fits and numerical losses/aggregation are delegated to Rust. Python keeps
dataframe recipes, fold membership and the public prediction interface.
"""

from __future__ import annotations

import numpy as np

from mboostrs._control import boost_control
from mboostrs._frame import column_names, take_rows
from mboostrs._names import partition_tilde
from mboostrs.families import AdaExp, Binomial, HingeLoss, Multinomial


def _ran_on_gram_route(model) -> bool:
    """Whether a fitted model was computed on the Gaussian Gram route.

    That route is chosen from the same gates as `try_fit_gaussian_gram`, which
    do not depend on the step count, so a refit at a larger end takes it too.
    """
    report = model.execution_report()
    return report is not None and report.route == "gaussian_gram"


def tune_response_mstop(formula, data, family, *, folds, start, nu, extensions,
                         max_mstop, seed, step, score, score_parameter,
                         unseen_levels, full_fit=None, _warn=True,
                         extrapolation="warn", **fit_kwargs):
    from mboostrs import _rust
    from mboostrs.cvrisk import BOUNDARY_SHARE
    from mboostrs.explore import (
        MstopTuning,
        _doubling_ends,
        _evaluation_weights,
        _fitted_family_score,
        _folds,
        _missing_rows,
        _multinomial_response,
        _response_values,
        _warn_search_ended_falling,
    )
    from mboostrs.gamboost import gamboost

    response = partition_tilde(formula)[0].strip()
    if response not in column_names(data):
        raise ValueError("response-score tuning needs a plain response column; "
                         "transformed response scales are not inferred")
    if hasattr(family, "param_names") or type(family).__name__ in (
            "CoxPH", "Gehan", "Cindex", "Weibull", "Loglog", "Lognormal", "PropOdds"):
        raise NotImplementedError("response-score tuning currently needs scalar responses or Multinomial probabilities")
    classes = None
    if isinstance(family, Multinomial):
        y, classes = _multinomial_response(data, response)
        if score not in ("brier", "log_loss"):
            raise ValueError("Multinomial tuning supports brier or log_loss")
        if family.n_classes is not None and family.n_classes != len(classes):
            raise ValueError("Multinomial n_classes disagrees with observed labels")
        family = Multinomial(n_classes=len(classes))
    else:
        y = np.asarray(_response_values(data, response), dtype=float)
    weights = _evaluation_weights(len(y), folds, fit_kwargs.pop("weights", None))
    if score != "family" and isinstance(family, (Binomial, AdaExp, HingeLoss)) and np.any(y[weights > 0] < 0):
        y = (y + 1.0) / 2.0
    # Validate the scoring contract before any fitting, using a valid dummy
    # prediction. All numerical validation remains in the core scorer.
    if score != "family":
        _rust.response_score(np.array([0.]), np.array([[0.5]]), np.ones(1), score, score_parameter)
    partitions = _folds(len(y), folds, seed, weights)
    na_action = fit_kwargs.get("na_action", "error")
    offset = fit_kwargs.pop("offset", None)
    vector_offset = offset is not None and np.ndim(offset) > 0
    if vector_offset:
        offset = np.asarray(offset, dtype=float)
        if offset.shape != y.shape:
            raise ValueError("offset must match response rows")
    prepared = []
    for b, (training_mask, held_mask, training_weights) in enumerate(partitions):
        train_rows, held_rows = np.flatnonzero(training_mask), np.flatnonzero(held_mask)
        if na_action in ("omit", "drop"):
            train_rows = train_rows[~_missing_rows(formula, data, train_rows)]
            held_rows = held_rows[~_missing_rows(formula, data, held_rows)]
        if not len(train_rows) or not len(held_rows):
            raise ValueError(f"fold {b} has no training/evaluation rows after missing-value handling")
        if classes is not None and set(y[train_rows].tolist()) != set(range(len(classes))):
            raise ValueError(f"fold {b} training rows omit classes; use stratified folds")
        train = take_rows(data, train_rows)
        held = take_rows(data, held_rows)
        options = dict(fit_kwargs)
        prediction_options = {"unseen_levels": unseen_levels, "extrapolation": extrapolation}
        if offset is not None:
            options["offset"] = offset[train_rows] if vector_offset else offset
        if vector_offset:
            prediction_options["offset"] = offset[held_rows]
        prepared.append((train, held, y[held_rows], weights[held_rows],
                         training_weights[train_rows], options, prediction_options))
    from mboostrs._workflow import checkpoint

    ends = _doubling_ends(start, max_mstop, extensions)
    grids = [list(range(end + 1)) for end in ends]
    # Each fold's model is fitted once, at the first grid's end, and grown with
    # continue_to at each later grid. Only the steps a grid adds are scored; the
    # steps before them were scored at an earlier grid and do not change.
    models = [None] * len(prepared)
    fits = 0
    scored_to = -1

    def advance(to):
        """Grow every fold's model to `to` steps; return the new scores per fold."""
        nonlocal fits, scored_to
        rows = []
        for index, row in enumerate(prepared):
            (train, held, observed, evaluation_weights, training_weights,
             options, prediction_options) = row
            model = models[index]
            if model is None or _ran_on_gram_route(model):
                # A fold fitted on the Gaussian Gram route keeps no residual to
                # resume from, and its general-route continuation costs more per
                # step than the Gram refit it replaces, so it is refitted from
                # zero, as the refit-from-zero search did.
                model = gamboost(formula, data=train, family=family, weights=training_weights,
                                 control=boost_control(to, nu, step=step), **options)
                fits += 1
            else:
                model = model.continue_to(to)
            models[index] = model
            if classes is None and score != "family":
                path = list(model._response_score_path(
                    observed, evaluation_weights, held, score, score_parameter, **prediction_options))
                path.extend([path[-1]] * (to + 1 - len(path)))
                rows.append(path[scored_to + 1:to + 1])
                continue
            # Limit temporary prediction memory. The model's public grid
            # replay preserves its training-fitted transforms and categories.
            values = []
            chunk = max(1, min(256, 1_000_000 // len(observed)))
            for first in range(scored_to + 1, to + 1, chunk):
                checkpoint()
                for position in range(first, min(first + chunk, to + 1)):
                    prediction = np.asarray(model.predict(
                        held, mstop=position, type="link" if score == "family" else "response",
                        **prediction_options), dtype=float)
                    if score == "family":
                        values.append(_fitted_family_score(
                            model, family, observed, prediction, evaluation_weights,
                            mstop=position))
                        continue
                    if prediction.ndim == 1:
                        prediction = prediction[:, None]
                    mean, _ = _rust.response_score(observed, np.ascontiguousarray(prediction),
                                                   evaluation_weights, score, score_parameter)
                    values.append(mean)
            rows.append(values)
        scored_to = to
        return rows

    fold_scores, curve, _grid, best, at_boundary, used = _rust.search_fold_scores(
        advance, ends, grids, BOUNDARY_SHARE)
    largest = int(used[-1])
    if full_fit is None:
        options = dict(fit_kwargs)
        if offset is not None:
            options["offset"] = offset
        full_fit = gamboost(formula, data=data, family=family, weights=weights,
                            control=boost_control(largest, nu, step=step), **options)
        full_fits = 1
    else:
        full_fits = 0
    if _warn:
        _warn_search_ended_falling(at_boundary, largest, max_mstop, extensions)
    return MstopTuning(mstop=int(best), risk=tuple(curve), largest=largest, first_largest=start,
                       at_boundary=bool(at_boundary), fits=fits, _fit=full_fit,
                       score=score, score_parameter=score_parameter, full_data_fits=full_fits,
                       fold_scores=tuple(tuple(row) for row in fold_scores),
                       grid_ends=tuple(int(end) for end in used))
