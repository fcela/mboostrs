"""Diagnostic call observers: preserve actual inputs/results without modifying them."""

from contextlib import contextmanager

import numpy as np

from mboostrs import _rust


@contextmanager
def native_calls(events):
    names = (
        "conformal_location_scale_scores",
        "conformal_location_scale_band",
        "conformal_count_default_score",
        "conformal_count_reference",
        "conformal_count_uniforms",
        "conformal_count_rank_scores",
        "conformal_count_rank_band",
        "conformal_poisson_scores",
        "conformal_absolute_residuals",
        "conformal_quantile_scores",
        "conformal_quantile",
        "conformal_poisson_band",
        "conformal_symmetric_band",
        "conformal_widen_band",
    )
    originals = {name: getattr(_rust, name) for name in names}

    def wrap(name, function):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            events.append(dict(function=name, args=args, kwargs=kwargs, result=result))
            return result

        return call

    try:
        for name, function in originals.items():
            setattr(_rust, name, wrap(name, function))
        yield
    finally:
        for name, function in originals.items():
            setattr(_rust, name, function)


@contextmanager
def control_predictions(events):
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    originals = {cls: cls.predict for cls in (CatBoostRegressor, LGBMRegressor, GradientBoostingRegressor)}
    identifiers = {}

    def wrap(function):
        def call(model, data, *args, **kwargs):
            result = function(model, data, *args, **kwargs)
            identifier = identifiers.setdefault(id(model), len(identifiers))
            events.append(
                dict(
                    model=identifier,
                    rows=len(data),
                    parameters=model.get_params(),
                    prediction=np.asarray(result).copy(),
                )
            )
            return result

        return call

    try:
        for cls, function in originals.items():
            cls.predict = wrap(function)
        yield
    finally:
        for cls, function in originals.items():
            cls.predict = function
