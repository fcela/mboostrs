"""A matched-margin gain must not replace a better actual incumbent."""

import importlib
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

MODULE = importlib.import_module("mboostrs.discover")


@pytest.mark.parametrize("incumbent,expected", [(0.4, False), (0.5, False),
    (np.nextafter(0.5, np.inf), True), (0.6, True), (None, False)])
def test_actual_incumbent_is_preserved_unless_candidate_strictly_improves(monkeypatch, incumbent, expected):
    value = MODULE._Analysis.__new__(MODULE._Analysis)
    value.mean_terms = ["bbs(x1, df=4.0)"]
    value.complete = value.frame = pl.DataFrame({"x1": [0., 1., 2.], "x2": [2., 1., 0.], "y": [1., 2., 3.]})
    value.response, value.family, value.seed, value.folds = "y", mb.NBinomial(), 679, 5
    value.na_action, value.fold_extrapolation = "omit", "clip"
    value.transforms, value.kind_of = {}, {"x1": "continuous", "x2": "continuous"}
    value.log, value.interaction_records = [], []
    value._fit_mean = lambda: object()
    value._admissible_pairs = lambda pairs: pairs
    value._admit = lambda *args, **kwargs: True
    observed = []
    value._pair_tree = lambda pair, tensor: observed.append((list(value.mean_terms), tensor))
    monkeypatch.setattr(MODULE, "pearson_residual", lambda *args: np.zeros(3))
    pair = MODULE.ScreenedPair(left="x1", right="x2", removed=1.)
    monkeypatch.setattr(MODULE, "_screen_pairs", lambda *args, **kwargs: [pair])

    def compare(specifications, *args, **kwargs):
        assert len(specifications) == 3 and specifications[-1] == "bbs(x1, df=4.0)"
        assert kwargs["include_evaluations"] and kwargs["nu"] == .02
        rows = [dict(change=0., **{"change se": 0., "out-of-fold risk": 1.}),
                dict(change=-.5, evaluation=SimpleNamespace(fold_scores=[.5]*5),
                     **{"change se": 0., "out-of-fold risk": .5}),
                dict(evaluation=None if incumbent is None else SimpleNamespace(fold_scores=[incumbent]*5))]
        return rows

    monkeypatch.setattr(MODULE, "compare_specifications", compare)
    value._interactions()
    assert value.log[-1]["adopted"] == expected
    assert value.log[-1]["incumbent_check"]["strictly_improves_incumbent"] == expected
    if not expected:
        assert value.mean_terms == ["bbs(x1, df=4.0)"]
        assert observed == [(["bbs(x1, df=4.0)"], None)]
    else:
        assert len(value.interaction_records) == 1
        assert "%X%" in value.mean_terms[-1]
