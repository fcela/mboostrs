"""Public effect reports replay composed recipes from the caller's raw columns."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.explore import _derived_column_sources


@pytest.mark.parametrize("restore", [False, True])
def test_sensitivity_accepts_raw_inputs_of_a_transformed_factor_by(restore):
    rng = np.random.default_rng(618210)
    x = np.exp(rng.uniform(0, 4, 240))
    g = rng.choice(["a", "b"], len(x))
    z = rng.uniform(-1, 1, len(x))
    data = {"x": x, "g": g, "z": z, "y": np.log(x) * (g == "b") + z}
    model = mb.gamboost("y ~ bbs(log(x), by=g) + bols(z)", data,
                        family=mb.Gaussian(), control=mb.boost_control(40, .1))
    if restore:
        model = pickle.loads(pickle.dumps(model))
    before = model.predict(data)
    rows = mb.sensitivity(model, data)
    assert {r["variable"] for r in rows} == {"x", "g", "z"}
    assert all(np.isfinite(r["range of the effect"]) for r in rows)
    np.testing.assert_array_equal(model.predict(data), before)


def test_transitive_shared_sources_keep_order_and_legitimate_generated_looking_inputs():
    recipe = SimpleNamespace(steps=[SimpleNamespace(name="t", sources=("x",)),
                                    SimpleNamespace(name="u", sources=("t", "z"))])
    def side(name):
        return SimpleNamespace(column=name)
    interaction = SimpleNamespace(derived="joint", left=side("u"), right=side("t"))
    model = SimpleNamespace(_transform_recipe=recipe, _interactions=[interaction])
    assert _derived_column_sources(model, ["joint", "u", "log__user"]) == ["x", "z", "log__user"]


def test_cyclic_recipe_metadata_is_rejected_without_recursion_failure():
    recipe = SimpleNamespace(steps=[SimpleNamespace(name="a", sources=("b",)),
                                    SimpleNamespace(name="b", sources=("a",))])
    model = SimpleNamespace(_transform_recipe=recipe, _interactions=[])
    with pytest.raises(ValueError, match="cyclic derived-column"):
        _derived_column_sources(model, ["a"])
