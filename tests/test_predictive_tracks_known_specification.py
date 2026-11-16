"""A supplied factor specification must represent the reference level too."""
import numpy as np

import mboostrs as mb
from benches.bench_predictive_tracks import FORMULAS


def test_factor_known_specification_represents_both_opposite_curves():
    rng = np.random.default_rng(618204)
    x = rng.uniform(-1, 1, 500)
    g = rng.choice(["a", "b"], len(x))
    z = rng.uniform(-1, 1, len(x))
    y = 2 * np.sin(np.pi * x) * np.where(g == "a", 1, -1) + .5 * z
    model = mb.gamboost(FORMULAS["factor"], {"x1": x, "x2": z, "g": g, "y": y},
                        family=mb.Gaussian(), control=mb.boost_control(400, .1))
    grid = np.linspace(-.95, .95, 100)
    for label, sign in (("a", 1), ("b", -1)):
        prediction = model.predict({"x1": grid, "x2": np.zeros(len(grid)),
                                    "g": np.repeat(label, len(grid))})
        assert np.mean((prediction - sign * 2 * np.sin(np.pi * grid)) ** 2) < .1
