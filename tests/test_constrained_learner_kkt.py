"""Shape-constrained learners solve their constrained step exactly (issue #69).

Each boosting step of `bmono(..., type="quad.prog")` solves
    min_b  1/2 b'(X'WX + P) b - (X'W u)' b   subject to  D b >= 0
for the step's negative gradient u, with D the constraint's difference
operator on the B-spline coefficients. The step's coefficients must be
feasible, and no feasible point may do better: SciPy's SLSQP solves the same
problem independently and its objective must not undercut ours. The
cumulative fitted effect must also satisfy the shape on a fine grid.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

import mboostrs as mb

CONSTRAINTS = {
    "increasing": lambda p: np.diff(np.eye(p), axis=0),
    "decreasing": lambda p: -np.diff(np.eye(p), axis=0),
    "convex": lambda p: np.diff(np.eye(p), n=2, axis=0),
    "concave": lambda p: -np.diff(np.eye(p), n=2, axis=0),
}


def _data(shape, n=150, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    signal = {"increasing": np.sqrt(x), "decreasing": -x ** 2,
              "convex": (x - 0.4) ** 2, "concave": -(x - 0.6) ** 2}[shape]
    return {"x": x, "y": signal + rng.normal(0, 0.15, n)}


@pytest.mark.parametrize("shape", sorted(CONSTRAINTS))
def test_each_step_is_feasible_and_optimal(shape):
    data = _data(shape)
    fit = mb.gamboost(f'y ~ bmono(x, constraint="{shape}", knots=8)', data=data,
                      control=mb.boost_control(25, 0.1))
    store = fit._learner_store
    X = np.asarray(store.extract_design(0))
    P = np.asarray(store.extract_penalty(0))
    D = CONSTRAINTS[shape](X.shape[1])
    A = X.T @ X + P
    y = data["y"]
    previous = np.full(len(y), float(np.asarray(fit.offset_).ravel()[0]))
    for step in range(25):
        u = y - previous                       # Gaussian negative gradient
        b = X.T @ u
        beta = np.asarray(fit._coefficients[step], dtype=float)
        assert np.min(D @ beta) >= -1e-9, (shape, step)

        def objective(v, b=b):
            return 0.5 * v @ A @ v - b @ v

        oracle = minimize(objective, np.zeros_like(beta), jac=lambda v, b=b: A @ v - b,
                          constraints=[{"type": "ineq", "fun": lambda v: D @ v,
                                        "jac": lambda v: D}],
                          method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        scale = max(1.0, abs(objective(beta)))
        assert objective(beta) <= oracle.fun + 1e-8 * scale, (shape, step)
        previous = previous + 0.1 * (X @ beta)
    np.testing.assert_allclose(previous, fit.fitted_, atol=1e-10)


@pytest.mark.parametrize("shape", sorted(CONSTRAINTS))
def test_the_fitted_effect_has_the_shape_on_a_fine_grid(shape):
    data = _data(shape)
    fit = mb.gamboost(f'y ~ bmono(x, constraint="{shape}")', data=data,
                      control=mb.boost_control(200, 0.1))
    grid = np.linspace(data["x"].min(), data["x"].max(), 2001)
    curve = np.asarray(fit.predict(newdata={"x": grid}))
    d1, d2 = np.diff(curve), np.diff(curve, n=2)
    check = {"increasing": d1.min() >= -1e-10, "decreasing": d1.max() <= 1e-10,
             "convex": d2.min() >= -1e-9, "concave": d2.max() <= 1e-9}[shape]
    assert check, shape


def test_bconstrained_with_a_caller_matrix_is_feasible_and_optimal():
    # C = I: every B-spline coefficient non-negative.
    data = _data("increasing")
    p = 20 + 3 + 1
    learner = mb.Bconstrained("x", C=np.eye(p))
    fit = mb.gamboost("y ~ x", data=data, learners=[learner],
                      control=mb.boost_control(25, 0.1))
    store = fit._learner_store
    X = np.asarray(store.extract_design(0))
    P = np.asarray(store.extract_penalty(0))
    A = X.T @ X + P
    y = data["y"]
    previous = np.full(len(y), float(np.asarray(fit.offset_).ravel()[0]))
    for step in range(25):
        b = X.T @ (y - previous)
        beta = np.asarray(fit._coefficients[step], dtype=float)
        assert beta.min() >= -1e-9, step
        oracle = minimize(lambda v, b=b: 0.5 * v @ A @ v - b @ v, np.zeros(p),
                          jac=lambda v, b=b: A @ v - b, bounds=[(0, None)] * p,
                          method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12})
        mine = 0.5 * beta @ A @ beta - b @ beta
        assert mine <= oracle.fun + 1e-8 * max(1.0, abs(mine)), step
        previous = previous + 0.1 * (X @ beta)
