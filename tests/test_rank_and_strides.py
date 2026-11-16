"""Rank-deficient designs and strided inputs (issue #69).

A learner whose design repeats a column has a singular X'X. mboost stops
there ("system is exactly singular"); mboostrs fits. Every step is a
projection onto the design's column span, so the fitted path must equal the
one from the design without the repeated column, and the repeated column's
coefficients, though their split is not unique, must sum to the reduced
design's coefficient.

Columns taken as strided views of a larger array must fit as their
contiguous copies do.
"""

import numpy as np

import mboostrs as mb


def _xyz(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    return x, z, x + 0.5 * z + rng.normal(size=n)


def test_a_repeated_column_leaves_the_fitted_path_unchanged():
    x, z, y = _xyz()
    data = {"y": y, "u": np.zeros(len(y))}

    def fit(design):
        p = design.shape[1]
        learner = mb.Buser("u", X=design, K=np.zeros((p, p)), lambda_value=0.0)
        return mb.gamboost("y ~ u", data=data, learners=[learner],
                           control=mb.boost_control(60, 0.1))

    repeated = fit(np.column_stack([x, x, z]))
    reduced = fit(np.column_stack([x, z]))
    np.testing.assert_allclose(repeated.fitted_, reduced.fitted_, atol=1e-8)
    np.testing.assert_allclose(repeated.risk_, reduced.risk_, rtol=1e-9)
    split = np.asarray(repeated.coef()["u"])
    whole = np.asarray(reduced.coef()["u"])
    assert np.all(np.isfinite(split))
    np.testing.assert_allclose([split[0] + split[1], split[2]], whole, atol=1e-8)


def test_strided_columns_fit_as_their_contiguous_copies():
    rng = np.random.default_rng(1)
    block = rng.normal(size=(400, 3))
    strided = {"x": block[::2, 0], "z": block[::2, 1], "y": block[::2, 2] + block[::2, 0]}
    assert not strided["x"].flags["C_CONTIGUOUS"]
    contiguous = {k: np.ascontiguousarray(v) for k, v in strided.items()}
    for formula in ("y ~ bbs(x) + bols(z)", "y ~ x + z"):
        fit = mb.gamboost if "bbs" in formula else mb.glmboost
        a = fit(formula, data=strided, control=mb.boost_control(50))
        b = fit(formula, data=contiguous, control=mb.boost_control(50))
        np.testing.assert_array_equal(a.fitted_, b.fitted_)
        new = {"x": block[1::4, 0], "z": block[1::4, 1]}
        np.testing.assert_array_equal(
            a.predict(newdata=new),
            b.predict(newdata={k: np.ascontiguousarray(v) for k, v in new.items()}))
