"""Hat values and AIC degrees of freedom for fits containing a tree term.

A `btree` step's hat matrix is the averaging matrix of the partition it
was fitted with, conditional on that partition; split selection is not
counted. Before this, any fit whose path selected a tree raised
`NotImplementedError` from `hatvalues()` and `aic()`. The smooth-only
pins below were recorded on the pre-change code and must not move: the
tree work adds a resolution path for tree steps and changes nothing
about the matrices of smooth learners. A `bmono` term still refuses,
with the message naming its constrained solve.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import Binomial, Gaussian


def _data():
    """The exact generator the smooth-only pins were recorded with."""
    rng = np.random.default_rng(20261109)
    n = 60
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    _w = rng.uniform(0.5, 1.5, size=n)  # drawn for stream parity; unused
    y_g = 1.2 * x + np.sin(2.0 * z) + rng.normal(scale=0.3, size=n)
    latent = 1.5 * x + np.sin(2.0 * z)
    p = 1.0 / (1.0 + np.exp(-latent))
    y_b = (rng.uniform(size=n) < p).astype(float)
    y_t = (x + (z > 0.0) * 1.5 + rng.normal(scale=0.4, size=n) > 0.6).astype(float)
    return x, z, y_g, y_b, y_t


def test_btree_binomial_fit_reports_hatvalues_and_aic_df():
    x, z, _y_g, _y_b, y_t = _data()
    model = mb.gamboost(
        "y ~ bbs(x, df=4) + btree(z)", data={"y": y_t, "x": x, "z": z},
        family=Binomial(), control=mb.boost_control(30, 0.1),
    )
    assert "z" in set(model.selected), "the tree term must be on the path"

    hat = model.hatvalues()
    assert hat.shape == (60,)
    assert np.all(np.isfinite(hat))
    assert np.all(hat >= 0.0)

    result = model.aic(method="classical")
    assert np.all(np.isfinite(result.df))
    assert result.df[-1] > 0.0
    # The weighted hat diagonal sums to the final step's df.
    assert abs(hat.sum() - result.df[-1]) < 1e-8


def test_btree_gaussian_fit_reports_hatvalues_and_aic_df():
    x, z, y_g, _y_b, _y_t = _data()
    model = mb.gamboost(
        "y ~ bbs(x, df=4) + btree(z)", data={"y": y_g, "x": x, "z": z},
        family=Gaussian(), control=mb.boost_control(30, 0.1),
    )
    assert "z" in set(model.selected), "the tree term must be on the path"

    hat = model.hatvalues()
    assert hat.shape == (60,)
    assert np.all(np.isfinite(hat))

    result = model.aic(method="corrected")
    assert np.all(np.isfinite(result.df))
    assert abs(hat.sum() - result.df[-1]) < 1e-8


def test_smooth_only_gaussian_hatvalues_unchanged():
    x, z, y_g, _y_b, _y_t = _data()
    model = mb.gamboost(
        "y ~ bols(x) + bbs(z, df=4)", data={"y": y_g, "x": x, "z": z},
        family=Gaussian(), control=mb.boost_control(25, 0.1),
    )
    hat = model.hatvalues()
    np.testing.assert_allclose(
        hat[:6],
        [0.052645740555170337, 0.034064330649230846, 0.064265814209330613,
         0.030903722936318601, 0.40870085007644208, 0.034219040469418975],
        rtol=0, atol=1e-12,
    )
    assert abs(hat.sum() - 3.6885967654691991) < 1e-12
    result = model.aic(method="corrected")
    assert abs(result.df[-1] - 3.6885967654691996) < 1e-12


def test_smooth_only_binomial_hatvalues_unchanged():
    x, z, _y_g, y_b, _y_t = _data()
    model = mb.gamboost(
        "y ~ bols(x) + bbs(z, df=4)", data={"y": y_b, "x": x, "z": z},
        family=Binomial(), control=mb.boost_control(25, 0.1),
    )
    hat = model.hatvalues()
    np.testing.assert_allclose(
        hat[:6],
        [0.049767260339675573, 0.034958863425734517, 0.055950895675789619,
         0.031712064175948039, 0.27492299837953005, 0.0341403621300033],
        rtol=0, atol=1e-12,
    )
    assert abs(hat.sum() - 3.2464860631439789) < 1e-12
    result = model.aic(method="classical")
    assert abs(result.df[-1] - 3.2464860631439785) < 1e-12


def test_bmono_still_refuses_with_the_constrained_solve_reason():
    x, z, y_g, _y_b, _y_t = _data()
    model = mb.gamboost(
        "y ~ bbs(x, df=4) + bmono(z)", data={"y": y_g, "x": x, "z": z},
        family=Gaussian(), control=mb.boost_control(25, 0.1),
    )
    assert "z" in set(model.selected), "the bmono term must be on the path"
    with pytest.raises(NotImplementedError) as excinfo:
        model.hatvalues()
    message = str(excinfo.value)
    assert "`bmono` solves a constrained problem" in message
    assert "conditional on the partition" in message
    with pytest.raises(NotImplementedError):
        model.aic(method="corrected")
