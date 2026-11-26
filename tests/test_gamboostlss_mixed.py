"""End-to-end coverage for heterogeneous gamboostLSS learner sets."""

import pickle

import numpy as np
import pytest
from mboostrs.gamlss import (
    DirichletLSS,
    GaussianLSS,
    LogNormalLSS,
    PoissonLSS,
    StudentTLSS,
    cvrisk_gamboostLSS,
    gamboostLSS,
    stabsel_gamboostLSS,
)

import mboostrs as mb
from conftest import r_call


def _data(seed=123, n=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.normal(size=n)
    y = 1.0 + np.sin(x) + 0.6 * z + np.exp(-0.1 + 0.25 * z) * rng.normal(size=n)
    return {"x": x, "z": z, "y": y}


def _mixed_formulas():
    return {
        "mu": "y ~ bbs(x, df=4) + bols(z)",
        "sigma": "y ~ bols(z) + bbs(x, df=4)",
    }


def test_mixed_bols_bbs_fit_replay_and_formula_order():
    data = _data()
    model = gamboostLSS(
        _mixed_formulas(),
        data,
        families=GaussianLSS(),
        control=mb.boost_control(40, 0.1),
    )

    assert model.mstop == 40
    assert model.risk_.shape == (41,)
    assert np.all(np.isfinite(model.risk_))
    assert model.risk_[-1] < model.risk_[0]
    assert set(model.selected["mu"]) <= {"x", "z"}
    assert set(model.selected["sigma"]) <= {"x", "z"}
    report = model.execution_report()
    assert report.route == "lss_cyclic"
    assert report.n_learners == 4
    assert report.scalar_learner_count == 0
    assert report.block_learner_count == 4
    assert report.requested_mstop == 80
    assert report.completed_mstop == 80

    # Training replay uses the exact prepared learners retained by Rust.
    for parameter in ("mu", "sigma"):
        fitted = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x": data["x"], "z": data["z"]},
        )
        np.testing.assert_allclose(replay, fitted, rtol=0.0, atol=2e-12)
        truncated = model.predict(parameter=parameter, type="link", mstop=11)
        assert truncated.shape == fitted.shape
        assert np.all(np.isfinite(truncated))

    # The second parameter deliberately reverses formula order. Coefficient
    # aggregation and selection names must still use that formula order.
    sigma = model.coef("sigma")
    assert list(sigma) == ["z", "x", "(offset)"]
    assert np.asarray(sigma["z"]).shape == (2,)
    assert np.asarray(sigma["x"]).ndim == 1

    effects = model.partial_effects("sigma")
    reconstructed = np.full(len(data["y"]), sigma["(offset)"], dtype=np.float64)
    for _, effect in effects.values():
        reconstructed += effect
    np.testing.assert_allclose(
        reconstructed,
        model.predict(parameter="sigma", type="link"),
        rtol=0.0,
        atol=2e-12,
    )


def test_centered_bbs_and_bols_newdata_prediction():
    data = _data(n=120)
    model = gamboostLSS(
        "y ~ bbs(x, df=4, center=True) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(25, 0.1),
    )
    newdata = {"x": data["x"][:9], "z": data["z"][:9]}
    for parameter in ("mu", "sigma"):
        prediction = model.predict(parameter=parameter, newdata=newdata, type="link")
        assert prediction.shape == (9,)
        assert np.all(np.isfinite(prediction))


def test_bns_mixes_with_bols_and_replays_weighted_centering():
    data = _data(n=96)
    model = gamboostLSS(
        "y ~ bns(x, df=7, center=True) + bols(z)",
        data,
        families=GaussianLSS(),
        weights=np.linspace(0.5, 1.5, 96),
        control=mb.boost_control(12, 0.1),
    )
    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x": data["x"], "z": data["z"]},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-12)
        assert set(model.coef(parameter)) == {"x", "z", "(offset)"}
    folds = np.ones((96, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        "y ~ bns(x, df=7, center=True) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=folds,
        grid=[[2, 2], [4, 4]],
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        "y ~ bns(x, df=7, center=True) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
    )
    assert stability.learner_names == ["x.mu", "z.mu", "x.sigma", "z.sigma"]


def test_brandom_complete_surface_and_unknown_level_error():
    rng = np.random.default_rng(733)
    n = 96
    group = np.arange(n) % 4
    z = rng.normal(size=n)
    group_effect = np.asarray([-0.8, 0.2, 0.9, -0.1])[group]
    y = 1.0 + group_effect + 0.4 * z + rng.normal(scale=0.45, size=n)
    data = {"y": y, "group": group, "z": z}
    formula = "y ~ brandom(group, df=3, center=True) + bols(z)"
    weights = np.linspace(0.6, 1.4, n)
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        weights=weights,
        control=mb.boost_control(12, 0.1),
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"group": group, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-12)
        effects = model.partial_effects(parameter)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-12)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", mstop=7),
        model.predict(parameter="mu", type="link", mstop=7),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="unknown Brandom level 4"):
        model.predict(
            parameter="mu",
            type="link",
            newdata={"group": np.asarray([0, 4]), "z": np.asarray([0.0, 0.0])},
        )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=folds,
        grid=[[2, 2], [4, 4]],
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
    )
    assert stability.learner_names == [
        "group.mu",
        "z.mu",
        "group.sigma",
        "z.sigma",
    ]


def test_bmrf_complete_surface_and_unknown_level_error():
    rng = np.random.default_rng(904)
    n = 96
    region = np.arange(n) % 4
    z = rng.normal(size=n)
    region_effect = np.asarray([-0.7, 0.1, 0.8, -0.2])[region]
    y = 1.0 + region_effect + 0.35 * z + rng.normal(scale=0.4, size=n)
    k = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    data = {"y": y, "region": region, "z": z}
    formula = "y ~ bmrf(region, K=K_regions, df=3, center=True) + bols(z)"
    env = {"K_regions": k}
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        weights=np.linspace(0.6, 1.4, n),
        control=mb.boost_control(12, 0.1),
        env=env,
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"region": region, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-12)
        effects = model.partial_effects(parameter)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-12)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", mstop=7),
        model.predict(parameter="mu", type="link", mstop=7),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="unknown Bmrf level 4"):
        model.predict(
            parameter="mu",
            type="link",
            newdata={"region": np.asarray([0, 4]), "z": np.asarray([0.0, 0.0])},
        )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=folds,
        grid=[[2, 2], [4, 4]],
        env=env,
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
        env=env,
    )
    assert stability.learner_names == [
        "region.mu",
        "z.mu",
        "region.sigma",
        "z.sigma",
    ]


def test_bspatial_complete_surface_replays_and_resamples():
    rng = np.random.default_rng(1776)
    n = 96
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    z = rng.normal(size=n)
    surface = np.sin(np.pi * x1) * np.cos(np.pi * x2)
    y = 0.8 + surface + 0.25 * z + rng.normal(scale=0.35, size=n)
    data = {"y": y, "x1": x1, "x2": x2, "z": z}
    formula = (
        "y ~ bspatial(x1, x2, knots=3, degree=2, differences=1, "
        "df=5, center=True) + bols(z)"
    )
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        weights=np.linspace(0.7, 1.3, n),
        control=mb.boost_control(10, 0.1),
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x1": x1, "x2": x2, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-11)
        effects = model.partial_effects(parameter)
        coordinates, spatial_effect = effects["x1,x2"]
        assert coordinates.shape == (n, 2)
        assert spatial_effect.shape == (n,)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-11)

    newdata = {"x1": x1[:7] + 0.01, "x2": x2[:7] - 0.01, "z": z[:7]}
    prediction = model.predict(parameter="mu", type="link", newdata=newdata)
    assert prediction.shape == (7,)
    assert np.all(np.isfinite(prediction))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", newdata=newdata),
        prediction,
        rtol=0.0,
        atol=0.0,
    )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        grid=[[2, 2], [3, 3]],
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
    )
    assert stability.learner_names == [
        "x1,x2.mu",
        "z.mu",
        "x1,x2.sigma",
        "z.sigma",
    ]


def test_bmono_complete_surface_replays_constraint_and_resamples():
    rng = np.random.default_rng(510)
    n = 96
    x = rng.uniform(0.0, 4.0, n)
    z = rng.normal(size=n)
    y = 1.2 + 1.8 * np.log(x + 0.5) + 0.2 * z + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x": x, "z": z}
    formula = (
        'y ~ bmono(x, constraint="increasing", knots=8, degree=3, '
        "differences=2, df=4) + bols(z)"
    )
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        weights=np.linspace(0.7, 1.3, n),
        control=mb.boost_control(10, 0.1),
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x": x, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-11)
        effects = model.partial_effects(parameter)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-11)

    grid_x = np.linspace(0.0, 4.0, 50)
    x_effect, effect = model.partial_effects(
        "mu", which="x", newdata={"x": grid_x, "z": np.zeros_like(grid_x)}
    )
    np.testing.assert_array_equal(x_effect, grid_x)
    assert np.all(np.diff(effect) >= -1e-9)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", mstop=6),
        model.predict(parameter="mu", type="link", mstop=6),
        rtol=0.0,
        atol=0.0,
    )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        grid=[[2, 2], [3, 3]],
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
    )
    assert stability.learner_names == ["x.mu", "z.mu", "x.sigma", "z.sigma"]


def test_bconstrained_complete_surface_replays_and_resamples():
    rng = np.random.default_rng(511)
    n = 80
    x = rng.uniform(0.0, 2.0, n)
    z = rng.normal(size=n)
    y = 0.8 + 0.5 * x**2 + 0.2 * z + rng.normal(scale=0.25, size=n)
    data = {"y": y, "x": x, "z": z}
    constraint = np.eye(3 + 2 + 1, dtype=np.float64)
    formula = (
        "y ~ bconstrained(x, C=C_positive, knots=3, degree=2, "
        "differences=1, df=4) + bols(z)"
    )
    env = {"C_positive": constraint}
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(8, 0.1),
        env=env,
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x": x, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-11)
        effects = model.partial_effects(parameter)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-11)

    grid_x = np.linspace(0.0, 2.0, 30)
    _, effect = model.partial_effects(
        "mu", which="x", newdata={"x": grid_x, "z": np.zeros_like(grid_x)}
    )
    # The default iterative violation-penalty solver is approximate; lambda2=1e6
    # keeps the residual feasibility error around 1e-5 for this design.
    assert np.all(effect >= -2e-5)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", mstop=5),
        model.predict(parameter="mu", type="link", mstop=5),
        rtol=0.0,
        atol=0.0,
    )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        grid=[[2, 2], [3, 3]],
        env=env,
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
        env=env,
    )
    assert stability.learner_names == ["x.mu", "z.mu", "x.sigma", "z.sigma"]


def test_btree_complete_surface_replays_multicolumn_design_and_resamples():
    rng = np.random.default_rng(512)
    n = 96
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.8 + 1.1 * (x > 0.0) - 0.7 * (z > 0.5) + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x": x, "z": z}
    formula = "y ~ btree(x, z, minbucket=4, minsplit=10, maxdepth=2) + bols(z)"
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(10, 0.1),
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x": x, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-12)
        effects = model.partial_effects(parameter)
        coordinates, tree_effect = effects["x,z"]
        assert coordinates.shape == (n, 2)
        assert tree_effect.shape == (n,)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-12)

    newdata = {"x": x[:7] + 0.05, "z": z[:7] - 0.05}
    prediction = model.predict(parameter="mu", type="link", newdata=newdata)
    assert prediction.shape == (7,)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", newdata=newdata),
        prediction,
        rtol=0.0,
        atol=0.0,
    )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        grid=[[2, 2], [3, 3]],
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
    )
    assert stability.learner_names == ["x,z.mu", "z.mu", "x,z.sigma", "z.sigma"]


def test_buser_complete_surface_requires_explicit_newdata_design_and_resamples():
    rng = np.random.default_rng(513)
    n = 72
    x = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(size=n)
    design = np.column_stack((x, x**2, np.sin(x)))
    penalty = np.diag([0.1, 1.0, 1.0])
    y = 0.6 + design @ np.array([0.7, -0.4, 0.5]) + 0.2 * z
    y += rng.normal(scale=0.25, size=n)
    data = {"y": y, "z": z}
    formula = "y ~ buser(custom, X=X_custom, K=K_custom, df=2.5, center=True) + bols(z)"
    env = {"X_custom": design, "K_custom": penalty}
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(8, 0.1),
        env=env,
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"z": z},
            buser_newdata={"custom": design},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=2e-12)
        effects = model.partial_effects(parameter)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=2e-12)

    with pytest.raises(ValueError, match="requires buser_newdata"):
        model.predict(parameter="mu", newdata={"z": z[:5]})
    with pytest.raises(ValueError, match="shape"):
        model.predict(
            parameter="mu",
            newdata={"z": z[:5]},
            buser_newdata={"custom": np.ones((5, 2))},
        )

    new_x = np.linspace(-1.0, 1.0, 7)
    new_design = np.column_stack((new_x, new_x**2, np.sin(new_x)))
    newdata = {"z": np.zeros(7)}
    prediction = model.predict(
        parameter="mu",
        type="link",
        newdata=newdata,
        buser_newdata={"custom": new_design},
    )
    assert prediction.shape == (7,)
    coordinates, effect = model.partial_effects(
        "mu",
        which="custom",
        newdata=newdata,
        buser_newdata={"custom": new_design},
    )
    np.testing.assert_array_equal(coordinates, np.arange(7, dtype=np.float64))
    assert effect.shape == (7,)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(
            parameter="mu",
            type="link",
            newdata=newdata,
            buser_newdata={"custom": new_design},
        ),
        prediction,
        rtol=0.0,
        atol=0.0,
    )
    updated = model.update(mstop=5)
    assert updated.mstop == 5

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        grid=[[2, 2], [3, 3]],
        env=env,
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
        env=env,
    )
    assert stability.learner_names == [
        "custom.mu",
        "z.mu",
        "custom.sigma",
        "z.sigma",
    ]


def test_explicit_bkernel_complete_surface_replays_cross_kernel_and_resamples():
    rng = np.random.default_rng(514)
    n = 56
    x = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(size=n)
    y = 0.5 + np.sin(2.0 * x) + 0.15 * z + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x": x, "z": z}
    formula = (
        "y ~ bkernel(x, bandwidth=0.45, df=4, kernel=gaussian, center=True) "
        "+ bols(z)"
    )
    model = gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(7, 0.1),
    )

    for parameter in model.param_names:
        training = model.predict(parameter=parameter, type="link")
        replay = model.predict(
            parameter=parameter,
            type="link",
            newdata={"x": x, "z": z},
        )
        np.testing.assert_allclose(replay, training, rtol=0.0, atol=3e-11)
        effects = model.partial_effects(parameter)
        reconstructed = np.full(n, model.coef(parameter)["(offset)"])
        for _, effect in effects.values():
            reconstructed += effect
        np.testing.assert_allclose(reconstructed, training, rtol=0.0, atol=3e-11)

    newdata = {"x": np.linspace(-1.0, 1.0, 6), "z": np.zeros(6)}
    prediction = model.predict(parameter="mu", type="link", newdata=newdata)
    assert prediction.shape == (6,)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict(parameter="mu", type="link", newdata=newdata),
        prediction,
        rtol=0.0,
        atol=0.0,
    )

    folds = np.ones((n, 4), dtype=np.float64)
    for fold in range(4):
        folds[fold::4, fold] = 0.0
    cv = cvrisk_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        grid=[[2, 2], [3, 3]],
    )
    assert np.all(np.isfinite(cv.risk_matrix))
    stability = stabsel_gamboostLSS(
        formula,
        data,
        families=GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
    )
    assert stability.learner_names == ["x.mu", "z.mu", "x.sigma", "z.sigma"]


def test_mixed_model_pickle_rebuilds_prediction_stores():
    data = _data(n=48)
    model = gamboostLSS(
        "y ~ bbs(x, df=4, center=True) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(7, 0.1),
    )
    restored = pickle.loads(pickle.dumps(model))
    newdata = {"x": data["x"][:8] + 0.02, "z": data["z"][:8] - 0.03}
    for parameter in model.param_names:
        np.testing.assert_allclose(
            restored.predict(newdata=newdata, parameter=parameter, type="link"),
            model.predict(newdata=newdata, parameter=parameter, type="link"),
            rtol=1e-12,
            atol=1e-12,
        )


def test_mixed_noncyclic_is_deterministic_and_uses_global_budget():
    data = _data(n=100)
    kwargs = dict(
        formula=_mixed_formulas(),
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(30, 0.1),
        method="noncyclic",
    )
    first = gamboostLSS(**kwargs)
    second = gamboostLSS(**kwargs)

    assert sum(len(parameter["selected"]) for parameter in first._params) == 30
    assert first.selected == second.selected
    np.testing.assert_array_equal(first.risk_, second.risk_)
    for parameter in ("mu", "sigma"):
        np.testing.assert_array_equal(
            first.predict(parameter=parameter), second.predict(parameter=parameter)
        )

    short = gamboostLSS(**(kwargs | {"control": mb.boost_control(11, 0.1)}))
    for parameter in first.param_names:
        np.testing.assert_allclose(
            first.predict(parameter=parameter, type="link", mstop=11),
            short.predict(parameter=parameter, type="link"),
            rtol=0.0,
            atol=2e-12,
        )


def test_weighted_and_by_mixed_fit_replays_and_predicts_newdata_with_by():
    data = _data(n=72)
    weights = np.linspace(0.4, 1.8, 72)
    model = gamboostLSS(
        "y ~ bbs(x, df=4, by=z) + bols(z)",
        data,
        families=GaussianLSS(),
        weights=weights,
        control=mb.boost_control(8, 0.1),
    )
    for parameter in model.param_names:
        assert np.all(np.isfinite(model.predict(parameter=parameter, mstop=5)))

    # newdata that duplicates the first 5 training rows must reproduce the
    # in-sample prediction: by= newdata prediction is no longer refused.
    in_sample = model.predict(parameter="mu", mstop=5)
    newdata_pred = model.predict(
        parameter="mu",
        mstop=5,
        newdata={"x": data["x"][:5], "z": data["z"][:5]},
    )
    assert newdata_pred.shape == (5,)
    np.testing.assert_allclose(newdata_pred, in_sample[:5], atol=1e-10)

    # partial_effects(newdata=...) must also succeed and reproduce the
    # in-sample bbs(x, by=z) contribution for the same rows.
    _, in_sample_effect = model.partial_effects("mu", which="x", mstop=5)
    _, newdata_effect = model.partial_effects(
        "mu",
        which="x",
        mstop=5,
        newdata={"x": data["x"][:5], "z": data["z"][:5]},
    )
    np.testing.assert_allclose(newdata_effect, in_sample_effect[:5], atol=1e-10)


@pytest.mark.parametrize("family_case", ["student", "count", "survival", "dirichlet"])
def test_mixed_blueprints_do_not_depend_on_response_shape_or_parameter_count(family_case):
    rng = np.random.default_rng(901)
    n = 54
    x = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(size=n)
    if family_case == "student":
        family = StudentTLSS()
        data = {"y": 0.4 * x + rng.standard_t(7, n), "x": x, "z": z}
        formula = "y ~ bbs(x, df=4) + bols(z)"
    elif family_case == "count":
        family = PoissonLSS()
        data = {"y": rng.poisson(np.exp(0.2 + 0.25 * x)), "x": x, "z": z}
        formula = "y ~ bbs(x, df=4) + bols(z)"
    elif family_case == "survival":
        family = LogNormalLSS()
        data = {
            "time": np.exp(0.2 * x + rng.normal(scale=0.4, size=n)),
            "status": rng.binomial(1, 0.8, n),
            "x": x,
            "z": z,
        }
        formula = "Surv(time, status) ~ bbs(x, df=4) + bols(z)"
    else:
        family = DirichletLSS(3)
        response = rng.dirichlet([2.0, 3.0, 4.0], size=n)
        data = {
            "y1": response[:, 0],
            "y2": response[:, 1],
            "y3": response[:, 2],
            "x": x,
            "z": z,
        }
        formula = "cbind(y1, y2, y3) ~ bbs(x, df=4) + bols(z)"
    model = gamboostLSS(
        formula,
        data,
        families=family,
        control=mb.boost_control(4, 0.05),
    )
    assert len(model.param_names) == family.n_params
    assert np.all(np.isfinite(model.risk_))
    for parameter in model.param_names:
        assert model.predict(parameter=parameter, type="link").shape == (n,)


def test_mixed_cvrisk_reprepares_both_learner_types_per_fold():
    data = _data(n=90)
    folds = np.ones((90, 3), dtype=np.float64)
    for fold in range(3):
        folds[fold::3, fold] = 0.0
    grid = [[5, 5], [10, 10]]
    result = cvrisk_gamboostLSS(
        _mixed_formulas(),
        data,
        families=GaussianLSS(),
        control=mb.boost_control(10, 0.1),
        folds=folds,
        grid=grid,
    )
    assert result.risk_matrix.shape == (3, 2)
    assert np.all(np.isfinite(result.risk_matrix))
    assert result.optimal_mstop in grid


def test_mixed_stability_selection_tracks_parameter_qualified_learners():
    data = _data(n=80)
    folds = np.zeros((80, 8), dtype=np.float64)
    for fold in range(8):
        folds[np.arange(80) % 8 != fold, fold] = 1.0
    result = stabsel_gamboostLSS(
        _mixed_formulas(),
        data,
        families=GaussianLSS(),
        control=mb.boost_control(12, 0.1),
        folds=folds,
        q=1,
        cutoff=0.6,
        sampling_type="MB",
    )
    assert result.phat.shape == (4, 12)
    assert result.learner_names == ["x.mu", "z.mu", "z.sigma", "x.sigma"]
    assert np.all(np.isfinite(result.phat))


@pytest.mark.r_parity
def test_buser_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(515); n <- 84
            x <- runif(n, -1.5, 1.5); z <- rnorm(n)
            X_custom <- cbind(x, x^2, sin(x))
            K_custom <- diag(c(0.1, 1, 1))
            y <- 0.6 + X_custom %*% c(0.7, -0.4, 0.5) + 0.2 * z +
                 rnorm(n, sd = 0.25)
            dat <- data.frame(y = as.double(y), z = z)
            fit <- gamboostLSS(
                y ~ buser(X_custom, K = K_custom, df = 2.5) + bols(z),
                data = dat, families = GaussianLSS(),
                control = boost_control(mstop = 9, nu = 0.1)
            )
            list(
                X = X_custom, K = K_custom, z = z, y = as.double(y),
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    design = np.asarray(result.rx2("X"), dtype=np.float64)
    penalty = np.asarray(result.rx2("K"), dtype=np.float64)
    data = {
        "z": np.asarray(result.rx2("z"), dtype=np.float64),
        "y": np.asarray(result.rx2("y"), dtype=np.float64),
    }
    model = gamboostLSS(
        "y ~ buser(custom, X=X_custom, K=K_custom, df=2.5) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(9, 0.1),
        env={"X_custom": design, "K_custom": penalty},
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=2e-6,
            atol=2e-6,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


@pytest.mark.r_parity
def test_mixed_bols_bbs_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(123); n <- 180
            x <- runif(n, -2, 2); z <- rnorm(n)
            y <- 1 + sin(x) + 0.6 * z + exp(-0.1 + 0.25 * z) * rnorm(n)
            dat <- data.frame(y = y, x = x, z = z)
            fit <- gamboostLSS(
                y ~ bbs(x, df = 4) + bols(z), data = dat,
                families = GaussianLSS(),
                control = boost_control(mstop = 40, nu = 0.1)
            )
            list(
                x = x, z = z, y = y,
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    data = {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in ("x", "z", "y")
    }
    model = gamboostLSS(
        "y ~ bbs(x, df=4) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(40, 0.1),
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=1e-8,
            atol=1e-8,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


@pytest.mark.r_parity
def test_brandom_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(733); n <- 96
            group <- factor(rep(letters[1:4], length.out = n))
            z <- rnorm(n)
            ge <- c(-0.8, 0.2, 0.9, -0.1)[as.integer(group)]
            y <- 1 + ge + 0.4 * z + rnorm(n, sd = 0.45)
            dat <- data.frame(y = y, group = group, z = z)
            fit <- gamboostLSS(
                y ~ brandom(group, df = 3) + bols(z), data = dat,
                families = GaussianLSS(),
                control = boost_control(mstop = 12, nu = 0.1)
            )
            list(
                group = as.integer(group), z = z, y = y,
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    data = {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in ("group", "z", "y")
    }
    model = gamboostLSS(
        "y ~ brandom(group, df=3) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(12, 0.1),
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=2e-6,
            atol=2e-6,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


@pytest.mark.r_parity
def test_bmrf_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(904); n <- 96
            group <- factor(rep(letters[1:4], length.out = n))
            z <- rnorm(n)
            ge <- c(-0.7, 0.1, 0.8, -0.2)[as.integer(group)]
            y <- 1 + ge + 0.35 * z + rnorm(n, sd = 0.4)
            K <- matrix(c(
                 1, -1,  0,  0,
                -1,  2, -1,  0,
                 0, -1,  2, -1,
                 0,  0, -1,  1
            ), 4, 4, byrow = TRUE)
            rownames(K) <- colnames(K) <- letters[1:4]
            dat <- data.frame(y = y, group = group, z = z)
            fit <- gamboostLSS(
                y ~ bmrf(group, bnd = K, df = 3) + bols(z), data = dat,
                families = GaussianLSS(),
                control = boost_control(mstop = 12, nu = 0.1)
            )
            list(
                group = as.integer(group), z = z, y = y, K = K,
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    data = {
        "group": np.asarray(result.rx2("group"), dtype=np.float64) - 1.0,
        "z": np.asarray(result.rx2("z"), dtype=np.float64),
        "y": np.asarray(result.rx2("y"), dtype=np.float64),
    }
    k = np.asarray(result.rx2("K"), dtype=np.float64)
    model = gamboostLSS(
        "y ~ bmrf(group, K=K_regions, df=3) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(12, 0.1),
        env={"K_regions": k},
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=2e-6,
            atol=2e-6,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


@pytest.mark.r_parity
def test_bspatial_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(1776); n <- 96
            x1 <- runif(n, -1, 1); x2 <- runif(n, -1, 1); z <- rnorm(n)
            surface <- sin(pi * x1) * cos(pi * x2)
            y <- 0.8 + surface + 0.25 * z + rnorm(n, sd = 0.35)
            dat <- data.frame(y = y, x1 = x1, x2 = x2, z = z)
            fit <- gamboostLSS(
                y ~ bspatial(x1, x2, knots = 3, degree = 2,
                             differences = 1, df = 5, center = TRUE) + bols(z),
                data = dat, families = GaussianLSS(),
                control = boost_control(mstop = 10, nu = 0.1)
            )
            list(
                x1 = x1, x2 = x2, z = z, y = y,
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    data = {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in ("x1", "x2", "z", "y")
    }
    model = gamboostLSS(
        "y ~ bspatial(x1, x2, knots=3, degree=2, differences=1, "
        "df=5, center=True) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(10, 0.1),
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=2e-6,
            atol=2e-6,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


@pytest.mark.r_parity
def test_bmono_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(510); n <- 96
            x <- runif(n, 0, 4); z <- rnorm(n)
            y <- 1.2 + 1.8 * log(x + 0.5) + 0.2 * z + rnorm(n, sd = 0.3)
            dat <- data.frame(y = y, x = x, z = z)
            fit <- gamboostLSS(
                y ~ bmono(x, constraint = "increasing", knots = 8,
                          degree = 3, differences = 2, df = 4) + bols(z),
                data = dat, families = GaussianLSS(),
                control = boost_control(mstop = 10, nu = 0.1)
            )
            list(
                x = x, z = z, y = y,
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    data = {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in ("x", "z", "y")
    }
    model = gamboostLSS(
        'y ~ bmono(x, constraint="increasing", knots=8, degree=3, '
        "differences=2, df=4) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(10, 0.1),
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=1e-3,
            atol=1e-3,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


@pytest.mark.r_parity
def test_btree_matches_r_gamboostlss():
    import rpy2.robjects as ro

    reference = ro.r(
        """
        function() {
            if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
            suppressMessages(library(gamboostLSS))
            set.seed(512); n <- 96
            x <- rnorm(n); z <- rnorm(n)
            y <- 0.8 + 1.1 * (x > 0) + 0.2 * z + rnorm(n, sd = 0.3)
            dat <- data.frame(y = y, x = x, z = z)
            fit <- gamboostLSS(
                y ~ btree(x) + bols(z), data = dat, families = GaussianLSS(),
                control = boost_control(mstop = 10, nu = 0.1)
            )
            list(
                x = x, z = z, y = y,
                mu = as.numeric(predict(fit, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(fit, parameter = "sigma", type = "link")),
                selected_mu = as.integer(selected(fit$mu)),
                selected_sigma = as.integer(selected(fit$sigma))
            )
        }
        """
    )
    result = r_call(reference, packages=("gamboostLSS",))

    data = {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in ("x", "z", "y")
    }
    model = gamboostLSS(
        "y ~ btree(x) + bols(z)",
        data,
        families=GaussianLSS(),
        control=mb.boost_control(10, 0.1),
    )
    for index, (parameter, selected_key) in enumerate(
        (("mu", "selected_mu"), ("sigma", "selected_sigma"))
    ):
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            np.asarray(result.rx2(parameter), dtype=np.float64),
            rtol=2e-6,
            atol=2e-6,
        )
        assert model._params[index]["selected"] == (
            np.asarray(result.rx2(selected_key), dtype=np.int64) - 1
        ).tolist()


def test_tiny_scale_response_stops_with_guided_error():
    # Decimal-fraction returns (sd ~ 0.02) make GaussianLSS's mu gradient
    # (y - mu)/sigma^2 overshoot by ~1/sigma^2 per boosting cycle, so the
    # joint risk grows geometrically. The engine must stop the path with a
    # typed error naming the remedy instead of returning a meaningless fit.
    rng = np.random.default_rng(7)
    n = 600
    x = rng.uniform(-2.0, 2.0, n)
    y = 0.0002 + 0.02 * rng.standard_normal(n)
    formulas = {"mu": "y ~ bols(x)", "sigma": "y ~ bols(x)"}
    with pytest.raises(RuntimeError, match="standard deviation"):
        gamboostLSS(
            formulas,
            {"x": x, "y": y},
            family=GaussianLSS(),
            control=mb.boost_control(100, 0.1),
            response_scale=None,
        )
    # The default fits the response divided by its SD, where the same steps
    # are of order one, and maps the model back to the original units.
    fit = gamboostLSS(formulas, {"x": x, "y": y}, family=GaussianLSS(),
                      control=mb.boost_control(100, 0.1))
    sigma = np.asarray(fit.predict(type="response")["sigma"], dtype=float)
    assert np.all(np.isfinite(sigma))
    assert 0.01 < float(np.median(sigma)) < 0.04


def _per_step_eta(model, k, newdata):
    """`predict(newdata=)` the slow way: one basis rebuild per boosting step.

    `LSSModel.predict` groups a learner's coefficients over the steps that
    chose it and evaluates its basis once. This is the ungrouped replay it
    replaced, kept here as the reference the grouped path has to reproduce.
    """
    from mboostrs.gamlss import _gamboostlss_by_data, _gamboostlss_x_data

    p = model._params[k]
    x_data = [
        _gamboostlss_x_data(newdata, learner, require_external_buser=True)
        for learner in p["learners"]
    ]
    by_data = _gamboostlss_by_data(newdata, p["learners"])
    coefs = [np.asarray(value, dtype=np.float64).tolist() for value in p["coefs"]]
    return np.asarray(
        p["learner_store"].predict_newdata(
            p["offset"], list(p["selected"]), coefs, x_data, None, by_data
        ),
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    "formulas",
    [
        {"mu": "y ~ bbs(x1) + bbs(x2) + bols(x3)", "sigma": "y ~ bbs(x3) + bols(x1)"},
        {"mu": "y ~ bbs(x1, bins=25) + bols(x2)", "sigma": "y ~ bbs(x2, bins=25)"},
        {"mu": "y ~ bbs(x1) + btree(x2)", "sigma": "y ~ bbs(x2)"},
        {"mu": "y ~ bbs(x1) + bols(x2, by='x3')", "sigma": "y ~ bols(x2)"},
    ],
    ids=["splines", "binned", "tree", "varying-coefficient"],
)
@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
def test_grouped_newdata_predict_matches_the_per_step_replay(formulas, method):
    rng = np.random.default_rng(5)
    n = 400
    data = {name: rng.uniform(0.0, 1.0, n) for name in ("x1", "x2", "x3")}
    data["y"] = rng.normal(
        1.0 + 2.0 * np.sin(2 * np.pi * data["x1"]) + data["x2"],
        np.exp(-0.5 + 0.5 * data["x3"]),
    )
    model = gamboostLSS(formulas, data=data, families=GaussianLSS(),
                        control=mb.boost_control(60, 0.1), method=method)

    newdata = {name: data[name][:120] for name in ("x1", "x2", "x3")}
    for k, name in enumerate(model.param_names):
        grouped = np.asarray(model.predict(newdata=newdata, parameter=name),
                             dtype=np.float64)
        np.testing.assert_allclose(grouped, _per_step_eta(model, k, newdata),
                                   rtol=0, atol=1e-10)


def test_newdata_predict_on_the_training_frame_reproduces_the_fitted_values():
    rng = np.random.default_rng(6)
    n = 300
    data = {name: rng.uniform(0.0, 1.0, n) for name in ("x1", "x2")}
    data["y"] = rng.normal(1.0 + data["x1"], np.exp(-0.5 + 0.5 * data["x2"]))
    model = gamboostLSS({"mu": "y ~ bbs(x1) + bols(x2)", "sigma": "y ~ bbs(x2)"},
                        data=data, families=GaussianLSS(),
                        control=mb.boost_control(80, 0.1), method="noncyclic")

    frame = {"x1": data["x1"], "x2": data["x2"]}
    for k, name in enumerate(model.param_names):
        np.testing.assert_allclose(
            np.asarray(model.predict(newdata=frame, parameter=name), dtype=np.float64),
            np.asarray(model._params[k]["fitted"], dtype=np.float64),
            rtol=0, atol=1e-10,
        )
