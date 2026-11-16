"""Structured engine-route reporting and persistence contracts."""

import json
import pickle

import numpy as np

import mboostrs as mb
import pytest


def _data(n=40):
    x = np.linspace(-1.0, 1.0, n)
    return {"x": x, "y": x**2 + 0.2 * x}


def test_glmboost_reports_packed_scalar_route_and_json_schema():
    model = mb.glmboost(
        "y ~ x",
        data=_data(),
        control=mb.boost_control(7, 0.1),
    )
    report = model.execution_report()

    assert isinstance(report, mb.ExecutionReport)
    assert report.schema_version == 1
    assert report.route == "packed_scalar"
    assert report.n_observations == 40
    assert report.n_learners == 2  # intercept + x
    assert report.completed_mstop == 7
    assert report.gram_eligible is False
    assert report.fallback_reasons == ("pure_scalar_model",)
    assert report.scalar_learner_count == 2
    assert report.block_learner_count == 0
    assert report.tree_learner_count == 0
    assert report.cholesky_solver_count == 0
    assert report.lu_solver_count == 0
    assert report.regularized_solver_count == 0
    assert report.max_solver_jitter is None
    assert report.factor_cache_bytes == 0
    assert report.gram_cache_bytes == 0
    assert report.calibration_count == 0
    assert report.tie_selection_count == 0
    assert report.calibration_converged_count == 0
    assert report.calibration_limit_count == 0
    assert report.calibration_failure_count == 0
    assert report.total_ns is None
    json.dumps(report.to_dict())


def test_gamboost_reports_gram_and_family_fallback_routes():
    data = _data()
    gaussian = mb.gamboost(
        "y ~ bbs(x)",
        data=data,
        family=mb.Gaussian(),
        control=mb.boost_control(6, 0.1),
    )
    assert gaussian.execution_report().route == "gaussian_gram"
    assert gaussian.execution_report().gram_eligible
    assert gaussian.execution_report().block_learner_count == 1
    assert gaussian.execution_report().cholesky_solver_count == 1
    assert gaussian.execution_report().factor_cache_bytes > 0
    assert gaussian.execution_report().gram_cache_bytes > 0

    poisson_data = {"x": data["x"], "y": np.arange(1, 41, dtype=float)}
    poisson = mb.gamboost(
        "y ~ bbs(x)",
        data=poisson_data,
        family=mb.Poisson(),
        control=mb.boost_control(3, 0.01),
    )
    report = poisson.execution_report()
    assert report.route == "general"
    assert "family_does_not_support_gram" in report.fallback_reasons
    assert report.block_learner_count == 1
    assert report.cholesky_solver_count == 1
    assert report.gram_cache_bytes == 0


def test_summary_instrumentation_preserves_results_and_records_timings(capsys):
    data = _data()
    off = mb.glmboost("y ~ x", data=data, control=mb.boost_control(5, 0.1))
    summary = mb.glmboost(
        "y ~ x",
        data=data,
        control=mb.boost_control(5, 0.1, trace=True),
    )

    np.testing.assert_array_equal(summary.fitted_, off.fitted_)
    np.testing.assert_array_equal(summary.risk_, off.risk_)
    assert summary.selected == off.selected
    report = summary.execution_report()
    rendered = capsys.readouterr().err
    assert "mboostrs execution: route=packed_scalar" in rendered
    assert "timing_ms:" in rendered
    assert report.instrumentation == "summary"
    assert report.setup_ns is not None
    assert report.iteration_ns is not None
    assert report.binding_ns is not None
    assert report.basis_ns is not None
    assert report.factorization_ns is not None
    assert report.cross_gram_ns is None
    assert report.reconstruction_ns is None
    assert report.total_ns is not None
    # Equality intentionally ignores machine-dependent timings.
    assert report == mb.ExecutionReport.from_mapping(
        {**report.to_dict(), "setup_ns": 1, "iteration_ns": 2, "total_ns": 3}
    )


def test_trace_renders_gram_route(capsys):
    model = mb.gamboost(
        "y ~ bbs(x)",
        data=_data(),
        control=mb.boost_control(3, 0.1, trace=True),
    )
    rendered = capsys.readouterr().err
    report = model.execution_report()
    assert "mboostrs execution: route=gaussian_gram" in rendered
    assert "cross_gram=" in rendered
    assert report.cross_gram_ns is not None
    assert report.reconstruction_ns is not None
    assert report.binding_ns is not None
    assert report.basis_ns is not None
    assert report.factorization_ns is not None
    assert report.factorization_ns > 0


def test_lss_reports_aggregate_cyclic_parameter_updates_and_pickle():
    model = mb.glmboostLSS(
        "y ~ x",
        data=_data(),
        families=mb.GaussianLSS(),
        control=mb.boost_control(4, 0.1),
        mstop={"mu": 4, "sigma": 2},
    )
    report = model.execution_report()

    assert report.route == "lss_cyclic"
    assert report.n_observations == 40
    assert report.n_learners == 4  # intercept + x for each parameter
    assert report.total_basis_columns == 4
    assert report.requested_mstop == 6
    assert report.completed_mstop == 6
    assert report.instrumentation == "off"
    assert report.scalar_learner_count == 4
    assert report.block_learner_count == 0
    assert report.calibration_count == 0
    assert pickle.loads(pickle.dumps(model)).execution_report() == report


def test_lss_noncyclic_and_smooth_routes_render_trace(capsys):
    data = _data()
    noncyclic_off = mb.glmboostLSS(
        "y ~ x",
        data=data,
        families=mb.GaussianLSS(),
        method="noncyclic",
        control=mb.boost_control(5, 0.1),
    )
    noncyclic = mb.glmboostLSS(
        "y ~ x",
        data=data,
        families=mb.GaussianLSS(),
        method="noncyclic",
        control=mb.boost_control(5, 0.1, trace=True),
    )
    noncyclic_report = noncyclic.execution_report()
    rendered = capsys.readouterr().err
    assert noncyclic_report.route == "lss_noncyclic"
    assert noncyclic_report.requested_mstop == 5
    assert noncyclic_report.completed_mstop == 5
    assert noncyclic_report.total_ns is not None
    assert noncyclic_report.binding_ns is not None
    assert "mboostrs execution: route=lss_noncyclic" in rendered
    np.testing.assert_array_equal(noncyclic.risk_, noncyclic_off.risk_)
    assert noncyclic.selected == noncyclic_off.selected
    for parameter in noncyclic.param_names:
        np.testing.assert_array_equal(
            noncyclic.predict(parameter=parameter, type="link"),
            noncyclic_off.predict(parameter=parameter, type="link"),
        )

    smooth = mb.gamboostLSS(
        "y ~ bbs(x)",
        data=data,
        families=mb.GaussianLSS(),
        control=mb.boost_control(3, 0.1),
    )
    smooth_report = smooth.execution_report()
    assert smooth_report.route == "lss_cyclic"
    assert smooth_report.n_learners == 2
    assert smooth_report.requested_mstop == 6
    assert smooth_report.completed_mstop == 6
    assert smooth_report.block_learner_count == 2
    assert smooth_report.cholesky_solver_count == 2


def test_standard_path_frame_is_derived_and_json_serializable():
    model = mb.gamboost(
        "y ~ bbs(x)", data=_data(), control=mb.boost_control(4, 0.1)
    )
    rows = model.path_frame(output="records")

    assert len(rows) == 4
    assert [row["iteration"] for row in rows] == [1, 2, 3, 4]
    assert all(row["learner"] == "x" for row in rows)
    assert rows[0]["risk"] == model.risk_[1]
    assert rows[0]["risk_change"] == model.risk_[0] - model.risk_[1]
    assert rows[0]["scaled_coefficient_norm"] == 0.1 * rows[0]["coefficient_norm"]
    json.dumps(rows)

    sliced = model[:2]
    assert len(sliced.path_frame(output="records")) == 2


def test_standard_path_frame_unifies_scalar_nuisance_values():
    x = np.linspace(-1.0, 1.0, 40)
    model = mb.glmboost(
        "y ~ x",
        data={"x": x, "y": np.maximum(0, np.rint(np.exp(0.5 + x))).astype(float)},
        family=mb.NBinomial(),
        control=mb.boost_control(4, 0.1),
    )
    rows = model.path_frame(output="records")

    assert len(model._nuisance_vec_path) == 5
    assert all(len(row["nuisance"]) == 1 for row in rows)
    assert all(np.isfinite(row["nuisance"][0]) for row in rows)
    assert model.execution_report().calibration_count == 5


def test_execution_report_counts_only_active_huber_calibration():
    adaptive = mb.glmboost(
        "y ~ x", data=_data(), family=mb.Huber(), control=mb.boost_control(3, 0.1)
    )
    fixed = mb.glmboost(
        "y ~ x",
        data=_data(),
        family=mb.Huber(delta=1.5),
        control=mb.boost_control(3, 0.1),
    )

    assert adaptive.execution_report().calibration_count == 4
    assert fixed.execution_report().calibration_count == 0


def test_execution_report_counts_ties_and_propodds_calibration_stops():
    x = np.linspace(-1.0, 1.0, 30)
    tied = mb.glmboost(
        "y ~ x1 + x2",
        data={"y": x, "x1": x, "x2": x.copy()},
        control=mb.boost_control(4, 0.1),
    )
    assert tied.selected == ["x1", "x1", "x1", "x1"]
    assert tied.execution_report().tie_selection_count == 4

    ordered = mb.glmboost(
        "y ~ x",
        data={
            "y": np.tile(np.array([1.0, 2.0, 3.0]), 10),
            "x": x,
        },
        family=mb.PropOdds(n_levels=3),
        control=mb.boost_control(3, 0.1),
    )
    report = ordered.execution_report()
    assert report.calibration_count == 4
    assert (
        report.calibration_converged_count
        + report.calibration_limit_count
        + report.calibration_failure_count
        == report.calibration_count
    )
    assert report.calibration_failure_count == 0


def test_lss_path_frame_preserves_global_update_and_risk_granularity():
    cyclic = mb.glmboostLSS(
        "y ~ x",
        data=_data(),
        families=mb.GaussianLSS(),
        control=mb.boost_control(3, 0.1),
        mstop={"mu": 3, "sigma": 2},
    )
    rows = cyclic.path_frame(output="records")
    assert [row["parameter"] for row in rows] == ["mu", "sigma", "mu", "sigma", "mu"]
    assert [row["cycle"] for row in rows] == [1, 1, 2, 2, 3]
    assert rows[0]["risk"] is None
    assert rows[1]["risk"] == cyclic.risk_[1]
    assert rows[4]["risk"] == cyclic.risk_[3]

    noncyclic = mb.glmboostLSS(
        "y ~ x",
        data=_data(),
        families=mb.GaussianLSS(),
        method="noncyclic",
        control=mb.boost_control(3, 0.1),
    )
    noncyclic_rows = noncyclic.path_frame(output="records")
    assert all(row["cycle"] is None for row in noncyclic_rows)
    assert all(row["risk"] is not None for row in noncyclic_rows)


def test_execution_report_survives_slice_and_pickle():
    model = mb.gamboost(
        "y ~ bbs(x)",
        data=_data(),
        control=mb.boost_control(8, 0.1),
    )
    sliced = model[:3]
    assert sliced.execution_report().route == model.execution_report().route
    assert sliced.execution_report().completed_mstop == 3

    restored = pickle.loads(pickle.dumps(model))
    assert restored.execution_report() == model.execution_report()


def test_path_row_mirrors_the_core_schema_field_for_field():
    """Enhancement-hunt 12.0b. `PathRow` existed in Rust and Python with the
    same name and three field differences, the sharpest being `parameter`:
    an index on one side and a name on the other. The core struct is now the
    schema — `parameter_index`, `learner_index` — and the bindings mirror it,
    adding the two resolved names and the nu-scaled norm on top."""
    import dataclasses

    from mboostrs.report import PathRow

    names = [f.name for f in dataclasses.fields(PathRow)]
    core_schema = ["iteration", "cycle", "parameter_index", "learner_index",
                   "risk", "risk_change", "coefficient_norm", "nuisance"]
    for field_name in core_schema:
        assert field_name in names, f"{field_name} is in the core row but not here"
    # the binding-only enrichments, each resolving one of the core's indices
    for field_name in ("parameter", "learner", "scaled_coefficient_norm"):
        assert field_name in names
    # no field carries a different meaning under a shared name
    assert names.index("parameter_index") < names.index("parameter")


def test_lss_path_rows_carry_both_the_parameter_index_and_its_name():
    from mboostrs.gamlss import GaussianLSS

    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    data = {"x": x, "y": 1.0 + 2.0 * x + rng.normal(0, 0.5, n)}
    fit = mb.glmboostLSS(
        {"mu": "y ~ x", "sigma": "y ~ x"}, data=data, families=GaussianLSS(),
        control=mb.boost_control(20, 0.1),
    )
    rows = fit.path_frame(output="records")
    assert rows
    param_names = list(fit.param_names) if hasattr(fit, "param_names") else ["mu", "sigma"]
    for row in rows:
        assert isinstance(row["parameter_index"], int)
        assert row["parameter"] == param_names[row["parameter_index"]]


def test_path_frame_works_for_every_gamboostlss_shape():
    """A pre-existing defect, surfaced by `LSSModel.varimp()` becoming its
    first internal caller: `lss_path_rows` branched on
    `param.get("kind") != "bbs"`, testing a value from the *learner-spec*
    vocabulary against a params dict whose `kind` is `"gam"` or absent. No
    `gamboostLSS` parameter ever matched, so every `gamboostLSS` fit took the
    `glmboostLSS` branch and `path_frame()` raised `KeyError: 'coefs_flat'`.
    The discriminator is now which coefficient shape the parameter stores."""
    from mboostrs.gamlss import GaussianLSS

    rng = np.random.default_rng(4)
    n = 200
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"x": x, "z": z,
            "y": 1 + 2 * x + rng.normal(0, np.exp(0.2 + 0.15 * z), n)}

    smooth_only = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bbs(z, df=4)"},
        data=data, families=GaussianLSS(), control=mb.boost_control(20, 0.1))
    mixed = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4) + bols(z)", "sigma": "y ~ bols(z)"},
        data=data, families=GaussianLSS(), control=mb.boost_control(20, 0.1))
    noncyclic = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4) + bols(z)", "sigma": "y ~ bols(z)"},
        data=data, families=GaussianLSS(), control=mb.boost_control(20, 0.1),
        method="noncyclic")
    linear = mb.glmboostLSS(
        {"mu": "y ~ x + z", "sigma": "y ~ z"},
        data=data, families=GaussianLSS(), control=mb.boost_control(20, 0.1))

    for model in (smooth_only, mixed, noncyclic, linear):
        rows = model.path_frame(output="records")
        assert rows
        assert all(row["learner"] for row in rows)

    # A gamboostLSS parameter names every learner; a glmboostLSS one has the
    # intercept learner prepended, since it is not in `predictor_names`.
    assert "(Intercept)" not in {r["learner"] for r in mixed.path_frame(output="records")}
    assert "(Intercept)" in {r["learner"] for r in linear.path_frame(output="records")}


def test_lss_varimp_refuses_a_cyclic_fit_rather_than_misattributing():
    """A cyclic fit records risk once per cycle, so the whole cycle's drop
    would land on whichever update came last — the dominant predictor scores
    0.0 in a confident-looking table."""
    from mboostrs.gamlss import GaussianLSS

    rng = np.random.default_rng(4)
    n = 300
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"x": x, "z": z,
            "y": 1 + 2 * x + rng.normal(0, np.exp(0.2 + 0.15 * z), n)}
    formulas = {"mu": "y ~ bbs(x, df=4) + bols(z)", "sigma": "y ~ bols(z)"}

    cyclic = mb.gamboostLSS(formulas, data=data, families=GaussianLSS(),
                            control=mb.boost_control(60, 0.1))
    with pytest.raises(NotImplementedError, match="once per cycle"):
        cyclic.varimp()

    noncyclic = mb.gamboostLSS(formulas, data=data, families=GaussianLSS(),
                               control=mb.boost_control(60, 0.1),
                               method="noncyclic")
    importance = noncyclic.varimp()
    # x drives mu in this data, so it must carry the larger share.
    assert importance["reduction"]["x.mu"] > importance["reduction"]["z.sigma"]
    assert sum(abs(v) for v in importance["reduction"].values()) == pytest.approx(1.0)
    assert sum(importance["selfreq"].values()) == pytest.approx(1.0)
