import json

import pytest

import mboostrs as mb


def _by_name(rows):
    return {row["name"]: row for row in rows}


def test_capabilities_are_json_serializable():
    payload = mb.capabilities(as_frame=False)

    assert set(payload) == {"families", "learners", "methods"}
    json.dumps(payload)


def test_validation_ledger_is_explicit_and_json_serializable():
    rows = mb.validation_capabilities(as_frame=False)

    assert len(rows) >= 40
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["node_id"] for row in rows)
    assert all(row["oracle"] for row in rows)
    assert all(row["compared_outputs"] for row in rows)
    assert all(float(row["atol"]) >= 0.0 and float(row["rtol"]) >= 0.0 for row in rows)
    json.dumps(rows)


def test_family_capabilities_cover_public_families_and_known_gaps():
    rows = mb.family_capabilities(as_frame=False)
    by_name = _by_name(rows)

    for name in [
        "Gaussian",
        "Binomial",
        "Poisson",
        "GammaReg",
        "CoxPH",
        "Multinomial",
        "AUC",
        "Gehan",
        "Cindex",
        "GaussianLSS",
        "GammaLSS",
        "BetaLSS",
        "ZINBLSS",
        "DirichletLSS",
        "WeibullLSS",
    ]:
        assert name in by_name
        assert by_name[name]["fit"] == "yes"

    assert by_name["GaussianLSS"]["cdf"] == "yes"
    assert by_name["GaussianLSS"]["deviance_increment"] == "yes"
    assert by_name["GammaLSS"]["pdf_or_pmf"] == "yes"
    assert by_name["BetaLSS"]["simulate"] == "yes"
    assert by_name["NBinomialLSS"]["deviance_increment"] == "yes"
    assert by_name["NBinomialLSS"]["cdf"] == "yes"
    assert by_name["NBinomialLSS"]["quantile"] == "yes"
    assert by_name["PoissonLSS"]["deviance_increment"] == "yes"
    assert by_name["PoissonLSS"]["cdf"] == "yes"
    assert by_name["PoissonLSS"]["quantile"] == "yes"
    # GammaLSS, StudentTLSS, BetaLSS, and ZINBLSS gained CDF/quantile once the
    # regularized incomplete gamma and beta functions were implemented in
    # gamlss.py (SciPy is not a runtime dependency). Validated against SciPy as a
    # test-time oracle -- see tests/test_lss_special_functions.py and the
    # py.lss.cdf_quantile.scipy_oracle validation record.
    for _lss in ("GammaLSS", "StudentTLSS", "BetaLSS", "ZINBLSS"):
        assert by_name[_lss]["cdf"] == "yes", _lss
        assert by_name[_lss]["quantile"] == "yes", _lss
        assert "py.lss.cdf_quantile.scipy_oracle" in by_name[_lss]["validation_ids"], _lss
    assert by_name["Gaussian"]["python_fit_status"] == "verified"
    assert by_name["GaussianLSS"]["r_fit_status"] == "verified"
    assert by_name["GumbelLSS"]["python_fit_status"] == "unverified"
    # `unsupported` until P4.1 item 1 replaced the Poisson approximation with
    # the exact density and gradients. `smoke` rather than `verified`: the
    # numerics are checked against gamlss.dist in Rust, and the Python
    # frontend's own evidence is a simulation recovery against R's maximum
    # likelihood, which is a tolerance rather than a parity claim.
    assert by_name["PIGLSS"]["python_fit_status"] == "smoke"
    assert by_name["PIGLSS"]["fit"] == "yes"
    assert "parity.python.gaussian" in by_name["Gaussian"]["validation_ids"]
    assert by_name["AUC"]["python_fit_status"] == "verified"
    assert "parity.python.auc" in by_name["AUC"]["validation_ids"]
    # Gehan's offset is unidentifiable (see families.rs Gehan doc comment),
    # so mboostrs deliberately deviates from R's arbitrary offset() value
    # rather than being "verified" against it -- fit is flagged
    # intentional-deviation, with a dedicated fit-parity id for everything
    # that IS offset-invariant (risk, selected, centered fitted) plus a
    # direct element-wise ngradient oracle.
    assert by_name["Gehan"]["python_fit_status"] == "intentional-deviation"
    assert "parity.python.gehan" in by_name["Gehan"]["validation_ids"]
    assert "parity.python.gehan_ngradient" in by_name["Gehan"]["validation_ids"]
    assert "deviation.gehan_offset" in by_name["Gehan"]["validation_ids"]
    # Unlike Gehan, Cindex's offset matches R exactly (0 both sides, verified
    # directly against R) -- fit is "verified", not "intentional-deviation".
    assert by_name["Cindex"]["python_fit_status"] == "verified"
    assert "parity.python.cindex" in by_name["Cindex"]["validation_ids"]
    assert "parity.python.cindex_ngradient" in by_name["Cindex"]["validation_ids"]

    for name in ["RCG", "FP", "Family"]:
        assert name in by_name
        assert by_name[name]["fit"] == "no"
        assert "not implemented" in by_name[name]["notes"]

    # IPCweights is not a boosting family (no fit/predict/coef), so it stays
    # fit="no" here, but it is implemented as the standalone
    # mboostrs.ipc_weights() preprocessing helper -- see the diagnostics
    # methods table and tests/test_diagnostics.py::test_ipc_weights_matches_r_mboost.
    assert by_name["IPCweights"]["fit"] == "no"
    assert "ipc_weights" in by_name["IPCweights"]["notes"]
    assert "not implemented" not in by_name["IPCweights"]["notes"]


def test_learner_capabilities_cover_public_learners_and_known_gaps():
    rows = mb.learner_capabilities(as_frame=False)
    by_name = _by_name(rows)

    for name in [
        "Bols",
        "Bbs",
        "Bmono",
        "Btree",
        "Bmrf",
        "Bspatial",
        "Brandom",
        "Bns",
        "Bkernel",
        "Bconstrained",
        "Buser",
    ]:
        assert name in by_name

    assert by_name["Bols"]["glmboost"] == "yes"
    assert by_name["Bbs"]["gamboostLSS"] == "yes"

    # The genuine gaps: two upstream learners mboostrs does not implement.
    # `brad` was listed here too, and should not have been -- mboostrs
    # implements it as `Brad` (Matern(3/2) against space-filling centres,
    # which is what upstream `brad` is: `fields::stationary.cov` with
    # smoothness 1.5). The ledger carried both rows, one verified and one
    # "not implemented", and the README's generated summary counted the
    # second.
    for name in ["blackboost", "bss"]:
        assert name in by_name
        assert by_name[name]["gamboost"] == "no"

    assert "brad" not in by_name
    assert by_name["Brad"]["r_name"] == "brad"
    assert by_name["Brad"]["gamboost"] == "yes"


def test_method_capabilities_track_lss_object_surface_and_gaps():
    rows = mb.method_capabilities(as_frame=False)
    lss_rows = {row["method"]: row for row in rows if row["model_class"] == "LSSModel"}

    assert lss_rows["predict"]["implemented"] == "yes"
    assert lss_rows["predict"]["partial"] == "yes"
    assert lss_rows["selected"]["implemented"] == "yes"
    assert lss_rows["fitted"]["implemented"] == "yes"
    assert lss_rows["risk"]["implemented"] == "yes"
    assert lss_rows["risk"]["partial"] == "yes"
    assert lss_rows["model_weights"]["implemented"] == "yes"
    assert lss_rows["summary"]["implemented"] == "yes"
    assert lss_rows["update"]["implemented"] == "yes"
    assert lss_rows["gaic"]["implemented"] == "yes"
    assert lss_rows["predint"]["implemented"] == "yes"
    assert lss_rows["prediction_interval"]["implemented"] == "yes"
    assert lss_rows["centiles"]["implemented"] == "yes"
    assert lss_rows["predict_all"]["implemented"] == "yes"
    assert lss_rows["coef_all"]["implemented"] == "yes"
    assert lss_rows["pit"]["implemented"] == "yes"
    assert lss_rows["residual_frame"]["implemented"] == "yes"
    assert lss_rows["q_stats"]["implemented"] == "yes"
    assert lss_rows["quantile_residuals"]["implemented"] == "yes"


def _load_manifest():
    from pathlib import Path

    return json.loads(
        (Path(__file__).resolve().parents[1] / "python/mboostrs/capability_manifest.json").read_text()
    )


def test_node_capabilities_has_one_record_per_cross():
    manifest = _load_manifest()
    nodes = manifest.get("node_capabilities", [])
    assert len(nodes) > 0

    from collections import Counter

    keys = Counter(
        (n["frontend"], n["family"] or n["learner"] or n["operation"],
         n["operation"], n.get("model_class", ""))
        for n in nodes
    )
    assert all(count == 1 for count in keys.values()), "duplicate node_capabilities entries"

    statuses = {n["status"] for n in nodes}
    assert statuses <= {"verified", "partial", "smoke", "intentional-deviation", "unsupported", "unverified"}

    validation_ids = {v["id"] for v in manifest.get("validations", [])}
    for n in nodes:
        bad = set(n["validation_ids"]) - validation_ids
        assert not bad, f"node {n['frontend']}/{n['family'] or n['learner'] or n['operation']} refs unknown: {bad}"
        assert n["weight_kind"] in ("unknown", "unit", "fractional", "integer-count", "bootstrap", "subsampling")


def test_node_capabilities_covers_all_families():
    manifest = _load_manifest()
    nodes = manifest.get("node_capabilities", [])
    node_families = {n["family"] for n in nodes if n["family"]}
    table_families = {f["name"] for f in manifest["tables"]["families"]}
    assert node_families == table_families


def test_node_capabilities_covers_all_learners():
    manifest = _load_manifest()
    nodes = manifest.get("node_capabilities", [])
    node_learners = {n["learner"] for n in nodes if n["learner"]}
    table_learners = {row["name"] for row in manifest["tables"]["learners"]}
    assert node_learners == table_learners


def test_capabilities_default_returns_frame_like_tables():
    tables = mb.capabilities()

    for table in tables.values():
        assert hasattr(table, "shape")
        assert table.shape[0] > 0


def test_capabilities_auto_prefers_polars_when_available():
    pl = pytest.importorskip("polars")

    table = mb.family_capabilities(output="auto")

    assert isinstance(table, pl.DataFrame)


def test_method_capabilities_can_expose_r_contract():
    rows = mb.method_capabilities(frontend="r", as_frame=False)

    assert any(row["model_class"] == "mboostrs_gamboostLSS" and row["method"] == "predict" for row in rows)
    assert {row["status"] for row in rows} <= {
        "verified",
        "smoke",
        "intentional-deviation",
        "unsupported",
        "unverified",
    }
