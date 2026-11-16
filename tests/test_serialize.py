"""Portable model artifacts.

The contract is that a reloaded model predicts identically to the original.
Everything else here guards the format: schema version, explicit rejection of
things it cannot represent, and human-readability.
"""

import json

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import serialize
from mboostrs.serialize import _content_digest

CTRL = mb.boost_control(100, 0.1)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = {
        "x": rng.uniform(1.0, 10.0, n),
        "z": rng.normal(size=n),
        "g": np.array(["a", "b", "c"])[rng.integers(0, 3, n)],
    }
    d["y"] = 2.0 * np.log(d["x"]) + 0.5 * d["z"] + rng.normal(scale=0.2, size=n)
    return d


def _newdata():
    return {"x": np.array([2.0, 5.0]), "z": np.zeros(2), "g": np.array(["a", "c"])}


def _models():
    d = _data()
    rng = np.random.default_rng(5)
    counts = {**d, "y": rng.poisson(np.exp(0.3 * d["z"])).astype(float)}
    return {
        "glm": mb.glmboost("y ~ x + z", data=d, family=mb.Gaussian(), control=CTRL),
        "glm_log": mb.glmboost(
            "y ~ log(x) + z", data=d, family=mb.Gaussian(), control=CTRL
        ),
        "glm_scale": mb.glmboost(
            "y ~ scale(x) + z", data=d, family=mb.Gaussian(), control=CTRL
        ),
        "gam_bbs": mb.gamboost(
            "y ~ bbs(x, df=4) + bols(z)", data=d, family=mb.Gaussian(), control=CTRL
        ),
        "gam_factor": mb.gamboost(
            "y ~ brandom(g) + bols(z)", data=d, family=mb.Gaussian(), control=CTRL
        ),
        # Bspatial is the only learner whose constructor takes two positional
        # names (name1, name2); the decoder used to guess a single `name` or
        # `names` key and crash on this one.
        "gam_bspatial": mb.gamboost(
            "y ~ bspatial(x, z, df=6)", data=d, family=mb.Gaussian(), control=CTRL
        ),
        "poisson": mb.glmboost(
            "y ~ x + z", data=counts, family=mb.Poisson(), control=CTRL
        ),
        "quantile": mb.glmboost(
            "y ~ x + z", data=d, family=mb.Quantile(tau=0.9), control=CTRL
        ),
    }


ALL_MODELS = _models()


@pytest.mark.parametrize("key", sorted(ALL_MODELS))
def test_round_trip_predicts_identically(key, tmp_path):
    model = ALL_MODELS[key]
    path = mb.save_model(model, tmp_path / f"{key}.json")
    restored = mb.load_model(path)

    before = np.asarray(model.predict(newdata=_newdata(), type="response"))
    after = np.asarray(restored.predict(newdata=_newdata(), type="response"))
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("key", sorted(ALL_MODELS))
def test_round_trip_preserves_structure(key, tmp_path):
    model = ALL_MODELS[key]
    restored = mb.load_model(mb.save_model(model, tmp_path / f"{key}.json"))

    assert restored.mstop == model.mstop
    assert restored.base_learner_names() == model.base_learner_names()
    np.testing.assert_array_equal(restored.risk_, model.risk_)
    np.testing.assert_array_equal(restored.fitted_, model.fitted_)
    assert restored.offset_ == model.offset_


def test_transform_recipe_survives_the_round_trip(tmp_path):
    """A reloaded model must replay `scale` with the original parameters."""
    model = ALL_MODELS["glm_scale"]
    restored = mb.load_model(mb.save_model(model, tmp_path / "m.json"))
    assert restored._transform_recipe.describe() == model._transform_recipe.describe()

    # The parameters are what makes predictions batch-independent.
    embedded = {
        "x": np.array([2.0, 5.0, 900.0]),
        "z": np.zeros(3),
        "g": np.array(["a", "c", "b"]),
    }
    np.testing.assert_array_equal(
        np.asarray(restored.predict(newdata=_newdata(), type="response")),
        np.asarray(restored.predict(newdata=embedded, type="response"))[:2],
    )


def test_factor_levels_survive_and_still_reject_unseen_levels(tmp_path):
    model = ALL_MODELS["gam_factor"]
    restored = mb.load_model(mb.save_model(model, tmp_path / "m.json"))
    with pytest.raises(ValueError, match=r"not present"):
        restored.predict(
            newdata={"g": np.array(["a", "zzz"]), "z": np.zeros(2)}
        )


def test_sliced_model_round_trips(tmp_path):
    model = ALL_MODELS["gam_bbs"][:40]
    restored = mb.load_model(mb.save_model(model, tmp_path / "m.json"))
    assert restored.mstop == 40
    np.testing.assert_array_equal(
        np.asarray(model.predict(newdata=_newdata(), type="response")),
        np.asarray(restored.predict(newdata=_newdata(), type="response")),
    )


def test_artifact_is_readable_json_with_a_schema_version(tmp_path):
    path = mb.save_model(ALL_MODELS["glm"], tmp_path / "m.json", indent=2)
    document = json.loads(path.read_text())

    assert document["schema_version"] == serialize.SCHEMA_VERSION
    assert document["model_class"] == "GLMBoostModel"
    assert document["mboostrs_version"] == mb.__version__
    # The formula is plainly visible, not buried in a binary blob.
    assert document["state"]["items"]["formula"] == "y ~ x + z"


def test_unsupported_schema_version_is_rejected(tmp_path):
    path = tmp_path / "m.json"
    mb.save_model(ALL_MODELS["glm"], path)
    document = json.loads(path.read_text())
    document["schema_version"] = 999
    # A writer that produced this content would sign it; restamp the digest
    # so the check under test is the one that answers.
    document["content_sha256"] = _content_digest(document)
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="schema version"):
        mb.load_model(path)


def test_unknown_model_class_is_rejected(tmp_path):
    path = tmp_path / "m.json"
    mb.save_model(ALL_MODELS["glm"], path)
    document = json.loads(path.read_text())
    document["model_class"] = "SomethingElse"
    # A writer that produced this content would sign it; restamp the digest
    # so the check under test is the one that answers.
    document["content_sha256"] = _content_digest(document)
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="unknown model class"):
        mb.load_model(path)


def test_an_edited_artifact_fails_its_checksum(tmp_path):
    path = tmp_path / "m.json"
    mb.save_model(ALL_MODELS["glm"], path)
    document = json.loads(path.read_text())
    document["state"]["items"]["offset"] = 123.0
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not match its checksum"):
        mb.load_model(path)


def test_the_checksum_ignores_indentation(tmp_path):
    compact, indented = tmp_path / "a.json", tmp_path / "b.json"
    mb.save_model(ALL_MODELS["glm"], compact)
    mb.save_model(ALL_MODELS["glm"], indented, indent=2)
    a = json.loads(compact.read_text())["content_sha256"]
    assert a == json.loads(indented.read_text())["content_sha256"]
    mb.load_model(indented)


def test_an_artifact_without_a_checksum_still_loads(tmp_path):
    path = tmp_path / "m.json"
    mb.save_model(ALL_MODELS["glm"], path)
    document = json.loads(path.read_text())
    del document["content_sha256"]
    path.write_text(json.dumps(document))
    loaded = mb.load_model(path)
    np.testing.assert_array_equal(loaded.predict(), ALL_MODELS["glm"].predict())


def test_a_failed_save_leaves_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "m.json"
    mb.save_model(ALL_MODELS["glm"], path)
    before = path.read_text()

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("os.fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        mb.save_model(ALL_MODELS["glm"], path)
    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_a_field_this_release_does_not_know_is_refused(tmp_path):
    path = tmp_path / "m.json"
    mb.save_model(ALL_MODELS["glm"], path)
    document = json.loads(path.read_text())
    family = document["state"]["items"]["family"]
    family["params"]["items"]["link_power"] = 2.0
    document["content_sha256"] = _content_digest(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="'link_power' this release"):
        mb.load_model(path)


def test_saving_an_unsupported_object_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="save_model supports"):
        mb.save_model("not a model", tmp_path / "m.json")


def test_family_parameters_survive(tmp_path):
    """Quantile carries tau; losing it would silently change the model."""
    model = ALL_MODELS["quantile"]
    restored = mb.load_model(mb.save_model(model, tmp_path / "m.json"))
    assert type(restored._family).__name__ == "Quantile"
    assert vars(restored._family) == vars(model._family)


def test_artifact_does_not_depend_on_pickle(tmp_path):
    """The whole point: no Python pickle opcodes in the file."""
    path = mb.save_model(ALL_MODELS["gam_bbs"], tmp_path / "m.json")
    text = path.read_text()
    json.loads(text)  # parses as plain JSON
    assert "__reduce__" not in text
    assert "copy_reg" not in text


def test_artifact_exposes_the_fields_the_r_reader_depends_on(tmp_path):
    """The R reader (mboostrs/R/mboostrs_load.R) reconstructs linear prediction
    from these state fields. Locking them here prevents a Python-side rename
    from silently breaking cross-language scoring."""
    model = ALL_MODELS["glm"]
    path = mb.save_model(model, tmp_path / "m.json", indent=1)
    doc = json.loads(path.read_text())

    assert doc["model_class"] == "GLMBoostModel"
    state = doc["state"]["items"]
    for key in ("offset", "nu", "learner_names", "predictor_names",
                "centering_means", "selected", "coefficients", "family"):
        assert key in state, f"artifact lost field {key!r} the R reader needs"
    assert state["family"]["__type__"] == "family"
    assert state["family"]["name"] == "Gaussian"


# ── 3.1: distributional models in the portable artifact format ───────────────
def _lss_cases():
    """One fitted model per interesting LSS shape, with its data."""
    from mboostrs import gamlss

    rng = np.random.default_rng(7)
    n = 250
    x, z = rng.normal(size=n), rng.normal(size=n)
    positive = np.abs(rng.normal(2, 0.5, n)) + 0.5
    counts = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
    unit = rng.beta(2, 5, n)

    cases = {}
    for label, family, y in (
        ("gaussian", gamlss.GaussianLSS(), 1 + 2 * x + rng.normal(0, 1, n)),
        ("studentt", gamlss.StudentTLSS(), 1 + 2 * x + rng.standard_t(5, n)),
        ("gamma", gamlss.GammaLSS(), positive),
        ("beta", gamlss.BetaLSS(), unit),
        ("poisson", gamlss.PoissonLSS(), counts),
        ("nbinomial", gamlss.NBinomialLSS(), counts),
        ("zipo", gamlss.ZIPoLSS(), counts),
    ):
        data = {"x": x, "z": z, "y": y}
        cases[f"gam_{label}"] = mb.gamboostLSS(
            {p: "y ~ bbs(x, df=4) + bols(z)" for p in family.param_names},
            data=data, families=family, control=mb.boost_control(20, 0.1))
    data = {"x": x, "z": z, "y": 1 + 2 * x + rng.normal(0, 1, n)}
    cases["glm_gaussian"] = mb.glmboostLSS(
        {"mu": "y ~ x + z", "sigma": "y ~ z"}, data=data,
        families=gamlss.GaussianLSS(), control=mb.boost_control(20, 0.1))
    cases["gam_noncyclic"] = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(z)"}, data=data,
        families=gamlss.GaussianLSS(), control=mb.boost_control(20, 0.1),
        method="noncyclic")
    return cases


@pytest.mark.parametrize("label", sorted(_lss_cases()))
def test_lss_models_round_trip_through_the_artifact_format(label, tmp_path):
    """Enhancement-hunt 3.1. Distributional models are the package's most
    distinctive capability and were the one model class that could not be
    archived beside a paper, moved to a production service, or reopened after a
    dependency bump."""
    fit = _lss_cases()[label]
    path = tmp_path / f"{label}.json"
    fit.save(path)

    reloaded = mb.load_model(path)
    assert type(reloaded).__name__ == "LSSModel"

    original, restored = fit.predict(type="response"), reloaded.predict(type="response")
    assert set(original) == set(restored)
    for parameter in original:
        np.testing.assert_array_equal(
            np.asarray(original[parameter]), np.asarray(restored[parameter])
        )
    np.testing.assert_array_equal(np.asarray(fit.risk), np.asarray(reloaded.risk))


def test_lss_artifact_scores_new_data_identically(tmp_path):
    fit = _lss_cases()["gam_gaussian"]
    path = fit.save(tmp_path / "m.json")
    reloaded = mb.load_model(path)
    newdata = {"x": np.array([-1.0, 0.0, 1.0]), "z": np.array([0.5, -0.5, 0.0])}
    original = fit.predict(newdata=newdata, type="response")
    restored = reloaded.predict(newdata=newdata, type="response")
    for parameter in original:
        np.testing.assert_array_equal(
            np.asarray(original[parameter]), np.asarray(restored[parameter])
        )
    np.testing.assert_array_equal(
        np.asarray(fit.centiles()), np.asarray(reloaded.centiles())
    )


def test_a_family_with_constructor_state_survives(tmp_path):
    """`DirichletLSS(K)` is the one LSS family that carries constructor state,
    and its `vars()` also holds `n_params` and `param_names`, which the
    constructor derives rather than accepts — passing those back raises
    `TypeError`, so the decoder filters to the signature."""
    from mboostrs.gamlss import DirichletLSS

    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    y = rng.dirichlet([2, 3, 4], n)
    data = {"x": x, "y1": y[:, 0], "y2": y[:, 1], "y3": y[:, 2]}
    family = DirichletLSS(3)
    fit = mb.glmboostLSS(
        {p: "cbind(y1, y2, y3) ~ x" for p in family.param_names},
        data=data, families=family, control=mb.boost_control(20, 0.1))

    reloaded = mb.load_model(fit.save(tmp_path / "dirichlet.json"))
    assert reloaded._family.K == 3
    assert list(reloaded.param_names) == ["a1", "a2", "a3"]
    original, restored = fit.predict(type="response"), reloaded.predict(type="response")
    for parameter in original:
        np.testing.assert_array_equal(
            np.asarray(original[parameter]), np.asarray(restored[parameter])
        )


def test_an_lss_artifact_names_its_class_so_other_readers_can_refuse(tmp_path):
    """The Rust and R readers score linear and additive models only. They must
    refuse a distributional artifact by *name* — the schema version is
    unchanged, so `model_class` is what tells them."""
    fit = _lss_cases()["glm_gaussian"]
    document = json.loads(fit.save(tmp_path / "m.json").read_text())
    assert document["model_class"] == "LSSModel"
    assert document["schema_version"] == serialize.SCHEMA_VERSION


class TestLSSFrameRoundTrip:
    """LSSModel.__getstate__ serialises its whole __dict__, which carries the
    training frame it needs on load to rebuild the learner stores. The encoder
    had no case for a data frame, so saving an LSS model fitted from polars or
    pandas raised TypeError while GLM and GAM models — which whitelist plain
    arrays — saved fine from the same input."""

    @pytest.mark.parametrize("backend", ["polars", "pandas", "records"])
    def test_lss_model_saves_from_every_input_backend(self, tmp_path, backend):
        import numpy as np

        import mboostrs as mb

        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(-2, 2, n)
        y = 1.0 + 0.5 * x + rng.normal(scale=np.exp(0.2 * x), size=n)
        records = {"y": y, "x": x}
        if backend == "polars":
            import polars as pl

            data = pl.DataFrame(records)
        elif backend == "pandas":
            pd = pytest.importorskip("pandas")
            data = pd.DataFrame(records)
        else:
            data = records

        model = mb.gamboostLSS("y ~ bols(x)", data=data,
                               families=mb.GaussianLSS(),
                               control=mb.boost_control(40, 0.1))
        path = tmp_path / "lss.pkl"
        mb.save_model(model, path)
        restored = mb.load_model(path)

        before = np.asarray(model.predict(parameter="mu"))
        after = np.asarray(restored.predict(parameter="mu"))
        np.testing.assert_array_equal(before, after)

        # Prediction on new data uses the rebuilt learner stores.
        newdata = {"y": y[:25], "x": x[:25]}
        np.testing.assert_array_equal(
            np.asarray(model.predict(newdata=newdata, parameter="mu")),
            np.asarray(restored.predict(newdata=newdata, parameter="mu")),
        )


def _stored_arrays(value, prefix=""):
    """Every array-like in a decoded artifact state, with its path."""
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _stored_arrays(item, f"{prefix}.{key}")
    elif hasattr(value, "columns") and hasattr(value, "shape"):
        for column in value.columns:
            yield f"{prefix}.{column}", np.asarray(value[column])
    elif isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, float)) for v in value):
        yield prefix, np.asarray(value)


@pytest.mark.parametrize("kind", ["glm", "gam", "gam log response", "lss"])
def test_a_scoring_only_artifact_stores_no_response_column(tmp_path, kind):
    rng = np.random.default_rng(4)
    n = 83
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = np.exp(data["x"] + rng.normal(size=n))
    if kind == "glm":
        model = mb.glmboost("y ~ x + z", data)
    elif kind == "gam":
        model = mb.gamboost("y ~ bbs(x) + bols(z)", data)
    elif kind == "gam log response":
        model = mb.gamboost("log(y) ~ bbs(x) + bols(z)", data)
    else:
        model = mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bols(z)"},
                               data=data, family=mb.GaussianLSS())
    path = tmp_path / "m.json"
    mb.save_model(model, path, scoring_only=True)
    state = serialize._decode(json.loads(path.read_text())["state"])
    for target in (data["y"], np.log(data["y"])):
        leaks = [name for name, array in _stored_arrays(state)
                 if array.shape == target.shape and np.allclose(array, target)]
        assert not leaks, f"response stored at {leaks}"
    loaded = mb.load_model(path)
    query = {"x": np.array([-0.5, 0.5]), "z": np.array([0.0, 1.0])}
    if kind == "lss":
        expected, got = model.predict(newdata=query), loaded.predict(newdata=query)
        for parameter in expected:
            np.testing.assert_allclose(got[parameter], expected[parameter])
    else:
        np.testing.assert_allclose(loaded.predict(newdata=query), model.predict(newdata=query))
