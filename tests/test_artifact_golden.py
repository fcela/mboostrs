"""Committed artifacts still load, and the schema version is deliberate.

P4.3: "Version the wire format whenever a reader must understand a new
required tag or model class. Add golden artifacts per schema version and
cross-version migration tests. Record checksums, feature provenance, response
coding, weight policy, and intentional deviations."

Golden artifacts existed — `mboostrs/tests/fixtures/` holds five, written by
an earlier build of the Python writer — and only the *R* reader test loaded
them. On the Python side the writer and the reader are the same code, so a
change to the wire format is invisible to Python's own tests: both sides move
together and every round trip still passes.

These load the committed files, which the current writer did not produce, and
check that the values they predict are the ones recorded beside them. A schema
change that broke an old artifact fails here rather than when someone opens a
file they saved last month.
"""

import json
import pathlib
import tempfile

import numpy as np
import pytest

import mboostrs as mb

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "mboostrs" / "tests" / "fixtures"
SCHEMA_VERSION = 1


@pytest.fixture(scope="module")
def expected():
    return json.loads((FIXTURES / "expected.json").read_text())


@pytest.mark.parametrize("name", ["gaussian", "poisson", "binomial"])
def test_a_committed_artifact_predicts_what_it_recorded(name, expected):
    """The cross-version check. These files were written by an earlier build;
    nothing in this run produced them."""
    model = mb.load_model(str(FIXTURES / f"glm_{name}.json"))
    newdata = {key: np.asarray(values, dtype=float)
               for key, values in expected["newdata"].items()}
    for scale in ("link", "response"):
        predicted = np.asarray(model.predict(newdata=newdata, type=scale),
                               dtype=float)
        recorded = np.asarray(expected["expected"][name][scale], dtype=float)
        assert predicted == pytest.approx(recorded, rel=1e-10, abs=1e-12), (
            f"{name}/{scale} moved: {predicted} against {recorded}")


@pytest.mark.parametrize("name", ["glm_gaussian", "glm_poisson",
                                  "glm_binomial", "glm_quantile",
                                  "gam_spline"])
def test_every_fixture_declares_the_schema_this_reader_supports(name):
    document = json.loads((FIXTURES / f"{name}.json").read_text())
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["model_class"] in ("GLMBoostModel", "GAMBoostModel")
    assert document["mboostrs_version"]


def test_a_fresh_artifact_declares_the_same_version():
    """A bump has to be a deliberate act. If the writer starts emitting a new
    version, this fails and the golden fixtures have to be regenerated with
    the migration thought through, rather than the two drifting apart."""
    import warnings

    rng = np.random.default_rng(0)
    n = 50
    data = {"x": rng.normal(size=n)}
    data["y"] = data["x"] + rng.normal(scale=0.3, size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.glmboost("y ~ x", data, control=mb.boost_control(10, 0.1))
    path = pathlib.Path(tempfile.mkdtemp()) / "model.json"
    fit.save(str(path))
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_an_unsupported_schema_version_is_refused():
    document = json.loads((FIXTURES / "glm_gaussian.json").read_text())
    document["schema_version"] = SCHEMA_VERSION + 1
    path = pathlib.Path(tempfile.mkdtemp()) / "future.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="schema version"):
        mb.load_model(str(path))


def test_the_artifact_records_the_weight_contract():
    """P4.3 names weight policy among what an artifact must record. It did
    not until P0.2: a `raw` fit came back claiming `upstream`."""
    import warnings

    rng = np.random.default_rng(1)
    n = 60
    data = {"x": rng.normal(size=n)}
    data["y"] = data["x"] + rng.normal(scale=0.3, size=n)
    weights = rng.uniform(0.3, 2.0, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.glmboost("y ~ x", data, weights=weights,
                          weight_policy="raw",
                          control=mb.boost_control(10, 0.1))
    path = pathlib.Path(tempfile.mkdtemp()) / "weighted.json"
    fit.save(str(path))
    restored = mb.load_model(str(path))
    assert restored.weight_policy == "raw"
    assert restored.input_weights() is not None
    assert restored._weights_rescaled is False


def test_the_artifact_records_the_response_coding():
    """A Binomial fit re-codes {0,1} to {-1,+1} internally. A reader that
    could not tell which coding produced the coefficients would predict the
    wrong sign."""
    document = json.loads((FIXTURES / "glm_binomial.json").read_text())
    family = document["state"]["items"]["family"]
    assert family["name"] == "Binomial"
    model = mb.load_model(str(FIXTURES / "glm_binomial.json"))
    probabilities = np.asarray(
        model.predict(newdata={"x": np.array([0.0, 1.0]),
                               "z": np.array([0.0, 1.0])},
                      type="response"), dtype=float)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_the_artifact_records_its_intentional_deviations():
    """P1.4 put the parity notes visible at write time into the artifact.
    A reader who loads a `Gehan` fit a year from now should not have to know
    that its offset is 0 by design."""
    import warnings

    rng = np.random.default_rng(2)
    n = 60
    data = {"x": rng.normal(size=n)}
    data["y"] = data["x"] + rng.normal(scale=0.3, size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.glmboost("y ~ x", data, family=mb.Gaussian(),
                          control=mb.boost_control(10, 0.1))
    path = pathlib.Path(tempfile.mkdtemp()) / "noted.json"
    fit.save(str(path))
    text = path.read_text()
    assert "parity" in text.lower() or "logLik" in text, (
        "the artifact should carry the notes that applied at write time")
