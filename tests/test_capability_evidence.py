"""The capability ledger's evidence links resolve, and `verified` means it.

P1.1 asks for a ledger whose rows name the exact test node behind them, and
for stale review dates and tests that no longer collect to fail CI. The links
existed and nothing checked them, so a renamed test or a deleted
parametrisation would leave the tables — and the review packet built from
them — asserting evidence that no longer runs.

P1.1 also records that the ledger "overstates at least one R prediction row".
It did. `GaussianKnownVarLSS` claimed a verified R fit and a verified R
prediction, and the script behind both compares `predict(type="link")` against
three numbers copied from a Python run. That check is worth having — it shows
the two frontends assemble the same call — and it is not a comparison against
an outside reference, which is what `verified` means in this vocabulary. The
Python side of the same family said `smoke`, and an R frontend cannot be
better verified than the implementation it is checked against.
"""

import datetime
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools" / "check_capability_evidence.py"


@pytest.fixture(scope="module")
def evidence():
    sys.path.insert(0, str(ROOT))
    from tools import check_capability_evidence
    return check_capability_evidence


@pytest.fixture(scope="module")
def document(evidence):
    return evidence.manifest()


def test_the_evidence_check_passes():
    result = subprocess.run(
        [sys.executable, str(CHECK)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_every_validation_names_a_node(document):
    for record in document["validations"]:
        assert record["node_id"], record["id"]
        assert record["oracle"], record["id"]


def test_every_r_script_carries_its_marker(evidence, document):
    """The link is bidirectional: the ledger names the script, and the script
    names the validation. One direction alone rots silently."""
    markers = evidence.r_markers()
    for record in document["validations"]:
        # Python nodes are pytest ids; Rust nodes are `<file>::<module>` and
        # carry their claim as the module itself.
        if record["node_id"].startswith(("tests/", "crates/")):
            continue
        assert record["node_id"] in markers.get(record["id"], []), record["id"]


def test_rust_evidence_names_a_module_that_exists(evidence, document):
    """The core crate is published on its own, so a contract it enforces can
    only be checked where the contract lives: a Python test cannot show that
    `QuantReg::try_new` rejects a `tau`, because the Python constructor
    rejects it first."""
    rust = [r for r in document["validations"]
            if r["node_id"].startswith("crates/")]
    assert rust, "the ledger should carry Rust evidence"
    assert evidence.check_rust_nodes(rust, show=False) == []


def test_a_missing_rust_module_fails_the_check(evidence):
    invented = [{"id": "invented",
                 "node_id": "crates/mboostrs-core/src/families.rs::no_such_mod",
                 "oracle": "mboost"}]
    problems = evidence.check_rust_nodes(invented, show=False)
    assert any("has no `mod no_such_mod`" in problem for problem in problems)


def test_rust_evidence_must_name_a_module(evidence):
    invented = [{"id": "invented",
                 "node_id": "crates/mboostrs-core/src/families.rs",
                 "oracle": "mboost"}]
    problems = evidence.check_rust_nodes(invented, show=False)
    assert any("which test module" in problem for problem in problems)


def test_a_renamed_test_fails_the_check(evidence, monkeypatch):
    records = [{"id": "invented", "node_id": "tests/test_nothing.py::test_x",
                "oracle": "mboost"}]
    problems = evidence.check_python_nodes(records, show=False)
    assert any("does not collect" in problem for problem in problems)


def test_an_orphan_marker_fails_the_check(evidence, monkeypatch):
    monkeypatch.setattr(evidence, "r_markers",
                        lambda: {"not.recorded": ["mboostrs/tests/x.R"]})
    problems = evidence.check_orphan_markers([], show=False)
    assert any("not a validation the manifest records" in problem
               for problem in problems)


def test_a_review_older_than_the_last_edit_fails(evidence, monkeypatch):
    monkeypatch.setattr(evidence, "manifest_commit_date",
                        lambda: datetime.date(2026, 9, 5))
    problems = evidence.check_review_date({"last_reviewed": "2026-07-19"},
                                          show=False)
    assert any("Re-review it" in problem for problem in problems)


def test_no_verified_status_rests_on_the_other_frontend(evidence, document):
    assert evidence.check_verified_has_an_outside_oracle(
        document, show=False) == []


def test_cross_frontend_agreement_is_not_verified(evidence):
    """The rule that caught the overstated row, shown failing."""
    document = {
        "tables": {"families": [{
            "name": "Invented",
            "support": {"r": {"operations": {"fit": "verified"},
                              "validation_ids": ["only"]}},
        }]},
        "validations": [{"id": "only", "node_id": "mboostrs/tests/x.R",
                         "oracle": "the mboostrs Python frontend"}],
    }
    problems = evidence.check_verified_has_an_outside_oracle(document,
                                                             show=False)
    assert any("Cross-frontend agreement is `smoke`" in problem
               for problem in problems)


def test_the_known_var_family_reports_what_its_evidence_supports():
    """The row P1.1 named. Its R fit and prediction were `verified` against
    three numbers copied out of a Python run."""
    import mboostrs as mb

    row = next(r for r in mb.family_capabilities(as_frame=False)
               if r["name"] == "GaussianKnownVarLSS")
    assert row["r_fit_status"] == "smoke"
    assert row["r_predict_status"] == "smoke"
    assert row["python_fit_status"] == "smoke"


def test_verified_r_predictions_have_an_external_r_parity_node(document):
    """A verified R prediction must be compared with an external oracle."""
    validations = {record["id"]: record for record in document["validations"]}
    for row in document["tables"]["families"]:
        support = row["support"]["r"]
        if support["operations"]["predict"] != "verified":
            continue
        evidence = [validations[item] for item in support["validation_ids"]
                    if item.startswith("parity.r.")]
        assert evidence, row["name"]
        assert any("gamboostLSS" in item["oracle"] for item in evidence), row["name"]


def test_the_oracle_says_what_it_compared(document):
    record = next(v for v in document["validations"]
                  if v["id"] == "parity.r.gaussianknownvarlss")
    assert "Python frontend" in record["oracle"]
    assert "nothing upstream" in record["oracle"]


def test_the_review_date_is_a_date(document):
    datetime.date.fromisoformat(document["last_reviewed"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", document["last_reviewed"])


def test_the_generated_tables_match_the_manifest():
    """The R capability file is generated from the manifest; a hand edit to
    one of them would make the two frontends report different statuses."""
    result = subprocess.run(
        [sys.executable, "tools/generate_capabilities.py", "--check"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_manifest_is_valid_json():
    json.loads((ROOT / "python" / "mboostrs"
                / "capability_manifest.json").read_text())
