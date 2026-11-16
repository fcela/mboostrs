"""Tests for the capability manifest's reference-version check.

The manifest's `reference` field attests to the oracle version a validation
was checked against (e.g. "gamboostLSS 2.1-0"). Nothing used to compare that
claim to what is actually installed, so it could silently outlive its oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_capabilities import (  # noqa: E402
    ContractError,
    validate_reference_versions,
    validate_upstream_names,
)


def _manifest(*references: str) -> dict:
    return {
        "validations": [
            {"id": f"v{i}", "reference": reference} for i, reference in enumerate(references)
        ]
    }


def test_a_stale_reference_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "tools.generate_capabilities._installed_r_package_versions",
        lambda: {"gamboostLSS": "2.2.0"},
    )
    manifest = _manifest("gamboostLSS 2.1-0")
    with pytest.raises(ContractError, match="2.1-0 claimed, 2.2.0 installed"):
        validate_reference_versions(manifest)


def test_a_matching_reference_is_accepted(monkeypatch):
    monkeypatch.setattr(
        "tools.generate_capabilities._installed_r_package_versions",
        lambda: {"mboost": "2.9.11"},
    )
    # "2.9-11" (R's dash convention) and "2.9.11" (packageVersion's dot
    # convention) name the same release.
    validate_reference_versions(_manifest("mboost 2.9-11"))


def test_a_composite_reference_checks_both_packages(monkeypatch):
    monkeypatch.setattr(
        "tools.generate_capabilities._installed_r_package_versions",
        lambda: {"gamboostLSS": "2.2.0", "mboost": "2.9.11"},
    )
    with pytest.raises(ContractError, match="gamboostLSS 2.1-0 claimed"):
        validate_reference_versions(_manifest("gamboostLSS 2.1-0 / mboost 2.9-11"))


def test_prose_that_mentions_a_version_in_passing_is_not_an_attestation(monkeypatch):
    monkeypatch.setattr(
        "tools.generate_capabilities._installed_r_package_versions",
        lambda: {"gamboostLSS": "2.2.0"},
    )
    # This reference genuinely mentions gamboostLSS 2.1-0, as the version
    # where upstream removed bns -- a historical note, not a claim about what
    # the suite ran against. It must not be flagged as a stale attestation.
    validate_reference_versions(
        _manifest("mboostrs Bns contract; upstream gamboostLSS 2.1-0 removes bns")
    )


def test_missing_r_skips_the_check_instead_of_failing(monkeypatch):
    monkeypatch.setattr(
        "tools.generate_capabilities._installed_r_package_versions",
        lambda: {},
    )
    # No R/Rscript in this environment (e.g. a docs-only build) -- nothing
    # to compare against, so this must not raise.
    validate_reference_versions(_manifest("gamboostLSS 2.1-0"))


def test_the_checked_in_manifest_has_no_stale_reference():
    """End-to-end: the real manifest, checked against whatever R actually
    has installed right now. Skips if R isn't available in this environment."""
    from tools.generate_capabilities import _installed_r_package_versions, load_manifest

    if not _installed_r_package_versions():
        pytest.skip("R/Rscript not available in this environment")
    validate_reference_versions(load_manifest())


# ─── one row per upstream name ─────────────────────────────────────
#
# The duplicate-entity check is keyed on the *mboostrs* name, so two rows
# could claim the same `r_name` and disagree about it. Two did: `Brad`
# recorded a verified Matern(3/2) radial basis against upstream `brad` --
# which is what `brad` is upstream, `fields::stationary.cov` with smoothness
# 1.5 -- while a second row named `brad` recorded "R base-learner not
# implemented" for the same learner. The second was written before `Brad`
# existed and never removed.
#
# It was not inert. The README's capability summary is generated from these
# rows, so it told readers the ledger held 15 learner rows with three
# unsupported, when two upstream learners are genuinely absent and the third
# was implemented and verified.


def _learner(name: str, r_name: str, status: str) -> dict:
    return {"name": name, "r_name": r_name,
            "support": {"python": {"operations": {"fit": status}}}}


def test_two_rows_for_one_upstream_name_are_rejected():
    manifest = {"tables": {"families": [], "learners": [
        _learner("Brad", "brad", "verified"),
        _learner("brad", "brad", "unsupported"),
    ]}}
    with pytest.raises(ContractError, match="both claim upstream 'brad'"):
        validate_upstream_names(manifest)


def test_the_message_says_which_two_rows_collide():
    manifest = {"tables": {"families": [
        _learner("Weibull", "Weibull", "verified"),
        _learner("WeibullOld", "Weibull", "unsupported"),
    ], "learners": []}}
    with pytest.raises(ContractError, match="'Weibull' and 'WeibullOld'"):
        validate_upstream_names(manifest)


def test_a_row_for_something_upstream_only_is_still_allowed():
    """The `unsupported` status exists to record what upstream has and
    mboostrs does not; `bss` and `blackboost` are real absences and keep
    their rows. What cannot stand is two rows for one upstream name."""
    manifest = {"tables": {"families": [], "learners": [
        _learner("Bbs", "bbs", "verified"),
        _learner("bss", "bss", "unsupported"),
        _learner("blackboost", "blackboost", "unsupported"),
    ]}}
    validate_upstream_names(manifest)


def test_a_row_with_no_upstream_counterpart_is_skipped():
    """An mboostrs-only learner has no `r_name`, and several sharing that
    emptiness is not a collision."""
    manifest = {"tables": {"families": [], "learners": [
        {"name": "OwnA", "support": {}},
        {"name": "OwnB", "r_name": "", "support": {}},
    ]}}
    validate_upstream_names(manifest)


def test_the_checked_in_manifest_has_one_row_per_upstream_name():
    import json

    manifest = json.loads(
        (ROOT / "python" / "mboostrs" / "capability_manifest.json").read_text()
    )
    validate_upstream_names(manifest)


def test_brad_is_recorded_as_implemented():
    """The specific row that was wrong, pinned: mboostrs implements upstream
    `brad` as `Brad`, so the ledger must not also claim it is absent."""
    import json

    manifest = json.loads(
        (ROOT / "python" / "mboostrs" / "capability_manifest.json").read_text()
    )
    rows = {row["name"]: row for row in manifest["tables"]["learners"]}
    assert "brad" not in rows, "the stale upstream-only row is back"
    assert rows["Brad"]["r_name"] == "brad"
    # Implemented and exercised, but not compared with mboost: mboostrs picks
    # centres by farthest-point selection where mboost uses
    # fields::cover.design, so the fits differ and `verified` would overstate.
    assert rows["Brad"]["support"]["python"]["operations"]["fit"] == "smoke"
