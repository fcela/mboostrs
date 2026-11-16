"""Check README support counts, evidence links and installation instructions.

Capability counts come from the generated summary. Unsupported paths remain
named, and performance claims remain scoped to the recorded measurements.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MANIFEST = ROOT / "python" / "mboostrs" / "capability_manifest.json"


@pytest.fixture(scope="module")
def readme():
    return README.read_text()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def _learners(manifest):
    return manifest["tables"]["learners"]


def _implemented(rows):
    return [r["name"] for r in rows
            if r["support"]["python"]["operations"]["fit"] != "unsupported"]


def _absent(rows):
    return [r["name"] for r in rows
            if r["support"]["python"]["operations"]["fit"] == "unsupported"]


# ─── the counts in the prose are the manifest's ────────────────────


def test_the_learner_count_matches_the_contract(readme, manifest):
    count = len(_learners(manifest))
    assert f"**{count} learner rows**" in readme


def test_the_family_row_counts_match_the_contract(readme, manifest):
    families = manifest["tables"]["families"]
    assert f"**{len(families)} family/helper rows**" in readme
    verified = sum(r["support"]["python"]["operations"]["fit"] == "verified" for r in families)
    unsupported = sum(r["support"]["python"]["operations"]["fit"] == "unsupported" for r in families)
    assert f"{verified} Python rows" in readme
    assert f"{unsupported} Python rows are explicitly unsupported" in readme


def test_the_absent_learners_are_named(readme, manifest):
    for name in _absent(_learners(manifest)):
        assert f"`{name}`" in readme, f"{name} is absent but the README does not say so"


def test_the_absent_families_are_named(readme, manifest):
    """`IPCweights` is deliberately excluded: it is not a boosting family, and
    the contract's own note says it is implemented as the standalone
    `mboostrs.ipc_weights` helper. Naming it as missing would be as wrong as
    the claim this test exists to replace."""
    missing = [r["name"] for r in manifest["tables"]["families"]
               if r["kind"] == "standard_missing"]
    assert set(missing) == {"FP", "Family", "IPCweights", "RCG"}
    for name in ("FP", "RCG"):
        assert f"`{name}`" in readme
    import mboostrs as mb
    assert hasattr(mb, "ipc_weights")


# ─── the withdrawn claims stay withdrawn ───────────────────────────


def test_the_unbounded_support_claim_is_gone(readme):
    assert "all base learners and families" not in readme
    assert "same formula interface and base learners" not in readme


def test_no_unbounded_speed_headline(readme):
    """The Performance section withdraws the measurement behind it, so the
    opening must not assert it."""
    opening = readme.split("## Install", 1)[0]
    for phrase in ("several times faster", "times faster", "x faster"):
        assert phrase not in opening, phrase
    # Wrapped across a line in the source, so compare on collapsed whitespace.
    flat = " ".join(readme.split())
    assert "no longer presented as current evidence" in flat


def test_the_reader_is_pointed_at_the_evidence(readme):
    opening = readme.split("## Install", 1)[0]
    assert "capabilities.ipynb" in opening
    assert "release-limitations.md" in opening
    assert "## Performance" in readme


# ─── the install section states what a bare install gives ──────────


def test_the_plot_extra_is_named_where_a_user_installs(readme):
    """`pip install mboostrs` brings no Altair, so the first `plot_risk()`
    call raises. The install section says so before that happens."""
    install = readme.split("## Install", 1)[1].split("## ", 1)[0]
    assert 'pip install "mboostrs[plot]"' in install
    assert "ImportError" in install


def test_the_extras_named_in_the_readme_exist(readme):
    """A named extra that `pyproject.toml` does not declare is an install
    command that fails."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    declared = set(re.findall(r"^(\w[\w-]*) = \[", pyproject, re.M))
    for extra in re.findall(r'pip install "mboostrs\[([\w,-]+)\]"', readme):
        for one in extra.split(","):
            assert one in declared, f"README installs [{one}], not in pyproject"


def test_the_base_dependencies_are_what_the_readme_says(readme):
    pyproject = (ROOT / "pyproject.toml").read_text()
    base = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    install = readme.split("## Install", 1)[1].split("## ", 1)[0]
    assert "NumPy and Polars and nothing else" in install
    assert "numpy" in base and "polars" in base
    for absent in ("altair", "pyarrow", "vl-convert"):
        assert absent not in base, f"{absent} is a base dependency again"


# ─── links resolve ─────────────────────────────────────────────────


def test_every_relative_link_resolves(readme):
    """The parity link pointed at `parity_and_bugs.md`; the source is an
    `.ipynb`, so the link had never resolved."""
    broken = []
    for target in re.findall(r"\]\((?!https?:|#)([^)\s]+)\)", readme):
        path = ROOT / target.split("#", 1)[0]
        if not path.exists():
            broken.append(target)
    assert broken == [], broken
