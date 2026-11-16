"""FAMILY_SETS and family_set() are reachable from the top level (#337)."""

import inspect
from pathlib import Path

import pytest

import mboostrs as mb
from mboostrs import model_selection

REPO = Path(__file__).resolve().parents[1]
# The quick reference moved from SKILL.md to this topic (SKILL.md is an entry point).
QUICKREF = (REPO / "docs" / "skills" / "analyzing-data-with-mboostrs" / "topics"
            / "quick-reference.md")


def test_family_sets_is_exported_at_top_level_and_is_the_same_object():
    assert mb.FAMILY_SETS is model_selection.FAMILY_SETS
    assert "FAMILY_SETS" in mb.__all__


def test_family_set_is_exported_at_top_level():
    assert mb.family_set is model_selection.family_set
    assert "family_set" in mb.__all__


def test_family_set_returns_the_families_for_one_support():
    realplus = mb.family_set("realplus")
    assert realplus[:4] == ["GammaLSS", "InverseGaussianLSS", "Pareto2LSS", "BCCGLSS"]
    assert {"LogNormalLSS", "WeibullLSS"} <= set(realplus)
    assert mb.family_set("real0to1") == ["BetaLSS"]


@pytest.mark.parametrize("support", sorted(mb.FAMILY_SETS))
def test_family_set_matches_the_table_for_every_support(support):
    assert mb.family_set(support) == list(mb.FAMILY_SETS[support])


@pytest.mark.parametrize("support", sorted(mb.FAMILY_SETS))
def test_every_family_named_by_a_support_is_a_public_class(support):
    for name in mb.family_set(support):
        assert inspect.isclass(getattr(mb, name)), name


def test_family_set_returns_a_copy_that_leaves_the_table_unchanged():
    before = tuple(mb.FAMILY_SETS["realplus"])
    names = mb.family_set("realplus")
    names.append("GaussianLSS")
    assert tuple(mb.FAMILY_SETS["realplus"]) == before


@pytest.mark.parametrize("bad", ["nonsense", "realPlus", "", None, ["realplus"]])
def test_unknown_support_raises_value_error_naming_the_valid_supports(bad):
    with pytest.raises(ValueError) as info:
        mb.family_set(bad)
    message = str(info.value)
    for valid in mb.FAMILY_SETS:
        assert valid in message, valid
    assert "family_set('realplus')" in message


def test_scan_lss_families_docstring_says_where_the_sets_live():
    doc = model_selection.scan_lss_families.__doc__
    assert "mboostrs.model_selection" in doc
    assert "mb.FAMILY_SETS" in doc
    assert "mb.family_set(support)" in doc


def test_skill_quick_reference_mentions_family_set():
    quickref = QUICKREF.read_text(encoding="utf-8")
    assert 'mb.family_set("realplus")' in quickref
