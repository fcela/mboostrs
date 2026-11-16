"""The agent entry points are reachable from the installed package (#306)."""

import inspect
import subprocess
import sys
from pathlib import Path

import mboostrs

REPO = Path(__file__).resolve().parents[1]
DOC_SKILL = REPO / "docs" / "skills" / "analyzing-data-with-mboostrs" / "SKILL.md"


def test_insights_is_the_function_not_the_module():
    assert inspect.isfunction(mboostrs.insights)
    assert mboostrs.insights.__module__ == "mboostrs.insights"
    from mboostrs.insights import Finding, insights
    assert insights is mboostrs.insights and Finding is mboostrs.Finding


def test_explicit_submodule_import_does_not_rebind_the_attribute():
    code = ("import inspect, mboostrs, mboostrs.insights; "
            "assert inspect.isfunction(mboostrs.insights)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_agent_guide_returns_the_skill():
    text = mboostrs.agent_guide()
    assert "Analyzing data with mboostrs" in text
    assert Path(mboostrs.agent_guide(path_only=True)).is_file()


def test_shipped_skill_matches_the_docs_copy():
    shipped = Path(mboostrs.agent_guide(path_only=True)).read_bytes()
    assert shipped == DOC_SKILL.read_bytes(), (
        "python/mboostrs/agent_docs/SKILL.md drifted from docs/skills/...; "
        "copy the docs file over it")


def test_package_docstring_points_at_the_guide():
    assert "agent_guide()" in mboostrs.__doc__
