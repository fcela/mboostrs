"""Altair output is rendered the same through expression and display forms."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "book" / "merge-outputs.py"


def _module():
    spec = importlib.util.spec_from_file_location("merge_outputs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("output_type", ["execute_result", "display_data"])
def test_altair_svg_replaces_the_plain_repr_for_both_output_forms(
    tmp_path, output_type
):
    svg = tmp_path / "build" / "chart.svg"
    svg.parent.mkdir()
    svg.write_text("<svg><title>Rendered chart</title></svg>")
    node = {
        "jupyter_data": {
            "output_type": output_type,
            "data": {
                "image/svg+xml": {"path": "/build/chart.svg", "hash": "abc"},
                "text/plain": {"content": "alt.LayerChart(...)"},
            },
        }
    }

    _module().inline_svg_refs(node, tmp_path)

    data = node["jupyter_data"]["data"]
    assert data["image/svg+xml"]["content"].startswith("<svg>")
    assert "text/plain" not in data


def test_non_chart_output_is_unchanged(tmp_path):
    node = {
        "jupyter_data": {
            "output_type": "stream",
            "data": {"text/plain": {"content": "alt.LayerChart(...)"}},
        }
    }

    _module().inline_svg_refs(node, tmp_path)

    assert "text/plain" in node["jupyter_data"]["data"]


def test_a_remaining_chart_fallback_is_reported():
    node = {
        "jupyter_data": {
            "output_type": "display_data",
            "data": {"text/plain": {"content": "alt.LayerChart(...)"}},
        }
    }

    problems = _module().unresolved_chart_outputs(node)

    assert problems == ["root: Altair text fallback remains"]


def test_svg_path_under_a_pages_base_url_resolves_to_the_build_directory(
    tmp_path, monkeypatch
):
    """With BASE_URL=/mboostrs, MyST writes /mboostrs/build/<hash>.svg; the file
    is at <html>/build/<hash>.svg."""
    svg = tmp_path / "build" / "chart.svg"
    svg.parent.mkdir()
    svg.write_text("<svg><title>Rendered chart</title></svg>")
    monkeypatch.setenv("BASE_URL", "/mboostrs")
    node = {
        "jupyter_data": {
            "output_type": "display_data",
            "data": {
                "image/svg+xml": {"path": "/mboostrs/build/chart.svg", "hash": "abc"},
                "text/plain": {"content": "alt.Chart(...)"},
            },
        }
    }

    _module().inline_svg_refs(node, tmp_path)

    data = node["jupyter_data"]["data"]
    assert data["image/svg+xml"]["content"].startswith("<svg>")
    assert "text/plain" not in data
