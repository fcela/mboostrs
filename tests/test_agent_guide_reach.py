"""The guide reaches an agent that has only the installed package (#468).

The worked examples ship with the package and are read through
``agent_guide("example:<name>")``; the fitting entry points and the warnings
that carry a remedy name the guide.
"""

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.agent_docs import EXAMPLES, TOPICS, guide_hint

REPO = Path(__file__).resolve().parents[1]
DOC_EXAMPLES = REPO / "docs" / "skills" / "analyzing-data-with-mboostrs" / "examples"


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_shipped_example_matches_the_docs_copy(name):
    shipped = Path(mb.agent_guide(f"example:{name}", path_only=True)).read_bytes()
    assert shipped == (DOC_EXAMPLES / EXAMPLES[name][0]).read_bytes()


def test_every_docs_example_is_registered():
    assert {p.name for p in DOC_EXAMPLES.glob("*.md")} == {f for f, _ in EXAMPLES.values()}


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_fits_in_one_read(name):
    assert len(mb.agent_guide(f"example:{name}")) < 9000


def test_index_lists_the_examples_and_skill_names_them():
    index = mb.agent_guide("index")
    skill = mb.agent_guide()
    for name in EXAMPLES:
        assert f"example:{name}" in index
        assert f"example:{name}" in skill


def test_unknown_example_names_the_examples():
    with pytest.raises(ValueError, match="example:rent-analysis"):
        mb.agent_guide("example:no-such-analysis")


def test_install_skill_copies_the_examples(tmp_path):
    target = mb.install_skill(tmp_path)
    for filename, _ in EXAMPLES.values():
        assert (target / "examples" / filename).is_file()


def test_wheel_includes_the_examples():
    text = (REPO / "pyproject.toml").read_text()
    assert 'python/mboostrs/agent_docs/examples/*.md' in text


@pytest.mark.parametrize("name", ["gamboost", "glmboost", "gam", "gamlss", "discover",
                                  "gamboostLSS", "glmboostLSS"])
def test_fitting_entry_points_name_the_guide(name):
    doc = getattr(mb, name).__doc__
    assert "agent_guide()" in doc and "reference_gap" in doc


def test_guide_hint_refuses_an_unknown_topic():
    with pytest.raises(ValueError, match="unknown agent_guide topic"):
        guide_hint("no-such-topic")


def test_guide_hint_names_a_registered_topic():
    assert guide_hint("pitfalls") == " Guide: python -m mboostrs guide pitfalls."
    assert "modeling-tactics" in TOPICS


def test_extrapolation_warning_names_the_guide_topic():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 300)
    data = pl.DataFrame({"x": x, "y": np.sin(4 * x) + rng.normal(0, 0.1, 300)})
    model = mb.gamboost("y ~ bbs(x)", data=data, control=mb.boost_control(mstop=50))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(pl.DataFrame({"x": [2.0]}))
    messages = [str(w.message) for w in caught]
    assert any("Guide: python -m mboostrs guide modeling-tactics." in m for m in messages)
