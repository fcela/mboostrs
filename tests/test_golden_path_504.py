"""The golden-path topic leads ``agent_guide()`` and stays short (#504).

D5 (experimental iteration speed) found that agents read many guide topics
before their first fit. The fix is one default topic, ``golden-path``, that is
the whole workflow on one page: data -> ``fit()`` -> ``report()`` -> three
follow-ups, pointing at a topic only when its evidence appears.
"""

from pathlib import Path

import mboostrs
import pytest

from mboostrs.agent_docs import TOPICS

REPO = Path(__file__).resolve().parents[1]
DOC_TOPICS = REPO / "docs" / "skills" / "analyzing-data-with-mboostrs" / "topics"


def test_golden_path_is_the_first_topic():
    assert next(iter(TOPICS)) == "golden-path"
    assert TOPICS["golden-path"][0] == "golden-path.md"


def test_golden_path_is_registered_and_installable(tmp_path):
    assert mboostrs.agent_guide("golden-path")
    target = mboostrs.install_skill(tmp_path)
    assert (target / "topics" / "golden-path.md").is_file()


def test_default_agent_guide_leads_with_the_golden_path():
    head = mboostrs.agent_guide()[:2000]
    assert "golden-path" in head
    assert "fit()" in head or "report()" in head
    assert '"golden-path"' in mboostrs.agent_guide()


def test_index_lists_golden_path_first():
    index = mboostrs.agent_guide("index")
    assert "golden-path" in index
    assert index.index("golden-path") < index.index("quick-reference")


def test_topic_stays_within_the_one_page_budget():
    text = mboostrs.agent_guide("golden-path")
    assert len(text.splitlines()) <= 150, "golden-path climbed over 150 lines"
    assert len(text) <= 8000, "golden-path climbed over 8,000 characters"


def test_topic_names_the_current_path_and_the_supersession():
    text = mboostrs.agent_guide("golden-path")
    for needle in ("hasattr", "fit", "report", "gamboost",
                   "tune_mstop", "reference_gap"):
        assert needle in text, f"golden-path does not name {needle!r}"
    assert "only if" in text, "golden-path has no 'open topic X only if Y' pointers"


def test_skill_and_agent_docs_trees_stay_in_sync():
    shipped_skill = Path(mboostrs.agent_guide(path_only=True)).read_bytes()
    assert shipped_skill == (DOC_TOPICS.parent / "SKILL.md").read_bytes()
    shipped_topic = Path(mboostrs.agent_guide("golden-path", path_only=True)).read_bytes()
    assert shipped_topic == (DOC_TOPICS / "golden-path.md").read_bytes()


@pytest.mark.parametrize("topic", list(TOPICS))
def test_every_topic_resolves(topic):
    assert mboostrs.agent_guide(topic)
