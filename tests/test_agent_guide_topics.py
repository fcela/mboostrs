"""The agent guide's topic sections ship, run, and install (claims-discovery benchmark)."""

import importlib.util
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import mboostrs
from mboostrs.agent_docs import TOPICS

REPO = Path(__file__).resolve().parents[1]
DOC_TOPICS = REPO / "docs" / "skills" / "analyzing-data-with-mboostrs" / "topics"


@pytest.mark.parametrize("topic", list(TOPICS))
def test_shipped_topic_matches_the_docs_copy(topic):
    shipped = Path(mboostrs.agent_guide(topic, path_only=True)).read_bytes()
    assert shipped == (DOC_TOPICS / TOPICS[topic][0]).read_bytes(), (
        f"python/mboostrs/agent_docs/topics/{TOPICS[topic][0]} drifted from "
        "docs/skills/.../topics/; copy the docs file over it")


def test_every_docs_topic_is_registered():
    on_disk = {p.name for p in DOC_TOPICS.glob("*.md")}
    assert on_disk == {filename for filename, _ in TOPICS.values()}


def test_index_lists_every_topic_and_the_skill_names_them():
    index = mboostrs.agent_guide("index")
    skill = mboostrs.agent_guide()
    for topic in TOPICS:
        assert topic in index
        assert f'"{topic}"' in skill, f"SKILL.md does not point at topic {topic!r}"


def test_unknown_topic_names_the_topics():
    with pytest.raises(ValueError, match="missing-data"):
        mboostrs.agent_guide("missing")


# Topics that state the method in prose and have no runnable block. A topic
# with a python block must not be listed here, and a topic listed here must not
# gain one without being removed from this set.
PROSE_TOPICS = {
    "insights", "before-the-first-fit", "screen-and-draft", "fit-and-tune",
    "interactions-and-terms", "specification-search", "distributional-branch",
    "sensitivity-and-contrasts", "explain", "pitfalls", "interpretation",
    "reading-a-fit", "closing-the-analysis", "discover-limits", "task-map",
}

#: Largest guide file an agent tool shows in one read is about 9,000 characters;
#: the entry point is held to a tighter limit so it fits with its index.
GUIDE_LIMIT = 9000
ENTRY_LIMIT = 8000


def test_prose_topics_are_registered():
    assert PROSE_TOPICS <= set(TOPICS)


# A python fence may name the packages it imports: ```python requires=econml.
# The example runs only when each named package is importable and is skipped
# otherwise, so the guide can show a route that needs an optional dependency.
_BLOCK_RE = re.compile(r"```python([^\n]*)\n(.*?)```", re.S)


def _blocks(topic):
    """(info, code) for each python fence of a topic."""
    return _BLOCK_RE.findall(mboostrs.agent_guide(topic))


def _requires(info):
    match = re.search(r"requires=([\w,]+)", info)
    return match.group(1).split(",") if match else []


@pytest.mark.parametrize("topic", list(TOPICS))
def test_topic_has_an_example_unless_it_is_prose(topic):
    blocks = _blocks(topic)
    if topic in PROSE_TOPICS:
        assert not blocks, f"topic {topic!r} has a runnable example; remove it from PROSE_TOPICS"
    else:
        assert blocks, f"topic {topic!r} has no runnable example; add one or list it in PROSE_TOPICS"


_EXAMPLES = [
    pytest.param(topic, k, _requires(info), code, id=f"{topic}-{k}")
    for topic in TOPICS
    for k, (info, code) in enumerate(_blocks(topic))
]


@pytest.mark.parametrize("topic, k, requires, code", _EXAMPLES)
def test_topic_example_runs(topic, k, requires, code):
    missing = [name for name in requires if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"the example needs {', '.join(missing)}, which is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, f"agent_guide:{topic}:{k}", "exec"), {})


def test_no_guide_file_exceeds_the_size_limit():
    files = {"SKILL.md": Path(mboostrs.agent_guide(path_only=True))}
    for topic in TOPICS:
        files[TOPICS[topic][0]] = Path(mboostrs.agent_guide(topic, path_only=True))
    sizes = {name: len(path.read_text(encoding="utf-8")) for name, path in files.items()}
    too_long = {name: size for name, size in sizes.items() if size > GUIDE_LIMIT}
    assert not too_long, f"guide files over {GUIDE_LIMIT} characters: {too_long}"


def test_discover_topic_keeps_the_scale_documentation_within_the_limit():
    # The size guard can be met by deleting prose, so pin the fact the discover
    # topic must state: a spread model adds a `sigma` column to `predictions`.
    # The topic grew past GUIDE_LIMIT when `sigma` was documented (#589).
    text = mboostrs.agent_guide("discover")
    assert "`sigma`" in text, "the discover topic no longer documents `sigma`"
    assert len(text) <= GUIDE_LIMIT, f"discover topic is {len(text)} characters"


def test_entry_point_stays_short_enough_to_read_whole():
    size = len(mboostrs.agent_guide())
    assert size < ENTRY_LIMIT, f"SKILL.md is {size} characters; keep it under {ENTRY_LIMIT}"


def test_install_skill_copies_skill_and_topics(tmp_path):
    target = mboostrs.install_skill(tmp_path / "skills")
    assert (target / "SKILL.md").read_text() == mboostrs.agent_guide()
    for topic, (filename, _) in TOPICS.items():
        assert (target / "topics" / filename).read_text() == mboostrs.agent_guide(topic)
    with pytest.raises(FileExistsError, match="overwrite"):
        mboostrs.install_skill(tmp_path / "skills")
    assert mboostrs.install_skill(tmp_path / "skills", overwrite=True) == target


def test_command_line_guide_and_install(tmp_path):
    out = subprocess.run([sys.executable, "-m", "mboostrs", "guide", "index"],
                         capture_output=True, text=True, check=True).stdout
    assert "missing-data" in out
    dest = tmp_path / "s"
    out = subprocess.run([sys.executable, "-m", "mboostrs", "skill", "install",
                          "--dest", str(dest)], capture_output=True, text=True,
                         check=True).stdout
    assert (dest / "analyzing-data-with-mboostrs" / "SKILL.md").is_file()
    assert "analyzing-data-with-mboostrs" in out
