"""The beginner path teaches decisions before catalogues.

D2 asks for one beginner path built around five decisions: what quantity is
the target; what the response support and censoring allow; what shape the
learner can represent and what its null space leaves unpenalised; how `mstop`
is chosen without using evaluation data twice; and which diagnostic can
falsify the assumptions.

Before this, a reader who had installed the package met the interface map — a
matrix of every entry point — and then a catalogue of families and another of
learners. Catalogues answer "what exists", and none of the five questions.

Every API name on the page is checked against the package, because a teaching
page that names a method which does not resolve teaches the wrong thing more
efficiently than prose.
"""

import json
import pathlib
import re
import warnings

import numpy as np
import pytest
import yaml

import mboostrs as mb

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
PAGE = BOOK / "usage" / "five_decisions.ipynb"


@pytest.fixture(scope="module")
def text():
    return "".join(json.loads(PAGE.read_text())["cells"][0]["source"])


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 120
    data = {"x": rng.normal(size=n)}
    data["y"] = 2 * data["x"] + rng.normal(size=n)
    fit = mb.gamboost("y ~ bbs(x, df=4)", data, control=mb.boost_control(30, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit, fit.cvrisk()


def test_the_page_is_in_the_toc_before_the_catalogues():
    config = yaml.safe_load((BOOK / "myst.yml").read_text())

    def walk(entries, out):
        for entry in entries:
            if "file" in entry:
                out.append(entry["file"])
            if "children" in entry:
                walk(entry["children"], out)
        return out

    pages = walk(config["project"]["toc"], [])
    assert "usage/five_decisions.ipynb" in pages
    order = pages.index("usage/five_decisions.ipynb")
    assert order > pages.index("usage/quickstart.ipynb")
    for catalogue in ("usage/families.ipynb", "usage/base_learners.ipynb",
                      "usage/interface_map.ipynb"):
        assert order < pages.index(catalogue), catalogue


def test_it_covers_the_five_decisions_d2_lists(text):
    for decision in ("## 1. Target quantity",
                     "## 2. Response type and family",
                     "## 3. Term shape and base learner",
                     "## 4. Choosing `mstop`",
                     "## 5. Diagnostics for each assumption"):
        assert decision in text, decision


def test_the_null_space_and_df_fairness_are_taught(text):
    """D2 names both explicitly: what the penalty leaves unpenalised, and how
    `df` affects selection fairness."""
    assert "null space" in text
    assert "same `df`" in text
    assert "learner_flexibility" in text, "the measurement should be cited"


def test_mstop_is_taught_without_using_evaluation_data_twice(text):
    assert "cvrisk" in text
    assert "mstop_1se" in text, "the one-standard-error rule"
    assert "still falling" in text, "the boundary trap"
    assert "grouped_folds" in text and "rolling_folds" in text


def test_every_api_name_on_the_page_resolves(text):
    public = set(mb.__all__)
    unresolved = sorted({n for n in re.findall(r"\bmb\.(\w+)", text)
                         if n not in public})
    assert unresolved == []


def test_every_method_named_on_the_page_exists(fitted, text):
    fit, cvr = fitted
    named = sorted(set(re.findall(r"\bfit\.(\w+)\(", text))
                   | set(re.findall(r"\bcvr\.(\w+)", text)))
    assert named, "the page should name methods"
    missing = [m for m in named if not (hasattr(fit, m) or hasattr(cvr, m))]
    assert missing == []


def test_the_refusal_message_it_quotes_is_the_real_one(text):
    """It quotes a `Binomial` refusal; if the wording changed, the page would
    be teaching a message the reader will never see."""
    assert "y must be in {-1, +1}, {0, 1}" in text
    rng = np.random.default_rng(1)
    frame = {"x": rng.normal(size=80), "y": rng.normal(size=80)}
    with pytest.raises(ValueError) as caught:
        mb.glmboost("y ~ x", frame, family=mb.Binomial(),
                    control=mb.boost_control(5, 0.1))
    assert "y must be in {-1, +1}, {0, 1}" in str(caught.value)


def test_every_cross_reference_resolves(text):
    broken = []
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if not (PAGE.parent / target).exists():
            broken.append(target)
    assert broken == []


def test_the_decision_headings_are_labels(text):
    """The five decisions are numbered sections whose headings name the
    decision, so the table of contents reads as a list of topics."""
    decisions = re.findall(r"^## \d\. (.+)$", text, re.M)
    assert len(decisions) == 5
    assert not any(d.rstrip().endswith("?") for d in decisions), decisions
