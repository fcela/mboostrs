"""D5 discoverability: api_card and the `sig` CLI give the signature, return
type and a minimal example for a public name; the high-churn docstrings open
with a short usage block (#505)."""

import inspect
import re
import subprocess
import sys

import pytest

import mboostrs as mb

HIGH_CHURN = [
    mb.tune_mstop,
    mb.compare_specifications,
    mb.scan_lss_families,
    mb.GamlssRefit,
    mb.conformal_interval,
    mb.GAMBoostModel.predict,
    mb.save_model,
    mb.load_model,
    mb.stabsel,
]


def test_api_card_is_exported():
    assert "api_card" in mb.__all__
    assert callable(mb.api_card)
    assert mb.api_card.__module__ == "mboostrs.api_card"
    from mboostrs.api_card import api_card

    assert api_card is mb.api_card


@pytest.mark.parametrize("obj", [mb.tune_mstop, mb.stabsel])
def test_api_card_is_one_bounded_read(obj):
    card = mb.api_card(obj)
    lines = card.splitlines()
    assert len(lines) <= 20
    assert obj.__name__ in card
    assert "(" in card
    assert "->" in card  # the return type is on the card
    assert "Example:" in card
    assert "import mboostrs" in card


def test_api_card_resolves_a_name_string():
    card = mb.api_card("tune_mstop")
    assert card == mb.api_card(mb.tune_mstop)
    assert "tune_mstop(" in card


def test_api_card_resolves_a_dotted_method_path():
    card = mb.api_card("GAMBoostModel.predict")
    assert "GAMBoostModel.predict(" in card
    assert "->" in card
    assert "import mboostrs" in card
    as_object = mb.api_card(mb.GAMBoostModel.predict)
    assert as_object.startswith("predict(")  # the object's own name


def test_api_card_unknown_name_raises():
    with pytest.raises(KeyError):
        mb.api_card("no_such_name_505")


def test_every_high_churn_name_resolves_and_is_bounded():
    for obj in HIGH_CHURN:
        card = mb.api_card(obj)
        assert len(card.splitlines()) <= 20, obj.__name__
        assert callable(obj) and "(" in card, obj.__name__


def test_cli_sig_exits_zero_and_prints_a_card():
    result = subprocess.run(
        [sys.executable, "-m", "mboostrs", "sig", "tune_mstop"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tune_mstop(" in result.stdout
    assert "(" in result.stdout


def test_cli_sig_refuses_an_unknown_name():
    result = subprocess.run(
        [sys.executable, "-m", "mboostrs", "sig", "no_such_name_505"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0


def _usage_like(doc: str) -> bool:
    """Heuristic: a `Usage::`-style marker near the top is followed by at
    least three code-like lines, and the docstring continues past them."""
    lines = doc.splitlines()
    for i, line in enumerate(lines[:12]):
        if not re.search(r"\bUsage::|>>>|(^|\s)import mboostrs", line):
            continue
        code = sum(
            1 for item in lines[i + 1:i + 12]
            if item.strip() and re.search(r"import |mb\.|[\(\)\[\]=]", item)
        )
        return code >= 3 and len(lines) - (i + 1) >= 3
    return False


@pytest.mark.parametrize("obj", HIGH_CHURN)
def test_high_churn_docstrings_open_with_a_usage_block(obj):
    doc = inspect.getdoc(obj)
    assert doc, obj.__name__
    assert _usage_like(doc), obj.__name__