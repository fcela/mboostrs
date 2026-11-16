"""Every packaged dataset has a provenance record, and it matches the data.

D3: "Add data cards for source, license, retrieval date, curation script,
exclusions, units, missingness, and known biases. The packaged datasets
currently rely mostly on loader docstrings rather than a machine-readable
provenance record."

The split is the point. Fields that can be **measured** from the shipped file
are generated, so they cannot drift from the data they describe: a column
added to a CSV fails the `--check`. Fields that cannot be measured are `None`
until somebody records them.

`None` means "not recorded", never "none apply". Ninety-six such fields are
open across sixteen datasets, and that number is reported rather than hidden:
a reader deciding whether they may redistribute a dataset needs to see that
the licence is unknown, not read a guess. Inventing one to make a table look
complete is the failure this structure exists to prevent.
"""

import inspect
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from mboostrs import datasets as ds
from mboostrs.datasets import data_card, data_cards

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from generate_data_cards import UNRECORDED, loaders  # noqa: E402


def test_every_packaged_loader_has_a_card():
    missing = [name for name in loaders() if name not in data_cards()]
    assert missing == [], missing


def test_no_card_describes_a_dataset_that_is_gone():
    extra = [name for name in data_cards() if name not in loaders()]
    assert extra == [], extra


def test_the_simulators_have_no_cards():
    """`friedman1` and friends generate their data, so there is no provenance
    to record — a card for one would be a claim about nothing."""
    cards = data_cards()
    for name in ("friedman1", "sparse_linear", "heteroscedastic_lss"):
        assert callable(getattr(ds, name))
        assert name not in cards


@pytest.mark.parametrize("name", loaders())
def test_the_measured_fields_match_the_shipped_data(name):
    """The half that cannot drift: regenerate and compare."""
    card = data_card(name)
    frame = getattr(ds, name)(output="polars")
    assert card["rows"] == frame.height
    assert list(card["columns"]) == list(frame.columns)
    for column, recorded in card["columns"].items():
        series = frame[column]
        assert recorded["dtype"] == str(series.dtype)
        missing = int(series.null_count())
        if series.dtype.is_numeric():
            values = series.drop_nulls().to_numpy()
            if values.dtype.kind == "f":
                missing += int(np.isnan(values).sum())
        assert recorded["missing"] == missing, column


@pytest.mark.parametrize("name", loaders())
def test_every_card_carries_a_summary(name):
    card = data_card(name)
    assert len(card["summary"]) > 15


def test_the_unrecorded_fields_are_present_and_none():
    """Absent keys would let a reader think the field does not apply. They are
    present and `None`, which says the answer is missing."""
    for name, card in data_cards().items():
        for field in UNRECORDED:
            assert field in card, f"{name} has no {field} key"
            assert card[field] is None or isinstance(card[field], str), name


def test_the_open_count_is_visible():
    """Ninety-six unrecorded fields is the honest state, and the generator
    reports it rather than presenting a complete-looking table."""
    open_fields = sum(1 for card in data_cards().values()
                      for field in UNRECORDED if card[field] is None)
    assert open_fields > 0
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_data_cards.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert str(open_fields) in result.stdout


def test_the_committed_cards_are_current():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_data_cards.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_check_fails_when_a_card_is_stale(tmp_path, monkeypatch):
    """A check that cannot fail is not a check."""
    import generate_data_cards

    stale = json.loads(generate_data_cards.OUTPUT.read_text())
    stale["bodyfat"]["rows"] = 1
    path = tmp_path / "data_cards.json"
    path.write_text(json.dumps(stale, indent=2) + "\n")
    monkeypatch.setattr(generate_data_cards, "OUTPUT", path)
    monkeypatch.setattr(sys, "argv", ["generate_data_cards.py", "--check"])
    assert generate_data_cards.main() == 1


def test_a_recorded_field_survives_regeneration(tmp_path, monkeypatch):
    """Someone filling in a licence must not have it wiped by the next run."""
    import generate_data_cards

    existing = {"bodyfat": {"license": "CC-BY-4.0"}}
    rebuilt = generate_data_cards.build(existing)
    assert rebuilt["bodyfat"]["license"] == "CC-BY-4.0"
    assert rebuilt["bodyfat"]["retrieved"] is None
    assert rebuilt["bodyfat"]["rows"] == ds.bodyfat(output="polars").height


def test_an_unknown_dataset_is_refused_by_name():
    with pytest.raises(KeyError, match="no data card for"):
        data_card("not_a_dataset")


def test_the_accessor_is_documented():
    assert "not recorded" in inspect.getdoc(data_card)
