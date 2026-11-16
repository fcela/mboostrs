"""Every standard family, across the aspects P1.2 lists.

P1.2 asks for each family to be checked on constructor defaults, response
domain and coding, offset, response link, negative gradient, risk, weights,
prediction, and diagnostics. The LSS half of that sweep existed —
`test_family_contract_is_reachable.py` fits and predicts every LSS family the
manifest claims. The standard families had no equivalent: their coverage sat
in per-family parity tests, which cover the families someone wrote a test for
and say nothing about the rest.

Running one contract over all 21 found two defects and one shared property
worth naming.
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest

import mboostrs as mb

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools" / "check_family_contract.py"


@pytest.fixture(scope="module")
def contract():
    sys.path.insert(0, str(ROOT))
    from tools import check_family_contract
    return check_family_contract


def test_the_contract_holds_for_every_standard_family():
    result = subprocess.run([sys.executable, str(CHECK)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "22 standard families" in result.stdout


def test_it_covers_the_aspects_p1_2_names(contract):
    assert set(contract.ASPECTS) == {
        "constructor", "domain", "refusal", "offset", "link", "gradient",
        "risk", "weights", "prediction", "diagnostics",
    }


def test_the_response_domain_is_discovered_not_mapped(contract):
    """A family-to-domain table would be another registry to keep in step,
    which is the failure `test_family_contract_is_reachable.py` was written to
    prevent. Each family is offered candidates and must accept one."""
    _, responses = contract.candidates()
    assert len(responses) >= 7
    families = contract.standard_families()
    assert len(families) == 22
    assert not hasattr(contract, "DOMAIN_FOR")


# --- what running it found ---------------------------------------------

def test_a_missing_multinomial_label_is_refused_by_name():
    """It raised `KeyError: np.float64(nan)` — an internal dictionary lookup
    escaping to the caller, where every other family names the column and the
    row."""
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    y = rng.integers(0, 3, size=n).astype(float)
    y[0] = np.nan
    with pytest.raises(ValueError, match=r"Column 'y' contains 1 NaN"):
        mb.gamboost("y ~ bols(x)", {"x": x, "y": y},
                    family=mb.Multinomial(n_classes=3),
                    control=mb.boost_control(5, 0.1))


def test_a_missing_label_in_an_object_column_is_refused_too():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=n)
    y = np.array(["a", "b", "c"] * 20, dtype=object)
    y[3] = None
    with pytest.raises(ValueError, match="missing label"):
        mb.gamboost("y ~ bols(x)", {"x": x, "y": y},
                    family=mb.Multinomial(n_classes=3),
                    control=mb.boost_control(5, 0.1))


def test_string_labels_still_fit():
    """The refusal must not catch a clean categorical response."""
    rng = np.random.default_rng(2)
    n = 60
    x = rng.normal(size=n)
    y = np.array(["a", "b", "c"] * 20, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboost("y ~ bols(x)", {"x": x, "y": y},
                          family=mb.Multinomial(n_classes=3),
                          control=mb.boost_control(5, 0.1))
    assert fit.mstop == 5


def test_fitting_does_not_write_the_class_count_onto_the_caller_s_family():
    """`family.n_classes = K` wrote the inferred count onto the caller's
    object, so a `Multinomial()` reused on a second dataset carried the first
    one's count. A fourth label then came back as `IndexError: index 3 is out
    of bounds for axis 0 with size 3`, from inside NumPy."""
    rng = np.random.default_rng(3)
    n = 90
    x = rng.normal(size=n)
    family = mb.Multinomial()
    assert family.n_classes is None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        three = mb.gamboost(
            "y ~ bols(x)", {"x": x, "y": rng.integers(0, 3, n).astype(float)},
            family=family, control=mb.boost_control(5, 0.1))
        assert family.n_classes is None, "the caller's object was modified"
        assert three._family.n_classes == 3, "the model records what it used"

        four = mb.gamboost(
            "y ~ bols(x)", {"x": x, "y": rng.integers(0, 4, n).astype(float)},
            family=family, control=mb.boost_control(5, 0.1))
    assert four._family.n_classes == 4


def test_a_class_count_smaller_than_the_data_says_so():
    rng = np.random.default_rng(4)
    n = 90
    x = rng.normal(size=n)
    with pytest.raises(ValueError, match="holds 5 distinct labels"):
        mb.gamboost("y ~ bols(x)",
                    {"x": x, "y": rng.integers(0, 5, n).astype(float)},
                    family=mb.Multinomial(n_classes=3),
                    control=mb.boost_control(5, 0.1))


def test_adaptive_huber_risk_is_not_monotone():
    """The one the sweep flagged that is not a defect. With no `delta`, the
    changeover point is re-estimated from the current residuals each
    iteration, so the loss changes as the path proceeds and the risk at step k
    is measured under a different function than at k-1. `mboost` does the
    same: 5 increases in 25 steps against our 4 on the same fixture shape."""
    rng = np.random.default_rng(11)
    n = 160
    x = rng.normal(size=n)
    y = 0.5 + 0.4 * x + rng.normal(0, 0.8, n)
    data = {"x": x, "y": y}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adaptive = mb.glmboost("y ~ x", data, family=mb.Huber(),
                               control=mb.boost_control(25, 0.1))
        fixed = mb.glmboost("y ~ x", data, family=mb.Huber(delta=1.0),
                            control=mb.boost_control(25, 0.1))

    assert np.any(np.diff(np.asarray(adaptive.risk_)) > 1e-8)
    assert np.all(np.diff(np.asarray(fixed.risk_)) <= 1e-8)


def test_the_adaptive_risk_is_a_recorded_parity_note():
    from mboostrs._parity_notes import FAMILY_NOTES

    note = next(n for n in FAMILY_NOTES["Huber"]
                if "not comparable" in n.name)
    assert note.shared_with_upstream, "mboost re-estimates delta the same way"
    assert "cvrisk" in note.consequence, "it should say what to do instead"


def test_the_checker_exempts_only_adaptive_losses(contract):
    assert contract.ADAPTIVE_LOSS == {"Huber"}


def test_multinomial_needs_the_gamboost_entry_point(contract):
    """`glmboost` refuses Multinomial up front and names gamboost, as mboost's
    glmboost refuses it (issue #98)."""
    assert contract.ENTRY_POINT["Multinomial"] == "gamboost"
    rng = np.random.default_rng(5)
    n = 60
    with pytest.raises(ValueError, match="glmboost does not fit family=Multinomial"):
        mb.glmboost("y ~ x",
                    {"x": rng.normal(size=n),
                     "y": rng.integers(0, 3, n).astype(float)},
                    family=mb.Multinomial(n_classes=3),
                    control=mb.boost_control(5, 0.1))
