"""The end-to-end matrix P1.2 specifies, and the asymmetry it found.

P1.2 names a minimum matrix crossing the four model kinds, four weight kinds,
three prediction kinds, and exact/binned/serialized fits "where applicable".
The pieces were covered — a weighted-parity test here, a serialization round
trip there — and nothing said which of the 144 cells actually run, so an
uncovered cell and an inapplicable one looked the same from outside.

The invariant every cell holds is that a model asked to predict its own
training rows returns its own fitted values. Everything the matrix varies is a
way of storing or replaying a fit — a weight rescaling, a binned design, a
JSON round trip, a truncated path — and each is a chance to lose the mapping
from a row to its prediction. It needs no oracle, and it is what a replay bug
breaks first.

Running it found one: `fit.with_mstop(fit.mstop)` raised on a non-cyclic
distributional fit.
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest

import mboostrs as mb

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools" / "check_semantic_matrix.py"


@pytest.fixture(scope="module")
def matrix():
    sys.path.insert(0, str(ROOT))
    from tools import check_semantic_matrix
    return check_semantic_matrix


@pytest.fixture(scope="module")
def noncyclic():
    rng = np.random.default_rng(0)
    n = 150
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = (data["x"] + np.sin(2 * data["z"])
                 + rng.normal(scale=0.3, size=n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS("y ~ bbs(x, df=4) + bbs(z, df=4)", data,
                             family=mb.GaussianLSS(),
                             control=mb.boost_control(30, 0.1),
                             method="noncyclic")
    return fit, data


def test_the_matrix_covers_what_p1_2_names(matrix):
    assert set(matrix.MODELS) == {"glmboost", "gamboost", "glmboostLSS",
                                  "gamboostLSS"}
    assert set(matrix.WEIGHT_KINDS) == {"unit", "integer", "fractional",
                                        "zeroone"}
    assert set(matrix.FIT_MODES) == {"exact", "binned", "serialized"}
    assert set(matrix.PREDICTIONS) == {"in_sample", "truncated", "newdata"}


def test_every_skipped_cell_says_why(matrix):
    """"Where applicable" has to name the exception, or an uncovered cell
    hides behind an inapplicable one."""
    assert matrix.INAPPLICABLE
    for (model, mode), reason in matrix.INAPPLICABLE.items():
        assert model in matrix.MODELS
        assert mode in matrix.FIT_MODES
        assert len(reason) > 40, reason


def test_glmboost_really_has_no_bins_argument():
    """The recorded reason, checked rather than trusted."""
    import inspect

    assert "bins" not in inspect.signature(mb.glmboost).parameters
    assert "bins" not in inspect.signature(mb.glmboostLSS).parameters
    assert "bins" in inspect.signature(mb.gamboost).parameters
    assert "bins" in inspect.signature(mb.gamboostLSS).parameters


def test_the_whole_matrix_holds():
    result = subprocess.run([sys.executable, str(CHECK)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "120 cell(s)" in result.stdout


@pytest.mark.parametrize("weight_kind", ["unit", "integer", "fractional",
                                         "zeroone"])
def test_a_serialized_gamboost_replays_under_every_weight_kind(matrix,
                                                               weight_kind):
    """The cell most likely to break quietly: the artifact carries the fitted
    path and not the weights, so a weight rule applied at fit time has to
    already be baked into what it stores."""
    assert matrix.run_cell("gamboost", weight_kind, "serialized",
                           "newdata") is None


def test_a_noncyclic_fit_accepts_its_own_mstop(noncyclic):
    """A non-cyclic fit is given one budget and decides how to divide it, so
    `mstop` reports an allocation while the control takes a total. Passing the
    model its own reported value raised `noncyclic LSS update requires a
    scalar mstop` — the truncation refusing the number the model had just
    printed."""
    fit, _ = noncyclic
    assert isinstance(fit.mstop, dict)
    same = fit.with_mstop(fit.mstop)
    assert same.mstop == fit.mstop
    for name, values in fit.fitted().items():
        assert np.allclose(values, same.fitted()[name])


def test_an_allocation_the_algorithm_would_not_choose_is_refused(noncyclic):
    """Accepting the reported allocation must not turn into pretending a
    non-cyclic fit can be steered per parameter. It cannot, and the message
    says what to use instead."""
    fit, _ = noncyclic
    total = sum(fit.mstop.values())
    impossible = {"mu": 1, "sigma": total - 1}
    assert impossible != fit.mstop
    with pytest.raises(ValueError, match="chooses how to divide its budget"):
        fit.with_mstop(impossible)


def test_a_cyclic_fit_still_takes_a_per_parameter_mstop():
    """The behaviour that was already right, and that the message points at."""
    rng = np.random.default_rng(1)
    n = 120
    data = {"x": rng.normal(size=n)}
    data["y"] = data["x"] + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS("y ~ bbs(x, df=4)", data,
                             family=mb.GaussianLSS(),
                             control=mb.boost_control(30, 0.1),
                             method="cyclic")
        truncated = fit.with_mstop({"mu": 12, "sigma": 5})
    assert truncated.mstop == {"mu": 12, "sigma": 5}


def test_a_scalar_still_means_the_total_for_a_noncyclic_fit(noncyclic):
    fit, _ = noncyclic
    smaller = fit.with_mstop(10)
    assert sum(smaller.mstop.values()) == 10
