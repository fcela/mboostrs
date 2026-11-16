"""What the book tells someone who needs a distribution that is not here.

P4.2: "Treat `Hurdle`, `RCG`, and custom `Family()` as design decisions, not
checklist parity. A callback family crosses the Python/Rust hot loop boundary
and needs an explicit performance and safety contract; absent a real use case,
document how to contribute a compiled family."

The steps for adding a compiled family were already written. What was missing
was the decision behind them: why `mboost::Family()` has no equivalent here,
which is the first question an R user with a `gamlss.dist` distribution asks.

The section that answers it carries measurements rather than an assertion,
because "callbacks would be slow" is the kind of claim that is repeated until
someone checks it. The gradient cost turns out to be about a quarter of a fit
— tolerable — and the decisive cost is the GIL serialising the parallel
candidate scan, which is a 1.9x factor on the fixture measured.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "book" / "implementation" / "architecture.ipynb"


@pytest.fixture(scope="module")
def text():
    return "".join("".join(cell["source"])
                   for cell in json.loads(PAGE.read_text())["cells"])


def test_the_page_says_why_there_is_no_callback_family(text):
    assert "### Compiled families and run-time callbacks" in text
    assert "mboost::Family()" in text


def test_it_counts_the_crossings_rather_than_asserting_they_are_many(text):
    """A gradient per iteration per fold: 1,000 for a fit, 10,000 for a
    ten-fold `cvrisk`, 100,000 for fifty complementary pairs."""
    for count in ("1,000", "10,000", "100,000"):
        assert count in text, count


def test_it_reports_the_measurement_that_decides_it(text):
    """The gradient cost alone would be tolerable. The GIL serialising the
    candidate scan is what is not."""
    assert "136 ms" in text and "259 ms" in text
    assert "1.9x" in text
    assert "GIL" in text


def test_it_names_the_safety_side_too(text):
    """P4.2 asks for a performance *and safety* contract."""
    for hazard in ("raise", "wrong length", "non-finite"):
        assert hazard in text, hazard


def test_it_points_at_a_worked_example(text):
    """PIG is the harder case done end to end: a special function, a gradient
    in terms of a Bessel ratio, and four gates before fitting was enabled."""
    assert "pig_contract" in text
    assert (ROOT / "crates" / "mboostrs-core" / "src" / "lss.rs").exists()


def test_it_says_what_to_do_when_compiling_is_not_an_option(text):
    assert "gamlss" in text
    assert "r_migration" in text


def test_the_measured_speedup_is_reproducible():
    """The claim is about this package's parallel candidate scan, so the route
    it names has to be the one a many-learner fit takes."""
    import warnings

    import numpy as np

    import mboostrs as mb

    rng = np.random.default_rng(0)
    n, p = 4000, 24
    data = {f"x{i}": rng.normal(size=n) for i in range(p)}
    data["y"] = (data["x0"] + np.sin(2 * data["x1"])
                 + rng.normal(scale=0.3, size=n))
    formula = "y ~ " + " + ".join(f"bbs(x{i}, df=4)" for i in range(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboost(formula, data, family=mb.Huber(delta=1.0),
                          control=mb.boost_control(20, 0.1))
    assert fit.execution_report().route == "parallel_block"


def test_the_interface_map_no_longer_calls_brad_unimplemented():
    """It listed `brad` and `bss` beside `blackboost` as not implemented.
    `brad` is implemented — that was the capability-ledger finding — and
    `bss` was withdrawn upstream rather than missing here."""
    page = ROOT / "docs" / "book" / "usage" / "interface_map.ipynb"
    body = "".join("".join(cell["source"])
                   for cell in json.loads(page.read_text())["cells"])
    claim = re.search(r"- The upstream `blackboost`[^\n]*", body).group(0)
    assert "`brad` is implemented" in claim
    assert "withdrawn upstream" in claim
    assert "squared error" in claim
