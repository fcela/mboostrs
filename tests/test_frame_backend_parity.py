"""Operations must behave the same whichever frame library is in play.

This package prefers polars and returns polars frames, but several code paths
were written against pandas and reached for a pandas-only method. Two of them
were live defects, and they failed in opposite ways:

* `outlier_frame` guarded its sort on `hasattr(frame, "sort_values")`. A polars
  frame does not have it, so the guard was false and the function returned its
  rows unranked — no error, just the wrong answer, from a function whose one
  job is to rank.
* stability selection's Python refit path indexed rows with `.iloc`, so every
  polars frame raised `AttributeError` before a fold could be fitted.

The first shape is the dangerous one: a pandas-only method name is often just
absent on polars rather than wrong, so a guarded call quietly does nothing.
These tests run the same operation across dict, polars, and pandas inputs and
require the same answer.
"""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

pandas = pytest.importorskip("pandas", reason="pandas is an optional dependency")


@pytest.fixture
def data_with_one_outlier():
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(0, 0.3, n)
    y[5] += 8.0            # one unmistakable outlier, at a known row
    return {"x": x, "y": y}, 5


def _frames(columns):
    return {
        "dict": columns,
        "polars": pl.DataFrame(columns),
        "pandas": pandas.DataFrame(columns),
    }


def _column(frame, name):
    if isinstance(frame, list):
        return np.asarray([row[name] for row in frame])
    return np.asarray(frame[name])


@pytest.mark.parametrize("output", ["auto", "polars", "pandas", "records"])
def test_outlier_frame_ranks_in_every_output_format(data_with_one_outlier, output):
    columns, outlier_row = data_with_one_outlier
    fit = mb.glmboost(
        "y ~ x", data=columns, family=mb.Gaussian(),
        control=mb.boost_control(100, 0.1),
    )
    frame = mb.outlier_frame(fit, output=output)

    magnitude = np.abs(_column(frame, "residual"))
    assert np.all(np.diff(magnitude) <= 1e-12), "rows are not ranked by |residual|"
    assert int(_column(frame, "index")[0]) == outlier_row


@pytest.mark.parametrize("kind", ["dict", "polars", "pandas"])
def test_outlier_frame_ranks_whatever_the_input_frame_is(data_with_one_outlier, kind):
    columns, outlier_row = data_with_one_outlier
    fit = mb.glmboost(
        "y ~ x", data=_frames(columns)[kind], family=mb.Gaussian(),
        control=mb.boost_control(100, 0.1),
    )
    frame = mb.outlier_frame(fit)
    assert int(_column(frame, "index")[0]) == outlier_row


def test_stability_selection_refit_accepts_every_frame_type():
    """The refit path subsets rows itself, so it has to know how each frame
    type does that. `gamboost` takes the Python refit path rather than the
    Rust selected-path fast path, which is where the `.iloc` call sat."""
    rng = np.random.default_rng(3)
    n = 200
    columns = {f"x{j}": rng.normal(size=n) for j in range(5)}
    columns["y"] = np.sin(3 * columns["x0"]) - columns["x1"] + rng.normal(0, 0.3, n)
    formula = "y ~ bbs(x0, df=4) + bols(x1) + bols(x2) + bols(x3) + bols(x4)"

    selected = {}
    for kind, frame in _frames(columns).items():
        fit = mb.gamboost(
            formula, data=frame, family=mb.Gaussian(),
            control=mb.boost_control(80, 0.1),
        )
        with pytest.warns(UserWarning, match="mstop too small"):
            selected[kind] = mb.stabsel(fit, q=3, PFER=1.0, B=6).selected

    assert selected["polars"] == selected["dict"] == selected["pandas"]


def test_the_worm_plot_builds_from_polars():
    """It converted its polars input to pandas, alone among the charts in
    `plot.py`; altair consumes polars directly."""
    from mboostrs.plot import plot_worm

    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, np.exp(-0.5 + 0.4 * x), n)
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"},
        data={"x": x, "y": y}, families=mb.GaussianLSS(),
        control=mb.boost_control(60, 0.1),
    )
    for n_intervals in (1, 4):
        plot_worm(fit, xvar="x", n_intervals=n_intervals, random_state=0).to_dict()


def test_no_module_reaches_for_a_pandas_only_method():
    """A standing guard for the shape both bugs had.

    `sort_values`, `iloc` and `to_pandas` are pandas-only. Calling one on a
    frame this package produced is either a crash or, when it sits behind a
    `hasattr` guard, a silent no-op. Row subsetting and sorting must dispatch
    on the frame type — see `stabsel._take_rows` — or, better, happen on the
    records before a frame is built, as `outlier_frame` does.

    The check reads attribute accesses out of the parsed syntax tree, so prose
    naming these methods (including this docstring) does not trip it.
    """
    import ast
    import pathlib

    banned = {"sort_values", "iloc", "to_pandas"}
    # `_frame.take_rows` is the one permitted use: it dispatches on frame type
    # and reaches `.iloc` only after ruling polars out. `stabsel._take_rows`
    # delegates to it.
    permitted = {("_frame.py", "take_rows")}

    package = pathlib.Path(mb.__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text())
        scopes = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    scopes[id(inner)] = node.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                if (path.name, scopes.get(id(node))) in permitted:
                    continue
                offenders.append(f"{path.name}:{node.lineno}: .{node.attr}")

    assert offenders == []
