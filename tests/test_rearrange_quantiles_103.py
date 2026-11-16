"""rearrange_quantiles: sorting separately fitted quantile/expectile levels (#103)."""

import numpy as np
import polars as pl
import pytest

import mboostrs as mb


def _data(n=300, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 2 * x + (0.05 + np.abs(x)) * rng.standard_normal(n)
    return {"x": x, "y": y}


def _fits(family, taus=(0.1, 0.5, 0.9), mstop=3):
    d = _data()
    # few boosting steps leave the levels close together, so some rows cross
    fits = [
        mb.glmboost(
            "y ~ bols(x)", data=d, family=family(tau=t),
            control=mb.boost_control(mstop=mstop, nu=0.1),
        )
        for t in taus
    ]
    return fits, d


def _hand_count(unsorted):
    m = unsorted.to_numpy()
    cross = m[:, :-1] > m[:, 1:]
    return cross.any(axis=1).sum(), cross.sum(axis=0), (m[:, :-1] - m[:, 1:]).max()


def test_quantile_sorted_monotone_and_report_matches_hand_count():
    fits, d = _fits(mb.Quantile)
    r = mb.rearrange_quantiles(fits, d)
    assert r.kind == "quantile" and r.levels == (0.1, 0.5, 0.9)
    assert r.predictions.columns == ["q10", "q50", "q90"]
    assert isinstance(r.predictions, pl.DataFrame)
    sorted_m = r.predictions.to_numpy()
    assert (np.diff(sorted_m, axis=1) >= 0).all()
    rows, pairs, biggest = _hand_count(r.unsorted)
    assert rows > 0 and pairs.sum() > 0 and biggest > 0
    assert r.crossing_rows == rows and r.n == 300
    assert r.crossing_share == pytest.approx(rows / 300)
    assert r.pair_crossings["n_crossing"].to_list() == pairs.tolist()
    assert r.max_crossing == pytest.approx(max(biggest, 0.0), abs=1e-12)
    # sorted rows are the row-wise sort of the unsorted predictions
    assert np.array_equal(sorted_m, np.sort(r.unsorted.to_numpy(), axis=1))
    # the unsorted columns equal each model's own prediction
    for j, f in enumerate(fits):
        assert np.allclose(r.unsorted.to_numpy()[:, j], f.predict(d), atol=1e-14)


def test_some_rows_cross_in_the_fixture_whatever_the_model_order():
    fits, d = _fits(mb.Quantile)
    r = mb.rearrange_quantiles([fits[2], fits[1], fits[0]], d)
    assert r.crossing_rows == mb.rearrange_quantiles(fits, d).crossing_rows
    assert r.levels == (0.1, 0.5, 0.9)
    assert r.crossing_rows > 0


def test_expectile_and_dict_input():
    fits, d = _fits(mb.Expectile)
    r = mb.rearrange_quantiles({0.1: fits[0], 0.5: fits[1], 0.9: fits[2]}, d)
    assert r.kind == "expectile"
    assert r.predictions.columns == ["e10", "e50", "e90"]
    assert (np.diff(r.predictions.to_numpy(), axis=1) >= 0).all()
    assert r.crossing_rows > 0


def test_records_output_and_levels_check():
    fits, d = _fits(mb.Quantile, mstop=300)
    r = mb.rearrange_quantiles(fits, d, levels=[0.1, 0.5, 0.9], output="records")
    assert isinstance(r.predictions, list)
    assert set(r.predictions[0]) == {"q10", "q50", "q90"}


def test_error_cases():
    q, d = _fits(mb.Quantile)
    e, _ = _fits(mb.Expectile)
    with pytest.raises(mb.MboostrsValueError, match="mix Quantile and Expectile"):
        mb.rearrange_quantiles([q[0], e[1]], d)
    with pytest.raises(mb.MboostrsValueError, match="share a tau"):
        mb.rearrange_quantiles([q[0], q[0]], d)
    with pytest.raises(mb.MboostrsValueError, match="at least two"):
        mb.rearrange_quantiles([q[0]], d)
    with pytest.raises(mb.MboostrsValueError, match="do not match"):
        mb.rearrange_quantiles(q, d, levels=[0.2, 0.5, 0.9])
    g = mb.glmboost("y ~ bols(x)", data=d, family=mb.Gaussian(),
                    control=mb.boost_control(mstop=3))
    with pytest.raises(mb.MboostrsValueError, match="Gaussian"):
        mb.rearrange_quantiles([q[0], g], d)


def test_core_refuses_nan():
    from mboostrs import _rust

    with pytest.raises(Exception, match="NaN"):
        _rust.rearrange_levels(np.array([[1.0, np.nan]]))


@pytest.mark.r_parity
def test_r_function_matches_python_to_1e_12(tmp_path):
    """The R function, run on the same data, returns the same sorted levels."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript not available")
    fits, d = _fits(mb.Quantile)
    py = mb.rearrange_quantiles(fits, d)
    src = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    pl.DataFrame(d).write_csv(src)
    env = dict(os.environ, MBOOSTRS_REARRANGE_IN=str(src), MBOOSTRS_REARRANGE_OUT=str(out))
    script = Path(__file__).resolve().parents[1] / "mboostrs" / "tests" / "test_rearrange_quantiles.R"
    done = subprocess.run([rscript, str(script)], env=env, capture_output=True, text=True)
    if "there is no package called" in done.stderr:
        pytest.skip("R package mboostrs is not installed")
    assert done.returncode == 0, done.stderr
    r_out = pl.read_csv(out)
    for name in py.predictions.columns:
        assert np.max(np.abs(r_out[name].to_numpy() - py.predictions[name].to_numpy())) < 1e-12


def test_to_dict_is_json_safe_and_gates_the_arrays():
    import json

    fits, d = _fits(mb.Quantile)
    r = mb.rearrange_quantiles(fits, d)
    small = r.to_dict()
    assert "predictions" not in small and small["crossing_rows"] == r.crossing_rows
    full = r.to_dict(include_arrays=True)
    assert set(full["predictions"]) == {"q10", "q50", "q90"}
    json.dumps(full)
