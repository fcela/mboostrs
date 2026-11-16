"""Provenance records: `mboostrs.provenance()` and the entry it adds to to_dict().

The data fingerprint is a SHA-256 over a fixed byte layout of the columns, so
it must match for equal content in any container, repeat across calls, and
change when a value, a name or a dtype changes.
"""

import json
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mboostrs as mb


def _table():
    rng = np.random.default_rng(7)
    n = 120
    x0 = rng.uniform(-1, 1, n)
    x1 = rng.uniform(-1, 1, n)
    return {
        "y": np.sin(2 * x0) + rng.normal(scale=0.3, size=n),
        "x0": x0,
        "x1": x1,
        "flag": rng.integers(0, 2, n).astype(bool),
        "count": rng.integers(0, 9, n),
        "group": np.array(["a", "b", "c"] * (n // 3) + ["a"] * (n % 3)),
    }


def _sha(data):
    return mb.provenance(data)["data"]["sha256"]


def test_fingerprint_is_the_same_on_two_calls():
    table = _table()
    first = mb.provenance(table, seed=5)
    second = mb.provenance(table, seed=5)
    assert first == second


def test_polars_pandas_and_dict_with_equal_content_hash_alike():
    table = _table()
    digest = _sha(table)
    assert _sha(pl.DataFrame(table)) == digest
    assert _sha(pd.DataFrame(table)) == digest


def test_text_missing_values_hash_alike_across_containers():
    table = {"y": np.array([1.0, 2.0, 3.0]), "label": ["u", None, "v"]}
    digest = _sha(table)
    assert _sha(pl.DataFrame(table)) == digest
    assert _sha(pd.DataFrame({"y": table["y"], "label": pd.Series(["u", None, "v"],
                                                                  dtype=object)})) == digest


def test_a_changed_value_changes_the_hash():
    table = _table()
    changed = {key: np.array(value, copy=True) for key, value in table.items()}
    changed["x1"][17] += 1e-9
    assert _sha(changed) != _sha(table)


def test_a_changed_text_value_changes_the_hash():
    table = _table()
    changed = dict(table)
    changed["group"] = np.where(np.arange(len(table["group"])) == 4, "z", table["group"])
    assert _sha(changed) != _sha(table)


def test_a_renamed_column_changes_the_hash():
    table = _table()
    renamed = dict(table)
    renamed["x2"] = renamed.pop("x1")
    assert _sha(renamed) != _sha(table)


def test_a_changed_dtype_changes_the_hash():
    table = _table()
    widened = dict(table)
    widened["count"] = table["count"].astype(np.int32)
    assert _sha(widened) != _sha(table)


def test_data_entry_reports_rows_and_dtypes():
    table = _table()
    data = mb.provenance(pl.DataFrame(table))["data"]
    assert data["n_rows"] == 120
    assert [c["name"] for c in data["columns"]] == list(table)
    dtypes = {c["name"]: c["dtype"] for c in data["columns"]}
    assert dtypes == {"y": "float64", "x0": "float64", "x1": "float64",
                      "flag": "bool", "count": "int64", "group": "string"}


def test_record_has_library_and_environment_and_no_timestamp():
    record = mb.provenance()
    assert record["mboostrs_version"] == mb.__version__
    assert record["build"]["build_id"] == mb.build_info()["build_id"]
    assert record["python_version"]
    assert record["platform"]
    assert record["packages"]["numpy"] and record["packages"]["polars"]
    assert "data" not in record and "seed" not in record
    assert not any("time" in key or "date" in key for key in record)
    json.dumps(record)


def test_seed_is_recorded_when_given():
    assert mb.provenance(seed=11)["seed"] == 11
    assert "seed" not in mb.provenance()


def test_pandas_version_is_recorded_because_pandas_is_installed():
    assert mb.provenance()["packages"]["pandas"] == pd.__version__


def test_data_must_be_a_table():
    with pytest.raises(TypeError, match="Polars frame, a pandas frame"):
        mb.provenance(data=[1, 2, 3])


def test_gamboost_to_dict_carries_provenance_of_its_training_data():
    table = _table()
    fit = mb.gamboost("y ~ bbs(x0) + bbs(x1)", data=table,
                      control=mb.boost_control(40, 0.1))
    record = fit.to_dict()
    json.dumps(record)
    assert record["provenance"]["data"] == mb.provenance(table)["data"]
    assert record["provenance"]["mboostrs_version"] == mb.__version__


def test_glmboost_to_dict_carries_provenance_of_its_training_data():
    table = pl.DataFrame(_table())
    fit = mb.glmboost("y ~ x0 + x1", data=table, control=mb.boost_control(40, 0.1))
    record = fit.to_dict()
    assert record["provenance"]["data"] == mb.provenance(table)["data"]


def test_lss_to_dict_carries_provenance_of_its_training_data():
    table = _table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x0)", "sigma": "y ~ bols(x1)"}, data=table,
                             families=mb.GaussianLSS(), control=mb.boost_control(30, 0.1))
    record = fit.to_dict()
    json.dumps(record)
    assert record["provenance"]["data"] == mb.provenance(table)["data"]


def test_gam_refit_to_dict_carries_provenance_of_its_training_data():
    table = _table()
    fit = mb.gamboost("y ~ bbs(x0) + bbs(x1)", data=table,
                      control=mb.boost_control(40, 0.1))
    refit = mb.gam_refit(fit)
    record = refit.to_dict()
    assert record["provenance"]["data"] == mb.provenance(table)["data"]


def test_gamlss_refit_to_dict_carries_provenance_of_its_training_data():
    table = _table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS({"mu": "y ~ bbs(x0)", "sigma": "y ~ bols(x1)"}, data=table,
                             families=mb.GaussianLSS(), control=mb.boost_control(30, 0.1))
        refit = mb.gamlss_refit(fit, which="all")
    record = refit.to_dict()
    assert record["provenance"]["data"] == mb.provenance(table)["data"]


def test_conformal_interval_to_dict_carries_provenance_without_a_data_fingerprint():
    table = _table()
    fit = mb.glmboost("y ~ x0 + x1", data=table, control=mb.boost_control(40, 0.1))
    newdata = {"x0": np.linspace(-1, 1, 10), "x1": np.zeros(10)}
    interval = mb.conformal_interval(fit, newdata=newdata, level=0.8, seed=2)
    record = interval.to_dict()
    json.dumps(record)
    assert record["provenance"]["mboostrs_version"] == mb.__version__
    assert "data" not in record["provenance"]
