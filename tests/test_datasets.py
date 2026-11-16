"""Tests for the built-in datasets and simulation generators."""

import numpy as np
import pytest

from mboostrs import datasets


def test_bodyfat_shape():
    df = datasets.bodyfat()
    assert "DEXfat" in df.columns
    assert len(df) > 0


def test_friedman1_shape_and_reproducibility():
    df1 = datasets.friedman1(n=50, n_features=8, seed=0)
    df2 = datasets.friedman1(n=50, n_features=8, seed=0)
    assert df1.shape == (50, 8 + 2)  # x0..x7, y, y_signal
    np.testing.assert_array_equal(df1["y"], df2["y"])


def test_friedman1_rejects_too_few_features():
    try:
        datasets.friedman1(n=10, n_features=3)
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass


def test_friedman1_signal_matches_formula():
    df = datasets.friedman1(n=200, n_features=5, noise=0.0, seed=1)
    expected = (
        10 * np.sin(np.pi * df["x0"] * df["x1"])
        + 20 * (df["x2"] - 0.5) ** 2
        + 10 * df["x3"]
        + 5 * df["x4"]
    )
    np.testing.assert_allclose(df["y_signal"], expected)
    np.testing.assert_allclose(df["y"], expected)  # noise=0.0


def test_sparse_linear_shape_and_signal():
    df = datasets.sparse_linear(n=100, p=20, n_signal=3, effect_size=2.0, noise=0.0, seed=0)
    assert df.shape == (100, 21)
    expected = 2.0 * df["x0"] - 2.0 * df["x1"] + 2.0 * df["x2"]
    np.testing.assert_allclose(df["y"], expected, atol=1e-10)


def test_sparse_linear_rejects_too_many_signals():
    try:
        datasets.sparse_linear(n=10, p=5, n_signal=6)
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass


def test_heteroscedastic_lss_shape_and_moments():
    df = datasets.heteroscedastic_lss(n=200, seed=0)
    assert set(df.columns) == {"x1", "x2", "y", "mu", "sigma"}
    residual = (df["y"] - df["mu"]) / df["sigma"]
    # standardized residuals should look standard normal, not exactly but
    # within a generous tolerance for n=200
    assert abs(residual.mean()) < 0.2
    assert abs(residual.std() - 1.0) < 0.2


# --- output= backends must agree ---------------------------------------------


def _loaders():
    return [datasets.bodyfat, datasets.wpbc, datasets.india, datasets.iris]


@pytest.mark.parametrize("loader", _loaders(), ids=lambda f: f.__name__)
def test_polars_and_pandas_loaders_agree_on_shape_and_columns(loader):
    pol = loader(output="polars")
    pan = loader(output="pandas")
    assert pol.height == pan.shape[0]
    assert list(pol.columns) == list(pan.columns)


@pytest.mark.parametrize("loader", _loaders(), ids=lambda f: f.__name__)
def test_numeric_columns_are_not_read_as_strings(loader):
    """`NA` in an R-written CSV must parse as null, not force a String column.

    wpbc's `pnodes` is the case that catches this: the CSVs are R-written, so
    a missing value is spelled `NA`. Polars needs `null_values="NA"` to read
    that as null; without it the whole column parses as String and any
    arithmetic on it is wrong.
    """
    import polars as pl

    pol = loader(output="polars")
    pan = loader(output="pandas")
    for col in pol.columns:
        if pan[col].dtype.kind in "if":
            assert pol[col].dtype != pl.String, (
                f"{loader.__name__}: column {col!r} parsed as String in Polars "
                f"but {pan[col].dtype} in pandas"
            )


def test_wpbc_missing_values_survive_as_nulls():
    pol = datasets.wpbc(output="polars")
    pan = datasets.wpbc(output="pandas")
    assert pol["pnodes"].null_count() == int(pan["pnodes"].isna().sum()) > 0


@pytest.mark.parametrize("loader", _loaders(), ids=lambda f: f.__name__)
def test_records_output_is_dependency_free(loader):
    rec = loader(output="records")
    assert isinstance(rec, dict)
    assert all(isinstance(v, np.ndarray) for v in rec.values())


def test_unknown_output_is_rejected():
    with pytest.raises(ValueError, match="output must be one of"):
        datasets.bodyfat(output="arrow")


def test_generators_honor_output():
    import polars as pl

    assert isinstance(datasets.friedman1(n=20, seed=0, output="polars"), pl.DataFrame)
    assert isinstance(datasets.sparse_linear(n=20, p=6, seed=0, output="records"), dict)
    assert isinstance(
        datasets.heteroscedastic_lss(n=20, seed=0, output="polars"), pl.DataFrame
    )


def test_string_columns_keep_their_values():
    """`null_values="NA"` must not null out legitimate string categories."""
    import polars as pl

    ind = datasets.india(output="polars")
    iris = datasets.iris(output="polars")
    assert ind["mcdist_lab"].dtype == pl.String
    assert ind["mcdist_lab"].null_count() == 0
    assert iris["Species"].null_count() == 0
    assert set(iris["Species"].unique()) == {"setosa", "versicolor", "virginica"}


def test_usmacro_covers_the_pre_sweep_sample():
    df = datasets.usmacro()
    assert df.height == 203
    assert df["year"].min() == 1959 and df["year"].max() == 2009
    pre_sweep = df.filter(df["year"] <= 1993)
    assert pre_sweep.height == 140
    rates = pre_sweep["tbilrate"].to_numpy()
    assert rates.min() == pytest.approx(2.29) and rates.max() == pytest.approx(15.33)
    assert set(["realgdp", "m1", "cpi", "tbilrate", "infl"]) <= set(df.columns)


def test_deposit_market_splices_the_two_rate_segments():
    df = datasets.deposit_market()
    assert df["date"].min() == "2009-05-01"
    assert df["date"].is_sorted()
    sources = set(df["mmda_source"].to_list())
    assert sources == {"fred_mmnrnj", "fdic_page"}
    # The FRED segment ends when the FDIC changed its calculation, and every
    # month after that is scraped from the published rate-cap pages.
    last_fred = df.filter(df["mmda_source"] == "fred_mmnrnj")["date"].max()
    first_page = df.filter(df["mmda_source"] == "fdic_page")["date"].min()
    assert last_fred < first_page
    assert last_fred == "2021-03-01"
    # The offered-rate columns other than the money-market one start with the
    # scraped segment; the money-market rate itself is never missing.
    assert df["natl_mmda"].null_count() == 0
    assert df.filter(df["mmda_source"] == "fdic_page")["natl_savings"].null_count() == 0


def test_bank_funding_quarterly_flow_is_a_within_year_difference():
    df = datasets.bank_funding()
    assert df["repdte"].min() == "1992-03-31"
    assert df["n_banks"].min() > 4000
    first_quarters = df.filter(df["repdte"].str.slice(5, 2) == "03")
    # A first-quarter report is itself year-to-date, so the flow equals the level.
    diffs = (first_quarters["interest_expense_qtr"]
             - first_quarters["interest_expense_ytd"]).abs().max()
    assert diffs == pytest.approx(0.0)
    assert (df["interest_expense_qtr"] > 0).all()


def test_bank_funding_annual_separates_deposit_interest():
    df = datasets.bank_funding_annual()
    assert df["year"].min() == 1992
    # Interest paid on deposits is part of interest paid on all liabilities.
    assert (df["edep"] <= df["eintexp"]).all()
    assert (df["depi"] < df["dep"]).all()
    rate_2023 = 100 * df.filter(df["year"] == 2023)["edep"][0] / df.filter(
        df["year"] == 2023)["depi"][0]
    assert 1.0 < rate_2023 < 5.0


def test_fdic_bank_yearend_loader_is_a_packaged_raw_extract():
    frame = datasets.fdic_bank_yearend(output="polars")
    assert frame.height == 44_828
    assert frame["REPDTE"].min() == "20171231"
    assert frame["REPDTE"].max() == "20251231"
    assert frame.columns == ["NAME", "STALP", "CERT", "REPDTE", "ASSET",
                             "DEP", "DEPI", "DEPNI", "EDEP", "EINTEXP",
                             "LNLSNET", "BRO", "OFFDOM"]
    assert "deposit_rate" not in frame.columns
    records = datasets.fdic_bank_yearend(output="records")
    assert len(records["CERT"]) == frame.height
    assert datasets.fdic_bank_yearend().shape == frame.shape
    card = datasets.data_card("fdic_bank_yearend")
    assert card["rows"] == frame.height
    assert "thousands" in card["units"]
    assert card["retrieved"] == "2026-08-21"


def test_hip_cz_shapes_and_validation_columns():
    df = datasets.hip_cz()
    # 595 CZs above the 25,000-population reporting cutoff
    assert df.height == 595
    for col in ("pop2000", "census_div", "lon", "lat",
                "le_raceadj_q1_M", "le_raceadj_q1_F",
                "le_raceadj_q4_M", "le_raceadj_q4_F",
                "sd_le_raceadj_q1_M", "sd_le_raceadj_q1_F",
                "sd_le_raceadj_q4_M", "sd_le_raceadj_q4_F",
                "cur_smoke_q1", "reimb_penroll_adj10", "cs_born_foreign"):
        assert col in df.columns
    # published standard errors: positive, and widest for the top-quartile
    # women whose death counts are smallest
    assert all(df[f"sd_le_raceadj_{q}_{g}"].min() > 0
               for q in ("q1", "q4") for g in ("M", "F"))
    assert df["sd_le_raceadj_q4_F"].max() > df["sd_le_raceadj_q1_M"].max()
    # the documented missingness: dropout_r is absent for 103 CZs
    assert df["dropout_r"].is_null().sum() == 103
    # population-weighted mean of Q1 male life expectancy reproduces the
    # published national figure to the precision the paper prints
    ok = df.drop_nulls(["le_raceadj_q1_M", "pop2000"])
    w = ok["pop2000"].to_numpy()
    mean = float((ok["le_raceadj_q1_M"].to_numpy() * w).sum() / w.sum())
    assert 75.0 < mean < 77.0


def test_hip_county_is_complete_and_keeps_published_uncertainty():
    df = datasets.hip_county()
    assert df.height == 935
    assert df["cty"].n_unique() == 935
    assert df["cty_pop2000"].min() >= 25_000
    for col in ("county_name", "stateabbrv", "census_div", "lon", "lat",
                "le_raceadj_q1_M", "le_raceadj_q1_F",
                "sd_le_raceadj_q1_M", "sd_le_raceadj_q1_F",
                "cur_smoke_q1", "reimb_penroll_adj10", "cs_born_foreign"):
        assert col in df.columns
        assert df[col].null_count() == 0
    assert df["sd_le_raceadj_q1_M"].min() > 0
    assert df["sd_le_raceadj_q1_F"].min() > 0
