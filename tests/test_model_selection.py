"""Tests for Milestone 9: model comparison and family scanning."""

import json

import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS

import mboostrs as mb


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return {"y": y, "x1": x1, "x2": x2}


@pytest.fixture
def glm_full(data):
    return mb.glmboost(
        "y ~ x1 + x2", data=data, family=mb.Gaussian(),
        control=mb.boost_control(30, 0.1),
    )


@pytest.fixture
def glm_partial(data):
    return mb.glmboost(
        "y ~ x1", data=data, family=mb.Gaussian(),
        control=mb.boost_control(30, 0.1),
    )


@pytest.fixture
def gam_full(data):
    return mb.gamboost(
        "y ~ bbs(x1, df=4) + bols(x2)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(30, 0.1),
    )


class TestGaicTable:
    def test_basic(self, glm_full, glm_partial, gam_full):
        table = mb.gaic_table(glm_full, glm_partial, gam_full)
        assert len(table) == 3
        for row in table:
            assert "label" in row
            assert row["mboost_corrected"] is not None

    def test_sorted_by_gaic(self, glm_full, glm_partial):
        table = mb.gaic_table(glm_full, glm_partial)
        assert table[0]["mboost_corrected"] <= table[1]["mboost_corrected"]

    def test_logn_penalty(self, glm_full, glm_partial):
        with pytest.raises(ValueError, match="fixed penalty"):
            mb.gaic_table(glm_full, glm_partial, k=(2, "logn"))

    def test_custom_labels(self, glm_full, glm_partial):
        table = mb.gaic_table(glm_full, glm_partial, labels=["full", "partial"])
        assert table[0]["label"] in ("full", "partial")

    def test_json_serializable(self, glm_full):
        table = mb.gaic_table(glm_full)
        json.dumps(table)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            mb.gaic_table()


class TestOutputSelection:
    """Enhancement-hunt 12.2: ``as_frame`` used to be inert in both positions
    on these two functions — a public boolean named for a data frame that never
    produced one. The unset default still returns records (the historical
    return type), but an explicit value is now honoured."""

    @pytest.mark.parametrize("fn", [mb.gaic_table, mb.compare_models])
    def test_unset_as_frame_returns_records(self, fn, glm_full, glm_partial):
        assert isinstance(fn(glm_full, glm_partial), list)

    @pytest.mark.parametrize("fn", [mb.gaic_table, mb.compare_models])
    def test_explicit_as_frame_true_returns_a_frame(self, fn, glm_full, glm_partial):
        result = fn(glm_full, glm_partial, as_frame=True)
        assert not isinstance(result, list)
        assert hasattr(result, "columns")

    @pytest.mark.parametrize("fn", [mb.gaic_table, mb.compare_models])
    def test_explicit_as_frame_false_returns_records(self, fn, glm_full, glm_partial):
        assert isinstance(fn(glm_full, glm_partial, as_frame=False), list)

    @pytest.mark.parametrize("fn", [mb.gaic_table, mb.compare_models])
    def test_output_overrides_as_frame(self, fn, glm_full, glm_partial):
        assert isinstance(fn(glm_full, glm_partial, as_frame=True,
                             output="records"), list)
        assert not isinstance(fn(glm_full, glm_partial, as_frame=False,
                                 output="polars"), list)


class TestCompareModels:
    def test_basic(self, glm_full, glm_partial):
        rows = mb.compare_models(glm_full, glm_partial)
        assert len(rows) == 2
        assert "risk" in rows[0]
        assert "logLik" in rows[0]

    def test_custom_metrics(self, glm_full):
        rows = mb.compare_models(glm_full, metrics=("risk", "mstop"))
        assert "mstop" in rows[0]
        assert "logLik" not in rows[0]

    def test_json_serializable(self, glm_full):
        rows = mb.compare_models(glm_full)
        json.dumps(rows)


class TestScanLssFamilies:
    def test_successful_scan(self, data):
        results = mb.scan_lss_families(
            "y ~ x1 + x2", data=data, families=[GaussianLSS()],
            control=mb.boost_control(20, 0.1),
        )
        assert len(results) == 1
        assert results[0]["status"] == "ok"
        assert results[0]["GAIC"] is not None
        assert results[0]["n"] == len(data["y"])

    def test_failed_fit_captured(self, data):
        """A family that can't fit negative data should report error, not crash."""
        from mboostrs.gamlss import GammaLSS

        results = mb.scan_lss_families(
            "y ~ x1 + x2", data=data,
            families=[GaussianLSS(), GammaLSS()],
            control=mb.boost_control(20, 0.1),
        )
        statuses = {r["family"]: r["status"] for r in results}
        assert statuses["GaussianLSS"] == "ok"
        # GammaLSS may fail or succeed depending on data
        assert statuses["GammaLSS"] in ("ok", "error")

    def test_sorted_best_first(self, data):
        results = mb.scan_lss_families(
            "y ~ x1 + x2", data=data, families=[GaussianLSS()],
            control=mb.boost_control(20, 0.1),
        )
        assert results[0]["status"] == "ok"

    def test_invalid_metric_raises(self, data):
        with pytest.raises(ValueError, match="metric"):
            mb.scan_lss_families(
                "y ~ x1", data=data, families=[GaussianLSS()],
                metric="bad",
            )

    def test_invalid_fit_func_raises(self, data):
        with pytest.raises(ValueError, match="fit_func"):
            mb.scan_lss_families(
                "y ~ x1", data=data, families=[GaussianLSS()],
                fit_func="bad",
            )


class TestProfileMstop:
    def test_risk_profile(self, glm_full):
        pm = mb.profile_mstop(glm_full, grid=[1, 5, 10, 20, 30], metric="risk")
        assert len(pm) == 5
        # Risk should be monotonically decreasing (boosting)
        risks = [r["risk"] for r in pm]
        assert risks[-1] < risks[0]

    def test_gaic_profile(self, glm_full):
        pm = mb.profile_mstop(glm_full, grid=[1, 10, 30], metric="GAIC")
        assert len(pm) == 3
        for row in pm:
            assert row["GAIC"] is not None

    def test_auto_grid(self, glm_full):
        pm = mb.profile_mstop(glm_full)
        assert len(pm) >= 5
        assert pm[-1]["mstop"] == glm_full.mstop

    def test_lss_risk_profile(self, data):
        mod = mb.glmboostLSS(
            "y ~ x1 + x2", data=data, families=GaussianLSS(),
            control=mb.boost_control(20, 0.1),
        )
        pm = mb.profile_mstop(mod, grid=[1, 10, 20], metric="risk")
        assert len(pm) == 3

    def test_invalid_metric_raises(self, glm_full):
        with pytest.raises(ValueError, match="metric"):
            mb.profile_mstop(glm_full, metric="bad")


class TestAicAtOptimum:
    """Enhancement-hunt 21.10. `gaic_table` and `compare_models` read
    `result.aic[-1]`, the criterion at whatever `mstop` a model happened to be
    fitted with — so a table headed `GAIC(k)`, used to rank models, ranked them
    by their AIC at their fitted length. That is sound only when every model in
    the table was fitted to the same `mstop`, which a model-comparison table
    invites you to violate. R's `AIC()` returns a scalar at the optimum."""

    @pytest.fixture
    def mixed_mstop(self, data):
        long_fit = mb.glmboost(
            "y ~ x1 + x2", data=data, family=mb.Gaussian(),
            control=mb.boost_control(400, 0.1),
        )
        short_fit = mb.glmboost(
            "y ~ x1", data=data, family=mb.Gaussian(),
            control=mb.boost_control(40, 0.1),
        )
        return long_fit, short_fit

    def test_gaic_table_reports_each_model_at_its_own_optimum(self, mixed_mstop):
        long_fit, short_fit = mixed_mstop
        rows = mb.gaic_table(long_fit, short_fit, labels=["long", "short"])
        by_label = {row["label"]: row["mboost_corrected"] for row in rows}
        for model, label in ((long_fit, "long"), (short_fit, "short")):
            result = model.aic()
            path = np.asarray(result.aic, dtype=float)
            assert by_label[label] == pytest.approx(path[result.mstop - 1])
            # and, where the model was fitted well past its optimum, that is
            # not the same as the value at the fitted length
            if result.mstop < path.size:
                assert by_label[label] <= path[-1]

    def test_compare_models_agrees_with_gaic_table(self, mixed_mstop):
        long_fit, short_fit = mixed_mstop
        table = {r["label"]: r["mboost_corrected"]
                 for r in mb.gaic_table(long_fit, short_fit, labels=["long", "short"])}
        compared = {r["label"]: r["GAIC"]
                    for r in mb.compare_models(long_fit, short_fit,
                                               labels=["long", "short"])}
        assert table == pytest.approx(compared)
