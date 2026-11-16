"""compare_models: sorting, refusal of risks on different scales, complexity,
timing and cross-validated rows (issue #294)."""

import json

import numpy as np
import pytest

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
def gaussian_full(data):
    return mb.glmboost("y ~ x1 + x2", data=data, family=mb.Gaussian(),
                       control=mb.boost_control(30, 0.1))


@pytest.fixture
def gaussian_one(data):
    return mb.glmboost("y ~ x1", data=data, family=mb.Gaussian(),
                       control=mb.boost_control(30, 0.1))


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
    return {"y": y, "x": x}


class TestSort:
    def test_risk_sorts_smallest_first(self, gaussian_full, gaussian_one):
        rows = mb.compare_models(gaussian_one, gaussian_full,
                                 metrics=("risk",), sort="risk")
        risks = [r["risk"] for r in rows]
        assert risks == sorted(risks)

    def test_log_likelihood_sorts_largest_first(self, gaussian_full, gaussian_one):
        rows = mb.compare_models(gaussian_one, gaussian_full,
                                 metrics=("logLik",), sort="logLik")
        values = [r["logLik"] for r in rows]
        assert values == sorted(values, reverse=True)

    def test_unset_sort_keeps_the_given_order(self, gaussian_full, gaussian_one):
        rows = mb.compare_models(gaussian_one, gaussian_full, metrics=("risk",),
                                 labels=["one", "full"])
        assert [r["label"] for r in rows] == ["one", "full"]

    def test_missing_values_go_last(self):
        from mboostrs.model_selection import _sort_comparison_rows

        rows = [
            {"label": "none", "risk": None},
            {"label": "high", "risk": 2.0},
            {"label": "nan", "risk": float("nan")},
            {"label": "low", "risk": 1.0},
        ]
        ordered = _sort_comparison_rows(rows, "risk", None)
        assert [r["label"] for r in ordered] == ["low", "high", "none", "nan"]

    def test_unknown_column_is_refused(self, gaussian_full):
        with pytest.raises(ValueError, match="not a column"):
            mb.compare_models(gaussian_full, metrics=("risk",), sort="GAIC")

    def test_gaic_sort_refuses_mixed_criteria(self):
        from mboostrs.model_selection import _sort_comparison_rows

        # Rows that use different criteria cannot be ordered by GAIC.

        rows = [
            {"label": "a", "GAIC": 1.0, "criterion": "mboost_corrected"},
            {"label": "b", "GAIC": 2.0, "criterion": "classical"},
        ]
        with pytest.raises(ValueError, match="different criteria"):
            _sort_comparison_rows(rows, "GAIC", None)


class TestRiskScale:
    def test_same_family_and_settings_keep_risk(self, gaussian_full, gaussian_one):
        rows = mb.compare_models(gaussian_full, gaussian_one, metrics=("risk",))
        assert all(r["risk"] is not None for r in rows)
        assert all("risk_unavailable" not in r for r in rows)

    def test_gaussian_and_binomial_withhold_risk(self, binary_data):
        gaussian = mb.glmboost("y ~ x", data=binary_data, family=mb.Gaussian(),
                               control=mb.boost_control(30, 0.1))
        binomial = mb.glmboost("y ~ x", data=binary_data, family=mb.Binomial(),
                               control=mb.boost_control(30, 0.1))
        rows = mb.compare_models(gaussian, binomial, metrics=("risk", "mstop"))
        assert [r["risk"] for r in rows] == [None, None]
        assert all("Gaussian" in r["risk_unavailable"] or "Binomial" in r["risk_unavailable"]
                   for r in rows)
        assert all(r["mstop"] is not None for r in rows)
        json.dumps(rows)

    def test_different_quantile_levels_withhold_risk(self, data):
        q10 = mb.glmboost("y ~ x1", data=data, family=mb.Quantile(tau=0.1),
                          control=mb.boost_control(20, 0.1))
        q50 = mb.glmboost("y ~ x1", data=data, family=mb.Quantile(tau=0.5),
                          control=mb.boost_control(20, 0.1))
        rows = mb.compare_models(q10, q50, metrics=("risk",))
        assert [r["risk"] for r in rows] == [None, None]

    def test_same_quantile_level_keeps_risk(self, data):
        a = mb.glmboost("y ~ x1", data=data, family=mb.Quantile(tau=0.3),
                        control=mb.boost_control(20, 0.1))
        b = mb.glmboost("y ~ x1 + x2", data=data, family=mb.Quantile(tau=0.3),
                        control=mb.boost_control(20, 0.1))
        rows = mb.compare_models(a, b, metrics=("risk",))
        assert all(r["risk"] is not None for r in rows)

    def test_sorting_by_withheld_risk_is_refused(self, binary_data):
        gaussian = mb.glmboost("y ~ x", data=binary_data, family=mb.Gaussian(),
                               control=mb.boost_control(10, 0.1))
        binomial = mb.glmboost("y ~ x", data=binary_data, family=mb.Binomial(),
                               control=mb.boost_control(10, 0.1))
        with pytest.raises(ValueError, match="cannot sort by risk"):
            mb.compare_models(gaussian, binomial, metrics=("risk",), sort="risk")


class TestComplexityAndTiming:
    def test_selected_learners_counts_distinct_learners(self, gaussian_full, gaussian_one):
        rows = mb.compare_models(gaussian_full, gaussian_one,
                                 metrics=("selected_learners",))
        for row, model in zip(rows, (gaussian_full, gaussian_one), strict=True):
            assert row["selected_learners"] == len(set(int(s) for s in model._selected))
        assert rows[1]["selected_learners"] == 1

    def test_default_metrics_include_complexity(self, gaussian_full):
        (row,) = mb.compare_models(gaussian_full)
        assert row["selected_learners"] == len(set(int(s) for s in gaussian_full._selected))


class TestCrossValidatedRows:
    def test_cv_columns_report_the_cv_optimum(self, data):
        model = mb.glmboost("y ~ x1 + x2", data=data, family=mb.Gaussian(),
                            control=mb.boost_control(40, 0.1))
        folds = mb.cv_weights(len(data["y"]), type="kfold", B=4, seed=0)
        cv = model.cvrisk(folds=folds)
        (row,) = mb.compare_models(model, metrics=("mstop",), cv=[cv])
        aggregate = np.asarray(cv.aggregate, dtype=float)
        grid = np.asarray(cv.grid)
        best = int(np.argmin(np.where(grid >= 1, aggregate, np.inf)))
        assert row["cv_mstop"] == int(grid[best])
        assert row["cv_risk"] == pytest.approx(aggregate[best])

    def test_cv_length_must_match_models(self, gaussian_full):
        with pytest.raises(ValueError, match="one entry per model"):
            mb.compare_models(gaussian_full, cv=[])

    def test_cv_results_on_different_folds_are_refused(self, data):
        model = mb.glmboost("y ~ x1", data=data, family=mb.Gaussian(),
                            control=mb.boost_control(10, 0.1))
        a = model.cvrisk(folds=mb.cv_weights(len(data["y"]), type="kfold", B=4, seed=0))
        b = model.cvrisk(folds=mb.cv_weights(len(data["y"]), type="kfold", B=4, seed=1))
        with pytest.raises(ValueError, match="same folds"):
            mb.compare_models(model, model, cv=[a, b])

    def test_cv_result_without_fingerprint_is_refused(self, data):
        model = mb.glmboost("y ~ x1", data=data, family=mb.Gaussian(),
                            control=mb.boost_control(10, 0.1))
        cv = model.cvrisk(folds=mb.cv_weights(len(data["y"]), type="kfold", B=4, seed=0))
        cv.fold_fingerprint = ""
        with pytest.raises(ValueError, match="same folds"):
            mb.compare_models(model, cv=[cv])


class TestSummaryTableValues:
    def test_as_dict_holds_numbers_not_strings(self, data):
        model = mb.glmboost("y ~ x1 + x2", data=data, family=mb.Gaussian(),
                            control=mb.boost_control(20, 0.1))
        info = model.summary().as_dict()
        numeric = [cell for table in info["tables"] for row in table["rows"]
                   for cell in row[1:] if isinstance(cell, (int, float))
                   and not isinstance(cell, bool)]
        assert numeric, "summary tables should expose numeric cells"
        json.dumps(info)
