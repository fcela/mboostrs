"""Python smoke tests for bmrf and bspatial learners (no R required)."""

import numpy as np
import pytest
from mboostrs._rust import fit_gamboost_specs as _fit_gamboost_specs
from mboostrs.families import Gaussian
from mboostrs.gamlss import learner_spec
from mboostrs.gamboost import _extract_column, gamboost
from mboostrs.learners import Bbs, Bmrf, Bspatial


def _make_simple_k(n_levels):
    """Build the graph Laplacian of a chain topology (1-2-...-n).

    K[i,i] is the number of neighbours of level i (1 at the two ends, 2 in
    the middle) and K[i,i+1] = K[i+1,i] = -1 for adjacent levels, so every
    row sums to zero. This is the format `bmrf` requires; the adjacency
    matrix this helper used to build is indefinite and is now rejected.
    """
    k = np.zeros((n_levels, n_levels), dtype=np.float64)
    for i in range(n_levels - 1):
        k[i, i + 1] = -1.0
        k[i + 1, i] = -1.0
        k[i, i] += 1.0
        k[i + 1, i + 1] += 1.0
    return k


def _fit_bmrf(y, w, levels, K, df=4.0, lam=-1.0, center=False, mstop=50, nu=0.1, by=None):
    """Fit one bmrf learner through the production learner-spec boundary."""
    K_arr = np.asarray(K, dtype=np.float64)
    data = {"region": np.asarray(levels, dtype=float)}
    by_name = None
    if by is not None:
        by_name = "__by"
        data[by_name] = np.asarray(by, dtype=float)
    learner = Bmrf(
        "region",
        K=K_arr,
        df=df,
        lambda_value=None if lam < 0 else lam,
        center=center,
        by=by_name,
    )
    return _fit_gamboost_specs(
        [learner_spec(learner, data)], y, w, mstop, nu, "gaussian", {},
        None, None, "inbag", False, None, 0,
    )


def _fit_bspatial(
    y, w, x1, x2, knots=20, degree=3, differences=2, df=6.0, lam=-1.0, center=False, mstop=50, nu=0.1, by=None
):
    """Fit one bspatial learner through the production spec boundary."""
    data = {"x1": np.asarray(x1, dtype=float), "x2": np.asarray(x2, dtype=float)}
    by_name = None
    if by is not None:
        by_name = "__by"
        data[by_name] = np.asarray(by, dtype=float)
    learner = Bspatial(
        "x1",
        "x2",
        knots=knots,
        degree=degree,
        differences=differences,
        df=df,
        lambda_value=None if lam < 0 else lam,
        center=center,
        by=by_name,
    )
    return _fit_gamboost_specs(
        [learner_spec(learner, data)], y, w, mstop, nu, "gaussian", {},
        None, None, "inbag", False, None, 0,
    )


# ── bmrf smoke tests ─────────────────────────────────────────────────


class TestBmrfSmoke:
    def test_bmrf_formula_raises_without_k(self):
        """Formula-parsed Bmrf without K raises ValueError."""
        rng = np.random.RandomState(42)
        n = 100
        n_levels = 4
        region = rng.randint(0, n_levels, n).astype(np.float64)
        y = rng.normal(0, 1, n)
        data = {"y": y, "region": region}

        with pytest.raises(ValueError, match="requires a neighborhood"):
            gamboost(
                "y ~ bmrf(region, df=4)",
                data=data,
                family=Gaussian(),
                control=(50, 0.1),
            )

    def test_bmrf_rejects_adjacency_matrix(self):
        """An adjacency matrix in place of the Laplacian is refused.

        This is the substitution worth guarding: same shape, same sparsity
        pattern, so nothing downstream would object — but the adjacency
        matrix is indefinite, so `beta' K beta` can be negative and the
        penalty rewards the divergence it exists to suppress. R's `mboost`
        applies the same `sum == 0` test.
        """
        n_levels = 5
        adjacency = np.zeros((n_levels, n_levels))
        for i in range(n_levels - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0

        assert np.linalg.eigvalsh(adjacency).min() < -1e-6, (
            "the premise of this test is that adjacency is indefinite"
        )

        with pytest.raises(ValueError, match="must be a graph Laplacian"):
            Bmrf("region", K=adjacency)

        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        assert np.linalg.eigvalsh(laplacian).min() > -1e-8
        Bmrf("region", K=laplacian)  # the converted form is accepted

    def test_bmrf_rejects_non_laplacian_shapes(self):
        """Square-but-wrong and non-square penalties are both refused."""
        with pytest.raises(ValueError, match="square"):
            Bmrf("region", K=np.ones((3, 4)))

        # A ridge penalty: valid for `buser`, but not a neighbourhood graph.
        with pytest.raises(ValueError, match="must be a graph Laplacian"):
            Bmrf("region", K=np.eye(4))

        asymmetric = np.array([[1.0, -1.0], [-0.5, 0.5]])
        with pytest.raises(ValueError, match="symmetric"):
            Bmrf("region", K=asymmetric)

    def test_bmrf_fit_with_k(self):
        """Full bmrf fit with explicit K matrix works end-to-end."""
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        y = np.array([3.0 * r for r in region]) + rng.normal(0, 1.0, n)
        K = _make_simple_k(n_levels)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        levels = region.astype(int).tolist()

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, _hat, _report, _residual = _fit_bmrf(y_arr, w, levels, K)

        assert len(fitted) == n
        assert not any(np.isnan(fitted))
        assert len(selected) == 50
        assert all(s == 0 for s in selected)

    def test_bmrf_predict_training(self):
        """bmrf predict on training data matches fitted values."""
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        y = np.array([3.0 * r for r in region]) + rng.normal(0, 1.0, n)
        K = _make_simple_k(n_levels)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        levels = region.astype(int).tolist()

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, store, _report, _residual = _fit_bmrf(y_arr, w, levels, K)
        pred = np.asarray(store.predict(offset, selected, coefs))

        np.testing.assert_allclose(pred, fitted, rtol=1e-10, err_msg="bmrf predict on training data != fitted")

    def test_bmrf_center_flag(self):
        """bmrf with center=True fits and drops null space."""
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        y = np.array([3.0 * r for r in region]) + rng.normal(0, 1.0, n)
        K = _make_simple_k(n_levels)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        levels = region.astype(int).tolist()

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, _hat, _report, _residual = _fit_bmrf(
            y_arr, w, levels, K, center=True
        )

        assert len(fitted) == n
        assert not any(np.isnan(fitted))

    def test_bmrf_with_by(self):
        """bmrf with by= varying-coefficient works end-to-end."""
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        z = rng.binomial(1, 0.5, n).astype(np.float64)
        y = np.array([3.0 * r for r in region]) * z + rng.normal(0, 1.0, n)
        K = _make_simple_k(n_levels)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        levels = region.astype(int).tolist()
        by_vals = z.tolist()

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, store, _report, _residual = _fit_bmrf(
            y_arr, w, levels, K, by=by_vals
        )

        assert len(fitted) == n
        assert not any(np.isnan(fitted))

        pred = np.asarray(store.predict(offset, selected, coefs))
        np.testing.assert_allclose(pred, fitted, rtol=1e-10, err_msg="bmrf by predict != fitted")

    def test_bmrf_different_df(self):
        """bmrf with different df values produces different results."""
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        y = np.array([3.0 * r for r in region]) + rng.normal(0, 1.0, n)
        K = _make_simple_k(n_levels)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        levels = region.astype(int).tolist()

        results = {}
        for df_val in [2.0, 4.0, 8.0]:
            _, _, _, _, fitted, _nuis, _nuis_path, _hat, _report, _residual = _fit_bmrf(y_arr, w, levels, K, df=df_val)
            results[df_val] = np.asarray(fitted)

        assert not np.allclose(results[2.0], results[4.0], rtol=1e-6)
        assert not np.allclose(results[4.0], results[8.0], rtol=1e-6)

    def test_bmrf_explicit_lambda(self):
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        y = np.array([3.0 * r for r in region]) + rng.normal(0, 1.0, n)
        K = _make_simple_k(n_levels)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        levels = region.astype(int).tolist()

        _, _, _, _, fitted1, _nuis, _nuis_path1, _hat1, _report1, _residual1 = _fit_bmrf(y_arr, w, levels, K, lam=0.1)
        _, _, _, _, fitted2, _nuis, _nuis_path2, _hat2, _report2, _residual2 = _fit_bmrf(y_arr, w, levels, K, lam=10.0)

        f1 = np.asarray(fitted1)
        f2 = np.asarray(fitted2)
        assert np.var(f1) > np.var(f2) + 0.01


# ── bspatial smoke tests ─────────────────────────────────────────────


class TestBspatialSmoke:
    def test_bspatial_fit_basic(self):
        """bspatial with default settings fits without error."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + rng.normal(0, 0.3, n)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, _hat, _report, _residual = _fit_bspatial(y_arr, w, x1, x2)

        assert len(fitted) == n
        assert not any(np.isnan(fitted))
        assert all(s == 0 for s in selected)

    def test_bspatial_predict_training(self):
        """bspatial predict on training data matches fitted values."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + rng.normal(0, 0.3, n)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, store, _report, _residual = _fit_bspatial(y_arr, w, x1, x2)
        pred = np.asarray(store.predict(offset, selected, coefs))

        np.testing.assert_allclose(pred, fitted, rtol=1e-10, err_msg="bspatial predict != fitted")

    def test_bspatial_center_flag(self):
        """bspatial with center=True fits without error."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + rng.normal(0, 0.3, n)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, _hat, _report, _residual = _fit_bspatial(
            y_arr, w, x1, x2, center=True
        )

        assert len(fitted) == n
        assert not any(np.isnan(fitted))

    def test_bspatial_with_by(self):
        """bspatial with by= varying-coefficient works end-to-end."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        z = rng.binomial(1, 0.5, n).astype(np.float64)
        y = np.sin(2 * np.pi * x1) * z + rng.normal(0, 0.3, n)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)
        by_vals = z.tolist()

        offset, selected, coefs, risk, fitted, _nuis, _nuis_path, store, _report, _residual = _fit_bspatial(
            y_arr, w, x1, x2, by=by_vals
        )

        assert len(fitted) == n
        assert not any(np.isnan(fitted))

        pred = np.asarray(store.predict(offset, selected, coefs))
        np.testing.assert_allclose(pred, fitted, rtol=1e-10)

    def test_bspatial_different_knots(self):
        """bspatial with different knot counts fits correctly."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + rng.normal(0, 0.3, n)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)

        for n_knots in [10, 20]:
            _, _, _, _, fitted, _nuis, _nuis_path, _hat, _report, _residual = _fit_bspatial(y_arr, w, x1, x2, knots=n_knots)
            assert len(fitted) == n
            assert not any(np.isnan(fitted))

    def test_bspatial_explicit_lambda(self):
        """bspatial with explicit lambda: large lambda => smoother."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + rng.normal(0, 0.3, n)

        y_arr = _extract_column({"y": y}, "y")
        w = np.ones(n, dtype=np.float64)

        _, _, _, _, f1, _nuis, _nuis_path1, _hat1, _report1, _residual1 = _fit_bspatial(y_arr, w, x1, x2, lam=0.1)
        _, _, _, _, f2, _nuis, _nuis_path2, _hat2, _report2, _residual2 = _fit_bspatial(y_arr, w, x1, x2, lam=100.0)

        f1_arr = np.asarray(f1)
        f2_arr = np.asarray(f2)
        assert np.var(f1_arr) > np.var(f2_arr) + 0.01

    def test_bspatial_mixed_with_bbs(self):
        """bspatial + bbs together in a gamboost model works."""
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        x3 = rng.normal(0, 1, n)
        y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + 0.5 * x3 + rng.normal(0, 0.3, n)

        model = gamboost(
            "y ~ x1 + x3",
            {"y": y, "x1": x1, "x2": x2, "x3": x3},
            family=Gaussian(),
            control=(100, 0.1),
            learners=[Bspatial("x1", "x2", df=6), Bbs("x3", df=4)],
        )

        assert len(model.fitted_) == n
        assert not np.any(np.isnan(model.fitted_))
        assert set(model._selected) == {0, 1}

    def test_bmrf_mixed_with_bbs(self):
        """bmrf + bbs together in a gamboost model works."""
        rng = np.random.RandomState(42)
        n = 200
        n_levels = 5
        region = rng.randint(0, n_levels, n).astype(np.float64)
        x1 = rng.uniform(0, 1, n)
        y = np.array([3.0 * r for r in region]) + np.sin(2 * np.pi * x1) + rng.normal(0, 0.5, n)
        K = _make_simple_k(n_levels)

        model = gamboost(
            "y ~ x1 + region",
            {"y": y, "x1": x1, "region": region},
            family=Gaussian(),
            control=(100, 0.1),
            learners=[Bbs("x1", df=4), Bmrf("region", K=K, df=4)],
        )

        assert len(model.fitted_) == n
        assert not np.any(np.isnan(model.fitted_))
        assert set(model._selected) == {0, 1}
