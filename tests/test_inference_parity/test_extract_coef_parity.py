"""R-parity tests for extract() and coef() methods."""

import mboostrs as mboost
import numpy as np
import pytest


def _make_gaussian_data(n=100, seed=42):
    np.random.seed(seed)
    x1 = np.random.uniform(size=n)
    x2 = np.random.uniform(size=n)
    y = 2*x1 - 1*x2 + np.random.normal(size=n, scale=0.5)
    return {"y": y, "x1": x1, "x2": x2}


class TestGLMBoostExtract:
    def test_extract_coefficients(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        coefs = mod.extract("coefficients")
        assert isinstance(coefs, dict)
        assert len(coefs) > 0
        # Non-zero coefficients for selected predictors
        for name, val in coefs.items():
            if name == "(Intercept)":
                continue
            assert val != 0.0, f"{name} should have non-zero coefficient"

    def test_extract_offset(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        offset = mod.extract("offset")
        assert isinstance(offset, float)

    def test_extract_weights(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        w = mod.extract("weights")
        assert isinstance(w, np.ndarray)
        assert len(w) == 100
        np.testing.assert_array_equal(w, np.ones(100))

    def test_extract_design(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        design = mod.extract("design")
        assert isinstance(design, dict)
        assert "x1" in design
        assert "x2" in design
        # mboost reports the fitted columns: centered predictors and intercept.
        np.testing.assert_allclose(design["x1"], data["x1"] - np.mean(data["x1"]), rtol=1e-10)
        np.testing.assert_array_equal(design["(Intercept)"], np.ones(100))

    def test_extract_residuals(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        res = mod.extract("residuals")
        assert isinstance(res, np.ndarray)
        assert len(res) == 100
        # Residuals should be roughly centered for Gaussian
        assert abs(np.mean(res)) < 0.5

    def test_extract_invalid_what(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        with pytest.raises(ValueError, match="what must be one of"):
            mod.extract("nonexistent")


class TestGLMBoostCoef:
    def test_coef_default(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        coefs = mod.coef()
        assert isinstance(coefs, dict)
        assert len(coefs) > 0

    def test_coef_which_name(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        coefs = mod.coef(which="x1")
        assert "x1" in coefs
        assert coefs["x1"] != 0.0

    def test_coef_which_index(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        names = mod._learner_names
        idx = names.index("x1")
        coefs = mod.coef(which=idx)
        assert "x1" in coefs

    def test_coef_off2int(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        coefs_plain = mod.coef()
        coefs_off2int = mod.coef(off2int=True)
        # Off2int should add offset to intercept coefficient
        if "(Intercept)" in coefs_plain:
            diff = coefs_off2int.get("(Intercept)", 0.0) - coefs_plain.get("(Intercept)", 0.0)
            np.testing.assert_allclose(diff, mod.offset_, rtol=1e-10)

    def test_coef_matches_coefficients_property(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        prop_coefs = mod.coefficients_
        method_coefs = mod.coef()
        # Should give the same values
        for name in prop_coefs:
            np.testing.assert_allclose(method_coefs.get(name, 0.0), prop_coefs[name], rtol=1e-10)


class TestGAMBoostExtract:
    def test_extract_coefficients(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        coefs = mod.extract("coefficients")
        assert isinstance(coefs, dict)
        assert len(coefs) > 0

    def test_extract_design(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        design = mod.extract("design")
        assert isinstance(design, dict)
        for mat in design.values():
            if mat is not None:
                assert isinstance(mat, np.ndarray)
                assert mat.ndim == 2

    def test_extract_penalty(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        penalty = mod.extract("penalty")
        assert isinstance(penalty, dict)
        for mat in penalty.values():
            if mat is not None:
                assert isinstance(mat, np.ndarray)
                assert mat.ndim == 2

    def test_extract_lambda(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        x2 = np.random.normal(size=n)
        y = 3*np.sin(2*np.pi*x1) + 0.5*x2 + np.random.normal(size=n, scale=0.2)
        data = {"y": y, "x1": x1, "x2": x2}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bols(x2) + bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        lam = mod.extract("lambda")
        assert isinstance(lam, dict)
        # bols reports its (zero) ridge penalty; bbs reports the λ from the
        # df=4 calibration (was None before the worker). Both scalar and block
        # learners report a value, matching R's extract(what="lambda").
        assert "x2" in lam and lam["x2"] == 0.0
        assert any(v is not None for v in lam.values())

    @pytest.mark.r_parity
    def test_extract_lambda_matches_r(self):
        import rpy2.robjects as ro

        ro.r('suppressMessages(library(mboost))')
        res = ro.r('''function(){
          set.seed(1); n<-120; z<-runif(n,-3,3); x1<-rnorm(n)
          y<-sin(z)+0.5*x1+rnorm(n,sd=0.3)
          m<-gamboost(y~bbs(z,df=5)+bols(x1), data=data.frame(y=y,z=z,x1=x1),
                      control=boost_control(80,0.1))
          lam<-extract(m, what="lambda")
          list(z=z, x1=x1, y=y, lamvals=as.numeric(unlist(lam)))
        }''')()
        def g(k):
            return np.asarray(res.rx2(k), dtype=np.float64)

        data = {"y": g("y"), "z": g("z"), "x1": g("x1")}
        mod = mboost.gamboost(
            "y ~ bbs(z, df=5) + bols(x1)", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(80, 0.1),
        )
        py_vals = sorted(v for v in mod.extract("lambda").values() if v is not None)
        r_vals = sorted(g("lamvals").tolist())
        np.testing.assert_allclose(py_vals, r_vals, rtol=1e-6, atol=1e-12)

    def test_extract_df(self):
        np.random.seed(42)
        n = 120
        x1 = np.random.uniform(-3, 3, size=n)
        x2 = np.random.normal(size=n)
        # distinct predictors so both learners get selected (and thus appear in
        # extract(df), which defaults to selected learners)
        y = np.sin(x1) + 0.5 * x2 + np.random.normal(size=n, scale=0.3)
        data = {"y": y, "x1": x1, "x2": x2}

        ctrl = mboost.boost_control(mstop=80, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4) + bols(x2)", data=data, control=ctrl, family=mboost.Gaussian())
        df_dict = mod.extract("df")
        assert isinstance(df_dict, dict)
        for val in df_dict.values():
            if val is not None:
                assert isinstance(val, float)
                assert val > 0
        # R's extract(what="df") returns the SPECIFIED df, not the effective
        # trace-df of the smoother. bbs(x1, df=4) -> 4.0 (was reported ~3.4 from
        # the hat-matrix trace); bols(x2) -> 2.0 (its column count, intercept +
        # slope, which there equals the trace).
        np.testing.assert_allclose(sorted(df_dict.values()), [2.0, 4.0], atol=1e-9)

    @pytest.mark.r_parity
    def test_extract_df_matches_r(self):
        import rpy2.robjects as ro

        ro.r('suppressMessages(library(mboost))')
        res = ro.r('''function(){
          set.seed(1); n<-120; z<-runif(n,-3,3); x1<-rnorm(n)
          y<-sin(z)+0.5*x1+rnorm(n,sd=0.3)
          m<-gamboost(y~bbs(z,df=5)+bols(x1), data=data.frame(y=y,z=z,x1=x1),
                      control=boost_control(80,0.1))
          df<-extract(m, what="df")
          list(z=z, x1=x1, y=y, dfvals=as.numeric(unlist(df)))
        }''')()
        def g(k):
            return np.asarray(res.rx2(k), dtype=np.float64)

        data = {"y": g("y"), "z": g("z"), "x1": g("x1")}
        mod = mboost.gamboost(
            "y ~ bbs(z, df=5) + bols(x1)", data=data, family=mboost.Gaussian(),
            control=mboost.boost_control(80, 0.1),
        )
        py_vals = sorted(mod.extract("df").values())
        r_vals = sorted(g("dfvals").tolist())
        np.testing.assert_allclose(py_vals, r_vals, atol=1e-9)

    def test_extract_offset(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        offset = mod.extract("offset")
        assert isinstance(offset, float)

    def test_extract_residuals(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        res = mod.extract("residuals")
        assert isinstance(res, np.ndarray)
        assert len(res) == 100


class TestGAMBoostCoef:
    def test_coef_default(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        coefs = mod.coef()
        assert isinstance(coefs, dict)
        assert len(coefs) > 0
        assert "(offset)" in coefs
        assert coefs["(offset)"] == mod.offset_

    def test_coef_off2int_warns(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        with pytest.warns(UserWarning, match="off2int"):
            coefs = mod.coef(off2int=True)
        assert "(offset)" in coefs


class TestResiduals:
    def test_residuals_gaussian_glmboost(self):
        data = _make_gaussian_data()
        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
        res = mod.residuals()
        y = data["y"]
        np.testing.assert_allclose(res, y - mod.fitted_, rtol=1e-10)

    def test_residuals_gaussian_gamboost(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = 3*np.sin(2*np.pi*x1) + np.random.normal(size=n, scale=0.5)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.gamboost("y ~ bbs(x1, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
        res = mod.residuals()
        np.testing.assert_allclose(res, y - mod.fitted_, rtol=1e-10)

    def test_residuals_non_gaussian(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(size=n)
        y = np.random.binomial(1, 0.5, size=n).astype(float)
        data = {"y": y, "x1": x1}

        ctrl = mboost.boost_control(mstop=50, nu=0.1)
        mod = mboost.glmboost("y ~ x1", data=data, control=ctrl, family=mboost.Binomial())
        resid = mod.residuals()
        assert resid.shape == (n,)
        assert np.all(np.isfinite(resid))
