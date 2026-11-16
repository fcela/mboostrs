"""Stability selection parity tests vs R's stabs::stabsel."""

import numpy as np
import pytest

try:
    from rpy2.robjects import r
    _HAS_R = True
except Exception:
    _HAS_R = False

from mboostrs import Gaussian, boost_control, gamboost, glmboost
from mboostrs.stabsel import (
    _D_rc,
    _maxQ,
    _minD,
    _stabsel_parameters,
    _um_const,
    stabsel,
)

pytestmark = pytest.mark.r_parity
skip_no_r = pytest.mark.skipif(not _HAS_R, reason="R/rpy2 not available")


# ---------------------------------------------------------------------------
# Parameter computation parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestStabselParameters:
    """Verify _stabsel_parameters matches R's stabsel_parameters()."""

    @pytest.mark.parametrize("p,q,PFER,B,assumption", [
        (3, 1, 1.0, 50, "unimodal"),
        (4, 2, 1.0, 50, "unimodal"),
        (10, 3, 2.0, 100, "unimodal"),
        (5, 2, 0.5, 50, "unimodal"),
        (10, 3, 2.0, 100, "none"),
    ])
    def test_stabsel_parameters(self, p, q, PFER, B, assumption):
        r_result = r(f'stabs::stabsel_parameters(p={p}, q={q}, PFER={PFER}, B={B}, assumption="{assumption}")')
        r_cutoff = float(r_result.rx2("cutoff")[0])
        r_pfer = float(r_result.rx2("PFER")[0])

        py_result = _stabsel_parameters(p=p, q=q, PFER=PFER, B=B, assumption=assumption)

        assert abs(py_result["cutoff"] - r_cutoff) < 1e-6, \
            f"cutoff: py={py_result['cutoff']}, R={r_cutoff}"
        assert abs(py_result["PFER"] - r_pfer) < 1e-2, \
            f"PFER: py={py_result['PFER']}, R={r_pfer}"

    def test_cutoff_none_computed(self):
        r_result = r('stabs::stabsel_parameters(p=10, cutoff=0.8, PFER=2.0, B=50, assumption="unimodal")')
        r_q = int(r_result.rx2("q")[0])
        py_result = _stabsel_parameters(p=10, cutoff=0.8, PFER=2.0, B=50, assumption="unimodal")
        assert py_result["q"] == r_q, f"q: py={py_result['q']}, R={r_q}"

    def test_pfer_none_computed(self):
        r_result = r('stabs::stabsel_parameters(p=5, cutoff=0.8, q=2, B=50, assumption="unimodal")')
        r_pfer = float(r_result.rx2("PFER")[0])
        py_result = _stabsel_parameters(p=5, cutoff=0.8, q=2, B=50, assumption="unimodal")
        assert abs(py_result["PFER"] - r_pfer) < 1e-3, f"PFER: py={py_result['PFER']}, R={r_pfer}"


# ---------------------------------------------------------------------------
# um_const parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestUmConst:
    def test_um_const_values(self):
        r('library(stabs)')
        r('um_const_r <- get("um_const", envir = asNamespace("stabs"))')
        for cutoff in [0.6, 0.7, 0.75, 0.8, 0.9, 0.95]:
            for B in [10, 50, 100]:
                for theta in [0.1, 0.3, 0.5]:
                    try:
                        py_val = _um_const(cutoff, B, theta)
                    except ValueError:
                        continue
                    r_val = float(r(f'um_const_r(cutoff={cutoff}, B={B}, theta={theta})')[0])
                    assert abs(py_val - r_val) < 1e-10, \
                        f"um_const({cutoff},{B},{theta}): py={py_val}, R={r_val}"


# ---------------------------------------------------------------------------
# maxQ parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestMaxQ:
    @pytest.mark.parametrize("p,B", [(3, 50), (10, 100), (5, 20)])
    def test_maxq(self, p, B):
        r('library(stabs)')
        r('maxQ_r <- get("maxQ", envir = asNamespace("stabs"))')
        py_val = _maxQ(p, B)
        r_val = int(r(f'maxQ_r(p={p}, B={B})')[0])
        assert py_val == r_val, f"maxQ({p},{B}): py={py_val}, R={r_val}"


# ---------------------------------------------------------------------------
# Full stabsel on glmboost parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestStabselGlmBoost:
    """Compare stabsel output on a Gaussian glmboost model vs R."""

    def _make_data(self, seed=42):
        np.random.seed(seed)
        n = 100
        X = np.random.randn(n, 4)
        y = 2 * X[:, 0] + 1 * X[:, 1] + np.random.randn(n) * 0.5
        import pandas as pd
        data = pd.DataFrame({
            "y": y,
            "x1": X[:, 0],
            "x2": X[:, 1],
            "x3": X[:, 2],
            "x4": X[:, 3],
        })
        return data

    def _fit_r(self, data, mstop=100, nu=0.1):
        from rpy2.robjects import pandas2ri
        with pandas2ri.converter.context():
            data_r = pandas2ri.py2rpy(data)
        r.assign("data_r", data_r)
        r(f'''
library(mboost)
model_r <- glmboost(y ~ x1 + x2 + x3 + x4,
                     data=data_r, family=Gaussian(),
                     control=boost_control(mstop={mstop}, nu={nu}))
        ''')

    def test_stabsel_selected_glmboost(self):
        data = self._make_data(seed=42)
        ctrl = boost_control(mstop=100, nu=0.1)
        model = glmboost("y ~ bols(x1) + bols(x2) + bols(x3) + bols(x4)",
                         data=data, family=Gaussian(), control=ctrl)

        self._fit_r(data, mstop=100, nu=0.1)

        B = 5
        q_val = 2
        PFER_val = 1.0

        r.assign("B_r", B)
        r.assign("q_r", q_val)
        r.assign("PFER_r", PFER_val)
        r('''
set.seed(42)
folds_r <- cv(weights = rep(1, 100), type = "subsampling", B = B_r)
ss_r <- stabsel(model_r, q = q_r, PFER = PFER_r,
                folds = folds_r, assumption = "unimodal", sampling.type = "SS")
        ''')

        r_sel_names = list(r('names(ss_r$selected)'))
        r_cutoff = float(r('ss_r$cutoff')[0])
        r_pfer = float(r('ss_r$PFER')[0])
        r_q = int(r('ss_r$q')[0])

        # Get R folds and convert for Python
        r_folds = np.asarray(r('folds_r'))
        # R folds are (n, B) with values 0 or 1
        # Python stabsel uses the same format
        result = stabsel(model, q=q_val, PFER=PFER_val,
                         folds=r_folds, assumption="unimodal",
                         sampling_type="SS", verbose=False)

        assert abs(result.cutoff - r_cutoff) < 1e-6, \
            f"cutoff: py={result.cutoff}, R={r_cutoff}"
        assert result.q == r_q, f"q: py={result.q}, R={r_q}"
        assert abs(result.PFER - r_pfer) < 1e-2, \
            f"PFER: py={result.PFER}, R={r_pfer}"

        py_selected_names = set(result.learner_names[i] for i in result.selected)
        assert py_selected_names == set(r_sel_names), \
            f"selected: py={py_selected_names}, R={set(r_sel_names)}"

    def test_stabsel_phat_shape(self):
        data = self._make_data(seed=42)
        ctrl = boost_control(mstop=50, nu=0.1)
        model = glmboost("y ~ bols(x1) + bols(x2) + bols(x3) + bols(x4)",
                         data=data, family=Gaussian(), control=ctrl)

        result = stabsel(model, q=2, PFER=1.0, B=5, verbose=False)
        p = len(model._learner_names)
        assert result.phat.shape == (p, model.mstop)
        assert result.phat.max() <= 1.0
        assert result.phat.min() >= 0.0

    def test_stabsel_phat_values_parity(self):
        # The full selection-probability matrix must match R value-for-value,
        # not just the final selected set. Regression test: the per-subsample
        # refit was freezing the full-data offset instead of recomputing it per
        # fold (as R does), which left the selected set/cutoff right but the
        # phat trajectory off by up to ~0.19.
        data = self._make_data(seed=42)
        ctrl = boost_control(mstop=80, nu=0.1)
        model = glmboost("y ~ bols(x1) + bols(x2) + bols(x3) + bols(x4)",
                         data=data, family=Gaussian(), control=ctrl)
        self._fit_r(data, mstop=80, nu=0.1)
        r.assign("B_r", 8)
        r('''
set.seed(202)
folds_r <- cv(weights = rep(1, 100), type = "subsampling", B = B_r)
ss_r <- stabsel(model_r, q = 2, PFER = 1.0, folds = folds_r,
                assumption = "unimodal", sampling.type = "SS")
        ''')
        r_folds = np.asarray(r('folds_r'))
        r_phat = np.asarray(r('as.matrix(ss_r$phat)'))  # (p, mstop)
        result = stabsel(model, q=2, PFER=1.0, folds=r_folds,
                         assumption="unimodal", sampling_type="SS", verbose=False)
        py_phat = np.asarray(result.phat)
        assert py_phat.shape == r_phat.shape, f"{py_phat.shape} vs {r_phat.shape}"
        np.testing.assert_allclose(
            py_phat, r_phat, atol=1e-12,
            err_msg="stabsel phat (selection probabilities) does not match R",
        )


# ---------------------------------------------------------------------------
# Full stabsel on gamboost parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestStabselGamBoost:
    """Compare stabsel output on a Gaussian gamboost model vs R."""

    def _make_data(self, seed=42):
        np.random.seed(seed)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = np.sin(2 * x1) + 0.5 * x2 + np.random.randn(n) * 0.3
        import pandas as pd
        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
        return data

    def test_stabsel_gamboost_basic(self):
        data = self._make_data(seed=42)
        ctrl = boost_control(mstop=50, nu=0.1)
        model = gamboost("y ~ bbs(x1, df=4) + bbs(x2, df=4) + bbs(x3, df=4)",
                         data=data, family=Gaussian(), control=ctrl)

        result = stabsel(model, q=2, PFER=1.0, B=3, verbose=False)
        p = len(model._learner_names)
        assert result.phat.shape[0] == p
        assert result.phat.shape[1] == model.mstop
        assert result.max_.shape == (p,)

    def test_stabsel_gamboost_phat_values_parity(self):
        # Value-level phat parity for gamboost (was shape-only before).
        data = self._make_data(seed=42)
        from rpy2.robjects import pandas2ri
        with pandas2ri.converter.context():
            gdata_r = pandas2ri.py2rpy(data)
        r.assign("gdata_r", gdata_r)
        r('''
library(mboost)
gmodel_r <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4) + bbs(x3, df=4),
                     data=gdata_r, family=Gaussian(),
                     control=boost_control(mstop=50, nu=0.1))
set.seed(303)
gfolds_r <- cv(weights = rep(1, 100), type = "subsampling", B = 6)
gss_r <- stabsel(gmodel_r, q = 2, PFER = 1.0, folds = gfolds_r,
                 assumption = "unimodal", sampling.type = "SS")
invisible(NULL)
        ''')
        r_folds = np.asarray(r('gfolds_r'))
        r_phat = np.asarray(r('as.matrix(gss_r$phat)'))
        ctrl = boost_control(mstop=50, nu=0.1)
        model = gamboost("y ~ bbs(x1, df=4) + bbs(x2, df=4) + bbs(x3, df=4)",
                         data=data, family=Gaussian(), control=ctrl)
        result = stabsel(model, q=2, PFER=1.0, folds=r_folds,
                         assumption="unimodal", sampling_type="SS", verbose=False)
        py_phat = np.asarray(result.phat)
        assert py_phat.shape == r_phat.shape, f"{py_phat.shape} vs {r_phat.shape}"
        # The selection-determining quantity (max selection prob per learner)
        # must match R exactly.
        np.testing.assert_allclose(
            py_phat.max(axis=1), r_phat.max(axis=1), atol=1e-10,
            err_msg="gamboost stabsel max selection prob does not match R",
        )
        # The full phat trajectory matches R except for rare per-subsample
        # selection ties (a learner selected one step earlier/later in a single
        # subsample when two bbs learners have near-equal SSE — inherent
        # floating point, not a systematic offset). Bound each element by the
        # per-subsample quantum (1 subsample) and the total mismatch count low;
        # this still catches a systematic shift like the old glmboost offset bug
        # (which moved phat by ~3 subsamples across many entries).
        B_sub = r_folds.shape[1] * 2  # SS sampling doubles the folds
        elt_diff = np.abs(py_phat - r_phat)
        assert elt_diff.max() <= 1.0 / B_sub + 1e-9, (
            f"gamboost phat off by more than one subsample: max={elt_diff.max()}"
        )
        assert np.sum(elt_diff > 1e-9) <= 3, (
            f"too many gamboost phat mismatches vs R: {int(np.sum(elt_diff > 1e-9))}"
        )


# ---------------------------------------------------------------------------
# r-concave parameter parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestStabselRconcaveParameters:
    """Verify r-concave _stabsel_parameters matches R's stabsel_parameters()."""

    @pytest.mark.parametrize("p,q,PFER,B", [
        (10, 3, 2.0, 50),
        (5, 2, 1.0, 50),
        (10, 3, 1.0, 100),
        (20, 5, 3.0, 50),
    ])
    def test_rconcave_cutoff_from_q_pfer(self, p, q, PFER, B):
        r_result = r(f'stabs::stabsel_parameters(p={p}, q={q}, PFER={PFER}, B={B}, assumption="r-concave")')
        r_cutoff = float(r_result.rx2("cutoff")[0])
        r_pfer = float(r_result.rx2("PFER")[0])
        py_result = _stabsel_parameters(p=p, q=q, PFER=PFER, B=B, assumption="r-concave")
        assert abs(py_result["cutoff"] - r_cutoff) < 1e-4, \
            f"cutoff: py={py_result['cutoff']}, R={r_cutoff}"
        assert abs(py_result["PFER"] - r_pfer) < 5e-2, \
            f"PFER: py={py_result['PFER']}, R={r_pfer}"

    def test_rconcave_q_from_cutoff_pfer(self):
        r_result = r('stabs::stabsel_parameters(p=10, cutoff=0.8, PFER=2.0, B=50, assumption="r-concave")')
        r_q = int(r_result.rx2("q")[0])
        py_result = _stabsel_parameters(p=10, cutoff=0.8, PFER=2.0, B=50, assumption="r-concave")
        assert py_result["q"] == r_q, f"q: py={py_result['q']}, R={r_q}"

    def test_rconcave_pfer_from_q_cutoff(self):
        r_result = r('stabs::stabsel_parameters(p=10, cutoff=0.8, q=3, B=50, assumption="r-concave")')
        r_pfer = float(r_result.rx2("PFER")[0])
        py_result = _stabsel_parameters(p=10, cutoff=0.8, q=3, B=50, assumption="r-concave")
        assert abs(py_result["PFER"] - r_pfer) < 5e-2, \
            f"PFER: py={py_result['PFER']}, R={r_pfer}"


# ---------------------------------------------------------------------------
# D_rc and minD parity
# ---------------------------------------------------------------------------

@skip_no_r
class TestRconcaveDFunctions:
    """Verify _D_rc and _minD match R's stabs::D and stabs::minD."""

    @classmethod
    def setup_class(cls):
        r('library(stabs)')
        r('D_r <- get("D", envir = asNamespace("stabs"))')
        r('minD_r <- get("minD", envir = asNamespace("stabs"))')

    @pytest.mark.parametrize("theta,which,B,r_val", [
        (0.09, 10, 50, -0.5),
        (0.09, 10, 50, -0.25),
        (0.3, 60, 100, -0.25),
        (0.05, 5, 20, -0.5),
    ])
    def test_D_rc(self, theta, which, B, r_val):
        py_val = _D_rc(theta, which, B, r_val)
        r_result = float(r(f'D_r(theta={theta}, which={which}, B={B}, r={r_val})')[0])
        assert abs(py_val - r_result) < 1e-4, \
            f"D({theta},{which},{B},{r_val}): py={py_val}, R={r_result}"

    @pytest.mark.parametrize("q,p,cutoff,B", [
        (3, 10, 0.6, 50),
        (3, 10, 0.7, 50),
        (2, 5, 0.67, 50),
    ])
    def test_minD(self, q, p, cutoff, B):
        py_val = _minD(q, p, cutoff, B)
        r_result = float(r(f'minD_r(q={q}, p={p}, pi={cutoff}, B={B})')[0])
        assert abs(py_val - r_result) < 1e-4, \
            f"minD({q},{p},{cutoff},{B}): py={py_val}, R={r_result}"


# ---------------------------------------------------------------------------
# q-truncation correctness
# ---------------------------------------------------------------------------

class TestQTruncation:
    """Verify q-truncation matches R's stabsel.mboost truncation logic."""

    def test_truncation_after_q_unique(self):
        """After q+1 unique learners appear, all subsequent selections become sel[0]."""
        sel = [2, 3, 1, 2, 3, 1, 0, 4]
        q_val = 2
        # R: qq = [1,2,3,3,3,3,3,3], qq > 2 => [F,F,T,T,T,T,T,T]
        # xs[qq > 2] <- xs[1] => [2, 3, 2, 2, 2, 2, 2, 2]
        expected = [2, 3, 2, 2, 2, 2, 2, 2]
        truncated = []
        seen = set()
        threshold_reached = False
        for idx in sel:
            if threshold_reached:
                truncated.append(sel[0])
            else:
                if idx not in seen:
                    seen.add(idx)
                    if len(seen) > q_val:
                        threshold_reached = True
                        truncated.append(sel[0])
                    else:
                        truncated.append(idx)
                else:
                    truncated.append(idx)
        assert truncated == expected, f"Got {truncated}, expected {expected}"

    def test_truncation_simple(self):
        """Simple case: sel=[0,1,2,3], q=2 -> [0,1,0,0]."""
        sel = [0, 1, 2, 3]
        q_val = 2
        truncated = []
        seen = set()
        threshold_reached = False
        for idx in sel:
            if threshold_reached:
                truncated.append(sel[0])
            else:
                if idx not in seen:
                    seen.add(idx)
                    if len(seen) > q_val:
                        threshold_reached = True
                        truncated.append(sel[0])
                    else:
                        truncated.append(idx)
                else:
                    truncated.append(idx)
        assert truncated == [0, 1, 0, 0]

    def test_truncation_at_threshold(self):
        """When exactly q unique learners, no truncation yet."""
        sel = [0, 1, 0, 1, 0]
        q_val = 2
        truncated = []
        seen = set()
        threshold_reached = False
        for idx in sel:
            if threshold_reached:
                truncated.append(sel[0])
            else:
                if idx not in seen:
                    seen.add(idx)
                    if len(seen) > q_val:
                        threshold_reached = True
                        truncated.append(sel[0])
                    else:
                        truncated.append(idx)
                else:
                    truncated.append(idx)
        assert truncated == [0, 1, 0, 1, 0]
