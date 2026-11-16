"""Reproductions of the examples in mboost's and gamboostLSS's help pages.

The documented examples are what a user copies first, and they combine
features in ways the unit tests do not. Targets are the values R prints when
the example is run, recorded here so the comparison needs no live R.
"""
import numpy as np
import pytest
from mboostrs.families import Gaussian, Laplace

import mboostrs as mboost

# R: datasets::cars, used by ?glmboost.
CARS_SPEED = np.array([
    4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13,
    14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19,
    19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25], dtype=float)
CARS_DIST = np.array([
    2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
    26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46,
    68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85], dtype=float)


@pytest.fixture
def cars():
    return {"dist": CARS_DIST, "speed": CARS_SPEED}


def _off2int(model):
    """R's coef(fit, off2int=TRUE): the offset folded into the intercept."""
    coefficients = model.coef()
    intercept = float(coefficients.get("(Intercept)", 0.0)) + float(model._offset)
    return intercept, float(coefficients["speed"])


class TestGlmboostCarsExample:
    """?glmboost: boosting `dist ~ speed` toward the least-squares fit."""

    def test_centred_fit_reaches_the_least_squares_solution(self, cars):
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(2000, 0.1))
        intercept, slope = _off2int(model)
        # R: coef(lm(dist ~ speed, data = cars))
        assert intercept == pytest.approx(-17.5791, abs=1e-3)
        assert slope == pytest.approx(3.9324, abs=1e-3)

    def test_uncentred_fit_matches_r_and_has_not_converged(self, cars):
        """The example passes center=FALSE to show slower convergence; R's own
        comment is "almost converged"."""
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(2000, 0.1),
                                center=False)
        intercept, slope = _off2int(model)
        # R: coef(glmboost(..., center = FALSE), off2int = TRUE)
        assert intercept == pytest.approx(-17.3512, abs=1e-3)
        assert slope == pytest.approx(3.9184, abs=1e-3)

    def test_laplace_variant_matches_r(self, cars):
        model = mboost.glmboost("dist ~ speed", data=cars, family=Laplace(),
                                control=mboost.boost_control(1000, 0.1))
        intercept, slope = _off2int(model)
        assert intercept == pytest.approx(-11.6310, abs=1e-3)
        assert slope == pytest.approx(3.4020, abs=1e-3)

    def test_offset_matches_r(self, cars):
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(100, 0.1))
        assert float(model._offset) == pytest.approx(42.98, abs=1e-6)

    @pytest.mark.parametrize("center", [True, False])
    def test_replay_agrees_with_the_fit_under_either_centring(self, cars, center):
        """Centring is applied inside the engine; the means it used must reach
        the model, or a replayed prediction disagrees with its own fit."""
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(200, 0.1),
                                center=center)
        np.testing.assert_allclose(
            np.asarray(model.fitted()),
            np.asarray(model.predict(newdata=cars)),
            rtol=0, atol=1e-9,
        )


class TestBaselearnersExample:
    """?baselearners exercises learner spellings the unit tests do not."""

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(290875)
        n = 150
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        z = rng.normal(size=n)
        return {"y": 2 * x1 - x2 + rng.normal(scale=0.3, size=n),
                "x1": x1, "x2": x2, "z": z}

    def test_multi_column_bols_is_refused_not_silently_truncated(self, frame):
        """R's bols(x1, x2) is ONE learner over both columns. The kwargs
        parser matched `key=value` only, so the bare second name matched
        nothing and was dropped: the fit used x1 alone and lost x2 with no
        warning."""
        with pytest.raises(ValueError, match="names a second column"):
            mboost.gamboost("y ~ bols(x1, x2)", data=frame, family=Gaussian(),
                            control=mboost.boost_control(10, 0.1))

    def test_a_list_of_names_is_refused_at_construction(self):
        """Reached the fit path unchecked and failed with
        "unhashable type: 'list'" when used as a dict key."""
        with pytest.raises(TypeError, match="takes one column name"):
            mboost.Bols(["x1", "x2"])

    @pytest.mark.parametrize("formula", [
        "y ~ bols(x1)",
        "y ~ bols(x1, intercept=False)",
        "y ~ bols(x1, by=z)",
        "y ~ bols(x1) + bols(x2)",
        "y ~ bbs(x1, df=4) + bbs(x1, by=z, df=8)",
        "y ~ bspatial(x1, x2, knots=12, df=6)",
    ])
    def test_the_supported_spellings_still_parse(self, frame, formula):
        model = mboost.gamboost(formula, data=frame, family=Gaussian(),
                                control=mboost.boost_control(20, 0.1))
        assert np.isfinite(np.asarray(model.predict())).all()

    def test_a_multi_column_learner_is_available_through_buser(self, frame):
        """The engine supports a multi-column linear learner; only the
        bols(a, b) spelling is missing, so the error points here."""
        n = len(frame["y"])
        design = np.column_stack([frame["x1"], frame["x2"]])
        model = mboost.gamboost(
            "y ~ .", data={**frame, "u": np.zeros(n)}, family=Gaussian(),
            learners=[mboost.Buser("u", X=design, K=np.eye(2), df=2)],
            control=mboost.boost_control(300, 0.1))
        coefficients = np.asarray(model.coef()["u"])
        assert coefficients[0] == pytest.approx(2.0, abs=0.15)
        assert coefficients[1] == pytest.approx(-1.0, abs=0.15)


class TestMethodsExample:
    """?methods: the accessor surface, against the values R prints."""

    def test_corrected_aic_matches_r(self, cars):
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(2000, 0.1),
                                center=False)
        result = model.aic(method="corrected")
        assert float(np.min(np.asarray(result.aic))) == pytest.approx(6.5554, abs=1e-3)
        assert result.mstop == 1549

    def test_centred_intercept_is_the_implied_one(self, cars):
        """R: coef(fit) returns an intercept equal to -mean(x) * coef(x),
        estimated implicitly by the centring rather than selected."""
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(100, 0.1))
        coefficients = model.coef()
        implied = -CARS_SPEED.mean() * float(coefficients["speed"])
        assert float(coefficients["(Intercept)"]) == pytest.approx(implied, abs=1e-9)
        # R's printed values for this fit.
        assert float(coefficients["(Intercept)"]) == pytest.approx(-60.557486, abs=1e-4)
        assert float(coefficients["speed"]) == pytest.approx(3.932304, abs=1e-4)

    def test_intercept_is_never_selected_when_centred(self, cars):
        model = mboost.glmboost("dist ~ speed", data=cars, family=Gaussian(),
                                control=mboost.boost_control(100, 0.1))
        assert set(np.asarray(model.selected).tolist()) == {"speed"}


class TestAicDegreesOfFreedom:
    """The AIC hat trace decided whether a learner carried an intercept by
    testing `centering != 0`. glmboost centres every predictor but keeps the
    intercept as its own learner, so a centred fit — the default — counted a
    column that is not there and reported 2*nu per selected learner where R
    reports nu."""

    @pytest.fixture
    def simple(self):
        rng = np.random.default_rng(1)
        n = 60
        x = rng.normal(size=n)
        return {"x": x, "y": 2 * x + rng.normal(scale=0.3, size=n)}

    @pytest.mark.parametrize("center", [True, False])
    def test_glmboost_df_is_nu_per_selected_learner(self, simple, center):
        model = mboost.glmboost("y ~ x", data=simple, family=Gaussian(),
                                control=mboost.boost_control(20, 0.1),
                                center=center)
        df = np.asarray(model.aic().df)
        # R gives the same values for both centring settings.
        np.testing.assert_allclose(df[:3], [0.100000, 0.190000, 0.271000], atol=1e-6)

    def test_gamboost_bols_keeps_its_intercept(self, simple):
        """R's bols(x) does include an intercept column, so its trace is 2."""
        model = mboost.gamboost("y ~ bols(x)", data=simple, family=Gaussian(),
                                control=mboost.boost_control(20, 0.1))
        df = np.asarray(model.aic().df)
        np.testing.assert_allclose(df[:3], [0.200000, 0.380000, 0.542000], atol=1e-6)

    def test_bodyfat_aic_matches_r(self):
        """?cvrisk fits this model and reads mstop off the AIC."""
        bodyfat = mboost.datasets.bodyfat()
        predictors = [c for c in bodyfat.columns if c != "DEXfat"]
        model = mboost.glmboost("DEXfat ~ " + " + ".join(predictors),
                                data=bodyfat, family=Gaussian(),
                                control=mboost.boost_control(100, 0.1))
        result = model.aic()
        assert float(np.min(np.asarray(result.aic))) == pytest.approx(3.3527, abs=1e-3)
        assert result.mstop == 45
        assert float(np.asarray(result.df)[44]) == pytest.approx(1.917234, abs=1e-5)


class TestCvriskExample:
    """?cvrisk: with the same fold matrix the risk path must agree exactly."""

    def test_kfold_weights_have_r_s_shape(self):
        bodyfat = mboost.datasets.bodyfat()
        folds = mboost.cv_weights(bodyfat.height, type="kfold", B=10, seed=1)
        assert folds.shape == (71, 10)
        # 71 rows do not divide by 10, so each fold holds out 7 or 8 and the
        # in-bag totals are 63 or 64, as R's are.
        assert set(folds.sum(axis=0).astype(int).tolist()) <= {63, 64}
        # Every row is held out exactly once across the ten folds.
        held_out = (10 - folds.sum(axis=1)).astype(int)
        assert set(held_out.tolist()) == {1}


# R: the negative-binomial simulation from ?mboostLSS, with the values that
# example's own model2 produces.
class TestMboostLSSExample:
    """?mboostLSS: a named list of formulas, one per distribution parameter."""

    @pytest.fixture(scope="class")
    def nbinomial(self):
        """The example's data-generating process, reproduced in numpy.

        The draws differ from R's RNG, so slope targets are not comparable;
        what is comparable is the internal identity below, which must hold on
        any data.
        """
        rng = np.random.default_rng(1907)
        n = 1000
        x = {f"x{i}": rng.normal(size=n) for i in range(1, 7)}
        mu = np.exp(1.5 + x["x1"] + 0.5 * x["x2"] - 0.5 * x["x3"] - x["x4"])
        sigma = np.exp(-0.4 * x["x3"] - 0.2 * x["x4"]
                       + 0.2 * x["x5"] + 0.4 * x["x6"])
        # rnbinom(size = sigma, mu = mu) as a gamma-Poisson mixture.
        y = rng.poisson(rng.gamma(shape=sigma, scale=mu / sigma))
        return {**x, "y": y.astype(float)}

    def test_named_formulas_give_each_parameter_its_own_learners(self, nbinomial):
        model = mboost.glmboostLSS(
            {"mu": "y ~ x1 + x2 + x3 + x4", "sigma": "y ~ x3 + x4 + x5 + x6"},
            data=nbinomial, families=mboost.NBinomialLSS(),
            control=mboost.boost_control(200, 0.1))
        assert list(model.param_names) == ["mu", "sigma"]
        assert list(model._params[0]["predictor_names"]) == ["x1", "x2", "x3", "x4"]
        assert list(model._params[1]["predictor_names"]) == ["x3", "x4", "x5", "x6"]

    def test_mu_offset_is_the_log_mean_response(self, nbinomial):
        """R: NBinomialLSS uses a log link, and the mu offset is log(mean(y))."""
        model = mboost.glmboostLSS(
            {"mu": "y ~ x1 + x2 + x3 + x4", "sigma": "y ~ x3 + x4 + x5 + x6"},
            data=nbinomial, families=mboost.NBinomialLSS(),
            control=mboost.boost_control(50, 0.1))
        expected = float(np.log(np.mean(nbinomial["y"])))
        assert float(model._params[0]["offset"]) == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("parameter,columns", [
        ("mu", ["x1", "x2", "x3", "x4"]),
        ("sigma", ["x3", "x4", "x5", "x6"]),
    ])
    def test_reported_coefficients_reproduce_the_linear_predictor(
            self, nbinomial, parameter, columns):
        """Each predictor is fitted on its centred column, so its slope carries
        an implied shift of -mean*beta belonging to the intercept. The LSS coef
        path accumulated the intercept learner's own updates only, so the
        reported coefficients were short by a constant — while predict() stayed
        right, because the engine applies the centring itself."""
        model = mboost.glmboostLSS(
            {"mu": "y ~ x1 + x2 + x3 + x4", "sigma": "y ~ x3 + x4 + x5 + x6"},
            data=nbinomial, families=mboost.NBinomialLSS(),
            control=mboost.boost_control(300, 0.1))

        index = list(model.param_names).index(parameter)
        coefficients = {k: float(np.asarray(v).ravel()[0])
                        for k, v in model.coef()[parameter].items()}
        rebuilt = (coefficients["(Intercept)"]
                   + float(model._params[index]["offset"])
                   + sum(coefficients[c] * nbinomial[c] for c in columns))
        np.testing.assert_allclose(
            rebuilt,
            np.asarray(model.predict(parameter=parameter, type="link")),
            rtol=0, atol=1e-9,
        )


class TestAsFamiliesExample:
    """?as.families: gamlss's "NO" is the Gaussian location-scale family, so
    the example's fit is reproducible even though the as.families bridge
    itself is not implemented."""

    @pytest.fixture
    def normal_data(self):
        rng = np.random.default_rng(123)
        n = 1000
        x = rng.uniform(size=n)
        y = rng.normal(loc=2 + 3 * x, scale=np.exp(1 - x))
        return {"x": x, "y": y}

    def test_recovers_the_generating_process(self, normal_data):
        model = mboost.glmboostLSS("y ~ x", data=normal_data,
                                   families=mboost.GaussianLSS(),
                                   control=mboost.boost_control(100, 0.1))
        coefficients = model.coef()
        mu_slope = float(np.asarray(coefficients["mu"]["x"]).ravel()[0])
        sigma_slope = float(np.asarray(coefficients["sigma"]["x"]).ravel()[0])
        # True values 3 and -1. The draws differ from R's RNG, so the fitted
        # slopes differ too; what must hold is that both are recovered near
        # their generating values.
        assert 2.0 < mu_slope < 3.5
        assert -1.4 < sigma_slope < -0.6

    def test_off2int_folds_the_offset_into_the_intercept(self, normal_data):
        """R reports coef(fit, off2int = TRUE); ours reports the intercept and
        the offset separately, and the two must agree."""
        model = mboost.glmboostLSS("y ~ x", data=normal_data,
                                   families=mboost.GaussianLSS(),
                                   control=mboost.boost_control(100, 0.1))
        for index, name in enumerate(model.param_names):
            coefficients = model.coef()[name]
            intercept = float(np.asarray(coefficients["(Intercept)"]).ravel()[0])
            offset = float(model._params[index]["offset"])
            slope = float(np.asarray(coefficients["x"]).ravel()[0])
            rebuilt = intercept + offset + slope * normal_data["x"]
            np.testing.assert_allclose(
                rebuilt,
                np.asarray(model.predict(parameter=name, type="link")),
                rtol=0, atol=1e-9)


class TestVarimpExample:
    """?varimp on bodyfat. R returns raw risk reductions summing to the total
    reduction; mboostrs returns shares summing to 1. The shares agree."""

    def test_shares_match_r(self):
        bodyfat = mboost.datasets.bodyfat()
        predictors = [c for c in bodyfat.columns if c != "DEXfat"]
        model = mboost.glmboost("DEXfat ~ " + " + ".join(predictors),
                                data=bodyfat, family=Gaussian(),
                                control=mboost.boost_control(100, 0.1))
        reduction = model.varimp()["reduction"]
        assert sum(float(v) for v in reduction.values()) == pytest.approx(1.0)
        # R: varimp(glmboost(DEXfat ~ ., data = bodyfat)), normalised.
        for name, share in (("hipcirc", 0.4400829), ("waistcirc", 0.3069008),
                            ("anthro3a", 0.1427191), ("anthro3b", 0.0656165),
                            ("kneebreadth", 0.0372508)):
            assert float(reduction[name]) == pytest.approx(share, abs=1e-6)


class TestSurvFitExample:
    """?survFit on the ovarian data."""

    def test_survival_curve_matches_r(self):
        # R: survival::ovarian, the six columns the example uses.
        futime = np.array([59, 115, 156, 421, 431, 448, 464, 475, 477, 563,
                           638, 744, 769, 770, 803, 855, 1040, 1106, 1129,
                           1206, 1227, 268, 329, 353, 365, 377], dtype=float)
        fustat = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
        age = np.array([72.3315, 74.4932, 66.4658, 53.3644, 50.3397, 56.4301,
                        56.9370, 59.8548, 64.1753, 55.1781, 56.7562, 50.1096,
                        59.6301, 57.0521, 39.2712, 43.1233, 38.8932, 44.6000,
                        53.9068, 44.2055, 59.5890, 74.5041, 43.1370, 63.2192,
                        64.4247, 58.3096])
        resid_ds = np.array([2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1,
                             2, 1, 1, 2, 1, 2, 2, 1, 2, 1], dtype=float)
        data = {"futime": futime, "fustat": fustat, "age": age,
                "resid.ds": resid_ds}

        model = mboost.glmboost("Surv(futime, fustat) ~ age + resid.ds",
                                data=data, family=mboost.CoxPH(),
                                control=mboost.boost_control(50, 0.1))
        coefficients = model.coef()
        assert float(coefficients["age"]) == pytest.approx(0.095014, abs=1e-5)
        assert float(coefficients["resid.ds"]) == pytest.approx(0.250679, abs=1e-5)

        fit = model.survfit()
        times = np.asarray(fit["time"] if isinstance(fit, dict) else fit.time)
        np.testing.assert_array_equal(
            times.astype(int),
            [59, 115, 156, 268, 329, 353, 365, 431, 464, 475, 563, 638])
        surv = np.asarray(fit["surv"] if isinstance(fit, dict) else fit.surv)
        np.testing.assert_allclose(
            surv[:4, 0], [0.976466, 0.950306, 0.919635, 0.886821], atol=1e-6)


class TestStabselParametersExample:
    """?stabsel: the error-control parameters are deterministic given q and
    PFER, so they compare exactly even though the selection is stochastic."""

    def test_mb_parameters_match_r(self):
        bodyfat = mboost.datasets.bodyfat()
        predictors = [c for c in bodyfat.columns if c != "DEXfat"]
        model = mboost.glmboost("DEXfat ~ " + " + ".join(predictors),
                                data=bodyfat, family=Gaussian(),
                                control=mboost.boost_control(100, 0.1))
        parameters = mboost.stabsel(model, q=3, PFER=1, sampling_type="MB",
                                    B=20, eval=False)
        # R: stabsel_parameters(q = 3, PFER = 1, p = 10, sampling.type = "MB")
        assert parameters["cutoff"] == pytest.approx(0.95)
        assert parameters["PFER"] == pytest.approx(1.0)
        assert parameters["q"] == 3
        assert parameters["p"] == 10


class TestLssMethodsExample:
    """?methods (gamboostLSS): the accessor surface, including R's
    `mstop(model) <- c(mu = 100, sigma = 200)`."""

    @pytest.fixture
    def lss(self):
        rng = np.random.default_rng(1907)
        n = 600
        x = {f"x{i}": rng.normal(size=n) for i in range(1, 5)}
        mu = np.exp(1.5 + x["x1"] + 0.5 * x["x2"])
        sigma = np.exp(-0.2 * x["x3"] + 0.4 * x["x4"])
        y = rng.poisson(rng.gamma(shape=sigma, scale=mu / sigma)).astype(float)
        return mboost.glmboostLSS(
            "y ~ x1 + x2 + x3 + x4", data={**x, "y": y},
            families=mboost.NBinomialLSS(),
            control=mboost.boost_control(400, 0.1))

    def test_a_single_mstop_truncates_every_parameter(self, lss):
        assert lss.with_mstop(200).mstop == 200

    @pytest.mark.parametrize("spec", [{"mu": 100, "sigma": 200}, [100, 200]])
    def test_per_parameter_mstop(self, lss, spec):
        """R: mstop(model) <- c(mu = 100, sigma = 200)."""
        truncated = lss.with_mstop(spec)
        assert truncated.mstop == {"mu": 100, "sigma": 200}
        lengths = [len(truncated._params[k]["selected"])
                   for k in range(len(truncated.param_names))]
        assert lengths == [100, 200]

    def test_coef_by_parameter(self, lss):
        """R: coef(model, parameter = "mu")."""
        assert set(lss.coef(parameter="mu")) <= {
            "(Intercept)", "x1", "x2", "x3", "x4"}
