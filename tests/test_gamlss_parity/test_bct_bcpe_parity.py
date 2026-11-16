"""R parity for the Box--Cox t and power-exponential LSS families.

The distribution helpers are compared with ``gamlss.dist`` directly.  The
fit test additionally exercises offsets, all four analytical gradients, and
link handling through a complete cyclic ``glmboostLSS`` path.
"""

import numpy as np
import pytest

import mboostrs as mb


FAMILIES = [("BCT", mb.BCTLSS), ("BCPE", mb.BCPELSS)]
PARAM_NAMES = ("mu", "sigma", "nu", "tau")


def _r_distribution(family_name):
    import rpy2.robjects as ro

    run = ro.r(
        """
        function(family_name) {
            y <- c(0.15, 0.7, 1.4, 3.5)
            p <- c(0.01, 0.2, 0.7, 0.99)
            mu <- c(1.7, 2.4, 0.9, 1.2)
            sigma <- c(0.4, 0.7, 0.25, 0.6)
            nu <- c(0.3, -0.5, 0.01, 0)
            tau <- c(5, 12, 2.2, 4)
            if (family_name == "BCT") {
                density <- gamlss.dist::dBCT(y, mu, sigma, nu, tau)
                cdf <- gamlss.dist::pBCT(y, mu, sigma, nu, tau)
                quantile <- gamlss.dist::qBCT(p, mu, sigma, nu, tau)
            } else {
                density <- gamlss.dist::dBCPE(y, mu, sigma, nu, tau)
                cdf <- gamlss.dist::pBCPE(y, mu, sigma, nu, tau)
                quantile <- gamlss.dist::qBCPE(p, mu, sigma, nu, tau)
            }
            list(y = y, p = p, mu = mu, sigma = sigma, nu = nu, tau = tau,
                 density = density, cdf = cdf, quantile = quantile)
        }
        """
    )
    result = run(family_name)
    return {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in (*PARAM_NAMES, "y", "p", "density", "cdf", "quantile")
    }


def _r_fit(family_name, mstop=20, step_length=0.05):
    import rpy2.robjects as ro

    run = ro.r(
        """
        function(family_name, mstop, step_length) {
            suppressMessages(library(gamboostLSS))
            set.seed(73)
            n <- 120
            x1 <- rnorm(n)
            x2 <- runif(n, -1, 1)
            mu <- 2 + 0.25 * x1
            sigma <- 0.18
            nu <- if (family_name == "BCT") 0.4 else 0.7
            tau <- if (family_name == "BCT") 6 else 2.5
            u <- runif(n, 0.01, 0.99)
            y <- if (family_name == "BCT") {
                gamlss.dist::qBCT(u, mu, sigma, nu, tau)
            } else {
                gamlss.dist::qBCPE(u, mu, sigma, nu, tau)
            }
            model <- glmboostLSS(
                y ~ x1 + x2,
                data = data.frame(y = y, x1 = x1, x2 = x2),
                families = as.families(family_name),
                control = boost_control(mstop = mstop, nu = step_length)
            )
            list(
                x1 = x1, x2 = x2, y = y,
                mu = as.numeric(predict(model, parameter = "mu", type = "link")),
                sigma = as.numeric(predict(model, parameter = "sigma", type = "link")),
                nu = as.numeric(predict(model, parameter = "nu", type = "link")),
                tau = as.numeric(predict(model, parameter = "tau", type = "link"))
            )
        }
        """
    )
    result = run(family_name, mstop, step_length)
    return {
        name: np.asarray(result.rx2(name), dtype=np.float64)
        for name in ("x1", "x2", "y", *PARAM_NAMES)
    }


@pytest.mark.r_parity
@pytest.mark.parametrize(("family_name", "family_cls"), FAMILIES)
def test_distribution_helpers_match_gamlss_dist(family_name, family_cls):
    expected = _r_distribution(family_name)
    params = {name: expected[name] for name in PARAM_NAMES}
    family = family_cls()
    np.testing.assert_allclose(
        family.pdf(expected["y"], params), expected["density"], rtol=2e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        family.cdf(expected["y"], params), expected["cdf"], rtol=2e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        family.quantile(expected["p"], params),
        expected["quantile"],
        rtol=0,
        atol=5e-10,
    )


@pytest.mark.r_parity
@pytest.mark.parametrize(("family_name", "family_cls"), FAMILIES)
def test_glmboostlss_path_matches_gamboostlss(family_name, family_cls):
    expected = _r_fit(family_name)
    model = mb.glmboostLSS(
        "y ~ x1 + x2",
        data={name: expected[name] for name in ("y", "x1", "x2")},
        families=family_cls(),
        control=mb.boost_control(20, 0.05),
    )
    for parameter in PARAM_NAMES:
        np.testing.assert_allclose(
            model.predict(parameter=parameter, type="link"),
            expected[parameter],
            rtol=2e-10,
            atol=2e-11,
            err_msg=f"{family_name} {parameter} link-scale path differs from R",
        )
