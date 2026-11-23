# glmboostLSS: boosting for GAMLSS (location/scale/shape).
# Thin R wrapper over the shared Rust core (mboostrs_core::lss::fit_lss), via the
# extendr `fit_glmboostlss` binding. Mirrors python/mboostrs/gamlss.py.

# ─── Shared LSS entry-point helpers ──────────────────────────────────
#
# The six LSS entry points (fit, cvrisk, stabsel — each in a glmboostLSS and a
# gamboostLSS spelling) differ in their middles: the linear ones build a design
# matrix inline, the additive ones go through `.gamboostlss_frontend`. Their
# heads and tails do not, and the tail in particular is the subtle part.

# Resolve the `families=` / `family=` pair, which every LSS entry point accepts
# for R compatibility. Six verbatim copies of this before.
.lss_resolve_families <- function(families, family) {
  if (!is.null(families) && !is.null(family)) {
    stop("Specify family= or families=, not both")
  }
  if (!is.null(family)) {
    return(family)
  }
  if (is.null(families)) {
    return(GaussianLSS())
  }
  families
}

# Per-learner selection frequencies from stability-selection subsample runs.
#
# `seqs` is one selection sequence per subsample. Each is truncated at the
# point where the number of *distinct* learners selected would exceed `q`,
# matching `stabs`: everything after that point is charged to the first
# learner so the sequence keeps its length without adding new selections.
# `phat[l, m]` is then the fraction of subsamples in which learner `l` had been
# selected by iteration `m`, and it is monotone in `m` by construction.
#
# The core computes it (`mboostrs_core::stabsel::try_selection_frequencies`).
.lss_stabsel_frequencies <- function(seqs, p, m, q_val) {
  phat <- .Call("wrap__stabsel_frequencies_r", lapply(seqs, as.integer),
                as.integer(p), as.integer(m), as.integer(q_val), PACKAGE = "mboostrs")
  matrix(phat, nrow = p, ncol = m)
}

#' Gaussian location-scale family
#'
#' This family fits predictors for both the conditional mean and standard
#' deviation. `mu` uses the identity link and `sigma` the log link. A
#' [Gaussian()] fit assumes constant spread; the location-scale model allows
#' the standard deviation to vary with the predictors.
#'
#' Matches `gamboostLSS::GaussianLSS()`.
#'
#' @return An LSS family object (list with class `mboostrs_lss_family`) for
#'   use with `glmboostLSS()` or `gamboostLSS()`.
#' @param stabilization How each negative gradient is scaled before its
#'   step, as upstream `gamboostLSS`'s family argument: `"none"` (default),
#'   `"MAD"` divides by the weighted median absolute deviation, `"L2"` by the
#'   weighted root mean square. It keeps the parameters' steps on comparable
#'   scales when the gradients' magnitudes differ widely.
#' @seealso [glmboostLSS()], [gamboostLSS()], [GaussianKnownVarLSS()]
#' @examples
#' set.seed(1)
#' n <- 200
#' x <- rnorm(n)
#' y <- rnorm(n, mean = 1 + x, sd = exp(0.3 * x))
#' d <- data.frame(y = y, x = x)
#' fit <- glmboostLSS(y ~ x, data = d, families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' coef(fit)$sigma
#' @export
GaussianLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "GaussianLSS", rust_name = "gaussianlss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' GammaLSS family (mu log link = mean, sigma log link = shape). Requires y > 0.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rgamma(n, 2, 1), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = GammaLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
GammaLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "GammaLSS", rust_name = "gammalss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' StudentTLSS family (mu identity link, sigma + df log links).
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rt(n, df = 5), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = StudentTLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
StudentTLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "StudentTLSS", rust_name = "studenttlss",
         param_names = c("mu", "sigma", "df")),
    class = "mboostrs_lss_family"
  )
}

#' BetaLSS family (mu logit link = mean, phi log link = precision). Requires
#' 0 < y < 1.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rbeta(n, 2, 5), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = BetaLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
BetaLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "BetaLSS", rust_name = "betalss",
         param_names = c("mu", "phi")),
    class = "mboostrs_lss_family"
  )
}

#' NBinomialLSS family (mu log link = mean, sigma log link = size). Counts.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnbinom(n, mu = 5, size = 2), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = NBinomialLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
NBinomialLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "NBinomialLSS", rust_name = "nbinomiallss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' ZIPoLSS family: zero-inflated Poisson. mu (log link, Poisson mean), sigma
#' (logit link, zero-inflation probability). Matches R \code{gamboostLSS::ZIPoLSS()}.
#' Requires non-negative integer counts.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rpois(n, lambda = 3), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = ZIPoLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
ZIPoLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "ZIPoLSS", rust_name = "zipolss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' DirichletLSS family: Dirichlet model for a compositional response, given as
#' \code{cbind(y1, ..., yK) ~ ...}. K concentration parameters
#' \code{alpha_k = exp(eta_k)} (log links). Matches R
#' \code{gamboostLSS::DirichletLSS(K)}. Requires \code{0 < y < 1}.
#' @param K Number of categories (2..20).
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(
#'   y1 = runif(n), y2 = runif(n), y3 = runif(n),
#'   x1 = rnorm(n), x2 = rnorm(n)
#' )
#' dat[, c("y1", "y2", "y3")] <- dat[, c("y1", "y2", "y3")] / rowSums(dat[, c("y1", "y2", "y3")])
#' mod <- glmboostLSS(cbind(y1, y2, y3) ~ x1 + x2, data = dat,
#'                    families = DirichletLSS(K = 3),
#'                    control = boost_control(mstop = 50))
#' @export
DirichletLSS <- function(K) {
  if (!(K >= 2 && K <= 20)) stop("DirichletLSS supports K in 2..20")
  structure(
    list(name = "DirichletLSS", rust_name = "dirichletlss",
         param_names = paste0("a", seq_len(K)), K = K),
    class = "mboostrs_lss_family"
  )
}

#' Normal location-scale family for outcomes carrying a known sampling variance
#'
#' `v_i = s2_i + exp(eta_sigma_i)`: the known sampling variance plus a boosted
#' *excess* variance component. Made for regressing published small-area
#' estimates whose standard errors vary by orders of magnitude across areas,
#' where a plain [GaussianLSS()] mistakes sampling noise for heterogeneity and
#' its sigma-submodel selects proxies for sample size.
#'
#' The response is a two-column `cbind(outcome, s2)`, with `s2` the squared
#' standard error -- the same shape the survival families take for
#' `Surv(time, status)`. Both submodels are boosted: `mu` carries its own
#' additive predictor as in [GaussianLSS()], and `eta_sigma` models only
#' the *unknown excess* variance on top of the known `s2`. The gradient
#' toward `eta_sigma` is attenuated by `exp(eta_sigma) / v` wherever the
#' known component dominates, which is the correction.
#'
#' **No `gamboostLSS` counterpart.** This family is an mboostrs addition, not
#' a reimplementation of an upstream one, so there is no oracle to check it
#' against and no upstream behaviour to match.
#'
#' @return A distributional family object for [glmboostLSS()] / [gamboostLSS()].
#' @seealso [GaussianLSS()]
#' @examples
#' set.seed(1)
#' n <- 100
#' x <- rnorm(n)
#' s2 <- runif(n, 0.05, 0.5)
#' y <- rnorm(n, 1 + x, sqrt(s2 + 0.25))
#' d <- data.frame(y = y, s2 = s2, x = x)
#' fit <- glmboostLSS(cbind(y, s2) ~ x, data = d,
#'                    families = GaussianKnownVarLSS(),
#'                    control = boost_control(mstop = 30))
#' coef(fit)$mu
#' @inheritParams GaussianLSS
#' @export
GaussianKnownVarLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "GaussianKnownVarLSS", rust_name = "gaussianknownvarlss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' Zero-inflated negative binomial family
#'
#' Type I: `mu` on the log link is the mean of the count component, `sigma` on
#' the log link its dispersion, and `nu` on the logit link the probability of
#' an excess zero -- a zero that comes from the inflation rather than from the
#' count distribution. Requires non-negative integer counts.
#'
#' Matches `gamboostLSS::ZINBLSS()`.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnbinom(n, mu = 5, size = 2), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = ZINBLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
ZINBLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "ZINBLSS", rust_name = "zinblss",
         param_names = c("mu", "sigma", "nu")),
    class = "mboostrs_lss_family"
  )
}

#' PoissonLSS family: Poisson model with mu log link.
#' @return An LSS family object for use with \code{glmboostLSS()} or \code{gamboostLSS()}.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rpois(n, lambda = 3), x = rnorm(n))
#' mod <- glmboostLSS(y ~ x, data = dat, families = PoissonLSS(),
#'                    control = boost_control(mstop = 50))
#' @inheritParams GaussianLSS
#' @export
PoissonLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "PoissonLSS", rust_name = "poissonlss",
         param_names = c("mu")),
    class = "mboostrs_lss_family"
  )
}

#' LogNormalLSS family: log-normal accelerated-failure-time model with a
#' right-censored \code{Surv(time, status)} response. mu (identity link,
#' log-location), sigma (log link, scale).
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' \dontrun{
#' library(survival)
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(
#'   time = rexp(n), status = rbinom(n, 1, 0.8),
#'   x1 = rnorm(n), x2 = rnorm(n)
#' )
#' mod <- glmboostLSS(Surv(time, status) ~ x1 + x2, data = dat,
#'                    families = LogNormalLSS(),
#'                    control = boost_control(mstop = 50))
#' }
#' @inheritParams GaussianLSS
#' @export
LogNormalLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "LogNormalLSS", rust_name = "lognormallss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' LogLogLSS family: log-logistic accelerated-failure-time model with a
#' right-censored \code{Surv(time, status)} response. mu (identity link,
#' log-location), sigma (log link, scale). The automatic sigma offset
#' minimizes the log-logistic negative log-likelihood. This differs from
#' gamboostLSS, which uses a standard-normal survival loss for this offset.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' \dontrun{
#' library(survival)
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(
#'   time = rexp(n), status = rbinom(n, 1, 0.8),
#'   x1 = rnorm(n), x2 = rnorm(n)
#' )
#' mod <- glmboostLSS(Surv(time, status) ~ x1 + x2, data = dat,
#'                    families = LogLogLSS(),
#'                    control = boost_control(mstop = 50))
#' }
#' @inheritParams GaussianLSS
#' @export
LogLogLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "LogLogLSS", rust_name = "logloglss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' WeibullLSS family: Weibull accelerated-failure-time model with a
#' right-censored \code{Surv(time, status)} response. mu (identity link,
#' log-location), sigma (log link, scale). The automatic mu offset is the
#' identity-link location that minimizes the Weibull negative log-likelihood.
#' This differs from gamboostLSS, which stores the logarithm of that location.
#' @return An LSS family object (list with class `mboostrs_lss_family`) for use with `glmboostLSS()` or `gamboostLSS()`.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' \dontrun{
#' library(survival)
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(
#'   time = rexp(n), status = rbinom(n, 1, 0.8),
#'   x1 = rnorm(n), x2 = rnorm(n)
#' )
#' mod <- glmboostLSS(Surv(time, status) ~ x1 + x2, data = dat,
#'                    families = WeibullLSS(),
#'                    control = boost_control(mstop = 50))
#' }
#' @inheritParams GaussianLSS
#' @export
WeibullLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "WeibullLSS", rust_name = "weibulllss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' NegativeBinomialLSS family (mu log link = mean, sigma log link = size).
#' Counts.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
NegativeBinomialLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "NegativeBinomialLSS", rust_name = "negativebinomiallss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' GeometricLSS family (mu log link = mean). Counts.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
GeometricLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "GeometricLSS", rust_name = "geometriclss",
         param_names = c("mu")),
    class = "mboostrs_lss_family"
  )
}

#' BinomialLSS family (mu logit link = probability). Requires y in \[0, 1\].
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
BinomialLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "BinomialLSS", rust_name = "binomiallss",
         param_names = c("mu")),
    class = "mboostrs_lss_family"
  )
}

#' GumbelLSS family (mu identity link = location, sigma log link = scale).
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
GumbelLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "GumbelLSS", rust_name = "gumbellss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' ZeroAdjustedGammaLSS family (mu log = mean, sigma log = dispersion,
#' nu logit = probability of an exact zero). Requires y >= 0.
#' Matches `gamlss.dist::ZAGA()`.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
ZeroAdjustedGammaLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "ZeroAdjustedGammaLSS", rust_name = "zeroadjustedgammalss",
         param_names = c("mu", "sigma", "nu")),
    class = "mboostrs_lss_family"
  )
}

#' Inverse Gaussian location-scale family
#'
#' `mu` on the log link is the mean and `sigma` on the log link the
#' dispersion. For positive continuous responses whose variance grows with the
#' cube of the mean, which is steeper than the Gamma's square. Requires
#' `y > 0`.
#'
#' Matches `gamlss.dist::IG()`.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
InverseGaussianLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "InverseGaussianLSS", rust_name = "inversegaussianlss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' Logistic location-scale family
#'
#' `mu` on the identity link is the location and `sigma` on the log link the
#' scale. Symmetric like the Gaussian and heavier in the tails, so it absorbs
#' moderate outliers that would otherwise pull a Gaussian mean. `y` is
#' unrestricted.
#'
#' Matches `gamlss.dist::LO()`.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
LogisticLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "LogisticLSS", rust_name = "logisticlss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' Pareto2LSS family (mu log link = location, sigma log link = scale).
#' Requires y > 0.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
Pareto2LSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "Pareto2LSS", rust_name = "pareto2lss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' ZeroAdjustedPoissonLSS family (mu log link = mean, sigma logit link).
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
ZeroAdjustedPoissonLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "ZeroAdjustedPoissonLSS", rust_name = "zeroadjustedpoissonlss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' PIGLSS family (mu log link = mean, sigma log link = dispersion). Counts.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
PIGLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "PIGLSS", rust_name = "piglss",
         param_names = c("mu", "sigma")),
    class = "mboostrs_lss_family"
  )
}

#' BCCGLSS family (mu identity link = median, sigma log link = scale,
#' nu identity link = Box-Cox power). Requires y > 0.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
BCCGLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "BCCGLSS", rust_name = "bccglss",
         param_names = c("mu", "sigma", "nu")),
    class = "mboostrs_lss_family"
  )
}

#' JSULSS family (mu identity, sigma log, nu identity, tau log). 4 parameters.
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
JSULSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "JSULSS", rust_name = "jsulss",
         param_names = c("mu", "sigma", "nu", "tau")),
    class = "mboostrs_lss_family"
  )
}

#' BCTLSS family (Box-Cox t; mu identity, sigma log, nu identity, tau log).
#' 4 parameters. Requires y > 0. Matches gamlss.dist BCT().
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
BCTLSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "BCTLSS", rust_name = "bctlss",
         param_names = c("mu", "sigma", "nu", "tau")),
    class = "mboostrs_lss_family"
  )
}

#' BCPELSS family (Box-Cox Power Exponential; mu identity, sigma log, nu
#' identity, tau log). 4 parameters. Requires y > 0. Matches gamlss.dist BCPE().
#' @return Family list.
#' @inheritParams GaussianLSS
#' @export
BCPELSS <- function(stabilization = c("none", "MAD", "L2")) {
  structure(
    list(stabilization = match.arg(stabilization), name = "BCPELSS", rust_name = "bcpelss",
         param_names = c("mu", "sigma", "nu", "tau")),
    class = "mboostrs_lss_family"
  )
}

# Inverse link for parameter `parameter` of an LSS family.
.lss_response <- function(eta, family, parameter) {
  if (family$rust_name == "gaussianlss") {
    if (parameter == "mu") return(eta)        # identity
    if (parameter == "sigma") return(exp(eta)) # log link
  }
  if (family$rust_name == "gammalss") {
    return(exp(eta))  # mu and sigma both use the log link
  }
  if (family$rust_name == "studenttlss") {
    if (parameter == "mu") return(eta)        # identity
    return(exp(eta))                          # sigma, df: log link
  }
  if (family$rust_name == "betalss") {
    if (parameter == "mu") return(plogis(eta)) # logit link
    return(exp(eta))                           # phi: log link
  }
  if (family$rust_name == "nbinomiallss") {
    return(exp(eta))  # mu and sigma both use the log link
  }
  if (family$rust_name == "negativebinomiallss") {
    return(exp(eta))  # mu and sigma both use the log link
  }
  if (family$rust_name == "geometriclss") {
    return(exp(eta))  # mu: log link
  }
  if (family$rust_name == "binomiallss") {
    return(plogis(eta))  # mu: logit link
  }
  if (family$rust_name == "gumbellss") {
    if (parameter == "mu") return(eta)
    return(exp(eta))
  }
  if (family$rust_name == "pareto2lss") {
    return(exp(eta))
  }
  if (family$rust_name == "inversegaussianlss") {
    return(exp(eta))  # mu and sigma both use the log link
  }
  if (family$rust_name == "logisticlss") {
    if (parameter == "mu") return(eta)        # identity
    return(exp(eta))                          # sigma: log link
  }
  if (family$rust_name == "zeroadjustedgammalss") {
    # mu, sigma log; nu logit (the probability of an exact zero). Clamp the
    # logit the same way zeroadjustedpoissonlss does, so an extreme eta cannot
    # return exactly 0 or 1 and make downstream log() infinite.
    if (parameter %in% c("mu", "sigma")) return(exp(eta))
    thresh <- -qlogis(.Machine$double.eps)
    return(plogis(pmin(thresh, pmax(eta, -thresh))))
  }
  if (family$rust_name == "zeroadjustedpoissonlss") {
    if (parameter == "mu") return(pmax(exp(eta), .Machine$double.eps))
    thresh <- -qlogis(.Machine$double.eps)
    return(plogis(pmin(thresh, pmax(eta, -thresh))))
  }
  if (family$rust_name == "piglss") {
    return(pmax(exp(eta), .Machine$double.eps))
  }
  if (family$rust_name == "bccglss") {
    if (parameter %in% c("mu", "nu")) return(eta)
    return(exp(eta))
  }
  if (family$rust_name == "jsulss") {
    if (parameter %in% c("mu", "nu")) return(eta)
    return(exp(eta))
  }
  if (family$rust_name %in% c("bctlss", "bcpelss")) {
    if (parameter %in% c("mu", "nu")) return(eta)
    return(exp(eta))
  }
  if (family$rust_name == "zipolss") {
    if (parameter == "mu") return(pmax(exp(eta), .Machine$double.eps))  # log link
    thresh <- -qlogis(.Machine$double.eps)                             # logit link
    return(plogis(pmin(thresh, pmax(eta, -thresh))))
  }
  if (family$rust_name == "zinblss") {
    if (parameter %in% c("mu", "sigma")) return(pmax(exp(eta), .Machine$double.eps))  # log
    thresh <- -qlogis(.Machine$double.eps)                                            # logit (nu)
    return(plogis(pmin(thresh, pmax(eta, -thresh))))
  }
  if (family$rust_name == "dirichletlss") {
    return(exp(eta))  # all alpha_k use the log link
  }
  if (family$rust_name == "poissonlss") {
    return(exp(eta))
  }
  if (family$rust_name %in% c("lognormallss", "logloglss", "weibulllss")) {
    if (parameter == "mu") return(eta)        # identity
    return(exp(eta))                          # sigma: log link
  }
  stop(sprintf("unknown LSS family/parameter: %s/%s", family$rust_name, parameter))
}

.lss_param_value <- function(params, name) {
  value <- params[[name]]
  if (is.null(value)) stop("missing LSS parameter '", name, "'")
  as.double(value)
}

#' LSS conditional response mean
#' @param family An LSS family object.
#' @param params Named list of response-scale parameters.
#' @param ... Not used.
#' @return Numeric conditional response mean. Families whose mean is undefined
#'   for the supplied parameters, or whose parameter named `mu` is not a mean,
#'   raise an error.
#' @export
lss_mean <- function(family, params, ...) {
  .lss_core("mean", family, NULL, params)
}

# Evaluate an LSS distribution helper of the core
# (`mboostrs_core::lss_distribution`). `first` is the response or probability
# column, NULL for the mean; the
# family's parameters follow in `param_names` order; BinomialLSS takes one
# trial and GaussianKnownVarLSS its known variance `s2` after them, except
# for the mean. The columns are recycled to a common length, as R's d/p/q functions
# recycle their arguments, and a zero-length column gives a zero-length result.
.lss_core <- function(kind, family, first, params) {
  cols <- lapply(family$param_names, function(pn) .lss_param_value(params, pn))
  if (!is.null(first)) cols <- c(list(as.double(first)), cols)
  if (kind != "mean" && identical(family$name, "BinomialLSS")) cols <- c(cols, list(1))
  if (kind != "mean" && identical(family$name, "GaussianKnownVarLSS")) {
    cols <- c(cols, list(.lss_param_value(params, "s2")))
  }
  lens <- lengths(cols)
  n <- if (any(lens == 0L)) 0L else max(lens)
  cols <- lapply(cols, rep_len, length.out = n)
  out <- .Call("wrap__r_lss_distribution", kind, family$name,
               as.double(unlist(cols, use.names = FALSE)), as.integer(n),
               as.integer(length(cols)), PACKAGE = "mboostrs")
  if (out$unconverged > 0L) {
    warning(sprintf(
      "%s %s: %d of %d value(s) reached the iteration limit before converging; they may be inaccurate.",
      family$name, kind, out$unconverged, n
    ), call. = FALSE)
  }
  out$values
}

#' LSS probability density or mass
#' @param family An LSS family object.
#' @param y Response values.
#' @param params Named list of response-scale parameters.
#' @param ... Not used.
#' @return Numeric density/mass vector.
#' @export
lss_pdf <- function(family, y, params, ...) {
  .lss_core("pdf", family, y, params)
}

#' LSS cumulative distribution function
#' @inheritParams lss_pdf
#' @return Numeric CDF vector.
#' @export
lss_cdf <- function(family, y, params, ...) {
  .lss_core("cdf", family, y, params)
}

#' LSS quantile function
#' @param p Probability values.
#' @inheritParams lss_pdf
#' @return Numeric quantile vector.
#' @export
lss_quantile <- function(family, p, params, ...) {
  p <- as.double(p)
  if (any(p <= 0.0 | p >= 1.0)) stop("LSS quantile probabilities must be in (0, 1)")
  .lss_core("quantile", family, p, params)
}

#' Simulate from an LSS family
#' @inheritParams lss_pdf
#' @param n Optional number of samples. Defaults to the broadcast parameter length.
#' @return Numeric sample vector.
#' @export
lss_simulate <- function(family, params, n = NULL, ...) {
  mu <- .lss_param_value(params, "mu")
  if (identical(family$rust_name, "poissonlss")) {
    if (is.null(n)) n <- length(mu)
    return(stats::rpois(n, lambda = rep_len(mu, n)))
  }
  if (identical(family$rust_name, "binomiallss")) {
    if (is.null(n)) n <- length(mu)
    return(stats::rbinom(n, size = 1, prob = rep_len(mu, n)))
  }
  if (identical(family$rust_name, "geometriclss")) {
    if (is.null(n)) n <- length(mu)
    prob <- rep_len(1.0 / (1.0 + mu), n)
    return(stats::rgeom(n, prob = prob))
  }
  if (identical(family$rust_name, "studenttlss")) {
    sigma <- .lss_param_value(params, "sigma")
    df <- .lss_param_value(params, "df")
    if (is.null(n)) n <- max(length(mu), length(sigma))
    return(rep_len(mu, n) + rep_len(sigma, n) *
           stats::rt(n, df = rep_len(df, n)))
  }
  sigma_name <- if (identical(family$rust_name, "betalss")) "phi" else "sigma"
  sigma <- .lss_param_value(params, sigma_name)
  if (is.null(n)) n <- max(length(mu), length(sigma))
  switch(family$rust_name,
    gaussianlss = stats::rnorm(n, mean = rep_len(mu, n), sd = rep_len(sigma, n)),
    gammalss = stats::rgamma(n, shape = rep_len(sigma, n), scale = rep_len(mu / sigma, n)),
    betalss = stats::rbeta(n, shape1 = rep_len(mu * sigma, n),
                           shape2 = rep_len((1.0 - mu) * sigma, n)),
    nbinomiallss = stats::rnbinom(n, mu = rep_len(mu, n), size = rep_len(sigma, n)),
    negativebinomiallss = stats::rnbinom(n, mu = rep_len(mu, n), size = rep_len(1 / sigma, n)),
    zipolss = {
      sigma <- .lss_param_value(params, "sigma")
      if (is.null(n)) n <- max(length(mu), length(sigma))
      mask <- stats::runif(n) < rep_len(sigma, n)
      ifelse(mask, 0.0, stats::rpois(n, lambda = rep_len(mu, n)))
    },
    zinblss = {
      sigma <- .lss_param_value(params, "sigma")
      nu <- .lss_param_value(params, "nu")
      if (is.null(n)) n <- max(length(mu), length(sigma))
      mask <- stats::runif(n) < rep_len(nu, n)
      ifelse(mask, 0.0,
             stats::rnbinom(n, mu = rep_len(mu, n), size = rep_len(1 / sigma, n)))
    },
    # Inverse-CDF draws: R's uniforms through the core quantile.
    bctlss = ,
    bcpelss = {
      if (is.null(n)) n <- max(length(mu), length(sigma))
      lss_quantile(family, stats::runif(n), lapply(params, rep_len, n))
    },
    stop(family$name, " simulation helper is not implemented yet")
  )
}

#' Per-observation LSS deviance contribution
#' @inheritParams lss_pdf
#' @return Numeric vector equal to \code{-2 * log(pdf(y))}.
#' @export
lss_deviance_increment <- function(family, y, params, ...) {
  .lss_core("deviance", family, y, params)
}

#' Fit a linear GAMLSS by cyclic component-wise boosting (glmboostLSS).
#'
#' @param formula One formula shared by all parameters, or a named list
#'   \code{list(mu = ..., sigma = ...)} for per-parameter predictors.
#' @param data A data frame.
#' @param families An LSS family (default \code{GaussianLSS()}).
#' @param family Alias for `families`; specify at most one of them.
#' @param control A \code{boost_control()} object (scalar mstop, nu).
#' @param offset Optional numeric vector of per-parameter offset overrides.
#' @param mstop Optional per-parameter step budget (R's mstop vector): a scalar,
#'   an unnamed vector (family order), or a named vector \code{c(mu=, sigma=)}.
#' @param weights Optional observation weights, one per row, non-negative.
#'   Integer weights are frequencies -- weight 3 stands for three identical
#'   rows -- and are used at the scale given; fractional weights are rescaled
#'   (see `weight_policy`). A zero weight excludes a row while keeping its
#'   position. An all-zero vector is rejected: the fit it produces reports a
#'   completed path over an empty sample.
#' @param weight_policy How the scale of `weights` is settled before fitting.
#'   `"upstream"` (the default) reproduces `mboost`'s `rescale_weights`:
#'   integer weights are frequencies and pass through, fractional weights are
#'   analytic and are rescaled to sum to the number of weight-positive rows.
#'   `"raw"` passes the vector through unchanged, for a caller whose weight
#'   scale is itself meaningful (survey weights summing to a population total,
#'   for example). The scale is not cosmetic -- it enters `X'WX`, the
#'   `df -> lambda` calibration, and the risk -- so `"raw"` gives a different
#'   path from R's on the same fractional weights.
#' @param method Boosting method: "cyclic" (default) or "noncyclic".
#' @details Gradient stabilization and per-parameter step lengths follow
#'   upstream `gamboostLSS` 2.2-0. Stabilization is a family argument,
#'   `GaussianLSS(stabilization = c("none", "MAD", "L2"))`, which scales each
#'   negative gradient before its step. Per-parameter step lengths come from
#'   `boost_control(nu = c(mu = 0.1, sigma = 0.05))`: a named vector is
#'   reordered to the family's parameters, an unnamed one taken in family
#'   order, and a scalar applies to every parameter. Both are carried into
#'   `cvrisk` and `stabsel`.
#' @return An object of class \code{"mboostrs_glmboostLSS"}.
#' @seealso [gamboostLSS()], [GaussianLSS()], [predict.mboostrs_glmboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' head(predict(mod, type = "response"))
#' coef(mod)
#' @export
glmboostLSS <- function(formula, data, families = NULL, family = NULL,
                        control = boost_control(), offset = NULL, mstop = NULL,
                        weights = NULL,
                        method = c("cyclic", "noncyclic"),
                        weight_policy = "upstream") {
  families <- .lss_resolve_families(families, family)
  method <- match.arg(method)
  cl <- match.call()
  pnames <- families$param_names
  K <- length(pnames)

  if (is.list(formula) && !inherits(formula, "formula")) {
    formulas <- lapply(pnames, function(p) formula[[p]])
  } else {
    formulas <- rep(list(formula), K)
  }
  names(formulas) <- pnames

  is_survival <- families$rust_name %in% c("lognormallss", "logloglss", "weibulllss")
  is_multivariate <- families$rust_name %in% c("dirichletlss", "gaussianknownvarlss")
  y <- NULL
  n <- NULL
  blocks <- vector("list", K)
  param_ncols <- integer(K)
  cmeans <- vector("list", K)
  for (k in seq_len(K)) {
    parsed <- .parse_formula(formulas[[k]], data, has_survival = is_survival, multivariate = is_multivariate)
    if (is.null(y)) { y <- parsed$y; n <- parsed$n }
    xk <- parsed$x_matrix
    param_ncols[k] <- ncol(xk)
    # gamboostLSS centers every covariate by its simple column mean; predict()
    # and the [ slice re-apply the same means, so storing colMeans keeps
    # fitted/predict consistent even for weighted fits.
    cmeans[[k]] <- colMeans(xk)
    blocks[[k]] <- xk
  }
  x_all <- do.call(cbind, blocks)
  if (is.null(x_all)) x_all <- matrix(0.0, n, 0)

  normalized <- .normalize_weights(weights, n, weight_policy)
  w <- normalized$fit
  mstop_pp <- integer(0)
  if (!is.null(mstop)) {
    if (is.null(names(mstop))) {
      mstop_pp <- as.integer(rep_len(mstop, K))
    } else {
      mstop_pp <- as.integer(mstop[pnames])
    }
  }
  offs <- if (is.null(offset)) numeric(0) else as.double(offset)

  if (any(!is.finite(y))) stop("y contains NA, NaN, or Inf values")
  if (any(!is.finite(x_all))) stop("design contains NA, NaN, or Inf values")

  res <- if (method == "noncyclic") {
    .Call("wrap__fit_glmboostlss_noncyclic",
      as.double(t(x_all)), as.integer(n), as.integer(param_ncols),
      as.double(y), as.double(w), as.integer(control$mstop), as.double(.lss_nu_scalar(control$nu)),
      families$rust_name, offs, control$trace %||% FALSE, .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
      control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
 PACKAGE = "mboostrs")
  } else {
    .Call("wrap__fit_glmboostlss",
      as.double(t(x_all)),  # transpose: R is col-major, Rust reads row-major
      as.integer(n), as.integer(param_ncols), as.double(y), as.double(w),
      as.integer(control$mstop), as.double(.lss_nu_scalar(control$nu)),
      families$rust_name, offs, mstop_pp, control$trace %||% FALSE,
      .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
      control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
      PACKAGE = "mboostrs")
  }
  if (isTRUE(control$trace)) .render_execution_report(res$execution_report)

  params <- lapply(seq_len(K), function(k) {
    r <- res$results[[k]]
    list(offset = r$offset, selected = r$selected, coefs = r$coefs,
         fitted = r$fitted, centering_means = cmeans[[k]],
         risk = as.numeric(res$param_risk[[k]]))
  })
  names(params) <- pnames

  structure(
    list(
      params = params,
      risk = res$risk,
      selection_order = res$selection_order,
      execution_report = res$execution_report,
      nu = control$nu,
      family = families,
      formulas = formulas,
      param_names = pnames,
      data = data,
      n = n,
      weights = w,
      control = control,
      offset = offset,
      method = method,
      call = cl
    ),
    class = "mboostrs_glmboostLSS"
  )
}

#' Predict from a fitted glmboostLSS model
#'
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param newdata Optional new data frame (default: training data).
#' @param parameter Parameter name (e.g. "mu"); NULL returns all as a named list.
#' @param mstop Iteration count to predict at (per parameter); NULL = full model.
#' @param type "link" or "response".
#' @param ... Not used.
#' @return A named list of numeric vectors (one per parameter), or a single vector if `parameter` is specified.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' head(predict(mod, type = "response"))
#' head(predict(mod, parameter = "mu", type = "response"))
#' @export
predict.mboostrs_glmboostLSS <- function(object, newdata = NULL, parameter = NULL,
                                         mstop = NULL, type = c("link", "response"),
                                         ...) {
  .refuse_unused("predict", ...)
  type <- match.arg(type)
  mstop_val <- if (is.null(mstop)) 0L else as.integer(mstop)
  want <- if (is.null(parameter)) object$param_names else parameter

  out <- list()
  for (k in seq_along(object$param_names)) {
    pn <- object$param_names[k]
    if (!(pn %in% want)) next
    p <- object$params[[pn]]
    src <- if (is.null(newdata)) object$data else newdata
    parsed <- .parse_formula(object$formulas[[k]], src, has_survival = FALSE,
                             need_response = FALSE)
    xk <- parsed$x_matrix
    eta <- .Call("wrap__predict_glmboost",
      p$offset, p$selected, p$coefs, .lss_param_nu(object, pn),
      as.double(t(xk)),
      as.double(p$centering_means),
      mstop_val,
      PACKAGE = "mboostrs"
    )
    if (type == "response") eta <- .lss_response(eta, object$family, pn)
    out[[pn]] <- eta
  }
  if (!is.null(parameter)) return(out[[parameter]])
  out
}

#' Print method for glmboostLSS
#' @param x A fitted \code{mboostrs_glmboostLSS} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x = rnorm(n))
#' mod <- glmboostLSS(y ~ x, data = dat, families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' print(mod)
#' @export
print.mboostrs_glmboostLSS <- function(x, ...) {
  cat("glmboostLSS model\n")
  cat("  Family:", x$family$name, "\n")
  cat("  Parameters:", paste(x$param_names, collapse = ", "), "\n")
  cat("  mstop:", paste(sprintf("%s=%d", x$param_names,
        vapply(x$params, function(p) length(p$selected), integer(1))),
        collapse = ", "), "\n")
  invisible(x)
}

# Legacy BBS-only parser retained for compatibility with saved calls. New fits
# use `.parse_gamboost_formula()` so ordinary gamboost and LSS normalize learner
# options through the same R frontend.
.parse_bbs_formula <- function(formula) {
  tf <- terms(formula)
  labels <- attr(tf, "term.labels")
  lapply(labels, function(label) {
    m <- regmatches(label, regexec("^bbs\\((.+)\\)$", label))[[1]]
    if (length(m) != 2)
      stop("gamboostLSS supports only bbs() base-learners; got: ", label)
    args <- .parse_bl_args(m[2])
    `%or%` <- function(a, b) if (is.null(a)) b else a
    list(
      name = args$positional[1],
      knots = as.integer(args$named$knots %or% 20),
      degree = as.integer(args$named$degree %or% 3),
      differences = as.integer(args$named$differences %or% 2),
      df = as.double(args$named$df %or% 4),
      center = isTRUE(tolower(args$named$center %or% "false") %in% c("true", "1"))
    )
  })
}

.gamboostlss_frontend <- function(formula, data, families, weights = NULL,
                                 weight_policy = "upstream") {
  pnames <- families$param_names
  formulas <- if (is.list(formula) && !inherits(formula, "formula")) {
    missing <- setdiff(pnames, names(formula))
    if (length(missing)) {
      stop("formula list must contain: ", paste(pnames, collapse = ", "))
    }
    lapply(pnames, function(parameter) formula[[parameter]])
  } else {
    rep(list(formula), length(pnames))
  }
  names(formulas) <- pnames
  response_terms <- lapply(formulas, function(value) all.vars(value[[2L]]))
  if (!all(vapply(response_terms[-1L], identical, logical(1), response_terms[[1L]]))) {
    stop("all parameter formulas must use the same response")
  }

  parsed <- lapply(formulas, .parse_gamboost_formula, data = data)
  specs <- lapply(parsed, function(part) lapply(part$learner_specs, .lss_by_spec))
  unsupported <- unique(unlist(lapply(specs, function(parameter_specs) {
    vapply(parameter_specs, function(spec) {
      if (spec$type %in% c("scalar", "bbs", "bns", "brandom", "bmrf", "bspatial", "bmono", "bconstrained", "tree", "bkernel", "buser")) "" else spec$type
    }, character(1))
  })))
  unsupported <- unsupported[nzchar(unsupported)]
  if (length(unsupported)) {
    stop("gamboostLSS currently supports bols(), bbs(), bns(), brandom(), bmrf(), bspatial(), bmono(), bconstrained(), btree(), bkernel(), and buser() base-learners; got: ",
         paste(unsupported, collapse = ", "))
  }

  is_survival <- families$rust_name %in% c("lognormallss", "logloglss", "weibulllss")
  is_multivariate <- families$rust_name %in% c("dirichletlss", "gaussianknownvarlss")
  if (is_survival) {
    vars <- response_terms[[1L]]
    y <- c(as.double(data[[vars[1L]]]), as.double(data[[vars[2L]]]))
    n <- length(data[[vars[1L]]])
  } else if (is_multivariate) {
    vars <- response_terms[[1L]]
    y <- as.double(unlist(lapply(vars, function(value) as.double(data[[value]]))))
    n <- length(data[[vars[1L]]])
  } else {
    y <- as.double(data[[response_terms[[1L]][1L]]])
    n <- length(y)
  }
  normalized <- .normalize_weights(weights, n, weight_policy)
  w <- normalized$fit
  if (any(!is.finite(y))) stop("y contains NA, NaN, or Inf values")
  list(formulas = formulas, specs = specs, y = y, n = n, weights = w,
       normalized = normalized)
}

# Learner specs for the distributional fit. A numeric bbs(x, by = z) keeps its
# `by_values`, which the binding's bbs arm applies. A `block` spec (a numeric
# bols by, a factor by on bbs or bols, and a plain bols in the block form) is
# an explicit design with a penalty; the distributional binding reads that
# shape as `buser`, whose penalty is taken as given or calibrated from `df`
# against the fit weights.
.lss_by_spec <- function(spec) {
  # A plain bols beside a by= learner is a block too: the menu switches to
  # the block form [1, x] (see .parse_gamboost_formula).
  if (identical(spec$type, "brandom") && !is.null(spec$by_name)) {
    # brandom(g, by = z): the indicator design scaled by z, which the
    # distributional binding's brandom arm, building indicators from the
    # levels, would drop. Sent as buser with an identity penalty and df.
    n <- spec$dims[1L]
    p <- spec$dims[2L]
    lambda <- spec$lambda %||% -1
    return(list(type = "buser", name = spec$name,
                X = matrix(spec$design, nrow = n, ncol = p, byrow = TRUE),
                dims = as.integer(c(n, p)), K = diag(p), K_dims = as.integer(c(p, p)),
                df = as.double(spec$df), lambda = as.double(lambda), center = FALSE,
                block = list(name = spec$name, by_name = spec$by_name,
                             brandom_levels = as.integer(p),
                             brandom_labels = spec$levels)))
  }
  if (!identical(spec$type, "block")) return(spec)
  n <- spec$dims[1L]
  p <- spec$dims[2L]
  df <- spec$df
  calibrate <- !is.null(df) && is.finite(df) && df > 0
  list(type = "buser", name = spec$name,
       X = matrix(spec$design, nrow = n, ncol = p, byrow = TRUE),
       dims = as.integer(c(n, p)),
       K = matrix(spec$penalty, p, p), K_dims = as.integer(c(p, p)),
       df = if (calibrate) as.double(df) else 4.0,
       # lambda = 1 keeps a penalty that already carries its lambda (or is zero).
       lambda = if (calibrate) -1.0 else 1.0,
       center = FALSE,
       # How to rebuild the design on new rows (.rebuild_block_design).
       block = spec[intersect(names(spec), c("name", "by_name", "by_levels",
                                             "by_factor", "bols_by", "factor_levels",
                                             "bols_block", "intercept"))])
}

#' Fit a GAMLSS with linear, spline, and random-intercept base-learners.
#'
#' Mirrors R \code{gamboostLSS::gamboostLSS}. Each parameter formula may mix
#' \code{bols(...)}, \code{bbs(...)}, \code{bns(...)},
#' \code{brandom(...)}, \code{bmrf(...)}, \code{bspatial(...)},
#' \code{bmono(...)}, \code{bconstrained(...)}, \code{btree(...)},
#' \code{bkernel(...)}, and \code{buser(...)} terms. The mboostrs
#' \code{bkernel} is an explicit kernel-ridge learner, not the upstream
#' pathway/GWAS construction with the same name.
#'
#' Gradient stabilization (a family argument) and per-parameter step lengths
#' (`boost_control(nu = c(mu = ..., sigma = ...))`) work as in
#' [glmboostLSS()]; see its Details.
#'
#' @inheritParams glmboostLSS
#' @return An object of class \code{"mboostrs_gamboostLSS"}.
#' @seealso [glmboostLSS()], [GaussianLSS()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' @export
gamboostLSS <- function(formula, data, families = NULL, family = NULL,
                        control = boost_control(), offset = NULL, mstop = NULL,
                        weights = NULL, method = c("cyclic", "noncyclic"),
                        weight_policy = "upstream") {
  families <- .lss_resolve_families(families, family)
  cl <- match.call()
  method <- match.arg(method)
  pnames <- families$param_names
  K <- length(pnames)
  frontend <- .gamboostlss_frontend(formula, data, families, weights = weights,
                                    weight_policy = weight_policy)
  formulas <- frontend$formulas
  specs_per_param <- frontend$specs
  y <- frontend$y
  n <- frontend$n
  w <- frontend$weights

  mstop_pp <- integer(0)
  if (!is.null(mstop)) {
    if (method == "noncyclic" && length(mstop) != 1L)
      stop("noncyclic gamboostLSS requires a scalar mstop")
    mstop_pp <- if (is.null(names(mstop)))
      as.integer(rep_len(mstop, K)) else as.integer(mstop[pnames])
  }
  if (method == "noncyclic") {
    if (!is.null(mstop)) control$mstop <- as.integer(mstop)[1]
    mstop_pp <- integer(0)
  }
  offs <- if (is.null(offset)) numeric(0) else as.double(offset)

  res <- .Call("wrap__fit_gamboostlss_mixed",
    specs_per_param,
    as.double(y),
    as.double(w),
    as.integer(control$mstop),
    as.double(.lss_nu_scalar(control$nu)),
    families$rust_name,
    offs,
    mstop_pp,
    method,
    control$trace %||% FALSE,
    .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
    PACKAGE = "mboostrs"
  )
  if (isTRUE(control$trace)) .render_execution_report(res$execution_report)

  params <- lapply(seq_len(K), function(k) {
    r <- res$results[[k]]
    list(offset = r$offset, selected = r$selected, coefs = r$coefs,
         fitted = r$fitted, learner_specs = specs_per_param[[k]],
         risk = as.numeric(res$param_risk[[k]]),
         learner_names = vapply(specs_per_param[[k]], `[[`, character(1), "name"))
  })
  names(params) <- pnames

  structure(
    list(params = params, risk = res$risk,
         selection_order = res$selection_order,
         execution_report = res$execution_report,
         nu = control$nu, family = families,
         formulas = formulas, param_names = pnames, data = data, n = n,
         weights = w, control = control, offset = offset, method = method,
         call = cl),
    class = "mboostrs_gamboostLSS"
  )
}

.gamboostlss_prediction_steps <- function(object, mstop) {
  full_steps <- vapply(object$params, function(value) length(value$selected), integer(1))
  if (is.null(mstop)) return(full_steps)
  if (identical(object$method, "noncyclic")) {
    if (length(mstop) != 1L || is.na(mstop) || mstop < 0) {
      stop("noncyclic prediction requires one non-negative global mstop")
    }
    global_steps <- min(as.integer(mstop), length(object$selection_order))
    steps <- tabulate(object$selection_order[seq_len(global_steps)],
                      nbins = length(object$param_names))
    names(steps) <- object$param_names
  } else {
    steps <- if (is.null(names(mstop))) {
      stats::setNames(as.integer(rep_len(mstop, length(object$param_names))),
                      object$param_names)
    } else {
      stats::setNames(as.integer(mstop[object$param_names]), object$param_names)
    }
    if (any(is.na(steps)) || any(steps < 0)) {
      stop("mstop must provide a non-negative budget for every LSS parameter")
    }
  }
  if (any(steps > full_steps)) stop("prediction mstop exceeds the fitted path")
  steps
}

.gamboostlss_replay_data <- function(spec, source, newdata = NULL,
                                      buser_newdata = NULL,
                                      context = "prediction",
                                      unseen_levels = "error") {
  if (identical(spec$type, "bspatial")) {
    x1 <- source[[spec$name1]]
    x2 <- source[[spec$name2]]
    if (is.null(x1)) stop(context, " data is missing variable: ", spec$name1)
    if (is.null(x2)) stop(context, " data is missing variable: ", spec$name2)
    return(c(as.double(x1), as.double(x2)))
  }
  if (identical(spec$type, "buser")) {
    rebuilt <- if (!is.null(newdata) && !is.null(spec$block))
      .rebuild_block_design(spec$block, newdata, unseen = unseen_levels) else NULL
    X <- if (is.null(newdata)) {
      spec$X
    } else if (!is.null(rebuilt)) {
      rebuilt
    } else {
      if (is.null(buser_newdata) || is.null(buser_newdata[[spec$name]])) {
        stop("new-data ", context, " for buser learner '", spec$name,
             "' requires buser_newdata=list(", spec$name, "=X_new)")
      }
      buser_newdata[[spec$name]]
    }
    X <- as.matrix(X)
    expected_columns <- as.integer(spec$dims[[2L]])
    if (!is.numeric(X) || ncol(X) != expected_columns || any(!is.finite(X))) {
      stop("buser design for '", spec$name, "' must be a finite numeric matrix with ",
           expected_columns, " columns")
    }
    return(as.double(t(X)))
  }
  if (spec$type %in% c("tree", "btree")) {
    # A tree reads several columns, factors coded against the training labels
    # (-1 for a level the fit never saw).
    for (name in spec$names %||% spec$name) {
      if (is.null(source[[name]])) stop(context, " data is missing variable: ", name)
    }
    design <- .tree_design(spec, source)
    if (!is.null(newdata) && identical(unseen_levels, "error") &&
        .tree_has_unseen(spec, design)) {
      stop("btree(", spec$name, ") has a level the fit did not see. Pass ",
           "unseen_levels = \"average\" or \"zero_effect\" to predict it.", call. = FALSE)
    }
    return(design)
  }
  values <- source[[spec$name]]
  if (is.null(values)) stop(context, " data is missing variable: ", spec$name)
  if (spec$type %in% c("brandom", "bmrf") && !is.null(newdata)) {
    # Coded against the training labels: a factor's own integer codes follow
    # its level set, so a subset of the levels was read as the wrong groups.
    # A level the fit never saw is code 0, which the core treats under the
    # unseen_levels policy (brandom); bmrf has no region outside its graph.
    n_levels <- spec$n_levels %||% spec$ncol
    labels <- if (identical(spec$type, "bmrf")) spec$level_labels else spec$levels
    codes <- if (!is.null(labels)) {
      match(as.character(values), labels)
    } else {
      v <- suppressWarnings(as.integer(values))
      v[!is.na(v) & (v < 1L | v > n_levels)] <- NA_integer_
      v
    }
    new <- is.na(codes) & !is.na(values)
    if (any(new) && (identical(unseen_levels, "error") || identical(spec$type, "bmrf"))) {
      stop(spec$type, "(", spec$name, ") has level(s) ",
           paste(unique(as.character(values[new])), collapse = ", "),
           " the fit did not see",
           if (identical(spec$type, "brandom"))
             ". Pass unseen_levels = \"average\" or \"zero_effect\" to predict them."
           else ": a bmrf region outside the neighbourhood graph has no effect.",
           call. = FALSE)
    }
    codes[new] <- 0L
    return(if (identical(spec$type, "bmrf")) as.double(codes) - 1.0 else as.double(codes))
  }
  if (identical(spec$type, "bmrf")) as.double(values) - 1.0 else as.double(values)
}

#' Predict from a fitted gamboostLSS model.
#'
#' @param object A fitted \code{mboostrs_gamboostLSS} object.
#' @param newdata Optional data frame containing the fitted learner variables.
#' @param parameter Parameter name (e.g. "mu"); NULL returns all as a named list.
#' @param type "link" or "response".
#' @param mstop Optional scalar or named per-parameter iteration budget. For a
#'   noncyclic model, a scalar denotes the global number of boosting steps.
#' @param buser_newdata Named list of explicit new-data design matrices for
#'   fitted \code{buser} learners. Each matrix must have the same number of
#'   columns as its training design.
#' @param unseen_levels What a \code{brandom}, factor \code{bols}, factor
#'   \code{by=} or \code{btree} term does with a level in \code{newdata}
#'   that the fit never saw: \code{"error"} (default) refuses,
#'   \code{"average"} gives the mean of the term's contributions over the
#'   fitted levels, \code{"zero_effect"} gives no effect of its own (a tree
#'   sends it to the branch that held more training weight). As in
#'   \code{predict.mboostrs_gamboost}.
#' @param extrapolation What to do when \code{newdata} holds values of a
#'   spline learner's variable outside the range the basis was fitted on, for
#'   every parameter: as in [predict.mboostrs_gamboost()] (\code{"warn"},
#'   \code{"error"}, \code{"clip"} or \code{"allow"}).
#' @param ... Not used.
#' @return A named list of numeric vectors (one per parameter), or a single vector if `parameter` is specified.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' head(predict(mod, type = "response"))
#' @export
predict.mboostrs_gamboostLSS <- function(object, newdata = NULL, parameter = NULL,
                                         type = c("link", "response"),
                                         mstop = NULL, buser_newdata = NULL,
                                         extrapolation = c("warn", "error", "clip", "allow"),
                                         unseen_levels = c("error", "zero_effect", "average"),
                                         ...) {
  .refuse_unused("predict", ...)
  type <- match.arg(type)
  extrapolation <- match.arg(extrapolation)
  unseen_levels <- match.arg(unseen_levels)
  extrap <- .extrapolation_policy(extrapolation)
  want <- if (is.null(parameter)) object$param_names else parameter
  unknown <- setdiff(want, object$param_names)
  if (length(unknown)) stop("unknown LSS parameter(s): ", paste(unknown, collapse = ", "))
  if (!is.null(newdata) && any(vapply(object$params, function(param) {
    # A numeric by= carries its training modifier in by_values and is
    # replayed with the new rows' modifier; only a by= without one is refused.
    any(vapply(param$learner_specs, function(spec) {
      !is.null(spec$by) && is.null(spec$by_values) && is.null(spec$block)
    }, logical(1)))
  }, logical(1)))) {
    stop("new-data prediction for gamboostLSS learners with by= is not implemented")
  }

  full_steps <- vapply(object$params, function(value) length(value$selected), integer(1))
  steps <- .gamboostlss_prediction_steps(object, mstop)

  prediction_data <- if (is.null(newdata)) object$data else newdata
  out <- list()
  for (pn in object$param_names) {
    if (!(pn %in% want)) next
    param <- object$params[[pn]]
    if (is.null(newdata) && steps[[pn]] == full_steps[[pn]]) {
      eta <- param$fitted
    } else {
      x_data <- lapply(param$learner_specs, function(spec) {
        values <- .gamboostlss_replay_data(spec, prediction_data, newdata,
                                           buser_newdata, "prediction", unseen_levels)
        if (is.null(newdata)) values else .lss_extrapolate(spec, values, extrap, pn)
      })
      eta <- .Call("wrap__predict_gamboostlss_mixed",
        list(param$learner_specs), as.double(object$weights), as.double(param$offset),
        as.integer(param$selected), param$coefs, as.double(.lss_param_nu(object, pn)), x_data,
        as.integer(steps[[pn]]), .lss_by_rows(param$learner_specs, newdata),
        unseen_levels, PACKAGE = "mboostrs")
    }
    if (type == "response") eta <- .lss_response(eta, object$family, pn)
    out[[pn]] <- eta
  }
  extrap$report()
  if (!is.null(parameter)) return(out[[parameter]])
  out
}

#' Partial effects for one gamboostLSS distribution parameter
#'
#' Replays only the selected updates belonging to each requested learner. The
#' effects exclude the parameter offset; adding the effects to that offset
#' reconstructs the link-scale predictor at the requested stopping point.
#'
#' @param object A fitted \code{mboostrs_gamboostLSS} object.
#' @param parameter Distribution parameter name or 1-based index.
#' @param which Optional learner names or 1-based indices. NULL returns every
#'   learner selected before \code{mstop}.
#' @param newdata Optional data frame for evaluation.
#' @param buser_newdata Named list of explicit design matrices for fitted
#'   \code{buser} learners when \code{newdata} is supplied.
#' @param mstop Optional stopping budget with the same semantics as \code{predict}.
#' @param ... Not used.
#' @return A named list of data frames with columns \code{x} and \code{effect}.
#' @export
partial_effects.mboostrs_gamboostLSS <- function(object, parameter,
                                                  which = NULL, newdata = NULL,
                                                  mstop = NULL,
                                                  buser_newdata = NULL, ...) {
  parameter <- .lss_resolve_parameters(object, parameter)
  if (length(parameter) != 1L) stop("partial_effects requires exactly one parameter")
  pn <- parameter[[1L]]
  param <- object$params[[pn]]
  learner_names <- param$learner_names %||% .lss_param_learner_names(object, pn)
  if (is.null(which)) {
    indices <- unique(as.integer(param$selected))
  } else if (is.numeric(which)) {
    indices <- as.integer(which)
  } else {
    indices <- match(as.character(which), learner_names)
  }
  if (any(is.na(indices)) || any(indices < 1L | indices > length(learner_names))) {
    stop("which contains an unknown gamboostLSS learner")
  }
  source <- if (is.null(newdata)) object$data else newdata
  if (!is.null(newdata) && any(vapply(param$learner_specs, function(spec) {
    !is.null(spec$by)
  }, logical(1)))) {
    stop("new-data partial effects for gamboostLSS learners with by= are not implemented")
  }
  x_data <- lapply(param$learner_specs, function(spec) {
    .gamboostlss_replay_data(spec, source, newdata, buser_newdata,
                             "partial-effect")
  })
  limit <- .gamboostlss_prediction_steps(object, mstop)[[pn]]
  result <- list()
  for (index in indices) {
    step_indices <- base::which(seq_along(param$selected) <= limit &
                               as.integer(param$selected) == index)
    effect <- .Call("wrap__predict_gamboostlss_mixed",
      list(param$learner_specs), as.double(object$weights), 0.0,
      as.integer(param$selected[step_indices]), param$coefs[step_indices],
      as.double(.lss_param_nu(object, pn)), x_data, as.integer(length(step_indices)),
      .lss_by_rows(param$learner_specs, newdata), "error", PACKAGE = "mboostrs")
    spec <- param$learner_specs[[index]]
    if (identical(spec$type, "bspatial")) {
      result[[learner_names[index]]] <- data.frame(
        x1 = as.double(source[[spec$name1]]),
        x2 = as.double(source[[spec$name2]]),
        effect = as.double(effect)
      )
    } else if (identical(spec$type, "buser")) {
      rows <- if (is.null(newdata)) nrow(spec$X) else
        nrow(as.matrix(buser_newdata[[spec$name]]))
      result[[learner_names[index]]] <- data.frame(
        x = seq_len(rows), effect = as.double(effect)
      )
    } else {
      result[[learner_names[index]]] <- data.frame(
        x = x_data[[index]], effect = as.double(effect)
      )
    }
  }
  result
}

#' Print method for gamboostLSS
#' @param x A fitted \code{mboostrs_gamboostLSS} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' print(mod)
#' @export
print.mboostrs_gamboostLSS <- function(x, ...) {
  cat("gamboostLSS model (smooth)\n")
  cat("  Family:", x$family$name, "\n")
  cat("  Parameters:", paste(x$param_names, collapse = ", "), "\n")
  invisible(x)
}

.lss_cvrisk_result <- function(risk, grid, fold_type = "custom") {
  risk <- as.matrix(risk)
  grid <- as.matrix(grid)
  storage.mode(grid) <- "integer"
  aggregate <- colMeans(risk, na.rm = TRUE)
  best_idx <- which.min(aggregate)
  attr(risk, "grid") <- grid
  attr(risk, "aggregate") <- aggregate
  attr(risk, "optimal_mstop") <- as.integer(grid[best_idx, ])
  attr(risk, "fold_type") <- fold_type
  class(risk) <- c("mboostrs_lss_cvrisk", class(risk))
  risk
}

#' Print LSS cvrisk result
#' @param x A \code{mboostrs_lss_cvrisk} object.
#' @param ... Not used.
#' @return Invisibly returns \code{x}.
#' @export
print.mboostrs_lss_cvrisk <- function(x, ...) {
  aggregate <- attr(x, "aggregate")
  cat("LSS cross-validated risk\n")
  cat("  Number of folds:", nrow(x), "\n")
  cat("  Grid points:", ncol(x), "\n")
  cat("  Optimal mstop:", paste(attr(x, "optimal_mstop"), collapse = ", "), "\n")
  cat("  Minimum aggregate risk:", formatC(min(aggregate), digits = 6), "\n")
  invisible(x)
}

#' Extract optimal mstop from LSS cvrisk
#' @param object A \code{mboostrs_lss_cvrisk} object.
#' @param ... Not used.
#' @return Integer vector of per-parameter mstop values.
#' @export
mstop.mboostrs_lss_cvrisk <- function(object, ...) {
  opt <- attr(object, "optimal_mstop")
  grid <- attr(object, "grid")
  names(opt) <- colnames(grid)
  opt
}

#' Convert LSS cvrisk result to a data frame
#' @param x A \code{mboostrs_lss_cvrisk} object.
#' @param row.names,optional,... Passed by the base generic; ignored.
#' @return A long data frame with fold, grid, risk, and aggregate columns.
#' @export
as.data.frame.mboostrs_lss_cvrisk <- function(x, row.names = NULL,
                                               optional = FALSE, ...) {
  risk <- unclass(x)
  grid <- attr(x, "grid")
  aggregate <- attr(x, "aggregate")
  optimal <- attr(x, "optimal_mstop")
  rows <- vector("list", length = nrow(risk) * ncol(risk))
  cursor <- 1L
  for (fold in seq_len(nrow(risk))) {
    for (g in seq_len(ncol(risk))) {
      row <- data.frame(
        fold = fold,
        grid_index = g,
        mstop = paste(grid[g, ], collapse = ","),
        risk = risk[fold, g],
        aggregate = aggregate[g],
        optimal = identical(as.integer(grid[g, ]), as.integer(optimal)),
        stringsAsFactors = FALSE
      )
      for (k in seq_len(ncol(grid))) {
        nm <- colnames(grid)[k]
        if (is.null(nm) || !nzchar(nm)) nm <- paste0("mstop_", k)
        row[[nm]] <- grid[g, k]
      }
      rows[[cursor]] <- row
      cursor <- cursor + 1L
    }
  }
  do.call(rbind, rows)
}

#' Plot an LSS cvrisk result
#' @param x A \code{mboostrs_lss_cvrisk} object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL.
#' @export
plot.mboostrs_lss_cvrisk <- function(x, ...) {
  plot_lss_cvrisk(x, ...)
}

.lss_stabsel_result <- function(phat, selected, max_phat, cutoff, q, PFER,
                                learner_names, assumption = "unimodal",
                                sampling_type = "MB", B = NA_integer_,
                                n_fits = NA_integer_) {
  names(max_phat) <- learner_names
  structure(
    list(
      phat = phat,
      selected = selected,
      max = max_phat,
      cutoff = cutoff,
      q = q,
      PFER = PFER,
      names = learner_names,
      learner_names = learner_names,
      assumption = assumption,
      sampling_type = sampling_type,
      # Initial subsamples the error bound was computed on: complementary
      # pairs under "SS", independent half-samples under "MB".
      B = as.integer(B),
      # Models actually fitted: 2 * B under "SS", B under "MB".
      n_fits = as.integer(n_fits)
    ),
    class = c("mboostrs_lss_stabsel", "mboostrs_stabsel")
  )
}

#' Print LSS stability selection result
#' @param x A \code{mboostrs_lss_stabsel} object.
#' @param ... Not used.
#' @return Invisibly returns \code{x}.
#' @export
print.mboostrs_lss_stabsel <- function(x, ...) {
  cat("LSS stability selection\n")
  cat("  Selected learners:", length(x$selected), "\n")
  cat("  Cutoff:", x$cutoff, "\n")
  cat("  q:", x$q, "\n")
  cat("  PFER:", x$PFER, "\n")
  if (!is.null(x$B) && !is.na(x$B)) {
    label <- if (identical(x$sampling_type, "SS")) "complementary pairs" else "subsamples"
    cat("  B:", x$B, label, "->", x$n_fits, "fits\n")
  }
  invisible(x)
}

#' Convert LSS stability selection result to a data frame
#' @param x A \code{mboostrs_lss_stabsel} object.
#' @param row.names,optional,... Passed by the base generic; ignored.
#' @return A data frame with one row per learner.
#' @export
as.data.frame.mboostrs_lss_stabsel <- function(x, row.names = NULL,
                                                optional = FALSE, ...) {
  data.frame(
    learner = names(x$max),
    max = as.numeric(x$max),
    selected = names(x$max) %in% x$selected,
    cutoff = x$cutoff,
    q = x$q,
    PFER = x$PFER,
    sampling_type = x$sampling_type,
    assumption = x$assumption,
    stringsAsFactors = FALSE
  )
}

#' Plot an LSS stability selection result
#' @param x A \code{mboostrs_lss_stabsel} object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL.
#' @export
plot.mboostrs_lss_stabsel <- function(x, ...) {
  plot_stabsel(x, ...)
}

#' Cross-validated risk for a linear GAMLSS (cvrisk for glmboostLSS).
#'
#' Mirrors R \code{gamboostLSS::cvrisk} for \code{glmboostLSS}: out-of-bag risk
#' over a grid of per-parameter mstop combinations.
#'
#' @param formula,data,families,family,control As in \code{\link{glmboostLSS}}.
#' @param folds An \code{n x B} matrix of in-bag weights (one column per fold).
#' @param grid An \code{n_grid x K} matrix of per-parameter mstop values.
#' @return A \code{B x n_grid} matrix of per-fold out-of-bag risks, with the
#'   grid points as column names.
#' @seealso [make_grid_lss()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' folds <- cv_weights(n, b = 5)
#' grid <- expand.grid(c(10, 20, 30), c(10, 20, 30))
#' cr <- cvrisk_glmboostLSS(y ~ x1 + x2, data = dat,
#'                          families = GaussianLSS(),
#'                          control = boost_control(mstop = 30),
#'                          folds = folds, grid = grid)
#' @export
cvrisk_glmboostLSS <- function(formula, data, families = NULL, family = NULL,
                               control = boost_control(), folds, grid) {
  families <- .lss_resolve_families(families, family)
  pnames <- families$param_names
  K <- length(pnames)
  if (is.list(formula) && !inherits(formula, "formula")) {
    formulas <- lapply(pnames, function(p) formula[[p]])
  } else {
    formulas <- rep(list(formula), K)
  }
  is_survival <- families$rust_name %in% c("lognormallss", "logloglss", "weibulllss")
  is_multivariate <- families$rust_name %in% c("dirichletlss", "gaussianknownvarlss")
  y <- NULL; n <- NULL; blocks <- vector("list", K); param_ncols <- integer(K)
  for (k in seq_len(K)) {
    parsed <- .parse_formula(formulas[[k]], data, has_survival = is_survival, multivariate = is_multivariate)
    if (is.null(y)) { y <- parsed$y; n <- parsed$n }
    blocks[[k]] <- parsed$x_matrix
    param_ncols[k] <- ncol(parsed$x_matrix)
  }
  x_all <- do.call(cbind, blocks); if (is.null(x_all)) x_all <- matrix(0.0, n, 0)
  w <- rep(1.0, n)
  grid <- as.matrix(grid)
  res <- .Call("wrap__cvrisk_glmboostlss",
    as.double(t(x_all)), as.integer(n), as.integer(param_ncols),
    as.double(y), as.double(w), as.integer(control$mstop), as.double(.lss_nu_scalar(control$nu)),
    families$rust_name,
    as.double(t(folds)), as.integer(ncol(folds)),
    as.integer(t(grid)), as.integer(nrow(grid)),
    .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
    PACKAGE = "mboostrs")
  m <- matrix(res, nrow = ncol(folds), byrow = TRUE)
  colnames(m) <- apply(grid, 1, paste, collapse = ",")
  .lss_cvrisk_result(m, grid)
}

#' Stability selection for a linear GAMLSS (stabsel for glmboostLSS).
#'
#' Mirrors R \code{gamboostLSS::stabsel} for \code{glmboostLSS} (non-cyclic):
#' selection frequencies over subsamples for the combined
#' \code{"<predictor>.<parameter>"} learner set.
#'
#' @param formula,data,families,family,control As in \code{\link{glmboostLSS}}.
#' @param folds An \code{n x B} matrix of in-bag weights holding the
#'   \emph{initial} subsamples, one column per subsample (\code{"MB"}) or per
#'   complementary pair (\code{"SS"}). Under \code{"SS"} the exact complements
#'   are appended here, so \code{B} columns produce \code{2 * B} fits; pairing
#'   is never inferred from an even column count. \code{\link{stabsel_folds}}
#'   builds one.
#' @param q,PFER,cutoff Stability-selection error-control parameters (as in
#'   \code{stabs::stabsel}).
#' @param assumption,sampling_type Passed to the error-control computation.
#'   \code{sampling_type = "SS"} uses complementary pairs and fits
#'   \code{2 * B} models; \code{"MB"} fits \code{B}. The bound is computed on
#'   \code{B}, and both counts are on the result as \code{B} and
#'   \code{n_fits}.
#' @return A list with \code{selected} (stable combined learners), \code{max}
#'   (per-learner max selection frequency), and \code{cutoff}/\code{q}/\code{PFER}.
#' @seealso [stabsel_params()], [stabsel()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' folds <- cv_weights(n, b = 50)
#' sel <- stabsel_glmboostLSS(y ~ x1 + x2, data = dat,
#'                            families = GaussianLSS(),
#'                            control = boost_control(mstop = 50),
#'                            folds = folds)
#' sel$selected
#' }
#' @export
stabsel_glmboostLSS <- function(formula, data, families = NULL, family = NULL,
                                control = boost_control(), folds,
                                q = -1L, PFER = -1.0, cutoff = -1.0,
                                assumption = "unimodal", sampling_type = "MB") {
  families <- .lss_resolve_families(families, family)
  pnames <- families$param_names
  K <- length(pnames)
  formulas <- if (is.list(formula) && !inherits(formula, "formula"))
    lapply(pnames, function(p) formula[[p]]) else rep(list(formula), K)
  is_survival <- families$rust_name %in% c("lognormallss", "logloglss", "weibulllss")
  is_multivariate <- families$rust_name %in% c("dirichletlss", "gaussianknownvarlss")
  y <- NULL; n <- NULL; blocks <- vector("list", K); param_ncols <- integer(K)
  combined <- character(0)
  for (k in seq_len(K)) {
    parsed <- .parse_formula(formulas[[k]], data, has_survival = is_survival, multivariate = is_multivariate)
    if (is.null(y)) { y <- parsed$y; n <- parsed$n }
    blocks[[k]] <- parsed$x_matrix
    param_ncols[k] <- ncol(parsed$x_matrix)
    combined <- c(combined, paste0(c("(Intercept)", colnames(parsed$x_matrix)),
                                   ".", pnames[k]))
  }
  x_all <- do.call(cbind, blocks); if (is.null(x_all)) x_all <- matrix(0.0, n, 0)
  p <- length(combined)
  sampling_type <- .stabsel_sampling_type(sampling_type)
  prepared <- .stabsel_fit_folds(folds, sampling_type, n)
  folds <- prepared$folds; B <- prepared$B; n_fits <- prepared$n_fits

  par <- stabsel_params(p = p, b = B,
                        cutoff = cutoff, q = q, pfer = PFER,
                        assumption = assumption, sampling_type = sampling_type)
  cutoff_val <- par$cutoff; q_val <- par$q

  seqs <- .Call("wrap__stabsel_selected_glmboostlss",
    as.double(t(x_all)), as.integer(n), as.integer(param_ncols),
    as.double(y), rep(1.0, n), as.integer(control$mstop), as.double(.lss_nu_scalar(control$nu)),
    families$rust_name, as.double(t(folds)), as.integer(n_fits), .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
 PACKAGE = "mboostrs")

  m <- control$mstop
  phat <- .lss_stabsel_frequencies(seqs, p, m, q_val)
  max_phat <- apply(phat, 1, max)
  selected <- combined[max_phat >= cutoff_val]
  .lss_stabsel_result(
    phat = phat,
    selected = selected,
    max_phat = max_phat,
    cutoff = cutoff_val,
    q = q_val,
    PFER = par$pfer,
    learner_names = combined,
    assumption = assumption,
    sampling_type = sampling_type,
    B = B,
    n_fits = n_fits
  )
}

# ─── LSS S3 methods ────────────────────────────────────────────────

.lss_resolve_parameters <- function(object, parameter = NULL) {
  if (is.null(parameter)) return(object$param_names)
  if (is.numeric(parameter)) {
    idx <- as.integer(parameter)
    if (any(idx < 1L | idx > length(object$param_names))) {
      stop("parameter index out of range")
    }
    return(object$param_names[idx])
  }
  parameter <- as.character(parameter)
  unknown <- setdiff(parameter, object$param_names)
  if (length(unknown)) {
    stop("unknown LSS parameter(s): ", paste(unknown, collapse = ", "))
  }
  parameter
}

.lss_param_learner_names <- function(object, parameter) {
  k <- match(parameter, object$param_names)
  if (inherits(object, "mboostrs_gamboostLSS")) {
    stored <- object$params[[parameter]]$learner_names
    if (!is.null(stored)) return(stored)
    specs <- .parse_bbs_formula(object$formulas[[k]])
    return(vapply(specs, function(s) s$name, character(1)))
  }
  parsed <- .parse_formula(object$formulas[[k]], object$data,
                           need_response = FALSE)
  c("(Intercept)", parsed$learner_names)
}

.lss_selected <- function(object, which = NULL, parameter = NULL, merge = FALSE) {
  params <- .lss_resolve_parameters(object, parameter)
  by_param <- stats::setNames(lapply(params, function(pn) {
    learner_names <- .lss_param_learner_names(object, pn)
    idx <- as.integer(object$params[[pn]]$selected)
    valid <- idx >= 1L & idx <= length(learner_names)
    path <- learner_names[idx[valid]]
    if (!is.null(which)) {
      targets <- if (is.numeric(which)) learner_names[as.integer(which)] else as.character(which)
      path <- path[path %in% targets]
    }
    path
  }), params)
  if (!is.null(parameter)) return(by_param[[1]])
  if (!merge) return(by_param)
  unlist(Map(function(pn, path) paste0(path, ".", pn), names(by_param), by_param),
         use.names = FALSE)
}

#' Selected learners for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param which Optional learner name or index filter.
#' @param parameter Optional LSS parameter name or 1-based index.
#' @param merge If TRUE, return one flat vector with \code{<learner>.<parameter>} names.
#' @param ... Not used.
#' @return A named list by parameter, or a character vector when \code{parameter} or \code{merge} is used.
#' @export
selected.mboostrs_glmboostLSS <- function(object, which = NULL, parameter = NULL,
                                          merge = FALSE, ...) {
  .lss_selected(object, which = which, parameter = parameter, merge = merge)
}

#' Selected learners for gamboostLSS
#' @inheritParams selected.mboostrs_glmboostLSS
#' @export
selected.mboostrs_gamboostLSS <- function(object, which = NULL, parameter = NULL,
                                          merge = FALSE, ...) {
  .lss_selected(object, which = which, parameter = parameter, merge = merge)
}

.lss_fitted <- function(object, parameter = NULL, mstop = NULL,
                        type = c("link", "response")) {
  type <- match.arg(type)
  if (!is.null(mstop)) object <- update(object, mstop = mstop)
  params <- if (is.null(parameter)) NULL else .lss_resolve_parameters(object, parameter)
  if (is.null(params)) return(predict(object, type = type))
  if (length(params) == 1L) return(predict(object, parameter = params, type = type))
  out <- predict(object, type = type)
  out[params]
}

#' Fitted values for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param parameter Optional LSS parameter name or 1-based index.
#' @param mstop Optional scalar or per-parameter mstop for a refitted/sliced view.
#' @param type "link" or "response".
#' @param ... Not used.
#' @return A named list by parameter, or a vector when \code{parameter} is supplied.
#' @export
fitted.mboostrs_glmboostLSS <- function(object, parameter = NULL, mstop = NULL,
                                        type = c("link", "response"), ...) {
  .lss_fitted(object, parameter = parameter, mstop = mstop, type = type)
}

#' Fitted values for gamboostLSS
#' @inheritParams fitted.mboostrs_glmboostLSS
#' @export
fitted.mboostrs_gamboostLSS <- function(object, parameter = NULL, mstop = NULL,
                                        type = c("link", "response"), ...) {
  .lss_fitted(object, parameter = parameter, mstop = mstop, type = type)
}

#' Risk paths of a glmboostLSS fit
#'
#' The joint negative log-likelihood after each update, as gamboostLSS's
#' \code{risk()} reports it.
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param merge FALSE (default) returns a list with one path per distribution
#'   parameter: the offset-only risk, then the joint risk after each update of
#'   that parameter. TRUE returns one named vector: for a cyclic fit the
#'   per-parameter paths interleaved by position, for a non-cyclic fit the
#'   offset-only risk once per parameter followed by every step's risk in
#'   order, each named by the parameter it followed.
#' @param parameter Parameters to include (names or positions); all by default.
#' @param ... Not used.
#' @return A list of numeric paths, or a named numeric vector with
#'   \code{merge = TRUE}. The \code{risk} element of the fitted object is the
#'   risk after each full cycle.
#' @export
risk.mboostrs_glmboostLSS <- function(object, merge = FALSE, parameter = NULL, ...) {
  .refuse_unused("risk", ...)
  names_all <- object$param_names
  paths <- lapply(object$params, function(p) p$risk)
  if (any(vapply(paths, is.null, logical(1)))) {
    stop("this model was fitted before per-parameter risk was recorded; ",
         "refit it to obtain risk()", call. = FALSE)
  }
  names(paths) <- names_all
  parameter <- if (is.null(parameter)) names_all else parameter
  if (is.numeric(parameter)) parameter <- names_all[parameter]
  if (!all(parameter %in% names_all)) {
    stop("unknown parameter: ", paste(setdiff(parameter, names_all), collapse = ", "),
         call. = FALSE)
  }
  if (!merge) {
    out <- paths[parameter]
    class(out) <- "inbag"
    return(out)
  }
  # As gamboostLSS: a cyclic fit interleaves the per-parameter paths by
  # position; a non-cyclic fit lists the offset-only risk once per parameter
  # and then every step's risk in order, each named by the parameter updated.
  if (identical(object$method, "noncyclic")) {
    values <- vapply(names_all, function(nm) paths[[nm]][1], numeric(1))
    labels <- names_all
    position <- setNames(rep(1L, length(names_all)), names_all)
    for (k in object$selection_order) {
      nm <- names_all[k]
      position[[nm]] <- position[[nm]] + 1L
      values <- c(values, paths[[nm]][position[[nm]]])
      labels <- c(labels, nm)
    }
  } else {
    longest <- max(lengths(paths[parameter]))
    values <- numeric(0); labels <- character(0)
    for (i in seq_len(longest)) for (nm in parameter) {
      if (i <= length(paths[[nm]])) {
        values <- c(values, paths[[nm]][i]); labels <- c(labels, nm)
      }
    }
  }
  out <- setNames(values, labels)
  class(out) <- "inbag"
  out
}

#' Risk paths of a gamboostLSS fit
#' @inheritParams risk.mboostrs_glmboostLSS
#' @export
risk.mboostrs_gamboostLSS <- function(object, merge = FALSE, parameter = NULL, ...) {
  risk.mboostrs_glmboostLSS(object, merge = merge, parameter = parameter, ...)
}

#' Observation weights for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param ... Not used.
#' @return Numeric observation weights.
#' @export
model_weights.mboostrs_glmboostLSS <- function(object, ...) {
  if (!is.null(object$weights)) return(object$weights)
  rep(1.0, object$n)
}

#' Observation weights for gamboostLSS
#' @inheritParams model_weights.mboostrs_glmboostLSS
#' @export
model_weights.mboostrs_gamboostLSS <- function(object, ...) {
  model_weights.mboostrs_glmboostLSS(object, ...)
}

.lss_params <- function(object, newdata = NULL, type = c("response", "link"),
                        mstop = NULL) {
  type <- match.arg(type)
  if (!is.null(mstop)) object <- update(object, mstop = mstop)
  predict(object, newdata = newdata, type = type)
}

#' Predicted parameters for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param newdata Optional data frame.
#' @param type "response" or "link".
#' @param mstop Optional scalar or per-parameter mstop for a refitted/sliced view.
#' @param ... Not used.
#' @return Named list of parameter vectors.
#' @export
lss_params.mboostrs_glmboostLSS <- function(object, newdata = NULL,
                                            type = c("response", "link"),
                                            mstop = NULL, ...) {
  .lss_params(object, newdata = newdata, type = type, mstop = mstop)
}

#' Predicted parameters for gamboostLSS
#' @inheritParams lss_params.mboostrs_glmboostLSS
#' @export
lss_params.mboostrs_gamboostLSS <- function(object, newdata = NULL,
                                            type = c("response", "link"),
                                            mstop = NULL, ...) {
  .lss_params(object, newdata = newdata, type = type, mstop = mstop)
}

.lss_predict_all_model <- function(object, newdata = NULL,
                                   type = c("response", "link"), mstop = NULL,
                                   output = c("data.frame", "list"),
                                   include_response = TRUE) {
  type <- match.arg(type)
  output <- match.arg(output)
  params <- lss_params(object, newdata = newdata, type = type, mstop = mstop)
  if (identical(output, "list")) return(params)
  n <- length(params[[1L]])
  out <- data.frame(index = seq_len(n))
  if (isTRUE(include_response)) {
    y <- tryCatch(
      .lss_plain_response(object, if (is.null(newdata)) object$data else newdata),
      error = function(e) NULL
    )
    if (!is.null(y)) out$y <- y
  }
  for (nm in names(params)) out[[nm]] <- params[[nm]]
  out
}

#' All-parameter predictions for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param newdata Optional data frame.
#' @param type "response" or "link".
#' @param mstop Optional scalar or per-parameter mstop.
#' @param output "data.frame" or "list".
#' @param include_response Include the response column when available.
#' @param ... Not used.
#' @return A data frame or list of predicted parameters.
#' @export
predictAll.mboostrs_glmboostLSS <- function(object, newdata = NULL,
                                            type = c("response", "link"),
                                            mstop = NULL,
                                            output = c("data.frame", "list"),
                                            include_response = TRUE, ...) {
  .lss_predict_all_model(object, newdata = newdata, type = type, mstop = mstop,
                         output = output, include_response = include_response)
}

#' All-parameter predictions for gamboostLSS
#' @inheritParams predictAll.mboostrs_glmboostLSS
#' @export
predictAll.mboostrs_gamboostLSS <- function(object, newdata = NULL,
                                            type = c("response", "link"),
                                            mstop = NULL,
                                            output = c("data.frame", "list"),
                                            include_response = TRUE, ...) {
  .lss_predict_all_model(object, newdata = newdata, type = type, mstop = mstop,
                         output = output, include_response = include_response)
}

.lss_plain_response <- function(object, data) {
  is_survival <- object$family$rust_name %in% c("lognormallss", "logloglss", "weibulllss")
  is_multivariate <- object$family$rust_name == "dirichletlss"
  if (is_survival || is_multivariate) {
    stop("distribution diagnostics are not implemented for survival or multivariate LSS responses")
  }
  resp_name <- all.vars(object$formulas[[1]])[1]
  as.double(data[[resp_name]])
}

.lss_deviance_increment_model <- function(object, newdata = NULL) {
  data <- if (is.null(newdata)) object$data else newdata
  y <- .lss_plain_response(object, data)
  params <- lss_params(object, newdata = newdata, type = "response")
  lss_deviance_increment(object$family, y, params)
}

#' Per-observation deviance contribution for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param newdata Optional data frame containing the response and predictors.
#' @param ... Not used.
#' @return Numeric vector of per-observation deviance contributions.
#' @export
deviance_increment.mboostrs_glmboostLSS <- function(object, newdata = NULL, ...) {
  .lss_deviance_increment_model(object, newdata = newdata)
}

#' Per-observation deviance contribution for gamboostLSS
#' @inheritParams deviance_increment.mboostrs_glmboostLSS
#' @export
deviance_increment.mboostrs_gamboostLSS <- function(object, newdata = NULL, ...) {
  .lss_deviance_increment_model(object, newdata = newdata)
}

.lss_prob_col <- function(prob) {
  gsub("\\.", "_", paste0("q", format(100.0 * prob, trim = TRUE, scientific = FALSE)))
}

.lss_centiles_model <- function(object, probs = c(0.025, 0.5, 0.975),
                                newdata = NULL, mstop = NULL) {
  probs <- as.double(probs)
  if (!length(probs) || any(probs <= 0.0 | probs >= 1.0)) {
    stop("centile probabilities must be in (0, 1)")
  }
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  n <- length(params[[1L]])
  out <- data.frame(index = seq_len(n))
  for (prob in probs) {
    out[[.lss_prob_col(prob)]] <- lss_quantile(object$family, rep(prob, n), params)
  }
  out
}

#' LSS centile table for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param probs Probability values in (0, 1).
#' @param newdata Optional data frame.
#' @param mstop Optional scalar or per-parameter mstop.
#' @param ... Not used.
#' @return A data frame with one row per observation and one column per centile.
#' @export
centiles.mboostrs_glmboostLSS <- function(object, probs = c(0.025, 0.5, 0.975),
                                          newdata = NULL, mstop = NULL, ...) {
  .lss_centiles_model(object, probs = probs, newdata = newdata, mstop = mstop)
}

#' LSS centile table for gamboostLSS
#' @inheritParams centiles.mboostrs_glmboostLSS
#' @export
centiles.mboostrs_gamboostLSS <- function(object, probs = c(0.025, 0.5, 0.975),
                                          newdata = NULL, mstop = NULL, ...) {
  .lss_centiles_model(object, probs = probs, newdata = newdata, mstop = mstop)
}

.lss_profile_weights <- function(object, n, newdata = NULL, weights = NULL) {
  if (is.null(weights)) {
    weights <- if (is.null(newdata)) model_weights(object) else rep(1.0, n)
  }
  weights <- as.double(weights)
  if (length(weights) != n || any(!is.finite(weights)) || any(weights < 0) ||
      !any(weights > 0)) {
    stop("weights must be finite, non-negative, aligned with the response, and contain a positive value")
  }
  weights
}

#' Observed and average fitted LSS distribution
#'
#' The fitted curve averages every row-specific density or mass. It does not
#' evaluate one distribution at average parameters. For a semicontinuous
#' family, atom probability is returned as separate rows because probability
#' mass and density have different units.
#' @param object A fitted LSS model.
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter stopping point.
#' @param grid Optional increasing response grid.
#' @param n_grid Number of default grid points or maximum default discrete support.
#' @param weights Optional non-negative evaluation weights.
#' @return A data frame with columns `x`, `value`, `series`, and `kind`.
#' @export
density_overlay_data <- function(object, newdata = NULL, mstop = NULL,
                                 grid = NULL, n_grid = 200L, weights = NULL) {
  if (!inherits(object, c("mboostrs_glmboostLSS", "mboostrs_gamboostLSS"))) {
    stop("density_overlay_data requires an mboostrs LSS model")
  }
  data <- if (is.null(newdata)) object$data else newdata
  y <- .lss_plain_response(object, data)
  if (!length(y) || any(!is.finite(y))) stop("the observed response must be finite and non-empty")
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  w <- .lss_profile_weights(object, length(y), newdata, weights)
  sw <- sum(w)
  n_grid <- as.integer(n_grid)
  if (length(n_grid) != 1L || n_grid < 2L) stop("n_grid must be at least 2")
  discrete <- object$family$rust_name %in% c(
    "nbinomiallss", "negativebinomiallss", "poissonlss", "geometriclss",
    "binomiallss", "zipolss", "zinblss", "zeroadjustedpoissonlss"
  )
  atom <- if (object$family$rust_name == "zeroadjustedgammalss") 0.0 else NULL

  if (is.null(grid) && discrete) {
    lo <- floor(min(y))
    hi <- ceiling(max(y))
    upper <- tryCatch(
      lss_quantile(object$family, rep(0.995, length(y)), params),
      error = function(e) numeric()
    )
    if (any(is.finite(upper))) hi <- max(hi, max(upper[is.finite(upper)]))
    if (hi - lo + 1 > n_grid) {
      stop("the default discrete support has more than n_grid points; raise n_grid or pass grid")
    }
    grid <- seq(lo, hi)
  } else if (is.null(grid)) {
    continuous_y <- if (is.null(atom)) y else y[y != atom]
    if (!length(continuous_y)) stop("a mixed overlay needs at least one non-atom response")
    limits <- continuous_y
    for (prob in c(0.005, 0.995)) {
      q <- tryCatch(lss_quantile(object$family, rep(prob, length(y)), params),
                    error = function(e) numeric())
      limits <- c(limits, q[is.finite(q)])
    }
    lo <- min(limits)
    hi <- max(limits)
    padding <- if (hi > lo) 0.03 * (hi - lo) else 0.1 * max(abs(lo), 1)
    grid <- seq(lo - padding, hi + padding, length.out = n_grid)
    if (!is.null(atom)) grid <- grid[grid > atom]
  } else {
    grid <- as.double(grid)
    if (length(grid) < 2L || any(!is.finite(grid)) || any(diff(grid) <= 0)) {
      stop("grid must be a finite, strictly increasing vector with at least two values")
    }
  }

  rows <- list()
  add <- function(x, observed, fitted, kind) {
    data.frame(
      x = rep(x, each = 2L), value = c(observed, fitted),
      series = c("observed", "fitted"), kind = rep(kind, 2L),
      stringsAsFactors = FALSE
    )
  }
  if (discrete) {
    rows <- lapply(grid, function(value) {
      fitted <- lss_pdf(object$family, rep(value, length(y)), params)
      add(value, sum(w[y == value]) / sw, sum(w * fitted) / sw, "mass")
    })
  } else {
    keep <- w > 0
    if (!is.null(atom)) keep <- keep & y != atom
    yy <- y[keep]
    ww <- w[keep]
    sww <- sum(ww)
    if (!sww) stop("a continuous density overlay needs positive weight off the atom")
    center <- sum(ww * yy) / sww
    variance <- sum(ww * (yy - center)^2) / sww
    iqr <- diff(stats::quantile(yy, c(0.25, 0.75), names = FALSE))
    scale <- if (iqr > 0) min(sqrt(max(variance, 0)), iqr / 1.34) else sqrt(max(variance, 0))
    if (!is.finite(scale) || scale <= 0) scale <- max(diff(range(yy)), abs(center), 1) / 4
    effective_n <- sww^2 / sum(ww^2)
    bandwidth <- max(0.9 * scale * effective_n^(-0.2), .Machine$double.eps)
    normalizer <- sqrt(2 * pi) * bandwidth * sw
    rows <- lapply(grid, function(value) {
      if (!is.null(atom) && value == atom) return(NULL)
      fitted <- lss_pdf(object$family, rep(value, length(y)), params)
      kernel <- exp(-0.5 * ((value - yy) / bandwidth)^2)
      add(value, sum(ww * kernel) / normalizer, sum(w * fitted) / sw, "density")
    })
    if (!is.null(atom)) {
      fitted_atom <- lss_pdf(object$family, rep(atom, length(y)), params)
      rows[[length(rows) + 1L]] <- add(
        atom, sum(w[y == atom]) / sw, sum(w * fitted_atom) / sw, "atom"
      )
    }
  }
  do.call(rbind, rows)
}

#' Fitted LSS centile-fan data
#' @inheritParams density_overlay_data
#' @param xvar Numeric covariate used to order the centiles.
#' @param probs Symmetric probabilities containing 0.5.
#' @param split_by Optional column retained for split panels. This does not refit models.
#' @return A row-preserving data frame with the observed response and centile columns.
#' @export
centile_fan_data <- function(object, xvar,
                             probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
                             split_by = NULL, newdata = NULL, mstop = NULL) {
  probs <- sort(unique(as.double(probs)))
  if (length(probs) < 3L || any(!is.finite(probs)) || any(probs <= 0 | probs >= 1)) {
    stop("probs must contain at least three finite probabilities in (0, 1), including 0.5")
  }
  if (!any(abs(probs - 0.5) < 1e-12)) stop("probs must contain 0.5 for the fan median")
  lower <- probs[probs < 0.5]
  upper <- probs[probs > 0.5]
  if (length(lower) != length(upper) || any(abs(lower - rev(1 - upper)) > 1e-12)) {
    stop("probs below and above 0.5 must form symmetric pairs")
  }
  data <- if (is.null(newdata)) object$data else newdata
  y <- .lss_plain_response(object, data)
  x <- as.double(data[[xvar]])
  if (length(x) != length(y) || any(!is.finite(x))) {
    stop("xvar must name a finite numeric column aligned with the response")
  }
  split <- if (is.null(split_by)) rep("all", length(y)) else as.character(data[[split_by]])
  if (length(split) != length(y) || anyNA(split)) {
    stop("split_by must name a complete column aligned with the response")
  }
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  out <- data.frame(index = seq_along(y), x = x, observed = y, split = split,
                    xvar = as.character(xvar), stringsAsFactors = FALSE)
  for (prob in probs) {
    out[[.lss_prob_col(prob)]] <- lss_quantile(
      object$family, rep(prob, length(y)), params
    )
  }
  out[order(out$split, out$x), , drop = FALSE]
}

# Split `ord` into `k` equal-count intervals under the rule the Python
# frontend uses: the first `n %% k` intervals take `n %/% k + 1` elements and
# the rest take `n %/% k`. This is numpy's `array_split` allocation.
#
# It is written out rather than delegated to `cut(seq_len(n), breaks = k)`,
# which places the remainder in different intervals: for n = 23 and k = 10,
# `cut` gives sizes 3,2,2,2,3,2,2,2,2,3 while `array_split` gives
# 3,3,3,2,2,2,2,2,2,2. Both are "equal-count", but the two frontends then
# report different bin membership, and so different summaries, for the same
# model whenever `n %% k != 0`.
.equal_count_groups <- function(ord, k) {
  n <- length(ord)
  k <- as.integer(k)
  if (k < 1L) stop("equal-count binning needs at least one interval")
  sizes <- rep.int(n %/% k, k)
  remainder <- n %% k
  if (remainder > 0L) {
    sizes[seq_len(remainder)] <- sizes[seq_len(remainder)] + 1L
  }
  labels <- factor(rep.int(seq_len(k), sizes), levels = seq_len(k))
  unname(split(ord, labels))
}

#' Joint LSS deviance profile
#' @inheritParams density_overlay_data
#' @param xvar Optional numeric covariate; row order is used when NULL.
#' @param bins Number of equal-count bins.
#' @return A data frame of weighted joint-deviance summaries by bin.
#' @export
deviance_profile <- function(object, xvar = NULL, bins = 10L, newdata = NULL,
                             mstop = NULL, weights = NULL) {
  data <- if (is.null(newdata)) object$data else newdata
  y <- .lss_plain_response(object, data)
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  contribution <- lss_deviance_increment(object$family, y, params)
  if (length(contribution) != length(y) || any(!is.finite(contribution))) {
    stop("deviance contributions must be finite and aligned with the response")
  }
  if (is.null(xvar)) {
    x <- seq_along(y)
    xvar_name <- "index"
  } else {
    x <- as.double(data[[xvar]])
    xvar_name <- as.character(xvar)
  }
  if (length(x) != length(y) || any(!is.finite(x))) {
    stop("xvar must name a finite numeric column aligned with the response")
  }
  w <- .lss_profile_weights(object, length(y), newdata, weights)
  bins <- as.integer(bins)
  if (length(bins) != 1L || bins <= 0L) stop("bins must be positive")
  ord <- order(x)
  groups <- .equal_count_groups(ord, min(bins, length(y)))
  rows <- lapply(seq_along(groups), function(interval) {
    idx <- groups[[interval]]
    ww <- w[idx]
    sw <- sum(ww)
    if (sw <= 0) return(NULL)
    d <- contribution[idx]
    data.frame(
      interval = interval, xvar = xvar_name,
      x_min = min(x[idx]), x_max = max(x[idx]), x_mid = sum(ww * x[idx]) / sw,
      n = length(idx), weight = sw, mean_deviance = sum(ww * d) / sw,
      total_deviance = sum(ww * d), median_deviance = stats::median(d),
      q90_deviance = as.double(stats::quantile(d, 0.9, names = FALSE)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

.lss_prediction_interval_model <- function(object, level = 0.95,
                                           newdata = NULL, mstop = NULL) {
  level <- as.double(level)
  if (length(level) != 1L || level <= 0.0 || level >= 1.0) {
    stop("level must be a scalar in (0, 1)")
  }
  alpha <- (1.0 - level) / 2.0
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  n <- length(params[[1L]])
  data.frame(
    index = seq_len(n),
    lower = lss_quantile(object$family, rep(alpha, n), params),
    median = lss_quantile(object$family, rep(0.5, n), params),
    upper = lss_quantile(object$family, rep(1.0 - alpha, n), params),
    level = level,
    lower_prob = alpha,
    upper_prob = 1.0 - alpha
  )
}

#' LSS prediction interval for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param level Central prediction interval level.
#' @param newdata Optional data frame.
#' @param mstop Optional scalar or per-parameter mstop.
#' @param ... Not used.
#' @return A data frame with lower, median, and upper interval columns.
#' @export
prediction_interval.mboostrs_glmboostLSS <- function(object, level = 0.95,
                                                     newdata = NULL, mstop = NULL, ...) {
  .lss_prediction_interval_model(object, level = level, newdata = newdata, mstop = mstop)
}

#' LSS prediction interval for gamboostLSS
#' @inheritParams prediction_interval.mboostrs_glmboostLSS
#' @export
prediction_interval.mboostrs_gamboostLSS <- function(object, level = 0.95,
                                                     newdata = NULL, mstop = NULL, ...) {
  .lss_prediction_interval_model(object, level = level, newdata = newdata, mstop = mstop)
}

#' Alias for LSS prediction intervals
#' @inheritParams prediction_interval.mboostrs_glmboostLSS
#' @export
predint.mboostrs_glmboostLSS <- function(object, level = 0.95,
                                         newdata = NULL, mstop = NULL, ...) {
  prediction_interval(object, level = level, newdata = newdata, mstop = mstop, ...)
}

#' Alias for LSS prediction intervals
#' @inheritParams prediction_interval.mboostrs_glmboostLSS
#' @export
predint.mboostrs_gamboostLSS <- function(object, level = 0.95,
                                         newdata = NULL, mstop = NULL, ...) {
  prediction_interval(object, level = level, newdata = newdata, mstop = mstop, ...)
}

#' Alias for LSS prediction intervals
#' @inheritParams prediction_interval.mboostrs_glmboostLSS
#' @export
PI.mboostrs_glmboostLSS <- function(object, level = 0.95,
                                    newdata = NULL, mstop = NULL, ...) {
  prediction_interval(object, level = level, newdata = newdata, mstop = mstop, ...)
}

#' Alias for LSS prediction intervals
#' @inheritParams prediction_interval.mboostrs_glmboostLSS
#' @export
PI.mboostrs_gamboostLSS <- function(object, level = 0.95,
                                    newdata = NULL, mstop = NULL, ...) {
  prediction_interval(object, level = level, newdata = newdata, mstop = mstop, ...)
}

.lss_pit_model <- function(object, newdata = NULL, mstop = NULL,
                           randomized = TRUE) {
  data <- if (is.null(newdata)) object$data else newdata
  y <- .lss_plain_response(object, data)
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  upper <- lss_cdf(object$family, y, params)
  discrete <- object$family$rust_name %in% c(
    "nbinomiallss", "negativebinomiallss", "poissonlss", "geometriclss",
    "binomiallss", "zipolss", "zinblss", "zeroadjustedpoissonlss"
  )
  if (discrete) {
    lower <- lss_cdf(object$family, y - 1.0, params)
  } else if (identical(object$family$rust_name, "zeroadjustedgammalss")) {
    lower <- ifelse(y <= 0, 0, upper)
  } else {
    lower <- upper
  }
  u <- if (isTRUE(randomized)) stats::runif(length(y), min = lower, max = upper)
       else 0.5 * (lower + upper)
  pmin(pmax(u, .Machine$double.eps), 1.0 - .Machine$double.eps)
}

#' PIT values for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter mstop.
#' @param randomized For discrete families, draw uniformly between CDF jumps.
#'   If FALSE, use the midpoint for deterministic diagnostics.
#' @param ... Not used.
#' @return Numeric PIT vector.
#' @export
pit.mboostrs_glmboostLSS <- function(object, newdata = NULL, mstop = NULL,
                                     randomized = TRUE, ...) {
  .lss_pit_model(object, newdata = newdata, mstop = mstop, randomized = randomized)
}

#' PIT values for gamboostLSS
#' @inheritParams pit.mboostrs_glmboostLSS
#' @export
pit.mboostrs_gamboostLSS <- function(object, newdata = NULL, mstop = NULL,
                                     randomized = TRUE, ...) {
  .lss_pit_model(object, newdata = newdata, mstop = mstop, randomized = randomized)
}

.lss_quantile_residuals_model <- function(object, newdata = NULL, mstop = NULL,
                                          randomized = TRUE) {
  stats::qnorm(.lss_pit_model(object, newdata = newdata, mstop = mstop,
                              randomized = randomized))
}

#' Randomized quantile residuals for glmboostLSS
#' @inheritParams pit.mboostrs_glmboostLSS
#' @param ... Not used.
#' @return Numeric quantile residual vector.
#' @export
quantile_residuals.mboostrs_glmboostLSS <- function(object, newdata = NULL, mstop = NULL,
                                                    randomized = TRUE, ...) {
  .lss_quantile_residuals_model(object, newdata = newdata, mstop = mstop,
                                randomized = randomized)
}

#' Randomized quantile residuals for gamboostLSS
#' @inheritParams quantile_residuals.mboostrs_glmboostLSS
#' @export
quantile_residuals.mboostrs_gamboostLSS <- function(object, newdata = NULL, mstop = NULL,
                                                    randomized = TRUE, ...) {
  .lss_quantile_residuals_model(object, newdata = newdata, mstop = mstop,
                                randomized = randomized)
}

.lss_residual_frame_model <- function(object, kind = c("quantile", "response"),
                                      newdata = NULL, mstop = NULL,
                                      randomized = TRUE) {
  kind <- match.arg(kind)
  data <- if (is.null(newdata)) object$data else newdata
  y <- .lss_plain_response(object, data)
  params <- lss_params(object, newdata = newdata, type = "response", mstop = mstop)
  pit_values <- NULL
  if (identical(kind, "quantile")) {
    pit_values <- .lss_pit_model(object, newdata = newdata, mstop = mstop,
                                 randomized = randomized)
    residual <- stats::qnorm(pit_values)
  } else {
    residual <- y - params[[object$param_names[1L]]]
    pit_values <- tryCatch(
      .lss_pit_model(object, newdata = newdata, mstop = mstop,
                     randomized = randomized),
      error = function(e) NULL
    )
  }
  deviance <- tryCatch(
    lss_deviance_increment(object$family, y, params),
    error = function(e) NULL
  )
  out <- data.frame(
    index = seq_along(y),
    y = y,
    residual_kind = kind,
    residual = residual,
    stringsAsFactors = FALSE
  )
  for (nm in names(params)) out[[nm]] <- params[[nm]]
  if (!is.null(pit_values)) out$pit <- pit_values
  if (!is.null(deviance)) out$deviance <- deviance
  out
}

#' Diagnostic residual table for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param kind "quantile" or "response".
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter mstop.
#' @param randomized For discrete families, draw uniformly between CDF jumps.
#' @param ... Not used.
#' @return A data frame with response, parameters, residuals, PIT, and deviance when available.
#' @export
residual_frame.mboostrs_glmboostLSS <- function(object, kind = c("quantile", "response"),
                                                newdata = NULL, mstop = NULL,
                                                randomized = TRUE, ...) {
  .lss_residual_frame_model(object, kind = kind, newdata = newdata,
                            mstop = mstop, randomized = randomized)
}

#' Diagnostic residual table for gamboostLSS
#' @inheritParams residual_frame.mboostrs_glmboostLSS
#' @export
residual_frame.mboostrs_gamboostLSS <- function(object, kind = c("quantile", "response"),
                                                newdata = NULL, mstop = NULL,
                                                randomized = TRUE, ...) {
  .lss_residual_frame_model(object, kind = kind, newdata = newdata,
                            mstop = mstop, randomized = randomized)
}

.lss_q_stats_model <- function(object, xvar = NULL, n_intervals = 10L,
                               newdata = NULL, mstop = NULL,
                               randomized = TRUE) {
  n_intervals <- as.integer(n_intervals)
  if (length(n_intervals) != 1L || n_intervals <= 0L) {
    stop("n_intervals must be positive")
  }
  data <- if (is.null(newdata)) object$data else newdata
  residuals <- quantile_residuals(object, newdata = newdata, mstop = mstop,
                                  randomized = randomized)
  if (is.null(xvar)) {
    x <- seq_along(residuals)
    xvar_name <- "index"
  } else {
    x <- as.double(data[[xvar]])
    xvar_name <- as.character(xvar)
  }
  ord <- order(x)
  groups <- split(ord, cut(seq_along(ord), breaks = min(n_intervals, length(ord)),
                           labels = FALSE))
  rows <- lapply(seq_along(groups), function(i) {
    idx <- groups[[i]]
    r <- residuals[idx]
    m <- mean(r)
    v <- if (length(r) > 1L) stats::var(r) else 0.0
    if (is.finite(v) && v > 0.0) {
      centered <- r - m
      sdv <- sqrt(v)
      skewness <- mean(centered^3) / (sdv^3)
      kurtosis <- mean(centered^4) / (sdv^4) - 3.0
    } else {
      skewness <- 0.0
      kurtosis <- 0.0
    }
    data.frame(
      interval = i,
      xvar = xvar_name,
      x_min = min(x[idx]),
      x_max = max(x[idx]),
      n = length(idx),
      mean = m,
      variance = v,
      skewness = skewness,
      kurtosis = kurtosis,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

#' Binned quantile-residual summaries for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param xvar Optional covariate name used to order bins. Defaults to row index.
#' @param n_intervals Number of bins.
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter mstop.
#' @param randomized For discrete families, draw uniformly between CDF jumps.
#' @param ... Not used.
#' @return A data frame of residual moments by interval.
#' @export
q_stats.mboostrs_glmboostLSS <- function(object, xvar = NULL, n_intervals = 10L,
                                         newdata = NULL, mstop = NULL,
                                         randomized = TRUE, ...) {
  .lss_q_stats_model(object, xvar = xvar, n_intervals = n_intervals,
                     newdata = newdata, mstop = mstop, randomized = randomized)
}

#' Binned quantile-residual summaries for gamboostLSS
#' @inheritParams q_stats.mboostrs_glmboostLSS
#' @export
q_stats.mboostrs_gamboostLSS <- function(object, xvar = NULL, n_intervals = 10L,
                                         newdata = NULL, mstop = NULL,
                                         randomized = TRUE, ...) {
  .lss_q_stats_model(object, xvar = xvar, n_intervals = n_intervals,
                     newdata = newdata, mstop = mstop, randomized = randomized)
}

.lss_control_for_mstop <- function(object, mstop) {
  if (is.list(mstop)) mstop <- unlist(mstop, use.names = TRUE)
  if (identical(object$method, "noncyclic")) {
    if (length(mstop) != 1L) stop("noncyclic LSS update requires a scalar mstop")
    mstop_val <- as.integer(mstop)
    mstop_arg <- NULL
  } else {
    mstop_vec <- if (is.null(names(mstop))) {
      as.integer(rep_len(mstop, length(object$param_names)))
    } else {
      as.integer(mstop[object$param_names])
    }
    if (any(is.na(mstop_vec))) stop("mstop must include all LSS parameters")
    names(mstop_vec) <- object$param_names
    mstop_val <- max(mstop_vec)
    mstop_arg <- mstop_vec
  }
  list(control = boost_control(mstop = mstop_val, nu = object$nu,
                               risk = object$control$risk %||% "inbag"),
       mstop = mstop_arg)
}

.lss_update <- function(object, mstop = NULL, ...) {
  dots <- list(...)
  if (length(dots)) {
    stop("LSS update currently supports only mstop changes")
  }
  if (is.null(mstop)) return(object)
  ctl <- .lss_control_for_mstop(object, mstop)
  formulas <- stats::setNames(object$formulas, object$param_names)
  if (inherits(object, "mboostrs_gamboostLSS")) {
    return(gamboostLSS(formulas, data = object$data, families = object$family,
                       control = ctl$control, offset = object$offset,
                       mstop = ctl$mstop, weights = object$weights,
                       method = object$method))
  }
  glmboostLSS(formulas, data = object$data, families = object$family,
              control = ctl$control, offset = object$offset,
              mstop = ctl$mstop, method = object$method)
}

# Upstream gamboostLSS's mstop(): a named per-parameter vector for a cyclic
# fit, and for a non-cyclic fit the total number of updates with their
# division between parameters in attr(, "partitions"). This used to return
# the first parameter's count alone, which for a non-cyclic fit is neither.
.lss_mstop <- function(object) {
  counts <- vapply(object$params, function(p) length(p$selected), integer(1))
  names(counts) <- object$param_names
  if (identical(object$method, "noncyclic")) {
    total <- sum(counts)
    attr(total, "partitions") <- counts
    return(total)
  }
  counts
}

#' @rdname mstop
#' @export
mstop.mboostrs_glmboostLSS <- function(object, ...) .lss_mstop(object)

#' @rdname mstop
#' @export
mstop.mboostrs_gamboostLSS <- function(object, ...) .lss_mstop(object)

#' @rdname mstop
#' @export
`mstop<-.mboostrs_glmboostLSS` <- function(object, ..., value) {
  update(object, mstop = value)
}

#' @rdname mstop
#' @export
`mstop<-.mboostrs_gamboostLSS` <- function(object, ..., value) {
  update(object, mstop = value)
}

#' Update a glmboostLSS model
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param mstop New scalar or per-parameter mstop.
#' @param ... Not used.
#' @return A refitted LSS model.
#' @export
update.mboostrs_glmboostLSS <- function(object, mstop = NULL, ...) {
  .lss_update(object, mstop = mstop, ...)
}

#' Update a gamboostLSS model
#' @inheritParams update.mboostrs_glmboostLSS
#' @export
update.mboostrs_gamboostLSS <- function(object, mstop = NULL, ...) {
  .lss_update(object, mstop = mstop, ...)
}

#' Residuals for a fitted glmboostLSS model
#'
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param parameter Location-parameter name (usually "mu"). NULL returns a
#'   named list containing the location residuals.
#' @param type "response" (default) or "pearson".
#' @param ... Not used.
#' @return A location residual vector when \code{parameter} is supplied, or a
#'   one-element named list otherwise.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' head(residuals(mod))
#' @export
residuals.mboostrs_glmboostLSS <- function(object, parameter = NULL,
                                            type = "response", ...) {
  if (!type %in% c("response", "pearson")) {
    stop("type must be 'response' or 'pearson'")
  }
  rust <- object$family$rust_name
  if (rust %in% c("lognormallss", "logloglss", "weibulllss")) {
    stop("Response residuals are undefined for ", object$family$name,
         ": the response is censored and the location eta is on the ",
         "log-time scale, so y - mu is not a residual. Use the censored ",
         "contract: quantile residuals (residual_frame(kind='quantile')) ",
         "where available, or score quantities on the fitted etas.")
  }
  if (rust == "dirichletlss") {
    stop("Response residuals are undefined for DirichletLSS: the response ",
         "is a composition vector, not a scalar, so y - mu is not one ",
         "residual.")
  }
  location <- object$param_names[[1L]]
  if (!is.null(parameter) && !identical(parameter, location)) {
    stop("LSS residuals are only defined for the location parameter ('",
         location, "'); score residuals for '", parameter,
         "' are not implemented")
  }

  p <- object$params[[location]]
  mu <- .lss_response(p$fitted, object$family, location)
  y <- .lss_get_response(object)
  r <- if (type == "pearson") {
    .lss_pearson(y, mu, object)
  } else {
    y - mu
  }
  if (!is.null(parameter)) return(r)
  setNames(list(r), location)
}

#' Residuals for a fitted gamboostLSS model
#' @param object A fitted \code{mboostrs_gamboostLSS} object.
#' @param parameter Parameter name (e.g. "mu"). NULL = all parameters as list.
#' @param type "response" (default) or "pearson".
#' @param ... Not used.
#' @return A named list of residual vectors (one per parameter).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' head(residuals(mod))
#' @export
residuals.mboostrs_gamboostLSS <- function(object, parameter = NULL,
                                            type = "response", ...) {
  residuals.mboostrs_glmboostLSS(object, parameter = parameter, type = type, ...)
}

#' Extract LSS response vector
#' @noRd
.lss_get_response <- function(object) {
  is_survival <- object$family$rust_name %in% c("lognormallss", "logloglss", "weibulllss")
  is_multivariate <- object$family$rust_name == "dirichletlss"
  if (is_survival) {
    sv <- all.vars(object$formulas[[1]][[2]])
    return(as.double(object$data[[sv[1]]]))
  }
  if (is_multivariate) {
    rv <- all.vars(object$formulas[[1]][[2]])
    return(as.double(object$data[[rv[1]]]))
  }
  resp_name <- all.vars(object$formulas[[1]])[1]
  as.double(object$data[[resp_name]])
}

#' Pearson residuals for supported LSS location families
#' @noRd
.lss_pearson <- function(y, mu, object) {
  if (object$family$rust_name == "gaussianlss") {
    sigma_name <- object$param_names[[2L]]
    sigma <- .lss_response(
      object$params[[sigma_name]]$fitted,
      object$family,
      sigma_name
    )
    return((y - mu) / pmax(sigma, 1e-10))
  }
  if (object$family$rust_name == "poissonlss") {
    return((y - mu) / sqrt(pmax(mu, 1e-10)))
  }
  stop("Pearson residuals are undefined for ", object$family$name,
       ": they are (y - mu) / sqrt(V(mu)), and this family declares no ",
       "variance function (only gaussianlss and poissonlss do). ")
}

#' Coefficients for a fitted glmboostLSS model
#'
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param parameter Parameter name (e.g. "mu"). NULL = all as list.
#' @param off2int Logical: add offset to intercept?
#' @param ... Not used.
#' @return A named list of coefficient vectors.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' coef(mod)
#' @export
coef.mboostrs_glmboostLSS <- function(object, parameter = NULL, off2int = FALSE, ...) {
  .refuse_unused("coef", ...)
  want <- if (is.null(parameter)) object$param_names else parameter
  out <- list()
  for (pn in want) {
    p <- object$params[[pn]]
    sel <- p$selected
    coefs <- p$coefs
    parsed <- .parse_formula(object$formulas[[match(pn, object$param_names)]],
                             object$data, has_survival = FALSE, need_response = FALSE)
    lnames <- c("(Intercept)", parsed$learner_names)
    result <- setNames(rep(0.0, length(lnames)), lnames)
    for (i in seq_along(sel)) {
      idx <- sel[i]
      nm <- lnames[idx]
      result[nm] <- result[nm] + .lss_param_nu(object, pn) * coefs[i]
    }

    # `glmboostLSS` centers its predictors, so a slope fitted on the centered
    # column implies an intercept shift of `-sum(beta_j * mean_j)` when the
    # coefficients are reported on the original scale. Omitting it reported
    # the intercept as 0 whenever the intercept learner itself was never
    # selected -- which is the usual case, because the offset already carries
    # the level. Measured against `gamboostLSS` on a 200-row fixture: it
    # reports -0.036374 where this reported 0, and -mean(x) * slope is
    # -0.036374 exactly. `mboost::coef.glmboost` applies the same correction,
    # and the mboostrs Python frontend already did.
    centering <- parsed$col_means %||% NULL
    if (is.null(centering)) {
      centering <- vapply(parsed$learner_names, function(nm) {
        column <- object$data[[nm]]
        if (is.numeric(column)) mean(column) else 0.0
      }, numeric(1))
    }
    if (length(centering) == length(parsed$learner_names) &&
        any(centering != 0)) {
      for (j in seq_along(parsed$learner_names)) {
        result["(Intercept)"] <- result["(Intercept)"] -
          result[[parsed$learner_names[j]]] * centering[j]
      }
    }

    if (off2int) result["(Intercept)"] <- result["(Intercept)"] + p$offset
    attr(result, "offset") <- p$offset
    out[[pn]] <- result
  }
  if (!is.null(parameter)) return(out[[parameter]])
  out
}

#' Coefficients for a fitted gamboostLSS model
#'
#' @inheritParams coef.mboostrs_glmboostLSS
#' @return A named list of coefficient blocks.
#' @export
coef.mboostrs_gamboostLSS <- function(object, parameter = NULL, off2int = FALSE, ...) {
  .refuse_unused("coef", ...)
  want <- if (is.null(parameter)) object$param_names else parameter
  out <- list()
  for (pn in want) {
    p <- object$params[[pn]]
    learner_names <- .lss_param_learner_names(object, pn)
    result <- stats::setNames(vector("list", length(learner_names)), learner_names)
    for (i in seq_along(p$selected)) {
      idx <- as.integer(p$selected[i])
      if (idx < 1L || idx > length(learner_names)) next
      value <- .lss_param_nu(object, pn) * as.numeric(p$coefs[[i]])
      if (is.null(result[[idx]])) {
        result[[idx]] <- value
      } else {
        result[[idx]] <- result[[idx]] + value
      }
    }
    for (i in seq_along(result)) {
      if (is.null(result[[i]])) result[[i]] <- 0.0
    }
    attr(result, "offset") <- p$offset
    out[[pn]] <- result
  }
  if (!is.null(parameter)) return(out[[parameter]])
  out
}

.lss_coef_all_model <- function(object, output = c("data.frame", "list")) {
  output <- match.arg(output)
  coefs <- coef(object)
  if (identical(output, "list")) return(coefs)
  rows <- list()
  cursor <- 1L
  for (pn in names(coefs)) {
    for (learner in names(coefs[[pn]])) {
      values <- as.numeric(coefs[[pn]][[learner]])
      for (j in seq_along(values)) {
        rows[[cursor]] <- data.frame(
          parameter = pn,
          learner = learner,
          coefficient_index = j,
          coefficient = values[j],
          stringsAsFactors = FALSE
        )
        cursor <- cursor + 1L
      }
    }
  }
  if (!length(rows)) {
    return(data.frame(parameter = character(), learner = character(),
                      coefficient_index = integer(), coefficient = numeric()))
  }
  do.call(rbind, rows)
}

#' All-parameter coefficients for glmboostLSS
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param output "data.frame" or "list".
#' @param ... Not used.
#' @return A data frame or list of coefficients by parameter.
#' @export
coefAll.mboostrs_glmboostLSS <- function(object, output = c("data.frame", "list"), ...) {
  .lss_coef_all_model(object, output = output)
}

#' All-parameter coefficients for gamboostLSS
#' @inheritParams coefAll.mboostrs_glmboostLSS
#' @export
coefAll.mboostrs_gamboostLSS <- function(object, output = c("data.frame", "list"), ...) {
  .lss_coef_all_model(object, output = output)
}

#' Slice a glmboostLSS model at iteration k
#'
#' @param x A fitted \code{mboostrs_glmboostLSS} object.
#' @param k Iteration number (scalar applied to all parameters, or named vector).
#' @return A sliced LSS model object containing only the first \code{i} iterations per parameter.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboostLSS(y ~ x1 + x2, data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' mod20 <- mod[20]
#' @export
`[.mboostrs_glmboostLSS` <- function(x, k) {
  if (is.null(names(k))) {
    k <- setNames(rep(as.integer(k), length(x$param_names)), x$param_names)
  }
  if (!is.null(x$selection_order)) {
    seen <- integer(length(x$param_names))
    keep <- logical(length(x$selection_order))
    for (step in seq_along(x$selection_order)) {
      parameter <- x$selection_order[step]
      seen[parameter] <- seen[parameter] + 1L
      keep[step] <- seen[parameter] <= k[x$param_names[parameter]]
    }
    x$selection_order <- x$selection_order[keep]
  }
  for (pn in x$param_names) {
    ki <- k[pn]
    if (ki <= 0 || ki > length(x$params[[pn]]$selected))
      stop("k for ", pn, " must be between 1 and ", length(x$params[[pn]]$selected))
    p <- x$params[[pn]]
    p$selected <- p$selected[1:ki]
    p$coefs <- p$coefs[1:ki]
    p$mstop <- ki
    p$fitted <- .Call("wrap__predict_glmboost",
      p$offset, p$selected, p$coefs, .lss_param_nu(x, pn),
      as.double(t(.parse_formula(x$formulas[[match(pn, x$param_names)]],
        x$data, need_response = FALSE)$x_matrix)),
      as.double(p$centering_means), as.integer(ki), PACKAGE = "mboostrs")
    x$params[[pn]] <- p
  }
  if (!is.null(x$execution_report)) {
    x$execution_report$completed_mstop <- if (!is.null(x$selection_order)) {
      length(x$selection_order)
    } else {
      sum(vapply(x$params, function(param) length(param$selected), integer(1)))
    }
  }
  x
}

#' Slice a gamboostLSS model at iteration k
#'
#' @param x A fitted \code{mboostrs_gamboostLSS} object.
#' @param k Iteration number (scalar applied to all parameters, named vector for
#'   cyclic fits, or scalar global step count for noncyclic fits).
#' @return A sliced LSS model object containing only the first \code{k} iterations.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' mod20 <- mod[20]
#' @export
`[.mboostrs_gamboostLSS` <- function(x, k) {
  steps <- .gamboostlss_prediction_steps(x, k)
  fitted_map <- predict(x, type = "link", mstop = k)
  if (identical(x$method, "noncyclic")) {
    global_steps <- min(as.integer(k), length(x$selection_order))
    x$selection_order <- x$selection_order[seq_len(global_steps)]
  } else if (!is.null(x$selection_order)) {
    seen <- integer(length(x$param_names))
    keep <- logical(length(x$selection_order))
    for (step in seq_along(x$selection_order)) {
      parameter <- x$selection_order[step]
      seen[parameter] <- seen[parameter] + 1L
      keep[step] <- seen[parameter] <= steps[[x$param_names[parameter]]]
    }
    x$selection_order <- x$selection_order[keep]
  }
  for (pn in x$param_names) {
    ki <- as.integer(steps[[pn]])
    p <- x$params[[pn]]
    p$selected <- p$selected[seq_len(ki)]
    p$coefs <- p$coefs[seq_len(ki)]
    p$mstop <- ki
    p$fitted <- fitted_map[[pn]]
    x$params[[pn]] <- p
  }
  if (!is.null(x$execution_report)) {
    x$execution_report$completed_mstop <- if (!is.null(x$selection_order)) {
      length(x$selection_order)
    } else {
      sum(vapply(x$params, function(param) length(param$selected), integer(1)))
    }
  }
  x
}

#' @rdname execution_report
#' @export
execution_report.mboostrs_glmboostLSS <- function(object, ...) object$execution_report

#' @rdname execution_report
#' @export
execution_report.mboostrs_gamboostLSS <- function(object, ...) object$execution_report

#' Summary for a fitted glmboostLSS model
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x = rnorm(n))
#' mod <- glmboostLSS(y ~ x, data = dat, families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' summary(mod)
#' @export
summary.mboostrs_glmboostLSS <- function(object, ...) {
  cat("glmboostLSS model\n")
  cat(" Family:", object$family$name, "\n")
  cat(" Parameters:", paste(object$param_names, collapse = ", "), "\n")
  for (pn in object$param_names) {
    p <- object$params[[pn]]
    ms <- length(p$selected)
    cat(sprintf("\n  %s: mstop=%d, offset=%.4f\n", pn, ms, p$offset))
  }
  invisible(object)
}

#' Summary for a fitted gamboostLSS model
#' @param object A fitted \code{mboostrs_gamboostLSS} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' summary(mod)
#' @export
summary.mboostrs_gamboostLSS <- function(object, ...) {
  cat("gamboostLSS model (smooth)\n")
  cat(" Family:", object$family$name, "\n")
  cat(" Parameters:", paste(object$param_names, collapse = ", "), "\n")
  for (pn in object$param_names) {
    p <- object$params[[pn]]
    ms <- length(p$selected)
    cat(sprintf("  %s: mstop=%d, offset=%.4f\n", pn, ms, p$offset))
  }
  invisible(object)
}

#' Log-likelihood for a fitted glmboostLSS model
#' @param object A fitted \code{mboostrs_glmboostLSS} object.
#' @param ... Not used.
#' @return An object of class \code{logLik}. Effective degrees of freedom
#'   are unavailable (\code{df = NA}); \code{update_count} records the
#'   heuristic number of boosting updates plus one separately.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x = rnorm(n))
#' mod <- glmboostLSS(y ~ x, data = dat, families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' logLik(mod)
#' @export
logLik.mboostrs_glmboostLSS <- function(object, ...) {
  risk <- object$risk
  ll <- -risk[length(risk)]
  df <- sum(vapply(object$params, function(p) length(p$selected), integer(1))) + 1
  attr(ll, "df") <- NA_real_
  attr(ll, "update_count") <- df
  attr(ll, "df_convention") <- "unavailable"
  attr(ll, "nobs") <- object$n
  class(ll) <- "logLik"
  ll
}

#' Log-likelihood for a fitted gamboostLSS model
#' @param object A fitted \code{mboostrs_gamboostLSS} object.
#' @param ... Not used.
#' @return An object of class \code{logLik}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                    families = GaussianLSS(),
#'                    control = boost_control(mstop = 50))
#' logLik(mod)
#' @export
logLik.mboostrs_gamboostLSS <- function(object, ...) {
  logLik.mboostrs_glmboostLSS(object, ...)
}

#' cvrisk for gamboostLSS (smooth base-learners)
#'
#' Each fold refits on its in-bag weight column in the engine and the
#' out-of-bag risk is evaluated there at every grid point, so the reported
#' quantity is the fitted families' risk rather than a second loss written in
#' R.
#'
#' @param formula,data,families,family,control As in \code{\link{gamboostLSS}}.
#' @param folds An \code{n x B} matrix of in-bag weights.
#' @param grid An \code{n_grid x K} matrix of per-parameter mstop values.
#' @return A \code{B x n_grid} matrix of per-fold OOB risks.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 50
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' folds <- cv_weights(n, b = 3)
#' grid <- expand.grid(c(10, 20), c(10, 20))
#' cr <- cvrisk_gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                          families = GaussianLSS(),
#'                          control = boost_control(mstop = 20),
#'                          folds = folds, grid = grid)
#' }
#' @export
cvrisk_gamboostLSS <- function(formula, data, families = NULL, family = NULL,
                                control = boost_control(), folds, grid) {
  families <- .lss_resolve_families(families, family)
  pnames <- families$param_names
  K <- length(pnames)
  frontend <- .gamboostlss_frontend(formula, data, families)
  B <- ncol(folds)
  if (is.null(B) || nrow(folds) != frontend$n || B < 1L) {
    stop("folds must be an n x B matrix with at least one column")
  }
  if (any(!is.finite(folds)) || any(folds < 0)) {
    stop("fold weights must be finite and non-negative")
  }
  grid <- as.matrix(grid)
  n_grid <- nrow(grid)
  if (n_grid < 1L || ncol(grid) != K || any(!is.finite(grid)) || any(grid < 0)) {
    stop(sprintf("grid must have at least one row and exactly %d non-negative columns", K))
  }
  if (is.null(colnames(grid))) colnames(grid) <- pnames

  risk_vec <- .Call("wrap__cvrisk_gamboostlss_mixed",
    frontend$specs,
    as.double(frontend$y),
    as.double(frontend$weights),
    as.integer(control$mstop),
    as.double(.lss_nu_scalar(control$nu)),
    families$rust_name,
    as.double(t(folds)),
    as.integer(B),
    as.integer(t(grid)),
    as.integer(n_grid),
    .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
    PACKAGE = "mboostrs"
  )

  risk_mat <- matrix(risk_vec, nrow = B, ncol = n_grid, byrow = TRUE)
  colnames(risk_mat) <- apply(grid, 1, paste, collapse = ",")
  .lss_cvrisk_result(risk_mat, grid)
}

#' Stability selection for a smooth GAMLSS (gamboostLSS).
#'
#' @param formula,data,families,family,control As in \code{\link{gamboostLSS}}.
#' @param folds An \code{n x B} matrix of in-bag weights holding the
#'   \emph{initial} subsamples, one column per subsample (\code{"MB"}) or per
#'   complementary pair (\code{"SS"}). Under \code{"SS"} the exact complements
#'   are appended here, so \code{B} columns produce \code{2 * B} fits.
#'   \code{\link{stabsel_folds}} builds one.
#' @param q,PFER,cutoff Stability-selection parameters.
#' @param assumption,sampling_type Passed to error-control computation.
#'   \code{sampling_type = "SS"} uses complementary pairs and fits
#'   \code{2 * B} models; \code{"MB"} fits \code{B}.
#' @return A list with \code{selected}, \code{max}, \code{cutoff}, \code{q}, \code{PFER}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' folds <- cv_weights(n, b = 50)
#' sel <- stabsel_gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dat,
#'                            families = GaussianLSS(),
#'                            control = boost_control(mstop = 50),
#'                            folds = folds)
#' sel$selected
#' }
#' @export
stabsel_gamboostLSS <- function(formula, data, families = NULL, family = NULL,
                                 control = boost_control(), folds,
                                 q = -1L, PFER = -1.0, cutoff = -1.0,
                                 assumption = "unimodal", sampling_type = "MB") {
  families <- .lss_resolve_families(families, family)
  frontend <- .gamboostlss_frontend(formula, data, families)
  sampling_type <- .stabsel_sampling_type(sampling_type)
  prepared <- .stabsel_fit_folds(folds, sampling_type, frontend$n)
  folds <- prepared$folds; B <- prepared$B; n_fits <- prepared$n_fits
  combined <- unlist(Map(function(specs, parameter) {
    paste0(vapply(specs, `[[`, character(1), "name"), ".", parameter)
  }, frontend$specs, families$param_names), use.names = FALSE)
  p <- length(combined)
  par <- stabsel_params(p = p, b = B,
                        cutoff = cutoff, q = q, pfer = PFER,
                        assumption = assumption, sampling_type = sampling_type)
  cutoff_val <- par$cutoff
  q_val <- par$q

  seqs <- .Call("wrap__stabsel_selected_gamboostlss_mixed",
    frontend$specs, as.double(frontend$y), as.double(frontend$weights),
    as.integer(control$mstop), as.double(.lss_nu_scalar(control$nu)), families$rust_name,
    as.double(t(folds)), as.integer(n_fits), .lss_stabilization(families), .lss_nu_vector(control$nu, families$param_names),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
 PACKAGE = "mboostrs")

  m <- control$mstop
  phat <- .lss_stabsel_frequencies(seqs, p, m, q_val)
  max_phat <- apply(phat, 1, max)
  selected <- combined[max_phat >= cutoff_val]
  .lss_stabsel_result(
    phat = phat,
    selected = selected,
    max_phat = max_phat,
    cutoff = cutoff_val,
    q = q_val,
    PFER = par$pfer,
    learner_names = combined,
    assumption = assumption,
    sampling_type = sampling_type,
    B = B,
    n_fits = n_fits
  )
}

# Upstream's LSS-only settings, passed to every fit, cross-validation and
# stability-selection binding. Stabilization is a family argument, as in
# gamboostLSS 2.2-0 (`GaussianLSS(stabilization = "MAD")`).
.lss_stabilization <- function(families) families$stabilization %||% "none"

# boost_control(nu = ...) for a distributional fit, read as upstream reads it:
# a scalar is every parameter's step; a named vector is reordered to the
# family's parameters; an unnamed one is taken in family order. Empty means
# "use the scalar".
.lss_nu_vector <- function(nu, params) {
  if (length(nu) <= 1L) return(numeric(0))
  if (!is.null(names(nu)) && all(nzchar(names(nu)))) {
    if (!setequal(names(nu), params)) {
      stop("boost_control(nu = ...) names ", paste(names(nu), collapse = ", "),
           "; the family's parameters are ", paste(params, collapse = ", "),
           call. = FALSE)
    }
    nu <- nu[params]
  } else if (length(nu) != length(params)) {
    stop("boost_control(nu = ...) has ", length(nu), " values for ",
         length(params), " parameters (", paste(params, collapse = ", "), ")",
         call. = FALSE)
  }
  as.double(nu)
}

# The scalar step the binding's control is built with; per-parameter steps,
# when given, replace it for every update.
.lss_nu_scalar <- function(nu) as.double(nu[[1L]])

# The step length of parameter `pn` of a fitted LSS model: its entry of a
# per-parameter boost_control(nu = ...), else the scalar.
.lss_param_nu <- function(object, pn) {
  per <- .lss_nu_vector(object$nu, object$param_names)
  if (length(per)) per[[match(pn, object$param_names)]] else as.double(object$nu)
}

# The new rows' `by` modifier for each learner of a parameter (NULL where it
# has none, or when predicting on the training rows). A numeric
# bbs(x, by = z) multiplies its design by z; without this the binding used the
# training rows' z, 8.5 from gamboostLSS on new data.
.lss_by_rows <- function(specs, newdata) {
  lapply(specs, function(spec) {
    if (is.null(newdata) || is.null(spec$by_name) || is.null(spec$by_values)) return(NULL)
    z <- newdata[[spec$by_name]]
    if (is.null(z)) stop(.missing_column_message(spec$by_name, newdata))
    as.double(z)
  })
}


# predict()'s extrapolation= policy for one learner of distribution parameter
# `pn`: spline learners (bbs, bmono, bns, and each bspatial coordinate) are
# measured against their training boundary knots, as in gamboost.
.lss_extrapolate <- function(spec, values, extrap, pn) {
  label <- function(name) sprintf("%s: %s", pn, name)
  if (spec$type %in% c("bbs", "bmono", "bns") && !is.null(spec$bk_lo)) {
    return(extrap$check(values, spec$bk_lo, spec$bk_hi, label(spec$name)))
  }
  if (identical(spec$type, "bspatial") && !is.null(spec$bk1_lo)) {
    n <- length(values) / 2
    term <- label(sprintf("bspatial(%s, %s)", spec$name1, spec$name2))
    return(c(extrap$check(values[seq_len(n)], spec$bk1_lo, spec$bk1_hi, term, spec$name1),
             extrap$check(values[n + seq_len(n)], spec$bk2_lo, spec$bk2_hi, term, spec$name2)))
  }
  values
}
