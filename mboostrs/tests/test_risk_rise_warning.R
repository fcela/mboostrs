# A fixed step too large for the loss's curvature is reported (issue #94).
library(mboostrs)
set.seed(0); x <- runif(300); d <- data.frame(x = x, y = rpois(300, 3 * exp(x)))
caught <- NULL
withCallingHandlers(
  gamboost(y ~ bbs(x), data = d, family = Poisson(), control = boost_control(mstop = 100, nu = 0.5)),
  warning = function(w) { caught <<- conditionMessage(w); invokeRestart("muffleWarning") })
stopifnot(!is.null(caught), grepl("training risk rose", caught, fixed = TRUE))
quiet <- TRUE
withCallingHandlers(
  gamboost(y ~ bbs(x), data = d, family = Poisson(), control = boost_control(mstop = 100, nu = 0.1)),
  warning = function(w) { if (grepl("training risk rose", conditionMessage(w))) quiet <<- FALSE
                          invokeRestart("muffleWarning") })
stopifnot(quiet)

# Small increases can accumulate to a material profiled-likelihood backslide.
for (family in list(NBinomial(), Hurdle(), Weibull(), Loglog(), Lognormal(),
                    PropOdds(n_levels = 3))) {
  params <- if (is.null(family$params)) list() else family$params
  found <- .Call("wrap__r_risk_path_rises", family$name, params,
                 c(1000, 900, 900.5, 901.1), PACKAGE = "mboostrs")
  stopifnot(length(found) == 4L, found[1] == 2, found[2] < .01,
            found[3] > .001, found[4] == 1)
}
set.seed(2); d <- data.frame(x = rnorm(400))
mu <- 200 * exp(.5 * d$x)
d$y <- rnbinom(nrow(d), size = 20, mu = mu)
for (kind in c("glm", "gam")) {
  fit <- if (kind == "glm") glmboost else gamboost
  formula <- if (kind == "glm") y ~ x else y ~ bols(x)
  caught <- character()
  fixed <- withCallingHandlers(
    fit(formula, d, family = NBinomial(), control = boost_control(400, .3)),
    warning = function(w) { caught <<- c(caught, conditionMessage(w)); invokeRestart("muffleWarning") })
  stopifnot(any(grepl("training risk rose", caught)),
            fixed$risk[length(fixed$risk)] > min(fixed$risk) * 1.001,
            fixed$control$step == "fixed", fixed$mstop == 400)
  searched <- fit(formula, d, family = NBinomial(),
                   control = boost_control(400, .3, step = "line_search"))
  stopifnot(max(diff(searched$risk)) < 1e-7)
}
cat("test_risk_rise_warning.R: OK\n")
