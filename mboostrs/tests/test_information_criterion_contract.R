library(mboostrs)

set.seed(91)
dat <- data.frame(x = rnorm(100), y = rnorm(100))
for (kind in c("glm", "gam")) {
  model <- if (kind == "glm") glmboost(y ~ x, dat) else gamboost(y ~ bols(x), dat)
  criterion <- AIC(model)
  df <- attr(criterion, "df")
  expected <- log(model$risk[-1] / 100) + (1 + df / 100) / (1 - (df + 2) / 100)
  stopifnot(isTRUE(all.equal(aic_path(criterion), expected)))
  stopifnot(identical(attr(criterion, "criterion"), "mboost_corrected"))
  stopifnot(identical(attr(criterion, "normalization"), "per_observation"))
  stopifnot(inherits(tryCatch(AIC(model, method = "classical"), error = identity), "error"))
  stopifnot(inherits(tryCatch(AIC(model, k = log(100)), error = identity), "error"))
  gmdl <- AIC(model, method = "gmdl")
  residual <- model$risk[-1]
  sigma2 <- residual / (100 - df)
  expected <- log(sigma2) + df / 100 * log((sum(dat$y^2) - residual) / (df * sigma2))
  stopifnot(isTRUE(all.equal(aic_path(gmdl), expected)))
}

for (family in list(Binomial(), Poisson())) {
  observed <- dat
  observed$y <- if (family$name == "binomial") as.numeric(dat$y > 0) else rpois(100, 2)
  model <- glmboost(y ~ x, observed, family = family)
  criterion <- AIC(model, method = "classical", k = 3)
  stopifnot(isTRUE(all.equal(aic_path(criterion), 2 * model$risk[-1] + 3 * attr(criterion, "df"))))
  stopifnot(identical(attr(criterion, "normalization"), "total"))
  stopifnot(inherits(tryCatch(AIC(model), error = identity), "error"))
}

model <- glmboostLSS(y ~ x, dat, family = GaussianLSS())
ll <- logLik(model)
stopifnot(is.na(attr(ll, "df")))
stopifnot(identical(attr(ll, "df_convention"), "unavailable"))
stopifnot(attr(ll, "update_count") == sum(vapply(model$params, function(p) length(p$selected), integer(1))) + 1)
