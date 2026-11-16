# Gradient stabilization and per-parameter nu in the R LSS frontend (#68).
# They were Python-only; R and Python fits now agree exactly
# (tests/test_lss_r_settings.py checks that from Python).
suppressMessages(library(mboostrs))
set.seed(0); n <- 200
d <- data.frame(x = rnorm(n), z = rnorm(n))
d$y <- rnorm(n, sin(d$x), exp(0.3 * d$z))
f <- list(mu = y ~ bbs(x) + bols(z), sigma = y ~ bols(z))
ctl <- boost_control(mstop = 40, nu = c(sigma = 0.05, mu = 0.1))

plain <- gamboostLSS(f, data = d, families = GaussianLSS(), control = boost_control(mstop = 40))
mad <- gamboostLSS(f, data = d, families = GaussianLSS(stabilization = "MAD"), control = ctl)
stopifnot(max(abs(predict(plain, type = "link")$mu - predict(mad, type = "link")$mu)) > 1e-3)

# Named and unnamed nu in family order give the same fit.
unnamed <- gamboostLSS(f, data = d, families = GaussianLSS(stabilization = "MAD"),
                       control = boost_control(mstop = 40, nu = c(0.1, 0.05)))
stopifnot(identical(predict(mad, type = "link"), predict(unnamed, type = "link")))

# glmboostLSS takes them too; cvrisk and stabsel accept the settings.
g <- glmboostLSS(y ~ x + z, data = d, families = GaussianLSS(stabilization = "L2"),
                 control = ctl)
stopifnot(all(is.finite(unlist(predict(g, type = "link")))))
folds <- cv_weights(n, fold_type = "kfold", b = 3)
invisible(cvrisk_gamboostLSS(f, data = d, families = GaussianLSS(stabilization = "MAD"),
                             control = ctl, folds = folds, grid = rbind(c(20, 20), c(40, 40))))
invisible(cvrisk_glmboostLSS(y ~ x + z, data = d, families = GaussianLSS(stabilization = "L2"),
                             control = ctl, folds = folds, grid = rbind(c(20, 20), c(40, 40))))

# Refusals: a wrongly named nu, and a vector nu for a standard fit.
msg <- tryCatch(gamboostLSS(f, data = d, families = GaussianLSS(),
                            control = boost_control(mstop = 10, nu = c(mu = 0.1, tau = 0.1))),
                error = conditionMessage)
stopifnot(grepl("parameters are mu, sigma", msg))
msg <- tryCatch(gamboost(y ~ bbs(x), data = d, control = boost_control(nu = c(0.1, 0.2))),
                error = conditionMessage)
stopifnot(grepl("glmboostLSS\\(\\) and gamboostLSS\\(\\) only", msg))
msg <- tryCatch(GaussianLSS(stabilization = "max"), error = conditionMessage)
stopifnot(grepl("should be one of", msg))
cat("test_lss_stabilization_nu.R: OK\n")
