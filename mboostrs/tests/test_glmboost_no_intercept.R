# glmboost(y ~ x - 1) against mboost (issue #101): no intercept learner, the
# offset kept, the covariates centered.
if (!requireNamespace("mboost", quietly = TRUE)) {
  cat("test_glmboost_no_intercept.R: skipped (mboost not installed)\n"); quit(save = "no")
}
suppressMessages(library(mboost))
set.seed(0); n <- 200
d <- data.frame(x0 = rnorm(n, 1), x1 = rnorm(n), x2 = rnorm(n))
d$y <- 1.5 + 2 * d$x0 - d$x1 + rnorm(n, sd = 0.5)
d$b <- factor(rbinom(n, 1, plogis(d$x0 - 1))); d$b01 <- as.integer(d$b) - 1L
nd <- data.frame(x0 = c(0, 2), x1 = c(1, -1), x2 = c(0, 0.5))
f <- cv(rep(1, n), type = "kfold", B = 5)
for (case in list(list(up = y ~ x0 + x1 + x2 - 1, ours = y ~ x0 + x1 + x2 - 1, fam = "Gaussian"),
                  list(up = b ~ 0 + x0 + x1 + x2, ours = b01 ~ 0 + x0 + x1 + x2, fam = "Binomial"))) {
  up <- suppressWarnings(mboost::glmboost(case$up, data = d, family = get(case$fam, asNamespace("mboost"))(),
                                          control = mboost::boost_control(mstop = 80)))
  ours <- suppressWarnings(mboostrs::glmboost(case$ours, data = d,
                                              family = get(case$fam, asNamespace("mboostrs"))(),
                                              control = mboostrs::boost_control(mstop = 80)))
  cu <- suppressMessages(coef(up))
  gap <- max(abs(fitted(up) - ours$fitted),
             abs(predict(up, newdata = nd) - predict(ours, newdata = nd)),
             abs(cu - coef(ours)[names(cu)]),
             abs(colMeans(cvrisk(up, folds = f, papply = lapply)) -
                 colMeans(as.matrix(mboostrs::cvrisk(ours, folds = f)$risk))))
  if (gap > 1e-12) stop(case$fam, ": gap ", gap)
  stopifnot(!("(Intercept)" %in% names(coef(ours))))
}
cat("test_glmboost_no_intercept.R: OK\n")

# glmboost(center = FALSE), as mboost: the raw columns in the fit and in every
# cvrisk fold (the folds were refitted centered, 0.30 from mboost's cvrisk).
for (cen in c(TRUE, FALSE)) {
  up <- mboost::glmboost(y ~ x0 + x1 + x2, data = d, center = cen, control = mboost::boost_control(mstop = 60))
  ours <- mboostrs::glmboost(y ~ x0 + x1 + x2, data = d, center = cen, control = mboostrs::boost_control(mstop = 60))
  gap <- max(abs(fitted(up) - ours$fitted),
             abs(colMeans(cvrisk(up, folds = f, papply = lapply)) -
                 colMeans(as.matrix(mboostrs::cvrisk(ours, folds = f)$risk))))
  if (gap > 1e-12) stop("center = ", cen, ": gap ", gap)
}
cat("test_glmboost_no_intercept.R (center): OK\n")
