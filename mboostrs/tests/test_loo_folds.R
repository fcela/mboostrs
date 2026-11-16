# loo_folds(): leave-one-out, mboost's cv(type = "kfold", B = n) (issue #100).
# mboost draws its columns in a random order, so the risk is compared averaged
# over folds, which does not depend on that order.
if (!requireNamespace("mboost", quietly = TRUE)) {
  cat("test_loo_folds.R: skipped (mboost not installed)\n"); quit(save = "no")
}
set.seed(3); n <- 30
d <- data.frame(x1 = runif(n), x2 = rnorm(n))
d$y <- 2 * d$x1 - d$x2 + rnorm(n, sd = 0.3)

folds <- mboostrs::loo_folds(n)
stopifnot(all(dim(folds) == n), all(diag(folds) == 0),
          all(folds[row(folds) != col(folds)] == 1))

ours <- mboostrs::glmboost(y ~ x1 + x2, data = d,
                           control = mboostrs::boost_control(mstop = 40))
up <- mboost::glmboost(y ~ x1 + x2, data = d, control = mboost::boost_control(mstop = 40))
r_ours <- colMeans(as.matrix(mboostrs::cvrisk(ours, folds = folds)$risk))
r_up <- colMeans(mboost::cvrisk(up, folds = mboost::cv(model.weights(up), type = "kfold", B = n),
                                papply = lapply))
gap <- max(abs(r_ours - as.numeric(r_up)) / abs(as.numeric(r_up)))
if (gap > 1e-10) stop("leave-one-out risk differs from mboost: relative gap ", gap)

# A zero-weight row is out of the sample and held out by no fold.
w <- rep(1, 6); w[2] <- 0
f0 <- mboostrs::loo_folds(6, weights = w)
stopifnot(ncol(f0) == 5L, all(f0[2, ] == 0))
stopifnot(inherits(try(mboostrs::loo_folds(1), silent = TRUE), "try-error"))
cat("test_loo_folds.R: OK\n")
