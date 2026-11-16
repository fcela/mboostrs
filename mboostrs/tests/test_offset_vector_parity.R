# A per-row offset (log exposure) in the R frontends, against mboost.
#
# `glmboost(offset = log(e))` and `gamboost(offset = log(e))` refused a vector
# ("Expected Scalar, got Doubles"), so a count per unit of exposure could not
# be fitted in R at all. The offset is one known contribution per row: it
# starts the fit, is never re-estimated, enters every cross-validation fold,
# and has to be supplied again for new rows.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping offset vector parity check")
  quit(save = "no")
}
suppressMessages(library(mboost))
library(mboostrs, warn.conflicts = FALSE)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(1)
n <- 300
x <- runif(n, -1, 1); z <- runif(n, -1, 1); e <- runif(n, 0.2, 5)
d <- data.frame(y = rpois(n, e * exp(0.3 + 0.6 * x)), x = x, z = z)
folds <- mboost::cv(rep(1, n), type = "kfold", B = 5)

g <- mboostrs::glmboost(y ~ x + z, data = d, family = mboostrs::Poisson(),
                        offset = log(e), control = mboostrs::boost_control(mstop = 100))
gm <- mboost::glmboost(y ~ x + z, data = d, family = mboost::Poisson(),
                       offset = log(e), control = mboost::boost_control(mstop = 100))
ok(max(abs(predict(g) - predict(gm))) < 1e-7, "glmboost fitted values differ from mboost")
cv_g <- mboostrs::cvrisk(g, folds = folds)
cv_gm <- mboost::cvrisk(gm, folds = folds, papply = lapply)
ok(max(abs(cv_g$aggregate - colMeans(cv_gm))) < 1e-7, "glmboost cvrisk differs from mboost")

b <- mboostrs::gamboost(y ~ bbs(x) + bols(z), data = d, family = mboostrs::Poisson(),
                        offset = log(e), control = mboostrs::boost_control(mstop = 100))
bm <- mboost::gamboost(y ~ mboost::bbs(x) + mboost::bols(z), data = d,
                       family = mboost::Poisson(), offset = log(e),
                       control = mboost::boost_control(mstop = 100))
ok(max(abs(predict(b) - predict(bm))) < 1e-10, "gamboost fitted values differ from mboost")
cv_b <- mboostrs::cvrisk(b, folds = folds)
cv_bm <- mboost::cvrisk(bm, folds = folds, papply = lapply)
ok(max(abs(cv_b$aggregate - colMeans(cv_bm))) < 1e-10, "gamboost cvrisk differs from mboost")

# New rows need their own offset; with it, the prediction is the training
# prediction's shape shifted by the new log exposure.
nd <- data.frame(x = d$x[1:3], z = d$z[1:3])
err <- tryCatch(predict(b, newdata = nd), error = function(e) conditionMessage(e))
ok(grepl("per-row offset", err), "predict on new rows without offset should refuse")
p_new <- predict(b, newdata = nd, offset = log(e[1:3]))
ok(max(abs(p_new - predict(b)[1:3])) < 1e-10, "predict(offset=) on training rows disagrees")
p_glm <- predict(g, newdata = nd, offset = log(e[1:3]))
ok(max(abs(p_glm - predict(g)[1:3])) < 1e-10, "glmboost predict(offset=) disagrees")

# New-data prediction of block-form learners (penalized `bols`, `brandom`)
# read a column-major design as row-major and was wrong without any offset.
g6 <- sample(1:6, n, TRUE)
d6 <- data.frame(y = rnorm(n) + x + z + g6 / 3, x = x, z = z, g = g6)
for (f in c("y ~ bols(z, lambda = 1)", "y ~ bbs(x) + brandom(g) + bols(z, lambda = 2)")) {
  fit <- mboostrs::gamboost(as.formula(f), data = d6)
  rows <- c(5, 1, 3)
  ok(max(abs(predict(fit, newdata = d6[rows, ]) - predict(fit)[rows])) < 1e-10,
     paste("new-data prediction disagrees with the fitted values for", f))
}

# A wrong-length offset is refused by name.
err <- tryCatch(mboostrs::glmboost(y ~ x, data = d, offset = rep(0, 5)),
                error = function(e) conditionMessage(e))
ok(grepl("offset has 5 entries", err), "wrong-length offset should be refused")

# Stability selection passes the offset to every subsample fit.
wide <- cbind(d, as.data.frame(setNames(replicate(6, runif(n, -1, 1), simplify = FALSE),
                                          paste0("u", 1:6))))
st <- mboostrs::stabsel(y ~ ., data = wide, family = mboostrs::Poisson(),
                        offset = log(e), control = mboostrs::boost_control(mstop = 100),
                        cutoff = 0.75, q = 3L, B = 25L)
ok("x" %in% st$selected, "stabsel with offset should select x")
cat("offset vector parity: ok\n")
