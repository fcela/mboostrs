# One weight contract, matching upstream's, in the R frontend.
#
# A boosting fit uses weights in three places at once: the base-learner normal
# equations (X'WX), the df -> lambda calibration that reads the hat matrix of
# those same equations, and the empirical risk. All three scale with the
# weight vector, so the *scale* of the weights, not only their ratios, changes
# the fitted path.
#
# `mboost_fit` settles that scale before anything else runs, with
#
#   rescale_weights <- function(w) {
#       if (max(abs(w - floor(w))) < sqrt(.Machine$double.eps)) return(w)
#       return(w / sum(w) * sum(w > 0))
#   }
#
# Integer weights are frequencies and keep their scale; fractional weights are
# analytic and are rescaled to sum to the number of weight-positive rows.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
near <- function(a, b, tol = 1e-6) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = tol))

# A ten-row fixture whose weights sum to 8.4 over 10 positive rows, so the
# rescaling factor is 10 / 8.4 and no value survives it unchanged.
y <- c(-0.3718140203505656, -0.22288374723536017, -1.0441854737522702,
       -0.008991429626466318, -2.3172496935752873, -0.009077244884482139,
       2.0637411074029646, 1.5778617698893298, -0.9958399352398524,
       -1.4883295726494274)
x <- c(0.1257302210933933, -0.1321048632913019, 0.6404226504432821,
       0.10490011715303971, -0.535669373161111, 0.36159505490948474,
       1.3040000451301372, 0.9470809631292422, -0.7037352358069926,
       -1.2654214710460525)
d <- data.frame(y = y, x = x)
w <- c(0.3, 0.7, 1.2, 0.5, 2.4, 0.1, 0.9, 1.1, 0.4, 0.8)

# ─── the policy itself ─────────────────────────────────────────────

ok(mboostrs:::.weights_are_integral(c(1, 3, 0, 2)), "integers should be integral")
ok(mboostrs:::.weights_are_integral(c(1, 2 + 1e-12)), "tolerance should absorb 1e-12")
ok(!mboostrs:::.weights_are_integral(c(1, 2.5)), "2.5 is not an integer")

ok(identical(mboostrs:::.rescale_weights(c(1, 3, 0, 2)), c(1, 3, 0, 2)),
   "integer frequencies must reach the fit unchanged")

rescaled <- mboostrs:::.rescale_weights(w)
ok(near(sum(rescaled), 10), "fractional weights should sum to the positive-row count")
ok(near(rescaled / w, rep(10 / 8.4, 10)), "only the scale should move, not the ratios")

# A zero-weight row is not one of the rows the scale is set to.
z <- mboostrs:::.rescale_weights(c(0.5, 0, 1.5, 0))
ok(near(sum(z), 2), "the scale should count positive rows only")
ok(z[2] == 0 && z[4] == 0, "zero weights should stay zero")

norm <- mboostrs:::.normalize_weights(w, 10)
ok(near(norm$given, w), "the caller's vector should be kept")
ok(near(sum(norm$fit), 10), "the effective vector should be rescaled")
ok(isTRUE(norm$rescaled), "the rescaling should be recorded")
ok(grepl("rescaled to sum to 10", mboostrs:::.weights_note(norm)),
   "the note should say what happened")

raw <- mboostrs:::.normalize_weights(w, 10, policy = "raw")
ok(near(raw$fit, w), "policy=raw should pass the vector through")
ok(grepl("no rescaling", mboostrs:::.weights_note(raw)), "the raw note should say so")

unit <- mboostrs:::.normalize_weights(NULL, 5)
ok(is.null(unit$given) && near(unit$fit, rep(1, 5)), "NULL means unit weights")
ok(identical(mboostrs:::.weights_note(unit), "unit weights"), "unit note")

for (case in list(
  list(w = rep(1, 3), n = 10, msg = "wrong length"),
  list(w = c(rep(1, 9), NA), n = 10, msg = "non-finite"),
  list(w = c(rep(1, 9), -1), n = 10, msg = "negative"),
  list(w = rep(0, 10), n = 10, msg = "all zero")
)) {
  res <- try(mboostrs:::.normalize_weights(case$w, case$n), silent = TRUE)
  ok(inherits(res, "try-error"), paste("should reject:", case$msg))
}
ok(inherits(try(mboostrs:::.normalize_weights(rep(1, 4), 4, policy = "frequency"),
                silent = TRUE), "try-error"),
   "an unknown policy should be rejected")

# ─── the fitting entry points ──────────────────────────────────────

# Taken from R:
#   glmboost(y ~ x, data = d, weights = w, control = boost_control(mstop = 30, nu = 0.1))
fit <- glmboost(y ~ x, data = d, weights = w,
                control = boost_control(mstop = 30, nu = 0.1))
ok(near(unname(coef(fit)["x"]), 1.529009),
   sprintf("glmboost x coefficient should match mboost: got %.7f", coef(fit)["x"]))

# A global rescaling of analytic weights is inert under the default policy:
# it is the same effective vector, so the fitted values *and* the risk agree.
scaled <- glmboost(y ~ x, data = d, weights = 2 * w,
                   control = boost_control(mstop = 30, nu = 0.1))
ok(near(fitted(scaled), fitted(fit), 1e-10),
   "rescaled analytic weights: fitted should match")
ok(near(scaled$risk, fit$risk, 1e-8),
   "rescaled analytic weights: risk should match too")

# Under policy="raw" the caller's scale survives, so the risk scales with it
# while the fitted values do not.
raw_fit <- glmboost(y ~ x, data = d, weights = w, weight_policy = "raw",
                    control = boost_control(mstop = 30, nu = 0.1))
raw_2w <- glmboost(y ~ x, data = d, weights = 2 * w, weight_policy = "raw",
                   control = boost_control(mstop = 30, nu = 0.1))
ok(near(fitted(raw_2w), fitted(raw_fit), 1e-10), "raw: fitted should match")
ok(near(raw_2w$risk, 2 * raw_fit$risk, 1e-8), "raw: risk should scale by 2x")
ok(!near(raw_fit$risk, fit$risk, 1e-6),
   "raw and upstream must differ, or the policy does nothing")

# Integer weights are frequencies: their scale is meaningful and is kept.
wi <- c(1, 2, 1, 3, 1, 2, 1, 1, 2, 1)
fi <- glmboost(y ~ x, data = d, weights = wi,
               control = boost_control(mstop = 30, nu = 0.1))
f2i <- glmboost(y ~ x, data = d, weights = 2 * wi,
                control = boost_control(mstop = 30, nu = 0.1))
ok(near(fitted(f2i), fitted(fi), 1e-10), "integer weights: fitted should match")
ok(near(f2i$risk, 2 * fi$risk, 1e-8),
   "integer frequency weights: risk should scale with them")

# gamboost applies the same policy. Its explicit `lambda=1` is the case that
# exposes a weight-scale error: `df -> lambda` would absorb one by moving
# lambda, an explicit lambda has nothing to absorb it with.
g <- gamboost(y ~ bols(x, lambda = 1), data = d, weights = w,
              control = boost_control(mstop = 20, nu = 0.1))
g2 <- gamboost(y ~ bols(x, lambda = 1), data = d, weights = 3 * w,
               control = boost_control(mstop = 20, nu = 0.1))
ok(near(fitted(g2), fitted(g), 1e-10),
   "gamboost rescaled weights: fitted should match")
graw <- gamboost(y ~ bols(x, lambda = 1), data = d, weights = w,
                 weight_policy = "raw",
                 control = boost_control(mstop = 20, nu = 0.1))
ok(!near(fitted(graw), fitted(g), 1e-6),
   "gamboost raw and upstream must differ on fractional weights")

# All-zero weights are rejected on every entry point.
for (call in list(
  function() glmboost(y ~ x, data = d, weights = rep(0, 10),
                      control = boost_control(mstop = 5)),
  function() gamboost(y ~ bols(x), data = d, weights = rep(0, 10),
                      control = boost_control(mstop = 5))
)) {
  ok(inherits(try(call(), silent = TRUE), "try-error"),
     "all-zero weights should be rejected")
}

# ─── LSS entry points ──────────────────────────────────────────────

set.seed(11); n <- 100
dl <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
dl$y <- rnorm(n, 1 + 0.8 * dl$x1, exp(0.3 * dl$x2))
wl <- runif(n, 0.4, 2.5)

l1 <- glmboostLSS(y ~ x1 + x2, data = dl, families = GaussianLSS(),
                  weights = wl, control = boost_control(mstop = 30, nu = 0.1))
l2 <- glmboostLSS(y ~ x1 + x2, data = dl, families = GaussianLSS(),
                  weights = 2 * wl, control = boost_control(mstop = 30, nu = 0.1))
ok(near(l1$risk, l2$risk, 1e-8),
   "glmboostLSS: a global rescaling of analytic weights should be inert")

gl1 <- gamboostLSS(y ~ bbs(x1, df = 3) + bbs(x2, df = 3), data = dl,
                   families = GaussianLSS(), weights = wl,
                   control = boost_control(mstop = 20, nu = 0.1))
gl2 <- gamboostLSS(y ~ bbs(x1, df = 3) + bbs(x2, df = 3), data = dl,
                   families = GaussianLSS(), weights = 2 * wl,
                   control = boost_control(mstop = 20, nu = 0.1))
ok(near(gl1$risk, gl2$risk, 1e-6),
   "gamboostLSS: a global rescaling of analytic weights should be inert")

ok(inherits(try(glmboostLSS(y ~ x1, data = dl, families = GaussianLSS(),
                            weights = rep(0, n),
                            control = boost_control(mstop = 5)), silent = TRUE),
            "try-error"),
   "LSS should reject all-zero weights")


# A 0/1 fold mask given to a weighted glmboost refits each fold with the
# model's weights: the curve equals the one from folds that carry them.
set.seed(4); n_fm <- 120; x_fm <- runif(n_fm, -1, 1); w_fm <- runif(n_fm, 0.2, 3)
d_fm <- data.frame(y = sin(2 * x_fm) + rnorm(n_fm, 0, 0.3), x = x_fm)
fit_fm <- glmboost(y ~ x, data = d_fm, weights = w_fm, control = boost_control(mstop = 40))
mask_fm <- cv_weights(n_fm, "kfold", 4, 1)
stopifnot(max(abs(cvrisk(fit_fm, folds = mask_fm)$risk_matrix -
                  cvrisk(fit_fm, folds = mask_fm * w_fm)$risk_matrix)) < 1e-10)

# A weighted gamboost cross-validates: each fold refits with the model's
# weights and scores its held-out rows with them, so a 0/1 mask and folds
# that carry the weights give the same curve.
fit_gw <- gamboost(y ~ bbs(x, df = 4), data = d_fm, weights = w_fm,
                   control = boost_control(mstop = 20))
stopifnot(max(abs(cvrisk(fit_gw, folds = mask_fm)$risk_matrix -
                  cvrisk(fit_gw, folds = mask_fm * w_fm)$risk_matrix)) < 1e-10)

cat("weight contract (R): PASS\n")
