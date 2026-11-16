# Two learners on one column stay apart in coef, varimp and partial_effects.
# coef() accumulated by name, adding bbs(x)'s steps into bols(x)'s entry;
# partial_effects() stored both under "x".
suppressMessages(library(mboostrs))
set.seed(0); n <- 200
d <- data.frame(x = runif(n, -2, 2), z = rnorm(n))
d$y <- sin(2 * d$x) + 2 * d$x + 0.5 * d$z + rnorm(n, sd = 0.3)
m <- gamboost(y ~ bols(x) + bbs(x, df = 1) + bols(z), data = d,
              control = boost_control(mstop = 300))
stopifnot(setequal(unique(m$selected), 1:3))

cf <- coef(m)
stopifnot(identical(names(cf), c("x (bols)", "x (bbs)", "z")),
          # bols(x) reports [(Intercept), x], as mboost does.
          length(cf[["x (bols)"]]) == 2L,
          length(cf[["x (bbs)"]]) > 1)
# Each entry is nu times the sum of that learner's own steps.
bbs_steps <- Reduce(`+`, m$coefs[m$selected == 2])
stopifnot(max(abs(cf[["x (bbs)"]] - m$nu * bbs_steps)) < 1e-12)
stopifnot(identical(names(coef(m, which = "x")), c("x (bols)", "x (bbs)")),
          identical(names(coef(m, which = "x (bbs)")), "x (bbs)"))

vi <- varimp(m)
stopifnot(identical(names(vi$reduction), c("x (bols)", "x (bbs)", "z")),
          abs(sum(vi$selfreq) - 1) < 1e-12)

stopifnot(identical(names(partial_effects(m)), c("x (bols)", "x (bbs)", "z")))

# Unshared names keep their keys.
m2 <- gamboost(y ~ bbs(x) + bols(z), data = d, control = boost_control(mstop = 50))
stopifnot(identical(names(coef(m2)), c("x", "z")))
cat("test_shared_column_learners.R: OK\n")
