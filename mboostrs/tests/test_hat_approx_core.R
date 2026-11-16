# hatvalues(approximate = TRUE) for gamboost added nu times each selected
# learner's leverage, but only for single-column learners: a spline, a factor
# block or a brandom step added nothing. The approximation is now computed by
# the core over every learner with a hat matrix (issue #848).
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
close <- function(got, want, label, tol = 1e-8) {
  got <- unname(as.numeric(got)); want <- as.numeric(want)
  ok(length(got) == length(want), paste(label, ": length", length(got), "vs", length(want)))
  err <- max(abs(got - want) / pmax(abs(want), 1e-12))
  ok(err <= tol, sprintf("%s: relative difference %.3g", label, err))
}

set.seed(849)
n2 <- 150
d2 <- data.frame(x1 = runif(n2), x2 = rnorm(n2),
                 g = sample(1:5, n2, replace = TRUE, prob = c(.3, .25, .2, .15, .1)))
geff <- c(-1, 0.5, 1, -0.3, 0.2)
d2$y <- sin(2 * pi * d2$x1) + 0.5 * d2$x2 + geff[d2$g] + rnorm(n2, sd = 0.3)

# Single-column learners only: the values of the R implementation before the
# port (1e-8 relative).
m5 <- gamboost(y ~ bols(x2) + bols(x1), data = d2, control = boost_control(mstop = 25))
h5 <- hatvalues(m5, approximate = TRUE)
close(h5[1:8], c(0.0166365738809202, 0.0264518008590885, 0.0164791570144911,
                 0.0083832847172612, 0.0271057889417922, 0.02456313387063,
                 0.0376679336708365, 0.00742547425039856), "bols-only approximation")
close(sum(h5), 3.5, "bols-only approximation sum")

# One spline step: the exact hat matrix is nu * H, so the approximation is
# the exact diagonal plus the offset's 1/n.
m7 <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d2,
               control = boost_control(mstop = 1))
ok(m7$selected[1] == 1L, "fixture: the first step should select the spline")
exact <- mboostrs:::.core_hat(m7)
h7 <- hatvalues(m7, approximate = TRUE)
close(h7 - 1 / n2, exact$diag, "one-step approximation vs exact diagonal", tol = 1e-10)
close(sum(h7) - 1, exact$trace[1], "one-step approximation vs exact trace", tol = 1e-10)

# Many steps with a small nu: the first-order sum is at or above the exact
# trace, and the relative gap is the size of the dropped second-order terms,
# below nu * mstop. The spline and brandom steps used to count for nothing.
nu8 <- 0.01; ms8 <- 30L
m8 <- gamboost(y ~ bbs(x1, df = 4) + bols(x2) + brandom(g, df = 3), data = d2,
               control = boost_control(mstop = ms8, nu = nu8))
ok(all(c(1L, 3L) %in% m8$selected), "fixture: the spline and brandom should be selected")
tr_exact <- tail(mboostrs:::.core_hat(m8)$trace, 1)
tr_approx <- sum(hatvalues(m8, approximate = TRUE)) - 1
ok(tr_approx >= tr_exact - 1e-12, "approximate trace below the exact trace")
ok((tr_approx - tr_exact) / tr_exact < nu8 * ms8,
   sprintf("approximate trace %.4f is far from the exact %.4f", tr_approx, tr_exact))

cat("test_hat_approx_core.R: OK\n")
