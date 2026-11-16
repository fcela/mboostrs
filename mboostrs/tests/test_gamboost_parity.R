# Parity check: our gamboost (bols / bbs) vs R's mboost, for fitting and for
# prediction (in-sample, in-range newdata, out-of-range newdata).
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping gamboost parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost))); data(iris)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-32s maxdiff=%.2e\n", what, md))
}

d <- iris
ctrl  <- mboostrs::boost_control(mstop = 100, nu = 0.1)
rctrl <- mboost::boost_control(mstop = 100, nu = 0.1)

#@validates parity.r.learner.bbs
# ── bbs (single smooth) ──────────────────────────────────────────────
m1 <- mboostrs::gamboost(Sepal.Width ~ bbs(Sepal.Length, df = 4), data = d,
                         family = mboostrs::Gaussian(), control = ctrl)
r1 <- mboost::gamboost(Sepal.Width ~ bbs(Sepal.Length, df = 4), data = d,
                       control = rctrl)
chk(fitted(m1), fitted(r1), "bbs fitted")
chk(predict(m1), fitted(m1), "bbs in-sample predict vs fitted")

nd <- data.frame(Sepal.Length = seq(min(d$Sepal.Length), max(d$Sepal.Length),
                                    length.out = 50))
chk(predict(m1, newdata = nd), predict(r1, newdata = nd), "bbs predict in-range")

nd2 <- data.frame(Sepal.Length = c(3.5, 9.0))   # outside the training range
chk(predict(m1, newdata = nd2), suppressWarnings(predict(r1, newdata = nd2)),
    "bbs predict out-of-range", tol = 1e-5)

#@validates parity.r.learner.bols
#@validates r.learner.contract
# ── bbs + bols (mixed) ───────────────────────────────────────────────
m2 <- mboostrs::gamboost(Sepal.Width ~ bbs(Sepal.Length, df = 4) + bols(Petal.Length),
                         data = d, family = mboostrs::Gaussian(), control = ctrl)
r2 <- mboost::gamboost(Sepal.Width ~ bbs(Sepal.Length, df = 4) + bols(Petal.Length),
                       data = d, control = rctrl)
chk(fitted(m2), fitted(r2), "bbs+bols fitted")
nd3 <- d[1:20, c("Sepal.Length", "Petal.Length")]
chk(predict(m2, newdata = nd3), predict(r2, newdata = nd3), "bbs+bols predict newdata")

cat("gamboost R-binding parity: PASS\n")
