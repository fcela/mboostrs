# risk() of an LSS fit matches gamboostLSS's risk(merge = FALSE / TRUE) for
# cyclic and non-cyclic fits. The non-cyclic case also checks the update
# sequence: gamboostLSS scores a candidate parameter with the gradient its
# model last computed, and the fit follows that (1e-3 apart before).
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping LSS risk parity check")
  quit(save = "no")
}
suppressMessages(library(gamboostLSS))
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

for (seed in c(5, 11)) {
  set.seed(seed); n <- 300
  d <- data.frame(x = runif(n, -1, 1), z = rnorm(n))
  d$y <- rnorm(n, mean = sin(2 * d$x) + 0.5 * d$z, sd = exp(0.2 + 0.4 * d$x))
  for (kind in c("glm", "gam")) for (method in c("cyclic", "noncyclic")) {
    f <- if (kind == "glm") y ~ x + z else y ~ bbs(x) + bols(z)
    up_fit <- if (kind == "glm") gamboostLSS::glmboostLSS else gamboostLSS::gamboostLSS
    our_fit <- if (kind == "glm") mboostrs::glmboostLSS else mboostrs::gamboostLSS
    up <- up_fit(f, data = d, families = gamboostLSS::GaussianLSS(),
                 control = mboost::boost_control(mstop = 150), method = method)
    ours <- our_fit(f, data = d, families = mboostrs::GaussianLSS(),
                    control = mboostrs::boost_control(mstop = 150), method = method)
    a <- risk(ours); b <- risk(up)
    d1 <- max(abs(unlist(a) - unlist(b)))
    am <- risk(ours, merge = TRUE); bm <- risk(up, merge = TRUE)
    d2 <- max(abs(unclass(am) - unclass(bm)))
    cat(sprintf("seed %d %s %-9s per-parameter %.1e merged %.1e\n", seed, kind, method, d1, d2))
    ok(d1 < 1e-9 && d2 < 1e-9, paste(kind, method, "risk differs from gamboostLSS"))
    ok(identical(names(am), names(bm)), paste(kind, method, "merged names differ"))
  }
}
cat("LSS risk parity (R): PASS\n")
