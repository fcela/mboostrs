# stabsel() ran its subsample fits in an R loop and computed the selection
# frequencies in R (`.lss_stabsel_frequencies`, also used by the LSS
# stabsel functions). Both now run in the core (issue #848). The expected
# values below are the results of the R implementation before the port, on
# the same fixtures; they must agree to 1e-8 relative.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
close <- function(got, want, label) {
  got <- unname(as.numeric(got)); want <- as.numeric(want)
  ok(length(got) == length(want), paste(label, ": length", length(got), "vs", length(want)))
  err <- max(abs(got - want) / pmax(abs(want), 1e-300) * (want != 0) + abs(got) * (want == 0))
  ok(err <= 1e-8, sprintf("%s: relative difference %.3g from the earlier R value", label, err))
}

set.seed(848)
n <- 120
X <- matrix(rnorm(n * 6), n, 6); colnames(X) <- paste0("x", 1:6)
d <- as.data.frame(X)
d$y <- 1.5 * d$x1 - d$x3 + 0.5 * d$x5 + rnorm(n)
d$yb <- as.integer(d$x1 + rnorm(n) > 0)
w <- runif(n, 0.5, 2)
f <- y ~ x1 + x2 + x3 + x4 + x5 + x6

# Gaussian, complementary pairs.
s1 <- stabsel(f, data = d, control = boost_control(mstop = 40), B = 10, q = 2, cutoff = 0.9)
close(s1$max, c(0, 1, 0, 1, 0, 0, 0), "s1 max")
close(colSums(s1$phat), c(1, 1, 1, 1, 1.2, 1.3, 1.45, 1.55, 1.6, 1.8, 1.95, rep(2, 29)),
      "s1 phat column sums")
ok(identical(s1$selected, c("x1", "x3")), "s1 selected")
ok(identical(dim(s1$phat), c(7L, 40L)), "s1 phat dimensions")

# Fractional weights: each fold carries them, rescaled as glmboost rescales.
s2 <- stabsel(f, data = d, control = boost_control(mstop = 40), B = 10, q = 2,
              cutoff = 0.9, weights = w)
close(s2$max, c(0, 1, 0, 1, 0, 0, 0), "s2 max")
close(colSums(s2$phat), c(1, 1, 1, 1, 1.15, 1.25, 1.4, 1.45, 1.65, 1.8, 1.95, 1.95,
                          rep(2, 28)), "s2 phat column sums")

# Binomial, independent half-samples.
s3 <- suppressWarnings(stabsel(yb ~ x1 + x2 + x3 + x4, data = d, family = Binomial(),
                               control = boost_control(mstop = 30), B = 8, q = 2,
                               cutoff = 0.9, sampling_type = "MB"))
close(s3$max, c(0.25, 1, 0.25, 0.5, 0), "s3 max")
close(colSums(s3$phat), c(rep(1, 12), 1.25, 1.375, 1.5, 1.625, 1.75, 1.75, 1.875,
                          1.875, 1.875, rep(2, 9)), "s3 phat column sums")

# No intercept: the learners are the columns alone.
s4 <- stabsel(y ~ x1 + x2 + x3 - 1, data = d, control = boost_control(mstop = 20),
              B = 6, q = 2, cutoff = 0.9)
close(s4$max, c(1, 0, 1), "s4 max")
ok(identical(names(s4$max), c("x1", "x2", "x3")), "s4 names")

# The frequency rule on hand-written paths; the third path is a failed fit.
seqs <- list(c(3L, 1L, 3L, 2L), c(2L, 2L, 2L, 2L), integer(0), c(1L, 4L, 2L, 3L))
close(mboostrs:::.lss_stabsel_frequencies(seqs, 4, 4, 2),
      c(0.25, 0.25, 0.25, 0, 0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25,
        0.5, 0.25, 0.25, 0.25), "LSS frequencies")

# glmboostLSS stability selection, which goes through the same function.
folds <- cv_weights(n, "subsampling", b = 6, prob = 0.5, seed = 3L)
s5 <- stabsel_glmboostLSS(y ~ x1 + x2 + x3, data = d, families = GaussianLSS(),
                          control = boost_control(mstop = 30), folds = folds, q = 2,
                          cutoff = 0.9)
close(s5$max, c(0, 1, 0, 0, 1, 0, 0, 0), "s5 max")
close(colSums(s5$phat), c(rep(1, 11), 7 / 6, 7 / 6, rep(2, 17)), "s5 phat column sums")

cat("test_stabsel_core_frequencies.R: OK\n")
