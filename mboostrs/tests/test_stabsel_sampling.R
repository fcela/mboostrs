# Complementary-pairs sampling in the R frontend.
#
# Under sampling_type = "SS" a run fits B half-samples and each one's exact
# complement: column b and column B + b of the fold matrix sum elementwise to
# the model's full weights, which is what the Shah-Samworth bound is derived
# on. `B` counts those initial subsamples and is what the bound is computed
# on; `n_fits` counts models actually fitted, so it is 2 * B under "SS" and B
# under "MB".
#
# `folds` supplied by a caller is always the *initial* subsamples: B columns.
# Under "SS" the complements are appended here, matching stabs:::check_folds
# (`folds <- cbind(folds, weights - folds)`). Pairing is never inferred from
# an even column count.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

# ─── the fold constructor ──────────────────────────────────────────

for (n in c(20L, 21L, 200L)) {
  B <- 4L
  f <- stabsel_folds(n, B = B, sampling_type = "SS")
  ok(identical(dim(f), c(n, 2L * B)), sprintf("SS folds should be n x 2B (n=%d)", n))
  for (b in seq_len(B)) {
    ok(isTRUE(all.equal(f[, b] + f[, B + b], rep(1.0, n))),
       sprintf("SS pair %d does not sum to the full weights (n=%d)", b, n))
  }
}

# Pairs sum to the model's weights, not to one.
n <- 40L
w <- seq(0.5, 3.0, length.out = n)
f <- stabsel_folds(n, B = 3L, sampling_type = "SS", weights = w)
for (b in 1:3) {
  ok(isTRUE(all.equal(f[, b] + f[, 3L + b], w)),
     sprintf("weighted SS pair %d does not sum to the weights", b))
}

# MB draws are independent: no column completes another.
n <- 40L
f <- stabsel_folds(n, B = 5L, sampling_type = "MB")
ok(identical(dim(f), c(n, 5L)), "MB folds should be n x B")
paired <- FALSE
for (i in 1:4) for (j in (i + 1L):5L) {
  if (isTRUE(all.equal(f[, i] + f[, j], rep(1.0, n)))) paired <- TRUE
}
ok(!paired, "MB folds must not contain complementary pairs")

# SS is MB plus complements, from the same draw and seed.
ss <- stabsel_folds(30L, B = 4L, sampling_type = "SS", seed = 11L)
mbf <- stabsel_folds(30L, B = 4L, sampling_type = "MB", seed = 11L)
ok(isTRUE(all.equal(ss[, 1:4], mbf)), "SS and MB should share their first B columns")

# prob != 0.5 has no complement, so "SS" rejects it and "MB" accepts it.
ok(inherits(try(stabsel_folds(20L, 2L, "SS", prob = 0.7), silent = TRUE), "try-error"),
   "SS with prob != 0.5 should be rejected")
ok(identical(dim(stabsel_folds(20L, 2L, "MB", prob = 0.7)), c(20L, 2L)),
   "MB should accept prob != 0.5")

ok(inherits(try(stabsel_folds(20L, 2L, "pairs"), silent = TRUE), "try-error"),
   "an unknown sampling_type should be rejected")

# ─── glmboost stabsel ──────────────────────────────────────────────

set.seed(42); n <- 120L; p <- 8L
x <- matrix(rnorm(n * p), n, p); colnames(x) <- paste0("x", seq_len(p))
d <- as.data.frame(x); d$y <- 2 * d$x1 - 1.5 * d$x3 + rnorm(n)

s_ss <- stabsel(y ~ ., data = d, family = Gaussian(),
                control = boost_control(mstop = 60L), B = 5L, q = 3L)
ok(s_ss$B == 5L && s_ss$n_fits == 10L,
   sprintf("SS should report B=5 and n_fits=10, got B=%s n_fits=%s",
           s_ss$B, s_ss$n_fits))
ok(identical(s_ss$sampling_type, "SS"), "sampling_type should be recorded")

s_mb <- stabsel(y ~ ., data = d, family = Gaussian(),
                control = boost_control(mstop = 60L), B = 5L, q = 3L,
                sampling_type = "MB", assumption = "none")
ok(s_mb$B == 5L && s_mb$n_fits == 5L,
   sprintf("MB should report B=5 and n_fits=5, got B=%s n_fits=%s",
           s_mb$B, s_mb$n_fits))

# The bound is computed on B, not on the fit count.
on_pairs <- stabsel_params(p = s_ss$p, b = 5L, cutoff = 0.7, q = 3L,
                           sampling_type = "SS")
on_fits <- stabsel_params(p = s_ss$p, b = 10L, cutoff = 0.7, q = 3L,
                          sampling_type = "SS")
ok(isTRUE(all.equal(s_ss$PFER, on_pairs$pfer)),
   "the reported PFER should be the one computed on B pairs")
ok(!isTRUE(all.equal(on_pairs$pfer, on_fits$pfer)),
   "the B and 2B bounds must differ, or this test proves nothing")

# Supplied folds are the initial subsamples; SS appends complements.
half <- stabsel_folds(n, B = 4L, sampling_type = "MB", seed = 8L)
s_sup <- stabsel(y ~ ., data = d, family = Gaussian(),
                 control = boost_control(mstop = 60L), q = 3L, folds = half)
ok(s_sup$B == 4L && s_sup$n_fits == 8L,
   sprintf("supplied folds should give B=4, n_fits=8, got B=%s n_fits=%s",
           s_sup$B, s_sup$n_fits))

# SS equals MB run over the pair-completed matrix.
paired_folds <- cbind(half, 1 - half)
s_as_mb <- stabsel(y ~ ., data = d, family = Gaussian(),
                   control = boost_control(mstop = 60L), q = 3L,
                   folds = paired_folds, sampling_type = "MB",
                   assumption = "none")
ok(isTRUE(all.equal(unname(s_sup$max), unname(s_as_mb$max))),
   "SS on B folds must equal MB on the same folds plus their complements")
ok(setequal(s_sup$selected, s_as_mb$selected),
   "SS and the equivalent MB run must select the same learners")

# ─── LSS stabsel ───────────────────────────────────────────────────

set.seed(1); n <- 100L; p <- 6L
X <- matrix(rnorm(n * p), n, p); colnames(X) <- paste0("x", seq_len(p))
dl <- data.frame(y = rnorm(n, 1 + 0.8 * X[, 1], exp(0.3 * X[, 2])), X)
half <- stabsel_folds(n, B = 4L, sampling_type = "MB", seed = 5L)

lss_ss <- stabsel_glmboostLSS(y ~ ., data = dl, families = GaussianLSS(),
                              control = boost_control(mstop = 40L, nu = 0.1),
                              folds = half, q = 4L, PFER = 2, sampling_type = "SS")
ok(lss_ss$B == 4L && lss_ss$n_fits == 8L,
   sprintf("LSS SS should report B=4 and n_fits=8, got B=%s n_fits=%s",
           lss_ss$B, lss_ss$n_fits))

lss_as_mb <- stabsel_glmboostLSS(y ~ ., data = dl, families = GaussianLSS(),
                                 control = boost_control(mstop = 40L, nu = 0.1),
                                 folds = cbind(half, 1 - half), q = 4L, PFER = 2,
                                 sampling_type = "MB")
ok(isTRUE(all.equal(lss_ss$phat, lss_as_mb$phat)),
   "LSS SS must equal MB over the same folds plus their complements")
ok(lss_as_mb$B == 8L && lss_as_mb$n_fits == 8L,
   "LSS MB should report B and n_fits equal to the column count")

gam_ss <- stabsel_gamboostLSS(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = dl,
                              families = GaussianLSS(),
                              control = boost_control(mstop = 30L),
                              folds = stabsel_folds(n, B = 3L, sampling_type = "MB"),
                              q = 2L, PFER = 1, sampling_type = "SS")
ok(gam_ss$B == 3L && gam_ss$n_fits == 6L,
   sprintf("gamboostLSS SS should report B=3 and n_fits=6, got B=%s n_fits=%s",
           gam_ss$B, gam_ss$n_fits))

cat("stabsel complementary-pairs sampling (R): PASS\n")

# stabsel() runs its subsample fits through glmboost(), so a base-learner
# formula used to fail inside term evaluation with "could not find function
# bols", naming neither the limitation nor the alternative.
learner_error <- tryCatch({
  stabsel(y ~ bbs(x1, df = 4), data = dl, family = Gaussian(),
          control = boost_control(mstop = 10L), q = 2L, PFER = 1)
  NULL
}, error = identity)
ok(inherits(learner_error, "error") &&
     grepl("glmboost", conditionMessage(learner_error), fixed = TRUE) &&
     grepl("y ~ x1 \\+ x2", conditionMessage(learner_error)),
   sprintf("stabsel must refuse a base-learner formula by name, got: %s",
           if (inherits(learner_error, "error")) conditionMessage(learner_error) else "no error"))

cat("stabsel base-learner refusal (R): PASS\n")
