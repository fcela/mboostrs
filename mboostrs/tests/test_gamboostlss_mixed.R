library(mboostrs)

# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
set.seed(412)
n <- 64L
dat <- data.frame(x = seq(-1, 1, length.out = n), z = rnorm(n))
dat$y <- 0.8 * sin(2 * dat$x) + 0.35 * dat$z +
  rnorm(n, sd = exp(0.15 * dat$z))
forms <- list(
  mu = y ~ bbs(x, df = 4) + bols(z),
  sigma = y ~ bols(z) + bbs(x, df = 4)
)

#@validates parity.r.gamboostlss.mixed_bols_bbs
fit <- gamboostLSS(
  forms, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 10, nu = 0.1)
)
stopifnot(length(fit$params$mu$selected) == 10L)
stopifnot(length(fit$params$sigma$selected) == 10L)
stopifnot(identical(names(selected(fit)), c("mu", "sigma")))

full <- predict(fit, type = "link")
truncated <- predict(fit, type = "link", mstop = c(mu = 4L, sigma = 6L))
newdata <- transform(dat[1:7, ], x = x + 0.02, z = z - 0.05)
new_pred <- predict(fit, newdata = newdata, type = "response")
stopifnot(length(full$mu) == n, length(truncated$mu) == n)
stopifnot(length(new_pred$mu) == 7L, all(new_pred$sigma > 0))
stopifnot(max(abs(full$mu - fit$params$mu$fitted)) < 1e-12)
effects <- partial_effects(fit, parameter = "mu")
replayed_mu <- rep(fit$params$mu$offset, n)
for (effect in effects) replayed_mu <- replayed_mu + effect$effect
stopifnot(max(abs(replayed_mu - full$mu)) < 1e-12)

saved <- tempfile(fileext = ".rds")
saveRDS(fit, saved)
restored <- readRDS(saved)
unlink(saved)
stopifnot(max(abs(predict(restored, newdata = newdata, parameter = "mu") -
                      predict(fit, newdata = newdata, parameter = "mu"))) < 1e-12)

no_intercept <- gamboostLSS(
  y ~ bols(z, intercept = FALSE), data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1)
)
stopifnot(all(vapply(no_intercept$params$mu$coefs, length, integer(1)) == 1L))
stopifnot(length(predict(no_intercept, newdata = dat[1:4, ], parameter = "mu")) == 4L)

#@validates smoke.r.gamboostlss.bns
bns_fit <- gamboostLSS(
  y ~ bns(x, df = 7, center = TRUE) + bols(z),
  data = dat, families = GaussianLSS(), weights = seq(0.5, 1.5, length.out = n),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(bns_fit, parameter = "mu") -
                      predict(bns_fit, newdata = dat, parameter = "mu"))) < 1e-12)

random_dat <- transform(dat, group = factor(rep(letters[1:4], length.out = n)))
random_dat$y <- random_dat$y + c(-0.8, 0.2, 0.9, -0.1)[as.integer(random_dat$group)]
#@validates parity.r.gamboostlss.brandom
random_formula <- y ~ brandom(group, df = 3, center = TRUE) + bols(z)
random_fit <- gamboostLSS(
  random_formula, data = random_dat, families = GaussianLSS(),
  weights = seq(0.6, 1.4, length.out = n),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(random_fit, parameter = "mu") -
                      predict(random_fit, newdata = random_dat,
                              parameter = "mu"))) < 1e-12)
random_effects <- partial_effects(random_fit, parameter = "mu")
random_replay <- rep(random_fit$params$mu$offset, n)
for (effect in random_effects) random_replay <- random_replay + effect$effect
stopifnot(max(abs(random_replay - predict(random_fit, parameter = "mu",
                                         type = "link"))) < 1e-12)
unknown_dat <- random_dat[1:2, ]
unknown_dat$group <- factor(c("a", "e"), levels = c(letters[1:4], "e"))
unknown_error <- try(predict(random_fit, newdata = unknown_dat,
                             parameter = "mu"), silent = TRUE)
stopifnot(inherits(unknown_error, "try-error"),
          grepl("level(s) e the fit did not see", as.character(unknown_error), fixed = TRUE))

# brandom(group, by = z) is fitted (issue #164); test_lss_by_parity.R
# compares it with gamboostLSS.
random_by <- gamboostLSS(
  y ~ brandom(group, df = 3, by = z) + bols(z),
  data = random_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1)
)
stopifnot(all(is.finite(unlist(predict(random_by, type = "link")))))

K_regions <- matrix(c(
   1, -1,  0,  0,
  -1,  2, -1,  0,
   0, -1,  2, -1,
   0,  0, -1,  1
), nrow = 4L, byrow = TRUE)
#@validates smoke.r.gamboostlss.bmrf
mrf_formula <- y ~ bmrf(group, K = K_regions, df = 3, center = TRUE) + bols(z)
mrf_fit <- gamboostLSS(
  mrf_formula, data = random_dat, families = GaussianLSS(),
  weights = seq(0.6, 1.4, length.out = n),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(mrf_fit, parameter = "mu") -
                      predict(mrf_fit, newdata = random_dat,
                              parameter = "mu"))) < 1e-12)
mrf_effects <- partial_effects(mrf_fit, parameter = "mu")
mrf_replay <- rep(mrf_fit$params$mu$offset, n)
for (effect in mrf_effects) mrf_replay <- mrf_replay + effect$effect
stopifnot(max(abs(mrf_replay - predict(mrf_fit, parameter = "mu",
                                       type = "link"))) < 1e-12)
mrf_unknown_error <- try(predict(mrf_fit, newdata = unknown_dat,
                                 parameter = "mu"), silent = TRUE)
stopifnot(inherits(mrf_unknown_error, "try-error"),
          grepl("the fit did not see", as.character(mrf_unknown_error), fixed = TRUE))

spatial_dat <- transform(dat, x1 = x, x2 = seq(1, -1, length.out = n))
spatial_dat$y <- spatial_dat$y + sin(pi * spatial_dat$x1) * cos(pi * spatial_dat$x2)
#@validates smoke.r.gamboostlss.bspatial
spatial_formula <- y ~ bspatial(x1, x2, knots = 3, degree = 2,
                                differences = 1, df = 5, center = TRUE) + bols(z)
spatial_fit <- gamboostLSS(
  spatial_formula, data = spatial_dat, families = GaussianLSS(),
  weights = seq(0.7, 1.3, length.out = n),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(spatial_fit, parameter = "mu") -
                      predict(spatial_fit, newdata = spatial_dat,
                              parameter = "mu"))) < 2e-11)
spatial_effects <- partial_effects(spatial_fit, parameter = "mu")
stopifnot(identical(names(spatial_effects[["x1,x2"]]), c("x1", "x2", "effect")))
spatial_replay <- rep(spatial_fit$params$mu$offset, n)
for (effect in spatial_effects) spatial_replay <- spatial_replay + effect$effect
stopifnot(max(abs(spatial_replay - predict(spatial_fit, parameter = "mu",
                                           type = "link"))) < 2e-11)
spatial_new <- transform(spatial_dat[1:7, ], x1 = x1 + 0.01, x2 = x2 - 0.01)
stopifnot(length(predict(spatial_fit, newdata = spatial_new,
                         parameter = "mu")) == 7L)

mono_dat <- transform(dat, xmono = seq(0, 4, length.out = n))
mono_dat$y <- 1.2 + 1.8 * log(mono_dat$xmono + 0.5) + 0.2 * mono_dat$z +
  rnorm(n, sd = 0.3)
#@validates smoke.r.gamboostlss.bmono
mono_formula <- y ~ bmono(xmono, constraint = "increasing", knots = 8,
                          degree = 3, differences = 2, df = 4) + bols(z)
mono_fit <- gamboostLSS(
  mono_formula, data = mono_dat, families = GaussianLSS(),
  weights = seq(0.7, 1.3, length.out = n),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(mono_fit, parameter = "mu") -
                      predict(mono_fit, newdata = mono_dat,
                              parameter = "mu"))) < 2e-11)
mono_effect <- partial_effects(mono_fit, parameter = "mu", which = "xmono",
                               newdata = mono_dat)[["xmono"]]
stopifnot(all(diff(mono_effect$effect) >= -1e-9))

C_positive <- diag(3L + 2L + 1L)
#@validates smoke.r.gamboostlss.bconstrained
constrained_formula <- y ~ bconstrained(xmono, C = C_positive, knots = 3,
                                        degree = 2, differences = 1, df = 4) + bols(z)
constrained_fit <- gamboostLSS(
  constrained_formula, data = mono_dat, families = GaussianLSS(),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(constrained_fit, parameter = "mu") -
                      predict(constrained_fit, newdata = mono_dat,
                              parameter = "mu"))) < 2e-11)
constrained_effect <- partial_effects(
  constrained_fit, parameter = "mu", which = "xmono", newdata = mono_dat
)[["xmono"]]
stopifnot(all(constrained_effect$effect >= -2e-5))

tree_dat <- transform(dat, xtree = seq(-2, 2, length.out = n))
tree_dat$y <- 0.8 + 1.1 * (tree_dat$xtree > 0) + 0.2 * tree_dat$z +
  rnorm(n, sd = 0.3)
#@validates smoke.r.gamboostlss.btree
tree_formula <- y ~ btree(xtree, minbucket = 4, minsplit = 10, maxdepth = 2) + bols(z)
tree_fit <- gamboostLSS(
  tree_formula, data = tree_dat, families = GaussianLSS(),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(tree_fit, parameter = "mu") -
                      predict(tree_fit, newdata = tree_dat,
                              parameter = "mu"))) < 2e-12)
tree_effects <- partial_effects(tree_fit, parameter = "mu")
tree_replay <- rep(tree_fit$params$mu$offset, n)
for (effect in tree_effects) tree_replay <- tree_replay + effect$effect
stopifnot(max(abs(tree_replay - predict(tree_fit, parameter = "mu"))) < 2e-12)

X_custom <- cbind(dat$x, dat$x^2, sin(dat$x))
K_custom <- diag(c(0.1, 1.0, 1.0))
#@validates smoke.r.gamboostlss.buser
user_formula <- y ~ buser(custom, X = X_custom, K = K_custom,
                          df = 2.5, center = TRUE) + bols(z)
user_fit <- gamboostLSS(
  user_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(user_fit, parameter = "mu") -
                      predict(user_fit, newdata = dat, parameter = "mu",
                              buser_newdata = list(custom = X_custom)))) < 2e-12)
user_effects <- partial_effects(user_fit, parameter = "mu")
user_replay <- rep(user_fit$params$mu$offset, n)
for (effect in user_effects) user_replay <- user_replay + effect$effect
stopifnot(max(abs(user_replay - predict(user_fit, parameter = "mu"))) < 2e-12)
user_missing <- try(predict(user_fit, newdata = dat[1:5, ], parameter = "mu"),
                    silent = TRUE)
stopifnot(inherits(user_missing, "try-error"),
          grepl("requires buser_newdata", as.character(user_missing), fixed = TRUE))
user_new_x <- seq(-1, 1, length.out = 7L)
user_new_X <- cbind(user_new_x, user_new_x^2, sin(user_new_x))
user_new_data <- data.frame(z = rep(0.0, 7L))
user_prediction <- predict(
  user_fit, newdata = user_new_data, parameter = "mu",
  buser_newdata = list(custom = user_new_X)
)
stopifnot(length(user_prediction) == 7L)
user_restored <- unserialize(serialize(user_fit, NULL))
stopifnot(max(abs(user_prediction - predict(
  user_restored, newdata = user_new_data, parameter = "mu",
  buser_newdata = list(custom = user_new_X)
))) == 0.0)

#@validates smoke.r.gamboostlss.bkernel
kernel_formula <- y ~ bkernel(x, bandwidth = 0.45, df = 4,
                              kernel = gaussian, center = TRUE) + bols(z)
kernel_fit <- gamboostLSS(
  kernel_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(kernel_fit, parameter = "mu") -
                      predict(kernel_fit, newdata = dat,
                              parameter = "mu"))) < 3e-11)
kernel_effects <- partial_effects(kernel_fit, parameter = "mu")
kernel_replay <- rep(kernel_fit$params$mu$offset, n)
for (effect in kernel_effects) kernel_replay <- kernel_replay + effect$effect
stopifnot(max(abs(kernel_replay - predict(kernel_fit, parameter = "mu"))) < 3e-11)

#@validates smoke.r.gamboostlss.bkernel_nystrom
nyst_formula <- y ~ bkernel(x, bandwidth = 0.45, df = 4,
                            landmarks = 15L, center = TRUE) + bols(z)
nyst_fit <- gamboostLSS(
  nyst_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 6, nu = 0.1)
)
stopifnot(max(abs(predict(nyst_fit, parameter = "mu") -
                    predict(nyst_fit, newdata = dat,
                            parameter = "mu"))) < 3e-11)
nyst_effects <- partial_effects(nyst_fit, parameter = "mu")
nyst_replay <- rep(nyst_fit$params$mu$offset, n)
for (effect in nyst_effects) nyst_replay <- nyst_replay + effect$effect
stopifnot(max(abs(nyst_replay - predict(nyst_fit, parameter = "mu"))) < 3e-11)
stopifnot(length(nyst_fit$params$mu$selected) == 6L,
          all(is.finite(nyst_fit$params$mu$selected)))

# a zero landmark count is rejected before the fit starts
nyst_bad <- try(gamboostLSS(
  y ~ bkernel(x, bandwidth = 0.45, df = 4, landmarks = 0L),
  data = dat, families = GaussianLSS(), control = boost_control(mstop = 3)
), silent = TRUE)
stopifnot(inherits(nyst_bad, "try-error"))

noncyclic <- gamboostLSS(
  forms, data = dat, families = GaussianLSS(), method = "noncyclic",
  control = boost_control(mstop = 9, nu = 0.1)
)
noncyclic_short <- gamboostLSS(
  forms, data = dat, families = GaussianLSS(), method = "noncyclic",
  control = boost_control(mstop = 4, nu = 0.1)
)
stopifnot(max(abs(predict(noncyclic, mstop = 4, parameter = "mu") -
                      predict(noncyclic_short, parameter = "mu"))) < 1e-12)
stopifnot(max(abs(predict(noncyclic, mstop = 4, parameter = "sigma") -
                      predict(noncyclic_short, parameter = "sigma"))) < 1e-12)

updated <- update(fit, mstop = c(mu = 5L, sigma = 7L))
stopifnot(length(updated$params$mu$selected) == 5L)
stopifnot(length(updated$params$sigma$selected) == 7L)

folds <- matrix(0, n, 4L)
for (b in seq_len(ncol(folds))) {
  folds[-seq(b, n, by = ncol(folds)), b] <- 1
}
bns_cv <- cvrisk_gamboostLSS(
  y ~ bns(x, df = 7, center = TRUE) + bols(z),
  data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(bns_cv)))
bns_stab <- stabsel_gamboostLSS(
  y ~ bns(x, df = 7, center = TRUE) + bols(z),
  data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(bns_stab$max),
                    c("x.mu", "z.mu", "x.sigma", "z.sigma")))
random_cv <- cvrisk_gamboostLSS(
  random_formula, data = random_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(random_cv)))
random_stab <- stabsel_gamboostLSS(
  random_formula, data = random_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(random_stab$max),
                    c("group.mu", "z.mu", "group.sigma", "z.sigma")))
mrf_cv <- cvrisk_gamboostLSS(
  mrf_formula, data = random_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(mrf_cv)))
mrf_stab <- stabsel_gamboostLSS(
  mrf_formula, data = random_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(mrf_stab$max),
                    c("group.mu", "z.mu", "group.sigma", "z.sigma")))
spatial_cv <- cvrisk_gamboostLSS(
  spatial_formula, data = spatial_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(spatial_cv)))
spatial_stab <- stabsel_gamboostLSS(
  spatial_formula, data = spatial_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(spatial_stab$max),
                    c("x1,x2.mu", "z.mu", "x1,x2.sigma", "z.sigma")))
mono_cv <- cvrisk_gamboostLSS(
  mono_formula, data = mono_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(mono_cv)))
mono_stab <- stabsel_gamboostLSS(
  mono_formula, data = mono_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(mono_stab$max),
                    c("xmono.mu", "z.mu", "xmono.sigma", "z.sigma")))
constrained_cv <- cvrisk_gamboostLSS(
  constrained_formula, data = mono_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(constrained_cv)))
constrained_stab <- stabsel_gamboostLSS(
  constrained_formula, data = mono_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(constrained_stab$max),
                    c("xmono.mu", "z.mu", "xmono.sigma", "z.sigma")))
tree_cv <- cvrisk_gamboostLSS(
  tree_formula, data = tree_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(tree_cv)))
tree_stab <- stabsel_gamboostLSS(
  tree_formula, data = tree_dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(tree_stab$max),
                    c("xtree.mu", "z.mu", "xtree.sigma", "z.sigma")))
user_cv <- cvrisk_gamboostLSS(
  user_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(user_cv)))
user_stab <- stabsel_gamboostLSS(
  user_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(user_stab$max),
                    c("custom.mu", "z.mu", "custom.sigma", "z.sigma")))
kernel_cv <- cvrisk_gamboostLSS(
  kernel_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  grid = rbind(c(mu = 2L, sigma = 2L), c(mu = 3L, sigma = 3L))
)
stopifnot(all(is.finite(kernel_cv)))
kernel_stab <- stabsel_gamboostLSS(
  kernel_formula, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 3, nu = 0.1), folds = folds,
  q = 1L, cutoff = 0.75
)
stopifnot(identical(names(kernel_stab$max),
                    c("x.mu", "z.mu", "x.sigma", "z.sigma")))
grid <- rbind(c(mu = 3L, sigma = 3L), c(mu = 5L, sigma = 5L))
cv <- cvrisk_gamboostLSS(
  forms, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 5, nu = 0.1),
  folds = folds, grid = grid
)
stopifnot(identical(dim(cv), c(4L, 2L)), all(is.finite(cv)))

stab <- stabsel_gamboostLSS(
  forms, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 5, nu = 0.1),
  folds = folds, q = 1L, cutoff = 0.75
)
expected <- c("x.mu", "z.mu", "z.sigma", "x.sigma")
stopifnot(length(stab$max) == length(expected))
stopifnot(identical(names(stab$max), expected))

if (requireNamespace("gamboostLSS", quietly = TRUE) &&
    requireNamespace("mboost", quietly = TRUE)) {
  suppressPackageStartupMessages(library(gamboostLSS))
  reference <- gamboostLSS::gamboostLSS(
    forms, data = dat, families = gamboostLSS::GaussianLSS(),
    control = mboost::boost_control(mstop = 10, nu = 0.1)
  )
  reference_link <- predict(reference, type = "link")
  stopifnot(max(abs(full$mu - as.double(reference_link$mu))) < 2e-6)
  stopifnot(max(abs(full$sigma - as.double(reference_link$sigma))) < 2e-6)

  random_reference <- gamboostLSS::gamboostLSS(
    y ~ brandom(group, df = 3) + bols(z), data = random_dat,
    families = gamboostLSS::GaussianLSS(),
    control = mboost::boost_control(mstop = 6, nu = 0.1)
  )
  random_local <- mboostrs::gamboostLSS(
    y ~ brandom(group, df = 3) + bols(z), data = random_dat,
    families = mboostrs::GaussianLSS(),
    control = mboostrs::boost_control(mstop = 6, nu = 0.1)
  )
  random_reference_link <- predict(random_reference, type = "link")
  stopifnot(max(abs(predict(random_local, parameter = "mu", type = "link") -
                        as.double(random_reference_link$mu))) < 2e-6)
  stopifnot(max(abs(predict(random_local, parameter = "sigma", type = "link") -
                        as.double(random_reference_link$sigma))) < 2e-6)
}
