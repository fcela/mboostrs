# Numeric by= on bmono, bns, bspatial, brandom, bmrf and btree in R gamboost
# (issue #164), against mboost (all but bns) with new rows taken from the middle of the data and
# the modifier changed. bns, which mboost withdrew, is checked against Python
# in tests/test_r_by_other_learners.py.
suppressMessages(library(mboostrs))
if (!requireNamespace("mboost", quietly = TRUE)) {
  cat("test_by_other_learners.R: skipped (mboost not installed)\n"); quit(save = "no")
}
suppressMessages(library(mboost))
set.seed(4); n <- 300
d <- data.frame(x = runif(n), w = runif(n), z = rnorm(n))
d$y <- sqrt(d$x) * d$z + sin(3 * d$w) + rnorm(n, sd = 0.2)
nd <- d[c(100:110, 280:290), ]; nd$z <- nd$z * 2
cases <- list(list(f = y ~ bmono(x, by = z) + bbs(w), tol = 1e-10),
              # bspatial without by= is within 1e-6 of mboost as well
              # (test_factor_bols_bspatial_parity.R).
              list(f = y ~ bspatial(x, w, by = z, df = 6) + bols(z), tol = 1e-6))
for (case in cases) {
  ours <- mboostrs::gamboost(case$f, data = d, control = mboostrs::boost_control(mstop = 100))
  up <- mboost::gamboost(case$f, data = d, control = mboost::boost_control(mstop = 100))
  gap <- max(abs(fitted(up) - ours$fitted), abs(predict(up, newdata = nd) - predict(ours, newdata = nd)))
  if (gap > case$tol) stop(deparse(case$f), ": gap ", gap)
}
# brandom(g, by = z): the indicator design scaled by z, df on the scaled design.
d$g <- factor(sample(letters[1:6], n, TRUE))
d$y2 <- as.numeric(d$g) * 0.3 * d$z + sin(3 * d$x) + rnorm(n, sd = 0.2)
nd <- d[c(100:110, 280:290), ]; nd$z <- nd$z * 2
for (f in list(y2 ~ brandom(g, by = z) + bbs(x), y2 ~ brandom(g, by = z, df = 2) + bols(x))) {
  ours <- mboostrs::gamboost(f, data = d, control = mboostrs::boost_control(mstop = 100))
  up <- mboost::gamboost(f, data = d, control = mboost::boost_control(mstop = 100))
  gap <- max(abs(fitted(up) - ours$fitted), abs(predict(up, newdata = nd) - predict(ours, newdata = nd)))
  if (gap > 1e-10) stop(deparse(f), ": gap ", gap)
}
# mboost's bmrf needs the BayesX package, and BayesX needs foreign; both are
# Suggests, and under R CMD check the subprocess may not find them.
if (requireNamespace("BayesX", quietly = TRUE) && requireNamespace("foreign", quietly = TRUE)) {
  # bmrf(r, bnd = K, by = z): the region indicators scaled by z, df on the
  # scaled design. mboost needs K's dimnames to name the regions.
  L <- 8
  K <- matrix(0, L, L); for (i in 1:(L - 1)) K[i, i + 1] <- K[i + 1, i] <- -1
  diag(K) <- -rowSums(K); dimnames(K) <- list(as.character(1:L), as.character(1:L))
  d$r <- factor(sample(1:L, n, TRUE), levels = 1:L)
  d$y3 <- (as.integer(d$r) / L) * d$z + sin(3 * d$w) + rnorm(n, sd = 0.3)
  nd <- d[c(100:110, 280:290), ]; nd$z <- nd$z * 2
  for (f in list(y3 ~ bmrf(r, bnd = K, by = z), y3 ~ bmrf(r, bnd = K, by = z, df = 3) + bbs(w))) {
    ours <- mboostrs::gamboost(f, data = d, control = mboostrs::boost_control(mstop = 80))
    up <- mboost::gamboost(f, data = d, control = mboost::boost_control(mstop = 80))
    gap <- max(abs(fitted(up) - ours$fitted), abs(predict(up) - predict(ours)),
               abs(predict(up, newdata = nd) - predict(ours, newdata = nd)))
    if (gap > 1e-10) stop(deparse(f), ": gap ", gap)
  }
} else {
  cat("test_by_other_learners.R: bmrf block skipped (BayesX or foreign not installed)\n")
}

# btree(x, by = b): b is 0/1 or a two-level factor; the tree is grown on the
# rows where b is 1 and its prediction is multiplied by b. New rows flip b.
d$b <- rbinom(n, 1, 0.5)
d$fb <- factor(ifelse(d$b == 1, "treated", "control"))
d$y4 <- sqrt(d$x) * d$b + sin(3 * d$w) + rnorm(n, sd = 0.3)
nd <- d[c(100:110, 280:290), ]; nd$b <- 1 - nd$b
nd$fb <- factor(ifelse(nd$b == 1, "treated", "control"))
tc <- partykit::ctree_control(maxdepth = 2, minbucket = 4, minsplit = 10,
                              mincriterion = 0, saveinfo = FALSE)
for (case in list(list(ours = y4 ~ btree(x, by = b) + bbs(w), up = y4 ~ btree(x, by = b) + bbs(w)),
                  list(ours = y4 ~ btree(x, w, by = fb, maxdepth = 2),
                       up = y4 ~ btree(x, w, by = fb, tree_controls = tc)))) {
  ours <- mboostrs::gamboost(case$ours, data = d, control = mboostrs::boost_control(mstop = 60))
  up <- mboost::gamboost(case$up, data = d, control = mboost::boost_control(mstop = 60))
  gap <- max(abs(fitted(up) - ours$fitted), abs(predict(up) - predict(ours)),
             abs(predict(up, newdata = nd) - predict(ours, newdata = nd)))
  if (gap > 1e-10) stop(deparse(case$ours), ": gap ", gap)
}
cat("test_by_other_learners.R: OK\n")
