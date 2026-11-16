# `bols` takes one engine form per formula, and it matches mboost.
#
# `bols(x, intercept = TRUE)` has two engine forms. The *centered* form fits
# one weight-centered column and takes its constant from the model offset; the
# *block* form fits `[1, x]` and carries its own. Alone the two are
# equivalent, because a constant is available either way. Two things break
# that, and both were getting it wrong.
#
# A penalty breaks it: R shrinks the learner's own intercept along with its
# slope, and a constant supplied by the offset is not shrinkable. This
# frontend sent the centered form for a penalized `bols` regardless.
#
# Mixing the two forms in one formula breaks it: the constant is then
# represented twice and the two representations compete along the path. Both
# frontends had this one.
#
# The reference is `mboost` itself, so these numbers cannot go stale against a
# hand-copied value.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping bols form parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost))

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(11)
n <- 80
d <- data.frame(x = rnorm(n), z = rnorm(n))
d$y <- 2 * d$x - d$z + rnorm(n)

compare <- function(formula, label, tolerance = 1e-12) {
  ours <- mboostrs::gamboost(formula, data = d,
                             control = mboostrs::boost_control(mstop = 30, nu = 0.1))
  theirs <- mboost::gamboost(formula, data = d,
                             control = mboost::boost_control(mstop = 30, nu = 0.1))
  difference <- max(abs(fitted(ours) - fitted(theirs)))
  ok(difference < tolerance,
     sprintf("%s: fitted values differ from mboost by %.3e (tolerance %.0e)",
             label, difference, tolerance))
  cat(sprintf("  %-38s maxdiff %.2e\n", label, difference))
  invisible(difference)
}

cat("bols form parity against mboost:\n")

# The case that was wrong on its own: a penalized bols sent the centered form.
compare(y ~ bols(x, lambda = 1), "bols(x, lambda=1)")
compare(y ~ bols(x, lambda = 0.1), "bols(x, lambda=0.1)")
compare(y ~ bols(x, lambda = 10), "bols(x, lambda=10)")

# The case that was wrong in both frontends: one penalized term beside an
# unpenalized one, so a block learner and a centered learner shared a formula
# and the constant was represented twice. This was the 1.1e-3 gap.
compare(y ~ bols(x, lambda = 2) + bols(z), "bols(x, lambda=2) + bols(z)")
compare(y ~ bols(z) + bols(x, lambda = 2), "bols(z) + bols(x, lambda=2)")

# The cases that were already right, kept so a fix cannot break them.
compare(y ~ bols(x), "bols(x)")
compare(y ~ bols(x) + bols(z), "bols(x) + bols(z)")
compare(y ~ bols(x, intercept = FALSE), "bols(x, intercept=FALSE)")
compare(y ~ bols(x, lambda = 1, intercept = FALSE), "bols(x, lambda=1, no intercept)")
compare(y ~ bols(x, lambda = 2) + bols(z, lambda = 0), "both blocks")

# The defect moved fitted values and left the selection sequence untouched, so
# a parity test comparing only `selected()` would have passed throughout. This
# records why the check has to be on values.
mixed <- mboostrs::gamboost(y ~ bols(x, lambda = 2) + bols(z), data = d,
                            control = mboostrs::boost_control(mstop = 30, nu = 0.1))
upstream <- mboost::gamboost(y ~ bols(x, lambda = 2) + bols(z), data = d,
                             control = mboost::boost_control(mstop = 30, nu = 0.1))
ok(identical(as.integer(mixed$selected), as.integer(mboost::selected(upstream))),
   "the selection sequences should match, as they did even while the values did not")

# `df=` on a penalized bols needs a solved lambda the R frontend does not
# build, and is refused rather than fitted with the wrong penalty.
refused <- try(mboostrs::gamboost(y ~ bols(x, df = 2), data = d,
                                  control = mboostrs::boost_control(mstop = 5)),
               silent = TRUE)
ok(inherits(refused, "try-error"), "bols(df=) should be refused, not mis-fitted")
ok(grepl("df -> lambda|solved lambda", conditionMessage(attr(refused, "condition"))) ||
     grepl("not supported by the R", conditionMessage(attr(refused, "condition"))),
   paste("the refusal should say why:", conditionMessage(attr(refused, "condition"))))

# A ridge-penalized brandom shrinks the constant direction as well, so the
# residual mean drifts while it is selected and a centered bols beside it
# cannot follow. `brandom(g) + bols(x)` sat 7e-4 from mboost before the menu
# switched to the block form for it. An unpenalized brandom (a plain
# `bols(factor)`) keeps the mean and leaves the centered form in place.
d$g <- factor(sample(letters[1:6], n, replace = TRUE))
d$y <- d$y + c(a = 0, b = 1, c = -1, d = 0.5, e = 0, f = 2)[as.character(d$g)]
compare(y ~ brandom(g) + bols(x), "brandom(g) + bols(x)")
compare(y ~ brandom(g, by = z) + bols(x), "brandom(g, by=z) + bols(x)")
compare(y ~ bols(g) + bols(x), "bols(g) + bols(x)")

# A loss other than squared error moves the mean of its negative gradient,
# so the centered form cannot follow it: bbs(x) + bols(z) was 6.4e-4 from
# mboost for Binomial and 0.053 for Laplace before the block form was used.
d$b <- factor(rbinom(n, 1, plogis(d$x)))
for (fam in c("Binomial", "Laplace", "Poisson")) {
  resp <- switch(fam, Binomial = "b", Laplace = "y", Poisson = "cnt")
  d$cnt <- rpois(n, exp(0.3 * d$x))
  f <- as.formula(paste(resp, "~ bbs(x) + bols(z)"))
  ours <- mboostrs::gamboost(f, data = d, family = get(fam, asNamespace("mboostrs"))(),
                             control = mboostrs::boost_control(mstop = 30, nu = 0.1))
  theirs <- mboost::gamboost(f, data = d, family = get(fam, asNamespace("mboost"))(),
                             control = mboost::boost_control(mstop = 30, nu = 0.1))
  gap <- max(abs(ours$fitted - fitted(theirs)))
  # bbs(x) alone sits about 2e-7 from mboost for Binomial on these 80 rows;
  # the bols(z) term is exact (3e-16 for bols(x) + bols(z)).
  if (gap > 1e-6) stop(fam, ": bbs(x) + bols(z) differs from mboost by ", gap)
}

# The bols form is re-derived when predict() re-parses the formula on new
# data. A factor bols counted as a two-column block when the new rows held two
# of its levels, which switched every bols to the block form: three new rows
# from bols(x) + bols(z) + bols(g) were predicted as six values.
d$g5 <- factor(sample(letters[1:5], n, TRUE))
f <- y ~ bols(x) + bols(z) + bols(g5)
ours <- mboostrs::gamboost(f, data = d, control = mboostrs::boost_control(mstop = 30))
theirs <- mboost::gamboost(f, data = d, control = mboost::boost_control(mstop = 30))
two_levels <- which(d$g5 %in% c("a", "b"))[1:3]
for (rows in list(two_levels, 1L, seq_len(n))) {
  nd <- d[rows, , drop = FALSE]
  gap <- max(abs(predict(ours, newdata = nd) - predict(theirs, newdata = nd)))
  if (length(predict(ours, newdata = nd)) != length(rows) || gap > 1e-10)
    stop("bols(x) + bols(z) + bols(g5) on ", length(rows), " new rows: gap ", gap)
}

cat("bols form parity (R): PASS\n")
