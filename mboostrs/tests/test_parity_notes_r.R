#@validates contract.r.parity_notes
# The R frontend reports the same documented differences the Python one does.
#
# P1.4's last open bullet: "R models and serialized artifacts do not carry the
# same first-contact information." Serialized artifacts do. R did not: there
# was no `parity_notes()` at all, and `summary()` printed nothing about them,
# while Python's summary ends with "This fit carries 1 documented parity
# note(s) ... Call parity_notes()".
#
# So a difference between a fit and what its name suggests -- `bns` being
# ridge-penalised, `btree` splitting on squared error, a Gaussian `logLik`
# that is not `mboost`'s number -- was reachable from one frontend and not the
# other.
#
# The notes are *generated* into this package from `mboostrs._parity_notes`,
# the same registry Python reads. Copying them would have been a second prose
# registry, which is the failure this project has paid for repeatedly.
library(mboostrs)

set.seed(1)
n <- 80
d <- data.frame(x = runif(n))
d$y <- sin(2 * pi * d$x) + rnorm(n, sd = 0.2)
ctl <- boost_control(mstop = 15, nu = 0.1)

expect_notes <- function(fit, expected, label) {
  notes <- parity_notes(fit)
  missing <- setdiff(expected, notes$name)
  if (length(missing) > 0) {
    stop(sprintf("%s: missing note(s) %s; got %s", label,
                 paste(missing, collapse = ", "),
                 paste(notes$name, collapse = ", ")))
  }
  for (column in c("mechanism", "consequence", "evidence")) {
    if (!column %in% names(notes)) stop(label, ": no ", column, " column")
  }
  if (!is.logical(notes$shared_with_upstream)) {
    stop(label, ": shared_with_upstream should be logical")
  }
  cat(sprintf("OK %-22s %d note(s)\n", label, nrow(notes)))
}

# The family's note travels with any fit that uses it.
glm_fit <- glmboost(y ~ x, data = d, family = Gaussian(), control = ctl)
expect_notes(glm_fit, "gaussian logLik is a real log-likelihood", "Gaussian")

huber <- glmboost(y ~ x, data = d, family = Huber(), control = ctl)
expect_notes(huber, "adaptive huber risk is not comparable across iterations",
             "Huber")

# The learner's note travels with the term. `btree` is the one that was
# unreachable: its spec `type` is "tree", not "btree", so a lookup keyed by
# the formula spelling missed it.
tree <- gamboost(y ~ btree(x), data = d, family = Gaussian(), control = ctl)
expect_notes(tree, "btree splits on squared error, not conditional inference",
             "btree")

bns_fit <- gamboost(y ~ bns(x, df = 4), data = d, family = Gaussian(),
                    control = ctl)
expect_notes(bns_fit, "bns is ridge-penalized, not unpenalized", "bns")

# A fit with nothing to say says nothing, rather than a row with empty fields.
poisson_d <- data.frame(x = d$x, y = rpois(n, lambda = 3))
poisson_fit <- glmboost(y ~ x, data = poisson_d, family = Poisson(),
                        control = ctl)
if (nrow(parity_notes(poisson_fit)) != 0L) {
  stop("Poisson has no recorded note; parity_notes should be empty")
}
cat("OK Poisson                0 notes\n")

# The global notes are opt-in, as in Python.
if (nrow(parity_notes(glm_fit, include_global = TRUE)) <=
    nrow(parity_notes(glm_fit))) {
  stop("include_global should add the package-level notes")
}
cat("OK include_global         adds package-level notes\n")

# `summary()` is where a reader meets this first, and it said nothing.
for (fit in list(glm_fit, tree)) {
  printed <- paste(capture.output(summary(fit)), collapse = " ")
  if (!grepl("documented parity note", printed, fixed = TRUE)) {
    stop("summary() should point at parity_notes()")
  }
  if (!grepl("Call parity_notes()", printed, fixed = TRUE)) {
    stop("summary() should name the accessor")
  }
}
cat("OK summary()              points at parity_notes()\n")

# The registry is generated, not copied: every note the R package carries has
# a name, a mechanism and a consequence, which is what makes it actionable.
rows <- mboostrs:::.parity_note_rows
if (length(rows) < 15) stop("too few notes generated: ", length(rows))
for (row in rows) {
  if (!nzchar(row$name) || !nzchar(row$mechanism) || !nzchar(row$consequence)) {
    stop("a generated note is missing a field: ", row$name)
  }
  if (!row$shared_with_upstream %in% c("yes", "no")) {
    stop("shared_with_upstream must be yes or no: ", row$name)
  }
}
cat(sprintf("OK generated registry     %d note(s), each complete\n", length(rows)))

cat("R parity notes: the two frontends say the same thing\n")
