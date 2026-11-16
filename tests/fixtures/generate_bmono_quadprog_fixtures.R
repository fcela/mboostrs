# Fixtures for tests/test_bmono_quadprog_parity.py
#
# mboost's `bmono(type = "quad.prog")` and ours both run a primal active-set QP
# (Nocedal-Wright 16.3) over the same H = X'X + lambda*K, but through different
# Cholesky code, so agreement is a numerical claim that needs pinning rather
# than an identity.
#
# The design is what makes the comparison mean anything. A monotone constraint
# fitted to data that already respects it never activates a single constraint,
# and the QP collapses to the unconstrained solve — a test built on that data
# passes whatever the solver does. The signal here rises overall but dips
# locally, so an `increasing` constraint pins some segments and leaves others
# free, and which ones changes from iteration to iteration.
#
# Run:  Rscript tests/fixtures/generate_bmono_quadprog_fixtures.R
# Needs: mboost 2.9.11, jsonlite
suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

set.seed(11)
n <- 80
x <- seq(0, 10, length.out = n)
y <- 0.35 * x + 1.4 * sin(x) + rnorm(n, 0, 0.25)
d <- data.frame(x = x, y = y)

NU <- 0.1

specs <- list(
  list(constraint = "increasing", df = 8, mstop = 60),
  list(constraint = "increasing", df = 12, mstop = 60),
  list(constraint = "convex", df = 4, mstop = 60),
  # The same case run four times longer, to record what the accumulation does
  # rather than assume it stays where it was at iteration 60.
  list(constraint = "convex", df = 4, mstop = 300)
)

cases <- list()
for (spec in specs) {
  fit <- mboost(
    y ~ bmono(x, constraint = spec$constraint, df = spec$df, type = "quad.prog"),
    data = d, control = boost_control(mstop = spec$mstop, nu = NU)
  )
  # `fit[k]` truncates in place, so walk the path upwards and read each fit.
  per_iteration <- lapply(seq_len(spec$mstop), function(k) as.numeric(fitted(fit[k])))
  fit <- mboost(
    y ~ bmono(x, constraint = spec$constraint, df = spec$df, type = "quad.prog"),
    data = d, control = boost_control(mstop = spec$mstop, nu = NU)
  )
  cases[[length(cases) + 1]] <- list(
    constraint = spec$constraint,
    df = spec$df,
    mstop = spec$mstop,
    fitted_path = per_iteration,
    fitted = as.numeric(fitted(fit)),
    risk = as.numeric(risk(fit))
  )
}

writeLines(
  toJSON(list(x = x, y = y, nu = NU, cases = cases),
         digits = 17, auto_unbox = TRUE),
  "tests/fixtures/bmono_quadprog.json"
)
cat("wrote tests/fixtures/bmono_quadprog.json:", length(cases), "cases\n")
