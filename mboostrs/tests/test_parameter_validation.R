# Every public family parameter is checked before it reaches a numerical
# routine.
#
# A constructor is the last place a bad value can be named against the
# argument the caller wrote. Past that point the value becomes a wrong number
# rather than an error: a non-positive Huber `delta` makes the loss decrease
# with the residual and reports a negative value as risk; a non-positive
# C-index `sigma` inverts the smoothing sigmoid so the loss rewards discordant
# pairs; `n_levels = 1` describes a categorical response with one category,
# which the binding used to repair with `.max(2)` and then fit a different
# model than the caller asked for.
#
# The check runs in the constructor and again in the binding, because a family
# also reaches the engine from cross-validation and stability-selection
# dispatch without passing a constructor.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

rejects <- function(expr, pattern, label) {
  r <- try(eval(expr), silent = TRUE)
  ok(inherits(r, "try-error"), paste("should have been rejected:", label))
  message_text <- conditionMessage(attr(r, "condition"))
  ok(grepl(pattern, message_text, fixed = TRUE),
     sprintf("%s: expected a message containing '%s', got '%s'",
             label, pattern, message_text))
}

# ─── the checkers ──────────────────────────────────────────────────

rejects(quote(mboostrs:::.check_unit_interval("tau", 0)), "tau must be in (0, 1)", "tau = 0")
rejects(quote(mboostrs:::.check_unit_interval("tau", 1)), "tau must be in (0, 1)", "tau = 1")
rejects(quote(mboostrs:::.check_unit_interval("tau", NA)), "tau must be in (0, 1)", "tau = NA")
rejects(quote(mboostrs:::.check_unit_interval("tau", c(0.2, 0.3))), "tau must be in (0, 1)", "tau vector")
ok(mboostrs:::.check_unit_interval("tau", 0.25) == 0.25, "0.25 should be accepted")

rejects(quote(mboostrs:::.check_positive("delta", 0)), "delta must be positive", "delta = 0")
rejects(quote(mboostrs:::.check_positive("delta", -1)), "delta must be positive", "delta = -1")
rejects(quote(mboostrs:::.check_positive("delta", Inf)), "delta must be positive", "delta = Inf")
ok(mboostrs:::.check_positive("delta", 2) == 2, "2 should be accepted")

rejects(quote(mboostrs:::.check_int_at_least("k", 2, 3L)), "k must be at least 3", "k = 2")
rejects(quote(mboostrs:::.check_int_at_least("k", 3.5, 3L)), "k must be a whole number", "k = 3.5")
ok(mboostrs:::.check_int_at_least("k", 4, 3L) == 4, "4 should be accepted")

# ─── family constructors ───────────────────────────────────────────

rejects(quote(Quantile(qoffset = 2)), "qoffset must be in (0, 1)", "Quantile(qoffset=2)")
rejects(quote(Quantile(qoffset = 0)), "qoffset must be in (0, 1)", "Quantile(qoffset=0)")
rejects(quote(Quantile(tau = 1)), "tau must be in (0, 1)", "Quantile(tau=1)")
rejects(quote(Expectile(tau = -0.2)), "tau must be in (0, 1)", "Expectile(tau=-0.2)")
rejects(quote(Huber(delta = -1)), "delta must be positive", "Huber(delta=-1)")
rejects(quote(Huber(delta = 0)), "delta must be positive", "Huber(delta=0)")
rejects(quote(PropOdds(n_levels = 1)), "n_levels must be at least 3", "PropOdds(1)")
rejects(quote(PropOdds(n_levels = 2)), "n_levels must be at least 3", "PropOdds(2)")
rejects(quote(Multinomial(classes = 1)), "classes must be at least 3", "Multinomial(1)")
rejects(quote(Multinomial(classes = 2)), "classes must be at least 3", "Multinomial(2)")
rejects(quote(Multinomial(classes = 3.5)), "classes must be a whole number", "Multinomial(3.5)")
rejects(quote(Cindex(sigma = -1)), "sigma must be positive", "Cindex(sigma=-1)")
rejects(quote(Cindex(sigma = 0)), "sigma must be positive", "Cindex(sigma=0)")

# The valid cases still construct, and carry the value through.
ok(identical(Quantile(tau = 0.9, qoffset = 0.1)$params$qoffset, 0.1),
   "Quantile should carry qoffset")
ok(length(Huber()$params) == 0L, "Huber() should send no delta")
ok(identical(Huber(delta = 2)$params$delta, 2), "Huber(2) should carry delta")
ok(length(PropOdds()$params) == 0L, "PropOdds() should send no n_levels")
ok(identical(PropOdds(n_levels = 4)$params$n_levels, 4), "PropOdds(4) should carry it")
ok(identical(Cindex()$params$sigma, 0.1), "Cindex() default sigma")

# ─── the binding dispatch ──────────────────────────────────────────
#
# A family object built by hand -- as one arrives from a serialized model --
# is checked at the binding rather than trusted.

set.seed(3); n <- 40
d <- data.frame(x = rnorm(n))
d$y <- 2 * d$x + rnorm(n)

hand_built <- function(name, params) {
  structure(list(name = name, params = params), class = "mboostrs_family")
}

# A survival fixture, so the response check for the ranking families passes
# and the parameter check is the one that fires. `Surv` lives in the survival
# package; without it the C-index case is skipped rather than reported as a
# validation failure.
ds <- data.frame(x = d$x, time = abs(d$y) + 1, event = rep(c(1, 0), length.out = n))
have_surv <- requireNamespace("survival", quietly = TRUE)
if (have_surv) suppressMessages(library(survival))

for (case in list(
  list(fam = hand_built("quantreg", list(tau = 0)), pattern = "tau must be in (0, 1)",
       formula = y ~ x, data = d),
  list(fam = hand_built("quantreg", list(qoffset = 2)), pattern = "qoffset must be in (0, 1)",
       formula = y ~ x, data = d),
  list(fam = hand_built("expectreg", list(tau = 1)), pattern = "tau must be in (0, 1)",
       formula = y ~ x, data = d),
  list(fam = hand_built("huber", list(delta = -1)), pattern = "delta must be positive",
       formula = y ~ x, data = d),
  list(fam = hand_built("propodds", list(n_levels = 1)), pattern = "n_levels must be at least 3",
       formula = y ~ x, data = d),
  # No Multinomial case: glmboost refuses the family before the binding, and
  # gamboost sets n_classes from the response's levels.
  if (have_surv)
    list(fam = hand_built("cindex", list(sigma = -1)), pattern = "sigma must be positive",
         formula = Surv(time, event) ~ x, data = ds)
)) {
  if (is.null(case)) next
  r <- try(glmboost(case$formula, data = case$data, family = case$fam,
                    control = boost_control(mstop = 5)), silent = TRUE)
  ok(inherits(r, "try-error"),
     paste("binding should reject", case$fam$name, case$pattern))
  ok(grepl(case$pattern, conditionMessage(attr(r, "condition")), fixed = TRUE),
     sprintf("binding message for %s should contain '%s', got '%s'",
             case$fam$name, case$pattern,
             conditionMessage(attr(r, "condition"))))
}

# Rejecting bad values must not have made the defaults unreachable.
for (fam in list(Quantile(), Expectile(), Huber(), Gaussian())) {
  fit <- glmboost(y ~ x, data = d, family = fam,
                  control = boost_control(mstop = 10, nu = 0.1))
  ok(all(is.finite(fitted(fit))), paste("default", fam$name, "should still fit"))
}

cat("parameter validation (R): PASS\n")
