#@validates parity.r.narrower_surface
# What the R frontend does not do, it says by name.
#
# P1.3 item 2: "Give R the same validated formula, `by`, binning, and factor
# behavior or mark the narrower surface directly in constructors and help
# pages." Marking it means the refusal has to name the feature and say where
# it lives, because the reader arrives from the Python documentation.
#
# Three of the four were doing neither, and one was not refusing at all:
#
#   `bbs(x, df=4) %X% bols(z)` FITTED. The learner regex `^(bbs)\((.+)\)$`
#   matched with the argument group swallowing `x, df = 4) %X% bols(z`, so the
#   term parsed as a plain `bbs(x)` -- 24 coefficients named `x`, the
#   interaction silently gone, and only an "NAs introduced by coercion"
#   warning from the argument reader. Python fits the tensor product, so a
#   ported script got a different model with no error.
#
#   `bins = 32` gave R's generic `unused argument (bins = 32)`, which says
#   nothing about which frontend has it.
#
#   `bols(log(x))` gave "Variable not found: log(x)" -- true, and it reads as
#   a typo in the data rather than as a frontend that does not evaluate
#   transforms.
library(mboostrs)

set.seed(1)
n <- 60
d <- data.frame(x = rnorm(n), z = rnorm(n),
                f = factor(rep(letters[1:3], length.out = n)))
d$y <- d$x + rnorm(n, sd = 0.3)
ctl <- boost_control(mstop = 5, nu = 0.1)

refuses <- function(label, expr, must_say) {
  message_text <- tryCatch({
    eval(expr)
    NULL
  }, error = function(e) conditionMessage(e))
  if (is.null(message_text)) {
    stop(sprintf("%s: accepted, where it should be refused by name", label))
  }
  for (phrase in must_say) {
    if (!grepl(phrase, message_text, fixed = TRUE)) {
      stop(sprintf("%s: the refusal does not say %s; got: %s",
                   label, sQuote(phrase), message_text))
    }
  }
  cat(sprintf("OK %-28s %s\n", label, substr(message_text, 1, 52)))
}

refuses("%X% interaction",
        quote(gamboost(y ~ bbs(x, df = 4) %X% bols(z), data = d,
                       family = Gaussian(), control = ctl)),
        c("%X%", "Python frontend"))

refuses("%O% interaction",
        quote(gamboost(y ~ bbs(x, df = 4) %O% bols(z), data = d,
                       family = Gaussian(), control = ctl)),
        c("%O%", "Python frontend"))

refuses("bins=",
        quote(gamboost(y ~ bbs(x, df = 4), data = d, family = Gaussian(),
                       control = ctl, bins = 32)),
        c("bins=", "Python frontend"))

# Numeric by= is supported for bbs(), bols(), bmono(), bns(), bspatial(),
# brandom(), bmrf() and btree(), factor by= for bbs() and bols() (issues
# #160, #164); bkernel is refused.
d$zby <- rnorm(nrow(d))
refuses("by= on bkernel",
        quote(gamboost(y ~ bkernel(x, by = zby), data = d, family = Gaussian(),
                       control = ctl)),
        c("by=", "bkernel"))

refuses("inline transform",
        quote(gamboost(y ~ bols(log(abs(x) + 1)), data = d,
                       family = Gaussian(), control = ctl)),
        c("is an expression", "Python frontend"))

# A column that is genuinely absent still gets the plain answer, and now says
# which columns there are. Confusing the two messages would be the obvious way
# to get this wrong.
absent <- tryCatch({
  gamboost(y ~ bols(nope), data = d, family = Gaussian(), control = ctl)
  NULL
}, error = function(e) conditionMessage(e))
if (is.null(absent) || !grepl("Variable not found: nope", absent, fixed = TRUE)) {
  stop("a missing column should still say so plainly; got: ", absent)
}
if (!grepl("Columns available", absent, fixed = TRUE)) {
  stop("a missing column should list the columns; got: ", absent)
}
cat(sprintf("OK %-28s %s\n", "missing column", substr(absent, 1, 52)))

# What the R frontend *does* support has to keep working, or the refusals
# above would be easy to get by refusing too much.
for (label in c("y ~ bbs(x, df = 4)", "y ~ bols(x) + bols(z)",
                "y ~ bols(f)", "y ~ f", "y ~ btree(x)")) {
  fit <- gamboost(as.formula(label), data = d, family = Gaussian(),
                  control = ctl)
  if (length(fitted(fit)) != n) stop(label, ": did not fit")
  cat(sprintf("OK %-28s fits\n", label))
}

cat("R narrower surface: every gap refused by name\n")
