# Column names that are not plain identifiers are written in backticks, as in
# base R: bbs(`house age`) + bols(`price/sqft`). Inside backticks a backslash
# escapes a backtick or a backslash (R's parser rule). The backticked name
# resolves to the column with that exact name in fit, predict and coef.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

# (name, formula text of the name)
awkward <- c(
  "house age", "price/sqft", "a+b", "net-gain", "a*b", "a:b", "area^2",
  "pct%", "a~b", "f(x)", "(adj", "a,b", "it's", "say \"hi\"", "a`b", "a\\b",
  "a=1", "1st", "2020", "bbs", "bols", "brandom", "I", "log", "scale", "if",
  "TRUE"
)

set.seed(3)
n <- 200
for (name in awkward) {
  d <- data.frame(y = rnorm(n), z = rnorm(n), check.names = FALSE)
  d[[name]] <- rnorm(n)
  d$y <- 1 + sin(d[[name]]) + 0.5 * d$z + 0.3 * d$y
  q <- mboostrs:::.quote_name(name)
  ok(identical(mboostrs:::.unquote_name(q), name), paste("round trip failed for", name))

  fit <- gamboost(as.formula(paste0("y ~ bbs(", q, ", df = 4) + bols(z)")), data = d,
                  control = boost_control(mstop = 30))
  ok(name %in% names(coef(fit)), paste("gamboost coef names lack", name))
  nd <- d[1:20, ]
  p <- predict(fit, newdata = nd)
  ok(length(p) == 20 && all(is.finite(p)), paste("gamboost predict failed for", name))

  lin <- glmboost(as.formula(paste0("y ~ ", q, " + z")), data = d,
                  control = boost_control(mstop = 30))
  ok(name %in% names(coef(lin)), paste("glmboost coef names lack", name))
  pl <- predict(lin, newdata = nd)
  ok(length(pl) == 20 && all(is.finite(pl)), paste("glmboost predict failed for", name))
}

# A name written without backticks gives an error that shows the backticked form.
d <- data.frame(y = rnorm(n), z = rnorm(n), check.names = FALSE)
d[["price per sqft"]] <- rnorm(n)
msg <- tryCatch(gamboost(y ~ bols(price/sqft), data = d), error = function(e) conditionMessage(e))
ok(grepl("`price/sqft`", msg, fixed = TRUE), paste("no backtick hint in:", substr(msg, 1, 200)))

# A by= column in backticks.
d[["my group"]] <- rbinom(n, 1, 0.5)
fit <- gamboost(y ~ bols(z, by = `my group`), data = d, control = boost_control(mstop = 20))
ok(all(is.finite(predict(fit, newdata = d))), "by = `my group` failed")

# The argument tokenizer keeps a comma or parenthesis inside a name.
args <- mboostrs:::.parse_bl_args("`a,b`, df = 4, boundary.knots = c(0, 10)")
ok(identical(args$positional, "a,b"), "positional name with a comma")
ok(identical(args$named$df, "4"), "named df")
ok(identical(args$named$boundary.knots, "c(0, 10)"), "boundary.knots kept whole")
cat("backtick names: ok\n")
