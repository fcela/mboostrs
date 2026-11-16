# btree over several columns, one of them a factor: the tree splits the
# factor into two sets of its levels, predict() evaluates the stored trees on
# new rows, and a level the fit never saw follows the heavier child.
suppressMessages(library(mboostrs))
set.seed(3); n <- 600
g <- factor(sample(c("a", "b", "c", "d", "e"), n, TRUE)); x <- runif(n, -1, 1)
y <- ifelse(g %in% c("a", "c"), 2 * x, -x) + rnorm(n, 0, 0.2)
d <- data.frame(y = y, g = g, x = x)
fit <- gamboost(y ~ btree(g, x, maxdepth = 3), data = d, control = boost_control(mstop = 150))
stopifnot(max(abs(predict(fit, newdata = d) - fitted(fit))) < 1e-10)
nd <- data.frame(g = c("a", "b", "z"), x = c(0.8, 0.8, 0.8))
# An unseen level is refused by default; "zero_effect" sends it down the
# branch that held more training weight.
stopifnot(inherits(try(predict(fit, newdata = nd), silent = TRUE), "try-error"))
p <- predict(fit, newdata = nd, unseen_levels = "zero_effect")
stopifnot(p[1] > 1, p[2] < -0.4, p[3] < -0.4)
cat("btree categorical (R): PASS\n")
