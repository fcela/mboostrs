# support_report(): joint support and unseen levels (issue #67), as Python.
suppressMessages(library(mboostrs))
set.seed(0); n <- 300
d <- data.frame(x1 = runif(n), g = factor(sample(c("a", "b", "c"), n, TRUE)))
d$x2 <- d$x1 + rnorm(n, sd = 0.03)
d$y <- d$x1 + d$x2 + (d$g == "b") + rnorm(n, sd = 0.2)
new <- data.frame(x1 = c(0.5, 0.9), x2 = c(0.5, 0.1), g = c("a", "z"))
for (fit in list(gamboost(y ~ bbs(x1) + bbs(x2) + bols(g), data = d),
                 glmboost(y ~ x1 + x2 + g, data = d))) {
  r <- support_report(fit, new)
  stopifnot(nrow(r) == 2, !r$outside_joint_support[1], r$outside_joint_support[2],
            r$training_share_beyond[2] == 0, r$unseen_levels[2] == "g=z",
            r$unseen_levels[1] == "")
}
cat("test_support_report.R: OK\n")
