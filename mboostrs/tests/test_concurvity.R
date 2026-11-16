# concurvity() on a gam_refit() result and on the boosted fits (issue #118).
library(mboostrs)
set.seed(0); n <- 300; x <- runif(n)
d <- data.frame(x = x, z = x + rnorm(n, 0, 0.1), w = runif(n))
d$y <- sin(3 * x) + d$w + rnorm(n, 0, 0.3)
m <- gamboost(y ~ bbs(x) + bbs(z) + bbs(w), data = d, control = boost_control(mstop = 200))
cc <- concurvity(gam_refit(m, which = "all"))
stopifnot(identical(names(cc), c("term", "worst", "observed", "estimate")),
          nrow(cc) == 3L, cc$worst[1] > 0.8, cc$worst[3] < cc$worst[1],
          all(cc$estimate <= cc$worst + 1e-12))

# The boosted fit reports the same measures from its own designs and path.
bc <- concurvity(m)
stopifnot(identical(names(bc), c("term", "worst", "observed", "estimate")),
          nrow(bc) == 3L, bc$worst[2] > 0.8, bc$worst[3] < 0.5,
          all(bc$estimate <= bc$worst + 1e-12))
bp <- concurvity(m, pairwise = TRUE)
stopifnot(identical(names(bp), c("term", "against", "worst", "observed", "estimate")),
          nrow(bp) == 6L,
          bp$worst[bp$term == "z" & bp$against == "x"] > 0.8,
          all(bp$worst[bp$term == "x" & bp$against == "z"] > 0.8))

# A glmboost's terms are one column each, so its concurvity is the
# collinearity of the centered columns, with a NaN row for the intercept.
g <- glmboost(y ~ x + z + w, data = d, control = boost_control(mstop = 100))
gc <- concurvity(g)
stopifnot(identical(gc$term, c("(Intercept)", "x", "z", "w")),
          is.nan(gc$worst[1]), gc$worst[2] > 0.8, gc$worst[4] < 0.05,
          all(abs(gc$worst[2:4] - gc$estimate[2:4]) < 1e-9))
gp <- concurvity(g, pairwise = TRUE)
stopifnot(nrow(gp) == 12L,
          all(is.nan(gp$worst[grepl("(Intercept)", paste(gp$term, gp$against))])))

cat("test_concurvity.R: OK\n")
