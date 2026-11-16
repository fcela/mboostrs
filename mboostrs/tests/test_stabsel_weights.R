# stabsel() took `...` and dropped it, so weights = was ignored without a
# message: here y depends on x1 in the first half of the rows and on x2 in the
# second, and with weight zero on the second half the run still selected x2
# in every subsample (max frequency 1, identical to the unweighted run). A
# misspelled argument was dropped the same way (issue #823, item 5). The
# subsample folds now carry the fit's weights, and `...` is refused.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(5)
n <- 200
x <- matrix(rnorm(n * 6), n, 6)
colnames(x) <- paste0("x", 1:6)
d <- as.data.frame(x)
first <- seq_len(n) <= n / 2
d$y <- ifelse(first, 2 * d$x1, 2 * d$x2) + rnorm(n, sd = 0.5)
w <- as.double(first)
ctl <- boost_control(mstop = 30)

s0 <- stabsel(y ~ ., data = d, control = ctl, B = 20, q = 2, cutoff = 0.7)
s1 <- stabsel(y ~ ., data = d, control = ctl, B = 20, q = 2, cutoff = 0.7, weights = w)
ok(s0$max[["x2"]] == 1, "unweighted run should select x2 in every subsample")
ok(s1$max[["x1"]] == 1, "weighted run should select x1 in every subsample")
ok(s1$max[["x2"]] < 0.5,
   sprintf("weighted run selects x2 with frequency %.2f; weights were ignored", s1$max[["x2"]]))
ok(!("x2" %in% s1$selected), "x2 is stable although its rows have weight zero")

# Each subsample fit is the glmboost fit on that fold's weights.
folds <- stabsel_folds(n, B = 20, sampling_type = "SS", weights = w, seed = 42L)
fit1 <- glmboost(y ~ ., data = d, control = ctl, weights = folds[, 1])
ok(all(folds[!first, ] == 0), "weighted folds put weight on zero-weight rows")
ok(length(fit1$selected) == 30L, "subsample fit failed")

# Unit weights reproduce the unweighted run.
s2 <- stabsel(y ~ ., data = d, control = ctl, B = 20, q = 2, cutoff = 0.7,
              weights = rep(1, n))
ok(identical(s2$max, s0$max), "unit weights changed the selection frequencies")

msg <- tryCatch({ stabsel(y ~ ., data = d, control = ctl, B = 5, q = 2,
                          cutoff = 0.7, wieghts = w); NULL },
                error = conditionMessage)
ok(!is.null(msg) && grepl("wieghts", msg), "a misspelled argument was dropped silently")

cat("test_stabsel_weights: OK\n")
