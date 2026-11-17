# predict(which = ) for a grouped learner (brandom, bmrf) returned the raw
# coefficient b_g of each row's group. A centered term's contribution to the
# fit is b_g - sum_j c_j b_j for brandom(center = TRUE), with c the weighted
# group shares, and Z b for bmrf(center = TRUE), whose coefficients index the
# penalized eigen-directions Z of the neighbourhood matrix K. The core's
# learner-effect function now computes both (issue #879). The expected values
# are Python's predict(which = 1) on the same data (1e-8 relative); the data
# are built without a random number generator so both languages hold them.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
close <- function(got, want, label, tol = 1e-8) {
  got <- unname(as.numeric(got)); want <- as.numeric(want)
  ok(length(got) == length(want), paste(label, ": length", length(got), "vs", length(want)))
  err <- max(abs(got - want) / pmax(abs(want), 1e-12))
  ok(err <= tol, sprintf("%s: relative difference %.3g", label, err))
}

n <- 120
i <- seq_len(n)
x <- (i * 0.6180339887498949) %% 1
g <- c(1, 1, 1, 2, 2, 3, 3, 1, 4, 5, 2, 1)[(i - 1) %% 12 + 1]
y <- sin(2 * pi * x) + c(-1, 0.5, 1, -0.3, 0.2)[g] + 0.3 * sin(1.7 * i)
d <- data.frame(x = x, g = as.integer(g), y = y)
newdata <- data.frame(x = c(0.1, 0.5, 0.9, 0.3), g = c(3L, 1L, 5L, 2L))
# Chain neighbourhood 1-2-3-4-5 as a graph Laplacian.
K <- matrix(0, 5, 5)
for (a in 1:4) {
  K[a, a + 1] <- K[a + 1, a] <- -1
  K[a, a] <- K[a, a] + 1
  K[a + 1, a + 1] <- K[a + 1, a + 1] + 1
}
ctrl <- boost_control(mstop = 60)

check <- function(model, label, in_sample, new_rows) {
  close(predict(model, which = 2)[1:12, 1], in_sample, paste(label, "in-sample"))
  close(predict(model, newdata = newdata, which = 2)[, 1], new_rows, paste(label, "newdata"))
  # The two terms' contributions and the offset add up to the fitted values.
  parts <- predict(model, which = 1)[, 1] + predict(model, which = 2)[, 1]
  ok(max(abs(parts + model$offset - fitted(model))) < 1e-10,
     paste(label, ": contributions do not add up to the fitted values"))
}

m <- gamboost(y ~ bbs(x, df = 4) + brandom(g, df = 3, center = TRUE), data = d, control = ctrl)
check(m, "brandom(center = TRUE)",
      c(-0.7509064871757845, -0.7509064871757845, -0.7509064871757845, 0.5709169924806738,
        0.5709169924806738, 0.9188812928605415, 0.9188812928605415, -0.7509064871757845,
        -0.034793424589192656, 0.23881229730501083, 0.5709169924806738, -0.7509064871757845),
      c(0.9188812928605415, -0.7509064871757845, 0.23881229730501083, 0.5709169924806738))
# At mstop = 30 the expected values are a Python fit with mstop = 30 recorded
# on arm64 macOS. Thirty boosting steps accumulate floating-point rounding
# differently on other CPUs (3.3e-8 relative on x86_64 Linux), so this
# comparison uses 1e-6 where the one-step comparisons above use 1e-8.
close(predict(m, which = 2, mstop = 30)[1:12, 1],
      c(-0.6062962275272437, -0.6062962275272437, -0.6062962275272437, 0.464152042981799,
        0.464152042981799, 0.7191852034855635, 0.7191852034855635, -0.6062962275272437,
        0.0008217228756568718, 0.19983287884403755, 0.464152042981799, -0.6062962275272437),
      "brandom(center = TRUE) mstop = 30", tol = 1e-6)
# The value before the fix was the uncentered coefficient b_g.
b <- predict(m, which = 2)[c(1, 4, 6, 9, 10), 1]
share <- as.numeric(table(d$g)) / n
ok(abs(sum(share * b)) < 1e-12, "centered brandom contribution has nonzero weighted mean")

m <- gamboost(y ~ bbs(x, df = 4) + brandom(g, df = 3), data = d, control = ctrl)
check(m, "brandom",
      c(-0.787368021082395, -0.787368021082395, -0.787368021082395, 0.4916813728060925,
        0.4916813728060925, 0.7844558828017418, 0.7844558828017418, -0.787368021082395,
        -0.0979475576412055, 0.12967068885047708, 0.4916813728060925, -0.787368021082395),
      c(0.7844558828017418, -0.787368021082395, 0.12967068885047708, 0.4916813728060925))

m <- gamboost(y ~ bbs(x, df = 4) + bmrf(g, K = K, df = 3), data = d, control = ctrl)
check(m, "bmrf",
      c(-0.7111847831562077, -0.7111847831562077, -0.7111847831562077, 0.46878230770106866,
        0.46878230770106866, 0.8114687341644524, 0.8114687341644524, -0.7111847831562077,
        0.27240322587632115, 0.2542362984725762, 0.46878230770106866, -0.7111847831562077),
      c(0.8114687341644524, -0.7111847831562077, 0.2542362984725762, 0.46878230770106866))

m <- gamboost(y ~ bbs(x, df = 4) + bmrf(g, K = K, df = 3, center = TRUE), data = d, control = ctrl)
check(m, "bmrf(center = TRUE)",
      c(-0.9073806891720063, -0.9073806891720063, -0.9073806891720063, 0.3980310697646603,
        0.3980310697646603, 0.7393938548903435, 0.7393938548903435, -0.9073806891720063,
        -0.16244258925421945, -0.06760164622877794, 0.3980310697646603, -0.9073806891720063),
      c(0.7393938548903435, -0.9073806891720063, -0.06760164622877794, 0.3980310697646603))
close(predict(m, which = 2, mstop = 30)[1:12, 1],
      c(-0.6958916391801755, -0.6958916391801755, -0.6958916391801755, 0.30150769719114684,
        0.30150769719114684, 0.5507592856816633, 0.5507592856816633, -0.6958916391801755,
        -0.08685799831229224, -0.06951734538034238, 0.30150769719114684, -0.6958916391801755),
      "bmrf(center = TRUE) mstop = 30", tol = 1e-6)

cat("test_predict_which_centred.R: OK\n")
