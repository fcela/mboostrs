library(mboostrs)

# Unbalanced groups ensure centering changes every level by the same nonzero
# constant. Compare the reported effects with independently replayed predictions.
g <- rep(1:3, c(9, 20, 41))
d <- data.frame(g = g, y = c(-3, 1, 5)[g] + sin(seq_along(g)) / 10)
K <- matrix(c(1, -1, 0, -1, 2, -1, 0, -1, 1), 3)
for (formula in list(y ~ brandom(g, df = 2, center = TRUE),
                     y ~ bmrf(g, K = K, df = 2, center = FALSE),
                     y ~ brandom(g, df = 2, center = FALSE))) {
  for (weights in list(rep(1, length(g)), rep(c(1, 2, 4), length.out = length(g)))) {
    fit <- gamboost(formula, d, weights = weights,
                    control = boost_control(mstop = 40))
    pe <- partial_effects(fit)[[1]]
    expected <- as.numeric(predict(fit, newdata = data.frame(g = 1:3), type = "link")) - fit$offset
    stopifnot(max(abs(pe$effect - expected)) < 1e-10)
    # Exercise the stored column-centering representation directly. Spectral
    # centering can reduce the coefficient dimension; that distinct route is
    # already refused by partial_effects and is outside this regression.
    replay <- fit
    replay$selected <- 1L
    replay$coefs <- list(c(1, 3, -2))
    replay$nu <- 1
    replay$mstop <- 1L
    replay$learner_centering <- list(c(.1, .2, .7))
    pe_replay <- partial_effects(replay)[[1]]$effect
    expected_replay <- as.numeric(predict(replay, newdata = data.frame(g = 1:3),
                                          type = "link")) - replay$offset
    stopifnot(max(abs(pe_replay - expected_replay)) < 1e-10)
    stopifnot(max(abs(pe_replay - c(1.7, 3.7, -1.3))) < 1e-10)
    centers <- fit$learner_centering[[1]]
    if (any(centers != 0)) {
      stopifnot(abs(sum(centers * pe$effect)) < 1e-10)
    }
  }
}

# Native boundary: no centering, dimension errors, and a hand-calculated shift.
call <- function(b, c) .Call("wrap__r_centered_group_effects", b, c, PACKAGE = "mboostrs")
stopifnot(isTRUE(all.equal(call(c(1, 3, -2), c(.1, .2, .7)), c(1.7, 3.7, -1.3))))
stopifnot(identical(call(c(1, 3), numeric()), c(1, 3)))
stopifnot(inherits(tryCatch(call(c(1, 3), .1), error = identity), "error"))
