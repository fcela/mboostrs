library(mboostrs)
expected <- c(gaussian="fixed", poisson="fixed", nbinomial="line_search", binomial="fixed", gammasreg="fixed")
for (family in names(expected)) {
    actual <- .Call("wrap__r_discovery_mean_step", family, PACKAGE="mboostrs")
    stopifnot(identical(actual, unname(expected[[family]])))
    nu <- .Call("wrap__r_discovery_learning_rate", family, PACKAGE="mboostrs")
    stopifnot(identical(nu, if (family == "gaussian") .1 else .02))
}
for (scale in c(1e-200, 1, 1e200)) {
    for (incumbent in c(.4, .5, .5+.Machine$double.eps)) {
        result <- .Call("wrap__r_prefer_simpler_cv", rep(incumbent*scale,5), rep(.5*scale,5), 0, PACKAGE="mboostrs")
        stopifnot(identical(result$prefer_simple, incumbent <= .5))
    }
}
identity <- mboostrs_build_info()
stopifnot(identical(identity$source_digest, "sha256:8a1e1afb305459b3a8b4abd1e983a4a7c2609bef251def2db016f646a8330d31"))
cat("R family policy, preserved shrinkage, strict incumbent/tie/scaling checks pass\n")
cat(identity$source_digest, "\n")
