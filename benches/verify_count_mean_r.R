library(mboostrs)
for (scale in c(1e-200, 1, 1e200)) {
  result <- .Call('wrap__r_prefer_simpler_cv', c(1,2,3,4,5)*scale, rep(0,5), 2, PACKAGE='mboostrs')
  stopifnot(identical(result$prefer_simple, FALSE), abs(result$difference/scale-3)<1e-14,
            abs(result$se/scale-sqrt(.5))<1e-14)
}
tie <- .Call('wrap__r_prefer_simpler_cv', rep(1,5), rep(1,5), 2, PACKAGE='mboostrs')
adjacent <- .Call('wrap__r_prefer_simpler_cv', rep(1+.Machine$double.eps,5), rep(1,5), 2, PACKAGE='mboostrs')
stopifnot(isTRUE(tie$prefer_simple), identical(adjacent$prefer_simple,FALSE))
for (family in c('gaussian','poisson','nbinomial','binomial','gammasreg')) {
  actual <- .Call('wrap__r_discovery_learning_rate',family,PACKAGE='mboostrs')
  stopifnot(identical(actual, if(family=='gaussian') .1 else .02))
}
stopifnot(inherits(try(.Call('wrap__r_prefer_simpler_cv', c(1,NA_real_), c(1,2), 2, PACKAGE='mboostrs'),silent=TRUE),'try-error'))
identity <- mboostrs_build_info()
stopifnot(identical(identity$source_digest, 'sha256:3648983f0839dd3a5fb7560853ad696741e27d7adf025d97e25b2a6a5b6d11d7'))
cat('R numerical choice, scale, exact tie, adjacent minimum, invalid-score and unchanged-rate checks passed\n')
cat(identity$source_digest,'\n')
