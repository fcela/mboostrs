suppressMessages(library(mboostrs))
ns <- asNamespace("mboostrs")
mu <- c(0, 1e-12, .01, .3, 1, 4, 25, 100, 1e5)
y <- c(0, 1, 0, 2, 1, 10, 20, 130, 100100)
for (level in c(.5, .8, .9, .99)) {
  for (score in c("poisson_normalized", "poisson_quantile")) {
    if (score == "poisson_normalized") {
      scale <- sqrt(pmax(mu, 1e-12))
      expected <- abs(y-mu)/scale
      lo <- mu-2*scale
      hi <- mu+2*scale
    } else {
      lo <- qpois((1-level)/2, mu)
      hi <- qpois((1+level)/2, mu)
      expected <- pmax(lo-y, y-hi)
      lo <- lo-2
      hi <- hi+2
    }
    got <- ns$.conformal_split_scores(y, mu, level, score)
    band <- ns$.conformal_split_band(mu, 2, level, score)
    stopifnot(isTRUE(all.equal(got, expected, tolerance=1e-12)),
              isTRUE(all.equal(band$lower, lo, tolerance=1e-12)),
              isTRUE(all.equal(band$upper, hi, tolerance=1e-12)))
  }
}
set.seed(636)
d <- data.frame(x=runif(240, -1, 1))
d$y <- rpois(240, exp(.3+.6*d$x))
model <- gamboost(y ~ bols(x), d[1:120, ], family=Poisson(), control=boost_control(60))
for (score in c("absolute", "poisson_normalized", "poisson_quantile")) {
  result <- conformal_interval(model, d[221:240, ], calibration=d[121:220, ], score=score, mstop=30)
  stopifnot(identical(result$score, score), length(result$lower)==20,
            all(result$lower<=result$upper), result$n_calibration==100)
}
cat("Poisson conformal R numerical references and public calibration passed\n")
