# partial_effects() for gamboost formed design %*% coefficients in R and
# computed a brandom/bmrf effect as b * (1 - c), where the fit's
# contribution of group g is b_g - sum_j c_j b_j (c the column centering).
# The R code also never saw the centering the core applies for
# brandom(center = TRUE), so it returned b. The effects are now computed by
# the core (issue #848). Expected values for the uncentered learners are the
# results of the R implementation before the port (1e-8 relative).
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
close <- function(got, want, label, tol = 1e-8) {
  got <- unname(as.numeric(got)); want <- as.numeric(want)
  ok(length(got) == length(want), paste(label, ": length", length(got), "vs", length(want)))
  err <- max(abs(got - want) / pmax(abs(want), 1e-12))
  ok(err <= tol, sprintf("%s: relative difference %.3g", label, err))
}

set.seed(849)
n2 <- 150
d2 <- data.frame(x1 = runif(n2), x2 = rnorm(n2),
                 g = sample(1:5, n2, replace = TRUE, prob = c(.3, .25, .2, .15, .1)))
geff <- c(-1, 0.5, 1, -0.3, 0.2)
d2$y <- sin(2 * pi * d2$x1) + 0.5 * d2$x2 + geff[d2$g] + rnorm(n2, sd = 0.3)

m2 <- gamboost(y ~ bbs(x1, df = 4) + bols(x2) + brandom(g, df = 3), data = d2,
               control = boost_control(mstop = 60))
pe <- partial_effects(m2, n_points = 7)
ok(identical(names(pe), c("x1", "g")), "partial_effects names")
close(pe$x1$x, c(0.00872125453315675, 0.171716823591851, 0.334712392650545,
                 0.497707961709239, 0.660703530767933, 0.823699099826626,
                 0.98669466888532), "bbs grid")
close(pe$x1$effect, c(0.322215959643961, 0.803600883605644, 0.731972935617485,
                      0.0863275324655536, -0.52067826217858, -0.640505158539418,
                      -0.339746109294416), "bbs effect")
# Uncentered brandom: the effect is the summed coefficient vector.
close(pe$g$effect, c(-0.746005725324418, 0.297893076698479, 0.644393018060834,
                     -0.162561920187678, 0.107407367717283), "brandom effect")

m3 <- gamboost(y ~ bns(x1, df = 4) + bols(x2), data = d2,
               control = boost_control(mstop = 30))
close(partial_effects(m3, n_points = 5)$x1$effect,
      c(-0.0221167464298692, 0.518428364611671, -0.0663476921478875,
        -0.380811351041645, -0.0490409929436521), "bns effect")

# brandom(center = TRUE): the effect of group g is the fit's contribution of
# the rows in g, and its group-share-weighted mean is zero.
m6 <- gamboost(y ~ bbs(x1, df = 4) + brandom(g, df = 3, center = TRUE), data = d2,
               control = boost_control(mstop = 60))
eff <- partial_effects(m6)$g$effect
contribution <- fitted(m6) - m6$offset - as.numeric(predict(m6, which = 1))
ok(max(abs(contribution - eff[d2$g])) < 1e-12,
   "centered brandom effect differs from its contribution to the fit")
share <- as.numeric(table(d2$g)) / n2
ok(abs(sum(share * eff)) < 1e-12, "centered brandom effect has nonzero weighted mean")
# The earlier R value (the uncentered coefficient sum) is off by c'b.
old <- c(-0.904334111029919, 0.285373979547653, 0.705419664947338,
         -0.191133778112699, 0.104674244647623)
close(eff, old - sum(share * old), "centered brandom effect vs b - c'b")

cat("test_partial_effects_core.R: OK\n")
