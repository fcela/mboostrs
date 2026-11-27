\name{base_learners}
\alias{base_learners}
\alias{bols}
\alias{bbs}
\alias{bns}
\alias{bmono}
\alias{bconstrained}
\alias{brandom}
\alias{bmrf}
\alias{bspatial}
\alias{btree}
\alias{bkernel}
\alias{buser}
\title{Base learners in mboostrs formulas}

\description{
The component-wise fitting functions choose, at each boosting iteration, the
one base learner that reduces the risk most. A base learner names a predictor
and says how to expand it: linearly, into a spline basis, into one indicator
per group, and so on.

In mboostrs these are \emph{formula syntax}, not callable functions. Writing
\code{bols(x)} at the R prompt is an error, because no such object is exported;
the term is parsed out of the formula you pass to \code{\link{gamboost}} or
\code{\link{gamboostLSS}}. This differs from upstream \pkg{mboost}, where
\code{bols} is an exported function returning a base-learner object, so an
mboost tutorial that assigns \code{bl <- bols(x)} does not transfer. The
formula spelling is otherwise the same.
}

\section{Catalogue}{
\describe{
  \item{\code{bols(x)}}{Linear effect. The design is the column itself plus a
    per-learner intercept. A factor expands to one indicator per level. This is
    the only learner \code{\link{glmboost}} uses, and a bare column name in a
    \code{glmboost} formula means this.}
  \item{\code{bbs(x, df = 4, knots = 20, degree = 3, differences = 2, knot_placement = c("equidistant", "quantile"), bins = NULL)}}{
    Penalized B-spline (P-spline, or quantile O-spline when
    \code{knot_placement = "quantile"}). Optionally discretizes continuous
    values into \code{bins} quantile bins (e.g. \code{bins = 256}) for fast
    \eqn{O(n + B \cdot K)} evaluation. A bare column name in a \code{gamboost}
    formula means \code{bbs(x)}.}
  \item{\code{bns(x, df = NULL)}}{The same B-spline basis with an identity
    ridge penalty instead of a difference penalty. Having no null space, the
    fitted curve can shrink to zero entirely, and any positive \code{df} is
    attainable.}
  \item{\code{bmono(x, constraint = "increasing")}}{P-spline constrained to be
    increasing, decreasing, convex, concave, positive, or negative.}
  \item{\code{bconstrained(x, C = C)}}{P-spline subject to a caller-supplied
    linear constraint matrix \eqn{C\beta \ge 0}. Bind \code{C} by name from the
    calling environment.}
  \item{\code{brandom(g, df = 4)}}{Random effect: one indicator column per
    level of \code{g}, ridge-penalized.}
  \item{\code{bmrf(region, K = K)}}{Markov random field — a graph-smoothed
    categorical effect using a supplied neighbourhood penalty matrix. \code{K}
    is resolved from the formula environment, or from an upstream-style
    \code{bnd =} argument.}
  \item{\code{bspatial(x1, x2, df = 6)}}{Tensor-product P-spline surface over
    two predictors, with the additive Kronecker penalty. Its penalty null space
    has dimension \code{differences^2}, so \code{df} must be at least 4 at the
    defaults.}
  \item{\code{btree(x, maxdepth = 1, selection = "ctree", split_method = c("exact", "histogram"), lambda_leaf = 0, colsample = 1)}}{
    Decision stump or shallow interaction tree over one or more predictors,
    supporting exact or 256-bin histogram split scanning, L2 leaf shrinkage
    (\code{lambda_leaf}), and deterministic per-step column subsampling
    (\code{colsample}, \code{colsample_seed}).}
  \item{\code{bkernel(x, bandwidth = 1)}}{Kernel ridge effect built from a
    dense kernel matrix. Upstream \code{mboost::bkernel} constructs its kernel
    from genotype data for pathway and GWAS analysis; this learner does not
    reproduce that construction, so a model written against upstream
    \code{bkernel} will not transfer unchanged.}
  \item{\code{buser(x, X = X, K = K)}}{Effect built from caller-supplied design
    and penalty matrices.}
}
Supported \code{by =} modifiers create varying-coefficient terms. For example,
\code{bols(x, by = z)} multiplies the design by \code{z} element-wise.
Support depends on the learner and whether \code{z} is numeric or a factor;
unsupported combinations raise an error naming the learner.
}

\section{Fitting functions and learner support}{
\code{gamboost} and \code{gamboostLSS} accept learner terms in their formulas;
support depends on the learner and response family. \code{glmboost} and
\code{glmboostLSS} fit one linear term per predictor, so they take \code{bols}
or a bare column name and reject the rest with an error naming the learner.

The R frontend implements \code{bols}, \code{bbs}, \code{bns}, \code{bmono},
\code{bconstrained}, \code{btree}, \code{brandom}, \code{bmrf}, and \code{bspatial}.
The standard R \code{gamboost} fitting path refuses \code{bkernel} and
\code{buser} with \dQuote{Unknown learner type}. See
\code{\link{mboostrs_capabilities}} for support by family and operation.
}

\section{R formula restrictions}{
The following Python formula features are refused by the R frontend:

\describe{
  \item{\code{\%X\%} and \code{\%O\%}}{Tensor interactions between two
    learners. Supply a precomputed interaction column where appropriate.}
  \item{\code{bins =}}{Quantile binning of a continuous design, which is a
    \code{gamboost()} argument in Python.}
  \item{Inline transforms}{\code{bols(log(x))} and the like. A formula term
    here is a column name, not an expression; add the computed column to
    \code{data} and name it. Python stores the transform recipe with the
    model and replays it on new data.}
}

Tree prediction on \code{newdata} reuses the fitted split rules and factor
levels. The prediction method's \code{unseen_levels} argument controls the
handling of factor levels absent from the fitting rows.
}

\examples{
set.seed(1)
n <- 200
d <- data.frame(
  x  = runif(n, 0, 1),
  z  = rnorm(n),
  g  = rep(1:5, length.out = n)
)
d$y <- sin(2 * pi * d$x) + 0.5 * d$z + rnorm(n, sd = 0.3)

# A smooth term, a linear term, and a random effect in one formula.
fit <- gamboost(y ~ bbs(x, df = 4) + bols(z) + brandom(g, df = 4),
                data = d, control = boost_control(mstop = 100, nu = 0.1))
head(coef(fit))

# bols(x) at the prompt is an error: these are formula syntax, not functions.
\dontrun{
bols(d$x)
}
}

\seealso{
\code{\link{gamboost}}, \code{\link{glmboost}}, \code{\link{gamboostLSS}},
\code{\link{mboostrs_capabilities}} for the machine-readable support table.
}
