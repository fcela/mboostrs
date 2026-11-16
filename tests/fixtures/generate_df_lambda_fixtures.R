#!/usr/bin/env Rscript
# R reference values for the df -> lambda map.
#
# Run from the repo root:  Rscript tests/fixtures/generate_df_lambda_fixtures.R
# Output: tests/fixtures/df_lambda.json
#
# `mboost:::df2lambda` is called exactly as `X_bbs` calls it, so the recorded
# lambda is the one a fitted base learner would use. The cases include df at
# and below the rank of the penalty's null space, where the effective-df curve
# is flat in lambda and the root is therefore decided by the root-finder's
# stopping rule rather than by the data.

suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

OUT <- file.path("tests", "fixtures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

macro <- read.csv(file.path("python", "mboostrs", "datasets", "usmacro.csv"))
macro <- macro[macro$year <= 1993, ]
rate <- macro$tbilrate

set.seed(11)
cases <- list(
    list(label = "tbill_rate", x = rate - mean(rate),
         weights = rep(1, length(rate)), knots = 20, degree = 3, differences = 2),
    list(label = "uniform_60", x = seq(0, 10, length.out = 60),
         weights = rep(1, 60), knots = 20, degree = 3, differences = 2),
    list(label = "uniform_60_weighted", x = seq(0, 10, length.out = 60),
         weights = rep(c(1, 2, 3, 0.5), length.out = 60),
         knots = 20, degree = 3, differences = 2),
    list(label = "clustered_200", x = c(rnorm(150, 2, 0.4), rnorm(50, 8, 1.5)),
         weights = rep(1, 200), knots = 20, degree = 3, differences = 2),
    list(label = "first_differences", x = seq(-3, 3, length.out = 80),
         weights = rep(1, 80), knots = 15, degree = 3, differences = 1),
    list(label = "degree2_diff3", x = seq(0, 1, length.out = 120),
         weights = rep(1, 120), knots = 25, degree = 2, differences = 3)
)

# Dense just above 2, which is where the plateau of the second-difference
# penalty ends and lambda becomes identified again.
DF_GRID <- c(0.5, 1, 1.5, 1.9, 2, 2.0000001, 2.000001, 2.00001, 2.0001,
             2.001, 2.01, 2.05, 2.1, 2.5, 3, 4, 6, 10, 24)

# A two-margin tensor, whose penalty null space is the product of the margins':
# 2 x 2 = 4. Included because a tensor pinned at df = 4 is a case where R and
# any other implementation agree on the fitted values for the wrong reason —
# both return the projection onto {1, a, b, ab}, from lambdas that differ by
# decades, because neither lambda matters.
tensor_case <- function() {
    n <- 400
    a <- seq(0, 10, length.out = n)
    b <- ((seq_len(n) * 7) %% n) / (n - 1) * 5
    basis <- function(v, upper) {
        inner <- seq(0, upper, length = 8)[2:7]
        X <- mboost:::bsplines(v, knots = inner, boundary.knots = c(0, upper), degree = 3)
        class(X) <- "matrix"
        X
    }
    B1 <- basis(a, 10)
    B2 <- basis(b, 5)
    p1 <- ncol(B1); p2 <- ncol(B2)
    X <- matrix(0, n, p1 * p2)
    for (i in seq_len(n)) X[i, ] <- as.vector(outer(B1[i, ], B2[i, ]))
    D1 <- crossprod(diff(diag(p1), differences = 2))
    D2 <- crossprod(diff(diag(p2), differences = 2))
    K <- kronecker(D1, diag(p2)) + kronecker(diag(p1), D2)
    grid <- c(2, 3, 4, 4.2, 4.4, 5, 6, 9, 16)
    lambdas <- sapply(grid, function(df) {
        value <- suppressWarnings(
            mboost:::df2lambda(X, df = df, dmat = K, weights = rep(1, n)))
        as.numeric(value["lambda"])
    })
    list(label = "tensor_2x2", a = a, b = b, weights = rep(1, n),
         knots = 6, degree = 3, differences = 2,
         n_columns = p1 * p2, null_space_rank = 4,
         boundary_a = 10, boundary_b = 5,
         df = grid, lambda = lambdas)
}

records <- lapply(cases, function(case) {
    boundary <- range(case$x)
    inner <- seq(boundary[1], boundary[2], length = case$knots + 2)
    inner <- inner[2:(length(inner) - 1)]
    X <- mboost:::bsplines(case$x, knots = inner, boundary.knots = boundary,
                           degree = case$degree)
    class(X) <- "matrix"
    K <- diff(diag(ncol(X)), differences = case$differences)
    K <- crossprod(K)
    lambdas <- sapply(DF_GRID, function(df) {
        if (df >= ncol(X)) return(0)
        value <- suppressWarnings(
            mboost:::df2lambda(X, df = df, dmat = K, weights = case$weights))
        as.numeric(value["lambda"])
    })
    list(label = case$label, x = case$x, weights = case$weights,
         knots = case$knots, degree = case$degree,
         differences = case$differences, n_columns = ncol(X),
         null_space_rank = case$differences,
         df = DF_GRID, lambda = lambdas)
})

records <- c(records, list(tensor_case()))

write_json(list(
    mboost_version = as.character(packageVersion("mboost")),
    r_version = R.version.string,
    lambda_max = options("mboost_lambdaMax")[[1]],
    df_trace_s = options("mboost_dftraceS")[[1]],
    eps = options("mboost_eps")[[1]],
    uniroot_tol = sqrt(.Machine$double.eps),
    cases = records
), file.path(OUT, "df_lambda.json"), auto_unbox = TRUE, digits = 17)

cat("wrote df_lambda.json:", length(records), "designs x", length(DF_GRID), "df values\n")
