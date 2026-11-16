# Both bindings expose the same Rust report; R matrices are flattened by row.
suppressPackageStartupMessages(library(mboostrs))
x <- expand.grid(x = seq(-1, 1, length.out = 20), z = seq(-1, 1, length.out = 20))
y <- 3 * x$x
explore <- function(engine = "native", categories = c(0L, 0L)) {
    .Call("wrap__r_preexplore", as.double(t(as.matrix(x))), y, categories,
          c(0L, 0L), engine, 7L, 60L, TRUE, PACKAGE = "mboostrs")
}
result <- jsonlite::fromJSON(explore())
stopifnot(result$engine == "native", result$fits == 1L,
          result$priority[1] == 0L, result$training_r2 > 0.99,
          identical(result$columns$shapes[[1]], c("bols", "bbs")))
again <- jsonlite::fromJSON(explore())
stopifnot(identical(result, again))
bad <- try(explore(categories = c(-1L, 0L)), silent = TRUE)
stopifnot(inherits(bad, "try-error"))
neural <- jsonlite::fromJSON(explore(engine = "neural"))
stopifnot(neural$engine == "neural", neural$fits == 1L,
          neural$neural_fit$encoded_inputs == 2L,
          identical(neural, jsonlite::fromJSON(explore(engine = "neural"))))
dual <- jsonlite::fromJSON(explore(engine = "dual"))
stopifnot(dual$engine == "dual", dual$fits == 2L,
          dual$shape_source == "native", !is.null(dual$neural))
message("pre-exploration R binding: OK")
