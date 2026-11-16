# Generics that mboost also defines.
#
# mboostrs and mboost each export `mstop`, `risk`, `varimp`, `nuisance`,
# `extract` and `cvrisk` as S3 generics, and a method is found only through
# the generic it was registered on. With both packages attached, whichever
# was attached last masks the other's generic, and every call on the other
# package's models failed with "no applicable method". Two things fix that:
# a default method on each mboostrs generic hands any object it has no
# method for to mboost's generic, and mboostrs registers its own methods on
# mboost's generics whenever mboost is loaded.
.SHARED_GENERICS <- c("mstop", "risk", "varimp", "nuisance", "extract", "cvrisk")

.upstream_call <- function(name, object, ...) {
  if (!requireNamespace("mboost", quietly = TRUE)) {
    stop(sprintf("no %s() method for an object of class %s", name,
                 paste(class(object), collapse = "/")), call. = FALSE)
  }
  getExportedValue("mboost", name)(object, ...)
}

#' @export
mstop.default <- function(object, ...) .upstream_call("mstop", object, ...)

#' @export
risk.default <- function(object, ...) .upstream_call("risk", object, ...)

#' @export
varimp.default <- function(object, ...) .upstream_call("varimp", object, ...)

#' @export
nuisance.default <- function(object) .upstream_call("nuisance", object)

#' @export
extract.default <- function(object, what, which = NULL, ...) {
  if (missing(what)) .upstream_call("extract", object, which = which, ...)
  else .upstream_call("extract", object, what = what, which = which, ...)
}

#' @export
cvrisk.default <- function(object, folds = NULL, fold_type = "subsampling", b = 25,
                           prob = 0.5, seed = 42L,
                           preprocessing = c("fixed", "refit"), ...) {
  if (match.arg(preprocessing) != "fixed") {
    stop("preprocessing = 'refit' requires an mboostrs glmboost or gamboost model", call. = FALSE)
  }
  # mboost's cvrisk() takes `folds` and its own arguments; this generic's
  # defaults describe mboostrs's fold construction and are not passed on.
  if (is.null(folds)) .upstream_call("cvrisk", object, ...)
  else .upstream_call("cvrisk", object, folds = folds, ...)
}

.register_on_mboost <- function(...) {
  ns <- asNamespace("mboost")
  own <- asNamespace("mboostrs")
  for (generic in .SHARED_GENERICS) {
    if (!exists(generic, envir = ns, inherits = FALSE)) next
    prefix <- paste0(generic, ".mboostrs_")
    for (method in ls(own, pattern = paste0("^", generic, "\\.mboostrs_"), all.names = TRUE)) {
      registerS3method(generic, sub(paste0("^", generic, "\\."), "", method),
                       get(method, envir = own), envir = ns)
    }
  }
  invisible(NULL)
}

.onLoad <- function(libname, pkgname) {
  if (isNamespaceLoaded("mboost")) .register_on_mboost()
  setHook(packageEvent("mboost", "onLoad"), .register_on_mboost)
  invisible(NULL)
}
