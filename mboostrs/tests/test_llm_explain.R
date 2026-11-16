#@validates r.method.contract
library(mboostrs)

set.seed(7)
n <- 200
dat <- data.frame(
  x1 = runif(n, 0, 10),
  x2 = runif(n, 0, 1),
  unused = rnorm(n),
  g = factor(sample(c("low", "mid", "high"), n, replace = TRUE))
)
dat$y <- sin(dat$x1) + 2 * dat$x2 + (dat$g == "high") + rnorm(n, 0, 0.3)

gam <- gamboost(y ~ bbs(x1, df = 4) + bols(x2) + bbs(unused, df = 4) + brandom(g),
                data = dat, control = boost_control(mstop = 120, nu = 0.1))
glm <- glmboost(y ~ x1 + x2 + unused, data = dat,
                control = boost_control(mstop = 100, nu = 0.1))

# ── every section is written, and `include` selects among them ──────

text <- llm_explain_prompt(gam)
stopifnot(is.character(text), length(text) == 1)
for (marker in c("How to read what follows", "boosting results", "Response scale:",
                 "Base learners, ranked:", "Effects (", "Limits of this description",
                 "Write the brief a statistician")) {
  stopifnot(grepl(marker, text, fixed = TRUE))
}

effects_only <- llm_explain_prompt(gam, include = c("effects"))
stopifnot(startsWith(effects_only, "Effects ("))
stopifnot(!grepl("How to read", effects_only, fixed = TRUE))

# An unknown section is refused rather than silently dropped.
stopifnot(inherits(try(llm_explain_prompt(gam, include = c("effects", "nope")),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(llm_explain_prompt(gam, which = "not_a_column"),
                       silent = TRUE), "try-error"))

# ── the effect blocks ──────────────────────────────────────────────

stopifnot(grepl("bbs: B-spline basis of degree 3 on 20 inner knots, df=4", text,
                fixed = TRUE))
stopifnot(grepl("second-difference penalty", text, fixed = TRUE))
stopifnot(grepl("Covariate: x1 (observed range", text, fixed = TRUE))
stopifnot(grepl("Shape:", text, fixed = TRUE))
# No padded numbers inside the serialized values.
stopifnot(!grepl('"[[:space:]]', text))
stopifnot(!grepl("df=[[:space:]]", text))

# A grid of six points gives six pairs.
six <- llm_explain_prompt(gam, include = c("effects"), n_points = 6, which = "x1")
pairs_line <- grep("^Contribution: \\{", strsplit(six, "\n")[[1]], value = TRUE)
stopifnot(length(pairs_line) == 1)
stopifnot(lengths(regmatches(pairs_line, gregexpr('": ', pairs_line))) == 6)

# A learner the path never chose is named as contributing exactly zero.
ranked <- llm_explain_prompt(gam, include = c("importance"))
stopifnot(grepl("unused", ranked, fixed = TRUE))

# A factor learner reports its level labels, and claims no shape for them.
stopifnot(grepl("Covariate: g (categorical, 3 levels)", text, fixed = TRUE))
for (level in c("low", "mid", "high")) {
  stopifnot(grepl(paste0('"', level, '"'), text, fixed = TRUE))
}

# ── glmboost reports a coefficient per term ────────────────────────

glm_text <- llm_explain_prompt(glm, include = c("effects"), top = 1)
stopifnot(grepl("[linear, unpenalized]", glm_text, fixed = TRUE))
stopifnot(grepl("Coefficient:", glm_text, fixed = TRUE))
stopifnot(grepl("contribution = ", glm_text, fixed = TRUE))

# ── importance is per learner, and sums to the whole path ──────────

# Two learners on one column share a name; the ranked list must tell them apart.
twin <- gamboost(y ~ bbs(x1, df = 4) + bmono(x1, df = 4), data = dat,
                 control = boost_control(mstop = 80, nu = 0.1))
twin_text <- llm_explain_prompt(twin, include = c("importance"))
stopifnot(grepl("x1 (bbs)", twin_text, fixed = TRUE))
stopifnot(grepl("x1 (bmono)", twin_text, fixed = TRUE))

shares <- mboostrs:::.llm_importance(gam)
stopifnot(abs(sum(abs(shares)) - 1) < 1e-12)

# ── context the caller supplies, and the message shapes ───────────

with_context <- llm_explain_prompt(
  gam,
  dataset_description = "Simulated data, n = 200.",
  response_description = "y is a lab measurement in mg/dL.",
  task = "List the three most important covariates."
)
stopifnot(grepl("Simulated data, n = 200.", with_context, fixed = TRUE))
stopifnot(grepl("y is a lab measurement in mg/dL.", with_context, fixed = TRUE))
stopifnot(endsWith(trimws(with_context), "List the three most important covariates."))

messages <- llm_explain_prompt(gam, output = "messages")
stopifnot(length(messages) == 2)
stopifnot(vapply(messages, function(m) m$role, character(1)) == c("system", "user"))
stopifnot(grepl("expert statistician", messages[[1]]$content, fixed = TRUE))
stopifnot(identical(messages[[2]]$content, text))

plan <- llm_explain_prompt(gam, output = "turns", num_sentences = 4)
stopifnot(identical(sort(names(plan)), c("system", "turns")))
stopifnot(length(plan$turns) == 3)
stopifnot(identical(plan$turns[1], text))
stopifnot(grepl("surprising or", plan$turns[2], fixed = TRUE))
stopifnot(grepl("at most 4 sentences", plan$turns[3], fixed = TRUE))

# ── llm_describe sends nothing without a key ──────────────────────

old <- Sys.getenv(c("MBOOSTRS_LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"))
Sys.unsetenv(c("MBOOSTRS_LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"))
failure <- try(llm_describe(gam), silent = TRUE)
stopifnot(inherits(failure, "try-error"))
stopifnot(grepl("needs an API key", conditionMessage(attr(failure, "condition")),
                fixed = TRUE))
for (name in names(old)) {
  if (nzchar(old[[name]])) do.call(Sys.setenv, setNames(list(old[[name]]), name))
}

# ── partial_effects: a filtered call is the same curve as the whole ─

all_effects <- partial_effects(gam, n_points = 20)
one_effect <- partial_effects(gam, which = "x2", n_points = 20)
stopifnot(isTRUE(all.equal(one_effect[["x2"]], all_effects[["x2"]])))
by_index <- partial_effects(gam, which = 2L, n_points = 20)
stopifnot(isTRUE(all.equal(by_index[["x2"]], all_effects[["x2"]])))

# ── the shrinkage floor ────────────────────────────────────────────

# A second-difference penalty leaves a 2-dimensional null space, and df(lambda)
# falls to 2 and stops. At df=2 the learner can only fit straight lines, so
# "monotone increasing" alone would read as a shape the data chose.
floor_fit <- suppressWarnings(gamboost(
  y ~ bbs(x1, df = 2), data = dat,
  control = boost_control(mstop = 60, nu = 0.1)
))
floor_text <- llm_explain_prompt(floor_fit, include = c("effects"), n_points = 5)
stopifnot(grepl("At the shrinkage floor: df=2", floor_text, fixed = TRUE))
stopifnot(grepl("can only represent straight lines", floor_text, fixed = TRUE))

# center = TRUE projects the null space out, so there is no floor to report.
centered_fit <- gamboost(y ~ bbs(x1, df = 2, center = TRUE), data = dat,
                         control = boost_control(mstop = 60, nu = 0.1))
centered_text <- llm_explain_prompt(centered_fit, include = c("effects"),
                                    n_points = 5)
stopifnot(!grepl("shrinkage floor", centered_text, fixed = TRUE))

cat("test_llm_explain.R: all assertions passed\n")
