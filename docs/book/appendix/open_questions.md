# Open questions about the R reference implementations

This appendix records points that need further investigation and, where
appropriate, clarification with the authors of `mboost`, `gamboostLSS`,
`gamlss` and `gamlss.dist`. These packages provide the methodological and
implementation references for many of the comparisons in this book.

The observations below come from checks against specific versions. They are
not claims about the current releases, nor a list of author-confirmed defects.
Before contacting the authors, each example should be rerun in a clean R
session with the package versions recorded. Differences may reflect an
intended convention, an unsupported way of calling a function, or a mistake
in our interpretation. No author-confirmed resolution is recorded here.

## Scope of the recorded checks

The boosting questions were recorded against `mboost` 2.9.11; a separate
`HingeLoss` check used 2.9.13. Distributional checks used `gamboostLSS` 2.2.0,
`gamlss` 5.5.0, `gamlss.dist` 6.1.1 and R 4.5.1. These are historical
observations. The [capability matrix](capabilities.ipynb) identifies which
mboostrs operations have separate numerical validation.

## `mboost`: weighted AUC gradients

In the recorded 2.9.11 check, a call to `AUC()@ngradient(y, f, w)` with
40 observations and integer weights summing to 60 returned a vector of
length 60. The implementation used indices from the expanded response
`rep(y, w)` while the supplied predictor had one value per original row.

The remaining question is whether non-unit weights are intended to be
accepted by this gradient interface. If they are, should its result have
one element per original observation or per expanded observation? The
meaning of fractional weights also needs to be established before using
this interface as a reference for weighted cross-validation.

mboostrs uses a weighted pairwise formulation. That choice needs to remain
an explicit difference until the intended reference convention is confirmed.

## `mboost`: weighted Gehan gradients

The recorded source inspection found repeated-index assignment when the
response was expanded according to integer weights. R keeps the last value
assigned to a repeated index, as this small example illustrates:

```r
g <- numeric(4)
g[c(1, 1, 1, 2, 3, 4)] <- c(10, 20, 30, 40, 50, 60)
g  # 30 40 50 60
```

This assignment behavior alone does not establish the intended Gehan
weighting convention. The investigation should reconstruct the complete
weighted gradient, check how repeated contributions are combined, and
confirm whether bootstrap multiplicities are supported by the family.

## `mboost`: the Gehan offset

Gehan loss depends on pairwise differences of residuals. Adding the same
constant to all fitted values therefore leaves the loss unchanged, so this
criterion does not identify a constant offset.

The recorded reference implementation optimized that flat criterion;
mboostrs sets the offset to zero. Coefficient increments and risk can agree
while fitted values differ by a constant. Is zero an appropriate convention
for this unidentified level, or does another part of the reference API rely
on the optimizer's selected value? This needs clarification before treating
agreement of absolute fitted levels as a parity requirement.

## `mboost`: the `HingeLoss` starting value

A check against 2.9.13 found that the internal `HingeLoss()` family used the
same offset expression as `AdaExp()`. For an imbalanced binary response,
that log-odds starting value need not minimize constant-predictor hinge loss.
For the recorded five-positive, three-negative fixture, the reported risks
were 3.744587 at the offset and 3.000000 at a constant predictor of 1.

The question is whether this starting value is intentional and whether the
internal family is meant to be used directly. An offset is an initialization,
so a difference here does not by itself establish an error in a fitted model.
Its accessibility and practical effect should be checked before requesting
a change.

## `mboost`: the withdrawn spline learners

The 2.9.11 reference refused `bns` and `bss` and directed callers to `bbs`.
The recorded release note states that they were deprecated, without giving
the methodological reason.

What motivated that decision? In particular, is there a reason that also
applies to mboostrs's `Bns` variant? The latter uses an identity ridge penalty
and should not be interpreted as an unpenalized natural spline merely because
of its name. Clarification would help determine which comparisons and naming
conventions are appropriate.

## `gamboostLSS`: censored `LogLogLSS` scale risk

In the recorded 2.2.0 source inspection, the scale-parameter loss combined a
logistic density for uncensored observations with a normal survival function
for censored observations. The corresponding location loss used logistic
survival. The recorded closure can be inspected with:

```r
library(gamboostLSS)
get("loss", envir = environment(LogLogLSS()$sigma@risk))
```

For a log-logistic accelerated-failure-time model with standardized log time
$e=(\log t-\mu)/\sigma$, the survival function associated with the logistic
density is `plogis(e, lower.tail = FALSE)`. At $e=2$, this is about 0.119203,
whereas normal survival is about 0.022750.

Is there a parameter-specific convention that explains the normal survival
term, or should both losses use logistic survival? The risk, offset and
negative gradient should be checked separately: a discrepancy in one does
not establish that the others share it. Current source and censored fits
need rechecking before proposing a change.

## `gamboostLSS`: the Weibull location offset

The recorded `WeibullLSS` location predictor had an identity response map,
and its loss used $(\log t-\mu)/\sigma$. Its offset code included a search
interval from zero to the largest observed time. That interval appears to
be on a different scale from a log-time location, especially when every
observed time is below one.

What is the intended scale of the offset argument and of a supplied `mu`?
Which initialization branch is used when parameters are created through the
family closure? The previous investigation did not resolve those questions.
A clean-session check should distinguish closure initialization from the
optimizer itself; an offset returned outside an inspected interval can
indicate that a different branch ran.

## `gamlss.dist`: scalar parameters in vector calculations

Two recorded examples raise questions about the accepted parameter shapes:

- `BCCG()` derivative closures produced different results for scalar `nu`
  and for `rep(nu, length(y))`. The inspected expression used `ifelse` with
  `nu != 0` as its condition; its result length follows that condition.
- `dNBI(0, mu = c(1, 2, 3), sigma = 1)` returned length one in the recorded
  version. A zero-inflation calculation may instead need one zero-mass
  probability for each value of `mu`.

Are scalar parameters intended to recycle in these interfaces, or must
callers supply aligned vectors? Are the direct density functions and the
internal derivative closures subject to different shape requirements?
Those conventions should be established before interpreting a difference
as a numerical error.

With aligned parameter vectors, the recorded BCCG derivatives agreed with
finite differences; an earlier suspicion of a general derivative error was
withdrawn. The remaining boundary question is whether the `nu = 0` score
should be evaluated by its finite limit or whether callers are expected to
avoid direct evaluation at that point.

## Investigation and confirmation

For each question, a follow-up should retain the package versions, a minimal
reproducer, the documented input convention, and the numerical consequence
for fitting or inference. Proposed explanations should be tested against
independent calculations where possible. Any author clarification should be
recorded with its source and scope; a question should be closed when that
evidence resolves it, rather than when one implementation happens to agree
with another.
