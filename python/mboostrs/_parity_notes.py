"""Where an mboostrs object differs from the thing its name refers to.

A user who reaches for `Gehan` or `Bkernel` or an AFT `logLik` has a meaning in
mind, taken from `mboost`, from `gamboostLSS`, or from the statistical
literature. Sometimes mboostrs computes something else. Sometimes `mboost`
computes something else too, and mboostrs reproduces it for parity, so the
object differs from the *name* rather than from upstream.

Both kinds mislead the same way, and both were recorded only in a generated
appendix a user would have to think to look for. A note here is attached to
the family or learner that carries it, so `model.parity_notes()` surfaces it
for the fit in hand.

Each note says four things, because three of them are not enough:

`name`
    A short handle, so the deviation can be named in a conversation or a
    changelog rather than re-described.

`mechanism`
    What the code actually does. Not "differs from R" -- *how*.

`consequence`
    What breaks for the reader who assumed otherwise. A deviation with no
    consequence does not need a note; one whose consequence is unstated leaves
    the reader to work out whether it matters.

`shared_with_upstream`
    Whether `mboost`/`gamboostLSS` does the same. This is the distinction that
    decides what a user should do: a deviation mboostrs invented might be a
    bug to report, while one shared with upstream is a property of the method
    as implemented everywhere, and switching packages will not escape it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParityNote:
    """One documented difference between an object and its name."""

    name: str
    mechanism: str
    consequence: str
    #: True when `mboost` or `gamboostLSS` behaves the same way, so the
    #: difference is between the implementation and the name rather than
    #: between mboostrs and upstream.
    shared_with_upstream: bool
    #: The test that holds the behaviour in place, so a reader can see what is
    #: actually checked rather than trusting the prose.
    evidence: str = ""
    #: One sentence on what the deviation changes in the outputs of a fit that
    #: carries it. When set, the summary prints this sentence in place of the
    #: generic pointer to `parity_notes()`.
    effect: str = ""

    def format_text(self) -> str:
        origin = ("shared with upstream" if self.shared_with_upstream
                  else "mboostrs-specific")
        lines = [f"{self.name} ({origin})",
                 f"  what it does: {self.mechanism}",
                 f"  what follows: {self.consequence}"]
        if self.evidence:
            lines.append(f"  checked by:   {self.evidence}")
        return "\n".join(lines)


#: Notes attached to a family, keyed by the Python class name.
FAMILY_NOTES: dict[str, tuple[ParityNote, ...]] = {
    "Multinomial": (
        ParityNote(
            name="multinomial risk excludes the reference class",
            mechanism=(
                "the loss is summed over the K-1 non-reference columns, where "
                "an observation in the reference class is zero in every one, "
                "so it contributes exactly nothing. Measured: the risk equals "
                "the categorical negative log-likelihood over non-reference "
                "rows alone, to 3e-14"
            ),
            consequence=(
                "the reported risk, the risk path, and any information "
                "criterion are all missing a term that moves with the fit. "
                "aic() and logLik() refuse; negative_risk() exposes the "
                "partial risk as the mboost-comparable reported quantity. "
                "cvrisk does not use it: see the next note"
            ),
            shared_with_upstream=True,
            evidence="tests/test_likelihood_contract.py::test_multinomial_risk_omits_the_reference_class",
        ),
        ParityNote(
            name="multinomial negative gradient is the derivative of the full nll",
            mechanism=(
                "the fit descends the complete categorical NLL: the negative "
                "gradient `y_k - p_k` is the derivative of the full "
                "cross-entropy including the reference-class term "
                "`-log(p_reference)`, so reference-class observations "
                "contribute to the gradient even though they contribute zero "
                "to the reported risk. Measured by central finite differences "
                "of the full NLL, agreeing to 1e-6 on rows from every class"
            ),
            consequence=(
                "the fitted coefficient path follows a different objective "
                "than the reported risk path. The deviation is intentional "
                "(upstream parity: mboost::Multinomial reports the same "
                "partial loss), so both are kept and documented as distinct, "
                "and every downstream selector refuses rather than mix them"
            ),
            shared_with_upstream=True,
            evidence="tests/test_likelihood_contract.py::test_multinomial_negative_gradient_is_the_derivative_of_the_full_nll",
        ),
        ParityNote(
            name="multinomial cvrisk scores the full negative log-likelihood",
            mechanism=(
                "held-out rows are scored with the complete categorical "
                "negative log-likelihood, reference class included "
                "(Family::held_out_risk); mboost's cvrisk sums its partial "
                "loss, which omits the reference class"
            ),
            consequence=(
                "the chosen mstop differs from mboost's, and later: on a "
                "three-class simulation mboost's criterion chose 257 steps "
                "where held-out log-loss was lowest at about 1,000 (test "
                "log-loss 1.036 against 1.026). The in-bag risk path still "
                "matches mboost"
            ),
            shared_with_upstream=False,
            evidence="tests/test_multinomial_parity/test_multinomial_basic.py::test_multinomial_cvrisk_matches_r",
        ),
        ParityNote(
            name="multinomial is gamboost-only",
            mechanism=(
                "the Kronecker-structured design the family needs is built on "
                "the gamboost path; glmboost rejects any response outside "
                "{0, 1} whatever n_classes says"
            ),
            consequence=(
                "a linear multi-class fit is written as "
                "gamboost(\"y ~ bols(x)\"), whose learners the Kronecker lift "
                "wraps; cvrisk and stability selection then run on that fit"
            ),
            shared_with_upstream=False,
        ),
    ),
    "Gehan": (
        ParityNote(
            name="gehan offset is zero rather than arbitrary",
            mechanism=(
                "the Gehan loss depends only on pairwise residual differences, "
                "so a constant offset cancels and the risk is flat in it -- "
                "the offset is unidentifiable. mboostrs defines it as 0; "
                "mboost runs optimize() on that flat function and keeps "
                "whatever it lands on"
            ),
            consequence=(
                "fitted values differ from R by an additive constant. "
                "Gradients, coefficient increments, selected() and risk all "
                "match exactly, so anything that depends on differences or on "
                "the path is unaffected; anything that reads a fitted level "
                "is not"
            ),
            shared_with_upstream=False,
        ),
        ParityNote(
            name="gehan is a ranking loss, not a likelihood",
            mechanism="the loss scores pairwise orderings, not a density",
            consequence=(
                "logLik() and aic() refuse. Use Weibull, Loglog or Lognormal "
                "for an AFT model with a likelihood, or CoxPH for a partial "
                "one"
            ),
            shared_with_upstream=True,
        ),
    ),
    "Cindex": (
        ParityNote(
            name="cindex is a ranking loss, not a likelihood",
            mechanism=(
                "the smoothed concordance loss depends only on the order of "
                "the fitted values"
            ),
            consequence="logLik() and aic() refuse; compare fits by cvrisk",
            shared_with_upstream=True,
        ),
    ),
    "AUC": (
        ParityNote(
            name="auc is a ranking loss, not a likelihood",
            mechanism="the smoothed 1 - AUC loss depends only on the ordering",
            consequence="logLik() and aic() refuse; compare fits by cvrisk",
            shared_with_upstream=True,
        ),
        ParityNote(
            name="weighted auc is an mboostrs extension",
            mechanism=(
                "each observation enters the pairwise sums with weight w_i "
                "(w_i == 0 drops out), the f/sd(f) standardisation is a "
                "weighted sample sd over the participating observations, and "
                "every pair picks up a w_i*w_k factor. At w == 1 this reduces "
                "bit-identically to the unit-weight formulas"
            ),
            consequence=(
                "a weighted AUC fit runs, and is not R's. R's own weighted "
                "AUC() gradient returns a compacted vector of length sum(w) "
                "rather than n, which is an upstream defect rather than a "
                "definition to match, so there is no R number to compare a "
                "weighted fit against. cvrisk() and stabsel() therefore work "
                "here for both bootstrap and subsampling folds, checked for "
                "finiteness and for selecting signal over noise rather than "
                "against R"
            ),
            shared_with_upstream=False,
            evidence=("tests/test_auc_family.py::"
                      "test_glmboost_auc_accepts_non_unit_weights_and_is_finite"),
        ),
    ),
    "Weibull": (
        ParityNote(
            name="aft likelihood is on the log-time scale",
            mechanism=(
                "the risk is the negative log-likelihood of log(T); the 1/t "
                "Jacobian that would put it on the time scale is absent "
                "(measured against an independently computed reference)"
            ),
            consequence=(
                "logLik() is a proper log-likelihood, but of a different "
                "quantity than a Gaussian fit of the same time column, so "
                "gaic_table() refuses to rank the two. exp(eta) is the fitted "
                "distribution's scale, not its mean"
            ),
            shared_with_upstream=True,
            evidence="tests/test_likelihood_contract.py::test_aft_risk_is_the_nll_of_log_time_not_of_time",
        ),
    ),
    "CoxPH": (
        ParityNote(
            name="cox reports a partial likelihood",
            mechanism=(
                "the risk is the Breslow negative log partial likelihood; the "
                "baseline hazard is profiled out, not estimated"
            ),
            consequence=(
                "logLik() ranks Cox models against each other and nothing "
                "else -- it is not a density of the data, so an AIC against "
                "any density-based fit is not a likelihood ratio. Response "
                "residuals are refused: exp(eta) is a relative risk, not a "
                "time"
            ),
            shared_with_upstream=True,
            evidence="tests/test_likelihood_contract.py::test_coxph_risk_is_the_breslow_partial_likelihood",
        ),
    ),
    "GammaReg": (
        ParityNote(
            name="gamma risk omits the log y term",
            mechanism=(
                "the risk is R's plloss with the shape re-estimated at every "
                "step (nuisance() reports it); it is the Gamma negative "
                "log-likelihood minus the sum of w log y"
            ),
            consequence=(
                "negative_risk() differs from the log-likelihood by that "
                "constant; logLik() adds it back. The shape is one value for "
                "all rows: use GammaLSS when it should depend on predictors"
            ),
            shared_with_upstream=True,
            evidence="tests/test_likelihood_contract.py::test_gammareg_loglik_is_the_gamma_density_at_the_estimated_shape",
        ),
    ),
    "Binomial": (
        ParityNote(
            name="binomial gradient carries a log(2) factor",
            mechanism=(
                "mboost defines the Binomial negative gradient with a "
                "1/log(2) factor its loss does not carry, so the working "
                "residual is about 1.44 times the 2(y - p) the risk implies"
            ),
            consequence=(
                "nu on a Binomial fit is effectively divided by log(2) "
                "relative to the natural-log gradient, so the same nu takes a "
                "larger step here than on a Gaussian fit"
            ),
            shared_with_upstream=True,
            effect=(
                "the boosting steps are about 1.44 times (1/log 2) those of a "
                "natural-log gradient at the same nu, so a fixed mstop fits "
                "further than a natural-log fit would; the reported risk is "
                "the natural-log negative log-likelihood, unscaled, as is "
                "logLik()"
            ),
        ),
    ),
    "Huber": (
        ParityNote(
            name="adaptive huber risk is not comparable across iterations",
            mechanism=(
                "with no delta given, the changeover point is re-estimated "
                "from the current residuals at every iteration, so the loss "
                "being minimised changes as the path proceeds and the risk at "
                "step k is measured under a different function than at k-1"
            ),
            consequence=(
                "the in-bag risk path can rise -- 4 of 25 steps on a plain "
                "linear fixture -- and a minimum in it is not necessarily a "
                "better fit. Pass an explicit delta for a monotone path, or "
                "choose mstop by cvrisk, whose folds are scored on held-out "
                "rows under whichever delta the fit reached"
            ),
            shared_with_upstream=True,
            evidence=("tests/test_family_contract.py::"
                      "test_adaptive_huber_risk_is_not_monotone"),
        ),
        ParityNote(
            name="adaptive huber cvrisk starts each fold afresh",
            mechanism=(
                "each cross-validation fold is fitted with its own family "
                "state, so its first delta comes from that fold's own offset. "
                "mboost keeps the last fitted predictor in the family closure "
                "and reuses one family object for every fold, so each fold's "
                "first delta comes from the previous fold's final fit"
            ),
            consequence=(
                "mboost's adaptive-Huber cvrisk depends on the fold order and "
                "differs from mboostrs by about 1% of the held-out risk; "
                "mboostrs equals mboost refitted fold by fold with a fresh "
                "Huber() (to 3e-14). With an explicit delta the two agree"
            ),
            shared_with_upstream=False,
            evidence=("tests/test_huber_cv_fold_state.py::"
                      "test_matches_mboost_refitted_fold_by_fold"),
        ),
    ),
    "PropOdds": (
        ParityNote(
            name="ordinal response is 1..K",
            mechanism=(
                "the engine validates the response against {1, ..., K}, "
                "matching R's 1-based factor levels"
            ),
            consequence=(
                "a zero-based column is refused; shift it by one before "
                "fitting. n_levels is required rather than inferred"
            ),
            shared_with_upstream=True,
        ),
    ),
    "Gaussian": (
        ParityNote(
            name="gaussian logLik is a real log-likelihood",
            mechanism=(
                "the Gaussian risk is the residual sum of squares, which is "
                "not the negative log-likelihood; logLik() reconstructs one by "
                "plugging in sigma^2 = RSS/n. mboost returns -risk instead"
            ),
            consequence=(
                "logLik() is smaller than R's for the same fit by exactly the "
                "constants the risk drops, and is not comparable with a value "
                "quoted from mboost. negative_risk() is R's quantity"
            ),
            shared_with_upstream=False,
            evidence="tests/test_likelihood_contract.py::test_gaussian_risk_is_rss_and_loglik_reconstructs_the_rest",
        ),
    ),
}

#: Notes attached to a learner, keyed by the Python class name.
LEARNER_NOTES: dict[str, tuple[ParityNote, ...]] = {
    "Bkernel": (
        ParityNote(
            name="bkernel is a kernel-ridge learner",
            mechanism=(
                "it builds an explicit kernel design against the observed x "
                "values and fits it with a ridge penalty"
            ),
            consequence=(
                "it is not mboost's bkernel, which is a pathway/GWAS "
                "construction over marker sets. A formula written for one "
                "fits a different model under the other, with no error"
            ),
            shared_with_upstream=False,
        ),
    ),
    "Brad": (
        ParityNote(
            name="brad centre choice differs",
            mechanism=(
                "centres are chosen by a farthest-point rule over the "
                "observed coordinates"
            ),
            consequence=(
                "the basis differs from mboost's for the same data, so "
                "coefficients are not comparable term by term even when the "
                "fitted surfaces are close"
            ),
            shared_with_upstream=False,
        ),
    ),
    "Btree": (
        ParityNote(
            name="btree splits on squared error, not conditional inference",
            mechanism=(
                "splits are chosen to minimise the within-node sum of squares"
            ),
            consequence=(
                "mboost's btree uses a conditional-inference criterion, which "
                "selects different splits and has different variable-selection "
                "bias. blackboost built on this is not mboost's blackboost"
            ),
            shared_with_upstream=False,
        ),
    ),
    "Bns": (
        ParityNote(
            name="bns is ridge-penalized, not unpenalized",
            mechanism=(
                "the B-spline basis is fitted with an identity (ridge) "
                "penalty rather than left unpenalized"
            ),
            consequence=(
                "the fit is smoother than an unpenalized natural spline of "
                "the same basis, and df counts the penalized trace rather "
                "than the basis dimension"
            ),
            shared_with_upstream=False,
        ),
    ),
}

#: Notes that describe the package rather than one object, surfaced by
#: `parity_notes(include_global=True)`.
GLOBAL_NOTES: tuple[ParityNote, ...] = (
    ParityNote(
        name="continue_to refits from zero on routes without a retained residual",
        mechanism=(
            "a gamboost fit resumes the retained gradient when its route kept "
            "one: every family whose calibration is a pure function of "
            "(y, f, w), which excludes PropOdds, Multinomial and the survival "
            "families. The Gaussian Gram route keeps per-learner projections "
            "instead of a length-n residual, and binned learners, glmboost "
            "and the distributional models have no resume path, so those grow "
            "by refitting from iteration zero"
        ),
        consequence=(
            "the grown model equals a cold fit to machine precision either "
            "way, so only the cost differs: a resumed continuation does not "
            "recompute the retained iterations, a refitting one does"
        ),
        shared_with_upstream=False,
    ),
)


#: Attached to a distributional fit that divided its response by a scale.
LSS_RESPONSE_SCALE_NOTE = ParityNote(
    name="distributional fits are run on the response divided by a scale",
    mechanism=(
        "gamboostLSS boosts on the response as given: once gradients are "
        "stabilized each step moves a location parameter by about nu in "
        "response units, and without stabilization by nu (y - mu)/sigma^2, so "
        "the fit depends on the units and a response in large units (dollars) "
        "barely leaves its offset. With response_scale='auto' (the default) "
        "mboostrs fits y / s, s the weighted SD of the response each fit sees, "
        "and maps every parameter's predictor back exactly "
        "(eta = a * eta_s + b: a = s for an identity-link location, "
        "b = log(s) for a log-link mean or scale); counts, proportions and "
        "compositions are not scaled"
    ),
    consequence=(
        "predictions, coefficients, risk and logLik are in the original units "
        "and the same model results in any units, but the boosting path at a "
        "given mstop is R's path on y / s, not on y: coefficients at an mstop "
        "tuned for gamboostLSS on raw data differ. response_scale=None (or "
        "'none') reproduces gamboostLSS's raw-units path"
    ),
    shared_with_upstream=False,
    evidence="tests/test_lss_response_scale.py::test_a_fit_in_other_units_is_the_same_model_in_those_units",
    effect=(
        "this fit ran on the response divided by its SD and was mapped back to "
        "the original units; response_scale=None gives gamboostLSS's raw-units path"
    ),
)


def notes_for(model, *, include_global: bool = False) -> list[ParityNote]:
    """Every documented deviation that applies to this fit.

    Ordered family first, then learners in formula order, then the global
    notes when asked for -- so the ones specific to what the caller fitted come
    before the ones about the package.
    """
    collected: list[ParityNote] = []

    family = getattr(model, "_family", None)
    if family is not None:
        collected.extend(FAMILY_NOTES.get(type(family).__name__, ()))

    seen: set[str] = set()
    learners = list(getattr(model, "_learners_spec", None) or [])
    for parameter in getattr(model, "_params", None) or []:
        learners.extend(parameter.get("learners", []) or [])
    for learner in learners:
        kind = type(learner).__name__
        if kind in seen:
            continue
        seen.add(kind)
        collected.extend(LEARNER_NOTES.get(kind, ()))

    scale = getattr(model, "_response_scale", None)
    if scale is not None and scale != "none" and \
            getattr(model, "_response_scale_used", None) is not None:
        collected.append(LSS_RESPONSE_SCALE_NOTE)

    if include_global:
        collected.extend(GLOBAL_NOTES)
    return collected


def format_notes(notes) -> str:
    """The notes as text, or a line saying there are none."""
    notes = list(notes)
    if not notes:
        return (
            "No documented parity deviations apply to this fit: its family and "
            "learners compute what their names refer to."
        )
    header = (
        f"{len(notes)} documented deviation(s) between what this fit computes "
        f"and what its names refer to:"
    )
    return "\n\n".join([header, *(note.format_text() for note in notes)])
