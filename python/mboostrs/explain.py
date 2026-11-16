"""Model descriptions written to be read by a language model.

:func:`llm_explain_prompt` renders a fitted model as one prompt: how the fit was
produced, which base learners it kept, how much of the risk each one removed,
the shape of every effect it estimated, and which questions the fit cannot
answer. It sends nothing anywhere — it returns the text (or a list of chat
messages) for the caller to pass to whatever model they use.
:func:`llm_describe` is the one function here that does make a request: it sends
that prompt to an OpenAI-compatible chat endpoint and returns the reply.

The numbers all come from methods the model already exposes: ``summary``,
``varimp``, ``partial_effects``, ``coef``. What this module adds is a serialized
form that survives the trip through a language model: an effect becomes a short
table of covariate value to contribution, rounded to the precision the effect's
own variation supports, with the covariate's observed range, the learner's
mechanics, and a one-line statement of the shape beside it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

__all__ = ["llm_describe", "llm_explain_prompt"]

# The sections of the prompt, in the order they are written. `include=` accepts
# any subset of these names.
SECTIONS = ("reading", "facts", "importance", "effects", "caveats", "task")

_DEFAULT_TASK = (
    "Write the brief a statistician would hand a colleague who has already "
    "read the tables above. Structure the reply in four labelled parts. "
    "**Findings**: the two or three results that matter most, each with the "
    "numbers that establish it, in the units named above. **Surprising "
    "patterns**: what runs against expectation for this subject matter, with "
    "the numbers that make it notable. **What could mislead**: features of "
    "the fit that could turn a finding into an artefact — a sparse or "
    "extrapolated covariate range, two learners carrying the same signal, an "
    "effect standing in for an omitted variable, a stopping iteration that "
    "may be too early or too late. **Next check**: the one analysis or datum "
    "that would most change your reading. Keep the whole reply under 300 "
    "words. A reply that walks the effects one by one, or that rebuilds a "
    "table the description already contains, fails this task however "
    "accurate it is; cite a number only in support of a point. Describe "
    "associations: this is a predictive fit on observational data unless the "
    "notes above say otherwise, so write what moves with what, not what "
    "causes what. Where the fit does not support a claim, say so instead of "
    "qualifying it. Markdown with bold labels and bullets; no code fence."
)

_SYSTEM_MESSAGE = (
    "You are {expert}. You read the global structure of a fitted component-wise "
    "boosting model — an additive model whose terms were selected one at a time "
    "— and you answer from the numbers you are given, the context supplied with "
    "them, and your knowledge of the subject matter."
)

#: The second and third turns of the reasoning sequence `output="turns"` builds.
#: The first turn is the findings reply itself. Splitting the work this way —
#: report findings, then hunt for what does not fit, then compress — is the structure
#: `t2ebm` uses for EBM graphs, and it exists because a model asked for a short
#: summary first tends to produce a fluent one that skips the anomalies.
_PROBE_SURPRISES = (
    "Now examine the patterns you reported and the fit behind them, and "
    "surface what a first reading misses: regions surprising or "
    "counterintuitive for this subject matter, effects that look like data "
    "artefacts rather than mechanisms, and importance shares or update "
    "counts that could mislead — credit split between correlated covariates, "
    "many updates buying little risk reduction, a learner absorbing the "
    "signal of a variable absent from the model. For each item say which "
    "numbers make it notable, which explanations could produce it — a real "
    "mechanism, a data artefact such as a sparse covariate range, a "
    "covariate standing in for another, confounding by something not in the "
    "model, or the path having stopped where it did — which one you consider "
    "most likely on subject-matter grounds, and what observation, analysis "
    "or experiment would settle it. The fit alone cannot settle it, so "
    "commit to a judgment instead of listing possibilities."
)

_PROBE_CONDENSE = (
    "Now write the brief for a reader who has the charts and tables in hand: "
    "at most {num_sentences} sentences. Open with the finding that matters "
    "most, give the other findings worth acting on, name the issues that "
    "could make them artefacts, and end with the single next check to run. "
    "A sentence that only reports what a table already shows is a wasted "
    "sentence: spend none on the method and none on walking the effects one "
    "by one. Write it in Markdown with bold labels or bullets where they "
    "help. Do not wrap it in a code fence."
)

_READING = """\
How to read what follows.

The fit is an additive model. The prediction for one observation is an offset
plus one effect per base learner, summed on the link scale named below — one
such scale per distribution parameter, where the fit has several. Component-wise
boosting built it by repeating one step: fit every
candidate base learner to the current negative gradient of the loss, keep the
single learner that reduces the loss most, and add `nu` times that fit to the
model. A learner picked in many steps accumulated a large effect. A learner
never picked contributes exactly zero — it is absent from the model, not merely
small.

Effects are **not** all centered on zero, so do not read one effect's average
level as meaningful on its own. Where an effect sits vertically depends on its
learner: a `bbs(..., center=True)` term has its unpenalized part removed and
does sit near zero, while a `bols` term carries its own intercept and a plain
`bbs` keeps the level it accumulated. What is always meaningful is the
*difference* between two points within one effect: that is the change in the
linear predictor between those covariate values, and it does not depend on
where the effect sits. Compare within an effect, not across effects and not
against zero.

An effect is printed as `covariate value: contribution` pairs; a key in
parentheses is an interval, and a quoted key is a category label."""

_CAVEATS = """\
Limits of this description.

- The fit reports no standard errors, confidence intervals or p-values. Each
  base learner was selected and estimated on the same data, so the usual
  interval formulas do not apply to it. Treat every number here as a point
  estimate with unquantified uncertainty.
- Importance is the share of in-bag risk reduction credited to a learner over
  the path. Two correlated covariates split that credit, so a small share can
  mean redundancy rather than irrelevance, and a learner's share depends on
  which other learners were offered.
- Effect shapes hold over the covariate range printed with them. Outside it the
  learner extrapolates by construction, not from data.
- Everything here is conditional on the stopping iteration `mstop`. The same
  data stopped at a different iteration gives different effects, and a longer
  path gives larger ones.
- These are associations, not causal effects. The fit minimises a loss over
  the covariates it was offered; nothing in it identifies what would happen if
  one of them were changed. A covariate can carry a large effect because it
  causes the response, because it shares a cause with the response, because it
  stands in for something absent from the model, or because of how the data
  were collected, and the numbers here do not distinguish those. Say "moves
  with" rather than "drives" or "causes" unless the notes accompanying this
  description state a design that identifies an effect."""


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _num(value, digits=4):
    """A short decimal form of one number, without exponent noise."""
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    text = f"{value:.{digits}g}"
    return "0" if text in ("-0", "-0.0") else text


def _effect_precision(values):
    """Decimals that keep about three significant digits of an effect's span.

    An effect that moves by 0.004 needs more decimals than one that moves by
    40, and rounding both to a fixed width either loses the first or pads the
    second with noise. This is the rule `t2ebm` uses for the same reason.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 3
    span = float(values.max() - values.min())
    if span <= 0:
        return 3
    precision, fraction = 1, span / 100.0
    while fraction < 1 and precision < 8:
        fraction *= 10.0
        precision += 1
    return precision


def _pct(share):
    """A percentage that does not round a small nonzero share away to nothing."""
    if share == 0:
        return "0.0%"
    if abs(share) < 0.001:
        return "<0.1%" if share > 0 else ">-0.1%"
    return f"{share:.1%}"


def _plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _pairs_block(keys, values, precision):
    """`{"key": value, ...}` on one line, as a language model reads best."""
    body = ", ".join(
        f'"{key}": {round(float(value), precision)}'
        for key, value in zip(keys, values, strict=True)
    )
    return "{" + body + "}"


# ---------------------------------------------------------------------------
# model facts
# ---------------------------------------------------------------------------

#: What one unit of the additive predictor means for each family, and what the
#: response scale is. Only families whose inverse link is defined in
#: `families._inverse_link` make a claim here; anything else says so.
_RESPONSE_SCALE = {
    "Gaussian": ("the response itself (identity link)",
                 "A contribution of +1 adds 1 to the predicted mean."),
    "GaussReg": ("the response itself (identity link)",
                 "A contribution of +1 adds 1 to the predicted mean."),
    "Laplace": ("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted median."),
    "Huber": ("the response itself (identity link)",
              "A contribution of +1 adds 1 to the predicted centre."),
    "Quantile": ("the response itself (identity link)",
                 "A contribution of +1 adds 1 to the predicted quantile."),
    "Expectile": ("the response itself (identity link)",
                  "A contribution of +1 adds 1 to the predicted expectile."),
    "Binomial": ("half the log-odds of the positive class",
                 "The response is 1/(1 + exp(-2*eta)), so a contribution of "
                 "+1 multiplies the odds by exp(2) = 7.39."),
    "AdaExp": ("half the log-odds of the positive class",
               "The response is 1/(1 + exp(-2*eta))."),
    "Poisson": ("the log of the expected count",
                "A contribution of +1 multiplies the expected count by "
                "exp(1) = 2.72."),
    "NBinomial": ("the log of the expected count",
                  "A contribution of +1 multiplies the expected count by 2.72."),
    "GammaReg": ("the log of the expected response",
                 "A contribution of +1 multiplies the expected response by 2.72."),
    "CoxPH": ("the log relative risk",
              "A contribution of +1 multiplies the hazard by exp(1) = 2.72; the "
              "baseline hazard is not part of the additive predictor."),
    # The AFT predictor is a *location* on the log-time scale, not the log of
    # the expected time. `exp(eta)` is the scale parameter of the fitted
    # distribution; the mean is that times a family-specific factor -- a Gamma
    # function term for Weibull, `exp(sigma^2 / 2)` for Lognormal -- which the
    # additive predictor does not carry. What *is* exact is the ratio: the
    # whole time distribution is multiplied by `exp(contribution)`, so every
    # quantile of it moves by the same factor, the mean included.
    "Weibull": ("a location on the log-time scale (an accelerated-failure-time "
                "predictor, not the log of the expected time)",
                "A contribution of +1 multiplies every survival quantile by "
                "2.72; the mean moves by the same factor, but exp(eta) itself "
                "is the distribution's scale, not its mean."),
    "Loglog": ("a location on the log-time scale (an accelerated-failure-time "
               "predictor, not the log of the expected time)",
               "A contribution of +1 multiplies every survival quantile by "
               "2.72; exp(eta) is the distribution's scale, not its mean."),
    "Lognormal": ("a location on the log-time scale (an accelerated-failure-time "
                  "predictor, not the log of the expected time)",
                  "A contribution of +1 multiplies every survival quantile by "
                  "2.72; the mean is exp(eta) * exp(sigma^2 / 2), so exp(eta) "
                  "alone understates it."),
}


#: What a +1 contribution does on each link scale an LSS parameter can carry.
_LINK_UNIT = {
    "identity": "a contribution of +1 adds 1 to {parameter}",
    "log": "a contribution of +1 multiplies {parameter} by exp(1) = 2.72",
    "logit": "a contribution of +1 multiplies the odds of {parameter} by 2.72",
    "sqrt": "the parameter is the square of the additive predictor",
}


def _lss_link(family, parameter):
    """The link one distribution parameter carries, from the capability table.

    The links live in the generated capability manifest, which is the one place
    that records them for every LSS family, so this reads them there rather than
    restating them.
    """
    from mboostrs.capabilities import family_capabilities

    name = type(family).__name__
    for row in family_capabilities(as_frame=False):
        if row["name"] != name:
            continue
        parameters = [item.strip() for item in str(row.get("parameters", "")).split(",")]
        links = [item.strip() for item in str(row.get("links", "")).split(",")]
        if parameter in parameters and len(links) == len(parameters):
            link = links[parameters.index(parameter)]
            unit = _LINK_UNIT.get(link)
            unit = unit.format(parameter=parameter) if unit else None
            return link, unit
    return None, None


def _response_scale(family):
    name = type(family).__name__
    scale, unit = _RESPONSE_SCALE.get(
        name,
        (f"the {name} family's link scale",
         "The relation between a contribution and the response is the family's "
         "inverse link, which is not summarized here."),
    )
    return scale, unit


def _facts_block(model, cv, response_description):
    """The fit's own summary, plus what its numbers are measured in."""
    family = getattr(model, "_family", None) or getattr(model, "_families", None)
    scale, unit = _response_scale(family)
    lines = [model.summary(cv=cv).as_text()]
    lines.append("")
    lines.append(f"Response scale: {scale}. {unit}")
    if response_description:
        lines.append(f"Response meaning: {response_description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# learner mechanics
# ---------------------------------------------------------------------------

def _ordinal(k):
    return {1: "first", 2: "second", 3: "third"}.get(int(k), f"{int(k)}th")


def _learner_note(spec, kind, tensor):
    """One clause describing what the learner computes and how it is penalized."""
    if tensor is not None:
        left, right = tensor.left, tensor.right
        return (
            f"row-wise tensor product of a {left.kind} basis on {left.column} "
            f"and a {right.kind} basis on {right.column}, penalized by the "
            "Kronecker sum of the two marginal penalties"
        )
    if spec is None:
        return "linear term, unpenalized"
    name = type(spec).__name__
    df = getattr(spec, "df", None)
    lam = getattr(spec, "lambda_value", None)
    smoothing = (
        f"df={_num(df)}" if df is not None
        else (f"lambda={_num(lam)}" if lam is not None else "unpenalized")
    )
    if name == "Bols":
        base = "linear term" + ("" if getattr(spec, "intercept", True) else ", no intercept")
        note = f"{base}, {smoothing}"
    elif name in ("Bbs", "Bns"):
        basis = "natural cubic spline" if name == "Bns" else (
            f"B-spline basis of degree {getattr(spec, 'degree', 3)}"
        )
        note = (
            f"{basis} on {getattr(spec, 'knots', 20)} inner knots, {smoothing}, "
            f"{_ordinal(getattr(spec, 'differences', 2))}-difference penalty"
        )
    elif name == "Bmono":
        note = (
            f"B-spline basis under a monotonicity constraint "
            f"({getattr(spec, 'constraint', 'increasing')} by construction), {smoothing}"
        )
    elif name == "Bconstrained":
        note = f"B-spline basis under a caller-supplied constraint matrix, {smoothing}"
    elif name == "Brandom":
        note = f"ridge-penalized effect over the levels of the column, {smoothing}"
    elif name == "Bspatial":
        note = (
            f"tensor-product spline over ({spec.name1}, {spec.name2}), {smoothing}"
        )
    elif name == "Btree":
        note = (
            f"regression tree, maxdepth={getattr(spec, 'maxdepth', 1)}, "
            f"minsplit={getattr(spec, 'minsplit', 10)} (a stump at maxdepth=1)"
        )
    elif name == "Bkernel":
        note = (
            f"kernel ridge term, {getattr(spec, 'kernel', 'gaussian')} kernel, "
            f"bandwidth={_num(getattr(spec, 'bandwidth', 1.0))}, {smoothing}"
        )
    elif name == "Bmrf":
        note = f"Markov random field over a supplied neighbourhood matrix, {smoothing}"
    elif name == "Brad":
        note = f"radial-basis term on {getattr(spec, 'knots', 100)} knots, {smoothing}"
    elif name == "Buser":
        note = f"caller-supplied design and penalty matrices, {smoothing}"
    else:
        note = smoothing
    if getattr(spec, "by", None) is not None:
        note += f"; multiplied by {spec.by} (varying coefficient)"
    if getattr(spec, "center", False):
        note += "; centered"
    return f"{kind}: {note}" if kind else note


def _null_space_note(spec, tensor):
    """What a learner can represent when its `df` sits at the shrinkage floor.

    A P-spline penalty of difference order `d` leaves a `d`-dimensional
    unpenalized subspace, and `df(lambda)` falls to `d` and stops. A learner
    asked for `df = d` is therefore fitted at maximum shrinkage and confined to
    that subspace: second differences leave straight lines, first differences
    leave constants, and a two-margin tensor leaves the product of the marginal
    null spaces. The effect still has a shape, but not a shape the data chose,
    and a reader told only "monotone increasing" would take it for one. The
    learners warn about this at fit time; this repeats it where the effect is
    read. `center=True` projects the null space out, and then no floor applies.
    """
    if tensor is not None:
        from mboostrs.interactions import _marginal_null_dim

        floor = _marginal_null_dim(tensor.left) * _marginal_null_dim(tensor.right)
        df = getattr(spec, "df", None)
        if floor and df is not None and float(df) <= floor:
            left, right = tensor.left.column, tensor.right.column
            span = (
                f"{{1, {left}, {right}, {left}*{right}}}" if floor == 4
                else f"the {floor}-dimensional unpenalized part of its penalty"
            )
            return (
                f"At the shrinkage floor: df={_num(df)} is at or below the "
                f"{floor}-dimensional null space of the tensor penalty, which no "
                f"finite lambda reaches, so this term is fitted at maximum "
                f"shrinkage and spans only {span}. The surface below is that "
                f"span, not a freely estimated one."
            )
        return None

    if spec is None or getattr(spec, "center", False):
        return None
    kind = type(spec).__name__
    differences = getattr(spec, "differences", None)
    if kind == "Bspatial" and differences is not None:
        null_rank = int(differences) ** 2
    elif kind in ("Bbs", "Bmono", "Bconstrained") and differences is not None:
        null_rank = int(differences)
    else:
        return None
    df = getattr(spec, "df", None)
    if df is None or float(df) > null_rank:
        return None
    leaves = {1: "constants", 2: "straight lines", 3: "quadratics"}
    span = leaves.get(null_rank, f"the {null_rank}-dimensional unpenalized subspace")
    if kind == "Bspatial":
        span = f"the {null_rank}-dimensional unpenalized subspace of the tensor penalty"
    return (
        f"At the shrinkage floor: df={_num(df)} is the rank of this penalty's "
        f"null space, which no finite lambda reaches, so the learner is fitted "
        f"at maximum shrinkage and can only represent {span}."
    )


def _interaction_for(model, column):
    for interaction in getattr(model, "_interactions", None) or ():
        if interaction.derived == column:
            return interaction
    return None


def _covariates(model, index, name):
    """Columns one learner reads, and the readable name of the term."""
    specs = getattr(model, "_learners_spec", None)
    spec = specs[index] if specs else None
    if spec is not None:
        tensor = _interaction_for(model, getattr(spec, "name", None))
        if tensor is not None:
            return [tensor.left.column, tensor.right.column], tensor.name, spec, tensor
        kind = type(spec).__name__
        if kind == "Bspatial":
            return [spec.name1, spec.name2], name, spec, None
        if kind == "Btree":
            return list(spec.names), name, spec, None
        if kind == "Brad":
            return list(spec.names), name, spec, None
        if kind == "Buser":
            # A caller-supplied design has no covariate to plot against.
            return [], name, spec, None
        return [spec.name], name, spec, None
    return [name], name, None, None


def _label(model, index):
    """How one learner is named in this description.

    An interaction reports as the term the user wrote rather than as its derived
    column. A learner over several columns reports as its type and all of them,
    because the stored name keeps only the first — a `btree(x1, x3)` is named
    `x1`, which is also what a `bbs(x1)` in the same formula is named. A name two
    learners still share — which the decomposition idiom
    ``bols(x) + bbs(x, df=1, center=True)`` produces — carries the learner type
    so the two blocks stay distinguishable.
    """
    names = list(getattr(model, "_learner_names", []))
    name = names[index]
    covariates, label, spec, tensor = _covariates(model, index, name)
    if tensor is None and spec is not None and len(covariates) > 1:
        return f"{type(spec).__name__.lower()}({', '.join(covariates)})"
    if label == name and names.count(name) > 1 and spec is not None:
        return f"{name} ({type(spec).__name__.lower()})"
    return label


def _levels_of(model, column):
    """Level labels of an encoded factor column, or None if it is numeric."""
    recipe = getattr(model, "_transform_recipe", None)
    for step in getattr(recipe, "steps", ()) or ():
        if step.kind == "factor" and step.name == column and step.params:
            return tuple(step.params[0])
    return None


def _column(model, name):
    data = getattr(model, "_data", None)
    if data is None:
        raise ValueError("this description needs the fitting data the model stored")
    if isinstance(data, dict):
        values = data[name]
    else:
        values = data[name]
    values = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    return np.asarray(values, dtype=float).ravel()


# ---------------------------------------------------------------------------
# effect shape
# ---------------------------------------------------------------------------

def _shape_note(x, y):
    """One sentence on the monotonicity and turning points of an effect."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    y = y[order]
    x = x[order]
    span = float(y.max() - y.min()) if y.size else 0.0
    if y.size < 3 or span <= 0:
        return "flat" if span <= 0 else "too few points to characterize"
    steps = np.diff(y)
    # A step under one per cent of the effect's own span is flat for reading
    # purposes; without this every rounding wiggle counts as a turn.
    live = np.where(np.abs(steps) > 0.01 * span, np.sign(steps), 0.0)
    live = live[live != 0]
    turns = int(np.sum(live[1:] != live[:-1])) if live.size > 1 else 0
    if turns == 0:
        direction = "increasing" if live.size and live[0] > 0 else "decreasing"
        return f"monotone {direction} over the observed range"
    where_max, where_min = _num(x[int(np.argmax(y))]), _num(x[int(np.argmin(y))])
    return (
        f"{turns} turning point{'s' if turns > 1 else ''}; highest near "
        f"{where_max}, lowest near {where_min}"
    )


def _condense(x, y, n_points, levels=None):
    """Reduce (x, effect) to at most `n_points` labelled values.

    Distinct covariate values are reported as they are while they are few
    enough to read — which is what a factor, an indicator or a stump needs.
    A continuous covariate with more distinct values than that is summarized by
    equal-count intervals, each carrying the mean effect of its rows, so the
    printed table stays the same size whatever `n` is.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    unique = np.unique(x)
    if unique.size <= max(n_points, 2):
        keys, values = [], []
        for value in unique:
            rows = x == value
            label = (
                f'{levels[int(value)]}' if levels is not None
                and 0 <= int(value) < len(levels) else _num(value)
            )
            keys.append(label)
            values.append(float(np.mean(y[rows])))
        return keys, values, np.array(unique, dtype=float), np.array(values)

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_points + 1)))
    keys, values, centres = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        last = hi == edges[-1]
        rows = (x >= lo) & (x <= hi) if last else (x >= lo) & (x < hi)
        if not np.any(rows):
            continue
        keys.append(f"({_num(lo)}, {_num(hi)})")
        values.append(float(np.mean(y[rows])))
        centres.append(0.5 * (float(lo) + float(hi)))
    return keys, values, np.array(centres), np.array(values)


def _grid_block(x1, x2, effect, name1, name2, cells):
    """Mean effect per cell of an equal-count grid over two covariates."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    effect = np.asarray(effect, dtype=float)
    edges1 = np.unique(np.quantile(x1, np.linspace(0.0, 1.0, cells + 1)))
    edges2 = np.unique(np.quantile(x2, np.linspace(0.0, 1.0, cells + 1)))
    precision = _effect_precision(effect)

    header = [f"{name1} \\ {name2}"] + [
        f"({_num(lo)}, {_num(hi)})" for lo, hi in zip(edges2[:-1], edges2[1:], strict=True)
    ]
    lines = [", ".join(header)]
    for lo1, hi1 in zip(edges1[:-1], edges1[1:], strict=True):
        rows1 = (x1 >= lo1) & (x1 <= hi1) if hi1 == edges1[-1] else (x1 >= lo1) & (x1 < hi1)
        cells_out = [f"({_num(lo1)}, {_num(hi1)})"]
        for lo2, hi2 in zip(edges2[:-1], edges2[1:], strict=True):
            rows2 = (x2 >= lo2) & (x2 <= hi2) if hi2 == edges2[-1] else (x2 >= lo2) & (x2 < hi2)
            rows = rows1 & rows2
            cells_out.append(
                _num(np.mean(effect[rows]), precision + 1) if np.any(rows) else "NA"
            )
        lines.append(", ".join(cells_out))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# per-learner blocks
# ---------------------------------------------------------------------------

def _selection_facts(model, name, index):
    """Updates spent on one learner, its share of the path, and its first one."""
    selected = list(getattr(model, "_selected", []))
    mstop = len(selected)
    hits = [step for step, chosen in enumerate(selected, start=1) if chosen == index]
    if not hits:
        return None
    return {
        "updates": len(hits),
        "share": len(hits) / mstop if mstop else float("nan"),
        "first": hits[0],
        "mstop": mstop,
    }


def _effect_values(model, index, name, n_points):
    """(x, effect) for one learner, on a grid where that is exact.

    A smooth learner in one covariate is evaluated on an equispaced grid, which
    is the effect itself rather than a summary of it. Everything else is
    evaluated at the training rows and condensed afterwards.
    """
    smooth_kinds = ("Bbs", "Bmono", "Bconstrained", "Bns", "Bkernel")
    specs = getattr(model, "_learners_spec", None)
    spec = specs[index] if specs else None
    on_grid = spec is not None and type(spec).__name__ in smooth_kinds
    effects = model.partial_effects(
        which=index, n_points=n_points if on_grid else None
    )
    if not effects:
        return None, None
    x, y = next(iter(effects.values()))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _glm_effect_block(model, name, index, importance, n_points):
    """A glmboost term: one coefficient, so the effect is the coefficient."""
    del n_points
    coefficients = model.coef()
    coefficient = float(coefficients.get(name, 0.0))
    x = _column(model, name)
    contributions = coefficient * x
    lines = [
        f"Term: {name}  [linear, unpenalized]",
        f"Coefficient: {_num(coefficient, 6)}  (contribution = "
        f"{_num(coefficient, 6)} * {name})",
        f"Covariate range: [{_num(x.min())}, {_num(x.max())}]",
        f"Contribution over that range: [{_num(contributions.min())}, "
        f"{_num(contributions.max())}]",
    ]
    lines.extend(_selection_lines(model, name, index, importance))
    return "\n".join(lines)


def _selection_lines(model, name, index, importance):
    lines = []
    facts = _selection_facts(model, name, index)
    if facts is None:
        lines.append("Never selected: this term contributes exactly zero.")
        return lines
    lines.append(
        f"Selected in {facts['updates']} of {facts['mstop']} updates "
        f"({facts['share']:.1%}), first at iteration {facts['first']}"
    )
    if importance is not None:
        lines.append(f"Importance: {_pct(importance)} of total risk reduction")
    return lines


def _effect_block(model, name, index, importance, n_points, cells):
    """The block for one gamboost / gamboostLSS learner."""
    covariates, _, spec, tensor = _covariates(model, index, name)
    # The learner's own class names the term (`bols`, `bbs`, `bmono`), not the
    # engine block it was packed into (`scalar`, `bbs`, `buser`).
    kind = None if spec is None else type(spec).__name__.lower()
    header = [
        f"Learner: {_label(model, index)}  [{_learner_note(spec, kind, tensor)}]",
    ]
    header.extend(_selection_lines(model, name, index, importance))
    floor = _null_space_note(spec, tensor)
    if floor is not None:
        header.append(floor)

    x, y = _effect_values(model, index, name, n_points)
    if y is None:
        return "\n".join(header)

    if len(covariates) == 1:
        column = covariates[0]
        levels = _levels_of(model, column)
        kind_note = (
            f"categorical, {len(levels)} levels" if levels is not None
            else f"observed range [{_num(x.min())}, {_num(x.max())}]"
        )
        header.append(f"Covariate: {column} ({kind_note})")
        keys, values, xs, ys = _condense(x, y, n_points, levels)
        precision = _effect_precision(values)
        header.append("Contribution: " + _pairs_block(keys, values, precision))
        header.append(
            f"Contribution range: [{_num(y.min())}, {_num(y.max())}] "
            f"(span {_num(y.max() - y.min())})"
        )
        if levels is None:
            header.append(f"Shape: {_shape_note(xs, ys)}")
        return "\n".join(header)

    if len(covariates) == 2:
        first, second = covariates
        x1, x2 = _column(model, first), _column(model, second)
        header.append(
            f"Covariates: {first} in [{_num(x1.min())}, {_num(x1.max())}], "
            f"{second} in [{_num(x2.min())}, {_num(x2.max())}]"
        )
        header.append(
            f"Contribution range: [{_num(y.min())}, {_num(y.max())}] "
            f"(span {_num(y.max() - y.min())})"
        )
        header.append(
            f"Mean contribution per cell of an equal-count grid; rows are "
            f"{first}, columns are {second}:"
        )
        header.append(_grid_block(x1, x2, y, first, second, cells))
        return "\n".join(header)

    if not covariates:
        header.append(
            "The design was supplied directly, so there is no covariate to "
            "report the contribution against."
        )
        header.append(
            f"Contribution range: [{_num(y.min())}, {_num(y.max())}] "
            f"(span {_num(y.max() - y.min())})"
        )
        return "\n".join(header)

    header.append(f"Covariates: {', '.join(covariates)}")
    header.append(
        f"Contribution range: [{_num(y.min())}, {_num(y.max())}] "
        f"(span {_num(y.max() - y.min())})"
    )
    header.append(
        "The contribution is joint in those covariates. Each line below is the "
        "mean contribution within equal-count intervals of one covariate, with "
        "the others left as they are — a marginal view of a joint effect:"
    )
    for column in covariates:
        values_x = _column(model, column)
        keys, values, _, _ = _condense(values_x, y, min(n_points, 6),
                                       _levels_of(model, column))
        header.append(
            f"  {column}: " + _pairs_block(keys, values, _effect_precision(values))
        )
    return "\n".join(header)


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------

def _importance(model):
    """Learner index -> share of risk reduction, or None when unavailable.

    This is `varimp`'s quantity — the drop in in-bag risk at each update,
    credited to the learner that update chose, normalized over the path — but
    keyed by learner index rather than by name. Two learners can carry one name
    (``bols(x) + bbs(x, df=1, center=True)`` is the standard decomposition
    idiom), and a name-keyed share cannot separate them.
    """
    risk = np.asarray(getattr(model, "_risk", []), dtype=float)
    selected = [int(index) for index in getattr(model, "_selected", [])]
    n_learners = len(getattr(model, "_learner_names", []))
    if risk.size >= 2 and selected and n_learners:
        reduction = np.zeros(n_learners)
        # risk[i] is the risk after i updates, so -diff is what each update
        # removed. The path can be longer than the risk vector only if the model
        # was sliced, so pair them by position and stop at the shorter one.
        steps = np.minimum(len(selected), risk.size - 1)
        np.add.at(reduction, np.array(selected[:steps]), -np.diff(risk)[:steps])
        total = float(np.sum(np.abs(reduction)))
        if total > 0:
            return {index: float(reduction[index]) / total for index in range(n_learners)}
        return {index: 0.0 for index in range(n_learners)}
    # An LSS fit keeps its risk path per cycle, not per update, and its own
    # `varimp` decides whether attribution is possible at all.
    try:
        shares = model.varimp()["reduction"]
    except (NotImplementedError, AttributeError, ValueError, KeyError):
        return None
    names = list(getattr(model, "_learner_names", []))
    return {
        index: float(shares[name])
        for index, name in enumerate(names)
        if name in shares
    }


def _importance_block(model, shares, stabsel):
    """A ranked table: what each learner removed, and how often it was picked."""
    names = list(getattr(model, "_learner_names", []))
    display = {index: _label(model, index) for index in range(len(names))}

    rows = []
    for index, name in enumerate(names):
        facts = _selection_facts(model, name, index)
        share = None if shares is None else shares.get(index)
        rows.append({
            "label": display[index],
            "share": share,
            "updates": 0 if facts is None else facts["updates"],
            "first": None if facts is None else facts["first"],
        })
    rows.sort(key=lambda row: (
        -(row["share"] if row["share"] is not None else 0.0), -row["updates"]
    ))

    phat = None
    if stabsel is not None:
        phat = {
            stabsel.learner_names[i]: float(stabsel.max_[i])
            for i in range(len(stabsel.learner_names))
        }

    lines = ["Base learners, ranked:"]
    for row in rows:
        parts = [f"  {row['label']}"]
        if row["share"] is not None:
            parts.append(f"importance {_pct(row['share'])}")
        parts.append(
            _plural(row["updates"], "update") if row["updates"]
            else "never selected (contributes zero)"
        )
        if row["first"] is not None:
            parts.append(f"entered at iteration {row['first']}")
        if phat is not None and row["label"] in phat:
            parts.append(
                f"selected in {phat[row['label']]:.0%} of stability-selection "
                f"subsamples"
            )
        lines.append(" — ".join(parts))
    if shares is None:
        lines.append(
            "Risk-reduction shares are unavailable for this fit; a cyclic LSS "
            "path records risk once per cycle, not once per update, so only "
            "selection counts are exact. Refit with method='noncyclic' for "
            "per-update attribution."
        )
    if stabsel is not None:
        lines.append(
            f"Stability selection: {stabsel.n_fits} subsample fits, cutoff "
            f"{stabsel.cutoff:.2f}, PFER {stabsel.PFER:.3g}. PFER bounds the "
            f"expected *number* of falsely selected learners from above -- it "
            f"is a count, not a probability and not a rate. The figure "
            f"reported per learner is the largest fraction of subsamples in "
            f"which it was selected: an empirical frequency, not the "
            f"probability that the learner belongs in the model."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _check_belongs_to_model(model, cv, stabsel):
    """Refuse a CV or stability-selection result from a different fit.

    Both are passed in by the caller, and nothing tied them to the model being
    described: a result from another model would be rendered into the prompt
    as if it were this one's, and the reader has no way to tell. The learner
    count catches a result for another formula; the digest of the response
    and weights, recorded on both results, catches the same formula fitted to
    other data. A result built without a digest (by hand, or by an older
    release) is checked by learner count alone.
    """
    learner_count = len(getattr(model, "_learner_names", []) or [])
    if learner_count == 0:
        return

    if stabsel is not None:
        p = getattr(stabsel, "p", None)
        if p is None:
            p = len(getattr(stabsel, "learner_names", []) or [])
        if p and p != learner_count:
            raise ValueError(
                f"the stabsel= result describes {p} learners and this model "
                f"has {learner_count}, so it is not this model's. Pass the "
                f"result of model.stabsel(...) for the model being described."
            )
        recorded = getattr(stabsel, "data_fingerprint", "")
        own = model._data_fingerprint() if hasattr(model, "_data_fingerprint") else ""
        if recorded and own and recorded != own:
            raise ValueError(
                "the stabsel= result was computed on another response or other "
                "weights than this model was fitted to (or the same rows in "
                "another order), so its selection frequencies do not describe "
                "this model. Pass the result of model.stabsel(...) for the "
                "model being described."
            )

    if cv is not None:
        names = getattr(cv, "learner_names", None)
        if names is not None and len(names) != learner_count:
            raise ValueError(
                f"the cv= result describes {len(names)} learners and this "
                f"model has {learner_count}, so it is not this model's. Pass "
                f"the result of model.cvrisk(...) for the model being "
                f"described."
            )
        # The learner count cannot tell apart two models of the same
        # formula on different data; the response-and-weights digest can.
        recorded = getattr(cv, "data_fingerprint", "")
        own = model._data_fingerprint() if hasattr(model, "_data_fingerprint") else ""
        if recorded and own and recorded != own:
            raise ValueError(
                "the cv= result was computed on another response or other "
                "weights than this model was fitted to (or the same rows in "
                "another order), so its risk does not describe this model. Pass "
                "the result of model.cvrisk(...) for the model being described."
            )


def _ordered_learners(model, shares, which, top):
    """Learner indices to describe, most important first."""
    names = list(getattr(model, "_learner_names", []))
    if which is None:
        selected = list(dict.fromkeys(getattr(model, "_selected", [])))
        indices = [int(index) for index in selected]
    else:
        requested = [which] if isinstance(which, str | int | np.integer) else list(which)
        indices = []
        for item in requested:
            if isinstance(item, str | np.str_):
                if item not in names:
                    raise ValueError(
                        f"unknown base learner {item!r}; the model has {names}"
                    )
                indices.append(names.index(item))
            else:
                index = int(item)
                if not 0 <= index < len(names):
                    raise ValueError(f"learner index {index} is out of range")
                indices.append(index)
    if shares is not None:
        indices.sort(key=lambda index: -abs(shares.get(index, 0.0)))
    if top is not None:
        indices = indices[: int(top)]
    return indices


def _effects_block(model, shares, which, top, n_points, cells):
    names = list(getattr(model, "_learner_names", []))
    indices = _ordered_learners(model, shares, which, top)
    from mboostrs._dispatch import is_glmboost

    is_glm = is_glmboost(model)
    blocks = []
    for index in indices:
        name = names[index]
        share = None if shares is None else shares.get(index)
        if is_glm:
            blocks.append(_glm_effect_block(model, name, index, share, n_points))
        else:
            blocks.append(
                _effect_block(model, name, index, share, n_points, cells)
            )
    if not blocks:
        return "Effects: the path selected no base learner, so the model is the offset alone."
    kept = len(indices)
    total = len(list(dict.fromkeys(getattr(model, "_selected", []))))
    heading = f"Effects ({kept} of {total} selected learners, most important first):"
    return "\n\n".join([heading, *blocks])


def _lss_prompt(model, kwargs):
    """One description per distribution parameter of an LSS fit."""
    parameter = kwargs.pop("parameter", None)
    parameters = list(model.param_names) if parameter is None else (
        [parameter] if isinstance(parameter, str) else list(parameter)
    )
    blocks = []
    for name in parameters:
        view = _LSSParameterView(model, name)
        link, unit = _lss_link(model._family, name)
        scale = f"{link} link" if link else "the parameter's own link scale"
        blocks.append(
            f"=== Distribution parameter: {name} ({scale}"
            + (f"; {unit}" if unit else "")
            + ") ===\n\n"
            + _body(view, dict(kwargs), skip=("facts",))
        )
    facts = model.summary().as_text()
    return facts, blocks


class _LSSParameterView:
    """One LSS parameter presented with the standard-model attribute names.

    The block writers read `_learner_names`, `_selected`, `_learners_spec` and
    `partial_effects(which=...)`. An LSS fit keeps one such set per distribution
    parameter, so this adapter exposes one parameter's set under those names
    instead of duplicating every writer for the LSS shape.
    """

    def __init__(self, model, parameter):
        self._model = model
        self._parameter = parameter
        block = model._params[list(model.param_names).index(parameter)]
        self._learner_names = list(block["predictor_names"])
        self._selected = [int(index) for index in block["selected"]]
        self._learners_spec = list(block.get("learners") or [])
        self._learner_types = [type(spec).__name__.lower() for spec in self._learners_spec]
        self._data = getattr(model, "_data", None)
        self._interactions = getattr(model, "_interactions", None)
        self._transform_recipe = getattr(model, "_transform_recipe", None)

    def partial_effects(self, which=None, n_points=None):
        # An LSS fit has no grid evaluation, so a smooth effect is read off the
        # training rows and condensed like any other.
        del n_points
        effects = self._model.partial_effects(self._parameter, which=which)
        if isinstance(effects, tuple):
            return {self._learner_names[int(which)]: effects}
        return effects

    def varimp(self):
        return self._model.varimp(parameter=self._parameter)


def _body(model, kwargs, skip=()):
    """The sections that describe one predictor: importance, then effects."""
    include = kwargs["include"]
    parts = []
    shares = _importance(model) if {"importance", "effects"} & set(include) else None
    if "importance" in include and "importance" not in skip:
        parts.append(_importance_block(model, shares, kwargs["stabsel"]))
    if "effects" in include and "effects" not in skip:
        parts.append(_effects_block(
            model, shares, kwargs["which"], kwargs["top"],
            kwargs["n_points"], kwargs["cells"],
        ))
    return "\n\n".join(parts)


def llm_explain_prompt(
    model,
    *,
    which=None,
    top=None,
    n_points=12,
    cells=5,
    parameter=None,
    dataset_description=None,
    response_description=None,
    task=None,
    cv=None,
    stabsel=None,
    include=SECTIONS,
    expert="an expert statistician and applied data analyst",
    output="text",
    num_sentences=7,
):
    """Render a fitted model as a prompt for a language model.

    The prompt states how the fit was produced, what each retained base learner
    contributes over the observed covariate range, how much of the risk it
    removed, and which questions the fit cannot answer. Nothing is sent
    anywhere: the return value is text (or chat messages) for the caller to pass
    to a model of their choosing.

    Parameters
    ----------
    model : GLMBoostModel, GAMBoostModel, or LSSModel
        A fitted model.
    which : str, int, or list of those, optional
        Base learners to describe. Default: every selected learner.
    top : int, optional
        Keep only this many learners, most important first. Default: all.
    n_points : int
        Values per effect. A smooth effect in one covariate is evaluated on an
        equispaced grid of this size; anything else is condensed to at most this
        many equal-count intervals.
    cells : int
        Grid resolution per axis for an effect over two covariates.
    parameter : str or list of str, optional
        For an LSS fit, the distribution parameters to describe. Default: all.
    dataset_description : str, optional
        What the data are, where they came from, what the columns mean. A model
        cannot recover this from the fit and reads it as context.
    response_description : str, optional
        What the response measures, in the caller's own terms.
    task : str, optional
        The closing instruction. Default: an analyst's brief in four parts —
        findings, surprising patterns, what could mislead, next check.
    cv : CvRiskResult, optional
        A cross-validation result for this fit. Included so the stopping
        iteration can be judged; without it the prompt says so.
    stabsel : StabSelResult, optional
        A stability-selection result for this fit. Adds a selection probability
        per learner.
    include : sequence of str
        Sections to write, from :data:`SECTIONS`.
    expert : str
        Who the reader is told to be, in the system message.
    output : {"text", "messages", "turns"}
        ``"text"`` returns one string. ``"messages"`` returns
        ``[{"role": "system", ...}, {"role": "user", ...}]``, the shape the
        OpenAI-compatible and Anthropic APIs take. ``"turns"`` returns
        ``{"system": str, "turns": [str, str, str]}``: the description, then a
        request to name what is surprising and why, then a request to compress
        the two into ``num_sentences``. Send the turns in order, keeping each
        reply in the conversation; :func:`llm_describe` does that for you.
    num_sentences : int
        Sentence budget for the closing turn of ``output="turns"``.

    Returns
    -------
    str, list of dict, or dict

    Examples
    --------
    >>> prompt = fit.llm_explain_prompt(          # doctest: +SKIP
    ...     dataset_description="Body fat, 71 men, DEXA reference measurement.",
    ...     top=5,
    ... )
    >>> fit.llm_describe(dataset_description="...")   # doctest: +SKIP
    """
    unknown = [name for name in include if name not in SECTIONS]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    _check_belongs_to_model(model, cv, stabsel)
    if output not in ("text", "messages", "turns"):
        raise ValueError(
            f"output must be 'text', 'messages' or 'turns', got {output!r}"
        )

    kwargs = {
        "which": which, "top": top, "n_points": n_points, "cells": cells,
        "stabsel": stabsel, "include": tuple(include),
    }
    parts = []
    if "reading" in include:
        parts.append(_READING)

    if type(model).__name__ == "LSSModel":
        facts, blocks = _lss_prompt(model, {**kwargs, "parameter": parameter})
        if "facts" in include:
            scale_note = (
                "Each distribution parameter has its own additive predictor on "
                "its own link scale; the blocks below take one parameter each."
            )
            parts.append(f"{facts}\n\n{scale_note}")
            if response_description:
                parts.append(f"Response meaning: {response_description}")
        parts.extend(blocks)
    else:
        if "facts" in include:
            parts.append(_facts_block(model, cv, response_description))
        body = _body(model, kwargs)
        if body:
            parts.append(body)

    if dataset_description and "facts" in include:
        parts.append(f"What the data are:\n{dataset_description}")
    if "caveats" in include:
        caveats = _CAVEATS
        if cv is None:
            caveats += (
                "\n- No cross-validated risk was supplied, so nothing here says "
                "whether the fit stopped too early or too late."
            )
        parts.append(caveats)
    if "task" in include:
        parts.append(task or _DEFAULT_TASK)

    text = "\n\n".join(parts)
    if output == "text":
        return text
    system = _SYSTEM_MESSAGE.format(expert=expert)
    if output == "turns":
        return {
            "system": system,
            "turns": [
                text,
                _PROBE_SURPRISES,
                _PROBE_CONDENSE.format(num_sentences=int(num_sentences)),
            ],
        }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


# ---------------------------------------------------------------------------
# sending the prompt
# ---------------------------------------------------------------------------

#: NVIDIA's hosted inference endpoint, which speaks the OpenAI chat-completions
#: protocol. Any endpoint that speaks it works: pass `base_url=` for a local
#: NIM container, vLLM, llama.cpp, Ollama, or OpenAI itself.
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: The default model. `llm=` overrides it, and the endpoint decides which ids it
#: serves — `GET {base_url}/models` lists them.
DEFAULT_LLM = "nvidia/nemotron-3-ultra-550b-a55b"

#: Environment variables read for the endpoint and the key, in order.
_BASE_URL_VARS = ("MBOOSTRS_LLM_BASE_URL", "NVIDIA_BASE_URL")
_API_KEY_VARS = ("MBOOSTRS_LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")


def _from_env(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def _reply_text(payload):
    """The assistant text of one chat-completions response.

    A reasoning model returns its chain of thought in `reasoning_content` and its
    answer in `content`; only the answer is wanted, and an empty `content` beside
    a filled `reasoning_content` means the reply was cut off before the model
    finished thinking, which is worth saying rather than returning "".
    """
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"the endpoint returned no choices: {payload!r:.400}")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if content:
        return content
    if (message.get("reasoning_content") or "").strip():
        raise RuntimeError(
            "the model spent the whole token budget reasoning and returned no "
            "answer; raise max_tokens"
        )
    raise RuntimeError(f"the endpoint returned an empty reply: {payload!r:.400}")


def _post_chat(messages, *, llm, base_url, api_key, max_tokens, temperature, timeout):
    """One chat-completions request over the standard library.

    `urllib` rather than an SDK: this is a single POST, and mboostrs should not
    acquire a dependency to make it. Callers who already build a client can pass
    it as `client=` instead.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": llm,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(
            f"{url} returned HTTP {error.code} for model {llm!r}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach {url}: {error.reason}") from error


def _client_chat(client, messages, *, llm, max_tokens, temperature, return_payload=False):
    """One request through a caller-supplied OpenAI-compatible client."""
    response = client.chat.completions.create(
        model=llm,
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )
    if isinstance(response, dict):
        payload = response
    # The OpenAI SDK returns objects; `model_dump` is the documented way to a
    # mapping, and plain attribute access covers clients that lack it.
    elif hasattr(response, "model_dump"):
        payload = response.model_dump()
    else:
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage = {name: getattr(usage, name, None) for name in
                     ("prompt_tokens", "completion_tokens", "total_tokens")}
        payload = {"choices": [{"message": {
            "content": response.choices[0].message.content}}], "usage": usage}
    return payload if return_payload else _reply_text(payload)


def llm_describe(
    model,
    *,
    llm=DEFAULT_LLM,
    base_url=None,
    api_key=None,
    client=None,
    reasoning=False,
    num_sentences=7,
    max_tokens=4000,
    temperature=0.7,
    timeout=180,
    return_conversation=False,
    **prompt_kwargs,
):
    """Send the model's description to a language model and return the reply.

    This is the one function in the module that makes a network request. It
    builds the prompt with :func:`llm_explain_prompt`, posts it to an
    OpenAI-compatible chat-completions endpoint, and returns the text that comes
    back. The default endpoint is NVIDIA's hosted inference API and the default
    model is ``nvidia/nemotron-3-ultra-550b-a55b``; both are arguments, and the API key comes from the
    environment.

    The prompt instructs the model to reply in Markdown — short headings or bold
    labels, bullets, and small tables where they help. Render the returned text
    with a Markdown renderer (``display(Markdown(...))`` in a notebook, or a
    browser) rather than printing it as plain text.

    Parameters
    ----------
    model : GLMBoostModel, GAMBoostModel, or LSSModel
        A fitted model.
    llm : str
        Model id the endpoint serves. Default ``"nvidia/nemotron-3-ultra-550b-a55b"``.
    base_url : str, optional
        Endpoint root, without ``/chat/completions``. Default: the environment
        variable ``MBOOSTRS_LLM_BASE_URL`` or ``NVIDIA_BASE_URL``, else
        :data:`DEFAULT_BASE_URL`.
    api_key : str, optional
        Bearer token. Default: the first of ``MBOOSTRS_LLM_API_KEY``,
        ``NVIDIA_API_KEY``, ``OPENAI_API_KEY`` that is set.
    client : object, optional
        An OpenAI-compatible client — anything with
        ``chat.completions.create(model=, messages=, max_tokens=, temperature=)``.
        Given one, ``base_url`` and ``api_key`` are ignored and the client's own
        configuration is used.
    reasoning : bool
        ``False`` sends one request. ``True`` runs the three-turn sequence of
        ``output="turns"``: describe the effects, name what is surprising and
        why, then compress both into ``num_sentences``. Three requests, and the
        summary is the return value.
    num_sentences : int
        Sentence budget for the closing turn when ``reasoning=True``.
    max_tokens, temperature, timeout : int, float, float
        Per-request generation and socket settings.
    return_conversation : bool
        Return the full message list — system, user and assistant turns in order
        — instead of only the final reply.
    **prompt_kwargs
        Passed to :func:`llm_explain_prompt`: ``dataset_description``,
        ``response_description``, ``top``, ``n_points``, ``cv``, ``stabsel``,
        ``task``, ``expert``, and the rest.

    Returns
    -------
    str or list of dict

    Raises
    ------
    ValueError
        No API key was given and none of the environment variables is set.
    RuntimeError
        The endpoint refused the request, could not be reached, or returned a
        reply with no content.

    Examples
    --------
    >>> fit.llm_describe(                                  # doctest: +SKIP
    ...     dataset_description="Body fat of 71 women; covariates in cm.",
    ...     reasoning=True,
    ... )
    """
    for name in ("output", "num_sentences"):
        if name in prompt_kwargs:
            raise TypeError(
                f"llm_describe() controls {name!r} itself; use "
                f"{'reasoning=' if name == 'output' else 'num_sentences='} instead"
            )
    # A prompt with no closing instruction spends a request and comes back asking
    # what the reader wanted, so the task section is not optional when sending.
    include = prompt_kwargs.get("include")
    if include is not None and "task" not in include:
        prompt_kwargs["include"] = (*tuple(include), "task")

    if client is None:
        if api_key is None:
            api_key, _ = _from_env(_API_KEY_VARS)
        if not api_key:
            raise ValueError(
                "llm_describe() needs an API key: pass api_key=, or set one of "
                + ", ".join(_API_KEY_VARS)
                + ". Use llm_explain_prompt() to get the prompt text without "
                "sending it anywhere."
            )
        if base_url is None:
            base_url, _ = _from_env(_BASE_URL_VARS)
        base_url = base_url or DEFAULT_BASE_URL

    def send(messages):
        if client is not None:
            return _client_chat(
                client, messages, llm=llm, max_tokens=max_tokens,
                temperature=temperature,
            )
        return _reply_text(_post_chat(
            messages, llm=llm, base_url=base_url, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature, timeout=timeout,
        ))

    if reasoning:
        plan = llm_explain_prompt(
            model, output="turns", num_sentences=num_sentences, **prompt_kwargs
        )
        conversation = [{"role": "system", "content": plan["system"]}]
        for turn in plan["turns"]:
            conversation.append({"role": "user", "content": turn})
            conversation.append({"role": "assistant", "content": send(conversation)})
    else:
        conversation = list(llm_explain_prompt(model, output="messages", **prompt_kwargs))
        conversation.append({"role": "assistant", "content": send(conversation)})

    if return_conversation:
        return conversation
    return conversation[-1]["content"]
