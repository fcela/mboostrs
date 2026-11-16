"""Single-holdout adoption rules (issue #541).

A supplied design with one fold carries no across-fold variation: the
paired fold standard error is undefined, so before this fix the
residual tree was adopted on a bare 2% margin on the aggregate curve.
On the frozen cancer battery that rule validated at 0.032 and tested
at 0.42: the tree was adopted on a 34.7% margin, carried the whole
adopted prediction, and scored worse on the disjoint calibration rows
(0.56) than on the test rows it was reported on. Its per-row paired z
on the very rows that adopted it was 0.86.

On a single holdout the tree step now reads the per-row evidence of
the holdout itself: adoption needs its margin AND a paired per-row z
of at least 2.0, the bar the fold route uses when folds exist. The
interaction-pair rule was audited the same way and left unchanged:
on the battery draws the adopted pairs' per-row evidence is positive
and a pair-free counterfactual does no better, so pairs were not the
failure. A one-SE step rule for the same design was also tried on the
frozen battery, in level and paired form; each variant traded one
draw's gain for another's loss, so the stopping rule is unchanged.

Two sibling routes decide on the same design and were flagged by the
tree fix: the shape probe, whose paired fold comparison is inert on
one fold, leaving its aggregate margin alone; and the pair-form
picker, whose adoption test degenerates the same way. Both were
measured on the frozen battery before any change: all 20 shape
replacements the probe made across the three draws carry a
full-formula per-row z under 2.0 (range -0.84 to 1.66), and the
picker's two adoptions (draws 0 and 2) carry per-row z of 0.50 and
0.65 against margins of 3.5% and 4.0%. Each route now reads the
per-row differences of the comparison that produced its margin and
requires the same 2.0 bar, scoped exactly as the tree rule is: a
single supplied holdout only, row-coupled families untouched.

The coordinate-surface step was audited last, after the constructive
test for the judging-layer census recorded it adopting on the same
design: in 4 of 40 additive-panel draws it adopted a surface over two
noise columns on the margin route, the fold z inert, and a fresh-data
audit refused all four (per-row z on 20,000 fresh rows between -10.7
and 1.3; three of the four lose significantly out of sample). The
step now reads the per-row differences of its adoption comparison
and requires the same 2.0 bar, with the same scoping. One of the
four draws clears even the per-row bar on its own holdout (margin
6.2%, z 2.35) while fresh rows reverse it: the recorded limit of an
audit that reads the supplied rows, not a reason to leave the bar
off the other three.

The distributional adoption checks (census C6, `_validate_lss`, and
C7, `compare_finished_routes`) are the same defect in another form:
their paired standard error is NaN at one fold, and both SE clauses
waived on a non-finite SE, so adoption rested on the 2% margin
alone. Their held-out score is a log score with no per-row audit
form in the probes, so there is no substitute proof to read: the
checks now require the paired proof itself, and a standard error
that cannot be computed fails the check it belongs to. On a
Gaussian bell clipped inside (0, 1) -- a panel where the mean family
is the truth -- the unfixed checks adopted BetaLSS in 5 of 40
single-holdout draws; a fresh-data audit refuses all five (per-row
z on 20,000 fresh rows between -12.0 and -1.9). The study is
recorded in docs/book/benchmarks/lss_single_holdout_se.md.
"""

import warnings

import numpy as np

import mboostrs as mb
from mboostrs._autospec import (
    _held_out_pair_z,
    _path_held_out_evidence,
    _single_fold_design,
    probe_mstop,
)



def _tree_term(p):
    return ("btree(" + ", ".join(f"x{j}" for j in range(p))
            + ", maxdepth=5, selection=sse)")


def _single_holdout(rng, n, n_hold):
    folds = np.ones((n, 1))
    folds[rng.permutation(n)[:n_hold], 0] = 0.0
    return folds


def _tree_audit(data, folds, mains, p, largest=1200):
    """(relative aggregate gain, per-row paired z) of adding the tree."""
    family = mb.Binomial()
    candidate = f"{mains} + {_tree_term(p)}"
    base, _ = _path_held_out_evidence(data, "y", mains, family, folds,
                                      largest, 0.1)
    cand, _ = _path_held_out_evidence(data, "y", candidate, family, folds,
                                      largest, 0.1)
    rel = ((base.risk[base.chosen] - cand.risk[cand.chosen])
           / abs(base.risk[base.chosen]))
    z, _mean, _se, _fits = _held_out_pair_z(
        data, "y", mains, base.chosen, largest, 0.1,
        candidate, cand.chosen, largest, 0.1, family, folds)
    return rel, z


def test_single_fold_design_scoping():
    """The per-row route applies to one supplied holdout only: not to
    several folds, and not to a scheme (which carries a judge set)."""
    folds = np.ones((50, 1))
    assert _single_fold_design(folds, None, mb.Binomial())
    assert not _single_fold_design(np.ones((50, 3)), None, mb.Binomial())

    class _Scheme:
        pass

    assert not _single_fold_design(folds, _Scheme(), mb.Binomial())


def test_row_audit_certifies_a_genuine_tree():
    """A depth-5 tree over a sign-product truth the additive terms
    cannot represent: the aggregate margin is large and the per-row z
    clears the adoption bar, so the gate is a noise bar, not a ban."""
    rng = np.random.default_rng(0)
    n, p = 300, 6
    X = rng.normal(size=(n, p))
    eta = 2.6 * np.sign(X[:, 0] * X[:, 1]) + 0.7 * X[:, 2]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = {f"x{j}": X[:, j] for j in range(p)} | {"y": y}
    folds = _single_holdout(rng, n, 75)
    mains = "y ~ " + " + ".join(f"bbs(x{j}, df=4)" for j in range(p))
    rel, z = _tree_audit(data, folds, mains, p)
    assert rel >= 0.02
    assert z is not None and z >= 2.0


def test_row_audit_does_not_certify_a_valueless_tree():
    """Linear truth, weak signal: the all-column tree adds nothing the
    additive terms miss, and the per-row z stays below the adoption
    bar -- the audit cannot be read as certifying it."""
    rng = np.random.default_rng(101)
    n, p = 320, 20
    X = rng.normal(size=(n, p))
    eta = 0.55 * X[:, 0] - 0.5 * X[:, 1] + 0.45 * X[:, 2] \
        + 0.4 * np.sin(X[:, 3])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = {f"x{j}": X[:, j] for j in range(p)} | {"y": y}
    folds = _single_holdout(rng, n, 80)
    mains = "y ~ " + " + ".join(f"bbs(x{j}, df=4)" for j in range(p))
    rel, z = _tree_audit(data, folds, mains, p)
    assert z is not None and z < 2.0


def test_probe_mstop_multi_fold_keeps_the_argmin():
    """The per-row rules are scoped to single-holdout designs: the
    stopping sweep itself is unchanged, and with several folds it
    keeps the plain argmin of the aggregate curve."""
    rng = np.random.default_rng(3)
    n, p = 200, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=2.0, size=n)
    data = {f"x{j}": X[:, j] for j in range(p)} | {"y": y}
    formula = "y ~ " + " + ".join(f"bbs(x{j}, df=4)" for j in range(p))
    folds = np.zeros((n, 3))
    order = rng.permutation(n)
    for k in range(3):
        folds[order[k::3], k] = 1.0
    # Fold weights: 1 marks the rows a fold's fit trains on.
    folds = 1.0 - folds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = probe_mstop(data, "y", formula, mb.Gaussian(), folds=folds,
                          largest=1500, nu=0.1)
    risk = np.asarray(row.risk, dtype=float)
    assert row.chosen == int(np.nanargmin(risk[1:])) + 1


def _shape_probe_row(kind, data_seed, holdout_seed, n):
    import polars as pl

    from mboostrs._autospec import probe_shapes
    from mboostrs._frame import read_column
    from mboostrs.autospec import _term_for

    rng = np.random.default_rng(data_seed)
    x1 = rng.uniform(0, 1, n)
    if kind == "smooth":
        y = 2.0 * np.sin(np.pi * x1) + rng.normal(scale=1.0, size=n)
    else:  # "step"
        y = 3.0 * (x1 > 0.5) + rng.normal(scale=1.0, size=n)
    frame = pl.DataFrame({"x1": x1, "y": y})
    hold = np.random.default_rng(10_000 + holdout_seed)
    folds = np.ones((n, 1))
    folds[hold.permutation(n)[: n // 4], 0] = 0.0
    column = read_column(frame, "x1")
    terms = [_term_for(column, 4.0, "bbs")]
    evidence, _fits = probe_shapes(
        frame, "y", [column], terms, mb.Gaussian(), folds=folds,
        mstop=200, nu=0.1, df=4.0, max_fits=400, fixed_names=(), env=None)
    return evidence[0]


def test_shape_probe_blocks_a_valueless_replacement_on_a_single_holdout():
    """Smooth truth: btree's 6.3% aggregate margin on one holdout is a
    selection artifact (one-term per-row z 1.7), so bbs stands and the
    sentence says which evidence decided it."""
    row = _shape_probe_row("smooth", 714, 14, 200)
    assert row.incumbent == "bbs"
    assert row.chosen == "bbs"
    assert "per-row" in row.sentence


def test_shape_probe_keeps_a_genuine_replacement_on_a_single_holdout():
    """Step truth: btree's margin is real (one-term per-row z 3.6), so
    the replacement still happens on a single holdout."""
    row = _shape_probe_row("step", 702, 2, 364)
    assert row.incumbent == "bbs"
    assert row.chosen == "btree"
    assert "per-row" in row.sentence


def _pair_panel(kind, seed):
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(-1, 1, (n, 4))
    interaction = (1.5 * x[:, 0] * x[:, 1] if kind == "tensor" else 0.0)
    y = np.sin(2 * x[:, 0]) + x[:, 1] + interaction \
        + rng.normal(0, .3, n)
    data = {f"x{j + 1}": x[:, j] for j in range(4)} | {"y": y}
    folds = np.ones((n, 1))
    folds[rng.permutation(n)[: n // 4], 0] = 0.0
    return data, folds


def test_pair_form_picker_blocks_a_margin_only_adoption_on_a_single_holdout(
        monkeypatch):
    """The picker's adoption needs the per-row bar on one holdout.

    With the shape gate in place, the additive panels swept for this
    fix no longer present the picker a live margin case (the shape
    probe's own audit removes the incumbent misfit the margin fed
    on), so this test forces the margin verdict: the tensor form's
    scored risk is doctored 5% under the additive incumbent's, with
    the real formulas and selected steps untouched. The per-row audit
    is then the only route that can veto, and it does (z -0.9 on this
    panel): the winner is "neither" and the recorded z is the
    audit's, not the fold route's inert 0.0.
    """
    import dataclasses

    import mboostrs._autospec as autospec_evidence

    original = autospec_evidence.score_coordinate_candidates

    def doctored(data, response, candidates, family, folds, mstop, nu,
                 **kwargs):
        scores, fits = original(data, response, candidates, family,
                                folds, mstop, nu, **kwargs)
        if set(candidates) == {"additive", "tensor", "joint_tree"} \
                and np.isfinite(scores["additive"].risk) \
                and scores["tensor"].stopping is not None:
            low = scores["additive"].risk * 0.95
            scores = dict(scores)
            scores["tensor"] = dataclasses.replace(
                scores["tensor"], risk=low, fold_risk=(low,))
        return scores, fits

    monkeypatch.setattr(autospec_evidence, "score_coordinate_candidates",
                        doctored)
    data, folds = _pair_panel("additive", 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", family=mb.Gaussian(), budget=150,
                          seed=3, interactions="adopt",
                          preexploration="off", folds=folds)
    form = next(v for k, v in rec.evidence.items()
                if k.startswith("pair_form_"))
    assert form["held_out_gain"] >= 0.02  # the margin alone would adopt
    assert form["winner"] == "neither"
    assert form["paired_z"] != 0.0 and form["paired_z"] < 2.0
    assert "%X%" not in rec.formula


def test_pair_form_picker_adopts_a_genuine_form_on_a_single_holdout():
    """A planted tensor interaction clears the audit (z 4.7 on this
    panel) and is adopted, its per-row z recorded as the deciding
    evidence."""
    data, folds = _pair_panel("tensor", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", family=mb.Gaussian(), budget=150,
                          seed=0, interactions="adopt",
                          preexploration="off", folds=folds)
    form = next(v for k, v in rec.evidence.items()
                if k.startswith("pair_form_"))
    assert form["winner"] == "tensor"
    assert form["paired_z"] >= 2.0
    assert "%X%" in rec.formula


def _surface_panel(seed):
    """The constructive test's secondary panel: two noise columns
    named as a coordinate pair (x, y), additive truth on the other
    two columns, a single supplied holdout of 91 rows."""
    rng = np.random.default_rng(seed)
    n = 364
    X = rng.uniform(0, 1, size=(n, 4))
    resp = (1.5 * np.sin(np.pi * X[:, 2]) + X[:, 3]
            + rng.normal(scale=0.7, size=n))
    data = {"x": X[:, 0], "y": X[:, 1], "x2": X[:, 2], "x3": X[:, 3],
            "resp": resp}
    folds = np.ones((n, 1))
    folds[rng.permutation(n)[:91], 0] = 0.0
    return data, folds


def test_coordinate_surface_blocks_a_valueless_adoption_on_a_single_holdout():
    """Seed 7914025 of the constructive test: the stump over the two
    noise columns clears the 2% margin (+3.8%) on one holdout, where
    the fold z is inert, but its per-row z is 0.95, so the surface
    step refuses it, records the audit's z, and the additive pair
    stands."""
    data, folds = _surface_panel(7914025)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "resp", family=mb.Gaussian(), budget=40,
                          nu=0.1, folds=folds, interactions="adopt",
                          residual_tree=False, seed=7914025)
    surface = rec.evidence["coordinate_surface"]
    assert surface["chosen"] == "btree"
    assert surface["chosen_relative_gain"] >= 0.02
    assert surface["adopted"] is False
    assert surface["chosen_z"] != 0.0 and surface["chosen_z"] < 2.0
    assert "btree(x, y)" not in rec.formula
    assert "bspatial" not in rec.formula


def test_coordinate_surface_keeps_a_genuine_adoption_on_a_single_holdout():
    """A planted product surface over the pair: the tensor clears the
    margin (28%) and the per-row bar (z 3.8), so the adoption stands
    and the recorded z is the audit's, not the fold route's inert
    0.0. interactions="report" keeps the pair picker from taking the
    pair first; the picker has its own tests above."""
    rng = np.random.default_rng(777)
    n = 364
    X = rng.uniform(0, 1, size=(n, 4))
    resp = (3.0 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
            + X[:, 2] + 0.5 * X[:, 3] + rng.normal(scale=0.5, size=n))
    data = {"x": X[:, 0], "y": X[:, 1], "x2": X[:, 2], "x3": X[:, 3],
            "resp": resp}
    folds = np.ones((n, 1))
    folds[rng.permutation(n)[:91], 0] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "resp", family=mb.Gaussian(), budget=40,
                          nu=0.1, folds=folds, interactions="report",
                          residual_tree=False, seed=777)
    surface = rec.evidence["coordinate_surface"]
    assert surface["adopted"] is True
    assert surface["chosen_z"] >= 2.0
    assert "bspatial(x, y" in rec.formula


# ── distributional adoption checks (census C6 and C7) ─────────────────────


def _beta_bell_panel(seed):
    """A Gaussian bell clipped inside (0, 1): the mean family is the
    truth, and BetaLSS is a flexible alternative density with no
    second-parameter signal behind it. n = 364 with a single supplied
    holdout of 91 rows, the constructive test's geometry."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, 364)
    y = np.clip(0.5 + 0.1 * x + rng.normal(0, 0.1, 364), 0.02, 0.98)
    data = {"y": y, "x": x}
    hold = np.random.default_rng(seed + 777_000)
    folds = np.ones((364, 1))
    folds[hold.permutation(364)[:91], 0] = 0.0
    return data, folds


def test_lss_family_is_not_adopted_on_a_single_holdout():
    """Seed 8031006 of the recorded sweep: BetaLSS's held-out log score
    beats Gaussian's by 0.037 nats/row, over the 2% margin, and the
    unfixed check adopted it because the paired standard error is NaN
    at one fold and the clause waived. The adoption is worthless: the
    per-row z of the same comparison is 1.0 on the holdout itself and
    -10.3 on 20,000 fresh rows. With no computable proof the check
    now refuses, and the evidence records the NaN standard error
    behind the refusal."""
    data, folds = _beta_bell_panel(8031006)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", budget=40, mstop=100, seed=8031006,
                          folds=folds, nu=0.1)
    evidence = rec.evidence["family"]
    assert isinstance(rec.family, mb.Gaussian)
    assert evidence.lss_score < evidence.mean_log_score
    assert (evidence.mean_log_score - evidence.lss_score
            > 0.02 * abs(evidence.mean_log_score))
    assert not np.isfinite(evidence.score_difference_se)
    assert "not clearly better, so Gaussian" in evidence.sentence


def test_finished_routes_do_not_adopt_on_a_single_holdout():
    """The finished-route comparison on the same panel: the unfixed
    code returned winner "lss" with both checks passed and both
    standard errors NaN. The comparison is an adoption gate, so the
    uncomputable proofs now fail their checks and the mean route
    stands."""
    from mboostrs._autospec import _lss_candidate, compare_finished_routes

    data, folds = _beta_bell_panel(8031006)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compare_finished_routes(
            data, "y", mean_family=mb.Gaussian(),
            mean_formula="y ~ bbs(x, df=4.0)", mean_mstop=16, mean_nu=0.1,
            lss_family=_lss_candidate("BetaLSS"),
            lss_formula="y ~ bbs(x, df=4.0)", lss_nu=0.1, lss_largest=100,
            folds=folds, fold_scheme=None, env=None)
    assert out is not None
    assert not np.isfinite(out["score_se"])
    assert out["log_score_check"] == "failed"
    assert out["winner"] == "mean"


def _patched_paired_se(monkeypatch, values):
    """Replace the paired-SE helper with a queue of returned values, so
    a test can give one check a proof and the next check none."""
    import mboostrs.explore as explore

    queue = list(values)
    real = explore._paired_se

    def fake(first, second, partitions=1):
        return queue.pop(0) if queue else real(first, second, partitions)

    monkeypatch.setattr(explore, "_paired_se", fake)
    return queue


def test_lss_mean_check_refuses_when_its_se_cannot_be_computed(monkeypatch):
    """The mean check in `_validate_lss` is an adoption gate: when the
    log-score check has a proof but the squared-error difference's
    standard error cannot be computed, the check cannot pass. (At one
    fold the log-score gate refuses first, so this state is reached
    here by giving the first check a finite standard error and the
    second none.)"""
    from mboostrs._autospec import FamilyEvidence, _validate_lss

    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(-2, 2, n)
    mu = 1 / (1 + np.exp(-(0.3 + 0.6 * x)))
    phi = np.exp(2.2 + 0.8 * x)
    data = {"y": rng.beta(mu * phi, (1 - mu) * phi), "x": x}
    folds = mb.cv_weights(n, "kfold", 5, seed=0)
    evidence = FamilyEvidence(
        recommended=mb.Gaussian(), dispersion=float("nan"),
        spread_tracks_mean=float("nan"), n_scored=n, fits=0,
        sentence="every response lies in (0, 1)")
    _patched_paired_se(monkeypatch, [0.001, float("nan")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _validate_lss(
            "BetaLSS", data, "y", "y ~ bbs(x, df=4.0)", folds, 100, 0.1,
            None, evidence, mean_mstop=100)
    assert not hasattr(out.recommended, "param_names")
    assert out.lss_mean_check == "failed"
    assert "cannot be computed" in out.sentence


def test_finished_routes_mean_check_refuses_without_an_se(monkeypatch):
    """The finished-route mean check follows the same rule: a finite
    log-score proof does not carry a mean check whose standard error
    cannot be computed."""
    from mboostrs._autospec import _lss_candidate, compare_finished_routes

    data, _ = _beta_bell_panel(8031006)
    folds = mb.cv_weights(364, "kfold", 5, seed=0)
    _patched_paired_se(monkeypatch, [0.001, float("nan")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compare_finished_routes(
            data, "y", mean_family=mb.Gaussian(),
            mean_formula="y ~ bbs(x, df=4.0)", mean_mstop=16, mean_nu=0.1,
            lss_family=_lss_candidate("BetaLSS"),
            lss_formula="y ~ bbs(x, df=4.0)", lss_nu=0.1, lss_largest=100,
            folds=folds, fold_scheme=None, env=None)
    assert out is not None
    assert out["log_score_check"] == "passed"
    assert out["mean_check"] == "failed"
    assert out["winner"] == "mean"


def test_lss_family_is_still_adopted_when_the_proof_exists():
    """The gate is a proof bar, not a ban: beta data whose precision
    moves with x is adopted on the default five-fold design, where
    the paired standard error exists and the improvement clears it."""
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(-2, 2, n)
    mu = 1 / (1 + np.exp(-(0.3 + 0.6 * x)))
    phi = np.exp(2.2 + 0.8 * x)
    data = {"y": rng.beta(mu * phi, (1 - mu) * phi), "x": x}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(data, "y", budget=50, mstop=100, seed=0)
    assert isinstance(rec.family, mb.BetaLSS)
    assert np.isfinite(rec.evidence["family"].score_difference_se)
