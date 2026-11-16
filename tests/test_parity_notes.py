"""Documented deviations reach the user who is affected by one.

A user reaching for `Gehan`, `Bkernel` or an AFT `logLik` has a meaning in
mind, taken from `mboost` or from the literature. Sometimes mboostrs computes
something else; sometimes `mboost` computes something else too and mboostrs
reproduces it, so the object differs from its *name* rather than from
upstream. Both mislead the same way.

They were recorded in a generated appendix, which a user has to think to look
for — and the point of these notes is that the user does not know to. They are
attached to the family or learner that carries them, surfaced by
`model.parity_notes()`, and named in `summary()`.

Each note carries four things, and the fourth is the one that changes what a
reader should do: whether upstream shares the behaviour. A deviation mboostrs
invented might be a bug to report; one shared with upstream is a property of
the method as implemented everywhere, and switching packages will not escape
it.
"""

import numpy as np
import pytest
from mboostrs._parity_notes import (
    FAMILY_NOTES,
    GLOBAL_NOTES,
    LEARNER_NOTES,
    ParityNote,
    notes_for,
)

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    return {
        "x": x,
        "y": 2.0 * x + rng.normal(size=n),
        "time": np.exp(0.5 * x + rng.exponential(size=n)),
        "event": (rng.uniform(size=n) < 0.7).astype(float),
    }


# ─── the notes themselves ──────────────────────────────────────────


def test_every_note_says_all_four_things():
    """A note missing its consequence leaves the reader to work out whether it
    matters, which is the work the note exists to save."""
    everything = [
        *(note for notes in FAMILY_NOTES.values() for note in notes),
        *(note for notes in LEARNER_NOTES.values() for note in notes),
        *GLOBAL_NOTES,
    ]
    assert len(everything) > 10, "expected a non-trivial set of notes"
    for note in everything:
        assert note.name and note.name == note.name.strip(), note
        assert len(note.mechanism) > 30, f"{note.name}: mechanism too thin"
        assert len(note.consequence) > 30, f"{note.name}: consequence too thin"
        assert isinstance(note.shared_with_upstream, bool), note.name


def test_names_are_unique():
    """A note is meant to be citable by name, so two must not share one."""
    names = [
        note.name
        for notes in (*FAMILY_NOTES.values(), *LEARNER_NOTES.values())
        for note in notes
    ] + [note.name for note in GLOBAL_NOTES]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, duplicates


def test_a_note_names_its_origin_in_its_rendered_form():
    shared = ParityNote("n", "m" * 40, "c" * 40, shared_with_upstream=True)
    mine = ParityNote("n", "m" * 40, "c" * 40, shared_with_upstream=False)
    assert "shared with upstream" in shared.format_text()
    assert "mboostrs-specific" in mine.format_text()


def test_the_notes_that_cite_evidence_cite_a_test_that_exists():
    """A note claiming a measurement should point at the measurement."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    cited = [
        note
        for notes in (*FAMILY_NOTES.values(), *LEARNER_NOTES.values())
        for note in notes
        if note.evidence
    ]
    assert cited, "expected some notes to cite their evidence"
    for note in cited:
        path, _, name = note.evidence.partition("::")
        assert (root / path).exists(), f"{note.name} cites a missing file: {path}"
        if name:
            source = (root / path).read_text()
            assert f"def {name}(" in source, (
                f"{note.name} cites {name}, which {path} does not define"
            )


# ─── they reach the fit ────────────────────────────────────────────


def test_an_aft_fit_carries_the_log_scale_note(data):
    model = mb.glmboost(
        "Surv(time, event) ~ x",
        data={k: data[k] for k in ("x", "time", "event")},
        family=mb.Weibull(), control=mb.boost_control(20, 0.1),
    )
    names = [note.name for note in notes_for(model)]
    assert "aft likelihood is on the log-time scale" in names


def test_a_tree_learner_carries_its_split_criterion_note(data):
    model = mb.gamboost("y ~ btree(x)", data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(15, 0.1))
    names = [note.name for note in notes_for(model)]
    assert "btree splits on squared error, not conditional inference" in names


def test_notes_cover_family_and_learners_together(data):
    model = mb.gamboost("y ~ btree(x) + bns(x)",
                        data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(15, 0.1))
    names = {note.name for note in notes_for(model)}
    assert "gaussian logLik is a real log-likelihood" in names   # family
    assert "btree splits on squared error, not conditional inference" in names
    assert "bns is ridge-penalized, not unpenalized" in names


def test_a_learner_kind_is_reported_once_however_many_terms_use_it(data):
    model = mb.gamboost("y ~ btree(x) + btree(x)",
                        data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(10, 0.1))
    names = [note.name for note in notes_for(model)]
    assert names.count(
        "btree splits on squared error, not conditional inference"
    ) == 1


def test_global_notes_are_opt_in(data):
    model = mb.glmboost("y ~ x", data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(10, 0.1))
    without = {note.name for note in notes_for(model)}
    with_global = {note.name for note in notes_for(model, include_global=True)}
    assert "continue_to refits from zero on routes without a retained residual" not in without
    assert "continue_to refits from zero on routes without a retained residual" in with_global


# ─── the model surface ─────────────────────────────────────────────


def test_parity_notes_is_reachable_from_a_standard_model(data):
    model = mb.glmboost("y ~ x", data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(10, 0.1))
    records = model.parity_notes(output="records")
    assert records and set(records[0]) == {
        "name", "mechanism", "consequence", "shared_with_upstream", "evidence",
    }


def test_parity_notes_is_reachable_from_a_distributional_model(data):
    model = mb.glmboostLSS("y ~ x", data={"x": data["x"], "y": data["y"]},
                           families=mb.GaussianLSS(),
                           control=mb.boost_control(20, 0.1))
    assert isinstance(model.parity_notes(output="text"), str)


@pytest.mark.parametrize("output", ["polars", "pandas", "records", "text"])
def test_parity_notes_honours_output(data, output):
    model = mb.glmboost("y ~ x", data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(10, 0.1))
    result = model.parity_notes(output=output)
    expected = {
        "polars": "polars", "pandas": "pandas",
        "records": "builtins", "text": "builtins",
    }
    assert type(result).__module__.split(".")[0] == expected[output]


def test_an_unknown_output_is_rejected(data):
    model = mb.glmboost("y ~ x", data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(10, 0.1))
    with pytest.raises(ValueError, match="output must be"):
        model.parity_notes(output="frame")


def test_a_fit_with_no_deviations_says_so():
    """`Poisson` with `bols` carries none, and the message says that rather
    than returning an empty table with no explanation."""
    rng = np.random.default_rng(5)
    n = 120
    x = rng.normal(size=n)
    data = {"x": x, "y": rng.poisson(np.exp(0.5 * x)).astype(float)}
    model = mb.gamboost("y ~ bols(x)", data=data, family=mb.Poisson(),
                        control=mb.boost_control(15, 0.1))
    assert notes_for(model) == []
    assert "No documented parity deviations" in model.parity_notes(output="text")


# ─── summary() names them ──────────────────────────────────────────


def test_summary_points_at_the_notes(data):
    """A deviation recorded only in an appendix is one the user has to think
    to look for, and the point is that they do not know to."""
    model = mb.gamboost("y ~ btree(x)", data={"x": data["x"], "y": data["y"]},
                        control=mb.boost_control(15, 0.1))
    text = model.summary().as_text()
    assert "parity note" in text
    assert "parity_notes()" in text


def test_summary_stays_quiet_when_there_is_nothing_to_say():
    rng = np.random.default_rng(5)
    n = 120
    x = rng.normal(size=n)
    data = {"x": x, "y": rng.poisson(np.exp(0.5 * x)).astype(float)}
    model = mb.gamboost("y ~ bols(x)", data=data, family=mb.Poisson(),
                        control=mb.boost_control(15, 0.1))
    assert "parity note" not in model.summary().as_text()


# ─── the notes agree with the code ─────────────────────────────────


def test_the_refusals_the_notes_promise_actually_happen(data):
    """Two notes say `logLik()` refuses. If it did not, the note would be
    describing a version of the package that no longer exists. (A third,
    GammaReg's, said the shape was never estimated; it is, and `logLik()`
    now returns the Gamma log-likelihood at the estimated shape.)"""
    survival = {k: data[k] for k in ("x", "time", "event")}
    for family, fragment in [
        (mb.Gehan(), "ranking objective"),
        (mb.Cindex(), "ranking objective"),
    ]:
        model = mb.glmboost("Surv(time, event) ~ x", data=survival,
                            family=family, control=mb.boost_control(15, 0.1))
        with pytest.raises(NotImplementedError, match=fragment):
            model.logLik()

    gamma = {"x": data["x"], "y": np.abs(data["y"]) + 0.5}
    model = mb.glmboost("y ~ x", data=gamma, family=mb.GammaReg(),
                        control=mb.boost_control(15, 0.1))
    assert np.isfinite(model.logLik())


def test_the_multinomial_note_matches_what_the_family_does():
    rng = np.random.default_rng(9)
    n = 150
    data = {"x": rng.normal(size=n),
            "y": rng.integers(0, 3, size=n).astype(float)}
    model = mb.gamboost("y ~ bols(x)", data=data, family=mb.Multinomial(),
                        control=mb.boost_control(15, 0.1))
    names = {note.name for note in notes_for(model)}
    assert "multinomial risk excludes the reference class" in names
    assert "multinomial is gamboost-only" in names
    with pytest.raises(NotImplementedError, match="reference class"):
        model.logLik()


def test_note_content_agrees_with_the_executable_contract():
    multinomial = FAMILY_NOTES["Multinomial"][0]
    assert "refuse" in multinomial.consequence
    gradient_note = next(
        note for note in FAMILY_NOTES["Multinomial"]
        if "negative gradient" in note.name
    )
    assert "derivative" in gradient_note.name
    assert "finite" in gradient_note.mechanism.lower()

    auc = next(note for note in FAMILY_NOTES["AUC"]
               if note.name.startswith("weighted auc"))
    assert "fit runs" in auc.consequence
    assert not auc.shared_with_upstream

    gamma = FAMILY_NOTES["GammaReg"][0]
    assert "GammaLSS" in gamma.consequence
    assert "NBinomial" not in gamma.consequence


def test_lss_collects_notes_from_each_parameters_learners(data):
    model = mb.gamboostLSS(
        "y ~ btree(x)", data={"x": data["x"], "y": data["y"]},
        families=mb.GaussianLSS(), control=mb.boost_control(10, 0.1),
    )
    names = {note.name for note in notes_for(model)}
    assert "btree splits on squared error, not conditional inference" in names
    assert "parity note" in model.summary().as_text()


def test_artifact_carries_the_notes_visible_when_it_was_written(data, tmp_path):
    import json

    model = mb.gamboost(
        "y ~ btree(x)", data={"x": data["x"], "y": data["y"]},
        control=mb.boost_control(10, 0.1),
    )
    path = mb.save_model(model, tmp_path / "model.json")
    document = json.loads(path.read_text())
    names = {note["name"] for note in document["parity_notes"]}
    assert "btree splits on squared error, not conditional inference" in names
    assert "continue_to refits from zero on routes without a retained residual" in names
