"""Search-wide constraints: forbidden pairs, mandatory terms and heredity (#74).

The truths are a main-effects-plus-pair surface, ``sin(2 x1) + x2 + 1.5 x1 x2``,
and a pure interaction, ``sin(3 x1 x2)``, which has no main effect of x1 or x2.
Under ``heredity="none"`` both searches find the x1 x2 pair. Under
``heredity="strong"`` a search whose specification lacks the main effects of
x1 and x2 does not nominate or adopt the pure pair. ``autospec`` is not in
that group: it keeps every column's main term in its formula during adoption,
so the parents are present and strong heredity admits the pair there.
"""

import json
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.explore import pair_term_for


def _frame(kind, seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 4))
    if kind == "pure":
        f = np.sin(3 * X[:, 0] * X[:, 1])
    else:
        f = np.sin(2 * X[:, 0]) + X[:, 1] + 1.5 * X[:, 0] * X[:, 1]
    y = f + rng.normal(0, 0.3, n)
    return {"x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3], "y": y}


def _is_pair(names):
    return set(names) == {"x1", "x2"}


def _discover(kind, **kwargs):
    kwargs.setdefault("spread", False)
    kwargs.setdefault("seed", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.discover(pl.DataFrame(_frame(kind)), "y", **kwargs)


def _session(kind="main", **kwargs):
    kwargs.setdefault("family", mb.Gaussian())
    kwargs.setdefault("folds", 4)
    kwargs.setdefault("mstop", 120)
    kwargs.setdefault("budget", mb.CostBudget(fits=200))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.open_search(_frame(kind), "y", **kwargs)


# --- discover -------------------------------------------------------------

def test_discover_adopts_the_pair_by_default_and_never_under_forbidden_pairs():
    assert any(_is_pair(r["variables"]) for r in _discover("main").interactions)
    result = _discover("main", forbidden_pairs=[("x2", "x1")])
    assert not any(_is_pair(r["variables"]) for r in result.interactions)
    assert "forbidden pair(s) not nominated" in " ".join(
        r["reason"] for r in result.log if r["stage"] == "interactions")
    assert not any("x1" in t and "x2" in t and ("%X%" in t or "by=" in t)
                   for t in result.formula.split(" + "))


def test_discover_mandatory_term_survives_backward_elimination():
    assert "x4" not in _discover("main").formula
    result = _discover("main", mandatory_terms=["x4"])
    assert "x4" in result.formula
    assert any(r["reason"].startswith("kept: mandatory") for r in result.log
               if r["candidate"] and r["candidate"].startswith("bbs(x4"))


def test_strong_heredity_misses_the_pure_pair_and_none_finds_it():
    found = _discover("pure", heredity="none")
    missed = _discover("pure", heredity="strong")
    assert any(_is_pair(r["variables"]) for r in found.interactions)
    assert missed.interactions == []
    reasons = [r["reason"] for r in missed.log if r["stage"] == "interactions"]
    assert any("heredity strong needs a main effect of x1, x2" in reason for reason in reasons)


def test_strong_and_weak_keep_a_pair_whose_main_effects_are_in_the_model():
    for rule in ("strong", "weak"):
        result = _discover("main", heredity=rule)
        assert any(_is_pair(r["variables"]) for r in result.interactions), rule


def test_discover_records_the_rule_in_its_log_and_result():
    result = _discover("main", heredity="strong", forbidden_pairs=[("x3", "x4")],
                       mandatory_terms=["x4"])
    rule = next(r["reason"] for r in result.log if r["stage"] == "rule")
    assert rule.startswith("rule in force: mu heredity strong")
    assert "forbidden pairs (mu): x3 x x4" in rule
    assert "mandatory (mu): x4" in rule
    assert result.constraints["heredity"] == {"mu": "strong"}
    assert result.constraints["forbidden_pairs"] == {"mu": [["x3", "x4"]]}
    assert json.loads(json.dumps(result.to_dict()))["constraints"] == result.constraints


def test_discover_sigma_mandatory_term_is_included_without_the_spread_test():
    rng = np.random.default_rng(3)
    n = 300
    frame = pl.DataFrame({"x1": rng.uniform(-1, 1, n), "x2": rng.uniform(-1, 1, n),
                          "x3": rng.uniform(-1, 1, n)})
    frame = frame.with_columns(y=pl.col("x1") + pl.Series(rng.normal(0, 0.3, n)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mb.discover(frame, "y", seed=0, mandatory_terms={"sigma": ["x3"]})
    assert result.spread_formula is not None and "x3" in result.spread_formula
    assert result.spread_drivers[0]["mandatory"] is True
    assert any(r["stage"] == "spread" and "mandatory" in r["reason"] for r in result.log)


def test_discover_candidates_restrict_the_mean_columns():
    result = _discover("main", candidates={"mu": ["x1", "x3"]})
    assert "x2" not in result.formula and "x4" not in result.formula
    assert result.constraints["candidates"]["mu"] == ["x1", "x3"]


def test_discover_refuses_constraints_it_cannot_apply():
    frame = pl.DataFrame(_frame("main"))
    with pytest.raises(ValueError, match="accepts the keys"):
        mb.discover(frame, "y", spread=False, forbidden_pairs={"sigma": [("x1", "x2")]})
    with pytest.raises(ValueError, match="heredity must be"):
        mb.discover(frame, "y", spread=False, heredity="sometimes")
    with pytest.raises(ValueError, match="response"):
        mb.discover(frame, "y", spread=False, forbidden_pairs=[("x1", "y")])
    with pytest.raises(ValueError, match="not a column"):
        mb.discover(frame, "y", spread=False, mandatory_terms=["nope"])
    with pytest.raises(ValueError, match="spread"):
        mb.discover(frame, "y", spread=False, mandatory_terms={"sigma": ["x3"]})


# --- SearchSession ----------------------------------------------------------

def test_session_never_nominates_or_accepts_a_forbidden_pair():
    search = _session(forbidden_pairs=[("x1", "x2")])
    search.run("set_spec", rhs="bbs(x1) + bbs(x2) + bols(x3)")
    term = pair_term_for("x1", "x2", data=_frame("main"))
    with pytest.raises(ValueError, match="forbidden pair"):
        search.run("compare", candidates={"pair": f"bbs(x1) + bbs(x2) + bols(x3) + {term}"})
    with pytest.raises(ValueError, match="forbidden pair"):
        search.run("set_spec", rhs=f"bbs(x1) + bbs(x2) + {term}")
    search.run("fit")
    leads = search.run("interaction_screen")
    assert not any(_is_pair([lead.left, lead.right]) for lead in leads.value)
    assert leads.summary["constraint_barred"].get("x1 x x2") == "forbidden pair x1 x x2"


def test_session_mandatory_term_survives_every_drop():
    search = _session(mandatory_terms=["x3"])
    with pytest.raises(ValueError, match="mandatory"):
        search.run("set_spec", rhs="bbs(x1) + bbs(x2)")
    search.run("set_spec", rhs="bbs(x1) + bbs(x2) + bols(x3)")
    with pytest.raises(ValueError, match="mandatory"):
        search.run("compare", candidates={"drop_noise": "bbs(x1) + bbs(x2)"})


def test_strong_heredity_refuses_the_pure_pair_and_none_adopts_it():
    spec = "bols(x3) + bols(x4)"
    term = pair_term_for("x1", "x2", data=_frame("pure"))

    strong = _session("pure", heredity="strong")
    strong.run("set_spec", rhs=spec)
    strong.run("fit")
    leads = strong.run("interaction_screen").value
    assert not any(_is_pair([lead.left, lead.right]) for lead in leads)
    with pytest.raises(ValueError, match="heredity strong"):
        strong.run("compare", candidates={"pair": f"{spec} + {term}"})

    none = _session("pure", heredity="none")
    none.run("set_spec", rhs=spec)
    none.run("fit")
    leads = none.run("interaction_screen").value
    assert any(_is_pair([lead.left, lead.right]) for lead in leads[:6])
    compared = none.run("compare", candidates={"pair": f"{spec} + {term}"})
    change, se = compared.value[1]["change"], compared.value[1]["change se"]
    assert change < -2 * se
    adopted = none.run("adopt", comparison_id=compared.evidence_id, candidate="pair")
    assert term in adopted.summary["spec"]


def test_strong_heredity_protects_the_parents_of_an_adopted_pair():
    spec = "bbs(x1) + bbs(x2) + bols(x3)"
    term = pair_term_for("x1", "x2", data=_frame("main"))
    search = _session("main", heredity="strong")
    search.run("set_spec", rhs=spec)
    compared = search.run("compare", candidates={"pair": f"{spec} + {term}"})
    search.run("adopt", comparison_id=compared.evidence_id, candidate="pair")
    with pytest.raises(ValueError, match="heredity strong"):
        search.run("compare", candidates={"drop_x1": f"bols(x2) + bols(x3) + {term}"})


def test_session_records_the_rule_in_state_transcript_and_checkpoint():
    rule = {"heredity": "strong", "forbidden_pairs": [["x1", "x2"]],
            "mandatory_terms": ["x3"]}
    search = _session(heredity="strong", forbidden_pairs=[("x2", "x1")],
                      mandatory_terms=["x3"])
    assert search.state()["constraints"] == rule
    assert search.export_transcript()["constraints"] == rule
    saved = json.loads(json.dumps(search.checkpoint()))
    assert saved["constraints"] == rule
    resumed = mb.SearchSession.resume(_frame("main"), saved, family=mb.Gaussian())
    assert resumed.state()["constraints"] == rule
    assert resumed.state()["state_id"] == search.state()["state_id"]


def test_session_default_rule_is_stated_and_leaves_the_state_unchanged():
    search = _session()
    assert search.state()["constraints"] == {"heredity": "none", "forbidden_pairs": [],
                                             "mandatory_terms": []}


def test_session_refuses_bad_constraints_at_open():
    with pytest.raises(ValueError, match="heredity must be"):
        _session(heredity="always")
    with pytest.raises(ValueError, match="not a column"):
        _session(forbidden_pairs=[("x1", "nope")])
    with pytest.raises(ValueError, match="response"):
        _session(mandatory_terms=["y"])


# --- autospec -------------------------------------------------------------

def test_autospec_states_the_rule_it_applied_on_every_run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = mb.autospec(_frame("main"), "y", seed=0)
    assert rec.evidence["interaction_rule"] == {"heredity": "none", "forbid": (),
                                                "mandatory": ()}
    assert "interaction rule: heredity none" in rec.report()
