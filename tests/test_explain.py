"""Tests for the language-model-facing model description."""
import numpy as np
import pytest
from mboostrs.explain import DEFAULT_LLM, SECTIONS, _effect_precision, _shape_note

import mboostrs as mboost


@pytest.fixture
def gam_model():
    rng = np.random.default_rng(0)
    n = 200
    x, z, unused = rng.uniform(0, 10, n), rng.normal(size=n), rng.normal(size=n)
    y = np.sin(x) + 0.8 * z + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x": x, "z": z, "unused": unused}
    return mboost.gamboost(
        "y ~ bbs(x) + bols(z) + bbs(unused)", data=data,
        control=mboost.boost_control(120, 0.1),
    )


@pytest.fixture
def glm_model():
    rng = np.random.default_rng(1)
    n = 150
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.4, size=n)
    return mboost.glmboost(
        "y ~ x1 + x2 + x3", data={"y": y, "x1": x1, "x2": x2, "x3": x3},
        control=mboost.boost_control(100, 0.1),
    )


@pytest.fixture
def factor_model():
    rng = np.random.default_rng(2)
    n = 200
    g = np.array(["low", "mid", "high"])[rng.integers(0, 3, n)]
    x = rng.uniform(0, 1, n)
    y = (g == "high") * 1.2 + 0.5 * x + rng.normal(scale=0.2, size=n)
    return mboost.gamboost(
        "y ~ brandom(g) + bbs(x)", data={"y": y, "g": g, "x": x},
        control=mboost.boost_control(80, 0.1),
    )


@pytest.fixture
def interaction_model():
    rng = np.random.default_rng(3)
    n = 250
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(6 * x1) * x2 + rng.normal(scale=0.2, size=n)
    return mboost.gamboost(
        "y ~ bbs(x1) %X% bbs(x2)", data={"y": y, "x1": x1, "x2": x2},
        control=mboost.boost_control(80, 0.1),
    )


@pytest.fixture
def lss_model():
    rng = np.random.default_rng(4)
    n = 250
    x = rng.uniform(0, 1, n)
    y = rng.normal(2.0 * x, 0.3 + 0.5 * x)
    return mboost.gamboostLSS(
        "y ~ bbs(x)", data={"y": y, "x": x}, families=mboost.GaussianLSS(),
        control=mboost.boost_control(60, 0.1),
    )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_prompt_carries_every_section(gam_model):
    text = gam_model.llm_explain_prompt()

    assert "How to read what follows" in text          # reading
    assert "Gaussian boosting results" in text         # facts
    assert "Response scale:" in text
    assert "Base learners, ranked:" in text            # importance
    assert "Effects (" in text                         # effects
    assert "Limits of this description" in text        # caveats
    assert "Write the brief a statistician" in text      # task


def test_include_selects_sections(gam_model):
    text = gam_model.llm_explain_prompt(include=("effects",))

    assert text.startswith("Effects (")
    assert "How to read what follows" not in text
    assert "Limits of this description" not in text


def test_unknown_section_and_output_are_refused(gam_model):
    with pytest.raises(ValueError, match="unknown section"):
        gam_model.llm_explain_prompt(include=("effects", "nonsense"))
    with pytest.raises(ValueError, match="output must be"):
        gam_model.llm_explain_prompt(output="json")
    with pytest.raises(ValueError, match="unknown base learner"):
        gam_model.llm_explain_prompt(which="not_a_column")


def test_sections_constant_matches_the_default(gam_model):
    # `include=SECTIONS` is the default, so the two must render identically.
    assert gam_model.llm_explain_prompt() == gam_model.llm_explain_prompt(
        include=SECTIONS
    )


def test_messages_output_is_a_chat_exchange(gam_model):
    messages = gam_model.llm_explain_prompt(output="messages")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "expert statistician" in messages[0]["content"]
    assert messages[1]["content"] == gam_model.llm_explain_prompt()


def test_description_is_deterministic(gam_model):
    first = gam_model.llm_explain_prompt(include=("importance", "effects"))
    second = gam_model.llm_explain_prompt(include=("importance", "effects"))

    assert first == second


def test_module_function_and_method_agree(gam_model):
    assert mboost.llm_explain_prompt(gam_model) == gam_model.llm_explain_prompt()


# ---------------------------------------------------------------------------
# what the effect blocks say
# ---------------------------------------------------------------------------

def test_smooth_effect_reports_grid_shape_and_range(gam_model):
    text = gam_model.llm_explain_prompt(include=("effects",), n_points=8)

    assert "[bbs: B-spline basis of degree 3 on 20 inner knots, df=4" in text
    assert "second-difference penalty" in text
    assert "Covariate: x (observed range" in text
    assert "Contribution: {" in text
    assert "Shape:" in text
    # An equispaced grid of eight points gives eight key/value pairs.
    block = [b for b in text.split("\n\n") if "Covariate: x " in b][0]
    pairs = [line for line in block.split("\n") if line.startswith("Contribution: {")][0]
    assert pairs.count(":") == 9  # eight pairs plus the "Contribution:" label


def test_never_selected_learner_is_named_as_contributing_zero(gam_model):
    text = gam_model.llm_explain_prompt(include=("importance",))

    assert "unused — importance 0.0% — never selected (contributes zero)" in text


def test_top_keeps_the_most_important_learners(gam_model):
    text = gam_model.llm_explain_prompt(include=("effects",), top=1)

    assert text.startswith("Effects (1 of 2 selected learners")
    assert text.count("Learner: ") == 1


def test_importance_is_keyed_per_learner_not_per_name():
    # Two learners on one column share a name. `varimp` collapses them into one
    # dictionary key; the description must still tell them apart.
    rng = np.random.default_rng(5)
    n = 200
    x = rng.uniform(0, 10, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    fit = mboost.gamboost(
        "y ~ bbs(x) + bmono(x)", data={"y": y, "x": x},
        control=mboost.boost_control(80, 0.1),
    )

    text = fit.llm_explain_prompt(include=("importance",))

    assert "x (bbs)" in text
    assert "x (bmono)" in text
    shares = [
        float(part.split("%")[0])
        for part in text.split("importance ")[1:]
    ]
    assert sum(shares) == pytest.approx(100.0, abs=0.2)


def test_varimp_keeps_same_name_learners_apart():
    # Regression (2026-08-13 defect note): the reduction dict overwrote
    # same-named learners, so bbs(x1) could report 0% importance while
    # another x1 learner that was never selected determined its value. Each
    # learner now has its own entry, as in mboost, and `variable_names`
    # aggregates them by variable.
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    x3 = rng.uniform(0, 10, n)
    y = np.sin(x1) + 0.7 * x2 + rng.normal(scale=0.3, size=n)
    fit = mboost.gamboost(
        "y ~ bbs(x1) + bols(x2) + bbs(x3) + bmono(x1)",
        data={"y": y, "x1": x1, "x2": x2, "x3": x3},
        control=mboost.boost_control(150, 0.1),
    )
    vi = fit.varimp(percent=False)

    # Ground truth: the per-index attribution summed over x1's learners.
    riskdiff = -np.diff(fit._risk) / n
    per_index = np.zeros(len(fit._learner_names))
    np.add.at(per_index, np.asarray(fit._selected), riskdiff)
    names = fit._learner_names
    keys = fit.base_learner_names()
    assert "x1 (bbs)" in keys and "x1 (bmono)" in keys
    for index, key in enumerate(keys):
        assert vi["reduction"][key] == pytest.approx(per_index[index], rel=1e-12)
    expected_x1 = sum(per_index[i] for i, nm in enumerate(names) if nm == "x1")
    assert expected_x1 > 0
    by_variable = sum(v for k, v in vi["reduction"].items()
                      if vi["variable_names"][k] == "x1")
    assert by_variable == pytest.approx(expected_x1, rel=1e-12)

    # Selection frequencies still partition the path across names.
    assert sum(vi["selfreq"].values()) == pytest.approx(1.0)

    # The summary table keeps same-named learners apart, one row per learner
    # tagged with its kind, instead of merging them under the first kind.
    text = str(fit.summary())
    selected_kinds = {
        fit._learner_types[i] for i in set(fit._selected) if names[i] == "x1"
    }
    for kind in selected_kinds:
        assert f"x1 [{kind}]" in text


def test_factor_effect_reports_level_labels(factor_model):
    text = factor_model.llm_explain_prompt(include=("effects",))

    assert "Covariate: g (categorical, 3 levels)" in text
    for level in ("low", "mid", "high"):
        assert f'"{level}"' in text
    # A level has no ordering, so no monotonicity claim is made about it.
    block = [b for b in text.split("\n\n") if "Covariate: g " in b][0]
    assert "Shape:" not in block


def test_interaction_reports_the_term_and_a_grid(interaction_model):
    text = interaction_model.llm_explain_prompt(n_points=6, cells=4)

    assert "Learner: bbs(x1) %X% bbs(x2)" in text
    assert "row-wise tensor product" in text
    assert "rows are x1, columns are x2" in text
    # The summary tables name the term as written; the derived column the
    # engine fits through appears nowhere in the prompt.
    assert any(line.startswith("bbs(x1) %X% bbs(x2) ") for line in text.splitlines())
    assert "__ix0" not in text
    grid = text.split("rows are x1, columns are x2:\n")[1].split("\n\n")[0]
    assert grid.count("\n") == 4  # a header line plus four rows


def test_two_covariate_learner_reports_both_ranges():
    rng = np.random.default_rng(6)
    n = 250
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = x1 * x2 + rng.normal(scale=0.2, size=n)
    fit = mboost.gamboost(
        "y ~ btree(x1, x2)", data={"y": y, "x1": x1, "x2": x2},
        control=mboost.boost_control(60, 0.1),
    )

    text = fit.llm_explain_prompt(include=("effects",), cells=3)

    assert "Learner: btree(x1, x2)" in text
    assert "regression tree, maxdepth=1" in text
    assert "Covariates: x1 in [" in text and "x2 in [" in text


def test_glmboost_reports_a_coefficient_per_term(glm_model):
    text = glm_model.llm_explain_prompt(include=("effects",), top=1)

    assert "[linear, unpenalized]" in text
    assert "Coefficient:" in text
    assert "contribution = " in text
    assert "Covariate range:" in text


# ---------------------------------------------------------------------------
# context the caller supplies
# ---------------------------------------------------------------------------

def test_supplied_context_is_included(gam_model):
    text = gam_model.llm_explain_prompt(
        dataset_description="Simulated data, n=200.",
        response_description="y is a lab measurement in mg/dL.",
        task="List the three most important covariates.",
    )

    assert "Simulated data, n=200." in text
    assert "y is a lab measurement in mg/dL." in text
    assert text.rstrip().endswith("List the three most important covariates.")
    assert "Write the brief a statistician" not in text


def test_missing_cross_validation_is_stated_and_a_result_is_used(gam_model):
    without = gam_model.llm_explain_prompt(include=("caveats",))
    assert "No cross-validated risk was supplied" in without

    cv = gam_model.cvrisk(folds=mboost.cv_weights(200, type="kfold", B=3, seed=1))
    with_cv = gam_model.llm_explain_prompt(include=("facts", "caveats"), cv=cv)
    assert "Cross-validation" in with_cv
    assert "Optimal mstop" in with_cv
    assert "No cross-validated risk was supplied" not in with_cv


def test_stability_selection_reports_a_frequency_not_a_probability(glm_model):
    """`phat` is the fraction of subsamples a learner was selected in.

    Calling it a probability invites the reader to treat it as the chance the
    learner belongs in the model, which stability selection does not claim.
    PFER likewise bounds a *count* of false selections, not a rate.
    """
    result = glm_model.stabsel(q=2, PFER=1)

    text = glm_model.llm_explain_prompt(include=("importance",), stabsel=result)

    assert "of stability-selection subsamples" in text
    assert "probability" not in text.lower().split("pfer")[0].split(
        "stability selection:"
    )[-1] or "not the probability" in text
    assert "expected *number* of falsely selected" in text
    assert "per-family error rate" not in text


# ---------------------------------------------------------------------------
# distributional fits
# ---------------------------------------------------------------------------

def test_lss_describes_every_parameter_with_its_link(lss_model):
    text = lss_model.llm_explain_prompt(n_points=5)

    assert "=== Distribution parameter: mu (identity link" in text
    assert "=== Distribution parameter: sigma (log link" in text
    assert text.count("Effects (") == 2


def test_lss_states_why_importance_is_missing_for_a_cyclic_fit(lss_model):
    text = lss_model.llm_explain_prompt(include=("importance",))

    assert "Risk-reduction shares are unavailable" in text
    assert "method='noncyclic'" in text
    assert "importance " not in text


def test_lss_parameter_can_be_selected(lss_model):
    text = lss_model.llm_explain_prompt(parameter="sigma", include=("effects",))

    assert "Distribution parameter: sigma" in text
    assert "Distribution parameter: mu" not in text


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 100.0], 1),       # a hundredth of the span is already 1
        ([0.0, 40.0], 2),        # a span of tens needs two decimals
        ([0.0, 1.0], 3),         # a span of one needs three
        ([0.0, 0.004], 6),       # a span of thousandths needs six
        ([2.0, 2.0], 3),         # a flat effect has no span to scale by
    ],
)
def test_effect_precision_follows_the_span(values, expected):
    assert _effect_precision(values) == expected


def test_shape_note_names_direction_and_turning_points():
    x = np.linspace(0, 1, 21)

    assert "monotone increasing" in _shape_note(x, x)
    assert "monotone decreasing" in _shape_note(x, -x)
    assert _shape_note(x, np.zeros_like(x)) == "flat"

    hump = -((x - 0.5) ** 2)
    note = _shape_note(x, hump)
    assert "1 turning point" in note
    assert "highest near 0.5" in note


# ---------------------------------------------------------------------------
# the three-turn reasoning sequence
# ---------------------------------------------------------------------------

def test_turns_output_is_a_system_plus_three_user_turns(gam_model):
    plan = gam_model.llm_explain_prompt(output="turns", num_sentences=4)

    assert set(plan) == {"system", "turns"}
    assert "expert statistician" in plan["system"]
    assert len(plan["turns"]) == 3
    # The first turn is the description; the other two probe and then compress.
    assert plan["turns"][0] == gam_model.llm_explain_prompt()
    assert "surprising or" in plan["turns"][1]
    assert "at most 4 sentences" in plan["turns"][2]


def test_instructions_ask_for_findings_not_a_restatement(gam_model):
    """Issue #181, second pass: replies still walked the effects one by one
    even after the task asked for findings, because a lone prohibition loses
    to the prompt's own table structure. The instructions now give the reply
    an explicit four-part skeleton (findings, surprising patterns, what could
    mislead, next check), cap its length, make the failure mode explicit, and
    the surprises probe commits to a judgment instead of listing possibilities
    without ranking them."""
    plan = gam_model.llm_explain_prompt(output="turns", num_sentences=5)
    task, surprises, condense = plan["turns"][0], plan["turns"][1], plan["turns"][2]
    for part in ("**Findings**", "**Surprising patterns**",
                 "**What could mislead**", "**Next check**"):
        assert part in task
    assert "under 300 words" in task
    assert "walks the effects one by one" in task
    for issue in ("sparse or extrapolated", "same signal", "stopping iteration"):
        assert issue in task
    assert "commit to a judgment" in surprises
    assert "most likely on subject-matter grounds" in surprises
    assert "wasted sentence" in condense
    assert "next check" in condense
    for retired in ("Describe what this model says", "shape and size of each effect",
                    "effects that carry the model", "Lead with the two or three findings",
                    "do not rank them"):
        assert retired not in task + surprises + condense


# ---------------------------------------------------------------------------
# sending the prompt
# ---------------------------------------------------------------------------

class _FakeCompletions:
    """The `chat.completions.create` surface of an OpenAI-compatible client."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, *, model, messages, max_tokens, temperature):
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages],
             "max_tokens": max_tokens, "temperature": temperature}
        )
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


class _FakeClient:
    def __init__(self, *replies):
        self.chat = type("chat", (), {})()
        self.chat.completions = _FakeCompletions(replies)

    @property
    def calls(self):
        return self.chat.completions.calls


def test_describe_sends_one_request_and_returns_the_reply(gam_model):
    client = _FakeClient("x drives the response.")

    reply = gam_model.llm_describe(client=client, llm="z-ai/glm-5.2")

    assert reply == "x drives the response."
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "z-ai/glm-5.2"
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "Base learners, ranked:" in call["messages"][1]["content"]


def test_describe_with_reasoning_runs_the_three_turns_in_one_conversation(gam_model):
    client = _FakeClient("first pass", "the surprises", "the summary")

    conversation = gam_model.llm_describe(
        client=client, reasoning=True, num_sentences=3, return_conversation=True,
    )

    assert len(client.calls) == 3
    # Each request carries every earlier turn, so the model sees its own replies.
    assert [m["role"] for m in client.calls[-1]["messages"]] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert [m["role"] for m in conversation] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant",
    ]
    assert conversation[-1]["content"] == "the summary"
    assert "at most 3 sentences" in conversation[-2]["content"]


def test_describe_passes_prompt_arguments_through(gam_model):
    client = _FakeClient("ok")

    gam_model.llm_describe(
        client=client, dataset_description="Simulated, n=200.", top=1,
    )

    prompt = client.calls[0]["messages"][1]["content"]
    assert "Simulated, n=200." in prompt
    assert prompt.count("Learner: ") == 1


def test_describe_refuses_arguments_it_controls(gam_model):
    client = _FakeClient("ok")

    with pytest.raises(TypeError, match="controls 'output'"):
        gam_model.llm_describe(client=client, output="messages")


def test_describe_without_a_key_says_so_and_sends_nothing(gam_model, monkeypatch):
    for name in ("MBOOSTRS_LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("llm_describe opened a connection without a key")

    monkeypatch.setattr("urllib.request.urlopen", refuse)

    with pytest.raises(ValueError, match="needs an API key"):
        gam_model.llm_describe()


def test_describe_posts_to_the_configured_endpoint(gam_model, monkeypatch):
    import json
    import urllib.request

    seen = {}

    class _Response:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "the reply"}}]}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def capture(request, timeout=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", capture)

    reply = gam_model.llm_describe(
        api_key="test-key", base_url="https://integrate.api.nvidia.com/v1",
        max_tokens=1234, temperature=0.2, timeout=42,
    )

    assert reply == "the reply"
    assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == DEFAULT_LLM
    assert seen["body"]["max_tokens"] == 1234
    assert seen["body"]["temperature"] == 0.2
    assert seen["timeout"] == 42


def test_describe_reports_an_http_failure_with_the_body(gam_model, monkeypatch):
    import io
    import urllib.error
    import urllib.request

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {},
            io.BytesIO(b'{"detail":"unknown model"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    with pytest.raises(RuntimeError, match="HTTP 404.*unknown model"):
        gam_model.llm_describe(api_key="test-key", llm="nope/nope")


def test_describe_reports_a_reply_that_is_all_reasoning(gam_model):
    class _Thinking(_FakeClient):
        def __init__(self):
            super().__init__()
            self.chat.completions.create = lambda **kwargs: {
                "choices": [{"message": {"content": "",
                                         "reasoning_content": "thinking..."}}]
            }

    with pytest.raises(RuntimeError, match="raise max_tokens"):
        gam_model.llm_describe(client=_Thinking())


def test_describe_restores_the_task_section_it_needs(gam_model):
    # A prompt with no closing instruction spends a request and comes back asking
    # what the reader wanted, so `llm_describe` puts the task back.
    client = _FakeClient("ok")

    gam_model.llm_describe(client=client, include=("importance", "effects"))

    prompt = client.calls[0]["messages"][1]["content"]
    assert prompt.rstrip().endswith("no code fence.")


# ---------------------------------------------------------------------------
# the shrinkage floor
# ---------------------------------------------------------------------------

def test_effect_block_says_when_df_is_at_the_shrinkage_floor():
    # A second-difference penalty leaves a 2-dimensional null space, and
    # `df(lambda)` falls to 2 and stops. At df=2 the learner is confined to
    # straight lines, so "monotone increasing" alone would read as a shape the
    # data chose.
    rng = np.random.default_rng(8)
    n = 200
    x = rng.uniform(0, 10, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    with pytest.warns(UserWarning, match="rank of the penalty's null space"):
        fit = mboost.gamboost(
            "y ~ bbs(x, df=2)", data={"y": y, "x": x},
            control=mboost.boost_control(60, 0.1),
        )

    text = fit.llm_explain_prompt(include=("effects",), n_points=5)

    assert "At the shrinkage floor: df=2" in text
    assert "can only represent straight lines" in text


def test_tensor_at_the_floor_names_the_span_it_is_confined_to():
    rng = np.random.default_rng(9)
    n = 250
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = x1 * x2 + rng.normal(scale=0.2, size=n)
    with pytest.warns(UserWarning, match="unpenalized part of the tensor penalty"):
        fit = mboost.gamboost(
            "y ~ bbs(x1, df=2) %X% bbs(x2, df=2)",
            data={"y": y, "x1": x1, "x2": x2},
            control=mboost.boost_control(60, 0.1),
        )

    text = fit.llm_explain_prompt(include=("effects",), n_points=5, cells=3)

    assert "At the shrinkage floor" in text
    assert "{1, x1, x2, x1*x2}" in text
    assert "not a freely estimated one" in text


def test_a_centered_learner_has_no_floor_to_report():
    # `center=True` projects the null space out, so every positive df is
    # attainable and there is nothing to warn about.
    rng = np.random.default_rng(10)
    n = 200
    x = rng.uniform(0, 10, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    fit = mboost.gamboost(
        "y ~ bbs(x, df=2, center=True)", data={"y": y, "x": x},
        control=mboost.boost_control(60, 0.1),
    )

    text = fit.llm_explain_prompt(include=("effects",), n_points=5)

    assert "shrinkage floor" not in text


# ---------------------------------------------------------------------------
# learner names that identify learners rather than columns
# ---------------------------------------------------------------------------

def test_base_learner_names_disambiguates_a_shared_name():
    # `bols(x) + bbs(x, df=1, center=True)` is the standard decomposition, and
    # both halves are stored under the column name "x".
    rng = np.random.default_rng(11)
    n = 150
    x = rng.uniform(0, 10, n)
    y = 0.5 * x + np.sin(x) + rng.normal(scale=0.3, size=n)
    fit = mboost.gamboost(
        "y ~ bols(x) + bbs(x, df=1, center=True)", data={"y": y, "x": x},
        control=mboost.boost_control(80, 0.1),
    )

    assert fit.base_learner_names() == ["x (bols)", "x (bbs)"]
    # The stored names are still the columns; coef() keeps both learners,
    # keyed by the disambiguated names (it used to keep one under "x").
    assert fit.base_learner_names(disambiguate=False) == ["x", "x"]
    assert set(fit.coef()) == {"x (bols)", "x (bbs)", "(offset)"}

    # A disambiguated name has to lead back to an index, or it is a dead end.
    assert fit.learner_index("x (bols)") == 0
    assert fit.learner_index("x (bbs)") == 1
    assert fit.learner_index("x", kind="bbs") == 1
    effects = fit.partial_effects(which=fit.learner_index("x (bbs)"))
    assert len(effects) == 1


def test_base_learner_names_leaves_unique_names_alone():
    rng = np.random.default_rng(12)
    n = 120
    x, z = rng.uniform(0, 10, n), rng.normal(size=n)
    y = np.sin(x) + z + rng.normal(scale=0.3, size=n)
    fit = mboost.gamboost(
        "y ~ bbs(x) + bols(z)", data={"y": y, "x": x, "z": z},
        control=mboost.boost_control(60, 0.1),
    )

    assert fit.base_learner_names() == ["x", "z"]
    assert fit.bnames() == fit.base_learner_names()
    assert fit.learner_index("x") == 0


def test_a_parenthesised_column_name_is_not_mistaken_for_a_kind():
    # `I(x^2)` reports under the expression the user wrote, parentheses and all.
    rng = np.random.default_rng(13)
    n = 120
    x = rng.uniform(1, 5, n)
    y = x**2 + rng.normal(scale=0.5, size=n)
    fit = mboost.gamboost(
        "y ~ bols(I(x^2))", data={"y": y, "x": x},
        control=mboost.boost_control(50, 0.1),
    )

    names = fit.base_learner_names()
    assert names == ["I(x^2)"]
    assert fit.learner_index("I(x^2)") == 0


# ─── per-learner contributions carry the step length (found by the agent
# ─── pilot, 2026-09-14: hand-summed branches omitted nu) ─────────────────


def test_tensor_partial_effect_carries_nu():
    """`predict(which=...)` for a %X% term was 1/nu times too large.

    `partial_effects`' Buser branch summed the unshrunken per-step
    coefficients and multiplied them into the design without the step
    length, so any decomposition over learners of a model containing a
    tensor interaction was wrong by a factor of 1/nu.
    """
    rng = np.random.default_rng(1)
    n = 400
    x1, x4 = rng.normal(size=(2, n))
    frame = {"x1": x1, "x4": x4,
             "y": 0.8 * x1 * x4 + rng.normal(scale=0.3, size=n)}
    fit = mboost.gamboost("y ~ bbs(x1, df=2.5, knots=5) %X% bbs(x4, df=2.5, knots=5)", data=frame,
                      control=mboost.boost_control(200, 0.1))
    full = np.asarray(fit.predict()).ravel()
    part = np.asarray(fit.predict(which=["__ix0"])).ravel()
    scale = np.linalg.norm(part) / np.linalg.norm(full)
    assert scale < 1.5  # was ~10.0 before the fix


def test_learner_contributions_sum_to_the_full_prediction():
    """With the offset added, the decomposition reassembles the prediction."""
    rng = np.random.default_rng(42)
    n = 600
    x1, x2, x4 = rng.normal(size=(3, n))
    frame = {"x1": x1, "x2": x2, "x4": x4,
             "y": (np.sin(2 * x1) + 0.5 * x2 + 0.8 * x1 * x4
                   + rng.normal(scale=0.4, size=n))}
    spec = ("y ~ bbs(x1) + bbs(x2) + bbs(x4)"
            " + bbs(x1, df=2.5, knots=5) %X% bbs(x4, df=2.5, knots=5)")
    fit = mboost.gamboost(spec, data=frame, control=mboost.boost_control(300, 0.1))
    full = np.asarray(fit.predict()).ravel()
    total = np.zeros(n)
    for name in sorted(set(fit.selected)):
        total += np.asarray(fit.predict(which=[name])).ravel()
    total += fit.offset_
    assert np.max(np.abs(total - full)) < 1e-8


def test_bkernel_partial_effect_carries_nu():
    """The exact-branch bkernel decomposition had the same 1/nu defect."""
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    frame = {"x": x, "y": np.sin(2 * x) + rng.normal(scale=0.3, size=n)}
    fit = mboost.gamboost("y ~ bkernel(x)", data=frame,
                      control=mboost.boost_control(100, 0.1))
    full = np.asarray(fit.predict()).ravel()
    part = np.asarray(fit.predict(which=["x"])).ravel()
    assert np.linalg.norm(part) / np.linalg.norm(full) < 1.5  # was ~10.0


def test_tensor_partial_effect_on_newdata_matches_training_rows():
    rng = np.random.default_rng(3)
    n = 300
    x1, x4 = rng.normal(size=(2, n))
    frame = {"x1": x1, "x4": x4,
             "y": 0.8 * x1 * x4 + rng.normal(scale=0.3, size=n)}
    fit = mboost.gamboost("y ~ bbs(x1, df=2.5, knots=5) %X% bbs(x4, df=2.5, knots=5)", data=frame,
                      control=mboost.boost_control(150, 0.1))
    train = np.asarray(fit.predict(which=["__ix0"])).ravel()
    replay = np.asarray(fit.predict(newdata=frame, which=["__ix0"])).ravel()
    assert train.shape == replay.shape
    assert np.max(np.abs(train - replay)) < 1e-10


def test_plain_buser_partial_effect_on_newdata_refuses():
    """A user-supplied design cannot be rebuilt silently; refuse with the fix."""
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, x ** 2])
    K = np.diag([0.0, 0.0, 1.0])
    frame = {"x": x, "y": X @ np.array([1.0, -2.0, 1.5])
             + rng.normal(scale=0.1, size=n)}
    fit = mboost.gamboost("y ~ buser(x, X=X, K=K)", data=frame, env={"X": X, "K": K},
                      control=mboost.boost_control(50, 0.1))
    with pytest.raises(ValueError, match="not rebuilt automatically"):
        fit.partial_effects(which=["x"], newdata={"x": x[:10]})


def test_lss_prompt_describes_a_one_covariate_tree():
    """A `btree` in an LSS predictor used to crash `_condense` with an
    IndexError: its covariate came back as an `(n, 1)` column, not a vector."""
    rng = np.random.default_rng(5)
    n = 300
    x, z = rng.uniform(0, 4, n), rng.uniform(0, 1, n)
    y = rng.normal(np.where(x > 2, 1.0, 0.0) + z, 0.5)
    model = mboost.gamboostLSS(
        {"mu": "y ~ btree(x) + bols(z)", "sigma": "y ~ 1"},
        data={"y": y, "x": x, "z": z}, families=mboost.GaussianLSS(),
        control=mboost.boost_control(80, 0.1),
    )

    coordinates, effect = model.partial_effects("mu", which=0)
    assert coordinates.shape == effect.shape == (n,)
    text = model.llm_explain_prompt()
    assert "Covariate: x (observed range" in text
    # The same `(n, 1)` column made plot_lss fail to cast a row to a float.
    pytest.importorskip("altair")
    from mboostrs.plot import plot_lss
    assert plot_lss(model, "mu").to_dict()
