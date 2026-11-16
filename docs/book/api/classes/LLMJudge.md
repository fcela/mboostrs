# `LLMJudge`

Defined in `mboostrs.modeler`; exported as `mboostrs.LLMJudge`.


*Class*

```python
LLMJudge(
    llm='nvidia/nemotron-3-ultra-550b-a55b',
    base_url=None,
    api_key=None,
    client=None,
    temperature=0.0,
    max_tokens=2000,
    timeout=180,
    retain_text=False,
    max_calls=None,
    max_total_tokens=None
)
```

A judge backed by any OpenAI-compatible chat endpoint.

Parameters mirror `mboostrs.explain.llm_describe`: the default
endpoint is NVIDIA's hosted inference API and the model id is
configuration, so GLM, Nemotron or any served model works; a
`client=` with `chat.completions.create` (the OpenAI SDK, LiteLLM,
a local vLLM container) replaces the endpoint entirely. Each call is
appended to `calls` with the prompt and reply SHA-256 digests,
so a run's judgments are auditable without retaining their text
(`retain_text=True` keeps the text too).

`max_calls` limits provider requests, including failed requests.
`max_total_tokens` is checked between requests using provider-reported
usage, so the last request can exceed it. Missing usage prevents another
request when a token budget is set; it is never treated as zero cost.

Source: [`python/mboostrs/modeler.py` lines 370-485](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L370-L485)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`tokens_used`](#llmjudge-tokens_used) | Property | Provider-reported tokens, or None when usage is unavailable. |

(llmjudge-tokens_used)=
## `LLMJudge.tokens_used`

*Property* of `LLMJudge`

```python
LLMJudge.tokens_used
```

Provider-reported tokens, or None when usage is unavailable.

Source: [`python/mboostrs/modeler.py` lines 404-413](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/modeler.py#L404-L413)
