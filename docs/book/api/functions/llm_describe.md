# `llm_describe`

Defined in `mboostrs.explain`; exported as `mboostrs.llm_describe`.


*Function*

```python
llm_describe(
    model,
    llm='nvidia/nemotron-3-ultra-550b-a55b',
    base_url=None,
    api_key=None,
    client=None,
    reasoning=False,
    num_sentences=7,
    max_tokens=4000,
    temperature=0.7,
    timeout=180,
    return_conversation=False,
    **prompt_kwargs
)
```

Send the model's description to a language model and return the reply.

This is the one function in the module that makes a network request. It
builds the prompt with `llm_explain_prompt`, posts it to an
OpenAI-compatible chat-completions endpoint, and returns the text that comes
back. The default endpoint is NVIDIA's hosted inference API and the default
model is `nvidia/nemotron-3-ultra-550b-a55b`; both are arguments, and the API key comes from the
environment.

The prompt instructs the model to reply in Markdown — short headings or bold
labels, bullets, and small tables where they help. Render the returned text
with a Markdown renderer (`display(Markdown(...))` in a notebook, or a
browser) rather than printing it as plain text.

**Parameters**

- `model` (GLMBoostModel, GAMBoostModel, or LSSModel): A fitted model.
- `llm` (str): Model id the endpoint serves. Default `"nvidia/nemotron-3-ultra-550b-a55b"`.
- `base_url` (str, optional): Endpoint root, without `/chat/completions`. Default: the environment
  variable `MBOOSTRS_LLM_BASE_URL` or `NVIDIA_BASE_URL`, else
  `DEFAULT_BASE_URL`.
- `api_key` (str, optional): Bearer token. Default: the first of `MBOOSTRS_LLM_API_KEY`,
  `NVIDIA_API_KEY`, `OPENAI_API_KEY` that is set.
- `client` (object, optional): An OpenAI-compatible client — anything with
  `chat.completions.create(model=, messages=, max_tokens=, temperature=)`.
  Given one, `base_url` and `api_key` are ignored and the client's own
  configuration is used.
- `reasoning` (bool): `False` sends one request. `True` runs the three-turn sequence of
  `output="turns"`: describe the effects, name what is surprising and
  why, then compress both into `num_sentences`. Three requests, and the
  summary is the return value.
- `num_sentences` (int): Sentence budget for the closing turn when `reasoning=True`.
- `max_tokens, temperature, timeout` (int, float, float): Per-request generation and socket settings.
- `return_conversation` (bool): Return the full message list — system, user and assistant turns in order
  — instead of only the final reply.
- `**prompt_kwargs`: Passed to `llm_explain_prompt`: `dataset_description`,
  `response_description`, `top`, `n_points`, `cv`, `stabsel`,
  `task`, `expert`, and the rest.

**Returns**

str or list of dict

**Raises**

- `ValueError`: No API key was given and none of the environment variables is set.
- `RuntimeError`: The endpoint refused the request, could not be reached, or returned a
reply with no content.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`

**Examples**

```python
>>> fit.llm_describe(                                  # doctest: +SKIP
...     dataset_description="Body fat of 71 women; covariates in cm.",
...     reasoning=True,
... )
```

Source: [`python/mboostrs/explain.py` lines 1319-1453](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explain.py#L1319-L1453)
