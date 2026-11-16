# `llm_explain_prompt`

Defined in `mboostrs.explain`; exported as `mboostrs.llm_explain_prompt`.


*Function*

```python
llm_explain_prompt(
    model,
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
    include=('reading', 'facts', 'importance', 'effects', 'caveats', 'task'),
    expert='an expert statistician and applied data analyst',
    output='text',
    num_sentences=7
)
```

Render a fitted model as a prompt for a language model.

The prompt states how the fit was produced, what each retained base learner
contributes over the observed covariate range, how much of the risk it
removed, and which questions the fit cannot answer. Nothing is sent
anywhere: the return value is text (or chat messages) for the caller to pass
to a model of their choosing.

**Parameters**

- `model` (GLMBoostModel, GAMBoostModel, or LSSModel): A fitted model.
- `which` (str, int, or list of those, optional): Base learners to describe. Default: every selected learner.
- `top` (int, optional): Keep only this many learners, most important first. Default: all.
- `n_points` (int): Values per effect. A smooth effect in one covariate is evaluated on an
  equispaced grid of this size; anything else is condensed to at most this
  many equal-count intervals.
- `cells` (int): Grid resolution per axis for an effect over two covariates.
- `parameter` (str or list of str, optional): For an LSS fit, the distribution parameters to describe. Default: all.
- `dataset_description` (str, optional): What the data are, where they came from, what the columns mean. A model
  cannot recover this from the fit and reads it as context.
- `response_description` (str, optional): What the response measures, in the caller's own terms.
- `task` (str, optional): The closing instruction. Default: an analyst's brief in four parts —
  findings, surprising patterns, what could mislead, next check.
- `cv` (CvRiskResult, optional): A cross-validation result for this fit. Included so the stopping
  iteration can be judged; without it the prompt says so.
- `stabsel` (StabSelResult, optional): A stability-selection result for this fit. Adds a selection probability
  per learner.
- `include` (sequence of str): Sections to write, from `SECTIONS`.
- `expert` (str): Who the reader is told to be, in the system message.
- `output` ({"text", "messages", "turns"}): `"text"` returns one string. `"messages"` returns
  `[{"role": "system", ...}, {"role": "user", ...}]`, the shape the
  OpenAI-compatible and Anthropic APIs take. `"turns"` returns
  `{"system": str, "turns": [str, str, str]}`: the description, then a
  request to name what is surprising and why, then a request to compress
  the two into `num_sentences`. Send the turns in order, keeping each
  reply in the conversation; `llm_describe` does that for you.
- `num_sentences` (int): Sentence budget for the closing turn of `output="turns"`.

**Returns**

str, list of dict, or dict

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Examples**

>>> prompt = fit.llm_explain_prompt(          # doctest: +SKIP
...     dataset_description="Body fat, 71 men, DEXA reference measurement.",
...     top=5,
... )
>>> fit.llm_describe(dataset_description="...")   # doctest: +SKIP

Source: [`python/mboostrs/explain.py` lines 1060-1207](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explain.py#L1060-L1207)
