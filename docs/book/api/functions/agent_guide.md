# `agent_guide`

Defined in `mboostrs.agent_docs`; exported as `mboostrs.agent_guide`.


*Function*

```python
agent_guide(topic: str | None = None, path_only: bool = False) -> str
```

Return the agent skill text, a topic section, or a file path.

`topic=None` returns the full workflow (`SKILL.md`); `"index"`
lists the topics and the worked examples; a topic name returns that
section; `"example:<name>"` returns a worked analysis. `path_only=True`
returns the file's path instead of its text (for `"index"`, the topics
directory).

Point an agent at it before its first fit:

```python
import mboostrs
print(mboostrs.agent_guide())
print(mboostrs.agent_guide("index"))
```


**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/agent_docs/__init__.py` lines 196-230](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/agent_docs/__init__.py#L196-L230)
