# `install_skill`

Defined in `mboostrs.agent_docs`; exported as `mboostrs.install_skill`.


*Function*

```python
install_skill(
    dest: str | pathlib.Path = '.claude/skills',
    overwrite: bool = False
) -> Path
```

Copy the skill, its topics and examples to `dest/analyzing-data-with-mboostrs/`.

`dest` defaults to `.claude/skills` under the current directory, the
project skills directory that Claude Code reads; pass
`~/.claude/skills` for every project of the current user. An existing
copy is refused unless `overwrite=True`, so a hand-edited skill is not
replaced silently. Returns the directory written.

The same is available from a shell:

```python
python -m mboostrs skill install [--dest DIR] [--overwrite]
```


**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `FileExistsError`

Source: [`python/mboostrs/agent_docs/__init__.py` lines 233-263](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/agent_docs/__init__.py#L233-L263)
