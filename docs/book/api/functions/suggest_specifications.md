# `suggest_specifications`

Defined in `mboostrs.explore`; exported as `mboostrs.suggest_specifications`.


*Function*

```python
suggest_specifications(
    report,
    screened=None,
    drop_below=0.0,
    top=None,
    model=None,
    noise=2.0
)
```

Rank candidate changes by their expected held-out gain.

Draws on a `term_report` and, optionally, the output of
`terms.screen_interactions`.

Two warnings travel with the output. Gains are measured one change at a
time and do not add up, because two changes to the same specification
interact. And a suggestion marked `detected` rather than `measured`
reports evidence of structure, not the effect of any particular term — see
`Suggestion`. Both mean the same thing in practice: apply one change,
refit, and score it before believing it. `compare_specifications` is the
way to do that.

A term is suggested for removal when removing it does not raise the
out-of-fold error, or when what it earns is less than `noise` standard
errors above zero: on these folds such a term cannot be told from one that
earns nothing. The second rule prefers the smaller specification, and a
genuinely weak effect can be removed by it; score the removal before
accepting it. Terms suggested for removal are also offered together, as one
specification without all of them, because removing them one per round
takes as many rounds as there are terms. A group term that `term_report`
could not measure on its folds is never suggested for removal.

A shape change whose measured gain is less than `noise` standard errors
(the paired fold standard error `term_report` records) is not suggested:
on these folds it cannot be told apart from no change. Pass `noise=0` to
list every positive gain. A drop is suggested whenever removing the term
does not raise the out-of-fold error, since that is a claim about no
change.

An interaction's term follows the columns' types (see `pair_term_for`):
a tensor surface for two continuous columns, a smooth switched by a
two-valued column, a random slope for a group column. A continuous x
continuous pair yields *three* suggestions — the tensor plus both
varying-coefficient directions, `bols(a, by=b)` and `bols(b, by=a)` —
because the tensor costs 6.25 df and can miss a bilinear interaction that
a 1-df varying coefficient captures. Score them against each other
with `compare_specifications` rather than taking the first.

Source: [`python/mboostrs/explore.py` lines 2897-3066](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2897-L3066)
